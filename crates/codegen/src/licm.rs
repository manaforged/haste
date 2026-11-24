use std::sync::Arc;

use crate::cfg::{self, Cfg, Loop, Loops};
use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, Op, Type, Value};
use crate::lsr::is_in_loop;
use crate::mem::{LoopMem, MemRoot, Provenance, in_range};

mod mem_promote;
mod reassoc;
mod unswitch;

fn insert_rpo(rpo: &mut Vec<Block>, anchor: Block, block: Block, after: bool) {
    if rpo.contains(&block) {
        return;
    }
    if let Some(pos) = rpo.iter().position(|&b| b == anchor) {
        rpo.insert(pos + usize::from(after), block);
    } else {
        rpo.push(block);
    }
}

fn can_hoist_global_load(
    func: &Function,
    mem: &LoopMem,
    prov: &Provenance,
    head_prefix: bool,
    inst: Inst,
) -> bool {
    let data = &func.insts[inst.index()];
    let Some(res) = data.result else { return false };
    let Op::Load(addr, offset) = data.op else {
        return false;
    };
    if mem.calls || mem.atomics || mem.unknown_mem_writes {
        return false;
    }
    let width = func.ty(res).bytes();
    let deref_ok = in_range(
        i64::from(offset),
        width,
        prov.ro_bytes.get(addr.index()).copied().unwrap_or(0),
    );
    if !head_prefix && !deref_ok {
        return false;
    }
    matches!(
        prov.loc(addr, i64::from(offset), width).root,
        MemRoot::Global(_)
    )
}

const MAX_HOISTED_SYMBOLS: usize = 4;
const MAX_LOOP_GP_LIVE: usize = 18;
const MAX_LOOP_VEC_LIVE: usize = 11;

struct Invariants<'a> {
    prov: &'a Provenance,
    is_const: &'a mut Vec<bool>,
    zeros: &'a [bool],
}

pub(crate) fn always_executes_in_loop(cfg: &Cfg, each: &Loop, blk: Block) -> bool {
    each.latches
        .iter()
        .all(|&latch| cfg::dominates(&cfg.idom, blk, latch))
}

pub(crate) fn cheap_call_addr(op: &Op, is_const: impl Fn(Value) -> bool) -> bool {
    match *op {
        Op::StackAddr(..) => true,
        Op::Binary(Binary::Iadd | Binary::Isub, a, b) => is_const(a) || is_const(b),
        _ => false,
    }
}

fn dominates_all_exits(func: &Function, cfg: &Cfg, each: &Loop, blk: Block) -> bool {
    (0..each.body.len()).all(|bi| {
        !each.body[bi]
            || !func
                .successors(Block(bi as u32))
                .any(|s| !is_in_loop(each, s))
            || cfg::dominates(&cfg.idom, blk, Block(bi as u32))
    })
}

fn loop_live_counts(
    func: &Function,
    each: &Loop,
    is_const: &[bool],
    home: &[Option<Block>],
) -> (usize, usize, Vec<u32>) {
    let mut seen = vec![false; func.types.len()];
    let mut uses = vec![0_u32; func.types.len()];
    let mut gp_live = 0_usize;
    let mut vec_live = 0_usize;
    let mut count = |v: Value| {
        if !seen[v.index()] {
            seen[v.index()] = true;
            if func.ty(v).in_float_register() {
                vec_live += 1;
            } else {
                gp_live += 1;
            }
        }
    };
    for &p in &func.blocks[each.head.index()].params {
        count(p);
    }
    for b in (0..each.body.len()).filter(|&b| each.body[b]) {
        for &inst in &func.blocks[b].insts {
            func.insts[inst.index()].op.for_operands(|v| {
                let vi = v.index();
                if is_const.get(vi).copied().unwrap_or(false) {
                    return;
                }
                uses[vi] += 1;
                if home
                    .get(vi)
                    .copied()
                    .flatten()
                    .is_none_or(|at| !is_in_loop(each, at))
                {
                    count(v);
                }
            });
        }
    }
    (gp_live, vec_live, uses)
}

struct HoistCtx<'a> {
    each: &'a Loop,
    prov: &'a Provenance,
    is_const: &'a [bool],
    zeros: &'a [bool],
    mem: &'a LoopMem,
}

#[derive(Clone, Copy)]
struct BlockHoistInfo {
    always_exec: bool,
    has_reg_call: bool,
    cold_call_blk: bool,
    head_prefix: bool,
}

impl BlockHoistInfo {
    fn new(func: &Function, cfg: &Cfg, ctx: &HoistCtx<'_>, blk: Block) -> Self {
        let always_exec = always_executes_in_loop(cfg, ctx.each, blk);
        let has_reg_call = crate::live::has_regular_call(func, blk);
        let cold_call_blk = is_cold_call_block(func, cfg, ctx, blk, always_exec, has_reg_call);
        Self {
            always_exec,
            has_reg_call,
            cold_call_blk,
            head_prefix: blk == ctx.each.head,
        }
    }

    fn update_head_prefix(&mut self, op: &Op, tc_res: Option<Value>) {
        if tc_res.is_none()
            && matches!(
                op,
                Op::Call { .. } | Op::TryCall { .. } | Op::Trap | Op::Trapz(_) | Op::Brif(..)
            )
        {
            self.head_prefix = false;
        }
    }
}

fn is_cold_call_block(
    func: &Function,
    cfg: &Cfg,
    ctx: &HoistCtx<'_>,
    blk: Block,
    always_exec: bool,
    has_reg_call: bool,
) -> bool {
    if always_exec && dominates_all_exits(func, cfg, ctx.each, blk) {
        return false;
    }
    has_reg_call
        || func.blocks[blk.index()].insts.iter().any(|&i| {
            let op = &func.insts[i.index()].op;
            match op {
                Op::Call { .. } => func
                    .pure_thread_cell_result(op, |v| ctx.zeros.get(v.index()) == Some(&true))
                    .is_none(),
                Op::TryCall { .. } | Op::Trap => true,
                _ => false,
            }
        })
}

fn thread_cell_res(func: &Function, zeros: &[bool], op: &Op, always_exec: bool) -> Option<Value> {
    (!zeros.is_empty() && always_exec)
        .then(|| func.pure_thread_cell_result(op, |v| zeros.get(v.index()) == Some(&true)))
        .flatten()
}

fn is_vec_mat_op(op: &Op, res_ty: Option<Type>) -> bool {
    res_ty.is_some_and(Type::in_float_register)
        && matches!(
            op,
            Op::Insert(..)
                | Op::Shuffle(..)
                | Op::Const(_)
                | Op::Load(..)
                | Op::StackLoad(..)
                | Op::Convert(Convert::Bitcast, _)
        )
}

fn is_cheap_cold_op(op: &Op, ctx: &HoistCtx<'_>, info: BlockHoistInfo) -> bool {
    let check_const = |v: Value| ctx.is_const.get(v.index()) == Some(&true);
    if info.has_reg_call && cheap_call_addr(op, check_const) {
        return true;
    }
    ctx.mem.calls
        && !info.always_exec
        && (cheap_call_addr(op, check_const)
            || matches!(
                op,
                Op::Const(_)
                    | Op::Convert(Convert::Ireduce | Convert::Uextend | Convert::Sextend, _)
            ))
}

fn operand_hoist_info(
    func: &Function,
    each: &Loop,
    home: &[Option<Block>],
    loop_uses: &[u32],
    op: &Op,
) -> (bool, bool) {
    let mut invariant = true;
    let mut frees_gp = false;
    op.for_operands(|v| {
        invariant &= home
            .get(v.index())
            .copied()
            .flatten()
            .is_none_or(|at| !is_in_loop(each, at));
        frees_gp |= loop_uses[v.index()] == 1 && !func.ty(v).in_float_register();
    });
    (invariant, frees_gp)
}

struct HoistBudget {
    gp_live: usize,
    vec_live: usize,
    loop_uses: Vec<u32>,
    uses: Option<Vec<u32>>,
    hoisted_vec: Vec<Value>,
    hoisted_symbols: usize,
    hoisted_gp: usize,
    hoisted_vecs: usize,
}

impl HoistBudget {
    fn new(func: &Function, ctx: &HoistCtx<'_>, home: &[Option<Block>]) -> Self {
        let (gp_live, vec_live, loop_uses) = loop_live_counts(func, ctx.each, ctx.is_const, home);
        let has_insert = (0..ctx.each.body.len())
            .filter(|&b| ctx.each.body[b])
            .flat_map(|b| &func.blocks[b].insts)
            .any(|inst| matches!(func.insts[inst.index()].op, Op::Insert(..)));
        Self {
            gp_live,
            vec_live,
            loop_uses,
            uses: has_insert.then(|| func.use_counts()),
            hoisted_vec: Vec::new(),
            hoisted_symbols: 0,
            hoisted_gp: 0,
            hoisted_vecs: 0,
        }
    }

    fn is_movable(
        &self,
        func: &Function,
        ctx: &HoistCtx<'_>,
        info: BlockHoistInfo,
        inst: Inst,
        has_tc: bool,
    ) -> bool {
        if has_tc {
            return true;
        }
        let op = &func.insts[inst.index()].op;
        let is_symbol = matches!(op, Op::Symbol(_));
        let static_loc =
            matches!(op, Op::StackAddr(..)) || (matches!(op, Op::SlotAddr(_)) && !info.always_exec);
        let sym_ok =
            !ctx.mem.calls || (info.always_exec && self.hoisted_symbols < MAX_HOISTED_SYMBOLS);
        let speculable_ok = op.is_speculable()
            && !info.cold_call_blk
            && !is_cheap_cold_op(op, ctx, info)
            && !(ctx.mem.calls && static_loc)
            && (!is_symbol || sym_ok);
        speculable_ok
            || ctx
                .mem
                .can_hoist_load(func, ctx.prov, info.head_prefix, inst)
            || can_hoist_global_load(func, ctx.mem, ctx.prov, info.head_prefix, inst)
    }

    fn gp_ok(
        &self,
        func: &Function,
        ctx: &HoistCtx<'_>,
        head_prefix: bool,
        inst: Inst,
        adds_gp: bool,
        has_tc: bool,
    ) -> bool {
        if !adds_gp {
            return true;
        }
        let op = &func.insts[inst.index()].op;
        let gp_total = self.gp_live + self.hoisted_gp;
        if (op.is_speculable() || has_tc) && gp_total >= MAX_LOOP_GP_LIVE {
            return false;
        }
        !(ctx.mem.unique_heap_writes
            && matches!(op, Op::Load(..))
            && gp_total >= 12
            && !ctx
                .mem
                .hoists_past_heap_writes(func, ctx.prov, head_prefix, inst))
    }

    fn try_hoist(
        &mut self,
        func: &Function,
        ctx: &HoistCtx<'_>,
        info: BlockHoistInfo,
        home: &[Option<Block>],
        inst: Inst,
        tc_res: Option<Value>,
    ) -> Option<Value> {
        let op = &func.insts[inst.index()].op;
        let res = func.insts[inst.index()].result.or(tc_res)?;
        let res_ty = Some(func.ty(res));
        let is_vec_mat = is_vec_mat_op(op, res_ty);
        let replaces_single_use = matches!(
            op,
            Op::Insert(prev, ..)
                if self.hoisted_vec.contains(prev)
                    && self.uses.as_ref().is_some_and(|u| u[prev.index()] == 1)
        );
        let vec_call_block = ctx.mem.calls && res_ty.is_some_and(Type::is_vector);
        let vec_ok = !is_vec_mat
            || (!vec_call_block
                && (replaces_single_use || self.vec_live + self.hoisted_vecs < MAX_LOOP_VEC_LIVE));
        let (operands_invariant, frees_gp) =
            operand_hoist_info(func, ctx.each, home, &self.loop_uses, op);
        let adds_gp = !frees_gp
            && !matches!(op, Op::Const(_))
            && res_ty.is_some_and(|ty| !ty.in_float_register());
        let movable = self.is_movable(func, ctx, info, inst, tc_res.is_some());
        let gp_ok = self.gp_ok(func, ctx, info.head_prefix, inst, adds_gp, tc_res.is_some());
        if !(movable && vec_ok && gp_ok && operands_invariant) {
            return None;
        }
        self.hoisted_gp += usize::from(adds_gp);
        if matches!(op, Op::Symbol(_)) {
            self.hoisted_symbols += 1;
        }
        if is_vec_mat {
            if !replaces_single_use {
                self.hoisted_vecs += 1;
            }
            self.hoisted_vec.push(res);
        }
        Some(res)
    }
}

fn hoist(
    func: &mut Function,
    cfg: &Cfg,
    each: &Loop,
    entry: Block,
    inv: &mut Invariants<'_>,
    mem: &LoopMem,
    home: &mut Vec<Option<Block>>,
) -> bool {
    let reassociated = reassoc::reassociate_loop_adds(func, cfg, each, home, inv.is_const);
    let ctx = HoistCtx {
        each,
        prov: inv.prov,
        is_const: inv.is_const,
        zeros: inv.zeros,
        mem,
    };
    let mut budget = HoistBudget::new(func, &ctx, home);
    let mut moved: Vec<Inst> = Vec::new();
    for &blk in &cfg.rpo {
        if !is_in_loop(each, blk) {
            continue;
        }
        let mut info = BlockHoistInfo::new(func, cfg, &ctx, blk);
        let insts = std::mem::take(&mut func.blocks[blk.index()].insts);
        let mut kept = Vec::with_capacity(insts.len());
        for inst in insts {
            let op = &func.insts[inst.index()].op;
            let tc_res = thread_cell_res(func, ctx.zeros, op, info.always_exec);
            info.update_head_prefix(op, tc_res);
            if let Some(res) = budget.try_hoist(func, &ctx, info, home, inst, tc_res) {
                home[res.index()] = Some(entry);
                moved.push(inst);
            } else {
                kept.push(inst);
            }
        }
        func.blocks[blk.index()].insts = kept;
    }
    if moved.is_empty() {
        return reassociated;
    }
    let insts = &mut func.blocks[entry.index()].insts;
    let at = insts.len().saturating_sub(1);
    insts.splice(at..at, moved);
    true
}

fn loops_touch_memory(func: &Function, loops: &Loops) -> bool {
    func.blocks
        .iter()
        .zip(&loops.depths)
        .filter(|(_, depth)| **depth > 0)
        .flat_map(|(block, _)| &block.insts)
        .any(|inst| {
            matches!(
                func.insts[inst.index()].op,
                Op::Load(..)
                    | Op::StackLoad(..)
                    | Op::Store(..)
                    | Op::StackStore(..)
                    | Op::Move(..)
                    | Op::Copy(..)
            )
        })
}

/// Moves loop-invariant work of every natural loop that one jump enters out to the block that enters it, then tests each invariant exit there once.
pub(crate) fn run(func: &mut Function) -> Domain {
    if func.entry.is_none() || func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut cfg = func.cfg();
    if cfg.loops().list.is_empty() {
        return Domain::NONE;
    }
    let mut found = cfg.loops().clone();
    let prov = if loops_touch_memory(func, &found) {
        Provenance::new(func, &cfg)
    } else {
        Provenance::default()
    };
    let mut home = func.def_blocks();
    let mut is_const = vec![false; func.types.len()];
    for data in &func.insts {
        if let (&Op::Const(_), Some(res)) = (&data.op, data.result) {
            is_const[res.index()] = true;
        }
    }
    let zeros = func.thread_cell_zeros();
    let mut inv = Invariants {
        prov: &prov,
        is_const: &mut is_const,
        zeros: &zeros,
    };
    let mut dirty = Domain::NONE;
    let mut made = Vec::new();
    for idx in 0..found.list.len() {
        let before = func.blocks.len();
        let Some(pre) = cfg::preheader(func, &cfg, &found.list[idx]) else {
            continue;
        };
        if func.blocks.len() > before {
            made.push((pre, found.list[idx].head));
            home.resize(func.types.len(), Some(pre));
            let head_idx = found.list[idx].head.index();
            let cfg = Arc::make_mut(&mut cfg);
            cfg.split_preds_for_preheader(found.list[idx].head, pre, |p| {
                is_in_loop(&found.list[idx], p)
            });
            cfg.reachable.push(true);
            cfg.idom.push(cfg.idom[head_idx]);
            cfg.idom[head_idx] = Some(pre);
            insert_rpo(&mut cfg.rpo, found.list[idx].head, pre, false);
            found.list[idx].body.resize(pre.index() + 1, false);
            for later in &mut found.list[idx + 1..] {
                let inside = later.body.get(head_idx).copied().unwrap_or(false);
                later.body.resize(pre.index(), false);
                later.body.push(inside);
            }
        }
        let (left, later) = found.list.split_at_mut(idx + 1);
        let each = &left[idx];
        let mut mem = LoopMem::analyze(func, each, &prov);
        let mut state = mem_promote::PromoteState {
            func,
            cfg: &mut cfg,
            each,
            later,
            home: &mut home,
        };
        let blocks_before = state.func.blocks.len();
        if mem_promote::promote_mem(&mut state, &mem, &prov, pre) {
            dirty |= Domain::VALUES | Domain::MEM;
            if state.func.blocks.len() > blocks_before {
                dirty |= Domain::CFG;
            }
            mem = LoopMem::analyze(func, &found.list[idx], &prov);
        }
        if hoist(func, &cfg, &found.list[idx], pre, &mut inv, &mem, &mut home) {
            dirty |= Domain::VALUES;
        }
    }
    if !made.is_empty() || dirty.contains(Domain::CFG) {
        func.bump_cfg();
    }
    if unswitch::unswitch(func) {
        dirty |= Domain::CFG | Domain::VALUES;
    }
    if cfg::drop_bare_preheaders(func, &made) {
        dirty |= Domain::CFG;
    }
    dirty
}
