mod cmp;
mod leaf;

use cmp::build_cmp_pack;
use leaf::{build_pack, build_vec_pack};

use crate::cfg::Cfg;
use crate::ir::{
    Binary, Block, Convert, Function, Inst, IntCC, Op, StackSlot, TargetOpId, Type, Unary, Value,
};
use crate::mem::{MemLoc, Provenance};

struct PackCtx<'a> {
    func: &'a Function,
    cfg: &'a Cfg,
    prov: &'a Provenance,
    consts: &'a [Option<i64>],
    def_info: &'a [DefInfo],
}

const MAX_PRED_CHAIN: usize = 4;

#[derive(Clone, Copy)]
enum RawLoad {
    Stack,
    Addr(Value, i32),
}

#[derive(Clone, PartialEq, Eq)]
enum AddrKey {
    Slot(StackSlot),
    Global(u32),
    Expr(Vec<(u32, i64)>),
}

#[derive(Clone, Copy)]
struct Leaf {
    blk_idx: usize,
    pos: usize,
    loc: MemLoc,
    canon_off: i64,
    raw: RawLoad,
    shift_byte: i64,
    bytes: i64,
    bswapped: bool,
}

enum PackOut {
    Load {
        big_endian: bool,
    },
    Cmp {
        cc: IntCC,
        mask: Option<i64>,
        rhs_const: Option<i64>,
        rhs_op: Option<Op>,
    },
}

struct PackPlan {
    wide_op: Op,
    wide_ty: Type,
    out: PackOut,
    touched: Vec<Value>,
}

type DefInfo = Option<(Block, usize, Inst)>;

pub(super) fn pred_chain(cfg: &Cfg, blk: Block) -> Vec<Block> {
    let mut chain = vec![blk];
    let mut cur = blk;
    while chain.len() < MAX_PRED_CHAIN
        && let Some(pred) = cfg.single_pred(cur)
        && cfg.reachable.get(pred.index()).copied().unwrap_or(false)
        && !chain.contains(&pred)
    {
        chain.push(pred);
        cur = pred;
    }
    chain.reverse();
    chain
}

fn sweep_dead(
    func: &mut Function,
    blk_idx: usize,
    dead_pool: &[bool],
    uses: &mut [u32],
    def_info: &mut [DefInfo],
) {
    let insts = std::mem::take(&mut func.blocks[blk_idx].insts);
    let mut live = Vec::with_capacity(insts.len());
    for &inst in insts.iter().rev() {
        let data = &func.insts[inst.index()];
        if let Some(res) = data.result
            && dead_pool.get(res.index()).copied().unwrap_or(false)
            && uses[res.index()] == 0
            && data.op.is_removable()
        {
            data.op
                .for_operands(|v| uses[v.index()] = uses[v.index()].saturating_sub(1));
            def_info[res.index()] = None;
            continue;
        }
        live.push(inst);
    }
    live.reverse();
    let blk = Block(blk_idx as u32);
    for (pos, &inst) in live.iter().enumerate() {
        if let Some(res) = func.insts[inst.index()].result {
            def_info[res.index()] = Some((blk, pos, inst));
        }
    }
    func.blocks[blk_idx].insts = live;
}

pub(super) fn has_vec_pack_cand(func: &Function) -> bool {
    pack_cands(func).1
}

fn pack_cands(func: &Function) -> (bool, bool) {
    let (mut narrow_loads, mut vec_cand) = (0_u32, false);
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let ty = func.ty(res);
            match data.op {
                Op::Load(..) | Op::StackLoad(..) if ty.is_small_int() && ty.bytes() <= 4 => {
                    narrow_loads += 1
                }
                Op::Insert(..) | Op::TargetOp(TargetOpId::X64Concat128, _) if ty.is_vector() => {
                    vec_cand = true
                }
                _ => {}
            }
            if narrow_loads >= 2 && vec_cand {
                return (true, true);
            }
        }
    }
    (narrow_loads >= 2, vec_cand)
}

struct PackTables {
    consts: Vec<Option<i64>>,
    def_info: Vec<DefInfo>,
    uses: Vec<u32>,
    dead_pool: Vec<bool>,
    touched_blk: Vec<bool>,
    consumed: Vec<bool>,
    consumed_marks: Vec<usize>,
}

fn build_pack_tables(func: &Function, cfg: &Cfg) -> PackTables {
    let n = func.types.len();
    let (mut consts, mut def_info) = (vec![None; n], vec![None; n]);
    for &blk in &cfg.rpo {
        for (pos, &inst) in func.blocks[blk.index()].insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            def_info[res.index()] = Some((blk, pos, inst));
            if let Op::Const(imm) = data.op {
                consts[res.index()] = Some(imm);
            }
        }
    }
    PackTables {
        consts,
        def_info,
        uses: func.use_counts(),
        dead_pool: vec![false; n],
        touched_blk: vec![false; func.blocks.len()],
        consumed: vec![false; n],
        consumed_marks: Vec::new(),
    }
}

fn build_scalar_pack(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (pos, inst, res, res_ty): (usize, Inst, Value, Type),
) -> Option<PackPlan> {
    if matches!(
        ctx.func.insts[inst.index()].op,
        Op::Binary(Binary::Bor | Binary::Iadd, ..)
    ) && matches!(res_ty, Type::I16 | Type::I32 | Type::I64)
        && let Some(plan) = build_pack(ctx, chain, (pos, res, res_ty))
    {
        return Some(plan);
    }
    build_cmp_pack(ctx, chain, (pos, inst, res_ty))
}

fn plan_one_pack(
    ctx: &PackCtx<'_>,
    target: Option<&crate::Target>,
    chain: &[Block],
    uses: &[u32],
    (pos, inst): (usize, Inst),
) -> Option<(Value, Type, PackPlan)> {
    let data = &ctx.func.insts[inst.index()];
    let res = data.result?;
    let res_ty = ctx.func.ty(res);
    let plan = match target {
        None if matches!(
            data.op,
            Op::Binary(Binary::Bor | Binary::Iadd | Binary::Band, ..) | Op::Icmp(..)
        ) =>
        {
            build_scalar_pack(ctx, chain, (pos, inst, res, res_ty))
        }
        Some(tgt)
            if matches!(
                data.op,
                Op::Insert(..) | Op::TargetOp(TargetOpId::X64Concat128, _)
            ) || matches!(data.op, Op::Convert(Convert::Bitcast, x) if uses[x.index()] == 1) =>
        {
            build_vec_pack(ctx, tgt, chain, (pos, res, res_ty))
        }
        _ => None,
    }?;
    Some((res, res_ty, plan))
}

fn has_block_pack_cand(func: &Function, target: Option<&crate::Target>, insts: &[Inst]) -> bool {
    insts.iter().any(|&inst| {
        let data = &func.insts[inst.index()];
        data.result.is_some()
            && match target {
                None => matches!(
                    data.op,
                    Op::Binary(Binary::Bor | Binary::Iadd | Binary::Band, ..) | Op::Icmp(..)
                ),
                Some(_) => {
                    matches!(
                        data.op,
                        Op::Insert(..)
                            | Op::TargetOp(TargetOpId::X64Concat128, _)
                            | Op::Convert(Convert::Bitcast, _)
                    )
                }
            }
    })
}

fn plan_block_packs(
    ctx: &PackCtx<'_>,
    target: Option<&crate::Target>,
    blk: Block,
    uses: &[u32],
    (consumed, marks): (&mut [bool], &mut Vec<usize>),
) -> Vec<Option<(Value, Type, PackPlan)>> {
    let insts = &ctx.func.blocks[blk.index()].insts;
    if !has_block_pack_cand(ctx.func, target, insts) {
        return Vec::new();
    }
    let chain = pred_chain(ctx.cfg, blk);
    let mut plans: Vec<Option<(Value, Type, PackPlan)>> = Vec::new();
    for (pos, &inst) in insts.iter().enumerate().rev() {
        let Some(res) = ctx.func.insts[inst.index()]
            .result
            .filter(|r| !consumed[r.index()])
        else {
            continue;
        };
        let Some((_, res_ty, p)) = plan_one_pack(ctx, target, &chain, uses, (pos, inst)) else {
            continue;
        };
        if plans.is_empty() {
            plans.resize_with(insts.len(), || None);
        }
        for &v in &p.touched {
            if v != res && uses.get(v.index()).copied().unwrap_or(0) <= 1 && !consumed[v.index()] {
                consumed[v.index()] = true;
                marks.push(v.index());
            }
        }
        plans[pos] = Some((res, res_ty, p));
    }
    for idx in marks.drain(..) {
        consumed[idx] = false;
    }
    plans
}

fn push_fresh_val(
    func: &mut Function,
    op: Op,
    ty: Type,
    out: &mut Vec<Inst>,
    t: &mut PackTables,
) -> Value {
    let val = func.value(ty);
    t.uses.push(1);
    t.def_info.push(None);
    t.dead_pool.push(false);
    t.consts.push(None);
    t.consumed.push(false);
    out.push(func.add(op, Some(val)));
    val
}

fn emit_cmp_inst(
    func: &mut Function,
    inst: Inst,
    res_ty: Type,
    plan: PackPlan,
    (out, t): (&mut Vec<Inst>, &mut PackTables),
) {
    let PackOut::Cmp {
        cc,
        mask,
        rhs_const,
        rhs_op,
    } = plan.out
    else {
        return;
    };
    let mut lhs = push_fresh_val(func, plan.wide_op, plan.wide_ty, out, t);
    if let Some(m) = mask {
        let mv = push_fresh_val(func, Op::Const(m), plan.wide_ty, out, t);
        lhs = push_fresh_val(
            func,
            Op::Binary(Binary::Band, lhs, mv),
            plan.wide_ty,
            out,
            t,
        );
    }
    let rhs = push_fresh_val(
        func,
        rhs_const.map(Op::Const).or(rhs_op).unwrap_or(Op::Const(0)),
        plan.wide_ty,
        out,
        t,
    );
    let cmp = Op::Icmp(cc, lhs, rhs);
    func.insts[inst.index()].op = if res_ty == Type::I8 {
        cmp
    } else {
        let cv = push_fresh_val(func, cmp, Type::I8, out, t);
        Op::Convert(Convert::Uextend, cv)
    };
}

fn emit_packed_inst(
    func: &mut Function,
    inst: Inst,
    res_ty: Type,
    plan: PackPlan,
    (out, t): (&mut Vec<Inst>, &mut PackTables),
) {
    let PackOut::Load { big_endian } = plan.out else {
        emit_cmp_inst(func, inst, res_ty, plan, (out, t));
        return;
    };
    if !big_endian && plan.wide_ty == res_ty {
        func.insts[inst.index()].op = plan.wide_op;
        return;
    }
    let wide_val = push_fresh_val(func, plan.wide_op, plan.wide_ty, out, t);
    func.insts[inst.index()].op = if big_endian && plan.wide_ty == res_ty {
        Op::Unary(Unary::Bswap, wide_val)
    } else if big_endian {
        let sw = push_fresh_val(
            func,
            Op::Unary(Unary::Bswap, wide_val),
            plan.wide_ty,
            out,
            t,
        );
        Op::Convert(Convert::Uextend, sw)
    } else {
        Op::Convert(Convert::Uextend, wide_val)
    };
}

fn mark_dead_touched(touched: &[Value], res: Value, t: &mut PackTables) {
    for &v in touched {
        if v == res {
            continue;
        }
        t.dead_pool[v.index()] = true;
        if let Some((db, _, _)) = t.def_info[v.index()] {
            t.touched_blk[db.index()] = true;
        }
    }
}

fn apply_block_packs(
    func: &mut Function,
    b_idx: usize,
    mut plans: Vec<Option<(Value, Type, PackPlan)>>,
    t: &mut PackTables,
) -> bool {
    let insts = std::mem::take(&mut func.blocks[b_idx].insts);
    let mut out = Vec::with_capacity(insts.len());
    let mut changed = false;
    for (pos, &inst) in insts.iter().enumerate() {
        if let Some((res, res_ty, plan)) = plans[pos].take() {
            func.insts[inst.index()]
                .op
                .for_operands(|v| t.uses[v.index()] = t.uses[v.index()].saturating_sub(1));
            mark_dead_touched(&plan.touched, res, t);
            emit_packed_inst(func, inst, res_ty, plan, (&mut out, t));
            changed = true;
        }
        out.push(inst);
    }
    func.blocks[b_idx].insts = out;
    t.touched_blk[b_idx] = true;
    sweep_dead(func, b_idx, &t.dead_pool, &mut t.uses, &mut t.def_info);
    changed
}

pub(super) fn coalesce_vec_rescan(func: &mut Function, cfg: &Cfg, target: &crate::Target) -> bool {
    if !has_vec_pack_cand(func) {
        return false;
    }
    let prov = Provenance::new(func, cfg);
    coalesce_loads(func, cfg, &prov, target)
}

fn run_pack_round(
    func: &mut Function,
    cfg: &Cfg,
    prov: &Provenance,
    target: Option<&crate::Target>,
    t: &mut PackTables,
) -> bool {
    let mut changed = false;
    for &blk in &cfg.rpo {
        let plans = {
            let ctx = PackCtx {
                func,
                cfg,
                prov,
                consts: &t.consts,
                def_info: &t.def_info,
            };
            plan_block_packs(
                &ctx,
                target,
                blk,
                &t.uses,
                (&mut t.consumed, &mut t.consumed_marks),
            )
        };
        if !plans.is_empty() {
            changed |= apply_block_packs(func, blk.index(), plans, t);
        }
    }
    if changed {
        for &blk in cfg.rpo.iter().rev() {
            if t.touched_blk[blk.index()] {
                sweep_dead(
                    func,
                    blk.index(),
                    &t.dead_pool,
                    &mut t.uses,
                    &mut t.def_info,
                );
            }
        }
    }
    changed
}

pub(super) fn coalesce_loads(
    func: &mut Function,
    cfg: &Cfg,
    prov: &Provenance,
    target: &crate::Target,
) -> bool {
    let (scalar_cand, vec_cand) = pack_cands(func);
    if !scalar_cand && !vec_cand {
        return false;
    }
    let mut t = build_pack_tables(func, cfg);
    let mut changed = scalar_cand && run_pack_round(func, cfg, prov, None, &mut t);
    if vec_cand || (changed && has_vec_pack_cand(func)) {
        t.touched_blk.fill(false);
        changed |= run_pack_round(func, cfg, prov, Some(target), &mut t);
    }
    changed
}
