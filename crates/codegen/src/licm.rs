use std::sync::Arc;

use crate::cfg::{self, Cfg, Loop, Loops};
use crate::ir::{Block, BlockData, Domain, Function, Inst, Op, Type, Value};
use crate::lsr::{EdgeArm, is_in_loop};
use crate::mem::{LoopMem, MemLoc, MemRoot, PromoteBase, PromoteCand, Provenance, in_range};
use crate::merge::{find, identity, rename};

mod reassoc;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SymVal {
    Phi,
    Merge(Block),
    Inst(Value),
}

struct ExitEdge {
    src: Block,
    branch: EdgeArm,
    target: Block,
    val: Value,
}

struct LoopShape {
    rpo: Vec<Block>,
    in_preds: Vec<Vec<Block>>,
}

struct PromoteState<'a> {
    func: &'a mut Function,
    cfg: &'a mut Arc<Cfg>,
    each: &'a Loop,
    later: &'a mut [Loop],
    home: &'a mut Vec<Option<Block>>,
}

fn loop_rpo_preds(func: &Function, cfg: &Cfg, each: &Loop) -> Option<LoopShape> {
    let n = func.blocks.len();
    for b in (0..each.body.len()).filter(|b| each.body[*b]) {
        let block = Block(b as u32);
        let needs_args = func.successors(block).any(|t| t == each.head || !is_in_loop(each, t));
        if needs_args && !matches!(func.last(block), Some(Op::Jump(..) | Op::Brif(..))) {
            return None;
        }
    }
    let mut seen = vec![false; n];
    let mut post = Vec::new();
    let mut stack = vec![(each.head, 0_usize)];
    seen[each.head.index()] = true;
    while let Some(top) = stack.last_mut() {
        let (b, next) = (top.0, top.1);
        top.1 += 1;
        match func.successors(b).nth(next) {
            Some(t) if is_in_loop(each, t) && !seen[t.index()] => {
                seen[t.index()] = true;
                stack.push((t, 0));
            }
            Some(_) => {}
            None => {
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    let mut in_preds = vec![Vec::new(); n];
    for &b in &post {
        if b == each.head {
            continue;
        }
        for &p in cfg.preds_of(b) {
            if is_in_loop(each, p) && !in_preds[b.index()].contains(&p) {
                in_preds[b.index()].push(p);
            }
        }
    }
    Some(LoopShape { rpo: post, in_preds })
}

fn inst_matches_cand(func: &Function, prov: &Provenance, op: &Op, res: Option<Value>, cand: PromoteCand) -> Option<(bool, Option<Value>)> {
    let (loc, ty, is_write, stored) = match (op, res) {
        (&Op::StackLoad(s, off), Some(r)) => (MemLoc::slot(s, i64::from(off), func.ty(r).bytes()), func.ty(r), false, None),
        (&Op::Load(a, off), Some(r)) => (prov.loc(a, i64::from(off), func.ty(r).bytes()), func.ty(r), false, None),
        (&Op::StackStore(v, s, off), _) => (MemLoc::slot(s, i64::from(off), func.ty(v).bytes()), func.ty(v), true, Some(v)),
        (&Op::Store(v, a, off), _) => (prov.loc(a, i64::from(off), func.ty(v).bytes()), func.ty(v), true, Some(v)),
        _ => return None,
    };
    ((loc.base, loc.offset, ty) == (cand.loc.base, cand.loc.offset, cand.ty)).then_some((is_write, stored))
}

fn can_zero_trip_store(prov: &Provenance, cand: PromoteCand) -> bool {
    match cand.base {
        PromoteBase::Slot(_) => true,
        PromoteBase::Addr(addr) => {
            let deref = prov.ro_bytes.get(addr.index()).copied().unwrap_or(0);
            in_range(i64::from(cand.offset), cand.ty.bytes(), deref)
        }
    }
}

fn can_merge_block(func: &Function, cfg: &Cfg, each: &Loop, b: Block) -> bool {
    let preds = cfg.preds_of(b);
    !preds.is_empty()
        && preds.iter().all(|p| {
            is_in_loop(each, *p)
                && matches!(func.last(*p), Some(Op::Jump(..) | Op::Brif(..)))
        })
}

fn join_block_preds(
    out_state: &[Option<(SymVal, bool)>],
    preds: &[Block],
    block: Block,
    mergeable: bool,
) -> Result<Option<(SymVal, bool)>, ()> {
    let mut joined: Option<(SymVal, bool)> = None;
    let mut all_ready = !preds.is_empty();
    let mut any_dirty = false;
    let mut same = true;
    for &p in preds {
        let Some(st) = out_state[p.index()] else {
            all_ready = false;
            continue;
        };
        any_dirty |= st.1;
        match joined {
            None => joined = Some(st),
            Some(prev) if prev != st => same = false,
            Some(_) => {}
        }
    }
    let Some(first) = joined else {
        return Ok(None);
    };
    if all_ready && same {
        return Ok(Some(first));
    }
    if mergeable {
        return Ok(Some((SymVal::Merge(block), any_dirty)));
    }
    Err(())
}

fn check_cand_flow(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    each: &Loop,
    shape: &LoopShape,
    cand: PromoteCand,
) -> Option<(Vec<Option<(SymVal, bool)>>, Vec<bool>)> {
    let n = func.blocks.len();
    let mut in_state = vec![None; n];
    let mut out_state = vec![None; n];
    let head_in_preds: Vec<Block> = cfg
        .preds_of(each.head)
        .iter()
        .copied()
        .filter(|p| is_in_loop(each, *p))
        .collect();
    let mergeable: Vec<bool> = (0..n).map(|i| can_merge_block(func, cfg, each, Block(i as u32))).collect();
    let zero_trip_ok = can_zero_trip_store(prov, cand);
    in_state[each.head.index()] = Some((SymVal::Phi, false));
    for _ in 0..=shape.rpo.len() + 2 {
        let mut changed = false;
        if in_state[each.head.index()].is_some_and(|(_, d)| !d)
            && head_in_preds.iter().any(|p| out_state[p.index()].is_some_and(|(_, d)| d))
        {
            let head_writes = out_state[each.head.index()].is_some_and(|(_, d)| d);
            if !zero_trip_ok && !head_writes {
                return None;
            }
            in_state[each.head.index()] = Some((SymVal::Phi, true));
            changed = true;
        }
        for &b in &shape.rpo {
            if b != each.head {
                let Ok(maybe_st) = join_block_preds(&out_state, &shape.in_preds[b.index()], b, mergeable[b.index()]) else {
                    return None;
                };
                let Some(st) = maybe_st else {
                    continue;
                };
                if in_state[b.index()] != Some(st) {
                    in_state[b.index()] = Some(st);
                    changed = true;
                }
            }
            let Some((mut val, mut dirty)) = in_state[b.index()] else { continue };
            for &inst in &func.blocks[b.index()].insts {
                let data = &func.insts[inst.index()];
                if let Some((true, Some(v))) = inst_matches_cand(func, prov, &data.op, data.result, cand) {
                    val = SymVal::Inst(v);
                    dirty = true;
                }
            }
            if out_state[b.index()] != Some((val, dirty)) {
                out_state[b.index()] = Some((val, dirty));
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if shape.rpo.iter().any(|b| out_state[b.index()].is_none()) {
        return None;
    }
    if shape.rpo.iter().any(|b| {
        !matches!(func.last(*b), Some(Op::Jump(..) | Op::Brif(..) | Op::Return(_) | Op::Trap))
    }) {
        return None;
    }
    let dirty_out = out_state.iter().map(|st| st.is_some_and(|(_, d)| d)).collect();
    Some((in_state, dirty_out))
}

fn base_mem_op(cand: PromoteCand, val: Option<Value>) -> Op {
    match (cand.base, val) {
        (PromoteBase::Slot(s), None) => Op::StackLoad(s, cand.offset),
        (PromoteBase::Addr(a), None) => Op::Load(a, cand.offset),
        (PromoteBase::Slot(s), Some(v)) => Op::StackStore(v, s, cand.offset),
        (PromoteBase::Addr(a), Some(v)) => Op::Store(v, a, cand.offset),
    }
}

fn push_edge_arg(op: &mut Op, branch: EdgeArm, arg: Value) {
    match (op, branch) {
        (Op::Jump(_, args), EdgeArm::Jump) => args.push(arg),
        (Op::Brif(_, _, then_args, _, _), EdgeArm::Then) => then_args.push(arg),
        (Op::Brif(_, _, _, _, else_args), EdgeArm::Else) => else_args.push(arg),
        _ => {}
    }
}

fn take_edge_to_pad(op: &mut Op, branch: EdgeArm, pad: Block) -> Vec<Value> {
    match (op, branch) {
        (Op::Jump(target, args), EdgeArm::Jump) => {
            *target = pad;
            std::mem::take(args)
        }
        (Op::Brif(_, then_b, then_args, _, _), EdgeArm::Then) => {
            *then_b = pad;
            std::mem::take(then_args)
        }
        (Op::Brif(_, _, _, else_b, else_args), EdgeArm::Else) => {
            *else_b = pad;
            std::mem::take(else_args)
        }
        _ => Vec::new(),
    }
}

fn emit_exit_stores(state: &mut PromoteState<'_>, cand: PromoteCand, exits: Vec<ExitEdge>) {
    for edge in exits {
        let store_op = base_mem_op(cand, Some(edge.val));
        let single = state.cfg.preds.get(edge.target.index()).is_some_and(|ps| ps.as_slice() == [edge.src])
            && state.func.entry != Some(edge.target);
        if single {
            let store_inst = state.func.add(store_op, None);
            state.func.blocks[edge.target.index()].insts.insert(0, store_inst);
            state.home.resize(state.func.types.len(), None);
            continue;
        }
        let Some(&term_inst) = state.func.blocks[edge.src.index()].insts.last() else { continue };
        let pad = Block(state.func.blocks.len() as u32);
        let moved_args = take_edge_to_pad(&mut state.func.insts[term_inst.index()].op, edge.branch, pad);
        let store_inst = state.func.add(store_op, None);
        let jump_inst = state.func.add(Op::Jump(edge.target, moved_args), None);
        state.func.blocks.push(BlockData { params: Vec::new(), insts: vec![store_inst, jump_inst] });
        state.home.resize(state.func.types.len(), None);
        let cfg = Arc::make_mut(&mut *state.cfg);
        if let Some(ps) = cfg.preds.get_mut(edge.target.index())
            && let Some(slot) = ps.iter_mut().find(|p| **p == edge.src)
        {
            *slot = pad;
        }
        cfg.preds.push(vec![edge.src]);
        cfg.reachable.push(true);
        cfg.idom.push(Some(edge.src));
        insert_rpo(&mut cfg.rpo, edge.src, pad, true);
        for outer in state.later.iter_mut() {
            let inside = is_in_loop(outer, edge.src) && is_in_loop(outer, edge.target);
            outer.body.resize(pad.index(), false);
            outer.body.push(inside);
            if outer.head == edge.target {
                outer.latches.iter_mut().filter(|l| **l == edge.src).for_each(|l| *l = pad);
            }
        }
    }
}

fn setup_preheader_phi(state: &mut PromoteState<'_>, pre: Block, cand: PromoteCand) -> Value {
    let init_val = state.func.value(cand.ty);
    let init_inst = state.func.add(base_mem_op(cand, None), Some(init_val));
    let pre_insts = &mut state.func.blocks[pre.index()].insts;
    let at = pre_insts.len().saturating_sub(1);
    pre_insts.insert(at, init_inst);
    if let Some(&pre_term) = pre_insts.last() {
        push_edge_arg(&mut state.func.insts[pre_term.index()].op, EdgeArm::Jump, init_val);
    }
    let phi = state.func.value(cand.ty);
    state.func.blocks[state.each.head.index()].params.push(phi);
    state.home.resize(state.func.types.len(), None);
    state.home[init_val.index()] = Some(pre);
    state.home[phi.index()] = Some(state.each.head);
    phi
}

fn rewrite_cand(
    state: &mut PromoteState<'_>,
    prov: &Provenance,
    pre: Block,
    shape: &LoopShape,
    cand: PromoteCand,
) -> bool {
    let Some((in_state, dirty_out)) = check_cand_flow(state.func, state.cfg, prov, state.each, shape, cand) else {
        return false;
    };
    let phi = setup_preheader_phi(state, pre, cand);
    let mut merge_phis = vec![None; state.func.blocks.len()];
    for &b in &shape.rpo {
        if let Some((SymVal::Merge(mb), _)) = in_state[b.index()]
            && mb == b
        {
            let m_phi = state.func.value(cand.ty);
            state.func.blocks[b.index()].params.push(m_phi);
            state.home.resize(state.func.types.len(), None);
            state.home[m_phi.index()] = Some(b);
            merge_phis[b.index()] = Some(m_phi);
        }
    }
    let mut subst = identity(state.func.types.len());
    let mut renamed = false;
    let mut exits = Vec::new();
    for &b in &shape.rpo {
        let Some((sym_in, _)) = in_state[b.index()] else { continue };
        let mut cur = match sym_in {
            SymVal::Phi => phi,
            SymVal::Merge(mb) => match merge_phis[mb.index()] {
                Some(m_phi) => m_phi,
                None => continue,
            },
            SymVal::Inst(v) => find(&subst, v),
        };
        let insts = std::mem::take(&mut state.func.blocks[b.index()].insts);
        let mut kept = Vec::with_capacity(insts.len());
        for inst in insts {
            let data = &state.func.insts[inst.index()];
            if let Some((is_write, stored)) = inst_matches_cand(state.func, prov, &data.op, data.result, cand) {
                if is_write {
                    if let Some(v) = stored {
                        cur = find(&subst, v);
                    }
                } else if let Some(res) = data.result {
                    subst[res.index()] = cur;
                    renamed = true;
                }
                continue;
            }
            kept.push(inst);
        }
        state.func.blocks[b.index()].insts = kept;
        if let Some(&term) = state.func.blocks[b.index()].insts.last() {
            if dirty_out[b.index()]
                && matches!(cand.base, PromoteBase::Addr(_))
                && matches!(state.func.insts[term.index()].op, Op::Return(_))
            {
                let store_inst = state.func.add(base_mem_op(cand, Some(cur)), None);
                let blk = &mut state.func.blocks[b.index()].insts;
                let at = blk.len() - 1;
                blk.insert(at, store_inst);
            }
            record_term_edges(state.func, state.each, &dirty_out, &merge_phis, (b, term, cur), &mut exits);
        }
    }
    emit_exit_stores(state, cand, exits);
    if renamed {
        rename(state.func, &subst);
    }
    true
}

fn needs_phi_arg(each: &Loop, merge_phis: &[Option<Value>], target: Block) -> bool {
    target == each.head || merge_phis.get(target.index()).copied().flatten().is_some()
}

fn record_term_edges(
    func: &mut Function,
    each: &Loop,
    dirty_out: &[bool],
    merge_phis: &[Option<Value>],
    (b, term, cur): (Block, Inst, Value),
    exits: &mut Vec<ExitEdge>,
) {
    let dirty = dirty_out[b.index()];
    match &mut func.insts[term.index()].op {
        Op::Jump(t, args) => {
            if needs_phi_arg(each, merge_phis, *t) {
                args.push(cur);
            } else if dirty && !is_in_loop(each, *t) {
                exits.push(ExitEdge { src: b, branch: EdgeArm::Jump, target: *t, val: cur });
            }
        }
        Op::Brif(_, t1, a1, t2, a2) => {
            if needs_phi_arg(each, merge_phis, *t1) {
                a1.push(cur);
            } else if dirty && !is_in_loop(each, *t1) {
                exits.push(ExitEdge { src: b, branch: EdgeArm::Then, target: *t1, val: cur });
            }
            if needs_phi_arg(each, merge_phis, *t2) {
                a2.push(cur);
            } else if dirty && !is_in_loop(each, *t2) {
                exits.push(ExitEdge { src: b, branch: EdgeArm::Else, target: *t2, val: cur });
            }
        }
        _ => {}
    }
}

fn promote_mem(state: &mut PromoteState<'_>, mem: &LoopMem, prov: &Provenance, pre: Block) -> bool {
    let cands = mem.promote_candidates(state.func, state.each, prov, state.home);
    if cands.is_empty() {
        return false;
    }
    let Some(shape) = loop_rpo_preds(state.func, state.cfg, state.each) else {
        return false;
    };
    let mut changed = false;
    for cand in cands {
        changed |= rewrite_cand(state, prov, pre, &shape, cand);
    }
    changed
}

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

fn can_hoist_global_load(func: &Function, mem: &LoopMem, prov: &Provenance, head_prefix: bool, inst: Inst) -> bool {
    let data = &func.insts[inst.index()];
    let Some(res) = data.result else { return false };
    let Op::Load(addr, offset) = data.op else { return false };
    if mem.calls || mem.atomics || mem.unknown_mem_writes {
        return false;
    }
    let width = func.ty(res).bytes();
    let deref_ok = in_range(i64::from(offset), width, prov.ro_bytes.get(addr.index()).copied().unwrap_or(0));
    if !head_prefix && !deref_ok {
        return false;
    }
    matches!(prov.loc(addr, i64::from(offset), width).root, MemRoot::Global(_))
}

const MAX_HOISTED_SYMBOLS: usize = 4;
const MAX_LOOP_GP_LIVE: usize = 18;
const MAX_LOOP_VEC_LIVE: usize = 11;

struct Invariants<'a> {
    prov: &'a Provenance,
    is_const: &'a mut Vec<bool>,
}

fn always_executes_in_loop(cfg: &Cfg, each: &Loop, blk: Block) -> bool {
    each.latches.iter().all(|&latch| cfg::dominates(&cfg.idom, blk, latch))
}

fn loop_live_counts(func: &Function, each: &Loop, is_const: &[bool], home: &[Option<Block>]) -> (usize, usize, Vec<u32>) {
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
                if home.get(vi).copied().flatten().is_none_or(|at| !is_in_loop(each, at)) {
                    count(v);
                }
            });
        }
    }
    (gp_live, vec_live, uses)
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
    let prov = inv.prov;
    let (gp_live, vec_live, loop_uses) = loop_live_counts(func, each, inv.is_const, home);
    let has_insert = (0..each.body.len())
        .filter(|&b| each.body[b])
        .flat_map(|b| &func.blocks[b].insts)
        .any(|inst| matches!(func.insts[inst.index()].op, Op::Insert(..)));
    let uses = has_insert.then(|| func.use_counts());
    let mut hoisted_vec: Vec<Value> = Vec::new();
    let mut hoisted_symbols = 0_usize;
    let mut hoisted_gp = 0_usize;
    let mut hoisted_vecs = 0_usize;
    let mut moved: Vec<Inst> = Vec::new();
    for &blk in &cfg.rpo {
        let block = blk.index();
        if !is_in_loop(each, blk) {
            continue;
        }
        let head = block == each.head.index();
        let always_exec = always_executes_in_loop(cfg, each, blk);
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut kept = Vec::with_capacity(insts.len());
        let mut head_prefix = head;
        for inst in insts {
            let op = &func.insts[inst.index()].op;
            if matches!(op, Op::Call { .. } | Op::TryCall { .. } | Op::Trap | Op::Trapz(_) | Op::Brif(..)) {
                head_prefix = false;
            }
            let is_symbol = matches!(op, Op::Symbol(_));
            let static_loc = matches!(op, Op::StackAddr(..) | Op::SlotAddr(_));
            let sym_ok = !mem.calls || (always_exec && hoisted_symbols < MAX_HOISTED_SYMBOLS);
            let res_ty = func.insts[inst.index()].result.map(|r| func.ty(r));
            let is_vec_mat = res_ty.is_some_and(Type::in_float_register)
                && matches!(op, Op::Insert(..) | Op::Shuffle(..) | Op::Const(_) | Op::Load(..) | Op::StackLoad(..));
            let replaces_single_use = matches!(op, Op::Insert(prev, ..) if hoisted_vec.contains(prev) && uses.as_ref().is_some_and(|u| u[prev.index()] == 1));
            let vec_ok = !is_vec_mat || replaces_single_use || vec_live + hoisted_vecs < MAX_LOOP_VEC_LIVE;
            let movable = (op.is_speculable() && !(mem.calls && static_loc) && (!is_symbol || sym_ok))
                || mem.can_hoist_load(func, prov, head_prefix, inst)
                || can_hoist_global_load(func, mem, prov, head_prefix, inst);
            let mut operands_invariant = true;
            let mut frees_operand = false;
            op.for_operands(|v| {
                operands_invariant &= home.get(v.index()).copied().flatten().is_none_or(|at| !is_in_loop(each, at));
                frees_operand |= loop_uses[v.index()] == 1 && !func.ty(v).in_float_register();
            });
            let adds_gp = !frees_operand && !matches!(op, Op::Const(_)) && res_ty.is_some_and(|ty| !ty.in_float_register());
            let gp_ok = !(adds_gp && op.is_speculable() && gp_live + hoisted_gp >= MAX_LOOP_GP_LIVE)
                && !(adds_gp
                    && mem.unique_heap_writes
                    && matches!(op, Op::Load(..))
                    && gp_live + hoisted_gp >= 12
                    && !mem.hoists_past_heap_writes(func, prov, head_prefix, inst));
            if movable && vec_ok && gp_ok && operands_invariant && let Some(res) = func.insts[inst.index()].result {
                hoisted_gp += usize::from(adds_gp);
                if is_symbol {
                    hoisted_symbols += 1;
                }
                if is_vec_mat {
                    if !replaces_single_use {
                        hoisted_vecs += 1;
                    }
                    hoisted_vec.push(res);
                }
                home[res.index()] = Some(entry);
                moved.push(inst);
            } else {
                kept.push(inst);
            }
        }
        func.blocks[block].insts = kept;
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
    func.blocks.iter().zip(&loops.depths).filter(|(_, depth)| **depth > 0).flat_map(|(block, _)| &block.insts).any(|inst| {
        matches!(func.insts[inst.index()].op, Op::Load(..) | Op::StackLoad(..) | Op::Store(..) | Op::StackStore(..) | Op::Move(..) | Op::Copy(..))
    })
}

struct ExitTest {
    head: Block,
    entry: Block,
    test: Block,
    cond: Value,
    exit_on_true: bool,
    exit: Block,
    exit_args: Vec<Value>,
}

fn exit_arm(each: &Loop, op: &Op) -> Option<(Value, bool, Block, Vec<Value>)> {
    let Op::Brif(cond, then, then_args, otherwise, otherwise_args) = op else { return None };
    match (is_in_loop(each, *then), is_in_loop(each, *otherwise)) {
        (false, true) => Some((*cond, true, *then, then_args.clone())),
        (true, false) => Some((*cond, false, *otherwise, otherwise_args.clone())),
        _ => None,
    }
}

fn exits_only_to(func: &Function, each: &Loop, block: Block, exit: Block, args: &[Value]) -> bool {
    let stays_or_matches = |target: &Block, target_args: &[Value]| is_in_loop(each, *target) || (*target == exit && target_args == args);
    match func.last(block) {
        Some(Op::Jump(target, target_args)) => stays_or_matches(target, target_args.as_slice()),
        Some(Op::Brif(_, then, then_args, otherwise, otherwise_args)) => {
            stays_or_matches(then, then_args.as_slice()) && stays_or_matches(otherwise, otherwise_args.as_slice())
        }
        _ => false,
    }
}

fn effect_free(func: &Function, block: Block) -> bool {
    func.blocks[block.index()].insts.split_last().is_some_and(|(_, body)| {
        body.iter().all(|inst| {
            let op = &func.insts[inst.index()].op;
            op.is_speculable() || matches!(op, Op::Load(..) | Op::StackLoad(..))
        })
    })
}

fn region_escapes(func: &Function, head: Block, exit: Block, region: &[bool], defs: &[Option<Block>]) -> bool {
    let from_region = |v: Value| defs.get(v.index()).copied().flatten().is_some_and(|b| region[b.index()]);
    let mut seen = vec![false; func.blocks.len()];
    seen[exit.index()] = true;
    let mut todo = vec![exit];
    while let Some(block) = todo.pop() {
        for &inst in &func.blocks[block.index()].insts {
            let mut hit = false;
            func.insts[inst.index()].op.for_operands(|v| hit |= from_region(v));
            if hit {
                return true;
            }
        }
        for next in func.successors(block) {
            if next != head && !std::mem::replace(&mut seen[next.index()], true) {
                todo.push(next);
            }
        }
    }
    false
}

fn find_exit_test(func: &Function, cfg: &Cfg, each: &Loop, defs: &[Option<Block>]) -> Option<ExitTest> {
    let entry = cfg::existing_preheader(func, cfg, each)?;
    let invariant = |v: Value| defs.get(v.index()).copied().flatten().is_none_or(|b| !is_in_loop(each, b));
    cfg.rpo.iter().copied().filter(|&b| is_in_loop(each, b)).find_map(|test| {
        let (cond, exit_on_true, exit, exit_args) = exit_arm(each, func.last(test)?)?;
        if !invariant(cond) || !exit_args.iter().all(|&v| invariant(v)) || !always_executes_in_loop(cfg, each, test) {
            return None;
        }
        let region: Vec<bool> = (0..func.blocks.len())
            .map(|b| Block(b as u32))
            .map(|b| is_in_loop(each, b) && (b == test || !cfg.dominates(test, b)))
            .collect();
        let acyclic = |b: Block| b == test || func.successors(b).all(|next| !(region[next.index()] && cfg.dominates(next, b)));
        let clean = (0..region.len())
            .map(|b| Block(b as u32))
            .filter(|b| region[b.index()])
            .all(|b| acyclic(b) && effect_free(func, b) && exits_only_to(func, each, b, exit, &exit_args));
        (clean && !region_escapes(func, each.head, exit, &region, defs)).then(|| ExitTest { head: each.head, entry, test, cond, exit_on_true, exit, exit_args })
    })
}

fn unswitch(func: &mut Function) -> bool {
    let mut changed = false;
    for _ in 0..64 {
        let cfg = func.cfg();
        let defs = func.def_blocks();
        let Some(found) = cfg.loops().list.iter().find_map(|each| find_exit_test(&*func, &cfg, each, &defs)) else {
            break;
        };
        let ExitTest { head, entry, test, cond, exit_on_true, exit, exit_args } = found;
        let (Some(&jump), Some(&branch)) = (func.blocks[entry.index()].insts.last(), func.blocks[test.index()].insts.last()) else {
            break;
        };
        let Op::Jump(_, head_args) = &func.insts[jump.index()].op else { break };
        let head_args = head_args.clone();
        let Op::Brif(_, then, then_args, otherwise, otherwise_args) = &func.insts[branch.index()].op else { break };
        let (stay, stay_args) = if exit_on_true { (*otherwise, otherwise_args.clone()) } else { (*then, then_args.clone()) };
        func.insts[jump.index()].op = if exit_on_true {
            Op::Brif(cond, exit, exit_args, head, head_args)
        } else {
            Op::Brif(cond, head, head_args, exit, exit_args)
        };
        func.insts[branch.index()].op = Op::Jump(stay, stay_args);
        func.bump_cfg();
        changed = true;
    }
    changed
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
    let prov = if loops_touch_memory(func, &found) { Provenance::new(func, &cfg) } else { Provenance::default() };
    let mut home = func.def_blocks();
    let mut is_const = vec![false; func.types.len()];
    for data in &func.insts {
        if let (&Op::Const(_), Some(res)) = (&data.op, data.result) {
            is_const[res.index()] = true;
        }
    }
    let mut inv = Invariants { prov: &prov, is_const: &mut is_const };
    let mut dirty = Domain::NONE;
    for idx in 0..found.list.len() {
        let before = func.blocks.len();
        let Some(pre) = cfg::preheader(func, &cfg, &found.list[idx]) else { continue };
        if func.blocks.len() > before {
            dirty |= Domain::CFG;
            home.resize(func.types.len(), Some(pre));
            let head_idx = found.list[idx].head.index();
            let mut pre_preds = Vec::new();
            let cfg = Arc::make_mut(&mut cfg);
            cfg.preds[head_idx].retain(|p| {
                let inside = is_in_loop(&found.list[idx], *p);
                if !inside {
                    pre_preds.push(*p);
                }
                inside
            });
            cfg.preds[head_idx].push(pre);
            cfg.preds.push(pre_preds);
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
        let mut state = PromoteState { func, cfg: &mut cfg, each, later, home: &mut home };
        let blocks_before = state.func.blocks.len();
        if promote_mem(&mut state, &mem, &prov, pre) {
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
    if dirty.contains(Domain::CFG) {
        func.bump_cfg();
    }
    if unswitch(func) {
        dirty |= Domain::CFG | Domain::VALUES;
    }
    dirty
}
