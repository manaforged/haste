use std::sync::Arc;

use super::insert_rpo;
use crate::cfg::{Cfg, Loop};
use crate::ir::{Block, BlockData, Function, Inst, Op, Value};
use crate::lsr::{EdgeArm, is_in_loop};
use crate::mem::{LoopMem, MemLoc, PromoteBase, PromoteCand, Provenance, in_range};
use crate::merge::{find, identity, rename};

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

pub(super) struct PromoteState<'a> {
    pub(super) func: &'a mut Function,
    pub(super) cfg: &'a mut Arc<Cfg>,
    pub(super) each: &'a Loop,
    pub(super) later: &'a mut [Loop],
    pub(super) home: &'a mut Vec<Option<Block>>,
}

fn loop_rpo_preds(func: &Function, cfg: &Cfg, each: &Loop) -> Option<LoopShape> {
    let n = func.blocks.len();
    for b in (0..each.body.len()).filter(|b| each.body[*b]) {
        let block = Block(b as u32);
        let needs_args = func
            .successors(block)
            .any(|t| t == each.head || !is_in_loop(each, t));
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
    Some(LoopShape {
        rpo: post,
        in_preds,
    })
}

fn inst_matches_cand(
    func: &Function,
    prov: &Provenance,
    op: &Op,
    res: Option<Value>,
    cand: PromoteCand,
) -> Option<(bool, Option<Value>)> {
    let (loc, ty, is_write, stored) = match (op, res) {
        (&Op::StackLoad(s, off), Some(r)) => (
            MemLoc::slot(s, i64::from(off), func.ty(r).bytes()),
            func.ty(r),
            false,
            None,
        ),
        (&Op::Load(a, off), Some(r)) => (
            prov.loc(a, i64::from(off), func.ty(r).bytes()),
            func.ty(r),
            false,
            None,
        ),
        (&Op::StackStore(v, s, off), _) => (
            MemLoc::slot(s, i64::from(off), func.ty(v).bytes()),
            func.ty(v),
            true,
            Some(v),
        ),
        (&Op::Store(v, a, off), _) => (
            prov.loc(a, i64::from(off), func.ty(v).bytes()),
            func.ty(v),
            true,
            Some(v),
        ),
        _ => return None,
    };
    ((loc.base, loc.offset, ty) == (cand.loc.base, cand.loc.offset, cand.ty))
        .then_some((is_write, stored))
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
            is_in_loop(each, *p) && matches!(func.last(*p), Some(Op::Jump(..) | Op::Brif(..)))
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
    let mergeable: Vec<bool> = (0..n)
        .map(|i| can_merge_block(func, cfg, each, Block(i as u32)))
        .collect();
    let zero_trip_ok = can_zero_trip_store(prov, cand);
    in_state[each.head.index()] = Some((SymVal::Phi, false));
    for _ in 0..=shape.rpo.len() + 2 {
        let mut changed = false;
        if in_state[each.head.index()].is_some_and(|(_, d)| !d)
            && head_in_preds
                .iter()
                .any(|p| out_state[p.index()].is_some_and(|(_, d)| d))
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
                let Ok(maybe_st) = join_block_preds(
                    &out_state,
                    &shape.in_preds[b.index()],
                    b,
                    mergeable[b.index()],
                ) else {
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
            let Some((mut val, mut dirty)) = in_state[b.index()] else {
                continue;
            };
            for &inst in &func.blocks[b.index()].insts {
                let data = &func.insts[inst.index()];
                if let Some((true, Some(v))) =
                    inst_matches_cand(func, prov, &data.op, data.result, cand)
                {
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
        !matches!(
            func.last(*b),
            Some(Op::Jump(..) | Op::Brif(..) | Op::Return(_) | Op::Trap)
        )
    }) {
        return None;
    }
    let dirty_out = out_state
        .iter()
        .map(|st| st.is_some_and(|(_, d)| d))
        .collect();
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
        let single = edge.target.index() < state.cfg.reachable.len()
            && state.cfg.preds_of(edge.target) == [edge.src]
            && state.func.entry != Some(edge.target);
        if single {
            let store_inst = state.func.add(store_op, None);
            state.func.blocks[edge.target.index()]
                .insts
                .insert(0, store_inst);
            state.home.resize(state.func.types.len(), None);
            continue;
        }
        let Some(&term_inst) = state.func.blocks[edge.src.index()].insts.last() else {
            continue;
        };
        let pad = Block(state.func.blocks.len() as u32);
        let moved_args = take_edge_to_pad(
            &mut state.func.insts[term_inst.index()].op,
            edge.branch,
            pad,
        );
        let store_inst = state.func.add(store_op, None);
        let jump_inst = state.func.add(Op::Jump(edge.target, moved_args), None);
        state.func.blocks.push(BlockData {
            params: Vec::new(),
            insts: vec![store_inst, jump_inst],
        });
        state.home.resize(state.func.types.len(), None);
        let cfg = Arc::make_mut(&mut *state.cfg);
        cfg.replace_pred_and_add(edge.target, edge.src, pad);
        cfg.reachable.push(true);
        cfg.idom.push(Some(edge.src));
        insert_rpo(&mut cfg.rpo, edge.src, pad, true);
        for outer in state.later.iter_mut() {
            let inside = is_in_loop(outer, edge.src) && is_in_loop(outer, edge.target);
            outer.body.resize(pad.index(), false);
            outer.body.push(inside);
            if outer.head == edge.target {
                outer
                    .latches
                    .iter_mut()
                    .filter(|l| **l == edge.src)
                    .for_each(|l| *l = pad);
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
        push_edge_arg(
            &mut state.func.insts[pre_term.index()].op,
            EdgeArm::Jump,
            init_val,
        );
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
    let Some((in_state, dirty_out)) =
        check_cand_flow(state.func, state.cfg, prov, state.each, shape, cand)
    else {
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
        let Some((sym_in, _)) = in_state[b.index()] else {
            continue;
        };
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
            if let Some((is_write, stored)) =
                inst_matches_cand(state.func, prov, &data.op, data.result, cand)
            {
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
            record_term_edges(
                state.func,
                state.each,
                &dirty_out,
                &merge_phis,
                (b, term, cur),
                &mut exits,
            );
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
                exits.push(ExitEdge {
                    src: b,
                    branch: EdgeArm::Jump,
                    target: *t,
                    val: cur,
                });
            }
        }
        Op::Brif(_, t1, a1, t2, a2) => {
            if needs_phi_arg(each, merge_phis, *t1) {
                a1.push(cur);
            } else if dirty && !is_in_loop(each, *t1) {
                exits.push(ExitEdge {
                    src: b,
                    branch: EdgeArm::Then,
                    target: *t1,
                    val: cur,
                });
            }
            if needs_phi_arg(each, merge_phis, *t2) {
                a2.push(cur);
            } else if dirty && !is_in_loop(each, *t2) {
                exits.push(ExitEdge {
                    src: b,
                    branch: EdgeArm::Else,
                    target: *t2,
                    val: cur,
                });
            }
        }
        _ => {}
    }
}

pub(super) fn promote_mem(
    state: &mut PromoteState<'_>,
    mem: &LoopMem,
    prov: &Provenance,
    pre: Block,
) -> bool {
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
