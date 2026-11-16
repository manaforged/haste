use crate::cfg::Loop;
use crate::ir::{Binary, Block, Inst, IntCC, Op, Type, Value};

use super::{EdgeArm, Group, IvInfo, State, cached_end, edge_arg, edge_args_mut, emit_init_ptr, is_in_loop};

pub(super) struct LftrMatch {
    pub(super) blk: Block,
    pub(super) inst: Inst,
    pub(super) new_cc: IntCC,
    pub(super) use_next: bool,
    pub(super) len: Value,
}

fn prove_len_bound(state: &State<'_>, init_pred: Block, init_arm: EdgeArm, len: Value) -> (bool, bool) {
    if let Some(c) = state.const_i64(len) {
        return (c != 0, c >= 0);
    }
    let target_len = state.value_of(len);
    let check_brif = |blk: Block, taken_then: bool| -> Option<(bool, bool)> {
        let &term = state.func.blocks[blk.index()].insts.last()?;
        let Op::Brif(cond, t_blk, _, e_blk, _) = state.func.insts[term.index()].op else { return None };
        let c_inst = (t_blk != e_blk).then(|| state.defs.get(state.value_of(cond).index()).copied().flatten()).flatten()?;
        let Op::Icmp(cc, a, b) = state.func.insts[c_inst.index()].op else { return None };
        let (cc, other) = state.norm_cmp(cc, a, b, target_len)?;
        let eff_cc = if taken_then { cc } else { cc.inverse() };
        let c = state.const_i64(other)?;
        match eff_cc {
            IntCC::NotEqual if c == 0 => Some((true, false)),
            IntCC::UnsignedGreaterThan => Some((true, false)),
            IntCC::UnsignedGreaterThanOrEqual if c as u64 >= 1 => Some((true, false)),
            IntCC::SignedGreaterThan if c >= 0 => Some((true, true)),
            IntCC::SignedGreaterThanOrEqual if c >= 1 => Some((true, true)),
            _ => None,
        }
    };
    if !matches!(init_arm, EdgeArm::Jump)
        && let Some(b) = check_brif(init_pred, matches!(init_arm, EdgeArm::Then))
    {
        return b;
    }
    let mut cur = init_pred;
    for _ in 0..3 {
        let preds = state.cfg.preds_of(cur);
        if preds.len() != 1 {
            break;
        }
        let parent = preds[0];
        match state.func.last(parent) {
            Some(Op::Brif(_, t_blk, _, e_blk, _)) if *t_blk != *e_blk && (*t_blk == cur || *e_blk == cur) => {
                return check_brif(parent, *t_blk == cur).unwrap_or((false, false));
            }
            Some(Op::Jump(target, _)) if *target == cur => cur = parent,
            _ => break,
        }
    }
    (false, false)
}

pub(super) fn collect_lftr_matches(state: &State<'_>, each: &Loop, iv_info: &IvInfo, stride: i64) -> Vec<LftrMatch> {
    if iv_info.init_edges.len() != 1 || iv_info.back_steps.is_empty() {
        return Vec::new();
    }
    let (pre, pre_arm, init_val) = iv_info.init_edges[0];
    let step = iv_info.back_steps[0].2;
    if step == 0 || !iv_info.back_steps.iter().all(|&(_, _, s)| s == step) {
        return Vec::new();
    }
    let init_const = state.const_i64(init_val);
    let single_latch = (iv_info.back_steps.len() == 1 && step == 1).then_some(iv_info.back_steps[0]);
    let iv_next = single_latch.and_then(|(pred, arm, _)| edge_arg(state.func, pred, arm, iv_info.iv_idx)).map(|v| state.value_of(v));
    let mut matches = Vec::new();
    for &blk in state.cfg.rpo.iter().filter(|&&b| is_in_loop(each, b)) {
        for &inst in &state.func.blocks[blk.index()].insts {
            let data = &state.func.insts[inst.index()];
            let Some(cmp_res) = data.result.filter(|&r| state.value_of(r) == r) else { continue };
            let Op::Icmp(cc, a, b) = data.op else { continue };
            let Some((use_next, cc_norm, len)) = state
                .norm_cmp(cc, a, b, iv_info.iv)
                .map(|(c, o)| (false, c, o))
                .or_else(|| iv_next.and_then(|n| state.norm_cmp(cc, a, b, n).map(|(c, o)| (true, c, o))))
            else {
                continue;
            };
            let len_dom = state.const_i64(len).is_some()
                || state.def_blocks.get(len.index()).copied().flatten().is_some_and(|d| state.cfg.dominates(d, pre));
            let stride_ok = (stride & 1) != 0
                || (init_const == Some(0)
                    && step == 1
                    && state.const_i64(len).is_some_and(|c| c >= 0 && (c as u64).checked_mul(stride.unsigned_abs()).is_some()));
            if len == iv_info.iv || Some(len) == iv_next || state.func.ty(len) != Type::I64 || !len_dom || !stride_ok {
                continue;
            }
            let new_cc = match cc_norm {
                IntCC::Equal | IntCC::NotEqual if !use_next || single_latch.is_some_and(|(lb, _, _)| lb == blk) => cc_norm,
                IntCC::UnsignedLessThan | IntCC::UnsignedGreaterThanOrEqual | IntCC::SignedLessThan | IntCC::SignedGreaterThanOrEqual => {
                    if init_const != Some(0) || step != 1 {
                        continue;
                    }
                    let (nonzero, nonneg) = prove_len_bound(state, pre, pre_arm, len);
                    if matches!(cc_norm, IntCC::SignedLessThan | IntCC::SignedGreaterThanOrEqual) && !nonneg {
                        continue;
                    }
                    let Some(Op::Brif(cond, t_blk, _, e_blk, _)) = state.func.last(blk) else { continue };
                    if state.value_of(*cond) != cmp_res {
                        continue;
                    }
                    let exit_on_false = matches!(cc_norm, IntCC::UnsignedLessThan | IntCC::SignedLessThan);
                    let exit_blk = if exit_on_false { *e_blk } else { *t_blk };
                    let want_arm = if exit_on_false { EdgeArm::Then } else { EdgeArm::Else };
                    let arm_ok = if use_next {
                        single_latch.is_some_and(|(lb, la, _)| nonzero && blk == lb && la == want_arm)
                    } else {
                        blk == each.head
                    };
                    if !arm_ok || is_in_loop(each, exit_blk) {
                        continue;
                    }
                    if exit_on_false { IntCC::NotEqual } else { IntCC::Equal }
                }
                _ => continue,
            };
            matches.push(LftrMatch { blk, inst, new_cc, use_next, len });
        }
    }
    matches
}

pub(super) fn dead_iv_chain(state: &State<'_>, each: &Loop, iv_info: &IvInfo, groups: &[&Group], lftr: &[LftrMatch]) -> Option<Vec<Inst>> {
    let mut rem_uses = vec![0_u32; state.func.types.len()];
    let mut retired = vec![false; state.func.insts.len()];
    for &inst in state.func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &state.func.insts[inst.index()];
        if data.result.is_some_and(|r| state.value_of(r) != r) {
            retired[inst.index()] = true;
        } else {
            data.op.for_operands(|v| rem_uses[state.value_of(v).index()] += 1);
        }
    }
    let dec = |rem: &mut [u32], v: Value| {
        let i = state.value_of(v).index();
        rem[i] = rem[i].saturating_sub(1);
    };
    for &(pred, arm, _) in &iv_info.back_steps {
        if let Some(arg) = edge_arg(state.func, pred, arm, iv_info.iv_idx) {
            dec(&mut rem_uses, arg);
        }
    }
    for inst in groups.iter().flat_map(|g| g.occs.iter().map(|o| o.inst)).chain(lftr.iter().map(|m| m.inst)) {
        if !std::mem::replace(&mut retired[inst.index()], true) {
            state.func.insts[inst.index()].op.for_operands(|v| dec(&mut rem_uses, v));
        }
    }
    let mut chain = Vec::new();
    loop {
        let mut progressed = false;
        for &blk in state.cfg.rpo.iter().rev().filter(|&&b| is_in_loop(each, b)) {
            for &inst in state.func.blocks[blk.index()].insts.iter().rev() {
                let data = &state.func.insts[inst.index()];
                if !retired[inst.index()] && data.op.is_speculable() && data.result.is_some_and(|r| rem_uses[r.index()] == 0) {
                    retired[inst.index()] = true;
                    progressed = true;
                    chain.push(inst);
                    data.op.for_operands(|v| dec(&mut rem_uses, v));
                }
            }
        }
        if !progressed {
            break;
        }
    }
    (rem_uses[iv_info.iv.index()] == 0).then_some(chain)
}

fn same_steps(a: &IvInfo, b: &IvInfo) -> bool {
    let entry = |info: &IvInfo| info.init_edges.first().map(|&(pred, arm, _)| (pred, arm));
    a.init_edges.len() == 1 && b.init_edges.len() == 1 && entry(a) == entry(b) && a.back_steps == b.back_steps
}

fn next_before(state: &mut State<'_>, (latch, step): (Block, i64), sib: Value, next: Value, at: Inst) -> Value {
    let next = state.value_of(next);
    let insts = &state.func.blocks[latch.index()].insts;
    let late = state.def_blocks.get(next.index()).copied().flatten() == Some(latch)
        && state.defs.get(next.index()).copied().flatten().is_some_and(|def| insts.iter().position(|&i| i == def) > insts.iter().position(|&i| i == at));
    if !late {
        return next;
    }
    let c = state.emit_before_inst(latch, at, Op::Const(step), Type::I64);
    let fresh = state.emit_before_inst(latch, at, Op::Binary(Binary::Iadd, sib, c), Type::I64);
    state.subst[next.index()] = fresh;
    state.renamed = true;
    fresh
}

pub(super) fn retire_counter(state: &mut State<'_>, each: &Loop, edges: &[(Block, EdgeArm, bool)], ivs: &[IvInfo]) -> bool {
    if state.renamed {
        return false;
    }
    for counter in ivs {
        let Some(sib) = ivs.iter().find(|sib| sib.iv != counter.iv && same_steps(counter, sib)) else { continue };
        let lftr = collect_lftr_matches(state, each, counter, 1);
        let latch = counter.back_steps[0];
        let sib_next = edge_arg(state.func, latch.0, latch.1, sib.iv_idx);
        if lftr.is_empty() || (sib_next.is_none() && lftr.iter().any(|m| m.use_next)) {
            continue;
        }
        let Some(chain) = dead_iv_chain(state, each, counter, &[], &lftr) else { continue };
        let (pre, _, start) = counter.init_edges[0];
        let mut ends: Vec<(Value, Value)> = Vec::new();
        for m in &lftr {
            let len = state.value_of(m.len);
            let end = cached_end(state, &mut ends, len, |s| {
                let span = if s.const_i64(start) == Some(0) { len } else { s.emit_before_term(pre, Op::Binary(Binary::Isub, len, start), Type::I64) };
                emit_init_ptr(s, pre, sib.init_edges[0].2, span, 1, 0)
            });
            let lhs = match sib_next.filter(|_| m.use_next) {
                Some(next) => next_before(state, (latch.0, latch.2), sib.iv, next, m.inst),
                None => sib.iv,
            };
            state.func.insts[m.inst.index()].op = Op::Icmp(m.new_cc, lhs, end);
        }
        for inst in chain {
            if let Some(blk) = state.func.insts[inst.index()].result.and_then(|r| state.def_blocks[r.index()]) {
                state.func.blocks[blk.index()].insts.retain(|&i| i != inst);
            }
            state.func.insts[inst.index()].op = Op::Trap;
        }
        state.func.blocks[each.head.index()].params.remove(counter.iv_idx);
        for &(pred, arm, _) in edges {
            if let Some(args) = edge_args_mut(state.func, pred, arm).filter(|args| counter.iv_idx < args.len()) {
                args.remove(counter.iv_idx);
            }
        }
        return true;
    }
    false
}
