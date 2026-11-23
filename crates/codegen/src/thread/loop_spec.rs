use crate::cfg::Cfg;
use crate::fold::extend;
use crate::ir::{Block, BlockData, Function, Inst, Op, StackSlot, Type, Value};
use crate::mem::{MemRoot, Provenance};

use super::tail::{SsaRepairPlan, apply_ssa_repair, localize_consts, plan_ssa_joins};

const MAX_FUNC_BLOCKS: usize = 512;
const MAX_CLONE_BLOCKS: usize = 12;
const MAX_CLONE_INSTS: usize = 64;

#[derive(Clone, Copy)]
struct SlotConst {
    slot: StackSlot,
    off: i64,
    ty: Type,
    imm: i64,
}

fn stack_addr_of(func: &Function, defs: &[Option<Inst>], addr: Value) -> Option<(StackSlot, i64)> {
    let inst = defs.get(addr.index()).copied().flatten()?;
    match func.insts[inst.index()].op {
        Op::StackAddr(slot, off) => Some((slot, i64::from(off))),
        _ => None,
    }
}

fn pred_const_slots(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], pred: Block) -> Vec<SlotConst> {
    let Some((_, body)) = func.blocks[pred.index()].insts.split_last() else {
        return Vec::new();
    };
    let mut clobbered: Vec<StackSlot> = Vec::new();
    let mut out = Vec::new();
    for &inst in body.iter().rev() {
        match func.insts[inst.index()].op {
            Op::StackStore(val, slot, off) => {
                if !clobbered.contains(&slot) {
                    if let Some(imm) = consts.get(val.index()).copied().flatten()
                        && func.ty(val).is_small_int()
                    {
                        out.push(SlotConst { slot, off: i64::from(off), ty: func.ty(val), imm });
                    }
                    clobbered.push(slot);
                }
            }
            Op::Store(val, addr, off2) => {
                if let Some((slot, off1)) = stack_addr_of(func, defs, addr)
                    && let Some(off) = off1.checked_add(i64::from(off2))
                    && !clobbered.contains(&slot)
                {
                    if let Some(imm) = consts.get(val.index()).copied().flatten()
                        && func.ty(val).is_small_int()
                    {
                        out.push(SlotConst { slot, off, ty: func.ty(val), imm });
                    }
                    clobbered.push(slot);
                }
            }
            Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..) | Op::Move(..) => break,
            _ => {}
        }
    }
    out
}

fn matches_slot_load(func: &Function, defs: &[Option<Inst>], op: &Op, res: Value, sc: SlotConst) -> bool {
    if func.ty(res) != sc.ty {
        return false;
    }
    match *op {
        Op::StackLoad(slot, off) => slot == sc.slot && i64::from(off) == sc.off,
        Op::Load(addr, off2) => stack_addr_of(func, defs, addr)
            .and_then(|(s, o1)| o1.checked_add(i64::from(off2)).map(|o| (s, o)))
            .is_some_and(|(s, o)| s == sc.slot && o == sc.off),
        _ => false,
    }
}

fn block_clean(func: &Function, prov: &Provenance, block: Block, slot: StackSlot) -> bool {
    for &inst in &func.blocks[block.index()].insts {
        let op = &func.insts[inst.index()].op;
        match *op {
            Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..) | Op::Move(..) | Op::BrTable(..) => return false,
            Op::StackStore(_, s, _) | Op::StackAddr(s, _) if s == slot => return false,
            Op::Load(addr, _) => {
                let _ = addr;
            }
            _ => {
                let mut touches = false;
                op.for_operands(|v| {
                    if prov.loc(v, 0, 1).root == MemRoot::Slot(slot) {
                        touches = true;
                    }
                });
                if touches {
                    return false;
                }
            }
        }
    }
    true
}

fn decide_block_with_slot(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    block: Block,
    sc: SlotConst,
) -> Option<(Block, Vec<Value>)> {
    let (&last, body) = func.blocks[block.index()].insts.split_last()?;
    let Op::Brif(cond, then, ref then_args, other, ref other_args) = func.insts[last.index()].op else {
        return None;
    };
    if then == other {
        return None;
    }
    let mut local: Vec<(Value, i64)> = Vec::new();
    let mut hit = false;
    let val_const = |v: Value, local: &[(Value, i64)]| -> Option<i64> {
        local
            .iter()
            .rev()
            .find(|&&(k, _)| k == v)
            .map(|&(_, c)| c)
            .or_else(|| consts.get(v.index()).copied().flatten())
    };
    for &inst in body {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if matches_slot_load(func, defs, &data.op, res, sc) {
            local.push((res, sc.imm));
            hit = true;
            continue;
        }
        let folded = match data.op {
            Op::Icmp(cc, a, b) => val_const(a, &local)
                .zip(val_const(b, &local))
                .map(|(ca, cb)| i64::from(crate::fold::holds(cc, func.ty(a), ca, cb))),
            Op::Binary(bin, a, b) => val_const(a, &local)
                .zip(val_const(b, &local))
                .and_then(|(ca, cb)| crate::fold::constant(bin, ca, cb)),
            Op::Convert(crate::ir::Convert::Ireduce | crate::ir::Convert::Uextend, x) => {
                val_const(x, &local).map(|c| extend(c, func.ty(x), false))
            }
            Op::Convert(crate::ir::Convert::Sextend, x) => {
                val_const(x, &local).map(|c| extend(c, func.ty(x), true))
            }
            _ => None,
        };
        if let Some(c) = folded {
            local.push((res, c));
        }
    }
    if !hit {
        return None;
    }
    let c = val_const(cond, &local)?;
    let taken = extend(c, func.ty(cond), false) != 0;
    if taken {
        Some((then, then_args.clone()))
    } else {
        Some((other, other_args.clone()))
    }
}

struct SpecPlan {
    pred: Block,
    head: Block,
    sc: SlotConst,
    blocks: Vec<Block>,
    decided: Vec<Option<(Block, Vec<Value>)>>,
    repairs: Vec<(Block, Block, SsaRepairPlan)>,
}

fn find_plan(func: &Function) -> Option<SpecPlan> {
    if func.blocks.len() > MAX_FUNC_BLOCKS {
        return None;
    }
    let cfg = func.cfg();
    let loops = cfg.loops();
    if loops.list.is_empty() {
        return None;
    }
    let defs = crate::fold::defs(func);
    let consts = func.const_values();
    let mut prov: Option<Provenance> = None;

    for each in &loops.list {
        let head = each.head;
        let preds = cfg.preds_of(head);
        if preds.len() < 2 {
            continue;
        }
        for &pred in preds {
            if each.body[pred.index()] {
                continue;
            }
            let Some(Op::Jump(target, _)) = func.last(pred) else { continue };
            if *target != head {
                continue;
            }
            let cands = pred_const_slots(func, &defs, &consts, pred);
            if cands.is_empty() {
                continue;
            }
            let prov = prov.get_or_insert_with(|| Provenance::compute(func));
            let pred_last = func.blocks[pred.index()].insts.last().copied();
            for sc in cands {
                if !prov.slot_addr_unescaped(sc.slot) && prov.slot_escaped_at(sc.slot, pred_last) {
                    continue;
                }
                if let Some(plan) = try_build_plan(func, &cfg, prov, (&defs, &consts), (pred, head, sc)) {
                    return Some(plan);
                }
            }
        }
    }
    None
}

fn try_build_plan(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    (defs, consts): (&[Option<Inst>], &[Option<i64>]),
    (pred, head, sc): (Block, Block, SlotConst),
) -> Option<SpecPlan> {
    if !block_clean(func, prov, head, sc.slot) {
        return None;
    }
    let mut blocks = Vec::new();
    let mut decided = Vec::new();
    let mut in_r = vec![false; func.blocks.len()];
    let mut work = vec![head];
    let mut total_insts = 0usize;
    let mut has_backedge = false;
    let mut folded_any = false;

    while let Some(b) = work.pop() {
        if in_r[b.index()] {
            continue;
        }
        if !cfg.dominates(head, b) || !block_clean(func, prov, b, sc.slot) {
            continue;
        }
        if blocks.len() >= MAX_CLONE_BLOCKS || total_insts + func.blocks[b.index()].insts.len() > MAX_CLONE_INSTS {
            return None;
        }
        in_r[b.index()] = true;
        blocks.push(b);
        total_insts += func.blocks[b.index()].insts.len();
        if let Some((target, args)) = decide_block_with_slot(func, defs, consts, b, sc) {
            folded_any = true;
            if target == head {
                has_backedge = true;
            }
            decided.push(Some((target, args)));
        } else {
            decided.push(None);
            for s in func.successors(b) {
                if s == head {
                    has_backedge = true;
                } else if !in_r[s.index()] {
                    work.push(s);
                }
            }
        }
    }

    if !folded_any || !has_backedge {
        return None;
    }

    let mut exits: Vec<(Block, Block)> = Vec::new();
    for (idx, &b) in blocks.iter().enumerate() {
        if let Some((t, _)) = &decided[idx] {
            if !in_r[t.index()] {
                exits.push((b, *t));
            }
        } else {
            for s in func.successors(b) {
                if !in_r[s.index()] {
                    exits.push((b, s));
                }
            }
        }
    }

    let mut r_vals = Vec::new();
    for &b in &blocks {
        r_vals.extend(func.blocks[b.index()].params.iter().copied());
        for &inst in &func.blocks[b.index()].insts {
            if !matches!(func.insts[inst.index()].op, Op::Const(_))
                && let Some(r) = func.insts[inst.index()].result
            {
                r_vals.push(r);
            }
        }
    }

    let mut unique_targets: Vec<Block> = exits.iter().map(|&(_, t)| t).collect();
    unique_targets.sort_unstable_by_key(|b| b.0);
    unique_targets.dedup();

    let mut repairs: Vec<(Block, Block, SsaRepairPlan)> = Vec::new();
    for t in unique_targets {
        if !cfg.dominates(head, t) {
            continue;
        }
        let ssa = plan_ssa_joins(func, cfg, head, t, &blocks, &r_vals)?;
        if ssa.need.is_empty() {
            continue;
        }
        if repairs.len() >= 4 {
            return None;
        }
        let mut srcs = exits.iter().filter(|&&(_, dst)| dst == t).map(|&(src, _)| src);
        let src = srcs.next()?;
        if srcs.next().is_some() {
            return None;
        }
        if exits.iter().any(|&(os, ot)| (os != src || ot != t) && ssa.joins.contains(&ot)) {
            return None;
        }
        let disjoint = repairs.iter().all(|(_, _, prev)| {
            !prev.need.iter().any(|v| ssa.need.contains(v))
                && !prev.from_exit.iter().zip(&ssa.from_exit).any(|(&a, &b)| a && b)
        });
        if !disjoint {
            return None;
        }
        let src_idx = blocks.iter().position(|&b| b == src)?;
        let term_ok = decided[src_idx].is_some() || matches!(func.last(src), Some(Op::Jump(..) | Op::Brif(..)));
        if !term_ok {
            return None;
        }
        repairs.push((src, t, ssa));
    }

    Some(SpecPlan { pred, head, sc, blocks, decided, repairs })
}

fn apply_plan(func: &mut Function, plan: SpecPlan) {
    for &b in &plan.blocks {
        localize_consts(func, b);
    }
    let defs = crate::fold::defs(func);
    let mut block_map: Vec<(Block, Block)> = Vec::with_capacity(plan.blocks.len());
    let mut val_map: Vec<(Value, Value)> = Vec::new();
    for &b in &plan.blocks {
        let fresh_b = Block(func.blocks.len() as u32);
        block_map.push((b, fresh_b));
        let params: Vec<Value> = func.blocks[b.index()]
            .params
            .clone()
            .into_iter()
            .map(|p| {
                let fresh = func.value(func.ty(p));
                val_map.push((p, fresh));
                fresh
            })
            .collect();
        func.blocks.push(BlockData { params, insts: Vec::new() });
        let insts = func.blocks[b.index()].insts.clone();
        for inst in insts {
            if let Some(r) = func.insts[inst.index()].result {
                let fresh = func.value(func.ty(r));
                val_map.push((r, fresh));
            }
        }
    }
    let map_val = |v: Value| val_map.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into);
    let map_block = |b: Block| block_map.iter().find(|&&(from, _)| from == b).map_or(b, |&(_, into)| into);

    for (idx, &b) in plan.blocks.iter().enumerate() {
        let fresh_b = block_map[idx].1;
        let orig_insts = func.blocks[b.index()].insts.clone();
        let mut new_insts = Vec::with_capacity(orig_insts.len());
        for (pos, inst) in orig_insts.iter().copied().enumerate() {
            let is_last = pos + 1 == orig_insts.len();
            let data = func.insts[inst.index()].clone();
            let fresh_res = data.result.map(map_val);
            let op = if is_last && let Some((target, args)) = &plan.decided[idx] {
                Op::Jump(map_block(*target), args.iter().copied().map(map_val).collect())
            } else if let Some(orig_r) = data.result
                && matches_slot_load(func, &defs, &data.op, orig_r, plan.sc)
            {
                Op::Const(plan.sc.imm)
            } else {
                let mut op = data.op;
                op.map_operands(map_val);
                match &mut op {
                    Op::Jump(t, _) => *t = map_block(*t),
                    Op::Brif(_, t, _, o, _) => {
                        *t = map_block(*t);
                        *o = map_block(*o);
                    }
                    _ => {}
                }
                op
            };
            let added = func.add(op, fresh_res);
            func.insts[added.index()].loc = data.loc;
            new_insts.push(added);
        }
        func.blocks[fresh_b.index()].insts = new_insts;
    }

    let head_clone = map_block(plan.head);
    if let Some(&pred_last) = func.blocks[plan.pred.index()].insts.last()
        && let Op::Jump(ref mut target, _) = func.insts[pred_last.index()].op
        && *target == plan.head
    {
        *target = head_clone;
    }

    for (src, exit, ssa) in plan.repairs {
        let pairs: Vec<(Value, Value)> = ssa.need.iter().map(|&orig| (orig, map_val(orig))).collect();
        apply_ssa_repair(func, &ssa, exit, map_block(src), &pairs);
    }
    func.bump_cfg();
}

struct DupJoinPlan {
    pred: Block,
    join: Block,
    repairs: Vec<(Block, SsaRepairPlan)>,
}

fn collect_param_loads(func: &Function, blk: Block, params: &[Value], mapped: &[Value], out: &mut Vec<(Value, i32)>) -> bool {
    for &inst in &func.blocks[blk.index()].insts {
        match func.insts[inst.index()].op {
            Op::Load(addr, off) => {
                if let Some(pos) = params.iter().position(|&p| p == addr) {
                    out.push((mapped[pos], off));
                }
            }
            Op::Call { .. } | Op::TryCall { .. } => return false,
            ref op if crate::mem::is_atomic_barrier(op) => return false,
            _ => {}
        }
    }
    true
}

fn pred_forwards_join_load(func: &Function, pred: Block, join: Block, args: &[Value]) -> bool {
    let params = &func.blocks[join.index()].params;
    let mut wanted: Vec<(Value, i32)> = Vec::new();
    if !collect_param_loads(func, join, params, args, &mut wanted) {
        return false;
    }
    if wanted.is_empty()
        && func.blocks[join.index()].insts.len() <= 4
        && let Some(Op::Jump(next, next_args)) = func.last(join)
        && *next != join
        && *next != pred
        && func.blocks[next.index()].insts.len() <= 12
    {
        let next_params = &func.blocks[next.index()].params;
        if next_params.len() == next_args.len() {
            let mut next_wanted = Vec::new();
            for &inst in &func.blocks[next.index()].insts {
                match func.insts[inst.index()].op {
                    Op::Load(addr, off) => {
                        if let Some(np) = next_params.iter().position(|&p| p == addr)
                            && let Some(jp) = params.iter().position(|&p| p == next_args[np])
                        {
                            next_wanted.push((params[jp], args[jp], off));
                        }
                    }
                    Op::Call { .. } | Op::TryCall { .. } => return false,
                    ref op if crate::mem::is_atomic_barrier(op) => return false,
                    _ => {}
                }
            }
            for (jparam, parg, off) in next_wanted {
                let overwritten = func.blocks[join.index()].insts.iter().any(|&i| match func.insts[i.index()].op {
                    Op::Store(_, a, o) => a == jparam && o == off,
                    Op::Copy(..) | Op::Move(..) => true,
                    _ => false,
                });
                if !overwritten {
                    wanted.push((parg, off));
                }
            }
        }
    }
    if wanted.is_empty() {
        return false;
    }
    for &inst in func.blocks[pred.index()].insts.iter().rev() {
        match func.insts[inst.index()].op {
            Op::Store(_, addr, off) if wanted.contains(&(addr, off)) => return true,
            Op::Call { .. } | Op::TryCall { .. } | Op::Copy(..) | Op::Move(..) => return false,
            ref op if crate::mem::is_atomic_barrier(op) => return false,
            _ => {}
        }
    }
    false
}

fn find_forwarding_join(func: &Function) -> Option<DupJoinPlan> {
    let cfg = func.cfg();
    for at in 0..func.blocks.len() {
        let join = Block(at as u32);
        let jdata = &func.blocks[at];
        let npreds = cfg.preds_of(join).len();
        if !cfg.reachable[at] || jdata.params.is_empty() || jdata.insts.len() > 12 || !(2..=4).contains(&npreds) {
            continue;
        }
        let Some(&last) = jdata.insts.last() else { continue };
        let succs: Vec<Block> = match func.insts[last.index()].op {
            Op::Jump(t, _) if t != join => vec![t],
            Op::Brif(_, t, _, f, _) if t != f && t != join && f != join => vec![t, f],
            _ => continue,
        };
        let mut own_vals: Vec<Value> = jdata.params.clone();
        for &inst in &jdata.insts {
            if !matches!(func.insts[inst.index()].op, Op::Const(_))
                && let Some(r) = func.insts[inst.index()].result
            {
                own_vals.push(r);
            }
        }
        let mut repairs: Vec<(Block, SsaRepairPlan)> = Vec::new();
        let mut ssa_ok = true;
        for &t in &succs {
            if !cfg.dominates(join, t) {
                continue;
            }
            let Some(ssa) = plan_ssa_joins(func, &cfg, join, t, &[], &own_vals) else {
                ssa_ok = false;
                break;
            };
            if ssa.need.is_empty() {
                continue;
            }
            if repairs.iter().any(|(_, prev)| prev.from_exit.iter().zip(&ssa.from_exit).any(|(&a, &b)| a && b)) {
                ssa_ok = false;
                break;
            }
            repairs.push((t, ssa));
        }
        if !ssa_ok {
            continue;
        }
        for &pred in cfg.preds_of(join) {
            if pred == join || cfg.dominates(join, pred) {
                continue;
            }
            let Some(Op::Jump(target, args)) = func.last(pred) else { continue };
            if *target != join || args.len() != jdata.params.len() {
                continue;
            }
            if pred_forwards_join_load(func, pred, join, args) {
                return Some(DupJoinPlan { pred, join, repairs });
            }
        }
    }
    None
}

fn apply_forwarding_join(func: &mut Function, plan: DupJoinPlan) {
    localize_consts(func, plan.join);
    let Some(&jump) = func.blocks[plan.pred.index()].insts.last() else { return };
    let Op::Jump(_, ref args) = func.insts[jump.index()].op else { return };
    let mut map: Vec<(Value, Value)> = func.blocks[plan.join.index()].params.iter().copied().zip(args.iter().copied()).collect();
    let jinsts = func.blocks[plan.join.index()].insts.clone();
    let (&last, body) = jinsts.split_last().expect("invariant: join has terminator");
    let mut copied = Vec::with_capacity(body.len());
    for &inst in body {
        let data = func.insts[inst.index()].clone();
        let mut op = data.op;
        op.map_operands(|v| map.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
        let res = data.result.map(|r| {
            let fresh = func.value(func.ty(r));
            map.push((r, fresh));
            fresh
        });
        let added = func.add(op, res);
        func.insts[added.index()].loc = data.loc;
        copied.push(added);
    }
    let mut term = func.insts[last.index()].op.clone();
    term.map_operands(|v| map.iter().find(|&&(from, _)| from == v).map_or(v, |&(_, into)| into));
    func.insts[jump.index()].op = term;
    let pinsts = &mut func.blocks[plan.pred.index()].insts;
    let at = pinsts.len() - 1;
    pinsts.splice(at..at, copied);
    func.bump_cfg();
    for (exit, ssa) in plan.repairs {
        let pairs: Vec<(Value, Value)> = ssa
            .need
            .iter()
            .filter_map(|&orig| map.iter().find(|&&(from, _)| from == orig).map(|&(_, into)| (orig, into)))
            .collect();
        apply_ssa_repair(func, &ssa, exit, plan.pred, &pairs);
    }
    func.bump_cfg();
}

/// Specializes a small call-free inner loop on a pre-loop constant stack-slot invariant and duplicates small store-forwarding joins.
pub(super) fn run(func: &mut Function) -> bool {
    let mut any = false;
    for _ in 0..2 {
        let Some(plan) = find_plan(func) else { break };
        apply_plan(func, plan);
        any = true;
    }
    for _ in 0..4 {
        let Some(plan) = find_forwarding_join(func) else { break };
        apply_forwarding_join(func, plan);
        any = true;
    }
    any
}
