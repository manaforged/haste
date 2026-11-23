use crate::cfg::{self, Cfg, Loop};
use crate::ir::{Block, BlockData, Domain, Function, Inst, Op, Value};
use crate::merge::{identity, rename};
use crate::unroll::remap_val;

fn safe_head_inst(func: &Function, inst: Inst) -> bool {
    let op = &func.insts[inst.index()].op;
    op.is_speculable() || matches!(op, Op::Load(..) | Op::StackLoad(..) | Op::Trapz(..) | Op::Assume(..))
}

struct RotatePlan {
    head: Block,
    body_entry: Block,
    exit: Block,
    body_on_true: bool,
    body_args: Vec<Value>,
    exit_args: Vec<Value>,
    cond: Value,
    head_body: Vec<Inst>,
    need_body: Vec<Value>,
    need_exit: Vec<Value>,
}

fn find_plan(func: &Function, cfg: &Cfg, each: &Loop, inner: bool) -> Option<RotatePlan> {
    let head = each.head;
    if Some(head) == func.entry || each.latches.is_empty() || each.latches.contains(&head) {
        return None;
    }
    let inside = |block: Block| each.body.get(block.index()).copied().unwrap_or(false);
    let head_insts = &func.blocks[head.index()].insts;
    let (&last, head_body) = head_insts.split_last()?;
    let exiting = each.latches.iter().any(|&latch| matches!(func.last(latch), Some(&Op::Brif(_, t, _, f, _)) if inside(t) != inside(f)));
    if head_body.len() > if exiting { 1 } else { 32 } || exiting && (each.size <= 2 || !inner) {
        return None;
    }
    let Op::Brif(cond, t_blk, ref t_args, f_blk, ref f_args) = func.insts[last.index()].op else {
        return None;
    };
    let (t_in, f_in) = (inside(t_blk), inside(f_blk));
    if t_in == f_in {
        return None;
    }
    let (body_entry, body_args, exit, exit_args, body_on_true) = if t_in {
        (t_blk, t_args.clone(), f_blk, f_args.clone(), true)
    } else {
        (f_blk, f_args.clone(), t_blk, t_args.clone(), false)
    };
    if body_entry == head || cfg.preds_of(body_entry) != [head] {
        return None;
    }
    let mut has_outside = false;
    for &p in cfg.preds_of(head) {
        if !cfg.reachable[p.index()] {
            return None;
        }
        if each.body[p.index()] {
            if !each.latches.contains(&p) {
                return None;
            }
        } else {
            has_outside = true;
        }
    }
    if !has_outside {
        return None;
    }
    let head_params = func.blocks[head.index()].params.len();
    for &latch in &each.latches {
        let ok = match func.last(latch) {
            Some(Op::Jump(target, args)) => *target == head && args.len() == head_params,
            Some(Op::Brif(_, t, ta, f, fa)) => {
                (*t == head || *f == head) && (*t != head || ta.len() == head_params) && (*f != head || fa.len() == head_params)
            }
            _ => false,
        };
        if !ok {
            return None;
        }
    }
    if !head_body.iter().all(|&i| safe_head_inst(func, i)) {
        return None;
    }
    if each.size == 2
        && head_body.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Load(..)))
        && head_body.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Trapz(..)))
    {
        return None;
    }
    let mut is_head_val = vec![false; func.types.len()];
    for &p in &func.blocks[head.index()].params {
        is_head_val[p.index()] = true;
    }
    for &i in head_body {
        if let Some(r) = func.insts[i.index()].result {
            is_head_val[r.index()] = true;
        }
    }
    let (mut need_body, mut need_exit) = (Vec::new(), Vec::new());
    let mut exit_uses_ok = true;
    let exit_preds = cfg.preds_of(exit);
    let exit_all_in_loop = !exit_preds.is_empty()
        && exit_preds.iter().all(|&p| {
            each.body.get(p.index()).copied().unwrap_or(false)
                && (p == head || matches!(func.last(p), Some(Op::Jump(..) | Op::Brif(..))))
        });
    let multi_exit_pred = exit_preds != [head];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        if !cfg.reachable[b_idx] || b_idx == head.index() {
            continue;
        }
        let in_loop = each.body[b_idx];
        for &inst in &block.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                if !is_head_val[v.index()] {
                    return;
                }
                if in_loop {
                    if !need_body.contains(&v) {
                        need_body.push(v);
                    }
                } else {
                    if !exit_all_in_loop || !cfg.dominates(exit, Block(b_idx as u32)) {
                        exit_uses_ok = false;
                    }
                    if !need_exit.contains(&v) {
                        need_exit.push(v);
                    }
                    if multi_exit_pred && !need_body.contains(&v) {
                        need_body.push(v);
                    }
                }
            });
            if !exit_uses_ok {
                return None;
            }
        }
    }
    Some(RotatePlan {
        head,
        body_entry,
        exit,
        body_on_true,
        body_args,
        exit_args,
        cond,
        head_body: head_body.to_vec(),
        need_body,
        need_exit,
    })
}

fn rewrite_blocks(func: &mut Function, each: &Loop, head: Block, body_remap: &[(Value, Value)], exit_remap: &[(Value, Value)]) {
    for b_idx in 0..func.blocks.len() {
        if b_idx == head.index() {
            continue;
        }
        let map = if each.body.get(b_idx).copied().unwrap_or(false) { body_remap } else { exit_remap };
        if map.is_empty() {
            continue;
        }
        for idx in 0..func.blocks[b_idx].insts.len() {
            let inst = func.blocks[b_idx].insts[idx];
            func.insts[inst.index()].op.map_operands(|v| remap_val(map, v));
        }
    }
}

fn clone_guard_into(func: &mut Function, target_blk: Block, plan: &RotatePlan, incoming_args: &[Value]) {
    let mut map: Vec<(Value, Value)> = func.blocks[plan.head.index()]
        .params
        .iter()
        .copied()
        .zip(incoming_args.iter().copied())
        .collect();
    if let Some(old_jump) = func.blocks[target_blk.index()].insts.pop() {
        func.insts[old_jump.index()].op = Op::Trap;
    }
    for &orig in &plan.head_body {
        let mut op = func.insts[orig.index()].op.clone();
        op.map_operands(|v| remap_val(&map, v));
        let res = func.insts[orig.index()].result.map(|r| {
            let nr = func.value_like(r);
            map.push((r, nr));
            nr
        });
        let ni = func.add(op, res);
        func.blocks[target_blk.index()].insts.push(ni);
    }
    let cond = remap_val(&map, plan.cond);
    let mut b_args: Vec<Value> = plan.body_args.iter().map(|&v| remap_val(&map, v)).collect();
    b_args.extend(plan.need_body.iter().map(|&v| remap_val(&map, v)));
    let mut e_args: Vec<Value> = plan.exit_args.iter().map(|&v| remap_val(&map, v)).collect();
    e_args.extend(plan.need_exit.iter().map(|&v| remap_val(&map, v)));
    let brif = if plan.body_on_true {
        Op::Brif(cond, plan.body_entry, b_args, plan.exit, e_args)
    } else {
        Op::Brif(cond, plan.exit, e_args, plan.body_entry, b_args)
    };
    let term = func.add(brif, None);
    func.blocks[target_blk.index()].insts.push(term);
}

fn split_latch_edge(func: &mut Function, plan: &RotatePlan, target: &mut Block, args: &mut Vec<Value>) {
    if *target != plan.head {
        return;
    }
    let split = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: Vec::new(), insts: Vec::new() });
    clone_guard_into(func, split, plan, args);
    *target = split;
    args.clear();
}

fn rotate_trap_prefix(func: &mut Function, cfg: &Cfg, each: &Loop) -> bool {
    if each.size != 2 || each.latches.len() != 1 {
        return false;
    }
    let (head, latch) = (each.head, each.latches[0]);
    if Some(head) == func.entry || latch == head {
        return false;
    }
    let Some(Op::Jump(lt, latch_args)) = func.last(latch).cloned() else { return false };
    let params = func.blocks[head.index()].params.clone();
    if lt != head || latch_args.len() != params.len() {
        return false;
    }
    let head_insts = &func.blocks[head.index()].insts;
    let Some((&last, head_body)) = head_insts.split_last() else { return false };
    if !matches!(func.insts[last.index()].op, Op::Brif(..)) || !head_body.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Load(..))) {
        return false;
    }
    let Some(trap_pos) = head_body.iter().position(|&i| matches!(func.insts[i.index()].op, Op::Trapz(_))) else {
        return false;
    };
    let Op::Trapz(cond) = func.insts[head_body[trap_pos].index()].op else { return false };
    let mut need_val = vec![false; func.types.len()];
    need_val[cond.index()] = true;
    let mut slice = Vec::new();
    for &inst in head_body[..trap_pos].iter().rev() {
        let data = &func.insts[inst.index()];
        if data.result.is_some_and(|r| need_val[r.index()]) {
            if !data.op.is_speculable() {
                return false;
            }
            data.op.for_operands(|v| need_val[v.index()] = true);
            slice.push(inst);
        }
    }
    slice.reverse();
    let Some(pre) = cfg::preheader(func, cfg, each) else { return false };
    let Some(Op::Jump(pt, pre_args)) = func.last(pre).cloned() else { return false };
    if pt != head || pre_args.len() != params.len() {
        return false;
    }
    let clone_slice = |func: &mut Function, dst: Block, incoming: &[Value], insert_before_last: bool| -> Value {
        let mut map: Vec<(Value, Value)> = params.iter().copied().zip(incoming.iter().copied()).collect();
        let saved = insert_before_last.then(|| func.blocks[dst.index()].insts.pop()).flatten();
        for &orig in &slice {
            let mut op = func.insts[orig.index()].op.clone();
            op.map_operands(|v| remap_val(&map, v));
            let res = func.insts[orig.index()].result.map(|r| {
                let nr = func.value_like(r);
                map.push((r, nr));
                nr
            });
            let ni = func.add(op, res);
            func.blocks[dst.index()].insts.push(ni);
        }
        let mapped_cond = remap_val(&map, cond);
        if let Some(j) = saved {
            let t = func.add(Op::Trapz(mapped_cond), None);
            func.blocks[dst.index()].insts.push(t);
            func.blocks[dst.index()].insts.push(j);
        }
        mapped_cond
    };
    let _ = clone_slice(func, pre, &pre_args, true);
    if let Some(old_jump) = func.blocks[latch.index()].insts.pop() {
        func.insts[old_jump.index()].op = Op::Trap;
    }
    let latch_cond = clone_slice(func, latch, &latch_args, false);
    let trap_blk = Block(func.blocks.len() as u32);
    let trap_inst = func.add(Op::Trap, None);
    func.blocks.push(BlockData { params: Vec::new(), insts: vec![trap_inst] });
    let brif = func.add(Op::Brif(latch_cond, head, latch_args, trap_blk, Vec::new()), None);
    func.blocks[latch.index()].insts.push(brif);
    let removed = func.blocks[head.index()].insts.remove(trap_pos);
    func.insts[removed.index()].op = Op::Trap;
    true
}

fn rotate_one(func: &mut Function, cfg: &Cfg, each: &Loop, inner: bool) -> bool {
    if rotate_trap_prefix(func, cfg, each) {
        return true;
    }
    let Some(plan) = find_plan(func, cfg, each, inner) else {
        return false;
    };
    let Some(pre) = cfg::preheader(func, cfg, each) else {
        return false;
    };
    let Some(Op::Jump(pre_target, pre_args)) = func.last(pre).cloned() else {
        return false;
    };
    if pre_target != plan.head || pre_args.len() != func.blocks[plan.head.index()].params.len() {
        return false;
    }
    let mut body_remap = Vec::with_capacity(plan.need_body.len());
    for &v in &plan.need_body {
        let np = func.value(func.ty(v));
        func.blocks[plan.body_entry.index()].params.push(np);
        body_remap.push((v, np));
    }
    let mut exit_remap = Vec::with_capacity(plan.need_exit.len());
    for &v in &plan.need_exit {
        let np = func.value(func.ty(v));
        func.blocks[plan.exit.index()].params.push(np);
        exit_remap.push((v, np));
    }
    rewrite_blocks(func, each, plan.head, &body_remap, &exit_remap);
    if !plan.need_exit.is_empty() {
        let extra: Vec<Value> = plan.need_exit.iter().map(|&v| remap_val(&body_remap, v)).collect();
        for &pred in cfg.preds_of(plan.exit) {
            if pred == plan.head {
                continue;
            }
            let Some(&last) = func.blocks[pred.index()].insts.last() else { continue };
            match &mut func.insts[last.index()].op {
                Op::Jump(t, args) if *t == plan.exit => args.extend_from_slice(&extra),
                Op::Brif(_, t, ta, f, fa) => {
                    if *t == plan.exit {
                        ta.extend_from_slice(&extra);
                    }
                    if *f == plan.exit {
                        fa.extend_from_slice(&extra);
                    }
                }
                _ => {}
            }
        }
    }
    clone_guard_into(func, pre, &plan, &pre_args);
    for &latch in &each.latches {
        match func.last(latch).cloned() {
            Some(Op::Jump(_, latch_args)) => clone_guard_into(func, latch, &plan, &latch_args),
            Some(Op::Brif(cond, mut t, mut ta, mut f, mut fa)) => {
                split_latch_edge(func, &plan, &mut t, &mut ta);
                split_latch_edge(func, &plan, &mut f, &mut fa);
                if let Some(&last) = func.blocks[latch.index()].insts.last() {
                    func.insts[last.index()].op = Op::Brif(cond, t, ta, f, fa);
                }
            }
            _ => {}
        }
    }
    for inst in std::mem::take(&mut func.blocks[plan.head.index()].insts) {
        if func.insts[inst.index()].op.is_terminator() {
            func.insts[inst.index()].op = Op::Trap;
        }
    }
    func.blocks[plan.head.index()].params.clear();
    true
}

struct NestPlan {
    head: Block,
    inner: Vec<Block>,
    fixed: u64,
}

fn has_bit(mask: u64, at: usize) -> bool {
    mask >> at & 1 == 1
}

fn pass_mask(func: &Function, latch: Block, head: Block) -> u64 {
    let params = &func.blocks[head.index()].params;
    let arm = |target: Block, args: &[Value]| {
        if target != head {
            return u64::MAX;
        }
        params.iter().zip(args).enumerate().filter(|(_, (p, a))| p == a).fold(0_u64, |mask, (at, _)| mask | 1 << at)
    };
    match func.last(latch) {
        Some(Op::Jump(t, ta)) => arm(*t, ta),
        Some(Op::Brif(_, t, ta, f, fa)) => arm(*t, ta) & arm(*f, fa),
        _ => 0,
    }
}

fn find_nest(func: &Function, cfg: &Cfg, each: &Loop) -> Option<NestPlan> {
    let head = each.head;
    let params = &func.blocks[head.index()].params;
    if each.latches.len() < 2 || params.is_empty() || params.len() > 64 || Some(head) == func.entry {
        return None;
    }
    if !cfg.preds_of(head).iter().all(|&p| matches!(func.last(p), Some(Op::Jump(..) | Op::Brif(..)))) {
        return None;
    }
    let masks: Vec<u64> = each.latches.iter().map(|&latch| pass_mask(func, latch, head)).collect();
    let common = masks.iter().fold(u64::MAX, |acc, m| acc & m);
    let covers = |s: u64| masks.iter().filter(|&&m| m & s == s).count();
    let best = masks.iter().map(|m| m & !common).filter(|&s| s != 0).max_by_key(|&s| (covers(s), s.count_ones()))?;
    let inner: Vec<Block> = each.latches.iter().zip(&masks).filter(|(_, m)| *m & best == best).map(|(latch, _)| *latch).collect();
    let body = cfg::natural_loop(cfg, head, inner.clone()).body;
    (hoists_from(func, &body, params, best | common, best) || rotatable_inner(func, head, &inner, &body))
        .then_some(NestPlan { head, inner, fixed: best | common })
}

fn rotatable_inner(func: &Function, head: Block, inner: &[Block], body: &[bool]) -> bool {
    let Some((&last, head_body)) = func.blocks[head.index()].insts.split_last() else { return false };
    let Op::Brif(_, t_blk, _, f_blk, _) = func.insts[last.index()].op else { return false };
    let inside = |b: Block| body.get(b.index()).copied().unwrap_or(false);
    let exiting = inner.iter().any(|&latch| matches!(func.last(latch), Some(&Op::Brif(_, t, _, f, _)) if inside(t) != inside(f)));
    !exiting && inside(t_blk) != inside(f_blk) && head_body.len() <= 32 && head_body.iter().all(|&i| safe_head_inst(func, i))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    Varies,
    Fixed,
    FromSplit,
}

fn hoists_from(func: &Function, body: &[bool], params: &[Value], fixed: u64, split: u64) -> bool {
    let mut reach: Vec<Reach> = func.def_blocks().iter().map(|def| if def.is_some_and(|b| body[b.index()]) { Reach::Varies } else { Reach::Fixed }).collect();
    for (at, &p) in params.iter().enumerate().filter(|(at, _)| has_bit(fixed, *at)) {
        reach[p.index()] = if has_bit(split, at) { Reach::FromSplit } else { Reach::Fixed };
    }
    let insts: Vec<Inst> = (0..body.len()).filter(|&b| body[b]).flat_map(|b| func.blocks[b].insts.iter().copied()).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &inst in &insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result.filter(|r| reach[r.index()] == Reach::Varies && data.op.is_speculable()) else { continue };
            let mut joined = Reach::Fixed;
            data.op.for_operands(|v| {
                joined = match (joined, reach[v.index()]) {
                    (Reach::Varies, _) | (_, Reach::Varies) => Reach::Varies,
                    (Reach::FromSplit, _) | (_, Reach::FromSplit) => Reach::FromSplit,
                    _ => Reach::Fixed,
                }
            });
            if joined == Reach::Varies {
                continue;
            }
            if joined == Reach::FromSplit && !matches!(data.op, Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) | Op::Convert(..) | Op::Const(_)) {
                return true;
            }
            reach[res.index()] = joined;
            changed = true;
        }
    }
    false
}

fn drop_fixed(args: &mut Vec<Value>, fixed: u64) {
    let mut at = 0;
    args.retain(|_| {
        at += 1;
        !has_bit(fixed, at - 1)
    });
}

fn nest_one(func: &mut Function, cfg: &Cfg, plan: &NestPlan) {
    let head = plan.head;
    let params = func.blocks[head.index()].params.clone();
    let outer_params: Vec<Value> = params.iter().map(|&p| func.value(func.ty(p))).collect();
    let mut jump_args = outer_params.clone();
    drop_fixed(&mut jump_args, plan.fixed);
    let jump = func.add(Op::Jump(head, jump_args), None);
    let outer = Block(func.blocks.len() as u32);
    func.blocks.push(BlockData { params: outer_params.clone(), insts: vec![jump] });
    let mut preds = cfg.preds_of(head).to_vec();
    preds.sort_unstable_by_key(|b| b.index());
    preds.dedup();
    for pred in preds {
        let Some(&last) = func.blocks[pred.index()].insts.last() else { continue };
        let op = &mut func.insts[last.index()].op;
        if !plan.inner.contains(&pred) {
            op.map_targets(|t| if t == head { outer } else { t });
            continue;
        }
        match op {
            Op::Jump(t, ta) if *t == head => drop_fixed(ta, plan.fixed),
            Op::Brif(_, t, ta, f, fa) => {
                if *t == head {
                    drop_fixed(ta, plan.fixed);
                }
                if *f == head {
                    drop_fixed(fa, plan.fixed);
                }
            }
            _ => {}
        }
    }
    let mut subst = identity(func.types.len());
    for (at, (&p, &q)) in params.iter().zip(&outer_params).enumerate() {
        if has_bit(plan.fixed, at) {
            subst[p.index()] = q;
        }
    }
    rename(func, &subst);
    drop_fixed(&mut func.blocks[head.index()].params, plan.fixed);
}

fn nest(func: &mut Function) -> bool {
    let mut changed = false;
    for _ in 0..64 {
        let cfg = func.cfg();
        let Some(plan) = cfg.loops().list.iter().find_map(|each| find_nest(&*func, &cfg, each)) else {
            break;
        };
        nest_one(func, &cfg, &plan);
        func.bump_cfg();
        changed = true;
    }
    changed
}

/// Separates nested loops that share a header, then rotates top-tested loops with small pure headers so loop guards precede entry and latches end in a single branch.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else {
        return Domain::NONE;
    };
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut changed = nest(func);
    for _ in 0..64 {
        let cfg = func.cfg();
        let loops = cfg.loops();
        let mut step = false;
        for each in &loops.list {
            let inner = !loops.list.iter().any(|other| other.head != each.head && each.body[other.head.index()]);
            if rotate_one(func, &cfg, each, inner) {
                step = true;
                break;
            }
        }
        if !step {
            break;
        }
        func.bump_cfg();
        changed = true;
    }
    if changed {
        crate::fold::prune(func, entry);
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
}
