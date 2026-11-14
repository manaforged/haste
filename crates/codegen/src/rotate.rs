use crate::cfg::{self, Cfg, Loop};
use crate::ir::{Block, BlockData, Domain, Function, Inst, Op, Value};
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

fn find_plan(func: &Function, cfg: &Cfg, each: &Loop) -> Option<RotatePlan> {
    let head = each.head;
    if Some(head) == func.entry || each.latches.is_empty() || each.latches.contains(&head) {
        return None;
    }
    let inside = |block: Block| each.body.get(block.index()).copied().unwrap_or(false);
    if each.latches.iter().any(|&latch| matches!(func.last(latch), Some(&Op::Brif(_, t, _, f, _)) if inside(t) != inside(f))) {
        return None;
    }
    let head_insts = &func.blocks[head.index()].insts;
    let (&last, head_body) = head_insts.split_last()?;
    if head_body.len() > 32 {
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
    let exit_single_pred = cfg.preds_of(exit) == [head];
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
                    if !exit_single_pred || !cfg.dominates(exit, Block(b_idx as u32)) {
                        exit_uses_ok = false;
                    }
                    if !need_exit.contains(&v) {
                        need_exit.push(v);
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
            let nr = func.value(func.ty(r));
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

fn rotate_one(func: &mut Function, cfg: &Cfg, each: &Loop) -> bool {
    let Some(plan) = find_plan(func, cfg, each) else {
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

/// Rotates top-tested loops with small pure headers so loop guards precede entry and latches end in a single branch.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else {
        return Domain::NONE;
    };
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut changed = false;
    for _ in 0..64 {
        let cfg = func.cfg();
        let loops = cfg.loops();
        let mut step = false;
        for each in &loops.list {
            if rotate_one(func, &cfg, each) {
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
