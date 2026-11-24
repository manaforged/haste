use crate::cfg::{self, Cfg, Loop};
use crate::ir::{Block, Function, Op, Value};
use crate::lsr::is_in_loop;

use super::always_executes_in_loop;

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
        (clean && !region_escapes(func, each.head, exit, &region, defs))
            .then(|| ExitTest { head: each.head, entry, test, cond, exit_on_true, exit, exit_args })
    })
}

pub(super) fn unswitch(func: &mut Function) -> bool {
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
