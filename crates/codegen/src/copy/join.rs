use super::{Edge, collect_sret_info, move_to_sret_src, resolve_slot, retarget_edge};
use crate::ir::{Block, BlockData, Function, Inst, Op, Value};

/// A copy into the return memory that `hoist` moved from the head of a join onto each edge that reaches it.
pub(super) struct Hoist {
    join: Block,
    copy: Inst,
    edges: Vec<(Block, Edge, Inst, Option<Block>)>,
}

/// Moves each copy of a join parameter into the return memory onto every edge that reaches the join, when each edge passes a slot address and nothing before the copy touches memory, so each slot meets its own copy.
pub(super) fn hoist(func: &mut Function, entry: Block, sret_ptr: Value, bytes: u32) -> Vec<Hoist> {
    let info = collect_sret_info(func, entry);
    let mut plans: Vec<(Block, Inst, Vec<(Block, Edge, Option<Value>)>)> = Vec::new();
    let mut reads = Vec::new();
    for &(join, pos, copy) in info.uses_of(sret_ptr) {
        let Some(param) = move_to_sret_src(func, &info.defs, copy, sret_ptr, bytes) else { continue };
        let block = &func.blocks[join.index()];
        if join == entry || !block.params.contains(&param) || !block.insts[..pos].iter().all(|&inst| pure(&func.insts[inst.index()].op)) {
            continue;
        }
        let Some(loads) = later_loads(func, &info, (join, pos), param, copy) else { continue };
        let at = block.params.iter().position(|&p| p == param).unwrap_or(usize::MAX);
        let mut edges = Vec::new();
        for &pred in info.preds_of(join) {
            let found = match func.last(pred) {
                Some(Op::Jump(target, args)) if *target == join => vec![(Edge::Jump, args.get(at).copied())],
                Some(Op::Brif(_, then, then_args, other, other_args)) => {
                    let mut found = Vec::new();
                    if *then == join {
                        found.push((Edge::Then, then_args.get(at).copied()));
                    }
                    if *other == join {
                        found.push((Edge::Else, other_args.get(at).copied()));
                    }
                    found
                }
                _ => vec![(Edge::Jump, None)],
            };
            for (edge, addr) in found {
                edges.push((pred, edge, addr.filter(|&addr| resolve_slot(func, &info.defs, addr).is_some())));
            }
        }
        if !edges.is_empty() && edges.iter().all(|&(.., addr)| addr.is_some()) && !plans.iter().any(|&(seen, ..)| seen == join) {
            plans.push((join, copy, edges));
            reads.extend(loads);
        }
    }
    for load in reads {
        if let Op::Load(from, _) = &mut func.insts[load.index()].op {
            *from = sret_ptr;
        }
    }
    let hoisted: Vec<Hoist> = plans.into_iter().map(|(join, copy, edges)| hoist_one(func, sret_ptr, bytes, join, copy, &edges)).collect();
    if !hoisted.is_empty() {
        func.bump_cfg();
    }
    hoisted
}

/// The loads of `param` in `join` after its copy at `pos`, when they are its only other uses and nothing between the copy and them writes memory, so each can read the copy's target instead.
fn later_loads(func: &Function, info: &super::SretUseInfo, (join, pos): (Block, usize), param: Value, copy: Inst) -> Option<Vec<Inst>> {
    let insts = &func.blocks[join.index()].insts;
    let mut loads = Vec::new();
    for &(block, at, user) in info.uses_of(param) {
        if user == copy {
            continue;
        }
        let quiet = |inst: &Inst| pure(&func.insts[inst.index()].op) || matches!(func.insts[inst.index()].op, Op::Load(..));
        if block != join || at <= pos || !matches!(func.insts[user.index()].op, Op::Load(from, _) if from == param) || !insts[pos + 1..at].iter().all(quiet) {
            return None;
        }
        loads.push(user);
    }
    Some(loads)
}

/// Whether `op` neither touches memory nor branches.
fn pure(op: &Op) -> bool {
    matches!(op, Op::Const(_) | Op::StackAddr(..) | Op::SlotAddr(_) | Op::Binary(..) | Op::Convert(..))
}

/// Removes `copy` from the head of `join` and puts a copy of each edge's slot before that edge.
fn hoist_one(func: &mut Function, sret_ptr: Value, bytes: u32, join: Block, copy: Inst, edges: &[(Block, Edge, Option<Value>)]) -> Hoist {
    func.blocks[join.index()].insts.retain(|&inst| inst != copy);
    let mut placed = Vec::new();
    for &(pred, edge, addr) in edges {
        let Some(addr) = addr else { continue };
        let moved = func.add(Op::Move(sret_ptr, addr, bytes), None);
        if edge == Edge::Jump {
            let insts = &mut func.blocks[pred.index()].insts;
            insts.insert(insts.len() - 1, moved);
            placed.push((pred, edge, moved, None));
            continue;
        }
        let Some(Op::Brif(_, then, then_args, other, other_args)) = func.last(pred) else { continue };
        let args = if edge == Edge::Then { then_args.clone() } else { other_args.clone() };
        let target = if edge == Edge::Then { *then } else { *other };
        let jump = func.add(Op::Jump(target, args), None);
        let block = Block(func.blocks.len() as u32);
        func.blocks.push(BlockData { params: Vec::new(), insts: vec![moved, jump] });
        retarget_edge(func, pred, edge, block, Vec::new());
        placed.push((pred, edge, moved, Some(block)));
    }
    Hoist { join, copy, edges: placed }
}

/// Puts back at its join each hoisted copy none of whose edge copies was forwarded; returns whether any hoist stayed.
pub(super) fn settle(func: &mut Function, hoisted: Vec<Hoist>) -> bool {
    let mut kept = false;
    for one in hoisted {
        if !one.edges.iter().all(|&(_, _, moved, _)| matches!(func.insts[moved.index()].op, Op::Move(..))) {
            kept = true;
            continue;
        }
        for (pred, edge, moved, block) in one.edges {
            match block {
                None => func.blocks[pred.index()].insts.retain(|&inst| inst != moved),
                Some(block) => {
                    let Some(&jump) = func.blocks[block.index()].insts.last() else { continue };
                    let Op::Jump(_, ref args) = func.insts[jump.index()].op else { continue };
                    let args = args.clone();
                    retarget_edge(func, pred, edge, one.join, args);
                    func.blocks[block.index()].insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
                }
            }
            func.insts[moved.index()].op = Op::Trap;
        }
        let at = func.blocks[one.join.index()].insts.iter().position(|&inst| !pure(&func.insts[inst.index()].op)).unwrap_or(0);
        func.blocks[one.join.index()].insts.insert(at, one.copy);
    }
    func.bump_cfg();
    kept
}
