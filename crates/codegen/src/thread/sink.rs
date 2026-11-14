use crate::cfg::Cfg;
use crate::fold::for_edges;
use crate::ir::{Block, Function, Inst, Op, Value};

use super::carry::Carry;
use super::eval::{EvalState, ThreadCx, bind, lookup, lookup_const, seed_edge, step_inst};
use super::{Eval, body, emit_needed, emit_select, needed_ops, note_use, redefines, select_split, single_pred_brif_seed};

struct Edge {
    pred: Block,
    arm: usize,
    args: Vec<Value>,
    plain: EvalState,
    split: Option<(Value, EvalState, EvalState)>,
}

fn fed(func: &Function, defs: &[Option<Inst>], args: &[Value]) -> bool {
    args.iter().any(|arg| defs.get(arg.index()).copied().flatten().is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(_) | Op::Select(..))))
}

fn candidates(func: &Function, disp: Block) -> (Vec<(Inst, Value)>, Vec<usize>) {
    let data = &func.blocks[disp.index()];
    let mut local: Vec<usize> = data.insts.iter().flat_map(|inst| crate::live::results(&func.insts[inst.index()])).map(|value| value.index()).collect();
    local.sort_unstable();
    let mut found: Vec<(Inst, Value)> = Vec::new();
    for &inst in body(func, disp) {
        let op = &func.insts[inst.index()].op;
        let Some(result) = func.insts[inst.index()].result else { continue };
        let (mut reads, mut free) = (false, true);
        op.for_operands(|v| {
            let chained = found.iter().any(|&(_, value)| value == v);
            reads |= chained || data.params.contains(&v);
            free &= chained || local.binary_search(&v.index()).is_err();
        });
        if reads && free && op.is_speculable() {
            found.push((inst, result));
        }
    }
    (found, local)
}

fn evaluate(func: &Function, cx: &ThreadCx<'_>, disp: Block, found: &[(Inst, Value)], args: &[Value], seeds: &[(Value, bool)]) -> EvalState {
    let mut state = EvalState::default();
    for &(cond, taken) in seeds {
        seed_edge(func, cx, cond, taken, 0, (&mut state.local_consts, &mut state.edge_cmps));
    }
    if redefines(func, disp, &state.seeded()) {
        state = EvalState::default();
    }
    bind(cx, &mut state, &func.blocks[disp.index()].params, args);
    for &(inst, _) in found {
        let _ = step_inst(func, cx, inst, &mut state);
    }
    state
}

fn settled(state: &EvalState, local: &[usize], value: Value) -> bool {
    let known = |v: Value| state.local_consts.iter().any(|&(k, _)| k == v);
    let to = lookup(&state.map, value);
    known(value) || (to != value && (known(to) || local.binary_search(&to.index()).is_err()))
}

fn settles(edge: &Edge, local: &[usize], value: Value, plain: bool) -> bool {
    if plain {
        return settled(&edge.plain, local, value);
    }
    edge.split.as_ref().is_some_and(|(_, then, otherwise)| settled(then, local, value) && settled(otherwise, local, value))
}

fn incoming(func: &Function, cx: &ThreadCx<'_>, (cfg, single): (&Cfg, &[Option<Block>]), disp: Block, found: &[(Inst, Value)]) -> Option<Vec<Edge>> {
    let params = func.blocks[disp.index()].params.len();
    let mut preds = cfg.preds[disp.index()].clone();
    preds.sort_unstable();
    preds.dedup();
    let mut edges = Vec::new();
    for pred in preds {
        if pred == disp || !cfg.reachable[pred.index()] || crate::cfg::dominates(cx.idom, disp, pred) {
            return None;
        }
        let cond = match func.last(pred)? {
            Op::Jump(..) => None,
            Op::Brif(cond, ..) => Some(*cond),
            _ => return None,
        };
        let mut into: Vec<Option<Vec<Value>>> = Vec::new();
        for_edges(func, pred.index(), |target, args| into.push((target == disp).then(|| args.to_vec())));
        for (arm, args) in into.into_iter().enumerate() {
            let Some(args) = args else { continue };
            if args.len() != params {
                return None;
            }
            let seeds: Vec<(Value, bool)> = cond.map_or_else(|| single_pred_brif_seed(func, single, pred), |cond| Some((cond, arm == 0))).into_iter().collect();
            let plain = evaluate(func, cx, disp, found, &args, &seeds);
            let choice = if cond.is_none() { (0..args.len()).find_map(|at| select_split(func, cx.defs, &args, at)) } else { None };
            let split = choice.map(|(test, then, otherwise)| {
                let side = |taken: bool, given: &[Value]| {
                    let all: Vec<(Value, bool)> = std::iter::once((test, taken)).chain(seeds.iter().copied()).collect();
                    evaluate(func, cx, disp, found, given, &all)
                };
                (test, side(true, &then[..]), side(false, &otherwise[..]))
            });
            edges.push(Edge { pred, arm, args, plain, split });
        }
    }
    Some(edges)
}

fn emit(func: &mut Function, cx: &mut ThreadCx<'_>, pred: Block, disp: Block, state: EvalState, outputs: &[Value]) -> Vec<Value> {
    let chosen_args: Vec<Value> = outputs.iter().map(|&value| lookup(&state.map, value)).collect();
    let needed = needed_ops(&state, &chosen_args);
    let eval = Eval { chosen_target: disp, chosen_args, local_ops: state.local_ops, local_consts: state.local_consts, needed, resolved_branch: false, carry: Carry::default() };
    emit_needed(func, pred, (&mut *cx.consts, &mut *cx.uses), eval).1
}

fn same(func: &Function, cx: &ThreadCx<'_>, a: Value, b: Value) -> bool {
    a == b || (func.ty(a) == func.ty(b) && lookup_const(&[], cx.consts, a).is_some_and(|imm| lookup_const(&[], cx.consts, b) == Some(imm)))
}

fn pass(func: &mut Function, pred: Block, arm: usize, next: Vec<Value>) {
    let Some(&term) = func.blocks[pred.index()].insts.last() else { return };
    match (&mut func.insts[term.index()].op, arm) {
        (Op::Jump(_, args), _) | (Op::Brif(_, _, args, ..), 0) | (Op::Brif(.., args), _) => *args = next,
        _ => {}
    }
}

fn sink(func: &mut Function, cx: &mut ThreadCx<'_>, graph: (&Cfg, &[Option<Block>]), disp: Block, renames: &mut Vec<(Value, Value)>) -> bool {
    let (found, local) = candidates(func, disp);
    if found.is_empty() {
        return false;
    }
    let Some(mut edges) = incoming(func, cx, graph, disp, &found) else { return false };
    if !edges.iter().any(|edge| fed(func, cx.defs, &edge.args)) {
        return false;
    }
    let kept: Vec<(Inst, Value)> =
        found.into_iter().filter(|&(_, value)| edges.iter().all(|edge| settles(edge, &local, value, true) || settles(edge, &local, value, false))).collect();
    if kept.is_empty() {
        return false;
    }
    for edge in &mut edges {
        if kept.iter().all(|&(_, value)| settles(edge, &local, value, true)) {
            edge.split = None;
        } else if !kept.iter().all(|&(_, value)| settles(edge, &local, value, false)) {
            return false;
        }
    }
    let data = &func.blocks[disp.index()];
    let stays = |inst: &Inst| !kept.iter().any(|(gone, _)| gone == inst);
    let read = |value: Value| {
        cx.uses.get(value.index()).is_some_and(|mask| *mask != 0)
            || data.insts.iter().filter(|inst| stays(*inst)).any(|inst| {
                let mut hit = false;
                func.insts[inst.index()].op.for_operands(|v| hit |= v == value);
                hit
            })
    };
    let outputs: Vec<Value> = kept.iter().map(|&(_, value)| value).filter(|&value| read(value)).collect();
    let dead: Vec<bool> = data.params.iter().map(|&param| !read(param)).collect();
    for Edge { pred, arm, args, plain, split } in edges {
        let values = match split {
            None => emit(func, cx, pred, disp, plain, &outputs),
            Some((test, then, otherwise)) => {
                note_use(cx.uses, pred, test);
                let then = emit(func, cx, pred, disp, then, &outputs);
                let otherwise = emit(func, cx, pred, disp, otherwise, &outputs);
                let mut joined = Vec::with_capacity(then.len());
                for (a, b) in then.into_iter().zip(otherwise) {
                    joined.push(if same(func, cx, a, b) { a } else { emit_select(func, cx, pred, test, a, b) });
                }
                joined
            }
        };
        let mut next: Vec<Value> = args.iter().zip(&dead).filter(|&(_, &gone)| !gone).map(|(&arg, _)| arg).collect();
        next.extend(values);
        pass(func, pred, arm, next);
    }
    let block = &mut func.blocks[disp.index()];
    let mut at = 0;
    block.params.retain(|_| {
        at += 1;
        !dead[at - 1]
    });
    block.insts.retain(|inst| !kept.iter().any(|(gone, _)| gone == inst));
    for &(inst, _) in &kept {
        func.insts[inst.index()].op = Op::Trap;
    }
    for value in outputs {
        let param = func.value(func.ty(value));
        cx.consts.push(None);
        func.blocks[disp.index()].params.push(param);
        renames.push((value, param));
    }
    true
}

pub(super) fn wanted(func: &Function, defs: &[Option<Inst>], pinned: &[bool]) -> bool {
    (0..func.blocks.len()).any(|pred| {
        let mut hit = false;
        for_edges(func, pred, |target, args| hit |= !pinned[target.index()] && fed(func, defs, args) && !candidates(func, target).0.is_empty());
        hit
    })
}

pub(super) fn run(func: &mut Function, cx: &mut ThreadCx<'_>, graph: (&Cfg, &[Option<Block>])) -> bool {
    let mut renames: Vec<(Value, Value)> = Vec::new();
    let mut changed = false;
    for index in 0..func.blocks.len() {
        if !cx.pinned[index] && graph.0.reachable[index] && !func.blocks[index].params.is_empty() {
            changed |= sink(func, cx, graph, Block(index as u32), &mut renames);
        }
    }
    for &(old, new) in &renames {
        let mask = cx.uses.get(old.index()).copied().unwrap_or(0);
        if cx.uses.len() <= new.index() {
            cx.uses.resize(new.index() + 1, 0);
        }
        cx.uses[new.index()] |= mask;
    }
    if !renames.is_empty() {
        for block in &func.blocks {
            for inst in &block.insts {
                func.insts[inst.index()].op.map_operands(|v| lookup(&renames, v));
            }
        }
    }
    changed
}
