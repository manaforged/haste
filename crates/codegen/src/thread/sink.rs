use crate::cfg::{Cfg, Loops};
use crate::fold::for_edges;
use crate::ir::{Block, BlockData, Domain, Function, Inst, Op, Value};

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
        let _ = step_inst(func, cx, None, inst, &mut state);
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
    let mut preds = cfg.preds_of(disp).to_vec();
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

fn can_split_exit(func: &Function, cfg: &Cfg, uses: &[u32], block: Block, (target, args): (Block, &[Value])) -> bool {
    if args.is_empty() || cfg.loops().depth_of(target) >= cfg.loops().depth_of(block) {
        return false;
    }
    body(func, block).iter().any(|&inst| {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { return false };
        !matches!(data.op, Op::Const(_) | Op::StackAddr(..))
            && data.op.is_speculable()
            && uses[res.index()] > 0
            && uses[res.index()] == args.iter().filter(|&&a| a == res).count() as u32
    })
}

fn add_jump_block(func: &mut Function, target: Block, args: Vec<Value>) -> Block {
    let split = Block(func.blocks.len() as u32);
    let jump = func.add(Op::Jump(target, args), None);
    func.blocks.push(BlockData { params: Vec::new(), insts: vec![jump] });
    split
}

fn split_cold_loop_exits(func: &mut Function) -> bool {
    let orig_len = func.blocks.len();
    let has_cand = {
        let cfg = func.cfg();
        !cfg.loops().list.is_empty()
            && (0..orig_len).any(|bi| {
                let block = Block(bi as u32);
                let depth = cfg.loops().depth_of(block);
                cfg.reachable[bi]
                    && depth > 0
                    && matches!(func.last(block), Some(Op::Brif(_, t_b, t_a, e_b, e_a)) if t_b != e_b && ((!t_a.is_empty() && cfg.loops().depth_of(*t_b) < depth) || (!e_a.is_empty() && cfg.loops().depth_of(*e_b) < depth)))
            })
    };
    if !has_cand {
        return false;
    }
    let uses = func.use_counts();
    let cfg = func.cfg().clone();
    let mut split_any = false;
    for bi in 0..orig_len {
        let block = Block(bi as u32);
        if !cfg.reachable[bi] || cfg.loops().depth_of(block) == 0 {
            continue;
        }
        let Some(&term) = func.blocks[bi].insts.last() else { continue };
        let Op::Brif(cond, t_b, t_a, e_b, e_a) = func.insts[term.index()].op.clone() else { continue };
        if t_b == e_b {
            continue;
        }
        let (mut nt_b, mut nt_a, mut ne_b, mut ne_a) = (t_b, t_a, e_b, e_a);
        let mut changed = false;
        if can_split_exit(func, &cfg, &uses, block, (nt_b, &nt_a)) {
            nt_b = add_jump_block(func, nt_b, std::mem::take(&mut nt_a));
            changed = true;
        }
        if can_split_exit(func, &cfg, &uses, block, (ne_b, &ne_a)) {
            ne_b = add_jump_block(func, ne_b, std::mem::take(&mut ne_a));
            changed = true;
        }
        if changed {
            func.insts[term.index()].op = Op::Brif(cond, nt_b, nt_a, ne_b, ne_a);
            split_any = true;
        }
    }
    split_any
}

fn loop_has_calls(func: &Function, loops: &Loops) -> Vec<bool> {
    loops
        .list
        .iter()
        .map(|each| {
            each.body.iter().enumerate().any(|(b, &in_l)| {
                in_l && func.blocks[b].insts.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Call { .. } | Op::TryCall { .. }))
            })
        })
        .collect()
}

fn vec_const_defs(func: &Function, cfg: &Cfg) -> Vec<bool> {
    let mut is_scalar_const = vec![false; func.types.len()];
    let mut is_vec_const = vec![false; func.types.len()];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let ty = func.ty(res);
            if ty.is_vector() {
                is_vec_const[res.index()] = match &data.op {
                    Op::Const(_) => true,
                    Op::Insert(base, scalar, _) => is_vec_const[base.index()] && is_scalar_const[scalar.index()],
                    Op::Shuffle(a, b, crate::ir::ShuffleMask::Const(_)) => is_vec_const[a.index()] && is_vec_const[b.index()],
                    Op::Convert(crate::ir::Convert::Bitcast, x) => is_vec_const[x.index()],
                    _ => false,
                };
            } else if matches!(data.op, Op::Const(_)) {
                is_scalar_const[res.index()] = true;
            }
        }
    }
    is_vec_const
}

fn loops_sink_ok(cfg: &Cfg, loop_calls: &[bool], (inner, outer): (Block, Block), (is_vec_const, is_const): (bool, bool)) -> bool {
    cfg.loops().list.iter().enumerate().all(|(idx, each)| {
        !each.body[inner.index()]
            || each.body[outer.index()]
            || (is_vec_const && loop_calls[idx])
            || (is_const && loop_calls[idx] && !crate::licm::always_executes_in_loop(cfg, each, inner))
    })
}

fn non_trivial_op(op: &Op) -> bool {
    !matches!(op, Op::Const(_) | Op::Select(..))
}

fn diamond_safe(func: &Function, cfg: &Cfg, (block, target): (Block, Block), (op, queued): (&Op, &[Inst])) -> bool {
    if cfg.loops().depth_of(target) != cfg.loops().depth_of(block)
        || cfg.pred_count(target) != 1
        || !matches!(func.last(target), Some(Op::Jump(..)))
        || !non_trivial_op(op)
    {
        return true;
    }
    let insts = body(func, target);
    if insts.iter().any(|&i| {
        let op = &func.insts[i.index()].op;
        !op.is_speculable() && !matches!(op, Op::Load(..) | Op::StackLoad(..))
    }) {
        return true;
    }
    let base = insts.iter().filter(|&&i| non_trivial_op(&func.insts[i.index()].op)).count();
    base >= 6 || base + queued.iter().filter(|&&i| non_trivial_op(&func.insts[i.index()].op)).count() < 6
}

fn cheap_call_addr(func: &Function, defs: &[Option<Inst>], op: &Op) -> bool {
    crate::licm::cheap_call_addr(op, |v| defs.get(v.index()).copied().flatten().is_some_and(|i| matches!(func.insts[i.index()].op, Op::Const(_))))
}

struct SinkInfo<'a> {
    defs: Option<&'a [Option<Inst>]>,
    loop_calls: &'a [bool],
    vec_consts: &'a [bool],
}

fn pick_sink_target(func: &Function, cfg: &Cfg, info: &SinkInfo<'_>, block: Block, sites: &[(Inst, Block)], (res, op, moves): (Value, &Op, &[Vec<Inst>])) -> Option<Block> {
    let mut lca = sites.first()?.1;
    for &(_, site) in sites {
        if !cfg.reachable[site.index()] {
            return None;
        }
        while !cfg.dominates(lca, site) {
            lca = cfg.idom[lca.index()]?;
        }
    }
    let is_vec_const = info.vec_consts.get(res.index()).copied().unwrap_or(false);
    let is_const = !is_vec_const && matches!(op, Op::Const(_));
    let mut target = lca;
    if !(target != block
        && !loops_sink_ok(cfg, info.loop_calls, (target, block), (false, false))
        && info.defs.is_some_and(|d| cheap_call_addr(func, d, op))
        && crate::live::has_regular_call(func, target))
    {
        while target != block && !loops_sink_ok(cfg, info.loop_calls, (target, block), (is_vec_const, is_const)) {
            target = cfg.idom[target.index()]?;
        }
    }
    if target == block || !cfg.dominates(block, target) {
        return None;
    }
    (is_vec_const || diamond_safe(func, cfg, (block, target), (op, &moves[target.index()]))).then_some(target)
}

fn build_users(func: &Function, spec_def: &[bool]) -> Vec<Vec<(Inst, Block)>> {
    let mut users = vec![Vec::new(); func.types.len()];
    for (bi, data) in func.blocks.iter().enumerate() {
        let blk = Block(bi as u32);
        for &inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|v| {
                if spec_def[v.index()] {
                    users[v.index()].push((inst, blk));
                }
            });
        }
    }
    users
}

fn topo_visit(func: &Function, moves: &[Inst], defs: &[(u32, usize)], state: &mut [u8], ordered: &mut Vec<Inst>, i: usize) {
    if state[i] != 0 {
        return;
    }
    state[i] = 1;
    func.insts[moves[i].index()].op.for_operands(|v| {
        if let Ok(pos) = defs.binary_search_by_key(&v.0, |&(r, _)| r) {
            topo_visit(func, moves, defs, state, ordered, defs[pos].1);
        }
    });
    state[i] = 2;
    ordered.push(moves[i]);
}

fn topo_sort_moves(func: &Function, moves: &mut Vec<Inst>) {
    let n = moves.len();
    if n <= 1 {
        return;
    }
    let mut defs: Vec<(u32, usize)> = moves
        .iter()
        .enumerate()
        .filter_map(|(i, &inst)| func.insts[inst.index()].result.map(|r| (r.0, i)))
        .collect();
    if defs.len() <= 1 {
        return;
    }
    defs.sort_unstable_by_key(|&(r, _)| r);
    let mut ordered = Vec::with_capacity(n);
    let mut state = vec![0_u8; n];
    for i in 0..n {
        topo_visit(func, moves, &defs, &mut state, &mut ordered, i);
    }
    *moves = ordered;
}

fn flush_moves(func: &mut Function, block: Block, moves: &mut Vec<Inst>) {
    if moves.is_empty() {
        return;
    }
    topo_sort_moves(func, moves);
    let mut next: Vec<Inst> = moves.drain(..).collect();
    next.extend_from_slice(&func.blocks[block.index()].insts);
    func.blocks[block.index()].insts = next;
}

fn sink_block(func: &mut Function, cfg: &Cfg, info: &SinkInfo<'_>, block: Block, has_child: bool, users: &mut [Vec<(Inst, Block)>], moves: &mut [Vec<Inst>]) -> bool {
    flush_moves(func, block, &mut moves[block.index()]);
    if !has_child {
        return false;
    }
    let bi = block.index();
    let len = func.blocks[bi].insts.len();
    let body_len = len.saturating_sub(1);
    if body_len == 0 {
        return false;
    }
    let mut kept: Vec<bool> = Vec::new();
    let mut touched: Vec<(usize, usize)> = Vec::new();
    for pos in (0..body_len).rev() {
        let inst = func.blocks[bi].insts[pos];
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if !data.op.is_speculable() {
            continue;
        }
        let Some(target) = pick_sink_target(func, cfg, info, block, &users[res.index()], (res, &data.op, moves)) else {
            continue;
        };
        if kept.is_empty() {
            kept.resize(len, true);
        }
        kept[pos] = false;
        let ti = target.index();
        if !touched.iter().any(|&(t, _)| t == ti) {
            touched.push((ti, moves[ti].len()));
        }
        moves[ti].push(inst);
        data.op.for_operands(|v| {
            for entry in &mut users[v.index()] {
                if entry.0 == inst {
                    entry.1 = target;
                }
            }
        });
    }
    for (ti, start) in touched {
        moves[ti][start..].reverse();
    }
    if !kept.is_empty() {
        let mut idx = 0;
        func.blocks[bi].insts.retain(|_| {
            let keep = kept[idx];
            idx += 1;
            keep
        });
        return true;
    }
    false
}

fn sink_dominated(func: &mut Function) -> bool {
    if func.blocks.len() <= 1 {
        return false;
    }
    let cfg = func.cfg().clone();
    let mut has_child = vec![false; func.blocks.len()];
    let mut any_child = false;
    for &b in &cfg.rpo {
        if let Some(p) = cfg.idom[b.index()]
            && p != b
        {
            has_child[p.index()] = true;
            any_child = true;
        }
    }
    if !any_child {
        return false;
    }
    let mut spec_def = vec![false; func.types.len()];
    let mut any_spec = false;
    for &b in &cfg.rpo {
        if !has_child[b.index()] {
            continue;
        }
        let insts = &func.blocks[b.index()].insts;
        for &inst in &insts[..insts.len().saturating_sub(1)] {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && data.op.is_speculable()
            {
                spec_def[res.index()] = true;
                any_spec = true;
            }
        }
    }
    if !any_spec {
        return false;
    }
    let has_loops = !cfg.loops().list.is_empty();
    let defs = has_loops.then(|| crate::fold::defs(func));
    let loop_calls = if has_loops { loop_has_calls(func, cfg.loops()) } else { Vec::new() };
    let vec_consts = if has_loops && func.types.iter().any(|ty| ty.is_vector()) { vec_const_defs(func, &cfg) } else { Vec::new() };
    let info = SinkInfo { defs: defs.as_deref(), loop_calls: &loop_calls, vec_consts: &vec_consts };
    let mut users = build_users(func, &spec_def);
    let mut moves = vec![Vec::new(); func.blocks.len()];
    let mut changed = false;
    for &block in &cfg.rpo {
        changed |= sink_block(func, &cfg, &info, block, has_child[block.index()], &mut users, &mut moves);
    }
    changed
}

pub(super) fn sink_cold(func: &mut Function) -> Domain {
    let mut out = Domain::NONE;
    if split_cold_loop_exits(func) {
        func.bump_cfg();
        out |= Domain::CFG | Domain::VALUES;
    }
    if sink_dominated(func) {
        out |= Domain::VALUES;
    }
    out
}
