use super::symbolic::{evaluate, tagged};
use super::{MAX_CHAIN_BLOCKS, MAX_CHAIN_TRIPS, MAX_CHAIN_UNROLLED_INSTS, MAX_FP_FULL_INSTS, MAX_FP_FULL_TRIPS, MAX_VEC_FULL_TRIPS, block_has_float, block_has_libcall_mem, block_has_vec, collect_global_consts, edge_args, eval_block, has_bad_non_term, meet, outside_const_params, push_params_map, remap_val};
use crate::cfg::{Cfg, Loop};
use crate::ir::{Block, BlockData, Function, Op, Value};

struct Region {
    blocks: Vec<Block>,
    open: Vec<bool>,
}

impl Region {
    fn at(&self, block: Block) -> Option<usize> {
        self.blocks.iter().position(|&each| each == block)
    }
}

struct Step {
    reached: Vec<bool>,
    decided: Vec<Option<bool>>,
    consts: Vec<Vec<Option<i64>>>,
    residual: usize,
    next: Option<Vec<Option<i64>>>,
}

struct Proof {
    heads: Vec<Vec<Option<i64>>>,
    steps: Vec<Step>,
}

struct Rounds<'a> {
    region: &'a Region,
    base: usize,
    backs: usize,
    heads: &'a [Vec<Option<i64>>],
    steps: &'a [Step],
}

impl Rounds<'_> {
    fn block(&self, round: usize, at: usize) -> Block {
        match round {
            0 => self.region.blocks[at],
            _ => Block((self.base + (round - 1) * self.region.blocks.len() + at) as u32),
        }
    }

    fn image(&self, round: usize, target: Block) -> Block {
        let Some(at) = self.region.at(target) else { return target };
        let round = round + usize::from(at == 0);
        if round > self.backs { target } else { self.block(round, at) }
    }

    fn contains(&self, block: Block) -> bool {
        self.region.at(block).is_some() || block.index() >= self.base
    }
}

struct LiveOut {
    value: Value,
    def: Block,
    users: Vec<Block>,
}

fn inside(each: &Loop, block: Block) -> bool {
    each.body.get(block.index()).copied().unwrap_or(false)
}

fn takes_args(func: &Function, pred: Block) -> bool {
    matches!(func.last(pred), Some(Op::Jump(..) | Op::Brif(..)))
}

fn shape(func: &Function, cfg: &Cfg, each: &Loop) -> Option<Region> {
    let head = each.head;
    let blocks: Vec<Block> = cfg.rpo.iter().copied().filter(|&block| inside(each, block)).collect();
    if blocks.len() != each.size || blocks.len() > MAX_CHAIN_BLOCKS || blocks.first() != Some(&head) {
        return None;
    }
    for &block in &blocks {
        let clean = !func.blocks[block.index()].insts.is_empty()
            && takes_args(func, block)
            && !has_bad_non_term(func, block.index())
            && (block == head || cfg.preds_of(block).iter().all(|&pred| inside(each, pred)));
        if !clean {
            return None;
        }
    }
    let params = func.blocks[head.index()].params.len();
    if !each.latches.iter().all(|&latch| edge_args(func, latch, head).is_some_and(|args| args.len() == params)) {
        return None;
    }
    let open = blocks
        .iter()
        .enumerate()
        .map(|(at, &block)| at > 0 && cfg.preds_of(block).iter().any(|pred| blocks[at..].contains(pred)))
        .collect();
    Some(Region { blocks, open })
}

fn meet_known(acc: &mut Option<Vec<Option<i64>>>, vals: &[Option<i64>]) {
    match acc {
        None => *acc = Some(vals.to_vec()),
        Some(acc) => {
            for (slot, &val) in acc.iter_mut().zip(vals) {
                *slot = slot.filter(|&imm| val == Some(imm));
            }
        }
    }
}

fn record_block_consts(func: &Function, block: Block, step: &mut Step, at: usize, env: &[Option<i64>]) {
    let Some((_, body)) = func.blocks[block.index()].insts.split_last() else { return };
    let mut row = Vec::with_capacity(body.len());
    for &inst in body {
        let data = &func.insts[inst.index()];
        let imm = data.result.and_then(|res| env[res.index()]).filter(|&imm| !tagged(imm));
        if imm.is_none() && !matches!(data.op, Op::Const(_)) {
            step.residual += 1;
        }
        row.push(imm);
    }
    step.consts[at] = row;
}

fn step_edges(
    func: &Function,
    region: &Region,
    at: usize,
    env: &[Option<i64>],
    incoming: &mut [Option<Vec<Option<i64>>>],
) -> (Option<bool>, Option<Vec<Option<i64>>>) {
    let mut next = None;
    let mut edge = |target: Block, args: &[Value]| match region.at(target) {
        Some(0) => meet(&mut next, args, env),
        Some(to) => meet(&mut incoming[to], args, env),
        None => {}
    };
    let mut decided = None;
    match func.last(region.blocks[at]) {
        Some(Op::Jump(target, args)) => edge(*target, args),
        Some(Op::Brif(cond, then, then_args, otherwise, else_args)) => {
            let taken = env[cond.index()].map(|known| known != 0);
            if taken != Some(false) {
                edge(*then, then_args);
            }
            if taken != Some(true) {
                edge(*otherwise, else_args);
            }
            decided = taken;
        }
        _ => {}
    }
    (decided, next)
}

fn step(func: &Function, region: &Region, params: &[Option<i64>], env: &mut [Option<i64>]) -> Step {
    let n = region.blocks.len();
    let mut incoming: Vec<Option<Vec<Option<i64>>>> = vec![None; n];
    incoming[0] = Some(params.to_vec());
    let mut out = Step {
        reached: vec![false; n],
        decided: vec![None; n],
        consts: vec![Vec::new(); n],
        residual: 0,
        next: None,
    };
    for (at, &block) in region.blocks.iter().enumerate() {
        let Some(given) = incoming[at].take() else { continue };
        let given = if region.open[at] { vec![None; func.blocks[block.index()].params.len()] } else { given };
        out.reached[at] = true;
        eval_block(func, block.index(), &given, env, evaluate);
        record_block_consts(func, block, &mut out, at, env);
        let (decided, next) = step_edges(func, region, at, env, &mut incoming);
        out.decided[at] = decided;
        if let Some(vals) = next {
            meet_known(&mut out.next, &vals);
        }
    }
    out
}

fn prove(func: &Function, region: &Region, init: Vec<Option<i64>>, global_consts: &[Option<i64>]) -> Option<Proof> {
    let has_mem = region.blocks.iter().any(|&b| block_has_libcall_mem(func, b.index()));
    let has_fp = region.blocks.iter().any(|&b| block_has_float(func, b.index()));
    let has_vec = region.blocks.iter().any(|&b| block_has_vec(func, b.index()));
    let max_trips = if has_mem { 4 } else if has_fp { MAX_FP_FULL_TRIPS } else if has_vec { MAX_VEC_FULL_TRIPS } else { MAX_CHAIN_TRIPS };
    let max_insts = if has_fp { MAX_FP_FULL_INSTS } else { MAX_CHAIN_UNROLLED_INSTS };
    let mut heads = vec![init];
    let mut steps = Vec::new();
    let mut total_residual = 0_usize;
    while heads.len() <= max_trips {
        let mut env = global_consts.to_vec();
        let params = heads.last()?;
        let cur = step(func, region, params, &mut env);
        total_residual += cur.residual;
        if total_residual > max_insts {
            return None;
        }
        let next = cur.next.clone();
        steps.push(cur);
        let Some(next) = next else { return Some(Proof { heads, steps }) };
        if next == *params {
            return None;
        }
        heads.push(next);
    }
    None
}

fn live_outs(func: &Function, cfg: &Cfg, each: &Loop) -> Vec<LiveOut> {
    let defs = func.def_blocks();
    let mut live: Vec<LiveOut> = Vec::new();
    for (index, data) in func.blocks.iter().enumerate() {
        let user = Block(index as u32);
        if inside(each, user) || !cfg.reachable.get(index).copied().unwrap_or(false) {
            continue;
        }
        for inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|value| {
                let Some(def) = defs[value.index()].filter(|&def| inside(each, def)) else { return };
                match live.iter().position(|out| out.value == value) {
                    Some(at) if live[at].users.contains(&user) => {}
                    Some(at) => live[at].users.push(user),
                    None => live.push(LiveOut { value, def, users: vec![user] }),
                }
            });
        }
    }
    live
}

fn multi_pred(cfg: &Cfg, block: Block) -> bool {
    let mut first = None;
    for &pred in cfg.preds_of(block) {
        if !cfg.reachable.get(pred.index()).copied().unwrap_or(false) {
            continue;
        }
        match first {
            None => first = Some(pred),
            Some(prev) if prev != pred => return true,
            _ => {}
        }
    }
    false
}

fn collect_post(cfg: &Cfg, users: &[Block], in_region: &dyn Fn(Block) -> bool) -> (Vec<bool>, Vec<bool>) {
    let n = cfg.reachable.len();
    let mut post = vec![false; n];
    let mut exits = vec![false; n];
    let mut stack: Vec<Block> = users
        .iter()
        .copied()
        .filter(|&u| cfg.reachable.get(u.index()).copied().unwrap_or(false) && !in_region(u))
        .collect();
    while let Some(block) = stack.pop() {
        let idx = block.index();
        if post[idx] {
            continue;
        }
        post[idx] = true;
        for &pred in cfg.preds_of(block) {
            if !cfg.reachable.get(pred.index()).copied().unwrap_or(false) {
                continue;
            }
            if in_region(pred) {
                exits[idx] = true;
            } else if !post[pred.index()] {
                stack.push(pred);
            }
        }
    }
    (post, exits)
}

fn idf_hits_set(cfg: &Cfg, block: Block, in_s: &[bool], in_region: &dyn Fn(Block) -> bool) -> bool {
    let idom = cfg.idom[block.index()];
    for &pred in cfg.preds_of(block) {
        if !cfg.reachable.get(pred.index()).copied().unwrap_or(false) {
            continue;
        }
        let mut cur = Some(pred);
        while let Some(node) = cur {
            if Some(node) == idom || in_region(node) {
                break;
            }
            if in_s[node.index()] {
                return true;
            }
            cur = cfg.idom[node.index()].filter(|&p| p != node);
        }
    }
    false
}

fn post_exit_phis(cfg: &Cfg, users: &[Block], force_exits: bool, in_region: &dyn Fn(Block) -> bool) -> Vec<bool> {
    let (post, exits) = collect_post(cfg, users, in_region);
    let n = post.len();
    let mut in_s = exits.clone();
    let mut phi = vec![false; n];
    for idx in 0..n {
        if exits[idx] && (force_exits || multi_pred(cfg, Block(idx as u32))) {
            phi[idx] = true;
        }
    }
    let candidates: Vec<usize> = (0..n).filter(|&i| post[i] && !phi[i] && multi_pred(cfg, Block(i as u32))).collect();
    if candidates.is_empty() {
        return phi;
    }
    loop {
        let mut changed = false;
        for &idx in &candidates {
            if !phi[idx] && idf_hits_set(cfg, Block(idx as u32), &in_s, in_region) {
                phi[idx] = true;
                in_s[idx] = true;
                changed = true;
            }
        }
        if !changed {
            return phi;
        }
    }
}

fn repairable(func: &Function, cfg: &Cfg, each: &Loop, out: &LiveOut) -> bool {
    let phi = post_exit_phis(cfg, &out.users, true, &|block| inside(each, block));
    for (idx, &need) in phi.iter().enumerate() {
        if !need {
            continue;
        }
        let preds = cfg.preds_of(Block(idx as u32));
        let has_pred = preds.iter().any(|&p| cfg.reachable.get(p.index()).copied().unwrap_or(false));
        let all_ok = preds.iter().all(|&p| !cfg.reachable.get(p.index()).copied().unwrap_or(false) || takes_args(func, p));
        if !has_pred || !all_ok {
            return false;
        }
    }
    true
}

fn pin(func: &mut Function, head: Block, params: &[Value], known: &[Option<i64>], map: &mut Vec<(Value, Value)>) {
    for (&param, known) in params.iter().zip(known) {
        let Some(imm) = known.filter(|&imm| !tagged(imm)) else { continue };
        let value = func.value(func.ty(param));
        let inst = func.add(Op::Const(imm), Some(value));
        func.blocks[head.index()].insts.push(inst);
        map.push((param, value));
    }
}

fn clone_body_consts(
    func: &mut Function,
    block: Block,
    to: Block,
    consts: &[Option<i64>],
    map: &mut Vec<(Value, Value)>,
) {
    let Some((_, body)) = func.blocks[block.index()].insts.split_last() else { return };
    let body = body.to_vec();
    for (pos, orig) in body.into_iter().enumerate() {
        let mut op = match consts.get(pos).copied().flatten() {
            Some(imm) if !matches!(func.insts[orig.index()].op, Op::Const(_)) => Op::Const(imm),
            _ => func.insts[orig.index()].op.clone(),
        };
        op.map_operands(|value| remap_val(map, value));
        let new_res = func.insts[orig.index()].result.map(|res| {
            let fresh = func.value_like(res);
            map.push((res, fresh));
            fresh
        });
        let inst = func.add(op, new_res);
        func.blocks[to.index()].insts.push(inst);
    }
}

fn decide_term(op: Op, decided: Option<bool>) -> Op {
    match (decided, op) {
        (Some(true), Op::Brif(_, then, then_args, _, _)) => Op::Jump(then, then_args),
        (Some(false), Op::Brif(_, _, _, otherwise, else_args)) => Op::Jump(otherwise, else_args),
        (_, op) => op,
    }
}

fn clone_round_block(func: &mut Function, rounds: &Rounds<'_>, round: usize, at: usize, map: &mut Vec<(Value, Value)>) {
    let to = rounds.block(round, at);
    let step = &rounds.steps[round];
    if !step.reached[at] {
        let trap = func.add(Op::Trap, None);
        func.blocks[to.index()].insts.push(trap);
        return;
    }
    let block = rounds.region.blocks[at];
    clone_body_consts(func, block, to, &step.consts[at], map);
    let Some(&term) = func.blocks[block.index()].insts.last() else { return };
    let mut op = decide_term(func.insts[term.index()].op.clone(), step.decided[at]);
    op.map_operands(|value| remap_val(map, value));
    op.map_targets(|target| rounds.image(round, target));
    let inst = func.add(op, None);
    func.blocks[to.index()].insts.push(inst);
}

fn clone_rounds(func: &mut Function, rounds: &Rounds<'_>) -> Vec<Vec<(Value, Value)>> {
    let blocks = &rounds.region.blocks;
    let mut maps = Vec::with_capacity(rounds.backs);
    for round in 1..=rounds.backs {
        let mut map = Vec::new();
        for (at, &block) in blocks.iter().enumerate() {
            let params = func.blocks[block.index()].params.clone();
            let fresh: Vec<Value> = params.iter().map(|&param| func.value(func.ty(param))).collect();
            push_params_map(&mut map, &params, &fresh);
            func.blocks.push(BlockData { params: fresh, insts: Vec::new() });
            if at == 0 {
                pin(func, rounds.block(round, 0), &params, &rounds.heads[round], &mut map);
            }
        }
        maps.push(map);
    }
    for (index, map) in maps.iter_mut().enumerate() {
        let round = index + 1;
        for at in 0..blocks.len() {
            clone_round_block(func, rounds, round, at, map);
        }
    }
    maps
}

fn specialize_round_zero(func: &mut Function, rounds: &Rounds<'_>) {
    let step = &rounds.steps[0];
    for (at, &block) in rounds.region.blocks.iter().enumerate() {
        if let Some(&term) = func.blocks[block.index()].insts.last() {
            let decided = if step.reached[at] { step.decided[at] } else { None };
            let mut op = decide_term(func.insts[term.index()].op.clone(), decided);
            op.map_targets(|target| rounds.image(0, target));
            func.insts[term.index()].op = op;
        }
        if !step.reached[at] {
            continue;
        }
        for (pos, &imm) in step.consts[at].iter().enumerate() {
            let Some(imm) = imm else { continue };
            let inst = func.blocks[block.index()].insts[pos];
            if !matches!(func.insts[inst.index()].op, Op::Const(_)) {
                func.insts[inst.index()].op = Op::Const(imm);
            }
        }
    }
}

fn resolve_idom(cfg: &Cfg, starts: &[Option<Value>], ends: &[Option<Value>], fallback: Value, mut cur: Block) -> Value {
    for _ in 0..cfg.idom.len() {
        if let Some(param) = starts[cur.index()] {
            return param;
        }
        if let Some(def) = ends[cur.index()] {
            return def;
        }
        let Some(parent) = cfg.idom[cur.index()].filter(|&p| p != cur) else { break };
        cur = parent;
    }
    fallback
}

fn rewrite_users(func: &mut Function, cfg: &Cfg, out: &LiveOut, resolve: &dyn Fn(Block) -> Value) {
    for &user in &out.users {
        if !cfg.reachable.get(user.index()).copied().unwrap_or(false) {
            continue;
        }
        let new = resolve(user);
        if new == out.value {
            continue;
        }
        for at in 0..func.blocks[user.index()].insts.len() {
            let inst = func.blocks[user.index()].insts[at];
            func.insts[inst.index()].op.map_operands(|value| if value == out.value { new } else { value });
        }
    }
}

fn collect_active_ends(
    after: &Cfg,
    rounds: &Rounds<'_>,
    out: &LiveOut,
    maps: &[Vec<(Value, Value)>],
) -> (Vec<Option<Value>>, usize, Value) {
    let mut ends = vec![None; after.reachable.len()];
    let mut active = 0_usize;
    let mut only = out.value;
    let Some(at) = rounds.region.at(out.def) else { return (ends, 0, out.value) };
    if after.reachable.get(out.def.index()).copied().unwrap_or(false) {
        ends[out.def.index()] = Some(out.value);
        active += 1;
    }
    for (index, map) in maps.iter().enumerate() {
        let block = rounds.block(index + 1, at);
        if after.reachable.get(block.index()).copied().unwrap_or(false) {
            let val = remap_val(map, out.value);
            ends[block.index()] = Some(val);
            only = val;
            active += 1;
        }
    }
    (ends, active, only)
}

fn fill_phi_edges(func: &mut Function, after: &Cfg, phi: &[bool], resolve: &dyn Fn(Block) -> Value) {
    for (idx, &need) in phi.iter().enumerate() {
        if !need {
            continue;
        }
        let target = Block(idx as u32);
        let mut preds: Vec<Block> = after
            .preds_of(target)
            .iter()
            .copied()
            .filter(|p| after.reachable.get(p.index()).copied().unwrap_or(false))
            .collect();
        preds.sort_unstable();
        preds.dedup();
        for pred in preds {
            func.push_edge_arg(pred, target, resolve(pred));
        }
    }
}

fn repair_one(func: &mut Function, after: &Cfg, rounds: &Rounds<'_>, out: &LiveOut, maps: &[Vec<(Value, Value)>]) {
    let (ends, active, only) = collect_active_ends(after, rounds, out, maps);
    if active == 0 {
        return;
    }
    if active == 1 {
        rewrite_users(func, after, out, &|_| only);
        return;
    }
    let phi = post_exit_phis(after, &out.users, false, &|block| rounds.contains(block));
    let mut starts = vec![None; func.blocks.len()];
    for (idx, &need) in phi.iter().enumerate() {
        if need {
            let param = func.value(func.ty(out.value));
            func.blocks[idx].params.push(param);
            starts[idx] = Some(param);
        }
    }
    rewrite_users(func, after, out, &|user| resolve_idom(after, &starts, &ends, out.value, user));
    fill_phi_edges(func, after, &phi, &|pred| resolve_idom(after, &starts, &ends, out.value, pred));
}

fn rebind(func: &mut Function, after: &Cfg, rounds: &Rounds<'_>, live: &[LiveOut], maps: &[Vec<(Value, Value)>]) {
    for out in live {
        repair_one(func, after, rounds, out, maps);
    }
}

pub(super) fn unroll(func: &mut Function, cfg: &Cfg, each: &Loop) -> bool {
    let Some(entry) = func.entry else { return false };
    let Some(region) = shape(func, cfg, each) else { return false };
    let global_consts = &collect_global_consts(func, cfg, evaluate);
    let Some(init) = outside_const_params(func, cfg, each.head, &each.latches, global_consts) else { return false };
    let Some(proof) = prove(func, &region, init, global_consts) else { return false };
    let backs = proof.heads.len() - 1;
    let live = if backs == 0 { Vec::new() } else { live_outs(func, cfg, each) };
    if !live.iter().all(|out| repairable(func, cfg, each, out)) {
        return false;
    }
    let rounds = Rounds { region: &region, base: func.blocks.len(), backs, heads: &proof.heads, steps: &proof.steps };
    let maps = clone_rounds(func, &rounds);
    specialize_round_zero(func, &rounds);
    func.bump_cfg();
    crate::fold::prune(func, entry);
    if !live.is_empty() {
        let after = func.cfg();
        rebind(func, &after, &rounds, &live, &maps);
    }
    true
}
