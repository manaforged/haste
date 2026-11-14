use rustc_hash::FxHashMap as HashMap;

use super::{MAX_CHAIN_BLOCKS, MAX_CHAIN_TRIPS, MAX_CHAIN_UNROLLED_INSTS, clone_body, edge_args, eval_block, has_bad_non_term, meet, outside_const_params, push_params_map, remap_val};
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
    decided: Vec<(usize, bool)>,
    next: Option<Vec<Option<i64>>>,
}

struct Proof {
    heads: Vec<Vec<Option<i64>>>,
    decided: Vec<(usize, bool)>,
}

struct Rounds<'a> {
    region: &'a Region,
    base: usize,
    backs: usize,
    heads: &'a [Vec<Option<i64>>],
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
}

struct LiveOut {
    value: Value,
    def: Block,
    users: Vec<Block>,
}

#[derive(Default)]
struct Repair {
    ends: HashMap<(Value, Block), Value>,
    starts: HashMap<(Value, Block), Value>,
    queue: Vec<(Value, Block)>,
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

fn step(func: &Function, region: &Region, params: &[Option<i64>], env: &mut [Option<i64>]) -> Step {
    let mut incoming: Vec<Option<Vec<Option<i64>>>> = vec![None; region.blocks.len()];
    incoming[0] = Some(params.to_vec());
    let mut step = Step { decided: Vec::new(), next: None };
    for (at, &block) in region.blocks.iter().enumerate() {
        let given = match incoming[at].take() {
            _ if region.open[at] => vec![None; func.blocks[block.index()].params.len()],
            Some(given) => given,
            None => continue,
        };
        eval_block(func, block.index(), &given, env);
        let mut edge = |target: Block, args: &[Value]| match region.at(target) {
            Some(0) => meet(&mut step.next, args, env),
            Some(to) => meet(&mut incoming[to], args, env),
            None => {}
        };
        match func.last(block) {
            Some(Op::Jump(target, args)) => edge(*target, args),
            Some(Op::Brif(cond, then, then_args, otherwise, else_args)) => {
                let taken = env[cond.index()].map(|known| known != 0);
                if taken != Some(false) {
                    edge(*then, then_args);
                }
                if taken != Some(true) {
                    edge(*otherwise, else_args);
                }
                step.decided.extend(taken.map(|taken| (at, taken)));
            }
            _ => {}
        }
    }
    step
}

fn prove(func: &Function, region: &Region, init: Vec<Option<i64>>, global_consts: &[Option<i64>]) -> Option<Proof> {
    let mut heads = vec![init];
    while heads.len() <= MAX_CHAIN_TRIPS {
        let mut env = global_consts.to_vec();
        let params = heads.last()?;
        let Step { decided, next } = step(func, region, params, &mut env);
        let Some(next) = next else { return Some(Proof { heads, decided }) };
        if next == *params {
            return None;
        }
        heads.push(next);
    }
    None
}

fn live_outs(func: &Function, each: &Loop) -> Vec<LiveOut> {
    let defs = func.def_blocks();
    let mut live: Vec<LiveOut> = Vec::new();
    for (index, data) in func.blocks.iter().enumerate() {
        let user = Block(index as u32);
        if inside(each, user) {
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

fn repairable(func: &Function, cfg: &Cfg, out: &LiveOut) -> bool {
    let mut seen = vec![false; func.blocks.len()];
    seen[out.def.index()] = true;
    let mut stack = out.users.clone();
    while let Some(block) = stack.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        let preds = cfg.preds_of(block);
        if preds.is_empty() || (preds.len() > 1 && !preds.iter().all(|&pred| takes_args(func, pred))) {
            return false;
        }
        stack.extend_from_slice(preds);
    }
    true
}

fn pin(func: &mut Function, head: Block, params: &[Value], known: &[Option<i64>], map: &mut Vec<(Value, Value)>) {
    for (&param, known) in params.iter().zip(known) {
        let Some(imm) = *known else { continue };
        let value = func.value(func.ty(param));
        let inst = func.add(Op::Const(imm), Some(value));
        func.blocks[head.index()].insts.push(inst);
        map.push((param, value));
    }
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
        for (at, &block) in blocks.iter().enumerate() {
            let Some((&term, body)) = func.blocks[block.index()].insts.split_last() else { continue };
            let body = body.to_vec();
            let to = rounds.block(round, at);
            clone_body(func, &body, to, map);
            let mut op = func.insts[term.index()].op.clone();
            op.map_operands(|value| remap_val(map, value));
            op.map_targets(|target| rounds.image(round, target));
            let inst = func.add(op, None);
            func.blocks[to.index()].insts.push(inst);
        }
    }
    maps
}

fn reach(func: &mut Function, cfg: &Cfg, repair: &mut Repair, var: Value, mut block: Block) -> Value {
    for _ in 0..cfg.preds.len() {
        if let Some(&known) = repair.starts.get(&(var, block)) {
            return known;
        }
        match *cfg.preds_of(block) {
            [] => return var,
            [pred] => {
                if let Some(&def) = repair.ends.get(&(var, pred)) {
                    return def;
                }
                block = pred;
            }
            _ => {
                let param = func.value(func.ty(var));
                func.blocks[block.index()].params.push(param);
                repair.starts.insert((var, block), param);
                repair.queue.push((var, block));
                return param;
            }
        }
    }
    var
}

fn push_edge(func: &mut Function, pred: Block, target: Block, arg: Value) {
    let Some(&term) = func.blocks[pred.index()].insts.last() else { return };
    match &mut func.insts[term.index()].op {
        Op::Jump(to, args) if *to == target => args.push(arg),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            if *then == target {
                then_args.push(arg);
            }
            if *otherwise == target {
                else_args.push(arg);
            }
        }
        _ => {}
    }
}

fn rebind(func: &mut Function, cfg: &Cfg, rounds: &Rounds<'_>, live: &[LiveOut], maps: &[Vec<(Value, Value)>]) {
    let mut repair = Repair::default();
    for out in live {
        let Some(at) = rounds.region.at(out.def) else { continue };
        repair.ends.insert((out.value, out.def), out.value);
        for (index, map) in maps.iter().enumerate() {
            repair.ends.insert((out.value, rounds.block(index + 1, at)), remap_val(map, out.value));
        }
    }
    for out in live {
        for &user in &out.users {
            let new = reach(func, cfg, &mut repair, out.value, user);
            for at in 0..func.blocks[user.index()].insts.len() {
                let inst = func.blocks[user.index()].insts[at];
                func.insts[inst.index()].op.map_operands(|value| if value == out.value { new } else { value });
            }
        }
    }
    let mut next = 0;
    while let Some(&(var, block)) = repair.queue.get(next) {
        next += 1;
        let mut preds = cfg.preds_of(block).to_vec();
        preds.sort_unstable();
        preds.dedup();
        for pred in preds {
            let arg = match repair.ends.get(&(var, pred)).copied() {
                Some(def) => def,
                None => reach(func, cfg, &mut repair, var, pred),
            };
            push_edge(func, pred, block, arg);
        }
    }
}

pub(super) fn unroll(func: &mut Function, cfg: &Cfg, each: &Loop, global_consts: &[Option<i64>]) -> bool {
    let Some(entry) = func.entry else { return false };
    let Some(region) = shape(func, cfg, each) else { return false };
    let Some(init) = outside_const_params(func, cfg, each.head, &each.latches, global_consts) else { return false };
    let Some(proof) = prove(func, &region, init, global_consts) else { return false };
    let backs = proof.heads.len() - 1;
    let body_insts: usize = region.blocks.iter().map(|block| func.blocks[block.index()].insts.len() - 1).sum();
    if (backs + 1) * body_insts > MAX_CHAIN_UNROLLED_INSTS {
        return false;
    }
    let live = if backs == 0 { Vec::new() } else { live_outs(func, each) };
    if !live.iter().all(|out| repairable(func, cfg, out)) {
        return false;
    }
    let rounds = Rounds { region: &region, base: func.blocks.len(), backs, heads: &proof.heads };
    let maps = clone_rounds(func, &rounds);
    for &latch in &each.latches {
        if let Some(&term) = func.blocks[latch.index()].insts.last() {
            func.insts[term.index()].op.map_targets(|target| rounds.image(0, target));
        }
    }
    for &(at, taken) in &proof.decided {
        let last = rounds.block(backs, at);
        if let Some(&term) = func.blocks[last.index()].insts.last()
            && let Op::Brif(_, then, then_args, otherwise, else_args) = func.insts[term.index()].op.clone()
        {
            func.insts[term.index()].op = if taken { Op::Jump(then, then_args) } else { Op::Jump(otherwise, else_args) };
        }
    }
    func.bump_cfg();
    crate::fold::prune(func, entry);
    if !live.is_empty() {
        let after = func.cfg();
        rebind(func, &after, &rounds, &live, &maps);
    }
    true
}
