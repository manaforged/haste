use crate::cfg::{Cfg, Loops};
use crate::ir::{Block, Function, Op};

pub(crate) fn cold(func: &Function) -> Vec<bool> {
    let cfg = func.cfg();
    cold_with(func, &cfg)
}

fn cold_with(func: &Function, cfg: &Cfg) -> Vec<bool> {
    let mut cold = vec![true; func.blocks.len()];
    let mut work = Vec::new();
    for b in 0..func.blocks.len() {
        let block = Block(b as u32);
        if matches!(func.last(block), Some(Op::Return(_))) {
            cold[b] = false;
            work.push(block);
        }
    }
    while let Some(block) = work.pop() {
        for &pred in cfg.preds_of(block) {
            if std::mem::replace(&mut cold[pred.index()], false) {
                work.push(pred);
            }
        }
    }
    if let Some(entry) = func.entry {
        cold[entry.index()] = false;
    }
    cold
}

pub(crate) fn has_regular_call(func: &Function, block: Block) -> bool {
    func.blocks[block.index()]
        .insts
        .iter()
        .any(|&inst| match &func.insts[inst.index()].op {
            Op::Call { callee, .. } | Op::TryCall { callee, .. } => !func.is_thread_cell(*callee),
            _ => false,
        })
}

fn has_caller_loc_call(func: &Function, zeros: &[bool], block: Block) -> bool {
    !zeros.is_empty()
        && func.blocks[block.index()].insts.iter().any(|&inst| {
            func.caller_loc_result(&func.insts[inst.index()].op, |v| zeros[v.index()])
                .is_some()
        })
}

fn has_atomic_rmw(func: &Function, block: Block) -> bool {
    func.blocks[block.index()].insts.iter().any(|&inst| {
        matches!(
            func.insts[inst.index()].op,
            Op::AtomicCas(..) | Op::AtomicRmw(..)
        )
    })
}

fn arm_has_atomic_call_detour(func: &Function, cfg: &Cfg, arm: Block, stop: Block) -> bool {
    if !has_atomic_rmw(func, arm) {
        return false;
    }
    let mut visited = [arm; 16];
    let mut len = 1;
    let mut stack = vec![arm];
    while let Some(cur) = stack.pop() {
        for s in func.successors(cur) {
            if s == stop || !cfg.dominates(arm, s) {
                continue;
            }
            if has_regular_call(func, s) {
                return true;
            }
            if len < visited.len() && !visited[..len].contains(&s) {
                visited[len] = s;
                len += 1;
                stack.push(s);
            }
        }
    }
    false
}

fn arm_inevitably_calls(
    func: &Function,
    cfg: &Cfg,
    cold: &[bool],
    (root, cur, stop, depth): (Block, Block, Option<Block>, u8),
) -> bool {
    if has_regular_call(func, cur) {
        return true;
    }
    if depth >= 3 {
        return false;
    }
    match func.last(cur) {
        Some(&Op::Jump(t, _)) if Some(t) != stop && t != cur && cfg.dominates(root, t) => {
            cold[t.index()] || arm_inevitably_calls(func, cfg, cold, (root, t, stop, depth + 1))
        }
        Some(&Op::Brif(_, t, _, e, _)) if t != e => [t, e].into_iter().all(|s| {
            Some(s) != stop
                && s != cur
                && cfg.dominates(root, s)
                && (cold[s.index()]
                    || arm_inevitably_calls(func, cfg, cold, (root, s, stop, depth + 1)))
        }),
        _ => false,
    }
}

struct DetourCtx<'a> {
    cfg: &'a Cfg,
    loops: &'a Loops,
    cold: &'a [bool],
    is_latch: &'a [bool],
    is_head: &'a [bool],
    zeros: &'a [bool],
}

fn is_call_detour_arm(
    func: &Function,
    ctx: &DetourCtx<'_>,
    parent: Block,
    arm: Block,
    other: Block,
) -> bool {
    let direct_rejoin = ctx
        .cfg
        .preds_of(other)
        .iter()
        .any(|&q| ctx.cfg.dominates(arm, q));
    let jump_rejoin = !direct_rejoin
        && matches!(
            func.last(other),
            Some(&Op::Jump(m, _)) if m != other && ctx.cfg.preds_of(m).iter().any(|&q| ctx.cfg.dominates(arm, q))
        );
    let fast_ret = !direct_rejoin
        && !jump_rejoin
        && matches!(func.last(other), Some(Op::Return(_)))
        && !has_regular_call(func, other)
        && !has_caller_loc_call(func, ctx.zeros, other);
    let rejoins = direct_rejoin || jump_rejoin || fast_ret;
    let stop = if direct_rejoin {
        Some(other)
    } else if let Some(&Op::Jump(m, _)) = func.last(other).filter(|_| jump_rejoin) {
        Some(m)
    } else {
        None
    };
    let arm_calls = arm_inevitably_calls(func, ctx.cfg, ctx.cold, (arm, arm, stop, 0))
        || (direct_rejoin && arm_has_atomic_call_detour(func, ctx.cfg, arm, other));
    let colder = (arm_calls && (direct_rejoin || !has_regular_call(func, other)))
        || (has_caller_loc_call(func, ctx.zeros, arm)
            && (direct_rejoin || !has_caller_loc_call(func, ctx.zeros, other)));
    !(Some(arm) == func.entry
        || ctx.cold[arm.index()]
        || ctx.cold[other.index()]
        || ctx.is_head[arm.index()]
        || ctx.is_latch[arm.index()]
        || ctx.cfg.single_pred(arm) != Some(parent)
        || ctx.loops.depth_of(other) < ctx.loops.depth_of(arm)
        || !colder
        || !rejoins
        || ctx.loops.list.iter().any(|l| {
            l.body[parent.index()] && l.latches.iter().any(|&lt| ctx.cfg.dominates(arm, lt))
        }))
}

fn call_detours(
    func: &Function,
    cfg: &Cfg,
    loops: &Loops,
    cold: &[bool],
    is_latch: &[bool],
) -> Vec<bool> {
    let n = func.blocks.len();
    let zeros = func.thread_cell_zeros();
    let mut deferred = vec![false; n];
    let mut is_head = vec![false; n];
    loops
        .list
        .iter()
        .for_each(|each| is_head[each.head.index()] = true);
    let ctx = DetourCtx {
        cfg,
        loops,
        cold,
        is_latch,
        is_head: &is_head,
        zeros: &zeros,
    };
    let mut is_root = vec![false; n];
    let mut any = false;
    for b in 0..n {
        let p = Block(b as u32);
        let Some(&Op::Brif(_, then, _, otherwise, _)) = func.last(p) else {
            continue;
        };
        if then == otherwise {
            continue;
        }
        for (arm, other) in [(then, otherwise), (otherwise, then)] {
            if is_call_detour_arm(func, &ctx, p, arm, other) {
                is_root[arm.index()] = true;
                any = true;
            }
        }
    }
    if any {
        for b in 0..n {
            let mut cur = Some(Block(b as u32));
            while let Some(c) = cur {
                if is_root[c.index()] {
                    deferred[b] = true;
                    break;
                }
                cur = cfg.idom[c.index()].filter(|&d| d != c);
            }
        }
    }
    deferred
}

/// Returns `(cold, deferred)` blocks for `func` using precomputed `cfg` and `loops`.
pub(crate) fn cold_and_detours(
    func: &Function,
    cfg: &Cfg,
    loops: &Loops,
) -> (Vec<bool>, Vec<bool>) {
    let cold = cold_with(func, cfg);
    let mut is_latch = vec![false; func.blocks.len()];
    loops
        .list
        .iter()
        .flat_map(|each| &each.latches)
        .for_each(|latch| is_latch[latch.index()] = true);
    let deferred = call_detours(func, cfg, loops, &cold, &is_latch);
    (cold, deferred)
}

/// Fast reverse-postorder block layout for `Effort::Quick`, skipping CFG dominators, loops, and cold-block partitioning.
pub(crate) fn quick_order(func: &Function) -> Vec<Block> {
    let Some(entry) = func.entry else {
        return Vec::new();
    };
    let n = func.blocks.len();
    let mut visited = vec![false; n];
    let mut post = Vec::with_capacity(n);
    let mut stack = vec![(entry, false)];
    while let Some((block, done)) = stack.pop() {
        if done {
            post.push(block);
            continue;
        }
        if std::mem::replace(&mut visited[block.index()], true) {
            continue;
        }
        stack.push((block, true));
        for succ in func.successors(block) {
            if !visited[succ.index()] {
                stack.push((succ, false));
            }
        }
    }
    post.reverse();
    post.retain(|block| *block == entry || !func.blocks[block.index()].insts.is_empty());
    post.extend(
        (0..n as u32)
            .map(Block)
            .filter(|b| !visited[b.index()] && !func.blocks[b.index()].insts.is_empty()),
    );
    post
}

pub(crate) fn order(func: &Function) -> Vec<Block> {
    let Some(entry) = func.entry else {
        return Vec::new();
    };
    let cfg = func.cfg();
    let loops = cfg.loops();
    let cold = cold_with(func, &cfg);
    let mut is_latch = vec![false; func.blocks.len()];
    loops
        .list
        .iter()
        .flat_map(|each| &each.latches)
        .for_each(|latch| is_latch[latch.index()] = true);
    let deferred = call_detours(func, &cfg, loops, &cold, &is_latch);
    let is_cold = |block: Block| cold[block.index()] || deferred[block.index()];
    let mut visited = vec![false; func.blocks.len()];
    let mut post = Vec::with_capacity(func.blocks.len());
    let mut stack = vec![(entry, false)];
    while let Some((block, done)) = stack.pop() {
        if done {
            post.push(block);
            continue;
        }
        if std::mem::replace(&mut visited[block.index()], true) {
            continue;
        }
        stack.push((block, true));
        let mut succs: Vec<Block> = func
            .successors(block)
            .filter(|succ| !visited[succ.index()])
            .collect();
        succs.sort_by_key(|&succ| {
            std::cmp::Reverse((!is_cold(succ), loops.depth_of(succ), is_latch[succ.index()]))
        });
        for succ in succs {
            stack.push((succ, false));
        }
    }
    post.reverse();
    let mut order = post;
    let mut placed = vec![false; func.blocks.len()];
    order.iter().for_each(|block| placed[block.index()] = true);
    order.retain(|block| *block == entry || !func.blocks[block.index()].insts.is_empty());
    order.extend(
        (0..func.blocks.len() as u32)
            .map(Block)
            .filter(|block| !placed[block.index()] && !func.blocks[block.index()].insts.is_empty()),
    );
    let mut hot = Vec::with_capacity(order.len());
    let mut cold = Vec::new();
    for block in order {
        if is_cold(block) {
            cold.push(block);
        } else {
            hot.push(block);
        }
    }
    hot.extend(cold);
    hot
}

const TEST_LIMIT: usize = 4;

pub(crate) fn copied_tests(func: &Function, order: &[Block]) -> Vec<bool> {
    let rank = Cfg::ranks(order, func.blocks.len());
    let mut copied = vec![false; func.insts.len()];
    for (at, &latch) in order.iter().enumerate() {
        let Some(&last) = func.blocks[latch.index()].insts.last() else {
            continue;
        };
        let Op::Jump(head, _) = func.insts[last.index()].op else {
            continue;
        };
        let insts = &func.blocks[head.index()].insts;
        let calls = insts
            .iter()
            .any(|inst| matches!(func.insts[inst.index()].op, Op::Call { .. }));
        copied[last.index()] = rank[head.index()] < at
            && insts.len() <= TEST_LIMIT
            && !calls
            && matches!(func.last(head), Some(Op::Brif(..)));
    }
    copied
}
