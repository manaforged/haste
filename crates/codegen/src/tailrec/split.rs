use std::cmp::Reverse;

use rustc_hash::FxHashMap;

use super::FOLD_LIMIT;
use crate::cfg::{Cfg, Loop, Loops};
use crate::ir::{Binary, Block, Convert, Domain, Function, Inst, Op, Type, Value};
use crate::live::{BitMatrix, results, set_bits, solve_backward_liveness};

const MAX_SPLIT: usize = 4;

fn reg_budgets(arch: crate::target::Arch) -> (usize, usize) {
    match arch {
        crate::target::Arch::X64 => (5, 9),
        crate::target::Arch::Arm64 => (usize::MAX, 0),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StaticVal {
    Dynamic,
    Imm(i64),
    Frame,
}

fn is_call_site(func: &Function, insts: &[Inst], idx: usize) -> bool {
    match func.insts[insts[idx].index()].op {
        Op::TryCall { .. } => true,
        Op::Call { .. } => !insts
            .get(idx + 1)
            .is_some_and(|&n| matches!(func.insts[n.index()].op, Op::Trap)),
        _ => false,
    }
}

fn static_step(func: &Function, op: &Op, ty: Type, vals: &[StaticVal]) -> StaticVal {
    match *op {
        Op::Const(imm) if ty.is_small_int() => StaticVal::Imm(imm),
        Op::StackAddr(..) | Op::FramePointer if ty == Type::I64 => StaticVal::Frame,
        Op::Icmp(cc, x, y) => match (vals[x.index()], vals[y.index()]) {
            (StaticVal::Imm(a), StaticVal::Imm(b)) if func.ty(x).is_small_int() => {
                StaticVal::Imm(i64::from(crate::fold::holds(cc, func.ty(x), a, b)))
            }
            _ => StaticVal::Dynamic,
        },
        Op::Binary(bin, x, y) => match (vals[x.index()], vals[y.index()]) {
            (StaticVal::Frame, StaticVal::Imm(imm)) | (StaticVal::Imm(imm), StaticVal::Frame)
                if bin == Binary::Iadd
                    && ty == Type::I64
                    && (-FOLD_LIMIT..=FOLD_LIMIT).contains(&imm) =>
            {
                StaticVal::Frame
            }
            (StaticVal::Imm(a), StaticVal::Imm(b)) if func.ty(x).is_small_int() => {
                crate::fold::constant(bin, a, b).map_or(StaticVal::Dynamic, StaticVal::Imm)
            }
            _ => StaticVal::Dynamic,
        },
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => match vals[x.index()] {
            StaticVal::Imm(imm) if func.ty(x).is_small_int() => {
                StaticVal::Imm(crate::fold::extend(imm, func.ty(x), false))
            }
            _ => StaticVal::Dynamic,
        },
        _ => StaticVal::Dynamic,
    }
}

fn static_values(func: &Function) -> Vec<bool> {
    let mut vals = vec![StaticVal::Dynamic; func.types.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for data in &func.insts {
            let Some(res) = data.result else {
                continue;
            };
            if vals[res.index()] != StaticVal::Dynamic {
                continue;
            }
            let next = static_step(func, &data.op, func.ty(res), &vals);
            if next != StaticVal::Dynamic {
                vals[res.index()] = next;
                changed = true;
            }
        }
    }
    vals.into_iter().map(|v| v != StaticVal::Dynamic).collect()
}

fn fusable_cmp_flags(func: &Function, uses: &[u32]) -> Vec<bool> {
    let mut out = vec![false; func.types.len()];
    for block in &func.blocks {
        let Some(&last) = block.insts.last() else {
            continue;
        };
        let (Op::Brif(cond, ..) | Op::Trapz(cond)) = func.insts[last.index()].op else {
            continue;
        };
        if uses[cond.index()] == 1
            && let Some(&prev) = block.insts.iter().rev().nth(1)
            && func.insts[prev.index()].result == Some(cond)
            && matches!(
                func.insts[prev.index()].op,
                Op::Icmp(_, a, _) | Op::Fits(_, _, a, _) if !func.ty(a).is_vector()
            )
        {
            out[cond.index()] = true;
        }
    }
    out
}

fn unique_head_preds(cfg: &Cfg, head: Block) -> Vec<Block> {
    let mut preds = Vec::new();
    for &p in cfg.preds_of(head) {
        if !preds.contains(&p) {
            preds.push(p);
        }
    }
    preds
}

fn eligible_loop(func: &Function, cfg: &Cfg, loops: &Loops, each: &Loop) -> Option<Vec<Block>> {
    if Some(each.head) == func.entry || !cfg.reachable[each.head.index()] {
        return None;
    }
    if loops
        .list
        .iter()
        .any(|o| o.head != each.head && each.body[o.head.index()])
    {
        return None;
    }
    let body_ok = (0..func.blocks.len()).filter(|&b| each.body[b]).all(|b| {
        cfg.reachable[b]
            && cfg.dominates(each.head, Block(b as u32))
            && !func.blocks[b].insts.iter().any(|&i| {
                matches!(
                    func.insts[i.index()].op,
                    Op::Call { .. } | Op::TryCall { .. }
                )
            })
    });
    if !body_ok {
        return None;
    }
    let preds = unique_head_preds(cfg, each.head);
    let ok_preds = preds.iter().all(|&p| {
        cfg.reachable[p.index()] && matches!(func.last(p), Some(Op::Jump(..) | Op::Brif(..)))
    });
    let (has_out, has_in) = (
        preds.iter().any(|&p| !each.body[p.index()]),
        preds.iter().any(|&p| each.body[p.index()]),
    );
    (ok_preds && has_out && has_in).then_some(preds)
}

fn for_loop_operands(op: &Op, body: &[bool], mut f: impl FnMut(Value)) {
    match op {
        Op::Jump(target, args) => {
            if body[target.index()] {
                args.iter().for_each(|&v| f(v));
            }
        }
        Op::Brif(cond, t, ta, fb, fa) => {
            f(*cond);
            if body[t.index()] {
                ta.iter().for_each(|&v| f(v));
            }
            if body[fb.index()] {
                fa.iter().for_each(|&v| f(v));
            }
        }
        _ => op.for_operands(f),
    }
}

fn local_live_sets(
    func: &Function,
    dense: &[Option<usize>],
    count: usize,
) -> (BitMatrix, BitMatrix) {
    let blocks = func.blocks.len();
    let (mut uses, mut defs) = (BitMatrix::new(blocks, count), BitMatrix::new(blocks, count));
    for (b, block) in func.blocks.iter().enumerate() {
        block.params.iter().for_each(|&p| {
            if let Some(d) = dense[p.index()] {
                defs.set(b, d);
            }
        });
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| {
                if let Some(d) = dense[v.index()]
                    && !defs.get(b, d)
                {
                    uses.set(b, d);
                }
            });
            results(data).for_each(|r| {
                if let Some(d) = dense[r.index()] {
                    defs.set(b, d);
                }
            });
        }
    }
    (uses, defs)
}

fn compute_live_in(func: &Function, cfg: &Cfg, dense: &[Option<usize>], count: usize) -> BitMatrix {
    if count == 0 {
        return BitMatrix::new(func.blocks.len(), 0);
    }
    let (uses, defs) = local_live_sets(func, dense, count);
    solve_backward_liveness(func, cfg.rpo.iter().copied(), &uses, &defs)
}

fn find_crosses_call(
    func: &Function,
    cfg: &Cfg,
    candidates: &[usize],
) -> (Vec<bool>, BitMatrix, Vec<Option<usize>>) {
    let mut dense = vec![None; func.types.len()];
    candidates
        .iter()
        .enumerate()
        .for_each(|(i, &v)| dense[v] = Some(i));
    let live_in = compute_live_in(func, cfg, &dense, candidates.len());
    let mut crosses = vec![false; func.types.len()];
    let words = live_in.words();
    if words == 0 {
        return (crosses, live_in, dense);
    }
    let mut cur = vec![0_u64; words];
    for &block in &cfg.rpo {
        live_in.fill_succ_union(func, block, &mut cur);
        let insts = &func.blocks[block.index()].insts;
        for idx in (0..insts.len()).rev() {
            let data = &func.insts[insts[idx].index()];
            results(data).for_each(|r| {
                if let Some(d) = dense[r.index()] {
                    cur[d >> 6] &= !(1_u64 << (d & 63));
                }
            });
            if is_call_site(func, insts, idx) {
                for i in set_bits(&cur) {
                    crosses[candidates[i]] = true;
                }
            }
            data.op.for_operands(|v| {
                if let Some(d) = dense[v.index()] {
                    cur[d >> 6] |= 1_u64 << (d & 63);
                }
            });
        }
    }
    (crosses, live_in, dense)
}

fn collect_loop_uses(
    func: &Function,
    cfg: &Cfg,
    each: &Loop,
    def_blocks: &[Option<Block>],
    static_vals: &[bool],
) -> Vec<(usize, u32)> {
    let mut uses_in_l = FxHashMap::default();
    for (b, block) in func.blocks.iter().enumerate() {
        if !each.body[b] {
            continue;
        }
        for &inst in &block.insts {
            for_loop_operands(&func.insts[inst.index()].op, &each.body, |v| {
                let idx = v.index();
                if func.ty(v).is_small_int()
                    && !static_vals[idx]
                    && let Some(db) = def_blocks[idx]
                    && cfg.reachable[db.index()]
                    && !each.body[db.index()]
                    && cfg.dominates(db, each.head)
                {
                    *uses_in_l.entry(idx).or_insert(0_u32) += 1;
                }
            });
        }
    }
    uses_in_l.into_iter().collect()
}

fn block_peak_gprs(
    func: &Function,
    live_in: &BitMatrix,
    dense: &[Option<usize>],
    block: Block,
    cur: &mut [u64],
) -> usize {
    live_in.fill_succ_union(func, block, cur);
    let mut peak: usize = cur.iter().map(|w| w.count_ones() as usize).sum();
    for &inst in func.blocks[block.index()].insts.iter().rev() {
        let data = &func.insts[inst.index()];
        results(data).for_each(|r| {
            if let Some(d) = dense[r.index()] {
                cur[d >> 6] &= !(1_u64 << (d & 63));
            }
        });
        data.op.for_operands(|v| {
            if let Some(d) = dense[v.index()] {
                cur[d >> 6] |= 1_u64 << (d & 63);
            }
        });
        peak = peak.max(cur.iter().map(|w| w.count_ones() as usize).sum());
    }
    peak
}

fn loop_dense_gprs(
    func: &Function,
    each: &Loop,
    (def_blocks, excluded, crosses): (&[Option<Block>], &[bool], &[bool]),
) -> (Vec<Option<usize>>, usize) {
    let (mut dense, mut count) = (vec![None; func.types.len()], 0_usize);
    let mut add = |v: usize| {
        if func.types[v].is_small_int() && !excluded[v] && dense[v].is_none() {
            dense[v] = Some(count);
            count += 1;
        }
    };
    for block in func
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(b, blk)| each.body[b].then_some(blk))
    {
        block.params.iter().for_each(|p| add(p.index()));
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            results(data).for_each(|r| add(r.index()));
            data.op.for_operands(|v| {
                if let Some(db) = def_blocks[v.index()]
                    && !each.body[db.index()]
                    && !crosses[v.index()]
                {
                    add(v.index());
                }
            });
        }
    }
    (dense, count)
}

fn loop_peak_gprs(
    func: &Function,
    cfg: &Cfg,
    each: &Loop,
    loop_budget: usize,
    ctx: (&[Option<Block>], &[bool], &[bool]),
) -> usize {
    let excluded = ctx.1;
    let head_params = func.blocks[each.head.index()]
        .params
        .iter()
        .filter(|&&p| func.ty(p).is_small_int() && !excluded[p.index()])
        .count();
    if head_params >= loop_budget {
        return head_params;
    }
    let (dense, count) = loop_dense_gprs(func, each, ctx);
    let live_in = compute_live_in(func, cfg, &dense, count);
    let words = live_in.words();
    if words == 0 {
        return head_params;
    }
    let (mut cur, mut peak) = (vec![0_u64; words], head_params);
    for (b, &inside) in each.body.iter().enumerate() {
        if inside {
            peak = peak.max(block_peak_gprs(
                func,
                &live_in,
                &dense,
                Block(b as u32),
                &mut cur,
            ));
        }
    }
    peak
}

fn apply_loop_split(func: &mut Function, each: &Loop, preds: &[Block], chosen: &[usize]) {
    let (mut remap, mut pairs) = (FxHashMap::default(), Vec::with_capacity(chosen.len()));
    for &v_idx in chosen {
        let v = Value(v_idx as u32);
        let v_loop = func.value(func.ty(v));
        func.blocks[each.head.index()].params.push(v_loop);
        remap.insert(v, v_loop);
        pairs.push((v, v_loop));
    }
    for block in func
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(b, blk)| each.body[b].then_some(blk))
    {
        block.insts.iter().for_each(|&i| {
            func.insts[i.index()]
                .op
                .map_operands(|x| remap.get(&x).copied().unwrap_or(x));
        });
    }
    for &pred in preds {
        let is_back = each.body[pred.index()];
        func.for_edge_args_mut(pred, each.head, |args| {
            args.extend(
                pairs
                    .iter()
                    .map(|&(v, v_loop)| if is_back { v_loop } else { v }),
            );
        });
    }
}

fn plan_loop_split(
    (func, cfg): (&Function, &Cfg),
    (each, preds, mut cands): (Loop, Vec<Block>, Vec<(usize, u32)>),
    (saved_floor, loop_budget): (usize, usize),
    (all_gprs, live_in, dense): (&[usize], &BitMatrix, &[Option<usize>]),
    ctx: (&[Option<Block>], &[bool], &[bool]),
) -> Option<(Loop, Vec<Block>, Vec<usize>)> {
    let crosses = ctx.2;
    let hrow = each.head.index();
    let live_crosses = all_gprs
        .iter()
        .filter(|&&v| crosses[v] && dense[v].is_some_and(|d| live_in.get(hrow, d)))
        .count();
    let overflow = live_crosses.saturating_sub(saved_floor);
    if overflow == 0 {
        return None;
    }
    let out_preds = preds.iter().filter(|&&p| !each.body[p.index()]).count();
    cands.retain(|&(v, c)| crosses[v] && (c >= 2 || (c >= 1 && each.depth >= 2 && out_preds == 1)));
    if cands.is_empty() {
        return None;
    }
    let budget = loop_budget
        .saturating_sub(loop_peak_gprs(func, cfg, &each, loop_budget, ctx))
        .min(MAX_SPLIT)
        .min(overflow);
    if budget == 0 {
        return None;
    }
    cands.sort_unstable_by_key(|&(v, count)| (Reverse(count), v));
    Some((
        each,
        preds,
        cands.into_iter().take(budget).map(|(v, _)| v).collect(),
    ))
}

pub(super) fn split_innermost_loops(func: &mut Function, target: &crate::Target) -> Domain {
    let (saved_floor, loop_budget) = reg_budgets(target.arch);
    if loop_budget == 0 {
        return Domain::NONE;
    }
    let has_call = func
        .blocks
        .iter()
        .any(|b| (0..b.insts.len()).any(|i| is_call_site(func, &b.insts, i)));
    if func.coroutine.is_some() || func.entry.is_none() || func.blocks.len() < 2 || !has_call {
        return Domain::NONE;
    }
    let uses = func.use_counts();
    let (mut static_vals, def_blocks, fusable) = (
        static_values(func),
        func.def_blocks(),
        fusable_cmp_flags(func, &uses),
    );
    if target.arch == crate::Arch::X64 {
        let defs = crate::fold::defs(func);
        for (s, a) in static_vals
            .iter_mut()
            .zip(crate::alloc::atomic_elidable_addrs(func, &defs, &uses))
        {
            *s |= a;
        }
    }
    let excluded: Vec<bool> = static_vals
        .iter()
        .zip(&fusable)
        .map(|(&s, &f)| s || f)
        .collect();
    let plans: Vec<(Loop, Vec<Block>, Vec<usize>)> = {
        let cfg = func.cfg();
        let loops = cfg.loops();
        let eligible: Vec<_> = loops
            .list
            .iter()
            .filter_map(|each| {
                let preds = eligible_loop(func, &cfg, loops, each)?;
                let cands = collect_loop_uses(func, &cfg, each, &def_blocks, &static_vals);
                (!cands.is_empty()).then(|| (each.clone(), preds, cands))
            })
            .collect();
        if eligible.is_empty() {
            return Domain::NONE;
        }
        let all_gprs: Vec<usize> = (0..func.types.len())
            .filter(|&v| func.types[v].is_small_int() && !excluded[v])
            .collect();
        let (crosses, live_in, dense) = find_crosses_call(func, &cfg, &all_gprs);
        eligible
            .into_iter()
            .filter_map(|item| {
                plan_loop_split(
                    (func, &cfg),
                    item,
                    (saved_floor, loop_budget),
                    (&all_gprs, &live_in, &dense),
                    (&def_blocks, &excluded, &crosses),
                )
            })
            .collect()
    };
    if plans.is_empty() {
        return Domain::NONE;
    }
    for (each, preds, chosen) in plans {
        apply_loop_split(func, &each, &preds, &chosen);
    }
    Domain::VALUES
}
