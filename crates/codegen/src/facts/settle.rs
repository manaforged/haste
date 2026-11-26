use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, Inst, InstData, IntCC, Op, Type, Value};

use super::exits;
use super::live;
use super::loops::{Env, ExtraPath};
use super::ranges::{Ranges, collect_incoming};
use super::rel::{Path, RelKind, cond_truth, fits_truth};
use super::transfer;
use super::{Fact, MAX_RELS, decide, truth};

#[derive(Clone, Copy)]
pub(super) struct SelectRec {
    res: Value,
    cond: Value,
    t_val: Value,
    f_val: Value,
    pub(super) block: Block,
}

#[derive(Clone, Default)]
pub(super) struct DomState {
    aliases: Vec<(Value, Value)>,
}

impl DomState {
    fn resolve(&self, mut v: Value) -> Value {
        for _ in 0..4 {
            let Some(&(_, to)) = self.aliases.iter().rev().find(|&&(from, _)| from == v) else {
                break;
            };
            v = to;
        }
        v
    }

    fn record_alias(&mut self, from: Value, to: Value) -> bool {
        let to = self.resolve(to);
        if to == from || self.resolve(from) == to {
            return false;
        }
        if self.aliases.len() >= MAX_RELS {
            self.aliases.remove(0);
        }
        self.aliases.push((from, to));
        true
    }

    fn propagate_selects(
        &mut self,
        func: &Function,
        (cfg, ranges, env, path, extra): (&Cfg, &Ranges, &Env, &mut Path, &ExtraPath),
        selects: &[SelectRec],
        block: Block,
        seen: &[bool],
    ) {
        for _ in 0..3 {
            let mut progress = false;
            for &rec in selects {
                if !seen[rec.res.index()] || self.resolve(rec.res) != rec.res {
                    continue;
                }
                let (cond, t_val, f_val) = (
                    self.resolve(rec.cond),
                    self.resolve(rec.t_val),
                    self.resolve(rec.f_val),
                );
                if let Some(taken) = cond_truth(func, ranges, path, cond)
                    .or_else(|| extra.cond_truth(func, (cfg, ranges, env, path, block), cond, 0))
                {
                    progress |= self.record_alias(rec.res, if taken { t_val } else { f_val });
                    continue;
                }
                let opt_fact = path.fact(ranges, rec.res);
                let (t_exact, f_exact) = (
                    path.fact(ranges, t_val).exact(),
                    path.fact(ranges, f_val).exact(),
                );
                if f_exact.is_some_and(|k| !opt_fact.admits(k)) {
                    progress |= self.record_alias(rec.res, t_val);
                    path.assume(func, ranges, cond, true);
                    if f_exact == Some(0) {
                        path.assume(func, ranges, t_val, true);
                    }
                } else if t_exact.is_some_and(|k| !opt_fact.admits(k)) {
                    progress |= self.record_alias(rec.res, f_val);
                    path.assume(func, ranges, cond, false);
                    if t_exact == Some(0) {
                        path.assume(func, ranges, f_val, true);
                    }
                } else if let Some(k) = opt_fact.exact() {
                    if f_exact == Some(k) && !path.fact(ranges, t_val).admits(k) {
                        progress |= self.record_alias(rec.res, f_val);
                        path.assume(func, ranges, cond, false);
                    } else if t_exact == Some(k) && !path.fact(ranges, f_val).admits(k) {
                        progress |= self.record_alias(rec.res, t_val);
                        path.assume(func, ranges, cond, true);
                    }
                }
            }
            if !progress {
                break;
            }
        }
    }
}

fn collect_selects(func: &Function) -> Option<Vec<SelectRec>> {
    let (mut selects, mut open) = (Vec::new(), false);
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            open |= matches!(
                data.op,
                Op::Icmp(..) | Op::Fits(..) | Op::Trapz(_) | Op::Assume(_) | Op::Brif(..)
            );
            if let (Op::Select(cond, t_val, f_val), Some(res)) = (&data.op, data.result)
                && func.ty(res).is_small_int()
            {
                selects.push(SelectRec {
                    res,
                    cond: *cond,
                    t_val: *t_val,
                    f_val: *f_val,
                    block: Block(b_idx as u32),
                });
            }
        }
    }
    (open || !selects.is_empty()).then_some(selects)
}

fn join_incoming_rels(
    func: &Function,
    (cfg, ranges, incoming, complete, exits, def_blocks): (
        &Cfg,
        &Ranges,
        &[Vec<(Block, &[Value])>],
        &[bool],
        &[Option<Path>],
        &std::cell::OnceCell<Vec<Option<Block>>>,
    ),
    block: Block,
    path: &mut Path,
) {
    if incoming.is_empty()
        || !complete[block.index()]
        || func.blocks[block.index()].params.is_empty()
    {
        return;
    }
    let edges = &incoming[block.index()];
    if edges.len() < 2 || edges.len() > 16 {
        return;
    }
    if !edges
        .iter()
        .all(|&(pred, _)| exits.get(pred.index()).and_then(Option::as_ref).is_some())
    {
        return;
    }
    if !edges.iter().any(|&(pred, _)| {
        exits[pred.index()].as_ref().is_some_and(|p| !p.rels().is_empty())
            || matches!(func.last(pred), Some(&Op::Brif(_, then, _, otherwise, _)) if then != otherwise)
    }) {
        return;
    }
    let edge_paths: Vec<Path> = edges
        .iter()
        .map(|&(pred, _)| {
            let mut p = exits[pred.index()].clone().unwrap_or_default();
            if matches!(func.last(pred), Some(&Op::Brif(_, then, _, otherwise, _)) if then != otherwise)
            {
                p.follow(func, ranges, pred, block);
            }
            p
        })
        .collect();
    let db_map = def_blocks.get_or_init(|| func.def_blocks());
    let mut cands: Vec<(Value, Type)> = Vec::new();
    for ep in &edge_paths {
        for r in ep.rels().iter().rev() {
            if matches!(r.kind, RelKind::Ult | RelKind::Ule)
                && db_map
                    .get(r.rhs.index())
                    .copied()
                    .flatten()
                    .is_some_and(|db| db != block && cfg.dominates(db, block))
                && !cands.contains(&(r.rhs, r.ty))
            {
                cands.push((r.rhs, r.ty));
                if cands.len() >= 16 {
                    break;
                }
            }
        }
        if cands.len() >= 16 {
            break;
        }
    }
    for (k, &param) in func.blocks[block.index()].params.iter().enumerate() {
        let ty = func.ty(param);
        if !ty.is_small_int() {
            continue;
        }
        for &(bound, bty) in &cands {
            if bty != ty {
                continue;
            }
            for (kind, cc) in [
                (RelKind::Ult, IntCC::UnsignedLessThan),
                (RelKind::Ule, IntCC::UnsignedLessThanOrEqual),
            ] {
                let all_ok = edge_paths.iter().zip(edges).all(|(ep, &(_, args))| {
                    args.get(k).is_some_and(|&arg| {
                        ep.decide_rel(func, ranges, cc, ty, arg, bound) == Some(true)
                            || decide(cc, ty, &ep.fact(ranges, arg), &ep.fact(ranges, bound))
                                == Some(true)
                    })
                });
                if all_ok {
                    path.push_rel_kind(kind, param, bound, ty);
                    break;
                }
            }
        }
    }
}

/// Settles the compares, traps and branches that the facts on entry to each block and its earlier instructions decide: a settled compare's result becomes a constant in `consts`, a trap that cannot fire goes, and a settled branch becomes a jump. A block's facts on entry are its only predecessor's facts on exit plus that edge's branch. Returns whether anything changed.
pub(crate) fn settle(
    func: &mut Function,
    defs: &[Option<Inst>],
    consts: &mut [Option<i64>],
) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let Some(entry) = func.entry else {
        return false;
    };
    let Some(selects) = collect_selects(func) else {
        return false;
    };
    let cfg = func.cfg();
    let single: Vec<Option<Block>> = (0..func.blocks.len())
        .map(|b| cfg.single_pred(Block(b as u32)))
        .collect();
    let (mut decided, mut compares, mut rewrites) = (Vec::new(), Vec::new(), Vec::new());
    {
        let (incoming, complete, edge_args) = collect_incoming(func);
        let ranges = Ranges::over(
            func,
            (defs, consts),
            (&cfg.rpo, &single, &cfg.idom),
            (&incoming, &complete, &edge_args),
        );
        let def_blocks = std::cell::OnceCell::new();
        let mut live = live::Live::default();
        let loops = cfg.loops();
        let mut env = Env::default();
        let mut exits = exits::Exits::new(func, &cfg, (&single, entry), (loops, &incoming));
        let mut seen = if selects.is_empty() {
            Vec::new()
        } else {
            vec![false; func.types.len()]
        };
        for &block in &cfg.rpo {
            if !seen.is_empty() {
                for &param in &func.blocks[block.index()].params {
                    seen[param.index()] = true;
                }
            }
            live.enter(&cfg, &selects, block);
            for each in &loops.list {
                if each.head == block {
                    env.collect_loop(
                        func,
                        (
                            &cfg,
                            &ranges,
                            &incoming,
                            &complete,
                            exits.paths(),
                            &def_blocks,
                        ),
                        each,
                    );
                }
            }
            let (follow, mut path, mut extra, mut dom) = exits.start(&cfg, block, entry);
            if let Some(pred) = follow {
                if single[block.index()] == Some(pred)
                    && let Some(&[(_, args)]) = incoming.get(block.index()).map(Vec::as_slice)
                {
                    for (&param, &arg) in func.blocks[block.index()].params.iter().zip(args) {
                        if func.ty(param).is_small_int() && param != arg {
                            path.push_rel_kind(RelKind::Eq, param, arg, func.ty(param));
                        }
                    }
                }
                if matches!(func.last(pred), Some(&Op::Brif(_, then, _, otherwise, _)) if then != otherwise)
                {
                    path.follow(func, &ranges, pred, block);
                    extra.follow(func, (&cfg, &ranges, &env, &path), pred, block);
                    dom.propagate_selects(
                        func,
                        (&cfg, &ranges, &env, &mut path, &extra),
                        &live.recs,
                        block,
                        &seen,
                    );
                }
            } else {
                join_incoming_rels(
                    func,
                    (
                        &cfg,
                        &ranges,
                        &incoming,
                        &complete,
                        exits.paths(),
                        &def_blocks,
                    ),
                    block,
                    &mut path,
                );
            }
            for inv in &env.rel_invs {
                if inv.head == block {
                    path.push_rel_kind(inv.kind, inv.param, inv.bound, inv.ty);
                }
            }
            for &inst in &func.blocks[block.index()].insts {
                settle_one(
                    func,
                    (
                        &cfg, &ranges, &env, &mut path, &mut extra, &mut dom, &live.recs, block,
                        &seen,
                    ),
                    inst,
                    (&mut decided, &mut compares, &mut rewrites),
                );
                if !seen.is_empty() {
                    let data = &func.insts[inst.index()];
                    if let Some(res) = data.result {
                        seen[res.index()] = true;
                    }
                    if let Op::Call { results, .. } = &data.op {
                        for &res in results {
                            seen[res.index()] = true;
                        }
                    }
                }
            }
            exits.store(block, (path, extra, dom));
        }
    }
    let settled = !compares.is_empty();
    for (result, known) in compares {
        consts[result.index()] = Some(i64::from(known));
    }
    apply(func, &decided, &rewrites) | settled
}

fn sub_range(
    func: &Function,
    (cfg, ranges, env, path, extra, block): (&Cfg, &Ranges, &Env, &Path, &ExtraPath, Block),
    ty: Type,
    (hi_v, lo_v): (Value, Value),
) -> Option<Fact> {
    let proves = |cc| {
        path.decide_rel(func, ranges, cc, ty, lo_v, hi_v) == Some(true)
            || extra.decide(func, (cfg, ranges, env, path, block), cc, lo_v, hi_v) == Some(true)
    };
    let (f_hi, f_lo) = (
        path.refined_fact(func, ranges, hi_v, 0),
        path.refined_fact(func, ranges, lo_v, 0),
    );
    if proves(IntCC::UnsignedLessThan) {
        let max_diff = if f_hi.hi > f_lo.lo {
            f_hi.hi - f_lo.lo
        } else {
            ty.unsigned_max()
        };
        return Some(Fact::range(1, max_diff));
    }
    if proves(IntCC::UnsignedLessThanOrEqual)
        && f_hi.hi >= f_lo.lo
        && f_hi.hi - f_lo.lo < ty.unsigned_max()
    {
        return Some(Fact::range(0, f_hi.hi - f_lo.lo));
    }
    None
}

fn sub_fact(
    func: &Function,
    ctx: (&Cfg, &Ranges, &Env, &Path, &ExtraPath, Block),
    v: Value,
) -> Fact {
    let (_, ranges, _, path, _, _) = ctx;
    let base = path.refined_fact(func, ranges, v, 0);
    if base.lo >= 1 || !func.ty(v).is_small_int() {
        return base;
    }
    match ranges.op(func, v) {
        Some(&Op::Binary(Binary::Isub, hi_v, lo_v)) => {
            sub_range(func, ctx, func.ty(v), (hi_v, lo_v)).map_or(base.clone(), |f| base.meet(&f))
        }
        _ => base,
    }
}

fn sub_cond_truth(
    func: &Function,
    ctx: (&Cfg, &Ranges, &Env, &Path, &ExtraPath, Block),
    cond: Value,
    depth: u8,
) -> Option<bool> {
    if depth > 2 {
        return None;
    }
    let (_, ranges, _, path, _, _) = ctx;
    match ranges.op(func, cond)? {
        &Op::Icmp(cc, x, y) if func.ty(x).is_small_int() => decide(
            cc,
            func.ty(x),
            &sub_fact(func, ctx, x),
            &sub_fact(func, ctx, y),
        ),
        &Op::Binary(Binary::Isub, ..) => truth(&sub_fact(func, ctx, cond)),
        &Op::Binary(Binary::Bxor, a, b) => {
            [(a, b), (b, a)].into_iter().find_map(|(inner, mask)| {
                (path.fact(ranges, mask).exact() == Some(1) && path.fact(ranges, inner).hi <= 1)
                    .then(|| sub_cond_truth(func, ctx, inner, depth + 1).map(|t| !t))
                    .flatten()
            })
        }
        _ => None,
    }
}

fn settle_one(
    func: &Function,
    (cfg, ranges, env, path, extra, dom, selects, block, seen): (
        &Cfg,
        &Ranges,
        &Env,
        &mut Path,
        &mut ExtraPath,
        &mut DomState,
        &[SelectRec],
        Block,
        &[bool],
    ),
    inst: Inst,
    out: (
        &mut Vec<(Inst, Option<bool>)>,
        &mut Vec<(Value, bool)>,
        &mut Vec<(Inst, Value, Value)>,
    ),
) {
    let data = &func.insts[inst.index()];
    if !dom.aliases.is_empty() {
        data.op.for_operands(|v| {
            let mapped = dom.resolve(v);
            if mapped != v {
                out.2.push((inst, v, mapped));
            }
        });
    }
    if !settle_value(
        func,
        (cfg, ranges, env, path, extra, dom, block),
        data,
        out.1,
    ) {
        settle_ctrl(
            func,
            (cfg, ranges, env, path, extra, dom, selects, block, seen),
            inst,
            data,
            out.0,
        );
    }
}

fn settle_value(
    func: &Function,
    (cfg, ranges, env, path, extra, dom, block): (
        &Cfg,
        &Ranges,
        &Env,
        &Path,
        &ExtraPath,
        &DomState,
        Block,
    ),
    data: &InstData,
    compares: &mut Vec<(Value, bool)>,
) -> bool {
    let known = match (&data.op, data.result) {
        (&Op::Icmp(cc, raw_x, raw_y), Some(result))
            if ranges.constant(result).is_none() && func.ty(raw_x).is_small_int() =>
        {
            let (x, y) = (dom.resolve(raw_x), dom.resolve(raw_y));
            let ty = func.ty(x);
            let ctx = (cfg, ranges, env, path, extra, block);
            decide(cc, ty, &sub_fact(func, ctx, x), &sub_fact(func, ctx, y))
                .or_else(|| path.decide_rel(func, ranges, cc, ty, x, y))
                .or_else(|| extra.decide(func, (cfg, ranges, env, path, block), cc, x, y))
                .map(|b| (result, b))
        }
        (&Op::Fits(op, signed, raw_x, raw_y), Some(result))
            if ranges.constant(result).is_none() && func.ty(raw_x).is_small_int() =>
        {
            fits_truth(
                func,
                ranges,
                path,
                op,
                signed,
                dom.resolve(raw_x),
                dom.resolve(raw_y),
            )
            .map(|b| (result, b))
        }
        _ => return false,
    };
    if let Some((result, b)) = known {
        compares.push((result, b));
    }
    true
}

fn settle_ctrl(
    func: &Function,
    (cfg, ranges, env, path, extra, dom, selects, block, seen): (
        &Cfg,
        &Ranges,
        &Env,
        &mut Path,
        &mut ExtraPath,
        &mut DomState,
        &[SelectRec],
        Block,
        &[bool],
    ),
    inst: Inst,
    data: &InstData,
    decided: &mut Vec<(Inst, Option<bool>)>,
) {
    let truth_of = |c, p: &Path| {
        cond_truth(func, ranges, p, c)
            .or_else(|| extra.cond_truth(func, (cfg, ranges, env, p, block), c, 0))
            .or_else(|| sub_cond_truth(func, (cfg, ranges, env, p, extra, block), c, 0))
    };
    match (&data.op, data.result) {
        (&Op::Select(raw_cond, raw_t, raw_f), Some(result)) if func.ty(result).is_small_int() => {
            if let Some(taken) = truth_of(dom.resolve(raw_cond), path) {
                dom.record_alias(result, dom.resolve(if taken { raw_t } else { raw_f }));
            }
        }
        (&Op::Trapz(raw_cond) | &Op::Assume(raw_cond), _) => {
            let condition = dom.resolve(raw_cond);
            let self_proved = matches!(data.op, Op::Assume(_)) && ranges.raised.contains(&raw_cond);
            if !self_proved && truth_of(condition, path) == Some(true) {
                decided.push((inst, None));
            }
            path.assume(func, ranges, condition, true);
            extra.assume(func, (cfg, ranges, env, path, block), condition, true, 0);
            dom.propagate_selects(func, (cfg, ranges, env, path, extra), selects, block, seen);
        }
        (&Op::Brif(raw_cond, then, _, otherwise, _), _) if ranges.constant(raw_cond).is_none() => {
            if let Some(taken) = truth_of(dom.resolve(raw_cond), path)
                .or_else(|| transfer::avoided(func, ranges, then, otherwise))
            {
                decided.push((inst, Some(taken)));
            }
        }
        _ => {
            if let Some(addr) = transfer::deref_ptr(&data.op)
                && func.ty(addr) == Type::I64
            {
                let resolved = dom.resolve(addr);
                if path.fact(ranges, resolved).lo == 0 {
                    path.add(ranges, resolved, Fact::range(1, u64::MAX));
                }
            }
        }
    }
}

fn apply(
    func: &mut Function,
    decided: &[(Inst, Option<bool>)],
    rewrites: &[(Inst, Value, Value)],
) -> bool {
    if decided.is_empty() && rewrites.is_empty() {
        return false;
    }
    let mut dropped = vec![false; func.insts.len()];
    for &(inst, taken) in decided {
        let op = &mut func.insts[inst.index()].op;
        match (taken, std::mem::replace(op, Op::Trap)) {
            (Some(taken), Op::Brif(_, then, then_args, otherwise, else_args)) => {
                *op = if taken {
                    Op::Jump(then, then_args)
                } else {
                    Op::Jump(otherwise, else_args)
                };
            }
            (_, other) => {
                dropped[inst.index()] = taken.is_none();
                *op = other;
            }
        }
    }
    for &(inst, old, new) in rewrites {
        if !dropped[inst.index()] {
            func.insts[inst.index()]
                .op
                .map_operands(|v| if v == old { new } else { v });
        }
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dropped[inst.index()]);
    }
    !decided.is_empty() || !rewrites.is_empty()
}
