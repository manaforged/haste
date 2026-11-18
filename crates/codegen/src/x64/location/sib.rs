use super::fuse::{FuseCtx, preserved};
use super::patterns::{FOLD_LIMIT, SibMatch, odd_scale_source, scaled_op};
use super::super::asm::Reg;
use crate::alloc::{Location, Plan};
use crate::ir::{Convert, Function, Inst, Op, Type, Value};
use crate::live::{Sib, SibBase};

pub(super) fn sib_ok(ctx: &FuseCtx<'_>, m: &SibMatch, extra_skips: &[usize], end_pos: usize) -> bool {
    let mut skip: Vec<usize> = m.elide.iter().map(|&(_, p)| p).collect();
    skip.extend_from_slice(extra_skips);
    let base_ok = match m.sib.base {
        SibBase::Frame(_) => true,
        SibBase::Value(base) => preserved(ctx.func, ctx.plan, ctx.b_idx, m.base_start, end_pos, &skip, base),
    };
    base_ok && preserved(ctx.func, ctx.plan, ctx.b_idx, m.idx_start, end_pos, &skip, m.sib.index)
}

pub(super) fn idx_source(func: &Function, plan: &Plan<Reg>, defs: &[Option<Inst>], idx: Value) -> Value {
    odd_scale_source(func, plan, |v| defs[v.index()], idx).map_or(idx, |(x, _)| x)
}

fn collect_sib_chain(
    ctx: &FuseCtx<'_>,
    root: Value,
    root_pos: usize,
    ls_uses: &[Vec<(usize, usize, i32)>],
    const_kids: &[Vec<(usize, usize, Value, i64)>],
) -> Option<(Vec<(Value, usize, i64)>, usize, bool)> {
    let (mut nodes, mut work) = (Vec::new(), vec![(root, 0_i64)]);
    let (mut max_end, mut total_mem) = (0_usize, 0_usize);
    while let Some((cur, delta)) = work.pop() {
        let mut kids = 0_usize;
        for &(kb, kpos, kid, kdelta) in &const_kids[cur.index()] {
            if kb == ctx.b_idx && kpos > root_pos && !ctx.plan.elided[kid.index()] && ctx.plan.sib[kid.index()].is_none() {
                let next = delta.checked_add(kdelta).filter(|d| (-FOLD_LIMIT..=FOLD_LIMIT).contains(d))?;
                nodes.push((kid, kpos, next));
                work.push((kid, next));
                kids += 1;
            }
        }
        if ctx.uses[cur.index()] as usize != ls_uses[cur.index()].len() + kids {
            return None;
        }
        for &(ub, upos, _) in &ls_uses[cur.index()] {
            if ub != ctx.b_idx || upos <= root_pos {
                return None;
            }
            max_end = max_end.max(upos);
        }
        total_mem += ls_uses[cur.index()].len();
    }
    (total_mem > 0 && max_end > root_pos).then_some((nodes, max_end, total_mem > 1))
}

fn chain_fits(disp: i32, root: Value, nodes: &[(Value, usize, i64)], ls_uses: &[Vec<(usize, usize, i32)>]) -> bool {
    std::iter::once((root, 0_i64))
        .chain(nodes.iter().map(|&(v, _, d)| (v, d)))
        .all(|(v, d)| ls_uses[v.index()].iter().all(|&(_, _, off)| (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(i64::from(disp) + d + i64::from(off)))))
}

pub(super) fn find_cand_sib(
    ctx: &FuseCtx<'_>,
    (pos, addr): (usize, Value),
    ls_uses: &[Vec<(usize, usize, i32)>],
    const_kids: &[Vec<(usize, usize, Value, i64)>],
) -> Option<(SibMatch, Vec<(Value, usize, i64)>)> {
    if ctx.plan.elided[addr.index()] {
        return None;
    }
    let (nodes, end_pos, multi_use) = collect_sib_chain(ctx, addr, pos, ls_uses, const_kids)?;
    let m = ctx.match_sib(addr, pos, multi_use)?;
    if !chain_fits(m.sib.disp, addr, &nodes, ls_uses) {
        return None;
    }
    if multi_use && matches!(m.sib.base, SibBase::Value(b) if ctx.plan.reg_of(b).is_none()) {
        return None;
    }
    let mut skips = vec![pos];
    skips.extend(nodes.iter().map(|&(_, p, _)| p));
    sib_ok(ctx, &m, &skips, end_pos).then_some((m, nodes))
}

struct SibUpgrade {
    elide: Vec<Value>,
    targets: Vec<(Value, Sib)>,
}

fn collect_sib_groups(func: &Function, plan: &Plan<Reg>, b_idx: usize, unscaled_only: bool) -> Vec<(Value, Vec<(Value, usize, Sib)>)> {
    let mut groups: Vec<(Value, Vec<(Value, usize, Sib)>)> = Vec::new();
    for (pos, &inst) in func.blocks[b_idx].insts.iter().enumerate() {
        let Some(res) = func.insts[inst.index()].result else { continue };
        let Some(mode) = plan.sib[res.index()] else { continue };
        if !plan.elided[res.index()] || (unscaled_only && mode.shift != 0) || plan.elided[mode.index.index()] {
            continue;
        }
        if let Some((_, list)) = groups.iter_mut().find(|(k, _)| *k == mode.index) {
            list.push((res, pos, mode));
        } else {
            groups.push((mode.index, vec![(res, pos, mode)]));
        }
    }
    groups
}

fn group_mem_span(ctx: &FuseCtx<'_>, core: Value, items: &[(Value, usize, Sib)], ls_uses: &[Vec<(usize, usize, i32)>]) -> Option<(usize, usize)> {
    let (mut direct, mut total_mem, mut max_end) = (0_usize, 0_usize, 0_usize);
    for &(v, vpos, mode) in items {
        if matches!(mode.base, SibBase::Value(b) if b == core) {
            return None;
        }
        let (_, _, inst) = ctx.pos_of[v.index()]?;
        ctx.func.insts[inst.index()].op.for_operands(|op_v| {
            if op_v == core {
                direct += 1;
            }
        });
        for &(ub, upos, _) in &ls_uses[v.index()] {
            if ub != ctx.b_idx || upos <= vpos {
                return None;
            }
            max_end = max_end.max(upos);
            total_mem += 1;
        }
    }
    (direct == ctx.uses[core.index()] as usize && total_mem > 0).then_some((total_mem, max_end))
}

fn odd_group_ok(ctx: &FuseCtx<'_>, items: &[(Value, usize, Sib)], shift: u8, total_mem: usize) -> bool {
    let profitable = (shift > 0 && total_mem <= 2) || (shift == 0 && total_mem == 1);
    profitable && items.iter().all(|&(_, _, m)| !matches!(m.base, SibBase::Value(b) if ctx.plan.elided[b.index()] || ctx.plan.reg_of(b).is_none()))
}

fn group_disp_fits(items: &[(Value, usize, Sib)], ls_uses: &[Vec<(usize, usize, i32)>], disp_delta: i64) -> bool {
    disp_delta == 0
        || items.iter().all(|&(v, _, m)| {
            let next = i64::from(m.disp) + disp_delta;
            (-FOLD_LIMIT..=FOLD_LIMIT).contains(&next)
                && ls_uses[v.index()].iter().all(|&(_, _, off)| (-FOLD_LIMIT..=FOLD_LIMIT).contains(&(next + i64::from(off))))
        })
}

fn eval_sib_group(ctx: &FuseCtx<'_>, core: Value, items: &[(Value, usize, Sib)], ls_uses: &[Vec<(usize, usize, i32)>]) -> Option<SibUpgrade> {
    let (sb, spos, sinst) = ctx.pos_of[core.index()]?;
    if sb != ctx.b_idx || ctx.func.ty(core) != Type::I64 {
        return None;
    }
    let (raw_idx, shift, odd) = scaled_op(ctx.func, ctx.plan, sinst)?;
    let (total_mem, max_end) = group_mem_span(ctx, core, items, ls_uses)?;
    if odd.is_some() && !odd_group_ok(ctx, items, shift, total_mem) {
        return None;
    }
    let mut elide_pos = vec![(core, spos)];
    let (idx, disp_delta, idx_start) = if odd.is_some() {
        let (idx, idx_start) = ctx.unwrap_uext(raw_idx, spos, true, &mut elide_pos);
        (idx, 0_i64, idx_start)
    } else {
        let (idx, delta, idx_start) = ctx.unwrap_uext_const(raw_idx, spos, (shift, 0), total_mem > 1, &mut elide_pos);
        if group_disp_fits(items, ls_uses, delta) {
            (idx, delta, idx_start)
        } else {
            elide_pos.truncate(1);
            let (idx, idx_start) = ctx.unwrap_uext(raw_idx, spos, total_mem > 1, &mut elide_pos);
            (idx, 0_i64, idx_start)
        }
    };
    let skip: Vec<usize> = elide_pos.iter().map(|&(_, p)| p).collect();
    if !ctx.index_ok(idx) || !preserved(ctx.func, ctx.plan, ctx.b_idx, idx_start, max_end, &skip, idx) {
        return None;
    }
    let index = if odd.is_some() { core } else { idx };
    let targets = items.iter().map(|&(v, _, m)| (v, Sib { index, shift, disp: (i64::from(m.disp) + disp_delta) as i32, ..m })).collect();
    Some(SibUpgrade { elide: elide_pos.into_iter().map(|(v, _)| v).collect(), targets })
}

fn uext_group_inner(ctx: &FuseCtx<'_>, sinst: Inst) -> Option<Value> {
    let inner = match ctx.func.insts[sinst.index()].op {
        Op::Convert(Convert::Uextend, inner) => inner,
        Op::Convert(Convert::Sextend, inner) if ctx.nonneg_small(inner) => inner,
        _ => return None,
    };
    let ty = ctx.func.ty(inner);
    (ty.is_small_int() && ty.bytes() <= 4 && !ctx.plan.elided[inner.index()]).then_some(inner)
}

fn eval_uext_group(ctx: &FuseCtx<'_>, core: Value, items: &[(Value, usize, Sib)], ls_uses: &[Vec<(usize, usize, i32)>]) -> Option<SibUpgrade> {
    let (sb, spos, sinst) = ctx.pos_of[core.index()]?;
    if sb != ctx.b_idx || ctx.func.ty(core) != Type::I64 {
        return None;
    }
    let inner = uext_group_inner(ctx, sinst)?;
    let (_, max_end) = group_mem_span(ctx, core, items, ls_uses)?;
    let mut elide_pos = vec![(core, spos)];
    let (idx, disp_delta, idx_start) = match ctx.peel_small_chain(inner, spos, (0, 0), true, &mut elide_pos) {
        Some((idx, delta, idx_start)) if ctx.plan.zero_ext[idx.index()] && ctx.plan.reg_of(idx).is_some() && group_disp_fits(items, ls_uses, delta) => {
            (idx, delta, idx_start)
        }
        _ => {
            elide_pos.truncate(1);
            if !ctx.plan.zero_ext[inner.index()] || ctx.plan.reg_of(inner).is_none() {
                return None;
            }
            (inner, 0, spos)
        }
    };
    let skip: Vec<usize> = elide_pos.iter().map(|&(_, p)| p).collect();
    if !preserved(ctx.func, ctx.plan, ctx.b_idx, idx_start, max_end, &skip, idx) {
        return None;
    }
    let targets = items.iter().map(|&(v, _, m)| (v, Sib { index: idx, disp: (i64::from(m.disp) + disp_delta) as i32, ..m })).collect();
    Some(SibUpgrade { elide: elide_pos.into_iter().map(|(v, _)| v).collect(), targets })
}

pub(super) fn apply_sib_groups(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[Option<(usize, usize, Inst)>], &[Option<Inst>]),
    ls_uses: &[Vec<(usize, usize, i32)>],
) {
    for unscaled_only in [true, false] {
        for b_idx in 0..func.blocks.len() {
            for (core, items) in collect_sib_groups(func, plan, b_idx, unscaled_only) {
                let ctx = FuseCtx { func, plan, uses, pos_of, defs, b_idx };
                let up = if unscaled_only { eval_sib_group(&ctx, core, &items, ls_uses) } else { eval_uext_group(&ctx, core, &items, ls_uses) };
                let Some(up) = up else { continue };
                for v in up.elide {
                    plan.elided[v.index()] = true;
                }
                for (v, sib) in up.targets {
                    plan.sib[v.index()] = Some(sib);
                }
            }
        }
    }
}

/// `x * 2 + c` as the address form `c(x,x)`, when `root` adds a constant to a doubling of its own block that nothing else reads.
pub(super) fn double(ctx: &FuseCtx<'_>, root: Value, root_pos: usize) -> Option<SibMatch> {
    if ctx.plan.elided[root.index()] || ctx.plan.is_static(root) || ctx.func.ty(root) != Type::I64 {
        return None;
    }
    let mut elide = Vec::new();
    let (node, disp, _) = ctx.peel_const(root, root_pos, false, &mut elide)?;
    let (block, pos, inst) = ctx.pos_of[node.index()]?;
    if node == root || block != ctx.b_idx || ctx.plan.elided[node.index()] || ctx.uses[node.index()] != 1 || pos >= root_pos {
        return None;
    }
    let (x, 1, None) = scaled_op(ctx.func, ctx.plan, inst)? else { return None };
    elide.push((node, pos));
    let (x, start) = ctx.unwrap_base_uext(x, pos, &mut elide);
    let base = !ctx.plan.elided[x.index()] && !matches!(ctx.plan.location(x), Location::Frame(..) | Location::Imm(..));
    base.then_some(SibMatch { sib: Sib { base: SibBase::Value(x), index: x, shift: 0, disp: i32::try_from(disp).ok()? }, elide, base_start: start, idx_start: start })
}
