use super::fuse::{FuseCtx, preserved};
use super::patterns::{FOLD_LIMIT, SibMatch, odd_scale_source, scaled_op};
use super::super::asm::Reg;
use crate::alloc::{Location, Plan};
use crate::ir::{AtomicRmwOp, Binary, Convert, Function, Inst, IntCC, Op, Type, Value};
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
    (uses, pos_of, defs): (&[u32], &[super::Site], &[Option<Inst>]),
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

/// Elides single-use `Load + Alu + Store` (and stack equivalents) to the same address so `Store` emits one memory read-modify-write instruction.
pub(super) fn fuse_mem_rmw(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[super::Site], &[Option<Inst>]),
) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (spos, &sinst) in block.insts.iter().enumerate() {
            let (val, addr_match): (Value, Option<(Value, i32)>) = match func.insts[sinst.index()].op {
                Op::Store(v, a, off) => (v, Some((a, off))),
                Op::StackStore(v, _, _) => (v, None),
                _ => continue,
            };
            let ty = func.ty(val);
            if !ty.is_small_int() || uses[val.index()] != 1 || plan.is_static(val) || plan.fused[val.index()] || plan.sib[val.index()].is_some() {
                continue;
            }
            let Some((rb, rpos, rinst)) = pos_of[val.index()] else { continue };
            if rb != b_idx || rpos >= spos { continue; }
            let Op::Binary(op @ (Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor), a, b) = func.insts[rinst.index()].op else { continue };
            let matches_load = |v: Value| -> Option<usize> {
                if uses[v.index()] != 1 || plan.elided[v.index()] || func.ty(v) != ty { return None; }
                let (lb, lpos, linst) = pos_of[v.index()]?;
                if lb != b_idx || lpos >= rpos { return None; }
                let same = match (func.insts[sinst.index()].op.clone(), func.insts[linst.index()].op.clone()) {
                    (Op::Store(_, sa, so), Op::Load(la, lo)) => sa == la && so == lo,
                    (Op::StackStore(_, ss, so), Op::StackLoad(ls, lo)) => ss == ls && so == lo,
                    _ => false,
                };
                same.then_some(lpos)
            };
            let Some((lpos, load_v, other)) = matches_load(a).map(|lp| (lp, a, b)).or_else(|| (op != Binary::Isub).then(|| matches_load(b).map(|lp| (lp, b, a))).flatten()) else { continue };
            if block.insts[lpos + 1..spos].iter().any(|&mid| super::fuse::may_barrier_mem(&func.insts[mid.index()].op)) {
                continue;
            }
            if let Some((addr, _)) = addr_match {
                let addr_ok = if plan.elided[addr.index()] {
                    plan.sib[addr.index()].is_some_and(|m| {
                        let base_ok = match m.base {
                            SibBase::Frame(_) => true,
                            SibBase::Value(bv) => preserved(func, plan, b_idx, lpos, spos, &[rpos], bv),
                        };
                        base_ok && preserved(func, plan, b_idx, lpos, spos, &[rpos], idx_source(func, plan, defs, m.index))
                    })
                } else {
                    preserved(func, plan, b_idx, lpos, spos, &[rpos], addr)
                };
                if !addr_ok { continue; }
            }
            let imm_ok = plan.const_of(other).is_some_and(|c| ty.bytes() < 8 || i32::try_from(c).is_ok());
            let reg_ok = !plan.elided[other.index()] && plan.reg_of(other).is_some() && preserved(func, plan, b_idx, rpos, spos, &[], other);
            if !imm_ok && !reg_ok { continue; }
            plan.elided[load_v.index()] = true;
            plan.elided[val.index()] = true;
        }
    }
}

/// Elides single-use variable `Ushr(word, k)` feeding a flag-tested `Band(ushr, 1)` so x64 emits `bt k, word`.
pub(super) fn fuse_bt_shifts(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[super::Site]),
    tst_end: &[Option<usize>],
    pure_test: &[bool],
) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (bpos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let (Op::Binary(Binary::Band, x, y), Some(band)) = (&data.op, data.result) else { continue };
            let end = if pure_test[band.index()] {
                bpos
            } else if plan.elided[band.index()] {
                let Some(e) = tst_end[band.index()] else { continue };
                e
            } else {
                continue;
            };
            if !matches!(func.ty(band), Type::I32 | Type::I64) { continue; }
            let Some(shifted) = [(*x, *y), (*y, *x)].into_iter().find_map(|(s, m)| (plan.const_of(m) == Some(1)).then_some(s)) else { continue };
            if uses[shifted.index()] != 1 || plan.elided[shifted.index()] { continue; }
            let Some((sb, upos, uinst)) = pos_of[shifted.index()] else { continue };
            if sb != b_idx || upos >= end { continue; }
            let Op::Binary(Binary::Ushr, word, k) = func.insts[uinst.index()].op else { continue };
            if plan.const_of(k).is_some() || plan.elided[word.index()] || plan.elided[k.index()] || plan.reg_of(word).is_none() {
                continue;
            }
            if preserved(func, plan, b_idx, upos, end, &[], word) && preserved(func, plan, b_idx, upos, end, &[], k) {
                plan.elided[shifted.index()] = true;
            }
        }
    }
}

/// Whether a spilled single-use load may fold directly into a store without scratch-register collision.
pub(super) fn store_load_ok(func: &Function, plan: &Plan<Reg>, linst: Inst, load_res: Value, op: &Op) -> bool {
    if !matches!(plan.location(load_res), Location::Slot) || !func.ty(load_res).is_small_int() {
        return false;
    }
    match *op {
        Op::StackStore(val, ..) => val == load_res,
        Op::Store(val, addr, _) => {
            val == load_res
                && addr != load_res
                && (matches!(func.insts[linst.index()].op, Op::StackLoad(..))
                    || (!plan.elided[addr.index()] && matches!(plan.location(addr), Location::Reg(_) | Location::Frame(..))))
        }
        _ => false,
    }
}

fn eq_val(plan: &Plan<Reg>, ty: Type, a: Value, b: Value) -> bool {
    a == b
        || plan.const_of(a).zip(plan.const_of(b)).is_some_and(|(c1, c2)| {
            crate::fold::extend(c1, ty, false) == crate::fold::extend(c2, ty, false)
        })
}

fn quiet_between(func: &Function, plan: &Plan<Reg>, block: usize, mut range: std::ops::Range<usize>, skip: Option<usize>) -> bool {
    range.all(|p| {
        skip == Some(p) || {
            let mid = &func.insts[func.blocks[block].insts[p].index()];
            !matches!(mid.op, Op::Call { .. } | Op::TryCall { .. })
                && mid.result.is_some_and(|r| {
                    plan.elided[r.index()]
                        || (plan.is_static(r) && matches!(mid.op, Op::Const(_) | Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_)))
                })
        }
    })
}

fn cas_brif_edge_ok(func: &Function, plan: &Plan<Reg>, uses: &[u32], (block, start, res, expected, ty): (usize, usize, Value, Value, Type)) -> bool {
    let insts = &func.blocks[block].insts;
    let Some(bpos) = insts.len().checked_sub(1) else { return false };
    if bpos <= start + 1 { return false; }
    let Op::Brif(cond, then, ref ta, otherwise, ref ea) = func.insts[insts[bpos].index()].op else { return false };
    let arm_uses = ta.iter().chain(ea).filter(|&&a| a == res).count() as u32;
    if arm_uses == 0 || uses[res.index()] != 1 + arm_uses || !(plan.fused[cond.index()] || plan.elided[cond.index()]) {
        return false;
    }
    let Some(cpos) = (start + 1..bpos).find(|&p| func.insts[insts[p].index()].result == Some(cond)) else { return false };
    let Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) = func.insts[insts[cpos].index()].op else { return false };
    if !((lhs == res && eq_val(plan, ty, rhs, expected)) || (rhs == res && eq_val(plan, ty, lhs, expected)))
        || !quiet_between(func, plan, block, start + 1..bpos, Some(cpos))
    {
        return false;
    }
    for (target, args) in [(then, ta.as_slice()), (otherwise, ea.as_slice())] {
        if !args.contains(&res) { continue; }
        let params = &func.blocks[target.index()].params;
        if args.len() != params.len() { return false; }
        for (&arg, &param) in args.iter().zip(params) {
            if arg != res { continue; }
            if !matches!(plan.location(param), Location::Reg(_) | Location::Slot)
                || plan.reg_of(res).is_some_and(|r| plan.reg_of(param) == Some(r))
            {
                return false;
            }
            let pr = plan.reg_of(param);
            if args.iter().any(|&o| o != res && (o == param || pr.is_some_and(|r| plan.reg_of(o) == Some(r)))) {
                return false;
            }
        }
    }
    true
}

/// Marks dead or flag-only `AtomicRmw`/`AtomicCas` results (and `AtomicCas` results consumed only by a fused compare and its `Brif` edge) as fused.
pub(super) fn fuse_atomic_rmws(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, use_of): (&[u32], &[super::Site], &[super::Site]),
) {
    for (v, def) in pos_of.iter().enumerate() {
        let Some((block, start, inst)) = *def else { continue };
        let (res, ty) = (Value(v as u32), func.ty(Value(v as u32)));
        if !ty.is_small_int() || plan.elided[v] || plan.fused[v] {
            continue;
        }
        let expected = match func.insts[inst.index()].op {
            Op::AtomicRmw(rmw, _, _, value) => {
                if uses[v] == 0 && matches!(rmw, AtomicRmwOp::Add | AtomicRmwOp::Sub | AtomicRmwOp::Or | AtomicRmwOp::And | AtomicRmwOp::Xor | AtomicRmwOp::Xchg) {
                    plan.fused[v] = true;
                    if let Some((_, bit)) = super::super::bits::lock_bt_shift_bit(func, plan, |x| pos_of[x.index()].map(|(_, _, i)| i), rmw, value)
                        && uses[value.index()] == 1
                        && !plan.elided[value.index()]
                        && let Some((mb, mpos, _)) = pos_of[value.index()]
                        && mb == block
                        && mpos < start
                        && preserved(func, plan, block, mpos, start, &[], bit)
                    {
                        plan.elided[value.index()] = true;
                    }
                    continue;
                }
                if rmw != AtomicRmwOp::Sub || uses[v] != 1 { continue; }
                value
            }
            Op::AtomicCas(_, _, expected, _) => {
                if uses[v] == 0 || (uses[v] >= 2 && cas_brif_edge_ok(func, plan, uses, (block, start, res, expected, ty))) {
                    plan.fused[v] = true;
                    continue;
                }
                if uses[v] != 1 { continue; }
                expected
            }
            _ => continue,
        };
        let Some((used_in, end, user)) = use_of[v] else { continue };
        let udata = &func.insts[user.index()];
        let zero_expected = plan.const_of(expected).is_some_and(|c| crate::fold::extend(c, ty, false) == 0);
        let ok_user = match udata.op {
            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                udata.result.is_some_and(|r| !plan.elided[r.index()] && !plan.is_static(r))
                    && ((lhs == res && eq_val(plan, ty, rhs, expected)) || (rhs == res && eq_val(plan, ty, lhs, expected)))
            }
            Op::Brif(cond, _, ref ta, _, ref ea) => cond == res && zero_expected && ta.is_empty() && ea.is_empty(),
            Op::Trapz(cond) => cond == res && zero_expected,
            _ => false,
        };
        if used_in == block && end > start && ok_user && quiet_between(func, plan, block, start + 1..end, None) {
            plan.fused[v] = true;
        }
    }
}
