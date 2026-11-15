use super::patterns::{
    FOLD_LIMIT, Root, SibMatch, adc_parts, bextr_parts, flag_cmp, is_zero_const, mid_preserves_flags, odd_scale_source, parked, scaled_op, shld_parts,
};
use super::super::{FRAME_REGS, REGS};
use super::super::asm::Reg;
use super::super::atomic::uses_scratch;
use super::super::features::Features;
use super::super::frame::SAVED;
use super::super::wide::WIDE_SCRATCH;
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Block, Convert, Function, Inst, InstData, IntCC, Op, Type, Unary, Value};
use crate::live::{Sib, SibBase};

pub(super) struct FuseCtx<'a> {
    pub(super) func: &'a Function,
    pub(super) plan: &'a Plan<Reg>,
    pub(super) uses: &'a [u32],
    pub(super) pos_of: &'a [Option<(usize, usize, Inst)>],
    pub(super) defs: &'a [Option<Inst>],
    pub(super) b_idx: usize,
}

fn scratches(func: &Function, data: &InstData, reg: Reg) -> bool {
    let mut wide = data.result.is_some_and(|r| func.ty(r) == Type::I128);
    data.op.for_operands(|v| wide |= func.ty(v) == Type::I128);
    (wide && WIDE_SCRATCH.contains(&reg)) || ((wide || uses_scratch(func, &data.op)) && REGS.op_scratch.contains(&reg))
}

fn preserved(func: &Function, plan: &Plan<Reg>, b_idx: usize, start: usize, end: usize, skip: &[usize], val: Value) -> bool {
    if plan.elided[val.index()] {
        return false;
    }
    if start + 1 >= end || plan.location(val).is_static() {
        return true;
    }
    if end - start > 128 {
        return false;
    }
    let (reg, slot_id) = match plan.location(val) {
        Location::Reg(reg) => (Some(reg), None),
        Location::Slot => {
            let Some(id) = plan.shared.get(val.index()).copied().flatten() else { return true };
            (None, Some(id))
        }
        _ => return false,
    };
    let clobbers = |r: Value| {
        reg.is_some_and(|reg| plan.reg_of(r) == Some(reg))
            || slot_id.is_some_and(|id| matches!(plan.location(r), Location::Slot) && plan.shared.get(r.index()).copied().flatten() == Some(id))
    };
    for pos in start + 1..end {
        if skip.contains(&pos) {
            continue;
        }
        let data = &func.insts[func.blocks[b_idx].insts[pos].index()];
        if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if (reg.is_some() && matches!(data.op, Op::Call { .. } | Op::TryCall { .. }))
            || reg.is_some_and(|reg| scratches(func, data, reg))
            || data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(clobbers)
            || crate::live::results(data).any(clobbers)
        {
            return false;
        }
    }
    true
}

fn band_ok(func: &Function, plan: &Plan<Reg>, b_idx: usize, (mpos, end, skip): (usize, usize, &[usize]), (x, y): (Value, Value)) -> bool {
    func.ty(x).is_small_int() && preserved(func, plan, b_idx, mpos, end, skip, x) && preserved(func, plan, b_idx, mpos, end, skip, y)
}

fn zero_cmp_band(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    pos_of: &[Option<(usize, usize, Inst)>],
    b_idx: usize,
    lhs: Value,
    rhs: Value,
) -> Option<(Value, usize, Value, Value)> {
    let masked = is_zero_const(func, plan, rhs).then_some(lhs).or_else(|| is_zero_const(func, plan, lhs).then_some(rhs))?;
    if uses[masked.index()] != 1 || plan.elided[masked.index()] {
        return None;
    }
    let (mb, mpos, minst) = pos_of[masked.index()]?;
    if mb != b_idx {
        return None;
    }
    let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op else { return None };
    Some((masked, mpos, x, y))
}

fn sib_ok(ctx: &FuseCtx<'_>, m: &SibMatch, extra_skips: &[usize], end_pos: usize) -> bool {
    let mut skip: Vec<usize> = m.elide.iter().map(|&(_, p)| p).collect();
    skip.extend_from_slice(extra_skips);
    let base_ok = match m.sib.base {
        SibBase::Frame(_) => true,
        SibBase::Value(base) => preserved(ctx.func, ctx.plan, ctx.b_idx, m.base_start, end_pos, &skip, base),
    };
    base_ok && preserved(ctx.func, ctx.plan, ctx.b_idx, m.idx_start, end_pos, &skip, m.sib.index)
}

fn idx_source(func: &Function, plan: &Plan<Reg>, defs: &[Option<Inst>], idx: Value) -> Value {
    odd_scale_source(func, plan, |v| defs[v.index()], idx).map_or(idx, |(x, _)| x)
}

fn may_barrier_mem(op: &Op) -> bool {
    matches!(
        op,
        Op::Store(..)
            | Op::StackStore(..)
            | Op::Copy(..)
            | Op::Move(..)
            | Op::Fence
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Trap
            | Op::Trapz(..)
            | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..)
    )
}

fn load_fuses(ctx: &FuseCtx<'_>, lpos: usize, eff_pos: usize, skip: &[usize]) -> bool {
    let insts = &ctx.func.blocks[ctx.b_idx].insts;
    if eff_pos <= lpos || insts[lpos + 1..eff_pos].iter().any(|&mid| may_barrier_mem(&ctx.func.insts[mid.index()].op)) {
        return false;
    }
    match ctx.func.insts[insts[lpos].index()].op {
        Op::StackLoad(..) => true,
        Op::Load(addr, _) if ctx.plan.elided[addr.index()] => ctx.plan.sib[addr.index()].is_some_and(|m| {
            let base_ok = match m.base {
                SibBase::Frame(_) => true,
                SibBase::Value(b) => preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, b),
            };
            let idx = idx_source(ctx.func, ctx.plan, ctx.defs, m.index);
            base_ok && preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, idx)
        }),
        Op::Load(addr, _) => preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, addr),
        _ => false,
    }
}

fn addr_reads(func: &Function, plan: &Plan<Reg>, defs: &[Option<Inst>], addr: Value, reg: Reg) -> bool {
    if plan.elided[addr.index()] {
        return plan.sib[addr.index()].is_some_and(|m| {
            let idx = idx_source(func, plan, defs, m.index);
            matches!(m.base, SibBase::Value(base) if !plan.elided[base.index()] && plan.reg_of(base) == Some(reg))
                || (!plan.elided[idx.index()] && plan.reg_of(idx) == Some(reg))
        });
    }
    plan.reg_of(addr) == Some(reg)
}

fn reads_reg(ctx: &FuseCtx<'_>, v: Value, reg: Reg) -> bool {
    ctx.plan.reg_of(v) == Some(reg)
        || ctx.pos_of[v.index()].is_some_and(|(_, _, inst)| matches!(ctx.func.insts[inst.index()].op, Op::Load(addr, _) if addr_reads(ctx.func, ctx.plan, ctx.defs, addr, reg)))
}

fn load_dst_clobber(ctx: &FuseCtx<'_>, linst: Inst, load_res: Value, (u_res, op): (Option<Value>, Binary), (a, b): (Value, Value)) -> bool {
    let Some(r_dst) = u_res.and_then(|r| ctx.plan.reg_of(r)) else { return false };
    if op == Binary::Isub && a == load_res {
        return false;
    }
    let other = if a == load_res { b } else { a };
    if ctx.plan.reg_of(other) == Some(r_dst) {
        return false;
    }
    let Op::Load(addr, _) = ctx.func.insts[linst.index()].op else { return false };
    addr_reads(ctx.func, ctx.plan, ctx.defs, addr, r_dst)
}

fn iadd_sub_const(plan: &Plan<Reg>, op: &Op) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => match (plan.const_of(a), plan.const_of(b)) {
            (None, Some(c)) => Some((a, c)),
            (Some(c), None) => Some((b, c)),
            _ => None,
        },
        Op::Binary(Binary::Isub, a, b) if plan.const_of(a).is_none() => plan.const_of(b).and_then(|c| c.checked_neg()).map(|c| (a, c)),
        _ => None,
    }
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

fn find_cand_sib(
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
    let mut skips = vec![pos];
    skips.extend(nodes.iter().map(|&(_, p, _)| p));
    sib_ok(ctx, &m, &skips, end_pos).then_some((m, nodes))
}

struct SibUpgrade {
    elide: Vec<Value>,
    targets: Vec<(Value, Sib)>,
}

fn collect_sib_groups(func: &Function, plan: &Plan<Reg>, b_idx: usize) -> Vec<(Value, Vec<(Value, usize, Sib)>)> {
    let mut groups: Vec<(Value, Vec<(Value, usize, Sib)>)> = Vec::new();
    for (pos, &inst) in func.blocks[b_idx].insts.iter().enumerate() {
        let Some(res) = func.insts[inst.index()].result else { continue };
        let Some(mode) = plan.sib[res.index()] else { continue };
        if !plan.elided[res.index()] || mode.shift != 0 || plan.elided[mode.index.index()] {
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
    let (idx, idx_start) = ctx.unwrap_uext(raw_idx, spos, total_mem > 1 || odd.is_some(), &mut elide_pos);
    if !ctx.index_ok(idx) || !preserved(ctx.func, ctx.plan, ctx.b_idx, idx_start, max_end, &[spos, idx_start], idx) {
        return None;
    }
    let index = if odd.is_some() { core } else { idx };
    let targets = items.iter().map(|&(v, _, m)| (v, Sib { index, shift, ..m })).collect();
    Some(SibUpgrade { elide: elide_pos.into_iter().map(|(v, _)| v).collect(), targets })
}

fn cmp_operand_ok(ctx: &FuseCtx<'_>, cpos: usize, last_pos: usize, v: Value) -> bool {
    if !ctx.plan.elided[v.index()] {
        return preserved(ctx.func, ctx.plan, ctx.b_idx, cpos, last_pos, &[], v);
    }
    ctx.pos_of[v.index()].is_some_and(|(lb, lpos, _)| lb == ctx.b_idx && lpos < cpos && load_fuses(ctx, lpos, last_pos, &[]))
}

fn bnot_elided_ok(ctx: &FuseCtx<'_>, features: &Features, op: Binary, other: Value) -> bool {
    op == Binary::Band
        && features.bmi1
        && ctx.defs[other.index()]
            .is_some_and(|i| matches!(ctx.func.insts[i.index()].op, Op::Unary(Unary::Bnot, inner) if !ctx.plan.elided[inner.index()]))
}

fn load_binop_ok(ctx: &FuseCtx<'_>, features: &Features, linst: Inst, load_res: Value, udata: &InstData) -> bool {
    let Op::Binary(op, a, b) = udata.op else { return false };
    let ty = ctx.func.ty(load_res);
    if !matches!(ty, Type::I32 | Type::I64) {
        return false;
    }
    match op {
        Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Imul => {
            let other = if a == load_res { b } else { a };
            udata.result.is_some_and(|r| ctx.plan.sib[r.index()].is_none())
                && (!ctx.plan.elided[other.index()] || bnot_elided_ok(ctx, features, op, other))
                && !load_dst_clobber(ctx, linst, load_res, (udata.result, op), (a, b))
        }
        Binary::Rotl | Binary::Rotr => {
            features.bmi2 && a == load_res && ctx.plan.const_of(b).is_some_and(|c| (c & i64::from(ty.bits() - 1)) != 0)
        }
        Binary::Ishl | Binary::Ushr | Binary::Sshr => features.bmi2 && a == load_res && ctx.plan.const_of(b).is_none(),
        _ => false,
    }
}

fn load_elide_ok(ctx: &FuseCtx<'_>, features: &Features, linst: Inst, load_res: Value, udata: &InstData) -> bool {
    let ty = ctx.func.ty(load_res);
    let non_wide = udata.result.is_some_and(|r| ctx.func.ty(r) != Type::I128);
    match udata.op {
        Op::Icmp(..) => true,
        Op::Convert(Convert::Uextend | Convert::Sextend, _) => ty.bytes() <= 4 && non_wide,
        Op::Convert(Convert::Ireduce, _) => ty.bytes() <= 8 && non_wide,
        Op::Binary(..) => load_binop_ok(ctx, features, linst, load_res, udata),
        Op::Unary(Unary::Clz, _) => matches!(ty, Type::I32 | Type::I64) && features.lzcnt,
        Op::Unary(Unary::Ctz, _) => matches!(ty, Type::I32 | Type::I64) && features.bmi1,
        Op::Unary(Unary::Popcnt, _) => matches!(ty, Type::I32 | Type::I64) && features.popcnt,
        _ => false,
    }
}

pub(in super::super) fn fuse(func: &Function, plan: &mut Plan<Reg>, features: &Features) -> Vec<Option<Root>> {
    let n = func.types.len();
    let mut roots = vec![None; n];
    if func.size() > 8192 {
        return roots;
    }
    let defs = crate::fold::defs(func);
    let (mut uses, mut pos_of, mut ls_uses, mut use_of) = (vec![0_u32; n], vec![None; n], vec![Vec::new(); n], vec![None; n]);
    let (mut last_flag_use, mut non_flag_use, mut const_kids) = (vec![None; n], vec![false; n], vec![Vec::new(); n]);
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let flag_cond = match &data.op {
                Op::Select(c, ..) | Op::Brif(c, ..) | Op::Trapz(c) => Some(*c),
                _ => None,
            };
            data.op.for_operands(|v| {
                uses[v.index()] += 1;
                use_of[v.index()] = Some((b_idx, pos, inst));
                if flag_cond == Some(v) && !non_flag_use[v.index()] {
                    match last_flag_use[v.index()] {
                        Some((pb, ppos)) if pb != b_idx || ppos == pos => non_flag_use[v.index()] = true,
                        _ => last_flag_use[v.index()] = Some((b_idx, pos)),
                    }
                } else {
                    non_flag_use[v.index()] = true;
                }
            });
            match data.op {
                Op::Load(addr, off) => ls_uses[addr.index()].push((b_idx, pos, off)),
                Op::Store(val, addr, off) if val != addr => ls_uses[addr.index()].push((b_idx, pos, off)),
                _ => {}
            }
            if let Some(res) = data.result {
                pos_of[res.index()] = Some((b_idx, pos, inst));
                if func.ty(res) == Type::I64
                    && plan.sib[res.index()].is_none()
                    && !plan.elided[res.index()]
                    && let Some((parent, delta)) = iadd_sub_const(plan, &data.op)
                {
                    const_kids[parent.index()].push((b_idx, pos, res, delta));
                }
            }
        }
    }
    let (mut branch_tst, mut cmp_tst, mut cand_sibs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cand_shld, mut cand_adc, mut cand_bextr, mut cand_loads, mut cand_alu) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else {
                if let Op::Brif(cond_val, ..) | Op::Trapz(cond_val) = data.op
                    && uses[cond_val.index()] == 1
                    && !plan.elided[cond_val.index()]
                    && let Some((cb, cpos, cinst)) = pos_of[cond_val.index()]
                    && cb == b_idx
                    && cpos < pos
                {
                    match func.insts[cinst.index()].op {
                        Op::Binary(Binary::Band, x, y) if band_ok(func, plan, b_idx, (cpos, pos, &[]), (x, y)) => {
                            branch_tst.push((pos, cond_val, None));
                        }
                        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                            if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
                                && mpos < cpos
                                && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y))
                            {
                                branch_tst.push((pos, cond_val, Some(m)));
                            }
                        }
                        _ => {}
                    }
                }
                continue;
            };
            let ty = func.ty(res);
            match data.op {
                Op::Binary(Binary::Band, x, y) if features.bmi1 && !plan.elided[res.index()] && ty.is_small_int() => {
                    for (cand, other) in [(x, y), (y, x)] {
                        if !plan.const_of(other).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))
                            && uses[cand.index()] == 1
                            && !plan.elided[cand.index()]
                            && let Some((cb, cpos, cinst)) = pos_of[cand.index()]
                            && cb == b_idx
                            && cpos < pos
                            && let Op::Unary(Unary::Bnot, inner) = func.insts[cinst.index()].op
                            && preserved(func, plan, b_idx, cpos, pos, &[], inner)
                        {
                            plan.elided[cand.index()] = true;
                            break;
                        }
                    }
                    if matches!(ty, Type::I32 | Type::I64) {
                        cand_bextr.push((b_idx, pos, res, Binary::Band, x, y));
                        cand_alu.push((b_idx, pos, res, Binary::Band, x, y));
                    }
                }
                Op::Binary(op @ (Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor), x, y) if !plan.elided[res.index()] => {
                    if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd) && matches!(ty, Type::I32 | Type::I64) {
                        cand_shld.push((b_idx, pos, res, x, y));
                    }
                    if matches!(op, Binary::Iadd | Binary::Isub) && matches!(ty, Type::I32 | Type::I64) {
                        cand_adc.push((b_idx, pos, res, op, x, y));
                    }
                    if matches!(op, Binary::Iadd | Binary::Isub) && plan.sib[res.index()].is_none() && ty == Type::I64 {
                        cand_sibs.push((b_idx, pos, res));
                    }
                    if matches!(ty, Type::I32 | Type::I64) {
                        cand_alu.push((b_idx, pos, res, op, x, y));
                    }
                }
                Op::Binary(op @ (Binary::Band | Binary::Ushr), x, y) if !plan.elided[res.index()] && matches!(ty, Type::I32 | Type::I64) => {
                    if features.bmi1 {
                        cand_bextr.push((b_idx, pos, res, op, x, y));
                    }
                    if op == Binary::Band {
                        cand_alu.push((b_idx, pos, res, op, x, y));
                    }
                }
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) if !plan.elided[res.index()] => {
                    cmp_tst.push((b_idx, pos, res, lhs, rhs));
                }
                Op::Load(..) | Op::StackLoad(..) if uses[res.index()] == 1 && !plan.elided[res.index()] && ty.is_small_int() => {
                    cand_loads.push((b_idx, pos, res));
                }
                _ => {}
            }
        }
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let (Op::Select(cond, ..) | Op::Brif(cond, ..) | Op::Trapz(cond)) = func.insts[inst.index()].op else { continue };
            let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
            let Some(tree) = ctx.cond_tree(cond, pos) else { continue };
            let [leaf] = tree.leaves[..] else { continue };
            let Some((_, leaf_pos, _)) = pos_of[leaf.index()] else { continue };
            let skip: Vec<usize> = tree.nodes.iter().map(|node| node.1).collect();
            if !preserved(func, plan, b_idx, leaf_pos, pos, &skip, leaf) {
                continue;
            }
            for &(v, _, is_leaf, _) in &tree.nodes {
                plan.elided[v.index()] = !is_leaf;
            }
            roots[cond.index()] = Some((leaf, None, tree.table));
            last_flag_use[leaf.index()] = Some((b_idx, pos));
            non_flag_use[leaf.index()] = false;
        }
    }
    let mut tst_end = vec![None; n];
    for (bpos, cond_val, m) in branch_tst {
        if roots[cond_val.index()].is_some() {
            continue;
        }
        plan.elided[cond_val.index()] = true;
        tst_end[cond_val.index()] = Some(bpos);
        if let Some(m) = m {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(bpos);
        }
    }
    for (b_idx, cpos, res, lhs, rhs) in cmp_tst {
        if !plan.elided[res.index()]
            && let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && band_ok(func, plan, b_idx, (mpos, cpos, &[]), (x, y))
        {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(cpos);
        }
    }
    let mut skip_flags = vec![false; n];
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    for (b_idx, pos, res, x, y) in cand_shld {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() {
            continue;
        }
        let Some((hi, lo, _, _, _)) = shld_parts(func, plan, func.ty(res), (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let (Some((bx, px, _)), Some((by, py, _))) = (pos_of[x.index()], pos_of[y.index()]) else { continue };
        if bx != b_idx || by != b_idx || px >= pos || py >= pos {
            continue;
        }
        let skip = [px, py];
        let min_p = px.min(py);
        let ok = if hi == lo {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi)
        } else {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi) && preserved(func, plan, b_idx, min_p, pos, &skip, lo)
        };
        if ok {
            plan.elided[x.index()] = true;
            plan.elided[y.index()] = true;
            skip_flags[res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_adc {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((cand, inner, _, a, b)) = adc_parts(func, plan, func.ty(res), op, (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let Some((cb, cpos, _)) = pos_of[cand.index()] else { continue };
        if cb != b_idx || cpos >= pos {
            continue;
        }
        let (cmp_pos, skip): (usize, &[usize]) = match inner {
            Some(iv) => match pos_of[iv.index()] {
                Some((ib, ipos, _)) if ib == b_idx && ipos < cpos => (ipos, &[cpos]),
                _ => continue,
            },
            None => (cpos, &[]),
        };
        if preserved(func, plan, b_idx, cmp_pos, pos, skip, a) && preserved(func, plan, b_idx, cmp_pos, pos, skip, b) {
            plan.elided[cand.index()] = true;
            if let Some(iv) = inner {
                plan.elided[iv.index()] = true;
            }
            skip_flags[res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_bextr {
        if plan.elided[res.index()] || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((Some(mid), src, _, _)) = bextr_parts(func, plan, func.ty(res), op, (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let Some((mb, mpos, _)) = pos_of[mid.index()] else { continue };
        if mb == b_idx && mpos < pos && preserved(func, plan, b_idx, mpos, pos, &[], src) {
            plan.elided[mid.index()] = true;
        }
    }
    for &(b_idx, pos, addr) in cand_sibs.iter().rev() {
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
        let Some((m, nodes)) = find_cand_sib(&ctx, (pos, addr), &ls_uses, &const_kids) else { continue };
        plan.sib[addr.index()] = Some(m.sib);
        plan.elided[addr.index()] = true;
        for (kid, _, delta) in nodes {
            plan.sib[kid.index()] = Some(Sib { disp: (i64::from(m.sib.disp) + delta) as i32, ..m.sib });
            plan.elided[kid.index()] = true;
        }
        for (v, _) in m.elide {
            plan.elided[v.index()] = true;
        }
    }
    for b_idx in 0..func.blocks.len() {
        for (core, items) in collect_sib_groups(func, plan, b_idx) {
            let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
            let Some(up) = eval_sib_group(&ctx, core, &items, &ls_uses) else { continue };
            for v in up.elide {
                plan.elided[v.index()] = true;
            }
            for (v, sib) in up.targets {
                plan.sib[v.index()] = Some(sib);
            }
        }
    }
    for &(b_idx, pos, res) in cand_sibs.iter().rev() {
        if plan.elided[res.index()] || uses[res.index()] == 0 || skip_flags[res.index()] {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
        let Some(m) = ctx.match_sib(res, pos, false) else { continue };
        if m.elide.is_empty() && !matches!(m.sib.base, SibBase::Frame(_)) {
            continue;
        }
        if sib_ok(&ctx, &m, &[], pos) {
            plan.sib[res.index()] = Some(m.sib);
            for (v, _) in m.elide {
                plan.elided[v.index()] = true;
            }
        }
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let (Op::Select(cond, x, y), Some(res)) = (&data.op, data.result) else { continue };
            let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
            let Some(tree) = ctx.cond_tree(*cond, pos) else { continue };
            let ([first, second], Location::Reg(dst)) = (&tree.leaves[..], plan.location(res)) else { continue };
            let other = if tree.table.count_ones() == 1 { *y } else { *x };
            let skip: Vec<usize> = tree.nodes.iter().map(|node| node.1).collect();
            let folds = |v: Value| {
                uses[v.index()] == 1
                    && !plan.elided[v.index()]
                    && func.ty(v).is_small_int()
                    && pos_of[v.index()].is_some_and(|(vb, lpos, linst)| {
                        vb == b_idx && matches!(func.insts[linst.index()].op, Op::Load(..) | Op::StackLoad(..)) && load_fuses(&ctx, lpos, pos, &skip)
                    })
            };
            let kept = |leaf: Value, free: bool| {
                pos_of[leaf.index()].is_some_and(|(_, leaf_pos, leaf_inst)| {
                    matches!(func.insts[leaf_inst.index()].op, Op::Icmp(_, a, b) if [a, b].into_iter().all(|v| {
                        (folds(v) || preserved(func, plan, b_idx, leaf_pos, pos, &skip, v)) && (free || !reads_reg(&ctx, v, dst))
                    }))
                })
            };
            if !func.ty(*x).is_small_int() || x == y || plan.reg_of(other) == Some(dst) || !kept(*first, true) || !kept(*second, false) {
                continue;
            }
            for &(v, ..) in &tree.nodes {
                plan.elided[v.index()] = true;
            }
            tst_end[first.index()] = Some(pos);
            tst_end[second.index()] = Some(pos);
            roots[cond.index()] = Some((*first, Some(*second), tree.table));
        }
    }
    for (b_idx, lpos, load_res) in cand_loads {
        let Some((ub, upos, uinst)) = use_of[load_res.index()] else { continue };
        if ub != b_idx || upos <= lpos {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
        let udata = &func.insts[uinst.index()];
        let eff_pos = if let Some(ur) = udata.result.filter(|r| plan.elided[r.index()]) {
            let Some(fpos) = tst_end[ur.index()] else { continue };
            fpos
        } else {
            if !load_elide_ok(&ctx, features, func.blocks[b_idx].insts[lpos], load_res, udata) {
                continue;
            }
            upos
        };
        if load_fuses(&ctx, lpos, eff_pos, &[]) {
            plan.elided[load_res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_alu {
        if plan.elided[res.index()] || plan.deferred[res.index()] || plan.fused[res.index()] || plan.sib[res.index()].is_some() || skip_flags[res.index()] {
            continue;
        }
        if op == Binary::Band && [x, y].into_iter().any(|v| plan.const_of(v).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))) {
            continue;
        }
        let shrink_safe = matches!(plan.location(res), Location::Reg(r) if !SAVED.contains(&r) && !FRAME_REGS.contains(&r))
            || (func.entry != Some(Block(b_idx as u32))
                && !matches!(func.blocks[b_idx].insts.last().map(|i| &func.insts[i.index()].op), Some(Op::Return(_))));
        if !shrink_safe {
            continue;
        }
        let insts = &func.blocks[b_idx].insts;
        for &next_inst in &insts[pos + 1..] {
            let nd = &func.insts[next_inst.index()];
            if nd.result.is_some_and(|r| plan.elided[r.index()]) {
                continue;
            }
            let flagless = nd.result.is_some_and(|r| !func.ty(r).is_vector())
                && matches!(nd.op, Op::Const(_) | Op::StackAddr(..) | Op::FramePointer | Op::Symbol(_) | Op::SlotAddr(_));
            if flagless {
                continue;
            }
            let zf_match = match nd.op {
                Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..) => c == res,
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                    (lhs == res && is_zero_const(func, plan, rhs)) || (rhs == res && is_zero_const(func, plan, lhs))
                }
                _ => false,
            };
            if zf_match {
                plan.fused[res.index()] = true;
            }
            break;
        }
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (cpos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(cmp_res) = data.result else { continue };
            if plan.elided[cmp_res.index()] || plan.fused[cmp_res.index()] || uses[cmp_res.index()] == 0 || non_flag_use[cmp_res.index()] {
                continue;
            }
            if !flag_cmp(func, &data.op) {
                continue;
            }
            let Some((ub, last_pos)) = last_flag_use[cmp_res.index()] else { continue };
            if ub != b_idx || last_pos <= cpos {
                continue;
            }
            if (cpos + 1..=last_pos).all(|pos| mid_preserves_flags(func, plan, &defs, &roots, cmp_res, block.insts[pos], pos == last_pos)) {
                plan.fused[cmp_res.index()] = true;
            } else if let Op::Icmp(_, a, b) = data.op {
                let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
                if !parked(plan, cmp_res) && [a, b].into_iter().all(|v| cmp_operand_ok(&ctx, cpos, last_pos, v)) {
                    plan.elided[cmp_res.index()] = true;
                }
            }
        }
    }
    roots
}
