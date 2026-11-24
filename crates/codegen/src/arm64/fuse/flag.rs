use super::super::asm::Reg;
use super::super::int::add_imm_parts;
use super::super::Location;
use super::bits::can_elide_tst_band;
use super::{
    alu_shift, cmn_parts, cmn_sum_parts, const_of, def_entry, def_inst, ext_band_of, ext_parts, local_single, preserved,
    shift_parts, unwrap_elided_tst, zero_cmp_band,
};
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

fn peel_uext(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], val: Value) -> Value {
    if plan.elided[val.index()]
        && let Some((_, _, inst)) = def_entry(pos_of, val)
        && let Op::Convert(Convert::Uextend, inner) = func.insts[inst.index()].op
    {
        return inner;
    }
    val
}

fn cmp_shift_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cmp_ty, v, other): (Type, Value, Value),
) -> bool {
    if !matches!(cmp_ty, Type::I32 | Type::I64) || !plan.elided[v.index()] {
        return false;
    }
    let Some((_, _, inst)) = def_entry(pos_of, v) else {
        return false;
    };
    let Some((sop, mid, imm)) = shift_parts(&func.insts[inst.index()].op, plan) else {
        return false;
    };
    if alu_shift(false, cmp_ty, sop, imm).is_none() {
        return false;
    }
    let root = peel_uext(func, plan, pos_of, mid);
    let other_root = peel_uext(func, plan, pos_of, other);
    preserved(func, plan, b_idx, start, pos, skip, root) && preserved(func, plan, b_idx, start, pos, skip, other_root)
}

fn cmp_operand_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cmp_ty, cc, v): (Type, IntCC, Value),
) -> bool {
    if !plan.elided[v.index()] {
        return preserved(func, plan, b_idx, start, pos, skip, v);
    }
    let def = |val| def_inst(pos_of, val);
    if let Some((_, _, inner)) = ext_parts(func, plan, cmp_ty, v, def) {
        return preserved(func, plan, b_idx, start, pos, skip, inner);
    }
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && matches!(cmp_ty, Type::I32 | Type::I64)
        && let Some((root, _)) = cmn_parts(func, plan, cmp_ty, v, &def)
    {
        return preserved(func, plan, b_idx, start, pos, skip, root);
    }
    false
}

fn tst_single_insn(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], (x, y): (Value, Value)) -> bool {
    let def = |v| def_inst(pos_of, v);
    let ty = func.ty(x);
    let ty_mask = ty.unsigned_max();
    for (raw, mask) in [(x, y), (y, x)] {
        if let Some(imm) = const_of(func, plan, mask, &def) {
            return unwrap_elided_tst(func, plan, raw, (imm as u64) & ty_mask, &def)
                .is_some_and(|(_, eff)| super::super::int::encode_bitmask_imm(eff, ty == Type::I64).is_some());
        }
    }
    true
}

fn cmp_band_tst(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cc, a, b, cmp_uses): (IntCC, Value, Value, u32),
) -> Option<Value> {
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual) {
        return None;
    }
    let (m, mpos, bx, by) = zero_cmp_band(func, plan, uses, pos_of, b_idx, a, b)?;
    let ok = mpos < start
        && can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, pos, skip), (bx, by))
        && (cmp_uses == 1 || tst_single_insn(func, plan, pos_of, (bx, by)));
    ok.then_some(m)
}

pub(super) fn try_cmp_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    span @ (b_idx, start, pos, skip): (usize, usize, usize, Option<usize>),
    (cc, a, b, cmp_uses): (IntCC, Value, Value, u32),
) -> Option<Option<Value>> {
    let cmp_ty = func.ty(a);
    if !cmp_ty.is_small_int() {
        return None;
    }
    let def = |val| def_inst(pos_of, val);
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some((sa, sb)) = cmn_sum_parts(func, plan, (a, b), &def)
        && preserved(func, plan, b_idx, start, pos, skip, sa)
        && preserved(func, plan, b_idx, start, pos, skip, sb)
    {
        return Some(None);
    }
    if !(plan.elided[a.index()] && plan.elided[b.index()])
        && cmp_operand_preserved(func, plan, pos_of, span, (cmp_ty, cc, a))
        && cmp_operand_preserved(func, plan, pos_of, span, (cmp_ty, cc, b))
    {
        return Some(None);
    }
    if cmp_shift_preserved(func, plan, pos_of, span, (cmp_ty, b, a)) || cmp_shift_preserved(func, plan, pos_of, span, (cmp_ty, a, b)) {
        return Some(None);
    }
    cmp_band_tst(func, plan, info, span, (cc, a, b, cmp_uses)).map(Some)
}

pub(super) fn try_fuse_flag_operand(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    flag_val: Value,
) -> bool {
    let Some((fpos, finst)) = local_single(plan, info, b_idx, pos, flag_val) else {
        return false;
    };
    match func.insts[finst.index()].op {
        Op::Icmp(cc, a, b) => {
            let Some(band) = try_cmp_preserved(func, plan, info, (b_idx, fpos, pos, None), (cc, a, b, 1)) else {
                return false;
            };
            plan.elided[flag_val.index()] = true;
            if let Some(m) = band {
                plan.elided[m.index()] = true;
            }
            true
        }
        Op::Convert(Convert::Uextend, inner)
            if let Some((ipos, iinst)) = local_single(plan, info, b_idx, fpos, inner)
                && let Op::Icmp(cc, a, b) = func.insts[iinst.index()].op
                && let Some(band) = try_cmp_preserved(func, plan, info, (b_idx, ipos, pos, Some(fpos)), (cc, a, b, 1)) =>
        {
            plan.elided[flag_val.index()] = true;
            plan.elided[inner.index()] = true;
            if let Some(m) = band {
                plan.elided[m.index()] = true;
            }
            true
        }
        _ => false,
    }
}

fn flag_ok(plan: &Plan<Reg>, val: Value) -> bool {
    !plan.elided[val.index()] && !plan.fused[val.index()] && !plan.is_static(val)
}

fn icmp_select_root(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    (b_idx, ipos, icmp_res): (usize, usize, Value),
) -> Option<(Value, usize, Option<usize>)> {
    if !flag_ok(plan, icmp_res) {
        return None;
    }
    if uses[icmp_res.index()] != 1 {
        return Some((icmp_res, ipos, None));
    }
    for (u_pos, &ni) in func.blocks[b_idx].insts.iter().enumerate().skip(ipos + 1) {
        let ud = &func.insts[ni.index()];
        if let Op::Convert(Convert::Uextend, inner) = ud.op
            && inner == icmp_res
            && let Some(uext) = ud.result
            && flag_ok(plan, uext)
            && func.ty(uext).is_small_int()
        {
            return Some((uext, u_pos, Some(u_pos)));
        }
        if matches!(ud.op, Op::Select(c, ..) if c == icmp_res) {
            return Some((icmp_res, ipos, None));
        }
    }
    None
}

fn find_select_consumers(
    func: &Function,
    plan: &Plan<Reg>,
    (b_idx, start_pos): (usize, usize),
    flag_val: Value,
    total_uses: u32,
) -> Option<(u32, usize, bool)> {
    let insts = &func.blocks[b_idx].insts;
    let (mut sels, mut brif, mut last_sel) = (0_u32, None, start_pos);
    for (s_pos, &ni) in insts.iter().enumerate().skip(start_pos + 1) {
        match &func.insts[ni.index()].op {
            Op::Select(c, x, y) if *c == flag_val && *x != flag_val && *y != flag_val => {
                sels += 1;
                last_sel = s_pos;
                if sels == total_uses {
                    return Some((sels, last_sel, false));
                }
            }
            Op::Brif(c, _, ta, _, ea)
                if sels == 1 && total_uses == 2 && *c == flag_val && !ta.contains(&flag_val) && !ea.contains(&flag_val) =>
            {
                brif = Some(s_pos);
                break;
            }
            _ => {}
        }
    }
    let b_pos = brif?;
    insts[last_sel + 1..b_pos]
        .iter()
        .all(|&ni| {
            let nd = &func.insts[ni.index()];
            (!matches!(nd.op, Op::Call { .. }) && nd.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]))
                || preserves_flags(func, plan, &nd.op, nd.result, false)
        })
        .then_some((1, last_sel, true))
}

fn brif_band_conflict(plan: &Plan<Reg>, cc: IntCC, (a, b): (Value, Value)) -> bool {
    matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && ((plan.const_of(b) == Some(0) && plan.elided[a.index()]) || (plan.const_of(a) == Some(0) && plan.elided[b.index()]))
}

fn root_preserved_around(
    func: &Function,
    plan: &Plan<Reg>,
    (b_idx, upos, pos, last_sel, skip_pos): (usize, usize, usize, usize, Option<usize>),
    val: Value,
) -> bool {
    preserved(func, plan, b_idx, upos, pos, None, val) && preserved(func, plan, b_idx, pos, last_sel, skip_pos, val)
}

fn shift_root_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    span @ (b_idx, upos, pos, last_sel, skip_pos): (usize, usize, usize, usize, Option<usize>),
    (ty, op): (Type, &Op),
) -> bool {
    let Some((sop, mid, imm)) = shift_parts(op, plan) else { return false };
    if alu_shift(false, ty, sop, imm).is_none() {
        return false;
    }
    if root_preserved_around(func, plan, span, mid) {
        return true;
    }
    let Some((mpos, minst)) = local_single(plan, info, b_idx, upos, mid) else { return false };
    let Op::Convert(Convert::Uextend, inner) = func.insts[minst.index()].op else { return false };
    let ty_bits = ty.bits();
    let amt = (imm as u32) & (ty_bits - 1);
    let uext_ok = func.ty(inner).is_small_int()
        && (super::zero_ext_loc(plan, inner)
            || (sop == Binary::Ishl && (amt >= ty_bits - func.ty(inner).bits() || amt <= 4)));
    uext_ok
        && preserved(func, plan, b_idx, mpos, pos, Some(upos), inner)
        && preserved(func, plan, b_idx, pos, last_sel, skip_pos, inner)
}

fn sum_fusible_to_select(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, last_sel, skip_pos): (usize, usize, usize, Option<usize>),
    (a, b): (Value, Value),
) -> bool {
    let def = |v| def_inst(pos_of, v);
    let Some(sum) = (const_of(func, plan, b, &def) == Some(0))
        .then_some(a)
        .or_else(|| (const_of(func, plan, a, &def) == Some(0)).then_some(b))
    else {
        return false;
    };
    let Some((spos, sinst)) = local_single(plan, info, b_idx, pos, sum) else { return false };
    let Op::Binary(Binary::Iadd, sa, sb) = func.insts[sinst.index()].op else { return false };
    let span = (b_idx, spos, pos, last_sel, skip_pos);
    root_preserved_around(func, plan, span, sa) && root_preserved_around(func, plan, span, sb)
}

fn cand_fusible_to_select(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, last_sel, skip_pos): (usize, usize, usize, Option<usize>),
    (ty, cc, cand): (Type, IntCC, Value),
) -> bool {
    let Some((cpos, cinst)) = local_single(plan, info, b_idx, pos, cand) else { return false };
    let op = &func.insts[cinst.index()].op;
    let span = (b_idx, cpos, pos, last_sel, skip_pos);
    if shift_root_preserved(func, plan, info, span, (ty, op)) {
        return true;
    }
    let def = |v| def_inst(pos_of, v);
    let ext_root = match *op {
        Op::Convert(Convert::Uextend | Convert::Sextend, inner) if func.ty(inner).bits() < ty.bits() => Some(inner),
        Op::Binary(Binary::Band, a, b) => ext_band_of(func, plan, ty, (a, b), &def).map(|(_, inner)| inner),
        Op::Unary(Unary::Ineg, inner) if matches!(cc, IntCC::Equal | IntCC::NotEqual) => Some(inner),
        _ => None,
    };
    ext_root.is_some_and(|root| !plan.elided[root.index()] && root_preserved_around(func, plan, span, root))
}

fn should_defer_icmp_select(
    func: &Function,
    plan: &Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    span @ (b_idx, pos, last_sel, skip_pos): (usize, usize, usize, Option<usize>),
    (cc, a, b): (IntCC, Value, Value),
) -> bool {
    let ty = func.ty(a);
    if !matches!(ty, Type::I32 | Type::I64) || plan.elided[a.index()] || plan.elided[b.index()] {
        return false;
    }
    if matches!(cc, IntCC::Equal | IntCC::NotEqual) && sum_fusible_to_select(func, plan, info, span, (a, b)) {
        return true;
    }
    let has_imm = |v: Value| plan.const_of(v).and_then(|imm| add_imm_parts(true, crate::fold::extend(imm, ty, cc.is_signed()))).is_some();
    !has_imm(a)
        && !has_imm(b)
        && (cand_fusible_to_select(func, plan, info, (b_idx, pos, last_sel, skip_pos), (ty, cc, b))
            || cand_fusible_to_select(func, plan, info, (b_idx, pos, last_sel, skip_pos), (ty, cc, a)))
}

fn try_fuse_icmp_select_with(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, _): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (a, b, allow_defer): (Value, Value, bool),
) {
    let Some((flag_val, search_pos, skip_pos)) = icmp_select_root(func, plan, uses, (b_idx, pos, res)) else {
        return;
    };
    let flag_uses = uses[flag_val.index()];
    if !(1..=2).contains(&flag_uses) {
        return;
    }
    let Op::Icmp(cc, _, _) = func.insts[func.blocks[b_idx].insts[pos].index()].op else {
        return;
    };
    let Some((cmp_count, last_sel, has_brif)) = find_select_consumers(func, plan, (b_idx, search_pos), flag_val, flag_uses) else {
        return;
    };
    if allow_defer && should_defer_icmp_select(func, plan, info, (b_idx, pos, last_sel, skip_pos), (cc, a, b)) {
        return;
    }
    if has_brif && skip_pos.is_none() && brif_band_conflict(plan, cc, (a, b)) {
        return;
    }
    if let Some(band) = try_cmp_preserved(func, plan, info, (b_idx, pos, last_sel, skip_pos), (cc, a, b, cmp_count))
        && (!has_brif || skip_pos.is_some() || band.is_none())
    {
        plan.elided[res.index()] = true;
        plan.elided[flag_val.index()] = true;
        if let Some(m) = band {
            plan.elided[m.index()] = true;
        }
    }
}

pub(super) fn try_fuse_icmp_select(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (a, b): (Value, Value),
) {
    try_fuse_icmp_select_with(func, plan, info, (b_idx, pos, res), (a, b, true));
}

pub(super) fn fuse_icmp_selects(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>])) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && !plan.elided[res.index()]
                && let Op::Icmp(_, a, b) = data.op
                && func.ty(a).is_small_int()
            {
                try_fuse_icmp_select_with(func, plan, info, (b_idx, pos, res), (a, b, false));
            }
        }
    }
}

fn fits_consumer(func: &Function, plan: &Plan<Reg>, tail: &[Inst], res: Value) -> bool {
    for &ni in tail {
        let nd = &func.insts[ni.index()];
        if !matches!(nd.op, Op::Call { .. }) && nd.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if consumes_flag_only(&nd.op, res) {
            return true;
        }
        if !preserves_flags(func, plan, &nd.op, nd.result, false) {
            return false;
        }
    }
    false
}

pub(super) fn fuse_fits(func: &Function, plan: &mut Plan<Reg>, uses: &[u32]) {
    for block in &func.blocks {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let Op::Fits(_, _, a, _) = data.op else { continue };
            if uses[res.index()] == 1
                && !plan.elided[res.index()]
                && !plan.fused[res.index()]
                && matches!(plan.location(res), Location::Reg(_) | Location::Slot)
                && func.ty(a).is_small_int()
                && fits_consumer(func, plan, &block.insts[pos + 1..], res)
            {
                plan.fused[res.index()] = true;
            }
        }
    }
}

pub(super) fn consumes_flag_only(op: &Op, res: Value) -> bool {
    match op {
        Op::Select(c, x, y) => *c == res && *x != res && *y != res,
        Op::Brif(c, _, ta, _, ea) => *c == res && !ta.contains(&res) && !ea.contains(&res),
        Op::Trapz(c) => *c == res,
        _ => false,
    }
}

/// Whether emitting `op` leaves ARM64 NZCV flags untouched.
pub(in super::super) fn preserves_flags(func: &Function, plan: &Plan<Reg>, op: &Op, result: Option<Value>, allow_i8_bitwise: bool) -> bool {
    matches!(
        *op,
        Op::Load(..) | Op::Store(..) | Op::StackAddr(..) | Op::StackLoad(..) | Op::StackStore(..)
            | Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_) | Op::FramePointer
    ) || match *op {
        Op::Binary(b, x, y) => !matches!(func.ty(x), Type::I128) && !func.ty(x).is_vector() && match b {
            Binary::Iadd | Binary::Isub | Binary::Imul | Binary::Bxor | Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr => true,
            Binary::Band | Binary::Bor => {
                (allow_i8_bitwise || func.ty(x) != Type::I8)
                    && !plan.elided[x.index()]
                    && !plan.elided[y.index()]
                    && result.is_none_or(|r| !plan.fused[r.index()])
            }
            _ => false,
        },
        Op::Unary(u, x) => !matches!(func.ty(x), Type::I128) && !func.ty(x).is_vector()
            && matches!(u, Unary::Ineg | Unary::Bnot | Unary::Clz | Unary::Ctz | Unary::Popcnt | Unary::Bswap),
        Op::Convert(c, x) => !matches!(func.ty(x), Type::I128) && !func.ty(x).is_vector()
            && result.is_some_and(|r| !func.ty(r).is_vector() && !func.ty(r).is_float())
            && matches!(c, Convert::Uextend | Convert::Sextend | Convert::Ireduce | Convert::Bitcast),
        _ => false,
    }
}

fn flag_consumer(func: &Function, plan: &Plan<Reg>, tail: &[Inst], res: Value, allow_clobber_branch: bool) -> Option<(usize, bool, bool)> {
    let mut flags_ok = true;
    let capped = func.size() > 2000;
    for (off, &ni) in tail.iter().enumerate() {
        let nd = &func.insts[ni.index()];
        if !matches!(nd.op, Op::Call { .. }) && nd.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if consumes_flag_only(&nd.op, res) {
            let is_branch = !matches!(nd.op, Op::Select(..));
            return (flags_ok || (allow_clobber_branch && is_branch)).then_some((off, is_branch, flags_ok));
        }
        if !preserves_flags(func, plan, &nd.op, nd.result, false) {
            if !allow_clobber_branch {
                return None;
            }
            flags_ok = false;
        }
        if !flags_ok && capped && off >= 64 {
            return None;
        }
    }
    None
}

fn multi_flag_consumers(func: &Function, plan: &Plan<Reg>, tail: &[Inst], res: Value, total_uses: u32) -> bool {
    let mut found = 0_u32;
    for &ni in tail {
        let nd = &func.insts[ni.index()];
        if !matches!(nd.op, Op::Call { .. }) && nd.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]) {
            continue;
        }
        if consumes_flag_only(&nd.op, res) {
            found += 1;
            if found == total_uses {
                return true;
            }
            continue;
        }
        if !preserves_flags(func, plan, &nd.op, nd.result, false) {
            return false;
        }
    }
    false
}

pub(super) fn icmp_consumer(func: &Function, plan: &Plan<Reg>, tail: &[Inst], res: Value) -> bool {
    flag_consumer(func, plan, tail, res, false).is_some()
}

pub(in super::super) fn zero_cmp_operand(plan: &Plan<Reg>, ty: Type, a: Value, b: Value) -> Option<(Value, bool)> {
    if !ty.is_small_int() {
        return None;
    }
    let is_imm = |v: Value, k: i64| plan.const_of(v).is_some_and(|imm| crate::fold::extend(imm, ty, false) == k);
    let (val, one) = if is_imm(b, 0) {
        (a, false)
    } else if is_imm(a, 0) {
        (b, false)
    } else if is_imm(b, 1) && plan.clean_bits[a.index()] <= 1 {
        (a, true)
    } else if is_imm(a, 1) && plan.clean_bits[b.index()] <= 1 {
        (b, true)
    } else {
        return None;
    };
    (!plan.elided[val.index()] && plan.const_of(val).is_none()).then_some((val, one))
}

fn can_elide_cbz(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, c_pos): (usize, usize, usize),
    val: Value,
    one: bool,
) -> bool {
    if !preserved(func, plan, b_idx, pos, c_pos, None, val) {
        return false;
    }
    if !one
        && uses[val.index()] == 1
        && let Some((vb, vpos, vinst)) = def_entry(pos_of, val)
        && vb == b_idx
        && let Op::Binary(Binary::Band, bx, by) = func.insts[vinst.index()].op
        && (super::bits::single_bit_raw(func, plan, bx, by).is_some()
            || can_elide_tst_band(func, plan, pos_of, (b_idx, vpos, c_pos, Some(pos)), (bx, by)))
    {
        return false;
    }
    true
}

pub(super) fn fuse_icmp_flags(func: &Function, plan: &mut Plan<Reg>, uses: &[u32]) {
    let mut pos_of = vec![None; func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            pos_of[res.index()] = Some((b_idx, pos, inst));
            let Op::Icmp(cc, a, b) = data.op else { continue };
            if plan.elided[res.index()] || plan.is_static(res) || func.ty(a).is_vector() {
                continue;
            }
            let use_count = uses[res.index()];
            if use_count > 1 {
                if !plan.fused[res.index()] && multi_flag_consumers(func, plan, &block.insts[pos + 1..], res, use_count) {
                    plan.fused[res.index()] = true;
                }
                continue;
            }
            if use_count != 1 {
                continue;
            }
            let was_fused = plan.fused[res.index()];
            let zero_val = matches!(cc, IntCC::Equal | IntCC::NotEqual)
                .then(|| zero_cmp_operand(plan, func.ty(a), a, b))
                .flatten();
            if was_fused && zero_val.is_none() {
                continue;
            }
            let Some((off, is_branch, flags_ok)) = flag_consumer(func, plan, &block.insts[pos + 1..], res, zero_val.is_some()) else {
                continue;
            };
            if is_branch
                && let Some((val, one)) = zero_val
                && can_elide_cbz(func, plan, (uses, &pos_of), (b_idx, pos, pos + 1 + off), val, one)
            {
                plan.fused[res.index()] = false;
                plan.elided[res.index()] = true;
            } else if !was_fused && flags_ok {
                plan.fused[res.index()] = true;
            }
        }
    }
}
