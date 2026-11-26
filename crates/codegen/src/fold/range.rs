mod bounds;
mod guard;

use super::extend;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

struct RangeState {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    consts: Vec<Option<i64>>,
}

fn def_inst(state: &RangeState, v: Value) -> Option<Inst> {
    state.defs.get(v.index()).copied().flatten()
}

fn const_val(state: &RangeState, v: Value) -> Option<i64> {
    state.consts.get(v.index()).copied().flatten()
}

fn cmp_const(func: &Function, state: &RangeState, v: Value) -> Option<(IntCC, Value, i64)> {
    let Op::Icmp(cc, l, r) = func.insts[def_inst(state, v)?.index()].op else {
        return None;
    };
    if !func.ty(l).is_small_int() {
        return None;
    }
    const_val(state, r)
        .map(|c| (cc, l, c))
        .or_else(|| const_val(state, l).map(|c| (cc.swapped(), r, c)))
}

fn band_args(
    func: &Function,
    state: &RangeState,
    v: Value,
    eq_cc: IntCC,
) -> Option<(Value, Value)> {
    let Op::Icmp(cc, l, r) = func.insts[def_inst(state, v)?.index()].op else {
        return None;
    };
    if cc != eq_cc {
        return None;
    }
    let arg = match (const_val(state, l), const_val(state, r)) {
        (_, Some(0)) => l,
        (Some(0), _) => r,
        _ => return None,
    };
    if state.uses.get(arg.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let Op::Binary(Binary::Band, x, m) = func.insts[def_inst(state, arg)?.index()].op else {
        return None;
    };
    func.ty(x).is_small_int().then_some((x, m))
}

fn shared_mask_pair(a: (Value, Value), b: (Value, Value)) -> Option<(Value, Value, Value)> {
    if a.0 == b.0 {
        Some((a.0, a.1, b.1))
    } else if a.0 == b.1 {
        Some((a.0, a.1, b.0))
    } else if a.1 == b.0 {
        Some((a.1, a.0, b.1))
    } else if a.1 == b.1 {
        Some((a.1, a.0, b.0))
    } else {
        None
    }
}

fn try_fold_mask_zero(
    func: &mut Function,
    inst: Inst,
    state: &mut RangeState,
    next: &mut Vec<Inst>,
) -> bool {
    let Op::Binary(op @ (Binary::Band | Binary::Bor), a, b) = func.insts[inst.index()].op else {
        return false;
    };
    if state.uses[a.index()] != 1 || state.uses[b.index()] != 1 {
        return false;
    }
    let eq_cc = if op == Binary::Band {
        IntCC::Equal
    } else {
        IntCC::NotEqual
    };
    let (Some(p1), Some(p2)) = (
        band_args(func, state, a, eq_cc),
        band_args(func, state, b, eq_cc),
    ) else {
        return false;
    };
    let Some((x, m1, m2)) = shared_mask_pair(p1, p2) else {
        return false;
    };
    let ty = func.ty(x);
    let mask = match (const_val(state, m1), const_val(state, m2)) {
        (Some(c1), Some(c2)) => {
            let mv = func.value(ty);
            let imm = c1 | c2;
            state.consts.push(Some(extend(imm, ty, false)));
            next.push(func.add(Op::Const(imm), Some(mv)));
            mv
        }
        _ => {
            let mv = func.value(ty);
            state.consts.push(None);
            next.push(func.add(Op::Binary(Binary::Bor, m1, m2), Some(mv)));
            mv
        }
    };
    let masked = func.value(ty);
    let zero = func.value(ty);
    state.consts.extend([None, Some(0)]);
    next.push(func.add(Op::Binary(Binary::Band, x, mask), Some(masked)));
    next.push(func.add(Op::Const(0), Some(zero)));
    func.insts[inst.index()].op = Op::Icmp(eq_cc, masked, zero);
    true
}

type FusedCmp = (Option<(Binary, i64)>, IntCC, i64);

fn fuse_eq_pair(is_and: bool, ty: Type, u1: u64, u2: u64) -> Option<FusedCmp> {
    let eq_cc = if is_and {
        IntCC::NotEqual
    } else {
        IntCC::Equal
    };
    let out_cc = if is_and {
        IntCC::UnsignedGreaterThan
    } else {
        IntCC::UnsignedLessThanOrEqual
    };
    let (lo, hi) = if u1 <= u2 { (u1, u2) } else { (u2, u1) };
    if lo == hi {
        return Some((None, eq_cc, lo as i64));
    }
    let umax = ty.unsigned_max();
    if umax >= 2 && lo == 0 && hi == 1 {
        return Some((None, out_cc, 1));
    }
    if umax >= 2 && hi - lo == 1 {
        return Some((Some((Binary::Isub, lo as i64)), out_cc, 1));
    }
    if umax >= 2 && lo == 0 && hi == umax {
        return Some((Some((Binary::Iadd, 1)), out_cc, 1));
    }
    let diff = hi - lo;
    (diff.is_power_of_two() && (lo & diff) == 0).then_some((
        Some((Binary::Bor, diff as i64)),
        eq_cc,
        hi as i64,
    ))
}

fn unsigned_lo(is_and: bool, ty: Type, side: (IntCC, i64)) -> Option<u64> {
    let umax = ty.unsigned_max();
    match (is_and, side.0, extend(side.1, ty, false) as u64) {
        (true, IntCC::UnsignedGreaterThanOrEqual, u) | (false, IntCC::UnsignedLessThan, u) => {
            Some(u)
        }
        (true, IntCC::UnsignedGreaterThan, u) | (false, IntCC::UnsignedLessThanOrEqual, u)
            if u < umax =>
        {
            Some(u + 1)
        }
        _ => None,
    }
}

fn unsigned_hi(is_and: bool, ty: Type, side: (IntCC, i64)) -> Option<u64> {
    match (is_and, side.0, extend(side.1, ty, false) as u64) {
        (true, IntCC::UnsignedLessThanOrEqual, u) | (false, IntCC::UnsignedGreaterThan, u) => {
            Some(u)
        }
        (true, IntCC::UnsignedLessThan, u) | (false, IntCC::UnsignedGreaterThanOrEqual, u)
            if u > 0 =>
        {
            Some(u - 1)
        }
        _ => None,
    }
}

fn fuse_unsigned_pair(
    is_and: bool,
    ty: Type,
    l: (IntCC, i64),
    h: (IntCC, i64),
) -> Option<FusedCmp> {
    let u_lo = unsigned_lo(is_and, ty, l)?;
    let u_hi = unsigned_hi(is_and, ty, h)?;
    if u_lo > u_hi || u_hi - u_lo >= ty.unsigned_max() {
        return None;
    }
    if u_lo == u_hi {
        let eq_cc = if is_and {
            IntCC::Equal
        } else {
            IntCC::NotEqual
        };
        return Some((None, eq_cc, u_lo as i64));
    }
    let out_cc = if is_and {
        IntCC::UnsignedLessThanOrEqual
    } else {
        IntCC::UnsignedGreaterThan
    };
    Some((
        (u_lo != 0).then_some((Binary::Isub, u_lo as i64)),
        out_cc,
        (u_hi - u_lo) as i64,
    ))
}

fn fuse_signed_pair(is_and: bool, ty: Type, l: (IntCC, i64), h: (IntCC, i64)) -> Option<FusedCmp> {
    let umax = ty.unsigned_max();
    let (smin, smax) = if ty.bits() == 64 {
        (i64::MIN, i64::MAX)
    } else {
        (-(1_i64 << (ty.bits() - 1)), (1_i64 << (ty.bits() - 1)) - 1)
    };
    let s_lo = match (is_and, l.0, extend(l.1, ty, true)) {
        (true, IntCC::SignedGreaterThanOrEqual, s) | (false, IntCC::SignedLessThan, s) => Some(s),
        (true, IntCC::SignedGreaterThan, s) | (false, IntCC::SignedLessThanOrEqual, s)
            if s < smax =>
        {
            Some(s + 1)
        }
        _ => None,
    }?;
    let s_hi = match (is_and, h.0, extend(h.1, ty, true)) {
        (true, IntCC::SignedLessThanOrEqual, s) | (false, IntCC::SignedGreaterThan, s) => Some(s),
        (true, IntCC::SignedLessThan, s) | (false, IntCC::SignedGreaterThanOrEqual, s)
            if s > smin =>
        {
            Some(s - 1)
        }
        _ => None,
    }?;
    if s_lo > s_hi {
        return None;
    }
    let (span, lo_u) = (
        (s_hi as u64).wrapping_sub(s_lo as u64) & umax,
        (s_lo as u64) & umax,
    );
    if span >= umax {
        return None;
    }
    if span == 0 {
        let eq_cc = if is_and {
            IntCC::Equal
        } else {
            IntCC::NotEqual
        };
        return Some((None, eq_cc, lo_u as i64));
    }
    let out_cc = if is_and {
        IntCC::UnsignedLessThanOrEqual
    } else {
        IntCC::UnsignedGreaterThan
    };
    Some((
        (lo_u != 0).then_some((Binary::Isub, lo_u as i64)),
        out_cc,
        span as i64,
    ))
}

fn signed_zero_half(is_and: bool, ty: Type, s_side: (IntCC, i64)) -> Option<bool> {
    match (is_and, s_side.0, extend(s_side.1, ty, true)) {
        (true, IntCC::SignedGreaterThanOrEqual, 0)
        | (true, IntCC::SignedGreaterThan, -1)
        | (false, IntCC::SignedLessThan, 0)
        | (false, IntCC::SignedLessThanOrEqual, -1) => Some(true),
        (true, IntCC::SignedLessThan, 0)
        | (true, IntCC::SignedLessThanOrEqual, -1)
        | (false, IntCC::SignedGreaterThanOrEqual, 0)
        | (false, IntCC::SignedGreaterThan, -1) => Some(false),
        _ => None,
    }
}

fn fuse_mixed_sign_pair(
    is_and: bool,
    ty: Type,
    s_side: (IntCC, i64),
    u_side: (IntCC, i64),
) -> Option<FusedCmp> {
    let umax = ty.unsigned_max();
    if umax < 2 {
        return None;
    }
    let smax = umax >> 1;
    if signed_zero_half(is_and, ty, s_side)? {
        let u_hi = unsigned_hi(is_and, ty, u_side)?;
        let out_cc = if is_and {
            IntCC::UnsignedLessThanOrEqual
        } else {
            IntCC::UnsignedGreaterThan
        };
        Some((None, out_cc, u_hi.min(smax) as i64))
    } else {
        let u_lo = unsigned_lo(is_and, ty, u_side)?;
        let out_cc = if is_and {
            IntCC::UnsignedGreaterThanOrEqual
        } else {
            IntCC::UnsignedLessThan
        };
        Some((None, out_cc, u_lo.max(smax + 1) as i64))
    }
}

fn fuse_cmp_pair(is_and: bool, ty: Type, p1: (IntCC, i64), p2: (IntCC, i64)) -> Option<FusedCmp> {
    let eq_cc = if is_and {
        IntCC::NotEqual
    } else {
        IntCC::Equal
    };
    if p1.0 == eq_cc && p2.0 == eq_cc {
        return fuse_eq_pair(
            is_and,
            ty,
            extend(p1.1, ty, false) as u64,
            extend(p2.1, ty, false) as u64,
        );
    }
    for (l, h) in [(p1, p2), (p2, p1)] {
        if let Some(res) = fuse_unsigned_pair(is_and, ty, l, h)
            .or_else(|| fuse_signed_pair(is_and, ty, l, h))
            .or_else(|| fuse_mixed_sign_pair(is_and, ty, l, h))
        {
            return Some(res);
        }
    }
    None
}

fn has_candidate(func: &Function) -> bool {
    let mut is_cmp = vec![false; func.types.len()];
    for data in func
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .map(|i| &func.insts[i.index()])
    {
        if let Op::Convert(Convert::Sextend, inner) = data.op
            && func.ty(inner).is_small_int()
            && func.ty(inner) != Type::I64
        {
            return true;
        }
        if let (&Op::Icmp(_, l, _), Some(res)) = (&data.op, data.result)
            && func.ty(l).is_small_int()
        {
            is_cmp[res.index()] = true;
        }
    }
    func.blocks.iter().flat_map(|b| &b.insts).any(|i| {
        matches!(
            func.insts[i.index()].op,
            Op::Binary(Binary::Band | Binary::Bor, a, b) if is_cmp[a.index()] && is_cmp[b.index()]
        )
    })
}

fn try_fold_cmp_pair(
    func: &mut Function,
    inst: Inst,
    state: &mut RangeState,
    next: &mut Vec<Inst>,
) -> bool {
    let Op::Binary(op @ (Binary::Band | Binary::Bor), a, b) = func.insts[inst.index()].op else {
        return false;
    };
    if state.uses[a.index()] != 1 || state.uses[b.index()] != 1 {
        return false;
    }
    let (Some((cc1, x1, c1)), Some((cc2, x2, c2))) =
        (cmp_const(func, state, a), cmp_const(func, state, b))
    else {
        return false;
    };
    if x1 != x2 {
        return false;
    }
    let x_ty = func.ty(x1);
    let Some((pre, out_cc, out_imm)) =
        fuse_cmp_pair(op == Binary::Band, x_ty, (cc1, c1), (cc2, c2))
    else {
        return false;
    };
    let lhs = if let Some((pre_op, pre_imm)) = pre {
        let (c_pre, r_pre) = (func.value(x_ty), func.value(x_ty));
        state
            .consts
            .extend([Some(extend(pre_imm, x_ty, false)), None]);
        next.push(func.add(Op::Const(pre_imm), Some(c_pre)));
        next.push(func.add(Op::Binary(pre_op, x1, c_pre), Some(r_pre)));
        r_pre
    } else {
        x1
    };
    let c_out = func.value(x_ty);
    state.consts.push(Some(extend(out_imm, x_ty, false)));
    next.push(func.add(Op::Const(out_imm), Some(c_out)));
    func.insts[inst.index()].op = Op::Icmp(out_cc, lhs, c_out);
    true
}

/// Fuses each `and` or `or` of two single-use compares of one value against constants or masks into one compare, and narrows provably non-negative `Sextend` to `Uextend`.
pub(super) fn fold_range_checks(func: &mut Function) -> bool {
    if !has_candidate(func) {
        return false;
    }
    let mut consts = vec![None; func.types.len()];
    let mut def_tab = vec![None; func.types.len()];
    let mut uses = vec![0_u32; func.types.len()];
    for &i in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[i.index()];
        data.op.for_operands(|v| uses[v.index()] += 1);
        if let Some(res) = data.result {
            def_tab[res.index()] = Some(i);
            if let Op::Const(imm) = data.op
                && func.ty(res).is_small_int()
            {
                consts[res.index()] = Some(extend(imm, func.ty(res), false));
            }
        }
    }
    let mut state = RangeState {
        defs: def_tab,
        uses,
        consts,
    };
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        let orig = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut next = Vec::with_capacity(orig.len());
        for inst in orig {
            if try_fold_cmp_pair(func, inst, &mut state, &mut next)
                || try_fold_mask_zero(func, inst, &mut state, &mut next)
                || bounds::try_fold_sext_to_uext(func, inst, &state)
            {
                changed = true;
            }
            next.push(inst);
        }
        func.blocks[b_idx].insts = next;
    }
    changed |= guard::fold_guarded_sext(func, &mut state);
    changed
}
