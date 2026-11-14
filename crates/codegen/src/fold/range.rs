use super::{defs, extend};
use crate::ir::{Binary, Function, Inst, IntCC, Op, Type, Value};

fn cmp_const(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], v: Value) -> Option<(IntCC, Value, i64)> {
    let Op::Icmp(cc, l, r) = func.insts[defs.get(v.index()).copied().flatten()?.index()].op else { return None };
    if !func.ty(l).is_small_int() { return None; }
    consts.get(r.index()).copied().flatten().map(|c| (cc, l, c)).or_else(|| consts.get(l.index()).copied().flatten().map(|c| (crate::cse::flip_int_cc(cc), r, c)))
}

fn fuse_cmp_pair(is_and: bool, ty: Type, (cc1, c1): (IntCC, i64), (cc2, c2): (IntCC, i64)) -> Option<(Option<(Binary, i64)>, IntCC, i64)> {
    let umax = ty.unsigned_max();
    let eq_cc = if is_and { IntCC::NotEqual } else { IntCC::Equal };
    if cc1 == eq_cc && cc2 == eq_cc {
        let (u1, u2) = (extend(c1, ty, false) as u64, extend(c2, ty, false) as u64);
        let (lo, hi) = if u1 <= u2 { (u1, u2) } else { (u2, u1) };
        let diff = hi - lo;
        return (diff != 0 && diff.is_power_of_two() && (lo & diff) == 0).then_some((Some((Binary::Bor, diff as i64)), eq_cc, hi as i64));
    }
    let out_cc = if is_and { IntCC::UnsignedLessThanOrEqual } else { IntCC::UnsignedGreaterThan };
    let (smin, smax) = if ty.bits() == 64 { (i64::MIN, i64::MAX) } else { (-(1_i64 << (ty.bits() - 1)), (1_i64 << (ty.bits() - 1)) - 1) };
    for ((l_cc, lc), (h_cc, hc)) in [((cc1, c1), (cc2, c2)), ((cc2, c2), (cc1, c1))] {
        let u_lo = match (is_and, l_cc, extend(lc, ty, false) as u64) {
            (true, IntCC::UnsignedGreaterThanOrEqual, u) | (false, IntCC::UnsignedLessThan, u) => Some(u),
            (true, IntCC::UnsignedGreaterThan, u) | (false, IntCC::UnsignedLessThanOrEqual, u) if u < umax => Some(u + 1),
            _ => None,
        };
        let u_hi = match (is_and, h_cc, extend(hc, ty, false) as u64) {
            (true, IntCC::UnsignedLessThanOrEqual, u) | (false, IntCC::UnsignedGreaterThan, u) => Some(u),
            (true, IntCC::UnsignedLessThan, u) | (false, IntCC::UnsignedGreaterThanOrEqual, u) if u > 0 => Some(u - 1),
            _ => None,
        };
        if let (Some(lo), Some(hi)) = (u_lo, u_hi) && lo <= hi && hi - lo < umax {
            return Some(((lo != 0).then_some((Binary::Isub, lo as i64)), out_cc, (hi - lo) as i64));
        }
        let s_lo = match (is_and, l_cc, extend(lc, ty, true)) {
            (true, IntCC::SignedGreaterThanOrEqual, s) | (false, IntCC::SignedLessThan, s) => Some(s),
            (true, IntCC::SignedGreaterThan, s) | (false, IntCC::SignedLessThanOrEqual, s) if s < smax => Some(s + 1),
            _ => None,
        };
        let s_hi = match (is_and, h_cc, extend(hc, ty, true)) {
            (true, IntCC::SignedLessThanOrEqual, s) | (false, IntCC::SignedGreaterThan, s) => Some(s),
            (true, IntCC::SignedLessThan, s) | (false, IntCC::SignedGreaterThanOrEqual, s) if s > smin => Some(s - 1),
            _ => None,
        };
        if let (Some(lo), Some(hi)) = (s_lo, s_hi) && lo <= hi {
            let (span, lo_u) = ((hi as u64).wrapping_sub(lo as u64) & umax, (lo as u64) & umax);
            if span < umax { return Some(((lo_u != 0).then_some((Binary::Isub, lo_u as i64)), out_cc, span as i64)); }
        }
    }
    None
}

/// Fuses each `and` or `or` of two single-use compares of one value against constants into one range or mask compare.
pub(super) fn fold_range_checks(func: &mut Function) -> bool {
    let mut compare = vec![false; func.types.len()];
    for data in func.blocks.iter().flat_map(|block| &block.insts).map(|inst| &func.insts[inst.index()]) {
        if let (&Op::Icmp(_, l, _), Some(res)) = (&data.op, data.result) && func.ty(l).is_small_int() { compare[res.index()] = true; }
    }
    let pair = |inst: &Inst| matches!(func.insts[inst.index()].op, Op::Binary(Binary::Band | Binary::Bor, a, b) if compare[a.index()] && compare[b.index()]);
    if !func.blocks.iter().flat_map(|block| &block.insts).any(pair) { return false; }
    let (defs, uses) = (defs(func), func.use_counts());
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) && func.ty(res).is_small_int() {
            consts[res.index()] = Some(extend(imm, func.ty(res), false));
        }
    }
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        let orig = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut next = Vec::with_capacity(orig.len());
        for inst in orig {
            if let Op::Binary(op @ (Binary::Band | Binary::Bor), a, b) = func.insts[inst.index()].op
                && uses[a.index()] == 1 && uses[b.index()] == 1
                && let (Some((cc1, x1, c1)), Some((cc2, x2, c2))) = (cmp_const(func, &defs, &consts, a), cmp_const(func, &defs, &consts, b))
                && x1 == x2
                && let Some((pre, out_cc, out_imm)) = fuse_cmp_pair(op == Binary::Band, func.ty(x1), (cc1, c1), (cc2, c2))
            {
                let x_ty = func.ty(x1);
                let lhs = if let Some((pre_op, pre_imm)) = pre {
                    let (c_pre, r_pre) = (func.value(x_ty), func.value(x_ty));
                    consts.extend([Some(extend(pre_imm, x_ty, false)), None]);
                    next.push(func.add(Op::Const(pre_imm), Some(c_pre)));
                    next.push(func.add(Op::Binary(pre_op, x1, c_pre), Some(r_pre)));
                    r_pre
                } else { x1 };
                let c_out = func.value(x_ty);
                consts.push(Some(extend(out_imm, x_ty, false)));
                next.push(func.add(Op::Const(out_imm), Some(c_out)));
                func.insts[inst.index()].op = Op::Icmp(out_cc, lhs, c_out);
                changed = true;
            }
            next.push(inst);
        }
        func.blocks[b_idx].insts = next;
    }
    changed
}
