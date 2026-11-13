use crate::fold::extend;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

fn emit_op(func: &mut Function, out: &mut Vec<Inst>, ty: Type, op: Op) -> Value {
    let val = func.value(ty);
    out.push(func.add(op, Some(val)));
    val
}

fn magic_udiv(func: &mut Function, out: &mut Vec<Inst>, ty: Type, x: Value, d: u64) -> Op {
    if ty != Type::I64 {
        let x64 = emit_op(func, out, Type::I64, Op::Convert(Convert::Uextend, x));
        let m = ((1_u128 << 64) / u128::from(d) + 1) as u64;
        let m_val = emit_op(func, out, Type::I64, Op::Const(m as i64));
        let q64 = emit_op(func, out, Type::I64, Op::Binary(Binary::Umulhi, x64, m_val));
        return Op::Convert(Convert::Ireduce, q64);
    }
    let l = 64 - d.leading_zeros();
    let s = l - 1;
    let d128 = u128::from(d);
    let m_cand = (1_u128 << (64 + s)) / d128 + 1;
    let err = m_cand * d128 - (1_u128 << (64 + s));
    if m_cand < (1_u128 << 64) && (u128::from(u64::MAX / d) + 1) * err < m_cand {
        let m_val = emit_op(func, out, Type::I64, Op::Const(m_cand as u64 as i64));
        let hi = emit_op(func, out, Type::I64, Op::Binary(Binary::Umulhi, x, m_val));
        let sh = emit_op(func, out, Type::I64, Op::Const(i64::from(s)));
        return Op::Binary(Binary::Ushr, hi, sh);
    }
    let m_full = (1_u128 << (64 + l)) / d128 + 1;
    let m_low = (m_full - (1_u128 << 64)) as u64;
    let m_val = emit_op(func, out, Type::I64, Op::Const(m_low as i64));
    let t1 = emit_op(func, out, Type::I64, Op::Binary(Binary::Umulhi, x, m_val));
    let diff = emit_op(func, out, Type::I64, Op::Binary(Binary::Isub, x, t1));
    let one = emit_op(func, out, Type::I64, Op::Const(1));
    let half = emit_op(func, out, Type::I64, Op::Binary(Binary::Ushr, diff, one));
    let t2 = emit_op(func, out, Type::I64, Op::Binary(Binary::Iadd, t1, half));
    let sh = emit_op(func, out, Type::I64, Op::Const(i64::from(l - 1)));
    Op::Binary(Binary::Ushr, t2, sh)
}

fn has_clear_low_byte_bits(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], v: Value) -> bool {
    let Some(inst) = defs.get(v.index()).copied().flatten() else { return false };
    let is_const = |val: Value, want: u64| consts.get(val.index()).copied().flatten().is_some_and(|c| c as u64 == want);
    match func.insts[inst.index()].op {
        Op::Unary(Unary::Bnot, inner) => {
            let Some(inner_inst) = defs.get(inner.index()).copied().flatten() else { return false };
            matches!(
                func.insts[inner_inst.index()].op,
                Op::Binary(Binary::Bor, a, b) if is_const(a, 0x7f7f_7f7f_7f7f_7f7f) || is_const(b, 0x7f7f_7f7f_7f7f_7f7f)
            )
        }
        Op::Binary(Binary::Band, a, b) => is_const(a, 0x8080_8080_8080_8080) || is_const(b, 0x8080_8080_8080_8080),
        _ => false,
    }
}

fn zero_compare(func: &mut Function, out: &mut Vec<Inst>, consts: &[Option<i64>], cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let uconst = |v: Value| consts.get(v.index()).copied().flatten().map(|c| extend(c, ty, false) as u64);
    let (val, new_cc, existing_zero) = match (cc, uconst(x), uconst(y)) {
        (IntCC::UnsignedLessThanOrEqual, _, Some(0)) => (x, IntCC::Equal, Some(y)),
        (IntCC::UnsignedGreaterThanOrEqual, Some(0), _) => (y, IntCC::Equal, Some(x)),
        (IntCC::UnsignedGreaterThan, _, Some(0)) => (x, IntCC::NotEqual, Some(y)),
        (IntCC::UnsignedLessThan, Some(0), _) => (y, IntCC::NotEqual, Some(x)),
        (IntCC::UnsignedLessThan, _, Some(1)) => (x, IntCC::Equal, None),
        (IntCC::UnsignedGreaterThan, Some(1), _) => (y, IntCC::Equal, None),
        (IntCC::UnsignedGreaterThanOrEqual, _, Some(1)) => (x, IntCC::NotEqual, None),
        (IntCC::UnsignedLessThanOrEqual, Some(1), _) => (y, IntCC::NotEqual, None),
        _ => return None,
    };
    let zero = existing_zero.unwrap_or_else(|| emit_op(func, out, ty, Op::Const(0)));
    Some(Op::Icmp(new_cc, val, zero))
}

fn negated_shift_add(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], ty: Type, a: Value, b: Value) -> Option<(Value, Value, i64)> {
    let is_zero = |v: Value| consts.get(v.index()).copied().flatten().is_some_and(|c| extend(c, ty, false) == 0);
    let uconst = |v: Value| consts.get(v.index()).copied().flatten().map(|c| extend(c, ty, false) as u64);
    for (base, rhs) in [(a, b), (b, a)] {
        let Some(shl_inst) = defs.get(rhs.index()).copied().flatten() else { continue };
        let (neg, shift) = match func.insts[shl_inst.index()].op {
            Op::Binary(Binary::Ishl, neg, k) => (neg, consts.get(k.index()).copied().flatten()?),
            Op::Binary(Binary::Imul, m1, m2) => {
                if let Some(m) = uconst(m2).filter(|&m| m > 1 && m.is_power_of_two()) {
                    (m1, i64::from(m.trailing_zeros()))
                } else if let Some(m) = uconst(m1).filter(|&m| m > 1 && m.is_power_of_two()) {
                    (m2, i64::from(m.trailing_zeros()))
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let Some(neg_inst) = defs.get(neg.index()).copied().flatten() else { continue };
        let idx = match func.insts[neg_inst.index()].op {
            Op::Binary(Binary::Isub, zero, idx) if is_zero(zero) => idx,
            Op::Unary(Unary::Ineg, idx) => idx,
            _ => continue,
        };
        return Some((base, idx, shift));
    }
    None
}

fn byte_msb_mask(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], a: Value, b: Value) -> Option<Op> {
    if func.ty(a) != Type::I64 {
        return None;
    }
    let is_const = |v: Value, want: u64| consts.get(v.index()).copied().flatten().is_some_and(|c| c as u64 == want);
    for (lhs, mask_v) in [(a, b), (b, a)] {
        if !is_const(mask_v, 0x8080_8080_8080_8080) {
            continue;
        }
        let Some(mul_inst) = defs.get(lhs.index()).copied().flatten() else { continue };
        let Op::Binary(Binary::Imul, m1, m2) = func.insts[mul_inst.index()].op else { continue };
        for (shifted, mul_c) in [(m1, m2), (m2, m1)] {
            if !is_const(mul_c, 0xFF) {
                continue;
            }
            let Some(shr_inst) = defs.get(shifted.index()).copied().flatten() else { continue };
            let Op::Binary(Binary::Ushr, x, sh) = func.insts[shr_inst.index()].op else { continue };
            if consts.get(sh.index()).copied().flatten() == Some(7) && has_clear_low_byte_bits(func, defs, consts, x) {
                return Some(Op::Binary(Binary::Band, x, mask_v));
            }
        }
    }
    None
}

/// Rewrites constant unsigned divisions/remainders, unsigned zero compares, negated shift-adds, and redundant byte-MSB masks.
pub(crate) fn rewrite(func: &mut Function, defs: &[Option<Inst>]) {
    let mut consts = vec![None; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(*imm);
        }
    }
    for b in 0..func.blocks.len() {
        let old = std::mem::take(&mut func.blocks[b].insts);
        let mut out = Vec::with_capacity(old.len());
        let mut block_udivs: Vec<(Value, Value, Value)> = Vec::new();
        for inst in old {
            let op = func.insts[inst.index()].op.clone();
            let res = func.insts[inst.index()].result;
            let replacement = match op {
                Op::Icmp(cc, x, y) => zero_compare(func, &mut out, &consts, cc, x, y),
                Op::Binary(Binary::Iadd, a, b) => negated_shift_add(func, defs, &consts, func.ty(a), a, b).map(|(base, idx, shift)| {
                    let ty = func.ty(base);
                    let k = emit_op(func, &mut out, ty, Op::Const(shift));
                    let shl = emit_op(func, &mut out, ty, Op::Binary(Binary::Ishl, idx, k));
                    Op::Binary(Binary::Isub, base, shl)
                }),
                Op::Binary(Binary::Band, a, b) => byte_msb_mask(func, defs, &consts, a, b),
                Op::Binary(Binary::Udiv, x, y) if matches!(func.ty(x), Type::I8 | Type::I16 | Type::I32 | Type::I64) => {
                    let ty = func.ty(x);
                    if let Some(d) = consts.get(y.index()).copied().flatten().map(|c| extend(c, ty, false) as u64).filter(|&d| d > 1) {
                        if d.is_power_of_two() {
                            let sh = emit_op(func, &mut out, ty, Op::Const(i64::from(d.trailing_zeros())));
                            Some(Op::Binary(Binary::Ushr, x, sh))
                        } else {
                            Some(magic_udiv(func, &mut out, ty, x, d))
                        }
                    } else {
                        if let Some(r) = res {
                            block_udivs.push((x, y, r));
                        }
                        None
                    }
                }
                Op::Binary(Binary::Urem, x, y) if matches!(func.ty(x), Type::I8 | Type::I16 | Type::I32 | Type::I64) => {
                    let ty = func.ty(x);
                    if let Some(d) = consts.get(y.index()).copied().flatten().map(|c| extend(c, ty, false) as u64).filter(|&d| d > 1) {
                        if d.is_power_of_two() {
                            let mask = emit_op(func, &mut out, ty, Op::Const((d - 1) as i64));
                            Some(Op::Binary(Binary::Band, x, mask))
                        } else {
                            let q_op = magic_udiv(func, &mut out, ty, x, d);
                            let q = emit_op(func, &mut out, ty, q_op);
                            let prod = emit_op(func, &mut out, ty, Op::Binary(Binary::Imul, q, y));
                            Some(Op::Binary(Binary::Isub, x, prod))
                        }
                    } else if let Some(&(_, _, q)) = block_udivs.iter().find(|&&(ux, uy, _)| ux == x && uy == y) {
                        let prod = emit_op(func, &mut out, ty, Op::Binary(Binary::Imul, q, y));
                        Some(Op::Binary(Binary::Isub, x, prod))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(new_op) = replacement {
                func.insts[inst.index()].op = new_op;
            }
            out.push(inst);
        }
        func.blocks[b].insts = out;
    }
}

/// Devirtualizes loads of static vtable slots into direct slot/symbol loads.
pub(crate) fn devirt_vtables(func: &mut Function, defs: &[Option<Inst>]) -> bool {
    if func.vtable_slots.is_empty() {
        return false;
    }
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let mut i = 0;
        while i < func.blocks[b].insts.len() {
            let inst = func.blocks[b].insts[i];
            if let (Op::Load(addr, off), Some(res)) = (func.insts[inst.index()].op.clone(), func.insts[inst.index()].result)
                && func.ty(res) == Type::I64
                && let Some(def_inst) = defs.get(addr.index()).copied().flatten()
                && let Op::Symbol(vt_sym) = func.insts[def_inst.index()].op
                && let Some(&(_, _, slot_sym, low)) = func.vtable_slots.iter().find(|&&(v, o, _, _)| v == vt_sym && o == off)
            {
                let slot_val = func.value(Type::I64);
                let slot_op = if low { Op::SlotAddr(slot_sym) } else { Op::Symbol(slot_sym) };
                let slot_inst = func.add(slot_op, Some(slot_val));
                func.blocks[b].insts.insert(i, slot_inst);
                func.insts[inst.index()].op = Op::Load(slot_val, 0);
                changed = true;
                i += 2;
                continue;
            }
            i += 1;
        }
    }
    changed
}

/// Whether `value` is the floating-point constant `2.0`.
pub(crate) fn is_float_two(func: &Function, defs: &[Option<Inst>], value: Value) -> bool {
    let Some(inst) = defs[value.index()] else { return false };
    let Op::Const(imm) = func.insts[inst.index()].op else { return false };
    match func.ty(value) {
        Type::F64 => imm as u64 == 0x4000_0000_0000_0000,
        Type::F32 => imm as u32 == 0x4000_0000,
        _ => false,
    }
}

/// The boolean value `op` negates (`v ^ 1`, `v == 0`, `v != 1`, or `select(v, 0, 1)`), plus any existing `1` constant.
fn negated(op: &Op, consts: &[Option<i64>], bools: &[bool]) -> Option<(Value, Option<Value>)> {
    let is = |value: Value, want: i64| consts[value.index()] == Some(want);
    match *op {
        Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) if is(y, 1) => Some((x, Some(y))),
        Op::Binary(Binary::Bxor, y, x) | Op::Icmp(IntCC::NotEqual, y, x) if is(y, 1) => Some((x, Some(y))),
        Op::Icmp(IntCC::Equal, x, y) if is(y, 0) => Some((x, None)),
        Op::Icmp(IntCC::Equal, y, x) if is(y, 0) => Some((x, None)),
        Op::Select(c, x, y) if is(x, 0) && is(y, 1) && bools[c.index()] => Some((c, Some(y))),
        _ => None,
    }
}

/// Rewrites a negation of a compare, `x ^ 1`, `x == 0`, `x != 1`, or `select(x, 0, 1)`, as the inverse compare or `x ^ 1`, and simplifies branches on `x == 0` / `x != 0`.
pub(crate) fn invert(func: &mut Function) {
    let count = func.types.len();
    let (mut compares, mut consts, mut xors, mut bools) = (vec![None; count], vec![None; count], vec![None; count], vec![false; count]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        match (&data.op, data.result) {
            (&Op::Icmp(cc, x, y), Some(result)) => {
                compares[result.index()] = Some((cc, x, y));
                bools[result.index()] = true;
            }
            (&Op::Fcmp(..) | &Op::Fits(..), Some(result)) => bools[result.index()] = true,
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Binary(Binary::Bxor, x, y), Some(result)) => xors[result.index()] = Some((x, y)),
            _ => {}
        }
    }
    let is_01 = |v: Value, bools: &[bool]| bools[v.index()] || matches!(consts[v.index()], Some(0 | 1));
    for _ in 0..2 {
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result else { continue };
            bools[result.index()] |= match data.op {
                Op::Binary(Binary::Band, x, y) => matches!(consts[x.index()], Some(0 | 1)) || matches!(consts[y.index()], Some(0 | 1)) || (is_01(x, &bools) && is_01(y, &bools)),
                Op::Binary(Binary::Bor | Binary::Bxor, x, y) | Op::Select(_, x, y) => is_01(x, &bools) && is_01(y, &bools),
                Op::Convert(Convert::Uextend, inner) => bools[inner.index()],
                _ => false,
            };
        }
    }
    let types = &func.types;
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let res = func.insts[inst.index()].result;
        let op = &mut func.insts[inst.index()].op;
        if let (Some(res), Some((inner, opt_one))) = (res, negated(op, &consts, &bools)) {
            let res_ty = types[res.index()];
            if res_ty == Type::I8
                && let Some((cc, a, b)) = compares[inner.index()]
            {
                *op = Op::Icmp(cc.inverse(), a, b);
                continue;
            }
            if bools[inner.index()]
                && let Some(one) = opt_one
                && types[inner.index()] == res_ty
                && types[one.index()] == res_ty
                && !matches!(*op, Op::Binary(Binary::Bxor, ..))
            {
                *op = Op::Binary(Binary::Bxor, inner, one);
                continue;
            }
        }
        let Op::Brif(condition, then, then_args, otherwise, else_args) = op else { continue };
        if let Some((x, y)) = xors[condition.index()] {
            let inner = if consts[y.index()] == Some(1) {
                Some(x)
            } else if consts[x.index()] == Some(1) {
                Some(y)
            } else {
                None
            };
            if let Some(inner) = inner.filter(|v| bools[v.index()]) {
                *condition = inner;
                std::mem::swap(then, otherwise);
                std::mem::swap(then_args, else_args);
                continue;
            }
        }
        let Some((cc, x, y)) = compares[condition.index()] else { continue };
        let (val, flipped) = if consts[y.index()] == Some(0) && types[x.index()].is_small_int() {
            (x, false)
        } else if consts[x.index()] == Some(0) && types[y.index()].is_small_int() {
            (y, true)
        } else {
            continue;
        };
        match (flipped, cc) {
            (_, IntCC::NotEqual) | (false, IntCC::UnsignedGreaterThan) | (true, IntCC::UnsignedLessThan) => *condition = val,
            (_, IntCC::Equal) | (false, IntCC::UnsignedLessThanOrEqual) | (true, IntCC::UnsignedGreaterThanOrEqual) => {
                *condition = val;
                std::mem::swap(then, otherwise);
                std::mem::swap(then_args, else_args);
            }
            _ => {}
        }
    }
}
