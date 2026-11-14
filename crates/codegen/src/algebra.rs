use crate::fold::{extend, holds};
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, ShuffleMask, Type, Unary, Value};

mod icmp;
mod select;
mod vec;
pub(crate) use icmp::{is_non_null_addr, non_null_cmp};
pub(crate) use select::{fold_bxor_conds, invert};

struct RewriteCx<'a> {
    func: &'a mut Function,
    defs: Vec<Option<Inst>>,
    consts: Vec<Option<i64>>,
    uses: Vec<u32>,
    out: Vec<Inst>,
    subst: Vec<Value>,
}

impl<'a> RewriteCx<'a> {
    fn def(&self, v: Value) -> Option<Inst> { self.defs.get(v.index()).copied().flatten() }
    fn sconst(&self, v: Value) -> Option<i64> { self.consts.get(v.index()).copied().flatten() }
    fn uconst(&self, v: Value, ty: Type) -> Option<u64> { self.sconst(v).map(|c| extend(c, ty, false) as u64) }
    fn known(&self, v: Value) -> (u64, u64) { crate::fold::known_bits(self.func, &|x| self.def(x), &|x| self.sconst(x), v, 0) }
    fn op(&self, v: Value) -> Option<&Op> { self.def(v).map(|inst| &self.func.insts[inst.index()].op) }
    fn count(&self, sh: Value, ty: Type) -> Option<u32> { self.sconst(sh).map(|imm| (imm as u32) & (ty.bits() - 1)) }
    fn single_use(&self, v: Value) -> bool { self.uses.get(v.index()).copied().unwrap_or(1) <= 1 }
    fn is_unit_bool(&self, v: Value) -> bool { let t = self.func.ty(v); t.is_small_int() && (self.known(v).0 & (t.unsigned_max() & !1)) == (t.unsigned_max() & !1) }

    fn emit(&mut self, ty: Type, op: Op) -> Value {
        let val = self.func.value(ty);
        if let Op::Const(imm) = op { self.consts.resize(self.func.types.len(), None); self.consts[val.index()] = Some(imm); }
        let inst = self.func.add(op, Some(val));
        self.defs.resize(self.func.types.len(), None);
        self.defs[val.index()] = Some(inst);
        self.out.push(inst);
        val
    }

    fn alias(&mut self, res: Option<Value>, target: Value) {
        if let Some(r) = res {
            let len = self.func.types.len();
            self.subst.extend((self.subst.len()..len).map(|i| Value(i as u32)));
            self.subst[r.index()] = crate::merge::find(&self.subst, target);
        }
    }

    fn redundant_mask(&self, src: Value, ty: Type, mask: u64) -> bool {
        let active = !self.known(src).0 & ty.unsigned_max();
        self.func.ty(src) == ty && (active & mask) == active
    }

    fn unwrap_uextend(&self, mut v: Value) -> Value {
        while let Some(inst) = self.def(v) && let Op::Convert(Convert::Uextend, inner) = self.func.insts[inst.index()].op { v = inner; }
        v
    }
}

fn magic_udiv(cx: &mut RewriteCx, ty: Type, x: Value, d: u64) -> Op {
    if ty != Type::I64 {
        let x64 = cx.emit(Type::I64, Op::Convert(Convert::Uextend, x));
        let m = ((1_u128 << 64) / u128::from(d) + 1) as u64;
        let m_val = cx.emit(Type::I64, Op::Const(m as i64));
        let q64 = cx.emit(Type::I64, Op::Binary(Binary::Umulhi, x64, m_val));
        return Op::Convert(Convert::Ireduce, q64);
    }
    let l = 64 - d.leading_zeros();
    let s = l - 1;
    let d128 = u128::from(d);
    let m_cand = (1_u128 << (64 + s)) / d128 + 1;
    let err = m_cand * d128 - (1_u128 << (64 + s));
    if m_cand < (1_u128 << 64) && (u128::from(u64::MAX / d) + 1) * err < m_cand {
        let m_val = cx.emit(Type::I64, Op::Const(m_cand as u64 as i64));
        let hi = cx.emit(Type::I64, Op::Binary(Binary::Umulhi, x, m_val));
        let sh = cx.emit(Type::I64, Op::Const(i64::from(s)));
        return Op::Binary(Binary::Ushr, hi, sh);
    }
    let m_full = (1_u128 << (64 + l)) / d128 + 1;
    let m_low = (m_full - (1_u128 << 64)) as u64;
    let m_val = cx.emit(Type::I64, Op::Const(m_low as i64));
    let t1 = cx.emit(Type::I64, Op::Binary(Binary::Umulhi, x, m_val));
    let diff = cx.emit(Type::I64, Op::Binary(Binary::Isub, x, t1));
    let one = cx.emit(Type::I64, Op::Const(1));
    let half = cx.emit(Type::I64, Op::Binary(Binary::Ushr, diff, one));
    let t2 = cx.emit(Type::I64, Op::Binary(Binary::Iadd, t1, half));
    let sh = cx.emit(Type::I64, Op::Const(i64::from(l - 1)));
    Op::Binary(Binary::Ushr, t2, sh)
}

enum Demand {
    Const(u64),
    Bits(Value),
}

fn shift_demand(op: Binary, w: u64, k: u32, mask: u64) -> u64 {
    if op == Binary::Ushr { (mask << k) & w } else { mask >> k }
}

fn demanded(cx: &RewriteCx, v: Value, mask: u64, depth: u8) -> Demand {
    let (kz, ko) = cx.known(v);
    if ((kz | ko) & mask) == mask {
        return Demand::Const(ko & mask);
    }
    let Some(inst) = cx.def(v).filter(|_| depth < 6) else { return Demand::Bits(v) };
    let ty = cx.func.ty(v);
    let next = |u: Value, m: u64| demanded(cx, u, m, depth + 1);
    match cx.func.insts[inst.index()].op {
        Op::Binary(bin @ (Binary::Bor | Binary::Bxor | Binary::Band), a, b) => [(a, b), (b, a)].into_iter().find_map(|(keep, drop)| {
            let bits = if bin == Binary::Band { cx.known(drop).1 } else { cx.known(drop).0 };
            ((bits & mask) == mask).then(|| next(keep, mask))
        }).unwrap_or(Demand::Bits(v)),
        Op::Unary(Unary::Bnot, inner) => match cx.op(inner) {
            Some(&Op::Binary(Binary::Bxor, x, y)) => [(x, y), (y, x)].into_iter().find_map(|(src, cv)| {
                cx.uconst(cv, ty).filter(|c| (c & mask) == mask).map(|_| next(src, mask))
            }).unwrap_or(Demand::Bits(v)),
            _ => Demand::Bits(v),
        },
        Op::Binary(op @ (Binary::Ishl | Binary::Ushr), a, sh) => {
            let Some(k) = cx.count(sh, ty) else { return Demand::Bits(v) };
            let (w, inner) = (ty.unsigned_max(), shift_demand(op, ty.unsigned_max(), k, mask));
            match next(a, inner) {
                Demand::Const(c) => Demand::Const(if op == Binary::Ushr { c >> k } else { (c << k) & mask }),
                Demand::Bits(s) => match cx.op(s) {
                    _ if k == 0 => Demand::Bits(s),
                    Some(&Op::Binary(inv @ (Binary::Ishl | Binary::Ushr), t, sh2)) if inv != op && cx.count(sh2, ty) == Some(k) && shift_demand(inv, w, k, inner) == mask => next(t, mask),
                    _ => Demand::Bits(v),
                },
            }
        }
        Op::Convert(kind @ (Convert::Uextend | Convert::Ireduce), a) if cx.func.ty(a).is_small_int() => {
            let inner = mask & cx.func.ty(a).unsigned_max();
            match next(a, inner) {
                Demand::Const(c) => Demand::Const(c),
                Demand::Bits(s) => match cx.op(s) {
                    Some(&Op::Convert(back @ (Convert::Uextend | Convert::Ireduce), t)) if back != kind && inner == mask && cx.func.ty(t) == ty => next(t, mask),
                    _ => Demand::Bits(v),
                },
            }
        }
        _ => Demand::Bits(v),
    }
}

fn field_of(cx: &RewriteCx, v: Value, depth: u8) -> (Value, u64) {
    let ty = cx.func.ty(v);
    let w = ty.unsigned_max();
    let part = |s: Value, m: u64| { let (src, sm) = field_of(cx, s, depth + 1); (src, sm & m) };
    match cx.op(v).filter(|_| depth < 6 && ty.is_small_int()) {
        Some(&Op::Binary(Binary::Band, a, b)) => [(a, b), (b, a)].into_iter().find_map(|(s, c)| cx.uconst(c, ty).map(|m| part(s, m))).unwrap_or((v, w)),
        Some(&Op::Convert(Convert::Uextend, n)) => match cx.op(n) {
            Some(&Op::Convert(Convert::Ireduce, s)) if cx.func.ty(s) == ty => part(s, cx.func.ty(n).unsigned_max()),
            _ => (v, w),
        },
        Some(&Op::Binary(op @ (Binary::Ishl | Binary::Ushr), f, sh)) => {
            let (s1, m1) = field_of(cx, f, depth + 1);
            match (cx.count(sh, ty), cx.op(s1)) {
                (Some(k), Some(&Op::Binary(inv @ (Binary::Ishl | Binary::Ushr), s, sh2))) if inv != op && cx.count(sh2, ty) == Some(k) => part(s, shift_demand(inv, w, k, m1)),
                _ => (v, w),
            }
        }
        _ => (v, w),
    }
}

fn merge_fields(cx: &mut RewriteCx, res: Option<Value>, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    let ((src, ma), (other, mb)) = (field_of(cx, a, 0), field_of(cx, b, 0));
    if src != other || !ty.is_small_int() { return None; }
    if cx.redundant_mask(src, ty, ma | mb) { cx.alias(res, src); return None; }
    let mask = cx.emit(ty, Op::Const(extend((ma | mb) as i64, ty, true)));
    Some(Op::Binary(Binary::Band, src, mask))
}

fn known_cond(cx: &mut RewriteCx, cond: Value) -> Option<Value> {
    let (ty, (kz, ko)) = (cx.func.ty(cond), cx.known(cond));
    (cx.sconst(cond).is_none() && ty.is_small_int() && (ko != 0 || kz == ty.unsigned_max())).then(|| cx.emit(ty, Op::Const(i64::from(ko != 0))))
}

fn match_swar_msb_broadcast(cx: &RewriteCx, v: Value) -> Option<Value> {
    if cx.func.ty(v) != Type::I64 {
        return None;
    }
    let Op::Binary(Binary::Imul, m1, m2) = cx.func.insts[cx.def(v)?.index()].op else { return None };
    [(m1, m2), (m2, m1)].into_iter().find_map(|(shifted, mul_c)| {
        (cx.sconst(mul_c) == Some(0xFF)).then(|| {
            let Op::Binary(Binary::Ushr, x, sh) = cx.func.insts[cx.def(shifted)?.index()].op else { return None };
            (cx.sconst(sh) == Some(7) && (cx.known(x).0 & 0x7f7f_7f7f_7f7f_7f7f) == 0x7f7f_7f7f_7f7f_7f7f).then_some(x)
        }).flatten()
    })
}

fn zero_compare(cx: &mut RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let (val, new_cc, existing_zero) = match (cc, cx.uconst(x, ty), cx.uconst(y, ty)) {
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
    Some(Op::Icmp(new_cc, val, existing_zero.unwrap_or_else(|| cx.emit(ty, Op::Const(0)))))
}

fn narrow_pair_icmp(cx: &RewriteCx, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let (&Op::Convert(k1, a), &Op::Convert(k2, b)) = (&cx.func.insts[cx.def(x)?.index()].op, &cx.func.insts[cx.def(y)?.index()].op) else {
        return None;
    };
    if k1 != k2 || cx.func.ty(a) != cx.func.ty(b) || !cx.func.ty(a).is_small_int() {
        return None;
    }
    let ok = match k1 {
        Convert::Uextend => !cc.is_signed(),
        Convert::Sextend => cc.is_signed() || matches!(cc, IntCC::Equal | IntCC::NotEqual),
        _ => false,
    };
    ok.then_some(Op::Icmp(cc, a, b))
}

fn narrow_const_icmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    let ty = cx.func.ty(x);
    for (norm, ext_v, c_v) in [(cc, x, y), (crate::cse::flip_int_cc(cc), y, x)] {
        let Some(c_imm) = cx.sconst(c_v) else { continue };
        match demanded(cx, ext_v, ty.unsigned_max(), 0) {
            Demand::Const(k) if res.is_some_and(|r| cx.func.ty(r) == Type::I8) => return Some(Op::Const(i64::from(holds(norm, ty, k as i64, c_imm)))),
            Demand::Bits(v) if v != ext_v => return Some(Op::Icmp(norm, v, c_v)),
            _ => {}
        }
        let Some(inst) = cx.def(ext_v) else { continue };
        let Op::Convert(kind @ (Convert::Uextend | Convert::Sextend), a) = cx.func.insts[inst.index()].op else { continue };
        let sub_ty = cx.func.ty(a);
        if !sub_ty.is_small_int() || sub_ty.bits() >= ty.bits() { continue; }
        let signed = norm.is_signed();
        if kind == Convert::Uextend && !signed {
            let uc = extend(c_imm, ty, false) as u64;
            if uc <= sub_ty.unsigned_max() {
                let nc = cx.emit(sub_ty, Op::Const(extend(uc as i64, sub_ty, true)));
                return Some(Op::Icmp(norm, a, nc));
            }
            if res.is_some_and(|r| cx.func.ty(r) == Type::I8) {
                return Some(Op::Const(i64::from(matches!(norm, IntCC::NotEqual | IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual))));
            }
        } else if kind == Convert::Sextend && (signed || matches!(norm, IntCC::Equal | IntCC::NotEqual)) {
            let (sc, half) = (extend(c_imm, ty, true), 1_i64 << (sub_ty.bits() - 1));
            if (-half..half).contains(&sc) {
                let nc = cx.emit(sub_ty, Op::Const(extend(sc, sub_ty, true)));
                return Some(Op::Icmp(norm, a, nc));
            }
        }
    }
    None
}

fn simplify_icmp(cx: &mut RewriteCx, res: Option<Value>, cc: IntCC, x: Value, y: Value) -> Option<Op> {
    if matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some(op) = [(x, y), (y, x)].into_iter().find_map(|(m, z)| {
            (cx.sconst(z) == Some(0)).then(|| match_swar_msb_broadcast(cx, m)).flatten().map(|w| Op::Icmp(cc, w, z))
        })
    {
        return Some(op);
    }
    if cx.func.ty(x).is_small_int()
        && let Some(op) = narrow_pair_icmp(cx, cc, x, y).or_else(|| narrow_const_icmp(cx, res, cc, x, y))
    {
        return match op {
            Op::Icmp(ncc, nx, ny) => {
                if let Some(folded) = icmp::eval_select(cx, res, ncc, nx, ny).or_else(|| select::eval_select_zero_cmp(cx, res, ncc, nx, ny)) {
                    return folded;
                }
                Some(zero_compare(cx, ncc, nx, ny).unwrap_or(Op::Icmp(ncc, nx, ny)))
            }
            other => Some(other),
        };
    }
    if let Some(folded) = icmp::eval_select(cx, res, cc, x, y).or_else(|| select::eval_select_zero_cmp(cx, res, cc, x, y)) {
        return folded;
    }
    zero_compare(cx, cc, x, y)
}

fn negated_shift_add(cx: &RewriteCx, ty: Type, a: Value, b: Value) -> Option<(Value, Value, i64)> {
    for (base, rhs) in [(a, b), (b, a)] {
        let Some(shl_inst) = cx.def(rhs) else { continue };
        let (neg, shift) = match cx.func.insts[shl_inst.index()].op {
            Op::Binary(Binary::Ishl, neg, k) => (neg, cx.sconst(k)?),
            Op::Binary(Binary::Imul, m1, m2) => {
                [(m1, m2), (m2, m1)]
                    .into_iter()
                    .find_map(|(v, cv)| cx.uconst(cv, ty).filter(|&m| m > 1 && m.is_power_of_two()).map(|m| (v, i64::from(m.trailing_zeros()))))?
            }
            _ => continue,
        };
        let Some(neg_inst) = cx.def(neg) else { continue };
        match cx.func.insts[neg_inst.index()].op {
            Op::Binary(Binary::Isub, zero, idx) if cx.uconst(zero, ty) == Some(0) => return Some((base, idx, shift)),
            Op::Unary(Unary::Ineg, idx) => return Some((base, idx, shift)),
            _ => {}
        }
    }
    None
}

fn simplify_band(cx: &mut RewriteCx, res: Option<Value>, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() { return None; }
    for (lhs, mask_v) in [(a, b), (b, a)] {
        let Some(mask) = cx.uconst(mask_v, ty) else { continue };
        if cx.redundant_mask(lhs, ty, mask) { cx.alias(res, lhs); return None; }
        if mask == 0x8080_8080_8080_8080 && let Some(x) = match_swar_msb_broadcast(cx, lhs) { return Some(Op::Binary(Binary::Band, x, mask_v)); }
        match demanded(cx, lhs, mask, 0) {
            Demand::Const(c) => return Some(Op::Const(extend(c as i64, ty, true))),
            Demand::Bits(src) if src == lhs => {}
            Demand::Bits(src) if cx.redundant_mask(src, ty, mask) => { cx.alias(res, src); return None; }
            Demand::Bits(src) => return Some(Op::Binary(Binary::Band, src, mask_v)),
        }
    }
    icmp::fuse_cmp_ops(cx, res, true, a, b)
}

fn simplify_bxor(cx: &mut RewriteCx, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() { return None; }
    let all_ones = ty.unsigned_max();
    for (inner, c2_val) in [(a, b), (b, a)] {
        let Some(c2) = cx.sconst(c2_val) else { continue };
        if (extend(c2, ty, false) as u64) == all_ones { return Some(Op::Unary(Unary::Bnot, inner)); }
        let Some(inner_inst) = cx.def(inner) else { continue };
        let Op::Binary(Binary::Bxor, x, y) = cx.func.insts[inner_inst.index()].op else { continue };
        if let Some((src, c1)) = [(x, y), (y, x)].into_iter().find_map(|(s, cv)| cx.sconst(cv).map(|c| (s, c))) {
            let xor_imm = extend(c1 ^ c2, ty, true);
            if (xor_imm as u64 & all_ones) == all_ones { return Some(Op::Unary(Unary::Bnot, src)); }
            let combined = cx.emit(ty, Op::Const(xor_imm));
            return Some(Op::Binary(Binary::Bxor, src, combined));
        }
    }
    None
}

fn simplify_ireduce(cx: &mut RewriteCx, res: Option<Value>, dst: Type, x: Value) -> Option<Op> {
    let cur = match demanded(cx, x, dst.unsigned_max(), 0) {
        Demand::Const(c) => return Some(Op::Const(extend(c as i64, dst, true))),
        Demand::Bits(v) => v,
    };
    if let Some(inst) = cx.def(cur) {
        match cx.func.insts[inst.index()].op {
            Op::Convert(k @ (Convert::Uextend | Convert::Sextend), inner) => {
                let ity = cx.func.ty(inner);
                if ity == dst { cx.alias(res, inner); return None; }
                if ity.is_small_int() { return Some(Op::Convert(if ity.bits() < dst.bits() { k } else { Convert::Ireduce }, inner)); }
            }
            Op::Convert(Convert::Ireduce, inner) if cx.func.ty(inner).is_small_int() => return Some(Op::Convert(Convert::Ireduce, inner)),
            _ => {}
        }
    }
    ireduce_rotate(cx, dst, cur).or_else(|| (cur != x).then_some(Op::Convert(Convert::Ireduce, cur)))
}

fn simplify_convert(cx: &mut RewriteCx, res: Option<Value>, kind: Convert, x: Value) -> Option<Op> {
    let (dst, src) = (cx.func.ty(res?), cx.func.ty(x));
    if !dst.is_small_int() || !src.is_small_int() { return None; }
    if kind == Convert::Ireduce && dst.bits() < src.bits() { return simplify_ireduce(cx, res, dst, x); }
    if dst.bits() <= src.bits() { return None; }
    match (kind, &cx.func.insts[cx.def(x)?.index()].op) {
        (Convert::Uextend, &Op::Convert(Convert::Uextend, inner)) if cx.func.ty(inner).is_small_int() => Some(Op::Convert(Convert::Uextend, inner)),
        (Convert::Sextend, &Op::Convert(Convert::Sextend, inner)) if cx.func.ty(inner).is_small_int() => Some(Op::Convert(Convert::Sextend, inner)),
        (Convert::Uextend, &Op::Convert(Convert::Ireduce, inner)) if cx.func.ty(inner) == dst => {
            let hi = dst.unsigned_max() & !src.unsigned_max();
            if (cx.known(inner).0 & hi) == hi { cx.alias(res, inner); }
            None
        }
        _ => None,
    }
}

fn fold_nested_shift(cx: &mut RewriteCx, res: Option<Value>, (op, ty): (Binary, Type), (cur, sh): (Value, Value), k: u32) -> Option<Op> {
    let Op::Binary(inner_op, inner, sh1) = cx.func.insts[cx.def(cur)?.index()].op else { return None };
    let k1 = cx.count(sh1, ty)?;
    let inv_op = if op == Binary::Ushr { Binary::Ishl } else { Binary::Ushr };
    if inner_op == op {
        if k1 == 0 { return Some(Op::Binary(op, inner, sh)); }
        if k1 + k < ty.bits() {
            let sum = cx.emit(ty, Op::Const(i64::from(k1 + k)));
            return Some(Op::Binary(op, inner, sum));
        }
        return Some(Op::Const(0));
    }
    if inner_op == inv_op && k1 == k {
        let keep = if op == Binary::Ushr { ty.unsigned_max() >> k } else { ty.unsigned_max() & !((1_u64 << k) - 1) };
        if cx.redundant_mask(inner, ty, keep) { cx.alias(res, inner); return None; }
        let mv = cx.emit(ty, Op::Const(extend(keep as i64, ty, true)));
        return Some(Op::Binary(Binary::Band, inner, mv));
    }
    None
}

fn simplify_shift(cx: &mut RewriteCx, res: Option<Value>, op: Binary, x: Value, sh: Value) -> Option<Op> {
    if op == Binary::Ushr && cx.func.ty(x) == Type::I64 && cx.sconst(sh) == Some(3)
        && let Some(inst) = cx.def(x) && let Op::Unary(Unary::Ctz, m) = cx.func.insts[inst.index()].op
        && let Some(word) = match_swar_msb_broadcast(cx, m)
    {
        return Some(Op::Binary(Binary::Ushr, cx.emit(Type::I64, Op::Unary(Unary::Ctz, word)), sh));
    }
    let ty = cx.func.ty(x);
    if matches!(ty, Type::I32 | Type::I64) && cx.sconst(sh).is_none()
        && let count = strip_rot_mask(cx, cx.func.ty(sh), sh, u64::from(ty.bits() - 1)) && count != sh
    {
        return Some(Op::Binary(op, x, count));
    }
    let raw_k = cx.sconst(sh).filter(|_| ty.is_small_int())?;
    let k = (raw_k as u32) & (ty.bits() - 1);
    if k == 0 { cx.alias(res, x); return None; }
    let cur = match demanded(cx, x, shift_demand(op, ty.unsigned_max(), k, ty.unsigned_max()), 0) {
        Demand::Const(c) => return crate::fold::binary(op, ty, c as i64, raw_k).map(|v| Op::Const(extend(v, ty, true))),
        Demand::Bits(v) => v,
    };
    fold_nested_shift(cx, res, (op, ty), (cur, sh), k).or_else(|| (cur != x).then_some(Op::Binary(op, cur, sh)))
}

fn fold_rodata_load(cx: &RewriteCx, ty: Type, addr: Value, off: i32) -> Option<Op> {
    if off < 0 { return None; }
    let size = if ty.is_vector() { ty.bytes() as usize } else { match ty { Type::I8 => 1, Type::I16 => 2, Type::I32 | Type::F32 => 4, Type::I64 | Type::F64 => 8, _ => return None } };
    let Op::Symbol(sym) = cx.func.insts[cx.def(addr)?.index()].op else { return None };
    let start = off as usize;
    let (_, bytes) = cx.func.rodata_bytes.iter().find(|&&(s, _)| s == sym)?;
    let slice = bytes.get(start..start.checked_add(size)?)?;
    if ty.is_vector() {
        return slice.iter().all(|&b| b == 0).then_some(Op::Const(0));
    }
    let mut raw = [0_u8; 8];
    raw[..size].copy_from_slice(slice);
    let u = u64::from_le_bytes(raw);
    Some(Op::Const(if ty.is_small_int() { extend(u as i64, ty, true) } else { u as i64 }))
}

fn reassociate_add(cx: &mut RewriteCx, fits_used: &[bool], a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() { return None; }
    for (lhs, c2_v) in [(a, b), (b, a)] {
        let Some(c2) = cx.sconst(c2_v).filter(|_| !fits_used.get(lhs.index()).copied().unwrap_or(false)) else { continue };
        let Some(inst) = cx.def(lhs) else { continue };
        match cx.func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, x, y) => if let Some((base, c1)) = [(x, y), (y, x)].into_iter().find_map(|(v, cv)| cx.sconst(cv).map(|c| (v, c))) {
                let c = cx.emit(ty, Op::Const(extend(c1.wrapping_add(c2), ty, true)));
                return Some(Op::Binary(Binary::Iadd, base, c));
            },
            Op::Binary(Binary::Isub, x, y) => if let Some(c1) = cx.sconst(y) {
                let c = cx.emit(ty, Op::Const(extend(c2.wrapping_sub(c1), ty, true)));
                return Some(Op::Binary(Binary::Iadd, x, c));
            },
            _ => {}
        }
    }
    None
}

fn reassociate_sub(cx: &mut RewriteCx, fits_used: &[bool], a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    let c2 = cx.sconst(b).filter(|_| ty.is_small_int() && !fits_used.get(a.index()).copied().unwrap_or(false))?;
    match cx.func.insts[cx.def(a)?.index()].op {
        Op::Binary(Binary::Iadd, x, y) => {
            let (base, c1) = [(x, y), (y, x)].into_iter().find_map(|(v, cv)| cx.sconst(cv).map(|c| (v, c)))?;
            let c = cx.emit(ty, Op::Const(extend(c1.wrapping_sub(c2), ty, true)));
            Some(Op::Binary(Binary::Iadd, base, c))
        }
        Op::Binary(Binary::Isub, x, y) => {
            let c = cx.emit(ty, Op::Const(extend(cx.sconst(y)?.wrapping_add(c2), ty, true)));
            Some(Op::Binary(Binary::Isub, x, c))
        }
        _ => None,
    }
}

fn simplify_div_rem(cx: &mut RewriteCx, block_udivs: &mut Vec<(Value, Value, Value)>, res: Option<Value>, (bin, x, y): (Binary, Value, Value)) -> Option<Op> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() { return None; }
    if let Some(d) = cx.uconst(y, ty).filter(|&d| d > 1) {
        if d.is_power_of_two() {
            return Some(if bin == Binary::Udiv {
                let sh = cx.emit(ty, Op::Const(i64::from(d.trailing_zeros())));
                Op::Binary(Binary::Ushr, x, sh)
            } else {
                let mask = cx.emit(ty, Op::Const((d - 1) as i64));
                Op::Binary(Binary::Band, x, mask)
            });
        }
        let q_op = magic_udiv(cx, ty, x, d);
        if bin == Binary::Udiv { return Some(q_op); }
        let q = cx.emit(ty, q_op);
        let prod = cx.emit(ty, Op::Binary(Binary::Imul, q, y));
        return Some(Op::Binary(Binary::Isub, x, prod));
    }
    if bin == Binary::Udiv {
        if let Some(r) = res { block_udivs.push((x, y, r)); }
        return None;
    }
    let &(_, _, q) = block_udivs.iter().find(|&&(ux, uy, _)| ux == x && uy == y)?;
    let prod = cx.emit(ty, Op::Binary(Binary::Imul, q, y));
    Some(Op::Binary(Binary::Isub, x, prod))
}

fn unmask_shift(cx: &RewriteCx, mut v: Value, bits: u32) -> Value {
    let m = u64::from(bits - 1);
    for _ in 0..4 {
        let Some(inst) = cx.def(v) else { break };
        let ty = cx.func.ty(v);
        v = match cx.func.insts[inst.index()].op {
            Op::Binary(Binary::Band, x, y) if cx.uconst(y, ty).is_some_and(|c| (c & m) == m) => x,
            Op::Binary(Binary::Band, x, y) if cx.uconst(x, ty).is_some_and(|c| (c & m) == m) => y,
            Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce, x) if cx.func.ty(x).is_small_int() => x,
            _ => break,
        };
    }
    v
}

fn unwrap_shift_arg(cx: &RewriteCx, v: Value, bits: u32) -> (Value, bool) {
    let v = unmask_shift(cx, v, bits);
    let mod_zero = |c: Value| cx.uconst(c, cx.func.ty(c)).is_some_and(|k| (k & u64::from(bits - 1)) == 0);
    if let Some(inst) = cx.def(v) {
        match cx.func.insts[inst.index()].op {
            Op::Unary(Unary::Ineg, inner) => return (unmask_shift(cx, inner, bits), true),
            Op::Binary(Binary::Isub, c, inner) if mod_zero(c) => return (unmask_shift(cx, inner, bits), true),
            Op::Binary(Binary::Iadd, p, q) => {
                for (c, neg) in [(p, q), (q, p)] {
                    if mod_zero(c) && let Some(ni) = cx.def(neg) && let Op::Unary(Unary::Ineg, inner) = cx.func.insts[ni.index()].op {
                        return (unmask_shift(cx, inner, bits), true);
                    }
                }
            }
            _ => {}
        }
    }
    (v, false)
}

fn match_shift_pair(cx: &RewriteCx, a: Value, b: Value) -> Option<((Value, Value), (Value, Value))> {
    let (ia, ib) = (cx.def(a)?, cx.def(b)?);
    [(ia, ib), (ib, ia)].into_iter().find_map(|(l, r)| {
        if let (Op::Binary(Binary::Ishl, x1, sh1), Op::Binary(Binary::Ushr, x2, sh2)) = (&cx.func.insts[l.index()].op, &cx.func.insts[r.index()].op) {
            Some(((*x1, *sh1), (*x2, *sh2)))
        } else { None }
    })
}

fn strip_rot_mask(cx: &RewriteCx, ty: Type, v: Value, keep: u64) -> Value {
    let Some(inst) = cx.def(v) else { return v };
    let Op::Binary(Binary::Band, x, y) = cx.func.insts[inst.index()].op else { return v };
    if cx.uconst(y, ty).is_some_and(|m| (m & keep) == keep) { x } else if cx.uconst(x, ty).is_some_and(|m| (m & keep) == keep) { y } else { v }
}

fn ireduce_rotate(cx: &mut RewriteCx, dst: Type, v: Value) -> Option<Op> {
    if dst != Type::I32 { return None; }
    let Op::Binary(Binary::Bor | Binary::Bxor | Binary::Iadd, a, b) = cx.func.insts[cx.def(v)?.index()].op else { return None };
    let ((x1, sh1), (x2, sh2)) = match_shift_pair(cx, a, b)?;
    let (c1, c2) = (cx.sconst(sh1)?, cx.sconst(sh2)?);
    if c1 <= 0 || c2 <= 0 || c1 + c2 != 32 { return None; }
    let (k1, k2) = (c1 as u32, c2 as u32);
    let hi_mask = ((1_u64 << k2) - 1) << 32;
    if (cx.known(x2).0 & hi_mask) != hi_mask { return None; }
    let ty = cx.func.ty(x1);
    let (s1, s2) = (strip_rot_mask(cx, ty, x1, dst.unsigned_max() >> k1), strip_rot_mask(cx, ty, x2, dst.unsigned_max() ^ ((1_u64 << k2) - 1)));
    let peel = |val: Value| match cx.def(val).map(|i| &cx.func.insts[i.index()].op) {
        Some(&Op::Convert(Convert::Uextend, inner)) => {
            if cx.func.ty(inner) == dst {
                Ok(inner)
            } else if let Some(ii) = cx.def(inner)
                && let Op::Convert(Convert::Ireduce, raw) = cx.func.insts[ii.index()].op
                && cx.func.ty(raw) == ty
            {
                Err(raw)
            } else {
                Err(val)
            }
        }
        _ => Err(val),
    };
    let base = match (peel(s1), peel(s2)) {
        (Ok(u1), Ok(u2)) if u1 == u2 => u1,
        (Err(r1), Err(r2)) if r1 == r2 => cx.emit(dst, Op::Convert(Convert::Ireduce, r1)),
        _ => return None,
    };
    let sh = cx.emit(dst, Op::Const(i64::from(k2)));
    Some(Op::Binary(Binary::Rotr, base, sh))
}

fn simplify_rotate(cx: &mut RewriteCx, bin: Binary, a: Value, b: Value) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !matches!(ty, Type::I32 | Type::I64) { return None; }
    let bits = ty.bits();
    let ((x1, sh1), (x2, sh2)) = match_shift_pair(cx, a, b)?;
    if let (Some(c1), Some(c2)) = (cx.sconst(sh1), cx.sconst(sh2))
        && let (k1, k2) = ((c1 as u32) & (bits - 1), (c2 as u32) & (bits - 1)) && k1 > 0 && k2 > 0 && k1 + k2 == bits
        && let (ux1, ux2) = (strip_rot_mask(cx, ty, x1, ty.unsigned_max() >> k1), strip_rot_mask(cx, ty, x2, ty.unsigned_max() ^ ((1_u64 << k2) - 1)))
        && ux1 == ux2
    {
        let sh_val = if c2 as u32 == k2 { sh2 } else { cx.emit(ty, Op::Const(i64::from(k2))) };
        return Some(Op::Binary(Binary::Rotr, ux1, sh_val));
    }
    if bin == Binary::Bor
        && let (ux1, ux2) = (strip_rot_mask(cx, ty, x1, ty.unsigned_max()), strip_rot_mask(cx, ty, x2, ty.unsigned_max()))
        && ux1 == ux2
    {
        let ((b1, n1), (b2, n2)) = (unwrap_shift_arg(cx, sh1, bits), unwrap_shift_arg(cx, sh2, bits));
        if b1 == b2 && n1 != n2 {
            return Some(Op::Binary(if n1 { Binary::Rotr } else { Binary::Rotl }, ux1, if n1 { sh2 } else { sh1 }));
        }
    }
    None
}

fn simplify_binary(cx: &mut RewriteCx, fits_used: &[bool], block_udivs: &mut Vec<(Value, Value, Value)>, res: Option<Value>, (bin, a, b): (Binary, Value, Value)) -> Option<Op> {
    match bin {
        Binary::Iadd => negated_shift_add(cx, cx.func.ty(a), a, b).map(|(base, idx, shift)| {
            let ty = cx.func.ty(base);
            let k = cx.emit(ty, Op::Const(shift));
            Op::Binary(Binary::Isub, base, cx.emit(ty, Op::Binary(Binary::Ishl, idx, k)))
        }).or_else(|| reassociate_add(cx, fits_used, a, b)).or_else(|| simplify_rotate(cx, bin, a, b)),
        Binary::Isub => reassociate_sub(cx, fits_used, a, b),
        Binary::Band => simplify_band(cx, res, a, b),
        Binary::Bor => simplify_rotate(cx, bin, a, b).or_else(|| icmp::fuse_cmp_ops(cx, res, false, a, b)).or_else(|| merge_fields(cx, res, a, b)),
        Binary::Bxor => simplify_bxor(cx, a, b).or_else(|| simplify_rotate(cx, bin, a, b)),
        Binary::Ushr | Binary::Ishl => simplify_shift(cx, res, bin, a, b),
        Binary::Udiv | Binary::Urem => simplify_div_rem(cx, block_udivs, res, (bin, a, b)),
        _ => None,
    }
}

fn simplify_inst(cx: &mut RewriteCx, fits_used: &[bool], block_udivs: &mut Vec<(Value, Value, Value)>, res: Option<Value>, op: Op) -> Option<Op> {
    match op {
        Op::Icmp(cc, x, y) => simplify_icmp(cx, res, cc, x, y),
        Op::Binary(bin, a, b) => {
            if res.is_some_and(|r| cx.func.ty(r).is_vector()) {
                vec::simplify_vec_binary(cx, res, bin, a, b)
            } else {
                simplify_binary(cx, fits_used, block_udivs, res, (bin, a, b))
            }
        }
        Op::Convert(Convert::Bitcast, x) => vec::simplify_bitcast(cx, res, x),
        Op::Convert(kind, x) => simplify_convert(cx, res, kind, x),
        Op::Insert(v, x, lane) => vec::simplify_insert(cx, res, v, x, lane),
        Op::Extract(v, lane) => vec::simplify_extract(cx, res, v, lane),
        Op::Shuffle(a, b, ShuffleMask::Const(m)) => vec::simplify_shuffle(cx, res, a, b, m),
        Op::Select(cond, t, f) => select::simplify_select(cx, res, cond, t, f),
        Op::Brif(cond, tb, ta, eb, ea) => known_cond(cx, cond).or_else(|| match_swar_msb_broadcast(cx, cond)).or_else(|| { let inner = cx.unwrap_uextend(cond); (inner != cond).then_some(inner) }).map(|c| Op::Brif(c, tb, ta, eb, ea)),
        Op::Load(addr, off) if !cx.func.rodata_bytes.is_empty() && off >= 0 => res.and_then(|r| fold_rodata_load(cx, cx.func.ty(r), addr, off)),
        _ => None,
    }
}

/// Rewrites constant unsigned divisions/remainders, unsigned zero compares, negated shift-adds, converts, shifts, selects, and bitwise/SWAR masks.
pub(crate) fn rewrite(func: &mut Function, initial_defs: &[Option<Inst>]) {
    let len = func.types.len();
    let (mut consts, mut fits_used) = (vec![None; len], vec![false; len]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::Const(imm), Some(res)) = (&data.op, data.result) {
            consts[res.index()] = Some(*imm);
        }
        if let Op::Fits(_, _, x, y) = data.op {
            fits_used[x.index()] = true;
            fits_used[y.index()] = true;
        }
    }
    let uses = func.use_counts();
    let mut cx = RewriteCx { func, defs: initial_defs.to_vec(), consts, uses, out: Vec::new(), subst: Vec::new() };
    for b in 0..cx.func.blocks.len() {
        let old = std::mem::take(&mut cx.func.blocks[b].insts);
        cx.out = Vec::with_capacity(old.len());
        let mut block_udivs = Vec::new();
        for inst in old {
            if !cx.subst.is_empty() {
                cx.func.insts[inst.index()].op.map_operands(|v| crate::merge::find(&cx.subst, v));
            }
            let (op, res) = (cx.func.insts[inst.index()].op.clone(), cx.func.insts[inst.index()].result);
            if let Some(new_op) = simplify_inst(&mut cx, &fits_used, &mut block_udivs, res, op) {
                if let (Op::Const(imm), Some(r)) = (&new_op, res) {
                    cx.consts.resize(cx.func.types.len(), None);
                    cx.consts[r.index()] = Some(*imm);
                }
                cx.func.insts[inst.index()].op = new_op;
            }
            cx.out.push(inst);
        }
        cx.func.blocks[b].insts = std::mem::take(&mut cx.out);
    }
    if !cx.subst.is_empty() {
        for inst in cx.func.blocks.iter().flat_map(|b| &b.insts) {
            cx.func.insts[inst.index()].op.map_operands(|v| crate::merge::find(&cx.subst, v));
        }
    }
}

/// Devirtualizes loads of static vtable slots into direct slot/symbol loads.
pub(crate) fn devirt_vtables(func: &mut Function, defs: &[Option<Inst>]) -> bool {
    if func.vtable_slots.is_empty() { return false; }
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
                let slot_inst = func.add(if low { Op::SlotAddr(slot_sym) } else { Op::Symbol(slot_sym) }, Some(slot_val));
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
    let Some(inst) = defs.get(value.index()).copied().flatten() else { return false };
    let Op::Const(imm) = func.insts[inst.index()].op else { return false };
    match func.ty(value) { Type::F64 => imm as u64 == 0x4000_0000_0000_0000, Type::F32 => imm as u32 == 0x4000_0000, _ => false }
}
