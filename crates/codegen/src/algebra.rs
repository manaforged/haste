use crate::fold::extend;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

mod icmp;

struct RewriteCx<'a> {
    func: &'a mut Function,
    defs: Vec<Option<Inst>>,
    consts: Vec<Option<i64>>,
    out: Vec<Inst>,
    subst: Vec<Value>,
}

impl<'a> RewriteCx<'a> {
    fn def(&self, v: Value) -> Option<Inst> { self.defs.get(v.index()).copied().flatten() }
    fn sconst(&self, v: Value) -> Option<i64> { self.consts.get(v.index()).copied().flatten() }
    fn uconst(&self, v: Value, ty: Type) -> Option<u64> { self.sconst(v).map(|c| extend(c, ty, false) as u64) }
    fn known(&self, v: Value) -> (u64, u64) { crate::fold::known_bits(self.func, &|x| self.def(x), &|x| self.sconst(x), v, 0) }
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

enum Masked {
    Const(u64),
    And(Value, u64),
}

fn simplify_masked(cx: &RewriteCx, v: Value, mask: u64, depth: u8) -> Option<Masked> {
    if mask == 0 {
        return Some(Masked::Const(0));
    }
    let (kz, ko) = cx.known(v);
    if (kz & mask) == mask {
        return Some(Masked::Const(0));
    }
    if (ko & mask) == mask {
        return Some(Masked::Const(mask));
    }
    if depth >= 6 {
        return None;
    }
    let inst = cx.def(v)?;
    match cx.func.insts[inst.index()].op {
        Op::Binary(bin @ (Binary::Bor | Binary::Bxor | Binary::Band), a, b) => {
            [(a, b), (b, a)].into_iter().find_map(|(keep, drop)| {
                let bits = if bin == Binary::Band { cx.known(drop).1 } else { cx.known(drop).0 };
                ((bits & mask) == mask).then(|| simplify_masked(cx, keep, mask, depth + 1).unwrap_or(Masked::And(keep, mask)))
            })
        }
        Op::Unary(Unary::Bnot, inner) => {
            let Op::Binary(Binary::Bxor, x, y) = cx.func.insts[cx.def(inner)?.index()].op else { return None };
            [(x, y), (y, x)].into_iter().find_map(|(src, cv)| {
                let c = cx.uconst(cv, cx.func.ty(v))?;
                ((c & mask) == mask).then(|| simplify_masked(cx, src, mask, depth + 1).unwrap_or(Masked::And(src, mask)))
            })
        }
        _ => None,
    }
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

fn swap_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal | IntCC::NotEqual => cc,
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
    }
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
    for (norm, ext_v, c_v) in [(cc, x, y), (swap_cc(cc), y, x)] {
        let Some(c_imm) = cx.sconst(c_v) else { continue };
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
                if let Some(folded) = icmp::eval_select(cx, res, ncc, nx, ny) {
                    return folded;
                }
                Some(zero_compare(cx, ncc, nx, ny).unwrap_or(Op::Icmp(ncc, nx, ny)))
            }
            other => Some(other),
        };
    }
    if let Some(folded) = icmp::eval_select(cx, res, cc, x, y) {
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
        if mask == 0x8080_8080_8080_8080 && let Some(x) = match_swar_msb_broadcast(cx, lhs) {
            return Some(Op::Binary(Binary::Band, x, mask_v));
        }
        if let Some(m) = simplify_masked(cx, lhs, mask, 0) {
            return match m {
                Masked::Const(c) => Some(Op::Const(extend(c as i64, ty, true))),
                Masked::And(src, nm) if cx.redundant_mask(src, ty, nm) => { cx.alias(res, src); None }
                Masked::And(src, nm) => Some(Op::Binary(Binary::Band, src, if nm == mask { mask_v } else { cx.emit(ty, Op::Const(extend(nm as i64, ty, true))) })),
            };
        }
    }
    None
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

fn strip_dead_bits(cx: &RewriteCx, mut cur: Value, mask: u64) -> Value {
    for _ in 0..4 {
        let Some(inst) = cx.def(cur) else { break };
        cur = match cx.func.insts[inst.index()].op {
            Op::Binary(Binary::Bor | Binary::Bxor, a, b) if (cx.known(b).0 & mask) == mask => a,
            Op::Binary(Binary::Bor | Binary::Bxor, a, b) if (cx.known(a).0 & mask) == mask => b,
            Op::Binary(Binary::Band, a, b) if (cx.known(b).1 & mask) == mask => a,
            Op::Binary(Binary::Band, a, b) if (cx.known(a).1 & mask) == mask => b,
            _ => break,
        };
    }
    cur
}

fn simplify_ireduce(cx: &mut RewriteCx, res: Option<Value>, dst: Type, x: Value) -> Option<Op> {
    let cur = strip_dead_bits(cx, x, dst.unsigned_max());
    if let Some(imm) = cx.sconst(cur) { return Some(Op::Const(extend(imm, dst, true))); }
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
    (cur != x).then_some(Op::Convert(Convert::Ireduce, cur))
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
    let k1 = (cx.sconst(sh1)? as u32) & (ty.bits() - 1);
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
        let new_ctz = cx.emit(Type::I64, Op::Unary(Unary::Ctz, word));
        return Some(Op::Binary(Binary::Ushr, new_ctz, sh));
    }
    let ty = cx.func.ty(x);
    let raw_k = cx.sconst(sh).filter(|_| ty.is_small_int())?;
    let k = (raw_k as u32) & (ty.bits() - 1);
    if k == 0 { cx.alias(res, x); return None; }
    let w = ty.unsigned_max();
    let survive = if op == Binary::Ushr { w & !((1_u64 << k) - 1) } else { w >> k };
    let cur = strip_dead_bits(cx, x, survive);
    if let Some(imm) = cx.sconst(cur) { return crate::fold::binary(op, ty, imm, raw_k).map(|v| Op::Const(extend(v, ty, true))); }
    fold_nested_shift(cx, res, (op, ty), (cur, sh), k).or_else(|| (cur != x).then_some(Op::Binary(op, cur, sh)))
}

fn simplify_select(cx: &mut RewriteCx, res: Option<Value>, cond: Value, t: Value, f: Value) -> Option<Op> {
    if t == f { cx.alias(res, t); return None; }
    let inner = cx.unwrap_uextend(cond);
    if res.is_some_and(|r| cx.func.ty(r) == Type::I8) && cx.func.ty(inner) == Type::I8 && cx.is_unit_bool(inner) {
        match (cx.uconst(t, Type::I8), cx.uconst(f, Type::I8)) {
            (Some(1), Some(0)) => { cx.alias(res, inner); return None; }
            (Some(1), _) if cx.is_unit_bool(f) => return Some(Op::Binary(Binary::Bor, inner, f)),
            (_, Some(0)) if cx.is_unit_bool(t) => return Some(Op::Binary(Binary::Band, inner, t)),
            (Some(0), Some(1)) => return Some(Op::Binary(Binary::Bxor, inner, f)),
            _ => {}
        }
    }
    (inner != cond).then_some(Op::Select(inner, t, f))
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

fn simplify_binary(cx: &mut RewriteCx, fits_used: &[bool], block_udivs: &mut Vec<(Value, Value, Value)>, res: Option<Value>, (bin, a, b): (Binary, Value, Value)) -> Option<Op> {
    match bin {
        Binary::Iadd => negated_shift_add(cx, cx.func.ty(a), a, b)
            .map(|(base, idx, shift)| {
                let ty = cx.func.ty(base);
                let k = cx.emit(ty, Op::Const(shift));
                let shl = cx.emit(ty, Op::Binary(Binary::Ishl, idx, k));
                Op::Binary(Binary::Isub, base, shl)
            })
            .or_else(|| reassociate_add(cx, fits_used, a, b)),
        Binary::Isub => reassociate_sub(cx, fits_used, a, b),
        Binary::Band => simplify_band(cx, res, a, b),
        Binary::Bxor => simplify_bxor(cx, a, b),
        Binary::Ushr | Binary::Ishl => simplify_shift(cx, res, bin, a, b),
        Binary::Udiv | Binary::Urem => simplify_div_rem(cx, block_udivs, res, (bin, a, b)),
        _ => None,
    }
}

fn simplify_inst(cx: &mut RewriteCx, fits_used: &[bool], block_udivs: &mut Vec<(Value, Value, Value)>, res: Option<Value>, op: Op) -> Option<Op> {
    match op {
        Op::Icmp(cc, x, y) => simplify_icmp(cx, res, cc, x, y),
        Op::Binary(bin, a, b) => simplify_binary(cx, fits_used, block_udivs, res, (bin, a, b)),
        Op::Convert(kind, x) => simplify_convert(cx, res, kind, x),
        Op::Select(cond, t, f) => simplify_select(cx, res, cond, t, f),
        Op::Brif(cond, tb, ta, eb, ea) => match_swar_msb_broadcast(cx, cond)
            .or_else(|| { let inner = cx.unwrap_uextend(cond); (inner != cond).then_some(inner) })
            .map(|c| Op::Brif(c, tb, ta, eb, ea)),
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
    let mut cx = RewriteCx { func, defs: initial_defs.to_vec(), consts, out: Vec::new(), subst: Vec::new() };
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

struct BoolFacts {
    compares: Vec<Option<(IntCC, Value, Value)>>,
    consts: Vec<Option<i64>>,
    xors: Vec<Option<(Value, Value)>>,
    uextends: Vec<Option<Value>>,
    bools: Vec<bool>,
}

impl BoolFacts {
    fn collect(func: &Function) -> Self {
        let n = func.types.len();
        let mut f = Self { compares: vec![None; n], consts: vec![None; n], xors: vec![None; n], uextends: vec![None; n], bools: vec![false; n] };
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let d = &func.insts[inst.index()];
            match (&d.op, d.result) {
                (&Op::Icmp(cc, x, y), Some(r)) if func.ty(r) == Type::I8 => { f.compares[r.index()] = Some((cc, x, y)); f.bools[r.index()] = true; }
                (&Op::Fcmp(..) | &Op::Fits(..), Some(r)) if func.ty(r) == Type::I8 => f.bools[r.index()] = true,
                (&Op::Const(imm), Some(r)) if func.ty(r).is_small_int() => f.consts[r.index()] = Some(imm),
                (&Op::Binary(Binary::Bxor, x, y), Some(r)) => f.xors[r.index()] = Some((x, y)),
                (&Op::Convert(Convert::Uextend, inner), Some(r)) => f.uextends[r.index()] = Some(inner),
                _ => {}
            }
        }
        f.propagate(func);
        f
    }

    fn propagate(&mut self, func: &Function) {
        let is_01 = |v: Value, bools: &[bool], consts: &[Option<i64>]| bools[v.index()] || matches!(consts[v.index()], Some(0 | 1));
        for _ in 0..2 {
            for inst in func.blocks.iter().flat_map(|block| &block.insts) {
                let d = &func.insts[inst.index()];
                let Some(r) = d.result else { continue };
                self.bools[r.index()] |= match d.op {
                    Op::Binary(Binary::Band, x, y) => matches!(self.consts[x.index()], Some(0 | 1)) || matches!(self.consts[y.index()], Some(0 | 1)) || (is_01(x, &self.bools, &self.consts) && is_01(y, &self.bools, &self.consts)),
                    Op::Binary(Binary::Bor | Binary::Bxor, x, y) | Op::Select(_, x, y) => is_01(x, &self.bools, &self.consts) && is_01(y, &self.bools, &self.consts),
                    Op::Convert(Convert::Uextend, inner) => self.bools[inner.index()],
                    _ => false,
                };
            }
        }
    }

    fn unwrap_uext(&self, mut v: Value) -> Value {
        while let Some(inner) = self.uextends.get(v.index()).copied().flatten() { v = inner; }
        v
    }

    fn xor_one_src(&self, x: Value, y: Value) -> Option<Value> {
        if self.consts[y.index()] == Some(1) && self.bools[x.index()] { Some(x) }
        else if self.consts[x.index()] == Some(1) && self.bools[y.index()] { Some(y) }
        else { None }
    }
}

fn negated(op: &Op, facts: &BoolFacts) -> Option<(Value, Option<Value>)> {
    let is = |v: Value, want: i64| facts.consts[v.index()] == Some(want);
    match *op {
        Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) if is(y, 1) => Some((x, Some(y))),
        Op::Binary(Binary::Bxor, y, x) | Op::Icmp(IntCC::NotEqual, y, x) if is(y, 1) => Some((x, Some(y))),
        Op::Icmp(IntCC::Equal, x, y) if is(y, 0) => Some((x, None)),
        Op::Icmp(IntCC::Equal, y, x) if is(y, 0) => Some((x, None)),
        Op::Select(c, x, y) if is(x, 0) && is(y, 1) && facts.bools[c.index()] => Some((c, Some(y))),
        _ => None,
    }
}

fn invert_brif(op: &mut Op, facts: &BoolFacts, types: &[Type]) {
    let Op::Brif(cond, then, then_args, otherwise, else_args) = op else { return };
    if let Some((x, y)) = facts.xors[cond.index()] && let Some(inner) = facts.xor_one_src(x, y) {
        *cond = inner;
        std::mem::swap(then, otherwise);
        std::mem::swap(then_args, else_args);
        return;
    }
    let Some((cc, x, y)) = facts.compares[cond.index()] else { return };
    let (val, flipped) = if facts.consts[y.index()] == Some(0) && types[x.index()].is_small_int() { (x, false) }
        else if facts.consts[x.index()] == Some(0) && types[y.index()].is_small_int() { (y, true) }
        else { return };
    let invert_arms = matches!((flipped, cc), (_, IntCC::Equal) | (false, IntCC::UnsignedLessThanOrEqual) | (true, IntCC::UnsignedGreaterThanOrEqual));
    if invert_arms || matches!((flipped, cc), (_, IntCC::NotEqual) | (false, IntCC::UnsignedGreaterThan) | (true, IntCC::UnsignedLessThan)) {
        *cond = val;
        if invert_arms { std::mem::swap(then, otherwise); std::mem::swap(then_args, else_args); }
    }
}

fn invert_inst(op: &mut Op, res: Option<Value>, facts: &BoolFacts, types: &[Type]) {
    if let (Some(res), Some((inner, opt_one))) = (res, negated(op, facts)) {
        let res_ty = types[res.index()];
        if res_ty == Type::I8 && let Some((cc, a, b)) = facts.compares[inner.index()] { *op = Op::Icmp(cc.inverse(), a, b); return; }
        if facts.bools[inner.index()] && let Some(one) = opt_one && types[inner.index()] == res_ty && types[one.index()] == res_ty && !matches!(*op, Op::Binary(Binary::Bxor, ..)) {
            *op = Op::Binary(Binary::Bxor, inner, one);
            return;
        }
    }
    invert_brif(op, facts, types);
}

/// Rewrites a negation of a compare, `x ^ 1`, `x == 0`, `x != 1`, or `select(x, 0, 1)`, as the inverse compare or `x ^ 1`, and simplifies branches on `x == 0` / `x != 0`.
pub(crate) fn invert(func: &mut Function) {
    let facts = BoolFacts::collect(func);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let res = func.insts[inst.index()].result;
        invert_inst(&mut func.insts[inst.index()].op, res, &facts, &func.types);
    }
    sink_brif_cmp(func);
}

fn xor_ne_pair(func: &Function, facts: &BoolFacts, x: Value, y: Value, allow_const: bool) -> Option<(Value, Value)> {
    if allow_const && let Some(pair) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.consts[c.index()].is_some()) {
        return Some(pair);
    }
    if facts.bools[x.index()] && facts.bools[y.index()] {
        let (a, b) = (facts.unwrap_uext(x), facts.unwrap_uext(y));
        return (func.ty(a) == func.ty(b) && func.ty(a).is_small_int()).then_some((a, b));
    }
    None
}

fn fold_bxor_select(func: &mut Function, facts: &BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Select(cond, t_val, f_val) = func.insts[inst.index()].op else { return false };
    let inner = facts.unwrap_uext(cond);
    if let Some((x, y)) = facts.xors[inner.index()] {
        if let Some((lhs, _)) = [(x, y), (y, x)].into_iter().find(|&(_, c)| facts.consts[c.index()] == Some(1))
            && let bool_src = facts.unwrap_uext(lhs) && facts.bools[bool_src.index()]
        {
            func.insts[inst.index()].op = Op::Select(bool_src, f_val, t_val);
            return true;
        }
        if let Some((a, b)) = xor_ne_pair(func, facts, x, y, false) {
            let cmp_val = func.value(Type::I8);
            out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
            func.insts[inst.index()].op = Op::Select(cmp_val, t_val, f_val);
            return true;
        }
    }
    if inner != cond { func.insts[inst.index()].op = Op::Select(inner, t_val, f_val); return true; }
    false
}

fn fold_bxor_brif(func: &mut Function, facts: &BoolFacts, out: &mut Vec<Inst>, inst: Inst) -> bool {
    let Op::Brif(cond, ..) = func.insts[inst.index()].op else { return false };
    let inner = facts.unwrap_uext(cond);
    if let Some((x, y)) = facts.xors[inner.index()]
        && let Some((a, b)) = xor_ne_pair(func, facts, x, y, true) && func.ty(a) == func.ty(b) && func.ty(a).is_small_int()
    {
        let cmp_val = func.value(Type::I8);
        out.push(func.add(Op::Icmp(IntCC::NotEqual, a, b), Some(cmp_val)));
        if let Op::Brif(c_mut, ..) = &mut func.insts[inst.index()].op { *c_mut = cmp_val; }
        return true;
    }
    if inner != cond && let Op::Brif(c_mut, ..) = &mut func.insts[inst.index()].op { *c_mut = inner; return true; }
    false
}

pub(crate) fn fold_bxor_conds(func: &mut Function) -> bool {
    let facts = BoolFacts::collect(func);
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts { changed |= fold_bxor_select(func, &facts, &mut out, inst) || fold_bxor_brif(func, &facts, &mut out, inst); out.push(inst); }
        func.blocks[b_idx].insts = out;
    }
    if changed { sink_brif_cmp(func); }
    changed
}

fn sink_brif_cmp(func: &mut Function) {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        func.insts[inst.index()].op.for_operands(|v| uses[v.index()] += 1);
    }
    for block in &mut func.blocks {
        let len = block.insts.len();
        if len < 2 {
            continue;
        }
        let Op::Brif(cond, ..) = func.insts[block.insts[len - 1].index()].op else { continue };
        if uses[cond.index()] != 1 {
            continue;
        }
        if let Some(cpos) = block.insts[..len - 1].iter().rposition(|&inst| {
            let data = &func.insts[inst.index()];
            data.result == Some(cond) && matches!(data.op, Op::Icmp(..) | Op::Fcmp(..))
        }) && cpos + 1 < len - 1
        {
            let cinst = block.insts.remove(cpos);
            block.insts.insert(len - 2, cinst);
        }
    }
}
