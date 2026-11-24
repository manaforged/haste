use super::vector;
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

/// `imm` of type `ty` extended to 64 bits, by its sign when `signed`.
pub(crate) fn extend(imm: i64, ty: Type, signed: bool) -> i64 {
    let bits = ty.bits();
    if bits >= 64 {
        return imm;
    }
    let shift = 64 - bits;
    if signed {
        imm << shift >> shift
    } else {
        ((imm as u64) << shift >> shift) as i64
    }
}

/// The base and addend of `op` when it is `base + c`, `c + base` or `base - c` with `c` known to `konst`.
pub(crate) fn const_offset(op: &Op, konst: impl Fn(Value) -> Option<i64>) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, x, y) => [(x, y), (y, x)]
            .into_iter()
            .find_map(|(base, c)| konst(c).map(|c| (base, c))),
        Op::Binary(Binary::Isub, x, y) => konst(y).map(|c| (x, c.wrapping_neg())),
        _ => None,
    }
}

/// Whether `imm` fits a sign-extended 32-bit immediate.
pub(crate) fn fits_imm32(imm: i64) -> bool {
    i32::try_from(imm).is_ok()
}

/// Whether `cc` holds between the constants `x` and `y` of type `ty`.
pub(crate) fn holds(cc: IntCC, ty: Type, x: i64, y: i64) -> bool {
    let (x, y) = (extend(x, ty, cc.is_signed()), extend(y, ty, cc.is_signed()));
    match cc {
        IntCC::Equal => x == y,
        IntCC::NotEqual => x != y,
        IntCC::SignedLessThan => x < y,
        IntCC::SignedGreaterThanOrEqual => x >= y,
        IntCC::SignedGreaterThan => x > y,
        IntCC::SignedLessThanOrEqual => x <= y,
        IntCC::UnsignedLessThan => (x as u64) < y as u64,
        IntCC::UnsignedGreaterThanOrEqual => x as u64 >= y as u64,
        IntCC::UnsignedGreaterThan => (x as u64) > y as u64,
        IntCC::UnsignedLessThanOrEqual => (x as u64) <= y as u64,
    }
}

/// `a op b` on two constants, for the operations that wrap the same at every width.
pub(crate) fn constant(op: Binary, a: i64, b: i64) -> Option<i64> {
    match op {
        Binary::Iadd => Some(a.wrapping_add(b)),
        Binary::Isub => Some(a.wrapping_sub(b)),
        Binary::Imul => Some(a.wrapping_mul(b)),
        Binary::Band => Some(a & b),
        Binary::Bor => Some(a | b),
        Binary::Bxor => Some(a ^ b),
        _ => None,
    }
}

/// `a op b` on constants of type `ty`, with shift counts masked to the width as the encoders mask them, or `None` for a division that traps.
pub(crate) fn binary(op: Binary, ty: Type, a: i64, b: i64) -> Option<i64> {
    if let Some(value) = constant(op, a, b) {
        return Some(value);
    }
    let bits = ty.bits();
    let count = (b & i64::from(bits - 1)) as u32;
    let (sa, sb) = (extend(a, ty, true), extend(b, ty, true));
    let (ua, ub) = (extend(a, ty, false) as u64, extend(b, ty, false) as u64);
    match op {
        Binary::Ishl => Some(a << count),
        Binary::Ushr => Some((ua >> count) as i64),
        Binary::Sshr => Some(sa >> count),
        Binary::Rotl => Some(vector::rotate(ty, ua, count)),
        Binary::Rotr => Some(vector::rotate(ty, ua, bits - count)),
        Binary::Smin => Some(sa.min(sb)),
        Binary::Smax => Some(sa.max(sb)),
        Binary::Umulhi => Some((u128::from(ua) * u128::from(ub) >> bits) as i64),
        Binary::Smulhi => Some((i128::from(sa) * i128::from(sb) >> bits) as i64),
        Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => vector::divide(op, ty, a, b),
        Binary::SaddSat
        | Binary::UaddSat
        | Binary::SsubSat
        | Binary::UsubSat
        | Binary::AvgRound => vector::sat_binary(op, ty, sa, sb, ua, ub),
        _ => None,
    }
}

fn unary(op: Unary, ty: Type, a: i64) -> Option<i64> {
    let bits = ty.bits();
    let unused = 64 - bits;
    let value = extend(a, ty, false) as u64;
    match op {
        Unary::Ineg => Some(a.wrapping_neg()),
        Unary::Bnot => Some(!a),
        Unary::Bswap => Some((value.swap_bytes() >> unused) as i64),
        Unary::Bitrev => Some((value.reverse_bits() >> unused) as i64),
        Unary::Clz => Some(i64::from(value.leading_zeros() - unused)),
        Unary::Ctz => Some(i64::from(value.trailing_zeros().min(bits))),
        Unary::Popcnt => Some(i64::from(value.count_ones())),
        _ => None,
    }
}

/// The constant that leaves the left operand of `op` unchanged when it stands on the right.
pub(super) fn unit(op: Binary) -> Option<i64> {
    match op {
        Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor => Some(0),
        Binary::Imul | Binary::Sdiv | Binary::Udiv => Some(1),
        Binary::Band => Some(-1),
        _ => None,
    }
}

/// Whether `x op y` on constants of type `ty` fits that type, read as signed when `signed`.
pub(crate) fn fits(op: Binary, signed: bool, ty: Type, (x, y): (i64, i64)) -> Option<bool> {
    let widen = |value: i64| {
        if signed {
            i128::from(extend(value, ty, true))
        } else {
            i128::from(extend(value, ty, false) as u64)
        }
    };
    let (x, y) = (widen(x), widen(y));
    let result = match op {
        Binary::Iadd => x + y,
        Binary::Isub => x - y,
        Binary::Imul => x * y,
        _ => return None,
    };
    let bits = ty.bits();
    let (low, high) = if signed {
        (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
    } else {
        (0, (1_i128 << bits) - 1)
    };
    Some((low..=high).contains(&result))
}

/// `op` on small integers when `known` gives the constants it reads.
pub(crate) fn evaluate(
    func: &Function,
    op: &Op,
    known: &dyn Fn(Value) -> Option<i64>,
) -> Option<i64> {
    let (Op::Unary(_, x)
    | Op::Binary(_, x, _)
    | Op::Icmp(_, x, _)
    | Op::Convert(_, x)
    | Op::Fits(_, _, x, _)) = *op
    else {
        return if let Op::Const(imm) = *op {
            Some(imm)
        } else {
            None
        };
    };
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let both = |x: Value, y: Value| Some((known(x)?, known(y)?));
    match *op {
        Op::Unary(op, x) => unary(op, ty, known(x)?),
        Op::Binary(op, x, y) => both(x, y).and_then(|(a, b)| binary(op, ty, a, b)),
        Op::Icmp(cc, x, y) => both(x, y).map(|(a, b)| i64::from(holds(cc, ty, a, b))),
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => Some(extend(known(x)?, ty, false)),
        Op::Convert(Convert::Sextend, x) => Some(extend(known(x)?, ty, true)),
        Op::Fits(op, signed, x, y) => both(x, y)
            .and_then(|pair| fits(op, signed, ty, pair))
            .map(i64::from),
        _ => None,
    }
}

/// Evaluates `op` for `ty`, sign-wrapping small-integer results to the canonical `Op::Const` representation.
pub(crate) fn evaluate_wrapped(
    func: &Function,
    op: &Op,
    ty: Type,
    known: &dyn Fn(Value) -> Option<i64>,
) -> Option<i64> {
    if !ty.is_small_int() {
        return None;
    }
    if let Op::Convert(Convert::NarrowSat { signed }, x) = *op {
        return vector::narrow_sat(func, ty, signed, x, known);
    }
    evaluate(func, op, known).map(|v| extend(v, ty, true))
}

fn known_shift_bits(op: Binary, ty: Type, (za, oa): (u64, u64), count: i64) -> (u64, u64) {
    let w = ty.unsigned_max();
    let k = (count as u32) & (ty.bits() - 1);
    match op {
        Binary::Ushr => (
            ((za >> k) | if k == 0 { 0 } else { !0_u64 << (ty.bits() - k) }) & w,
            (oa >> k) & w,
        ),
        _ => (
            ((za << k) | if k == 0 { 0 } else { (1_u64 << k) - 1 }) & w,
            (oa << k) & w,
        ),
    }
}

/// Returns `(known_zero_bits, known_one_bits)` within `v`'s integer type width.
pub(crate) fn known_bits(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
    depth: u8,
) -> (u64, u64) {
    let ty = func.ty(v);
    if !ty.is_small_int() {
        return (0, 0);
    }
    let w = ty.unsigned_max();
    if let Some(imm) = known(v) {
        let u = extend(imm, ty, false) as u64;
        return (!u & w, u & w);
    }
    if depth >= 6 {
        return (0, 0);
    }
    let Some(inst) = def_of(v) else {
        return (0, 0);
    };
    let rec = |u: Value| known_bits(func, def_of, known, u, depth + 1);
    match func.insts[inst.index()].op {
        Op::Unary(Unary::Bnot, a) => {
            let (z, o) = rec(a);
            (o & w, z & w)
        }
        Op::Binary(Binary::Band, a, b) => {
            let ((za, oa), (zb, ob)) = (rec(a), rec(b));
            ((za | zb) & w, oa & ob & w)
        }
        Op::Binary(Binary::Bor, a, b) => {
            let ((za, oa), (zb, ob)) = (rec(a), rec(b));
            (za & zb & w, (oa | ob) & w)
        }
        Op::Binary(Binary::Bxor, a, b) => {
            let ((za, oa), (zb, ob)) = (rec(a), rec(b));
            (((za & zb) | (oa & ob)) & w, ((za & ob) | (oa & zb)) & w)
        }
        Op::Binary(op @ (Binary::Ushr | Binary::Ishl), a, sh) => {
            known(sh).map_or((0, 0), |c| known_shift_bits(op, ty, rec(a), c))
        }
        Op::Binary(Binary::Imul, a, b) => {
            let tz = (rec(a).0.trailing_ones() + rec(b).0.trailing_ones()).min(ty.bits());
            (
                if tz == 64 {
                    u64::MAX
                } else {
                    ((1_u64 << tz) - 1) & w
                },
                0,
            )
        }
        Op::Convert(Convert::Uextend, a) if func.ty(a).is_small_int() => {
            let ((za, oa), aw) = (rec(a), func.ty(a).unsigned_max());
            ((za | !aw) & w, oa & aw)
        }
        Op::Convert(Convert::Ireduce, a) if func.ty(a).is_small_int() => {
            let (za, oa) = rec(a);
            (za & w, oa & w)
        }
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => (w & !1, 0),
        Op::Select(_, a, b) => {
            let ((za, oa), (zb, ob)) = (rec(a), rec(b));
            (za & zb & w, oa & ob & w)
        }
        _ => (0, 0),
    }
}

/// Whether `v` is proven to hold only `0` or `1` via `known_bits`.
pub(crate) fn is_unit_bool(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    v: Value,
) -> bool {
    let ty = func.ty(v);
    ty.is_small_int() && (known_bits(func, def_of, known, v, 0).0 | 1) == ty.unsigned_max()
}

const DEREF_SCAN: usize = 256;

/// Where a value is read: after `prefix` in its block, and below every block that dominates `block`.
#[derive(Clone, Copy)]
pub(crate) struct Before<'a> {
    pub(crate) prefix: &'a [Inst],
    pub(crate) block: Option<Block>,
}

/// Whether `v` is a non-null address where `at` reads it: by its definition, or because a load or store that ran before touched a byte range that holds or ends at `v`.
pub(crate) fn non_null_at(
    func: &Function,
    def_of: &dyn Fn(Value) -> Option<Inst>,
    known: &dyn Fn(Value) -> Option<i64>,
    at: Before,
    v: Value,
) -> bool {
    if crate::algebra::is_non_null_addr(func, def_of, known, v, 0) {
        return true;
    }
    if func.ty(v) != Type::I64 {
        return false;
    }
    let (base, disp) = match def_of(v).map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Binary(Binary::Iadd, a, b)) => known(b)
            .map(|c| (a, c))
            .or_else(|| known(a).map(|c| (b, c)))
            .unwrap_or((v, 0)),
        Some(&Op::Binary(Binary::Isub, a, b)) => known(b).map_or((v, 0), |c| (a, c.wrapping_neg())),
        _ => (v, 0),
    };
    let stop = def_of(base);
    let mut left = DEREF_SCAN;
    let check_slice = |insts: &[Inst], left: &mut usize| -> Option<bool> {
        for &inst in insts.iter().rev() {
            if *left == 0 {
                return Some(false);
            }
            *left -= 1;
            if stop == Some(inst) {
                return Some(false);
            }
            let data = &func.insts[inst.index()];
            let (addr, off, bytes) = match data.op {
                Op::Load(addr, off) => (
                    addr,
                    off,
                    data.result.map_or(0, |result| func.ty(result).bytes()),
                ),
                Op::Store(value, addr, off) => (addr, off, func.ty(value).bytes()),
                _ => continue,
            };
            let touched = i64::from(off)..=i64::from(off) + i64::from(bytes);
            if bytes != 0
                && ((addr == v && touched.contains(&0))
                    || (addr == base && touched.contains(&disp)))
            {
                return Some(true);
            }
        }
        None
    };
    if let Some(hit) = check_slice(at.prefix, &mut left) {
        return hit;
    }
    let Some(mut cur) = at.block else {
        return false;
    };
    let cfg = func.cfg();
    while let Some(parent) = cfg
        .idom
        .get(cur.index())
        .copied()
        .flatten()
        .filter(|p| *p != cur)
    {
        if let Some(hit) = check_slice(&func.blocks[parent.index()].insts, &mut left) {
            return hit;
        }
        cur = parent;
    }
    false
}
