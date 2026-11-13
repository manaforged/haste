use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

mod vector;

/// `imm` of type `ty` extended to 64 bits, by its sign when `signed`.
pub(crate) fn extend(imm: i64, ty: Type, signed: bool) -> i64 {
    let bits = ty.bits();
    if bits >= 64 {
        return imm;
    }
    let shift = 64 - bits;
    match signed {
        true => imm << shift >> shift,
        false => ((imm as u64) << shift >> shift) as i64,
    }
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
        IntCC::UnsignedGreaterThan => x as u64 > y as u64,
        IntCC::UnsignedLessThanOrEqual => x as u64 <= y as u64,
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
        Binary::Rotl => Some(rotate(ty, ua, count)),
        Binary::Rotr => Some(rotate(ty, ua, bits - count)),
        Binary::Smin => Some(sa.min(sb)),
        Binary::Smax => Some(sa.max(sb)),
        Binary::Umulhi => Some((u128::from(ua) * u128::from(ub) >> bits) as i64),
        Binary::Smulhi => Some((i128::from(sa) * i128::from(sb) >> bits) as i64),
        Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => divide(op, ty, a, b),
        _ => None,
    }
}

/// The low bits of `value` of type `ty` rotated left by `left` modulo the width.
fn rotate(ty: Type, value: u64, left: u32) -> i64 {
    match ty {
        Type::I8 => i64::from((value as u8).rotate_left(left)),
        Type::I16 => i64::from((value as u16).rotate_left(left)),
        Type::I32 => i64::from((value as u32).rotate_left(left)),
        _ => value.rotate_left(left) as i64,
    }
}

/// `a / b` or `a % b` on constants of type `ty`, or `None` when `b` is zero or the operation is the signed minimum by -1, where the machine traps.
fn divide(op: Binary, ty: Type, a: i64, b: i64) -> Option<i64> {
    let signed = matches!(op, Binary::Sdiv | Binary::Srem);
    let (x, y) = (extend(a, ty, signed), extend(b, ty, signed));
    if y == 0 || (signed && y == -1 && x == -1_i64 << (ty.bits() - 1)) {
        return None;
    }
    Some(match op {
        Binary::Sdiv => x / y,
        Binary::Srem => x % y,
        Binary::Udiv => (x as u64 / y as u64) as i64,
        _ => (x as u64 % y as u64) as i64,
    })
}

/// `op` of the constant `a` of type `ty`, counting bits at that width.
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
fn unit(op: Binary) -> Option<i64> {
    match op {
        Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor => Some(0),
        Binary::Imul => Some(1),
        Binary::Band => Some(-1),
        _ => None,
    }
}

/// Whether `x op y` on constants of type `ty` fits that type, read as signed when `signed`.
pub(crate) fn fits(op: Binary, signed: bool, ty: Type, (x, y): (i64, i64)) -> Option<bool> {
    let widen = |value: i64| if signed { i128::from(extend(value, ty, true)) } else { i128::from(extend(value, ty, false) as u64) };
    let (x, y) = (widen(x), widen(y));
    let result = match op {
        Binary::Iadd => x + y,
        Binary::Isub => x - y,
        Binary::Imul => x * y,
        _ => return None,
    };
    let bits = ty.bits();
    let (low, high) = if signed { (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1) } else { (0, (1_i128 << bits) - 1) };
    Some((low..=high).contains(&result))
}

/// `op` on small integers when `known` gives the constants it reads.
pub(crate) fn evaluate(func: &Function, op: &Op, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    let (Op::Unary(_, x) | Op::Binary(_, x, _) | Op::Icmp(_, x, _) | Op::Convert(_, x) | Op::Fits(_, _, x, _)) = *op else {
        return if let Op::Const(imm) = *op { Some(imm) } else { None };
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
        Op::Fits(op, signed, x, y) => both(x, y).and_then(|pair| fits(op, signed, ty, pair)).map(i64::from),
        _ => None,
    }
}

/// Evaluates `op` for `ty`, sign-wrapping small-integer results to the canonical `Op::Const` representation.
pub(crate) fn evaluate_wrapped(func: &Function, op: &Op, ty: Type, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    if !ty.is_small_int() {
        return None;
    }
    evaluate(func, op, known).map(|v| extend(v, ty, true))
}

fn known_shift_bits(op: Binary, ty: Type, (za, oa): (u64, u64), count: i64) -> (u64, u64) {
    let w = ty.unsigned_max();
    let k = (count as u32) & (ty.bits() - 1);
    match op {
        Binary::Ushr => (((za >> k) | if k == 0 { 0 } else { !0_u64 << (ty.bits() - k) }) & w, (oa >> k) & w),
        _ => (((za << k) | if k == 0 { 0 } else { (1_u64 << k) - 1 }) & w, (oa << k) & w),
    }
}

/// Returns `(known_zero_bits, known_one_bits)` within `v`'s integer type width.
pub(crate) fn known_bits(func: &Function, def_of: &dyn Fn(Value) -> Option<Inst>, known: &dyn Fn(Value) -> Option<i64>, v: Value, depth: u8) -> (u64, u64) {
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
    let Some(inst) = def_of(v) else { return (0, 0) };
    let rec = |u: Value| known_bits(func, def_of, known, u, depth + 1);
    match func.insts[inst.index()].op {
        Op::Unary(Unary::Bnot, a) => { let (z, o) = rec(a); (o & w, z & w) }
        Op::Binary(Binary::Band, a, b) => { let ((za, oa), (zb, ob)) = (rec(a), rec(b)); ((za | zb) & w, oa & ob & w) }
        Op::Binary(Binary::Bor, a, b) => { let ((za, oa), (zb, ob)) = (rec(a), rec(b)); (za & zb & w, (oa | ob) & w) }
        Op::Binary(Binary::Bxor, a, b) => {
            let ((za, oa), (zb, ob)) = (rec(a), rec(b));
            (((za & zb) | (oa & ob)) & w, ((za & ob) | (oa & zb)) & w)
        }
        Op::Binary(op @ (Binary::Ushr | Binary::Ishl), a, sh) => known(sh).map_or((0, 0), |c| known_shift_bits(op, ty, rec(a), c)),
        Op::Convert(Convert::Uextend, a) if func.ty(a).is_small_int() => {
            let ((za, oa), aw) = (rec(a), func.ty(a).unsigned_max());
            ((za | !aw) & w, oa & aw)
        }
        Op::Convert(Convert::Ireduce, a) if func.ty(a).is_small_int() => { let (za, oa) = rec(a); (za & w, oa & w) }
        Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => (w & !1, 0),
        Op::Select(_, a, b) => { let ((za, oa), (zb, ob)) = (rec(a), rec(b)); (za & zb & w, oa & ob & w) }
        _ => (0, 0),
    }
}

pub(super) struct Fold {
    consts: Vec<Option<i64>>,
    subst: Vec<Value>,
    incoming: Vec<Option<Value>>,
    mixed: Vec<bool>,
    defs: Vec<Option<Inst>>,
}

impl Fold {
    pub(super) fn find(&self, value: Value) -> Value {
        crate::merge::find(&self.subst, value)
    }

    pub(super) fn known(&self, value: Value) -> Option<i64> {
        self.consts[self.find(value).index()]
    }

    fn both(&self, x: Value, y: Value) -> Option<(i64, i64)> {
        Some((self.known(x)?, self.known(y)?))
    }

    fn evaluate(&self, func: &Function, op: &Op, ty: Type) -> Option<i64> {
        evaluate_wrapped(func, op, ty, &|value| self.known(value)).or_else(|| match *op {
            Op::Binary(op, x, y) if ty.is_small_int() => self.absorbed(op, ty, x, y).map(|v| extend(v, ty, true)),
            _ => None,
        })
    }

    /// The constant `x op y` of type `ty` gives whatever one operand holds: a product, high product or mask with zero, or the difference or xor of a value with itself.
    fn absorbed(&self, op: Binary, ty: Type, x: Value, y: Value) -> Option<i64> {
        let zero = |value: Value| self.known(value).is_some_and(|imm| extend(imm, ty, false) == 0);
        match op {
            Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Band if zero(x) || zero(y) => Some(0),
            Binary::Isub | Binary::Bxor if self.find(x) == self.find(y) => Some(0),
            _ => None,
        }
    }

    /// The operation of the instruction that defines `value`, when one in a block does.
    pub(super) fn def<'a>(&self, func: &'a Function, value: Value) -> Option<&'a Op> {
        self.defs[self.find(value).index()].map(|inst| &func.insts[inst.index()].op)
    }

    /// Whether `value` is 0 or 1 in its low byte because a comparison, overflow check, or boolean op defines it.
    fn boolean(&self, func: &Function, value: Value) -> bool {
        self.boolean_at(func, value, 0)
    }

    fn boolean_at(&self, func: &Function, value: Value, depth: u8) -> bool {
        if depth >= 8 || func.ty(value).is_vector() {
            return false;
        }
        let is_01 = |v: Value| matches!(self.known(v), Some(0 | 1)) || self.boolean_at(func, v, depth + 1);
        match self.def(func, value) {
            Some(Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..)) => true,
            Some(Op::Convert(Convert::Uextend, inner)) => self.boolean_at(func, *inner, depth + 1),
            Some(Op::Binary(Binary::Band, a, b)) => {
                matches!(self.known(*a), Some(0 | 1)) || matches!(self.known(*b), Some(0 | 1)) || (is_01(*a) && is_01(*b))
            }
            Some(Op::Binary(Binary::Bor | Binary::Bxor, a, b) | Op::Select(_, a, b)) => is_01(*a) && is_01(*b),
            _ => false,
        }
    }

    /// Whether `x` and `y` of type `ty` hold the same bits at that width.
    fn same(&self, ty: Type, x: Value, y: Value) -> bool {
        self.find(x) == self.find(y) || matches!(self.both(x, y), Some((a, b)) if extend(a, ty, false) == extend(b, ty, false))
    }

    /// The existing value the result of `op` always equals, when there is one.
    fn identity(&self, func: &Function, op: &Op) -> Option<Value> {
        match *op {
            Op::Select(condition, x, y) => {
                if self.find(x) == self.find(y) {
                    return Some(x);
                }
                if let Some(known) = self.known(condition) {
                    return Some(if known != 0 { x } else { y });
                }
                if func.ty(condition) == func.ty(x)
                    && self.boolean(func, condition)
                    && self.known(x) == Some(1)
                    && self.known(y) == Some(0)
                {
                    return Some(condition);
                }
                if func.ty(condition) == func.ty(x)
                    && self.find(condition) == self.find(x)
                    && self.known(y).is_some_and(|imm| extend(imm, func.ty(x), false) == 0)
                {
                    return Some(x);
                }
                if let Some(&Op::Icmp(cc, lhs, rhs)) = self.def(func, condition) {
                    for (idx, (var, imm_val)) in [(lhs, rhs), (rhs, lhs)].into_iter().enumerate() {
                        if let Some(c) = self.known(imm_val) {
                            let ty = func.ty(var);
                            let imm = extend(c, ty, false);
                            let norm = match (cc, idx == 1) {
                                (IntCC::Equal, _) => Some((IntCC::Equal, imm)),
                                (IntCC::NotEqual, _) => Some((IntCC::NotEqual, imm)),
                                (IntCC::UnsignedGreaterThan, false) | (IntCC::UnsignedLessThan, true) if imm == 0 => Some((IntCC::NotEqual, 0)),
                                (IntCC::UnsignedGreaterThanOrEqual, false) | (IntCC::UnsignedLessThanOrEqual, true) if imm == 1 => Some((IntCC::NotEqual, 0)),
                                (IntCC::UnsignedLessThan, false) | (IntCC::UnsignedGreaterThan, true) if imm == 1 => Some((IntCC::Equal, 0)),
                                (IntCC::UnsignedLessThanOrEqual, false) | (IntCC::UnsignedGreaterThanOrEqual, true) if imm == 0 => Some((IntCC::Equal, 0)),
                                _ => None,
                            };
                            if let Some((eq_cc, eq_c)) = norm {
                                let is_c = |v: Value| self.known(v).is_some_and(|v_imm| extend(v_imm, ty, false) == eq_c);
                                let is_var = |v: Value| self.find(v) == self.find(var);
                                if (eq_cc == IntCC::Equal && is_c(x) && is_var(y)) || (eq_cc == IntCC::NotEqual && is_var(x) && is_c(y)) {
                                    return Some(var);
                                }
                            }
                        }
                    }
                }
                None
            }
            Op::Convert(Convert::Bitcast, x) => match *self.def(func, x)? {
                Op::Convert(Convert::Bitcast, value) if func.ty(value) != func.ty(x) => Some(value),
                _ => None,
            },
            Op::Extract(..) | Op::Shuffle(..) => vector::vec_identity(self, func, op),
            Op::Binary(Binary::Fmin | Binary::Fmax, x, y) if self.find(x) == self.find(y) => Some(x),
            Op::Binary(kind, x, y) if func.ty(x).is_small_int() => self.neutral(func, kind, func.ty(x), x, y).or_else(|| self.cancel(func, kind, x, y)),
            Op::Unary(kind @ (Unary::Ineg | Unary::Bnot), x) => match *self.def(func, x)? {
                Op::Unary(inner, value) if inner == kind => Some(value),
                _ => None,
            },
            Op::Icmp(cc, x, y) => [(x, y), (y, x)].into_iter().find_map(|(b, c)| {
                let imm = extend(self.known(c)?, func.ty(b), false);
                (func.ty(b) == Type::I8 && self.boolean(func, b) && matches!((cc, imm), (IntCC::NotEqual, 0) | (IntCC::Equal, 1))).then_some(b)
            }),
            _ => None,
        }
    }

    /// `x` when `y` is the unit of `op` at type `ty` or a shift count that masks to zero, or `y` when `x` is the unit of a commuting `op`.
    fn neutral(&self, func: &Function, op: Binary, ty: Type, x: Value, y: Value) -> Option<Value> {
        let is = |value: Value, want: i64| self.known(value).is_some_and(|imm| extend(imm, ty, false) == extend(want, ty, false));
        if let Some(unit) = unit(op) {
            if is(y, unit) {
                return Some(x);
            }
            return (op != Binary::Isub && is(x, unit)).then_some(y);
        }
        let shift = matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr);
        if shift && self.known(y).is_some_and(|count| count & i64::from(ty.bits() - 1) == 0) {
            return Some(x);
        }
        if op == Binary::Band {
            for (val, other) in [(x, y), (y, x)] {
                if let Some(mask) = self.known(other) {
                    let (kz, _) = known_bits(func, &|v| self.defs[self.find(v).index()], &|v| self.known(v), val, 0);
                    let active = !kz & ty.unsigned_max();
                    if (active & (extend(mask, ty, false) as u64)) == active {
                        return Some(val);
                    }
                }
            }
        }
        None
    }

    /// `a` when `x ^ y` is `(a ^ c) ^ c`, or `b` when `(a + b) - a` in either order of `Iadd`.
    fn cancel(&self, func: &Function, op: Binary, x: Value, y: Value) -> Option<Value> {
        let ty = func.ty(x);
        if op == Binary::Isub {
            return match *self.def(func, x)? {
                Op::Binary(Binary::Iadd, a, b) if self.same(ty, a, y) => Some(b),
                Op::Binary(Binary::Iadd, a, b) if self.same(ty, b, y) => Some(a),
                _ => None,
            };
        }
        if op != Binary::Bxor {
            return None;
        }
        [(x, y), (y, x)].into_iter().find_map(|(inner, other)| match *self.def(func, inner)? {
            Op::Binary(Binary::Bxor, a, b) if self.same(ty, b, other) => Some(a),
            Op::Binary(Binary::Bxor, a, b) if self.same(ty, a, other) => Some(b),
            _ => None,
        })
    }

    /// Finds, in one pass in block order, the small integers that are constants, wrapped to their type, and the results that equal an existing value.
    fn constants(&mut self, func: &Function) {
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result.filter(|result| self.subst[result.index()] == *result && self.consts[result.index()].is_none()) else { continue };
            let ty = func.ty(result);
            if let Some(imm) = self.evaluate(func, &data.op, ty) {
                self.consts[result.index()] = Some(imm);
            } else if let Some(value) = self.identity(func, &data.op) {
                self.subst[result.index()] = value;
            }
        }
    }

    /// Turns the branches `decide` settles into jumps, inverts `xor(b, 1)` branches, drops traps that cannot fire, rewrites always-firing traps to `Op::Trap`, and truncates after `Op::Trap`.
    fn branches(&self, func: &mut Function) -> bool {
        let mut changed = false;
        for block in 0..func.blocks.len() {
            let mut trapped = false;
            let Function { blocks, insts, types, .. } = func;
            blocks[block].insts.retain(|&inst| {
                let op = &mut insts[inst.index()].op;
                if trapped { *op = Op::Trap; changed = true; return false; }
                if let Op::Trapz(x) = *op && let Some(k) = self.known(x) {
                    *op = Op::Trap; changed = true; trapped = extend(k, types[x.index()], false) == 0; return trapped;
                }
                trapped = *op == Op::Trap;
                true
            });
            let Some(&last) = func.blocks[block].insts.last() else { continue };
            let invert = match func.insts[last.index()].op {
                Op::Brif(cond, ..) if self.known(cond).is_none() => match self.def(func, cond) {
                    Some(&Op::Binary(Binary::Bxor, a, b)) => [(a, b), (b, a)].into_iter().find_map(|(inner, one)| {
                        (func.ty(inner) == Type::I8 && self.boolean(func, inner) && self.known(one).is_some_and(|imm| extend(imm, Type::I8, false) == 1)).then_some(inner)
                    }),
                    _ => None,
                },
                _ => None,
            };
            if let Some(inner) = invert
                && let Op::Brif(cond, then, then_args, otherwise, else_args) = &mut func.insts[last.index()].op
            {
                *cond = inner;
                std::mem::swap(then, otherwise);
                std::mem::swap(then_args, else_args);
                changed = true;
            }
            changed |= self.decide(&func.types, &mut func.insts[last.index()].op);
        }
        changed
    }

    /// Turns `op` into a jump when it branches on a constant or sends both edges to the same block with the same arguments.
    fn decide(&self, types: &[Type], op: &mut Op) -> bool {
        match op {
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let same = then == otherwise && then_args.len() == else_args.len() && then_args.iter().zip(else_args.iter()).all(|(a, b)| self.find(*a) == self.find(*b));
                let taken = match self.known(*condition) {
                    Some(known) => known != 0,
                    None if same => true,
                    None => return false,
                };
                let (target, args) = if taken { (*then, std::mem::take(then_args)) } else { (*otherwise, std::mem::take(else_args)) };
                *op = Op::Jump(target, args);
                true
            }
            Op::BrTable(index, default, targets) => {
                let target = if let Some(known) = self.known(*index) {
                    let i = extend(known, types[index.index()], false) as u64;
                    usize::try_from(i).ok().and_then(|at| targets.get(at).copied()).unwrap_or(*default)
                } else if targets.iter().all(|target| target == default) {
                    *default
                } else {
                    return false;
                };
                *op = Op::Jump(target, Vec::new());
                true
            }
            _ => false,
        }
    }

    /// Sends each edge that passes a constant to a block holding only a branch on that parameter straight to the target the branch takes.
    fn thread(&self, func: &mut Function, pinned: &[bool]) -> bool {
        let mut uses = func.use_counts();
        let mut changed = false;
        for block in 0..func.blocks.len() {
            let Some(&last) = func.blocks[block].insts.last() else { continue };
            let mut op = std::mem::replace(&mut func.insts[last.index()].op, Op::Trap);
            match &mut op {
                Op::Jump(target, args) => changed |= self.retarget(func, (&mut uses, pinned), target, args),
                Op::Brif(_, then, then_args, otherwise, else_args) => {
                    changed |= self.retarget(func, (&mut uses, pinned), then, then_args);
                    changed |= self.retarget(func, (&mut uses, pinned), otherwise, else_args);
                }
                _ => {}
            }
            func.insts[last.index()].op = op;
        }
        changed
    }

    /// Moves the edge to `target` with `args` along the blocks that only jump on, or only branch on a parameter the edge makes a constant, to the first block that does more, keeping `uses` counting the edge's new arguments.
    fn retarget(&self, func: &Function, (uses, pinned): (&mut [u32], &[bool]), target: &mut Block, args: &mut Vec<Value>) -> bool {
        let Some((first, first_values)) = self.step(func, (uses, pinned), *target, args) else { return false };
        if first == *target {
            return false;
        }
        let (mut at, mut values, mut seen) = (first, first_values, vec![*target, first]);
        while let Some((next, next_values)) = self.step(func, (uses, pinned), at, &values) {
            if seen.contains(&next) {
                return false;
            }
            seen.push(next);
            (at, values) = (next, next_values);
        }
        if !self.agrees(func, at, &values) {
            return false;
        }
        args.iter().for_each(|value| uses[value.index()] -= 1);
        values.iter().for_each(|value| uses[value.index()] += 1);
        (*target, *args) = (at, values);
        true
    }

    /// Whether the parameters of `block` that an earlier round replaced stand for the values `args` gives them.
    fn agrees(&self, func: &Function, block: Block, args: &[Value]) -> bool {
        func.blocks[block.index()].params.iter().zip(args).all(|(param, arg)| self.subst[param.index()] == *param || self.find(*arg) == self.find(*param))
    }

    /// Where an edge into `block` with `args` goes next when `block` holds only a jump, or a branch on a parameter `args` makes a constant or on a comparison of such a parameter with a constant, and reads its parameters nowhere else.
    fn step(&self, func: &Function, (uses, pinned): (&[u32], &[bool]), block: Block, args: &[Value]) -> Option<(Block, Vec<Value>)> {
        let data = &func.blocks[block.index()];
        let (compare, last) = match data.insts[..] {
            [only] => (None, only),
            [compare, last] => (Some(compare), last),
            _ => return None,
        };
        if !self.agrees(func, block, args) {
            return None;
        }
        let op = &func.insts[last.index()].op;
        let param = |value: &Value| data.params.iter().position(|param| param == value);
        let (next, next_args) = match op {
            Op::Jump(next, next_args) if compare.is_none() => (*next, next_args),
            Op::Brif(condition, then, then_args, otherwise, else_args) => match self.edge_condition(func, (compare, uses), *condition, &|value| param(&value).map(|at| args[at]))? {
                0 => (*otherwise, else_args),
                _ => (*then, then_args),
            },
            _ => return None,
        };
        let reads = |param: Value| {
            let mut count = 0_u32;
            compare.iter().map(|inst| &func.insts[inst.index()].op).chain([op]).for_each(|op| op.for_operands(|value| count += u32::from(value == param)));
            count
        };
        if pinned[next.index()] || data.params.iter().any(|param| uses[param.index()] != reads(*param)) {
            return None;
        }
        Some((next, next_args.iter().map(|value| param(value).map_or(*value, |at| args[at])).collect()))
    }

    /// The constant a branch on `condition` sees when the block's parameters hold what `arg` gives them: `condition` itself is such a parameter, or `compare` alone uses it and compares parameters or constants.
    fn edge_condition(&self, func: &Function, (compare, uses): (Option<Inst>, &[u32]), condition: Value, arg: &dyn Fn(Value) -> Option<Value>) -> Option<i64> {
        let Some(compare) = compare else { return self.known(arg(condition)?) };
        let data = &func.insts[compare.index()];
        let (Op::Icmp(cc, x, y), Some(result)) = (&data.op, data.result) else { return None };
        if result != condition || uses[result.index()] != 1 || !func.ty(*x).is_small_int() {
            return None;
        }
        let value = |operand: Value| arg(operand).map_or_else(|| self.known(operand), |given| self.known(given));
        Some(i64::from(holds(*cc, func.ty(*x), value(*x)?, value(*y)?)))
    }

    /// Replaces each block parameter that every reachable edge gives one value, other than itself.
    fn params(&mut self, func: &Function, reachable: &[bool], pinned: &[bool]) -> bool {
        for param in func.blocks.iter().flat_map(|block| &block.params) {
            self.incoming[param.index()] = None;
            self.mixed[param.index()] = false;
        }
        for (block, _) in reachable.iter().enumerate().filter(|(_, live)| **live) {
            for_edges(func, block, |target, args| {
                for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                    let arg = self.find(*arg);
                    match self.incoming[param.index()] {
                        _ if arg == *param => {}
                        None => self.incoming[param.index()] = Some(arg),
                        Some(seen) if seen != arg => self.mixed[param.index()] = true,
                        Some(_) => {}
                    }
                }
            });
        }
        let mut changed = false;
        for (_, data) in func.blocks.iter().enumerate().filter(|(block, _)| reachable[*block] && !pinned[*block]) {
            for param in &data.params {
                if let (Some(value), false) = (self.incoming[param.index()], self.mixed[param.index()])
                    && self.subst[param.index()] == *param
                {
                    self.subst[param.index()] = value;
                    changed = true;
                }
            }
        }
        changed
    }
}

/// Calls `f` with each block the terminator of `block` jumps to with arguments, and those arguments.
fn for_edges<'a>(func: &'a Function, block: usize, mut f: impl FnMut(Block, &'a [Value])) {
    let Some(last) = func.blocks[block].insts.last() else { return };
    match &func.insts[last.index()].op {
        Op::Jump(target, args) => f(*target, args),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            f(*then, then_args);
            f(*otherwise, else_args);
        }
        _ => {}
    }
}

/// The entry and every block a call enters, whose parameters are not edge arguments.
fn pinned(func: &Function, entry: Block) -> Vec<bool> {
    let mut pinned = vec![false; func.blocks.len()];
    pinned[entry.index()] = true;
    for block in 0..func.blocks.len() as u32 {
        match func.last(Block(block)) {
            Some(Op::TryCall { normal, handler, .. }) => {
                pinned[normal.index()] = true;
                pinned[handler.index()] = true;
            }
            Some(Op::BrTable(_, default, targets)) => {
                pinned[default.index()] = true;
                targets.iter().for_each(|target| pinned[target.index()] = true);
            }
            _ => {}
        }
    }
    pinned
}

/// Drops the parameters `fold` replaced, and their arguments on every edge.
fn drop_params(func: &mut Function, fold: &Fold) {
    let gone = |param: &Value| fold.subst[param.index()] != *param;
    let blocks = &func.blocks;
    let keep = |args: &mut Vec<Value>, target: Block| {
        let params = &blocks[target.index()].params;
        let mut index = 0;
        args.retain(|_| {
            index += 1;
            !params.get(index - 1).is_some_and(gone)
        });
    };
    for block in blocks {
        let Some(&last) = block.insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, args) => keep(args, *target),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                keep(then_args, *then);
                keep(else_args, *otherwise);
            }
            _ => {}
        }
    }
    for block in &mut func.blocks {
        block.params.retain(|param| !gone(param));
    }
}

/// The instruction in a block that defines each value.
pub(crate) fn defs(func: &Function) -> Vec<Option<Inst>> {
    let mut defs = vec![None; func.types.len()];
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Some(result) = func.insts[inst.index()].result {
            defs[result.index()] = Some(inst);
        }
    }
    defs
}

fn prune_changed(func: &mut Function, entry: Block) -> (Vec<bool>, bool) {
    let reachable = crate::cfg::Cfg::from_entry(func, entry).reachable;
    let mut changed = false;
    for (block, live) in func.blocks.iter_mut().zip(&reachable) {
        if !live && !block.insts.is_empty() {
            block.insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
            changed = true;
        }
    }
    (reachable, changed)
}

/// Empties the blocks no path from `entry` reaches and returns which blocks a path reaches.
pub(crate) fn prune(func: &mut Function, entry: Block) -> Vec<bool> {
    prune_changed(func, entry).0
}

/// Drops pure instructions whose results nothing reads, in one backward pass that also frees the operands they read.
fn dead(func: &mut Function) -> bool {
    let mut uses = func.use_counts();
    let mut keep = Vec::new();
    let mut dropped = false;
    for block in func.blocks.iter_mut().rev() {
        keep.clear();
        keep.resize(block.insts.len(), true);
        for (at, inst) in block.insts.iter().enumerate().rev() {
            let data = &func.insts[inst.index()];
            if data.op.is_removable() && data.result.is_some_and(|result| uses[result.index()] == 0) {
                keep[at] = false;
                dropped = true;
                data.op.for_operands(|value| uses[value.index()] -= 1);
            }
        }
        let mut at = 0;
        block.insts.retain(|_| { at += 1; keep[at - 1] });
    }
    dropped
}

fn fold_bitcast_loads(func: &mut Function) -> bool {
    let uses = func.use_counts();
    let defs = defs(func);
    let mut remap: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut dropped = vec![false; func.insts.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::Convert(Convert::Bitcast, src), Some(res)) = (&data.op, data.result)
            && uses[src.index()] == 1
            && func.ty(*src).bytes() == func.ty(res).bytes()
            && func.ty(*src) != func.ty(res)
            && let Some(src_inst) = defs[src.index()]
            && matches!(func.insts[src_inst.index()].op, Op::Load(..))
        {
            func.types[src.index()] = func.ty(res);
            remap[res.index()] = Some(*src);
            dropped[inst.index()] = true;
        }
    }
    if remap.iter().all(Option::is_none) {
        return false;
    }
    for block in &mut func.blocks {
        block.insts.retain(|inst| !dropped[inst.index()]);
        for &inst in &block.insts {
            func.insts[inst.index()].op.map_operands(|v| remap[v.index()].unwrap_or(v));
        }
    }
    true
}

/// Folds constants, identities and branches on them, settles the compares, traps and branches the facts on each path decide when `effort` is full, empties blocks no path reaches, replaces block parameters every edge agrees on, drops dead pure instructions, and threads the edges through the blocks that leaves holding only a jump; returns whether `func` was mutated.
pub(crate) fn run(func: &mut Function, effort: crate::Effort) -> bool {
    let Some(entry) = func.entry else { return false };
    let count = func.types.len();
    let defs = defs(func);
    let mut fold = Fold { consts: vec![None; count], subst: (0..count as u32).map(Value).collect(), incoming: vec![None; count], mixed: vec![false; count], defs };
    let pinned = pinned(func, entry);
    let (mut reachable, mut changed) = (Vec::new(), false);
    loop {
        fold.constants(func);
        let settled = effort == crate::Effort::Full && reachable.is_empty() && crate::facts::settle(func, &fold.defs, &mut fold.consts);
        let branched = settled | fold.branches(func) | fold.thread(func, &pinned);
        if branched || reachable.is_empty() {
            let (r, pruned) = prune_changed(func, entry);
            reachable = r; changed |= pruned;
        }
        let replaced = fold.params(func, &reachable, &pinned);
        changed |= branched | replaced;
        if !branched && !replaced {
            break;
        }
    }
    drop_params(func, &fold);
    for block in &func.blocks {
        for inst in &block.insts {
            func.insts[inst.index()].op.map_operands(|v| { let n = fold.find(v); changed |= n != v; n });
            let replacement = match func.insts[inst.index()].op {
                Op::Binary(Binary::Fmul, x, y) if crate::algebra::is_float_two(func, &fold.defs, y) => Some(Op::Binary(Binary::Fadd, x, x)),
                Op::Binary(Binary::Fmul, x, y) if crate::algebra::is_float_two(func, &fold.defs, x) => Some(Op::Binary(Binary::Fadd, y, y)),
                Op::Binary(Binary::Isub, x, y) if func.ty(x).is_small_int() && fold.known(x).is_some_and(|imm| extend(imm, func.ty(x), false) == 0) => Some(Op::Unary(Unary::Ineg, y)),
                _ => vector::vec_zero_fold(func, &fold, &func.insts[inst.index()].op)
                    .or_else(|| func.insts[inst.index()].result.and_then(|result| fold.consts[result.index()]).map(Op::Const)),
            };
            if let Some(op) = replacement && func.insts[inst.index()].op != op {
                func.insts[inst.index()].op = op;
                changed = true;
            }
        }
    }
    let snap = (!changed).then(|| (func.blocks.iter().map(|b| b.insts.clone()).collect::<Vec<_>>(), func.insts.iter().map(|d| d.op.clone()).collect::<Vec<_>>()));
    crate::algebra::rewrite(func, &fold.defs);
    crate::algebra::invert(func);
    if let Some((b0, o0)) = snap {
        changed = o0.len() != func.insts.len() || b0.iter().zip(&func.blocks).any(|(a, b)| *a != b.insts) || o0.iter().zip(&func.insts).any(|(a, b)| *a != b.op);
    }
    changed |= fold_bitcast_loads(func) | dead(func);
    if fold.thread(func, &pinned) | fold.branches(func) { prune(func, entry); dead(func); changed = true; }
    if crate::algebra::devirt_vtables(func, &fold.defs) { dead(func); changed = true; }
    if crate::algebra::fold_bxor_conds(func) { dead(func); changed = true; }
    changed
}

pub(crate) use crate::thread::compact;
