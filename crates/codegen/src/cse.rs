use rustc_hash::FxHashMap as HashMap;

use crate::fold::extend;
use crate::ir::{Binary, Block, Convert, Domain, FloatCC, Function, Inst, IntCC, Op, ReduceOp, ShuffleMask, StackSlot, TargetOpId, Type, Unary, Value};
use crate::merge::{find, identity, rename};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Operand {
    Val(Value),
    Imm(Type, i64),
}

fn order_pair(a: Operand, b: Operand) -> (Operand, Operand) {
    if a > b { (b, a) } else { (a, b) }
}

fn is_commutative_binary(op: Binary) -> bool {
    matches!(
        op,
        Binary::Iadd | Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax
            | Binary::SaddSat | Binary::UaddSat | Binary::AvgRound | Binary::Fmin | Binary::Fmax | Binary::Band | Binary::Bor | Binary::Bxor
    )
}

pub(crate) fn flip_int_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal | IntCC::NotEqual => cc,
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
    }
}

fn invert_int_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal => IntCC::NotEqual,
        IntCC::NotEqual => IntCC::Equal,
        IntCC::SignedLessThan => IntCC::SignedGreaterThanOrEqual,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThan,
        IntCC::SignedGreaterThan => IntCC::SignedLessThanOrEqual,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThan,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThanOrEqual,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThan,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThan,
    }
}

fn canonical_bound(cc: IntCC, v: i128, min: i128, max: i128, (lt, le, gt, ge): (IntCC, IntCC, IntCC, IntCC)) -> (IntCC, i128) {
    if cc == lt {
        if v == min + 1 {
            return (IntCC::Equal, min);
        }
        if v == max {
            return (IntCC::NotEqual, max);
        }
    } else if cc == le {
        if v == min {
            return (IntCC::Equal, min);
        }
        if v < max - 1 {
            return (lt, v + 1);
        }
        if v == max - 1 {
            return (IntCC::NotEqual, max);
        }
    } else if cc == gt {
        if v == min {
            return (IntCC::NotEqual, min);
        }
        if v < max - 1 {
            return (ge, v + 1);
        }
        if v == max - 1 {
            return (IntCC::Equal, max);
        }
    } else if cc == ge {
        if v == min + 1 {
            return (IntCC::NotEqual, min);
        }
        if v == max {
            return (IntCC::Equal, max);
        }
    }
    (cc, v)
}

fn canonical_icmp(cc: IntCC, x: Operand, y: Operand) -> (IntCC, Operand, Operand) {
    let (cc, a, b) = if x > y { (flip_int_cc(cc), y, x) } else { (cc, x, y) };
    let (Operand::Val(_), Operand::Imm(ity, imm)) = (a, b) else {
        return (cc, a, b);
    };
    if !ity.is_small_int() {
        return (cc, a, b);
    }
    let bits = ity.bits();
    let (new_cc, new_v) = if cc.is_signed() {
        let min = -(1_i128 << (bits - 1));
        let max = (1_i128 << (bits - 1)) - 1;
        let v = i128::from(extend(imm, ity, true));
        canonical_bound(
            cc,
            v,
            min,
            max,
            (IntCC::SignedLessThan, IntCC::SignedLessThanOrEqual, IntCC::SignedGreaterThan, IntCC::SignedGreaterThanOrEqual),
        )
    } else {
        let min = 0_i128;
        let max = (1_i128 << bits) - 1;
        let v = i128::from(extend(imm, ity, false) as u64);
        canonical_bound(
            cc,
            v,
            min,
            max,
            (IntCC::UnsignedLessThan, IntCC::UnsignedLessThanOrEqual, IntCC::UnsignedGreaterThan, IntCC::UnsignedGreaterThanOrEqual),
        )
    };
    (new_cc, a, Operand::Imm(ity, extend(new_v as i64, ity, false)))
}

fn flip_float_cc(cc: FloatCC) -> FloatCC {
    match cc {
        FloatCC::Equal | FloatCC::NotEqual => cc,
        FloatCC::LessThan => FloatCC::GreaterThan,
        FloatCC::GreaterThan => FloatCC::LessThan,
        FloatCC::LessThanOrEqual => FloatCC::GreaterThanOrEqual,
        FloatCC::GreaterThanOrEqual => FloatCC::LessThanOrEqual,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ShuffleKey {
    Const([u8; 16]),
    Dyn(Operand),
}

/// What a pure instruction computes, by its operation and the values it reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Const(Type, i64),
    Unary(Unary, Operand),
    Binary(Binary, Operand, Operand),
    Convert(Convert, Type, Operand),
    Icmp(IntCC, Type, Operand, Operand),
    Fits(Binary, bool, Operand, Operand),
    Fcmp(FloatCC, Type, Operand, Operand),
    Shuffle(Type, Operand, Operand, ShuffleKey),
    Extract(Operand, u8),
    Insert(Operand, Operand, u8),
    Reduce(ReduceOp, Operand),
    TargetOp(TargetOpId, [Option<Operand>; 3]),
    Select(Operand, Operand, Operand),
    Fma(Operand, Operand, Operand),
    StackAddr(StackSlot, i32),
    SlotAddr(u32),
    Symbol(u32),
    ThreadCell(Operand, Operand, Operand),
}

fn operand(consts: &[Option<i64>], types: &[Type], subst: &[Value], value: Value) -> Operand {
    let v = find(subst, value);
    match consts[v.index()] {
        Some(imm) => Operand::Imm(types[v.index()], imm),
        None => Operand::Val(v),
    }
}

fn unwrap_bool_neg(bool_neg: &[Option<Value>], subst: &[Value], c: Value, x: Value, y: Value) -> (Value, Value, Value) {
    let mut cond = find(subst, c);
    let mut inv = false;
    while let Some(inner) = bool_neg[cond.index()] {
        cond = find(subst, inner);
        inv = !inv;
    }
    if inv { (cond, y, x) } else { (cond, x, y) }
}

/// The key of `op` with result type `ty` and its operands replaced by `subst`, when it reads no memory and has no effect.
fn key(op: &Op, consts: &[Option<i64>], bool_neg: &[Option<Value>], types: &[Type], ty: Type, subst: &[Value]) -> Option<Key> {
    let opnd = |value: Value| operand(consts, types, subst, value);
    Some(match *op {
        Op::Const(imm) => Key::Const(ty, if ty.is_small_int() { extend(imm, ty, false) } else { imm }),
        Op::Unary(op, x) => Key::Unary(op, opnd(x)),
        Op::Binary(Binary::Isub, x, y) => match (opnd(x), opnd(y)) {
            (a, Operand::Imm(sub_ty, imm)) => {
                let neg = extend(imm.wrapping_neg(), sub_ty, false);
                let (a, b) = order_pair(a, Operand::Imm(sub_ty, neg));
                Key::Binary(Binary::Iadd, a, b)
            }
            (a, b) => Key::Binary(Binary::Isub, a, b),
        },
        Op::Binary(op, x, y) => {
            let (a, b) = (opnd(x), opnd(y));
            let (a, b) = if is_commutative_binary(op) { order_pair(a, b) } else { (a, b) };
            Key::Binary(op, a, b)
        }
        Op::Convert(op, x) => Key::Convert(op, ty, opnd(x)),
        Op::Icmp(cc, x, y) => {
            let x_ty = types[find(subst, x).index()];
            let (cc, a, b) = canonical_icmp(cc, opnd(x), opnd(y));
            Key::Icmp(cc, x_ty, a, b)
        }
        Op::Fits(op, signed, x, y) => {
            let (a, b) = (opnd(x), opnd(y));
            let (a, b) = if matches!(op, Binary::Iadd | Binary::Imul) { order_pair(a, b) } else { (a, b) };
            Key::Fits(op, signed, a, b)
        }
        Op::Fcmp(cc, x, y) => {
            let x_ty = types[find(subst, x).index()];
            let (a, b) = (opnd(x), opnd(y));
            if a > b {
                Key::Fcmp(flip_float_cc(cc), x_ty, b, a)
            } else {
                Key::Fcmp(cc, x_ty, a, b)
            }
        }
        Op::Shuffle(a, b, mask) => {
            let mkey = match mask {
                ShuffleMask::Const(bytes) => ShuffleKey::Const(bytes),
                ShuffleMask::Dyn(idx) => ShuffleKey::Dyn(opnd(idx)),
            };
            Key::Shuffle(ty, opnd(a), opnd(b), mkey)
        }
        Op::Extract(v, lane) => Key::Extract(opnd(v), lane),
        Op::Insert(vec, scalar, lane) => Key::Insert(opnd(vec), opnd(scalar), lane),
        Op::Reduce(kind, v) => Key::Reduce(kind, opnd(v)),
        Op::TargetOp(id, args) => {
            let s = args.as_slice();
            Key::TargetOp(id, [s.first().map(|v| opnd(*v)), s.get(1).map(|v| opnd(*v)), s.get(2).map(|v| opnd(*v))])
        }
        Op::Select(c, x, y) => {
            let (cond, sx, sy) = unwrap_bool_neg(bool_neg, subst, c, x, y);
            Key::Select(opnd(cond), opnd(sx), opnd(sy))
        }
        Op::Fma(a, b, c) => Key::Fma(opnd(a), opnd(b), opnd(c)),
        Op::StackAddr(slot, offset) => Key::StackAddr(slot, offset),
        Op::SlotAddr(sym) => Key::SlotAddr(sym),
        Op::Symbol(sym) => Key::Symbol(sym),
        _ => return None,
    })
}

fn block_tag(block: Block) -> u32 {
    block.0 + 1
}

/// The values the dominating blocks computed, by key, and the keys to forget on leaving each block.
#[derive(Default)]
struct Table {
    known: HashMap<Key, Value>,
    added: Vec<(Key, Option<Value>)>,
    consts: Vec<Option<i64>>,
    bool_neg: Vec<Option<Value>>,
    icmp_def: Vec<Option<(IntCC, Type, Value, Value)>>,
    def_block: Vec<u32>,
    local_flag_only: Vec<bool>,
    bool_val: Vec<bool>,
    affine: Vec<(Value, i64)>,
    stack_addrs: Vec<Option<(StackSlot, i32)>>,
    assoc: Vec<Option<(Binary, Value, i64)>>,
    unary_src: Vec<Option<(Unary, Value)>>,
    conv_src: Vec<Option<(Convert, Value, Type)>>,
    select_def: Vec<Option<(Value, Value, Value)>>,
    drop_buf: Vec<Key>,
    const_count: usize,
}

impl Table {
    fn alt_select(&self, op: &Op, types: &[Type], subst: &[Value]) -> Option<Value> {
        let Op::Select(c, x, y) = *op else { return None };
        let (cond, sx, sy) = unwrap_bool_neg(&self.bool_neg, subst, c, x, y);
        let (cc, x_ty, ix, iy) = self.icmp_def[cond.index()]?;
        let (inv_cc, ia, ib) = canonical_icmp(invert_int_cc(cc), operand(&self.consts, types, subst, ix), operand(&self.consts, types, subst, iy));
        let &inv_c = self.known.get(&Key::Icmp(inv_cc, x_ty, ia, ib))?;
        let key = Key::Select(Operand::Val(inv_c), operand(&self.consts, types, subst, sy), operand(&self.consts, types, subst, sx));
        self.known.get(&key).copied()
    }

    fn simplify_select(&mut self, c: Value, x: Value, y: Value, types: &[Type], result: Value, subst: &[Value]) -> (Value, Value, Value, bool) {
        let (cond, raw_x, raw_y) = unwrap_bool_neg(&self.bool_neg, subst, c, x, y);
        let (mut sx, mut sy) = (find(subst, raw_x), find(subst, raw_y));
        let mut flat = false;
        if !types[cond.index()].is_vector() {
            if let Some((c2, a, _)) = self.select_def[sx.index()]
                && find(subst, c2) == cond
            {
                sx = find(subst, a);
                flat = true;
            }
            if let Some((c2, _, b)) = self.select_def[sy.index()]
                && find(subst, c2) == cond
            {
                sy = find(subst, b);
                flat = true;
            }
            self.select_def[result.index()] = Some((cond, sx, sy));
        }
        (cond, sx, sy, flat)
    }

    fn normalized_key(&mut self, op: &Op, types: &[Type], ty: Type, result: Value, subst: &[Value]) -> Option<Result<Key, Value>> {
        match *op {
            Op::StackAddr(slot, offset) => {
                self.stack_addrs[result.index()] = Some((slot, offset));
                Some(Ok(Key::StackAddr(slot, offset)))
            }
            Op::Unary(u @ (Unary::Ineg | Unary::Bnot | Unary::Bswap), x) => {
                let xv = find(subst, x);
                if let Some((prev_u, inner)) = self.unary_src[xv.index()]
                    && prev_u == u
                {
                    return Some(Err(find(subst, inner)));
                }
                self.unary_src[result.index()] = Some((u, xv));
                Some(Ok(Key::Unary(u, operand(&self.consts, types, subst, xv))))
            }
            Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr, x, y) => {
                if let Operand::Imm(_, 0) = operand(&self.consts, types, subst, y) {
                    return Some(Err(find(subst, x)));
                }
                key(op, &self.consts, &self.bool_neg, types, ty, subst).map(Ok)
            }
            Op::Binary(bin @ (Binary::Iadd | Binary::Isub), x, y) if ty.is_small_int() => {
                let ox = operand(&self.consts, types, subst, x);
                let oy = operand(&self.consts, types, subst, y);
                let val_imm = match (bin, ox, oy) {
                    (Binary::Iadd, Operand::Val(base), Operand::Imm(_, imm))
                    | (Binary::Iadd, Operand::Imm(_, imm), Operand::Val(base)) => Some((base, imm)),
                    (Binary::Isub, Operand::Val(base), Operand::Imm(_, imm)) => Some((base, imm.wrapping_neg())),
                    (Binary::Isub, Operand::Imm(_, 1), Operand::Val(v)) if self.bool_val[v.index()] => {
                        let (a, b) = order_pair(Operand::Val(v), Operand::Imm(ty, 1));
                        return Some(Ok(Key::Binary(Binary::Bxor, a, b)));
                    }
                    _ => None,
                };
                if let Some((base, delta)) = val_imm {
                    let delta = extend(delta, ty, false);
                    if delta == 0 {
                        return Some(Err(base));
                    }
                    if ty == Type::I64
                        && let Some((slot, base_off)) = self.stack_addrs[base.index()]
                        && let Some(total) = i32::try_from(delta as i64).ok().and_then(|d| base_off.checked_add(d))
                    {
                        self.stack_addrs[result.index()] = Some((slot, total));
                        return Some(Ok(Key::StackAddr(slot, total)));
                    }
                    let (root, base_off) = self.affine[base.index()];
                    let root = find(subst, root);
                    let total = extend(base_off.wrapping_add(delta), ty, false);
                    self.affine[result.index()] = (root, total);
                    if total == 0 {
                        return Some(Err(root));
                    }
                    let (a, b) = order_pair(Operand::Val(root), Operand::Imm(ty, total));
                    return Some(Ok(Key::Binary(Binary::Iadd, a, b)));
                }
                key(op, &self.consts, &self.bool_neg, types, ty, subst).map(Ok)
            }
            Op::Binary(bin @ (Binary::Band | Binary::Bor | Binary::Bxor | Binary::Imul), x, y) if ty.is_small_int() => {
                let ox = operand(&self.consts, types, subst, x);
                let oy = operand(&self.consts, types, subst, y);
                let val_imm = match (ox, oy) {
                    (Operand::Val(v), Operand::Imm(_, imm)) | (Operand::Imm(_, imm), Operand::Val(v)) => Some((v, imm)),
                    _ => None,
                };
                if let Some((v, raw_imm)) = val_imm {
                    let imm = extend(raw_imm, ty, false);
                    let all_ones = extend(-1, ty, false);
                    let (root, cur_imm) = match self.assoc[v.index()] {
                        Some((prev_bin, r, c)) if prev_bin == bin => (find(subst, r), c),
                        _ => (
                            v,
                            match bin {
                                Binary::Band => all_ones,
                                Binary::Imul => 1,
                                _ => 0,
                            },
                        ),
                    };
                    let combined = match bin {
                        Binary::Band => extend(cur_imm & imm, ty, false),
                        Binary::Bor => extend(cur_imm | imm, ty, false),
                        Binary::Bxor => extend(cur_imm ^ imm, ty, false),
                        _ => extend(cur_imm.wrapping_mul(imm), ty, false),
                    };
                    let identity_val = match bin {
                        Binary::Band => all_ones,
                        Binary::Imul => 1,
                        _ => 0,
                    };
                    if combined == identity_val || (bin == Binary::Band && self.bool_val[root.index()] && (combined & 1) == 1) {
                        return Some(Err(root));
                    }
                    let known_const = if combined == 0 && matches!(bin, Binary::Band | Binary::Imul) {
                        Some(0)
                    } else if bin == Binary::Band && self.bool_val[root.index()] && (combined & 1) == 0 {
                        Some(0)
                    } else if bin == Binary::Bor && combined == all_ones {
                        Some(all_ones)
                    } else {
                        None
                    };
                    if let Some(cval) = known_const
                        && let Some(&existing) = self.known.get(&Key::Const(ty, cval))
                    {
                        return Some(Err(existing));
                    }
                    self.assoc[result.index()] = Some((bin, root, combined));
                    let (a, b) = order_pair(Operand::Val(root), Operand::Imm(ty, combined));
                    return Some(Ok(Key::Binary(bin, a, b)));
                }
                key(op, &self.consts, &self.bool_neg, types, ty, subst).map(Ok)
            }
            Op::Convert(conv, x) => {
                let xv = find(subst, x);
                let x_ty = types[xv.index()];
                if matches!(conv, Convert::Bitcast | Convert::Uextend | Convert::Sextend | Convert::Ireduce) && ty == x_ty {
                    return Some(Err(xv));
                }
                if let Some((prev_conv, inner, inner_ty)) = self.conv_src[xv.index()] {
                    let inner = find(subst, inner);
                    match (prev_conv, conv) {
                        (Convert::Bitcast, Convert::Bitcast) if ty == inner_ty => return Some(Err(inner)),
                        (Convert::Bitcast, Convert::Bitcast) => {
                            self.conv_src[result.index()] = Some((Convert::Bitcast, inner, inner_ty));
                            return Some(Ok(Key::Convert(Convert::Bitcast, ty, operand(&self.consts, types, subst, inner))));
                        }
                        (Convert::Uextend, Convert::Uextend | Convert::Sextend) => {
                            self.conv_src[result.index()] = Some((Convert::Uextend, inner, inner_ty));
                            return Some(Ok(Key::Convert(Convert::Uextend, ty, operand(&self.consts, types, subst, inner))));
                        }
                        (Convert::Sextend, Convert::Sextend) => {
                            self.conv_src[result.index()] = Some((Convert::Sextend, inner, inner_ty));
                            return Some(Ok(Key::Convert(Convert::Sextend, ty, operand(&self.consts, types, subst, inner))));
                        }
                        (Convert::Ireduce, Convert::Ireduce) => {
                            self.conv_src[result.index()] = Some((Convert::Ireduce, inner, inner_ty));
                            return Some(Ok(Key::Convert(Convert::Ireduce, ty, operand(&self.consts, types, subst, inner))));
                        }
                        (Convert::Uextend | Convert::Sextend, Convert::Ireduce) => {
                            if ty == inner_ty {
                                return Some(Err(inner));
                            }
                            let eff = if ty.bytes() < inner_ty.bytes() { Convert::Ireduce } else { prev_conv };
                            self.conv_src[result.index()] = Some((eff, inner, inner_ty));
                            return Some(Ok(Key::Convert(eff, ty, operand(&self.consts, types, subst, inner))));
                        }
                        (Convert::Ireduce, Convert::Uextend) if ty == inner_ty && matches!(x_ty, Type::I8 | Type::I16 | Type::I32) => {
                            let mask = x_ty.unsigned_max() as i64;
                            self.assoc[result.index()] = Some((Binary::Band, inner, mask));
                            let (a, b) = order_pair(operand(&self.consts, types, subst, inner), Operand::Imm(ty, mask));
                            return Some(Ok(Key::Binary(Binary::Band, a, b)));
                        }
                        _ => {}
                    }
                }
                self.conv_src[result.index()] = Some((conv, xv, x_ty));
                let eff_conv = if conv == Convert::Sextend && self.bool_val[xv.index()] {
                    Convert::Uextend
                } else {
                    conv
                };
                Some(Ok(Key::Convert(eff_conv, ty, operand(&self.consts, types, subst, xv))))
            }
            Op::Icmp(cc, x, y) => {
                let x_ty = types[find(subst, x).index()];
                let (cc, a, b) = canonical_icmp(cc, operand(&self.consts, types, subst, x), operand(&self.consts, types, subst, y));
                if a == b {
                    let cval = i64::from(crate::fold::holds(cc, x_ty, 0, 0));
                    if let Some(&existing) = self.known.get(&Key::Const(ty, cval)) {
                        return Some(Err(existing));
                    }
                }
                if let (Operand::Val(v), Operand::Imm(ity, imm)) = (a, b)
                    && self.bool_val[v.index()]
                {
                    if (cc == IntCC::NotEqual && imm == 0) || (cc == IntCC::Equal && imm == 1) {
                        if ity == Type::I8 {
                            return Some(Err(v));
                        }
                        if let Some((Convert::Uextend, inner, Type::I8)) = self.conv_src[v.index()] {
                            return Some(Err(find(subst, inner)));
                        }
                    }
                    if ity == Type::I8 && ((cc == IntCC::Equal && imm == 0) || (cc == IntCC::NotEqual && imm == 1)) {
                        let (pa, pb) = order_pair(Operand::Val(v), Operand::Imm(Type::I8, 1));
                        return Some(Ok(Key::Binary(Binary::Bxor, pa, pb)));
                    }
                }
                Some(Ok(Key::Icmp(cc, x_ty, a, b)))
            }
            Op::Select(c, x, y) => {
                let (cond, sx, sy, _) = self.simplify_select(c, x, y, types, result, subst);
                if sx == sy {
                    return Some(Err(sx));
                }
                if !types[cond.index()].is_vector()
                    && let Some(imm) = self.consts[cond.index()]
                {
                    return Some(Err(if imm == 0 { sy } else { sx }));
                }
                Some(Ok(Key::Select(operand(&self.consts, types, subst, cond), operand(&self.consts, types, subst, sx), operand(&self.consts, types, subst, sy))))
            }
            _ => key(op, &self.consts, &self.bool_neg, types, ty, subst).map(Ok),
        }
    }

    /// Drops from `block` every pure instruction a dominating one already computed, recording its replacement in `subst`; true when it dropped or simplified any.
    fn scan(&mut self, func: &mut Function, block: Block, subst: &mut [Value]) -> bool {
        let mut insts = std::mem::take(&mut func.blocks[block.index()].insts);
        let mut mutated = false;
        insts.retain(|&inst| {
            let data = &func.insts[inst.index()];
            if matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
                if let Op::Call { callee, ref args, ref results, .. } = data.op
                    && (func.is_thread_cell(callee) || func.is_thread_cell(find(subst, callee)))
                    && let ([slot, size, init], [result]) = (args.as_slice(), results.as_slice())
                {
                    let opnd = |v: Value| operand(&self.consts, &func.types, subst, v);
                    let tc_key = Key::ThreadCell(opnd(*slot), opnd(*size), opnd(*init));
                    if let Some(&earlier) = self.known.get(&tc_key) {
                        subst[result.index()] = earlier;
                        mutated = true;
                        return false;
                    }
                    let prev = self.known.insert(tc_key, *result);
                    self.added.push((tc_key, prev));
                }
                if self.const_count > 0 {
                    self.drop_buf.clear();
                    self.drop_buf.extend(self.known.keys().copied().filter(|k| matches!(k, Key::Const(..))));
                    for k in self.drop_buf.drain(..) {
                        let prev = self.known.remove(&k);
                        self.const_count -= usize::from(prev.is_some());
                        self.added.push((k, prev));
                    }
                }
                return true;
            }
            let Some(result) = data.result else { return true };
            let Some(norm) = self.normalized_key(&data.op, &func.types, func.ty(result), result, subst) else { return true };
            let key = match norm {
                Ok(key) => key,
                Err(alias) => { subst[result.index()] = alias; mutated = true; return false; }
            };
            if let Some(&earlier) = self.known.get(&key) {
                if self.local_flag_only[result.index()] && self.def_block[earlier.index()] != block_tag(block) { return true; }
                subst[result.index()] = earlier;
                mutated = true;
                false
            } else if let Some(earlier) = self.alt_select(&func.insts[inst.index()].op, &func.types, subst) {
                subst[result.index()] = earlier;
                mutated = true;
                false
            } else {
                if let Op::Select(c, x, y) = func.insts[inst.index()].op {
                    let (cond, sx, sy, flat) = self.simplify_select(c, x, y, &func.types, result, subst);
                    if flat { func.insts[inst.index()].op = Op::Select(cond, sx, sy); mutated = true; }
                }
                let prev = self.known.insert(key, result);
                self.const_count += usize::from(prev.is_none() && matches!(key, Key::Const(..)));
                self.added.push((key, prev));
                true
            }
        });
        func.blocks[block.index()].insts = insts;
        mutated
    }

    fn forget(&mut self, mark: usize) {
        for (key, prev) in self.added.drain(mark..).rev() {
            let is_const = matches!(key, Key::Const(..));
            match prev {
                Some(v) => self.const_count += usize::from(self.known.insert(key, v).is_none() && is_const),
                None => self.const_count -= usize::from(self.known.remove(&key).is_some() && is_const),
            }
        }
    }
}

enum Visit { Enter(Block), Leave(usize) }

/// Replaces every pure instruction that a dominating instruction already computed with that instruction's result, returning the invalidated IR domain.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else { return Domain::NONE };
    if func.coroutine.is_some() { return Domain::NONE; }
    let cfg = func.cfg();
    let children = cfg.children();
    let mut subst = identity(func.types.len());
    let mut consts = vec![None; func.types.len()];
    let mut bool_val = vec![false; func.types.len()];
    let mut bool_neg = vec![None; func.types.len()];
    let mut icmp_def = vec![None; func.types.len()];
    let mut def_block = vec![0_u32; func.types.len()];
    let mut local_flag_only = vec![false; func.types.len()];
    let mut uses = vec![0_u32; func.types.len()];
    let mut flag_cands: Vec<(Value, u32)> = Vec::new();
    let mut has_narrow_uext = false;
    for &b in &cfg.rpo {
        let blk = &func.blocks[b.index()];
        let btag = block_tag(b);
        for &inst in &blk.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| uses[v.index()] += 1);
            if let Some(res) = data.result {
                let ri = res.index();
                def_block[ri] = btag;
                match data.op {
                    Op::Const(imm) if func.ty(res).is_small_int() => {
                        let z = extend(imm, func.ty(res), false);
                        consts[ri] = Some(z);
                        bool_val[ri] = z == 0 || z == 1;
                    }
                    Op::Icmp(cc, x, y) => {
                        bool_val[ri] = true;
                        icmp_def[ri] = Some((cc, func.ty(x), x, y));
                        if cc == IntCC::Equal {
                            if consts[x.index()] == Some(0) && bool_val[y.index()] { bool_neg[ri] = Some(y); }
                            else if consts[y.index()] == Some(0) && bool_val[x.index()] { bool_neg[ri] = Some(x); }
                        }
                    }
                    Op::Fcmp(..) | Op::Fits(..) => bool_val[ri] = true,
                    Op::Convert(Convert::Uextend, x) => {
                        if bool_val[x.index()] { bool_val[ri] = true; }
                        if matches!(func.ty(x), Type::I8 | Type::I16) { has_narrow_uext = true; }
                    }
                    Op::Binary(Binary::Band, x, y) if bool_val[x.index()] || bool_val[y.index()] => bool_val[ri] = true,
                    Op::Binary(Binary::Bor, x, y) if bool_val[x.index()] && bool_val[y.index()] => bool_val[ri] = true,
                    Op::Binary(Binary::Bxor, x, y) => {
                        bool_val[ri] = bool_val[x.index()] && bool_val[y.index()];
                        if consts[y.index()] == Some(1) && bool_val[x.index()] { bool_neg[ri] = Some(x); }
                        else if consts[x.index()] == Some(1) && bool_val[y.index()] { bool_neg[ri] = Some(y); }
                    }
                    Op::Binary(Binary::Isub, x, y) if consts[x.index()] == Some(1) && bool_val[y.index()] => { bool_val[ri] = true; bool_neg[ri] = Some(y); }
                    Op::Select(_, x, y) if bool_val[x.index()] && bool_val[y.index()] => bool_val[ri] = true,
                    _ => {}
                }
            }
            if let Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..) = data.op && def_block[c.index()] == btag && icmp_def[c.index()].is_some() {
                flag_cands.push((c, btag));
            }
        }
    }
    for (c, _) in flag_cands {
        if uses[c.index()] == 1 { local_flag_only[c.index()] = true; }
    }
    let affine = identity(func.types.len()).into_iter().map(|v| (v, 0_i64)).collect();
    let nvals = func.types.len();
    let mut table = Table {
        consts, bool_neg, icmp_def, def_block, local_flag_only, bool_val, affine,
        stack_addrs: vec![None; nvals], assoc: vec![None; nvals], unary_src: vec![None; nvals], conv_src: vec![None; nvals], select_def: vec![None; nvals],
        ..Table::default()
    };
    let (mut dropped, mut stack) = (false, vec![Visit::Enter(entry)]);
    while let Some(visit) = stack.pop() {
        match visit {
            Visit::Leave(mark) => table.forget(mark),
            Visit::Enter(block) => {
                stack.push(Visit::Leave(table.added.len()));
                dropped |= table.scan(func, block, &mut subst);
                stack.extend(children[block.index()].iter().map(|child| Visit::Enter(*child)));
            }
        }
    }
    if dropped { rename(func, &subst); }
    let changed = has_narrow_uext && narrow_widened_subs(func, &cfg, &table.consts, &table.bool_val);
    if dropped || changed { Domain::VALUES } else { Domain::NONE }
}

const COND_PEEL_DEPTH: u8 = 4;
const DOM_PROVE_STEPS: u8 = 24;

/// Operand or constant represented on a branch edge.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeVal { Const(i64), Val(Value) }

/// Predicate context mapping values through substitutions, constants, and known booleans.
pub(crate) struct CmpEnv<'a> { pub(crate) defs: &'a [Option<Inst>], pub(crate) consts: &'a [Option<i64>], pub(crate) subst: &'a [Value], pub(crate) bool_val: Option<&'a [bool]> }

impl CmpEnv<'_> {
    /// Resolves `v` to its canonical representative in `subst`.
    pub(crate) fn find(&self, v: Value) -> Value { crate::merge::find(self.subst, v) }
    fn known(&self, v: Value) -> Option<i64> { self.consts.get(self.find(v).index()).copied().flatten() }
    /// Returns `v` as a constant `EdgeVal` when known, or its canonical `Value`.
    pub(crate) fn edge_val(&self, v: Value) -> EdgeVal { self.known(v).map_or_else(|| EdgeVal::Val(self.find(v)), EdgeVal::Const) }
    fn is_bool(&self, func: &Function, v: Value) -> bool { self.bool_val.map_or_else(|| self.defs.get(v.index()).copied().flatten().is_some_and(|i| matches!(func.insts[i.index()].op, Op::Icmp(..) | Op::Fits(..))), |b| b.get(v.index()).copied().unwrap_or(false)) }

    /// Peels zero-extensions and boolean xor-1 inversions off `cond`, returning the inner op, polarity, and depth.
    pub(crate) fn peel_cond<'a>(&self, func: &'a Function, mut cond: Value, mut taken: bool, mut depth: u8) -> Option<(&'a Op, bool, u8)> {
        while depth < COND_PEEL_DEPTH {
            let op = &func.insts[self.defs.get(cond.index()).copied().flatten()?.index()].op;
            match *op {
                Op::Convert(Convert::Uextend, inner) if func.ty(inner).is_small_int() => cond = inner,
                Op::Binary(Binary::Bxor, a, b) => { let (i, _) = [(a, b), (b, a)].into_iter().find(|&(i, k)| self.is_bool(func, i) && self.known(k).is_some_and(|v| crate::fold::extend(v, func.ty(k), false) == 1))?; (cond, taken) = (i, !taken); }
                _ => return Some((op, taken, depth)),
            }
            depth += 1;
        }
        None
    }
}

/// Returns whether holding `hcc(hx, hy)` implies `wcc(wx, wy)` at integer type `ty`.
pub(crate) fn cmp_proves(ty: Type, (mut hcc, mut hx, mut hy): (IntCC, EdgeVal, EdgeVal), (mut wcc, mut wx, mut wy): (IntCC, EdgeVal, EdgeVal)) -> bool {
    use IntCC::*;
    if let (EdgeVal::Const(a), EdgeVal::Const(b)) = (wx, wy) { return crate::fold::holds(wcc, ty, a, b); }
    if matches!((hx, hy), (EdgeVal::Const(_), EdgeVal::Val(_))) { (hcc, hx, hy) = (flip_int_cc(hcc), hy, hx); }
    if matches!((wx, wy), (EdgeVal::Const(_), EdgeVal::Val(_))) { (wcc, wx, wy) = (flip_int_cc(wcc), wy, wx); }
    let cc_ok = |h, w| h == w || matches!((h, w), (Equal, UnsignedLessThanOrEqual | UnsignedGreaterThanOrEqual | SignedLessThanOrEqual | SignedGreaterThanOrEqual) | (UnsignedLessThan, UnsignedLessThanOrEqual | NotEqual) | (UnsignedGreaterThan, UnsignedGreaterThanOrEqual | NotEqual) | (SignedLessThan, SignedLessThanOrEqual | NotEqual) | (SignedGreaterThan, SignedGreaterThanOrEqual | NotEqual));
    if ((hx, hy) == (wx, wy) && cc_ok(hcc, wcc)) || ((hy, hx) == (wx, wy) && cc_ok(flip_int_cc(hcc), wcc)) { return true; }
    let (EdgeVal::Val(hv), EdgeVal::Const(hc), EdgeVal::Val(wv), EdgeVal::Const(wc)) = (hx, hy, wx, wy) else { return false };
    if hv != wv { return false; }
    let (um, ch, cw) = (ty.unsigned_max(), crate::fold::extend(hc, ty, false) as u64, crate::fold::extend(wc, ty, false) as u64);
    let (lo, hi) = match hcc {
        Equal => (ch, ch), UnsignedLessThan if ch > 0 => (0, ch - 1), UnsignedLessThanOrEqual => (0, ch), UnsignedGreaterThan if ch < um => (ch + 1, um),
        UnsignedGreaterThanOrEqual => (ch, um), NotEqual if ch == 0 && um >= 1 => (1, um), NotEqual if ch == um && um >= 1 => (0, um - 1), _ => return false,
    };
    matches!((wcc, cw), (Equal, c) if lo == c && hi == c) || matches!(wcc, NotEqual if cw < lo || cw > hi)
        || matches!(wcc, UnsignedLessThan if hi < cw) || matches!(wcc, UnsignedLessThanOrEqual if hi <= cw)
        || matches!(wcc, UnsignedGreaterThan if lo > cw) || matches!(wcc, UnsignedGreaterThanOrEqual if lo >= cw)
}

/// Returns whether assuming `cond == taken` implies comparison `want` at `ty`.
pub(crate) fn cond_proves_want(func: &Function, env: &CmpEnv<'_>, (cond, taken, depth): (Value, bool, u8), ty: Type, want: (IntCC, EdgeVal, EdgeVal)) -> bool {
    let Some((op, taken, depth)) = env.peel_cond(func, cond, taken, depth) else { return false };
    match *op {
        Op::Binary(Binary::Band, a, b) if taken => cond_proves_want(func, env, (a, true, depth + 1), ty, want) || cond_proves_want(func, env, (b, true, depth + 1), ty, want),
        Op::Binary(Binary::Bor, a, b) if !taken => cond_proves_want(func, env, (a, false, depth + 1), ty, want) || cond_proves_want(func, env, (b, false, depth + 1), ty, want),
        Op::Icmp(cc, x, y) if func.ty(x) == ty => cmp_proves(ty, (if taken { cc } else { cc.inverse() }, env.edge_val(x), env.edge_val(y)), want),
        _ => false,
    }
}

/// Returns whether a dominating edge or trap/assume proves `want` at `ty` reaching `cur`.
pub(crate) fn dom_proves_want(func: &Function, cfg: &crate::cfg::Cfg, env: &CmpEnv<'_>, mut cur: Block, skip: Option<Block>, ty: Type, want: (IntCC, EdgeVal, EdgeVal)) -> bool {
    let start = cur;
    for _ in 0..DOM_PROVE_STEPS {
        if Some(cur) != skip && func.blocks[cur.index()].insts.iter().any(|&i| matches!(func.insts[i.index()].op, Op::Trapz(c) | Op::Assume(c) if cond_proves_want(func, env, (c, true, 0), ty, want))) { return true; }
        let Some(parent) = cfg.idom[cur.index()].filter(|&p| p != cur) else { break };
        if let Some(&Op::Brif(c, tb, _, eb, _)) = func.last(parent) && tb != eb
            && let Some(t) = (cfg.single_pred(tb) == Some(parent) && cfg.dominates(tb, start)).then_some(true).or_else(|| (cfg.single_pred(eb) == Some(parent) && cfg.dominates(eb, start)).then_some(false))
            && cond_proves_want(func, env, (c, t, 0), ty, want) { return true; }
        cur = parent;
    }
    false
}

fn sub_add_const(consts: &[Option<i64>], op: &Op, ty: Type) -> Option<(Value, i64)> {
    let um = ty.unsigned_max() as i64;
    match *op {
        Op::Binary(Binary::Isub, x, y) => consts[y.index()].map(|c| (x, crate::fold::extend(c, ty, false).wrapping_neg() & um)),
        Op::Binary(Binary::Iadd, x, y) => consts[y.index()].map(|c| (x, crate::fold::extend(c, ty, false) & um)).or_else(|| consts[x.index()].map(|c| (y, crate::fold::extend(c, ty, false) & um))),
        _ => None,
    }
}

fn narrow_widened_subs(func: &mut Function, cfg: &crate::cfg::Cfg, consts: &[Option<i64>], bool_val: &[bool]) -> bool {
    let defs = crate::fold::defs(func);
    let mut wide = Vec::new();
    for &b in &cfg.rpo {
        for (p, &i) in func.blocks[b.index()].insts.iter().enumerate() {
            if let Some(r) = func.insts[i.index()].result && matches!(func.ty(r), Type::I32 | Type::I64)
                && let Some((u, d)) = sub_add_const(consts, &func.insts[i.index()].op, func.ty(r))
                && let Some(ui) = defs[u.index()] && let Op::Convert(Convert::Uextend, base) = func.insts[ui.index()].op && matches!(func.ty(base), Type::I8 | Type::I16)
            { wide.push((base, func.ty(base), d & (func.ty(base).unsigned_max() as i64), r, b, p)); }
        }
    }
    if wide.is_empty() { return false; }
    let id_subst: Vec<Value> = (0..func.types.len() as u32).map(Value).collect();
    let env = CmpEnv { defs: &defs, consts, subst: &id_subst, bool_val: Some(bool_val) };
    let (mut changed, mut subst, mut has_subst) = (false, id_subst.clone(), false);
    for &block in &cfg.rpo {
        for pos in 0..func.blocks[block.index()].insts.len() {
            let inst = func.blocks[block.index()].insts[pos];
            let Some(res) = func.insts[inst.index()].result else { continue };
            let ty = func.ty(res);
            if matches!(ty, Type::I8 | Type::I16) && let Some((base, delta)) = sub_add_const(consts, &func.insts[inst.index()].op, ty)
                && let Some(&(_, _, _, dw, _, _)) = wide.iter().find(|&&(bw, st, d, _, wb, wp)| bw == base && st == ty && d == delta && (wb == block && wp < pos || wb != block && cfg.dominates(wb, block)))
            { func.insts[inst.index()].op = Op::Convert(Convert::Ireduce, dw); changed = true; }
            if let Op::Convert(Convert::Uextend, inner) = func.insts[inst.index()].op && let Some(ii) = defs[inner.index()] && let Op::Convert(Convert::Ireduce, src) = func.insts[ii.index()].op
                && func.ty(inner).bits() < func.ty(src).bits() && dom_proves_want(func, cfg, &env, block, Some(block), func.ty(src), (IntCC::UnsignedLessThanOrEqual, EdgeVal::Val(src), EdgeVal::Const(func.ty(inner).unsigned_max() as i64)))
            { func.insts[inst.index()].op = if ty.bits() > func.ty(src).bits() { Op::Convert(Convert::Uextend, src) } else if ty.bits() < func.ty(src).bits() { Op::Convert(Convert::Ireduce, src) } else { subst[res.index()] = src; has_subst = true; continue }; changed = true; }
        }
    }
    if has_subst { rename(func, &subst); changed = true; }
    if changed { let u = func.use_counts(); for b in &mut func.blocks { b.insts.retain(|&i| !matches!(func.insts[i.index()].op, Op::Convert(Convert::Ireduce | Convert::Uextend, _)) || func.insts[i.index()].result.is_none_or(|r| u[r.index()] > 0)); } }
    changed
}
