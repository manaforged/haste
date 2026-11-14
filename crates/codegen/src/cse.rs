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
        Binary::Iadd
            | Binary::Imul
            | Binary::Umulhi
            | Binary::Smulhi
            | Binary::Smin
            | Binary::Smax
            | Binary::Umin
            | Binary::Umax
            | Binary::SaddSat
            | Binary::UaddSat
            | Binary::AvgRound
            | Binary::Fmin
            | Binary::Fmax
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
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

/// The values the dominating blocks computed, by key, and the keys to forget on leaving each block.
#[derive(Default)]
struct Table {
    known: HashMap<Key, Value>,
    added: Vec<(Key, Option<Value>)>,
    consts: Vec<Option<i64>>,
    bool_neg: Vec<Option<Value>>,
    icmp_def: Vec<Option<(IntCC, Type, Value, Value)>>,
    def_block: Vec<Block>,
    local_flag_only: Vec<bool>,
    drop_buf: Vec<Key>,
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

    /// Drops from `block` every pure instruction a dominating one already computed, recording its replacement in `subst`; true when it dropped any.
    fn scan(&mut self, func: &mut Function, block: Block, subst: &mut [Value]) -> bool {
        let insts = std::mem::take(&mut func.blocks[block.index()].insts);
        let before = insts.len();
        let kept: Vec<Inst> = insts
            .into_iter()
            .filter(|inst| {
                let data = &func.insts[inst.index()];
                if matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
                    self.drop_buf.clear();
                    self.drop_buf.extend(self.known.keys().copied().filter(|k| matches!(k, Key::Const(..))));
                    for k in self.drop_buf.drain(..) {
                        let prev = self.known.remove(&k);
                        self.added.push((k, prev));
                    }
                    return true;
                }
                let Some(result) = data.result else { return true };
                let Some(key) = key(&data.op, &self.consts, &self.bool_neg, &func.types, func.ty(result), subst) else {
                    return true;
                };
                if let Some(&earlier) = self.known.get(&key) {
                    if self.local_flag_only[result.index()] && self.def_block[earlier.index()] != block {
                        return true;
                    }
                    subst[result.index()] = earlier;
                    false
                } else if let Some(earlier) = self.alt_select(&data.op, &func.types, subst) {
                    subst[result.index()] = earlier;
                    false
                } else {
                    let prev = self.known.insert(key, result);
                    self.added.push((key, prev));
                    true
                }
            })
            .collect();
        let dropped = kept.len() != before;
        func.blocks[block.index()].insts = kept;
        dropped
    }

    fn forget(&mut self, mark: usize) {
        for (key, prev) in self.added.drain(mark..).rev() {
            match prev {
                Some(v) => {
                    self.known.insert(key, v);
                }
                None => {
                    self.known.remove(&key);
                }
            }
        }
    }
}

enum Visit {
    Enter(Block),
    Leave(usize),
}

/// Replaces every pure instruction that a dominating instruction already computed with that instruction's result, returning the invalidated IR domain.
pub(crate) fn run(func: &mut Function) -> Domain {
    let Some(entry) = func.entry else { return Domain::NONE };
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let cfg = func.cfg();
    let children = cfg.children();
    let mut subst = identity(func.types.len());
    let mut consts = vec![None; func.types.len()];
    let mut bool_val = vec![false; func.types.len()];
    let mut bool_neg = vec![None; func.types.len()];
    let mut icmp_def = vec![None; func.types.len()];
    let mut def_block = vec![Block(u32::MAX); func.types.len()];
    let mut local_flag_only = vec![false; func.types.len()];
    let uses = func.use_counts();
    for &b in &cfg.rpo {
        let blk = &func.blocks[b.index()];
        for &inst in &blk.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                let ri = res.index();
                def_block[ri] = b;
                match data.op {
                    Op::Const(imm) => {
                        let ty = func.ty(res);
                        if ty.is_small_int() {
                            let z = extend(imm, ty, false);
                            consts[ri] = Some(z);
                            bool_val[ri] = z == 0 || z == 1;
                        }
                    }
                    Op::Icmp(cc, x, y) => {
                        bool_val[ri] = true;
                        icmp_def[ri] = Some((cc, func.ty(x), x, y));
                        if cc == IntCC::Equal {
                            if consts[x.index()] == Some(0) && bool_val[y.index()] {
                                bool_neg[ri] = Some(y);
                            } else if consts[y.index()] == Some(0) && bool_val[x.index()] {
                                bool_neg[ri] = Some(x);
                            }
                        }
                    }
                    Op::Fcmp(..) | Op::Fits(..) => bool_val[ri] = true,
                    Op::Convert(Convert::Uextend, x) if bool_val[x.index()] => bool_val[ri] = true,
                    Op::Binary(Binary::Band, x, y) if bool_val[x.index()] || bool_val[y.index()] => bool_val[ri] = true,
                    Op::Binary(Binary::Bor, x, y) if bool_val[x.index()] && bool_val[y.index()] => bool_val[ri] = true,
                    Op::Binary(Binary::Bxor, x, y) => {
                        bool_val[ri] = bool_val[x.index()] && bool_val[y.index()];
                        if consts[y.index()] == Some(1) && bool_val[x.index()] {
                            bool_neg[ri] = Some(x);
                        } else if consts[x.index()] == Some(1) && bool_val[y.index()] {
                            bool_neg[ri] = Some(y);
                        }
                    }
                    Op::Binary(Binary::Isub, x, y) if consts[x.index()] == Some(1) && bool_val[y.index()] => {
                        bool_val[ri] = true;
                        bool_neg[ri] = Some(y);
                    }
                    Op::Select(_, x, y) if bool_val[x.index()] && bool_val[y.index()] => bool_val[ri] = true,
                    _ => {}
                }
            }
            if let Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..) = data.op
                && uses[c.index()] == 1
                && def_block[c.index()] == b
                && let Some(c_def) = blk.insts.iter().find(|&&i| func.insts[i.index()].result == Some(c))
                && matches!(func.insts[c_def.index()].op, Op::Icmp(..))
            {
                local_flag_only[c.index()] = true;
            }
        }
    }
    let mut table = Table {
        consts,
        bool_neg,
        icmp_def,
        def_block,
        local_flag_only,
        ..Table::default()
    };
    let mut dropped = false;
    let mut stack = vec![Visit::Enter(entry)];
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
    if dropped {
        rename(func, &subst);
        Domain::VALUES
    } else {
        Domain::NONE
    }
}
