use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::fold::extend;
use crate::ir::{Binary, Block, Convert, FloatCC, Function, Inst, IntCC, Op, ReduceOp, ShuffleMask, StackSlot, TargetOpId, Type, Unary, Value};
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
            | Binary::Fmin
            | Binary::Fmax
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
    )
}

fn flip_int_cc(cc: IntCC) -> IntCC {
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
    Fcmp(FloatCC, Type, Operand, Operand),
    Shuffle(Type, Operand, Operand, ShuffleKey),
    Extract(Operand, u8),
    Insert(Operand, Operand, u8),
    Reduce(ReduceOp, Operand),
    TargetOp(TargetOpId, [Option<Operand>; 3]),
    Select(Operand, Operand, Operand),
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

/// The key of `op` with result type `ty` and its operands replaced by `subst`, when it reads no memory and has no effect.
fn key(op: &Op, consts: &[Option<i64>], types: &[Type], ty: Type, subst: &[Value]) -> Option<Key> {
    let opnd = |value: Value| operand(consts, types, subst, value);
    Some(match *op {
        Op::Const(imm) => Key::Const(ty, imm),
        Op::Unary(op, x) => Key::Unary(op, opnd(x)),
        Op::Binary(op, x, y) => {
            let (a, b) = (opnd(x), opnd(y));
            let (a, b) = if is_commutative_binary(op) { order_pair(a, b) } else { (a, b) };
            Key::Binary(op, a, b)
        }
        Op::Convert(op, x) => Key::Convert(op, ty, opnd(x)),
        Op::Icmp(cc, x, y) => {
            let x_ty = types[find(subst, x).index()];
            let (a, b) = (opnd(x), opnd(y));
            if a > b {
                Key::Icmp(flip_int_cc(cc), x_ty, b, a)
            } else {
                Key::Icmp(cc, x_ty, a, b)
            }
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
        Op::Select(c, x, y) => Key::Select(opnd(c), opnd(x), opnd(y)),
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
    drop_buf: Vec<Key>,
}

impl Table {
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
                let Some(key) = key(&data.op, &self.consts, &func.types, func.ty(result), subst) else { return true };
                match self.known.get(&key) {
                    Some(&earlier) => {
                        subst[result.index()] = earlier;
                        false
                    }
                    None => {
                        let prev = self.known.insert(key, result);
                        self.added.push((key, prev));
                        true
                    }
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

/// Replaces every pure instruction that a dominating instruction already computed with that instruction's result, returning whether any instruction was dropped.
pub(crate) fn run(func: &mut Function) -> bool {
    let Some(entry) = func.entry else { return false };
    if func.coroutine.is_some() {
        return false;
    }
    let children = Cfg::from_entry(func, entry).children();
    let mut subst = identity(func.types.len());
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (&Op::Const(imm), Some(res)) = (&data.op, data.result) {
            let ty = func.ty(res);
            if ty.is_small_int() {
                consts[res.index()] = Some(extend(imm, ty, false));
            }
        }
    }
    let mut table = Table { consts, ..Table::default() };
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
    }
    dropped
}
