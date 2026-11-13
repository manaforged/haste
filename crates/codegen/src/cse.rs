use rustc_hash::FxHashMap as HashMap;

use crate::dom::dominated;
use crate::ir::{Binary, Block, Convert, FloatCC, Function, Inst, IntCC, Op, StackSlot, Type, Unary, Value};
use crate::merge::{find, identity, rename};

/// What a pure instruction computes, by its operation and the values it reads.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Const(Type, i64),
    Unary(Unary, Value),
    Binary(Binary, Value, Value),
    Convert(Convert, Type, Value),
    Icmp(IntCC, Type, Value, Value),
    Fcmp(FloatCC, Type, Value, Value),
    Splat(Type, Value),
    Select(Value, Value, Value),
    StackAddr(StackSlot, i32),
    SlotAddr(u32),
    Symbol(u32),
}

/// The key of `op` with result type `ty` and its operands replaced by `subst`, when it reads no memory and has no effect.
fn key(op: &Op, types: &[Type], ty: Type, subst: &[Value]) -> Option<Key> {
    let f = |value: Value| find(subst, value);
    Some(match *op {
        Op::Const(imm) => Key::Const(ty, imm),
        Op::Unary(op, x) => Key::Unary(op, f(x)),
        Op::Binary(op, x, y) => Key::Binary(op, f(x), f(y)),
        Op::Convert(op, x) => Key::Convert(op, ty, f(x)),
        Op::Icmp(cc, x, y) => {
            let (x, y) = (f(x), f(y));
            Key::Icmp(cc, types[x.index()], x, y)
        }
        Op::Fcmp(cc, x, y) => {
            let (x, y) = (f(x), f(y));
            Key::Fcmp(cc, types[x.index()], x, y)
        }
        Op::Splat(x) => Key::Splat(ty, f(x)),
        Op::Select(c, x, y) => Key::Select(f(c), f(x), f(y)),
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
                    let const_keys: Vec<Key> = self.known.keys().copied().filter(|k| matches!(k, Key::Const(..))).collect();
                    for k in const_keys {
                        let prev = self.known.remove(&k);
                        self.added.push((k, prev));
                    }
                    return true;
                }
                let Some(result) = data.result else { return true };
                let Some(key) = key(&data.op, &func.types, func.ty(result), subst) else { return true };
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

/// Replaces every pure instruction that a dominating instruction already computed with that instruction's result.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.coroutine.is_some() {
        return;
    }
    let children = dominated(func, entry);
    let mut subst = identity(func.types.len());
    let mut table = Table::default();
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
}
