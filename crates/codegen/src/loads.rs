//! Reuses what memory holds along straight-line code: a load of the bytes an earlier load or store on the same path read or wrote, with no write of unknown reach between them, takes that value.

use crate::ir::{Block, Function, Inst, Op, StackSlot, Type, Value};
use crate::merge::{find, identity, pred_counts, rename};

/// The most cells one path remembers, so a long block costs linear time.
const CELLS: usize = 64;

/// Where remembered bytes start: a frame slot, or an address held in a value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Base {
    Slot(StackSlot),
    Addr(Value),
}

/// Bytes of memory read or written as one value of `ty`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell {
    base: Base,
    offset: i64,
    ty: Type,
}

impl Cell {
    fn overlaps(&self, other: &Cell) -> bool {
        self.offset < other.offset + i64::from(other.ty.bytes()) && other.offset < self.offset + i64::from(self.ty.bytes())
    }

    /// Whether a write of `self` may change the bytes of `other`: frame slots never overlap each other, an address may point anywhere.
    fn clobbers(&self, other: &Cell) -> bool {
        match (self.base, other.base) {
            (Base::Slot(a), Base::Slot(b)) => a == b && self.overlaps(other),
            (Base::Addr(a), Base::Addr(b)) if a == b => self.overlaps(other),
            _ => true,
        }
    }
}

/// What memory holds along one path, by cell.
#[derive(Clone, Default)]
struct Known(Vec<(Cell, Value)>);

impl Known {
    fn get(&self, cell: &Cell) -> Option<Value> {
        self.0.iter().find(|(known, _)| known == cell).map(|(_, value)| *value)
    }

    fn remember(&mut self, cell: Cell, value: Value) {
        if self.0.len() < CELLS {
            self.0.push((cell, value));
        }
    }

    fn write(&mut self, cell: Cell, value: Value) {
        self.0.retain(|(known, _)| !cell.clobbers(known));
        self.remember(cell, value);
    }
}

/// What one instruction does to memory.
enum Effect {
    None,
    Read(Cell, Value),
    Write(Cell, Value),
    Unknown,
}

/// Whether `op` may write memory it does not name by a plain address, or orders accesses around it.
pub(crate) fn clobbers(op: &Op) -> bool {
    matches!(op, Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence | Op::Copy(..) | Op::Move(..) | Op::Call { .. } | Op::TryCall { .. })
}

fn effect(func: &Function, slots: &[Option<(StackSlot, i32)>], subst: &[Value], inst: Inst) -> Effect {
    let data = &func.insts[inst.index()];
    let at = |addr: Value, offset: i32, ty: Type| match slots[addr.index()] {
        Some((slot, base)) => Cell { base: Base::Slot(slot), offset: i64::from(base) + i64::from(offset), ty },
        None => Cell { base: Base::Addr(find(subst, addr)), offset: i64::from(offset), ty },
    };
    let slot = |slot: StackSlot, offset: i32, ty: Type| Cell { base: Base::Slot(slot), offset: i64::from(offset), ty };
    match (&data.op, data.result) {
        (&Op::Load(addr, offset), Some(result)) => Effect::Read(at(addr, offset, func.ty(result)), result),
        (&Op::StackLoad(from, offset), Some(result)) => Effect::Read(slot(from, offset, func.ty(result)), result),
        (&Op::Store(value, addr, offset), _) => Effect::Write(at(addr, offset, func.ty(value)), find(subst, value)),
        (&Op::StackStore(value, to, offset), _) => Effect::Write(slot(to, offset, func.ty(value)), find(subst, value)),
        (op, _) if clobbers(op) => Effect::Unknown,
        _ => Effect::None,
    }
}

/// Drops from `block` each load whose bytes `known` holds, recording its replacement in `subst`, and leaves in `known` what memory holds at the block's end.
fn scan(func: &mut Function, block: Block, (slots, subst): (&[Option<(StackSlot, i32)>], &mut [Value]), known: &mut Known) -> bool {
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let before = insts.len();
    let mut kept = Vec::with_capacity(before);
    for inst in insts {
        match effect(func, slots, subst, inst) {
            Effect::Read(cell, result) => match known.get(&cell) {
                Some(value) => {
                    subst[result.index()] = value;
                    continue;
                }
                None => known.remember(cell, result),
            },
            Effect::Write(cell, value) => known.write(cell, value),
            Effect::Unknown => known.0.clear(),
            Effect::None => {}
        }
        kept.push(inst);
    }
    let dropped = kept.len() != before;
    func.blocks[block.index()].insts = kept;
    dropped
}

/// The frame slot and offset each value holds the address of, when an instruction computes it as one.
fn slot_addresses(func: &Function) -> Vec<Option<(StackSlot, i32)>> {
    let mut slots = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::StackAddr(slot, offset), Some(result)) = (&data.op, data.result) {
            slots[result.index()] = Some((*slot, *offset));
        }
    }
    slots
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between: through each block and on into a successor that only it enters.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.coroutine.is_some() {
        return;
    }
    let preds = pred_counts(func);
    let slots = slot_addresses(func);
    let mut subst = identity(func.types.len());
    let mut dropped = false;
    let mut seen = vec![false; func.blocks.len()];
    let mut stack = vec![(entry, Known::default())];
    while let Some((block, mut known)) = stack.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        dropped |= scan(func, block, (&slots, &mut subst), &mut known);
        let ends_in_call = func.last(block).is_some_and(|op| matches!(op, Op::TryCall { .. }));
        for next in func.successors(block) {
            let inherits = preds[next.index()] == 1 && next != entry && !ends_in_call;
            stack.push((next, if inherits { known.clone() } else { Known::default() }));
        }
    }
    if dropped {
        rename(func, &subst);
    }
}
