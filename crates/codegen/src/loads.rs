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

    /// Whether a write of `self` may change the bytes of `other`: frame slots never overlap each other, an unescaped slot never aliases an address.
    fn clobbers(&self, other: &Cell, unescaped: &[bool]) -> bool {
        match (self.base, other.base) {
            (Base::Slot(a), Base::Slot(b)) => a == b && self.overlaps(other),
            (Base::Addr(a), Base::Addr(b)) if a == b => self.overlaps(other),
            (Base::Addr(_), Base::Slot(b)) | (Base::Slot(b), Base::Addr(_)) => !unescaped[b.index()],
            (Base::Addr(_), Base::Addr(_)) => true,
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

    fn assemble_const(&self, cell: &Cell, consts: &[Option<i64>]) -> Option<i64> {
        let Base::Slot(slot) = cell.base else { return None };
        let width = i64::from(cell.ty.bytes());
        if !cell.ty.is_small_int() || !matches!(width, 2 | 4 | 8) {
            return None;
        }
        let mut assembled = 0_u64;
        for byte in 0..width {
            let at = cell.offset + byte;
            let (kcell, kval) = self.0.iter().find(|(c, _)| {
                c.base == Base::Slot(slot) && c.ty.is_small_int() && c.offset <= at && at < c.offset + i64::from(c.ty.bytes())
            })?;
            let imm = consts[kval.index()]?;
            let shift = 8 * (at - kcell.offset);
            let piece = ((imm as u64) >> shift) & 0xFF;
            assembled |= piece << (8 * byte);
        }
        Some(assembled as i64)
    }

    fn write(&mut self, cell: Cell, value: Value, unescaped: &[bool]) {
        self.0.retain(|(known, _)| !cell.clobbers(known, unescaped));
        self.remember(cell, value);
    }

    fn clobber_slot(&mut self, slot: StackSlot, offset: i64, bytes: i64, unescaped: &[bool]) {
        let end = offset + bytes;
        let slot_escaped = !unescaped[slot.index()];
        self.0.retain(|(known, _)| match known.base {
            Base::Slot(s) if s == slot => !(known.offset < end && offset < known.offset + i64::from(known.ty.bytes())),
            Base::Addr(_) if slot_escaped => false,
            _ => true,
        });
    }
}

/// What one instruction does to memory.
enum Effect {
    None,
    Read(Cell, Value),
    Write(Cell, Value),
    ClobberSlot { slot: StackSlot, offset: i64, bytes: i64 },
    Unknown,
}

/// Whether `op` may write memory it does not name by a plain address, or orders accesses around it.
pub(crate) fn clobbers(op: &Op) -> bool {
    matches!(op, Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence | Op::Copy(..) | Op::Move(..) | Op::Call { .. } | Op::TryCall { .. })
}

struct State<'a> {
    slots: &'a [Option<(StackSlot, i32)>],
    unescaped: &'a [bool],
    selects: &'a [Option<(Value, Value, Value)>],
    consts: &'a mut [Option<i64>],
    subst: &'a mut [Value],
}

fn addr_cell(slots: &[Option<(StackSlot, i32)>], subst: &[Value], addr: Value, offset: i32, ty: Type) -> Cell {
    match slots[addr.index()] {
        Some((slot, base)) => Cell { base: Base::Slot(slot), offset: i64::from(base) + i64::from(offset), ty },
        None => Cell { base: Base::Addr(find(subst, addr)), offset: i64::from(offset), ty },
    }
}

fn effect(func: &Function, slots: &[Option<(StackSlot, i32)>], subst: &[Value], inst: Inst) -> Effect {
    let data = &func.insts[inst.index()];
    let slot = |slot: StackSlot, offset: i32, ty: Type| Cell { base: Base::Slot(slot), offset: i64::from(offset), ty };
    match (&data.op, data.result) {
        (&Op::Load(addr, offset), Some(result)) => Effect::Read(addr_cell(slots, subst, addr, offset, func.ty(result)), result),
        (&Op::StackLoad(from, offset), Some(result)) => Effect::Read(slot(from, offset, func.ty(result)), result),
        (&Op::Store(value, addr, offset), _) => Effect::Write(addr_cell(slots, subst, addr, offset, func.ty(value)), find(subst, value)),
        (&Op::StackStore(value, to, offset), _) => Effect::Write(slot(to, offset, func.ty(value)), find(subst, value)),
        (&Op::Move(to, _, bytes), _) => match slots[to.index()] {
            Some((slot, base)) => Effect::ClobberSlot { slot, offset: i64::from(base), bytes: i64::from(bytes) },
            None => Effect::Unknown,
        },
        (op, _) if clobbers(op) => Effect::Unknown,
        _ => Effect::None,
    }
}

fn flush_reads(pending: &mut Vec<(StackSlot, i64, i64, usize)>, slot: StackSlot, offset: i64, bytes: i64) {
    let end = offset + bytes;
    pending.retain(|&(s, off, len, _)| s != slot || !(off < end && offset < off + len));
}

/// Drops from `block` each load whose bytes `known` holds, recording its replacement in `subst`, eliminates dead overwritten stores to unescaped slots, and leaves in `known` what memory holds at the block's end.
fn scan(func: &mut Function, block: Block, state: &mut State<'_>, known: &mut Known) -> bool {
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut kept: Vec<Option<Inst>> = Vec::with_capacity(insts.len());
    let mut pending_stores: Vec<(StackSlot, i64, i64, usize)> = Vec::new();
    let mut renamed = false;
    for inst in insts {
        match func.insts[inst.index()].op {
            Op::Move(_, from, bytes) => {
                if let Some((slot, base)) = state.slots[from.index()] {
                    flush_reads(&mut pending_stores, slot, i64::from(base), i64::from(bytes));
                }
            }
            Op::TryCall { .. } => pending_stores.clear(),
            _ => {}
        }
        match effect(func, state.slots, state.subst, inst) {
            Effect::Read(cell, result) => match known.get(&cell) {
                Some(value) => {
                    state.subst[result.index()] = value;
                    renamed = true;
                    continue;
                }
                None => {
                    if let Some(imm) = known.assemble_const(&cell, state.consts) {
                        func.insts[inst.index()].op = Op::Const(imm);
                        state.consts[result.index()] = Some(imm);
                        known.remember(cell, result);
                    } else if let Op::Load(addr, offset) = func.insts[inst.index()].op
                        && let Some((c, ptr_a, ptr_b)) = state.selects[find(state.subst, addr).index()]
                        && let (Some(val_a), Some(val_b)) = (
                            known.get(&addr_cell(state.slots, state.subst, ptr_a, offset, cell.ty)),
                            known.get(&addr_cell(state.slots, state.subst, ptr_b, offset, cell.ty)),
                        )
                    {
                        func.insts[inst.index()].op = Op::Select(find(state.subst, c), val_a, val_b);
                        known.remember(cell, result);
                    } else {
                        known.remember(cell, result);
                        if let Base::Slot(s) = cell.base {
                            flush_reads(&mut pending_stores, s, cell.offset, i64::from(cell.ty.bytes()));
                        }
                    }
                }
            },
            Effect::Write(cell, value) => {
                known.write(cell, value, state.unescaped);
                if let Base::Slot(s) = cell.base
                    && state.unescaped[s.index()]
                {
                    let bytes = i64::from(cell.ty.bytes());
                    let end = cell.offset + bytes;
                    pending_stores.retain(|&(ps, off, len, kept_idx)| {
                        if ps == s && off >= cell.offset && off + len <= end {
                            if let Some(dead_store) = kept[kept_idx].take() {
                                func.insts[dead_store.index()].op = Op::Trap;
                            }
                            false
                        } else {
                            true
                        }
                    });
                    pending_stores.push((s, cell.offset, bytes, kept.len()));
                }
            }
            Effect::ClobberSlot { slot, offset, bytes } => known.clobber_slot(slot, offset, bytes, state.unescaped),
            Effect::Unknown => known.0.retain(|(c, _)| matches!(c.base, Base::Slot(s) if state.unescaped[s.index()])),
            Effect::None => {}
        }
        kept.push(Some(inst));
    }
    func.blocks[block.index()].insts = kept.into_iter().flatten().collect();
    renamed
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

/// Whether each stack slot's address never escapes outside direct loads, stores (as address), and moves.
fn unescaped_slots(func: &Function) -> Vec<bool> {
    let mut total_uses = vec![0_u32; func.types.len()];
    let mut safe_uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &func.insts[inst.index()].op;
        op.for_operands(|value| total_uses[value.index()] += 1);
        match *op {
            Op::Load(addr, _) => safe_uses[addr.index()] += 1,
            Op::Store(val, addr, _) if val != addr => safe_uses[addr.index()] += 1,
            Op::Move(to, from, _) => {
                safe_uses[to.index()] += 1;
                safe_uses[from.index()] += 1;
            }
            _ => {}
        }
    }
    let mut unescaped = vec![true; func.slots.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::StackAddr(slot, _), Some(result)) = (&data.op, data.result)
            && safe_uses[result.index()] != total_uses[result.index()]
        {
            unescaped[slot.index()] = false;
        }
    }
    unescaped
}

/// Replaces each load of bytes that an earlier access on every path to it read or wrote with no write of unknown reach between: through each block and on into a successor that only it enters.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    if func.coroutine.is_some() {
        return;
    }
    let preds = pred_counts(func);
    let slots = slot_addresses(func);
    let unescaped = unescaped_slots(func);
    let mut consts = vec![None; func.types.len()];
    let mut selects = vec![None; func.types.len()];
    for data in &func.insts {
        match (&data.op, data.result) {
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Select(c, a, b), Some(result)) => selects[result.index()] = Some((c, a, b)),
            _ => {}
        }
    }
    let mut subst = identity(func.types.len());
    let mut state = State { slots: &slots, unescaped: &unescaped, selects: &selects, consts: &mut consts, subst: &mut subst };
    let mut dropped = false;
    let mut seen = vec![false; func.blocks.len()];
    let mut stack = vec![(entry, Known::default())];
    while let Some((block, mut known)) = stack.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        dropped |= scan(func, block, &mut state, &mut known);
        let ends_in_call = func.last(block).is_some_and(|op| matches!(op, Op::TryCall { .. }));
        for next in func.successors(block) {
            let inherits = preds[next.index()] == 1 && next != entry && !ends_in_call;
            stack.push((next, if inherits { known.clone() } else { Known::default() }));
        }
    }
    if dropped {
        rename(func, state.subst);
    }
}
