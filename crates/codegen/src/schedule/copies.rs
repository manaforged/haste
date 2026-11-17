use crate::ir::{Binary, Function, Inst, Lane, Op, Type, Value};

const PAIR: Type = Type::Vec { lane: Lane::I64, count: 2 };

/// Whether `op` neither writes memory nor has another effect that a load may not move across.
fn movable(op: &Op) -> bool {
    match op {
        Op::Binary(bin, ..) => !matches!(bin, Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem),
        Op::Load(..) | Op::StackLoad(..) | Op::Unary(..) | Op::Convert(..) | Op::Icmp(..) | Op::Select(..) | Op::Const(_) | Op::Symbol(_) | Op::StackAddr(..) | Op::SlotAddr(_) => true,
        _ => false,
    }
}

/// Whether `addr` points into a stack slot, through up to four sums.
fn on_stack(func: &Function, defs: &[Option<Inst>], mut addr: Value) -> bool {
    for _ in 0..4 {
        let Some(inst) = defs.get(addr.index()).copied().flatten() else { return false };
        match func.insts[inst.index()].op {
            Op::StackAddr(..) | Op::SlotAddr(_) => return true,
            Op::Binary(Binary::Iadd, base, _) => addr = base,
            _ => return false,
        }
    }
    false
}

/// The `I64` store at `inst` as its value, address and offset.
fn store(func: &Function, inst: Inst) -> Option<(Value, Value, i32)> {
    match func.insts[inst.index()].op {
        Op::Store(value, addr, off) if func.ty(value) == Type::I64 => Some((value, addr, off)),
        _ => None,
    }
}

/// The position in `insts` of the single-use `I64` load that defines `value`, and its address and offset.
fn load(func: &Function, insts: &[Inst], uses: &[u32], value: Value) -> Option<(usize, Value, i32)> {
    let at = insts.iter().position(|&inst| func.insts[inst.index()].result == Some(value))?;
    match func.insts[insts[at].index()].op {
        Op::Load(addr, off) if uses[value.index()] == 1 => Some((at, addr, off)),
        _ => None,
    }
}

/// The new block order when the stores at `at` and `at + 1` copy two adjacent quadwords loaded earlier in the block with nothing between that writes memory.
fn pair_at(func: &mut Function, (insts, uses, defs): (&[Inst], &[u32], &[Option<Inst>]), at: usize) -> Option<Vec<Inst>> {
    let (mut first, mut second) = (store(func, insts[at])?, store(func, *insts.get(at + 1)?)?);
    if first.2 > second.2 {
        std::mem::swap(&mut first, &mut second);
    }
    let (low, high) = (load(func, insts, uses, first.0)?, load(func, insts, uses, second.0)?);
    let adjacent = |a: (Value, i32), b: (Value, i32)| a.0 == b.0 && i64::from(a.1) + 8 == i64::from(b.1);
    if !adjacent((first.1, first.2), (second.1, second.2)) || !adjacent((low.1, low.2), (high.1, high.2)) || on_stack(func, defs, first.1) || on_stack(func, defs, low.1) {
        return None;
    }
    let from = low.0.min(high.0);
    if from >= at || !insts[from..at].iter().all(|&inst| movable(&func.insts[inst.index()].op)) {
        return None;
    }
    let wide = func.value(PAIR);
    let read = func.add(Op::Load(low.1, low.2), Some(wide));
    let write = func.add(Op::Store(wide, first.1, first.2), None);
    let mut out: Vec<Inst> = insts.iter().enumerate().filter(|&(pos, _)| pos != low.0 && pos != high.0 && pos != at && pos != at + 1).map(|(_, &inst)| inst).collect();
    let before = (0..at).filter(|&pos| pos != low.0 && pos != high.0).count();
    out.splice(before..before, [read, write]);
    Some(out)
}

/// Turns each pair of stores in a loop that copies two adjacent quadwords loaded in the same block into one 16-byte load and store.
pub(super) fn pair(func: &mut Function) {
    let uses = func.use_counts();
    let defs = crate::fold::defs(func);
    if func.entry.is_none() {
        return;
    }
    let depths = func.cfg().loops().depths.clone();
    for block in (0..func.blocks.len()).filter(|&block| depths.get(block).is_some_and(|&depth| depth > 0)) {
        let mut at = 0;
        while at + 1 < func.blocks[block].insts.len() {
            let insts = func.blocks[block].insts.clone();
            match pair_at(func, (&insts, &uses, &defs), at) {
                Some(out) => func.blocks[block].insts = out,
                None => at += 1,
            }
        }
    }
}
