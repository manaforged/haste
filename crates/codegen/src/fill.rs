use crate::fold::{constant, defs, extend};
use crate::ir::{Convert, Domain, Function, Inst, Op, StackSlot, Type, Value};
use crate::reloc::RuntimeSymbol;

/// The most bytes a `memset` of a constant size is stored inline with.
const FILL_INLINE: i64 = 64;

/// How many definitions deep a size or byte is followed to find its constant.
const DEPTH: u32 = 4;

/// A 64-bit integer whose eight bytes are each one.
pub(crate) const BYTES: i64 = 0x0101_0101_0101_0101;

/// The constant `value` holds when constants, extensions and arithmetic on constants compute it.
fn known(func: &Function, defs: &[Option<Inst>], value: Value, depth: u32) -> Option<i64> {
    let op = &func.insts[defs[value.index()]?.index()].op;
    let next = depth.checked_sub(1)?;
    match *op {
        Op::Const(imm) => Some(imm),
        Op::Convert(Convert::Uextend, x) => Some(extend(known(func, defs, x, next)?, func.ty(x), false)),
        Op::Convert(Convert::Sextend | Convert::Ireduce, x) => known(func, defs, x, next),
        Op::Binary(op, x, y) if func.ty(x).is_small_int() => constant(op, known(func, defs, x, next)?, known(func, defs, y, next)?),
        _ => None,
    }
}

/// The target, byte and size of `inst` when it calls `memset` with a constant byte and a small constant size.
fn fill(func: &Function, defs: &[Option<Inst>], inst: Inst) -> Option<(Value, i64, u32)> {
    let Op::Call { callee, args, results, .. } = &func.insts[inst.index()].op else { return None };
    let (&[to, byte, bytes], true) = (&args[..], results.is_empty()) else { return None };
    if !matches!(func.insts[defs[callee.index()]?.index()].op, Op::Runtime(RuntimeSymbol::Memset)) {
        return None;
    }
    let size = known(func, defs, bytes, DEPTH).filter(|size| (0..=FILL_INLINE).contains(size))?;
    Some((to, known(func, defs, byte, DEPTH)? & 0xFF, size as u32))
}

fn resolve_slot(slots: &[Option<(StackSlot, i64)>], ptr: Value, off: i32) -> Option<(StackSlot, i32)> {
    let (slot, base) = slots.get(ptr.index()).copied().flatten()?;
    i32::try_from(base).ok()?.checked_add(off).map(|total| (slot, total))
}

fn emit_load(func: &mut Function, slots: &[Option<(StackSlot, i64)>], (src, off): (Value, i32), ty: Type, out: &mut Vec<Inst>) -> Value {
    let val = func.value(ty);
    let op = match resolve_slot(slots, src, off) {
        Some((slot, total)) => Op::StackLoad(slot, total),
        None => Op::Load(src, off),
    };
    out.push(func.add(op, Some(val)));
    val
}

fn emit_store(func: &mut Function, slots: &[Option<(StackSlot, i64)>], val: Value, (dst, off): (Value, i32), out: &mut Vec<Inst>) {
    let op = match resolve_slot(slots, dst, off) {
        Some((slot, total)) => Op::StackStore(val, slot, total),
        None => Op::Store(val, dst, off),
    };
    out.push(func.add(op, None));
}

fn slot_vec_chunks(func: &Function, slots: &[Option<(StackSlot, i64)>]) -> Vec<Vec<(i32, Type)>> {
    let mut chunks: Vec<Vec<(i32, Type)>> = vec![Vec::new(); func.slots.len()];
    let mut record = |slot: StackSlot, off: i32, ty: Type| {
        if ty.is_vector()
            && matches!(ty.bytes(), 8 | 16)
            && let Some(list) = chunks.get_mut(slot.index())
            && !list.iter().any(|&(o, t)| o == off && t.bytes() == ty.bytes())
        {
            list.push((off, ty));
        }
    };
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        match (&data.op, data.result) {
            (&Op::StackLoad(slot, off), Some(res)) => record(slot, off, func.ty(res)),
            (&Op::StackStore(val, slot, off), _) => record(slot, off, func.ty(val)),
            (&Op::Load(addr, off), Some(res)) => {
                if let Some((slot, total)) = resolve_slot(slots, addr, off) {
                    record(slot, total, func.ty(res));
                }
            }
            (&Op::Store(val, addr, off), _) => {
                if let Some((slot, total)) = resolve_slot(slots, addr, off) {
                    record(slot, total, func.ty(val));
                }
            }
            _ => {}
        }
    }
    for list in &mut chunks {
        list.sort_unstable_by_key(|&(off, ty)| (off, std::cmp::Reverse(ty.bytes())));
    }
    chunks
}

fn match_vec_at(
    vec_chunks: &[Vec<(i32, Type)>],
    slots: &[Option<(StackSlot, i64)>],
    (dst, src): (Value, Option<Value>),
    at: u32,
    bytes: u32,
) -> Option<Type> {
    let rem = bytes - at;
    let at_i32 = at as i32;
    let find_end = |ptr: Value, want_bytes: u32| -> Option<Type> {
        let (slot, total) = resolve_slot(slots, ptr, at_i32)?;
        vec_chunks
            .get(slot.index())?
            .iter()
            .copied()
            .find(|&(off, ty)| off == total && ty.bytes() == want_bytes)
            .map(|(_, ty)| ty)
    };
    let find_any = |want_bytes: u32| -> Option<Type> {
        find_end(dst, want_bytes).or_else(|| src.and_then(|s| find_end(s, want_bytes)))
    };
    if rem >= 16
        && let Some(ty) = find_any(16)
    {
        return Some(ty);
    }
    if rem >= 8
        && let Some(ty) = find_any(8)
    {
        return Some(ty);
    }
    None
}

fn vec_pieces(
    vec_chunks: &[Vec<(i32, Type)>],
    slots: &[Option<(StackSlot, i64)>],
    src: Option<Value>,
    dst: Value,
    bytes: u32,
) -> Vec<(Type, u32)> {
    if vec_chunks.iter().all(Vec::is_empty) {
        return crate::copy::words(bytes);
    }
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes {
        if let Some(ty) = match_vec_at(vec_chunks, slots, (dst, src), at, bytes) {
            out.push((ty, at));
            at += ty.bytes();
            continue;
        }
        let next_vec = (at + 1..bytes)
            .find(|&pos| match_vec_at(vec_chunks, slots, (dst, src), pos, bytes).is_some())
            .unwrap_or(bytes);
        out.extend(crate::copy::words(next_vec - at).into_iter().map(|(ty, off)| (ty, at + off)));
        at = next_vec;
    }
    out
}

/// Appends to `out` a store of `byte` repeated over each piece of the `bytes` bytes at `to`.
fn stores(
    func: &mut Function,
    slots: &[Option<(StackSlot, i64)>],
    vec_chunks: &[Vec<(i32, Type)>],
    (to, byte, bytes): (Value, i64, u32),
    out: &mut Vec<Inst>,
) {
    let pieces = if byte == 0 { vec_pieces(vec_chunks, slots, None, to, bytes) } else { crate::copy::words(bytes) };
    for (ty, at) in pieces {
        let imm = if ty.is_vector() { 0 } else { extend(byte.wrapping_mul(BYTES), ty, false) };
        let word = func.value(ty);
        out.push(func.add(Op::Const(imm), Some(word)));
        emit_store(func, slots, word, (to, at as i32), out);
    }
}

fn expand_copy(
    func: &mut Function,
    defs: &[Option<Inst>],
    slots: &[Option<(StackSlot, i64)>],
    vec_chunks: &[Vec<(i32, Type)>],
    inst: Inst,
    out: &mut Vec<Inst>,
) -> bool {
    let Op::Copy(dst, src, len) = func.insts[inst.index()].op else { return false };
    let Some(bytes) = known(func, defs, len, DEPTH).and_then(|n| u32::try_from(n).ok()).filter(|&b| b <= crate::copy::MOVE_INLINE) else {
        return false;
    };
    let pieces = vec_pieces(vec_chunks, slots, Some(src), dst, bytes);
    let loaded: Vec<_> = pieces.iter().map(|&(ty, at)| (emit_load(func, slots, (src, at as i32), ty, out), at as i32)).collect();
    for (val, at) in loaded {
        emit_store(func, slots, val, (dst, at), out);
    }
    true
}

/// Replaces each `memset` of a constant byte and small constant size by stores, and scalarizes small constant-length `Copy` operations, so promotion sees the bytes they access.
pub(crate) fn run(func: &mut Function) -> Domain {
    if !func.insts.iter().any(|data| matches!(data.op, Op::Runtime(RuntimeSymbol::Memset) | Op::Copy(..))) {
        return Domain::NONE;
    }
    let defs = defs(func);
    let (slots, _) = func.slot_offsets(|v| known(func, &defs, v, DEPTH));
    let vec_chunks = slot_vec_chunks(func, &slots);
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            if let Some(found) = fill(func, &defs, inst) {
                stores(func, &slots, &vec_chunks, found, &mut out);
                changed = true;
            } else if expand_copy(func, &defs, &slots, &vec_chunks, inst, &mut out) {
                changed = true;
            } else {
                out.push(inst);
            }
        }
        func.blocks[block].insts = out;
    }
    if changed { Domain::MEM | Domain::VALUES } else { Domain::NONE }
}
