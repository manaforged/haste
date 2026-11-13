use crate::fold::{constant, defs, extend};
use crate::ir::{Convert, Function, Inst, Op, StackSlot, Type, Value};
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

/// Appends to `out` a store of `byte` repeated over each word of the `bytes` bytes at `to`.
fn stores(func: &mut Function, slots: &[Option<(StackSlot, i64)>], (to, byte, bytes): (Value, i64, u32), out: &mut Vec<Inst>) {
    for (ty, at) in crate::copy::words(bytes) {
        let word = func.value(ty);
        out.push(func.add(Op::Const(extend(byte.wrapping_mul(BYTES), ty, false)), Some(word)));
        emit_store(func, slots, word, (to, at as i32), out);
    }
}

fn expand_copy(func: &mut Function, defs: &[Option<Inst>], slots: &[Option<(StackSlot, i64)>], inst: Inst, out: &mut Vec<Inst>) -> bool {
    let Op::Copy(dst, src, len) = func.insts[inst.index()].op else { return false };
    let Some(bytes) = known(func, defs, len, DEPTH) else { return false };
    match bytes {
        0 => true,
        1 | 2 | 4 | 8 => {
            let ty = match bytes {
                1 => Type::I8,
                2 => Type::I16,
                4 => Type::I32,
                _ => Type::I64,
            };
            let val = emit_load(func, slots, (src, 0), ty, out);
            emit_store(func, slots, val, (dst, 0), out);
            true
        }
        16 => {
            let lo = emit_load(func, slots, (src, 0), Type::I64, out);
            let hi = emit_load(func, slots, (src, 8), Type::I64, out);
            emit_store(func, slots, lo, (dst, 0), out);
            emit_store(func, slots, hi, (dst, 8), out);
            true
        }
        _ => false,
    }
}

/// Replaces each `memset` of a constant byte and small constant size by stores, and scalarizes small constant-length `Copy` operations, so promotion sees the bytes they access.
pub(crate) fn run(func: &mut Function) {
    if !func.insts.iter().any(|data| matches!(data.op, Op::Runtime(RuntimeSymbol::Memset) | Op::Copy(..))) {
        return;
    }
    let defs = defs(func);
    let (slots, _) = func.slot_offsets(|v| known(func, &defs, v, DEPTH));
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            if let Some(found) = fill(func, &defs, inst) {
                stores(func, &slots, found, &mut out);
            } else if !expand_copy(func, &defs, &slots, inst, &mut out) {
                out.push(inst);
            }
        }
        func.blocks[block].insts = out;
    }
}
