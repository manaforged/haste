use crate::fold::{constant, defs, extend};
use crate::ir::{Convert, Function, Inst, Op, Value};
use crate::reloc::RuntimeSymbol;

/// The most bytes a `memset` of a constant size is stored inline with.
const FILL_INLINE: i64 = 64;

/// How many definitions deep a size or byte is followed to find its constant.
const DEPTH: u32 = 4;

/// A 64-bit integer whose eight bytes are each one.
const BYTES: i64 = 0x0101_0101_0101_0101;

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

/// Appends to `out` a store of `byte` repeated over each word of the `bytes` bytes at `to`.
fn stores(func: &mut Function, (to, byte, bytes): (Value, i64, u32), out: &mut Vec<Inst>) {
    for (ty, at) in crate::copy::words(bytes) {
        let word = func.value(ty);
        out.push(func.add(Op::Const(extend(byte.wrapping_mul(BYTES), ty, false)), Some(word)));
        out.push(func.add(Op::Store(word, to, at as i32), None));
    }
}

/// Replaces each call of `memset` with a constant byte and a constant size of at most `FILL_INLINE` bytes by stores, so promotion sees the bytes it writes.
pub(crate) fn run(func: &mut Function) {
    if !func.insts.iter().any(|data| matches!(data.op, Op::Runtime(RuntimeSymbol::Memset))) {
        return;
    }
    let defs = defs(func);
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            match fill(func, &defs, inst) {
                Some(found) => stores(func, found, &mut out),
                None => out.push(inst),
            }
        }
        func.blocks[block].insts = out;
    }
}
