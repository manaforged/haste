use crate::ir::{
    AbiParam, Binary, CallConv, Convert, Function, InstData, Op, SigRef, Signature, TargetOpId,
    TinyArgs, Type, Value,
};
use crate::reloc::RuntimeSymbol;

/// A helper compiled into haste that one instruction becomes a call to.
struct Helper {
    symbol: RuntimeSymbol,
    args: Vec<Value>,
    result: Option<Value>,
}

/// Turns each instruction the encoder has no inline form for into a call to a native helper: copies of a size `copy::chunks` does not cover, moves above `copy::MOVE_INLINE` bytes, and 128-bit division and float conversions.
pub(crate) fn lower(func: &mut Function) {
    let mut consts = vec![None; func.types.len()];
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        if let (Op::Const(imm), Some(result)) = (&data.op, data.result) {
            consts[result.index()] = Some(*imm);
        }
    }
    for block in 0..func.blocks.len() {
        let mut at = 0;
        while at < func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[at];
            if let Op::Move(to, from, bytes) = func.insts[inst.index()].op
                && bytes > crate::copy::MOVE_INLINE
            {
                let size = func.value(Type::I64);
                let constant = func.add(Op::Const(i64::from(bytes)), Some(size));
                func.blocks[block].insts.insert(at, constant);
                func.insts[inst.index()].op = Op::Copy(to, from, size);
                at += 1;
            }
            if let Some(helper) = helper(func, &consts, &func.insts[inst.index()]) {
                let sig = signature(func, &helper.args, helper.result);
                let callee = func.value(Type::I64);
                let runtime = func.add(Op::Runtime(helper.symbol), Some(callee));
                func.blocks[block].insts.insert(at, runtime);
                let data = &mut func.insts[inst.index()];
                data.op = Op::Call {
                    sig,
                    callee,
                    args: helper.args,
                    results: helper.result.into_iter().collect(),
                };
                data.result = None;
                at += 1;
            }
            at += 1;
        }
    }
}

fn helper(func: &Function, consts: &[Option<i64>], data: &InstData) -> Option<Helper> {
    let (symbol, args) = match data.op {
        Op::Copy(to, from, bytes)
            if consts
                .get(bytes.index())
                .copied()
                .flatten()
                .and_then(crate::copy::chunks)
                .is_none() =>
        {
            (RuntimeSymbol::Memmove, vec![to, from, bytes])
        }
        Op::Binary(op, x, y) if func.ty(x) == Type::I128 => (wide_binary(op)?, vec![x, y]),
        Op::Convert(op, x) => (convert(op, func.ty(x), func.ty(data.result?))?, vec![x]),
        _ => return None,
    };
    Some(Helper {
        symbol,
        args,
        result: data.result,
    })
}

/// Adds the platform C signature that takes `args` and returns `result` to `func`.
pub(crate) fn signature(func: &mut Function, args: &[Value], result: Option<Value>) -> SigRef {
    let mut signature = Signature::new(CallConv::Native);
    signature.params = args
        .iter()
        .map(|arg| AbiParam::new(func.ty(*arg)))
        .collect();
    signature.returns = result
        .iter()
        .map(|result| AbiParam::new(func.ty(*result)))
        .collect();
    func.sigs.push(signature);
    SigRef(func.sigs.len() as u32 - 1)
}

/// The helper for a 128-bit division or remainder. MIR asserts a nonzero divisor and no signed overflow first, so the helpers wrap instead of panicking.
fn wide_binary(op: Binary) -> Option<RuntimeSymbol> {
    match op {
        Binary::Sdiv => Some(RuntimeSymbol::I128Sdiv),
        Binary::Udiv => Some(RuntimeSymbol::I128Udiv),
        Binary::Srem => Some(RuntimeSymbol::I128Srem),
        Binary::Urem => Some(RuntimeSymbol::I128Urem),
        _ => None,
    }
}

fn convert(op: Convert, from: Type, to: Type) -> Option<RuntimeSymbol> {
    match (op, from, to) {
        (Convert::ToSintSat, Type::F32, Type::I128) => Some(RuntimeSymbol::F32ToI128),
        (Convert::ToSintSat, Type::F64, Type::I128) => Some(RuntimeSymbol::F64ToI128),
        (Convert::ToUintSat, Type::F32, Type::I128) => Some(RuntimeSymbol::F32ToU128),
        (Convert::ToUintSat, Type::F64, Type::I128) => Some(RuntimeSymbol::F64ToU128),
        (Convert::FromSint, Type::I128, Type::F32) => Some(RuntimeSymbol::I128ToF32),
        (Convert::FromSint, Type::I128, Type::F64) => Some(RuntimeSymbol::I128ToF64),
        (Convert::FromUint, Type::I128, Type::F32) => Some(RuntimeSymbol::U128ToF32),
        (Convert::FromUint, Type::I128, Type::F64) => Some(RuntimeSymbol::U128ToF64),
        _ => None,
    }
}

/// Expands each 128-bit unsigned division or remainder by a zero-extended word into a word division of the high half and one x64 `div` of its remainder and the low half, in place of a helper call.
pub(crate) fn divide_by_word(func: &mut Function) -> bool {
    let defs = crate::fold::defs(func);
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let mut at = 0;
        while at < func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[at];
            let (Op::Binary(op @ (Binary::Udiv | Binary::Urem), x, y), Some(_)) = (
                &func.insts[inst.index()].op,
                func.insts[inst.index()].result,
            ) else {
                at += 1;
                continue;
            };
            let (op, x, y) = (*op, *x, *y);
            let word = match defs[y.index()].map(|def| &func.insts[def.index()].op) {
                Some(&Op::Convert(Convert::Uextend, inner)) if func.ty(inner) == Type::I64 => {
                    Some(inner)
                }
                _ => None,
            };
            let Some(by) = word.filter(|_| func.ty(x) == Type::I128) else {
                at += 1;
                continue;
            };
            let mut emit = |func: &mut Function, op: Op, ty: Type| {
                let value = func.value(ty);
                let added = func.add(op, Some(value));
                func.blocks[block].insts.insert(at, added);
                at += 1;
                value
            };
            let shift = emit(func, Op::Const(64), Type::I64);
            let wide_high = emit(func, Op::Binary(Binary::Ushr, x, shift), Type::I128);
            let high = emit(func, Op::Convert(Convert::Ireduce, wide_high), Type::I64);
            let low = emit(func, Op::Convert(Convert::Ireduce, x), Type::I64);
            let quotient = emit(func, Op::Binary(Binary::Udiv, high, by), Type::I64);
            let product = emit(func, Op::Binary(Binary::Imul, quotient, by), Type::I64);
            let rest = emit(func, Op::Binary(Binary::Isub, high, product), Type::I64);
            let id = if op == Binary::Urem {
                TargetOpId::X64RemWide
            } else {
                TargetOpId::X64DivWide
            };
            let part = emit(
                func,
                Op::TargetOp(id, TinyArgs::three(rest, low, by)),
                Type::I64,
            );
            func.insts[inst.index()].op = if op == Binary::Urem {
                Op::Convert(Convert::Uextend, part)
            } else {
                Op::Binary(Binary::Iconcat, part, quotient)
            };
            changed = true;
            at += 1;
        }
    }
    changed
}
