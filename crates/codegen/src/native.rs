use crate::ir::{AbiParam, Binary, CallConv, Convert, Function, Inst, InstData, Op, SigRef, Signature, Type, Value};
use crate::reloc::RuntimeSymbol;

/// A helper compiled into haste that one instruction becomes a call to.
struct Helper {
    symbol: RuntimeSymbol,
    args: Vec<Value>,
    result: Option<Value>,
}

/// Turns each instruction the encoder has no inline form for into a call to a native helper: copies of a size `copy::chunks` does not cover, and 128-bit division and float conversions.
pub(crate) fn lower(func: &mut Function) {
    let mut consts = vec![None; func.types.len()];
    for data in &func.insts {
        if let (Op::Const(imm), Some(result)) = (&data.op, data.result) {
            consts[result.index()] = Some(*imm);
        }
    }
    for block in 0..func.blocks.len() {
        let mut at = 0;
        while at < func.blocks[block].insts.len() {
            let inst = func.blocks[block].insts[at];
            if let Some(helper) = helper(func, &consts, &func.insts[inst.index()]) {
                let sig = signature(func, &helper);
                let callee = func.value(Type::I64);
                func.insts.push(InstData { op: Op::Runtime(helper.symbol), result: Some(callee) });
                func.blocks[block].insts.insert(at, Inst(func.insts.len() as u32 - 1));
                let data = &mut func.insts[inst.index()];
                data.op = Op::Call { sig, callee, args: helper.args, results: helper.result.into_iter().collect() };
                data.result = None;
                at += 1;
            }
            at += 1;
        }
    }
}

fn helper(func: &Function, consts: &[Option<i64>], data: &InstData) -> Option<Helper> {
    let (symbol, args) = match data.op {
        Op::Copy(to, from, bytes) if consts.get(bytes.index()).copied().flatten().and_then(crate::copy::chunks).is_none() => {
            (RuntimeSymbol::Memmove, vec![to, from, bytes])
        }
        Op::Binary(op, x, y) if func.ty(x) == Type::I128 => (wide_binary(op)?, vec![x, y]),
        Op::Convert(op, x) => (convert(op, func.ty(x), func.ty(data.result?))?, vec![x]),
        _ => return None,
    };
    Some(Helper { symbol, args, result: data.result })
}

fn signature(func: &mut Function, helper: &Helper) -> SigRef {
    let mut signature = Signature::new(CallConv::Native);
    signature.params = helper.args.iter().map(|arg| AbiParam::new(func.ty(*arg))).collect();
    signature.returns = helper.result.iter().map(|result| AbiParam::new(func.ty(*result))).collect();
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
