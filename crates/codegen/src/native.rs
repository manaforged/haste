use crate::ir::{AbiParam, Binary, CallConv, Convert, Function, Inst, InstData, Op, SigRef, Signature, Type, Value};

/// A helper compiled into haste that one instruction becomes a call to.
struct Helper {
    address: usize,
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
                let callee = Value(func.types.len() as u32);
                func.types.push(Type::I64);
                func.insts.push(InstData { op: Op::Const(helper.address as i64), result: Some(callee) });
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
    let (address, args) = match data.op {
        Op::Copy(to, from, bytes) if consts.get(bytes.index()).copied().flatten().and_then(crate::copy::chunks).is_none() => {
            (libc::memmove as *const () as usize, vec![to, from, bytes])
        }
        Op::Binary(op, x, y) if func.ty(x) == Type::I128 => (wide_binary(op)?, vec![x, y]),
        Op::Convert(op, x) => (convert(op, func.ty(x), func.ty(data.result?))?, vec![x]),
        _ => return None,
    };
    Some(Helper { address, args, result: data.result })
}

fn signature(func: &mut Function, helper: &Helper) -> SigRef {
    let mut signature = Signature::new(CallConv::Native);
    signature.params = helper.args.iter().map(|arg| AbiParam::new(func.ty(*arg))).collect();
    signature.returns = helper.result.iter().map(|result| AbiParam::new(func.ty(*result))).collect();
    func.sigs.push(signature);
    SigRef(func.sigs.len() as u32 - 1)
}

/// The helper for a 128-bit division or remainder. MIR asserts a nonzero divisor and no signed overflow first, so the helpers wrap instead of panicking.
fn wide_binary(op: Binary) -> Option<usize> {
    let address: extern "C" fn(i128, i128) -> i128 = match op {
        Binary::Sdiv => sdiv,
        Binary::Udiv => udiv,
        Binary::Srem => srem,
        Binary::Urem => urem,
        _ => return None,
    };
    Some(address as usize)
}

fn convert(op: Convert, from: Type, to: Type) -> Option<usize> {
    let address = match (op, from, to) {
        (Convert::ToSintSat, Type::F32, Type::I128) => f32_to_i128 as extern "C" fn(f32) -> i128 as usize,
        (Convert::ToSintSat, Type::F64, Type::I128) => f64_to_i128 as extern "C" fn(f64) -> i128 as usize,
        (Convert::ToUintSat, Type::F32, Type::I128) => f32_to_u128 as extern "C" fn(f32) -> u128 as usize,
        (Convert::ToUintSat, Type::F64, Type::I128) => f64_to_u128 as extern "C" fn(f64) -> u128 as usize,
        (Convert::FromSint, Type::I128, Type::F32) => i128_to_f32 as extern "C" fn(i128) -> f32 as usize,
        (Convert::FromSint, Type::I128, Type::F64) => i128_to_f64 as extern "C" fn(i128) -> f64 as usize,
        (Convert::FromUint, Type::I128, Type::F32) => u128_to_f32 as extern "C" fn(u128) -> f32 as usize,
        (Convert::FromUint, Type::I128, Type::F64) => u128_to_f64 as extern "C" fn(u128) -> f64 as usize,
        _ => return None,
    };
    Some(address)
}

extern "C" fn sdiv(x: i128, y: i128) -> i128 {
    x.wrapping_div(y)
}

extern "C" fn udiv(x: i128, y: i128) -> i128 {
    ((x as u128) / (y as u128)) as i128
}

extern "C" fn srem(x: i128, y: i128) -> i128 {
    x.wrapping_rem(y)
}

extern "C" fn urem(x: i128, y: i128) -> i128 {
    ((x as u128) % (y as u128)) as i128
}

extern "C" fn f32_to_i128(x: f32) -> i128 {
    x as i128
}

extern "C" fn f64_to_i128(x: f64) -> i128 {
    x as i128
}

extern "C" fn f32_to_u128(x: f32) -> u128 {
    x as u128
}

extern "C" fn f64_to_u128(x: f64) -> u128 {
    x as u128
}

extern "C" fn i128_to_f32(x: i128) -> f32 {
    x as f32
}

extern "C" fn i128_to_f64(x: i128) -> f64 {
    x as f64
}

extern "C" fn u128_to_f32(x: u128) -> f32 {
    x as f32
}

extern "C" fn u128_to_f64(x: u128) -> f64 {
    x as f64
}
