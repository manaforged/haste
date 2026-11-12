use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, CallConv, Function, FunctionBuilder, Signature, Type};

use crate::lower::{Repr, repr, signature_of};
use crate::sema::Context;
use crate::sema::ty::Ty;

/// How one argument that Haste's convention passes reaches rustc's code: as it is, or by the address of a value rustc takes in two registers, the second at a byte offset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Carry {
    Same,
    Split(Type, Type, i32),
}

/// A function in Haste's convention for a function taking `params` and returning `ret` that calls rustc's code at `address`, passing each argument as `carries` says.
pub(super) fn adapter(cx: &mut Context<'_>, (params, ret): (&[Ty], Option<Ty>), carries: &[Carry], address: usize) -> Result<Function> {
    let haste = signature_of(cx, ret, params.iter().copied())?;
    let mut native = Signature::new(CallConv::Native);
    native.returns = haste.returns.clone();
    let mut passed = haste.params.iter();
    let sret = haste.params.first().filter(|param| param.purpose == ArgumentPurpose::StructReturn).copied();
    if let Some(sret) = sret {
        native.params.push(sret);
        passed.next();
    }
    let mut kept = Vec::new();
    for (&param, &carry) in params.iter().zip(carries) {
        if matches!(repr(cx, param)?, Repr::Zst) {
            continue;
        }
        let Some(&haste_param) = passed.next() else { bail!("an adapter with fewer parameters than its function") };
        match carry {
            Carry::Same => native.params.push(haste_param),
            Carry::Split(first, second, _) => native.params.extend([AbiParam::new(first), AbiParam::new(second)]),
        }
        kept.push(carry);
    }
    let mut func = Function::new(haste);
    let mut b = FunctionBuilder::new(&mut func);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let values = b.block_params(entry).to_vec();
    let (mut args, mut values) = (Vec::new(), values.into_iter());
    args.extend(sret.and_then(|_| values.next()));
    for (carry, value) in kept.into_iter().zip(values) {
        match carry {
            Carry::Same => args.push(value),
            Carry::Split(first, second, offset) => {
                let low = b.ins().load(first, value, 0);
                let high = b.ins().load(second, value, offset);
                args.extend([low, high]);
            }
        }
    }
    let sig = b.import_signature(native);
    let callee = b.ins().iconst(Type::I64, address as i64);
    let call = b.ins().call_indirect(sig, callee, &args);
    let results = b.inst_results(call).to_vec();
    b.ins().return_(&results);
    Ok(func)
}
