use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, CallConv, Function, FunctionBuilder, Signature, Type, Value};

use crate::lower::{Repr, repr, signature_of};
use crate::sema::Context;
use crate::sema::ty::Ty;

/// How one argument that Haste's convention passes reaches rustc's code: as it is, or by the address of a value rustc takes in two registers, the second at a byte offset.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Carry {
    Same,
    Split(Type, Type, i32),
}

/// A call's two signatures: Haste's, and rustc's for the same parameters when each reaches it as its carry says, with the carries of the parameters that pass a value and whether the first parameter of both is a result pointer.
pub(crate) struct Shapes {
    pub haste: Signature,
    pub native: Signature,
    pub kept: Vec<Carry>,
    pub sret: bool,
}

/// The signatures of a function taking `params` and returning `ret` in Haste's convention and in rustc's when argument `i` reaches it as `carries[i]` says.
pub(crate) fn shapes(cx: &mut Context<'_>, (params, ret): (&[Ty], Option<Ty>), carries: &[Carry]) -> Result<Shapes> {
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
        let Some(&haste_param) = passed.next() else { bail!("a call with fewer parameters than its function") };
        match carry {
            Carry::Same => native.params.push(haste_param),
            Carry::Split(first, second, _) => native.params.extend([AbiParam::new(first), AbiParam::new(second)]),
        }
        kept.push(carry);
    }
    Ok(Shapes { haste, native, kept, sret: sret.is_some() })
}

/// Calls rustc's code at `code` from Haste's code with the Haste-convention argument values `values`, loading both halves of each argument that `shapes` splits, and returns the call's results.
pub(crate) fn call_native(b: &mut FunctionBuilder<'_>, shapes: &Shapes, code: Value, values: &[Value]) -> Vec<Value> {
    let mut values = values.iter().copied();
    let mut args: Vec<Value> = values.by_ref().take(usize::from(shapes.sret)).collect();
    for (&carry, value) in shapes.kept.iter().zip(values) {
        match carry {
            Carry::Same => args.push(value),
            Carry::Split(first, second, offset) => {
                let low = b.ins().load(first, value, 0);
                let high = b.ins().load(second, value, offset);
                args.extend([low, high]);
            }
        }
    }
    let sig = b.import_signature(shapes.native.clone());
    let call = b.ins().call_indirect(sig, code, &args);
    b.inst_results(call).to_vec()
}

/// A function in Haste's convention for a function taking `params` and returning `ret` that calls rustc's code at `address`, passing each argument as `carries` says.
pub(super) fn adapter(cx: &mut Context<'_>, types: (&[Ty], Option<Ty>), carries: &[Carry], address: usize) -> Result<Function> {
    let shapes = shapes(cx, types, carries)?;
    let mut func = Function::new(shapes.haste.clone());
    let mut b = FunctionBuilder::new(&mut func);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let values = b.block_params(entry).to_vec();
    let callee = b.ins().iconst(Type::I64, address as i64);
    let results = call_native(&mut b, &shapes, callee, &values);
    b.ins().return_(&results);
    Ok(func)
}

/// A function in rustc's convention for a function taking `params` and returning `ret`, receiving each argument as `carries` says, that calls the Haste code whose address the call table entry at `entry` holds, storing both halves of each split argument to memory and passing its address.
pub(super) fn entry(cx: &mut Context<'_>, types: (&[Ty], Option<Ty>), carries: &[Carry], entry: i64) -> Result<Function> {
    let shapes = shapes(cx, types, carries)?;
    let mut func = Function::new(shapes.native.clone());
    let mut b = FunctionBuilder::new(&mut func);
    let block = b.create_block();
    b.append_block_params_for_function_params(block);
    b.switch_to_block(block);
    let mut values = b.block_params(block).to_vec().into_iter();
    let mut args: Vec<Value> = values.by_ref().take(usize::from(shapes.sret)).collect();
    for &carry in &shapes.kept {
        let Some(value) = values.next() else { bail!("an entry with fewer arguments than its function") };
        match carry {
            Carry::Same => args.push(value),
            Carry::Split(first, second, offset) => {
                let Some(high) = values.next() else { bail!("an entry with a split argument missing its second half") };
                let size = offset as u32 + second.bytes();
                let slot = b.create_stack_slot(size, first.bytes().max(second.bytes()).trailing_zeros() as u8);
                let address = b.ins().stack_addr(Type::I64, slot, 0);
                b.ins().store(value, address, 0);
                b.ins().store(high, address, offset);
                args.push(address);
            }
        }
    }
    let table = b.ins().iconst(Type::I64, entry);
    let callee = b.ins().load(Type::I64, table, 0);
    let sig = b.import_signature(shapes.haste.clone());
    let call = b.ins().call_indirect(sig, callee, &args);
    let results = b.inst_results(call).to_vec();
    b.ins().return_(&results);
    Ok(func)
}
