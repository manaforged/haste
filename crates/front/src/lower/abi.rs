use anyhow::{Result, bail};
use haste_codegen::{AbiParam, CallConv, Signature, Type, Value};
use rustc_abi::{BackendRepr, Float, Primitive, Scalar, TargetDataLayout};

use crate::sema::layout::target;
use crate::sema::ty::{Kind, Ty};

use super::adt::tag_type;
use super::place::Place;
use super::{Lower, Repr, repr};

/// The machine values that a value of one type travels as in the platform convention, each with its byte offset inside the value.
type Parts = Vec<(Type, i32)>;

fn scalar_type(dl: &TargetDataLayout, scalar: &Scalar) -> Result<Type> {
    match scalar.primitive() {
        Primitive::Float(Float::F32) => Ok(Type::F32),
        Primitive::Float(Float::F64) => Ok(Type::F64),
        Primitive::Float(other) => bail!("a native argument of float type {other:?}"),
        other => tag_type(other.size(dl).bytes()),
    }
}

impl Lower<'_, '_> {
    fn native_parts(&mut self, dl: &TargetDataLayout, ty: Ty) -> Result<Parts> {
        let size = match repr(self.cx, ty)? {
            Repr::Scalar(machine) => return Ok(vec![(machine, 0)]),
            Repr::Zst => return Ok(Vec::new()),
            Repr::Mem { size, .. } => size,
        };
        let layout = self.cx.layout(ty)?;
        match &layout.backend_repr {
            BackendRepr::Scalar(scalar) => Ok(vec![(scalar_type(dl, scalar)?, 0)]),
            BackendRepr::ScalarPair(first, second) => {
                let offset = first.size(dl).align_to(second.align(dl).abi).bytes() as i32;
                Ok(vec![(scalar_type(dl, first)?, 0), (scalar_type(dl, second)?, offset)])
            }
            _ => bail!("a {size}-byte aggregate across a native call is not supported yet"),
        }
    }

    /// Calls the native function at `code`, which takes `params` and returns `ret` in the platform convention, with the generated-code `values` of its arguments.
    pub(super) fn native_call(&mut self, code: Value, (params, ret): (Vec<Ty>, Option<Ty>), values: Vec<Value>) -> Result<Option<Value>> {
        let dl = target()?;
        let mut signature = Signature::new(CallConv::Native);
        let (mut args, mut values) = (Vec::new(), values.into_iter());
        for param in params {
            let parts = self.native_parts(&dl, param)?;
            if parts.is_empty() {
                continue;
            }
            let Some(value) = values.next() else { bail!("a native call with too few arguments") };
            match repr(self.cx, param)? {
                Repr::Scalar(_) => args.push(value),
                _ => args.extend(parts.iter().map(|&(ty, offset)| self.ins().load(ty, value, offset))),
            }
            signature.params.extend(parts.iter().map(|&(ty, _)| AbiParam::new(ty)));
        }
        for value in values {
            signature.params.push(AbiParam::new(self.b.func.ty(value)));
            args.push(value);
        }
        let returned = ret.map(|ret| self.native_parts(&dl, ret)).transpose()?.unwrap_or_default();
        signature.returns.extend(returned.iter().map(|&(ty, _)| AbiParam::new(ty)));
        let sig = self.b.import_signature(signature);
        let call = self.ins().call_indirect(sig, code, &args);
        let results = self.b.inst_results(call).to_vec();
        match ret {
            Some(ret) if matches!(self.cx.types.kind(ret), Kind::Never) => {
                self.ins().trap();
                self.close();
                Ok(None)
            }
            Some(ret) => self.native_result(ret, &returned, &results),
            None => Ok(None),
        }
    }

    fn native_result(&mut self, ret: Ty, parts: &Parts, results: &[Value]) -> Result<Option<Value>> {
        let whole = repr(self.cx, ret)?;
        if !matches!(whole, Repr::Mem { .. }) {
            return Ok(results.first().copied());
        }
        let Some(slot) = self.slot(whole) else { return Ok(None) };
        let place = Place::slot(slot, whole);
        for (&(ty, offset), &value) in parts.iter().zip(results) {
            self.write(place.at(offset, Repr::Scalar(ty)), value);
        }
        Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
    }
}
