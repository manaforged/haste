use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, CallConv, Signature, Type, Value};
use rustc_abi::{BackendRepr, Float, Primitive, Scalar, TargetDataLayout};

use crate::sema::layout::{Layout, target};
use crate::sema::ty::{Kind, Ty};

use super::adt::tag_type;
use super::place::Place;
use super::{Lower, Parts, Repr, repr};

/// The machine values that a value of one type travels as in the platform convention, each with its byte offset inside the value.
type Passed = Vec<(Type, i32)>;

/// The canonical discriminant and optional payload of `purpose` for signature hashing.
pub(crate) fn purpose_tag(purpose: ArgumentPurpose) -> (u8, Option<u32>) {
    match purpose {
        ArgumentPurpose::Normal => (0, None),
        ArgumentPurpose::StructReturn => (1, None),
        ArgumentPurpose::StructArgument(n) => (2, Some(n)),
        ArgumentPurpose::Dict => (3, None),
        ArgumentPurpose::Variadic => (4, None),
    }
}

pub(super) fn scalar_type(dl: &TargetDataLayout, scalar: &Scalar) -> Result<Type> {
    match scalar.primitive() {
        Primitive::Float(Float::F32) => Ok(Type::F32),
        Primitive::Float(Float::F64) => Ok(Type::F64),
        Primitive::Float(other) => bail!("a native argument of float type {other:?}"),
        other => tag_type(other.size(dl).bytes()),
    }
}

/// The one or two scalars that `layout` names, each with its byte offset, or `Bytes` when it names none.
pub(super) fn parts(layout: &Layout) -> Result<Parts> {
    let dl = target()?;
    Ok(match &layout.backend_repr {
        BackendRepr::Scalar(scalar) => Parts::One(scalar_type(dl, scalar)?),
        BackendRepr::ScalarPair { a, b, b_offset } => Parts::Pair(scalar_type(dl, a)?, scalar_type(dl, b)?, b_offset.bytes() as i32),
        _ => Parts::Bytes,
    })
}

/// The parts that a typed copy of a value of `layout` moves: its scalars, or every byte when both scalars are a union's, whose bytes between them may hold data.
pub(super) fn copied_parts(layout: &Layout) -> Parts {
    match &layout.backend_repr {
        BackendRepr::ScalarPair { a: Scalar::Union { .. }, b: Scalar::Union { .. }, .. } => Parts::Bytes,
        BackendRepr::SimdVector { .. } => match layout.size.bytes() {
            8 => Parts::One(Type::I64),
            16 => Parts::Pair(Type::I64, Type::I64, 8),
            _ => Parts::Bytes,
        },
        _ => parts(layout).unwrap_or(Parts::Bytes),
    }
}

impl Lower<'_, '_> {
    fn native_parts(&mut self, ty: Ty) -> Result<Passed> {
        let size = match repr(self.cx, ty)? {
            Repr::Scalar(machine) => return Ok(vec![(machine, 0)]),
            Repr::Zst => return Ok(Vec::new()),
            Repr::Mem { size, .. } => size,
        };
        let layout = self.cx.layout(ty)?;
        match parts(&layout)? {
            Parts::Bytes => {
                bail!("a {size}-byte aggregate across a native call is not supported yet")
            }
            found => Ok(found.scalars().collect()),
        }
    }

    /// Calls the native function at `code`, which takes `params` and returns `ret` in the platform convention, with the generated-code `values` of its arguments.
    pub(super) fn native_call(&mut self, code: Value, (params, ret): (Vec<Ty>, Option<Ty>), values: Vec<Value>) -> Result<Option<Value>> {
        let mut signature = Signature::new(CallConv::Native);
        let (mut args, mut values) = (Vec::new(), values.into_iter());
        for param in params {
            let parts = self.native_parts(param)?;
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
            let (ty, value) = match self.b.func.ty(value) {
                Type::F32 => (Type::F64, self.ins().fpromote(Type::F64, value)),
                ty => (ty, value),
            };
            signature.params.push(AbiParam::variadic(ty));
            args.push(value);
        }
        let returned = ret.map(|ret| self.native_parts(ret)).transpose()?.unwrap_or_default();
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
            Some(ret) => self.received(ret, &returned, &results),
            None => Ok(None),
        }
    }

    /// The value of a result of type `ret` that a call left in the registers `results`, one per part in `parts`: a frame slot holding them when the result lives in memory.
    pub(super) fn received(&mut self, ret: Ty, parts: &[(Type, i32)], results: &[Value]) -> Result<Option<Value>> {
        let whole = repr(self.cx, ret)?;
        if !matches!(whole, Repr::Mem { .. }) {
            return Ok(results.first().copied());
        }
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let place = Place::slot(slot, whole);
        for (&(ty, offset), &value) in parts.iter().zip(results) {
            self.write(place.at(offset, Repr::Scalar(ty)), value);
        }
        Ok(Some(self.ins().stack_addr(Type::I64, slot, 0)))
    }
}
