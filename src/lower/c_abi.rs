use haste_codegen::{AbiParam, ArgumentPurpose, Signature, Type};
use hir_def::{AdtId, VariantId};
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::inherent::IntoKind;

use super::Lower;

const WORD: u64 = 8;
const REGISTER_BYTES: u64 = 16;
#[cfg(target_arch = "aarch64")]
const HFA_MEMBERS: usize = 4;

/// How one C argument or return value travels.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Pass {
    Words(Vec<(Type, u32)>),
    Stack(u32),
    Pointer(u32),
}

impl Pass {
    /// Adds the parameters for this value; a return value in memory becomes a struct return parameter.
    pub fn add_param(&self, signature: &mut Signature) {
        match self {
            Pass::Words(words) => signature.params.extend(words.iter().map(|(ty, _)| AbiParam::new(*ty))),
            Pass::Stack(size) => signature.params.push(AbiParam::special(Type::I64, ArgumentPurpose::StructArgument(*size))),
            Pass::Pointer(_) => signature.params.push(AbiParam::new(Type::I64)),
        }
    }

    pub fn add_return(&self, signature: &mut Signature) {
        match self {
            Pass::Words(words) => signature.returns.extend(words.iter().map(|(ty, _)| AbiParam::new(*ty))),
            Pass::Stack(_) | Pass::Pointer(_) => signature.params.insert(0, AbiParam::special(Type::I64, ArgumentPurpose::StructReturn)),
        }
    }

    pub fn in_memory(&self) -> bool {
        !matches!(self, Pass::Words(_))
    }
}

fn int_of(bytes: u64) -> Type {
    match bytes {
        1 => Type::I8,
        2 => Type::I16,
        3 | 4 => Type::I32,
        _ => Type::I64,
    }
}

impl<'db> Lower<'_, '_, 'db> {
    fn leaves(&self, ty: Ty<'db>, base: u64, into: &mut Vec<(u64, Option<Type>)>) -> Result<(), String> {
        let layout = self.layout(ty)?;
        if layout.size.bytes() == 0 {
            return Ok(());
        }
        if let Ok(scalar) = self.scalar(ty) {
            into.push((base, scalar.ty.is_float().then_some(scalar.ty)));
            return Ok(());
        }
        let fields = match ty.kind() {
            TyKind::Adt(def, args) => match def.def_id() {
                AdtId::StructId(id) => self.fields_of(VariantId::StructId(id), args),
                _ => return Ok(into.push((base, None))),
            },
            TyKind::Tuple(items) => items.iter().collect(),
            TyKind::Array(item, _) => {
                let stride = self.layout(item)?.size.bytes().max(1);
                for index in 0..layout.size.bytes() / stride {
                    self.leaves(item, base + index * stride, into)?;
                }
                return Ok(());
            }
            _ => return Ok(into.push((base, None))),
        };
        for (index, field) in fields.into_iter().enumerate() {
            self.leaves(field, base + layout.fields.offset(index).bytes(), into)?;
        }
        Ok(())
    }

    /// Classifies a C value of type `ty`, or `None` when it takes no space.
    pub(super) fn pass_of(&self, ty: Ty<'db>) -> Result<Option<Pass>, String> {
        let size = self.layout(ty)?.size.bytes();
        if size == 0 {
            return Ok(None);
        }
        if let Ok(scalar) = self.scalar(ty) {
            return Ok(Some(Pass::Words(vec![(scalar.ty, 0)])));
        }
        let mut leaves = Vec::new();
        self.leaves(ty, 0, &mut leaves)?;
        Ok(Some(self.classify(size, &leaves)))
    }

    #[cfg(target_arch = "x86_64")]
    fn classify(&self, size: u64, leaves: &[(u64, Option<Type>)]) -> Pass {
        if size > REGISTER_BYTES {
            return Pass::Stack(size as u32);
        }
        let words = (0..size.div_ceil(WORD)).map(|index| {
            let start = index * WORD;
            let bytes = (size - start).min(WORD);
            let inside: Vec<_> = leaves.iter().filter(|(at, _)| *at >= start && *at < start + WORD).collect();
            let all_float = !inside.is_empty() && inside.iter().all(|(_, float)| float.is_some());
            let ty = match (all_float, bytes > 4) {
                (true, true) => Type::F64,
                (true, false) => Type::F32,
                (false, _) => int_of(bytes),
            };
            (ty, start as u32)
        });
        Pass::Words(words.collect())
    }

    #[cfg(target_arch = "aarch64")]
    fn classify(&self, size: u64, leaves: &[(u64, Option<Type>)]) -> Pass {
        let first = leaves.first().and_then(|(_, float)| *float);
        let uniform = first.is_some() && leaves.len() <= HFA_MEMBERS && leaves.iter().all(|(_, float)| *float == first);
        if let (true, Some(float)) = (uniform, first) {
            return Pass::Words(leaves.iter().map(|(at, _)| (float, *at as u32)).collect());
        }
        if size > REGISTER_BYTES {
            return Pass::Pointer(size as u32);
        }
        let words = (0..size.div_ceil(WORD)).map(|index| (int_of((size - index * WORD).min(WORD)), (index * WORD) as u32));
        Pass::Words(words.collect())
    }
}
