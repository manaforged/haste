use rustc_hash::FxHashMap as HashMap;

use crate::names::{ItemRef, Primitive};

/// An interned type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ty(pub u32);

/// Whether a reference or pointer allows writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mut {
    Not,
    Mut,
}

/// The shape of a type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Prim(Primitive),
    Never,
    Tuple(Vec<Ty>),
    Ref(Mut, Ty),
    Ptr(Mut, Ty),
    Array(Ty, u64),
    Slice(Ty),
    Adt(ItemRef, Vec<Ty>),
    FnDef(ItemRef, Vec<Ty>),
    FnPtr(Vec<Ty>, Ty),
    Param(u32),
    Var(u32),
    IntVar(u32),
    FloatVar(u32),
    Error,
}

/// The type interner.
#[derive(Default)]
pub struct Types {
    kinds: Vec<Kind>,
    map: HashMap<Kind, Ty>,
}

impl Types {
    /// The type of shape `kind`.
    pub fn intern(&mut self, kind: Kind) -> Ty {
        if let Some(&ty) = self.map.get(&kind) {
            return ty;
        }
        let ty = Ty(self.kinds.len() as u32);
        self.kinds.push(kind.clone());
        self.map.insert(kind, ty);
        ty
    }

    /// The shape of `ty`.
    pub fn kind(&self, ty: Ty) -> &Kind {
        &self.kinds[ty.0 as usize]
    }

    /// The primitive type `primitive`.
    pub fn prim(&mut self, primitive: Primitive) -> Ty {
        self.intern(Kind::Prim(primitive))
    }

    /// The unit type.
    pub fn unit(&mut self) -> Ty {
        self.intern(Kind::Tuple(Vec::new()))
    }

    /// The type `bool`.
    pub fn bool(&mut self) -> Ty {
        self.prim(Primitive::Bool)
    }

    /// Whether `ty` is `()`.
    pub fn is_unit(&self, ty: Ty) -> bool {
        matches!(self.kind(ty), Kind::Tuple(items) if items.is_empty())
    }

    /// `ty` with every `Param(i)` replaced by `substs[i]`.
    pub fn subst(&mut self, ty: Ty, substs: &[Ty]) -> Ty {
        let kind = self.kind(ty).clone();
        let map = |types: &mut Types, items: &[Ty]| items.iter().map(|item| types.subst(*item, substs)).collect::<Vec<_>>();
        let kind = match kind {
            Kind::Param(index) => return substs.get(index as usize).copied().unwrap_or(ty),
            Kind::Tuple(items) => Kind::Tuple(map(self, &items)),
            Kind::Ref(mutable, inner) => Kind::Ref(mutable, self.subst(inner, substs)),
            Kind::Ptr(mutable, inner) => Kind::Ptr(mutable, self.subst(inner, substs)),
            Kind::Array(inner, len) => Kind::Array(self.subst(inner, substs), len),
            Kind::Slice(inner) => Kind::Slice(self.subst(inner, substs)),
            Kind::Adt(item, args) => Kind::Adt(item, map(self, &args)),
            Kind::FnDef(item, args) => Kind::FnDef(item, map(self, &args)),
            Kind::FnPtr(params, ret) => Kind::FnPtr(map(self, &params), self.subst(ret, substs)),
            other => return self.intern(other),
        };
        self.intern(kind)
    }
}

/// The width in bits and signedness of an integer primitive, with `isize` and `usize` at 64 bits.
pub fn int_info(primitive: Primitive) -> Option<(u32, bool)> {
    use Primitive::*;
    Some(match primitive {
        I8 => (8, true),
        I16 => (16, true),
        I32 => (32, true),
        I64 | Isize => (64, true),
        I128 => (128, true),
        U8 => (8, false),
        U16 => (16, false),
        U32 => (32, false),
        U64 | Usize => (64, false),
        U128 => (128, false),
        _ => return None,
    })
}

/// The width in bits of a float primitive.
pub fn float_bits(primitive: Primitive) -> Option<u32> {
    match primitive {
        Primitive::F16 => Some(16),
        Primitive::F32 => Some(32),
        Primitive::F64 => Some(64),
        Primitive::F128 => Some(128),
        _ => None,
    }
}
