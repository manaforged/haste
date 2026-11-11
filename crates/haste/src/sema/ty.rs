use std::sync::Arc;

use crate::names::{ItemRef, Primitive};
use crate::sync::Interner;

/// An interned type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ty(pub u32);

impl Ty {
    /// The type of an expression or result that inference gave no type.
    pub const NONE: Ty = Ty(u32::MAX);
}

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
    /// An array of `0` whose length is the const value `1`.
    Array(Ty, Ty),
    Slice(Ty),
    Adt(ItemRef, Vec<Ty>),
    FnDef(ItemRef, Vec<Ty>),
    FnPtr(Vec<Ty>, Ty),
    /// The closure expression `2` of the body of `0` at type arguments `1`.
    Closure(ItemRef, Vec<Ty>, u32),
    /// A trait object of the trait `0`: its trait arguments, then the type bound to each associated type in `2`, with the sorted auto traits `3`.
    Dyn(ItemRef, Vec<Ty>, Vec<ItemRef>, Vec<ItemRef>),
    Pat(Ty, u128, u128),
    /// The value of a const generic argument, as the bits of its type.
    Value(u128),
    Var(u32),
    IntVar(u32),
    FloatVar(u32),
}

/// The trait arguments of a trait object's types `args`, and the types bound to its associated types `members`.
pub fn object_parts<'k>(args: &'k [Ty], members: &'k [ItemRef]) -> (&'k [Ty], impl Iterator<Item = (ItemRef, Ty)> + 'k) {
    let (trait_args, bound) = args.split_at(args.len() - members.len());
    (trait_args, members.iter().copied().zip(bound.iter().copied()))
}

impl Kind {
    /// The types directly inside this shape.
    pub fn children(&self) -> impl Iterator<Item = Ty> + '_ {
        let (items, last): (&[Ty], Option<&Ty>) = match self {
            Kind::Tuple(items) | Kind::Adt(_, items) | Kind::FnDef(_, items) | Kind::Closure(_, items, _) | Kind::Dyn(_, items, ..) => (items, None),
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Pat(inner, ..) => (std::slice::from_ref(inner), None),
            Kind::Array(inner, len) => (std::slice::from_ref(inner), Some(len)),
            Kind::FnPtr(params, ret) => (params, Some(ret)),
            _ => (&[], None),
        };
        items.iter().chain(last).copied()
    }

    /// This shape with each type directly inside it replaced by `f` of that type.
    pub fn map(self, mut f: impl FnMut(Ty) -> Ty) -> Kind {
        match self {
            Kind::Tuple(items) => Kind::Tuple(items.into_iter().map(&mut f).collect()),
            Kind::Ref(mutable, inner) => Kind::Ref(mutable, f(inner)),
            Kind::Ptr(mutable, inner) => Kind::Ptr(mutable, f(inner)),
            Kind::Array(inner, len) => Kind::Array(f(inner), f(len)),
            Kind::Slice(inner) => Kind::Slice(f(inner)),
            Kind::Adt(item, args) => Kind::Adt(item, args.into_iter().map(&mut f).collect()),
            Kind::FnDef(item, args) => Kind::FnDef(item, args.into_iter().map(&mut f).collect()),
            Kind::Closure(item, args, id) => Kind::Closure(item, args.into_iter().map(&mut f).collect(), id),
            Kind::Dyn(of, args, members, autos) => Kind::Dyn(of, args.into_iter().map(&mut f).collect(), members, autos),
            Kind::FnPtr(params, ret) => {
                let params = params.into_iter().map(&mut f).collect();
                Kind::FnPtr(params, f(ret))
            }
            Kind::Pat(base, start, end) => Kind::Pat(f(base), start, end),
            other => other,
        }
    }

    /// Whether a value of this shape has no size known before it runs, so a pointer to it carries metadata.
    pub fn is_unsized(&self) -> bool {
        matches!(self, Kind::Slice(_) | Kind::Prim(Primitive::Str) | Kind::Dyn(..))
    }
}

const TY_SHARDS: usize = 64;
const TY_SLAB: usize = 65_536;
const TY_SLABS: usize = 256;

const ALL_PRIMITIVES: [Primitive; 19] = [
    Primitive::Bool,
    Primitive::Char,
    Primitive::Str,
    Primitive::I8,
    Primitive::I16,
    Primitive::I32,
    Primitive::I64,
    Primitive::I128,
    Primitive::Isize,
    Primitive::U8,
    Primitive::U16,
    Primitive::U32,
    Primitive::U64,
    Primitive::U128,
    Primitive::Usize,
    Primitive::F16,
    Primitive::F32,
    Primitive::F64,
    Primitive::F128,
];

const UNIT_TY: Ty = Ty(19);
const NEVER_TY: Ty = Ty(20);
const VAR_BASE: u32 = 21;
const FAST_VARS: u32 = 64;
const INT_VAR_BASE: u32 = VAR_BASE + FAST_VARS;
const FAST_INT_VARS: u32 = 32;
const FLOAT_VAR_BASE: u32 = INT_VAR_BASE + FAST_INT_VARS;
const FAST_FLOAT_VARS: u32 = 16;
const VALUE_BASE: u32 = FLOAT_VAR_BASE + FAST_FLOAT_VARS;
const FAST_VALUES: u32 = 16;

/// The type interner.
#[derive(Clone)]
pub struct Types {
    inner: Arc<Interner<Kind, TY_SHARDS, TY_SLAB, TY_SLABS>>,
}

impl Default for Types {
    fn default() -> Self {
        let types = Self { inner: Arc::new(Interner::default()) };
        let fixed = ALL_PRIMITIVES
            .into_iter()
            .map(Kind::Prim)
            .chain([Kind::Tuple(Vec::new()), Kind::Never])
            .chain((0..FAST_VARS).map(Kind::Var))
            .chain((0..FAST_INT_VARS).map(Kind::IntVar))
            .chain((0..FAST_FLOAT_VARS).map(Kind::FloatVar))
            .chain((0..FAST_VALUES).map(|bits| Kind::Value(u128::from(bits))));
        for kind in fixed {
            types.intern_slow(kind);
        }
        types
    }
}

fn prim_index(primitive: Primitive) -> u32 {
    primitive as u32
}

fn fast_ty(kind: &Kind) -> Option<Ty> {
    match *kind {
        Kind::Prim(primitive) => Some(Ty(prim_index(primitive))),
        Kind::Tuple(ref items) if items.is_empty() => Some(UNIT_TY),
        Kind::Never => Some(NEVER_TY),
        Kind::Var(index) if index < FAST_VARS => Some(Ty(VAR_BASE + index)),
        Kind::IntVar(index) if index < FAST_INT_VARS => Some(Ty(INT_VAR_BASE + index)),
        Kind::FloatVar(index) if index < FAST_FLOAT_VARS => Some(Ty(FLOAT_VAR_BASE + index)),
        Kind::Value(bits) if bits < u128::from(FAST_VALUES) => Some(Ty(VALUE_BASE + bits as u32)),
        _ => None,
    }
}

impl Types {
    /// The type of shape `kind`.
    pub fn intern(&self, kind: Kind) -> Ty {
        if let Some(ty) = fast_ty(&kind) {
            return ty;
        }
        self.intern_slow(kind)
    }

    fn intern_slow(&self, kind: Kind) -> Ty {
        Ty(self.inner.intern(&kind, Kind::clone))
    }

    /// The shape of `ty`.
    pub fn kind(&self, ty: Ty) -> &Kind {
        self.inner.get(ty.0)
    }

    /// The primitive type `primitive`.
    pub fn prim(&self, primitive: Primitive) -> Ty {
        Ty(prim_index(primitive))
    }

    /// The unit type.
    pub fn unit(&self) -> Ty {
        UNIT_TY
    }

    /// The type `bool`.
    pub fn bool(&self) -> Ty {
        Ty(0)
    }

    /// The array of `element` with the known length `len`.
    pub fn array(&self, element: Ty, len: u64) -> Ty {
        let len = self.intern(Kind::Value(u128::from(len)));
        self.intern(Kind::Array(element, len))
    }

    /// The length that the const value `len` of an array type holds, once it is known.
    pub fn len(&self, len: Ty) -> Option<u64> {
        match *self.kind(len) {
            Kind::Value(bits) => u64::try_from(bits).ok(),
            _ => None,
        }
    }

    /// Whether `ty` holds any unbound inference variable.
    pub fn has_vars(&self, ty: Ty) -> bool {
        match self.kind(ty) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => true,
            other => other.children().any(|child| self.has_vars(child)),
        }
    }
}

/// The value with the low `bits` bits set: the mask of an integer that wide.
pub fn mask(bits: u32) -> u128 {
    if bits == 128 { u128::MAX } else { (1 << bits) - 1 }
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
