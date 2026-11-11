use rustc_hash::FxHashMap as HashMap;

use crate::names::{ItemRef, Primitive};

use super::substs::{Substs, SubstsPool};

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
    Tuple(Substs),
    Ref(Mut, Ty),
    Ptr(Mut, Ty),
    /// An array of `0` whose length is the const value `1`.
    Array(Ty, Ty),
    Slice(Ty),
    Adt(ItemRef, Substs),
    FnDef(ItemRef, Substs),
    FnPtr(Substs, Ty),
    /// The closure expression `2` of the body of `0` at type arguments `1`.
    Closure(ItemRef, Substs, u32),
    /// A trait object of the trait `0`: its trait arguments, then the type bound to each associated type in `2`, with the sorted auto traits `3`.
    Dyn(ItemRef, Substs, Vec<ItemRef>, Vec<ItemRef>),
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
    pub fn children<'a>(&'a self, types: &'a Types) -> impl Iterator<Item = Ty> + 'a {
        let (items, last): (&[Ty], Option<&Ty>) = match self {
            Kind::Tuple(items) | Kind::Adt(_, items) | Kind::FnDef(_, items) | Kind::Dyn(_, items, ..) => (types.substs(*items), None),
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Slice(inner) | Kind::Pat(inner, ..) => (std::slice::from_ref(inner), None),
            Kind::Array(inner, len) => (std::slice::from_ref(inner), Some(len)),
            Kind::FnPtr(params, ret) => (types.substs(*params), Some(ret)),
            _ => (&[], None),
        };
        items.iter().chain(last).copied()
    }

    /// This shape with each type directly inside it replaced by `f` of that type, interning slices in `pool`.
    pub fn map_pool(&self, pool: &SubstsPool, mut f: impl FnMut(Ty) -> Ty) -> Kind {
        match self {
            Kind::Tuple(items) => Kind::Tuple(pool.map(*items, &mut f)),
            Kind::Ref(mutable, inner) => Kind::Ref(*mutable, f(*inner)),
            Kind::Ptr(mutable, inner) => Kind::Ptr(*mutable, f(*inner)),
            Kind::Array(inner, len) => Kind::Array(f(*inner), f(*len)),
            Kind::Slice(inner) => Kind::Slice(f(*inner)),
            Kind::Adt(item, args) => Kind::Adt(*item, pool.map(*args, &mut f)),
            Kind::FnDef(item, args) => Kind::FnDef(*item, pool.map(*args, &mut f)),
            Kind::Dyn(of, args, members, autos) => Kind::Dyn(*of, pool.map(*args, &mut f), members.clone(), autos.clone()),
            Kind::FnPtr(params, ret) => {
                let params = pool.map(*params, &mut f);
                Kind::FnPtr(params, f(*ret))
            }
            Kind::Pat(base, start, end) => Kind::Pat(f(*base), *start, *end),
            other => other.clone(),
        }
    }

    /// Whether a value of this shape has no size known before it runs, so a pointer to it carries metadata.
    pub fn is_unsized(&self) -> bool {
        matches!(self, Kind::Slice(_) | Kind::Prim(Primitive::Str) | Kind::Dyn(..))
    }
}

/// The type interner.
#[derive(Default)]
pub struct Types {
    kinds: Vec<Kind>,
    map: HashMap<Kind, Ty>,
    substs: SubstsPool,
}

impl Types {
    /// Clones the handle to the `[Ty]` slice interner.
    pub fn pool(&self) -> SubstsPool {
        self.substs.clone()
    }

    /// Interns `slice` into a `Substs` handle.
    pub fn intern_substs(&self, slice: &[Ty]) -> Substs {
        self.substs.intern(slice)
    }

    /// The slice of type arguments that `id` names.
    pub fn substs(&self, id: Substs) -> &[Ty] {
        self.substs.get(id)
    }

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
        self.intern(Kind::Tuple(Substs::EMPTY))
    }

    /// The tuple type of `items`.
    pub fn tuple(&mut self, items: &[Ty]) -> Ty {
        let substs = self.intern_substs(items);
        self.intern(Kind::Tuple(substs))
    }

    /// The ADT type `item` at `args`.
    pub fn adt(&mut self, item: ItemRef, args: &[Ty]) -> Ty {
        let substs = self.intern_substs(args);
        self.intern(Kind::Adt(item, substs))
    }

    /// The function pointer type `(params) -> ret`.
    pub fn fn_ptr(&mut self, params: &[Ty], ret: Ty) -> Ty {
        let substs = self.intern_substs(params);
        self.intern(Kind::FnPtr(substs, ret))
    }

    /// The type `bool`.
    pub fn bool(&mut self) -> Ty {
        self.prim(Primitive::Bool)
    }

    /// The array of `element` with the known length `len`.
    pub fn array(&mut self, element: Ty, len: u64) -> Ty {
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
