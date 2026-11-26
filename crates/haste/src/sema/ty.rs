use std::sync::Arc;

use crate::names::{ItemRef, PRIMITIVES, Primitive};
use crate::sync::{AtomicU8Slab, AtomicU32Slab, Interner};

use super::substs::{
    ItemList, ItemListPool, PatPool, PatRange, ProjDef, ProjId, ProjPool, Substs, SubstsPool,
};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    UnsafeFnPtr(Substs, Ty),
    /// The closure expression `2` of the body of `0` at type arguments `1`.
    Closure(ItemRef, Substs, u32),
    /// A trait object of the trait `0`: its trait arguments, then the type bound to each associated type in `2`, with the sorted auto traits `3`.
    Dyn(ItemRef, Substs, ItemList, ItemList),
    Pat(Ty, PatRange),
    /// The value of a const generic argument, as the bits of its type.
    Value(u128),
    /// A rigid generic type or const parameter by its slot in the item's substitutions.
    Param(u16),
    /// A rigid unnormalized associated type projection.
    Proj(ProjId),
    Var(u32),
    IntVar(u32),
    FloatVar(u32),
}

/// The trait arguments of a trait object's types `args`, and the types bound to its associated types `members`.
pub fn object_parts<'k>(
    args: &'k [Ty],
    members: &'k [ItemRef],
) -> (&'k [Ty], impl Iterator<Item = (ItemRef, Ty)> + 'k) {
    let (trait_args, bound) = args.split_at(args.len() - members.len());
    (
        trait_args,
        members.iter().copied().zip(bound.iter().copied()),
    )
}

impl Kind {
    /// The types directly inside this shape.
    pub fn children<'a>(&'a self, types: &'a Types) -> impl Iterator<Item = Ty> + 'a {
        let (first, items, extra): (Option<Ty>, &[Ty], &[Ty]) = match self {
            Kind::Tuple(items)
            | Kind::Adt(_, items)
            | Kind::FnDef(_, items)
            | Kind::Closure(_, items, _)
            | Kind::Dyn(_, items, ..) => (None, types.substs(*items), &[]),
            Kind::Ref(_, inner)
            | Kind::Ptr(_, inner)
            | Kind::Slice(inner)
            | Kind::Pat(inner, ..) => (Some(*inner), &[], &[]),
            Kind::Array(inner, len) => (Some(*inner), std::slice::from_ref(len), &[]),
            Kind::FnPtr(params, ret) | Kind::UnsafeFnPtr(params, ret) => {
                (Some(*ret), types.substs(*params), &[])
            }
            Kind::Proj(id) => {
                let proj = types.proj(*id);
                (
                    Some(proj.self_ty),
                    types.substs(proj.args),
                    types.substs(proj.own),
                )
            }
            _ => (None, &[], &[]),
        };
        first
            .into_iter()
            .chain(items.iter().copied())
            .chain(extra.iter().copied())
    }

    /// Whether a value of this shape has no size known before it runs, so a pointer to it carries metadata.
    pub fn is_unsized(&self) -> bool {
        matches!(
            self,
            Kind::Slice(_) | Kind::Prim(Primitive::Str) | Kind::Dyn(..)
        )
    }
}

const FLAG_PARAM: u8 = 1;
const FLAG_VAR: u8 = 2;
const FLAG_PROJ: u8 = 4;
const FLAG_CLOSURE: u8 = 8;
const FLAG_NONE: u8 = 16;
const FLAG_SET: u8 = 0x80;

const TY_SHARDS: usize = 128;
const TY_SLAB: usize = 65_536;
const TY_SLABS: usize = 256;

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
const PARAM_BASE: u32 = VALUE_BASE + FAST_VALUES;
const FAST_PARAMS: u32 = 32;

#[derive(Default)]
struct UnaryCache([AtomicU32Slab<TY_SLAB, TY_SLABS>; 7]);

/// The type interner.
#[derive(Clone)]
pub struct Types {
    inner: Arc<Interner<Kind, TY_SHARDS, TY_SLAB, TY_SLABS>>,
    flags: Arc<AtomicU8Slab<TY_SLAB, TY_SLABS>>,
    depths: Arc<AtomicU8Slab<TY_SLAB, TY_SLABS>>,
    unary: Arc<UnaryCache>,
    substs: SubstsPool,
    projs: ProjPool,
    items: ItemListPool,
    pats: PatPool,
}

impl Default for Types {
    fn default() -> Self {
        let types = Self {
            inner: Arc::default(),
            flags: Arc::default(),
            depths: Arc::default(),
            unary: Arc::default(),
            substs: SubstsPool::default(),
            projs: ProjPool::default(),
            items: ItemListPool::default(),
            pats: PatPool::default(),
        };
        let fixed = PRIMITIVES
            .iter()
            .map(|&(_, p)| Kind::Prim(p))
            .chain([Kind::Tuple(Substs::EMPTY), Kind::Never])
            .chain((0..FAST_VARS).map(Kind::Var))
            .chain((0..FAST_INT_VARS).map(Kind::IntVar))
            .chain((0..FAST_FLOAT_VARS).map(Kind::FloatVar))
            .chain((0..FAST_VALUES).map(|bits| Kind::Value(u128::from(bits))))
            .chain((0..FAST_PARAMS as u16).map(Kind::Param));
        for kind in fixed {
            types.intern_slow(kind);
        }
        types
    }
}

fn unary_slot(kind: &Kind) -> Option<(usize, u32)> {
    match *kind {
        Kind::Ref(Mut::Not, inner) if inner != Ty::NONE => Some((0, inner.0)),
        Kind::Ref(Mut::Mut, inner) if inner != Ty::NONE => Some((1, inner.0)),
        Kind::Ptr(Mut::Not, inner) if inner != Ty::NONE => Some((2, inner.0)),
        Kind::Ptr(Mut::Mut, inner) if inner != Ty::NONE => Some((3, inner.0)),
        Kind::Slice(inner) if inner != Ty::NONE => Some((4, inner.0)),
        Kind::Tuple(substs) if !substs.is_empty() => Some((5, substs.0)),
        Kind::Proj(id) => Some((6, id.0)),
        _ => None,
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
        Kind::Param(index) if u32::from(index) < FAST_PARAMS => {
            Some(Ty(PARAM_BASE + u32::from(index)))
        }
        _ => None,
    }
}

impl Types {
    /// Interns `slice` into a `Substs` handle.
    pub fn intern_substs(&self, slice: &[Ty]) -> Substs {
        self.substs.intern(slice)
    }

    /// The slice of type arguments that `id` names.
    pub fn substs(&self, id: Substs) -> &[Ty] {
        self.substs.get(id)
    }

    /// Maps `f` across `id` in place without cloning the interner handle.
    pub fn map_substs(&self, id: Substs, f: &mut impl FnMut(Ty) -> Ty) -> Substs {
        self.substs.map(id, f)
    }

    /// Interns `def` into a `ProjId` handle.
    pub fn intern_proj(&self, def: ProjDef) -> ProjId {
        self.projs.intern(def)
    }

    /// The projection definition that `id` names.
    pub fn proj(&self, id: ProjId) -> ProjDef {
        self.projs.get(id)
    }

    /// Interns `slice` into an `ItemList` handle.
    pub fn intern_items(&self, slice: &[ItemRef]) -> ItemList {
        self.items.intern(slice)
    }

    /// The slice of item references that `id` names.
    pub fn items(&self, id: ItemList) -> &[ItemRef] {
        self.items.get(id)
    }

    /// Interns `range` into a `PatRange` handle.
    pub fn intern_pat(&self, range: (u128, u128)) -> PatRange {
        self.pats.intern(range)
    }

    /// The `(start, end)` range that `id` names.
    pub fn pat(&self, id: PatRange) -> (u128, u128) {
        self.pats.get(id)
    }

    #[inline]
    fn flags_of(&self, ty: Ty) -> u8 {
        if ty == Ty::NONE {
            FLAG_NONE
        } else {
            self.flags.get(ty.0 as usize).unwrap_or(0) & !FLAG_SET
        }
    }

    fn substs_flags(&self, substs: Substs) -> u8 {
        self.substs(substs)
            .iter()
            .fold(0, |acc, &ty| acc | self.flags_of(ty))
    }

    /// Whether `ty` contains any rigid `Param` or `Proj` at any depth.
    pub fn has_params(&self, ty: Ty) -> bool {
        (self.flags_of(ty) & FLAG_PARAM) != 0
    }

    /// Whether `ty` syntactically contains any inference variable slot at any depth.
    pub fn has_var_flag(&self, ty: Ty) -> bool {
        (self.flags_of(ty) & FLAG_VAR) != 0
    }

    /// Whether `ty` contains any rigid `Proj` at any depth.
    pub fn has_projs(&self, ty: Ty) -> bool {
        (self.flags_of(ty) & FLAG_PROJ) != 0
    }

    /// Whether `ty` contains no parameters, inference variables, closures, or `Ty::NONE` at any depth.
    pub fn is_ground(&self, ty: Ty) -> bool {
        (self.flags_of(ty) & (FLAG_PARAM | FLAG_VAR | FLAG_CLOSURE | FLAG_NONE)) == 0
    }

    /// Whether `ty` contains a `Closure` type at any depth.
    pub fn has_closure(&self, ty: Ty) -> bool {
        (self.flags_of(ty) & FLAG_CLOSURE) != 0
    }

    /// Whether any type in `substs` contains a rigid `Param` or `Proj`.
    pub fn substs_have_params(&self, substs: Substs) -> bool {
        (self.substs_flags(substs) & FLAG_PARAM) != 0
    }

    fn kind_flags(&self, kind: &Kind) -> u8 {
        match kind {
            Kind::Param(_) => FLAG_PARAM,
            Kind::Proj(id) => {
                let proj = self.proj(*id);
                let sub = self.flags_of(proj.self_ty)
                    | self.substs_flags(proj.args)
                    | self.substs_flags(proj.own);
                FLAG_PARAM | FLAG_PROJ | (sub & (FLAG_VAR | FLAG_CLOSURE | FLAG_NONE))
            }
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => FLAG_VAR,
            Kind::Closure(..) => {
                FLAG_CLOSURE
                    | kind
                        .children(self)
                        .fold(0, |acc, child| acc | self.flags_of(child))
            }
            other => other
                .children(self)
                .fold(0, |acc, child| acc | self.flags_of(child)),
        }
    }

    /// The type of shape `kind`.
    pub fn intern(&self, kind: Kind) -> Ty {
        if let Some(ty) = fast_ty(&kind) {
            return ty;
        }
        if let Some((slot, key)) = unary_slot(&kind) {
            if let Some(raw) = self.unary.0[slot].get(key as usize) {
                return Ty(raw - 1);
            }
            let ty = self.intern_slow(kind);
            self.unary.0[slot].set(key as usize, ty.0 + 1);
            return ty;
        }
        self.intern_slow(kind)
    }

    /// The nesting depth of `ty`, saturated at `u8::MAX`.
    pub fn ty_depth(&self, ty: Ty) -> usize {
        if ty == Ty::NONE {
            0
        } else {
            usize::from(self.depths.get(ty.0 as usize).unwrap_or(0))
        }
    }

    /// The maximum nesting depth of any type in `substs`.
    pub fn substs_depth(&self, substs: Substs) -> usize {
        self.substs(substs)
            .iter()
            .fold(0usize, |acc, &ty| acc.max(self.ty_depth(ty)))
    }

    fn kind_depth(&self, kind: &Kind) -> u8 {
        kind.children(self)
            .fold(0u8, |acc, child| acc.max(self.ty_depth(child) as u8))
            .saturating_add(1)
    }

    fn intern_slow(&self, kind: Kind) -> Ty {
        Ty(self.inner.intern_owned(kind, |kind, id| {
            self.flags
                .set(id as usize, self.kind_flags(kind) | FLAG_SET);
            self.depths.set(id as usize, self.kind_depth(kind));
        }))
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

    /// The tuple type of `items`.
    pub fn tuple(&self, items: &[Ty]) -> Ty {
        let substs = self.intern_substs(items);
        self.intern(Kind::Tuple(substs))
    }

    /// The ADT type `item` at `args`.
    pub fn adt(&self, item: ItemRef, args: &[Ty]) -> Ty {
        let substs = self.intern_substs(args);
        self.intern(Kind::Adt(item, substs))
    }

    /// The function pointer type `(params) -> ret`.
    pub fn fn_ptr(&self, params: &[Ty], ret: Ty) -> Ty {
        let substs = self.intern_substs(params);
        self.intern(Kind::FnPtr(substs, ret))
    }

    /// The unsafe function pointer type `unsafe fn(params) -> ret`.
    pub fn unsafe_fn_ptr(&self, params: &[Ty], ret: Ty) -> Ty {
        let substs = self.intern_substs(params);
        self.intern(Kind::UnsafeFnPtr(substs, ret))
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
}

/// Maps each element of `id` via `f` without cloning pool handles or heap-allocating for up to 8 arguments.
pub(in crate::sema) fn try_map_substs<S, E>(
    state: &mut S,
    id: Substs,
    types_of: fn(&S) -> &Types,
    f: &mut impl FnMut(&mut S, Ty) -> Result<Ty, E>,
) -> Result<Substs, E> {
    if id.is_empty() {
        return Ok(Substs::EMPTY);
    }
    enum SliceCopy {
        One(Ty),
        Small([Ty; 8], usize),
        Large(Vec<Ty>),
    }
    let copied = {
        let slice = types_of(state).substs(id);
        match slice.len() {
            1 => SliceCopy::One(slice[0]),
            len if len <= 8 => {
                let mut buf = [Ty::NONE; 8];
                buf[..len].copy_from_slice(slice);
                SliceCopy::Small(buf, len)
            }
            _ => SliceCopy::Large(slice.to_vec()),
        }
    };
    match copied {
        SliceCopy::One(ty) => {
            let mapped = f(state, ty)?;
            Ok(if mapped == ty {
                id
            } else {
                types_of(state).intern_substs(&[mapped])
            })
        }
        SliceCopy::Small(mut buf, len) => {
            let mut changed = false;
            for slot in &mut buf[..len] {
                let mapped = f(state, *slot)?;
                if mapped != *slot {
                    *slot = mapped;
                    changed = true;
                }
            }
            Ok(if changed {
                types_of(state).intern_substs(&buf[..len])
            } else {
                id
            })
        }
        SliceCopy::Large(mut vec) => {
            let mut changed = false;
            for slot in &mut vec {
                let mapped = f(state, *slot)?;
                if mapped != *slot {
                    *slot = mapped;
                    changed = true;
                }
            }
            Ok(if changed {
                types_of(state).intern_substs(&vec)
            } else {
                id
            })
        }
    }
}

/// Maps direct child types of `ty` via `f` without cloning pool handles, returning `ty` unchanged when no child changed.
pub(in crate::sema) fn try_map_children<S, E>(
    state: &mut S,
    ty: Ty,
    types_of: fn(&S) -> &Types,
    f: &mut impl FnMut(&mut S, Ty) -> Result<Ty, E>,
) -> Result<Ty, E> {
    let next = match *types_of(state).kind(ty) {
        Kind::Tuple(items) => {
            let mapped = try_map_substs(state, items, types_of, f)?;
            if mapped == items {
                return Ok(ty);
            }
            Kind::Tuple(mapped)
        }
        Kind::Ref(mutable, inner) => {
            let mapped = f(state, inner)?;
            if mapped == inner {
                return Ok(ty);
            }
            Kind::Ref(mutable, mapped)
        }
        Kind::Ptr(mutable, inner) => {
            let mapped = f(state, inner)?;
            if mapped == inner {
                return Ok(ty);
            }
            Kind::Ptr(mutable, mapped)
        }
        Kind::Array(inner, len) => {
            let m_inner = f(state, inner)?;
            let m_len = f(state, len)?;
            if m_inner == inner && m_len == len {
                return Ok(ty);
            }
            Kind::Array(m_inner, m_len)
        }
        Kind::Slice(inner) => {
            let mapped = f(state, inner)?;
            if mapped == inner {
                return Ok(ty);
            }
            Kind::Slice(mapped)
        }
        Kind::Adt(item, args) => {
            let mapped = try_map_substs(state, args, types_of, f)?;
            if mapped == args {
                return Ok(ty);
            }
            Kind::Adt(item, mapped)
        }
        Kind::FnDef(item, args) => {
            let mapped = try_map_substs(state, args, types_of, f)?;
            if mapped == args {
                return Ok(ty);
            }
            Kind::FnDef(item, mapped)
        }
        Kind::Closure(parent, args, at) => {
            let mapped = try_map_substs(state, args, types_of, f)?;
            if mapped == args {
                return Ok(ty);
            }
            Kind::Closure(parent, mapped, at)
        }
        Kind::Dyn(of, args, members, autos) => {
            let mapped = try_map_substs(state, args, types_of, f)?;
            if mapped == args {
                return Ok(ty);
            }
            Kind::Dyn(of, mapped, members, autos)
        }
        Kind::FnPtr(params, ret) => {
            let m_params = try_map_substs(state, params, types_of, f)?;
            let m_ret = f(state, ret)?;
            if m_params == params && m_ret == ret {
                return Ok(ty);
            }
            Kind::FnPtr(m_params, m_ret)
        }
        Kind::UnsafeFnPtr(params, ret) => {
            let m_params = try_map_substs(state, params, types_of, f)?;
            let m_ret = f(state, ret)?;
            if m_params == params && m_ret == ret {
                return Ok(ty);
            }
            Kind::UnsafeFnPtr(m_params, m_ret)
        }
        Kind::Pat(base, range) => {
            let mapped = f(state, base)?;
            if mapped == base {
                return Ok(ty);
            }
            Kind::Pat(mapped, range)
        }
        Kind::Proj(id) => {
            let def = types_of(state).proj(id);
            let self_ty = f(state, def.self_ty)?;
            let args = try_map_substs(state, def.args, types_of, f)?;
            let own = try_map_substs(state, def.own, types_of, f)?;
            if self_ty == def.self_ty && args == def.args && own == def.own {
                return Ok(ty);
            }
            Kind::Proj(types_of(state).intern_proj(ProjDef {
                self_ty,
                args,
                own,
                ..def
            }))
        }
        _ => return Ok(ty),
    };
    Ok(types_of(state).intern(next))
}

/// The value with the low `bits` bits set: the mask of an integer that wide.
pub fn mask(bits: u32) -> u128 {
    if bits == 128 {
        u128::MAX
    } else {
        (1 << bits) - 1
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
