use std::hash::{Hash as _, Hasher as _};

use rustc_hash::FxHasher;

use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Mut, Ty, Types};

use super::wire;

/// Computes a workspace-invariant 64-bit hash of concrete sysroot `substs`, or `None` if non-sysroot or non-concrete.
pub fn substs_key(
    outline: &Outline,
    types: &Types,
    substs: Substs,
    mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>,
) -> Option<u64> {
    let mut hasher = FxHasher::default();
    hash_substs(outline, types, substs, &mut key_of, &mut hasher, false)?;
    Some(hasher.finish())
}

/// The `TypeId` of concrete `ty`: a hash of its structure and item identities, independent of interning order.
pub fn type_id_key(outline: &Outline, types: &Types, ty: Ty) -> u64 {
    let mut hasher = FxHasher::default();
    let mut key_of = |_: &Outline, item: ItemRef| {
        let mut h = FxHasher::default();
        item.hash(&mut h);
        Some(h.finish())
    };
    match hash_ty(outline, types, ty, &mut key_of, &mut hasher, true) {
        Some(()) => hasher.finish(),
        None => u64::from(ty.0),
    }
}

fn hash_substs(
    outline: &Outline,
    types: &Types,
    substs: Substs,
    key_of: &mut impl FnMut(&Outline, ItemRef) -> Option<u64>,
    hasher: &mut FxHasher,
    closures: bool,
) -> Option<()> {
    let slice = types.substs(substs);
    slice.len().hash(hasher);
    for &ty in slice {
        hash_ty(outline, types, ty, key_of, hasher, closures)?;
    }
    Some(())
}

fn hash_items(
    outline: &Outline,
    items: &[ItemRef],
    key_of: &mut impl FnMut(&Outline, ItemRef) -> Option<u64>,
    hasher: &mut FxHasher,
) -> Option<()> {
    items.len().hash(hasher);
    for &x in items {
        key_of(outline, x)?.hash(hasher);
    }
    Some(())
}

fn hash_ty(
    outline: &Outline,
    types: &Types,
    ty: Ty,
    key_of: &mut impl FnMut(&Outline, ItemRef) -> Option<u64>,
    hasher: &mut FxHasher,
    closures: bool,
) -> Option<()> {
    if ty == Ty::NONE {
        0u8.hash(hasher);
        return Some(());
    }
    match types.kind(ty) {
        Kind::Prim(p) => {
            1u8.hash(hasher);
            wire::encode_prim(*p)?.hash(hasher);
        }
        Kind::Never => {
            2u8.hash(hasher);
        }
        Kind::Tuple(s) => {
            3u8.hash(hasher);
            hash_substs(outline, types, *s, key_of, hasher, closures)?;
        }
        Kind::Ref(m, i) => {
            4u8.hash(hasher);
            (*m == Mut::Mut).hash(hasher);
            hash_ty(outline, types, *i, key_of, hasher, closures)?;
        }
        Kind::Ptr(m, i) => {
            5u8.hash(hasher);
            (*m == Mut::Mut).hash(hasher);
            hash_ty(outline, types, *i, key_of, hasher, closures)?;
        }
        Kind::Array(e, l) => {
            6u8.hash(hasher);
            hash_ty(outline, types, *e, key_of, hasher, closures)?;
            hash_ty(outline, types, *l, key_of, hasher, closures)?;
        }
        Kind::Slice(e) => {
            7u8.hash(hasher);
            hash_ty(outline, types, *e, key_of, hasher, closures)?;
        }
        Kind::Adt(i, s) => {
            8u8.hash(hasher);
            key_of(outline, *i)?.hash(hasher);
            hash_substs(outline, types, *s, key_of, hasher, closures)?;
        }
        Kind::FnDef(i, s) => {
            9u8.hash(hasher);
            key_of(outline, *i)?.hash(hasher);
            hash_substs(outline, types, *s, key_of, hasher, closures)?;
        }
        Kind::FnPtr(p, r) => {
            10u8.hash(hasher);
            hash_substs(outline, types, *p, key_of, hasher, closures)?;
            hash_ty(outline, types, *r, key_of, hasher, closures)?;
        }
        Kind::UnsafeFnPtr(p, r) => {
            16u8.hash(hasher);
            hash_substs(outline, types, *p, key_of, hasher, closures)?;
            hash_ty(outline, types, *r, key_of, hasher, closures)?;
        }
        Kind::Dyn(o, s, m, a) => {
            12u8.hash(hasher);
            key_of(outline, *o)?.hash(hasher);
            hash_substs(outline, types, *s, key_of, hasher, closures)?;
            hash_items(outline, types.items(*m), key_of, hasher)?;
            hash_items(outline, types.items(*a), key_of, hasher)?;
        }
        Kind::Pat(b, range) => {
            let (lo, hi) = types.pat(*range);
            13u8.hash(hasher);
            lo.hash(hasher);
            hi.hash(hasher);
            hash_ty(outline, types, *b, key_of, hasher, closures)?;
        }
        Kind::Value(v) => {
            14u8.hash(hasher);
            v.hash(hasher);
        }
        Kind::Closure(parent, s, at) if closures => {
            11u8.hash(hasher);
            key_of(outline, *parent)?.hash(hasher);
            at.hash(hasher);
            hash_substs(outline, types, *s, key_of, hasher, closures)?;
        }
        Kind::Closure(..)
        | Kind::Param(_)
        | Kind::Proj(_)
        | Kind::Var(_)
        | Kind::IntVar(_)
        | Kind::FloatVar(_) => return None,
    }
    Some(())
}
