//! Binary codec for workspace-invariant frozen sysroot `PolySummary` entries.

use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::body::{BinOp, ExprId, PatId};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::infer::{Adjust, Awaited, BodyTables, By, Capture, Closure, Coercion, For, Local, Res, SiteKey, SyntaxTables, Tried, Vtable};
use crate::sema::substs::{PackedMap, PackedSet, ProjDef, ProjId, Substs};
use crate::sema::ty::{Kind, Mut, Ty, Types};

use super::{CallOverride, ConstHole, OpCheck, PolyInst, PolyObligation, PolySummary};

mod files;
mod mono;
mod wire;

pub use files::FileIndex;
pub use mono::{decode_typed, decode_verdict, encode_typed, encode_verdict};
pub(crate) use files::ItemRecord;
use files::{read_items, write_items};
pub(crate) use wire::{
    Cursor, TyTable, read_mut, read_site_key, read_str, read_substs, read_ty, write_site_key, write_str, write_u16,
    write_u32, write_u32s, write_u64,
};

const CODEC_MAGIC: [u8; 4] = *b"HPS2";
const INST_MAGIC: [u8; 4] = *b"HFI2";

/// Encodes `summary` into a workspace-invariant byte vector, or `None` if it references non-sysroot items.
pub fn encode_summary(outline: &Outline, types: &Types, summary: &PolySummary, mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>) -> Result<Option<Vec<u8>>> {
    Ok(framed((outline, types, &mut key_of), CODEC_MAGIC, |enc, body| enc.write_summary(body, summary)))
}

/// Decodes a frozen `PolySummary` from `bytes` into `types`, or `None` if any sysroot item is absent.
pub fn decode_summary(outline: &Outline, index: &FileIndex, types: &Types, bytes: &[u8]) -> Result<Option<PolySummary>> {
    unframed((outline, index, types), CODEC_MAGIC, bytes, |dec, cur| dec.read_summary(cur).map(Some))
}

fn framed<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    (outline, types, key_of): (&Outline, &Types, &mut F),
    magic: [u8; 4],
    write: impl FnOnce(&mut Encoder<'_, F>, &mut Vec<u8>) -> Option<()>,
) -> Option<Vec<u8>> {
    let mut enc = Encoder { outline, types, key_of, item_map: HashMap::default(), items: Vec::new(), tys: TyTable::default() };
    let mut body = Vec::new();
    write(&mut enc, &mut body)?;
    let mut out = Vec::with_capacity(16 + enc.items.len() * ItemRecord::LEN + enc.tys.bytes().len() + body.len());
    out.extend_from_slice(&magic);
    write_items(&mut out, &enc.items);
    write_u32(&mut out, enc.tys.count());
    out.extend_from_slice(enc.tys.bytes());
    out.extend_from_slice(&body);
    Some(out)
}

fn unframed<T>(
    (outline, index, types): (&Outline, &FileIndex, &Types),
    magic: [u8; 4],
    bytes: &[u8],
    read: impl FnOnce(&Decoder<'_>, &mut Cursor<'_>) -> Result<Option<T>>,
) -> Result<Option<T>> {
    let mut cur = Cursor::new(bytes);
    if cur.take(4)? != magic {
        bail!("invalid substrate entry magic");
    }
    let Some(items) = read_items(&mut cur, outline, index)? else { return Ok(None) };
    let ty_count = cur.count()?;
    let mut tys = Vec::with_capacity(ty_count);
    for _ in 0..ty_count {
        tys.push(read_ty(&mut cur, types, &items, &tys)?);
    }
    read(&Decoder { types, items: &items, tys: &tys }, &mut cur)
}

/// Decoded concrete sysroot instance resolution data from a substrate.
pub struct FrozenInst {
    pub projs: Box<[(ProjId, Ty)]>,
    pub resolved_calls: Box<[(SiteKey, ItemRef, Substs)]>,
    pub builtin_ops: Box<[ExprId]>,
    pub call_overrides: Box<[CallOverride]>,
    pub offset_overrides: Box<[(ExprId, u64)]>,
    pub drops: Vec<Res>,
}

/// Computes a workspace-invariant 64-bit hash of concrete sysroot `substs`, or `None` if non-sysroot or non-concrete.
pub fn substs_key(outline: &Outline, types: &Types, substs: Substs, mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>) -> Option<u64> {
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

fn hash_substs(outline: &Outline, types: &Types, substs: Substs, key_of: &mut impl FnMut(&Outline, ItemRef) -> Option<u64>, hasher: &mut FxHasher, closures: bool) -> Option<()> {
    let slice = types.substs(substs);
    slice.len().hash(hasher);
    for &ty in slice {
        hash_ty(outline, types, ty, key_of, hasher, closures)?;
    }
    Some(())
}

fn hash_ty(outline: &Outline, types: &Types, ty: Ty, key_of: &mut impl FnMut(&Outline, ItemRef) -> Option<u64>, hasher: &mut FxHasher, closures: bool) -> Option<()> {
    if ty == Ty::NONE {
        0u8.hash(hasher);
        return Some(());
    }
    match types.kind(ty) {
        Kind::Prim(p) => { 1u8.hash(hasher); wire::encode_prim(*p)?.hash(hasher); }
        Kind::Never => { 2u8.hash(hasher); }
        Kind::Tuple(s) => { 3u8.hash(hasher); hash_substs(outline, types, *s, key_of, hasher, closures)?; }
        Kind::Ref(m, i) => { 4u8.hash(hasher); (*m == Mut::Mut).hash(hasher); hash_ty(outline, types, *i, key_of, hasher, closures)?; }
        Kind::Ptr(m, i) => { 5u8.hash(hasher); (*m == Mut::Mut).hash(hasher); hash_ty(outline, types, *i, key_of, hasher, closures)?; }
        Kind::Array(e, l) => { 6u8.hash(hasher); hash_ty(outline, types, *e, key_of, hasher, closures)?; hash_ty(outline, types, *l, key_of, hasher, closures)?; }
        Kind::Slice(e) => { 7u8.hash(hasher); hash_ty(outline, types, *e, key_of, hasher, closures)?; }
        Kind::Adt(i, s) => { 8u8.hash(hasher); key_of(outline, *i)?.hash(hasher); hash_substs(outline, types, *s, key_of, hasher, closures)?; }
        Kind::FnDef(i, s) => { 9u8.hash(hasher); key_of(outline, *i)?.hash(hasher); hash_substs(outline, types, *s, key_of, hasher, closures)?; }
        Kind::FnPtr(p, r) => { 10u8.hash(hasher); hash_substs(outline, types, *p, key_of, hasher, closures)?; hash_ty(outline, types, *r, key_of, hasher, closures)?; }
        Kind::Dyn(o, s, m, a) => {
            12u8.hash(hasher);
            key_of(outline, *o)?.hash(hasher);
            hash_substs(outline, types, *s, key_of, hasher, closures)?;
            let (ms, aus) = (types.items(*m), types.items(*a));
            ms.len().hash(hasher);
            for &x in ms { key_of(outline, x)?.hash(hasher); }
            aus.len().hash(hasher);
            for &x in aus { key_of(outline, x)?.hash(hasher); }
        }
        Kind::Pat(b, range) => { let (lo, hi) = types.pat(*range); 13u8.hash(hasher); lo.hash(hasher); hi.hash(hasher); hash_ty(outline, types, *b, key_of, hasher, closures)?; }
        Kind::Value(v) => { 14u8.hash(hasher); v.hash(hasher); }
        Kind::Closure(parent, s, at) if closures => { 11u8.hash(hasher); key_of(outline, *parent)?.hash(hasher); at.hash(hasher); hash_substs(outline, types, *s, key_of, hasher, closures)?; }
        Kind::Closure(..) | Kind::Param(_) | Kind::Proj(_) | Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return None,
    }
    Some(())
}

/// Encodes a concrete sysroot `PolyInst` and its `drops`, or `None` if it references non-sysroot items.
pub fn encode_instance(
    outline: &Outline,
    types: &Types,
    substs: Substs,
    inst: &PolyInst,
    drops: &[Res],
    mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>,
) -> Result<Option<Vec<u8>>> {
    Ok(framed((outline, types, &mut key_of), INST_MAGIC, |enc, body| enc.write_instance(body, substs, inst, drops)))
}

/// Decodes a frozen concrete sysroot instance from `bytes`, returning `None` if `expected_substs` or any item mismatches.
pub fn decode_instance(outline: &Outline, index: &FileIndex, types: &Types, expected_substs: Substs, bytes: &[u8]) -> Result<Option<FrozenInst>> {
    unframed((outline, index, types), INST_MAGIC, bytes, |dec, cur| dec.read_instance(cur, expected_substs))
}

struct Encoder<'a, F> {
    outline: &'a Outline,
    types: &'a Types,
    key_of: &'a mut F,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<ItemRecord>,
    tys: TyTable,
}

fn intern_item<F: FnMut(&Outline, ItemRef) -> Option<u64>>(
    outline: &Outline,
    key_of: &mut F,
    item_map: &mut HashMap<ItemRef, u32>,
    items: &mut Vec<ItemRecord>,
    item: ItemRef,
) -> Option<u32> {
    if let Some(&idx) = item_map.get(&item) {
        return Some(idx);
    }
    let record = ItemRecord::of(outline, item, key_of)?;
    let idx = items.len() as u32;
    items.push(record);
    item_map.insert(item, idx);
    Some(idx)
}

impl<F: FnMut(&Outline, ItemRef) -> Option<u64>> Encoder<'_, F> {
    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        intern_item(self.outline, self.key_of, &mut self.item_map, &mut self.items, item)
    }

    fn substs_ids(&mut self, substs: Substs) -> Option<Vec<u32>> {
        let (outline, key_of, item_map, items) = (self.outline, &mut *self.key_of, &mut self.item_map, &mut self.items);
        self.tys.substs_ids(self.types, substs, &mut |i| intern_item(outline, key_of, item_map, items, i))
    }

    fn ty_id(&mut self, ty: Ty) -> Option<u32> {
        let (outline, key_of, item_map, items) = (self.outline, &mut *self.key_of, &mut self.item_map, &mut self.items);
        self.tys.ty_id(self.types, ty, &mut |i| intern_item(outline, key_of, item_map, items, i))
    }
    fn write_substs(&mut self, out: &mut Vec<u8>, s: Substs) -> Option<()> { let ids = self.substs_ids(s)?; write_u32s(out, &ids); Some(()) }

    fn write_res(&mut self, out: &mut Vec<u8>, r: &Res) -> Option<()> {
        match *r {
            Res::Local(id) => { out.push(0); write_u32(out, id); }
            Res::Fn(item, s) => { out.push(1); write_u32(out, self.item_id(item)?); self.write_substs(out, s)?; }
            Res::Ctor(item, idx, s) => { out.push(2); write_u32(out, self.item_id(item)?); write_u32(out, idx); self.write_substs(out, s)?; }
            Res::Const(ty) => { out.push(3); write_u32(out, self.ty_id(ty)?); }
        }
        Some(())
    }

    fn write_vtable(&mut self, out: &mut Vec<u8>, v: &Vtable) -> Option<()> {
        write_u32(out, self.ty_id(v.ty)?); write_u32(out, self.ty_id(v.object)?);
        write_opt(out, v.drop.as_ref(), |out, r| self.write_res(out, r))?;
        write_u32(out, v.methods.len() as u32);
        for m in &v.methods { write_opt(out, m.as_ref(), |out, r| self.write_res(out, r))?; }
        write_u32(out, v.supers.len() as u32);
        for s in &v.supers { self.write_vtable(out, s)?; }
        Some(())
    }

    fn write_summary(&mut self, out: &mut Vec<u8>, s: &PolySummary) -> Option<()> {
        write_u16(out, s.param_count);
        write_opt(out, s.ret.as_ref(), |out, t| { write_u32(out, self.ty_id(*t)?); Some(()) })?;
        write_u32(out, s.locals.len() as u32);
        for l in &s.locals { write_u32(out, self.ty_id(l.ty)?); }
        write_u32s(out, &s.params);
        write_syntax(out, &s.template.syntax);
        self.write_tables(out, &s.template)?;
        write_u32(out, s.obligations.len() as u32);
        for ob in &s.obligations {
            match ob {
                PolyObligation::Prove { of, self_ty, args } => {
                    out.push(0); write_u32(out, self.item_id(*of)?); write_u32(out, self.ty_id(*self_ty)?); self.write_substs(out, *args)?;
                }
                PolyObligation::Call { expr, found, tuple } => {
                    out.push(1); write_u32(out, expr.0); write_u32(out, self.ty_id(*found)?); write_u32(out, self.ty_id(*tuple)?);
                }
            }
        }
        write_u32(out, s.projections.len() as u32);
        for &p in &s.projections {
            let d = self.types.proj(p);
            write_u32(out, self.item_id(d.of)?); write_u32(out, self.ty_id(d.self_ty)?); self.write_substs(out, d.args)?;
            write_u32(out, self.item_id(d.assoc)?); self.write_substs(out, d.own)?;
        }
        write_u32(out, s.layout_tys.len() as u32);
        for &t in &s.layout_tys { write_u32(out, self.ty_id(t)?); }
        write_u32(out, s.const_holes.len() as u32);
        for ch in &s.const_holes {
            match ch {
                ConstHole::Value(ty) => { out.push(0); write_u32(out, self.ty_id(*ty)?); }
                ConstHole::OffsetOf { expr, steps } => {
                    out.push(1); write_u32(out, expr.0); write_u32(out, steps.len() as u32);
                    for &(ty, f) in steps { write_u32(out, self.ty_id(ty)?); write_u32(out, f); }
                }
                ConstHole::Intrinsic { item, substs } => { out.push(2); write_u32(out, self.item_id(*item)?); self.write_substs(out, *substs)?; }
            }
        }
        write_u32(out, s.fixed_calls.len() as u32);
        for &(item, substs) in &s.fixed_calls { write_u32(out, self.item_id(item)?); self.write_substs(out, substs)?; }
        write_u32(out, s.direct_calls.len() as u32);
        for &(site, item, substs) in &s.direct_calls { write_site_key(out, site); write_u32(out, self.item_id(item)?); self.write_substs(out, substs)?; }
        write_u32(out, s.call_holes.len() as u32);
        for &(site, item, substs) in &s.call_holes { write_site_key(out, site); write_u32(out, self.item_id(item)?); self.write_substs(out, substs)?; }
        write_u32(out, s.op_holes.len() as u32);
        for &(expr, op, res) in &s.op_holes { write_u32(out, expr.0); self.write_op_check(out, op)?; self.write_res(out, &res)?; }
        write_u32(out, s.fixed_drops.len() as u32);
        for r in &s.fixed_drops { self.write_res(out, r)?; }
        write_u32(out, s.drop_tys.len() as u32);
        for &t in &s.drop_tys { write_u32(out, self.ty_id(t)?); }
        write_u32(out, s.parametric_tys.len() as u32);
        for &t in &s.parametric_tys { write_u32(out, self.ty_id(t)?); }
        Some(())
    }

    fn write_instance(&mut self, out: &mut Vec<u8>, substs: Substs, inst: &PolyInst, drops: &[Res]) -> Option<()> {
        self.write_substs(out, substs)?;
        write_u32(out, inst.projs.len() as u32);
        for &(pid, ty) in &inst.projs {
            let d = self.types.proj(pid);
            write_u32(out, self.item_id(d.of)?);
            write_u32(out, self.ty_id(d.self_ty)?);
            self.write_substs(out, d.args)?;
            write_u32(out, self.item_id(d.assoc)?);
            self.write_substs(out, d.own)?;
            write_u32(out, self.ty_id(ty)?);
        }
        write_u32(out, inst.resolved_calls.len() as u32);
        for &(site, item, s) in &inst.resolved_calls {
            write_site_key(out, site);
            write_u32(out, self.item_id(item)?);
            self.write_substs(out, s)?;
        }
        write_u32(out, inst.builtin_ops.len() as u32);
        for e in &inst.builtin_ops { write_u32(out, e.0); }
        write_u32(out, inst.call_overrides.len() as u32);
        for ov in &inst.call_overrides {
            write_u32(out, ov.expr.0);
            write_opt(out, ov.res.as_ref(), |o, r| self.write_res(o, r))?;
            write_opt(out, ov.autoderefs.as_ref(), |o, a| { write_u32(o, *a); Some(()) })?;
            out.push(u8::from(ov.overloaded));
            write_opt(out, ov.box_free.as_ref(), |o, r| self.write_res(o, r))?;
        }
        write_u32(out, inst.offset_overrides.len() as u32);
        for &(e, v) in &inst.offset_overrides {
            write_u32(out, e.0);
            write_u64(out, v);
        }
        write_u32(out, drops.len() as u32);
        for r in drops { self.write_res(out, r)?; }
        Some(())
    }

    fn write_op_check(&mut self, out: &mut Vec<u8>, op: OpCheck) -> Option<()> {
        match op {
            OpCheck::Unary(t) => { out.push(0); write_u32(out, self.ty_id(t)?); }
            OpCheck::Binary(b, l, r) => { out.push(1); out.push(b as u8); write_u32(out, self.ty_id(l)?); write_u32(out, self.ty_id(r)?); }
            OpCheck::AssignOp(b, l, r) => { out.push(2); out.push(b as u8); write_u32(out, self.ty_id(l)?); write_u32(out, self.ty_id(r)?); }
            OpCheck::Index(b, i) => { out.push(3); write_opt(out, b.as_ref(), |o, t| { write_u32(o, self.ty_id(*t)?); Some(()) })?; write_u32(out, self.ty_id(i)?); }
            OpCheck::Never => out.push(4),
        }
        Some(())
    }

    fn write_tables(&mut self, out: &mut Vec<u8>, t: &BodyTables) -> Option<()> {
        write_u32(out, t.exprs.len() as u32);
        for &ty in &t.exprs { write_u32(out, self.ty_id(ty)?); }
        write_map_opt(out, &t.res, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
        write_map_opt(out, &t.autoderefs, |out, k, v| { write_u32(out, k.0); write_u32(out, *v); Some(()) })?;
        write_map_opt(out, &t.derefs, |out, k, v| { write_u32(out, k.0.0); write_u32(out, k.1); self.write_res(out, v) })?;
        write_map_opt(out, &t.operators, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
        write_map_opt(out, &t.coercions, |out, k, v| {
            write_u32(out, k.0);
            self.write_coercion(out, v)
        })?;
        write_map_opt(out, &t.offsets, |out, k, v| { write_u32(out, k.0); write_u64(out, *v); Some(()) })?;
        write_map_opt(out, &t.pat_res, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
        write_u32(out, t.overloaded.as_slice().len() as u32);
        for k in t.overloaded.as_slice() { write_u32(out, k.0); }
        write_map_opt(out, &t.pats, |out, k, v| { write_u32(out, k.0); write_u32(out, self.ty_id(*v)?); Some(()) })?;
        write_map_opt(out, &t.fors, |out, k, v| {
            write_u32(out, k.0); self.write_res(out, &v.into_iter)?; self.write_res(out, &v.next)?;
            write_u32(out, self.ty_id(v.iter)?); write_u32(out, self.ty_id(v.option)?); write_u32(out, v.some); Some(())
        })?;
        write_map_opt(out, &t.tries, |out, k, v| {
            write_u32(out, k.0); self.write_res(out, &v.branch)?; self.write_res(out, &v.from_residual)?;
            write_u32(out, self.ty_id(v.flow)?); write_u32(out, v.proceed); write_u32(out, v.stop);
            write_opt(out, v.target.as_ref(), |out, e| { write_u32(out, e.0); Some(()) })
        })?;
        write_map_opt(out, &t.awaits, |out, k, v| {
            write_u32(out, k.0); self.write_res(out, &v.into_future)?; write_u32(out, self.ty_id(v.future)?);
            self.write_res(out, &v.poll)?; write_u32(out, self.ty_id(v.polled)?); Some(())
        })?;
        write_map_opt(out, &t.closures, |out, k, v| {
            write_u32(out, k.0); write_u32s(out, &v.params); write_u32(out, self.ty_id(v.ret)?);
            write_u32(out, v.body.0); write_u32(out, v.captures.len() as u32);
            for cap in &v.captures {
                write_u32(out, cap.local);
                out.push(match cap.by { By::Ref => 0, By::Mut => 1, By::Value => 2 });
                write_u32(out, cap.path.len() as u32);
                for &(variant, index) in &cap.path {
                    write_u32(out, variant);
                    write_u32(out, index);
                }
            }
            out.push(u8::from(v.consumes)); out.push(u8::from(v.coroutine)); Some(())
        })?;
        write_map_opt(out, &t.try_blocks, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
        write_map_opt(out, &t.box_frees, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
        Some(())
    }

    fn write_coercion(&mut self, out: &mut Vec<u8>, v: &Coercion) -> Option<()> {
        match v {
            Coercion::Len(ty) => { out.push(0); write_u32(out, self.ty_id(*ty)?); }
            Coercion::Vtable(vt) => { out.push(1); self.write_vtable(out, vt)?; }
            Coercion::Reify(r) => { out.push(2); self.write_res(out, r)?; }
            Coercion::Deref(steps, ty) => {
                out.push(3); write_u32(out, steps.len() as u32);
                for (st, sr) in steps { write_u32(out, self.ty_id(*st)?); write_opt(out, sr.as_ref(), |out, r| self.write_res(out, r))?; }
                write_u32(out, self.ty_id(*ty)?);
            }
            Coercion::Autos => out.push(4),
            Coercion::Upcast(idx) => { out.push(5); write_u32(out, *idx); }
            Coercion::Field { target, inner } => {
                out.push(6);
                write_u32(out, self.ty_id(*target)?);
                self.write_coercion(out, inner)?;
            }
        }
        Some(())
    }
}

fn write_syntax(out: &mut Vec<u8>, syn: &SyntaxTables) {
    write_map(out, &syn.binds, |o, k, v| { write_u32(o, k.0); write_u32(o, *v); });
    write_map(out, &syn.adjusts, |o, k, v| {
        write_u32(o, k.0); write_u32(o, v.derefs);
        match v.autoref { Some(m) => { o.push(1); o.push(u8::from(m == Mut::Mut)); } None => o.push(0) }
    });
    write_map(out, &syn.fields, |o, k, v| { write_u32(o, k.0); write_u32(o, *v); });
    write_map(out, &syn.inits, |o, k, v| { write_u32(o, k.0); write_u32(o, *v); });
    write_map(out, &syn.ranges, |o, k, v| { write_u32(o, k.0); write_u32s(o, v); });
    write_map(out, &syn.field_pats, |o, k, v| { write_u32(o, k.0); write_u32(o, *v); });
    write_map(out, &syn.pat_derefs, |o, k, v| { write_u32(o, k.0); write_u32(o, *v); });
    write_map(out, &syn.ref_binds, |o, k, v| { write_u32(o, k.0); o.push(u8::from(*v == Mut::Mut)); });
    write_u32(out, syn.or_binds.as_slice().len() as u32);
    syn.or_binds.as_slice().iter().for_each(|k| write_u32(out, k.0));
    write_u32(out, syn.const_args.as_slice().len() as u32);
    syn.const_args.as_slice().iter().for_each(|k| write_u32(out, k.0));
    write_map(out, &syn.unpacks, |o, k, v| { write_u32(o, *k); write_u32(o, v.0); });
}

struct Decoder<'a> {
    types: &'a Types,
    items: &'a [ItemRef],
    tys: &'a [Ty],
}

impl Decoder<'_> {
    fn read_coercion(&self, c: &mut Cursor<'_>) -> Result<Coercion> {
        Ok(match c.u8()? {
            0 => Coercion::Len(self.ty(c.u32()?)?),
            1 => Coercion::Vtable(self.vtable(c)?),
            2 => Coercion::Reify(self.res(c)?),
            3 => Coercion::Deref((0..c.u32()?).map(|_| Ok((self.ty(c.u32()?)?, read_opt(c, |c2| self.res(c2))?))).collect::<Result<Vec<_>>>()?, self.ty(c.u32()?)?),
            4 => Coercion::Autos,
            5 => Coercion::Upcast(c.u32()?),
            6 => Coercion::Field { target: self.ty(c.u32()?)?, inner: Box::new(self.read_coercion(c)?) },
            _ => bail!("invalid coercion tag"),
        })
    }
    fn ty(&self, idx: u32) -> Result<Ty> {
        if idx == u32::MAX { return Ok(Ty::NONE); }
        self.tys.get(idx as usize).copied().context("type index out of bounds")
    }
    fn item(&self, idx: u32) -> Result<ItemRef> { self.items.get(idx as usize).copied().context("item index out of bounds") }
    fn substs(&self, cur: &mut Cursor<'_>) -> Result<Substs> {
        let ids = cur.u32s()?.into_iter().map(|i| self.ty(i)).collect::<Result<Vec<_>>>()?;
        Ok(self.types.intern_substs(&ids))
    }
    fn res(&self, cur: &mut Cursor<'_>) -> Result<Res> {
        Ok(match cur.u8()? {
            0 => Res::Local(cur.u32()?),
            1 => Res::Fn(self.item(cur.u32()?)?, self.substs(cur)?),
            2 => Res::Ctor(self.item(cur.u32()?)?, cur.u32()?, self.substs(cur)?),
            3 => Res::Const(self.ty(cur.u32()?)?),
            _ => bail!("invalid res tag"),
        })
    }
    fn vtable(&self, cur: &mut Cursor<'_>) -> Result<Vtable> {
        let (ty, object) = (self.ty(cur.u32()?)?, self.ty(cur.u32()?)?);
        let drop = read_opt(cur, |c| self.res(c))?;
        let methods = (0..cur.u32()?).map(|_| read_opt(cur, |c| self.res(c))).collect::<Result<Vec<_>>>()?;
        let supers = (0..cur.u32()?).map(|_| self.vtable(cur)).collect::<Result<Vec<_>>>()?;
        Ok(Vtable { ty, object, drop, methods, supers })
    }
    fn read_summary(&self, cur: &mut Cursor<'_>) -> Result<PolySummary> {
        let param_count = cur.u16()?;
        let ret = read_opt(cur, |c| self.ty(c.u32()?))?;
        let locals = (0..cur.u32()?).map(|_| Ok(Local { ty: self.ty(cur.u32()?)? })).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let params = cur.u32s()?.into_boxed_slice();
        let syntax = Arc::new(read_syntax(cur)?);
        let template = Arc::new(self.read_tables(cur, syntax)?);
        let obligations = (0..cur.u32()?).map(|_| match cur.u8()? {
            0 => Ok(PolyObligation::Prove { of: self.item(cur.u32()?)?, self_ty: self.ty(cur.u32()?)?, args: self.substs(cur)? }),
            1 => Ok(PolyObligation::Call { expr: ExprId(cur.u32()?), found: self.ty(cur.u32()?)?, tuple: self.ty(cur.u32()?)? }),
            _ => bail!("invalid obligation tag"),
        }).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let projections = (0..cur.u32()?).map(|_| Ok(self.types.intern_proj(ProjDef {
            of: self.item(cur.u32()?)?, self_ty: self.ty(cur.u32()?)?, args: self.substs(cur)?, assoc: self.item(cur.u32()?)?, own: self.substs(cur)?,
        }))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let layout_tys = (0..cur.u32()?).map(|_| self.ty(cur.u32()?)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let const_holes = (0..cur.u32()?).map(|_| match cur.u8()? {
            0 => Ok(ConstHole::Value(self.ty(cur.u32()?)?)),
            1 => Ok(ConstHole::OffsetOf { expr: ExprId(cur.u32()?), steps: (0..cur.u32()?).map(|_| Ok((self.ty(cur.u32()?)?, cur.u32()?))).collect::<Result<Vec<_>>>()?.into_boxed_slice() }),
            2 => Ok(ConstHole::Intrinsic { item: self.item(cur.u32()?)?, substs: self.substs(cur)? }),
            _ => bail!("invalid const hole tag"),
        }).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let fixed_calls = (0..cur.u32()?).map(|_| Ok((self.item(cur.u32()?)?, self.substs(cur)?))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let direct_calls = (0..cur.u32()?).map(|_| Ok((read_site_key(cur)?, self.item(cur.u32()?)?, self.substs(cur)?))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let call_holes = (0..cur.u32()?).map(|_| Ok((read_site_key(cur)?, self.item(cur.u32()?)?, self.substs(cur)?))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let op_holes = (0..cur.u32()?).map(|_| Ok((ExprId(cur.u32()?), self.read_op_check(cur)?, self.res(cur)?))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let fixed_drops = (0..cur.u32()?).map(|_| self.res(cur)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let drop_tys = (0..cur.u32()?).map(|_| self.ty(cur.u32()?)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let parametric_tys = (0..cur.u32()?).map(|_| self.ty(cur.u32()?)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        Ok(PolySummary { param_count, template, locals, params, ret, obligations, projections, layout_tys, const_holes, fixed_calls, direct_calls, call_holes, op_holes, fixed_drops, drop_tys, parametric_tys })
    }

    fn read_instance(&self, cur: &mut Cursor<'_>, expected_substs: Substs) -> Result<Option<FrozenInst>> {
        let substs = self.substs(cur)?;
        if substs != expected_substs {
            return Ok(None);
        }
        let mut projs = (0..cur.u32()?)
            .map(|_| {
                let pid = self.types.intern_proj(ProjDef {
                    of: self.item(cur.u32()?)?,
                    self_ty: self.ty(cur.u32()?)?,
                    args: self.substs(cur)?,
                    assoc: self.item(cur.u32()?)?,
                    own: self.substs(cur)?,
                });
                let ty = self.ty(cur.u32()?)?;
                Ok((pid, ty))
            })
            .collect::<Result<Vec<_>>>()?;
        projs.sort_unstable_by_key(|&(pid, _)| pid.0);
        let projs = projs.into_boxed_slice();
        let resolved_calls = (0..cur.u32()?)
            .map(|_| Ok((read_site_key(cur)?, self.item(cur.u32()?)?, self.substs(cur)?)))
            .collect::<Result<Vec<_>>>()?
            .into_boxed_slice();
        let builtin_ops = (0..cur.u32()?).map(|_| cur.u32().map(ExprId)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let call_overrides = (0..cur.u32()?)
            .map(|_| {
                Ok(CallOverride {
                    expr: ExprId(cur.u32()?),
                    res: read_opt(cur, |c| self.res(c))?,
                    autoderefs: read_opt(cur, |c| c.u32())?,
                    overloaded: cur.u8()? != 0,
                    box_free: read_opt(cur, |c| self.res(c))?,
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_boxed_slice();
        let offset_overrides = (0..cur.u32()?)
            .map(|_| Ok((ExprId(cur.u32()?), cur.u64()?)))
            .collect::<Result<Vec<_>>>()?
            .into_boxed_slice();
        let drops = (0..cur.u32()?).map(|_| self.res(cur)).collect::<Result<Vec<_>>>()?;
        Ok(Some(FrozenInst { projs, resolved_calls, builtin_ops, call_overrides, offset_overrides, drops }))
    }

    fn read_op_check(&self, cur: &mut Cursor<'_>) -> Result<OpCheck> {
        Ok(match cur.u8()? {
            0 => OpCheck::Unary(self.ty(cur.u32()?)?),
            1 => OpCheck::Binary(decode_bin_op(cur.u8()?)?, self.ty(cur.u32()?)?, self.ty(cur.u32()?)?),
            2 => OpCheck::AssignOp(decode_bin_op(cur.u8()?)?, self.ty(cur.u32()?)?, self.ty(cur.u32()?)?),
            3 => OpCheck::Index(read_opt(cur, |c| self.ty(c.u32()?))?, self.ty(cur.u32()?)?),
            4 => OpCheck::Never,
            _ => bail!("invalid op check tag"),
        })
    }

    fn read_tables(&self, cur: &mut Cursor<'_>, syntax: Arc<SyntaxTables>) -> Result<BodyTables> {
        let exprs = (0..cur.u32()?).map(|_| self.ty(cur.u32()?)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let res = read_map(cur, |c| Ok((ExprId(c.u32()?), self.res(c)?)))?;
        let autoderefs = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32()?)))?;
        let derefs = read_map(cur, |c| Ok(((ExprId(c.u32()?), c.u32()?), self.res(c)?)))?;
        let operators = read_map(cur, |c| Ok((ExprId(c.u32()?), self.res(c)?)))?;
        let coercions = read_map(cur, |c| Ok((ExprId(c.u32()?), self.read_coercion(c)?)))?;
        let offsets = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u64()?)))?;
        let pat_res = read_map(cur, |c| Ok((PatId(c.u32()?), self.res(c)?)))?;
        let overloaded = PackedSet::pack((0..cur.u32()?).map(|_| cur.u32().map(ExprId)).collect::<Result<Vec<_>>>()?);
        let pats = read_map(cur, |c| Ok((PatId(c.u32()?), self.ty(c.u32()?)?)))?;
        let fors = read_map(cur, |c| Ok((ExprId(c.u32()?), For { into_iter: self.res(c)?, next: self.res(c)?, iter: self.ty(c.u32()?)?, option: self.ty(c.u32()?)?, some: c.u32()? })))?;
        let tries = read_map(cur, |c| Ok((ExprId(c.u32()?), Tried { branch: self.res(c)?, from_residual: self.res(c)?, flow: self.ty(c.u32()?)?, proceed: c.u32()?, stop: c.u32()?, target: read_opt(c, |c2| c2.u32().map(ExprId))? })))?;
        let awaits = read_map(cur, |c| Ok((ExprId(c.u32()?), Awaited { into_future: self.res(c)?, future: self.ty(c.u32()?)?, poll: self.res(c)?, polled: self.ty(c.u32()?)? })))?;
        let closures = read_map(cur, |c| {
            let (k, params, ret, body) = (ExprId(c.u32()?), c.u32s()?, self.ty(c.u32()?)?, ExprId(c.u32()?));
            let captures = (0..c.u32()?)
                .map(|_| {
                    let local = c.u32()?;
                    let by = match c.u8()? { 0 => By::Ref, 1 => By::Mut, 2 => By::Value, _ => bail!("invalid capture by") };
                    let path = (0..c.u32()?).map(|_| Ok((c.u32()?, c.u32()?))).collect::<Result<Vec<_>>>()?;
                    Ok(Capture { local, by, path })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((k, Closure { params, ret, body, captures, consumes: c.u8()? != 0, coroutine: c.u8()? != 0 }))
        })?;
        let try_blocks = read_map(cur, |c| Ok((ExprId(c.u32()?), self.res(c)?)))?;
        let box_frees = read_map(cur, |c| Ok((ExprId(c.u32()?), self.res(c)?)))?;
        Ok(BodyTables { syntax, exprs, res, autoderefs, derefs, operators, coercions, offsets, pat_res, overloaded, pats, fors, tries, awaits, closures, try_blocks, box_frees })
    }
}

fn read_syntax(cur: &mut Cursor<'_>) -> Result<SyntaxTables> {
    let binds = read_map(cur, |c| Ok((PatId(c.u32()?), c.u32()?)))?;
    let adjusts = read_map(cur, |c| Ok((ExprId(c.u32()?), Adjust { derefs: c.u32()?, autoref: read_opt(c, read_mut)? })))?;
    let fields = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32()?)))?;
    let inits = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32()?)))?;
    let ranges = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u32s()?.into_boxed_slice())))?;
    let field_pats = read_map(cur, |c| Ok((PatId(c.u32()?), c.u32()?)))?;
    let pat_derefs = read_map(cur, |c| Ok((PatId(c.u32()?), c.u32()?)))?;
    let ref_binds = read_map(cur, |c| Ok((PatId(c.u32()?), read_mut(c)?)))?;
    let or_binds = PackedSet::pack((0..cur.u32()?).map(|_| cur.u32().map(PatId)).collect::<Result<Vec<_>>>()?);
    let const_args = PackedSet::pack((0..cur.u32()?).map(|_| cur.u32().map(ExprId)).collect::<Result<Vec<_>>>()?);
    let unpacks = read_map(cur, |c| Ok((c.u32()?, PatId(c.u32()?))))?;
    Ok(SyntaxTables { binds, adjusts, fields, inits, ranges, field_pats, pat_derefs, ref_binds, or_binds, const_args, unpacks })
}

fn write_opt<T>(out: &mut Vec<u8>, opt: Option<&T>, mut f: impl FnMut(&mut Vec<u8>, &T) -> Option<()>) -> Option<()> {
    match opt { Some(v) => { out.push(1); f(out, v) } None => { out.push(0); Some(()) } }
}
fn read_opt<T>(cur: &mut Cursor<'_>, f: impl FnOnce(&mut Cursor<'_>) -> Result<T>) -> Result<Option<T>> {
    Ok(if cur.u8()? != 0 { Some(f(cur)?) } else { None })
}
fn decode_bin_op(raw: u8) -> Result<BinOp> {
    const OPS: [BinOp; 18] = [
        BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Rem, BinOp::And, BinOp::Or, BinOp::BitXor, BinOp::BitAnd,
        BinOp::BitOr, BinOp::Shl, BinOp::Shr, BinOp::Eq, BinOp::Lt, BinOp::Le, BinOp::Ne, BinOp::Ge, BinOp::Gt,
    ];
    OPS.get(raw as usize).copied().context("invalid bin op tag")
}
fn write_map<K, V>(out: &mut Vec<u8>, map: &PackedMap<K, V>, mut f: impl FnMut(&mut Vec<u8>, &K, &V)) {
    write_u32(out, map.len() as u32);
    for (k, v) in map { f(out, k, v); }
}
fn write_map_opt<K, V>(out: &mut Vec<u8>, map: &PackedMap<K, V>, mut f: impl FnMut(&mut Vec<u8>, &K, &V) -> Option<()>) -> Option<()> {
    write_u32(out, map.len() as u32);
    for (k, v) in map { f(out, k, v)?; }
    Some(())
}
fn read_map<K: Ord, V>(cur: &mut Cursor<'_>, mut f: impl FnMut(&mut Cursor<'_>) -> Result<(K, V)>) -> Result<PackedMap<K, V>> {
    let len = cur.u32()? as usize;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len { vec.push(f(cur)?); }
    Ok(PackedMap::from_sorted(vec))
}
