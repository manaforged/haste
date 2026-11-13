//! Binary codec for workspace-invariant frozen sysroot `PolySummary` entries.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::{BinOp, ExprId, PatId};
use crate::names::{ItemRef, Primitive};
use crate::outline::Outline;
use crate::sema::infer::{Adjust, Awaited, BodyTables, By, Capture, Closure, Coercion, For, Local, Res, SiteKey, SyntaxTables, Tried, Vtable};
use crate::sema::substs::{PackedMap, PackedSet, ProjDef, Substs};
use crate::sema::ty::{Kind, Mut, Ty, Types};

use super::{ConstHole, OpCheck, PolyObligation, PolySummary};

const CODEC_MAGIC: [u8; 4] = *b"HPS1";

/// Reverse lookup table mapping canonical sysroot item keys back to workspace `ItemRef`s.
#[derive(Debug, Default)]
pub struct SysrootIndex {
    items: HashMap<u64, ItemRef>,
}

impl SysrootIndex {
    /// Builds an index of all sysroot items in `outline` using `key_of`.
    pub fn build(outline: &Outline, mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>) -> Self {
        let mut items = HashMap::default();
        for (krate, tree) in outline.crates.iter().enumerate() {
            if !tree.env.release || !tree.env.fixed {
                continue;
            }
            for (file_idx, file) in tree.files.iter().enumerate() {
                for item_idx in 0..file.items.items.len() {
                    let item = ItemRef { krate: krate as u32, file: file_idx as u32, item: item_idx as u32 };
                    if let Some(key) = key_of(outline, item) {
                        items.insert(key, item);
                    }
                }
            }
        }
        Self { items }
    }

    /// Resolves a canonical sysroot item key to its `ItemRef` in this workspace.
    pub fn get(&self, key: u64) -> Option<ItemRef> {
        self.items.get(&key).copied()
    }
}

/// Encodes `summary` into a workspace-invariant byte vector, or `None` if it references non-sysroot items.
pub fn encode_summary(outline: &Outline, types: &Types, summary: &PolySummary, mut key_of: impl FnMut(&Outline, ItemRef) -> Option<u64>) -> Result<Option<Vec<u8>>> {
    let mut enc = Encoder { outline, types, key_of: &mut key_of, item_map: HashMap::default(), items: Vec::new(), ty_map: HashMap::default(), ty_count: 0, ty_buf: Vec::new() };
    let mut body = Vec::new();
    if enc.write_summary(&mut body, summary).is_none() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(16 + enc.items.len() * 8 + enc.ty_buf.len() + body.len());
    out.extend_from_slice(&CODEC_MAGIC);
    write_u32(&mut out, enc.items.len() as u32);
    enc.items.iter().for_each(|&key| write_u64(&mut out, key));
    write_u32(&mut out, enc.ty_count);
    out.extend_from_slice(&enc.ty_buf);
    out.extend_from_slice(&body);
    Ok(Some(out))
}

/// Decodes a frozen `PolySummary` from `bytes` into `types`, or `None` if any sysroot item is absent.
pub fn decode_summary(index: &SysrootIndex, types: &Types, bytes: &[u8]) -> Result<Option<PolySummary>> {
    let mut cur = Cursor { data: bytes, pos: 0 };
    if cur.take(4)? != CODEC_MAGIC {
        bail!("invalid poly summary magic");
    }
    let item_count = cur.u32()? as usize;
    let mut items = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let Some(item) = index.get(cur.u64()?) else { return Ok(None) };
        items.push(item);
    }
    let ty_count = cur.u32()? as usize;
    let mut tys = Vec::with_capacity(ty_count);
    for _ in 0..ty_count {
        tys.push(read_ty(&mut cur, types, &items, &tys)?);
    }
    let dec = Decoder { types, items: &items, tys: &tys };
    Ok(Some(dec.read_summary(&mut cur)?))
}

struct Encoder<'a, F> {
    outline: &'a Outline,
    types: &'a Types,
    key_of: &'a mut F,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<u64>,
    ty_map: HashMap<Ty, u32>,
    ty_count: u32,
    ty_buf: Vec<u8>,
}

impl<F: FnMut(&Outline, ItemRef) -> Option<u64>> Encoder<'_, F> {
    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        if let Some(&idx) = self.item_map.get(&item) { return Some(idx); }
        let key = (self.key_of)(self.outline, item)?;
        let idx = self.items.len() as u32;
        self.items.push(key);
        self.item_map.insert(item, idx);
        Some(idx)
    }
    fn substs_ids(&mut self, substs: Substs) -> Option<Vec<u32>> { self.types.substs(substs).iter().map(|&ty| self.ty_id(ty)).collect() }

    fn ty_id(&mut self, ty: Ty) -> Option<u32> {
        if ty == Ty::NONE { return Some(u32::MAX); }
        if let Some(&idx) = self.ty_map.get(&ty) { return Some(idx); }
        let mut rec = Vec::new();
        match self.types.kind(ty).clone() {
            Kind::Prim(p) => { rec.push(0); rec.push(p as u8); }
            Kind::Never => rec.push(1),
            Kind::Tuple(s) => { rec.push(2); write_u32s(&mut rec, &self.substs_ids(s)?); }
            Kind::Ref(m, i) => { rec.push(3); rec.push(u8::from(m == Mut::Mut)); write_u32(&mut rec, self.ty_id(i)?); }
            Kind::Ptr(m, i) => { rec.push(4); rec.push(u8::from(m == Mut::Mut)); write_u32(&mut rec, self.ty_id(i)?); }
            Kind::Array(e, l) => { rec.push(5); write_u32(&mut rec, self.ty_id(e)?); write_u32(&mut rec, self.ty_id(l)?); }
            Kind::Slice(e) => { rec.push(6); write_u32(&mut rec, self.ty_id(e)?); }
            Kind::Adt(i, s) => { rec.push(7); write_u32(&mut rec, self.item_id(i)?); write_u32s(&mut rec, &self.substs_ids(s)?); }
            Kind::FnDef(i, s) => { rec.push(8); write_u32(&mut rec, self.item_id(i)?); write_u32s(&mut rec, &self.substs_ids(s)?); }
            Kind::FnPtr(p, r) => { rec.push(9); write_u32s(&mut rec, &self.substs_ids(p)?); write_u32(&mut rec, self.ty_id(r)?); }
            Kind::Closure(i, s, a) => { rec.push(10); write_u32(&mut rec, self.item_id(i)?); write_u32s(&mut rec, &self.substs_ids(s)?); write_u32(&mut rec, a); }
            Kind::Dyn(o, s, m, a) => {
                rec.push(11); write_u32(&mut rec, self.item_id(o)?); write_u32s(&mut rec, &self.substs_ids(s)?);
                let ms = m.into_iter().map(|x| self.item_id(x)).collect::<Option<Vec<_>>>()?;
                let aus = a.into_iter().map(|x| self.item_id(x)).collect::<Option<Vec<_>>>()?;
                write_u32s(&mut rec, &ms); write_u32s(&mut rec, &aus);
            }
            Kind::Pat(b, lo, hi) => { rec.push(12); write_u32(&mut rec, self.ty_id(b)?); write_u128(&mut rec, lo); write_u128(&mut rec, hi); }
            Kind::Value(v) => { rec.push(13); write_u128(&mut rec, v); }
            Kind::Param(p) => { rec.push(14); write_u16(&mut rec, p); }
            Kind::Proj(id) => {
                let d = self.types.proj(id);
                rec.push(15); write_u32(&mut rec, self.item_id(d.of)?); write_u32(&mut rec, self.ty_id(d.self_ty)?);
                write_u32s(&mut rec, &self.substs_ids(d.args)?); write_u32(&mut rec, self.item_id(d.assoc)?); write_u32s(&mut rec, &self.substs_ids(d.own)?);
            }
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return None,
        }
        let idx = self.ty_count;
        self.ty_count += 1;
        self.ty_buf.extend_from_slice(&rec);
        self.ty_map.insert(ty, idx);
        Some(idx)
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
            }
            Some(())
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
            }
            out.push(u8::from(v.consumes)); out.push(u8::from(v.coroutine)); Some(())
        })?;
        write_map_opt(out, &t.try_blocks, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
        write_map_opt(out, &t.box_frees, |out, k, v| { write_u32(out, k.0); self.write_res(out, v) })?;
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
        let call_holes = (0..cur.u32()?).map(|_| Ok((read_site_key(cur)?, self.item(cur.u32()?)?, self.substs(cur)?))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let op_holes = (0..cur.u32()?).map(|_| Ok((ExprId(cur.u32()?), self.read_op_check(cur)?, self.res(cur)?))).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let fixed_drops = (0..cur.u32()?).map(|_| self.res(cur)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let drop_tys = (0..cur.u32()?).map(|_| self.ty(cur.u32()?)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        let parametric_tys = (0..cur.u32()?).map(|_| self.ty(cur.u32()?)).collect::<Result<Vec<_>>>()?.into_boxed_slice();
        Ok(PolySummary { param_count, template, locals, params, ret, obligations, projections, layout_tys, const_holes, fixed_calls, call_holes, op_holes, fixed_drops, drop_tys, parametric_tys })
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
        let coercions = read_map(cur, |c| {
            let k = ExprId(c.u32()?);
            let v = match c.u8()? {
                0 => Coercion::Len(self.ty(c.u32()?)?),
                1 => Coercion::Vtable(self.vtable(c)?),
                2 => Coercion::Reify(self.res(c)?),
                3 => Coercion::Deref((0..c.u32()?).map(|_| Ok((self.ty(c.u32()?)?, read_opt(c, |c2| self.res(c2))?))).collect::<Result<Vec<_>>>()?, self.ty(c.u32()?)?),
                4 => Coercion::Autos,
                5 => Coercion::Upcast(c.u32()?),
                _ => bail!("invalid coercion tag"),
            };
            Ok((k, v))
        })?;
        let offsets = read_map(cur, |c| Ok((ExprId(c.u32()?), c.u64()?)))?;
        let pat_res = read_map(cur, |c| Ok((PatId(c.u32()?), self.res(c)?)))?;
        let overloaded = PackedSet::pack((0..cur.u32()?).map(|_| cur.u32().map(ExprId)).collect::<Result<Vec<_>>>()?);
        let pats = read_map(cur, |c| Ok((PatId(c.u32()?), self.ty(c.u32()?)?)))?;
        let fors = read_map(cur, |c| Ok((ExprId(c.u32()?), For { into_iter: self.res(c)?, next: self.res(c)?, iter: self.ty(c.u32()?)?, option: self.ty(c.u32()?)?, some: c.u32()? })))?;
        let tries = read_map(cur, |c| Ok((ExprId(c.u32()?), Tried { branch: self.res(c)?, from_residual: self.res(c)?, flow: self.ty(c.u32()?)?, proceed: c.u32()?, stop: c.u32()?, target: read_opt(c, |c2| c2.u32().map(ExprId))? })))?;
        let awaits = read_map(cur, |c| Ok((ExprId(c.u32()?), Awaited { into_future: self.res(c)?, future: self.ty(c.u32()?)?, poll: self.res(c)?, polled: self.ty(c.u32()?)? })))?;
        let closures = read_map(cur, |c| {
            let (k, params, ret, body) = (ExprId(c.u32()?), c.u32s()?, self.ty(c.u32()?)?, ExprId(c.u32()?));
            let captures = (0..c.u32()?).map(|_| Ok(Capture { local: c.u32()?, by: match c.u8()? { 0 => By::Ref, 1 => By::Mut, 2 => By::Value, _ => bail!("invalid capture by") } })).collect::<Result<Vec<_>>>()?;
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

fn read_ty(cur: &mut Cursor<'_>, types: &Types, items: &[ItemRef], tys: &[Ty]) -> Result<Ty> {
    let ty_at = |id: u32| if id == u32::MAX { Ok(Ty::NONE) } else { tys.get(id as usize).copied().context("ty ref out of bounds") };
    let item_at = |id: u32| items.get(id as usize).copied().context("item ref out of bounds");
    let substs_of = |c: &mut Cursor<'_>| -> Result<Substs> {
        let ids = c.u32s()?.into_iter().map(ty_at).collect::<Result<Vec<_>>>()?;
        Ok(types.intern_substs(&ids))
    };
    let kind = match cur.u8()? {
        0 => Kind::Prim(decode_prim(cur.u8()?)?),
        1 => Kind::Never,
        2 => Kind::Tuple(substs_of(cur)?),
        3 => Kind::Ref(read_mut(cur)?, ty_at(cur.u32()?)?),
        4 => Kind::Ptr(read_mut(cur)?, ty_at(cur.u32()?)?),
        5 => Kind::Array(ty_at(cur.u32()?)?, ty_at(cur.u32()?)?),
        6 => Kind::Slice(ty_at(cur.u32()?)?),
        7 => Kind::Adt(item_at(cur.u32()?)?, substs_of(cur)?),
        8 => Kind::FnDef(item_at(cur.u32()?)?, substs_of(cur)?),
        9 => Kind::FnPtr(substs_of(cur)?, ty_at(cur.u32()?)?),
        10 => Kind::Closure(item_at(cur.u32()?)?, substs_of(cur)?, cur.u32()?),
        11 => {
            let (of, s) = (item_at(cur.u32()?)?, substs_of(cur)?);
            let members = cur.u32s()?.into_iter().map(item_at).collect::<Result<Vec<_>>>()?;
            let autos = cur.u32s()?.into_iter().map(item_at).collect::<Result<Vec<_>>>()?;
            Kind::Dyn(of, s, members, autos)
        }
        12 => Kind::Pat(ty_at(cur.u32()?)?, cur.u128()?, cur.u128()?),
        13 => Kind::Value(cur.u128()?),
        14 => Kind::Param(cur.u16()?),
        15 => Kind::Proj(types.intern_proj(ProjDef {
            of: item_at(cur.u32()?)?, self_ty: ty_at(cur.u32()?)?, args: substs_of(cur)?, assoc: item_at(cur.u32()?)?, own: substs_of(cur)?,
        })),
        _ => bail!("invalid type node tag"),
    };
    Ok(types.intern(kind))
}

fn decode_prim(raw: u8) -> Result<Primitive> {
    const ALL: [Primitive; 19] = [
        Primitive::Bool, Primitive::Char, Primitive::Str, Primitive::I8, Primitive::I16, Primitive::I32, Primitive::I64, Primitive::I128, Primitive::Isize,
        Primitive::U8, Primitive::U16, Primitive::U32, Primitive::U64, Primitive::U128, Primitive::Usize, Primitive::F16, Primitive::F32, Primitive::F64, Primitive::F128,
    ];
    ALL.get(raw as usize).copied().context("invalid primitive tag")
}

fn read_mut(cur: &mut Cursor<'_>) -> Result<Mut> { Ok(if cur.u8()? != 0 { Mut::Mut } else { Mut::Not }) }

struct Cursor<'a> { data: &'a [u8], pos: usize }

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("cursor overflow")?;
        let slice = self.data.get(self.pos..end).context("unexpected eof")?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into()?)) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn u128(&mut self) -> Result<u128> { Ok(u128::from_le_bytes(self.take(16)?.try_into()?)) }
    fn u32s(&mut self) -> Result<Vec<u32>> { (0..self.u32()?).map(|_| self.u32()).collect() }
}

fn write_u16(out: &mut Vec<u8>, v: u16) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_u32(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_u64(out: &mut Vec<u8>, v: u64) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_u128(out: &mut Vec<u8>, v: u128) { out.extend_from_slice(&v.to_le_bytes()); }
fn write_u32s(out: &mut Vec<u8>, s: &[u32]) { write_u32(out, s.len() as u32); s.iter().for_each(|&v| write_u32(out, v)); }

fn write_opt<T>(out: &mut Vec<u8>, opt: Option<&T>, mut f: impl FnMut(&mut Vec<u8>, &T) -> Option<()>) -> Option<()> {
    match opt { Some(v) => { out.push(1); f(out, v) } None => { out.push(0); Some(()) } }
}
fn read_opt<T>(cur: &mut Cursor<'_>, f: impl FnOnce(&mut Cursor<'_>) -> Result<T>) -> Result<Option<T>> {
    Ok(if cur.u8()? != 0 { Some(f(cur)?) } else { None })
}
fn write_site_key(out: &mut Vec<u8>, s: SiteKey) {
    match s {
        SiteKey::ExprRes(e) => { out.push(0); write_u32(out, e.0); }
        SiteKey::PatRes(p) => { out.push(1); write_u32(out, p.0); }
        SiteKey::Deref(e, i) => { out.push(2); write_u32(out, e.0); write_u32(out, i); }
        SiteKey::Operator(e) => { out.push(3); write_u32(out, e.0); }
        SiteKey::TryBlock(e) => { out.push(4); write_u32(out, e.0); }
        SiteKey::BoxFree(e) => { out.push(5); write_u32(out, e.0); }
        SiteKey::ForInto(e) => { out.push(6); write_u32(out, e.0); }
        SiteKey::ForNext(e) => { out.push(7); write_u32(out, e.0); }
        SiteKey::TryBranch(e) => { out.push(8); write_u32(out, e.0); }
        SiteKey::TryFrom(e) => { out.push(9); write_u32(out, e.0); }
        SiteKey::AwaitInto(e) => { out.push(10); write_u32(out, e.0); }
        SiteKey::AwaitPoll(e) => { out.push(11); write_u32(out, e.0); }
        SiteKey::CoercionReify(e) => { out.push(12); write_u32(out, e.0); }
        SiteKey::CoercionDeref(e, i) => { out.push(13); write_u32(out, e.0); write_u16(out, i); }
    }
}
fn read_site_key(cur: &mut Cursor<'_>) -> Result<SiteKey> {
    Ok(match cur.u8()? {
        0 => SiteKey::ExprRes(ExprId(cur.u32()?)),
        1 => SiteKey::PatRes(PatId(cur.u32()?)),
        2 => SiteKey::Deref(ExprId(cur.u32()?), cur.u32()?),
        3 => SiteKey::Operator(ExprId(cur.u32()?)),
        4 => SiteKey::TryBlock(ExprId(cur.u32()?)),
        5 => SiteKey::BoxFree(ExprId(cur.u32()?)),
        6 => SiteKey::ForInto(ExprId(cur.u32()?)),
        7 => SiteKey::ForNext(ExprId(cur.u32()?)),
        8 => SiteKey::TryBranch(ExprId(cur.u32()?)),
        9 => SiteKey::TryFrom(ExprId(cur.u32()?)),
        10 => SiteKey::AwaitInto(ExprId(cur.u32()?)),
        11 => SiteKey::AwaitPoll(ExprId(cur.u32()?)),
        12 => SiteKey::CoercionReify(ExprId(cur.u32()?)),
        13 => SiteKey::CoercionDeref(ExprId(cur.u32()?), cur.u16()?),
        _ => bail!("invalid site key tag"),
    })
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
