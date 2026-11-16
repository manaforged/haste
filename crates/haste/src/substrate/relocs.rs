//! Workspace-invariant serialization of `RelocPlan` for cross-workspace sysroot stencil reuse.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap;

use crate::body::{ExprId, PatId};
use crate::host::{HostSymbol, MathSymbol, X86Symbol};
use crate::lower::reloc::{RelocPlan, RelocStep, SiteOrigin, TyOrigin};
use crate::lower::{RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId, Word};
use crate::names::{ItemRef, Primitive};
use crate::outline::Outline;
use crate::sema::infer::SiteKey;
use crate::sema::reach::Reached;
use crate::sema::substs::{ProjDef, Substs};
use crate::sema::summary::codec::{Cursor, FileIndex, ItemRecord, read_mut, write_u16, write_u32, write_u32s, write_u64, write_u128};
use crate::sema::ty::{Kind, Mut, Ty, Types};

const PLAN_MAGIC: [u8; 4] = *b"HRP1";

const PRIMS: [Primitive; 19] = [
    Primitive::Bool, Primitive::Char, Primitive::Str, Primitive::I8, Primitive::I16, Primitive::I32, Primitive::I64, Primitive::I128, Primitive::Isize,
    Primitive::U8, Primitive::U16, Primitive::U32, Primitive::U64, Primitive::U128, Primitive::Usize, Primitive::F16, Primitive::F32, Primitive::F64, Primitive::F128,
];

/// Encodes `plan` into workspace-invariant bytes, or `None` if any symbol or type leaves the allowed crate scope.
pub fn encode_reloc_plan<F, FR, FV>(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    plan: &RelocPlan,
    include_deps: bool,
    mut key_of: F,
    mut rodata_of: FR,
    mut vtable_of: FV,
) -> Option<Vec<u8>>
where
    F: FnMut(&Outline, ItemRef) -> Option<u64>,
    FR: FnMut(RodataId) -> Option<(Vec<u8>, Vec<RodataReloc>)>,
    FV: FnMut(VtableId) -> Option<Vec<VtableEntry>>,
{
    let mut enc = Encoder {
        outline,
        types,
        reached,
        key_of: &mut key_of,
        rodata_of: &mut rodata_of,
        vtable_of: &mut vtable_of,
        item_map: HashMap::new(),
        items: Vec::new(),
        ty_map: HashMap::new(),
        ty_count: 0,
        ty_buf: Vec::new(),
        rodata_map: HashMap::new(),
        rodatas: Vec::new(),
        vtable_map: HashMap::new(),
        vtables: Vec::new(),
    };
    let mut steps_buf = Vec::new();
    enc.write_plan(&mut steps_buf, plan)?;

    let mut out = Vec::with_capacity(32 + enc.items.len() * (ItemRecord::LEN + 4) + enc.ty_buf.len() + steps_buf.len());
    out.extend_from_slice(&PLAN_MAGIC);
    out.push(u8::from(include_deps));
    write_u32(&mut out, u32::try_from(enc.items.len()).ok()?);
    for &(record, sub) in &enc.items {
        record.write(&mut out);
        write_u32(&mut out, sub);
    }
    write_u32(&mut out, enc.ty_count);
    write_u32(&mut out, u32::try_from(enc.ty_buf.len()).ok()?);
    out.extend_from_slice(&enc.ty_buf);
    write_u32(&mut out, u32::try_from(enc.rodatas.len()).ok()?);
    out.extend_from_slice(&enc.rodatas_bytes()?);
    write_u32(&mut out, u32::try_from(enc.vtables.len()).ok()?);
    out.extend_from_slice(&enc.vtables_bytes()?);
    out.extend_from_slice(&steps_buf);
    Some(out)
}

/// Reconstructs a `RelocPlan` from `bytes` in `reached`, returning `None` cleanly if any dependency or item is absent.
pub fn decode_reloc_plan(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    slots: &dyn Slots,
    index: &FileIndex,
    sysroot_only: bool,
    bytes: &[u8],
) -> Result<Option<RelocPlan>> {
    let mut cur = Cursor::new(bytes);
    if cur.take(4)? != PLAN_MAGIC {
        bail!("invalid portable reloc plan magic");
    }
    let include_deps = cur.u8()? != 0;
    if sysroot_only && include_deps {
        return Ok(None);
    }
    let key_of = if include_deps { crate::substrate::sysroot_item_key } else { crate::substrate::sysroot_only_item_key };
    let item_count = cur.count()?;
    let mut items = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let record = ItemRecord::read(&mut cur)?;
        let sub = cur.u32()?;
        let Some(base) = index.resolve(outline, record, key_of) else { return Ok(None) };
        let item = if sub == 0 {
            base
        } else {
            let sub_idx = sub - 1;
            if base.item >= (1 << 15) || sub_idx >= (1 << 16) {
                bail!("invalid sub-item index");
            }
            base.inner(sub_idx)
        };
        items.push(item);
    }

    let ty_count = cur.count()?;
    let ty_len = cur.u32()? as usize;
    let ty_data = cur.take(ty_len)?;
    let mut ty_cur = Cursor::new(ty_data);
    let mut tys = Vec::with_capacity(ty_count);
    for _ in 0..ty_count {
        tys.push(read_ty(&mut ty_cur, types, &items, &tys)?);
    }
    if ty_cur.pos != ty_data.len() {
        bail!("trailing bytes in reloc plan type table");
    }

    let rodata_count = cur.count()?;
    let mut rodatas = Vec::with_capacity(rodata_count);
    for _ in 0..rodata_count {
        let len = cur.u32()? as usize;
        let data = cur.take(len)?;
        let reloc_count = cur.count()?;
        let mut relocs = Vec::with_capacity(reloc_count);
        for _ in 0..reloc_count {
            let offset = cur.u32()?;
            let target_idx = cur.u32()? as usize;
            let target_id = *rodatas.get(target_idx).context("rodata reloc out of bounds")?;
            relocs.push(RodataReloc { offset, target: RodataTarget::Rodata(target_id) });
        }
        rodatas.push(slots.rodata(data, &relocs));
    }

    let vtable_count = cur.count()?;
    let mut vtables = Vec::with_capacity(vtable_count);
    for _ in 0..vtable_count {
        let entry_count = cur.count()?;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            match cur.u8()? {
                0 => entries.push(VtableEntry::Word(Word::Int(cur.u64()?))),
                1 => {
                    let item = *items.get(cur.u32()? as usize).context("vtable item out of bounds")?;
                    let substs = read_substs(&mut cur, types, &tys)?;
                    let Some(idx) = resolve_reached(reached, (item, substs)) else { return Ok(None) };
                    entries.push(VtableEntry::Word(Word::Code(idx)));
                }
                2 => {
                    let super_id = *vtables.get(cur.u32()? as usize).context("super vtable out of bounds")?;
                    entries.push(VtableEntry::Super(super_id));
                }
                _ => bail!("invalid vtable entry tag"),
            }
        }
        vtables.push(slots.vtable_id(&entries));
    }

    let mut dec = Decoder { types, reached, items: &items, tys: &tys, rodatas: &rodatas, vtables: &vtables, bytes, seen: Seen::default() };
    let Some(plan) = dec.read_plan(&mut cur)? else { return Ok(None) };
    if cur.pos != bytes.len() {
        bail!("trailing bytes in portable reloc plan");
    }
    Ok(Some(plan))
}

struct EncRodata {
    bytes: Vec<u8>,
    relocs: Vec<(u32, u32)>,
}

enum EncVtableEntry {
    Int(u64),
    Code(u32, Vec<u32>),
    Super(u32),
}

struct Encoder<'a, F, FR, FV> {
    outline: &'a Outline,
    types: &'a Types,
    reached: &'a Reached,
    key_of: &'a mut F,
    rodata_of: &'a mut FR,
    vtable_of: &'a mut FV,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<(ItemRecord, u32)>,
    ty_map: HashMap<Ty, u32>,
    ty_count: u32,
    ty_buf: Vec<u8>,
    rodata_map: HashMap<RodataId, u32>,
    rodatas: Vec<EncRodata>,
    vtable_map: HashMap<VtableId, u32>,
    vtables: Vec<Vec<EncVtableEntry>>,
}

impl<F, FR, FV> Encoder<'_, F, FR, FV>
where
    F: FnMut(&Outline, ItemRef) -> Option<u64>,
    FR: FnMut(RodataId) -> Option<(Vec<u8>, Vec<RodataReloc>)>,
    FV: FnMut(VtableId) -> Option<Vec<VtableEntry>>,
{
    fn rodatas_bytes(&self) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for entry in &self.rodatas {
            write_u32(&mut out, u32::try_from(entry.bytes.len()).ok()?);
            out.extend_from_slice(&entry.bytes);
            write_u32(&mut out, u32::try_from(entry.relocs.len()).ok()?);
            for &(offset, target) in &entry.relocs {
                write_u32(&mut out, offset);
                write_u32(&mut out, target);
            }
        }
        Some(out)
    }

    fn vtables_bytes(&self) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for entries in &self.vtables {
            write_u32(&mut out, u32::try_from(entries.len()).ok()?);
            for entry in entries {
                match entry {
                    EncVtableEntry::Int(val) => { out.push(0); write_u64(&mut out, *val); }
                    EncVtableEntry::Code(item, substs) => { out.push(1); write_u32(&mut out, *item); write_u32s(&mut out, substs); }
                    EncVtableEntry::Super(id) => { out.push(2); write_u32(&mut out, *id); }
                }
            }
        }
        Some(out)
    }

    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        if let Some(&idx) = self.item_map.get(&item) {
            return Some(idx);
        }
        let (base, sub) = match item.outer() {
            Some((outer, sub)) => (outer, sub.checked_add(1)?),
            None => (item, 0),
        };
        let record = ItemRecord::of(self.outline, base, &mut *self.key_of)?;
        let idx = u32::try_from(self.items.len()).ok()?;
        self.items.push((record, sub));
        self.item_map.insert(item, idx);
        Some(idx)
    }

    fn reached_ref(&mut self, index: usize) -> Option<(u32, Vec<u32>)> {
        let ((item, substs), _) = self.reached.list.get(index)?;
        Some((self.item_id(*item)?, self.substs_ids(*substs)?))
    }

    fn substs_ids(&mut self, substs: Substs) -> Option<Vec<u32>> {
        self.types.substs(substs).iter().map(|&ty| self.ty_id(ty)).collect()
    }

    fn ty_id(&mut self, ty: Ty) -> Option<u32> {
        if ty == Ty::NONE {
            return Some(u32::MAX);
        }
        if let Some(&idx) = self.ty_map.get(&ty) {
            return Some(idx);
        }
        let mut rec = Vec::new();
        match self.types.kind(ty).clone() {
            Kind::Prim(p) => { rec.push(0); rec.push(u8::try_from(PRIMS.iter().position(|&x| x == p)?).ok()?); }
            Kind::Never => rec.push(1),
            Kind::Tuple(s) => { rec.push(2); write_u32s(&mut rec, &self.substs_ids(s)?); }
            Kind::Ref(m, i) => { rec.push(3); rec.push(u8::from(m == Mut::Mut)); write_u32(&mut rec, self.ty_id(i)?); }
            Kind::Ptr(m, i) => { rec.push(4); rec.push(u8::from(m == Mut::Mut)); write_u32(&mut rec, self.ty_id(i)?); }
            Kind::Array(e, l) => { rec.push(5); write_u32(&mut rec, self.ty_id(e)?); write_u32(&mut rec, self.ty_id(l)?); }
            Kind::Slice(e) => { rec.push(6); write_u32(&mut rec, self.ty_id(e)?); }
            Kind::Adt(i, s) => { rec.push(7); write_u32(&mut rec, self.item_id(i)?); write_u32s(&mut rec, &self.substs_ids(s)?); }
            Kind::FnDef(i, s) => { rec.push(8); write_u32(&mut rec, self.item_id(i)?); write_u32s(&mut rec, &self.substs_ids(s)?); }
            Kind::FnPtr(p, r) => { rec.push(9); write_u32s(&mut rec, &self.substs_ids(p)?); write_u32(&mut rec, self.ty_id(r)?); }
            Kind::Closure(i, s, idx) => { rec.push(10); write_u32(&mut rec, self.item_id(i)?); write_u32s(&mut rec, &self.substs_ids(s)?); write_u32(&mut rec, idx); }
            Kind::Dyn(o, s, m, a) => {
                rec.push(11);
                write_u32(&mut rec, self.item_id(o)?);
                write_u32s(&mut rec, &self.substs_ids(s)?);
                let ms = m.into_iter().map(|x| self.item_id(x)).collect::<Option<Vec<_>>>()?;
                let aus = a.into_iter().map(|x| self.item_id(x)).collect::<Option<Vec<_>>>()?;
                write_u32s(&mut rec, &ms);
                write_u32s(&mut rec, &aus);
            }
            Kind::Pat(b, lo, hi) => { rec.push(12); write_u32(&mut rec, self.ty_id(b)?); write_u128(&mut rec, lo); write_u128(&mut rec, hi); }
            Kind::Value(v) => { rec.push(13); write_u128(&mut rec, v); }
            Kind::Param(p) => { rec.push(14); write_u16(&mut rec, p); }
            Kind::Proj(id) => {
                let d = self.types.proj(id);
                rec.push(15);
                write_u32(&mut rec, self.item_id(d.of)?);
                write_u32(&mut rec, self.ty_id(d.self_ty)?);
                write_u32s(&mut rec, &self.substs_ids(d.args)?);
                write_u32(&mut rec, self.item_id(d.assoc)?);
                write_u32s(&mut rec, &self.substs_ids(d.own)?);
            }
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return None,
        }
        let idx = self.ty_count;
        self.ty_count = idx.checked_add(1)?;
        self.ty_buf.extend_from_slice(&rec);
        self.ty_map.insert(ty, idx);
        Some(idx)
    }

    fn rodata_id(&mut self, id: RodataId) -> Option<u32> {
        if let Some(&idx) = self.rodata_map.get(&id) {
            return Some(idx);
        }
        let (bytes, relocs) = (self.rodata_of)(id)?;
        let mut enc_relocs = Vec::with_capacity(relocs.len());
        for reloc in relocs {
            let RodataTarget::Rodata(child) = reloc.target;
            if child == id {
                return None;
            }
            enc_relocs.push((reloc.offset, self.rodata_id(child)?));
        }
        let idx = u32::try_from(self.rodatas.len()).ok()?;
        self.rodatas.push(EncRodata { bytes, relocs: enc_relocs });
        self.rodata_map.insert(id, idx);
        Some(idx)
    }

    fn vtable_id(&mut self, id: VtableId) -> Option<u32> {
        if let Some(&idx) = self.vtable_map.get(&id) {
            return Some(idx);
        }
        let entries = (self.vtable_of)(id)?;
        let mut enc_entries = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry {
                VtableEntry::Word(Word::Int(val)) => enc_entries.push(EncVtableEntry::Int(val)),
                VtableEntry::Word(Word::Code(idx)) => {
                    let (item, substs) = self.reached_ref(idx)?;
                    enc_entries.push(EncVtableEntry::Code(item, substs));
                }
                VtableEntry::Super(super_id) => {
                    if super_id == id {
                        return None;
                    }
                    enc_entries.push(EncVtableEntry::Super(self.vtable_id(super_id)?));
                }
            }
        }
        let idx = u32::try_from(self.vtables.len()).ok()?;
        self.vtables.push(enc_entries);
        self.vtable_map.insert(id, idx);
        Some(idx)
    }

    fn write_plan(&mut self, out: &mut Vec<u8>, plan: &RelocPlan) -> Option<()> {
        write_u32(out, u32::try_from(plan.steps.len()).ok()?);
        for step in &plan.steps {
            self.write_step(out, step)?;
        }
        Some(())
    }

    fn write_step(&mut self, out: &mut Vec<u8>, step: &RelocStep) -> Option<()> {
        match step {
            RelocStep::Const(src) => { out.push(0); self.write_symbol(out, src)?; }
            RelocStep::Promoted { seq, size } => { out.push(1); write_u32(out, *seq); write_u32(out, *size); }
            RelocStep::VtableAt(expr) => { out.push(2); write_u32(out, expr.0); }
            RelocStep::VtableSlot { expr, slot } => { out.push(3); write_u32(out, expr.0); write_u16(out, *slot); }
            RelocStep::SiteSlot(origin) => { out.push(4); self.write_site_origin(out, origin)?; }
            RelocStep::SiteInline { origin, plan } => { out.push(5); self.write_site_origin(out, origin)?; self.write_plan(out, plan)?; }
            RelocStep::TypeId(origin) => { out.push(6); self.write_ty_origin(out, origin)?; }
            RelocStep::TypeName(origin) => { out.push(7); self.write_ty_origin(out, origin)?; }
        }
        Some(())
    }

    fn write_symbol(&mut self, out: &mut Vec<u8>, src: &SymbolSource) -> Option<()> {
        match src {
            SymbolSource::Slot(index) => {
                out.push(0);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
            }
            SymbolSource::Vtable(id) => { out.push(1); write_u32(out, self.vtable_id(*id)?); }
            SymbolSource::Rodata(id) => { out.push(2); write_u32(out, self.rodata_id(*id)?); }
            SymbolSource::StaticCell { index, size } => {
                out.push(3);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
                write_u32(out, *size);
            }
            SymbolSource::PromotedCell { owner, seq, size } => {
                out.push(4);
                let (item, substs) = self.reached_ref(*owner)?;
                write_u32(out, item);
                write_u32s(out, &substs);
                write_u32(out, *seq);
                write_u32(out, *size);
            }
            SymbolSource::Host(host) => { out.push(5); write_host(out, host)?; }
            SymbolSource::Word(word) => { out.push(6); write_u64(out, *word); }
            SymbolSource::ThreadSlot(index) => {
                out.push(7);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
            }
        }
        Some(())
    }

    fn write_ty_origin(&mut self, out: &mut Vec<u8>, origin: &TyOrigin) -> Option<()> {
        match origin {
            TyOrigin::Subst(idx) => { out.push(0); write_u16(out, *idx); }
            TyOrigin::Local(idx) => { out.push(1); write_u32(out, *idx); }
            TyOrigin::Expr(expr) => { out.push(2); write_u32(out, expr.0); }
            TyOrigin::ForIter(expr) => { out.push(3); write_u32(out, expr.0); }
            TyOrigin::AwaitFuture(expr) => { out.push(4); write_u32(out, expr.0); }
            TyOrigin::SiteSubst(site, idx) => { out.push(5); self.write_site_origin(out, site)?; out.push(*idx); }
            TyOrigin::Step { base, variant, field } => { out.push(6); self.write_ty_origin(out, base)?; write_u32(out, *variant); write_u32(out, *field); }
        }
        Some(())
    }

    fn write_site_origin(&mut self, out: &mut Vec<u8>, origin: &SiteOrigin) -> Option<()> {
        match origin {
            SiteOrigin::Table(key) => { out.push(0); write_site_key(out, *key); }
            SiteOrigin::PointerFnDef { expr, callee } => { out.push(1); write_u32(out, expr.0); write_u32(out, callee.0); }
            SiteOrigin::DropGlue(ty) => { out.push(2); self.write_ty_origin(out, ty)?; }
            SiteOrigin::DropFn(ty) => { out.push(3); self.write_ty_origin(out, ty)?; }
            SiteOrigin::ConstEvalSelect(inner) => { out.push(4); self.write_site_origin(out, inner)?; }
            SiteOrigin::ShapeGlueDrop => out.push(5),
            SiteOrigin::ShapeGlueResume => out.push(6),
            SiteOrigin::ShapeUntuple => out.push(7),
            SiteOrigin::ShapeClone(idx) => { out.push(8); write_u16(out, *idx); }
            SiteOrigin::VtableMethod { expr, slot } => { out.push(9); write_u32(out, expr.0); write_u16(out, *slot); }
            SiteOrigin::Known(item, substs) => { out.push(10); write_u32(out, self.item_id(*item)?); write_u32s(out, &self.substs_ids(*substs)?); }
        }
        Some(())
    }
}

struct Decoder<'a> {
    types: &'a Types,
    reached: &'a Reached,
    items: &'a [ItemRef],
    tys: &'a [Ty],
    rodatas: &'a [RodataId],
    vtables: &'a [VtableId],
    bytes: &'a [u8],
    seen: Seen<'a>,
}

/// Inline subplans already decoded from one plan, keyed by their encoded bytes.
#[derive(Default)]
struct Seen<'a> {
    long: FxHashMap<[u8; 8], Vec<(&'a [u8], Arc<RelocPlan>)>>,
    short: Vec<(&'a [u8], Arc<RelocPlan>)>,
}

impl<'a> Seen<'a> {
    fn find(&self, rest: &[u8]) -> Option<(usize, Arc<RelocPlan>)> {
        let bucket = rest.first_chunk::<8>().and_then(|prefix| self.long.get(prefix));
        self.short.iter().chain(bucket.into_iter().flatten()).find(|(enc, _)| rest.starts_with(enc)).map(|(enc, plan)| (enc.len(), Arc::clone(plan)))
    }

    fn insert(&mut self, enc: &'a [u8], plan: &Arc<RelocPlan>) {
        let entry = (enc, Arc::clone(plan));
        match enc.first_chunk::<8>() {
            Some(prefix) => self.long.entry(*prefix).or_default().push(entry),
            None => self.short.push(entry),
        }
    }
}

impl<'a> Decoder<'a> {
    fn item_at(&self, id: u32) -> Result<ItemRef> {
        self.items.get(id as usize).copied().context("reloc plan item out of bounds")
    }

    fn read_instance(&self, cur: &mut Cursor<'_>) -> Result<Option<usize>> {
        let item = self.item_at(cur.u32()?)?;
        let substs = read_substs(cur, self.types, self.tys)?;
        Ok(resolve_reached(self.reached, (item, substs)))
    }

    fn read_plan(&mut self, cur: &mut Cursor<'_>) -> Result<Option<RelocPlan>> {
        let count = cur.count()?;
        let mut steps = Vec::with_capacity(count);
        for _ in 0..count {
            let Some(step) = self.read_step(cur)? else { return Ok(None) };
            steps.push(step);
        }
        Ok(Some(RelocPlan::from_vec(steps)))
    }

    fn read_subplan(&mut self, cur: &mut Cursor<'_>) -> Result<Option<Arc<RelocPlan>>> {
        let (start, bytes) = (cur.pos, self.bytes);
        if let Some((len, plan)) = self.seen.find(bytes.get(start..).context("reloc subplan out of bounds")?) {
            cur.take(len)?;
            return Ok(Some(plan));
        }
        let Some(plan) = self.read_plan(cur)? else { return Ok(None) };
        let plan = Arc::new(plan);
        self.seen.insert(&bytes[start..cur.pos], &plan);
        Ok(Some(plan))
    }

    fn read_step(&mut self, cur: &mut Cursor<'_>) -> Result<Option<RelocStep>> {
        Ok(Some(match cur.u8()? {
            0 => {
                let Some(sym) = self.read_symbol(cur)? else { return Ok(None) };
                RelocStep::Const(sym)
            }
            1 => RelocStep::Promoted { seq: cur.u32()?, size: cur.u32()? },
            2 => RelocStep::VtableAt(ExprId(cur.u32()?)),
            3 => RelocStep::VtableSlot { expr: ExprId(cur.u32()?), slot: cur.u16()? },
            4 => RelocStep::SiteSlot(self.read_site_origin(cur)?),
            5 => {
                let origin = self.read_site_origin(cur)?;
                let Some(plan) = self.read_subplan(cur)? else { return Ok(None) };
                RelocStep::SiteInline { origin, plan }
            }
            6 => RelocStep::TypeId(self.read_ty_origin(cur)?),
            7 => RelocStep::TypeName(self.read_ty_origin(cur)?),
            _ => bail!("invalid reloc step tag"),
        }))
    }

    fn read_symbol(&self, cur: &mut Cursor<'_>) -> Result<Option<SymbolSource>> {
        Ok(Some(match cur.u8()? {
            0 => match self.read_instance(cur)? {
                Some(idx) => SymbolSource::Slot(idx),
                None => return Ok(None),
            },
            1 => SymbolSource::Vtable(*self.vtables.get(cur.u32()? as usize).context("vtable ref out of bounds")?),
            2 => SymbolSource::Rodata(*self.rodatas.get(cur.u32()? as usize).context("rodata ref out of bounds")?),
            3 => {
                let resolved = self.read_instance(cur)?;
                let size = cur.u32()?;
                let Some(index) = resolved else { return Ok(None) };
                SymbolSource::StaticCell { index, size }
            }
            4 => {
                let resolved = self.read_instance(cur)?;
                let (seq, size) = (cur.u32()?, cur.u32()?);
                let Some(owner) = resolved else { return Ok(None) };
                SymbolSource::PromotedCell { owner, seq, size }
            }
            5 => SymbolSource::Host(read_host(cur)?),
            6 => SymbolSource::Word(cur.u64()?),
            7 => match self.read_instance(cur)? {
                Some(index) => SymbolSource::ThreadSlot(index),
                None => return Ok(None),
            },
            _ => bail!("invalid symbol source tag"),
        }))
    }

    fn read_ty_origin(&self, cur: &mut Cursor<'_>) -> Result<TyOrigin> {
        Ok(match cur.u8()? {
            0 => TyOrigin::Subst(cur.u16()?),
            1 => TyOrigin::Local(cur.u32()?),
            2 => TyOrigin::Expr(ExprId(cur.u32()?)),
            3 => TyOrigin::ForIter(ExprId(cur.u32()?)),
            4 => TyOrigin::AwaitFuture(ExprId(cur.u32()?)),
            5 => TyOrigin::SiteSubst(Box::new(self.read_site_origin(cur)?), cur.u8()?),
            6 => TyOrigin::Step { base: Box::new(self.read_ty_origin(cur)?), variant: cur.u32()?, field: cur.u32()? },
            _ => bail!("invalid ty origin tag"),
        })
    }

    fn read_site_origin(&self, cur: &mut Cursor<'_>) -> Result<SiteOrigin> {
        Ok(match cur.u8()? {
            0 => SiteOrigin::Table(read_site_key(cur)?),
            1 => SiteOrigin::PointerFnDef { expr: ExprId(cur.u32()?), callee: ExprId(cur.u32()?) },
            2 => SiteOrigin::DropGlue(self.read_ty_origin(cur)?),
            3 => SiteOrigin::DropFn(self.read_ty_origin(cur)?),
            4 => SiteOrigin::ConstEvalSelect(Box::new(self.read_site_origin(cur)?)),
            5 => SiteOrigin::ShapeGlueDrop,
            6 => SiteOrigin::ShapeGlueResume,
            7 => SiteOrigin::ShapeUntuple,
            8 => SiteOrigin::ShapeClone(cur.u16()?),
            9 => SiteOrigin::VtableMethod { expr: ExprId(cur.u32()?), slot: cur.u16()? },
            10 => SiteOrigin::Known(self.item_at(cur.u32()?)?, read_substs(cur, self.types, self.tys)?),
            _ => bail!("invalid site origin tag"),
        })
    }
}

fn resolve_reached(reached: &Reached, instance: (ItemRef, Substs)) -> Option<usize> {
    reached.index.get(&instance).copied().filter(|&idx| !reached.is_untyped(idx))
}

fn ty_at(tys: &[Ty], id: u32) -> Result<Ty> {
    if id == u32::MAX {
        Ok(Ty::NONE)
    } else {
        tys.get(id as usize).copied().context("reloc plan ty out of bounds")
    }
}

fn read_substs(cur: &mut Cursor<'_>, types: &Types, tys: &[Ty]) -> Result<Substs> {
    let count = cur.count()?;
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(ty_at(tys, cur.u32()?)?);
    }
    Ok(types.intern_substs(&ids))
}

fn read_ty(cur: &mut Cursor<'_>, types: &Types, items: &[ItemRef], tys: &[Ty]) -> Result<Ty> {
    let item_at = |id: u32| items.get(id as usize).copied().context("reloc plan item out of bounds");
    let kind = match cur.u8()? {
        0 => Kind::Prim(PRIMS.get(cur.u8()? as usize).copied().context("invalid prim tag")?),
        1 => Kind::Never,
        2 => Kind::Tuple(read_substs(cur, types, tys)?),
        3 => Kind::Ref(read_mut(cur)?, ty_at(tys, cur.u32()?)?),
        4 => Kind::Ptr(read_mut(cur)?, ty_at(tys, cur.u32()?)?),
        5 => Kind::Array(ty_at(tys, cur.u32()?)?, ty_at(tys, cur.u32()?)?),
        6 => Kind::Slice(ty_at(tys, cur.u32()?)?),
        7 => Kind::Adt(item_at(cur.u32()?)?, read_substs(cur, types, tys)?),
        8 => Kind::FnDef(item_at(cur.u32()?)?, read_substs(cur, types, tys)?),
        9 => Kind::FnPtr(read_substs(cur, types, tys)?, ty_at(tys, cur.u32()?)?),
        10 => Kind::Closure(item_at(cur.u32()?)?, read_substs(cur, types, tys)?, cur.u32()?),
        11 => {
            let (of, s) = (item_at(cur.u32()?)?, read_substs(cur, types, tys)?);
            let m_count = cur.count()?;
            let mut members = Vec::with_capacity(m_count);
            for _ in 0..m_count {
                members.push(item_at(cur.u32()?)?);
            }
            let a_count = cur.count()?;
            let mut autos = Vec::with_capacity(a_count);
            for _ in 0..a_count {
                autos.push(item_at(cur.u32()?)?);
            }
            Kind::Dyn(of, s, members, autos)
        }
        12 => Kind::Pat(ty_at(tys, cur.u32()?)?, cur.u128()?, cur.u128()?),
        13 => Kind::Value(cur.u128()?),
        14 => Kind::Param(cur.u16()?),
        15 => Kind::Proj(types.intern_proj(ProjDef {
            of: item_at(cur.u32()?)?,
            self_ty: ty_at(tys, cur.u32()?)?,
            args: read_substs(cur, types, tys)?,
            assoc: item_at(cur.u32()?)?,
            own: read_substs(cur, types, tys)?,
        })),
        _ => bail!("invalid reloc plan type tag"),
    };
    Ok(types.intern(kind))
}

fn write_host(out: &mut Vec<u8>, host: &HostSymbol) -> Option<()> {
    match host {
        HostSymbol::Alloc => out.push(0),
        HostSymbol::AllocZeroed => out.push(1),
        HostSymbol::Dealloc => out.push(2),
        HostSymbol::Realloc => out.push(3),
        HostSymbol::Noop => out.push(4),
        HostSymbol::Barrier => out.push(5),
        HostSymbol::Cpuid => out.push(6),
        HostSymbol::ExtendedState => out.push(7),
        HostSymbol::DsoHandle => out.push(8),
        HostSymbol::Abort => out.push(9),
        HostSymbol::FillOnce => out.push(10),
        HostSymbol::ThreadCell => out.push(11),
        HostSymbol::Syscall => out.push(12),
        HostSymbol::Math(m) => { out.push(13); write_str(out, m.link_name())?; }
        HostSymbol::X86(x) => { out.push(14); write_str(out, x.link_name())?; }
        HostSymbol::Dynamic(s) => { out.push(15); write_str(out, s)?; }
    }
    Some(())
}

fn read_host(cur: &mut Cursor<'_>) -> Result<HostSymbol> {
    Ok(match cur.u8()? {
        0 => HostSymbol::Alloc,
        1 => HostSymbol::AllocZeroed,
        2 => HostSymbol::Dealloc,
        3 => HostSymbol::Realloc,
        4 => HostSymbol::Noop,
        5 => HostSymbol::Barrier,
        6 => HostSymbol::Cpuid,
        7 => HostSymbol::ExtendedState,
        8 => HostSymbol::DsoHandle,
        9 => HostSymbol::Abort,
        10 => HostSymbol::FillOnce,
        11 => HostSymbol::ThreadCell,
        12 => HostSymbol::Syscall,
        13 => HostSymbol::Math(MathSymbol::from_name(read_str(cur)?).context("unknown math symbol")?),
        14 => HostSymbol::X86(X86Symbol::from_link(read_str(cur)?).context("unknown x86 symbol")?),
        15 => HostSymbol::Dynamic(read_str(cur)?.into()),
        _ => bail!("invalid host symbol tag"),
    })
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

fn write_str(out: &mut Vec<u8>, s: &str) -> Option<()> {
    write_u32(out, u32::try_from(s.len()).ok()?);
    out.extend_from_slice(s.as_bytes());
    Some(())
}

fn read_str<'a>(cur: &mut Cursor<'a>) -> Result<&'a str> {
    let len = cur.u32()? as usize;
    std::str::from_utf8(cur.take(len)?).context("invalid utf-8 symbol")
}

