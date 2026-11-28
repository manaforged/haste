use std::io::Write as _;
use std::path::Path;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph::store::write_new_with;
use crate::image::aot_cache::{LocalItemRecord, Stamps};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::infer::{Coercion, LazyTables, Shape, Typed};
use crate::sema::intrinsic::Intrinsic;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{TyTable, write_u32, write_u64};
use crate::sema::ty::{Kind, Ty, Types};

use super::super::unwind::{UnwindScratch, direct_may_unwind};
use super::super::{Instance, Reached, is_std_unwind_entry};
use super::{HEADER_LEN, MEMO_MAGIC, MEMO_VERSION, RECORD_LEN};

pub(super) struct ExtractedRecord {
    param_tys: Vec<Ty>,
    ret: Option<Ty>,
    vdce_dyn_tys: Vec<Ty>,
    calls: Vec<Instance>,
    indirect: Vec<Instance>,
    unwind_split: u16,
    flags: u16,
}

pub(super) fn extract_record(
    cx: &mut Context<'_>,
    reached: &Reached,
    idx: usize,
    (item, _substs): Instance,
    typed: &Typed,
    scratch: &mut UnwindScratch,
) -> Option<ExtractedRecord> {
    if let LazyTables::ReachMemo(memo) = &typed.tables {
        let param_tys: Vec<Ty> = typed
            .params
            .iter()
            .map(|&p| typed.locals[p as usize].ty)
            .collect();
        let flags = u16::from(memo.direct_unwinds)
            | (u16::from(memo.skip_drops) << 1)
            | (u16::from(memo.has_catch_unwind) << 2)
            | (u16::from(typed.ret.is_some()) << 3);
        return Some(ExtractedRecord {
            param_tys,
            ret: typed.ret,
            vdce_dyn_tys: memo.vdce_dyn_tys.to_vec(),
            calls: memo.calls.to_vec(),
            indirect: memo.indirect.to_vec(),
            unwind_split: memo.unwind_split,
            flags,
        });
    }
    if item.outer().is_some() || cx.item(item).1.kind != crate::outline::item::ItemKind::Fn {
        return None;
    }
    let vdce_dyn_tys = collect_vdce_dyn_tys(cx, item, typed)?;
    let (direct_unwinds, skip_drops) = direct_may_unwind(cx, reached, idx, scratch);
    let mut calls = Vec::new();
    let mut seen_calls = HashSet::default();
    let unwind_split = if skip_drops {
        for inst in typed.calls_without_drops() {
            if cx.intrinsic(inst.0).is_none() && seen_calls.insert(inst) {
                calls.push(inst);
            }
        }
        let split = u16::try_from(calls.len()).ok()?;
        for inst in typed.calls() {
            if cx.intrinsic(inst.0).is_none() && seen_calls.insert(inst) {
                calls.push(inst);
            }
        }
        split
    } else {
        for inst in typed.calls() {
            if cx.intrinsic(inst.0).is_none() && seen_calls.insert(inst) {
                calls.push(inst);
            }
        }
        u16::try_from(calls.len()).ok()?
    };
    let (_, ind_ids) = typed.collect_link_edges(&reached.index);
    let mut indirect = Vec::new();
    let mut seen_ind = HashSet::default();
    for id in ind_ids {
        if let Some((inst, _)) = reached.list.get(id)
            && cx.intrinsic(inst.0).is_none()
            && seen_ind.insert(*inst)
        {
            indirect.push(*inst);
        }
    }
    if calls.len() > u16::MAX as usize || indirect.len() > u16::MAX as usize {
        return None;
    }
    if calls
        .iter()
        .chain(&indirect)
        .any(|&(c, s)| c.outer().is_some() || substs_has_outer(&cx.types, s))
    {
        return None;
    }
    let has_catch_unwind = !cx.outline.is_sysroot(item.krate)
        && typed.calls().any(|(callee, _)| {
            if matches!(cx.intrinsic(callee), Some(Intrinsic::CatchUnwind)) {
                return true;
            }
            let (src, entry) = cx.item(callee);
            entry.name.is_some_and(|at| {
                let name = src.text(at);
                name == "catch_unwind"
                    || (cx.outline.roots[callee.krate as usize].name == "std"
                        && is_std_unwind_entry(name, src))
            })
        });
    let mut param_tys = Vec::with_capacity(typed.params.len());
    for &p in &typed.params {
        let ty = typed.locals.get(p as usize)?.ty;
        if ty_has_outer(&cx.types, ty) {
            return None;
        }
        param_tys.push(ty);
    }
    if param_tys.len() > u16::MAX as usize {
        return None;
    }
    if let Some(r) = typed.ret
        && ty_has_outer(&cx.types, r)
    {
        return None;
    }
    if vdce_dyn_tys.len() > u16::MAX as usize
        || vdce_dyn_tys.iter().any(|&ty| ty_has_outer(&cx.types, ty))
    {
        return None;
    }
    let flags = u16::from(direct_unwinds)
        | (u16::from(skip_drops) << 1)
        | (u16::from(has_catch_unwind) << 2)
        | (u16::from(typed.ret.is_some()) << 3);
    Some(ExtractedRecord {
        param_tys,
        ret: typed.ret,
        vdce_dyn_tys,
        calls,
        indirect,
        unwind_split,
        flags,
    })
}

fn coercion_has_vdce(c: &Coercion) -> bool {
    match c {
        Coercion::Vtable(_) | Coercion::Upcast(_) => true,
        Coercion::Field { inner, .. } => coercion_has_vdce(inner),
        _ => false,
    }
}

fn collect_dyn_in_ty(types: &Types, root: Ty, seen: &mut HashSet<Ty>, out: &mut Vec<Ty>) {
    if root == Ty::NONE || !seen.insert(root) {
        return;
    }
    let mut stack = vec![root];
    while let Some(ty) = stack.pop() {
        match *types.kind(ty) {
            Kind::Dyn(..) => out.push(ty),
            ref other => {
                for child in other.children(types) {
                    if child != Ty::NONE && seen.insert(child) {
                        stack.push(child);
                    }
                }
            }
        }
    }
}

fn collect_vdce_dyn_tys(cx: &Context<'_>, item: ItemRef, typed: &Typed) -> Option<Vec<Ty>> {
    if matches!(typed.shape, Shape::Virtual(_) | Shape::GlobalAlloc(_)) {
        return None;
    }
    if cx.foreign(item) && cx.symbol_name(item).as_deref() == Some("__rust_start_panic") {
        return None;
    }
    let (coercions, exprs) = typed.raw_coercions_and_exprs(&cx.types);
    if coercions.values().any(coercion_has_vdce) {
        return None;
    }
    let mut seen = HashSet::default();
    let mut out = Vec::new();
    if let Some(r) = typed.ret {
        collect_dyn_in_ty(&cx.types, r, &mut seen, &mut out);
    }
    for l in &typed.locals {
        collect_dyn_in_ty(&cx.types, l.ty, &mut seen, &mut out);
    }
    for &ty in exprs {
        collect_dyn_in_ty(&cx.types, ty, &mut seen, &mut out);
    }
    Some(out)
}

fn ty_has_outer(types: &Types, root: Ty) -> bool {
    if root == Ty::NONE {
        return false;
    }
    tys_has_outer(types, vec![root])
}

fn substs_has_outer(types: &Types, substs: Substs) -> bool {
    if substs.is_empty() {
        return false;
    }
    tys_has_outer(types, types.substs(substs).to_vec())
}

fn tys_has_outer(types: &Types, mut stack: Vec<Ty>) -> bool {
    let mut seen = HashSet::default();
    while let Some(ty) = stack.pop() {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        match *types.kind(ty) {
            Kind::Adt(i, s) | Kind::FnDef(i, s) | Kind::Closure(i, s, _) => {
                if i.outer().is_some() {
                    return true;
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Dyn(o, s, m, a) => {
                if o.outer().is_some()
                    || types
                        .items(m)
                        .iter()
                        .chain(types.items(a))
                        .any(|x| x.outer().is_some())
                {
                    return true;
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Proj(id) => {
                let p = types.proj(id);
                if p.of.outer().is_some() || p.assoc.outer().is_some() {
                    return true;
                }
                stack.push(p.self_ty);
                stack.extend_from_slice(types.substs(p.args));
                stack.extend_from_slice(types.substs(p.own));
            }
            ref other => stack.extend(other.children(types)),
        }
    }
    false
}

pub(super) fn has_local_closure(outline: &Outline, types: &Types, substs: Substs) -> bool {
    let mut stack: Vec<Ty> = types.substs(substs).to_vec();
    let mut seen = HashSet::default();
    while let Some(ty) = stack.pop() {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        match *types.kind(ty) {
            Kind::Closure(parent, s, _) => {
                if !outline.is_fixed(parent.krate) {
                    return true;
                }
                stack.extend_from_slice(types.substs(s));
            }
            ref other => stack.extend(other.children(types)),
        }
    }
    false
}

fn ty_pure_sysroot(outline: &Outline, types: &Types, root: Ty) -> bool {
    if root == Ty::NONE {
        return true;
    }
    tys_pure_sysroot(outline, types, vec![root])
}

fn substs_pure_sysroot(outline: &Outline, types: &Types, substs: Substs) -> bool {
    if substs.is_empty() {
        return true;
    }
    tys_pure_sysroot(outline, types, types.substs(substs).to_vec())
}

fn tys_pure_sysroot(outline: &Outline, types: &Types, mut stack: Vec<Ty>) -> bool {
    let mut seen = HashSet::default();
    while let Some(ty) = stack.pop() {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        match *types.kind(ty) {
            Kind::Adt(i, s) | Kind::FnDef(i, s) | Kind::Closure(i, s, _) => {
                if !outline.is_sysroot(i.krate) {
                    return false;
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Dyn(o, s, m, a) => {
                if !outline.is_sysroot(o.krate)
                    || types
                        .items(m)
                        .iter()
                        .chain(types.items(a))
                        .any(|x| !outline.is_sysroot(x.krate))
                {
                    return false;
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Proj(id) => {
                let p = types.proj(id);
                if !outline.is_sysroot(p.of.krate) || !outline.is_sysroot(p.assoc.krate) {
                    return false;
                }
                stack.push(p.self_ty);
                stack.extend_from_slice(types.substs(p.args));
                stack.extend_from_slice(types.substs(p.own));
            }
            ref other => stack.extend(other.children(types)),
        }
    }
    true
}

pub(super) fn is_pure_sysroot_record(
    outline: &Outline,
    types: &Types,
    (item, substs): Instance,
    rec: &ExtractedRecord,
) -> bool {
    outline.is_sysroot(item.krate)
        && substs_pure_sysroot(outline, types, substs)
        && rec
            .param_tys
            .iter()
            .chain(rec.ret.iter())
            .chain(&rec.vdce_dyn_tys)
            .all(|&ty| ty_pure_sysroot(outline, types, ty))
        && rec
            .calls
            .iter()
            .chain(&rec.indirect)
            .all(|&(i, s)| outline.is_sysroot(i.krate) && substs_pure_sysroot(outline, types, s))
}

pub(super) struct MemoEncoder<'a> {
    outline: &'a Outline,
    types: &'a Types,
    stamps: &'a Stamps,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<(LocalItemRecord, u32)>,
    tys: TyTable,
    records: Vec<(u64, u64, u32, u32)>,
    blob: Vec<u8>,
}

impl<'a> MemoEncoder<'a> {
    pub(super) fn new(outline: &'a Outline, types: &'a Types, stamps: &'a Stamps) -> Self {
        Self {
            outline,
            types,
            stamps,
            item_map: HashMap::default(),
            items: Vec::new(),
            tys: TyTable::default(),
            records: Vec::new(),
            blob: Vec::new(),
        }
    }

    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        if let Some(&id) = self.item_map.get(&item) {
            return Some(id);
        }
        let (base, sub) = match item.outer() {
            Some((o, s)) => (o, s.checked_add(1)?),
            None => (item, 0),
        };
        let rec = LocalItemRecord::of(self.outline, self.stamps, base)?;
        let id = u32::try_from(self.items.len()).ok()?;
        self.items.push((rec, sub));
        self.item_map.insert(item, id);
        Some(id)
    }

    fn write_instance(&mut self, out: &mut Vec<u8>, (item, substs): Instance) -> Option<()> {
        let item_id = self.item_id(item)?;
        let (outline, stamps, item_map, items) = (
            self.outline,
            self.stamps,
            &mut self.item_map,
            &mut self.items,
        );
        let substs_ids = self.tys.substs_ids(self.types, substs, &mut |i| {
            if let Some(&id) = item_map.get(&i) {
                return Some(id);
            }
            let (base, sub) = match i.outer() {
                Some((o, s)) => (o, s.checked_add(1)?),
                None => (i, 0),
            };
            let rec = LocalItemRecord::of(outline, stamps, base)?;
            let id = u32::try_from(items.len()).ok()?;
            items.push((rec, sub));
            item_map.insert(i, id);
            Some(id)
        })?;
        let count = u16::try_from(substs_ids.len()).ok()?;
        write_u32(out, item_id);
        out.extend_from_slice(&count.to_le_bytes());
        for tid in substs_ids {
            write_u32(out, tid);
        }
        Some(())
    }

    fn write_ty_id(&mut self, out: &mut Vec<u8>, ty: Ty) -> Option<()> {
        if ty == Ty::NONE {
            write_u32(out, u32::MAX);
            return Some(());
        }
        let (outline, stamps, item_map, items) = (
            self.outline,
            self.stamps,
            &mut self.item_map,
            &mut self.items,
        );
        let tid = self.tys.ty_id(self.types, ty, &mut |i| {
            if let Some(&id) = item_map.get(&i) {
                return Some(id);
            }
            let (base, sub) = match i.outer() {
                Some((o, s)) => (o, s.checked_add(1)?),
                None => (i, 0),
            };
            let rec = LocalItemRecord::of(outline, stamps, base)?;
            let id = u32::try_from(items.len()).ok()?;
            items.push((rec, sub));
            item_map.insert(i, id);
            Some(id)
        })?;
        write_u32(out, tid);
        Some(())
    }

    pub(super) fn add_record(
        &mut self,
        (key_hi, key_lo, item_key, substs_hash, stamps_digest): (u64, u64, u64, u64, u64),
        rec: &ExtractedRecord,
    ) -> Option<()> {
        let mut payload = Vec::with_capacity(
            36 + (rec.param_tys.len() + 1) * 4 + (rec.calls.len() + rec.indirect.len()) * 8,
        );
        write_u64(&mut payload, item_key);
        write_u64(&mut payload, substs_hash);
        write_u64(&mut payload, stamps_digest);
        payload.extend_from_slice(&rec.flags.to_le_bytes());
        payload.extend_from_slice(&rec.unwind_split.to_le_bytes());
        payload.extend_from_slice(&(rec.calls.len() as u16).to_le_bytes());
        payload.extend_from_slice(&(rec.indirect.len() as u16).to_le_bytes());
        payload.extend_from_slice(&(rec.param_tys.len() as u16).to_le_bytes());
        payload.extend_from_slice(&(rec.vdce_dyn_tys.len() as u16).to_le_bytes());
        for &ty in &rec.param_tys {
            self.write_ty_id(&mut payload, ty)?;
        }
        if let Some(r) = rec.ret {
            self.write_ty_id(&mut payload, r)?;
        }
        for &ty in &rec.vdce_dyn_tys {
            self.write_ty_id(&mut payload, ty)?;
        }
        for &inst in &rec.calls {
            self.write_instance(&mut payload, inst)?;
        }
        for &inst in &rec.indirect {
            self.write_instance(&mut payload, inst)?;
        }
        let off = u32::try_from(self.blob.len()).ok()?;
        let len = u32::try_from(payload.len()).ok()?;
        self.blob.extend_from_slice(&payload);
        self.records.push((key_hi, key_lo, off, len));
        Some(())
    }
}

pub(super) fn write_memo_bundle(
    path: &Path,
    config_digest: u64,
    ineligible_count: u32,
    mut enc: MemoEncoder<'_>,
) -> std::io::Result<()> {
    enc.records.sort_unstable_by_key(|&(hi, lo, _, _)| (hi, lo));
    enc.records.dedup_by_key(|&mut (hi, lo, _, _)| (hi, lo));
    let mut items_bytes = Vec::with_capacity(enc.items.len() * 24);
    for (rec, sub) in &enc.items {
        rec.write(&mut items_bytes);
        write_u32(&mut items_bytes, *sub);
    }
    let tys_count = enc.tys.count();
    let tys_bytes = enc.tys.bytes();
    let items_off = HEADER_LEN as u32;
    let tys_off = items_off + items_bytes.len() as u32;
    let records_off = tys_off + tys_bytes.len() as u32;
    let blob_off = records_off + (enc.records.len() * RECORD_LEN) as u32;
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&MEMO_MAGIC);
    write_u32(&mut header, MEMO_VERSION);
    write_u32(&mut header, 0);
    write_u64(&mut header, config_digest);
    write_u32(&mut header, items_off);
    write_u32(&mut header, enc.items.len() as u32);
    write_u32(&mut header, tys_off);
    write_u32(&mut header, tys_count);
    write_u32(&mut header, records_off);
    write_u32(&mut header, enc.records.len() as u32);
    write_u32(&mut header, blob_off);
    write_u32(&mut header, enc.blob.len() as u32);
    write_u32(&mut header, ineligible_count);
    write_u32(&mut header, 0);
    debug_assert_eq!(header.len(), HEADER_LEN);
    write_new_with(path, |w| {
        w.write_all(&header)?;
        w.write_all(&items_bytes)?;
        w.write_all(tys_bytes)?;
        for &(hi, lo, off, len) in &enc.records {
            let mut buf = [0u8; RECORD_LEN];
            buf[0..8].copy_from_slice(&lo.to_le_bytes());
            buf[8..16].copy_from_slice(&hi.to_le_bytes());
            buf[16..20].copy_from_slice(&off.to_le_bytes());
            buf[20..24].copy_from_slice(&len.to_le_bytes());
            w.write_all(&buf)?;
        }
        w.write_all(&enc.blob)
    })
}
