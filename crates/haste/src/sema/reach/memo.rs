use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::graph::store::home;
use crate::image::aot_cache::{
    LocalFileIndex, LocalItemRecord, Stamps, find_record, hash_entry_128, hash_str,
    prune_stale_bundles, read_u32_le, read_u64_le, stable_entry_item_key, stable_substs_hash,
    touched_stamps_digest,
};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::tree::Profile;
use crate::sema::Context;
use crate::sema::infer::{LazyTables, Local, ReachMemo, Shape, Typed};
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{Cursor, read_ty};
use crate::sema::ty::{Kind, Ty, Types};
use crate::substrate::{Backing, target_hash};

use super::unwind::UnwindScratch;
mod encode;

use super::{Instance, Reached};
use encode::{
    MemoEncoder, extract_record, has_local_closure, is_pure_sysroot_record, write_memo_bundle,
};

pub(super) const MEMO_MAGIC: [u8; 8] = *b"HSTRCH01";
pub(super) const MEMO_VERSION: u32 = 6;
pub(super) const HEADER_LEN: usize = 64;
pub(super) const RECORD_LEN: usize = 24;
const MAX_BUNDLES: usize = 16;

fn memo_dir() -> PathBuf {
    home().join("reach-memo")
}

fn ws_bundle_path(key: u64) -> PathBuf {
    memo_dir().join(format!("{key:016x}.rmemo"))
}

fn tc_bundle_path(key: u64) -> PathBuf {
    memo_dir().join(format!("tc-{key:016x}.rmemo"))
}

fn enabled() -> bool {
    crate::image::aot_cache::enabled() && !crate::env_flag("HASTE_NO_REACH_MEMO")
}

fn debug_enabled() -> bool {
    crate::env_flag("HASTE_DEBUG_REACH_MEMO")
        || std::env::var_os("HASTE_DEBUG").is_some_and(|v| v == "reach-memo" || v == "1")
}

#[derive(Clone, Copy)]
struct MemoHeader {
    items_off: usize,
    items_count: usize,
    tys_off: usize,
    tys_count: usize,
    tys_len: usize,
    records_off: usize,
    records_count: usize,
    blob_off: usize,
    blob_len: usize,
    ineligible_count: usize,
}

impl MemoHeader {
    fn parse(data: &[u8], expected_config: u64) -> Option<Self> {
        if data.len() < HEADER_LEN || data[0..8] != MEMO_MAGIC {
            return None;
        }
        if read_u32_le(data, 8) != MEMO_VERSION || read_u64_le(data, 16) != expected_config {
            return None;
        }
        let items_off = read_u32_le(data, 24) as usize;
        let items_count = read_u32_le(data, 28) as usize;
        let tys_off = read_u32_le(data, 32) as usize;
        let tys_count = read_u32_le(data, 36) as usize;
        let records_off = read_u32_le(data, 40) as usize;
        let records_count = read_u32_le(data, 44) as usize;
        let blob_off = read_u32_le(data, 48) as usize;
        let blob_len = read_u32_le(data, 52) as usize;
        let ineligible_count = read_u32_le(data, 56) as usize;
        let items_end = items_off.checked_add(items_count.checked_mul(24)?)?;
        let records_end = records_off.checked_add(records_count.checked_mul(RECORD_LEN)?)?;
        let blob_end = blob_off.checked_add(blob_len)?;
        if items_off < HEADER_LEN
            || items_end > tys_off
            || tys_off > records_off
            || records_end > blob_off
            || blob_end > data.len()
        {
            return None;
        }
        Some(Self {
            items_off,
            items_count,
            tys_off,
            tys_count,
            tys_len: records_off - tys_off,
            records_off,
            records_count,
            blob_off,
            blob_len,
            ineligible_count,
        })
    }
}

struct MemoTables {
    items: Vec<ItemRef>,
    item_valid: Vec<bool>,
    tys: Vec<Ty>,
    ty_valid: Vec<bool>,
}

const INVALID_ITEM: ItemRef = ItemRef {
    krate: u32::MAX,
    file: 0,
    item: 0,
};

impl MemoTables {
    fn decode(
        data: &[u8],
        hdr: &MemoHeader,
        outline: &Outline,
        types: &Types,
        file_index: &LocalFileIndex,
        stamps: &Stamps,
    ) -> Option<Self> {
        let mut cur = Cursor::new(data.get(hdr.items_off..hdr.items_off + hdr.items_count * 24)?);
        let mut items = Vec::with_capacity(hdr.items_count);
        let mut item_valid = Vec::with_capacity(hdr.items_count);
        for _ in 0..hdr.items_count {
            let rec = LocalItemRecord::read(&mut cur).ok()?;
            let sub = cur.u32().ok()?;
            let resolved = file_index.resolve(outline, stamps, rec).and_then(|base| {
                if sub == 0 {
                    Some(base)
                } else {
                    let sub_idx = sub - 1;
                    (base.item < (1 << 15) && sub_idx < (1 << 16)).then(|| base.inner(sub_idx))
                }
            });
            match resolved {
                Some(item) => {
                    items.push(item);
                    item_valid.push(true);
                }
                None => {
                    items.push(INVALID_ITEM);
                    item_valid.push(false);
                }
            }
        }
        let mut ty_cur = Cursor::new(data.get(hdr.tys_off..hdr.tys_off + hdr.tys_len)?);
        let mut tys = Vec::with_capacity(hdr.tys_count);
        let mut ty_valid = Vec::with_capacity(hdr.tys_count);
        for _ in 0..hdr.tys_count {
            let ty = read_ty(&mut ty_cur, types, &items, &tys).ok()?;
            let valid = ty_has_no_invalid(types, ty);
            tys.push(ty);
            ty_valid.push(valid);
        }
        Some(Self {
            items,
            item_valid,
            tys,
            ty_valid,
        })
    }

    fn read_ty_id(&self, cur: &mut Cursor<'_>) -> Option<Ty> {
        let id = cur.u32().ok()?;
        if id == u32::MAX {
            return Some(Ty::NONE);
        }
        let idx = id as usize;
        if !*self.ty_valid.get(idx)? {
            return None;
        }
        Some(self.tys[idx])
    }

    fn read_instance(&self, types: &Types, cur: &mut Cursor<'_>) -> Option<Instance> {
        let item_id = cur.u32().ok()? as usize;
        if !*self.item_valid.get(item_id)? {
            return None;
        }
        let item = self.items[item_id];
        let count = cur.u16().ok()? as usize;
        if count == 0 {
            return Some((item, Substs::EMPTY));
        }
        let mut resolved = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            let id = cur.u32().ok()?;
            if id == u32::MAX {
                resolved.push(Ty::NONE);
            } else {
                let idx = id as usize;
                if !*self.ty_valid.get(idx)? {
                    return None;
                }
                resolved.push(self.tys[idx]);
            }
        }
        Some((item, types.intern_substs(&resolved)))
    }
}

fn ty_has_no_invalid(types: &Types, root: Ty) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::default();
    while let Some(ty) = stack.pop() {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        match *types.kind(ty) {
            Kind::Adt(i, s) | Kind::FnDef(i, s) | Kind::Closure(i, s, _) => {
                if i.krate == u32::MAX {
                    return false;
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Dyn(o, s, m, a) => {
                if o.krate == u32::MAX
                    || types
                        .items(m)
                        .iter()
                        .chain(types.items(a))
                        .any(|x| x.krate == u32::MAX)
                {
                    return false;
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Proj(id) => {
                let p = types.proj(id);
                if p.of.krate == u32::MAX || p.assoc.krate == u32::MAX {
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

struct MemoLayer {
    backing: Option<Backing>,
    header: Option<MemoHeader>,
    tables: OnceLock<Option<MemoTables>>,
}

impl MemoLayer {
    fn open(path: &Path, config_digest: u64) -> Self {
        let loaded = Backing::open(path, HEADER_LEN)
            .ok()
            .and_then(|b| MemoHeader::parse(b.as_slice(), config_digest).map(|h| (b, h)));
        let (backing, header) = match loaded {
            Some((b, h)) => (Some(b), Some(h)),
            None => (None, None),
        };
        Self {
            backing,
            header,
            tables: OnceLock::new(),
        }
    }

    fn record_count(&self) -> usize {
        self.header.map_or(0, |h| h.records_count)
    }

    fn ineligible_count(&self) -> usize {
        self.header.map_or(0, |h| h.ineligible_count)
    }

    fn lookup(
        &self,
        outline: &Outline,
        types: &Types,
        file_index: &OnceLock<LocalFileIndex>,
        stamps: &Stamps,
        key: (u64, u64, u64, u64, u64),
    ) -> Option<Typed> {
        let backing = self.backing.as_ref()?;
        let hdr = self.header.as_ref()?;
        let data = backing.as_slice();
        let (key_hi, key_lo, item_key, substs_hash, stamps_digest) = key;
        let idx = find_record(data, hdr.records_off, hdr.records_count, (key_hi, key_lo))?;
        let rec = hdr.records_off + idx * RECORD_LEN;
        let off = read_u32_le(data, rec + 16) as usize;
        let len = read_u32_le(data, rec + 20) as usize;
        if off.checked_add(len)? > hdr.blob_len || len < 36 {
            return None;
        }
        let payload = data.get(hdr.blob_off + off..hdr.blob_off + off + len)?;
        if read_u64_le(payload, 0) != item_key
            || read_u64_le(payload, 8) != substs_hash
            || read_u64_le(payload, 16) != stamps_digest
        {
            return None;
        }
        let flags = u16::from_le_bytes([payload[24], payload[25]]);
        let unwind_split = u16::from_le_bytes([payload[26], payload[27]]);
        let calls_count = u16::from_le_bytes([payload[28], payload[29]]) as usize;
        let indirect_count = u16::from_le_bytes([payload[30], payload[31]]) as usize;
        let param_count = u16::from_le_bytes([payload[32], payload[33]]) as usize;
        let vdce_dyn_count = u16::from_le_bytes([payload[34], payload[35]]) as usize;
        let tables = self
            .tables
            .get_or_init(|| {
                let fidx = file_index.get_or_init(|| LocalFileIndex::build(outline));
                MemoTables::decode(data, hdr, outline, types, fidx, stamps)
            })
            .as_ref()?;
        let mut cur = Cursor::new(payload.get(36..)?);
        let mut locals = Vec::with_capacity(param_count);
        let mut params = Vec::with_capacity(param_count);
        for i in 0..param_count {
            let ty = tables.read_ty_id(&mut cur)?;
            locals.push(Local { ty });
            params.push(i as u32);
        }
        let ret = if (flags & 8) != 0 {
            Some(tables.read_ty_id(&mut cur)?)
        } else {
            None
        };
        let mut vdce_dyn_tys = Vec::with_capacity(vdce_dyn_count);
        for _ in 0..vdce_dyn_count {
            vdce_dyn_tys.push(tables.read_ty_id(&mut cur)?);
        }
        let mut calls = Vec::with_capacity(calls_count);
        for _ in 0..calls_count {
            calls.push(tables.read_instance(types, &mut cur)?);
        }
        let mut indirect = Vec::with_capacity(indirect_count);
        for _ in 0..indirect_count {
            indirect.push(tables.read_instance(types, &mut cur)?);
        }
        if 36 + cur.pos != payload.len() {
            return None;
        }
        let memo = ReachMemo {
            calls: calls.into_boxed_slice(),
            indirect: indirect.into_boxed_slice(),
            vdce_dyn_tys: vdce_dyn_tys.into_boxed_slice(),
            unwind_split,
            direct_unwinds: (flags & 1) != 0,
            skip_drops: (flags & 2) != 0,
            has_catch_unwind: (flags & 4) != 0,
        };
        Some(Typed {
            tables: LazyTables::ReachMemo(Box::new(memo)),
            locals,
            params,
            ret,
            drops: Vec::new(),
            shape: Shape::Body,
        })
    }
}

#[derive(Default)]
pub(crate) struct MemoLookupScratch {
    ty_memo: HashMap<Ty, u64>,
    item_keys: HashMap<ItemRef, Option<u64>>,
    substs_hashes: HashMap<Substs, Option<u64>>,
}

pub(crate) struct ReachMemoCache {
    ws_path: PathBuf,
    tc_path: PathBuf,
    config_digest: u64,
    stamps: Stamps,
    file_index: OnceLock<LocalFileIndex>,
    ws: MemoLayer,
    tc: MemoLayer,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

fn compute_config_digest(outline: &Outline, stamps: &Stamps, global_alloc: Option<ItemRef>) -> u64 {
    let target = crate::host_target().direct_calls(true);
    let t_hash = target_hash(&target);
    let triple_hash = hash_str(target.triple);
    let release = crate::profile() == Profile::Release;
    let alloc_stamp = global_alloc
        .and_then(|item| {
            let k = stable_entry_item_key(outline, stamps, item)?;
            let mut h = FxHasher::default();
            (k, stamps.iface(item.krate)).hash(&mut h);
            Some(h.finish())
        })
        .unwrap_or(0);
    let mut h = FxHasher::default();
    (
        MEMO_MAGIC,
        MEMO_VERSION,
        t_hash,
        triple_hash,
        release,
        alloc_stamp,
    )
        .hash(&mut h);
    h.finish()
}

fn compute_ws_key(outline: &Outline, config_digest: u64) -> u64 {
    let mut locals: Vec<(&str, u8, &Path)> = outline
        .crates
        .iter()
        .zip(&outline.roots)
        .filter(|(c, _)| !c.env.fixed)
        .map(|(_, r)| (r.name.as_str(), r.role as u8, r.path.as_path()))
        .collect();
    locals.sort_unstable();
    let mut h = FxHasher::default();
    (config_digest, locals).hash(&mut h);
    h.finish()
}

fn compute_tc_key(tc_key: u64, config_digest: u64) -> u64 {
    let mut h = FxHasher::default();
    (0x7C00_5EED_u64, config_digest, tc_key).hash(&mut h);
    h.finish()
}

impl ReachMemoCache {
    pub(crate) fn open(cx: &Context<'_>) -> Option<Arc<Self>> {
        if !enabled() {
            return None;
        }
        let target = crate::host_target().direct_calls(true);
        let t_hash = target_hash(&target);
        let tc_key = cx.store().substrate().map_or(0, |s| s.toolchain_hash());
        let stamps = Stamps::compute(cx.outline, tc_key, t_hash);
        let config_digest = compute_config_digest(cx.outline, &stamps, cx.global_allocator());
        let ws_path = ws_bundle_path(compute_ws_key(cx.outline, config_digest));
        let tc_path = tc_bundle_path(compute_tc_key(tc_key, config_digest));
        let ws = MemoLayer::open(&ws_path, config_digest);
        let tc = MemoLayer::open(&tc_path, config_digest);
        Some(Arc::new(Self {
            ws_path,
            tc_path,
            config_digest,
            stamps,
            file_index: OnceLock::new(),
            ws,
            tc,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }))
    }

    pub(crate) fn has_backing(&self) -> bool {
        self.ws.backing.is_some() || self.tc.backing.is_some()
    }

    pub(crate) fn lookup(
        &self,
        cx: &Context<'_>,
        instance: Instance,
        scratch: &mut MemoLookupScratch,
    ) -> Option<Arc<Typed>> {
        if !self.has_backing()
            || !cx.outline.is_fixed(instance.0.krate)
            || instance.0.outer().is_some()
            || cx.item(instance.0).1.kind != crate::outline::item::ItemKind::Fn
        {
            return None;
        }
        let key = self.instance_key_cached(cx.outline, &cx.types, instance, scratch)?;
        if let Some(typed) = self
            .ws
            .lookup(cx.outline, &cx.types, &self.file_index, &self.stamps, key)
            .or_else(|| {
                self.tc
                    .lookup(cx.outline, &cx.types, &self.file_index, &self.stamps, key)
            })
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(Arc::new(typed));
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    fn instance_key_cached(
        &self,
        outline: &Outline,
        types: &Types,
        (item, substs): Instance,
        scratch: &mut MemoLookupScratch,
    ) -> Option<(u64, u64, u64, u64, u64)> {
        let item_key = *scratch
            .item_keys
            .entry(item)
            .or_insert_with(|| stable_entry_item_key(outline, &self.stamps, item));
        let item_key = item_key?;
        let substs_hash = if substs.is_empty() {
            0
        } else {
            let ty_memo = &mut scratch.ty_memo;
            let stamps = &self.stamps;
            (*scratch
                .substs_hashes
                .entry(substs)
                .or_insert_with(|| stable_substs_hash(outline, types, stamps, substs, ty_memo)))?
        };
        let stamps_digest = touched_stamps_digest(types, &self.stamps, item, substs)?;
        let (key_hi, key_lo) =
            hash_entry_128(self.config_digest, item_key, substs_hash, stamps_digest, 0);
        Some((key_hi, key_lo, item_key, substs_hash, stamps_digest))
    }

    fn instance_key(
        &self,
        outline: &Outline,
        types: &Types,
        (item, substs): Instance,
        ty_memo: &mut HashMap<Ty, u64>,
    ) -> Option<(u64, u64, u64, u64, u64)> {
        let item_key = stable_entry_item_key(outline, &self.stamps, item)?;
        let substs_hash = if substs.is_empty() {
            0
        } else {
            stable_substs_hash(outline, types, &self.stamps, substs, ty_memo)?
        };
        let stamps_digest = touched_stamps_digest(types, &self.stamps, item, substs)?;
        let (key_hi, key_lo) =
            hash_entry_128(self.config_digest, item_key, substs_hash, stamps_digest, 0);
        Some((key_hi, key_lo, item_key, substs_hash, stamps_digest))
    }

    pub(crate) fn flush(&self, cx: &mut Context<'_>, reached: &Reached) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let ws_loaded = self.ws.record_count();
        let tc_loaded = self.tc.record_count();
        let ws_inel = self.ws.ineligible_count();
        let tc_inel = self.tc.ineligible_count();
        let new_ws_misses = misses.saturating_sub(ws_inel);
        let need_ws = ws_loaded == 0 || new_ws_misses > 16;
        let need_tc = tc_loaded == 0 || (misses > tc_inel + 64 && hits + misses > tc_loaded + 64);
        if !need_ws && !need_tc {
            self.log_stats(hits, misses, ws_loaded, tc_loaded, 0, 0);
            return;
        }
        let mut scratch = UnwindScratch::new();
        let mut extracted = Vec::new();
        let mut ineligible = 0u32;
        for (idx, &(instance, ref res)) in reached.list.iter().enumerate() {
            let Ok(typed) = res else { continue };
            if !cx.outline.is_fixed(instance.0.krate)
                || instance.0.outer().is_some()
                || cx.item(instance.0).1.kind != crate::outline::item::ItemKind::Fn
            {
                continue;
            }
            if !instance.1.is_empty() && has_local_closure(cx.outline, &cx.types, instance.1) {
                ineligible = ineligible.saturating_add(1);
                continue;
            }
            if let Some(rec) = extract_record(cx, reached, idx, instance, typed, &mut scratch) {
                extracted.push((instance, rec));
            } else {
                ineligible = ineligible.saturating_add(1);
            }
        }
        let mut ty_memo = HashMap::default();
        let mut ws_enc = need_ws.then(|| MemoEncoder::new(cx.outline, &cx.types, &self.stamps));
        let mut tc_enc = need_tc.then(|| MemoEncoder::new(cx.outline, &cx.types, &self.stamps));
        let mut ws_count = 0usize;
        let mut tc_count = 0usize;
        for (instance, rec) in &extracted {
            let Some(key) = self.instance_key(cx.outline, &cx.types, *instance, &mut ty_memo)
            else {
                continue;
            };
            if let Some(enc) = ws_enc.as_mut()
                && enc.add_record(key, rec).is_some()
            {
                ws_count += 1;
            }
            if let Some(enc) = tc_enc.as_mut()
                && is_pure_sysroot_record(cx.outline, &cx.types, *instance, rec)
                && enc.add_record(key, rec).is_some()
            {
                tc_count += 1;
            }
        }
        if let Some(enc) = ws_enc
            && ws_count > 0
            && write_memo_bundle(&self.ws_path, self.config_digest, ineligible, enc).is_ok()
            && let Some(dir) = self.ws_path.parent()
        {
            prune_stale_bundles(dir, &self.ws_path, "rmemo", MAX_BUNDLES - 1);
        }
        if let Some(enc) = tc_enc
            && tc_count > tc_loaded + 64
        {
            let _ = write_memo_bundle(&self.tc_path, self.config_digest, ineligible, enc);
        }
        self.log_stats(hits, misses, ws_loaded, tc_loaded, ws_count, tc_count);
    }

    fn log_stats(
        &self,
        hits: usize,
        misses: usize,
        ws_loaded: usize,
        tc_loaded: usize,
        ws_flushed: usize,
        tc_flushed: usize,
    ) {
        if debug_enabled() {
            eprintln!(
                "[reach-memo] ws_loaded={ws_loaded} tc_loaded={tc_loaded} hits={hits} misses={misses} ws_flushed={ws_flushed} tc_flushed={tc_flushed}"
            );
        }
    }
}
