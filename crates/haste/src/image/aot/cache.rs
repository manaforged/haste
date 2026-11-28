mod codec;
mod relocs;
mod tables;

use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::graph::store::home;
use crate::image::Image;
use crate::image::aot::{AotCode, AotFn, AotState};
use crate::image::stencil::DictConfig;
use crate::lower::{
    RelocPlan, RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId,
    Word,
};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::tree::Profile;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sema::summary::codec::Cursor;
use crate::sema::ty::Ty;
use crate::substrate::{Backing, target_hash};
use crate::sync::lock;

use codec::{
    CacheInstanceKey, LoadedHeader, Stamps, decode_code_blob, find_record, hash_entry_128,
    parse_bundle_header, prune_stale_bundles, read_u32_le, read_u64_le, stable_entry_item_key,
    stable_substs_hash, touched_stamps_digest, write_bundle,
};
use relocs::{LocalFileIndex, collect_inlined_local_files, hash_str, stable_file_key};
use tables::{
    BundleEncoder, SharedTables, decode_symbol_list, read_code_span, read_rodata_span,
    read_vtable_span,
};

const CACHE_MAGIC: [u8; 8] = *b"HSTAOT03";
const CACHE_VERSION: u32 = 3;
const HEADER_LEN: usize = 112;
const RECORD_LEN: usize = 24;
const MAX_BUNDLES: usize = 16;

pub(in crate::image) fn enabled() -> bool {
    !crate::env_flag("HASTE_NO_AOT_CACHE") && !crate::image::verify_stencil()
}

pub(in crate::image) fn debug_enabled() -> bool {
    crate::env_flag("HASTE_DEBUG_AOT")
        || std::env::var_os("HASTE_DEBUG").is_some_and(|v| v == "aot" || v == "1")
}

fn cache_dir() -> PathBuf {
    home().join("aot-cache")
}

fn bundle_path(bundle_key: u64) -> PathBuf {
    cache_dir().join(format!("{bundle_key:016x}.aot"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct AotConfigKey {
    target_hash: u64,
    triple_hash: u64,
    flags_bits: u64,
    global_alloc_stamp: u64,
    dict_cfg: (usize, usize),
    config_digest: u64,
}

impl AotConfigKey {
    fn compute(
        outline: &Outline,
        stamps: &Stamps,
        unwind_enabled: bool,
        global_alloc: Option<ItemRef>,
        dict: DictConfig,
    ) -> Self {
        let target = crate::host_target().direct_calls(true);
        let t_hash = target_hash(&target);
        let triple_hash = hash_str(target.triple);
        let release = crate::profile() == Profile::Release;
        let x64_baseline = crate::with_cfg(|c| c.x64_baseline);
        let allow_inline = release;
        let low_entries = true;
        let flags_bits = u64::from(target.wide_vectors())
            | (u64::from(release) << 1)
            | (u64::from(unwind_enabled) << 2)
            | (u64::from(global_alloc.is_some()) << 3)
            | (u64::from(low_entries) << 4)
            | (u64::from(allow_inline) << 5)
            | (u64::from(x64_baseline) << 6);
        let global_alloc_stamp = global_alloc
            .and_then(|item| {
                let k = stable_entry_item_key(outline, stamps, item)?;
                let mut h = FxHasher::default();
                k.hash(&mut h);
                stamps.iface(item.krate).hash(&mut h);
                Some(h.finish())
            })
            .unwrap_or(0);
        let dict_cfg = (dict.min_bytes, dict.min_fanout);
        let mut h = FxHasher::default();
        CACHE_MAGIC.hash(&mut h);
        CACHE_VERSION.hash(&mut h);
        t_hash.hash(&mut h);
        triple_hash.hash(&mut h);
        flags_bits.hash(&mut h);
        global_alloc_stamp.hash(&mut h);
        dict_cfg.hash(&mut h);
        Self {
            target_hash: t_hash,
            triple_hash,
            flags_bits,
            global_alloc_stamp,
            dict_cfg,
            config_digest: h.finish(),
        }
    }
}

#[derive(Default)]
struct CacheStats {
    hits: AtomicUsize,
    misses: AtomicUsize,
    key_none: AtomicUsize,
    key_item_none: AtomicUsize,
    key_substs_none: AtomicUsize,
    key_stamps_none: AtomicUsize,
    tables_none: AtomicUsize,
    lookup_none: AtomicUsize,
    lookup_find_miss: AtomicUsize,
    lookup_meta_mismatch: AtomicUsize,
    lookup_code_none: AtomicUsize,
    lookup_inlined_stale: AtomicUsize,
    lookup_relocs_none: AtomicUsize,
    lookup_tail_mismatch: AtomicUsize,
    warm_runs: AtomicUsize,
    open_us: AtomicUsize,
    tables_us: AtomicUsize,
    tables_err: OnceLock<String>,
}

pub(in crate::image) struct AotCache {
    path: PathBuf,
    config: AotConfigKey,
    stamps: Stamps,
    file_index: OnceLock<LocalFileIndex>,
    local_file_stamps: OnceLock<HashMap<u64, u64>>,
    backing: Option<Backing>,
    header: LoadedHeader,
    tables: OnceLock<Option<SharedTables>>,
    decoded_codes: Box<[OnceLock<Option<Arc<AotCode>>>]>,
    decoded_rodatas: Box<[OnceLock<Option<RodataId>>]>,
    decoded_vtables: Box<[OnceLock<Option<VtableId>>]>,
    compiled_inlined: Mutex<HashMap<usize, Vec<(u32, u32)>>>,
    hit_inlined: Mutex<HashMap<usize, Vec<(u64, u64)>>>,
    has_misses: AtomicBool,
    stats: CacheStats,
}

impl AotCache {
    pub(in crate::image) fn open(
        cx: &mut Context<'_>,
        reached: &Reached,
        roots: &[usize],
        dict: DictConfig,
    ) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let started = Instant::now();
        let target = crate::host_target().direct_calls(true);
        let t_hash = target_hash(&target);
        let tc_key = cx.store().substrate().map_or(0, |s| s.toolchain_hash());
        let stamps = Stamps::compute(cx.outline, tc_key, t_hash);
        let unwind_enabled = crate::sema::reach::unwind_enabled(cx, reached);
        let config = AotConfigKey::compute(
            cx.outline,
            &stamps,
            unwind_enabled,
            cx.global_allocator(),
            dict,
        );
        let bundle_key = compute_bundle_key(cx.outline, reached, roots, config);
        let path = bundle_path(bundle_key);
        let (backing, header) = match open_bundle(&path, config) {
            Some((b, h)) => (Some(b), h),
            None => (None, LoadedHeader::empty()),
        };
        let decoded_codes = (0..header.codes_count).map(|_| OnceLock::new()).collect();
        let decoded_rodatas = (0..header.rodatas_count).map(|_| OnceLock::new()).collect();
        let decoded_vtables = (0..header.vtables_count).map(|_| OnceLock::new()).collect();
        let stats = CacheStats::default();
        stats
            .open_us
            .store(started.elapsed().as_micros() as usize, Ordering::Relaxed);
        Some(Self {
            path,
            config,
            stamps,
            file_index: OnceLock::new(),
            local_file_stamps: OnceLock::new(),
            backing,
            header,
            tables: OnceLock::new(),
            decoded_codes,
            decoded_rodatas,
            decoded_vtables,
            compiled_inlined: Mutex::default(),
            hit_inlined: Mutex::default(),
            has_misses: AtomicBool::new(false),
            stats,
        })
    }

    pub(in crate::image) fn has_loaded_bundle(&self) -> bool {
        self.backing.is_some()
    }

    pub(in crate::image) fn mark_miss(&self) {
        self.has_misses.store(true, Ordering::Relaxed);
    }

    pub(in crate::image) fn note_hit(&self) {
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub(in crate::image) fn note_miss(&self) {
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(in crate::image) fn note_warm(&self) {
        self.stats.warm_runs.fetch_add(1, Ordering::Relaxed);
    }

    pub(in crate::image) fn record_compiled_inlined(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        plan: &RelocPlan,
        index: usize,
        relocs: &[SymbolSource],
    ) {
        let files = collect_inlined_local_files(cx, reached, plan, index, relocs);
        if !files.is_empty() {
            lock(&self.compiled_inlined).insert(index, files);
        }
    }

    pub(in crate::image) fn instance_key(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        ty_memo: &mut HashMap<Ty, u64>,
    ) -> Option<CacheInstanceKey> {
        let ((item, substs), res) = reached.list.get(index)?;
        if res.is_err() {
            self.stats.key_none.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let Some(item_key) = stable_entry_item_key(cx.outline, &self.stamps, *item) else {
            self.stats.key_none.fetch_add(1, Ordering::Relaxed);
            self.stats.key_item_none.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let substs_hash = if substs.is_empty() {
            0
        } else {
            let Some(sh) =
                stable_substs_hash(cx.outline, &cx.types, &self.stamps, *substs, ty_memo)
            else {
                self.stats.key_none.fetch_add(1, Ordering::Relaxed);
                self.stats.key_substs_none.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            sh
        };
        let Some(stamps_digest) = touched_stamps_digest(&cx.types, &self.stamps, *item, *substs)
        else {
            self.stats.key_none.fetch_add(1, Ordering::Relaxed);
            self.stats.key_stamps_none.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let is_cold = cx.has_attr(*item, "cold");
        let may_unwind = reached.may_unwind(cx, index);
        let flags = u32::from(is_cold) | (u32::from(may_unwind) << 1);
        let (key_hi, key_lo) = hash_entry_128(
            self.config.config_digest,
            item_key,
            substs_hash,
            stamps_digest,
            flags,
        );
        Some(CacheInstanceKey {
            key_hi,
            key_lo,
            item_key,
            substs_hash,
            stamps_digest,
            flags,
        })
    }

    fn local_file_stamps(&self, outline: &Outline) -> &HashMap<u64, u64> {
        self.local_file_stamps.get_or_init(|| {
            let mut map = HashMap::default();
            for (krate, tree) in outline.crates.iter().enumerate() {
                if tree.env.fixed {
                    continue;
                }
                for file in 0..tree.files.len() as u32 {
                    if let Some(key) = stable_file_key(outline, krate as u32, file)
                        && let Some(stamp) = self.stamps.file(krate as u32, file)
                    {
                        map.insert(key, stamp);
                    }
                }
            }
            map
        })
    }

    fn shared_tables(&self, cx: &Context<'_>) -> Option<&SharedTables> {
        let data = self.backing.as_ref()?.as_slice();
        let res = self
            .tables
            .get_or_init(|| {
                let started = Instant::now();
                let file_index = self
                    .file_index
                    .get_or_init(|| LocalFileIndex::build(cx.outline));
                let decoded = match SharedTables::decode(
                    data,
                    &self.header,
                    cx.outline,
                    &cx.types,
                    file_index,
                    &self.stamps,
                ) {
                    Ok(t) => Some(t),
                    Err(err) => {
                        drop(self.stats.tables_err.set(err.to_string()));
                        None
                    }
                };
                self.stats
                    .tables_us
                    .store(started.elapsed().as_micros() as usize, Ordering::Relaxed);
                decoded
            })
            .as_ref();
        if res.is_none() {
            self.stats.tables_none.fetch_add(1, Ordering::Relaxed);
        }
        res
    }

    fn blob_slice(&self, off: u32, len: u32) -> Option<&[u8]> {
        let data = self.backing.as_ref()?.as_slice();
        let off = off as usize;
        let end = off.checked_add(len as usize)?;
        if end > self.header.blob_len {
            return None;
        }
        data.get(self.header.blob_off + off..self.header.blob_off + end)
    }

    fn decode_code(&self, code_id: u32) -> Option<Arc<AotCode>> {
        let slot = self.decoded_codes.get(code_id as usize)?;
        slot.get_or_init(|| {
            let data = self.backing.as_ref()?.as_slice();
            let span = read_code_span(data, &self.header, code_id as usize)?;
            let blob = self.blob_slice(span.code_off, span.code_len)?;
            decode_code_blob(blob)
        })
        .clone()
    }

    fn decode_rodata(
        &self,
        tables: &SharedTables,
        image: &Image,
        cx: &Context<'_>,
        reached: &Reached,
        id: u32,
    ) -> Option<RodataId> {
        let slot = self.decoded_rodatas.get(id as usize)?;
        *slot.get_or_init(|| {
            let data = self.backing.as_ref()?.as_slice();
            let span = read_rodata_span(data, &self.header, id as usize)?;
            let bytes = self.blob_slice(span.data_off, span.data_len)?;
            let reloc_base = self.header.blob_off + span.reloc_off as usize;
            let reloc_slice = data.get(reloc_base..self.header.blob_off + self.header.blob_len)?;
            let mut cur = Cursor::new(reloc_slice);
            let mut relocs = Vec::with_capacity((span.reloc_count as usize).min(4096));
            for _ in 0..span.reloc_count {
                let offset = cur.u32().ok()?;
                let target = match cur.u8().ok()? {
                    0 => {
                        let child_id = cur.u32().ok()?;
                        if child_id >= id {
                            return None;
                        }
                        RodataTarget::Rodata(
                            self.decode_rodata(tables, image, cx, reached, child_id)?,
                        )
                    }
                    1 => {
                        RodataTarget::Fn(tables.read_instance(&cx.types, reached, image, &mut cur)?)
                    }
                    _ => return None,
                };
                relocs.push(RodataReloc { offset, target });
            }
            Some(image.rodata(bytes, &relocs))
        })
    }

    fn decode_vtable(
        &self,
        tables: &SharedTables,
        image: &Image,
        cx: &Context<'_>,
        reached: &Reached,
        id: u32,
    ) -> Option<VtableId> {
        let slot = self.decoded_vtables.get(id as usize)?;
        *slot.get_or_init(|| {
            let data = self.backing.as_ref()?.as_slice();
            let span = read_vtable_span(data, &self.header, id as usize)?;
            let entry_base = self.header.blob_off + span.entry_off as usize;
            let entry_slice = data.get(entry_base..self.header.blob_off + self.header.blob_len)?;
            let mut cur = Cursor::new(entry_slice);
            let mut entries = Vec::with_capacity((span.entry_count as usize).min(1024));
            for _ in 0..span.entry_count {
                let entry = match cur.u8().ok()? {
                    0 => VtableEntry::Word(Word::Int(cur.u64().ok()?)),
                    1 => VtableEntry::Word(Word::Code(
                        tables.read_instance(&cx.types, reached, image, &mut cur)?,
                    )),
                    2 => {
                        let super_id = cur.u32().ok()?;
                        if super_id >= id {
                            return None;
                        }
                        VtableEntry::Super(
                            self.decode_vtable(tables, image, cx, reached, super_id)?,
                        )
                    }
                    _ => return None,
                };
                entries.push(entry);
            }
            Some(image.vtable_id(&entries))
        })
    }

    pub(in crate::image) fn lookup_instance(
        &self,
        index: usize,
        key: &CacheInstanceKey,
        image: &Image,
        cx: &Context<'_>,
        reached: &Reached,
    ) -> Option<(AotFn, u32)> {
        let backing = self.backing.as_ref()?;
        let data = backing.as_slice();
        let Some(idx) = find_record(
            data,
            self.header.records_off,
            self.header.records_count,
            (key.key_hi, key.key_lo),
        ) else {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            self.stats.lookup_find_miss.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let rec = self.header.records_off + idx * RECORD_LEN;
        let off = read_u32_le(data, rec + 16);
        let len = read_u32_le(data, rec + 20);
        let Some(payload) = self.blob_slice(off, len).filter(|p| p.len() >= 40) else {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let item_key = read_u64_le(payload, 0);
        let substs_hash = read_u64_le(payload, 8);
        let stamps_digest = read_u64_le(payload, 16);
        let flags = read_u32_le(payload, 24);
        if item_key != key.item_key
            || substs_hash != key.substs_hash
            || stamps_digest != key.stamps_digest
            || (flags & 3) != key.flags
        {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            self.stats
                .lookup_meta_mismatch
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let is_dict = (flags & 4) != 0;
        let code_id = read_u32_le(payload, 28);
        let reloc_count = read_u32_le(payload, 32) as usize;
        let inlined_count = read_u32_le(payload, 36) as usize;
        let inlined_end = 40usize.checked_add(inlined_count.checked_mul(16)?)?;
        if payload.len() < inlined_end {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut inlined_pairs = Vec::new();
        if inlined_count > 0 {
            let local_stamps = self.local_file_stamps(cx.outline);
            inlined_pairs.reserve(inlined_count);
            for i in 0..inlined_count {
                let at = 40 + i * 16;
                let f_key = read_u64_le(payload, at);
                let f_stamp = read_u64_le(payload, at + 8);
                if local_stamps.get(&f_key).copied() != Some(f_stamp) {
                    self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .lookup_inlined_stale
                        .fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                inlined_pairs.push((f_key, f_stamp));
            }
        }
        let Some(code) = self.decode_code(code_id) else {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            self.stats.lookup_code_none.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let relocs = if reloc_count == 0 {
            if inlined_end != payload.len() {
                self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .lookup_tail_mismatch
                    .fetch_add(1, Ordering::Relaxed);
                return None;
            }
            Vec::new()
        } else {
            let Some(tables) = self.shared_tables(cx) else {
                self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            let mut cur = Cursor::new(payload.get(inlined_end..)?);
            let Some(r) = decode_symbol_list(
                tables,
                &cx.types,
                reached,
                image,
                &mut cur,
                reloc_count,
                |id| self.decode_rodata(tables, image, cx, reached, id),
                |id| self.decode_vtable(tables, image, cx, reached, id),
            ) else {
                self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .lookup_relocs_none
                    .fetch_add(1, Ordering::Relaxed);
                return None;
            };
            if inlined_end + cur.pos != payload.len() {
                self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .lookup_tail_mismatch
                    .fetch_add(1, Ordering::Relaxed);
                return None;
            }
            r
        };
        if !inlined_pairs.is_empty() {
            lock(&self.hit_inlined).insert(index, inlined_pairs);
        }
        let len_u32 = code.bytes().len() as u32;
        let entry = if is_dict {
            AotFn::Dict { body: code, relocs }
        } else {
            AotFn::Direct { code, relocs }
        };
        Some((entry, len_u32))
    }

    fn inlined_pairs_for(&self, outline: &Outline, idx: usize, leader: usize) -> Vec<(u64, u64)> {
        let compiled = {
            let map = lock(&self.compiled_inlined);
            map.get(&idx).or_else(|| map.get(&leader)).cloned()
        };
        if let Some(files) = compiled {
            let mut pairs: Vec<(u64, u64)> = files
                .into_iter()
                .filter_map(|(k, f)| {
                    Some((stable_file_key(outline, k, f)?, self.stamps.file(k, f)?))
                })
                .collect();
            pairs.sort_unstable();
            pairs.dedup();
            return pairs;
        }
        let map = lock(&self.hit_inlined);
        map.get(&idx)
            .or_else(|| map.get(&leader))
            .cloned()
            .unwrap_or_default()
    }

    pub(in crate::image) fn flush(
        mut self,
        state: &AotState,
        cx: &mut Context<'_>,
        reached: &Reached,
        should_write: bool,
    ) {
        let lookup_key_none = self.stats.key_none.load(Ordering::Relaxed);
        let lookup_item_none = self.stats.key_item_none.load(Ordering::Relaxed);
        let lookup_substs_none = self.stats.key_substs_none.load(Ordering::Relaxed);
        let lookup_stamps_none = self.stats.key_stamps_none.load(Ordering::Relaxed);
        let mut flushed_records = 0usize;
        let mut flush_us = 0usize;
        if should_write && self.has_misses.load(Ordering::Relaxed) {
            let started = Instant::now();
            drop(self.backing.take());
            let mut ty_memo = HashMap::default();
            let mut entries = Vec::with_capacity(state.functions.len());
            for idx in 0..state.functions.len() {
                let Some((leader, is_dict, code, relocs)) = resolve_aot_fn(&state.functions, idx)
                else {
                    continue;
                };
                let Some(key) = self.instance_key(cx, reached, idx, &mut ty_memo) else {
                    continue;
                };
                let inlined = self.inlined_pairs_for(cx.outline, idx, leader);
                entries.push((key, is_dict, code, relocs, inlined));
            }
            let mut enc = BundleEncoder::new(cx.outline, &cx.types, reached, state, &self.stamps);
            for (key, is_dict, code, relocs, inlined) in entries {
                if enc
                    .add_instance(&key, is_dict, code, relocs, &inlined)
                    .is_some()
                {
                    flushed_records += 1;
                }
            }
            if let Some(sections) = enc.finish()
                && write_bundle(&self.path, self.config, &sections).is_ok()
                && let Some(dir) = self.path.parent()
            {
                prune_stale_bundles(dir, &self.path, MAX_BUNDLES - 1);
            }
            flush_us = started.elapsed().as_micros() as usize;
        }
        if debug_enabled() {
            let s = &self.stats;
            let err = s.tables_err.get().map_or("none", String::as_str);
            let flush_key_none = s
                .key_none
                .load(Ordering::Relaxed)
                .saturating_sub(lookup_key_none);
            eprintln!(
                "[aot-cache] bundle={} loaded={} records={} hits={} misses={} key_none={} (item={} substs={} stamps={}) tables_none={} tables_err={err} lookup_none={} (find={} meta={} inlined={} code={} relocs={} tail={}) warm={} flushed={} flush_key_none={} open_ms={:.1} tables_ms={:.1} flush_ms={:.1}",
                self.path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                self.header.records_count > 0,
                self.header.records_count,
                s.hits.load(Ordering::Relaxed),
                s.misses.load(Ordering::Relaxed),
                lookup_key_none,
                lookup_item_none,
                lookup_substs_none,
                lookup_stamps_none,
                s.tables_none.load(Ordering::Relaxed),
                s.lookup_none.load(Ordering::Relaxed),
                s.lookup_find_miss.load(Ordering::Relaxed),
                s.lookup_meta_mismatch.load(Ordering::Relaxed),
                s.lookup_inlined_stale.load(Ordering::Relaxed),
                s.lookup_code_none.load(Ordering::Relaxed),
                s.lookup_relocs_none.load(Ordering::Relaxed),
                s.lookup_tail_mismatch.load(Ordering::Relaxed),
                s.warm_runs.load(Ordering::Relaxed),
                flushed_records,
                flush_key_none,
                s.open_us.load(Ordering::Relaxed) as f64 / 1000.0,
                s.tables_us.load(Ordering::Relaxed) as f64 / 1000.0,
                flush_us as f64 / 1000.0,
            );
        }
    }
}

fn resolve_aot_fn(
    functions: &[Option<AotFn>],
    mut idx: usize,
) -> Option<(usize, bool, &Arc<AotCode>, &[SymbolSource])> {
    for _ in 0..functions.len() {
        match functions.get(idx).and_then(Option::as_ref)? {
            AotFn::Alias(next) if *next < functions.len() => idx = *next,
            AotFn::Direct { code, relocs } => return Some((idx, false, code, relocs)),
            AotFn::Dict { body, relocs } => return Some((idx, true, body, relocs)),
            AotFn::Alias(_) => return None,
        }
    }
    None
}

fn open_bundle(path: &Path, config: AotConfigKey) -> Option<(Backing, LoadedHeader)> {
    let backing = Backing::open(path, HEADER_LEN).ok()?;
    let header = parse_bundle_header(backing.as_slice(), config)?;
    Some((backing, header))
}

fn compute_bundle_key(
    outline: &Outline,
    reached: &Reached,
    roots: &[usize],
    config: AotConfigKey,
) -> u64 {
    let mut locals: Vec<(&str, u8, &Path)> = outline
        .crates
        .iter()
        .zip(&outline.roots)
        .filter(|(c, _)| !c.env.fixed)
        .map(|(_, r)| (r.name.as_str(), r.role as u8, r.path.as_path()))
        .collect();
    locals.sort_unstable();
    let mut root_tags: Vec<(&str, &Path, u32, u32)> = roots
        .iter()
        .take(2)
        .filter_map(|&idx| {
            let (item, _) = reached.list.get(idx)?.0;
            let r = outline.roots.get(item.krate as usize)?;
            Some((r.name.as_str(), r.path.as_path(), item.file, item.item))
        })
        .collect();
    root_tags.sort_unstable();
    let mut h = FxHasher::default();
    config.config_digest.hash(&mut h);
    locals.hash(&mut h);
    root_tags.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests;
