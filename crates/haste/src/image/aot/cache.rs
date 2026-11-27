mod codec;
mod inline;
mod plan;
mod relocs;
mod stamps;
mod tables;
mod view;

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
use crate::lower::stencil::InlineTemplate;
use crate::lower::{FootprintId, RelocPlan, Slots, SymbolSource};
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
pub(crate) use inline::InlineCacheHit;
use inline::{InlCache, inline_bundle_path};
#[cfg(test)]
use relocs::stable_file_key;
use relocs::{LocalFileIndex, collect_inlined_local_files, hash_str, relocs_unwind_digest};
use stamps::{compute_bundle_key, resolve_aot_fn, should_skip_aot_flush};
use tables::{BundleEncoder, SharedTables, decode_symbol_list, read_code_span};
use view::BundleView;

const CACHE_MAGIC: [u8; 8] = *b"HSTAOT03";
const CACHE_VERSION: u32 = 4;
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

pub(crate) struct AotCache {
    path: PathBuf,
    config: AotConfigKey,
    stamps: Stamps,
    file_index: OnceLock<LocalFileIndex>,
    local_file_stamps: OnceLock<HashMap<u64, u64>>,
    view: BundleView,
    decoded_codes: Box<[OnceLock<Option<Arc<AotCode>>>]>,
    compiled_inlined: Mutex<HashMap<usize, Vec<(u32, u32)>>>,
    hit_inlined: Mutex<HashMap<usize, Vec<(u64, u64)>>>,
    inl: InlCache,
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
        let bundle_key = compute_bundle_key(cx.outline, reached, roots, config.config_digest);
        let path = bundle_path(bundle_key);
        let view = BundleView::open(&path, CACHE_MAGIC, CACHE_VERSION, config);
        let decoded_codes = (0..view.header.codes_count)
            .map(|_| OnceLock::new())
            .collect();
        let inl = InlCache::open(inline_bundle_path(&path), config);
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
            view,
            decoded_codes,
            compiled_inlined: Mutex::default(),
            hit_inlined: Mutex::default(),
            inl,
            has_misses: AtomicBool::new(false),
            stats,
        })
    }

    pub(in crate::image) fn has_loaded_bundle(&self) -> bool {
        self.view.has_backing()
    }

    pub(in crate::image) fn mark_miss(&self) {
        self.has_misses.store(true, Ordering::Relaxed);
    }

    pub(in crate::image) fn note_hits(&self, n: usize) {
        if n > 0 {
            self.stats.hits.fetch_add(n, Ordering::Relaxed);
        }
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

    fn key_parts(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        ty_memo: &mut HashMap<Ty, u64>,
        count_stats: bool,
    ) -> Option<(u64, u64, u64, bool, bool)> {
        let ((item, substs), res) = reached.list.get(index)?;
        if res.is_err() {
            if count_stats {
                self.stats.key_none.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        }
        let Some(item_key) = stable_entry_item_key(cx.outline, &self.stamps, *item) else {
            if count_stats {
                self.stats.key_none.fetch_add(1, Ordering::Relaxed);
                self.stats.key_item_none.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        };
        let substs_hash = if substs.is_empty() {
            0
        } else {
            let Some(sh) =
                stable_substs_hash(cx.outline, &cx.types, &self.stamps, *substs, ty_memo)
            else {
                if count_stats {
                    self.stats.key_none.fetch_add(1, Ordering::Relaxed);
                    self.stats.key_substs_none.fetch_add(1, Ordering::Relaxed);
                }
                return None;
            };
            sh
        };
        let Some(stamps_digest) = touched_stamps_digest(&cx.types, &self.stamps, *item, *substs)
        else {
            if count_stats {
                self.stats.key_none.fetch_add(1, Ordering::Relaxed);
                self.stats.key_stamps_none.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        };
        let is_cold = cx.has_attr(*item, "cold");
        let may_unwind = reached.may_unwind(cx, index);
        Some((item_key, substs_hash, stamps_digest, is_cold, may_unwind))
    }

    pub(in crate::image) fn instance_key(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
        ty_memo: &mut HashMap<Ty, u64>,
    ) -> Option<CacheInstanceKey> {
        let (item_key, substs_hash, stamps_digest, is_cold, may_unwind) =
            self.key_parts(cx, reached, index, ty_memo, true)?;
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

    pub(crate) fn lookup_inline(
        &self,
        (slots, cx, reached): (&dyn Slots, &mut Context<'_>, &Reached),
        index: usize,
        (limit, low, compact, sole, shape_only): (usize, bool, bool, bool, bool),
    ) -> Option<InlineCacheHit> {
        if !self.inl.view.has_backing() {
            self.inl.note_miss();
            return None;
        }
        let mut ty_memo = HashMap::default();
        let Some((item_key, substs_hash, stamps_digest, is_cold, may_unwind)) =
            self.key_parts(cx, reached, index, &mut ty_memo, false)
        else {
            self.inl.note_miss();
            return None;
        };
        let Some(key) = self.inl.key_for_entry(
            (item_key, substs_hash, stamps_digest),
            self.config.config_digest,
            (limit, low, compact, sole, is_cold, may_unwind),
        ) else {
            self.inl.note_miss();
            return None;
        };
        let local_stamps = self.local_file_stamps(cx.outline);
        let hit = self.inl.lookup(
            (slots, cx, reached),
            key,
            &self.file_index,
            &self.stamps,
            local_stamps,
            shape_only,
        );
        if hit.is_none() {
            self.inl.note_miss();
        }
        hit
    }

    pub(crate) fn record_inline(
        &self,
        (cx, reached): (&mut Context<'_>, &Reached),
        index: usize,
        (limit, low, compact, sole): (usize, bool, bool, bool),
        footprint: FootprintId,
        template: Option<&Arc<InlineTemplate>>,
        plan: &RelocPlan,
        relocs: &[SymbolSource],
    ) {
        let mut ty_memo = HashMap::default();
        let Some((item_key, substs_hash, stamps_digest, is_cold, may_unwind)) =
            self.key_parts(cx, reached, index, &mut ty_memo, false)
        else {
            return;
        };
        self.inl.record(
            (cx, reached),
            (item_key, substs_hash, stamps_digest),
            self.config.config_digest,
            (index, limit, low, compact, sole, is_cold, may_unwind),
            footprint,
            template,
            plan,
            relocs,
        );
    }

    fn local_file_stamps(&self, outline: &Outline) -> &HashMap<u64, u64> {
        self.local_file_stamps.get_or_init(|| {
            let mut map = HashMap::default();
            for (krate, tree) in outline.crates.iter().enumerate() {
                if tree.env.fixed {
                    continue;
                }
                for file in 0..tree.files.len() as u32 {
                    if let Some(key) = self.stamps.file_key(outline, krate as u32, file)
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
        let started = Instant::now();
        let res = self.view.shared_tables(
            cx.outline,
            &cx.types,
            &self.file_index,
            &self.stamps,
            Some(&self.stats.tables_err),
        );
        self.stats
            .tables_us
            .compare_exchange(
                0,
                started.elapsed().as_micros() as usize,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .ok();
        if res.is_none() {
            self.stats.tables_none.fetch_add(1, Ordering::Relaxed);
        }
        res
    }

    fn decode_code(&self, code_id: u32) -> Option<Arc<AotCode>> {
        let slot = self.decoded_codes.get(code_id as usize)?;
        slot.get_or_init(|| {
            let data = self.view.backing.as_ref()?.as_slice();
            let span = read_code_span(data, &self.view.header, code_id as usize)?;
            let blob = self.view.blob_slice(span.code_off, span.code_len)?;
            decode_code_blob(blob)
        })
        .clone()
    }

    pub(in crate::image) fn record_hit_inlined_batch(
        &self,
        entries: Vec<(usize, Vec<(u64, u64)>)>,
    ) {
        if entries.is_empty() {
            return;
        }
        let mut map = lock(&self.hit_inlined);
        for (index, pairs) in entries {
            map.insert(index, pairs);
        }
    }

    pub(in crate::image) fn lookup_instance_batched(
        &self,
        index: usize,
        key: &CacheInstanceKey,
        image: &Image,
        cx: &mut Context<'_>,
        reached: &Reached,
        inlined_batch: &mut Vec<(usize, Vec<(u64, u64)>)>,
    ) -> Option<(AotFn, u32)> {
        let backing = self.view.backing.as_ref()?;
        let data = backing.as_slice();
        let Some(idx) = find_record(
            data,
            self.view.header.records_off,
            self.view.header.records_count,
            (key.key_hi, key.key_lo),
        ) else {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            self.stats.lookup_find_miss.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let rec = self.view.header.records_off + idx * RECORD_LEN;
        let off = read_u32_le(data, rec + 16);
        let len = read_u32_le(data, rec + 20);
        let Some(payload) = self.view.blob_slice(off, len).filter(|p| p.len() >= 48) else {
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
        let unwind_digest = read_u64_le(payload, 40);
        let inlined_end = 48usize.checked_add(inlined_count.checked_mul(16)?)?;
        if payload.len() < inlined_end {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut inlined_pairs = Vec::new();
        if inlined_count > 0 {
            let local_stamps = self.local_file_stamps(cx.outline);
            inlined_pairs.reserve(inlined_count);
            for i in 0..inlined_count {
                let at = 48 + i * 16;
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
                |id| {
                    self.view
                        .decode_rodata(tables, image, &cx.types, reached, id)
                },
                |id| {
                    self.view
                        .decode_vtable(tables, image, &cx.types, reached, id)
                },
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
        if relocs_unwind_digest(cx, reached, &relocs) != unwind_digest {
            self.stats.lookup_none.fetch_add(1, Ordering::Relaxed);
            self.stats
                .lookup_meta_mismatch
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }
        if !inlined_pairs.is_empty() {
            inlined_batch.push((index, inlined_pairs));
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
                    Some((
                        self.stamps.file_key(outline, k, f)?,
                        self.stamps.file(k, f)?,
                    ))
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
        let (inl_loaded, inl_records, inl_hits, inl_misses) = self.inl.stats();
        let aot_loaded = self.view.header.records_count > 0;
        let aot_records = self.view.header.records_count;
        let mut flushed_records = 0usize;
        let mut flush_us = 0usize;
        let skip_rewrite = should_skip_aot_flush(
            aot_loaded,
            self.stats.hits.load(Ordering::Relaxed),
            self.stats.misses.load(Ordering::Relaxed),
            aot_records,
        );
        if should_write && self.has_misses.load(Ordering::Relaxed) && !skip_rewrite {
            let started = Instant::now();
            drop(self.view.backing.take());
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
                let unwind_digest = relocs_unwind_digest(cx, reached, relocs);
                entries.push((key, is_dict, code, relocs, inlined, unwind_digest));
            }
            let mut enc = BundleEncoder::new(cx.outline, &cx.types, reached, state, &self.stamps);
            for (key, is_dict, code, relocs, inlined, unwind_digest) in entries {
                if enc
                    .add_instance(&key, is_dict, code, relocs, &inlined, unwind_digest)
                    .is_some()
                {
                    flushed_records += 1;
                }
            }
            if let Some(sections) = enc.finish()
                && write_bundle(
                    &self.path,
                    CACHE_MAGIC,
                    CACHE_VERSION,
                    self.config,
                    &sections,
                )
                .is_ok()
                && let Some(dir) = self.path.parent()
            {
                prune_stale_bundles(dir, &self.path, "aot", MAX_BUNDLES - 1);
            }
            flush_us = started.elapsed().as_micros() as usize;
        }
        let (inl_flushed, inl_flush_us) = if should_write {
            let _ = self.local_file_stamps(cx.outline);
            if let Some(local_stamps) = self.local_file_stamps.get() {
                self.inl.flush(
                    state,
                    (cx, reached),
                    self.config,
                    &self.file_index,
                    &self.stamps,
                    local_stamps,
                )
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };
        if debug_enabled() {
            let s = &self.stats;
            let err = s.tables_err.get().map_or("none", String::as_str);
            let flush_key_none = s
                .key_none
                .load(Ordering::Relaxed)
                .saturating_sub(lookup_key_none);
            eprintln!(
                "[aot-cache] bundle={} loaded={} records={} hits={} misses={} key_none={} (item={} substs={} stamps={}) tables_none={} tables_err={err} lookup_none={} (find={} meta={} inlined={} code={} relocs={} tail={}) warm={} flushed={} flush_key_none={} open_ms={:.1} tables_ms={:.1} flush_ms={:.1} inl_loaded={} inl_records={} inl_hits={} inl_misses={} inl_flushed={} inl_flush_ms={:.1}",
                self.path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                aot_loaded,
                aot_records,
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
                inl_loaded,
                inl_records,
                inl_hits,
                inl_misses,
                inl_flushed,
                inl_flush_us as f64 / 1000.0,
            );
        }
    }
}

fn open_bundle(
    path: &Path,
    magic: [u8; 8],
    version: u32,
    config: AotConfigKey,
) -> Option<(Backing, LoadedHeader)> {
    let backing = Backing::open(path, HEADER_LEN).ok()?;
    let header = parse_bundle_header(backing.as_slice(), magic, version, config)?;
    Some((backing, header))
}

#[cfg(test)]
mod inline_tests;
#[cfg(test)]
mod tests;
