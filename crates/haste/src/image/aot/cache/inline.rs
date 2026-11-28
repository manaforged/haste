use std::hash::{Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use haste_codegen::Function;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::image::aot::AotState;
use crate::lower::stencil::{Form, InlineTemplate};
use crate::lower::{
    FootprintId, RelocPlan, RodataId, RodataReloc, Slots, SymbolSource, VtableEntry, VtableId, Word,
};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{Cursor, write_u32, write_u32s, write_u64};
use crate::sync::lock;

use super::codec::{
    Stamps, find_record, prune_stale_bundles, read_u32_le, read_u64_le, write_bundle,
};
use super::plan::{
    decode_inline_shape, decode_reloc_plan, encode_reloc_plan, pack_inline_blob, split_inline_blob,
};
use super::relocs::{LocalFileIndex, collect_inlined_local_files};
use super::tables::{BundleEncoder, CARRY_BIT, SharedTables, read_code_span};
use super::{AotConfigKey, BundleView, MAX_BUNDLES, RECORD_LEN};

pub(super) const INL_MAGIC: [u8; 8] = *b"HSTINL01";
pub(super) const INL_VERSION: u32 = 2;
const NEG_BODY_ID: u32 = u32::MAX;
const INL_HEADER_LEN: usize = 60;

pub(crate) enum InlineCacheHit {
    Negative(FootprintId),
    Shape(FootprintId, bool, u8, u16),
    Body(FootprintId, Arc<InlineTemplate>),
}

pub(super) struct InlKey {
    key_hi: u64,
    key_lo: u64,
    item_key: u64,
    substs_hash: u64,
    stamps_digest: u64,
    extra_flags: u64,
}

struct CompiledInline {
    key: InlKey,
    footprint: FootprintId,
    template: Option<Arc<InlineTemplate>>,
    inlined: Vec<(u32, u32)>,
    no_unwind: Vec<(ItemRef, Substs)>,
}

pub(super) struct InlCache {
    path: PathBuf,
    pub(super) view: BundleView,
    decoded_bodies: Box<[OnceLock<Option<Arc<InlineTemplate>>>]>,
    compiled: Mutex<HashMap<(u64, u64), CompiledInline>>,
    has_misses: AtomicBool,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl InlCache {
    pub(super) fn open(path: PathBuf, config: AotConfigKey) -> Self {
        let view = BundleView::open(&path, INL_MAGIC, INL_VERSION, config);
        let decoded_bodies = (0..view.header.codes_count)
            .map(|_| OnceLock::new())
            .collect();
        Self {
            path,
            view,
            decoded_bodies,
            compiled: Mutex::default(),
            has_misses: AtomicBool::new(false),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    pub(super) fn stats(&self) -> (bool, usize, usize, usize) {
        (
            self.view.has_backing(),
            self.view.header.records_count,
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    fn key_for(
        &self,
        key_parts: (u64, u64, u64),
        config_digest: u64,
        (limit, low, compact, sole, is_cold, may_unwind): (usize, bool, bool, bool, bool, bool),
    ) -> Option<InlKey> {
        let limit_u32 = u32::try_from(limit).ok()?;
        let extra_flags = u64::from(limit_u32)
            | (u64::from(low) << 32)
            | (u64::from(compact) << 33)
            | (u64::from(sole) << 34)
            | (u64::from(is_cold) << 35)
            | (u64::from(may_unwind) << 36);
        let (key_hi, key_lo) = hash_inline_128(
            config_digest,
            key_parts.0,
            key_parts.1,
            key_parts.2,
            extra_flags,
        );
        Some(InlKey {
            key_hi,
            key_lo,
            item_key: key_parts.0,
            substs_hash: key_parts.1,
            stamps_digest: key_parts.2,
            extra_flags,
        })
    }

    pub(super) fn lookup(
        &self,
        (slots, cx, reached): (&dyn Slots, &mut Context<'_>, &Reached),
        key: InlKey,
        file_index: &OnceLock<LocalFileIndex>,
        stamps: &Stamps,
        local_stamps: &HashMap<u64, u64>,
        shape_only: bool,
    ) -> Option<InlineCacheHit> {
        let backing = self.view.backing.as_ref()?;
        let data = backing.as_slice();
        let idx = find_record(
            data,
            self.view.header.records_off,
            self.view.header.records_count,
            (key.key_hi, key.key_lo),
        )?;
        let rec = self.view.header.records_off + idx * RECORD_LEN;
        let off = read_u32_le(data, rec + 16);
        let len = read_u32_le(data, rec + 20);
        let payload = self
            .view
            .blob_slice(off, len)
            .filter(|p| p.len() >= INL_HEADER_LEN)?;
        if read_u64_le(payload, 0) != key.item_key
            || read_u64_le(payload, 8) != key.substs_hash
            || read_u64_le(payload, 16) != key.stamps_digest
            || read_u64_le(payload, 24) != key.extra_flags
        {
            return None;
        }
        let fp_lo = read_u64_le(payload, 32);
        let fp_hi = read_u64_le(payload, 40);
        let footprint = FootprintId(u128::from(fp_lo) | (u128::from(fp_hi) << 64));
        let body_id = read_u32_le(payload, 48);
        let inlined_count = read_u32_le(payload, 52) as usize;
        let no_unwind_count = read_u32_le(payload, 56) as usize;
        let inlined_end = INL_HEADER_LEN.checked_add(inlined_count.checked_mul(16)?)?;
        if payload.len() < inlined_end {
            return None;
        }
        for i in 0..inlined_count {
            let at = INL_HEADER_LEN + i * 16;
            let f_key = read_u64_le(payload, at);
            let f_stamp = read_u64_le(payload, at + 8);
            if local_stamps.get(&f_key).copied() != Some(f_stamp) {
                return None;
            }
        }
        if no_unwind_count == 0 {
            if payload.len() != inlined_end {
                return None;
            }
        } else {
            let tables = self
                .view
                .shared_tables(cx.outline, &cx.types, file_index, stamps, None)?;
            decode_valid_no_unwind(
                tables,
                cx,
                reached,
                payload.get(inlined_end..)?,
                no_unwind_count,
            )?;
        }
        if body_id == NEG_BODY_ID {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(InlineCacheHit::Negative(footprint));
        }
        if shape_only {
            let span = read_code_span(data, &self.view.header, body_id as usize)?;
            let blob = self.view.blob_slice(span.code_off, span.code_len)?;
            let (returns, height, size) = decode_inline_shape(blob)?;
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(InlineCacheHit::Shape(footprint, returns, height, size));
        }
        let template = self.decode_body(body_id, slots, cx, reached, file_index, stamps)?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(InlineCacheHit::Body(footprint, template))
    }

    pub(super) fn note_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn record(
        &self,
        (cx, reached): (&mut Context<'_>, &Reached),
        key_parts: (u64, u64, u64),
        config_digest: u64,
        (index, limit, low, compact, sole, is_cold, may_unwind): (
            usize,
            usize,
            bool,
            bool,
            bool,
            bool,
            bool,
        ),
        footprint: FootprintId,
        template: Option<&Arc<InlineTemplate>>,
        plan: &RelocPlan,
        relocs: &[SymbolSource],
    ) {
        if template.is_some() && relocs.is_empty() && (plan.has_calls() || plan.reloc_len() > 0) {
            return;
        }
        let Some(key) = self.key_for(
            key_parts,
            config_digest,
            (limit, low, compact, sole, is_cold, may_unwind),
        ) else {
            return;
        };
        let inlined = if relocs.is_empty() {
            Vec::new()
        } else {
            collect_inlined_local_files(cx, reached, plan, index, relocs)
        };
        let mut no_unwind = Vec::new();
        if template.is_some() {
            let mut seen = HashSet::default();
            for sym in relocs {
                if let SymbolSource::Slot(c_idx) = *sym
                    && seen.insert(c_idx)
                    && !reached.may_unwind(cx, c_idx)
                    && let Some(((item, substs), _)) = reached.list.get(c_idx)
                {
                    no_unwind.push((*item, *substs));
                }
            }
        }
        self.has_misses.store(true, Ordering::Relaxed);
        let pair = (key.key_hi, key.key_lo);
        lock(&self.compiled).entry(pair).or_insert(CompiledInline {
            key,
            footprint,
            template: template.map(Arc::clone),
            inlined,
            no_unwind,
        });
    }

    pub(super) fn key_for_entry(
        &self,
        key_parts: (u64, u64, u64),
        config_digest: u64,
        flags: (usize, bool, bool, bool, bool, bool),
    ) -> Option<InlKey> {
        self.key_for(key_parts, config_digest, flags)
    }

    fn decode_body(
        &self,
        body_id: u32,
        slots: &dyn Slots,
        cx: &Context<'_>,
        reached: &Reached,
        file_index: &OnceLock<LocalFileIndex>,
        stamps: &Stamps,
    ) -> Option<Arc<InlineTemplate>> {
        let slot = self.decoded_bodies.get(body_id as usize)?;
        slot.get_or_init(|| {
            let data = self.view.backing.as_ref()?.as_slice();
            let span = read_code_span(data, &self.view.header, body_id as usize)?;
            let blob = self.view.blob_slice(span.code_off, span.code_len)?;
            let (returns, height, size, form, plan_bytes, func_bytes) = split_inline_blob(blob)?;
            let tables = self
                .view
                .shared_tables(cx.outline, &cx.types, file_index, stamps, None)?;
            let reloc_plan = decode_reloc_plan(
                tables,
                &cx.types,
                reached,
                slots,
                plan_bytes,
                |id| {
                    self.view
                        .decode_rodata(tables, slots, &cx.types, reached, id)
                },
                |id| {
                    self.view
                        .decode_vtable(tables, slots, &cx.types, reached, id)
                },
            )?;
            let body = Function::decode(func_bytes)?;
            Some(Arc::new(InlineTemplate {
                body: Arc::new(body),
                returns,
                reloc_plan: Arc::new(reloc_plan),
                height,
                size,
                form,
            }))
        })
        .clone()
    }

    pub(super) fn flush(
        mut self,
        state: &AotState,
        (cx, reached): (&mut Context<'_>, &Reached),
        config: AotConfigKey,
        file_index: &OnceLock<LocalFileIndex>,
        stamps: &Stamps,
        local_stamps: &HashMap<u64, u64>,
    ) -> (usize, usize) {
        if !self.has_misses.load(Ordering::Relaxed) {
            return (0, 0);
        }
        let compiled = std::mem::take(&mut *lock(&self.compiled));
        let existing_records = self.view.header.records_count;
        if compiled.is_empty()
            || (self.view.has_backing()
                && compiled.len() < 128
                && compiled.len().saturating_mul(16) < existing_records)
        {
            return (0, 0);
        }
        let started = Instant::now();
        let carry_slots = CarrySlots::default();
        let mut carried_blobs = Vec::new();
        let mut seen_keys = HashSet::default();
        for (pair, _) in &compiled {
            seen_keys.insert(*pair);
        }
        self.collect_carry_over(
            &seen_keys,
            &carry_slots,
            (cx, reached, file_index, stamps, local_stamps),
            &mut carried_blobs,
        );
        let extra_rodata = lock(&carry_slots.rodata);
        let extra_vtables = lock(&carry_slots.vtables);
        let mut enc = BundleEncoder::with_extras(
            cx.outline,
            &cx.types,
            reached,
            state,
            stamps,
            &extra_rodata,
            &extra_vtables,
        );
        let mut func_cache: HashMap<usize, Option<Vec<u8>>> = HashMap::default();
        let mut flushed = 0usize;
        for (pair, entry) in compiled {
            let body_id = match &entry.template {
                Some(t) => {
                    let Some(id) = encode_template(&mut enc, &mut func_cache, t) else {
                        continue;
                    };
                    id
                }
                None => NEG_BODY_ID,
            };
            let Some(no_unwind_count) = u32::try_from(entry.no_unwind.len()).ok() else {
                continue;
            };
            let Some(no_unwind_bytes) = encode_no_unwind(&mut enc, &entry.no_unwind) else {
                continue;
            };
            let inlined = convert_inlined_pairs(cx.outline, stamps, &entry.inlined);
            let payload = pack_payload(
                &entry.key,
                entry.footprint,
                body_id,
                &inlined,
                no_unwind_count,
                &no_unwind_bytes,
            );
            if enc.add_raw_record(pair, &payload).is_some() {
                flushed += 1;
            }
        }
        let mut code_remap: HashMap<u32, Option<u32>> = HashMap::default();
        for (pair, mut head, no_unwind, body) in carried_blobs {
            let Some(no_unwind_count) = u32::try_from(no_unwind.len()).ok() else {
                continue;
            };
            let Some(no_unwind_bytes) = encode_no_unwind(&mut enc, &no_unwind) else {
                continue;
            };
            let new_body_id = match body {
                None => NEG_BODY_ID,
                Some((old_id, returns, height, size, form, plan, func_bytes)) => {
                    let id_opt = *code_remap.entry(old_id).or_insert_with(|| {
                        let new_plan_bytes = encode_reloc_plan(&mut enc, &plan)?;
                        let packed = pack_inline_blob(
                            returns,
                            height,
                            size,
                            form,
                            &new_plan_bytes,
                            func_bytes,
                        )?;
                        enc.push_code_bytes((1usize << 62) | (old_id as usize), &packed)
                    });
                    let Some(id) = id_opt else {
                        continue;
                    };
                    id
                }
            };
            head[48..52].copy_from_slice(&new_body_id.to_le_bytes());
            head[56..60].copy_from_slice(&no_unwind_count.to_le_bytes());
            head.extend_from_slice(&no_unwind_bytes);
            if enc.add_raw_record(pair, &head).is_some() {
                flushed += 1;
            }
        }
        let sections = enc.finish();
        drop(extra_rodata);
        drop(extra_vtables);
        drop(self.view.backing.take());
        if let Some(sections) = sections
            && write_bundle(&self.path, INL_MAGIC, INL_VERSION, config, &sections).is_ok()
            && let Some(dir) = self.path.parent()
        {
            prune_stale_bundles(dir, &self.path, "inl", MAX_BUNDLES - 1);
        }
        (flushed, started.elapsed().as_micros() as usize)
    }

    fn collect_carry_over<'a>(
        &'a self,
        seen_keys: &HashSet<(u64, u64)>,
        carry_slots: &CarrySlots,
        (cx, reached, file_index, stamps, local_stamps): (
            &mut Context<'_>,
            &Reached,
            &OnceLock<LocalFileIndex>,
            &Stamps,
            &HashMap<u64, u64>,
        ),
        out: &mut Vec<CarriedEntry<'a>>,
    ) {
        let Some(backing) = self.view.backing.as_ref() else {
            return;
        };
        let data = backing.as_slice();
        let old_tables = self
            .view
            .shared_tables(cx.outline, &cx.types, file_index, stamps, None);
        let mut decoded: HashMap<u32, Option<RelocPlan>> = HashMap::default();
        for i in 0..self.view.header.records_count {
            let rec = self.view.header.records_off + i * RECORD_LEN;
            let key_lo = read_u64_le(data, rec);
            let key_hi = read_u64_le(data, rec + 8);
            if seen_keys.contains(&(key_hi, key_lo)) {
                continue;
            }
            let off = read_u32_le(data, rec + 16);
            let len = read_u32_le(data, rec + 20);
            let Some(payload) = self
                .view
                .blob_slice(off, len)
                .filter(|p| p.len() >= INL_HEADER_LEN)
            else {
                continue;
            };
            let body_id = read_u32_le(payload, 48);
            let inlined_count = read_u32_le(payload, 52) as usize;
            let no_unwind_count = read_u32_le(payload, 56) as usize;
            let Some(inlined_end) = INL_HEADER_LEN.checked_add(inlined_count.saturating_mul(16))
            else {
                continue;
            };
            if payload.len() < inlined_end {
                continue;
            }
            if !(0..inlined_count).all(|j| {
                let at = INL_HEADER_LEN + j * 16;
                let f_key = read_u64_le(payload, at);
                let f_stamp = read_u64_le(payload, at + 8);
                local_stamps.get(&f_key).copied() == Some(f_stamp)
            }) {
                continue;
            }
            let no_unwind = if no_unwind_count == 0 {
                if payload.len() != inlined_end {
                    continue;
                }
                Vec::new()
            } else {
                let Some(tables) = old_tables else {
                    continue;
                };
                let Some(tail) = payload.get(inlined_end..) else {
                    continue;
                };
                let Some(entries) =
                    decode_valid_no_unwind(tables, cx, reached, tail, no_unwind_count)
                else {
                    continue;
                };
                entries
            };
            if body_id == NEG_BODY_ID {
                out.push((
                    (key_hi, key_lo),
                    payload[..inlined_end].to_vec(),
                    no_unwind,
                    None,
                ));
                continue;
            }
            let Some(tables) = old_tables else {
                continue;
            };
            let Some(span) = read_code_span(data, &self.view.header, body_id as usize) else {
                continue;
            };
            let Some(blob) = self.view.blob_slice(span.code_off, span.code_len) else {
                continue;
            };
            let Some((returns, height, size, form, plan_bytes, func_bytes)) =
                split_inline_blob(blob)
            else {
                continue;
            };
            let plan = decoded
                .entry(body_id)
                .or_insert_with(|| {
                    decode_reloc_plan(
                        tables,
                        &cx.types,
                        reached,
                        carry_slots,
                        plan_bytes,
                        |id| {
                            self.view
                                .decode_rodata(tables, carry_slots, &cx.types, reached, id)
                        },
                        |id| {
                            self.view
                                .decode_vtable(tables, carry_slots, &cx.types, reached, id)
                        },
                    )
                })
                .clone();
            let Some(plan) = plan else {
                continue;
            };
            out.push((
                (key_hi, key_lo),
                payload[..inlined_end].to_vec(),
                no_unwind,
                Some((body_id, returns, height, size, form, plan, func_bytes)),
            ));
        }
    }
}

type CarriedBody<'a> = (u32, bool, u8, u16, Form, RelocPlan, &'a [u8]);
type CarriedEntry<'a> = (
    (u64, u64),
    Vec<u8>,
    Vec<(ItemRef, Substs)>,
    Option<CarriedBody<'a>>,
);

#[derive(Default)]
struct CarrySlots {
    rodata: Mutex<Vec<(Vec<u8>, Vec<RodataReloc>)>>,
    vtables: Mutex<Vec<Vec<VtableEntry>>>,
    empty_bodies: OnceLock<crate::lower::Bodies>,
}

impl Slots for CarrySlots {
    fn entry(&self, index: usize) -> i64 {
        index as i64
    }
    fn data(&self, _: &[u8]) -> i64 {
        0
    }
    fn zeroed(&self, _: usize) -> i64 {
        0
    }
    fn table(&self, _: &[Word]) -> i64 {
        0
    }
    fn cell(&self, _: usize, _: u32) -> i64 {
        0
    }
    fn bodies(&self) -> &crate::lower::Bodies {
        self.empty_bodies.get_or_init(Default::default)
    }
    fn rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        let mut g = lock(&self.rodata);
        let id = RodataId((g.len() as u32) | CARRY_BIT);
        g.push((bytes.to_vec(), relocs.to_vec()));
        id
    }
    fn vtable_id(&self, entries: &[VtableEntry]) -> VtableId {
        let mut g = lock(&self.vtables);
        let id = VtableId((g.len() as u32) | CARRY_BIT);
        g.push(entries.to_vec());
        id
    }
    fn vtable_entries(&self, id: VtableId) -> Option<Vec<VtableEntry>> {
        lock(&self.vtables)
            .get((id.0 & !CARRY_BIT) as usize)
            .cloned()
    }
}

pub(super) fn inline_bundle_path(aot_path: &Path) -> PathBuf {
    aot_path.with_extension("inl")
}

pub(super) fn hash_inline_128(
    config_digest: u64,
    item_key: u64,
    substs_hash: u64,
    stamps_digest: u64,
    extra_flags: u64,
) -> (u64, u64) {
    let mut hi = FxHasher::default();
    (
        config_digest,
        item_key,
        substs_hash,
        stamps_digest,
        extra_flags,
    )
        .hash(&mut hi);
    let mut lo = FxHasher::default();
    (
        0x517C_C1B7_2722_0A95_u64,
        extra_flags,
        stamps_digest,
        substs_hash,
        item_key,
        config_digest,
    )
        .hash(&mut lo);
    (hi.finish(), lo.finish())
}

fn convert_inlined_pairs(
    outline: &Outline,
    stamps: &Stamps,
    files: &[(u32, u32)],
) -> Vec<(u64, u64)> {
    let mut pairs: Vec<(u64, u64)> = files
        .iter()
        .filter_map(|&(k, f)| Some((stamps.file_key(outline, k, f)?, stamps.file(k, f)?)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn encode_no_unwind(enc: &mut BundleEncoder<'_>, entries: &[(ItemRef, Substs)]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(entries.len() * 12);
    for &(item, substs) in entries {
        let item_id = enc.item_id(item)?;
        let substs_ids = enc.substs_ids(substs)?;
        write_u32(&mut out, item_id);
        write_u32s(&mut out, &substs_ids);
    }
    Some(out)
}

fn decode_valid_no_unwind(
    tables: &SharedTables,
    cx: &mut Context<'_>,
    reached: &Reached,
    bytes: &[u8],
    count: usize,
) -> Option<Vec<(ItemRef, Substs)>> {
    if count == 0 {
        return bytes.is_empty().then(Vec::new);
    }
    let mut cur = Cursor::new(bytes);
    let mut out = Vec::with_capacity(count.min(256));
    for _ in 0..count {
        let item = tables.item_at(cur.u32().ok()?)?;
        let substs = tables.read_substs_checked(&cx.types, &mut cur)?;
        let idx = reached.index.get(&(item, substs)).copied()?;
        if reached.may_unwind(cx, idx) {
            return None;
        }
        out.push((item, substs));
    }
    (cur.pos == bytes.len()).then_some(out)
}

fn pack_payload(
    key: &InlKey,
    footprint: FootprintId,
    body_id: u32,
    inlined: &[(u64, u64)],
    no_unwind_count: u32,
    no_unwind_bytes: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(INL_HEADER_LEN + inlined.len() * 16 + no_unwind_bytes.len());
    write_u64(&mut out, key.item_key);
    write_u64(&mut out, key.substs_hash);
    write_u64(&mut out, key.stamps_digest);
    write_u64(&mut out, key.extra_flags);
    write_u64(&mut out, footprint.0 as u64);
    write_u64(&mut out, (footprint.0 >> 64) as u64);
    write_u32(&mut out, body_id);
    write_u32(&mut out, inlined.len() as u32);
    write_u32(&mut out, no_unwind_count);
    for &(f_key, f_stamp) in inlined {
        write_u64(&mut out, f_key);
        write_u64(&mut out, f_stamp);
    }
    out.extend_from_slice(no_unwind_bytes);
    out
}

fn encode_template(
    enc: &mut BundleEncoder<'_>,
    func_cache: &mut HashMap<usize, Option<Vec<u8>>>,
    template: &Arc<InlineTemplate>,
) -> Option<u32> {
    let ptr = Arc::as_ptr(template) as usize;
    let plan_bytes = encode_reloc_plan(enc, &template.reloc_plan)?;
    let func_ptr = Arc::as_ptr(&template.body) as usize;
    let func_bytes = func_cache
        .entry(func_ptr)
        .or_insert_with(|| template.body.encode())
        .as_deref()?;
    let blob = pack_inline_blob(
        template.returns,
        template.height,
        template.size,
        template.form,
        &plan_bytes,
        func_bytes,
    )?;
    enc.push_code_bytes(ptr, &blob)
}
