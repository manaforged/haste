use std::hash::{Hash as _, Hasher as _};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use rustc_hash::{FxHashSet as HashSet, FxHasher};

use crate::image::Image;
use crate::image::aot::{AotCode, AotFn, AotState};
use crate::lower::{RodataTarget, SymbolSource, VtableEntry, Word};
use crate::outline::Outline;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::Cursor;
use crate::sema::ty::{Kind, Ty, Types};

use super::codec::{
    CacheInstanceKey, decode_code_blob, find_record, read_u32_le, read_u64_le, write_bundle,
};
use super::relocs::{LocalFileIndex, relocs_unwind_digest};
use super::stamps::Stamps;
use super::tables::{BundleEncoder, decode_symbol_list, read_code_span};
use super::view::BundleView;
use super::{AotConfigKey, CACHE_MAGIC, CACHE_VERSION, RECORD_LEN, cache_dir};

fn tc_bundle_path(tc_key: u64, config_digest: u64) -> PathBuf {
    let mut h = FxHasher::default();
    (0xA077_C000_u64, config_digest, tc_key).hash(&mut h);
    cache_dir().join(format!("tc-{:016x}.aot", h.finish()))
}

pub(super) struct ToolchainAotLayer {
    path: PathBuf,
    view: BundleView,
    decoded_codes: Box<[OnceLock<Option<Arc<AotCode>>>]>,
}

impl ToolchainAotLayer {
    pub(super) fn open(tc_key: u64, config: AotConfigKey) -> Self {
        let path = tc_bundle_path(tc_key, config.config_digest);
        let view = BundleView::open(&path, CACHE_MAGIC, CACHE_VERSION, config);
        let decoded_codes = (0..view.header.codes_count)
            .map(|_| OnceLock::new())
            .collect();
        Self {
            path,
            view,
            decoded_codes,
        }
    }

    #[cfg(test)]
    pub(super) fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            view: BundleView::empty(),
            decoded_codes: Box::new([]),
        }
    }

    pub(super) fn has_backing(&self) -> bool {
        self.view.has_backing()
    }

    pub(super) fn record_count(&self) -> usize {
        self.view.header.records_count
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

    pub(super) fn lookup(
        &self,
        key: &CacheInstanceKey,
        image: &Image,
        cx: &mut Context<'_>,
        reached: &Reached,
        file_index: &OnceLock<LocalFileIndex>,
        stamps: &Stamps,
    ) -> Option<(AotFn, u32)> {
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
        let payload = self.view.blob_slice(off, len).filter(|p| p.len() >= 48)?;
        if read_u64_le(payload, 0) != key.item_key
            || read_u64_le(payload, 8) != key.substs_hash
            || read_u64_le(payload, 16) != key.stamps_digest
            || (read_u32_le(payload, 24) & 3) != key.flags
        {
            return None;
        }
        let is_dict = (read_u32_le(payload, 24) & 4) != 0;
        let code_id = read_u32_le(payload, 28);
        let reloc_count = read_u32_le(payload, 32) as usize;
        let inlined_count = read_u32_le(payload, 36) as usize;
        let unwind_digest = read_u64_le(payload, 40);
        if inlined_count != 0 {
            return None;
        }
        let code = self.decode_code(code_id)?;
        let relocs = if reloc_count == 0 {
            if payload.len() != 48 {
                return None;
            }
            Vec::new()
        } else {
            let tables = self
                .view
                .shared_tables(cx.outline, &cx.types, file_index, stamps, None)?;
            let mut cur = Cursor::new(payload.get(48..)?);
            let r = decode_symbol_list(
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
            )?;
            if 48 + cur.pos != payload.len() {
                return None;
            }
            r
        };
        if relocs_unwind_digest(cx, reached, &relocs) != unwind_digest {
            return None;
        }
        let len_u32 = code.bytes().len() as u32;
        let entry = if is_dict {
            AotFn::Dict { body: code, relocs }
        } else {
            AotFn::Direct { code, relocs }
        };
        Some((entry, len_u32))
    }

    pub(super) fn flush_pure_sysroot(
        &mut self,
        config: AotConfigKey,
        state: &AotState,
        cx: &Context<'_>,
        reached: &Reached,
        stamps: &Stamps,
        entries: &[(
            usize,
            &CacheInstanceKey,
            bool,
            &Arc<AotCode>,
            &[SymbolSource],
            &[(u64, u64)],
            u64,
        )],
    ) {
        let existing = self.record_count();
        let pure_count = entries
            .iter()
            .filter(|&&(idx, _, _, _, relocs, inlined, _)| {
                inlined.is_empty()
                    && is_pure_sysroot_entry(cx.outline, &cx.types, reached, state, idx, relocs)
            })
            .count();
        if pure_count == 0 || (existing > 0 && pure_count <= existing + 128) {
            return;
        }
        drop(self.view.backing.take());
        let mut enc = BundleEncoder::new(cx.outline, &cx.types, reached, state, stamps);
        for &(idx, key, is_dict, code, relocs, inlined, unwind_digest) in entries {
            if inlined.is_empty()
                && is_pure_sysroot_entry(cx.outline, &cx.types, reached, state, idx, relocs)
            {
                let _ = enc.add_instance(key, is_dict, code, relocs, &[], unwind_digest);
            }
        }
        if let Some(sections) = enc.finish() {
            let _ = write_bundle(&self.path, CACHE_MAGIC, CACHE_VERSION, config, &sections);
        }
    }
}

fn substs_pure_sysroot(outline: &Outline, types: &Types, substs: Substs) -> bool {
    if substs.is_empty() {
        return true;
    }
    let mut stack: Vec<Ty> = types.substs(substs).to_vec();
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

fn inst_pure_sysroot(outline: &Outline, types: &Types, reached: &Reached, idx: usize) -> bool {
    let Some(((item, substs), _)) = reached.list.get(idx) else {
        return false;
    };
    outline.is_sysroot(item.krate) && substs_pure_sysroot(outline, types, *substs)
}

fn rodata_pure_sysroot(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    state: &AotState,
    id: u32,
) -> bool {
    let Some((_, relocs)) = state.rodata.get(id as usize) else {
        return false;
    };
    relocs.iter().all(|r| match r.target {
        RodataTarget::Fn(idx) => inst_pure_sysroot(outline, types, reached, idx),
        RodataTarget::Rodata(child) => {
            child.0 < id && rodata_pure_sysroot(outline, types, reached, state, child.0)
        }
    })
}

fn vtable_pure_sysroot(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    state: &AotState,
    id: u32,
) -> bool {
    let Some(entries) = state.vtables.get(id as usize) else {
        return false;
    };
    entries.iter().all(|e| match *e {
        VtableEntry::Word(Word::Int(_)) => true,
        VtableEntry::Word(Word::Code(idx)) => inst_pure_sysroot(outline, types, reached, idx),
        VtableEntry::Super(sup) => {
            sup.0 < id && vtable_pure_sysroot(outline, types, reached, state, sup.0)
        }
    })
}

fn is_pure_sysroot_entry(
    outline: &Outline,
    types: &Types,
    reached: &Reached,
    state: &AotState,
    idx: usize,
    relocs: &[SymbolSource],
) -> bool {
    if !inst_pure_sysroot(outline, types, reached, idx) {
        return false;
    }
    relocs.iter().all(|sym| match *sym {
        SymbolSource::Slot(t)
        | SymbolSource::StaticCell { index: t, .. }
        | SymbolSource::PromotedCell { owner: t, .. }
        | SymbolSource::ThreadSlot(t) => inst_pure_sysroot(outline, types, reached, t),
        SymbolSource::Rodata(id) => rodata_pure_sysroot(outline, types, reached, state, id.0),
        SymbolSource::Vtable(id) => vtable_pure_sysroot(outline, types, reached, state, id.0),
        SymbolSource::Host(_) | SymbolSource::Word(_) => true,
    })
}
