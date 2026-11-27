use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::image::aot::{AotCode, AotState};
use crate::lower::{RodataId, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId, Word};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{Cursor, TyTable, read_ty, write_u32, write_u32s, write_u64};
use crate::sema::ty::{Kind, Ty, Types};

use super::codec::{Stamps, encode_code_blob, read_u32_le};
use super::relocs::{LocalFileIndex, LocalItemRecord, read_host, resolve_or_want, write_host};

pub(super) struct RodataEntrySpan {
    pub(super) data_off: u32,
    pub(super) data_len: u32,
    pub(super) reloc_off: u32,
    pub(super) reloc_count: u32,
}

pub(super) struct VtableEntrySpan {
    pub(super) entry_off: u32,
    pub(super) entry_count: u32,
}

pub(super) struct CodeEntrySpan {
    pub(super) code_off: u32,
    pub(super) code_len: u32,
}

pub(super) struct BundleSections {
    pub(super) items_bytes: Vec<u8>,
    pub(super) items_count: u32,
    pub(super) tys_bytes: Vec<u8>,
    pub(super) tys_count: u32,
    pub(super) rodatas: Vec<RodataEntrySpan>,
    pub(super) vtables: Vec<VtableEntrySpan>,
    pub(super) codes: Vec<CodeEntrySpan>,
    pub(super) records: Vec<(u64, u64, u32, u32)>,
    pub(super) blob: Vec<u8>,
}

pub(super) struct BundleEncoder<'a> {
    outline: &'a Outline,
    types: &'a Types,
    reached: &'a Reached,
    state: &'a AotState,
    stamps: &'a Stamps,
    item_map: HashMap<ItemRef, u32>,
    items: Vec<(LocalItemRecord, u32)>,
    tys: TyTable,
    rodata_map: HashMap<RodataId, u32>,
    rodata_visiting: HashSet<RodataId>,
    rodatas: Vec<RodataEntrySpan>,
    vtable_map: HashMap<VtableId, u32>,
    vtable_visiting: HashSet<VtableId>,
    vtables: Vec<VtableEntrySpan>,
    code_map: HashMap<usize, u32>,
    codes: Vec<CodeEntrySpan>,
    records: Vec<(u64, u64, u32, u32)>,
    blob: Vec<u8>,
}

impl<'a> BundleEncoder<'a> {
    pub(super) fn new(
        outline: &'a Outline,
        types: &'a Types,
        reached: &'a Reached,
        state: &'a AotState,
        stamps: &'a Stamps,
    ) -> Self {
        Self {
            outline,
            types,
            reached,
            state,
            stamps,
            item_map: HashMap::default(),
            items: Vec::new(),
            tys: TyTable::default(),
            rodata_map: HashMap::default(),
            rodata_visiting: HashSet::default(),
            rodatas: Vec::new(),
            vtable_map: HashMap::default(),
            vtable_visiting: HashSet::default(),
            vtables: Vec::new(),
            code_map: HashMap::default(),
            codes: Vec::new(),
            records: Vec::new(),
            blob: Vec::new(),
        }
    }

    fn push_blob(&mut self, bytes: &[u8]) -> Option<(u32, u32)> {
        let off = u32::try_from(self.blob.len()).ok()?;
        let len = u32::try_from(bytes.len()).ok()?;
        self.blob.extend_from_slice(bytes);
        Some((off, len))
    }

    fn item_id(&mut self, item: ItemRef) -> Option<u32> {
        intern_item(
            self.outline,
            self.stamps,
            &mut self.item_map,
            &mut self.items,
            item,
        )
    }

    fn substs_ids(&mut self, substs: Substs) -> Option<Vec<u32>> {
        let (outline, stamps, item_map, items) = (
            self.outline,
            self.stamps,
            &mut self.item_map,
            &mut self.items,
        );
        self.tys.substs_ids(self.types, substs, &mut |i| {
            intern_item(outline, stamps, item_map, items, i)
        })
    }

    fn reached_ref(&mut self, index: usize) -> Option<(u32, Vec<u32>)> {
        let ((item, substs), _) = self.reached.list.get(index)?;
        Some((self.item_id(*item)?, self.substs_ids(*substs)?))
    }

    fn code_id(&mut self, code: &Arc<AotCode>) -> Option<u32> {
        let ptr = Arc::as_ptr(code) as usize;
        if let Some(&id) = self.code_map.get(&ptr) {
            return Some(id);
        }
        let encoded = encode_code_blob(code)?;
        let (code_off, code_len) = self.push_blob(&encoded)?;
        let id = u32::try_from(self.codes.len()).ok()?;
        self.codes.push(CodeEntrySpan { code_off, code_len });
        self.code_map.insert(ptr, id);
        Some(id)
    }

    fn rodata_id(&mut self, id: RodataId) -> Option<u32> {
        if let Some(&idx) = self.rodata_map.get(&id) {
            return Some(idx);
        }
        if !self.rodata_visiting.insert(id) {
            return None;
        }
        let res = (|| {
            let (bytes, relocs) = self.state.rodata.get(id.0 as usize)?;
            let mut reloc_buf = Vec::new();
            for reloc in relocs {
                write_u32(&mut reloc_buf, reloc.offset);
                match reloc.target {
                    RodataTarget::Rodata(child) => {
                        reloc_buf.push(0);
                        write_u32(&mut reloc_buf, self.rodata_id(child)?);
                    }
                    RodataTarget::Fn(fn_idx) => {
                        reloc_buf.push(1);
                        let (item, substs) = self.reached_ref(fn_idx)?;
                        write_u32(&mut reloc_buf, item);
                        write_u32s(&mut reloc_buf, &substs);
                    }
                }
            }
            let (data_off, data_len) = self.push_blob(bytes)?;
            let (reloc_off, _) = self.push_blob(&reloc_buf)?;
            let reloc_count = u32::try_from(relocs.len()).ok()?;
            let idx = u32::try_from(self.rodatas.len()).ok()?;
            self.rodatas.push(RodataEntrySpan {
                data_off,
                data_len,
                reloc_off,
                reloc_count,
            });
            self.rodata_map.insert(id, idx);
            Some(idx)
        })();
        self.rodata_visiting.remove(&id);
        res
    }

    fn vtable_id(&mut self, id: VtableId) -> Option<u32> {
        if let Some(&idx) = self.vtable_map.get(&id) {
            return Some(idx);
        }
        if !self.vtable_visiting.insert(id) {
            return None;
        }
        let res = (|| {
            let entries = self.state.vtables.get(id.0 as usize)?;
            let mut entry_buf = Vec::new();
            for entry in entries {
                match *entry {
                    VtableEntry::Word(Word::Int(val)) => {
                        entry_buf.push(0);
                        write_u64(&mut entry_buf, val);
                    }
                    VtableEntry::Word(Word::Code(fn_idx)) => {
                        entry_buf.push(1);
                        let (item, substs) = self.reached_ref(fn_idx)?;
                        write_u32(&mut entry_buf, item);
                        write_u32s(&mut entry_buf, &substs);
                    }
                    VtableEntry::Super(super_id) => {
                        entry_buf.push(2);
                        write_u32(&mut entry_buf, self.vtable_id(super_id)?);
                    }
                }
            }
            let (entry_off, _) = self.push_blob(&entry_buf)?;
            let entry_count = u32::try_from(entries.len()).ok()?;
            let idx = u32::try_from(self.vtables.len()).ok()?;
            self.vtables.push(VtableEntrySpan {
                entry_off,
                entry_count,
            });
            self.vtable_map.insert(id, idx);
            Some(idx)
        })();
        self.vtable_visiting.remove(&id);
        res
    }

    fn write_symbol(&mut self, out: &mut Vec<u8>, src: &SymbolSource) -> Option<()> {
        match src {
            SymbolSource::Slot(index) => {
                out.push(0);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
            }
            SymbolSource::Vtable(id) => {
                out.push(1);
                write_u32(out, self.vtable_id(*id)?);
            }
            SymbolSource::Rodata(id) => {
                out.push(2);
                write_u32(out, self.rodata_id(*id)?);
            }
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
            SymbolSource::Host(host) => {
                out.push(5);
                write_host(out, host)?;
            }
            SymbolSource::Word(word) => {
                out.push(6);
                write_u64(out, *word);
            }
            SymbolSource::ThreadSlot(index) => {
                out.push(7);
                let (item, substs) = self.reached_ref(*index)?;
                write_u32(out, item);
                write_u32s(out, &substs);
            }
        }
        Some(())
    }

    pub(super) fn add_instance(
        &mut self,
        key: &super::codec::CacheInstanceKey,
        is_dict: bool,
        code: &Arc<AotCode>,
        relocs: &[SymbolSource],
        inlined: &[(u64, u64)],
    ) -> Option<()> {
        let code_id = self.code_id(code)?;
        let mut payload = Vec::with_capacity(40 + inlined.len() * 16 + relocs.len() * 12);
        write_u64(&mut payload, key.item_key);
        write_u64(&mut payload, key.substs_hash);
        write_u64(&mut payload, key.stamps_digest);
        let flags = key.flags | (u32::from(is_dict) << 2);
        write_u32(&mut payload, flags);
        write_u32(&mut payload, code_id);
        let reloc_count = u32::try_from(relocs.len()).ok()?;
        write_u32(&mut payload, reloc_count);
        let inlined_count = u32::try_from(inlined.len()).ok()?;
        write_u32(&mut payload, inlined_count);
        for &(f_key, f_stamp) in inlined {
            write_u64(&mut payload, f_key);
            write_u64(&mut payload, f_stamp);
        }
        for sym in relocs {
            self.write_symbol(&mut payload, sym)?;
        }
        let (payload_off, payload_len) = self.push_blob(&payload)?;
        self.records
            .push((key.key_hi, key.key_lo, payload_off, payload_len));
        Some(())
    }

    pub(super) fn finish(mut self) -> Option<BundleSections> {
        let mut items_bytes = Vec::with_capacity(self.items.len() * 24);
        for &(rec, sub) in &self.items {
            rec.write(&mut items_bytes);
            write_u32(&mut items_bytes, sub);
        }
        self.records.sort_unstable_by_key(|r| (r.0, r.1));
        self.records.dedup_by_key(|r| (r.0, r.1));
        Some(BundleSections {
            items_bytes,
            items_count: u32::try_from(self.items.len()).ok()?,
            tys_bytes: self.tys.bytes().to_vec(),
            tys_count: self.tys.count(),
            rodatas: self.rodatas,
            vtables: self.vtables,
            codes: self.codes,
            records: self.records,
            blob: self.blob,
        })
    }
}

fn intern_item(
    outline: &Outline,
    stamps: &Stamps,
    item_map: &mut HashMap<ItemRef, u32>,
    items: &mut Vec<(LocalItemRecord, u32)>,
    item: ItemRef,
) -> Option<u32> {
    if let Some(&idx) = item_map.get(&item) {
        return Some(idx);
    }
    let (base, sub) = match item.outer() {
        Some((outer, sub)) => (outer, sub.checked_add(1)?),
        None => (item, 0),
    };
    let record = LocalItemRecord::of(outline, stamps, base)?;
    let idx = u32::try_from(items.len()).ok()?;
    items.push((record, sub));
    item_map.insert(item, idx);
    Some(idx)
}

pub(super) struct SharedTables {
    pub(super) items: Vec<ItemRef>,
    pub(super) item_valid: Vec<bool>,
    pub(super) tys: Vec<Ty>,
    pub(super) ty_valid: Vec<bool>,
}

const INVALID_ITEM: ItemRef = ItemRef {
    krate: u32::MAX,
    file: 0,
    item: 0,
};

impl SharedTables {
    pub(super) fn decode(
        data: &[u8],
        header: &super::codec::LoadedHeader,
        outline: &Outline,
        types: &Types,
        index: &LocalFileIndex,
        stamps: &Stamps,
    ) -> Result<Self> {
        let items_end = header.items_off + header.items_count * 24;
        let mut item_cur = Cursor::new(
            data.get(header.items_off..items_end)
                .context("items out of bounds")?,
        );
        let mut items = Vec::with_capacity(header.items_count);
        let mut item_valid = Vec::with_capacity(header.items_count);
        for _ in 0..header.items_count {
            let record = LocalItemRecord::read(&mut item_cur)?;
            let sub = item_cur.u32()?;
            let resolved = index.resolve(outline, stamps, record).and_then(|base| {
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
        let tys_end = header.tys_off + header.tys_len;
        let mut ty_cur = Cursor::new(
            data.get(header.tys_off..tys_end)
                .context("tys out of bounds")?,
        );
        let mut tys = Vec::with_capacity(header.tys_count);
        let mut ty_valid = Vec::with_capacity(header.tys_count);
        for _ in 0..header.tys_count {
            let ty = read_ty(&mut ty_cur, types, &items, &tys)?;
            let valid = is_ty_valid(types, ty);
            tys.push(ty);
            ty_valid.push(valid);
        }
        if ty_cur.pos != header.tys_len {
            bail!("trailing bytes in bundle type table");
        }
        Ok(Self {
            items,
            item_valid,
            tys,
            ty_valid,
        })
    }

    pub(super) fn item_at(&self, id: u32) -> Option<ItemRef> {
        let idx = id as usize;
        (*self.item_valid.get(idx)?).then(|| self.items[idx])
    }

    pub(super) fn substs_from_ids(&self, types: &Types, ids: &[u32]) -> Option<Substs> {
        let mut resolved = Vec::with_capacity(ids.len());
        for &id in ids {
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
        Some(types.intern_substs(&resolved))
    }

    pub(super) fn read_substs_checked(
        &self,
        types: &Types,
        cur: &mut Cursor<'_>,
    ) -> Option<Substs> {
        let ids = cur.u32s().ok()?;
        self.substs_from_ids(types, &ids)
    }

    pub(super) fn read_instance(
        &self,
        types: &Types,
        reached: &Reached,
        slots: &dyn Slots,
        cur: &mut Cursor<'_>,
    ) -> Option<usize> {
        let item = self.item_at(cur.u32().ok()?)?;
        let substs = self.read_substs_checked(types, cur)?;
        resolve_or_want(reached, slots, (item, substs))
    }
}

fn is_ty_valid(types: &Types, ty: Ty) -> bool {
    if ty == Ty::NONE {
        return true;
    }
    let mut stack = vec![ty];
    let mut seen = HashSet::default();
    while let Some(cur) = stack.pop() {
        if cur == Ty::NONE || !seen.insert(cur) {
            continue;
        }
        match *types.kind(cur) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return false,
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
            ref other => {
                for child in other.children(types) {
                    stack.push(child);
                }
            }
        }
    }
    true
}

pub(super) fn read_rodata_span(
    data: &[u8],
    header: &super::codec::LoadedHeader,
    id: usize,
) -> Option<RodataEntrySpan> {
    if id >= header.rodatas_count {
        return None;
    }
    let at = header.rodatas_off + id * 16;
    Some(RodataEntrySpan {
        data_off: read_u32_le(data, at),
        data_len: read_u32_le(data, at + 4),
        reloc_off: read_u32_le(data, at + 8),
        reloc_count: read_u32_le(data, at + 12),
    })
}

pub(super) fn read_vtable_span(
    data: &[u8],
    header: &super::codec::LoadedHeader,
    id: usize,
) -> Option<VtableEntrySpan> {
    if id >= header.vtables_count {
        return None;
    }
    let at = header.vtables_off + id * 8;
    Some(VtableEntrySpan {
        entry_off: read_u32_le(data, at),
        entry_count: read_u32_le(data, at + 4),
    })
}

pub(super) fn read_code_span(
    data: &[u8],
    header: &super::codec::LoadedHeader,
    id: usize,
) -> Option<CodeEntrySpan> {
    if id >= header.codes_count {
        return None;
    }
    let at = header.codes_off + id * 8;
    Some(CodeEntrySpan {
        code_off: read_u32_le(data, at),
        code_len: read_u32_le(data, at + 4),
    })
}

pub(super) fn decode_symbol_list(
    tables: &SharedTables,
    types: &Types,
    reached: &Reached,
    slots: &dyn Slots,
    cur: &mut Cursor<'_>,
    count: usize,
    mut decode_rodata: impl FnMut(u32) -> Option<RodataId>,
    mut decode_vtable: impl FnMut(u32) -> Option<VtableId>,
) -> Option<Vec<SymbolSource>> {
    let mut out = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let sym = match cur.u8().ok()? {
            0 => SymbolSource::Slot(tables.read_instance(types, reached, slots, cur)?),
            1 => SymbolSource::Vtable(decode_vtable(cur.u32().ok()?)?),
            2 => SymbolSource::Rodata(decode_rodata(cur.u32().ok()?)?),
            3 => {
                let index = tables.read_instance(types, reached, slots, cur)?;
                let size = cur.u32().ok()?;
                SymbolSource::StaticCell { index, size }
            }
            4 => {
                let owner = tables.read_instance(types, reached, slots, cur)?;
                let seq = cur.u32().ok()?;
                let size = cur.u32().ok()?;
                SymbolSource::PromotedCell { owner, seq, size }
            }
            5 => SymbolSource::Host(read_host(cur).ok()?),
            6 => SymbolSource::Word(cur.u64().ok()?),
            7 => SymbolSource::ThreadSlot(tables.read_instance(types, reached, slots, cur)?),
            _ => return None,
        };
        out.push(sym);
    }
    Some(out)
}
