use std::path::Path;
use std::sync::OnceLock;

use crate::lower::{RodataId, RodataReloc, RodataTarget, Slots, VtableEntry, VtableId, Word};
use crate::outline::Outline;
use crate::sema::reach::Reached;
use crate::sema::summary::codec::Cursor;
use crate::sema::ty::Types;
use crate::substrate::Backing;

use super::codec::{LoadedHeader, Stamps};
use super::relocs::LocalFileIndex;
use super::tables::{SharedTables, read_rodata_span, read_vtable_span};
use super::{AotConfigKey, open_bundle};

pub(super) struct BundleView {
    pub(super) backing: Option<Backing>,
    pub(super) header: LoadedHeader,
    tables: OnceLock<Option<SharedTables>>,
    rodatas: Box<[OnceLock<Option<RodataId>>]>,
    vtables: Box<[OnceLock<Option<VtableId>>]>,
}

impl BundleView {
    pub(super) fn open(path: &Path, magic: [u8; 8], version: u32, config: AotConfigKey) -> Self {
        let (backing, header) = match open_bundle(path, magic, version, config) {
            Some((b, h)) => (Some(b), h),
            None => (None, LoadedHeader::empty()),
        };
        let rodatas = (0..header.rodatas_count).map(|_| OnceLock::new()).collect();
        let vtables = (0..header.vtables_count).map(|_| OnceLock::new()).collect();
        Self {
            backing,
            header,
            tables: OnceLock::new(),
            rodatas,
            vtables,
        }
    }

    #[cfg(test)]
    pub(super) fn empty() -> Self {
        Self {
            backing: None,
            header: LoadedHeader::empty(),
            tables: OnceLock::new(),
            rodatas: Box::new([]),
            vtables: Box::new([]),
        }
    }

    pub(super) fn has_backing(&self) -> bool {
        self.backing.is_some()
    }

    pub(super) fn blob_slice(&self, off: u32, len: u32) -> Option<&[u8]> {
        let data = self.backing.as_ref()?.as_slice();
        let off = off as usize;
        let end = off.checked_add(len as usize)?;
        if end > self.header.blob_len {
            return None;
        }
        data.get(self.header.blob_off + off..self.header.blob_off + end)
    }

    pub(super) fn shared_tables(
        &self,
        outline: &Outline,
        types: &Types,
        file_index: &OnceLock<LocalFileIndex>,
        stamps: &Stamps,
        tables_err: Option<&OnceLock<String>>,
    ) -> Option<&SharedTables> {
        let data = self.backing.as_ref()?.as_slice();
        self.tables
            .get_or_init(|| {
                let idx = file_index.get_or_init(|| LocalFileIndex::build(outline));
                match SharedTables::decode(data, &self.header, outline, types, idx, stamps) {
                    Ok(t) => Some(t),
                    Err(err) => {
                        if let Some(slot) = tables_err {
                            drop(slot.set(err.to_string()));
                        }
                        None
                    }
                }
            })
            .as_ref()
    }

    pub(super) fn decode_rodata(
        &self,
        tables: &SharedTables,
        slots: &dyn Slots,
        types: &Types,
        reached: &Reached,
        id: u32,
    ) -> Option<RodataId> {
        let slot = self.rodatas.get(id as usize)?;
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
                            self.decode_rodata(tables, slots, types, reached, child_id)?,
                        )
                    }
                    1 => RodataTarget::Fn(tables.read_instance(types, reached, slots, &mut cur)?),
                    _ => return None,
                };
                relocs.push(RodataReloc { offset, target });
            }
            Some(slots.rodata(bytes, &relocs))
        })
    }

    pub(super) fn decode_vtable(
        &self,
        tables: &SharedTables,
        slots: &dyn Slots,
        types: &Types,
        reached: &Reached,
        id: u32,
    ) -> Option<VtableId> {
        let slot = self.vtables.get(id as usize)?;
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
                        tables.read_instance(types, reached, slots, &mut cur)?,
                    )),
                    2 => {
                        let super_id = cur.u32().ok()?;
                        if super_id >= id {
                            return None;
                        }
                        VtableEntry::Super(
                            self.decode_vtable(tables, slots, types, reached, super_id)?,
                        )
                    }
                    _ => return None,
                };
                entries.push(entry);
            }
            Some(slots.vtable_id(&entries))
        })
    }
}
