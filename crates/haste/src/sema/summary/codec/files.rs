//! Workspace-invariant records of fixed items and the file map that resolves them.

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;
use crate::outline::Outline;

use super::{Cursor, write_u32, write_u64};

/// Map from fixed file keys to the loaded `(crate, file)` they name in this workspace.
#[derive(Debug, Default)]
pub struct FileIndex {
    files: HashMap<u64, (u32, u32)>,
}

impl FileIndex {
    /// Builds the map over every loaded file of every fixed crate in `outline`.
    pub fn build(outline: &Outline) -> Self {
        let mut files = HashMap::default();
        for (krate, tree) in outline
            .crates
            .iter()
            .enumerate()
            .filter(|(_, tree)| tree.env.fixed)
        {
            for file in 0..tree.files.len() as u32 {
                if let Some(key) = crate::substrate::file_key(outline, krate as u32, file) {
                    files.entry(key).or_insert((krate as u32, file));
                }
            }
        }
        Self { files }
    }

    /// Builds a `FileIndex` directly from pre-validated `(file_key -> (krate, file))` pairs.
    pub(crate) fn from_files(files: HashMap<u64, (u32, u32)>) -> Self {
        Self { files }
    }

    /// Iterates every `(file_key, (krate, file))` entry in this index.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u64, &(u32, u32))> {
        self.files.iter()
    }

    /// Resolves `record` to its `ItemRef` in this workspace, or `None` when it is body-local, its file is absent or `key_of` disagrees with its key.
    pub(crate) fn resolve(
        &self,
        outline: &Outline,
        record: ItemRecord,
        key_of: impl FnOnce(&Outline, ItemRef) -> Option<u64>,
    ) -> Option<ItemRef> {
        let &(krate, file) = self.files.get(&record.file)?;
        let item = ItemRef {
            krate,
            file,
            item: record.item,
        };
        (item.outer().is_none() && key_of(outline, item)? == record.key).then_some(item)
    }
}

/// A fixed item as written to disk: its file key, its index in that file, and its full key as a check.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ItemRecord {
    file: u64,
    item: u32,
    key: u64,
}

impl ItemRecord {
    /// Bytes one record takes on disk.
    pub(crate) const LEN: usize = 20;

    /// The record of `item`, or `None` when `key_of` or its file key gives none.
    pub(crate) fn of(
        outline: &Outline,
        item: ItemRef,
        key_of: impl FnOnce(&Outline, ItemRef) -> Option<u64>,
    ) -> Option<Self> {
        let key = key_of(outline, item)?;
        let file = crate::substrate::file_key(outline, item.krate, item.file)?;
        Some(Self {
            file,
            item: item.item,
            key,
        })
    }

    /// Appends this record to `out`.
    pub(crate) fn write(self, out: &mut Vec<u8>) {
        write_u64(out, self.file);
        write_u32(out, self.item);
        write_u64(out, self.key);
    }

    /// Reads one record from `cur`.
    pub(crate) fn read(cur: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            file: cur.u64()?,
            item: cur.u32()?,
            key: cur.u64()?,
        })
    }
}

pub(super) fn write_items(out: &mut Vec<u8>, items: &[ItemRecord]) {
    write_u32(out, items.len() as u32);
    items.iter().for_each(|record| record.write(out));
}

pub(super) fn read_items(
    cur: &mut Cursor<'_>,
    outline: &Outline,
    index: &FileIndex,
) -> Result<Option<Vec<ItemRef>>> {
    let item_count = cur.u32()? as usize;
    let mut items = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let Some(item) = index.resolve(
            outline,
            ItemRecord::read(cur)?,
            crate::substrate::sysroot_item_key,
        ) else {
            return Ok(None);
        };
        items.push(item);
    }
    Ok(Some(items))
}
