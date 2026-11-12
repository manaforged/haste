use std::collections::BTreeMap;
use std::hash::BuildHasher as _;
use std::path::Path;

use anyhow::{Context as _, Result};
use rustc_hash::FxBuildHasher;

use crate::graph::Workspace;
use crate::names::ItemRef;
use crate::outline::Outline;

use super::ident::item_path;

const FUNCTION: &str = "fn";
const REST: &str = "rest";
const FIELD: char = '\t';

/// The source of a build's local crates as prep saw it: a hash of each module-level function's text by its definition path, and a hash of all other text, which no edit may change for any prepped code to stay valid.
#[derive(PartialEq, Eq, Debug, Default)]
pub(super) struct Snapshot {
    functions: BTreeMap<String, u64>,
    rest: u64,
}

impl Snapshot {
    /// The snapshot of every local crate of `outline` now.
    pub(super) fn take(outline: &Outline, workspace: &Workspace) -> Snapshot {
        let mut snapshot = Snapshot::default();
        let mut rest = Vec::new();
        for krate in (0..outline.crates.len()).filter(|&krate| outline.package(workspace, krate).local()) {
            for (index, file) in outline.crates[krate].files.iter().enumerate() {
                let text = file.source.text.as_str();
                let mut cut = Vec::new();
                for at in 0..file.items.items.len() {
                    let item = ItemRef { krate: krate as u32, file: index as u32, item: at as u32 };
                    let Some((path, range)) = item_path(outline, item).zip(range(file, at)) else { continue };
                    if let std::collections::btree_map::Entry::Vacant(entry) = snapshot.functions.entry(path) {
                        entry.insert(FxBuildHasher.hash_one(&text[range.clone()]));
                        cut.push(range);
                    }
                }
                rest.push((file.source.path.clone(), outside(text, cut)));
            }
        }
        snapshot.rest = FxBuildHasher.hash_one(&rest);
        snapshot
    }

    /// The definition paths of the functions whose text differs from `earlier`, or none when text outside functions differs.
    pub(super) fn changed(&self, earlier: &Snapshot) -> Option<Vec<String>> {
        if self.rest != earlier.rest {
            return None;
        }
        let mut paths: Vec<String> = self.functions.iter().filter(|(path, hash)| earlier.functions.get(*path) != Some(hash)).map(|(path, _)| path.clone()).collect();
        paths.extend(earlier.functions.keys().filter(|path| !self.functions.contains_key(*path)).cloned());
        Some(paths)
    }

    /// Writes this snapshot to `path`.
    pub(super) fn write(&self, path: &Path) -> Result<()> {
        let mut text = format!("{REST}{FIELD}{:016x}\n", self.rest);
        for (name, hash) in &self.functions {
            text.push_str(&format!("{FUNCTION}{FIELD}{hash:016x}{FIELD}{name}\n"));
        }
        crate::graph::store::write_new(path, &text).with_context(|| format!("writing {}", path.display()))
    }

    /// The snapshot written at `path`.
    pub(super) fn read(path: &Path) -> Result<Snapshot> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut snapshot = Snapshot::default();
        for line in text.lines() {
            let mut fields = line.splitn(3, FIELD);
            let (word, hash) = (fields.next(), fields.next().and_then(|hash| u64::from_str_radix(hash, 16).ok()));
            match (word, hash, fields.next()) {
                (Some(REST), Some(hash), None) => snapshot.rest = hash,
                (Some(FUNCTION), Some(hash), Some(name)) => _ = snapshot.functions.insert(name.to_owned(), hash),
                _ => anyhow::bail!("{} holds a malformed line: {line}", path.display()),
            }
        }
        Ok(snapshot)
    }
}

/// The byte range of item `at` of `file` in its own text, from its first attribute to its last token, whose index is one below the item's end.
fn range(file: &crate::outline::tree::File, at: usize) -> Option<std::ops::Range<usize>> {
    let item = &file.items.items[at];
    let last = item.end.checked_sub(1).filter(|&last| last >= item.lead)?;
    let (text, span) = (file.source.text.as_str(), file.source.span(item.lead, last));
    let start = (span.as_ptr() as usize).checked_sub(text.as_ptr() as usize)?;
    (start + span.len() <= text.len()).then_some(start..start + span.len())
}

/// `text` without the byte ranges `cut`.
fn outside(text: &str, mut cut: Vec<std::ops::Range<usize>>) -> String {
    cut.sort_by_key(|range| range.start);
    let (mut kept, mut from) = (String::with_capacity(text.len()), 0);
    for range in cut {
        kept.push_str(&text[from.min(range.start)..range.start]);
        from = from.max(range.end);
    }
    kept.push_str(&text[from.min(text.len())..]);
    kept
}
