use std::collections::{BTreeMap, BTreeSet};
use std::hash::BuildHasher as _;
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxBuildHasher;

use crate::graph::Workspace;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::outline::tree::File;

use super::ident::identity;
use super::key::{Place, clean};
use super::source::normalize;

const FUNCTION: &str = "fn";
const ITEM: &str = "item";
const REST: &str = "rest";
const FIELD: char = '\t';

/// The source of a build's local crates as prep saw it: each function by its identity with the hash of its text, where its name was and the byte range it spanned; where each other named item was; and a hash of all other text, which no edit may change for any prepped code to stay valid.
#[derive(PartialEq, Eq, Debug, Default)]
pub(super) struct Snapshot {
    functions: BTreeMap<String, Function>,
    items: BTreeMap<String, Place>,
    rest: u64,
}

/// One function of a snapshot: the hash of its text, where its name is, and the byte range of its text in its file.
#[derive(PartialEq, Eq, Debug, Clone)]
struct Function {
    hash: u64,
    place: Place,
    start: u32,
    end: u32,
}

/// What an edit since a snapshot changed: the byte ranges, in the snapshot's files, of the functions whose text differs, and the crates that hold them.
#[derive(Default)]
pub(super) struct Changed {
    pub ranges: Vec<(String, u32, u32)>,
    pub crates: BTreeSet<String>,
}

impl Snapshot {
    /// The snapshot of every local crate of `outline` now.
    pub(super) fn take(outline: &Outline, workspace: &Workspace) -> Snapshot {
        let mut snapshot = Snapshot::default();
        let mut rest = Vec::new();
        for krate in (0..outline.crates.len()).filter(|&krate| outline.package(workspace, krate).local()) {
            for (index, file) in outline.crates[krate].files.iter().enumerate() {
                let (functions, cut) = snapshot.file(outline, (krate, index), file);
                rest.push((file.source.path.clone(), outside(&file.source.text, cut), functions));
            }
        }
        snapshot.rest = FxBuildHasher.hash_one(&rest);
        snapshot
    }

    /// Adds the named items of `file` and returns the text of functions whose identity is not unique, which joins the rest, with the byte ranges of the functions it holds apart.
    fn file(&mut self, outline: &Outline, (krate, index): (usize, usize), file: &File) -> (Vec<String>, Vec<std::ops::Range<usize>>) {
        let (text, mut doubled, mut cut) = (file.source.text.as_str(), Vec::new(), Vec::new());
        let Some(name) = normalize(&file.source.path).filter(|_| clean(text)) else { return (vec![text.to_owned()], cut) };
        for at in 0..file.items.items.len() {
            let item = ItemRef { krate: krate as u32, file: index as u32, item: at as u32 };
            let Some((path, lo)) = identity(outline, item).zip(name_offset(file, at)) else { continue };
            let place = Place { file: name.clone(), lo };
            if file.items.items[at].kind != ItemKind::Fn {
                self.items.insert(path, place);
                continue;
            }
            let Some(range) = range(file, at) else { continue };
            let function = Function { hash: FxBuildHasher.hash_one(&text[range.clone()]), place, start: range.start as u32, end: range.end as u32 };
            if self.functions.insert(path.clone(), function).is_some() {
                doubled.push(path);
            }
            cut.push(range);
        }
        let doubled = doubled.into_iter().filter_map(|path| self.functions.remove(&path)).map(|function| text[function.start as usize..function.end as usize].to_owned());
        (doubled.collect(), cut)
    }

    /// What changed from `earlier` to this snapshot, or none when text outside functions differs.
    pub(super) fn changed(&self, earlier: &Snapshot) -> Option<Changed> {
        if self.rest != earlier.rest {
            return None;
        }
        let mut changed = Changed::default();
        for (path, function) in &earlier.functions {
            if self.functions.get(path).is_none_or(|now| now.hash != function.hash) {
                changed.ranges.push((function.place.file.clone(), function.start, function.end));
                changed.crates.insert(crate_of(path).to_owned());
            }
        }
        changed.crates.extend(self.functions.keys().filter(|path| !earlier.functions.contains_key(*path)).map(|path| crate_of(path).to_owned()));
        Some(changed)
    }

    /// Where the item named `identity` was when this snapshot was taken.
    pub(super) fn place(&self, identity: &str) -> Option<&Place> {
        self.functions.get(identity).map(|function| &function.place).or_else(|| self.items.get(identity))
    }

    /// Writes this snapshot to `path`.
    pub(super) fn write(&self, path: &Path) -> Result<()> {
        let mut text = format!("{REST}{FIELD}{:016x}\n", self.rest);
        for (name, function) in &self.functions {
            let Function { hash, place, start, end } = function;
            text.push_str(&format!("{FUNCTION}{FIELD}{hash:016x}{FIELD}{}{FIELD}{}{FIELD}{start}{FIELD}{end}{FIELD}{name}\n", place.file, place.lo));
        }
        for (name, place) in &self.items {
            text.push_str(&format!("{ITEM}{FIELD}{}{FIELD}{}{FIELD}{name}\n", place.file, place.lo));
        }
        crate::graph::store::write_new(path, &text).with_context(|| format!("writing {}", path.display()))
    }

    /// The snapshot written at `path`.
    pub(super) fn read(path: &Path) -> Result<Snapshot> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut snapshot = Snapshot::default();
        for line in text.lines() {
            if snapshot.line(line).is_none() {
                bail!("{} holds a malformed line: {line}", path.display());
            }
        }
        Ok(snapshot)
    }

    fn line(&mut self, line: &str) -> Option<()> {
        let fields: Vec<&str> = line.split(FIELD).collect();
        let number = |at: usize| fields.get(at)?.parse::<u32>().ok();
        match fields.as_slice() {
            [REST, hash] => self.rest = u64::from_str_radix(hash, 16).ok()?,
            [FUNCTION, hash, file, _, _, _, name] => {
                let place = Place { file: (*file).to_owned(), lo: number(3)? };
                let function = Function { hash: u64::from_str_radix(hash, 16).ok()?, place, start: number(4)?, end: number(5)? };
                self.functions.insert((*name).to_owned(), function);
            }
            [ITEM, file, _, name] => _ = self.items.insert((*name).to_owned(), Place { file: (*file).to_owned(), lo: number(2)? }),
            _ => return None,
        }
        Some(())
    }
}

/// Whether every local source file of `outline` still reads on disk as it did when `outline` was loaded.
pub(super) fn settled(outline: &Outline, workspace: &Workspace) -> Result<bool> {
    for krate in (0..outline.crates.len()).filter(|&krate| outline.package(workspace, krate).local()) {
        for file in outline.crates[krate].files.iter().filter(|file| !file.expanded) {
            let path = &file.source.path;
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            if Source::parse(path.clone(), text)?.text != file.source.text {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn crate_of(identity: &str) -> &str {
    identity.split("::").next().unwrap_or(identity)
}

/// The byte offset of the name of item `at` of `file` in its own text, when the name is in that text and not in a macro's output.
fn name_offset(file: &File, at: usize) -> Option<u32> {
    let name = file.items.items[at].name?;
    let (text, span) = (file.source.text.as_str(), file.source.span(name, name));
    let start = (span.as_ptr() as usize).checked_sub(text.as_ptr() as usize)?;
    (start + span.len() <= text.len()).then_some(start as u32)
}

/// The byte range of item `at` of `file` in its own text, from its first attribute to its last token, whose index is one below the item's end.
fn range(file: &File, at: usize) -> Option<std::ops::Range<usize>> {
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
