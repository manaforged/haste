//! Reading a source file again after an edit, in place when only function bodies or token places changed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

use super::Outline;
use super::item::ItemKind;
use super::tree::{File, read};

/// What reading a file again found.
#[derive(Debug, PartialEq, Eq)]
pub enum Edit {
    /// The file has the same tokens at the same places.
    Same,
    /// Only function bodies or token places changed; the outline now holds the new text in each of these `(crate, file)` slots.
    Bodies(Vec<(u32, u32)>),
    /// Tokens outside function bodies changed in a file of each of these crates, which must be read again.
    Items(Vec<u32>),
}

impl Outline {
    /// Reads the file at `path` again and swaps it in where the outline holds it when only function bodies changed.
    pub fn reread(&mut self, path: &Path) -> Result<Edit> {
        let slots = self.slots_of(path);
        let Some(&(krate, file)) = slots.first() else { return Ok(Edit::Same) };
        let old = self.crates[krate as usize].files[file as usize].clone();
        let new = read(path)?;
        if same_places(&old, &new) {
            return Ok(Edit::Same);
        }
        if old.expanded || !old.source.marks.is_empty() || outside_bodies(&old) != outside_bodies(&new) {
            let mut crates: Vec<u32> = slots.iter().map(|&(krate, _)| krate).collect();
            crates.dedup();
            return Ok(Edit::Items(crates));
        }
        let new = Arc::new(new);
        for &(krate, file) in &slots {
            self.crates[krate as usize].files[file as usize] = new.clone();
        }
        Ok(Edit::Bodies(slots))
    }

    /// Reads the file at `path` again in place, dropping the macro expansions of its bodies, as an edit to a file it includes needs; returns each place that holds it.
    pub fn reexpand(&mut self, path: &Path) -> Result<Vec<(u32, u32)>> {
        let slots = self.slots_of(path);
        let new = Arc::new(read(path)?);
        for &(krate, file) in &slots {
            self.crates[krate as usize].files[file as usize] = new.clone();
        }
        Ok(slots)
    }

    /// Each file that an `include_str!` or `include_bytes!` call with a literal path in crate `krate` reads, with the file that holds the call.
    pub fn included(&self, krate: usize) -> Vec<(PathBuf, PathBuf)> {
        let tree = &self.crates[krate];
        let mut found = Vec::new();
        for src in tree.files.iter().filter(|file| !file.expanded).map(|file| &file.source) {
            let calls = (2..src.tokens.len() as u32).filter(|&open| src.text(open) == "(" && src.text(open - 1) == "!" && matches!(src.text(open - 2), "include_str" | "include_bytes"));
            let paths = calls.filter_map(|open| super::build::path_text(src, *src.args(open).first()?, &tree.env));
            found.extend(paths.map(|path| (src.beside(&path), src.path.clone())));
        }
        found
    }

    /// The crate and file index of each place that holds the file at `path`.
    pub fn slots_of(&self, path: &Path) -> Vec<(u32, u32)> {
        let mut slots = Vec::new();
        for (krate, tree) in self.crates.iter().enumerate() {
            for (file, entry) in tree.files.iter().enumerate() {
                if entry.source.path == path {
                    slots.push((krate as u32, file as u32));
                }
            }
        }
        slots
    }
}

fn same_places(old: &File, new: &File) -> bool {
    let (a, b) = (&old.source, &new.source);
    a.tokens.len() == b.tokens.len() && (0..a.tokens.len() as u32).all(|at| a.text(at) == b.text(at) && a.tokens[at as usize].start == b.tokens[at as usize].start)
}

/// The tokens of `file` outside the bodies that callers cannot see: those of functions that are not `const` or `async` and name no `impl Trait`, whose body decides a value or type that callers use.
fn outside_bodies(file: &File) -> Vec<&str> {
    let src = &file.source;
    let mut skipped = vec![false; src.tokens.len()];
    for item in file.items.items.iter().filter(|item| item.kind == ItemKind::Fn) {
        if let Some(open) = item.body
            && (item.start..open).all(|at| !matches!(src.text(at), "const" | "async" | "impl"))
        {
            skipped[open as usize + 1..src.pair(open) as usize].fill(true);
        }
    }
    (0..src.tokens.len() as u32).filter(|at| !skipped[*at as usize]).map(|at| src.text(at)).collect()
}
