//! Reading a source file again after an edit, in place when only function bodies or token places changed.

use std::hash::Hash as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashMap;
use rustc_lexer::TokenKind;

use super::Outline;
use super::item::{Attr, Import, Item, ItemKind, Items};
use super::source::{Mark, Origins, Source, Token};
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
        let (slots, crates) = self.places_of(path);
        if crates.is_empty() {
            return Ok(Edit::Same);
        }
        let Some(&(krate, file)) = slots.first() else {
            return Ok(Edit::Items(crates));
        };
        let old = self.crates[krate as usize].files[file as usize].clone();
        let new = read(path)?;
        if same_places(&old, &new) {
            return Ok(Edit::Same);
        }
        if !old.source.marks.is_empty() || outside_bodies(&old) != outside_bodies(&new) {
            return Ok(Edit::Items(crates));
        }
        let new = Arc::new(new);
        for &(krate, file) in &slots {
            self.crates[krate as usize].files[file as usize] = new.clone();
            self.crates[krate as usize].cached_included = None;
        }
        Ok(Edit::Bodies(slots))
    }

    /// Refreshes the file at `path` after a file its `include_str!` or `include_bytes!` calls read changed.
    pub fn reexpand(&mut self, path: &Path) -> Result<Edit> {
        let (slots, crates) = self.places_of(path);
        if crates.is_empty() {
            return Ok(Edit::Same);
        }
        let Some(&(krate, file)) = slots.first() else {
            return Ok(Edit::Items(crates));
        };
        let expanded_include = self.crates.iter().enumerate().any(|(ck, tree)| {
            let mut scan = IncludeScan::new(self, ck);
            tree.files.iter().any(|entry| {
                entry.expanded
                    && entry.source.path == path
                    && !scan.calls(&entry.source, None, 0).is_empty()
            })
        });
        let old = &self.crates[krate as usize].files[file as usize];
        let skipped = body_skipped(old);
        let mut scan = IncludeScan::new(self, krate as usize);
        let outside_include = (0..old.source.tokens.len() as u32).any(|at| {
            !skipped[at as usize]
                && old.source.kind(at) == TokenKind::Ident
                && matches!(
                    old.source.text(at),
                    "include" | "include_str" | "include_bytes"
                )
        }) || !scan.calls(&old.source, Some(&skipped), 0).is_empty();
        if expanded_include || !old.source.marks.is_empty() || outside_include {
            return Ok(Edit::Items(crates));
        }
        let new = Arc::new(read(path)?);
        for &(krate, file) in &slots {
            self.crates[krate as usize].files[file as usize] = new.clone();
            self.crates[krate as usize].cached_included = None;
        }
        Ok(Edit::Bodies(slots))
    }

    /// Each file that an `include!`, `include_str!` or `include_bytes!` call (including macro-generated calls) in crate `krate` reads, with the file that holds the call.
    pub fn included(&self, krate: usize) -> Vec<(PathBuf, PathBuf)> {
        let tree = &self.crates[krate];
        if let Some(cached) = &tree.cached_included {
            return cached.clone();
        }
        let mut scan = IncludeScan::new(self, krate);
        let mut out = Vec::new();
        for file in &tree.files {
            for path in scan.calls(&file.source, None, 0) {
                out.push((file.source.beside(&path), file.source.path.clone()));
            }
        }
        out
    }

    fn places_of(&self, path: &Path) -> (Vec<(u32, u32)>, Vec<u32>) {
        let (mut slots, mut crates) = (Vec::new(), Vec::new());
        for (krate, tree) in self.crates.iter().enumerate() {
            for (file, entry) in tree
                .files
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.source.path == path)
            {
                if !entry.expanded {
                    slots.push((krate as u32, file as u32));
                }
                if crates.last() != Some(&(krate as u32)) {
                    crates.push(krate as u32);
                }
            }
        }
        (slots, crates)
    }
}

struct IncludeScan<'a> {
    outline: &'a Outline,
    krate: usize,
    defined: Option<FxHashMap<&'a str, &'a Arc<super::tree::Macro>>>,
    may_include: FxHashMap<*const File, bool>,
}

impl<'a> IncludeScan<'a> {
    fn new(outline: &'a Outline, krate: usize) -> Self {
        Self {
            outline,
            krate,
            defined: None,
            may_include: FxHashMap::default(),
        }
    }

    fn find_macro(&mut self, name: &str) -> Option<&'a Arc<super::tree::Macro>> {
        let tree = &self.outline.crates[self.krate];
        if let Some(mac) = tree.exports.get(name) {
            return Some(mac);
        }
        if !tree.defined.is_empty() {
            let defined = self.defined.get_or_insert_with(|| {
                let mut map = FxHashMap::default();
                for (&(f, i), mac) in &tree.defined {
                    if let Some(file) = tree.files.get(f as usize)
                        && let Some(ntok) = file.items.items.get(i as usize).and_then(|e| e.name)
                    {
                        map.entry(crate::names::plain(file.source.text(ntok)))
                            .or_insert(mac);
                    }
                }
                map
            });
            if let Some(&mac) = defined.get(name) {
                return Some(mac);
            }
        }
        self.outline.deps[self.krate]
            .iter()
            .find_map(|&(_, dep)| self.outline.crates[dep].exports.get(name))
    }

    fn file_may_include(&mut self, file: &Arc<File>) -> bool {
        let key = Arc::as_ptr(file);
        if let Some(&cached) = self.may_include.get(&key) {
            return cached;
        }
        let yes = ["include", "include_str", "include_bytes"]
            .iter()
            .any(|word| file.source.holds_word((0, file.source.len()), word));
        self.may_include.insert(key, yes);
        yes
    }

    fn resolve_path(
        &mut self,
        src: &Source,
        (from, to): (u32, u32),
        depth: usize,
    ) -> Option<String> {
        if depth >= 16 || from >= to {
            return None;
        }
        if let Some(plain) =
            super::build::path_text(src, (from, to), &self.outline.crates[self.krate].env)
        {
            return Some(plain);
        }
        if src.text(from) == "(" && src.pair(from) == to - 1 {
            return self.resolve_path(src, (from + 1, to - 1), depth + 1);
        }
        if to < from + 3 || src.kind(from + 1) != TokenKind::Bang || src.pair(from + 2) != to - 1 {
            return None;
        }
        let name = crate::names::plain(src.text(from));
        if name == "concat" {
            return src
                .args(from + 2)
                .into_iter()
                .map(|arg| self.resolve_path(src, arg, depth + 1))
                .collect();
        }
        let def = self.find_macro(name)?;
        let expanded = super::mbe::expand(
            &def.file.source,
            &def.rules,
            src,
            (from + 2, self.krate as u32),
            None,
        )
        .ok()?;
        self.resolve_path(&expanded, (0, expanded.len()), depth + 1)
    }

    fn calls(&mut self, src: &Source, skipped: Option<&[bool]>, depth: usize) -> Vec<String> {
        if depth >= 16 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for open in 2..src.tokens.len() as u32 {
            if src.kind(open - 1) != TokenKind::Bang || !super::source::opens(src.kind(open)) {
                continue;
            }
            if skipped.is_some_and(|mask| mask[(open - 2) as usize]) {
                continue;
            }
            let name = crate::names::plain(src.text(open - 2));
            if matches!(name, "include" | "include_str" | "include_bytes") {
                if let Some(arg) = src.args(open).first().copied()
                    && let Some(path) = self.resolve_path(src, arg, depth)
                {
                    out.push(path);
                }
                continue;
            }
            if let Some(def) = self.find_macro(name)
                && self.file_may_include(&def.file)
                && let Ok(expanded) = super::mbe::expand(
                    &def.file.source,
                    &def.rules,
                    src,
                    (open, self.krate as u32),
                    None,
                )
            {
                out.extend(self.calls(&expanded, None, depth + 1));
            }
        }
        out
    }
}

fn same_places(old: &File, new: &File) -> bool {
    let (a, b) = (&old.source, &new.source);
    a.tokens.len() == b.tokens.len()
        && (0..a.tokens.len() as u32).all(|at| {
            a.text(at) == b.text(at) && a.tokens[at as usize].start == b.tokens[at as usize].start
        })
}

fn body_skipped(file: &File) -> Vec<bool> {
    let src = &file.source;
    let mut skipped = vec![false; src.tokens.len()];
    for item in file
        .items
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::Fn)
    {
        if let Some(open) = item.body
            && (item.start..open).all(|at| !matches!(src.text(at), "const" | "async" | "impl"))
            && (item.lead..item.start).all(|at| src.text(at) != "define_opaque")
            && !src.holds_word((open, src.pair(open)), "impl")
        {
            skipped[open as usize + 1..src.pair(open) as usize].fill(true);
        }
    }
    skipped
}

/// The tokens of `file` outside the bodies that callers cannot see: those of functions that are not `const` or `async` and name no `impl Trait`, whose body decides a value or type that callers use.
fn outside_bodies(file: &File) -> Vec<&str> {
    let src = &file.source;
    let skipped = body_skipped(file);
    (0..src.tokens.len() as u32)
        .filter(|at| !skipped[*at as usize])
        .map(|at| src.text(at))
        .collect()
}

/// Hashes all tokens of `file` outside regular function bodies into `hasher`.
pub(crate) fn hash_outside_bodies(file: &File, hasher: &mut impl std::hash::Hasher) {
    let src = &file.source;
    let skipped = body_skipped(file);
    let mut count = 0usize;
    for at in (0..src.tokens.len() as u32).filter(|at| !skipped[*at as usize]) {
        src.text(at).hash(hasher);
        count += 1;
    }
    count.hash(hasher);
}

fn origins_identical(a: &Origins, b: &Origins) -> bool {
    match (a, b) {
        (Origins::Read, Origins::Read) => true,
        (Origins::Copied(x), Origins::Copied(y)) => x == y,
        _ => false,
    }
}

fn tokens_identical(a: &[Token], b: &[Token]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.kind == y.kind && x.start == y.start && x.raw() == y.raw())
}

fn items_vec_identical(a: &[Item], b: &[Item]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.kind == y.kind
                && x.name == y.name
                && x.vis == y.vis
                && x.attrs == y.attrs
                && x.lead == y.lead
                && x.start == y.start
                && x.end == y.end
                && x.body == y.body
                && x.parent == y.parent
        })
}

fn imports_identical(a: &[Import], b: &[Import]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.item == y.item && x.absolute == y.absolute && x.path == y.path && x.leaf == y.leaf
        })
}

fn items_identical(a: &Items, b: &Items) -> bool {
    a.inner == b.inner
        && a.marks == b.marks
        && items_vec_identical(&a.items, &b.items)
        && imports_identical(&a.imports, &b.imports)
}

fn base_identical(a: &Option<Arc<File>>, b: &Option<Arc<File>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => Arc::ptr_eq(x, y) || files_identical(x, y),
        _ => false,
    }
}

fn map_scope_id(
    a_to_b: &mut Vec<(u32, u32)>,
    b_to_a: &mut Vec<(u32, u32)>,
    a: u32,
    b: u32,
) -> bool {
    if a == 0 || b == 0 {
        return a == b;
    }
    if let Some(&(_, mapped)) = a_to_b.iter().find(|&&(k, _)| k == a) {
        return mapped == b;
    }
    if b_to_a.iter().any(|&(k, _)| k == b) {
        return false;
    }
    a_to_b.push((a, b));
    b_to_a.push((b, a));
    true
}

fn marks_equivalent(a: &[(u32, Mark)], b: &[(u32, Mark)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let (mut a_to_b, mut b_to_a) = (Vec::new(), Vec::new());
    a.iter().zip(b).all(|(&(pa, ma), &(pb, mb))| {
        pa == pb
            && match (ma, mb) {
                (Mark::Crate(x), Mark::Crate(y)) => x == y,
                (
                    Mark::Home {
                        krate: k1,
                        file: f1,
                        item: i1,
                    },
                    Mark::Home {
                        krate: k2,
                        file: f2,
                        item: i2,
                    },
                ) => (k1, f1, i1) == (k2, f2, i2),
                (
                    Mark::Scope {
                        id: ia,
                        parent: pa_id,
                    },
                    Mark::Scope {
                        id: ib,
                        parent: pb_id,
                    },
                ) => {
                    map_scope_id(&mut a_to_b, &mut b_to_a, ia, ib)
                        && map_scope_id(&mut a_to_b, &mut b_to_a, pa_id, pb_id)
                }
                _ => false,
            }
    })
}

/// Returns whether two parsed `File`s have identical tokens, spans, items, and origins.
pub(crate) fn files_identical(old_f: &File, new_f: &File) -> bool {
    let (os, ns) = (&old_f.source, &new_f.source);
    if old_f.expanded != new_f.expanded
        || os.path != ns.path
        || os.offset != ns.offset
        || os.text.len() != ns.text.len()
        || os.tokens.len() != ns.tokens.len()
        || old_f.items.items.len() != new_f.items.items.len()
        || old_f.items.imports.len() != new_f.items.imports.len()
    {
        return false;
    }
    if !old_f.expanded
        && (old_f.expand_origin != new_f.expand_origin
            || !origins_identical(&os.origins, &ns.origins))
    {
        return false;
    }
    marks_equivalent(&os.marks, &ns.marks)
        && os.opaque == ns.opaque
        && items_identical(&old_f.items, &new_f.items)
        && (old_f.expanded || base_identical(&os.base, &ns.base))
        && os.text.as_str() == ns.text.as_str()
        && tokens_identical(os.tokens.as_slice(), ns.tokens.as_slice())
}

fn attr_tokens_eq(old_src: &Source, a: &Attr, new_src: &Source, b: &Attr) -> bool {
    match (a, b) {
        (Attr::Cfg(x), Attr::Cfg(y)) => x == y,
        (Attr::CfgAttr(cx, ax), Attr::CfgAttr(cy, ay)) => {
            cx == cy
                && ax.len() == ay.len()
                && ax
                    .iter()
                    .zip(ay)
                    .all(|(x, y)| attr_tokens_eq(old_src, x, new_src, y))
        }
        (Attr::Path(x), Attr::Path(y)) => x == y,
        (Attr::MacroUse, Attr::MacroUse)
        | (Attr::NoStd, Attr::NoStd)
        | (Attr::NoCore, Attr::NoCore)
        | (Attr::NoImplicitPrelude, Attr::NoImplicitPrelude)
        | (Attr::Test, Attr::Test)
        | (Attr::Bench, Attr::Bench) => true,
        (Attr::MacroExport(x), Attr::MacroExport(y)) => x == y,
        (Attr::Derive(x), Attr::Derive(y)) | (Attr::Other(x), Attr::Other(y)) => {
            old_src.text(*x) == new_src.text(*y)
        }
        _ => false,
    }
}

/// Returns whether two attribute lists match token-for-token across sources.
pub(crate) fn attrs_tokens_eq(old_src: &Source, a: &[Attr], new_src: &Source, b: &[Attr]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| attr_tokens_eq(old_src, x, new_src, y))
}

fn token_slice_eq(
    old_src: &Source,
    old_start: u32,
    old_end: u32,
    new_src: &Source,
    new_start: u32,
    new_end: u32,
) -> bool {
    if old_end.saturating_sub(old_start) != new_end.saturating_sub(new_start) {
        return false;
    }
    (old_start..old_end)
        .zip(new_start..new_end)
        .all(|(a, b)| old_src.text(a) == new_src.text(b))
}

fn tokens_has_word(src: &Source, start: u32, end: u32, words: &[&str]) -> bool {
    (start..end).any(|at| words.contains(&src.text(at)))
}

/// Returns `(sig_changed, body_changed)` for corresponding items across two file versions.
pub(crate) fn item_tokens_diff(
    old_file: &File,
    old_item: &Item,
    new_file: &File,
    new_item: &Item,
) -> (bool, bool) {
    if old_item.kind != new_item.kind || old_item.parent != new_item.parent {
        return (true, true);
    }
    let (old_src, new_src) = (&old_file.source, &new_file.source);
    match old_item.kind {
        ItemKind::Fn => {
            if let (Some(o_open), Some(n_open)) = (old_item.body, new_item.body) {
                let is_opaque_or_special =
                    tokens_has_word(old_src, old_item.start, o_open, &["const", "async", "impl"])
                        || tokens_has_word(
                            old_src,
                            old_item.lead,
                            old_item.start,
                            &["define_opaque"],
                        )
                        || tokens_has_word(
                            new_src,
                            new_item.start,
                            n_open,
                            &["const", "async", "impl"],
                        )
                        || tokens_has_word(
                            new_src,
                            new_item.lead,
                            new_item.start,
                            &["define_opaque"],
                        );
                if is_opaque_or_special {
                    let all_same = token_slice_eq(
                        old_src,
                        old_item.lead,
                        old_item.end,
                        new_src,
                        new_item.lead,
                        new_item.end,
                    );
                    (!all_same, !all_same)
                } else {
                    let sig_same = token_slice_eq(
                        old_src,
                        old_item.lead,
                        o_open,
                        new_src,
                        new_item.lead,
                        n_open,
                    );
                    let body_same = token_slice_eq(
                        old_src,
                        o_open,
                        old_item.end,
                        new_src,
                        n_open,
                        new_item.end,
                    );
                    (!sig_same, !body_same)
                }
            } else {
                let all_same = token_slice_eq(
                    old_src,
                    old_item.lead,
                    old_item.end,
                    new_src,
                    new_item.lead,
                    new_item.end,
                );
                (!all_same, !all_same)
            }
        }
        ItemKind::Mod | ItemKind::Impl | ItemKind::Trait | ItemKind::ExternBlock => {
            match (old_item.body, new_item.body) {
                (Some(o_open), Some(n_open)) => {
                    let head_same = token_slice_eq(
                        old_src,
                        old_item.lead,
                        o_open + 1,
                        new_src,
                        new_item.lead,
                        n_open + 1,
                    );
                    (!head_same, !head_same)
                }
                (None, None) => {
                    let all_same = token_slice_eq(
                        old_src,
                        old_item.lead,
                        old_item.end,
                        new_src,
                        new_item.lead,
                        new_item.end,
                    );
                    (!all_same, !all_same)
                }
                _ => (true, true),
            }
        }
        _ => {
            let all_same = token_slice_eq(
                old_src,
                old_item.lead,
                old_item.end,
                new_src,
                new_item.lead,
                new_item.end,
            );
            (!all_same, !all_same)
        }
    }
}

/// Returns whether an `ItemKind` can affect global trait, impl, macro, or const resolution.
pub(crate) fn is_global_kind(kind: ItemKind) -> bool {
    matches!(
        kind,
        ItemKind::Impl
            | ItemKind::Trait
            | ItemKind::TraitAlias
            | ItemKind::Type
            | ItemKind::Const
            | ItemKind::MacroRules
            | ItemKind::Macro
            | ItemKind::MacroCall
    )
}

/// Returns whether item `item_idx` in `file` carries a crate-wide special marker attribute.
pub(crate) fn has_critical_mark(file: &File, item_idx: usize) -> bool {
    let bits = super::item::mark("lang")
        | super::item::mark("global_allocator")
        | super::item::mark("fundamental")
        | super::item::mark("repr")
        | super::item::mark("panic_handler");
    file.items
        .marks
        .iter()
        .any(|&(idx, mask)| idx as usize == item_idx && (mask & bits) != 0)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fresh_dir() -> TempDir {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("haste_edit_{}_{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        TempDir(dir)
    }

    #[test]
    fn body_impl_and_macro_generated_includes_invalidate_properly() {
        let dir = fresh_dir();
        let src_path = dir.0.join("lib.rs");
        let inc_a = dir.0.join("a.txt");
        let inc_b = dir.0.join("b.txt");
        let inc_top = dir.0.join("top.txt");
        std::fs::write(&inc_a, "alpha").expect("write a");
        std::fs::write(&inc_b, "beta").expect("write b");
        std::fs::write(&inc_top, "top").expect("write top");
        std::fs::write(
            &src_path,
            "pub trait Tr { fn f(&self) -> u8; }\npub struct S;\nmacro_rules! my_path { () => { \"a.txt\" }; }\nmacro_rules! gen_inc { ($p:expr) => { include_str!($p) }; }\nconst TOP: &str = gen_inc!(\"top.txt\");\npub fn holder() -> u8 {\n    let _x = include_str!(my_path!());\n    let _y = gen_inc!(\"b.txt\");\n    1\n}\n",
        )
        .expect("write lib.rs");

        let mut outline = Outline::file(&src_path, crate::graph::cfg::Target::default());
        outline.load_all().expect("load_all");
        let mut included = outline.included(0);
        included.sort();
        let mut expected = vec![
            (inc_a.clone(), src_path.clone()),
            (inc_b.clone(), src_path.clone()),
            (inc_top.clone(), src_path.clone()),
        ];
        expected.sort();
        assert_eq!(included, expected);

        std::fs::write(
            &src_path,
            "pub trait Tr { fn f(&self) -> u8; }\npub struct S;\nmacro_rules! my_path { () => { \"a.txt\" }; }\nmacro_rules! gen_inc { ($p:expr) => { include_str!($p) }; }\nconst TOP: &str = gen_inc!(\"top.txt\");\npub fn holder() -> u8 {\n    let _x = include_str!(my_path!());\n    let _y = gen_inc!(\"b.txt\");\n    impl Tr for S { fn f(&self) -> u8 { 2 } }\n    2\n}\n",
        )
        .expect("write impl in body");
        assert_eq!(
            outline.reread(&src_path).expect("reread"),
            Edit::Items(vec![0])
        );

        let body_only = "pub fn plain() -> u8 { 1 }\npub fn with_inc() -> &'static str { include_str!(\"b.txt\") }\nconst ITEM_TOP: &str = \"x\";\n";
        std::fs::write(&src_path, body_only).expect("write body_only");
        let mut body_outline = Outline::file(&src_path, crate::graph::cfg::Target::default());
        body_outline.load_all().expect("load body_only");
        assert_eq!(
            body_outline
                .reexpand(&src_path)
                .expect("reexpand body include"),
            Edit::Bodies(vec![(0, 0)])
        );

        let top_macro_inc = "macro_rules! gen_inc { ($p:expr) => { include_str!($p) }; }\nconst TOP: &str = gen_inc!(\"top.txt\");\n";
        std::fs::write(&src_path, top_macro_inc).expect("write top_macro_inc");
        let mut top_outline = Outline::file(&src_path, crate::graph::cfg::Target::default());
        top_outline.load_all().expect("load top_macro_inc");
        assert_eq!(
            top_outline
                .reexpand(&src_path)
                .expect("reexpand top macro include"),
            Edit::Items(vec![0])
        );

        let mut f1 = File::new(
            Source::parse(src_path.clone(), "pub fn inner() -> u64 { 10 }".into()).expect("f1"),
            Items::default(),
            true,
        );
        f1.source.marks = vec![
            (1, Mark::Scope { id: 100, parent: 0 }),
            (2, Mark::Scope { id: 100, parent: 0 }),
        ]
        .into();
        let mut f2 = File::new(
            Source::parse(src_path, "pub fn inner() -> u64 { 10 }".into()).expect("f2"),
            Items::default(),
            true,
        );
        f2.source.marks = vec![
            (1, Mark::Scope { id: 250, parent: 0 }),
            (2, Mark::Scope { id: 250, parent: 0 }),
        ]
        .into();
        assert!(files_identical(&f1, &f2));
        f2.source.marks = vec![
            (1, Mark::Scope { id: 250, parent: 0 }),
            (2, Mark::Scope { id: 251, parent: 0 }),
        ]
        .into();
        assert!(!files_identical(&f1, &f2));
    }
}
