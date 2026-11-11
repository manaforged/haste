use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rustc_lexer::TokenKind;

use super::item::{Attr, Item, ItemKind};
use super::source::Source;
use super::tree::{Files, Crate, Decl, File, Held, Impl, Macro, Module, State, Textual, Waiting, textual};
use super::{derive, mbe, rules, scan};

/// The item kinds that an attribute macro can stand on.
const ATTRIBUTED: [ItemKind; 9] = [ItemKind::Fn, ItemKind::Struct, ItemKind::Enum, ItemKind::Union, ItemKind::Impl, ItemKind::Trait, ItemKind::Const, ItemKind::Static, ItemKind::Type];
/// Builtin attributes, which never name a macro; any other attribute is looked up as one.
const INERT: &[&str] = &[
    "allow", "warn", "deny", "forbid", "expect", "inline", "cold", "must_use", "doc", "repr", "test", "ignore", "should_panic", "deprecated", "non_exhaustive", "track_caller", "automatically_derived", "no_mangle", "export_name", "link_section", "link_name", "used", "target_feature", "stable", "unstable", "const_stable", "feature", "lang", "diagnostic", "rustfmt", "clippy",
];
/// Whether the attribute `name` is builtin, so it never names an attribute macro.
pub(super) fn inert(name: &str) -> bool {
    INERT.contains(&name) || name.starts_with("rustc_")
}
const BUILTIN: [&str; 6] = ["include", "global_asm", "compile_error", "concat", "env", "asm"];

/// What a macro lookup in a crate's exports found.
enum Found {
    Macro(Arc<Macro>),
    NotYet(usize),
    Missing,
}

/// Loads modules of one crate, which it holds while the other crates stay readable.
pub struct Loader<'a> {
    pub krate: usize,
    pub cache: &'a Files,
    pub others: &'a [Crate],
    pub deps: &'a [(String, usize)],
    pub out: &'a mut Crate,
    pub depth: usize,
    pub scope: Option<Arc<Textual>>,
}

impl Loader<'_> {
    /// Reads the declared module `module` and expands its item macros, loading its `#[macro_use]` children with it.
    pub fn load(&mut self, module: usize) -> Result<()> {
        let State::Declared(decl) = self.out.modules[module].state.clone() else { return Ok(()) };
        let (index, owner) = match decl {
            Decl::Inline { file, at } => (file, Some(at)),
            Decl::File { candidates, by_attr } => match self.open(module, &candidates, by_attr) {
                Some(index) => (index, None),
                None => return Ok(()),
            },
        };
        let entry = &mut self.out.modules[module];
        entry.state = State::Loaded;
        entry.file = index;
        let saved = std::mem::replace(&mut self.scope, entry.textual.clone());
        let file = self.out.files[index].clone();
        let result = self.children(&file, index, module, owner);
        self.out.modules[module].textual = std::mem::replace(&mut self.scope, saved);
        result
    }

    fn open(&mut self, module: usize, candidates: &[PathBuf], by_attr: bool) -> Option<usize> {
        let Some(found) = candidates.iter().find(|candidate| super::tree::exists(candidate)) else {
            self.out.missing.extend_from_slice(candidates);
            self.out.modules[module].state = State::Missing;
            return None;
        };
        let file = match self.cache.file(found) {
            Ok(file) => file,
            Err(error) => {
                self.out.errors.push(format!("{error:#}"));
                self.out.modules[module].state = State::Missing;
                return None;
            }
        };
        if !self.out.env.enabled(&file.items.inner) {
            self.out.gated.push(found.clone());
            self.out.modules[module].state = State::Gated;
            return None;
        }
        let entry = &mut self.out.modules[module];
        if entry.parent.is_some() {
            entry.dir = child_dir(found, by_attr, &entry.name);
        }
        self.out.files.push(file);
        Some(self.out.files.len() - 1)
    }

    fn children(&mut self, file: &Arc<File>, index: usize, module: usize, owner: Option<u32>) -> Result<()> {
        for (at, item) in file.items.items.iter().enumerate() {
            if item.parent != owner || !self.out.env.enabled(&item.attrs) {
                continue;
            }
            if self.out.consumed.contains(&(index as u32, at as u32)) {
                continue;
            }
            self.attributes(file, module, (at as u32, item));
            self.module_item(file, (index, module), (at as u32, item))?;
        }
        Ok(())
    }

    fn module_item(&mut self, file: &Arc<File>, (index, module): (usize, usize), (at, item): (u32, &Item)) -> Result<()> {
        match item.kind {
            ItemKind::Mod => self.child(file, index, module, (at, item))?,
            ItemKind::MacroRules | ItemKind::Macro => self.define(file, (index as u32, at), item),
            ItemKind::MacroCall => self.call(file, module, at)?,
            ItemKind::Impl => self.impl_block(index, module, at, file)?,
            ItemKind::Const if item.name.is_some_and(|name| file.source.text(name) == "_") => self.out.impl_bodies.push((index, at)),
            ItemKind::Fn if item.body.is_some_and(|open| file.source.holds_word((open, file.source.pair(open)), "impl")) => self.out.impl_bodies.push((index, at)),
            ItemKind::Struct | ItemKind::Enum | ItemKind::Union => self.derive(file, module, (at, item))?,
            _ => {}
        }
        Ok(())
    }

    fn attributes(&mut self, file: &Arc<File>, module: usize, (at, item): (u32, &Item)) {
        if !ATTRIBUTED.contains(&item.kind) {
            return;
        }
        let src = &file.source;
        for attr in &item.attrs {
            let Attr::Other(first) = *attr else { continue };
            let name = src.text(first);
            if inert(name) {
                continue;
            }
            let mut last = first;
            while src.kind(last + 1) == TokenKind::Colon && src.kind(last + 2) == TokenKind::Colon && src.kind(last + 3) == TokenKind::Ident {
                last += 3;
            }
            self.out.attributed.push(Held { file: file.clone(), owner: module, at, path: (first, last) });
        }
    }

    fn derive(&mut self, file: &Arc<File>, module: usize, (at, item): (u32, &Item)) -> Result<()> {
        let (expanded, foreign) = derive::derived(&file.source, item, &self.out.env, (self.krate, self.deps));
        self.out.derived.extend(foreign.into_iter().map(|path| Held { file: file.clone(), owner: module, at, path }));
        match expanded {
            Some(expanded) => self.splice(module, expanded, "derive"),
            None => Ok(()),
        }
    }

    fn impl_block(&mut self, index: usize, module: usize, at: u32, file: &Arc<File>) -> Result<()> {
        self.out.impls.push(Impl { file: index, at, module, parts: Vec::new() });
        self.members(file, self.out.impls.len() - 1, Some(at))
    }

    fn members(&mut self, file: &Arc<File>, block: usize, owner: Option<u32>) -> Result<()> {
        for (at, item) in file.items.items.iter().enumerate() {
            if item.parent == owner && item.kind == ItemKind::MacroCall && self.out.env.enabled(&item.attrs) {
                self.member_call(file, block, at as u32)?;
            }
        }
        Ok(())
    }

    fn member_call(&mut self, file: &Arc<File>, block: usize, at: u32) -> Result<()> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(_)) = (item.name, item.body) else { return Ok(()) };
        let name = crate::names::plain(file.source.text(name));
        let Some(definition) = self.lookup(file, item, name) else {
            self.out.members.push(self.waiting(file, block, at));
            return Ok(());
        };
        self.member_at(file, block, at, &definition)
    }

    /// Expands the waiting macro call `call` in an impl body with `definition`, in the textual scope it was written in.
    pub fn expand_member(&mut self, call: Waiting, definition: &Macro) -> Result<()> {
        self.scope = call.scope;
        self.member_at(&call.file, call.owner, call.at, definition)
    }

    fn member_at(&mut self, file: &Arc<File>, block: usize, at: u32, definition: &Macro) -> Result<()> {
        let Some((name, expanded)) = self.expand_call(file, at, definition) else { return Ok(()) };
        let Some(expanded) = self.scanned(expanded, name) else { return Ok(()) };
        self.out.impls[block].parts.push(self.out.files.len() - 1);
        self.depth += 1;
        let result = self.members(&expanded, block, None);
        self.depth -= 1;
        result
    }

    fn child(&mut self, file: &Arc<File>, index: usize, parent: usize, (at, item): (u32, &Item)) -> Result<()> {
        let name = crate::names::plain(file.source.text(item.name.unwrap_or(item.start))).to_owned();
        let base = &self.out.modules[parent];
        let path = self.out.env.path(&item.attrs);
        let (decl, dir) = if item.body.is_some() {
            (Decl::Inline { file: index, at }, base.dir.join(path.unwrap_or(&name)))
        } else {
            let candidates = match path {
                Some(path) if base.item.is_some() => vec![base.dir.join(path)],
                Some(path) => vec![file.source.beside(path)],
                None => vec![base.dir.join(format!("{name}.rs")), base.dir.join(&name).join("mod.rs")],
            };
            (Decl::File { candidates, by_attr: path.is_some() }, PathBuf::new())
        };
        let macro_use = item.attrs.contains(&Attr::MacroUse);
        let inline = item.body.map(|_| at);
        let child = Module { file: usize::MAX, item: inline, parent: Some(parent), name, dir, parts: Vec::new(), vis: item.vis, macro_use, state: State::Declared(decl), textual: self.scope.clone() };
        self.out.modules.push(child);
        if macro_use {
            let module = self.out.modules.len() - 1;
            self.load(module)?;
            self.scope = self.out.modules[module].textual.clone();
        }
        Ok(())
    }

    fn define(&mut self, file: &Arc<File>, place: (u32, u32), item: &Item) {
        let (Some(name), Some(open)) = (item.name, item.body) else { return };
        let builtin = item.attrs.iter().any(|attr| matches!(attr, Attr::Other(at) if file.source.text(*at) == "rustc_builtin_macro"));
        if builtin {
            return;
        }
        let parsed = if item.kind == ItemKind::Macro { rules::parse_macro(&file.source, open) } else { rules::parse(&file.source, open) };
        match parsed {
            Ok(rules) => {
                let rules = if item.attrs.contains(&Attr::MacroExport(true)) { rules::local_inner(&file.source, rules) } else { rules };
                let name = crate::names::plain(file.source.text(name)).to_owned();
                let definition = Arc::new(Macro { file: file.clone(), rules, krate: self.krate });
                self.out.defined.insert(place, definition.clone());
                let public = item.kind == ItemKind::Macro && item.vis != super::item::Vis::Private;
                if public || item.attrs.iter().any(|attr| matches!(attr, Attr::MacroExport(_))) {
                    self.out.exports.insert(name.clone(), definition.clone());
                }
                if item.kind == ItemKind::MacroRules {
                    self.scope = Some(Arc::new(Textual { name, def: definition, next: self.scope.take() }));
                }
            }
            Err(error) => self.out.errors.push(format!("{error:#}")),
        }
    }

    fn exported(&self, krate: usize, name: &str) -> Found {
        let holder = if krate == self.krate { &*self.out } else { &self.others[krate] };
        match holder.exports.get(name) {
            Some(found) => Found::Macro(found.clone()),
            None if holder.partial() || holder.modules.is_empty() => Found::NotYet(krate),
            None => Found::Missing,
        }
    }

    fn lookup(&mut self, file: &File, item: &Item, name: &str) -> Option<Arc<Macro>> {
        if name == "cfg_select" {
            return textual(self.scope.as_ref(), name).filter(|_| item.name == Some(item.start));
        }
        let crates = match self.qualified(file, item) {
            Some(crates) => crates?,
            None => {
                if let Some(found) = textual(self.scope.as_ref(), name) {
                    return Some(found);
                }
                self.out.macro_use.clone()
            }
        };
        for krate in crates {
            match self.exported(krate, name) {
                Found::Macro(found) => return Some(found),
                Found::NotYet(krate) => self.out.fallback.push(krate),
                Found::Missing => {}
            }
        }
        None
    }

    /// For a call through a path, the crates whose exports it names: `$crate::m!`, `crate::m!` or `dep::m!`. Longer paths resolve through names instead.
    fn qualified(&self, file: &File, item: &Item) -> Option<Option<Vec<usize>>> {
        let src = &file.source;
        let first = item.start;
        if src.kind(first) == TokenKind::Dollar {
            return Some((item.name == Some(first + 4)).then(|| src.dollar(first).map(|krate| vec![krate as usize])).flatten());
        }
        if item.name == Some(first) || src.kind(first) == TokenKind::Colon {
            return None;
        }
        if item.name != Some(first + 3) {
            return Some(None);
        }
        Some(match src.text(first) {
            "crate" | "self" | "super" => Some(vec![self.krate]),
            segment => self.deps.iter().find(|(dep, _)| dep == segment).map(|(_, krate)| vec![*krate]),
        })
    }

    /// Expands again the calls that found no definition, now that more modules are read.
    pub fn retry(&mut self) -> Result<()> {
        self.out.fallback.clear();
        for call in std::mem::take(&mut self.out.deferred) {
            self.scope = call.scope;
            self.call(&call.file, call.owner, call.at)?;
        }
        for call in std::mem::take(&mut self.out.members) {
            self.scope = call.scope;
            self.member_call(&call.file, call.owner, call.at)?;
        }
        self.scope = None;
        Ok(())
    }

    fn waiting(&self, file: &Arc<File>, owner: usize, at: u32) -> Waiting {
        Waiting { file: file.clone(), owner, at, scope: self.scope.clone() }
    }

    fn call(&mut self, file: &Arc<File>, module: usize, at: u32) -> Result<()> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(open)) = (item.name, item.body) else { return Ok(()) };
        let name = crate::names::plain(file.source.text(name));
        if name == "include" {
            return self.include(file, module, open);
        }
        if BUILTIN.contains(&name) {
            return Ok(());
        }
        let Some(definition) = self.lookup(file, item, name) else {
            if name == "cfg_select" {
                return self.cfg_select(file, module, open);
            }
            self.out.deferred.push(self.waiting(file, module, at));
            return Ok(());
        };
        self.expand_at(file, module, at, &definition)
    }

    /// Expands the waiting macro call `call` with `definition`, in the textual scope it was written in.
    pub fn expand(&mut self, call: Waiting, definition: &Macro) -> Result<()> {
        self.scope = call.scope;
        self.expand_at(&call.file, call.owner, call.at, definition)
    }

    /// Adds `source`, the output of procedural macro `name`, to module `module` in the module's textual scope.
    pub fn insert(&mut self, module: usize, source: Source, name: &str) -> Result<()> {
        self.scope = self.out.modules[module].textual.clone();
        let result = self.splice(module, Ok(source), name);
        self.scope = None;
        result
    }

    /// The name and expansion by `definition` of the macro call item `at` of `file`, or none when it has no body or recursed too deep.
    fn expand_call<'f>(&mut self, file: &'f Arc<File>, at: u32, definition: &Macro) -> Option<(&'f str, Result<Source>)> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(open)) = (item.name, item.body) else { return None };
        let name = crate::names::plain(file.source.text(name));
        if self.depth >= crate::RECURSION_LIMIT {
            self.out.errors.push(format!("{}: macro {name} recursed past {} levels", file.source.path.display(), crate::RECURSION_LIMIT));
            return None;
        }
        Some((name, mbe::expand(&definition.file.source, &definition.rules, &file.source, (open, definition.krate as u32), None)))
    }

    fn expand_at(&mut self, file: &Arc<File>, module: usize, at: u32, definition: &Macro) -> Result<()> {
        let Some((name, expanded)) = self.expand_call(file, at, definition) else { return Ok(()) };
        self.splice(module, expanded, name)
    }

    fn cfg_select(&mut self, file: &Arc<File>, module: usize, open: u32) -> Result<()> {
        let selected = match mbe::select(&file.source, open, &self.out.env) {
            Ok(Some(selected)) => selected,
            Ok(None) => return Ok(()),
            Err(error) => {
                self.out.errors.push(format!("cfg_select!: {error:#}"));
                return Ok(());
            }
        };
        self.splice(module, Ok(selected), "cfg_select")
    }

    fn include(&mut self, file: &Arc<File>, module: usize, open: u32) -> Result<()> {
        let src = &file.source;
        let path = path_text(src, (open + 1, src.pair(open)), &self.out.env).ok_or_else(|| anyhow::anyhow!("a path that is not a literal, `concat!` or `env!`"));
        let source = path.and_then(|path| Source::read(&src.beside(&path)));
        self.splice(module, source, "include")
    }

    fn scanned(&mut self, source: Result<Source>, name: &str) -> Option<Arc<File>> {
        let expanded = source.and_then(|source| scan::scan(&source).map(|items| File::new(source, items, true)));
        match expanded {
            Ok(expanded) => {
                let expanded = Arc::new(expanded);
                self.out.files.push(expanded.clone());
                Some(expanded)
            }
            Err(error) => {
                self.out.errors.push(format!("{name}!: {error:#}"));
                None
            }
        }
    }

    fn splice(&mut self, module: usize, source: Result<Source>, name: &str) -> Result<()> {
        let Some(expanded) = self.scanned(source, name) else { return Ok(()) };
        let index = self.out.files.len() - 1;
        self.out.modules[module].parts.push(index);
        self.out.expansions += 1;
        self.depth += 1;
        let result = self.children(&expanded, index, module, None);
        self.depth -= 1;
        result
    }
}

/// The text of the tokens from `from` up to `to` when they are a string literal or a `concat!`, `env!` or `stringify!` of them, inside the parentheses that a macro puts around an expression fragment.
pub(crate) fn path_text(src: &Source, (from, to): (u32, u32), env: &super::tree::Env) -> Option<String> {
    if to == from + 1 {
        return src.literal_text(from).ok().flatten();
    }
    if src.text(from) == "(" && src.pair(from) == to - 1 {
        return path_text(src, (from + 1, to - 1), env);
    }
    if to < from + 3 || src.text(from + 1) != "!" {
        return None;
    }
    let parts = src.args(from + 2);
    match src.text(from) {
        "concat" => parts.into_iter().map(|range| path_text(src, range, env)).collect(),
        "env" => env.var(&path_text(src, *parts.first()?, env)?),
        "stringify" => Some(src.spelled((from + 3, src.pair(from + 2)))),
        _ => None,
    }
}

fn child_dir(found: &Path, by_attr: bool, name: &str) -> PathBuf {
    let parent = found.parent().unwrap_or(Path::new("")).to_owned();
    let mod_rs = found.file_name().is_some_and(|file| file == "mod.rs");
    if by_attr || mod_rs { parent } else { parent.join(name) }
}
