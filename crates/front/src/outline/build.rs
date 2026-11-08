use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rustc_lexer::TokenKind;

use super::item::{Attr, Item, ItemKind};
use super::source::Source;
use super::tree::{Cache, Crate, Decl, File, Impl, Macro, Module, State, Textual, Truth, textual};
use super::{mbe, rules, scan};

const DEPTH: usize = 128;
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
    pub cache: &'a Cache,
    pub others: &'a [Crate],
    pub deps: &'a [(String, usize)],
    pub out: &'a mut Crate,
    pub demands: &'a mut Vec<usize>,
    pub depth: usize,
    pub scope: Option<Arc<Textual>>,
}

fn strip_raw(name: &str) -> &str {
    name.strip_prefix("r#").unwrap_or(name)
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
        let Some(found) = candidates.iter().find(|candidate| candidate.is_file()) else {
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
            match item.kind {
                ItemKind::Mod => self.child(file, index, module, (at as u32, item))?,
                ItemKind::MacroRules | ItemKind::Macro => self.define(file, (index as u32, at as u32), item),
                ItemKind::MacroCall => self.call(file, module, at as u32)?,
                ItemKind::Impl => self.impl_block(index, module, at as u32, file)?,
                _ => {}
            }
        }
        Ok(())
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
        let (Some(name), Some(open)) = (item.name, item.body) else { return Ok(()) };
        let name = strip_raw(file.source.text(name));
        let Some(definition) = self.lookup(file, item, name) else {
            self.out.members.push((file.clone(), block, at));
            return Ok(());
        };
        if self.depth >= DEPTH {
            self.out.errors.push(format!("{}: macro {name} recursed past {DEPTH} levels", file.source.path.display()));
            return Ok(());
        }
        let expanded = mbe::expand(&definition.file.source, &definition.rules, &file.source, (open, definition.krate as u32));
        let Some(expanded) = self.scanned(expanded, name) else { return Ok(()) };
        self.out.impls[block].parts.push(self.out.files.len() - 1);
        self.depth += 1;
        let result = self.members(&expanded, block, None);
        self.depth -= 1;
        result
    }

    fn child(&mut self, file: &Arc<File>, index: usize, parent: usize, (at, item): (u32, &Item)) -> Result<()> {
        let name = strip_raw(file.source.text(item.name.unwrap_or(item.start))).to_owned();
        let base = &self.out.modules[parent];
        let path = self.out.env.path(&item.attrs);
        let (decl, dir) = if item.body.is_some() {
            (Decl::Inline { file: index, at }, base.dir.join(path.unwrap_or(&name)))
        } else {
            let candidates = match path {
                Some(path) if base.item.is_some() => vec![base.dir.join(path)],
                Some(path) => vec![file.source.path.parent().unwrap_or(Path::new("")).join(path)],
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
                let name = strip_raw(file.source.text(name)).to_owned();
                let definition = Arc::new(Macro { file: file.clone(), rules, krate: self.krate });
                self.out.defined.insert(place, definition.clone());
                let public = item.kind == ItemKind::Macro && item.vis != super::item::Vis::Private;
                if public || item.attrs.iter().any(|attr| *attr == Attr::MacroExport) {
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
        self.scope = None;
        self.out.fallback.clear();
        for (file, module, at) in std::mem::take(&mut self.out.deferred) {
            self.call(&file, module, at)?;
        }
        for (file, block, at) in std::mem::take(&mut self.out.members) {
            self.member_call(&file, block, at)?;
        }
        Ok(())
    }

    fn call(&mut self, file: &Arc<File>, module: usize, at: u32) -> Result<()> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(open)) = (item.name, item.body) else { return Ok(()) };
        let name = strip_raw(file.source.text(name));
        if BUILTIN.contains(&name) {
            return Ok(());
        }
        let Some(definition) = self.lookup(file, item, name) else {
            if name == "cfg_select" {
                return self.cfg_select(file, module, open);
            }
            self.out.deferred.push((file.clone(), module, at));
            return Ok(());
        };
        self.expand(file, module, at, &definition)
    }

    /// Expands the macro call item `at` of `file` in `module` with `definition`.
    pub fn expand(&mut self, file: &Arc<File>, module: usize, at: u32, definition: &Macro) -> Result<()> {
        let item = &file.items.items[at as usize];
        let (Some(name), Some(open)) = (item.name, item.body) else { return Ok(()) };
        let name = strip_raw(file.source.text(name));
        if self.depth >= DEPTH {
            self.out.errors.push(format!("{}: macro {name} recursed past {DEPTH} levels", file.source.path.display()));
            return Ok(());
        }
        let expanded = mbe::expand(&definition.file.source, &definition.rules, &file.source, (open, definition.krate as u32));
        self.splice(module, expanded, name)
    }

    fn cfg_select(&mut self, file: &Arc<File>, module: usize, open: u32) -> Result<()> {
        let selected = match mbe::select(&file.source, open, |cfg| self.out.env.eval(cfg) != Truth::No) {
            Ok(Some(selected)) => selected,
            Ok(None) => return Ok(()),
            Err(error) => {
                self.out.errors.push(format!("cfg_select!: {error:#}"));
                return Ok(());
            }
        };
        self.splice(module, Ok(selected), "cfg_select")
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

fn child_dir(found: &Path, by_attr: bool, name: &str) -> PathBuf {
    let parent = found.parent().unwrap_or(Path::new("")).to_owned();
    let mod_rs = found.file_name().is_some_and(|file| file == "mod.rs");
    if by_attr || mod_rs { parent } else { parent.join(name) }
}
