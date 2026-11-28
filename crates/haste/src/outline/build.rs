mod expand;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rustc_lexer::TokenKind;

use super::attr::split_commas;
use super::derive;
use super::item::{Attr, Item, ItemKind};
use super::scan::syntax_error;
use super::source::{Origins, Source};
use super::tree::{
    Crate, Decl, Env, File, Files, Held, Impl, MemberBlock, Module, State, Textual, Truth, View,
};

pub(crate) use expand::{path_text, spelled_fragment};

const ATTRIBUTED: [ItemKind; 9] = [
    ItemKind::Fn,
    ItemKind::Struct,
    ItemKind::Enum,
    ItemKind::Union,
    ItemKind::Impl,
    ItemKind::Trait,
    ItemKind::Const,
    ItemKind::Static,
    ItemKind::Type,
];
const INERT: &[&str] = &[
    "allow",
    "warn",
    "deny",
    "forbid",
    "expect",
    "inline",
    "cold",
    "must_use",
    "doc",
    "repr",
    "ignore",
    "should_panic",
    "deprecated",
    "non_exhaustive",
    "track_caller",
    "automatically_derived",
    "no_mangle",
    "export_name",
    "link_section",
    "link_name",
    "used",
    "target_feature",
    "stable",
    "unstable",
    "const_stable",
    "feature",
    "lang",
    "diagnostic",
    "rustfmt",
    "clippy",
];

/// Whether the attribute `name` is builtin, so it never names an attribute macro.
pub(super) fn inert(name: &str) -> bool {
    INERT.contains(&name) || name.starts_with("rustc_")
}

/// What a loader sees of the other crates: the crates as they are, or, by crate, the views that a parallel load fixed for one level.
#[derive(Clone, Copy)]
pub enum Others<'a> {
    Crates(&'a [Crate]),
    Views(&'a [Option<Arc<View>>]),
}

/// Loads modules of one crate, which it holds while the other crates stay readable.
pub struct Loader<'a> {
    pub krate: usize,
    pub cache: &'a Files,
    pub others: Others<'a>,
    pub deps: &'a [(String, usize)],
    pub out: &'a mut Crate,
    pub depth: usize,
    pub scope: Option<Arc<Textual>>,
    edition_2024: Option<bool>,
}

fn prelude_crates(deps: &[(String, usize)]) -> Vec<usize> {
    let named = |name: &str| deps.iter().find(|(dep, _)| dep == name).map(|(_, at)| *at);
    ["std", "core"]
        .iter()
        .find_map(|name| named(name))
        .into_iter()
        .collect()
}

fn has_impl_body(src: &Source, item: &Item) -> bool {
    match item.kind {
        ItemKind::Const => item.name.is_some_and(|name| src.text(name) == "_"),
        ItemKind::Fn => item
            .body
            .is_some_and(|open| src.holds_word((open, src.pair(open)), "impl")),
        _ => false,
    }
}

fn definitely_enabled(env: &Env, attrs: &[Attr]) -> bool {
    attrs.iter().all(|attr| match attr {
        Attr::Cfg(cfg) => env.eval(cfg) == Truth::Yes,
        Attr::CfgAttr(cfg, inner) => match env.eval(cfg) {
            Truth::No => true,
            Truth::Yes => definitely_enabled(env, inner),
            Truth::Unknown => false,
        },
        _ => true,
    })
}

fn is_compile_error(
    src: &Source,
    item: &Item,
    scope: Option<&Arc<Textual>>,
    deps: &[(String, usize)],
    krate: usize,
) -> bool {
    let Some(name_tok) = item.name else {
        return false;
    };
    if crate::names::plain(src.text(name_tok)) != "compile_error" {
        return false;
    }
    if name_tok == item.start {
        return super::tree::textual(scope, "compile_error").is_none();
    }
    if name_tok == item.start + 3 && src.kind(item.start + 1) == TokenKind::Colon {
        return matches!(src.text(item.start), "core" | "std");
    }
    if name_tok == item.start + 5
        && src.kind(item.start) == TokenKind::Colon
        && src.kind(item.start + 1) == TokenKind::Colon
    {
        return matches!(src.text(item.start + 2), "core" | "std");
    }
    if name_tok == item.start + 4
        && src.kind(item.start) == TokenKind::Dollar
        && let Some(target) = src.dollar(item.start)
    {
        let target = target as usize;
        return target == krate
            || deps
                .iter()
                .any(|(dep, at)| *at == target && matches!(dep.as_str(), "core" | "std"));
    }
    false
}

impl<'a> Loader<'a> {
    /// A loader of crate `krate` into `out` that sees the other crates through `others`; a crate with nothing read yet first gets its prelude crate among `deps`.
    pub fn new(
        krate: usize,
        cache: &'a Files,
        others: Others<'a>,
        deps: &'a [(String, usize)],
        out: &'a mut Crate,
    ) -> Loader<'a> {
        if out.modules.len() == 1 && out.files.is_empty() && out.macro_use.is_empty() {
            out.macro_use = prelude_crates(deps);
        }
        Loader {
            krate,
            cache,
            others,
            deps,
            out,
            depth: 0,
            scope: None,
            edition_2024: None,
        }
    }
}

impl Loader<'_> {
    /// Reads the declared module `module` as `load` does, and after the root module adds the `#[macro_use]` extern crates it names.
    pub fn load_module(&mut self, module: usize) -> Result<()> {
        self.load(module)?;
        if module == 0 {
            self.macro_use_externs();
        }
        Ok(())
    }

    fn macro_use_externs(&mut self) {
        let Some(root) = self.out.modules.first().filter(|root| root.loaded()) else {
            return;
        };
        let files: Vec<Arc<File>> = std::iter::once(root.file)
            .chain(root.parts.iter().copied())
            .filter_map(|idx| self.out.files.get(idx).cloned())
            .collect();
        for file in files {
            let externs = file.items.items.iter().filter(|item| {
                item.kind == ItemKind::ExternCrate
                    && item.parent.is_none()
                    && item.attrs.contains(&Attr::MacroUse)
            });
            for item in externs {
                let name = file.source.text(item.start + 2);
                let Some(&(_, at)) = self.deps.iter().find(|(dep, _)| dep == name) else {
                    continue;
                };
                if !self.out.macro_use.contains(&at) {
                    self.out.macro_use.push(at);
                }
            }
        }
    }

    /// Reads the declared module `module` and expands its item macros, loading its `#[macro_use]` children with it.
    pub fn load(&mut self, module: usize) -> Result<()> {
        let State::Declared(decl) = self.out.modules[module].state.clone() else {
            return Ok(());
        };
        let (index, owner) = match decl {
            Decl::Inline { file, at } => (file, Some(at)),
            Decl::File {
                candidates,
                by_attr,
            } => match self.open(module, &candidates, by_attr) {
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
        let Some(found) = candidates
            .iter()
            .find(|candidate| super::tree::exists(candidate))
        else {
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
        self.check_file_2024(&file, false);
        let entry = &mut self.out.modules[module];
        if entry.parent.is_some() {
            entry.dir = child_dir(found, by_attr, &entry.name);
        }
        self.out.files.push(file);
        Some(self.out.files.len() - 1)
    }

    fn children(
        &mut self,
        file: &Arc<File>,
        index: usize,
        module: usize,
        owner: Option<u32>,
    ) -> Result<()> {
        if file.expanded && owner.is_none() {
            let unmarked_only = matches!(file.source.origins, Origins::Copied(_));
            self.check_file_2024(file, unmarked_only);
        }
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

    fn module_item(
        &mut self,
        file: &Arc<File>,
        (index, module): (usize, usize),
        (at, item): (u32, &Item),
    ) -> Result<()> {
        match item.kind {
            ItemKind::Mod => self.child(file, index, module, (at, item))?,
            ItemKind::MacroRules | ItemKind::Macro => self.define(file, (index as u32, at), item),
            ItemKind::MacroCall => {
                if !self.emit_compile_error(file, item) {
                    self.call(file, module, at)?;
                }
            }
            ItemKind::Impl => self.impl_block(index, module, at, file)?,
            ItemKind::Trait => self.trait_block(index, module, at, file)?,
            ItemKind::Struct | ItemKind::Enum | ItemKind::Union => {
                self.derive(file, module, (at, item))?;
            }
            _ if has_impl_body(&file.source, item) => self.out.impl_bodies.push((index, at)),
            _ => {}
        }
        Ok(())
    }

    fn attributes(&mut self, file: &Arc<File>, module: usize, (at, item): (u32, &Item)) {
        if !ATTRIBUTED.contains(&item.kind) {
            return;
        }
        self.collect_attr_sites(file, module, at, &item.attrs);
    }

    fn collect_attr_sites(&mut self, file: &Arc<File>, module: usize, at: u32, attrs: &[Attr]) {
        let src = &file.source;
        for attr in attrs {
            match attr {
                Attr::CfgAttr(cfg, inner) if self.out.env.eval(cfg) == Truth::Yes => {
                    self.collect_attr_sites(file, module, at, inner);
                }
                &Attr::Other(first) => {
                    let absolute = src.kind(first) == TokenKind::Colon
                        && src.kind(first + 1) == TokenKind::Colon
                        && src.kind(first + 2) == TokenKind::Ident;
                    if !absolute && inert(src.text(first)) {
                        continue;
                    }
                    let mut last = if absolute { first + 2 } else { first };
                    while src.kind(last + 1) == TokenKind::Colon
                        && src.kind(last + 2) == TokenKind::Colon
                        && src.kind(last + 3) == TokenKind::Ident
                    {
                        last += 3;
                    }
                    self.out.attributed.push(Held {
                        file: file.clone(),
                        owner: module,
                        at,
                        path: (first, last),
                    });
                }
                _ => {}
            }
        }
    }

    fn derive(&mut self, file: &Arc<File>, module: usize, (at, item): (u32, &Item)) -> Result<()> {
        let (expanded, foreign) =
            derive::derived(&file.source, item, &self.out.env, (self.krate, self.deps));
        self.out
            .derived
            .extend(foreign.into_iter().map(|path| Held {
                file: file.clone(),
                owner: module,
                at,
                path,
            }));
        match expanded {
            Some(expanded) => self.splice(module, expanded, "derive", child_origin(file, 1, at)),
            None => Ok(()),
        }
    }

    fn impl_block(&mut self, index: usize, module: usize, at: u32, file: &Arc<File>) -> Result<()> {
        self.out.impls.push(Impl {
            file: index,
            at,
            module,
            parts: Vec::new(),
        });
        self.check_member_compile_errors(file, Some(at));
        self.members(
            file,
            MemberBlock::Impl(self.out.impls.len() - 1),
            module,
            Some(at),
        )
    }

    fn trait_block(
        &mut self,
        index: usize,
        module: usize,
        at: u32,
        file: &Arc<File>,
    ) -> Result<()> {
        if !file
            .items
            .items
            .iter()
            .any(|item| item.parent == Some(at) && item.kind == ItemKind::MacroCall)
        {
            return Ok(());
        }
        self.out.traits.push(Impl {
            file: index,
            at,
            module,
            parts: Vec::new(),
        });
        self.check_member_compile_errors(file, Some(at));
        self.members(
            file,
            MemberBlock::Trait(self.out.traits.len() - 1),
            module,
            Some(at),
        )
    }

    fn check_member_compile_errors(&mut self, file: &File, owner: Option<u32>) {
        for item in &file.items.items {
            if item.parent == owner
                && item.kind == ItemKind::MacroCall
                && self.out.env.enabled(&item.attrs)
            {
                let _ = self.emit_compile_error(file, item);
            }
        }
    }

    fn emit_compile_error(&mut self, file: &File, item: &Item) -> bool {
        let Some(open) = item.body else {
            return false;
        };
        if !definitely_enabled(&self.out.env, &item.attrs)
            || !definitely_enabled(&self.out.env, &file.items.inner)
            || !is_compile_error(
                &file.source,
                item,
                self.scope.as_ref(),
                self.deps,
                self.krate,
            )
        {
            return false;
        }
        let args = file.source.args(open);
        let msg = match args.as_slice() {
            [arg] => path_text(&file.source, *arg, &self.out.env)
                .unwrap_or_else(|| spelled_fragment(&file.source, *arg)),
            [] => "compile_error! requires a string argument".to_owned(),
            _ => spelled_fragment(&file.source, (open + 1, file.source.pair(open))),
        };
        self.out.errors.push(syntax_error(
            &file.source,
            item.start,
            item.end,
            &msg,
            "compile_error!",
        ));
        true
    }

    fn child(
        &mut self,
        file: &Arc<File>,
        index: usize,
        parent: usize,
        (at, item): (u32, &Item),
    ) -> Result<()> {
        let name =
            crate::names::plain(file.source.text(item.name.unwrap_or(item.start))).to_owned();
        let base = &self.out.modules[parent];
        let path = self.out.env.path(&item.attrs);
        let (decl, dir) = if item.body.is_some() {
            (
                Decl::Inline { file: index, at },
                base.dir.join(path.unwrap_or(&name)),
            )
        } else {
            let candidates = match path {
                Some(path) if base.item.is_some() => vec![base.dir.join(path)],
                Some(path) => vec![file.source.beside(path)],
                None => vec![
                    base.dir.join(format!("{name}.rs")),
                    base.dir.join(&name).join("mod.rs"),
                ],
            };
            (
                Decl::File {
                    candidates,
                    by_attr: path.is_some(),
                },
                PathBuf::new(),
            )
        };
        let macro_use = item.attrs.contains(&Attr::MacroUse);
        let inline = item.body.map(|_| at);
        let child = Module {
            file: usize::MAX,
            item: inline,
            parent: Some(parent),
            name,
            dir,
            parts: Vec::new(),
            vis: item.vis,
            macro_use,
            state: State::Declared(decl),
            textual: self.scope.clone(),
        };
        self.out.modules.push(child);
        if macro_use {
            let module = self.out.modules.len() - 1;
            self.load(module)?;
            self.scope = self.out.modules[module].textual.clone();
        }
        Ok(())
    }
}

fn child_dir(found: &Path, by_attr: bool, name: &str) -> PathBuf {
    let parent = found.parent().unwrap_or(Path::new("")).to_owned();
    let mod_rs = found.file_name().is_some_and(|file| file == "mod.rs");
    if by_attr || mod_rs {
        parent
    } else {
        parent.join(name)
    }
}

fn child_origin(parent: &File, kind: u8, at: u32) -> u64 {
    use std::hash::{Hash as _, Hasher as _};
    let mut hasher = rustc_hash::FxHasher::default();
    parent.expand_origin.hash(&mut hasher);
    kind.hash(&mut hasher);
    at.hash(&mut hasher);
    hasher.finish().max(1)
}

impl Loader<'_> {
    fn is_edition_2024(&mut self) -> bool {
        if let Some(flag) = self.edition_2024 {
            return flag;
        }
        let flag = !self.out.env.fixed && manifest_is_2024(&self.out.env.vars);
        self.edition_2024 = Some(flag);
        flag
    }

    fn check_file_2024(&mut self, file: &File, unmarked_only: bool) {
        if !self.is_edition_2024() {
            return;
        }
        super::scan::check_edition_2024_source(
            &file.source,
            unmarked_only,
            &mut |from, to, errors| {
                check_attr_2024(&file.source, &self.out.env, unmarked_only, from, to, errors);
            },
            &mut self.out.errors,
        );
    }
}

fn manifest_is_2024(vars: &[(String, String)]) -> bool {
    let manifest_path = vars
        .iter()
        .find(|(k, _)| k == "CARGO_MANIFEST_PATH")
        .map(|(_, v)| PathBuf::from(v))
        .or_else(|| {
            vars.iter()
                .find(|(k, _)| k == "CARGO_MANIFEST_DIR")
                .map(|(_, d)| Path::new(d).join("Cargo.toml"))
        });
    let Some(path) = manifest_path else {
        return false;
    };
    edition_from_manifest(&path).unwrap_or(false)
}

fn edition_from_manifest(path: &Path) -> Option<bool> {
    let text = std::fs::read_to_string(path).ok()?;
    let root = toml::from_str::<toml::Table>(&text).ok()?;
    let edition = root.get("package")?.as_table()?.get("edition")?;
    if let Some(ed) = edition.as_str() {
        return Some(ed == "2024");
    }
    let inherited = edition
        .as_table()
        .and_then(|t| t.get("workspace"))
        .and_then(toml::Value::as_bool)
        == Some(true);
    if !inherited {
        return Some(false);
    }
    for dir in path.parent()?.ancestors().skip(1) {
        let ws_path = dir.join("Cargo.toml");
        let Ok(ws_text) = std::fs::read_to_string(&ws_path) else {
            continue;
        };
        let Ok(ws_root) = toml::from_str::<toml::Table>(&ws_text) else {
            continue;
        };
        if let Some(ws) = ws_root.get("workspace").and_then(toml::Value::as_table) {
            return Some(
                ws.get("package")
                    .and_then(toml::Value::as_table)
                    .and_then(|p| p.get("edition"))
                    .and_then(toml::Value::as_str)
                    == Some("2024"),
            );
        }
    }
    Some(false)
}

fn check_attr_2024(
    src: &Source,
    env: &Env,
    unmarked_only: bool,
    from: u32,
    to: u32,
    errors: &mut Vec<String>,
) {
    if from >= to || (unmarked_only && src.mark(from).is_some()) {
        return;
    }
    let mut after = from;
    while after < to
        && matches!(
            src.kind(after),
            TokenKind::Ident | TokenKind::RawIdent | TokenKind::Colon
        )
    {
        after += 1;
    }
    if after == from {
        return;
    }
    if after == from + 1 && src.kind(from) == TokenKind::Ident {
        let name = src.text(from);
        if name == "unsafe" && after < to && src.kind(after) == TokenKind::OpenParen {
            return;
        }
        if name == "cfg_attr" && after < to && src.kind(after) == TokenKind::OpenParen {
            let parts = split_commas(src, after + 1, src.pair(after));
            if let Some(&(first, last)) = parts.first() {
                let cond: String = (first..last)
                    .map(|tok| src.text(tok))
                    .collect::<Vec<_>>()
                    .join(" ");
                if let Ok(cfg) = crate::graph::cfg::parse(&cond)
                    && env.eval(&cfg) == Truth::Yes
                {
                    for &(f, t) in &parts[1..] {
                        check_attr_2024(src, env, unmarked_only, f, t, errors);
                    }
                }
            }
            return;
        }
        if matches!(name, "no_mangle" | "export_name" | "link_section") {
            errors.push(syntax_error(
                src,
                from,
                after,
                "unsafe attribute used without unsafe",
                "usage of unsafe attribute",
            ));
        }
    }
}
