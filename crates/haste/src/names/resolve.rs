use std::sync::Arc;

use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::graph::Workspace;
use crate::outline::Outline;
use crate::outline::item::{Attr, ItemKind, Vis};
use crate::outline::source::Source;
use crate::outline::tree::{Crate as Tree, File, Macro};

use super::collect::{self, Context, Pending};
use super::path::{Path, Paths, Requests};
use super::{Binding, Crate, Def, ItemRef, ModRef, Names, Ns, Origin, PerNs, SELF, Scope, Sym, Symbols, UNDERSCORE, Unresolved, plain};

type Found = Vec<(Sym, Ns, Binding)>;

/// What resolving one import found: nothing, the same source as last time, or bindings with the source they came from.
enum Outcome {
    Missing,
    Unchanged,
    Bound(Found, Option<(Def, u32)>),
}

fn stamp(paths: &Paths<'_>, target: Def) -> Option<(Def, u32)> {
    match target {
        Def::Module(home) => paths.scope(home).map(|scope| (target, scope.version())),
        Def::Item(_) => Some((target, 0)),
        Def::Variant(..) | Def::Primitive(_) => None,
    }
}

const NAMESPACES: [Ns; 3] = [Ns::Types, Ns::Values, Ns::Macros];

/// Whether a name of visibility `vis` declared in module `home` is visible to module `importer`.
pub(crate) fn visible(paths: &Paths<'_>, vis: Vis, home: ModRef, importer: ModRef) -> bool {
    match vis {
        Vis::Pub => true,
        _ if home.krate != importer.krate => false,
        Vis::Private => paths.within(importer, home),
        _ => true,
    }
}

fn glob(paths: &Paths<'_>, target: Def, (at, vis): (ModRef, Vis), since: u32) -> Found {
    let mut found = Found::new();
    match target {
        Def::Module(home) => {
            let Some(scope) = paths.scope(home) else {
                return found;
            };
            let names: Box<dyn Iterator<Item = &Sym>> = if since == 0 { Box::new(scope.names.keys()) } else { Box::new(scope.log[since as usize..].iter()) };
            for name in names {
                let entry = &scope.names[name];
                for ns in NAMESPACES {
                    let Some(binding) = entry.get(ns) else {
                        continue;
                    };
                    if visible(paths, binding.vis, home, at) {
                        found.push((*name, ns, Binding { def: binding.def, vis, origin: Origin::Glob }));
                    }
                }
            }
        }
        Def::Item(item) => {
            for (name, binding) in paths.all_variants(item) {
                let binding = Binding { vis, ..binding };
                found.push((name, Ns::Types, binding));
                found.push((name, Ns::Values, binding));
            }
        }
        Def::Variant(..) | Def::Primitive(_) => {}
    }
    found
}

fn named(paths: &Paths<'_>, path: &Path<'_>, at: ModRef) -> Option<(PerNs, Option<ModRef>, Option<Def>)> {
    let (last, prefix) = path.segments.split_last()?;
    if prefix.is_empty() {
        return Some((paths.first(*last, at, path), None, None));
    }
    let target = paths.resolve(&Path { segments: prefix, ..*path }, at)?;
    if *last == SELF {
        let binding = Binding { def: target, vis: Vis::Pub, origin: Origin::Item };
        return Some((PerNs { types: Some(binding), values: None, macros: None }, None, None));
    }
    let home = match target {
        Def::Module(home) => Some(home),
        _ => None,
    };
    Some((paths.member(target, *last), home, Some(target)))
}

fn import(paths: &Paths<'_>, pending: &Pending) -> Outcome {
    let path = paths.use_path(Path { segments: &pending.segments, absolute: pending.absolute, dollar: pending.dollar });
    let at = ModRef { krate: paths.krate, module: pending.module };
    let Some(name) = pending.name else {
        let Some(target) = paths.resolve(&path, at) else {
            return Outcome::Missing;
        };
        let stamp = stamp(paths, target);
        if stamp.is_some() && stamp == pending.stamp {
            return Outcome::Unchanged;
        }
        let since = pending.stamp.filter(|(seen, _)| *seen == target).map_or(0, |(_, version)| version);
        return Outcome::Bound(glob(paths, target, (at, pending.vis), since), stamp);
    };
    let Some((found, home, target)) = named(paths, &path, at) else {
        return Outcome::Missing;
    };
    if name == UNDERSCORE {
        let Some(binding) = found.types else {
            return Outcome::Missing;
        };
        let stamp = Some((binding.def, 0));
        return if stamp == pending.stamp { Outcome::Unchanged } else { Outcome::Bound(Found::new(), stamp) };
    }
    let stamp = target.and_then(|target| stamp(paths, target));
    if stamp.is_some() && stamp == pending.stamp {
        return Outcome::Unchanged;
    }
    let mut out = Found::new();
    for ns in NAMESPACES {
        let Some(binding) = found.get(ns) else {
            continue;
        };
        if home.is_none_or(|home| visible(paths, binding.vis, home, at)) {
            out.push((name, ns, Binding { def: binding.def, vis: pending.vis, origin: Origin::Import }));
        }
    }
    if out.is_empty() { Outcome::Missing } else { Outcome::Bound(out, stamp) }
}

fn apply(scopes: &mut [Scope], module: u32, found: Found) -> bool {
    let mut changed = false;
    for (name, ns, binding) in found {
        changed |= scopes[module as usize].bind(name, ns, binding);
    }
    changed
}

const PRELUDES: [(&str, &str); 4] = [("2015", "rust_2015"), ("2018", "rust_2018"), ("2021", "rust_2021"), ("2024", "rust_2024")];

fn prelude(paths: &Paths<'_>, tree: &Tree, edition: &str) -> Option<ModRef> {
    if let Some(found) = own_prelude(paths, tree) {
        return Some(found);
    }
    let no_std = tree.files.get(tree.modules.first()?.file)?.items.inner.iter().any(|attr| matches!(attr, Attr::NoStd | Attr::NoCore));
    let (core, std) = (paths.symbols.intern("core"), paths.symbols.intern("std"));
    let dep = |root| paths.deps.iter().find(|(name, _)| *name == root);
    let (_, krate) = if no_std { dep(core) } else { dep(std).or_else(|| dep(core)) }?;
    let module = PRELUDES.iter().find(|(year, _)| *year == edition).map_or("rust_2021", |(_, module)| module);
    let prelude = paths.member(Def::Module(ModRef { krate: *krate as u32, module: 0 }), paths.symbols.intern("prelude")).types?.def;
    match paths.member(prelude, paths.symbols.intern(module)).types?.def {
        Def::Module(found) => Some(found),
        _ => None,
    }
}

fn own_prelude(paths: &Paths<'_>, tree: &Tree) -> Option<ModRef> {
    let file = tree.files.get(tree.modules.first()?.file)?;
    let src = &file.source;
    let marked = |item: &&crate::outline::item::Item| item.attrs.iter().any(|attr| matches!(attr, Attr::Other(at) if src.text(*at) == "prelude_import"));
    let import = file.items.items.iter().filter(|item| item.kind == ItemKind::Use && item.parent.is_none()).find(marked)?;
    let words = (import.start..import.end).map(|at| src.text(at)).skip_while(|word| *word != "use").skip(1);
    let mut def = Def::Module(ModRef { krate: paths.krate, module: 0 });
    for word in words.filter(|word| !matches!(*word, ":" | "::" | "*" | ";")) {
        def = paths.member(def, paths.symbols.intern(word)).types?.def;
    }
    match def {
        Def::Module(found) => Some(found),
        _ => None,
    }
}

fn resolve_macro(paths: &Paths<'_>, file: &File, (call, at): (u32, ModRef)) -> Option<ItemRef> {
    let item = &file.items.items[call as usize];
    macro_at(paths, &file.source, (item.start, item.name?), at)
}

/// The macro that the path from token `start` to name token `name` of `src` names in module `at`.
pub fn macro_at(paths: &Paths<'_>, src: &Source, (start, name): (u32, u32), at: ModRef) -> Option<ItemRef> {
    let dollar = if src.kind(start) == TokenKind::Dollar { src.dollar(start) } else { None };
    let words = (start..=name).filter(|token| matches!(src.kind(*token), TokenKind::Ident | TokenKind::RawIdent));
    let segments: Vec<Sym> = words.map(|token| paths.symbols.intern(plain(src.text(token)))).collect();
    macro_path(paths, &Path { segments: &segments, absolute: src.kind(start) == TokenKind::Colon, dollar }, at)
}

/// The macro that `path` names in module `at`; a crate root that names no such macro asks for its whole crate, since `#[macro_export]` puts a macro from any module there.
pub fn macro_path(paths: &Paths<'_>, path: &Path<'_>, at: ModRef) -> Option<ItemRef> {
    let (last, prefix) = path.segments.split_last()?;
    let binding = if prefix.is_empty() {
        paths.first(*last, at, path).macros.or_else(|| paths.macro_use(*last))?
    } else {
        let owner = paths.resolve(&Path { segments: prefix, ..*path }, at)?;
        let found = paths.member(owner, *last).macros;
        if let (None, Def::Module(root)) = (found, owner) {
            paths.ask_crate(root);
        }
        found?
    };
    match binding.def {
        Def::Item(item) => Some(item),
        _ => None,
    }
}

pub(super) fn paths_of<'a>(names: &'a Names, outline: &'a Outline, krate: usize, demands: &'a Requests) -> Paths<'a> {
    Paths { demands: Some(demands), ..names.paths(outline, krate as u32) }
}

fn declared(outline: &Outline, krate: usize) -> impl Iterator<Item = ModRef> + '_ {
    let modules = outline.crates[krate].modules.iter().enumerate();
    let declared = modules.filter(|(_, module)| matches!(module.state, crate::outline::tree::State::Declared(_)));
    declared.map(move |(module, _)| ModRef { krate: krate as u32, module: module as u32 })
}

impl Names {
    /// Empty scopes for every crate of `outline`; `resolve` fills them as modules load.
    pub fn new(workspace: &Workspace, outline: &Outline) -> Names {
        let crates = outline.roots.iter().map(|root| {
            let manifest = &workspace.packages[workspace.graph.units[root.unit].package].manifest;
            Crate { edition_2015: manifest.edition == "2015", edition: manifest.edition.clone(), ..Crate::default() }
        });
        let mut names = Names { crates: crates.collect(), symbols: Symbols::default() };
        for (entry, deps) in names.crates.iter_mut().zip(&outline.deps) {
            entry.deps = deps.iter().map(|(name, at)| (names.symbols.intern(name), *at)).collect();
        }
        names
    }

    /// Forgets what crate `krate` resolved, keeping its edition and dependencies, so that `resolve` resolves it again from its reloaded outline.
    pub fn reset(&mut self, krate: usize) {
        let old = &self.crates[krate];
        self.crates[krate] = Crate { edition_2015: old.edition_2015, edition: old.edition.clone(), deps: old.deps.clone(), eager: old.eager, ..Crate::default() };
    }

    /// Collects every loaded module not collected yet and resolves imports until nothing changes, loading the modules that lookups reach; each pass resolves again only the imports of crates whose own scopes, or the scopes of a crate they depend on, changed.
    pub fn resolve(&mut self, outline: &mut Outline) -> Result<()> {
        let users = outline.users();
        let mut changed = vec![true; self.crates.len()];
        loop {
            self.collect(outline, &mut changed);
            let demands = Requests::default();
            loop {
                let mut sweep = vec![false; self.crates.len()];
                crate::outline::dependents_of(&users, &mut sweep, (0..changed.len()).filter(|&krate| changed[krate]));
                changed = self.pass(outline, &demands, &sweep);
                if self.want(demands.names.take()) | self.spread_whole() {
                    changed.fill(true);
                } else if !changed.contains(&true) {
                    break;
                }
            }
            self.preludes(outline, &demands, &mut changed);
            let expanded = self.expand_imported(outline, &demands)?;
            let wanted = self.want(demands.names.take());
            if wanted {
                changed.fill(true);
            }
            let crates = demands.crates.take();
            let mut demands = demands.modules.into_inner();
            demands.extend(crates.into_iter().flat_map(|krate| declared(outline, krate as usize)));
            for krate in (0..outline.crates.len()).filter(|krate| self.crates[*krate].eager) {
                demands.extend(declared(outline, krate));
            }
            if demands.is_empty() && !expanded && !wanted {
                if outline.fall_back()? {
                    continue;
                }
                return Ok(());
            }
            demands.sort_unstable_by_key(|module| (module.krate, module.module));
            demands.dedup();
            let modules: Vec<(usize, usize)> = demands.iter().map(|module| (module.krate as usize, module.module as usize)).collect();
            outline.load_many(&modules)?;
        }
    }

    fn collect(&mut self, outline: &Outline, collected: &mut [bool]) {
        let symbols = &self.symbols;
        for (krate, tree) in outline.crates.iter().enumerate() {
            let entry = &mut self.crates[krate];
            entry.scopes.resize_with(tree.modules.len(), Scope::default);
            entry.collected.resize(tree.modules.len(), None);
            if tree.modules.iter().zip(&entry.collected).all(|(module, seen)| !module.loaded() || *seen == Some(module.parts.len())) {
                continue;
            }
            let mut children = vec![Vec::new(); tree.modules.len()];
            for (index, module) in tree.modules.iter().enumerate() {
                if let Some(parent) = module.parent {
                    children[parent].push(index);
                }
            }
            let context = Context { krate: krate as u32, env: &tree.env, deps: &outline.deps[krate], symbols, children: &children };
            let mut changed = false;
            for (index, module) in tree.modules.iter().enumerate() {
                let seen = entry.collected[index];
                if !module.loaded() || seen == Some(module.parts.len()) {
                    continue;
                }
                let own = seen.is_none().then_some((module.file as u32, module.item));
                let parts = module.parts[seen.unwrap_or(0)..].iter().map(|&part| (part as u32, None));
                let files: Vec<(u32, Option<u32>)> = own.into_iter().chain(parts).collect();
                let found = collect::collect_module(tree, index, &files, &context, &mut entry.scopes);
                entry.add_pending(found);
                entry.collected[index] = Some(module.parts.len());
                changed = true;
            }
            if changed {
                collected[krate] = true;
                collect::lift_macro_use(tree, &mut entry.scopes);
                if let Some(root) = entry.scopes.first() {
                    entry.externs = collect::externs(root);
                }
            }
        }
    }

    /// Resolves again the active imports of the crates that `sweep` marks and returns the crates whose scopes changed.
    fn pass(&mut self, outline: &Outline, demands: &Requests, sweep: &[bool]) -> Vec<bool> {
        let mut found: Vec<(usize, usize, Found, Option<(Def, u32)>)> = Vec::new();
        let mut unresolved = vec![Vec::new(); self.crates.len()];
        for (krate, entry) in self.crates.iter().enumerate().filter(|(krate, _)| sweep[*krate]) {
            let paths = paths_of(self, outline, krate, demands);
            for (index, pending) in entry.pending.iter().enumerate().filter(|(_, pending)| pending.active) {
                match import(&paths, pending) {
                    Outcome::Bound(bindings, stamp) => found.push((krate, index, bindings, stamp)),
                    Outcome::Unchanged => {}
                    Outcome::Missing => unresolved[krate].push(Unresolved { file: pending.file, import: pending.import }),
                }
            }
        }
        let mut changed = vec![false; self.crates.len()];
        for (krate, index, bindings, stamp) in found {
            let entry = &mut self.crates[krate];
            let pending = &mut entry.pending[index];
            let moved = pending.name.is_none() && pending.stamp.map(|(target, _)| target) != stamp.map(|(target, _)| target);
            pending.stamp = stamp;
            let module = pending.module;
            changed[krate] |= apply(&mut entry.scopes, module, bindings);
            if let (true, Some((Def::Module(target), _))) = (moved, stamp) {
                let asked = entry.wanted_in(module).into_iter().map(|name| (target, name));
                demands.names.borrow_mut().extend(asked);
            }
        }
        for ((entry, unresolved), _) in self.crates.iter_mut().zip(unresolved).zip(sweep).filter(|(_, swept)| **swept) {
            entry.unresolved = unresolved;
        }
        changed
    }

    fn expand_imported(&self, outline: &mut Outline, demands: &Requests) -> Result<bool> {
        let mut expanded = false;
        for krate in 0..outline.crates.len() {
            let (found, foreign) = self.imported_macros(outline, krate, demands, false);
            outline.set_foreign(krate, &foreign);
            if !found.is_empty() {
                let found = found.into_iter().map(|(index, definition)| (index - foreign.iter().filter(|at| **at < index).count(), definition)).collect();
                outline.expand(krate, found)?;
                expanded = true;
            }
            let (members, _) = self.imported_macros(outline, krate, demands, true);
            if !members.is_empty() {
                outline.expand_members(krate, members)?;
                expanded = true;
            }
        }
        Ok(expanded)
    }

    /// The waiting macro calls of crate `krate` that name resolution binds, in modules or, when `members`, in impl or trait bodies: those with a definition, and the indexes of the others.
    fn imported_macros(&self, outline: &Outline, krate: usize, demands: &Requests, members: bool) -> (Vec<(usize, Arc<Macro>)>, Vec<usize>) {
        let paths = paths_of(self, outline, krate, demands);
        let tree = &outline.crates[krate];
        let (mut found, mut foreign) = (Vec::new(), Vec::new());
        let check = |(index, call): (usize, &crate::outline::tree::Waiting)| {
            let item = resolve_macro(&paths, &call.file, (call.at, ModRef { krate: krate as u32, module: call.owner as u32 }))?;
            Some((index, outline.crates[item.krate as usize].defined.get(&(item.file, item.item)).cloned()))
        };
        let resolved: Vec<_> = match members {
            true => tree.members.iter().map(|(_, call)| call).enumerate().filter_map(check).collect(),
            false => tree.deferred.iter().enumerate().filter_map(check).collect(),
        };
        for (index, definition) in resolved {
            match definition {
                Some(definition) => found.push((index, definition)),
                None => foreign.push(index),
            }
        }
        (found, foreign)
    }

    fn preludes(&mut self, outline: &Outline, demands: &Requests, found: &mut [bool]) {
        for krate in 0..self.crates.len() {
            if self.crates[krate].prelude.is_some() || !outline.crates[krate].modules.first().is_some_and(|root| root.loaded()) {
                continue;
            }
            let paths = paths_of(self, outline, krate, demands);
            self.crates[krate].prelude = prelude(&paths, &outline.crates[krate], &self.crates[krate].edition);
            found[krate] |= self.crates[krate].prelude.is_some();
        }
    }
}
