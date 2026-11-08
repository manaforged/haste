pub mod attr;
pub mod build;
pub mod fragment;
pub mod item;
pub mod mbe;
pub mod rules;
pub mod scan;
pub mod source;
pub mod tree;

use rustc_hash::FxHashMap as HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::graph::manifest::Kind;
use crate::graph::{Targets, Workspace};
use build::Loader;
use item::{Attr, ItemKind, Vis};
use tree::{Cache, Crate, Decl, Env, Macro, Module, State};

/// One crate to outline: a library or binary target of a unit.
#[derive(Clone, Debug)]
pub struct Root {
    pub unit: usize,
    pub name: String,
    pub path: PathBuf,
    pub lib: bool,
}

/// The module trees of every crate in a workspace graph, read as far as lookups needed them.
pub struct Outline {
    pub roots: Vec<Root>,
    pub crates: Vec<Crate>,
    pub deps: Vec<Vec<(String, usize)>>,
    pub cache: Cache,
}

/// The library and member binary targets of every unit.
pub fn roots(workspace: &Workspace) -> Vec<Root> {
    let mut roots = Vec::new();
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let manifest = &workspace.packages[entry.package].manifest;
        if let Some(lib) = &manifest.lib {
            let name = manifest.lib_name.as_ref().unwrap_or(&manifest.name);
            roots.push(Root { unit, name: name.replace('-', "_"), path: lib.clone(), lib: true });
        }
        if workspace.graph.members.contains(&entry.package) {
            let bins = manifest.bins.iter().map(|(name, path)| Root { unit, name: name.clone(), path: path.clone(), lib: false });
            roots.extend(bins);
        }
    }
    roots
}

const IMPLICIT: [&str; 3] = ["std", "core", "alloc"];

fn implicit(workspace: &Workspace, roots: &[Root]) -> Vec<(String, usize)> {
    let named = |name: &str| {
        let found = roots.iter().position(|root| root.lib && workspace.packages[workspace.graph.units[root.unit].package].sysroot && root.name == name);
        found.map(|at| (name.to_owned(), at))
    };
    IMPLICIT.iter().chain(&["proc_macro"]).filter_map(|name| named(name)).collect()
}

fn dependencies(workspace: &Workspace, roots: &[Root]) -> Vec<Vec<(String, usize)>> {
    let libs: HashMap<usize, usize> = roots.iter().enumerate().filter(|(_, root)| root.lib).map(|(index, root)| (root.unit, index)).collect();
    let implicit = implicit(workspace, roots);
    roots
        .iter()
        .map(|root| {
            let unit = &workspace.graph.units[root.unit];
            let package = &workspace.packages[unit.package];
            let normal = unit.deps.iter().filter(|(_, _, kind)| *kind == Kind::Normal);
            let extern_name = |key: &str, dep: usize, lib: usize| {
                let renamed = workspace.packages[workspace.graph.units[dep].package].manifest.name != key;
                if renamed { key.replace('-', "_") } else { roots[lib].name.clone() }
            };
            let mut deps: Vec<(String, usize)> = normal.filter_map(|(key, dep, _)| libs.get(dep).map(|&lib| (extern_name(key, *dep, lib), lib))).collect();
            if !root.lib && let Some(&lib) = libs.get(&root.unit) {
                deps.push((root.name.replace('-', "_"), lib));
            }
            if !package.sysroot {
                let wanted = |name: &str| IMPLICIT.contains(&name) || (name == "proc_macro" && package.manifest.proc_macro);
                deps.extend(implicit.iter().filter(|(name, _)| wanted(name)).cloned());
            }
            deps
        })
        .collect()
}

/// Creates the outline of every root with only the root modules declared; `load` reads modules as lookups need them.
pub fn new(workspace: &Workspace, targets: &Targets) -> Outline {
    let roots = roots(workspace);
    let deps = dependencies(workspace, &roots);
    let crates = roots.iter().map(|root| root_crate(workspace, targets, root)).collect();
    Outline { roots, crates, deps, cache: Cache::default() }
}

fn root_crate(workspace: &Workspace, targets: &Targets, root: &Root) -> Crate {
    let unit = &workspace.graph.units[root.unit];
    let env = Env { target: targets.of(unit.side).clone(), features: unit.features.clone() };
    let dir = root.path.parent().map(PathBuf::from).unwrap_or_default();
    let decl = Decl::File { candidates: vec![root.path.clone()], by_attr: true };
    let module = Module { file: usize::MAX, item: None, parent: None, name: String::new(), dir, parts: Vec::new(), vis: Vis::Pub, macro_use: false, state: State::Declared(decl), textual: None };
    Crate { env, modules: vec![module], ..Crate::default() }
}

impl Outline {
    /// The outline of the single crate whose root file is `path`, with no dependencies.
    pub fn file(path: &std::path::Path, target: crate::graph::cfg::Target) -> Outline {
        let root = Root { unit: 0, name: String::new(), path: path.to_owned(), lib: true };
        let dir = path.parent().map(PathBuf::from).unwrap_or_default();
        let decl = Decl::File { candidates: vec![path.to_owned()], by_attr: true };
        let module = Module { file: usize::MAX, item: None, parent: None, name: String::new(), dir, parts: Vec::new(), vis: Vis::Pub, macro_use: false, state: State::Declared(decl), textual: None };
        let krate = Crate { env: Env { target, features: Default::default() }, modules: vec![module], ..Crate::default() };
        Outline { roots: vec![root], crates: vec![krate], deps: vec![Vec::new()], cache: Cache::default() }
    }

    /// Reads module `module` of crate `krate` if it is only declared, then expands what that made resolvable.
    pub fn load(&mut self, krate: usize, module: usize) -> Result<()> {
        let mut demands = Vec::new();
        self.load_one(krate, module, &mut demands)?;
        self.settle(demands)
    }

    /// Reads each listed `(crate, module)` that is only declared, then expands what that made resolvable.
    pub fn load_many(&mut self, modules: &[(usize, usize)]) -> Result<()> {
        self.read_ahead(modules);
        let mut demands = Vec::new();
        for &(krate, module) in modules {
            self.load_one(krate, module, &mut demands)?;
        }
        self.settle(demands)
    }

    /// Reads every module of every crate.
    pub fn load_all(&mut self) -> Result<()> {
        loop {
            let declared = |(krate, tree): (usize, &Crate)| (0..tree.modules.len()).filter(|at| matches!(tree.modules[*at].state, State::Declared(_))).map(move |at| (krate, at)).collect::<Vec<_>>();
            let modules: Vec<(usize, usize)> = self.crates.iter().enumerate().flat_map(declared).collect();
            self.load_many(&modules)?;
            if !self.crates.iter().any(Crate::partial) {
                return Ok(());
            }
        }
    }

    /// Expands the deferred calls of crate `krate` at the given indexes with the definitions that name resolution found for them.
    pub fn expand(&mut self, krate: usize, found: Vec<(usize, Arc<Macro>)>) -> Result<()> {
        let deferred = std::mem::take(&mut self.crates[krate].deferred);
        let mut found = found.into_iter().peekable();
        let mut calls = Vec::new();
        for (index, call) in deferred.into_iter().enumerate() {
            match found.next_if(|(at, _)| *at == index) {
                Some((_, definition)) => calls.push((call, definition)),
                None => self.crates[krate].deferred.push(call),
            }
        }
        let mut demands = Vec::new();
        self.with_loader(krate, &mut demands, |loader| calls.into_iter().try_for_each(|((file, module, at), definition)| loader.expand(&file, module, at, &definition)))?;
        self.settle(demands)
    }

    /// Reads every module of the crates whose exports may define a still-deferred macro call, once name resolution can bind nothing more; returns whether it read any.
    pub fn fall_back(&mut self) -> Result<bool> {
        let mut demands = Vec::new();
        for krate in 0..self.crates.len() {
            if !self.crates[krate].deferred.is_empty() {
                self.with_loader(krate, &mut demands, |loader| loader.retry())?;
            }
        }
        let waiting = self.crates.iter().filter(|krate| !krate.deferred.is_empty());
        demands.extend(waiting.flat_map(|krate| krate.fallback.iter().copied()));
        demands.sort_unstable();
        demands.dedup();
        demands.retain(|krate| self.crates[*krate].partial());
        if demands.is_empty() {
            return Ok(false);
        }
        self.settle(demands)?;
        Ok(true)
    }

    /// Moves the deferred calls of crate `krate` at the given indexes, which name resolution bound to procedural macros, out of the calls left to expand.
    pub fn set_foreign(&mut self, krate: usize, indexes: &[usize]) {
        let holder = &mut self.crates[krate];
        let deferred = std::mem::take(&mut holder.deferred);
        for (index, call) in deferred.into_iter().enumerate() {
            if indexes.binary_search(&index).is_ok() { holder.foreign.push(call) } else { holder.deferred.push(call) }
        }
    }

    fn with_loader<T>(&mut self, krate: usize, demands: &mut Vec<usize>, run: impl FnOnce(&mut Loader<'_>) -> Result<T>) -> Result<T> {
        let mut holder = std::mem::take(&mut self.crates[krate]);
        if holder.modules.len() == 1 && holder.files.is_empty() && holder.macro_use.is_empty() {
            holder.macro_use = self.prelude_crates(krate);
        }
        let mut loader = Loader { krate, cache: &self.cache, others: &self.crates, deps: &self.deps[krate], out: &mut holder, demands, depth: 0, scope: None };
        let result = run(&mut loader);
        self.crates[krate] = holder;
        result
    }

    fn prelude_crates(&self, krate: usize) -> Vec<usize> {
        let named = |name: &str| self.deps[krate].iter().find(|(dep, _)| dep == name).map(|(_, at)| *at);
        ["std", "core"].iter().find_map(|name| named(name)).into_iter().collect()
    }

    fn read_ahead(&self, modules: &[(usize, usize)]) {
        let files = modules.iter().filter_map(|&(krate, module)| match &self.crates[krate].modules[module].state {
            State::Declared(Decl::File { candidates, .. }) => Some(candidates),
            _ => None,
        });
        let paths: Vec<PathBuf> = files.flatten().cloned().collect();
        self.cache.read_all(&paths);
    }

    fn load_one(&mut self, krate: usize, module: usize, demands: &mut Vec<usize>) -> Result<()> {
        self.with_loader(krate, demands, |loader| loader.load(module))?;
        if module == 0 {
            self.macro_use_externs(krate);
        }
        Ok(())
    }

    fn macro_use_externs(&mut self, krate: usize) {
        let holder = &self.crates[krate];
        let Some(file) = holder.modules.first().filter(|root| root.loaded()).map(|root| holder.files[root.file].clone()) else { return };
        let externs = file.items.items.iter().filter(|item| item.kind == ItemKind::ExternCrate && item.parent.is_none() && item.attrs.contains(&Attr::MacroUse));
        let names: Vec<&str> = externs.map(|item| file.source.text(item.start + 2)).collect();
        let found: Vec<usize> = names.iter().filter_map(|name| self.deps[krate].iter().find(|(dep, _)| dep == name).map(|(_, at)| *at)).collect();
        let list = &mut self.crates[krate].macro_use;
        for at in found {
            if !list.contains(&at) {
                list.push(at);
            }
        }
    }

    fn settle(&mut self, mut demands: Vec<usize>) -> Result<()> {
        while !demands.is_empty() {
            demands.sort_unstable();
            demands.dedup();
            let mut next = Vec::new();
            for krate in demands {
                let declared: Vec<usize> = (0..self.crates[krate].modules.len()).filter(|at| matches!(self.crates[krate].modules[*at].state, State::Declared(_))).collect();
                for module in declared {
                    self.load_one(krate, module, &mut next)?;
                }
            }
            for krate in 0..self.crates.len() {
                if !self.crates[krate].deferred.is_empty() {
                    self.with_loader(krate, &mut next, |loader| loader.retry())?;
                }
            }
            demands = next;
        }
        Ok(())
    }
}
