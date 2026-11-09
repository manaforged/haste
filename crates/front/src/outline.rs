pub mod attr;
pub mod build;
pub mod derive;
pub mod edit;
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

/// What a crate is to its package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Lib,
    Bin,
    Script,
}

/// One crate to outline: the library, a binary or the build script of a unit.
#[derive(Clone, Debug)]
pub struct Root {
    pub unit: usize,
    pub name: String,
    pub path: PathBuf,
    pub role: Role,
}

/// The module trees of every crate in a workspace graph, read as far as lookups needed them.
pub struct Outline {
    pub roots: Vec<Root>,
    pub crates: Vec<Crate>,
    pub deps: Vec<Vec<(String, usize)>>,
    pub cache: Cache,
    /// The output text of each procedural macro called in a body, by macro function and input text, or why it failed.
    pub procs: HashMap<(crate::names::ItemRef, String), Result<String, String>>,
}

/// The library and member binary targets of every unit, and the build scripts of the units `scripts`.
pub fn roots(workspace: &Workspace, scripts: &[usize]) -> Vec<Root> {
    let mut roots = Vec::new();
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let manifest = &workspace.packages[entry.package].manifest;
        if let Some(lib) = &manifest.lib {
            let name = manifest.lib_name.as_ref().unwrap_or(&manifest.name).replace('-', "_");
            let supplied = workspace.packages[entry.package].sysroot && name == "proc_macro";
            let path = if supplied { PathBuf::from(tree::PROC_MACRO_ROOT) } else { lib.clone() };
            roots.push(Root { unit, name, path, role: Role::Lib });
        }
        if workspace.graph.members.contains(&entry.package) {
            let bins = manifest.bins.iter().map(|(name, path)| Root { unit, name: name.clone(), path: path.clone(), role: Role::Bin });
            roots.extend(bins);
        }
        if let Some(path) = manifest.build.as_ref().filter(|_| scripts.contains(&unit)) {
            roots.push(Root { unit, name: "build_script_build".to_owned(), path: path.clone(), role: Role::Script });
        }
    }
    roots
}

const IMPLICIT: [&str; 3] = ["std", "core", "alloc"];

fn implicit(workspace: &Workspace, roots: &[Root]) -> Vec<(String, usize)> {
    let named = |name: &str| {
        let found = roots.iter().position(|root| root.role == Role::Lib && workspace.packages[workspace.graph.units[root.unit].package].sysroot && root.name == name);
        found.map(|at| (name.to_owned(), at))
    };
    IMPLICIT.iter().chain(&["proc_macro"]).filter_map(|name| named(name)).collect()
}

fn dependencies(workspace: &Workspace, roots: &[Root]) -> Vec<Vec<(String, usize)>> {
    let libs: HashMap<usize, usize> = roots.iter().enumerate().filter(|(_, root)| root.role == Role::Lib).map(|(index, root)| (root.unit, index)).collect();
    let implicit = implicit(workspace, roots);
    roots
        .iter()
        .map(|root| {
            let unit = &workspace.graph.units[root.unit];
            let package = &workspace.packages[unit.package];
            let wanted = if root.role == Role::Script { Kind::Build } else { Kind::Normal };
            let normal = unit.deps.iter().filter(|(_, _, kind)| *kind == wanted);
            let extern_name = |key: &str, dep: usize, lib: usize| {
                let renamed = workspace.packages[workspace.graph.units[dep].package].manifest.name != key;
                if renamed { key.replace('-', "_") } else { roots[lib].name.clone() }
            };
            let mut deps: Vec<(String, usize)> = normal.filter_map(|(key, dep, _)| libs.get(dep).map(|&lib| (extern_name(key, *dep, lib), lib))).collect();
            if root.role == Role::Bin && let Some(&lib) = libs.get(&root.unit) {
                deps.push((root.name.replace('-', "_"), lib));
            }
            if !package.sysroot || root.role == Role::Script {
                let wanted = |name: &str| IMPLICIT.contains(&name) || (name == "proc_macro" && package.manifest.proc_macro);
                deps.extend(implicit.iter().filter(|(name, _)| wanted(name)).cloned());
            }
            deps
        })
        .collect()
}

/// Creates the outline of every root, with the build scripts of the units `scripts`, with only the root modules declared; `load` reads modules as lookups need them.
pub fn new(workspace: &Workspace, targets: &Targets, scripts: &[usize]) -> Outline {
    let roots = roots(workspace, scripts);
    let deps = dependencies(workspace, &roots);
    let crates = roots.iter().map(|root| root_crate(workspace, targets, root)).collect();
    Outline { roots, crates, deps, cache: Cache::default(), procs: HashMap::default() }
}

fn root_crate(workspace: &Workspace, targets: &Targets, root: &Root) -> Crate {
    let unit = &workspace.graph.units[root.unit];
    let script = root.role == Role::Script;
    let known = script || unit.script.is_some() || workspace.packages[unit.package].manifest.build.is_none();
    let side = if script { crate::graph::Side::Host } else { unit.side };
    let env = Env { target: targets.of(side).clone(), features: unit.features.clone(), script: unit.script.clone().filter(|_| !script), known };
    Crate { env, modules: vec![root_module(&root.path)], ..Crate::default() }
}

fn root_module(path: &std::path::Path) -> Module {
    let dir = path.parent().map(PathBuf::from).unwrap_or_default();
    let decl = Decl::File { candidates: vec![path.to_owned()], by_attr: true };
    Module { file: usize::MAX, item: None, parent: None, name: String::new(), dir, parts: Vec::new(), vis: Vis::Pub, macro_use: false, state: State::Declared(decl), textual: None }
}

impl Outline {
    /// The outline of the single crate whose root file is `path`, with no dependencies.
    pub fn file(path: &std::path::Path, target: crate::graph::cfg::Target) -> Outline {
        let root = Root { unit: 0, name: String::new(), path: path.to_owned(), role: Role::Lib };
        let krate = Crate { env: Env { target, ..Env::default() }, modules: vec![root_module(path)], ..Crate::default() };
        Outline { roots: vec![root], crates: vec![krate], deps: vec![Vec::new()], cache: Cache::default(), procs: HashMap::default() }
    }

    /// The top-level `fn main` in the root file of crate `krate`.
    pub fn main(&self, krate: usize) -> Option<crate::names::ItemRef> {
        let tree = &self.crates[krate];
        let (file_at, file) = tree.files.iter().enumerate().find(|(_, file)| !file.expanded && file.source.path == self.roots[krate].path)?;
        let is_main = |item: &item::Item| item.kind == ItemKind::Fn && item.parent.is_none() && item.name.is_some_and(|name| file.source.text(name) == "main");
        let at = file.items.items.iter().position(is_main)?;
        Some(crate::names::ItemRef { krate: krate as u32, file: file_at as u32, item: at as u32 })
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

    /// Reads crate `krate` again from its root after its files changed outside function bodies; no other crate may depend on it.
    pub fn reload(&mut self, krate: usize) -> Result<()> {
        if self.deps.iter().any(|deps| deps.iter().any(|(_, at)| *at == krate)) {
            anyhow::bail!("crate {} has dependents", self.roots[krate].name);
        }
        let paths: Vec<PathBuf> = self.crates[krate].files.iter().map(|file| file.source.path.clone()).collect();
        self.cache.forget(&paths);
        let env = std::mem::take(&mut self.crates[krate].env);
        self.crates[krate] = Crate { env, modules: vec![root_module(&self.roots[krate].path)], ..Crate::default() };
        self.load_crate(krate)
    }

    /// Reads every module of crate `krate`, as a lookup over all of a crate's impls needs.
    pub fn load_crate(&mut self, krate: usize) -> Result<()> {
        while self.crates[krate].partial() {
            self.settle(vec![krate])?;
        }
        Ok(())
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
        self.with_loader(krate, &mut demands, |loader| calls.into_iter().try_for_each(|(call, definition)| loader.expand(call, &definition)))?;
        self.settle(demands)
    }

    /// Reads every module of the crates whose exports may define a still-deferred macro call, once name resolution can bind nothing more; returns whether it read any.
    pub fn fall_back(&mut self) -> Result<bool> {
        let mut demands = Vec::new();
        for krate in 0..self.crates.len() {
            if !self.crates[krate].deferred.is_empty() || !self.crates[krate].members.is_empty() {
                self.with_loader(krate, &mut demands, |loader| loader.retry())?;
            }
        }
        let waiting = self.crates.iter().filter(|krate| !krate.deferred.is_empty() || !krate.members.is_empty());
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

    /// Adds to crate `krate` the items that procedural macros wrote, each `(module, source, macro name)`, and reads what they declare.
    pub fn insert(&mut self, krate: usize, outputs: Vec<(usize, source::Source, String)>) -> Result<()> {
        let mut demands = Vec::new();
        self.with_loader(krate, &mut demands, |loader| outputs.into_iter().try_for_each(|(module, source, name)| loader.insert(module, source, &name)))?;
        self.settle(demands)
    }

    /// Takes item `at` of `file` in crate `krate` out of the crate, as an attribute macro does with the item it replaces.
    pub fn consume(&mut self, krate: usize, file: &Arc<tree::File>, at: u32) {
        let holder = &mut self.crates[krate];
        let Some(index) = holder.files.iter().position(|entry| Arc::ptr_eq(entry, file)) else { return };
        holder.consumed.insert((index as u32, at));
        holder.impls.retain(|block| (block.file, block.at) != (index, at));
    }

    /// The crates that depend on crate `krate`, directly or through others, with `krate` itself.
    pub fn dependents(&self, krate: usize) -> Vec<usize> {
        let mut found = vec![krate];
        let mut at = 0;
        while let Some(&next) = found.get(at) {
            let users = (0..self.deps.len()).filter(|user| self.deps[*user].iter().any(|(_, dep)| *dep == next));
            let fresh: Vec<usize> = users.filter(|user| !found.contains(user)).collect();
            found.extend(fresh);
            at += 1;
        }
        found
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
            let crates = &self.crates;
            let declared: Vec<(usize, usize)> = demands.iter().flat_map(|&krate| (0..crates[krate].modules.len()).filter(move |at| matches!(crates[krate].modules[*at].state, State::Declared(_))).map(move |at| (krate, at))).collect();
            self.read_ahead(&declared);
            for (krate, module) in declared {
                self.load_one(krate, module, &mut next)?;
            }
            for krate in 0..self.crates.len() {
                if !self.crates[krate].deferred.is_empty() || !self.crates[krate].members.is_empty() {
                    self.with_loader(krate, &mut next, |loader| loader.retry())?;
                }
            }
            demands = next;
        }
        Ok(())
    }
}
