pub mod attr;
pub mod build;
pub mod derive;
pub mod edit;
pub mod fragment;
pub mod item;
pub mod mbe;
mod parallel;
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
use build::{Loader, Others};
use item::{ItemKind, Vis};
use tree::{Files, Crate, Decl, Env, Macro, Module, State};

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
    pub cache: Files,
    /// The output text of each procedural macro called in a body, by macro function and input text, or why it failed; dropped for a macro whose crate reloads.
    pub procs: HashMap<(crate::names::ItemRef, String), Result<String, String>>,
    /// For each root whose sources cannot change, a key of its path and of the resolved graph, under which what its procedural macros print for one input is kept.
    pub pins: Vec<Option<u64>>,
}

/// The library and member binary targets of every unit; with build scripts `scripts`, those scripts and only the libraries they build from.
pub fn roots(workspace: &Workspace, scripts: &[usize]) -> Vec<Root> {
    let mut roots = Vec::new();
    let needed = (!scripts.is_empty()).then(|| script_closure(workspace, scripts));
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let manifest = &workspace.packages[entry.package].manifest;
        if let Some(lib) = manifest.lib.as_ref().filter(|_| needed.as_ref().is_none_or(|needed| needed.contains(&unit))) {
            let name = manifest.lib_name.as_ref().unwrap_or(&manifest.name).replace('-', "_");
            let supplied = workspace.packages[entry.package].sysroot && name == "proc_macro";
            let path = if supplied { PathBuf::from(tree::PROC_MACRO_ROOT) } else { lib.clone() };
            roots.push(Root { unit, name, path, role: Role::Lib });
        }
        if needed.is_none() && workspace.graph.members.contains(&entry.package) {
            let bins = manifest.bins.iter().map(|(name, path)| Root { unit, name: name.clone(), path: path.clone(), role: Role::Bin });
            roots.extend(bins);
        }
        if let Some(path) = manifest.build.as_ref().filter(|_| scripts.contains(&unit)) {
            roots.push(Root { unit, name: "build_script_build".to_owned(), path: path.clone(), role: Role::Script });
        }
    }
    roots
}

/// The units whose libraries the build scripts of `scripts` build from: the sysroot and every library their build dependencies reach.
fn script_closure(workspace: &Workspace, scripts: &[usize]) -> rustc_hash::FxHashSet<usize> {
    let units = &workspace.graph.units;
    let sysroot = (0..units.len()).filter(|&unit| workspace.packages[units[unit].package].sysroot);
    let builds = scripts.iter().flat_map(|&unit| units[unit].deps.iter().filter(|(_, _, kind)| *kind == Kind::Build).map(|(_, dep, _)| *dep));
    let mut left: Vec<usize> = sysroot.chain(builds).collect();
    let mut needed = rustc_hash::FxHashSet::default();
    while let Some(unit) = left.pop() {
        if needed.insert(unit) {
            left.extend(units[unit].deps.iter().filter(|(_, _, kind)| *kind == Kind::Normal).map(|(_, dep, _)| *dep));
        }
    }
    needed
}

const IMPLICIT: [&str; 4] = ["std", "core", "alloc", "proc_macro"];

fn implicit(workspace: &Workspace, roots: &[Root]) -> Vec<(String, usize)> {
    let named = |name: &str| {
        let found = roots.iter().position(|root| root.role == Role::Lib && workspace.packages[workspace.graph.units[root.unit].package].sysroot && root.name == name);
        found.map(|at| (name.to_owned(), at))
    };
    IMPLICIT.iter().filter_map(|name| named(name)).collect()
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
                deps.extend(implicit.iter().cloned());
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
    let pins = pins(workspace, &roots);
    Outline { roots, crates, deps, cache: Files::default(), procs: HashMap::default(), pins }
}

/// The key of each root from a registry or git source: its path, and every unit's package, version, source, features and side.
fn pins(workspace: &Workspace, roots: &[Root]) -> Vec<Option<u64>> {
    use std::hash::BuildHasher as _;
    let units: Vec<_> = workspace.graph.units.iter().map(|unit| (&workspace.packages[unit.package].locked, &unit.features, unit.side == crate::graph::Side::Host)).map(|(locked, features, host)| (&locked.name, locked.version.to_string(), &locked.source, features, host)).collect();
    let graph = rustc_hash::FxBuildHasher.hash_one(&units);
    roots.iter().map(|root| workspace.packages[workspace.graph.units[root.unit].package].locked.source.is_some().then(|| rustc_hash::FxBuildHasher.hash_one((graph, &root.path)))).collect()
}

fn root_crate(workspace: &Workspace, targets: &Targets, root: &Root) -> Crate {
    let unit = &workspace.graph.units[root.unit];
    let script = root.role == Role::Script;
    let known = script || unit.script.is_some() || workspace.packages[unit.package].manifest.build.is_none();
    let side = if script { crate::graph::Side::Host } else { unit.side };
    let vars = if script { crate::graph::script::package_vars(workspace, root.unit) } else { crate::graph::script::crate_vars(workspace, root.unit) };
    let env = Env { target: targets.of(side).clone(), features: unit.features.clone(), script: unit.script.clone().filter(|_| !script), known, vars, release: workspace.packages[unit.package].sysroot };
    Crate { env, modules: vec![root_module(&root.path)], ..Crate::default() }
}

/// Takes out of `waiting` the calls at the indexes of `found`, each with its definition, and keeps the rest waiting.
fn take_found(waiting: &mut Vec<tree::Waiting>, found: Vec<(usize, Arc<Macro>)>) -> Vec<(tree::Waiting, Arc<Macro>)> {
    let mut found = found.into_iter().peekable();
    let mut calls = Vec::new();
    for (index, call) in std::mem::take(waiting).into_iter().enumerate() {
        match found.next_if(|(at, _)| *at == index) {
            Some((_, definition)) => calls.push((call, definition)),
            None => waiting.push(call),
        }
    }
    calls
}

fn root_module(path: &std::path::Path) -> Module {
    let dir = path.parent().map(PathBuf::from).unwrap_or_default();
    let decl = Decl::File { candidates: vec![path.to_owned()], by_attr: true };
    Module { file: usize::MAX, item: None, parent: None, name: String::new(), dir, parts: Vec::new(), vis: Vis::Pub, macro_use: false, state: State::Declared(decl), textual: None }
}

impl Outline {
    /// The path that `module_path!` spells for module `at`: the crate name, then each module name from the root down.
    pub fn module_path(&self, at: crate::names::ModRef) -> String {
        let modules = &self.crates[at.krate as usize].modules;
        let mut names = Vec::new();
        let mut current = Some(at.module as usize);
        while let Some(module) = current.filter(|&module| module != 0) {
            names.push(modules[module].name.as_str());
            current = modules[module].parent;
        }
        names.push(&self.roots[at.krate as usize].name);
        names.reverse();
        names.join("::")
    }

    /// The outline of the single crate whose root file is `path`, with no dependencies.
    pub fn file(path: &std::path::Path, target: crate::graph::cfg::Target) -> Outline {
        let root = Root { unit: 0, name: String::new(), path: path.to_owned(), role: Role::Lib };
        let krate = Crate { env: Env { target, ..Env::default() }, modules: vec![root_module(path)], ..Crate::default() };
        Outline { roots: vec![root], crates: vec![krate], deps: vec![Vec::new()], cache: Files::default(), procs: HashMap::default(), pins: vec![None] }
    }

    /// The top-level `fn main` that the cfgs of crate `krate` keep in its root file.
    pub fn main(&self, krate: usize) -> Option<crate::names::ItemRef> {
        let tree = &self.crates[krate];
        let (file_at, file) = tree.files.iter().enumerate().find(|(_, file)| !file.expanded && file.source.path == self.roots[krate].path)?;
        let is_main = |item: &item::Item| item.kind == ItemKind::Fn && item.parent.is_none() && item.name.is_some_and(|name| file.source.text(name) == "main") && tree.env.enabled(&item.attrs);
        let at = file.items.items.iter().position(is_main)?;
        Some(crate::names::ItemRef { krate: krate as u32, file: file_at as u32, item: at as u32 })
    }

    /// Reads module `module` of crate `krate` if it is only declared, then expands what that made resolvable.
    pub fn load(&mut self, krate: usize, module: usize) -> Result<()> {
        self.load_one(krate, module)
    }

    /// Reads each listed `(crate, module)` that is only declared, then expands what that made resolvable.
    pub fn load_many(&mut self, modules: &[(usize, usize)]) -> Result<()> {
        self.read_ahead(modules);
        modules.iter().try_for_each(|&(krate, module)| self.load_one(krate, module))
    }

    /// Reads every module of every crate, level by level of each module tree; with more than one job, crates load on several threads to the same result.
    pub fn load_all(&mut self) -> Result<()> {
        let workers = crate::jobs().min(self.crates.len());
        if workers > 1 {
            return parallel::load_all(self, workers);
        }
        while self.crates.iter().any(Crate::partial) {
            let modules: Vec<(usize, usize)> = self.crates.iter().enumerate().flat_map(|(krate, tree)| tree.declared().map(move |at| (krate, at))).collect();
            self.load_many(&modules)?;
        }
        Ok(())
    }

    /// Reads crate `krate` and every crate that depends on it again from their roots after files of `krate` changed outside function bodies, and returns those crates.
    pub fn reload(&mut self, krate: usize) -> Result<Vec<usize>> {
        let mut seen = vec![false; self.deps.len()];
        let crates = dependents_of(&self.users(), &mut seen, [krate]);
        let paths: Vec<PathBuf> = crates.iter().flat_map(|&at| self.crates[at].files.iter().map(|file| file.source.path.clone())).collect();
        self.cache.forget(&paths);
        for &at in &crates {
            let env = std::mem::take(&mut self.crates[at].env);
            self.crates[at] = Crate { env, modules: vec![root_module(&self.roots[at].path)], ..Crate::default() };
        }
        self.procs.retain(|(run, _), _| !seen[run.krate as usize]);
        crates.iter().try_for_each(|&at| self.load_crate(at))?;
        Ok(crates)
    }

    /// Reads every module of crate `krate`, as a lookup over all of a crate's impls needs.
    pub fn load_crate(&mut self, krate: usize) -> Result<()> {
        while self.crates[krate].partial() {
            self.load_crates(vec![krate])?;
        }
        Ok(())
    }

    /// Expands the deferred calls of crate `krate` at the given indexes with the definitions that name resolution found for them.
    pub fn expand(&mut self, krate: usize, found: Vec<(usize, Arc<Macro>)>) -> Result<()> {
        let calls = take_found(&mut self.crates[krate].deferred, found);
        self.with_loader(krate, |loader| calls.into_iter().try_for_each(|(call, definition)| loader.expand(call, &definition)))
    }

    /// Expands the waiting macro calls in impl bodies of crate `krate` at the given indexes with the definitions that name resolution found for them.
    pub fn expand_members(&mut self, krate: usize, found: Vec<(usize, Arc<Macro>)>) -> Result<()> {
        let calls = take_found(&mut self.crates[krate].members, found);
        self.with_loader(krate, |loader| calls.into_iter().try_for_each(|(call, definition)| loader.expand_member(call, &definition)))
    }

    /// Reads every module of the crates whose exports may define a still-deferred macro call, once name resolution can bind nothing more; returns whether it read any.
    pub fn fall_back(&mut self) -> Result<bool> {
        self.retry_waiting()?;
        let waiting = self.crates.iter().filter(|krate| !krate.deferred.is_empty() || !krate.members.is_empty());
        let mut fallback: Vec<usize> = waiting.flat_map(|krate| krate.fallback.iter().copied()).collect();
        fallback.sort_unstable();
        fallback.dedup();
        fallback.retain(|krate| self.crates[*krate].partial());
        if fallback.is_empty() {
            return Ok(false);
        }
        self.load_crates(fallback)?;
        Ok(true)
    }

    /// Adds to crate `krate` the items that procedural macros wrote, each `(module, source, macro name)`, and reads what they declare.
    pub fn insert(&mut self, krate: usize, outputs: Vec<(usize, source::Source, String)>) -> Result<()> {
        self.with_loader(krate, |loader| outputs.into_iter().try_for_each(|(module, source, name)| loader.insert(module, source, &name)))
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
        let mut seen = vec![false; self.deps.len()];
        dependents_of(&self.users(), &mut seen, [krate])
    }

    /// The crates whose package is a member of `workspace`.
    pub fn members<'o>(&'o self, workspace: &'o Workspace) -> impl Iterator<Item = usize> + 'o {
        (0..self.roots.len()).filter(|&krate| workspace.graph.members.contains(&workspace.graph.units[self.roots[krate].unit].package))
    }

    /// The package that crate `krate` belongs to.
    pub fn package<'w>(&self, workspace: &'w Workspace, krate: usize) -> &'w crate::graph::Package {
        &workspace.packages[workspace.graph.units[self.roots[krate].unit].package]
    }

    /// The outline errors and missing module files of every crate whose package is local, where rustc would stop the build.
    pub fn local_errors(&self, workspace: &Workspace) -> Vec<String> {
        let local = self.crates.iter().enumerate().filter(|(krate, _)| self.package(workspace, *krate).local());
        let mut found: Vec<String> = local.flat_map(|(_, tree)| {
            let missing = tree.missing.chunks(2).map(|tried| format!("file not found for module: tried {}", tried.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(" and ")));
            tree.errors.iter().cloned().chain(missing)
        }).collect();
        found.sort();
        found.dedup();
        found
    }

    /// The crates that depend directly on each crate.
    pub fn users(&self) -> Vec<Vec<usize>> {
        let mut users = vec![Vec::new(); self.deps.len()];
        for (user, deps) in self.deps.iter().enumerate() {
            deps.iter().for_each(|&(_, dep)| users[dep].push(user));
        }
        users
    }

    /// Moves the deferred calls of crate `krate` at the given indexes, which name resolution bound to procedural macros, out of the calls left to expand.
    pub fn set_foreign(&mut self, krate: usize, indexes: &[usize]) {
        let holder = &mut self.crates[krate];
        let deferred = std::mem::take(&mut holder.deferred);
        for (index, call) in deferred.into_iter().enumerate() {
            if indexes.binary_search(&index).is_ok() { holder.foreign.push(call) } else { holder.deferred.push(call) }
        }
    }

    fn with_loader<T>(&mut self, krate: usize, run: impl FnOnce(&mut Loader<'_>) -> Result<T>) -> Result<T> {
        let mut holder = std::mem::take(&mut self.crates[krate]);
        let mut loader = Loader::new(krate, &self.cache, Others::Crates(&self.crates), &self.deps[krate], &mut holder);
        let result = run(&mut loader);
        self.crates[krate] = holder;
        result
    }

    fn read_ahead(&self, modules: &[(usize, usize)]) {
        let paths: Vec<PathBuf> = modules.iter().flat_map(|&(krate, module)| self.crates[krate].unread_files(module)).cloned().collect();
        self.cache.read_all(&paths);
    }

    fn load_one(&mut self, krate: usize, module: usize) -> Result<()> {
        self.with_loader(krate, |loader| loader.load_module(module))
    }

    /// Reads every declared module of `crates`, then retries the macro calls of every crate that still waits on one.
    fn load_crates(&mut self, mut crates: Vec<usize>) -> Result<()> {
        crates.sort_unstable();
        crates.dedup();
        let trees = &self.crates;
        let declared: Vec<(usize, usize)> = crates.iter().flat_map(|&krate| trees[krate].declared().map(move |at| (krate, at))).collect();
        self.load_many(&declared)?;
        self.retry_waiting()
    }

    /// Retries the deferred and impl-body macro calls of every crate that has some.
    fn retry_waiting(&mut self) -> Result<()> {
        for krate in 0..self.crates.len() {
            if !self.crates[krate].deferred.is_empty() || !self.crates[krate].members.is_empty() {
                self.with_loader(krate, |loader| loader.retry())?;
            }
        }
        Ok(())
    }
}

/// The crates `from` and the crates that depend on one of them, by the direct `users` of each crate, that `seen` does not hold yet; marks them in `seen`.
pub fn dependents_of(users: &[Vec<usize>], seen: &mut [bool], from: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut found: Vec<usize> = from.into_iter().filter(|&krate| !std::mem::replace(&mut seen[krate], true)).collect();
    let mut at = 0;
    while let Some(&next) = found.get(at) {
        for &user in &users[next] {
            if !std::mem::replace(&mut seen[user], true) {
                found.push(user);
            }
        }
        at += 1;
    }
    found
}
