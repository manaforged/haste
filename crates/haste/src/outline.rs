pub mod attr;
pub mod build;
pub mod derive;
pub mod edit;
pub mod fragment;
mod graph;
mod include;
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

use crate::graph::Workspace;
use build::{Loader, Others};
use graph::root_module;
use include::resolve_macro_includes;
use item::ItemKind;
use tree::{Crate, Env, Files, Macro};

pub(crate) use graph::build_closure;
pub use graph::{dependents_of, new};

/// What a crate is to its package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Lib,
    Bin,
    Example,
    Test,
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

fn take_found<T>(waiting: &mut Vec<T>, found: Vec<(usize, Arc<Macro>)>) -> Vec<(T, Arc<Macro>)> {
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

impl Outline {
    /// Whether crate `krate` is an immutable release sysroot crate.
    pub fn is_sysroot(&self, krate: u32) -> bool {
        self.crates
            .get(krate as usize)
            .is_some_and(tree::Crate::is_sysroot)
    }

    /// Whether crate `krate` is an immutable sysroot or pinned registry/git crate.
    pub fn is_fixed(&self, krate: u32) -> bool {
        self.crates
            .get(krate as usize)
            .is_some_and(tree::Crate::is_fixed)
    }

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
        let root = Root {
            unit: 0,
            name: String::new(),
            path: path.to_owned(),
            role: Role::Lib,
        };
        let krate = Crate {
            env: Env {
                target,
                ..Env::default()
            },
            modules: vec![root_module(path)],
            ..Crate::default()
        };
        Outline {
            roots: vec![root],
            crates: vec![krate],
            deps: vec![Vec::new()],
            cache: Files::default(),
            procs: HashMap::default(),
            pins: vec![None],
        }
    }

    /// The top-level `fn main` that the cfgs of crate `krate` keep in its root file, or that an attribute macro on it wrote in its place.
    pub fn main(&self, krate: usize) -> Option<crate::names::ItemRef> {
        let tree = &self.crates[krate];
        let root = tree
            .files
            .iter()
            .position(|file| !file.expanded && file.source.path == self.roots[krate].path)?;
        let parts = tree
            .modules
            .iter()
            .find(|module| module.file == root && module.item.is_none())
            .map_or(&[][..], |module| &module.parts[..]);
        std::iter::once(root)
            .chain(parts.iter().copied())
            .find_map(|file_at| {
                let file = &tree.files[file_at];
                let is_main = |item: &item::Item| {
                    item.kind == ItemKind::Fn
                        && item.parent.is_none()
                        && item
                            .name
                            .is_some_and(|name| file.source.text(name) == "main")
                        && tree.env.enabled(&item.attrs)
                };
                let at = (0..file.items.items.len()).find(|&at| {
                    is_main(&file.items.items[at])
                        && !tree.consumed.contains(&(file_at as u32, at as u32))
                })?;
                Some(crate::names::ItemRef {
                    krate: krate as u32,
                    file: file_at as u32,
                    item: at as u32,
                })
            })
    }

    /// Reads module `module` of crate `krate` if it is only declared, then expands what that made resolvable.
    pub fn load(&mut self, krate: usize, module: usize) -> Result<()> {
        self.load_one(krate, module)
    }

    /// Reads each listed `(crate, module)` that is only declared, then expands what that made resolvable.
    pub fn load_many(&mut self, modules: &[(usize, usize)]) -> Result<()> {
        self.read_ahead(modules);
        modules
            .iter()
            .try_for_each(|&(krate, module)| self.load_one(krate, module))
    }

    /// Reads every module of every crate, level by level of each module tree; with more than one job, crates load on several threads to the same result.
    pub fn load_all(&mut self) -> Result<()> {
        self.load_selected(&vec![true; self.crates.len()])
    }

    /// Reads every module of each crate where `selected[krate]` is true, level by level; with more than one job, crates load on several threads.
    pub fn load_selected(&mut self, selected: &[bool]) -> Result<()> {
        let count = selected.iter().filter(|&&on| on).count();
        let workers = crate::jobs().min(count);
        if workers > 1 {
            return parallel::load_selected(self, workers, selected);
        }
        while self
            .crates
            .iter()
            .enumerate()
            .any(|(at, tree)| selected.get(at).copied().unwrap_or(false) && tree.partial())
        {
            let modules: Vec<(usize, usize)> = self
                .crates
                .iter()
                .enumerate()
                .filter(|(at, _)| selected.get(*at).copied().unwrap_or(false))
                .flat_map(|(krate, tree)| tree.declared().map(move |at| (krate, at)))
                .collect();
            self.load_many(&modules)?;
        }
        Ok(())
    }

    /// Reads crate `krate` and every crate that depends on it again from their roots after files of `krate` changed outside function bodies, and returns those crates.
    pub fn reload(&mut self, krate: usize) -> Result<Vec<usize>> {
        let mut seen = vec![false; self.deps.len()];
        let crates = dependents_of(&self.users(), &mut seen, [krate]);
        let paths: Vec<PathBuf> = crates
            .iter()
            .flat_map(|&at| {
                self.crates[at]
                    .files
                    .iter()
                    .map(|file| file.source.path.clone())
            })
            .collect();
        self.cache.forget(&paths);
        for &at in &crates {
            let env = std::mem::take(&mut self.crates[at].env);
            self.crates[at] = Crate {
                env,
                modules: vec![root_module(&self.roots[at].path)],
                ..Crate::default()
            };
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
        self.with_loader(krate, |loader| {
            calls
                .into_iter()
                .try_for_each(|(call, definition)| loader.expand(call, &definition))
        })
    }

    /// Expands the waiting macro calls in impl or trait bodies of crate `krate` at the given indexes with the definitions that name resolution found for them.
    pub fn expand_members(&mut self, krate: usize, found: Vec<(usize, Arc<Macro>)>) -> Result<()> {
        let calls = take_found(&mut self.crates[krate].members, found);
        self.with_loader(krate, |loader| {
            calls
                .into_iter()
                .try_for_each(|((block, call), definition)| {
                    loader.expand_member(block, call, &definition)
                })
        })
    }

    /// Reads every module of the crates whose exports may define a still-deferred macro call, once name resolution can bind nothing more; returns whether it read any.
    pub fn fall_back(&mut self) -> Result<bool> {
        self.retry_waiting()?;
        let waiting = self
            .crates
            .iter()
            .filter(|krate| !krate.deferred.is_empty() || !krate.members.is_empty());
        let mut fallback: Vec<usize> = waiting
            .flat_map(|krate| krate.fallback.iter().copied())
            .collect();
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
    pub fn insert(
        &mut self,
        krate: usize,
        outputs: Vec<(usize, source::Source, String)>,
    ) -> Result<()> {
        self.with_loader(krate, |loader| {
            outputs
                .into_iter()
                .try_for_each(|(module, source, name)| loader.insert(module, source, &name))
        })
    }

    /// Takes item `at` of `file` in crate `krate` out of the crate, as an attribute macro does with the item it replaces.
    pub fn consume(&mut self, krate: usize, file: &Arc<tree::File>, at: u32) {
        let holder = &mut self.crates[krate];
        let Some(index) = holder
            .files
            .iter()
            .position(|entry| Arc::ptr_eq(entry, file))
        else {
            return;
        };
        holder.consumed.insert((index as u32, at));
        holder
            .impls
            .retain(|block| (block.file, block.at) != (index, at));
        holder
            .traits
            .retain(|block| (block.file, block.at) != (index, at));
    }

    /// The crates that depend on crate `krate`, directly or through others, with `krate` itself.
    pub fn dependents(&self, krate: usize) -> Vec<usize> {
        let mut seen = vec![false; self.deps.len()];
        dependents_of(&self.users(), &mut seen, [krate])
    }

    /// The crates whose package is a member of `workspace`.
    pub fn members<'o>(&'o self, workspace: &'o Workspace) -> impl Iterator<Item = usize> + 'o {
        (0..self.roots.len()).filter(|&krate| {
            workspace
                .graph
                .members
                .contains(&workspace.graph.units[self.roots[krate].unit].package)
        })
    }

    /// The package that crate `krate` belongs to.
    pub fn package<'w>(&self, workspace: &'w Workspace, krate: usize) -> &'w crate::graph::Package {
        &workspace.packages[workspace.graph.units[self.roots[krate].unit].package]
    }

    /// The outline errors and missing module files of every crate whose package is local, where rustc would stop the build.
    pub fn local_errors(&self, workspace: &Workspace) -> Vec<String> {
        let local = self
            .crates
            .iter()
            .enumerate()
            .filter(|(krate, _)| self.package(workspace, *krate).local());
        let mut found: Vec<String> = local
            .flat_map(|(_, tree)| {
                let missing = tree.missing.chunks(2).map(|tried| {
                    format!(
                        "file not found for module: tried {}",
                        tried
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>()
                            .join(" and ")
                    )
                });
                tree.errors.iter().cloned().chain(missing)
            })
            .collect();
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
            if indexes.binary_search(&index).is_ok() {
                holder.foreign.push(call);
            } else {
                holder.deferred.push(call);
            }
        }
    }

    fn with_loader<T>(
        &mut self,
        krate: usize,
        run: impl FnOnce(&mut Loader<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut holder = std::mem::take(&mut self.crates[krate]);
        let mut loader = Loader::new(
            krate,
            &self.cache,
            Others::Crates(&self.crates),
            &self.deps[krate],
            &mut holder,
        );
        let result = run(&mut loader);
        self.crates[krate] = holder;
        result
    }

    fn read_ahead(&self, modules: &[(usize, usize)]) {
        let paths: Vec<PathBuf> = modules
            .iter()
            .flat_map(|&(krate, module)| self.crates[krate].unread_files(module))
            .cloned()
            .collect();
        self.cache.read_all(&paths);
    }

    fn load_one(&mut self, krate: usize, module: usize) -> Result<()> {
        let mut holder = std::mem::take(&mut self.crates[krate]);
        let others = Others::Crates(&self.crates);
        let result = Loader::new(krate, &self.cache, others, &self.deps[krate], &mut holder)
            .load_module(module)
            .and_then(|()| {
                resolve_macro_includes(
                    krate,
                    &self.cache,
                    others,
                    &self.deps[krate],
                    &mut holder,
                    module,
                )
            });
        self.crates[krate] = holder;
        result
    }

    fn load_crates(&mut self, mut crates: Vec<usize>) -> Result<()> {
        crates.sort_unstable();
        crates.dedup();
        let trees = &self.crates;
        let declared: Vec<(usize, usize)> = crates
            .iter()
            .flat_map(|&krate| trees[krate].declared().map(move |at| (krate, at)))
            .collect();
        self.load_many(&declared)?;
        self.retry_waiting()
    }

    fn retry_waiting(&mut self) -> Result<()> {
        for krate in 0..self.crates.len() {
            if !self.crates[krate].deferred.is_empty() || !self.crates[krate].members.is_empty() {
                self.with_loader(krate, |loader| loader.retry())?;
            }
        }
        Ok(())
    }
}
