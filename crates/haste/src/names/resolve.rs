mod import;
mod macros;

pub(crate) use import::visible;
pub use macros::{macro_at, macro_path};

use anyhow::Result;

use crate::graph::Workspace;
use crate::outline::Outline;

use super::collect::{self, Context, ImportGuard};
use super::path::{Paths, Requests};
use super::{Crate, Def, ModRef, Names, Scope, Symbols};
use import::{apply, sweep_crate_imports};

pub(super) fn paths_of<'a>(
    names: &'a Names,
    outline: &'a Outline,
    krate: usize,
    demands: &'a Requests,
) -> Paths<'a> {
    Paths {
        demands: Some(demands),
        ..names.paths(outline, krate as u32)
    }
}

fn declared(outline: &Outline, krate: usize) -> impl Iterator<Item = ModRef> + '_ {
    let modules = outline.crates[krate].modules.iter().enumerate();
    let declared = modules
        .filter(|(_, module)| matches!(module.state, crate::outline::tree::State::Declared(_)));
    declared.map(move |(module, _)| ModRef {
        krate: krate as u32,
        module: module as u32,
    })
}

impl Names {
    /// Empty scopes for every crate of `outline`, seeded with any cached sysroot crates.
    pub(crate) fn new_with(
        workspace: &Workspace,
        outline: &Outline,
        cached: Option<(Symbols, Vec<(usize, Crate)>)>,
    ) -> Names {
        let (symbols, sysroot) = cached.unwrap_or_default();
        let mut by_krate: Vec<Option<Crate>> = (0..outline.crates.len()).map(|_| None).collect();
        for (krate, entry) in sysroot {
            if krate < by_krate.len() {
                by_krate[krate] = Some(entry);
            }
        }
        let crates = outline
            .roots
            .iter()
            .enumerate()
            .map(|(krate, root)| {
                if let Some(entry) = by_krate[krate].take() {
                    return entry;
                }
                let manifest =
                    &workspace.packages[workspace.graph.units[root.unit].package].manifest;
                let deps = outline.deps[krate]
                    .iter()
                    .map(|(name, at)| (symbols.intern(name), *at))
                    .collect();
                Crate {
                    edition_2015: manifest.edition == "2015",
                    edition: manifest.edition.clone(),
                    deps,
                    ..Crate::default()
                }
            })
            .collect();
        Names { crates, symbols }
    }

    /// Forgets what crate `krate` resolved, keeping its edition and dependencies, so that `resolve` resolves it again from its reloaded outline.
    pub fn reset(&mut self, krate: usize) {
        let old = &self.crates[krate];
        self.crates[krate] = Crate {
            edition_2015: old.edition_2015,
            edition: old.edition.clone(),
            deps: old.deps.clone(),
            eager: old.eager,
            ..Crate::default()
        };
    }

    /// Collects every loaded module not collected yet and resolves imports until nothing changes, loading the modules that lookups reach; each pass resolves again only the imports of crates whose own scopes, or the scopes of a crate they depend on, changed.
    pub fn resolve(&mut self, outline: &mut Outline) -> Result<()> {
        let users = outline.users();
        let mut changed: Vec<bool> = self.crates.iter().map(|entry| !entry.settled).collect();
        loop {
            self.collect(outline, &mut changed);
            let demands = Requests::default();
            loop {
                let mut sweep = vec![false; self.crates.len()];
                crate::outline::dependents_of(
                    &users,
                    &mut sweep,
                    (0..changed.len()).filter(|&krate| changed[krate]),
                );
                changed = self.pass(outline, &demands, &sweep);
                if self.want(demands.names.take()) | self.spread_whole() {
                    for (flag, entry) in changed.iter_mut().zip(&self.crates) {
                        *flag = !entry.settled;
                    }
                } else if !changed.contains(&true) {
                    break;
                }
            }
            self.preludes(outline, &demands, &mut changed);
            let expanded = self.expand_imported(outline, &demands)?;
            let wanted = self.want(demands.names.take());
            if wanted {
                for (flag, entry) in changed.iter_mut().zip(&self.crates) {
                    *flag = !entry.settled;
                }
            }
            let crates = demands.crates.take();
            let mut demands = demands.modules.into_inner();
            demands.extend(
                crates
                    .into_iter()
                    .flat_map(|krate| declared(outline, krate as usize)),
            );
            for krate in (0..outline.crates.len())
                .filter(|krate| self.crates[*krate].eager && !self.crates[*krate].settled)
            {
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
            let modules: Vec<(usize, usize)> = demands
                .iter()
                .map(|module| (module.krate as usize, module.module as usize))
                .collect();
            outline.load_many(&modules)?;
        }
    }

    fn collect(&mut self, outline: &Outline, collected: &mut [bool]) {
        let symbols = &self.symbols;
        for (krate, tree) in outline.crates.iter().enumerate() {
            let entry = &mut self.crates[krate];
            if entry.settled {
                continue;
            }
            entry.scopes.resize_with(tree.modules.len(), Scope::default);
            entry.collected.resize(tree.modules.len(), None);
            if tree
                .modules
                .iter()
                .zip(&entry.collected)
                .all(|(module, seen)| !module.loaded() || *seen == Some(module.parts.len()))
            {
                continue;
            }
            let mut children = vec![Vec::new(); tree.modules.len()];
            for (index, module) in tree.modules.iter().enumerate() {
                if let Some(parent) = module.parent {
                    children[parent].push(index);
                }
            }
            let context = Context {
                krate: krate as u32,
                env: &tree.env,
                deps: &outline.deps[krate],
                symbols,
                children: &children,
            };
            let mut changed = false;
            for (index, module) in tree.modules.iter().enumerate() {
                let seen = entry.collected[index];
                if !module.loaded() || seen == Some(module.parts.len()) {
                    continue;
                }
                let own = seen.is_none().then_some((module.file as u32, module.item));
                let parts = module.parts[seen.unwrap_or(0)..]
                    .iter()
                    .map(|&part| (part as u32, None));
                let files: Vec<(u32, Option<u32>)> = own.into_iter().chain(parts).collect();
                let found = collect::collect_module(tree, index, &files, &context, entry);
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

    fn pass(&mut self, outline: &Outline, demands: &Requests, sweep: &[bool]) -> Vec<bool> {
        let swept: Vec<usize> = (0..self.crates.len())
            .filter(|&krate| sweep[krate] && !self.crates[krate].settled)
            .collect();
        let mut found = Vec::new();
        let mut unresolved = vec![Vec::new(); self.crates.len()];
        if swept.len() >= 2 && crate::jobs() > 1 {
            let results = crate::sync::parallel_map(swept, |krate| {
                let local_asks = Requests::default();
                let out = sweep_crate_imports(self, outline, krate, &local_asks);
                (out, local_asks.into_parts())
            });
            for ((krate, local_found, local_unresolved), parts) in results {
                demands.extend_parts(parts);
                found.extend(local_found);
                unresolved[krate] = local_unresolved;
            }
        } else {
            for krate in swept {
                let (_, local_found, local_unresolved) =
                    sweep_crate_imports(self, outline, krate, demands);
                found.extend(local_found);
                unresolved[krate] = local_unresolved;
            }
        }
        let mut changed = vec![false; self.crates.len()];
        let mut refresh_extern = Vec::new();
        for (krate, index, bindings, stamp, guard) in found {
            let entry = &mut self.crates[krate];
            let pending = &mut entry.pending[index];
            let moved = pending.name.is_none()
                && pending.stamp.map(|(target, _)| target) != stamp.map(|(target, _)| target);
            pending.stamp = stamp;
            pending.guard = guard;
            if matches!(guard, ImportGuard::Extern(..)) && !pending.absolute {
                refresh_extern.push((krate, index));
            }
            let module = pending.module;
            changed[krate] |= apply(&mut entry.scopes, module, bindings);
            if let (true, Some((Def::Module(target), _))) = (moved, stamp) {
                let asked = entry
                    .wanted_in(module)
                    .into_iter()
                    .map(|name| (target, name));
                demands.names.borrow_mut().extend(asked);
            }
        }
        for (krate, index) in refresh_extern {
            let entry = &mut self.crates[krate];
            let pending = &mut entry.pending[index];
            if let ImportGuard::Extern(_, externs_len) = pending.guard {
                let ver = entry.scopes[pending.module as usize].version();
                if pending.segments.first().is_some_and(|first| {
                    entry.scopes[pending.module as usize]
                        .names
                        .get(first)
                        .is_none_or(|e| e.types.is_none())
                }) {
                    pending.guard = ImportGuard::Extern(ver, externs_len);
                } else {
                    pending.guard = ImportGuard::None;
                }
            }
        }
        for ((entry, unresolved), _) in self
            .crates
            .iter_mut()
            .zip(unresolved)
            .zip(sweep)
            .filter(|(_, swept)| **swept)
        {
            entry.unresolved = unresolved;
        }
        changed
    }
}
