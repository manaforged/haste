use anyhow::Result;

use super::collect::Pending;
use super::path::Requests;
use super::{Crate, Def, ModRef, Names, Sym};
use crate::outline::Outline;

impl Crate {
    /// Adds imports found by collection, active at once in an eager crate or where a lookup already asked for their name.
    pub fn add_pending(&mut self, found: Vec<Pending>) {
        for mut pending in found {
            let index = self.pending.len();
            match pending.name {
                Some(name) => self.named.entry((pending.module, name)).or_default().push(index),
                None => self.globs.entry(pending.module).or_default().push(index),
            }
            let asked = self.wanted.get(&pending.module).is_some_and(|names| pending.name.is_none_or(|name| names.contains(&name)));
            pending.active = asked || self.resolves_all(pending.module);
            self.pending.push(pending);
        }
    }

    /// The names that lookups asked for in `module`.
    pub fn wanted_in(&self, module: u32) -> Vec<Sym> {
        self.wanted.get(&module).map(|names| names.iter().copied().collect()).unwrap_or_default()
    }
}

impl Names {
    /// Records the names that lookups asked for, activates the imports that can bind them, and passes each name on through the globs of its module; returns whether anything was new.
    pub fn want(&mut self, mut asked: Vec<(ModRef, Sym)>) -> bool {
        let mut new = false;
        while let Some((at, name)) = asked.pop() {
            let Some(entry) = self.crates.get_mut(at.krate as usize) else {
                continue;
            };
            if !entry.wanted.entry(at.module).or_default().insert(name) {
                continue;
            }
            new = true;
            let named = entry.named.get(&(at.module, name)).cloned().unwrap_or_default();
            let globs = entry.globs.get(&at.module).cloned().unwrap_or_default();
            for index in named.into_iter().chain(globs.iter().copied()) {
                entry.pending[index].active = true;
            }
            let targets = globs.iter().filter_map(|index| match entry.pending[*index].stamp {
                Some((Def::Module(target), _)) => Some((target, name)),
                _ => None,
            });
            asked.extend(targets);
        }
        new
    }

    /// Activates every import of the modules `modules` and of the modules their globs read; returns whether anything was new.
    pub fn want_whole(&mut self, mut modules: Vec<ModRef>) -> bool {
        let mut new = false;
        while let Some(at) = modules.pop() {
            let Some(entry) = self.crates.get_mut(at.krate as usize) else {
                continue;
            };
            if entry.whole.insert(at.module) {
                new = true;
                entry.pending.iter_mut().filter(|pending| pending.module == at.module).for_each(|pending| pending.active = true);
            }
            let globs = entry.globs.get(&at.module).into_iter().flatten();
            let targets: Vec<ModRef> = globs
                .filter_map(|index| match entry.pending[*index].stamp {
                    Some((Def::Module(target), _)) => Some(target),
                    _ => None,
                })
                .collect();
            modules.extend(targets.into_iter().filter(|target| self.crates.get(target.krate as usize).is_some_and(|other| !other.whole.contains(&target.module))));
        }
        new
    }

    pub(super) fn spread_whole(&mut self) -> bool {
        let modules = self.crates.iter().enumerate().flat_map(|(krate, entry)| entry.whole.iter().map(move |&module| ModRef { krate: krate as u32, module }));
        let modules: Vec<ModRef> = modules.collect();
        self.want_whole(modules)
    }

    /// Loads the modules and resolves the names that later lookups asked for, then settles; returns whether anything was new.
    pub fn load(&mut self, outline: &mut Outline, asks: Requests) -> Result<bool> {
        let wanted = self.want(asks.names.into_inner()) | self.want_whole(asks.scopes.into_inner());
        let mut modules: Vec<(usize, usize)> = asks.modules.into_inner().iter().map(|module| (module.krate as usize, module.module as usize)).collect();
        modules.sort_unstable();
        modules.dedup();
        outline.load_many(&modules)?;
        let mut crates = asks.crates.into_inner();
        crates.sort_unstable();
        crates.dedup();
        crates.retain(|krate| outline.crates[*krate as usize].partial());
        for &krate in &crates {
            outline.load_crate(krate as usize)?;
        }
        self.resolve(outline)?;
        Ok(wanted || !modules.is_empty() || !crates.is_empty())
    }
}
