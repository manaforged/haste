//! Binding the procedural macro calls that the outline holds back to the macro functions their paths name.

use crate::outline::Outline;
use crate::outline::item::ProcKind;
use crate::outline::source::Source;
use crate::outline::tree::{Held, Waiting};

use super::path::{Asks, Paths};
use super::resolve::{macro_at, paths_of};
use super::{ItemRef, ModRef, Names, collect};

/// Where a procedural macro call is written: a bang call item, one path of a `#[derive]`, or an attribute on an item.
pub enum Site {
    Bang(Waiting),
    Derive(Held),
    Attribute(Held),
}

/// A procedural macro call bound to the macro function `run` of kind `kind`, in crate `krate`.
pub struct Bound {
    pub krate: usize,
    pub run: ItemRef,
    pub kind: ProcKind,
    pub site: Site,
}

impl Site {
    /// The module the call is written in.
    pub fn module(&self) -> usize {
        match self {
            Site::Bang(call) => call.owner,
            Site::Derive(held) | Site::Attribute(held) => held.owner,
        }
    }

    fn path(&self) -> Option<(&Source, (u32, u32))> {
        match self {
            Site::Bang(call) => {
                let item = &call.file.items.items[call.at as usize];
                Some((&call.file.source, (item.start, item.name?)))
            }
            Site::Derive(held) | Site::Attribute(held) => Some((&held.file.source, held.path)),
        }
    }

    fn wanted(&self, kind: ProcKind) -> bool {
        match self {
            Site::Bang(_) => kind == ProcKind::Bang,
            Site::Derive(_) => kind == ProcKind::Derive,
            Site::Attribute(_) => kind == ProcKind::Attribute,
        }
    }
}

impl Names {
    /// Takes from the outline every held-back procedural macro call whose path names a macro function now; bang calls and derives that name none stay held back, and attributes that name none are inert.
    pub fn bind_procs(&self, outline: &mut Outline) -> Vec<Bound> {
        let mut bound = Vec::new();
        for krate in 0..outline.crates.len() {
            let holder = &mut outline.crates[krate];
            let bangs = std::mem::take(&mut holder.foreign).into_iter().map(Site::Bang);
            let derives = std::mem::take(&mut holder.derived).into_iter().map(Site::Derive);
            let sites: Vec<Site> = bangs.chain(derives).chain(std::mem::take(&mut holder.attributed).into_iter().map(Site::Attribute)).collect();
            let demands = Asks::default();
            let paths = paths_of(self, outline, krate, &demands);
            let (found, left): (Vec<_>, Vec<_>) = sites.into_iter().map(|site| (proc_at(outline, &paths, &site, krate), site)).partition(|(found, _)| found.is_some());
            bound.extend(found.into_iter().filter_map(|(found, site)| found.map(|(run, kind)| Bound { krate, run, kind, site })));
            drop(paths);
            for (_, site) in left {
                match site {
                    Site::Bang(call) => outline.crates[krate].foreign.push(call),
                    Site::Derive(held) => outline.crates[krate].derived.push(held),
                    Site::Attribute(_) => {}
                }
            }
        }
        bound
    }
}

fn proc_at(outline: &Outline, paths: &Paths<'_>, site: &Site, krate: usize) -> Option<(ItemRef, ProcKind)> {
    let (src, path) = site.path()?;
    let run = macro_at(paths, src, path, ModRef { krate: krate as u32, module: site.module() as u32 })?;
    if run.outer().is_some() {
        return None;
    }
    let file = &outline.crates[run.krate as usize].files[run.file as usize];
    let (_, kind) = collect::proc_kind(&file.source, &file.items.items[run.item as usize])?;
    site.wanted(kind).then_some((run, kind))
}
