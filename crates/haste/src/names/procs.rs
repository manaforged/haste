//! Binding the procedural macro calls that the outline holds back to the macro functions their paths name.

use std::sync::Arc;

use crate::outline::Outline;
use crate::outline::item::ProcKind;
use crate::outline::source::Source;
use crate::outline::tree::{File, Held, Waiting};

use super::path::{Paths, Requests};
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
    /// Takes from the outline every held-back procedural macro call whose path names a macro function now, with what the lookups asked to load; a call that names none stays held back while its lookup asked for something or it is a bang call or derive, and is inert otherwise.
    pub fn bind_procs(&self, outline: &mut Outline) -> (Vec<Bound>, Requests) {
        let (mut bound, demands) = (Vec::new(), Requests::default());
        for krate in 0..outline.crates.len() {
            let holder = &mut outline.crates[krate];
            let bangs = std::mem::take(&mut holder.foreign).into_iter().map(Site::Bang);
            let derives = std::mem::take(&mut holder.derived).into_iter().map(Site::Derive);
            let sites: Vec<Site> = bangs.chain(derives).chain(std::mem::take(&mut holder.attributed).into_iter().map(Site::Attribute)).collect();
            let paths = paths_of(self, outline, krate, &demands);
            let (mut left, start) = (Vec::new(), bound.len());
            for site in sites {
                let before = demands.count();
                match proc_at(outline, &paths, &site, krate) {
                    Some((run, kind)) => bound.push(Bound { krate, run, kind, site }),
                    None => left.push((demands.count() > before, site)),
                }
            }
            drop(paths);
            defer_derives(&mut bound, start, &mut left);
            left.into_iter().for_each(|(asked, site)| hold(outline, krate, asked, site));
        }
        (bound, demands)
    }
}

/// Leaves out each derive of crate calls `bound[start..]` and `left` whose item an attribute macro written before it rewrites, since the derive runs on that macro's output, and holds back the derives whose attribute is still being resolved.
fn defer_derives(bound: &mut Vec<Bound>, start: usize, left: &mut Vec<(bool, Site)>) {
    let ready = bound[start..].iter().map(|call| (&call.site, true));
    let pending = left.iter().filter(|(asked, _)| *asked).map(|(_, site)| (site, false));
    let attrs: Vec<(Arc<File>, u32, u32, bool)> = ready
        .chain(pending)
        .filter_map(|(site, ready)| match site {
            Site::Attribute(held) => Some((held.file.clone(), held.at, held.path.0, ready)),
            _ => None,
        })
        .collect();
    let over = |site: &Site| match site {
        Site::Derive(held) => attrs.iter().find(|(file, at, path, _)| Arc::ptr_eq(file, &held.file) && *at == held.at && *path < held.path.0).map(|entry| entry.3),
        _ => None,
    };
    left.retain(|(_, site)| over(site) != Some(true));
    let mut at = start;
    while at < bound.len() {
        match over(&bound[at].site) {
            Some(true) => drop(bound.remove(at)),
            Some(false) => left.push((true, bound.remove(at).site)),
            None => at += 1,
        }
    }
}

/// Puts back a call that names no macro function yet: a bang call or derive always, an attribute only while its lookup asked for something to load.
fn hold(outline: &mut Outline, krate: usize, asked: bool, site: Site) {
    let holder = &mut outline.crates[krate];
    match site {
        Site::Bang(call) => holder.foreign.push(call),
        Site::Derive(held) => holder.derived.push(held),
        Site::Attribute(held) if asked => holder.attributed.push(held),
        Site::Attribute(_) => {}
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
