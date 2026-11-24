use crate::outline::Outline;
use crate::outline::item::Vis;

use super::super::collect::{ImportGuard, Pending};
use super::super::path::{Path, Paths, Requests};
use super::super::{
    Binding, Def, ModRef, Names, Ns, Origin, PerNs, SELF, Scope, Sym, UNDERSCORE, Unresolved,
};
use super::paths_of;

pub(super) type Found = Vec<(Sym, Ns, Binding)>;

enum Outcome {
    Missing,
    Unchanged,
    Bound(Found, Option<(Def, u32)>, ImportGuard),
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
        Vis::Private | Vis::SelfMod => paths.within(importer, home),
        Vis::Super => paths.within(importer, paths.parent(home).unwrap_or(home)),
        Vis::Crate | Vis::In(_) => true,
    }
}

fn glob(paths: &Paths<'_>, target: Def, (at, vis): (ModRef, Vis), since: u32) -> Found {
    let mut found = Found::new();
    match target {
        Def::Module(home) => {
            let Some(scope) = paths.scope(home) else {
                return found;
            };
            let names: Box<dyn Iterator<Item = &Sym>> = if since == 0 {
                Box::new(scope.names.keys())
            } else {
                Box::new(scope.log[since as usize..].iter())
            };
            for name in names {
                let entry = &scope.names[name];
                for ns in NAMESPACES {
                    let Some(binding) = entry.get(ns) else {
                        continue;
                    };
                    if visible(paths, binding.vis, home, at) {
                        found.push((
                            *name,
                            ns,
                            Binding {
                                def: binding.def,
                                vis,
                                origin: Origin::Glob,
                            },
                        ));
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

fn guard_valid(paths: &Paths<'_>, pending: &Pending, at: ModRef) -> bool {
    let Some((target, ver)) = pending.stamp else {
        return false;
    };
    match pending.guard {
        ImportGuard::None => return false,
        ImportGuard::Solid => {}
        ImportGuard::Extern(local_ver, externs_len) => {
            if paths.externs.len() as u32 != externs_len {
                return false;
            }
            if !pending.absolute {
                let Some(scope) = paths.scope(at) else {
                    return false;
                };
                if scope.version() != local_ver
                    && pending.segments.first().is_none_or(|first| {
                        scope.names.get(first).is_some_and(|e| e.types.is_some())
                    })
                {
                    return false;
                }
            }
        }
    }
    match target {
        Def::Item(_) => true,
        Def::Module(home) => paths
            .scope(home)
            .is_some_and(|scope| scope.version() == ver),
        Def::Variant(..) | Def::Primitive(_) => false,
    }
}

fn import(paths: &Paths<'_>, pending: &Pending) -> Outcome {
    let at = ModRef {
        krate: paths.krate,
        module: pending.module,
    };
    if pending.guard != ImportGuard::None && guard_valid(paths, pending, at) {
        return Outcome::Unchanged;
    }
    let path = paths.use_path(Path {
        segments: &pending.segments,
        absolute: pending.absolute,
        dollar: pending.dollar,
    });
    let Some(name) = pending.name else {
        let Some((target, guard)) = paths.resolve_guarded(&path, at) else {
            return Outcome::Missing;
        };
        let stamp = stamp(paths, target);
        if stamp.is_some() && stamp == pending.stamp {
            return Outcome::Unchanged;
        }
        let since = pending
            .stamp
            .filter(|(seen, _)| *seen == target)
            .map_or(0, |(_, version)| version);
        return Outcome::Bound(glob(paths, target, (at, pending.vis), since), stamp, guard);
    };
    let Some((last, prefix)) = path.segments.split_last() else {
        return Outcome::Missing;
    };
    let (found, home, target, guard) = if prefix.is_empty() {
        (paths.first(*last, at, &path), None, None, ImportGuard::None)
    } else {
        let Some((target, guard)) = paths.resolve_guarded(
            &Path {
                segments: prefix,
                ..path
            },
            at,
        ) else {
            return Outcome::Missing;
        };
        if *last == SELF {
            let binding = Binding {
                def: target,
                vis: Vis::Pub,
                origin: Origin::Item,
            };
            (
                PerNs {
                    types: Some(binding),
                    values: None,
                    macros: None,
                },
                None,
                None,
                ImportGuard::None,
            )
        } else {
            let st = stamp(paths, target);
            if name != UNDERSCORE && st.is_some() && st == pending.stamp {
                return Outcome::Unchanged;
            }
            let home = match target {
                Def::Module(home) => Some(home),
                _ => None,
            };
            (paths.member(target, *last), home, Some(target), guard)
        }
    };
    if name == UNDERSCORE {
        let Some(binding) = found.types else {
            return Outcome::Missing;
        };
        let stamp = Some((binding.def, 0));
        return if stamp == pending.stamp {
            Outcome::Unchanged
        } else {
            Outcome::Bound(Found::new(), stamp, ImportGuard::None)
        };
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
            out.push((
                name,
                ns,
                Binding {
                    def: binding.def,
                    vis: pending.vis,
                    origin: Origin::Import,
                },
            ));
        }
    }
    if out.is_empty() {
        if let Some(Def::Module(root)) = target {
            paths.demand_macro_in(root, found);
        }
        Outcome::Missing
    } else {
        Outcome::Bound(
            out,
            stamp,
            if stamp.is_some() {
                guard
            } else {
                ImportGuard::None
            },
        )
    }
}

pub(super) fn apply(scopes: &mut [Scope], module: u32, found: Found) -> bool {
    let mut changed = false;
    for (name, ns, binding) in found {
        changed |= scopes[module as usize].bind(name, ns, binding);
    }
    changed
}

pub(super) type CrateSweep = (
    usize,
    Vec<(usize, usize, Found, Option<(Def, u32)>, ImportGuard)>,
    Vec<Unresolved>,
);

pub(super) fn sweep_crate_imports(
    names: &Names,
    outline: &Outline,
    krate: usize,
    asks: &Requests,
) -> CrateSweep {
    let paths = paths_of(names, outline, krate, asks);
    let mut local_found = Vec::new();
    let mut local_unresolved = Vec::new();
    for (index, pending) in names.crates[krate]
        .pending
        .iter()
        .enumerate()
        .filter(|(_, pending)| pending.active)
    {
        match import(&paths, pending) {
            Outcome::Bound(bindings, stamp, guard) => {
                local_found.push((krate, index, bindings, stamp, guard))
            }
            Outcome::Unchanged => {}
            Outcome::Missing => local_unresolved.push(Unresolved {
                file: pending.file,
                import: pending.import,
            }),
        }
    }
    (krate, local_found, local_unresolved)
}
