use std::sync::Arc;

use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::outline::Outline;
use crate::outline::item::{Attr, ItemKind};
use crate::outline::source::Source;
use crate::outline::tree::{Crate as Tree, File, Macro};

use super::super::path::{Path, Paths, Requests};
use super::super::{Binding, Def, ItemRef, ModRef, Names, Origin, Sym, plain};
use super::paths_of;

const PRELUDES: [(&str, &str); 4] = [
    ("2015", "rust_2015"),
    ("2018", "rust_2018"),
    ("2021", "rust_2021"),
    ("2024", "rust_2024"),
];

fn prelude(paths: &Paths<'_>, tree: &Tree, edition: &str) -> Option<ModRef> {
    if let Some(found) = own_prelude(paths, tree) {
        return Some(found);
    }
    let no_std = tree
        .files
        .get(tree.modules.first()?.file)?
        .items
        .inner
        .iter()
        .any(|attr| matches!(attr, Attr::NoStd | Attr::NoCore));
    let (core, std) = (paths.symbols.intern("core"), paths.symbols.intern("std"));
    let dep = |root| paths.deps.iter().find(|(name, _)| *name == root);
    let (_, krate) = if no_std {
        dep(core)
    } else {
        dep(std).or_else(|| dep(core))
    }?;
    let module = PRELUDES
        .iter()
        .find(|(year, _)| *year == edition)
        .map_or("rust_2021", |(_, m)| m);
    let prelude = paths
        .member(
            Def::Module(ModRef {
                krate: *krate as u32,
                module: 0,
            }),
            paths.symbols.intern("prelude"),
        )
        .types?
        .def;
    match paths
        .member(prelude, paths.symbols.intern(module))
        .types?
        .def
    {
        Def::Module(found) => Some(found),
        _ => None,
    }
}

fn own_prelude(paths: &Paths<'_>, tree: &Tree) -> Option<ModRef> {
    let file = tree.files.get(tree.modules.first()?.file)?;
    let src = &file.source;
    let marked = |item: &&crate::outline::item::Item| {
        item.attrs
            .iter()
            .any(|attr| matches!(attr, Attr::Other(at) if src.text(*at) == "prelude_import"))
    };
    let import = file
        .items
        .items
        .iter()
        .filter(|item| item.kind == ItemKind::Use && item.parent.is_none())
        .find(marked)?;
    let words = (import.start..import.end)
        .map(|at| src.text(at))
        .skip_while(|word| *word != "use")
        .skip(1);
    let mut def = Def::Module(ModRef {
        krate: paths.krate,
        module: 0,
    });
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
pub fn macro_at(
    paths: &Paths<'_>,
    src: &Source,
    (start, name): (u32, u32),
    at: ModRef,
) -> Option<ItemRef> {
    let dollar = if src.kind(start) == TokenKind::Dollar {
        src.dollar(start)
    } else {
        None
    };
    let words = (start..=name)
        .filter(|token| matches!(src.kind(*token), TokenKind::Ident | TokenKind::RawIdent));
    let segments: Vec<Sym> = words
        .map(|token| paths.symbols.intern(plain(src.text(token))))
        .collect();
    macro_path(
        paths,
        &Path {
            segments: &segments,
            absolute: src.kind(start) == TokenKind::Colon,
            dollar,
        },
        at,
    )
}

/// The macro that `path` names in module `at`; a crate root that names no such macro asks for its whole crate, since `#[macro_export]` puts a macro from any module there.
pub fn macro_path(paths: &Paths<'_>, path: &Path<'_>, at: ModRef) -> Option<ItemRef> {
    let (last, prefix) = path.segments.split_last()?;
    let binding = if prefix.is_empty() {
        let first = paths.first(*last, at, path);
        let found = first.macros.or_else(|| paths.macro_use(*last));
        if found.is_none()
            && let Some(Binding {
                def: Def::Module(submod),
                origin: Origin::Import | Origin::Glob,
                ..
            }) = first.types
        {
            paths.demand_macro_module(submod);
        }
        found?
    } else {
        let owner = paths.resolve(
            &Path {
                segments: prefix,
                ..*path
            },
            at,
        )?;
        let found = paths.member(owner, *last);
        if let (None, Def::Module(root)) = (found.macros, owner) {
            paths.demand_macro_in(root, found);
        }
        found.macros?
    };
    match binding.def {
        Def::Item(item) => Some(item),
        _ => None,
    }
}

impl Names {
    pub(super) fn expand_imported(
        &self,
        outline: &mut Outline,
        demands: &Requests,
    ) -> Result<bool> {
        let mut expanded = false;
        for krate in 0..outline.crates.len() {
            if self.crates[krate].settled {
                continue;
            }
            let (found, foreign) = self.imported_macros(outline, krate, demands, false);
            outline.set_foreign(krate, &foreign);
            if !found.is_empty() {
                let found = found
                    .into_iter()
                    .map(|(index, definition)| {
                        (
                            index - foreign.iter().filter(|at| **at < index).count(),
                            definition,
                        )
                    })
                    .collect();
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

    fn imported_macros(
        &self,
        outline: &Outline,
        krate: usize,
        demands: &Requests,
        members: bool,
    ) -> (Vec<(usize, Arc<Macro>)>, Vec<usize>) {
        let paths = paths_of(self, outline, krate, demands);
        let tree = &outline.crates[krate];
        let (mut found, mut foreign) = (Vec::new(), Vec::new());
        let check = |(index, call): (usize, &crate::outline::tree::Waiting)| {
            let item = resolve_macro(
                &paths,
                &call.file,
                (
                    call.at,
                    ModRef {
                        krate: krate as u32,
                        module: call.owner as u32,
                    },
                ),
            )?;
            Some((
                index,
                outline.crates[item.krate as usize]
                    .defined
                    .get(&(item.file, item.item))
                    .cloned(),
            ))
        };
        let resolved: Vec<_> = match members {
            true => tree
                .members
                .iter()
                .map(|(_, call)| call)
                .enumerate()
                .filter_map(check)
                .collect(),
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

    pub(super) fn preludes(&mut self, outline: &Outline, demands: &Requests, found: &mut [bool]) {
        for krate in 0..self.crates.len() {
            if self.crates[krate].settled
                || self.crates[krate].prelude.is_some()
                || !outline.crates[krate]
                    .modules
                    .first()
                    .is_some_and(|root| root.loaded())
            {
                continue;
            }
            let paths = paths_of(self, outline, krate, demands);
            self.crates[krate].prelude =
                prelude(&paths, &outline.crates[krate], &self.crates[krate].edition);
            found[krate] |= self.crates[krate].prelude.is_some();
        }
    }
}
