mod scan;

pub(super) use scan::{declared, parse_in, scan_nested};

use crate::names::collect::{extern_crate_target, import_parts, namespaces};
use crate::names::path::{Path as NamePath, Paths};
use crate::names::{Binding, Def, ItemRef, ModRef, Ns, Origin, PerNs, SELF, Sym, Symbols, plain};
use crate::outline::item::{Item, ItemKind, Vis};
use crate::outline::tree::{File, Index};

use super::Context;

/// Where a path written inside a body leads.
pub(super) enum BodyPath {
    /// A definition that the body declares or links through `extern crate`.
    Def(Def),
    /// An enum that the body declares, whose variant the last segment names.
    Variant(ItemRef),
    /// Out of the body's modules through `super`: the rest of the path, after this many segments, resolves from the home module.
    Outside(usize),
}

/// The rest of `path` once its first `skip` segments led out of the body's modules, to resolve from the home module.
pub(super) fn outside<'p>(path: &NamePath<'p>, skip: usize) -> NamePath<'p> {
    NamePath {
        segments: &path.segments[skip..],
        absolute: false,
        dollar: path.dollar,
    }
}

/// The namespace that the first of `len` path segments is looked up in, for a path that names something in `ns`.
pub(super) fn head_ns(len: usize, ns: Ns) -> Ns {
    if len == 1 { ns } else { Ns::Types }
}

/// Whether the item `entry` binds a name in namespace `ns`.
pub(super) fn in_namespace(entry: &Item, ns: Ns) -> bool {
    if matches!(entry.kind, ItemKind::Mod | ItemKind::ExternCrate) {
        ns == Ns::Types
    } else {
        namespaces(entry).contains(&ns)
    }
}

impl<'a> Context<'a> {
    fn body_entries(&self, root: ItemRef) -> &'a [Item] {
        self.expansion(root)
            .map_or(&[][..], |found| &found.file.items.items[..])
    }

    fn body_child(
        &self,
        root: ItemRef,
        level: Option<u32>,
        from: Option<u32>,
        name: &str,
        ns: Option<Ns>,
    ) -> Option<ItemRef> {
        let found = self.expansion(root)?;
        let items = &found.file.items.items;
        let held = self.body_index(&found.file).items.get(&level)?.get(name)?;
        let matches = |at: u32| ns.is_none_or(|ns| in_namespace(&items[at as usize], ns));
        if level.is_some_and(|at| items[at as usize].kind == ItemKind::Mod) || from.is_none() {
            return held
                .iter()
                .copied()
                .find(|&at| matches(at))
                .map(|at| root.inner(at));
        }
        let mut cur = from.and_then(|at| found.item_scopes.get(at as usize).copied());
        while let Some(scope) = cur {
            if let Some(&at) = held
                .iter()
                .rev()
                .find(|&&at| found.item_scopes.get(at as usize) == Some(&scope) && matches(at))
            {
                return Some(root.inner(at));
            }
            cur = found.scopes.get(scope as usize).copied().flatten();
        }
        None
    }

    fn body_index(&self, file: &'a File) -> &'a Index {
        file.index(|file| index_of(file, &self.names.symbols))
    }

    fn scope_item(
        &self,
        root: ItemRef,
        from: Option<u32>,
        name: &str,
        ns: Option<Ns>,
    ) -> Option<ItemRef> {
        let items = self.body_entries(root);
        let mut level = from;
        loop {
            let searched = level.is_none_or(|at| {
                !matches!(items[at as usize].kind, ItemKind::Impl | ItemKind::Trait)
            });
            if searched && let Some(found) = self.body_child(root, level, from, name, ns) {
                return Some(found);
            }
            match level {
                Some(at) if items[at as usize].kind != ItemKind::Mod => {
                    level = items[at as usize].parent
                }
                _ => return None,
            }
        }
    }

    /// Where `path`, written at `at` in a body of module `home`, leads among the items that the body of `at.0` declares, where `at.1` is the body item it is written in, or none when it does not start among them.
    pub(super) fn local_path(
        &self,
        home: ModRef,
        at: Option<(ItemRef, Option<u32>)>,
        path: &NamePath<'_>,
        ns: Ns,
    ) -> Option<BodyPath> {
        let (root, from) = at.filter(|_| !path.absolute)?;
        let texts: Vec<&str> = path
            .segments
            .iter()
            .map(|&segment| self.names.symbols.text(segment))
            .collect();
        self.body_path((root, home), from, &texts, ns)
    }

    /// Where `rest` leads from the body item `item` of a body in module `home`, which a block around the path declares.
    pub(super) fn local_descend(
        &self,
        home: ModRef,
        item: ItemRef,
        rest: &[Sym],
        ns: Ns,
    ) -> Option<BodyPath> {
        let (root, at) = item.outer()?;
        let texts: Vec<&str> = rest
            .iter()
            .map(|&segment| self.names.symbols.text(segment))
            .collect();
        self.body_descend((root, home), at, &texts, ns)
    }

    fn body_path(
        &self,
        (root, home): (ItemRef, ModRef),
        from: Option<u32>,
        segments: &[&str],
        ns: Ns,
    ) -> Option<BodyPath> {
        let items = self.body_entries(root);
        let enclosing = |mut at: Option<u32>| {
            while let Some(index) = at.filter(|&index| items[index as usize].kind != ItemKind::Mod)
            {
                at = items[index as usize].parent;
            }
            at
        };
        let supers = segments
            .iter()
            .take_while(|segment| **segment == "super")
            .count();
        let (start, rest) = match (supers, segments.first()) {
            (0, None) => return None,
            (0, Some(&"self")) => (enclosing(from)?, &segments[1..]),
            (0, Some(first)) => (
                self.scope_item(root, from, first, Some(head_ns(segments.len(), ns)))?
                    .outer()?
                    .1,
                &segments[1..],
            ),
            _ => {
                let mut module = enclosing(from)?;
                for done in 1..=supers {
                    match enclosing(items[module as usize].parent) {
                        Some(outer) => module = outer,
                        None => return Some(BodyPath::Outside(done)),
                    }
                }
                (module, &segments[supers..])
            }
        };
        self.body_descend((root, home), start, rest, ns)
    }

    fn body_descend(
        &self,
        (root, home): (ItemRef, ModRef),
        mut at: u32,
        rest: &[&str],
        ns: Ns,
    ) -> Option<BodyPath> {
        let items = self.body_entries(root);
        if items[at as usize].kind == ItemKind::ExternCrate {
            return self.extern_descend((root, home), at, rest, ns);
        }
        for (index, name) in rest.iter().enumerate() {
            let last = index + 1 == rest.len();
            match items[at as usize].kind {
                ItemKind::Enum if last => return Some(BodyPath::Variant(root.inner(at))),
                ItemKind::Mod => {
                    let Some(next) = self.body_child(
                        root,
                        Some(at),
                        None,
                        name,
                        Some(if last { ns } else { Ns::Types }),
                    ) else {
                        return self.module_import((root, home), at, &rest[index..], ns);
                    };
                    let next = next.outer()?.1;
                    if items[next as usize].kind == ItemKind::ExternCrate {
                        return self.extern_descend((root, home), next, &rest[index + 1..], ns);
                    }
                    at = next;
                }
                _ => return None,
            }
        }
        Some(BodyPath::Def(Def::Item(root.inner(at))))
    }

    fn extern_descend(
        &self,
        (root, home): (ItemRef, ModRef),
        at: u32,
        rest: &[&str],
        ns: Ns,
    ) -> Option<BodyPath> {
        let found = self.expansion(root)?;
        let item = &found.file.items.items[at as usize];
        let deps = &self.outline.deps[root.krate as usize];
        let def = Def::Module(extern_crate_target(
            &found.file.source,
            item,
            root.krate,
            deps,
        )?);
        if rest.is_empty() {
            return (ns == Ns::Types).then_some(BodyPath::Def(def));
        }
        self.def_descend(home, def, rest, ns)
    }

    fn module_import(
        &self,
        (root, home): (ItemRef, ModRef),
        at: u32,
        rest: &[&str],
        ns: Ns,
    ) -> Option<BodyPath> {
        let file = &self.expansion(root)?.file;
        let (first, more) = rest.split_first()?;
        let word = self.names.symbols.intern(first);
        let paths = self.paths(home.krate);
        let held = |import: &&u32| {
            file.items.items[file.items.imports[**import as usize].item as usize].parent == Some(at)
        };
        let imports = self.body_index(file).imports.get(*first)?;
        let (_, entry) = imports.iter().filter(held).find_map(|&import| {
            self.import_entry((root, home), &paths, (file, import), Some(word))
        })?;
        let binding = if more.is_empty() {
            entry.get(ns)
        } else {
            entry.types
        };
        self.def_descend(home, binding?.def, more, ns)
    }

    fn def_descend(&self, home: ModRef, mut def: Def, rest: &[&str], ns: Ns) -> Option<BodyPath> {
        for (index, name) in rest.iter().enumerate() {
            let entry = self.path_member(home, def, self.names.symbols.intern(name));
            let binding = if index + 1 == rest.len() {
                entry.get(ns)
            } else {
                entry.types
            };
            def = binding?.def;
        }
        Some(BodyPath::Def(def))
    }

    /// The bindings that the `use` declarations of the body `root` in module `home` give `word`, or every name they import when none, as the body item `here` sees them: named imports first, then globs.
    pub(super) fn body_imports(
        &self,
        (root, home): (ItemRef, ModRef),
        word: Option<Sym>,
        (here, hidden): (Option<u32>, &[u32]),
    ) -> Vec<PerNs> {
        let Some(found) = self.expansion(root) else {
            return Vec::new();
        };
        let file = &found.file;
        if file.items.imports.is_empty() {
            return Vec::new();
        }
        let (paths, index) = (self.paths(home.krate), self.body_index(file));
        let named = word
            .and_then(|word| index.imports.get(self.names.symbols.text(word)))
            .map_or(&[][..], |list| &list[..]);
        let globbed = if word.is_some() {
            &index.globs[..]
        } else {
            &[][..]
        };
        let every = if word.is_none() {
            0..file.items.imports.len() as u32
        } else {
            0..0
        };
        let (mut out, mut globs) = (Vec::new(), Vec::new());
        for import in named.iter().chain(globbed).copied().chain(every) {
            if !import_visible(file, index, import, here)
                || hidden.contains(&file.items.imports[import as usize].item)
            {
                continue;
            }
            match self.import_entry((root, home), &paths, (file, import), word) {
                Some((true, entry)) => out.push(entry),
                Some((false, entry)) => globs.push(entry),
                None => {}
            }
        }
        out.extend(globs);
        out
    }

    fn import_entry(
        &self,
        (root, home): (ItemRef, ModRef),
        paths: &Paths<'_>,
        (file, import): (&File, u32),
        word: Option<Sym>,
    ) -> Option<(bool, PerNs)> {
        let (segments, absolute, dollar, name) = import_parts(file, import, &self.names.symbols);
        let segments = segments.strip_suffix(&[SELF]).unwrap_or(&segments);
        let absolute = paths
            .use_path(NamePath {
                segments,
                absolute,
                dollar,
            })
            .absolute;
        let from = file.items.items[file.items.imports[import as usize].item as usize].parent;
        let (head, leaf) = match (name, segments.split_last(), word) {
            (None, _, Some(word)) => (segments, word),
            (Some(_), Some((&leaf, head)), _) => (head, leaf),
            (None, Some(_), None) => {
                let path = NamePath {
                    segments,
                    absolute,
                    dollar,
                };
                let def = match self.local_path(home, Some((root, from)), &path, Ns::Types) {
                    Some(BodyPath::Outside(skip)) if skip == segments.len() => {
                        Some(Def::Module(home))
                    }
                    Some(BodyPath::Outside(skip)) => paths.resolve(&outside(&path, skip), home),
                    found => self
                        .body_binding(found, "")
                        .map(|binding| binding.def)
                        .or_else(|| paths.resolve(&path, home)),
                }?;
                return Some((
                    false,
                    PerNs {
                        types: Some(Binding {
                            def,
                            vis: Vis::Pub,
                            origin: Origin::Glob,
                        }),
                        values: None,
                        macros: None,
                    },
                ));
            }
            _ => return None,
        };
        let local = if absolute {
            None
        } else {
            self.body_import((root, home), from, head, leaf)
        };
        let entry = match head {
            _ if let Some(local) = local => local,
            [] => paths.first(
                leaf,
                home,
                &NamePath {
                    segments: &[leaf],
                    absolute,
                    dollar,
                },
            ),
            _ => {
                let target = if self.outline.crates[home.krate as usize].env.fixed
                    || dollar.is_some_and(|k| self.outline.is_sysroot(k))
                {
                    paths.resolve(
                        &NamePath {
                            segments: head,
                            absolute,
                            dollar,
                        },
                        home,
                    )?
                } else {
                    paths
                        .resolve_visible(
                            &NamePath {
                                segments: head,
                                absolute,
                                dollar,
                            },
                            home,
                        )
                        .ok()??
                };
                let mut found = self.path_member(home, target, leaf);
                if let Def::Module(target_mod) = target
                    && !self.outline.crates[home.krate as usize].env.fixed
                    && !dollar.is_some_and(|k| self.outline.is_sysroot(k))
                {
                    let keep = |b: Binding| {
                        crate::names::resolve::visible(&paths, b.vis, target_mod, home)
                    };
                    found.types = found.types.filter(|&b| keep(b));
                    found.values = found.values.filter(|&b| keep(b));
                    found.macros = found.macros.filter(|&b| keep(b));
                }
                found
            }
        };
        (entry.types.is_some() || entry.values.is_some()).then_some((name.is_some(), entry))
    }

    fn body_import(
        &self,
        (root, home): (ItemRef, ModRef),
        from: Option<u32>,
        head: &[Sym],
        leaf: Sym,
    ) -> Option<PerNs> {
        if head.is_empty() {
            return None;
        }
        let texts: Vec<&str> = head
            .iter()
            .chain([&leaf])
            .map(|&segment| self.names.symbols.text(segment))
            .collect();
        let found = |ns| {
            self.body_binding(
                self.body_path((root, home), from, &texts, ns),
                texts[texts.len() - 1],
            )
        };
        let entry = PerNs {
            types: found(Ns::Types),
            values: found(Ns::Values),
            macros: None,
        };
        (entry.types.is_some() || entry.values.is_some()).then_some(entry)
    }

    /// The bindings of `name` inside `def`, an enum's variants or a module's items, for a path written in module `home`, where a body may declare `def`.
    pub(super) fn path_member(&self, home: ModRef, def: Def, name: Sym) -> PerNs {
        let Some((root, at)) = (match def {
            Def::Item(item) => item.outer(),
            _ => None,
        }) else {
            return self.paths(home.krate).member(def, name);
        };
        let text = self.names.symbols.text(name);
        let found = |ns| self.body_binding(self.body_descend((root, home), at, &[text], ns), text);
        PerNs {
            types: found(Ns::Types),
            values: found(Ns::Values),
            macros: None,
        }
    }

    fn body_binding(&self, path: Option<BodyPath>, name: &str) -> Option<Binding> {
        let def = match path? {
            BodyPath::Def(def) => def,
            BodyPath::Variant(item) => self.body_variant(item, name)?,
            BodyPath::Outside(_) => return None,
        };
        Some(Binding {
            def,
            vis: Vis::Pub,
            origin: Origin::Item,
        })
    }

    fn body_variant(&self, item: ItemRef, name: &str) -> Option<Def> {
        let (root, at) = item.outer()?;
        let found = self.expansion(root)?;
        let open = found.file.items.items[at as usize].body?;
        let index = crate::names::path::variant_names(&found.file.source, open)
            .iter()
            .position(|variant| *variant == name)?;
        Some(Def::Variant(item, index as u32))
    }

    /// Resolves `path` in the type namespace through the `use` declarations of the body `root` in module `home`, as the body item `here` sees them, when one binds its first segment.
    pub(super) fn body_use_path(
        &self,
        (root, home): (ItemRef, ModRef),
        path: &NamePath<'_>,
        here: Option<u32>,
    ) -> Option<Def> {
        let (first, rest) = path.segments.split_first().filter(|_| !path.absolute)?;
        let binding = self
            .body_imports((root, home), Some(*first), (here, &[]))
            .into_iter()
            .find_map(|entry| entry.types)?;
        rest.iter().try_fold(binding.def, |def, segment| {
            self.path_member(home, def, *segment)
                .types
                .map(|found| found.def)
        })
    }
}

fn import_visible(file: &File, index: &Index, import: u32, here: Option<u32>) -> bool {
    index.holders[import as usize]
        .iter()
        .all(|&at| here.is_some_and(|here| encloses(&file.items.items, at, here)))
}

fn encloses(items: &[Item], at: u32, from: u32) -> bool {
    std::iter::successors(Some(from), |&now| items[now as usize].parent).any(|now| now == at)
}

fn index_of(file: &File, symbols: &Symbols) -> Index {
    let (src, items) = (&file.source, &file.items.items);
    let mut index = Index::default();
    for (at, entry) in items.iter().enumerate() {
        let Some(name) = entry.name else { continue };
        let block = entry
            .parent
            .filter(|&parent| items[parent as usize].kind == ItemKind::ExternBlock);
        for level in [
            Some(entry.parent),
            block.map(|parent| items[parent as usize].parent),
        ]
        .into_iter()
        .flatten()
        {
            index
                .items
                .entry(level)
                .or_default()
                .entry(plain(src.text(name)).into())
                .or_default()
                .push(at as u32);
        }
    }
    for import in 0..file.items.imports.len() as u32 {
        match import_parts(file, import, symbols).3 {
            Some(name) => index
                .imports
                .entry(symbols.text(name).into())
                .or_default()
                .push(import),
            None => index.globs.push(import),
        }
        let parent = items[file.items.imports[import as usize].item as usize].parent;
        index.holders.push(
            std::iter::successors(parent, |&at| items[at as usize].parent)
                .filter(|&at| matches!(items[at as usize].kind, ItemKind::Fn | ItemKind::Mod))
                .collect(),
        );
    }
    index
}
