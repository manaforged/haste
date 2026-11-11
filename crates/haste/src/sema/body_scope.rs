//! Names inside a function body: the items that its blocks and the modules it declares hold, found from the innermost scope out, as rustc resolves them.

use crate::names::collect::{import_parts, namespaces};
use crate::names::path::{Path as NamePath, Paths};
use crate::names::{Binding, Def, ItemRef, ModRef, Ns, Origin, PerNs, SELF, Sym, Symbols, plain};
use crate::outline::item::{Item, ItemKind, Vis};
use crate::outline::tree::{File, Index};

use super::Context;

/// Where a path written inside a body leads.
pub(super) enum BodyPath {
    /// An item that the body declares.
    Item(ItemRef),
    /// An enum that the body declares, whose variant the last segment names.
    Variant(ItemRef),
    /// Out of the body's modules through `super`: the rest of the path, after this many segments, resolves from the home module.
    Outside(usize),
}

/// The rest of `path` once its first `skip` segments led out of the body's modules, to resolve from the home module.
pub(super) fn outside<'p>(path: &NamePath<'p>, skip: usize) -> NamePath<'p> {
    NamePath { segments: &path.segments[skip..], absolute: false, dollar: path.dollar }
}

fn in_namespace(entry: &Item, ns: Ns) -> bool {
    if entry.kind == ItemKind::Mod { ns == Ns::Types } else { namespaces(entry).contains(&ns) }
}

impl<'a> Context<'a> {
    fn body_entries(&self, root: ItemRef) -> &'a [Item] {
        self.expansion(root).map_or(&[][..], |found| &found.file.items.items[..])
    }

    /// The item named `name`, in namespace `ns` or any when none, that the body item `level` of `root`, or the body itself when none, holds directly or in an extern block.
    fn body_child(&self, root: ItemRef, level: Option<u32>, name: &str, ns: Option<Ns>) -> Option<ItemRef> {
        let found = self.expansion(root)?;
        let items = &found.file.items.items;
        let held = self.body_index(&found.file).items.get(&level)?.get(name)?;
        held.iter().find(|&&at| ns.is_none_or(|ns| in_namespace(&items[at as usize], ns))).map(|&at| root.inner(at))
    }

    /// The name index of the body expansion `file`, made on first use and kept with it.
    fn body_index(&self, file: &'a File) -> &'a Index {
        file.index(|file| index_of(file, &self.names.symbols))
    }

    /// The item named `name` that code at the body item `from` of `root`, or in the body's own blocks when none, sees: in the blocks around it, then in its module, innermost first.
    fn scope_item(&self, root: ItemRef, from: Option<u32>, name: &str, ns: Option<Ns>) -> Option<ItemRef> {
        let items = self.body_entries(root);
        let mut level = from;
        loop {
            let searched = level.is_none_or(|at| !matches!(items[at as usize].kind, ItemKind::Impl | ItemKind::Trait));
            if searched && let Some(found) = self.body_child(root, level, name, ns) {
                return Some(found);
            }
            match level {
                Some(at) if items[at as usize].kind != ItemKind::Mod => level = items[at as usize].parent,
                _ => return None,
            }
        }
    }

    /// Where `path`, written at `at`, leads among the items that the body of `at.0` declares, where `at.1` is the body item it is written in, or none when it does not start among them.
    pub(super) fn local_path(&self, at: Option<(ItemRef, Option<u32>)>, path: &NamePath<'_>, ns: Ns) -> Option<BodyPath> {
        let (root, from) = at.filter(|_| !path.absolute)?;
        let texts: Vec<&str> = path.segments.iter().map(|&segment| self.names.symbols.text(segment)).collect();
        self.body_path(root, from, &texts, ns)
    }

    /// Where `rest` leads from the body item `item`, which a block around the path declares.
    pub(super) fn local_descend(&self, item: ItemRef, rest: &[crate::names::Sym], ns: Ns) -> Option<BodyPath> {
        let (root, at) = item.outer()?;
        let texts: Vec<&str> = rest.iter().map(|&segment| self.names.symbols.text(segment)).collect();
        self.body_descend(root, at, &texts, ns)
    }

    /// Where `segments`, written at the body item `from` of `root`, lead through the items and modules that the body declares, or none when the path does not start among them.
    fn body_path(&self, root: ItemRef, from: Option<u32>, segments: &[&str], ns: Ns) -> Option<BodyPath> {
        let items = self.body_entries(root);
        let enclosing = |mut at: Option<u32>| {
            while let Some(index) = at.filter(|&index| items[index as usize].kind != ItemKind::Mod) {
                at = items[index as usize].parent;
            }
            at
        };
        let supers = segments.iter().take_while(|segment| **segment == "super").count();
        let (start, rest) = match (supers, segments.first()) {
            (0, None) => return None,
            (0, Some(&"self")) => (enclosing(from)?, &segments[1..]),
            (0, Some(first)) => {
                let wanted = if segments.len() == 1 { ns } else { Ns::Types };
                (self.scope_item(root, from, first, Some(wanted))?.outer()?.1, &segments[1..])
            }
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
        self.body_descend(root, start, rest, ns)
    }

    fn body_descend(&self, root: ItemRef, mut at: u32, rest: &[&str], ns: Ns) -> Option<BodyPath> {
        let items = self.body_entries(root);
        for (index, name) in rest.iter().enumerate() {
            let last = index + 1 == rest.len();
            match items[at as usize].kind {
                ItemKind::Enum if last => return Some(BodyPath::Variant(root.inner(at))),
                ItemKind::Mod => at = self.body_child(root, Some(at), name, Some(if last { ns } else { Ns::Types }))?.outer()?.1,
                _ => return None,
            }
        }
        Some(BodyPath::Item(root.inner(at)))
    }

    /// The bindings that the `use` declarations of the body `root` in module `home` give `word`, or every name they import when none, as code at token `here` sees them: named imports first, then globs.
    pub(super) fn body_imports(&self, (root, home): (ItemRef, ModRef), word: Option<Sym>, here: Option<u32>) -> Vec<PerNs> {
        let Some(found) = self.expansion(root) else { return Vec::new() };
        let file = &found.file;
        if file.items.imports.is_empty() {
            return Vec::new();
        }
        let (paths, index) = (self.paths(home.krate), self.body_index(file));
        let named = word.and_then(|word| index.imports.get(self.names.symbols.text(word))).map_or(&[][..], |list| &list[..]);
        let globbed = if word.is_some() { &index.globs[..] } else { &[][..] };
        let every = if word.is_none() { 0..file.items.imports.len() as u32 } else { 0..0 };
        let (mut out, mut globs) = (Vec::new(), Vec::new());
        for import in named.iter().chain(globbed).copied().chain(every) {
            if !import_visible(file, index, import, here) {
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

    /// What the import `import` of the body expansion `file` gives `word`, or every name it imports when none, and whether it is a named import rather than a glob.
    fn import_entry(&self, (root, home): (ItemRef, ModRef), paths: &Paths<'_>, (file, import): (&File, u32), word: Option<Sym>) -> Option<(bool, PerNs)> {
        let (segments, absolute, dollar, name) = import_parts(file, import, &self.names.symbols);
        let segments = segments.strip_suffix(&[SELF]).unwrap_or(&segments);
        let absolute = paths.use_path(NamePath { segments, absolute, dollar }).absolute;
        let (head, leaf) = match (name, segments.split_last(), word) {
            (None, _, Some(word)) => (segments, word),
            (Some(_), Some((&leaf, head)), _) => (head, leaf),
            (None, Some(_), None) => {
                let def = paths.resolve(&NamePath { segments, absolute, dollar }, home)?;
                return Some((false, PerNs { types: Some(Binding { def, vis: Vis::Pub, origin: Origin::Glob }), values: None, macros: None }));
            }
            _ => return None,
        };
        let from = file.items.items[file.items.imports[import as usize].item as usize].parent;
        let local = if absolute { None } else { self.body_import(root, from, head, leaf) };
        let entry = match head {
            _ if let Some(local) = local => local,
            [] => paths.first(leaf, home, &NamePath { segments: &[leaf], absolute, dollar }),
            _ => self.path_member(home.krate, paths.resolve(&NamePath { segments: head, absolute, dollar }, home)?, leaf),
        };
        (entry.types.is_some() || entry.values.is_some()).then_some((name.is_some(), entry))
    }

    /// The bindings of `leaf` in the item that `head`, written at the body item `from` of `root`, names among the items the body declares: a variant of an enum or an item of a module.
    fn body_import(&self, root: ItemRef, from: Option<u32>, head: &[Sym], leaf: Sym) -> Option<PerNs> {
        if head.is_empty() {
            return None;
        }
        let texts: Vec<&str> = head.iter().chain([&leaf]).map(|&segment| self.names.symbols.text(segment)).collect();
        let found = |ns| self.body_binding(self.body_path(root, from, &texts, ns), texts[texts.len() - 1]);
        let entry = PerNs { types: found(Ns::Types), values: found(Ns::Values), macros: None };
        (entry.types.is_some() || entry.values.is_some()).then_some(entry)
    }

    /// The bindings of `name` inside `def`, an enum's variants or a module's items, for a path of crate `krate`, where a body may declare `def`.
    pub(super) fn path_member(&self, krate: u32, def: Def, name: Sym) -> PerNs {
        let Some((root, at)) = (match def { Def::Item(item) => item.outer(), _ => None }) else { return self.paths(krate).member(def, name) };
        let text = self.names.symbols.text(name);
        let found = |ns| self.body_binding(self.body_descend(root, at, &[text], ns), text);
        PerNs { types: found(Ns::Types), values: found(Ns::Values), macros: None }
    }

    /// The binding of the body item or enum variant `name` that `path` reached.
    fn body_binding(&self, path: Option<BodyPath>, name: &str) -> Option<Binding> {
        let def = match path? {
            BodyPath::Item(item) => Def::Item(item),
            BodyPath::Variant(item) => self.body_variant(item, name)?,
            BodyPath::Outside(_) => return None,
        };
        Some(Binding { def, vis: Vis::Pub, origin: Origin::Item })
    }

    /// The variant named `name` of the enum `item` that a body declares.
    fn body_variant(&self, item: ItemRef, name: &str) -> Option<Def> {
        let (root, at) = item.outer()?;
        let found = self.expansion(root)?;
        let open = found.file.items.items[at as usize].body?;
        let index = crate::names::path::variant_names(&found.file.source, open).iter().position(|variant| *variant == name)?;
        Some(Def::Variant(item, index as u32))
    }

    /// Resolves `path` in the type namespace through the `use` declarations of the body `root` in module `home`, as code at token `here` sees them, when one binds its first segment.
    pub(super) fn body_use_path(&self, (root, home): (ItemRef, ModRef), path: &NamePath<'_>, here: Option<u32>) -> Option<Def> {
        let (first, rest) = path.segments.split_first().filter(|_| !path.absolute)?;
        let binding = self.body_imports((root, home), Some(*first), here).into_iter().find_map(|entry| entry.types)?;
        rest.iter().try_fold(binding.def, |def, segment| self.path_member(home.krate, def, *segment).types.map(|found| found.def))
    }
}

/// Whether the `use` declaration `import` of a body's expansion `file` is in scope at token `here`, or in the body itself when `here` is none: it is unless a function declared in the body holds it and not `here`.
fn import_visible(file: &File, index: &Index, import: u32, here: Option<u32>) -> bool {
    index.holders[import as usize].iter().all(|&at| here.is_some_and(|here| holds(&file.items.items[at as usize], here)))
}

/// Whether the tokens of `entry` hold token `token`.
fn holds(entry: &Item, token: u32) -> bool {
    entry.start <= token && token < entry.end
}

/// The name index of a body's expansion `file`: its items by holder and plain name, its imports by the name `symbols` spells for them, and the functions around each import.
fn index_of(file: &File, symbols: &Symbols) -> Index {
    let (src, items) = (&file.source, &file.items.items);
    let mut index = Index::default();
    for (at, entry) in items.iter().enumerate() {
        let Some(name) = entry.name else { continue };
        let block = entry.parent.filter(|&parent| items[parent as usize].kind == ItemKind::ExternBlock);
        for level in [Some(entry.parent), block.map(|parent| items[parent as usize].parent)].into_iter().flatten() {
            index.items.entry(level).or_default().entry(plain(src.text(name)).into()).or_default().push(at as u32);
        }
    }
    let fns: Vec<u32> = (0..items.len() as u32).filter(|&at| items[at as usize].kind == ItemKind::Fn).collect();
    for import in 0..file.items.imports.len() as u32 {
        match import_parts(file, import, symbols).3 {
            Some(name) => index.imports.entry(symbols.text(name).into()).or_default().push(import),
            None => index.globs.push(import),
        }
        let start = items[file.items.imports[import as usize].item as usize].start;
        index.holders.push(fns.iter().copied().filter(|&at| holds(&items[at as usize], start)).collect());
    }
    index
}
