//! Names inside a function body: the items that its blocks and the modules it declares hold, found from the innermost scope out, as rustc resolves them.

use crate::names::collect::namespaces;
use crate::names::path::Path as NamePath;
use crate::names::{ItemRef, Ns, plain};
use crate::outline::item::{Item, ItemKind};

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
        let (src, items) = (&found.file.source, &found.file.items.items[..]);
        let held = |entry: &Item| entry.parent == level || entry.parent.is_some_and(|at| items[at as usize].kind == ItemKind::ExternBlock && items[at as usize].parent == level);
        let named = |entry: &Item| entry.name.is_some_and(|at| plain(src.text(at)) == name) && ns.is_none_or(|ns| in_namespace(entry, ns));
        items.iter().position(|entry| held(entry) && named(entry)).map(|at| root.inner(at as u32))
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
}
