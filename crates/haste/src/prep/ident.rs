use crate::body::{BlockKind, Expr};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Shape, Typed};
use crate::sema::reach::Instance;
use crate::sema::ty::Kind;

use super::record::{ClosureKind, InstanceKind, Record};

/// The separator of definition path segments.
pub(super) const SEPARATOR: &str = "::";
const CALL_ONCE: &str = "call_once";
const CALLS_BY_REFERENCE: [&str; 2] = ["call", "call_mut"];

/// What names one function instance in a driver manifest: the definition path of a function item, or a closure by the path of its typeck root and its place among the root's closures.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum Ident {
    Item(String),
    Closure(ClosureAt),
}

/// A closure by its typeck root's path, how it is called, and its place among the root's closures in the order their expressions end, with their count.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct ClosureAt {
    pub root: String,
    pub once: bool,
    pub order: usize,
    pub count: usize,
}

/// The definition path rustc prints for function item `item` outside the standard crates, when it is a function declared at module level.
pub(super) fn item_path(outline: &Outline, item: ItemRef) -> Option<String> {
    if item.outer().is_some() {
        return None;
    }
    let krate = &outline.crates[item.krate as usize];
    let file = krate.files.get(item.file as usize)?;
    let entry = file.items.items.get(item.item as usize)?;
    if entry.kind != ItemKind::Fn {
        return None;
    }
    let holder = match entry.parent {
        Some(parent) if file.items.items.get(parent as usize)?.kind == ItemKind::Mod => Some(parent),
        Some(_) => return None,
        None => None,
    };
    let mut at = krate.modules.iter().position(|module| module.file == item.file as usize && module.item == holder)?;
    let mut segments = vec![file.source.text(entry.name?).to_owned()];
    while let Some(parent) = krate.modules[at].parent {
        segments.push(krate.modules[at].name.clone());
        at = parent;
    }
    segments.push(outline.roots[item.krate as usize].name.replace('-', "_"));
    segments.reverse();
    Some(segments.join(SEPARATOR))
}

/// The identity of `instance` with body `typed` in a driver manifest, for a non-generic function item or a closure in one, outside the crates `sysroot` marks.
pub(super) fn ident(cx: &mut Context<'_>, (item, substs): Instance, typed: &Typed, sysroot: &[bool]) -> Option<Ident> {
    match &typed.shape {
        Shape::Body if cx.types.substs(substs).is_empty() && !sysroot[item.krate as usize] => item_path(cx.outline, item).map(Ident::Item),
        Shape::Closure(_) => closure(cx, (item, substs), sysroot).map(Ident::Closure),
        _ => None,
    }
}

fn closure(cx: &mut Context<'_>, (item, substs): Instance, sysroot: &[bool]) -> Option<ClosureAt> {
    let callee = *cx.types.substs(substs).first()?;
    let Kind::Closure(root, outer, at) = *cx.types.kind(callee) else { return None };
    if !cx.types.substs(outer).is_empty() || sysroot[root.krate as usize] {
        return None;
    }
    let method = cx.name(item);
    let once = method == CALL_ONCE;
    if !once && !CALLS_BY_REFERENCE.contains(&method) {
        return None;
    }
    let body = cx.body(root).ok()?;
    let mut ends = Vec::new();
    for (index, expr) in body.exprs.iter().enumerate() {
        match expr {
            Expr::Closure { .. } => ends.push(index as u32),
            Expr::Block { kind: BlockKind::Async | BlockKind::AsyncMove | BlockKind::Const, .. } => return None,
            _ => {}
        }
    }
    let order = ends.iter().position(|&end| end == at)?;
    Some(ClosureAt { root: item_path(cx.outline, root)?, once, order, count: ends.len() })
}

impl Ident {
    /// The path under which the records this identity may match are indexed: the item's path, or the closure's root.
    pub(super) fn key(&self) -> &str {
        match self {
            Ident::Item(path) => path,
            Ident::Closure(at) => &at.root,
        }
    }

    /// Whether `record` is the instance this identity names.
    pub(super) fn names(&self, record: &Record) -> bool {
        if !record.args.is_empty() {
            return false;
        }
        match (self, &record.closure) {
            (Ident::Item(path), None) => record.kind == InstanceKind::Item && *path == record.path,
            (Ident::Closure(at), Some(id)) => {
                let placed = id.root == at.root && id.order == Some(at.order) && id.count == Some(at.count);
                placed && called(at.once, record.kind, id.kind)
            }
            _ => false,
        }
    }
}

/// Whether a call by value (`once`) or by reference runs the code of a record of `kind` for a closure of call trait `trait_kind`.
fn called(once: bool, kind: InstanceKind, trait_kind: Option<ClosureKind>) -> bool {
    match (once, kind, trait_kind) {
        (false, InstanceKind::Item, Some(ClosureKind::Fn | ClosureKind::FnMut)) => true,
        (true, InstanceKind::ClosureOnce, Some(ClosureKind::Fn | ClosureKind::FnMut)) => true,
        (true, InstanceKind::Item, Some(ClosureKind::FnOnce)) => true,
        _ => false,
    }
}

/// Whether the definition path `path` is `changed` or lies inside it, as a closure or a nested item does.
pub(super) fn within(path: &str, changed: &str) -> bool {
    path.strip_prefix(changed).is_some_and(|rest| rest.is_empty() || rest.starts_with(SEPARATOR))
}
