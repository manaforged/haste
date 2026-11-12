use crate::body::{BlockKind, Expr};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::ty::{Kind, Ty};

use super::record::{ClosureKind, InstanceKind};

/// The separator of definition path segments.
const SEPARATOR: &str = "::";
const CALL_ONCE: &str = "call_once";
const CALLS_BY_REFERENCE: [&str; 2] = ["call", "call_mut"];

/// The item kinds whose definitions a key names: functions, and the types and traits a key's types name.
const NAMED: [ItemKind; 5] = [ItemKind::Fn, ItemKind::Struct, ItemKind::Enum, ItemKind::Union, ItemKind::Trait];

/// A closure type by its typeck root, the root's type arguments, and its place among the root's closures in the order their expressions end, with their count.
pub(super) struct ClosureAt {
    pub root: ItemRef,
    pub outer: Vec<Ty>,
    pub order: usize,
    pub count: usize,
}

/// The name of item `item` of a crate that no edit moves: its module path, the header of the impl or trait that holds it, and its name; none for an item declared in a body or of a kind no key names.
pub(super) fn identity(outline: &Outline, item: ItemRef) -> Option<String> {
    if item.outer().is_some() {
        return None;
    }
    let krate = &outline.crates[item.krate as usize];
    let file = krate.files.get(item.file as usize)?;
    let entry = file.items.items.get(item.item as usize)?;
    if !NAMED.contains(&entry.kind) {
        return None;
    }
    let mut segments = vec![file.source.text(entry.name?).to_owned()];
    let mut holder = entry.parent;
    if let Some(parent) = holder.and_then(|parent| file.items.items.get(parent as usize)).filter(|parent| matches!(parent.kind, ItemKind::Impl | ItemKind::Trait)) {
        segments.push(file.source.spelled((parent.start, parent.body?)));
        holder = parent.parent;
    }
    if let Some(parent) = holder {
        (file.items.items.get(parent as usize)?.kind == ItemKind::Mod).then_some(())?;
    }
    let mut at = krate.modules.iter().position(|module| module.file == item.file as usize && module.item == holder)?;
    while let Some(parent) = krate.modules[at].parent {
        segments.push(krate.modules[at].name.clone());
        at = parent;
    }
    segments.push(outline.roots[item.krate as usize].name.replace('-', "_"));
    segments.reverse();
    Some(segments.join(SEPARATOR))
}

/// The closure that closure type `ty` is, when its root holds no coroutine or inline constant.
pub(super) fn closure(cx: &mut Context<'_>, ty: Ty) -> Option<ClosureAt> {
    let Kind::Closure(root, outer, at) = *cx.types.kind(ty) else { return None };
    let outer = cx.types.substs(outer).to_vec();
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
    Some(ClosureAt { root, outer, order, count: ends.len() })
}

/// Whether the call method `item` of a closure runs it by value, or none when `item` is no closure call method.
pub(super) fn once(cx: &Context<'_>, item: ItemRef) -> Option<bool> {
    let method = cx.name(item);
    (method == CALL_ONCE || CALLS_BY_REFERENCE.contains(&method)).then_some(method == CALL_ONCE)
}

/// Whether a call by value (`once`) or by reference runs the code of a record of `kind` for a closure of call trait `trait_kind`.
pub(super) fn called(once: bool, kind: InstanceKind, trait_kind: Option<ClosureKind>) -> bool {
    match (once, kind, trait_kind) {
        (false, InstanceKind::Item, Some(ClosureKind::Fn | ClosureKind::FnMut)) => true,
        (true, InstanceKind::ClosureOnce, Some(ClosureKind::Fn | ClosureKind::FnMut)) => true,
        (true, InstanceKind::Item, Some(ClosureKind::FnOnce)) => true,
        _ => false,
    }
}
