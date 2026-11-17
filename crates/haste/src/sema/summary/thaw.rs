use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::names::ItemRef;
use crate::outline::Outline;

use super::super::Context;
use super::super::infer::{LazyTables, Shape, Typed, register};
use super::super::reach::{Instance, lay_out};
use super::super::table::Table;
use super::super::ty::Types;
use super::{SummaryKind, codec, frozen};

/// The persisted typed body of fixed `instance`, registered and laid out as a fresh inference leaves it.
pub(in crate::sema) fn thawed(cx: &mut Context<'_>, instance: &Instance, unlaid: &mut Vec<String>) -> Option<Result<Arc<Typed>, String>> {
    let typed = frozen::instance_bytes(cx, *instance, |index, bytes| codec::decode_typed(cx.outline, index, &cx.types, instance.1, bytes).ok().flatten())?;
    let (before, param_count) = (cx.quiet_mark(), cx.types.substs(instance.1).len());
    if matches!(typed.shape, Shape::Body)
        && let Err(err) = warm_body(cx, instance.0, param_count, before)
    {
        return Some(Err(err));
    }
    let typed = Arc::new(typed);
    register(cx, *instance, &typed);
    keep_laid(cx, *instance, &typed, unlaid, before);
    Some(Ok(typed))
}

/// Lays out `typed` and keeps it as the final body of `instance` when it laid out and nothing was asked since `before`.
pub(in crate::sema) fn keep_laid(cx: &mut Context<'_>, instance: Instance, typed: &Arc<Typed>, unlaid: &mut Vec<String>, before: (usize, usize)) {
    let laid = lay_out(cx, typed).map_err(|error| unlaid.push(format!("{}: {error:#}", cx.name(instance.0))));
    if laid.is_ok() && cx.quiet_mark() == before {
        cx.store.finals.insert(instance, Arc::clone(typed));
    }
}

/// Parses the body of fixed `item` while the pass can still answer what its macros ask, failing when it asked and did not parse.
pub(in crate::sema) fn warm_body(cx: &mut Context<'_>, item: ItemRef, param_count: usize, before: (usize, usize)) -> Result<(), String> {
    if cx.outline.is_fixed(item.krate)
        && !cx.store.lookups.skip_warm_sysroot_bodies
        && cx.store.bodies.get(&item).is_none()
        && cx.store.lookups.substrate.is_some()
        && !(cx.outline.is_sysroot(item.krate)
            && param_count == 0
            && (cx.attr_value(item, "cold").is_some() || cx.attr_value(item, "inline") == Some("never")))
        && let Err(err) = cx.body(item)
        && cx.quiet_mark() != before
    {
        return Err(format!("{}: {err:#}", cx.name(item)));
    }
    Ok(())
}

/// Flags a fresh eager body of a fixed `instance` that the substrate can keep, so the batch saves it.
pub(in crate::sema) fn note_fresh(cx: &Context<'_>, instance: Instance, typed: &Typed) {
    let lookups = &cx.store.lookups;
    let LazyTables::Eager(tables) = &typed.tables else { return };
    if lookups.has_new_sysroot_instances.load(Ordering::Relaxed)
        || frozen::instance_keys(cx.outline, &cx.types, instance).is_none()
        || frozen::is_template(&lookups.summaries, instance.0, tables)
        || !cx.store.finals.contains_key(&instance)
    {
        return;
    }
    let (_, key_fn) = frozen::layer_key_fn(cx.outline, &cx.types, instance);
    if matches!(codec::encode_typed(cx.outline, &cx.types, instance.1, typed, key_fn), Ok(Some(_))) {
        lookups.has_new_sysroot_instances.store(true, Ordering::Relaxed);
    }
}

/// The persisted non-polymorphic verdict of fixed `item`.
pub(in crate::sema) fn frozen_verdict(cx: &Context<'_>, item: ItemRef) -> Option<SummaryKind> {
    frozen::thaw_summary(cx, item, codec::decode_verdict)
}

/// Records the non-polymorphic verdict just built for fixed `item` as persistable when nothing was asked since `before`.
pub(in crate::sema) fn settle_verdict(cx: &Context<'_>, item: ItemRef, before: (usize, usize)) {
    if cx.outline.is_fixed(item.krate) && cx.quiet_mark() == before {
        cx.store.lookups.verdicts.insert(item, ());
        cx.store.lookups.has_new_sysroot_summaries.store(true, Ordering::Relaxed);
    }
}

/// The substrate bytes of the settled summary `kind` of `item`: a polymorphic summary, or a verdict built without asks.
pub(in crate::sema) fn encode_kind(outline: &Outline, types: &Types, (item, kind): (ItemRef, &SummaryKind), verdicts: &Table<ItemRef, ()>) -> Option<Vec<u8>> {
    let encoded = match kind {
        SummaryKind::Poly(summary) => codec::encode_summary(outline, types, summary, crate::substrate::sysroot_item_key),
        _ if verdicts.contains_key(&item) => codec::encode_verdict(outline, types, kind, crate::substrate::sysroot_item_key),
        _ => return None,
    };
    encoded.ok().flatten()
}
