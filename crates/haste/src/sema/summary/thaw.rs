use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::names::ItemRef;
use crate::outline::Outline;

use super::super::Context;
use super::super::infer::{Shape, Typed, register};
use super::super::reach::{Instance, lay_out};
use super::super::table::Table;
use super::super::ty::Types;
use super::{SummaryKind, codec, frozen};

/// The persisted typed body of fixed `instance`, registered and laid out as a fresh inference leaves it.
pub(in crate::sema) fn thawed(
    cx: &mut Context<'_>,
    instance: &Instance,
    unlaid: &mut Vec<String>,
) -> Option<Result<Arc<Typed>, String>> {
    let typed = frozen::instance_bytes(cx, *instance, |index, bytes| {
        codec::decode_typed(cx.outline, index, &cx.types, instance.1, bytes)
            .ok()
            .flatten()
    })?;
    let before = cx.quiet_mark();
    if matches!(typed.shape, Shape::Body)
        && let Err(err) = warm_body(cx, instance.0, before)
    {
        return Some(Err(err));
    }
    let typed = Arc::new(typed);
    register(cx, *instance, &typed);
    super::super::infer::record_parent_drops(cx, *instance, &typed);
    keep_laid(cx, *instance, &typed, unlaid, before);
    Some(Ok(typed))
}

/// Lays out `typed` and keeps it as the final body of `instance` when it laid out and nothing was asked since `before`.
pub(in crate::sema) fn keep_laid(
    cx: &mut Context<'_>,
    instance: Instance,
    typed: &Arc<Typed>,
    unlaid: &mut Vec<String>,
    before: (usize, usize),
) {
    let laid = lay_out(cx, typed)
        .map_err(|error| unlaid.push(format!("{}: {error:#}", cx.name(instance.0))));
    if laid.is_ok() {
        super::super::infer::mark_parent_laid(cx, instance, typed);
        if cx.quiet_mark() == before && !matches!(typed.shape, Shape::GlobalAlloc(_)) {
            cx.store.finals.insert(instance, Arc::clone(typed));
        }
    }
}

fn body_may_expand(cx: &Context<'_>, item: ItemRef) -> bool {
    let (src, entry) = cx.item(item);
    let (from, to) = match entry.body {
        Some(open) => (open + 1, src.pair(open)),
        None => (entry.start, entry.end),
    };
    (from..to).any(|at| {
        matches!(
            src.kind(at),
            rustc_lexer::TokenKind::Bang | rustc_lexer::TokenKind::Pound
        )
    })
}

/// Parses the body of fixed non-sysroot `item` while the pass can still answer what its macros ask, failing when it asked and did not parse.
pub(in crate::sema) fn warm_body(
    cx: &mut Context<'_>,
    item: ItemRef,
    before: (usize, usize),
) -> Result<(), String> {
    if cx.outline.is_fixed(item.krate)
        && !cx.outline.is_sysroot(item.krate)
        && !cx.store.lookups.skip_warm_sysroot_bodies
        && cx.store.bodies.get(&item).is_none()
        && cx.store.lookups.substrate.is_some()
        && body_may_expand(cx, item)
        && let Err(err) = cx.body(item)
        && cx.quiet_mark() != before
    {
        return Err(crate::diag::format_error(cx.name(item), &err));
    }
    Ok(())
}

/// Encodes and records a newly inferred fixed `instance` when it is absent from the substrate, so the batch saves it.
pub(in crate::sema) fn note_fresh(cx: &Context<'_>, instance: Instance, typed: &Typed) {
    let lookups = &cx.store.lookups;
    if !cx.store.finals.contains_key(&instance) {
        return;
    }
    let Some((pure_sysroot, item_key, substs_key, bytes)) =
        frozen::encode_fresh_instance((cx.outline, &cx.types), &lookups.summaries, instance, typed)
    else {
        return;
    };
    if lookups.substrate.as_ref().is_some_and(|sub| {
        sub.instance_candidates(item_key, substs_key)
            .any(|b| b == bytes.as_ref())
    }) {
        return;
    }
    lookups
        .fresh_instances
        .insert(instance, (pure_sysroot, item_key, substs_key, bytes));
    lookups
        .has_new_sysroot_instances
        .store(true, Ordering::Relaxed);
}

/// Encodes and records a newly built fixed `PolySummary` or settled verdict when absent from the substrate.
pub(in crate::sema) fn note_fresh_summary(cx: &Context<'_>, item: ItemRef, kind: &SummaryKind) {
    if !cx.outline.is_fixed(item.krate) {
        return;
    }
    let lookups = &cx.store.lookups;
    let Some(key) = crate::substrate::sysroot_item_key(cx.outline, item) else {
        return;
    };
    let Some(bytes) = encode_kind(cx.outline, &cx.types, (item, kind), &lookups.verdicts) else {
        return;
    };
    if lookups
        .substrate
        .as_ref()
        .is_some_and(|sub| sub.summary(key) == Some(bytes.as_slice()))
    {
        return;
    }
    lookups.fresh_summaries.insert(
        item,
        (
            cx.outline.is_sysroot(item.krate),
            key,
            bytes.into_boxed_slice(),
        ),
    );
    lookups
        .has_new_sysroot_summaries
        .store(true, Ordering::Relaxed);
}

/// The persisted non-polymorphic verdict of fixed `item`.
pub(in crate::sema) fn frozen_verdict(cx: &Context<'_>, item: ItemRef) -> Option<SummaryKind> {
    frozen::thaw_summary(cx, item, codec::decode_verdict)
}

/// Records the non-polymorphic verdict just built for fixed `item` as persistable when nothing was asked since `before`.
pub(in crate::sema) fn settle_verdict(
    cx: &Context<'_>,
    item: ItemRef,
    before: (usize, usize),
    kind: &SummaryKind,
) {
    if cx.outline.is_fixed(item.krate) && cx.quiet_mark() == before {
        cx.store.lookups.verdicts.insert(item, ());
        note_fresh_summary(cx, item, kind);
    }
}

/// The substrate bytes of the settled summary `kind` of `item`: a polymorphic summary, or a verdict built without asks.
pub(in crate::sema) fn encode_kind(
    outline: &Outline,
    types: &Types,
    (item, kind): (ItemRef, &SummaryKind),
    verdicts: &Table<ItemRef, ()>,
) -> Option<Vec<u8>> {
    let encoded = match kind {
        SummaryKind::Poly(summary) => {
            codec::encode_summary(outline, types, summary, crate::substrate::sysroot_item_key)
        }
        _ if verdicts.contains_key(&item) => {
            codec::encode_verdict(outline, types, kind, crate::substrate::sysroot_item_key)
        }
        _ => return None,
    };
    encoded.ok().flatten()
}
