use std::cell::RefCell;

use rustc_abi::Variants;

use crate::sema::infer::Form;
use crate::sema::table::{Flight, Table};
use crate::sema::ty::Ty;

use super::super::SummaryKind;
use super::WarmFilter;

fn count_table<K, V>(table: &Table<K, V>, mut ok: impl FnMut(&K, &V) -> bool) -> u32 {
    let mut n = 0u32;
    table.retain(|k, v| {
        if ok(k, v) {
            n += 1;
        }
        true
    });
    n
}

fn count_flight<K: Eq + std::hash::Hash, V>(
    flight: &Flight<K, V>,
    mut ok: impl FnMut(&K, &V) -> bool,
) -> u32 {
    let mut n = 0u32;
    flight.retain_entries(|k, v| {
        if ok(k, v) {
            n += 1;
        }
        true
    });
    n
}

fn count_raw_table<K, V>(table: &Table<K, V>, total: &mut u32, threshold: u32) -> bool {
    let mut n = *total;
    if n < threshold {
        table.retain(|_, _| {
            n = n.saturating_add(1);
            true
        });
    }
    *total = n;
    n >= threshold
}

fn count_raw_flight<K: Eq + std::hash::Hash, V>(
    flight: &Flight<K, V>,
    total: &mut u32,
    threshold: u32,
) -> bool {
    let mut n = *total;
    if n < threshold {
        flight.retain_entries(|_, _| {
            n = n.saturating_add(1);
            true
        });
    }
    *total = n;
    n >= threshold
}

pub(super) fn raw_entries_at_least(store: &crate::sema::Cache, threshold: u32) -> bool {
    let lk = &store.lookups;
    let mut total = 0u32;
    macro_rules! check {
        ($($t:expr),+ $(,)?) => {
            $(if count_raw_table($t, &mut total, threshold) { return true; })+
        };
    }
    check!(
        &store.signatures,
        &store.adts,
        &store.layouts,
        &store.tails,
        &store.headers,
        &store.evaluated,
        &lk.modules,
        &lk.owners,
        &lk.heads,
        &lk.trait_defs,
        &lk.discriminants,
        &lk.reprs,
        &lk.attrs,
        &lk.drops,
        &lk.own_drops,
        &lk.unpins,
        &lk.freezes,
        &lk.glues,
        &lk.fields,
        &lk.forms,
        &lk.futures,
        &lk.scope_traits,
        &lk.trait_autos,
        &lk.type_defs,
        &lk.impl_traits,
        &lk.selections,
        &lk.auto_traits,
        &lk.provisional_rets,
        &lk.projections,
        &lk.trait_calls,
        &lk.assoc_types,
        &lk.verdicts,
    );
    count_raw_flight(&lk.checked_impls, &mut total, threshold)
        || count_raw_flight(&lk.reveals, &mut total, threshold)
        || count_raw_flight(&lk.summaries, &mut total, threshold)
}

pub(super) fn count_fixed_entries(store: &crate::sema::Cache, filter: &WarmFilter<'_>) -> u32 {
    let memo = RefCell::new(Vec::new());
    let ok_ty = |t: Ty| filter.ty(&store.types, t, &mut memo.borrow_mut());
    let ok_s = |s| filter.substs(&store.types, s, &mut memo.borrow_mut());
    let lk = &store.lookups;
    let mut total = 0u32;
    total += count_table(&store.signatures, |k, _| filter.item(*k));
    total += count_table(&store.adts, |k, _| filter.item(*k));
    total += count_table(&store.layouts, |&ty, l| {
        !matches!(l.variants, Variants::Multiple { .. }) && ok_ty(ty)
    });
    total += count_table(&store.tails, |&ty, v| ok_ty(ty) && ok_ty(*v));
    total += count_table(&store.headers, |k, _| filter.impl_ref(*k));
    total += count_table(&store.evaluated, |&(i, s), _| filter.item(i) && ok_s(s));
    total += count_table(&lk.modules, |&k, _| filter.krate(k));
    total += count_table(&lk.owners, |&k, _| filter.krate(k));
    total += count_table(&lk.heads, |h, idx| filter.holder(*h) && idx.complete());
    total += count_table(&lk.trait_defs, |k, _| filter.item(*k));
    total += count_table(&lk.discriminants, |k, _| filter.item(*k));
    total += count_table(&lk.reprs, |k, _| filter.item(*k));
    total += count_table(&lk.attrs, |k, _| filter.item(*k));
    total += count_table(&lk.drops, |&ty, _| ok_ty(ty));
    total += count_table(&lk.own_drops, |&ty, _| ok_ty(ty));
    total += count_table(&lk.unpins, |&ty, _| ok_ty(ty));
    total += count_table(&lk.freezes, |&ty, _| ok_ty(ty));
    total += count_table(&lk.glues, |&ty, _| ok_ty(ty));
    total += count_table(&lk.fields, |&ty, _| ok_ty(ty));
    total += count_table(&lk.forms, |&(f, _), _| match f {
        Form::Adt(i) | Form::Fn(i) => filter.item(i),
        Form::Impl(b) => filter.impl_ref(b),
    });
    total += count_table(&lk.futures, |&ty, v| ok_ty(ty) && ok_ty(*v));
    total += count_table(&lk.scope_traits, |&m, _| filter.mod_ref(m));
    total += count_table(&lk.trait_autos, |k, _| filter.item(*k));
    total += count_table(&lk.type_defs, |&(sc, _), def| {
        filter.mod_ref(sc.0)
            && sc.1.is_none_or(|i| filter.item(i))
            && sc.2.is_none_or(|i| filter.item(i))
            && filter.def(def)
    });
    total += count_table(&lk.impl_traits, |k, v| {
        filter.impl_ref(*k) && v.is_none_or(|i| filter.item(i))
    });
    total += count_table(&lk.selections, |&(tr, ty, s), v| {
        filter.item(tr)
            && ok_ty(ty)
            && ok_s(s)
            && v.is_none_or(|(b, ss)| filter.impl_ref(b) && ok_s(ss))
    });
    total += count_table(&lk.auto_traits, |&(tr, ty), _| filter.item(tr) && ok_ty(ty));
    total += count_flight(&lk.checked_impls, |k, _| filter.impl_ref(*k));
    total += count_flight(&lk.reveals, |&(i, s), &t| {
        filter.item(i) && ok_s(s) && ok_ty(t)
    });
    total += count_table(&lk.provisional_rets, |&(i, s), &t| {
        filter.item(i) && ok_s(s) && ok_ty(t)
    });
    total += count_table(&lk.projections, |&(tr, ty, a1, _, a2), v| {
        filter.item(tr) && ok_ty(ty) && ok_s(a1) && ok_s(a2) && v.is_none_or(&ok_ty)
    });
    total += count_table(&lk.trait_calls, |&(m, s), _| filter.item(m) && ok_s(s));
    total += count_table(&lk.assoc_types, |&(tr, ty, a1, assoc, a2), val| {
        filter.item(tr) && ok_ty(ty) && ok_s(a1) && filter.item(assoc) && ok_s(a2) && ok_ty(*val)
    });
    total += count_table(&lk.verdicts, |k, _| filter.item(*k));
    total += count_flight(&lk.summaries, |k, v| {
        filter.item(*k) && !matches!(v, SummaryKind::Error(_))
    });
    total
}
