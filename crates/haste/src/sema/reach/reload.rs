use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashSet as HashSet;

use crate::names::{Def, ItemRef, ModRef};
use crate::outline::Outline;

use super::super::impls::{Holder, ImplRef};
use super::super::infer::{
    Coercion, Form, LazyTables, Res, Shape, Typed, is_concrete_fn, shape_calls,
};
use super::super::lookups::{Cache, Naming};
use super::super::substs::Substs;
use super::super::summary::{ConstHole, PolyObligation, PolySummary, SummaryKind};
use super::super::{Context, write_lock};
use super::{Instance, Reached, ReachedEntry};

/// Fine-grained invalidation scope produced when diffing an incremental reload.
pub struct ReloadScope {
    pub crates: Vec<u32>,
    pub dependents: Vec<u32>,
    pub dirty_crates: Vec<u32>,
    pub dirty_files: Vec<(u32, u32)>,
    pub old_dirty_items: HashSet<ItemRef>,
    pub all_dirty_items: HashSet<ItemRef>,
    pub dirty_modules: HashSet<ModRef>,
}

impl ReloadScope {
    #[inline]
    pub(crate) fn is_dirty_file(&self, krate: u32, file: u32) -> bool {
        self.dirty_files.contains(&(krate, file))
    }

    #[inline]
    fn dirty_in(&self, items: &HashSet<ItemRef>, mut item: ItemRef) -> bool {
        if self.is_dirty_file(item.krate, item.file) || items.contains(&item) {
            return true;
        }
        while let Some(owner) = item.declaring_body() {
            if items.contains(&owner) {
                return true;
            }
            item = owner;
        }
        false
    }

    #[inline]
    pub(crate) fn is_item_dirty(&self, item: ItemRef) -> bool {
        self.dirty_in(&self.all_dirty_items, item)
    }

    #[inline]
    pub(crate) fn is_old_ref_dirty(&self, item: ItemRef) -> bool {
        self.dirty_in(&self.old_dirty_items, item)
    }
}

fn res_dirty<F: Fn(ItemRef) -> bool>(res: Res, naming: &mut Naming<'_, F>) -> bool {
    match res {
        Res::Local(_) => false,
        Res::Fn(item, substs) | Res::Ctor(item, _, substs) => !naming.fn_kept(item, substs),
        Res::Const(ty) => naming.names(ty),
    }
}

fn coercion_dirty<F: Fn(ItemRef) -> bool>(coercion: &Coercion, naming: &mut Naming<'_, F>) -> bool {
    match coercion {
        Coercion::Len(ty) => naming.names(*ty),
        Coercion::Vtable(vt) => vt.calls().into_iter().any(|res| res_dirty(*res, naming)),
        Coercion::Reify(res) => res_dirty(*res, naming),
        Coercion::Deref(steps, target) => {
            naming.names(*target)
                || steps
                    .iter()
                    .any(|(ty, res)| naming.names(*ty) || res.is_some_and(|r| res_dirty(r, naming)))
        }
        Coercion::Autos | Coercion::Upcast(_) => false,
        Coercion::Field { target, inner } => naming.names(*target) || coercion_dirty(inner, naming),
    }
}

fn poly_summary_dirty<F: Fn(ItemRef) -> bool>(
    summary: &PolySummary,
    naming: &mut Naming<'_, F>,
    old_ref_dirty: &F,
) -> bool {
    if summary.locals.iter().any(|l| naming.names(l.ty))
        || summary.ret.is_some_and(|t| naming.names(t))
        || summary
            .layout_tys
            .iter()
            .chain(&summary.drop_tys)
            .chain(&summary.parametric_tys)
            .any(|&t| naming.names(t))
        || summary.fixed_drops.iter().any(|&r| res_dirty(r, naming))
        || summary
            .fixed_calls
            .iter()
            .any(|&(i, s)| !naming.fn_kept(i, s))
        || summary
            .direct_calls
            .iter()
            .chain(&summary.call_holes)
            .any(|&(_, i, s)| !naming.fn_kept(i, s))
        || summary.obligations.iter().any(|ob| match ob {
            PolyObligation::Prove { of, self_ty, args } => !naming.trait_kept(*of, *self_ty, *args),
            PolyObligation::Call { found, tuple, .. } => {
                naming.names(*found) || naming.names(*tuple)
            }
        })
        || summary.const_holes.iter().any(|hole| match hole {
            ConstHole::Value(ty) => naming.names(*ty),
            ConstHole::OffsetOf { steps, .. } => steps.iter().any(|&(ty, _)| naming.names(ty)),
            ConstHole::Intrinsic { item, substs } => !naming.fn_kept(*item, *substs),
        })
        || summary.projections.iter().any(|&proj_id| {
            let d = naming.types.proj(proj_id);
            old_ref_dirty(d.of)
                || old_ref_dirty(d.assoc)
                || naming.names(d.self_ty)
                || naming.substs(d.args)
                || naming.substs(d.own)
        })
        || summary
            .op_holes
            .iter()
            .any(|(_, _, r)| res_dirty(*r, naming))
    {
        return true;
    }
    let t = &summary.template;
    t.exprs.iter().any(|&ty| naming.names(ty))
        || t.pats.values().any(|&ty| naming.names(ty))
        || t.res
            .values()
            .chain(t.pat_res.values())
            .chain(t.derefs.values())
            .chain(t.operators.values())
            .chain(t.try_blocks.values())
            .chain(t.box_frees.values())
            .any(|&r| res_dirty(r, naming))
        || t.coercions.values().any(|c| coercion_dirty(c, naming))
}

fn typed_mentions_dirty<F: Fn(ItemRef) -> bool>(
    typed: &Typed,
    naming: &mut Naming<'_, F>,
    old_ref_dirty: &F,
) -> bool {
    if typed.locals.iter().any(|l| naming.names(l.ty))
        || typed.ret.is_some_and(|ty| naming.names(ty))
        || typed
            .drops
            .iter()
            .copied()
            .chain(shape_calls(&typed.shape).copied())
            .any(|r| res_dirty(r, naming))
    {
        return true;
    }
    if let Shape::Closure(cb) = &typed.shape
        && (old_ref_dirty(cb.parent) || naming.names(cb.closure.ret))
    {
        return true;
    }
    match &typed.tables {
        LazyTables::Eager(t) => {
            t.exprs.iter().any(|&ty| naming.names(ty))
                || t.pats.values().any(|&ty| naming.names(ty))
                || t.closures.values().any(|c| naming.names(c.ret))
                || t.fors.values().any(|f| {
                    naming.names(f.iter)
                        || naming.names(f.option)
                        || res_dirty(f.into_iter, naming)
                        || res_dirty(f.next, naming)
                })
                || t.tries.values().any(|tr| {
                    naming.names(tr.flow)
                        || res_dirty(tr.branch, naming)
                        || res_dirty(tr.from_residual, naming)
                })
                || t.awaits.values().any(|a| {
                    naming.names(a.future)
                        || naming.names(a.polled)
                        || res_dirty(a.into_future, naming)
                        || res_dirty(a.poll, naming)
                })
                || t.res
                    .values()
                    .chain(t.pat_res.values())
                    .chain(t.derefs.values())
                    .chain(t.operators.values())
                    .chain(t.try_blocks.values())
                    .chain(t.box_frees.values())
                    .any(|&r| res_dirty(r, naming))
                || t.coercions.values().any(|c| coercion_dirty(c, naming))
        }
        LazyTables::Poly(poly) => {
            poly.calls().any(|(i, s)| !naming.fn_kept(i, s))
                || poly.projs.iter().any(|&(_, ty)| naming.names(ty))
                || poly_summary_dirty(&poly.summary, naming, old_ref_dirty)
        }
        LazyTables::CheckOnly(check) => {
            check.calls().any(|(i, s)| !naming.fn_kept(i, s))
                || poly_summary_dirty(&check.summary, naming, old_ref_dirty)
        }
    }
}

/// Retains unaffected polymorphic failures and collects local items needing `check_poly_items`.
pub fn dirty_poly_items(
    cases: &[ItemRef],
    poly_failures: &mut Vec<(ItemRef, String)>,
    scope: &ReloadScope,
    evicted_summaries: &HashSet<ItemRef>,
) -> Vec<ItemRef> {
    let has_old_dirty = !scope.old_dirty_items.is_empty() || !scope.dirty_files.is_empty();
    let retry_crates = if has_old_dirty {
        &scope.dependents
    } else {
        &scope.crates
    };
    let mut retry = HashSet::default();
    poly_failures.retain(|(item, _)| {
        if retry_crates.contains(&item.krate) {
            retry.insert(*item);
            false
        } else {
            true
        }
    });
    cases
        .iter()
        .copied()
        .filter(|&item| {
            (scope.crates.contains(&item.krate) && scope.is_item_dirty(item))
                || (retry_crates.contains(&item.krate) && retry.contains(&item))
                || evicted_summaries.contains(&item)
        })
        .collect()
}

/// Splits concrete `cases` into unchanged root indices kept in place and dirty root instances to re-reach.
pub fn split_dirty_roots(
    cx: &mut Context<'_>,
    cases: &[ItemRef],
    reached: &Reached,
    prev_roots: &[usize],
    touched: &[usize],
) -> (Vec<usize>, Vec<Instance>) {
    let mut touched_mask = vec![false; reached.list.len()];
    for &idx in touched {
        if idx < touched_mask.len() {
            touched_mask[idx] = true;
        }
    }
    let mut was_root = vec![false; reached.list.len()];
    for &idx in prev_roots {
        if idx < was_root.len() {
            was_root[idx] = true;
        }
    }
    let mut preserved = Vec::with_capacity(prev_roots.len());
    let mut dirty = Vec::new();
    for &item in cases {
        let inst = (item, Substs::EMPTY);
        if let Some(&idx) = reached.index.get(&inst)
            && was_root.get(idx).copied().unwrap_or(false)
            && !touched_mask.get(idx).copied().unwrap_or(false)
        {
            preserved.push(idx);
        } else if is_concrete_fn(cx, item) {
            dirty.push(inst);
        }
    }
    (preserved, dirty)
}

/// Closes `dead_set` backwards along `calls` and `indirect` so no surviving entry points into `dead`.
pub(in crate::sema::reach) fn close_dead_callers(
    reached: &Reached,
    dead_set: &mut [bool],
) -> Vec<usize> {
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); dead_set.len()];
    for (caller, edges) in reached.calls.iter().enumerate() {
        for &c in edges {
            if c < rev.len() {
                rev[c].push(caller);
            }
        }
    }
    for (caller, edges) in reached.indirect.iter().enumerate() {
        for &c in edges {
            if c < rev.len() {
                rev[c].push(caller);
            }
        }
    }
    let mut stack: Vec<usize> = dead_set
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| d.then_some(i))
        .collect();
    while let Some(c) = stack.pop() {
        for &caller in &rev[c] {
            if !dead_set[caller] && reached.states.get(caller) != Some(&ReachedEntry::Replaced) {
                dead_set[caller] = true;
                stack.push(caller);
            }
        }
    }
    dead_set
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| d.then_some(i))
        .collect()
}

fn body_expansion_missing(outline: &Outline, crates: &[u32], item: ItemRef) -> bool {
    if item.outer().is_none() {
        return false;
    }
    let owner = item.body_owner();
    crates.contains(&owner.krate)
        && outline
            .crates
            .get(owner.krate as usize)
            .and_then(|c| c.files.get(owner.file as usize))
            .is_none_or(|f| !f.body_read(owner.krate, owner.item))
}

/// Drops cached items, types, lookups, and reached instances touched by `scope`, returning `(dead, in_place, evicted_summaries)`.
pub fn reload_invalidate(
    store: &mut Cache,
    outline: &Outline,
    reached: &Reached,
    scope: &ReloadScope,
) -> (Vec<usize>, Vec<usize>, HashSet<ItemRef>) {
    let mut missing_owners = HashSet::default();
    let mut note_missing = |item: ItemRef| {
        if body_expansion_missing(outline, &scope.crates, item) {
            missing_owners.insert(item.body_owner());
        }
    };
    for (idx, ((item, _), _)) in reached.list.iter().enumerate() {
        if reached.states.get(idx) != Some(&ReachedEntry::Replaced) {
            note_missing(*item);
        }
    }
    store.headers.retain(|block, _| {
        if let ImplRef::Body(item) = *block {
            note_missing(item);
        }
        true
    });
    store.signatures.retain(|item, _| {
        note_missing(*item);
        true
    });
    store.bodies.retain(|item, _| {
        note_missing(*item);
        true
    });
    store.adts.retain(|item, _| {
        note_missing(*item);
        true
    });
    let item_dirty = |item: ItemRef| {
        scope.is_item_dirty(item)
            || missing_owners.contains(&item)
            || body_expansion_missing(outline, &scope.crates, item)
    };
    let old_ref_dirty = |item: ItemRef| {
        scope.is_old_ref_dirty(item)
            || missing_owners.contains(&item)
            || body_expansion_missing(outline, &scope.crates, item)
    };
    let block_kept = |block: &ImplRef| match *block {
        ImplRef::Module { krate, index } => {
            scope.dirty_files.is_empty()
                || outline.crates[krate as usize]
                    .impls
                    .get(index as usize)
                    .is_some_and(|e| {
                        !scope.is_dirty_file(krate, e.file as u32)
                            && !e
                                .parts
                                .iter()
                                .any(|&p| scope.is_dirty_file(krate, p as u32))
                    })
        }
        ImplRef::Body(item) => !item_dirty(item),
    };
    let mut naming = Naming::new(&store.types, &old_ref_dirty);
    let has_old_dirty = !scope.old_dirty_items.is_empty()
        || !scope.dirty_files.is_empty()
        || !missing_owners.is_empty();
    let has_any_dirty = has_old_dirty || !scope.all_dirty_items.is_empty();
    let mut touched_set = vec![false; reached.list.len()];
    let mut touched_instances: HashSet<Instance> = HashSet::default();
    for (idx, ((item, substs), res)) in reached.list.iter().enumerate() {
        if reached.states.get(idx) == Some(&ReachedEntry::Replaced) {
            continue;
        }
        let dirty = res.is_err()
            || item_dirty(*item)
            || (has_old_dirty
                && (naming.substs(*substs)
                    || res.as_ref().is_ok_and(|typed| {
                        typed_mentions_dirty(typed, &mut naming, &old_ref_dirty)
                    })));
        if dirty {
            touched_set[idx] = true;
            touched_instances.insert((*item, *substs));
        }
    }
    for &idx in &reached.crate_errors {
        if let Some(((item, substs), _)) = reached.list.get(idx)
            && reached.states.get(idx) != Some(&ReachedEntry::Replaced)
        {
            touched_set[idx] = true;
            touched_instances.insert((*item, *substs));
        }
    }
    let mut opaque_dirty = HashSet::default();
    for (idx, &dirty) in touched_set.iter().enumerate() {
        if dirty
            && store
                .signatures
                .get(&reached.list[idx].0.0)
                .is_some_and(|s| s.opaque || s.is_async)
        {
            opaque_dirty.insert(idx);
        }
    }
    if !opaque_dirty.is_empty() {
        loop {
            let mut added = false;
            for (caller, callees) in reached.calls.iter().enumerate() {
                if !touched_set[caller] && callees.iter().any(|c| opaque_dirty.contains(c)) {
                    touched_set[caller] = true;
                    touched_instances.insert(reached.list[caller].0);
                    if store
                        .signatures
                        .get(&reached.list[caller].0.0)
                        .is_some_and(|s| s.opaque || s.is_async)
                    {
                        opaque_dirty.insert(caller);
                    }
                    added = true;
                }
            }
            if !added {
                break;
            }
        }
    }
    let mut dead_set = vec![false; reached.list.len()];
    let mut in_place = Vec::new();
    for (idx, &dirty) in touched_set.iter().enumerate() {
        if !dirty {
            continue;
        }
        let ((item, _), res) = &reached.list[idx];
        let keep_in_place = !has_old_dirty
            && reached.states.get(idx) == Some(&ReachedEntry::Settled)
            && res.is_ok()
            && !opaque_dirty.contains(&idx)
            && !old_ref_dirty(*item)
            && outline
                .crates
                .get(item.krate as usize)
                .and_then(|c| c.files.get(item.file as usize))
                .and_then(|f| f.items.items.get(item.item as usize))
                .is_some();
        if keep_in_place {
            in_place.push(idx);
        } else {
            dead_set[idx] = true;
        }
    }
    let touched = if dead_set.iter().any(|&d| d) {
        close_dead_callers(reached, &mut dead_set)
    } else {
        Vec::new()
    };

    macro_rules! retain_each {
        ($pred:expr; $($t:expr),+ $(,)?) => { $($t.retain($pred);)+ };
    }
    if has_any_dirty {
        retain_each!(
            |item, _| !item_dirty(*item);
            store.signatures, store.bodies, store.adts, store.lookups.trait_autos,
            store.lookups.trait_defs, store.lookups.discriminants, store.lookups.reprs,
            store.lookups.attrs
        );
        store.headers.retain(|block, _| block_kept(block));
    }
    if has_old_dirty || !touched_instances.is_empty() {
        retain_each!(
            |instance, _| {
                !item_dirty(instance.0)
                    && !(has_old_dirty && naming.substs(instance.1))
                    && !touched_instances.contains(instance)
            };
            store.finals, store.evaluated, store.lookups.provisional_rets
        );
    }
    if has_old_dirty {
        retain_each!(
            |ty, _| !naming.names(*ty);
            store.layouts, store.tails, store.closures, store.lookups.drops, store.lookups.own_drops,
            store.lookups.unpins, store.lookups.freezes, store.lookups.glues, store.lookups.fields,
            store.lookups.futures
        );
    }
    store.runner.clear();

    let lk = &mut store.lookups;
    if has_any_dirty {
        lk.forms.retain(|(form, _), found| {
            let own = match *form {
                Form::Adt(i) | Form::Fn(i) => !old_ref_dirty(i),
                Form::Impl(b) => block_kept(&b),
            };
            own && (!has_old_dirty
                || found
                    .iter()
                    .flat_map(|t| t.iter().flatten())
                    .all(|&ty| !naming.names(ty)))
        });
        lk.impl_traits.retain(|block, _| block_kept(block));
        lk.checked_impls.retain_keys(|block| block_kept(block));
        lk.selections.retain(|(tr, ty, s), found| {
            (!has_old_dirty || naming.trait_kept(*tr, *ty, *s))
                && found
                    .as_ref()
                    .is_none_or(|(b, a)| block_kept(b) && !(has_old_dirty && naming.substs(*a)))
        });
        lk.trait_calls.retain(|(i, s), r| {
            !has_old_dirty || (naming.fn_kept(*i, *s) && !res_dirty(*r, &mut naming))
        });
    }
    if has_old_dirty || !touched_instances.is_empty() {
        lk.reveals.retain(|instance| {
            !item_dirty(instance.0)
                && !(has_old_dirty && naming.substs(instance.1))
                && !touched_instances.contains(instance)
        });
    }
    if !scope.dirty_crates.is_empty() || has_any_dirty {
        lk.heads.retain(|holder, _| match *holder {
            Holder::Crate(krate) => !scope.dirty_crates.contains(&krate),
            Holder::Body(item) => !item_dirty(item),
        });
    }
    if !scope.dirty_crates.is_empty() {
        lk.modules.retain(|k, _| !scope.dirty_crates.contains(k));
        lk.owners.retain(|k, _| !scope.dirty_crates.contains(k));
        if let Some(heads) = &mut lk.sysroot_heads
            && heads.keys().any(|k| scope.dirty_crates.contains(k))
        {
            let map = Arc::make_mut(heads);
            map.retain(|k, _| !scope.dirty_crates.contains(k));
            if map.is_empty() {
                lk.sysroot_heads = None;
            }
        }
    }
    if !scope.dirty_modules.is_empty() {
        lk.scope_traits
            .retain(|home, _| !scope.dirty_modules.contains(home));
    }
    if !scope.dirty_modules.is_empty() || has_old_dirty {
        lk.type_defs.retain(|key, def| {
            !scope.dirty_modules.contains(&key.0.0)
                && !key.0.1.is_some_and(&old_ref_dirty)
                && !key.0.2.is_some_and(&old_ref_dirty)
                && match *def {
                    Def::Item(i) | Def::Variant(i, _) => !old_ref_dirty(i),
                    Def::Module(m) => !scope.dirty_modules.contains(&m),
                    Def::Primitive(_) => true,
                }
        });
    }
    if has_old_dirty {
        lk.auto_traits
            .retain(|(tr, ty), _| naming.trait_kept(*tr, *ty, Substs::EMPTY));
        lk.projections.retain(|(tr, ty, s, _, gat), found| {
            naming.trait_kept(*tr, *ty, *s)
                && !naming.substs(*gat)
                && found.is_none_or(|t| !naming.names(t))
        });
        lk.assoc_types.retain(|(tr, ty, s, a, gat), res| {
            naming.trait_kept(*tr, *ty, *s)
                && !old_ref_dirty(*a)
                && !naming.substs(*gat)
                && !naming.names(*res)
        });
    }
    let mut evicted_summaries = HashSet::default();
    lk.summaries.retain_entries(|item, kind| {
        let keep = !item_dirty(*item)
            && match kind {
                SummaryKind::Poly(summary) => {
                    !has_old_dirty || !poly_summary_dirty(summary, &mut naming, &old_ref_dirty)
                }
                SummaryKind::MonomorphicHints(hints) => {
                    !has_old_dirty || !hints.values().any(|&i| old_ref_dirty(i))
                }
                SummaryKind::MonomorphicFallback => true,
                SummaryKind::Error(_) => false,
            };
        if !keep {
            evicted_summaries.insert(*item);
        }
        keep
    });
    if !scope.dirty_crates.is_empty() {
        lk.file_index = Arc::new(OnceLock::new());
        let mut langs = write_lock(&lk.state.langs);
        if langs
            .as_ref()
            .is_some_and(|f| f.crates.iter().any(|k| scope.dirty_crates.contains(k)))
        {
            *langs = None;
        }
        drop(langs);
        let mut guard = write_lock(&lk.state.loaded);
        if guard.len() == outline.crates.len() {
            for &k in &scope.crates {
                guard[k as usize] = outline.crates[k as usize].extent();
            }
        }
    }
    (touched, in_place, evicted_summaries)
}
