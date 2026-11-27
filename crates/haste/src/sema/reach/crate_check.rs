use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_abi::BackendRepr;

use crate::body::{BinOp, Body, Expr};
use crate::names::ItemRef;
use crate::outline::item::{Item, ItemKind};
use crate::outline::tree::Crate;
use crate::sema::Context;
use crate::sema::evaluate;
use crate::sema::impls::ImplRef;
use crate::sema::infer::{Infer, Typed, check_union_fields, type_params};
use crate::sema::substs::Substs;
use crate::sema::summary::expected_param_count;
use crate::sema::ty::Ty;

use super::{Instance, Reached, ReachedEntry, typed_once};

#[derive(Clone, Copy)]
enum CheckTask {
    Const(ItemRef),
    Static(ItemRef),
    Adt(ItemRef, ItemKind),
    Impl(ImplRef, ItemRef),
}

#[derive(Default)]
struct CheckOutcome {
    unlaid: Vec<String>,
    note_at: Option<usize>,
    error: Option<(Instance, String)>,
    deferred: bool,
}

fn apply_outcome(reached: &mut Reached, outcome: CheckOutcome) {
    reached.unlaid.extend(outcome.unlaid);
    if let Some(at) = outcome.note_at {
        reached.note_crate_error(at);
    }
    if let Some((instance, error)) = outcome.error {
        reached.record_crate_error(instance, error);
    }
}

fn run_check_task(
    cx: &mut Context<'_>,
    reached: &Reached,
    task: CheckTask,
    force_commit: bool,
) -> CheckOutcome {
    let deferred0 = cx.deferred_evals.len();
    let mut outcome = match task {
        CheckTask::Const(item) => check_const_outcome(cx, reached, item, force_commit),
        CheckTask::Static(item) => check_with_error(cx, item, force_commit, |cx| {
            Infer::check_static_item_sync(cx, item)
        }),
        CheckTask::Adt(item, kind) => {
            check_with_error(cx, item, force_commit, |cx| check_adt_inner(cx, item, kind))
        }
        CheckTask::Impl(block, item) => check_with_error(cx, item, force_commit, |cx| {
            Infer::check_local_trait_impl(cx, block)
        }),
    };
    if cx.deferred_evals.len() > deferred0 {
        outcome.deferred = true;
        outcome.unlaid.clear();
        outcome.note_at = None;
        outcome.error = None;
    }
    outcome
}

/// Validates local crate constants, statics, trait impls, and ADTs.
pub(super) fn check_local_crate_items(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    roots: &[Instance],
    force_commit: bool,
) {
    reached.clear_crate_errors();
    let crates = local_crates(cx, roots);
    let mut items = Vec::new();
    let mut impls = Vec::new();
    for krate in crates {
        collect_crate_items(cx, krate, &mut items);
        collect_crate_impls(cx, krate, &mut impls);
    }
    let mut tasks = Vec::with_capacity(items.len() + impls.len());
    for (item, kind) in items {
        match kind {
            ItemKind::Const => tasks.push(CheckTask::Const(item)),
            ItemKind::Static => tasks.push(CheckTask::Static(item)),
            ItemKind::Enum | ItemKind::Union | ItemKind::Struct => {
                tasks.push(CheckTask::Adt(item, kind));
            }
            _ => {}
        }
    }
    for (block, item) in impls {
        tasks.push(CheckTask::Impl(block, item));
    }
    let workers = crate::jobs().min(tasks.len() / 4);
    if tasks.len() < 8 || workers <= 1 {
        for task in tasks {
            let outcome = run_check_task(cx, reached, task, force_commit);
            apply_outcome(reached, outcome);
        }
        return;
    }
    let forks: Vec<Context<'_>> = (0..workers - 1).map(|_| cx.fork()).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    cx.in_parallel = true;
    let worker_loop = |worker: &mut Context<'_>| {
        let mut out = Vec::new();
        loop {
            let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&task) = tasks.get(idx) else {
                break;
            };
            out.push((idx, run_check_task(worker, reached, task, force_commit)));
        }
        out
    };
    let (mut results, joined) = crate::sync::scope(|scope| {
        let handles: Vec<_> = forks
            .into_iter()
            .map(|mut fork| {
                scope.spawn(move || {
                    let out = worker_loop(&mut fork);
                    let deferred = fork.take_deferred_evals();
                    let (_, asks) = fork.release();
                    (asks, deferred, out)
                })
            })
            .collect();
        let own = worker_loop(cx);
        let joined: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().map_or(None, Some))
            .collect();
        (own, joined)
    });
    cx.in_parallel = false;
    let mut deferred = cx.take_deferred_evals();
    for (asks, worker_deferred, worker_out) in joined.into_iter().flatten() {
        cx.asks.repeat(&asks);
        deferred.extend(worker_deferred);
        results.extend(worker_out);
    }
    if !deferred.is_empty() {
        super::wave::eval_deferred_list(cx, deferred);
    }
    results.sort_unstable_by_key(|&(idx, _)| idx);
    for (idx, outcome) in results {
        if outcome.deferred {
            let retried = run_check_task(cx, reached, tasks[idx], force_commit);
            apply_outcome(reached, retried);
        } else {
            apply_outcome(reached, outcome);
        }
    }
}

fn local_crates(cx: &Context<'_>, roots: &[Instance]) -> Vec<u32> {
    let mut crates = Vec::new();
    for &(item, _) in roots {
        if !cx.outline.is_fixed(item.krate) && !crates.contains(&item.krate) {
            crates.push(item.krate);
        }
    }
    if crates.is_empty() {
        for krate in 0..cx.outline.crates.len() as u32 {
            if !cx.outline.is_fixed(krate) {
                crates.push(krate);
            }
        }
    }
    crates
}

fn collect_crate_items(cx: &Context<'_>, krate: u32, out: &mut Vec<(ItemRef, ItemKind)>) {
    let tree = &cx.outline.crates[krate as usize];
    for (file_at, file) in tree.files.iter().enumerate() {
        let items = &file.items.items;
        for (at, entry) in items.iter().enumerate() {
            if !matches!(
                entry.kind,
                ItemKind::Const
                    | ItemKind::Static
                    | ItemKind::Enum
                    | ItemKind::Union
                    | ItemKind::Struct
            ) {
                continue;
            }
            if !is_checkable_item(tree, items, at, entry.kind) {
                continue;
            }
            out.push((
                ItemRef {
                    krate,
                    file: file_at as u32,
                    item: at as u32,
                },
                entry.kind,
            ));
        }
    }
}

fn collect_crate_impls(cx: &Context<'_>, krate: u32, out: &mut Vec<(ImplRef, ItemRef)>) {
    let tree = &cx.outline.crates[krate as usize];
    for (index, entry) in tree.impls.iter().enumerate() {
        let block = ImplRef::Module {
            krate,
            index: index as u32,
        };
        let item = ItemRef {
            krate,
            file: entry.file as u32,
            item: entry.at,
        };
        out.push((block, item));
    }
}

fn is_checkable_item(tree: &Crate, items: &[Item], at: usize, kind: ItemKind) -> bool {
    let mut cur = Some(at);
    while let Some(idx) = cur {
        let item = &items[idx];
        if !tree.env.enabled(&item.attrs) {
            return false;
        }
        if idx != at {
            let allowed_parent = match kind {
                ItemKind::Const => matches!(item.kind, ItemKind::Mod | ItemKind::Impl),
                _ => item.kind == ItemKind::Mod,
            };
            if !allowed_parent {
                return false;
            }
        }
        cur = item.parent.map(|p| p as usize);
    }
    true
}

fn check_const_outcome(
    cx: &mut Context<'_>,
    reached: &Reached,
    item: ItemRef,
    force_commit: bool,
) -> CheckOutcome {
    let mut outcome = CheckOutcome::default();
    if cx.foreign(item) || expected_param_count(cx, item) != Some(0) {
        return outcome;
    }
    let asks0 = cx.asks.count();
    let Ok(body) = cx.body(item) else {
        return outcome;
    };
    if body.root.is_none() {
        return outcome;
    }
    let instance = (item, Substs::EMPTY);
    let typed = match reached.index.get(&instance).copied() {
        Some(at) if reached.states[at] == ReachedEntry::Settled => match &reached.list[at].1 {
            Ok(typed) => Arc::clone(typed),
            Err(_) => {
                outcome.note_at = Some(at);
                return outcome;
            }
        },
        _ => {
            let mut unlaid = Vec::new();
            match typed_once(cx, &instance, &mut unlaid) {
                Ok(typed) => {
                    outcome.unlaid = unlaid;
                    typed
                }
                Err(err) => {
                    if force_commit || cx.asks.count() == asks0 {
                        outcome.error = Some((instance, err));
                    }
                    return outcome;
                }
            }
        }
    };
    if !const_may_fail(&body, &typed) || !evaluate::is_installed() {
        return outcome;
    }
    let Some(bits) = const_eval_bits(cx, typed.ret) else {
        return outcome;
    };
    let eval_asks0 = cx.asks.count();
    if let Err(err) = evaluate::run(cx, &instance, bits)
        && (force_commit || cx.asks.count() == eval_asks0)
    {
        let err = anyhow::anyhow!("evaluation of constant value failed: {err:#}");
        outcome.error = Some((instance, crate::diag::format_error(cx.name(item), &err)));
    }
    outcome
}

fn const_may_fail(body: &Body, typed: &Typed) -> bool {
    if typed.calls().next().is_some() {
        return true;
    }
    body.exprs.iter().any(|expr| {
        matches!(
            expr,
            Expr::Binary(BinOp::Div | BinOp::Rem | BinOp::Shl | BinOp::Shr, ..)
                | Expr::Index(..)
                | Expr::Loop { .. }
                | Expr::While { .. }
        )
    })
}

fn const_eval_bits(cx: &mut Context<'_>, ret: Option<Ty>) -> Option<u32> {
    let ty = ret?;
    let layout = cx.layout(ty).ok()?;
    let bytes = layout.size.bytes();
    if bytes == 0 {
        return Some(0);
    }
    if bytes <= 16
        && matches!(
            layout.backend_repr,
            BackendRepr::Scalar(_) | BackendRepr::ScalarPair { .. }
        )
    {
        return Some((bytes * 8) as u32);
    }
    None
}

fn check_with_error(
    cx: &mut Context<'_>,
    item: ItemRef,
    force_commit: bool,
    run: impl FnOnce(&mut Context<'_>) -> Result<()>,
) -> CheckOutcome {
    let mut outcome = CheckOutcome::default();
    let asks0 = cx.asks.count();
    if let Err(err) = run(cx)
        && (force_commit || cx.asks.count() == asks0)
    {
        outcome.error = Some((
            (item, Substs::EMPTY),
            crate::diag::format_error(cx.name(item), &err),
        ));
    }
    outcome
}

fn check_adt_inner(cx: &mut Context<'_>, item: ItemRef, kind: ItemKind) -> Result<()> {
    if kind == ItemKind::Enum {
        cx.discriminants(item)?;
    }
    if kind == ItemKind::Union {
        check_union_fields(cx, item)?;
    }
    if !cx.foreign(item) {
        let def = cx.adt_def(item)?;
        if cx.repr(item).transparent() && kind == ItemKind::Enum && def.variants.len() != 1 {
            bail!(
                "transparent enum {} needs exactly one variant, but has {}",
                cx.name(item),
                def.variants.len()
            );
        }
        if type_params(&def.generics.params).next().is_none() {
            cx.adt_layout(item, &[])?;
            Infer::check_item_dyn_object_safety(cx, item)?;
        }
    }
    Ok(())
}
