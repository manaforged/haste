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
    for (item, kind) in items {
        match kind {
            ItemKind::Const => check_const_item(cx, reached, item, force_commit),
            ItemKind::Static => check_static_item(cx, reached, item, force_commit),
            ItemKind::Enum | ItemKind::Union | ItemKind::Struct => {
                check_adt_item(cx, reached, item, kind, force_commit);
            }
            _ => {}
        }
    }
    for (block, item) in impls {
        check_impl_block(cx, reached, block, item, force_commit);
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

fn check_const_item(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    item: ItemRef,
    force_commit: bool,
) {
    if cx.foreign(item) || expected_param_count(cx, item) != Some(0) {
        return;
    }
    let asks0 = cx.asks.count();
    let Ok(body) = cx.body(item) else {
        return;
    };
    if body.root.is_none() {
        return;
    }
    let instance = (item, Substs::EMPTY);
    let typed = match reached.index.get(&instance).copied() {
        Some(at) if reached.states[at] == ReachedEntry::Settled => match &reached.list[at].1 {
            Ok(typed) => Arc::clone(typed),
            Err(_) => {
                reached.note_crate_error(at);
                return;
            }
        },
        _ => {
            let mut unlaid = Vec::new();
            match typed_once(cx, &instance, &mut unlaid) {
                Ok(typed) => {
                    reached.unlaid.extend(unlaid);
                    typed
                }
                Err(err) => {
                    if force_commit || cx.asks.count() == asks0 {
                        reached.record_crate_error(instance, err);
                    }
                    return;
                }
            }
        }
    };
    if !const_may_fail(&body, &typed) || !evaluate::is_installed() {
        return;
    }
    let Some(bits) = const_eval_bits(cx, typed.ret) else {
        return;
    };
    let eval_asks0 = cx.asks.count();
    if let Err(err) = evaluate::run(cx, &instance, bits)
        && (force_commit || cx.asks.count() == eval_asks0)
    {
        let err = anyhow::anyhow!("evaluation of constant value failed: {err:#}");
        reached.record_crate_error(instance, crate::diag::format_error(cx.name(item), &err));
    }
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

fn check_static_item(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    item: ItemRef,
    force_commit: bool,
) {
    let asks0 = cx.asks.count();
    if let Err(err) = Infer::check_static_item_sync(cx, item)
        && (force_commit || cx.asks.count() == asks0)
    {
        reached.record_crate_error(
            (item, Substs::EMPTY),
            crate::diag::format_error(cx.name(item), &err),
        );
    }
}

fn check_impl_block(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    block: ImplRef,
    item: ItemRef,
    force_commit: bool,
) {
    let asks0 = cx.asks.count();
    if let Err(err) = Infer::check_local_trait_impl(cx, block)
        && (force_commit || cx.asks.count() == asks0)
    {
        reached.record_crate_error(
            (item, Substs::EMPTY),
            crate::diag::format_error(cx.name(item), &err),
        );
    }
}

fn check_adt_item(
    cx: &mut Context<'_>,
    reached: &mut Reached,
    item: ItemRef,
    kind: ItemKind,
    force_commit: bool,
) {
    let asks0 = cx.asks.count();
    if let Err(err) = check_adt_inner(cx, item, kind)
        && (force_commit || cx.asks.count() == asks0)
    {
        reached.record_crate_error(
            (item, Substs::EMPTY),
            crate::diag::format_error(cx.name(item), &err),
        );
    }
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
