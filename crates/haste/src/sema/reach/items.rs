use std::cell::RefCell;
use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{
    LazyTables, Shape, Typed, compiler_made, infer_after_compiler_made, mark_parent_laid,
    record_parent_drops, register,
};
use crate::sema::summary::{thaw, typed_once_poly};
use crate::sema::ty::{Kind, Ty};

use super::Instance;

pub(in crate::sema) fn infer_entry(
    cx: &mut Context<'_>,
    instance: &Instance,
    unlaid: &mut Vec<String>,
    check_undeclared: bool,
) -> Result<Arc<Typed>, String> {
    if check_undeclared && cx.undeclared(instance) {
        return Err("no longer declared".to_owned());
    }
    typed_once(cx, instance, unlaid)
}

std::thread_local! {
    static INFERRING: RefCell<Vec<Instance>> = const { RefCell::new(Vec::new()) };
}

struct InferGuard(bool);

impl InferGuard {
    fn enter(instance: Instance) -> Self {
        INFERRING.with(|active| active.borrow_mut().push(instance));
        Self(true)
    }
}

impl Drop for InferGuard {
    fn drop(&mut self) {
        if self.0 {
            INFERRING.with(|active| {
                active.borrow_mut().pop();
            });
        }
    }
}

pub(in crate::sema) fn is_inferring(instance: Instance) -> bool {
    INFERRING.with(|active| active.borrow().contains(&instance))
}

/// The inferred body of `instance`: the one an earlier pass kept, or a fresh inference that is kept when it type-checked, laid out, and asked for nothing.
pub(crate) fn typed_once(
    cx: &mut Context<'_>,
    instance: &Instance,
    unlaid: &mut Vec<String>,
) -> Result<Arc<Typed>, String> {
    if cx.types.substs_depth(instance.1) > crate::RECURSION_LIMIT {
        return Err(format!(
            "reached the recursion limit while instantiating `{}`",
            cx.name(instance.0)
        ));
    }
    let is_alloc_shim = cx.global_alloc_shim(instance.0).is_some();
    if !is_alloc_shim
        && let Some(typed) = cx.store.finals.get(instance).filter(|t| {
            cx.store.lookups.skip_warm_sysroot_bodies
                || !matches!(&t.tables, LazyTables::CheckOnly(_))
        })
    {
        register(cx, *instance, &typed);
        record_parent_drops(cx, *instance, &typed);
        mark_parent_laid(cx, *instance, &typed);
        return Ok(typed);
    }
    if !is_alloc_shim && let Some(result) = thaw::thawed(cx, instance, unlaid) {
        return result;
    }
    let typed = typed_fresh(cx, instance, unlaid)?;
    if !is_alloc_shim {
        thaw::note_fresh(cx, *instance, &typed);
    }
    Ok(typed)
}

fn typed_fresh(
    cx: &mut Context<'_>,
    instance: &Instance,
    unlaid: &mut Vec<String>,
) -> Result<Arc<Typed>, String> {
    let _guard = InferGuard::enter(*instance);
    let before = cx.quiet_mark();
    match compiler_made(cx, instance.0, instance.1) {
        Ok(Some(typed)) => {
            let typed = Arc::new(typed);
            thaw::keep_laid(cx, *instance, &typed, unlaid, before);
            return Ok(typed);
        }
        Err(error) => return Err(crate::diag::format_error(cx.name(instance.0), &error)),
        Ok(None) => {}
    }
    if let Some(result) = typed_once_poly(cx, instance, unlaid) {
        return result;
    }
    let typed = Arc::new(
        infer_after_compiler_made(cx, instance.0, instance.1, None)
            .map_err(|error| crate::diag::format_error(cx.name(instance.0), &error))?,
    );
    thaw::keep_laid(cx, *instance, &typed, unlaid, before);
    Ok(typed)
}

fn lay_out_ty(cx: &mut Context<'_>, ty: Ty) -> anyhow::Result<()> {
    if let Kind::Adt(item, _) = *cx.types.kind(ty)
        && cx.item(item).1.kind == ItemKind::Enum
    {
        cx.discriminants(item)?;
    }
    if matches!(
        cx.types.kind(ty),
        Kind::Adt(..) | Kind::Tuple(_) | Kind::Array(..) | Kind::Closure(..)
    ) {
        cx.layout(ty)?;
    }
    Ok(())
}

/// Computes the layout of each ADT, tuple and array type that lowering `typed` reads, so the names those layouts need are asked for while inference can still retry.
pub(crate) fn lay_out(cx: &mut Context<'_>, typed: &Typed) -> anyhow::Result<()> {
    if let Shape::Closure(body) = &typed.shape
        && body.parent_laid
    {
        for ty in typed.locals[body.parent_locals as usize..]
            .iter()
            .map(|local| local.ty)
            .chain(typed.ret)
        {
            if ty != Ty::NONE {
                lay_out_ty(cx, ty)?;
            }
        }
        return Ok(());
    }
    let scoped = typed.scoped(&cx.types);
    let tables = scoped.ensure_tables(&cx.types);
    let mut intrinsic_args = Vec::new();
    for (item, substs) in typed.calls() {
        if cx.intrinsic(item).is_some() {
            intrinsic_args.extend_from_slice(cx.types.substs(substs));
        }
    }
    let locals = typed.locals.iter().map(|local| local.ty);
    let pats = tables.pats.values().copied();
    let mut seen = FxHashSet::default();
    for ty in tables
        .exprs
        .iter()
        .copied()
        .chain(locals)
        .chain(pats)
        .chain(intrinsic_args)
    {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        lay_out_ty(cx, ty)?;
    }
    Ok(())
}
