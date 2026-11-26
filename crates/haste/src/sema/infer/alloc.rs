use anyhow::{Result, bail};

use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::Context;
use super::super::substs::Substs;
use super::super::traits::GlobalAllocOp;
use super::super::ty::Ty;
use super::{Local, Res, Shape, Typed, detached};

/// The resolved static allocator and `GlobalAlloc` trait method behind a `__rust_*alloc*` shim.
#[derive(Clone, Debug)]
pub struct GlobalAllocShim {
    pub op: GlobalAllocOp,
    pub allocator: Res,
    pub method: Res,
    pub alloc_ty: Ty,
    pub layout_ty: Ty,
}

fn find_global_alloc_trait(cx: &Context<'_>, core_krate: u32) -> Option<ItemRef> {
    let tree = &cx.outline.crates[core_krate as usize];
    for (file, entry) in tree.files.iter().enumerate() {
        for (at, item) in entry.items.items.iter().enumerate() {
            if item.kind == ItemKind::Trait
                && tree.env.enabled(&item.attrs)
                && item
                    .name
                    .is_some_and(|name| entry.source.text(name) == "GlobalAlloc")
            {
                return Some(ItemRef {
                    krate: core_krate,
                    file: file as u32,
                    item: at as u32,
                });
            }
        }
    }
    None
}

/// Synthesizes the typed shim body for `__rust_alloc`, `__rust_alloc_zeroed`, `__rust_dealloc`, or `__rust_realloc` forwarding to `#[global_allocator]`.
pub fn global_alloc_typed(
    cx: &mut Context<'_>,
    item: ItemRef,
    op: GlobalAllocOp,
    alloc_static: ItemRef,
) -> Result<Typed> {
    let (param_tys, ret_ty) = detached(cx, item, |infer| infer.fn_types(item, &[]))?;
    let alloc_ty = detached(cx, alloc_static, |infer| infer.fn_types(alloc_static, &[]))?.1;
    let Some(layout_item) = cx.lang_item("alloc_layout") else {
        bail!("core has no `Layout` lang item");
    };
    let layout_ty = cx.types.adt(layout_item, &[]);
    let Some(trait_item) = find_global_alloc_trait(cx, layout_item.krate) else {
        bail!("core has no `GlobalAlloc` trait");
    };
    let Some(trait_method) = cx.trait_member(trait_item, op.method_name()) else {
        bail!("`GlobalAlloc` has no `{}` method", op.method_name());
    };
    let method = detached(cx, item, |infer| {
        let substs = infer.cx.types.intern_substs(&[alloc_ty]);
        infer.trait_call(trait_method, substs)
    })?;
    let locals: Vec<Local> = param_tys.iter().map(|&ty| Local { ty }).collect();
    let params: Vec<u32> = (0..locals.len() as u32).collect();
    let shim = GlobalAllocShim {
        op,
        allocator: Res::Fn(alloc_static, Substs::EMPTY),
        method,
        alloc_ty,
        layout_ty,
    };
    Ok(Typed {
        locals,
        params,
        ret: Some(ret_ty),
        shape: Shape::GlobalAlloc(shim),
        ..Typed::default()
    })
}
