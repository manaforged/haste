use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::Signature;
use crate::diag::{Diagnostic, DiagnosticKind, Span};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::super::Context;
use super::super::super::impls::ImplRef;
use super::super::super::substs::Substs;
use super::super::super::ty::{Kind, Ty};
use super::super::scope::type_params;
use super::super::{Infer, detached};

fn block_is_user(cx: &Context<'_>, block: ImplRef) -> bool {
    let krate = match block {
        ImplRef::Module { krate, .. } => krate,
        ImplRef::Body(item) => item.krate,
    };
    !cx.outline.is_fixed(krate)
}

fn trait_has_methods(cx: &Context<'_>, of: ItemRef) -> bool {
    cx.children(of)
        .into_iter()
        .any(|member| cx.item(member).1.kind == ItemKind::Fn)
}

pub(super) fn check_item_trait_impls(
    cx: &mut Context<'_>,
    item: ItemRef,
    extra_traits: &[ItemRef],
) -> Result<()> {
    if cx.outline.is_fixed(item.krate) {
        return Ok(());
    }
    let mut checked_blocks = HashSet::default();
    if let Some(block) = cx.owner_impl(item) {
        checked_blocks.insert(block);
        check_trait_impl_block(cx, block)?;
    }
    let mut traits = super::super::scope::header_bound_traits(cx, item);
    for &of in extra_traits {
        if !traits.contains(&of) {
            traits.push(of);
        }
    }
    let user_crates: Vec<u32> = (0..cx.outline.crates.len() as u32)
        .filter(|&k| !cx.outline.is_fixed(k))
        .collect();
    if user_crates.is_empty() {
        return Ok(());
    }
    for of in traits {
        if !trait_has_methods(cx, of) {
            continue;
        }
        for block in cx.trait_impls(of, None, &user_crates) {
            if checked_blocks.insert(block) {
                check_trait_impl_block(cx, block)?;
            }
        }
    }
    Ok(())
}

pub(in super::super) fn check_trait_impl_block(cx: &mut Context<'_>, block: ImplRef) -> Result<()> {
    if !block_is_user(cx, block) {
        return Ok(());
    }
    let Some(of) = cx.impl_trait(block) else {
        return Ok(());
    };
    let Ok(header) = cx.impl_header(block) else {
        return Ok(());
    };
    if header.negative {
        return Ok(());
    }
    let outer = type_params(&header.generics).count();
    for impl_item in cx.members(block) {
        if cx.item(impl_item).1.kind != ItemKind::Fn {
            continue;
        }
        let name = cx.plain_name(impl_item).to_owned();
        let Some(trait_item) = cx.trait_member(of, &name) else {
            continue;
        };
        if cx.item(trait_item).1.kind != ItemKind::Fn {
            continue;
        }
        check_impl_method_compat(cx, block, outer, trait_item, impl_item, &name)?;
    }
    Ok(())
}

fn check_impl_method_compat(
    cx: &mut Context<'_>,
    block: ImplRef,
    outer: usize,
    trait_item: ItemRef,
    impl_item: ItemRef,
    name: &str,
) -> Result<()> {
    let (Ok(impl_sig), Ok(trait_sig)) = (cx.signature(impl_item), cx.signature(trait_item)) else {
        return Ok(());
    };
    let impl_own = type_params(&impl_sig.generics).count();
    let trait_own = type_params(&trait_sig.generics).count();
    let total = outer + impl_own;
    if total > u16::MAX as usize {
        return Ok(());
    }
    let diag = detached(cx, impl_item, |infer| {
        if infer.init_poly_for_item(impl_item, total as u16).is_err() {
            return Ok(None);
        }
        if let Some(poly) = &mut infer.poly {
            poly.skip_compat = true;
        }
        let Some(self_ty) = infer.self_ty else {
            return Ok(None);
        };
        let impl_args: Vec<Ty> = (0..outer)
            .map(|i| infer.cx.types.intern(Kind::Param(i as u16)))
            .collect();
        let Ok(trait_args) = infer.impl_trait_args(block, &impl_args, self_ty) else {
            return Ok(None);
        };
        let Ok((impl_params, impl_ret)) = infer.sig_types(&impl_sig) else {
            return Ok(None);
        };
        let mut trait_substs = Vec::with_capacity(1 + trait_args.len() + trait_own);
        trait_substs.push(self_ty);
        trait_substs.extend(trait_args);
        trait_substs.extend(
            (outer..outer + trait_own).map(|i| infer.cx.types.intern(Kind::Param(i as u16))),
        );
        let Ok(trait_frame) = infer.frame(trait_item, &trait_substs) else {
            return Ok(None);
        };
        let Ok((trait_params, trait_ret)) =
            infer.with_frame(trait_frame, |inner| inner.sig_types(&trait_sig))
        else {
            return Ok(None);
        };
        let of = infer.cx.impl_trait(block);
        let impl_params: Vec<Ty> = impl_params
            .into_iter()
            .map(|ty| normalize_block_proj(infer, block, of, self_ty, ty, 8))
            .collect();
        let impl_ret = normalize_block_proj(infer, block, of, self_ty, impl_ret, 8);
        let trait_params: Vec<Ty> = trait_params
            .into_iter()
            .map(|ty| normalize_block_proj(infer, block, of, self_ty, ty, 8))
            .collect();
        let trait_ret = normalize_block_proj(infer, block, of, self_ty, trait_ret, 8);
        Ok(compare_method_sigs(
            infer,
            name,
            (impl_item, &impl_sig, &impl_params, impl_ret, impl_own),
            (trait_item, &trait_sig, &trait_params, trait_ret, trait_own),
        ))
    });
    match diag {
        Ok(Some(err)) => Err(err.into()),
        _ => Ok(()),
    }
}

fn compat_diag(
    msg: String,
    primary_span: Option<Span>,
    primary_label: impl Into<String>,
    trait_span: Option<Span>,
) -> Diagnostic {
    Diagnostic::new(DiagnosticKind::Type, msg)
        .with_span(primary_span)
        .with_primary_label(primary_label)
        .with_label(trait_span, "type in trait")
}

fn compare_method_sigs(
    infer: &Infer<'_, '_>,
    name: &str,
    (impl_item, impl_sig, impl_params, impl_ret, impl_own): (ItemRef, &Signature, &[Ty], Ty, usize),
    (trait_item, trait_sig, trait_params, trait_ret, trait_own): (
        ItemRef,
        &Signature,
        &[Ty],
        Ty,
        usize,
    ),
) -> Option<Diagnostic> {
    let msg = format!("method `{name}` has an incompatible type for trait");
    let trait_head_span = item_name_span(infer.cx, trait_item);
    let impl_head_span = item_name_span(infer.cx, impl_item);
    if impl_params.len() != trait_params.len() || impl_sig.variadic != trait_sig.variadic {
        return Some(compat_diag(
            msg,
            impl_head_span,
            format!(
                "expected {} parameter(s), found {}",
                trait_params.len(),
                impl_params.len()
            ),
            trait_head_span,
        ));
    }
    for (idx, (&trait_p, &impl_p)) in trait_params.iter().zip(impl_params).enumerate() {
        if !types_compatible(infer, trait_p, impl_p) {
            return Some(compat_diag(
                msg,
                method_param_span(infer.cx, impl_item, impl_sig, idx),
                format!(
                    "expected `{}`, found `{}`",
                    infer.show(trait_p),
                    infer.show(impl_p)
                ),
                method_param_span(infer.cx, trait_item, trait_sig, idx),
            ));
        }
    }
    let opaque_or_async =
        trait_sig.is_async || trait_sig.opaque || impl_sig.is_async || impl_sig.opaque;
    if !opaque_or_async && !types_compatible(infer, trait_ret, impl_ret) {
        return Some(compat_diag(
            msg,
            impl_head_span,
            format!(
                "expected `{}`, found `{}`",
                infer.show(trait_ret),
                infer.show(impl_ret)
            ),
            trait_head_span,
        ));
    }
    if trait_own != impl_own || impl_sig.receiver.is_some() != trait_sig.receiver.is_some() {
        return Some(compat_diag(
            msg,
            impl_head_span,
            "method signature does not match trait",
            trait_head_span,
        ));
    }
    None
}

fn item_name_span(cx: &Context<'_>, item: ItemRef) -> Option<Span> {
    let (src, entry) = cx.item(item);
    Span::from_token(src, entry.name.unwrap_or(entry.start))
}

fn method_param_span(cx: &Context<'_>, item: ItemRef, sig: &Signature, idx: usize) -> Option<Span> {
    let (src, entry) = cx.item(item);
    let offset = usize::from(sig.receiver.is_some());
    if idx >= offset
        && let Some(&tok) = sig.param_starts.get(idx - offset)
        && let Some(span) = Span::from_token(src, tok)
    {
        return Some(span);
    }
    Span::from_token(src, entry.name.unwrap_or(entry.start))
}

fn normalize_block_proj(
    infer: &mut Infer<'_, '_>,
    block: ImplRef,
    of: Option<ItemRef>,
    self_ty: Ty,
    ty: Ty,
    depth: u8,
) -> Ty {
    if depth == 0 {
        return ty;
    }
    let ty = infer.resolve(ty);
    match *infer.cx.types.kind(ty) {
        Kind::Proj(proj_id) => {
            let proj = infer.cx.types.proj(proj_id);
            let norm_self =
                normalize_block_proj(infer, block, of, self_ty, proj.self_ty, depth - 1);
            let own: Vec<Ty> = infer
                .cx
                .types
                .substs(proj.own)
                .to_vec()
                .into_iter()
                .map(|t| normalize_block_proj(infer, block, of, self_ty, t, depth - 1))
                .collect();
            if of == Some(proj.of) && norm_self == self_ty {
                let assoc_name = infer.cx.plain_name(proj.assoc).to_owned();
                if let Ok(Some(resolved)) = infer.impl_alias(block, (&assoc_name, &own)) {
                    return normalize_block_proj(infer, block, of, self_ty, resolved, depth - 1);
                }
            }
            ty
        }
        Kind::Ref(m, inner) => {
            let next = normalize_block_proj(infer, block, of, self_ty, inner, depth - 1);
            if next == inner {
                ty
            } else {
                infer.cx.types.intern(Kind::Ref(m, next))
            }
        }
        Kind::Ptr(m, inner) => {
            let next = normalize_block_proj(infer, block, of, self_ty, inner, depth - 1);
            if next == inner {
                ty
            } else {
                infer.cx.types.intern(Kind::Ptr(m, next))
            }
        }
        Kind::Slice(inner) => {
            let next = normalize_block_proj(infer, block, of, self_ty, inner, depth - 1);
            if next == inner {
                ty
            } else {
                infer.cx.types.intern(Kind::Slice(next))
            }
        }
        Kind::Array(elem, len) => {
            let next = normalize_block_proj(infer, block, of, self_ty, elem, depth - 1);
            if next == elem {
                ty
            } else {
                infer.cx.types.intern(Kind::Array(next, len))
            }
        }
        Kind::Tuple(substs) => {
            let orig = infer.cx.types.substs(substs).to_vec();
            let next: Vec<Ty> = orig
                .iter()
                .map(|&t| normalize_block_proj(infer, block, of, self_ty, t, depth - 1))
                .collect();
            if next == orig {
                ty
            } else {
                let s = infer.cx.types.intern_substs(&next);
                infer.cx.types.intern(Kind::Tuple(s))
            }
        }
        Kind::Adt(item, substs) => {
            let orig = infer.cx.types.substs(substs).to_vec();
            let next: Vec<Ty> = orig
                .iter()
                .map(|&t| normalize_block_proj(infer, block, of, self_ty, t, depth - 1))
                .collect();
            if next == orig {
                ty
            } else {
                let s = infer.cx.types.intern_substs(&next);
                infer.cx.types.intern(Kind::Adt(item, s))
            }
        }
        Kind::FnPtr(params, ret) => {
            let orig = infer.cx.types.substs(params).to_vec();
            let next_p: Vec<Ty> = orig
                .iter()
                .map(|&t| normalize_block_proj(infer, block, of, self_ty, t, depth - 1))
                .collect();
            let next_r = normalize_block_proj(infer, block, of, self_ty, ret, depth - 1);
            if next_p == orig && next_r == ret {
                ty
            } else {
                let s = infer.cx.types.intern_substs(&next_p);
                infer.cx.types.intern(Kind::FnPtr(s, next_r))
            }
        }
        Kind::Pat(inner, range) => {
            let next = normalize_block_proj(infer, block, of, self_ty, inner, depth - 1);
            if next == inner {
                ty
            } else {
                infer.cx.types.intern(Kind::Pat(next, range))
            }
        }
        _ => ty,
    }
}

fn types_compatible(infer: &Infer<'_, '_>, a: Ty, b: Ty) -> bool {
    let a = infer.resolve(a);
    let b = infer.resolve(b);
    if a == b {
        return true;
    }
    match (*infer.cx.types.kind(a), *infer.cx.types.kind(b)) {
        (Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_), _)
        | (_, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) => true,
        (Kind::Proj(pa), Kind::Proj(pb)) => {
            let pa = infer.cx.types.proj(pa);
            let pb = infer.cx.types.proj(pb);
            pa.of == pb.of
                && pa.assoc == pb.assoc
                && types_compatible(infer, pa.self_ty, pb.self_ty)
                && substs_compatible(infer, pa.args, pb.args)
                && substs_compatible(infer, pa.own, pb.own)
        }
        (Kind::Prim(pa), Kind::Prim(pb)) => pa == pb,
        (Kind::Never, Kind::Never) => true,
        (Kind::Param(ia), Kind::Param(ib)) => ia == ib,
        (Kind::Value(va), Kind::Value(vb)) => va == vb,
        (Kind::Ref(ma, ia), Kind::Ref(mb, ib)) | (Kind::Ptr(ma, ia), Kind::Ptr(mb, ib)) => {
            ma == mb && types_compatible(infer, ia, ib)
        }
        (Kind::Slice(ia), Kind::Slice(ib)) => types_compatible(infer, ia, ib),
        (Kind::Array(ea, la), Kind::Array(eb, lb)) => {
            types_compatible(infer, ea, eb) && types_compatible(infer, la, lb)
        }
        (Kind::Tuple(sa), Kind::Tuple(sb)) => substs_compatible(infer, sa, sb),
        (Kind::Adt(ia, sa), Kind::Adt(ib, sb)) => ia == ib && substs_compatible(infer, sa, sb),
        (Kind::FnDef(ia, sa), Kind::FnDef(ib, sb)) => ia == ib && substs_compatible(infer, sa, sb),
        (Kind::FnPtr(pa, ra), Kind::FnPtr(pb, rb)) => {
            substs_compatible(infer, pa, pb) && types_compatible(infer, ra, rb)
        }
        (Kind::Closure(ia, sa, ca), Kind::Closure(ib, sb, cb)) => {
            ia == ib && ca == cb && substs_compatible(infer, sa, sb)
        }
        (Kind::Dyn(ta, sa, ma, aa), Kind::Dyn(tb, sb, mb, ab)) => {
            ta == tb && ma == mb && aa == ab && substs_compatible(infer, sa, sb)
        }
        (Kind::Pat(ia, ra), Kind::Pat(ib, rb)) => ra == rb && types_compatible(infer, ia, ib),
        _ => false,
    }
}

fn substs_compatible(infer: &Infer<'_, '_>, a: Substs, b: Substs) -> bool {
    let sa = infer.cx.types.substs(a);
    let sb = infer.cx.types.substs(b);
    sa.len() == sb.len()
        && sa
            .iter()
            .zip(sb)
            .all(|(&xa, &xb)| types_compatible(infer, xa, xb))
}
