use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{ImplHeader, Signature};
use crate::diag::{Diagnostic, DiagnosticKind, Span};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use rustc_lexer::TokenKind;

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

impl Infer<'_, '_> {
    /// Checks a local trait impl block for unsafe requirements, missing items, supertraits, and method signatures.
    pub(in crate::sema) fn check_local_trait_impl(
        cx: &mut Context<'_>,
        block: ImplRef,
    ) -> Result<()> {
        check_trait_impl_block(cx, block)
    }
}

pub(in super::super) fn check_trait_impl_block(cx: &mut Context<'_>, block: ImplRef) -> Result<()> {
    if !block_is_user(cx, block) {
        return Ok(());
    }
    let flight = cx.store.lookups.checked_impls.clone();
    flight
        .run_cached(
            cx,
            block,
            |cx| {
                let before = cx.quiet_mark();
                check_trait_impl_block_uncached(cx, block)?;
                Ok(cx.quiet_mark() == before)
            },
            |_| Ok(false),
        )
        .map(|_| ())
}

fn check_trait_impl_block_uncached(cx: &mut Context<'_>, block: ImplRef) -> Result<()> {
    let Some(of) = cx.impl_trait(block) else {
        return Ok(());
    };
    let Ok(header) = cx.impl_header(block) else {
        return Ok(());
    };
    if header.negative {
        return Ok(());
    }
    check_impl_unsafe(cx, block, of)?;
    check_missing_trait_items(cx, block, of)?;
    let outer = type_params(&header.generics).count();
    check_impl_supertraits(cx, block, of, &header, outer)?;
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

fn impl_item_ref(cx: &Context<'_>, block: ImplRef) -> ItemRef {
    match block {
        ImplRef::Body(item) => item,
        ImplRef::Module { krate, index } => {
            let entry = &cx.outline.crates[krate as usize].impls[index as usize];
            ItemRef {
                krate,
                file: entry.file as u32,
                item: entry.at,
            }
        }
    }
}

fn impl_block_span(cx: &Context<'_>, block: ImplRef) -> Option<Span> {
    let (src, entry) = cx.item(impl_item_ref(cx, block));
    Span::from_token(src, entry.start)
}

fn is_unsafe_trait(cx: &Context<'_>, of: ItemRef) -> bool {
    if matches!(cx.lang(of), Some("send" | "sync"))
        || (cx.outline.is_sysroot(of.krate) && matches!(cx.name(of), "Send" | "Sync"))
    {
        return true;
    }
    let (src, entry) = cx.item(of);
    entry
        .name
        .is_some_and(|name| (entry.start..name).any(|at| src.text(at) == "unsafe"))
}

fn is_unsafe_impl(cx: &Context<'_>, block: ImplRef) -> bool {
    let (src, entry) = cx.item(impl_item_ref(cx, block));
    let end = entry.body.unwrap_or(entry.end);
    let mut at = entry.start;
    while at < end && src.text(at) != "impl" {
        if src.text(at) == "unsafe" {
            return true;
        }
        at += 1;
    }
    false
}

fn check_impl_unsafe(cx: &Context<'_>, block: ImplRef, of: ItemRef) -> Result<()> {
    if !is_unsafe_trait(cx, of) || is_unsafe_impl(cx, block) {
        return Ok(());
    }
    let name = cx.name(of);
    Err(Diagnostic::new(
        DiagnosticKind::Type,
        format!("the trait `{name}` requires an `unsafe impl` declaration"),
    )
    .with_span(impl_block_span(cx, block))
    .with_primary_label(format!("implementing the trait `{name}` requires `unsafe`"))
    .into())
}

fn trait_item_has_default(cx: &Context<'_>, member: ItemRef) -> bool {
    let (src, entry) = cx.item(member);
    match entry.kind {
        ItemKind::Fn => entry.body.is_some(),
        ItemKind::Type | ItemKind::Const => {
            let Some(mut at) = entry.name.map(|n| n + 1) else {
                return true;
            };
            let mut angles = 0u32;
            while at < entry.end {
                match src.kind(at) {
                    TokenKind::Eq if angles == 0 => return true,
                    TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => {
                        at = src.pair(at);
                    }
                    TokenKind::Lt => angles += 1,
                    TokenKind::Gt
                        if !at.checked_sub(1).is_some_and(|prev| {
                            matches!(src.kind(prev), TokenKind::Minus | TokenKind::Eq)
                                && src.joint(prev)
                        }) =>
                    {
                        angles = angles.saturating_sub(1);
                    }
                    _ => {}
                }
                at += 1;
            }
            false
        }
        _ => true,
    }
}

fn check_missing_trait_items(cx: &Context<'_>, block: ImplRef, of: ItemRef) -> Result<()> {
    let tree = &cx.outline.crates[of.krate as usize];
    for member in cx.children(of) {
        let (_, entry) = cx.item(member);
        if !matches!(entry.kind, ItemKind::Fn | ItemKind::Type | ItemKind::Const)
            || !tree.env.enabled(&entry.attrs)
            || trait_item_has_default(cx, member)
        {
            continue;
        }
        let name = cx.plain_name(member);
        if cx.member(block, name).is_none() {
            return Err(Diagnostic::new(
                DiagnosticKind::Type,
                format!("not all trait items implemented, missing: `{name}`"),
            )
            .with_span(impl_block_span(cx, block))
            .with_primary_label(format!("missing `{name}` in implementation"))
            .into());
        }
    }
    Ok(())
}

fn check_impl_supertraits(
    cx: &mut Context<'_>,
    block: ImplRef,
    of: ItemRef,
    header: &ImplHeader,
    outer: usize,
) -> Result<()> {
    let Ok(def) = cx.trait_generics(of) else {
        return Ok(());
    };
    if def.1.supers.len == 0 {
        return Ok(());
    }
    let unbounded_generic = outer > 0
        && header.predicates.is_empty()
        && header.generics.iter().all(|g| g.bounds.len == 0);
    if outer != 0 && !unbounded_generic {
        return Ok(());
    }
    let anchor = impl_item_ref(cx, block);
    let diag = detached(cx, anchor, |infer| {
        let before = infer.cx.quiet_mark();
        let impl_args: Vec<Ty> = (0..outer)
            .map(|i| infer.cx.types.intern(Kind::Param(i as u16)))
            .collect();
        let Ok(frame) = infer.impl_frame_of(block, &impl_args, None) else {
            return Ok(None);
        };
        infer.with_frame(frame, |infer| {
            let Ok(self_ty) = infer.lower_ty(&header.body, header.self_ty) else {
                return Ok(None);
            };
            let Ok(trait_args) = infer.impl_trait_args(block, &impl_args, self_ty) else {
                return Ok(None);
            };
            let mut substs = Vec::with_capacity(1 + trait_args.len());
            substs.push(self_ty);
            substs.extend(trait_args);
            let Ok(supers) = infer.super_bounds(of, &substs) else {
                return Ok(None);
            };
            for (super_trait, super_substs) in supers {
                if super_trait == of {
                    continue;
                }
                if outer > 0
                    && matches!(
                        infer.cx.lang(super_trait),
                        Some("sized" | "meta_sized" | "pointee_sized")
                    )
                {
                    continue;
                }
                let target = infer.resolve(super_substs[0]);
                let args = &super_substs[1..];
                if infer.has_vars(target) || args.iter().any(|&a| infer.has_vars(a)) {
                    continue;
                }
                let held = if infer.compiler_decides(super_trait, target) {
                    infer.compiler_holds(super_trait, target)
                } else if infer.compiler_impl(super_trait, target)
                    || infer.object_holds(super_trait, target, args) == Some(true)
                {
                    true
                } else {
                    infer.select(super_trait, target, args)?.is_some()
                };
                if !held && infer.cx.quiet_mark() == before {
                    let ty_name = infer.show(target);
                    let trait_name = infer.cx.name(super_trait);
                    return Ok(Some(
                        Diagnostic::new(
                            DiagnosticKind::Trait,
                            format!("the trait bound `{ty_name}: {trait_name}` is not satisfied"),
                        )
                        .with_span(impl_block_span(infer.cx, block))
                        .with_primary_label(format!(
                            "the trait `{trait_name}` is not implemented for `{ty_name}`"
                        )),
                    ));
                }
            }
            Ok(None)
        })
    });
    match diag {
        Ok(Some(err)) => Err(err.into()),
        _ => Ok(()),
    }
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
    if impl_sig.is_unsafe != trait_sig.is_unsafe {
        let label = if trait_sig.is_unsafe {
            "expected unsafe fn, found safe fn"
        } else {
            "expected safe fn, found unsafe fn"
        };
        return Err(compat_diag(
            format!("method `{name}` has an incompatible type for trait"),
            item_name_span(cx, impl_item),
            label,
            item_name_span(cx, trait_item),
        )
        .into());
    }
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
    if impl_params.len() != trait_params.len() || impl_sig.variadic != trait_sig.variadic {
        return Some(compat_diag(
            format!("method `{name}` has an incompatible type for trait"),
            item_name_span(infer.cx, impl_item),
            format!(
                "expected {} parameter(s), found {}",
                trait_params.len(),
                impl_params.len()
            ),
            item_name_span(infer.cx, trait_item),
        ));
    }
    for (idx, (&trait_p, &impl_p)) in trait_params.iter().zip(impl_params).enumerate() {
        if !types_compatible(infer, trait_p, impl_p) {
            return Some(compat_diag(
                format!("method `{name}` has an incompatible type for trait"),
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
            format!("method `{name}` has an incompatible type for trait"),
            item_name_span(infer.cx, impl_item),
            format!(
                "expected `{}`, found `{}`",
                infer.show(trait_ret),
                infer.show(impl_ret)
            ),
            item_name_span(infer.cx, trait_item),
        ));
    }
    if trait_own != impl_own || impl_sig.receiver.is_some() != trait_sig.receiver.is_some() {
        return Some(compat_diag(
            format!("method `{name}` has an incompatible type for trait"),
            item_name_span(infer.cx, impl_item),
            "method signature does not match trait",
            item_name_span(infer.cx, trait_item),
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
    if !infer.cx.types.has_projs(ty) {
        return ty;
    }
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
