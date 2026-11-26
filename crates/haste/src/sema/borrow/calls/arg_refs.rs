use std::sync::Arc;

use crate::body::adt::{AdtDef, Shape};
use crate::body::sig::{Generic, ImplHeader, Predicate, SelfKind, Signature};
use crate::body::{Args, Body, Bound, ExprId, List, Stmt, Ty as AstTy, TyId};
use crate::names::{ItemRef, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::impls::ImplRef;
use crate::sema::infer::{Coercion, Res, type_params};
use crate::sema::substs::Substs;
use crate::sema::traits::TraitGenerics;
use crate::sema::ty::{Kind, Mut, Ty};

use super::lifetimes::{callable_param_inputs, pred_matches_name_in};

#[derive(Clone)]
enum HeaderSource {
    Impl(Arc<ImplHeader>, ImplRef),
    Trait(Arc<TraitGenerics>, ItemRef),
    Sig(Arc<Signature>),
}

impl HeaderSource {
    fn arena(&self) -> &Body {
        match self {
            HeaderSource::Impl(hdr, _) => &hdr.body,
            HeaderSource::Trait(tg, _) => &tg.0,
            HeaderSource::Sig(sig) => &sig.body,
        }
    }

    fn parts<'a>(
        &'a self,
        cx: &'a crate::sema::Context<'_>,
        item_src: &'a Source,
    ) -> (&'a Body, &'a Source) {
        let src = match self {
            HeaderSource::Impl(_, blk) => &cx.impl_file(*blk).source,
            HeaderSource::Trait(_, of) => &cx.file(*of).source,
            HeaderSource::Sig(_) => item_src,
        };
        (self.arena(), src)
    }
}

fn param_placeholders(cx: &crate::sema::Context<'_>, count: usize) -> Vec<Ty> {
    (0..count)
        .map(|i| cx.types.intern(Kind::Param(i as u16)))
        .collect()
}

pub(in super::super) fn ast_ref_mut(body: &Body, mut ty: TyId) -> Option<Mut> {
    while let AstTy::Paren(inner) = *body.ty(ty) {
        ty = inner;
    }
    match *body.ty(ty) {
        AstTy::Ref { mutable, .. } => Some(if mutable { Mut::Mut } else { Mut::Not }),
        _ => None,
    }
}

pub(in super::super) fn recv_ref_mut(sig: &Signature) -> Option<Mut> {
    match sig.receiver? {
        (SelfKind::Ref, _) => Some(Mut::Not),
        (SelfKind::RefMut, _) => Some(Mut::Mut),
        (SelfKind::Typed(ty), _) => ast_ref_mut(&sig.body, ty),
        (SelfKind::Value, _) => None,
    }
}

pub(in super::super) fn recv_is_by_ref(sig: &Signature) -> bool {
    recv_ref_mut(sig).is_some()
}

pub(in super::super) fn sig_param_idx(idx: usize, has_self_arg: bool) -> Option<usize> {
    if has_self_arg {
        idx.checked_sub(1)
    } else {
        Some(idx)
    }
}

pub(super) fn param_ref_mut(sig: &Signature, idx: usize, has_self_arg: bool) -> Option<Mut> {
    if has_self_arg && idx == 0 {
        return recv_ref_mut(sig);
    }
    sig.params
        .get(sig_param_idx(idx, has_self_arg)?)
        .and_then(|&(_, ty)| ast_ref_mut(&sig.body, ty))
}

impl super::super::Checker<'_, '_> {
    pub(in super::super) fn is_callable_trait_method(&self, item: ItemRef) -> bool {
        self.cx
            .owner_trait(item)
            .and_then(|t| self.cx.lang(t))
            .is_some_and(crate::sema::infer::is_callable_lang)
    }

    pub(in super::super) fn local_param_sig_ty(
        &mut self,
        callee: ExprId,
    ) -> Option<(Arc<Signature>, TyId)> {
        let local = self.expr_plain_local(callee)?;
        let pos = self.typed.params.iter().position(|&p| p == local)?;
        let sig = self.cx.signature(self.item).ok()?;
        let param_idx = sig_param_idx(pos, sig.receiver.is_some())?;
        let &(_, param_ty) = sig.params.get(param_idx)?;
        Some((sig, param_ty))
    }

    pub(in super::super) fn ptr_or_ref_mut(&self, mut ty: Ty) -> Option<Mut> {
        if ty == Ty::NONE {
            return None;
        }
        while let Kind::Pat(inner, ..) = *self.cx.types.kind(ty) {
            ty = inner;
        }
        match *self.cx.types.kind(ty) {
            Kind::Ref(m, _) | Kind::Ptr(m, _) => Some(m),
            _ => None,
        }
    }

    fn peel_call_receiver(&self, mut ty: Ty) -> Ty {
        while let Some(inner) = match *self.cx.types.kind(ty) {
            Kind::Ref(_, inner) => Some(inner),
            _ => self.cx.boxed(ty),
        } {
            ty = inner;
        }
        ty
    }

    fn find_local_type_alias(&self, name: &str) -> Option<ItemRef> {
        let root = self.item.body_owner();
        let exp = self.cx.expansion(self.item)?;
        let items = &exp.file.items.items;
        let src = &exp.file.source;
        for stmt in &self.body.stmts {
            if let Stmt::Item(start, end) = *stmt {
                for at in crate::sema::expand::stmt_items(items, (start, end)) {
                    let it = &items[at];
                    if it.kind == ItemKind::Type
                        && it.name.is_some_and(|n| plain(src.text(n)) == name)
                    {
                        return Some(root.inner(at as u32));
                    }
                }
            }
        }
        None
    }

    fn ast_ptr_or_ref_mut_local(&self, arena: &Body, src: &Source, ty: TyId) -> Option<Mut> {
        self.ast_ptr_or_ref_mut_local_depth(arena, src, ty, 16)
    }

    fn ast_ptr_or_ref_mut_local_depth(
        &self,
        arena: &Body,
        src: &Source,
        mut ty: TyId,
        depth: u8,
    ) -> Option<Mut> {
        if depth == 0 {
            return None;
        }
        while let AstTy::Paren(inner) = *arena.ty(ty) {
            ty = inner;
        }
        match *arena.ty(ty) {
            AstTy::Ref { mutable, .. } | AstTy::Ptr { mutable, .. } => {
                Some(if mutable { Mut::Mut } else { Mut::Not })
            }
            AstTy::Path(path_id) => {
                let path = arena.path(path_id);
                let segs = arena.list(path.segments);
                if path.qself.is_some() || segs.len() != 1 {
                    return None;
                }
                let name = plain(src.text(arena.segments[segs[0] as usize].name));
                let alias_item = self.find_local_type_alias(name)?;
                let alias = self.cx.alias_def(alias_item).ok()?;
                self.ast_ptr_or_ref_mut_local_depth(
                    &alias.arena,
                    self.cx.body_src(alias_item),
                    alias.ty,
                    depth - 1,
                )
            }
            _ => None,
        }
    }

    fn lower_ast_ty_hdr(&mut self, hdr: &HeaderSource, ty: TyId) -> Option<Ty> {
        let owner = self.item;
        let impl_count = self
            .cx
            .owner_impl(owner)
            .and_then(|b| self.cx.impl_header(b).ok())
            .map_or(0, |h| type_params(&h.generics).count());
        let trait_count = self
            .cx
            .owner_trait(owner)
            .and_then(|t| self.cx.trait_generics(t).ok())
            .map_or(0, |tg| 1 + type_params(&tg.1.params).count());
        let own_count = self
            .cx
            .signature(owner)
            .ok()
            .map_or(0, |s| type_params(&s.generics).count());
        let total = impl_count + trait_count + own_count;
        crate::sema::infer::detached(self.cx, owner, |infer| {
            let params = param_placeholders(infer.cx, total);
            let frame = infer.frame(owner, &params)?;
            infer.with_frame(frame, |i| i.lower_ty(hdr.arena(), ty))
        })
        .ok()
    }

    fn lower_ast_ptr_or_ref_mut_hdr(&mut self, hdr: &HeaderSource, ty: TyId) -> Option<Mut> {
        {
            let (arena, src) = hdr.parts(self.cx, self.src);
            if let Some(m) = self.ast_ptr_or_ref_mut_local(arena, src, ty) {
                return Some(m);
            }
        }
        self.lower_ast_ty_hdr(hdr, ty)
            .and_then(|t| self.ptr_or_ref_mut(t))
    }

    fn uninstantiated_fn_param_ty(
        &mut self,
        item: ItemRef,
        substs: Substs,
        fn_idx: usize,
    ) -> Option<Ty> {
        let count = self.cx.types.substs(substs).len();
        crate::sema::infer::detached(self.cx, item, |infer| {
            let placeholders = param_placeholders(infer.cx, count);
            infer.fn_types(item, &placeholders)
        })
        .ok()
        .and_then(|(params, _)| params.get(fn_idx).copied())
    }

    pub(super) fn uninstantiated_fn_param_ref_mut(
        &mut self,
        item: ItemRef,
        substs: Substs,
        fn_idx: usize,
        has_self_in_sig: bool,
    ) -> Option<Mut> {
        if let Some(t) = self.uninstantiated_fn_param_ty(item, substs, fn_idx) {
            return self.ptr_or_ref_mut(t);
        }
        let sig = self.cx.signature(item).ok()?;
        if has_self_in_sig && fn_idx == 0 {
            return match sig.receiver? {
                (SelfKind::Ref, _) => Some(Mut::Not),
                (SelfKind::RefMut, _) => Some(Mut::Mut),
                (SelfKind::Typed(ty), _) => {
                    self.ast_ptr_or_ref_mut_local(&sig.body, self.cx.item(item).0, ty)
                }
                (SelfKind::Value, _) => None,
            };
        }
        let param_idx = sig_param_idx(fn_idx, sig.receiver.is_some())?;
        let &(_, param_ty) = sig.params.get(param_idx)?;
        self.ast_ptr_or_ref_mut_local(&sig.body, self.cx.item(item).0, param_ty)
    }

    fn callable_trait_method_arg_ty(&self, call_expr: ExprId, idx: usize) -> Option<Ty> {
        let Some(Res::Fn(method, substs)) = self.typed.res.get(&call_expr).copied() else {
            return None;
        };
        if !self.is_callable_trait_method(method) {
            return None;
        }
        let tuple = *self.cx.types.substs(substs).get(1)?;
        match *self.cx.types.kind(tuple) {
            Kind::Tuple(elems) => self.cx.types.substs(elems).get(idx).copied(),
            _ => None,
        }
    }

    fn direct_callee_arg_ty(&mut self, ty: Ty, idx: usize) -> Option<Ty> {
        let peeled = self.peel_call_receiver(ty);
        match *self.cx.types.kind(peeled) {
            Kind::FnPtr(params, _) => self.cx.types.substs(params).get(idx).copied(),
            Kind::Dyn(of, args, ..)
                if self
                    .cx
                    .lang(of)
                    .is_some_and(crate::sema::infer::is_callable_lang) =>
            {
                let tup = *self.cx.types.substs(args).first()?;
                match *self.cx.types.kind(tup) {
                    Kind::Tuple(elems) => self.cx.types.substs(elems).get(idx).copied(),
                    _ => None,
                }
            }
            Kind::Closure(parent, _, at) => {
                if parent == self.item {
                    self.typed
                        .closures
                        .get(&ExprId(at))
                        .and_then(|c| c.params.get(idx))
                        .map(|&l| self.local_ty(l))
                } else {
                    self.cx.closure(peeled).and_then(|info| {
                        info.closure
                            .params
                            .get(idx)
                            .and_then(|&l| info.typed.locals.get(l as usize))
                            .map(|l| l.ty)
                    })
                }
            }
            _ => None,
        }
    }

    fn param_slot_candidates(
        &mut self,
        slot: u16,
        idx: usize,
    ) -> Option<Vec<(HeaderSource, TyId)>> {
        let mut rem = slot as usize;
        let mut candidates: Vec<(HeaderSource, TyId)> = Vec::new();
        let mut matched_name: Option<String> = None;
        let outer = if let Some(block) = self.cx.owner_impl(self.item)
            && let Ok(hdr) = self.cx.impl_header(block)
        {
            Some((HeaderSource::Impl(hdr, block), false))
        } else if let Some(of) = self.cx.owner_trait(self.item)
            && let Ok(tg) = self.cx.trait_generics(of)
        {
            Some((HeaderSource::Trait(tg, of), true))
        } else {
            None
        };
        if let Some((hdr_src, allow_self)) = outer {
            let (generics, predicates) = match &hdr_src {
                HeaderSource::Impl(hdr, _) => (&hdr.generics[..], &hdr.predicates[..]),
                HeaderSource::Trait(tg, _) => (&tg.1.params[..], &tg.1.predicates[..]),
                HeaderSource::Sig(s) => (&s.generics[..], &s.predicates[..]),
            };
            let src = hdr_src.parts(self.cx, self.src).1;
            match collect_header_slot(
                hdr_src.arena(),
                src,
                generics,
                predicates,
                rem,
                idx,
                allow_self,
            ) {
                Ok((name, ins)) => {
                    matched_name = name;
                    push_header_candidates(&hdr_src, ins, &mut candidates);
                }
                Err(next) => rem = next,
            }
        }
        let sig = self.cx.signature(self.item).ok()?;
        let sig_hdr = HeaderSource::Sig(sig.clone());
        if matched_name.is_none()
            && candidates.is_empty()
            && let Ok((_, ins)) = collect_header_slot(
                &sig.body,
                self.src,
                &sig.generics,
                &sig.predicates,
                rem,
                idx,
                false,
            )
        {
            push_header_candidates(&sig_hdr, ins, &mut candidates);
        } else if let Some(ref name) = matched_name {
            let mut ins = Vec::new();
            for pred in &sig.predicates {
                if pred_matches_name_in(&sig.body, self.src, pred, name) {
                    collect_bound_list_inputs(&sig.body, self.src, pred.bounds, idx, &mut ins);
                }
            }
            push_header_candidates(&sig_hdr, ins, &mut candidates);
        }
        Some(candidates)
    }

    fn param_slot_callable_ty(&mut self, slot: u16, idx: usize) -> Option<Ty> {
        let candidates = self.param_slot_candidates(slot, idx)?;
        for (hdr_src, ty_id) in candidates {
            if let Some(t) = self.lower_ast_ty_hdr(&hdr_src, ty_id) {
                return Some(t);
            }
        }
        None
    }

    fn param_slot_callable_ref_mut(&mut self, slot: u16, idx: usize) -> Option<Mut> {
        let candidates = self.param_slot_candidates(slot, idx)?;
        for (hdr_src, ty_id) in candidates {
            if let Some(m) = self.lower_ast_ptr_or_ref_mut_hdr(&hdr_src, ty_id) {
                return Some(m);
            }
        }
        None
    }

    pub(super) fn callee_param_ref_mut(
        &mut self,
        call_expr: Option<ExprId>,
        callee_ty: Option<Ty>,
        idx: usize,
    ) -> Option<Mut> {
        let ty = callee_ty.filter(|&t| t != Ty::NONE)?;
        let peeled = self.peel_call_receiver(ty);
        if let Some(t) = self.direct_callee_arg_ty(ty, idx) {
            return self.ptr_or_ref_mut(t);
        }
        match *self.cx.types.kind(peeled) {
            Kind::FnDef(fn_item, substs) => {
                self.uninstantiated_fn_param_ref_mut(fn_item, substs, idx, false)
            }
            Kind::Param(p) => self.param_slot_callable_ref_mut(p, idx),
            _ => call_expr
                .and_then(|e| self.callable_trait_method_arg_ty(e, idx))
                .and_then(|t| self.ptr_or_ref_mut(t)),
        }
    }

    fn callee_bound_param_hdr(
        &mut self,
        callee: ExprId,
        idx: usize,
    ) -> Option<(HeaderSource, TyId)> {
        let (sig, param_ty) = self.local_param_sig_ty(callee)?;
        let inputs = callable_param_inputs(&sig, self.src, param_ty);
        let &in_ty = inputs.get(idx)?;
        Some((HeaderSource::Sig(sig), in_ty))
    }

    pub(super) fn callee_bound_param_ref_mut(&mut self, callee: ExprId, idx: usize) -> Option<Mut> {
        let (hdr, in_ty) = self.callee_bound_param_hdr(callee, idx)?;
        self.lower_ast_ptr_or_ref_mut_hdr(&hdr, in_ty)
    }

    pub(in super::super) fn struct_adt_variant(
        &mut self,
        expr: ExprId,
    ) -> Option<(ItemRef, usize, Option<Substs>, Arc<AdtDef>)> {
        let (item, variant, substs) = self.struct_adt(expr)?;
        let def = self.cx.adt_def(item).ok()?;
        (variant < def.variants.len()).then_some((item, variant, Some(substs), def))
    }

    fn detached_adt_variant_fields(
        &mut self,
        item: ItemRef,
        variant: usize,
        args: Vec<Ty>,
    ) -> Option<Vec<Ty>> {
        crate::sema::infer::detached(self.cx, item, |infer| infer.adt_fields(item, &args))
            .ok()
            .and_then(|all| all.get(variant).cloned())
    }

    fn uninstantiated_adt_fields(
        &mut self,
        item: ItemRef,
        variant: usize,
        def: &AdtDef,
    ) -> Option<Vec<Ty>> {
        let args = param_placeholders(self.cx, type_params(&def.generics.params).count());
        self.detached_adt_variant_fields(item, variant, args)
    }

    fn adt_field_ast_ptr_or_ref_mut(
        &self,
        item: ItemRef,
        def: &AdtDef,
        variant: usize,
        idx: usize,
    ) -> Option<Mut> {
        let f_ty = def.variants.get(variant)?.fields.get(idx)?.ty;
        self.ast_ptr_or_ref_mut_local(&def.body, &self.cx.file(item).source, f_ty)
    }

    pub(super) fn ctor_field_ref_mut(
        &mut self,
        item: ItemRef,
        variant: u32,
        idx: usize,
    ) -> Option<Mut> {
        let def = self.cx.adt_def(item).ok()?;
        let v = variant as usize;
        if let Some(fields) = self.uninstantiated_adt_fields(item, v, &def) {
            return fields
                .get(idx)
                .copied()
                .and_then(|t| self.ptr_or_ref_mut(t));
        }
        self.adt_field_ast_ptr_or_ref_mut(item, &def, v, idx)
    }

    fn struct_field_index(
        &self,
        def: &AdtDef,
        variant: usize,
        item: ItemRef,
        fid: u32,
    ) -> Option<usize> {
        let f = self.body.fields[fid as usize];
        if let Some(&idx) = self.typed.inits.get(&f.value) {
            return Some(idx as usize);
        }
        let var = &def.variants[variant];
        let fname = plain(self.src.text(f.name));
        if var.shape == Shape::Tuple
            && let Ok(i) = fname.parse::<usize>()
            && i < var.fields.len()
        {
            return Some(i);
        }
        let adt_src = &self.cx.file(item).source;
        var.fields
            .iter()
            .position(|vf| vf.name.is_some_and(|n| plain(adt_src.text(n)) == fname))
    }

    pub(in super::super) fn struct_field_ref_muts(
        &mut self,
        expr: ExprId,
        field_ids: &[u32],
    ) -> (Vec<Option<Mut>>, Vec<Ty>) {
        let Some((item, variant, substs, def)) = self.struct_adt_variant(expr) else {
            return (Vec::new(), Vec::new());
        };
        let uninst = self.uninstantiated_adt_fields(item, variant, &def);
        let need_inst = field_ids.iter().any(|&fid| {
            let val = self.body.fields[fid as usize].value;
            self.typed.coercions.contains_key(&val)
                || matches!(
                    *self.body.expr(self.unwrap_parens(val)),
                    crate::body::Expr::Tuple(_)
                )
        });
        let inst = if need_inst {
            substs.and_then(|s| {
                let svec = self.cx.types.substs(s).to_vec();
                self.detached_adt_variant_fields(item, variant, svec)
            })
        } else {
            None
        };
        let mut refs = Vec::with_capacity(field_ids.len());
        let mut tys = Vec::with_capacity(field_ids.len());
        for &fid in field_ids {
            let Some(fidx) = self.struct_field_index(&def, variant, item, fid) else {
                refs.push(None);
                tys.push(Ty::NONE);
                continue;
            };
            let r = if let Some(ref fs) = uninst {
                fs.get(fidx).copied().and_then(|t| self.ptr_or_ref_mut(t))
            } else {
                self.adt_field_ast_ptr_or_ref_mut(item, &def, variant, fidx)
            };
            refs.push(r);
            tys.push(
                inst.as_ref()
                    .and_then(|fs| fs.get(fidx).copied())
                    .unwrap_or(Ty::NONE),
            );
        }
        (refs, tys)
    }

    pub(super) fn instantiated_arg_ty(
        &mut self,
        call_expr: Option<ExprId>,
        callee_ty: Option<Ty>,
        res: Option<Res>,
        fn_arg_offset: usize,
        idx: usize,
    ) -> Option<Ty> {
        if let Some(e) = call_expr
            && let Some(t) = self.callable_trait_method_arg_ty(e, idx)
        {
            return Some(t);
        }
        if let Some(ty) = callee_ty.filter(|&t| t != Ty::NONE)
            && let Some(t) = self.direct_callee_arg_ty(ty, idx)
        {
            return Some(t);
        }
        match res? {
            Res::Fn(item, substs) if !self.is_callable_trait_method(item) => {
                let svec = self.cx.types.substs(substs).to_vec();
                crate::sema::infer::detached(self.cx, item, |infer| infer.fn_types(item, &svec))
                    .ok()
                    .and_then(|(params, _)| params.get(fn_arg_offset + idx).copied())
            }
            Res::Ctor(item, v, substs) => {
                let svec = self.cx.types.substs(substs).to_vec();
                self.detached_adt_variant_fields(item, v as usize, svec)
                    .and_then(|fields| fields.get(idx).copied())
            }
            _ => None,
        }
    }

    fn fn_arg_offset(ctx: &super::ArgListCtx<'_>) -> usize {
        usize::from(ctx.recv_sources.is_some() && ctx.sig.is_some_and(|s| s.receiver.is_some()))
    }

    fn callee_param_ty(
        &mut self,
        call_expr: Option<ExprId>,
        callee_ty: Option<Ty>,
        idx: usize,
    ) -> Option<Ty> {
        let ty = callee_ty.filter(|&t| t != Ty::NONE)?;
        if let Some(t) = self.direct_callee_arg_ty(ty, idx) {
            return Some(t);
        }
        let peeled = self.peel_call_receiver(ty);
        match *self.cx.types.kind(peeled) {
            Kind::FnDef(fn_item, substs) => self.uninstantiated_fn_param_ty(fn_item, substs, idx),
            Kind::Param(p) => self.param_slot_callable_ty(p, idx),
            _ => call_expr.and_then(|e| self.callable_trait_method_arg_ty(e, idx)),
        }
    }

    pub(super) fn resolve_call_arg_ty(
        &mut self,
        ctx: &super::ArgListCtx<'_>,
        idx: usize,
        _arg: ExprId,
    ) -> Option<Ty> {
        if ctx.expected_from_arg_tys
            && let Some(ts) = ctx.arg_tys
        {
            return ts.get(idx).copied().filter(|&t| t != Ty::NONE);
        }
        let fn_offset = Self::fn_arg_offset(ctx);
        if let Some(Res::Fn(item, substs)) = ctx.call_res
            && !self.is_callable_trait_method(item)
        {
            return self.uninstantiated_fn_param_ty(item, substs, fn_offset + idx);
        }
        if let Some(c) = ctx.callee_expr
            && let Some((hdr, in_ty)) = self.callee_bound_param_hdr(c, idx)
            && let Some(t) = self.lower_ast_ty_hdr(&hdr, in_ty)
        {
            return Some(t);
        }
        if let Some(t) = self.callee_param_ty(ctx.call_expr, ctx.callee_ty, idx) {
            return Some(t);
        }
        if let Some((item, v)) = ctx.ctor_res
            && let Ok(def) = self.cx.adt_def(item)
        {
            return self
                .uninstantiated_adt_fields(item, v as usize, &def)
                .and_then(|fs| fs.get(idx).copied());
        }
        None
    }

    pub(super) fn resolve_call_arg_ref(
        &mut self,
        ctx: &super::ArgListCtx<'_>,
        idx: usize,
        arg: ExprId,
    ) -> Option<Mut> {
        let fn_offset = Self::fn_arg_offset(ctx);
        let param_ref = ctx
            .param_refs
            .and_then(|refs| refs.get(idx).copied().flatten())
            .or_else(|| {
                ctx.call_res.and_then(|r| match r {
                    Res::Fn(item, substs) if !self.is_callable_trait_method(item) => self
                        .uninstantiated_fn_param_ref_mut(
                            item,
                            substs,
                            fn_offset + idx,
                            ctx.has_self_arg || ctx.recv_sources.is_some(),
                        ),
                    _ => None,
                })
            })
            .or_else(|| {
                ctx.sig
                    .and_then(|s| param_ref_mut(s, idx, ctx.has_self_arg))
            })
            .or_else(|| {
                ctx.callee_expr
                    .and_then(|c| self.callee_bound_param_ref_mut(c, idx))
            })
            .or_else(|| self.callee_param_ref_mut(ctx.call_expr, ctx.callee_ty, idx))
            .or_else(|| {
                ctx.ctor_res
                    .and_then(|(item, v)| self.ctor_field_ref_mut(item, v, idx))
            });
        if param_ref.is_some() {
            return param_ref;
        }
        let field_target = match self.typed.coercions.get(&arg)? {
            Coercion::Field { target, .. } => Some(*target),
            _ => None,
        };
        if let Some(target) = field_target
            && let Some(m) = self.ptr_or_ref_mut(target)
        {
            return Some(m);
        }
        let expected_ty = ctx.arg_tys.and_then(|ts| ts.get(idx).copied()).or_else(|| {
            self.instantiated_arg_ty(ctx.call_expr, ctx.callee_ty, ctx.call_res, fn_offset, idx)
        });
        Some(
            expected_ty
                .and_then(|t| self.ptr_or_ref_mut(t))
                .unwrap_or(Mut::Mut),
        )
    }
}

fn push_header_candidates(
    hdr_src: &HeaderSource,
    ins: Vec<TyId>,
    out: &mut Vec<(HeaderSource, TyId)>,
) {
    out.extend(ins.into_iter().map(|t| (hdr_src.clone(), t)));
}

fn collect_header_slot(
    arena: &Body,
    src: &Source,
    generics: &[Generic],
    predicates: &[Predicate],
    slot: usize,
    idx: usize,
    allow_self: bool,
) -> Result<(Option<String>, Vec<TyId>), usize> {
    let mut rem = slot;
    let mut ins = Vec::new();
    let name = if allow_self && rem == 0 {
        Some("Self".to_owned())
    } else {
        if allow_self {
            rem -= 1;
        }
        let cnt = type_params(generics).count();
        let Some(g) = type_params(generics).nth(rem) else {
            return Err(rem - cnt);
        };
        let gname = src.text(g.name);
        collect_bound_list_inputs(arena, src, g.bounds, idx, &mut ins);
        (gname != "impl").then(|| gname.to_owned())
    };
    if let Some(ref gname) = name {
        for pred in predicates {
            if pred_matches_name_in(arena, src, pred, gname) {
                collect_bound_list_inputs(arena, src, pred.bounds, idx, &mut ins);
            }
        }
    }
    Ok((name, ins))
}

fn collect_bound_list_inputs(
    arena: &Body,
    src: &Source,
    list: List,
    idx: usize,
    out: &mut Vec<TyId>,
) {
    for &b_id in arena.list(list) {
        if let Bound::Trait { path, .. } = arena.bounds[b_id as usize]
            && let Some(&seg_id) = arena.list(arena.path(path).segments).last()
            && let seg = arena.segments[seg_id as usize]
            && matches!(
                src.text(seg.name),
                "Fn" | "FnMut" | "FnOnce" | "AsyncFn" | "AsyncFnMut" | "AsyncFnOnce"
            )
            && let Some(Args::Paren { inputs, .. }) = seg.args
            && let Some(&in_ty) = arena.list(inputs).get(idx)
        {
            out.push(TyId(in_ty));
        }
    }
}
