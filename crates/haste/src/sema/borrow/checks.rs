mod closure;

use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::{GenericKind, Signature};
use crate::body::{BlockKind, Expr, ExprId, List, Stmt, Ty as AstTy, UnOp};
use crate::outline::item::ItemKind;
use crate::sema::impls::ImplRef;
use crate::sema::infer::{By, Coercion, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::calls::{Lifetimes, collect_ty_lifetimes, ty_shares_name};
use super::places::{FieldKey, Place, PlaceStep, push_step};
use super::propagate::BorrowSource;
use super::{Checker, PartialMove, State, loan_conflicts_borrow};

pub(super) const TEMP_TARGET_BASE: u32 = u32::MAX - 262_144;

pub(super) use closure::ClosureEffects;

pub(super) fn is_temp_target(target: u32) -> Option<u32> {
    (target >= TEMP_TARGET_BASE && target < super::ITER_HOLDER_BASE).then(|| target - TEMP_TARGET_BASE)
}

pub(super) fn temp_target_for(depth: u32) -> u32 {
    TEMP_TARGET_BASE + depth.min(65_000)
}

impl Checker<'_, '_> {
    pub(super) fn ty_holds_borrow(&mut self, ty: Ty) -> bool {
        let mut seen = HashSet::default();
        self.ty_holds_borrow_inner(ty, &mut seen)
    }

    fn ty_holds_borrow_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(..) | Kind::Dyn(..) => true,
            Kind::Slice(elem) | Kind::Array(elem, _) | Kind::Pat(elem, ..) => {
                self.ty_holds_borrow_inner(elem, seen)
            }
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                items.into_iter().any(|item| self.ty_holds_borrow_inner(item, seen))
            }
            Kind::Closure(..) => self.closure_ty_holds_borrow(ty, seen),
            Kind::Adt(item, args) => {
                let args = self.cx.types.substs(args).to_vec();
                if args.into_iter().any(|arg| self.ty_holds_borrow_inner(arg, seen)) {
                    return true;
                }
                self.cx.adt_def(item).is_ok_and(|def| {
                    def.generics.params.iter().any(|g| g.kind == GenericKind::Lifetime)
                })
            }
            _ => false,
        }
    }

    fn closure_ty_holds_borrow(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        let Kind::Closure(parent, _, at) = *self.cx.types.kind(ty) else {
            return false;
        };
        if let Some(info) = self.cx.closure(ty) {
            return info.closure.captures.iter().any(|c| {
                c.by != By::Value
                    || self.ty_holds_borrow_inner(info.typed.locals.get(c.local as usize).map_or(Ty::NONE, |l| l.ty), seen)
            });
        }
        if parent == self.item
            && let Some(closure) = self.typed.closures.get(&ExprId(at))
        {
            return closure.captures.iter().any(|c| {
                c.by != By::Value || self.ty_holds_borrow_inner(self.local_ty(c.local), seen)
            });
        }
        true
    }

    pub(super) fn ty_holds_closure(&mut self, ty: Ty) -> bool {
        let mut seen = HashSet::default();
        self.ty_holds_closure_inner(ty, &mut seen)
    }

    fn ty_holds_closure_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Closure(..) | Kind::Dyn(..) | Kind::FnDef(..) | Kind::FnPtr(..) => true,
            Kind::Ref(_, inner) | Kind::Slice(inner) | Kind::Array(inner, _) | Kind::Pat(inner, ..) => {
                self.ty_holds_closure_inner(inner, seen)
            }
            Kind::Tuple(items) => {
                let items = self.cx.types.substs(items).to_vec();
                items.into_iter().any(|item| self.ty_holds_closure_inner(item, seen))
            }
            Kind::Adt(item, items) => {
                if self.is_iter_mod(item) {
                    return true;
                }
                let items = self.cx.types.substs(items).to_vec();
                items.into_iter().any(|item| self.ty_holds_closure_inner(item, seen))
            }
            _ => false,
        }
    }

    pub(super) fn item_in_std_mod(&mut self, item: crate::names::ItemRef, submod: &str) -> bool {
        let mod_path = self.cx.module_text(item);
        let Some(rest) = mod_path.strip_prefix("core::").or_else(|| mod_path.strip_prefix("std::")) else {
            return false;
        };
        rest == submod || (rest.starts_with(submod) && rest.as_bytes().get(submod.len()) == Some(&b':'))
    }

    pub(super) fn is_iter_mod(&mut self, item: crate::names::ItemRef) -> bool {
        self.item_in_std_mod(item, "iter")
    }

    pub(super) fn ty_has_borrow_drop(&mut self, ty: Ty) -> bool {
        if ty == Ty::NONE {
            return false;
        }
        if let Some(&cached) = self.drop_cache.get(&ty) {
            return cached;
        }
        let mut seen = HashSet::default();
        let res = self.ty_has_borrow_drop_inner(ty, &mut seen);
        self.drop_cache.insert(ty, res);
        res
    }

    fn ty_has_borrow_drop_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Dyn(..) => true,
            Kind::Tuple(items) => self.cx.types.substs(items).to_vec().into_iter().any(|t| self.ty_has_borrow_drop_inner(t, seen)),
            Kind::Array(elem, len) => self.cx.types.len(len) != Some(0) && self.ty_has_borrow_drop_inner(elem, seen),
            Kind::Adt(item, _) => {
                if self.cx.lang(item) == Some("manually_drop") || self.cx.item(item).1.kind == ItemKind::Union {
                    return false;
                }
                let (drop_blk, parts) = super::super::infer::detached(self.cx, self.item, |infer| {
                    let drop_blk = infer.drop_impl(ty)?.map(|(blk, _)| blk);
                    let parts = infer.drop_parts(ty)?;
                    Ok((drop_blk, parts))
                })
                .unwrap_or((None, Vec::new()));
                if let Some(block) = drop_blk
                    && self.cx.boxed(ty).is_none()
                    && !self.impl_has_may_dangle(block)
                {
                    return true;
                }
                parts.into_iter().any(|part| self.ty_has_borrow_drop_inner(part, seen))
            }
            _ => false,
        }
    }

    fn impl_has_may_dangle(&mut self, block: ImplRef) -> bool {
        let Ok(hdr) = self.cx.impl_header(block) else {
            return false;
        };
        let start = match block {
            ImplRef::Body(item) => self.cx.item(item).1.start,
            ImplRef::Module { krate, index } => {
                let entry = &self.cx.outline.crates[krate as usize].impls[index as usize];
                self.cx.outline.crates[krate as usize].files[entry.file].items.items[entry.at as usize].start
            }
        };
        let end = hdr.generics.last().map_or(start, |g| g.span.1);
        let src = &self.cx.impl_file(block).source;
        (start..end).any(|t| src.text(t) == "may_dangle")
    }

    pub(super) fn is_non_promotable_temp(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner)
            | Expr::Cast(inner, _)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Unary(UnOp::Deref, inner)
            | Expr::Ref { expr: inner, .. } => self.is_non_promotable_temp(inner),
            Expr::Call(callee, args) => {
                let is_ctor = matches!(self.typed.res.get(&callee), Some(Res::Ctor(..)))
                    || matches!(self.typed.res.get(&expr), Some(Res::Ctor(..)));
                if !is_ctor {
                    let is_const = matches!(self.typed.res.get(&callee).or_else(|| self.typed.res.get(&expr)), Some(Res::Fn(item, _)) if self.is_const_fn(*item));
                    if !is_const {
                        return true;
                    }
                }
                self.body.list(args).iter().any(|&a| self.is_non_promotable_temp(ExprId(a)))
            }
            Expr::Method { .. } | Expr::Await(_) | Expr::Try(_) => true,
            Expr::Block { kind, stmts, tail, .. } => {
                !matches!(kind, BlockKind::Const)
                    && (self.body.list(stmts).iter().any(|&s| match self.body.stmts[s as usize] {
                        Stmt::Let { init, .. } => init.is_some_and(|e| self.is_non_promotable_temp(e)),
                        Stmt::Expr(e, _) => self.is_non_promotable_temp(e),
                        _ => false,
                    }) || tail.is_some_and(|t| self.is_non_promotable_temp(t)))
            }
            Expr::Tuple(list) | Expr::Array(list) => self.body.list(list).iter().any(|&a| self.is_non_promotable_temp(ExprId(a))),
            Expr::Struct { fields, base, .. } => {
                self.body.list(fields).iter().any(|&f| self.is_non_promotable_temp(self.body.fields[f as usize].value))
                    || base.is_some_and(|b| self.is_non_promotable_temp(b))
            }
            Expr::Repeat(a, _) | Expr::Index(a, _) => self.is_non_promotable_temp(a),
            Expr::If(_, t, e) => self.is_non_promotable_temp(t) || e.is_some_and(|alt| self.is_non_promotable_temp(alt)),
            Expr::Match(_, arms) => self.body.list(arms).iter().any(|&a| self.is_non_promotable_temp(self.body.arms[a as usize].body)),
            _ => false,
        }
    }

    fn is_const_fn(&self, item: crate::names::ItemRef) -> bool {
        let (src, entry) = self.cx.item(item);
        (entry.start..entry.name.unwrap_or(entry.start)).any(|t| src.text(t) == "const")
    }

    pub(super) fn check_stmt_temp_escape(&self, state: &State, min_depth: u32, check_locals: &[u32], outer_live: &HashSet<u32>) -> Result<()> {
        for l in &state.loans {
            let Some(holder) = l.holder else { continue };
            let Some(t_depth) = is_temp_target(l.target) else { continue };
            if t_depth >= min_depth && (check_locals.contains(&holder) || outer_live.contains(&holder)) {
                bail!("temporary value dropped while borrowed");
            }
        }
        Ok(())
    }

    pub(super) fn check_assign_escape(&mut self, p: &Place, sources: &[BorrowSource], state: &mut State) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let mut targets: Vec<u32> = if p.through_ref {
            state.loans.iter().filter(|l| l.holder == Some(p.local) && l.mutable).map(|l| l.target).collect()
        } else {
            Vec::new()
        };
        if p.through_ref && (!self.is_param_local(p.local) || self.closure_depth > 0) && !targets.contains(&p.local) {
            targets.push(p.local);
        }
        if p.through_ref && (self.is_param_local(p.local) || targets.iter().any(|&t| self.is_param_local(t))) {
            for s in sources {
                let allowed_ref = self.is_ref_ty(self.local_ty(s.target)) && (s.through_ref || self.is_closure_param(s.target));
                if is_temp_target(s.target).is_some() || (!self.from_caller(s.target, s.through_ref) && !allowed_ref) {
                    bail!("`{}` does not live long enough", self.local_name(s.target));
                }
            }
        }
        if self.closure_depth > 0 && self.scope_depth.get(p.local as usize).copied().unwrap_or(0) < self.closure_depth {
            for s in sources {
                let is_cparam = self.is_closure_param(s.target);
                let cparam_borrow = is_cparam && self.ty_holds_borrow(self.local_ty(s.target));
                let allowed_ref = self.is_ref_ty(self.local_ty(s.target)) && (s.through_ref || is_cparam);
                if is_temp_target(s.target).is_some()
                    || (!s.through_ref && !self.is_ref_ty(self.local_ty(s.target)) && self.closure_by_val_caps.contains(&s.target))
                    || (!self.from_caller(s.target, s.through_ref)
                        && !allowed_ref
                        && !cparam_borrow
                        && self.scope_depth.get(s.target as usize).copied().unwrap_or(0) >= self.closure_depth)
                {
                    bail!("borrowed data escapes outside of closure");
                }
            }
        }
        if p.through_ref {
            for holder in targets {
                if self.ty_holds_borrow(self.local_ty(holder)) {
                    super::bind::push_source_loans_filtered(
                        &mut state.loans,
                        Some(holder),
                        sources,
                        false,
                        |s| s.target != holder,
                        |s| s.mutable,
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn propagate_assign_sources(&mut self, p: &Place, sources: &[BorrowSource], state: &mut State) -> Result<()> {
        if !sources.is_empty() {
            self.check_assign_escape(p, sources, state)?;
            if !p.through_ref {
                self.push_sources_for_local(p.local, sources, state);
            }
        }
        Ok(())
    }

    pub(super) fn check_mut_ref_local_move(&self, dst_local: u32, src: ExprId, state: &State) -> Result<()> {
        if let Some(src_local) = self.expr_plain_local(src)
            && src_local != dst_local
            && self.is_mut_ref_ty(self.local_ty(dst_local))
            && self.is_mut_ref_ty(self.local_ty(src_local))
            && state.loans.iter().any(|l| loan_conflicts_borrow(l, src_local, &[]))
        {
            bail!("cannot move out of `{}` because it is borrowed", self.local_name(src_local));
        }
        Ok(())
    }

    pub(super) fn check_mut_ref_reborrow(&self, dst: ExprId, src: ExprId, state: &State) -> Result<()> {
        if let Some(dst_local) = self.expr_plain_local(dst) {
            self.check_mut_ref_local_move(dst_local, src, state)?;
        }
        Ok(())
    }

    pub(super) fn check_adt_drop_move(&mut self, ty: Ty) -> Result<()> {
        if self.ty_has_own_drop(ty) {
            bail!("cannot move out of type which implements the `Drop` trait");
        }
        Ok(())
    }

    pub(super) fn check_fru_base(&mut self, expr: ExprId, fields: List, b: ExprId, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        let deref_base = matches!(*self.body.expr(b), Expr::Unary(UnOp::Deref, _));
        let Some(p) = self.root_place(b) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let Some((adt_item, variant, args)) = self.struct_adt(expr) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let args_vec = self.cx.types.substs(args).to_vec();
        let Ok(field_tys) = super::super::infer::detached(self.cx, self.item, |infer| infer.adt_fields(adt_item, &args_vec)) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let pending_fields: Vec<(FieldKey, Ty)> = {
            let Ok(def) = self.cx.adt_def(adt_item) else {
                return self.check_expr(b, state, deref_base, outer_live, true, depth);
            };
            let Some(var) = def.variants.get(variant) else {
                return self.check_expr(b, state, deref_base, outer_live, true, depth);
            };
            let overridden: HashSet<&str> = self.body.list(fields).iter().map(|&id| self.src.text(self.body.fields[id as usize].name)).collect();
            let adt_src = &self.cx.file(adt_item).source;
            var.fields
                .iter()
                .enumerate()
                .filter_map(|(idx, fdef)| {
                    let (fname, key) = match fdef.name {
                        Some(n) => {
                            let s = adt_src.text(n);
                            (s.to_owned(), FieldKey::Named(s.to_owned()))
                        }
                        None => (idx.to_string(), FieldKey::Tuple(idx as u32)),
                    };
                    if overridden.contains(fname.as_str()) {
                        return None;
                    }
                    let fty = field_tys.get(variant).and_then(|fs| fs.get(idx)).copied().unwrap_or(Ty::NONE);
                    Some((key, fty))
                })
                .collect()
        };
        self.check_place_subexprs(b, state, outer_live, depth)?;
        let behind_ref = self.place_behind_ref(b, false);
        let mut drop_checked = false;
        let adt_ty = self.expr_ty(expr);
        for (key, fty) in pending_fields {
            let mut steps = p.steps.clone();
            push_step(&mut steps, PlaceStep::Field(key));
            if self.ty_is_copy(fty) {
                self.check_can_read(p.local, &steps, state)?;
            } else {
                if !drop_checked {
                    drop_checked = true;
                    self.check_adt_drop_move(adt_ty)?;
                }
                if behind_ref {
                    bail!("cannot move out of a shared or mutable reference");
                }
                self.check_can_move(p.local, &steps, state)?;
                if !p.through_ref {
                    self.push_partial_move(state, p.local, steps);
                }
            }
        }
        Ok(())
    }

    fn struct_adt(&self, expr: ExprId) -> Option<(crate::names::ItemRef, usize, super::super::substs::Substs)> {
        match self.typed.res.get(&expr).copied() {
            Some(Res::Ctor(item, v, args)) => Some((item, v as usize, args)),
            _ => match *self.cx.types.kind(self.expr_ty(expr)) {
                Kind::Adt(item, args) => Some((item, 0, args)),
                _ => None,
            },
        }
    }

    pub(super) fn check_place_subexprs(&mut self, expr: ExprId, state: &mut State, outer_live: &HashSet<u32>, depth: u32) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) | Expr::Unary(_, inner) => {
                self.check_place_subexprs(inner, state, outer_live, depth)
            }
            Expr::Index(base, idx) => {
                self.check_place_subexprs(base, state, outer_live, depth)?;
                self.check_expr(idx, state, false, outer_live, true, depth)
            }
            _ => Ok(()),
        }
    }

    pub(super) fn attach_method_arg_sources(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        param_ids: &[u32],
        recv_mut: bool,
        sig: Option<&Signature>,
        state: &mut State,
    ) -> Result<()> {
        let Some(mut p) = self.root_place(receiver) else {
            return Ok(());
        };
        let adjust = self.typed.adjusts.get(&expr).copied();
        let derefs = adjust.map_or(0, |a| a.derefs);
        if derefs > 0 {
            self.apply_deref_steps(expr, self.expr_ty(receiver), derefs, &mut p);
        } else if adjust.and_then(|a| a.autoref).is_none()
            && self.is_mut_ref_ty(self.expr_ty(receiver))
            && sig.is_some_and(super::calls::recv_is_by_ref)
        {
            self.apply_deref_steps(expr, self.expr_ty(receiver), 1, &mut p);
        }
        let self_ty = self.method_self_ty(expr, receiver, sig);
        let recv_ty = if self_ty != Ty::NONE { self_ty } else { self.local_ty(p.local) };
        if (!recv_mut && !self.ty_holds_closure(recv_ty)) || !self.ty_holds_borrow(recv_ty) {
            return Ok(());
        }
        let fn_item = match self.typed.res.get(&expr).copied() {
            Some(Res::Fn(item, _)) => Some(item),
            _ => None,
        };
        let keep_env = self.ty_holds_closure(recv_ty);
        let mut self_lts = Lifetimes::default();
        if let Some(item) = fn_item
            && let Some(block) = self.cx.owner_impl(item)
            && let Ok(hdr) = self.cx.impl_header(block)
        {
            let impl_src = &self.cx.impl_file(block).source;
            collect_ty_lifetimes(&hdr.body, impl_src, hdr.self_ty, &mut self_lts);
        }
        let mut all_sources = Vec::new();
        for (idx, &id) in param_ids.iter().enumerate() {
            let arg = ExprId(id);
            let arg_ty = self.expr_ty(arg);
            if arg_ty != Ty::NONE && !self.ty_holds_borrow(arg_ty) {
                continue;
            }
            if let Some(s) = sig
                && let Some(item) = fn_item
                && let Some(&(_, param_ty)) = s.params.get(idx)
            {
                let (sig_src, _) = self.cx.item(item);
                let mut p_lts = Lifetimes::default();
                collect_ty_lifetimes(&s.body, sig_src, param_ty, &mut p_lts);
                let is_fresh_ref = matches!(s.body.ty(param_ty), AstTy::Ref { lifetime: None, .. });
                let shares_self = p_lts.named.iter().any(|n| self_lts.named.contains(n))
                    || (!is_fresh_ref && ty_shares_name(&s.body, sig_src, param_ty, &self_lts.names))
                    || self.generic_param_ties_to_self(s, sig_src, &p_lts, &self_lts);
                if !shares_self {
                    continue;
                }
            }
            let arg_sources = self.expr_borrow_sources(arg, state);
            for s in self.unwrap_collection_arg_sources(arg, arg_ty, recv_ty, arg_sources, state) {
                if !s.env_only || keep_env {
                    all_sources.push(s);
                }
            }
        }
        self.propagate_assign_sources(&p, &all_sources, state)
    }

    fn unwrap_collection_arg_sources(
        &mut self,
        arg: ExprId,
        arg_ty: Ty,
        recv_ty: Ty,
        sources: Vec<BorrowSource>,
        state: &State,
    ) -> Vec<BorrowSource> {
        if arg_ty == Ty::NONE || recv_ty == Ty::NONE || sources.is_empty() {
            return sources;
        }
        let Kind::Ref(_, inner) = *self.cx.types.kind(arg_ty) else {
            return sources;
        };
        let inner_elems: Vec<Ty> = match *self.cx.types.kind(inner) {
            Kind::Slice(e) | Kind::Array(e, _) => vec![e],
            Kind::Adt(_, args) => self.cx.types.substs(args).to_vec(),
            _ => return sources,
        };
        let Kind::Adt(recv_item, recv_args) = *self.cx.types.kind(recv_ty) else {
            return sources;
        };
        let recv_substs = self.cx.types.substs(recv_args).to_vec();
        let coerced_target = match self.typed.coercions.get(&arg) {
            Some(Coercion::Deref(_, to)) => Some(*to),
            Some(Coercion::Vtable(v)) => Some(v.object),
            Some(Coercion::Len(_)) => match *self.cx.types.kind(inner) {
                Kind::Array(elem, _) => Some(self.cx.types.intern(Kind::Slice(elem))),
                _ => None,
            },
            _ => None,
        };
        if inner_elems.is_empty()
            || recv_substs.is_empty()
            || self.cx.adt_def(recv_item).is_ok_and(|def| def.generics.params.iter().any(|g| g.kind == GenericKind::Lifetime))
            || recv_substs.iter().any(|&t| self.ty_contains_ref_to(t, inner, coerced_target, &inner_elems))
        {
            return sources;
        }
        let mut unwrapped = arg;
        while let Expr::Paren(e) = *self.body.expr(unwrapped) {
            unwrapped = e;
        }
        let container_local = match *self.body.expr(unwrapped) {
            Expr::Ref { expr: inner_e, .. } => self.root_place(inner_e).map(|p| p.local),
            _ => self.root_place(unwrapped).and_then(|p| {
                state.loans.iter().find(|l| l.holder == Some(p.local) && !self.is_ref_ty(self.local_ty(l.target))).map(|l| l.target)
            }),
        };
        super::propagate::unwrap_container_loans(sources, state, |s| {
            is_temp_target(s.target).is_some() || container_local == Some(s.target)
        })
    }

    fn ty_contains_ref_to(&self, ty: Ty, container: Ty, coerced: Option<Ty>, elems: &[Ty]) -> bool {
        match *self.cx.types.kind(ty) {
            Kind::Ref(_, target) => target == container || coerced == Some(target) || elems.contains(&target),
            Kind::Tuple(items) | Kind::Adt(_, items) => {
                self.cx.types.substs(items).iter().any(|&t| self.ty_contains_ref_to(t, container, coerced, elems))
            }
            Kind::Slice(t) | Kind::Array(t, _) | Kind::Pat(t, ..) => self.ty_contains_ref_to(t, container, coerced, elems),
            _ => false,
        }
    }

    fn method_self_ty(&self, expr: ExprId, receiver: ExprId, sig: Option<&Signature>) -> Ty {
        let mut cur = self.expr_ty(receiver);
        if cur == Ty::NONE {
            return Ty::NONE;
        }
        let adjust = self.typed.adjusts.get(&expr).copied();
        let derefs = adjust.map_or(0, |a| a.derefs);
        for step in 0..derefs {
            if self.typed.derefs.contains_key(&(expr, step)) {
                return Ty::NONE;
            }
            cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            if cur == Ty::NONE {
                return Ty::NONE;
            }
        }
        if adjust.and_then(|a| a.autoref).is_none()
            && sig.is_some_and(super::calls::recv_is_by_ref)
            && self.is_ref_ty(cur)
        {
            cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
        }
        cur
    }

    fn generic_param_ties_to_self(&self, sig: &Signature, src: &crate::outline::source::Source, p_lts: &Lifetimes<'_>, self_lts: &Lifetimes<'_>) -> bool {
        for g in &sig.generics {
            let gname = src.text(g.name);
            if !p_lts.names.contains(&gname) && !p_lts.named.contains(&gname) {
                continue;
            }
            let mut b_lts = Lifetimes::default();
            for &b_id in sig.body.list(g.bounds) {
                if let crate::body::Bound::Lifetime(lt) = sig.body.bounds[b_id as usize] {
                    if self_lts.named.contains(&src.text(lt)) {
                        return true;
                    }
                } else if let crate::body::Bound::Trait { path, .. } = sig.body.bounds[b_id as usize] {
                    for &seg in sig.body.list(sig.body.path(path).segments) {
                        if let Some(crate::body::Args::Angle(args)) = sig.body.segments[seg as usize].args {
                            for &a in sig.body.list(args) {
                                if let crate::body::GenericArg::Binding { ty, .. } = sig.body.args[a as usize] {
                                    collect_ty_lifetimes(&sig.body, src, ty, &mut b_lts);
                                }
                            }
                        }
                    }
                }
            }
            if b_lts.names.iter().any(|n| self_lts.names.contains(n)) || b_lts.named.iter().any(|n| self_lts.named.contains(n)) {
                return true;
            }
        }
        false
    }

    pub(super) fn attach_call_out_sources(&mut self, expr: ExprId, arg_ids: &[u32], sig: Option<&Signature>, fn_item: Option<crate::names::ItemRef>, has_self_arg: bool, state: &mut State) -> Result<()> {
        let Some(sig) = sig else { return Ok(()) };
        if has_self_arg && !arg_ids.is_empty() {
            let recv_mut = super::calls::recv_ref_mut(sig) == Some(Mut::Mut);
            let recv_expr = match *self.body.expr(ExprId(arg_ids[0])) {
                Expr::Ref { mutable: true, raw: false, expr: inner } => inner,
                _ => ExprId(arg_ids[0]),
            };
            self.attach_method_arg_sources(expr, recv_expr, &arg_ids[1..], recv_mut, Some(sig), state)?;
        }
        let param_args = if has_self_arg { arg_ids.get(1..).unwrap_or(&[]) } else { arg_ids };
        if param_args.len() < 2 || sig.params.len() < 2 {
            return Ok(());
        }
        let fn_src = fn_item.map_or(self.src, |item| self.cx.item(item).0);
        for (dst_idx, &dst_id) in param_args.iter().enumerate() {
            let Some(&(_, dst_ast)) = sig.params.get(dst_idx) else { continue };
            let AstTy::Ref { mutable: true, ty: dst_inner, .. } = *sig.body.ty(dst_ast) else { continue };
            let dst_expr = match *self.body.expr(ExprId(dst_id)) {
                Expr::Ref { mutable: true, raw: false, expr } => expr,
                _ => continue,
            };
            let Some(p) = self.root_place(dst_expr) else { continue };
            let dst_ty = self.expr_ty(dst_expr);
            if !self.ty_holds_borrow(if dst_ty != Ty::NONE { dst_ty } else { self.local_ty(p.local) }) {
                continue;
            }
            for (src_idx, &src_id) in param_args.iter().enumerate() {
                if src_idx == dst_idx {
                    continue;
                }
                let Some(&(_, src_ast)) = sig.params.get(src_idx) else { continue };
                if self.sig_params_share_lt_or_ty(sig, fn_src, dst_inner, src_ast) {
                    let src_expr = match (*sig.body.ty(src_ast), *self.body.expr(ExprId(src_id))) {
                        (AstTy::Ref { lifetime: None, mutable: true, ty: src_inner }, Expr::Ref { mutable: true, raw: false, expr })
                            if !matches!(*sig.body.ty(dst_inner), AstTy::Ref { .. })
                                && self.sig_params_share_lt_or_ty(sig, fn_src, dst_inner, src_inner) => expr,
                        _ => ExprId(src_id),
                    };
                    let sources = self.expr_borrow_sources(src_expr, state);
                    self.propagate_assign_sources(&p, &sources, state)?;
                }
            }
        }
        Ok(())
    }

    fn sig_params_share_lt_or_ty(&self, sig: &Signature, src: &crate::outline::source::Source, a: crate::body::TyId, b: crate::body::TyId) -> bool {
        let mut lts_a = Lifetimes::default();
        let mut lts_b = Lifetimes::default();
        collect_ty_lifetimes(&sig.body, src, a, &mut lts_a);
        collect_ty_lifetimes(&sig.body, src, b, &mut lts_b);
        if lts_a.named.iter().any(|n| lts_b.named.contains(n)) {
            return true;
        }
        sig.generics.iter().any(|g| {
            g.kind == GenericKind::Type && {
                let name = src.text(g.name);
                lts_a.names.contains(&name) && lts_b.names.contains(&name)
            }
        })
    }

    pub(super) fn has_new_moves(before_moved: &[bool], before_partial: &[PartialMove], after: &State, locals: &[u32]) -> bool {
        locals.iter().any(|&l| {
            let idx = l as usize;
            (after.moved.get(idx).copied().unwrap_or(false) && !before_moved.get(idx).copied().unwrap_or(false))
                || after.partial_moves.iter().any(|m| m.local == l && !before_partial.contains(m))
        })
    }
}
