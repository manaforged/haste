use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::sig::GenericKind;
use crate::body::{Expr, ExprId};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::ty::{Kind, Ty};

use super::super::bind::push_source_loans_filtered;
use super::super::places::{Place, PlaceStep};
use super::super::propagate::BorrowSource;
use super::super::{Checker, State, is_temp_holder};
use super::is_temp_target;

impl Checker<'_, '_> {
    pub(in super::super) fn ty_has_borrow_interior_mut(&mut self, ty: Ty) -> bool {
        if ty == Ty::NONE {
            return false;
        }
        if let Some(&cached) = self.variance.interior_mut_cache.get(&ty) {
            return cached;
        }
        let mut seen = HashSet::default();
        let res = self.ty_has_borrow_interior_mut_inner(ty, &mut seen);
        self.variance.interior_mut_cache.insert(ty, res);
        res
    }

    fn ty_has_borrow_interior_mut_inner(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(_, inner)
            | Kind::Slice(inner)
            | Kind::Array(inner, _)
            | Kind::Pat(inner, ..) => self.ty_has_borrow_interior_mut_inner(inner, seen),
            Kind::Tuple(items) => self
                .cx
                .types
                .substs(items)
                .to_vec()
                .into_iter()
                .any(|t| self.ty_has_borrow_interior_mut_inner(t, seen)),
            Kind::Adt(item, args) => self.adt_has_borrow_interior_mut(ty, item, args, seen),
            _ => false,
        }
    }

    fn adt_has_borrow_interior_mut(
        &mut self,
        ty: Ty,
        item: ItemRef,
        args: super::super::super::substs::Substs,
        seen: &mut HashSet<Ty>,
    ) -> bool {
        if matches!(self.cx.lang(item), Some("unsafe_cell" | "unsafe_pinned")) {
            let mut ptr_seen = HashSet::default();
            return self
                .cx
                .types
                .substs(args)
                .first()
                .copied()
                .is_some_and(|inner| self.ty_holds_borrow_or_ptr(inner, &mut ptr_seen));
        }
        if self.cx.lang(item) == Some("phantom_data") {
            return self
                .cx
                .types
                .substs(args)
                .first()
                .copied()
                .is_some_and(|inner| self.ty_has_borrow_interior_mut_inner(inner, seen));
        }
        if self.cx.item(item).1.kind == ItemKind::Union {
            return false;
        }
        if let Some(owned) = self.cx.boxed(ty) {
            return self.ty_has_borrow_interior_mut_inner(owned, seen);
        }
        let allow_ptr = matches!(self.cx.name(item), "NonNull" | "Unique");
        let args_vec = self.cx.types.substs(args).to_vec();
        let Ok(fields) = super::super::super::infer::detached(self.cx, item, |infer| {
            infer.adt_fields(item, &args_vec)
        }) else {
            return false;
        };
        fields
            .iter()
            .flatten()
            .any(|&fty| match *self.cx.types.kind(fty) {
                Kind::Ptr(_, pointee) if allow_ptr => {
                    self.ty_has_borrow_interior_mut_inner(pointee, seen)
                }
                _ => self.ty_has_borrow_interior_mut_inner(fty, seen),
            })
    }

    fn ty_holds_borrow_or_ptr(&mut self, ty: Ty, seen: &mut HashSet<Ty>) -> bool {
        if ty == Ty::NONE || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty).clone() {
            Kind::Ref(..) | Kind::Dyn(..) | Kind::Proj(_) => true,
            Kind::Param(idx) => !self.static_type_params.contains(&idx),
            Kind::Ptr(_, inner)
            | Kind::Slice(inner)
            | Kind::Array(inner, _)
            | Kind::Pat(inner, ..) => self.ty_holds_borrow_or_ptr(inner, seen),
            Kind::Tuple(items) => self
                .cx
                .types
                .substs(items)
                .to_vec()
                .into_iter()
                .any(|t| self.ty_holds_borrow_or_ptr(t, seen)),
            Kind::Closure(..) => self.ty_holds_borrow(ty),
            Kind::Adt(item, args) => {
                let args = self.cx.types.substs(args).to_vec();
                args.into_iter()
                    .any(|a| self.ty_holds_borrow_or_ptr(a, seen))
                    || self.cx.adt_def(item).is_ok_and(|def| {
                        def.generics
                            .params
                            .iter()
                            .any(|g| g.kind == GenericKind::Lifetime)
                    })
            }
            _ => false,
        }
    }

    pub(in super::super) fn collect_ptr_targets_expr(
        &mut self,
        mut expr: ExprId,
        state: &State,
        out: &mut Vec<u32>,
    ) {
        loop {
            expr = self.unwrap_parens(expr);
            match *self.body.expr(expr) {
                Expr::Cast(inner, _) => expr = inner,
                Expr::Ref { expr: inner, .. } => {
                    if let Some(p) = self.root_place(inner) {
                        self.collect_place_write_targets(&p, state, out);
                    }
                    return;
                }
                Expr::Method { receiver, .. } => {
                    if let Some(p) = self.root_place(receiver) {
                        self.collect_place_write_targets(&p, state, out);
                    }
                    return;
                }
                Expr::Call(_, args) if self.body.list(args).len() == 1 => {
                    expr = ExprId(self.body.list(args)[0]);
                }
                Expr::Path(_) => {
                    if let Some(local) = self.expr_plain_local(expr) {
                        if let Some(targets) = state.ptr_targets.get(&local) {
                            for &t in targets {
                                if !out.contains(&t) {
                                    out.push(t);
                                }
                            }
                        }
                        if self.is_param_local(local) && !out.contains(&local) {
                            out.push(local);
                        }
                    }
                    return;
                }
                _ => return,
            }
        }
    }

    fn collect_place_write_targets(&self, p: &Place, state: &State, out: &mut Vec<u32>) {
        let lty = self.local_ty(p.local);
        let follow = p.through_ref || self.is_ref_ty(lty) || self.is_mut_ptr_ty(lty);
        if !follow {
            if !out.contains(&p.local) {
                out.push(p.local);
            }
            return;
        }
        for l in &state.loans {
            if l.holder == Some(p.local) && !out.contains(&l.target) {
                out.push(l.target);
            }
        }
        if let Some(targets) = state.ptr_targets.get(&p.local) {
            for &t in targets {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        if self.is_param_local(p.local) && !out.contains(&p.local) {
            out.push(p.local);
        }
    }

    pub(in super::super) fn record_ptr_targets(
        &mut self,
        dst_local: u32,
        rhs: ExprId,
        state: &mut State,
    ) {
        let mut targets = Vec::new();
        self.collect_ptr_targets_expr(rhs, state, &mut targets);
        if targets.is_empty() {
            state.ptr_targets.remove(&dst_local);
        } else {
            state.ptr_targets.insert(dst_local, targets);
        }
    }

    pub(in super::super) fn update_assign_ptr_targets(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        state: &mut State,
    ) {
        if let Some(local) = self.expr_plain_local(lhs)
            && self.is_mut_ptr_ty(self.local_ty(local))
        {
            self.record_ptr_targets(local, rhs, state);
        }
    }

    pub(in super::super) fn check_drop_order(&self, min_depth: u32, state: &State) -> Result<()> {
        for l in &state.loans {
            let Some(holder) = l.holder else { continue };
            if is_temp_holder(Some(holder))
                || state.moved.get(holder as usize).copied().unwrap_or(false)
                || !self.drop_locals.contains(&holder)
            {
                continue;
            }
            if is_temp_target(l.target).is_some()
                || (l.through_ref
                    && (self.is_ref_ty(self.local_ty(l.target)) || self.is_closure_param(l.target)))
            {
                continue;
            }
            let target_depth = self
                .scope_depth
                .get(l.target as usize)
                .copied()
                .unwrap_or(0);
            if target_depth < min_depth {
                continue;
            }
            let holder_depth = self.scope_depth.get(holder as usize).copied().unwrap_or(0);
            let target_drops_first = holder_depth < target_depth
                || (holder_depth == target_depth
                    && self
                        .variance
                        .decl_seq
                        .get(holder as usize)
                        .copied()
                        .unwrap_or(0)
                        < self
                            .variance
                            .decl_seq
                            .get(l.target as usize)
                            .copied()
                            .unwrap_or(0));
            if target_drops_first {
                return Err(self.does_not_live_long_enough(l.site, l.target));
            }
        }
        Ok(())
    }

    pub(in super::super) fn check_target_write_escape(
        &mut self,
        p: &Place,
        targets: &[u32],
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        let param_target =
            self.is_param_local(p.local) || targets.iter().any(|&t| self.is_param_local(t));
        self.check_sources_against_targets(
            p.through_ref || p.through_deref,
            param_target,
            self.scope_depth.get(p.local as usize).copied().unwrap_or(0) < self.closure_depth,
            sources,
        )?;
        let has_field = p
            .steps
            .iter()
            .any(|st| matches!(st, PlaceStep::Field(..) | PlaceStep::SliceElem { .. }));
        if !p.through_ref {
            return self.check_sources_fit_target(
                self.current_expr,
                p.local,
                false,
                has_field,
                sources,
            );
        }
        if self.is_param_local(p.local) {
            self.check_sources_fit_target(self.current_expr, p.local, true, has_field, sources)?;
        }
        self.apply_write_to_targets(Some(p.local), has_field, targets, sources, state)
    }

    pub(in super::super) fn check_ptr_deref_write(
        &mut self,
        targets: &[u32],
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        let param_target = targets.iter().any(|&t| self.is_param_local(t));
        let closure_target = self.closure_depth > 0
            && targets.iter().any(|&t| {
                self.scope_depth.get(t as usize).copied().unwrap_or(0) < self.closure_depth
            });
        self.check_sources_against_targets(true, param_target, closure_target, sources)?;
        self.apply_write_to_targets(None, false, targets, sources, state)
    }

    fn check_sources_against_targets(
        &mut self,
        deref_write: bool,
        param_target: bool,
        closure_target: bool,
        sources: &[BorrowSource],
    ) -> Result<()> {
        if deref_write && param_target {
            for s in sources {
                let allowed_ref = self.is_ref_ty(self.local_ty(s.target))
                    && (s.through_ref || self.is_closure_param(s.target));
                if is_temp_target(s.target).is_some()
                    || (!self.from_caller(s.target, s.through_ref) && !allowed_ref)
                {
                    return Err(self.does_not_live_long_enough(None, s.target));
                }
            }
        }
        if self.closure_depth > 0 && closure_target {
            for s in sources {
                let is_cparam = self.is_closure_param(s.target);
                let cparam_borrow = is_cparam && self.ty_holds_borrow(self.local_ty(s.target));
                let allowed_ref =
                    self.is_ref_ty(self.local_ty(s.target)) && (s.through_ref || is_cparam);
                if is_temp_target(s.target).is_some()
                    || (!s.through_ref
                        && !self.is_ref_ty(self.local_ty(s.target))
                        && self.closure_by_val_caps.contains(&s.target))
                    || (!self.from_caller(s.target, s.through_ref)
                        && !allowed_ref
                        && !cparam_borrow
                        && self
                            .scope_depth
                            .get(s.target as usize)
                            .copied()
                            .unwrap_or(0)
                            >= self.closure_depth)
                {
                    return Err(self.borrow_err(None, "borrowed data escapes outside of closure"));
                }
            }
        }
        Ok(())
    }

    fn apply_write_to_targets(
        &mut self,
        root_local: Option<u32>,
        root_has_field: bool,
        targets: &[u32],
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        for &holder in targets {
            let is_root = root_local == Some(holder);
            let peel = is_root && self.is_ref_ty(self.local_ty(holder));
            let holder_field = if is_root { root_has_field } else { false };
            self.check_sources_fit_target(self.current_expr, holder, peel, holder_field, sources)?;
            state.set_tuple_sources(holder, None);
            if self.ty_holds_borrow(self.local_ty(holder)) {
                push_source_loans_filtered(
                    &mut state.loans,
                    Some(holder),
                    sources,
                    false,
                    self.current_expr,
                    |s| s.target != holder,
                    |s| s.mutable,
                );
            }
        }
        Ok(())
    }
}
