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
                if !allowed_ref
                    && let Some(origins) = self.closure_param_origins.get(&s.target).cloned()
                {
                    self.check_sources_against_targets(true, true, false, &origins)?;
                    continue;
                }
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
                        && !s.origin_only
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

    pub(in super::super) fn check_tail_drop_temps(
        &mut self,
        tail: ExprId,
        block_depth: u32,
        state: &State,
    ) -> Result<()> {
        let mut subs = Vec::new();
        self.collect_drop_temp_subexprs(tail, false, &mut subs);
        for sub in subs {
            let ty = self.expr_ty(sub);
            if ty == Ty::NONE || !self.ty_holds_borrow(ty) || !self.ty_has_borrow_drop(ty) {
                continue;
            }
            for src in self.expr_borrow_sources(sub, state) {
                if self.escapes_depth(src.target, src.through_ref, src.mutable, block_depth) {
                    return Err(self.does_not_live_long_enough(Some(sub), src.target));
                }
            }
        }
        Ok(())
    }

    pub(in super::super) fn collect_drop_temp_sources(
        &mut self,
        expr: ExprId,
        state: &State,
    ) -> Vec<BorrowSource> {
        let mut subs = Vec::new();
        self.collect_drop_temp_subexprs(expr, true, &mut subs);
        let mut out = Vec::new();
        for sub in subs {
            let ty = self.expr_ty(sub);
            if ty == Ty::NONE || !self.ty_holds_borrow(ty) || !self.ty_has_borrow_drop(ty) {
                continue;
            }
            out.extend(self.expr_borrow_sources(sub, state));
        }
        out
    }

    fn collect_drop_temp_subexprs(
        &mut self,
        expr: ExprId,
        include_self: bool,
        out: &mut Vec<ExprId>,
    ) {
        if include_self && self.root_place(expr).is_none() {
            out.push(expr);
        }
        match *self.body.expr(expr) {
            Expr::Paren(e)
            | Expr::Cast(e, _)
            | Expr::Unary(_, e)
            | Expr::Ref { expr: e, .. }
            | Expr::Field(e, _)
            | Expr::TupleField(e, _)
            | Expr::Try(e)
            | Expr::Await(e) => self.collect_drop_temp_subexprs(e, true, out),
            Expr::Binary(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) => {
                self.collect_drop_temp_subexprs(a, true, out);
                self.collect_drop_temp_subexprs(b, true, out);
            }
            Expr::Call(callee, args) => {
                self.collect_drop_temp_subexprs(callee, true, out);
                for a in self.body.list(args).to_vec() {
                    self.collect_drop_temp_subexprs(ExprId(a), true, out);
                }
            }
            Expr::Method {
                receiver, params, ..
            } => {
                self.collect_drop_temp_subexprs(receiver, true, out);
                for a in self.body.list(params).to_vec() {
                    self.collect_drop_temp_subexprs(ExprId(a), true, out);
                }
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for e in self.body.list(list).to_vec() {
                    self.collect_drop_temp_subexprs(ExprId(e), true, out);
                }
            }
            Expr::Struct { fields, base, .. } => {
                let vals: Vec<ExprId> = self
                    .body
                    .list(fields)
                    .iter()
                    .map(|&f| self.body.fields[f as usize].value)
                    .collect();
                for v in vals {
                    self.collect_drop_temp_subexprs(v, true, out);
                }
                if let Some(b) = base {
                    self.collect_drop_temp_subexprs(b, true, out);
                }
            }
            _ => {}
        }
    }

    pub(in super::super) fn lock_slice_index_base(
        &mut self,
        expr: ExprId,
        base: ExprId,
        state: &mut State,
    ) -> Option<u32> {
        if !self.expr_has_slice_index(base) {
            return None;
        }
        let p = self.root_place(base)?;
        let tid = self.next_temp_holder();
        super::super::bind::push_loan(
            &mut state.loans,
            super::super::Loan::new(p.local, Vec::new(), Some(tid), false, false, Some(expr)),
        );
        Some(tid)
    }

    fn expr_has_slice_index(&self, mut expr: ExprId) -> bool {
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) => {
                    expr = inner;
                }
                Expr::Index(inner, _) => {
                    let mut ty = self.expr_ty(inner);
                    while let Kind::Ref(_, pointee) = *self.cx.types.kind(ty) {
                        ty = pointee;
                    }
                    if matches!(*self.cx.types.kind(ty), Kind::Slice(_)) {
                        return true;
                    }
                    expr = inner;
                }
                _ => return false,
            }
        }
    }

    pub(in super::super) fn check_index_mut_impl(&mut self, expr: ExprId) -> Result<()> {
        let Some(crate::sema::infer::Res::Fn(method, substs)) =
            self.typed.operators.get(&expr).copied()
        else {
            return Ok(());
        };
        if self.cx.name(method) != "index_mut" {
            return Ok(());
        }
        let Some(tr) = self.cx.owner_trait(method) else {
            return Ok(());
        };
        let args = self.cx.types.substs(substs).to_vec();
        let Some((&recv_ty, trait_args)) = args.split_first() else {
            return Ok(());
        };
        if self.cx.types.has_params(recv_ty)
            || trait_args.iter().any(|&t| self.cx.types.has_params(t))
        {
            return Ok(());
        }
        let found = crate::sema::infer::detached(self.cx, self.item, |infer| {
            infer.select(tr, recv_ty, trait_args)
        })
        .ok()
        .flatten()
        .is_some();
        if !found {
            return Err(self.borrow_err(
                Some(expr),
                "cannot borrow data in an index as mutable: trait `IndexMut` is not implemented",
            ));
        }
        Ok(())
    }

    pub(in super::super) fn check_static_assoc_ref_ann(
        &mut self,
        ty_id: crate::body::TyId,
    ) -> Result<()> {
        let crate::body::Ty::Ref {
            lifetime: Some(lt),
            ty: inner,
            ..
        } = *self.body.ty(ty_id)
        else {
            return Ok(());
        };
        if self.src.text(lt) != "'static" {
            return Ok(());
        }
        let crate::body::Ty::Path(path_id) = *self.body.ty(inner) else {
            return Ok(());
        };
        let p = self.body.path(path_id);
        let segs = self.body.list(p.segments);
        if segs.len() < 2 {
            return Ok(());
        }
        let last = self.body.segments[segs[segs.len() - 1] as usize];
        let has_non_static_lt_arg = match last.args {
            Some(crate::body::Args::Angle(args)) => self.body.list(args).iter().any(|&a| {
                matches!(
                    self.body.args[a as usize],
                    crate::body::GenericArg::Lifetime(arg_lt) if self.src.text(arg_lt) != "'static"
                )
            }),
            _ => false,
        };
        if !has_non_static_lt_arg {
            return Ok(());
        }
        let first_name =
            crate::names::plain(self.src.text(self.body.segments[segs[0] as usize].name));
        let Ok(sig) = self.cx.signature(self.item) else {
            return Ok(());
        };
        let is_ty_param = sig.generics.iter().any(|g| {
            g.kind == crate::body::sig::GenericKind::Type && self.src.text(g.name) == first_name
        });
        if is_ty_param {
            return Err(self.borrow_err(
                None,
                "the associated type may not live long enough: `'static` required",
            ));
        }
        Ok(())
    }

    pub(in super::super) fn ret_is_elided_impl_self(&mut self) -> bool {
        let Ok(sig) = self.cx.signature(self.item) else {
            return false;
        };
        if sig.receiver.is_some() {
            return false;
        }
        let Some(ret_ast) = sig.ret else {
            return false;
        };
        let crate::body::Ty::Path(p) = *sig.body.ty(ret_ast) else {
            return false;
        };
        let segs = sig.body.list(sig.body.path(p).segments);
        if segs.len() != 1 || self.src.text(sig.body.segments[segs[0] as usize].name) != "Self" {
            return false;
        }
        let Some(impl_ref) = self.cx.owner_impl(self.item) else {
            return false;
        };
        let Ok(hdr) = self.cx.impl_header(impl_ref) else {
            return false;
        };
        let impl_src = &self.cx.impl_file(impl_ref).source;
        let mut lts = super::super::calls::Lifetimes::default();
        super::super::calls::collect_ty_lifetimes(&hdr.body, impl_src, hdr.self_ty, &mut lts);
        lts.has_elided && lts.named.is_empty()
    }
}
