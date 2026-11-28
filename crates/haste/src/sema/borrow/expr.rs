use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::{BinOp, BlockKind, Expr, ExprId, List, PatId, UnOp, jumps_to};
use crate::outline::item::ItemKind;

use super::super::infer::Res;
use super::super::ty::{Kind, Mut, Ty};
use super::places::PlaceStep;
use super::{Checker, State, merge_branch, prune_dead_loans, prune_moved_loans};

impl Checker<'_, '_> {
    fn finish_expr_divergence(&self, expr: ExprId, state: &mut State) {
        let expr_ty = self.expr_ty(expr);
        if expr_ty != Ty::NONE && matches!(*self.cx.types.kind(expr_ty), Kind::Never) {
            state.diverged = true;
        }
    }

    pub(super) fn check_expr(
        &mut self,
        expr: ExprId,
        state: &mut State,
        place_ctx: bool,
        outer_live: &HashSet<u32>,
        value_used: bool,
        depth: u32,
    ) -> Result<()> {
        if state.diverged {
            return Ok(());
        }
        let prev_expr = self.current_expr;
        self.current_expr = Some(expr);
        let res = self.check_expr_kind(expr, state, place_ctx, outer_live, value_used, depth);
        self.current_expr = prev_expr;
        res?;
        self.finish_expr_divergence(expr, state);
        Ok(())
    }

    fn check_expr_kind(
        &mut self,
        expr: ExprId,
        state: &mut State,
        place_ctx: bool,
        outer_live: &HashSet<u32>,
        value_used: bool,
        depth: u32,
    ) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Path(_) => self.check_path_expr(expr, state, place_ctx),
            Expr::Ref {
                mutable,
                raw,
                expr: inner,
            } => self.check_ref_expr(inner, mutable, raw, state, outer_live, depth),
            Expr::Assign(lhs, rhs) => self.check_assign_expr(lhs, rhs, state, outer_live, depth),
            Expr::AssignOp(_, lhs, rhs) => {
                self.check_assign_op_expr(expr, lhs, rhs, state, outer_live, depth)
            }
            Expr::Block {
                kind: BlockKind::Async | BlockKind::AsyncMove,
                ..
            } => self.check_closure(expr, None, expr, state, depth),
            Expr::Block {
                kind,
                stmts,
                tail,
                label,
            } => self.check_block(
                expr,
                kind,
                stmts,
                tail,
                label,
                state,
                (place_ctx, value_used),
                outer_live,
                depth,
            ),
            Expr::If(cond, then, otherwise) => self.check_if(
                expr,
                cond,
                then,
                otherwise,
                state,
                (place_ctx, value_used),
                outer_live,
                depth,
            ),
            Expr::Let(pat, init) => self.check_let_expr(pat, init, state, outer_live, depth),
            Expr::Match(scrut, arms) => self.check_match(
                expr,
                scrut,
                arms,
                state,
                (place_ctx, value_used),
                outer_live,
                depth,
            ),
            Expr::Loop { label, body } => self.check_loop(
                Some(expr),
                label,
                None,
                None,
                None,
                body,
                state,
                outer_live,
                depth,
            ),
            Expr::While { label, cond, body } => self.check_loop(
                None,
                label,
                Some(cond),
                None,
                None,
                body,
                state,
                outer_live,
                depth,
            ),
            Expr::For {
                label,
                pat,
                iter,
                body,
            } => self.check_for(label, pat, iter, body, state, outer_live, depth),
            Expr::Call(callee, args) => {
                self.check_call(expr, callee, args, state, outer_live, depth)
            }
            Expr::Method {
                receiver, params, ..
            } => self.check_method(expr, receiver, params, state, outer_live, depth),
            Expr::Unary(op, inner) => {
                self.check_unary_expr(expr, op, inner, state, place_ctx, outer_live, depth)
            }
            Expr::Index(a, b) => {
                self.check_index_expr(expr, a, b, state, place_ctx, outer_live, depth)
            }
            Expr::Binary(op, a, b) => self.check_binary_expr(op, a, b, state, outer_live, depth),
            Expr::Repeat(a, b) => self.check_repeat_expr(a, b, state, outer_live, depth),
            Expr::Field(base, _) | Expr::TupleField(base, _) => {
                self.check_field_expr(expr, base, state, place_ctx, outer_live, depth)
            }
            Expr::Cast(inner, _) => self.check_cast_expr(inner, state, outer_live, depth),
            Expr::Try(inner) | Expr::Await(inner) | Expr::Become(inner) => {
                self.check_expr(inner, state, false, outer_live, true, depth)
            }
            Expr::Paren(inner) => {
                self.check_expr(inner, state, place_ctx, outer_live, value_used, depth)
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                self.check_list_expr(list, state, outer_live, depth)
            }
            Expr::Struct { fields, base, .. } => {
                self.check_struct_expr(expr, fields, base, state, outer_live, depth)
            }
            Expr::Range { from, to, .. } => {
                self.check_range_expr(from, to, state, outer_live, depth)
            }
            Expr::Return(val) => self.check_return_expr(val, state, outer_live, depth),
            Expr::Break(label, val) => self.check_break_expr(label, val, state, outer_live, depth),
            Expr::Continue(label) => {
                if let Some(idx) = self
                    .breaks
                    .iter()
                    .rposition(|f| !f.is_block && jumps_to(self.src, (f.label, false), label))
                {
                    let cont_depth = self.breaks[idx].depth;
                    self.check_drop_order(cont_depth, state)?;
                    self.breaks[idx].continues.push(state.clone());
                }
                state.diverged = true;
                Ok(())
            }
            Expr::Yield(val) => self.check_yield_expr(val, state, outer_live, depth),
            Expr::Closure { params, body, .. } => {
                self.check_closure(expr, Some(params), body, state, depth)
            }
            Expr::Lit(..)
            | Expr::Macro { .. }
            | Expr::Underscore
            | Expr::FormatArgs(_)
            | Expr::OffsetOf { .. } => Ok(()),
        }
    }

    fn check_path_expr(&mut self, expr: ExprId, state: &mut State, place_ctx: bool) -> Result<()> {
        if let Some(Res::Local(local)) = self.typed.res.get(&expr)
            && (*local as usize) < self.is_copy.len()
        {
            if place_ctx || self.is_copy[*local as usize] {
                self.check_can_read(*local, &[], state)?;
            } else {
                self.check_can_move(*local, &[], state)?;
                self.mark_local_moved(state, *local, Some(expr));
            }
        } else if let Some(Res::Fn(item, _)) = self.typed.res.get(&expr)
            && self.cx.item(*item).1.kind == ItemKind::Static
        {
            if self.unsafe_depth == 0 && self.is_static_mut(*item) {
                return Err(self.unsafe_err(
                    Some(expr),
                    format!(
                        "use of mutable static `{}` is unsafe and requires unsafe function or block",
                        self.cx.name(*item)
                    ),
                ));
            }
            if !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
                return Err(self.borrow_err(
                    Some(expr),
                    format!("cannot move out of static item `{}`", self.cx.name(*item)),
                ));
            }
        }
        Ok(())
    }

    fn check_ref_expr(
        &mut self,
        inner: ExprId,
        mutable: bool,
        raw: bool,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if !raw {
            self.check_union_place(inner)?;
        }
        if mutable && self.place_behind_shared_ref(inner) {
            return Err(self.borrow_err(None, "cannot borrow data in a `&` reference as mutable"));
        }
        if mutable {
            self.check_static_mut(inner, false)?;
        }
        if let Some(p) = self.root_place(inner) {
            if mutable && p.behind_shared {
                return Err(self.borrow_err(
                    None,
                    "cannot borrow data in a `&` reference or `*const` pointer as mutable",
                ));
            }
            if let Some(prefix_len) = p.raw_ptr_prefix {
                self.check_can_read(p.local, &p.steps[..prefix_len], state)?;
            } else {
                self.check_borrow(
                    p.local,
                    p.through_ref,
                    p.mut_ref_root,
                    &p.steps,
                    mutable,
                    state,
                )?;
            }
            if mutable {
                state.set_tuple_sources(p.local, None);
            }
        }
        self.check_expr(inner, state, true, outer_live, true, depth)
    }

    fn check_assign_expr(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let mut after_rhs = outer_live.clone();
        self.collect_live_expr(lhs, &mut after_rhs);
        let reborrow_mut = self.assign_reborrow_or_place_ctx(lhs, rhs);
        self.check_expr(rhs, state, reborrow_mut, &after_rhs, true, depth)?;
        if !state.diverged {
            self.check_mut_ref_reborrow(lhs, rhs, state)?;
            let sources = self.expr_borrow_sources(rhs, state);
            let tuple_srcs = self
                .expr_plain_local(lhs)
                .and_then(|l| self.expr_tuple_sources(rhs, state, false).map(|ts| (l, ts)));
            prune_dead_loans(state, &after_rhs);
            self.assign_place(lhs, Some(rhs), &sources, state, outer_live, depth)?;
            self.update_assign_ptr_targets(lhs, rhs, state);
            if let Some((local, ts)) = tuple_srcs {
                state.set_tuple_sources(local, Some(ts));
            }
            prune_moved_loans(state);
        }
        Ok(())
    }

    fn check_assign_op_expr(
        &mut self,
        expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_union_place(lhs)?;
        let mut after_rhs = outer_live.clone();
        self.collect_live_expr(lhs, &mut after_rhs);
        self.check_expr(rhs, state, false, &after_rhs, true, depth)?;
        if !state.diverged {
            let sources = if self.typed.operators.contains_key(&expr)
                && self.ty_holds_borrow(self.expr_ty(lhs))
            {
                self.expr_borrow_sources(rhs, state)
            } else {
                Vec::new()
            };
            prune_dead_loans(state, &after_rhs);
            if self.place_behind_shared_ref(lhs) {
                return Err(self.borrow_err(Some(lhs), "cannot assign to data in a `&` reference"));
            }
            self.check_static_mut(lhs, false)?;
            if let Some(p) = self.root_place(lhs) {
                self.check_not_moved(p.local, &p.steps, state)?;
                self.check_can_mutate(
                    p.local,
                    p.through_ref,
                    p.mut_ref_root,
                    p.behind_shared,
                    &p.steps,
                    state,
                    false,
                )?;
                state.set_tuple_sources(p.local, None);
                if !sources.is_empty() {
                    self.propagate_assign_sources(&p, &sources, state)?;
                }
            } else {
                self.check_expr(lhs, state, true, outer_live, false, depth)?;
            }
        }
        Ok(())
    }

    fn check_let_expr(
        &mut self,
        pat: PatId,
        init: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_expr(init, state, true, outer_live, true, depth)?;
        let scrut_place = self.root_place(init);
        self.bind_pat(
            pat,
            Some(init),
            scrut_place.as_ref(),
            &[],
            &[],
            state,
            (true, true, depth),
        )
    }

    fn check_unary_expr(
        &mut self,
        expr: ExprId,
        op: UnOp,
        inner: ExprId,
        state: &mut State,
        place_ctx: bool,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if op == UnOp::Deref
            && self.unsafe_depth == 0
            && self.expr_ty(inner) != Ty::NONE
            && matches!(self.cx.types.kind(self.expr_ty(inner)), Kind::Ptr(..))
        {
            return Err(self.unsafe_err(
                Some(expr),
                "dereference of raw pointer is unsafe and requires unsafe function or block",
            ));
        }
        if op == UnOp::Deref && !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
            if self.typed.derefs.contains_key(&(expr, 0)) || self.place_behind_ref_or_ptr(expr) {
                return Err(self.borrow_err(
                    Some(expr),
                    "cannot move out of a shared or mutable reference",
                ));
            }
            self.check_expr(inner, state, true, outer_live, true, depth)?;
            if let Some(p) = self.root_place(expr) {
                self.check_can_move(p.local, &p.steps, state)?;
                self.push_partial_move(state, p.local, p.steps, Some(expr));
            }
            Ok(())
        } else {
            self.check_expr(inner, state, op == UnOp::Deref, outer_live, true, depth)
        }
    }

    fn check_index_expr(
        &mut self,
        expr: ExprId,
        a: ExprId,
        b: ExprId,
        state: &mut State,
        place_ctx: bool,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
            return Err(self.borrow_err(Some(expr), "cannot move out of index"));
        }
        self.check_expr(a, state, true, outer_live, true, depth)?;
        self.check_expr(b, state, false, outer_live, true, depth)
    }

    fn check_binary_expr(
        &mut self,
        op: BinOp,
        a: ExprId,
        b: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if matches!(op, BinOp::And | BinOp::Or) {
            let mut after_a = outer_live.clone();
            self.collect_live_expr(b, &mut after_a);
            self.check_expr(a, state, false, &after_a, true, depth)?;
            if !state.diverged {
                let before_rhs = state.clone();
                let mut rhs_state = state.clone();
                prune_dead_loans(&mut rhs_state, &after_a);
                self.check_expr(b, &mut rhs_state, false, outer_live, true, depth)?;
                merge_branch(state, &before_rhs, &rhs_state);
            }
            Ok(())
        } else {
            let cmp = matches!(
                op,
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
            );
            self.check_expr(a, state, cmp, outer_live, !cmp, depth)?;
            self.check_expr(b, state, cmp, outer_live, !cmp, depth)
        }
    }

    fn check_repeat_expr(
        &mut self,
        a: ExprId,
        b: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_expr(a, state, false, outer_live, true, depth)?;
        self.check_expr(b, state, false, outer_live, true, depth)
    }

    fn check_field_expr(
        &mut self,
        expr: ExprId,
        base: ExprId,
        state: &mut State,
        place_ctx: bool,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if !place_ctx && matches!(*self.body.expr(expr), Expr::Field(..)) {
            self.check_union_access(expr, base)?;
        }
        let moving = !place_ctx && !self.ty_is_copy(self.expr_ty(expr));
        if let Some(p) = self.root_place(expr)
            && !p.steps.is_empty()
            && !p.steps.iter().any(|s| matches!(s, PlaceStep::Opaque(_)))
        {
            return self.check_field_place(expr, base, p, moving, state);
        }
        if moving {
            if self.place_behind_ref_or_ptr(expr) {
                return Err(self.borrow_err(Some(expr), "cannot move out of a reference"));
            }
            if self.static_root(expr).is_some() {
                return Err(self.borrow_err(Some(expr), "cannot move out of static item"));
            }
            self.check_adt_drop_move(self.field_container_ty(expr, base))?;
        }
        self.check_expr(base, state, true, outer_live, true, depth)
    }

    fn check_field_place(
        &mut self,
        expr: ExprId,
        base: ExprId,
        p: super::places::Place,
        moving: bool,
        state: &mut State,
    ) -> Result<()> {
        if p.steps.last() == Some(&PlaceStep::Union) {
            let prefix = &p.steps[..p.steps.len() - 1];
            if !moving {
                self.check_can_read(p.local, prefix, state)?;
            } else if p.through_ref {
                return Err(self.borrow_err(
                    Some(expr),
                    format!(
                        "cannot move out of `{}` which is behind a reference",
                        self.local_name(p.local)
                    ),
                ));
            } else {
                self.check_can_move(p.local, prefix, state)?;
                if prefix.is_empty() {
                    self.mark_local_moved(state, p.local, Some(expr));
                } else {
                    self.push_partial_move(state, p.local, prefix.to_vec(), Some(expr));
                }
            }
        } else if !moving {
            self.check_can_read(p.local, &p.steps, state)?;
        } else if p.through_ref {
            return Err(self.borrow_err(
                Some(expr),
                format!(
                    "cannot move out of `{}` which is behind a reference",
                    self.local_name(p.local)
                ),
            ));
        } else {
            self.check_adt_drop_move(self.field_container_ty(expr, base))?;
            self.check_can_move(p.local, &p.steps, state)?;
            self.push_partial_move(state, p.local, p.steps, Some(expr));
        }
        Ok(())
    }

    fn check_cast_expr(
        &mut self,
        inner: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let reborrow_mut = self.is_mut_ref_ty(self.expr_ty(inner));
        self.check_expr(inner, state, reborrow_mut, outer_live, true, depth)
    }

    pub(super) fn check_tuple_with_expected_tys(
        &mut self,
        expr: ExprId,
        list: List,
        elem_tys: &[Ty],
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let prev_expr = self.current_expr;
        self.current_expr = Some(expr);
        let ids: Vec<u32> = self.body.list(list).to_vec();
        let res = if ids.len() == elem_tys.len() {
            let refs: Vec<Option<Mut>> = elem_tys.iter().map(|&t| self.ptr_or_ref_mut(t)).collect();
            self.check_arg_list_expected_tys(&ids, &refs, elem_tys, state, outer_live, depth)
        } else {
            self.check_list_expr(list, state, outer_live, depth)
        };
        self.current_expr = prev_expr;
        res?;
        self.finish_expr_divergence(expr, state);
        Ok(())
    }

    pub(super) fn check_let_init(
        &mut self,
        pat: PatId,
        ann_ty: Option<crate::body::TyId>,
        init: ExprId,
        (place_ctx, value_used): (bool, bool),
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let has_ty = ann_ty.is_some_and(|t| !matches!(*self.body.ty(t), crate::body::Ty::Infer));
        if let Some(at) = ann_ty
            && has_ty
            && let Expr::Tuple(list) = *self.body.expr(self.unwrap_parens(init))
            && let Some(elem_tys) = self
                .typed
                .pats
                .get(&pat)
                .copied()
                .filter(|&t| t != Ty::NONE)
                .or_else(|| Some(self.expr_ty(init)))
                .map(|t| self.mask_infer_tuple_ty(at, t))
                .and_then(|t| self.peeled_tuple_elems(t))
        {
            return self
                .check_tuple_with_expected_tys(init, list, &elem_tys, state, outer_live, depth);
        }
        let reborrow_mut = (has_ty || self.typed.coercions.contains_key(&init))
            && self.is_mut_ref_ty(self.expr_ty(init));
        self.check_expr(
            init,
            state,
            place_ctx || reborrow_mut,
            outer_live,
            value_used,
            depth,
        )
    }

    fn check_list_expr(
        &mut self,
        list: List,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        self.check_tuple_or_array_list(list, state, outer_live, depth)
    }

    fn check_struct_expr(
        &mut self,
        expr: ExprId,
        fields: List,
        base: Option<ExprId>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let field_ids: Vec<u32> = self.body.list(fields).to_vec();
        let val_ids: Vec<u32> = field_ids
            .iter()
            .map(|&f| self.body.fields[f as usize].value.0)
            .collect();
        let (field_refs, field_tys) = self.struct_field_ref_muts(expr, &field_ids);
        let mut fields_live = outer_live.clone();
        if let Some(b) = base {
            self.collect_live_expr(b, &mut fields_live);
        }
        self.check_arg_list_expected_tys(
            &val_ids,
            &field_refs,
            &field_tys,
            state,
            &fields_live,
            depth,
        )?;
        self.check_static_struct_fields(expr, &val_ids, state)?;
        if let Some(b) = base {
            self.check_fru_base(expr, fields, b, state, outer_live, depth)?;
        }
        Ok(())
    }

    fn check_static_struct_fields(
        &mut self,
        expr: ExprId,
        val_ids: &[u32],
        state: &State,
    ) -> Result<()> {
        let Some((adt_item, variant, _, def)) = self.struct_adt_variant(expr) else {
            return Ok(());
        };
        let var = &def.variants[variant];
        let adt_src = &self.cx.file(adt_item).source;
        let static_indices: Vec<u32> = var
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                matches!(
                    *def.body.ty(f.ty),
                    crate::body::Ty::Ref {
                        lifetime: Some(lt),
                        ..
                    } if adt_src.text(lt) == "'static"
                )
            })
            .map(|(i, _)| i as u32)
            .collect();
        if static_indices.is_empty() {
            return Ok(());
        }
        for &val_id in val_ids {
            let val = ExprId(val_id);
            if self
                .typed
                .inits
                .get(&val)
                .is_some_and(|idx| static_indices.contains(idx))
            {
                self.check_static_arg(val, state)?;
            }
        }
        Ok(())
    }

    fn check_range_expr(
        &mut self,
        from: Option<ExprId>,
        to: Option<ExprId>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if let Some(f) = from {
            self.check_expr(f, state, false, outer_live, true, depth)?;
        }
        if let Some(t) = to {
            self.check_expr(t, state, false, outer_live, true, depth)?;
        }
        Ok(())
    }

    fn check_return_expr(
        &mut self,
        val: Option<ExprId>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if let Some(v) = val {
            let reborrow_mut = self.return_allows_reborrow(v);
            self.check_expr(v, state, reborrow_mut, outer_live, true, depth)?;
            if self.returns_borrow || self.returns_invariant {
                self.check_return_escape(v, state)?;
            }
        }
        self.check_drop_order(self.return_min_depth, state)?;
        state.diverged = true;
        Ok(())
    }

    fn check_break_expr(
        &mut self,
        label: Option<u32>,
        val: Option<ExprId>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if let Some(v) = val {
            self.check_expr(v, state, false, outer_live, true, depth)?;
        }
        if let Some(idx) = self
            .breaks
            .iter()
            .rposition(|f| jumps_to(self.src, (f.label, f.is_block), label))
        {
            let target_depth = self.breaks[idx].depth;
            self.check_drop_order(target_depth, state)?;
            if let Some(v) = val {
                let srcs = self.expr_borrow_sources(v, state);
                for src in &srcs {
                    if self.escapes_depth(src.target, src.through_ref, src.mutable, target_depth)
                        || (super::checks::is_temp_target(src.target).is_some()
                            && !self.temp_target_allowed(src.mutable))
                    {
                        return Err(self.does_not_live_long_enough(Some(v), src.target));
                    }
                }
                self.breaks[idx].sources.extend(srcs);
            }
            self.breaks[idx].states.push(state.clone());
        }
        state.diverged = true;
        Ok(())
    }

    fn check_yield_expr(
        &mut self,
        val: Option<ExprId>,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        if let Some(v) = val {
            self.check_expr(v, state, false, outer_live, true, depth)?;
        }
        Ok(())
    }
}
