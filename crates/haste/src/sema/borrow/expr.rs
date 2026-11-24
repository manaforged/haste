use anyhow::{Result, bail};
use rustc_hash::FxHashSet as HashSet;

use crate::body::{BinOp, BlockKind, Expr, ExprId, List, PatId, UnOp, jumps_to};
use crate::outline::item::ItemKind;

use super::super::infer::Res;
use super::super::ty::{Kind, Ty};
use super::places::PlaceStep;
use super::{Checker, PartialMove, State, merge_branch, prune_dead_loans, prune_moved_loans};

impl Checker<'_, '_> {
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
        self.check_expr_kind(expr, state, place_ctx, outer_live, value_used, depth)?;
        let expr_ty = self.expr_ty(expr);
        if expr_ty != Ty::NONE && matches!(*self.cx.types.kind(expr_ty), Kind::Never) {
            state.diverged = true;
        }
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
            Expr::Loop { label, body } => {
                self.check_loop(label, None, None, None, body, state, outer_live, depth)
            }
            Expr::While { label, cond, body } => self.check_loop(
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
            Expr::Continue(_) => {
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
                state.moved[*local as usize] = true;
                state.partial_moves.retain(|m| m.local != *local);
            }
        } else if let Some(Res::Fn(item, _)) = self.typed.res.get(&expr)
            && self.cx.item(*item).1.kind == ItemKind::Static
        {
            if self.unsafe_depth == 0 && self.is_static_mut(*item) {
                bail!(
                    "use of mutable static `{}` is unsafe and requires unsafe function or block",
                    self.cx.name(*item)
                );
            }
            if !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
                bail!("cannot move out of static item `{}`", self.cx.name(*item));
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
            bail!("cannot borrow data in a `&` reference as mutable");
        }
        if mutable {
            self.check_static_mut(inner, false)?;
        }
        if let Some(p) = self.root_place(inner) {
            if mutable && p.behind_shared {
                bail!("cannot borrow data in a `&` reference or `*const` pointer as mutable");
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
        }
        self.check_expr(inner, state, true, outer_live, false, depth)
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
        let reborrow_mut =
            self.is_mut_ref_ty(self.expr_ty(lhs)) && self.is_mut_ref_ty(self.expr_ty(rhs));
        self.check_expr(rhs, state, reborrow_mut, &after_rhs, true, depth)?;
        if !state.diverged {
            self.check_mut_ref_reborrow(lhs, rhs, state)?;
            let sources = self.expr_borrow_sources(rhs, state);
            prune_dead_loans(state, &after_rhs);
            self.assign_place(lhs, &sources, state, outer_live, depth)?;
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
                bail!("cannot assign to data in a `&` reference");
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
            bail!("dereference of raw pointer is unsafe and requires unsafe function or block");
        }
        if op == UnOp::Deref && !place_ctx && !self.ty_is_copy(self.expr_ty(expr)) {
            if self.typed.derefs.contains_key(&(expr, 0)) || self.place_behind_ref_or_ptr(expr) {
                bail!("cannot move out of a shared or mutable reference");
            }
            self.check_expr(inner, state, true, outer_live, false, depth)?;
            if let Some(p) = self.root_place(expr) {
                self.check_can_move(p.local, &p.steps, state)?;
                state.partial_moves.push(PartialMove {
                    local: p.local,
                    steps: p.steps,
                });
            }
            Ok(())
        } else {
            self.check_expr(inner, state, op == UnOp::Deref, outer_live, false, depth)
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
            bail!("cannot move out of index");
        }
        self.check_expr(a, state, true, outer_live, false, depth)?;
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
                bail!("cannot move out of a reference");
            }
            if self.static_root(expr).is_some() {
                bail!("cannot move out of static item");
            }
            self.check_adt_drop_move(self.field_container_ty(expr, base))?;
        }
        self.check_expr(base, state, true, outer_live, false, depth)
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
                bail!(
                    "cannot move out of `{}` which is behind a reference",
                    self.local_name(p.local)
                );
            } else {
                self.check_can_move(p.local, prefix, state)?;
                if prefix.is_empty() {
                    state.moved[p.local as usize] = true;
                    state.partial_moves.retain(|m| m.local != p.local);
                } else {
                    self.push_partial_move(state, p.local, prefix.to_vec());
                }
            }
        } else if !moving {
            self.check_can_read(p.local, &p.steps, state)?;
        } else if p.through_ref {
            bail!(
                "cannot move out of `{}` which is behind a reference",
                self.local_name(p.local)
            );
        } else {
            self.check_adt_drop_move(self.field_container_ty(expr, base))?;
            self.check_can_move(p.local, &p.steps, state)?;
            self.push_partial_move(state, p.local, p.steps);
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

    fn check_list_expr(
        &mut self,
        list: List,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let ids: Vec<u32> = self.body.list(list).to_vec();
        self.check_arg_list(&ids, None, None, false, state, outer_live, depth)
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
        let val_ids: Vec<u32> = self
            .body
            .list(fields)
            .iter()
            .map(|&f| self.body.fields[f as usize].value.0)
            .collect();
        self.check_arg_list(&val_ids, None, None, false, state, outer_live, depth)?;
        if let Some(b) = base {
            self.check_fru_base(expr, fields, b, state, outer_live, depth)?;
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
            let reborrow_mut = self.is_mut_ref_ty(self.expr_ty(v))
                && self.typed.ret.is_some_and(|r| self.is_mut_ref_ty(r));
            self.check_expr(v, state, reborrow_mut, outer_live, true, depth)?;
            if self.returns_borrow {
                self.check_return_escape(v, state)?;
            }
        }
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
            if let Some(v) = val {
                for src in self.expr_borrow_sources(v, state) {
                    if self.escapes_depth(src.target, src.through_ref, target_depth) {
                        bail!(
                            "`{}` does not live long enough",
                            self.local_name(src.target)
                        );
                    }
                }
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
