mod args;
pub(super) mod closure;
mod types;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::body::{Expr, ExprId, List, UnOp};
use crate::outline::item::ItemKind;
use crate::sema::infer::Res;
use crate::sema::ty::{Kind, Ty};

use super::places::{FieldKey, Place, PlaceStep, push_step};
use super::propagate::BorrowSource;
use super::{Checker, PartialMove, State, loan_conflicts_borrow};

pub(super) const TEMP_TARGET_BASE: u32 = u32::MAX - 262_144;

pub(super) use closure::ClosureEffects;

pub(super) fn is_temp_target(target: u32) -> Option<u32> {
    (target >= TEMP_TARGET_BASE && target < super::ITER_HOLDER_BASE)
        .then(|| target - TEMP_TARGET_BASE)
}

pub(super) fn temp_target_for(depth: u32) -> u32 {
    TEMP_TARGET_BASE + depth.min(65_000)
}

impl Checker<'_, '_> {
    pub(super) fn check_stmt_temp_escape(
        &self,
        state: &State,
        min_depth: u32,
        check_locals: &[u32],
        outer_live: &HashSet<u32>,
    ) -> Result<()> {
        for l in &state.loans {
            let Some(holder) = l.holder else { continue };
            let Some(t_depth) = is_temp_target(l.target) else {
                continue;
            };
            if t_depth >= min_depth
                && (check_locals.contains(&holder) || outer_live.contains(&holder))
            {
                return Err(self.borrow_err(l.site, "temporary value dropped while borrowed"));
            }
        }
        Ok(())
    }

    pub(super) fn check_assign_escape(
        &mut self,
        p: &Place,
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        if sources.is_empty() {
            return Ok(());
        }
        let mut targets: Vec<u32> = if p.through_ref {
            state
                .loans
                .iter()
                .filter(|l| l.holder == Some(p.local) && l.mutable)
                .map(|l| l.target)
                .collect()
        } else {
            Vec::new()
        };
        if p.through_ref
            && (!self.is_param_local(p.local) || self.closure_depth > 0)
            && !targets.contains(&p.local)
        {
            targets.push(p.local);
        }
        if (p.through_ref || p.through_deref)
            && (self.is_param_local(p.local) || targets.iter().any(|&t| self.is_param_local(t)))
        {
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
        if self.closure_depth > 0
            && self.scope_depth.get(p.local as usize).copied().unwrap_or(0) < self.closure_depth
        {
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
        if p.through_ref {
            for holder in targets {
                state.set_tuple_sources(holder, None);
                if self.ty_holds_borrow(self.local_ty(holder)) {
                    super::bind::push_source_loans_filtered(
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
        }
        Ok(())
    }

    pub(super) fn propagate_assign_sources(
        &mut self,
        p: &Place,
        sources: &[BorrowSource],
        state: &mut State,
    ) -> Result<()> {
        if !sources.is_empty() {
            self.check_assign_escape(p, sources, state)?;
            if !p.through_ref {
                self.push_sources_for_local(p.local, sources, self.current_expr, state);
            }
        }
        Ok(())
    }

    pub(super) fn check_mut_ref_local_move(
        &self,
        dst_local: u32,
        src: ExprId,
        state: &State,
    ) -> Result<()> {
        if let Some(src_local) = self.expr_plain_local(src)
            && src_local != dst_local
            && self.is_mut_ref_ty(self.local_ty(dst_local))
            && self.is_mut_ref_ty(self.local_ty(src_local))
            && let Some(loan) = state
                .loans
                .iter()
                .find(|l| loan_conflicts_borrow(l, src_local, &[]))
        {
            return Err(self.borrow_err_labeled(
                Some(src),
                format!(
                    "cannot move out of `{}` because it is borrowed",
                    self.local_name(src_local)
                ),
                "move out of borrowed value occurs here",
                loan.site,
                "borrow occurs here",
            ));
        }
        Ok(())
    }

    pub(super) fn check_mut_ref_reborrow(
        &self,
        dst: ExprId,
        src: ExprId,
        state: &State,
    ) -> Result<()> {
        if let Some(dst_local) = self.expr_plain_local(dst) {
            self.check_mut_ref_local_move(dst_local, src, state)?;
        }
        Ok(())
    }

    pub(super) fn check_adt_drop_move(&mut self, ty: Ty) -> Result<()> {
        if self.ty_has_own_drop(ty) {
            return Err(self.borrow_err(
                None,
                "cannot move out of type which implements the `Drop` trait",
            ));
        }
        Ok(())
    }

    pub(super) fn check_fru_base(
        &mut self,
        expr: ExprId,
        fields: List,
        b: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        let deref_base = matches!(*self.body.expr(b), Expr::Unary(UnOp::Deref, _));
        let Some(p) = self.root_place(b) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let Some((adt_item, variant, args)) = self.struct_adt(expr) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let args_vec = self.cx.types.substs(args).to_vec();
        let Ok(field_tys) = super::super::infer::detached(self.cx, self.item, |infer| {
            infer.adt_fields(adt_item, &args_vec)
        }) else {
            return self.check_expr(b, state, deref_base, outer_live, true, depth);
        };
        let pending_fields: Vec<(FieldKey, Ty)> = {
            let Ok(def) = self.cx.adt_def(adt_item) else {
                return self.check_expr(b, state, deref_base, outer_live, true, depth);
            };
            let Some(var) = def.variants.get(variant) else {
                return self.check_expr(b, state, deref_base, outer_live, true, depth);
            };
            let overridden: HashSet<&str> = self
                .body
                .list(fields)
                .iter()
                .map(|&id| self.src.text(self.body.fields[id as usize].name))
                .collect();
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
                    let fty = field_tys
                        .get(variant)
                        .and_then(|fs| fs.get(idx))
                        .copied()
                        .unwrap_or(Ty::NONE);
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
                    return Err(self
                        .borrow_err(Some(b), "cannot move out of a shared or mutable reference"));
                }
                self.check_can_move(p.local, &steps, state)?;
                if !p.through_ref {
                    self.push_partial_move(state, p.local, steps, Some(b));
                }
            }
        }
        Ok(())
    }

    fn struct_adt(
        &self,
        expr: ExprId,
    ) -> Option<(crate::names::ItemRef, usize, super::super::substs::Substs)> {
        match self.typed.res.get(&expr).copied() {
            Some(Res::Ctor(item, v, args)) => Some((item, v as usize, args)),
            _ => match *self.cx.types.kind(self.expr_ty(expr)) {
                Kind::Adt(item, args) => Some((item, 0, args)),
                _ => None,
            },
        }
    }

    pub(super) fn check_place_subexprs(
        &mut self,
        expr: ExprId,
        state: &mut State,
        outer_live: &HashSet<u32>,
        depth: u32,
    ) -> Result<()> {
        match *self.body.expr(expr) {
            Expr::Paren(inner)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Unary(_, inner) => self.check_place_subexprs(inner, state, outer_live, depth),
            Expr::Index(base, idx) => {
                self.check_place_subexprs(base, state, outer_live, depth)?;
                self.check_expr(idx, state, false, outer_live, true, depth)
            }
            _ => Ok(()),
        }
    }

    pub(super) fn has_new_moves(
        before_moved: &[bool],
        before_partial: &[PartialMove],
        after: &State,
        locals: &[u32],
    ) -> bool {
        locals.iter().any(|&l| {
            let idx = l as usize;
            (after.moved.get(idx).copied().unwrap_or(false)
                && !before_moved.get(idx).copied().unwrap_or(false))
                || after
                    .partial_moves
                    .iter()
                    .any(|m| m.local == l && !before_partial.contains(m))
        })
    }

    pub(super) fn check_union_access(&self, expr: ExprId, base: ExprId) -> Result<()> {
        if self.unsafe_depth == 0 {
            let cont = self.field_container_ty(expr, base);
            if cont != Ty::NONE
                && matches!(self.cx.types.kind(cont), Kind::Adt(item, _) if self.cx.item(*item).1.kind == ItemKind::Union)
            {
                return Err(self.unsafe_err(
                    Some(expr),
                    "access to union field is unsafe and requires unsafe function or block",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn check_union_place(&self, mut expr: ExprId) -> Result<()> {
        if self.unsafe_depth > 0 {
            return Ok(());
        }
        loop {
            match *self.body.expr(expr) {
                Expr::Paren(inner) => expr = inner,
                Expr::Field(base, _) => {
                    self.check_union_access(expr, base)?;
                    expr = base;
                }
                Expr::TupleField(base, _) | Expr::Index(base, _) => expr = base,
                _ => return Ok(()),
            }
        }
    }

    pub(super) fn check_coercion_vtables(&self) -> Result<()> {
        for (&coercion_expr, coercion) in &self.typed.coercions {
            let Some(vtable) = coercion.vtable() else {
                continue;
            };
            let Kind::Closure(parent, _, at) = *self.cx.types.kind(vtable.ty) else {
                continue;
            };
            if parent != self.item {
                continue;
            }
            let Kind::Dyn(of, ..) = *self.cx.types.kind(vtable.object) else {
                continue;
            };
            let Some(eff) = self.closure_effects.get(&ExprId(at)) else {
                continue;
            };
            let lang = self.cx.lang(of);
            self.check_closure_effects(
                Some(coercion_expr),
                eff,
                lang == Some("fn"),
                lang == Some("fn_mut"),
                false,
            )?;
        }
        Ok(())
    }
}
