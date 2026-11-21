use crate::body::{BlockKind, Expr, ExprId, List, Stmt, UnOp};
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut, Ty};

use super::Checker;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum FieldKey {
    Named(String),
    Tuple(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PlaceStep {
    Field(FieldKey),
    Deref,
    Opaque,
}

#[derive(Clone, Debug)]
pub(super) struct Place {
    pub(super) local: u32,
    pub(super) through_ref: bool,
    pub(super) mut_ref_root: bool,
    pub(super) reborrows_shared_field: bool,
    pub(super) steps: Vec<PlaceStep>,
}

pub(super) fn steps_conflict(a: &[PlaceStep], b: &[PlaceStep]) -> bool {
    a.iter().zip(b).all(|(x, y)| x == y)
}

pub(super) fn push_step(steps: &mut Vec<PlaceStep>, step: PlaceStep) {
    if steps.last() != Some(&PlaceStep::Opaque) {
        steps.push(step);
    }
}

impl Checker<'_, '_> {
    pub(super) fn local_name(&self, local: u32) -> String {
        self.names.get(local as usize).and_then(|n| n.clone()).unwrap_or_else(|| format!("_{local}"))
    }

    pub(super) fn local_ty(&self, local: u32) -> Ty {
        self.typed.locals.get(local as usize).map_or(Ty::NONE, |l| l.ty)
    }

    pub(super) fn expr_ty(&self, expr: ExprId) -> Ty {
        self.typed.exprs.get(expr.0 as usize).copied().unwrap_or(Ty::NONE)
    }

    pub(super) fn is_ref_ty(&self, ty: Ty) -> bool {
        ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(..))
    }

    pub(super) fn is_mut_ref_ty(&self, ty: Ty) -> bool {
        ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Mut, _))
    }

    pub(super) fn is_shared_ref_or_ptr(&self, ty: Ty) -> bool {
        ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _) | Kind::Ptr(Mut::Not, _))
    }

    pub(super) fn is_mut_ref_or_ptr(&self, ty: Ty) -> bool {
        ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Mut, _) | Kind::Ptr(Mut::Mut, _))
    }

    pub(super) fn ty_is_copy(&mut self, ty: Ty) -> bool {
        if ty == Ty::NONE {
            return true;
        }
        if self.is_ref_ty(ty) {
            return true;
        }
        super::super::infer::is_copy(self.cx, self.item, ty).unwrap_or(true)
    }

    fn deref_ref_step(&self, ty: Ty, p: &mut Place) -> bool {
        if self.is_mut_ref_or_ptr(ty) {
            p.mut_ref_root = true;
            p.through_ref = true;
            true
        } else if self.is_shared_ref_or_ptr(ty) {
            if !p.steps.is_empty() && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _)) {
                p.reborrows_shared_field = true;
            }
            p.mut_ref_root = false;
            p.through_ref = true;
            true
        } else {
            false
        }
    }

    pub(super) fn apply_deref_steps(&self, site: ExprId, base_ty: Ty, count: u32, p: &mut Place) {
        let mut cur = base_ty;
        for step in 0..count {
            if self.typed.derefs.contains_key(&(site, step)) {
                push_step(&mut p.steps, PlaceStep::Opaque);
                if self.is_mut_ref_or_ptr(cur) {
                    p.mut_ref_root = true;
                }
                cur = Ty::NONE;
                continue;
            }
            if cur != Ty::NONE {
                if !self.deref_ref_step(cur, p) && self.cx.boxed(cur).is_some() {
                    p.through_ref = true;
                }
                cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            } else {
                p.through_ref = true;
            }
            push_step(&mut p.steps, PlaceStep::Deref);
        }
    }

    fn builtin_deref_ty(&self, ty: Ty) -> Option<Ty> {
        match *self.cx.types.kind(ty) {
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) => Some(inner),
            _ => self.cx.boxed(ty),
        }
    }

    pub(super) fn root_place(&self, expr: ExprId) -> Option<Place> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.root_place(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Some(Place {
                    local: *local,
                    through_ref: false,
                    mut_ref_root: false,
                    reborrows_shared_field: false,
                    steps: Vec::new(),
                }),
                _ => None,
            },
            Expr::Field(base, name) => {
                let mut p = self.root_place(base)?;
                self.project_field(expr, base, FieldKey::Named(self.src.text(name).to_owned()), &mut p);
                Some(p)
            }
            Expr::TupleField(base, idx) => {
                let mut p = self.root_place(base)?;
                self.project_field(expr, base, FieldKey::Tuple(idx), &mut p);
                Some(p)
            }
            Expr::Index(base, _) => {
                let mut p = self.root_place(base)?;
                let base_ty = self.expr_ty(base);
                self.deref_ref_step(base_ty, &mut p);
                push_step(&mut p.steps, PlaceStep::Opaque);
                Some(p)
            }
            Expr::Unary(UnOp::Deref, inner) => {
                let mut p = self.root_place(inner)?;
                let inner_ty = self.expr_ty(inner);
                if self.typed.derefs.contains_key(&(expr, 0)) {
                    push_step(&mut p.steps, PlaceStep::Opaque);
                } else {
                    self.deref_ref_step(inner_ty, &mut p);
                    p.through_ref = true;
                    push_step(&mut p.steps, PlaceStep::Deref);
                }
                Some(p)
            }
            _ => None,
        }
    }

    fn project_field(&self, expr: ExprId, base: ExprId, key: FieldKey, p: &mut Place) {
        let base_ty = self.expr_ty(base);
        let autoderefs = self.typed.autoderefs.get(&expr).copied().unwrap_or(0);
        if autoderefs > 0 {
            self.apply_deref_steps(expr, base_ty, autoderefs, p);
        } else if self.deref_ref_step(base_ty, p) {
            push_step(&mut p.steps, PlaceStep::Deref);
        }
        push_step(&mut p.steps, PlaceStep::Field(key));
    }

    pub(super) fn place_behind_shared_ref(&self, expr: ExprId) -> bool {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.place_behind_shared_ref(inner),
            Expr::Unary(UnOp::Deref, inner) => {
                let ty = self.expr_ty(inner);
                (ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _)))
                    || self.place_behind_shared_ref(inner)
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => {
                let ty = self.expr_ty(base);
                let base_shared = ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _));
                base_shared || self.place_behind_shared_ref(base)
            }
            _ => false,
        }
    }

    pub(super) fn closure_capture_info(&self, closure_expr: ExprId, by: By, target: u32) -> (Vec<Vec<PlaceStep>>, bool) {
        let mut walk = CapturePathWalk {
            checker: self,
            target,
            paths: Vec::new(),
            whole: false,
            mutates: false,
        };
        if let Expr::Closure { body, .. } = *self.body.expr(closure_expr) {
            walk.walk_expr(body);
        } else {
            walk.walk_expr(closure_expr);
        }
        let paths = if walk.whole || walk.paths.is_empty() {
            vec![Vec::new()]
        } else {
            walk.paths
        };
        (paths, by == By::Mut && walk.mutates)
    }
}

fn first_field_prefix(steps: &[PlaceStep]) -> Option<Vec<PlaceStep>> {
    for (idx, step) in steps.iter().enumerate() {
        match step {
            PlaceStep::Deref => {}
            PlaceStep::Field(_) => return Some(steps[..=idx].to_vec()),
            PlaceStep::Opaque => return None,
        }
    }
    None
}

struct CapturePathWalk<'a, 'c, 'x> {
    checker: &'a Checker<'c, 'x>,
    target: u32,
    paths: Vec<Vec<PlaceStep>>,
    whole: bool,
    mutates: bool,
}

impl CapturePathWalk<'_, '_, '_> {
    fn walk_list(&mut self, list: List) {
        for &id in self.checker.body.list(list) {
            self.walk_expr(ExprId(id));
        }
    }

    fn walk_place_indices(&mut self, expr: ExprId) {
        match *self.checker.body.expr(expr) {
            Expr::Paren(inner) | Expr::Field(inner, _) | Expr::TupleField(inner, _) | Expr::Unary(_, inner) => {
                self.walk_place_indices(inner);
            }
            Expr::Index(base, idx) => {
                self.walk_place_indices(base);
                self.walk_expr(idx);
            }
            _ => {}
        }
    }

    fn check_nested_capture(&mut self, closure_expr: ExprId) -> bool {
        if let Some(c) = self.checker.typed.closures.get(&closure_expr)
            && let Some(cap) = c.captures.iter().find(|cap| cap.local == self.target)
        {
            let is_copy = self.checker.is_copy.get(self.target as usize).copied().unwrap_or(true);
            let is_mut_ref = self.checker.is_mut_ref_ty(self.checker.local_ty(self.target));
            if cap.by == By::Value && (!is_copy || is_mut_ref) {
                self.whole = true;
                self.mutates = true;
                return false;
            }
            return true;
        }
        false
    }

    fn walk_expr(&mut self, expr: ExprId) {
        if self.whole && self.mutates {
            return;
        }
        match *self.checker.body.expr(expr) {
            Expr::Ref { mutable, raw, expr: inner } => {
                if !raw
                    && mutable
                    && self.checker.root_place(inner).is_some_and(|p| p.local == self.target)
                {
                    self.mutates = true;
                }
                self.walk_expr(inner);
            }
            Expr::Paren(inner)
            | Expr::Cast(inner, _)
            | Expr::Try(inner)
            | Expr::Await(inner)
            | Expr::Become(inner) => self.walk_expr(inner),
            Expr::Field(..) | Expr::TupleField(..) | Expr::Index(..) | Expr::Unary(UnOp::Deref, _) => {
                if let Some(p) = self.checker.root_place(expr)
                    && p.local == self.target
                {
                    if let Some(prefix) = first_field_prefix(&p.steps) {
                        if !self.paths.contains(&prefix) {
                            self.paths.push(prefix);
                        }
                        self.walk_place_indices(expr);
                    } else {
                        self.whole = true;
                    }
                    return;
                }
                match *self.checker.body.expr(expr) {
                    Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Unary(_, base) => self.walk_expr(base),
                    Expr::Index(base, idx) => {
                        self.walk_expr(base);
                        self.walk_expr(idx);
                    }
                    _ => {}
                }
            }
            Expr::Unary(_, inner) => self.walk_expr(inner),
            Expr::Path(_) => {
                if self.checker.typed.res.get(&expr) == Some(&Res::Local(self.target)) {
                    self.whole = true;
                }
            }
            Expr::Assign(lhs, rhs) | Expr::AssignOp(_, lhs, rhs) => {
                if self.checker.root_place(lhs).is_some_and(|p| p.local == self.target) {
                    self.mutates = true;
                }
                self.walk_expr(lhs);
                self.walk_expr(rhs);
            }
            Expr::Binary(_, a, b) | Expr::Repeat(a, b) => {
                self.walk_expr(a);
                self.walk_expr(b);
            }
            Expr::Call(callee, args) => {
                self.walk_expr(callee);
                self.walk_list(args);
            }
            Expr::Method { receiver, params, .. } => {
                if self.checker.root_place(receiver).is_some_and(|p| p.local == self.target) {
                    let autoref = self.checker.typed.adjusts.get(&expr).and_then(|a| a.autoref);
                    if autoref == Some(Mut::Mut)
                        || (autoref.is_none() && self.checker.is_mut_ref_ty(self.checker.expr_ty(receiver)))
                    {
                        self.mutates = true;
                    }
                }
                self.walk_expr(receiver);
                self.walk_list(params);
            }
            Expr::Block { kind, stmts, tail, .. } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) && !self.check_nested_capture(expr) {
                    return;
                }
                for &id in self.checker.body.list(stmts) {
                    match self.checker.body.stmts[id as usize] {
                        Stmt::Let { init, otherwise, .. } => {
                            if let Some(e) = init {
                                self.walk_expr(e);
                            }
                            if let Some(e) = otherwise {
                                self.walk_expr(e);
                            }
                        }
                        Stmt::Expr(e, _) => self.walk_expr(e),
                        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
                    }
                }
                if let Some(t) = tail {
                    self.walk_expr(t);
                }
            }
            Expr::If(c, t, e) => {
                self.walk_expr(c);
                self.walk_expr(t);
                if let Some(alt) = e {
                    self.walk_expr(alt);
                }
            }
            Expr::Let(_, init) => self.walk_expr(init),
            Expr::While { cond, body, .. } => {
                self.walk_expr(cond);
                self.walk_expr(body);
            }
            Expr::Loop { body, .. } => self.walk_expr(body),
            Expr::For { iter, body, .. } => {
                self.walk_expr(iter);
                self.walk_expr(body);
            }
            Expr::Match(scrut, arms) => {
                self.walk_expr(scrut);
                for &id in self.checker.body.list(arms) {
                    let arm = self.checker.body.arms[id as usize];
                    if let Some(g) = arm.guard {
                        self.walk_expr(g);
                    }
                    self.walk_expr(arm.body);
                }
            }
            Expr::Closure { body, .. } => {
                if self.check_nested_capture(expr) {
                    self.walk_expr(body);
                }
            }
            Expr::Tuple(list) | Expr::Array(list) => self.walk_list(list),
            Expr::Struct { fields, base, .. } => {
                for &id in self.checker.body.list(fields) {
                    self.walk_expr(self.checker.body.fields[id as usize].value);
                }
                if let Some(b) = base {
                    self.walk_expr(b);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.walk_expr(f);
                }
                if let Some(t) = to {
                    self.walk_expr(t);
                }
            }
            Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => {
                if let Some(e) = e {
                    self.walk_expr(e);
                }
            }
            Expr::Lit(..)
            | Expr::Continue(_)
            | Expr::Macro { .. }
            | Expr::Underscore
            | Expr::FormatArgs(_)
            | Expr::OffsetOf { .. } => {}
        }
    }
}
