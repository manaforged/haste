use crate::body::{BlockKind, Expr, ExprId, List, Pat, PatId, Stmt, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
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
    SliceElem { offset: u64, min_len: u64, from_end: bool, array_len: Option<u64> },
    SliceSub { from: u64, to: u64, array_len: Option<u64> },
    Deref,
    Opaque,
}

#[derive(Clone, Debug)]
pub(super) struct Place {
    pub(super) local: u32,
    pub(super) through_ref: bool,
    pub(super) mut_ref_root: bool,
    pub(super) behind_shared: bool,
    pub(super) reborrows_shared_field: bool,
    pub(super) raw_ptr_prefix: Option<usize>,
    pub(super) steps: Vec<PlaceStep>,
}

pub(super) fn steps_conflict(a: &[PlaceStep], b: &[PlaceStep]) -> bool {
    a.iter().zip(b).all(|(x, y)| step_conflict(x, y))
}

fn step_conflict(a: &PlaceStep, b: &PlaceStep) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (PlaceStep::Opaque, PlaceStep::SliceElem { .. } | PlaceStep::SliceSub { .. })
        | (PlaceStep::SliceElem { .. } | PlaceStep::SliceSub { .. }, PlaceStep::Opaque) => true,
        (
            PlaceStep::SliceElem { .. } | PlaceStep::SliceSub { .. },
            PlaceStep::SliceElem { .. } | PlaceStep::SliceSub { .. },
        ) => slice_steps_conflict(a, b),
        _ => false,
    }
}

fn slice_array_len(step: &PlaceStep) -> Option<u64> {
    match *step {
        PlaceStep::SliceElem { array_len, .. } | PlaceStep::SliceSub { array_len, .. } => array_len,
        _ => None,
    }
}

fn slice_range(step: &PlaceStep, len: u64) -> (u64, u64) {
    match *step {
        PlaceStep::SliceElem { offset, from_end: false, .. } => {
            let start = offset.min(len);
            (start, (start + 1).min(len))
        }
        PlaceStep::SliceElem { offset, from_end: true, .. } => {
            let start = len.saturating_sub(offset);
            (start, (start + 1).min(len))
        }
        PlaceStep::SliceSub { from, to, .. } => (from.min(len), len.saturating_sub(to)),
        _ => (0, len),
    }
}

fn slice_steps_conflict(a: &PlaceStep, b: &PlaceStep) -> bool {
    if let Some(len) = slice_array_len(a).or_else(|| slice_array_len(b)) {
        let (a1, a2) = slice_range(a, len);
        let (b1, b2) = slice_range(b, len);
        return a1 < b2 && b1 < a2;
    }
    match (a, b) {
        (
            PlaceStep::SliceElem { offset: o1, min_len: m1, from_end: e1, .. },
            PlaceStep::SliceElem { offset: o2, min_len: m2, from_end: e2, .. },
        ) => {
            if e1 == e2 {
                o1 == o2
            } else {
                o1 + o2 >= (*m1).max(*m2)
            }
        }
        (PlaceStep::SliceElem { offset, from_end, .. }, PlaceStep::SliceSub { from, to, .. })
        | (PlaceStep::SliceSub { from, to, .. }, PlaceStep::SliceElem { offset, from_end, .. }) => {
            if *from_end {
                offset > to
            } else {
                offset >= from
            }
        }
        (PlaceStep::SliceSub { .. }, PlaceStep::SliceSub { .. }) => true,
        _ => true,
    }
}

pub(super) fn push_step(steps: &mut Vec<PlaceStep>, step: PlaceStep) {
    if steps.last() != Some(&PlaceStep::Opaque) {
        steps.push(step);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaceBehind {
    SharedRef,
    AnyRef,
    AnyRefOrPtr,
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

    fn update_ref_flags(&self, ty: Ty, mut_ref_root: &mut bool, behind_shared: &mut bool) {
        if self.is_mut_ref_or_ptr(ty) {
            if matches!(self.cx.types.kind(ty), Kind::Ptr(Mut::Mut, _)) {
                *behind_shared = false;
            }
            *mut_ref_root = !*behind_shared;
        } else if self.is_shared_ref_or_ptr(ty) {
            *behind_shared = true;
            *mut_ref_root = false;
        }
    }

    pub(super) fn deref_ref_step(&self, ty: Ty, p: &mut Place) -> bool {
        if ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ptr(..)) && p.raw_ptr_prefix.is_none() {
            p.raw_ptr_prefix = Some(p.steps.len());
        }
        if self.is_mut_ref_or_ptr(ty) {
            self.update_ref_flags(ty, &mut p.mut_ref_root, &mut p.behind_shared);
            p.through_ref = true;
            true
        } else if self.is_shared_ref_or_ptr(ty) {
            if !p.steps.is_empty() && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _)) {
                p.reborrows_shared_field = true;
            }
            self.update_ref_flags(ty, &mut p.mut_ref_root, &mut p.behind_shared);
            p.through_ref = true;
            true
        } else {
            false
        }
    }

    pub(super) fn apply_deref_steps(&self, site: ExprId, base_ty: Ty, count: u32, p: &mut Place) -> Ty {
        let mut cur = base_ty;
        for step in 0..count {
            if self.typed.derefs.contains_key(&(site, step)) {
                push_step(&mut p.steps, PlaceStep::Opaque);
                if !p.behind_shared && self.is_mut_ref_or_ptr(cur) {
                    p.mut_ref_root = true;
                }
                p.through_ref = true;
                cur = Ty::NONE;
                continue;
            }
            if cur != Ty::NONE {
                self.deref_ref_step(cur, p);
                cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            } else {
                p.mut_ref_root = !p.behind_shared;
                p.through_ref = true;
            }
            push_step(&mut p.steps, PlaceStep::Deref);
        }
        cur
    }

    pub(super) fn builtin_deref_ty(&self, ty: Ty) -> Option<Ty> {
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
                    local: *local, through_ref: false, mut_ref_root: false, behind_shared: false,
                    reborrows_shared_field: false, raw_ptr_prefix: None, steps: Vec::new(),
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
                    p.through_ref = true;
                } else {
                    let was_ref = self.deref_ref_step(inner_ty, &mut p);
                    if !was_ref && inner_ty == Ty::NONE {
                        p.through_ref = true;
                        p.mut_ref_root = !p.behind_shared;
                    }
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
        let cur = if autoderefs > 0 {
            self.apply_deref_steps(expr, base_ty, autoderefs, p)
        } else if self.deref_ref_step(base_ty, p) {
            push_step(&mut p.steps, PlaceStep::Deref);
            self.builtin_deref_ty(base_ty).unwrap_or(Ty::NONE)
        } else {
            base_ty
        };
        let is_union = cur != Ty::NONE
            && matches!(self.cx.types.kind(cur), Kind::Adt(item, _) if self.cx.item(*item).1.kind == ItemKind::Union);
        push_step(&mut p.steps, if is_union { PlaceStep::Opaque } else { PlaceStep::Field(key) });
    }

    pub(super) fn is_static_mut(&self, item: ItemRef) -> bool {
        let (src, entry) = self.cx.item(item);
        (entry.start..entry.name.unwrap_or(entry.start)).any(|t| src.text(t) == "mut")
    }

    pub(super) fn static_root(&self, expr: ExprId) -> Option<(ItemRef, bool, bool)> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.static_root(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Fn(item, _)) if self.cx.item(*item).1.kind == ItemKind::Static => Some((*item, false, false)),
                _ => None,
            },
            Expr::Field(base, _) | Expr::TupleField(base, _) => {
                let (item, mut mr, mut bs) = self.static_root(base)?;
                let mut cur = self.expr_ty(base);
                let autoderefs = self.typed.autoderefs.get(&expr).copied().unwrap_or(0);
                let steps = if autoderefs > 0 { autoderefs } else { u32::from(self.is_ref_ty(cur) || matches!(self.cx.types.kind(cur), Kind::Ptr(..))) };
                for s in 0..steps {
                    if self.typed.derefs.contains_key(&(expr, s)) {
                        cur = Ty::NONE;
                        continue;
                    }
                    self.update_ref_flags(cur, &mut mr, &mut bs);
                    cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
                }
                Some((item, mr, bs))
            }
            Expr::Index(base, _) => {
                let (item, mut mr, mut bs) = self.static_root(base)?;
                self.update_ref_flags(self.expr_ty(base), &mut mr, &mut bs);
                Some((item, mr, bs))
            }
            Expr::Unary(UnOp::Deref, inner) => {
                let (item, mut mr, mut bs) = self.static_root(inner)?;
                if !self.typed.derefs.contains_key(&(expr, 0)) {
                    self.update_ref_flags(self.expr_ty(inner), &mut mr, &mut bs);
                }
                Some((item, mr, bs))
            }
            _ => None,
        }
    }

    pub(super) fn ty_has_own_drop(&mut self, ty: Ty) -> bool {
        ty != Ty::NONE
            && matches!(self.cx.types.kind(ty), Kind::Adt(..))
            && self.cx.boxed(ty).is_none()
            && super::super::infer::has_drop_impl(self.cx, ty).unwrap_or(false)
    }

    pub(super) fn field_container_ty(&self, expr: ExprId, base: ExprId) -> Ty {
        let mut cur = self.expr_ty(base);
        let autoderefs = self.typed.autoderefs.get(&expr).copied().unwrap_or(0);
        for s in 0..autoderefs {
            if self.typed.derefs.contains_key(&(expr, s)) {
                return Ty::NONE;
            }
            cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
        }
        cur
    }

    pub(super) fn place_behind_ref_or_ptr(&self, expr: ExprId) -> bool {
        self.place_behind(expr, PlaceBehind::AnyRefOrPtr)
    }

    pub(super) fn place_behind_shared_ref(&self, expr: ExprId) -> bool {
        self.place_behind(expr, PlaceBehind::SharedRef)
    }

    pub(super) fn place_behind_ref(&self, expr: ExprId, shared_only: bool) -> bool {
        self.place_behind(expr, if shared_only { PlaceBehind::SharedRef } else { PlaceBehind::AnyRef })
    }

    fn place_behind(&self, expr: ExprId, mode: PlaceBehind) -> bool {
        let matches_ty = |ty: Ty| {
            ty != Ty::NONE
                && match mode {
                    PlaceBehind::SharedRef => matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _)),
                    PlaceBehind::AnyRef => matches!(self.cx.types.kind(ty), Kind::Ref(..)),
                    PlaceBehind::AnyRefOrPtr => matches!(self.cx.types.kind(ty), Kind::Ref(..) | Kind::Ptr(..)),
                }
        };
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.place_behind(inner, mode),
            Expr::Unary(UnOp::Deref, inner) => {
                let ty = self.expr_ty(inner);
                if mode != PlaceBehind::AnyRefOrPtr && ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ptr(..)) {
                    return false;
                }
                (mode == PlaceBehind::AnyRefOrPtr && self.typed.derefs.contains_key(&(expr, 0)))
                    || matches_ty(ty)
                    || self.place_behind(inner, mode)
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => {
                let ty = self.expr_ty(base);
                if mode != PlaceBehind::AnyRefOrPtr && ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ptr(..)) {
                    return false;
                }
                if mode == PlaceBehind::AnyRefOrPtr {
                    let mut cur = ty;
                    for s in 0..self.typed.autoderefs.get(&expr).copied().unwrap_or(0) {
                        if self.typed.derefs.contains_key(&(expr, s)) || matches_ty(cur) {
                            return true;
                        }
                        cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
                    }
                    if matches_ty(cur) {
                        return true;
                    }
                }
                matches_ty(ty) || self.place_behind(base, mode)
            }
            _ => false,
        }
    }

    pub(super) fn slice_pat_steps(&self, pat: PatId, list: List) -> Vec<(PatId, PlaceStep)> {
        let pat_ids = self.body.list(list);
        let rest_pos = pat_ids.iter().position(|&id| self.is_rest_pat(PatId(id)));
        let min_len = (pat_ids.len() - usize::from(rest_pos.is_some())) as u64;
        let array_len = self.typed.pats.get(&pat).and_then(|&ty| match *self.cx.types.kind(ty) {
            Kind::Array(_, len) => self.cx.types.len(len),
            _ => None,
        });
        pat_ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| {
                let step = match rest_pos {
                    Some(r) if idx == r => PlaceStep::SliceSub { from: r as u64, to: (pat_ids.len() - r - 1) as u64, array_len },
                    Some(r) if idx > r => PlaceStep::SliceElem { offset: (pat_ids.len() - idx) as u64, min_len, from_end: true, array_len },
                    _ => PlaceStep::SliceElem { offset: idx as u64, min_len, from_end: false, array_len },
                };
                (PatId(id), step)
            })
            .collect()
    }

    pub(super) fn closure_capture_info(&mut self, closure_expr: ExprId, by: By, target: u32) -> Vec<(Vec<PlaceStep>, bool, bool, bool)> {
        let root_expr = match *self.body.expr(closure_expr) { Expr::Closure { body, .. } => body, _ => closure_expr };
        let mut walk = CapturePathWalk { checker: self, target, paths: Vec::new() };
        walk.walk_expr(root_expr);
        if walk.paths.is_empty() { return vec![(Vec::new(), by == By::Mut, false, false)]; }
        if by == By::Ref {
            for entry in &mut walk.paths {
                if !entry.3 { entry.1 = false; }
            }
        }
        walk.paths
    }
}

fn capture_prefix(p: &Place) -> Vec<PlaceStep> {
    let limit = p.raw_ptr_prefix.unwrap_or(p.steps.len());
    p.steps[..limit].iter().take_while(|s| !matches!(s, PlaceStep::Opaque)).cloned().collect()
}

struct CapturePathWalk<'a, 'c, 'x> {
    checker: &'a mut Checker<'c, 'x>,
    target: u32,
    paths: Vec<(Vec<PlaceStep>, bool, bool, bool)>,
}

impl CapturePathWalk<'_, '_, '_> {
    fn record_path(&mut self, steps: Vec<PlaceStep>, mutated: bool, through_ref: bool, mut_ref_root: bool) {
        if let Some(existing) = self.paths.iter_mut().find(|(s, ..)| *s == steps) {
            existing.1 |= mutated;
            existing.2 |= through_ref;
            existing.3 |= mut_ref_root;
        } else {
            self.paths.push((steps, mutated, through_ref, mut_ref_root));
        }
    }

    fn record_place(&mut self, p: &Place, mutated: bool) {
        let (through_ref, mut_ref_root) = if p.raw_ptr_prefix.is_some() {
            (false, false)
        } else {
            (p.through_ref, p.mut_ref_root && !p.reborrows_shared_field)
        };
        self.record_path(capture_prefix(p), mutated && p.raw_ptr_prefix.is_none(), through_ref, mut_ref_root);
    }

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
        if let Some(c) = self.checker.typed.closures.get(&closure_expr) {
            let caps: Vec<_> = c.captures.iter().filter(|cap| cap.local == self.target).cloned().collect();
            if !caps.is_empty() {
                let mut walk_body = false;
                for cap in caps {
                    let cap_ty = cap.ty(self.checker.cx, self.checker.local_ty(self.target)).unwrap_or_else(|_| self.checker.local_ty(self.target));
                    let is_copy = self.checker.ty_is_copy(cap_ty);
                    let is_mut_ref = self.checker.is_mut_ref_ty(cap_ty);
                    let mut steps: Vec<PlaceStep> = cap.path.iter().map(|&(_, f)| PlaceStep::Field(FieldKey::Tuple(f))).collect();
                    if cap.by == By::Value && is_mut_ref {
                        push_step(&mut steps, PlaceStep::Deref);
                        self.record_path(steps, true, true, true);
                    } else if cap.by == By::Value && !is_copy {
                        self.record_path(steps, true, false, false);
                    } else if cap.path.is_empty() || (cap.by == By::Mut && is_mut_ref) {
                        walk_body = true;
                    } else {
                        self.record_path(steps, cap.by == By::Mut, false, false);
                    }
                }
                return walk_body;
            }
        }
        false
    }

    fn pat_res_inspects(&mut self, pat: PatId) -> bool {
        match self.checker.typed.pat_res.get(&pat).copied() {
            Some(Res::Ctor(item, _, _)) => self.checker.cx.adt_def(item).is_ok_and(|def| def.variants.len() > 1),
            Some(Res::Fn(..)) => true,
            _ => false,
        }
    }

    fn walk_scrutinee_pat(&mut self, scrut: ExprId, pats: &[PatId]) {
        if let Some(p) = self.checker.root_place(scrut)
            && p.local == self.target
        {
            self.walk_place_indices(scrut);
            if p.raw_ptr_prefix.is_some() {
                self.record_place(&p, false);
                return;
            }
            let prefix = capture_prefix(&p);
            let mut_ref_root = p.mut_ref_root && !p.reborrows_shared_field;
            for &pat in pats {
                self.walk_pat(pat, &prefix, (p.through_ref, mut_ref_root, p.reborrows_shared_field));
            }
        } else {
            self.walk_expr(scrut);
        }
    }

    fn walk_pat(&mut self, pat: PatId, steps: &[PlaceStep], (mut through_ref, mut mut_ref_root, mut shared): (bool, bool, bool)) {
        let derefs = self.checker.typed.pat_derefs(pat);
        let mut cur_steps = steps.to_vec();
        if derefs > 0 {
            through_ref = true;
            if !shared && self.pat_has_mut_ref_bind(pat) {
                mut_ref_root = true;
            }
            for _ in 0..derefs {
                push_step(&mut cur_steps, PlaceStep::Deref);
            }
        }
        match *self.checker.body.pat(pat) {
            Pat::Wild | Pat::Rest => {}
            Pat::Paren(inner) | Pat::Box(inner) => {
                self.walk_pat(inner, &cur_steps, (through_ref, mut_ref_root, shared));
            }
            Pat::Ref { mutable, pat: inner } => {
                through_ref = true;
                if !mutable {
                    shared = true;
                    mut_ref_root = false;
                } else if !shared {
                    mut_ref_root = true;
                }
                push_step(&mut cur_steps, PlaceStep::Deref);
                self.walk_pat(inner, &cur_steps, (through_ref, mut_ref_root, shared));
            }
            Pat::Bind { sub, .. } => {
                if self.checker.typed.pat_res.contains_key(&pat) {
                    if self.pat_res_inspects(pat) {
                        self.record_path(cur_steps, false, through_ref, mut_ref_root && !shared);
                    }
                    return;
                }
                let is_leaf = sub.is_none_or(|s| matches!(*self.checker.body.pat(s), Pat::Rest | Pat::Wild));
                if let Some(&mutability) = self.checker.typed.ref_binds.get(&pat) {
                    self.record_path(cur_steps.clone(), mutability == Mut::Mut, through_ref, mut_ref_root && !shared);
                } else if is_leaf {
                    self.record_path(cur_steps.clone(), false, through_ref, mut_ref_root && !shared);
                }
                if let Some(sub) = sub {
                    self.walk_pat(sub, &cur_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Lit(_) | Pat::Range { .. } | Pat::Path(_) | Pat::Never | Pat::Macro { .. } => {
                self.record_path(cur_steps, false, through_ref, mut_ref_root && !shared);
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                if matches!(*self.checker.body.pat(pat), Pat::TupleStruct(..)) && self.pat_res_inspects(pat) {
                    self.record_path(cur_steps.clone(), false, through_ref, mut_ref_root && !shared);
                }
                let mut saw_rest = false;
                for (idx, &id) in self.checker.body.list(list).iter().enumerate() {
                    let sub = PatId(id);
                    if matches!(self.checker.body.pat(sub), Pat::Rest) {
                        saw_rest = true;
                        continue;
                    }
                    if saw_rest {
                        self.record_path(cur_steps.clone(), false, through_ref, mut_ref_root && !shared);
                    } else {
                        let mut next_steps = cur_steps.clone();
                        push_step(&mut next_steps, PlaceStep::Field(FieldKey::Tuple(idx as u32)));
                        self.walk_pat(sub, &next_steps, (through_ref, mut_ref_root, shared));
                    }
                }
            }
            Pat::Struct { fields, .. } => {
                if self.pat_res_inspects(pat) {
                    self.record_path(cur_steps.clone(), false, through_ref, mut_ref_root && !shared);
                }
                for &fp_id in self.checker.body.list(fields) {
                    let fp = self.checker.body.field_pats[fp_id as usize];
                    let mut next_steps = cur_steps.clone();
                    push_step(&mut next_steps, PlaceStep::Field(FieldKey::Named(self.checker.src.text(fp.name).to_owned())));
                    self.walk_pat(fp.pat, &next_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Slice(list) => {
                if self.checker.typed.pats.get(&pat).is_some_and(|&ty| matches!(self.checker.cx.types.kind(ty), Kind::Slice(_))) {
                    self.record_path(cur_steps.clone(), false, through_ref, mut_ref_root && !shared);
                }
                for (sub, step) in self.checker.slice_pat_steps(pat, list) {
                    let mut next_steps = cur_steps.clone();
                    push_step(&mut next_steps, step);
                    self.walk_pat(sub, &next_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Or(list) => {
                for &alt in self.checker.body.list(list) {
                    self.walk_pat(PatId(alt), &cur_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Deref(inner) => {
                self.record_path(cur_steps.clone(), false, through_ref, mut_ref_root && !shared);
                self.walk_pat(inner, &cur_steps, (through_ref, mut_ref_root, shared));
            }
        }
    }

    fn pat_has_mut_ref_bind(&self, pat: PatId) -> bool {
        match *self.checker.body.pat(pat) {
            Pat::Bind { sub, .. } => self.checker.typed.ref_binds.get(&pat) == Some(&Mut::Mut) || sub.is_some_and(|s| self.pat_has_mut_ref_bind(s)),
            Pat::Paren(inner) | Pat::Box(inner) | Pat::Deref(inner) => self.pat_has_mut_ref_bind(inner),
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list) | Pat::Or(list) => {
                self.checker.body.list(list).iter().any(|&id| self.pat_has_mut_ref_bind(PatId(id)))
            }
            Pat::Struct { fields, .. } => {
                self.checker.body.list(fields).iter().any(|&id| self.pat_has_mut_ref_bind(self.checker.body.field_pats[id as usize].pat))
            }
            _ => false,
        }
    }

    fn walk_expr(&mut self, expr: ExprId) {
        match *self.checker.body.expr(expr) {
            Expr::Ref { mutable, raw: _, expr: inner } => {
                if let Some(p) = self.checker.root_place(inner)
                    && p.local == self.target
                {
                    self.record_place(&p, mutable);
                    self.walk_place_indices(inner);
                    return;
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
                    self.record_place(&p, false);
                    self.walk_place_indices(expr);
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
                    self.record_path(Vec::new(), false, false, false);
                }
            }
            Expr::Assign(lhs, rhs) | Expr::AssignOp(_, lhs, rhs) => {
                if let Some(p) = self.checker.root_place(lhs)
                    && p.local == self.target
                {
                    self.record_place(&p, true);
                    self.walk_place_indices(lhs);
                } else {
                    self.walk_expr(lhs);
                }
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
                if let Some(mut p) = self.checker.root_place(receiver)
                    && p.local == self.target
                {
                    let adjust = self.checker.typed.adjusts.get(&expr).copied();
                    let autoref = adjust.and_then(|a| a.autoref);
                    let derefs = adjust.map_or(0, |a| a.derefs);
                    if derefs > 0 {
                        self.checker.apply_deref_steps(expr, self.checker.expr_ty(receiver), derefs, &mut p);
                    } else if autoref.is_none() && self.checker.is_mut_ref_ty(self.checker.expr_ty(receiver)) {
                        self.checker.deref_ref_step(self.checker.expr_ty(receiver), &mut p);
                        push_step(&mut p.steps, PlaceStep::Deref);
                    }
                    let is_mut = autoref == Some(Mut::Mut)
                        || (autoref.is_none() && self.checker.is_mut_ref_ty(self.checker.expr_ty(receiver)));
                    self.record_place(&p, is_mut);
                    self.walk_place_indices(receiver);
                } else {
                    self.walk_expr(receiver);
                }
                self.walk_list(params);
            }
            Expr::Block { kind, stmts, tail, .. } => {
                if matches!(kind, BlockKind::Async | BlockKind::AsyncMove) && !self.check_nested_capture(expr) {
                    return;
                }
                for &id in self.checker.body.list(stmts) {
                    match self.checker.body.stmts[id as usize] {
                        Stmt::Let { pat, init, otherwise, .. } => {
                            if let Some(e) = init {
                                self.walk_scrutinee_pat(e, &[pat]);
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
                if let Some(alt) = e { self.walk_expr(alt); }
            }
            Expr::Let(pat, init) => self.walk_scrutinee_pat(init, &[pat]),
            Expr::While { cond, body, .. } => { self.walk_expr(cond); self.walk_expr(body); }
            Expr::Loop { body, .. } => self.walk_expr(body),
            Expr::For { iter, body, .. } => { self.walk_expr(iter); self.walk_expr(body); }
            Expr::Match(scrut, arms) => {
                let arm_ids = self.checker.body.list(arms);
                let pats: Vec<PatId> = arm_ids.iter().map(|&id| self.checker.body.arms[id as usize].pat).collect();
                self.walk_scrutinee_pat(scrut, &pats);
                for &id in arm_ids {
                    let arm = self.checker.body.arms[id as usize];
                    if let Some(g) = arm.guard { self.walk_expr(g); }
                    self.walk_expr(arm.body);
                }
            }
            Expr::Closure { body, .. } => if self.check_nested_capture(expr) { self.walk_expr(body); },
            Expr::Tuple(list) | Expr::Array(list) => self.walk_list(list),
            Expr::Struct { fields, base, .. } => {
                for &id in self.checker.body.list(fields) {
                    self.walk_expr(self.checker.body.fields[id as usize].value);
                }
                if let Some(b) = base { self.walk_expr(b); }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from { self.walk_expr(f); }
                if let Some(t) = to { self.walk_expr(t); }
            }
            Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => if let Some(e) = e { self.walk_expr(e); },
            Expr::Lit(..) | Expr::Continue(_) | Expr::Macro { .. } | Expr::Underscore | Expr::FormatArgs(_) | Expr::OffsetOf { .. } => {}
        }
    }
}
