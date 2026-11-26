mod capture;

use crate::body::{Expr, ExprId, List, PatId, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::infer::Res;
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
    SliceElem {
        offset: u64,
        min_len: u64,
        from_end: bool,
        array_len: Option<u64>,
    },
    SliceSub {
        from: u64,
        to: u64,
        array_len: Option<u64>,
    },
    Deref,
    Union,
    Opaque(Option<u32>),
}

#[derive(Clone, Debug)]
pub(super) struct Place {
    pub(super) local: u32,
    pub(super) through_ref: bool,
    pub(super) through_deref: bool,
    pub(super) mut_ref_root: bool,
    pub(super) behind_shared: bool,
    pub(super) reborrows_shared_field: bool,
    pub(super) raw_ptr_prefix: Option<usize>,
    pub(super) steps: Vec<PlaceStep>,
}

pub(super) fn steps_conflict(a: &[PlaceStep], b: &[PlaceStep]) -> bool {
    for (x, y) in a.iter().zip(b) {
        if !step_conflict(x, y) {
            return false;
        }
        match (x, y) {
            (PlaceStep::Opaque(Some(a)), PlaceStep::Opaque(Some(b))) if a != b => return true,
            (PlaceStep::Opaque(None), _) | (_, PlaceStep::Opaque(None)) => return true,
            _ => {}
        }
    }
    true
}

fn step_conflict(a: &PlaceStep, b: &PlaceStep) -> bool {
    if a == b {
        return true;
    }
    match (a, b) {
        (PlaceStep::Union, _)
        | (_, PlaceStep::Union)
        | (PlaceStep::Opaque(_), PlaceStep::Opaque(_)) => true,
        (PlaceStep::Opaque(_), PlaceStep::SliceElem { .. } | PlaceStep::SliceSub { .. })
        | (PlaceStep::SliceElem { .. } | PlaceStep::SliceSub { .. }, PlaceStep::Opaque(_)) => true,
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
        PlaceStep::SliceElem {
            offset,
            from_end: false,
            ..
        } => {
            let start = offset.min(len);
            (start, (start + 1).min(len))
        }
        PlaceStep::SliceElem {
            offset,
            from_end: true,
            ..
        } => {
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
            PlaceStep::SliceElem {
                offset: o1,
                min_len: m1,
                from_end: e1,
                ..
            },
            PlaceStep::SliceElem {
                offset: o2,
                min_len: m2,
                from_end: e2,
                ..
            },
        ) => {
            if e1 == e2 {
                o1 == o2
            } else {
                o1 + o2 >= (*m1).max(*m2)
            }
        }
        (
            PlaceStep::SliceElem {
                offset, from_end, ..
            },
            PlaceStep::SliceSub { from, to, .. },
        )
        | (
            PlaceStep::SliceSub { from, to, .. },
            PlaceStep::SliceElem {
                offset, from_end, ..
            },
        ) => {
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
    if !(matches!(step, PlaceStep::Opaque(_)) && steps.last() == Some(&step)) {
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
        self.names
            .get(local as usize)
            .and_then(|n| n.clone())
            .unwrap_or_else(|| format!("_{local}"))
    }

    pub(super) fn local_ty(&self, local: u32) -> Ty {
        self.typed
            .locals
            .get(local as usize)
            .map_or(Ty::NONE, |l| l.ty)
    }

    pub(super) fn expr_ty(&self, expr: ExprId) -> Ty {
        self.typed
            .exprs
            .get(expr.0 as usize)
            .copied()
            .unwrap_or(Ty::NONE)
    }

    pub(super) fn is_ref_ty(&self, ty: Ty) -> bool {
        ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(..))
    }

    pub(super) fn is_mut_ref_ty(&self, ty: Ty) -> bool {
        ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Mut, _))
    }

    pub(super) fn is_shared_ref_or_ptr(&self, ty: Ty) -> bool {
        ty != Ty::NONE
            && matches!(
                self.cx.types.kind(ty),
                Kind::Ref(Mut::Not, _) | Kind::Ptr(Mut::Not, _)
            )
    }

    pub(super) fn is_mut_ref_or_ptr(&self, ty: Ty) -> bool {
        ty != Ty::NONE
            && matches!(
                self.cx.types.kind(ty),
                Kind::Ref(Mut::Mut, _) | Kind::Ptr(Mut::Mut, _)
            )
    }

    pub(super) fn is_param_local(&self, local: u32) -> bool {
        self.is_param.get(local as usize).copied().unwrap_or(false)
    }

    pub(super) fn is_closure_param(&self, local: u32) -> bool {
        self.typed
            .closures
            .values()
            .any(|c| c.params.contains(&local))
    }

    pub(super) fn ty_is_copy(&mut self, ty: Ty) -> bool {
        super::is_copy_ty(self.cx, self.item, self.typed, ty)
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
        if ty != Ty::NONE
            && matches!(self.cx.types.kind(ty), Kind::Ptr(..))
            && p.raw_ptr_prefix.is_none()
        {
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

    pub(super) fn deref_target_ty(&mut self, site: ExprId, step: u32) -> Option<Ty> {
        let Some(Res::Fn(d_fn, d_args)) = self.typed.derefs.get(&(site, step)).cloned() else {
            return None;
        };
        let args_vec = self.cx.types.substs(d_args).to_vec();
        super::super::infer::detached(self.cx, self.item, |infer| infer.fn_types(d_fn, &args_vec))
            .ok()
            .and_then(|(_, ret)| match *self.cx.types.kind(ret) {
                Kind::Ref(_, inner) => Some(inner),
                _ => None,
            })
    }

    pub(super) fn apply_deref_steps(
        &mut self,
        site: ExprId,
        base_ty: Ty,
        count: u32,
        p: &mut Place,
    ) -> Ty {
        let mut cur = base_ty;
        for step in 0..count {
            if self.typed.derefs.contains_key(&(site, step)) {
                push_step(&mut p.steps, PlaceStep::Opaque(Some(site.0)));
                if !p.behind_shared && self.is_mut_ref_or_ptr(cur) {
                    p.mut_ref_root = true;
                }
                p.through_deref = true;
                cur = self.deref_target_ty(site, step).unwrap_or(Ty::NONE);
                continue;
            }
            if cur != Ty::NONE {
                self.deref_ref_step(cur, p);
                cur = self.builtin_deref_ty(cur).unwrap_or(Ty::NONE);
            } else if !p.through_deref {
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

    pub(super) fn root_place(&mut self, expr: ExprId) -> Option<Place> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.root_place(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Local(local)) => Some(Place {
                    local: *local,
                    through_ref: false,
                    through_deref: false,
                    mut_ref_root: false,
                    behind_shared: false,
                    reborrows_shared_field: false,
                    raw_ptr_prefix: None,
                    steps: Vec::new(),
                }),
                _ => None,
            },
            Expr::Field(base, name) => {
                let mut p = self.root_place(base)?;
                self.project_field(
                    expr,
                    base,
                    FieldKey::Named(self.src.text(name).to_owned()),
                    &mut p,
                );
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
                let autoderefs = self.typed.autoderefs.get(&expr).copied().unwrap_or(0);
                if autoderefs > 0 {
                    self.apply_deref_steps(expr, base_ty, autoderefs, &mut p);
                } else {
                    self.deref_ref_step(base_ty, &mut p);
                }
                push_step(&mut p.steps, PlaceStep::Opaque(Some(expr.0)));
                Some(p)
            }
            Expr::Unary(UnOp::Deref, inner) => {
                let mut p = self.root_place(inner)?;
                let inner_ty = self.expr_ty(inner);
                if self.typed.derefs.contains_key(&(expr, 0)) {
                    push_step(&mut p.steps, PlaceStep::Opaque(Some(expr.0)));
                    p.through_deref = true;
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

    fn project_field(&mut self, expr: ExprId, base: ExprId, key: FieldKey, p: &mut Place) {
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
            && matches!(
                self.cx.types.kind(cur),
                Kind::Adt(item, _) if self.cx.item(*item).1.kind == ItemKind::Union
            );
        push_step(
            &mut p.steps,
            if is_union {
                PlaceStep::Union
            } else {
                PlaceStep::Field(key)
            },
        );
    }

    pub(super) fn is_static_mut(&self, item: ItemRef) -> bool {
        let (src, entry) = self.cx.item(item);
        (entry.start..entry.name.unwrap_or(entry.start)).any(|t| src.text(t) == "mut")
    }

    pub(super) fn static_root(&self, expr: ExprId) -> Option<(ItemRef, bool, bool)> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.static_root(inner),
            Expr::Path(_) => match self.typed.res.get(&expr) {
                Some(Res::Fn(item, _)) if self.cx.item(*item).1.kind == ItemKind::Static => {
                    Some((*item, false, false))
                }
                _ => None,
            },
            Expr::Field(base, _) | Expr::TupleField(base, _) => {
                let (item, mut mr, mut bs) = self.static_root(base)?;
                let mut cur = self.expr_ty(base);
                let autoderefs = self.typed.autoderefs.get(&expr).copied().unwrap_or(0);
                let steps = if autoderefs > 0 {
                    autoderefs
                } else {
                    u32::from(
                        self.is_ref_ty(cur) || matches!(self.cx.types.kind(cur), Kind::Ptr(..)),
                    )
                };
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
        self.place_behind(
            expr,
            if shared_only {
                PlaceBehind::SharedRef
            } else {
                PlaceBehind::AnyRef
            },
        )
    }

    fn place_behind(&self, expr: ExprId, mode: PlaceBehind) -> bool {
        let matches_ty = |ty: Ty| {
            ty != Ty::NONE
                && match mode {
                    PlaceBehind::SharedRef => {
                        matches!(self.cx.types.kind(ty), Kind::Ref(Mut::Not, _))
                    }
                    PlaceBehind::AnyRef => matches!(self.cx.types.kind(ty), Kind::Ref(..)),
                    PlaceBehind::AnyRefOrPtr => {
                        matches!(self.cx.types.kind(ty), Kind::Ref(..) | Kind::Ptr(..))
                    }
                }
        };
        match *self.body.expr(expr) {
            Expr::Paren(inner) => self.place_behind(inner, mode),
            Expr::Unary(UnOp::Deref, inner) => {
                let ty = self.expr_ty(inner);
                if mode != PlaceBehind::AnyRefOrPtr
                    && ty != Ty::NONE
                    && matches!(self.cx.types.kind(ty), Kind::Ptr(..))
                {
                    return false;
                }
                (mode == PlaceBehind::AnyRefOrPtr && self.typed.derefs.contains_key(&(expr, 0)))
                    || matches_ty(ty)
                    || self.place_behind(inner, mode)
            }
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => {
                let ty = self.expr_ty(base);
                if mode != PlaceBehind::AnyRefOrPtr
                    && ty != Ty::NONE
                    && matches!(self.cx.types.kind(ty), Kind::Ptr(..))
                {
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
        let array_len = self
            .typed
            .pats
            .get(&pat)
            .and_then(|&ty| match *self.cx.types.kind(ty) {
                Kind::Array(_, len) => self.cx.types.len(len),
                _ => None,
            });
        pat_ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| {
                let step = match rest_pos {
                    Some(r) if idx == r => PlaceStep::SliceSub {
                        from: r as u64,
                        to: (pat_ids.len() - r - 1) as u64,
                        array_len,
                    },
                    Some(r) if idx > r => PlaceStep::SliceElem {
                        offset: (pat_ids.len() - idx) as u64,
                        min_len,
                        from_end: true,
                        array_len,
                    },
                    _ => PlaceStep::SliceElem {
                        offset: idx as u64,
                        min_len,
                        from_end: false,
                        array_len,
                    },
                };
                (PatId(id), step)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{FieldKey, PlaceStep, steps_conflict};

    #[test]
    fn steps_conflict_handles_fields_prefixes_slices_and_unions() {
        let fa = PlaceStep::Field(FieldKey::Named("a".to_owned()));
        let fb = PlaceStep::Field(FieldKey::Named("b".to_owned()));
        let t0 = PlaceStep::Field(FieldKey::Tuple(0));
        let t1 = PlaceStep::Field(FieldKey::Tuple(1));

        assert!(!steps_conflict(&[fa.clone()], &[fb.clone()]));
        assert!(!steps_conflict(&[t0.clone()], &[t1]));
        assert!(steps_conflict(&[], &[fa.clone()]));
        assert!(steps_conflict(&[fa.clone()], &[fa.clone(), t0]));
        assert!(steps_conflict(&[PlaceStep::Union], &[fb]));
        assert!(steps_conflict(
            &[PlaceStep::Opaque(Some(1))],
            &[PlaceStep::Opaque(Some(2))]
        ));

        let elem0 = PlaceStep::SliceElem {
            offset: 0,
            min_len: 3,
            from_end: false,
            array_len: Some(3),
        };
        let elem2 = PlaceStep::SliceElem {
            offset: 2,
            min_len: 3,
            from_end: false,
            array_len: Some(3),
        };
        let tail1 = PlaceStep::SliceElem {
            offset: 1,
            min_len: 3,
            from_end: true,
            array_len: Some(3),
        };
        let tail2 = PlaceStep::SliceElem {
            offset: 2,
            min_len: 3,
            from_end: true,
            array_len: Some(3),
        };
        let dyn_head0 = PlaceStep::SliceElem {
            offset: 0,
            min_len: 1,
            from_end: false,
            array_len: None,
        };
        let dyn_tail1 = PlaceStep::SliceElem {
            offset: 1,
            min_len: 1,
            from_end: true,
            array_len: None,
        };
        assert!(!steps_conflict(&[elem0.clone()], &[elem2.clone()]));
        assert!(!steps_conflict(&[elem2.clone()], &[elem0.clone()]));
        assert!(steps_conflict(&[elem2.clone()], &[tail1.clone()]));
        assert!(steps_conflict(&[tail1.clone()], &[elem2]));
        assert!(!steps_conflict(&[tail1], &[tail2]));
        assert!(steps_conflict(&[dyn_head0], &[dyn_tail1]));
        assert!(steps_conflict(&[elem0], &[PlaceStep::Opaque(None)]));
    }
}
