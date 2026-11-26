use crate::body::{BlockKind, Expr, ExprId, UnOp};
use crate::names::ItemRef;
use crate::sema::infer::{Adjust, Res, field_types};
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::super::const_val::{ConstEnv, Frame, PlaceRef, PtrTarget, Val};
use super::super::literal::Scope;

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    pub(super) fn eval_place_mut(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        if let Expr::Path(_) = *scope.body.expr(expr)
            && let Some(&Res::Local(local)) = scope.typed.res.get(&expr)
        {
            let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
            let slot = frame.locals.get_mut(local as usize)?;
            let id = match *slot {
                Some(id) => id,
                None => {
                    let id = self.alloc_ty(ty)?;
                    *slot = Some(id);
                    id
                }
            };
            return Some(PlaceRef {
                target: PtrTarget::Obj { id, addend: 0 },
                ty,
                slice_len: None,
            });
        }
        self.eval_place(scope, expr, frame)
    }

    /// Evaluates `&inner` or `&raw inner` to a thin or wide pointer value.
    pub(in super::super) fn eval_ref_val(
        &mut self,
        scope: Scope<'_>,
        inner: ExprId,
        _raw: bool,
        ref_ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        let place = self.eval_place_or_temp(scope, inner, frame)?;
        if let Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) = *self.lower.cx.types.kind(ref_ty)
            && let Kind::Slice(_) = *self.lower.cx.types.kind(pointee)
            && let Kind::Array(_, len_ty) = *self.lower.cx.types.kind(place.ty)
        {
            let len = self.lower.cx.types.len(len_ty)?;
            return Some(Val::wide(place.target, len));
        }
        Some(self.make_ref(&place))
    }

    pub(in super::super) fn make_ref(&self, place: &PlaceRef) -> Val {
        match place.slice_len {
            Some(len) => Val::wide(place.target, len),
            None => Val::ptr(place.target),
        }
    }

    /// Returns a place for `expr`, materializing temporaries into a new object if needed.
    pub(in super::super) fn eval_place_or_temp(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        if scope.typed.coercions.get(&expr).is_none()
            && let Some(place) = self.eval_place(scope, expr, frame)
        {
            return Some(place);
        }
        let val = self.eval_expr_val(scope, expr, frame)?;
        let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
        let id = self.alloc_val(val)?;
        Some(PlaceRef {
            target: PtrTarget::Obj { id, addend: 0 },
            ty,
            slice_len: None,
        })
    }

    /// Resolves a place expression (local, immutable static, deref, field, index) to `PlaceRef`.
    pub(in super::super) fn eval_place(
        &mut self,
        scope: Scope<'_>,
        mut expr: ExprId,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        loop {
            match *scope.body.expr(expr) {
                Expr::Paren(inner) => expr = inner,
                Expr::Block {
                    stmts,
                    tail: Some(inner),
                    label: None,
                    kind,
                } if scope.body.list(stmts).is_empty()
                    && matches!(
                        kind,
                        BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe
                    ) =>
                {
                    expr = inner;
                }
                _ => break,
            }
        }
        let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
        match *scope.body.expr(expr) {
            Expr::Path(_) => self.eval_path_place(scope, expr, ty, frame),
            Expr::Unary(UnOp::Deref, operand) => {
                self.eval_deref_target(scope, expr, operand, frame)
            }
            Expr::TupleField(base, index) => self.eval_field_place(scope, expr, base, index, frame),
            Expr::Field(base, _) => {
                self.eval_field_place(scope, expr, base, *scope.typed.fields.get(&expr)?, frame)
            }
            Expr::Index(base, index) => self.eval_index_place(scope, expr, base, index, frame),
            _ => None,
        }
    }

    fn eval_path_place(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        ty: Ty,
        frame: &Frame,
    ) -> Option<PlaceRef> {
        let id = match *scope.typed.res.get(&expr)? {
            Res::Local(local) => frame.locals.get(local as usize).copied().flatten()?,
            Res::Fn(item, substs) if super::super::super::is_static(self.lower.cx, item) => {
                self.static_obj(item, substs, ty)?
            }
            _ => return None,
        };
        Some(PlaceRef {
            target: PtrTarget::Obj { id, addend: 0 },
            ty,
            slice_len: None,
        })
    }

    fn eval_deref_target(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        operand: ExprId,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        let place = self.eval_place_or_temp(scope, operand, frame)?;
        let ty = scope.typed.expr_ty(&self.lower.cx.types, operand);
        if matches!(*self.lower.cx.types.kind(ty), Kind::Ptr(..))
            && !scope.typed.derefs.contains_key(&(expr, 0))
        {
            return None;
        }
        self.step_deref_place(scope, (expr, 0), &place, ty)
    }

    pub(in super::super) fn eval_deref_read(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        operand: ExprId,
        frame: &mut Frame,
    ) -> Option<Val> {
        let place = self.eval_deref_target(scope, expr, operand, frame)?;
        self.read_place(&place)
    }

    fn step_deref_place(
        &mut self,
        scope: Scope<'_>,
        site: (ExprId, u32),
        place: &PlaceRef,
        ty: Ty,
    ) -> Option<PlaceRef> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.derefs.get(&site) {
            let ptr = self.make_ref(place);
            let out = self.call_fn(item, substs, vec![ptr])?;
            let inner = match *self.lower.cx.types.kind(ty) {
                Kind::Ref(_, inner) => inner,
                _ => scope.typed.expr_ty(&self.lower.cx.types, site.0),
            };
            return self.place_from_ref_val(&out, inner);
        }
        let Kind::Ref(_, inner) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        let ptr_val = self.read_place(place)?;
        self.place_from_ref_val(&ptr_val, inner)
    }

    pub(in super::super) fn place_from_ref_val(
        &mut self,
        ptr_val: &Val,
        inner: Ty,
    ) -> Option<PlaceRef> {
        let target = ptr_val.read_ptr(0)?;
        let slice_len = if self.lower.cx.is_unsized(inner).ok()? {
            Some(u64::from_le_bytes(
                ptr_val.bytes.get(8..16)?.try_into().ok()?,
            ))
        } else {
            None
        };
        Some(PlaceRef {
            target,
            ty: inner,
            slice_len,
        })
    }

    fn eval_autoderef_place(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        base: ExprId,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        let mut place = self.eval_place_or_temp(scope, base, frame)?;
        let steps = scope.typed.autoderefs.get(&expr).copied().unwrap_or(0);
        for step in 0..steps {
            let cur_ty = place.ty;
            place = self.step_deref_place(scope, (expr, step), &place, cur_ty)?;
        }
        Some(place)
    }

    fn eval_field_place(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        base: ExprId,
        index: u32,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        let place = self.eval_autoderef_place(scope, expr, base, frame)?;
        self.part_place_ref(&place, (0, index))
    }

    pub(in super::super) fn part_place_ref(
        &mut self,
        place: &PlaceRef,
        (variant, index): (u32, u32),
    ) -> Option<PlaceRef> {
        match *self.lower.cx.types.kind(place.ty) {
            Kind::Array(elem, _) => {
                let delta = self
                    .lower
                    .cx
                    .layout(elem)
                    .ok()?
                    .size
                    .bytes()
                    .checked_mul(u64::from(index))?;
                Some(PlaceRef {
                    target: place.target.offset(delta)?,
                    ty: elem,
                    slice_len: None,
                })
            }
            Kind::Tuple(items) => {
                let elem = *self.lower.cx.types.substs(items).get(index as usize)?;
                let layout = self.lower.cx.layout(place.ty).ok()?;
                let delta = u64::try_from(field_offset(&layout, 0, index)).ok()?;
                Some(PlaceRef {
                    target: place.target.offset(delta)?,
                    ty: elem,
                    slice_len: None,
                })
            }
            Kind::Adt(item, args) => self.adt_part_place(place, (item, args), (variant, index)),
            _ => None,
        }
    }

    fn adt_part_place(
        &mut self,
        place: &PlaceRef,
        (item, args): (ItemRef, Substs),
        (variant, index): (u32, u32),
    ) -> Option<PlaceRef> {
        let slice = self.lower.cx.types.substs(args).to_vec();
        let elem = *field_types(self.lower.cx, item, &slice)
            .ok()?
            .get(variant as usize)?
            .get(index as usize)?;
        let layout = self.lower.cx.layout(place.ty).ok()?;
        let delta = u64::try_from(field_offset(&layout, variant, index)).ok()?;
        Some(PlaceRef {
            target: place.target.offset(delta)?,
            ty: elem,
            slice_len: None,
        })
    }

    fn eval_index_place(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        base: ExprId,
        index: ExprId,
        frame: &mut Frame,
    ) -> Option<PlaceRef> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.operators.get(&expr) {
            let base_place = self.eval_autoderef_place(scope, expr, base, frame)?;
            let idx_val = self.eval_expr_val(scope, index, frame)?;
            let out = self.call_fn(item, substs, vec![self.make_ref(&base_place), idx_val])?;
            return self.place_from_ref_val(&out, scope.typed.expr_ty(&self.lower.cx.types, expr));
        }
        let place = self.eval_autoderef_place(scope, expr, base, frame)?;
        let idx = u64::try_from(self.eval_expr_val(scope, index, frame)?.as_u128()?).ok()?;
        self.eval_raw_index_place(&place, idx)
    }

    fn eval_raw_index_place(&mut self, place: &PlaceRef, idx: u64) -> Option<PlaceRef> {
        let (elem, len) = match *self.lower.cx.types.kind(place.ty) {
            Kind::Array(elem, len_ty) => (elem, self.lower.cx.types.len(len_ty)?),
            Kind::Slice(elem) => (elem, place.slice_len?),
            _ => return None,
        };
        if idx >= len {
            return None;
        }
        let stride = self.lower.cx.layout(elem).ok()?.size.bytes();
        Some(PlaceRef {
            target: place.target.offset(stride.checked_mul(idx)?)?,
            ty: elem,
            slice_len: None,
        })
    }

    /// Computes the adjusted receiver value of a method call.
    pub(in super::super) fn eval_receiver(
        &mut self,
        scope: Scope<'_>,
        (expr, receiver): (ExprId, ExprId),
        adjust: Adjust,
        frame: &mut Frame,
    ) -> Option<Val> {
        if adjust
            == (Adjust {
                derefs: 0,
                autoref: None,
            })
        {
            return self.eval_expr_val(scope, receiver, frame);
        }
        let mut place = self.eval_place_or_temp(scope, receiver, frame)?;
        for step in 0..adjust.derefs {
            if let Kind::Array(_, len_ty) = *self.lower.cx.types.kind(place.ty) {
                adjust.autoref?;
                return Some(Val::wide(place.target, self.lower.cx.types.len(len_ty)?));
            }
            let cur_ty = place.ty;
            place = self.step_deref_place(scope, (expr, step), &place, cur_ty)?;
        }
        match adjust.autoref {
            Some(_) => Some(self.make_ref(&place)),
            None => self.read_place(&place),
        }
    }
}
