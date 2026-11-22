use rustc_abi::FieldsShape;

use crate::body::literal::{byte_value, char_value, string_bytes};
use crate::body::{BinOp, Expr, ExprId, List, Lit, UnOp};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::infer::{Adjust, Coercion, Res, field_types, parse_int};
use crate::sema::layout::field_offset;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info, mask};

use super::const_val::{ConstEnv, Frame, PlaceRef, PtrTarget, Val};
use super::literal::{Scope, apply_int_binary, apply_shift, cast_scalar, cmp_scalar, prim_info};

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    /// Applies any inference coercion recorded on `expr` to `val`.
    pub(super) fn apply_coercion(&mut self, scope: Scope<'_>, expr: ExprId, val: Val) -> Option<Val> {
        let Some(coercion) = scope.typed.coercions.get(&expr) else {
            return Some(val);
        };
        match coercion {
            Coercion::Len(len_ty) => {
                let len = self.lower.cx.types.len(*len_ty)?;
                let target = val.read_ptr(0)?;
                Some(Val::wide(target, len))
            }
            Coercion::Deref(steps, _) => self.apply_deref_coercion(steps, val),
            Coercion::Reify(Res::Fn(item, substs)) => Some(Val::ptr(PtrTarget::Fn(self.lower.rodata_fn_target(*item, *substs)?))),
            _ => None,
        }
    }

    fn apply_deref_coercion(&mut self, steps: &[(Ty, Option<Res>)], mut val: Val) -> Option<Val> {
        for (ty, res) in steps {
            val = match res {
                Some(Res::Fn(item, substs)) => self.call_fn(*item, *substs, vec![val])?,
                Some(_) => return None,
                None => {
                    let target = val.read_ptr(0)?;
                    let inner = match *self.lower.cx.types.kind(*ty) {
                        Kind::Ref(_, inner) | Kind::Ptr(_, inner) => inner,
                        _ => return None,
                    };
                    let slice_len = if self.lower.cx.is_unsized(inner).ok()? {
                        Some(u64::from_le_bytes(val.bytes.get(8..16)?.try_into().ok()?))
                    } else {
                        None
                    };
                    self.read_place(&PlaceRef { target, ty: inner, slice_len })?
                }
            };
        }
        Some(val)
    }

    /// Evaluates a literal expression to a `Val`.
    pub(super) fn eval_lit_val(&mut self, scope: Scope<'_>, lit: Lit, at: u32, ty: Ty) -> Option<Val> {
        let text = scope.src.text(at);
        if matches!(lit, Lit::Str | Lit::ByteStr | Lit::CStr) {
            return self.eval_str_lit(lit, text);
        }
        let Kind::Prim(prim) = *self.lower.cx.types.kind(ty) else { return None };
        self.eval_prim_lit(lit, prim, text, ty)
    }

    fn eval_str_lit(&mut self, lit: Lit, text: &str) -> Option<Val> {
        let mut bytes = string_bytes(text).ok()?;
        if lit == Lit::CStr {
            bytes.push(0);
        }
        let len = bytes.len() as u64;
        let id = self.alloc_val(Val { bytes, ptrs: Vec::new() })?;
        let target = PtrTarget::Obj { id, addend: 0 };
        Some(if lit == Lit::ByteStr { Val::ptr(target) } else { Val::wide(target, len) })
    }

    fn eval_prim_lit(&mut self, lit: Lit, prim: Primitive, text: &str, ty: Ty) -> Option<Val> {
        let bits = match (lit, prim) {
            (Lit::Bool(v), Primitive::Bool) => u128::from(v),
            (Lit::Char, Primitive::Char) => u128::from(u32::from(char_value(text).ok()?)),
            (Lit::Byte, Primitive::U8) => u128::from(byte_value(text).ok()?),
            (Lit::Int, _) => parse_int(text).ok()?.0 & mask(int_info(prim)?.0),
            _ => return None,
        };
        Some(Val::scalar(self.ty_size(ty)?, bits))
    }

    /// Evaluates a path expression (local, const, static, or unit constructor).
    pub(super) fn eval_path_val(&mut self, scope: Scope<'_>, expr: ExprId, ty: Ty, frame: &Frame) -> Option<Val> {
        match *scope.typed.res.get(&expr)? {
            Res::Local(local) => {
                let id = frame.locals.get(local as usize).copied().flatten()?;
                self.read_place(&PlaceRef { target: PtrTarget::Obj { id, addend: 0 }, ty, slice_len: None })
            }
            Res::Const(val_ty) => {
                let Kind::Value(bits) = *self.lower.cx.types.kind(val_ty) else { return None };
                Some(Val::scalar(self.ty_size(ty)?, bits))
            }
            Res::Ctor(_, variant, _) => {
                (!matches!(*self.lower.cx.types.kind(ty), Kind::FnDef(..))).then(|| self.construct_adt(ty, variant, Vec::new(), None)).flatten()
            }
            Res::Fn(item, substs) => match self.lower.cx.item(item).1.kind {
                ItemKind::Static => {
                    let id = self.static_obj(item, substs, ty)?;
                    self.read_place(&PlaceRef { target: PtrTarget::Obj { id, addend: 0 }, ty, slice_len: None })
                }
                ItemKind::Const => self.eval_const_item_val(item, substs, ty),
                ItemKind::Fn if matches!(*self.lower.cx.types.kind(ty), Kind::FnDef(..)) => Some(Val::zst()),
                _ => None,
            },
        }
    }

    /// Evaluates an associated or free `const` item to a `Val`.
    pub(super) fn eval_const_item_val(&mut self, item: ItemRef, substs: Substs, ty: Ty) -> Option<Val> {
        let (source, typed) = self.const_body_source(item, substs)?;
        let typed = typed.scoped(&self.lower.cx.types);
        if typed.ret_ty() != ty {
            return None;
        }
        let body = self.lower.cx.body(source).ok()?;
        let root = body.root?;
        let src = self.lower.cx.body_src(source);
        self.depth += 1;
        let mut frame = Frame { locals: vec![None; typed.locals.len()] };
        let out = self.eval_expr_val(Scope { body: &body, src, typed: &typed, in_const: true }, root, &mut frame);
        self.depth -= 1;
        out
    }

    /// Evaluates a unary expression other than deref.
    pub(super) fn eval_unary_val(&mut self, scope: Scope<'_>, (expr, op, operand): (ExprId, UnOp, ExprId), ty: Ty, frame: &mut Frame) -> Option<Val> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.operators.get(&expr) {
            let arg = self.eval_expr_val(scope, operand, frame)?;
            return self.call_fn(item, substs, vec![arg]);
        }
        let Kind::Prim(prim) = *self.lower.cx.types.kind(ty) else { return None };
        let val = self.eval_expr_val(scope, operand, frame)?.as_u128()?;
        let out = match op {
            UnOp::Neg if int_info(prim)?.1 => val.wrapping_neg() & mask(int_info(prim)?.0),
            UnOp::Not if prim == Primitive::Bool => u128::from(val == 0),
            UnOp::Not => (!val) & mask(int_info(prim)?.0),
            _ => return None,
        };
        Some(Val::scalar(self.ty_size(ty)?, out))
    }

    /// Evaluates a binary expression including short-circuiting `&&` and `||`.
    pub(super) fn eval_binary_val(&mut self, scope: Scope<'_>, (expr, op, lhs, rhs): (ExprId, BinOp, ExprId, ExprId), ty: Ty, frame: &mut Frame) -> Option<Val> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.operators.get(&expr) {
            let l = self.eval_expr_val(scope, lhs, frame)?;
            let r = self.eval_expr_val(scope, rhs, frame)?;
            return self.call_fn(item, substs, vec![l, r]);
        }
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.eval_short_circuit(scope, op, lhs, rhs, frame);
        }
        let lval = self.eval_expr_val(scope, lhs, frame)?.as_u128()?;
        let rval = self.eval_expr_val(scope, rhs, frame)?.as_u128()?;
        let lty = scope.typed.expr_ty(&self.lower.cx.types, lhs);
        let rty = scope.typed.expr_ty(&self.lower.cx.types, rhs);
        self.apply_scalar_binop(op, (lval, lty), (rval, rty), ty)
    }

    fn eval_short_circuit(&mut self, scope: Scope<'_>, op: BinOp, lhs: ExprId, rhs: ExprId, frame: &mut Frame) -> Option<Val> {
        let l = self.eval_expr_val(scope, lhs, frame)?.as_u128()? != 0;
        if (op == BinOp::And && !l) || (op == BinOp::Or && l) {
            return Some(Val::scalar(1, u128::from(l)));
        }
        let r = self.eval_expr_val(scope, rhs, frame)?.as_u128()? != 0;
        Some(Val::scalar(1, u128::from(r)))
    }

    fn apply_scalar_binop(&mut self, op: BinOp, (lval, lty): (u128, Ty), (rval, rty): (u128, Ty), res_ty: Ty) -> Option<Val> {
        let (Kind::Prim(lprim), Kind::Prim(rprim)) = (self.lower.cx.types.kind(lty).clone(), self.lower.cx.types.kind(rty).clone()) else {
            return None;
        };
        let bits = match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge if lprim == rprim => {
                u128::from(cmp_scalar(op, lval, rval, prim_info(lprim)?))
            }
            BinOp::Shl | BinOp::Shr => {
                let Kind::Prim(dst_prim) = *self.lower.cx.types.kind(res_ty) else { return None };
                apply_shift(op, dst_prim, (lval, rval), int_info(rprim)?)?
            }
            _ => {
                let Kind::Prim(dst_prim) = *self.lower.cx.types.kind(res_ty) else { return None };
                apply_int_binary(op, dst_prim, lval, rval)?
            }
        };
        Some(Val::scalar(self.ty_size(res_ty)?, bits))
    }

    /// Evaluates a primitive cast expression.
    pub(super) fn eval_cast_val(&mut self, scope: Scope<'_>, inner: ExprId, dst_ty: Ty, frame: &mut Frame) -> Option<Val> {
        let src_ty = scope.typed.expr_ty(&self.lower.cx.types, inner);
        if self.lower.cx.is_unsized(dst_ty).ok()? {
            return None;
        }
        if let (Kind::FnDef(item, substs), Kind::FnPtr(..)) = (self.lower.cx.types.kind(src_ty).clone(), self.lower.cx.types.kind(dst_ty).clone()) {
            return Some(Val::ptr(PtrTarget::Fn(self.lower.rodata_fn_target(item, substs)?)));
        }
        let (Kind::Prim(src_prim), Kind::Prim(dst_prim)) = (self.lower.cx.types.kind(src_ty).clone(), self.lower.cx.types.kind(dst_ty).clone()) else {
            return None;
        };
        let src_val = self.eval_expr_val(scope, inner, frame)?.as_u128()?;
        let out = cast_scalar(src_val, src_prim, dst_prim)?;
        Some(Val::scalar(self.ty_size(dst_ty)?, out))
    }

    /// Evaluates simple assignment `lhs = rhs`.
    pub(super) fn eval_assign_val(&mut self, scope: Scope<'_>, (lhs, rhs): (ExprId, ExprId), frame: &mut Frame) -> Option<Val> {
        let val = self.eval_expr_val(scope, rhs, frame)?;
        let place = self.eval_place_mut(scope, lhs, frame)?;
        self.write_bytes(place.target, &val)?;
        Some(Val::zst())
    }

    /// Evaluates compound assignment `lhs op= rhs`.
    pub(super) fn eval_compound_val(&mut self, scope: Scope<'_>, (expr, op, lhs, rhs): (ExprId, BinOp, ExprId, ExprId), frame: &mut Frame) -> Option<Val> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.operators.get(&expr) {
            let lplace = self.eval_place_mut(scope, lhs, frame)?;
            let lref = self.make_ref(&lplace);
            let rval = self.eval_expr_val(scope, rhs, frame)?;
            return self.call_fn(item, substs, vec![lref, rval]);
        }
        let rval = self.eval_expr_val(scope, rhs, frame)?.as_u128()?;
        let place = self.eval_place_mut(scope, lhs, frame)?;
        let lval = self.read_place(&place)?.as_u128()?;
        let lty = scope.typed.expr_ty(&self.lower.cx.types, lhs);
        let rty = scope.typed.expr_ty(&self.lower.cx.types, rhs);
        let next = self.apply_scalar_binop(op, (lval, lty), (rval, rty), lty)?;
        self.write_bytes(place.target, &next)?;
        Some(Val::zst())
    }

    fn eval_place_mut(&mut self, scope: Scope<'_>, expr: ExprId, frame: &mut Frame) -> Option<PlaceRef> {
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
            return Some(PlaceRef { target: PtrTarget::Obj { id, addend: 0 }, ty, slice_len: None });
        }
        self.eval_place(scope, expr, frame)
    }

    /// Evaluates `&inner` or `&raw inner` to a thin or wide pointer value.
    pub(super) fn eval_ref_val(&mut self, scope: Scope<'_>, inner: ExprId, _raw: bool, ref_ty: Ty, frame: &mut Frame) -> Option<Val> {
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

    pub(super) fn make_ref(&self, place: &PlaceRef) -> Val {
        match place.slice_len {
            Some(len) => Val::wide(place.target, len),
            None => Val::ptr(place.target),
        }
    }

    /// Returns a place for `expr`, materializing temporaries into a new object if needed.
    pub(super) fn eval_place_or_temp(&mut self, scope: Scope<'_>, expr: ExprId, frame: &mut Frame) -> Option<PlaceRef> {
        if scope.typed.coercions.get(&expr).is_none()
            && let Some(place) = self.eval_place(scope, expr, frame)
        {
            return Some(place);
        }
        let val = self.eval_expr_val(scope, expr, frame)?;
        let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
        let id = self.alloc_val(val)?;
        Some(PlaceRef { target: PtrTarget::Obj { id, addend: 0 }, ty, slice_len: None })
    }

    /// Resolves a place expression (local, immutable static, deref, field, index) to `PlaceRef`.
    pub(super) fn eval_place(&mut self, scope: Scope<'_>, mut expr: ExprId, frame: &mut Frame) -> Option<PlaceRef> {
        loop {
            match *scope.body.expr(expr) {
                Expr::Paren(inner) => expr = inner,
                Expr::Block { stmts, tail: Some(inner), label: None, kind }
                    if scope.body.list(stmts).is_empty()
                        && matches!(kind, crate::body::BlockKind::Plain | crate::body::BlockKind::Expansion | crate::body::BlockKind::Unsafe) =>
                {
                    expr = inner;
                }
                _ => break,
            }
        }
        let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
        match *scope.body.expr(expr) {
            Expr::Path(_) => self.eval_path_place(scope, expr, ty, frame),
            Expr::Unary(UnOp::Deref, operand) => self.eval_deref_target(scope, expr, operand, frame),
            Expr::TupleField(base, index) => self.eval_field_place(scope, expr, base, index, frame),
            Expr::Field(base, _) => self.eval_field_place(scope, expr, base, *scope.typed.fields.get(&expr)?, frame),
            Expr::Index(base, index) => self.eval_index_place(scope, expr, base, index, frame),
            _ => None,
        }
    }

    fn eval_path_place(&mut self, scope: Scope<'_>, expr: ExprId, ty: Ty, frame: &Frame) -> Option<PlaceRef> {
        let id = match *scope.typed.res.get(&expr)? {
            Res::Local(local) => frame.locals.get(local as usize).copied().flatten()?,
            Res::Fn(item, substs) if super::super::is_static(self.lower.cx, item) => self.static_obj(item, substs, ty)?,
            _ => return None,
        };
        Some(PlaceRef { target: PtrTarget::Obj { id, addend: 0 }, ty, slice_len: None })
    }

    fn eval_deref_target(&mut self, scope: Scope<'_>, expr: ExprId, operand: ExprId, frame: &mut Frame) -> Option<PlaceRef> {
        let place = self.eval_place_or_temp(scope, operand, frame)?;
        let ty = scope.typed.expr_ty(&self.lower.cx.types, operand);
        if matches!(*self.lower.cx.types.kind(ty), Kind::Ptr(..)) && !scope.typed.derefs.contains_key(&(expr, 0)) {
            return None;
        }
        self.step_deref_place(scope, (expr, 0), &place, ty)
    }

    pub(super) fn eval_deref_read(&mut self, scope: Scope<'_>, expr: ExprId, operand: ExprId, frame: &mut Frame) -> Option<Val> {
        let place = self.eval_deref_target(scope, expr, operand, frame)?;
        self.read_place(&place)
    }

    fn step_deref_place(&mut self, scope: Scope<'_>, site: (ExprId, u32), place: &PlaceRef, ty: Ty) -> Option<PlaceRef> {
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

    pub(super) fn place_from_ref_val(&mut self, ptr_val: &Val, inner: Ty) -> Option<PlaceRef> {
        let target = ptr_val.read_ptr(0)?;
        let slice_len = if self.lower.cx.is_unsized(inner).ok()? {
            Some(u64::from_le_bytes(ptr_val.bytes.get(8..16)?.try_into().ok()?))
        } else {
            None
        };
        Some(PlaceRef { target, ty: inner, slice_len })
    }

    fn eval_autoderef_place(&mut self, scope: Scope<'_>, expr: ExprId, base: ExprId, frame: &mut Frame) -> Option<PlaceRef> {
        let mut place = self.eval_place_or_temp(scope, base, frame)?;
        let steps = scope.typed.autoderefs.get(&expr).copied().unwrap_or(0);
        for step in 0..steps {
            let cur_ty = place.ty;
            place = self.step_deref_place(scope, (expr, step), &place, cur_ty)?;
        }
        Some(place)
    }

    fn eval_field_place(&mut self, scope: Scope<'_>, expr: ExprId, base: ExprId, index: u32, frame: &mut Frame) -> Option<PlaceRef> {
        let place = self.eval_autoderef_place(scope, expr, base, frame)?;
        self.part_place_ref(&place, (0, index))
    }

    pub(super) fn part_place_ref(&mut self, place: &PlaceRef, (variant, index): (u32, u32)) -> Option<PlaceRef> {
        match *self.lower.cx.types.kind(place.ty) {
            Kind::Array(elem, _) => {
                let delta = self.lower.cx.layout(elem).ok()?.size.bytes().checked_mul(u64::from(index))?;
                Some(PlaceRef { target: place.target.offset(delta)?, ty: elem, slice_len: None })
            }
            Kind::Tuple(items) => {
                let elem = *self.lower.cx.types.substs(items).get(index as usize)?;
                let layout = self.lower.cx.layout(place.ty).ok()?;
                let delta = u64::try_from(field_offset(&layout, 0, index)).ok()?;
                Some(PlaceRef { target: place.target.offset(delta)?, ty: elem, slice_len: None })
            }
            Kind::Adt(item, args) => self.adt_part_place(place, (item, args), (variant, index)),
            _ => None,
        }
    }

    fn adt_part_place(&mut self, place: &PlaceRef, (item, args): (ItemRef, Substs), (variant, index): (u32, u32)) -> Option<PlaceRef> {
        let slice = self.lower.cx.types.substs(args).to_vec();
        let elem = *field_types(self.lower.cx, item, &slice).ok()?.get(variant as usize)?.get(index as usize)?;
        let layout = self.lower.cx.layout(place.ty).ok()?;
        let delta = u64::try_from(field_offset(&layout, variant, index)).ok()?;
        Some(PlaceRef { target: place.target.offset(delta)?, ty: elem, slice_len: None })
    }

    fn eval_index_place(&mut self, scope: Scope<'_>, expr: ExprId, base: ExprId, index: ExprId, frame: &mut Frame) -> Option<PlaceRef> {
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
        Some(PlaceRef { target: place.target.offset(stride.checked_mul(idx)?)?, ty: elem, slice_len: None })
    }

    /// Computes the adjusted receiver value of a method call.
    pub(super) fn eval_receiver(&mut self, scope: Scope<'_>, (expr, receiver): (ExprId, ExprId), adjust: Adjust, frame: &mut Frame) -> Option<Val> {
        if adjust == (Adjust { derefs: 0, autoref: None }) {
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

    pub(super) fn eval_tuple_val(&mut self, scope: Scope<'_>, list: List, ty: Ty, frame: &mut Frame) -> Option<Val> {
        let Kind::Tuple(_) = *self.lower.cx.types.kind(ty) else { return None };
        let layout = self.lower.cx.layout(ty).ok()?;
        let size = self.ty_size(ty)?;
        let id = self.alloc_val(Val::zeroed(size))?;
        for (idx, &raw) in scope.body.list(list).iter().enumerate() {
            let val = self.eval_expr_val(scope, ExprId(raw), frame)?;
            let off = u64::try_from(field_offset(&layout, 0, idx as u32)).ok()?;
            self.write_bytes(PtrTarget::Obj { id, addend: off }, &val)?;
        }
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    pub(super) fn eval_array_val(&mut self, scope: Scope<'_>, list: List, ty: Ty, frame: &mut Frame) -> Option<Val> {
        let Kind::Array(elem, _) = *self.lower.cx.types.kind(ty) else { return None };
        let stride = self.lower.cx.layout(elem).ok()?.size.bytes();
        let id = self.alloc_ty(ty)?;
        for (idx, &raw) in scope.body.list(list).iter().enumerate() {
            let val = self.eval_expr_val(scope, ExprId(raw), frame)?;
            self.write_bytes(PtrTarget::Obj { id, addend: stride.checked_mul(idx as u64)? }, &val)?;
        }
        let size = self.ty_size(ty)?;
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    pub(super) fn eval_repeat_val(&mut self, scope: Scope<'_>, item: ExprId, ty: Ty, frame: &mut Frame) -> Option<Val> {
        let Kind::Array(elem, count_ty) = *self.lower.cx.types.kind(ty) else { return None };
        let stride = self.lower.cx.layout(elem).ok()?.size.bytes();
        let count = self.lower.cx.types.len(count_ty)?;
        let val = self.eval_expr_val(scope, item, frame)?;
        let id = self.alloc_ty(ty)?;
        for idx in 0..count {
            self.write_bytes(PtrTarget::Obj { id, addend: stride.checked_mul(idx)? }, &val)?;
        }
        let size = self.ty_size(ty)?;
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    pub(super) fn eval_struct_val(&mut self, scope: Scope<'_>, (expr, fields, base): (ExprId, List, Option<ExprId>), ty: Ty, frame: &mut Frame) -> Option<Val> {
        let variant = match scope.typed.res.get(&expr) {
            Some(&Res::Ctor(_, variant, _)) => variant,
            _ => 0,
        };
        let base_val = match base {
            Some(b) => Some(self.eval_expr_val(scope, b, frame)?),
            None => None,
        };
        let mut pairs = Vec::new();
        for &raw in scope.body.list(fields) {
            let value = scope.body.fields[raw as usize].value;
            pairs.push((*scope.typed.inits.get(&value)?, self.eval_expr_val(scope, value, frame)?));
        }
        self.construct_adt(ty, variant, pairs, base_val)
    }

    /// Constructs an ADT value from field `(index, value)` pairs and optional `base`.
    pub(super) fn construct_adt(&mut self, ty: Ty, variant: u32, fields: Vec<(u32, Val)>, base: Option<Val>) -> Option<Val> {
        let Kind::Adt(item, _) = *self.lower.cx.types.kind(ty) else { return None };
        let layout = self.lower.cx.layout(ty).ok()?;
        if matches!(layout.fields, FieldsShape::Union(_)) {
            return None;
        }
        let size = self.ty_size(ty)?;
        let id = self.alloc_val(base.unwrap_or_else(|| Val::zeroed(size)))?;
        for (idx, val) in fields {
            let off = u64::try_from(field_offset(&layout, variant, idx)).ok()?;
            self.write_bytes(PtrTarget::Obj { id, addend: off }, &val)?;
        }
        self.write_adt_tag((id, item, variant), &layout)?;
        self.read_bytes(PtrTarget::Obj { id, addend: 0 }, size)
    }

    fn write_adt_tag(&mut self, (id, item, variant): (u32, ItemRef, u32), layout: &crate::sema::layout::Layout) -> Option<()> {
        if let Some((off, tag, bits)) = super::super::adt::tag_bits(self.lower.cx, item, layout, variant).ok()? {
            let tag_val = Val::scalar(tag.bytes() as usize, bits);
            self.write_bytes(PtrTarget::Obj { id, addend: u64::try_from(off).ok()? }, &tag_val)?;
        }
        Some(())
    }
}
