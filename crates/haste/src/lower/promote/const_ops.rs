mod aggregate;
mod place;

use crate::body::literal::{byte_value, char_value, string_bytes};
use crate::body::{BinOp, ExprId, Lit, UnOp};
use crate::names::{ItemRef, Primitive};
use crate::outline::item::ItemKind;
use crate::sema::infer::{Coercion, Res, parse_int};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, int_info, mask};

use super::const_val::{ConstEnv, Frame, PlaceRef, PtrTarget, Val};
use super::literal::{Scope, apply_int_binary, apply_shift, cast_scalar, cmp_scalar, prim_info};

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    /// Applies any inference coercion recorded on `expr` to `val`.
    pub(super) fn apply_coercion(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        val: Val,
    ) -> Option<Val> {
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
            Coercion::Reify(Res::Fn(item, substs)) => Some(Val::ptr(PtrTarget::Fn(
                self.lower.rodata_fn_target(*item, *substs)?,
            ))),
            _ => None,
        }
    }

    fn apply_deref_coercion(&mut self, steps: &[(Ty, Option<Res>)], mut val: Val) -> Option<Val> {
        for (ty, res) in steps {
            val = match res {
                Some(Res::Fn(item, substs)) => self.call_fn(*item, *substs, vec![val])?,
                Some(_) => return None,
                None => {
                    let inner = match *self.lower.cx.types.kind(*ty) {
                        Kind::Ref(_, inner) | Kind::Ptr(_, inner) => inner,
                        _ => return None,
                    };
                    let place = self.place_from_ref_val(&val, inner)?;
                    self.read_place(&place)?
                }
            };
        }
        Some(val)
    }

    /// Evaluates a literal expression to a `Val`.
    pub(super) fn eval_lit_val(
        &mut self,
        scope: Scope<'_>,
        lit: Lit,
        at: u32,
        ty: Ty,
    ) -> Option<Val> {
        let text = scope.src.text(at);
        if matches!(lit, Lit::Str | Lit::ByteStr | Lit::CStr) {
            return self.eval_str_lit(lit, text);
        }
        let Kind::Prim(prim) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
        self.eval_prim_lit(lit, prim, text, ty)
    }

    fn eval_str_lit(&mut self, lit: Lit, text: &str) -> Option<Val> {
        let mut bytes = string_bytes(text).ok()?;
        if lit == Lit::CStr {
            bytes.push(0);
        }
        let len = bytes.len() as u64;
        let id = self.alloc_val(Val {
            bytes,
            ptrs: Vec::new(),
        })?;
        let target = PtrTarget::Obj { id, addend: 0 };
        Some(if lit == Lit::ByteStr {
            Val::ptr(target)
        } else {
            Val::wide(target, len)
        })
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
    pub(super) fn eval_path_val(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        ty: Ty,
        frame: &Frame,
    ) -> Option<Val> {
        match *scope.typed.res.get(&expr)? {
            Res::Local(local) => {
                let id = frame.locals.get(local as usize).copied().flatten()?;
                self.read_place(&PlaceRef {
                    target: PtrTarget::Obj { id, addend: 0 },
                    ty,
                    slice_len: None,
                })
            }
            Res::Const(val_ty) => {
                let Kind::Value(bits) = *self.lower.cx.types.kind(val_ty) else {
                    return None;
                };
                Some(Val::scalar(self.ty_size(ty)?, bits))
            }
            Res::Ctor(_, variant, _) => (!matches!(*self.lower.cx.types.kind(ty), Kind::FnDef(..)))
                .then(|| self.construct_adt(ty, variant, Vec::new(), None))
                .flatten(),
            Res::Fn(item, substs) => match self.lower.cx.item(item).1.kind {
                ItemKind::Static => {
                    let id = self.static_obj(item, substs, ty)?;
                    self.read_place(&PlaceRef {
                        target: PtrTarget::Obj { id, addend: 0 },
                        ty,
                        slice_len: None,
                    })
                }
                ItemKind::Const => self.eval_const_item_val(item, substs, ty),
                ItemKind::Fn if matches!(*self.lower.cx.types.kind(ty), Kind::FnDef(..)) => {
                    Some(Val::zst())
                }
                _ => None,
            },
        }
    }

    /// Evaluates an associated or free `const` item to a `Val`.
    pub(super) fn eval_const_item_val(
        &mut self,
        item: ItemRef,
        substs: Substs,
        ty: Ty,
    ) -> Option<Val> {
        let (source, typed) = self.const_body_source(item, substs)?;
        let typed = typed.scoped(&self.lower.cx.types);
        if typed.ret_ty() != ty {
            return None;
        }
        let body = self.lower.cx.body(source).ok()?;
        let root = body.root?;
        let src = self.lower.cx.body_src(source);
        self.depth += 1;
        let mut frame = Frame {
            locals: vec![None; typed.locals.len()],
        };
        let out = self.eval_expr_val(
            Scope {
                body: &body,
                src,
                typed: &typed,
                in_const: true,
            },
            root,
            &mut frame,
        );
        self.depth -= 1;
        out
    }

    /// Evaluates a unary expression other than deref.
    pub(super) fn eval_unary_val(
        &mut self,
        scope: Scope<'_>,
        (expr, op, operand): (ExprId, UnOp, ExprId),
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.operators.get(&expr) {
            let arg = self.eval_expr_val(scope, operand, frame)?;
            return self.call_fn(item, substs, vec![arg]);
        }
        let Kind::Prim(prim) = *self.lower.cx.types.kind(ty) else {
            return None;
        };
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
    pub(super) fn eval_binary_val(
        &mut self,
        scope: Scope<'_>,
        (expr, op, lhs, rhs): (ExprId, BinOp, ExprId, ExprId),
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
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

    fn eval_short_circuit(
        &mut self,
        scope: Scope<'_>,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        frame: &mut Frame,
    ) -> Option<Val> {
        let l = self.eval_expr_val(scope, lhs, frame)?.as_u128()? != 0;
        if (op == BinOp::And && !l) || (op == BinOp::Or && l) {
            return Some(Val::scalar(1, u128::from(l)));
        }
        let r = self.eval_expr_val(scope, rhs, frame)?.as_u128()? != 0;
        Some(Val::scalar(1, u128::from(r)))
    }

    fn apply_scalar_binop(
        &mut self,
        op: BinOp,
        (lval, lty): (u128, Ty),
        (rval, rty): (u128, Ty),
        res_ty: Ty,
    ) -> Option<Val> {
        let (Kind::Prim(lprim), Kind::Prim(rprim)) = (
            self.lower.cx.types.kind(lty).clone(),
            self.lower.cx.types.kind(rty).clone(),
        ) else {
            return None;
        };
        let bits = match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                if lprim == rprim =>
            {
                u128::from(cmp_scalar(op, lval, rval, prim_info(lprim)?))
            }
            BinOp::Shl | BinOp::Shr => {
                let Kind::Prim(dst_prim) = *self.lower.cx.types.kind(res_ty) else {
                    return None;
                };
                apply_shift(op, dst_prim, (lval, rval), int_info(rprim)?)?
            }
            _ => {
                let Kind::Prim(dst_prim) = *self.lower.cx.types.kind(res_ty) else {
                    return None;
                };
                apply_int_binary(op, dst_prim, lval, rval)?
            }
        };
        Some(Val::scalar(self.ty_size(res_ty)?, bits))
    }

    /// Evaluates a primitive cast expression.
    pub(super) fn eval_cast_val(
        &mut self,
        scope: Scope<'_>,
        inner: ExprId,
        dst_ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        let src_ty = scope.typed.expr_ty(&self.lower.cx.types, inner);
        if self.lower.cx.is_unsized(dst_ty).ok()? {
            return None;
        }
        let (src_kind, dst_kind) = (
            self.lower.cx.types.kind(src_ty).clone(),
            self.lower.cx.types.kind(dst_ty).clone(),
        );
        match (src_kind, dst_kind) {
            (Kind::FnDef(item, substs), Kind::FnPtr(..)) => Some(Val::ptr(PtrTarget::Fn(
                self.lower.rodata_fn_target(item, substs)?,
            ))),
            (Kind::Prim(_), Kind::Ptr(_, _)) => Some(Val::scalar(
                8,
                self.eval_expr_val(scope, inner, frame)?.as_u128()?,
            )),
            (Kind::Ptr(_, _) | Kind::Ref(_, _), Kind::Ptr(_, _))
                if !self.lower.cx.is_unsized(src_ty).ok()? =>
            {
                self.eval_expr_val(scope, inner, frame)
            }
            (Kind::Prim(src_prim), Kind::Prim(dst_prim)) => {
                let src_val = self.eval_expr_val(scope, inner, frame)?.as_u128()?;
                let out = cast_scalar(src_val, src_prim, dst_prim)?;
                Some(Val::scalar(self.ty_size(dst_ty)?, out))
            }
            _ => None,
        }
    }

    /// Evaluates simple assignment `lhs = rhs`.
    pub(super) fn eval_assign_val(
        &mut self,
        scope: Scope<'_>,
        (lhs, rhs): (ExprId, ExprId),
        frame: &mut Frame,
    ) -> Option<Val> {
        let val = self.eval_expr_val(scope, rhs, frame)?;
        let place = self.eval_place_mut(scope, lhs, frame)?;
        self.write_bytes(place.target, &val)?;
        Some(Val::zst())
    }

    /// Evaluates compound assignment `lhs op= rhs`.
    pub(super) fn eval_compound_val(
        &mut self,
        scope: Scope<'_>,
        (expr, op, lhs, rhs): (ExprId, BinOp, ExprId, ExprId),
        frame: &mut Frame,
    ) -> Option<Val> {
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
}
