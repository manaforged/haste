use haste_codegen::Type;

use crate::body::literal::{byte_value, char_value};
use crate::body::{BinOp, Expr, ExprId, Lit, UnOp};
use crate::names::Primitive;
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::sema::infer::{Res, parse_int};
use crate::sema::ty::{Kind, int_info, mask};

use super::super::super::Lower;
use super::super::super::ops::parse_float;
use super::{MAX_CONST_DEPTH, Scope};

pub(in crate::lower) fn eval_lit_bits(
    src: &Source,
    lit: Lit,
    at: u32,
    primitive: Primitive,
) -> Option<(usize, u128)> {
    match lit {
        Lit::Bool(value) if primitive == Primitive::Bool => Some((1, u128::from(value))),
        Lit::Char if primitive == Primitive::Char => {
            Some((4, u128::from(u32::from(char_value(src.text(at)).ok()?))))
        }
        Lit::Byte if primitive == Primitive::U8 => {
            Some((1, u128::from(byte_value(src.text(at)).ok()?)))
        }
        Lit::Int => {
            let (bits, _) = int_info(primitive)?;
            Some((
                bits as usize / 8,
                parse_int(src.text(at)).ok()?.0 & mask(bits),
            ))
        }
        Lit::Float => float_lit_bits(primitive, src.text(at)),
        _ => None,
    }
}

pub(in crate::lower) fn float_lit_bits(primitive: Primitive, text: &str) -> Option<(usize, u128)> {
    match primitive {
        Primitive::F32 => Some((4, parse_float(Type::F32, text).ok()? as u64 as u128)),
        Primitive::F64 => Some((8, parse_float(Type::F64, text).ok()? as u64 as u128)),
        _ => None,
    }
}

pub(in crate::lower) fn neg_float_bits(primitive: Primitive, bits: u128) -> Option<(usize, u128)> {
    match primitive {
        Primitive::F32 => Some((4, (bits ^ (1u128 << 31)) & 0xFFFF_FFFF)),
        Primitive::F64 => Some((8, (bits ^ (1u128 << 63)) & 0xFFFF_FFFF_FFFF_FFFF)),
        _ => None,
    }
}

pub(super) fn prim_scalar_bytes(primitive: Primitive) -> Option<usize> {
    match primitive {
        Primitive::Bool => Some(1),
        Primitive::Char | Primitive::F32 => Some(4),
        Primitive::F64 => Some(8),
        _ => Some(int_info(primitive)?.0 as usize / 8),
    }
}

impl Lower<'_, '_> {
    fn eval_path_scalar(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        primitive: Primitive,
        depth: u32,
    ) -> Option<u128> {
        match scope.typed.res.get(&expr) {
            Some(&Res::Const(val_ty)) => match *self.cx.types.kind(val_ty) {
                Kind::Value(bits) => Some(bits),
                _ => None,
            },
            Some(&Res::Fn(item, substs))
                if depth < MAX_CONST_DEPTH && self.cx.item(item).1.kind == ItemKind::Const =>
            {
                let (source, typed) = self.literal_source(item, substs).ok()??;
                let const_typed = typed.scoped(&self.cx.types);
                if *self.cx.types.kind(const_typed.ret_ty()) != Kind::Prim(primitive) {
                    return None;
                }
                let body = self.cx.body(source).ok()?;
                let (root, src) = (body.root?, self.cx.body_src(source));
                self.eval_scalar(
                    Scope {
                        body: &body,
                        src,
                        typed: &const_typed,
                        in_const: true,
                    },
                    root,
                    primitive,
                    depth + 1,
                )
            }
            _ => None,
        }
    }

    fn eval_unary_scalar(
        &mut self,
        scope: Scope<'_>,
        (op, inner): (UnOp, ExprId),
        primitive: Primitive,
        depth: u32,
    ) -> Option<u128> {
        let inner_val = self.eval_scalar(scope, inner, primitive, depth)?;
        match op {
            UnOp::Neg if matches!(primitive, Primitive::F32 | Primitive::F64) => {
                Some(neg_float_bits(primitive, inner_val)?.1)
            }
            UnOp::Neg if int_info(primitive).is_some_and(|(_, signed)| signed) => {
                Some(inner_val.wrapping_neg())
            }
            UnOp::Not if primitive == Primitive::Bool => Some(u128::from(inner_val == 0)),
            UnOp::Not if int_info(primitive).is_some() => Some(!inner_val),
            _ => None,
        }
    }

    pub(super) fn eval_scalar(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        primitive: Primitive,
        depth: u32,
    ) -> Option<u128> {
        let expr = Self::peel_uncoerced(scope, expr);
        if depth > MAX_CONST_DEPTH || scope.typed.coercions.contains_key(&expr) {
            return None;
        }
        let value = match *scope.body.expr(expr) {
            Expr::Lit(lit, at) => eval_lit_bits(scope.src, lit, at, primitive)?.1,
            Expr::Path(_) => self.eval_path_scalar(scope, expr, primitive, depth)?,
            Expr::Unary(op, inner) if !scope.typed.operators.contains_key(&expr) => {
                self.eval_unary_scalar(scope, (op, inner), primitive, depth)?
            }
            Expr::Cast(inner, _) => {
                let Kind::Prim(src_prim) = *self
                    .cx
                    .types
                    .kind(scope.typed.expr_ty(&self.cx.types, inner))
                else {
                    return None;
                };
                return cast_scalar(
                    self.eval_scalar(scope, inner, src_prim, depth)?,
                    src_prim,
                    primitive,
                );
            }
            Expr::Binary(op, lhs, rhs) if !scope.typed.operators.contains_key(&expr) => {
                return self.eval_binary(scope, op, lhs, rhs, primitive, depth);
            }
            _ => return self.eval_const_scalar_fn(scope, expr, primitive, depth),
        };
        Some(int_info(primitive).map_or(value, |(bits, _)| value & mask(bits)))
    }

    fn eval_binary(
        &mut self,
        scope: Scope<'_>,
        op: BinOp,
        lhs: ExprId,
        rhs: ExprId,
        primitive: Primitive,
        depth: u32,
    ) -> Option<u128> {
        match op {
            BinOp::And | BinOp::Or if primitive == Primitive::Bool => {
                let l = self.eval_scalar(scope, lhs, Primitive::Bool, depth)? != 0;
                if (op == BinOp::And && !l) || (op == BinOp::Or && l) {
                    return Some(u128::from(l));
                }
                let r = self.eval_scalar(scope, rhs, Primitive::Bool, depth)? != 0;
                Some(u128::from(r))
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                if primitive == Primitive::Bool =>
            {
                let Kind::Prim(lp) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, lhs))
                else {
                    return None;
                };
                let Kind::Prim(rp) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, rhs))
                else {
                    return None;
                };
                if lp != rp {
                    return None;
                }
                let (bits, signed) = prim_info(lp)?;
                let l = self.eval_scalar(scope, lhs, lp, depth)?;
                let r = self.eval_scalar(scope, rhs, rp, depth)?;
                Some(u128::from(cmp_scalar(op, l, r, (bits, signed))))
            }
            BinOp::Shl | BinOp::Shr => {
                let Kind::Prim(rp) = *self.cx.types.kind(scope.typed.expr_ty(&self.cx.types, rhs))
                else {
                    return None;
                };
                let rinfo = int_info(rp)?;
                let l = self.eval_scalar(scope, lhs, primitive, depth)?;
                let r = self.eval_scalar(scope, rhs, rp, depth)?;
                apply_shift(op, primitive, (l, r), rinfo)
            }
            BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Rem => {
                let l = self.eval_scalar(scope, lhs, primitive, depth)?;
                let r = self.eval_scalar(scope, rhs, primitive, depth)?;
                apply_int_binary(op, primitive, l, r)
            }
            _ => None,
        }
    }
}

pub(in super::super) fn apply_shift(
    op: BinOp,
    primitive: Primitive,
    (l, r): (u128, u128),
    (rbits, rsigned): (u32, bool),
) -> Option<u128> {
    let (bits, signed) = int_info(primitive)?;
    let shift = if rsigned {
        u32::try_from(sign_extend(r, rbits)).ok()
    } else {
        u32::try_from(r).ok()
    }?;
    if shift >= bits {
        return None;
    }
    match op {
        BinOp::Shl => Some((l << shift) & mask(bits)),
        _ if signed => Some(((sign_extend(l, bits) >> shift) as u128) & mask(bits)),
        _ => Some((l & mask(bits)) >> shift),
    }
}

pub(in super::super) fn apply_int_binary(
    op: BinOp,
    primitive: Primitive,
    l: u128,
    r: u128,
) -> Option<u128> {
    if matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor) {
        let (bits, _) = if primitive == Primitive::Bool {
            (1, false)
        } else {
            int_info(primitive)?
        };
        let out = match op {
            BinOp::BitAnd => l & r,
            BinOp::BitOr => l | r,
            _ => l ^ r,
        };
        return Some(out & mask(bits));
    }
    let (bits, signed) = int_info(primitive)?;
    if signed {
        apply_checked_signed(op, bits, l, r)
    } else {
        apply_checked_unsigned(op, bits, l, r)
    }
}

fn apply_checked_signed(op: BinOp, bits: u32, l: u128, r: u128) -> Option<u128> {
    let (sa, sb) = (sign_extend(l, bits), sign_extend(r, bits));
    let res = match op {
        BinOp::Add => sa.checked_add(sb)?,
        BinOp::Sub => sa.checked_sub(sb)?,
        BinOp::Mul => sa.checked_mul(sb)?,
        BinOp::Div => sa.checked_div(sb)?,
        BinOp::Rem => sa.checked_rem(sb)?,
        _ => return None,
    };
    check_signed(res, bits)
}

fn apply_checked_unsigned(op: BinOp, bits: u32, l: u128, r: u128) -> Option<u128> {
    let res = match op {
        BinOp::Add => l.checked_add(r)?,
        BinOp::Sub => l.checked_sub(r)?,
        BinOp::Mul => l.checked_mul(r)?,
        BinOp::Div => l.checked_div(r)?,
        BinOp::Rem => l.checked_rem(r)?,
        _ => return None,
    };
    (res <= mask(bits)).then_some(res)
}

pub(in super::super) fn prim_info(prim: Primitive) -> Option<(u32, bool)> {
    match prim {
        Primitive::Bool => Some((1, false)),
        Primitive::Char => Some((32, false)),
        _ => int_info(prim),
    }
}

pub(in super::super) fn sign_extend(val: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((val << shift) as i128) >> shift
}

fn check_signed(res: i128, bits: u32) -> Option<u128> {
    if bits == 128 {
        return Some(res as u128);
    }
    let min = -(1i128 << (bits - 1));
    let max = (1i128 << (bits - 1)) - 1;
    (res >= min && res <= max).then_some((res as u128) & mask(bits))
}

pub(in super::super) fn cast_scalar(
    src_val: u128,
    src_prim: Primitive,
    dst_prim: Primitive,
) -> Option<u128> {
    if dst_prim == Primitive::Char {
        return matches!(src_prim, Primitive::U8 | Primitive::Char)
            .then_some(src_val & 0xFFFF_FFFF);
    }
    let (dst_bits, _) = int_info(dst_prim)?;
    let (src_bits, src_signed) = prim_info(src_prim)?;
    let bits = if src_signed {
        sign_extend(src_val, src_bits) as u128
    } else {
        src_val
    };
    Some(bits & mask(dst_bits))
}

pub(in super::super) fn cmp_scalar(
    op: BinOp,
    l: u128,
    r: u128,
    (bits, signed): (u32, bool),
) -> bool {
    if signed {
        let (sa, sb) = (sign_extend(l, bits), sign_extend(r, bits));
        match op {
            BinOp::Eq => sa == sb,
            BinOp::Ne => sa != sb,
            BinOp::Lt => sa < sb,
            BinOp::Le => sa <= sb,
            BinOp::Gt => sa > sb,
            _ => sa >= sb,
        }
    } else {
        match op {
            BinOp::Eq => l == r,
            BinOp::Ne => l != r,
            BinOp::Lt => l < r,
            BinOp::Le => l <= r,
            BinOp::Gt => l > r,
            _ => l >= r,
        }
    }
}
