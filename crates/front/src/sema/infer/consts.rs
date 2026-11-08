use anyhow::{Result, bail};

use crate::body::{BinOp, Body, Expr, ExprId, Lit, PathId, TyId, UnOp};
use crate::names::{Def, ItemRef, Primitive, Sym};
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, int_info};
use super::Infer;
use super::path::parse_int;

/// The width and signedness of an integer type.
type Int = (u32, bool);

fn mask(bits: u32) -> u128 {
    if bits == 128 { u128::MAX } else { (1 << bits) - 1 }
}

fn signed(value: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((value << shift) as i128) >> shift
}

fn fold(op: BinOp, x: u128, y: u128, (bits, is_signed): Int) -> Result<u128> {
    if matches!(op, BinOp::Div | BinOp::Rem) && y == 0 {
        bail!("a division by zero in a constant");
    }
    Ok(match op {
        BinOp::Div | BinOp::Rem | BinOp::Shr if is_signed => fold_signed(op, signed(x, bits), y, bits),
        BinOp::Div | BinOp::Rem | BinOp::Shr => fold_unsigned(op, x, y),
        BinOp::Add => x.wrapping_add(y),
        BinOp::Sub => x.wrapping_sub(y),
        BinOp::Mul => x.wrapping_mul(y),
        BinOp::Shl => x.checked_shl(y as u32).unwrap_or(0),
        BinOp::BitAnd => x & y,
        BinOp::BitOr => x | y,
        BinOp::BitXor => x ^ y,
        other => bail!("the operator {other:?} in a constant is not evaluated yet"),
    })
}

fn fold_signed(op: BinOp, x: i128, y: u128, bits: u32) -> u128 {
    match op {
        BinOp::Div => x.wrapping_div(signed(y, bits)) as u128,
        BinOp::Rem => x.wrapping_rem(signed(y, bits)) as u128,
        _ => (x >> (y as u32).min(127)) as u128,
    }
}

fn fold_unsigned(op: BinOp, x: u128, y: u128) -> u128 {
    match op {
        BinOp::Div => x / y,
        BinOp::Rem => x % y,
        _ => x.checked_shr(y as u32).unwrap_or(0),
    }
}

impl Infer<'_, '_> {
    /// The value of the integer constant item `item`, as the bits of its type.
    pub(super) fn const_value(&mut self, item: ItemRef) -> Result<(u128, Int)> {
        let sig = self.cx.signature(item)?;
        let Some(ret) = sig.ret else { bail!("constant {} has no type", self.cx.name(item)) };
        let body = self.cx.body(item)?;
        let Some(root) = body.root else { bail!("constant {} has no value", self.cx.name(item)) };
        let mut frame = self.frame(item, &[])?;
        frame.src = self.cx.body_src(item);
        self.with_frame(frame, |infer| {
            let ty = infer.lower_ty(&sig.body, ret)?;
            let Kind::Prim(primitive) = *infer.cx.types.kind(ty) else { bail!("constant of a type that is not an integer") };
            let Some(int) = int_info(primitive) else { bail!("constant of a type that is not an integer") };
            let (value, _) = infer.const_eval(&body, root, int)?;
            Ok((value & mask(int.0), int))
        })
    }

    /// The value of the integer constant expression `expr` of `arena`, read as `want` unless it says its own type.
    pub(super) fn const_eval(&mut self, arena: &Body, expr: ExprId, want: Int) -> Result<(u128, Int)> {
        match *arena.expr(expr) {
            Expr::Lit(Lit::Int, at) => {
                let (value, suffix) = parse_int(self.src.text(at))?;
                Ok((value, suffix.and_then(int_info).unwrap_or(want)))
            }
            Expr::Paren(inner) => self.const_eval(arena, inner, want),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.const_eval(arena, tail, want),
            Expr::Unary(op, inner) => self.const_unary(arena, op, inner, want),
            Expr::Binary(op, lhs, rhs) => self.const_binary(arena, op, (lhs, rhs), want),
            Expr::Cast(inner, ty) => self.const_cast(arena, inner, ty),
            Expr::Path(path) => self.const_path(arena, path),
            _ => bail!("constant expressions of this kind are not evaluated yet"),
        }
    }

    fn const_unary(&mut self, arena: &Body, op: UnOp, inner: ExprId, want: Int) -> Result<(u128, Int)> {
        let (value, int) = self.const_eval(arena, inner, want)?;
        let value = match op {
            UnOp::Neg => value.wrapping_neg(),
            UnOp::Not => !value,
            UnOp::Deref => bail!("a dereference in a constant"),
        };
        Ok((value & mask(int.0), int))
    }

    fn const_cast(&mut self, arena: &Body, inner: ExprId, ty: TyId) -> Result<(u128, Int)> {
        let target = self.lower_ty(arena, ty)?;
        let Kind::Prim(primitive) = *self.cx.types.kind(target) else { bail!("a cast to a type that is not an integer") };
        let Some(int) = int_info(primitive) else { bail!("a cast to a type that is not an integer") };
        let (value, from) = self.const_eval(arena, inner, int)?;
        let value = if from.1 { signed(value, from.0) as u128 } else { value };
        Ok((value & mask(int.0), int))
    }

    fn const_binary(&mut self, arena: &Body, op: BinOp, (lhs, rhs): (ExprId, ExprId), want: Int) -> Result<(u128, Int)> {
        let (x, int) = self.const_eval(arena, lhs, want)?;
        let shift = matches!(op, BinOp::Shl | BinOp::Shr);
        let (y, _) = self.const_eval(arena, rhs, if shift { (32, false) } else { int })?;
        Ok((fold(op, x, y, int)? & mask(int.0), int))
    }

    fn const_path(&mut self, arena: &Body, path: PathId) -> Result<(u128, Int)> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        self.const_named(&segments, absolute, dollar)
    }

    /// The value of the integer constant that the path `segments` names: an associated constant of an integer type, or a constant item.
    pub(super) fn const_named(&mut self, segments: &[Sym], absolute: bool, dollar: Option<u32>) -> Result<(u128, Int)> {
        let text = |at: usize| self.cx.names.symbols.text(segments[at]);
        if let [_, _] = segments[..]
            && let Some(int) = Primitive::named(text(0)).and_then(int_info)
        {
            let (bits, is_signed) = int;
            return match (text(1), is_signed) {
                ("MAX", false) => Ok((mask(bits), int)),
                ("MAX", true) => Ok((mask(bits) >> 1, int)),
                ("MIN", false) => Ok((0, int)),
                ("MIN", true) => Ok((1 << (bits - 1), int)),
                ("BITS", _) => Ok((u128::from(bits), (32, false))),
                (other, _) => bail!("the integer constant `{other}` is not known"),
            };
        }
        let def = self.value_def(segments, absolute, dollar).ok();
        match def {
            Some(Def::Item(item)) if self.cx.item(item).1.kind == ItemKind::Const => self.const_value(item),
            _ => bail!("constant `{}` is not an integer constant item", self.spell(segments)),
        }
    }
}
