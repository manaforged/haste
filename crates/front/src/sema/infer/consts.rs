use anyhow::{Result, bail};

use crate::body::{BinOp, Body, Expr, ExprId, Lit, PathId, TyId, UnOp};
use crate::names::{Def, ItemRef, Primitive, Sym};
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Ty, int_info};
use super::super::intrinsic::Intrinsic;
use super::{Infer, Res};
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
    pub(super) fn const_value(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(u128, Int)> {
        let sig = self.cx.signature(item)?;
        let Some(ret) = sig.ret else { bail!("constant {} has no type", self.cx.name(item)) };
        let body = self.cx.body(item)?;
        let Some(root) = body.root else { bail!("constant {} has no value", self.cx.name(item)) };
        let mut frame = self.frame(item, substs)?;
        (frame.src, frame.root) = (self.cx.body_src(item), Some(item.body_owner()));
        self.with_frame(frame, |infer| {
            let ty = infer.lower_ty(&sig.body, ret)?;
            let Kind::Prim(primitive) = *infer.cx.types.kind(ty) else { bail!("constant of a type that is not an integer") };
            let Some(int) = int_info(primitive) else { bail!("constant of a type that is not an integer") };
            let (value, _) = infer.const_eval(&body, root, int)?;
            Ok((value & mask(int.0), int))
        })
    }

    fn variant_value(&mut self, item: ItemRef, index: u32) -> Result<(u128, Int)> {
        let at = self.cx.variant_position(item, index)?;
        if !self.cx.adt_def(item)?.variants[at].fields.is_empty() {
            bail!("a constant variant of {} that has fields", self.cx.name(item));
        }
        let Some(int) = int_info(self.cx.discriminant_primitive(item)) else { bail!("an enum tag that is not an integer") };
        let value = self.cx.discriminants(item)?[at];
        Ok((value as u128 & mask(int.0), int))
    }

    /// The discriminant of each variant of the enum `item`: its written value read as `primitive`, or one more than the previous.
    pub(super) fn discriminants(&mut self, item: ItemRef, primitive: Primitive) -> Result<Vec<i128>> {
        let def = self.cx.adt_def(item)?;
        let Some(int) = int_info(primitive) else { bail!("an enum tag that is not an integer") };
        let mut next = 0_i128;
        let mut values = Vec::with_capacity(def.variants.len());
        for variant in &def.variants {
            if let Some(expr) = variant.discriminant {
                let (value, _) = self.const_eval(&def.body, expr, int)?;
                next = if int.1 { signed(value & mask(int.0), int.0) } else { (value & mask(int.0)) as i128 };
            }
            values.push(next);
            next = next.wrapping_add(1);
        }
        Ok(values)
    }

    /// The const value that `expr` of `arena` gives as a generic argument or an array length: a const parameter of the frame as it stands, or the value it evaluates to.
    pub(super) fn const_arg(&mut self, arena: &Body, expr: ExprId) -> Result<Ty> {
        if let Some(param) = self.named_param(arena, expr) {
            return Ok(param);
        }
        let (value, (bits, _)) = self.const_eval(arena, expr, (64, false))?;
        Ok(self.cx.types.intern(Kind::Value(value & mask(bits))))
    }

    fn named_param(&self, arena: &Body, expr: ExprId) -> Option<Ty> {
        match *arena.expr(expr) {
            Expr::Paren(inner) => self.named_param(arena, inner),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.named_param(arena, tail),
            Expr::Path(path) => {
                let entry = arena.path(path);
                let [segment] = arena.list(entry.segments) else { return None };
                let name = self.src.text(arena.segments[*segment as usize].name);
                let named = entry.qself.is_none() && arena.segments[*segment as usize].args.is_none();
                named.then(|| self.generics.iter().find(|(generic, _)| *generic == name).map(|&(_, ty)| ty)).flatten()
            }
            _ => None,
        }
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
            Expr::Call(callee, args) if arena.list(args).is_empty() => self.const_call(arena, callee, want),
            Expr::Path(path) => self.const_name(arena, (expr, path), want),
            other => bail!("constant expressions like {other:?} are not evaluated yet"),
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

    fn const_name(&mut self, arena: &Body, (expr, path): (ExprId, PathId), want: Int) -> Result<(u128, Int)> {
        let Some(param) = self.named_param(arena, expr) else { return self.const_path(arena, path) };
        match *self.cx.types.kind(self.resolve(param)) {
            Kind::Value(bits) => Ok((bits, want)),
            _ => bail!("a const parameter whose value is not known yet"),
        }
    }

    fn const_path(&mut self, arena: &Body, path: PathId) -> Result<(u128, Int)> {
        let named = match arena.path(path).qself {
            Some(_) => None,
            None => {
                let (segments, absolute, dollar) = self.segments(arena, path);
                Some(self.const_named(&segments, absolute, dollar))
            }
        };
        match named {
            Some(Ok(found)) => Ok(found),
            Some(Err(error)) => self.const_member(arena, path).map_err(|_| error),
            None => self.const_member(arena, path),
        }
    }

    fn const_member(&mut self, arena: &Body, path: PathId) -> Result<(u128, Int)> {
        let (item, substs) = self.const_target(arena, path)?;
        match self.cx.item(item).1.kind {
            ItemKind::Const => self.const_value(item, &substs),
            other => bail!("a {other:?} used as a constant"),
        }
    }

    fn const_call(&mut self, arena: &Body, callee: ExprId, want: Int) -> Result<(u128, Int)> {
        let Expr::Path(path) = *arena.expr(callee) else { bail!("a constant call of something that is not a path") };
        let (item, substs) = self.const_target(arena, path)?;
        let layout_of = |infer: &mut Self| match substs.first() {
            Some(&ty) => infer.cx.layout(ty),
            None => bail!("a layout intrinsic with no type"),
        };
        match self.cx.intrinsic(item) {
            Some(Intrinsic::SizeOf) => Ok((u128::from(layout_of(self)?.size.bytes()), (64, false))),
            Some(Intrinsic::AlignOf) => Ok((u128::from(layout_of(self)?.align.abi.bytes()), (64, false))),
            Some(other) => bail!("the intrinsic {other:?} in a constant is not evaluated yet"),
            None => self.const_body(item, &substs, want),
        }
    }

    fn const_target(&mut self, arena: &Body, path: PathId) -> Result<(ItemRef, Vec<Ty>)> {
        let (_, res) = self.path_value(arena, path)?;
        let Res::Fn(item, substs) = res else { bail!("a constant path that names no item") };
        let substs = substs.into_iter().map(|ty| self.zonk(ty)).collect();
        match self.trait_call(item, substs)? {
            Res::Fn(item, substs) => Ok((item, substs)),
            _ => bail!("a constant path that names no item"),
        }
    }

    fn const_body(&mut self, item: ItemRef, substs: &[Ty], want: Int) -> Result<(u128, Int)> {
        let body = self.cx.body(item)?;
        let Some(root) = body.root else { bail!("{} has no body to evaluate", self.cx.name(item)) };
        let mut frame = self.frame(item, substs)?;
        (frame.src, frame.root) = (self.cx.body_src(item), Some(item.body_owner()));
        self.with_frame(frame, |infer| infer.const_eval(&body, root, want))
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
            Some(Def::Item(item)) if self.cx.item(item).1.kind == ItemKind::Const => self.const_value(item, &[]),
            Some(Def::Variant(item, index)) => self.variant_value(item, index),
            _ => bail!("constant `{}` is not an integer constant item", self.spell(segments)),
        }
    }
}
