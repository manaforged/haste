use anyhow::{Result, anyhow, bail};

use crate::body::{BinOp, Body, Expr, ExprId, List, Lit, PathId, TyId, UnOp};
use crate::names::{Def, ItemRef};
use crate::outline::item::ItemKind;

use super::super::super::evaluate::{
    CONST_METHODS, ConstMethod, Int, OVERFLOW_OPS, OverflowOp, const_arithmetic_error, convert,
    fold, int_method, is_signed_min, overflowing_op, signed,
};
use super::super::super::intrinsic::Intrinsic;
use super::super::super::ty::{Kind, Ty, int_info, mask};
use super::super::path::parse_int;
use super::super::{Infer, Res};
use super::{BOOL, USIZE};

const U32: Int = (32, false);

const I32: Int = (32, true);

impl<'a> Infer<'_, 'a> {
    pub(in super::super) fn const_eval(
        &mut self,
        arena: &Body,
        expr: ExprId,
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        match *arena.expr(expr) {
            Expr::Lit(lit, at) => self.const_lit(lit, at, want),
            Expr::Paren(_) | Expr::Block { .. } | Expr::If(..) => {
                self.const_flow(arena, expr, want)
            }
            Expr::Unary(op, inner) => self.const_unary(arena, op, inner, want),
            Expr::Binary(op, lhs, rhs) => self.const_binary(arena, op, (lhs, rhs), want),
            Expr::Cast(inner, ty) => self.const_cast(arena, inner, ty),
            Expr::Call(callee, args) => self.const_call(arena, (callee, args), want),
            Expr::Path(path) => self.const_name(arena, (expr, path), want),
            Expr::TupleField(base, index) => self.const_tuple_field(arena, base, index, want),
            Expr::Method {
                receiver,
                name,
                args: None,
                params,
            } => self.const_method(arena, (receiver, name, params), want),
            Expr::Macro { path, open } => self.const_macro(arena, (path, open), want),
            other => bail!("constant expressions like {other:?} are not evaluated yet"),
        }
    }

    fn const_tuple_field(
        &mut self,
        arena: &Body,
        base: ExprId,
        index: u32,
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        match *arena.expr(base) {
            Expr::Paren(inner) => self.const_tuple_field(arena, inner, index, want),
            Expr::Block {
                stmts,
                tail: Some(tail),
                ..
            } if arena.list(stmts).is_empty() => self.const_tuple_field(arena, tail, index, want),
            Expr::Tuple(items) => {
                let &item = arena
                    .list(items)
                    .get(index as usize)
                    .ok_or_else(|| anyhow!("no field {index} on tuple constant"))?;
                self.const_eval(arena, ExprId(item), want)
            }
            Expr::Method {
                receiver,
                name,
                args: None,
                params,
            } => self.const_overflowing(arena, (receiver, name, params), index, want),
            Expr::Path(path) => self.const_path_field(arena, path, index, want),
            other => bail!("constant tuple field on {other:?} is not evaluated yet"),
        }
    }

    fn const_overflowing(
        &mut self,
        arena: &Body,
        (receiver, name, params): (ExprId, u32, List),
        index: u32,
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let text = crate::names::plain(self.src.text(name));
        let Some(&(_, op)) = OVERFLOW_OPS.iter().find(|(known, _)| *known == text) else {
            bail!("the method `{text}` is not evaluated in constants yet");
        };
        let &[rhs] = arena.list(params) else {
            bail!("`{text}` takes one operand");
        };
        let hint = (index == 0).then_some(want).flatten();
        let (x, lt) = self.const_eval(arena, receiver, hint)?;
        let int = lt.or(hint).unwrap_or(I32);
        let y = match op {
            OverflowOp::Shl | OverflowOp::Shr => self.const_defaulted(arena, ExprId(rhs), I32)?.0,
            _ => convert(self.const_defaulted(arena, ExprId(rhs), int)?.0, int, int),
        };
        let (val, over) = overflowing_op(op, x, y, int);
        match index {
            0 => Ok((val & mask(int.0), Some(int))),
            1 => Ok((u128::from(over), Some(BOOL))),
            _ => bail!("no field {index} on `{text}` result"),
        }
    }

    fn const_path_field(
        &mut self,
        arena: &Body,
        path: PathId,
        index: u32,
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let (item, substs) = self.const_target(arena, path)?;
        let body = self.cx.body(item)?;
        let frame = self.frame(item, &substs)?;
        self.with_frame(frame, |infer| {
            let root = body.root.ok_or_else(|| anyhow!("constant has no body"))?;
            let items = match *body.expr(root) {
                Expr::Tuple(items) | Expr::Call(_, items) => items,
                _ => bail!("constant tuple field on non-tuple constant"),
            };
            let &elem = body
                .list(items)
                .get(index as usize)
                .ok_or_else(|| anyhow!("no field {index} on constant"))?;
            infer.const_eval(&body, ExprId(elem), want)
        })
    }

    fn const_lit(&self, lit: Lit, at: u32, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let text = self.src.text(at);
        match lit {
            Lit::Int => {
                let (value, suffix) = parse_int(text)?;
                let ty = suffix.and_then(int_info).or(want);
                Ok((ty.map_or(value, |(bits, _)| value & mask(bits)), ty))
            }
            Lit::Bool(value) => Ok((u128::from(value), Some(BOOL))),
            Lit::Byte => Ok((
                u128::from(crate::body::literal::byte_value(text)?),
                Some((8, false)),
            )),
            Lit::Char => Ok((
                u128::from(crate::body::literal::char_value(text)?),
                Some((32, false)),
            )),
            other => bail!("constant {other:?} literals are not evaluated yet"),
        }
    }

    fn const_flow(
        &mut self,
        arena: &Body,
        expr: ExprId,
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        match *arena.expr(expr) {
            Expr::Paren(inner) => self.const_eval(arena, inner, want),
            Expr::Block {
                stmts,
                tail: Some(tail),
                ..
            } if arena.list(stmts).is_empty() => self.const_eval(arena, tail, want),
            Expr::If(cond, then, Some(otherwise)) => {
                let (test, _) = self.const_typed(arena, cond, BOOL)?;
                self.const_eval(arena, if test != 0 { then } else { otherwise }, want)
            }
            _ => bail!(
                "constant blocks with statements and `if` with no `else` are not evaluated yet"
            ),
        }
    }

    fn const_macro(
        &mut self,
        arena: &Body,
        call: (PathId, u32),
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let (output, inner, root) = self.macro_expr(arena, call)?;
        self.in_source(output, |infer| infer.const_eval(&inner, root, want))
    }

    fn macro_expr(
        &mut self,
        arena: &Body,
        (path, open): (PathId, u32),
    ) -> Result<(&'a crate::outline::source::Source, Body, ExprId)> {
        let (segments, found) = self.resolve_macro(arena, path);
        let Some(output) = self.expand_macro_call(found, open)? else {
            bail!(
                "constant macro `{}` is not supported",
                self.spell(&segments)
            );
        };
        let config = crate::body::Config {
            calls: None,
            env: Some(&self.cx.outline.crates[self.home.krate as usize].env),
        };
        let inner = crate::body::parse_expr(output, 0, output.len(), config)?;
        let Some(root) = inner.root else {
            bail!(
                "macro `{}` wrote no constant expression",
                self.spell(&segments)
            );
        };
        Ok((output, inner, root))
    }

    fn const_method(
        &mut self,
        arena: &Body,
        (receiver, name, params): (ExprId, u32, List),
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let text = crate::names::plain(self.src.text(name));
        let Some(&(_, method)) = CONST_METHODS.iter().find(|(known, _)| *known == text) else {
            bail!("the method `{text}` is not evaluated in constants yet");
        };
        if method == ConstMethod::Len {
            return Ok((self.const_len(arena, receiver)?, Some(USIZE)));
        }
        let fallback = if method == ConstMethod::Pow {
            want.unwrap_or(I32)
        } else {
            I32
        };
        let (x, int) = self.const_defaulted(arena, receiver, fallback)?;
        let wanted = if method == ConstMethod::Pow { U32 } else { int };
        let mut args = Vec::new();
        for &arg in arena.list(params) {
            args.push(self.const_typed(arena, ExprId(arg), wanted)?.0);
        }
        int_method(method, x, &args, int).map(|(value, ty)| (value, Some(ty)))
    }

    fn const_unary(
        &mut self,
        arena: &Body,
        op: UnOp,
        inner: ExprId,
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let (value, ty) = self.const_eval(arena, inner, want)?;
        let value = match op {
            UnOp::Neg => {
                if let Some((bits, true)) = ty {
                    if matches!(arena.expr(inner), Expr::Lit(..)) {
                        if value > (1_u128 << (bits - 1)) {
                            return Err(const_arithmetic_error(
                                "literal out of range for signed integer in a constant",
                            ));
                        }
                    } else if is_signed_min(value, bits) {
                        return Err(const_arithmetic_error(
                            "attempt to negate with overflow in a constant",
                        ));
                    }
                }
                value.wrapping_neg()
            }
            UnOp::Not => !value,
            UnOp::Deref => bail!("a dereference in a constant"),
        };
        let bits = ty.map_or(128, |(bits, _)| bits);
        Ok((value & mask(bits), ty))
    }

    fn const_cast(&mut self, arena: &Body, inner: ExprId, ty: TyId) -> Result<(u128, Option<Int>)> {
        let target = self.lower_ty(arena, ty)?;
        let Kind::Prim(primitive) = *self.cx.types.kind(target) else {
            bail!("a cast to a type that is not an integer");
        };
        let Some(int) = int_info(primitive) else {
            bail!("a cast to a type that is not an integer");
        };
        let fallback = if super::super::expr::cast_int_lit_fits(arena, self.src, inner, int)? {
            int
        } else {
            I32
        };
        let (value, from) = self.const_defaulted(arena, inner, fallback)?;
        Ok((convert(value, from, int), Some(int)))
    }

    fn const_binary(
        &mut self,
        arena: &Body,
        op: BinOp,
        pair: (ExprId, ExprId),
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.const_logic(arena, op, pair);
        }
        if super::super::expr::is_compare(op) {
            return self.const_compare(arena, op, pair);
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            return self.const_shift(arena, op, pair, want);
        }
        let (x, y, ty) = self.const_operands(arena, pair, want)?;
        let int = ty.unwrap_or((128, false));
        Ok((fold(op, x, y, int)? & mask(int.0), ty))
    }

    fn const_logic(
        &mut self,
        arena: &Body,
        op: BinOp,
        (lhs, rhs): (ExprId, ExprId),
    ) -> Result<(u128, Option<Int>)> {
        let (x, _) = self.const_typed(arena, lhs, BOOL)?;
        if (x == 1) == (op == BinOp::Or) {
            return Ok((x, Some(BOOL)));
        }
        Ok((self.const_typed(arena, rhs, BOOL)?.0, Some(BOOL)))
    }

    fn const_shift(
        &mut self,
        arena: &Body,
        op: BinOp,
        (lhs, rhs): (ExprId, ExprId),
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let (x, lt) = self.const_eval(arena, lhs, want)?;
        let (y, _) = self.const_defaulted(arena, rhs, I32)?;
        let int = lt.unwrap_or((128, false));
        Ok((fold(op, x, y, int)? & mask(int.0), lt))
    }

    fn const_operands(
        &mut self,
        arena: &Body,
        (lhs, rhs): (ExprId, ExprId),
        want: Option<Int>,
    ) -> Result<(u128, u128, Option<Int>)> {
        let (mut x, lt) = self.const_eval(arena, lhs, None)?;
        let (y, rt) = self.const_eval(arena, rhs, lt.or(want))?;
        let ty = lt.or(rt).or(want);
        if lt.is_none() && ty.is_some() {
            (x, _) = self.const_eval(arena, lhs, ty)?;
        }
        Ok((x, y, ty))
    }

    fn const_compare(
        &mut self,
        arena: &Body,
        op: BinOp,
        (lhs, rhs): (ExprId, ExprId),
    ) -> Result<(u128, Option<Int>)> {
        let (mut x, mut y, ty) = self.const_operands(arena, (lhs, rhs), None)?;
        let int = match ty {
            Some(int) => int,
            None => {
                (x, _) = self.const_typed(arena, lhs, I32)?;
                (y, _) = self.const_typed(arena, rhs, I32)?;
                I32
            }
        };
        let order = if int.1 {
            signed(x, int.0).cmp(&signed(y, int.0))
        } else {
            x.cmp(&y)
        };
        let holds = match op {
            BinOp::Eq => order.is_eq(),
            BinOp::Ne => order.is_ne(),
            BinOp::Lt => order.is_lt(),
            BinOp::Le => order.is_le(),
            BinOp::Gt => order.is_gt(),
            _ => order.is_ge(),
        };
        Ok((u128::from(holds), Some(BOOL)))
    }

    fn const_name(
        &mut self,
        arena: &Body,
        (expr, path): (ExprId, PathId),
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        if let Some((value, int)) = self.const_arg_named(arena, path) {
            return Ok((value, Some(int)));
        }
        let Some(param) = self.named_param(arena, expr) else {
            let (value, int) = self.const_path(arena, path)?;
            return Ok((value, Some(int)));
        };
        let declared = match self.param_name(arena, expr) {
            Some(name) => self.const_param(name)?.and_then(|(ty, _)| self.int_of(ty)),
            None => None,
        };
        let ty = declared.or(want);
        match *self.cx.types.kind(self.resolve(param)) {
            Kind::Value(bits) => Ok((ty.map_or(bits, |(w, _)| bits & mask(w)), ty)),
            Kind::Param(_) | Kind::Proj(_) => {
                self.poly_fallback();
                bail!("a const parameter whose value is not known yet")
            }
            _ => bail!("a const parameter whose value is not known yet"),
        }
    }

    pub(super) fn const_path(&mut self, arena: &Body, path: PathId) -> Result<(u128, Int)> {
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

    fn const_arg_named(&self, arena: &Body, path: PathId) -> Option<(u128, Int)> {
        let entry = arena.path(path);
        let [segment] = arena.list(entry.segments) else {
            return None;
        };
        let name = crate::names::plain(self.src.text(arena.segments[*segment as usize].name));
        let unqualified = entry.qself.is_none() && !entry.global;
        unqualified
            .then(|| {
                self.const_args
                    .iter()
                    .rev()
                    .find(|(bound, _)| bound == name)
                    .map(|(_, value)| *value)
            })
            .flatten()
    }

    fn const_call(
        &mut self,
        arena: &Body,
        (callee, args): (ExprId, List),
        want: Option<Int>,
    ) -> Result<(u128, Option<Int>)> {
        let Expr::Path(path) = *arena.expr(callee) else {
            bail!("a constant call of something that is not a path");
        };
        let (item, substs) = self.const_target(arena, path)?;
        let params = match self.cx.intrinsic(item) {
            Some(_) => vec![want; arena.list(args).len()],
            None => self.const_params(item, &substs)?,
        };
        let mut values = Vec::new();
        for (&arg, &param) in arena.list(args).iter().zip(&params) {
            let int = param.or(want).unwrap_or(I32);
            let (value, from) = self.const_defaulted(arena, ExprId(arg), int)?;
            values.push((convert(value, from, int), int));
        }
        let layout_of = |infer: &mut Self| match substs.first() {
            Some(&ty) => infer.cx.layout(ty),
            None => bail!("a layout intrinsic with no type"),
        };
        match self.cx.intrinsic(item) {
            Some(Intrinsic::SizeOf) => Ok((u128::from(layout_of(self)?.size.bytes()), Some(USIZE))),
            Some(Intrinsic::AlignOf) => {
                Ok((u128::from(layout_of(self)?.align.bytes()), Some(USIZE)))
            }
            Some(other) => bail!("the intrinsic {other:?} in a constant is not evaluated yet"),
            None => self.const_body(item, &substs, (&values, want)),
        }
    }

    fn const_target(&mut self, arena: &Body, path: PathId) -> Result<(ItemRef, Vec<Ty>)> {
        let (_, res) = self.path_value(arena, path)?;
        let Res::Fn(item, substs) = res else {
            bail!("a constant path that names no item");
        };
        let substs = self.map_substs(substs, Self::zonk);
        if self.substs_have_params(substs) {
            self.poly_fallback();
            bail!(
                "constant target {} depends on type parameters",
                self.cx.name(item)
            );
        }
        match self.trait_call(item, substs)? {
            Res::Fn(item, substs) => Ok((item, self.cx.types.substs(substs).to_vec())),
            _ => bail!("a constant path that names no item"),
        }
    }

    fn const_body(
        &mut self,
        item: ItemRef,
        substs: &[Ty],
        (values, want): (&[(u128, Int)], Option<Int>),
    ) -> Result<(u128, Option<Int>)> {
        let sig = self.cx.signature(item)?;
        if sig.params.len() != values.len() {
            bail!(
                "a constant call of {} with {} arguments",
                self.cx.name(item),
                values.len()
            );
        }
        self.with_const_body(item, substs, |infer, body, root| {
            let ret = match sig.ret {
                Some(ty) => {
                    let lowered = infer.lower_ty(&sig.body, ty)?;
                    infer.int_of(lowered)
                }
                None => None,
            };
            let mark = infer.const_args.len();
            for (&(pat, _), &value) in sig.params.iter().zip(values) {
                let crate::body::Pat::Bind {
                    name, sub: None, ..
                } = *sig.body.pat(pat)
                else {
                    bail!("a const fn parameter that is not a plain name");
                };
                infer
                    .const_args
                    .push((crate::names::plain(infer.src.text(name)).to_owned(), value));
            }
            let result = infer.const_eval(body, root, ret.or(want));
            infer.const_args.truncate(mark);
            let (value, ty) = result?;
            let out = ret.or(ty);
            Ok((out.map_or(value, |(bits, _)| value & mask(bits)), out))
        })
    }

    fn const_len(&mut self, arena: &Body, expr: ExprId) -> Result<u128> {
        match *arena.expr(expr) {
            Expr::Lit(Lit::Str | Lit::ByteStr, at) => {
                crate::body::literal::string_bytes(self.src.text(at))
                    .map(|bytes| bytes.len() as u128)
            }
            Expr::Array(items) => Ok(arena.list(items).len() as u128),
            Expr::Repeat(_, count) => self.const_typed(arena, count, USIZE).map(|(len, _)| len),
            Expr::Paren(inner) | Expr::Ref { expr: inner, .. } => self.const_len(arena, inner),
            Expr::Block {
                stmts,
                tail: Some(tail),
                ..
            } if arena.list(stmts).is_empty() => self.const_len(arena, tail),
            Expr::Path(path) => self.const_len_path(arena, path),
            Expr::Macro { path, open } => self.const_len_macro(arena, (path, open)),
            ref other => bail!("the length of {other:?} in a constant"),
        }
    }

    fn const_len_macro(&mut self, arena: &Body, call: (PathId, u32)) -> Result<u128> {
        let (output, inner, root) = self.macro_expr(arena, call)?;
        self.in_source(output, |infer| infer.const_len(&inner, root))
    }

    fn const_len_path(&mut self, arena: &Body, path: PathId) -> Result<u128> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        let (item, substs) = match self.value_def(&segments, absolute, dollar) {
            Ok(Def::Item(item)) => (item, Vec::new()),
            _ => self.const_target(arena, path)?,
        };
        if self.cx.item(item).1.kind != ItemKind::Const {
            bail!(
                "`{}` is not a constant whose length is known",
                self.spell(&segments)
            );
        }
        self.with_const_body(item, &substs, |infer, body, root| {
            infer.const_len(body, root)
        })
    }
}
