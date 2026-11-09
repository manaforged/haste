use anyhow::{Result, bail};

use crate::body::{BinOp, Body, Expr, ExprId, Lit, PathId, TyId, UnOp};
use crate::names::{Def, ItemRef, Primitive, Sym};
use crate::outline::item::ItemKind;

use super::super::ty::{Kind, Ty, int_info, mask};
use super::super::intrinsic::Intrinsic;
use super::{Infer, Res};
use super::path::parse_int;

/// The width and signedness of an integer type.
type Int = (u32, bool);

const U32: Int = (32, false);

const USIZE: Int = (64, false);

const BOOL: Int = (1, false);

const I32: Int = (32, true);

/// A method that constant expressions call: a pure method of the integer types, or the length of a string, byte string or array.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConstMethod {
    Ilog,
    Ilog2,
    Ilog10,
    Pow,
    CountOnes,
    CountZeros,
    LeadingZeros,
    TrailingZeros,
    Len,
}

const CONST_METHODS: [(&str, ConstMethod); 9] = [
    ("ilog", ConstMethod::Ilog),
    ("ilog2", ConstMethod::Ilog2),
    ("ilog10", ConstMethod::Ilog10),
    ("pow", ConstMethod::Pow),
    ("count_ones", ConstMethod::CountOnes),
    ("count_zeros", ConstMethod::CountZeros),
    ("leading_zeros", ConstMethod::LeadingZeros),
    ("trailing_zeros", ConstMethod::TrailingZeros),
    ("len", ConstMethod::Len),
];

fn ilog(value: u128, base: u128) -> u128 {
    let (mut value, mut log) = (value, 0);
    while value >= base {
        value /= base;
        log += 1;
    }
    log
}

fn int_method(method: ConstMethod, x: u128, args: &[u128], int: Int) -> Result<(u128, Int)> {
    let arg = args.first().copied();
    match (method, arg) {
        (ConstMethod::Ilog | ConstMethod::Ilog2 | ConstMethod::Ilog10, _) => int_log(method, x, arg, int).map(|log| (log, U32)),
        (ConstMethod::Pow, Some(exp)) => Ok((x.wrapping_pow(exp as u32) & mask(int.0), int)),
        (ConstMethod::Pow, None) => bail!("`pow` in a constant has no exponent"),
        _ => Ok((int_bits(method, x & mask(int.0), int.0), U32)),
    }
}

fn int_log(method: ConstMethod, x: u128, base: Option<u128>, (bits, is_signed): Int) -> Result<u128> {
    if x == 0 || (is_signed && signed(x, bits) < 0) {
        bail!("the logarithm of a value that is not positive in a constant");
    }
    Ok(match (method, base) {
        (ConstMethod::Ilog2, _) => u128::from(127 - x.leading_zeros()),
        (ConstMethod::Ilog, Some(base)) if !(is_signed && signed(base, bits) < 2) && base >= 2 => ilog(x, base),
        (ConstMethod::Ilog, _) => bail!("a logarithm of base below 2 in a constant"),
        _ => ilog(x, 10),
    })
}

fn int_bits(method: ConstMethod, x: u128, bits: u32) -> u128 {
    u128::from(match method {
        ConstMethod::CountOnes => x.count_ones(),
        ConstMethod::CountZeros => bits - x.count_ones(),
        ConstMethod::LeadingZeros => x.leading_zeros() - (128 - bits),
        _ => x.trailing_zeros().min(bits),
    })
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
    /// The value of the integer or bool constant item `item`, as the bits of its type.
    pub(super) fn const_value(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(u128, Int)> {
        let sig = self.cx.signature(item)?;
        let Some(ret) = sig.ret else { bail!("constant {} has no type", self.cx.name(item)) };
        self.with_const_body(item, substs, |infer, body, root| {
            let ty = infer.lower_ty(&sig.body, ret)?;
            let Kind::Prim(primitive) = *infer.cx.types.kind(ty) else { bail!("constant of a type that is not an integer") };
            let Some(int) = int_info(primitive).or((primitive == Primitive::Bool).then_some(BOOL)) else { bail!("constant of a type that is not an integer") };
            let (value, _) = infer.const_eval(body, root, int)?;
            Ok((value & mask(int.0), int))
        })
    }

    /// Runs `run` on the body and value expression of the constant or const fn `item` at `substs`, with names read as that body reads them.
    fn with_const_body<T>(&mut self, item: ItemRef, substs: &[Ty], run: impl FnOnce(&mut Self, &Body, ExprId) -> Result<T>) -> Result<T> {
        let body = self.cx.body(item)?;
        let Some(root) = body.root else { bail!("{} has no value to evaluate", self.cx.name(item)) };
        let mut frame = self.frame(item, substs)?;
        (frame.src, frame.root) = (self.cx.body_src(item), Some(item.body_owner()));
        self.with_frame(frame, |infer| run(infer, &body, root))
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
        let (value, (bits, _)) = self.const_eval(arena, expr, USIZE)?;
        Ok(self.cx.types.intern(Kind::Value(value & mask(bits))))
    }

    /// The generic argument `ty`: a type, or, when a plain path names no type, the constant it names, as rustc reads an argument that could be either.
    pub(super) fn ty_or_const_arg(&mut self, arena: &Body, ty: TyId) -> Result<Ty> {
        let lowered = self.lower_ty(arena, ty);
        let (Err(error), &crate::body::Ty::Path(path)) = (&lowered, arena.ty(ty)) else { return lowered };
        match self.const_path(arena, path) {
            Ok((value, (bits, _))) => Ok(self.cx.types.intern(Kind::Value(value & mask(bits)))),
            Err(_) => bail!("{error:#}"),
        }
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
            Expr::Lit(lit, at) => self.const_lit(lit, at, want),
            Expr::Paren(_) | Expr::Block { .. } | Expr::If(..) => self.const_flow(arena, expr, want),
            Expr::Unary(op, inner) => self.const_unary(arena, op, inner, want),
            Expr::Binary(op, lhs, rhs) => self.const_binary(arena, op, (lhs, rhs), want),
            Expr::Cast(inner, ty) => self.const_cast(arena, inner, ty),
            Expr::Call(callee, args) => self.const_call(arena, (callee, args), want),
            Expr::Path(path) => self.const_name(arena, (expr, path), want),
            Expr::Method { receiver, name, args: None, params } => self.const_method(arena, (receiver, name, params), want),
            other => bail!("constant expressions like {other:?} are not evaluated yet"),
        }
    }

    fn const_lit(&self, lit: Lit, at: u32, want: Int) -> Result<(u128, Int)> {
        let text = self.src.text(at);
        match lit {
            Lit::Int => {
                let (value, suffix) = parse_int(text)?;
                Ok((value, suffix.and_then(int_info).unwrap_or(want)))
            }
            Lit::Bool(value) => Ok((u128::from(value), BOOL)),
            Lit::Byte => Ok((u128::from(crate::body::literal::byte_value(text)?), (8, false))),
            Lit::Char => Ok((u128::from(crate::body::literal::char_value(text)?), (32, false))),
            other => bail!("constant {other:?} literals are not evaluated yet"),
        }
    }

    fn const_flow(&mut self, arena: &Body, expr: ExprId, want: Int) -> Result<(u128, Int)> {
        match *arena.expr(expr) {
            Expr::Paren(inner) => self.const_eval(arena, inner, want),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.const_eval(arena, tail, want),
            Expr::If(cond, then, Some(otherwise)) => {
                let (test, _) = self.const_eval(arena, cond, BOOL)?;
                self.const_eval(arena, if test != 0 { then } else { otherwise }, want)
            }
            _ => bail!("constant blocks with statements and `if` with no `else` are not evaluated yet"),
        }
    }

    fn const_method(&mut self, arena: &Body, (receiver, name, params): (ExprId, u32, crate::body::List), want: Int) -> Result<(u128, Int)> {
        let text = crate::names::plain(self.src.text(name));
        let Some(&(_, method)) = CONST_METHODS.iter().find(|(known, _)| *known == text) else { bail!("the method `{text}` is not evaluated in constants yet") };
        if method == ConstMethod::Len {
            return Ok((self.const_len(arena, receiver)?, USIZE));
        }
        let (x, int) = self.const_eval(arena, receiver, want)?;
        let wanted = if method == ConstMethod::Pow { U32 } else { int };
        let mut args = Vec::new();
        for &arg in arena.list(params) {
            args.push(self.const_eval(arena, ExprId(arg), wanted)?.0);
        }
        int_method(method, x, &args, int)
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
        if matches!(op, BinOp::And | BinOp::Or) {
            let (x, _) = self.const_eval(arena, lhs, BOOL)?;
            let decided = (x == 1) == (op == BinOp::Or);
            return if decided { Ok((x, BOOL)) } else { Ok((self.const_eval(arena, rhs, BOOL)?.0, BOOL)) };
        }
        if super::expr::is_compare(op) {
            return self.const_compare(arena, op, (lhs, rhs));
        }
        let (x, int) = self.const_eval(arena, lhs, want)?;
        let shift = matches!(op, BinOp::Shl | BinOp::Shr);
        let (y, _) = self.const_eval(arena, rhs, if shift { (32, false) } else { int })?;
        Ok((fold(op, x, y, int)? & mask(int.0), int))
    }

    fn const_compare(&mut self, arena: &Body, op: BinOp, (lhs, rhs): (ExprId, ExprId)) -> Result<(u128, Int)> {
        let (x, int) = self.const_eval(arena, lhs, I32)?;
        let (y, _) = self.const_eval(arena, rhs, int)?;
        let order = if int.1 { signed(x, int.0).cmp(&signed(y, int.0)) } else { x.cmp(&y) };
        let holds = match op {
            BinOp::Eq => order.is_eq(),
            BinOp::Ne => order.is_ne(),
            BinOp::Lt => order.is_lt(),
            BinOp::Le => order.is_le(),
            BinOp::Gt => order.is_gt(),
            _ => order.is_ge(),
        };
        Ok((u128::from(holds), BOOL))
    }

    fn const_name(&mut self, arena: &Body, (expr, path): (ExprId, PathId), want: Int) -> Result<(u128, Int)> {
        if let Some(value) = self.const_arg_named(arena, path) {
            return Ok(value);
        }
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

    /// The value of the parameter that the one-segment `path` names in the const fn being evaluated.
    fn const_arg_named(&self, arena: &Body, path: PathId) -> Option<(u128, Int)> {
        let entry = arena.path(path);
        let [segment] = arena.list(entry.segments) else { return None };
        let name = crate::names::plain(self.src.text(arena.segments[*segment as usize].name));
        let unqualified = entry.qself.is_none() && !entry.global;
        unqualified.then(|| self.const_args.iter().rev().find(|(bound, _)| bound == name).map(|(_, value)| *value)).flatten()
    }

    fn const_call(&mut self, arena: &Body, (callee, args): (ExprId, crate::body::List), want: Int) -> Result<(u128, Int)> {
        let Expr::Path(path) = *arena.expr(callee) else { bail!("a constant call of something that is not a path") };
        let mut values = Vec::new();
        for &arg in arena.list(args) {
            values.push(self.const_eval(arena, ExprId(arg), want)?);
        }
        let (item, substs) = self.const_target(arena, path)?;
        let layout_of = |infer: &mut Self| match substs.first() {
            Some(&ty) => infer.cx.layout(ty),
            None => bail!("a layout intrinsic with no type"),
        };
        match self.cx.intrinsic(item) {
            Some(Intrinsic::SizeOf) => Ok((u128::from(layout_of(self)?.size.bytes()), USIZE)),
            Some(Intrinsic::AlignOf) => Ok((u128::from(layout_of(self)?.align.abi.bytes()), USIZE)),
            Some(other) => bail!("the intrinsic {other:?} in a constant is not evaluated yet"),
            None => self.const_body(item, &substs, (&values, want)),
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

    /// The value of the const fn or constant `item` at `substs`, with its parameters bound to `values`.
    fn const_body(&mut self, item: ItemRef, substs: &[Ty], (values, want): (&[(u128, Int)], Int)) -> Result<(u128, Int)> {
        let sig = self.cx.signature(item)?;
        if sig.params.len() != values.len() {
            bail!("a constant call of {} with {} arguments", self.cx.name(item), values.len());
        }
        self.with_const_body(item, substs, |infer, body, root| {
            let mark = infer.const_args.len();
            for (&(pat, _), &value) in sig.params.iter().zip(values) {
                let crate::body::Pat::Bind { name, sub: None, .. } = *sig.body.pat(pat) else { bail!("a const fn parameter that is not a plain name") };
                infer.const_args.push((crate::names::plain(infer.src.text(name)).to_owned(), value));
            }
            let result = infer.const_eval(body, root, want);
            infer.const_args.truncate(mark);
            result
        })
    }

    fn const_len(&mut self, arena: &Body, expr: ExprId) -> Result<u128> {
        Ok(match *arena.expr(expr) {
            Expr::Lit(Lit::Str | Lit::ByteStr, at) => crate::body::literal::string_bytes(self.src.text(at))?.len() as u128,
            Expr::Array(items) => arena.list(items).len() as u128,
            Expr::Repeat(_, count) => self.const_eval(arena, count, USIZE)?.0,
            Expr::Paren(inner) | Expr::Ref { expr: inner, .. } => self.const_len(arena, inner)?,
            Expr::Path(path) => {
                let (segments, absolute, dollar) = self.segments(arena, path);
                match self.value_def(&segments, absolute, dollar)? {
                    Def::Item(item) if self.cx.item(item).1.kind == ItemKind::Const => self.with_const_body(item, &[], |infer, body, root| infer.const_len(body, root))?,
                    _ => bail!("`{}` is not a constant whose length is known", self.spell(&segments)),
                }
            }
            ref other => bail!("the length of {other:?} in a constant"),
        })
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
