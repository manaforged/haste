use std::sync::Arc;

use anyhow::{Result, anyhow, bail};

use crate::body::adt::AdtDef;
use crate::body::sig::{ImplHeader, Signature};
use crate::body::{BinOp, Body, Expr, ExprId, Lit, PathId, TyId, UnOp};
use crate::names::{Def, ItemRef, Primitive, Sym};
use crate::outline::item::ItemKind;

use super::super::evaluate::{self, CONST_METHODS, ConstMethod, Int, OVERFLOW_OPS, OverflowOp, convert, fold, int_method, overflowing_op, signed};
use super::super::impls::ImplRef;
use super::super::intrinsic::Intrinsic;
use super::super::reach::Instance;
use super::super::ty::{Kind, Ty, int_info, mask};
use super::path::parse_int;
use super::scope::Frame;
use super::{Infer, Res, Shape, Typed};

const U32: Int = (32, false);

const USIZE: Int = (64, false);

const BOOL: Int = (1, false);

const I32: Int = (32, true);

/// The parsed syntax that holds an anonymous constant: a function or constant body, a function or constant header, a type definition, an impl header, or a type alias.
#[derive(Clone, Debug)]
pub enum Arena {
    Body(Arc<Body>),
    Signature(Arc<Signature>),
    Adt(Arc<AdtDef>),
    Header(Arc<ImplHeader>),
    Alias(Arc<super::super::Alias>),
}

impl Arena {
    /// The arena whose expressions the constant is made of.
    pub fn body(&self) -> &Body {
        match self {
            Arena::Body(body) => body,
            Arena::Signature(sig) => &sig.body,
            Arena::Adt(def) => &def.body,
            Arena::Header(header) => &header.body,
            Arena::Alias(alias) => &alias.arena,
        }
    }

    fn tag(&self) -> u128 {
        match self {
            Arena::Body(_) => 0,
            Arena::Signature(_) => 1,
            Arena::Adt(_) => 2,
            Arena::Header(_) => 3,
            Arena::Alias(_) => 4,
        }
    }
}

/// An anonymous constant compiled as its own function that takes nothing: the arena that holds it and its value expression.
#[derive(Clone, Debug)]
pub struct Anon {
    pub arena: Arena,
    pub root: ExprId,
}

impl<'a> Infer<'_, 'a> {
    fn const_typed(&mut self, arena: &Body, expr: ExprId, target: Int) -> Result<(u128, Int)> {
        let (value, ty) = self.const_eval(arena, expr, Some(target))?;
        let int = ty.unwrap_or(target);
        Ok((value & mask(int.0), int))
    }

    fn const_defaulted(&mut self, arena: &Body, expr: ExprId, fallback: Int) -> Result<(u128, Int)> {
        let (value, ty) = self.const_eval(arena, expr, None)?;
        match ty {
            Some(int) => Ok((value & mask(int.0), int)),
            None => self.const_typed(arena, expr, fallback),
        }
    }

    /// The value of the integer or bool constant item `item`, as the bits of its type: folded from its syntax, or else from running its body.
    pub(super) fn const_value(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(u128, Int)> {
        if self.any_has_params(substs) {
            self.poly_fallback();
            bail!("constant {} depends on type parameters", self.cx.name(item));
        }
        let sig = self.cx.signature(item)?;
        let Some(ret) = sig.ret else { bail!("constant {} has no type", self.cx.name(item)) };
        let mut width = None;
        let folded = self.with_const_body(item, substs, |infer, body, root| {
            let ty = infer.lower_ty(&sig.body, ret)?;
            let Some(int) = infer.int_of(ty) else { bail!("constant of a type that is not an integer") };
            width = Some(int);
            let (value, _) = infer.const_typed(body, root, int)?;
            Ok((value & mask(int.0), int))
        });
        match (folded, width) {
            (Err(error), Some(int)) if !substs.iter().any(|&ty| self.has_vars(ty)) => {
                let interned = self.cx.types.intern_substs(substs);
                self.run_instance(&(item, interned), int, error)
            }
            (found, _) => found,
        }
    }

    /// The width and signedness of `ty` when it is an integer type or `bool`.
    fn int_of(&self, ty: Ty) -> Option<Int> {
        let Kind::Prim(primitive) = *self.cx.types.kind(ty) else {
            return None;
        };
        int_info(primitive).or((primitive == Primitive::Bool).then_some(BOOL))
    }

    /// The value, `int` wide, of the constant or anonymous constant `instance` from running it, once the folder failed with `error`.
    fn run_instance(&mut self, instance: &Instance, int: Int, error: anyhow::Error) -> Result<(u128, Int)> {
        let ran = evaluate::run(self.cx, instance, int.0);
        let value = ran.map_err(|reason| anyhow!("{error:#}; running {}: {reason:#}", self.cx.name(instance.0)))?;
        Ok((value & mask(int.0), int))
    }

    /// The value of the anonymous constant `expr` of `arena`, which the folder failed on with `error`, from running it as its own function once per pass and frame arguments: when those arguments are known and an item holds `arena`.
    fn run_anon(&mut self, arena: &Body, expr: ExprId, error: anyhow::Error) -> Result<(u128, Int)> {
        let args: Vec<Ty> = self.generics.iter().map(|&(_, ty)| ty).chain(self.self_ty).collect();
        if self.any_has_params(&args) {
            self.poly_fallback();
            return Err(error);
        }
        if args.iter().any(|&ty| self.has_vars(ty)) {
            return Err(error);
        }
        let Some((owner, arena)) = self.holder(arena) else {
            return Err(error);
        };
        let mut args: Vec<Ty> = args.into_iter().map(|ty| self.zonk(ty)).collect();
        args.extend([arena.tag(), u128::from(expr.0)].map(|value| self.cx.types.intern(Kind::Value(value))));
        let (instance, asked) = ((owner, self.cx.types.intern_substs(&args)), self.cx.asks.count());
        let typed = match self.cx.store.finals.get(&instance) {
            Some(typed) => typed,
            None => self.infer_anon(Anon { arena, root: expr }, &instance).map_err(|reason| anyhow!("{error:#}; inferring it: {reason:#}"))?,
        };
        let Some(int) = typed.ret.and_then(|ty| self.int_of(ty)) else { bail!("{error:#}; it is not an integer") };
        if self.cx.asks.count() != asked {
            self.cx.store.finals.remove(&instance);
            bail!("{error:#}; inferring it needs names that are not loaded yet");
        }
        let ran = self.run_instance(&instance, int, error);
        if ran.is_err() && self.cx.asks.count() != asked {
            self.cx.store.finals.remove(&instance);
        }
        ran
    }

    /// The item whose parsed syntax is `arena`, with that syntax: the body being inferred, or a header of the frame; none for a static, a foreign function, or syntax that no item holds, such as a macro's output.
    fn holder(&self, arena: &Body) -> Option<(ItemRef, Arena)> {
        let (store, held) = (&self.cx.store, |found: &Arena| std::ptr::eq(found.body(), arena));
        let items = self.instance.iter().map(|(item, _)| *item).chain(self.frame_item);
        for item in items {
            let found = match self.cx.item(item).1.kind {
                ItemKind::Struct | ItemKind::Enum | ItemKind::Union => [store.adts.get(&item).map(Arena::Adt), None],
                ItemKind::Type => [self.cx.aliases.get(&item).map(Arena::Alias), None],
                ItemKind::Fn | ItemKind::Const | ItemKind::Static if !self.cx.foreign(item) => [store.signatures.get(&item).map(Arena::Signature), store.bodies.get(&item).map(Arena::Body)],
                _ => [None, None],
            };
            if let Some(found) = found.into_iter().flatten().find(held) {
                return Some((item, found));
            }
        }
        let block = self.block?;
        let header = Arena::Header(store.headers.get(&block)?);
        let owner = match block {
            ImplRef::Body(item) => item,
            ImplRef::Module { krate, index } => {
                let entry = &self.cx.outline.crates[krate as usize].impls[index as usize];
                ItemRef { krate, file: entry.file as u32, item: entry.at }
            }
        };
        held(&header).then_some((owner, header))
    }

    /// The typed form of the anonymous constant `anon` as the body of `instance`, inferred in the frame as it stands and kept in the store, where the runner finds it.
    fn infer_anon(&mut self, anon: Anon, instance: &Instance) -> Result<Arc<Typed>> {
        let before = self.cx.quiet_mark();
        let (src, home, frames) = (self.src, self.home, self.frames);
        let frame = Frame { src, home, generics: self.generics.clone(), self_ty: self.self_ty, block: self.block, item: self.frame_item, root: self.root };
        let (arena, sig) = (anon.arena.clone(), Signature::default());
        let body = arena.body();
        let mut infer = Infer::start(self.cx, (body, &sig, src), home);
        infer.swap_frame(frame);
        (infer.instance, infer.frames) = (Some(*instance), frames);
        infer.out.exprs.resize(body.exprs.len(), Ty::NONE);
        let ty = infer.fresh(Kind::Var);
        infer.out.ret = Some(ty);
        let found = infer.expr(anon.root, Some(ty))?;
        infer.coerce_expr(anon.root, found, ty)?;
        let mut typed = infer.finish()?;
        typed.shape = Shape::Anon(anon);
        let typed = Arc::new(super::settle(self.cx, *instance, typed)?);
        if self.cx.quiet_mark() != before {
            bail!("inferring it deferred constant evaluation or asked for unloaded names");
        }
        self.cx.store.finals.insert(*instance, typed.clone());
        Ok(typed)
    }

    /// Runs `run` on the body and value expression of the constant or const fn `item` at `substs`, with names read as that body reads them.
    fn with_const_body<T>(&mut self, item: ItemRef, substs: &[Ty], run: impl FnOnce(&mut Self, &Body, ExprId) -> Result<T>) -> Result<T> {
        let body = self.cx.body(item)?;
        let Some(root) = body.root else { bail!("{} has no value to evaluate", self.cx.name(item)) };
        let constant = self.cx.item(item).1.kind != ItemKind::Fn;
        if constant && self.evaluating.contains(&item) {
            bail!("the constant {} depends on itself", self.cx.name(item));
        }
        self.evaluating.extend(constant.then_some(item));
        let found = self.with_item_frame(item, substs, |infer| run(infer, &body, root));
        if constant {
            self.evaluating.pop();
        }
        found
    }

    fn with_item_frame<T>(&mut self, item: ItemRef, substs: &[Ty], run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let mut frame = self.frame(item, substs)?;
        (frame.src, frame.root) = (self.cx.body_src(item), Some(item.body_owner()));
        self.with_frame(frame, run)
    }

    /// The integer type of each parameter of the const fn `item` at `substs`.
    fn const_params(&mut self, item: ItemRef, substs: &[Ty]) -> Result<Vec<Option<Int>>> {
        let sig = self.cx.signature(item)?;
        self.with_item_frame(item, substs, |infer| {
            let lowered: Result<Vec<Ty>> = sig.params.iter().map(|&(_, ty)| infer.lower_ty(&sig.body, ty)).collect();
            Ok(lowered?.into_iter().map(|ty| infer.int_of(ty)).collect())
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
                let (value, _) = self.const_typed(&def.body, expr, int)?;
                next = if int.1 { signed(value & mask(int.0), int.0) } else { (value & mask(int.0)) as i128 };
            }
            values.push(next);
            next = next.wrapping_add(1);
        }
        Ok(values)
    }

    /// The const value that `expr` of `arena` gives as a generic argument or an array length: a const parameter of the frame as it stands, or the value it folds to, or else the value from running it.
    pub(super) fn const_arg(&mut self, arena: &Body, expr: ExprId) -> Result<Ty> {
        if let Some(param) = self.named_param(arena, expr) {
            return Ok(param);
        }
        let (value, (bits, _)) = match self.const_defaulted(arena, expr, USIZE) {
            Err(error) => self.run_anon(arena, expr, error)?,
            found => found?,
        };
        Ok(self.cx.types.intern(Kind::Value(value & mask(bits))))
    }

    /// The generic argument `ty`: a type, or, when a plain path names no type, the constant it names, as rustc reads an argument that could be either.
    pub(super) fn ty_or_const_arg(&mut self, arena: &Body, ty: TyId) -> Result<Ty> {
        let lowered = self.lower_ty(arena, ty);
        let (Err(error), &crate::body::Ty::Path(path)) = (&lowered, arena.ty(ty)) else {
            return lowered;
        };
        match self.const_path(arena, path) {
            Ok((value, (bits, _))) => Ok(self.cx.types.intern(Kind::Value(value & mask(bits)))),
            Err(reason) => bail!("{error:#}; as a constant: {reason:#}"),
        }
    }

    fn param_name(&self, arena: &Body, expr: ExprId) -> Option<&'a str> {
        match *arena.expr(expr) {
            Expr::Paren(inner) => self.param_name(arena, inner),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.param_name(arena, tail),
            Expr::Path(path) => {
                let entry = arena.path(path);
                let [segment] = arena.list(entry.segments) else {
                    return None;
                };
                let named = entry.qself.is_none() && !entry.global && arena.segments[*segment as usize].args.is_none();
                named.then(|| self.src.text(arena.segments[*segment as usize].name))
            }
            _ => None,
        }
    }

    fn named_param(&self, arena: &Body, expr: ExprId) -> Option<Ty> {
        let name = self.param_name(arena, expr)?;
        self.generics.iter().find(|(generic, _)| *generic == name).map(|&(_, ty)| ty)
    }

    /// The value of the integer constant expression `expr` of `arena`, read as `want` unless it says its own type.
    pub(super) fn const_eval(&mut self, arena: &Body, expr: ExprId, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        match *arena.expr(expr) {
            Expr::Lit(lit, at) => self.const_lit(lit, at, want),
            Expr::Paren(_) | Expr::Block { .. } | Expr::If(..) => self.const_flow(arena, expr, want),
            Expr::Unary(op, inner) => self.const_unary(arena, op, inner, want),
            Expr::Binary(op, lhs, rhs) => self.const_binary(arena, op, (lhs, rhs), want),
            Expr::Cast(inner, ty) => self.const_cast(arena, inner, ty),
            Expr::Call(callee, args) => self.const_call(arena, (callee, args), want),
            Expr::Path(path) => self.const_name(arena, (expr, path), want),
            Expr::TupleField(base, index) => self.const_tuple_field(arena, base, index, want),
            Expr::Method { receiver, name, args: None, params } => self.const_method(arena, (receiver, name, params), want),
            Expr::Macro { path, open } => self.const_macro(arena, (path, open), want),
            other => bail!("constant expressions like {other:?} are not evaluated yet"),
        }
    }

    fn const_tuple_field(&mut self, arena: &Body, base: ExprId, index: u32, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        match *arena.expr(base) {
            Expr::Paren(inner) => self.const_tuple_field(arena, inner, index, want),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.const_tuple_field(arena, tail, index, want),
            Expr::Tuple(items) => {
                let &item = arena.list(items).get(index as usize).ok_or_else(|| anyhow::anyhow!("no field {index} on tuple constant"))?;
                self.const_eval(arena, ExprId(item), want)
            }
            Expr::Method { receiver, name, args: None, params } => self.const_overflowing(arena, (receiver, name, params), index, want),
            Expr::Path(path) => self.const_path_field(arena, path, index, want),
            other => bail!("constant tuple field on {other:?} is not evaluated yet"),
        }
    }

    fn const_overflowing(&mut self, arena: &Body, (receiver, name, params): (ExprId, u32, crate::body::List), index: u32, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let text = crate::names::plain(self.src.text(name));
        let Some(&(_, op)) = OVERFLOW_OPS.iter().find(|(known, _)| *known == text) else { bail!("the method `{text}` is not evaluated in constants yet") };
        let &[rhs] = arena.list(params) else { bail!("`{text}` takes one operand") };
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

    fn const_path_field(&mut self, arena: &Body, path: PathId, index: u32, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let (item, substs) = self.const_target(arena, path)?;
        let body = self.cx.body(item)?;
        let frame = self.frame(item, &substs)?;
        self.with_frame(frame, |infer| {
            let root = body.root.ok_or_else(|| anyhow::anyhow!("constant has no body"))?;
            let items = match *body.expr(root) {
                Expr::Tuple(items) | Expr::Call(_, items) => items,
                _ => bail!("constant tuple field on non-tuple constant"),
            };
            let &elem = body.list(items).get(index as usize).ok_or_else(|| anyhow::anyhow!("no field {index} on constant"))?;
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
            Lit::Byte => Ok((u128::from(crate::body::literal::byte_value(text)?), Some((8, false)))),
            Lit::Char => Ok((u128::from(crate::body::literal::char_value(text)?), Some((32, false)))),
            other => bail!("constant {other:?} literals are not evaluated yet"),
        }
    }

    fn const_flow(&mut self, arena: &Body, expr: ExprId, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        match *arena.expr(expr) {
            Expr::Paren(inner) => self.const_eval(arena, inner, want),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.const_eval(arena, tail, want),
            Expr::If(cond, then, Some(otherwise)) => {
                let (test, _) = self.const_typed(arena, cond, BOOL)?;
                self.const_eval(arena, if test != 0 { then } else { otherwise }, want)
            }
            _ => bail!("constant blocks with statements and `if` with no `else` are not evaluated yet"),
        }
    }

    /// The value of the macro call `call`, folded from the expression that the macro writes.
    fn const_macro(&mut self, arena: &Body, call: (PathId, u32), want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let (output, inner, root) = self.macro_expr(arena, call)?;
        self.in_source(output, |infer| infer.const_eval(&inner, root, want))
    }

    fn macro_expr(&mut self, arena: &Body, (path, open): (PathId, u32)) -> Result<(&'a crate::outline::source::Source, Body, ExprId)> {
        let (segments, found) = self.resolve_macro(arena, path);
        let Some(output) = self.expand_macro_call(found, open)? else { bail!("constant macro `{}` is not supported", self.spell(&segments)) };
        let config = crate::body::Config { calls: None, env: Some(&self.cx.outline.crates[self.home.krate as usize].env) };
        let inner = crate::body::parse_expr(output, 0, output.len(), config)?;
        let Some(root) = inner.root else { bail!("macro `{}` wrote no constant expression", self.spell(&segments)) };
        Ok((output, inner, root))
    }

    fn const_method(&mut self, arena: &Body, (receiver, name, params): (ExprId, u32, crate::body::List), want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let text = crate::names::plain(self.src.text(name));
        let Some(&(_, method)) = CONST_METHODS.iter().find(|(known, _)| *known == text) else { bail!("the method `{text}` is not evaluated in constants yet") };
        if method == ConstMethod::Len {
            return Ok((self.const_len(arena, receiver)?, Some(USIZE)));
        }
        let fallback = if method == ConstMethod::Pow { want.unwrap_or(I32) } else { I32 };
        let (x, int) = self.const_defaulted(arena, receiver, fallback)?;
        let wanted = if method == ConstMethod::Pow { U32 } else { int };
        let mut args = Vec::new();
        for &arg in arena.list(params) {
            args.push(self.const_typed(arena, ExprId(arg), wanted)?.0);
        }
        int_method(method, x, &args, int).map(|(value, ty)| (value, Some(ty)))
    }

    fn const_unary(&mut self, arena: &Body, op: UnOp, inner: ExprId, want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let (value, ty) = self.const_eval(arena, inner, want)?;
        let value = match op {
            UnOp::Neg => value.wrapping_neg(),
            UnOp::Not => !value,
            UnOp::Deref => bail!("a dereference in a constant"),
        };
        let bits = ty.map_or(128, |(bits, _)| bits);
        Ok((value & mask(bits), ty))
    }

    fn const_cast(&mut self, arena: &Body, inner: ExprId, ty: TyId) -> Result<(u128, Option<Int>)> {
        let target = self.lower_ty(arena, ty)?;
        let Kind::Prim(primitive) = *self.cx.types.kind(target) else { bail!("a cast to a type that is not an integer") };
        let Some(int) = int_info(primitive) else { bail!("a cast to a type that is not an integer") };
        let (value, from) = self.const_defaulted(arena, inner, I32)?;
        Ok((convert(value, from, int), Some(int)))
    }

    fn const_binary(&mut self, arena: &Body, op: BinOp, pair: (ExprId, ExprId), want: Option<Int>) -> Result<(u128, Option<Int>)> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.const_logic(arena, op, pair);
        }
        if super::expr::is_compare(op) {
            return self.const_compare(arena, op, pair);
        }
        if matches!(op, BinOp::Shl | BinOp::Shr) {
            return self.const_shift(arena, op, pair, want);
        }
        let (x, y, ty) = self.const_operands(arena, pair, want)?;
        let int = ty.unwrap_or((128, false));
        Ok((fold(op, x, y, int)? & mask(int.0), ty))
    }

    fn const_logic(&mut self, arena: &Body, op: BinOp, (lhs, rhs): (ExprId, ExprId)) -> Result<(u128, Option<Int>)> {
        let (x, _) = self.const_typed(arena, lhs, BOOL)?;
        if (x == 1) == (op == BinOp::Or) {
            return Ok((x, Some(BOOL)));
        }
        Ok((self.const_typed(arena, rhs, BOOL)?.0, Some(BOOL)))
    }

    fn const_shift(&mut self, arena: &Body, op: BinOp, (lhs, rhs): (ExprId, ExprId), want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let (x, lt) = self.const_eval(arena, lhs, want)?;
        let (y, _) = self.const_defaulted(arena, rhs, I32)?;
        let int = lt.unwrap_or((128, false));
        Ok((fold(op, x, y, int)? & mask(int.0), lt))
    }

    fn const_operands(&mut self, arena: &Body, (lhs, rhs): (ExprId, ExprId), want: Option<Int>) -> Result<(u128, u128, Option<Int>)> {
        let (mut x, lt) = self.const_eval(arena, lhs, None)?;
        let (y, rt) = self.const_eval(arena, rhs, lt.or(want))?;
        let ty = lt.or(rt).or(want);
        if lt.is_none() && ty.is_some() {
            (x, _) = self.const_eval(arena, lhs, ty)?;
        }
        Ok((x, y, ty))
    }

    fn const_compare(&mut self, arena: &Body, op: BinOp, (lhs, rhs): (ExprId, ExprId)) -> Result<(u128, Option<Int>)> {
        let (mut x, mut y, ty) = self.const_operands(arena, (lhs, rhs), None)?;
        let int = match ty {
            Some(int) => int,
            None => {
                (x, _) = self.const_typed(arena, lhs, I32)?;
                (y, _) = self.const_typed(arena, rhs, I32)?;
                I32
            }
        };
        let order = if int.1 { signed(x, int.0).cmp(&signed(y, int.0)) } else { x.cmp(&y) };
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

    fn const_name(&mut self, arena: &Body, (expr, path): (ExprId, PathId), want: Option<Int>) -> Result<(u128, Option<Int>)> {
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
        let [segment] = arena.list(entry.segments) else {
            return None;
        };
        let name = crate::names::plain(self.src.text(arena.segments[*segment as usize].name));
        let unqualified = entry.qself.is_none() && !entry.global;
        unqualified.then(|| self.const_args.iter().rev().find(|(bound, _)| bound == name).map(|(_, value)| *value)).flatten()
    }

    fn const_call(&mut self, arena: &Body, (callee, args): (ExprId, crate::body::List), want: Option<Int>) -> Result<(u128, Option<Int>)> {
        let Expr::Path(path) = *arena.expr(callee) else { bail!("a constant call of something that is not a path") };
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
            Some(Intrinsic::AlignOf) => Ok((u128::from(layout_of(self)?.align.bytes()), Some(USIZE))),
            Some(other) => bail!("the intrinsic {other:?} in a constant is not evaluated yet"),
            None => self.const_body(item, &substs, (&values, want)),
        }
    }

    fn const_target(&mut self, arena: &Body, path: PathId) -> Result<(ItemRef, Vec<Ty>)> {
        let (_, res) = self.path_value(arena, path)?;
        let Res::Fn(item, substs) = res else { bail!("a constant path that names no item") };
        let substs = self.map_substs(substs, Self::zonk);
        if self.substs_have_params(substs) {
            self.poly_fallback();
            bail!("constant target {} depends on type parameters", self.cx.name(item));
        }
        match self.trait_call(item, substs)? {
            Res::Fn(item, substs) => Ok((item, self.cx.types.substs(substs).to_vec())),
            _ => bail!("a constant path that names no item"),
        }
    }

    /// The value of the const fn or constant `item` at `substs`, with its parameters bound to `values`.
    fn const_body(&mut self, item: ItemRef, substs: &[Ty], (values, want): (&[(u128, Int)], Option<Int>)) -> Result<(u128, Option<Int>)> {
        let sig = self.cx.signature(item)?;
        if sig.params.len() != values.len() {
            bail!("a constant call of {} with {} arguments", self.cx.name(item), values.len());
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
                let crate::body::Pat::Bind { name, sub: None, .. } = *sig.body.pat(pat) else { bail!("a const fn parameter that is not a plain name") };
                infer.const_args.push((crate::names::plain(infer.src.text(name)).to_owned(), value));
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
            Expr::Lit(Lit::Str | Lit::ByteStr, at) => crate::body::literal::string_bytes(self.src.text(at)).map(|bytes| bytes.len() as u128),
            Expr::Array(items) => Ok(arena.list(items).len() as u128),
            Expr::Repeat(_, count) => self.const_typed(arena, count, USIZE).map(|(len, _)| len),
            Expr::Paren(inner) | Expr::Ref { expr: inner, .. } => self.const_len(arena, inner),
            Expr::Block { stmts, tail: Some(tail), .. } if arena.list(stmts).is_empty() => self.const_len(arena, tail),
            Expr::Path(path) => self.const_len_path(arena, path),
            Expr::Macro { path, open } => self.const_len_macro(arena, (path, open)),
            ref other => bail!("the length of {other:?} in a constant"),
        }
    }

    fn const_len_macro(&mut self, arena: &Body, call: (PathId, u32)) -> Result<u128> {
        let (output, inner, root) = self.macro_expr(arena, call)?;
        self.in_source(output, |infer| infer.const_len(&inner, root))
    }

    /// The length of the constant array or string that `path` names: a constant item, or an associated constant such as `Kind::ALL`.
    fn const_len_path(&mut self, arena: &Body, path: PathId) -> Result<u128> {
        let (segments, absolute, dollar) = self.segments(arena, path);
        let (item, substs) = match self.value_def(&segments, absolute, dollar) {
            Ok(Def::Item(item)) => (item, Vec::new()),
            _ => self.const_target(arena, path)?,
        };
        if self.cx.item(item).1.kind != ItemKind::Const {
            bail!("`{}` is not a constant whose length is known", self.spell(&segments));
        }
        self.with_const_body(item, &substs, |infer, body, root| infer.const_len(body, root))
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
