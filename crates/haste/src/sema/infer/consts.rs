mod eval;

use std::sync::Arc;

use anyhow::{Result, anyhow, bail};

use crate::body::adt::AdtDef;
use crate::body::sig::{ImplHeader, Signature};
use crate::body::{Body, Expr, ExprId, TyId};
use crate::names::{Def, ItemRef, Primitive, Sym};
use crate::outline::item::ItemKind;

use super::super::evaluate::{self, Int, signed};
use super::super::impls::ImplRef;
use super::super::reach::Instance;
use super::super::ty::{Kind, Ty, int_info, mask};
use super::scope::Frame;
use super::{Infer, Shape, Typed};

const USIZE: Int = (64, false);

const BOOL: Int = (1, false);

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

    fn const_defaulted(
        &mut self,
        arena: &Body,
        expr: ExprId,
        fallback: Int,
    ) -> Result<(u128, Int)> {
        let (value, ty) = self.const_eval(arena, expr, None)?;
        match ty {
            Some(int) => Ok((value & mask(int.0), int)),
            None => self.const_typed(arena, expr, fallback),
        }
    }

    pub(super) fn const_value(&mut self, item: ItemRef, substs: &[Ty]) -> Result<(u128, Int)> {
        if self.any_has_params(substs) {
            self.poly_fallback();
            bail!("constant {} depends on type parameters", self.cx.name(item));
        }
        let sig = self.cx.signature(item)?;
        let Some(ret) = sig.ret else {
            bail!("constant {} has no type", self.cx.name(item));
        };
        let asks0 = self.cx.asks.count();
        let mut width = None;
        let folded = self.with_const_body(item, substs, |infer, body, root| {
            let ty = infer.lower_ty(&sig.body, ret)?;
            let Some(int) = infer.int_of(ty) else {
                bail!("constant of a type that is not an integer");
            };
            width = Some(int);
            let (value, _) = infer.const_typed(body, root, int)?;
            Ok((value & mask(int.0), int))
        });
        match (folded, width) {
            (Ok((value, int)), _) => {
                if !substs.iter().any(|&ty| self.has_vars(ty)) && self.cx.asks.count() == asks0 {
                    let interned = self.cx.types.intern_substs(substs);
                    self.cx.store.evaluated.insert((item, interned), value);
                }
                Ok((value, int))
            }
            (Err(error), _) if evaluate::is_const_arithmetic_error(&error) => Err(error),
            (Err(error), Some(int)) if !substs.iter().any(|&ty| self.has_vars(ty)) => {
                let interned = self.cx.types.intern_substs(substs);
                self.run_instance(&(item, interned), int, error)
            }
            (found, _) => found,
        }
    }

    pub(in crate::sema) fn fold_const_item(
        cx: &mut super::super::Context<'_>,
        item: ItemRef,
    ) -> Option<Result<u128>> {
        let asks0 = cx.asks.count();
        let folded = super::detached(cx, item, |infer| {
            let sig = infer.cx.signature(item)?;
            let Some(ret) = sig.ret else {
                bail!("constant has no type");
            };
            infer.with_const_body(item, &[], |infer, body, root| {
                let ty = infer.lower_ty(&sig.body, ret)?;
                let Some(int) = infer.int_of(ty) else {
                    bail!("constant of a type that is not an integer");
                };
                let (value, _) = infer.const_typed(body, root, int)?;
                Ok(value & mask(int.0))
            })
        });
        match folded {
            Ok(value) if cx.asks.count() == asks0 => Some(Ok(value)),
            Err(err) if evaluate::is_const_arithmetic_error(&err) => Some(Err(err)),
            _ => None,
        }
    }

    pub(super) fn check_const_body_overflow(&mut self, item: ItemRef) -> Result<()> {
        if !matches!(
            self.cx.item(item).1.kind,
            ItemKind::Const | ItemKind::Static
        ) {
            return Ok(());
        }
        let Some(root) = self.body.root else {
            return Ok(());
        };
        let body = self.body;
        self.check_const_expr_overflow(body, root)
    }

    fn check_const_expr_overflow(&mut self, arena: &Body, expr: ExprId) -> Result<()> {
        let want = self
            .out
            .exprs
            .get(expr.0 as usize)
            .copied()
            .filter(|&ty| ty != Ty::NONE)
            .and_then(|ty| self.int_of(self.resolve(ty)));
        if let Some(int) = want {
            match self.const_typed(arena, expr, int) {
                Ok(_) => return Ok(()),
                Err(err) if evaluate::is_const_arithmetic_error(&err) => {
                    return Err(self.diag_expr(
                        crate::diag::DiagnosticKind::Type,
                        Some(expr),
                        err.to_string(),
                        "evaluation of constant value failed",
                    ));
                }
                Err(_) => {}
            }
        }
        match *arena.expr(expr) {
            Expr::Paren(inner) | Expr::Unary(_, inner) | Expr::Cast(inner, _) => {
                self.check_const_expr_overflow(arena, inner)
            }
            Expr::Ref { expr: inner, .. } => self.check_const_expr_overflow(arena, inner),
            Expr::Binary(_, lhs, rhs) => {
                self.check_const_expr_overflow(arena, lhs)?;
                self.check_const_expr_overflow(arena, rhs)
            }
            Expr::Tuple(list) | Expr::Array(list) => {
                for &id in arena.list(list) {
                    self.check_const_expr_overflow(arena, ExprId(id))?;
                }
                Ok(())
            }
            Expr::Repeat(val, _) => self.check_const_expr_overflow(arena, val),
            Expr::Struct { fields, base, .. } => {
                for &id in arena.list(fields) {
                    self.check_const_expr_overflow(arena, arena.fields[id as usize].value)?;
                }
                if let Some(base) = base {
                    self.check_const_expr_overflow(arena, base)?;
                }
                Ok(())
            }
            Expr::Call(callee, args) => {
                self.check_const_expr_overflow(arena, callee)?;
                for &id in arena.list(args) {
                    self.check_const_expr_overflow(arena, ExprId(id))?;
                }
                Ok(())
            }
            Expr::Method {
                receiver, params, ..
            } => {
                self.check_const_expr_overflow(arena, receiver)?;
                for &id in arena.list(params) {
                    self.check_const_expr_overflow(arena, ExprId(id))?;
                }
                Ok(())
            }
            Expr::Block { stmts, tail, .. } => {
                for &id in arena.list(stmts) {
                    match arena.stmts[id as usize] {
                        crate::body::Stmt::Let {
                            init, otherwise, ..
                        } => {
                            if let Some(e) = init {
                                self.check_const_expr_overflow(arena, e)?;
                            }
                            if let Some(e) = otherwise {
                                self.check_const_expr_overflow(arena, e)?;
                            }
                        }
                        crate::body::Stmt::Expr(e, _) => {
                            self.check_const_expr_overflow(arena, e)?
                        }
                        crate::body::Stmt::Item(..)
                        | crate::body::Stmt::Macro { .. }
                        | crate::body::Stmt::Empty => {}
                    }
                }
                if let Some(tail) = tail {
                    self.check_const_expr_overflow(arena, tail)?;
                }
                Ok(())
            }
            Expr::If(cond, then, otherwise) => {
                self.check_const_expr_overflow(arena, cond)?;
                self.check_const_expr_overflow(arena, then)?;
                if let Some(alt) = otherwise {
                    self.check_const_expr_overflow(arena, alt)?;
                }
                Ok(())
            }
            Expr::Match(scrut, arms) => {
                self.check_const_expr_overflow(arena, scrut)?;
                for &id in arena.list(arms) {
                    self.check_const_expr_overflow(arena, arena.arms[id as usize].body)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn int_of(&self, ty: Ty) -> Option<Int> {
        let Kind::Prim(primitive) = *self.cx.types.kind(ty) else {
            return None;
        };
        int_info(primitive).or((primitive == Primitive::Bool).then_some(BOOL))
    }

    fn run_instance(
        &mut self,
        instance: &Instance,
        int: Int,
        error: anyhow::Error,
    ) -> Result<(u128, Int)> {
        let ran = evaluate::run(self.cx, instance, int.0);
        let value = ran.map_err(|reason| {
            if evaluate::is_const_arithmetic_error(&reason) {
                return reason;
            }
            anyhow!(
                "{error:#}; running {}: {reason:#}",
                self.cx.name(instance.0)
            )
        })?;
        Ok((value & mask(int.0), int))
    }

    fn run_anon(
        &mut self,
        arena: &Body,
        expr: ExprId,
        error: anyhow::Error,
    ) -> Result<(u128, Int)> {
        let args: Vec<Ty> = self
            .generics
            .iter()
            .map(|&(_, ty)| ty)
            .chain(self.self_ty)
            .collect();
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
        args.extend(
            [arena.tag(), u128::from(expr.0)].map(|value| self.cx.types.intern(Kind::Value(value))),
        );
        let (instance, asked) = (
            (owner, self.cx.types.intern_substs(&args)),
            self.cx.asks.count(),
        );
        let typed = match self.cx.store.finals.get(&instance) {
            Some(typed) => typed,
            None => self
                .infer_anon(Anon { arena, root: expr }, &instance)
                .map_err(|reason| anyhow!("{error:#}; inferring it: {reason:#}"))?,
        };
        let Some(int) = typed.ret.and_then(|ty| self.int_of(ty)) else {
            bail!("{error:#}; it is not an integer");
        };
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

    fn holder(&self, arena: &Body) -> Option<(ItemRef, Arena)> {
        let (store, held) = (&self.cx.store, |found: &Arena| {
            std::ptr::eq(found.body(), arena)
        });
        let items = self
            .instance
            .iter()
            .map(|(item, _)| *item)
            .chain(self.frame_item);
        for item in items {
            let found = match self.cx.item(item).1.kind {
                ItemKind::Struct | ItemKind::Enum | ItemKind::Union => {
                    [store.adts.get(&item).map(Arena::Adt), None]
                }
                ItemKind::Type => [self.cx.aliases.get(&item).map(Arena::Alias), None],
                ItemKind::Fn | ItemKind::Const | ItemKind::Static if !self.cx.foreign(item) => [
                    store.signatures.get(&item).map(Arena::Signature),
                    store.bodies.get(&item).map(Arena::Body),
                ],
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
                ItemRef {
                    krate,
                    file: entry.file as u32,
                    item: entry.at,
                }
            }
        };
        held(&header).then_some((owner, header))
    }

    fn infer_anon(&mut self, anon: Anon, instance: &Instance) -> Result<Arc<Typed>> {
        let before = self.cx.quiet_mark();
        let (src, home, frames) = (self.src, self.home, self.frames);
        let frame = Frame {
            src,
            home,
            generics: self.generics.clone(),
            self_ty: self.self_ty,
            block: self.block,
            item: self.frame_item,
            root: self.root,
        };
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

    fn with_const_body<T>(
        &mut self,
        item: ItemRef,
        substs: &[Ty],
        run: impl FnOnce(&mut Self, &Body, ExprId) -> Result<T>,
    ) -> Result<T> {
        let body = self.cx.body(item)?;
        let Some(root) = body.root else {
            bail!("{} has no value to evaluate", self.cx.name(item));
        };
        let constant = self.cx.item(item).1.kind != ItemKind::Fn;
        if constant && self.evaluating.contains(&item) {
            return Err(evaluate::const_arithmetic_error(format!(
                "cycle detected when evaluating `{}`",
                self.cx.name(item)
            )));
        }
        self.evaluating.extend(constant.then_some(item));
        let found = self.with_item_frame(item, substs, |infer| run(infer, &body, root));
        if constant {
            self.evaluating.pop();
        }
        found
    }

    fn with_item_frame<T>(
        &mut self,
        item: ItemRef,
        substs: &[Ty],
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let mut frame = self.frame(item, substs)?;
        (frame.src, frame.root) = (self.cx.body_src(item), Some(item.body_owner()));
        self.with_frame(frame, run)
    }

    fn const_params(&mut self, item: ItemRef, substs: &[Ty]) -> Result<Vec<Option<Int>>> {
        let sig = self.cx.signature(item)?;
        self.with_item_frame(item, substs, |infer| {
            let lowered: Result<Vec<Ty>> = sig
                .params
                .iter()
                .map(|&(_, ty)| infer.lower_ty(&sig.body, ty))
                .collect();
            Ok(lowered?.into_iter().map(|ty| infer.int_of(ty)).collect())
        })
    }

    fn variant_value(&mut self, item: ItemRef, index: u32) -> Result<(u128, Int)> {
        let at = self.cx.variant_position(item, index)?;
        let def = self.cx.adt_def(item)?;
        if !def.variants[at].fields.is_empty() {
            bail!(
                "a constant variant of {} that has fields",
                self.cx.name(item)
            );
        }
        let Some(int) = int_info(self.cx.discriminant_primitive(item)) else {
            bail!("an enum tag that is not an integer");
        };
        let value = if self
            .active_discr
            .as_ref()
            .is_some_and(|(active, ..)| *active == item)
        {
            self.eval_discr_at(item, &def, int, at)?
        } else {
            self.cx.discriminants(item)?[at]
        };
        Ok((value as u128 & mask(int.0), int))
    }

    fn eval_discr_at(
        &mut self,
        item: ItemRef,
        def: &crate::body::adt::AdtDef,
        int: Int,
        at: usize,
    ) -> Result<i128> {
        let mut anchor = at;
        while anchor > 0 && def.variants[anchor].discriminant.is_none() {
            if self
                .active_discr
                .as_ref()
                .is_some_and(|(_, cached, _)| cached[anchor].is_some())
            {
                break;
            }
            anchor -= 1;
        }
        let base = if let Some(value) = self
            .active_discr
            .as_ref()
            .and_then(|(_, cached, _)| cached[anchor])
        {
            value
        } else if let Some(expr) = def.variants[anchor].discriminant {
            if let Some((_, _, visiting)) = self.active_discr.as_mut() {
                if visiting[anchor] {
                    bail!("cyclic enum discriminant in {}", self.cx.name(item));
                }
                visiting[anchor] = true;
            }
            let computed = self.const_typed(&def.body, expr, int).map(|(value, _)| {
                if int.1 {
                    signed(value & mask(int.0), int.0)
                } else {
                    (value & mask(int.0)) as i128
                }
            });
            if let Some((_, cached, visiting)) = self.active_discr.as_mut() {
                visiting[anchor] = false;
                if let Ok(value) = computed {
                    cached[anchor] = Some(value);
                }
            }
            computed?
        } else {
            if let Some((_, cached, _)) = self.active_discr.as_mut() {
                cached[anchor] = Some(0);
            }
            0
        };
        let max_val: i128 = if int.1 {
            if int.0 >= 128 {
                i128::MAX
            } else {
                (1_i128 << (int.0 - 1)) - 1
            }
        } else if int.0 >= 127 {
            i128::MAX
        } else {
            (1_i128 << int.0) - 1
        };
        let mut value = base;
        for step in (anchor + 1)..=at {
            if value >= max_val {
                bail!("enum discriminant overflow in {}", self.cx.name(item));
            }
            value += 1;
            if let Some((_, cached, _)) = self.active_discr.as_mut() {
                cached[step] = Some(value);
            }
        }
        Ok(value)
    }

    pub(super) fn discriminants(
        &mut self,
        item: ItemRef,
        primitive: Primitive,
    ) -> Result<Vec<i128>> {
        let def = self.cx.adt_def(item)?;
        let Some(int) = int_info(primitive) else {
            bail!("an enum tag that is not an integer");
        };
        let count = def.variants.len();
        let saved = self
            .active_discr
            .replace((item, vec![None; count], vec![false; count]));
        let mut values = Vec::with_capacity(count);
        for at in 0..count {
            match self.eval_discr_at(item, &def, int, at) {
                Ok(value) => values.push(value),
                Err(err) => {
                    self.active_discr = saved;
                    return Err(err);
                }
            }
        }
        self.active_discr = saved;
        let mut seen = rustc_hash::FxHashSet::default();
        for &value in &values {
            if !seen.insert(value) {
                bail!(
                    "discriminant value `{value}` assigned more than once in {}",
                    self.cx.name(item)
                );
            }
        }
        Ok(values)
    }

    pub(super) fn const_arg(&mut self, arena: &Body, expr: ExprId) -> Result<Ty> {
        if let Some(param) = self.named_param(arena, expr) {
            return Ok(param);
        }
        let (value, (bits, _)) = match self.const_defaulted(arena, expr, USIZE) {
            Err(error) if evaluate::is_const_arithmetic_error(&error) => return Err(error),
            Err(error) => self.run_anon(arena, expr, error)?,
            found => found?,
        };
        Ok(self.cx.types.intern(Kind::Value(value & mask(bits))))
    }

    pub(super) fn ty_or_const_arg(&mut self, arena: &Body, ty: TyId) -> Result<Ty> {
        let lowered = self.lower_ty(arena, ty);
        let (Err(error), &crate::body::Ty::Path(path)) = (&lowered, arena.ty(ty)) else {
            return lowered;
        };
        let entry = arena.path(path);
        let [seg] = arena.list(entry.segments) else {
            return lowered;
        };
        if entry.qself.is_some() || entry.global || arena.segments[*seg as usize].args.is_some() {
            return lowered;
        }
        match self.const_path(arena, path) {
            Ok((value, (bits, _))) => Ok(self.cx.types.intern(Kind::Value(value & mask(bits)))),
            Err(reason) => bail!("{error:#}; as a constant: {reason:#}"),
        }
    }

    fn param_name(&self, arena: &Body, expr: ExprId) -> Option<&'a str> {
        match *arena.expr(expr) {
            Expr::Paren(inner) => self.param_name(arena, inner),
            Expr::Block {
                stmts,
                tail: Some(tail),
                ..
            } if arena.list(stmts).is_empty() => self.param_name(arena, tail),
            Expr::Path(path) => {
                let entry = arena.path(path);
                let [segment] = arena.list(entry.segments) else {
                    return None;
                };
                let named = entry.qself.is_none()
                    && !entry.global
                    && arena.segments[*segment as usize].args.is_none();
                named.then(|| self.src.text(arena.segments[*segment as usize].name))
            }
            _ => None,
        }
    }

    fn named_param(&self, arena: &Body, expr: ExprId) -> Option<Ty> {
        let name = self.param_name(arena, expr)?;
        self.generics
            .iter()
            .find(|(generic, _)| *generic == name)
            .map(|&(_, ty)| ty)
    }

    pub(super) fn const_named(
        &mut self,
        segments: &[Sym],
        absolute: bool,
        dollar: Option<u32>,
    ) -> Result<(u128, Int)> {
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
            Some(Def::Item(item)) if self.cx.item(item).1.kind == ItemKind::Const => {
                self.const_value(item, &[])
            }
            Some(Def::Variant(item, index)) => self.variant_value(item, index),
            _ => bail!(
                "constant `{}` is not an integer constant item",
                self.spell(segments)
            ),
        }
    }
}
