use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::{Body, Expr, ExprId, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;
use crate::sema::Context;
use crate::sema::infer::{Coercion, Res, SiteKey};
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::Kind;

pub(super) use super::budget::inline_limit;
use super::place::Place;
use super::reloc::{SiteOrigin, TyOrigin};
use super::stencil::resolve_inline;
use super::{Lower, Parts, Repr};

mod call;
mod flow;

/// Whether a call may try to copy reached function `index` inline.
pub fn may_inline(cx: &mut Context<'_>, reached: &Reached, index: usize) -> bool {
    inline_limit(cx, reached, index).is_some()
}

/// Whether reached function `index` lowers to a body small and copyable enough to inline.
pub fn can_inline_body(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn super::Slots,
    index: usize,
) -> bool {
    if !slots.allow_inline() {
        return false;
    }
    let Some(limit) = inline_limit(cx, reached, index) else {
        return false;
    };
    resolve_inline(
        cx,
        (reached, slots),
        &mut super::Inline::default(),
        index,
        (limit, false),
    )
    .is_ok_and(|(callee, _)| callee.is_some())
}

pub(super) fn unwrap_expr(
    body: &Body,
    mut expr: ExprId,
    allow_block: impl Fn(crate::body::BlockKind, bool) -> bool,
) -> ExprId {
    loop {
        match *body.expr(expr) {
            Expr::Paren(inner) => expr = inner,
            Expr::Block {
                stmts,
                tail: Some(inner),
                label,
                kind,
            } if body.list(stmts).is_empty() && allow_block(kind, label.is_some()) => {
                expr = inner;
            }
            _ => return expr,
        }
    }
}

fn unwrap_plain_expr(body: &Body, expr: ExprId, allow_unsafe: bool) -> ExprId {
    unwrap_expr(body, expr, |kind, labeled| {
        !labeled
            && (matches!(
                kind,
                crate::body::BlockKind::Plain | crate::body::BlockKind::Expansion
            ) || (allow_unsafe && kind == crate::body::BlockKind::Unsafe))
    })
}

pub(super) fn const_bool(body: &Body, expr: ExprId) -> Option<bool> {
    match *body.expr(unwrap_plain_expr(body, expr, false)) {
        Expr::Lit(crate::body::Lit::Bool(taken), _) => Some(taken),
        _ => None,
    }
}

impl Lower<'_, '_> {
    /// Lowers `expr` and returns its value, or none for `()`, `!` and code after a jump; its instructions carry its source place.
    pub fn expr(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let outer = self.b.loc();
        if let Some(loc) = self.expr_site_place(expr) {
            self.b.set_loc(loc);
        }
        let value = self.expr_placed(expr);
        self.b.set_loc(outer);
        value
    }

    fn expr_placed(&mut self, expr: ExprId) -> Result<Option<Value>> {
        if !self.open {
            return Ok(None);
        }
        let value = self.expr_kind(expr)?;
        let typed = self.typed;
        match (typed.coercions.get(&expr), value) {
            _ if !self.open => Ok(None),
            (
                Some(coercion @ (Coercion::Len(_) | Coercion::Vtable(_) | Coercion::Upcast(_))),
                Some(value),
            ) => {
                let coercion = coercion.clone();
                self.apply_leaf_coercion(expr, value, self.ty_of(expr), &coercion)
            }
            (Some(Coercion::Field { target, inner }), Some(value)) => {
                let (target, inner) = (*target, inner.as_ref().clone());
                self.coerce_struct(expr, value, self.ty_of(expr), target, &inner)
            }
            (Some(Coercion::Deref(steps, _)), Some(value)) => {
                self.deref_coerced(expr, value, steps.clone())
            }
            (Some(&Coercion::Reify(Res::Fn(item, substs))), _) => {
                let index = self.reached_index(item, substs)?;
                Ok(Some(self.code_of(
                    index,
                    SiteOrigin::Table(SiteKey::CoercionReify(expr)),
                )))
            }
            _ => Ok(value),
        }
    }

    fn expr_kind(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Lit(lit, at) => self.lit(expr, lit, at).map(Some),
            Expr::Path(_) => self.path(expr),
            Expr::Paren(inner) => self.expr(inner),
            Expr::Unary(UnOp::Deref, operand) => self.deref_read(expr, operand),
            Expr::Unary(op, operand) => self.unary(expr, op, operand),
            Expr::Binary(op, lhs, rhs) => self.binary(expr, op, lhs, rhs),
            Expr::Assign(lhs, rhs) => self.assign(expr, (lhs, rhs), None),
            Expr::AssignOp(op, lhs, rhs) => self.assign(expr, (lhs, rhs), Some(op)),
            Expr::Cast(inner, _) => self.cast(expr, inner),
            Expr::Call(callee, args) => self.call(expr, callee, args),
            Expr::Method {
                receiver, params, ..
            } => self.method(expr, receiver, params),
            Expr::OffsetOf { .. } => {
                let Some(&offset) = self.typed.offsets.get(&expr) else {
                    bail!("offset_of with no offset")
                };
                Ok(Some(
                    self.int_const(haste_codegen::Type::I64, u128::from(offset)),
                ))
            }
            _ => self.memory(expr),
        }
    }

    fn deref_read(&mut self, expr: ExprId, operand: ExprId) -> Result<Option<Value>> {
        if self.cx.boxed(self.ty_of(operand)).is_some()
            && crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))?
        {
            return self.move_out_of_box(expr, operand);
        }
        let place = self.place_of(expr)?;
        Ok(place.and_then(|place| self.read(place)))
    }

    fn move_out_of_box(&mut self, expr: ExprId, operand: ExprId) -> Result<Option<Value>> {
        let Some((root, path)) = self.scrutinee_path(operand) else {
            bail!("moving out of a `Box` behind a reference")
        };
        let boxed = self.ty_of(operand);
        let Some(Res::Fn(free, substs)) = crate::sema::infer::drop_fn(self.cx, boxed)? else {
            bail!("`Box` has no `Drop` impl")
        };
        let owner = self.place_or_temp(operand)?;
        let contents = self
            .deref_step((expr, 0), owner, boxed)?
            .map(|(place, _)| place);
        let value = match contents {
            Some(place) if matches!(place.repr, Repr::Mem { .. }) => {
                let read = self.read(place);
                self.spill(read, self.ty_of(expr))?
                    .and_then(|copy| self.read(copy))
            }
            place => place.and_then(|place| self.read(place)),
        };
        let target = self.address_or_dangling(owner, boxed)?;
        self.invoke(
            SiteOrigin::DropFn(TyOrigin::Expr(operand)),
            free,
            substs,
            vec![target],
        )?;
        self.moved_out(root, &path)?;
        Ok(value)
    }

    fn memory(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Tuple(list) => self.tuple(expr, list),
            Expr::Closure { .. } => self.closure(expr),
            Expr::Ref {
                expr: inner,
                mutable: false,
                raw: false,
            } if self.promotable(inner)? => {
                match self.materialize_const_expr(inner, (false, true))? {
                    Some(addr) => Ok(Some(addr)),
                    None => self.promoted(inner),
                }
            }
            Expr::Ref {
                expr: inner, raw, ..
            } => self.reference(inner, raw),
            Expr::TupleField(..) => self.field_value(expr),
            _ => self.flow(expr),
        }
    }

    fn flow(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Block {
                stmts,
                tail,
                label: None,
                kind: crate::body::BlockKind::Try,
            } => self.try_block(expr, stmts, tail),
            Expr::Block {
                kind: crate::body::BlockKind::Async | crate::body::BlockKind::AsyncMove,
                ..
            } => self.closure(expr),
            Expr::Await(inner) => self.await_expr(expr, inner),
            Expr::Block {
                stmts,
                tail,
                label: None,
                kind: crate::body::BlockKind::Expansion,
            } => self.expansion(stmts, tail),
            Expr::Block {
                stmts,
                tail,
                label: None,
                kind: crate::body::BlockKind::Const,
            } => self.const_block(expr, stmts, tail),
            Expr::Block {
                stmts,
                tail,
                label: None,
                ..
            } => self.block(stmts, tail),
            Expr::Block {
                stmts,
                tail,
                label: Some(label),
                ..
            } => self.labeled_block(expr, (stmts, tail), label),
            Expr::If(cond, then, otherwise) => self.if_expr(expr, cond, then, otherwise),
            Expr::While { cond, body, label } => self.while_expr(cond, body, label),
            Expr::Loop { body, label } => self.loop_expr(expr, body, label),
            Expr::Break(label, value) => self.break_expr(label, value),
            Expr::Continue(label) => self.continue_expr(label),
            Expr::Return(value) => self.return_expr(value),
            _ => self.data(expr),
        }
    }

    fn data(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match *self.body.expr(expr) {
            Expr::Struct { fields, base, .. } => self.struct_literal(expr, fields, base),
            Expr::Field(..) => self.field_value(expr),
            Expr::Match(scrutinee, arms) => self.match_expr(expr, scrutinee, arms),
            Expr::Let(pat, init) => self.let_expr(pat, init),
            Expr::Array(items) => match self.const_rodata_expr(expr)? {
                Some(addr) => Ok(Some(addr)),
                None => self.array(expr, items),
            },
            Expr::Repeat(value, _) => match self.const_rodata_expr(expr)? {
                Some(addr) => Ok(Some(addr)),
                None => self.repeat(expr, value),
            },
            Expr::Index(..) => Ok(self.place_of(expr)?.and_then(|place| self.read(place))),
            Expr::For {
                label,
                pat,
                iter,
                body,
            } => self.for_expr(expr, pat, iter, (label, body)),
            Expr::Try(inner) => self.try_expr(expr, inner),
            Expr::Range { from, to, .. } => self.range(expr, from, to),
            ref other => bail!("cannot lower expression {other:?} yet"),
        }
    }

    fn field_value(&mut self, expr: ExprId) -> Result<Option<Value>> {
        let ty = self.ty_of(expr);
        let value = self
            .place_of(expr)?
            .and_then(|place| self.read_typed(place, ty));
        if let Some((root, path)) = self.move_path(expr)
            && crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))?
        {
            self.moved_out(root, &path)?;
        }
        Ok(value)
    }

    fn path(&mut self, expr: ExprId) -> Result<Option<Value>> {
        match self.typed.res.get(&expr) {
            Some(&Res::Local(local)) => {
                let value = self.place_of(expr)?.and_then(|place| self.read(place));
                if crate::sema::infer::needs_drop(self.cx, self.ty_of(expr))? {
                    self.moved_out(super::moves::Root::Local(local), &[])?;
                }
                Ok(value)
            }
            Some(&Res::Fn(item, substs)) => self.item_path(expr, item, substs),
            Some(&Res::Const(value)) => {
                let (Kind::Value(bits), Repr::Scalar(machine)) =
                    (self.cx.types.kind(value).clone(), self.repr_of(expr)?)
                else {
                    bail!("a const parameter whose value is not known")
                };
                Ok(Some(self.int_const(machine, bits)))
            }
            Some(Res::Ctor(..))
                if matches!(self.cx.types.kind(self.ty_of(expr)), Kind::FnDef(..)) =>
            {
                Ok(None)
            }
            Some(&Res::Ctor(_, variant, _)) => match self.const_rodata_expr(expr)? {
                Some(addr) => Ok(Some(addr)),
                None => self.construct(self.ty_of(expr), variant, Vec::new(), None),
            },
            None => bail!("path has no resolution"),
        }
    }

    fn item_path(&mut self, expr: ExprId, item: ItemRef, substs: Substs) -> Result<Option<Value>> {
        match self.cx.item(item).1.kind {
            ItemKind::Static => Ok(self.place_of(expr)?.and_then(|place| self.read(place))),
            ItemKind::Const => match self.const_rodata_item(expr, item, substs)? {
                Some(addr) => Ok(Some(addr)),
                None => self.constant(expr, item, substs),
            },
            _ => Ok(None),
        }
    }

    /// The register type that carries a value of `repr` across a block edge.
    pub fn carrier(repr: Repr) -> Option<Type> {
        match repr {
            Repr::Zst => None,
            Repr::Scalar(ty) => Some(ty),
            Repr::Mem { .. } => Some(Type::I64),
        }
    }

    pub(super) fn join_slot(&mut self, repr: Repr) -> Option<Place> {
        match repr {
            Repr::Mem { parts, .. } if parts != Parts::Bytes => {
                self.slot(repr).map(|slot| Place::slot(slot, repr))
            }
            _ => None,
        }
    }

    pub(super) fn unwrap_expr(&self, expr: ExprId, allow_unsafe: bool) -> ExprId {
        unwrap_plain_expr(self.body, expr, allow_unsafe)
    }
}
