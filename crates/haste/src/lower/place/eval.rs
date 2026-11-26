use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::{Expr, ExprId, List};
use crate::sema::infer::{Coercion, Res, cloned_tuple, needs_drop};
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::reloc::SiteOrigin;
use super::super::{Lower, Repr, repr};
use super::{Base, Place};

impl Lower<'_, '_> {
    /// Lowers the tuple expression `expr` into a new frame slot and returns its address.
    pub fn tuple(&mut self, expr: ExprId, list: List) -> Result<Option<Value>> {
        if let Some(addr) = self.const_rodata_expr(expr)? {
            return Ok(Some(addr));
        }
        self.tuple_of(self.ty_of(expr), list)
    }

    pub(in super::super) fn expr_lent_mutably(&self, expr: ExprId) -> bool {
        if self
            .typed
            .ref_binds
            .values()
            .any(|binding| *binding == Mut::Mut)
        {
            return true;
        }
        (0..self.body.exprs.len() as u32)
            .map(ExprId)
            .any(|at| match *self.body.expr(at) {
                Expr::Assign(lhs, _) => self.place_reaches_assign(lhs, expr),
                Expr::AssignOp(_, lhs, _)
                | Expr::Ref {
                    mutable: true,
                    expr: lhs,
                    ..
                } => self.place_reaches(lhs, expr),
                Expr::Method { receiver, .. } => {
                    self.typed
                        .adjusts
                        .get(&at)
                        .is_some_and(|adjust| adjust.autoref == Some(Mut::Mut))
                        && self.place_reaches(receiver, expr)
                }
                _ => false,
            })
    }

    fn place_reaches_assign(&self, lhs: ExprId, target: ExprId) -> bool {
        match *self.body.expr(lhs) {
            Expr::Paren(inner) => self.place_reaches_assign(inner, target),
            Expr::Tuple(items) => self
                .body
                .list(items)
                .iter()
                .any(|&at| self.place_reaches_assign(ExprId(at), target)),
            _ => self.place_reaches(lhs, target),
        }
    }

    fn place_reaches(&self, root: ExprId, target: ExprId) -> bool {
        let mut cur = root;
        loop {
            if cur == target {
                return true;
            }
            if let Some(inner) = self.transparent(cur) {
                cur = inner;
                continue;
            }
            match *self.body.expr(cur) {
                Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Index(base, _) => {
                    cur = base;
                }
                Expr::Block {
                    tail: Some(tail),
                    label: None,
                    ..
                } => cur = tail,
                Expr::Block { label: Some(_), .. } | Expr::Loop { .. } => return true,
                Expr::If(_, then, otherwise) => {
                    return self.place_reaches(then, target)
                        || otherwise.is_some_and(|alt| self.place_reaches(alt, target));
                }
                Expr::Match(_, arms) => {
                    return self
                        .body
                        .list(arms)
                        .iter()
                        .any(|&at| self.place_reaches(self.body.arms[at as usize].body, target));
                }
                _ => return false,
            }
        }
    }

    pub(in super::super) fn snapshot_repr(&mut self, repr: Repr, addr: Value) -> Value {
        let Some(slot) = self.slot(repr) else {
            return addr;
        };
        let dst = self.address(Place::slot(slot, repr));
        self.copy(dst, addr, repr);
        dst
    }

    pub(in super::super) fn snapshot_before_drops(
        &mut self,
        mark: usize,
        expr: ExprId,
        value: Option<Value>,
    ) -> Result<Option<Value>> {
        let Some(addr) = value else {
            return Ok(None);
        };
        if !self.open || self.owned.len() <= mark {
            return Ok(Some(addr));
        }
        let repr = self.coerced_repr(expr, self.typed.coercions.get(&expr))?;
        Ok(Some(if matches!(repr, Repr::Mem { .. }) {
            self.snapshot_repr(repr, addr)
        } else {
            addr
        }))
    }

    fn moved_local(&mut self, expr: ExprId) -> Result<bool> {
        if let Some(inner) = self.transparent(expr) {
            return self.moved_local(inner);
        }
        let local = matches!(self.body.expr(expr), Expr::Path(_))
            && matches!(self.typed.res.get(&expr), Some(Res::Local(_)));
        Ok(local && needs_drop(self.cx, self.ty_of(expr))?)
    }

    pub(in super::super) fn move_from_expr(&mut self, expr: ExprId) -> Result<()> {
        if !self.open || !needs_drop(self.cx, self.ty_of(expr))? {
            return Ok(());
        }
        if let Some((root, path)) = self.move_path(expr) {
            self.moved_out(root, &path)?;
        }
        Ok(())
    }

    pub(in super::super) fn eval_operand(&mut self, expr: ExprId) -> Result<Option<Value>> {
        self.operand(expr, false)
    }

    pub(in super::super) fn operand(&mut self, expr: ExprId, last: bool) -> Result<Option<Value>> {
        let in_place = last && self.moved_local(expr)?;
        let value = self.expr(expr)?;
        self.move_from_expr(expr)?;
        let Some(addr) = value else {
            return Ok(None);
        };
        let coerced = self.typed.coercions.get(&expr);
        let fresh_coercion = matches!(
            coerced,
            Some(
                Coercion::Len(_)
                    | Coercion::Vtable(_)
                    | Coercion::Upcast(_)
                    | Coercion::Reify(_)
                    | Coercion::Field { .. }
            )
        );
        if !self.open
            || !self.may_alias_place(expr)
            || fresh_coercion
            || (in_place && coerced.is_none())
        {
            return Ok(Some(addr));
        }
        let repr = self.coerced_repr(expr, coerced)?;
        Ok(Some(if matches!(repr, Repr::Mem { .. }) {
            self.snapshot_repr(repr, addr)
        } else {
            addr
        }))
    }

    pub(in super::super) fn tuple_of(&mut self, ty: Ty, list: List) -> Result<Option<Value>> {
        let whole = repr(self.cx, ty)?;
        let Kind::Tuple(items) = *self.cx.types.kind(ty) else {
            bail!("a tuple of a type that is not a tuple");
        };
        let items = self.cx.types.substs(items).to_vec();
        let slot = self.slot(whole);
        let unwind = self.wants_field_unwind()?;
        let mut guards = Vec::new();
        for (index, item) in self.body.list(list).to_vec().into_iter().enumerate() {
            let field_ty = items[index];
            let field = repr(self.cx, field_ty)?;
            let offset = self.field_offset(ty, index as u32)?;
            let part = slot.map(|slot| Place::slot(slot, whole).at(offset, field));
            if !self.eval_and_write_part(part, field_ty, ExprId(item), unwind, &mut guards)? {
                return Ok(None);
            }
        }
        self.disarm_owned_guards(guards);
        Ok(slot.and_then(|slot| self.read(Place::slot(slot, whole))))
    }

    pub(in super::super) fn push_arg_guard(
        &mut self,
        expr: ExprId,
        value: Option<Value>,
        guards: &mut Vec<usize>,
    ) -> Result<()> {
        let ty = match self.typed.coercions.get(&expr) {
            Some(Coercion::Field { target, .. }) => *target,
            _ => self.ty_of(expr),
        };
        if !self.open || !needs_drop(self.cx, ty)? {
            return Ok(());
        }
        let whole = repr(self.cx, ty)?;
        let place = match (whole, value) {
            (Repr::Zst, _) | (_, None) => None,
            (Repr::Mem { .. }, Some(addr)) => Some(Place {
                base: Base::Addr(addr),
                offset: 0,
                repr: whole,
            }),
            (Repr::Scalar(Type::I64), Some(val))
                if self
                    .typed
                    .coercions
                    .get(&expr)
                    .is_some_and(Coercion::widens) =>
            {
                Some(Place {
                    base: Base::Addr(val),
                    offset: 0,
                    repr: whole,
                })
            }
            (Repr::Scalar(_), Some(val)) => self.spill(Some(val), ty)?,
        };
        if let Some(guard) = self.push_owned_guard(ty, place, expr)? {
            guards.push(guard);
        }
        Ok(())
    }

    pub(in super::super) fn eval_guarded_args(
        &mut self,
        args: List,
        mut values: Vec<Value>,
        mut guards: Vec<usize>,
        check_const: bool,
    ) -> Result<Vec<Value>> {
        let list = self.body.list(args).to_vec();
        let unwind = self.wants_field_unwind()?;
        for (index, &arg) in list.iter().enumerate() {
            let expr = ExprId(arg);
            if check_const && self.typed.const_args.contains(&expr) {
                continue;
            }
            let last = index + 1 == list.len();
            let value = self.operand(expr, last)?;
            if !self.open {
                return Ok(values);
            }
            if unwind && !last {
                self.push_arg_guard(expr, value, &mut guards)?;
            }
            values.extend(value);
        }
        self.disarm_owned_guards(guards);
        Ok(values)
    }

    /// Lowers the compiler-made `Clone::clone` of a tuple: each field of the referenced tuple cloned through its call in `calls`, into a new tuple.
    pub fn clone_body(&mut self, calls: Vec<Res>) -> Result<Option<Value>> {
        let (Some(ty), Some(local)) = (self.typed.ret, self.typed.params.first().copied()) else {
            bail!("a compiler-made clone with no parameter or result");
        };
        let Some(source) = self.locals[local as usize].and_then(|place| self.read(place)) else {
            bail!("a compiler-made clone with no source");
        };
        let mut values = Vec::new();
        for (index, call) in calls.into_iter().enumerate() {
            let Res::Fn(item, substs) = call else {
                bail!("a field clone that is not a call")
            };
            let offset = self.field_offset(ty, index as u32)?;
            let field = self.ins().iadd_imm(source, i64::from(offset));
            values.push(self.invoke(
                SiteOrigin::ShapeClone(index as u16),
                item,
                substs,
                vec![field],
            )?);
        }
        match cloned_tuple(self.cx, ty) {
            Some(tuple) => self.tuple_from(tuple, values),
            None => {
                let whole = repr(self.cx, ty)?;
                Ok(self.read(Place {
                    base: Base::Addr(source),
                    offset: 0,
                    repr: whole,
                }))
            }
        }
    }

    fn tuple_from(&mut self, ty: Ty, values: Vec<Option<Value>>) -> Result<Option<Value>> {
        let whole = repr(self.cx, ty)?;
        let Kind::Tuple(items) = *self.cx.types.kind(ty) else {
            bail!("a tuple of a type that is not a tuple");
        };
        let items = self.cx.types.substs(items).to_vec();
        let slot = self.slot(whole);
        for (index, value) in values.into_iter().enumerate() {
            let field = repr(self.cx, items[index])?;
            let offset = self.field_offset(ty, index as u32)?;
            if let (Some(slot), Some(value)) = (slot, value) {
                self.write(Place::slot(slot, whole).at(offset, field), value);
            }
        }
        Ok(slot.and_then(|slot| self.read(Place::slot(slot, whole))))
    }
}
