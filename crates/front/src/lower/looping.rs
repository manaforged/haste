use anyhow::{Result, bail};
use haste_codegen::{Type, Value};

use crate::body::{ExprId, PatId};
use crate::sema::infer::Res;

use super::place::Place;
use super::{Loop, Lower, repr};

impl Lower<'_, '_> {
    /// Lowers `for pat in iter { body }`: `into_iter` once, then `next` until it returns `None`.
    pub fn for_expr(&mut self, expr: ExprId, pat: PatId, iter: ExprId, body: ExprId) -> Result<Option<Value>> {
        let Some(each) = self.typed.fors.get(&expr).cloned() else { bail!("`for` loop has no iterator calls") };
        let (Res::Fn(into, into_substs), Res::Fn(next, next_substs)) = (&each.into_iter, &each.next) else { bail!("`for` loop calls are not functions") };
        let value = self.expr(iter)?;
        let iterator = self.invoke(*into, into_substs, value.into_iter().collect())?;
        let place = self.spill(iterator, each.iter)?;
        self.own(super::moves::Root::Temp(iter), each.iter, place, true)?;
        let depth = self.owned.len();
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.jump(head, &[]);
        self.enter(head);
        let address = self.address_or_dangling(place, each.iter)?;
        let option = self.invoke(*next, next_substs, vec![address])?;
        let Some(option) = self.spill(option, each.option)? else { bail!("`next` returned no value") };
        match self.is_variant(option, each.option, each.some)? {
            Some(some) => self.brif(some, (inner, &[]), (exit, &[])),
            None => self.jump(inner, &[]),
        };
        self.enter(inner);
        let item = self.variant_field(option, each.option, (each.some, 0))?;
        self.bind_matched(pat, Some(item), None)?;
        self.loops.push(Loop { head, exit, value: None, depth });
        let result = self.expr(body);
        self.loops.pop();
        result?;
        self.end_scope(depth)?;
        self.branch_value(None, head);
        self.enter(exit);
        Ok(None)
    }

    /// Lowers `inner?`: `Try::branch`, then the `Continue` value, or a return of `FromResidual::from_residual` on the `Break` value.
    pub fn try_expr(&mut self, expr: ExprId, inner: ExprId) -> Result<Option<Value>> {
        let Some(tried) = self.typed.tries.get(&expr).cloned() else { bail!("`?` has no `Try` calls") };
        let (Res::Fn(branch, branch_substs), Res::Fn(from, from_substs)) = (&tried.branch, &tried.from_residual) else { bail!("`?` calls are not functions") };
        let value = self.expr(inner)?;
        let flow = self.invoke(*branch, branch_substs, value.into_iter().collect())?;
        let Some(flow) = self.spill(flow, tried.flow)? else { bail!("`branch` returned no value") };
        let Some(stopped) = self.is_variant(flow, tried.flow, tried.stop)? else { bail!("`?` on a `ControlFlow` with one variant") };
        let (stop, proceed) = (self.b.create_block(), self.b.create_block());
        self.brif(stopped, (stop, &[]), (proceed, &[]));
        self.enter(stop);
        let residual = self.variant_field(flow, tried.flow, (tried.stop, 0))?;
        let residual = self.read(residual);
        let result = self.invoke(*from, from_substs, residual.into_iter().collect())?;
        self.finish(result)?;
        self.enter(proceed);
        let output = self.variant_field(flow, tried.flow, (tried.proceed, 0))?;
        Ok(self.read(output))
    }

    fn spill(&mut self, value: Option<Value>, ty: crate::sema::ty::Ty) -> Result<Option<Place>> {
        let whole = repr(self.cx, ty)?;
        let Some(slot) = self.slot(whole) else { return Ok(None) };
        let place = Place::slot(slot, whole);
        if let Some(value) = value {
            self.write(place, value);
        }
        Ok(Some(place))
    }

    /// Lowers the range expression `expr` as the struct it builds.
    pub fn range(&mut self, expr: ExprId, from: Option<ExprId>, to: Option<ExprId>) -> Result<Option<Value>> {
        let Some(Res::Ctor(_, variant, _)) = self.typed.res.get(&expr).cloned() else { bail!("range has no constructor") };
        let mut values = Vec::new();
        for end in [from, to].into_iter().flatten() {
            let Some(&index) = self.typed.fields.get(&end) else { bail!("range end has no field") };
            values.push((index, self.expr(end)?));
        }
        if let Some(&exhausted) = self.typed.fields.get(&expr) {
            values.push((exhausted, Some(self.ins().iconst(Type::I8, 0))));
        }
        self.construct(self.ty_of(expr), variant, values, None)
    }
}
