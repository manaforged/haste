use anyhow::{Result, bail};
use haste_codegen::{Block, IntCC, Type, Value};
use rustc_abi::Variants;
use rustc_index::Idx;

use crate::body::{ExprId, List, PatId};
use crate::names::{ItemRef, Primitive};
use crate::sema::infer::{Res, SiteKey, Tried};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty};

use super::place::Place;
use super::reloc::SiteOrigin;
use super::{Loop, Lower, Repr, repr};

fn range_signed(prim: Primitive) -> Option<bool> {
    match prim {
        Primitive::I8 | Primitive::I16 | Primitive::I32 | Primitive::I64 | Primitive::Isize => Some(true),
        Primitive::U8 | Primitive::U16 | Primitive::U32 | Primitive::U64 | Primitive::Usize | Primitive::Char => Some(false),
        _ => None,
    }
}

impl Lower<'_, '_> {
    /// Lowers `for pat in iter { body }`: `into_iter` once, then `next` until it returns `None`.
    pub fn for_expr(&mut self, expr: ExprId, pat: PatId, iter: ExprId, (label, body): (Option<u32>, ExprId)) -> Result<Option<Value>> {
        let Some(each) = self.typed.fors.get(&expr).cloned() else { bail!("`for` loop has no iterator calls") };
        let (Res::Fn(into, into_substs), Res::Fn(next, next_substs)) = (each.into_iter, each.next) else { bail!("`for` loop calls are not functions") };
        if let Kind::Adt(adt, args) = *self.cx.types.kind(each.iter)
            && let Some(inclusive) = match self.cx.lang(adt) {
                Some("Range") => Some(false),
                Some("RangeInclusive") => Some(true),
                _ => None,
            }
            && let [elem_ty] = *self.cx.types.substs(args)
            && let Kind::Prim(prim) = *self.cx.types.kind(elem_ty)
            && let Some(signed) = range_signed(prim)
            && let Repr::Scalar(ir_ty) = repr(self.cx, elem_ty)?
        {
            return self.for_range(pat, iter, (each.iter, elem_ty), (signed, inclusive, ir_ty), (label, body));
        }
        let value = self.expr(iter)?;
        let iterator = self.invoke(SiteOrigin::Table(SiteKey::ForInto(expr)), into, into_substs, value.into_iter().collect())?;
        let place = self.spill(iterator, each.iter)?;
        self.own(super::moves::Root::Held(expr), each.iter, place, true)?;
        let depth = self.owned.len();
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.jump(head, &[]);
        self.enter(head);
        let address = self.address_or_dangling(place, each.iter)?;
        let option = self.invoke(SiteOrigin::Table(SiteKey::ForNext(expr)), next, next_substs, vec![address])?;
        let Some(option) = self.spill(option, each.option)? else { bail!("`next` returned no value") };
        match self.is_variant(Some(option), each.option, each.some)? {
            Some(some) => self.brif(some, (inner, &[]), (exit, &[])),
            None => self.jump(inner, &[]),
        };
        self.enter(inner);
        let item = self.variant_field(option, each.option, (each.some, 0))?;
        self.bind_matched(pat, Some(item), None)?;
        self.loop_body((head, exit, depth), label, body)
    }

    fn for_range(
        &mut self,
        pat: PatId,
        iter: ExprId,
        (range_ty, elem_ty): (Ty, Ty),
        (signed, inclusive, ir_ty): (bool, bool, Type),
        (label, body): (Option<u32>, ExprId),
    ) -> Result<Option<Value>> {
        let value = self.expr(iter)?;
        let Some(place) = self.spill(value, range_ty)? else { bail!("range has no place") };
        let start_place = self.part_place(place, range_ty, (0, 0))?;
        let end_place = self.part_place(place, range_ty, (0, 1))?;
        let Some(start) = self.read(start_place) else { bail!("range start has no value") };
        let Some(end) = self.read(end_place) else { bail!("range end has no value") };
        let cur_repr = repr(self.cx, elem_ty)?;
        let Some(cur_slot) = self.slot(cur_repr) else { bail!("range element has no slot") };
        let cur_place = Place::slot(cur_slot, cur_repr);
        self.write(cur_place, start);
        let done_place = if inclusive {
            let ex_place = self.part_place(place, range_ty, (0, 2))?;
            let Some(exhausted) = self.read(ex_place) else { bail!("range exhausted has no value") };
            let done_repr = Repr::Scalar(Type::I8);
            let Some(done_slot) = self.slot(done_repr) else { bail!("range done has no slot") };
            let done_place = Place::slot(done_slot, done_repr);
            self.write(done_place, exhausted);
            Some(done_place)
        } else {
            None
        };
        let depth = self.owned.len();
        let (head, inner, exit) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.jump(head, &[]);
        self.enter(head);
        let Some(cur) = self.read(cur_place) else { bail!("range cursor has no value") };
        let cc = match (signed, inclusive) {
            (true, false) => IntCC::SignedLessThan,
            (false, false) => IntCC::UnsignedLessThan,
            (true, true) => IntCC::SignedLessThanOrEqual,
            (false, true) => IntCC::UnsignedLessThanOrEqual,
        };
        let in_bounds = self.ins().icmp(cc, cur, end);
        let cond = if let Some(done_place) = done_place {
            let Some(done) = self.read(done_place) else { bail!("range done has no value") };
            let zero = self.ins().iconst(Type::I8, 0);
            let not_done = self.ins().icmp(IntCC::Equal, done, zero);
            self.ins().band(not_done, in_bounds)
        } else {
            in_bounds
        };
        self.brif(cond, (inner, &[]), (exit, &[]));
        self.enter(inner);
        if let Some(done_place) = done_place {
            let at_end = self.ins().icmp(IntCC::Equal, cur, end);
            self.write(done_place, at_end);
        }
        let one = self.ins().iconst(ir_ty, 1);
        let next = self.ins().iadd(cur, one);
        self.write(cur_place, next);
        let item = self.spill(Some(cur), elem_ty)?;
        self.bind_matched(pat, item, None)?;
        self.loop_body((head, exit, depth), label, body)
    }

    /// Lowers the body of a loop whose iterations start at block `head` and leave to `exit`, with `depth` owned values outside it, then the jump back to `head`; the loop's value is `()`.
    pub(super) fn loop_body(&mut self, (head, exit, depth): (Block, Block, usize), label: Option<u32>, body: ExprId) -> Result<Option<Value>> {
        self.loops.push(Loop { head, exit, value: None, depth, label, block: false });
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
        let (Res::Fn(branch, branch_substs), Res::Fn(from, from_substs)) = (tried.branch, tried.from_residual) else { bail!("`?` calls are not functions") };
        let value = self.expr(inner)?;
        let flow = self.invoke(SiteOrigin::Table(SiteKey::TryBranch(expr)), branch, branch_substs, value.into_iter().collect())?;
        let Some(flow) = self.spill(flow, tried.flow)? else { return self.zero_sized_try(expr, &tried, (from, from_substs)) };
        let Some(stopped) = self.is_variant(Some(flow), tried.flow, tried.stop)? else { return self.always_stop(expr, Some(flow), &tried, (from, from_substs)) };
        let (stop, proceed) = (self.b.create_block(), self.b.create_block());
        self.brif(stopped, (stop, &[]), (proceed, &[]));
        self.enter(stop);
        self.stop_with(expr, Some(flow), &tried, (from, from_substs))?;
        self.enter(proceed);
        let output = self.variant_field(flow, tried.flow, (tried.proceed, 0))?;
        Ok(self.read(output))
    }

    /// Lowers `?` on a zero-sized `ControlFlow`, whose one inhabited variant says at compile time whether it proceeds or stops.
    fn zero_sized_try(&mut self, expr: ExprId, tried: &Tried, from: (ItemRef, Substs)) -> Result<Option<Value>> {
        if matches!(self.cx.layout(tried.flow)?.variants, Variants::Single { index } if index.index() == tried.proceed as usize) {
            return Ok(None);
        }
        self.always_stop(expr, None, tried, from)
    }

    /// Lowers `?` on a `ControlFlow` at `flow` whose one inhabited variant is `Break`: a return of its residual, after which nothing runs.
    fn always_stop(&mut self, expr: ExprId, flow: Option<Place>, tried: &Tried, from: (ItemRef, Substs)) -> Result<Option<Value>> {
        self.stop_with(expr, flow, tried, from)?;
        let unreachable = self.b.create_block();
        self.enter(unreachable);
        Ok(None)
    }

    /// Leaves with `FromResidual::from_residual` on the `Break` value of the `ControlFlow` at `flow`.
    fn stop_with(&mut self, expr: ExprId, flow: Option<Place>, tried: &Tried, from: (ItemRef, Substs)) -> Result<()> {
        let residual = match flow {
            Some(flow) => {
                let field = self.variant_field(flow, tried.flow, (tried.stop, 0))?;
                self.read(field)
            }
            None => None,
        };
        self.leave_residual(expr, tried, from, residual.into_iter().collect())
    }

    /// Returns `FromResidual::from_residual` on the `Break` value `residual` from the function, or leaves the `try` block with it.
    fn leave_residual(&mut self, expr: ExprId, tried: &Tried, (from, from_substs): (ItemRef, Substs), residual: Vec<Value>) -> Result<()> {
        let result = self.invoke(SiteOrigin::Table(SiteKey::TryFrom(expr)), from, from_substs, residual)?;
        match tried.target {
            Some(block) => self.leave_try(block, result),
            None => self.finish(result),
        }
    }

    /// Lowers `try { .. }`: the block, then `Try::from_output` on its value, joined with the values that each `?` inside it leaves with.
    pub fn try_block(&mut self, expr: ExprId, stmts: List, tail: Option<ExprId>) -> Result<Option<Value>> {
        let Some(&Res::Fn(from_output, substs)) = self.typed.try_blocks.get(&expr) else { bail!("`try` block has no `from_output` call") };
        let exit = self.b.create_block();
        let value = self.value_slot(expr)?;
        self.try_exits.push((expr, Loop { head: exit, exit, value, depth: self.owned.len(), label: None, block: true }));
        let output = self.block(stmts, tail);
        let wrapped = match output {
            Ok(output) if self.open => self.invoke(SiteOrigin::Table(SiteKey::TryBlock(expr)), from_output, substs, output.into_iter().collect()),
            other => other,
        };
        let left = wrapped.and_then(|wrapped| self.leave_try(expr, wrapped));
        self.try_exits.pop();
        left?;
        self.enter(exit);
        Ok(value.and_then(|place| self.read(place)))
    }

    fn leave_try(&mut self, block: ExprId, value: Option<Value>) -> Result<()> {
        let Some(&(_, Loop { exit, value: slot, depth, .. })) = self.try_exits.iter().rev().find(|(at, _)| *at == block) else { bail!("`?` outside its `try` block") };
        if !self.open {
            return Ok(());
        }
        if let (Some(value), Some(place)) = (value, slot) {
            self.write(place, value);
        }
        self.drop_to(depth)?;
        self.branch_value(None, exit);
        Ok(())
    }

    /// A new frame slot for a value of `ty`, holding `value` when one is given.
    pub(super) fn spill(&mut self, value: Option<Value>, ty: crate::sema::ty::Ty) -> Result<Option<Place>> {
        let whole = repr(self.cx, ty)?;
        let Some(slot) = self.slot(whole) else {
            return Ok(None);
        };
        let place = Place::slot(slot, whole);
        if let Some(value) = value {
            self.write(place, value);
        }
        Ok(Some(place))
    }

    /// Lowers the range expression `expr` as the struct it builds.
    pub fn range(&mut self, expr: ExprId, from: Option<ExprId>, to: Option<ExprId>) -> Result<Option<Value>> {
        let Some(&Res::Ctor(_, variant, _)) = self.typed.res.get(&expr) else { bail!("range has no constructor") };
        let Some(fields) = self.typed.ranges.get(&expr).cloned() else { bail!("range has no fields") };
        let ends: Vec<ExprId> = [from, to].into_iter().flatten().collect();
        let mut values = Vec::new();
        for (at, &index) in fields.iter().enumerate() {
            let value = match ends.get(at) {
                Some(&end) => self.expr(end)?,
                None => Some(self.ins().iconst(Type::I8, 0)),
            };
            values.push((index, value));
        }
        self.construct(self.ty_of(expr), variant, values, None)
    }
}
