use anyhow::{Result, bail};

use crate::body::{ExprId, List, PatId};
use crate::names::ItemRef;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

/// The calls a `for` loop makes, `IntoIterator::into_iter` on its value then `Iterator::next` on the iterator, and the types between them.
#[derive(Clone, Debug)]
pub struct For {
    pub into_iter: Res,
    pub next: Res,
    pub iter: Ty,
    pub option: Ty,
    pub some: u32,
}

/// The calls an `expr?` makes, `Try::branch` on its value and `FromResidual::from_residual` on a `Break`, and the `ControlFlow` between them.
#[derive(Clone, Debug)]
pub struct Tried {
    pub branch: Res,
    pub from_residual: Res,
    pub flow: Ty,
    pub proceed: u32,
    pub stop: u32,
    /// The `try` block that a `Break` leaves, or none when it returns from the function.
    pub target: Option<ExprId>,
}

impl Infer<'_, '_> {
    /// Infers `inner?` to the `Output` of its `Try` impl.
    pub fn try_expr(&mut self, expr: ExprId, inner: ExprId) -> Result<Ty> {
        let found = self.expr(inner, None)?;
        let found = self.resolve(found);
        let (branch, flow) = self.lang_call("branch", found)?;
        let Kind::Adt(item, args) = *self.cx.types.kind(self.resolve(flow)) else { bail!("`branch` does not return a `ControlFlow`") };
        let args = self.cx.types.substs(args).to_vec();
        let (Some(stop), Some(proceed)) = (self.variant_named(item, "Break")?, self.variant_named(item, "Continue")?) else { bail!("`ControlFlow` has no `Break` or `Continue`") };
        let Some(method) = self.cx.lang_item("from_residual") else { bail!("no `from_residual` lang item") };
        let target = self.try_blocks.last().copied();
        let ret = target.map(|(_, ty)| ty).or(self.out.ret).unwrap_or_else(|| self.cx.types.unit());
        if let Some(of) = self.cx.owner_trait(method)
            && (self.has_vars(ret) || self.has_vars(args[0]))
        {
            self.obligations.push((of, vec![ret, args[0]]));
        }
        let from_residual = Res::fn_of(&self.cx.types, method, &[ret, args[0]]);
        self.out.tries.insert(expr, Tried { branch, from_residual, flow, proceed, stop, target: target.map(|(block, _)| block) });
        Ok(args[1])
    }

    /// Infers `try { .. }` to a `Try` type whose `Output` its value is, and which each `?` inside it leaves with.
    pub fn try_block(&mut self, expr: ExprId, (stmts, tail): (List, Option<ExprId>), expected: Option<Ty>) -> Result<Ty> {
        let ty = expected.unwrap_or_else(|| self.fresh(Kind::Var));
        self.try_blocks.push((expr, ty));
        let found = self.block(stmts, tail, None);
        self.try_blocks.pop();
        let found = found?;
        let (Some(of), Some(method)) = (self.cx.lang_item("Try"), self.cx.lang_item("from_output")) else { bail!("core has no `Try::from_output`") };
        let output = self.project(of, ty, Vec::new(), "Output")?;
        self.coerce(found, output)?;
        self.out.try_blocks.insert(expr, Res::fn_of(&self.cx.types, method, &[ty]));
        Ok(ty)
    }

    /// Infers `for pat in iter { body }`.
    pub fn for_expr(&mut self, expr: ExprId, pat: PatId, iter: ExprId, (label, body): (Option<u32>, ExprId)) -> Result<Ty> {
        let found = self.expr(iter, None)?;
        let (into_iter, iter_ty) = self.lang_call("into_iter", found)?;
        let pending = self.opaque_item(found)?;
        if pending.is_some() {
            self.unify(iter_ty, found)?;
        }
        let by_mut = self.cx.types.intern(Kind::Ref(Mut::Mut, iter_ty));
        let (next, option) = self.lang_call("next", by_mut)?;
        let Kind::Adt(item, args) = *self.cx.types.kind(self.resolve(option)) else { bail!("`next` does not return an `Option`") };
        let elem = self.cx.types.substs(args)[0];
        if let Some(bound) = pending {
            self.unify(elem, bound)?;
        }
        let elem = self.known_ty(elem)?;
        let Some(some) = self.variant_named(item, "Some")? else { bail!("`Option` has no `Some`") };
        self.out.fors.insert(expr, For { into_iter, next, iter: iter_ty, option, some });
        let depth = self.scopes.len();
        self.bind_pat(pat, elem)?;
        self.unit_loop_body(body, label, depth)
    }

    /// The `Item` that a bound of the pending `impl Iterator` value `found` fixes, which a `for` loop over it yields through the blanket `IntoIterator` impl before its hidden type is known.
    fn opaque_item(&mut self, found: Ty) -> Result<Option<Ty>> {
        let Some(iterator) = self.cx.lang_item("next").and_then(|next| self.cx.owner_trait(next)) else { return Ok(None) };
        let found = self.resolve(found);
        self.opaque_binding(iterator, found, "Item")
    }

    pub(super) fn lang_call(&mut self, lang: &str, receiver: Ty) -> Result<(Res, Ty)> {
        let Some(method) = self.cx.lang_item(lang) else { bail!("no `{lang}` lang item") };
        let Some((item, substs)) = self.lang_method(method, receiver)? else { bail!("{} does not implement `{}`", self.show(receiver), self.cx.name(method)) };
        let (_, ret) = self.fn_types(item, &substs)?;
        Ok((Res::fn_of(&self.cx.types, item, &substs), ret))
    }

    /// Infers the range expression `expr` as a struct of the lang item it names.
    pub fn range_expr(&mut self, expr: ExprId, (from, to): (Option<ExprId>, Option<ExprId>), inclusive: bool) -> Result<Ty> {
        let lang = match (from.is_some(), to.is_some(), inclusive) {
            (true, true, false) => "Range",
            (true, false, _) => "RangeFrom",
            (false, true, false) => "RangeTo",
            (false, false, _) => "RangeFull",
            (true, true, true) => "RangeInclusive",
            (false, true, true) => "RangeToInclusive",
        };
        let Some(item) = self.cx.lang_item(lang) else { bail!("no `{lang}` lang item") };
        let args = self.adt_args(item, Vec::new(), true)?;
        let fields = self.adt_fields(item, &args)?[0].clone();
        let mut filled = Vec::new();
        for (end, name) in [(from, "start"), (to, "end")] {
            let Some(end) = end else { continue };
            let index = self.range_field(item, name)?;
            let found = self.expr(end, Some(fields[index as usize]))?;
            self.coerce_expr(end, found, fields[index as usize])?;
            filled.push(index);
        }
        if lang == "RangeInclusive" {
            filled.push(self.range_field(item, "exhausted")?);
        }
        let substs = self.cx.types.intern_substs(&args);
        self.out.ranges.insert(expr, filled.into_boxed_slice());
        self.out.res.insert(expr, Res::Ctor(item, 0, substs));
        Ok(self.cx.types.intern(Kind::Adt(item, substs)))
    }

    fn range_field(&mut self, item: ItemRef, name: &str) -> Result<u32> {
        self.field_index(item, 0, name)?.ok_or_else(|| anyhow::anyhow!("`{}` has no field `{name}`", self.cx.name(item)))
    }
}
