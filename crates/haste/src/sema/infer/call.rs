use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, List};
use crate::names::ItemRef;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Infers the call `expr` of the `dyn Fn` or `dyn FnMut` value of type `ty`, recording it as a call of `call` or `call_mut` on the trait object.
    pub(super) fn object_call(
        &mut self,
        (expr, found): (ExprId, Ty),
        ty: Ty,
        args: List,
    ) -> Result<Ty> {
        let Kind::Dyn(of, parts, members, _) = *self.cx.types.kind(ty) else {
            bail!("an object call of a type that is not a trait object")
        };
        let (tuple, bound_ret) = {
            let (given, mut bound) = super::super::ty::object_parts(
                self.cx.types.substs(parts),
                self.cx.types.items(members),
            );
            let Some(&tuple) = given.first() else {
                bail!("a callable trait object with no argument tuple")
            };
            (tuple, bound.next().map(|(_, ret)| ret))
        };
        let ret = bound_ret.unwrap_or_else(|| self.cx.types.unit());
        let Kind::Tuple(params) = *self.cx.types.kind(self.resolve(tuple)) else {
            bail!("a callable trait object whose arguments are not a tuple")
        };
        let params = self.cx.types.substs(params).to_vec();
        self.infer_call_args(args, &params, None)?;
        let lang = self.cx.lang(of);
        let name = super::callable::callable_method_name(lang);
        if matches!(name, "call_once" | "async_call_once") {
            self.free_box(expr, found)?;
        }
        let Some(method) = self.cx.trait_member(of, name) else {
            bail!("`{}` has no `{name}`", self.cx.name(of))
        };
        self.out
            .res
            .insert(expr, Res::fn_of(&self.cx.types, method, &[ty, tuple]));
        Ok(ret)
    }

    /// Infers the call `expr` of a value of type `found`, stepping through references, `Box`es and `Deref` impls to a callable impl or callable target.
    pub(super) fn overloaded_call(
        &mut self,
        expr: ExprId,
        found: Ty,
        args_list: List,
    ) -> Result<Ty> {
        let params: Vec<Ty> = self
            .body
            .list(args_list)
            .iter()
            .map(|_| self.fresh(Kind::Var))
            .collect();
        let tuple = self.cx.types.tuple(&params);
        let (mut ty, mut steps) = (self.resolve(found), 0);
        loop {
            if let Some(method) = self.callable_impl(ty, tuple)? {
                return self.finish_impl_call(
                    (expr, found, steps),
                    (method, ty, tuple),
                    (args_list, &params),
                );
            }
            if let Some(ret) = self.step_target_call((expr, found, steps), ty, args_list)? {
                return Ok(ret);
            }
            let next = match self.builtin_deref(ty) {
                Some(inner) => Some(inner),
                None => self.deref_step(expr, steps, ty)?,
            };
            let Some(next) = next else {
                return Err(self.diag_not_callable(expr, found));
            };
            if steps + 1 >= crate::RECURSION_LIMIT as u32 {
                return Err(self.diag_deref_limit(expr, found));
            }
            (ty, steps) = (self.resolve(next), steps + 1);
        }
    }

    fn finish_impl_call(
        &mut self,
        step: (ExprId, Ty, u32),
        (method, ty, tuple): (ItemRef, Ty, Ty),
        (args, params): (List, &[Ty]),
    ) -> Result<Ty> {
        self.record_call_steps(step, super::by_reference(self.cx, method) == Some(Mut::Mut))?;
        self.infer_call_args(args, params, None)?;
        self.out
            .res
            .insert(step.0, Res::fn_of(&self.cx.types, method, &[ty, tuple]));
        self.out.overloaded.insert(step.0);
        let Some(once) = self.cx.lang_item("fn_once") else {
            bail!("core has no `FnOnce`")
        };
        self.project(once, ty, vec![tuple], "Output")
    }

    fn step_target_call(
        &mut self,
        step: (ExprId, Ty, u32),
        ty: Ty,
        args: List,
    ) -> Result<Option<Ty>> {
        match self.cx.types.kind(ty).clone() {
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => {
                self.record_call_steps(
                    step,
                    matches!(self.cx.lang(of), Some("fn_mut" | "async_fn_mut")),
                )?;
                self.object_call((step.0, step.1), ty, args).map(Some)
            }
            Kind::Closure(..) => {
                let mutably = self.closure_mutates(ty)?;
                self.record_call_steps(step, mutably)?;
                self.closure_call(step.0, ty, args).map(Some)
            }
            Kind::FnDef(item, substs) => {
                self.record_call_steps(step, false)?;
                self.out.res.insert(step.0, Res::Fn(item, substs));
                let given = self.cx.types.substs(substs).to_vec();
                let (params, ret) = self.fn_types(item, &given)?;
                self.infer_call_args(args, &params, Some((item, &given)))?;
                Ok(Some(ret))
            }
            Kind::FnPtr(params, ret) => {
                self.record_call_steps(step, false)?;
                let params = self.cx.types.substs(params).to_vec();
                self.infer_call_args(args, &params, None)?;
                Ok(Some(ret))
            }
            _ => Ok(None),
        }
    }

    fn record_call_steps(
        &mut self,
        (expr, found, steps): (ExprId, Ty, u32),
        mutably: bool,
    ) -> Result<()> {
        self.out.autoderefs.remove(&expr);
        if steps > 0 {
            self.out.autoderefs.insert(expr, steps);
        }
        if mutably {
            self.steps_mutably(expr, found, steps)?;
            if let Expr::Call(callee, _) = *self.body.expr(expr) {
                self.note_mut(callee)?;
            }
        }
        Ok(())
    }

    pub(super) fn infer_call_args(
        &mut self,
        args: List,
        params: &[Ty],
        def: Option<(ItemRef, &[Ty])>,
    ) -> Result<()> {
        let args = self.body.list(args).to_vec();
        let variadic = match def {
            Some((item, _)) => self.cx.foreign(item) && self.cx.signature(item)?.variadic,
            None => false,
        };
        if args.len() < params.len() || (!variadic && args.len() != params.len()) {
            return Err(self.diag_arity(None, params.len(), args.len()));
        }
        for (&arg, &param) in args.iter().zip(params) {
            self.call_arg(def, ExprId(arg), (param, None), &mut None)?;
        }
        for &arg in &args[params.len()..] {
            self.expr(ExprId(arg), None)?;
        }
        Ok(())
    }

    /// `Fn::call`, `FnMut::call_mut`, `FnOnce::call_once`, or their `AsyncFn*` counterparts, the first that an impl gives `ty` at the argument tuple `tuple`.
    pub(in crate::sema) fn callable_impl(&mut self, ty: Ty, tuple: Ty) -> Result<Option<ItemRef>> {
        for (lang, name) in super::callable::CALLABLE_METHODS {
            let Some(of) = self.cx.lang_item(lang) else {
                continue;
            };
            if self
                .trial(|infer| Ok(infer.select(of, ty, &[tuple])?.map(|_| ())))?
                .is_some()
            {
                return Ok(self.cx.trait_member(of, name));
            }
        }
        Ok(None)
    }

    /// Takes back the `Self` obligation of a call of trait item `item` at `substs` while a trait argument is still unknown, so the arguments are typed before an impl is picked, as rustc selects a callee's obligations only after its arguments that are not closures.
    pub(super) fn hold_open_obligation(
        &mut self,
        def: Option<&(ItemRef, Vec<Ty>)>,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some((of, substs)) =
            def.and_then(|(item, substs)| Some((self.cx.owner_trait(*item)?, substs)))
        else {
            return Ok(None);
        };
        let count = 1 + super::scope::type_params(&self.cx.trait_generics(of)?.1.params).count();
        let Some(outer) = substs.get(..count) else {
            return Ok(None);
        };
        let open = outer[1..]
            .iter()
            .any(|&ty| matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)));
        let Some(at) = self
            .obligations
            .iter()
            .position(|(held, tys)| open && *held == of && tys[..] == *outer)
        else {
            return Ok(None);
        };
        Ok(Some(self.obligations.remove(at)))
    }

    /// Gives back the held obligation once an argument of type `found` is typed and before it is coerced to the open parameter `param`, as rustc selects pending obligations there, when the impl that picks takes that argument; an open trait argument may have impls here that selection does not see, so a pick the argument does not fit stays held until the arguments are coerced.
    pub(super) fn release_if_fits(
        &mut self,
        held: &mut Option<(ItemRef, Vec<Ty>)>,
        found: Ty,
        param: Ty,
    ) -> Result<()> {
        let Some(entry) = held.clone() else {
            return Ok(());
        };
        if !matches!(self.cx.types.kind(self.resolve(param)), Kind::Var(_)) {
            return Ok(());
        }
        let saved = self.save_deferred();
        let res = self.trial(|infer| {
            infer.obligations.push(entry);
            if infer.settle_obligations().is_err()
                || matches!(infer.cx.types.kind(infer.resolve(param)), Kind::Var(_))
            {
                return Ok(None);
            }
            Ok(infer.coerces_to(found, param).then_some(()))
        });
        if matches!(res, Ok(Some(()))) {
            *held = None;
        } else {
            self.restore_deferred(saved);
        }
        res.map(drop)
    }

    /// Gives back an obligation that `hold_open_obligation` took and settles what it now decides.
    pub(super) fn release_obligation(
        &mut self,
        held: &mut Option<(ItemRef, Vec<Ty>)>,
    ) -> Result<()> {
        let Some(entry) = held.take() else {
            return Ok(());
        };
        self.obligations.push(entry);
        self.settle_pending()
    }

    /// Infers the arguments `args` of a call of a function with `params` that returns `ret`, the function item `def` when it is one, and returns `ret`.
    pub(super) fn call_args(
        &mut self,
        args: Vec<u32>,
        (params, ret, def): (Vec<Ty>, Ty, Option<(ItemRef, Vec<Ty>)>),
        (expected, mut held): (Option<Ty>, Option<(ItemRef, Vec<Ty>)>),
    ) -> Result<Ty> {
        let extra = self.variadic_extra(
            def.as_ref().map(|(item, _)| *item),
            args.len(),
            params.len(),
        )?;
        let hints = self.expected_inputs(&params, ret, expected);
        let value_count = params.len();
        let def_ref = def.as_ref().map(|(item, substs)| (*item, &substs[..]));
        for (&arg, (&param, &hint)) in args[..value_count].iter().zip(params.iter().zip(&hints)) {
            if !self.is_deferred_closure_arg(ExprId(arg)) {
                self.call_arg(def_ref, ExprId(arg), (param, hint), &mut held)?;
            }
        }
        self.release_obligation(&mut held)?;
        for (&arg, (&param, &hint)) in args[..value_count].iter().zip(params.iter().zip(&hints)) {
            if self.is_deferred_closure_arg(ExprId(arg)) {
                self.call_arg(def_ref, ExprId(arg), (param, hint), &mut held)?;
            }
        }
        for &arg in &args[args.len() - extra..] {
            self.expr(ExprId(arg), None)?;
        }
        Ok(ret)
    }

    fn is_deferred_closure_arg(&self, expr: ExprId) -> bool {
        self.is_closure(expr)
            || matches!(*self.body.expr(expr), Expr::Ref { expr: inner, .. } if self.is_closure(inner))
    }

    /// How many arguments past its `params` a call of `callee` with `given` arguments passes, which only a C-variadic foreign function takes.
    fn variadic_extra(
        &mut self,
        callee: Option<ItemRef>,
        given: usize,
        params: usize,
    ) -> Result<usize> {
        let variadic = match callee {
            Some(item) => self.cx.foreign(item) && self.cx.signature(item)?.variadic,
            None => false,
        };
        match given.checked_sub(params) {
            Some(0) => Ok(0),
            Some(extra) if variadic => Ok(extra),
            _ => Err(self.diag_arity(None, params, given)),
        }
    }

    pub(super) fn expected_inputs(
        &mut self,
        params: &[Ty],
        ret: Ty,
        expected: Option<Ty>,
    ) -> Vec<Option<Ty>> {
        let Some(expected) = expected else {
            return vec![None; params.len()];
        };
        self.probe(|infer| match infer.unifies(ret, expected) {
            true => params
                .iter()
                .map(|&param| Some(infer.substitute(param)))
                .collect(),
            false => vec![None; params.len()],
        })
    }

    fn free_box(&mut self, expr: ExprId, found: Ty) -> Result<()> {
        let found = self.resolve(found);
        if self.cx.boxed(found).is_none() || self.out.autoderefs.get(&expr) != Some(&1) {
            bail!("a `dyn FnOnce` is called by value only through a `Box`");
        }
        let Some((block, args)) = self.drop_impl(found)? else {
            bail!("`Box` has no `Drop` impl")
        };
        let Some(member) = self.cx.member(block, "drop") else {
            bail!("the `Drop` impl of `Box` has no `drop`")
        };
        let args: Vec<Ty> = args.into_iter().map(|arg| self.zonk(arg)).collect();
        self.out
            .box_frees
            .insert(expr, Res::fn_of(&self.cx.types, member, &args));
        Ok(())
    }
}
