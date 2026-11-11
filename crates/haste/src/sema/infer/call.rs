use anyhow::{Result, bail};

use crate::body::{Expr, ExprId, List};
use crate::names::ItemRef;

use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

impl Infer<'_, '_> {
    /// Infers the call `expr` of the `dyn Fn` or `dyn FnMut` value of type `ty`, recording it as a call of `call` or `call_mut` on the trait object.
    pub(super) fn object_call(&mut self, (expr, found): (ExprId, Ty), ty: Ty, args: List) -> Result<Ty> {
        let Kind::Dyn(of, parts, members, _) = self.cx.types.kind(ty).clone() else { bail!("an object call of a type that is not a trait object") };
        let (tuple, bound_ret) = {
            let (given, mut bound) = super::super::ty::object_parts(self.cx.types.substs(parts), &members);
            let Some(&tuple) = given.first() else { bail!("a callable trait object with no argument tuple") };
            (tuple, bound.next().map(|(_, ret)| ret))
        };
        let ret = bound_ret.unwrap_or_else(|| self.cx.types.unit());
        let Kind::Tuple(params) = *self.cx.types.kind(self.resolve(tuple)) else { bail!("a callable trait object whose arguments are not a tuple") };
        let params = self.cx.types.substs(params).to_vec();
        self.infer_call_args(args, &params, None)?;
        let name = match self.cx.lang(of) {
            Some("fn") => "call",
            Some("fn_mut") => "call_mut",
            _ => {
                self.free_box(expr, found)?;
                "call_once"
            }
        };
        let Some(method) = self.cx.trait_member(of, name) else { bail!("`{}` has no `{name}`", self.cx.name(of)) };
        self.out.res.insert(expr, Res::fn_of(&self.cx.types, method, &[ty, tuple]));
        Ok(ret)
    }

    /// Infers the call `expr` of a value of type `found`, stepping through references, `Box`es and `Deref` impls to a callable impl or callable target.
    pub(super) fn overloaded_call(&mut self, expr: ExprId, found: Ty, args_list: List) -> Result<Ty> {
        let params: Vec<Ty> = self.body.list(args_list).iter().map(|_| self.fresh(Kind::Var)).collect();
        let tuple = self.cx.types.tuple(&params);
        let (mut ty, mut steps) = (self.resolve(found), 0);
        loop {
            if let Some(method) = self.callable_impl(ty, tuple)? {
                return self.finish_impl_call((expr, found, steps), (method, ty, tuple), (args_list, &params));
            }
            if let Some(ret) = self.step_target_call((expr, found, steps), ty, args_list)? {
                return Ok(ret);
            }
            let next = match self.builtin_deref(ty) {
                Some(inner) => Some(inner),
                None => self.deref_step(expr, steps, ty)?,
            };
            let Some(next) = next else { bail!("calling {} is not supported yet", self.show(found)) };
            if steps + 1 >= crate::RECURSION_LIMIT as u32 {
                bail!("reached the recursion limit while dereferencing {}", self.show(found));
            }
            (ty, steps) = (self.resolve(next), steps + 1);
        }
    }

    fn finish_impl_call(&mut self, step: (ExprId, Ty, u32), (method, ty, tuple): (ItemRef, Ty, Ty), (args, params): (List, &[Ty])) -> Result<Ty> {
        self.record_call_steps(step, super::by_reference(self.cx, method) == Some(Mut::Mut))?;
        self.infer_call_args(args, params, None)?;
        self.out.res.insert(step.0, Res::fn_of(&self.cx.types, method, &[ty, tuple]));
        self.out.overloaded.insert(step.0);
        let Some(once) = self.cx.lang_item("fn_once") else { bail!("core has no `FnOnce`") };
        self.project(once, ty, vec![tuple], "Output")
    }

    fn step_target_call(&mut self, step: (ExprId, Ty, u32), ty: Ty, args: List) -> Result<Option<Ty>> {
        match self.cx.types.kind(ty).clone() {
            Kind::Dyn(of, ..) if self.is_callable_trait(of) => {
                self.record_call_steps(step, self.cx.lang(of) == Some("fn_mut"))?;
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

    fn record_call_steps(&mut self, (expr, found, steps): (ExprId, Ty, u32), mutably: bool) -> Result<()> {
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

    pub(super) fn infer_call_args(&mut self, args: List, params: &[Ty], def: Option<(ItemRef, &[Ty])>) -> Result<()> {
        let args = self.body.list(args).to_vec();
        let variadic = match def {
            Some((item, _)) => self.cx.foreign(item) && self.cx.signature(item)?.variadic,
            None => false,
        };
        if args.len() < params.len() || (!variadic && args.len() != params.len()) {
            bail!("call takes {} arguments, {} given", params.len(), args.len());
        }
        for (&arg, &param) in args.iter().zip(params) {
            self.call_arg(def, ExprId(arg), (param, None))?;
        }
        for &arg in &args[params.len()..] {
            self.expr(ExprId(arg), None)?;
        }
        Ok(())
    }

    /// `Fn::call`, `FnMut::call_mut` or `FnOnce::call_once`, the first that an impl gives `ty` at the argument tuple `tuple`.
    pub(in crate::sema) fn callable_impl(&mut self, ty: Ty, tuple: Ty) -> Result<Option<ItemRef>> {
        for (lang, name) in [("fn", "call"), ("fn_mut", "call_mut"), ("fn_once", "call_once")] {
            let Some(of) = self.cx.lang_item(lang) else {
                continue;
            };
            if self.trial(|infer| Ok(infer.select(of, ty, &[tuple])?.map(|_| ())))?.is_some() {
                return Ok(self.cx.trait_member(of, name));
            }
        }
        Ok(None)
    }

    fn free_box(&mut self, expr: ExprId, found: Ty) -> Result<()> {
        let found = self.resolve(found);
        if self.cx.boxed(found).is_none() || self.out.autoderefs.get(&expr) != Some(&1) {
            bail!("a `dyn FnOnce` is called by value only through a `Box`");
        }
        let Some((block, args)) = self.drop_impl(found)? else { bail!("`Box` has no `Drop` impl") };
        let Some(member) = self.cx.member(block, "drop") else { bail!("the `Drop` impl of `Box` has no `drop`") };
        let args: Vec<Ty> = args.into_iter().map(|arg| self.zonk(arg)).collect();
        self.out.box_frees.insert(expr, Res::fn_of(&self.cx.types, member, &args));
        Ok(())
    }
}
