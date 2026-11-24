use anyhow::{Result, bail};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Ty};
use super::Infer;

pub(super) const CALLABLE: [&str; 3] = ["fn", "fn_mut", "fn_once"];
pub(super) const ASYNC_CALLABLE: [&str; 3] = ["async_fn", "async_fn_mut", "async_fn_once"];
pub(super) const CALLABLE_METHODS: [(&str, &str); 6] = [
    ("fn", "call"),
    ("fn_mut", "call_mut"),
    ("fn_once", "call_once"),
    ("async_fn", "async_call"),
    ("async_fn_mut", "async_call_mut"),
    ("async_fn_once", "async_call_once"),
];

/// Whether `lang` names any sync or async callable trait.
pub(crate) fn is_callable_lang(lang: &str) -> bool {
    CALLABLE.contains(&lang) || ASYNC_CALLABLE.contains(&lang)
}

/// Whether `item` is `AsyncFnOnce::async_call_once`.
pub(in crate::sema) fn is_async_call_once(cx: &Context<'_>, item: ItemRef) -> bool {
    cx.owner_trait(item).and_then(|of| cx.lang(of)) == Some("async_fn_once")
}

/// The call method name of callable lang trait `lang`.
pub(in crate::sema) fn callable_method_name(lang: Option<&str>) -> &'static str {
    lang.and_then(|lang| {
        CALLABLE_METHODS
            .iter()
            .find(|&&(trait_lang, _)| trait_lang == lang)
            .map(|&(_, name)| name)
    })
    .unwrap_or("call_once")
}

impl Infer<'_, '_> {
    /// Whether trait `of` is `Fn`, `FnMut` or `FnOnce`.
    pub(in crate::sema) fn is_sync_callable_trait(&self, of: ItemRef) -> bool {
        self.cx
            .lang(of)
            .is_some_and(|lang| CALLABLE.contains(&lang))
    }

    /// Whether trait `of` is `AsyncFn`, `AsyncFnMut` or `AsyncFnOnce`.
    pub(in crate::sema) fn is_async_callable_trait(&self, of: ItemRef) -> bool {
        self.cx
            .lang(of)
            .is_some_and(|lang| ASYNC_CALLABLE.contains(&lang))
    }

    /// Whether trait `of` is a sync or async callable trait, which callables implement with no impl to select.
    pub(in crate::sema) fn is_callable_trait(&self, of: ItemRef) -> bool {
        self.is_sync_callable_trait(of) || self.is_async_callable_trait(of)
    }

    /// The `FnOnce` or `AsyncFnOnce` trait that owns `Output` for callable trait `of`.
    pub(super) fn callable_output_trait(&mut self, of: ItemRef) -> Option<ItemRef> {
        self.cx.lang_item(if self.is_async_callable_trait(of) {
            "async_fn_once"
        } else {
            "fn_once"
        })
    }

    /// Replaces the `Output` return type of an async callable `FnPtr` bound with a fresh variable for closure inference.
    pub(super) fn closure_hint_sig(&mut self, of: ItemRef, sig: Ty) -> Ty {
        if !self.is_async_callable_trait(of) {
            return sig;
        }
        let Kind::FnPtr(inputs, _) = *self.cx.types.kind(sig) else {
            return sig;
        };
        let inputs = self.cx.types.substs(inputs).to_vec();
        let fut = self.fresh(Kind::Var);
        self.cx.types.fn_ptr(&inputs, fut)
    }

    /// Unifies the signature of callable `ty` with callable trait `of`'s `FnPtr` signature `bound`.
    pub(super) fn tie_callable_bound(&mut self, of: ItemRef, ty: Ty, bound: Ty) -> Result<()> {
        let Kind::FnPtr(inputs, output) = *self.cx.types.kind(bound) else {
            return Ok(());
        };
        let inputs = self.cx.types.substs(inputs).to_vec();
        let Some((params, ret)) = self.callable_sig(ty)? else {
            return self.tie_callable_output(of, ty, inputs, output);
        };
        if params.len() != inputs.len() {
            bail!(
                "a callable of {} arguments where its bound takes {}",
                params.len(),
                inputs.len()
            );
        }
        params
            .into_iter()
            .zip(inputs)
            .try_for_each(|(param, input)| self.unify(param, input))?;
        if self.is_async_callable_trait(of) {
            self.tie_future_output(ret, output)
        } else {
            self.unify(ret, output)
        }
    }

    /// Unifies `sync_sig`'s inputs and `<ret as Future>::Output` with `async_bound`'s inputs and `Output`.
    pub(super) fn tie_sync_sig_as_async(&mut self, sync_sig: Ty, async_bound: Ty) -> Result<()> {
        let (&Kind::FnPtr(params, ret), &Kind::FnPtr(inputs, output)) = (
            self.cx.types.kind(sync_sig),
            self.cx.types.kind(async_bound),
        ) else {
            bail!("expected function pointer signatures");
        };
        let (params, inputs) = (
            self.cx.types.substs(params).to_vec(),
            self.cx.types.substs(inputs).to_vec(),
        );
        if params.len() != inputs.len() {
            bail!(
                "a callable of {} arguments where its bound takes {}",
                params.len(),
                inputs.len()
            );
        }
        params
            .into_iter()
            .zip(inputs)
            .try_for_each(|(param, input)| self.unify(param, input))?;
        self.tie_future_output(ret, output)
    }

    fn tie_callable_output(
        &mut self,
        of: ItemRef,
        ty: Ty,
        inputs: Vec<Ty>,
        output: Ty,
    ) -> Result<()> {
        let once = self.callable_output_trait(of);
        let (Some(once), false) = (
            once,
            matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)),
        ) else {
            return Ok(());
        };
        let tuple = self.cx.types.tuple(&inputs);
        let found = self.project(once, ty, vec![tuple], "Output")?;
        self.unify(found, output)
    }

    /// Unifies `<fut as Future>::Output` with `output` and verifies `fut: Future`.
    pub(super) fn tie_future_output(&mut self, fut: Ty, output: Ty) -> Result<()> {
        let Some(future_trait) = self.cx.lang_item("future_trait") else {
            bail!("core has no `Future`");
        };
        let found = self.project(future_trait, fut, Vec::new(), "Output")?;
        self.unify(found, output)?;
        let resolved = self.resolve(fut);
        if !self.compiler_impl(future_trait, resolved)
            && self.object_holds(future_trait, resolved, &[]) != Some(true)
        {
            if self.has_vars(resolved) {
                self.obligations.push((future_trait, vec![resolved]));
            } else if !self.poly_may_implement(future_trait, resolved, &[])?
                && self.select(future_trait, resolved, &[])?.is_none()
            {
                bail!("{} is not a future", self.show(resolved));
            }
        }
        Ok(())
    }

    /// Whether `ty` implements `Future` without emitting new deferred obligations.
    pub(super) fn is_future_type(&mut self, ty: Ty) -> Result<bool> {
        let Some(future_trait) = self.cx.lang_item("future_trait") else {
            return Ok(false);
        };
        let resolved = self.resolve(ty);
        if self.compiler_impl(future_trait, resolved)
            || self.object_holds(future_trait, resolved, &[]) == Some(true)
        {
            return Ok(true);
        }
        if self.has_vars(resolved) {
            return Ok(false);
        }
        Ok(self.poly_may_implement(future_trait, resolved, &[])?
            || self.select(future_trait, resolved, &[])?.is_some())
    }

    /// Checks a compiler-implemented callable trait obligation `self_ty: of<args[0]>`.
    pub(in crate::sema) fn check_callable_obligation(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> Result<bool> {
        if !self.is_callable_trait(of) {
            return Ok(false);
        }
        if !matches!(
            self.cx.types.kind(self.resolve(self_ty)),
            Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..)
        ) {
            return Ok(false);
        }
        let Some((params, ret)) = self.callable_sig(self_ty)? else {
            return Ok(false);
        };
        let expected = self.cx.types.tuple(&params);
        let Some(&actual) = args.first() else {
            return Ok(false);
        };
        let mark = self.mark_vars();
        if !self.unifies(expected, actual) {
            self.rollback_vars(mark);
            return Ok(false);
        }
        if self.is_async_callable_trait(of) && !self.is_future_type(ret)? {
            self.rollback_vars(mark);
            return Ok(false);
        }
        self.commit_vars(mark);
        Ok(true)
    }

    /// Built-in associated type projection for `FnOnce::Output`, `AsyncFnOnce::Output`, `AsyncFnOnce::CallOnceFuture` and `AsyncFnMut::CallRefFuture`.
    pub(super) fn callable_assoc_alias(
        &mut self,
        of: ItemRef,
        ty: Ty,
        name: &str,
    ) -> Result<Option<Ty>> {
        enum CallableAssoc {
            FnOnceOutput,
            AsyncFuture,
            AsyncOutput,
        }
        let assoc = match (self.cx.lang(of), name) {
            (Some("fn_once"), "Output") => CallableAssoc::FnOnceOutput,
            (Some("async_fn_once"), "CallOnceFuture") => CallableAssoc::AsyncFuture,
            (Some("async_fn" | "async_fn_mut"), "CallRefFuture") => CallableAssoc::AsyncFuture,
            (Some("async_fn" | "async_fn_mut" | "async_fn_once"), "Output") => {
                CallableAssoc::AsyncOutput
            }
            _ => return Ok(None),
        };
        if matches!(assoc, CallableAssoc::FnOnceOutput) {
            return Ok(self.callable_sig(ty)?.map(|(_, ret)| ret));
        }
        let kind_ok = matches!(
            self.cx.types.kind(self.resolve(ty)),
            Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..) | Kind::Var(_)
        );
        if !kind_ok {
            return Ok(None);
        }
        let Some((_, ret)) = self.callable_sig(ty)? else {
            return Ok(None);
        };
        match assoc {
            CallableAssoc::AsyncFuture => Ok(Some(ret)),
            CallableAssoc::AsyncOutput => {
                let Some(future_trait) = self.cx.lang_item("future_trait") else {
                    bail!("core has no `Future`");
                };
                self.project(future_trait, ret, Vec::new(), "Output")
                    .map(Some)
            }
            CallableAssoc::FnOnceOutput => unreachable!(),
        }
    }

    /// Checks a compiler-implemented callable trait bound in `bound_holds`, including any associated type bindings on angle-bracketed syntax.
    pub(super) fn callable_bound_holds(
        &mut self,
        (arena, home, bound): (&crate::body::Body, crate::names::ModRef, u32),
        (of, path, ty): (ItemRef, crate::body::PathId, Ty),
        depth: u32,
    ) -> Result<bool> {
        let Some((_, sig)) = self.callable_bound(arena, home, bound)? else {
            return Ok(true);
        };
        let mark = self.mark_vars();
        if self.tie_callable_bound(of, ty, sig).is_err() {
            self.rollback_vars(mark);
            return Ok(false);
        }
        let args = self.bound_args(of, arena, path, ty)?;
        if super::project::angle_args(arena, path).is_some()
            && !self.assoc_bounds_hold((arena, home, path), (of, ty, &args), depth)?
        {
            self.rollback_vars(mark);
            return Ok(false);
        }
        self.commit_vars(mark);
        Ok(true)
    }

    pub(super) fn poly_callable_bound_holds(
        &mut self,
        (arena, home, bound): (&crate::body::Body, crate::names::ModRef, u32),
        (of, path, ty): (ItemRef, crate::body::PathId, Ty),
        depth: u32,
    ) -> Result<bool> {
        let ty = self.resolve(ty);
        if let Some((_, wanted)) = self.callable_bound(arena, home, bound)? {
            if let Some(sig) = self.poly_fn_bound_of(ty, of) {
                return Ok(self.unifies(sig, wanted));
            }
            if let Some(sync_of) = self.matching_sync_trait(of)
                && let Some(sync_sig) = self.poly_fn_bound_of(ty, sync_of)
            {
                let held = self.trial(|infer| {
                    Ok(infer
                        .tie_sync_sig_as_async(sync_sig, wanted)
                        .is_ok()
                        .then_some(()))
                })?;
                return Ok(held.is_some());
            }
        }
        let args = self.bound_args(of, arena, path, ty)?;
        self.poly_env_holds((arena, home, path), (of, ty, &args), depth)
    }

    fn poly_fn_bound_of(&self, ty: Ty, of: ItemRef) -> Option<Ty> {
        self.poly
            .as_ref()?
            .fn_bounds
            .iter()
            .find(|&&(t, o, _)| self.resolve(t) == ty && o == of)
            .map(|&(_, _, s)| s)
    }

    fn matching_sync_trait(&mut self, of: ItemRef) -> Option<ItemRef> {
        let sync_lang = match self.cx.lang(of)? {
            "async_fn" => "fn",
            "async_fn_mut" => "fn_mut",
            "async_fn_once" => "fn_once",
            _ => return None,
        };
        self.cx.lang_item(sync_lang)
    }

    pub(super) fn poly_callable_target(&mut self, callee_ty: Ty) -> Option<(ItemRef, Vec<Ty>, Ty)> {
        let callee_ty = self.resolve(callee_ty);
        let bounds = self.poly.as_ref()?.fn_bounds.clone();
        for (lang, name) in CALLABLE_METHODS {
            let Some(of) = self.cx.lang_item(lang) else {
                continue;
            };
            let Some(&(_, _, sig)) = bounds
                .iter()
                .find(|&&(t, o, _)| o == of && self.resolve(t) == callee_ty)
            else {
                continue;
            };
            let Kind::FnPtr(params, output) = *self.cx.types.kind(sig) else {
                continue;
            };
            let method = self.cx.trait_member(of, name)?;
            let params = self.cx.types.substs(params).to_vec();
            let ret = if ASYNC_CALLABLE.contains(&lang) {
                let fut_lang = if lang == "async_fn_once" {
                    "async_fn_once"
                } else {
                    "async_fn_mut"
                };
                let fut_name = if lang == "async_fn_once" {
                    "CallOnceFuture"
                } else {
                    "CallRefFuture"
                };
                let fut_trait = self.cx.lang_item(fut_lang)?;
                let tuple = self.cx.types.tuple(&params);
                self.project(fut_trait, callee_ty, vec![tuple], fut_name)
                    .ok()?
            } else {
                output
            };
            return Some((method, params, ret));
        }
        None
    }
}
