use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};

use crate::names::ItemRef;

use super::super::super::Context;
use super::super::super::substs::Substs;
use super::super::super::ty::{Kind, Mut, Ty};
use super::super::{Local, Res, Shape, Typed};
use super::{By, ClosureBody, ClosureInfo};

/// Whether `item` is `Fn::call`, `AsyncFn::async_call`, `FnMut::call_mut` or `AsyncFnMut::async_call_mut`, and how each takes the closure.
pub fn by_reference(cx: &Context<'_>, item: ItemRef) -> Option<Mut> {
    match cx.owner_trait(item).and_then(|of| cx.lang(of)) {
        Some("fn" | "async_fn") => Some(Mut::Not),
        Some("fn_mut" | "async_fn_mut") => Some(Mut::Mut),
        _ => None,
    }
}

/// The typed form of a call trait method at a closure, function item or function pointer and its argument tuple, which calls the closure's body, the function or the pointer that the receiver `Res::Local(0)` points at, as a vtable passes it, with the tuple's fields.
pub fn untuple_typed(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Typed> {
    let slice = cx.types.substs(substs);
    let (Some(&callee), Some(&tuple)) = (slice.first(), slice.get(1)) else {
        bail!("a call trait method with no callee and arguments")
    };
    let (target, ret) = match *cx.types.kind(callee) {
        Kind::FnPtr(_, ret) => (Res::Local(0), ret),
        Kind::FnDef(target, args) => {
            let arg_slice = cx.types.substs(args).to_vec();
            let (res, (_, ret)) = super::super::detached(cx, target, |infer| {
                Ok((
                    infer.trait_call(target, args)?,
                    infer.fn_types(target, &arg_slice)?,
                ))
            })?;
            (res, ret)
        }
        _ => {
            let info = cx.ensure_closure_res(callee, false)?;
            let Some(call_once) = cx.call_once() else {
                bail!("core has no `FnOnce::call_once`")
            };
            (
                Res::fn_of(&cx.types, call_once, &[callee, tuple]),
                info.closure.ret,
            )
        }
    };
    let pointer = matches!(cx.types.kind(callee), Kind::FnPtr(..)).then_some(Mut::Not);
    let receiver = by_reference(cx, item)
        .or(pointer)
        .map_or(callee, |by| cx.types.intern(Kind::Ref(by, callee)));
    Ok(Typed {
        locals: vec![Local { ty: receiver }, Local { ty: tuple }],
        params: vec![0, 1],
        ret: Some(ret),
        shape: Shape::Untuple(target),
        ..Typed::default()
    })
}

/// Records the closures of the finished body `typed` of `parent` at `substs` in `cx`, so other bodies can call them.
pub fn register(cx: &mut Context<'_>, (parent, substs): (ItemRef, Substs), typed: &Typed) {
    if !typed.has_closures() {
        return;
    }
    let typed = typed.scoped(&cx.types);
    typed.ensure_tables(&cx.types);
    let shared = Arc::new(Typed {
        drops: Vec::new(),
        ..(*typed).clone()
    });
    let (parent_drops, parent_laid) = (Arc::new(OnceLock::new()), Arc::new(AtomicBool::new(false)));
    for (at, closure) in &typed.closures {
        let ty = cx.types.intern(Kind::Closure(parent, substs, at.0));
        let parts: Vec<Ty> = closure
            .captures
            .iter()
            .map(|c| {
                let held = typed.locals[c.local as usize].ty;
                let held = c.ty(cx, held).unwrap_or(held);
                match c.by {
                    By::Ref => cx.types.intern(Kind::Ref(Mut::Not, held)),
                    By::Mut => cx.types.intern(Kind::Ref(Mut::Mut, held)),
                    By::Value => held,
                }
            })
            .collect();
        let mut env = cx.types.tuple(&parts);
        if closure.coroutine {
            let word = cx.types.prim(crate::names::Primitive::Usize);
            env = cx.types.tuple(&[word, word, env]);
        }
        cx.set_closure(
            ty,
            ClosureInfo {
                parent,
                typed: shared.clone(),
                closure: closure.clone(),
                env,
                parent_drops: parent_drops.clone(),
                parent_laid: parent_laid.clone(),
            },
        );
    }
}

fn finish_closure(
    cx: &mut Context<'_>,
    mut typed: Typed,
    info: &ClosureInfo,
    env: Option<u32>,
    ret: Ty,
) -> Result<Typed> {
    if env.is_some() {
        for cap in &info.closure.captures {
            if !cap.path.is_empty() {
                typed.locals.push(Local {
                    ty: cap.ty(cx, typed.locals[cap.local as usize].ty)?,
                });
            }
        }
    }
    typed.ret = Some(ret);
    typed.shape = Shape::Closure(ClosureBody {
        parent: info.parent,
        closure: info.closure.clone(),
        env,
        parent_laid: info.parent_laid.load(Ordering::Acquire),
        parent_locals: info.typed.locals.len() as u32,
    });
    typed.drops = super::super::drop::closure_drop_calls(cx, &typed, info)?;
    Ok(typed)
}

/// The typed form of `Future::poll` at the coroutine type `ty`: its defining body, resumed with the pinned coroutine and the task context, returning `Poll` of its output.
pub fn coroutine_typed(cx: &mut Context<'_>, poll: ItemRef, ty: Ty) -> Result<Typed> {
    let info = cx.ensure_closure_res(ty, true)?;
    let (params, ret) = super::super::detached(cx, poll, |infer| infer.fn_types(poll, &[ty]))?;
    let mut typed = (*info.typed).clone();
    let first = typed.locals.len() as u32;
    typed.locals.extend(params.iter().map(|&ty| Local { ty }));
    typed.params = vec![first, first + 1];
    finish_closure(cx, typed, &info, Some(first), ret)
}

/// The typed form of the function that the closure type `ty` runs: its defining body, with an environment parameter and the closure's own parameters.
pub fn closure_typed(cx: &mut Context<'_>, ty: Ty) -> Result<Typed> {
    let info = cx.ensure_closure_res(ty, true)?;
    let mut typed = (*info.typed).clone();
    let env = cx.closure_takes_env(ty).then(|| {
        typed.locals.push(Local {
            ty: cx.types.intern(Kind::Ref(Mut::Mut, info.env)),
        });
        typed.locals.len() as u32 - 1
    });
    typed.params = env
        .into_iter()
        .chain(info.closure.params.iter().copied())
        .collect();
    finish_closure(cx, typed, &info, env, info.closure.ret)
}

impl<'a> Context<'a> {
    /// Returns the recorded closure info for `ty`, lazily inferring its defining body once when needed.
    pub fn ensure_closure(&mut self, ty: Ty) -> Option<Arc<ClosureInfo>> {
        self.ensure_closure_res(ty, false).ok()
    }

    pub(in crate::sema) fn ensure_closure_env(&mut self, ty: Ty) -> Result<Ty> {
        if let Some(info) = self.closure_ready(ty, false) {
            return Ok(info.env);
        }
        if let Some(info) = self.load_closure_owner(ty, false)? {
            return Ok(info.env);
        }
        self.async_coroutine_env(ty)?.ok_or_else(|| {
            anyhow::anyhow!("layout of a closure whose defining body was not inferred")
        })
    }

    fn closure_ready(&self, ty: Ty, full: bool) -> Option<Arc<ClosureInfo>> {
        self.store
            .closures
            .get(&ty)
            .filter(|info| !full || !info.typed.exprs.is_empty())
    }

    fn ensure_closure_res(&mut self, ty: Ty, full: bool) -> Result<Arc<ClosureInfo>> {
        self.closure_ready(ty, full).map_or_else(
            || {
                self.load_closure_owner(ty, full)?.ok_or_else(|| {
                    anyhow::anyhow!("a closure whose defining body was not inferred")
                })
            },
            Ok,
        )
    }

    fn load_closure_owner(&mut self, ty: Ty, full: bool) -> Result<Option<Arc<ClosureInfo>>> {
        let Kind::Closure(parent, substs, _) = *self.types.kind(ty) else {
            bail!("not a closure")
        };
        if let Some(info) = self.register_final_closure((parent, substs), ty, full) {
            return Ok(Some(info));
        }
        if crate::sema::reach::is_inferring((parent, substs)) {
            return Ok(None);
        }
        let mut cycled = false;
        let reveals = self.store.lookups.reveals.clone();
        reveals.run(
            self,
            (parent, substs),
            |cx| {
                Ok(
                    crate::sema::reach::typed_once(cx, &(parent, substs), &mut Vec::new())
                        .map_err(|m| anyhow::anyhow!("{m}"))?
                        .ret
                        .unwrap_or(Ty::NONE),
                )
            },
            |_| {
                cycled = true;
                Ok(Ty::NONE)
            },
        )?;
        if let Some(info) = self.closure_ready(ty, full) {
            return Ok(Some(info));
        }
        if cycled {
            return Ok(None);
        }
        if let Some(info) = self.register_final_closure((parent, substs), ty, full) {
            return Ok(Some(info));
        }
        let typed = crate::sema::reach::typed_once(self, &(parent, substs), &mut Vec::new())
            .map_err(|m| anyhow::anyhow!("{m}"))?;
        register(self, (parent, substs), &typed);
        Ok(Some(self.closure_ready(ty, full).ok_or_else(|| {
            anyhow::anyhow!(
                "closure owner `{}` did not record closure",
                self.name(parent)
            )
        })?))
    }

    fn register_final_closure(
        &mut self,
        instance: (ItemRef, Substs),
        ty: Ty,
        full: bool,
    ) -> Option<Arc<ClosureInfo>> {
        let typed = self.store.finals.get(&instance)?;
        register(self, instance, &typed);
        self.closure_ready(ty, full)
    }

    fn async_coroutine_env(&mut self, ty: Ty) -> Result<Option<Ty>> {
        let Kind::Closure(parent, substs, at) = *self.types.kind(ty) else {
            return Ok(None);
        };
        let sig = self.signature(parent)?;
        if !sig.is_async || self.body(parent)?.root.is_none_or(|root| root.0 != at) {
            return Ok(None);
        }
        if sig.params.iter().any(|&(pat, _)| {
            !matches!(
                *sig.body.pat(pat),
                crate::body::Pat::Bind {
                    sub: None,
                    by_ref: false,
                    ..
                } | crate::body::Pat::Wild
            )
        }) {
            return Ok(None);
        }
        let slice = self.types.substs(substs).to_vec();
        let params = super::super::detached(self, parent, |infer| {
            let frame = infer.frame(parent, &slice)?;
            let (params, _) = infer.with_frame(frame, |infer| infer.sig_types(&sig))?;
            infer.settle_projections()?;
            Ok(params
                .into_iter()
                .map(|p| infer.zonk(p))
                .collect::<Vec<_>>())
        })?;
        let captures = self.types.tuple(&params);
        let word = self.types.prim(crate::names::Primitive::Usize);
        Ok(Some(self.types.tuple(&[word, word, captures])))
    }
}
