use anyhow::{Result, anyhow, bail};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Ty};
use super::closure::Closure;
use super::{Infer, Local, Typed};

/// A return type that holds `impl Trait`: the function, its type arguments, the type that stands for the return value at the call, and the function pointer type of its `Fn` bound.
pub(super) type Opaque = (ItemRef, Vec<Ty>, Ty, Option<Ty>);

impl Infer<'_, '_> {
    /// Ties `ret`, the return type of `item` at `substs` that holds `impl Trait`, to the type that its body returns, once `substs` are known.
    pub(super) fn opaque_ret(&mut self, item: ItemRef, substs: &[Ty], (ret, sig): (Ty, Option<Ty>)) -> Result<()> {
        self.opaques.push((item, substs.to_vec(), ret, sig));
        self.settle_opaques(false)
    }

    /// Reveals each `impl Trait` return type whose type arguments are known, or every one when `force` is set.
    pub(super) fn settle_opaques(&mut self, force: bool) -> Result<()> {
        let mut at = 0;
        while at < self.opaques.len() {
            let (item, substs, ret, _) = self.opaques[at].clone();
            if !force && substs.iter().any(|ty| self.has_vars(*ty)) {
                at += 1;
                continue;
            }
            self.opaques.swap_remove(at);
            let substs: Vec<Ty> = substs.into_iter().map(|ty| self.zonk(ty)).collect();
            self.share_closures(&substs);
            let found = revealed(self.cx, item, &substs)?;
            self.unify(ret, found)?;
        }
        Ok(())
    }

    /// Registers the closures of this body once their types are known, when `substs` hold one, so the body that reveals an `impl Trait` return can call them before this body finishes.
    fn share_closures(&mut self, substs: &[Ty]) {
        let Some((item, args)) = self.instance.clone() else { return };
        if !substs.iter().any(|&ty| holds_closure(self.cx, ty, (item, &args))) {
            return;
        }
        let locals: Vec<Local> = self.out.locals.clone().into_iter().map(|local| Local { ty: self.substitute(local.ty), ..local }).collect();
        let mut closures = self.out.closures.clone();
        closures.values_mut().for_each(|closure| closure.ret = self.substitute(closure.ret));
        let typed = |closure: &Closure| {
            let held = closure.params.iter().copied().chain(closure.captures.iter().map(|capture| capture.local));
            held.map(|local| locals[local as usize].ty).chain([closure.ret]).collect::<Vec<Ty>>()
        };
        if closures.values().flat_map(typed).any(|ty| self.has_vars(ty)) {
            return;
        }
        super::closure::register(self.cx, (item, &args), &Typed { locals, closures, ..Typed::default() });
    }

    /// Ties the associated types that the bounds of each `impl Trait` in this function's return type name, as `Output = T`, to its hidden type, which rustc reads to infer the body.
    pub(super) fn bind_hidden(&mut self) -> Result<()> {
        let sig = self.sig;
        for (at, hidden) in std::mem::take(&mut self.hidden) {
            let bounds = sig.body.tys.iter().find_map(|ty| match *ty {
                crate::body::Ty::ImplTrait(token, bounds) if token == at => Some(bounds),
                _ => None,
            });
            for &bound in bounds.map(|bounds| sig.body.list(bounds)).unwrap_or_default() {
                let Some((of, path)) = self.bound_trait_at(&sig.body, bound, self.home) else { continue };
                if self.is_callable_trait(of) {
                    continue;
                }
                let args = self.bound_args(of, &sig.body, path, hidden)?;
                self.bindings_hold(&sig.body, path, (of, hidden, &args))?;
            }
        }
        Ok(())
    }

    /// The function pointer type of the `Fn` bound on the pending `impl Trait` return value `ty`.
    pub(super) fn opaque_sig(&self, ty: Ty) -> Option<Ty> {
        self.opaques.iter().find(|(_, _, ret, _)| self.resolve(*ret) == ty).and_then(|opaque| opaque.3)
    }
}

/// The return type of `item` at `substs`, inferring its body once per pass, which also records the closures it defines.
pub(super) fn revealed(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Ty> {
    let key = (item, substs.to_vec());
    match cx.reveals.get(&key) {
        Some(Some(ty)) => return Ok(*ty),
        Some(None) => bail!("the `impl Trait` return type of `{}` depends on itself", cx.name(item)),
        None => {}
    }
    cx.reveals.insert(key.clone(), None);
    let ret = super::infer(cx, item, substs).and_then(|typed| typed.ret.ok_or_else(|| anyhow!("`{}` has no return type", cx.name(item))));
    match &ret {
        Ok(ty) => cx.reveals.insert(key, Some(*ty)),
        Err(_) => cx.reveals.remove(&key),
    };
    ret
}

fn holds_closure(cx: &Context<'_>, ty: Ty, (item, args): (ItemRef, &[Ty])) -> bool {
    let kind = cx.types.kind(ty);
    matches!(kind, Kind::Closure(parent, substs, _) if *parent == item && substs.as_slice() == args) || kind.children().any(|child| holds_closure(cx, child, (item, args)))
}
