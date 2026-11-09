use anyhow::{Result, anyhow};

use crate::names::ItemRef;

use super::super::Context;
use super::super::ty::{Kind, Ty};
use super::closure::Closure;
use super::{Infer, Local, Typed};

/// A return type that holds `impl Trait`, until its type arguments are known.
#[derive(Clone)]
pub(super) struct Opaque {
    pub item: ItemRef,
    pub substs: Vec<Ty>,
    /// The type that stands for the return value at the call.
    pub ret: Ty,
    /// The function pointer type of its `Fn` bound.
    pub sig: Option<Ty>,
    /// Its other trait bounds, each with `Self` first, which method calls on it read before it is revealed.
    pub bounds: Vec<(ItemRef, Vec<Ty>)>,
    /// The declared output of an async fn, which `.await` gives before the future is revealed.
    pub output: Option<Ty>,
}

impl Infer<'_, '_> {
    /// Ties the return value of `opaque` to the type that its body returns, once its type arguments are known.
    pub(super) fn opaque_ret(&mut self, opaque: Opaque) -> Result<()> {
        self.opaques.push(opaque);
        self.settle_opaques(false)
    }

    /// The trait bounds of the pending `impl Trait` return value `ty` and the supertrait bounds they imply, each with `Self` first.
    pub(super) fn opaque_bounds(&mut self, ty: Ty) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let mut bounds = self.opaques.iter().find(|opaque| self.resolve(opaque.ret) == ty).map(|opaque| opaque.bounds.clone()).unwrap_or_default();
        let mut at = 0;
        while at < bounds.len() {
            let (of, substs) = bounds[at].clone();
            for implied in self.super_bounds(of, &substs)? {
                if !bounds.contains(&implied) {
                    bounds.push(implied);
                }
            }
            at += 1;
        }
        Ok(bounds)
    }

    /// Ties `substs` of trait `of`, whose `Self` is a pending `impl Trait` value, to the one bound of it of that trait: whether they unify, or none when it has no one such bound.
    pub(super) fn opaque_holds(&mut self, of: ItemRef, substs: &[Ty]) -> Result<Option<bool>> {
        let held: Vec<Vec<Ty>> = self.opaque_bounds(self.resolve(substs[0]))?.into_iter().filter(|(bound, _)| *bound == of).map(|(_, types)| types).collect();
        let [types] = &held[..] else { return Ok(None) };
        let tied = self.trial(|infer| Ok(substs.iter().zip(types).all(|(&a, &b)| infer.unify(a, b).is_ok()).then_some(())))?;
        Ok(Some(tied.is_some()))
    }

    /// Reveals each `impl Trait` return type whose type arguments are known, or every one when `force` is set.
    pub(super) fn settle_opaques(&mut self, force: bool) -> Result<()> {
        let mut at = 0;
        while at < self.opaques.len() {
            let Opaque { item, substs, ret, output, .. } = self.opaques[at].clone();
            if !force && substs.iter().any(|ty| self.has_vars(*ty)) {
                at += 1;
                continue;
            }
            self.opaques.swap_remove(at);
            let substs: Vec<Ty> = substs.into_iter().map(|ty| self.zonk(ty)).collect();
            self.share_closures(&substs);
            let found = revealed(self.cx, item, &substs)?;
            if let Some(output) = output.filter(|_| self.cx.closure(found).is_none()) {
                let output = self.zonk(output);
                self.cx.futures.insert(found, output);
            }
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

    /// Holds the hidden type of each `impl Trait` in this function's return type to its bounds, tying the associated types they name, as `Output = T`, and their trait arguments, which rustc reads to infer the body.
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
                self.obligations.push((of, std::iter::once(hidden).chain(args).collect()));
            }
        }
        Ok(())
    }

    /// The declared output of the pending async fn future `ty`.
    pub(super) fn opaque_output(&self, ty: Ty) -> Option<Ty> {
        self.opaques.iter().find(|opaque| self.resolve(opaque.ret) == ty).and_then(|opaque| opaque.output)
    }

    /// The function pointer type of the `Fn` bound on the pending `impl Trait` return value `ty`.
    pub(super) fn opaque_sig(&self, ty: Ty) -> Option<Ty> {
        self.opaques.iter().find(|opaque| self.resolve(opaque.ret) == ty).and_then(|opaque| opaque.sig)
    }
}

/// The return type of `item` at `substs`, inferring its body once per pass, which also records the closures it defines.
pub(super) fn revealed(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Ty> {
    let key = (item, substs.to_vec());
    match cx.reveals.get(&key) {
        Some(Some(ty)) => return Ok(*ty),
        Some(None) => return async_future(cx, item, substs)?.ok_or_else(|| anyhow!("the `impl Trait` return type of `{}` depends on itself", cx.name(item))),
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

/// The future of the async fn `item` at `substs`, which names its body's coroutine without inferring it, as a call inside that body's own inference needs; none for other functions.
fn async_future(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Option<Ty>> {
    if !cx.signature(item)?.is_async {
        return Ok(None);
    }
    let root = cx.body(item)?.root.ok_or_else(|| anyhow!("`{}` has no body", cx.name(item)))?;
    Ok(Some(cx.types.intern(Kind::Closure(item, substs.to_vec(), root.0))))
}

/// The function that a `#[define_opaque]` attribute in the same file names as the definer of the type alias `alias`, whose body holds `impl Trait`.
pub(super) fn opaque_definer(cx: &Context<'_>, alias: ItemRef) -> Option<ItemRef> {
    let (src, entry) = cx.item(alias);
    let name = crate::names::plain(src.text(entry.name?));
    let items = &cx.file(alias).items.items;
    let definers = items.iter().enumerate().filter(|(_, item)| item.kind == crate::outline::item::ItemKind::Fn).map(|(at, _)| ItemRef { item: at as u32, ..alias });
    definers.into_iter().find(|&candidate| cx.attr_at(candidate, "define_opaque").is_some_and(|at| src.text(at + 2) == name))
}

fn holds_closure(cx: &Context<'_>, ty: Ty, (item, args): (ItemRef, &[Ty])) -> bool {
    let kind = cx.types.kind(ty);
    matches!(kind, Kind::Closure(parent, substs, _) if *parent == item && substs.as_slice() == args) || kind.children().any(|child| holds_closure(cx, child, (item, args)))
}
