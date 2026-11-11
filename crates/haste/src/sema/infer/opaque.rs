use anyhow::{Result, anyhow};

use crate::body::sig::Signature;
use crate::body::{Args, Body, PathId};
use crate::names::{ItemRef, ModRef};

use super::super::Context;
use super::super::substs::{PackedMap, Substs};
use super::super::ty::{Kind, Ty};
use super::closure::Closure;
use super::scope::Frame;
use super::{BodyTables, Infer, Local, Typed};

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
    /// The associated type bindings its bounds spell, as `(trait, name, ty)`.
    pub bindings: Vec<(ItemRef, String, Ty)>,
    /// The declared output of an async fn, which `.await` gives before the future is revealed.
    pub output: Option<Ty>,
}

impl<'a> Infer<'_, 'a> {
    /// Ties the return value of `opaque` to the type that its body returns, once its type arguments are known.
    pub(super) fn opaque_ret(&mut self, opaque: Opaque) -> Result<()> {
        self.opaques.push(opaque);
        self.settle_opaques(false)
    }

    /// The trait bounds of the pending `impl Trait` return value `ty` and the supertrait bounds they imply, each with `Self` first.
    pub(super) fn opaque_bounds(&mut self, ty: Ty) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let bounds = self.opaques.iter().find(|opaque| self.resolve(opaque.ret) == ty).map(|opaque| opaque.bounds.clone()).unwrap_or_default();
        self.expand_super_bounds(bounds)
    }

    /// Expands `bounds` with all supertrait bounds they transitively imply, keeping `Self` first in each entry.
    pub(super) fn expand_super_bounds(&mut self, mut bounds: Vec<(ItemRef, Vec<Ty>)>) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
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
        let tied = self.trial(|infer| Ok(substs.iter().zip(types).all(|(&a, &b)| infer.unifies(a, b)).then_some(())))?;
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
            if self.poly.is_some() && self.any_has_params(&substs) {
                self.poly_fallback();
                return Ok(());
            }
            self.share_closures(&substs);
            let interned = self.cx.types.intern_substs(&substs);
            let found = revealed(self.cx, item, interned)?;
            if let Some(output) = output.filter(|_| !self.cx.has_closure(found)) {
                let output = self.zonk(output);
                self.cx.store.lookups.futures.insert(found, output);
            }
            self.unify(ret, found)?;
        }
        Ok(())
    }

    /// Registers the closures of this body once their types are known, when `substs` hold one, so the body that reveals an `impl Trait` return can call them before this body finishes.
    fn share_closures(&mut self, substs: &[Ty]) {
        let Some((item, args)) = self.instance else { return };
        if !substs.iter().any(|&ty| holds_closure(self.cx, ty, (item, args))) {
            return;
        }
        let locals: Vec<Local> = self.out.locals.clone().into_iter().map(|local| Local { ty: self.zonk(local.ty) }).collect();
        let mut closures = self.out.closures.clone();
        closures.values_mut().for_each(|closure| closure.ret = self.zonk(closure.ret));
        let typed = |closure: &Closure| {
            let held = closure.params.iter().copied().chain(closure.captures.iter().map(|capture| capture.local));
            held.map(|local| locals[local as usize].ty).chain([closure.ret]).collect::<Vec<Ty>>()
        };
        if closures.values().flat_map(typed).any(|ty| self.has_vars(ty)) {
            return;
        }
        let tables = std::sync::Arc::new(BodyTables { closures: PackedMap::pack(closures), ..BodyTables::default() });
        super::closure::register(self.cx, (item, args), &Typed { tables, locals, ..Typed::default() });
    }

    /// Holds the hidden type of each `impl Trait` in this function's return type to its bounds, tying the associated types they name, as `Output = T`, and their trait arguments, which rustc reads to infer the body.
    pub(super) fn bind_hidden(&mut self) -> Result<()> {
        if !self.hidden.is_empty() {
            self.poly_fallback();
        }
        let sig = self.sig;
        for (at, hidden) in std::mem::take(&mut self.hidden) {
            let bounds = sig.body.tys.iter().find_map(|ty| match *ty {
                crate::body::Ty::ImplTrait(token, bounds) if token == at => Some(bounds),
                _ => None,
            });
            for &bound in bounds.map(|bounds| sig.body.list(bounds)).unwrap_or_default() {
                let Some((of, path)) = self.bound_trait_at(&sig.body, bound, self.home) else {
                    continue;
                };
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

    /// The declared output of the pending async fn or `impl Future<Output = T>` return `ty`.
    pub(super) fn opaque_output(&self, ty: Ty) -> Option<Ty> {
        self.opaques
            .iter()
            .find(|opaque| self.resolve(opaque.ret) == ty)
            .and_then(|opaque| opaque.output.or_else(|| opaque.bindings.iter().find(|(of, name, _)| self.cx.lang(*of) == Some("future_trait") && name == "Output").map(|&(_, _, out)| out)))
    }

    /// The supertraits of `of` in vtable order, each paired with whether `of`'s parameters uniquely fix its type arguments.
    pub(in crate::sema) fn vtable_traits(&mut self, of: ItemRef) -> Result<Vec<(ItemRef, bool)>> {
        let traits = self.implied_traits(vec![of])?;
        let count = super::scope::type_params(&self.cx.trait_generics(of)?.1.params).count();
        let initial: Vec<_> = (0..=count).map(|at| self.cx.types.intern(Kind::Value(at as u128))).collect();
        let bounds = self.expand_super_bounds(vec![(of, initial)])?;
        Ok(traits
            .into_iter()
            .map(|owner| {
                let fixed = owner == of || {
                    let hits: Vec<_> = bounds.iter().filter(|(bound, _)| *bound == owner).collect();
                    hits.len() == 1 && !hits[0].1[1..].iter().any(|&ty| self.has_vars(ty))
                };
                (owner, fixed)
            })
            .collect())
    }

    /// The associated type `name` of trait `of` that a bound of the pending `impl Trait` return `ty` fixes.
    pub(super) fn opaque_binding(&mut self, of: ItemRef, ty: Ty, name: &str) -> Result<Option<Ty>> {
        let bindings = self.opaques.iter().find(|opaque| self.resolve(opaque.ret) == ty).map(|opaque| opaque.bindings.clone()).unwrap_or_default();
        for (bound_of, bound_name, out) in bindings {
            if bound_name == name && (bound_of == of || self.implied_traits(vec![bound_of])?.contains(&of)) {
                return Ok(Some(out));
            }
        }
        Ok(None)
    }

    /// The function pointer type of the `Fn` bound on the pending `impl Trait` return value `ty`.
    pub(super) fn opaque_sig(&self, ty: Ty) -> Option<Ty> {
        self.opaques.iter().find(|opaque| self.resolve(opaque.ret) == ty).and_then(|opaque| opaque.sig)
    }

    /// The function pointer type that the first `Fn` bound among `bounds` in `arena` spells, read through `frame`.
    pub(super) fn first_fn_bound(&mut self, frame: Frame<'a>, arena: &Body, bounds: Vec<u32>) -> Result<Option<Ty>> {
        let home = frame.home;
        self.with_frame(frame, |infer| {
            for bound in bounds {
                if let Some(hint) = infer.fn_bound(arena, home, bound)? {
                    return Ok(Some(hint));
                }
            }
            Ok(None)
        })
    }

    /// The function pointer type that an `Fn` bound on the outer parameter `name` spells in a header of the frame: the function's own, or its impl's or trait's.
    pub(super) fn header_fn_bound(&mut self, name: &str, home: ModRef) -> Result<Option<Ty>> {
        for (header, src) in self.headers()? {
            let (arena, generics, predicates) = header.parts();
            for bound in super::scope::named_bounds(arena, src, (generics, predicates), name) {
                if let Some(hint) = self.in_source(src, |infer| infer.fn_bound(arena, home, bound))? {
                    return Ok(Some(hint));
                }
            }
        }
        Ok(None)
    }

    /// The function pointer type that an `impl Fn(A) -> B` return type of the body spells, which a closure it returns takes its signature from.
    pub(super) fn opaque_fn_sig(&mut self) -> Result<Option<Ty>> {
        self.ret_fn_sig(self.sig)
    }

    /// The function pointer type that an `impl Fn(A) -> B` return type of `sig` spells in the current frame.
    pub(super) fn ret_fn_sig(&mut self, sig: &Signature) -> Result<Option<Ty>> {
        let Some(crate::body::Ty::ImplTrait(_, bounds)) = sig.ret.map(|ret| *sig.body.ty(ret)) else {
            return Ok(None);
        };
        for &bound in sig.body.list(bounds) {
            if let Some(found) = self.fn_bound(&sig.body, self.home, bound)? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// The trait bounds other than `Fn` that an `impl Trait` return type of `sig` spells in the current frame, each with `ret` as `Self`.
    pub(super) fn ret_bounds(&mut self, sig: &Signature, ret: Ty) -> Result<Vec<(ItemRef, Vec<Ty>)>> {
        let Some(crate::body::Ty::ImplTrait(_, bounds)) = sig.ret.map(|ret| *sig.body.ty(ret)) else {
            return Ok(Vec::new());
        };
        let mut found = Vec::new();
        for &bound in sig.body.list(bounds) {
            let Some((of, path)) = self.bound_trait_at(&sig.body, bound, self.home) else {
                continue;
            };
            if !self.is_callable_trait(of)
                && let Ok(args) = self.bound_args(of, &sig.body, path, ret)
            {
                found.push((of, std::iter::once(ret).chain(args).collect()));
            }
        }
        Ok(found)
    }

    /// The associated type bindings that an `impl Trait` return type of `sig` spells in the current frame.
    pub(super) fn ret_bindings(&mut self, sig: &Signature) -> Result<Vec<(ItemRef, String, Ty)>> {
        let Some(crate::body::Ty::ImplTrait(_, bounds)) = sig.ret.map(|ret| *sig.body.ty(ret)) else {
            return Ok(Vec::new());
        };
        let mut found = Vec::new();
        for &bound in sig.body.list(bounds) {
            let Some((of, path)) = self.bound_trait_at(&sig.body, bound, self.home) else {
                continue;
            };
            if !self.is_callable_trait(of) {
                for (name, ty) in self.path_bindings(&sig.body, path)? {
                    found.push((of, name, ty));
                }
            }
        }
        Ok(found)
    }

    pub(super) fn fn_bound(&mut self, arena: &Body, home: ModRef, bound: u32) -> Result<Option<Ty>> {
        let Some((of, path)) = self.bound_trait_at(arena, bound, home) else {
            return Ok(None);
        };
        if !self.is_callable_trait(of) {
            return Ok(None);
        }
        let Some(&last) = arena.list(arena.path(path).segments).last() else {
            return Ok(None);
        };
        let (inputs, output) = match arena.segments[last as usize].args {
            Some(Args::Paren { inputs, output }) => self.paren_sig(arena, inputs, output)?,
            Some(Args::Angle(_)) => match self.angle_sig(arena, (of, path))? {
                Some(sig) => sig,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        Ok(Some(self.cx.types.fn_ptr(&inputs, output)))
    }

    /// The inputs and output that the angle form `FnOnce<(A, B), Output = R>` of callable trait `of` names, once its argument tuple is known.
    fn angle_sig(&mut self, arena: &Body, (of, path): (ItemRef, PathId)) -> Result<Option<(Vec<Ty>, Ty)>> {
        let self_ty = self.fresh(Kind::Var);
        let Some(&tuple) = self.bound_args(of, arena, path, self_ty)?.first() else {
            return Ok(None);
        };
        let Kind::Tuple(inputs) = *self.cx.types.kind(self.resolve(tuple)) else { return Ok(None) };
        let inputs = self.cx.types.substs(inputs).to_vec();
        let output = self.path_bindings(arena, path)?.into_iter().find(|(name, _)| name == "Output").map(|(_, ty)| ty);
        Ok(Some((inputs, output.unwrap_or_else(|| self.fresh(Kind::Var)))))
    }
}

/// The return type of `item` at `substs`, inferring its body once per pass, which also records the closures it defines.
pub(super) fn revealed(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Ty> {
    let reveals = cx.store.lookups.reveals.clone();
    reveals.run(
        cx,
        (item, substs),
        |cx| super::infer(cx, item, substs).and_then(|typed| typed.ret.ok_or_else(|| anyhow!("`{}` has no return type", cx.name(item)))),
        |cx| async_future(cx, item, substs)?.ok_or_else(|| anyhow!("the `impl Trait` return type of `{}` depends on itself", cx.name(item))),
    )
}

/// The future of the async fn `item` at `substs`, which names its body's coroutine without inferring it, as a call inside that body's own inference needs; none for other functions.
fn async_future(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Option<Ty>> {
    if !cx.signature(item)?.is_async {
        return Ok(None);
    }
    let root = cx.body(item)?.root.ok_or_else(|| anyhow!("`{}` has no body", cx.name(item)))?;
    Ok(Some(cx.types.intern(Kind::Closure(item, substs, root.0))))
}

/// The function that a `#[define_opaque]` attribute in the same file names as the definer of the type alias `alias`, whose body holds `impl Trait`.
pub(super) fn opaque_definer(cx: &Context<'_>, alias: ItemRef) -> Option<ItemRef> {
    let (src, entry) = cx.item(alias);
    let name = crate::names::plain(src.text(entry.name?));
    let items = &cx.file(alias).items.items;
    let definers = items.iter().enumerate().filter(|(_, item)| item.kind == crate::outline::item::ItemKind::Fn).map(|(at, _)| ItemRef { item: at as u32, ..alias });
    definers.into_iter().find(|&candidate| cx.attr_at(candidate, "define_opaque").is_some_and(|at| src.text(at + 2) == name))
}

fn holds_closure(cx: &Context<'_>, ty: Ty, (item, args): (ItemRef, Substs)) -> bool {
    let kind = cx.types.kind(ty);
    matches!(kind, Kind::Closure(parent, substs, _) if *parent == item && *substs == args) || kind.children(&cx.types).any(|child| holds_closure(cx, child, (item, args)))
}
