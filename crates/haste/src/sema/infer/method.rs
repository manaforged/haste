use anyhow::{Result, bail};

use crate::body::sig::SelfKind;
use crate::body::{ExprId, List};
use crate::names::{ItemRef, plain};

use super::super::impls::{ImplRef, head};
use super::super::ty::{Kind, Mut, Ty};
use super::scope::{Frame, named_args, type_params};
use super::{Infer, Res};

/// How a method call adjusts its receiver: through how many references it reads, then whether it borrows the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adjust {
    pub derefs: u32,
    pub autoref: Option<Mut>,
}

/// A trait method a call may pick: its trait, its declaration in the trait, how it takes `self`, and the type of the parameter whose bound names the trait, which is then its `Self`.
#[derive(Clone, Copy)]
struct TraitCandidate {
    of: ItemRef,
    item: ItemRef,
    kind: SelfKind,
    bound: Option<Ty>,
}

/// The method a call picked: its item, the arguments of its impl or trait, the arguments of its own type parameters when its receiver type named them, and how it borrows the receiver.
struct Pick {
    item: ItemRef,
    args: Vec<Ty>,
    own: Vec<Ty>,
    autoref: Option<Mut>,
}

/// One method a call may pick: its impl, its item and how it takes `self`.
#[derive(Clone, Copy)]
struct Candidate {
    block: ImplRef,
    item: ItemRef,
    kind: SelfKind,
}

impl<'a> Infer<'_, 'a> {
    /// Infers the method call `expr`: picks the method by the receiver's type through references and borrows, as rustc probes.
    pub fn method_call(&mut self, expr: ExprId, receiver: ExprId, name: u32, params: (Option<List>, List), expected: Option<Ty>) -> Result<Ty> {
        let found = self.expr(receiver, None)?;
        let name = plain(self.src.text(name));
        let mut steps = vec![self.known_ty(found)?];
        while let Some(next) = self.next_step(expr, steps.len() as u32 - 1, steps[steps.len() - 1])? {
            if steps.len() == crate::RECURSION_LIMIT {
                bail!("reached the recursion limit while dereferencing {}", self.show(found));
            }
            steps.push(self.known_ty(next)?);
        }
        if let Some(pick) = self.opaque_pick(steps[0], name)? {
            return self.picked(expr, (pick, 0), params, expected);
        }
        match self.probe_steps((expr, found), name, &steps)? {
            Some((pick, derefs)) => self.picked(expr, (pick, derefs), params, expected),
            None => Err(self.no_method(steps[steps.len() - 1], found, name)),
        }
    }

    /// The method `name` that the first of `steps` to have one picks, and how many references it reads through.
    fn probe_steps(&mut self, (expr, found): (ExprId, Ty), name: &str, steps: &[Ty]) -> Result<Option<(Pick, u32)>> {
        let traits = self.trait_candidates(name, steps)?;
        let mut candidates = Vec::new();
        for &step in steps {
            self.candidates(step, name, &mut candidates)?;
        }
        for (derefs, &step) in steps.iter().enumerate() {
            if let Some(pick) = self.probe_step(step, (&candidates, &traits))? {
                if pick.autoref == Some(Mut::Mut) {
                    self.steps_mutably(expr, found, derefs as u32)?;
                }
                return Ok(Some((pick, derefs as u32)));
            }
        }
        Ok(None)
    }

    fn no_method(&self, last: Ty, found: Ty, name: &str) -> anyhow::Error {
        match self.cx.types.kind(last) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => anyhow::anyhow!("the type of a method receiver must be known here"),
            _ => anyhow::anyhow!("no method `{name}` on {}", self.show(found)),
        }
    }

    /// The trait method `name` that a bound of the pending `impl Trait` value `ty`, or a supertrait of one, declares, as rustc picks it by value before the hidden type is known.
    fn opaque_pick(&mut self, ty: Ty, name: &str) -> Result<Option<Pick>> {
        for (of, substs) in self.opaque_bounds(ty)? {
            if let Some(item) = self.cx.trait_member(of, name)
                && let Some(autoref) = borrow_of(self.cx.signature(item)?.receiver.map(|(kind, _)| kind))
            {
                self.obligations.push((of, substs.clone()));
                return Ok(Some(Pick { item, args: substs, own: Vec::new(), autoref }));
            }
        }
        Ok(None)
    }

    fn probe_step(&mut self, step: Ty, (candidates, traits): (&[Candidate], &[TraitCandidate])) -> Result<Option<Pick>> {
        let (bounded, unbounded): (Vec<TraitCandidate>, Vec<TraitCandidate>) = traits.iter().partition(|candidate| candidate.bound.is_some());
        let step_ty = self.substitute(step);
        let bounded_at: Vec<TraitCandidate> = bounded.iter().map(|candidate| TraitCandidate { bound: candidate.bound.map(|bound| self.substitute(bound)), ..*candidate }).collect();
        let unbounded: Vec<TraitCandidate> = unbounded.into_iter().filter(|candidate| !shadowed(&bounded_at, candidate.of, step_ty)).collect();
        for autoref in [None, Some(Mut::Not), Some(Mut::Mut)] {
            let adjusted = match autoref {
                None => step,
                Some(mutability) => self.cx.types.intern(Kind::Ref(mutability, step)),
            };
            if let Some(pick) = self.probe_traits(&bounded, (adjusted, autoref))? {
                return Ok(Some(pick));
            }
            for &candidate in candidates {
                if let Some((args, own)) = self.try_receiver(candidate, adjusted)? {
                    return Ok(Some(Pick { item: candidate.item, args, own, autoref }));
                }
            }
            if let Some(pick) = self.probe_traits(&unbounded, (adjusted, autoref))? {
                return Ok(Some(pick));
            }
        }
        Ok(None)
    }

    fn probe_traits(&mut self, traits: &[TraitCandidate], (adjusted, autoref): (Ty, Option<Mut>)) -> Result<Option<Pick>> {
        for &candidate in traits {
            if let Some((item, args)) = self.try_trait(candidate, adjusted)? {
                return Ok(Some(Pick { item, args, own: Vec::new(), autoref }));
            }
        }
        Ok(None)
    }

    fn next_step(&mut self, expr: ExprId, derefs: u32, step: Ty) -> Result<Option<Ty>> {
        let next = match self.cx.types.kind(step).clone() {
            Kind::Array(element, _) => return Ok(Some(self.cx.types.intern(Kind::Slice(element)))),
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Ptr(..) | Kind::Slice(_) => None,
            _ => match self.builtin_deref(step) {
                Some(inner) => Some(inner),
                None => self.deref_step(expr, derefs, step)?,
            },
        };
        Ok(next.map(|next| self.resolve(next)))
    }

    fn candidates(&mut self, step: Ty, name: &str, found: &mut Vec<Candidate>) -> Result<()> {
        let Some(head) = head(self.cx.types.kind(step)) else { return Ok(()) };
        for block in self.cx.inherent(head) {
            if found.iter().any(|candidate| candidate.block == block) {
                continue;
            }
            let Some(item) = self.cx.member(block, name) else { continue };
            if let Some((kind, _)) = self.cx.signature(item)?.receiver {
                found.push(Candidate { block, item, kind });
            }
        }
        Ok(())
    }

    fn trait_candidates(&mut self, name: &str, steps: &[Ty]) -> Result<Vec<TraitCandidate>> {
        let mut found = Vec::new();
        let mut traits = self.cx.traits_in_scope(self.home);
        let principals: Vec<ItemRef> = steps.iter().filter_map(|&step| match self.cx.types.kind(step) {
            Kind::Dyn(of, ..) => Some(*of),
            _ => None,
        }).collect();
        let bounded = self.bound_candidates(name, steps)?;
        let current = self.current_trait();
        for local in self.body_traits().into_iter().chain(principals).chain(current) {
            if !traits.contains(&local) {
                traits.push(local);
            }
        }
        let scoped = traits.len();
        traits.extend(bounded.iter().map(|&(of, _)| of).filter(|of| !traits[..scoped].contains(of)).collect::<Vec<_>>());
        for (at, of) in traits.into_iter().enumerate() {
            let Some(item) = self.cx.trait_member(of, name) else { continue };
            let Some((kind, _)) = self.cx.signature(item)?.receiver else { continue };
            if let Some(&(_, param)) = bounded.iter().find(|&&(bounded, _)| bounded == of) {
                found.push(TraitCandidate { of, item, kind, bound: Some(param) });
            }
            if at < scoped {
                found.push(TraitCandidate { of, item, kind, bound: None });
            }
        }
        Ok(found)
    }

    pub(super) fn current_trait(&mut self) -> Option<ItemRef> {
        let owner = self.instance.as_ref().and_then(|(item, _)| self.cx.owner_trait(*item));
        self.block.and_then(|block| self.cx.impl_trait(block)).or(owner)
    }

    fn bound_candidates(&mut self, name: &str, steps: &[Ty]) -> Result<Vec<(ItemRef, Ty)>> {
        let steps: Vec<Ty> = steps.iter().map(|&step| self.substitute(step)).collect();
        let params: Vec<(&str, Ty)> = self.generics.clone().into_iter().filter(|&(_, ty)| steps.contains(&self.substitute(ty))).collect();
        let mut found = Vec::new();
        for param in params {
            if let Some((of, _)) = self.bound_trait(param, name, Self::reaches_member)? {
                let ty = self.substitute(param.1);
                found.extend(self.implied_traits(vec![of])?.into_iter().map(|of| (of, ty)));
            }
        }
        let mut bounds = self.type_bounds()?;
        bounds.extend(self.param_alias_bounds()?);
        for (of, substs) in bounds {
            let ty = self.substitute(substs[0]);
            if steps.contains(&ty) && self.reaches_member(of, name)? {
                found.extend(self.implied_traits(vec![of])?.into_iter().map(|of| (of, ty)));
            }
        }
        let owner = self.instance.as_ref().and_then(|(item, _)| self.cx.owner_trait(*item)).filter(|_| self.block.is_none());
        if let (Some(of), Some(ty)) = (owner, self.self_ty.map(|ty| self.substitute(ty)))
            && steps.contains(&ty)
        {
            found.extend(self.implied_traits(vec![of])?.into_iter().map(|of| (of, ty)));
        }
        Ok(found)
    }

    /// The function a call of trait method `method` with receiver type `receiver` runs, and its leading type arguments.
    pub fn trait_method(&mut self, method: ItemRef, receiver: Ty) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        self.method_of_trait(method, receiver, false)
    }

    /// Like `trait_method` for a call that desugaring writes, such as `Iterator::next` in a `for` loop, which takes the trait item itself with an obligation while the receiver's `Self` is still unknown.
    pub fn lang_method(&mut self, method: ItemRef, receiver: Ty) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        self.method_of_trait(method, receiver, true)
    }

    fn method_of_trait(&mut self, method: ItemRef, receiver: Ty, pending: bool) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some(of) = self.cx.owner_trait(method) else { bail!("{} is not a trait method", self.cx.name(method)) };
        let Some((kind, _)) = self.cx.signature(method)?.receiver else { bail!("{} takes no `self`", self.cx.name(method)) };
        let candidate = TraitCandidate { of, item: method, kind, bound: None };
        let found = match self.try_trait(candidate, receiver)? {
            None if pending => self.pending_trait(candidate, receiver)?,
            found => found,
        };
        let Some((item, mut substs)) = found else { return Ok(None) };
        let own = type_params(&self.cx.signature(item)?.generics).count();
        substs.extend((0..own).map(|_| self.fresh(Kind::Var)));
        Ok(Some((item, substs)))
    }

    fn try_trait(&mut self, candidate: TraitCandidate, adjusted: Ty) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let self_ty = match candidate.kind {
            SelfKind::Typed(ty) => self.typed_self(candidate, ty, adjusted)?,
            _ => self.trait_self(candidate, adjusted),
        };
        let Some(self_ty) = self_ty else { return Ok(None) };
        if candidate.bound.is_some_and(|bound| self.substitute(bound) != self.substitute(self_ty)) {
            return Ok(None);
        }
        let args = self.fresh_trait_args(candidate.of)?;
        if candidate.bound.is_some() || !args.is_empty() {
            self.assume(candidate.of, &std::iter::once(self_ty).chain(args.iter().copied()).collect::<Vec<_>>())?;
        }
        if self.compiler_impl(candidate.of, self_ty) || self.object_holds(candidate.of, self_ty, &args) == Some(true) {
            return Ok(Some((candidate.item, std::iter::once(self_ty).chain(args).collect())));
        }
        let Some(selected) = self.trial(|infer| infer.select(candidate.of, self_ty, &args))? else { return self.unselected_trait(candidate, self_ty, args) };
        Ok(Some(match self.impl_member(candidate.of, (self_ty, &args), selected, self.cx.name(candidate.item))? {
            Some((item, (_, impl_args))) => (item, impl_args),
            None => (candidate.item, std::iter::once(self_ty).chain(args).collect()),
        }))
    }

    fn pending_trait(&mut self, candidate: TraitCandidate, receiver: Ty) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some(self_ty) = self.trait_self(candidate, receiver) else { return Ok(None) };
        if !matches!(self.cx.types.kind(self.resolve(self_ty)), Kind::Var(_)) {
            return Ok(None);
        }
        let substs: Vec<Ty> = std::iter::once(self_ty).chain(self.fresh_trait_args(candidate.of)?).collect();
        self.obligations.push((candidate.of, substs.clone()));
        Ok(Some((candidate.item, substs)))
    }

    /// The trait item itself when no one impl of its trait is selected for `self_ty` yet but one may apply, with the obligation to prove later.
    fn unselected_trait(&mut self, candidate: TraitCandidate, self_ty: Ty, args: Vec<Ty>) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        if !self.may_implement(candidate.of, self_ty, &args)? {
            return Ok(None);
        }
        let substs: Vec<Ty> = std::iter::once(self_ty).chain(args).collect();
        self.obligations.push((candidate.of, substs.clone()));
        Ok(Some((candidate.item, substs)))
    }

    /// The `Self` that makes the declared receiver type `ty` of a trait method, such as `Pin<&mut Self>`, equal `adjusted`.
    fn typed_self(&mut self, candidate: TraitCandidate, ty: crate::body::TyId, adjusted: Ty) -> Result<Option<Ty>> {
        let sig = self.cx.signature(candidate.item)?;
        let self_ty = self.fresh(Kind::Var);
        let substs: Vec<Ty> = std::iter::once(self_ty).chain(self.fresh_trait_args(candidate.of)?).collect();
        let frame = self.frame(candidate.item, &substs)?;
        let declared = self.with_frame(frame, |infer| infer.lower_ty(&sig.body, ty))?;
        let found = self.trial(|infer| Ok(infer.unifies(declared, adjusted).then(|| infer.resolve(self_ty))))?;
        Ok(found.filter(|found| !matches!(self.cx.types.kind(*found), Kind::Var(_)) && candidate.bound.is_none_or(|param| param == self.substitute(*found))))
    }

    fn trait_self(&mut self, candidate: TraitCandidate, adjusted: Ty) -> Option<Ty> {
        let self_ty = match (candidate.kind, self.cx.types.kind(adjusted).clone()) {
            (SelfKind::Value, Kind::Slice(_) | Kind::Prim(crate::names::Primitive::Str)) => return None,
            (SelfKind::Value, _) => adjusted,
            (SelfKind::Ref, Kind::Ref(Mut::Not, inner)) | (SelfKind::RefMut, Kind::Ref(Mut::Mut, inner)) => inner,
            _ => return None,
        };
        candidate.bound.is_none_or(|param| param == self.substitute(self_ty)).then_some(self_ty)
    }

    fn try_receiver(&mut self, candidate: Candidate, adjusted: Ty) -> Result<Option<(Vec<Ty>, Vec<Ty>)>> {
        self.trial(|infer| Ok(infer.receiver_matches(candidate, adjusted)))
    }

    fn receiver_matches(&mut self, candidate: Candidate, adjusted: Ty) -> Option<(Vec<Ty>, Vec<Ty>)> {
        let count = self.impl_params(candidate.block).ok()?;
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let (mut generics, self_ty) = self.impl_frame(candidate.block, &args).ok()?;
        let sig = self.cx.signature(candidate.item).ok()?;
        let src = &self.cx.file(candidate.item).source;
        let own = match candidate.kind {
            SelfKind::Typed(_) => self.own_args(candidate.item, Vec::new()).ok()?,
            _ => Vec::new(),
        };
        generics.extend(named_args(src, &sig.generics, &own));
        let frame = Frame { src, home: self.cx.module_of(candidate.item).ok()?, generics, self_ty: Some(self_ty), block: Some(candidate.block), item: None, root: candidate.block.root() };
        let receiver = self.with_frame(frame, |infer| infer.receiver_ty(&sig.body, candidate.kind)).ok()?;
        self.unifies(receiver, adjusted).then_some((args, own))
    }

    fn picked(&mut self, expr: ExprId, (pick, derefs): (Pick, u32), (explicit, params): (Option<List>, List), expected: Option<Ty>) -> Result<Ty> {
        let Pick { item, args: mut substs, own, autoref } = pick;
        let adjust = Adjust { derefs, autoref };
        let body = self.body;
        let given = match explicit {
            Some(list) => self.arg_list(body, list)?,
            None => Vec::new(),
        };
        let given = self.own_args(item, given)?;
        for (&from, &to) in given.iter().zip(&own) {
            self.unify(from, to)?;
        }
        substs.extend(given);
        self.bind_bounds(item, &substs)?;
        self.out.res.insert(expr, Res::Fn(item, substs.clone()));
        self.out.adjusts.insert(expr, adjust);
        if let (Some(Mut::Mut), crate::body::Expr::Method { receiver, .. }) = (adjust.autoref, *self.body.expr(expr)) {
            self.note_mut(receiver)?;
        }
        let (types, ret) = self.fn_types(item, &substs)?;
        let given = self.body.list(params).to_vec();
        if given.len() + 1 != types.len() {
            bail!("{} takes {} arguments, {} given", self.cx.name(item), types.len() - 1, given.len());
        }
        let hints = self.expected_inputs(&types[1..], ret, expected);
        for ((arg, param), hint) in given.into_iter().zip(types.into_iter().skip(1)).zip(hints) {
            self.call_arg(Some((item, &substs[..])), ExprId(arg), (param, hint))?;
        }
        Ok(ret)
    }
}

/// Whether a where-clause bound of `of` on the type parameter that `step` instantiates hides the trait's impls, as rustc sees only the bound on a parameter.
fn shadowed(bounded: &[TraitCandidate], of: ItemRef, step: Ty) -> bool {
    bounded.iter().any(|candidate| candidate.of == of && candidate.bound == Some(step))
}

/// How a call borrows its receiver for a method that takes `self` as `kind`, or none for a typed `self`.
fn borrow_of(kind: Option<SelfKind>) -> Option<Option<Mut>> {
    match kind? {
        SelfKind::Value => Some(None),
        SelfKind::Ref => Some(Some(Mut::Not)),
        SelfKind::RefMut => Some(Some(Mut::Mut)),
        SelfKind::Typed(_) => None,
    }
}
