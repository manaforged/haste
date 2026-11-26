use anyhow::{Result, bail};

use crate::body::TyId;
use crate::body::sig::SelfKind;
use crate::names::{ItemRef, Primitive};
use crate::sema::impls::head;
use crate::sema::ty::{Kind, Mut, Ty};

use super::super::Infer;
use super::super::scope::{Frame, named_args, type_params};
use super::super::select::Picked;
use super::{Candidate, TraitCandidate};

impl Infer<'_, '_> {
    pub(super) fn candidates(
        &mut self,
        step: Ty,
        name: &str,
        found: &mut Vec<Candidate>,
    ) -> Result<()> {
        let Some(head) = head(self.cx.types.kind(step)) else {
            return Ok(());
        };
        for block in self.cx.inherent(head) {
            if found.iter().any(|candidate| candidate.block == block) {
                continue;
            }
            let Some(item) = self.cx.member(block, name) else {
                continue;
            };
            if !self.cx.member_visible(item, self.home)? {
                continue;
            }
            if let Some((kind, _)) = self.cx.signature(item)?.receiver {
                found.push(Candidate { block, item, kind });
            }
        }
        Ok(())
    }

    pub(super) fn trait_candidates(
        &mut self,
        name: &str,
        steps: &[Ty],
    ) -> Result<Vec<TraitCandidate>> {
        let mut found = Vec::new();
        let in_scope = self.cx.traits_in_scope(self.home);
        let bounded = self.bound_candidates(name, steps)?;
        let mut extra_scoped = Vec::new();
        let current = self.current_trait();
        for local in self.body_traits().into_iter().chain(current) {
            if !in_scope.contains(&local) && !extra_scoped.contains(&local) {
                extra_scoped.push(local);
            }
        }
        let mut dyn_scoped: Vec<(ItemRef, Ty)> = Vec::new();
        for &step in steps {
            if let Kind::Dyn(of, ..) = *self.cx.types.kind(step) {
                for implied in self.implied_traits(vec![of])? {
                    if !in_scope.contains(&implied)
                        && !extra_scoped.contains(&implied)
                        && !dyn_scoped.contains(&(implied, step))
                    {
                        dyn_scoped.push((implied, step));
                    }
                }
            }
        }
        let mut extra_bounded = Vec::new();
        for &(of, _) in &bounded {
            if !in_scope.contains(&of)
                && !extra_scoped.contains(&of)
                && !extra_bounded.contains(&of)
            {
                extra_bounded.push(of);
            }
        }
        let scoped_len = in_scope.len() + extra_scoped.len();
        for (at, of) in in_scope
            .iter()
            .copied()
            .chain(extra_scoped)
            .chain(extra_bounded)
            .enumerate()
        {
            let Some(item) = self.cx.trait_member(of, name) else {
                continue;
            };
            let Some((kind, _)) = self.cx.signature(item)?.receiver else {
                continue;
            };
            if let Some(&(_, param)) = bounded.iter().find(|&&(bounded, _)| bounded == of) {
                found.push(TraitCandidate {
                    of,
                    item,
                    kind,
                    bound: Some(param),
                    dyn_step: None,
                });
            }
            if at < scoped_len {
                found.push(TraitCandidate {
                    of,
                    item,
                    kind,
                    bound: None,
                    dyn_step: None,
                });
            }
        }
        for (of, step) in dyn_scoped {
            let Some(item) = self.cx.trait_member(of, name) else {
                continue;
            };
            let Some((kind, _)) = self.cx.signature(item)?.receiver else {
                continue;
            };
            found.push(TraitCandidate {
                of,
                item,
                kind,
                bound: None,
                dyn_step: Some(step),
            });
        }
        Ok(found)
    }

    pub(super) fn keep_fitting_bounds(
        &mut self,
        found: &mut Vec<TraitCandidate>,
        count: usize,
    ) -> Result<()> {
        let mut fits = Vec::with_capacity(found.len());
        for candidate in found.iter() {
            fits.push(self.cx.signature(candidate.item)?.params.len() == count);
        }
        let fitting: Vec<Option<Ty>> = found
            .iter()
            .zip(&fits)
            .filter(|(candidate, fit)| candidate.bound.is_some() && **fit)
            .map(|(candidate, _)| candidate.bound)
            .collect();
        let mut kept = fits.iter().map(|&fit| fit);
        found.retain(|candidate| {
            kept.next().unwrap_or(true)
                || candidate.bound.is_none()
                || !fitting.contains(&candidate.bound)
        });
        Ok(())
    }

    pub(in super::super) fn current_trait(&mut self) -> Option<ItemRef> {
        let owner = self
            .instance
            .as_ref()
            .and_then(|(item, _)| self.cx.owner_trait(*item));
        self.block
            .and_then(|block| self.cx.impl_trait(block))
            .or(owner)
    }

    fn bound_candidates(&mut self, name: &str, steps: &[Ty]) -> Result<Vec<(ItemRef, Ty)>> {
        let mut sub_buf = [Ty::NONE; 8];
        let sub_vec;
        let steps: &[Ty] = if steps.len() <= 8 {
            for (dst, &step) in sub_buf.iter_mut().zip(steps) {
                *dst = self.substitute(step);
            }
            &sub_buf[..steps.len()]
        } else {
            sub_vec = steps
                .iter()
                .map(|&step| self.substitute(step))
                .collect::<Vec<_>>();
            &sub_vec
        };
        let mut found = Vec::new();
        for idx in 0..self.generics.len() {
            let (param_name, param_ty) = self.generics[idx];
            let sub_ty = self.substitute(param_ty);
            if steps.contains(&sub_ty)
                && let Some((of, _)) =
                    self.bound_trait((param_name, param_ty), name, Self::reaches_member)?
            {
                let ty = self.substitute(param_ty);
                found.extend(
                    self.implied_traits(vec![of])?
                        .into_iter()
                        .map(|of| (of, ty)),
                );
            }
        }
        for (of, substs) in self.assumptions()? {
            let ty = self.substitute(substs[0]);
            if steps.contains(&ty) && self.reaches_member(of, name)? {
                for implied in self.implied_traits(vec![of])? {
                    if !found.contains(&(implied, ty)) {
                        found.push((implied, ty));
                    }
                }
            }
        }
        Ok(found)
    }

    /// The function a call of trait method `method` with receiver type `receiver` runs, and its leading type arguments.
    pub fn trait_method(
        &mut self,
        method: ItemRef,
        receiver: Ty,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        self.method_of_trait(method, receiver, false)
    }

    /// Like `trait_method` for a call that desugaring writes, such as `Iterator::next` in a `for` loop, which takes the trait item itself with an obligation while the receiver's `Self` is still unknown.
    pub fn lang_method(
        &mut self,
        method: ItemRef,
        receiver: Ty,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        self.method_of_trait(method, receiver, true)
    }

    fn method_of_trait(
        &mut self,
        method: ItemRef,
        receiver: Ty,
        pending: bool,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some(of) = self.cx.owner_trait(method) else {
            bail!("{} is not a trait method", self.cx.name(method));
        };
        let Some((kind, _)) = self.cx.signature(method)?.receiver else {
            bail!("{} takes no `self`", self.cx.name(method));
        };
        let candidate = TraitCandidate {
            of,
            item: method,
            kind,
            bound: None,
            dyn_step: None,
        };
        let found = match self.try_trait(candidate, receiver, true)? {
            None if pending => self.pending_trait(candidate, receiver)?,
            found => found,
        };
        let Some((item, mut substs)) = found else {
            return Ok(None);
        };
        let own = type_params(&self.cx.signature(item)?.generics).count();
        substs.extend((0..own).map(|_| self.fresh(Kind::Var)));
        Ok(Some((item, substs)))
    }

    pub(super) fn try_trait(
        &mut self,
        candidate: TraitCandidate,
        adjusted: Ty,
        use_bounds: bool,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some(self_ty) = self.candidate_self(candidate, adjusted)? else {
            return Ok(None);
        };
        if self.unsized_for_method(candidate.item, self_ty)? {
            return Ok(None);
        }
        let args = self.fresh_trait_args(candidate.of)?;
        if use_bounds && (candidate.bound.is_some() || !args.is_empty()) {
            self.assume(
                candidate.of,
                &std::iter::once(self_ty)
                    .chain(args.iter().copied())
                    .collect::<Vec<_>>(),
            )?;
        }
        if self.compiler_impl(candidate.of, self_ty)
            || self.object_holds(candidate.of, self_ty, &args) == Some(true)
            || self.check_callable_obligation(candidate.of, self_ty, &args)?
        {
            return Ok(Some((
                candidate.item,
                std::iter::once(self_ty).chain(args).collect(),
            )));
        }
        if candidate.kind == SelfKind::Value && matches!(self.cx.types.kind(self_ty), Kind::Dyn(..))
        {
            return Ok(None);
        }
        let Some(selected) = self.trial(|infer| infer.select(candidate.of, self_ty, &args))? else {
            return self.unselected_trait(candidate, self_ty, args);
        };
        self.selected_trait_method(candidate, self_ty, args, selected)
    }

    fn unsized_for_method(&mut self, method: ItemRef, self_ty: Ty) -> Result<bool> {
        if matches!(
            self.cx.types.kind(self.resolve(self_ty)),
            Kind::Ref(..) | Kind::Ptr(..)
        ) {
            return Ok(false);
        }
        let Some(sized) = self.cx.lang_item("sized") else {
            return Ok(false);
        };
        Ok(self.needs_sized_self(method)? && !self.compiler_holds(sized, self_ty))
    }

    fn selected_trait_method(
        &mut self,
        candidate: TraitCandidate,
        self_ty: Ty,
        args: Vec<Ty>,
        selected: Picked,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        self.check_trait_impl_block(selected.0)?;
        if self.poly.is_some() && (self.has_params(self_ty) || self.any_has_params(&args)) {
            let substs: Vec<Ty> = std::iter::once(self_ty).chain(args).collect();
            self.obligations.push((candidate.of, substs.clone()));
            return Ok(Some((candidate.item, substs)));
        }
        Ok(Some(
            match self.impl_member(
                candidate.of,
                (self_ty, &args),
                selected,
                self.cx.name(candidate.item),
            )? {
                Some((item, (_, impl_args))) => (item, impl_args),
                None => (
                    candidate.item,
                    std::iter::once(self_ty).chain(args).collect(),
                ),
            },
        ))
    }

    fn candidate_self(&mut self, candidate: TraitCandidate, adjusted: Ty) -> Result<Option<Ty>> {
        let self_ty = match candidate.kind {
            SelfKind::Typed(ty) => self.typed_self(candidate, ty, adjusted)?,
            _ => self.trait_self(candidate, adjusted),
        };
        Ok(self_ty.filter(|&self_ty| {
            candidate
                .bound
                .is_none_or(|bound| self.substitute(bound) == self.substitute(self_ty))
        }))
    }

    fn pending_trait(
        &mut self,
        candidate: TraitCandidate,
        receiver: Ty,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some(self_ty) = self.trait_self(candidate, receiver) else {
            return Ok(None);
        };
        if !matches!(self.cx.types.kind(self.resolve(self_ty)), Kind::Var(_)) {
            return Ok(None);
        }
        let substs: Vec<Ty> = std::iter::once(self_ty)
            .chain(self.fresh_trait_args(candidate.of)?)
            .collect();
        self.obligations.push((candidate.of, substs.clone()));
        Ok(Some((candidate.item, substs)))
    }

    fn unselected_trait(
        &mut self,
        candidate: TraitCandidate,
        self_ty: Ty,
        args: Vec<Ty>,
    ) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        if !self.may_implement(candidate.of, self_ty, &args)? {
            return Ok(None);
        }
        let substs: Vec<Ty> = std::iter::once(self_ty).chain(args).collect();
        self.obligations.push((candidate.of, substs.clone()));
        Ok(Some((candidate.item, substs)))
    }

    fn typed_self(
        &mut self,
        candidate: TraitCandidate,
        ty: TyId,
        adjusted: Ty,
    ) -> Result<Option<Ty>> {
        let sig = self.cx.signature(candidate.item)?;
        let self_ty = self.fresh(Kind::Var);
        let substs: Vec<Ty> = std::iter::once(self_ty)
            .chain(self.fresh_trait_args(candidate.of)?)
            .collect();
        let frame = self.frame(candidate.item, &substs)?;
        let declared = self.with_frame(frame, |infer| infer.lower_ty(&sig.body, ty))?;
        let found = self.trial(|infer| {
            Ok(infer
                .unifies(declared, adjusted)
                .then(|| infer.resolve(self_ty)))
        })?;
        Ok(found.filter(|found| {
            !matches!(self.cx.types.kind(*found), Kind::Var(_))
                && candidate
                    .bound
                    .is_none_or(|param| param == self.substitute(*found))
        }))
    }

    fn trait_self(&mut self, candidate: TraitCandidate, adjusted: Ty) -> Option<Ty> {
        let self_ty = match (candidate.kind, self.cx.types.kind(adjusted).clone()) {
            (SelfKind::Value, Kind::Slice(_) | Kind::Prim(Primitive::Str)) => {
                return None;
            }
            (SelfKind::Value, Kind::Adt(item, _))
                if self.cx.foreign(item) || self.cx.is_unsized(adjusted).unwrap_or(false) =>
            {
                return None;
            }
            (SelfKind::Value, _) => adjusted,
            (SelfKind::Ref, Kind::Ref(Mut::Not, inner))
            | (SelfKind::RefMut, Kind::Ref(Mut::Mut, inner)) => inner,
            _ => return None,
        };
        candidate
            .bound
            .is_none_or(|param| param == self.substitute(self_ty))
            .then_some(self_ty)
    }

    pub(super) fn try_receiver(
        &mut self,
        candidate: Candidate,
        adjusted: Ty,
    ) -> Result<Option<(Vec<Ty>, Vec<Ty>)>> {
        self.trial(|infer| Ok(infer.receiver_matches(candidate, adjusted)))
    }

    fn receiver_matches(
        &mut self,
        candidate: Candidate,
        adjusted: Ty,
    ) -> Option<(Vec<Ty>, Vec<Ty>)> {
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
        let frame = Frame {
            src,
            home: self.cx.module_of(candidate.item).ok()?,
            generics,
            self_ty: Some(self_ty),
            block: Some(candidate.block),
            item: None,
            root: candidate.block.root(),
        };
        let receiver = self
            .with_frame(frame, |infer| infer.receiver_ty(&sig.body, candidate.kind))
            .ok()?;
        self.unifies(receiver, adjusted).then_some((args, own))
    }
}
