use anyhow::{Result, bail};

use crate::body::sig::SelfKind;
use crate::body::{ExprId, List};
use crate::names::{ItemRef, plain};

use super::super::impls::{ImplRef, head};
use super::super::ty::{Kind, Mut, Ty};
use super::scope::{Frame, type_params};
use super::{Infer, Res};

/// How a method call adjusts its receiver: through how many references it reads, then whether it borrows the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adjust {
    pub derefs: u32,
    pub autoref: Option<Mut>,
}

/// A trait method a call may pick: its trait, its declaration in the trait and how it takes `self`.
#[derive(Clone, Copy)]
struct TraitCandidate {
    of: ItemRef,
    item: ItemRef,
    kind: SelfKind,
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
    pub fn method_call(&mut self, expr: ExprId, receiver: ExprId, name: u32, params: (Option<List>, List)) -> Result<Ty> {
        let found = self.expr(receiver, None)?;
        let name = plain(self.src.text(name));
        let steps = self.deref_steps(found)?;
        let candidates = self.candidates(&steps, name)?;
        let traits = self.trait_candidates(name)?;
        for (derefs, &step) in steps.iter().enumerate() {
            for autoref in [None, Some(Mut::Not), Some(Mut::Mut)] {
                let adjusted = match autoref {
                    None => step,
                    Some(mutability) => self.cx.types.intern(Kind::Ref(mutability, step)),
                };
                let adjust = Adjust { derefs: derefs as u32, autoref };
                for &candidate in &candidates {
                    if let Some(args) = self.try_receiver(candidate, adjusted)? {
                        return self.picked(expr, candidate.item, args, adjust, params);
                    }
                }
                for &candidate in &traits {
                    if let Some((item, args)) = self.try_trait(candidate, adjusted)? {
                        return self.picked(expr, item, args, adjust, params);
                    }
                }
            }
        }
        bail!("no method `{name}` on {}", self.show(found))
    }

    fn deref_steps(&mut self, ty: Ty) -> Result<Vec<Ty>> {
        let mut steps = vec![self.resolve(ty)];
        loop {
            let last = *steps.last().expect("steps start with the receiver");
            match self.cx.types.kind(last).clone() {
                Kind::Ref(_, inner) => steps.push(self.resolve(inner)),
                Kind::Array(element, _) => {
                    steps.push(self.cx.types.intern(Kind::Slice(element)));
                    return Ok(steps);
                }
                Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => bail!("the type of a method receiver must be known here"),
                _ => return Ok(steps),
            }
        }
    }

    fn candidates(&mut self, steps: &[Ty], name: &str) -> Result<Vec<Candidate>> {
        let mut blocks = Vec::new();
        for &step in steps {
            let Some(head) = head(self.cx.types.kind(step)) else { continue };
            for block in self.cx.inherent(head) {
                if !blocks.contains(&block) {
                    blocks.push(block);
                }
            }
        }
        let mut found = Vec::new();
        for block in blocks {
            let Some(item) = self.cx.member(block, name) else { continue };
            if let Some((kind, _)) = self.cx.signature(item)?.receiver {
                found.push(Candidate { block, item, kind });
            }
        }
        Ok(found)
    }

    fn trait_candidates(&mut self, name: &str) -> Result<Vec<TraitCandidate>> {
        let mut found = Vec::new();
        for of in self.cx.traits_in_scope(self.home) {
            let Some(item) = self.cx.trait_member(of, name) else { continue };
            if let Some((kind, _)) = self.cx.signature(item)?.receiver {
                found.push(TraitCandidate { of, item, kind });
            }
        }
        Ok(found)
    }

    /// The function a call of trait method `method` with receiver type `receiver` runs, and its leading type arguments.
    pub fn trait_method(&mut self, method: ItemRef, receiver: Ty) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let Some(of) = self.cx.owner_trait(method) else { bail!("{} is not a trait method", self.cx.name(method)) };
        let Some((kind, _)) = self.cx.signature(method)?.receiver else { bail!("{} takes no `self`", self.cx.name(method)) };
        let Some((item, mut substs)) = self.try_trait(TraitCandidate { of, item: method, kind }, receiver)? else { return Ok(None) };
        let own = type_params(&self.cx.signature(item)?.generics).count();
        substs.extend((0..own).map(|_| self.fresh(Kind::Var)));
        Ok(Some((item, substs)))
    }

    fn try_trait(&mut self, candidate: TraitCandidate, adjusted: Ty) -> Result<Option<(ItemRef, Vec<Ty>)>> {
        let self_ty = match (candidate.kind, self.cx.types.kind(adjusted).clone()) {
            (SelfKind::Value, _) => adjusted,
            (SelfKind::Ref, Kind::Ref(Mut::Not, inner)) | (SelfKind::RefMut, Kind::Ref(Mut::Mut, inner)) => inner,
            _ => return Ok(None),
        };
        let count = type_params(&self.cx.trait_generics(candidate.of)?.1.params).count();
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let saved = self.vars.clone();
        let Some((block, impl_args)) = self.select(candidate.of, self_ty, &args)? else {
            self.vars = saved;
            return Ok(None);
        };
        Ok(Some(match self.cx.member(block, self.cx.name(candidate.item)) {
            Some(item) => (item, impl_args),
            None => (candidate.item, std::iter::once(self_ty).chain(args).collect()),
        }))
    }

    fn try_receiver(&mut self, candidate: Candidate, adjusted: Ty) -> Result<Option<Vec<Ty>>> {
        let saved = self.vars.clone();
        let matched = self.receiver_matches(candidate, adjusted);
        if matched.is_none() {
            self.vars = saved;
        }
        Ok(matched)
    }

    fn receiver_matches(&mut self, candidate: Candidate, adjusted: Ty) -> Option<Vec<Ty>> {
        let count = self.impl_params(candidate.block).ok()?;
        let args: Vec<Ty> = (0..count).map(|_| self.fresh(Kind::Var)).collect();
        let (generics, self_ty) = self.impl_frame(candidate.block, &args).ok()?;
        let sig = self.cx.signature(candidate.item).ok()?;
        let frame = Frame { src: &self.cx.file(candidate.item).source, home: self.cx.module_of(candidate.item).ok()?, generics, self_ty: Some(self_ty), block: Some(candidate.block) };
        let receiver = self.with_frame(frame, |infer| infer.receiver_ty(&sig.body, candidate.kind)).ok()?;
        self.unify(receiver, adjusted).ok().map(|_| args)
    }

    fn picked(&mut self, expr: ExprId, item: ItemRef, args: Vec<Ty>, adjust: Adjust, (explicit, params): (Option<List>, List)) -> Result<Ty> {
        let sig = self.cx.signature(item)?;
        let own = super::scope::type_params(&sig.generics).count();
        let mut substs = args;
        let body = self.body;
        let given = match explicit {
            Some(list) => self.arg_list(body, list)?,
            None => Vec::new(),
        };
        match given.len() {
            0 => substs.extend((0..own).map(|_| self.fresh(Kind::Var))),
            count if count == own => substs.extend(given),
            count => bail!("{} takes {own} type arguments, {count} given", self.cx.name(item)),
        }
        self.out.res.insert(expr, Res::Fn(item, substs.clone()));
        self.out.adjusts.insert(expr, adjust);
        let (types, ret) = self.fn_types(item, &substs)?;
        let given = self.body.list(params).to_vec();
        if given.len() + 1 != types.len() {
            bail!("{} takes {} arguments, {} given", self.cx.name(item), types.len() - 1, given.len());
        }
        for (arg, param) in given.into_iter().zip(types.into_iter().skip(1)) {
            let found = self.expr(ExprId(arg), Some(param))?;
            self.coerce_expr(ExprId(arg), found, param)?;
        }
        Ok(ret)
    }
}
