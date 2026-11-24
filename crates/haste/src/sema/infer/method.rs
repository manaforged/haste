mod candidates;

use anyhow::{Result, bail};

use crate::body::sig::SelfKind;
use crate::body::{ExprId, List};
use crate::names::{ItemRef, plain};

use super::super::impls::ImplRef;
use super::super::ty::{Kind, Mut, Ty};
use super::{Infer, Res};

/// How a method call adjusts its receiver: through how many references it reads, then whether it borrows the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adjust {
    pub derefs: u32,
    pub autoref: Option<Mut>,
}

#[derive(Clone, Copy)]
struct TraitCandidate {
    of: ItemRef,
    item: ItemRef,
    kind: SelfKind,
    bound: Option<Ty>,
}

struct Pick {
    item: ItemRef,
    args: Vec<Ty>,
    own: Vec<Ty>,
    autoref: Option<Mut>,
    bound_trait: Option<ItemRef>,
}

#[derive(Clone, Copy)]
struct Candidate {
    block: ImplRef,
    item: ItemRef,
    kind: SelfKind,
}

impl<'a> Infer<'_, 'a> {
    /// Infers the method call `expr`: picks the method by the receiver's type through references and borrows, as rustc probes.
    pub fn method_call(
        &mut self,
        expr: ExprId,
        receiver: ExprId,
        name: u32,
        params: (Option<List>, List),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        if !self.shared_here(None) {
            self.poly_fallback();
        }
        let found = self.expr(receiver, None)?;
        let at = name;
        let name = plain(self.src.text(name));
        let mut steps = vec![self.known_ty(found)?];
        while let Some(next) =
            self.next_step(expr, steps.len() as u32 - 1, steps[steps.len() - 1])?
        {
            if steps.len() == crate::RECURSION_LIMIT {
                bail!(
                    "reached the recursion limit while dereferencing {}",
                    self.show(found)
                );
            }
            steps.push(self.known_ty(next)?);
        }
        if let Some(pick) = self.opaque_pick(steps[0], name)? {
            return self.picked(expr, (pick, 0), params, expected);
        }
        let count = self.body.list(params.1).len();
        let origin = self.receiver_origin(receiver);
        let outer = std::mem::replace(&mut self.receiver_param, origin);
        let mut probed = self.probe_steps((expr, found), (name, count), &steps, expected);
        if matches!(probed, Ok(None))
            && let Some(&last) = steps.last()
            && let Some(default_prim) = match self.cx.types.kind(last) {
                Kind::IntVar(_) => Some(crate::names::Primitive::I32),
                Kind::FloatVar(_) => Some(crate::names::Primitive::F64),
                _ => None,
            }
        {
            let default_ty = self.cx.types.prim(default_prim);
            if let Some((pick, resolved)) = self.trial(|infer| {
                if !infer.unifies(last, default_ty) {
                    return Ok(None);
                }
                let resolved: Vec<Ty> = steps.iter().map(|&s| infer.resolve(s)).collect();
                Ok(infer
                    .probe_steps((expr, found), (name, count), &resolved, expected)?
                    .map(|pick| (pick, resolved)))
            })? {
                steps = resolved;
                probed = Ok(Some(pick));
            }
        }
        self.receiver_param = outer;
        let probed = probed?;
        self.finish_method_call((expr, found), (name, at), &steps, probed, params, expected)
    }

    fn receiver_origin(&self, receiver: ExprId) -> Option<&'a str> {
        match *self.body.expr(receiver) {
            crate::body::Expr::Paren(inner) => self.receiver_origin(inner),
            crate::body::Expr::Path(_) => match self.out.res.get(&receiver) {
                Some(Res::Local(local)) => self
                    .param_origins
                    .iter()
                    .find(|(at, _)| at == local)
                    .map(|&(_, name)| name),
                _ => None,
            },
            _ => None,
        }
    }

    fn finish_method_call(
        &mut self,
        (expr, found): (ExprId, Ty),
        (name, at): (&str, u32),
        steps: &[Ty],
        probed: Option<(Pick, u32)>,
        params: (Option<List>, List),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        match probed {
            Some((pick, derefs)) => {
                self.prune_method_steps(expr, steps, derefs, pick.bound_trait.is_some());
                self.picked(expr, (pick, derefs), params, expected)
            }
            None => {
                if steps
                    .iter()
                    .any(|&s| matches!(self.cx.types.kind(s), Kind::Proj(_)))
                {
                    self.poly_fallback();
                }
                Err(self.no_method(steps[steps.len() - 1], found, (name, at)))
            }
        }
    }

    fn prune_method_steps(&mut self, expr: ExprId, steps: &[Ty], derefs: u32, from_bound: bool) {
        self.out.derefs.retain(|&(e, s), _| e != expr || s < derefs);
        if self.poly.is_some()
            && (steps[..derefs as usize]
                .iter()
                .any(|&s| matches!(self.cx.types.kind(s), Kind::Proj(_)))
                || (!from_bound
                    && matches!(self.cx.types.kind(steps[derefs as usize]), Kind::Proj(_))))
        {
            self.poly_fallback();
        }
    }

    fn probe_steps(
        &mut self,
        (expr, found): (ExprId, Ty),
        (name, count): (&str, usize),
        steps: &[Ty],
        expected: Option<Ty>,
    ) -> Result<Option<(Pick, u32)>> {
        let mut traits = self.trait_candidates(name, steps)?;
        self.keep_fitting_bounds(&mut traits, count)?;
        let mut candidates = Vec::new();
        if let Some(hinted_of) = self
            .method_hints
            .as_ref()
            .and_then(|m| m.get(&expr).copied())
            && traits.iter().any(|c| c.of == hinted_of)
        {
            traits.retain(|c| c.of == hinted_of);
        } else {
            for &step in steps {
                self.candidates(step, name, &mut candidates)?;
            }
        }
        for (derefs, &step) in steps.iter().enumerate() {
            if let Some(pick) = self.probe_step(step, (&candidates, &traits), expected)? {
                if pick.autoref == Some(Mut::Mut) {
                    self.steps_mutably(expr, found, derefs as u32)?;
                }
                return Ok(Some((pick, derefs as u32)));
            }
        }
        Ok(None)
    }

    fn no_method(&self, last: Ty, found: Ty, (name, at): (&str, u32)) -> anyhow::Error {
        match self.cx.types.kind(last) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => self.diag_tok(
                crate::diag::DiagnosticKind::Type,
                at,
                format!("type annotations needed: cannot resolve receiver type for `{name}`"),
                "cannot infer type for receiver",
            ),
            _ => self.diag_tok(
                crate::diag::DiagnosticKind::Name,
                at,
                format!(
                    "no method named `{name}` found for type `{}`",
                    self.show(found)
                ),
                "method not found",
            ),
        }
    }

    fn opaque_pick(&mut self, ty: Ty, name: &str) -> Result<Option<Pick>> {
        for (of, substs) in self.opaque_bounds(ty)? {
            if let Some(item) = self.cx.trait_member(of, name)
                && let Some(autoref) =
                    borrow_of(self.cx.signature(item)?.receiver.map(|(kind, _)| kind))
            {
                self.obligations.push((of, substs.clone()));
                return Ok(Some(Pick {
                    item,
                    args: substs,
                    own: Vec::new(),
                    autoref,
                    bound_trait: None,
                }));
            }
        }
        Ok(None)
    }

    fn probe_step(
        &mut self,
        step: Ty,
        (candidates, traits): (&[Candidate], &[TraitCandidate]),
        expected: Option<Ty>,
    ) -> Result<Option<Pick>> {
        let (bounded, all_unbounded): (Vec<TraitCandidate>, Vec<TraitCandidate>) = traits
            .iter()
            .partition(|candidate| candidate.bound.is_some());
        let step_ty = self.substitute(step);
        let is_var = matches!(self.cx.types.kind(self.resolve(step)), Kind::Var(_));
        let bounded_at: Vec<TraitCandidate> = bounded
            .iter()
            .map(|candidate| TraitCandidate {
                bound: candidate.bound.map(|bound| self.substitute(bound)),
                ..*candidate
            })
            .collect();
        let wanted = self.known(expected);
        let unshadowed: Vec<TraitCandidate> = all_unbounded
            .iter()
            .copied()
            .filter(|candidate| !shadowed(&bounded_at, candidate.of, step_ty))
            .collect();
        for autoref in [None, Some(Mut::Not), Some(Mut::Mut)] {
            if autoref.is_some() && is_var {
                continue;
            }
            if let Some(pick) = self.probe_adjusted(
                step,
                autoref,
                candidates,
                (&bounded, &all_unbounded, &unshadowed),
                wanted,
            )? {
                return Ok(Some(pick));
            }
        }
        if matches!(self.cx.types.kind(step), Kind::Proj(_)) {
            self.poly_fallback();
        }
        Ok(None)
    }

    fn probe_adjusted(
        &mut self,
        step: Ty,
        autoref: Option<Mut>,
        candidates: &[Candidate],
        (bounded, all_unbounded, unshadowed): (
            &[TraitCandidate],
            &[TraitCandidate],
            &[TraitCandidate],
        ),
        wanted: Option<Ty>,
    ) -> Result<Option<Pick>> {
        let adjusted = autoref.map_or(step, |mutability| {
            self.cx.types.intern(Kind::Ref(mutability, step))
        });
        let mut bypassed = false;
        if let Some(pick) = self.trial(|infer| {
            let Some(pick) = infer.probe_traits(bounded, (adjusted, autoref), true)? else {
                return Ok(None);
            };
            if all_unbounded.is_empty()
                || wanted.is_none_or(|wanted| infer.fits_expected(&pick, wanted))
            {
                return Ok(Some(pick));
            }
            bypassed = true;
            Ok(None)
        })? {
            return Ok(Some(pick));
        }
        for &candidate in candidates {
            if let Some((args, own)) = self.try_receiver(candidate, adjusted)? {
                return Ok(Some(Pick {
                    item: candidate.item,
                    args,
                    own,
                    autoref,
                    bound_trait: None,
                }));
            }
        }
        let pool = if bypassed { all_unbounded } else { unshadowed };
        self.probe_traits(pool, (adjusted, autoref), !bypassed)
    }

    fn probe_traits(
        &mut self,
        traits: &[TraitCandidate],
        (adjusted, autoref): (Ty, Option<Mut>),
        use_bounds: bool,
    ) -> Result<Option<Pick>> {
        for &candidate in traits {
            if let Some((item, args)) = self.try_trait(candidate, adjusted, use_bounds)? {
                return Ok(Some(Pick {
                    item,
                    args,
                    own: Vec::new(),
                    autoref,
                    bound_trait: candidate.bound.map(|_| candidate.of),
                }));
            }
        }
        Ok(None)
    }

    fn fits_expected(&mut self, pick: &Pick, wanted: Ty) -> bool {
        self.probe(|infer| {
            let Ok(own) = infer.own_args(pick.item, Vec::new()) else {
                return true;
            };
            let substs: Vec<Ty> = pick.args.iter().copied().chain(own).collect();
            let Ok((_, ret)) = infer.fn_types(pick.item, &substs) else {
                return true;
            };
            infer.is_never(ret) || infer.unifies(ret, wanted) || infer.coerces_to(ret, wanted)
        })
    }

    fn next_step(&mut self, expr: ExprId, derefs: u32, step: Ty) -> Result<Option<Ty>> {
        let next = match *self.cx.types.kind(step) {
            Kind::Array(element, _) => return Ok(Some(self.cx.types.intern(Kind::Slice(element)))),
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Ptr(..) | Kind::Slice(_) => {
                None
            }
            _ => match self.builtin_deref(step) {
                Some(inner) => Some(inner),
                None => self.deref_step_raw(expr, derefs, step)?,
            },
        };
        Ok(next.map(|next| self.resolve(next)))
    }

    fn picked(
        &mut self,
        expr: ExprId,
        (pick, derefs): (Pick, u32),
        (explicit, params): (Option<List>, List),
        expected: Option<Ty>,
    ) -> Result<Ty> {
        let Pick {
            item,
            args: mut substs,
            own,
            autoref,
            bound_trait,
        } = pick;
        if let (Some(of), Some(poly)) = (bound_trait, &mut self.poly) {
            poly.method_traits.insert(expr, of);
        }
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
        self.out
            .res
            .insert(expr, Res::fn_of(&self.cx.types, item, &substs));
        self.out.adjusts.insert(expr, adjust);
        if let (Some(Mut::Mut), crate::body::Expr::Method { receiver, .. }) =
            (adjust.autoref, *self.body.expr(expr))
        {
            self.note_mut(receiver)?;
        }
        let (types, ret) = self.fn_types(item, &substs)?;
        let given = self.body.list(params).to_vec();
        if given.len() + 1 != types.len() {
            return Err(self.diag_expr(
                crate::diag::DiagnosticKind::Type,
                Some(expr),
                format!(
                    "`{}` takes {} arguments, {} given",
                    self.cx.name(item),
                    types.len() - 1,
                    given.len()
                ),
                "unexpected number of arguments",
            ));
        }
        let hints = self.expected_inputs(&types[1..], ret, expected);
        for ((arg, param), hint) in given.into_iter().zip(types.into_iter().skip(1)).zip(hints) {
            self.call_arg(
                Some((item, &substs[..])),
                ExprId(arg),
                (param, hint),
                &mut None,
            )?;
        }
        Ok(ret)
    }
}

fn shadowed(bounded: &[TraitCandidate], of: ItemRef, step: Ty) -> bool {
    bounded
        .iter()
        .any(|candidate| candidate.of == of && candidate.bound == Some(step))
}

fn borrow_of(kind: Option<SelfKind>) -> Option<Option<Mut>> {
    match kind? {
        SelfKind::Value => Some(None),
        SelfKind::Ref => Some(Some(Mut::Not)),
        SelfKind::RefMut => Some(Some(Mut::Mut)),
        SelfKind::Typed(_) => None,
    }
}
