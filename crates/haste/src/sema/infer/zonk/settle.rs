use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::names::{ItemRef, Primitive};

use super::super::super::substs::Substs;
use super::super::super::ty::{Kind, Ty};
use super::super::poly::PolyPhase;
use super::super::{Infer, Res};

const INT_DEFAULTS: [Primitive; 12] = [
    Primitive::I32,
    Primitive::Usize,
    Primitive::U32,
    Primitive::U64,
    Primitive::I64,
    Primitive::U8,
    Primitive::U16,
    Primitive::I8,
    Primitive::I16,
    Primitive::Isize,
    Primitive::U128,
    Primitive::I128,
];

const FLOAT_DEFAULTS: [Primitive; 4] = [
    Primitive::F64,
    Primitive::F32,
    Primitive::F16,
    Primitive::F128,
];

impl Infer<'_, '_> {
    pub(super) fn settle_all(&mut self) -> Result<()> {
        self.settle_initial_passes()?;
        self.settle_final_passes()?;
        if let Some((of, substs)) = self.obligations.first() {
            let self_ty = substs[0];
            let resolved = self.resolve(self_ty);
            let site = self
                .obligation_sites
                .get(&(*of, self_ty))
                .or_else(|| self.obligation_sites.get(&(*of, resolved)))
                .copied();
            let ty = self.show(self_ty);
            let trait_name = self.cx.name(*of);
            let label = if self.is_thread_auto_trait(*of) && trait_name == "Send" {
                format!("`{ty}` cannot be sent between threads safely")
            } else if self.is_thread_auto_trait(*of) && trait_name == "Sync" {
                format!("`{ty}` cannot be shared between threads safely")
            } else {
                format!("the trait `{trait_name}` is not implemented for `{ty}`")
            };
            return Err(self.diag_expr(
                crate::diag::DiagnosticKind::Trait,
                site,
                format!("the trait bound `{ty}: {trait_name}` is not satisfied"),
                label,
            ));
        }
        Ok(())
    }

    pub(in super::super) fn settle_projection_loop(&mut self) -> Result<()> {
        for _ in 0..crate::RECURSION_LIMIT {
            let before = (
                self.projections.len(),
                self.obligations.len(),
                self.bound_vars(),
            );
            self.settle_projections()?;
            self.settle_obligations()?;
            let stalled = self.projections.len() >= before.0
                && self.obligations.len() >= before.1
                && self.bound_vars() == before.2;
            if self.obligations.is_empty() || stalled {
                break;
            }
        }
        Ok(())
    }

    fn settle_initial_passes(&mut self) -> Result<()> {
        self.settle_projection_loop()?;
        self.settle_opaques(true)?;
        self.settle_awaits()?;
        self.settle_operators(false)?;
        self.settle_projection_loop()?;
        self.settle_casts(false)
    }

    fn settle_casts(&mut self, final_pass: bool) -> Result<()> {
        for (inner, found, target) in std::mem::take(&mut self.casts) {
            let resolved_target = self.resolve(target);
            if matches!(*self.cx.types.kind(resolved_target), Kind::Var(_)) {
                if !final_pass {
                    self.casts.push((inner, found, target));
                }
                continue;
            }
            let Kind::Ptr(_, open) = *self.cx.types.kind(resolved_target) else {
                if !self.out.coercions.contains_key(&inner) {
                    if matches!(*self.cx.types.kind(self.resolve(found)), Kind::Var(_))
                        && !final_pass
                    {
                        self.casts.push((inner, found, target));
                    } else if let Some(coercion) = self.coercion(found, target)? {
                        self.out.coercions.insert(inner, coercion);
                    } else {
                        self.finish_identity_cast(found, target)?;
                    }
                }
                continue;
            };
            if !self.has_vars(open) {
                if matches!(*self.cx.types.kind(self.resolve(found)), Kind::Var(_)) && !final_pass {
                    self.casts.push((inner, found, target));
                } else if !self.out.coercions.contains_key(&inner)
                    && let Some(coercion) = self.coercion(found, target)?
                {
                    self.out.coercions.insert(inner, coercion);
                }
                continue;
            }
            match *self.cx.types.kind(self.resolve(found)) {
                Kind::Var(_) if !final_pass => self.casts.push((inner, found, target)),
                Kind::Ref(_, pointee) | Kind::Ptr(_, pointee) => {
                    if matches!(self.cx.types.kind(self.resolve(open)), Kind::Var(_))
                        || self.probe(|infer| infer.unifies(open, pointee))
                    {
                        self.unify(open, pointee)?;
                    } else if !self.out.coercions.contains_key(&inner)
                        && let Some(coercion) = self.coercion(found, target)?
                    {
                        self.out.coercions.insert(inner, coercion);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn settle_final_passes(&mut self) -> Result<()> {
        self.fall_back_residuals()?;
        self.settle_projection_loop()?;
        self.fall_back_operands()?;
        self.fall_back_numbers()?;
        self.fall_back_diverging()?;
        self.settle_obligations()?;
        self.settle_operators(true)?;
        self.settle_projection_loop()?;
        self.settle_casts(true)?;
        self.settle_offsets()?;
        self.settle_captures()?;
        self.captures_settled = true;
        self.settle_obligations()?;
        self.check_all_matches_exhaustive()?;
        self.settle_projection_loop()?;
        self.settle_obligations()
    }

    fn bound_vars(&self) -> usize {
        self.vars.iter().filter(|slot| slot.is_some()).count()
    }

    fn fall_back_numbers(&mut self) -> Result<()> {
        let exprs = self
            .out
            .exprs
            .iter()
            .copied()
            .filter(|&ty| ty != Ty::NONE && self.cx.types.has_var_flag(ty));
        let locals = self
            .out
            .locals
            .iter()
            .map(|local| local.ty)
            .filter(|&ty| self.cx.types.has_var_flag(ty));
        let roots: Vec<Ty> = exprs.chain(locals).collect();
        if roots.is_empty() {
            return Ok(());
        }
        self.fall_back_in(roots)
    }

    pub(in super::super) fn fall_back_in(&mut self, mut stack: Vec<Ty>) -> Result<()> {
        let mut seen = HashSet::default();
        while let Some(ty) = stack.pop() {
            if !self.cx.types.has_var_flag(ty) {
                continue;
            }
            let ty = self.resolve(ty);
            if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
                continue;
            }
            let fallback = match self.cx.types.kind(ty) {
                Kind::IntVar(_) => self.default_num_var(ty, &INT_DEFAULTS),
                Kind::FloatVar(_) => self.default_num_var(ty, &FLOAT_DEFAULTS),
                kind => {
                    stack.extend(kind.children(&self.cx.types));
                    continue;
                }
            };
            let prim = self.cx.types.prim(fallback);
            self.unify(ty, prim)?;
        }
        Ok(())
    }

    fn default_num_var(&mut self, var: Ty, candidates: &[Primitive]) -> Primitive {
        let default = candidates[0];
        let relevant: Vec<(ItemRef, Vec<Ty>)> = self
            .obligations
            .iter()
            .filter(|(of, substs)| {
                (self.captures_settled || !self.is_thread_auto_trait(*of))
                    && substs
                        .iter()
                        .any(|&ty| self.ty_mentions_var(ty, var, &mut HashSet::default()))
            })
            .cloned()
            .collect();
        if relevant.is_empty() || self.probe_num_candidate(var, default, &relevant) {
            return default;
        }
        let mut matched = None;
        for &prim in &candidates[1..] {
            if self.probe_num_candidate(var, prim, &relevant) {
                if matched.is_some() {
                    return default;
                }
                matched = Some(prim);
            }
        }
        matched.unwrap_or(default)
    }

    fn probe_num_candidate(
        &mut self,
        var: Ty,
        prim: Primitive,
        relevant: &[(ItemRef, Vec<Ty>)],
    ) -> bool {
        let prim_ty = self.cx.types.prim(prim);
        let saved = self.save_deferred();
        let saved_poly = saved.2;
        let ok = self.probe(|infer| {
            if !infer.unifies(var, prim_ty) {
                return false;
            }
            infer.obligations = relevant.to_vec();
            if infer.settle_projection_loop().is_err() || !infer.obligations.is_empty() {
                return false;
            }
            !saved_poly.is_some_and(|(was, _)| {
                was != PolyPhase::Fallback
                    && infer
                        .poly
                        .as_ref()
                        .is_some_and(|p| p.phase == PolyPhase::Fallback)
            })
        });
        self.restore_deferred(saved);
        ok
    }

    fn ty_mentions_var(&self, ty: Ty, var: Ty, seen: &mut HashSet<Ty>) -> bool {
        if !self.cx.types.has_var_flag(ty) {
            return false;
        }
        let ty = self.resolve(ty);
        if ty == var {
            return true;
        }
        if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
            return false;
        }
        self.cx
            .types
            .kind(ty)
            .children(&self.cx.types)
            .any(|child| self.ty_mentions_var(child, var, seen))
    }

    fn fall_back_residuals(&mut self) -> Result<()> {
        let Some(of) = self.cx.lang_item("Try") else {
            return Ok(());
        };
        let calls: Vec<Res> = self
            .out
            .tries
            .values()
            .map(|each| each.from_residual)
            .collect();
        for call in calls {
            let Res::Fn(_, substs) = call else {
                continue;
            };
            let &[ret, residual] = self.cx.types.substs(substs) else {
                continue;
            };
            if self.has_vars(ret) || !self.has_vars(residual) {
                continue;
            }
            let own = self.project(of, ret, Vec::new(), "Residual")?;
            if self.probe(|infer| infer.unifies(own, residual)) {
                self.unify(own, residual)?;
            }
        }
        Ok(())
    }

    pub(in super::super) fn settle_obligations(&mut self) -> Result<()> {
        let limit = self
            .obligations
            .len()
            .saturating_add(crate::RECURSION_LIMIT);
        let mut steps = 0;
        let mut at = 0;
        while at < self.obligations.len() && steps < limit {
            steps += 1;
            let (of, substs) = self.obligations[at].clone();
            if self.step_obligation(of, &substs)? {
                self.obligations.swap_remove(at);
            } else {
                at += 1;
            }
        }
        Ok(())
    }

    fn step_obligation(&mut self, of: ItemRef, substs: &[Ty]) -> Result<bool> {
        let self_ty = self.resolve(substs[0]);
        if self.is_thread_auto_trait(of) {
            if !self.captures_settled {
                return Ok(false);
            }
            if self.poly.is_some() && !self.has_vars(self_ty) && self.has_params(self_ty) {
                return Ok(self.thread_auto_holds(of, self_ty));
            }
        }
        if let Some(done) = self.step_var_obligation(of, self_ty, substs)? {
            return Ok(done);
        }
        let known = substs[1..].iter().all(|ty| !self.has_vars(*ty));
        let closed_key =
            (!self.has_vars(self_ty) && known).then(|| self.closed_obligation_key(of, substs));
        if closed_key.is_some_and(|key| self.proved.contains(&key)) {
            return Ok(true);
        }
        if self.poly.is_some() && closed_key.is_some() && self.any_has_params(substs) {
            return self.poly_settle_obligation(of, self_ty, &substs[1..]);
        }
        let done = ((!known || self.any_has_params(substs)) && self.assume(of, substs)?)
            || self.settled_by_impl(of, self_ty, &substs[1..])?;
        if done
            && self.trail_depth == 0
            && let Some(key) = closed_key
        {
            self.proved.insert(key);
        }
        Ok(done)
    }

    fn closed_obligation_key(&mut self, of: ItemRef, substs: &[Ty]) -> (ItemRef, Substs) {
        let resolved: Vec<Ty> = substs.iter().map(|&ty| self.substitute(ty)).collect();
        (of, self.cx.types.intern_substs(&resolved))
    }

    fn step_var_obligation(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        substs: &[Ty],
    ) -> Result<Option<bool>> {
        match self.cx.types.kind(self_ty) {
            Kind::Var(_) => {
                let held = self.opaque_holds(of, substs)?;
                Ok(Some(
                    held == Some(true)
                        || (held.is_none()
                            && self.callable_opaque_holds(of, self_ty, &substs[1..])?),
                ))
            }
            Kind::IntVar(_) | Kind::FloatVar(_) => Ok(Some(
                self.opaque_holds(of, substs)? == Some(true)
                    || self.settled_num_var(of, self_ty, &substs[1..])?,
            )),
            _ => Ok(None),
        }
    }

    fn settled_num_var(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        let settled = self.trial(|infer| {
            Ok(infer
                .select(of, self_ty, args)?
                .filter(|_| !infer.has_vars(self_ty))
                .map(|_| ()))
        })?;
        Ok(settled.is_some())
    }

    fn settled_by_impl(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        Ok(self.builtin_obligation_holds(of, self_ty, args)?
            || self.select(of, self_ty, args)?.is_some())
    }

    fn builtin_obligation_holds(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        if !self.has_vars(self_ty) {
            if self.cx.lang(of) == Some("unsize") {
                return Ok(args.first().is_some_and(|&to| !self.has_vars(to))
                    && self.unsize_holds(of, self_ty, args[0])?);
            }
            if self.compiler_decides(of, self_ty) {
                return Ok(self.compiler_holds(of, self_ty));
            }
            if self.compiler_impl(of, self_ty)
                || matches!(
                    self.cx.lang(of),
                    Some("discriminant_kind" | "bikeshed_guaranteed_no_drop")
                )
                || self.object_holds(of, self_ty, args) == Some(true)
            {
                return Ok(true);
            }
        }
        self.callable_obligation_holds(of, self_ty, args)
    }

    fn callable_obligation_holds(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        self.check_callable_obligation(of, self_ty, args)
    }

    fn fall_back_diverging(&mut self) -> Result<()> {
        let unit = self.cx.types.unit();
        for index in 0..self.var_diverges.len() {
            if self.var_diverges[index] && index < self.vars.len() && self.vars[index].is_none() {
                let var = self.cx.types.intern(Kind::Var(index as u32));
                self.unify(var, unit)?;
            }
        }
        Ok(())
    }

    pub(super) fn ensure_no_type_vars(&self) -> Result<()> {
        let mut seen = HashSet::default();
        let tys = self
            .out
            .exprs
            .iter()
            .copied()
            .filter(|&ty| ty != Ty::NONE)
            .chain(self.out.locals.iter().map(|local| local.ty))
            .chain(self.out.closures.values().map(|closure| closure.ret))
            .chain(self.out.pats.values().copied())
            .chain(self.out.ret);
        for ty in tys {
            self.check_no_type_vars(ty, &mut seen)?;
        }
        self.ensure_no_table_vars(&mut seen)
    }

    fn ensure_no_table_vars(&self, seen: &mut HashSet<Ty>) -> Result<()> {
        for map in [
            &self.out.res,
            &self.out.operators,
            &self.out.try_blocks,
            &self.out.box_frees,
        ] {
            for res in map.values() {
                self.check_res_vars(res, seen)?;
            }
        }
        for res in self.out.pat_res.values().chain(self.out.derefs.values()) {
            self.check_res_vars(res, seen)?;
        }
        self.ensure_no_flow_vars(seen)
    }

    fn ensure_no_flow_vars(&self, seen: &mut HashSet<Ty>) -> Result<()> {
        for each in self.out.fors.values() {
            self.check_res_vars(&each.into_iter, seen)?;
            self.check_res_vars(&each.next, seen)?;
            self.check_no_type_vars(each.iter, seen)?;
            self.check_no_type_vars(each.option, seen)?;
        }
        for each in self.out.awaits.values() {
            self.check_res_vars(&each.into_future, seen)?;
            self.check_res_vars(&each.poll, seen)?;
            self.check_no_type_vars(each.future, seen)?;
            self.check_no_type_vars(each.polled, seen)?;
        }
        for each in self.out.tries.values() {
            self.check_res_vars(&each.branch, seen)?;
            self.check_res_vars(&each.from_residual, seen)?;
            self.check_no_type_vars(each.flow, seen)?;
        }
        Ok(())
    }

    fn check_no_type_vars(&self, ty: Ty, seen: &mut HashSet<Ty>) -> Result<()> {
        if !self.cx.types.has_var_flag(ty) {
            return Ok(());
        }
        let ty = self.resolve(ty);
        if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
            return Ok(());
        }
        match self.cx.types.kind(ty) {
            Kind::Var(_) => anyhow::bail!("type annotations needed"),
            kind => {
                for child in kind.children(&self.cx.types) {
                    self.check_no_type_vars(child, seen)?;
                }
                Ok(())
            }
        }
    }

    fn check_res_vars(&self, res: &Res, seen: &mut HashSet<Ty>) -> Result<()> {
        match res {
            Res::Fn(_, substs) | Res::Ctor(_, _, substs) => {
                for &ty in self.cx.types.substs(*substs) {
                    self.check_no_type_vars(ty, seen)?;
                }
                Ok(())
            }
            Res::Const(ty) => self.check_no_type_vars(*ty, seen),
            _ => Ok(()),
        }
    }
}
