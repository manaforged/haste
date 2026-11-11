//! Filling inference variables into the final types of a body once inference settles, with the numeric fallbacks.

use super::*;

impl Infer<'_, '_> {
    pub(super) fn zonk(&mut self, ty: Ty) -> Ty {
        self.fill(ty, true)
    }

    /// `ty` with every bound inference variable replaced by its binding, and the unbound ones kept.
    pub(super) fn substitute(&mut self, ty: Ty) -> Ty {
        self.fill(ty, false)
    }

    fn fill(&mut self, ty: Ty, fall_back: bool) -> Ty {
        let ty = self.resolve(ty);
        let kind = self.cx.types.kind(ty).clone();
        let fallback = match kind {
            Kind::IntVar(_) => self.cx.types.prim(Primitive::I32),
            Kind::FloatVar(_) => self.cx.types.prim(Primitive::F64),
            Kind::Var(_) => self.cx.types.unit(),
            _ => {
                let kind = kind.map(|item| self.fill(item, fall_back));
                return self.cx.types.intern(kind);
            }
        };
        if fall_back { fallback } else { ty }
    }

    pub(super) fn finish(mut self) -> Result<Typed> {
        self.settle_projections()?;
        self.settle_opaques(true)?;
        self.settle_awaits()?;
        self.settle_operators(false)?;
        self.settle_obligations()?;
        self.fall_back_numbers()?;
        self.settle_obligations()?;
        self.settle_operators(true)?;
        self.settle_projections()?;
        self.settle_captures()?;
        self.zonk_types()?;
        self.zonk_calls()?;
        Ok(self.out)
    }

    fn fall_back_numbers(&mut self) -> Result<()> {
        let exprs = self.out.exprs.iter().copied().filter(|ty| *ty != Ty::NONE);
        let mut stack: Vec<Ty> = exprs.chain(self.out.locals.iter().map(|local| local.ty)).collect();
        let mut seen = rustc_hash::FxHashSet::default();
        while let Some(ty) = stack.pop() {
            let ty = self.resolve(ty);
            if !seen.insert(ty) {
                continue;
            }
            let fallback = match self.cx.types.kind(ty) {
                Kind::IntVar(_) => Primitive::I32,
                Kind::FloatVar(_) => Primitive::F64,
                kind => {
                    stack.extend(kind.children());
                    continue;
                }
            };
            let prim = self.cx.types.prim(fallback);
            self.unify(ty, prim)?;
        }
        Ok(())
    }

    fn zonk_types(&mut self) -> Result<()> {
        let exprs = std::mem::take(&mut self.out.exprs);
        self.out.exprs = exprs.into_iter().map(|ty| if ty == Ty::NONE { ty } else { self.zonk(ty) }).collect();
        let locals = std::mem::take(&mut self.out.locals);
        self.out.locals = locals.into_iter().map(|local| Local { ty: self.zonk(local.ty), ..local }).collect();
        let pats = std::mem::take(&mut self.out.pats);
        self.out.pats = pats.into_iter().map(|(pat, ty)| (pat, self.zonk(ty))).collect();
        let res = std::mem::take(&mut self.out.res);
        self.out.res = self.zonk_all(res)?;
        let pat_res = std::mem::take(&mut self.out.pat_res);
        self.out.pat_res = self.zonk_all(pat_res)?;
        self.out.ret = self.out.ret.map(|ret| self.zonk(ret));
        let closures = std::mem::take(&mut self.out.closures);
        self.out.closures = closures.into_iter().map(|(expr, closure)| (expr, Closure { ret: self.zonk(closure.ret), ..closure })).collect();
        Ok(())
    }

    fn zonk_calls(&mut self) -> Result<()> {
        let fors = std::mem::take(&mut self.out.fors);
        self.out.fors = fors.into_iter().map(|(expr, each)| Ok((expr, self.zonk_for(each)?))).collect::<Result<_>>()?;
        let awaits = std::mem::take(&mut self.out.awaits);
        self.out.awaits = awaits.into_iter().map(|(expr, each)| Ok((expr, self.zonk_await(each)?))).collect::<Result<_>>()?;
        let tries = std::mem::take(&mut self.out.tries);
        self.out.tries = tries.into_iter().map(|(expr, each)| Ok((expr, self.zonk_try(each)?))).collect::<Result<_>>()?;
        let derefs = std::mem::take(&mut self.out.derefs);
        self.out.derefs = self.zonk_all(derefs)?;
        let operators = std::mem::take(&mut self.out.operators);
        self.out.operators = self.zonk_all(operators)?;
        let try_blocks = std::mem::take(&mut self.out.try_blocks);
        self.out.try_blocks = self.zonk_all(try_blocks)?;
        let box_frees = std::mem::take(&mut self.out.box_frees);
        self.out.box_frees = self.zonk_all(box_frees)?;
        let coercions = std::mem::take(&mut self.out.coercions);
        self.out.coercions = coercions.into_iter().map(|(expr, coercion)| Ok((expr, self.zonk_coercion(coercion)?))).collect::<Result<_>>()?;
        Ok(())
    }

    fn zonk_all<K: Eq + std::hash::Hash>(&mut self, map: HashMap<K, Res>) -> Result<HashMap<K, Res>> {
        map.into_iter().map(|(key, res)| Ok((key, self.zonk_res(res)?))).collect()
    }

    fn zonk_for(&mut self, each: For) -> Result<For> {
        Ok(For { into_iter: self.zonk_res(each.into_iter)?, next: self.zonk_res(each.next)?, iter: self.zonk(each.iter), option: self.zonk(each.option), some: each.some })
    }

    fn zonk_await(&mut self, each: Awaited) -> Result<Awaited> {
        Ok(Awaited { into_future: self.zonk_res(each.into_future)?, future: self.zonk(each.future), poll: self.zonk_res(each.poll)?, polled: self.zonk(each.polled) })
    }

    fn zonk_try(&mut self, each: Tried) -> Result<Tried> {
        Ok(Tried { branch: self.zonk_res(each.branch)?, from_residual: self.zonk_res(each.from_residual)?, flow: self.zonk(each.flow), ..each })
    }

    pub(super) fn zonk_res(&mut self, res: Res) -> Result<Res> {
        match res {
            Res::Fn(item, substs) => {
                let substs = substs.into_iter().map(|ty| self.zonk(ty)).collect();
                self.trait_call(item, substs)
            }
            Res::Ctor(item, variant, args) => Ok(Res::Ctor(item, variant, args.into_iter().map(|ty| self.zonk(ty)).collect())),
            Res::Const(value) => Ok(Res::Const(self.zonk(value))),
            other => Ok(other),
        }
    }
}
