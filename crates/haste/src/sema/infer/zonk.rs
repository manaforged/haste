use std::cell::RefCell;
use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::super::substs::{PackedMap, PackedSet};
use super::*;

const MAX_POOLED: usize = 8;

thread_local! {
    static OUT_POOL: RefCell<Vec<InferOut>> = const { RefCell::new(Vec::new()) };
}

#[derive(Default)]
pub(super) struct InferOut {
    pub exprs: Vec<Ty>,
    pub res: HashMap<ExprId, Res>,
    pub locals: Vec<Local>,
    pub binds: HashMap<PatId, u32>,
    pub adjusts: HashMap<ExprId, Adjust>,
    pub autoderefs: HashMap<ExprId, u32>,
    pub derefs: HashMap<(ExprId, u32), Res>,
    pub operators: HashMap<ExprId, Res>,
    pub coercions: HashMap<ExprId, Coercion>,
    pub fields: HashMap<ExprId, u32>,
    pub inits: HashMap<ExprId, u32>,
    pub offsets: HashMap<ExprId, u64>,
    pub ranges: HashMap<ExprId, Box<[u32]>>,
    pub field_pats: HashMap<PatId, u32>,
    pub pat_res: HashMap<PatId, Res>,
    pub pat_derefs: HashMap<PatId, u32>,
    pub ref_binds: HashMap<PatId, Mut>,
    pub or_binds: HashSet<PatId>,
    pub overloaded: HashSet<ExprId>,
    pub const_args: HashSet<ExprId>,
    pub pats: HashMap<PatId, Ty>,
    pub params: Vec<u32>,
    pub ret: Option<Ty>,
    pub fors: HashMap<ExprId, For>,
    pub tries: HashMap<ExprId, Tried>,
    pub awaits: HashMap<ExprId, Awaited>,
    pub closures: HashMap<ExprId, Closure>,
    pub try_blocks: HashMap<ExprId, Res>,
    pub box_frees: HashMap<ExprId, Res>,
    pub unpacks: HashMap<u32, PatId>,
}

impl InferOut {
    fn clear(&mut self) {
        self.exprs.clear();
        self.res.clear();
        self.locals.clear();
        self.binds.clear();
        self.adjusts.clear();
        self.autoderefs.clear();
        self.derefs.clear();
        self.operators.clear();
        self.coercions.clear();
        self.fields.clear();
        self.inits.clear();
        self.offsets.clear();
        self.ranges.clear();
        self.field_pats.clear();
        self.pat_res.clear();
        self.pat_derefs.clear();
        self.ref_binds.clear();
        self.or_binds.clear();
        self.overloaded.clear();
        self.const_args.clear();
        self.pats.clear();
        self.params.clear();
        self.ret = None;
        self.fors.clear();
        self.tries.clear();
        self.awaits.clear();
        self.closures.clear();
        self.try_blocks.clear();
        self.box_frees.clear();
        self.unpacks.clear();
    }
}

pub(super) fn take_out() -> InferOut {
    OUT_POOL.with(|pool| pool.borrow_mut().pop()).unwrap_or_default()
}

pub(super) fn recycle_out(out: &mut InferOut) {
    out.clear();
    OUT_POOL.with(|pool| {
        let mut vec = pool.borrow_mut();
        if vec.len() < MAX_POOLED {
            vec.push(std::mem::take(out));
        }
    });
}

impl Infer<'_, '_> {
    pub(in crate::sema) fn zonk(&mut self, ty: Ty) -> Ty {
        self.fill(ty, true)
    }

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
                let kind = self.map_kind(&kind, |this, item| this.fill(item, fall_back));
                return self.cx.types.intern(kind);
            }
        };
        if fall_back { fallback } else { ty }
    }

    pub(super) fn map_substs(&mut self, id: super::super::substs::Substs, mut f: impl FnMut(&mut Self, Ty) -> Ty) -> super::super::substs::Substs {
        let pool = self.cx.types.pool();
        pool.map(id, &mut |ty| f(self, ty))
    }

    pub(super) fn map_kind(&mut self, kind: &Kind, mut f: impl FnMut(&mut Self, Ty) -> Ty) -> Kind {
        let pools = self.cx.types.pools();
        kind.map_pool(&pools, |ty| f(self, ty))
    }

    fn settle_all(&mut self) -> Result<()> {
        self.settle_initial_passes()?;
        self.settle_final_passes()?;
        if let Some((of, substs)) = self.obligations.first() {
            let ty = self.show(substs[0]);
            anyhow::bail!("the trait bound `{ty}: {}` is not satisfied", self.cx.name(*of));
        }
        Ok(())
    }

    fn settle_initial_passes(&mut self) -> Result<()> {
        self.settle_projections()?;
        self.settle_obligations()?;
        self.settle_projections()?;
        self.settle_opaques(true)?;
        self.settle_awaits()?;
        self.settle_operators(false)?;
        self.settle_obligations()?;
        self.settle_casts()
    }

    fn settle_final_passes(&mut self) -> Result<()> {
        self.fall_back_numbers()?;
        self.fall_back_diverging()?;
        self.fall_back_residuals()?;
        self.settle_obligations()?;
        self.settle_operators(true)?;
        for _ in 0..crate::RECURSION_LIMIT {
            let before = (self.projections.len(), self.obligations.len(), self.bound_vars());
            self.settle_projections()?;
            self.settle_obligations()?;
            let stalled = self.projections.len() >= before.0
                && self.obligations.len() >= before.1
                && self.bound_vars() == before.2;
            if self.obligations.is_empty() || stalled {
                break;
            }
        }
        self.settle_captures()
    }

    fn bound_vars(&self) -> usize {
        self.vars.iter().filter(|slot| slot.is_some()).count()
    }

    pub(super) fn finish(mut self) -> Result<Typed> {
        self.settle_all()?;
        self.ensure_no_type_vars()?;
        self.pack_typed()
    }

    pub(super) fn finish_poly(mut self) -> Result<(Typed, Option<Box<super::poly::PolyState>>, bool)> {
        self.settle_all()?;
        let clean = self.shared_here(None) && self.backlog() == 0 && !self.has_unbound_out_vars();
        let typed = self.pack_typed()?;
        let mut poly = self.poly.take();
        if let Some(state) = &mut poly {
            super::poly::zonk_poly_state(&mut self, state);
        }
        Ok((typed, poly, clean))
    }

    fn has_unbound_out_vars(&self) -> bool {
        let mut seen = HashSet::default();
        let exprs = self.out.exprs.iter().copied().filter(|&ty| ty != Ty::NONE);
        let locals = self.out.locals.iter().map(|local| local.ty);
        let pats = self.out.pats.values().copied();
        exprs.chain(locals).chain(pats).chain(self.out.ret).any(|ty| self.has_vars_seen(ty, &mut seen))
    }

    fn fall_back_numbers(&mut self) -> Result<()> {
        let exprs = self.out.exprs.iter().copied().filter(|ty| *ty != Ty::NONE);
        let mut stack: Vec<Ty> = exprs.chain(self.out.locals.iter().map(|local| local.ty)).collect();
        let mut seen = HashSet::default();
        while let Some(ty) = stack.pop() {
            let ty = self.resolve(ty);
            if !seen.insert(ty) {
                continue;
            }
            let fallback = match self.cx.types.kind(ty) {
                Kind::IntVar(_) => Primitive::I32,
                Kind::FloatVar(_) => Primitive::F64,
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

    /// Gives each `?` whose residual is still unknown the residual of the function's own `Try` type, the one `From` impl a generic body could use.
    fn fall_back_residuals(&mut self) -> Result<()> {
        let Some(of) = self.cx.lang_item("Try") else { return Ok(()) };
        let calls: Vec<Res> = self.out.tries.values().map(|each| each.from_residual).collect();
        for call in calls {
            let Res::Fn(_, substs) = call else { continue };
            let &[ret, residual] = self.cx.types.substs(substs) else { continue };
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

    fn pack_syntax(&mut self) -> Arc<SyntaxTables> {
        let syntax = SyntaxTables {
            binds: PackedMap::pack(self.out.binds.drain()),
            adjusts: PackedMap::pack(self.out.adjusts.drain()),
            fields: PackedMap::pack(self.out.fields.drain()),
            inits: PackedMap::pack(self.out.inits.drain()),
            ranges: PackedMap::pack(self.out.ranges.drain()),
            field_pats: PackedMap::pack(self.out.field_pats.drain()),
            pat_derefs: PackedMap::pack(self.out.pat_derefs.drain()),
            ref_binds: PackedMap::pack(self.out.ref_binds.drain()),
            or_binds: PackedSet::pack(self.out.or_binds.drain()),
            const_args: PackedSet::pack(self.out.const_args.drain()),
            unpacks: PackedMap::pack(self.out.unpacks.drain()),
        };
        Arc::new(syntax)
    }

    fn zonk_locals_and_closures(&mut self) {
        for idx in 0..self.out.locals.len() {
            let ty = self.out.locals[idx].ty;
            self.out.locals[idx].ty = self.zonk(ty);
        }
        let keys: Vec<ExprId> = self.out.closures.keys().copied().collect();
        for key in keys {
            let ret = self.out.closures[&key].ret;
            let zonked = self.zonk(ret);
            if let Some(closure) = self.out.closures.get_mut(&key) {
                closure.ret = zonked;
            }
        }
    }

    pub(super) fn pack_typed(&mut self) -> Result<Typed> {
        self.zonk_locals_and_closures();
        let syntax = self.pack_syntax();
        let mut raw_exprs = std::mem::take(&mut self.out.exprs);
        let exprs: Box<[Ty]> = raw_exprs.drain(..).map(|ty| if ty == Ty::NONE { ty } else { self.zonk(ty) }).collect();
        self.out.exprs = raw_exprs;
        let mut raw_pats = std::mem::take(&mut self.out.pats);
        let pats = PackedMap::pack(raw_pats.drain().map(|(pat, ty)| (pat, self.zonk(ty))));
        self.out.pats = raw_pats;
        let res = self.drain_res_field(|out| &mut out.res)?;
        let pat_res = self.drain_res_field(|out| &mut out.pat_res)?;
        let derefs = self.drain_res_field(|out| &mut out.derefs)?;
        let operators = self.drain_res_field(|out| &mut out.operators)?;
        let try_blocks = self.drain_res_field(|out| &mut out.try_blocks)?;
        let box_frees = self.drain_res_field(|out| &mut out.box_frees)?;
        let fors = self.drain_mapped(|out| &mut out.fors, Self::zonk_for)?;
        let awaits = self.drain_mapped(|out| &mut out.awaits, Self::zonk_await)?;
        let tries = self.drain_mapped(|out| &mut out.tries, Self::zonk_try)?;
        let coercions = self.drain_mapped(|out| &mut out.coercions, Self::zonk_coercion)?;
        let autoderefs = PackedMap::pack(self.out.autoderefs.drain());
        let offsets = PackedMap::pack(self.out.offsets.drain());
        let overloaded = PackedSet::pack(self.out.overloaded.drain());
        let closures = PackedMap::pack(self.out.closures.drain());
        let ret = self.out.ret.take().map(|ret| self.zonk(ret));
        let locals = std::mem::take(&mut self.out.locals);
        let params = std::mem::take(&mut self.out.params);
        let tables = BodyTables {
            syntax,
            exprs,
            res,
            autoderefs,
            derefs,
            operators,
            coercions,
            offsets,
            pat_res,
            overloaded,
            pats,
            fors,
            tries,
            awaits,
            closures,
            try_blocks,
            box_frees,
        };
        Ok(Typed { tables: super::LazyTables::Eager(Arc::new(tables)), locals, params, ret, drops: Vec::new(), shape: Shape::default() })
    }

    fn drain_res_field<K: Copy + Ord + std::hash::Hash>(&mut self, field: fn(&mut InferOut) -> &mut HashMap<K, Res>) -> Result<PackedMap<K, Res>> {
        self.drain_mapped(field, Self::zonk_res)
    }

    fn drain_mapped<K: Copy + Ord + std::hash::Hash, V>(&mut self, field: fn(&mut InferOut) -> &mut HashMap<K, V>, map_fn: fn(&mut Self, V) -> Result<V>) -> Result<PackedMap<K, V>> {
        let raw: Vec<(K, V)> = field(&mut self.out).drain().collect();
        if raw.is_empty() {
            return Ok(PackedMap::default());
        }
        let mut pairs = Vec::with_capacity(raw.len());
        for (key, value) in raw {
            pairs.push((key, map_fn(self, value)?));
        }
        Ok(PackedMap::pack(pairs))
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

    pub(super) fn settle_obligations(&mut self) -> Result<()> {
        let limit = self.obligations.len().saturating_add(crate::RECURSION_LIMIT);
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
        if let Some(done) = self.step_var_obligation(of, self_ty, substs)? {
            return Ok(done);
        }
        let known = substs[1..].iter().all(|ty| !self.has_vars(*ty));
        let closed_key = (!self.has_vars(self_ty) && known).then(|| self.closed_obligation_key(of, substs));
        if closed_key.is_some_and(|key| self.proved.contains(&key)) {
            return Ok(true);
        }
        if self.poly.is_some() && closed_key.is_some() && self.any_has_params(substs) {
            return self.poly_settle_obligation(of, self_ty, &substs[1..]);
        }
        let done = ((!known || self.any_has_params(substs)) && self.assume(of, substs)?)
            || self.settled_by_impl(of, self_ty, &substs[1..])?;
        if done && self.trail_depth == 0 && let Some(key) = closed_key {
            self.proved.insert(key);
        }
        Ok(done)
    }

    fn closed_obligation_key(&mut self, of: ItemRef, substs: &[Ty]) -> (ItemRef, Substs) {
        let resolved: Vec<Ty> = substs.iter().map(|&ty| self.substitute(ty)).collect();
        (of, self.cx.types.intern_substs(&resolved))
    }

    fn step_var_obligation(&mut self, of: ItemRef, self_ty: Ty, substs: &[Ty]) -> Result<Option<bool>> {
        match self.cx.types.kind(self_ty) {
            Kind::Var(_) => {
                let held = self.opaque_holds(of, substs)?;
                Ok(Some(held == Some(true) || (held.is_none() && self.callable_opaque_holds(of, self_ty, &substs[1..])?)))
            }
            Kind::IntVar(_) | Kind::FloatVar(_) => Ok(Some(self.opaque_holds(of, substs)? == Some(true) || self.settled_num_var(of, self_ty, &substs[1..])?)),
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
        Ok(self.builtin_obligation_holds(of, self_ty, args)? || self.select(of, self_ty, args)?.is_some())
    }

    fn builtin_obligation_holds(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        if !self.has_vars(self_ty) {
            if self.compiler_decides(of, self_ty) {
                return Ok(self.compiler_holds(of, self_ty));
            }
            if self.compiler_impl(of, self_ty)
                || matches!(self.cx.lang(of), Some("discriminant_kind" | "bikeshed_guaranteed_no_drop"))
                || self.object_holds(of, self_ty, args) == Some(true)
            {
                return Ok(true);
            }
        }
        self.callable_obligation_holds(of, self_ty, args)
    }

    fn callable_obligation_holds(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        if !self.is_callable_trait(of) {
            return Ok(false);
        }
        if !matches!(self.cx.types.kind(self.resolve(self_ty)), Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..)) {
            return Ok(false);
        }
        let Some((params, _)) = self.callable_sig(self_ty)? else { return Ok(false) };
        let expected = self.cx.types.tuple(&params);
        let Some(&actual) = args.first() else { return Ok(false) };
        let mark = self.mark_vars();
        if self.unifies(expected, actual) {
            self.commit_vars(mark);
            return Ok(true);
        }
        self.rollback_vars(mark);
        Ok(false)
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

    fn ensure_no_type_vars(&self) -> Result<()> {
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
        for map in [&self.out.res, &self.out.operators, &self.out.try_blocks, &self.out.box_frees] {
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
        let ty = self.resolve(ty);
        if !seen.insert(ty) {
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

    pub(super) fn zonk_res(&mut self, res: Res) -> Result<Res> {
        match res {
            Res::Fn(item, substs) => {
                let mapped = self.map_substs(substs, Self::zonk);
                self.trait_call(item, mapped)
            }
            Res::Ctor(item, variant, args) => Ok(Res::Ctor(item, variant, self.map_substs(args, Self::zonk))),
            Res::Const(value) => Ok(Res::Const(self.zonk(value))),
            other => Ok(other),
        }
    }
}

/// Whether `item` is a non-generic synchronous `fn(i64) -> i64` suitable as a JIT test case.
pub fn is_case_sig(cx: &mut Context<'_>, item: ItemRef) -> bool {
    let Ok(sig) = cx.signature(item) else { return false };
    if sig.params.len() != 1 || sig.ret.is_none() || sig.receiver.is_some() || sig.is_async || sig.variadic || sig.opaque || scope::type_params(&sig.generics).next().is_some() {
        return false;
    }
    let i64_ty = cx.types.prim(Primitive::I64);
    matches!(
        detached(cx, item, |infer| {
            let (params, ret) = infer.fn_types(item, &[])?;
            Ok(params.as_slice() == [i64_ty] && infer.resolve(ret) == i64_ty)
        }),
        Ok(true)
    )
}
