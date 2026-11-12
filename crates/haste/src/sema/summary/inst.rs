use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap as HashMap;

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;

use super::super::infer::{Awaited, BodyTables, Closure, Coercion, For, Res, SiteKey, Tried};
use super::super::substs::{PackedMap, PackedSet, ProjId, Substs};
use super::super::ty::{Kind, Ty, Types};
use super::{CallOverride, PolySummary};

/// Compact per-instance instantiation state that materializes full `BodyTables` only on demand.
#[derive(Clone, Debug)]
pub struct PolyInst {
    pub args: Substs,
    pub summary: Arc<PolySummary>,
    pub projs: Box<[(ProjId, Ty)]>,
    pub resolved_calls: Box<[(SiteKey, ItemRef, Substs)]>,
    pub builtin_ops: Box<[ExprId]>,
    pub call_overrides: Box<[CallOverride]>,
    pub offset_overrides: Box<[(ExprId, u64)]>,
    pub(super) materialized: OnceLock<Arc<BodyTables>>,
}

impl PolyInst {
    /// Returns already-materialized concrete body tables, panicking if `ensure_tables` was not called.
    pub fn materialized(&self) -> &BodyTables {
        let Some(tables) = self.materialized.get() else { unreachable!() };
        tables
    }

    /// Materializes concrete body tables on first access and caches them in `OnceLock`.
    pub fn ensure_tables(&self, types: &Types) -> &BodyTables {
        self.materialized.get_or_init(|| Arc::new(self.materialize(types)))
    }

    /// Iterates every concrete and resolved call edge without materializing `BodyTables`.
    pub fn calls(&self) -> impl Iterator<Item = (ItemRef, Substs)> + '_ {
        self.summary.fixed_calls.iter().copied().chain(self.resolved_calls.iter().map(|&(_, i, s)| (i, s)))
    }

    /// Substitutes this instance's concrete arguments and normalized projections into `ty`.
    pub fn sub_ty(&self, types: &Types, ty: Ty) -> Ty {
        if ty == Ty::NONE || !types.has_params(ty) {
            return ty;
        }
        match *types.kind(ty) {
            Kind::Param(idx) => types.substs(self.args)[idx as usize],
            Kind::Proj(id) => self.proj_ty(id),
            ref kind => {
                let pools = types.pools();
                types.intern(kind.map_pool(&pools, |child| self.sub_ty(types, child)))
            }
        }
    }

    /// Substitutes this instance's concrete arguments and normalized projections into `substs`.
    pub fn sub_substs(&self, types: &Types, substs: Substs) -> Substs {
        if !types.substs_have_params(substs) {
            return substs;
        }
        types.pool().map(substs, &mut |ty| self.sub_ty(types, ty))
    }

    /// The concrete type of `expr` in this instance without materializing `BodyTables`.
    pub fn expr_ty(&self, types: &Types, expr: ExprId) -> Ty {
        self.sub_ty(types, self.summary.template.exprs[expr.0 as usize])
    }

    /// The concrete type of `pat` in this instance without materializing `BodyTables`.
    pub fn pat_ty(&self, types: &Types, pat: PatId) -> Option<Ty> {
        self.summary.template.pats.get(&pat).map(|&ty| self.sub_ty(types, ty))
    }

    /// The autoderef count at `expr` in this instance without materializing `BodyTables`.
    pub fn autoderefs_at(&self, expr: ExprId) -> u32 {
        if let Some(ov) = self.call_overrides.iter().find(|c| c.expr == expr) {
            return ov.autoderefs.unwrap_or(0);
        }
        self.summary.template.autoderefs.get(&expr).copied().unwrap_or(0)
    }

    /// Whether `expr` is an overloaded callable invocation in this instance.
    pub fn is_overloaded_call(&self, expr: ExprId) -> bool {
        self.call_overrides.iter().any(|c| c.expr == expr && c.overloaded) || self.summary.template.overloaded.contains(&expr)
    }

    /// The resolved callee `(ItemRef, Substs)` at `key` in this instance without materializing `BodyTables`.
    pub fn site_fn(&self, key: SiteKey) -> Option<(ItemRef, Substs)> {
        if let Ok(idx) = self.resolved_calls.binary_search_by_key(&key, |&(k, ..)| k) {
            let (_, item, substs) = self.resolved_calls[idx];
            return Some((item, substs));
        }
        if matches!(key, SiteKey::Operator(_)) {
            return None;
        }
        if let SiteKey::ExprRes(expr) = key
            && self.call_overrides.iter().any(|c| c.expr == expr)
        {
            return None;
        }
        match self.summary.template.site_res(key)? {
            Res::Fn(item, substs) => Some((item, substs)),
            _ => None,
        }
    }

    /// The concrete `Res` at `key` in this instance without materializing `BodyTables`.
    pub fn site_res(&self, types: &Types, key: SiteKey) -> Option<Res> {
        if let Some((item, substs)) = self.site_fn(key) {
            return Some(Res::Fn(item, substs));
        }
        if matches!(key, SiteKey::Operator(_)) {
            return None;
        }
        if let SiteKey::ExprRes(expr) = key
            && let Some(ov) = self.call_overrides.iter().find(|c| c.expr == expr)
        {
            return ov.res;
        }
        match self.summary.template.site_res(key)? {
            Res::Local(l) => Some(Res::Local(l)),
            Res::Const(ty) => Some(Res::Const(self.sub_ty(types, ty))),
            Res::Ctor(i, v, s) => Some(Res::Ctor(i, v, self.sub_substs(types, s))),
            Res::Fn(..) => None,
        }
    }

    fn proj_ty(&self, id: ProjId) -> Ty {
        let Ok(idx) = self.projs.binary_search_by_key(&id.0, |&(p, _)| p.0) else { unreachable!() };
        self.projs[idx].1
    }

    fn sub_ty_cached(&self, types: &Types, cache: &mut HashMap<Ty, Ty>, ty: Ty) -> Ty {
        if ty == Ty::NONE || !types.has_params(ty) {
            return ty;
        }
        if let Some(&found) = cache.get(&ty) {
            return found;
        }
        let mapped = match *types.kind(ty) {
            Kind::Param(idx) => types.substs(self.args)[idx as usize],
            Kind::Proj(id) => self.proj_ty(id),
            ref kind => {
                let pools = types.pools();
                types.intern(kind.map_pool(&pools, |child| self.sub_ty_cached(types, cache, child)))
            }
        };
        cache.insert(ty, mapped);
        mapped
    }

    fn sub_res_cached(&self, types: &Types, cache: &mut HashMap<Ty, Ty>, key: SiteKey, r: &Res) -> Res {
        match *r {
            Res::Local(l) => Res::Local(l),
            Res::Const(ty) => Res::Const(self.sub_ty_cached(types, cache, ty)),
            Res::Ctor(i, v, s) => Res::Ctor(i, v, types.pool().map(s, &mut |t| self.sub_ty_cached(types, cache, t))),
            Res::Fn(i, s) if !types.substs_have_params(s) => Res::Fn(i, s),
            Res::Fn(..) => {
                let Some((ri, rs)) = self.site_fn(key) else { unreachable!() };
                Res::Fn(ri, rs)
            }
        }
    }

    /// Materializes a complete concrete `BodyTables` from the shared polymorphic template and hole solutions.
    pub fn materialize(&self, types: &Types) -> BodyTables {
        let tpl = &self.summary.template;
        let mut cache = HashMap::default();
        let exprs = tpl.exprs.iter().map(|&ty| self.sub_ty_cached(types, &mut cache, ty)).collect();
        let pats = PackedMap::from_sorted(tpl.pats.as_slice().iter().map(|&(k, ty)| (k, self.sub_ty_cached(types, &mut cache, ty))).collect());
        let pat_res = PackedMap::from_sorted(tpl.pat_res.as_slice().iter().map(|&(k, ref r)| (k, self.sub_res_cached(types, &mut cache, SiteKey::PatRes(k), r))).collect());
        let derefs = PackedMap::from_sorted(tpl.derefs.as_slice().iter().map(|&(k, ref r)| (k, self.sub_res_cached(types, &mut cache, SiteKey::Deref(k.0, k.1), r))).collect());
        let try_blocks = PackedMap::from_sorted(tpl.try_blocks.as_slice().iter().map(|&(k, ref r)| (k, self.sub_res_cached(types, &mut cache, SiteKey::TryBlock(k), r))).collect());
        let res = self.materialize_res(types, &mut cache, &tpl.res);
        let operators = self.materialize_operators(types, &mut cache, &tpl.operators);
        let fors = PackedMap::from_sorted(tpl.fors.as_slice().iter().map(|&(k, ref f)| (k, For { into_iter: self.sub_res_cached(types, &mut cache, SiteKey::ForInto(k), &f.into_iter), next: self.sub_res_cached(types, &mut cache, SiteKey::ForNext(k), &f.next), iter: self.sub_ty_cached(types, &mut cache, f.iter), option: self.sub_ty_cached(types, &mut cache, f.option), some: f.some })).collect());
        let tries = PackedMap::from_sorted(tpl.tries.as_slice().iter().map(|&(k, ref t)| (k, Tried { branch: self.sub_res_cached(types, &mut cache, SiteKey::TryBranch(k), &t.branch), from_residual: self.sub_res_cached(types, &mut cache, SiteKey::TryFrom(k), &t.from_residual), flow: self.sub_ty_cached(types, &mut cache, t.flow), ..*t })).collect());
        let awaits = PackedMap::from_sorted(tpl.awaits.as_slice().iter().map(|&(k, ref a)| (k, Awaited { into_future: self.sub_res_cached(types, &mut cache, SiteKey::AwaitInto(k), &a.into_future), future: self.sub_ty_cached(types, &mut cache, a.future), poll: self.sub_res_cached(types, &mut cache, SiteKey::AwaitPoll(k), &a.poll), polled: self.sub_ty_cached(types, &mut cache, a.polled) })).collect());
        let coercions = PackedMap::from_sorted(tpl.coercions.as_slice().iter().map(|&(k, ref c)| (k, self.materialize_coercion(types, &mut cache, k, c))).collect());
        let closures = PackedMap::from_sorted(tpl.closures.as_slice().iter().map(|&(k, ref c)| (k, Closure { ret: self.sub_ty_cached(types, &mut cache, c.ret), ..c.clone() })).collect());
        let (autoderefs, overloaded, box_frees) = self.materialize_call_tables(types, &mut cache, tpl);
        let offsets = if self.offset_overrides.is_empty() { tpl.offsets.clone() } else { let mut pairs = tpl.offsets.as_slice().to_vec(); pairs.extend_from_slice(&self.offset_overrides); PackedMap::pack(pairs) };
        BodyTables { syntax: Arc::clone(&tpl.syntax), exprs, res, autoderefs, derefs, operators, coercions, offsets, pat_res, overloaded, pats, fors, tries, awaits, closures, try_blocks, box_frees }
    }

    fn materialize_res(&self, types: &Types, cache: &mut HashMap<Ty, Ty>, tpl_res: &PackedMap<ExprId, Res>) -> PackedMap<ExprId, Res> {
        if self.call_overrides.is_empty() {
            return PackedMap::from_sorted(tpl_res.as_slice().iter().map(|&(k, ref r)| (k, self.sub_res_cached(types, cache, SiteKey::ExprRes(k), r))).collect());
        }
        let mut pairs = Vec::with_capacity(tpl_res.len() + self.call_overrides.len());
        for &(expr, ref r) in tpl_res.as_slice() {
            if let Some(ov) = self.call_overrides.iter().find(|c| c.expr == expr) {
                if let Some(new_res) = ov.res {
                    pairs.push((expr, new_res));
                }
            } else {
                pairs.push((expr, self.sub_res_cached(types, cache, SiteKey::ExprRes(expr), r)));
            }
        }
        for ov in &self.call_overrides {
            if !tpl_res.contains_key(&ov.expr)
                && let Some(new_res) = ov.res
            {
                pairs.push((ov.expr, new_res));
            }
        }
        PackedMap::pack(pairs)
    }

    fn materialize_operators(&self, types: &Types, cache: &mut HashMap<Ty, Ty>, tpl_ops: &PackedMap<ExprId, Res>) -> PackedMap<ExprId, Res> {
        if tpl_ops.is_empty() {
            return PackedMap::default();
        }
        let mut pairs = Vec::with_capacity(tpl_ops.len());
        for &(expr, ref r) in tpl_ops.as_slice() {
            if self.builtin_ops.binary_search(&expr).is_ok() {
                continue;
            }
            let mapped = match self.site_fn(SiteKey::Operator(expr)) {
                Some((ri, rs)) => Res::Fn(ri, rs),
                None => self.sub_res_cached(types, cache, SiteKey::Operator(expr), r),
            };
            pairs.push((expr, mapped));
        }
        PackedMap::from_sorted(pairs)
    }

    fn materialize_coercion(&self, types: &Types, cache: &mut HashMap<Ty, Ty>, id: ExprId, coercion: &Coercion) -> Coercion {
        match coercion {
            Coercion::Len(len) => Coercion::Len(self.sub_ty_cached(types, cache, *len)),
            Coercion::Autos | Coercion::Upcast(_) | Coercion::Vtable(_) => coercion.clone(),
            Coercion::Reify(r) => Coercion::Reify(self.sub_res_cached(types, cache, SiteKey::CoercionReify(id), r)),
            Coercion::Deref(steps, to) => {
                let mapped = steps.iter().enumerate().map(|(idx, (ty, r))| (self.sub_ty_cached(types, cache, *ty), r.as_ref().map(|r| self.sub_res_cached(types, cache, SiteKey::CoercionDeref(id, idx as u16), r)))).collect();
                Coercion::Deref(mapped, self.sub_ty_cached(types, cache, *to))
            }
        }
    }

    fn materialize_call_tables(&self, types: &Types, cache: &mut HashMap<Ty, Ty>, tpl: &BodyTables) -> (PackedMap<ExprId, u32>, PackedSet<ExprId>, PackedMap<ExprId, Res>) {
        let mut box_frees = PackedMap::from_sorted(tpl.box_frees.as_slice().iter().map(|&(k, ref r)| (k, self.sub_res_cached(types, cache, SiteKey::BoxFree(k), r))).collect());
        if self.call_overrides.is_empty() {
            return (tpl.autoderefs.clone(), tpl.overloaded.clone(), box_frees);
        }
        let mut deref_pairs: Vec<(ExprId, u32)> = tpl.autoderefs.as_slice().iter().copied().filter(|(e, _)| !self.call_overrides.iter().any(|c| c.expr == *e)).collect();
        let mut over_set: Vec<ExprId> = tpl.overloaded.as_slice().to_vec();
        let mut box_pairs: Vec<(ExprId, Res)> = box_frees.as_slice().to_vec();
        for ov in &self.call_overrides {
            if let Some(steps) = ov.autoderefs {
                deref_pairs.push((ov.expr, steps));
            }
            if ov.overloaded {
                over_set.push(ov.expr);
            }
            if let Some(free) = ov.box_free {
                box_pairs.push((ov.expr, free));
            }
        }
        if self.call_overrides.iter().any(|c| c.box_free.is_some()) {
            box_frees = PackedMap::pack(box_pairs);
        }
        (PackedMap::pack(deref_pairs), PackedSet::pack(over_set), box_frees)
    }
}
