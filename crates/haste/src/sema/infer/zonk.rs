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
        self.settle_projections()?;
        self.settle_opaques(true)?;
        self.settle_awaits()?;
        self.settle_operators(false)?;
        self.settle_obligations()?;
        self.fall_back_numbers()?;
        self.settle_obligations()?;
        self.settle_operators(true)?;
        self.settle_projections()?;
        if self.poly.is_some() {
            self.settle_obligations()?;
        }
        self.settle_captures()
    }

    pub(super) fn finish(mut self) -> Result<Typed> {
        self.settle_all()?;
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
        Ok(Typed { tables: Arc::new(tables), locals, params, ret, drops: Vec::new(), shape: Shape::default() })
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
