mod settle;

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
    OUT_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default()
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
        if !self.cx.types.has_var_flag(ty) {
            return ty;
        }
        let ty = self.resolve(ty);
        if !self.cx.types.has_var_flag(ty) {
            return ty;
        }
        match self.cx.types.kind(ty) {
            Kind::IntVar(_) => {
                if fall_back {
                    self.cx.types.prim(Primitive::I32)
                } else {
                    ty
                }
            }
            Kind::FloatVar(_) => {
                if fall_back {
                    self.cx.types.prim(Primitive::F64)
                } else {
                    ty
                }
            }
            Kind::Var(_) => {
                if fall_back {
                    self.cx.types.unit()
                } else {
                    ty
                }
            }
            _ => {
                let Ok(mapped) = super::super::ty::try_map_children::<_, std::convert::Infallible>(
                    self,
                    ty,
                    |this| &this.cx.types,
                    &mut |this, item| Ok(this.fill(item, fall_back)),
                );
                mapped
            }
        }
    }

    pub(super) fn map_substs(
        &mut self,
        id: super::super::substs::Substs,
        mut f: impl FnMut(&mut Self, Ty) -> Ty,
    ) -> super::super::substs::Substs {
        let Ok(mapped) = super::super::ty::try_map_substs::<_, std::convert::Infallible>(
            self,
            id,
            |this| &this.cx.types,
            &mut |this, ty| Ok(f(this, ty)),
        );
        mapped
    }

    pub(super) fn finish(mut self) -> Result<Typed> {
        self.settle_all()?;
        self.ensure_no_type_vars()?;
        self.pack_typed()
    }

    pub(super) fn finish_poly(
        mut self,
    ) -> std::result::Result<
        (Typed, Option<Box<super::poly::PolyState>>, bool),
        (anyhow::Error, bool),
    > {
        if let Err(err) = self.settle_all() {
            return Err((err, self.poly_fell_back()));
        }
        let clean = self.shared_here(None) && self.backlog() == 0 && !self.has_unbound_out_vars();
        let typed = match self.pack_typed() {
            Ok(typed) => typed,
            Err(err) => return Err((err, self.poly_fell_back())),
        };
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
        exprs
            .chain(locals)
            .chain(pats)
            .chain(self.out.ret)
            .any(|ty| self.has_vars_seen(ty, &mut seen))
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
        let exprs: Box<[Ty]> = raw_exprs
            .drain(..)
            .map(|ty| if ty == Ty::NONE { ty } else { self.zonk(ty) })
            .collect();
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
        Ok(Typed {
            tables: super::LazyTables::Eager(Arc::new(tables)),
            locals,
            params,
            ret,
            drops: Vec::new(),
            shape: Shape::default(),
        })
    }

    fn drain_res_field<K: Copy + Ord + std::hash::Hash>(
        &mut self,
        field: fn(&mut InferOut) -> &mut HashMap<K, Res>,
    ) -> Result<PackedMap<K, Res>> {
        self.drain_mapped(field, Self::zonk_res)
    }

    fn drain_mapped<K: Copy + Ord + std::hash::Hash, V>(
        &mut self,
        field: fn(&mut InferOut) -> &mut HashMap<K, V>,
        map_fn: fn(&mut Self, V) -> Result<V>,
    ) -> Result<PackedMap<K, V>> {
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
        Ok(For {
            into_iter: self.zonk_res(each.into_iter)?,
            next: self.zonk_res(each.next)?,
            iter: self.zonk(each.iter),
            option: self.zonk(each.option),
            some: each.some,
        })
    }

    fn zonk_await(&mut self, each: Awaited) -> Result<Awaited> {
        Ok(Awaited {
            into_future: self.zonk_res(each.into_future)?,
            future: self.zonk(each.future),
            poll: self.zonk_res(each.poll)?,
            polled: self.zonk(each.polled),
        })
    }

    fn zonk_try(&mut self, each: Tried) -> Result<Tried> {
        Ok(Tried {
            branch: self.zonk_res(each.branch)?,
            from_residual: self.zonk_res(each.from_residual)?,
            flow: self.zonk(each.flow),
            ..each
        })
    }

    pub(super) fn zonk_res(&mut self, res: Res) -> Result<Res> {
        match res {
            Res::Fn(item, substs) => {
                let mapped = self.map_substs(substs, Self::zonk);
                self.trait_call(item, mapped)
            }
            Res::Ctor(item, variant, args) => {
                Ok(Res::Ctor(item, variant, self.map_substs(args, Self::zonk)))
            }
            Res::Const(value) => Ok(Res::Const(self.zonk(value))),
            other => Ok(other),
        }
    }

    pub(super) fn init_poly_for_item(&mut self, item: ItemRef, param_count: u16) -> Result<()> {
        let args: Vec<Ty> = (0..param_count)
            .map(|idx| self.cx.types.intern(Kind::Param(idx)))
            .collect();
        let substs = self.cx.types.intern_substs(&args);
        let root = Some(item.body_owner());
        (self.root, self.instance) = (root, Some((item, substs)));
        self.poly = Some(Box::new(super::poly::PolyState::default()));
        let mut frame = self.frame(item, &args)?;
        (frame.src, frame.root) = (self.cx.body_src(item), root);
        self.swap_frame(frame);
        self.init_poly_env(item)
    }
}

/// Whether `ty` implements `Copy` in the context of `item`.
pub fn is_copy(cx: &mut Context<'_>, item: ItemRef, ty: Ty) -> Result<bool> {
    detached(cx, item, |infer| {
        if infer.cx.types.has_params(ty)
            && let Some(count) = super::super::summary::expected_param_count(infer.cx, item)
            && count > 0
            && count <= u16::MAX as usize
        {
            let _ = infer.init_poly_for_item(item, count as u16);
        }
        infer.is_copy(ty)
    })
}

/// Whether `item` is a non-generic synchronous `fn(i64) -> i64` suitable as a JIT test case.
#[cfg(feature = "diff-harness")]
pub fn is_case_sig(cx: &mut Context<'_>, item: ItemRef) -> bool {
    let Ok(sig) = cx.signature(item) else {
        return false;
    };
    if sig.params.len() != 1
        || sig.ret.is_none()
        || sig.receiver.is_some()
        || sig.is_async
        || sig.variadic
        || sig.opaque
        || scope::type_params(&sig.generics).next().is_some()
    {
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

/// Whether `item` is a non-generic synchronous zero-argument `#[test]` function (`fn()` or `fn() -> T`), returning `Some(returns_unit)`.
pub fn test_sig(cx: &mut Context<'_>, item: ItemRef) -> Option<bool> {
    let sig = cx.signature(item).ok()?;
    if !sig.params.is_empty()
        || sig.receiver.is_some()
        || sig.is_async
        || sig.variadic
        || sig.opaque
        || scope::type_params(&sig.generics).next().is_some()
    {
        return None;
    }
    let unit = cx.types.unit();
    detached(cx, item, |infer| {
        let (params, ret) = infer.fn_types(item, &[])?;
        Ok((params.is_empty()).then_some(infer.resolve(ret) == unit))
    })
    .ok()
    .flatten()
}

/// Whether `item` is a non-generic function suitable as a root for `haste build`.
pub fn is_concrete_fn(cx: &mut Context<'_>, item: ItemRef) -> bool {
    let Ok(sig) = cx.signature(item) else {
        return false;
    };
    !sig.variadic && super::super::summary::expected_param_count(cx, item) == Some(0)
}
