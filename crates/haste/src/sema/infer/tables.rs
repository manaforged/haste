use std::sync::{Arc, OnceLock};

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::sema::substs::{PackedMap, PackedSet, Substs};
use crate::sema::summary::{PolyCheck, PolyInst};
use crate::sema::ty::{Mut, Ty, Types};

use super::{Adjust, Awaited, Closure, Coercion, For, Local, Res, Shape, Tried};

/// A call or resolution table entry inside a typed body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SiteKey {
    ExprRes(ExprId),
    PatRes(PatId),
    Deref(ExprId, u32),
    Operator(ExprId),
    TryBlock(ExprId),
    BoxFree(ExprId),
    ForInto(ExprId),
    ForNext(ExprId),
    TryBranch(ExprId),
    TryFrom(ExprId),
    AwaitInto(ExprId),
    AwaitPoll(ExprId),
    CoercionReify(ExprId),
    CoercionDeref(ExprId, u16),
}

/// The instantiation-invariant syntax tables of an inferred body.
#[derive(Clone, Debug, Default)]
pub struct SyntaxTables {
    pub binds: PackedMap<PatId, u32>,
    pub adjusts: PackedMap<ExprId, Adjust>,
    pub fields: PackedMap<ExprId, u32>,
    pub inits: PackedMap<ExprId, u32>,
    pub ranges: PackedMap<ExprId, Box<[u32]>>,
    pub field_pats: PackedMap<PatId, u32>,
    pub pat_derefs: PackedMap<PatId, u32>,
    pub ref_binds: PackedMap<PatId, Mut>,
    pub or_binds: PackedSet<PatId>,
    pub const_args: PackedSet<ExprId>,
    pub unpacks: PackedMap<u32, PatId>,
}

impl SyntaxTables {
    /// A shared empty syntax table set for synthetic bodies.
    pub fn empty() -> Arc<Self> {
        static EMPTY: OnceLock<Arc<SyntaxTables>> = OnceLock::new();
        Arc::clone(EMPTY.get_or_init(|| Arc::new(Self::default())))
    }
}

/// The arena-packed expression, pattern, operator, coercion and closure tables of an inferred body.
#[derive(Clone, Debug)]
pub struct BodyTables {
    pub syntax: Arc<SyntaxTables>,
    pub exprs: Box<[Ty]>,
    pub res: PackedMap<ExprId, Res>,
    pub autoderefs: PackedMap<ExprId, u32>,
    pub derefs: PackedMap<(ExprId, u32), Res>,
    pub operators: PackedMap<ExprId, Res>,
    pub coercions: PackedMap<ExprId, Coercion>,
    pub offsets: PackedMap<ExprId, u64>,
    pub pat_res: PackedMap<PatId, Res>,
    pub overloaded: PackedSet<ExprId>,
    pub pats: PackedMap<PatId, Ty>,
    pub fors: PackedMap<ExprId, For>,
    pub tries: PackedMap<ExprId, Tried>,
    pub awaits: PackedMap<ExprId, Awaited>,
    pub closures: PackedMap<ExprId, Closure>,
    pub try_blocks: PackedMap<ExprId, Res>,
    pub box_frees: PackedMap<ExprId, Res>,
}

impl Default for BodyTables {
    fn default() -> Self {
        Self {
            syntax: SyntaxTables::empty(),
            exprs: Box::default(),
            res: PackedMap::default(),
            autoderefs: PackedMap::default(),
            derefs: PackedMap::default(),
            operators: PackedMap::default(),
            coercions: PackedMap::default(),
            offsets: PackedMap::default(),
            pat_res: PackedMap::default(),
            overloaded: PackedSet::default(),
            pats: PackedMap::default(),
            fors: PackedMap::default(),
            tries: PackedMap::default(),
            awaits: PackedMap::default(),
            closures: PackedMap::default(),
            try_blocks: PackedMap::default(),
            box_frees: PackedMap::default(),
        }
    }
}

impl BodyTables {
    /// A shared empty body table set for compiler-generated bodies.
    pub fn empty() -> Arc<Self> {
        static EMPTY: OnceLock<Arc<BodyTables>> = OnceLock::new();
        Arc::clone(EMPTY.get_or_init(|| Arc::new(Self::default())))
    }

    /// Every expression and desugared `for`/`await` temporary type that lowering may own and drop.
    pub fn owned_tys(&self) -> impl Iterator<Item = Ty> + '_ {
        self.exprs
            .iter()
            .copied()
            .chain(self.fors.values().map(|f| f.iter))
            .chain(self.awaits.values().map(|a| a.future))
    }

    /// Expression and desugared `for`/`await` temporary types owned at `expr`.
    pub fn expr_owned_tys(&self, expr: ExprId) -> impl Iterator<Item = Ty> + '_ {
        [
            Some(self.exprs[expr.0 as usize]),
            self.fors.get(&expr).map(|f| f.iter),
            self.awaits.get(&expr).map(|a| a.future),
        ]
        .into_iter()
        .flatten()
    }

    /// The resolution stored at `key` in these tables.
    pub fn site_res(&self, key: SiteKey) -> Option<Res> {
        match key {
            SiteKey::ExprRes(id) => self.res.get(&id).copied(),
            SiteKey::PatRes(id) => self.pat_res.get(&id).copied(),
            SiteKey::Deref(id, step) => self.derefs.get(&(id, step)).copied(),
            SiteKey::Operator(id) => self.operators.get(&id).copied(),
            SiteKey::TryBlock(id) => self.try_blocks.get(&id).copied(),
            SiteKey::BoxFree(id) => self.box_frees.get(&id).copied(),
            SiteKey::ForInto(id) => Some(self.fors.get(&id)?.into_iter),
            SiteKey::ForNext(id) => Some(self.fors.get(&id)?.next),
            SiteKey::TryBranch(id) => Some(self.tries.get(&id)?.branch),
            SiteKey::TryFrom(id) => Some(self.tries.get(&id)?.from_residual),
            SiteKey::AwaitInto(id) => Some(self.awaits.get(&id)?.into_future),
            SiteKey::AwaitPoll(id) => Some(self.awaits.get(&id)?.poll),
            SiteKey::CoercionReify(id) => match self.coercions.get(&id)? {
                Coercion::Reify(res) => Some(*res),
                _ => None,
            },
            SiteKey::CoercionDeref(id, idx) => match self.coercions.get(&id)? {
                Coercion::Deref(steps, _) => steps.get(idx as usize)?.1,
                _ => None,
            },
        }
    }
}

impl std::ops::Deref for BodyTables {
    type Target = SyntaxTables;

    fn deref(&self) -> &Self::Target {
        &self.syntax
    }
}

impl std::ops::DerefMut for BodyTables {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.syntax)
    }
}

/// Either eagerly materialized body tables, a lazy polymorphic instantiation, or a stripped check-only summary.
#[derive(Clone, Debug)]
pub enum LazyTables {
    Eager(Arc<BodyTables>),
    Poly(Box<PolyInst>),
    CheckOnly(Box<PolyCheck>),
}

/// The inferred types and resolutions of one function body at one set of type arguments.
#[derive(Clone, Debug)]
pub struct Typed {
    pub tables: LazyTables,
    pub locals: Vec<Local>,
    pub params: Vec<u32>,
    pub ret: Option<Ty>,
    pub drops: Vec<Res>,
    pub shape: Shape,
}

impl Default for Typed {
    fn default() -> Self {
        Self { tables: LazyTables::Eager(BodyTables::empty()), locals: Vec::new(), params: Vec::new(), ret: None, drops: Vec::new(), shape: Shape::default() }
    }
}

impl std::ops::Deref for Typed {
    type Target = BodyTables;

    fn deref(&self) -> &Self::Target {
        match &self.tables {
            LazyTables::Eager(tables) => tables,
            LazyTables::Poly(poly) => poly.materialized(),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no BodyTables"),
        }
    }
}

impl std::ops::DerefMut for Typed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let LazyTables::Eager(tables) = &mut self.tables else { unreachable!() };
        Arc::make_mut(tables)
    }
}

fn coercion_calls(coercion: &Coercion) -> Box<dyn Iterator<Item = &Res> + '_> {
    match coercion {
        Coercion::Vtable(vtable) => Box::new(vtable.calls().into_iter()),
        Coercion::Reify(res) => Box::new(std::iter::once(res)),
        Coercion::Len(_) | Coercion::Autos | Coercion::Upcast(_) => Box::new(std::iter::empty()),
        Coercion::Deref(steps, _) => Box::new(steps.iter().filter_map(|(_, res)| res.as_ref())),
    }
}

fn shape_calls(shape: &Shape) -> Box<dyn Iterator<Item = &Res> + '_> {
    match shape {
        Shape::Glue(glue) => Box::new(glue.drop.iter().chain(&glue.parts).chain(&glue.resume)),
        Shape::Untuple(res) => Box::new(std::iter::once(res)),
        Shape::Clone(calls) => Box::new(calls.iter()),
        _ => Box::new(std::iter::empty()),
    }
}

fn res_fn(res: &Res) -> Option<(ItemRef, Substs)> {
    match *res {
        Res::Fn(item, substs) => Some((item, substs)),
        _ => None,
    }
}

impl Typed {
    /// The return type, or `Ty::NONE` for a body that has none.
    pub fn ret_ty(&self) -> Ty {
        self.ret.unwrap_or(Ty::NONE)
    }

    /// Materializes and returns the concrete body tables of this instance.
    pub fn ensure_tables(&self, types: &Types) -> &BodyTables {
        match &self.tables {
            LazyTables::Eager(tables) => tables,
            LazyTables::Poly(poly) => poly.ensure_tables(types),
            LazyTables::CheckOnly(_) => panic!("check-only Typed cannot materialize BodyTables"),
        }
    }

    /// Returns a borrowed or transiently materialized view of this instance without pinning `OnceLock` tables.
    pub fn scoped(&self, types: &Types) -> std::borrow::Cow<'_, Self> {
        match &self.tables {
            LazyTables::Eager(_) => std::borrow::Cow::Borrowed(self),
            LazyTables::Poly(poly) => match poly.ephemeral_tables(types) {
                Some(tables) => std::borrow::Cow::Owned(Self {
                    tables: LazyTables::Eager(tables),
                    locals: self.locals.clone(),
                    params: self.params.clone(),
                    ret: self.ret,
                    drops: self.drops.clone(),
                    shape: self.shape.clone(),
                }),
                None => std::borrow::Cow::Borrowed(self),
            },
            LazyTables::CheckOnly(_) => panic!("check-only Typed cannot materialize scoped BodyTables"),
        }
    }

    /// The instantiation-invariant syntax tables of this body.
    pub fn syntax(&self) -> &SyntaxTables {
        match &self.tables {
            LazyTables::Eager(tables) => &tables.syntax,
            LazyTables::Poly(poly) => &poly.summary.template.syntax,
            LazyTables::CheckOnly(check) => &check.summary.template.syntax,
        }
    }

    /// Whether this body defines any closures or coroutines.
    pub fn has_closures(&self) -> bool {
        match &self.tables {
            LazyTables::Eager(tables) => !tables.closures.is_empty(),
            LazyTables::Poly(poly) => !poly.summary.template.closures.is_empty(),
            LazyTables::CheckOnly(check) => !check.summary.template.closures.is_empty(),
        }
    }

    /// Whether this body contains any `.await` suspension points.
    pub fn has_awaits(&self) -> bool {
        match &self.tables {
            LazyTables::Eager(tables) => !tables.awaits.is_empty(),
            LazyTables::Poly(poly) => !poly.summary.template.awaits.is_empty(),
            LazyTables::CheckOnly(check) => !check.summary.template.awaits.is_empty(),
        }
    }

    /// The inferred type of expression `expr`, or `Ty::NONE` if absent.
    pub fn expr_ty(&self, types: &Types, expr: ExprId) -> Ty {
        match &self.tables {
            LazyTables::Eager(tables) => tables.exprs[expr.0 as usize],
            LazyTables::Poly(poly) => poly.expr_ty(types, expr),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no expr_ty"),
        }
    }

    /// The inferred type of pattern `pat`, if present.
    pub fn pat_ty(&self, types: &Types, pat: PatId) -> Option<Ty> {
        match &self.tables {
            LazyTables::Eager(tables) => tables.pats.get(&pat).copied(),
            LazyTables::Poly(poly) => poly.pat_ty(types, pat),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no pat_ty"),
        }
    }

    /// The resolution at call or path site `key`, if any.
    pub fn site_res(&self, types: &Types, key: SiteKey) -> Option<Res> {
        match &self.tables {
            LazyTables::Eager(tables) => tables.site_res(key),
            LazyTables::Poly(poly) => poly.site_res(types, key),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no site_res"),
        }
    }

    /// The function item and type arguments at call site `key`, if resolved to a function.
    pub fn site_fn(&self, key: SiteKey) -> Option<(ItemRef, Substs)> {
        match &self.tables {
            LazyTables::Eager(tables) => match tables.site_res(key)? {
                Res::Fn(item, substs) => Some((item, substs)),
                _ => None,
            },
            LazyTables::Poly(poly) => poly.site_fn(key),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no site_fn"),
        }
    }

    /// The number of autoderef steps recorded for expression `expr`.
    pub fn autoderefs_at(&self, expr: ExprId) -> u32 {
        match &self.tables {
            LazyTables::Eager(tables) => tables.autoderefs.get(&expr).copied().unwrap_or(0),
            LazyTables::Poly(poly) => poly.autoderefs_at(expr),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no autoderefs"),
        }
    }

    /// The iterator type of `for` expression `expr`, if present.
    pub fn for_iter(&self, types: &Types, expr: ExprId) -> Option<Ty> {
        match &self.tables {
            LazyTables::Eager(tables) => tables.fors.get(&expr).map(|f| f.iter),
            LazyTables::Poly(poly) => poly.summary.template.fors.get(&expr).map(|f| poly.sub_ty(types, f.iter)),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no for_iter"),
        }
    }

    /// The future type of `.await` expression `expr`, if present.
    pub fn await_future(&self, types: &Types, expr: ExprId) -> Option<Ty> {
        match &self.tables {
            LazyTables::Eager(tables) => tables.awaits.get(&expr).map(|a| a.future),
            LazyTables::Poly(poly) => poly.summary.template.awaits.get(&expr).map(|a| poly.sub_ty(types, a.future)),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no await_future"),
        }
    }

    /// Whether expression `expr` is an `.await` site.
    pub fn has_await_at(&self, expr: ExprId) -> bool {
        match &self.tables {
            LazyTables::Eager(tables) => tables.awaits.contains_key(&expr),
            LazyTables::Poly(poly) => poly.summary.template.awaits.contains_key(&expr),
            LazyTables::CheckOnly(check) => check.summary.template.awaits.contains_key(&expr),
        }
    }

    /// Whether expression `expr` is an overloaded operator or `Fn*` call.
    pub fn is_overloaded_call(&self, expr: ExprId) -> bool {
        match &self.tables {
            LazyTables::Eager(tables) => tables.overloaded.contains(&expr),
            LazyTables::Poly(poly) => poly.is_overloaded_call(expr),
            LazyTables::CheckOnly(_) => panic!("check-only Typed has no overloaded call table"),
        }
    }

    /// Every function the body calls, with its interned type arguments.
    pub fn calls(&self) -> Box<dyn Iterator<Item = (ItemRef, Substs)> + '_> {
        let drops = self.drops.iter().filter_map(res_fn);
        let shaped = shape_calls(&self.shape).filter_map(res_fn);
        match &self.tables {
            LazyTables::Eager(t) => {
                let loops = t.fors.values().flat_map(|each| [&each.into_iter, &each.next]);
                let tries = t.tries.values().flat_map(|each| [&each.branch, &each.from_residual]);
                let awaits = t.awaits.values().flat_map(|each| [&each.into_future, &each.poll]);
                let coerced = t.coercions.values().flat_map(coercion_calls);
                let direct = t.res.values().chain(t.pat_res.values()).chain(t.derefs.values()).chain(t.operators.values()).chain(t.try_blocks.values()).chain(t.box_frees.values());
                Box::new(direct.chain(loops).chain(tries).chain(awaits).chain(coerced).filter_map(res_fn).chain(drops).chain(shaped))
            }
            LazyTables::Poly(poly) => Box::new(poly.calls().chain(drops).chain(shaped)),
            LazyTables::CheckOnly(check) => Box::new(check.calls().chain(drops).chain(shaped)),
        }
    }
}
