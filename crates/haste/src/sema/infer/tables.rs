use std::sync::{Arc, OnceLock};

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::sema::substs::{PackedMap, PackedSet, Substs};
use crate::sema::summary::PolyInst;
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

/// Either eagerly materialized body tables or a lazy polymorphic instantiation.
#[derive(Clone, Debug)]
pub enum LazyTables {
    Eager(Arc<BodyTables>),
    Poly(Box<PolyInst>),
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
        }
    }

    /// The instantiation-invariant syntax tables of this body.
    pub fn syntax(&self) -> &SyntaxTables {
        match &self.tables {
            LazyTables::Eager(tables) => &tables.syntax,
            LazyTables::Poly(poly) => &poly.summary.template.syntax,
        }
    }

    /// Whether this body defines any closures or coroutines.
    pub fn has_closures(&self) -> bool {
        match &self.tables {
            LazyTables::Eager(tables) => !tables.closures.is_empty(),
            LazyTables::Poly(poly) => !poly.summary.template.closures.is_empty(),
        }
    }

    /// Whether this body contains any `.await` suspension points.
    pub fn has_awaits(&self) -> bool {
        match &self.tables {
            LazyTables::Eager(tables) => !tables.awaits.is_empty(),
            LazyTables::Poly(poly) => !poly.summary.template.awaits.is_empty(),
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
        }
    }
}
