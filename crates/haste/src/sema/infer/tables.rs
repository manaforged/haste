use std::sync::{Arc, OnceLock};

use crate::body::{ExprId, PatId};
use crate::names::ItemRef;
use crate::sema::substs::{PackedMap, PackedSet, Substs};
use crate::sema::ty::{Mut, Ty};

use super::{Adjust, Awaited, Closure, Coercion, For, Local, Res, Shape, Tried};

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

/// The inferred types and resolutions of one function body at one set of type arguments.
#[derive(Clone, Debug)]
pub struct Typed {
    pub tables: Arc<BodyTables>,
    pub locals: Vec<Local>,
    pub params: Vec<u32>,
    pub ret: Option<Ty>,
    pub drops: Vec<Res>,
    pub shape: Shape,
}

impl Default for Typed {
    fn default() -> Self {
        Self { tables: BodyTables::empty(), locals: Vec::new(), params: Vec::new(), ret: None, drops: Vec::new(), shape: Shape::default() }
    }
}

impl std::ops::Deref for Typed {
    type Target = BodyTables;

    fn deref(&self) -> &Self::Target {
        &self.tables
    }
}

impl std::ops::DerefMut for Typed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.tables)
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

impl Typed {
    /// The return type, or `Ty::NONE` for a body that has none.
    pub fn ret_ty(&self) -> Ty {
        self.ret.unwrap_or(Ty::NONE)
    }

    /// Every function the body calls, with its interned type arguments.
    pub fn calls(&self) -> impl Iterator<Item = (ItemRef, Substs)> + '_ {
        let loops = self.fors.values().flat_map(|each| [&each.into_iter, &each.next]);
        let tries = self.tries.values().flat_map(|each| [&each.branch, &each.from_residual]);
        let awaits = self.awaits.values().flat_map(|each| [&each.into_future, &each.poll]);
        let coerced = self.coercions.values().flat_map(coercion_calls);
        let shaped = shape_calls(&self.shape);
        let direct = self.res.values().chain(self.pat_res.values()).chain(self.derefs.values()).chain(self.operators.values()).chain(self.try_blocks.values()).chain(self.box_frees.values());
        direct.chain(loops).chain(tries).chain(awaits).chain(&self.drops).chain(shaped).chain(coerced).filter_map(|res| match res {
            Res::Fn(item, substs) => Some((*item, *substs)),
            _ => None,
        })
    }
}
