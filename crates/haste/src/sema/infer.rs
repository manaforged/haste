mod adt;
mod assoc;
mod clone;
mod consts;
mod array;
mod closure;
mod coerce;
mod deref;
mod expr;
mod matching;
mod looping;
mod method;
mod object;
mod opaque;
mod pat;
mod path;
mod pattern;
mod project;
mod scope;
mod drop;
mod select;
mod stmt;
mod zonk;

pub use adt::Fields;
pub use clone::cloned_tuple;
pub use closure::{By, Capture, Closure, ClosureBody, ClosureInfo, by_reference, register};
pub use drop::{Glue, drop_fn, has_drop_impl, needs_drop};
pub use looping::{For, Tried};
pub use method::Adjust;
pub use coerce::Coercion;
pub use consts::{Anon, Arena};
pub use object::{ALIGN_SLOT, SIZE_SLOT, Vtable, virtual_slot};
pub use pat::{position, rest_split};
pub use path::{parse_int, segments};
pub use scope::{Form, type_params};
pub use select::header_bound_traits;

use std::sync::{Arc, OnceLock};

use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, bail};

use crate::body::sig::Signature;
use crate::body::{Body, BlockKind, Expr, ExprId, PatId};
use crate::names::{ItemRef, ModRef, Primitive};
use crate::outline::source::Source;

use super::Context;
use super::substs::{PackedMap, PackedSet, Substs};
use super::ty::{Kind, Mut, Ty, Types, int_info};

/// What a path expression names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Res {
    Local(u32),
    Fn(ItemRef, Substs),
    Ctor(ItemRef, u32, Substs),
    Const(Ty),
}

impl Res {
    /// Builds `Res::Fn` after interning `args` in `types`.
    pub fn fn_of(types: &Types, item: ItemRef, args: &[Ty]) -> Self {
        Res::Fn(item, types.intern_substs(args))
    }
}

/// One local variable of a body.
#[derive(Clone, Copy, Debug)]
pub struct Local {
    pub ty: Ty,
}

/// What the code of an instance is: a body, drop glue, a closure's body, a call through the vtable slot of a trait object, a constructor, or an anonymous constant.
#[derive(Clone, Debug, Default)]
pub enum Shape {
    #[default]
    Body,
    Glue(Glue),
    Closure(ClosureBody),
    Virtual(u32),
    Untuple(Res),
    Ctor(u32),
    Clone(Vec<Res>),
    Anon(Anon),
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

pub(super) type Name<'a> = (&'a str, u32);

/// The inference of one body.
pub struct Infer<'c, 'a> {
    cx: &'c mut Context<'a>,
    body: &'c Body,
    sig: &'c Signature,
    src: &'a Source,
    root: Option<ItemRef>,
    instance: Option<(ItemRef, Substs)>,
    home: ModRef,
    generics: Vec<(&'a str, Ty)>,
    self_ty: Option<Ty>,
    block: Option<super::impls::ImplRef>,
    frame_item: Option<ItemRef>,
    vars: Vec<Option<Ty>>,
    metadata: Vec<(Ty, Ty)>,
    obligations: Vec<(ItemRef, Vec<Ty>)>,
    projections: Vec<project::Projection>,
    opaques: Vec<opaque::Opaque>,
    operands: Vec<(ExprId, crate::body::BinOp, (Ty, Ty), Ty)>,
    try_blocks: Vec<(ExprId, Ty)>,
    awaits: Vec<(ExprId, Ty)>,
    ret_sig: Option<Ty>,
    out: zonk::InferOut,
    scopes: Vec<(Name<'a>, u32)>,
    items: Vec<(&'a str, ItemRef)>,
    loops: Vec<Breaks>,
    uses: Vec<closure::Captures>,
    unsure: Vec<(ExprId, closure::Captures, bool, Vec<u32>)>,
    diverges: bool,
    hidden: Vec<(u32, Ty)>,
    const_args: Vec<(String, (u128, (u32, bool)))>,
    selecting: Vec<String>,
    frames: usize,
    evaluating: Vec<ItemRef>,
    imports: HashMap<(ItemRef, Option<crate::names::Sym>), Vec<crate::names::PerNs>>,
}

impl Drop for Infer<'_, '_> {
    fn drop(&mut self) {
        zonk::recycle_out(&mut self.out);
    }
}

#[derive(Clone, Copy)]
struct Breaks {
    ty: Option<Ty>,
    broke: bool,
    label: Option<u32>,
    block: bool,
}

#[derive(Clone, Copy)]
struct Clash {
    a: Ty,
    b: Ty,
    cycle: bool,
}

/// Infers the body of the function `item` with its generic parameters set to `substs`.
pub fn infer(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Typed> {
    if let Some(typed) = compiler_made(cx, item, substs)? {
        return Ok(typed);
    }
    let sig = cx.signature(item)?;
    let slot = object::virtual_slot(cx, item, substs)?;
    let foreign = cx.foreign(item) || slot.is_some();
    let body = if foreign { std::rc::Rc::default() } else { cx.body(item)? };
    let src = cx.body_src(item);
    let home = cx.module_of(item)?;
    let root = Some(item.body_owner());
    let args = cx.types.substs(substs).to_vec();
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    (infer.root, infer.instance) = (root, Some((item, substs)));
    let mut frame = infer.frame(item, &args)?;
    (frame.src, frame.root) = (src, root);
    infer.swap_frame(frame);
    infer.out.exprs.resize(body.exprs.len(), Ty::NONE);
    infer.run(foreign)?;
    let mut typed = infer.finish()?;
    if let Some(slot) = slot {
        typed.shape = Shape::Virtual(slot);
    }
    settle(cx, (item, substs), typed)
}

pub(super) fn settle(cx: &mut Context<'_>, instance: (ItemRef, Substs), mut typed: Typed) -> Result<Typed> {
    closure::register(cx, instance, &typed);
    if object::fill_drops(cx, &mut typed)? {
        closure::register(cx, instance, &typed);
    }
    typed.drops = drop::drop_calls(cx, &typed)?;
    Ok(typed)
}

/// One `.await`: the `IntoFuture::into_future` of the awaited value, the future it gives, the `Future::poll` of that future, and the `Poll` it returns.
#[derive(Clone, Debug)]
pub struct Awaited {
    pub into_future: Res,
    pub future: Ty,
    pub poll: Res,
    pub polled: Ty,
}

fn compiler_made(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Option<Typed>> {
    if matches!(cx.item(item).1.kind, crate::outline::item::ItemKind::Struct | crate::outline::item::ItemKind::Enum) {
        return adt::ctor_typed(cx, item, substs).map(Some);
    }
    let slice = cx.types.substs(substs);
    if cx.lang(item) == Some("drop_glue") {
        let Some(&ty) = slice.first() else { bail!("drop glue with no type") };
        return drop::glue_typed(cx, item, ty).map(Some);
    }
    let Some(&ty) = slice.first() else { return Ok(None) };
    if cx.lang(item) == Some("clone_fn") && clone::compiler_clone(cx, ty) {
        return clone::clone_typed(cx, item, ty).map(Some);
    }
    if cx.lang(item) == Some("poll") && cx.closure(ty).is_some_and(|info| info.closure.coroutine) {
        return closure::coroutine_typed(cx, item, ty).map(Some);
    }
    let (once, by_ref) = (cx.call_once() == Some(item), closure::by_reference(cx, item).is_some());
    match cx.types.kind(ty) {
        Kind::Closure(..) if once => closure::closure_typed(cx, ty).map(Some),
        Kind::Closure(..) | Kind::FnDef(..) if once || by_ref => closure::untuple_typed(cx, item, substs).map(Some),
        _ => Ok(None),
    }
}

impl<'c, 'a> Infer<'c, 'a> {
    fn start(cx: &'c mut Context<'a>, (body, sig, src): (&'c Body, &'c Signature, &'a Source), home: ModRef) -> Self {
        let (generics, vars, metadata, obligations, projections, opaques, operands, try_blocks, scopes, items, loops, uses) = Default::default();
        let out = zonk::take_out();
        Infer { cx, body, sig, src, root: None, instance: None, home, generics, self_ty: None, block: None, frame_item: None, vars, metadata, obligations, projections, opaques, operands, try_blocks, awaits: Vec::new(), ret_sig: None, out, scopes, items, loops, uses, unsure: Vec::new(), diverges: false, hidden: Vec::new(), const_args: Vec::new(), selecting: Vec::new(), frames: 0, evaluating: Vec::new(), imports: HashMap::default() }
    }
}

impl Infer<'_, '_> {
    fn run(&mut self, foreign: bool) -> Result<()> {
        self.hidden.clear();
        let declared = match self.sig.ret {
            Some(ty) => self.lower_ty(&self.sig.body, ty)?,
            None => self.cx.types.unit(),
        };
        let ret = if self.sig.is_async { self.fresh(Kind::Var) } else { declared };
        self.out.ret = Some(ret);
        self.bind_hidden()?;
        self.ret_sig = self.opaque_fn_sig()?;
        self.bind_params()?;
        let Some(root) = self.body.root else {
            return if foreign { Ok(()) } else { bail!("body has no root") };
        };
        if self.sig.is_async {
            let params = self.out.params.clone();
            let found = self.coroutine_expr(root, (Some(declared), true), &params)?;
            self.out.exprs[root.0 as usize] = found;
            return self.unify(ret, found);
        }
        let found = self.expr(root, Some(ret))?;
        self.coerce_expr(root, found, ret)
    }

    fn bind_params(&mut self) -> Result<()> {
        let sig = self.sig;
        if let Some((kind, _)) = sig.receiver {
            let ty = self.receiver_ty(&sig.body, kind)?;
            let local = self.new_local(ty);
            self.scopes.push((("self", 0), local));
            self.out.params.push(local);
        }
        for (index, &(pat, ty)) in sig.params.iter().enumerate() {
            let ty = self.lower_ty(&sig.body, ty)?;
            let own = self.body.fn_params.get(index).copied();
            let local = self.param_pat(&sig.body, pat, ty, own)?;
            self.out.params.push(local);
        }
        Ok(())
    }

    fn param_pat(&mut self, arena: &Body, pat: PatId, ty: Ty, own: Option<PatId>) -> Result<u32> {
        match *arena.pat(pat) {
            crate::body::Pat::Bind { name, sub: None, by_ref: false, .. } => {
                let local = self.new_local(ty);
                self.scope_name(name, local);
                Ok(local)
            }
            crate::body::Pat::Wild => Ok(self.new_local(ty)),
            _ => {
                let Some(own) = own else { bail!("a parameter pattern outside the body's arena") };
                let local = self.new_local(ty);
                self.bind_pat(own, ty)?;
                self.out.unpacks.insert(local, own);
                Ok(local)
            }
        }
    }

    fn new_local(&mut self, ty: Ty) -> u32 {
        self.out.locals.push(Local { ty });
        self.out.locals.len() as u32 - 1
    }

    fn trial<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<Option<T>>) -> Result<Option<T>> {
        let saved = self.vars.clone();
        let out = f(self);
        if !matches!(out, Ok(Some(_))) {
            self.vars = saved;
        }
        out
    }

    fn probe<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = self.vars.clone();
        let out = f(self);
        self.vars = saved;
        out
    }

    fn fresh(&mut self, kind: fn(u32) -> Kind) -> Ty {
        self.vars.push(None);
        self.cx.types.intern(kind(self.vars.len() as u32 - 1))
    }

    fn var_of(&self, ty: Ty) -> Option<u32> {
        match self.cx.types.kind(ty) {
            Kind::Var(index) | Kind::IntVar(index) | Kind::FloatVar(index) => Some(*index),
            _ => None,
        }
    }

    /// `ty` with its outer inference variables followed; a variable of another inference stays as it is.
    pub fn resolve(&self, mut ty: Ty) -> Ty {
        while let Some(bound) = self.var_of(ty).and_then(|index| self.vars.get(index as usize).copied().flatten()) {
            ty = bound;
        }
        ty
    }

    fn bind(&mut self, var: Ty, to: Ty) -> Result<(), Clash> {
        let index = self.var_of(var).unwrap_or(u32::MAX) as usize;
        let accepts = match (self.cx.types.kind(var), self.cx.types.kind(to)) {
            (Kind::Var(_), _) => true,
            (Kind::IntVar(_), Kind::IntVar(_)) | (Kind::FloatVar(_), Kind::FloatVar(_)) => true,
            (Kind::IntVar(_), Kind::Prim(primitive)) => int_info(*primitive).is_some(),
            (Kind::FloatVar(_), Kind::Prim(primitive)) => super::ty::float_bits(*primitive).is_some(),
            _ => false,
        };
        if !accepts {
            return Err(Clash { a: var, b: to, cycle: false });
        }
        if self.occurs(index as u32, to) {
            return Err(Clash { a: var, b: to, cycle: true });
        }
        self.vars[index] = Some(to);
        Ok(())
    }

    fn occurs(&self, var: u32, ty: Ty) -> bool {
        let ty = self.resolve(ty);
        if self.var_of(ty) == Some(var) {
            return true;
        }
        self.cx.types.kind(ty).children(&self.cx.types).any(|item| self.occurs(var, item))
    }

    /// Makes `a` and `b` the same type.
    pub fn unify(&mut self, a: Ty, b: Ty) -> Result<()> {
        self.unify_quiet(a, b).map_err(|clash| self.clash_error(clash))
    }

    pub(super) fn unifies(&mut self, a: Ty, b: Ty) -> bool {
        self.unify_quiet(a, b).is_ok()
    }

    fn clash_error(&self, Clash { a, b, cycle }: Clash) -> anyhow::Error {
        match cycle {
            true => anyhow::anyhow!("cyclic type: {} occurs in {}", self.show(a), self.show(b)),
            false => anyhow::anyhow!("mismatched types: {} and {}", self.show(a), self.show(b)),
        }
    }

    fn unify_quiet(&mut self, a: Ty, b: Ty) -> Result<(), Clash> {
        let (a, b) = (self.resolve(a), self.resolve(b));
        if a == b {
            return Ok(());
        }
        let (kind_a, kind_b) = (self.cx.types.kind(a).clone(), self.cx.types.kind(b).clone());
        match (&kind_a, &kind_b) {
            (Kind::Var(_), _) => self.bind(a, b),
            (_, Kind::Var(_)) => self.bind(b, a),
            (Kind::IntVar(_) | Kind::FloatVar(_), _) => self.bind(a, b),
            (_, Kind::IntVar(_) | Kind::FloatVar(_)) => self.bind(b, a),
            _ => self.unify_shapes(&kind_a, &kind_b).map_err(|()| Clash { a, b, cycle: false }),
        }
    }

    fn unify_shapes(&mut self, a: &Kind, b: &Kind) -> Result<(), ()> {
        let pairs: Vec<(Ty, Ty)> = match (a, b) {
            (Kind::Tuple(x), Kind::Tuple(y)) => self.zip_substs(*x, *y)?,
            (Kind::Ref(m, x), Kind::Ref(n, y)) | (Kind::Ptr(m, x), Kind::Ptr(n, y)) if m == n => vec![(*x, *y)],
            (Kind::Array(x, m), Kind::Array(y, n)) => vec![(*x, *y), (*m, *n)],
            (Kind::Slice(x), Kind::Slice(y)) => vec![(*x, *y)],
            (Kind::Pat(x, s, e), Kind::Pat(y, t, f)) if (s, e) == (t, f) => vec![(*x, *y)],
            (Kind::Adt(i, x), Kind::Adt(j, y)) | (Kind::FnDef(i, x), Kind::FnDef(j, y)) | (Kind::Dyn(i, x, ..), Kind::Dyn(j, y, ..)) if i == j && dyn_members(a) == dyn_members(b) => self.zip_substs(*x, *y)?,
            (Kind::FnPtr(x, r), Kind::FnPtr(y, s)) => {
                let mut pairs = self.zip_substs(*x, *y)?;
                pairs.push((*r, *s));
                pairs
            }
            _ => return Err(()),
        };
        pairs.into_iter().try_for_each(|(x, y)| self.unify_quiet(x, y).map_err(drop))
    }

    fn zip_substs(&self, a: Substs, b: Substs) -> Result<Vec<(Ty, Ty)>, ()> {
        if a == b {
            return Ok(Vec::new());
        }
        let (xs, ys) = (self.cx.types.substs(a), self.cx.types.substs(b));
        if xs.len() != ys.len() {
            return Err(());
        }
        Ok(xs.iter().copied().zip(ys.iter().copied()).collect())
    }

    /// Checks that a value of type `found` can be used where `expected` is wanted, through the coercions that keep its bits.
    pub fn coerce(&mut self, found: Ty, expected: Ty) -> Result<()> {
        let (found, expected) = (self.resolve(found), self.resolve(expected));
        match (self.cx.types.kind(found), self.weakened(found, expected)) {
            (Kind::Never, _) => Ok(()),
            (_, Some((a, b))) => self.unify(a, b),
            _ => self.unify(found, expected),
        }?;
        self.settle_pending()
    }

    pub(super) fn weakened(&self, found: Ty, expected: Ty) -> Option<(Ty, Ty)> {
        match (self.cx.types.kind(self.resolve(found)), self.cx.types.kind(self.resolve(expected))) {
            (&Kind::Ref(from, a), &Kind::Ptr(to, b)) | (&Kind::Ref(from, a), &Kind::Ref(to, b)) | (&Kind::Ptr(from, a), &Kind::Ptr(to, b)) if from == Mut::Mut && to == Mut::Not => Some((a, b)),
            (&Kind::Ref(from, a), &Kind::Ptr(to, b)) if from == to => Some((a, b)),
            _ => None,
        }
    }

    fn settle_pending(&mut self) -> Result<()> {
        loop {
            let before = self.backlog();
            self.settle_metadata()?;
            self.settle_projections()?;
            self.settle_opaques(false)?;
            self.settle_operators(false)?;
            self.settle_obligations()?;
            if self.backlog() >= before {
                return Ok(());
            }
        }
    }

    fn backlog(&self) -> usize {
        self.metadata.len() + self.projections.len() + self.opaques.len() + self.operands.len() + self.obligations.len()
    }

    pub(super) fn known_ty(&mut self, ty: Ty) -> Result<Ty> {
        if matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)) {
            self.settle_pending()?;
        }
        Ok(self.resolve(ty))
    }

    /// Coerces the value of `expr`, of type `found`, to `expected`, at each branch value it ends in, recording an array-to-slice unsizing.
    pub fn coerce_expr(&mut self, expr: ExprId, found: Ty, expected: Ty) -> Result<()> {
        if let Some(ends) = self.branch_ends(expr) {
            for end in ends {
                let found = self.out.exprs[end.0 as usize];
                self.coerce_expr(end, found, expected)?;
            }
            self.out.exprs[expr.0 as usize] = expected;
            return Ok(());
        }
        match self.coercion(found, expected)? {
            Some(coercion) => {
                self.out.coercions.insert(expr, coercion);
                Ok(())
            }
            None => self.coerce(found, expected),
        }
    }

    fn branch_ends(&self, expr: ExprId) -> Option<Vec<ExprId>> {
        match *self.body.expr(expr) {
            Expr::Paren(inner) => Some(vec![inner]),
            Expr::Block { tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe, .. } => Some(vec![tail]),
            Expr::If(_, then, Some(otherwise)) => Some(vec![then, otherwise]),
            Expr::Match(_, arms) => Some(self.body.list(arms).iter().map(|&at| self.body.arms[at as usize].body).collect()),
            _ => None,
        }
    }

    pub(super) fn known(&self, expected: Option<Ty>) -> Option<Ty> {
        expected.filter(|ty| !matches!(self.cx.types.kind(self.resolve(*ty)), Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)))
    }

    fn settle_obligations(&mut self) -> Result<()> {
        let mut at = 0;
        while at < self.obligations.len() {
            let (of, substs) = self.obligations[at].clone();
            let self_ty = self.resolve(substs[0]);
            if matches!(self.cx.types.kind(self_ty), Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) {
                if self.opaque_holds(of, &substs)? == Some(true) {
                    self.obligations.swap_remove(at);
                } else {
                    at += 1;
                }
                continue;
            }
            let known = substs[1..].iter().all(|ty| !self.has_vars(*ty));
            if (!known && self.assume(of, &substs)?) || self.settled_by_impl(of, self_ty, &substs[1..])? || known {
                self.obligations.swap_remove(at);
            } else {
                at += 1;
            }
        }
        Ok(())
    }

    fn settled_by_impl(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        Ok(self.select(of, self_ty, args)?.is_some() && !self.has_vars(self_ty))
    }

    pub(super) fn assume(&mut self, of: ItemRef, substs: &[Ty]) -> Result<bool> {
        let mut matched = Vec::new();
        for (held, types) in self.assumptions()? {
            if held != of || matched.contains(&types) {
                continue;
            }
            let fits = |infer: &mut Self| -> Result<Option<()>> { Ok(substs.iter().zip(&types).all(|(&a, &b)| infer.unifies(a, b)).then_some(())) };
            if self.probe(fits)?.is_some() {
                matched.push(types);
            }
        }
        let [types] = &matched[..] else { return Ok(false) };
        substs.iter().zip(types).try_for_each(|(&a, &b)| self.unify(a, b))?;
        Ok(true)
    }

    fn settle_metadata(&mut self) -> Result<()> {
        let mut at = 0;
        while at < self.metadata.len() {
            let (pointee, result) = self.metadata[at];
            if matches!(self.cx.types.kind(self.resolve(pointee)), Kind::Var(_)) {
                at += 1;
                continue;
            }
            self.metadata.swap_remove(at);
            let found = self.metadata_of(pointee)?;
            self.unify(found, result)?;
        }
        Ok(())
    }

    fn unsettled(&self) -> ([usize; 5], usize, usize) {
        (self.cx.asks.mark(), self.projections.len(), self.obligations.len())
    }

    fn bodiless(&self) -> bool {
        self.instance.is_none() && self.items.is_empty() && self.scopes.is_empty() && self.const_args.is_empty() && self.evaluating.is_empty() && self.selecting.is_empty()
    }

    /// A short spelling of `ty` for messages.
    pub fn show(&self, ty: Ty) -> String {
        self.cx.type_text(ty, &|ty| self.resolve(ty))
    }
}

/// The field types of each variant of the struct, enum or union `item` at `args`.
pub fn field_types(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Fields> {
    detached(cx, item, |infer| infer.adt_fields(item, args))
}

/// The type of the last field of the struct `item` at `args`.
pub fn tail_field(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Option<Ty>> {
    detached(cx, item, |infer| infer.adt_tail_field(item, args))
}

fn dyn_members(kind: &Kind) -> (&[ItemRef], &[ItemRef]) {
    match kind {
        Kind::Dyn(_, _, members, autos) => (members, autos),
        _ => (&[], &[]),
    }
}

/// Whether the struct `item` may end in an unsized field, which its definition decides for every instantiation as rustc's layout does; with no `?Sized` parameter, a tail the stand-ins cannot lower, such as a projection on one, is sized.
pub fn maybe_unsized(cx: &mut Context<'_>, item: ItemRef) -> Result<bool> {
    if cx.item(item).1.kind != crate::outline::item::ItemKind::Struct {
        return Ok(false);
    }
    let def = cx.adt_def(item)?;
    let src = &cx.file(item).source;
    let byte = cx.types.intern(Kind::Prim(Primitive::U8));
    let stand_ins = [cx.types.intern(Kind::Value(1)), cx.types.intern(Kind::Slice(byte)), byte];
    let args: Vec<Ty> = scope::type_params(&def.generics.params).map(|generic| match generic.kind {
        crate::body::sig::GenericKind::Const(_) => stand_ins[0],
        _ if relaxed(&def.body, src, &def.generics.predicates, generic) => stand_ins[1],
        _ => stand_ins[2],
    }).collect();
    let any_relaxed = args.contains(&stand_ins[1]);
    let ty = cx.types.adt(item, &args);
    match cx.is_unsized(ty) {
        Err(_) if !any_relaxed => Ok(false),
        Err(_) => detached(cx, item, |infer| infer.open_tail_unsized(item, &args, stand_ins[2])),
        found => found,
    }
}

fn relaxed(arena: &Body, src: &Source, predicates: &[crate::body::sig::Predicate], generic: &crate::body::sig::Generic) -> bool {
    let bounds = scope::generic_bounds(arena, src, predicates, generic);
    bounds.iter().any(|&bound| matches!(arena.bounds[bound as usize], crate::body::Bound::Trait { maybe: true, .. }))
}

/// The discriminant of each variant of the enum `item`, whose tags are integers of type `primitive`.
pub fn discriminants(cx: &mut Context<'_>, item: ItemRef, primitive: Primitive) -> Result<Vec<i128>> {
    detached(cx, item, |infer| infer.discriminants(item, primitive))
}

/// The return type of the non-generic function `item`.
pub fn return_type(cx: &mut Context<'_>, item: ItemRef) -> Result<Ty> {
    detached(cx, item, |infer| Ok(infer.fn_types(item, &[])?.1))
}

pub(crate) fn detached<T>(cx: &mut Context<'_>, item: ItemRef, run: impl FnOnce(&mut Infer<'_, '_>) -> Result<T>) -> Result<T> {
    let (body, sig) = (Body::default(), Signature::default());
    let src = &cx.file(item).source;
    let home = cx.module_of(item)?;
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    run(&mut infer)
}
