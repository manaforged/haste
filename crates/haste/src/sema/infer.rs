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

pub use clone::cloned_tuple;
pub use closure::{By, Capture, Closure, ClosureBody, ClosureInfo, by_reference, register};
pub use drop::Glue;
pub use looping::{For, Tried};
pub use method::Adjust;
pub use coerce::Coercion;
pub use object::{ALIGN_SLOT, SIZE_SLOT, Vtable, virtual_slot};
pub use pat::{position, rest_split};
pub use path::{parse_int, segments};
pub use scope::type_params;

use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, bail};

use crate::body::sig::Signature;
use crate::body::{Body, BlockKind, Expr, ExprId, PatId};
use crate::names::{ItemRef, ModRef, Primitive};
use crate::outline::source::Source;

use super::Context;
use super::ty::{Kind, Mut, Ty, int_info};

/// What a path expression names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Res {
    Local(u32),
    Fn(ItemRef, Vec<Ty>),
    Ctor(ItemRef, u32, Vec<Ty>),
    /// The value of a const parameter, as the const value it is set to.
    Const(Ty),
}

/// One local variable of a body.
#[derive(Clone, Copy, Debug)]
pub struct Local {
    pub ty: Ty,
}

/// What the code of an instance is: a body, drop glue, a closure's body, a call through the vtable slot of a trait object, or a constructor.
#[derive(Clone, Debug, Default)]
pub enum Shape {
    #[default]
    Body,
    Glue(Glue),
    Closure(ClosureBody),
    Virtual(u32),
    /// A call of a closure or function item through a call trait method, which spreads the argument tuple into this call of the closure's body or the function.
    Untuple(Res),
    /// The tuple constructor of this variant, called as a function.
    Ctor(u32),
    /// The `Clone::clone` of a tuple that the compiler writes, with the clone call of each field.
    Clone(Vec<Res>),
}

/// The inferred types and resolutions of one function body at one set of type arguments.
#[derive(Clone, Debug, Default)]
pub struct Typed {
    pub exprs: Vec<Ty>,
    pub res: HashMap<ExprId, Res>,
    pub locals: Vec<Local>,
    pub binds: HashMap<PatId, u32>,
    pub adjusts: HashMap<ExprId, Adjust>,
    /// The field accesses and calls that dereference their base or callee this many times first, through references and `Box`es.
    pub autoderefs: HashMap<ExprId, u32>,
    /// The dereferences that call `Deref::deref` or `DerefMut::deref_mut`, by expression and step: step 0 of a `*`, or one step of an autoderef.
    pub derefs: HashMap<(ExprId, u32), Res>,
    /// The operator expressions that call a trait method: `Index::index`, `IndexMut::index_mut`, or a binary operator trait such as `Add::add` or `PartialEq::eq`.
    pub operators: HashMap<ExprId, Res>,
    /// The expressions whose value coerces to the type it is used at, and how.
    pub coercions: HashMap<ExprId, Coercion>,
    pub fields: HashMap<ExprId, u32>,
    /// The field that each struct literal initializer fills, by its value expression, apart from `fields` because that value may itself be a field read.
    pub inits: HashMap<ExprId, u32>,
    /// The byte offset that each `offset_of!` names.
    pub offsets: HashMap<ExprId, u64>,
    /// The fields that each range expression fills: one per end it gives, in order, then `exhausted` for an inclusive range.
    pub ranges: HashMap<ExprId, Vec<u32>>,
    pub field_pats: HashMap<PatId, u32>,
    pub pat_res: HashMap<PatId, Res>,
    pub pat_derefs: HashMap<PatId, u32>,
    pub ref_binds: HashMap<PatId, crate::sema::ty::Mut>,
    /// The or-patterns whose alternatives bind names, which every alternative binds to the same locals.
    pub or_binds: rustc_hash::FxHashSet<PatId>,
    /// The calls of values that implement `Fn` or `FnMut` themselves, which pass the callee's address and the argument tuple.
    pub overloaded: rustc_hash::FxHashSet<ExprId>,
    /// The call arguments that `#[rustc_legacy_const_generics]` turns into const generic arguments, which pass no value.
    pub const_args: rustc_hash::FxHashSet<ExprId>,
    pub pats: HashMap<PatId, Ty>,
    pub params: Vec<u32>,
    pub ret: Option<Ty>,
    pub fors: HashMap<ExprId, For>,
    pub tries: HashMap<ExprId, Tried>,
    /// The `.await` expressions of the body, with the calls each makes.
    pub awaits: HashMap<ExprId, Awaited>,
    /// The drop glue of each type that a local or a temporary of the body holds and that needs drop.
    pub drops: Vec<Res>,
    /// The closure expressions of the body.
    pub closures: HashMap<ExprId, Closure>,
    /// The `try` blocks of the body, each with the `Try::from_output` call that wraps its value.
    pub try_blocks: HashMap<ExprId, Res>,
    /// The `Drop::drop` of the `Box` that each by-value call of a `Box<dyn FnOnce>` frees once the call consumed its contents.
    pub box_frees: HashMap<ExprId, Res>,
    /// The parameter locals whose pattern takes the argument apart, with that pattern in the body's arena.
    pub unpacks: HashMap<u32, PatId>,
    /// What the code of this instance is.
    pub shape: Shape,
}

impl Typed {
    /// The return type, or `Ty::NONE` for a body that has none.
    pub fn ret_ty(&self) -> Ty {
        self.ret.unwrap_or(Ty::NONE)
    }

    /// Every function the body calls, with its type arguments.
    pub fn calls(&self) -> impl Iterator<Item = (ItemRef, &[Ty])> {
        let loops = self.fors.values().flat_map(|each| [&each.into_iter, &each.next]);
        let tries = self.tries.values().flat_map(|each| [&each.branch, &each.from_residual]);
        let awaits = self.awaits.values().flat_map(|each| [&each.into_future, &each.poll]);
        let glue = match &self.shape {
            Shape::Glue(glue) => Some(glue),
            _ => None,
        };
        let glue = glue.into_iter().flat_map(|glue| glue.drop.iter().chain(&glue.parts).chain(&glue.resume));
        let untuple = match &self.shape {
            Shape::Untuple(res) => Some(res),
            _ => None,
        };
        let cloned = match &self.shape {
            Shape::Clone(calls) => &calls[..],
            _ => &[],
        };
        let coerced = self.coercions.values().flat_map(|coercion| -> Box<dyn Iterator<Item = &Res>> {
            match coercion {
                Coercion::Vtable(vtable) => Box::new(vtable.calls().into_iter()),
                Coercion::Reify(res) => Box::new(std::iter::once(res)),
                Coercion::Len(_) | Coercion::Autos | Coercion::Upcast(_) => Box::new(std::iter::empty()),
                Coercion::Deref(steps, _) => Box::new(steps.iter().filter_map(|(_, res)| res.as_ref())),
            }
        });
        self.res.values().chain(self.pat_res.values()).chain(self.derefs.values()).chain(self.operators.values()).chain(self.try_blocks.values()).chain(self.box_frees.values()).chain(loops).chain(tries).chain(awaits).chain(&self.drops).chain(glue).chain(untuple).chain(cloned).chain(coerced).filter_map(|res| match res {
            Res::Fn(item, substs) => Some((*item, &substs[..])),
            _ => None,
        })
    }
}

/// A local variable's name and the `macro_rules!` hygiene scope that wrote it.
pub(super) type Name<'a> = (&'a str, u32);

/// The inference of one body.
pub struct Infer<'c, 'a> {
    cx: &'c mut Context<'a>,
    body: &'c Body,
    sig: &'c Signature,
    src: &'a Source,
    root: Option<ItemRef>,
    instance: Option<(ItemRef, Vec<Ty>)>,
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
    /// The `.await` expressions on a pending async fn's future, with that future, finished once it is revealed.
    awaits: Vec<(ExprId, Ty)>,
    ret_sig: Option<Ty>,
    out: Typed,
    scopes: Vec<(Name<'a>, u32)>,
    items: Vec<(&'a str, ItemRef)>,
    loops: Vec<Breaks>,
    uses: Vec<closure::Captures>,
    /// The closures whose captures were decided while a read's type was unknown: each one's uses, whether it moves, and what it always captures.
    unsure: Vec<(ExprId, closure::Captures, bool, Vec<u32>)>,
    diverges: bool,
    /// The `impl Trait` types that no frame parameter stands for: the token of each and the variable for its hidden type.
    hidden: Vec<(u32, Ty)>,
    /// The values of the parameters of the const fns that a constant expression is calling, innermost last.
    const_args: Vec<(String, (u128, (u32, bool)))>,
    selecting: Vec<String>,
    /// How many frames of aliases, constants and other items' syntax are open, which `RECURSION_LIMIT` bounds.
    frames: usize,
    /// The constants being evaluated, innermost last, so a constant that depends on itself is an error.
    evaluating: Vec<ItemRef>,
}

/// rustc's default recursion limit, which bounds dereference steps and nested alias and constant expansion.
pub(super) const RECURSION_LIMIT: usize = 128;

/// What the `break`s of one loop or labeled block give: the type they coerce to once it is known, whether any `break` was seen, its label, and whether it is a block.
#[derive(Clone, Copy)]
struct Breaks {
    ty: Option<Ty>,
    broke: bool,
    label: Option<u32>,
    block: bool,
}

/// Two types that do not unify, or with `cycle` a variable that occurs in the type it would stand for.
#[derive(Clone, Copy)]
struct Clash {
    a: Ty,
    b: Ty,
    cycle: bool,
}

/// Infers the body of the function `item` with its generic parameters set to `substs`.
pub fn infer(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Typed> {
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
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    (infer.root, infer.instance) = (root, Some((item, substs.to_vec())));
    let mut frame = infer.frame(item, substs)?;
    (frame.src, frame.root) = (src, root);
    infer.swap_frame(frame);
    infer.out.exprs = vec![Ty::NONE; body.exprs.len()];
    infer.run(foreign)?;
    let mut typed = infer.finish()?;
    if let Some(slot) = slot {
        typed.shape = Shape::Virtual(slot);
    }
    settle(cx, (item, substs), typed)
}

fn settle(cx: &mut Context<'_>, instance: (ItemRef, &[Ty]), mut typed: Typed) -> Result<Typed> {
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

/// The typed form of an instance that the compiler writes instead of a body: drop glue, or the `call_once` of a closure.
fn compiler_made(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Option<Typed>> {
    if matches!(cx.item(item).1.kind, crate::outline::item::ItemKind::Struct | crate::outline::item::ItemKind::Enum) {
        return adt::ctor_typed(cx, item, substs).map(Some);
    }
    if cx.lang(item) == Some("drop_glue") {
        let Some(&ty) = substs.first() else { bail!("drop glue with no type") };
        return drop::glue_typed(cx, item, ty).map(Some);
    }
    let Some(&ty) = substs.first() else { return Ok(None) };
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
        Infer { cx, body, sig, src, root: None, instance: None, home, generics, self_ty: None, block: None, frame_item: None, vars, metadata, obligations, projections, opaques, operands, try_blocks, awaits: Vec::new(), ret_sig: None, out: Typed::default(), scopes, items, loops, uses, unsure: Vec::new(), diverges: false, hidden: Vec::new(), const_args: Vec::new(), selecting: Vec::new(), frames: 0, evaluating: Vec::new() }
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

    /// Runs `f` and keeps the variable bindings it made only when it gives a value.
    fn trial<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<Option<T>>) -> Result<Option<T>> {
        let saved = self.vars.clone();
        let out = f(self);
        if !matches!(out, Ok(Some(_))) {
            self.vars = saved;
        }
        out
    }

    /// Runs `f` and undoes every variable binding it made.
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
        self.cx.types.kind(ty).children().any(|item| self.occurs(var, item))
    }

    /// Makes `a` and `b` the same type.
    pub fn unify(&mut self, a: Ty, b: Ty) -> Result<()> {
        self.unify_quiet(a, b).map_err(|clash| self.clash_error(clash))
    }

    /// Makes `a` and `b` the same type and says whether they unify, building no message when they do not.
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
            (Kind::Tuple(x), Kind::Tuple(y)) if x.len() == y.len() => x.iter().copied().zip(y.iter().copied()).collect(),
            (Kind::Ref(m, x), Kind::Ref(n, y)) | (Kind::Ptr(m, x), Kind::Ptr(n, y)) if m == n => vec![(*x, *y)],
            (Kind::Array(x, m), Kind::Array(y, n)) => vec![(*x, *y), (*m, *n)],
            (Kind::Slice(x), Kind::Slice(y)) => vec![(*x, *y)],
            (Kind::Pat(x, s, e), Kind::Pat(y, t, f)) if (s, e) == (t, f) => vec![(*x, *y)],
            (Kind::Adt(i, x), Kind::Adt(j, y)) | (Kind::FnDef(i, x), Kind::FnDef(j, y)) | (Kind::Dyn(i, x, ..), Kind::Dyn(j, y, ..)) if i == j && x.len() == y.len() && dyn_members(a) == dyn_members(b) => x.iter().copied().zip(y.iter().copied()).collect(),
            (Kind::FnPtr(x, r), Kind::FnPtr(y, s)) if x.len() == y.len() => x.iter().copied().zip(y.iter().copied()).chain([(*r, *s)]).collect(),
            _ => return Err(()),
        };
        pairs.into_iter().try_for_each(|(x, y)| self.unify_quiet(x, y).map_err(drop))
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

    /// The pointees of `found` and `expected` when a pointer of type `found` coerces to `expected` by dropping `mut` or by turning a reference into a raw pointer.
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

    /// How many deferred items inference still waits on, which each pass of `settle_pending` must cut to run again.
    fn backlog(&self) -> usize {
        self.metadata.len() + self.projections.len() + self.opaques.len() + self.operands.len() + self.obligations.len()
    }

    /// The type `ty` resolved, after settling what is pending when it is still an unknown variable, as rustc does before it needs a type.
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

    /// The expected type `expected` when it is already known, so that branches coerce to it instead of to each other.
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

    /// Whether the one impl that selection picks settles the obligation: its nested bounds wait while `self_ty` still holds variables, so it is tried again once they are known.
    fn settled_by_impl(&mut self, of: ItemRef, self_ty: Ty, args: &[Ty]) -> Result<bool> {
        Ok(self.select(of, self_ty, args)?.is_some() && !self.has_vars(self_ty))
    }

    /// Ties `substs` of trait `of` to the one bound of the frame they match, and whether exactly one did, as rustc prefers a where-clause over impls.
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

    /// A short spelling of `ty` for messages.
    pub fn show(&self, ty: Ty) -> String {
        self.cx.type_text(ty, &|ty| self.resolve(ty))
    }
}

/// The field types of each variant of the struct, enum or union `item` at `args`.
pub fn field_types(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Vec<Vec<Ty>>> {
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
    let ty = cx.types.intern(Kind::Adt(item, args.clone()));
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

/// Whether dropping a value of `ty` runs code.
pub fn needs_drop(cx: &mut Context<'_>, ty: Ty) -> Result<bool> {
    let Some(anchor) = cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
    detached(cx, anchor, |infer| infer.needs_drop(ty))
}

/// Whether `ty` has a `Drop` impl of its own, which forbids moving out of its parts.
pub fn has_drop_impl(cx: &mut Context<'_>, ty: Ty) -> Result<bool> {
    let Some(anchor) = cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
    detached(cx, anchor, |infer| infer.drop_impl(ty).map(|found| found.is_some()))
}

/// The `Drop::drop` of `ty`'s own `Drop` impl, which for a `Box` frees the allocation without dropping what it owns.
pub fn drop_fn(cx: &mut Context<'_>, ty: Ty) -> Result<Option<Res>> {
    let Some(anchor) = cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
    detached(cx, anchor, |infer| infer.own_drop(ty))
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
