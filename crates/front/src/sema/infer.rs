mod adt;
mod assoc;
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

pub use closure::{By, Capture, Closure, ClosureBody, ClosureInfo, closure_typed, register};
pub use drop::Glue;
pub use looping::{For, Tried};
pub use method::Adjust;
pub use coerce::Coercion;
pub use object::{ALIGN_SLOT, FIRST_METHOD, SIZE_SLOT, Vtable, virtual_slot};
pub use pat::{Mode, position, rest_split};
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
    pub mutable: bool,
}

/// What the code of an instance is: a body, drop glue, a closure's body, or a call through the vtable slot of a trait object.
#[derive(Clone, Debug, Default)]
pub enum Shape {
    #[default]
    Body,
    Glue(Glue),
    Closure(ClosureBody),
    Virtual(u32),
    /// A call of a closure through `Fn::call` or `FnMut::call_mut`, which spreads the argument tuple into this call of the closure's own body.
    Untuple(Res),
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
    /// The fields that each range expression fills: one per end it gives, in order, then `exhausted` for an inclusive range.
    pub ranges: HashMap<ExprId, Vec<u32>>,
    pub field_pats: HashMap<PatId, u32>,
    pub pat_res: HashMap<PatId, Res>,
    pub pat_derefs: HashMap<PatId, u32>,
    pub ref_binds: HashMap<PatId, crate::sema::ty::Mut>,
    pub pats: HashMap<PatId, Ty>,
    pub params: Vec<u32>,
    pub ret: Option<Ty>,
    pub fors: HashMap<ExprId, For>,
    pub tries: HashMap<ExprId, Tried>,
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
    /// Every function the body calls, with its type arguments.
    pub fn calls(&self) -> impl Iterator<Item = (ItemRef, &[Ty])> {
        let loops = self.fors.values().flat_map(|each| [&each.into_iter, &each.next]);
        let tries = self.tries.values().flat_map(|each| [&each.branch, &each.from_residual]);
        let glue = match &self.shape {
            Shape::Glue(glue) => Some(glue),
            _ => None,
        };
        let glue = glue.into_iter().flat_map(|glue| glue.drop.iter().chain(&glue.parts));
        let untuple = match &self.shape {
            Shape::Untuple(res) => Some(res),
            _ => None,
        };
        let coerced = self.coercions.values().flat_map(|coercion| -> Box<dyn Iterator<Item = &Res>> {
            match coercion {
                Coercion::Vtable(vtable) => Box::new(vtable.drop.iter().chain(vtable.methods.iter().flatten())),
                Coercion::Reify(res) => Box::new(std::iter::once(res)),
                Coercion::Len(_) => Box::new(std::iter::empty()),
                Coercion::Deref(steps, _) => Box::new(steps.iter().filter_map(|(_, res)| res.as_ref())),
            }
        });
        self.res.values().chain(self.pat_res.values()).chain(self.derefs.values()).chain(self.operators.values()).chain(self.try_blocks.values()).chain(self.box_frees.values()).chain(loops).chain(tries).chain(&self.drops).chain(glue).chain(untuple).chain(coerced).filter_map(|res| match res {
            Res::Fn(item, substs) => Some((*item, &substs[..])),
            _ => None,
        })
    }
}

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
    ret_sig: Option<Ty>,
    out: Typed,
    scopes: Vec<(&'a str, u32)>,
    items: Vec<(&'a str, ItemRef)>,
    loops: Vec<Breaks>,
    uses: Vec<closure::Uses>,
    diverges: bool,
}

/// What the `break`s of one loop give: the type they coerce to once it is known, and whether any `break` was seen.
#[derive(Clone, Copy)]
struct Breaks {
    ty: Option<Ty>,
    broke: bool,
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
    infer.out.exprs = vec![Ty(u32::MAX); body.exprs.len()];
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

/// The typed form of an instance that the compiler writes instead of a body: drop glue, or the `call_once` of a closure.
fn compiler_made(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Option<Typed>> {
    if cx.lang(item) == Some("drop_glue") {
        let Some(&ty) = substs.first() else { bail!("drop glue with no type") };
        return drop::glue_typed(cx, item, ty).map(Some);
    }
    match substs.first() {
        Some(&ty) if matches!(cx.types.kind(ty), Kind::Closure(..)) && cx.call_once() == Some(item) => closure::closure_typed(cx, ty).map(Some),
        Some(&ty) if matches!(cx.types.kind(ty), Kind::Closure(..)) && closure::by_reference(cx, item).is_some() => closure::untuple_typed(cx, item, substs).map(Some),
        _ => Ok(None),
    }
}

impl<'c, 'a> Infer<'c, 'a> {
    fn start(cx: &'c mut Context<'a>, (body, sig, src): (&'c Body, &'c Signature, &'a Source), home: ModRef) -> Self {
        let (generics, vars, metadata, obligations, projections, opaques, operands, try_blocks, scopes, items, loops, uses) = Default::default();
        Infer { cx, body, sig, src, root: None, instance: None, home, generics, self_ty: None, block: None, frame_item: None, vars, metadata, obligations, projections, opaques, operands, try_blocks, ret_sig: None, out: Typed::default(), scopes, items, loops, uses, diverges: false }
    }
}

impl Infer<'_, '_> {
    fn run(&mut self, foreign: bool) -> Result<()> {
        let ret = match self.sig.ret {
            Some(ty) => self.lower_ty(&self.sig.body, ty)?,
            None => self.cx.types.unit(),
        };
        self.out.ret = Some(ret);
        self.ret_sig = self.opaque_fn_sig()?;
        let sig = self.sig;
        if let Some((kind, mutable)) = sig.receiver {
            let ty = self.receiver_ty(&sig.body, kind)?;
            let local = self.new_local(ty, mutable);
            self.scopes.push(("self", local));
            self.out.params.push(local);
        }
        for (index, &(pat, ty)) in sig.params.iter().enumerate() {
            let ty = self.lower_ty(&sig.body, ty)?;
            let own = self.body.fn_params.get(index).copied();
            let local = self.param_pat(&sig.body, pat, ty, own)?;
            self.out.params.push(local);
        }
        let Some(root) = self.body.root else {
            return if foreign { Ok(()) } else { bail!("body has no root") };
        };
        let found = self.expr(root, Some(ret))?;
        self.coerce_expr(root, found, ret)
    }

    fn param_pat(&mut self, arena: &Body, pat: PatId, ty: Ty, own: Option<PatId>) -> Result<u32> {
        match *arena.pat(pat) {
            crate::body::Pat::Bind { name, mutable, sub: None, by_ref: false } => {
                let local = self.new_local(ty, mutable);
                self.scopes.push((self.src.text(name), local));
                Ok(local)
            }
            crate::body::Pat::Wild => Ok(self.new_local(ty, false)),
            _ => {
                let Some(own) = own else { bail!("a parameter pattern outside the body's arena") };
                let local = self.new_local(ty, false);
                self.bind_pat(own, ty)?;
                self.out.unpacks.insert(local, own);
                Ok(local)
            }
        }
    }

    fn new_local(&mut self, ty: Ty, mutable: bool) -> u32 {
        self.out.locals.push(Local { ty, mutable });
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

    /// `ty` with its outer inference variables followed.
    pub fn resolve(&self, mut ty: Ty) -> Ty {
        while let Some(bound) = self.var_of(ty).and_then(|index| self.vars[index as usize]) {
            ty = bound;
        }
        ty
    }

    fn bind(&mut self, var: Ty, to: Ty) -> Result<()> {
        let index = self.var_of(var).unwrap_or(u32::MAX) as usize;
        let accepts = match (self.cx.types.kind(var), self.cx.types.kind(to)) {
            (Kind::Var(_), _) => true,
            (Kind::IntVar(_), Kind::IntVar(_)) | (Kind::FloatVar(_), Kind::FloatVar(_)) => true,
            (Kind::IntVar(_), Kind::Prim(primitive)) => int_info(*primitive).is_some(),
            (Kind::FloatVar(_), Kind::Prim(primitive)) => super::ty::float_bits(*primitive).is_some(),
            _ => false,
        };
        if !accepts {
            bail!("mismatched types: {} and {}", self.show(var), self.show(to));
        }
        if self.occurs(index as u32, to) {
            bail!("cyclic type: {} occurs in {}", self.show(var), self.show(to));
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
            _ => self.unify_shapes(&kind_a, &kind_b).map_err(|_| anyhow::anyhow!("mismatched types: {} and {}", self.show(a), self.show(b))),
        }
    }

    fn unify_shapes(&mut self, a: &Kind, b: &Kind) -> Result<()> {
        let pairs: Vec<(Ty, Ty)> = match (a, b) {
            (Kind::Tuple(x), Kind::Tuple(y)) if x.len() == y.len() => x.iter().copied().zip(y.iter().copied()).collect(),
            (Kind::Ref(m, x), Kind::Ref(n, y)) | (Kind::Ptr(m, x), Kind::Ptr(n, y)) if m == n => vec![(*x, *y)],
            (Kind::Array(x, m), Kind::Array(y, n)) => vec![(*x, *y), (*m, *n)],
            (Kind::Slice(x), Kind::Slice(y)) => vec![(*x, *y)],
            (Kind::Pat(x, s, e), Kind::Pat(y, t, f)) if (s, e) == (t, f) => vec![(*x, *y)],
            (Kind::Adt(i, x), Kind::Adt(j, y)) | (Kind::FnDef(i, x), Kind::FnDef(j, y)) | (Kind::Dyn(i, x, _), Kind::Dyn(j, y, _)) if i == j && x.len() == y.len() && dyn_members(a) == dyn_members(b) => x.iter().copied().zip(y.iter().copied()).collect(),
            (Kind::FnPtr(x, r), Kind::FnPtr(y, s)) if x.len() == y.len() => x.iter().copied().zip(y.iter().copied()).chain([(*r, *s)]).collect(),
            _ => bail!("mismatch"),
        };
        pairs.into_iter().try_for_each(|(x, y)| self.unify(x, y))
    }

    /// Checks that a value of type `found` can be used where `expected` is wanted, through the coercions that keep its bits.
    pub fn coerce(&mut self, found: Ty, expected: Ty) -> Result<()> {
        let (found, expected) = (self.resolve(found), self.resolve(expected));
        match (self.cx.types.kind(found).clone(), self.cx.types.kind(expected).clone()) {
            (Kind::Never, _) => Ok(()),
            (Kind::Ref(from, a), Kind::Ptr(to, b)) | (Kind::Ref(from, a), Kind::Ref(to, b)) | (Kind::Ptr(from, a), Kind::Ptr(to, b)) if from == Mut::Mut && to == Mut::Not => self.unify(a, b),
            (Kind::Ref(from, a), Kind::Ptr(to, b)) if from == to => self.unify(a, b),
            _ => self.unify(found, expected),
        }?;
        self.settle_metadata()?;
        self.settle_projections()?;
        self.settle_opaques(false)?;
        self.settle_operators(false)?;
        self.settle_obligations()
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
            Expr::Block { tail: Some(tail), label: None, kind: BlockKind::Plain | BlockKind::Unsafe, .. } => Some(vec![tail]),
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
                at += 1;
                continue;
            }
            let known = substs[1..].iter().all(|ty| !self.has_vars(*ty));
            if self.select(of, self_ty, &substs[1..])?.is_some() || known {
                self.obligations.swap_remove(at);
            } else {
                at += 1;
            }
        }
        Ok(())
    }

    fn has_vars(&self, ty: Ty) -> bool {
        let ty = self.resolve(ty);
        let kind = self.cx.types.kind(ty);
        matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)) || kind.children().any(|item| self.has_vars(item))
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
        format!("{:?}", self.cx.types.kind(self.resolve(ty)))
    }

    fn zonk(&mut self, ty: Ty) -> Ty {
        let ty = self.resolve(ty);
        let kind = self.cx.types.kind(ty).clone();
        let fallback = match kind {
            Kind::IntVar(_) => Some(self.cx.types.prim(Primitive::I32)),
            Kind::FloatVar(_) => Some(self.cx.types.prim(Primitive::F64)),
            Kind::Var(_) => Some(self.cx.types.unit()),
            _ => None,
        };
        if let Some(fallback) = fallback {
            return fallback;
        }
        let kind = kind.map(|item| self.zonk(item));
        self.cx.types.intern(kind)
    }

    fn finish(mut self) -> Result<Typed> {
        self.settle_projections()?;
        self.settle_opaques(true)?;
        self.settle_operators(true)?;
        self.fall_back_numbers()?;
        self.settle_projections()?;
        self.zonk_types()?;
        self.zonk_calls()?;
        Ok(self.out)
    }

    fn fall_back_numbers(&mut self) -> Result<()> {
        let exprs = self.out.exprs.iter().copied().filter(|ty| ty.0 != u32::MAX);
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
        self.out.exprs = exprs.into_iter().map(|ty| if ty.0 == u32::MAX { ty } else { self.zonk(ty) }).collect();
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

    fn zonk_try(&mut self, each: Tried) -> Result<Tried> {
        Ok(Tried { branch: self.zonk_res(each.branch)?, from_residual: self.zonk_res(each.from_residual)?, flow: self.zonk(each.flow), ..each })
    }

    fn zonk_res(&mut self, res: Res) -> Result<Res> {
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

/// The field types of each variant of the struct, enum or union `item` at `args`.
pub fn field_types(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Vec<Vec<Ty>>> {
    detached(cx, item, |infer| infer.adt_fields(item, args))
}

/// The type of the last field of the struct `item` at `args`.
pub fn tail_field(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Option<Ty>> {
    detached(cx, item, |infer| infer.adt_tail_field(item, args))
}

fn dyn_members(kind: &Kind) -> &[ItemRef] {
    match kind {
        Kind::Dyn(_, _, members) => members,
        _ => &[],
    }
}

/// Whether the struct `item` may end in an unsized field, which its definition decides for every instantiation as rustc's layout does.
pub fn maybe_unsized(cx: &mut Context<'_>, item: ItemRef) -> Result<bool> {
    if cx.item(item).1.kind != crate::outline::item::ItemKind::Struct {
        return Ok(false);
    }
    let def = cx.adt_def(item)?;
    let src = &cx.file(item).source;
    let byte = cx.types.intern(Kind::Prim(Primitive::U8));
    let stand_ins = [cx.types.intern(Kind::Value(1)), cx.types.intern(Kind::Slice(byte)), byte];
    let args = scope::type_params(&def.generics.params).map(|generic| match generic.kind {
        crate::body::sig::GenericKind::Const(_) => stand_ins[0],
        _ if relaxed(&def.body, src, &def.generics.predicates, generic) => stand_ins[1],
        _ => stand_ins[2],
    });
    let ty = cx.types.intern(Kind::Adt(item, args.collect()));
    cx.is_unsized(ty)
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

pub(crate) fn detached<T>(cx: &mut Context<'_>, item: ItemRef, run: impl FnOnce(&mut Infer<'_, '_>) -> Result<T>) -> Result<T> {
    let (body, sig) = (Body::default(), Signature::default());
    let src = &cx.file(item).source;
    let home = cx.module_of(item)?;
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    run(&mut infer)
}
