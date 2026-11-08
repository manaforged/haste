mod adt;
mod assoc;
mod array;
mod expr;
mod matching;
mod looping;
mod method;
mod pat;
mod path;
mod pattern;
mod scope;
mod drop;
mod select;
mod stmt;

pub use looping::{For, Tried};
pub use method::Adjust;
pub use pat::{Mode, position, rest_split};
pub use path::{parse_int, segments};

use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, bail};

use crate::body::sig::Signature;
use crate::body::{Body, ExprId, PatId};
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
}

/// One local variable of a body.
#[derive(Clone, Copy, Debug)]
pub struct Local {
    pub ty: Ty,
    pub mutable: bool,
}

/// The inferred types and resolutions of one function body at one set of type arguments.
#[derive(Debug, Default)]
pub struct Typed {
    pub exprs: Vec<Ty>,
    pub res: HashMap<ExprId, Res>,
    pub locals: Vec<Local>,
    pub binds: HashMap<PatId, u32>,
    pub adjusts: HashMap<ExprId, Adjust>,
    /// The expressions whose pointer to an array of this length coerces to a pointer to a slice.
    pub unsize: HashMap<ExprId, u64>,
    pub fields: HashMap<ExprId, u32>,
    pub field_pats: HashMap<PatId, u32>,
    pub pat_res: HashMap<PatId, Res>,
    pub pat_derefs: HashMap<PatId, u32>,
    pub ref_binds: HashMap<PatId, crate::sema::ty::Mut>,
    pub pats: HashMap<PatId, Ty>,
    pub params: Vec<u32>,
    pub ret: Option<Ty>,
    pub fors: HashMap<ExprId, For>,
    pub tries: HashMap<ExprId, Tried>,
}

impl Typed {
    /// Every function the body calls, with its type arguments.
    pub fn calls(&self) -> impl Iterator<Item = (ItemRef, &[Ty])> {
        let loops = self.fors.values().flat_map(|each| [&each.into_iter, &each.next]);
        let tries = self.tries.values().flat_map(|each| [&each.branch, &each.from_residual]);
        self.res.values().chain(loops).chain(tries).filter_map(|res| match res {
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
    home: ModRef,
    generics: Vec<(&'a str, Ty)>,
    self_ty: Option<Ty>,
    block: Option<super::impls::ImplRef>,
    vars: Vec<Option<Ty>>,
    metadata: Vec<(Ty, Ty)>,
    out: Typed,
    scopes: Vec<(&'a str, u32)>,
    items: Vec<(&'a str, ItemRef)>,
    loops: Vec<Option<Ty>>,
    diverges: bool,
}

/// Infers the body of the function `item` with its generic parameters set to `substs`.
pub fn infer(cx: &mut Context<'_>, item: ItemRef, substs: &[Ty]) -> Result<Typed> {
    let sig = cx.signature(item)?;
    let foreign = cx.foreign(item);
    let body = if foreign { std::rc::Rc::default() } else { cx.body(item)? };
    let src = cx.body_src(item);
    let home = cx.module_of(item)?;
    let root = Some(item.outer().map_or(item, |(outer, _)| outer));
    let mut infer = Infer { cx, body: &body, sig: &sig, src, root, home, generics: Vec::new(), self_ty: None, block: None, vars: Vec::new(), metadata: Vec::new(), out: Typed::default(), scopes: Vec::new(), items: Vec::new(), loops: Vec::new(), diverges: false };
    let mut frame = infer.frame(item, substs)?;
    frame.src = src;
    infer.swap_frame(frame);
    infer.out.exprs = vec![Ty(u32::MAX); body.exprs.len()];
    infer.run(foreign)?;
    Ok(infer.finish())
}

impl Infer<'_, '_> {
    fn run(&mut self, foreign: bool) -> Result<()> {
        let ret = match self.sig.ret {
            Some(ty) => self.lower_ty(&self.sig.body, ty)?,
            None => self.cx.types.unit(),
        };
        self.out.ret = Some(ret);
        let sig = self.sig;
        if let Some((kind, mutable)) = sig.receiver {
            let ty = self.receiver_ty(&sig.body, kind)?;
            let local = self.new_local(ty, mutable);
            self.scopes.push(("self", local));
            self.out.params.push(local);
        }
        for &(pat, ty) in &sig.params {
            let ty = self.lower_ty(&sig.body, ty)?;
            let local = self.param_pat(pat, ty)?;
            self.out.params.push(local);
        }
        let Some(root) = self.body.root else {
            return if foreign { Ok(()) } else { bail!("body has no root") };
        };
        let found = self.expr(root, Some(ret))?;
        self.coerce_expr(root, found, ret)
    }

    fn param_pat(&mut self, pat: PatId, ty: Ty) -> Result<u32> {
        match *self.sig.body.pat(pat) {
            crate::body::Pat::Bind { name, mutable, sub: None, by_ref: false } => {
                let local = self.new_local(ty, mutable);
                self.scopes.push((self.src.text(name), local));
                Ok(local)
            }
            crate::body::Pat::Wild => Ok(self.new_local(ty, false)),
            _ => bail!("parameter patterns other than names are not supported yet"),
        }
    }

    fn new_local(&mut self, ty: Ty, mutable: bool) -> u32 {
        self.out.locals.push(Local { ty, mutable });
        self.out.locals.len() as u32 - 1
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
        match self.cx.types.kind(ty) {
            Kind::Tuple(items) | Kind::Adt(_, items) | Kind::FnDef(_, items) => items.iter().any(|item| self.occurs(var, *item)),
            Kind::Ref(_, inner) | Kind::Ptr(_, inner) | Kind::Array(inner, _) | Kind::Slice(inner) | Kind::Pat(inner, ..) => self.occurs(var, *inner),
            Kind::FnPtr(params, ret) => params.iter().any(|param| self.occurs(var, *param)) || self.occurs(var, *ret),
            _ => false,
        }
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
            (Kind::Array(x, m), Kind::Array(y, n)) if m == n => vec![(*x, *y)],
            (Kind::Slice(x), Kind::Slice(y)) => vec![(*x, *y)],
            (Kind::Pat(x, s, e), Kind::Pat(y, t, f)) if (s, e) == (t, f) => vec![(*x, *y)],
            (Kind::Adt(i, x), Kind::Adt(j, y)) | (Kind::FnDef(i, x), Kind::FnDef(j, y)) if i == j && x.len() == y.len() => x.iter().copied().zip(y.iter().copied()).collect(),
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
        self.settle_metadata()
    }

    /// Coerces the value of `expr`, of type `found`, to `expected`, recording an array-to-slice unsizing.
    pub fn coerce_expr(&mut self, expr: ExprId, found: Ty, expected: Ty) -> Result<()> {
        match self.unsize(found, expected)? {
            Some(len) => {
                self.out.unsize.insert(expr, len);
                Ok(())
            }
            None => self.coerce(found, expected),
        }
    }

    fn unsize(&mut self, found: Ty, expected: Ty) -> Result<Option<u64>> {
        let (found, expected) = (self.resolve(found), self.resolve(expected));
        let (Kind::Ref(from, a) | Kind::Ptr(from, a), Kind::Ref(to, b) | Kind::Ptr(to, b)) = (self.cx.types.kind(found).clone(), self.cx.types.kind(expected).clone()) else { return Ok(None) };
        let (a, b) = (self.resolve(a), self.resolve(b));
        let (Kind::Array(element, len), Kind::Slice(target)) = (self.cx.types.kind(a).clone(), self.cx.types.kind(b).clone()) else { return Ok(None) };
        if from == Mut::Not && to == Mut::Mut {
            bail!("cannot coerce a shared pointer to a mutable one");
        }
        self.unify(element, target)?;
        Ok(Some(len))
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
        let items = |infer: &mut Self, items: &[Ty]| items.iter().map(|item| infer.zonk(*item)).collect::<Vec<_>>();
        let kind = match kind {
            Kind::Tuple(list) => Kind::Tuple(items(self, &list)),
            Kind::Ref(m, inner) => Kind::Ref(m, self.zonk(inner)),
            Kind::Ptr(m, inner) => Kind::Ptr(m, self.zonk(inner)),
            Kind::Array(inner, len) => Kind::Array(self.zonk(inner), len),
            Kind::Slice(inner) => Kind::Slice(self.zonk(inner)),
            Kind::Adt(item, args) => Kind::Adt(item, items(self, &args)),
            Kind::FnDef(item, args) => Kind::FnDef(item, items(self, &args)),
            Kind::FnPtr(params, ret) => Kind::FnPtr(items(self, &params), self.zonk(ret)),
            Kind::Pat(base, start, end) => Kind::Pat(self.zonk(base), start, end),
            other => other,
        };
        self.cx.types.intern(kind)
    }

    fn finish(mut self) -> Typed {
        let exprs = std::mem::take(&mut self.out.exprs);
        self.out.exprs = exprs.into_iter().map(|ty| if ty.0 == u32::MAX { ty } else { self.zonk(ty) }).collect();
        let locals = std::mem::take(&mut self.out.locals);
        self.out.locals = locals.into_iter().map(|local| Local { ty: self.zonk(local.ty), ..local }).collect();
        let pats = std::mem::take(&mut self.out.pats);
        self.out.pats = pats.into_iter().map(|(pat, ty)| (pat, self.zonk(ty))).collect();
        let res = std::mem::take(&mut self.out.res);
        self.out.res = res.into_iter().map(|(expr, res)| (expr, self.zonk_res(res))).collect();
        let pat_res = std::mem::take(&mut self.out.pat_res);
        self.out.pat_res = pat_res.into_iter().map(|(pat, res)| (pat, self.zonk_res(res))).collect();
        self.out.ret = self.out.ret.map(|ret| self.zonk(ret));
        let fors = std::mem::take(&mut self.out.fors);
        self.out.fors = fors.into_iter().map(|(expr, each)| (expr, self.zonk_for(each))).collect();
        let tries = std::mem::take(&mut self.out.tries);
        self.out.tries = tries.into_iter().map(|(expr, each)| (expr, self.zonk_try(each))).collect();
        self.out
    }

    fn zonk_for(&mut self, each: For) -> For {
        For { into_iter: self.zonk_res(each.into_iter), next: self.zonk_res(each.next), iter: self.zonk(each.iter), option: self.zonk(each.option), some: each.some }
    }

    fn zonk_try(&mut self, each: Tried) -> Tried {
        Tried { branch: self.zonk_res(each.branch), from_residual: self.zonk_res(each.from_residual), flow: self.zonk(each.flow), ..each }
    }

    fn zonk_res(&mut self, res: Res) -> Res {
        match res {
            Res::Fn(item, substs) => {
                let substs = substs.into_iter().map(|ty| self.zonk(ty)).collect();
                self.trait_call(item, substs)
            }
            Res::Ctor(item, variant, args) => Res::Ctor(item, variant, args.into_iter().map(|ty| self.zonk(ty)).collect()),
            other => other,
        }
    }
}

/// The field types of each variant of the struct, enum or union `item` at `args`.
pub fn field_types(cx: &mut Context<'_>, item: ItemRef, args: &[Ty]) -> Result<Vec<Vec<Ty>>> {
    detached(cx, item, |infer| infer.adt_fields(item, args))
}

/// Whether dropping a value of `ty` runs code.
pub fn needs_drop(cx: &mut Context<'_>, ty: Ty) -> Result<bool> {
    let Some(anchor) = cx.lang_item("drop") else { bail!("core has no `Drop` trait") };
    detached(cx, anchor, |infer| infer.needs_drop(ty))
}

fn detached<T>(cx: &mut Context<'_>, item: ItemRef, run: impl FnOnce(&mut Infer<'_, '_>) -> Result<T>) -> Result<T> {
    let (body, sig) = (Body::default(), Signature::default());
    let src = &cx.file(item).source;
    let home = cx.module_of(item)?;
    let mut infer = Infer { cx, body: &body, sig: &sig, src, root: None, home, generics: Vec::new(), self_ty: None, block: None, vars: Vec::new(), metadata: Vec::new(), out: Typed::default(), scopes: Vec::new(), items: Vec::new(), loops: Vec::new(), diverges: false };
    run(&mut infer)
}
