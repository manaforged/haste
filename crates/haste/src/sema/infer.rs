mod adt;
mod adt_check;
mod alloc;
mod array;
mod assign;
mod assoc;
mod call;
mod callable;
mod clone;
mod closure;
mod coerce;
mod consts;
mod deref;
mod drop;
mod expr;
mod looping;
mod matching;
mod method;
mod name;
mod object;
mod opaque;
mod pat;
mod path;
mod pattern;
pub(super) mod poly;
mod project;
pub(super) mod scope;
mod select;
mod stmt;
mod tables;
mod unify;
mod zonk;

pub use adt::{Fields, field_types, maybe_unsized};
pub(in crate::sema) use adt::{body_relaxes_sized, param_relaxes_sized};
pub(in crate::sema) use adt_check::{check_union_concrete_fields, check_union_fields};
pub use alloc::GlobalAllocShim;
pub(in crate::sema) use callable::callable_method_name;
pub(crate) use callable::is_callable_lang;
pub use clone::cloned_tuple;
pub use closure::{By, Capture, Closure, ClosureBody, ClosureInfo, by_reference, register};
pub use coerce::Coercion;
pub use consts::{Anon, Arena};
pub use drop::{Glue, drop_fn, glue_call, has_drop_impl, needs_drop};
pub(in crate::sema) use drop::{mark_parent_laid, record_parent_drops};
pub use looping::{For, Tried};
pub use method::Adjust;
pub use name::type_name;
pub use object::{ALIGN_SLOT, SIZE_SLOT, Vtable, virtual_slot};
pub use pat::{position, rest_split};
pub use path::{parse_int, segments};
pub use scope::{Form, header_bound_traits, type_params};
pub use tables::{BodyTables, LazyTables, SiteKey, SyntaxTables, Typed};
pub(super) use tables::{CallSites, recount_call_sites, shape_calls};

use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, bail};

use crate::body::sig::Signature;
use crate::body::{Body, ExprId, PatId};
use crate::diag::{Diagnostic, DiagnosticKind, Span};
use crate::names::{ItemRef, ModRef, Primitive};
use crate::outline::source::Source;

use super::Context;
use super::substs::Substs;
use super::ty::{Kind, Mut, Ty, Types};

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
    Asm(Vec<Res>),
    GlobalAlloc(GlobalAllocShim),
}

pub(super) type Name<'a> = (&'a str, u32);
type LocalObligations = Box<[(ItemRef, Vec<Ty>)]>;
type LocalSelectionKey = (ItemRef, Ty, Substs, Option<u32>, bool);
type LocalSelectionValue = (Option<(super::impls::ImplRef, Substs)>, LocalObligations);
type LocalProjectionKey = (ItemRef, Ty, Substs, crate::names::Sym, Substs, Option<u32>);
type LocalProjectionValue = (Option<Ty>, LocalObligations);

/// The inference of one body.
pub struct Infer<'c, 'a> {
    pub(in crate::sema) cx: &'c mut Context<'a>,
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
    var_diverges: Vec<bool>,
    var_trail: Vec<(u32, Option<Ty>)>,
    diverge_trail: Vec<(u32, u32)>,
    trail_depth: u32,
    cache_min_var: Vec<usize>,
    var_epoch: u64,
    capture_epoch: u64,
    local_cache_epoch: u64,
    local_closed_cache_epoch: u64,
    local_cache_var_limit: usize,
    metadata: Vec<(Ty, Ty)>,
    obligations: Vec<(ItemRef, Vec<Ty>)>,
    projections: Vec<project::Projection>,
    opaques: Vec<opaque::Opaque>,
    operands: Vec<(ExprId, crate::body::BinOp, (Ty, Ty), Ty)>,
    casts: Vec<(ExprId, Ty, Ty)>,
    deferred_offsets: Vec<(ExprId, Ty, crate::body::List)>,
    param_origins: Vec<(u32, &'a str)>,
    receiver_param: Option<&'a str>,
    try_blocks: Vec<(ExprId, Ty)>,
    awaits: Vec<(ExprId, Ty)>,
    ret_sig: Option<Ty>,
    out: zonk::InferOut,
    scopes: Vec<(Name<'a>, u32)>,
    items: Vec<(&'a str, ItemRef)>,
    block_uses: Vec<u32>,
    loops: Vec<Breaks>,
    uses: Vec<closure::Captures>,
    unsure: Vec<(ExprId, closure::Captures, bool, Vec<u32>)>,
    diverges: bool,
    hidden: Vec<(u32, Ty)>,
    const_args: Vec<(String, (u128, (u32, bool)))>,
    selecting: Vec<u64>,
    assoc_stack: Vec<(ItemRef, Ty, Substs, crate::body::PathId, Option<Ty>)>,
    cycle_hits: u32,
    frames: usize,
    evaluating: Vec<ItemRef>,
    proved: rustc_hash::FxHashSet<(ItemRef, Substs)>,
    imports: HashMap<
        (ItemRef, Option<crate::names::Sym>, Option<u32>, Vec<u32>),
        Vec<crate::names::PerNs>,
    >,
    local_selections: HashMap<LocalSelectionKey, LocalSelectionValue>,
    local_closed_selections: HashMap<LocalSelectionKey, LocalSelectionValue>,
    local_projections: HashMap<LocalProjectionKey, LocalProjectionValue>,
    local_closed_projections: HashMap<LocalProjectionKey, LocalProjectionValue>,
    method_hints: Option<std::sync::Arc<HashMap<ExprId, ItemRef>>>,
    pub(super) poly: Option<Box<poly::PolyState>>,
    body_src: &'a Source,
    current_expr: Option<ExprId>,
    current_pat: Option<PatId>,
    obligation_sites: HashMap<(ItemRef, Ty), ExprId>,
    captures_settled: bool,
    auto_stack: Vec<(ItemRef, Ty)>,
    auto_provisional: HashMap<(ItemRef, Ty), usize>,
    auto_dep: usize,
    matches: Vec<(ExprId, Ty, crate::body::List)>,
    let_pats: Vec<(PatId, Ty)>,
    uninhabited: HashMap<Ty, bool>,
    active_discr: Option<(ItemRef, Vec<Option<i128>>, Vec<bool>)>,
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

/// Infers the body of the function `item` with its generic parameters set to `substs`.
pub fn infer(cx: &mut Context<'_>, item: ItemRef, substs: Substs) -> Result<Typed> {
    infer_with_hints(cx, item, substs, None)
}

pub(super) fn infer_with_hints(
    cx: &mut Context<'_>,
    item: ItemRef,
    substs: Substs,
    method_hints: Option<std::sync::Arc<HashMap<ExprId, ItemRef>>>,
) -> Result<Typed> {
    if let Some(typed) = compiler_made(cx, item, substs)? {
        return Ok(typed);
    }
    infer_after_compiler_made(cx, item, substs, method_hints)
}

pub(super) fn infer_after_compiler_made(
    cx: &mut Context<'_>,
    item: ItemRef,
    substs: Substs,
    method_hints: Option<std::sync::Arc<HashMap<ExprId, ItemRef>>>,
) -> Result<Typed> {
    let sig = cx.signature(item)?;
    let slot = object::virtual_slot(cx, item, substs)?;
    let foreign = cx.foreign(item) || slot.is_some();
    let body = if foreign {
        std::sync::Arc::default()
    } else {
        cx.body(item)?
    };
    let src = cx.body_src(item);
    let home = cx.module_of(item)?;
    let root = Some(item.body_owner());
    let args = cx.types.substs(substs).to_vec();
    let mut typed = {
        let mut infer = Box::new(Infer::start(cx, (&body, &sig, src), home));
        (infer.root, infer.instance, infer.method_hints) =
            (root, Some((item, substs)), method_hints);
        let mut frame = infer.frame(item, &args)?;
        (frame.src, frame.root) = (src, root);
        infer.swap_frame(frame);
        infer.out.exprs.resize(body.exprs.len(), Ty::NONE);
        infer.run(foreign)?;
        if !foreign {
            infer.check_const_body_overflow(item)?;
        }
        (*infer).finish()?
    };
    if foreign {
        super::global_asm::bind(cx, item, &mut typed);
    }
    if let Some(slot) = slot {
        typed.shape = Shape::Virtual(slot);
    }
    if !foreign
        && !cx.outline.crates[item.krate as usize].env.fixed
        && matches!(typed.shape, Shape::Body)
    {
        if let Some(block) = cx.owner_impl(item) {
            poly::check_trait_impl_block(cx, block)?;
        }
        super::borrow::check(cx, item, (&body, &sig), src, &typed)?;
    }
    settle(cx, (item, substs), typed)
}

pub(super) fn settle(
    cx: &mut Context<'_>,
    instance: (ItemRef, Substs),
    mut typed: Typed,
) -> Result<Typed> {
    closure::register(cx, instance, &typed);
    if !typed.coercions.is_empty() && object::fill_drops(cx, &mut typed)? {
        closure::register(cx, instance, &typed);
    }
    typed.drops = drop::drop_calls(cx, &typed)?;
    drop::record_parent_drops(cx, instance, &typed);
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

pub(super) fn compiler_made(
    cx: &mut Context<'_>,
    item: ItemRef,
    substs: Substs,
) -> Result<Option<Typed>> {
    if let Some((op, alloc_static)) = cx.global_alloc_shim(item) {
        return alloc::global_alloc_typed(cx, item, op, alloc_static).map(Some);
    }
    if adt::is_ctor(cx, item) {
        return adt::ctor_typed(cx, item, substs).map(Some);
    }
    let slice = cx.types.substs(substs);
    if cx.lang(item) == Some("drop_glue") {
        let Some(&ty) = slice.first() else {
            bail!("drop glue with no type")
        };
        return drop::glue_typed(cx, item, ty).map(Some);
    }
    let Some(&ty) = slice.first() else {
        return Ok(None);
    };
    if cx.lang(item) == Some("clone_fn") && clone::compiler_clone(cx, ty) {
        return clone::clone_typed(cx, item, ty).map(Some);
    }
    if cx.lang(item) == Some("poll")
        && cx
            .ensure_closure(ty)
            .is_some_and(|info| info.closure.coroutine)
    {
        return closure::coroutine_typed(cx, item, ty).map(Some);
    }
    let (once, by_ref, async_once) = (
        cx.call_once() == Some(item),
        closure::by_reference(cx, item).is_some(),
        callable::is_async_call_once(cx, item),
    );
    match cx.types.kind(ty) {
        Kind::Closure(..) if once => closure::closure_typed(cx, ty).map(Some),
        Kind::Closure(..) | Kind::FnDef(..) | Kind::FnPtr(..) if once || by_ref || async_once => {
            closure::untuple_typed(cx, item, substs).map(Some)
        }
        _ => Ok(None),
    }
}

impl<'c, 'a> Infer<'c, 'a> {
    pub(super) fn start(
        cx: &'c mut Context<'a>,
        (body, sig, src): (&'c Body, &'c Signature, &'a Source),
        home: ModRef,
    ) -> Self {
        let (
            generics,
            vars,
            metadata,
            obligations,
            projections,
            opaques,
            operands,
            try_blocks,
            scopes,
            items,
            loops,
            uses,
        ) = Default::default();
        Infer {
            cx,
            body,
            sig,
            src,
            home,
            root: None,
            instance: None,
            generics,
            self_ty: None,
            block: None,
            frame_item: None,
            vars,
            var_diverges: Vec::new(),
            var_trail: Vec::new(),
            diverge_trail: Vec::new(),
            trail_depth: 0,
            cache_min_var: Vec::new(),
            var_epoch: 0,
            capture_epoch: 0,
            local_cache_epoch: 0,
            local_closed_cache_epoch: 0,
            local_cache_var_limit: 0,
            metadata,
            obligations,
            projections,
            opaques,
            operands,
            casts: Vec::new(),
            deferred_offsets: Vec::new(),
            param_origins: Vec::new(),
            receiver_param: None,
            try_blocks,
            awaits: Vec::new(),
            ret_sig: None,
            out: zonk::take_out(),
            scopes,
            items,
            block_uses: Vec::new(),
            loops,
            uses,
            unsure: Vec::new(),
            diverges: false,
            hidden: Vec::new(),
            const_args: Vec::new(),
            selecting: Vec::new(),
            assoc_stack: Vec::new(),
            cycle_hits: 0,
            frames: 0,
            evaluating: Vec::new(),
            proved: rustc_hash::FxHashSet::default(),
            imports: HashMap::default(),
            local_selections: HashMap::default(),
            local_closed_selections: HashMap::default(),
            local_projections: HashMap::default(),
            local_closed_projections: HashMap::default(),
            method_hints: None,
            poly: None,
            body_src: src,
            current_expr: None,
            current_pat: None,
            obligation_sites: HashMap::default(),
            captures_settled: false,
            auto_stack: Vec::new(),
            auto_provisional: HashMap::default(),
            auto_dep: usize::MAX,
            matches: Vec::new(),
            let_pats: Vec::new(),
            uninhabited: HashMap::default(),
            active_discr: None,
        }
    }

    pub(super) fn note_obligation_site(&mut self, of: ItemRef, self_ty: Ty) {
        if let Some(expr) = self.current_expr {
            let resolved = self.resolve(self_ty);
            self.obligation_sites.entry((of, resolved)).or_insert(expr);
            self.obligation_sites.entry((of, self_ty)).or_insert(expr);
        }
    }

    fn diag_span(
        kind: DiagnosticKind,
        span: Option<Span>,
        msg: impl Into<String>,
        label: impl Into<String>,
    ) -> anyhow::Error {
        let mut diag = Diagnostic::new(kind, msg).with_span(span);
        let label = label.into();
        if !label.is_empty() {
            diag = diag.with_primary_label(label);
        }
        diag.into()
    }

    pub(super) fn diag_expr(
        &self,
        kind: DiagnosticKind,
        expr: Option<ExprId>,
        msg: impl Into<String>,
        label: impl Into<String>,
    ) -> anyhow::Error {
        let span = expr
            .or(self.current_expr)
            .and_then(|id| Span::from_expr(self.body_src, self.body, id));
        Self::diag_span(kind, span, msg, label)
    }

    pub(super) fn diag_tok(
        &self,
        kind: DiagnosticKind,
        tok: u32,
        msg: impl Into<String>,
        label: impl Into<String>,
    ) -> anyhow::Error {
        Self::diag_span(kind, Span::from_token(self.body_src, tok), msg, label)
    }

    pub(super) fn diag_pat(
        &self,
        kind: DiagnosticKind,
        pat: PatId,
        msg: impl Into<String>,
        label: impl Into<String>,
    ) -> anyhow::Error {
        Self::diag_span(
            kind,
            Span::from_pat(self.body_src, self.body, pat),
            msg,
            label,
        )
    }

    pub(super) fn diag_arity(
        &self,
        expr: Option<ExprId>,
        want: usize,
        got: usize,
    ) -> anyhow::Error {
        self.diag_expr(
            DiagnosticKind::Type,
            expr,
            format!("call takes {want} arguments, {got} given"),
            "unexpected number of arguments",
        )
    }

    pub(super) fn diag_deref_limit(&self, expr: ExprId, found: Ty) -> anyhow::Error {
        self.diag_expr(
            DiagnosticKind::Type,
            Some(expr),
            format!(
                "reached the recursion limit while dereferencing {}",
                self.show(found)
            ),
            "dereference recursion limit exceeded",
        )
    }

    pub(super) fn diag_not_callable(&self, expr: ExprId, found: Ty) -> anyhow::Error {
        self.diag_expr(
            DiagnosticKind::Type,
            Some(expr),
            format!("calling `{}` is not supported", self.show(found)),
            "call expression requires function",
        )
    }

    pub(super) fn diag_unknown_field(&self, tok: u32, item: ItemRef, name: &str) -> anyhow::Error {
        self.diag_tok(
            DiagnosticKind::Name,
            tok,
            format!("no field `{name}` on `{}`", self.cx.name(item)),
            "unknown field",
        )
    }
}

impl Infer<'_, '_> {
    pub(super) fn run(&mut self, foreign: bool) -> Result<()> {
        self.hidden.clear();
        let declared = match self.sig.ret {
            Some(ty) => self.lower_ty(&self.sig.body, ty)?,
            None => self.cx.types.unit(),
        };
        let ret = if self.sig.is_async {
            self.fresh(Kind::Var)
        } else {
            declared
        };
        self.out.ret = Some(ret);
        self.bind_hidden()?;
        self.ret_sig = self.opaque_fn_sig()?;
        self.bind_params()?;
        let Some(root) = self.body.root else {
            return if foreign {
                Ok(())
            } else {
                bail!("body has no root")
            };
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
        for (index, &(pat, declared)) in sig.params.iter().enumerate() {
            let ty = self.lower_ty(&sig.body, declared)?;
            let own = self.body.fn_params.get(index).copied();
            let local = self.param_pat(&sig.body, pat, ty, own)?;
            if let Some(at) = self.type_param_named(&sig.body, declared) {
                self.param_origins.push((local, self.generics[at].0));
            }
            self.out.params.push(local);
        }
        Ok(())
    }

    fn type_param_named(&self, arena: &Body, ty: crate::body::TyId) -> Option<usize> {
        match *arena.ty(ty) {
            crate::body::Ty::ImplTrait(at, _) => {
                let key = self.src.text(at);
                self.generics
                    .iter()
                    .position(|(generic, _)| scope::same_param(generic, key))
            }
            crate::body::Ty::Path(path) => {
                let path = arena.path(path);
                let &[segment] = arena.list(path.segments) else {
                    return None;
                };
                let segment = &arena.segments[segment as usize];
                if path.qself.is_some() || segment.args.is_some() {
                    return None;
                }
                let name = self.src.text(segment.name);
                self.generics
                    .iter()
                    .position(|(generic, _)| *generic == name)
            }
            _ => None,
        }
    }

    fn param_pat(&mut self, arena: &Body, pat: PatId, ty: Ty, own: Option<PatId>) -> Result<u32> {
        match *arena.pat(pat) {
            crate::body::Pat::Bind {
                name,
                sub: None,
                by_ref: false,
                ..
            } => {
                let local = self.new_local(ty);
                self.scope_name(name, local);
                Ok(local)
            }
            crate::body::Pat::Wild => Ok(self.new_local(ty)),
            _ => {
                let Some(own) = own else {
                    bail!("a parameter pattern outside the body's arena")
                };
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
}

/// The discriminant of each variant of the enum `item`, whose tags are integers of type `primitive`.
pub fn discriminants(
    cx: &mut Context<'_>,
    item: ItemRef,
    primitive: Primitive,
) -> Result<Vec<i128>> {
    detached(cx, item, |infer| infer.discriminants(item, primitive))
}

#[cfg(feature = "diff-harness")]
pub use zonk::is_case_sig;
pub use zonk::{is_concrete_fn, is_copy, test_sig};

/// The return type of the non-generic function `item`.
pub fn return_type(cx: &mut Context<'_>, item: ItemRef) -> Result<Ty> {
    detached(cx, item, |infer| Ok(infer.fn_types(item, &[])?.1))
}

pub(crate) fn detached<T>(
    cx: &mut Context<'_>,
    item: ItemRef,
    run: impl FnOnce(&mut Infer<'_, '_>) -> Result<T>,
) -> Result<T> {
    let (body, sig) = (Body::default(), Signature::default());
    let src = &cx.file(item).source;
    let home = cx.module_of(item)?;
    let mut infer = Infer::start(cx, (&body, &sig, src), home);
    run(&mut infer)
}

pub(super) fn detached_summary<T>(
    cx: &mut Context<'_>,
    item: ItemRef,
    run: impl FnOnce(&mut Infer<'_, '_>) -> Result<T>,
) -> Result<T> {
    static EMPTY_BODY: std::sync::OnceLock<Body> = std::sync::OnceLock::new();
    static EMPTY_SIG: std::sync::OnceLock<Signature> = std::sync::OnceLock::new();
    static EMPTY_SRC: std::sync::OnceLock<crate::outline::source::Source> =
        std::sync::OnceLock::new();
    let body = EMPTY_BODY.get_or_init(Body::default);
    let sig = EMPTY_SIG.get_or_init(Signature::default);
    let src = EMPTY_SRC.get_or_init(|| crate::outline::source::Source {
        path: std::path::PathBuf::new(),
        text: Default::default(),
        tokens: Default::default(),
        marks: Vec::new(),
        opaque: Vec::new(),
        base: None,
        offset: 0,
        outputs: Default::default(),
        origins: crate::outline::source::Origins::Read,
        line_starts: std::sync::OnceLock::new(),
    });
    let home = cx.module_of(item).unwrap_or(ModRef {
        krate: item.krate,
        module: 0,
    });
    let mut infer = Infer::start(cx, (body, sig, src), home);
    infer.root = item.declaring_body();
    infer.frame_item = Some(item);
    run(&mut infer)
}
