use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, Function, FunctionBuilder, Type, Value};
use rustc_hash::FxHashMap;

use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::body::{Body, Expr, ExprId, List, Stmt};
use crate::sema::infer::{Arena, Coercion, Res, Shape, Typed};
use crate::sema::substs::Substs;
use crate::sema::reach::Reached;
use crate::sema::table::Table;
use crate::sync::{read_lock, write_lock};

use super::plan::{BodyFootprintPlan, eval_footprint_with_cycle};
use super::reloc::{RelocPlan, RelocStep, SiteOrigin, eval_relocs};
use super::{Inline, Lower, Slots, Word, is_static, signature, signature_of};

/// The 128-bit structural footprint digest of a lowered function and its inlined callees.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FootprintId(pub u128);

/// An interned read-only data blob identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RodataId(pub u32);

/// An interned vtable layout identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VtableId(pub u32);

/// The target of an 8-byte pointer relocation inside a read-only data blob.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RodataTarget {
    Rodata(RodataId),
}

/// One 8-byte pointer relocation inside a read-only data blob.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RodataReloc {
    pub offset: u32,
    pub target: RodataTarget,
}

/// One word inside a symbolic vtable definition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VtableEntry {
    Word(Word),
    Super(VtableId),
}

/// A symbolic relocation entry referenced by a function stencil or dictionary.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum SymbolSource {
    Slot(usize),
    Vtable(VtableId),
    Rodata(RodataId),
    StaticCell { index: usize, size: u32 },
    PromotedCell { owner: usize, seq: u32, size: u32 },
    Host(HostSymbol),
    Word(u64),
    ThreadSlot(usize),
}

/// The ordered symbolic relocation vector (`D`) of one lowered function instance.
pub type RelocVec = Vec<SymbolSource>;

/// The output of lowering one function into a reusable SSA stencil.
pub struct StencilOutput {
    pub func: Function,
    pub relocs: RelocVec,
    pub reloc_plan: Arc<RelocPlan>,
}

/// A cached, simplified callee body ready to splice inline into callers.
pub struct InlinedCallee {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub relocs: RelocVec,
    pub reloc_plan: Arc<RelocPlan>,
    pub body: Arc<Function>,
    pub returns: bool,
    pub height: u8,
    pub size: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum LowerMode {
    Stencil,
    Dict,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TemplateKey {
    item: ItemRef,
    footprint: FootprintId,
    limit: usize,
    low: bool,
}

struct InlineTemplate {
    body: Arc<Function>,
    returns: bool,
    reloc_plan: Arc<RelocPlan>,
    height: u8,
    size: u16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CycleState {
    Acyclic,
    CyclicIncomplete,
    CyclicComplete,
}

impl CycleState {
    fn is_complete(self) -> bool {
        self != Self::CyclicIncomplete
    }
}

const TRANSIENT_CAP: usize = 32_768;
const TRIM_SHARDS: usize = 4;
const MAX_INLINE_HEIGHT: u8 = 10;

pub(super) type InlineMeta = (ItemRef, FootprintId, bool, u8, u16);

/// Cached inlineable callee stencils and footprints across a compilation.
#[derive(Default)]
pub struct Bodies {
    pub(super) plans: Table<(ItemRef, Option<ExprId>), Arc<BodyFootprintPlan>>,
    inlined: Table<usize, Option<Arc<InlinedCallee>>>,
    inline_meta: Table<usize, Option<InlineMeta>>,
    templates: Table<TemplateKey, Option<Arc<InlineTemplate>>>,
    cyclic: RwLock<FxHashMap<usize, CycleState>>,
    transient_entries: AtomicUsize,
    evict_cursor: AtomicUsize,
}

#[derive(Clone, Copy)]
struct NodeInfo {
    index: u32,
    low: u32,
    on_stack: bool,
    self_loop: bool,
    complete: bool,
}

struct TarjanRun {
    nodes: FxHashMap<usize, NodeInfo>,
    stack: Vec<usize>,
    clock: u32,
}

impl TarjanRun {
    fn enter(&mut self, u: usize, (edges, self_loop, complete): (Vec<usize>, bool, bool)) -> (usize, usize, Vec<usize>) {
        self.clock += 1;
        let info = NodeInfo { index: self.clock, low: self.clock, on_stack: true, self_loop, complete };
        self.nodes.insert(u, info);
        self.stack.push(u);
        (u, 0, edges)
    }

    fn finish(&mut self, u: usize, parent: Option<usize>, cyclic: &mut FxHashMap<usize, CycleState>) {
        let u_info = self.nodes[&u];
        if u_info.low == u_info.index {
            self.pop_scc(u, cyclic);
        }
        if let Some(p) = parent {
            let u_now = self.nodes[&u];
            if let Some(p_info) = self.nodes.get_mut(&p) {
                p_info.low = p_info.low.min(u_now.low);
                p_info.complete &= u_now.complete;
            }
        }
    }

    fn pop_scc(&mut self, u: usize, cyclic: &mut FxHashMap<usize, CycleState>) {
        let mut scc = Vec::new();
        let mut self_loop = false;
        let mut complete = true;
        while let Some(w) = self.stack.pop() {
            if let Some(info) = self.nodes.get_mut(&w) {
                info.on_stack = false;
                self_loop |= info.self_loop;
                complete &= info.complete;
            }
            scc.push(w);
            if w == u {
                break;
            }
        }
        let is_cyc = scc.len() > 1 || self_loop;
        let state = match (is_cyc, complete) {
            (true, true) => Some(CycleState::CyclicComplete),
            (true, false) => Some(CycleState::CyclicIncomplete),
            (false, true) => Some(CycleState::Acyclic),
            (false, false) => None,
        };
        for &w in &scc {
            if let Some(st) = state {
                cyclic.insert(w, st);
            }
            if !complete && let Some(info) = self.nodes.get_mut(&w) {
                info.complete = false;
            }
        }
    }
}

impl Bodies {
    /// Drops all cached inlineable bodies and footprints.
    pub fn clear(&self) {
        self.clear_transient();
        self.plans.clear();
        write_lock(&self.cyclic).clear();
    }

    /// Drops cached inline templates while retaining footprint plans and precomputed cycle entries.
    pub fn clear_transient(&self) {
        self.transient_entries.store(0, Ordering::Relaxed);
        self.evict_cursor.store(0, Ordering::Relaxed);
        self.inlined.clear();
        self.inline_meta.clear();
        self.templates.clear();
    }

    /// Evicts a small round-robin slice of shards once transient entries reach the trim watermark.
    pub(crate) fn trim_transient(&self) {
        if self.transient_entries.load(Ordering::Relaxed) < TRANSIENT_CAP / 8 {
            return;
        }
        let start = self.evict_cursor.fetch_add(TRIM_SHARDS, Ordering::Relaxed);
        let mut removed = 0usize;
        for offset in 0..TRIM_SHARDS {
            let shard = (start + offset) % crate::sema::table::SHARDS;
            removed += self.inlined.clear_shard(shard);
            removed += self.inline_meta.clear_shard(shard);
            removed += self.templates.clear_shard(shard);
        }
        if removed > 0 {
            let mut cur = self.transient_entries.load(Ordering::Relaxed);
            while cur > 0 {
                match self.transient_entries.compare_exchange_weak(cur, cur.saturating_sub(removed), Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(next) => cur = next,
                }
            }
        }
    }

    fn bump_transient(&self) {
        self.transient_entries.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn is_cyclic(&self, cx: &mut Context<'_>, reached: &Reached, root: usize) -> bool {
        if let Some(&st) = read_lock(&self.cyclic).get(&root) {
            return st != CycleState::Acyclic;
        }
        if reached.is_untyped(root) {
            return false;
        }
        let mut view = CycleView { shared: &self.cyclic, found: FxHashMap::default() };
        if super::may_inline(cx, reached, root) {
            tarjan_from(cx, reached, &mut view, root);
        } else {
            view.found.insert(root, CycleState::Acyclic);
        }
        let mut shared = write_lock(&self.cyclic);
        for (node, state) in view.found {
            shared.entry(node).or_insert(state);
        }
        shared.get(&root).is_some_and(|&st| st != CycleState::Acyclic)
    }
}

struct CycleView<'b> {
    shared: &'b RwLock<FxHashMap<usize, CycleState>>,
    found: FxHashMap<usize, CycleState>,
}

impl CycleView<'_> {
    fn is_done(&self, v: usize) -> bool {
        let done = |st: &CycleState| st.is_complete();
        self.found.get(&v).is_some_and(done) || read_lock(self.shared).get(&v).is_some_and(done)
    }
}

fn tarjan_from(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, root: usize) {
    let mut run = TarjanRun { nodes: FxHashMap::default(), stack: Vec::new(), clock: 0 };
    let mut frames = vec![run.enter(root, node_edges(cx, reached, cyclic, root))];
    while let Some(top) = frames.last_mut() {
        let u = top.0;
        if top.1 < top.2.len() {
            let v = top.2[top.1];
            top.1 += 1;
            step_edge(cx, reached, cyclic, (&mut run, &mut frames), (u, v));
        } else if let Some((done, _, _)) = frames.pop() {
            run.finish(done, frames.last().map(|f| f.0), &mut cyclic.found);
        }
    }
}

type Frames = Vec<(usize, usize, Vec<usize>)>;

fn step_edge(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, (run, frames): (&mut TarjanRun, &mut Frames), (u, v): (usize, usize)) {
    if cyclic.is_done(v) {
        return;
    }
    if let Some(&v_info) = run.nodes.get(&v) {
        if let Some(u_info) = run.nodes.get_mut(&u) {
            if v_info.on_stack {
                u_info.low = u_info.low.min(v_info.index);
            } else {
                u_info.complete &= v_info.complete;
            }
        }
        return;
    }
    frames.push(run.enter(v, node_edges(cx, reached, cyclic, v)));
}

fn node_edges(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, u: usize) -> (Vec<usize>, bool, bool) {
    let mut info = NodeInfo { index: 0, low: 0, on_stack: false, self_loop: false, complete: true };
    let mut edges = Vec::new();
    let closure = crate::profile() == crate::outline::tree::Profile::Release && matches!(&reached.list[u].1, Ok(typed) if matches!(typed.shape, Shape::Closure(_)));
    if let Some(slice) = reached.direct_calls(u).filter(|_| !closure) {
        for &v in slice {
            scan_callee(cx, reached, cyclic, (u, v), (&mut info, &mut edges));
        }
        return (edges, info.self_loop, info.complete);
    }
    let Ok(typed) = &reached.list[u].1 else { return (edges, false, true) };
    let calls = closure.then(|| closure_calls(cx, typed)).flatten().unwrap_or_else(|| typed.calls().collect());
    for (item, substs) in calls {
        if cx.intrinsic(item).is_some() {
            continue;
        }
        match reached.index.get(&(item, substs)) {
            Some(&v) => scan_callee(cx, reached, cyclic, (u, v), (&mut info, &mut edges)),
            None => info.complete = false,
        }
    }
    (edges, info.self_loop, info.complete)
}

/// The functions a closure's own body calls; its typed data is its parent's, whose calls hold those of every closure in the parent's body, so each table is read at the closure's own expressions.
fn closure_calls(cx: &mut Context<'_>, typed: &Typed) -> Option<Vec<(ItemRef, Substs)>> {
    let typed = typed.scoped(&cx.types);
    let Shape::Closure(closure) = &typed.shape else { return None };
    let body = cx.body(closure.parent).ok()?;
    let own = own_exprs(&body, closure.closure.body);
    let mut calls: Vec<&Res> = Vec::new();
    for expr in &own {
        calls.extend([&typed.res, &typed.operators, &typed.try_blocks, &typed.box_frees].into_iter().filter_map(|map| map.get(expr)));
        calls.extend(typed.fors.get(expr).into_iter().flat_map(|each| [&each.into_iter, &each.next]));
        calls.extend(typed.tries.get(expr).into_iter().flat_map(|each| [&each.branch, &each.from_residual]));
        calls.extend(typed.awaits.get(expr).into_iter().flat_map(|each| [&each.into_future, &each.poll]));
        calls.extend(typed.coercions.get(expr).into_iter().flat_map(coercion_calls));
    }
    let derefs = typed.derefs.iter().filter(|((expr, _), _)| own.binary_search(expr).is_ok()).map(|(_, res)| res);
    let shared = typed.pat_res.values().chain(&typed.drops);
    let all = calls.into_iter().chain(derefs).chain(shared);
    Some(all.filter_map(|res| if let Res::Fn(item, substs) = res { Some((*item, *substs)) } else { None }).collect())
}

/// The functions a coercion calls.
fn coercion_calls(coercion: &Coercion) -> Vec<&Res> {
    match coercion {
        Coercion::Vtable(vtable) => vtable.calls(),
        Coercion::Reify(res) => vec![res],
        Coercion::Deref(steps, _) => steps.iter().filter_map(|(_, res)| res.as_ref()).collect(),
        Coercion::Len(_) | Coercion::Autos | Coercion::Upcast(_) => Vec::new(),
    }
}

/// The expressions of `body` in the tree under `root` outside any closure nested there, in increasing order.
fn own_exprs(body: &Body, root: ExprId) -> Vec<ExprId> {
    let mut own = Vec::new();
    let mut todo = vec![root];
    while let Some(expr) = todo.pop() {
        own.push(expr);
        children(body, expr, &mut todo);
    }
    own.sort_unstable();
    own.dedup();
    own
}

/// Pushes onto `out` the expressions directly under `expr`, except the body of a closure.
fn children(body: &Body, expr: ExprId, out: &mut Vec<ExprId>) {
    let list = |list: List| body.list(list).iter().map(|&at| ExprId(at));
    match *body.expr(expr) {
        Expr::Unary(_, a) | Expr::Field(a, _) | Expr::TupleField(a, _) | Expr::Cast(a, _) | Expr::Ref { expr: a, .. } | Expr::Let(_, a) | Expr::Loop { body: a, .. } | Expr::Try(a) | Expr::Await(a) | Expr::Become(a) | Expr::Paren(a) => out.push(a),
        Expr::Binary(_, a, b) | Expr::Assign(a, b) | Expr::AssignOp(_, a, b) | Expr::Index(a, b) | Expr::Repeat(a, b) | Expr::While { cond: a, body: b, .. } | Expr::For { iter: a, body: b, .. } => out.extend([a, b]),
        Expr::Call(callee, args) => out.extend(std::iter::once(callee).chain(list(args))),
        Expr::Method { receiver, params, .. } => out.extend(std::iter::once(receiver).chain(list(params))),
        Expr::Tuple(items) | Expr::Array(items) => out.extend(list(items)),
        Expr::If(cond, then, otherwise) => out.extend([cond, then].into_iter().chain(otherwise)),
        Expr::Range { from, to, .. } => out.extend(from.into_iter().chain(to)),
        Expr::Return(value) | Expr::Break(_, value) | Expr::Yield(value) => out.extend(value),
        Expr::Struct { fields, base, .. } => out.extend(body.list(fields).iter().map(|&at| body.fields[at as usize].value).chain(base)),
        Expr::Match(scrutinee, arms) => out.extend(std::iter::once(scrutinee).chain(body.list(arms).iter().flat_map(|&at| body.arms[at as usize].guard.into_iter().chain([body.arms[at as usize].body])))),
        Expr::Block { stmts, tail, .. } => out.extend(body.list(stmts).iter().flat_map(|&at| stmt_exprs(body.stmts[at as usize])).chain(tail)),
        Expr::Closure { .. } | Expr::Lit(..) | Expr::Path(_) | Expr::Continue(_) | Expr::Macro { .. } | Expr::Underscore | Expr::FormatArgs(_) | Expr::OffsetOf { .. } => {}
    }
}

/// The expressions directly under statement `stmt`.
fn stmt_exprs(stmt: Stmt) -> Vec<ExprId> {
    match stmt {
        Stmt::Let { init, otherwise, .. } => init.into_iter().chain(otherwise).collect(),
        Stmt::Expr(expr, _) => vec![expr],
        Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => Vec::new(),
    }
}

fn scan_callee(cx: &mut Context<'_>, reached: &Reached, cyclic: &mut CycleView<'_>, (u, v): (usize, usize), (info, edges): (&mut NodeInfo, &mut Vec<usize>)) {
    if v == u {
        info.self_loop |= !matches!(&reached.list[u].1, Ok(typed) if matches!(typed.shape, Shape::Closure(_)));
        return;
    }
    if cyclic.is_done(v) {
        return;
    }
    if reached.is_untyped(v) {
        info.complete = false;
        return;
    }
    if !super::may_inline(cx, reached, v) {
        cyclic.found.insert(v, CycleState::Acyclic);
        return;
    }
    edges.push(v);
}

pub(super) fn item_bits(item: ItemRef) -> u128 {
    (u128::from(item.krate) << 64) | (u128::from(item.file) << 32) | u128::from(item.item)
}

/// Lowers reached function `index` into an SSA stencil with symbolic relocations.
pub fn lower_stencil(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<StencilOutput> {
    let (func, relocs, reloc_plan, _) = run_with_cycle(cx, (reached, slots), &mut Inline::default(), index, LowerMode::Stencil)?;
    Ok(StencilOutput { func, relocs, reloc_plan })
}

/// Lowers reached function `index` into a shared dictionary-parameterized SSA body and its relocation vector.
pub fn lower_dict(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) -> Result<(Function, RelocVec, Arc<RelocPlan>)> {
    let (func, relocs, reloc_plan, _) = run_with_cycle(cx, (reached, slots), &mut Inline::default(), index, LowerMode::Dict)?;
    Ok((func, relocs, reloc_plan))
}

fn record_meta_height(inline: &mut Inline, meta: Option<(ItemRef, FootprintId, bool, u8, u16)>) -> Option<(ItemRef, FootprintId, bool)> {
    meta.map(|(item, fp, returns, h, sz)| {
        inline.max_callee_height = inline.max_callee_height.max(h);
        inline.max_callee_size = inline.max_callee_size.max(sz);
        (item, fp, returns)
    })
}

fn record_callee_height(inline: &mut Inline, callee: Option<Arc<InlinedCallee>>) -> Option<Arc<InlinedCallee>> {
    if let Some(c) = &callee {
        inline.max_callee_height = inline.max_callee_height.max(c.height);
        inline.max_callee_size = inline.max_callee_size.max(c.size);
    }
    callee
}

fn run_inline_probe(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, limit: usize) -> Result<(Option<InlineHit>, usize, usize, Quiet)> {
    let (depth, quiet) = (inline.stack.len(), Quiet::mark(cx, slots));
    inline.stack.push(index);
    let outcome = ensure_inline_template(cx, (reached, slots), inline, index, (limit, &quiet));
    inline.pop(depth);
    let (tmpl, cycle) = outcome?;
    Ok((tmpl, cycle, depth, quiet))
}

pub(super) fn resolve_inline_meta(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, limit: usize) -> Result<(Option<(ItemRef, FootprintId, bool)>, usize)> {
    let bodies = slots.bodies();
    if bodies.is_cyclic(cx, reached, index) {
        return Ok((None, usize::MAX));
    }
    let standard = super::expr::inline_limit(cx, reached, index) == Some(limit);
    if let Some(cached) = bodies.inlined.get(&index).filter(|c| c.is_some() || standard) {
        return Ok((record_meta_height(inline, cached.map(|c| (c.item, c.footprint, c.returns, c.height, c.size))), usize::MAX));
    }
    if let Some(cached) = bodies.inline_meta.get(&index).filter(|c| c.is_some() || standard) {
        return Ok((record_meta_height(inline, cached), usize::MAX));
    }
    if let Some(pos) = inline.stack.iter().position(|&at| at == index) {
        return Ok((None, pos));
    }
    if let Some((cached, cycle)) = inline.local.get(&index).filter(|(c, _)| c.is_some() || standard) {
        let (meta, cyc) = (cached.as_ref().map(|c| (c.item, c.footprint, c.returns, c.height, c.size)), *cycle);
        return Ok((record_meta_height(inline, meta), cyc));
    }
    if let Some(&(meta, cyc)) = inline.local_meta.get(&index).filter(|(m, _)| m.is_some() || standard) {
        return Ok((record_meta_height(inline, meta), cyc));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) = run_inline_probe(cx, (reached, slots), inline, index, limit)?;
    let meta = tmpl.map(|(footprint, t, _)| (item, footprint, t.returns, t.height, t.size));
    if standard {
        if cycle > depth && quiet.holds(cx, slots) {
            bodies.bump_transient();
            let stored = bodies.inline_meta.get_or_insert_with(index, || meta);
            return Ok((record_meta_height(inline, stored), cycle));
        }
        inline.local_meta.insert(index, (meta, if cycle < depth { cycle } else { usize::MAX }));
    }
    Ok((record_meta_height(inline, meta), cycle))
}

pub(super) fn resolve_inline(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, limit: usize) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let bodies = slots.bodies();
    if bodies.is_cyclic(cx, reached, index) {
        return Ok((None, usize::MAX));
    }
    let standard = super::expr::inline_limit(cx, reached, index) == Some(limit);
    if let Some(cached) = bodies.inlined.get(&index).filter(|c| c.is_some() || standard) {
        return Ok((record_callee_height(inline, cached), usize::MAX));
    }
    if let Some(pos) = inline.stack.iter().position(|&at| at == index) {
        return Ok((None, pos));
    }
    if let Some((cached, cycle)) = inline.local.get(&index).filter(|(c, _)| c.is_some() || standard) {
        let (hit, cyc) = (cached.clone(), *cycle);
        return Ok((record_callee_height(inline, hit), cyc));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) = run_inline_probe(cx, (reached, slots), inline, index, limit)?;
    let result = match tmpl {
        Some((footprint, t, Some(relocs))) => Some(Arc::new(InlinedCallee { item, footprint, relocs, reloc_plan: Arc::clone(&t.reloc_plan), body: Arc::clone(&t.body), returns: t.returns, height: t.height, size: t.size })),
        Some((footprint, t, None)) => {
            let evaluated = quiet.lowered(slots, eval_relocs(cx, reached, slots, &t.reloc_plan, index))?;
            evaluated.map(|relocs| Arc::new(InlinedCallee { item, footprint, relocs, reloc_plan: Arc::clone(&t.reloc_plan), body: Arc::clone(&t.body), returns: t.returns, height: t.height, size: t.size }))
        }
        None => None,
    };
    if standard {
        if cycle > depth && quiet.holds(cx, slots) {
            bodies.bump_transient();
            let stored = bodies.inlined.get_or_insert_with(index, || result);
            return Ok((record_callee_height(inline, stored), cycle));
        }
        inline.local.insert(index, (result.clone(), if cycle < depth { cycle } else { usize::MAX }));
    }
    Ok((record_callee_height(inline, result), cycle))
}

struct Quiet {
    wanted: usize,
    sema: (usize, usize),
}

impl Quiet {
    fn mark(cx: &Context<'_>, slots: &dyn Slots) -> Self {
        Self { wanted: slots.wanted_len(), sema: cx.quiet_mark() }
    }

    fn holds(&self, cx: &Context<'_>, slots: &dyn Slots) -> bool {
        slots.wanted_len() == self.wanted && cx.quiet_mark() == self.sema
    }

    fn lowered<T>(&self, slots: &dyn Slots, result: Result<T>) -> Result<Option<T>> {
        match result {
            Ok(out) => Ok(Some(out)),
            Err(err) if slots.wanted_len() > self.wanted => Err(err),
            Err(_) => Ok(None),
        }
    }
}

type InlineHit = (FootprintId, Arc<InlineTemplate>, Option<RelocVec>);

fn ensure_inline_template(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, quiet): (usize, &Quiet)) -> Result<(Option<InlineHit>, usize)> {
    let saved_h = std::mem::replace(&mut inline.max_callee_height, 0);
    let saved_sz = std::mem::replace(&mut inline.max_callee_size, 0);
    let res = ensure_inline_template_inner(cx, (reached, slots), inline, index, (limit, quiet));
    inline.max_callee_height = saved_h;
    inline.max_callee_size = saved_sz;
    res
}

fn ensure_inline_template_inner(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, quiet): (usize, &Quiet)) -> Result<(Option<InlineHit>, usize)> {
    let item = reached.list[index].0.0;
    let fp_res = eval_footprint_with_cycle(cx, (reached, slots), inline, index);
    let fp_h = std::mem::replace(&mut inline.max_callee_height, 0);
    let fp_sz = std::mem::replace(&mut inline.max_callee_size, 0);
    let Some((footprint, fp_cycle)) = quiet.lowered(slots, fp_res)? else { return Ok((None, usize::MAX)) };
    let key = TemplateKey { item, footprint, limit, low: slots.low_entries() };
    if let Some(template) = slots.bodies().templates.get(&key) {
        return Ok((template.map(|t| (footprint, t, None)), fp_cycle));
    }
    let lowered = run_with_cycle(cx, (reached, slots), inline, index, LowerMode::Stencil);
    let ir_h = inline.max_callee_height;
    let ir_sz = inline.max_callee_size;
    let Some((func, relocs, reloc_plan, ir_cycle)) = quiet.lowered(slots, lowered)? else { return Ok((None, fp_cycle)) };
    let cycle = fp_cycle.min(ir_cycle);
    let base_h = fp_h.max(ir_h);
    let base_size = fp_sz.max(ir_sz) as usize;
    let mut template = if reloc_plan.has_promoted() {
        None
    } else {
        simplify_inline(func, limit).and_then(|(body, returns)| {
            let body_size = body.size();
            let grew = (base_size == 0 && body_size > 32) || (base_size > 0 && body_size.saturating_sub(base_size) > 16);
            let height = if grew { base_h.saturating_add(1) } else { base_h };
            let size = body_size.min(u16::MAX as usize) as u16;
            (height <= MAX_INLINE_HEIGHT).then(|| Arc::new(InlineTemplate { body, returns, reloc_plan, height, size }))
        })
    };
    let depth = inline.stack.len().saturating_sub(1);
    if cycle > depth && quiet.holds(cx, slots) {
        slots.bodies().bump_transient();
        template = slots.bodies().templates.get_or_insert_with(key, || template);
    }
    Ok((template.map(|t| (footprint, t, Some(relocs))), cycle))
}

fn simplify_inline(mut func: Function, limit: usize) -> Option<(Arc<Function>, bool)> {
    if func.is_coroutine() {
        return None;
    }
    haste_codegen::simplify(&mut func, crate::profile().into());
    (func.can_inline() && func.size() <= limit).then(|| {
        let returns = func.has_return();
        (Arc::new(func), returns)
    })
}

fn run_with_cycle(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, mode: LowerMode) -> Result<(Function, RelocVec, Arc<RelocPlan>, usize)> {
    let ((item, _), typed) = &reached.list[index];
    let Ok(typed) = typed else { bail!("{} did not type-check", cx.name(*item)) };
    let typed = typed.scoped(&cx.types);
    let typed: &Typed = &typed;
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => *item,
    };
    let arena = match &typed.shape {
        Shape::Body | Shape::Closure(_) => Arena::Body(cx.body(source)?),
        Shape::Anon(anon) => anon.arena.clone(),
        Shape::Glue(_) | Shape::Virtual(_) | Shape::Untuple(_) | Shape::Ctor(_) | Shape::Clone(_) => Arena::Body(Arc::default()),
    };
    let src = cx.body_src(source);
    let overflow_checks = cx.outline.crates[source.krate as usize].env.overflow_checks();
    let mut sig = if is_static(cx, *item) { signature_of(cx, None, [])? } else { signature(cx, typed)? };
    if mode == LowerMode::Dict {
        sig.params.push(AbiParam::special(Type::I64, ArgumentPurpose::Dict));
    }
    let mut func = Function::new(sig);
    let b = FunctionBuilder::new(&mut func);
    let mut relocs = Vec::new();
    let mut reloc_plan = Vec::new();
    let cycle = {
        let mut lower = Lower {
            cx,
            reached,
            slots,
            inline,
            body: arena.body(),
            src,
            typed,
            b,
            locals: vec![None; typed.locals.len()],
            loops: Vec::new(),
            try_exits: Vec::new(),
            open: true,
            dead: None,
            sret: None,
            dict_param: None,
            mode,
            relocs: &mut relocs,
            reloc_plan: &mut reloc_plan,
            promoted_seq: 0,
            cycle_root: usize::MAX,
            owned: Vec::new(),
            flags: Vec::new(),
            moved: Default::default(),
            extended: Default::default(),
            pending: Vec::new(),
            index,
            resume: None,
            overflow_checks,
        };
        lower.run()?;
        lower.cycle_root
    };
    let cycle = if mode == LowerMode::Stencil
        && slots.allow_inline()
        && func.has_vtable_slots()
        && !func.is_coroutine()
    {
        splice_devirt_vtable_calls(cx, (reached, slots), inline, (&mut func, &mut relocs, &mut reloc_plan), cycle)?
    } else {
        cycle
    };
    Ok((func, relocs, Arc::new(RelocPlan::from_vec(reloc_plan)), cycle))
}

fn splice_devirt_vtable_calls(
    cx: &mut Context<'_>,
    (reached, slots): (&Reached, &dyn Slots),
    inline: &mut Inline,
    (func, relocs, reloc_plan): (&mut Function, &mut RelocVec, &mut Vec<RelocStep>),
    mut cycle_root: usize,
) -> Result<usize> {
    let mut any = false;
    for _ in 0..8 {
        haste_codegen::simplify(func, crate::profile().into());
        any = false;
        let current_plan = RelocPlan { steps: reloc_plan.clone().into_boxed_slice() };
        let mut spliced = false;
        for (block, pos, callee_sym) in func.devirt_call_sites() {
            let Some(&SymbolSource::Slot(callee_idx)) = relocs.get(callee_sym as usize) else { continue };
            let Some((expr, slot)) = current_plan.vtable_slot_at(callee_sym as usize) else { continue };
            let Some(limit) = super::expr::inline_limit(cx, reached, callee_idx) else { continue };
            let (hit, cycle) = resolve_inline(cx, (reached, slots), inline, callee_idx, limit)?;
            cycle_root = cycle_root.min(cycle);
            let Some(inlined) = hit else { continue };
            if !inlined.body.can_inline() {
                continue;
            }
            let sym_base = relocs.len() as u32;
            relocs.extend(inlined.relocs.iter().cloned());
            reloc_plan.push(RelocStep::SiteInline { origin: SiteOrigin::VtableMethod { expr, slot }, plan: Arc::clone(&inlined.reloc_plan) });
            func.splice_call(block, pos, &inlined.body, sym_base);
            spliced = true;
            any = true;
            break;
        }
        if !spliced {
            break;
        }
    }
    if any {
        haste_codegen::simplify(func, crate::profile().into());
    }
    Ok(cycle_root)
}

impl Lower<'_, '_> {
    pub(super) fn push_reloc(&mut self, src: SymbolSource, step: RelocStep) -> u32 {
        let idx = self.relocs.len() as u32;
        self.relocs.push(src);
        self.reloc_plan.push(step);
        idx
    }

    pub(super) fn emit_slot(&mut self, src: SymbolSource, origin: SiteOrigin) -> Value {
        self.emit_slot_with(src, RelocStep::SiteSlot(origin))
    }

    pub(super) fn emit_slot_with(&mut self, src: SymbolSource, step: RelocStep) -> Value {
        let self_call = matches!(self.mode, LowerMode::Stencil) && matches!(&src, SymbolSource::Slot(i) if *i == self.index);
        let low = self.slots.low_entries();
        self.emit_reloc(src, step, self_call, low)
    }

    pub(super) fn emit_symbol(&mut self, src: SymbolSource) -> Value {
        let step = RelocStep::Const(src.clone());
        self.emit_symbol_with(src, step)
    }

    pub(super) fn emit_const_rodata(&mut self, bytes: &[u8]) -> Value {
        let id = self.slots.rodata(bytes, &[]);
        let idx = self.relocs.len() as u32;
        let val = self.emit_symbol(SymbolSource::Rodata(id));
        if matches!(self.mode, LowerMode::Stencil) && bytes.len() <= 64 {
            self.b.func.add_rodata_bytes(idx, bytes.to_vec());
        }
        val
    }

    pub(super) fn emit_symbol_with(&mut self, src: SymbolSource, step: RelocStep) -> Value {
        self.emit_reloc(src, step, false, false)
    }

    fn emit_reloc(&mut self, src: SymbolSource, step: RelocStep, self_call: bool, low_slot: bool) -> Value {
        let idx = self.push_reloc(src, step);
        if self_call {
            self.b.func.add_self_slot(idx);
        }
        match self.mode {
            LowerMode::Dict => {
                let dict = self.dict_param.expect("dictionary parameter is set");
                self.ins().load(Type::I64, dict, (idx * 8) as i32)
            }
            LowerMode::Stencil if low_slot => self.ins().slot_addr(idx),
            LowerMode::Stencil => self.ins().symbol(idx),
        }
    }
}
