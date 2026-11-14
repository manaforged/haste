use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{Result, bail};
use haste_codegen::{AbiParam, ArgumentPurpose, Function, FunctionBuilder, Type, Value};
use rustc_hash::FxHashMap;

use crate::body::ExprId;
use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::infer::{Arena, Shape, Typed};
use crate::sema::reach::Reached;
use crate::sema::table::Table;
use crate::sync::write_lock;

mod cycles;

use super::plan::{BodyFootprintPlan, eval_footprint_with_cycle};
use super::reloc::{RelocPlan, RelocStep, SiteOrigin, eval_relocs};
use super::{Inline, Lower, Slots, Word, is_static, signature, signature_of};

/// The 128-bit structural footprint digest of a lowered function and its inlined callees.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FootprintId(pub u128);
impl FootprintId { pub fn is_portable(self) -> bool { self.0 & 1 == 0 } }

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

impl InlinedCallee {
    fn meta(&self) -> InlineMeta {
        (self.item, self.footprint, self.returns, self.height, self.size)
    }
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

const TRANSIENT_CAP: usize = 32_768;
const TRIM_SHARDS: usize = 8;
const MAX_INLINE_HEIGHT: u8 = 10;

pub(super) type InlineMeta = (ItemRef, FootprintId, bool, u8, u16);

#[derive(Clone, Copy)]
enum Tally {
    Hit,
    Miss,
    Built,
    Raced,
    OpenCycle,
    Deferred,
    Devirt,
    Trim,
    Flush,
}

const TALLY_NAMES: [&str; 9] = ["template_hit", "template_miss", "template_simplify", "template_raced", "unstored_cycle", "unstored_deferred", "devirt_simplify", "trim_partial", "trim_full"];
static TALLIES: [AtomicUsize; TALLY_NAMES.len()] = [const { AtomicUsize::new(0) }; TALLY_NAMES.len()];

fn tally(kind: Tally) {
    TALLIES[kind as usize].fetch_add(1, Ordering::Relaxed);
}

fn simplify_tallied(func: &mut Function, kind: Tally) {
    tally(kind);
    haste_codegen::simplify(func, crate::profile().into());
}

/// Cached inlineable callee stencils and footprints across a compilation.
#[derive(Default)]
pub struct Bodies {
    pub(super) plans: Table<(ItemRef, Option<ExprId>), Arc<BodyFootprintPlan>>,
    inlined: Table<usize, Option<Arc<InlinedCallee>>>,
    inline_meta: Table<usize, Option<InlineMeta>>,
    templates: Table<TemplateKey, Option<Arc<InlineTemplate>>>,
    cyclic: RwLock<FxHashMap<usize, cycles::CycleState>>,
    transient_entries: AtomicUsize,
    evict_cursor: AtomicUsize,
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

    /// Evicts a rolling slice of cached inlined callees and footprint plans at the trim watermark, and every transient entry at the cap.
    pub(crate) fn trim_transient(&self) {
        let count = self.transient_entries.load(Ordering::Relaxed);
        if count < TRANSIENT_CAP / 2 {
            return;
        }
        let start = self.evict_cursor.fetch_add(TRIM_SHARDS, Ordering::Relaxed);
        if count >= TRANSIENT_CAP {
            tally(Tally::Flush);
            self.clear_transient();
            for offset in 0..TRIM_SHARDS {
                self.plans.clear_shard((start + offset) % crate::sema::table::SHARDS);
            }
            return;
        }
        tally(Tally::Trim);
        let mut removed = 0usize;
        for offset in 0..TRIM_SHARDS {
            let shard = (start + offset) % crate::sema::table::SHARDS;
            removed += self.inlined.clear_shard(shard);
            removed += self.inline_meta.clear_shard(shard);
        }
        self.plans.clear_shard((start / TRIM_SHARDS) % crate::sema::table::SHARDS);
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

    /// Formats the inline template cache counters as one line, or nothing unless `HASTE_MISSED` is set.
    pub(crate) fn inline_report() -> String {
        if haste_codegen::missed_report().is_empty() {
            return String::new();
        }
        let mut out = String::from("inline:");
        for (name, count) in TALLY_NAMES.iter().zip(&TALLIES) {
            out.push_str(&format!(" {name}={}", count.load(Ordering::Relaxed)));
        }
        out.push('\n');
        out
    }
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

fn limit_flags(cx: &mut Context<'_>, reached: &Reached, index: usize, limit: usize) -> (bool, bool) {
    let std_limit = super::expr::inline_limit(cx, reached, index);
    (std_limit == Some(limit), std_limit.is_some_and(|sl| limit <= sl))
}

fn build_inlined(item: ItemRef, footprint: FootprintId, t: &InlineTemplate, relocs: RelocVec) -> Arc<InlinedCallee> {
    Arc::new(InlinedCallee {
        item,
        footprint,
        relocs,
        reloc_plan: Arc::clone(&t.reloc_plan),
        body: Arc::clone(&t.body),
        returns: t.returns,
        height: t.height,
        size: t.size,
    })
}

pub(super) fn resolve_inline_meta(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, limit: usize) -> Result<(Option<(ItemRef, FootprintId, bool)>, usize)> {
    let bodies = slots.bodies();
    if bodies.is_cyclic(cx, reached, index) {
        return Ok((None, usize::MAX));
    }
    let (standard, neg_valid) = limit_flags(cx, reached, index, limit);
    if let Some(cached) = bodies.inlined.get(&index).filter(|c| c.is_some() || neg_valid) {
        return Ok((record_meta_height(inline, cached.as_deref().map(InlinedCallee::meta)), usize::MAX));
    }
    if let Some(cached) = bodies.inline_meta.get(&index).filter(|c| c.is_some() || neg_valid) {
        return Ok((record_meta_height(inline, cached), usize::MAX));
    }
    if let Some(pos) = inline.stack.iter().position(|&at| at == index) {
        return Ok((None, pos));
    }
    if let Some((cached, cycle)) = inline.local.get(&index).filter(|(c, _)| c.is_some() || neg_valid) {
        let (meta, cyc) = (cached.as_deref().map(InlinedCallee::meta), *cycle);
        return Ok((record_meta_height(inline, meta), cyc));
    }
    if let Some(&(meta, cyc)) = inline.local_meta.get(&index).filter(|(m, _)| m.is_some() || neg_valid) {
        return Ok((record_meta_height(inline, meta), cyc));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) = run_inline_probe(cx, (reached, slots), inline, index, limit)?;
    let (meta, inlined) = match tmpl {
        Some((footprint, t, relocs)) => (
            Some((item, footprint, t.returns, t.height, t.size)),
            relocs.map(|r| Some(build_inlined(item, footprint, &t, r))),
        ),
        None => (None, Some(None)),
    };
    if standard {
        if cycle > depth && quiet.holds(cx, slots) {
            bodies.bump_transient();
            if let Some(cal) = inlined {
                let stored = bodies.inlined.get_or_insert_with(index, || cal);
                return Ok((record_meta_height(inline, stored.as_deref().map(InlinedCallee::meta)), cycle));
            }
            let stored = bodies.inline_meta.get_or_insert_with(index, || meta);
            return Ok((record_meta_height(inline, stored), cycle));
        }
        let local_cyc = if cycle < depth { cycle } else { usize::MAX };
        if let Some(cal) = inlined {
            inline.local.insert(index, (cal, local_cyc));
        } else {
            inline.local_meta.insert(index, (meta, local_cyc));
        }
    }
    Ok((record_meta_height(inline, meta), cycle))
}

pub(super) fn resolve_inline(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, limit: usize) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let bodies = slots.bodies();
    if bodies.is_cyclic(cx, reached, index) {
        return Ok((None, usize::MAX));
    }
    let (standard, neg_valid) = limit_flags(cx, reached, index, limit);
    if let Some(cached) = bodies.inlined.get(&index).filter(|c| c.is_some() || neg_valid) {
        return Ok((record_callee_height(inline, cached), usize::MAX));
    }
    if neg_valid && matches!(bodies.inline_meta.get(&index), Some(None)) {
        return Ok((None, usize::MAX));
    }
    if let Some(pos) = inline.stack.iter().position(|&at| at == index) {
        return Ok((None, pos));
    }
    if let Some((cached, cycle)) = inline.local.get(&index).filter(|(c, _)| c.is_some() || neg_valid) {
        let (hit, cyc) = (cached.clone(), *cycle);
        return Ok((record_callee_height(inline, hit), cyc));
    }
    if neg_valid && let Some(&(None, cyc)) = inline.local_meta.get(&index) {
        return Ok((None, cyc));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) = run_inline_probe(cx, (reached, slots), inline, index, limit)?;
    let result = match tmpl {
        Some((footprint, t, Some(relocs))) => Some(build_inlined(item, footprint, &t, relocs)),
        Some((footprint, t, None)) => {
            let evaluated = quiet.lowered(slots, eval_relocs(cx, reached, slots, &t.reloc_plan, index))?;
            evaluated.map(|relocs| build_inlined(item, footprint, &t, relocs))
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
        tally(Tally::Hit);
        return Ok((template.map(|t| (footprint, t, None)), fp_cycle));
    }
    tally(Tally::Miss);
    let lowered = run_with_cycle(cx, (reached, slots), inline, index, LowerMode::Stencil);
    let ir_h = inline.max_callee_height;
    let ir_sz = inline.max_callee_size;
    let Some((func, relocs, reloc_plan, ir_cycle)) = quiet.lowered(slots, lowered)? else { return Ok((None, fp_cycle)) };
    let cycle = fp_cycle.min(ir_cycle);
    let base_h = fp_h.max(ir_h);
    let base_size = fp_sz.max(ir_sz) as usize;
    let leaf = !plan_has_calls(&reloc_plan);
    let mut template = simplify_inline(func, limit, leaf).and_then(|(body, returns)| {
        let body_size = body.size();
        let grew = (base_size == 0 && body_size > 32) || (base_size > 0 && body_size.saturating_sub(base_size) > 16);
        let height = if grew { base_h.saturating_add(1) } else { base_h };
        let size = body_size.min(u16::MAX as usize) as u16;
        (height <= MAX_INLINE_HEIGHT).then(|| Arc::new(InlineTemplate { body, returns, reloc_plan, height, size }))
    });
    let depth = inline.stack.len().saturating_sub(1);
    if cycle <= depth {
        tally(Tally::OpenCycle);
    } else if quiet.holds(cx, slots) {
        slots.bodies().bump_transient();
        let built = template.as_ref().map(Arc::as_ptr);
        template = slots.bodies().templates.get_or_insert_with(key, || template);
        if built != template.as_ref().map(Arc::as_ptr) {
            tally(Tally::Raced);
        }
    } else {
        tally(Tally::Deferred);
    }
    Ok((template.map(|t| (footprint, t, Some(relocs))), cycle))
}

fn plan_has_calls(plan: &RelocPlan) -> bool {
    plan.steps.iter().any(|step| match step {
        RelocStep::SiteSlot(_)
        | RelocStep::VtableSlot { .. }
        | RelocStep::Const(SymbolSource::Slot(_) | SymbolSource::Host(_)) => true,
        RelocStep::SiteInline { plan, .. } => plan_has_calls(plan),
        _ => false,
    })
}

fn simplify_inline(mut func: Function, limit: usize, leaf: bool) -> Option<(Arc<Function>, bool)> {
    if func.is_coroutine() {
        return None;
    }
    let release = crate::profile() == crate::outline::tree::Profile::Release;
    let effective_limit = if release && leaf { limit.saturating_add(limit / 2).min(900).max(limit) } else { limit };
    if func.size() > effective_limit.saturating_mul(6).max(1600) {
        return None;
    }
    simplify_tallied(&mut func, Tally::Built);
    (func.can_inline() && func.inline_size() <= effective_limit).then(|| {
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
    if mode == LowerMode::Stencil {
        canonicalize_const_symbols(&mut func, &reloc_plan);
    }
    Ok((func, relocs, Arc::new(RelocPlan::from_vec(reloc_plan)), cycle))
}

fn collect_const_remaps(steps: &[RelocStep], self_slots: &[u32], offset: &mut usize, canon: &mut FxHashMap<SymbolSource, u32>, pairs: &mut Vec<(u32, u32)>) {
    for step in steps {
        match step {
            RelocStep::Const(src) => {
                let idx = *offset as u32;
                *offset += 1;
                if !self_slots.contains(&idx) {
                    if let Some(&first) = canon.get(src) {
                        pairs.push((idx, first));
                    } else {
                        canon.insert(src.clone(), idx);
                    }
                }
            }
            RelocStep::SiteInline { plan, .. } => collect_const_remaps(&plan.steps, self_slots, offset, canon, pairs),
            RelocStep::TypeName(_) => *offset += 2,
            RelocStep::Promoted { .. } | RelocStep::VtableAt(_) | RelocStep::VtableSlot { .. } | RelocStep::SiteSlot(_) | RelocStep::TypeId(_) => *offset += 1,
        }
    }
}

fn canonicalize_const_symbols(func: &mut Function, steps: &[RelocStep]) {
    func.remap_symbols(|self_slots| {
        let (mut offset, mut canon, mut pairs) = (0usize, FxHashMap::default(), Vec::new());
        collect_const_remaps(steps, self_slots, &mut offset, &mut canon, &mut pairs);
        if pairs.is_empty() {
            return None;
        }
        let mut remap: Vec<u32> = (0..offset as u32).collect();
        for (from, to) in pairs {
            remap[from as usize] = to;
        }
        Some(remap)
    });
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
        simplify_tallied(func, Tally::Devirt);
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
        canonicalize_const_symbols(func, reloc_plan);
        simplify_tallied(func, Tally::Devirt);
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
        self.emit_rodata_id(id, bytes)
    }

    pub(super) fn emit_static_rodata(&mut self, index: usize, bytes: &[u8]) -> Value {
        let id = self.slots.static_rodata(index, bytes);
        self.emit_rodata_id(id, bytes)
    }

    fn emit_rodata_id(&mut self, id: RodataId, bytes: &[u8]) -> Value {
        let idx = self.relocs.len() as u32;
        let val = self.emit_symbol(SymbolSource::Rodata(id));
        if matches!(self.mode, LowerMode::Stencil) && bytes.len() <= 1024 {
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
