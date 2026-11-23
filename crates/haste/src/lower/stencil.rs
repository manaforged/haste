use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::Result;
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
mod persist;
mod splice;

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
    Fn(usize),
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
    pub form: Form,
}

/// What an inline body holds: no loop, a loop, bulky callees left out of line, or copies of bulky callees.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Form {
    #[default]
    Lean,
    Loop,
    Compact,
    Dense,
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
    compact: bool,
    sole: bool,
}

struct InlineTemplate {
    body: Arc<Function>,
    returns: bool,
    reloc_plan: Arc<RelocPlan>,
    height: u8,
    size: u16,
    form: Form,
}

enum Miss {
    Shape,
    Over,
}

type Lowered = (std::result::Result<(Arc<Function>, bool), Miss>, RelocVec, Arc<RelocPlan>, usize);

impl InlineTemplate {
    fn shape(&self) -> (bool, u8, u16) {
        (self.returns, self.height, self.size)
    }
}

const TRANSIENT_CAP: usize = 32_768;
const TRIM_SHARDS: usize = 4;
const MAX_INLINE_HEIGHT: u8 = 12;

pub(super) type InlineMeta = (ItemRef, FootprintId, bool, u8, u16);

#[derive(Clone, Copy)]
pub(super) enum Tally {
    Hit,
    Miss,
    Built,
    Raced,
    OpenCycle,
    Deferred,
    Devirt,
    Trim,
    Flush,
    Retry,
    Compact,
    Disk,
    Stale,
}

const TALLY_NAMES: [&str; 13] = [
    "template_hit", "template_miss", "template_simplify", "template_raced", "unstored_cycle", "unstored_deferred", "devirt_simplify", "trim_partial", "trim_full", "compact_retry", "compact_fit",
    "template_disk", "template_stale",
];
static TALLIES: [AtomicUsize; TALLY_NAMES.len()] = [const { AtomicUsize::new(0) }; TALLY_NAMES.len()];

fn tally(kind: Tally) {
    TALLIES[kind as usize].fetch_add(1, Ordering::Relaxed);
}

pub(super) fn simplify_tallied(func: &mut Function, kind: Tally) {
    tally(kind);
    haste_codegen::simplify(func, crate::profile().into());
}

/// Cached inlineable callee stencils and footprints across a compilation.
#[derive(Default)]
pub struct Bodies {
    pub(super) plans: Table<(ItemRef, Option<ExprId>), Arc<BodyFootprintPlan>>,
    inlined: Table<(usize, usize), Option<Arc<InlinedCallee>>>,
    pub(super) partial: Table<(usize, bool), Option<Arc<InlinedCallee>>>,
    inline_meta: Table<(usize, usize), Option<InlineMeta>>,
    templates: Table<TemplateKey, Arc<InlineTemplate>>,
    rejected: Table<TemplateKey, ()>,
    cyclic: RwLock<FxHashMap<usize, cycles::CycleState>>,
    persist: std::sync::OnceLock<persist::Persist>,
    transient_entries: AtomicUsize,
    evict_cursor: AtomicUsize,
}

impl Bodies {
    /// Drops all cached inlineable bodies and footprints.
    pub fn clear(&self) {
        self.clear_transient();
        self.inline_meta.clear();
        self.rejected.clear();
        self.plans.clear();
        write_lock(&self.cyclic).clear();
    }

    /// Drops cached inline templates while retaining footprint plans, shape metadata, and precomputed cycle entries.
    pub fn clear_transient(&self) {
        self.transient_entries.store(0, Ordering::Relaxed);
        self.evict_cursor.store(0, Ordering::Relaxed);
        self.inlined.clear();
        self.partial.clear();
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
            removed += self.partial.clear_shard(shard);
            removed += self.templates.clear_shard(shard);
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

    pub(super) fn bump_transient(&self) {
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

/// Whether a template under construction must keep a callee of height `height` as a call, so that its own height stays within `MAX_INLINE_HEIGHT`.
fn at_height_cap(inline: &Inline, height: u8) -> bool {
    !inline.stack.is_empty() && height >= MAX_INLINE_HEIGHT
}

fn record_meta_height(inline: &mut Inline, meta: Option<(ItemRef, FootprintId, bool, u8, u16)>) -> Option<(ItemRef, FootprintId, bool)> {
    meta.filter(|meta| !at_height_cap(inline, meta.3)).map(|(item, fp, returns, h, sz)| {
        inline.max_callee_height = inline.max_callee_height.max(h);
        inline.max_callee_size = inline.max_callee_size.max(sz);
        (item, fp, returns)
    })
}

fn record_callee_height(inline: &mut Inline, callee: Option<Arc<InlinedCallee>>) -> Option<Arc<InlinedCallee>> {
    let callee = callee.filter(|c| !at_height_cap(inline, c.height));
    if let Some(c) = &callee {
        inline.max_callee_height = inline.max_callee_height.max(c.height);
        inline.max_callee_size = inline.max_callee_size.max(c.size);
    }
    callee
}

fn run_inline_probe(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, shape_only, compact): (usize, bool, bool)) -> Result<(Option<InlineHit>, usize, usize, Quiet)> {
    let (depth, quiet) = (inline.stack.len(), Quiet::mark(cx, slots));
    inline.stack.push(index);
    let saved = std::mem::take(&mut inline.host);
    inline.host.compact = compact;
    inline.host.sole = super::budget::stable_sole(cx, reached, index);
    let outcome = ensure_inline_template(cx, (reached, slots), inline, index, (limit, shape_only, &quiet));
    inline.host = saved;
    inline.pop(depth);
    let (tmpl, cycle) = outcome?;
    Ok((tmpl, cycle, depth, quiet))
}

fn limit_flags(cx: &mut Context<'_>, reached: &Reached, (bodies, inline): (&Bodies, &Inline), index: usize, limit: usize) -> Option<(usize, Option<usize>, bool, bool)> {
    let cap = if !reached.is_sole_called(index) && bodies.is_cyclic(cx, reached, index) {
        let host = inline.stack.last().copied().filter(|&host| host >= index);
        if host.is_some_and(|host| bodies.is_cyclic(cx, reached, host)) {
            return None;
        }
        super::budget::cycle_limit(cx, reached, index)?
    } else {
        usize::MAX
    };
    let std_limit = super::budget::inline_limit(cx, reached, index).map(|standard| standard.min(cap));
    let limit = limit.min(cap);
    Some((limit, std_limit, std_limit.is_some_and(|sl| limit >= sl), std_limit.is_some_and(|sl| limit <= sl)))
}

fn cached_inline(bodies: &Bodies, inline: &Inline, (index, limit, std_limit): (usize, usize, Option<usize>), (pos_valid, neg_valid, shape_only): (bool, bool, bool)) -> Option<(Option<Arc<InlinedCallee>>, Option<InlineMeta>, usize)> {
    let whole = |callee: Option<Arc<InlinedCallee>>, cycle: usize| {
        let meta = callee.as_deref().map(InlinedCallee::meta);
        (callee, meta, cycle)
    };
    let ok_cal = |callee: &Option<Arc<InlinedCallee>>| if callee.is_some() { pos_valid } else { neg_valid };
    if let Some(callee) = bodies.inlined.get(&(index, limit)) {
        return Some(whole(callee, usize::MAX));
    }
    if let Some(sl) = std_limit.filter(|&sl| sl != limit)
        && let Some(callee) = bodies.inlined.get(&(index, sl)).filter(ok_cal)
    {
        return Some(whole(callee, usize::MAX));
    }
    if let Some(meta) = bodies.inline_meta.get(&(index, limit)).filter(|m| shape_only || m.is_none()) {
        return Some((None, meta, usize::MAX));
    }
    if let Some(pos) = inline.stack.iter().position(|&at| at == index) {
        return Some((None, None, pos));
    }
    if let Some((callee, cycle)) = inline.local.get(&(index, limit)) {
        return Some(whole(callee.clone(), *cycle));
    }
    if let Some(sl) = std_limit.filter(|&sl| sl != limit)
        && let Some((callee, cycle)) = inline.local.get(&(index, sl)).filter(|(callee, _)| ok_cal(callee))
    {
        return Some(whole(callee.clone(), *cycle));
    }
    inline.local_meta.get(&(index, limit)).filter(|(m, _)| shape_only || m.is_none()).map(|&(meta, cycle)| (None, meta, cycle))
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
        form: t.form,
    })
}

pub(super) fn resolve_inline_meta(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, limit: usize) -> Result<(Option<(ItemRef, FootprintId, bool)>, usize)> {
    let bodies = slots.bodies();
    let Some((limit, std_limit, pos_valid, neg_valid)) = limit_flags(cx, reached, (bodies, &*inline), index, limit) else {
        return Ok((None, usize::MAX));
    };
    if let Some((_, meta, cycle)) = cached_inline(bodies, inline, (index, limit, std_limit), (pos_valid, neg_valid, true)) {
        return Ok((record_meta_height(inline, meta), cycle));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) = run_inline_probe(cx, (reached, slots), inline, index, (limit, true, false))?;
    let (meta, inlined) = match tmpl {
        Some((footprint, Found::Body(t, relocs))) => (
            Some((item, footprint, t.returns, t.height, t.size)),
            relocs.map(|r| Some(build_inlined(item, footprint, &t, r))),
        ),
        Some((footprint, Found::Shape(returns, height, size))) => (Some((item, footprint, returns, height, size)), None),
        None => (None, Some(None)),
    };
    if cycle > depth && quiet.holds(cx, slots) {
        if let Some(Some(cal)) = inlined {
            bodies.bump_transient();
            let stored = bodies.inlined.get_or_insert_with((index, limit), || Some(cal));
            return Ok((record_meta_height(inline, stored.as_deref().map(InlinedCallee::meta)), cycle));
        }
        let stored = bodies.inline_meta.get_or_insert_with((index, limit), || meta);
        return Ok((record_meta_height(inline, stored), cycle));
    }
    let local_cyc = if cycle < depth { cycle } else { usize::MAX };
    if let Some(cal) = inlined {
        inline.local.insert((index, limit), (cal, local_cyc));
    } else {
        inline.local_meta.insert((index, limit), (meta, local_cyc));
    }
    Ok((record_meta_height(inline, meta), cycle))
}

fn finish_inline(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, (index, callee, cycle): (usize, Option<Arc<InlinedCallee>>, usize)) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let callee = super::budget::maybe_partial(cx, (reached, slots), inline, (index, callee))?;
    Ok((record_callee_height(inline, callee), cycle))
}

pub(super) fn resolve_inline(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, compact): (usize, bool)) -> Result<(Option<Arc<InlinedCallee>>, usize)> {
    let bodies = slots.bodies();
    let Some((limit, std_limit, pos_valid, neg_valid)) = limit_flags(cx, reached, (bodies, &*inline), index, limit) else {
        return Ok((None, usize::MAX));
    };
    if !compact && let Some((callee, _, cycle)) = cached_inline(bodies, inline, (index, limit, std_limit), (pos_valid, neg_valid, false)) {
        return finish_inline(cx, (reached, slots), inline, (index, callee, cycle));
    }
    let item = reached.list[index].0.0;
    let (tmpl, cycle, depth, quiet) = run_inline_probe(cx, (reached, slots), inline, index, (limit, false, compact))?;
    let result = match tmpl {
        Some((footprint, Found::Body(t, Some(relocs)))) => Some(build_inlined(item, footprint, &t, relocs)),
        Some((footprint, Found::Body(t, None))) => {
            let evaluated = quiet.lowered(slots, eval_relocs(cx, reached, slots, &t.reloc_plan, index))?;
            evaluated.map(|relocs| build_inlined(item, footprint, &t, relocs))
        }
        Some((_, Found::Shape(..))) => unreachable!("a body probe never answers from recorded metadata"),
        None => None,
    };
    if !compact {
        if cycle > depth && quiet.holds(cx, slots) {
            if let Some(cal) = result {
                bodies.bump_transient();
                let stored = bodies.inlined.get_or_insert_with((index, limit), || Some(cal));
                return finish_inline(cx, (reached, slots), inline, (index, stored, cycle));
            }
            bodies.inline_meta.get_or_insert_with((index, limit), || None);
            return finish_inline(cx, (reached, slots), inline, (index, None, cycle));
        }
        inline.local.insert((index, limit), (result.clone(), if cycle < depth { cycle } else { usize::MAX }));
    }
    finish_inline(cx, (reached, slots), inline, (index, result, cycle))
}

pub(super) struct Quiet {
    wanted: usize,
    sema: (usize, usize),
}

impl Quiet {
    pub(super) fn mark(cx: &Context<'_>, slots: &dyn Slots) -> Self {
        Self { wanted: slots.wanted_len(), sema: cx.quiet_mark() }
    }

    pub(super) fn holds(&self, cx: &Context<'_>, slots: &dyn Slots) -> bool {
        slots.wanted_len() == self.wanted && cx.quiet_mark() == self.sema
    }

    pub(super) fn lowered<T>(&self, slots: &dyn Slots, result: Result<T>) -> Result<Option<T>> {
        match result {
            Ok(out) => Ok(Some(out)),
            Err(err) if slots.wanted_len() > self.wanted => Err(err),
            Err(_) => Ok(None),
        }
    }
}

enum Found {
    Body(Arc<InlineTemplate>, Option<RelocVec>),
    Shape(bool, u8, u16),
}

type InlineHit = (FootprintId, Found);

fn ensure_inline_template(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, probe: (usize, bool, &Quiet)) -> Result<(Option<InlineHit>, usize)> {
    let saved_h = std::mem::replace(&mut inline.max_callee_height, 0);
    let saved_sz = std::mem::replace(&mut inline.max_callee_size, 0);
    let res = ensure_inline_template_inner(cx, (reached, slots), inline, index, probe);
    inline.max_callee_height = saved_h;
    inline.max_callee_size = saved_sz;
    res
}

fn try_cached_template<'a>(cx: &Context<'_>, (reached, slots): (&Reached, &'a dyn Slots), key: &TemplateKey, (shape_only, open_ok, quiet): (bool, bool, &Quiet)) -> Result<Option<Option<Found>>, Option<persist::Disk<'a>>> {
    if slots.bodies().rejected.get(key).is_some() {
        tally(Tally::Hit);
        return Ok(Some(None));
    }
    if let Some(template) = slots.bodies().templates.get(key) {
        tally(Tally::Hit);
        return Ok(Some(Some(Found::Body(template, None))));
    }
    tally(Tally::Miss);
    let disk = if key.compact { None } else { slots.bodies().disk(cx, key) };
    if shape_only && open_ok && let Some(shape) = disk.as_ref().and_then(persist::Disk::hit) {
        return Ok(Some(shape.map(|(returns, height, size)| Found::Shape(returns, height, size))));
    }
    if !shape_only && open_ok && let Some(d) = &disk && let Some(mut found) = d.template(cx, (reached, slots)) {
        if quiet.holds(cx, slots) {
            match found {
                Some(t) => {
                    slots.bodies().bump_transient();
                    found = Some(slots.bodies().templates.get_or_insert_with(*key, || t));
                }
                None => {
                    slots.bodies().rejected.get_or_insert_with(*key, || ());
                }
            }
        }
        return Ok(Some(found.map(|t| Found::Body(t, None))));
    }
    Err(disk)
}

fn settle_built_template(cx: &Context<'_>, (reached, slots): (&Reached, &dyn Slots), key: TemplateKey, mut template: Option<Arc<InlineTemplate>>, (disk, calm, servable, in_cycle): (Option<persist::Disk<'_>>, bool, bool, bool)) -> Option<Arc<InlineTemplate>> {
    if in_cycle {
        tally(Tally::OpenCycle);
    } else if calm {
        match template {
            Some(t) => {
                slots.bodies().bump_transient();
                let built = Arc::as_ptr(&t);
                let stored = slots.bodies().templates.get_or_insert_with(key, || t);
                if built != Arc::as_ptr(&stored) {
                    tally(Tally::Raced);
                }
                template = Some(stored);
            }
            None => {
                slots.bodies().rejected.get_or_insert_with(key, || ());
            }
        }
    } else {
        tally(Tally::Deferred);
    }
    if let Some(disk) = disk {
        disk.settle(cx, (reached, slots), template.as_deref(), (calm && !in_cycle, servable && calm));
    }
    template
}

fn pack_template(fit: std::result::Result<(Arc<Function>, bool), Miss>, reloc_plan: Arc<RelocPlan>, form: Form, (base_h, base_size): (u8, usize)) -> Option<Arc<InlineTemplate>> {
    fit.ok().and_then(|(body, returns)| {
        let body_size = body.size();
        let grew = (base_size == 0 && body_size > 32) || (base_size > 0 && body_size.saturating_sub(base_size) > 16);
        let height = if grew { base_h.saturating_add(1) } else { base_h };
        let size = body_size.min(u16::MAX as usize) as u16;
        (height <= MAX_INLINE_HEIGHT).then(|| Arc::new(InlineTemplate { body, returns, reloc_plan, height, size, form }))
    })
}

fn lower_with_compact(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, compact, quiet): (usize, bool, &Quiet)) -> Result<(Option<Lowered>, bool)> {
    let mut lowered = lower_template(cx, (reached, slots), inline, index, (limit, quiet))?;
    let over = !compact && inline.host.retry && matches!(lowered, Some((Err(Miss::Over), ..)));
    let Some(roomy) = over.then(|| super::budget::compact_limit(cx, reached, index, limit)).flatten() else {
        return Ok((lowered, false));
    };
    let Some(first) = lowered.as_ref().map(|found| found.3) else { return Ok((lowered, false)) };
    tally(Tally::Retry);
    inline.host.compact = true;
    lowered = lower_template(cx, (reached, slots), inline, index, (roomy, quiet))?.map(|(fit, relocs, plan, cycle)| (fit, relocs, plan, cycle.min(first)));
    Ok((lowered, true))
}

fn ensure_inline_template_inner(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, shape_only, quiet): (usize, bool, &Quiet)) -> Result<(Option<InlineHit>, usize)> {
    let item = reached.list[index].0.0;
    let fp_res = eval_footprint_with_cycle(cx, (reached, slots), inline, index);
    let fp_h = std::mem::replace(&mut inline.max_callee_height, 0);
    let fp_sz = std::mem::replace(&mut inline.max_callee_size, 0);
    let Some((footprint, fp_cycle)) = quiet.lowered(slots, fp_res)? else { return Ok((None, usize::MAX)) };
    let key = TemplateKey { item, footprint, limit, low: slots.low_entries(), compact: inline.host.compact, sole: super::budget::stable_sole(cx, reached, index) };
    let depth = inline.stack.len().saturating_sub(1);
    let servable = shape_only && fp_cycle > depth;
    let disk = match try_cached_template(cx, (reached, slots), &key, (shape_only, fp_cycle > depth, quiet)) {
        Ok(Some(hit)) => return Ok((hit.map(|f| (footprint, f)), fp_cycle)),
        Ok(None) => None,
        Err(disk) => disk,
    };
    let (lowered, retried) = lower_with_compact(cx, (reached, slots), inline, index, (limit, key.compact, quiet))?;
    let form = if inline.host.compact { Form::Compact } else { inline.host.seen.max(super::budget::own_form(cx, reached, index)) };
    let (ir_h, ir_sz) = (inline.max_callee_height, inline.max_callee_size);
    let Some((fit, relocs, reloc_plan, ir_cycle)) = lowered else {
        if let Some(disk) = disk {
            disk.settle(cx, (reached, slots), None, (false, servable && quiet.holds(cx, slots)));
        }
        return Ok((None, fp_cycle));
    };
    let cycle = fp_cycle.min(ir_cycle);
    let template = pack_template(fit, reloc_plan, form, (fp_h.max(ir_h), fp_sz.max(ir_sz) as usize));
    if retried && template.is_some() {
        tally(Tally::Compact);
    }
    let template = settle_built_template(cx, (reached, slots), key, template, (disk, quiet.holds(cx, slots), servable, cycle <= depth));
    Ok((template.map(|t| (footprint, Found::Body(t, Some(relocs)))), cycle))
}

fn lower_template(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, (limit, quiet): (usize, &Quiet)) -> Result<Option<Lowered>> {
    let lowered = run_with_cycle(cx, (reached, slots), inline, index, LowerMode::Stencil);
    Ok(quiet.lowered(slots, lowered)?.map(|(func, relocs, plan, cycle)| {
        let limit = if plan.has_calls() { limit } else { super::budget::leaf_limit(cx, reached, index, limit) };
        let res = if haste_codegen::trace_enabled() {
            haste_codegen::with_trace_label(cx.name(reached.list[index].0.0), || simplify_inline(func, limit))
        } else {
            simplify_inline(func, limit)
        };
        (res, relocs, plan, cycle)
    }))
}

fn simplify_inline(mut func: Function, limit: usize) -> std::result::Result<(Arc<Function>, bool), Miss> {
    if func.size() > limit.saturating_mul(6).max(1600) {
        return Err(Miss::Shape);
    }
    haste_codegen::split_coroutine(&mut func);
    simplify_tallied(&mut func, Tally::Built);
    if !func.can_inline() {
        return Err(Miss::Shape);
    }
    if func.inline_size() > limit {
        return Err(Miss::Over);
    }
    let returns = func.has_return();
    Ok((Arc::new(func), returns))
}

fn body_arena(cx: &mut Context<'_>, source: ItemRef, shape: &Shape) -> Result<Arena> {
    match shape {
        Shape::Body | Shape::Closure(_) => Ok(Arena::Body(cx.body(source)?)),
        Shape::Anon(anon) => Ok(anon.arena.clone()),
        Shape::Glue(_) | Shape::Virtual(_) | Shape::Untuple(_) | Shape::Ctor(_) | Shape::Clone(_) | Shape::Asm(_) | Shape::GlobalAlloc(_) => Ok(Arena::Body(Arc::default())),
    }
}

fn run_with_cycle(cx: &mut Context<'_>, (reached, slots): (&Reached, &dyn Slots), inline: &mut Inline, index: usize, mode: LowerMode) -> Result<(Function, RelocVec, Arc<RelocPlan>, usize)> {
    let (item, substs) = &reached.list[index].0;
    let concrete = substs.is_empty();
    let typed = reached.typed(index, cx.name(*item))?;
    let typed = typed.scoped(&cx.types);
    let typed: &Typed = &typed;
    let source = match &typed.shape {
        Shape::Closure(closure) => closure.parent,
        _ => *item,
    };
    let arena = body_arena(cx, source, &typed.shape)?;
    let mut sig = if is_static(cx, *item) { signature_of(cx, None, [])? } else { signature(cx, typed)? };
    if mode == LowerMode::Dict {
        sig.params.push(AbiParam::special(Type::I64, ArgumentPurpose::Dict));
    }
    let mut func = Function::new(sig);
    func.keep_fp = crate::sema::reach::unwind_enabled(cx, reached);
    let b = FunctionBuilder::new(&mut func);
    let (mut relocs, mut reloc_plan) = (Vec::new(), Vec::new());
    let mut cycle = {
        let mut lower = Lower::new(cx, (reached, slots, inline), (index, source, arena.body(), typed), (b, mode), (&mut relocs, &mut reloc_plan));
        lower.run()?;
        lower.cycle_root
    };
    if splice::should_splice_devirt(slots, &func, mode, &reloc_plan) {
        cycle = splice::splice_devirt_vtable_calls(cx, (reached, slots), inline, (&mut func, &mut relocs, &mut reloc_plan), (cycle, concrete))?;
    }
    if mode == LowerMode::Stencil {
        splice::canonicalize_const_symbols((cx, reached), &mut func, (&reloc_plan, &relocs), (concrete, inline.stack.is_empty()));
        if func.is_coroutine() {
            haste_codegen::split_coroutine(&mut func);
        }
        if !func.unwind_drops().is_empty() {
            simplify_tallied(&mut func, Tally::Built);
            let mut sym_may_unwind = |sym: u32| match relocs.get(sym as usize) {
                Some(SymbolSource::Slot(idx)) => reached.may_unwind(cx, *idx),
                Some(SymbolSource::Host(h)) => crate::sema::reach::host_symbol_may_unwind(h),
                _ => true,
            };
            func.elide_dead_unwind_drops(&mut sym_may_unwind);
        }
        func.zero_init_unwind_flags();
    }
    Ok((func, relocs, Arc::new(RelocPlan::from_vec(reloc_plan)), cycle))
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

    pub(super) fn emit_rodata_id(&mut self, id: RodataId, bytes: &[u8]) -> Value {
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

    pub(super) fn emit_reloc(&mut self, src: SymbolSource, step: RelocStep, self_call: bool, low_slot: bool) -> Value {
        let image = match src {
            SymbolSource::Vtable(_) | SymbolSource::Rodata(_) | SymbolSource::PromotedCell { .. } => true,
            SymbolSource::StaticCell { index, .. } => !self.cx.foreign(self.reached.list[index].0.0),
            _ => false,
        };
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
            LowerMode::Stencil => {
                let addr = self.ins().symbol(idx);
                if image {
                    self.b.func.mark_image_addr(addr);
                }
                addr
            }
        }
    }
}
