use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use haste_codegen::Function;
use rustc_hash::FxHashMap;

use crate::body::ExprId;
use crate::host::HostSymbol;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sema::table::Table;
use crate::sync::write_lock;

mod cycles;
mod emit;
mod inflight;
mod inline;
mod persist;
mod splice;
mod tally;

use emit::run_with_cycle;
pub(super) use inline::{Quiet, resolve_inline, resolve_inline_meta};
use tally::tally;
pub(super) use tally::{Tally, simplify_tallied};

use super::plan::BodyFootprintPlan;
use super::reloc::RelocPlan;
use super::{Inline, Slots, Word};

/// The 128-bit structural footprint digest of a lowered function and its inlined callees.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FootprintId(pub u128);

impl FootprintId {
    /// Whether this footprint depends only on portable items and can be persisted across builds.
    pub fn is_portable(self) -> bool {
        self.0 & 1 == 0
    }
}

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
        (
            self.item,
            self.footprint,
            self.returns,
            self.height,
            self.size,
        )
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

pub(crate) struct InlineTemplate {
    pub(crate) body: Arc<Function>,
    pub(crate) returns: bool,
    pub(crate) reloc_plan: Arc<RelocPlan>,
    pub(crate) height: u8,
    pub(crate) size: u16,
    pub(crate) form: Form,
}

impl InlineTemplate {
    pub(crate) fn shape(&self) -> (bool, u8, u16) {
        (self.returns, self.height, self.size)
    }
}

const TRANSIENT_CAP: usize = 32_768;
const TRIM_SHARDS: usize = 4;

pub(super) type InlineMeta = (ItemRef, FootprintId, bool, u8, u16);

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
    inflight: inflight::InFlight,
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
                self.plans
                    .clear_shard((start + offset) % crate::sema::table::SHARDS);
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
        self.plans
            .clear_shard((start / TRIM_SHARDS) % crate::sema::table::SHARDS);
        if removed > 0 {
            let mut cur = self.transient_entries.load(Ordering::Relaxed);
            while cur > 0 {
                match self.transient_entries.compare_exchange_weak(
                    cur,
                    cur.saturating_sub(removed),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(next) => cur = next,
                }
            }
        }
    }

    pub(super) fn bump_transient(&self) {
        self.transient_entries.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn inline_report() -> String {
        tally::report()
    }
}

pub(super) fn item_bits(item: ItemRef) -> u128 {
    (u128::from(item.krate) << 64) | (u128::from(item.file) << 32) | u128::from(item.item)
}

/// Lowers reached function `index` into an SSA stencil with symbolic relocations.
pub fn lower_stencil(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    index: usize,
) -> Result<StencilOutput> {
    let (func, relocs, reloc_plan, _, _) = run_with_cycle(
        cx,
        (reached, slots),
        &mut Inline::default(),
        index,
        LowerMode::Stencil,
        false,
    )?;
    Ok(StencilOutput {
        func,
        relocs,
        reloc_plan,
    })
}

/// Lowers reached function `index` into a shared dictionary-parameterized SSA body and its relocation vector.
pub fn lower_dict(
    cx: &mut Context<'_>,
    reached: &Reached,
    slots: &dyn Slots,
    index: usize,
) -> Result<(Function, RelocVec, Arc<RelocPlan>)> {
    let (func, relocs, reloc_plan, _, _) = run_with_cycle(
        cx,
        (reached, slots),
        &mut Inline::default(),
        index,
        LowerMode::Dict,
        false,
    )?;
    Ok((func, relocs, reloc_plan))
}

/// Pre-populates the shared inline template cache for reached function `index` when it is inlineable.
pub fn warm_inline(cx: &mut Context<'_>, reached: &Reached, slots: &dyn Slots, index: usize) {
    if let Some(limit) = super::budget::inline_limit(cx, reached, index) {
        let mut inline = Inline {
            warming: true,
            ..Inline::default()
        };
        let _ = resolve_inline(cx, (reached, slots), &mut inline, index, (limit, false));
    }
}
