use std::sync::Arc;

use haste_codegen::{Code, CodeUnwindDrop, Effort, Site, Target};
use rustc_hash::FxHashMap as HashMap;

use crate::lower::{
    self, FootprintId, RelocPlan, RodataId, RodataReloc, Slots, SymbolSource, VtableEntry,
    VtableId, Word,
};
use crate::names::ItemRef;
use crate::outline::tree::Profile;
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::{Image, ImageMode};

mod compile;
mod data;
mod group;

/// Minimum template code size in bytes before sharing a dictionary body.
pub const DICT_MIN_BYTES: usize = 512;

/// Minimum distinct relocation vectors before sharing a dictionary body.
pub const DICT_MIN_FANOUT: usize = 8;

/// Whether `HASTE_VERIFY_STENCIL` asks for stencil instantiations to be checked against concrete lowering.
pub fn verify_stencil() -> bool {
    crate::with_cfg(|c| c.verify_stencil)
}

const DEFAULT_TIER_UP: u32 = 64;
const DICT_STUB_SPACE: usize = 48;

#[derive(Clone, Copy)]
pub(super) struct DictConfig {
    pub min_bytes: usize,
    pub min_fanout: usize,
    pub tier_up: u32,
    pub verify: bool,
}

impl DictConfig {
    pub fn from_env() -> Self {
        let min_bytes = env_usize("HASTE_DICT_MIN_BYTES").unwrap_or(DICT_MIN_BYTES);
        let default_fanout = if min_bytes == 0 { 1 } else { DICT_MIN_FANOUT };
        let min_fanout = env_usize("HASTE_DICT_MIN_FANOUT")
            .unwrap_or(default_fanout)
            .max(1);
        let tier_up = env_usize("HASTE_JIT_TIER_UP").map_or(DEFAULT_TIER_UP, |v| v as u32);
        Self {
            min_bytes,
            min_fanout,
            tier_up,
            verify: verify_stencil(),
        }
    }

    pub(super) fn shares_dict(self, bytes: usize, fanout: usize) -> bool {
        bytes > self.min_bytes && fanout >= self.min_fanout
    }
}

pub(super) fn target() -> Target {
    crate::host_target()
}

impl From<Profile> for Effort {
    fn from(profile: Profile) -> Effort {
        match profile {
            Profile::Debug => Effort::Quick,
            Profile::Release => Effort::Full,
        }
    }
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

#[derive(Clone)]
pub(super) enum CachedTemplate {
    Stencil {
        code: Arc<Code>,
        reloc_plan: Arc<RelocPlan>,
    },
    Dict {
        body_exec: usize,
        code: Arc<Code>,
        reloc_plan: Arc<RelocPlan>,
    },
}

impl CachedTemplate {
    pub(super) fn reloc_plan(&self) -> &Arc<RelocPlan> {
        match self {
            Self::Stencil { reloc_plan, .. } | Self::Dict { reloc_plan, .. } => reloc_plan,
        }
    }
}

pub(super) struct JitTemplate {
    pub stencil: Arc<Code>,
    pub dict: Option<(usize, Arc<Code>, Arc<RelocPlan>)>,
    pub fanout: usize,
    pub reloc_plan: Arc<RelocPlan>,
}

pub(super) struct TierUpEntry {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub symbols: Vec<u64>,
}

#[derive(Clone)]
pub(super) struct FirstJit {
    pub index: usize,
    pub code: Arc<Code>,
    pub symbols: Vec<u64>,
    pub exec: usize,
    pub reloc_plan: Arc<RelocPlan>,
}

#[derive(Clone)]
pub(super) enum FirstJitState {
    First(FirstJit),
    Promoted,
}

pub(super) struct StencilState {
    pub cfg: DictConfig,
    pub aliases: HashMap<(ItemRef, FootprintId, Vec<u64>), (usize, bool)>,
    pub templates: HashMap<(ItemRef, FootprintId), CachedTemplate>,
    pub jit_templates: HashMap<(ItemRef, FootprintId), JitTemplate>,
    pub first_jit: HashMap<ItemRef, FirstJitState>,
    pub tier_ups: HashMap<usize, TierUpEntry>,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            cfg: DictConfig::from_env(),
            aliases: HashMap::default(),
            templates: HashMap::default(),
            jit_templates: HashMap::default(),
            first_jit: HashMap::default(),
            tier_ups: HashMap::default(),
        }
    }
}

impl StencilState {
    pub fn clear(&mut self) {
        self.cfg = DictConfig::from_env();
        self.aliases.clear();
        self.templates.clear();
        self.jit_templates.clear();
        self.first_jit.clear();
        self.tier_ups.clear();
    }
}

pub(super) struct DistinctInst {
    pub leader: usize,
    pub followers: Vec<usize>,
    pub symbols: Vec<u64>,
    pub relocs: Option<Vec<SymbolSource>>,
}

impl DistinctInst {
    fn reloc_count(&self) -> usize {
        self.relocs.as_ref().map_or(self.symbols.len(), Vec::len)
    }
}

pub(super) struct GroupWork {
    pub item: ItemRef,
    pub footprint: FootprintId,
    pub distinct: Vec<DistinctInst>,
    pub leader_out: Option<lower::StencilOutput>,
}

impl GroupWork {
    pub fn representative(&self) -> usize {
        self.distinct[0].leader
    }

    pub fn all_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.distinct
            .iter()
            .flat_map(|inst| std::iter::once(inst.leader).chain(inst.followers.iter().copied()))
    }
}

pub(super) enum VerifyRecord {
    Patched {
        exec: usize,
        len: usize,
        sites: Vec<Site>,
        unwind_drops: Vec<CodeUnwindDrop>,
    },
    Compiled {
        code: Arc<Code>,
        symbols: Vec<u64>,
    },
    Dict {
        body_exec: usize,
        stub_exec: usize,
        stub_bytes: Vec<u8>,
        code: Arc<Code>,
    },
}

pub(super) struct FreshSlots<'a>(pub(super) &'a Image, pub(super) lower::Bodies);

impl Slots for FreshSlots<'_> {
    fn entry(&self, index: usize) -> i64 {
        self.0.entry(index)
    }
    fn data(&self, bytes: &[u8]) -> i64 {
        self.0.data(bytes)
    }
    fn zeroed(&self, len: usize) -> i64 {
        self.0.zeroed(len)
    }
    fn table(&self, words: &[Word]) -> i64 {
        self.0.table(words)
    }
    fn cell(&self, index: usize, size: u32) -> i64 {
        self.0.cell(index, size)
    }
    fn want(&self, instance: crate::sema::reach::Instance) {
        self.0.want(instance);
    }
    fn wanted_len(&self) -> usize {
        self.0.wanted_len()
    }
    fn bodies(&self) -> &lower::Bodies {
        &self.1
    }
    fn allow_inline(&self) -> bool {
        self.0.allow_inline()
    }
    fn low_entries(&self) -> bool {
        self.0.low_entries()
    }
    fn rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        self.0.rodata(bytes, relocs)
    }
    fn static_rodata(&self, index: usize, bytes: &[u8]) -> RodataId {
        self.0.static_rodata(index, bytes)
    }
    fn vtable_id(&self, entries: &[VtableEntry]) -> VtableId {
        self.0.vtable_id(entries)
    }
    fn vtable_entries(&self, id: VtableId) -> Option<Vec<VtableEntry>> {
        self.0.vtable_entries(id)
    }
    fn resolve_symbol(&self, sym: &SymbolSource) -> u64 {
        self.0.resolve_symbol(sym)
    }
}

pub(super) fn item_err(cx: &Context<'_>, item: ItemRef, err: impl std::fmt::Display) -> String {
    format!("{}: {err:#}", cx.name(item))
}

pub(super) fn aot_relocs_valid(code: &Code, relocs_len: usize) -> bool {
    code.relocs()
        .iter()
        .filter_map(|r| r.symbol_index())
        .all(|idx| idx < relocs_len)
        && code
            .unwind_drops()
            .iter()
            .all(|u| (u.drop_sym as usize) < relocs_len)
}

pub(super) fn mode_dict_key(direct_key: u64) -> u64 {
    direct_key ^ 0xD1C7_A070_D1C7_A070
}

pub(super) fn mode_stencil_key(mode: ImageMode, item_key: u64) -> u64 {
    if mode == ImageMode::Aot {
        item_key ^ 0xA070_A070_A070_A070
    } else {
        item_key
    }
}

fn compile_traced(
    cx: &mut Context<'_>,
    reached: &Reached,
    index: usize,
    func: haste_codegen::Function,
    target: &haste_codegen::Target,
) -> std::result::Result<haste_codegen::Code, haste_codegen::Error> {
    if !haste_codegen::trace_enabled() {
        return haste_codegen::compile(func, target);
    }
    let label = cx.symbol_text(reached.list[index].0);
    haste_codegen::with_trace_label(&label, || haste_codegen::compile(func, target))
}
