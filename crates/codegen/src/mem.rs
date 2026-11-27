mod alias;
pub(crate) mod escape;
mod frame;
mod loop_mem;
mod roots;

use rustc_hash::FxHashMap as HashMap;

pub(crate) use alias::{
    KnownLoads, clobbers, clobbers_load, in_range, is_atomic_barrier, is_hard_barrier, is_rodata,
    may_alias, may_alias_at,
};
pub(crate) use frame::drop_unread_frame_stores;
pub(crate) use loop_mem::{LoopMem, PromoteBase, PromoteCand};
pub(crate) use roots::compute_affine;

use crate::cfg::Cfg;
use crate::ir::{Function, Inst, StackSlot, Value};
use crate::reloc::RuntimeSymbol;
use roots::{TrackRoot, compute_affine_in, compute_tracked_roots, slot_addresses};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MemRoot {
    Slot(StackSlot),
    NoAliasParam(u16),
    ReadonlyParam(u16),
    Global(u32),
    UniqueHeap(Option<Value>),
    ThreadCell(u32),
    CallerLoc,
    Unknown,
}

impl MemRoot {
    /// Whether the root names one known object: a stack slot, a noalias or readonly parameter, or a unique heap allocation.
    pub(crate) fn is_identified(&self) -> bool {
        matches!(
            self,
            Self::Slot(_) | Self::NoAliasParam(_) | Self::ReadonlyParam(_) | Self::UniqueHeap(_)
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MemBase {
    Slot(StackSlot),
    Global(u32),
    Addr(Value),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MemLoc {
    pub root: MemRoot,
    pub base: MemBase,
    pub offset: i64,
    pub size: u32,
}

impl MemLoc {
    pub(crate) fn new(root: MemRoot, base: MemBase, offset: i64, size: u32) -> Self {
        Self {
            root,
            base,
            offset,
            size,
        }
    }

    pub(crate) fn slot(slot: StackSlot, offset: i64, size: u32) -> Self {
        Self {
            root: MemRoot::Slot(slot),
            base: MemBase::Slot(slot),
            offset,
            size,
        }
    }

    pub(crate) fn overlaps(self, other: Self) -> bool {
        if self.size == 0 || other.size == 0 {
            return false;
        }
        let (a0, b0) = (i128::from(self.offset), i128::from(other.offset));
        let (a1, b1) = (a0 + i128::from(self.size), b0 + i128::from(other.size));
        a0 < b1 && b0 < a1
    }
}

#[derive(Default)]
pub(crate) struct Provenance {
    pub slot_offsets: Vec<Option<(StackSlot, i32)>>,
    pub unescaped_slots: Vec<bool>,
    pub addr_unescaped_slots: Vec<bool>,
    pub unwind_target_slots: Vec<bool>,
    pub call_readonly_slots: Vec<bool>,
    pub return_only_slots: Vec<bool>,
    pub noalias_escaped: Vec<bool>,
    pub param_roots: Vec<MemRoot>,
    pub symbol_defs: Vec<Option<u32>>,
    frame_runtime: Vec<Option<RuntimeSymbol>>,
    pub ro_bytes: Vec<u32>,
    /// Values that are parameters of an entry block with no predecessors: addresses the caller made before this frame existed.
    pub incoming: Vec<bool>,
    pub affine: Vec<(Value, i64)>,
    pub spans: Vec<Option<(MemBase, i64, i64)>>,
    pub has_caller_loc: bool,
    frontiers: HashMap<u32, escape::Frontier>,
    inst_at: Vec<(u32, u32)>,
    frame: frame::FrameFacts,
}

impl Provenance {
    /// The coroutine frame runtime that `callee` names, if any: `Malloc` writes no memory the function can read, and `Free` writes only the block it frees.
    pub(crate) fn frame_runtime(&self, callee: Value) -> Option<RuntimeSymbol> {
        self.frame_runtime.get(callee.index()).copied().flatten()
    }

    fn build(
        func: &Function,
        cfg: &Cfg,
        (is_spill, positions): (Option<&[bool]>, bool),
        affine: Option<Vec<(Value, i64)>>,
    ) -> Self {
        let affine = affine.unwrap_or_else(|| compute_affine_in(func, Some(&cfg.rpo)));
        let (tracked, has_caller_loc, need_spans, symbol_defs, frame_runtime) =
            compute_tracked_roots(func, cfg, &affine);
        let mut prov = Self::from_roots(
            func,
            cfg,
            (&tracked, has_caller_loc, symbol_defs, frame_runtime),
            affine,
        );
        let esc = escape::classify_uses(
            func,
            cfg,
            &prov.slot_offsets,
            &tracked,
            (is_spill, positions),
        );
        if positions {
            prov.frontiers = escape::frontiers(func, &esc);
        }
        prov.unescaped_slots = esc.unescaped;
        prov.addr_unescaped_slots = esc.addr_unescaped;
        prov.unwind_target_slots = esc.unwind_target;
        prov.call_readonly_slots = esc.call_readonly;
        prov.return_only_slots = esc.return_only;
        prov.noalias_escaped = esc.noalias;
        if !prov.frontiers.is_empty() {
            prov.inst_at = vec![(u32::MAX, 0); func.insts.len()];
            for &b in &cfg.rpo {
                for (pos, &inst) in func.blocks[b.index()].insts.iter().enumerate() {
                    prov.inst_at[inst.index()] = (b.index() as u32, pos as u32);
                }
            }
        }
        if need_spans {
            prov.spans = loop_mem::compute_spans(func, cfg, &prov.slot_offsets, &prov.symbol_defs);
        }
        prov
    }

    fn from_roots(
        func: &Function,
        cfg: &Cfg,
        (tracked, has_caller_loc, symbol_defs, frame_runtime): (
            &[TrackRoot],
            bool,
            Vec<Option<u32>>,
            Vec<Option<RuntimeSymbol>>,
        ),
        affine: Vec<(Value, i64)>,
    ) -> Self {
        let slot_offsets = slot_addresses(func);
        let mut param_roots = vec![MemRoot::Unknown; func.types.len()];
        for (i, t) in tracked.iter().enumerate() {
            if let TrackRoot::Set(root) = *t {
                param_roots[i] = root;
            }
        }
        let mut ro_bytes = vec![0_u32; func.types.len()];
        let mut incoming = vec![false; func.types.len()];
        if let Some(entry) = cfg.entry {
            let fresh = cfg.pred_count(entry) == 0;
            for (&param, abi) in func.blocks[entry.index()]
                .params
                .iter()
                .zip(&func.signature.params)
            {
                ro_bytes[param.index()] = abi.deref_bytes;
                incoming[param.index()] = fresh;
            }
        }
        Self {
            slot_offsets,
            param_roots,
            symbol_defs,
            frame_runtime,
            ro_bytes,
            incoming,
            affine,
            has_caller_loc,
            frame: frame::FrameFacts::new(func, cfg),
            ..Self::default()
        }
    }

    /// The address facts alone, for `loc` and `ro_bytes`: no escape classification and no loop spans.
    pub(crate) fn addresses(func: &Function, cfg: &Cfg) -> Self {
        let affine = compute_affine_in(func, Some(&cfg.rpo));
        let (tracked, has_caller_loc, _, symbol_defs, frame_runtime) =
            compute_tracked_roots(func, cfg, &affine);
        Self::from_roots(
            func,
            cfg,
            (&tracked, has_caller_loc, symbol_defs, frame_runtime),
            affine,
        )
    }

    pub(crate) fn new(func: &Function, cfg: &Cfg) -> Self {
        Self::build(func, cfg, (None, false), None)
    }

    /// Like `new`, taking `affine` when the caller already holds `compute_affine` of this exact function, and also finding where each slot escapes so a call before that point may skip it.
    pub(crate) fn with_affine(
        func: &Function,
        cfg: &Cfg,
        affine: Option<Vec<(Value, i64)>>,
    ) -> Self {
        Self::build(func, cfg, (None, true), affine)
    }

    pub(crate) fn compute(func: &Function) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, (None, false), None)
    }

    pub(crate) fn compute_with_spills(func: &Function, is_spill: Option<&[bool]>) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, (is_spill, false), None)
    }

    pub(crate) fn loc(&self, addr: Value, offset: i64, size: u32) -> MemLoc {
        if let Some((slot, base)) = self.slot_offsets.get(addr.index()).copied().flatten()
            && let Some(total) = i64::from(base).checked_add(offset)
        {
            return MemLoc::slot(slot, total, size);
        }
        let (root_val, base_off) = self.affine.get(addr.index()).copied().unwrap_or((addr, 0));
        let (eff_val, eff_off) = match base_off.checked_add(offset) {
            Some(total) => (root_val, total),
            None => (addr, offset),
        };
        if let Some((slot, base)) = self.slot_offsets.get(eff_val.index()).copied().flatten()
            && let Some(total) = i64::from(base).checked_add(eff_off)
        {
            return MemLoc::slot(slot, total, size);
        }
        let base = match self.symbol_defs.get(eff_val.index()).copied().flatten() {
            Some(sym) => MemBase::Global(sym),
            None => MemBase::Addr(eff_val),
        };
        let root = self
            .param_roots
            .get(eff_val.index())
            .copied()
            .unwrap_or(MemRoot::Unknown);
        MemLoc::new(root, base, eff_off, size)
    }

    pub(crate) fn loc_span(&self, loc: MemLoc) -> (MemBase, i128, i128) {
        let size = i128::from(loc.size);
        let off = i128::from(loc.offset);
        if let MemBase::Addr(v) = loc.base
            && let Some((base, blo, bhi)) = self.spans.get(v.index()).copied().flatten()
        {
            return (base, i128::from(blo) + off, i128::from(bhi) + off + size);
        }
        (loc.base, off, off + size)
    }

    /// The `(block, pos)` recorded for `inst` when this provenance was built, if any.
    pub(crate) fn inst_loc(&self, inst: Inst) -> Option<(u32, u32)> {
        self.inst_at
            .get(inst.index())
            .copied()
            .filter(|&(block, _)| block != u32::MAX)
    }

    /// Whether `slot`'s address may have left the function by the time `at` runs; `None` means anywhere.
    pub(crate) fn slot_escaped_at(&self, slot: StackSlot, at: Option<Inst>) -> bool {
        let (Some(inst), Some(frontier)) = (at, self.frontiers.get(&slot.0)) else {
            return true;
        };
        self.inst_loc(inst).is_none_or(|loc| frontier.reached(loc))
    }

    /// The precomputed escape frontier of `slot`, when one exists.
    pub(crate) fn slot_frontier(&self, slot: StackSlot) -> Option<&escape::Frontier> {
        self.frontiers.get(&slot.0)
    }

    /// Whether `slot` has escaped by `at`, returning `false` for slots whose address never escapes.
    pub(crate) fn slot_is_escaped_at(&self, slot: StackSlot, at: Option<Inst>) -> bool {
        !self.slot_addr_unescaped(slot) && self.slot_escaped_at(slot, at)
    }

    /// Whether `base` is an address the caller passed in, which cannot point into this frame's slots.
    pub(crate) fn is_incoming(&self, base: MemBase) -> bool {
        matches!(base, MemBase::Addr(v) if self.incoming.get(v.index()).copied().unwrap_or(false))
    }

    pub(crate) fn slot_unescaped(&self, slot: StackSlot) -> bool {
        self.unescaped_slots
            .get(slot.index())
            .copied()
            .unwrap_or(false)
    }

    /// Whether `slot`'s address never escapes to normal memory ops or calls, ignoring unwind-drop table reads.
    pub(crate) fn slot_addr_unescaped(&self, slot: StackSlot) -> bool {
        self.addr_unescaped_slots
            .get(slot.index())
            .copied()
            .unwrap_or(false)
    }

    /// Whether `slot`'s address is stored into an indirect/wide unwind-drop slot and therefore read on panic unwind.
    pub(crate) fn slot_unwind_target(&self, slot: StackSlot) -> bool {
        self.unwind_target_slots
            .get(slot.index())
            .copied()
            .unwrap_or(false)
    }

    pub(crate) fn escaped_slots(&self) -> Vec<bool> {
        self.unescaped_slots.iter().map(|&u| !u).collect()
    }

    pub(crate) fn addr_escaped_slots(&self) -> Vec<bool> {
        self.addr_unescaped_slots.iter().map(|&u| !u).collect()
    }

    /// Whether every call that takes `slot`'s address takes it as the memory for its result, so only those calls write it.
    pub(crate) fn slot_return_only(&self, slot: StackSlot) -> bool {
        self.return_only_slots
            .get(slot.index())
            .copied()
            .unwrap_or(false)
    }

    pub(crate) fn slot_call_readonly(&self, slot: StackSlot) -> bool {
        self.call_readonly_slots
            .get(slot.index())
            .copied()
            .unwrap_or(false)
    }

    pub(crate) fn noalias_unescaped(&self, param: u16) -> bool {
        !self
            .noalias_escaped
            .get(usize::from(param))
            .copied()
            .unwrap_or(true)
    }
}
