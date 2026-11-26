use std::hash::Hash;

use haste_codegen::RelocKind;
use rustc_hash::FxHashSet as HashSet;

use crate::lower::{RodataTarget, SymbolSource, VtableEntry, Word};

use super::vdce::VtableMask;
use super::{AotFn, AotState};

pub(super) enum FnEdge<'a> {
    Sym(&'a SymbolSource, bool),
    Alias(usize),
}

pub(super) fn for_each_fn_edge(entry: &AotFn, mut on_edge: impl FnMut(FnEdge<'_>)) {
    match entry {
        AotFn::Direct { code, relocs } => {
            for reloc in code.relocs() {
                if reloc.as_runtime().is_some() {
                    continue;
                }
                if let Some(sym) = relocs.get(reloc.symbol as usize) {
                    on_edge(FnEdge::Sym(
                        sym,
                        matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32),
                    ));
                }
            }
            for drop in code.unwind_drops() {
                if let Some(sym) = relocs.get(drop.drop_sym as usize) {
                    on_edge(FnEdge::Sym(sym, true));
                }
            }
        }
        AotFn::Dict { relocs, .. } => {
            for sym in relocs {
                on_edge(FnEdge::Sym(sym, false));
            }
        }
        AotFn::Alias(leader) => on_edge(FnEdge::Alias(*leader)),
    }
}

pub(super) struct FirstSeen<K> {
    seen: HashSet<K>,
    pub(super) list: Vec<K>,
}

impl<K> Default for FirstSeen<K> {
    fn default() -> Self {
        Self {
            seen: HashSet::default(),
            list: Vec::new(),
        }
    }
}

impl<K: Copy + Eq + Hash> FirstSeen<K> {
    fn insert(&mut self, key: K) {
        if self.seen.insert(key) {
            self.list.push(key);
        }
    }
}

pub(in crate::image) struct AotReach {
    pub(super) fns: Vec<bool>,
    pub(super) order: Vec<usize>,
    rodata: Vec<bool>,
    pub(super) rodata_order: Vec<usize>,
    vtables: Vec<bool>,
    pub(super) vtable_order: Vec<usize>,
    pub(super) slots: FirstSeen<usize>,
    pub(super) statics: FirstSeen<usize>,
    pub(super) promoted: FirstSeen<(usize, u32)>,
    pub(super) thread_slots: FirstSeen<usize>,
    work_fns: Vec<usize>,
    work_rodata: Vec<usize>,
    work_vtables: Vec<usize>,
}

fn mark_id(seen: &mut Vec<bool>, order: &mut Vec<usize>, work: &mut Vec<usize>, id: usize) {
    if id >= seen.len() {
        seen.resize(id + 1, false);
    }
    if !seen[id] {
        seen[id] = true;
        order.push(id);
        work.push(id);
    }
}

impl AotReach {
    pub(in crate::image) fn new(len: usize, roots: &[usize]) -> Self {
        let mut this = Self {
            fns: vec![false; len],
            order: Vec::new(),
            rodata: Vec::new(),
            rodata_order: Vec::new(),
            vtables: Vec::new(),
            vtable_order: Vec::new(),
            slots: FirstSeen::default(),
            statics: FirstSeen::default(),
            promoted: FirstSeen::default(),
            thread_slots: FirstSeen::default(),
            work_fns: Vec::new(),
            work_rodata: Vec::new(),
            work_vtables: Vec::new(),
        };
        for &root in roots {
            this.mark_fn(root);
        }
        this
    }

    pub(super) fn collect(state: &AotState, roots: &[usize], vmask: &VtableMask) -> Self {
        let mut this = Self::new(state.functions.len(), roots);
        this.rodata.resize(state.rodata.len(), false);
        this.vtables.resize(state.vtables.len(), false);
        this.drain(state, None, vmask);
        this
    }

    fn mark_fn(&mut self, idx: usize) {
        if idx < self.fns.len() && !self.fns[idx] {
            self.fns[idx] = true;
            self.order.push(idx);
            self.work_fns.push(idx);
        }
    }

    fn mark_sym(&mut self, sym: &SymbolSource) {
        match sym {
            SymbolSource::Slot(idx) => {
                self.slots.insert(*idx);
                self.mark_fn(*idx);
            }
            SymbolSource::Vtable(id) => mark_id(
                &mut self.vtables,
                &mut self.vtable_order,
                &mut self.work_vtables,
                id.0 as usize,
            ),
            SymbolSource::Rodata(id) => mark_id(
                &mut self.rodata,
                &mut self.rodata_order,
                &mut self.work_rodata,
                id.0 as usize,
            ),
            SymbolSource::StaticCell { index, .. } => {
                self.statics.insert(*index);
                self.mark_fn(*index);
            }
            SymbolSource::PromotedCell { owner, seq, .. } => {
                self.promoted.insert((*owner, *seq));
            }
            SymbolSource::ThreadSlot(index) => {
                self.thread_slots.insert(*index);
            }
            SymbolSource::Host(_) | SymbolSource::Word(_) => {}
        }
    }

    fn drain_fns(&mut self, state: &AotState, pending: &mut Option<(&[bool], &mut Vec<usize>)>) {
        while let Some(idx) = self.work_fns.pop() {
            if let Some((compiled, next_wave)) = pending.as_mut()
                && !compiled.get(idx).copied().unwrap_or(true)
            {
                next_wave.push(idx);
                continue;
            }
            let Some(entry) = state.functions.get(idx).and_then(Option::as_ref) else {
                continue;
            };
            for_each_fn_edge(entry, |edge| match edge {
                FnEdge::Sym(sym, direct) => {
                    if direct && let SymbolSource::Slot(callee) = *sym {
                        self.mark_fn(callee);
                    } else {
                        self.mark_sym(sym);
                    }
                }
                FnEdge::Alias(leader) => self.mark_fn(leader),
            });
        }
    }

    fn drain_rodata_vtables(&mut self, state: &AotState, vmask: &VtableMask) {
        while let Some(id) = self.work_rodata.pop() {
            for reloc in state.rodata.get(id).into_iter().flat_map(|(_, r)| r) {
                match reloc.target {
                    RodataTarget::Rodata(target) => mark_id(
                        &mut self.rodata,
                        &mut self.rodata_order,
                        &mut self.work_rodata,
                        target.0 as usize,
                    ),
                    RodataTarget::Fn(fn_idx) => self.mark_fn(fn_idx),
                }
            }
        }
        while let Some(id) = self.work_vtables.pop() {
            for (slot, entry) in state.vtables.get(id).into_iter().flatten().enumerate() {
                if !vmask.slot_live(id, slot) {
                    continue;
                }
                match *entry {
                    VtableEntry::Word(Word::Int(_)) => {}
                    VtableEntry::Word(Word::Code(fn_idx)) => self.mark_fn(fn_idx),
                    VtableEntry::Super(super_id) => mark_id(
                        &mut self.vtables,
                        &mut self.vtable_order,
                        &mut self.work_vtables,
                        super_id.0 as usize,
                    ),
                }
            }
        }
    }

    fn drain(
        &mut self,
        state: &AotState,
        mut pending: Option<(&[bool], &mut Vec<usize>)>,
        vmask: &VtableMask,
    ) {
        while !self.work_fns.is_empty()
            || !self.work_rodata.is_empty()
            || !self.work_vtables.is_empty()
        {
            self.drain_fns(state, &mut pending);
            self.drain_rodata_vtables(state, vmask);
        }
        if let Some((_, next_wave)) = pending {
            self.work_fns.extend_from_slice(next_wave);
        }
    }
}
