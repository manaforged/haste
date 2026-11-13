use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use haste_codegen::{Code, RelocKind, Target};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::aot::{Executable, Symbol};
use crate::host::HostSymbol;
use crate::lower::{RodataReloc, RodataTarget, SymbolSource, VtableEntry, Word};
use crate::sync::lock;

use super::Image;

#[derive(Clone)]
pub(super) enum AotFn {
    Direct { code: Arc<Code>, relocs: Vec<SymbolSource> },
    Dict { body: Arc<Code>, relocs: Vec<SymbolSource> },
    Alias(usize),
}

pub(super) enum AotStatic {
    Local(u32),
    Foreign { weak: bool, host: Option<HostSymbol> },
}

pub(crate) struct AotState {
    pub(super) rodata: Vec<(Vec<u8>, Vec<RodataReloc>)>,
    pub(super) vtables: Vec<Vec<VtableEntry>>,
    pub(super) statics: HashMap<usize, AotStatic>,
    pub(super) promoted: Vec<((usize, u32), u32)>,
    pub(super) promoted_seen: HashMap<(usize, u32), usize>,
    pub(super) foreign_fns: HashMap<usize, HostSymbol>,
    pub(super) functions: Vec<Option<AotFn>>,
}

impl AotState {
    pub(super) fn new(len: usize) -> Mutex<Self> {
        Mutex::new(Self {
            rodata: Vec::new(),
            vtables: Vec::new(),
            statics: HashMap::default(),
            promoted: Vec::new(),
            promoted_seen: HashMap::default(),
            foreign_fns: HashMap::default(),
            functions: vec![None; len],
        })
    }

    pub(super) fn ensure_len(&mut self, len: usize) {
        if self.functions.len() < len {
            self.functions.resize(len, None);
        }
    }
}

pub(super) struct AotReach {
    fns: Vec<bool>,
    rodata: Vec<bool>,
    vtables: Vec<bool>,
    slots: HashSet<usize>,
    slot_list: Vec<usize>,
    statics: HashSet<usize>,
    promoted: HashSet<(usize, u32)>,
    work_fns: Vec<usize>,
    work_rodata: Vec<usize>,
    work_vtables: Vec<usize>,
}

impl AotReach {
    pub(super) fn new(len: usize, roots: [usize; 2]) -> Self {
        let mut this = Self {
            fns: vec![false; len],
            rodata: Vec::new(),
            vtables: Vec::new(),
            slots: HashSet::default(),
            slot_list: Vec::new(),
            statics: HashSet::default(),
            promoted: HashSet::default(),
            work_fns: Vec::new(),
            work_rodata: Vec::new(),
            work_vtables: Vec::new(),
        };
        for root in roots {
            this.mark_fn(root);
        }
        this
    }

    fn collect(state: &AotState, roots: [usize; 2]) -> Self {
        let mut this = Self::new(state.functions.len(), roots);
        this.rodata.resize(state.rodata.len(), false);
        this.vtables.resize(state.vtables.len(), false);
        this.drain(state, None);
        this.slot_list.sort_unstable();
        this
    }

    fn mark_fn(&mut self, idx: usize) {
        if idx < self.fns.len() && !self.fns[idx] {
            self.fns[idx] = true;
            self.work_fns.push(idx);
        }
    }

    fn mark_rodata(&mut self, id: usize) {
        if id >= self.rodata.len() {
            self.rodata.resize(id + 1, false);
        }
        if !self.rodata[id] {
            self.rodata[id] = true;
            self.work_rodata.push(id);
        }
    }

    fn mark_vtable(&mut self, id: usize) {
        if id >= self.vtables.len() {
            self.vtables.resize(id + 1, false);
        }
        if !self.vtables[id] {
            self.vtables[id] = true;
            self.work_vtables.push(id);
        }
    }

    fn mark_sym(&mut self, sym: &SymbolSource) {
        match sym {
            SymbolSource::Slot(idx) => {
                if self.slots.insert(*idx) {
                    self.slot_list.push(*idx);
                }
                self.mark_fn(*idx);
            }
            SymbolSource::Vtable(id) => self.mark_vtable(id.0 as usize),
            SymbolSource::Rodata(id) => self.mark_rodata(id.0 as usize),
            SymbolSource::StaticCell { index, .. } => {
                self.statics.insert(*index);
                self.mark_fn(*index);
            }
            SymbolSource::PromotedCell { owner, seq, .. } => {
                self.promoted.insert((*owner, *seq));
            }
            SymbolSource::Host(_) | SymbolSource::Word(_) | SymbolSource::ThreadSlot(_) => {}
        }
    }

    pub(super) fn drain(&mut self, state: &AotState, mut pending: Option<(&[bool], &mut Vec<usize>)>) {
        while !self.work_fns.is_empty() || !self.work_rodata.is_empty() || !self.work_vtables.is_empty() {
            while let Some(idx) = self.work_fns.pop() {
                if let Some((compiled, next_wave)) = pending.as_mut()
                    && !compiled.get(idx).copied().unwrap_or(true)
                {
                    next_wave.push(idx);
                    continue;
                }
                match state.functions.get(idx).and_then(Option::as_ref) {
                    Some(AotFn::Direct { code, relocs }) => {
                        for reloc in code.relocs() {
                            if reloc.as_runtime().is_some() {
                                continue;
                            }
                            let Some(sym) = relocs.get(reloc.symbol as usize) else { continue };
                            if matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32)
                                && let SymbolSource::Slot(callee) = *sym
                            {
                                self.mark_fn(callee);
                            } else {
                                self.mark_sym(sym);
                            }
                        }
                    }
                    Some(AotFn::Dict { relocs, .. }) => {
                        for sym in relocs {
                            self.mark_sym(sym);
                        }
                    }
                    Some(AotFn::Alias(leader)) => self.mark_fn(*leader),
                    None => {}
                }
            }
            while let Some(id) = self.work_rodata.pop() {
                if let Some((_, relocs)) = state.rodata.get(id) {
                    for reloc in relocs {
                        let RodataTarget::Rodata(target) = reloc.target;
                        self.mark_rodata(target.0 as usize);
                    }
                }
            }
            while let Some(id) = self.work_vtables.pop() {
                if let Some(entries) = state.vtables.get(id) {
                    for entry in entries {
                        match *entry {
                            VtableEntry::Word(Word::Int(_)) => {}
                            VtableEntry::Word(Word::Code(fn_idx)) => self.mark_fn(fn_idx),
                            VtableEntry::Super(super_id) => self.mark_vtable(super_id.0 as usize),
                        }
                    }
                }
            }
        }
        if let Some((_, next_wave)) = pending {
            self.work_fns.extend_from_slice(next_wave);
        }
    }
}

struct SymTable<'a> {
    slot_offs: &'a HashMap<usize, u32>,
    rodata_offs: &'a [Option<u32>],
    vtable_offs: &'a [Option<u32>],
    static_syms: &'a HashMap<usize, Symbol>,
    promoted_syms: &'a HashMap<(usize, u32), Symbol>,
    thread_slots: &'a HashMap<usize, u64>,
}

impl SymTable<'_> {
    fn translate(&self, sym: &SymbolSource) -> Result<Symbol> {
        Ok(match sym {
            SymbolSource::Slot(index) => Symbol::Rodata(*self.slot_offs.get(index).context("missing slot offset")?),
            SymbolSource::Vtable(id) => Symbol::Rodata(self.vtable_offs.get(id.0 as usize).copied().flatten().context("missing vtable offset")?),
            SymbolSource::Rodata(id) => Symbol::Rodata(self.rodata_offs.get(id.0 as usize).copied().flatten().context("missing rodata offset")?),
            SymbolSource::StaticCell { index, .. } => self.static_syms.get(index).cloned().context("missing static cell")?,
            SymbolSource::PromotedCell { owner, seq, .. } => self.promoted_syms.get(&(*owner, *seq)).cloned().context("missing promoted cell")?,
            SymbolSource::Host(host) => Symbol::Host(host.clone()),
            SymbolSource::Word(word) => Symbol::Word(*word),
            SymbolSource::ThreadSlot(index) => Symbol::Word(self.thread_slots.get(index).copied().unwrap_or(*index as u64)),
        })
    }
}

fn emit_dict_bodies(state: &AotState, live_fns: &[bool], exe: &mut Executable) -> Result<HashMap<usize, u32>> {
    let mut dict_bodies: HashMap<usize, u32> = HashMap::default();
    for (idx, item) in state.functions.iter().enumerate() {
        if !live_fns[idx] {
            continue;
        }
        let Some(AotFn::Dict { body, .. }) = item else { continue };
        let key = Arc::as_ptr(body) as usize;
        if dict_bodies.contains_key(&key) {
            continue;
        }
        let off = exe.text(body.bytes());
        for reloc in body.relocs() {
            let rt = reloc.as_runtime().context("dict body contains non-runtime relocation")?;
            exe.patch(off + reloc.offset, reloc.kind, Symbol::Runtime(rt));
        }
        dict_bodies.insert(key, off);
    }
    Ok(dict_bodies)
}

fn slot_target(fn_offs: &[Option<u32>], foreign_fns: &HashMap<usize, HostSymbol>, idx: usize) -> Symbol {
    if let Some(off) = fn_offs.get(idx).copied().flatten() {
        Symbol::Text(off)
    } else if let Some(host) = foreign_fns.get(&idx) {
        Symbol::Host(host.clone())
    } else {
        Symbol::Host(HostSymbol::Abort)
    }
}

fn emit_functions(state: &AotState, live_fns: &[bool], target: &Target, syms: &SymTable<'_>, dict_bodies: &HashMap<usize, u32>, exe: &mut Executable) -> Result<Vec<Option<u32>>> {
    let mut fn_offs = vec![None; state.functions.len()];
    let mut aliases = Vec::new();
    let mut pending = Vec::new();
    for (idx, item) in state.functions.iter().enumerate() {
        if !live_fns[idx] {
            continue;
        }
        match item {
            Some(AotFn::Direct { code, relocs }) => {
                let off = exe.text(code.bytes());
                fn_offs[idx] = Some(off);
                pending.push((off, code, relocs));
            }
            Some(AotFn::Dict { body, relocs }) => {
                let dict_off = exe.rodata(&vec![0u8; relocs.len() * 8], 8);
                for (i, reloc) in relocs.iter().enumerate() {
                    exe.pointer(true, dict_off + (i * 8) as u32, syms.translate(reloc)?);
                }
                let stub = target.dict_entry_stub(0, 1usize << 40, 0, None)?;
                let [(dict_rel, dict_kind), (body_rel, body_kind)] = target.dict_stub_relocs()?;
                let off = exe.text(&stub);
                fn_offs[idx] = Some(off);
                let body_off = dict_bodies[&(Arc::as_ptr(body) as usize)];
                exe.patch(off + dict_rel, dict_kind, Symbol::Rodata(dict_off));
                exe.patch(off + body_rel, body_kind, Symbol::Text(body_off));
            }
            Some(AotFn::Alias(leader)) => aliases.push((idx, *leader)),
            None => {}
        }
    }
    for (idx, leader) in aliases {
        fn_offs[idx] = fn_offs[leader];
    }
    for (off, code, relocs) in pending {
        for reloc in code.relocs() {
            let target_sym = match reloc.as_runtime() {
                Some(rt) => Symbol::Runtime(rt),
                None if matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32) => match relocs[reloc.symbol as usize] {
                    SymbolSource::Slot(callee_idx) => slot_target(&fn_offs, &state.foreign_fns, callee_idx),
                    ref other => syms.translate(other)?,
                },
                None => syms.translate(&relocs[reloc.symbol as usize])?,
            };
            exe.patch(off + reloc.offset, reloc.kind, target_sym);
        }
    }
    Ok(fn_offs)
}

impl Image {
    pub(super) fn record_aot(&self, index: usize, entry: AotFn, followers: &[usize]) {
        if let Some(aot) = self.aot() {
            let mut state = lock(aot);
            state.ensure_len(index + 1);
            state.functions[index] = Some(entry);
            for &follower in followers {
                state.ensure_len(follower + 1);
                state.functions[follower] = Some(AotFn::Alias(index));
            }
        }
    }

    /// Freezes a compiled AOT image into a standalone executable starting at `(lang_start, user_main)`.
    pub(crate) fn to_aot(&self, target: &Target, (lang_start, user_main): (usize, usize)) -> Result<Executable> {
        let state = lock(self.aot().context("image has no AOT recording")?);
        let thread_slots = lock(&self.thread_slots);
        let live = AotReach::collect(&state, [lang_start, user_main]);
        let mut exe = Executable::new(target);

        let mut slot_offs = HashMap::default();
        for &idx in &live.slot_list {
            slot_offs.insert(idx, exe.rodata(&[0; 8], 8));
        }
        let rodata_offs: Vec<Option<u32>> = state.rodata.iter().enumerate().map(|(id, (bytes, _))| live.rodata[id].then(|| exe.rodata(bytes, 16))).collect();
        for (id, (_, relocs)) in state.rodata.iter().enumerate() {
            let Some(base) = rodata_offs[id] else { continue };
            for reloc in relocs {
                let RodataTarget::Rodata(target) = reloc.target;
                let target_off = rodata_offs.get(target.0 as usize).copied().flatten().context("missing rodata target")?;
                exe.pointer(true, base + reloc.offset, Symbol::Rodata(target_off));
            }
        }
        let vtable_offs: Vec<Option<u32>> = state.vtables.iter().enumerate().map(|(id, entries)| live.vtables[id].then(|| exe.rodata(&vec![0u8; entries.len() * 8], 8))).collect();

        let mut static_keys: Vec<usize> = state.statics.keys().copied().filter(|idx| live.statics.contains(idx)).collect();
        static_keys.sort_unstable();
        let mut static_syms = HashMap::default();
        for index in static_keys {
            let sym = match &state.statics[&index] {
                AotStatic::Local(size) => Symbol::Bss(exe.bss(*size + 1, 16)),
                AotStatic::Foreign { weak: false, host: Some(host) } => Symbol::Host(host.clone()),
                AotStatic::Foreign { weak: false, host: None } => Symbol::Data(exe.data(&[0; 8], 8)),
                AotStatic::Foreign { weak: true, host } => {
                    let off = exe.data(&[0; 8], 8);
                    if let Some(host) = host {
                        exe.pointer(false, off, Symbol::Host(host.clone()));
                    }
                    Symbol::Data(off)
                }
            };
            static_syms.insert(index, sym);
        }
        let mut promoted_syms = HashMap::default();
        for &((owner, seq), size) in &state.promoted {
            if live.promoted.contains(&(owner, seq)) {
                promoted_syms.insert((owner, seq), Symbol::Bss(exe.bss(size, 16)));
            }
        }

        let syms = SymTable {
            slot_offs: &slot_offs,
            rodata_offs: &rodata_offs,
            vtable_offs: &vtable_offs,
            static_syms: &static_syms,
            promoted_syms: &promoted_syms,
            thread_slots: &thread_slots,
        };
        let dict_bodies = emit_dict_bodies(&state, &live.fns, &mut exe)?;
        let fn_offs = emit_functions(&state, &live.fns, target, &syms, &dict_bodies, &mut exe)?;

        for (id, entries) in state.vtables.iter().enumerate() {
            let Some(base) = vtable_offs[id] else { continue };
            for (i, entry) in entries.iter().enumerate() {
                let target_sym = match *entry {
                    VtableEntry::Word(Word::Int(val)) => Symbol::Word(val),
                    VtableEntry::Word(Word::Code(fn_idx)) => slot_target(&fn_offs, &state.foreign_fns, fn_idx),
                    VtableEntry::Super(super_id) => Symbol::Rodata(vtable_offs.get(super_id.0 as usize).copied().flatten().context("missing super vtable")?),
                };
                exe.pointer(true, base + (i * 8) as u32, target_sym);
            }
        }
        for &idx in &live.slot_list {
            exe.pointer(true, slot_offs[&idx], slot_target(&fn_offs, &state.foreign_fns, idx));
        }

        let Some(start_off) = fn_offs.get(lang_start).copied().flatten() else {
            bail!("`std::rt::lang_start` failed to compile");
        };
        let Some(main_off) = fn_offs.get(user_main).copied().flatten() else {
            bail!("`main` failed to compile");
        };
        exe.entry(start_off, main_off);
        Ok(exe)
    }
}
