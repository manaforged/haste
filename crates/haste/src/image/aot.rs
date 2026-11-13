use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use haste_codegen::{Code, RelocKind, Target};
use rustc_hash::FxHashMap as HashMap;

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

pub(super) struct AotState {
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

struct SymTable<'a> {
    slot_offs: &'a [u32],
    rodata_offs: &'a [u32],
    vtable_offs: &'a [u32],
    static_syms: &'a HashMap<usize, Symbol>,
    promoted_syms: &'a HashMap<(usize, u32), Symbol>,
    thread_slots: &'a HashMap<usize, u64>,
}

impl SymTable<'_> {
    fn translate(&self, sym: &SymbolSource) -> Result<Symbol> {
        Ok(match sym {
            SymbolSource::Slot(index) => Symbol::Rodata(*self.slot_offs.get(*index).context("missing slot offset")?),
            SymbolSource::Vtable(id) => Symbol::Rodata(*self.vtable_offs.get(id.0 as usize).context("missing vtable offset")?),
            SymbolSource::Rodata(id) => Symbol::Rodata(*self.rodata_offs.get(id.0 as usize).context("missing rodata offset")?),
            SymbolSource::StaticCell { index, .. } => self.static_syms.get(index).cloned().context("missing static cell")?,
            SymbolSource::PromotedCell { owner, seq, .. } => self.promoted_syms.get(&(*owner, *seq)).cloned().context("missing promoted cell")?,
            SymbolSource::Host(host) => Symbol::Host(host.clone()),
            SymbolSource::Word(word) => Symbol::Word(*word),
            SymbolSource::ThreadSlot(index) => Symbol::Word(self.thread_slots.get(index).copied().unwrap_or(*index as u64)),
        })
    }
}

fn emit_dict_bodies(state: &AotState, exe: &mut Executable) -> Result<HashMap<usize, u32>> {
    let mut dict_bodies: HashMap<usize, u32> = HashMap::default();
    for item in state.functions.iter().flatten() {
        let AotFn::Dict { body, .. } = item else { continue };
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

fn emit_functions(state: &AotState, target: &Target, syms: &SymTable<'_>, dict_bodies: &HashMap<usize, u32>, exe: &mut Executable) -> Result<Vec<Option<u32>>> {
    let mut fn_offs = vec![None; state.functions.len()];
    let mut aliases = Vec::new();
    let mut pending = Vec::new();
    for (idx, item) in state.functions.iter().enumerate() {
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
        if let Some(aot) = &self.aot {
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
        let state = lock(self.aot.as_ref().context("image has no AOT recording")?);
        let thread_slots = lock(&self.thread_slots);
        let mut exe = Executable::new(target);

        let slot_offs: Vec<u32> = (0..state.functions.len()).map(|_| exe.rodata(&[0; 8], 8)).collect();
        let rodata_offs: Vec<u32> = state.rodata.iter().map(|(bytes, _)| exe.rodata(bytes, 16)).collect();
        for (id, (_, relocs)) in state.rodata.iter().enumerate() {
            for reloc in relocs {
                let RodataTarget::Rodata(target) = reloc.target;
                exe.pointer(true, rodata_offs[id] + reloc.offset, Symbol::Rodata(rodata_offs[target.0 as usize]));
            }
        }
        let vtable_offs: Vec<u32> = state.vtables.iter().map(|entries| exe.rodata(&vec![0u8; entries.len() * 8], 8)).collect();

        let mut static_syms = HashMap::default();
        for (&index, kind) in &state.statics {
            let sym = match kind {
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
            promoted_syms.insert((owner, seq), Symbol::Bss(exe.bss(size, 16)));
        }

        let syms = SymTable {
            slot_offs: &slot_offs,
            rodata_offs: &rodata_offs,
            vtable_offs: &vtable_offs,
            static_syms: &static_syms,
            promoted_syms: &promoted_syms,
            thread_slots: &thread_slots,
        };
        let dict_bodies = emit_dict_bodies(&state, &mut exe)?;
        let fn_offs = emit_functions(&state, target, &syms, &dict_bodies, &mut exe)?;

        for (id, entries) in state.vtables.iter().enumerate() {
            for (i, entry) in entries.iter().enumerate() {
                let target_sym = match *entry {
                    VtableEntry::Word(Word::Int(val)) => Symbol::Word(val),
                    VtableEntry::Word(Word::Code(fn_idx)) => slot_target(&fn_offs, &state.foreign_fns, fn_idx),
                    VtableEntry::Super(super_id) => Symbol::Rodata(vtable_offs[super_id.0 as usize]),
                };
                exe.pointer(true, vtable_offs[id] + (i * 8) as u32, target_sym);
            }
        }
        for (idx, &slot_off) in slot_offs.iter().enumerate() {
            exe.pointer(true, slot_off, slot_target(&fn_offs, &state.foreign_fns, idx));
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
