use std::sync::Arc;

use anyhow::{Context, Result};
use haste_codegen::RelocKind;
use rustc_hash::FxHashMap as HashMap;

use crate::aot::{Executable, Symbol};
use crate::host::HostSymbol;
use crate::lower::{RodataTarget, SymbolSource, VtableEntry, VtableId, Word};

use super::vdce::VtableMask;
use super::{AotCode, AotFn, AotReach, AotState};

pub(super) struct PreSyms<'a> {
    pub(super) rodata_offs: &'a [Option<u32>],
    pub(super) static_syms: &'a HashMap<usize, Symbol>,
    pub(super) promoted_syms: &'a HashMap<(usize, u32), Symbol>,
    pub(super) thread_slots: &'a HashMap<usize, u64>,
}

impl PreSyms<'_> {
    fn translate(&self, sym: &SymbolSource) -> Result<Symbol> {
        Ok(match sym {
            SymbolSource::Rodata(id) => Symbol::Rodata(
                self.rodata_offs
                    .get(id.0 as usize)
                    .copied()
                    .flatten()
                    .context("missing rodata offset")?,
            ),
            SymbolSource::StaticCell { index, .. } => self
                .static_syms
                .get(index)
                .cloned()
                .context("missing static cell")?,
            SymbolSource::PromotedCell { owner, seq, .. } => self
                .promoted_syms
                .get(&(*owner, *seq))
                .cloned()
                .context("missing promoted cell")?,
            SymbolSource::Host(host) => Symbol::Host(host.clone()),
            SymbolSource::Word(word) => Symbol::Word(*word),
            SymbolSource::ThreadSlot(index) => Symbol::Word(
                self.thread_slots
                    .get(index)
                    .copied()
                    .unwrap_or(*index as u64),
            ),
            SymbolSource::Slot(_) | SymbolSource::Vtable(_) => {
                unreachable!("slots and vtables use structural ICF keys")
            }
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum RelocTargetKey {
    SelfSlot(bool),
    Fn(bool, usize),
    ForeignSlot(bool, Symbol),
    Vtable(VtableId),
    Sym(Symbol),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum FnIcfKey<'a> {
    Direct(
        &'a [u8],
        Vec<(u32, RelocKind, RelocTargetKey)>,
        Vec<(i32, i32, RelocTargetKey, bool)>,
    ),
    Dict(usize, Vec<RelocTargetKey>),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum SlotKey {
    Fn(usize),
    Sym(Symbol),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum VtableSlotKey {
    Dead,
    Int(u64),
    Fn(usize),
    Sym(Symbol),
    Super(u32),
}

pub(super) fn foreign_sym(foreign_fns: &HashMap<usize, HostSymbol>, idx: usize) -> Symbol {
    foreign_fns
        .get(&idx)
        .map_or(Symbol::Host(HostSymbol::Abort), |host| {
            Symbol::Host(host.clone())
        })
}

pub(super) fn alias_roots(functions: &[Option<AotFn>]) -> Vec<usize> {
    (0..functions.len())
        .map(|mut cur| {
            for _ in 0..functions.len() {
                match functions.get(cur).and_then(Option::as_ref) {
                    Some(AotFn::Alias(next)) if *next < functions.len() => cur = *next,
                    _ => break,
                }
            }
            cur
        })
        .collect()
}

fn slot_reloc_key(
    self_idx: usize,
    (callee, direct): (usize, bool),
    state: &AotState,
    roots: &[usize],
    canon: &[usize],
) -> RelocTargetKey {
    let root = roots.get(callee).copied().unwrap_or(callee);
    if state.functions.get(root).and_then(Option::as_ref).is_some() {
        if root == self_idx {
            RelocTargetKey::SelfSlot(direct)
        } else {
            RelocTargetKey::Fn(direct, canon[root])
        }
    } else {
        RelocTargetKey::ForeignSlot(direct, foreign_sym(&state.foreign_fns, callee))
    }
}

fn sym_reloc_key(
    self_idx: usize,
    (sym, direct): (&SymbolSource, bool),
    state: &AotState,
    pre: &PreSyms<'_>,
    (roots, canon): (&[usize], &[usize]),
) -> Result<RelocTargetKey> {
    Ok(match sym {
        SymbolSource::Slot(callee) => {
            slot_reloc_key(self_idx, (*callee, direct), state, roots, canon)
        }
        SymbolSource::Vtable(id) => RelocTargetKey::Vtable(*id),
        other => RelocTargetKey::Sym(pre.translate(other)?),
    })
}

fn direct_icf_key<'a>(
    self_idx: usize,
    (code, relocs): (&'a AotCode, &[SymbolSource]),
    state: &AotState,
    pre: &PreSyms<'_>,
    (roots, canon): (&[usize], &[usize]),
) -> Result<FnIcfKey<'a>> {
    let mut sig = Vec::with_capacity(code.relocs().len());
    for reloc in code.relocs() {
        let target = if let Some(rt) = reloc.as_runtime() {
            RelocTargetKey::Sym(Symbol::Runtime(rt))
        } else {
            let direct = matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32);
            sym_reloc_key(
                self_idx,
                (&relocs[reloc.symbol as usize], direct),
                state,
                pre,
                (roots, canon),
            )?
        };
        sig.push((reloc.offset, reloc.kind, target));
    }
    let mut drops = Vec::with_capacity(code.unwind_drops().len());
    for d in code.unwind_drops() {
        let target = sym_reloc_key(
            self_idx,
            (&relocs[d.drop_sym as usize], true),
            state,
            pre,
            (roots, canon),
        )?;
        drops.push((d.local_off, d.flag_off, target, d.indirect));
    }
    Ok(FnIcfKey::Direct(code.bytes(), sig, drops))
}

fn dict_icf_key<'a>(
    self_idx: usize,
    (body, relocs): (&Arc<AotCode>, &[SymbolSource]),
    state: &AotState,
    pre: &PreSyms<'_>,
    (roots, canon): (&[usize], &[usize]),
) -> Result<FnIcfKey<'a>> {
    let mut sig = Vec::with_capacity(relocs.len());
    for sym in relocs {
        sig.push(sym_reloc_key(
            self_idx,
            (sym, false),
            state,
            pre,
            (roots, canon),
        )?);
    }
    Ok(FnIcfKey::Dict(Arc::as_ptr(body) as usize, sig))
}

pub(super) fn fold_functions(
    state: &AotState,
    live_fns: &[bool],
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Vec<usize>> {
    let mut canon = roots.to_vec();
    for _ in 0..5 {
        let mut seen: HashMap<FnIcfKey<'_>, usize> = HashMap::default();
        let mut changed = false;
        let prev = canon.clone();
        for idx in 0..state.functions.len() {
            if !live_fns[idx] || roots[idx] != idx {
                continue;
            }
            let key = match &state.functions[idx] {
                Some(AotFn::Direct { code, relocs }) => {
                    direct_icf_key(idx, (code, relocs), state, pre, (roots, &prev))?
                }
                Some(AotFn::Dict { body, relocs }) => {
                    dict_icf_key(idx, (body, relocs), state, pre, (roots, &prev))?
                }
                Some(AotFn::Alias(_)) | None => continue,
            };
            let leader = *seen.entry(key).or_insert(idx);
            if canon[idx] != leader {
                canon[idx] = leader;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(canon)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RodataRef {
    Rodata(u32),
    Fn(usize),
}

type RodataKey<'a> = (&'a [u8], Vec<(u32, RodataRef)>);
type VtablePlacement = (Vec<Option<u32>>, Vec<(u32, usize)>);

fn emit_rodata_relocs(
    base: u32,
    resolved: &[(u32, RodataRef)],
    exe: &mut Executable,
    fn_relocs: &mut Vec<(u32, usize)>,
) {
    for &(reloc_off, target) in resolved {
        match target {
            RodataRef::Rodata(child_off) => {
                exe.pointer(true, base + reloc_off, Symbol::Rodata(child_off))
            }
            RodataRef::Fn(fn_idx) => fn_relocs.push((base + reloc_off, fn_idx)),
        }
    }
}

fn place_rodata_node<'a>(
    id: usize,
    state: &'a AotState,
    rodata_offs: &mut [Option<u32>],
    visiting: &mut [bool],
    seen: &mut HashMap<RodataKey<'a>, u32>,
    fn_relocs: &mut Vec<(u32, usize)>,
    exe: &mut Executable,
) -> Result<u32> {
    if let Some(off) = rodata_offs[id] {
        return Ok(off);
    }
    let entry = state.rodata.get(id).context("missing rodata entry")?;
    if visiting[id] {
        let off = exe.rodata(&entry.0, 16);
        rodata_offs[id] = Some(off);
        return Ok(off);
    }
    visiting[id] = true;
    let mut resolved = Vec::with_capacity(entry.1.len());
    for reloc in &entry.1 {
        let target = match reloc.target {
            RodataTarget::Rodata(child) => RodataRef::Rodata(place_rodata_node(
                child.0 as usize,
                state,
                rodata_offs,
                visiting,
                seen,
                fn_relocs,
                exe,
            )?),
            RodataTarget::Fn(fn_idx) => RodataRef::Fn(fn_idx),
        };
        resolved.push((reloc.offset, target));
    }
    visiting[id] = false;
    if let Some(off) = rodata_offs[id] {
        emit_rodata_relocs(off, &resolved, exe, fn_relocs);
        return Ok(off);
    }
    let off = if let Some(&existing) = seen.get(&(entry.0.as_slice(), resolved.clone())) {
        existing
    } else {
        let fresh = exe.rodata(&entry.0, 16);
        emit_rodata_relocs(fresh, &resolved, exe, fn_relocs);
        seen.insert((entry.0.as_slice(), resolved), fresh);
        fresh
    };
    rodata_offs[id] = Some(off);
    Ok(off)
}

pub(super) fn emit_rodata(
    state: &AotState,
    live: &AotReach,
    exe: &mut Executable,
) -> Result<(Vec<Option<u32>>, Vec<(u32, usize)>)> {
    let mut rodata_offs = vec![None; state.rodata.len()];
    let mut visiting = vec![false; state.rodata.len()];
    let mut seen: HashMap<RodataKey<'_>, u32> = HashMap::default();
    let mut fn_relocs = Vec::new();
    for &id in &live.rodata_order {
        place_rodata_node(
            id,
            state,
            &mut rodata_offs,
            &mut visiting,
            &mut seen,
            &mut fn_relocs,
            exe,
        )?;
    }
    Ok((rodata_offs, fn_relocs))
}

pub(super) fn alloc_slots(
    state: &AotState,
    slots: &[usize],
    (roots, canon): (&[usize], &[usize]),
    exe: &mut Executable,
) -> (HashMap<usize, u32>, Vec<(u32, usize)>) {
    let mut by_key: HashMap<SlotKey, u32> = HashMap::default();
    let mut slot_offs = HashMap::default();
    let mut unique = Vec::new();
    for &idx in slots {
        let root = roots.get(idx).copied().unwrap_or(idx);
        let (key, rep) = if state.functions.get(root).and_then(Option::as_ref).is_some() {
            let leader = canon[root];
            (SlotKey::Fn(leader), leader)
        } else {
            (SlotKey::Sym(foreign_sym(&state.foreign_fns, idx)), idx)
        };
        let off = *by_key.entry(key).or_insert_with(|| {
            let fresh = exe.rodata(&[0; 8], 8);
            unique.push((fresh, rep));
            fresh
        });
        slot_offs.insert(idx, off);
    }
    (slot_offs, unique)
}

struct VtableAlloc<'a> {
    vtable_offs: Vec<Option<u32>>,
    seen: HashMap<Vec<VtableSlotKey>, u32>,
    unique: Vec<(u32, usize)>,
    roots: &'a [usize],
    canon: &'a [usize],
}

fn place_vtable_node(
    id: usize,
    state: &AotState,
    vmask: &VtableMask,
    alloc: &mut VtableAlloc<'_>,
    exe: &mut Executable,
) -> Result<u32> {
    if let Some(off) = alloc.vtable_offs[id] {
        return Ok(off);
    }
    let entries = state.vtables.get(id).context("missing vtable")?;
    let mut key = Vec::with_capacity(entries.len());
    for (slot, entry) in entries.iter().enumerate() {
        if !vmask.slot_live(id, slot) {
            key.push(VtableSlotKey::Dead);
            continue;
        }
        let slot_key = match *entry {
            VtableEntry::Word(Word::Int(val)) => VtableSlotKey::Int(val),
            VtableEntry::Word(Word::Code(fn_idx)) => {
                let root = alloc.roots.get(fn_idx).copied().unwrap_or(fn_idx);
                if state.functions.get(root).and_then(Option::as_ref).is_some() {
                    VtableSlotKey::Fn(alloc.canon[root])
                } else {
                    VtableSlotKey::Sym(foreign_sym(&state.foreign_fns, fn_idx))
                }
            }
            VtableEntry::Super(super_id) => VtableSlotKey::Super(place_vtable_node(
                super_id.0 as usize,
                state,
                vmask,
                alloc,
                exe,
            )?),
        };
        key.push(slot_key);
    }
    let off = if let Some(&existing) = alloc.seen.get(&key) {
        existing
    } else {
        let fresh = exe.rodata(&vec![0u8; entries.len() * 8], 8);
        alloc.seen.insert(key, fresh);
        alloc.unique.push((fresh, id));
        fresh
    };
    alloc.vtable_offs[id] = Some(off);
    Ok(off)
}

pub(super) fn alloc_vtables(
    state: &AotState,
    order: &[usize],
    vmask: &VtableMask,
    (roots, canon): (&[usize], &[usize]),
    exe: &mut Executable,
) -> Result<VtablePlacement> {
    let mut alloc = VtableAlloc {
        vtable_offs: vec![None; state.vtables.len()],
        seen: HashMap::default(),
        unique: Vec::new(),
        roots,
        canon,
    };
    for &id in order {
        place_vtable_node(id, state, vmask, &mut alloc, exe)?;
    }
    Ok((alloc.vtable_offs, alloc.unique))
}

pub(super) fn emit_unwind_table(
    state: &AotState,
    live: &AotReach,
    (roots, canon): (&[usize], &[usize]),
    fn_offs: &[Option<u32>],
    exe: &mut Executable,
) {
    let mut seen: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
    let mut fns: Vec<(u32, u32, Vec<(i32, i32, u32, u32)>)> = Vec::new();
    for &idx in &live.order {
        let leader = canon[roots[idx]];
        if !seen.insert(leader) {
            continue;
        }
        let Some(fn_start) = fn_offs.get(leader).copied().flatten() else {
            continue;
        };
        let Some(AotFn::Direct { code, relocs }) = &state.functions[leader] else {
            continue;
        };
        if code.unwind_drops().is_empty() {
            continue;
        }
        let mut drops = Vec::with_capacity(code.unwind_drops().len());
        for d in code.unwind_drops() {
            if let Some(&SymbolSource::Slot(callee_idx)) = relocs.get(d.drop_sym as usize)
                && let Some(drop_off) = fn_offs.get(callee_idx).copied().flatten()
            {
                drops.push((d.local_off, d.flag_off, drop_off, u32::from(d.indirect)));
            }
        }
        if !drops.is_empty() {
            fns.push((fn_start, fn_start + code.bytes().len() as u32, drops));
        }
    }
    if fns.is_empty() {
        return;
    }
    fns.sort_unstable_by_key(|&(start, _, _)| start);
    let total_drops: usize = fns.iter().map(|(_, _, d)| d.len()).sum();
    let mut buf = Vec::with_capacity(8 + (fns.len() + total_drops) * 16);
    buf.extend_from_slice(&(fns.len() as u64).to_le_bytes());
    let mut cursor = 0u32;
    for &(start, end, ref drops) in &fns {
        for word in [start, end, cursor, drops.len() as u32] {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        cursor += drops.len() as u32;
    }
    for (_, _, drops) in fns {
        for (local_off, flag_off, drop_off, ind) in drops {
            for word in [local_off as u32, flag_off as u32, drop_off, ind] {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    exe.unwind_table_off = Some(exe.rodata(&buf, 8));
}
