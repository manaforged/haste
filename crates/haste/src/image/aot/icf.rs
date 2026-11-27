use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

use anyhow::{Context, Result};
use haste_codegen::{RelocKind, UnwindArg};
use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::aot::{Executable, Symbol, UNWIND_FN_REC_BYTES};
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
    FnSlot(bool),
    ForeignSlot(bool, Symbol),
    Vtable(VtableId),
    Sym(Symbol),
}

#[derive(PartialEq, Eq)]
enum StaticFnKey<'a> {
    Direct(
        &'a [u8],
        Vec<(u32, RelocKind, RelocTargetKey)>,
        Vec<(i32, i32, RelocTargetKey, UnwindArg)>,
        Vec<(u32, u32, u32)>,
        u32,
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

struct PreparedFn {
    idx: usize,
    static_hash: u64,
    fn_slots: Vec<usize>,
}

fn slot_static_target(
    self_idx: usize,
    (callee, direct): (usize, bool),
    state: &AotState,
    roots: &[usize],
    fn_slots: &mut Vec<usize>,
) -> RelocTargetKey {
    let root = roots.get(callee).copied().unwrap_or(callee);
    if state.functions.get(root).and_then(Option::as_ref).is_some() {
        fn_slots.push(if root == self_idx { usize::MAX } else { root });
        RelocTargetKey::FnSlot(direct)
    } else {
        RelocTargetKey::ForeignSlot(direct, foreign_sym(&state.foreign_fns, callee))
    }
}

fn sym_static_target(
    self_idx: usize,
    (sym, direct): (&SymbolSource, bool),
    state: &AotState,
    pre: &PreSyms<'_>,
    roots: &[usize],
    fn_slots: &mut Vec<usize>,
) -> Result<RelocTargetKey> {
    Ok(match sym {
        SymbolSource::Slot(callee) => {
            slot_static_target(self_idx, (*callee, direct), state, roots, fn_slots)
        }
        SymbolSource::Vtable(id) => RelocTargetKey::Vtable(*id),
        other => RelocTargetKey::Sym(pre.translate(other)?),
    })
}

fn prepare_fn(
    idx: usize,
    state: &AotState,
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Option<PreparedFn>> {
    let mut h = FxHasher::default();
    let mut fn_slots = Vec::new();
    match &state.functions[idx] {
        Some(AotFn::Direct { code, relocs }) => {
            0u8.hash(&mut h);
            code.bytes().hash(&mut h);
            code.saved_regs().hash(&mut h);
            code.sites().len().hash(&mut h);
            for s in code.sites() {
                (s.ret, s.handler, s.frame_offset).hash(&mut h);
            }
            code.relocs().len().hash(&mut h);
            for reloc in code.relocs() {
                reloc.offset.hash(&mut h);
                reloc.kind.hash(&mut h);
                if let Some(rt) = reloc.as_runtime() {
                    0u8.hash(&mut h);
                    rt.hash(&mut h);
                } else {
                    1u8.hash(&mut h);
                    let direct = matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32);
                    let target = sym_static_target(
                        idx,
                        (&relocs[reloc.symbol as usize], direct),
                        state,
                        pre,
                        roots,
                        &mut fn_slots,
                    )?;
                    target.hash(&mut h);
                }
            }
            code.unwind_drops().len().hash(&mut h);
            for d in code.unwind_drops() {
                (d.local_off, d.flag_off, d.arg).hash(&mut h);
                let target = sym_static_target(
                    idx,
                    (&relocs[d.drop_sym as usize], true),
                    state,
                    pre,
                    roots,
                    &mut fn_slots,
                )?;
                target.hash(&mut h);
            }
        }
        Some(AotFn::Dict { body, relocs }) => {
            1u8.hash(&mut h);
            (Arc::as_ptr(body) as usize).hash(&mut h);
            relocs.len().hash(&mut h);
            for sym in relocs {
                let target =
                    sym_static_target(idx, (sym, false), state, pre, roots, &mut fn_slots)?;
                target.hash(&mut h);
            }
        }
        Some(AotFn::Alias(_)) | None => return Ok(None),
    }
    Ok(Some(PreparedFn {
        idx,
        static_hash: h.finish(),
        fn_slots,
    }))
}

fn build_static_key<'a>(
    idx: usize,
    state: &'a AotState,
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Option<StaticFnKey<'a>>> {
    let mut unused = Vec::new();
    Ok(match &state.functions[idx] {
        Some(AotFn::Direct { code, relocs }) => {
            let mut sig = Vec::with_capacity(code.relocs().len());
            for reloc in code.relocs() {
                let target = if let Some(rt) = reloc.as_runtime() {
                    RelocTargetKey::Sym(Symbol::Runtime(rt))
                } else {
                    let direct = matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32);
                    sym_static_target(
                        idx,
                        (&relocs[reloc.symbol as usize], direct),
                        state,
                        pre,
                        roots,
                        &mut unused,
                    )?
                };
                sig.push((reloc.offset, reloc.kind, target));
            }
            let mut drops = Vec::with_capacity(code.unwind_drops().len());
            for d in code.unwind_drops() {
                let target = sym_static_target(
                    idx,
                    (&relocs[d.drop_sym as usize], true),
                    state,
                    pre,
                    roots,
                    &mut unused,
                )?;
                drops.push((d.local_off, d.flag_off, target, d.arg));
            }
            let sites = code
                .sites()
                .iter()
                .map(|s| (s.ret, s.handler, s.frame_offset))
                .collect();
            Some(StaticFnKey::Direct(
                code.bytes(),
                sig,
                drops,
                sites,
                code.saved_regs(),
            ))
        }
        Some(AotFn::Dict { body, relocs }) => {
            let mut sig = Vec::with_capacity(relocs.len());
            for sym in relocs {
                sig.push(sym_static_target(
                    idx,
                    (sym, false),
                    state,
                    pre,
                    roots,
                    &mut unused,
                )?);
            }
            Some(StaticFnKey::Dict(Arc::as_ptr(body) as usize, sig))
        }
        Some(AotFn::Alias(_)) | None => None,
    })
}

pub(super) fn fold_functions(
    state: &AotState,
    live_fns: &[bool],
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Vec<usize>> {
    let mut canon = roots.to_vec();
    let live_roots: Vec<usize> = (0..state.functions.len())
        .filter(|&idx| live_fns[idx] && roots[idx] == idx && state.functions[idx].is_some())
        .collect();
    if live_roots.len() < 2 {
        return Ok(canon);
    }
    let workers = crate::jobs().min(live_roots.len() / 64);
    let prepared: Vec<PreparedFn> = if workers <= 1 {
        let mut out = Vec::with_capacity(live_roots.len());
        for &idx in &live_roots {
            if let Some(p) = prepare_fn(idx, state, pre, roots)? {
                out.push(p);
            }
        }
        out
    } else {
        let chunk_len = live_roots.len().div_ceil(workers).max(1);
        let chunks: Result<Vec<Vec<PreparedFn>>> = crate::sync::scope(|scope| {
            let handles: Vec<_> = live_roots
                .chunks(chunk_len)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::with_capacity(chunk.len());
                        for &idx in chunk {
                            if let Some(p) = prepare_fn(idx, state, pre, roots)? {
                                local.push(p);
                            }
                        }
                        Ok::<_, anyhow::Error>(local)
                    })
                })
                .collect();
            let mut out = Vec::with_capacity(handles.len());
            for h in handles {
                let res = h.join().map_err(|panic| {
                    anyhow::anyhow!(
                        "AOT ICF prepare worker panicked: {}",
                        crate::sync::panic_text(&*panic)
                    )
                })??;
                out.push(res);
            }
            Ok(out)
        });
        let mut out = Vec::with_capacity(live_roots.len());
        for part in chunks? {
            out.extend(part);
        }
        out
    };

    let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::default();
    for (pos, p) in prepared.iter().enumerate() {
        by_hash.entry(p.static_hash).or_default().push(pos);
    }

    let mut cand_groups: Vec<Vec<usize>> = Vec::new();
    for (_, bucket) in by_hash {
        if bucket.len() < 2 {
            continue;
        }
        let mut exact_groups: Vec<(StaticFnKey<'_>, Vec<usize>)> = Vec::new();
        for pos in bucket {
            let idx = prepared[pos].idx;
            let Some(key) = build_static_key(idx, state, pre, roots)? else {
                continue;
            };
            if let Some((_, members)) = exact_groups.iter_mut().find(|(k, _)| *k == key) {
                members.push(pos);
            } else {
                exact_groups.push((key, vec![pos]));
            }
        }
        for (_, mut members) in exact_groups {
            if members.len() < 2 {
                continue;
            }
            members.sort_unstable_by_key(|&pos| prepared[pos].idx);
            if prepared[members[0]].fn_slots.is_empty() {
                let leader = prepared[members[0]].idx;
                for &pos in &members {
                    canon[prepared[pos].idx] = leader;
                }
            } else {
                cand_groups.push(members);
            }
        }
    }
    if cand_groups.is_empty() {
        return Ok(canon);
    }

    for _ in 0..5 {
        let prev = canon.clone();
        let mut changed = false;
        for group in &cand_groups {
            let mut seen: HashMap<Vec<usize>, usize> =
                HashMap::with_capacity_and_hasher(group.len(), Default::default());
            for &pos in group {
                let p = &prepared[pos];
                let resolved: Vec<usize> = p
                    .fn_slots
                    .iter()
                    .map(|&r| if r == usize::MAX { usize::MAX } else { prev[r] })
                    .collect();
                let leader = *seen.entry(resolved).or_insert(p.idx);
                if canon[p.idx] != leader {
                    canon[p.idx] = leader;
                    changed = true;
                }
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
    let off = match seen.entry((entry.0.as_slice(), resolved)) {
        std::collections::hash_map::Entry::Occupied(hit) => *hit.get(),
        std::collections::hash_map::Entry::Vacant(slot) => {
            let fresh = exe.rodata(&entry.0, 16);
            emit_rodata_relocs(fresh, &slot.key().1, exe, fn_relocs);
            slot.insert(fresh);
            fresh
        }
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
    (fn_offs, dict_bodies): (&[Option<u32>], &HashMap<usize, u32>),
    exe: &mut Executable,
) {
    let mut seen: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
    let mut seen_dict: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
    let mut fns: Vec<(u32, u32, u32, Vec<(i32, i32, u32, u32)>)> = Vec::new();
    let mut sites: Vec<(u32, u32, u32, u32)> = Vec::new();
    let push_fn = |base_off: u32,
                   code: &AotCode,
                   drops: Vec<(i32, i32, u32, u32)>,
                   sites: &mut Vec<(u32, u32, u32, u32)>,
                   fns: &mut Vec<(u32, u32, u32, Vec<(i32, i32, u32, u32)>)>| {
        for s in code.sites() {
            sites.push((base_off + s.ret, base_off + s.handler, s.frame_offset, 0));
        }
        if !drops.is_empty() || code.saved_regs() != 0 {
            fns.push((
                base_off,
                base_off + code.bytes().len() as u32,
                code.saved_regs(),
                drops,
            ));
        }
    };
    for &idx in &live.order {
        let leader = canon[roots[idx]];
        if !seen.insert(leader) {
            continue;
        }
        match &state.functions[leader] {
            Some(AotFn::Direct { code, relocs }) => {
                let Some(fn_start) = fn_offs.get(leader).copied().flatten() else {
                    continue;
                };
                let mut drops = Vec::with_capacity(code.unwind_drops().len());
                for d in code.unwind_drops() {
                    if let Some(&SymbolSource::Slot(callee_idx)) = relocs.get(d.drop_sym as usize)
                        && let Some(drop_off) = fn_offs.get(callee_idx).copied().flatten()
                    {
                        drops.push((d.local_off, d.flag_off, drop_off, u32::from(d.arg.as_u8())));
                    }
                }
                push_fn(fn_start, code, drops, &mut sites, &mut fns);
            }
            Some(AotFn::Dict { body, .. }) => {
                let ptr = Arc::as_ptr(body) as usize;
                if seen_dict.insert(ptr)
                    && let Some(&body_off) = dict_bodies.get(&ptr)
                {
                    push_fn(body_off, body, Vec::new(), &mut sites, &mut fns);
                }
            }
            _ => {}
        }
    }
    if fns.is_empty() && sites.is_empty() {
        return;
    }
    fns.sort_unstable_by_key(|&(start, _, _, _)| start);
    sites.sort_unstable_by_key(|&(ret, _, _, _)| ret);
    let total_drops: usize = fns.iter().map(|(_, _, _, d)| d.len()).sum();
    let sites_off = 24 + fns.len() * UNWIND_FN_REC_BYTES + total_drops * 16;
    let mut buf = Vec::with_capacity(sites_off + sites.len() * 16);
    buf.extend_from_slice(&(fns.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(sites.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(sites_off as u64).to_le_bytes());
    let mut cursor = 0u32;
    for &(start, end, saved_regs, ref drops) in &fns {
        for word in [start, end, cursor, drops.len() as u32, saved_regs] {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        cursor += drops.len() as u32;
    }
    for (_, _, _, drops) in fns {
        for (local_off, flag_off, drop_off, ind) in drops {
            for word in [local_off as u32, flag_off as u32, drop_off, ind] {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    for (ret, handler, frame_offset, pad) in sites {
        for word in [ret, handler, frame_offset, pad] {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }
    exe.unwind_table_off = Some(exe.rodata(&buf, 8));
}
