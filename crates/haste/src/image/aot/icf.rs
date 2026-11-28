use std::hash::{Hash as _, Hasher as _};
use std::sync::Arc;

use anyhow::{Context, Result};
use haste_codegen::{RelocKind, UnwindArg};
use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::aot::{Executable, Symbol};
use crate::host::HostSymbol;
use crate::lower::{SymbolSource, VtableEntry, Word};

use super::vdce::VtableMask;
use super::{AotFn, AotReach, AotState};

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
    Vtable,
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
enum VtableStaticSlot {
    Dead,
    Int(u64),
    FnSlot,
    Foreign(Symbol),
    SuperSlot,
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
    vtable_slots: Vec<usize>,
}

struct PreparedVtable {
    id: usize,
    fn_slots: Vec<usize>,
    super_slots: Vec<usize>,
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
    vtable_slots: &mut Vec<usize>,
) -> Result<RelocTargetKey> {
    Ok(match sym {
        SymbolSource::Slot(callee) => {
            slot_static_target(self_idx, (*callee, direct), state, roots, fn_slots)
        }
        SymbolSource::Vtable(id) => {
            vtable_slots.push(id.0 as usize);
            RelocTargetKey::Vtable
        }
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
    let mut vtable_slots = Vec::new();
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
                        &mut vtable_slots,
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
                    &mut vtable_slots,
                )?;
                target.hash(&mut h);
            }
        }
        Some(AotFn::Dict { body, relocs }) => {
            1u8.hash(&mut h);
            (Arc::as_ptr(body) as usize).hash(&mut h);
            relocs.len().hash(&mut h);
            for sym in relocs {
                let target = sym_static_target(
                    idx,
                    (sym, false),
                    state,
                    pre,
                    roots,
                    &mut fn_slots,
                    &mut vtable_slots,
                )?;
                target.hash(&mut h);
            }
        }
        Some(AotFn::Alias(_)) | None => return Ok(None),
    }
    Ok(Some(PreparedFn {
        idx,
        static_hash: h.finish(),
        fn_slots,
        vtable_slots,
    }))
}

fn build_static_key<'a>(
    idx: usize,
    state: &'a AotState,
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Option<StaticFnKey<'a>>> {
    let mut unused_fn = Vec::new();
    let mut unused_vt = Vec::new();
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
                        &mut unused_fn,
                        &mut unused_vt,
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
                    &mut unused_fn,
                    &mut unused_vt,
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
                    &mut unused_fn,
                    &mut unused_vt,
                )?);
            }
            Some(StaticFnKey::Dict(Arc::as_ptr(body) as usize, sig))
        }
        Some(AotFn::Alias(_)) | None => None,
    })
}

fn prepare_live_fns(
    live_roots: &[usize],
    state: &AotState,
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Vec<PreparedFn>> {
    let workers = crate::jobs().min(live_roots.len() / 64);
    if workers <= 1 {
        let mut out = Vec::with_capacity(live_roots.len());
        for &idx in live_roots {
            if let Some(p) = prepare_fn(idx, state, pre, roots)? {
                out.push(p);
            }
        }
        return Ok(out);
    }
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
    Ok(out)
}

fn prepare_vtable_groups(
    state: &AotState,
    order: &[usize],
    vmask: &VtableMask,
    roots: &[usize],
) -> (Vec<PreparedVtable>, Vec<Vec<usize>>, Vec<usize>) {
    let mut vt_class: Vec<usize> = (0..state.vtables.len()).collect();
    let mut prepared = Vec::with_capacity(order.len());
    let mut by_key: HashMap<Vec<VtableStaticSlot>, Vec<usize>> = HashMap::default();
    for &id in order {
        let Some(entries) = state.vtables.get(id) else {
            continue;
        };
        let mut key = Vec::with_capacity(entries.len());
        let mut fn_slots = Vec::new();
        let mut super_slots = Vec::new();
        for (slot, entry) in entries.iter().enumerate() {
            if !vmask.slot_live(id, slot) {
                key.push(VtableStaticSlot::Dead);
                continue;
            }
            match *entry {
                VtableEntry::Word(Word::Int(val)) => key.push(VtableStaticSlot::Int(val)),
                VtableEntry::Word(Word::Code(fn_idx)) => {
                    let root = roots.get(fn_idx).copied().unwrap_or(fn_idx);
                    if state.functions.get(root).and_then(Option::as_ref).is_some() {
                        fn_slots.push(root);
                        key.push(VtableStaticSlot::FnSlot);
                    } else {
                        key.push(VtableStaticSlot::Foreign(foreign_sym(
                            &state.foreign_fns,
                            fn_idx,
                        )));
                    }
                }
                VtableEntry::Super(super_id) => {
                    super_slots.push(super_id.0 as usize);
                    key.push(VtableStaticSlot::SuperSlot);
                }
            }
        }
        let pos = prepared.len();
        prepared.push(PreparedVtable {
            id,
            fn_slots,
            super_slots,
        });
        by_key.entry(key).or_default().push(pos);
    }
    let mut groups = Vec::new();
    for (_, mut members) in by_key {
        if members.len() < 2 {
            continue;
        }
        members.sort_unstable_by_key(|&pos| prepared[pos].id);
        let leader = prepared[members[0]].id;
        for &pos in &members {
            vt_class[prepared[pos].id] = leader;
        }
        if !prepared[members[0]].fn_slots.is_empty() || !prepared[members[0]].super_slots.is_empty()
        {
            groups.push(members);
        }
    }
    (prepared, groups, vt_class)
}

fn refine_fn_groups(
    groups: &mut Vec<Vec<usize>>,
    prepared: &[PreparedFn],
    prev_fn: &[usize],
    prev_vt: &[usize],
    fn_class: &mut [usize],
) -> bool {
    let mut split = false;
    let mut next_groups = Vec::with_capacity(groups.len());
    for group in groups.drain(..) {
        let mut buckets: HashMap<(Vec<usize>, Vec<usize>), Vec<usize>> =
            HashMap::with_capacity_and_hasher(group.len(), Default::default());
        for pos in group {
            let p = &prepared[pos];
            let f_sig: Vec<usize> = p
                .fn_slots
                .iter()
                .map(|&r| {
                    if r == usize::MAX {
                        usize::MAX
                    } else {
                        prev_fn[r]
                    }
                })
                .collect();
            let v_sig: Vec<usize> = p
                .vtable_slots
                .iter()
                .map(|&vt| prev_vt.get(vt).copied().unwrap_or(vt))
                .collect();
            buckets.entry((f_sig, v_sig)).or_default().push(pos);
        }
        if buckets.len() == 1 {
            next_groups.extend(buckets.into_values());
            continue;
        }
        split = true;
        for (_, mut sub) in buckets {
            sub.sort_unstable_by_key(|&pos| prepared[pos].idx);
            let leader = prepared[sub[0]].idx;
            for &pos in &sub {
                fn_class[prepared[pos].idx] = leader;
            }
            if sub.len() >= 2 {
                next_groups.push(sub);
            }
        }
    }
    *groups = next_groups;
    split
}

fn refine_vt_groups(
    groups: &mut Vec<Vec<usize>>,
    prepared: &[PreparedVtable],
    prev_fn: &[usize],
    prev_vt: &[usize],
    vt_class: &mut [usize],
) -> bool {
    let mut split = false;
    let mut next_groups = Vec::with_capacity(groups.len());
    for group in groups.drain(..) {
        let mut buckets: HashMap<(Vec<usize>, Vec<usize>), Vec<usize>> =
            HashMap::with_capacity_and_hasher(group.len(), Default::default());
        for pos in group {
            let p = &prepared[pos];
            let f_sig: Vec<usize> = p.fn_slots.iter().map(|&r| prev_fn[r]).collect();
            let s_sig: Vec<usize> = p
                .super_slots
                .iter()
                .map(|&vt| prev_vt.get(vt).copied().unwrap_or(vt))
                .collect();
            buckets.entry((f_sig, s_sig)).or_default().push(pos);
        }
        if buckets.len() == 1 {
            next_groups.extend(buckets.into_values());
            continue;
        }
        split = true;
        for (_, mut sub) in buckets {
            sub.sort_unstable_by_key(|&pos| prepared[pos].id);
            let leader = prepared[sub[0]].id;
            for &pos in &sub {
                vt_class[prepared[pos].id] = leader;
            }
            if sub.len() >= 2 {
                next_groups.push(sub);
            }
        }
    }
    *groups = next_groups;
    split
}

pub(super) fn fold_functions(
    state: &AotState,
    live: &AotReach,
    vmask: &VtableMask,
    pre: &PreSyms<'_>,
    roots: &[usize],
) -> Result<Vec<usize>> {
    let mut fn_class = roots.to_vec();
    let live_roots: Vec<usize> = (0..state.functions.len())
        .filter(|&idx| live.fns[idx] && roots[idx] == idx && state.functions[idx].is_some())
        .collect();
    if live_roots.len() < 2 {
        return Ok(fn_class);
    }
    let prepared = prepare_live_fns(&live_roots, state, pre, roots)?;
    let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::default();
    for (pos, p) in prepared.iter().enumerate() {
        by_hash.entry(p.static_hash).or_default().push(pos);
    }

    let mut fn_groups: Vec<Vec<usize>> = Vec::new();
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
            let leader = prepared[members[0]].idx;
            for &pos in &members {
                fn_class[prepared[pos].idx] = leader;
            }
            if !prepared[members[0]].fn_slots.is_empty()
                || !prepared[members[0]].vtable_slots.is_empty()
            {
                fn_groups.push(members);
            }
        }
    }
    if fn_groups.is_empty() {
        return Ok(fn_class);
    }

    let (vt_prepared, mut vt_groups, mut vt_class) =
        prepare_vtable_groups(state, &live.vtable_order, vmask, roots);

    loop {
        let prev_fn = fn_class.clone();
        let prev_vt = vt_class.clone();
        let fn_split =
            refine_fn_groups(&mut fn_groups, &prepared, &prev_fn, &prev_vt, &mut fn_class);
        let vt_split = refine_vt_groups(
            &mut vt_groups,
            &vt_prepared,
            &prev_fn,
            &prev_vt,
            &mut vt_class,
        );
        if !fn_split && !vt_split {
            break;
        }
    }
    Ok(fn_class)
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

type VtablePlacement = (Vec<Option<u32>>, Vec<(u32, usize)>);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::aot::{AotCode, AotFn};
    use haste_codegen::StencilReloc;

    #[test]
    fn icf_bisimulation_runs_to_fixpoint_past_16_hops() {
        const HOPS: usize = 18;
        let diff_a = HOPS * 2;
        let diff_b = HOPS * 2 + 1;
        let total = diff_b + 1;

        let mut state = AotState::new(total).into_inner().expect("lock");
        let call_reloc = StencilReloc {
            offset: 1,
            symbol: 0,
            kind: RelocKind::PcRel32,
        };
        let hop_code = AotCode::for_test(vec![0xE8, 0, 0, 0, 0, 0xC3], vec![call_reloc]);
        let leaf_a = AotCode::for_test(vec![0x90, 0xC3], Vec::new());
        let leaf_b = AotCode::for_test(vec![0xCC, 0xC3], Vec::new());

        for i in 0..HOPS {
            let next_a = if i + 1 < HOPS { i + 1 } else { diff_a };
            let next_b = if i + 1 < HOPS { HOPS + i + 1 } else { diff_b };
            state.functions[i] = Some(AotFn::Direct {
                code: Arc::clone(&hop_code),
                relocs: vec![SymbolSource::Slot(next_a)],
            });
            state.functions[HOPS + i] = Some(AotFn::Direct {
                code: Arc::clone(&hop_code),
                relocs: vec![SymbolSource::Slot(next_b)],
            });
        }
        state.functions[diff_a] = Some(AotFn::Direct {
            code: leaf_a,
            relocs: Vec::new(),
        });
        state.functions[diff_b] = Some(AotFn::Direct {
            code: leaf_b,
            relocs: Vec::new(),
        });

        let roots: Vec<usize> = (0..total).collect();
        let live = AotReach::new(total, &roots);
        let pre = PreSyms {
            rodata_offs: &[],
            static_syms: &HashMap::default(),
            promoted_syms: &HashMap::default(),
            thread_slots: &HashMap::default(),
        };
        let canon = fold_functions(&state, &live, &VtableMask::all_live(), &pre, &roots)
            .expect("fold_functions");
        for i in 0..HOPS {
            assert_ne!(
                canon[i],
                canon[HOPS + i],
                "hop {i} must not merge across 18-hop distinct leaves"
            );
        }
    }
}
