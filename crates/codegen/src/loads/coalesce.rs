use super::{State, emit_op, fits_mem_imm};
use crate::ir::{Function, Inst, Op, StackSlot, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance};
use crate::merge::find;

#[derive(Clone, Copy)]
pub(super) enum StoreBase {
    Slot(StackSlot),
    Addr(Value),
}

#[derive(Clone, Copy)]
pub(super) struct ConstStore {
    pub loc: MemLoc,
    pub imm: u64,
    pub kept_idx: usize,
    pub base: StoreBase,
}

pub(super) struct ScanBuf {
    pub kept: Vec<Option<Inst>>,
    pub pending_stores: Vec<(MemLoc, usize)>,
    pub const_stores: Vec<ConstStore>,
}

pub(super) fn slot_unescaped(prov: &Provenance, loc: MemLoc) -> bool {
    matches!(loc.root, MemRoot::Slot(s) if prov.slot_unescaped(s))
}

fn store_base(prov: &Provenance, subst: &[Value], op: &Op, loc: MemLoc) -> Option<StoreBase> {
    match (op, loc.base) {
        (_, MemBase::Slot(s)) => Some(StoreBase::Slot(s)),
        (Op::Store(..), MemBase::Addr(v)) => Some(StoreBase::Addr(v)),
        (&Op::Store(_, addr, _), MemBase::Global(_)) => {
            let a = find(subst, addr);
            let (eff_val, _) = prov.affine.get(a.index()).copied().unwrap_or((a, 0));
            prov.symbol_defs.get(eff_val.index()).copied().flatten().map(|_| StoreBase::Addr(eff_val))
        }
        _ => None,
    }
}

fn masked_imm(ty: Type, imm: i64) -> Option<u64> {
    match ty {
        Type::I8 => Some((imm as u64) & 0xFF),
        Type::I16 => Some((imm as u64) & 0xFFFF),
        Type::I32 => Some((imm as u64) & 0xFFFF_FFFF),
        _ => None,
    }
}

fn canonical_wide_imm(wide_ty: Type, wide_imm: u64) -> i64 {
    match wide_ty {
        Type::I16 => i64::from(wide_imm as i16),
        Type::I32 => i64::from(wide_imm as i32),
        _ => wide_imm as i64,
    }
}

fn find_coalesce_slice(const_stores: &[ConstStore], target_idx: usize) -> Option<(Vec<usize>, i64, Type)> {
    let target = const_stores[target_idx];
    let mut same: Vec<(usize, i64, u32)> = const_stores
        .iter()
        .enumerate()
        .filter(|(_, cs)| cs.loc.base == target.loc.base)
        .map(|(i, cs)| (i, cs.loc.offset, cs.loc.size))
        .collect();
    if same.len() < 2 {
        return None;
    }
    same.sort_by_key(|&(_, off, _)| off);
    let pos = same.iter().position(|&(i, _, _)| i == target_idx)?;
    let (mut l, mut r) = (pos, pos);
    while l > 0 && same[l - 1].1 + i64::from(same[l - 1].2) == same[l].1 {
        l -= 1;
    }
    while r + 1 < same.len() && same[r].1 + i64::from(same[r].2) == same[r + 1].1 {
        r += 1;
    }
    let contig = &same[l..=r];
    let t_rel = pos - l;
    for target_w in [8_i64, 4, 2] {
        let Some(wide_ty) = Type::int_of_bytes(target_w) else { continue };
        for a in 0..=t_rel {
            let lo = contig[a].1;
            if i32::try_from(lo).is_err() {
                continue;
            }
            if matches!(target.base, StoreBase::Addr(_)) && !fits_mem_imm(wide_ty, lo) {
                continue;
            }
            for b in t_rel..contig.len() {
                if b > a && contig[b].1 + i64::from(contig[b].2) - lo == target_w {
                    return Some((contig[a..=b].iter().map(|&(idx, _, _)| idx).collect(), lo, wide_ty));
                }
            }
        }
    }
    None
}

fn collect_coalesce_bytes(func: &mut Function, bufs: &mut ScanBuf, indices: &[usize], lo: i64, fallback: StoreBase) -> (u64, StoreBase) {
    let mut wide_imm = 0_u64;
    let mut chosen = fallback;
    let mut min_kept = usize::MAX;
    let mut drop_kept = Vec::with_capacity(indices.len());
    for &idx in indices {
        let cs = bufs.const_stores[idx];
        wide_imm |= cs.imm << ((cs.loc.offset - lo) * 8);
        if cs.kept_idx < min_kept {
            min_kept = cs.kept_idx;
            chosen = cs.base;
        }
        if cs.kept_idx != bufs.kept.len() {
            drop_kept.push(cs.kept_idx);
            if let Some(dead) = bufs.kept[cs.kept_idx].take() {
                func.insts[dead.index()].op = Op::Trap;
            }
        }
    }
    bufs.pending_stores.retain(|&(_, kidx)| !drop_kept.contains(&kidx));
    let mut rm = vec![false; bufs.const_stores.len()];
    for &idx in indices {
        rm[idx] = true;
    }
    let mut cur_idx = 0_usize;
    bufs.const_stores.retain(|_| {
        let keep = !rm[cur_idx];
        cur_idx += 1;
        keep
    });
    (wide_imm, chosen)
}

fn emit_wide_const_store(
    func: &mut Function,
    state: &mut State<'_>,
    known: &mut KnownLoads,
    bases: &[(MemBase, MemRoot)],
    bufs: &mut ScanBuf,
    (inst, loc, chosen, wide_ty, wide_imm): (Inst, MemLoc, StoreBase, Type, u64),
) -> MemLoc {
    let canonical = canonical_wide_imm(wide_ty, wide_imm);
    let c_val = emit_op(func, state, &mut bufs.kept, wide_ty, Op::Const(canonical));
    let lo_i32 = loc.offset as i32;
    func.insts[inst.index()].op = match chosen {
        StoreBase::Slot(s) => Op::StackStore(c_val, s, lo_i32),
        StoreBase::Addr(a) => Op::Store(c_val, a, lo_i32),
    };
    super::dom::write_preserving(func, known, bases, loc, wide_ty, c_val, state.prov);
    bufs.const_stores.push(ConstStore { loc, imm: wide_imm, kept_idx: bufs.kept.len(), base: chosen });
    loc
}

fn retain_const_stores(prov: &Provenance, const_stores: &mut Vec<ConstStore>, loc: MemLoc) {
    let safe_write = slot_unescaped(prov, loc);
    const_stores.retain(|cs| {
        !crate::mem::may_alias(loc, cs.loc, prov)
            && (slot_unescaped(prov, cs.loc) || safe_write || cs.loc.base == loc.base)
    });
}

fn try_patch_const_store(
    func: &mut Function,
    state: &mut State<'_>,
    known: &mut KnownLoads,
    bases: &[(MemBase, MemRoot)],
    bufs: &mut ScanBuf,
    (inst, loc, ty, value): (Inst, MemLoc, Type, Value),
) -> Option<MemLoc> {
    let imm = state.consts[value.index()].and_then(|v| masked_imm(ty, v))?;
    let loc_end = loc.offset.checked_add(i64::from(loc.size))?;
    let hit_idx = bufs.const_stores.iter().position(|cs| {
        cs.loc.base == loc.base
            && cs.loc.size > loc.size
            && cs.loc.offset <= loc.offset
            && cs.loc.offset.checked_add(i64::from(cs.loc.size)).is_some_and(|e| loc_end <= e)
            && cs.kept_idx < bufs.kept.len()
            && bufs.kept[cs.kept_idx].is_some()
    })?;
    let cs = bufs.const_stores.swap_remove(hit_idx);
    let wide_ty = Type::int_of_bytes(i64::from(cs.loc.size))?;
    if let Some(dead) = bufs.kept[cs.kept_idx].take() {
        func.insts[dead.index()].op = Op::Trap;
    }
    bufs.pending_stores.retain(|&(_, kidx)| kidx != cs.kept_idx);
    retain_const_stores(state.prov, &mut bufs.const_stores, cs.loc);
    let shift = (loc.offset - cs.loc.offset) * 8;
    let width_mask = (1_u64 << (loc.size * 8)) - 1;
    let wide_imm = (cs.imm & !(width_mask << shift)) | ((imm & width_mask) << shift);
    Some(emit_wide_const_store(func, state, known, bases, bufs, (inst, cs.loc, cs.base, wide_ty, wide_imm)))
}

pub(super) fn try_coalesce_store(
    func: &mut Function,
    state: &mut State<'_>,
    known: &mut KnownLoads,
    bases: &[(MemBase, MemRoot)],
    bufs: &mut ScanBuf,
    (inst, loc, ty, value, track): (Inst, MemLoc, Type, Value, bool),
) -> Option<MemLoc> {
    if track && let Some(patched) = try_patch_const_store(func, state, known, bases, bufs, (inst, loc, ty, value)) {
        return Some(patched);
    }
    retain_const_stores(state.prov, &mut bufs.const_stores, loc);
    if !track {
        return None;
    }
    let imm = state.consts[value.index()].and_then(|v| masked_imm(ty, v))?;
    let base = store_base(state.prov, state.subst, &func.insts[inst.index()].op, loc)?;
    let new_idx = bufs.const_stores.len();
    bufs.const_stores.push(ConstStore { loc, imm, kept_idx: bufs.kept.len(), base });
    let (indices, lo, wide_ty) = find_coalesce_slice(&bufs.const_stores, new_idx)?;
    let (wide_imm, chosen) = collect_coalesce_bytes(func, bufs, &indices, lo, base);
    let wide_loc = MemLoc::new(loc.root, loc.base, lo, wide_ty.bytes());
    Some(emit_wide_const_store(func, state, known, bases, bufs, (inst, wide_loc, chosen, wide_ty, wide_imm)))
}
