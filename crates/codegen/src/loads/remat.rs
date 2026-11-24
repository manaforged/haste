mod split;

pub(super) use split::split_long_loads;

use crate::ir::{Convert, Function, Inst, Op, Type, Value};
use crate::mem::{KnownLoads, MemLoc, MemRoot, Provenance, may_alias_at};
use crate::merge::find;

use super::coalesce::ScanBuf;
use super::dom::{clobbers_loc, loc_immutable};
use super::join::{SubwordPlan, find_subword_source, int_of};

const MIN_BLOCK_INSTS: usize = 128;
const FAR_MIN_DIST: usize = 128;
const MIN_GAP_DEFS: usize = 24;

#[derive(Clone, Copy)]
struct LoadTrack {
    last_use_pos: usize,
    last_use_defs: usize,
}

pub(super) struct FarCseGuard {
    active: bool,
    cur_pos: usize,
    cur_defs: usize,
    loads: Vec<Option<LoadTrack>>,
    touched: Vec<usize>,
}

impl FarCseGuard {
    pub(super) fn empty() -> Self {
        Self {
            active: false,
            cur_pos: 0,
            cur_defs: 0,
            loads: Vec::new(),
            touched: Vec::new(),
        }
    }

    pub(super) fn reset(&mut self, block_len: usize, num_values: usize) {
        for idx in self.touched.drain(..) {
            if idx < self.loads.len() {
                self.loads[idx] = None;
            }
        }
        (self.cur_pos, self.cur_defs) = (0, 0);
        self.active = block_len >= MIN_BLOCK_INSTS;
        if self.active && self.loads.len() < num_values {
            self.loads.resize(num_values, None);
        }
    }

    pub(super) fn should_skip_value(&self, op: &Op, ty: Type, dom_val: Value) -> bool {
        if !self.active || !is_scalar_remat_ty(ty) || !matches!(op, Op::Load(..)) {
            return false;
        }
        let Some(Some(track)) = self.loads.get(dom_val.index()) else {
            return false;
        };
        self.cur_pos >= track.last_use_pos + FAR_MIN_DIST
            && self.cur_defs.saturating_sub(track.last_use_defs) >= MIN_GAP_DEFS
    }

    pub(super) fn check_cse(
        &self,
        op: &Op,
        ty: Type,
        forwarded: Option<Value>,
    ) -> (Option<Value>, bool) {
        let Some(dom_val) = forwarded else {
            return (None, false);
        };
        if self.should_skip_value(op, ty, dom_val) {
            return (None, true);
        }
        (Some(dom_val), false)
    }

    pub(super) fn sync_kept(&mut self, func: &Function, subst: &[Value], insts: &[Option<Inst>]) {
        if !self.active {
            return;
        }
        for inst in insts.iter().flatten() {
            self.record_inst(func, subst, *inst);
        }
    }

    fn record_inst(&mut self, func: &Function, subst: &[Value], inst: Inst) {
        let data = &func.insts[inst.index()];
        let next_defs = self.cur_defs + usize::from(data.result.is_some());
        if self.loads.len() < func.types.len() {
            self.loads.resize(func.types.len(), None);
        }
        data.op.for_operands(|raw| {
            let v = find(subst, raw);
            if let Some(Some(track)) = self.loads.get_mut(v.index()) {
                track.last_use_pos = self.cur_pos;
                track.last_use_defs = next_defs;
            }
        });
        if let Some(res) = data.result
            && matches!(data.op, Op::Load(..))
            && is_scalar_remat_ty(func.ty(res))
        {
            if self.loads[res.index()].is_none() {
                self.touched.push(res.index());
            }
            self.loads[res.index()] = Some(LoadTrack {
                last_use_pos: self.cur_pos,
                last_use_defs: next_defs,
            });
        }
        self.cur_pos += 1;
        self.cur_defs = next_defs;
    }

    pub(super) fn on_drop_store(&mut self) {
        if self.active {
            self.cur_pos = self.cur_pos.saturating_sub(1);
        }
    }
}

fn is_scalar_remat_ty(ty: Type) -> bool {
    ty.is_small_int() || ty.is_float()
}

fn kill_kept_store(
    func: &mut Function,
    kept: &mut [Option<Inst>],
    guard: &mut FarCseGuard,
    kept_idx: usize,
) -> bool {
    let Some(dead_store) = kept[kept_idx].take() else {
        return false;
    };
    func.insts[dead_store.index()].op = Op::Trap;
    guard.on_drop_store();
    true
}

pub(super) fn drop_stores_if(
    func: &mut Function,
    bufs: &mut ScanBuf,
    guard: &mut FarCseGuard,
    mut kill: impl FnMut(MemLoc) -> bool,
) -> bool {
    let mut changed = false;
    bufs.pending_stores.retain(|&(ploc, kept_idx)| {
        if !kill(ploc) {
            return true;
        }
        changed |= kill_kept_store(func, &mut bufs.kept, guard, kept_idx);
        false
    });
    changed
}

/// Drops the pending stores into the allocation that `inst` frees with the runtime `free`, since no later read can see them.
pub(super) fn drop_freed_stores(
    func: &mut Function,
    state: &super::State<'_>,
    (bufs, guard): (&mut ScanBuf, &mut FarCseGuard),
    inst: Inst,
) -> bool {
    let Op::Call {
        callee, ref args, ..
    } = func.insts[inst.index()].op
    else {
        return false;
    };
    let Some(&freed) = args.first() else {
        return false;
    };
    let Some(def) = state.defs.get(callee.index()).copied().flatten() else {
        return false;
    };
    if !matches!(
        func.insts[def.index()].op,
        Op::Runtime(crate::reloc::RuntimeSymbol::Free)
    ) {
        return false;
    }
    let base = super::addr_loc(state.prov, state.subst, freed, 0, 1).base;
    drop_stores_if(func, bufs, guard, |ploc| ploc.base == base)
}

fn merge_interval(cov: &mut Vec<(i128, i128)>, mut lo: i128, mut hi: i128) {
    cov.retain(|&(ilo, ihi)| {
        if ihi < lo || hi < ilo {
            return true;
        }
        lo = lo.min(ilo);
        hi = hi.max(ihi);
        false
    });
    let pos = cov.partition_point(|&(ilo, _)| ilo < lo);
    cov.insert(pos, (lo, hi));
}

pub(super) fn drop_covered_stores(
    func: &mut Function,
    bufs: &mut ScanBuf,
    guard: &mut FarCseGuard,
    (new_loc, multi): (MemLoc, bool),
) -> bool {
    let nlo = i128::from(new_loc.offset);
    let nhi = nlo + i128::from(new_loc.size);
    if nhi <= nlo
        || !bufs
            .pending_stores
            .iter()
            .any(|&(p, _)| p.base == new_loc.base)
    {
        return false;
    }
    if !multi {
        return drop_stores_if(func, bufs, guard, |ploc| {
            let pend = i128::from(ploc.offset) + i128::from(ploc.size);
            ploc.base == new_loc.base && ploc.offset >= new_loc.offset && pend <= nhi
        });
    }
    let mut cov = vec![(nlo, nhi)];
    let mut dead = vec![false; bufs.pending_stores.len()];
    let mut any_dead = false;
    for (i, &(ploc, _)) in bufs.pending_stores.iter().enumerate().rev() {
        if ploc.base != new_loc.base {
            continue;
        }
        let plo = i128::from(ploc.offset);
        let phi = plo + i128::from(ploc.size);
        if cov.iter().any(|&(ilo, ihi)| ilo <= plo && phi <= ihi) {
            dead[i] = true;
            any_dead = true;
        } else if phi > plo {
            merge_interval(&mut cov, plo, phi);
        }
    }
    if !any_dead {
        return false;
    }
    let mut idx = 0_usize;
    let mut changed = false;
    bufs.pending_stores.retain(|&(_, kept_idx)| {
        let drop_it = dead[idx];
        idx += 1;
        if !drop_it {
            return true;
        }
        changed |= kill_kept_store(func, &mut bufs.kept, guard, kept_idx);
        false
    });
    changed
}

#[derive(Clone, Copy)]
pub(super) struct SubwordSlice {
    pub loc: MemLoc,
    pub orig_off: i64,
    pub ty: Type,
    pub val: Value,
}

pub(super) struct SubwordSlices(Vec<SubwordSlice>);

fn clip_span((lo, hi): (i64, i64), (wlo, whi): (i64, i64)) -> [Option<(i64, i64)>; 2] {
    if wlo >= hi || whi <= lo {
        return [Some((lo, hi)), None];
    }
    [
        (wlo > lo).then_some((lo, wlo)),
        (whi < hi).then_some((whi, hi)),
    ]
}

fn push_clipped_slice(
    out: &mut Vec<SubwordSlice>,
    wloc: MemLoc,
    w_span: (i64, i64),
    (orig_off, ty, val): (i64, Type, Value),
    span: (i64, i64),
) {
    for sub in clip_span(span, w_span).into_iter().flatten() {
        if let Ok(size) = u32::try_from(sub.1 - sub.0) {
            out.push(SubwordSlice {
                loc: MemLoc::new(wloc.root, wloc.base, sub.0, size),
                orig_off,
                ty,
                val,
            });
        }
    }
}

pub(super) fn const_slice_imm(byte_shift: i64, ty: Type, imm: i64) -> Option<i64> {
    if !(0..=7).contains(&byte_shift) || int_of(ty).is_none() {
        return None;
    }
    let shifted = (imm as u64) >> (byte_shift * 8);
    let masked = match ty {
        Type::F32 => shifted & 0xFFFF_FFFF,
        Type::F64 => shifted,
        _ => shifted & ty.unsigned_max(),
    };
    Some(masked as i64)
}

pub(super) fn ext_inner_source(
    func: &Function,
    defs: &[Option<Inst>],
    subst: &[Value],
    val: Value,
) -> Option<(Convert, Value, Type)> {
    let inst = defs.get(val.index()).copied().flatten()?;
    if let Op::Convert(kind @ (Convert::Uextend | Convert::Sextend), inner) =
        func.insts[inst.index()].op
    {
        let canon = find(subst, inner);
        return Some((kind, canon, func.ty(canon)));
    }
    None
}

impl SubwordSlices {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn prune_on_write(
        &mut self,
        func: &Function,
        prov: &Provenance,
        wloc: MemLoc,
        at: Option<Inst>,
    ) {
        let Some(w_hi) = wloc.offset.checked_add(i64::from(wloc.size)) else {
            self.0.retain(|s| loc_immutable(func, s.loc));
            return;
        };
        let old = std::mem::take(&mut self.0);
        for s in old {
            if s.loc.base == wloc.base {
                if let Some(s_hi) = s.loc.offset.checked_add(i64::from(s.loc.size)) {
                    push_clipped_slice(
                        &mut self.0,
                        wloc,
                        (wloc.offset, w_hi),
                        (s.orig_off, s.ty, s.val),
                        (s.loc.offset, s_hi),
                    );
                }
            } else if (loc_immutable(func, s.loc)
                && (matches!(s.loc.root, MemRoot::Global(_)) || wloc.root != s.loc.root))
                || !may_alias_at(wloc, s.loc, prov, at)
            {
                self.0.push(s);
            }
        }
    }

    pub(super) fn save_split_slices(&mut self, known: &KnownLoads, subst: &[Value], wloc: MemLoc) {
        let Some(w_hi) = wloc.offset.checked_add(i64::from(wloc.size)) else {
            return;
        };
        for &(koff, kty, kval) in known.entries_for(wloc.base) {
            if !kty.is_small_int() || kty.bytes() < 2 {
                continue;
            }
            let Some(k_hi) = koff.checked_add(i64::from(kty.bytes())) else {
                continue;
            };
            if koff < w_hi && wloc.offset < k_hi && !(wloc.offset <= koff && k_hi <= w_hi) {
                push_clipped_slice(
                    &mut self.0,
                    wloc,
                    (wloc.offset, w_hi),
                    (koff, kty, find(subst, kval)),
                    (koff, k_hi),
                );
            }
        }
    }

    pub(super) fn on_unknown(&mut self, func: &Function, op: &Op, prov: &Provenance) {
        self.0
            .retain(|s| !clobbers_loc(func, (op, None), s.loc, prov));
    }

    pub(super) fn find(&self, loc: MemLoc, ty: Type) -> Option<SubwordSlice> {
        int_of(ty)?;
        let end = loc.offset.checked_add(i64::from(ty.bytes()))?;
        self.0
            .iter()
            .filter(|s| {
                s.loc.base == loc.base
                    && s.loc.offset <= loc.offset
                    && s.loc
                        .offset
                        .checked_add(i64::from(s.loc.size))
                        .is_some_and(|e| end <= e)
            })
            .min_by_key(|s| (s.ty.bytes(), loc.offset - s.orig_off))
            .copied()
    }
}

pub(super) enum SubwordAction {
    Forwarded(Value),
    Const(i64),
    Plan(SubwordPlan),
}

fn try_subword_source(
    known: &KnownLoads,
    slices: &SubwordSlices,
    subst: &[Value],
    loc: MemLoc,
    ty: Type,
) -> Option<(i64, Type, Value)> {
    if let Some((koff, kty, kval)) = find_subword_source(known, loc, ty) {
        return Some((koff, kty, find(subst, kval)));
    }
    let slice = slices.find(loc, ty)?;
    Some((slice.orig_off, slice.ty, find(subst, slice.val)))
}

fn subword_shortcut(
    func: &Function,
    (loc, ty): (MemLoc, Type),
    (koff, kty, kval): (i64, Type, Value),
    (defs, subst, consts): (&[Option<Inst>], &[Value], &[Option<i64>]),
) -> Option<SubwordAction> {
    if kty.is_small_int()
        && let Some(imm) = consts.get(kval.index()).copied().flatten()
        && let Some(cval) = const_slice_imm(loc.offset - koff, ty, imm)
    {
        return Some(SubwordAction::Const(cval));
    }
    let (kind, inner, inner_ty) = ext_inner_source(func, defs, subst, kval)?;
    if loc.offset == koff && inner_ty == ty {
        return Some(SubwordAction::Forwarded(inner));
    }
    if kind == Convert::Uextend
        && inner_ty.is_small_int()
        && koff
            .checked_add(i64::from(inner_ty.bytes()))
            .is_some_and(|end| loc.offset >= end)
    {
        return Some(SubwordAction::Const(0));
    }
    None
}

pub(super) fn resolve_subword_plan(
    func: &Function,
    (inst, loc, ty): (Inst, MemLoc, Type),
    (known, slices, guard): (&KnownLoads, &SubwordSlices, &FarCseGuard),
    (post_inline, defs, subst, consts): (bool, &[Option<Inst>], &[Value], &[Option<i64>]),
) -> Option<SubwordAction> {
    if !post_inline {
        let (koff, kty, kval) = find_subword_source(known, loc, ty)?;
        return SubwordPlan::new(loc.offset, ty, koff, kty, kval).map(SubwordAction::Plan);
    }
    let (koff, kty, kval) = try_subword_source(known, slices, subst, loc, ty)?;
    if guard.should_skip_value(&func.insts[inst.index()].op, ty, kval) {
        return None;
    }
    if let Some(hit) = subword_shortcut(func, (loc, ty), (koff, kty, kval), (defs, subst, consts)) {
        return Some(hit);
    }
    SubwordPlan::new(loc.offset, ty, koff, kty, kval).map(SubwordAction::Plan)
}

fn for_term_args(op: &Op, mut f: impl FnMut(Value)) -> bool {
    match op {
        Op::Jump(_, args) => args.iter().for_each(|&v| f(v)),
        Op::Brif(_, _, t_args, _, e_args) => t_args.iter().chain(e_args).for_each(|&v| f(v)),
        _ => return false,
    }
    true
}

fn has_sink_cand(func: &Function) -> bool {
    for b in &func.blocks {
        if b.insts.len() < 2 {
            continue;
        }
        let last = b.insts[b.insts.len() - 1];
        match &func.insts[last.index()].op {
            Op::Jump(_, args) if !args.is_empty() => return true,
            Op::Brif(c, _, t_args, _, e_args)
                if !t_args.is_empty()
                    || !e_args.is_empty()
                    || func.insts[b.insts[b.insts.len() - 2].index()].result != Some(*c) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn collect_fits_used(func: &Function) -> Vec<bool> {
    let mut fits_used = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::Fits(_, _, x, y) = func.insts[inst.index()].op {
            if fits_used.is_empty() {
                fits_used.resize(func.types.len(), false);
            }
            fits_used[x.index()] = true;
            fits_used[y.index()] = true;
        }
    }
    fits_used
}

fn is_sunk_cond(func: &Function, inst: Inst, cond: Option<Value>) -> bool {
    let res = func.insts[inst.index()].result;
    res.is_some() && res == cond && crate::schedule::is_condition(func, inst)
}

fn is_sunk_arg(
    func: &Function,
    inst: Inst,
    fits_used: &[bool],
    uses: &[u32],
    term_uses: &[u32],
) -> bool {
    let data = &func.insts[inst.index()];
    data.result.is_some_and(|res| {
        data.op.is_speculable()
            && !fits_used.get(res.index()).copied().unwrap_or(false)
            && term_uses[res.index()] > 0
            && uses[res.index()] == term_uses[res.index()]
    })
}

fn sink_block_term_defs(
    func: &mut Function,
    b: usize,
    uses: &[u32],
    fits_used: &[bool],
    term_uses: &mut [u32],
) -> bool {
    let len = func.blocks[b].insts.len();
    if len < 2 {
        return false;
    }
    let last = func.blocks[b].insts[len - 1];
    let mut arg_count = 0_u32;
    if !for_term_args(&func.insts[last.index()].op, |v| {
        term_uses[v.index()] += 1;
        arg_count += 1;
    }) {
        return false;
    }
    let cond = match func.insts[last.index()].op {
        Op::Brif(c, ..) if uses[c.index()] == 1 => Some(c),
        _ => None,
    };
    if arg_count == 0
        && cond.is_none_or(|c| func.insts[func.blocks[b].insts[len - 2].index()].result == Some(c))
    {
        return false;
    }
    let needs_move = func.blocks[b].insts[..len - 1]
        .iter()
        .enumerate()
        .any(|(pos, &inst)| {
            if is_sunk_cond(func, inst, cond) {
                pos != len - 2
            } else {
                arg_count > 0 && is_sunk_arg(func, inst, fits_used, uses, term_uses)
            }
        });
    if !needs_move {
        if arg_count > 0 {
            for_term_args(&func.insts[last.index()].op, |v| term_uses[v.index()] = 0);
        }
        return false;
    }
    let insts = std::mem::take(&mut func.blocks[b].insts);
    let mut kept = Vec::with_capacity(len);
    let mut sunk = Vec::new();
    let mut tail = None;
    for &inst in &insts[..len - 1] {
        if is_sunk_cond(func, inst, cond) {
            tail = Some(inst);
        } else if is_sunk_arg(func, inst, fits_used, uses, term_uses) {
            sunk.push(inst);
        } else {
            kept.push(inst);
        }
    }
    kept.extend(sunk);
    kept.extend(tail);
    kept.push(last);
    let changed = kept != insts;
    func.blocks[b].insts = kept;
    if arg_count > 0 {
        for_term_args(&func.insts[last.index()].op, |v| term_uses[v.index()] = 0);
    }
    changed
}

pub(super) fn sink_terminator_defs(func: &mut Function) -> bool {
    if !has_sink_cand(func) {
        return false;
    }
    let uses = func.use_counts();
    let fits_used = collect_fits_used(func);
    let mut term_uses = vec![0_u32; func.types.len()];
    let mut changed = false;
    for b in 0..func.blocks.len() {
        changed |= sink_block_term_defs(func, b, &uses, &fits_used, &mut term_uses);
    }
    changed
}
