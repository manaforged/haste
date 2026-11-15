use crate::ir::{Block, Convert, Function, Inst, Op, Type, Value};
use crate::mem::{KnownLoads, MemLoc, MemRoot, Provenance, is_hard_barrier, may_alias};
use crate::merge::find;

use super::coalesce::ScanBuf;
use super::dom::{clobbers_loc, loc_immutable};
use super::join::{SubwordPlan, find_subword_source, int_of};

const MIN_BLOCK_INSTS: usize = 128;
const FAR_MIN_DIST: usize = 128;
const MIN_GAP_DEFS: usize = 24;
const PAIR_MAX_WINDOW: usize = 16;

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
}

impl FarCseGuard {
    pub(super) fn new(block_len: usize, num_values: usize) -> Self {
        if block_len < MIN_BLOCK_INSTS {
            return Self { active: false, cur_pos: 0, cur_defs: 0, loads: Vec::new() };
        }
        Self { active: true, cur_pos: 0, cur_defs: 0, loads: vec![None; num_values] }
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

    pub(super) fn check_cse(&self, op: &Op, ty: Type, forwarded: Option<Value>) -> (Option<Value>, bool) {
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

fn is_mem_write(op: &Op) -> bool {
    matches!(op, Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..))
}

struct RawCand {
    def_pos: usize,
    val: Value,
    ptr: Value,
    off: i32,
    ty: Type,
}

#[derive(Clone, Copy)]
struct Plan {
    def_pos: usize,
    orig_val: Value,
    ptr: Value,
    off: i32,
    ty: Type,
    last_near_use: usize,
    first_far_use: usize,
}

struct Insertion {
    pos: usize,
    plans: Vec<Plan>,
}

struct BlockScan {
    def_prefix: Vec<usize>,
    hard_barriers: Vec<usize>,
    writes: Vec<usize>,
    in_block_uses: Vec<u32>,
    last_use: Vec<Option<usize>>,
    def_in_block: Vec<bool>,
    raw_cands: Vec<RawCand>,
    cand_uses: Vec<Vec<usize>>,
}

struct ScanCtx<'a> {
    func: &'a Function,
    prov: &'a Provenance,
    insts: &'a [Inst],
    all_uses: &'a [u32],
    scan: &'a BlockScan,
}

fn scan_block(func: &Function, insts: &[Inst]) -> BlockScan {
    let n_vals = func.types.len();
    let len = insts.len();
    let mut def_prefix = Vec::with_capacity(len + 1);
    let mut hard_barriers = Vec::with_capacity(len + 1);
    def_prefix.push(0);
    hard_barriers.push(0);
    let mut writes = Vec::new();
    let mut in_block_uses = vec![0_u32; n_vals];
    let mut last_use = vec![None; n_vals];
    let mut def_in_block = vec![false; n_vals];
    let mut cand_idx = vec![usize::MAX; n_vals];
    let mut raw_cands = Vec::new();
    let mut cand_uses: Vec<Vec<usize>> = Vec::new();

    for (pos, &inst) in insts.iter().enumerate() {
        let data = &func.insts[inst.index()];
        def_prefix.push(def_prefix[pos] + usize::from(data.result.is_some()));
        hard_barriers.push(hard_barriers[pos] + usize::from(is_hard_barrier(&data.op)));
        if is_mem_write(&data.op) {
            writes.push(pos);
        }
        data.op.for_operands(|opnd| {
            let idx = opnd.index();
            in_block_uses[idx] += 1;
            last_use[idx] = Some(pos);
            let ci = cand_idx[idx];
            if ci != usize::MAX && cand_uses[ci].last() != Some(&pos) {
                cand_uses[ci].push(pos);
            }
        });
        if let Some(res) = data.result {
            def_in_block[res.index()] = true;
            if let Op::Load(ptr, off) = data.op {
                let ty = func.ty(res);
                if is_scalar_remat_ty(ty) {
                    cand_idx[res.index()] = raw_cands.len();
                    raw_cands.push(RawCand { def_pos: pos, val: res, ptr, off, ty });
                    cand_uses.push(Vec::new());
                }
            }
        }
    }

    BlockScan { def_prefix, hard_barriers, writes, in_block_uses, last_use, def_in_block, raw_cands, cand_uses }
}

fn has_clobber_between(ctx: &ScanCtx<'_>, raw: &RawCand, end_pos: usize) -> bool {
    let w_lo = ctx.scan.writes.partition_point(|&w| w <= raw.def_pos);
    let w_hi = ctx.scan.writes.partition_point(|&w| w < end_pos);
    if w_lo >= w_hi {
        return false;
    }
    let read_loc = ctx.prov.loc(raw.ptr, i64::from(raw.off), raw.ty.bytes());
    ctx.scan.writes[w_lo..w_hi]
        .iter()
        .any(|&w| clobbers_loc(ctx.func, &ctx.func.insts[ctx.insts[w].index()].op, read_loc, ctx.prov))
}

fn ptr_live_at_or_past(ctx: &ScanCtx<'_>, ptr: Value, pos: usize) -> bool {
    let idx = ptr.index();
    ctx.scan.last_use[idx].is_some_and(|u| u >= pos)
        || (ctx.scan.def_in_block[idx] && ctx.scan.in_block_uses[idx] < ctx.all_uses[idx])
}

fn first_chasm(uses: &[usize], def_prefix: &[usize]) -> Option<(usize, usize)> {
    uses.windows(2).find_map(|pair| {
        let (last_near, first_far) = (pair[0], pair[1]);
        (first_far >= last_near + FAR_MIN_DIST
            && def_prefix[first_far] - def_prefix[last_near + 1] >= MIN_GAP_DEFS)
            .then_some((last_near, first_far))
    })
}

fn validate_cand(ctx: &ScanCtx<'_>, raw: &RawCand, uses: &[usize]) -> Option<Plan> {
    if ctx.scan.in_block_uses[raw.val.index()] != ctx.all_uses[raw.val.index()] {
        return None;
    }
    let (last_near_use, first_far_use) = first_chasm(uses, &ctx.scan.def_prefix)?;
    if ctx.scan.hard_barriers[first_far_use] != ctx.scan.hard_barriers[raw.def_pos + 1] {
        return None;
    }
    if has_clobber_between(ctx, raw, first_far_use) {
        return None;
    }
    Some(Plan { def_pos: raw.def_pos, orig_val: raw.val, ptr: raw.ptr, off: raw.off, ty: raw.ty, last_near_use, first_far_use })
}

fn can_pair(a: &Plan, b: &Plan, def_prefix: &[usize]) -> bool {
    if a.ptr != b.ptr || a.ty.bytes() != b.ty.bytes() || a.ty.is_float() != b.ty.is_float() || !matches!(a.ty.bytes(), 4 | 8) {
        return false;
    }
    if i64::from(a.off).checked_add(i64::from(a.ty.bytes())) != Some(i64::from(b.off)) {
        return false;
    }
    if a.first_far_use.abs_diff(b.first_far_use) > PAIR_MAX_WINDOW {
        return false;
    }
    let pos = a.first_far_use.min(b.first_far_use);
    let later = if a.first_far_use > pos { a } else { b };
    pos >= later.last_near_use + FAR_MIN_DIST && def_prefix[pos] - def_prefix[later.last_near_use + 1] >= MIN_GAP_DEFS
}

fn find_pair_partner(plans: &[Plan], paired: &[bool], i: usize, def_prefix: &[usize]) -> Option<usize> {
    let a = &plans[i];
    let max_off = i64::from(a.off).checked_add(i64::from(a.ty.bytes()))?;
    for j in (i + 1)..plans.len() {
        if plans[j].ptr != a.ptr || i64::from(plans[j].off) > max_off {
            break;
        }
        if !paired[j] && can_pair(a, &plans[j], def_prefix) {
            return Some(j);
        }
    }
    None
}

fn group_insertions(mut plans: Vec<Plan>, def_prefix: &[usize]) -> Vec<Insertion> {
    plans.sort_by_key(|p| (p.ptr.0, p.off, p.first_far_use));
    let mut paired = vec![false; plans.len()];
    let mut insertions = Vec::with_capacity(plans.len());
    for i in 0..plans.len() {
        if paired[i] {
            continue;
        }
        paired[i] = true;
        if let Some(j) = find_pair_partner(&plans, &paired, i, def_prefix) {
            paired[j] = true;
            let pos = plans[i].first_far_use.min(plans[j].first_far_use);
            insertions.push(Insertion { pos, plans: vec![plans[i], plans[j]] });
        } else {
            insertions.push(Insertion { pos: plans[i].first_far_use, plans: vec![plans[i]] });
        }
    }
    insertions.sort_by_key(|ins| (ins.pos, ins.plans[0].def_pos));
    insertions
}

fn emit_group(func: &mut Function, group: &[Plan], out: &mut Vec<Inst>, remap: &mut [Option<Value>]) {
    for plan in group {
        let ptr = remap.get(plan.ptr.index()).copied().flatten().unwrap_or(plan.ptr);
        let new_val = func.value(plan.ty);
        out.push(func.add(Op::Load(ptr, plan.off), Some(new_val)));
        remap[plan.orig_val.index()] = Some(new_val);
    }
}

fn apply_insertions(func: &mut Function, block: Block, insertions: Vec<Insertion>) {
    let extra: usize = insertions.iter().map(|ins| ins.plans.len()).sum();
    let insts = std::mem::take(&mut func.blocks[block.index()].insts);
    let mut out = Vec::with_capacity(insts.len() + extra);
    let mut remap: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut any_remap = false;
    let mut ins_idx = 0;

    for (pos, &inst) in insts.iter().enumerate() {
        while ins_idx < insertions.len() && insertions[ins_idx].pos == pos {
            emit_group(func, &insertions[ins_idx].plans, &mut out, &mut remap);
            any_remap = true;
            ins_idx += 1;
        }
        if any_remap {
            func.insts[inst.index()].op.map_operands(|v| remap.get(v.index()).copied().flatten().unwrap_or(v));
        }
        out.push(inst);
    }
    func.blocks[block.index()].insts = out;
}

fn plan_survives(ctx: &ScanCtx<'_>, plans: &[Plan], i: usize) -> bool {
    let plan = &plans[i];
    if ptr_live_at_or_past(ctx, plan.ptr, plan.first_far_use) {
        return true;
    }
    plans.iter().enumerate().any(|(j, other)| {
        j != i
            && other.orig_val != plan.orig_val
            && other.ptr == plan.ptr
            && plan.last_near_use.max(other.last_near_use) < plan.first_far_use.min(other.first_far_use)
    })
}

fn plan_block(func: &Function, prov: &Provenance, all_uses: &[u32], insts: &[Inst]) -> Vec<Insertion> {
    let scan = scan_block(func, insts);
    if scan.raw_cands.is_empty() {
        return Vec::new();
    }
    let ctx = ScanCtx { func, prov, insts, all_uses, scan: &scan };
    let mut plans: Vec<Plan> = scan
        .raw_cands
        .iter()
        .zip(&scan.cand_uses)
        .filter_map(|(raw, uses)| validate_cand(&ctx, raw, uses))
        .collect();
    let keep: Vec<bool> = (0..plans.len()).map(|i| plan_survives(&ctx, &plans, i)).collect();
    let mut idx = 0_usize;
    plans.retain(|_| {
        let ok = keep[idx];
        idx += 1;
        ok
    });
    if plans.is_empty() {
        return Vec::new();
    }
    group_insertions(plans, &scan.def_prefix)
}

fn split_long_loads_round(func: &mut Function, prov: &Provenance) -> bool {
    let all_uses = func.use_counts();
    let mut changed = false;
    for b_idx in 0..func.blocks.len() {
        if func.blocks[b_idx].insts.len() < MIN_BLOCK_INSTS {
            continue;
        }
        let insertions = plan_block(func, prov, &all_uses, &func.blocks[b_idx].insts);
        if !insertions.is_empty() {
            apply_insertions(func, Block(b_idx as u32), insertions);
            changed = true;
        }
    }
    changed
}

/// Splits long same-block scalar load live ranges across barrier-free instruction chasms by rematerializing before distant uses.
pub(super) fn split_long_loads(func: &mut Function, prov: &Provenance) -> bool {
    const MAX_SPLIT_ROUNDS: usize = 4;
    if !func.blocks.iter().any(|b| b.insts.len() >= MIN_BLOCK_INSTS) {
        return false;
    }
    let mut changed = false;
    for _ in 0..MAX_SPLIT_ROUNDS {
        if !split_long_loads_round(func, prov) {
            break;
        }
        changed = true;
    }
    changed
}

fn kill_kept_store(func: &mut Function, kept: &mut [Option<Inst>], guard: &mut FarCseGuard, kept_idx: usize) -> bool {
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

pub(super) fn drop_covered_stores(func: &mut Function, bufs: &mut ScanBuf, guard: &mut FarCseGuard, (new_loc, multi): (MemLoc, bool)) -> bool {
    let nlo = i128::from(new_loc.offset);
    let nhi = nlo + i128::from(new_loc.size);
    if nhi <= nlo || !bufs.pending_stores.iter().any(|&(p, _)| p.base == new_loc.base) {
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
    [(wlo > lo).then_some((lo, wlo)), (whi < hi).then_some((whi, hi))]
}

fn push_clipped_slice(out: &mut Vec<SubwordSlice>, wloc: MemLoc, w_span: (i64, i64), (orig_off, ty, val): (i64, Type, Value), span: (i64, i64)) {
    for sub in clip_span(span, w_span).into_iter().flatten() {
        if let Ok(size) = u32::try_from(sub.1 - sub.0) {
            out.push(SubwordSlice { loc: MemLoc::new(wloc.root, wloc.base, sub.0, size), orig_off, ty, val });
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

pub(super) fn ext_inner_source(func: &Function, defs: &[Option<Inst>], subst: &[Value], val: Value) -> Option<(Convert, Value, Type)> {
    let inst = defs.get(val.index()).copied().flatten()?;
    if let Op::Convert(kind @ (Convert::Uextend | Convert::Sextend), inner) = func.insts[inst.index()].op {
        let canon = find(subst, inner);
        return Some((kind, canon, func.ty(canon)));
    }
    None
}

impl SubwordSlices {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn prune_on_write(&mut self, func: &Function, prov: &Provenance, wloc: MemLoc) {
        let Some(w_hi) = wloc.offset.checked_add(i64::from(wloc.size)) else {
            self.0.retain(|s| loc_immutable(func, s.loc));
            return;
        };
        let old = std::mem::take(&mut self.0);
        for s in old {
            if s.loc.base == wloc.base {
                if let Some(s_hi) = s.loc.offset.checked_add(i64::from(s.loc.size)) {
                    push_clipped_slice(&mut self.0, wloc, (wloc.offset, w_hi), (s.orig_off, s.ty, s.val), (s.loc.offset, s_hi));
                }
            } else if (loc_immutable(func, s.loc) && (matches!(s.loc.root, MemRoot::Global(_)) || wloc.root != s.loc.root))
                || !may_alias(wloc, s.loc, prov)
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
            let Some(k_hi) = koff.checked_add(i64::from(kty.bytes())) else { continue };
            if koff < w_hi && wloc.offset < k_hi && !(wloc.offset <= koff && k_hi <= w_hi) {
                push_clipped_slice(&mut self.0, wloc, (wloc.offset, w_hi), (koff, kty, find(subst, kval)), (koff, k_hi));
            }
        }
    }

    pub(super) fn on_unknown(&mut self, func: &Function, op: &Op, prov: &Provenance) {
        self.0.retain(|s| !clobbers_loc(func, op, s.loc, prov));
    }

    pub(super) fn find(&self, loc: MemLoc, ty: Type) -> Option<SubwordSlice> {
        int_of(ty)?;
        let end = loc.offset.checked_add(i64::from(ty.bytes()))?;
        self.0
            .iter()
            .filter(|s| {
                s.loc.base == loc.base
                    && s.loc.offset <= loc.offset
                    && s.loc.offset.checked_add(i64::from(s.loc.size)).is_some_and(|e| end <= e)
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
        && koff.checked_add(i64::from(inner_ty.bytes())).is_some_and(|end| loc.offset >= end)
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

pub(super) fn sink_terminator_defs(func: &mut Function) -> bool {
    let uses = func.use_counts();
    let mut fits_used = vec![false; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::Fits(_, _, x, y) = func.insts[inst.index()].op {
            fits_used[x.index()] = true;
            fits_used[y.index()] = true;
        }
    }
    let mut term_uses = vec![0_u32; func.types.len()];
    let mut changed = false;
    for b in 0..func.blocks.len() {
        if func.blocks[b].insts.len() < 2 {
            continue;
        }
        let Some(&last) = func.blocks[b].insts.last() else { continue };
        if !for_term_args(&func.insts[last.index()].op, |v| term_uses[v.index()] += 1) {
            continue;
        }
        let insts = std::mem::take(&mut func.blocks[b].insts);
        let len = insts.len();
        let mut kept = Vec::with_capacity(len);
        let mut sunk = Vec::new();
        for &inst in &insts[..len - 1] {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result
                && data.op.is_speculable()
                && !fits_used[res.index()]
                && term_uses[res.index()] > 0
                && uses[res.index()] == term_uses[res.index()]
            {
                sunk.push(inst);
            } else {
                kept.push(inst);
            }
        }
        if !sunk.is_empty() && sunk.iter().enumerate().any(|(idx, &inst)| insts[len - 1 - sunk.len() + idx] != inst) {
            changed = true;
        }
        kept.extend(sunk);
        kept.push(last);
        func.blocks[b].insts = kept;
        for_term_args(&func.insts[last.index()].op, |v| term_uses[v.index()] = 0);
    }
    changed
}
