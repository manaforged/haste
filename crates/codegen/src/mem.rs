mod loop_mem;

use std::cell::OnceCell;

use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::ir::{AbiParam, ArgumentPurpose, Binary, Block, BlockData, Convert, Function, Inst, Op, StackSlot, Type, Value};
pub(crate) use loop_mem::{LoopMem, PromoteBase, PromoteCand};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum MemRoot {
    Slot(StackSlot),
    NoAliasParam(u16),
    ReadonlyParam(u16),
    Global(u32),
    Unknown,
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
        Self { root, base, offset, size }
    }

    pub(crate) fn slot(slot: StackSlot, offset: i64, size: u32) -> Self {
        Self { root: MemRoot::Slot(slot), base: MemBase::Slot(slot), offset, size }
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrackRoot {
    Unset,
    Set(MemRoot),
    Conflict,
}

impl TrackRoot {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unset, x) | (x, Self::Unset) => x,
            (Self::Set(a), Self::Set(b)) if a == b => Self::Set(a),
            _ => Self::Conflict,
        }
    }
}

#[derive(Default)]
pub(crate) struct Provenance {
    pub slot_offsets: Vec<Option<(StackSlot, i32)>>,
    pub unescaped_slots: Vec<bool>,
    pub call_readonly_slots: Vec<bool>,
    pub noalias_escaped: Vec<bool>,
    pub param_roots: Vec<MemRoot>,
    pub symbol_defs: Vec<Option<u32>>,
    pub ro_bytes: Vec<u32>,
    pub affine: Vec<(Value, i64)>,
    pub spans: Vec<Option<(MemBase, i64, i64)>>,
}

struct UseCounts {
    total: Vec<u32>,
    safe: Vec<u32>,
    ro_calls: Vec<u32>,
}

struct SlotEscapes {
    unescaped: Vec<bool>,
    call_readonly: Vec<bool>,
    noalias: Vec<bool>,
}

impl SlotEscapes {
    fn new(func: &Function) -> Self {
        Self { unescaped: vec![true; func.slots.len()], call_readonly: vec![true; func.slots.len()], noalias: vec![false; func.signature.params.len()] }
    }
}

fn slot_addresses(func: &Function) -> Vec<Option<(StackSlot, i32)>> {
    func.slot_offsets(|_| None)
        .0
        .into_iter()
        .map(|entry| entry.and_then(|(slot, off)| i32::try_from(off).ok().map(|o| (slot, o))))
        .collect()
}

fn finish_slot_escape(
    func: &Function,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    counts: &UseCounts,
    is_spill: Option<&[bool]>,
) -> SlotEscapes {
    let mut esc = SlotEscapes::new(func);
    for (v, slot) in slots.iter().enumerate() {
        let resolved = slot.map(|(s, _)| s).or(match tracked[v] {
            TrackRoot::Set(MemRoot::Slot(s)) => Some(s),
            _ => None,
        });
        if let Some(s) = resolved {
            let spilled = is_spill.is_some_and(|sp| sp.get(v).copied().unwrap_or(false));
            if spilled || counts.safe[v] != counts.total[v] {
                esc.unescaped[s.index()] = false;
            }
            if spilled || counts.safe[v] + counts.ro_calls[v] != counts.total[v] {
                esc.call_readonly[s.index()] = false;
            }
        }
        if let TrackRoot::Set(MemRoot::NoAliasParam(p)) = tracked[v]
            && counts.safe[v] != counts.total[v]
        {
            esc.noalias[usize::from(p)] = true;
        }
    }
    esc
}

fn count_safe_args(func: &Function, target: Block, args: &[Value], tracked: &[TrackRoot], safe: &mut [u32]) {
    for (&param, &arg) in func.blocks[target.index()].params.iter().zip(args) {
        if let TrackRoot::Set(root) = tracked[param.index()]
            && tracked[arg.index()] == TrackRoot::Set(root)
        {
            safe[arg.index()] += 1;
        }
    }
}

fn is_ptr_capable(ty: Type) -> bool {
    matches!(ty, Type::I64 | Type::I128 | Type::Vec { .. })
}

fn non_ptr_values(func: &Function, cfg: &Cfg) -> Vec<bool> {
    let n = func.types.len();
    let mut non_ptr = vec![true; n];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let op = &func.insts[inst.index()].op;
            if matches!(op, Op::Icmp(..) | Op::Trapz(_)) {
                continue;
            }
            op.for_operands(|v| non_ptr[v.index()] = false);
        }
    }
    non_ptr
}

fn is_unknown_mem_op(op: &Op, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot]) -> bool {
    let known = |v: Value| {
        slots[v.index()].is_some()
            || matches!(
                tracked[v.index()],
                TrackRoot::Set(MemRoot::Slot(_) | MemRoot::NoAliasParam(_) | MemRoot::ReadonlyParam(_))
            )
    };
    match *op {
        Op::Load(addr, _) | Op::Store(_, addr, _) => !known(addr),
        Op::Move(dst, src, _) | Op::Copy(dst, src, _) => !known(dst) || !known(src),
        Op::Call { .. }
        | Op::TryCall { .. }
        | Op::AtomicLoad(_)
        | Op::AtomicStore(..)
        | Op::AtomicRmw(..)
        | Op::AtomicCas(..)
        | Op::Fence => true,
        _ => false,
    }
}

fn compute_block_danger(func: &Function, cfg: &Cfg, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot]) -> (Vec<bool>, Vec<bool>, Vec<usize>) {
    let n = func.blocks.len();
    let loops = cfg.loops();
    let mut in_loop = vec![false; n];
    for (i, &depth) in loops.depths.iter().enumerate() {
        in_loop[i] = depth > 0;
    }
    let mut danger = vec![false; n];
    let mut unknown_end = vec![0_usize; n];
    let mut queue = Vec::new();
    for &b in &cfg.rpo {
        let bi = b.index();
        unknown_end[bi] = func.blocks[bi].insts.iter().rposition(|&inst| is_unknown_mem_op(&func.insts[inst.index()].op, slots, tracked)).map_or(0, |pos| pos + 1);
        if in_loop[bi] || unknown_end[bi] > 0 {
            danger[bi] = true;
            queue.push(b);
        }
    }
    while let Some(b) = queue.pop() {
        for &pred in cfg.preds_of(b) {
            let pi = pred.index();
            if !danger[pi] {
                danger[pi] = true;
                queue.push(pred);
            }
        }
    }
    (in_loop, danger, unknown_end)
}

fn post_call_safe(func: &Function, block: Block, pos: usize, in_loop: &[bool], danger: &[bool], unknown_end: &[usize]) -> bool {
    if in_loop[block.index()] || unknown_end[block.index()] > pos + 1 {
        return false;
    }
    let Some(term) = func.blocks[block.index()].insts.last() else { return false };
    !func.insts[term.index()].op.targets().any(|t| danger[t.index()])
}

fn non_capturing_call_arg(
    func: &Function,
    in_loop: bool,
    (is_const, non_ptr, tracked): (&[bool], &[bool], &[TrackRoot]),
    (params, args, result, results): (&[AbiParam], &[Value], Option<Value>, &[Value]),
    (arg_idx, p): (usize, u16),
) -> bool {
    let Some(pa) = params.get(arg_idx) else { return false };
    if !pa.readonly && (in_loop || !pa.noalias) {
        return false;
    }
    if params.iter().any(|param| param.purpose == ArgumentPurpose::StructReturn) {
        return false;
    }
    for r in result.into_iter().chain(results.iter().copied()) {
        if is_ptr_capable(func.ty(r)) && !non_ptr[r.index()] {
            return false;
        }
    }
    args.iter().enumerate().all(|(j, &other)| {
        j == arg_idx
            || is_const[other.index()]
            || !is_ptr_capable(func.ty(other))
            || params.get(j).is_some_and(|q| q.readonly && q.purpose == ArgumentPurpose::Normal)
            || tracked[other.index()] == TrackRoot::Set(MemRoot::NoAliasParam(p))
    })
}

fn classify_uses(
    func: &Function,
    cfg: &Cfg,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    is_spill: Option<&[bool]>,
) -> SlotEscapes {
    if func.slots.is_empty() && !func.signature.params.iter().any(|abi| abi.noalias) {
        return SlotEscapes::new(func);
    }
    let n = func.types.len();
    let mut is_const = vec![false; n];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let data = &func.insts[inst.index()];
            if let (&Op::Const(_), Some(res)) = (&data.op, data.result) {
                is_const[res.index()] = true;
            }
        }
    }
    let non_ptr = OnceCell::new();
    let (in_loop, danger, unknown_end) = compute_block_danger(func, cfg, slots, tracked);
    let mut counts = UseCounts { total: vec![0_u32; n], safe: vec![0_u32; n], ro_calls: vec![0_u32; n] };
    let mark_safe = |v: Value, safe: &mut [u32]| {
        if slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_)) {
            safe[v.index()] += 1;
        }
    };
    let mark_param_safe = |v: Value, safe: &mut [u32]| {
        if slots[v.index()].is_none()
            && matches!(tracked[v.index()], TrackRoot::Set(MemRoot::NoAliasParam(_) | MemRoot::ReadonlyParam(_)))
        {
            safe[v.index()] += 1;
        }
    };
    for &b in &cfg.rpo {
        for (pos, &inst) in func.blocks[b.index()].insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| counts.total[v.index()] += 1);
            match &data.op {
                Op::Load(addr, _) => mark_safe(*addr, &mut counts.safe),
                Op::Trapz(addr) => mark_param_safe(*addr, &mut counts.safe),
                Op::Store(val, addr, _) if val != addr => mark_safe(*addr, &mut counts.safe),
                Op::Move(dst, src, _) | Op::Copy(dst, src, _) => {
                    mark_param_safe(*dst, &mut counts.safe);
                    mark_param_safe(*src, &mut counts.safe);
                }
                Op::Binary(Binary::Iadd | Binary::Isub, a, b) => {
                    if data.result.is_some_and(|r| slots[r.index()].is_some()) {
                        let target = if slots[a.index()].is_some() { a } else { b };
                        counts.safe[target.index()] += 1;
                    } else if data.result.is_some_and(|r| matches!(tracked[r.index()], TrackRoot::Set(_))) {
                        let target = if matches!(tracked[a.index()], TrackRoot::Set(_)) { a } else { b };
                        counts.safe[target.index()] += 1;
                    }
                }
                Op::Convert(Convert::Bitcast, x) if data.result.is_some_and(|r| matches!(tracked[r.index()], TrackRoot::Set(_))) => {
                    counts.safe[x.index()] += 1;
                }
                Op::Select(_, a, b) => {
                    if let Some(r) = data.result
                        && let TrackRoot::Set(root) = tracked[r.index()]
                    {
                        for v in [a, b] {
                            if tracked[v.index()] == TrackRoot::Set(root) {
                                counts.safe[v.index()] += 1;
                            }
                        }
                    }
                }
                Op::Icmp(_, a, b) => {
                    for (v, other) in [(a, b), (b, a)] {
                        let is_noalias = matches!(tracked[v.index()], TrackRoot::Set(MemRoot::NoAliasParam(_)));
                        let same_or_const = (slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_)))
                            && (is_const[other.index()]
                                || matches!((tracked[v.index()], tracked[other.index()]), (TrackRoot::Set(ra), TrackRoot::Set(rb)) if ra == rb));
                        if is_noalias || same_or_const {
                            counts.safe[v.index()] += 1;
                        }
                    }
                }
                Op::Jump(target, args) => count_safe_args(func, *target, args, tracked, &mut counts.safe),
                Op::Brif(_, then_b, then_args, else_b, else_args) => {
                    count_safe_args(func, *then_b, then_args, tracked, &mut counts.safe);
                    count_safe_args(func, *else_b, else_args, tracked, &mut counts.safe);
                }
                Op::Call { sig, args, results, .. } => {
                    let params = &func.sigs[sig.index()].params;
                    let post_safe = post_call_safe(func, b, pos, &in_loop, &danger, &unknown_end);
                    for (i, &arg) in args.iter().enumerate() {
                        if params.get(i).is_some_and(|p| p.readonly) {
                            counts.ro_calls[arg.index()] += 1;
                        }
                        if slots[arg.index()].is_none()
                            && let TrackRoot::Set(MemRoot::NoAliasParam(p)) = tracked[arg.index()]
                            && (post_safe || non_capturing_call_arg(func, in_loop[b.index()], (&is_const, non_ptr.get_or_init(|| non_ptr_values(func, cfg)), tracked), (params, args, data.result, results), (i, p)))
                        {
                            counts.safe[arg.index()] += 1;
                        }
                    }
                }
                Op::TryCall { sig, args, .. } => {
                    let params = &func.sigs[sig.index()].params;
                    let post_safe = post_call_safe(func, b, pos, &in_loop, &danger, &unknown_end);
                    for (i, &arg) in args.iter().enumerate() {
                        if params.get(i).is_some_and(|p| p.readonly) {
                            counts.ro_calls[arg.index()] += 1;
                        }
                        if slots[arg.index()].is_none()
                            && let TrackRoot::Set(MemRoot::NoAliasParam(_)) = tracked[arg.index()]
                            && post_safe
                        {
                            counts.safe[arg.index()] += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    finish_slot_escape(func, slots, tracked, &counts, is_spill)
}

fn step_track(op: &Op, ty: Type, tracked: &[TrackRoot]) -> TrackRoot {
    if ty != Type::I64 {
        return TrackRoot::Conflict;
    }
    match *op {
        Op::StackAddr(slot, _) => TrackRoot::Set(MemRoot::Slot(slot)),
        Op::Symbol(sym) | Op::SlotAddr(sym) => TrackRoot::Set(MemRoot::Global(sym)),
        Op::Convert(Convert::Bitcast, x) => tracked[x.index()],
        Op::Binary(Binary::Iadd, a, b) => match (tracked[a.index()], tracked[b.index()]) {
            (TrackRoot::Unset, _) | (_, TrackRoot::Unset) => TrackRoot::Unset,
            (TrackRoot::Set(r), TrackRoot::Conflict) | (TrackRoot::Conflict, TrackRoot::Set(r)) => TrackRoot::Set(r),
            _ => TrackRoot::Conflict,
        },
        Op::Binary(Binary::Isub, a, b) => match (tracked[a.index()], tracked[b.index()]) {
            (TrackRoot::Unset, _) | (_, TrackRoot::Unset) => TrackRoot::Unset,
            (TrackRoot::Set(r), TrackRoot::Conflict) => TrackRoot::Set(r),
            _ => TrackRoot::Conflict,
        },
        Op::Select(_, a, b) => tracked[a.index()].join(tracked[b.index()]),
        _ => TrackRoot::Conflict,
    }
}

fn join_track_args(tracked: &mut [TrackRoot], blocks: &[BlockData], target: Block, args: &[Value]) -> bool {
    let mut changed = false;
    for (&param, &arg) in blocks[target.index()].params.iter().zip(args) {
        if arg != param {
            let next = tracked[param.index()].join(tracked[arg.index()]);
            if next != tracked[param.index()] {
                tracked[param.index()] = next;
                changed = true;
            }
        }
    }
    changed
}

fn seed_entry_roots(func: &Function, cfg: &Cfg, tracked: &mut [TrackRoot]) {
    let Some(entry) = cfg.entry else { return };
    for (i, (&param, abi)) in func.blocks[entry.index()].params.iter().zip(&func.signature.params).enumerate() {
        let root = u16::try_from(i).ok().and_then(|idx| {
            if abi.noalias {
                Some(MemRoot::NoAliasParam(idx))
            } else if abi.readonly {
                Some(MemRoot::ReadonlyParam(idx))
            } else {
                None
            }
        });
        tracked[param.index()] = root.map_or(TrackRoot::Conflict, TrackRoot::Set);
    }
}

fn is_derived_i64(op: &Op, ty: Type) -> bool {
    ty == Type::I64 && matches!(op, Op::Convert(Convert::Bitcast, _) | Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Select(..))
}

fn track_def(func: &Function, inst: Inst, tracked: &mut [TrackRoot]) -> bool {
    let data = &func.insts[inst.index()];
    let Some(res) = data.result else { return false };
    let next = step_track(&data.op, func.ty(res), tracked);
    let changed = next != tracked[res.index()];
    tracked[res.index()] = next;
    changed
}

fn track_edges(func: &Function, inst: Inst, tracked: &mut [TrackRoot]) -> bool {
    match &func.insts[inst.index()].op {
        Op::Jump(t, args) => join_track_args(tracked, &func.blocks, *t, args),
        Op::Brif(_, t, ta, f, fa) => join_track_args(tracked, &func.blocks, *t, ta) | join_track_args(tracked, &func.blocks, *f, fa),
        _ => false,
    }
}

fn compute_tracked_roots(func: &Function, cfg: &Cfg) -> Vec<TrackRoot> {
    let mut tracked = vec![TrackRoot::Unset; func.types.len()];
    seed_entry_roots(func, cfg, &mut tracked);
    let mut steps = Vec::new();
    let mut changed = false;
    for &b in &cfg.rpo {
        let insts = &func.blocks[b.index()].insts;
        for &inst in insts {
            changed |= track_def(func, inst, &mut tracked);
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|res| is_derived_i64(&data.op, func.ty(res))) {
                steps.push((inst, false));
            }
        }
        if let Some(&last) = insts.last()
            && matches!(func.insts[last.index()].op, Op::Jump(..) | Op::Brif(..))
        {
            changed |= track_edges(func, last, &mut tracked);
            steps.push((last, true));
        }
    }
    let mut rounds = 1;
    while changed && rounds < 64 {
        changed = false;
        for &(inst, edges) in &steps {
            changed |= if edges { track_edges(func, inst, &mut tracked) } else { track_def(func, inst, &mut tracked) };
        }
        rounds += 1;
    }
    if changed {
        tracked.fill(TrackRoot::Conflict);
    } else {
        tracked.iter_mut().filter(|item| **item == TrackRoot::Unset).for_each(|item| *item = TrackRoot::Conflict);
    }
    tracked
}

fn step_affine(op: &Op, out: &[(Value, i64)], consts: &[Option<i64>], types: &[Type]) -> Option<(Value, i64)> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => {
            let (base, k) = if let Some(k) = consts[b.index()] {
                (a, k)
            } else {
                (b, consts[a.index()]?)
            };
            let (root, off) = out[base.index()];
            off.checked_add(k).map(|total| (root, total))
        }
        Op::Binary(Binary::Isub, a, b) => {
            let k = consts[b.index()]?.checked_neg()?;
            let (root, off) = out[a.index()];
            off.checked_add(k).map(|total| (root, total))
        }
        Op::Convert(Convert::Bitcast, x) => (types[x.index()] == Type::I64).then_some(out[x.index()]),
        Op::Select(_, a, b) => (out[a.index()] == out[b.index()]).then_some(out[a.index()]),
        _ => None,
    }
}

pub(crate) fn compute_affine(func: &Function) -> Vec<(Value, i64)> {
    let count = func.types.len();
    let mut consts = vec![None; count];
    let mut out: Vec<(Value, i64)> = (0..count as u32).map(|i| (Value(i), 0)).collect();
    let mut steps = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if let Op::Const(imm) = data.op {
            consts[res.index()] = Some(imm);
        } else if is_derived_i64(&data.op, func.ty(res)) {
            steps.push((data, res));
        }
    }
    for _ in 0..8 {
        let mut changed = false;
        for &(data, res) in &steps {
            if let Some(cand) = step_affine(&data.op, &out, &consts, &func.types)
                && cand != out[res.index()]
            {
                out[res.index()] = cand;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

impl Provenance {
    fn build(func: &Function, cfg: &Cfg, is_spill: Option<&[bool]>, affine: Option<Vec<(Value, i64)>>) -> Self {
        let slot_offsets = slot_addresses(func);
        let tracked = compute_tracked_roots(func, cfg);
        let esc = classify_uses(func, cfg, &slot_offsets, &tracked, is_spill);
        let mut param_roots = vec![MemRoot::Unknown; func.types.len()];
        for (i, t) in tracked.iter().enumerate() {
            if let TrackRoot::Set(root) = *t {
                param_roots[i] = root;
            }
        }
        let mut symbol_defs = vec![None; func.types.len()];
        for inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let data = &func.insts[inst.index()];
            if let (Op::Symbol(sym) | Op::SlotAddr(sym), Some(res)) = (&data.op, data.result) {
                symbol_defs[res.index()] = Some(*sym);
            }
        }
        let mut ro_bytes = vec![0_u32; func.types.len()];
        if let Some(entry) = cfg.entry {
            for (&param, abi) in func.blocks[entry.index()].params.iter().zip(&func.signature.params) {
                ro_bytes[param.index()] = abi.deref_bytes;
            }
        }
        let affine = affine.unwrap_or_else(|| compute_affine(func));
        let spans = loop_mem::compute_spans(func, cfg, &slot_offsets, &symbol_defs);
        Self {
            slot_offsets,
            unescaped_slots: esc.unescaped,
            call_readonly_slots: esc.call_readonly,
            noalias_escaped: esc.noalias,
            param_roots,
            symbol_defs,
            ro_bytes,
            affine,
            spans,
        }
    }

    pub(crate) fn new(func: &Function, cfg: &Cfg) -> Self {
        Self::build(func, cfg, None, None)
    }

    /// Like `new`, taking `affine` when the caller already holds `compute_affine` of this exact function.
    pub(crate) fn with_affine(func: &Function, cfg: &Cfg, affine: Option<Vec<(Value, i64)>>) -> Self {
        Self::build(func, cfg, None, affine)
    }

    pub(crate) fn compute(func: &Function) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, None, None)
    }

    pub(crate) fn compute_with_spills(func: &Function, is_spill: Option<&[bool]>) -> Self {
        let cfg = func.cfg();
        Self::build(func, &cfg, is_spill, None)
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
        let root = self.param_roots.get(eff_val.index()).copied().unwrap_or(MemRoot::Unknown);
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

    pub(crate) fn slot_unescaped(&self, slot: StackSlot) -> bool {
        self.unescaped_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn escaped_slots(&self) -> Vec<bool> {
        self.unescaped_slots.iter().map(|&u| !u).collect()
    }

    pub(crate) fn slot_call_readonly(&self, slot: StackSlot) -> bool {
        self.call_readonly_slots.get(slot.index()).copied().unwrap_or(false)
    }

    pub(crate) fn noalias_unescaped(&self, param: u16) -> bool {
        !self.noalias_escaped.get(usize::from(param)).copied().unwrap_or(true)
    }
}

pub(crate) fn may_alias(a: MemLoc, b: MemLoc, prov: &Provenance) -> bool {
    if a.size == 0 || b.size == 0 {
        return false;
    }
    if a.base == b.base {
        return a.overlaps(b);
    }
    if a.size != u32::MAX && b.size != u32::MAX {
        let (ba, alo, ahi) = prov.loc_span(a);
        let (bb, blo, bhi) = prov.loc_span(b);
        if ba == bb {
            return alo < bhi && blo < ahi;
        }
    }
    match (a.root, b.root) {
        (MemRoot::Slot(sa), MemRoot::Slot(sb)) => sa == sb,
        (MemRoot::NoAliasParam(pa), MemRoot::NoAliasParam(pb)) => pa == pb,
        (MemRoot::Global(ga), MemRoot::Global(gb)) => ga == gb,
        (MemRoot::NoAliasParam(_), MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_))
        | (MemRoot::Slot(_) | MemRoot::Global(_) | MemRoot::ReadonlyParam(_), MemRoot::NoAliasParam(_)) => false,
        (MemRoot::NoAliasParam(p), MemRoot::Unknown) | (MemRoot::Unknown, MemRoot::NoAliasParam(p)) => !prov.noalias_unescaped(p),
        (MemRoot::Slot(s), _) | (_, MemRoot::Slot(s)) => !prov.slot_unescaped(s),
        _ => true,
    }
}

fn call_writes_noalias(params: &[AbiParam], args: &[Value], prov: &Provenance, p: u16) -> bool {
    args.iter().enumerate().any(|(i, arg)| {
        prov.param_roots.get(arg.index()).copied() == Some(MemRoot::NoAliasParam(p))
            && !params.get(i).is_some_and(|pa| pa.readonly)
    })
}

/// Whether `loc` lies in a read-only global section (`func.rodata_bytes`).
pub(crate) fn is_rodata(func: &Function, loc: MemLoc) -> bool {
    matches!(loc.root, MemRoot::Global(sym) if func.rodata_bytes.iter().any(|&(s, _)| s == sym))
}

/// Whether `[off, off + bytes)` lies within `[0, size)` without underflow or overflow.
pub(crate) fn in_range(off: i64, bytes: u32, size: u32) -> bool {
    off >= 0 && (off as u64).checked_add(u64::from(bytes)).is_some_and(|end| end <= u64::from(size))
}

/// Whether `op` is a `StackStore` whose byte range in `slot` overlaps `[off, off + bytes)`.
pub(crate) fn stack_store_overlaps(func: &Function, op: &Op, slot: StackSlot, (off, bytes): (i32, u32)) -> Option<bool> {
    let Op::StackStore(val, wslot, woff) = *op else { return None };
    Some(wslot == slot && MemLoc::slot(slot, i64::from(off), bytes).overlaps(MemLoc::slot(wslot, i64::from(woff), func.ty(val).bytes())))
}

/// Whether `op` is an atomic memory op or fence.
pub(crate) fn is_atomic_barrier(op: &Op) -> bool {
    matches!(op, Op::AtomicLoad(_) | Op::AtomicStore(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fence)
}

/// Whether `op` prevents hoisting, sinking, or rematerializing memory accesses across it.
pub(crate) fn is_hard_barrier(op: &Op) -> bool {
    is_atomic_barrier(op)
        || matches!(op, Op::Call { .. } | Op::TryCall { .. } | Op::Trap | Op::Trapz(_) | Op::Assume(_)
            | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..))
}

fn stack_store_clobbers(func: &Function, op: &Op, (val, slot, off): (Value, StackSlot, i32), loc: MemLoc, prov: &Provenance) -> bool {
    if let MemBase::Slot(lslot) = loc.base && loc.root == MemRoot::Slot(lslot)
        && let Ok(loff) = i32::try_from(loc.offset) && let Some(hit) = stack_store_overlaps(func, op, lslot, (loff, loc.size))
    {
        return hit;
    }
    may_alias(MemLoc::slot(slot, i64::from(off), func.ty(val).bytes()), loc, prov)
}

pub(crate) fn clobbers(func: &Function, op: &Op, loc: MemLoc, prov: &Provenance) -> bool {
    match *op {
        Op::StackStore(val, slot, off) => stack_store_clobbers(func, op, (val, slot, off), loc, prov),
        Op::Store(val, addr, off) => may_alias(prov.loc(addr, i64::from(off), func.ty(val).bytes()), loc, prov),
        Op::Move(dst, _, bytes) => may_alias(prov.loc(dst, 0, bytes), loc, prov),
        Op::Copy(dst, _, _) => {
            let w = prov.loc(dst, 0, u32::MAX);
            w.base == loc.base || (w.root != MemRoot::Unknown && w.root == loc.root) || may_alias(w, loc, prov)
        }
        Op::Call { sig, ref args, .. } | Op::TryCall { sig, ref args, .. } => match loc.root {
            MemRoot::Slot(s) => !prov.slot_unescaped(s) && !prov.slot_call_readonly(s),
            MemRoot::NoAliasParam(p) => {
                !prov.noalias_unescaped(p) || call_writes_noalias(&func.sigs[sig.index()].params, args, prov, p)
            }
            _ => true,
        },
        _ => is_atomic_barrier(op),
    }
}

#[derive(Clone)]
pub(crate) struct KnownLoads {
    by_base: HashMap<MemBase, (MemRoot, Vec<(i64, Type, Value)>)>,
}

impl KnownLoads {
    pub(crate) fn new() -> Self {
        Self { by_base: HashMap::default() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.by_base.is_empty()
    }

    pub(crate) fn get(&self, base: MemBase, offset: i64, ty: Type) -> Option<Value> {
        self.by_base.get(&base)?.1.iter().find(|&&(off, t, _)| off == offset && t == ty).map(|&(_, _, val)| val)
    }

    pub(crate) fn entries_for(&self, base: MemBase) -> &[(i64, Type, Value)] {
        self.by_base.get(&base).map_or(&[], |entry| entry.1.as_slice())
    }

    pub(crate) fn remember(&mut self, loc: MemLoc, ty: Type, value: Value) {
        let (_, list) = self.by_base.entry(loc.base).or_insert_with(|| (loc.root, Vec::new()));
        if let Some(entry) = list.iter_mut().find(|(off, t, _)| *off == loc.offset && *t == ty) {
            entry.2 = value;
        } else {
            list.push((loc.offset, ty, value));
        }
    }

    pub(crate) fn write(&mut self, loc: MemLoc, ty: Type, value: Value, prov: &Provenance) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| !may_alias(loc, MemLoc::new(*root, base, off, t.bytes()), prov));
            !entries.is_empty()
        });
        self.remember(loc, ty, value);
    }

    pub(crate) fn invalidate(&mut self, func: &Function, op: &Op, prov: &Provenance) {
        self.by_base.retain(|&base, (root, entries)| {
            entries.retain(|&(off, t, _)| !clobbers(func, op, MemLoc::new(*root, base, off, t.bytes()), prov));
            !entries.is_empty()
        });
    }
}
