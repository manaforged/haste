use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::ir::{
    AbiParam, ArgumentPurpose, Binary, Block, Convert, Function, InstData, Op, StackSlot, Value,
};

const FRONTIER_WORDS: usize = 1 << 20;

use super::{MemRoot, TrackRoot};

struct UseCounts {
    total: Vec<u32>,
    safe: Vec<u32>,
    unwind_stores: Vec<u32>,
    ro_calls: Vec<u32>,
    returns: Vec<u32>,
}

pub(super) struct SlotEscapes {
    pub(super) unescaped: Vec<bool>,
    pub(super) addr_unescaped: Vec<bool>,
    pub(super) call_readonly: Vec<bool>,
    pub(super) return_only: Vec<bool>,
    pub(super) noalias: Vec<bool>,
    pub(super) spilled: Vec<bool>,
    pub(super) points: HashMap<u32, Vec<(u32, u32)>>,
}

/// Where an escaped slot's address leaves the function: blocks entered after an escape, and the escaping instructions.
pub(crate) struct Frontier {
    seen: Vec<u64>,
    points: Vec<(u32, u32)>,
}

impl Frontier {
    /// Whether the slot may have escaped by the instruction at `pos` in `block`, counting that instruction.
    pub(crate) fn reached(&self, (block, pos): (u32, u32)) -> bool {
        self.seen[block as usize / 64] >> (block % 64) & 1 == 1
            || self.points.iter().any(|&(b, p)| b == block && p <= pos)
    }
}

impl SlotEscapes {
    fn new(func: &Function) -> Self {
        Self {
            unescaped: vec![true; func.slots.len()],
            addr_unescaped: vec![true; func.slots.len()],
            call_readonly: vec![true; func.slots.len()],
            return_only: vec![true; func.slots.len()],
            noalias: vec![false; func.signature.params.len()],
            spilled: vec![false; func.slots.len()],
            points: HashMap::default(),
        }
    }
}

fn slot_root(
    v: Value,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
) -> Option<StackSlot> {
    slots[v.index()]
        .map(|(s, _)| s)
        .or(match tracked[v.index()] {
            TrackRoot::Set(MemRoot::Slot(s)) => Some(s),
            _ => None,
        })
}

fn finish_slot_escape(
    func: &Function,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    counts: &UseCounts,
    is_spill: Option<&[bool]>,
) -> SlotEscapes {
    let mut esc = SlotEscapes::new(func);
    for v in 0..slots.len() {
        if let Some(s) = slot_root(Value(v as u32), slots, tracked) {
            let spilled = is_spill.is_some_and(|sp| sp.get(v).copied().unwrap_or(false));
            esc.spilled[s.index()] |= spilled;
            if spilled || counts.safe[v] != counts.total[v] {
                esc.unescaped[s.index()] = false;
            }
            let uw = counts.unwind_stores.get(v).copied().unwrap_or(0);
            if spilled || counts.safe[v] + uw != counts.total[v] {
                esc.addr_unescaped[s.index()] = false;
            }
            let ro = counts.ro_calls.get(v).copied().unwrap_or(0);
            if spilled || counts.safe[v] + ro != counts.total[v] {
                esc.call_readonly[s.index()] = false;
            }
            let ret = counts.returns.get(v).copied().unwrap_or(0);
            if spilled || counts.safe[v] + ret != counts.total[v] {
                esc.return_only[s.index()] = false;
            }
        }
        if let TrackRoot::Set(MemRoot::NoAliasParam(p)) = tracked[v]
            && counts.safe[v] != counts.total[v]
        {
            esc.noalias[usize::from(p)] = true;
        }
    }
    for u in &func.unwind_drops {
        for s in [u.local, u.flag] {
            esc.unescaped[s.index()] = false;
            esc.call_readonly[s.index()] = false;
            esc.return_only[s.index()] = false;
        }
    }
    esc
}

fn count_safe_args(
    func: &Function,
    target: Block,
    args: &[Value],
    tracked: &[TrackRoot],
    safe: &mut [u32],
) {
    for (&param, &arg) in func.blocks[target.index()].params.iter().zip(args) {
        if let TrackRoot::Set(root) = tracked[param.index()]
            && tracked[arg.index()] == TrackRoot::Set(root)
        {
            safe[arg.index()] += 1;
        }
    }
}

fn is_unknown_mem_op(op: &Op, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot]) -> bool {
    let known = |v: Value| {
        slots[v.index()].is_some()
            || matches!(tracked[v.index()], TrackRoot::Set(root) if root.is_identified())
    };
    match *op {
        Op::Load(addr, _) | Op::Store(_, addr, _) => !known(addr),
        Op::Move(dst, src, _) | Op::Copy(dst, src, _) => !known(dst) || !known(src),
        Op::Call { .. } | Op::TryCall { .. } => true,
        _ => super::is_atomic_barrier(op),
    }
}

fn compute_block_danger(
    func: &Function,
    cfg: &Cfg,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
) -> (Vec<bool>, Vec<bool>, Vec<usize>) {
    let n = func.blocks.len();
    let in_loop: Vec<bool> = cfg.loops().depths.iter().map(|&d| d > 0).collect();
    let mut danger = vec![false; n];
    let mut unknown_end = vec![0_usize; n];
    let mut queue = Vec::new();
    for &b in &cfg.rpo {
        let bi = b.index();
        unknown_end[bi] = func.blocks[bi]
            .insts
            .iter()
            .rposition(|&inst| is_unknown_mem_op(&func.insts[inst.index()].op, slots, tracked))
            .map_or(0, |pos| pos + 1);
        if in_loop[bi] || unknown_end[bi] > 0 {
            danger[bi] = true;
            queue.push(b);
        }
    }
    while let Some(b) = queue.pop() {
        for &pred in cfg.preds_of(b) {
            if !danger[pred.index()] {
                danger[pred.index()] = true;
                queue.push(pred);
            }
        }
    }
    (in_loop, danger, unknown_end)
}

fn post_call_safe(
    func: &Function,
    block: Block,
    pos: usize,
    in_loop: &[bool],
    danger: &[bool],
    unknown_end: &[usize],
) -> bool {
    if in_loop[block.index()] || unknown_end[block.index()] > pos + 1 {
        return false;
    }
    let Some(term) = func.blocks[block.index()].insts.last() else {
        return false;
    };
    !func.insts[term.index()]
        .op
        .targets()
        .any(|t| danger[t.index()])
}

fn is_rooted(v: Value, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot]) -> bool {
    slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_))
}

fn mark_safe(
    v: Value,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    safe: &mut [u32],
) {
    if is_rooted(v, slots, tracked) {
        safe[v.index()] += 1;
    }
}

fn mark_param_safe(
    v: Value,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    safe: &mut [u32],
) {
    if slots[v.index()].is_none()
        && matches!(
            tracked[v.index()],
            TrackRoot::Set(MemRoot::NoAliasParam(_) | MemRoot::ReadonlyParam(_))
        )
    {
        safe[v.index()] += 1;
    }
}

fn classify_icmp(
    (a, b): (Value, Value),
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    is_const: &[bool],
    safe: &mut [u32],
) {
    for (v, other) in [(a, b), (b, a)] {
        let is_noalias = matches!(tracked[v.index()], TrackRoot::Set(MemRoot::NoAliasParam(_)));
        let same_or_const = is_rooted(v, slots, tracked)
            && (is_const[other.index()]
                || matches!((tracked[v.index()], tracked[other.index()]), (TrackRoot::Set(ra), TrackRoot::Set(rb)) if ra == rb));
        if is_noalias || same_or_const {
            safe[v.index()] += 1;
        }
    }
}

fn classify_select((a, b): (Value, Value), res: Value, tracked: &[TrackRoot], safe: &mut [u32]) {
    if let TrackRoot::Set(root) = tracked[res.index()] {
        for v in [a, b] {
            if tracked[v.index()] == TrackRoot::Set(root) {
                safe[v.index()] += 1;
            }
        }
    }
}

fn classify_arith_select(
    data: &InstData,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    safe: &mut [u32],
) {
    let Some(res) = data.result else { return };
    match data.op {
        Op::Binary(Binary::Iadd | Binary::Isub, a, b) => {
            if slots[res.index()].is_some() {
                let target = if slots[a.index()].is_some() { a } else { b };
                safe[target.index()] += 1;
            } else if matches!(tracked[res.index()], TrackRoot::Set(_)) {
                let target = if matches!(tracked[a.index()], TrackRoot::Set(_)) {
                    a
                } else {
                    b
                };
                safe[target.index()] += 1;
            }
        }
        Op::Convert(Convert::Bitcast, x) if matches!(tracked[res.index()], TrackRoot::Set(_)) => {
            safe[x.index()] += 1;
        }
        Op::Select(_, a, b) => classify_select((a, b), res, tracked, safe),
        _ => {}
    }
}

struct EscapeCx<'a> {
    func: &'a Function,
    cfg: &'a Cfg,
    slots: &'a [Option<(StackSlot, i32)>],
    tracked: &'a [TrackRoot],
    is_const: &'a [bool],
    unwind_write_only: &'a [bool],
    has_noalias: bool,
    danger: &'a std::cell::OnceCell<(Vec<bool>, Vec<bool>, Vec<usize>)>,
}

fn classify_call(
    cx: &EscapeCx<'_>,
    counts: &mut UseCounts,
    (b, pos): (Block, usize),
    (params, args): (&[AbiParam], &[Value]),
) {
    let mut post_safe: Option<bool> = None;
    for (i, &arg) in args.iter().enumerate() {
        if params.get(i).is_some_and(|p| p.readonly) {
            if counts.ro_calls.is_empty() {
                counts.ro_calls.resize(counts.total.len(), 0);
            }
            counts.ro_calls[arg.index()] += 1;
        }
        if params
            .get(i)
            .is_some_and(|p| p.purpose == ArgumentPurpose::StructReturn)
        {
            if counts.returns.is_empty() {
                counts.returns.resize(counts.total.len(), 0);
            }
            counts.returns[arg.index()] += 1;
        }
        if cx.has_noalias
            && cx.slots[arg.index()].is_none()
            && matches!(
                cx.tracked[arg.index()],
                TrackRoot::Set(MemRoot::NoAliasParam(_))
            )
        {
            let safe = *post_safe.get_or_insert_with(|| {
                let (in_loop, danger, unknown_end) = cx
                    .danger
                    .get_or_init(|| compute_block_danger(cx.func, cx.cfg, cx.slots, cx.tracked));
                post_call_safe(cx.func, b, pos, in_loop, danger, unknown_end)
            });
            if safe {
                counts.safe[arg.index()] += 1;
            }
        }
    }
}

fn classify_inst(cx: &EscapeCx<'_>, counts: &mut UseCounts, b: Block, pos: usize, data: &InstData) {
    match &data.op {
        Op::Load(addr, _) => mark_safe(*addr, cx.slots, cx.tracked, &mut counts.safe),
        Op::Trapz(addr) => mark_param_safe(*addr, cx.slots, cx.tracked, &mut counts.safe),
        Op::Store(val, addr, _) if val != addr => {
            mark_safe(*addr, cx.slots, cx.tracked, &mut counts.safe)
        }
        Op::StackStore(val, slot, 0) if cx.unwind_write_only.get(slot.index()) == Some(&true) => {
            mark_safe(*val, cx.slots, cx.tracked, &mut counts.unwind_stores);
        }
        Op::Move(dst, src, _) | Op::Copy(dst, src, _) => {
            mark_param_safe(*dst, cx.slots, cx.tracked, &mut counts.safe);
            mark_param_safe(*src, cx.slots, cx.tracked, &mut counts.safe);
        }
        Op::Binary(Binary::Iadd | Binary::Isub, ..)
        | Op::Convert(Convert::Bitcast, _)
        | Op::Select(..) => {
            classify_arith_select(data, cx.slots, cx.tracked, &mut counts.safe);
        }
        Op::Icmp(_, a, b) => classify_icmp(
            (*a, *b),
            cx.slots,
            cx.tracked,
            cx.is_const,
            &mut counts.safe,
        ),
        Op::Jump(target, args) => {
            count_safe_args(cx.func, *target, args, cx.tracked, &mut counts.safe)
        }
        Op::Brif(_, then_b, then_args, else_b, else_args) => {
            count_safe_args(cx.func, *then_b, then_args, cx.tracked, &mut counts.safe);
            count_safe_args(cx.func, *else_b, else_args, cx.tracked, &mut counts.safe);
        }
        Op::Call { sig, args, .. } | Op::TryCall { sig, args, .. } => {
            classify_call(
                cx,
                counts,
                (b, pos),
                (&cx.func.sigs[sig.index()].params, args),
            );
        }
        _ => {}
    }
}

pub(super) fn classify_uses(
    func: &Function,
    cfg: &Cfg,
    slots: &[Option<(StackSlot, i32)>],
    tracked: &[TrackRoot],
    (is_spill, positions): (Option<&[bool]>, bool),
) -> SlotEscapes {
    let has_noalias = func.signature.params.iter().any(|abi| abi.noalias);
    if func.slots.is_empty() && !has_noalias {
        return SlotEscapes::new(func);
    }
    let any_target = slots.iter().any(Option::is_some)
        || tracked.iter().any(|t| {
            matches!(
                t,
                TrackRoot::Set(MemRoot::Slot(_) | MemRoot::NoAliasParam(_))
            )
        });
    if !any_target {
        let mut esc = SlotEscapes::new(func);
        for u in &func.unwind_drops {
            for s in [u.local, u.flag] {
                esc.unescaped[s.index()] = false;
                esc.call_readonly[s.index()] = false;
                esc.return_only[s.index()] = false;
            }
        }
        return esc;
    }
    let n = func.types.len();
    let has_indirect_unwind = func.unwind_drops.iter().any(|u| u.indirect);
    let mut unwind_write_only = if has_indirect_unwind {
        let mut v = vec![false; func.slots.len()];
        for u in &func.unwind_drops {
            if u.indirect {
                v[u.local.index()] = true;
            }
        }
        v
    } else {
        Vec::new()
    };
    let mut is_const = vec![false; n];
    if has_indirect_unwind {
        for &b in &cfg.rpo {
            for &inst in &func.blocks[b.index()].insts {
                match func.insts[inst.index()].op {
                    Op::StackLoad(slot, _) | Op::StackAddr(slot, _) => {
                        unwind_write_only[slot.index()] = false
                    }
                    _ => {}
                }
            }
        }
    }
    let mut counts = UseCounts {
        total: vec![0_u32; n],
        safe: vec![0_u32; n],
        unwind_stores: if has_indirect_unwind {
            vec![0_u32; n]
        } else {
            Vec::new()
        },
        ro_calls: Vec::new(),
        returns: Vec::new(),
    };
    let danger = std::cell::OnceCell::new();
    let mut points: HashMap<u32, Vec<(u32, u32)>> = HashMap::default();
    let mut used: Vec<(Value, StackSlot, u32)> = Vec::new();
    for &b in &cfg.rpo {
        for (pos, &inst) in func.blocks[b.index()].insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::Const(_) => {
                    if let Some(res) = data.result {
                        is_const[res.index()] = true;
                    }
                    continue;
                }
                Op::StackLoad(..)
                | Op::StackAddr(..)
                | Op::Symbol(_)
                | Op::SlotAddr(_)
                | Op::Runtime(_)
                | Op::Trap => continue,
                Op::Load(addr, _) => {
                    if is_rooted(addr, slots, tracked) {
                        counts.total[addr.index()] += 1;
                        counts.safe[addr.index()] += 1;
                    }
                    continue;
                }
                Op::Store(val, addr, _) if !is_rooted(val, slots, tracked) => {
                    if is_rooted(addr, slots, tracked) {
                        counts.total[addr.index()] += 1;
                        counts.safe[addr.index()] += 1;
                    }
                    continue;
                }
                Op::StackStore(val, ..) if !is_rooted(val, slots, tracked) => continue,
                _ => {}
            }
            if positions {
                used.clear();
            }
            let mut has_tracked = false;
            data.op.for_operands(|v| {
                let vi = v.index();
                if slots[vi].is_some() || matches!(tracked[vi], TrackRoot::Set(_)) {
                    has_tracked = true;
                    counts.total[vi] += 1;
                    if positions && let Some(s) = slot_root(v, slots, tracked) {
                        let uw = if has_indirect_unwind {
                            counts.unwind_stores[vi]
                        } else {
                            0
                        };
                        used.push((v, s, counts.safe[vi] + uw));
                    }
                }
            });
            if !has_tracked {
                continue;
            }
            let cx = EscapeCx {
                func,
                cfg,
                slots,
                tracked,
                is_const: &is_const,
                unwind_write_only: &unwind_write_only,
                has_noalias,
                danger: &danger,
            };
            classify_inst(&cx, &mut counts, b, pos, data);
            if positions && !used.is_empty() {
                for (i, &(v, s, before)) in used.iter().enumerate() {
                    if used[..i].iter().any(|&(u, _, _)| u == v) {
                        continue;
                    }
                    let uses = used[i..].iter().filter(|&&(u, _, _)| u == v).count() as u32;
                    let after = counts.safe[v.index()]
                        + if has_indirect_unwind {
                            counts.unwind_stores[v.index()]
                        } else {
                            0
                        };
                    if after - before < uses {
                        points
                            .entry(s.0)
                            .or_default()
                            .push((b.index() as u32, pos as u32));
                    }
                }
            }
        }
    }
    let mut esc = finish_slot_escape(func, slots, tracked, &counts, is_spill);
    esc.points = points;
    esc
}

/// The escape frontier of every escaped, unspilled slot, omitted when any call may already see it.
pub(super) fn frontiers(func: &Function, esc: &SlotEscapes) -> HashMap<u32, Frontier> {
    let words = func.blocks.len().div_ceil(64);
    let wanted = |s: u32, pts: &Vec<(u32, u32)>| {
        let idx = s as usize;
        !esc.unescaped[idx] && !esc.spilled[idx] && !pts.is_empty()
    };
    let active_count = esc
        .points
        .iter()
        .filter(|&(&s, pts)| wanted(s, pts))
        .count();
    if active_count == 0 || active_count * words > FRONTIER_WORDS {
        return HashMap::default();
    }
    let mut queue = Vec::new();
    let mut out = HashMap::default();
    for (&s, pts) in &esc.points {
        if !wanted(s, pts) {
            continue;
        }
        let mut seen = vec![0_u64; words];
        queue.extend(pts.iter().map(|&(b, _)| Block(b)));
        while let Some(b) = queue.pop() {
            let Some(&term) = func.blocks[b.index()].insts.last() else {
                continue;
            };
            for t in func.insts[term.index()].op.targets() {
                let (w, bit) = (t.index() / 64, 1_u64 << (t.index() % 64));
                if seen[w] & bit == 0 {
                    seen[w] |= bit;
                    queue.push(t);
                }
            }
        }
        out.insert(
            s,
            Frontier {
                seen,
                points: pts.clone(),
            },
        );
    }
    out
}

pub(super) fn finish_fresh_owners(
    func: &Function,
    cfg: &Cfg,
    entry: &[Value],
    affine: &[(Value, i64)],
    (frees, called): (&[Value], &[(Value, bool, Block, usize)]),
    tracked: &mut [TrackRoot],
) -> bool {
    let mut freed_in_non_ret: Vec<Value> = Vec::new();
    if !frees.is_empty() {
        for &b in &cfg.rpo {
            let insts = &func.blocks[b.index()].insts;
            if insts
                .last()
                .is_some_and(|&i| matches!(func.insts[i.index()].op, Op::Return(_)))
            {
                continue;
            }
            for &inst in insts {
                if let Op::Call { callee, args, .. } = &func.insts[inst.index()].op
                    && frees.contains(callee)
                    && let Some(&ptr) = args.first()
                    && !freed_in_non_ret.contains(&ptr)
                {
                    freed_in_non_ret.push(ptr);
                }
            }
        }
    }
    let mut updated = false;
    for &b in &cfg.rpo {
        let insts = &func.blocks[b.index()].insts;
        for (spos, &inst) in insts.iter().enumerate() {
            if let Op::Store(value, addr, _) = func.insts[inst.index()].op
                && let Some(&(_, is_m, cb, cpos)) = called.iter().find(|&&(v, ..)| v == value)
                && let Some(&(owner, _)) = affine.get(addr.index())
                && owner != value
                && (entry.contains(&owner)
                    || (is_m
                        && !freed_in_non_ret.contains(&owner)
                        && (func.is_frame_ptr(owner)
                            || (cb == b
                                && spos > cpos
                                && !insts[cpos..spos].iter().any(|&i| {
                                    crate::live::results(&func.insts[i.index()]).any(|r| r == owner)
                                })))))
            {
                let next = TrackRoot::Set(MemRoot::UniqueHeap(Some(owner)));
                if tracked[value.index()] != next {
                    tracked[value.index()] = next;
                    updated = true;
                }
            }
        }
    }
    updated
}

#[inline]
pub(super) fn inst_escapes_caller_loc(op: &Op, tracked: &[TrackRoot]) -> bool {
    let is_cl = |v: Value| tracked[v.index()] == TrackRoot::Set(MemRoot::CallerLoc);
    match *op {
        Op::Load(addr, 0) if is_cl(addr) => false,
        Op::Store(val, addr, 0) if is_cl(addr) && !is_cl(val) => false,
        Op::Convert(crate::ir::Convert::Bitcast, _) | Op::Jump(..) | Op::Brif(..) => false,
        _ => {
            let mut bad = false;
            op.for_operands(|v| bad |= is_cl(v));
            bad
        }
    }
}

pub(super) fn caller_loc_escaped(func: &Function, cfg: &Cfg, tracked: &[TrackRoot]) -> bool {
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            if inst_escapes_caller_loc(&func.insts[inst.index()].op, tracked) {
                return true;
            }
        }
    }
    false
}
