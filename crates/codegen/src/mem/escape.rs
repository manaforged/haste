use std::cell::OnceCell;

use crate::cfg::Cfg;
use crate::ir::{AbiParam, ArgumentPurpose, Binary, Block, Convert, Function, InstData, Op, StackSlot, Type, Value};

use super::{MemRoot, TrackRoot};

struct UseCounts {
    total: Vec<u32>,
    safe: Vec<u32>,
    ro_calls: Vec<u32>,
}

pub(super) struct SlotEscapes {
    pub(super) unescaped: Vec<bool>,
    pub(super) call_readonly: Vec<bool>,
    pub(super) noalias: Vec<bool>,
}

impl SlotEscapes {
    fn new(func: &Function) -> Self {
        Self {
            unescaped: vec![true; func.slots.len()],
            call_readonly: vec![true; func.slots.len()],
            noalias: vec![false; func.signature.params.len()],
        }
    }
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
    let mut non_ptr = vec![true; func.types.len()];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            let op = &func.insts[inst.index()].op;
            if !matches!(op, Op::Icmp(..) | Op::Trapz(_)) {
                op.for_operands(|v| non_ptr[v.index()] = false);
            }
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
    let in_loop: Vec<bool> = cfg.loops().depths.iter().map(|&d| d > 0).collect();
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
            if !danger[pred.index()] {
                danger[pred.index()] = true;
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
    (is_const, non_ptr, tracked): (&[bool], &[bool], &[TrackRoot]),
    (params, args, result, results): (&[AbiParam], &[Value], Option<Value>, &[Value]),
    (arg_idx, p): (usize, u16),
) -> bool {
    let Some(pa) = params.get(arg_idx) else { return false };
    if (!pa.readonly && !pa.noalias) || params.iter().any(|param| param.purpose == ArgumentPurpose::StructReturn) {
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

fn mark_safe(v: Value, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot], safe: &mut [u32]) {
    if slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_)) {
        safe[v.index()] += 1;
    }
}

fn mark_param_safe(v: Value, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot], safe: &mut [u32]) {
    if slots[v.index()].is_none()
        && matches!(tracked[v.index()], TrackRoot::Set(MemRoot::NoAliasParam(_) | MemRoot::ReadonlyParam(_)))
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
        let same_or_const = (slots[v.index()].is_some() || matches!(tracked[v.index()], TrackRoot::Set(_)))
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

fn classify_arith_select(data: &InstData, slots: &[Option<(StackSlot, i32)>], tracked: &[TrackRoot], safe: &mut [u32]) {
    let Some(res) = data.result else { return };
    match data.op {
        Op::Binary(Binary::Iadd | Binary::Isub, a, b) => {
            if slots[res.index()].is_some() {
                let target = if slots[a.index()].is_some() { a } else { b };
                safe[target.index()] += 1;
            } else if matches!(tracked[res.index()], TrackRoot::Set(_)) {
                let target = if matches!(tracked[a.index()], TrackRoot::Set(_)) { a } else { b };
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
    non_ptr: &'a OnceCell<Vec<bool>>,
    in_loop: &'a [bool],
    danger: &'a [bool],
    unknown_end: &'a [usize],
}

fn classify_call(
    cx: &EscapeCx<'_>,
    counts: &mut UseCounts,
    (b, pos): (Block, usize),
    (params, args): (&[AbiParam], &[Value]),
    allow_non_cap: Option<(Option<Value>, &[Value])>,
) {
    let post_safe = post_call_safe(cx.func, b, pos, cx.in_loop, cx.danger, cx.unknown_end);
    for (i, &arg) in args.iter().enumerate() {
        if params.get(i).is_some_and(|p| p.readonly) {
            counts.ro_calls[arg.index()] += 1;
        }
        if cx.slots[arg.index()].is_none()
            && let TrackRoot::Set(MemRoot::NoAliasParam(p)) = cx.tracked[arg.index()]
        {
            let ok = post_safe
                || allow_non_cap.is_some_and(|(result, results)| {
                    let non_ptr = cx.non_ptr.get_or_init(|| non_ptr_values(cx.func, cx.cfg));
                    non_capturing_call_arg(cx.func, (cx.is_const, non_ptr, cx.tracked), (params, args, result, results), (i, p))
                });
            if ok {
                counts.safe[arg.index()] += 1;
            }
        }
    }
}

fn classify_inst(cx: &EscapeCx<'_>, counts: &mut UseCounts, b: Block, pos: usize, data: &InstData) {
    match &data.op {
        Op::Load(addr, _) => mark_safe(*addr, cx.slots, cx.tracked, &mut counts.safe),
        Op::Trapz(addr) => mark_param_safe(*addr, cx.slots, cx.tracked, &mut counts.safe),
        Op::Store(val, addr, _) if val != addr => mark_safe(*addr, cx.slots, cx.tracked, &mut counts.safe),
        Op::Move(dst, src, _) | Op::Copy(dst, src, _) => {
            mark_param_safe(*dst, cx.slots, cx.tracked, &mut counts.safe);
            mark_param_safe(*src, cx.slots, cx.tracked, &mut counts.safe);
        }
        Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Convert(Convert::Bitcast, _) | Op::Select(..) => {
            classify_arith_select(data, cx.slots, cx.tracked, &mut counts.safe);
        }
        Op::Icmp(_, a, b) => classify_icmp((*a, *b), cx.slots, cx.tracked, cx.is_const, &mut counts.safe),
        Op::Jump(target, args) => count_safe_args(cx.func, *target, args, cx.tracked, &mut counts.safe),
        Op::Brif(_, then_b, then_args, else_b, else_args) => {
            count_safe_args(cx.func, *then_b, then_args, cx.tracked, &mut counts.safe);
            count_safe_args(cx.func, *else_b, else_args, cx.tracked, &mut counts.safe);
        }
        Op::Call { sig, args, results, .. } => {
            classify_call(cx, counts, (b, pos), (&cx.func.sigs[sig.index()].params, args), Some((data.result, results)));
        }
        Op::TryCall { sig, args, .. } => {
            classify_call(cx, counts, (b, pos), (&cx.func.sigs[sig.index()].params, args), None);
        }
        _ => {}
    }
}

pub(super) fn classify_uses(
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
    let cx = EscapeCx { func, cfg, slots, tracked, is_const: &is_const, non_ptr: &non_ptr, in_loop: &in_loop, danger: &danger, unknown_end: &unknown_end };
    let mut counts = UseCounts { total: vec![0_u32; n], safe: vec![0_u32; n], ro_calls: vec![0_u32; n] };
    for &b in &cfg.rpo {
        for (pos, &inst) in func.blocks[b.index()].insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| counts.total[v.index()] += 1);
            classify_inst(&cx, &mut counts, b, pos, data);
        }
    }
    finish_slot_escape(func, slots, tracked, &counts, is_spill)
}
