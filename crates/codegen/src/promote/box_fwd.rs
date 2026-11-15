use rustc_hash::FxHashSet as HashSet;
use std::sync::Arc;

use crate::cfg::Cfg;
use crate::copy::MOVE_INLINE;
use crate::fold::defs;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, SigRef, StackSlot, Value};
use crate::reloc::RuntimeSymbol;

use super::join_slots::resolve_addrs;
use super::subword::eval_consts;

enum WriteKind {
    Scalar { val: Value, off: i32 },
    Memset { sig: SigRef, byte: Value, len: Value, off: i64 },
}

struct SlotWrite {
    block: Block,
    pos: usize,
    inst: Inst,
    kind: WriteKind,
}

struct SlotCopy {
    block: Block,
    pos: usize,
    inst: Inst,
    dst: Value,
}

struct Collect {
    slot: StackSlot,
    size: u32,
    writes: Vec<SlotWrite>,
    copy: Option<SlotCopy>,
}

struct Plan {
    slot: StackSlot,
    writes: Vec<SlotWrite>,
    copy: SlotCopy,
}

struct Env {
    consts: Vec<Option<i64>>,
    addrs: Vec<Option<(StackSlot, i64)>>,
    addr_insts: Vec<(Inst, StackSlot)>,
    addr_slot: Vec<Option<StackSlot>>,
    defs: Vec<Option<Inst>>,
    cfg: Arc<Cfg>,
    rank: Vec<usize>,
}

impl Env {
    fn new(func: &Function) -> Self {
        let consts = eval_consts(func);
        let (addrs, addr_insts) = resolve_addrs(func, &consts);
        let mut addr_slot = vec![None; func.insts.len()];
        for &(inst, slot) in &addr_insts {
            addr_slot[inst.index()] = Some(slot);
        }
        let defs = defs(func);
        let cfg = func.cfg();
        let rank = Cfg::ranks(&cfg.rpo, func.blocks.len());
        Self { consts, addrs, addr_insts, addr_slot, defs, cfg, rank }
    }
}

pub(super) fn run(func: &mut Function) -> bool {
    let mut changed = false;
    while forward_one(func) {
        changed = true;
    }
    changed
}

fn forward_one(func: &mut Function) -> bool {
    let env = Env::new(func);
    for idx in 0..func.slots.len() {
        if func.slots[idx].size < MOVE_INLINE {
            continue;
        }
        let slot = StackSlot(idx as u32);
        let Some(plan) = find_plan(func, &env, slot) else { continue };
        apply_plan(func, &env, plan);
        return true;
    }
    false
}

fn find_plan(func: &Function, env: &Env, slot: StackSlot) -> Option<Plan> {
    let size = func.slots[slot.index()].size;
    let mut st = Collect { slot, size, writes: Vec::new(), copy: None };
    for (b_idx, block) in func.blocks.iter().enumerate() {
        let b = Block(b_idx as u32);
        for (pos, &inst) in block.insts.iter().enumerate() {
            if !visit_inst(func, env, (b, pos, inst), &mut st) {
                return None;
            }
        }
    }
    let copy = st.copy?;
    if st.writes.is_empty() || !env.cfg.reachable[copy.block.index()] {
        return None;
    }
    let ok = st.writes.iter().all(|w| {
        precedes(&env.cfg, w.block, w.pos, copy.block, copy.pos) && fits_dst(func, env, w, copy.dst)
    });
    if !ok || nonconst_values(env, &st.writes) > 4 {
        return None;
    }
    st.writes.sort_unstable_by_key(|w| (env.rank[w.block.index()], w.pos));
    Some(Plan { slot, writes: st.writes, copy })
}

fn nonconst_values(env: &Env, writes: &[SlotWrite]) -> usize {
    let mut vals = HashSet::default();
    for w in writes {
        if let WriteKind::Scalar { val, .. } = w.kind
            && env.consts[val.index()].is_none()
        {
            vals.insert(val);
        }
    }
    vals.len()
}

fn visit_inst(func: &Function, env: &Env, at: (Block, usize, Inst), st: &mut Collect) -> bool {
    let op = &func.insts[at.2.index()].op;
    if env.addr_slot[at.2.index()] == Some(st.slot) {
        return addr_inst_ok(op, env, st.slot);
    }
    match op {
        &Op::StackStore(val, s, off) if s == st.slot => push_scalar(func, env, at, (val, off), st),
        &Op::Store(val, ptr, off) if is_slot(env, ptr, st.slot) => push_ptr_store(func, env, at, (val, ptr, off), st),
        Op::Call { sig, args, .. } if is_memset(func, env, op, st.slot) => {
            push_memset(env, at, *sig, args, st)
        }
        Op::Move(..) | Op::Copy(..) => visit_move_or_copy(env, at, op, st),
        Op::StackAddr(s, _) | Op::StackLoad(s, _) if *s == st.slot => false,
        _ => !op_uses_slot(op, env, st.slot),
    }
}

fn visit_move_or_copy(env: &Env, at: (Block, usize, Inst), op: &Op, st: &mut Collect) -> bool {
    match *op {
        Op::Move(dst, src, bytes) if is_slot(env, src, st.slot) => {
            set_copy(env, at, dst, bytes == st.size && env.addrs[src.index()] == Some((st.slot, 0)), st)
        }
        Op::Copy(dst, src, len) if is_slot(env, src, st.slot) => {
            let exact = !is_slot(env, len, st.slot)
                && env.addrs[src.index()] == Some((st.slot, 0))
                && env.consts[len.index()] == Some(i64::from(st.size));
            set_copy(env, at, dst, exact, st)
        }
        _ => !op_uses_slot(op, env, st.slot),
    }
}

fn addr_inst_ok(op: &Op, env: &Env, slot: StackSlot) -> bool {
    match op {
        Op::Select(cond, _, _) => !is_slot(env, *cond, slot),
        Op::StackAddr(..) | Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Convert(Convert::Bitcast, _) => true,
        _ => false,
    }
}

fn is_slot(env: &Env, v: Value, slot: StackSlot) -> bool {
    env.addrs[v.index()].is_some_and(|(s, _)| s == slot)
}

fn op_uses_slot(op: &Op, env: &Env, slot: StackSlot) -> bool {
    let mut hit = false;
    op.for_operands(|v| hit |= is_slot(env, v, slot));
    hit
}

fn push_scalar(func: &Function, env: &Env, at: (Block, usize, Inst), (val, off): (Value, i32), st: &mut Collect) -> bool {
    if is_slot(env, val, st.slot) || off < 0 {
        return false;
    }
    let Some(end) = i64::from(off).checked_add(i64::from(func.ty(val).bytes())) else {
        return false;
    };
    if end > i64::from(st.size) {
        return false;
    }
    st.writes.push(SlotWrite { block: at.0, pos: at.1, inst: at.2, kind: WriteKind::Scalar { val, off } });
    true
}

fn push_ptr_store(
    func: &Function,
    env: &Env,
    at: (Block, usize, Inst),
    (val, ptr, off): (Value, Value, i32),
    st: &mut Collect,
) -> bool {
    let Some((_, base)) = env.addrs[ptr.index()] else { return false };
    let Some(total) = base.checked_add(i64::from(off)).and_then(|v| i32::try_from(v).ok()) else {
        return false;
    };
    push_scalar(func, env, at, (val, total), st)
}

fn is_memset(func: &Function, env: &Env, op: &Op, slot: StackSlot) -> bool {
    let Op::Call { callee, args, results, .. } = op else { return false };
    if args.len() != 3 || !results.is_empty() || !is_slot(env, args[0], slot) {
        return false;
    }
    let Some(def) = env.defs[callee.index()] else { return false };
    matches!(func.insts[def.index()].op, Op::Runtime(RuntimeSymbol::Memset))
}

fn push_memset(env: &Env, at: (Block, usize, Inst), sig: SigRef, args: &[Value], st: &mut Collect) -> bool {
    let (ptr, byte, len) = (args[0], args[1], args[2]);
    if is_slot(env, byte, st.slot) || is_slot(env, len, st.slot) {
        return false;
    }
    let Some((_, base)) = env.addrs[ptr.index()] else { return false };
    let Some(len_bytes) = env.consts[len.index()] else { return false };
    if base < 0 || len_bytes < 0 || base.checked_add(len_bytes).is_none_or(|end| end > i64::from(st.size)) {
        return false;
    }
    st.writes.push(SlotWrite { block: at.0, pos: at.1, inst: at.2, kind: WriteKind::Memset { sig, byte, len, off: base } });
    true
}

fn set_copy(env: &Env, at: (Block, usize, Inst), dst: Value, valid: bool, st: &mut Collect) -> bool {
    if !valid || is_slot(env, dst, st.slot) || st.copy.is_some() {
        return false;
    }
    st.copy = Some(SlotCopy { block: at.0, pos: at.1, inst: at.2, dst });
    true
}

fn precedes(cfg: &Cfg, wb: Block, wpos: usize, cb: Block, cpos: usize) -> bool {
    (wb == cb && wpos < cpos) || (wb != cb && cfg.dominates(wb, cb))
}

fn fits_dst(func: &Function, env: &Env, w: &SlotWrite, dst: Value) -> bool {
    let Some((d_slot, d_base)) = env.addrs[dst.index()] else {
        return true;
    };
    let (off, write_len) = match w.kind {
        WriteKind::Scalar { val, off } => (i64::from(off), i64::from(func.ty(val).bytes())),
        WriteKind::Memset { len, off, .. } => (off, env.consts[len.index()].unwrap_or(i64::MAX)),
    };
    let Some(total) = d_base.checked_add(off).and_then(|v| i32::try_from(v).ok()) else {
        return false;
    };
    let Some(end) = i64::from(total).checked_add(write_len) else {
        return false;
    };
    total >= 0 && end <= i64::from(func.slots[d_slot.index()].size)
}

fn apply_plan(func: &mut Function, env: &Env, plan: Plan) {
    let mut emitted = Vec::with_capacity(plan.writes.len() * 3);
    let mut removed = HashSet::default();
    let mut operand_seeds = Vec::new();
    func.insts[plan.copy.inst.index()].op.for_operands(|v| operand_seeds.push(v));
    for w in &plan.writes {
        removed.insert(w.inst);
        func.insts[w.inst.index()].op.for_operands(|v| operand_seeds.push(v));
        emit_write(func, env, w, plan.copy.dst, &mut emitted);
    }
    for b_idx in 0..func.blocks.len() {
        func.blocks[b_idx].insts.retain(|i| !removed.contains(i));
    }
    let c_insts = &mut func.blocks[plan.copy.block.index()].insts;
    if let Some(pos) = c_insts.iter().position(|&i| i == plan.copy.inst) {
        c_insts.splice(pos..=pos, emitted);
    }
    cleanup_dead(func, env, plan.slot, &operand_seeds);
}

fn emit_write(func: &mut Function, env: &Env, w: &SlotWrite, dst: Value, out: &mut Vec<Inst>) {
    match w.kind {
        WriteKind::Scalar { val, off } => {
            let v = remat_const(func, env, val, out);
            let op = match dst_slot_off(env, dst, i64::from(off)) {
                Some((d_slot, total)) => Op::StackStore(v, d_slot, total),
                None => Op::Store(v, dst, off),
            };
            out.push(func.add(op, None));
        }
        WriteKind::Memset { sig, byte, len, off } => {
            let ptr = emit_dst_ptr(func, env, dst, off, out);
            let b = remat_const(func, env, byte, out);
            let l = remat_const(func, env, len, out);
            let callee = func.value(crate::ir::Type::I64);
            out.push(func.add(Op::Runtime(RuntimeSymbol::Memset), Some(callee)));
            out.push(func.add(Op::Call { sig, callee, args: vec![ptr, b, l], results: Vec::new() }, None));
        }
    }
}

fn dst_slot_off(env: &Env, dst: Value, off: i64) -> Option<(StackSlot, i32)> {
    let (d_slot, d_base) = env.addrs[dst.index()]?;
    let total = d_base.checked_add(off).and_then(|v| i32::try_from(v).ok())?;
    Some((d_slot, total))
}

fn emit_dst_ptr(func: &mut Function, env: &Env, dst: Value, off: i64, out: &mut Vec<Inst>) -> Value {
    if off == 0 {
        return dst;
    }
    let ptr_ty = func.ty(dst);
    if let Some((d_slot, total)) = dst_slot_off(env, dst, off) {
        let ptr = func.value(ptr_ty);
        out.push(func.add(Op::StackAddr(d_slot, total), Some(ptr)));
        return ptr;
    }
    let off_v = func.value(ptr_ty);
    out.push(func.add(Op::Const(off), Some(off_v)));
    let ptr = func.value(ptr_ty);
    out.push(func.add(Op::Binary(Binary::Iadd, dst, off_v), Some(ptr)));
    ptr
}

fn remat_const(func: &mut Function, env: &Env, v: Value, out: &mut Vec<Inst>) -> Value {
    let Some(def) = env.defs[v.index()] else { return v };
    let Op::Const(imm) = func.insts[def.index()].op else { return v };
    let fresh = func.value(func.ty(v));
    out.push(func.add(Op::Const(imm), Some(fresh)));
    fresh
}

fn cleanup_dead(func: &mut Function, env: &Env, slot: StackSlot, seeds: &[Value]) {
    let mut cand = HashSet::default();
    let mut all_seeds = seeds.to_vec();
    for &(inst, s) in &env.addr_insts {
        if s == slot {
            cand.insert(inst);
            func.insts[inst.index()].op.for_operands(|v| all_seeds.push(v));
        }
    }
    for v in all_seeds {
        if let Some(def) = env.defs[v.index()]
            && matches!(func.insts[def.index()].op, Op::Const(_) | Op::Runtime(_))
        {
            cand.insert(def);
        }
    }
    while drop_unused(func, &cand) {}
}

fn drop_unused(func: &mut Function, cand: &HashSet<Inst>) -> bool {
    let uses = func.use_counts();
    let mut dropped = false;
    for b_idx in 0..func.blocks.len() {
        let before = func.blocks[b_idx].insts.len();
        func.blocks[b_idx].insts.retain(|inst| {
            !cand.contains(inst) || func.insts[inst.index()].result.is_none_or(|r| uses[r.index()] > 0)
        });
        dropped |= func.blocks[b_idx].insts.len() < before;
    }
    dropped
}
