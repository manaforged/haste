use crate::builder::FunctionBuilder;
use crate::ir::{Binary, Block, Function, Op, SlotData, StackSlot, Type, Value};
use crate::reloc::RuntimeSymbol;

use super::{Coroutine, Frame, Home, call, push};

/// The largest frame a first poll keeps on the native stack.
const MAX_STACK_FRAME: u32 = 4096;

/// A frame that starts in a native stack buffer and moves to the heap at the first suspension.
pub(super) struct Lazy {
    pub(super) buf: StackSlot,
    pub(super) bytes: u32,
    pub(super) may_be_heap: Vec<bool>,
}

fn slot_in_bounds(func: &Function, slot: StackSlot, off: i64, bytes: u32) -> Option<i32> {
    crate::mem::in_range(off, bytes, func.slots[slot.index()].size)
        .then(|| i32::try_from(off).ok())
        .flatten()
}

fn scan_holders(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
    holder_ok: &mut [bool],
    holder_target: &mut [Option<(StackSlot, i64)>],
) {
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::StackAddr(s, _) => holder_ok[s.index()] = false,
                Op::StackLoad(s, off) => {
                    if off != 0 || data.result.is_none_or(|r| func.ty(r) != Type::I64) {
                        holder_ok[s.index()] = false;
                    }
                }
                Op::StackStore(v, s, off) => {
                    let si = s.index();
                    if !holder_ok[si] {
                        continue;
                    }
                    if off != 0 || func.ty(v) != Type::I64 {
                        holder_ok[si] = false;
                    } else if let Some((target, toff)) = addrs[v.index()]
                        && target != s
                    {
                        match holder_target[si] {
                            None => holder_target[si] = Some((target, toff)),
                            Some(prev) if prev == (target, toff) => {}
                            Some(_) => holder_ok[si] = false,
                        }
                    } else {
                        holder_ok[si] = false;
                    }
                }
                _ => {}
            }
        }
    }
}

fn propagate_holder_addrs(
    func: &Function,
    addrs: &mut [Option<(StackSlot, i64)>],
    holder_ok: &[bool],
    holder_target: &[Option<(StackSlot, i64)>],
) -> bool {
    let mut changed = false;
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let (Op::StackLoad(s, 0), Some(res)) = (&data.op, data.result)
                && holder_ok[s.index()]
                && let Some(target) = holder_target[s.index()]
                && addrs[res.index()].is_none()
            {
                addrs[res.index()] = Some(target);
                changed = true;
            }
        }
    }
    if !changed {
        return false;
    }
    let consts = func.const_values();
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if addrs[res.index()].is_some() {
                continue;
            }
            if let Op::Binary(Binary::Iadd, x, y) = data.op {
                let hit = addrs[x.index()]
                    .zip(consts[y.index()])
                    .or_else(|| addrs[y.index()].zip(consts[x.index()]));
                if let Some(((s, base), c)) = hit
                    && let Some(sum) = base.checked_add(c)
                {
                    addrs[res.index()] = Some((s, sum));
                }
            }
        }
    }
    true
}

fn resolve_slot_addrs(func: &Function) -> (Vec<Option<(StackSlot, i64)>>, Vec<bool>) {
    let (mut addrs, _) = func.slot_offsets(|_| None);
    let nslots = func.slots.len();
    let mut holder_ok: Vec<bool> = (0..nslots)
        .map(|s| func.slots[s].size == 8 && !func.is_unwind_slot(StackSlot(s as u32)))
        .collect();
    let mut holder_target = vec![None; nslots];
    for _ in 0..4 {
        scan_holders(func, &addrs, &mut holder_ok, &mut holder_target);
        if !propagate_holder_addrs(func, &mut addrs, &holder_ok, &holder_target) {
            break;
        }
    }
    let is_holder = holder_ok
        .into_iter()
        .zip(holder_target)
        .map(|(ok, target)| ok && target.is_some())
        .collect();
    (addrs, is_holder)
}

fn try_fold_slot_access(
    func: &mut Function,
    inst_idx: usize,
    addrs: &[Option<(StackSlot, i64)>],
) -> bool {
    let data = &func.insts[inst_idx];
    match data.op {
        Op::Load(addr, off) => {
            if let (Some((slot, base)), Some(res)) = (addrs[addr.index()], data.result)
                && let Some(sum) = base.checked_add(i64::from(off))
                && let Some(total) = slot_in_bounds(func, slot, sum, func.ty(res).bytes())
            {
                func.insts[inst_idx].op = Op::StackLoad(slot, total);
                return true;
            }
        }
        Op::Store(val, addr, off) if val != addr => {
            if let Some((slot, base)) = addrs[addr.index()]
                && let Some(sum) = base.checked_add(i64::from(off))
                && let Some(total) = slot_in_bounds(func, slot, sum, func.ty(val).bytes())
            {
                func.insts[inst_idx].op = Op::StackStore(val, slot, total);
            }
        }
        _ => {}
    }
    false
}

fn rewrite_block_addrs(
    func: &mut Function,
    addrs: &[Option<(StackSlot, i64)>],
    is_holder: &[bool],
) {
    let mut local_addr_block = vec![None; func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let (Op::StackAddr(..), Some(res)) = (&data.op, data.result) {
                local_addr_block[res.index()] = Some(Block(b_idx as u32));
            }
        }
    }
    for b_idx in 0..func.blocks.len() {
        let cur_block = Block(b_idx as u32);
        let old = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut out = Vec::with_capacity(old.len());
        for inst in old {
            let ii = inst.index();
            if try_fold_slot_access(func, ii, addrs) {
                out.push(inst);
                continue;
            }
            let skip_remat = func.insts[ii]
                .result
                .is_some_and(|r| addrs.get(r.index()).copied().flatten().is_some())
                || matches!(func.insts[ii].op, Op::StackStore(_, s, 0) if is_holder[s.index()]);
            if !skip_remat {
                let mut remat: Vec<(Value, StackSlot, i32)> = Vec::new();
                func.insts[ii].op.for_operands(|v| {
                    if let Some((slot, base)) = addrs.get(v.index()).copied().flatten()
                        && local_addr_block[v.index()] != Some(cur_block)
                        && let Some(total) = slot_in_bounds(func, slot, base, 0)
                        && !remat.iter().any(|&(old_v, _, _)| old_v == v)
                    {
                        remat.push((v, slot, total));
                    }
                });
                if !remat.is_empty() {
                    let pairs: Vec<(Value, Value)> = remat
                        .into_iter()
                        .map(|(old_v, slot, total)| {
                            let fresh =
                                push(func, &mut out, Op::StackAddr(slot, total), Some(Type::I64))
                                    .expect("stack_addr has a result");
                            (old_v, fresh)
                        })
                        .collect();
                    func.insts[ii].op.map_operands(|v| {
                        pairs
                            .iter()
                            .find_map(|&(o, n)| (o == v).then_some(n))
                            .unwrap_or(v)
                    });
                }
            }
            out.push(inst);
        }
        func.blocks[b_idx].insts = out;
    }
}

fn sweep_dead_addrs(func: &mut Function, addrs: &[Option<(StackSlot, i64)>], is_holder: &[bool]) {
    let nslots = is_holder.len();
    for _ in 0..3 {
        let mut used = vec![false; func.types.len()];
        let mut holder_loaded = vec![false; nslots];
        for block in &func.blocks {
            for &inst in &block.insts {
                let op = &func.insts[inst.index()].op;
                op.for_operands(|v| used[v.index()] = true);
                if let Op::StackLoad(s, _) = *op {
                    holder_loaded[s.index()] = true;
                }
            }
        }
        let mut removed = false;
        for block in &mut func.blocks {
            block.insts.retain(|&inst| {
                let data = &func.insts[inst.index()];
                let dead_def = data.result.is_some_and(|r| !used[r.index()])
                    && (matches!(data.op, Op::StackAddr(..))
                        || matches!(data.op, Op::StackLoad(s, 0) if is_holder[s.index()])
                        || (matches!(data.op, Op::Binary(Binary::Iadd, _, _))
                            && data.result.is_some_and(|r| {
                                addrs.get(r.index()).copied().flatten().is_some()
                            })));
                let dead_store = matches!(
                    data.op,
                    Op::StackStore(_, s, 0) if is_holder[s.index()] && !holder_loaded[s.index()]
                );
                if dead_def || dead_store {
                    func.insts[inst.index()].op = Op::Trap;
                    removed = true;
                    false
                } else {
                    true
                }
            });
        }
        if !removed {
            break;
        }
    }
}

/// Folds loads/stores through constant-offset stack addresses and single-address holder slots into direct `StackLoad`/`StackStore`, and sinks cross-block `StackAddr` uses into the blocks that consume them.
pub(super) fn localize_slot_addrs(func: &mut Function) {
    if func.slots.is_empty() {
        return;
    }
    let (init_addrs, _) = func.slot_offsets(|_| None);
    let no_holders = vec![false; func.slots.len()];
    for b_idx in 0..func.blocks.len() {
        let insts = func.blocks[b_idx].insts.clone();
        for inst in insts {
            try_fold_slot_access(func, inst.index(), &init_addrs);
        }
    }
    sweep_dead_addrs(func, &init_addrs, &no_holders);
    let (addrs, is_holder) = resolve_slot_addrs(func);
    rewrite_block_addrs(func, &addrs, &is_holder);
    sweep_dead_addrs(func, &addrs, &is_holder);
}

/// Adds the stack buffer when no address into the frame leaves the function or crosses a suspension.
pub(super) fn plan(
    func: &Function,
    spec: &Coroutine,
    frame: &Frame,
    (is_spill, stack): (&[bool], &mut Vec<SlotData>),
) -> Option<Lazy> {
    let bytes = frame.bytes();
    if bytes > MAX_STACK_FRAME {
        return None;
    }
    let framed = |slot: StackSlot| matches!(frame.slots[slot.index()], Home::Frame(_));
    let addressed =
        func.blocks.iter().flat_map(|block| &block.insts).any(
            |inst| matches!(func.insts[inst.index()].op, Op::StackAddr(slot, _) if framed(slot)),
        );
    if addressed && frame_escapes(func, &framed, is_spill) {
        return None;
    }
    let mut may_be_heap = vec![false; func.blocks.len()];
    let mut work: Vec<Block> = spec.resumes.clone();
    for &r in &spec.resumes {
        may_be_heap[r.index()] = true;
    }
    while let Some(b) = work.pop() {
        for succ in func.successors(b) {
            if !std::mem::replace(&mut may_be_heap[succ.index()], true) {
                work.push(succ);
            }
        }
    }
    stack[0].size = 24;
    stack.push(SlotData {
        size: bytes,
        align_shift: 4,
        poll_local: true,
    });
    Some(Lazy {
        buf: StackSlot(stack.len() as u32 - 1),
        bytes,
        may_be_heap,
    })
}

/// Before each return, moves a frame still in the stack buffer to the heap when suspending, and frees a heap frame when it ends.
pub(super) fn settle(
    func: &mut Function,
    spec: &Coroutine,
    anchor: StackSlot,
    lazy: &Lazy,
    blocks: usize,
) {
    for idx in 0..blocks {
        let block = Block(idx as u32);
        let done = spec.done.contains(&block);
        if done && !lazy.may_be_heap[idx] {
            continue;
        }
        let Some(&last) = func.blocks[idx].insts.last() else {
            continue;
        };
        let op = &func.insts[last.index()].op;
        if !(op.is_terminator() && (done || matches!(op, Op::Return(_)))) {
            continue;
        }
        let term = func.blocks[idx]
            .insts
            .pop()
            .expect("a block with a terminator");
        let mut b = FunctionBuilder::new(func);
        let (cold, tail) = (b.create_block(), b.create_block());
        b.switch_to_block(block);
        let on_stack = b.ins().stack_load(Type::I8, anchor, 16);
        if done {
            b.ins().brif(on_stack, tail, &[], cold, &[]);
        } else {
            b.ins().brif(on_stack, cold, &[], tail, &[]);
        }
        let mut insts = Vec::new();
        let size = push(
            b.func,
            &mut insts,
            Op::Const(i64::from(lazy.bytes)),
            Some(Type::I64),
        )
        .expect("a constant has a result");
        if done {
            let fp = push(
                b.func,
                &mut insts,
                Op::StackLoad(anchor, 0),
                Some(Type::I64),
            )
            .expect("a load has a result");
            b.func.mark_frame_ptr(fp);
            call(b.func, &mut insts, RuntimeSymbol::Free, &[fp, size], false);
        } else {
            let buf = push(
                b.func,
                &mut insts,
                Op::StackAddr(lazy.buf, 0),
                Some(Type::I64),
            )
            .expect("stack_addr has a result");
            b.func.mark_frame_ptr(buf);
            let heap = call(b.func, &mut insts, RuntimeSymbol::Malloc, &[size], true)
                .expect("malloc returns");
            b.func.mark_frame_ptr(heap);
            push(b.func, &mut insts, Op::Move(heap, buf, lazy.bytes), None);
            let object = push(
                b.func,
                &mut insts,
                Op::StackLoad(anchor, 8),
                Some(Type::I64),
            )
            .expect("a load has a result");
            push(
                b.func,
                &mut insts,
                Op::Store(heap, object, spec.header.1),
                None,
            );
        }
        b.func.blocks[cold.index()].insts = insts;
        b.switch_to_block(cold);
        b.ins().jump(tail, &[]);
        b.func.blocks[tail.index()].insts.push(term);
    }
}

/// Whether an address into a framed slot, followed through local slots that hold it, reaches anything but a load, a store address, a copy or a local slot, or lives across a suspension.
fn frame_escapes(func: &Function, framed: &dyn Fn(StackSlot) -> bool, is_spill: &[bool]) -> bool {
    let (addrs, _) = func.slot_offsets(|_| None);
    let slot_of = |v: Value| {
        addrs
            .get(v.index())
            .copied()
            .flatten()
            .map(|(slot, _)| slot)
    };
    let mut tainted: Vec<bool> = (0..func.types.len())
        .map(|v| slot_of(Value(v as u32)).is_some_and(framed))
        .collect();
    let mut holders = vec![false; func.slots.len()];
    let insts: Vec<_> = func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .map(|inst| &func.insts[inst.index()])
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for data in &insts {
            let held = match data.op {
                Op::StackStore(value, slot, _) if tainted[value.index()] => Some(slot),
                Op::Store(value, addr, _) if tainted[value.index()] => slot_of(addr),
                _ => None,
            };
            if let Some(slot) = held.filter(|&slot| !framed(slot) && !holders[slot.index()]) {
                holders[slot.index()] = true;
                changed = true;
            }
            let from = match data.op {
                Op::StackLoad(slot, _) => holders[slot.index()],
                Op::Load(addr, _) => slot_of(addr).is_some_and(|slot| holders[slot.index()]),
                Op::Binary(Binary::Iadd | Binary::Isub, a, _) => tainted[a.index()],
                _ => false,
            };
            if let Some(result) = data.result.filter(|r| from && !tainted[r.index()]) {
                tainted[result.index()] = true;
                changed = true;
            }
        }
    }
    let local = |v: Value| slot_of(v).is_some_and(|slot| !framed(slot));
    insts.iter().any(|data| {
        let mut bad = false;
        data.op.for_operands(|v| {
            let holder_addr = slot_of(v).is_some_and(|slot| holders[slot.index()]);
            if !tainted[v.index()] && !holder_addr {
                return;
            }
            bad |= match &data.op {
                Op::Load(..) => false,
                Op::Move(..) => holder_addr,
                Op::Copy(_, _, len) => holder_addr || *len == v,
                Op::Store(value, addr, _) => *value == v && (holder_addr || !local(*addr)),
                Op::StackStore(_, slot, _) => holder_addr || framed(*slot),
                Op::Binary(Binary::Iadd | Binary::Isub, a, b) => {
                    *a != v
                        || tainted[b.index()]
                        || (holder_addr && data.result.and_then(|r| slot_of(r)).is_none())
                }
                _ => true,
            };
        });
        bad || data.result.is_some_and(|r| {
            tainted[r.index()] && is_spill.get(r.index()).copied().unwrap_or(false)
        })
    })
}
