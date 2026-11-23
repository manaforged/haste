use crate::ir::{Function, Inst, Op, Value};

use super::Emitter;

/// Returns `(frame_offset, width_bytes)` when `inst` stores constant integer zero into a stack slot.
pub(crate) fn stack_zero_store(func: &Function, slots: &[i32], inst: Inst, const_of: impl FnOnce(Value) -> Option<i64>) -> Option<(i32, u32)> {
    let Op::StackStore(val, slot, off) = func.insts[inst.index()].op else {
        return None;
    };
    let ty = func.ty(val);
    (ty.is_small_int() && const_of(val) == Some(0)).then(|| (slots[slot.index()] + off, ty.bytes()))
}

fn emit_zero_interval<'a, E: Emitter<'a>>(emit: &mut E, start: i32, stop: i32) {
    let mut cur = start;
    while stop - cur >= 16 {
        emit.emit_zero_frame(cur, 8);
        cur += 8;
    }
    let rem = stop - cur;
    match rem {
        0 => {}
        1 | 2 | 4 | 8 => emit.emit_zero_frame(cur, rem as u32),
        3 => {
            emit.emit_zero_frame(cur, 2);
            emit.emit_zero_frame(cur + 2, 1);
        }
        6 if (cur & 3) == 2 => {
            emit.emit_zero_frame(cur, 2);
            emit.emit_zero_frame(cur + 2, 4);
        }
        6 if (cur & 3) == 0 => {
            emit.emit_zero_frame(cur, 4);
            emit.emit_zero_frame(cur + 4, 2);
        }
        5..=7 => {
            emit.emit_zero_frame(cur, 4);
            emit.emit_zero_frame(stop - 4, 4);
        }
        _ => {
            emit.emit_zero_frame(cur, 8);
            emit.emit_zero_frame(stop - 8, 8);
        }
    }
}

fn try_emit_zero_run<'a, E: Emitter<'a>>(emit: &mut E, insts: &[Inst], start: usize) -> Option<usize> {
    let first = emit.zero_stack_store(insts[start])?;
    let func = emit.func();
    let mut runs = vec![(first.0, first.0 + first.1 as i32)];
    let mut k = start + 1;
    while k < insts.len() {
        let inst = insts[k];
        let data = &func.insts[inst.index()];
        if !matches!(data.op, Op::Call { .. }) && data.result.is_some_and(|res| emit.is_elided(res)) {
            k += 1;
            continue;
        }
        let Some((off, width)) = emit.zero_stack_store(inst) else {
            break;
        };
        runs.push((off, off + width as i32));
        k += 1;
    }
    if runs.len() < 2 {
        return None;
    }
    runs.sort_unstable_by_key(|&(lo, _)| lo);
    let (mut cur_lo, mut cur_hi) = runs[0];
    for &(lo, hi) in &runs[1..] {
        if lo <= cur_hi {
            cur_hi = cur_hi.max(hi);
        } else {
            emit_zero_interval(emit, cur_lo, cur_hi);
            cur_lo = lo;
            cur_hi = hi;
        }
    }
    emit_zero_interval(emit, cur_lo, cur_hi);
    Some(k)
}

pub(crate) fn emit_body<'a, E: Emitter<'a>>(emit: &mut E, block: crate::ir::Block) -> Result<(), String> {
    emit.clear_flags();
    let func = emit.func();
    let insts = &func.blocks[block.index()].insts;
    let (mut ahead, mut loc, mut k) = (0, 0, 0);
    while k < insts.len() {
        let inst = insts[k];
        let data = &func.insts[inst.index()];
        if !matches!(data.op, Op::Call { .. }) && data.result.is_some_and(|res| emit.is_elided(res)) {
            k += 1;
            continue;
        }
        if ahead <= k {
            ahead = insts[k..].iter().position(|i| func.insts[i.index()].loc != 0).map_or(insts.len(), |at| k + at);
            loc = insts.get(ahead).map_or(loc, |i| func.insts[i.index()].loc);
            if loc == 0 {
                loc = func.home_loc();
            }
        }
        mark_line(emit, loc);
        if let Some(next_k) = try_emit_zero_run(emit, insts, k) {
            k = next_k;
            continue;
        }
        emit.inst(inst)?;
        k += 1;
    }
    if !insts.last().is_some_and(|inst| func.insts[inst.index()].op.is_terminator()) {
        emit.trap();
    }
    Ok(())
}

/// Starts a line-table row for `loc` at the current offset unless the open row already has it; 0 marks code with no source.
pub(super) fn mark_line<'a, E: Emitter<'a>>(emit: &mut E, loc: u64) {
    let at = emit.pos();
    let lines = emit.lines();
    match lines.last_mut() {
        Some(last) if last.1 == loc => {}
        Some(last) if last.0 == at => last.1 = loc,
        _ => lines.push((at, loc)),
    }
}
