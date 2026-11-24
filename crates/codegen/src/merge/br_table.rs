use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

use super::{emit_bin, emit_const, emit_icmp, prune_dead_local};

#[derive(Clone, Copy)]
struct MaskedRange {
    mask: u64,
    lo: u64,
    hi: u64,
}

enum BinaryPlan {
    Jump(Block),
    Ranges {
        idx: Value,
        ty: Type,
        hit: Block,
        default: Block,
        items: Vec<MaskedRange>,
    },
}

fn collect_hit_intervals(
    default: Block,
    targets: &[Block],
    u_max: u64,
) -> Option<(Option<Block>, Vec<(u64, u64)>)> {
    let mut hit: Option<Block> = None;
    let mut intervals: Vec<(u64, u64)> = Vec::new();
    for (i, &target) in targets.iter().enumerate() {
        let u = i as u64;
        if u > u_max {
            break;
        }
        if target == default {
            continue;
        }
        match hit {
            None => hit = Some(target),
            Some(prev) if prev != target => return None,
            Some(_) => {}
        }
        if let Some(last) = intervals.last_mut().filter(|last| last.1 + 1 == u) {
            last.1 = u;
        } else {
            intervals.push((u, u));
        }
    }
    Some((hit, intervals))
}

fn try_coalesce_pair(a: MaskedRange, b: MaskedRange) -> Option<MaskedRange> {
    if a.mask != b.mask || a.hi - a.lo != b.hi - b.lo {
        return None;
    }
    let d = b.lo - a.lo;
    if d == 0 || !d.is_power_of_two() || (d & a.mask) != d || (a.lo & d) != 0 || (a.lo ^ a.hi) >= d
    {
        return None;
    }
    Some(MaskedRange {
        mask: a.mask & !d,
        lo: a.lo,
        hi: a.hi,
    })
}

fn coalesce_pass(items: &mut Vec<MaskedRange>) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i < items.len() {
        if let Some((j, merged)) = ((i + 1)..items.len())
            .find_map(|j| try_coalesce_pair(items[i], items[j]).map(|m| (j, m)))
        {
            items[i] = merged;
            items.remove(j);
            changed = true;
        }
        i += 1;
    }
    changed
}

fn coalesce_masked_ranges(intervals: &[(u64, u64)], u_max: u64) -> Vec<MaskedRange> {
    let mut items: Vec<MaskedRange> = intervals
        .iter()
        .map(|&(lo, hi)| MaskedRange {
            mask: u_max,
            lo,
            hi,
        })
        .collect();
    if items.len() <= 512 {
        while coalesce_pass(&mut items) {}
    }
    items
}

fn local_def_op(func: &Function, b: usize, val: Value) -> Option<&Op> {
    func.blocks[b].insts.iter().find_map(|&inst| {
        let item = &func.insts[inst.index()];
        (item.result == Some(val)).then_some(&item.op)
    })
}

fn unshifted_br_index(
    func: &Function,
    b: usize,
    idx: Value,
    table_len: usize,
) -> Option<(Value, u64)> {
    if table_len == 0 {
        return None;
    }
    let &Op::Binary(Binary::Isub, base_v, min_v) = local_def_op(func, b, idx)? else {
        return None;
    };
    let &Op::Const(min) = local_def_op(func, b, min_v)? else {
        return None;
    };
    if min <= 0 {
        return None;
    }
    let orig = match local_def_op(func, b, base_v) {
        Some(&Op::Convert(Convert::Uextend, narrow)) if func.ty(narrow).is_small_int() => narrow,
        _ => base_v,
    };
    let shift = min as u64;
    let end = shift.checked_add(table_len as u64)?;
    if end > i64::MAX as u64 || end - 1 > func.ty(orig).unsigned_max() {
        return None;
    }
    Some((orig, shift))
}

fn profitable_masked_ranges(items: &[MaskedRange], u_max: u64) -> bool {
    items.len() == 1 || (items.len() <= 2 && items.iter().any(|it| it.mask != u_max))
}

fn binary_br_table_plan(func: &Function, b: usize) -> Option<BinaryPlan> {
    let inst = func.blocks[b].insts.last().copied()?;
    let Op::BrTable(idx, default, ref targets) = func.insts[inst.index()].op else {
        return None;
    };
    let ty = func.ty(idx);
    if !ty.is_small_int() {
        return None;
    }
    let u_max = ty.unsigned_max();
    let (hit_opt, intervals) = collect_hit_intervals(default, targets, u_max)?;
    let Some(hit) = hit_opt else {
        return Some(BinaryPlan::Jump(default));
    };
    if let Some((orig_idx, base)) = unshifted_br_index(func, b, idx, targets.len()) {
        let orig_ty = func.ty(orig_idx);
        let orig_umax = orig_ty.unsigned_max();
        let shifted: Vec<(u64, u64)> = intervals
            .iter()
            .map(|&(lo, hi)| (lo + base, hi + base))
            .collect();
        let items = coalesce_masked_ranges(&shifted, orig_umax);
        if profitable_masked_ranges(&items, orig_umax) {
            return Some(BinaryPlan::Ranges {
                idx: orig_idx,
                ty: orig_ty,
                hit,
                default,
                items,
            });
        }
    }
    let items = coalesce_masked_ranges(&intervals, u_max);
    profitable_masked_ranges(&items, u_max).then_some(BinaryPlan::Ranges {
        idx,
        ty,
        hit,
        default,
        items,
    })
}

fn emit_masked_range(
    func: &mut Function,
    insts: &mut Vec<Inst>,
    (idx, ty, u_max): (Value, Type, u64),
    item: MaskedRange,
) -> Value {
    let v = if item.mask == u_max {
        idx
    } else {
        let mask_c = emit_const(func, insts, ty, item.mask);
        emit_bin(func, insts, ty, Binary::Band, (idx, mask_c))
    };
    if item.lo == item.hi {
        let lo_c = emit_const(func, insts, ty, item.lo);
        emit_icmp(func, insts, IntCC::Equal, (v, lo_c))
    } else if item.lo == 0 {
        let hi_c = emit_const(func, insts, ty, item.hi);
        emit_icmp(func, insts, IntCC::UnsignedLessThanOrEqual, (v, hi_c))
    } else {
        let lo_c = emit_const(func, insts, ty, item.lo);
        let diff = emit_bin(func, insts, ty, Binary::Isub, (v, lo_c));
        let span_c = emit_const(func, insts, ty, item.hi - item.lo);
        emit_icmp(func, insts, IntCC::UnsignedLessThanOrEqual, (diff, span_c))
    }
}

fn apply_binary_plan(func: &mut Function, b: usize, plan: BinaryPlan) {
    let mut insts = Vec::new();
    let new_term = match plan {
        BinaryPlan::Jump(default) => Op::Jump(default, Vec::new()),
        BinaryPlan::Ranges {
            idx,
            ty,
            hit,
            default,
            items,
        } => {
            let u_max = ty.unsigned_max();
            let mut acc: Option<Value> = None;
            for item in items {
                let cmp = emit_masked_range(func, &mut insts, (idx, ty, u_max), item);
                acc = Some(match acc {
                    None => cmp,
                    Some(prev) => emit_bin(func, &mut insts, Type::I8, Binary::Bor, (prev, cmp)),
                });
            }
            let Some(cond) = acc else { return };
            Op::Brif(cond, hit, Vec::new(), default, Vec::new())
        }
    };
    let Some(term) = func.blocks[b].insts.pop() else {
        return;
    };
    func.insts[term.index()].op = new_term;
    func.blocks[b].insts.extend(insts);
    func.blocks[b].insts.push(term);
    prune_dead_local(func, b);
}

pub(crate) fn lower_binary_br_tables(func: &mut Function) -> bool {
    let has_br_table =
        (0..func.blocks.len()).any(|b| matches!(func.last(Block(b as u32)), Some(Op::BrTable(..))));
    if !has_br_table {
        return false;
    }
    let cfg = func.cfg();
    let mut changed = false;
    for b in 0..func.blocks.len() {
        if !cfg.reachable[b] {
            continue;
        }
        if let Some(plan) = binary_br_table_plan(func, b) {
            apply_binary_plan(func, b, plan);
            changed = true;
        }
    }
    changed
}
