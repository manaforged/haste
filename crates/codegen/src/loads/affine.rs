use super::fits_mem_imm;
use super::join::for_edge_args;
use crate::ir::{Binary, Block, Function, Inst, Op, StackSlot, Type, Value};
use crate::mem::compute_affine;
use crate::merge::{identity, rename};

fn collect_incoming(func: &Function, target: Block, param_count: usize) -> Option<Vec<Vec<Value>>> {
    let mut incoming: Vec<Vec<Value>> = Vec::new();
    for pred in &func.blocks {
        let Some(&last) = pred.insts.last() else {
            continue;
        };
        if !for_edge_args(&func.insts[last.index()].op, target, |a| {
            incoming.push(a.to_vec())
        }) {
            return None;
        }
    }
    if incoming.is_empty() || incoming.iter().any(|args| args.len() != param_count) {
        return None;
    }
    Some(incoming)
}

fn find_affine_delta(
    affine: &[(Value, i64)],
    incoming: &[Vec<Value>],
    i: usize,
    j: usize,
) -> Option<i64> {
    let (r0_i, o0_i) = affine[incoming[0][i].index()];
    let (r0_j, o0_j) = affine[incoming[0][j].index()];
    if r0_i != r0_j {
        return None;
    }
    let delta = o0_i.wrapping_sub(o0_j);
    let all_match = incoming.iter().skip(1).all(|args| {
        let (ri, oi) = affine[args[i].index()];
        let (rj, oj) = affine[args[j].index()];
        ri == rj && oi.wrapping_sub(oj) == delta
    });
    all_match.then_some(delta)
}

fn eliminate_block_affine_params(
    func: &mut Function,
    b_idx: usize,
    affine: &[(Value, i64)],
    subst: &mut Vec<Value>,
) -> bool {
    let target = Block(b_idx as u32);
    let params = func.blocks[b_idx].params.clone();
    let Some(incoming) = collect_incoming(func, target, params.len()) else {
        return false;
    };
    let mut eliminated = vec![false; params.len()];
    let mut prepended: Vec<Inst> = Vec::new();
    for i in 1..params.len() {
        if func.ty(params[i]) != Type::I64 {
            continue;
        }
        for j in 0..i {
            if eliminated[j] || func.ty(params[j]) != Type::I64 {
                continue;
            }
            let Some(delta) = find_affine_delta(affine, &incoming, i, j) else {
                continue;
            };
            eliminated[i] = true;
            if delta == 0 {
                subst[params[i].index()] = params[j];
            } else {
                let c_val = func.value(Type::I64);
                let c_inst = func.add(Op::Const(delta), Some(c_val));
                let derived = func.value(Type::I64);
                let d_inst = func.add(Op::Binary(Binary::Iadd, params[j], c_val), Some(derived));
                subst.push(c_val);
                subst.push(derived);
                subst[params[i].index()] = derived;
                prepended.push(c_inst);
                prepended.push(d_inst);
            }
            break;
        }
    }
    if !eliminated.iter().any(|&e| e) {
        return false;
    }
    if !prepended.is_empty() {
        prepended.append(&mut func.blocks[b_idx].insts);
        func.blocks[b_idx].insts = prepended;
    }
    let retain_mask = |args: &mut Vec<Value>| {
        let mut k = 0;
        args.retain(|_| {
            k += 1;
            !eliminated[k - 1]
        });
    };
    retain_mask(&mut func.blocks[b_idx].params);
    for pred_idx in 0..func.blocks.len() {
        func.for_edge_args_mut(Block(pred_idx as u32), target, retain_mask);
    }
    true
}

pub(super) fn eliminate_affine_block_params(
    func: &mut Function,
) -> (bool, Option<Vec<(Value, i64)>>) {
    let Some(entry) = func.entry else {
        return (false, None);
    };
    let mergeable = |func: &Function, b_idx: usize| {
        Block(b_idx as u32) != entry && func.blocks[b_idx].params.len() >= 2
    };
    if !(0..func.blocks.len()).any(|b_idx| mergeable(func, b_idx)) {
        return (false, None);
    }
    let affine = compute_affine(func);
    let mut subst = identity(func.types.len());
    let mut any_changed = false;
    for b_idx in 0..func.blocks.len() {
        if mergeable(func, b_idx) {
            any_changed |= eliminate_block_affine_params(func, b_idx, &affine, &mut subst);
        }
    }
    if any_changed {
        rename(func, &subst);
    }
    (any_changed, (!any_changed).then_some(affine))
}

pub(super) fn fold_affine_mem(
    func: &mut Function,
    slots: &[Option<(StackSlot, i32)>],
    affine: &[(Value, i64)],
) -> bool {
    let mut changed = false;
    for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let Some(ty) = (match func.insts[inst.index()].op {
            Op::Load(..) => func.insts[inst.index()].result.map(|r| func.ty(r)),
            Op::Store(val, ..) => Some(func.ty(val)),
            _ => None,
        }) else {
            continue;
        };
        let (Op::Load(addr, off) | Op::Store(_, addr, off)) = &mut func.insts[inst.index()].op
        else {
            continue;
        };
        if let Some((slot, base_off)) = slots[addr.index()] {
            if let Some(total) = i64::from(base_off).checked_add(i64::from(*off))
                && crate::mem::in_range(total, ty.bytes(), func.slots[slot.index()].size)
                && let Ok(disp) = i32::try_from(total)
            {
                func.insts[inst.index()].op = match func.insts[inst.index()].op {
                    Op::Load(..) => Op::StackLoad(slot, disp),
                    Op::Store(v, ..) => Op::StackStore(v, slot, disp),
                    _ => unreachable!(),
                };
                changed = true;
            }
            continue;
        }
        let (root, base_off) = affine[addr.index()];
        if root != *addr
            && func.types[root.index()] == Type::I64
            && slots[root.index()].is_none()
            && let Some(new_off) = i64::from(*off).checked_add(base_off)
            && fits_mem_imm(ty, new_off)
        {
            *addr = root;
            *off = new_off as i32;
            changed = true;
        }
    }
    changed
}
