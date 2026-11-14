use super::{Kind, Scan};
use crate::ir::{Convert, Function, Inst, Lane, Op, StackSlot, Type, Value};

#[derive(Clone, Default)]
struct SlotSummary {
    invalid: bool,
    vec_ty: Option<Type>,
    has_scalar: bool,
    uniform_scalar: Option<Lane>,
}

fn slot_homes(func: &Function, scan: &Scan) -> Vec<Option<Type>> {
    let mut slots = vec![SlotSummary::default(); func.slots.len()];
    for access in &scan.accesses {
        let var = &scan.vars[access.var];
        let slot_idx = var.slot.index();
        if scan.escaped[slot_idx] {
            continue;
        }
        let info = &mut slots[slot_idx];
        if info.invalid {
            continue;
        }
        let size = func.slots[slot_idx].size;
        if !matches!(size, 8 | 16) {
            info.invalid = true;
            continue;
        }
        let ty = access.ty;
        if ty.is_vector() {
            if var.offset != 0 || ty.bytes() != size {
                info.invalid = true;
            } else if info.vec_ty.is_none() {
                info.vec_ty = Some(ty);
            }
            continue;
        }
        let Some(lane) = Lane::from_type(ty) else {
            info.invalid = true;
            continue;
        };
        let bytes = i64::from(lane.bytes());
        if var.offset < 0 || var.offset + bytes > i64::from(size) || var.offset % bytes != 0 {
            info.invalid = true;
            continue;
        }
        if !info.has_scalar {
            info.has_scalar = true;
            info.uniform_scalar = Some(lane);
        } else if info.uniform_scalar != Some(lane) {
            info.uniform_scalar = None;
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(slot_idx, info)| {
            if info.invalid || !info.has_scalar {
                return None;
            }
            let vec_ty = info.vec_ty?;
            let size = func.slots[slot_idx].size;
            Some(
                info.uniform_scalar
                    .map(|lane| Type::Vec { lane, count: (size / lane.bytes()) as u8 })
                    .unwrap_or(vec_ty),
            )
        })
        .collect()
}

fn bitcast_if_needed(func: &mut Function, val: Value, from: Type, to: Type, out: &mut Vec<Inst>) -> Value {
    if from == to {
        return val;
    }
    let cast = func.value(to);
    out.push(func.add(Op::Convert(Convert::Bitcast, val), Some(cast)));
    cast
}

fn emit_scalar_load(
    func: &mut Function,
    inst: Inst,
    slot: StackSlot,
    (home_vec, lane_vec, lane_idx): (Type, Type, u8),
    out: &mut Vec<Inst>,
) {
    let vec_val = func.value(home_vec);
    out.push(func.add(Op::StackLoad(slot, 0), Some(vec_val)));
    let src = bitcast_if_needed(func, vec_val, home_vec, lane_vec, out);
    func.insts[inst.index()].op = Op::Extract(src, lane_idx);
    out.push(inst);
}

fn emit_scalar_store(
    func: &mut Function,
    inst: Inst,
    val: Value,
    slot: StackSlot,
    (home_vec, lane_vec, lane_idx): (Type, Type, u8),
    out: &mut Vec<Inst>,
) {
    let old_vec = func.value(home_vec);
    out.push(func.add(Op::StackLoad(slot, 0), Some(old_vec)));
    let src = bitcast_if_needed(func, old_vec, home_vec, lane_vec, out);
    let updated = func.value(lane_vec);
    out.push(func.add(Op::Insert(src, val, lane_idx), Some(updated)));
    let stored = bitcast_if_needed(func, updated, lane_vec, home_vec, out);
    func.insts[inst.index()].op = Op::StackStore(stored, slot, 0);
    out.push(inst);
}

/// Rewrites aligned scalar lane accesses on unescaped vector-touched slots into full-vector lane ops.
pub(super) fn split(func: &mut Function, scan: &Scan) -> bool {
    let homes = slot_homes(func, scan);
    if homes.iter().all(Option::is_none) {
        return false;
    }
    let mut changed = false;
    for block in 0..func.blocks.len() {
        let insts = std::mem::take(&mut func.blocks[block].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let Some(touch) = scan.touch(&func.insts[inst.index()]) else {
                out.push(inst);
                continue;
            };
            let Some(home_vec) = homes[touch.slot.index()] else {
                out.push(inst);
                continue;
            };
            let ty = func.ty(touch.kind.value());
            if ty.is_vector() {
                out.push(inst);
                continue;
            }
            let Some(off) = touch.offset else {
                out.push(inst);
                continue;
            };
            let Some(lane) = Lane::from_type(ty) else {
                out.push(inst);
                continue;
            };
            let size = func.slots[touch.slot.index()].size;
            let lane_vec = Type::Vec { lane, count: (size / lane.bytes()) as u8 };
            let lane_idx = (off / i64::from(lane.bytes())) as u8;
            match touch.kind {
                Kind::Load(_) => emit_scalar_load(func, inst, touch.slot, (home_vec, lane_vec, lane_idx), &mut out),
                Kind::Store(val) => emit_scalar_store(func, inst, val, touch.slot, (home_vec, lane_vec, lane_idx), &mut out),
            }
            changed = true;
        }
        func.blocks[block].insts = out;
    }
    changed
}
