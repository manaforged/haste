use super::{Kind, Scan};
use crate::ir::{Convert, Function, Inst, Lane, Op, StackSlot, Type, Value};

#[derive(Clone, Copy)]
struct SlotHome {
    vec_ty: Type,
    lane: Lane,
}

#[derive(Clone, Default)]
struct SlotSummary {
    invalid: bool,
    has_vec: bool,
    has_scalar: bool,
    scalar_bytes: Option<u32>,
    uniform_scalar: Option<Lane>,
    first_scalar: Option<Lane>,
}

fn slot_homes(func: &Function, scan: &Scan) -> Vec<Option<SlotHome>> {
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
            } else {
                info.has_vec = true;
            }
            continue;
        }
        let Some(lane) = Lane::from_type(ty) else {
            info.invalid = true;
            continue;
        };
        let bytes = lane.bytes();
        if var.offset < 0 || var.offset + i64::from(bytes) > i64::from(size) || var.offset % i64::from(bytes) != 0 {
            info.invalid = true;
            continue;
        }
        if let Some(prev) = info.scalar_bytes {
            if prev != bytes {
                info.invalid = true;
                continue;
            }
        } else {
            info.scalar_bytes = Some(bytes);
        }
        info.has_scalar = true;
        match info.first_scalar {
            None => {
                info.first_scalar = Some(lane);
                info.uniform_scalar = Some(lane);
            }
            Some(_) if info.uniform_scalar != Some(lane) => info.uniform_scalar = None,
            Some(_) => {}
        }
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(slot_idx, info)| {
            if info.invalid || !info.has_vec || !info.has_scalar {
                return None;
            }
            let bytes = info.scalar_bytes?;
            let count = (func.slots[slot_idx].size / bytes) as u8;
            let vec_match = scan.accesses.iter().find_map(|access| {
                if scan.vars[access.var].slot.index() != slot_idx {
                    return None;
                }
                access.ty.vec_shape().and_then(|(vl, _)| (vl.bytes() == bytes).then_some(vl))
            });
            let lane = info.uniform_scalar.or(vec_match).or(info.first_scalar)?;
            Some(SlotHome { vec_ty: Type::Vec { lane, count }, lane })
        })
        .collect()
}

fn emit_scalar_load(
    func: &mut Function,
    inst: Inst,
    slot: StackSlot,
    lane_idx: u8,
    home: SlotHome,
    ty: Type,
    out: &mut Vec<Inst>,
) {
    let vec_val = func.value(home.vec_ty);
    out.push(func.add(Op::StackLoad(slot, 0), Some(vec_val)));
    let home_scalar = home.lane.to_type();
    if ty == home_scalar {
        func.insts[inst.index()].op = Op::Extract(vec_val, lane_idx);
    } else {
        let ext = func.value(home_scalar);
        out.push(func.add(Op::Extract(vec_val, lane_idx), Some(ext)));
        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, ext);
    }
    out.push(inst);
}

fn emit_scalar_store(
    func: &mut Function,
    inst: Inst,
    val: Value,
    slot: StackSlot,
    lane_idx: u8,
    home: SlotHome,
    ty: Type,
    out: &mut Vec<Inst>,
) {
    let old_vec = func.value(home.vec_ty);
    out.push(func.add(Op::StackLoad(slot, 0), Some(old_vec)));
    let home_scalar = home.lane.to_type();
    let elem = if ty == home_scalar {
        val
    } else {
        let cast = func.value(home_scalar);
        out.push(func.add(Op::Convert(Convert::Bitcast, val), Some(cast)));
        cast
    };
    let new_vec = func.value(home.vec_ty);
    out.push(func.add(Op::Insert(old_vec, elem, lane_idx), Some(new_vec)));
    func.insts[inst.index()].op = Op::StackStore(new_vec, slot, 0);
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
            let Some(home) = homes[touch.slot.index()] else {
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
            let lane_idx = (off / i64::from(home.lane.bytes())) as u8;
            match touch.kind {
                Kind::Load(_) => emit_scalar_load(func, inst, touch.slot, lane_idx, home, ty, &mut out),
                Kind::Store(val) => emit_scalar_store(func, inst, val, touch.slot, lane_idx, home, ty, &mut out),
            }
            changed = true;
        }
        func.blocks[block].insts = out;
    }
    changed
}
