use super::{Kind, Scan};
use crate::ir::{Convert, Function, Inst, Lane, Op, StackSlot, Type, Value};

use std::cmp::Reverse;

#[derive(Clone, Copy)]
struct VecChunk {
    offset: i64,
    bytes: u32,
    vec_ty: Type,
    needs_split: bool,
    uniform_scalar: Option<Lane>,
    invalid: bool,
}

#[derive(Clone, Copy)]
struct ChunkHome {
    offset: i64,
    bytes: u32,
    home_vec: Type,
}

fn slot_homes(func: &Function, scan: &Scan) -> Vec<Vec<ChunkHome>> {
    let mut vec_accesses: Vec<(usize, i64, u32, Type)> = Vec::new();
    for access in &scan.accesses {
        let var = &scan.vars[access.var];
        let slot_idx = var.slot.index();
        if scan.escaped[slot_idx] || !access.ty.is_vector() {
            continue;
        }
        vec_accesses.push((slot_idx, var.offset, access.ty.bytes(), access.ty));
    }
    if vec_accesses.is_empty() {
        return Vec::new();
    }
    vec_accesses.sort_unstable_by_key(|&(slot_idx, off, bytes, _)| (slot_idx, off, Reverse(bytes)));
    let mut slots: Vec<Vec<VecChunk>> = vec![Vec::new(); func.slots.len()];
    for (slot_idx, off, bytes, ty) in vec_accesses {
        let slot_size = i64::from(func.slots[slot_idx].size);
        let end = off.checked_add(i64::from(bytes));
        let valid_geom = off >= 0
            && matches!(bytes, 8 | 16)
            && end.is_some_and(|e| e <= slot_size)
            && i32::try_from(off).is_ok();
        let list = &mut slots[slot_idx];
        if !valid_geom {
            for chunk in list.iter_mut() {
                let c_end = chunk.offset + i64::from(chunk.bytes);
                if off < c_end && end.is_none_or(|e| chunk.offset < e) {
                    chunk.invalid = true;
                }
            }
            continue;
        }
        let end = off + i64::from(bytes);
        if list.iter().any(|c| c.offset == off && c.bytes == bytes) {
            continue;
        }
        let covering = list
            .iter()
            .filter(|c| off < c.offset + i64::from(c.bytes) && c.offset < end)
            .count();
        if covering == 1
            && bytes == 8
            && let Some(chunk) = list.iter_mut().find(|c| {
                !c.invalid
                    && c.bytes == 16
                    && off >= c.offset
                    && end <= c.offset + 16
                    && (off - c.offset) % 8 == 0
            })
        {
            chunk.needs_split = true;
            chunk.uniform_scalar = None;
            continue;
        }
        let mut overlapped = false;
        for chunk in list.iter_mut() {
            if off < chunk.offset + i64::from(chunk.bytes) && chunk.offset < end {
                chunk.invalid = true;
                overlapped = true;
            }
        }
        list.push(VecChunk {
            offset: off,
            bytes,
            vec_ty: ty,
            needs_split: false,
            uniform_scalar: None,
            invalid: overlapped,
        });
    }
    for access in &scan.accesses {
        let var = &scan.vars[access.var];
        let slot_idx = var.slot.index();
        if scan.escaped[slot_idx] || access.ty.is_vector() || slots[slot_idx].is_empty() {
            continue;
        }
        let acc_off = var.offset;
        let acc_end = acc_off + i64::from(access.ty.bytes());
        for chunk in &mut slots[slot_idx] {
            let c_end = chunk.offset + i64::from(chunk.bytes);
            if acc_off >= c_end || chunk.offset >= acc_end || chunk.invalid {
                continue;
            }
            if acc_off < chunk.offset || acc_end > c_end {
                chunk.invalid = true;
                continue;
            }
            let Some(lane) = Lane::from_type(access.ty) else {
                chunk.invalid = true;
                continue;
            };
            let lbytes = i64::from(lane.bytes());
            if (acc_off - chunk.offset) % lbytes != 0 {
                chunk.invalid = true;
                continue;
            }
            if !chunk.needs_split {
                chunk.needs_split = true;
                chunk.uniform_scalar = Some(lane);
            } else if chunk.uniform_scalar != Some(lane) {
                chunk.uniform_scalar = None;
            }
        }
    }
    slots
        .into_iter()
        .map(|list| {
            let mut homes: Vec<ChunkHome> = list
                .into_iter()
                .filter(|c| !c.invalid && c.needs_split)
                .map(|c| ChunkHome {
                    offset: c.offset,
                    bytes: c.bytes,
                    home_vec: c
                        .uniform_scalar
                        .map(|lane| Type::Vec {
                            lane,
                            count: (c.bytes / lane.bytes()) as u8,
                        })
                        .unwrap_or(c.vec_ty),
                })
                .collect();
            homes.sort_unstable_by_key(|h| h.offset);
            homes
        })
        .collect()
}

fn bitcast_if_needed(
    func: &mut Function,
    val: Value,
    from: Type,
    to: Type,
    out: &mut Vec<Inst>,
) -> Value {
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
    (slot, chunk_off): (StackSlot, i32),
    (home_vec, lane_vec, lane_idx): (Type, Type, u8),
    out: &mut Vec<Inst>,
) {
    let vec_val = func.value(home_vec);
    out.push(func.add(Op::StackLoad(slot, chunk_off), Some(vec_val)));
    let src = bitcast_if_needed(func, vec_val, home_vec, lane_vec, out);
    func.insts[inst.index()].op = Op::Extract(src, lane_idx);
    out.push(inst);
}

fn emit_scalar_store(
    func: &mut Function,
    inst: Inst,
    val: Value,
    (slot, chunk_off): (StackSlot, i32),
    (home_vec, lane_vec, lane_idx): (Type, Type, u8),
    out: &mut Vec<Inst>,
) {
    let old_vec = func.value(home_vec);
    out.push(func.add(Op::StackLoad(slot, chunk_off), Some(old_vec)));
    let src = bitcast_if_needed(func, old_vec, home_vec, lane_vec, out);
    let updated = func.value(lane_vec);
    out.push(func.add(Op::Insert(src, val, lane_idx), Some(updated)));
    let stored = bitcast_if_needed(func, updated, lane_vec, home_vec, out);
    func.insts[inst.index()].op = Op::StackStore(stored, slot, chunk_off);
    out.push(inst);
}

/// Rewrites aligned scalar and sub-vector accesses inside unescaped vector chunks into full-chunk lane ops.
pub(super) fn split(func: &mut Function, scan: &Scan) -> bool {
    let homes = slot_homes(func, scan);
    if homes.iter().all(Vec::is_empty) {
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
            let ty = func.ty(touch.kind.value());
            let Some(off) = touch.offset else {
                out.push(inst);
                continue;
            };
            let end = off + i64::from(ty.bytes());
            let Some(home) = homes[touch.slot.index()]
                .iter()
                .find(|h| off >= h.offset && end <= h.offset + i64::from(h.bytes))
            else {
                out.push(inst);
                continue;
            };
            if ty.is_vector() && off == home.offset && ty.bytes() == home.bytes {
                out.push(inst);
                continue;
            }
            let chunk_off = home.offset as i32;
            if ty.is_vector() {
                let lane_vec = Type::Vec {
                    lane: Lane::I64,
                    count: 2,
                };
                let lane_idx = ((off - home.offset) / 8) as u8;
                match touch.kind {
                    Kind::Load(_) => {
                        let vec_val = func.value(home.home_vec);
                        out.push(func.add(Op::StackLoad(touch.slot, chunk_off), Some(vec_val)));
                        let src =
                            bitcast_if_needed(func, vec_val, home.home_vec, lane_vec, &mut out);
                        let extracted = func.value(Type::I64);
                        out.push(func.add(Op::Extract(src, lane_idx), Some(extracted)));
                        func.insts[inst.index()].op = Op::Convert(Convert::Bitcast, extracted);
                        out.push(inst);
                    }
                    Kind::Store(val) => {
                        let scalar = bitcast_if_needed(func, val, ty, Type::I64, &mut out);
                        emit_scalar_store(
                            func,
                            inst,
                            scalar,
                            (touch.slot, chunk_off),
                            (home.home_vec, lane_vec, lane_idx),
                            &mut out,
                        );
                    }
                }
                changed = true;
                continue;
            }
            let Some(lane) = Lane::from_type(ty) else {
                out.push(inst);
                continue;
            };
            let lane_vec = Type::Vec {
                lane,
                count: (home.bytes / lane.bytes()) as u8,
            };
            let lane_idx = ((off - home.offset) / i64::from(lane.bytes())) as u8;
            match touch.kind {
                Kind::Load(_) => emit_scalar_load(
                    func,
                    inst,
                    (touch.slot, chunk_off),
                    (home.home_vec, lane_vec, lane_idx),
                    &mut out,
                ),
                Kind::Store(val) => emit_scalar_store(
                    func,
                    inst,
                    val,
                    (touch.slot, chunk_off),
                    (home.home_vec, lane_vec, lane_idx),
                    &mut out,
                ),
            }
            changed = true;
        }
        func.blocks[block].insts = out;
    }
    changed
}
