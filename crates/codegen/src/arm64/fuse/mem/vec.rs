use super::super::{const_of, def_entry, def_inst, local_single, preserved};
use crate::alloc::Plan;
use crate::arm64::asm::Reg;
use crate::ir::{Binary, Function, Inst, Lane, Op, ShuffleMask, StackSlot, Type, Value};

fn movi_enc(q: u32, op: u32, cmode: u32, byte: u8) -> u32 {
    let imm = u32::from(byte);
    0x0F00_0400 | (q << 30) | (op << 29) | (cmode << 12) | ((imm >> 5) << 16) | ((imm & 0x1F) << 5)
}

fn byte_bitmask64(v: u64) -> Option<u8> {
    let mut mask = 0_u8;
    for i in 0..8 {
        let byte = (v >> (i * 8)) & 0xFF;
        if byte == 0xFF {
            mask |= 1 << i;
        } else if byte != 0 {
            return None;
        }
    }
    Some(mask)
}

fn splat_pattern64(lane: Lane, imm: i64) -> u64 {
    let raw = imm as u64;
    match lane {
        Lane::I8 => u64::from(raw as u8) * 0x0101_0101_0101_0101,
        Lane::I16 => u64::from(raw as u16) * 0x0001_0001_0001_0001,
        Lane::I32 | Lane::F32 => u64::from(raw as u32) * 0x0000_0001_0000_0001,
        Lane::I64 | Lane::F64 => raw,
    }
}

fn encode_neon_movi_16(q: u32, v16: u16) -> Option<u32> {
    let v8 = v16 as u8;
    if (v16 >> 8) as u8 == v8 {
        return Some(movi_enc(q, 0, 0xE, v8));
    }
    for (op, raw) in [(0_u32, v16), (1_u32, !v16)] {
        if raw & 0xFF00 == 0 {
            return Some(movi_enc(q, op, 0x8, raw as u8));
        }
        if raw & 0x00FF == 0 {
            return Some(movi_enc(q, op, 0xA, (raw >> 8) as u8));
        }
    }
    None
}

fn encode_neon_movi_32(q: u32, v32: u32) -> Option<u32> {
    for (op, raw) in [(0_u32, v32), (1_u32, !v32)] {
        for (idx, cmode) in [0x0_u32, 0x2, 0x4, 0x6].into_iter().enumerate() {
            let shift = idx * 8;
            if raw == ((raw >> shift) & 0xFF) << shift {
                return Some(movi_enc(q, op, cmode, (raw >> shift) as u8));
            }
        }
        for (idx, cmode) in [0xC_u32, 0xD].into_iter().enumerate() {
            let fill_bits = (idx + 1) * 8;
            let fill = (1_u32 << fill_bits) - 1;
            let top = (raw >> fill_bits) & 0xFF;
            if raw == ((top << fill_bits) | fill) {
                return Some(movi_enc(q, op, cmode, top as u8));
            }
        }
    }
    None
}

pub(in crate::arm64) fn encode_neon_movi(q: u32, pattern: u64) -> Option<u32> {
    if pattern == 0 {
        return Some(movi_enc(q, 1, 0xE, 0));
    }
    let v32 = pattern as u32;
    if v32 == (pattern >> 32) as u32 {
        let v16 = v32 as u16;
        if (v32 >> 16) as u16 == v16
            && let Some(word) = encode_neon_movi_16(q, v16)
        {
            return Some(word);
        }
        if let Some(word) = encode_neon_movi_32(q, v32) {
            return Some(word);
        }
    }
    byte_bitmask64(pattern).map(|mask| movi_enc(q, 1, 0xE, mask))
}

struct PairHalf {
    val: Value,
    pos: usize,
    base_pos: usize,
    base: Option<Value>,
    slot: Option<StackSlot>,
    off: i32,
    addr_elide: Option<Value>,
}

fn splat_insert(
    func: &Function,
    pos_of: &[Option<(usize, usize, Inst)>],
    res: Value,
) -> Option<(Value, Value)> {
    let Op::Shuffle(a, b, ShuffleMask::Const(m)) = func.insts[def_inst(pos_of, res)?.index()].op
    else {
        return None;
    };
    let Type::Vec { lane, count } = func.ty(res) else {
        return None;
    };
    let (count, first) = (usize::from(count), usize::from(m[0]));
    if !matches!(lane.bytes() as usize * count, 8 | 16)
        || first >= 2 * count
        || !m[..count].iter().all(|&x| usize::from(x) == first)
    {
        return None;
    }
    let src = if first < count { a } else { b };
    let Op::Insert(_, scalar, idx) = func.insts[def_inst(pos_of, src)?.index()].op else {
        return None;
    };
    (usize::from(idx) == first % count).then_some((src, scalar))
}

pub(in crate::arm64::fuse) fn const_splat(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    res: Value,
) -> Option<(Value, i64)> {
    let (src, scalar) = splat_insert(func, pos_of, res)?;
    const_of(func, plan, scalar, |v| def_inst(pos_of, v)).map(|imm| (src, imm))
}

fn try_fuse_splat_shuffle(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (a, b): (Value, Value),
) {
    let Type::Vec { lane, count } = func.ty(res) else {
        return;
    };
    let Some((src, scalar)) = splat_insert(func, pos_of, res) else {
        return;
    };
    if uses[src.index()] != u32::from(a == src) + u32::from(b == src) || plan.elided[src.index()] {
        return;
    }
    let movi_ok = const_of(func, plan, scalar, |v| def_inst(pos_of, v)).is_some_and(|imm| {
        encode_neon_movi(
            u32::from(lane.bytes() * u32::from(count) == 16),
            splat_pattern64(lane, imm),
        )
        .is_some()
    });
    let gpr_ok = lane.is_int()
        && count > 1
        && def_entry(pos_of, src).is_some_and(|(sb, spos, _)| {
            sb == b_idx && spos < pos && preserved(func, plan, b_idx, spos, pos, None, scalar)
        });
    if movi_ok || gpr_ok {
        plan.elided[src.index()] = true;
    }
}

fn unwrap_iadd_const(
    func: &Function,
    plan: &Plan<Reg>,
    ctx: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    addr: Value,
) -> (Value, i64, Option<(usize, Value)>) {
    if plan.sib[addr.index()].is_none()
        && let Some((apos, ainst)) = local_single(plan, ctx, b_idx, pos, addr)
        && let Op::Binary(Binary::Iadd, base, c) = func.insts[ainst.index()].op
        && func.ty(base) == Type::I64
        && let Some(imm) = const_of(func, plan, c, |v| ctx.1[v.index()].map(|(_, _, i)| i))
    {
        return (base, imm, Some((apos, addr)));
    }
    (addr, 0, None)
}

fn load_pair_half(
    func: &Function,
    plan: &Plan<Reg>,
    ctx: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos): (usize, usize),
    val: Value,
) -> Option<PairHalf> {
    if func.ty(val).bytes() != 8 {
        return None;
    }
    let (lpos, linst) = local_single(plan, ctx, b_idx, pos, val)?;
    match func.insts[linst.index()].op {
        Op::StackLoad(slot, off) => Some(PairHalf {
            val,
            pos: lpos,
            base_pos: lpos,
            base: None,
            slot: Some(slot),
            off,
            addr_elide: None,
        }),
        Op::Load(addr, off) => {
            let (base, add_imm, addr_elide) =
                unwrap_iadd_const(func, plan, ctx, (b_idx, lpos), addr);
            if plan.elided[base.index()] || plan.sib[base.index()].is_some() {
                return None;
            }
            let eff = i64::from(off)
                .checked_add(add_imm)
                .and_then(|v| i32::try_from(v).ok())?;
            Some(PairHalf {
                val,
                pos: lpos,
                base_pos: addr_elide.map_or(lpos, |(ap, _)| ap),
                base: Some(base),
                slot: None,
                off: eff,
                addr_elide: addr_elide.map(|(_, a)| a),
            })
        }
        _ => None,
    }
}

fn try_fuse_insert_loads(
    func: &Function,
    plan: &mut Plan<Reg>,
    ctx: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, ins0): (usize, usize, Value),
    (lo_val, hi_val): (Value, Value),
) -> bool {
    let Some(lo) = load_pair_half(func, plan, ctx, (b_idx, pos), lo_val) else {
        return false;
    };
    let Some(hi) = load_pair_half(func, plan, ctx, (b_idx, pos), hi_val) else {
        return false;
    };
    let same_anchor =
        (lo.slot.is_some() && lo.slot == hi.slot) || (lo.base.is_some() && lo.base == hi.base);
    if !same_anchor || i64::from(lo.off).checked_add(8) != Some(i64::from(hi.off)) {
        return false;
    }
    let min_pos = lo.pos.min(hi.pos);
    if func.blocks[b_idx].insts[min_pos + 1..pos]
        .iter()
        .any(|&i| func.insts[i.index()].op.is_mem_barrier())
    {
        return false;
    }
    for v in [
        Some(ins0),
        Some(lo.val),
        Some(hi.val),
        lo.addr_elide,
        hi.addr_elide,
    ]
    .into_iter()
    .flatten()
    {
        plan.elided[v.index()] = true;
    }
    if let Some(base) = lo.base
        && !preserved(
            func,
            plan,
            b_idx,
            lo.base_pos.max(hi.base_pos),
            pos,
            None,
            base,
        )
    {
        for v in [
            Some(ins0),
            Some(lo.val),
            Some(hi.val),
            lo.addr_elide,
            hi.addr_elide,
        ]
        .into_iter()
        .flatten()
        {
            plan.elided[v.index()] = false;
        }
        return false;
    }
    true
}

fn try_fuse_vec_insert(
    func: &Function,
    plan: &mut Plan<Reg>,
    ctx: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (ins0, val_b, lane_b): (Value, Value, u8),
) {
    let Type::Vec { lane, count: 2 } = func.ty(res) else {
        return;
    };
    if !matches!(lane, Lane::I64 | Lane::F64) || lane_b >= 2 {
        return;
    }
    let Some((pos0, inst0)) = local_single(plan, ctx, b_idx, pos, ins0) else {
        return;
    };
    let Op::Insert(_, val_a, lane_a) = func.insts[inst0.index()].op else {
        return;
    };
    if lane_a + lane_b != 1 {
        return;
    }
    let (lo, hi) = if lane_a == 0 {
        (val_a, val_b)
    } else {
        (val_b, val_a)
    };
    if try_fuse_insert_loads(func, plan, ctx, (b_idx, pos, ins0), (lo, hi)) {
        return;
    }
    if plan.elided[val_a.index()] || plan.elided[val_b.index()] {
        return;
    }
    let ok = match lane {
        Lane::I64 => {
            plan.const_of(val_a) == Some(0) || preserved(func, plan, b_idx, pos0, pos, None, val_a)
        }
        Lane::F64 => pos0 + 1 == pos,
        _ => false,
    };
    if ok {
        plan.elided[ins0.index()] = true;
    }
}

pub(in crate::arm64::fuse) fn try_fuse_vec_op(
    func: &Function,
    plan: &mut Plan<Reg>,
    ctx: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    op: &Op,
) -> bool {
    match *op {
        Op::Shuffle(a, b, ShuffleMask::Const(_)) => {
            try_fuse_splat_shuffle(func, plan, ctx, (b_idx, pos, res), (a, b))
        }
        Op::Insert(vec, val, idx) => {
            try_fuse_vec_insert(func, plan, ctx, (b_idx, pos, res), (vec, val, idx))
        }
        _ => return false,
    }
    true
}
