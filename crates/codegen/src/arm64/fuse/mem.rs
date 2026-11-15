use super::super::asm::{ExtOp, Reg, encode_sib};
use super::{const_of, local_single, preserved, shift_parts, zero_ext_loc};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, Lane, Op, ShuffleMask, StackSlot, Type, Value};
use crate::live::{Sib, SibBase};

struct SibCand {
    base: SibBase,
    base_pos: usize,
    idx: Value,
    ext: ExtOp,
    shift: i64,
    idx_pos: usize,
    disp: i32,
    folded_const: bool,
    elide: [Option<Value>; 4],
}

fn mem_imm_fits(disp: i32, bytes: u32) -> bool {
    let unit = bytes as i32;
    (-256..256).contains(&disp) || (disp >= 0 && disp % unit == 0 && disp / unit < 4096)
}

fn slot_align_bytes(func: &Function, slot: StackSlot) -> u32 {
    let data = &func.slots[slot.index()];
    let min_align = if data.size >= 8 { 8 } else if data.size >= 4 { 4 } else if data.size >= 2 { 2 } else { 1 };
    (1_u32 << data.align_shift.min(4)).max(min_align)
}

fn sib_use_ok(func: &Function, cand: &SibCand, expected: u32, ty: Type, off: i32) -> bool {
    let Some(total) = cand.disp.checked_add(off) else { return false };
    if ty == Type::I128 {
        return false;
    }
    let bytes = ty.bytes();
    let scale_ok = cand.shift == 0 || bytes == (1 << cand.shift);
    match cand.base {
        SibBase::Value(_) => (total == 0 && scale_ok) || (expected == 1 && cand.folded_const && mem_imm_fits(total, bytes)),
        SibBase::Frame(slot) => expected == 1 && (-65536..=65536).contains(&total) && slot_align_bytes(func, slot) >= bytes && total % (bytes as i32) == 0,
    }
}

fn sib_uses_span(func: &Function, (b_idx, pos, res): (usize, usize, Value), cand: &SibCand, expected: u32) -> Option<usize> {
    let (mut seen, mut last_pos) = (0_u32, pos);
    for (u_pos, &u_inst) in func.blocks[b_idx].insts.iter().enumerate().skip(pos + 1) {
        let u_data = &func.insts[u_inst.index()];
        let hit = match (&u_data.op, u_data.result) {
            (&Op::Load(addr, off), Some(r)) if addr == res => Some((func.ty(r), off)),
            (&Op::Store(val, addr, off), _) if addr == res && val != res => Some((func.ty(val), off)),
            _ => None,
        };
        if hit.is_some_and(|(t, off)| sib_use_ok(func, cand, expected, t, off)) {
            seen += 1;
            last_pos = u_pos;
        }
    }
    (seen == expected).then_some(last_pos)
}

fn sib_preserved(func: &Function, plan: &Plan<Reg>, (b_idx, last_pos): (usize, usize), cand: &SibCand) -> bool {
    let shift_clobbers = cand.elide.into_iter().flatten().any(|kept| kept != cand.idx && !plan.elided[kept.index()] && plan.reg_of(kept) == plan.reg_of(cand.idx));
    if shift_clobbers || !preserved(func, plan, b_idx, cand.idx_pos, last_pos, None, cand.idx) {
        return false;
    }
    match cand.base {
        SibBase::Frame(_) => true,
        SibBase::Value(b) => preserved(func, plan, b_idx, cand.base_pos, last_pos, None, b),
    }
}

fn try_commit_sib(func: &Function, plan: &mut Plan<Reg>, uses: &[u32], (b_idx, pos, res): (usize, usize, Value), cand: SibCand) -> bool {
    let base_bad = matches!(cand.base, SibBase::Value(b) if plan.reg_of(b).is_none() || plan.elided[b.index()]);
    if base_bad || plan.reg_of(cand.idx).is_none() || plan.elided[cand.idx.index()] {
        return false;
    }
    let Some(last_pos) = sib_uses_span(func, (b_idx, pos, res), &cand, uses[res.index()]) else {
        return false;
    };
    plan.elided[res.index()] = true;
    let mut extra = [None; 4];
    for (slot, v) in extra.iter_mut().zip(cand.elide) {
        if let Some(ev) = v
            && uses[ev.index()] == 1
            && !plan.elided[ev.index()]
        {
            plan.elided[ev.index()] = true;
            *slot = Some(ev);
        }
    }
    if sib_preserved(func, plan, (b_idx, last_pos), &cand) {
        plan.sib[res.index()] = Some(Sib { base: cand.base, index: cand.idx, shift: encode_sib(cand.ext, cand.shift as u8), disp: cand.disp });
        return true;
    }
    plan.elided[res.index()] = false;
    for ev in extra.into_iter().flatten() {
        plan.elided[ev.index()] = false;
    }
    false
}

fn unwrap_sib_op(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    op: &Op,
) -> Option<(Value, ExtOp)> {
    match *op {
        Op::Convert(Convert::Sextend, inner) if func.ty(inner) == Type::I32 && !plan.elided[inner.index()] => Some((inner, ExtOp::Sxtw)),
        Op::Convert(Convert::Uextend, inner) if !plan.elided[inner.index()] && func.ty(inner).is_small_int() => {
            if func.ty(inner) == Type::I32 {
                Some((inner, if zero_ext_loc(plan, inner) { ExtOp::Uxtx } else { ExtOp::Uxtw }))
            } else {
                zero_ext_loc(plan, inner).then_some((inner, ExtOp::Uxtx))
            }
        }
        Op::Binary(Binary::Band, a, b) => {
            let def = |u: Value| pos_of[u.index()].map(|(_, _, i)| i);
            let (ext, inner) = super::ext_band_of(func, plan, Type::I64, (a, b), &def)?;
            (ext == ExtOp::Uxtw).then_some((inner, ExtOp::Uxtw))
        }
        _ => None,
    }
}

fn unwrap_sib_ext(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit): (usize, usize),
    (v, allow_elided): (Value, bool),
) -> Option<(Value, usize, ExtOp, Option<Value>)> {
    if uses[v.index()] != 1 || (!allow_elided && plan.elided[v.index()]) {
        return None;
    }
    let (sb, epos, einst) = pos_of[v.index()]?;
    if sb != b_idx || epos >= limit {
        return None;
    }
    let (inner, ext) = unwrap_sib_op(func, plan, pos_of, &func.insts[einst.index()].op)?;
    Some((inner, epos, ext, Some(v)))
}

fn parse_iadd_const(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], (x, y): (Value, Value)) -> Option<(Value, i32)> {
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    for (val, c) in [(x, y), (y, x)] {
        if let Some(imm) = const_of(func, plan, c, def).and_then(|v| i32::try_from(v).ok()) {
            return Some((val, imm));
        }
    }
    None
}

fn parse_sib_base_op(func: &Function, plan: &Plan<Reg>, pos_of: &[Option<(usize, usize, Inst)>], op: &Op) -> Option<(Value, i32)> {
    match *op {
        Op::Binary(Binary::Iadd, x, y) => parse_iadd_const(func, plan, pos_of, (x, y)),
        Op::Binary(Binary::Isub, real, c) => {
            let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
            const_of(func, plan, c, def).and_then(|v| v.checked_neg()).and_then(|v| i32::try_from(v).ok()).map(|imm| (real, imm))
        }
        _ => None,
    }
}

fn parse_sib_base(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit): (usize, usize),
    v: Value,
) -> Option<(SibBase, usize, i32, bool, Option<Value>)> {
    if let Location::Frame(slot, disp) = plan.location(v) {
        return Some((SibBase::Frame(slot), limit, disp, true, None));
    }
    if let Some((bpos, binst)) = local_single(plan, info, b_idx, limit, v) {
        let op = &func.insts[binst.index()].op;
        if let Op::StackAddr(slot, disp) = *op {
            return Some((SibBase::Frame(slot), bpos, disp, true, Some(v)));
        }
        if let Some((real, imm)) = parse_sib_base_op(func, plan, pos_of, op) {
            if let Location::Frame(slot, base_disp) = plan.location(real)
                && let Some(total) = base_disp.checked_add(imm)
            {
                return Some((SibBase::Frame(slot), bpos, total, true, Some(v)));
            }
            if plan.reg_of(real).is_some() && !plan.elided[real.index()] {
                return Some((SibBase::Value(real), bpos, imm, true, Some(v)));
            }
        }
    }
    (plan.reg_of(v).is_some() && !plan.elided[v.index()]).then_some((SibBase::Value(v), limit, 0, false, None))
}

fn parse_sib_index(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit, allow_elided): (usize, usize, bool),
    rhs: Value,
) -> Option<(Value, ExtOp, i64, usize, [Option<Value>; 2])> {
    if let Some((sb, spos, sinst)) = pos_of[rhs.index()]
        && sb == b_idx
        && spos < limit
        && let Some((Binary::Ishl, mid, shift)) = shift_parts(&func.insts[sinst.index()].op, plan)
        && (1..=3).contains(&shift)
    {
        let usable = uses[rhs.index()] == 1 && (allow_elided || !plan.elided[rhs.index()]);
        if usable && let Some((idx, idx_pos, ext, uext)) = unwrap_sib_ext(func, plan, info, (b_idx, spos), (mid, true)) {
            return Some((idx, ext, shift, idx_pos, [Some(rhs), uext]));
        }
        if !plan.elided[mid.index()] {
            return Some((mid, ExtOp::Uxtx, shift, spos, [Some(rhs), None]));
        }
    }
    let (idx, idx_pos, ext, uext) = unwrap_sib_ext(func, plan, info, (b_idx, limit), (rhs, allow_elided))?;
    Some((idx, ext, 0, idx_pos, [uext, None]))
}

fn try_sib_arm(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, _): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, res_pos, sum_pos, res): (usize, usize, usize, Value),
    (base_val, rhs, outer, plain): (Value, Value, Option<(Value, i32)>, bool),
) -> bool {
    let (outer_elide, outer_disp) = outer.map_or((None, 0), |(p, d)| (Some(p), d));
    let (idx, ext, shift, idx_pos, [e0, e1]) = if plain {
        (rhs, ExtOp::Uxtx, 0, sum_pos, [None, None])
    } else if let Some(parsed) = parse_sib_index(func, plan, info, (b_idx, sum_pos, outer.is_some()), rhs) {
        parsed
    } else {
        return false;
    };
    let Some((base, base_pos, b_disp, b_folded, b_elide)) = parse_sib_base(func, plan, info, (b_idx, sum_pos), base_val) else {
        return false;
    };
    let Some(disp) = b_disp.checked_add(outer_disp) else { return false };
    let cand = SibCand { base, base_pos, idx, ext, shift, idx_pos, disp, folded_const: b_folded || outer.is_some(), elide: [e0, e1, b_elide, outer_elide] };
    try_commit_sib(func, plan, uses, (b_idx, res_pos, res), cand)
}

fn try_sib_sum(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, res_pos, sum_pos, res): (usize, usize, usize, Value),
    (x, y, outer): (Value, Value, Option<(Value, i32)>),
) -> bool {
    for plain in [false, true] {
        for (b_arm, i_arm) in [(x, y), (y, x)] {
            if try_sib_arm(func, plan, info, (b_idx, res_pos, sum_pos, res), (b_arm, i_arm, outer, plain)) {
                return true;
            }
        }
    }
    false
}

pub(super) fn try_fuse_sib(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (x, y): (Value, Value),
) -> bool {
    if func.ty(res) != Type::I64 || plan.sib[res.index()].is_some() || uses[res.index()] == 0 {
        return false;
    }
    if try_sib_sum(func, plan, info, (b_idx, pos, pos, res), (x, y, None)) {
        return true;
    }
    let Some((p, disp)) = parse_iadd_const(func, plan, pos_of, (x, y)) else {
        return false;
    };
    if func.ty(p) == Type::I64
        && let Some((p_pos, p_inst)) = local_single(plan, info, b_idx, pos, p)
        && let Op::Binary(Binary::Iadd, px, py) = func.insts[p_inst.index()].op
    {
        return try_sib_sum(func, plan, info, (b_idx, pos, p_pos, res), (px, py, Some((p, disp))));
    }
    false
}

fn trivial_mul_imm(_func: &Function, plan: &Plan<Reg>, ty: Type, val: Value) -> bool {
    let Some(raw) = plan.const_of(val) else {
        return false;
    };
    let uimm = crate::fold::extend(raw, ty, false) as u64;
    let simm = crate::fold::extend(raw, ty, true);
    let max_shift = if ty == Type::I64 { 63 } else { 31 };
    uimm <= 1
        || simm == -1
        || uimm.is_power_of_two()
        || (uimm - 1).is_power_of_two()
        || uimm.checked_add(1).is_some_and(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
        || simm
            .checked_neg()
            .filter(|&n| n > 2)
            .and_then(|n| (n as u64).checked_add(1))
            .is_some_and(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
}

pub(super) fn try_fuse_madd(
    func: &Function,
    plan: &mut Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (alu_op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(alu_op, Binary::Iadd | Binary::Isub) {
        return false;
    }
    let ty = func.ty(res);
    let candidates = if alu_op == Binary::Isub { [Some((y, x)), None] } else { [Some((y, x)), Some((x, y))] };
    for (mul, acc) in candidates.into_iter().flatten() {
        if plan.elided[acc.index()] {
            continue;
        }
        let Some((mpos, minst)) = local_single(plan, info, b_idx, pos, mul) else {
            continue;
        };
        let Op::Binary(Binary::Imul, a, b) = func.insts[minst.index()].op else {
            continue;
        };
        if trivial_mul_imm(func, plan, ty, a) || trivial_mul_imm(func, plan, ty, b) {
            continue;
        }
        plan.elided[mul.index()] = true;
        let ok = preserved(func, plan, b_idx, mpos, pos, None, a) && preserved(func, plan, b_idx, mpos, pos, None, b);
        if ok {
            return true;
        }
        plan.elided[mul.index()] = false;
    }
    false
}

/// Returns `(sub, a, b, acc)` when `x` and `y` under `op` emit `madd`/`msub` with an elided `Imul`.
pub(in super::super) fn madd_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    x: Value,
    y: Value,
    def: impl Fn(Value) -> Option<Inst>,
) -> Option<(bool, Value, Value, Value)> {
    if !matches!(op, Binary::Iadd | Binary::Isub) || !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let sub = op == Binary::Isub;
    let candidates = if sub { [Some((y, x)), None] } else { [Some((y, x)), Some((x, y))] };
    for (mul, acc) in candidates.into_iter().flatten() {
        if !plan.elided[mul.index()] || plan.elided[acc.index()] {
            continue;
        }
        let Some(inst) = def(mul) else { continue };
        if let Op::Binary(Binary::Imul, a, b) = func.insts[inst.index()].op
            && !plan.elided[a.index()]
            && !plan.elided[b.index()]
            && !trivial_mul_imm(func, plan, ty, a)
            && !trivial_mul_imm(func, plan, ty, b)
        {
            return Some((sub, a, b, acc));
        }
    }
    None
}

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

pub(in super::super) fn splat_pattern64(lane: Lane, imm: i64) -> u64 {
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

pub(in super::super) fn encode_neon_movi(q: u32, pattern: u64) -> Option<u32> {
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

fn try_fuse_splat_shuffle(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    res: Value,
    (a, b, m): (Value, Value, [u8; 16]),
) {
    let Type::Vec { lane, count } = func.ty(res) else { return };
    let bytes = lane.bytes() * u32::from(count);
    let count_u = usize::from(count);
    let first = m[0];
    if !matches!(bytes, 8 | 16) || usize::from(first) >= 2 * count_u || !m[..count_u].iter().all(|&x| x == first) {
        return;
    }
    let src = if usize::from(first) < count_u { a } else { b };
    if uses[src.index()] != 1 || plan.elided[src.index()] {
        return;
    }
    let Some((_, _, ins_inst)) = pos_of[src.index()] else { return };
    if let Op::Insert(_, scalar, idx) = func.insts[ins_inst.index()].op
        && usize::from(idx) == usize::from(first) % count_u
        && let Some(imm) = plan.const_of(scalar)
        && encode_neon_movi(u32::from(bytes == 16), splat_pattern64(lane, imm)).is_some()
    {
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

fn is_mem_barrier(op: &Op) -> bool {
    matches!(
        op,
        Op::Store(..)
            | Op::StackStore(..)
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Fence
            | Op::Copy(..)
            | Op::Move(..)
            | Op::Trap
            | Op::Trapz(..)
    )
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
            let (base, add_imm, addr_elide) = unwrap_iadd_const(func, plan, ctx, (b_idx, lpos), addr);
            if plan.elided[base.index()] || plan.sib[base.index()].is_some() {
                return None;
            }
            let eff = i64::from(off).checked_add(add_imm).and_then(|v| i32::try_from(v).ok())?;
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
    let Some(lo) = load_pair_half(func, plan, ctx, (b_idx, pos), lo_val) else { return false };
    let Some(hi) = load_pair_half(func, plan, ctx, (b_idx, pos), hi_val) else { return false };
    let same_anchor = (lo.slot.is_some() && lo.slot == hi.slot) || (lo.base.is_some() && lo.base == hi.base);
    if !same_anchor || i64::from(lo.off).checked_add(8) != Some(i64::from(hi.off)) {
        return false;
    }
    let min_pos = lo.pos.min(hi.pos);
    if func.blocks[b_idx].insts[min_pos + 1..pos].iter().any(|&i| is_mem_barrier(&func.insts[i.index()].op)) {
        return false;
    }
    for v in [Some(ins0), Some(lo.val), Some(hi.val), lo.addr_elide, hi.addr_elide].into_iter().flatten() {
        plan.elided[v.index()] = true;
    }
    if let Some(base) = lo.base
        && !preserved(func, plan, b_idx, lo.base_pos.max(hi.base_pos), pos, None, base)
    {
        for v in [Some(ins0), Some(lo.val), Some(hi.val), lo.addr_elide, hi.addr_elide].into_iter().flatten() {
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
    let Type::Vec { lane, count: 2 } = func.ty(res) else { return };
    if !matches!(lane, Lane::I64 | Lane::F64) || lane_b >= 2 {
        return;
    }
    let Some((pos0, inst0)) = local_single(plan, ctx, b_idx, pos, ins0) else { return };
    let Op::Insert(_, val_a, lane_a) = func.insts[inst0.index()].op else { return };
    if lane_a + lane_b != 1 {
        return;
    }
    let (lo, hi) = if lane_a == 0 { (val_a, val_b) } else { (val_b, val_a) };
    if try_fuse_insert_loads(func, plan, ctx, (b_idx, pos, ins0), (lo, hi)) {
        return;
    }
    if plan.elided[val_a.index()] || plan.elided[val_b.index()] {
        return;
    }
    let ok = match lane {
        Lane::I64 => plan.const_of(val_a) == Some(0) || preserved(func, plan, b_idx, pos0, pos, None, val_a),
        Lane::F64 => pos0 + 1 == pos,
        _ => false,
    };
    if ok {
        plan.elided[ins0.index()] = true;
    }
}

pub(super) fn try_fuse_vec_op(
    func: &Function,
    plan: &mut Plan<Reg>,
    ctx: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    op: &Op,
) -> bool {
    match *op {
        Op::Shuffle(a, b, ShuffleMask::Const(m)) => try_fuse_splat_shuffle(func, plan, ctx, res, (a, b, m)),
        Op::Insert(vec, val, idx) => try_fuse_vec_insert(func, plan, ctx, (b_idx, pos, res), (vec, val, idx)),
        _ => return false,
    }
    true
}
