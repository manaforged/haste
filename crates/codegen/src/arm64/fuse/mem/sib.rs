use super::super::{
    const_of, def_entry, def_inst, ext_band_of, local_single, preserved, shift_parts, zero_ext_loc,
};
use crate::alloc::{Location, Plan};
use crate::arm64::asm::{ExtOp, Reg, encode_sib};
use crate::ir::{Binary, Convert, Function, Inst, Op, StackSlot, Type, Value};
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
    let min_align = if data.size >= 8 {
        8
    } else if data.size >= 4 {
        4
    } else if data.size >= 2 {
        2
    } else {
        1
    };
    (1_u32 << data.align_shift.min(4)).max(min_align)
}

fn sib_use_ok(func: &Function, cand: &SibCand, expected: u32, ty: Type, off: i32) -> bool {
    let Some(total) = cand.disp.checked_add(off) else {
        return false;
    };
    if ty == Type::I128 {
        return false;
    }
    let bytes = ty.bytes();
    let scale_ok = cand.shift == 0 || bytes == (1 << cand.shift);
    match cand.base {
        SibBase::Value(_) => {
            (total == 0 && scale_ok)
                || (expected == 1 && cand.folded_const && mem_imm_fits(total, bytes))
        }
        SibBase::Frame(slot) => {
            expected == 1
                && (-65536..=65536).contains(&total)
                && slot_align_bytes(func, slot) >= bytes
                && total % (bytes as i32) == 0
        }
    }
}

fn sib_uses_span(
    func: &Function,
    (b_idx, pos, res): (usize, usize, Value),
    cand: &SibCand,
    expected: u32,
) -> Option<usize> {
    let (mut seen, mut last_pos) = (0_u32, pos);
    for (u_pos, &u_inst) in func.blocks[b_idx].insts.iter().enumerate().skip(pos + 1) {
        let u_data = &func.insts[u_inst.index()];
        let hit = match (&u_data.op, u_data.result) {
            (&Op::Load(addr, off), Some(r)) if addr == res => Some((func.ty(r), off)),
            (&Op::Store(val, addr, off), _) if addr == res && val != res => {
                Some((func.ty(val), off))
            }
            (&Op::AtomicLoad(addr, ord), Some(r)) if !ord.has_acquire() && addr == res => {
                Some((func.ty(r), 0))
            }
            (&Op::AtomicStore(val, addr, ord), _)
                if !ord.has_release() && addr == res && val != res =>
            {
                Some((func.ty(val), 0))
            }
            _ => None,
        };
        if hit.is_some_and(|(t, off)| sib_use_ok(func, cand, expected, t, off)) {
            seen += 1;
            last_pos = u_pos;
        }
    }
    (seen == expected).then_some(last_pos)
}

fn sib_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    (b_idx, last_pos): (usize, usize),
    cand: &SibCand,
) -> bool {
    let shift_clobbers = cand.elide.into_iter().flatten().any(|kept| {
        kept != cand.idx && !plan.elided[kept.index()] && plan.reg_of(kept) == plan.reg_of(cand.idx)
    });
    if shift_clobbers || !preserved(func, plan, b_idx, cand.idx_pos, last_pos, None, cand.idx) {
        return false;
    }
    match cand.base {
        SibBase::Frame(_) => true,
        SibBase::Value(b) => preserved(func, plan, b_idx, cand.base_pos, last_pos, None, b),
    }
}

fn try_commit_sib(
    func: &Function,
    plan: &mut Plan<Reg>,
    uses: &[u32],
    (b_idx, pos, res): (usize, usize, Value),
    cand: SibCand,
) -> bool {
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
        plan.sib[res.index()] = Some(Sib {
            base: cand.base,
            index: cand.idx,
            shift: encode_sib(cand.ext, cand.shift as u8),
            disp: cand.disp,
        });
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
        Op::Convert(Convert::Sextend, inner)
            if func.ty(inner) == Type::I32 && !plan.elided[inner.index()] =>
        {
            Some((inner, ExtOp::Sxtw))
        }
        Op::Convert(Convert::Uextend, inner)
            if !plan.elided[inner.index()] && func.ty(inner).is_small_int() =>
        {
            if func.ty(inner) == Type::I32 {
                Some((
                    inner,
                    if zero_ext_loc(plan, inner) {
                        ExtOp::Uxtx
                    } else {
                        ExtOp::Uxtw
                    },
                ))
            } else {
                zero_ext_loc(plan, inner).then_some((inner, ExtOp::Uxtx))
            }
        }
        Op::Binary(Binary::Band, a, b) => {
            let def = |u| def_inst(pos_of, u);
            let (ext, inner) = ext_band_of(func, plan, Type::I64, (a, b), &def)?;
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
    let (sb, epos, einst) = def_entry(pos_of, v)?;
    if sb != b_idx || epos >= limit {
        return None;
    }
    let (inner, ext) = unwrap_sib_op(func, plan, pos_of, &func.insts[einst.index()].op)?;
    Some((inner, epos, ext, Some(v)))
}

fn parse_iadd_const(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (x, y): (Value, Value),
) -> Option<(Value, i32)> {
    let def = |v| def_inst(pos_of, v);
    for (val, c) in [(x, y), (y, x)] {
        if let Some(imm) = const_of(func, plan, c, def).and_then(|v| i32::try_from(v).ok()) {
            return Some((val, imm));
        }
    }
    None
}

pub(super) fn parse_sib_base_op(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    op: &Op,
) -> Option<(Value, i32)> {
    match *op {
        Op::Binary(Binary::Iadd, x, y) => parse_iadd_const(func, plan, pos_of, (x, y)),
        Op::Binary(Binary::Isub, real, c) => {
            let def = |v| def_inst(pos_of, v);
            const_of(func, plan, c, def)
                .and_then(|v| v.checked_neg())
                .and_then(|v| i32::try_from(v).ok())
                .map(|imm| (real, imm))
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
    (plan.reg_of(v).is_some() && !plan.elided[v.index()]).then_some((
        SibBase::Value(v),
        limit,
        0,
        false,
        None,
    ))
}

fn parse_sib_index(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (uses, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit, allow_elided): (usize, usize, bool),
    rhs: Value,
) -> Option<(Value, ExtOp, i64, usize, [Option<Value>; 2])> {
    if let Some((sb, spos, sinst)) = def_entry(pos_of, rhs)
        && sb == b_idx
        && spos < limit
        && let Some((Binary::Ishl, mid, shift)) = shift_parts(&func.insts[sinst.index()].op, plan)
        && (1..=3).contains(&shift)
    {
        let usable = uses[rhs.index()] == 1 && (allow_elided || !plan.elided[rhs.index()]);
        if usable
            && let Some((idx, idx_pos, ext, uext)) =
                unwrap_sib_ext(func, plan, info, (b_idx, spos), (mid, true))
        {
            return Some((idx, ext, shift, idx_pos, [Some(rhs), uext]));
        }
        if !plan.elided[mid.index()] {
            return Some((mid, ExtOp::Uxtx, shift, spos, [Some(rhs), None]));
        }
    }
    let (idx, idx_pos, ext, uext) =
        unwrap_sib_ext(func, plan, info, (b_idx, limit), (rhs, allow_elided))?;
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
    } else if let Some(parsed) =
        parse_sib_index(func, plan, info, (b_idx, sum_pos, outer.is_some()), rhs)
    {
        parsed
    } else {
        return false;
    };
    let Some((base, base_pos, b_disp, b_folded, b_elide)) =
        parse_sib_base(func, plan, info, (b_idx, sum_pos), base_val)
    else {
        return false;
    };
    let Some(disp) = b_disp.checked_add(outer_disp) else {
        return false;
    };
    let cand = SibCand {
        base,
        base_pos,
        idx,
        ext,
        shift,
        idx_pos,
        disp,
        folded_const: b_folded || outer.is_some(),
        elide: [e0, e1, b_elide, outer_elide],
    };
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
            if try_sib_arm(
                func,
                plan,
                info,
                (b_idx, res_pos, sum_pos, res),
                (b_arm, i_arm, outer, plain),
            ) {
                return true;
            }
        }
    }
    false
}

pub(in crate::arm64::fuse) fn try_fuse_sib(
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
        return try_sib_sum(
            func,
            plan,
            info,
            (b_idx, pos, p_pos, res),
            (px, py, Some((p, disp))),
        );
    }
    false
}
