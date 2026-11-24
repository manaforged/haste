use super::super::{const_of, def_entry, def_inst, preserved, shift_parts};
use super::{PosMap, UseInfo, live_root_preserved};
use crate::alloc::Plan;
use crate::arm64::asm::Reg;
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Value};

fn extr_hi(
    func: &Function,
    plan: &Plan<Reg>,
    bits: u32,
    v: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
    extra_ok: &dyn Fn(Value) -> bool,
) -> Option<(Value, u32, Option<Value>)> {
    let (Binary::Ishl, a, ax) = shift_parts(&func.insts[def(v)?.index()].op, plan)? else {
        return None;
    };
    let kl = (ax as u32) & (bits - 1);
    if kl == 0 {
        return None;
    }
    let need_mask = (1_u64 << (bits - kl)) - 1;
    if extra_ok(a)
        && let Some(band) = def(a)
        && let Op::Binary(Binary::Band, x0, x1) = func.insts[band.index()].op
        && !plan.elided[x0.index()]
        && !plan.elided[x1.index()]
    {
        for (inner, m_val) in [(x0, x1), (x1, x0)] {
            if let Some(imm) = const_of(func, plan, m_val, def)
                && ((imm as u64) & need_mask) == need_mask
            {
                return Some((inner, kl, Some(a)));
            }
        }
    }
    Some((a, kl, None))
}

fn extr_lo(
    func: &Function,
    plan: &Plan<Reg>,
    bits: u32,
    v: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
    extra_ok: &dyn Fn(Value) -> bool,
) -> Option<(Value, u32, Option<Value>)> {
    let op = &func.insts[def(v)?.index()].op;
    if let Some((Binary::Ushr, b, ay)) = shift_parts(op, plan) {
        let kr = (ay as u32) & (bits - 1);
        return (kr > 0).then_some((b, kr, None));
    }
    let Op::Binary(Binary::Band, x0, x1) = *op else {
        return None;
    };
    for (s, m_val) in [(x0, x1), (x1, x0)] {
        if !extra_ok(s) {
            continue;
        }
        let Some(sinst) = def(s) else { continue };
        let Some((Binary::Ushr, inner, ay)) = shift_parts(&func.insts[sinst.index()].op, plan)
        else {
            continue;
        };
        let kr = (ay as u32) & (bits - 1);
        if kr == 0 {
            continue;
        }
        let need_mask = (1_u64 << (bits - kr)) - 1;
        if let Some(imm) = const_of(func, plan, m_val, def)
            && ((imm as u64) & need_mask) == need_mask
        {
            return Some((inner, kr, Some(s)));
        }
    }
    None
}

fn extr_match(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
    extra_ok: &dyn Fn(Value) -> bool,
) -> Option<(Value, Option<Value>, Value, Option<Value>, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let bits = ty.bits();
    for (hv, lv) in [(x, y), (y, x)] {
        if let Some((hi, kl, he)) = extr_hi(func, plan, bits, hv, def, extra_ok)
            && let Some((lo, kr, le)) = extr_lo(func, plan, bits, lv, def, extra_ok)
            && kl + kr == bits
        {
            return Some((hi, he, lo, le, kr as u8));
        }
    }
    None
}

pub(in crate::arm64) fn extr_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value, u8)> {
    let ok = |v: Value| plan.elided[v.index()];
    let (hi, _, lo, _, kr) = extr_match(func, plan, ty, (x, y), def, &ok)?;
    Some((hi, lo, kr))
}

fn extr_uext_ok(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosMap<'_>,
    (b_idx, pos, other_pos, branch_pos): (usize, usize, usize, usize),
    (leaf, has_extra, kr): (Value, bool, u32),
) -> Option<bool> {
    if !plan.elided[leaf.index()] {
        return None;
    }
    let (hb, hpos, hinst) = def_entry(pos_of, leaf)?;
    let Op::Convert(Convert::Uextend, inner) = func.insts[hinst.index()].op else {
        return None;
    };
    if hb != b_idx || kr > func.ty(inner).bits() {
        return None;
    }
    let start = if has_extra { branch_pos } else { hpos };
    let skip = has_extra.then_some(other_pos);
    Some(
        preserved(func, plan, b_idx, start, pos, skip, inner)
            && (has_extra || preserved(func, plan, b_idx, hpos, branch_pos, None, inner)),
    )
}

fn extr_branch_ok(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosMap<'_>,
    (b_idx, pos, other_pos, branch_pos): (usize, usize, usize, usize),
    (leaf, extra, limit): (Value, Option<Value>, Option<u32>),
) -> Option<bool> {
    if let Some(ev) = extra
        && !plan.elided[ev.index()]
    {
        let (eb, epos, _) = def_entry(pos_of, ev)?;
        if eb != b_idx || epos >= branch_pos {
            return None;
        }
        return (live_root_preserved(
            func,
            plan,
            pos_of,
            (b_idx, epos, branch_pos, Some(other_pos), None),
            leaf,
        ) && preserved(func, plan, b_idx, branch_pos, pos, Some(other_pos), leaf))
        .then_some(true);
    }
    if let Some(kr) = limit
        && let Some(ok) = extr_uext_ok(
            func,
            plan,
            pos_of,
            (b_idx, pos, other_pos, branch_pos),
            (leaf, extra.is_some(), kr),
        )
    {
        return ok.then_some(false);
    }
    live_root_preserved(
        func,
        plan,
        pos_of,
        (b_idx, branch_pos, pos, Some(other_pos), None),
        leaf,
    )
    .then_some(false)
}

fn extr_pair_positions(
    pos_of: PosMap<'_>,
    b_idx: usize,
    pos: usize,
    (x, y): (Value, Value),
) -> Option<(usize, usize)> {
    let (xb, xpos, _) = def_entry(pos_of, x)?;
    let (yb, ypos, _) = def_entry(pos_of, y)?;
    (xb == b_idx && yb == b_idx && xpos < pos && ypos < pos).then_some((xpos, ypos))
}

pub(in crate::arm64::fuse) fn try_fuse_extr(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
        || uses[x.index()] != 1
        || uses[y.index()] != 1
        || plan.elided[x.index()]
        || plan.elided[y.index()]
    {
        return false;
    }
    let Some((xpos, ypos)) = extr_pair_positions(pos_of, b_idx, pos, (x, y)) else {
        return false;
    };
    let def = |v| def_inst(pos_of, v);
    let extra_ok = |v: Value| plan.elided[v.index()] || uses[v.index()] == 1;
    let Some((hi, he, lo, le, kr)) = extr_match(func, plan, func.ty(res), (x, y), &def, &extra_ok)
    else {
        return false;
    };
    let (hpos, lpos) = match extr_hi(func, plan, func.ty(res).bits(), x, &def, &extra_ok) {
        Some((cand, _, _)) if cand == hi => (xpos, ypos),
        _ => (ypos, xpos),
    };
    let Some(el_he) = extr_branch_ok(
        func,
        plan,
        pos_of,
        (b_idx, pos, lpos, hpos),
        (hi, he, Some(u32::from(kr))),
    ) else {
        return false;
    };
    let Some(el_le) = extr_branch_ok(func, plan, pos_of, (b_idx, pos, hpos, lpos), (lo, le, None))
    else {
        return false;
    };
    plan.elided[x.index()] = true;
    plan.elided[y.index()] = true;
    if el_he && let Some(ev) = he {
        plan.elided[ev.index()] = true;
    }
    if el_le && let Some(ev) = le {
        plan.elided[ev.index()] = true;
    }
    true
}
