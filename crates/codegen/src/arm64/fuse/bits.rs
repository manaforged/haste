use super::super::asm::Reg;
use super::{can_peel_uextend, const_of, local_single, preserved, shift_parts, zero_ext_loc};
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, Type, Value};

pub(in crate::arm64) type PosMap<'a> = &'a [Option<(usize, usize, Inst)>];
pub(in crate::arm64) type UseInfo<'a> = (&'a [u32], PosMap<'a>);

#[derive(Clone, Copy)]
pub(in crate::arm64) struct Bfx {
    pub insert: bool,
    pub signed: bool,
    pub src: Value,
    pub lsb: u8,
    pub width: u8,
}

fn low_mask_width(m: u64, ty_mask: u64) -> Option<u32> {
    (m != 0 && m != ty_mask && (m & (m + 1)) == 0).then(|| m.trailing_ones())
}

fn bfx_band(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
    mid_ok: &dyn Fn(Value) -> bool,
) -> Option<(Bfx, Value, Option<Value>)> {
    let (bits, ty_mask) = (ty.bits(), ty.unsigned_max());
    for (mid, m_val) in [(x, y), (y, x)] {
        if !mid_ok(mid) {
            continue;
        }
        let Some(m) = const_of(func, plan, m_val, def).map(|imm| (imm as u64) & ty_mask) else {
            continue;
        };
        match func.insts[def(mid)?.index()].op {
            Op::Binary(sop @ (Binary::Ushr | Binary::Sshr), src, amt_val) => {
                if let Some(w) = low_mask_width(m, ty_mask)
                    && let Some(amt) = const_of(func, plan, amt_val, def)
                {
                    let lsb = (amt as u32) & (bits - 1);
                    if sop == Binary::Ushr || w <= bits - lsb {
                        let width = w.min(bits - lsb) as u8;
                        return Some((Bfx { insert: false, signed: false, src, lsb: lsb as u8, width }, mid, None));
                    }
                }
            }
            Op::Binary(Binary::Ishl, src, amt_val) => {
                if let Some(amt) = const_of(func, plan, amt_val, def) {
                    let lsb = (amt as u32) & (bits - 1);
                    let eff = (m | ((1_u64 << lsb) - 1)) & ty_mask;
                    let top = if eff == ty_mask { Some(bits) } else { low_mask_width(eff, ty_mask) };
                    if lsb > 0 && let Some(top) = top && top > lsb {
                        return Some((Bfx { insert: true, signed: false, src, lsb: lsb as u8, width: (top - lsb) as u8 }, mid, None));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn bfx_shr_ishl(
    (bits, op, shr_amt): (u32, Binary, u32),
    (mid, src, top): (Value, Value, u32),
) -> Option<(Bfx, Value, Option<Value>)> {
    if shr_amt >= top && (shr_amt > 0 || top > 0) {
        return Some((Bfx { insert: false, signed: op == Binary::Sshr, src, lsb: (shr_amt - top) as u8, width: (bits - shr_amt) as u8 }, mid, None));
    }
    (shr_amt > 0 && top > shr_amt)
        .then(|| (Bfx { insert: true, signed: op == Binary::Sshr, src, lsb: (top - shr_amt) as u8, width: (bits - top) as u8 }, mid, None))
}

fn bfx_shr(
    func: &Function,
    plan: &Plan<Reg>,
    (ty, op): (Type, Binary),
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
    mid_ok: &dyn Fn(Value) -> bool,
) -> Option<(Bfx, Value, Option<Value>)> {
    if !mid_ok(x) {
        return None;
    }
    let (bits, ty_mask) = (ty.bits(), ty.unsigned_max());
    let shr_amt = (const_of(func, plan, y, def)? as u32) & (bits - 1);
    match func.insts[def(x)?.index()].op {
        Op::Binary(Binary::Band, a, b) if op == Binary::Ushr => [(a, b), (b, a)].into_iter().find_map(|(src, m_val)| {
            let imm = const_of(func, plan, m_val, def)?;
            let w = low_mask_width(((imm as u64) & ty_mask) >> shr_amt, ty_mask)?;
            Some((Bfx { insert: false, signed: false, src, lsb: shr_amt as u8, width: w as u8 }, x, None))
        }),
        Op::Binary(Binary::Ishl, src, l_amt_val) => {
            let top = (const_of(func, plan, l_amt_val, def)? as u32) & (bits - 1);
            bfx_shr_ishl((bits, op, shr_amt), (x, src, top))
        }
        Op::Convert(Convert::Sextend, inner) if op == Binary::Sshr && func.ty(inner).is_small_int() && func.ty(inner).bits() < bits => {
            let ib = func.ty(inner).bits();
            let lsb = shr_amt.min(ib - 1);
            Some((Bfx { insert: false, signed: true, src: inner, lsb: lsb as u8, width: (ib - lsb) as u8 }, x, None))
        }
        _ => None,
    }
}

fn band_mask_src(
    func: &Function,
    plan: &Plan<Reg>,
    band_val: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u32)> {
    let Op::Binary(Binary::Band, a, b) = func.insts[def(band_val)?.index()].op else {
        return None;
    };
    if plan.elided[a.index()] || plan.elided[b.index()] {
        return None;
    }
    let mty = func.ty(band_val);
    let mask = mty.unsigned_max();
    [(a, b), (b, a)].into_iter().find_map(|(src, m_val)| {
        let imm = const_of(func, plan, m_val, def)?;
        let w = low_mask_width((imm as u64) & mask, mask)?;
        Some((src, w))
    })
}

fn bfx_shl(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
    hooks: (&dyn Fn(Value) -> bool, &dyn Fn(Value) -> bool),
) -> Option<(Bfx, Value, Option<Value>)> {
    let (mid_ok, inner_ok) = hooks;
    if !mid_ok(x) {
        return None;
    }
    let (bits, ty_mask) = (ty.bits(), ty.unsigned_max());
    let shl_amt = (const_of(func, plan, y, def)? as u32) & (bits - 1);
    if shl_amt == 0 {
        return None;
    }
    match func.insts[def(x)?.index()].op {
        Op::Binary(Binary::Band, a, b) => [(a, b), (b, a)].into_iter().find_map(|(src, m_val)| {
            let imm = const_of(func, plan, m_val, def)?;
            let w = low_mask_width((imm as u64) & ty_mask, ty_mask)?;
            Some((Bfx { insert: true, signed: false, src, lsb: shl_amt as u8, width: w.min(bits - shl_amt) as u8 }, x, None))
        }),
        Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner) if func.ty(inner).is_small_int() && func.ty(inner).bits() < bits => {
            if conv == Convert::Uextend && inner_ok(inner) && let Some((src, w)) = band_mask_src(func, plan, inner, def) {
                return Some((Bfx { insert: true, signed: false, src, lsb: shl_amt as u8, width: w.min(bits - shl_amt) as u8 }, x, Some(inner)));
            }
            (!plan.elided[inner.index()]).then(|| {
                (Bfx { insert: true, signed: conv == Convert::Sextend, src: inner, lsb: shl_amt as u8, width: func.ty(inner).bits().min(bits - shl_amt) as u8 }, x, None)
            })
        }
        _ => None,
    }
}

fn bfx_match(
    func: &Function,
    plan: &Plan<Reg>,
    (ty, op): (Type, Binary),
    pair: (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
    hooks: (&dyn Fn(Value) -> bool, &dyn Fn(Value) -> bool),
) -> Option<(Bfx, Value, Option<Value>)> {
    if !ty.is_small_int() {
        return None;
    }
    let (mid_ok, inner_ok) = hooks;
    match op {
        Binary::Band => bfx_band(func, plan, ty, pair, def, mid_ok),
        Binary::Ushr | Binary::Sshr => bfx_shr(func, plan, (ty, op), pair, def, mid_ok),
        Binary::Ishl => bfx_shl(func, plan, ty, pair, def, (mid_ok, inner_ok)),
        _ => None,
    }
}

pub(in crate::arm64) fn bfx_parts(func: &Function, plan: &Plan<Reg>, ty: Type, op: Binary, (x, y): (Value, Value), def: &dyn Fn(Value) -> Option<Inst>) -> Option<Bfx> {
    let ok = |v: Value| plan.elided[v.index()];
    bfx_match(func, plan, (ty, op), (x, y), def, (&ok, &ok)).map(|(bfx, _, _)| bfx)
}

fn check_bfx_src(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, pos, res, is_conv): (usize, usize, Value, bool),
    (spos, extra, bfx): (usize, Option<Value>, Bfx),
) -> Option<bool> {
    let need = if bfx.insert { u32::from(bfx.width) } else { u32::from(bfx.lsb) + u32::from(bfx.width) };
    let ext_ok = |inner: Value| need <= func.ty(inner).bits() || zero_ext_loc(plan, inner);
    if let Some(band_v) = extra {
        let (_, bpos, _) = pos_of[band_v.index()]?;
        return (bfx.src != res && preserved(func, plan, b_idx, bpos, pos, Some(spos), bfx.src)).then_some(false);
    }
    let peeled = !is_conv && can_peel_uextend(func, plan, info, (b_idx, spos, pos, Some(spos), res), (bfx.src, &ext_ok));
    if !peeled && !preserved(func, plan, b_idx, spos, pos, None, bfx.src) {
        return None;
    }
    Some(peeled)
}

pub(super) fn try_fuse_bfx(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, pos_of): UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    let mid_ok = |v: Value| uses[v.index()] == 1 && !plan.elided[v.index()] && pos_of[v.index()].is_some_and(|(sb, spos, _)| sb == b_idx && spos < pos);
    let inner_ok = |v: Value| uses[v.index()] == 1 && pos_of[v.index()].is_some_and(|(sb, spos, _)| sb == b_idx && spos < pos);
    let Some((bfx, mid, extra)) = bfx_match(func, plan, (func.ty(res), op), (x, y), &def, (&mid_ok, &inner_ok)) else {
        return false;
    };
    let Some((_, spos, min_inst)) = pos_of[mid.index()] else { return false };
    let is_conv = matches!(func.insts[min_inst.index()].op, Op::Convert(..));
    let Some(peeled) = check_bfx_src(func, plan, info, (b_idx, pos, res, is_conv), (spos, extra, bfx)) else {
        return false;
    };
    plan.elided[mid.index()] = true;
    if let Some(band_v) = extra {
        plan.elided[band_v.index()] = true;
    }
    if peeled {
        plan.elided[bfx.src.index()] = true;
    }
    true
}

fn conv_bfx_shift(
    func: &Function,
    plan: &Plan<Reg>,
    (src_bits, signed): (u32, bool),
    op: &Op,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<Bfx> {
    let (sop, src, amt_val) = match *op {
        Op::Binary(sop @ (Binary::Ushr | Binary::Sshr | Binary::Ishl), src, amt_val) => (sop, src, amt_val),
        _ => return None,
    };
    let lsb = (const_of(func, plan, amt_val, def)? as u32) & (src_bits - 1);
    if lsb == 0 {
        return None;
    }
    let width = (src_bits - lsb) as u8;
    match (sop, signed) {
        (Binary::Ushr, false) => Some(Bfx { insert: false, signed: false, src, lsb: lsb as u8, width }),
        (Binary::Sshr, true) => Some(Bfx { insert: false, signed: true, src, lsb: lsb as u8, width }),
        (Binary::Ishl, _) => Some(Bfx { insert: true, signed, src, lsb: lsb as u8, width }),
        _ => None,
    }
}

fn adapt_conv_bfx(mut bfx: Bfx, src_bits: u32, signed: bool) -> Option<Bfx> {
    if bfx.signed {
        return signed.then_some(bfx);
    }
    if bfx.insert && u32::from(bfx.lsb) + u32::from(bfx.width) == src_bits && signed {
        bfx.signed = true;
    }
    Some(bfx)
}

fn conv_bfx_band(
    func: &Function,
    plan: &Plan<Reg>,
    (src_ty, signed): (Type, bool),
    (x, a, b): (Value, Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<Bfx> {
    if let Some((src, w)) = band_mask_src(func, plan, x, def) {
        return Some(Bfx { insert: false, signed: false, src, lsb: 0, width: w as u8 });
    }
    let ok = |v: Value| plan.elided[v.index()];
    let (bfx, _, _) = bfx_band(func, plan, src_ty, (a, b), def, &ok)?;
    adapt_conv_bfx(bfx, src_ty.bits(), signed)
}

fn conv_bfx_match(
    func: &Function,
    plan: &Plan<Reg>,
    (dst_ty, conv): (Type, Convert),
    x: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
    mid_ok: &dyn Fn(Value) -> bool,
) -> Option<Bfx> {
    if !matches!(dst_ty, Type::I16 | Type::I32 | Type::I64) || !mid_ok(x) {
        return None;
    }
    let src_ty = func.ty(x);
    if !src_ty.is_small_int() || src_ty.bits() >= dst_ty.bits() {
        return None;
    }
    let signed = conv == Convert::Sextend;
    let op = &func.insts[def(x)?.index()].op;
    if let Op::Binary(Binary::Band, a, b) = *op {
        if let Some(bfx) = conv_bfx_band(func, plan, (src_ty, signed), (x, a, b), def) {
            return Some(bfx);
        }
    }
    if let Op::Binary(bop @ (Binary::Ushr | Binary::Sshr | Binary::Ishl), a, b) = *op {
        let ok = |v: Value| plan.elided[v.index()];
        if let Some((bfx, _, _)) = bfx_match(func, plan, (src_ty, bop), (a, b), def, (&ok, &ok))
            && let Some(out) = adapt_conv_bfx(bfx, src_ty.bits(), signed)
        {
            return Some(out);
        }
    }
    conv_bfx_shift(func, plan, (src_ty.bits(), signed), op, def)
}

pub(in crate::arm64) fn conv_bfx_parts(func: &Function, plan: &Plan<Reg>, (dst_ty, conv): (Type, Convert), x: Value, def: &dyn Fn(Value) -> Option<Inst>) -> Option<Bfx> {
    let ok = |v: Value| plan.elided[v.index()];
    conv_bfx_match(func, plan, (dst_ty, conv), x, def, &ok)
}

fn live_root_preserved(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosMap<'_>,
    (b_idx, start, end, skip, res): (usize, usize, usize, Option<usize>, Option<Value>),
    val: Value,
) -> bool {
    let mut cur = val;
    while plan.elided[cur.index()] {
        let Some((_, _, inst)) = pos_of[cur.index()] else { return false };
        let Op::Convert(Convert::Uextend, inner) = func.insts[inst.index()].op else { return false };
        cur = inner;
    }
    res != Some(cur) && preserved(func, plan, b_idx, start, end, skip, cur)
}

pub(super) fn try_fuse_conv_bfx(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (conv, x): (Convert, Value),
) -> bool {
    let Some((xpos, _)) = local_single(plan, info, b_idx, pos, x) else {
        return false;
    };
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    let ok = |v: Value| v == x;
    let Some(bfx) = conv_bfx_match(func, plan, (func.ty(res), conv), x, &def, &ok) else {
        return false;
    };
    let is_uext = !bfx.signed;
    let need = if bfx.insert { u32::from(bfx.width) } else { u32::from(bfx.lsb) + u32::from(bfx.width) };
    let ext_ok = |inner: Value| need <= func.ty(inner).bits() || zero_ext_loc(plan, inner);
    let peeled = is_uext && can_peel_uextend(func, plan, info, (b_idx, xpos, pos, Some(xpos), res), (bfx.src, &ext_ok));
    if !peeled && !live_root_preserved(func, plan, pos_of, (b_idx, xpos, pos, None, Some(res)), bfx.src) {
        return false;
    }
    plan.elided[x.index()] = true;
    if peeled {
        plan.elided[bfx.src.index()] = true;
    }
    true
}

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
            if let Some(imm) = const_of(func, plan, m_val, def) && ((imm as u64) & need_mask) == need_mask {
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
    let Op::Binary(Binary::Band, x0, x1) = *op else { return None };
    for (s, m_val) in [(x0, x1), (x1, x0)] {
        if !extra_ok(s) {
            continue;
        }
        let Some(sinst) = def(s) else { continue };
        let Some((Binary::Ushr, inner, ay)) = shift_parts(&func.insts[sinst.index()].op, plan) else {
            continue;
        };
        let kr = (ay as u32) & (bits - 1);
        if kr == 0 {
            continue;
        }
        let need_mask = (1_u64 << (bits - kr)) - 1;
        if let Some(imm) = const_of(func, plan, m_val, def) && ((imm as u64) & need_mask) == need_mask {
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

pub(in crate::arm64) fn extr_parts(func: &Function, plan: &Plan<Reg>, ty: Type, (x, y): (Value, Value), def: &dyn Fn(Value) -> Option<Inst>) -> Option<(Value, Value, u8)> {
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
    let (hb, hpos, hinst) = pos_of[leaf.index()]?;
    let Op::Convert(Convert::Uextend, inner) = func.insts[hinst.index()].op else {
        return None;
    };
    if hb != b_idx || kr > func.ty(inner).bits() {
        return None;
    }
    let start = if has_extra { branch_pos } else { hpos };
    let skip = has_extra.then_some(other_pos);
    Some(preserved(func, plan, b_idx, start, pos, skip, inner) && (has_extra || preserved(func, plan, b_idx, hpos, branch_pos, None, inner)))
}

fn extr_branch_ok(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosMap<'_>,
    (b_idx, pos, other_pos, branch_pos): (usize, usize, usize, usize),
    (leaf, extra, limit): (Value, Option<Value>, Option<u32>),
) -> Option<bool> {
    if let Some(ev) = extra && !plan.elided[ev.index()] {
        let (eb, epos, _) = pos_of[ev.index()]?;
        if eb != b_idx || epos >= branch_pos {
            return None;
        }
        return (live_root_preserved(func, plan, pos_of, (b_idx, epos, branch_pos, Some(other_pos), None), leaf)
            && preserved(func, plan, b_idx, branch_pos, pos, Some(other_pos), leaf))
        .then_some(true);
    }
    if let Some(kr) = limit && let Some(ok) = extr_uext_ok(func, plan, pos_of, (b_idx, pos, other_pos, branch_pos), (leaf, extra.is_some(), kr)) {
        return ok.then_some(false);
    }
    live_root_preserved(func, plan, pos_of, (b_idx, branch_pos, pos, Some(other_pos), None), leaf).then_some(false)
}

fn extr_pair_positions(pos_of: PosMap<'_>, b_idx: usize, pos: usize, (x, y): (Value, Value)) -> Option<(usize, usize)> {
    let (xb, xpos, _) = pos_of[x.index()]?;
    let (yb, ypos, _) = pos_of[y.index()]?;
    (xb == b_idx && yb == b_idx && xpos < pos && ypos < pos).then_some((xpos, ypos))
}

pub(super) fn try_fuse_extr(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
        || uses[x.index()] != 1 || uses[y.index()] != 1 || plan.elided[x.index()] || plan.elided[y.index()]
    {
        return false;
    }
    let Some((xpos, ypos)) = extr_pair_positions(pos_of, b_idx, pos, (x, y)) else { return false };
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    let extra_ok = |v: Value| plan.elided[v.index()] || uses[v.index()] == 1;
    let Some((hi, he, lo, le, kr)) = extr_match(func, plan, func.ty(res), (x, y), &def, &extra_ok) else {
        return false;
    };
    let (hpos, lpos) = match extr_hi(func, plan, func.ty(res).bits(), x, &def, &extra_ok) {
        Some((cand, _, _)) if cand == hi => (xpos, ypos),
        _ => (ypos, xpos),
    };
    let Some(el_he) = extr_branch_ok(func, plan, pos_of, (b_idx, pos, lpos, hpos), (hi, he, Some(u32::from(kr)))) else {
        return false;
    };
    let Some(el_le) = extr_branch_ok(func, plan, pos_of, (b_idx, pos, hpos, lpos), (lo, le, None)) else {
        return false;
    };
    plan.elided[x.index()] = true;
    plan.elided[y.index()] = true;
    if el_he && let Some(ev) = he { plan.elided[ev.index()] = true; }
    if el_le && let Some(ev) = le { plan.elided[ev.index()] = true; }
    true
}

fn find_def(func: &Function, val: Value) -> Option<Inst> {
    func.blocks.iter().flat_map(|b| &b.insts).copied().find(|&i| func.insts[i.index()].result == Some(val))
}

fn func_const(func: &Function, plan: &Plan<Reg>, val: Value) -> Option<i64> {
    plan.const_of(val).or_else(|| match func.insts[find_def(func, val)?.index()].op {
        Op::Const(imm) => Some(imm),
        _ => None,
    })
}

fn peel_bit_op(func: &Function, plan: &Plan<Reg>, val: Value, bit: u8, def: &dyn Fn(Value) -> Option<Inst>) -> Option<(Value, u8)> {
    let ty = func.ty(val);
    let bits = ty.bits();
    let get_amt = |v: Value| plan.const_of(v).or_else(|| match func.insts[def(v)?.index()].op {
        Op::Const(imm) => Some(imm),
        _ => None,
    });
    match func.insts[def(val)?.index()].op {
        Op::Binary(Binary::Ushr, inner, amt) => {
            let k = (get_amt(amt)? as u32) & (bits - 1);
            (u32::from(bit) + k < bits).then_some((inner, bit + k as u8))
        }
        Op::Binary(Binary::Sshr, inner, amt) => {
            let k = (get_amt(amt)? as u32) & (bits - 1);
            Some((inner, (u32::from(bit) + k).min(bits - 1) as u8))
        }
        Op::Binary(Binary::Ishl, inner, amt) => {
            let k = (get_amt(amt)? as u32) & (bits - 1);
            (u32::from(bit) >= k).then_some((inner, bit - k as u8))
        }
        Op::Convert(Convert::Uextend, inner) => (u32::from(bit) < func.ty(inner).bits()).then_some((inner, bit)),
        Op::Convert(Convert::Sextend, inner) => Some((inner, bit.min(func.ty(inner).bits() as u8 - 1))),
        _ => None,
    }
}

pub(in crate::arm64) fn single_bit_raw(func: &Function, plan: &Plan<Reg>, x: Value, y: Value) -> Option<(Value, u8)> {
    for (val, mask_val) in [(x, y), (y, x)] {
        let ty = func.ty(val);
        if ty.is_small_int() && let Some(imm) = func_const(func, plan, mask_val) {
            let u = (imm as u64) & ty.unsigned_max();
            if u.is_power_of_two() {
                return Some((val, u.trailing_zeros() as u8));
            }
        }
    }
    None
}

pub(in crate::arm64) fn single_bit_mask(func: &Function, plan: &Plan<Reg>, x: Value, y: Value) -> Option<(Value, u8)> {
    let (mut val, mut bit) = single_bit_raw(func, plan, x, y)?;
    let def = |v: Value| find_def(func, v);
    while plan.elided[val.index()] {
        (val, bit) = peel_bit_op(func, plan, val, bit, &def)?;
    }
    Some((val, bit))
}

pub(super) fn can_elide_band_bit(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, mpos, end_pos, skip_pos): (usize, usize, usize, Option<usize>),
    (x, y): (Value, Value),
) -> Option<Option<Value>> {
    let (mut val, mut bit) = single_bit_raw(func, plan, x, y)?;
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    if plan.elided[val.index()] {
        while plan.elided[val.index()] {
            (val, bit) = peel_bit_op(func, plan, val, bit, &def)?;
        }
        return preserved(func, plan, b_idx, mpos, end_pos, skip_pos, val).then_some(None);
    }
    if let Some((spos, _)) = local_single(plan, info, b_idx, mpos, val)
        && let Some((inner, _)) = peel_bit_op(func, plan, val, bit, &def)
        && !plan.elided[inner.index()]
        && preserved(func, plan, b_idx, spos, mpos, None, inner)
        && preserved(func, plan, b_idx, mpos, end_pos, skip_pos, inner)
    {
        return Some(Some(val));
    }
    preserved(func, plan, b_idx, mpos, end_pos, skip_pos, val).then_some(None)
}

pub(super) fn can_elide_tst_band(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: PosMap<'_>,
    (b_idx, mpos, end_pos, skip_pos): (usize, usize, usize, Option<usize>),
    (x, y): (Value, Value),
) -> bool {
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return false;
    }
    if preserved(func, plan, b_idx, mpos, end_pos, skip_pos, x)
        && preserved(func, plan, b_idx, mpos, end_pos, skip_pos, y)
    {
        return true;
    }
    let def = |v: Value| pos_of[v.index()].map(|(_, _, i)| i);
    let ty_mask = ty.unsigned_max();
    for (val, mask_val) in [(x, y), (y, x)] {
        if plan.elided[val.index()]
            && let Some(imm) = const_of(func, plan, mask_val, &def).map(|v| (v as u64) & ty_mask)
            && let Some((root, eff)) = unwrap_elided_tst(func, plan, val, imm, &def)
            && super::super::int::encode_bitmask_imm(eff, ty == Type::I64).is_some()
        {
            return preserved(func, plan, b_idx, mpos, end_pos, skip_pos, root);
        }
    }
    false
}

pub(in crate::arm64) fn unwrap_elided_tst(
    func: &Function,
    plan: &Plan<Reg>,
    mut val: Value,
    mut imm: u64,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u64)> {
    while plan.elided[val.index()] {
        let ty = func.ty(val);
        let bits = ty.bits();
        match func.insts[def(val)?.index()].op {
            Op::Binary(Binary::Ushr, inner, amt) => {
                let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
                if imm.leading_zeros() < (64 - bits) + k {
                    return None;
                }
                val = inner;
                imm <<= k;
            }
            Op::Binary(Binary::Ishl, inner, amt) => {
                let k = (const_of(func, plan, amt, def)? as u32) & (bits - 1);
                if k > 0 && (imm & ((1_u64 << k) - 1)) != 0 {
                    return None;
                }
                val = inner;
                imm >>= k;
            }
            Op::Convert(Convert::Uextend, inner) => {
                if imm > func.ty(inner).unsigned_max() {
                    return None;
                }
                val = inner;
            }
            _ => return None,
        }
    }
    Some((val, imm))
}

fn check_branch_tst(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (uses, pos_of): UseInfo<'_>,
    (b_idx, pos, cond_val): (usize, usize, Value),
) -> Option<(Value, Option<Value>, Option<Value>)> {
    let (cpos, cinst) = local_single(plan, info, b_idx, pos, cond_val)?;
    if func.size() > 2000 && pos - cpos > 64 {
        return None;
    }
    match func.insts[cinst.index()].op {
        Op::Binary(Binary::Band, x, y) => {
            if let Some(extra) = can_elide_band_bit(func, plan, info, (b_idx, cpos, pos, None), (x, y)) {
                return Some((cond_val, None, extra));
            }
            can_elide_tst_band(func, plan, pos_of, (b_idx, cpos, pos, None), (x, y)).then_some((cond_val, None, None))
        }
        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
            let (m, mpos, x, y) = super::zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)?;
            if mpos >= cpos || (func.size() > 2000 && pos - mpos > 64) {
                return None;
            }
            if let Some(extra) = can_elide_band_bit(func, plan, info, (b_idx, mpos, pos, Some(cpos)), (x, y)) {
                return Some((cond_val, Some(m), extra));
            }
            can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, pos, Some(cpos)), (x, y)).then_some((cond_val, Some(m), None))
        }
        _ => None,
    }
}

type BranchTstHit = (Value, Option<Value>, Option<Value>);
type CmpTstHit = (usize, usize, usize, Value, Value);

fn scan_block_tst(
    func: &Function,
    plan: &Plan<Reg>,
    info: UseInfo<'_>,
    (b_idx, small): (usize, bool),
    (branches, cmps): (&mut Vec<BranchTstHit>, &mut Vec<CmpTstHit>),
) {
    let block = &func.blocks[b_idx];
    for (pos, &inst) in block.insts.iter().enumerate() {
        match func.insts[inst.index()].op {
            Op::Brif(cond_val, ..) | Op::Trapz(cond_val) => {
                if let Some(hit) = check_branch_tst(func, plan, info, (b_idx, pos, cond_val)) {
                    branches.push(hit);
                }
            }
            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                if let Some(res) = func.insts[inst.index()].result
                    && let Some(end_pos) = if !plan.elided[res.index()] {
                        Some(pos)
                    } else if small || block.insts.len() - pos <= 64 {
                        super::elided_use_end(func, plan, &block.insts, pos, res)
                    } else {
                        None
                    }
                {
                    cmps.push((b_idx, pos, end_pos, lhs, rhs));
                }
            }
            _ => {}
        }
    }
}

pub(super) fn fuse_tst(func: &Function, plan: &mut Plan<Reg>, info @ (uses, pos_of): UseInfo<'_>) {
    let small = func.size() <= 2000;
    let (mut branch_tst, mut cmp_tst) = (Vec::new(), Vec::new());
    for b_idx in 0..func.blocks.len() {
        scan_block_tst(func, plan, info, (b_idx, small), (&mut branch_tst, &mut cmp_tst));
    }
    for (cond_val, m, extra) in branch_tst {
        plan.elided[cond_val.index()] = true;
        if let Some(m) = m { plan.elided[m.index()] = true; }
        if let Some(e) = extra { plan.elided[e.index()] = true; }
    }
    for (b_idx, cpos, end_pos, lhs, rhs) in cmp_tst {
        if let Some((m, mpos, x, y)) = super::zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && (small || end_pos - mpos <= 64)
            && can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, end_pos, None), (x, y))
        {
            plan.elided[m.index()] = true;
        }
    }
}
