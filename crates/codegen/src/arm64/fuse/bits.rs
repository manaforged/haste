mod extr;
mod tst;

pub(in crate::arm64) use self::extr::extr_parts;
pub(super) use self::extr::try_fuse_extr;
pub(super) use self::tst::{can_elide_tst_band, fuse_tst, peel_elided};
pub(in crate::arm64) use self::tst::{single_bit_mask, single_bit_raw, unwrap_elided_tst};
use super::super::asm::Reg;
use super::{
    can_peel_uextend, const_of, def_entry, def_inst, local_single, preserved, zero_ext_loc,
};
use crate::alloc::Plan;
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Value};

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
                        return Some((
                            Bfx {
                                insert: false,
                                signed: false,
                                src,
                                lsb: lsb as u8,
                                width,
                            },
                            mid,
                            None,
                        ));
                    }
                }
            }
            Op::Binary(Binary::Ishl, src, amt_val) => {
                if let Some(amt) = const_of(func, plan, amt_val, def) {
                    let lsb = (amt as u32) & (bits - 1);
                    let eff = (m | ((1_u64 << lsb) - 1)) & ty_mask;
                    let top = if eff == ty_mask {
                        Some(bits)
                    } else {
                        low_mask_width(eff, ty_mask)
                    };
                    if lsb > 0
                        && let Some(top) = top
                        && top > lsb
                    {
                        return Some((
                            Bfx {
                                insert: true,
                                signed: false,
                                src,
                                lsb: lsb as u8,
                                width: (top - lsb) as u8,
                            },
                            mid,
                            None,
                        ));
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
        return Some((
            Bfx {
                insert: false,
                signed: op == Binary::Sshr,
                src,
                lsb: (shr_amt - top) as u8,
                width: (bits - shr_amt) as u8,
            },
            mid,
            None,
        ));
    }
    (shr_amt > 0 && top > shr_amt).then(|| {
        (
            Bfx {
                insert: true,
                signed: op == Binary::Sshr,
                src,
                lsb: (top - shr_amt) as u8,
                width: (bits - top) as u8,
            },
            mid,
            None,
        )
    })
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
        Op::Binary(Binary::Band, a, b) if op == Binary::Ushr => {
            [(a, b), (b, a)].into_iter().find_map(|(src, m_val)| {
                let imm = const_of(func, plan, m_val, def)?;
                let w = low_mask_width(((imm as u64) & ty_mask) >> shr_amt, ty_mask)?;
                Some((
                    Bfx {
                        insert: false,
                        signed: false,
                        src,
                        lsb: shr_amt as u8,
                        width: w as u8,
                    },
                    x,
                    None,
                ))
            })
        }
        Op::Binary(Binary::Ishl, src, l_amt_val) => {
            let top = (const_of(func, plan, l_amt_val, def)? as u32) & (bits - 1);
            bfx_shr_ishl((bits, op, shr_amt), (x, src, top))
        }
        Op::Convert(Convert::Sextend, inner)
            if op == Binary::Sshr
                && func.ty(inner).is_small_int()
                && func.ty(inner).bits() < bits =>
        {
            let ib = func.ty(inner).bits();
            let lsb = shr_amt.min(ib - 1);
            Some((
                Bfx {
                    insert: false,
                    signed: true,
                    src: inner,
                    lsb: lsb as u8,
                    width: (ib - lsb) as u8,
                },
                x,
                None,
            ))
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
            Some((
                Bfx {
                    insert: true,
                    signed: false,
                    src,
                    lsb: shl_amt as u8,
                    width: w.min(bits - shl_amt) as u8,
                },
                x,
                None,
            ))
        }),
        Op::Convert(conv @ (Convert::Uextend | Convert::Sextend), inner)
            if func.ty(inner).is_small_int() && func.ty(inner).bits() < bits =>
        {
            if conv == Convert::Uextend
                && inner_ok(inner)
                && let Some((src, w)) = band_mask_src(func, plan, inner, def)
            {
                return Some((
                    Bfx {
                        insert: true,
                        signed: false,
                        src,
                        lsb: shl_amt as u8,
                        width: w.min(bits - shl_amt) as u8,
                    },
                    x,
                    Some(inner),
                ));
            }
            (!plan.elided[inner.index()]).then(|| {
                (
                    Bfx {
                        insert: true,
                        signed: conv == Convert::Sextend,
                        src: inner,
                        lsb: shl_amt as u8,
                        width: func.ty(inner).bits().min(bits - shl_amt) as u8,
                    },
                    x,
                    None,
                )
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

pub(in crate::arm64) fn bfx_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    op: Binary,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<Bfx> {
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
    let need = if bfx.insert {
        u32::from(bfx.width)
    } else {
        u32::from(bfx.lsb) + u32::from(bfx.width)
    };
    let ext_ok = |inner: Value| need <= func.ty(inner).bits() || zero_ext_loc(plan, inner);
    if let Some(band_v) = extra {
        let (_, bpos, _) = def_entry(pos_of, band_v)?;
        return (bfx.src != res && preserved(func, plan, b_idx, bpos, pos, Some(spos), bfx.src))
            .then_some(false);
    }
    let peeled = !is_conv
        && can_peel_uextend(
            func,
            plan,
            info,
            (b_idx, spos, pos, Some(spos), res),
            (bfx.src, &ext_ok),
        );
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
    let def = |v| def_inst(pos_of, v);
    let mid_ok = |v: Value| {
        uses[v.index()] == 1
            && !plan.elided[v.index()]
            && def_entry(pos_of, v).is_some_and(|(sb, spos, _)| sb == b_idx && spos < pos)
    };
    let inner_ok = |v: Value| {
        uses[v.index()] == 1
            && def_entry(pos_of, v).is_some_and(|(sb, spos, _)| sb == b_idx && spos < pos)
    };
    let Some((bfx, mid, extra)) = bfx_match(
        func,
        plan,
        (func.ty(res), op),
        (x, y),
        &def,
        (&mid_ok, &inner_ok),
    ) else {
        return false;
    };
    let Some((_, spos, min_inst)) = def_entry(pos_of, mid) else {
        return false;
    };
    let is_conv = matches!(func.insts[min_inst.index()].op, Op::Convert(..));
    let Some(peeled) = check_bfx_src(
        func,
        plan,
        info,
        (b_idx, pos, res, is_conv),
        (spos, extra, bfx),
    ) else {
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
        Op::Binary(sop @ (Binary::Ushr | Binary::Sshr | Binary::Ishl), src, amt_val) => {
            (sop, src, amt_val)
        }
        _ => return None,
    };
    let lsb = (const_of(func, plan, amt_val, def)? as u32) & (src_bits - 1);
    if lsb == 0 {
        return None;
    }
    let width = (src_bits - lsb) as u8;
    match (sop, signed) {
        (Binary::Ushr, false) => Some(Bfx {
            insert: false,
            signed: false,
            src,
            lsb: lsb as u8,
            width,
        }),
        (Binary::Sshr, true) => Some(Bfx {
            insert: false,
            signed: true,
            src,
            lsb: lsb as u8,
            width,
        }),
        (Binary::Ishl, _) => Some(Bfx {
            insert: true,
            signed,
            src,
            lsb: lsb as u8,
            width,
        }),
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
        return Some(Bfx {
            insert: false,
            signed: false,
            src,
            lsb: 0,
            width: w as u8,
        });
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
    if let Op::Binary(Binary::Band, a, b) = *op
        && let Some(bfx) = conv_bfx_band(func, plan, (src_ty, signed), (x, a, b), def)
    {
        return Some(bfx);
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

pub(in crate::arm64) fn conv_bfx_parts(
    func: &Function,
    plan: &Plan<Reg>,
    (dst_ty, conv): (Type, Convert),
    x: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<Bfx> {
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
        let Some((_, _, inst)) = def_entry(pos_of, cur) else {
            return false;
        };
        let Op::Convert(Convert::Uextend, inner) = func.insts[inst.index()].op else {
            return false;
        };
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
    let def = |v| def_inst(pos_of, v);
    let ok = |v: Value| v == x;
    let Some(bfx) = conv_bfx_match(func, plan, (func.ty(res), conv), x, &def, &ok) else {
        return false;
    };
    let is_uext = !bfx.signed;
    let need = if bfx.insert {
        u32::from(bfx.width)
    } else {
        u32::from(bfx.lsb) + u32::from(bfx.width)
    };
    let ext_ok = |inner: Value| need <= func.ty(inner).bits() || zero_ext_loc(plan, inner);
    let peeled = is_uext
        && can_peel_uextend(
            func,
            plan,
            info,
            (b_idx, xpos, pos, Some(xpos), res),
            (bfx.src, &ext_ok),
        );
    if !peeled
        && !live_root_preserved(
            func,
            plan,
            pos_of,
            (b_idx, xpos, pos, None, Some(res)),
            bfx.src,
        )
    {
        return false;
    }
    plan.elided[x.index()] = true;
    if peeled {
        plan.elided[bfx.src.index()] = true;
    }
    true
}
