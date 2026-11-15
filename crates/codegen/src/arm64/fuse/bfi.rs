use super::super::asm::Reg;
use super::bits::{UseInfo, bfx_parts};
use super::{const_of, def_entry, def_inst, local_single, preserved, zero_ext_loc};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Convert, Function, Inst, Op, Type, Value};

fn width_mask(width: u8) -> u64 {
    if u32::from(width) >= 64 { u64::MAX } else { (1_u64 << width) - 1 }
}

fn cleared_field(mask: u64, ty_mask: u64) -> Option<(u8, u8)> {
    let cleared = (!mask) & ty_mask;
    if cleared == 0 || cleared == ty_mask {
        return None;
    }
    let lsb = cleared.trailing_zeros();
    let shifted = cleared >> lsb;
    ((shifted & (shifted + 1)) == 0).then(|| (lsb as u8, shifted.trailing_ones() as u8))
}

fn val_fits_width(
    func: &Function,
    plan: &Plan<Reg>,
    (ty, width): (Type, u8),
    val: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> bool {
    if plan.elided[val.index()] || const_of(func, plan, val, def).is_some() {
        return false;
    }
    let Some(inst) = def(val) else { return false };
    let (bits, ty_mask, need_mask) = (ty.bits(), ty.unsigned_max(), width_mask(width));
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Band, a, b) => [a, b].into_iter().any(|mv| {
            const_of(func, plan, mv, def).is_some_and(|c| (((c as u64) & ty_mask) & !need_mask) == 0)
        }),
        Op::Convert(Convert::Uextend, inner) => {
            func.ty(inner).is_small_int() && func.ty(inner).bits() <= u32::from(width)
        }
        Op::Binary(Binary::Ushr, _, amt) => const_of(func, plan, amt, def)
            .is_some_and(|c| ((c as u32) & (bits - 1)) >= bits - u32::from(width)),
        _ => false,
    }
}

fn bfi_band_ishl(
    func: &Function,
    plan: &Plan<Reg>,
    (ty, lsb, width): (Type, u8, u8),
    (x0, x1): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>)> {
    let (bits, ty_mask) = (ty.bits(), ty.unsigned_max());
    let top_mask = width_mask(lsb + width) & ty_mask;
    let low_mask = if lsb == 0 { 0 } else { (1_u64 << lsb) - 1 };
    for (mid, mv) in [(x0, x1), (x1, x0)] {
        if plan.elided[mid.index()] {
            continue;
        }
        let Some(imm) = const_of(func, plan, mv, def) else { continue };
        if ((((imm as u64) & ty_mask) | low_mask) & ty_mask) != top_mask {
            continue;
        }
        let Some(mi) = def(mid) else { continue };
        if let Op::Binary(Binary::Ishl, src, amt) = func.insts[mi.index()].op
            && const_of(func, plan, amt, def).is_some_and(|c| ((c as u32) & (bits - 1)) == u32::from(lsb))
            && !plan.elided[src.index()]
        {
            return Some((src, Some(mid)));
        }
    }
    None
}

fn bfi_rhs_plain(
    func: &Function,
    plan: &Plan<Reg>,
    (ty, lsb, width): (Type, u8, u8),
    b_op: &Op,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>)> {
    let (bits, ty_mask, need_mask) = (ty.bits(), ty.unsigned_max(), width_mask(width));
    match *b_op {
        Op::Binary(Binary::Band, bx, by) if lsb == 0 => [(bx, by), (by, bx)].into_iter().find_map(|(src, mv)| {
            const_of(func, plan, mv, def)
                .is_some_and(|c| ((c as u64) & ty_mask) == need_mask)
                .then_some((src, None))
        }),
        Op::Binary(Binary::Band, bx, by) if lsb > 0 => bfi_band_ishl(func, plan, (ty, lsb, width), (bx, by), def),
        Op::Binary(Binary::Ishl, bx, by)
            if lsb > 0 && const_of(func, plan, by, def).is_some_and(|c| ((c as u32) & (bits - 1)) == u32::from(lsb)) =>
        {
            (u32::from(lsb) + u32::from(width) == bits || val_fits_width(func, plan, (ty, width), bx, def))
                .then_some((bx, None))
        }
        Op::Convert(Convert::Uextend, inner)
            if lsb == 0 && func.ty(inner).is_small_int() && func.ty(inner).bits() == u32::from(width) =>
        {
            Some((inner, None))
        }
        _ => None,
    }
}

fn bfi_rhs(
    func: &Function,
    plan: &Plan<Reg>,
    (ty, lsb, width): (Type, u8, u8),
    b_op: &Op,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Option<Value>)> {
    if let Op::Binary(bop @ (Binary::Band | Binary::Ishl), bx, by) = *b_op
        && let Some(bfx) = bfx_parts(func, plan, ty, bop, (bx, by), def)
        && bfx.insert
        && !bfx.signed
        && bfx.lsb == lsb
        && bfx.width == width
    {
        return Some((bfx.src, None));
    }
    bfi_rhs_plain(func, plan, (ty, lsb, width), b_op, def)
}

fn bfi_clear(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    ca: Value,
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, u8, u8)> {
    let Op::Binary(Binary::Band, x0, x1) = func.insts[def(ca)?.index()].op else { return None };
    let ty_mask = ty.unsigned_max();
    for (a, mv) in [(x0, x1), (x1, x0)] {
        let Some(imm) = const_of(func, plan, mv, def) else { continue };
        let Some((lsb, width)) = cleared_field((imm as u64) & ty_mask, ty_mask) else { continue };
        if !plan.elided[a.index()] && const_of(func, plan, a, def).is_none() {
            return Some((a, lsb, width));
        }
    }
    None
}

fn bfi_pair(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (ca, cb): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value, Option<Value>, u8, u8)> {
    let (a, lsb, width) = bfi_clear(func, plan, ty, ca, def)?;
    let bi = def(cb)?;
    let (b, mid) = bfi_rhs(func, plan, (ty, lsb, width), &func.insts[bi.index()].op, def)?;
    Some((a, b, mid, lsb, width))
}

pub(in crate::arm64) fn bfi_parts(
    func: &Function,
    plan: &Plan<Reg>,
    ty: Type,
    (x, y): (Value, Value),
    def: &dyn Fn(Value) -> Option<Inst>,
) -> Option<(Value, Value, u8, u8)> {
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    if plan.elided[x.index()] && plan.elided[y.index()] {
        return bfi_pair(func, plan, ty, (x, y), def)
            .or_else(|| bfi_pair(func, plan, ty, (y, x), def))
            .map(|(a, b, _, lsb, width)| (a, b, lsb, width));
    }
    if plan.elided[x.index()] == plan.elided[y.index()] {
        return None;
    }
    let (ca, cb) = if plan.elided[x.index()] { (x, y) } else { (y, x) };
    let (a, lsb, width) = bfi_clear(func, plan, ty, ca, def)?;
    (lsb == 0 && val_fits_width(func, plan, (ty, width), cb, def)).then_some((a, cb, 0, width))
}

fn inst_clobbers(plan: &Plan<Reg>, inst_val: Value, target: Value) -> bool {
    match plan.location(target) {
        Location::Reg(reg) => plan.reg_of(inst_val) == Some(reg),
        Location::Slot => {
            let sid = plan.shared.get(target.index()).copied().flatten();
            sid.is_some()
                && matches!(plan.location(inst_val), Location::Slot)
                && plan.shared.get(inst_val.index()).copied().flatten() == sid
        }
        _ => false,
    }
}

fn bfi_mid_root(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, apos, bpos, pos, res): (usize, usize, usize, usize, Value),
    (mid, src): (Value, Value),
) -> bool {
    let Some((mb, mpos, _)) = def_entry(pos_of, mid) else { return false };
    let def = |v| def_inst(pos_of, v);
    mb == b_idx
        && mpos < bpos
        && src != res
        && !plan.elided[src.index()]
        && const_of(func, plan, src, &def).is_none()
        && !inst_clobbers(plan, mid, src)
        && preserved(func, plan, b_idx, mpos, bpos, Some(apos), src)
        && preserved(func, plan, b_idx, bpos, pos, Some(apos), src)
}

fn bfi_root(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, bpos, pos, skip, width): (usize, usize, usize, usize, u8),
    val: Value,
) -> Option<Value> {
    let root = if plan.elided[val.index()] {
        let (_, _, uinst) = def_entry(pos_of, val)?;
        let Op::Convert(Convert::Uextend, inner) = func.insts[uinst.index()].op else { return None };
        if plan.elided[inner.index()] || (u32::from(width) > func.ty(inner).bits() && !zero_ext_loc(plan, inner)) {
            return None;
        }
        inner
    } else {
        val
    };
    let def = |v| def_inst(pos_of, v);
    (const_of(func, plan, root, &def).is_none() && preserved(func, plan, b_idx, bpos, pos, Some(skip), root))
        .then_some(root)
}

fn try_fuse_bfi_unelided(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (x, y): (Value, Value),
) -> bool {
    let (ty, def) = (func.ty(res), |v| def_inst(pos_of, v));
    for (ca, cb) in [(x, y), (y, x)] {
        let Some((apos, _)) = local_single(plan, info, b_idx, pos, ca) else { continue };
        let Some((a, lsb, width)) = bfi_clear(func, plan, ty, ca, &def) else { continue };
        if lsb == 0
            && !plan.elided[cb.index()]
            && a != res
            && val_fits_width(func, plan, (ty, width), cb, &def)
            && (plan.reg_of(a) == plan.reg_of(res) || plan.reg_of(cb) != plan.reg_of(res))
            && preserved(func, plan, b_idx, apos, pos, None, a)
        {
            plan.elided[ca.index()] = true;
            return true;
        }
    }
    false
}

pub(super) fn try_fuse_bfi(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (_, pos_of): UseInfo<'_>,
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) -> bool {
    if !matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd) || !matches!(func.ty(res), Type::I32 | Type::I64) {
        return false;
    }
    if let Some((xpos, _)) = local_single(plan, info, b_idx, pos, x)
        && let Some((ypos, _)) = local_single(plan, info, b_idx, pos, y)
    {
        let (ty, def) = (func.ty(res), |v| def_inst(pos_of, v));
        for (ca, cb, apos, bpos) in [(x, y, xpos, ypos), (y, x, ypos, xpos)] {
            let Some((a, b, mid_opt, _, width)) = bfi_pair(func, plan, ty, (ca, cb), &def) else { continue };
            if a == res || !preserved(func, plan, b_idx, apos, pos, Some(bpos), a) {
                continue;
            }
            let rhs_ok = match mid_opt {
                Some(mid) => bfi_mid_root(func, plan, pos_of, (b_idx, apos, bpos, pos, res), (mid, b)),
                None => bfi_root(func, plan, pos_of, (b_idx, bpos, pos, apos, width), b).is_some_and(|r| r != res),
            };
            if rhs_ok {
                plan.elided[x.index()] = true;
                plan.elided[y.index()] = true;
                return true;
            }
        }
    }
    try_fuse_bfi_unelided(func, plan, info, (b_idx, pos, res), (x, y))
}
