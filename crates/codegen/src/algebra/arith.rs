use crate::fold::extend;
use crate::ir::{Binary, Convert, Op, Type, Value};

use super::RewriteCx;

fn magic_udiv(cx: &mut RewriteCx, ty: Type, x: Value, d: u64) -> Op {
    if ty != Type::I64 {
        let x64 = cx.emit(Type::I64, Op::Convert(Convert::Uextend, x));
        let m = ((1_u128 << 64) / u128::from(d) + 1) as u64;
        let m_val = cx.emit(Type::I64, Op::Const(m as i64));
        let q64 = cx.emit(Type::I64, Op::Binary(Binary::Umulhi, x64, m_val));
        return Op::Convert(Convert::Ireduce, q64);
    }
    let l = 64 - d.leading_zeros();
    let s = l - 1;
    let d128 = u128::from(d);
    let m_cand = (1_u128 << (64 + s)) / d128 + 1;
    let err = m_cand * d128 - (1_u128 << (64 + s));
    if m_cand < (1_u128 << 64) && (u128::from(u64::MAX / d) + 1) * err < m_cand {
        let m_val = cx.emit(Type::I64, Op::Const(m_cand as u64 as i64));
        let hi = cx.emit(Type::I64, Op::Binary(Binary::Umulhi, x, m_val));
        let sh = cx.emit(Type::I64, Op::Const(i64::from(s)));
        return Op::Binary(Binary::Ushr, hi, sh);
    }
    let m_full = if l == 64 {
        u128::MAX / d128 + 1
    } else {
        (1_u128 << (64 + l)) / d128 + 1
    };
    let m_low = (m_full - (1_u128 << 64)) as u64;
    let m_val = cx.emit(Type::I64, Op::Const(m_low as i64));
    let t1 = cx.emit(Type::I64, Op::Binary(Binary::Umulhi, x, m_val));
    let diff = cx.emit(Type::I64, Op::Binary(Binary::Isub, x, t1));
    let one = cx.emit(Type::I64, Op::Const(1));
    let half = cx.emit(Type::I64, Op::Binary(Binary::Ushr, diff, one));
    let t2 = cx.emit(Type::I64, Op::Binary(Binary::Iadd, t1, half));
    let sh = cx.emit(Type::I64, Op::Const(i64::from(l - 1)));
    Op::Binary(Binary::Ushr, t2, sh)
}

pub(super) fn fold_rodata_load(cx: &RewriteCx, ty: Type, addr: Value, off: i32) -> Option<Op> {
    let Op::Symbol(sym) = cx.func.insts[cx.def(addr)?.index()].op else {
        return None;
    };
    if ty.is_vector() {
        let slice = cx
            .func
            .rodata_slice(sym, i64::from(off), usize::try_from(ty.bytes()).ok()?)?;
        return slice.iter().all(|&b| b == 0).then_some(Op::Const(0));
    }
    cx.func.rodata_read(sym, i64::from(off), ty).map(Op::Const)
}

pub(super) fn const_addend(cx: &RewriteCx, x: Value, y: Value) -> Option<(Value, i64)> {
    [(x, y), (y, x)]
        .into_iter()
        .find_map(|(v, cv)| cx.sconst(cv).map(|c| (v, c)))
}

pub(super) fn fold_disp(cx: &RewriteCx, addr: Value, off: i32, bytes: u32) -> Option<(Value, i32)> {
    let &Op::Binary(Binary::Iadd, x, y) = cx.op(addr)? else {
        return None;
    };
    let (base, c) = const_addend(cx, x, y)?;
    if cx.func.ty(base) != Type::I64 || matches!(cx.op(base), Some(Op::StackAddr(..))) {
        return None;
    }
    let disp = i64::from(off).checked_add(c)?;
    i32::try_from(disp.checked_add(i64::from(bytes))?).ok()?;
    i32::try_from(disp).ok().map(|disp| (base, disp))
}

pub(super) fn reassociate_add(
    cx: &mut RewriteCx,
    fits_used: &[bool],
    a: Value,
    b: Value,
) -> Option<Op> {
    let ty = cx.func.ty(a);
    if !ty.is_small_int() {
        return None;
    }
    for (lhs, c2_v) in [(a, b), (b, a)] {
        let Some(c2) = cx
            .sconst(c2_v)
            .filter(|_| !fits_used.get(lhs.index()).copied().unwrap_or(false))
        else {
            continue;
        };
        let Some(inst) = cx.def(lhs) else { continue };
        match cx.func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, x, y) => {
                if let Some((base, c1)) = const_addend(cx, x, y) {
                    let c = cx.emit(ty, Op::Const(extend(c1.wrapping_add(c2), ty, true)));
                    return Some(Op::Binary(Binary::Iadd, base, c));
                }
            }
            Op::Binary(Binary::Isub, x, y) => {
                if let Some(c1) = cx.sconst(y) {
                    let c = cx.emit(ty, Op::Const(extend(c2.wrapping_sub(c1), ty, true)));
                    return Some(Op::Binary(Binary::Iadd, x, c));
                }
            }
            _ => {}
        }
    }
    None
}

pub(super) fn reassociate_sub(
    cx: &mut RewriteCx,
    fits_used: &[bool],
    a: Value,
    b: Value,
) -> Option<Op> {
    let ty = cx.func.ty(a);
    let c2 = cx
        .sconst(b)
        .filter(|_| ty.is_small_int() && !fits_used.get(a.index()).copied().unwrap_or(false))?;
    match cx.func.insts[cx.def(a)?.index()].op {
        Op::Binary(Binary::Iadd, x, y) => {
            let (base, c1) = const_addend(cx, x, y)?;
            let c = cx.emit(ty, Op::Const(extend(c1.wrapping_sub(c2), ty, true)));
            Some(Op::Binary(Binary::Iadd, base, c))
        }
        Op::Binary(Binary::Isub, x, y) => {
            let c = cx.emit(
                ty,
                Op::Const(extend(cx.sconst(y)?.wrapping_add(c2), ty, true)),
            );
            Some(Op::Binary(Binary::Isub, x, c))
        }
        _ => None,
    }
}

pub(super) fn simplify_div_rem(
    cx: &mut RewriteCx,
    block_udivs: &mut Vec<(Value, Value, Value)>,
    res: Option<Value>,
    (bin, x, y): (Binary, Value, Value),
) -> Option<Op> {
    let ty = cx.func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    if let Some(d) = cx.uconst(y, ty).filter(|&d| d > 1) {
        if d.is_power_of_two() {
            return Some(if bin == Binary::Udiv {
                let sh = cx.emit(ty, Op::Const(i64::from(d.trailing_zeros())));
                Op::Binary(Binary::Ushr, x, sh)
            } else {
                let mask = cx.emit(ty, Op::Const((d - 1) as i64));
                Op::Binary(Binary::Band, x, mask)
            });
        }
        let q_op = magic_udiv(cx, ty, x, d);
        if bin == Binary::Udiv {
            return Some(q_op);
        }
        let q = cx.emit(ty, q_op);
        let prod = cx.emit(ty, Op::Binary(Binary::Imul, q, y));
        return Some(Op::Binary(Binary::Isub, x, prod));
    }
    if bin == Binary::Udiv {
        if let Some(r) = res {
            block_udivs.push((x, y, r));
        }
        return None;
    }
    let &(_, _, q) = block_udivs
        .iter()
        .find(|&&(ux, uy, _)| ux == x && uy == y)?;
    let prod = cx.emit(ty, Op::Binary(Binary::Imul, q, y));
    Some(Op::Binary(Binary::Isub, x, prod))
}

pub(super) fn fold_select_const_bin(
    cx: &mut RewriteCx,
    bin: Binary,
    a: Value,
    b: Value,
) -> Option<Op> {
    if !matches!(
        bin,
        Binary::Iadd
            | Binary::Isub
            | Binary::Imul
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
            | Binary::Ishl
            | Binary::Ushr
            | Binary::Sshr
    ) {
        return None;
    }
    let ty = cx.func.ty(a);
    if !ty.is_small_int() {
        return None;
    }
    for (sel_v, c_v, sel_lhs) in [(a, b, true), (b, a, false)] {
        if !sel_lhs && matches!(bin, Binary::Ishl | Binary::Ushr | Binary::Sshr) {
            continue;
        }
        let (Some(c), true) = (cx.sconst(c_v), cx.single_use(sel_v)) else {
            continue;
        };
        let Some(&Op::Select(cond, t, f)) = cx.op(sel_v) else {
            continue;
        };
        let (Some(kt), Some(kf)) = (cx.sconst(t), cx.sconst(f)) else {
            continue;
        };
        let (nt, nf) = if sel_lhs {
            (
                crate::fold::binary(bin, ty, kt, c)?,
                crate::fold::binary(bin, ty, kf, c)?,
            )
        } else {
            (
                crate::fold::binary(bin, ty, c, kt)?,
                crate::fold::binary(bin, ty, c, kf)?,
            )
        };
        let (et, ef) = (extend(nt, ty, true), extend(nf, ty, true));
        if et == ef {
            return Some(Op::Const(et));
        }
        let (tv, fv) = (cx.emit(ty, Op::Const(et)), cx.emit(ty, Op::Const(ef)));
        return Some(Op::Select(cond, tv, fv));
    }
    None
}
