use crate::ir::{Binary, Block, Convert, Function, Op, Type, Unary, Value};
use crate::live::{Live, overlaps};

use super::{Location, Regs, Uses, registered};

fn in_place(op: Binary) -> bool {
    matches!(
        op,
        Binary::Iadd
            | Binary::Isub
            | Binary::Imul
            | Binary::Band
            | Binary::Bor
            | Binary::Bxor
            | Binary::Fadd
            | Binary::Fsub
            | Binary::Fmul
            | Binary::Fdiv
    )
}

fn is_shift(op: Binary) -> bool {
    matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr)
}

fn is_commutative(op: Binary) -> bool {
    matches!(
        op,
        Binary::Iadd | Binary::Imul | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Fadd | Binary::Fmul
    )
}

fn reg_class<R>(regs: &Regs<R>, ty: Type) -> Option<bool> {
    (ty != Type::I128 && registered(regs, ty)).then(|| ty.is_small_int())
}

fn same_class<R>(regs: &Regs<R>, a: Type, b: Type) -> bool {
    reg_class(regs, a).is_some_and(|ca| reg_class(regs, b) == Some(ca))
}

/// Returns for each value a dying same-class operand whose register may be reused in place.
pub(super) fn reusable<R>(func: &Function, regs: &Regs<R>, locations: &[Location<R>], live: &Live, elided: &[bool]) -> Vec<Option<Value>> {
    let mut reuse = vec![None; func.types.len()];
    for block in &func.blocks {
        for (at, inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result else { continue };
            let rty = func.ty(result);
            if elided[result.index()] || reg_class(regs, rty).is_none() {
                continue;
            }
            let def = live.defs[result.index()];
            if def == 0 || live.ranges_of(result.index()).first().is_none_or(|r| r.start != def) {
                continue;
            }
            let end_at = def - 1;
            let dies = |op: Value| {
                same_class(regs, func.ty(op), rty)
                    && live.last_end(op.index()) == Some(end_at)
                    && !overlaps(live.ranges_of(op.index()), live.ranges_of(result.index()))
            };
            let chosen = match &data.op {
                Op::Binary(op, x, y) => {
                    let next = block.insts.get(at + 1).map(|next| &func.insts[next.index()].op);
                    if matches!(next, Some(Op::Fits(check, _, a, b)) if check == op && (a, b) == (x, y)) {
                        continue;
                    }
                    let can_x = in_place(*op) || (is_shift(*op) && (!regs.two_address || matches!(locations[y.index()], Location::Imm(_))));
                    if can_x && dies(*x) {
                        Some(*x)
                    } else if (is_commutative(*op) || (!regs.two_address && (matches!(*op, Binary::Isub) || is_shift(*op)))) && dies(*y) {
                        Some(*y)
                    } else {
                        None
                    }
                }
                Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce | Convert::Bitcast, x) if dies(*x) => Some(*x),
                Op::Unary(
                    Unary::Ineg | Unary::Bnot | Unary::Bswap | Unary::Bitrev | Unary::Clz | Unary::Ctz | Unary::Fabs | Unary::Fneg | Unary::Sqrt,
                    x,
                ) if dies(*x) => Some(*x),
                Op::Select(cond, x, y) => {
                    if *x != *cond && dies(*x) {
                        Some(*x)
                    } else if !regs.two_address && *y != *cond && dies(*y) {
                        Some(*y)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            reuse[result.index()] = chosen;
        }
    }
    reuse
}

/// Computes which small integer values already have zero-extended upper bits in 64-bit registers.
pub(super) fn zero_ext<R>(func: &Function, locations: &[Location<R>], census: &Uses) -> Vec<bool> {
    let count = func.types.len();
    let (mut out, mut incoming) = (vec![false; count], vec![Some(Vec::new()); count]);
    if let Some(entry) = func.entry {
        func.blocks[entry.index()].params.iter().for_each(|p| incoming[p.index()] = None);
    }
    for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let op = &func.insts[inst.index()].op;
        if let Op::TryCall { handler, .. } = *op {
            func.blocks[handler.index()].params.iter().for_each(|p| incoming[p.index()] = None);
        }
        crate::frame::terminator_edges(op, &mut |target: Block, args: &[Value]| {
            for (p, &a) in func.blocks[target.index()].params.iter().zip(args) {
                if a != *p && let Some(list) = &mut incoming[p.index()] {
                    list.push(a);
                }
            }
        });
    }
    for (v, ty) in func.types.iter().enumerate() {
        let single_def = census.def_inst[v].is_some_and(|d| func.insts[d.index()].result == Some(Value(v as u32)));
        out[v] = ty.is_small_int() && (single_def || incoming[v].as_ref().is_some_and(|l| !l.is_empty()));
    }
    let const_val = |v: Value| match locations[v.index()] {
        Location::Imm(imm) => Some(imm),
        _ => census.def_inst[v.index()].and_then(|d| match func.insts[d.index()].op { Op::Const(imm) => Some(imm), _ => None }),
    };
    let ty_mask = |ty: Type| if ty == Type::I64 { i64::MAX } else { (1_i64 << ty.bits()) - 1 };
    let fits_mask = |v: Value, mask: i64| const_val(v).is_some_and(|m| (0..=mask).contains(&m));
    let is_icmp = |v: Value| census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Icmp(..)));
    let is_cmp = |v: Value| is_icmp(v) || census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Convert(Convert::Uextend, inner) if is_icmp(inner)));
    let clean_bits = |out: &[bool], v: Value| -> Option<u32> {
        let imm_bits = |x: i64| (x >= 0).then(|| 64 - (x as u64).leading_zeros());
        if let Some(w) = const_val(v).and_then(imm_bits) {
            return Some(w);
        }
        match census.def_inst[v.index()].map(|d| &func.insts[d.index()].op) {
            Some(&Op::Convert(Convert::Uextend, inner)) => Some(func.types[inner.index()].bits()),
            Some(&Op::Binary(Binary::Band, a, b)) => const_val(a)
                .and_then(imm_bits)
                .or_else(|| const_val(b).and_then(imm_bits))
                .map(|w| w.min(func.types[v.index()].bits()))
                .or_else(|| out[v.index()].then(|| func.types[v.index()].bits())),
            _ => out[v.index()].then(|| func.types[v.index()].bits()),
        }
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let Some(res) = func.insts[inst.index()].result else { continue };
            let ty = func.types[res.index()];
            if !ty.is_small_int() {
                continue;
            }
            let z = match func.insts[inst.index()].op {
                Op::Const(imm) => (0..=ty_mask(ty)).contains(&imm),
                Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(_) | Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) | Op::Convert(Convert::Uextend, _) | Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => true,
                Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Unary(Unary::Bitrev, _) => ty == Type::I32,
                Op::Unary(Unary::Bswap, a) => ty == Type::I32 && out[a.index()],
                Op::Select(_, a, b) => out[a.index()] && out[b.index()],
                Op::Binary(Binary::Bor | Binary::Bxor, a, b) => ty == Type::I32 || (out[a.index()] && out[b.index()]),
                Op::Binary(Binary::Iadd, a, b) => ty == Type::I32 && !is_cmp(a) && !is_cmp(b),
                Op::Binary(Binary::Isub | Binary::Imul | Binary::Rotl | Binary::Rotr | Binary::Udiv | Binary::Urem | Binary::Umulhi | Binary::Umin | Binary::Umax | Binary::UaddSat | Binary::UsubSat | Binary::AvgRound, _, _) => ty == Type::I32,
                Op::Binary(Binary::Ushr, a, _) => ty == Type::I32 || out[a.index()],
                Op::Binary(Binary::Ishl, a, amt) => const_val(amt).is_some_and(|s| clean_bits(&out, a).is_some_and(|w| w + ((s as u32) & (ty.bits() - 1)) <= ty.bits())),
                Op::Binary(Binary::Band, a, b) => ty == Type::I32 || (matches!(ty, Type::I8 | Type::I16) && (out[a.index()] || out[b.index()] || fits_mask(a, ty_mask(ty)) || fits_mask(b, ty_mask(ty)))),
                Op::Convert(Convert::Ireduce, shifted) => {
                    let (src_ty, dst_bits) = (func.types[shifted.index()], ty.bits());
                    clean_bits(&out, shifted).is_some_and(|w| w <= dst_bits)
                        || (matches!(src_ty, Type::I64 | Type::I32)
                            && census.def_inst[shifted.index()].is_some_and(|def| match func.insts[def.index()].op {
                                Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => dst_bits >= 8,
                                Op::Binary(Binary::Ushr, _, c) => const_val(c).is_some_and(|s| (0..i64::from(src_ty.bits())).contains(&s) && src_ty.bits() - (s as u32) <= dst_bits),
                                _ => false,
                            }))
                }
                _ => false,
            };
            if out[res.index()] != z {
                out[res.index()] = z;
                changed = true;
            }
        }
        for (p, list) in incoming.iter().enumerate() {
            if out[p] && let Some(list) = list && !list.iter().all(|a| out[a.index()]) {
                out[p] = false;
                changed = true;
            }
        }
    }
    out
}
