use crate::ir::{Binary, Block, Convert, Function, Op, TargetOpId, Type, Unary, Value};
use crate::live::{Live, covers, overlaps};
use crate::park::crosses_ranges;

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

struct ReuseCtx<'a, R: 'static> {
    func: &'a Function,
    regs: &'a Regs<R>,
    locations: &'a [Location<R>],
    live: &'a Live,
    is_param: &'a [bool],
}

fn reuse_dies<R: 'static>(ctx: &ReuseCtx<'_, R>, result: Value, op: Value, end_at: u32) -> bool {
    same_class(ctx.regs, ctx.func.ty(op), ctx.func.ty(result))
        && ctx.live.last_end(op.index()) == Some(end_at)
        && !overlaps(ctx.live.ranges_of(op.index()), ctx.live.ranges_of(result.index()))
}

fn prefers_y(is_param: &[bool], reuse: &[Option<Value>], x: Value, y: Value) -> bool {
    is_param[y.index()] && !is_param[x.index()] && reuse[x.index()].is_none_or(|rx| !is_param[rx.index()])
}

fn followed_by_fits<R: 'static>(ctx: &ReuseCtx<'_, R>, block_idx: usize, at: usize, (op, x, y): (Binary, Value, Value)) -> bool {
    let next = ctx.func.blocks[block_idx].insts.get(at + 1).map(|next| &ctx.func.insts[next.index()].op);
    matches!(next, Some(Op::Fits(check, _, a, b)) if *check == op && (*a, *b) == (x, y))
}

fn binary_reuse_allowed<R: 'static>(ctx: &ReuseCtx<'_, R>, op: Binary, y: Value) -> (bool, bool) {
    let imm_y = matches!(ctx.locations[y.index()], Location::Imm(_));
    let can_x = in_place(op) || (is_shift(op) && (!ctx.regs.two_address || imm_y));
    let can_y = is_commutative(op) || (!ctx.regs.two_address && (matches!(op, Binary::Isub) || is_shift(op)));
    (can_x, can_y)
}

fn reuse_binary<R: 'static>(ctx: &ReuseCtx<'_, R>, reuse: &[Option<Value>], (block_idx, at, result): (usize, usize, Value), (op, x, y): (Binary, Value, Value), end_at: u32) -> Option<Value> {
    if followed_by_fits(ctx, block_idx, at, (op, x, y)) {
        return None;
    }
    let (can_x, can_y) = binary_reuse_allowed(ctx, op, y);
    let dies_x = can_x && reuse_dies(ctx, result, x, end_at);
    let dies_y = can_y && reuse_dies(ctx, result, y, end_at);
    if dies_x && dies_y && is_commutative(op) && prefers_y(ctx.is_param, reuse, x, y) {
        return Some(y);
    }
    dies_x.then_some(x).or_else(|| dies_y.then_some(y))
}

fn reuse_candidate<R: 'static>(ctx: &ReuseCtx<'_, R>, reuse: &[Option<Value>], (block_idx, at, result): (usize, usize, Value), end_at: u32) -> Option<Value> {
    let data = &ctx.func.insts[ctx.func.blocks[block_idx].insts[at].index()];
    match data.op {
        Op::TargetOp(TargetOpId::X64AddCarry | TargetOpId::X64SubBorrow, ref args) => match *args.as_slice() {
            [carry, a, b] if a != b && a != carry && reuse_dies(ctx, result, a, end_at) => Some(a),
            _ => None,
        },
        Op::Binary(op, x, y) => reuse_binary(ctx, reuse, (block_idx, at, result), (op, x, y), end_at),
        Op::Convert(Convert::Uextend | Convert::Sextend | Convert::Ireduce | Convert::Bitcast, x)
            if reuse_dies(ctx, result, x, end_at) => Some(x),
        Op::Unary(Unary::Ineg | Unary::Bnot | Unary::Bswap | Unary::Bitrev | Unary::Clz | Unary::Ctz | Unary::Fabs | Unary::Fneg | Unary::Sqrt, x)
            if reuse_dies(ctx, result, x, end_at) => Some(x),
        Op::Select(cond, x, y) => {
            if x != cond && reuse_dies(ctx, result, x, end_at) {
                let pick_y = !ctx.regs.two_address && y != cond && reuse_dies(ctx, result, y, end_at) && prefers_y(ctx.is_param, reuse, x, y);
                Some(if pick_y { y } else { x })
            } else {
                (!ctx.regs.two_address && y != cond && reuse_dies(ctx, result, y, end_at)).then_some(y)
            }
        }
        _ => None,
    }
}

/// Returns for each value a dying same-class operand whose register may be reused in place.
pub(super) fn reusable<R: 'static>(func: &Function, regs: &Regs<R>, locations: &[Location<R>], live: &Live, elided: &[bool]) -> Vec<Option<Value>> {
    let mut is_param = vec![false; func.types.len()];
    for block in &func.blocks {
        for &param in &block.params {
            is_param[param.index()] = true;
        }
    }
    let ctx = ReuseCtx { func, regs, locations, live, is_param: &is_param };
    let mut reuse = vec![None; func.types.len()];
    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (at, inst) in block.insts.iter().enumerate() {
            let Some(result) = func.insts[inst.index()].result else { continue };
            let rty = func.ty(result);
            if elided[result.index()] || reg_class(regs, rty).is_none() {
                continue;
            }
            let def = live.defs[result.index()];
            if def == 0 || live.ranges_of(result.index()).first().is_none_or(|r| r.start != def) {
                continue;
            }
            reuse[result.index()] = reuse_candidate(&ctx, &reuse, (block_idx, at, result), def - 1);
        }
    }
    reuse
}

/// Each zero-extension of a value with clean upper bits that stays live through it and no call, as (copy, source): the two hold the same register bits.
pub(super) fn extend_copies(func: &Function, live: &Live, zero_ext: &[bool], pooled: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        let (Op::Convert(Convert::Uextend, source), Some(copy)) = (&data.op, data.result) else { continue };
        let (copy, source) = (copy.index(), source.index());
        let ranges = live.ranges_of(copy);
        if pooled[copy]
            && pooled[source]
            && zero_ext[source]
            && func.types[copy].bytes() <= 8
            && !ranges.is_empty()
            && !crosses_ranges(&live.calls, ranges)
            && covers(live.ranges_of(source), ranges)
        {
            out.push((copy, source));
        }
    }
    out
}

fn reg_reduce<R: PartialEq>(func: &Function, locations: &[Location<R>], res: Value, src: Value) -> bool {
    matches!(func.types[res.index()], Type::I8 | Type::I16) && matches!((&locations[res.index()], &locations[src.index()]), (Location::Reg(a), Location::Reg(b)) if a != b)
}

/// Whether some 8- or 16-bit narrowing copies between two different registers, where an extending copy costs no more than a move.
pub(super) fn has_reg_reduce<R: PartialEq>(func: &Function, locations: &[Location<R>]) -> bool {
    func.insts.iter().any(|data| matches!((&data.op, data.result), (&Op::Convert(Convert::Ireduce, src), Some(res)) if reg_reduce(func, locations, res, src)))
}

/// Computes which small integer values already have zero-extended upper bits in 64-bit registers.
pub(super) fn zero_ext<R: PartialEq>(func: &Function, locations: &[Location<R>], census: &Uses, call_zext: bool) -> Vec<bool> {
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
        let narrow_call = call_zext && matches!(ty, Type::I8 | Type::I16) && census.def_inst[v].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Call { .. }));
        out[v] = ty.is_small_int() && (single_def || narrow_call || incoming[v].as_ref().is_some_and(|l| !l.is_empty()));
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
                    (call_zext && reg_reduce(func, locations, res, shifted))
                        || clean_bits(&out, shifted).is_some_and(|w| w <= dst_bits)
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
