use crate::frame::{Abi, Loc};
use crate::ir::{AtomicOrd, Binary, Block, Convert, Function, InstData, Op, Signature, TargetOpId, Type, Unary, Value};
use crate::live::{Live, covers, overlaps};
use crate::park::crosses_ranges;

use super::{Location, Regs, Uses, registered};

const RCPC2_BIT: u8 = 16;

/// Extracts `(addr, mask)` for an ARM64 relaxed or RCPC2 acquire/release atomic memory site.
pub(crate) fn rcpc2_atomic_site(func: &Function, data: &InstData, rcpc2: bool) -> Option<(Value, u8)> {
    let (addr, bytes, extra) = match (&data.op, data.result) {
        (&Op::AtomicLoad(addr, ord), Some(r)) if !ord.has_acquire() => (addr, func.ty(r).bytes(), 0),
        (&Op::AtomicLoad(addr, AtomicOrd::Acquire), Some(r)) if rcpc2 => (addr, func.ty(r).bytes(), RCPC2_BIT),
        (&Op::AtomicStore(val, addr, ord), _) if !ord.has_release() && val != addr => (addr, func.ty(val).bytes(), 0),
        (&Op::AtomicStore(val, addr, AtomicOrd::Release), _) if rcpc2 && val != addr => (addr, func.ty(val).bytes(), RCPC2_BIT),
        _ => return None,
    };
    matches!(bytes, 1 | 2 | 4 | 8).then_some((addr, bytes as u8 | extra))
}

/// Reports whether `delta` fits every ARM64 atomic offset encoded in `mask`.
pub(crate) fn rcpc2_delta_fits(delta: i64, mask: u8) -> bool {
    if (-256..=255).contains(&delta) {
        return true;
    }
    if (mask & RCPC2_BIT) != 0 || delta < 0 {
        return false;
    }
    [1_u8, 2, 4, 8].into_iter().filter(|&b| (mask & b) != 0).all(|b| delta % i64::from(b) == 0 && delta / i64::from(b) < 4096)
}

pub(super) fn rcpc2_atomic_uses(func: &Function) -> Vec<(u32, u8)> {
    let mut info = vec![(0_u32, 0_u8); func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            if let Some((addr, mask)) = rcpc2_atomic_site(func, &func.insts[inst.index()], true) {
                let entry = &mut info[addr.index()];
                entry.0 += 1;
                entry.1 |= mask;
            }
        }
    }
    info
}

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

fn narrow_call_needs_zext<R>(func: &Function, locations: &[Location<R>], census: &Uses) -> Vec<bool> {
    let is_zero = |v: Value| match locations[v.index()] {
        Location::Imm(0) => true,
        _ => census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Const(0))),
    };
    let mut needs = vec![false; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let op = &func.insts[inst.index()].op;
        match op {
            Op::Brif(cond, _, ta, _, ea) => {
                for &v in ta.iter().chain(ea.iter()) {
                    needs[v.index()] = true;
                }
                let _ = cond;
            }
            Op::Trapz(_) => {}
            Op::Store(val, addr, _) | Op::AtomicStore(val, addr, _) | Op::VolatileStore(val, addr, _) => {
                needs[addr.index()] = true;
                let _ = val;
            }
            Op::StackStore(..) => {}
            Op::Icmp(crate::ir::IntCC::Equal | crate::ir::IntCC::NotEqual, a, b) if is_zero(*a) || is_zero(*b) => {}
            _ => op.for_operands(|v| needs[v.index()] = true),
        }
    }
    needs
}

/// Computes `(zero_ext, clean_bits)` where `clean_bits[v]` upper-bounds the active low bits of `v`'s 64-bit register value (`<= 64` when upper bits are zero, `65` when dirty).
pub(super) fn zero_ext<R: PartialEq>(func: &Function, locations: &[Location<R>], census: &Uses, call_zext: bool) -> (Vec<bool>, Vec<u8>) {
    const DIRTY: u8 = 65;
    let count = func.types.len();
    let (mut bits, mut incoming) = (vec![DIRTY; count], vec![Some(Vec::new()); count]);
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
    let call_needs = call_zext.then(|| narrow_call_needs_zext(func, locations, census));
    for (v, ty) in func.types.iter().enumerate() {
        if !ty.is_small_int() {
            continue;
        }
        let single_def = census.def_inst[v].is_some_and(|d| func.insts[d.index()].result == Some(Value(v as u32)));
        let narrow_call = call_needs.as_ref().is_some_and(|n| n[v])
            && matches!(ty, Type::I8 | Type::I16)
            && census.def_inst[v].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Call { .. }));
        if narrow_call {
            bits[v] = ty.bits() as u8;
        } else if single_def || incoming[v].as_ref().is_some_and(|l| !l.is_empty()) {
            bits[v] = 0;
        }
    }
    let const_val = |v: Value| match locations[v.index()] {
        Location::Imm(imm) => Some(imm),
        _ => census.def_inst[v.index()].and_then(|d| match func.insts[d.index()].op { Op::Const(imm) => Some(imm), _ => None }),
    };
    let ty_mask = |ty: Type| if ty == Type::I64 { i64::MAX } else { (1_i64 << ty.bits()) - 1 };
    let imm_bits = |imm: i64, ty: Type| {
        if (0..=ty_mask(ty)).contains(&imm) {
            (64 - (imm as u64).leading_zeros()) as u8
        } else {
            DIRTY
        }
    };
    let is_icmp = |v: Value| census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Icmp(..)));
    let is_cmp = |v: Value| is_icmp(v) || census.def_inst[v.index()].is_some_and(|d| matches!(func.insts[d.index()].op, Op::Convert(Convert::Uextend, inner) if is_icmp(inner)));
    let mut changed = true;
    while changed {
        changed = false;
        for &inst in func.blocks.iter().flat_map(|b| &b.insts) {
            let Some(res) = func.insts[inst.index()].result else { continue };
            let ty = func.types[res.index()];
            if !ty.is_small_int() {
                continue;
            }
            let w = ty.bits() as u8;
            let b = if let Some(imm) = const_val(res) {
                imm_bits(imm, ty)
            } else {
                match func.insts[inst.index()].op {
                    Op::Const(imm) => imm_bits(imm, ty),
                    Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..) => w,
                    Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => 1,
                    Op::Unary(Unary::Clz | Unary::Ctz | Unary::Popcnt, _) => 7_u8.min(w),
                    Op::Convert(Convert::Uextend, inner) => bits[inner.index()].min(func.types[inner.index()].bits() as u8),
                    Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Unary(Unary::Bitrev, _) => if ty == Type::I32 { 32 } else { DIRTY },
                    Op::Unary(Unary::Bswap, a) => if ty == Type::I32 && bits[a.index()] <= 32 { 32 } else { DIRTY },
                    Op::Select(_, a, b) => bits[a.index()].max(bits[b.index()]),
                    Op::Binary(Binary::Bor | Binary::Bxor, a, b) => {
                        let m = bits[a.index()].max(bits[b.index()]);
                        if ty == Type::I32 { m.min(32) } else { m }
                    }
                    Op::Binary(Binary::Iadd, a, b) => if ty == Type::I32 && !is_cmp(a) && !is_cmp(b) { 32 } else { DIRTY },
                    Op::Binary(Binary::Isub | Binary::Imul | Binary::Rotl | Binary::Rotr | Binary::Udiv | Binary::Urem | Binary::Umulhi | Binary::Umin | Binary::Umax | Binary::UaddSat | Binary::UsubSat | Binary::AvgRound, _, _) => {
                        if ty == Type::I32 { 32 } else { DIRTY }
                    }
                    Op::Binary(Binary::Ushr, a, c) => {
                        let shift = const_val(c).map(|s| ((s as u32) & (ty.bits() - 1)) as u8);
                        match ty {
                            Type::I32 | Type::I64 => {
                                let base = bits[a.index()].min(w);
                                shift.map_or(base, |s| base.saturating_sub(s))
                            }
                            _ if bits[a.index()] <= w => shift.map_or(bits[a.index()], |s| bits[a.index()].saturating_sub(s)),
                            _ => DIRTY,
                        }
                    }
                    Op::Binary(Binary::Ishl, a, amt) => {
                        if let Some(s) = const_val(amt) {
                            let sh = ((s as u32) & (ty.bits() - 1)) as u8;
                            if bits[a.index()].saturating_add(sh) <= w {
                                bits[a.index()] + sh
                            } else if ty == Type::I32 {
                                32
                            } else {
                                DIRTY
                            }
                        } else if ty == Type::I32 {
                            32
                        } else {
                            DIRTY
                        }
                    }
                    Op::Binary(Binary::Band, a, b) => {
                        let m = bits[a.index()].min(bits[b.index()]);
                        match ty {
                            Type::I32 => m.min(32),
                            Type::I8 | Type::I16 => if m <= w { m } else { DIRTY },
                            _ => m,
                        }
                    }
                    Op::Convert(Convert::Ireduce, shifted) => {
                        if bits[shifted.index()] <= w {
                            bits[shifted.index()]
                        } else if call_zext && reg_reduce(func, locations, res, shifted) {
                            w
                        } else {
                            DIRTY
                        }
                    }
                    _ => DIRTY,
                }
            };
            let next = bits[res.index()].max(b);
            if bits[res.index()] != next {
                bits[res.index()] = next;
                changed = true;
            }
        }
        for (p, list) in incoming.iter().enumerate() {
            if let Some(list) = list && !list.is_empty() {
                let next = list.iter().map(|a| bits[a.index()]).max().unwrap_or(DIRTY).max(bits[p]);
                if bits[p] != next {
                    bits[p] = next;
                    changed = true;
                }
            }
        }
    }
    let out = func.types.iter().enumerate().map(|(v, ty)| ty.is_small_int() && bits[v] <= ty.bits() as u8).collect();
    (out, bits)
}

/// The register each value is passed or returned in at a call or return, and the register a high product lands in.
pub(super) fn passed<R: Copy>(func: &Function, classify: &impl Fn(&Signature) -> Result<Abi<R>, String>, high_product: Option<R>) -> (Vec<Option<(R, bool)>>, Vec<Option<(u8, bool)>>) {
    let (mut ints, mut floats) = (vec![None; func.types.len()], vec![None; func.types.len()]);
    let mark = |vals: &[Value], locs: &[Loc<R>], arg: bool, ints: &mut [Option<(R, bool)>], floats: &mut [Option<(u8, bool)>]| {
        for (val, loc) in vals.iter().zip(locs) {
            match *loc {
                Loc::Int(reg) if arg || ints[val.index()].is_none() => ints[val.index()] = Some((reg, arg)),
                Loc::Float(reg) if arg || floats[val.index()].is_none() => floats[val.index()] = Some((reg, arg)),
                _ => {}
            }
        }
    };
    let ret_abi = classify(&func.signature).ok();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        match &func.insts[inst.index()].op {
            Op::Call { sig, args, results, .. } => {
                if let Ok(abi) = classify(&func.sigs[sig.index()]) {
                    mark(args, &abi.params, true, &mut ints, &mut floats);
                    mark(results, &abi.returns, false, &mut ints, &mut floats);
                }
            }
            Op::TryCall { sig, args, .. } => {
                if let Ok(abi) = classify(&func.sigs[sig.index()]) {
                    mark(args, &abi.params, true, &mut ints, &mut floats);
                }
            }
            Op::Binary(Binary::Umulhi | Binary::Smulhi, ..) => {
                if let (Some(reg), Some(res)) = (high_product, func.insts[inst.index()].result)
                    && func.ty(res) == Type::I64
                    && ints[res.index()].is_none()
                {
                    ints[res.index()] = Some((reg, false));
                }
            }
            Op::Return(vals) => {
                if let Some(abi) = &ret_abi {
                    mark(vals, &abi.returns, false, &mut ints, &mut floats);
                }
            }
            _ => {}
        }
    }
    (ints, floats)
}
