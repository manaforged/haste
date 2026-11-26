use super::super::Site;
use super::super::patterns::product_halves;
use super::super::sib::{idx_source, store_load_ok};
use super::{FuseCtx, preserved};
use crate::alloc::Plan;
use crate::ir::{
    Binary, BlockData, Convert, Function, Inst, InstData, Lane, Op, ShuffleMask, TargetOpId, Type,
    Unary, Value,
};
use crate::live::SibBase;
use crate::x64::asm::Reg;
use crate::x64::bits::atomic_addr_base;
use crate::x64::features::Features;
use crate::x64::vector::{float_vec_opcode, is_commutative, vec_alu_opcode};

pub(super) fn load_fuses(ctx: &FuseCtx<'_>, lpos: usize, eff_pos: usize, skip: &[usize]) -> bool {
    let insts = &ctx.func.blocks[ctx.b_idx].insts;
    if eff_pos <= lpos
        || insts[lpos + 1..eff_pos]
            .iter()
            .any(|&mid| ctx.func.insts[mid.index()].op.is_load_barrier())
    {
        return false;
    }
    match ctx.func.insts[insts[lpos].index()].op {
        Op::StackLoad(..) => true,
        Op::Load(addr, _) | Op::AtomicLoad(addr, _) if ctx.plan.elided[addr.index()] => {
            if let Some(m) = ctx.plan.sib[addr.index()] {
                let base_ok = match m.base {
                    SibBase::Frame(_) => true,
                    SibBase::Value(b) => {
                        preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, b)
                    }
                };
                let idx = idx_source(ctx.func, ctx.plan, ctx.defs, m.index);
                base_ok && preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, idx)
            } else {
                atomic_addr_base(ctx.func, ctx.plan, ctx.defs, addr).is_some_and(|(base, _)| {
                    !ctx.plan.elided[base.index()]
                        && preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, base)
                })
            }
        }
        Op::Load(addr, _) | Op::AtomicLoad(addr, _) => {
            preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, addr)
        }
        _ => false,
    }
}

pub(super) fn addr_reads(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    addr: Value,
    reg: Reg,
) -> bool {
    if plan.elided[addr.index()] {
        if let Some(m) = plan.sib[addr.index()] {
            let idx = idx_source(func, plan, defs, m.index);
            return matches!(
                m.base,
                SibBase::Value(base) if !plan.elided[base.index()] && plan.reg_of(base) == Some(reg)
            ) || (!plan.elided[idx.index()] && plan.reg_of(idx) == Some(reg));
        }
        return atomic_addr_base(func, plan, defs, addr)
            .is_some_and(|(base, _)| plan.reg_of(base) == Some(reg));
    }
    plan.reg_of(addr) == Some(reg)
}

pub(super) fn reads_reg(ctx: &FuseCtx<'_>, v: Value, reg: Reg) -> bool {
    ctx.plan.reg_of(v) == Some(reg)
        || ctx.pos_of[v.index()].is_some_and(|(_, _, inst)| {
            matches!(
                ctx.func.insts[inst.index()].op,
                Op::Load(addr, _) | Op::AtomicLoad(addr, _)
                    if addr_reads(ctx.func, ctx.plan, ctx.defs, addr, reg)
            )
        })
}

fn load_dst_clobber(
    ctx: &FuseCtx<'_>,
    linst: Inst,
    load_res: Value,
    (u_res, op): (Option<Value>, Binary),
    (a, b): (Value, Value),
) -> bool {
    let Some(r_dst) = u_res.and_then(|r| ctx.plan.reg_of(r)) else {
        return false;
    };
    if op == Binary::Isub && a == load_res {
        return false;
    }
    let other = if a == load_res { b } else { a };
    if ctx.plan.reg_of(other) == Some(r_dst) {
        return false;
    }
    let (Op::Load(addr, _) | Op::AtomicLoad(addr, _)) = ctx.func.insts[linst.index()].op else {
        return false;
    };
    addr_reads(ctx.func, ctx.plan, ctx.defs, addr, r_dst)
}

fn bnot_elided_ok(ctx: &FuseCtx<'_>, features: &Features, op: Binary, other: Value) -> bool {
    op == Binary::Band
        && features.bmi1
        && ctx.defs[other.index()].is_some_and(|i| {
            matches!(
                ctx.func.insts[i.index()].op,
                Op::Unary(Unary::Bnot, inner) if !ctx.plan.elided[inner.index()]
            )
        })
}

fn load_binop_ok(
    ctx: &FuseCtx<'_>,
    features: &Features,
    linst: Inst,
    load_res: Value,
    udata: &InstData,
) -> bool {
    let Op::Binary(op, a, b) = udata.op else {
        return false;
    };
    let ty = ctx.func.ty(load_res);
    if !matches!(ty, Type::I32 | Type::I64) {
        return false;
    }
    match op {
        Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Imul => {
            let other = if a == load_res { b } else { a };
            udata
                .result
                .is_some_and(|r| ctx.plan.sib[r.index()].is_none())
                && (!ctx.plan.elided[other.index()] || bnot_elided_ok(ctx, features, op, other))
                && !load_dst_clobber(ctx, linst, load_res, (udata.result, op), (a, b))
        }
        Binary::Umulhi | Binary::Smulhi => {
            udata.result.is_some_and(|r| ctx.func.ty(r) == Type::I64)
        }
        Binary::Rotl | Binary::Rotr => {
            features.bmi2
                && a == load_res
                && ctx
                    .plan
                    .const_of(b)
                    .is_some_and(|c| (c & i64::from(ty.bits() - 1)) != 0)
        }
        Binary::Ishl | Binary::Ushr | Binary::Sshr => {
            features.bmi2 && a == load_res && ctx.plan.const_of(b).is_none()
        }
        _ => false,
    }
}

fn vec128_binary_op_ok(features: &Features, op: Binary, lane: Lane) -> bool {
    vec_alu_opcode(op, lane).is_some()
        || (float_vec_opcode(op).is_some() && lane.is_float())
        || (op == Binary::Imul && (lane == Lane::I16 || (lane == Lane::I32 && features.sse41)))
        || (matches!(
            op,
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax
        ) && lane != Lane::I64
            && (features.sse41
                || matches!(
                    (op, lane),
                    (Binary::Umin | Binary::Umax, Lane::I8)
                        | (Binary::Smin | Binary::Smax, Lane::I16)
                )))
        || (matches!(
            op,
            Binary::SaddSat
                | Binary::UaddSat
                | Binary::SsubSat
                | Binary::UsubSat
                | Binary::AvgRound
        ) && matches!(lane, Lane::I8 | Lane::I16))
}

fn vec_binary_load_ok(
    ctx: &FuseCtx<'_>,
    features: &Features,
    (lane, wide): (Lane, bool),
    load_res: Value,
    (op, a, b): (Binary, Value, Value),
) -> bool {
    let other = if a == load_res { b } else { a };
    if ctx.plan.elided[other.index()] || (a != load_res && b != load_res) {
        return false;
    }
    if wide {
        let packed = (vec_alu_opcode(op, lane).is_some()
            && (lane.is_int() || matches!(op, Binary::Band | Binary::Bor | Binary::Bxor)))
            || (float_vec_opcode(op).is_some() && lane.is_float());
        return (packed
            && (b == load_res || (is_commutative(op) && ctx.plan.float_of(b).is_some())))
            || (b == load_res
                && matches!(
                    (op, lane),
                    (Binary::Ishl | Binary::Ushr, Lane::I32 | Lane::I64)
                        | (Binary::Sshr, Lane::I32)
                ));
    }
    vec128_binary_op_ok(features, op, lane) && (b == load_res || is_commutative(op))
}

fn vec_target_load_ok(
    ctx: &FuseCtx<'_>,
    wide: bool,
    load_res: Value,
    id: TargetOpId,
    args: &[Value],
) -> bool {
    match (id, args) {
        (TargetOpId::X64Low128 | TargetOpId::X64High128, &[v]) => wide && v == load_res,
        (TargetOpId::X64Concat128, &[l, h]) => {
            !wide && h == load_res && !ctx.plan.elided[l.index()]
        }
        (
            TargetOpId::X64Pshufb256
            | TargetOpId::X64Pcmpeqb256
            | TargetOpId::X64Pcmpgtb256
            | TargetOpId::X64Psadbw256
            | TargetOpId::X64Pmaddubsw256
            | TargetOpId::X64Pmaddwd256,
            &[x, y],
        ) => wide && y == load_res && !ctx.plan.elided[x.index()],
        _ => false,
    }
}

fn wide_const_shuffle_load_ok(
    lane: Lane,
    count: u8,
    load_res: Value,
    (a, b): (Value, Value),
    indices: &[u8],
) -> bool {
    let q = [indices[0], indices[1], indices[2], indices[3]];
    if lane.bytes() != 8
        || count != 4
        || (q.iter().all(|&x| x == q[0]) && q[0] % 4 == 0)
        || q == [0, 1, 2, 3]
        || q == [4, 5, 6, 7]
    {
        return false;
    }
    if q.iter().all(|&x| x < 4) {
        return a == load_res;
    }
    if q.iter().all(|&x| x >= 4) {
        return b == load_res;
    }
    if (0..4).all(|i| usize::from(q[i]) % 4 == i)
        || (q[0] % 2 == 0 && q[1] == q[0] + 1 && q[2] % 2 == 0 && q[3] == q[2] + 1)
    {
        return b == load_res;
    }
    match q {
        [0, 4, 2, 6] | [1, 5, 3, 7] | [5, 0, 7, 2] => b == load_res,
        [4, 0, 6, 2] | [5, 1, 7, 3] | [1, 4, 3, 6] => a == load_res,
        _ => true,
    }
}

fn vec128_const_shuffle_load_ok(
    features: &Features,
    lane: Lane,
    count: u8,
    load_res: Value,
    (a, b): (Value, Value),
    indices: &[u8],
) -> bool {
    let (c, bytes) = (usize::from(count), lane.bytes() as usize);
    if indices[..c].iter().all(|&i| i < count)
        || indices[..c].iter().all(|&i| i >= count && i < 2 * count)
    {
        return false;
    }
    if lane.bytes() == 8 && count == 2 && matches!((indices[0], indices[1]), (0, 2) | (1, 3)) {
        return b == load_res;
    }
    let bidx = |k: usize| {
        let i = indices[k / bytes];
        if i < count {
            i * bytes as u8 + (k % bytes) as u8
        } else if i < 2 * count {
            16 + (i - count) * bytes as u8 + (k % bytes) as u8
        } else {
            0x80
        }
    };
    let palignr_ok = features.ssse3
        && (1..16).contains(&bidx(0))
        && (0..16).all(|k| bidx(k) == bidx(0) + k as u8)
        && a == load_res;
    let pblendw_ok = features.sse41
        && b == load_res
        && (0..8).all(|w| {
            let (lo, hi) = (bidx(2 * w), bidx(2 * w + 1));
            (lo == 16 + 2 * w as u8 && hi == 17 + 2 * w as u8)
                || (lo == 2 * w as u8 && hi == 2 * w as u8 + 1)
        });
    palignr_ok || pblendw_ok
}

fn vec_load_elide_ok(
    ctx: &FuseCtx<'_>,
    features: &Features,
    load_res: Value,
    udata: &InstData,
) -> bool {
    let ty = ctx.func.ty(load_res);
    let wide = ty.is_wide_vec();
    let Some((lane, count)) = ty
        .vec_shape()
        .filter(|_| features.avx && (wide || (ty.is_valid_vec() && ty.bytes() == 16)))
    else {
        return false;
    };
    match udata.op {
        Op::Unary(Unary::Ineg, x) => wide && lane.is_int() && x == load_res,
        Op::Binary(op, a, b) => {
            vec_binary_load_ok(ctx, features, (lane, wide), load_res, (op, a, b))
        }
        Op::TargetOp(id, ref args) => vec_target_load_ok(ctx, wide, load_res, id, args.as_slice()),
        Op::Shuffle(a, b, ShuffleMask::Dyn(m)) => {
            !wide
                && features.ssse3
                && a == b
                && m == load_res
                && ty
                    == Type::Vec {
                        lane: Lane::I8,
                        count: 16,
                    }
                && ctx.func.ty(a) == ty
                && !ctx.plan.elided[a.index()]
        }
        Op::Shuffle(a, b, ShuffleMask::Const(indices))
            if a != b
                && (a == load_res || b == load_res)
                && !ctx.plan.elided[(if a == load_res { b } else { a }).index()] =>
        {
            if wide {
                wide_const_shuffle_load_ok(lane, count, load_res, (a, b), &indices)
            } else {
                vec128_const_shuffle_load_ok(features, lane, count, load_res, (a, b), &indices)
            }
        }
        _ => false,
    }
}

fn carry_load_ok(
    ctx: &FuseCtx<'_>,
    linst: Inst,
    load_res: Value,
    (args, result): (&[Value], Option<Value>),
) -> bool {
    let &[carry, a, b] = args else {
        return false;
    };
    if b != load_res || a == b || carry == b || ctx.func.ty(load_res) != Type::I64 {
        return false;
    }
    match ctx.func.insts[linst.index()].op {
        Op::Load(addr, _) => {
            let result_reg = result.and_then(|r| ctx.plan.reg_of(r));
            ![Reg::Rax, Reg::Rcx, Reg::Rdx]
                .into_iter()
                .chain(result_reg)
                .any(|reg| addr_reads(ctx.func, ctx.plan, ctx.defs, addr, reg))
        }
        Op::StackLoad(..) => true,
        _ => false,
    }
}

fn load_elide_ok(
    ctx: &FuseCtx<'_>,
    features: &Features,
    linst: Inst,
    load_res: Value,
    udata: &InstData,
) -> bool {
    let ty = ctx.func.ty(load_res);
    if ty.is_vector() {
        return features.avx && vec_load_elide_ok(ctx, features, load_res, udata);
    }
    let non_wide = udata.result.is_some_and(|r| ctx.func.ty(r) != Type::I128);
    match udata.op {
        Op::Icmp(..) => true,
        Op::Brif(cond, ..) | Op::Trapz(cond) => cond == load_res,
        Op::Store(..) | Op::StackStore(..) => {
            !matches!(ctx.func.insts[linst.index()].op, Op::AtomicLoad(..))
                && store_load_ok(ctx.func, ctx.plan, linst, load_res, &udata.op)
        }
        Op::Convert(Convert::Uextend | Convert::Sextend, _) => ty.bytes() <= 4 && non_wide,
        Op::Convert(Convert::Ireduce, _) => ty.bytes() <= 8 && non_wide,
        Op::Binary(..) => load_binop_ok(ctx, features, linst, load_res, udata),
        Op::TargetOp(
            TargetOpId::X64AddCarry
            | TargetOpId::X64AddCarryOut
            | TargetOpId::X64SubBorrow
            | TargetOpId::X64SubBorrowOut,
            ref args,
        ) => carry_load_ok(ctx, linst, load_res, (args.as_slice(), udata.result)),
        Op::Unary(Unary::Clz, _) => matches!(ty, Type::I32 | Type::I64) && features.lzcnt,
        Op::Unary(Unary::Ctz, _) => matches!(ty, Type::I32 | Type::I64) && features.bmi1,
        Op::Unary(Unary::Popcnt, _) => matches!(ty, Type::I32 | Type::I64) && features.popcnt,
        _ => false,
    }
}

pub(super) fn elide_loads(
    func: &Function,
    plan: &mut Plan<Reg>,
    features: &Features,
    (uses, pos_of, defs, use_of): (&[u32], &[Site], &[Option<Inst>], &[Site]),
    tst_end: &[Option<usize>],
    cand_loads: Vec<(usize, usize, Value)>,
) {
    for (b_idx, lpos, load_res) in cand_loads {
        if plan.elided[load_res.index()] {
            continue;
        }
        let Some((ub, upos, uinst)) = use_of[load_res.index()] else {
            continue;
        };
        if ub != b_idx || upos <= lpos {
            continue;
        }
        let ctx = FuseCtx {
            func,
            plan,
            uses,
            pos_of,
            defs,
            b_idx,
        };
        let udata = &func.insts[uinst.index()];
        let eff_pos = if let Some(ur) = udata.result.filter(|r| plan.elided[r.index()]) {
            let Some(fpos) = tst_end[ur.index()] else {
                continue;
            };
            fpos
        } else {
            if !load_elide_ok(
                &ctx,
                features,
                func.blocks[b_idx].insts[lpos],
                load_res,
                udata,
            ) {
                continue;
            }
            upos
        };
        if load_fuses(&ctx, lpos, eff_pos, &[]) {
            plan.elided[load_res.index()] = true;
        }
    }
}

pub(super) fn elide_zero_bases(func: &Function, plan: &mut Plan<Reg>, defs: &[Option<Inst>]) {
    let mut read = vec![false; func.types.len()];
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &func.insts[inst.index()].op;
        let base = match *op {
            Op::Insert(base, scalar, 0)
                if base != scalar && func.ty(base).lane().is_some_and(Lane::is_int) =>
            {
                Some(base)
            }
            _ => None,
        };
        op.for_operands(|v| read[v.index()] |= Some(v) != base);
    }
    for (v, def) in defs.iter().enumerate() {
        if !read[v]
            && func.types[v].is_vector()
            && def.is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(0)))
        {
            plan.elided[v] = true;
        }
    }
}

pub(super) fn fold_products(
    func: &Function,
    plan: &mut Plan<Reg>,
    uses: &[u32],
    defs: &[Option<Inst>],
) {
    for block in &func.blocks {
        for pos in 2..block.insts.len() {
            let data = &func.insts[block.insts[pos].index()];
            let (Op::Binary(Binary::Bxor | Binary::Bor | Binary::Iadd, x, y), Some(result)) =
                (&data.op, data.result)
            else {
                continue;
            };
            let halves = [block.insts[pos - 2], block.insts[pos - 1]]
                .map(|inst| func.insts[inst.index()].result);
            if low_of_pair(func, block, pos - 1).is_none()
                || !halves.iter().all(|half| {
                    half.is_some_and(|half| [*x, *y].contains(&half) && uses[half.index()] == 1)
                })
                || plan.elided[result.index()]
                || plan.sib[result.index()].is_some()
                || product_halves(func, &|v| defs[v.index()], (*x, *y)).is_none()
            {
                continue;
            }
            for half in halves.into_iter().flatten() {
                plan.elided[half.index()] = true;
            }
        }
    }
}

pub(super) fn is_load(func: &Function, defs: &[Option<Inst>], value: Value) -> bool {
    defs[value.index()].is_some_and(|inst| {
        matches!(
            func.insts[inst.index()].op,
            Op::Load(..) | Op::StackLoad(..)
        )
    })
}

pub(super) fn low_of_pair(
    func: &Function,
    block: &BlockData,
    pos: usize,
) -> Option<(Value, [Value; 2])> {
    let data = &func.insts[block.insts[pos].index()];
    let (Op::Binary(Binary::Imul, x, y), Some(low)) = (&data.op, data.result) else {
        return None;
    };
    let before = &func.insts[block.insts[pos.checked_sub(1)?].index()];
    let (Op::Binary(Binary::Umulhi | Binary::Smulhi, a, b), Some(high)) =
        (&before.op, before.result)
    else {
        return None;
    };
    ((a, b) == (x, y) || (a, b) == (y, x))
        .then_some(())
        .filter(|()| func.ty(low) == Type::I64 && func.ty(high) == Type::I64)?;
    Some((low, [*x, *y]))
}

pub(super) fn restore_paired_loads(
    func: &Function,
    plan: &mut Plan<Reg>,
    defs: &[Option<Inst>],
    paired: &[(Value, [Value; 2])],
) {
    for &(low, operands) in paired {
        if plan.is_static(low) || plan.deferred[low.index()] {
            for value in operands
                .into_iter()
                .filter(|&value| is_load(func, defs, value))
            {
                plan.elided[value.index()] = false;
            }
        }
    }
}
