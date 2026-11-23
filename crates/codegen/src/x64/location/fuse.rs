use super::patterns::{
    Root, adc_fold, adc_parts, adc_sides, bextr_parts, flag_cmp, is_zero_const, mid_preserves_flags, parked, shld_parts,
};
use super::sib::{apply_sib_groups, double, find_cand_sib, idx_source, sib_ok};
use super::super::{FRAME_REGS, REGS};
use super::super::asm::Reg;
use super::super::features::Features;
use super::super::frame::SAVED;
use super::super::vector::{float_vec_opcode, is_commutative, vec_alu_opcode};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Block, Convert, Function, Inst, InstData, IntCC, Lane, Op, ShuffleMask, TargetOpId, Type, Unary, Value};
use crate::live::{Sib, SibBase};

pub(super) struct FuseCtx<'a> {
    pub(super) func: &'a Function,
    pub(super) plan: &'a Plan<Reg>,
    pub(super) uses: &'a [u32],
    pub(super) pos_of: &'a [super::Site],
    pub(super) defs: &'a [Option<Inst>],
    pub(super) b_idx: usize,
}

pub(super) fn preserved(func: &Function, plan: &Plan<Reg>, b_idx: usize, start: usize, end: usize, skip: &[usize], val: Value) -> bool {
    crate::emit::preserved(func, (plan, &REGS), (b_idx, start, end, 128), |pos| skip.contains(&pos), val)
}

fn band_ok(func: &Function, plan: &Plan<Reg>, b_idx: usize, (mpos, end, skip): (usize, usize, &[usize]), (x, y): (Value, Value)) -> bool {
    func.ty(x).is_small_int() && preserved(func, plan, b_idx, mpos, end, skip, x) && preserved(func, plan, b_idx, mpos, end, skip, y)
}

fn zero_cmp_band(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    pos_of: &[super::Site],
    b_idx: usize,
    lhs: Value,
    rhs: Value,
) -> Option<(Value, usize, Value, Value)> {
    crate::emit::zero_cmp_band(func, plan, (uses, pos_of), b_idx, (lhs, rhs), |v| is_zero_const(func, plan, v))
}

pub(super) fn may_barrier_mem(op: &Op) -> bool {
    matches!(
        op,
        Op::Store(..)
            | Op::StackStore(..)
            | Op::Copy(..)
            | Op::Move(..)
            | Op::Fence(..)
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Trap
            | Op::Trapz(..)
            | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..)
    )
}

fn load_fuses(ctx: &FuseCtx<'_>, lpos: usize, eff_pos: usize, skip: &[usize]) -> bool {
    let insts = &ctx.func.blocks[ctx.b_idx].insts;
    if eff_pos <= lpos || insts[lpos + 1..eff_pos].iter().any(|&mid| may_barrier_mem(&ctx.func.insts[mid.index()].op)) {
        return false;
    }
    match ctx.func.insts[insts[lpos].index()].op {
        Op::StackLoad(..) => true,
        Op::Load(addr, _) if ctx.plan.elided[addr.index()] => ctx.plan.sib[addr.index()].is_some_and(|m| {
            let base_ok = match m.base {
                SibBase::Frame(_) => true,
                SibBase::Value(b) => preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, b),
            };
            let idx = idx_source(ctx.func, ctx.plan, ctx.defs, m.index);
            base_ok && preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, idx)
        }),
        Op::Load(addr, _) => preserved(ctx.func, ctx.plan, ctx.b_idx, lpos, eff_pos, skip, addr),
        _ => false,
    }
}

fn addr_reads(func: &Function, plan: &Plan<Reg>, defs: &[Option<Inst>], addr: Value, reg: Reg) -> bool {
    if plan.elided[addr.index()] {
        return plan.sib[addr.index()].is_some_and(|m| {
            let idx = idx_source(func, plan, defs, m.index);
            matches!(m.base, SibBase::Value(base) if !plan.elided[base.index()] && plan.reg_of(base) == Some(reg))
                || (!plan.elided[idx.index()] && plan.reg_of(idx) == Some(reg))
        });
    }
    plan.reg_of(addr) == Some(reg)
}

fn reads_reg(ctx: &FuseCtx<'_>, v: Value, reg: Reg) -> bool {
    ctx.plan.reg_of(v) == Some(reg)
        || ctx.pos_of[v.index()].is_some_and(|(_, _, inst)| matches!(ctx.func.insts[inst.index()].op, Op::Load(addr, _) if addr_reads(ctx.func, ctx.plan, ctx.defs, addr, reg)))
}

fn load_dst_clobber(ctx: &FuseCtx<'_>, linst: Inst, load_res: Value, (u_res, op): (Option<Value>, Binary), (a, b): (Value, Value)) -> bool {
    let Some(r_dst) = u_res.and_then(|r| ctx.plan.reg_of(r)) else { return false };
    if op == Binary::Isub && a == load_res {
        return false;
    }
    let other = if a == load_res { b } else { a };
    if ctx.plan.reg_of(other) == Some(r_dst) {
        return false;
    }
    let Op::Load(addr, _) = ctx.func.insts[linst.index()].op else { return false };
    addr_reads(ctx.func, ctx.plan, ctx.defs, addr, r_dst)
}

pub(in crate::x64) fn iadd_sub_const(plan: &Plan<Reg>, op: &Op) -> Option<(Value, i64)> {
    crate::alloc::iadd_sub_const(op, |v| plan.const_of(v))
}

fn cmp_band_arg_ok(ctx: &FuseCtx<'_>, (mpos, cpos, last_pos): (usize, usize, usize), u: Value) -> bool {
    if !ctx.plan.elided[u.index()] {
        return preserved(ctx.func, ctx.plan, ctx.b_idx, mpos, last_pos, &[cpos], u);
    }
    ctx.pos_of[u.index()].is_some_and(|(lb, lpos, linst)| {
        lb == ctx.b_idx
            && lpos < mpos
            && matches!(ctx.func.insts[linst.index()].op, Op::Load(..) | Op::StackLoad(..))
            && load_fuses(ctx, lpos, last_pos, &[cpos])
    })
}

fn cmp_operand_ok(ctx: &FuseCtx<'_>, tst_end: &[Option<usize>], cpos: usize, last_pos: usize, v: Value) -> bool {
    if !ctx.plan.elided[v.index()] {
        return preserved(ctx.func, ctx.plan, ctx.b_idx, cpos, last_pos, &[], v);
    }
    let Some((lb, lpos, linst)) = ctx.pos_of[v.index()] else { return false };
    if lb != ctx.b_idx || lpos >= cpos {
        return false;
    }
    match ctx.func.insts[linst.index()].op {
        Op::Load(..) | Op::StackLoad(..) => load_fuses(ctx, lpos, last_pos, &[]),
        Op::Binary(Binary::Band, x, y) if tst_end[v.index()].is_some() => {
            [x, y].into_iter().all(|u| cmp_band_arg_ok(ctx, (lpos, cpos, last_pos), u))
        }
        _ => false,
    }
}

fn push_branch_tst(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[super::Site]),
    (b_idx, pos, cond_val): (usize, usize, Value),
    out: &mut Vec<(usize, Value, Option<Value>)>,
) {
    if uses[cond_val.index()] != 1 || plan.elided[cond_val.index()] { return; }
    let Some((cb, cpos, cinst)) = pos_of[cond_val.index()] else { return };
    if cb != b_idx || cpos >= pos { return; }
    match func.insts[cinst.index()].op {
        Op::Binary(Binary::Band, x, y) if band_ok(func, plan, b_idx, (cpos, pos, &[]), (x, y)) => out.push((pos, cond_val, None)),
        Op::Convert(Convert::Ireduce, m)
            if uses[m.index()] == 1 && !plan.elided[m.index()]
                && let Some((mb, mpos, minst)) = pos_of[m.index()]
                && mb == b_idx && mpos < cpos
                && let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op
                && narrows_exactly(func, plan, &|v| pos_of[v.index()].map(|(_, _, i)| i), m, func.ty(cond_val))
                && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y)) => out.push((pos, cond_val, Some(m))),
        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs)
            if let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)
                && mpos < cpos
                && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y)) => out.push((pos, cond_val, Some(m))),
        _ => {}
    }
}

fn select_arg_ok(ctx: &FuseCtx<'_>, (start, pos, skip): (usize, usize, &[usize]), dst: Option<Reg>, v: Value) -> bool {
    let folds = ctx.uses[v.index()] == 1
        && !ctx.plan.elided[v.index()]
        && ctx.func.ty(v).is_small_int()
        && ctx.pos_of[v.index()].is_some_and(|(vb, lpos, linst)| {
            vb == ctx.b_idx && matches!(ctx.func.insts[linst.index()].op, Op::Load(..) | Op::StackLoad(..)) && load_fuses(ctx, lpos, pos, skip)
        });
    (folds || preserved(ctx.func, ctx.plan, ctx.b_idx, start, pos, skip, v)) && dst.is_none_or(|d| !reads_reg(ctx, v, d))
}

fn elided_band_args(ctx: &FuseCtx<'_>, tst_end: &[Option<usize>], cc: IntCC, v: Value) -> Option<(usize, [Value; 2])> {
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual) || !ctx.plan.elided[v.index()] || tst_end[v.index()].is_none() {
        return None;
    }
    let (vb, mpos, minst) = ctx.pos_of[v.index()]?;
    let Op::Binary(Binary::Band, bx, by) = ctx.func.insts[minst.index()].op else { return None };
    (vb == ctx.b_idx).then_some((mpos, [bx, by]))
}

fn select_leaf_ok(ctx: &FuseCtx<'_>, tst_end: &[Option<usize>], (pos, skip): (usize, &[usize]), dst: Option<Reg>, leaf: Value) -> Option<Option<Value>> {
    let (_, leaf_pos, leaf_inst) = ctx.pos_of[leaf.index()]?;
    let Op::Icmp(cc, a, b) = ctx.func.insts[leaf_inst.index()].op else { return None };
    let mut band = None;
    for v in [a, b] {
        if let Some((mpos, args)) = elided_band_args(ctx, tst_end, cc, v) {
            if !args.into_iter().all(|u| select_arg_ok(ctx, (mpos, pos, skip), dst, u)) {
                return None;
            }
            band = Some(v);
        } else if !select_arg_ok(ctx, (leaf_pos, pos, skip), dst, v) {
            return None;
        }
    }
    Some(band)
}

fn apply_select_at(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[super::Site], &[Option<Inst>]),
    (b_idx, pos, inst): (usize, usize, Inst),
    (tst_end, roots): (&mut [Option<usize>], &mut [Option<Root>]),
) {
    let data = &func.insts[inst.index()];
    let (Op::Select(cond, x, y), Some(res)) = (&data.op, data.result) else { return };
    let (first, second, table, nodes, other, dst) = {
        let ctx = FuseCtx { func, plan, uses, pos_of, defs, b_idx };
        let Some(tree) = ctx.cond_tree(*cond, pos) else { return };
        let ([first, second], Location::Reg(dst)) = (&tree.leaves[..], plan.location(res)) else { return };
        let other = if tree.table.count_ones() == 1 { *y } else { *x };
        (*first, *second, tree.table, tree.nodes, other, dst)
    };
    if !func.ty(*x).is_small_int() || x == y || plan.reg_of(other) == Some(dst) {
        return;
    }
    let skip: Vec<usize> = nodes.iter().map(|node| node.1).collect();
    let bands = {
        let ctx = FuseCtx { func, plan, uses, pos_of, defs, b_idx };
        let Some(b1) = select_leaf_ok(&ctx, tst_end, (pos, &skip), None, first) else { return };
        let Some(b2) = select_leaf_ok(&ctx, tst_end, (pos, &skip), Some(dst), second) else { return };
        [b1, b2]
    };
    for &(v, ..) in &nodes {
        plan.elided[v.index()] = true;
    }
    tst_end[first.index()] = Some(pos);
    tst_end[second.index()] = Some(pos);
    for b in bands.into_iter().flatten() {
        tst_end[b.index()] = Some(pos);
    }
    roots[cond.index()] = Some((first, Some(second), table));
}

fn apply_select_trees(
    func: &Function,
    plan: &mut Plan<Reg>,
    tables: (&[u32], &[super::Site], &[Option<Inst>]),
    tst_end: &mut [Option<usize>],
    roots: &mut [Option<Root>],
) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            apply_select_at(func, plan, tables, (b_idx, pos, inst), (tst_end, roots));
        }
    }
}

fn bnot_elided_ok(ctx: &FuseCtx<'_>, features: &Features, op: Binary, other: Value) -> bool {
    op == Binary::Band
        && features.bmi1
        && ctx.defs[other.index()]
            .is_some_and(|i| matches!(ctx.func.insts[i.index()].op, Op::Unary(Unary::Bnot, inner) if !ctx.plan.elided[inner.index()]))
}

fn load_binop_ok(ctx: &FuseCtx<'_>, features: &Features, linst: Inst, load_res: Value, udata: &InstData) -> bool {
    let Op::Binary(op, a, b) = udata.op else { return false };
    let ty = ctx.func.ty(load_res);
    if !matches!(ty, Type::I32 | Type::I64) {
        return false;
    }
    match op {
        Binary::Iadd | Binary::Isub | Binary::Band | Binary::Bor | Binary::Bxor | Binary::Imul => {
            let other = if a == load_res { b } else { a };
            udata.result.is_some_and(|r| ctx.plan.sib[r.index()].is_none())
                && (!ctx.plan.elided[other.index()] || bnot_elided_ok(ctx, features, op, other))
                && !load_dst_clobber(ctx, linst, load_res, (udata.result, op), (a, b))
        }
        Binary::Umulhi | Binary::Smulhi => udata.result.is_some_and(|r| ctx.func.ty(r) == Type::I64),
        Binary::Rotl | Binary::Rotr => {
            features.bmi2 && a == load_res && ctx.plan.const_of(b).is_some_and(|c| (c & i64::from(ty.bits() - 1)) != 0)
        }
        Binary::Ishl | Binary::Ushr | Binary::Sshr => features.bmi2 && a == load_res && ctx.plan.const_of(b).is_none(),
        _ => false,
    }
}

fn vec_load_elide_ok(ctx: &FuseCtx<'_>, features: &Features, load_res: Value, udata: &InstData) -> bool {
    let (ty, wide) = (ctx.func.ty(load_res), ctx.func.ty(load_res).is_wide_vec());
    let Some((lane, count)) = ty.vec_shape().filter(|_| features.avx && (wide || (ty.is_valid_vec() && ty.bytes() == 16))) else { return false };
    match udata.op {
        Op::Unary(Unary::Ineg, x) => wide && lane.is_int() && x == load_res,
        Op::Binary(op, a, b) => {
            let other = if a == load_res { b } else { a };
            if ctx.plan.elided[other.index()] || (a != load_res && b != load_res) { return false; }
            if wide {
                let packed = (vec_alu_opcode(op, lane).is_some() && (lane.is_int() || matches!(op, Binary::Band | Binary::Bor | Binary::Bxor))) || (float_vec_opcode(op).is_some() && lane.is_float());
                return (packed && (b == load_res || (is_commutative(op) && ctx.plan.float_of(b).is_some()))) || (b == load_res && matches!((op, lane), (Binary::Ishl | Binary::Ushr, Lane::I32 | Lane::I64) | (Binary::Sshr, Lane::I32)));
            }
            let ok = vec_alu_opcode(op, lane).is_some() || (float_vec_opcode(op).is_some() && lane.is_float()) || (op == Binary::Imul && (lane == Lane::I16 || (lane == Lane::I32 && features.sse41))) || (matches!(op, Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax) && lane != Lane::I64 && (features.sse41 || matches!((op, lane), (Binary::Umin | Binary::Umax, Lane::I8) | (Binary::Smin | Binary::Smax, Lane::I16)))) || (matches!(op, Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat | Binary::AvgRound) && matches!(lane, Lane::I8 | Lane::I16));
            ok && (b == load_res || is_commutative(op))
        }
        Op::TargetOp(id, ref args) => match (id, args.as_slice()) {
            (TargetOpId::X64Low128 | TargetOpId::X64High128, &[v]) => wide && v == load_res,
            (TargetOpId::X64Concat128, &[l, h]) => !wide && h == load_res && !ctx.plan.elided[l.index()],
            (TargetOpId::X64Pshufb256 | TargetOpId::X64Pcmpeqb256 | TargetOpId::X64Pcmpgtb256 | TargetOpId::X64Psadbw256 | TargetOpId::X64Pmaddubsw256 | TargetOpId::X64Pmaddwd256, &[x, y]) => wide && y == load_res && !ctx.plan.elided[x.index()],
            _ => false,
        },
        Op::Shuffle(a, b, ShuffleMask::Dyn(m)) => !wide && features.ssse3 && a == b && m == load_res && ty == Type::Vec { lane: Lane::I8, count: 16 } && ctx.func.ty(a) == ty && !ctx.plan.elided[a.index()],
        Op::Shuffle(a, b, ShuffleMask::Const(indices)) if a != b && (a == load_res || b == load_res) && !ctx.plan.elided[(if a == load_res { b } else { a }).index()] => {
            if wide {
                let q = [indices[0], indices[1], indices[2], indices[3]];
                if lane.bytes() != 8 || count != 4 || (q.iter().all(|&x| x == q[0]) && q[0] % 4 == 0) || q == [0, 1, 2, 3] || q == [4, 5, 6, 7] { return false; }
                if q.iter().all(|&x| x < 4) { return a == load_res; } else if q.iter().all(|&x| x >= 4) { return b == load_res; }
                if (0..4).all(|i| usize::from(q[i]) % 4 == i) || (q[0] % 2 == 0 && q[1] == q[0] + 1 && q[2] % 2 == 0 && q[3] == q[2] + 1) { return b == load_res; }
                return match q { [0, 4, 2, 6] | [1, 5, 3, 7] | [5, 0, 7, 2] => b == load_res, [4, 0, 6, 2] | [5, 1, 7, 3] | [1, 4, 3, 6] => a == load_res, _ => true };
            }
            let (c, bytes) = (usize::from(count), lane.bytes() as usize);
            if indices[..c].iter().all(|&i| i < count) || indices[..c].iter().all(|&i| i >= count && i < 2 * count) { return false; }
            if lane.bytes() == 8 && count == 2 && matches!((indices[0], indices[1]), (0, 2) | (1, 3)) { return b == load_res; }
            let bidx = |k: usize| { let i = indices[k / bytes]; if i < count { i * bytes as u8 + (k % bytes) as u8 } else if i < 2 * count { 16 + (i - count) * bytes as u8 + (k % bytes) as u8 } else { 0x80 } };
            (features.ssse3 && (1..16).contains(&bidx(0)) && (0..16).all(|k| bidx(k) == bidx(0) + k as u8) && a == load_res) || (features.sse41 && b == load_res && (0..8).all(|w| { let (lo, hi) = (bidx(2 * w), bidx(2 * w + 1)); (lo == 16 + 2 * w as u8 && hi == 17 + 2 * w as u8) || (lo == 2 * w as u8 && hi == 2 * w as u8 + 1) }))
        }
        _ => false,
    }
}

fn load_elide_ok(ctx: &FuseCtx<'_>, features: &Features, linst: Inst, load_res: Value, udata: &InstData) -> bool {
    let ty = ctx.func.ty(load_res);
    if ty.is_vector() { return features.avx && vec_load_elide_ok(ctx, features, load_res, udata); }
    let non_wide = udata.result.is_some_and(|r| ctx.func.ty(r) != Type::I128);
    match udata.op {
        Op::Icmp(..) => true,
        Op::Brif(cond, ..) | Op::Trapz(cond) => cond == load_res,
        Op::Store(..) | Op::StackStore(..) => super::sib::store_load_ok(ctx.func, ctx.plan, linst, load_res, &udata.op),
        Op::Convert(Convert::Uextend | Convert::Sextend, _) => ty.bytes() <= 4 && non_wide,
        Op::Convert(Convert::Ireduce, _) => ty.bytes() <= 8 && non_wide,
        Op::Binary(..) => load_binop_ok(ctx, features, linst, load_res, udata),
        Op::TargetOp(TargetOpId::X64AddCarry | TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrow | TargetOpId::X64SubBorrowOut, ref args) => {
            carry_load_ok(ctx, linst, load_res, (args.as_slice(), udata.result))
        }
        Op::Unary(Unary::Clz, _) => matches!(ty, Type::I32 | Type::I64) && features.lzcnt,
        Op::Unary(Unary::Ctz, _) => matches!(ty, Type::I32 | Type::I64) && features.bmi1,
        Op::Unary(Unary::Popcnt, _) => matches!(ty, Type::I32 | Type::I64) && features.popcnt,
        _ => false,
    }
}

/// Whether narrowing `x` to `to` drops only bits known to be zero, so the zero flag of `x` is the zero flag of the narrowed value.
pub(in super::super) fn narrows_exactly(func: &Function, plan: &Plan<Reg>, def: &dyn Fn(Value) -> Option<Inst>, x: Value, to: Type) -> bool {
    let (zeros, _) = crate::fold::known_bits(func, def, &|v| plan.const_of(v), x, 0);
    func.ty(x).unsigned_max() & !to.unsigned_max() & !zeros == 0
}

pub(in super::super) fn fuse(func: &Function, plan: &mut Plan<Reg>, features: &Features) -> (Vec<Option<Root>>, Vec<bool>) {
    let n = func.types.len();
    let (mut roots, mut pure_test) = (vec![None; n], vec![false; n]);
    if func.size() > 8192 {
        return (roots, pure_test);
    }
    let defs = crate::fold::defs(func);
    let (mut uses, mut pos_of, mut ls_uses, mut use_of) = (vec![0_u32; n], vec![None; n], vec![Vec::new(); n], vec![None; n]);
    let (mut last_flag_use, mut non_flag_use, mut const_kids) = (vec![None; n], vec![false; n], vec![Vec::new(); n]);
    let mut paired = Vec::new();
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let flag_cond = match &data.op {
                Op::Select(c, ..) | Op::Brif(c, ..) | Op::Trapz(c) => Some(*c),
                _ => None,
            };
            let pair = low_of_pair(func, block, pos);
            paired.extend(pair);
            let reads_nothing = pair.is_some() || crate::live::carry_out_of_pair(func, &block.insts, pos);
            data.op.for_operands(|v| {
                if reads_nothing && is_load(func, &defs, v) {
                    return;
                }
                uses[v.index()] += 1;
                use_of[v.index()] = Some((b_idx, pos, inst));
                if flag_cond == Some(v) && !non_flag_use[v.index()] {
                    match last_flag_use[v.index()] {
                        Some((pb, ppos)) if pb != b_idx || ppos == pos => non_flag_use[v.index()] = true,
                        _ => last_flag_use[v.index()] = Some((b_idx, pos)),
                    }
                } else {
                    non_flag_use[v.index()] = true;
                }
            });
            match data.op {
                Op::Load(addr, off) => ls_uses[addr.index()].push((b_idx, pos, off)),
                Op::Store(val, addr, off) if val != addr => ls_uses[addr.index()].push((b_idx, pos, off)),
                _ => {}
            }
            if let Some(res) = data.result {
                pos_of[res.index()] = Some((b_idx, pos, inst));
                if func.ty(res) == Type::I64
                    && plan.sib[res.index()].is_none()
                    && !plan.elided[res.index()]
                    && let Some((parent, delta)) = iadd_sub_const(plan, &data.op)
                {
                    const_kids[parent.index()].push((b_idx, pos, res, delta));
                }
            }
        }
    }
    let (mut branch_tst, mut cmp_tst, mut cand_sibs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cand_shld, mut cand_adc, mut cand_bextr, mut cand_loads, mut cand_alu) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else {
                if let Op::Brif(cond_val, ..) | Op::Trapz(cond_val) = data.op {
                    push_branch_tst(func, plan, (&uses, &pos_of), (b_idx, pos, cond_val), &mut branch_tst);
                }
                continue;
            };
            let ty = func.ty(res);
            match data.op {
                Op::Select(cond_val, x, y) if x != y && func.ty(cond_val).is_small_int() => {
                    let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
                    if ctx.cond_tree(cond_val, pos).is_none() {
                        push_branch_tst(func, plan, (&uses, &pos_of), (b_idx, pos, cond_val), &mut branch_tst);
                    }
                }
                Op::Binary(Binary::Band, x, y) if features.bmi1 && !plan.elided[res.index()] && ty.is_small_int() => {
                    for (cand, other) in [(x, y), (y, x)] {
                        if !plan.const_of(other).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))
                            && uses[cand.index()] == 1
                            && !plan.elided[cand.index()]
                            && let Some((cb, cpos, cinst)) = pos_of[cand.index()]
                            && cb == b_idx
                            && cpos < pos
                            && let Op::Unary(Unary::Bnot, inner) = func.insts[cinst.index()].op
                            && preserved(func, plan, b_idx, cpos, pos, &[], inner)
                        {
                            plan.elided[cand.index()] = true;
                            break;
                        }
                    }
                    if matches!(ty, Type::I32 | Type::I64) {
                        cand_bextr.push((b_idx, pos, res, Binary::Band, x, y));
                        cand_alu.push((b_idx, pos, res, Binary::Band, x, y));
                    }
                }
                Op::Binary(op @ (Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor), x, y) if !plan.elided[res.index()] => {
                    if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd) && matches!(ty, Type::I32 | Type::I64) {
                        cand_shld.push((b_idx, pos, res, x, y));
                    }
                    if matches!(op, Binary::Iadd | Binary::Isub) && matches!(ty, Type::I32 | Type::I64) {
                        cand_adc.push((b_idx, pos, res, op, x, y));
                    }
                    if matches!(op, Binary::Iadd | Binary::Isub) && plan.sib[res.index()].is_none() && ty == Type::I64 {
                        cand_sibs.push((b_idx, pos, res));
                    }
                    if matches!(ty, Type::I32 | Type::I64) {
                        cand_alu.push((b_idx, pos, res, op, x, y));
                    }
                }
                Op::Binary(op @ (Binary::Band | Binary::Ushr | Binary::Ishl | Binary::Sshr), x, y) if !plan.elided[res.index()] && matches!(ty, Type::I32 | Type::I64) => {
                    if features.bmi1 && matches!(op, Binary::Band | Binary::Ushr) {
                        cand_bextr.push((b_idx, pos, res, op, x, y));
                    }
                    let nz_shift = matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr) && plan.const_of(y).is_some_and(|c| (c & i64::from(ty.bits() - 1)) != 0);
                    if op == Binary::Band || nz_shift || (op == Binary::Ushr && features.bmi1) {
                        cand_alu.push((b_idx, pos, res, op, x, y));
                    }
                }
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) if !plan.elided[res.index()] => {
                    cmp_tst.push((b_idx, pos, res, lhs, rhs));
                }
                Op::Load(..) | Op::StackLoad(..) if uses[res.index()] == 1 && !plan.elided[res.index()] && (ty.is_small_int() || (features.avx && (ty.is_wide_vec() || (ty.is_valid_vec() && ty.bytes() == 16)))) => {
                    cand_loads.push((b_idx, pos, res));
                }
                _ => {}
            }
        }
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let (Op::Select(cond, ..) | Op::Brif(cond, ..) | Op::Trapz(cond)) = func.insts[inst.index()].op else { continue };
            let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
            let Some(tree) = ctx.cond_tree(cond, pos) else { continue };
            let [leaf] = tree.leaves[..] else { continue };
            let Some((_, leaf_pos, _)) = pos_of[leaf.index()] else { continue };
            let skip: Vec<usize> = tree.nodes.iter().map(|node| node.1).collect();
            if !preserved(func, plan, b_idx, leaf_pos, pos, &skip, leaf) {
                continue;
            }
            for &(v, _, is_leaf, _) in &tree.nodes {
                plan.elided[v.index()] = !is_leaf;
            }
            roots[cond.index()] = Some((leaf, None, tree.table));
            last_flag_use[leaf.index()] = Some((b_idx, pos));
            non_flag_use[leaf.index()] = false;
        }
    }
    let mut tst_end = vec![None; n];
    for (bpos, cond_val, m) in branch_tst {
        if roots[cond_val.index()].is_some() {
            continue;
        }
        plan.elided[cond_val.index()] = true;
        tst_end[cond_val.index()] = Some(bpos);
        if let Some(m) = m {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(bpos);
        }
    }
    for (b_idx, cpos, res, lhs, rhs) in cmp_tst {
        if !plan.elided[res.index()]
            && let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, &uses, &pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && band_ok(func, plan, b_idx, (mpos, cpos, &[]), (x, y))
        {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(cpos);
        }
    }
    let mut skip_flags = vec![false; n];
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    for (b_idx, pos, res, x, y) in cand_shld {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() {
            continue;
        }
        let Some((hi, lo, _, _, _)) = shld_parts(func, plan, func.ty(res), (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let (Some((bx, px, _)), Some((by, py, _))) = (pos_of[x.index()], pos_of[y.index()]) else { continue };
        if bx != b_idx || by != b_idx || px >= pos || py >= pos {
            continue;
        }
        let skip = [px, py];
        let min_p = px.min(py);
        let ok = if hi == lo {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi)
        } else {
            preserved(func, plan, b_idx, min_p, pos, &skip, hi) && preserved(func, plan, b_idx, min_p, pos, &skip, lo)
        };
        if ok {
            plan.elided[x.index()] = true;
            plan.elided[y.index()] = true;
            skip_flags[res.index()] = true;
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_adc {
        if plan.elided[res.index()] || plan.sib[res.index()].is_some() || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((cand, inner, cc, a, b)) = adc_parts(func, plan, func.ty(res), op, (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let Some((cb, cpos, _)) = pos_of[cand.index()] else { continue };
        if cb != b_idx || cpos >= pos {
            continue;
        }
        let (cmp_pos, skip): (usize, &[usize]) = match inner {
            Some(iv) => match pos_of[iv.index()] {
                Some((ib, ipos, _)) if ib == b_idx && ipos < cpos => (ipos, &[cpos]),
                _ => continue,
            },
            None => (cpos, &[]),
        };
        if preserved(func, plan, b_idx, cmp_pos, pos, skip, a) && preserved(func, plan, b_idx, cmp_pos, pos, skip, b) {
            plan.elided[cand.index()] = true;
            if let Some(iv) = inner {
                plan.elided[iv.index()] = true;
            }
            tst_end[inner.unwrap_or(cand).index()] = Some(pos);
            skip_flags[res.index()] = true;
            let other = if cand == y { x } else { y };
            let quiet = [cmp_pos, cpos];
            if uses[other.index()] == 1
                && !plan.elided[other.index()]
                && let Some((invert, ..)) = adc_sides(func, plan, cc, a, b)
                && let Some((p, q)) = adc_fold(func, op, invert, other, def_inst)
                && let Some((ob, opos, _)) = pos_of[other.index()]
                && ob == b_idx
                && opos < pos
                && [p, q].into_iter().all(|v| !plan.elided[v.index()] && preserved(func, plan, b_idx, opos, pos, &quiet, v))
            {
                plan.elided[other.index()] = true;
            }
        }
    }
    for (b_idx, pos, res, op, x, y) in cand_bextr {
        if plan.elided[res.index()] || plan.elided[x.index()] || plan.elided[y.index()] {
            continue;
        }
        let Some((Some(mid), src, _, _)) = bextr_parts(func, plan, func.ty(res), op, (x, y), |v| uses[v.index()] == 1 && !plan.elided[v.index()], def_inst) else { continue };
        let Some((mb, mpos, _)) = pos_of[mid.index()] else { continue };
        if mb == b_idx && mpos < pos && preserved(func, plan, b_idx, mpos, pos, &[], src) {
            plan.elided[mid.index()] = true;
        }
    }
    for &(b_idx, pos, addr) in cand_sibs.iter().rev() {
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
        let Some((m, nodes)) = find_cand_sib(&ctx, (pos, addr), &ls_uses, &const_kids) else { continue };
        plan.sib[addr.index()] = Some(m.sib);
        plan.elided[addr.index()] = true;
        for (kid, _, delta) in nodes {
            plan.sib[kid.index()] = Some(Sib { disp: (i64::from(m.sib.disp) + delta) as i32, ..m.sib });
            plan.elided[kid.index()] = true;
        }
        for (v, _) in m.elide {
            plan.elided[v.index()] = true;
        }
    }
    apply_sib_groups(func, plan, (&uses, &pos_of, &defs), &ls_uses);
    for &(b_idx, pos, res) in cand_sibs.iter().rev() {
        if plan.elided[res.index()] || uses[res.index()] == 0 || skip_flags[res.index()] {
            continue;
        }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
        let Some(m) = ctx.sib_matches(res, pos, false).find(|m| (!m.elide.is_empty() || matches!(m.sib.base, SibBase::Frame(_))) && sib_ok(&ctx, m, &[], pos)).or_else(|| double(&ctx, res, pos).filter(|m| sib_ok(&ctx, m, &[], pos))) else { continue };
        plan.sib[res.index()] = Some(m.sib);
        for (v, _) in m.elide {
            plan.elided[v.index()] = true;
        }
    }
    apply_select_trees(func, plan, (&uses, &pos_of, &defs), &mut tst_end, &mut roots);
    super::sib::fuse_mem_rmw(func, plan, (&uses, &pos_of, &defs));
    for (b_idx, lpos, load_res) in cand_loads {
        if plan.elided[load_res.index()] { continue; }
        let Some((ub, upos, uinst)) = use_of[load_res.index()] else { continue };
        if ub != b_idx || upos <= lpos { continue; }
        let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
        let udata = &func.insts[uinst.index()];
        let eff_pos = if let Some(ur) = udata.result.filter(|r| plan.elided[r.index()]) {
            let Some(fpos) = tst_end[ur.index()] else { continue };
            fpos
        } else {
            if !load_elide_ok(&ctx, features, func.blocks[b_idx].insts[lpos], load_res, udata) { continue; }
            upos
        };
        if load_fuses(&ctx, lpos, eff_pos, &[]) { plan.elided[load_res.index()] = true; }
    }
    for (b_idx, pos, res, op, x, y) in cand_alu {
        if plan.elided[res.index()] || plan.deferred[res.index()] || plan.fused[res.index()] || plan.sib[res.index()].is_some() || skip_flags[res.index()] {
            continue;
        }
        if op == Binary::Band && ((uses[res.index()] > 0 && !non_flag_use[res.index()]) || [x, y].into_iter().any(|v| plan.const_of(v).is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF)))) {
            continue;
        }
        if op == Binary::Ushr && !plan.const_of(y).is_some_and(|c| (c & i64::from(func.ty(res).bits() - 1)) != 0) && bextr_parts(func, plan, func.ty(res), op, (x, y), |v| plan.elided[v.index()], def_inst).is_none() {
            continue;
        }
        let shrink_safe = matches!(plan.location(res), Location::Reg(r) if !SAVED.contains(&r) && !FRAME_REGS.contains(&r))
            || (func.entry != Some(Block(b_idx as u32))
                && !matches!(func.blocks[b_idx].insts.last().map(|i| &func.insts[i.index()].op), Some(Op::Return(_))));
        if !shrink_safe {
            continue;
        }
        let insts = &func.blocks[b_idx].insts;
        let mut zero = res;
        for &next_inst in &insts[pos + 1..] {
            let nd = &func.insts[next_inst.index()];
            if nd.result.is_some_and(|r| plan.elided[r.index()]) {
                continue;
            }
            let flagless = nd.result.is_some_and(|r| !func.ty(r).is_vector())
                && matches!(nd.op, Op::Const(_) | Op::StackAddr(..) | Op::FramePointer | Op::Symbol(_) | Op::SlotAddr(_));
            if flagless {
                continue;
            }
            if let (&Op::Convert(Convert::Ireduce, src), Some(narrow)) = (&nd.op, nd.result)
                && src == zero
                && narrows_exactly(func, plan, &|v| defs[v.index()], res, func.ty(narrow))
            {
                zero = narrow;
                continue;
            }
            let zf_match = match nd.op {
                Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..) => c == zero,
                Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                    (lhs == zero && is_zero_const(func, plan, rhs)) || (rhs == zero && is_zero_const(func, plan, lhs))
                }
                _ => false,
            };
            if zf_match {
                plan.fused[res.index()] = true;
            }
            break;
        }
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (cpos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(cmp_res) = data.result else { continue };
            if plan.elided[cmp_res.index()] || plan.fused[cmp_res.index()] || uses[cmp_res.index()] == 0 || non_flag_use[cmp_res.index()] {
                continue;
            }
            let is_band = matches!(data.op, Op::Binary(Binary::Band, a, _) if func.ty(a).is_small_int());
            if !flag_cmp(func, &data.op) && !is_band { continue; }
            let Some((ub, last_pos)) = last_flag_use[cmp_res.index()] else { continue };
            if ub != b_idx || last_pos <= cpos { continue; }
            if (cpos + 1..=last_pos).all(|pos| mid_preserves_flags(func, plan, &defs, &roots, cmp_res, block.insts[pos], pos == last_pos)) {
                plan.fused[cmp_res.index()] = true;
                pure_test[cmp_res.index()] = is_band;
            } else if let Op::Icmp(_, a, b) = data.op {
                let ctx = FuseCtx { func, plan, uses: &uses, pos_of: &pos_of, defs: &defs, b_idx };
                if !parked(plan, cmp_res) && [a, b].into_iter().all(|v| cmp_operand_ok(&ctx, &tst_end, cpos, last_pos, v)) {
                    plan.elided[cmp_res.index()] = true;
                    for v in [a, b] { if tst_end[v.index()].is_some() { tst_end[v.index()] = Some(last_pos); } }
                }
            }
        }
    }
    super::sib::fuse_bt_shifts(func, plan, (&uses, &pos_of), &tst_end, &pure_test);
    fold_products(func, plan, &uses, &defs);
    if features.avx2 { super::elide_broadcast_inserts(func, plan, &defs, (&uses, &pos_of, &use_of)); }
    super::fuse_atomic_rmws(func, plan, (&uses, &pos_of, &use_of));
    elide_zero_bases(func, plan, &defs);
    for (low, operands) in paired {
        if plan.is_static(low) || plan.deferred[low.index()] {
            for value in operands.into_iter().filter(|&value| is_load(func, &defs, value)) {
                plan.elided[value.index()] = false;
            }
        }
    }
    (roots, pure_test)
}

/// Elides each zero vector read only as the base of integer lane-0 inserts, which write their scalar into a zeroed register without reading the base.
fn elide_zero_bases(func: &Function, plan: &mut Plan<Reg>, defs: &[Option<Inst>]) {
    let mut read = vec![false; func.types.len()];
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &func.insts[inst.index()].op;
        let base = match *op {
            Op::Insert(base, scalar, 0) if base != scalar && func.ty(base).lane().is_some_and(Lane::is_int) => Some(base),
            _ => None,
        };
        op.for_operands(|v| read[v.index()] |= Some(v) != base);
    }
    for (v, def) in defs.iter().enumerate() {
        if !read[v] && func.types[v].is_vector() && def.is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(0))) {
            plan.elided[v] = true;
        }
    }
}

/// Elides the high and low halves of each 64-bit product whose only use is the add, or, or xor right after them, which then multiplies in place.
fn fold_products(func: &Function, plan: &mut Plan<Reg>, uses: &[u32], defs: &[Option<Inst>]) {
    for block in &func.blocks {
        for pos in 2..block.insts.len() {
            let data = &func.insts[block.insts[pos].index()];
            let (Op::Binary(Binary::Bxor | Binary::Bor | Binary::Iadd, x, y), Some(result)) = (&data.op, data.result) else { continue };
            let halves = [block.insts[pos - 2], block.insts[pos - 1]].map(|inst| func.insts[inst.index()].result);
            if low_of_pair(func, block, pos - 1).is_none()
                || !halves.iter().all(|half| half.is_some_and(|half| [*x, *y].contains(&half) && uses[half.index()] == 1))
                || plan.elided[result.index()]
                || plan.sib[result.index()].is_some()
                || super::patterns::product_halves(func, &|v| defs[v.index()], (*x, *y)).is_none()
            {
                continue;
            }
            for half in halves.into_iter().flatten() {
                plan.elided[half.index()] = true;
            }
        }
    }
}

/// Whether an `adc` or `sbb` may read `load_res` from memory as its added operand: the address survives the writes to rax, rcx, rdx and the result register that come first.
fn carry_load_ok(ctx: &FuseCtx<'_>, linst: Inst, load_res: Value, (args, result): (&[Value], Option<Value>)) -> bool {
    let &[carry, a, b] = args else { return false };
    if b != load_res || a == b || carry == b || ctx.func.ty(load_res) != Type::I64 {
        return false;
    }
    match ctx.func.insts[linst.index()].op {
        Op::Load(addr, _) => {
            let result_reg = result.and_then(|r| ctx.plan.reg_of(r));
            ![Reg::Rax, Reg::Rcx, Reg::Rdx].into_iter().chain(result_reg).any(|reg| addr_reads(ctx.func, ctx.plan, ctx.defs, addr, reg))
        }
        Op::StackLoad(..) => true,
        _ => false,
    }
}

fn is_load(func: &Function, defs: &[Option<Inst>], value: Value) -> bool {
    defs[value.index()].is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Load(..) | Op::StackLoad(..)))
}

/// The low product and operands of the `Imul` at `pos` when it follows the 64-bit `Umulhi` of the same operands, which leaves that product in rax so the `Imul` reads nothing.
fn low_of_pair(func: &Function, block: &crate::ir::BlockData, pos: usize) -> Option<(Value, [Value; 2])> {
    let data = &func.insts[block.insts[pos].index()];
    let (Op::Binary(Binary::Imul, x, y), Some(low)) = (&data.op, data.result) else { return None };
    let before = &func.insts[block.insts[pos.checked_sub(1)?].index()];
    let (Op::Binary(Binary::Umulhi | Binary::Smulhi, a, b), Some(high)) = (&before.op, before.result) else { return None };
    ((a, b) == (x, y) || (a, b) == (y, x)).then_some(()).filter(|()| func.ty(low) == Type::I64 && func.ty(high) == Type::I64)?;
    Some((low, [*x, *y]))
}
