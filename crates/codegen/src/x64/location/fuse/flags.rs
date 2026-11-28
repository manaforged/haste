use super::super::Site;
use super::super::patterns::{
    Root, bextr_parts, flag_cmp, is_zero_const, mid_preserves_flags, parked,
};
use super::loads::{load_fuses, reads_reg};
use super::{FuseCtx, narrows_exactly, preserved};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};
use crate::x64::FRAME_REGS;
use crate::x64::asm::Reg;
use crate::x64::frame::SAVED;

fn band_ok(
    func: &Function,
    plan: &Plan<Reg>,
    b_idx: usize,
    (mpos, end, skip): (usize, usize, &[usize]),
    (x, y): (Value, Value),
) -> bool {
    func.ty(x).is_small_int()
        && preserved(func, plan, b_idx, mpos, end, skip, x)
        && preserved(func, plan, b_idx, mpos, end, skip, y)
}

fn zero_cmp_band(
    func: &Function,
    plan: &Plan<Reg>,
    uses: &[u32],
    pos_of: &[Site],
    b_idx: usize,
    lhs: Value,
    rhs: Value,
) -> Option<(Value, usize, Value, Value)> {
    crate::emit::zero_cmp_band(func, plan, (uses, pos_of), b_idx, (lhs, rhs), |v| {
        is_zero_const(func, plan, v)
    })
}

fn cmp_band_arg_ok(
    ctx: &FuseCtx<'_>,
    (mpos, cpos, last_pos): (usize, usize, usize),
    u: Value,
) -> bool {
    if !ctx.plan.elided[u.index()] {
        return preserved(ctx.func, ctx.plan, ctx.b_idx, mpos, last_pos, &[cpos], u);
    }
    ctx.pos_of[u.index()].is_some_and(|(lb, lpos, linst)| {
        lb == ctx.b_idx
            && lpos < mpos
            && matches!(
                ctx.func.insts[linst.index()].op,
                Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..)
            )
            && load_fuses(ctx, lpos, last_pos, &[cpos])
    })
}

fn cmp_operand_ok(
    ctx: &FuseCtx<'_>,
    tst_end: &[Option<usize>],
    cpos: usize,
    last_pos: usize,
    v: Value,
) -> bool {
    if !ctx.plan.elided[v.index()] {
        return preserved(ctx.func, ctx.plan, ctx.b_idx, cpos, last_pos, &[], v);
    }
    let Some((lb, lpos, linst)) = ctx.pos_of[v.index()] else {
        return false;
    };
    if lb != ctx.b_idx || lpos >= cpos {
        return false;
    }
    match ctx.func.insts[linst.index()].op {
        Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..) => {
            load_fuses(ctx, lpos, last_pos, &[])
        }
        Op::Binary(Binary::Band, x, y) if tst_end[v.index()].is_some() => [x, y]
            .into_iter()
            .all(|u| cmp_band_arg_ok(ctx, (lpos, cpos, last_pos), u)),
        _ => false,
    }
}

pub(super) fn push_branch_tst(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of): (&[u32], &[Site]),
    (b_idx, pos, cond_val): (usize, usize, Value),
    out: &mut Vec<(usize, Value, Option<Value>)>,
) {
    if uses[cond_val.index()] != 1 || plan.elided[cond_val.index()] {
        return;
    }
    let Some((cb, cpos, cinst)) = pos_of[cond_val.index()] else {
        return;
    };
    if cb != b_idx || cpos >= pos {
        return;
    }
    match func.insts[cinst.index()].op {
        Op::Binary(Binary::Band, x, y) if band_ok(func, plan, b_idx, (cpos, pos, &[]), (x, y)) => {
            out.push((pos, cond_val, None))
        }
        Op::Convert(Convert::Ireduce, m)
            if uses[m.index()] == 1
                && !plan.elided[m.index()]
                && let Some((mb, mpos, minst)) = pos_of[m.index()]
                && mb == b_idx
                && mpos < cpos
                && let Op::Binary(Binary::Band, x, y) = func.insts[minst.index()].op
                && narrows_exactly(
                    func,
                    plan,
                    &|v| pos_of[v.index()].map(|(_, _, i)| i),
                    m,
                    func.ty(cond_val),
                )
                && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y)) =>
        {
            out.push((pos, cond_val, Some(m)))
        }
        Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs)
            if let Some((m, mpos, x, y)) =
                zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)
                && mpos < cpos
                && band_ok(func, plan, b_idx, (mpos, pos, &[cpos]), (x, y)) =>
        {
            out.push((pos, cond_val, Some(m)))
        }
        _ => {}
    }
}

fn select_arg_ok(
    ctx: &FuseCtx<'_>,
    (start, pos, skip): (usize, usize, &[usize]),
    dst: Option<Reg>,
    v: Value,
) -> bool {
    let folds = ctx.uses[v.index()] == 1
        && !ctx.plan.elided[v.index()]
        && ctx.func.ty(v).is_small_int()
        && ctx.pos_of[v.index()].is_some_and(|(vb, lpos, linst)| {
            vb == ctx.b_idx
                && matches!(
                    ctx.func.insts[linst.index()].op,
                    Op::Load(..) | Op::StackLoad(..)
                )
                && load_fuses(ctx, lpos, pos, skip)
        });
    (folds || preserved(ctx.func, ctx.plan, ctx.b_idx, start, pos, skip, v))
        && dst.is_none_or(|d| !reads_reg(ctx, v, d))
}

fn elided_band_args(
    ctx: &FuseCtx<'_>,
    tst_end: &[Option<usize>],
    cc: IntCC,
    v: Value,
) -> Option<(usize, [Value; 2])> {
    if !matches!(cc, IntCC::Equal | IntCC::NotEqual)
        || !ctx.plan.elided[v.index()]
        || tst_end[v.index()].is_none()
    {
        return None;
    }
    let (vb, mpos, minst) = ctx.pos_of[v.index()]?;
    let Op::Binary(Binary::Band, bx, by) = ctx.func.insts[minst.index()].op else {
        return None;
    };
    (vb == ctx.b_idx).then_some((mpos, [bx, by]))
}

fn select_leaf_ok(
    ctx: &FuseCtx<'_>,
    tst_end: &[Option<usize>],
    (pos, skip): (usize, &[usize]),
    dst: Option<Reg>,
    leaf: Value,
) -> Option<Option<Value>> {
    let (_, leaf_pos, leaf_inst) = ctx.pos_of[leaf.index()]?;
    let Op::Icmp(cc, a, b) = ctx.func.insts[leaf_inst.index()].op else {
        return None;
    };
    let mut band = None;
    for v in [a, b] {
        if let Some((mpos, args)) = elided_band_args(ctx, tst_end, cc, v) {
            if !args
                .into_iter()
                .all(|u| select_arg_ok(ctx, (mpos, pos, skip), dst, u))
            {
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
    (uses, pos_of, defs): (&[u32], &[Site], &[Option<Inst>]),
    (b_idx, pos, inst): (usize, usize, Inst),
    (tst_end, roots): (&mut [Option<usize>], &mut [Option<Root>]),
) {
    let data = &func.insts[inst.index()];
    let (Op::Select(cond, x, y), Some(res)) = (&data.op, data.result) else {
        return;
    };
    let (first, second, table, nodes, other, dst) = {
        let ctx = FuseCtx {
            func,
            plan,
            uses,
            pos_of,
            defs,
            b_idx,
        };
        let Some(tree) = ctx.cond_tree(*cond, pos) else {
            return;
        };
        let ([first, second], Location::Reg(dst)) = (&tree.leaves[..], plan.location(res)) else {
            return;
        };
        let other = if tree.table.count_ones() == 1 { *y } else { *x };
        (*first, *second, tree.table, tree.nodes, other, dst)
    };
    if !func.ty(*x).is_small_int() || x == y || plan.reg_of(other) == Some(dst) {
        return;
    }
    let skip: Vec<usize> = nodes.iter().map(|node| node.1).collect();
    let bands = {
        let ctx = FuseCtx {
            func,
            plan,
            uses,
            pos_of,
            defs,
            b_idx,
        };
        let Some(b1) = select_leaf_ok(&ctx, tst_end, (pos, &skip), None, first) else {
            return;
        };
        let Some(b2) = select_leaf_ok(&ctx, tst_end, (pos, &skip), Some(dst), second) else {
            return;
        };
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

pub(super) fn apply_select_trees(
    func: &Function,
    plan: &mut Plan<Reg>,
    tables: (&[u32], &[Site], &[Option<Inst>]),
    tst_end: &mut [Option<usize>],
    roots: &mut [Option<Root>],
) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            apply_select_at(func, plan, tables, (b_idx, pos, inst), (tst_end, roots));
        }
    }
}

pub(super) fn apply_single_leaf_trees(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[Site], &[Option<Inst>]),
    roots: &mut [Option<Root>],
    (last_flag_use, non_flag_use): (&mut [Option<(usize, usize)>], &mut [bool]),
) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            let (Op::Select(cond, ..) | Op::Brif(cond, ..) | Op::Trapz(cond)) =
                func.insts[inst.index()].op
            else {
                continue;
            };
            let ctx = FuseCtx {
                func,
                plan,
                uses,
                pos_of,
                defs,
                b_idx,
            };
            let Some(tree) = ctx.cond_tree(cond, pos) else {
                continue;
            };
            let [leaf] = tree.leaves[..] else {
                continue;
            };
            let Some((_, leaf_pos, _)) = pos_of[leaf.index()] else {
                continue;
            };
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
}

pub(super) fn apply_tst_fusions(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of): (&[u32], &[Site]),
    roots: &[Option<Root>],
    branch_tst: Vec<(usize, Value, Option<Value>)>,
    cmp_tst: Vec<(usize, usize, Value, Value, Value)>,
) -> Vec<Option<usize>> {
    let mut tst_end = vec![None; func.types.len()];
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
            && let Some((m, mpos, x, y)) = zero_cmp_band(func, plan, uses, pos_of, b_idx, lhs, rhs)
            && mpos < cpos
            && band_ok(func, plan, b_idx, (mpos, cpos, &[]), (x, y))
        {
            plan.elided[m.index()] = true;
            tst_end[m.index()] = Some(cpos);
        }
    }
    tst_end
}

fn alu_zf_eligible(
    func: &Function,
    plan: &Plan<Reg>,
    (uses, pos_of, non_flag_use, skip_flags): (&[u32], &[Site], &[bool], &[bool]),
    (b_idx, res, op, x, y): (usize, Value, Binary, Value, Value),
) -> bool {
    if plan.elided[res.index()]
        || plan.deferred[res.index()]
        || plan.fused[res.index()]
        || plan.sib[res.index()].is_some()
        || skip_flags[res.index()]
    {
        return false;
    }
    if op == Binary::Band
        && ((uses[res.index()] > 0 && !non_flag_use[res.index()])
            || [x, y].into_iter().any(|v| {
                plan.const_of(v)
                    .is_some_and(|m| matches!(m, 0xFF | 0xFFFF | 0xFFFF_FFFF))
            }))
    {
        return false;
    }
    let def_inst = |v: Value| pos_of[v.index()].map(|(_, _, inst)| inst);
    if op == Binary::Ushr
        && !plan
            .const_of(y)
            .is_some_and(|c| (c & i64::from(func.ty(res).bits() - 1)) != 0)
        && bextr_parts(
            func,
            plan,
            func.ty(res),
            op,
            (x, y),
            |v| plan.elided[v.index()],
            def_inst,
        )
        .is_none()
    {
        return false;
    }
    matches!(plan.location(res), Location::Reg(r) if !SAVED.contains(&r) && !FRAME_REGS.contains(&r))
        || (func.entry != Some(Block(b_idx as u32))
            && !matches!(
                func.blocks[b_idx]
                    .insts
                    .last()
                    .map(|i| &func.insts[i.index()].op),
                Some(Op::Return(_))
            ))
}

fn alu_zf_matches(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    (b_idx, pos, res): (usize, usize, Value),
) -> bool {
    let insts = &func.blocks[b_idx].insts;
    let mut zero = res;
    for &next_inst in &insts[pos + 1..] {
        let nd = &func.insts[next_inst.index()];
        if nd.result.is_some_and(|r| plan.elided[r.index()]) {
            continue;
        }
        let flagless = nd.result.is_some_and(|r| !func.ty(r).is_vector())
            && matches!(
                nd.op,
                Op::Const(_)
                    | Op::StackAddr(..)
                    | Op::FramePointer
                    | Op::Symbol(_)
                    | Op::SlotAddr(_)
            );
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
        return match nd.op {
            Op::Brif(c, ..) | Op::Trapz(c) | Op::Select(c, ..) => c == zero,
            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                (lhs == zero && is_zero_const(func, plan, rhs))
                    || (rhs == zero && is_zero_const(func, plan, lhs))
            }
            _ => false,
        };
    }
    false
}

pub(super) fn fuse_alu_zf(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs, non_flag_use, skip_flags): (
        &[u32],
        &[Site],
        &[Option<Inst>],
        &[bool],
        &[bool],
    ),
    cand_alu: Vec<(usize, usize, Value, Binary, Value, Value)>,
) {
    for (b_idx, pos, res, op, x, y) in cand_alu {
        if alu_zf_eligible(
            func,
            plan,
            (uses, pos_of, non_flag_use, skip_flags),
            (b_idx, res, op, x, y),
        ) && alu_zf_matches(func, plan, defs, (b_idx, pos, res))
        {
            plan.fused[res.index()] = true;
        }
    }
}

pub(super) fn fuse_flag_cmps(
    func: &Function,
    plan: &mut Plan<Reg>,
    (uses, pos_of, defs): (&[u32], &[Site], &[Option<Inst>]),
    (last_flag_use, non_flag_use): (&[Option<(usize, usize)>], &[bool]),
    roots: &[Option<Root>],
    (tst_end, pure_test): (&mut [Option<usize>], &mut [bool]),
) {
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (cpos, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(cmp_res) = data.result else {
                continue;
            };
            if plan.elided[cmp_res.index()]
                || plan.fused[cmp_res.index()]
                || uses[cmp_res.index()] == 0
                || non_flag_use[cmp_res.index()]
            {
                continue;
            }
            let is_band =
                matches!(data.op, Op::Binary(Binary::Band, a, _) if func.ty(a).is_small_int());
            if !flag_cmp(func, &data.op) && !is_band {
                continue;
            }
            let Some((ub, last_pos)) = last_flag_use[cmp_res.index()] else {
                continue;
            };
            if ub != b_idx || last_pos <= cpos {
                continue;
            }
            if (cpos + 1..=last_pos).all(|pos| {
                mid_preserves_flags(
                    func,
                    plan,
                    defs,
                    roots,
                    cmp_res,
                    block.insts[pos],
                    pos == last_pos,
                )
            }) {
                plan.fused[cmp_res.index()] = true;
                pure_test[cmp_res.index()] = is_band;
            } else if let Op::Icmp(_, a, b) = data.op {
                let ctx = FuseCtx {
                    func,
                    plan,
                    uses,
                    pos_of,
                    defs,
                    b_idx,
                };
                if !parked(plan, cmp_res)
                    && [a, b]
                        .into_iter()
                        .all(|v| cmp_operand_ok(&ctx, tst_end, cpos, last_pos, v))
                {
                    plan.elided[cmp_res.index()] = true;
                    for v in [a, b] {
                        if tst_end[v.index()].is_some() {
                            tst_end[v.index()] = Some(last_pos);
                        }
                    }
                }
            }
        }
    }
}

fn is_narrow_byte_def(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    inst: Inst,
    res: Value,
) -> bool {
    if func.ty(res) != Type::I8
        || plan.elided[res.index()]
        || plan.deferred[res.index()]
        || plan.is_static(res)
        || plan.reg_of(res).is_none()
    {
        return false;
    }
    match func.insts[inst.index()].op {
        Op::Icmp(_, x, _) => !plan.fused[res.index()] && !func.ty(x).is_vector(),
        Op::Fits(..) => !plan.fused[res.index()],
        Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, a, b) => {
            ![a, b].into_iter().any(|u| {
                plan.elided[u.index()]
                    && matches!(
                        defs[u.index()].map(|i| &func.insts[i.index()].op),
                        Some(Op::Unary(Unary::Bnot, _))
                    )
            })
        }
        _ => false,
    }
}

pub(super) fn find_narrow_bytes(
    func: &Function,
    plan: &mut Plan<Reg>,
    defs: &[Option<Inst>],
) -> Vec<bool> {
    let n = func.types.len();
    let mut cand = vec![false; n];
    let mut bitops: Vec<(Value, Value, Value)> = Vec::new();
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if !is_narrow_byte_def(func, plan, defs, inst, res) {
                continue;
            }
            cand[res.index()] = true;
            if let Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, a, b) = data.op {
                bitops.push((res, a, b));
            }
        }
    }
    let mut wide = vec![false; n];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            match &data.op {
                Op::Brif(_, _, ta, _, ea) => {
                    for &v in ta.iter().chain(ea) {
                        wide[v.index()] = true;
                    }
                }
                Op::Trapz(_) => {}
                Op::Select(_, t, f) => {
                    wide[t.index()] = true;
                    wide[f.index()] = true;
                }
                Op::Icmp(_, a, _) if func.ty(*a) == Type::I8 => {}
                Op::Store(_, addr, _) => wide[addr.index()] = true,
                Op::StackStore(..) => {}
                Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, _, _)
                    if data.result.is_some_and(|r| cand[r.index()]) => {}
                op => op.for_operands(|v| wide[v.index()] = true),
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &(res, a, b) in &bitops {
            if wide[res.index()] {
                for u in [a, b] {
                    if !wide[u.index()] {
                        wide[u.index()] = true;
                        changed = true;
                    }
                }
            }
        }
    }
    let mut narrow = vec![false; n];
    for v in 0..n {
        if cand[v] && !wide[v] {
            narrow[v] = true;
            plan.zero_ext[v] = false;
            plan.clean_bits[v] = 65;
        }
    }
    narrow
}
