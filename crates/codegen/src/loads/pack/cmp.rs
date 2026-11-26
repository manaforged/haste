use super::leaf::{checked_wide_load, contiguous_leaves, def_op, push_leaf_load, wide_mem_ty};
use super::{AddrKey, Leaf, PackCtx, PackOut, PackPlan};
use crate::ir::{Binary, Block, Convert, Inst, IntCC, Op, Type, Value};

#[derive(Default)]
struct CmpAcc {
    leaves_a: Vec<Leaf>,
    leaves_b: Vec<Leaf>,
    masks: Vec<(u64, u64)>,
    touched: Vec<Value>,
    key_a: Option<AddrKey>,
    key_b: Option<AddrKey>,
    two_ptr: Option<bool>,
}

fn const_pair(ctx: &PackCtx<'_>, a: Value, b: Value) -> Option<(Value, i64)> {
    let c = |v: Value| ctx.consts.get(v.index()).copied().flatten();
    c(b).map(|imm| (a, imm))
        .or_else(|| c(a).map(|imm| (b, imm)))
}

fn canon_const(raw: u64, ty: Type) -> i64 {
    if ty == Type::I16 {
        (raw as i16) as i64
    } else if ty == Type::I32 {
        (raw as i32) as i64
    } else {
        raw as i64
    }
}

fn peel_uext(ctx: &PackCtx<'_>, mut v: Value, touched: &mut Vec<Value>) -> Value {
    while let Some(&Op::Convert(Convert::Uextend, inner)) = def_op(ctx, v) {
        touched.push(v);
        v = inner;
    }
    v
}

fn peel_masked_load(
    ctx: &PackCtx<'_>,
    mut v: Value,
    mut mask: u64,
    touched: &mut Vec<Value>,
) -> Option<(Value, u64)> {
    loop {
        v = peel_uext(ctx, v, touched);
        let Some(&Op::Binary(Binary::Band, a, b)) = def_op(ctx, v) else {
            break;
        };
        let (inner, imm) = const_pair(ctx, a, b)?;
        touched.push(v);
        mask &= imm as u64;
        v = inner;
    }
    let ty = ctx.func.ty(v);
    (ty.is_small_int() && ty.bytes() <= 4).then_some((v, mask & (!0_u64 >> (64 - ty.bytes() * 8))))
}

fn push_cmp_masked(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    val: Value,
    (init_m, c_bits): (u64, u64),
    acc: &mut CmpAcc,
) -> Option<()> {
    if acc.two_ptr == Some(true) {
        return None;
    }
    acc.two_ptr = Some(false);
    let (load_v, mask) = peel_masked_load(ctx, val, init_m, &mut acc.touched)?;
    if mask == 0 || (c_bits & !mask) != 0 {
        return None;
    }
    let idx = acc.masks.len() as i64;
    push_leaf_load(
        ctx,
        chain,
        load_v,
        (idx, ctx.func.ty(load_v).bytes()),
        (&mut acc.leaves_a, &mut acc.touched, &mut acc.key_a),
    )?;
    acc.masks.push((mask, c_bits));
    Some(())
}

fn push_cmp_eq(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (a, b): (Value, Value),
    acc: &mut CmpAcc,
) -> Option<()> {
    let cmp_ty = ctx.func.ty(a);
    if !cmp_ty.is_small_int() {
        return None;
    }
    let cmp_mask = !0_u64 >> (64 - cmp_ty.bytes() * 8);
    if let Some((other, imm)) = const_pair(ctx, a, b) {
        return push_cmp_masked(ctx, chain, other, (cmp_mask, (imm as u64) & cmp_mask), acc);
    }
    if acc.two_ptr == Some(false) {
        return None;
    }
    acc.two_ptr = Some(true);
    let (la, lb) = (
        peel_uext(ctx, a, &mut acc.touched),
        peel_uext(ctx, b, &mut acc.touched),
    );
    let ty = ctx.func.ty(la);
    if ty != ctx.func.ty(lb) || !ty.is_small_int() || ty.bytes() > 4 {
        return None;
    }
    let idx = acc.leaves_a.len() as i64;
    push_leaf_load(
        ctx,
        chain,
        la,
        (idx, ty.bytes()),
        (&mut acc.leaves_a, &mut acc.touched, &mut acc.key_a),
    )?;
    push_leaf_load(
        ctx,
        chain,
        lb,
        (idx, ty.bytes()),
        (&mut acc.leaves_b, &mut acc.touched, &mut acc.key_b),
    )
}

fn parse_bor_zero_leaf(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    val: Value,
    acc: &mut CmpAcc,
) -> Option<()> {
    let cur = peel_uext(ctx, val, &mut acc.touched);
    match *def_op(ctx, cur)? {
        Op::Binary(Binary::Bxor, a, b) => {
            acc.touched.push(cur);
            push_cmp_eq(ctx, chain, (a, b), acc)
        }
        Op::Binary(Binary::Band, x, y) => {
            let (inner, m) = const_pair(ctx, x, y)?;
            let inner = peel_uext(ctx, inner, &mut acc.touched);
            if let Some(&Op::Binary(Binary::Bxor, a, b)) = def_op(ctx, inner) {
                let (load_side, c_imm) = const_pair(ctx, a, b)?;
                acc.touched.extend([cur, inner]);
                return push_cmp_masked(ctx, chain, load_side, (m as u64, (c_imm & m) as u64), acc);
            }
            push_cmp_masked(ctx, chain, cur, (!0_u64, 0), acc)
        }
        _ => push_cmp_masked(ctx, chain, cur, (!0_u64, 0), acc),
    }
}

fn collect_cmp_leaves(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    val: Value,
    (join_op, leaf_cc): (Binary, Option<IntCC>),
    acc: &mut CmpAcc,
) -> Option<()> {
    if acc.leaves_a.len() >= 8 || acc.touched.len() >= 64 {
        return None;
    }
    let cur = peel_uext(ctx, val, &mut acc.touched);
    match *def_op(ctx, cur)? {
        Op::Binary(op, a, b) if op == join_op => {
            acc.touched.push(cur);
            collect_cmp_leaves(ctx, chain, a, (join_op, leaf_cc), acc)?;
            collect_cmp_leaves(ctx, chain, b, (join_op, leaf_cc), acc)
        }
        Op::Icmp(cc, a, b) if leaf_cc == Some(cc) => {
            acc.touched.push(cur);
            push_cmp_eq(ctx, chain, (a, b), acc)
        }
        _ if leaf_cc.is_none() => parse_bor_zero_leaf(ctx, chain, cur, acc),
        _ => None,
    }
}

fn finish_two_ptr_cmp(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (root_pos, cc, wide_ty, min_off): (usize, IntCC, Type, i64),
    (wide_op, acc): (Op, CmpAcc),
) -> Option<PackPlan> {
    if acc.leaves_b.len() != acc.leaves_a.len() {
        return None;
    }
    let pairs_b: Vec<Leaf> = acc
        .leaves_a
        .iter()
        .map(|la| acc.leaves_b[la.shift_byte as usize])
        .collect();
    let min_b = pairs_b[0].canon_off;
    if !acc
        .leaves_a
        .iter()
        .zip(&pairs_b)
        .all(|(la, lb)| la.bytes == lb.bytes && lb.canon_off - min_b == la.canon_off - min_off)
    {
        return None;
    }
    let rhs_op = checked_wide_load(ctx, chain, root_pos, &pairs_b, wide_ty)?;
    Some(PackPlan {
        wide_op,
        wide_ty,
        out: PackOut::Cmp {
            cc,
            mask: None,
            rhs_const: None,
            rhs_op: Some(rhs_op),
        },
        touched: acc.touched,
    })
}

fn finish_cmp_pack(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (root_pos, cc): (usize, IntCC),
    mut acc: CmpAcc,
) -> Option<PackPlan> {
    let min_off = contiguous_leaves(&mut acc.leaves_a, None)?;
    let total_w: i64 = acc.leaves_a.iter().map(|l| l.bytes).sum();
    if !matches!(total_w, 2 | 4 | 8) {
        return None;
    }
    let wide_ty = wide_mem_ty(total_w);
    let wide_op = checked_wide_load(ctx, chain, root_pos, &acc.leaves_a, wide_ty)?;
    if acc.two_ptr == Some(true) {
        return finish_two_ptr_cmp(ctx, chain, (root_pos, cc, wide_ty, min_off), (wide_op, acc));
    }
    let (mut wide_mask, mut wide_val) = (0_u64, 0_u64);
    for la in &acc.leaves_a {
        let (m, c) = acc.masks[la.shift_byte as usize];
        let sh = (la.canon_off - min_off) * 8;
        wide_mask |= m << sh;
        wide_val |= c << sh;
    }
    let all_ones = !0_u64 >> (64 - total_w * 8);
    let mask = (wide_mask != all_ones).then(|| canon_const(wide_mask, wide_ty));
    let out = PackOut::Cmp {
        cc,
        mask,
        rhs_const: Some(canon_const(wide_val, wide_ty)),
        rhs_op: None,
    };
    Some(PackPlan {
        wide_op,
        wide_ty,
        out,
        touched: acc.touched,
    })
}

pub(super) fn build_cmp_pack(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (root_pos, inst, res_ty): (usize, Inst, Type),
) -> Option<PackPlan> {
    if !res_ty.is_small_int() {
        return None;
    }
    let res = ctx.func.insts[inst.index()].result?;
    let (join, leaf_cc, out_cc, root) = match ctx.func.insts[inst.index()].op {
        Op::Binary(Binary::Band, ..) => (Binary::Band, Some(IntCC::Equal), IntCC::Equal, res),
        Op::Binary(Binary::Bor, ..) => (Binary::Bor, Some(IntCC::NotEqual), IntCC::NotEqual, res),
        Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), a, b) => {
            let (bor, imm) = const_pair(ctx, a, b)?;
            if imm != 0 || !matches!(def_op(ctx, bor), Some(&Op::Binary(Binary::Bor, ..))) {
                return None;
            }
            (Binary::Bor, None, cc, bor)
        }
        _ => return None,
    };
    let mut acc = CmpAcc::default();
    collect_cmp_leaves(ctx, chain, root, (join, leaf_cc), &mut acc)?;
    (acc.leaves_a.len() >= 2)
        .then(|| finish_cmp_pack(ctx, chain, (root_pos, out_cc), acc))
        .flatten()
}
