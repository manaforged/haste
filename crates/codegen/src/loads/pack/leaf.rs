use super::super::dom::clobbers_loc;
use super::super::fits_mem_imm;
use super::{AddrKey, DefInfo, Leaf, PackCtx, PackOut, PackPlan, RawLoad};
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, TargetOpId, Type, Unary, Value};
use crate::mem::{MemBase, MemLoc, Provenance};

const MAX_ADDR_STEPS: usize = 16;

fn canon_addr_step(
    func: &Function,
    consts: &[Option<i64>],
    def_info: &[DefInfo],
    (v, sc): (Value, i64),
    work: &mut Vec<(Value, i64)>,
) -> Result<bool, ()> {
    let Some((_, _, inst)) = def_info.get(v.index()).copied().flatten() else {
        return Ok(false);
    };
    match func.insts[inst.index()].op {
        Op::Convert(Convert::Bitcast, x) if func.ty(x) == Type::I64 => work.push((x, sc)),
        Op::Binary(Binary::Iadd, a, b) => work.extend([(a, sc), (b, sc)]),
        Op::Binary(Binary::Isub, a, b) => work.extend([(a, sc), (b, sc.checked_neg().ok_or(())?)]),
        Op::Binary(Binary::Ishl, a, amt) => {
            let Some(sh) = consts
                .get(amt.index())
                .copied()
                .flatten()
                .filter(|&s| (0..63).contains(&s))
            else {
                return Ok(false);
            };
            work.push((a, sc.checked_mul(1_i64 << sh).ok_or(())?));
        }
        Op::Binary(Binary::Imul, a, b) => {
            let Some((inner, k)) = consts
                .get(b.index())
                .copied()
                .flatten()
                .map(|k| (a, k))
                .or_else(|| consts.get(a.index()).copied().flatten().map(|k| (b, k)))
            else {
                return Ok(false);
            };
            work.push((inner, sc.checked_mul(k).ok_or(())?));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn merge_canon_terms(mut terms: Vec<(u32, i64)>) -> Option<AddrKey> {
    terms.sort_unstable_by_key(|&(id, _)| id);
    let mut merged: Vec<(u32, i64)> = Vec::with_capacity(terms.len());
    for (id, sc) in terms {
        if let Some(last) = merged.last_mut().filter(|(lid, _)| *lid == id) {
            last.1 = last.1.checked_add(sc)?;
            if last.1 == 0 {
                merged.pop();
            }
        } else if sc != 0 {
            merged.push((id, sc));
        }
    }
    Some(AddrKey::Expr(merged))
}

fn canonical_addr(
    func: &Function,
    prov: &Provenance,
    consts: &[Option<i64>],
    def_info: &[DefInfo],
    addr: Value,
    off: i32,
) -> Option<(AddrKey, i64)> {
    let loc = prov.loc(addr, i64::from(off), 1);
    match loc.base {
        MemBase::Slot(slot) => return Some((AddrKey::Slot(slot), loc.offset)),
        MemBase::Global(sym) => return Some((AddrKey::Global(sym), loc.offset)),
        MemBase::Addr(_) => {}
    }
    let (mut const_off, mut terms, mut work, mut steps) =
        (i64::from(off), Vec::new(), vec![(addr, 1_i64)], 0_usize);
    while let Some((v, sc)) = work.pop() {
        if let Some(imm) = consts.get(v.index()).copied().flatten() {
            const_off = const_off.checked_add(sc.checked_mul(imm)?)?;
            continue;
        }
        if steps < MAX_ADDR_STEPS
            && func.ty(v) == Type::I64
            && canon_addr_step(func, consts, def_info, (v, sc), &mut work).ok()?
        {
            steps += 1;
            continue;
        }
        terms.push((v.index() as u32, sc));
    }
    Some((merge_canon_terms(terms)?, const_off))
}

pub(super) fn push_leaf_load(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    val: Value,
    (shift_byte, expect_bytes): (i64, u32),
    (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
    let (lblk, lpos, linst) = ctx.def_info.get(val.index()).copied().flatten()?;
    let blk_idx = chain.iter().position(|&b| b == lblk)?;
    let ty = ctx.func.ty(val);
    if ty.bytes() != expect_bytes {
        return None;
    }
    let (loc, raw, addr_key, canon_off) = match ctx.func.insts[linst.index()].op {
        Op::Load(addr, off) => {
            let (ak, coff) =
                canonical_addr(ctx.func, ctx.prov, ctx.consts, ctx.def_info, addr, off)?;
            (
                ctx.prov.loc(addr, i64::from(off), ty.bytes()),
                RawLoad::Addr(addr, off),
                ak,
                coff,
            )
        }
        Op::StackLoad(slot, off) => {
            let loc = MemLoc::slot(slot, i64::from(off), ty.bytes());
            (loc, RawLoad::Stack, AddrKey::Slot(slot), loc.offset)
        }
        _ => return None,
    };
    match key {
        None => *key = Some(addr_key),
        Some(expected) if *expected != addr_key => return None,
        _ => {}
    }
    touched.push(val);
    leaves.push(Leaf {
        blk_idx,
        pos: lpos,
        loc,
        canon_off,
        raw,
        shift_byte,
        bytes: i64::from(expect_bytes),
        bswapped: false,
    });
    Some(())
}

pub(super) fn def_op<'a>(ctx: &'a PackCtx<'_>, v: Value) -> Option<&'a Op> {
    let (_, _, inst) = ctx.def_info.get(v.index()).copied().flatten()?;
    Some(&ctx.func.insts[inst.index()].op)
}

fn peel_shift_ext(
    ctx: &PackCtx<'_>,
    mut val: Value,
    (mut shift, mut avail): (i64, i64),
    touched: &mut Vec<Value>,
) -> Option<(Value, i64, i64, bool)> {
    while let Some(op) = def_op(ctx, val) {
        match *op {
            Op::Convert(Convert::Uextend, inner) => {
                touched.push(val);
                avail = avail.min(i64::from(ctx.func.ty(inner).bytes()));
                val = inner;
            }
            Op::Binary(Binary::Ishl, inner, amt) => {
                let imm = ctx.consts.get(amt.index()).copied().flatten()?;
                let cur_bytes = i64::from(ctx.func.ty(val).bytes());
                if imm < 0 || imm >= cur_bytes * 8 || imm % 8 != 0 || imm / 8 >= avail {
                    return None;
                }
                touched.push(val);
                shift += imm / 8;
                avail = avail.min(cur_bytes) - imm / 8;
                val = inner;
            }
            Op::Unary(Unary::Bswap, inner) => {
                touched.push(val);
                avail = avail.min(i64::from(ctx.func.ty(inner).bytes()));
                return Some((inner, shift, avail, true));
            }
            _ => break,
        }
    }
    Some((val, shift, avail, false))
}

fn collect_leaves(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (val, shift, avail): (Value, i64, i64),
    (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
    if leaves.len() >= 8 || touched.len() >= 64 {
        return None;
    }
    let (cur, shift, avail, bswapped) = peel_shift_ext(ctx, val, (shift, avail), touched)?;
    if !bswapped && let Some(&Op::Binary(Binary::Bor | Binary::Iadd, a, b)) = def_op(ctx, cur) {
        touched.push(cur);
        collect_leaves(ctx, chain, (a, shift, avail), (leaves, touched, key))?;
        return collect_leaves(ctx, chain, (b, shift, avail), (leaves, touched, key));
    }
    let load_ty = ctx.func.ty(cur);
    let min_bytes = if bswapped { 2 } else { 1 };
    if !load_ty.is_small_int()
        || !(min_bytes..=4).contains(&load_ty.bytes())
        || i64::from(load_ty.bytes()) > avail
    {
        return None;
    }
    push_leaf_load(
        ctx,
        chain,
        cur,
        (shift, load_ty.bytes()),
        (leaves, touched, key),
    )?;
    if bswapped {
        leaves.last_mut()?.bswapped = true;
    }
    Some(())
}

fn has_clobber(
    func: &Function,
    prov: &Provenance,
    chain: &[Block],
    leaves: &[Leaf],
    root_pos: usize,
    wide_loc: MemLoc,
) -> bool {
    let Some(min_leaf) = leaves.iter().min_by_key(|l| (l.blk_idx, l.pos)) else {
        return true;
    };
    let last_chain = chain.len() - 1;
    let check_inst = |inst: Inst| {
        let op = &func.insts[inst.index()].op;
        clobbers_loc(func, (op, None), wide_loc, prov)
            || leaves
                .iter()
                .any(|l| clobbers_loc(func, (op, None), l.loc, prov))
    };
    for ci in min_leaf.blk_idx..=last_chain {
        let insts = &func.blocks[chain[ci].index()].insts;
        let lo = if ci == min_leaf.blk_idx {
            min_leaf.pos + 1
        } else {
            0
        };
        let hi = if ci == last_chain {
            root_pos
        } else {
            insts.len()
        };
        if lo < hi && insts[lo..hi].iter().copied().any(check_inst) {
            return true;
        }
    }
    false
}

fn build_wide_load(
    ctx: &PackCtx<'_>,
    root_blk: Block,
    root_pos: usize,
    first: Leaf,
    wide_ty: Type,
) -> Option<Op> {
    match (first.loc.base, first.raw) {
        (MemBase::Slot(slot), _) => {
            Some(Op::StackLoad(slot, i32::try_from(first.loc.offset).ok()?))
        }
        (_, RawLoad::Addr(addr, off)) if fits_mem_imm(wide_ty, i64::from(off)) => {
            Some(Op::Load(addr, off))
        }
        (MemBase::Addr(base), _)
            if ctx
                .def_info
                .get(base.index())
                .copied()
                .flatten()
                .is_none_or(|(db, dp, _)| {
                    ctx.cfg.dominates(db, root_blk) && (db != root_blk || dp < root_pos)
                }) =>
        {
            let off_i32 = i32::try_from(first.loc.offset).ok()?;
            fits_mem_imm(wide_ty, first.loc.offset).then_some(Op::Load(base, off_i32))
        }
        _ => None,
    }
}

pub(super) fn contiguous_leaves(leaves: &mut [Leaf], expected_bytes: Option<i64>) -> Option<i64> {
    leaves.sort_by_key(|l| l.canon_off);
    let min_off = leaves.first()?.canon_off;
    let mut cursor = min_off;
    for l in leaves.iter() {
        if l.canon_off != cursor {
            return None;
        }
        cursor += l.bytes;
    }
    let total = cursor - min_off;
    expected_bytes
        .is_none_or(|want| want == total)
        .then_some(min_off)
}

pub(super) fn wide_mem_ty(total_w: i64) -> Type {
    if total_w == 2 {
        Type::I16
    } else if total_w == 4 {
        Type::I32
    } else {
        Type::I64
    }
}

pub(super) fn checked_wide_load(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    root_pos: usize,
    leaves: &[Leaf],
    wide_ty: Type,
) -> Option<Op> {
    let first = leaves[0];
    let wide_op = build_wide_load(ctx, *chain.last()?, root_pos, first, wide_ty)?;
    let wide_loc = MemLoc::new(
        first.loc.root,
        first.loc.base,
        first.loc.offset,
        wide_ty.bytes(),
    );
    (!has_clobber(ctx.func, ctx.prov, chain, leaves, root_pos, wide_loc)).then_some(wide_op)
}

fn finish_pack_plan(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (root_pos, wide_ty, big_endian): (usize, Type, bool),
    leaves: &[Leaf],
    touched: Vec<Value>,
) -> Option<PackPlan> {
    let wide_op = checked_wide_load(ctx, chain, root_pos, leaves, wide_ty)?;
    Some(PackPlan {
        wide_op,
        wide_ty,
        out: PackOut::Load { big_endian },
        touched,
    })
}

pub(super) fn build_pack(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (root_pos, root_val, res_ty): (usize, Value, Type),
) -> Option<PackPlan> {
    let (mut leaves, mut touched, mut addr_key) = (Vec::new(), Vec::new(), None);
    let avail = i64::from(res_ty.bytes());
    collect_leaves(
        ctx,
        chain,
        (root_val, 0, avail),
        (&mut leaves, &mut touched, &mut addr_key),
    )?;
    if !(2..=8).contains(&leaves.len()) {
        return None;
    }
    let min_off = contiguous_leaves(&mut leaves, None)?;
    let total_w: i64 = leaves.iter().map(|l| l.bytes).sum();
    if !matches!(total_w, 2 | 4 | 8)
        || total_w > avail
        || leaves.iter().any(|l| l.shift_byte + l.bytes > total_w)
    {
        return None;
    }
    let little = leaves
        .iter()
        .all(|l| !l.bswapped && l.shift_byte == l.canon_off - min_off);
    let big = !little
        && leaves.iter().all(|l| {
            (l.bytes == 1 || l.bswapped)
                && l.shift_byte == total_w - (l.canon_off - min_off) - l.bytes
        });
    if !little && !big {
        return None;
    }
    finish_pack_plan(
        ctx,
        chain,
        (root_pos, wide_mem_ty(total_w), big),
        &leaves,
        touched,
    )
}

fn unwrap_bitcast_leaf(
    ctx: &PackCtx<'_>,
    mut val: Value,
    expect_bytes: u32,
    touched: &mut Vec<Value>,
) -> Value {
    for _ in 0..4 {
        let Some(&Op::Convert(Convert::Bitcast, inner)) = def_op(ctx, val) else {
            break;
        };
        if ctx.func.ty(inner).bytes() != expect_bytes {
            break;
        }
        touched.push(val);
        val = inner;
    }
    val
}

fn collect_insert_leaves(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (val, vec_ty, byte_base): (Value, Type, i64),
    (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
    let (lane, count) = vec_ty.vec_shape()?;
    let mut lanes = [None; 16];
    let (mut cur, mut filled) = (val, 0_u8);
    while filled < count {
        let (blk, _, inst) = ctx.def_info.get(cur.index()).copied().flatten()?;
        if !chain.contains(&blk) || ctx.func.ty(cur) != vec_ty {
            return None;
        }
        let Op::Insert(prev, scalar, idx) = ctx.func.insts[inst.index()].op else {
            return None;
        };
        if usize::from(idx) >= usize::from(count) {
            return None;
        }
        touched.push(cur);
        if lanes[usize::from(idx)].is_none() {
            lanes[usize::from(idx)] = Some(scalar);
            filled += 1;
        }
        cur = prev;
    }
    if ctx
        .def_info
        .get(cur.index())
        .copied()
        .flatten()
        .is_some_and(|(b, _, inst)| {
            chain.contains(&b) && matches!(ctx.func.insts[inst.index()].op, Op::Const(_))
        })
    {
        touched.push(cur);
    }
    for (i, slot) in lanes[..usize::from(count)].iter().enumerate() {
        let leaf = unwrap_bitcast_leaf(ctx, (*slot)?, lane.bytes(), touched);
        push_leaf_load(
            ctx,
            chain,
            leaf,
            (byte_base + i as i64 * i64::from(lane.bytes()), lane.bytes()),
            (leaves, touched, key),
        )?;
    }
    Some(())
}

fn collect_vec_source(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (val, byte_base, expect_bytes): (Value, i64, u32),
    out: (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
    let (leaves, touched, key) = out;
    let cur = unwrap_bitcast_leaf(ctx, val, expect_bytes, touched);
    if push_leaf_load(
        ctx,
        chain,
        cur,
        (byte_base, expect_bytes),
        (leaves, touched, key),
    )
    .is_some()
    {
        return Some(());
    }
    collect_insert_leaves(
        ctx,
        chain,
        (cur, ctx.func.ty(cur), byte_base),
        (leaves, touched, key),
    )
}

pub(super) fn build_vec_pack(
    ctx: &PackCtx<'_>,
    target: &crate::Target,
    chain: &[Block],
    (root_pos, root_val, res_ty): (usize, Value, Type),
) -> Option<PackPlan> {
    if !res_ty.is_vector() || !res_ty.is_register_vec(target) || !matches!(res_ty.bytes(), 16 | 32)
    {
        return None;
    }
    let (mut leaves, mut touched, mut key) = (Vec::new(), Vec::new(), None);
    let inner = unwrap_bitcast_leaf(ctx, root_val, res_ty.bytes(), &mut touched);
    let (blk, _, inst) = ctx.def_info.get(inner.index()).copied().flatten()?;
    if !chain.contains(&blk) {
        return None;
    }
    match (res_ty.bytes(), &ctx.func.insts[inst.index()].op) {
        (16, Op::Insert(..)) => collect_insert_leaves(
            ctx,
            chain,
            (inner, ctx.func.ty(inner), 0),
            (&mut leaves, &mut touched, &mut key),
        )?,
        (32, Op::TargetOp(TargetOpId::X64Concat128, args)) if args.as_slice().len() == 2 => {
            touched.push(inner);
            let [low, high] = [args.as_slice()[0], args.as_slice()[1]];
            collect_vec_source(
                ctx,
                chain,
                (low, 0, 16),
                (&mut leaves, &mut touched, &mut key),
            )?;
            collect_vec_source(
                ctx,
                chain,
                (high, 16, 16),
                (&mut leaves, &mut touched, &mut key),
            )?;
        }
        _ => return None,
    }
    let min_off = contiguous_leaves(&mut leaves, Some(i64::from(res_ty.bytes())))?;
    if !leaves.iter().all(|l| l.shift_byte == l.canon_off - min_off) {
        return None;
    }
    finish_pack_plan(ctx, chain, (root_pos, res_ty, false), &leaves, touched)
}
