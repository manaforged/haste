use super::fits_mem_imm;
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, StackSlot, TargetOpId, Type, Unary, Value};
use crate::mem::{MemBase, MemLoc, Provenance};

struct PackCtx<'a> {
    func: &'a Function,
    cfg: &'a Cfg,
    prov: &'a Provenance,
    consts: &'a [Option<i64>],
    def_info: &'a [DefInfo],
}

const MAX_PRED_CHAIN: usize = 4;
const MAX_ADDR_STEPS: usize = 16;

#[derive(Clone, Copy)]
enum RawLoad {
    Stack,
    Addr(Value, i32),
}

#[derive(Clone, PartialEq, Eq)]
enum AddrKey {
    Slot(StackSlot),
    Global(u32),
    Expr(Vec<(u32, i64)>),
}

#[derive(Clone, Copy)]
struct Leaf {
    blk_idx: usize,
    pos: usize,
    loc: MemLoc,
    canon_off: i64,
    raw: RawLoad,
    shift_byte: i64,
    bytes: i64,
    bswapped: bool,
}

enum PackOut {
    Load { big_endian: bool },
    Cmp { cc: IntCC, mask: Option<i64>, rhs_const: Option<i64>, rhs_op: Option<Op> },
}

struct PackPlan {
    wide_op: Op,
    wide_ty: Type,
    out: PackOut,
    touched: Vec<Value>,
}

type DefInfo = Option<(Block, usize, Inst)>;

fn canon_addr_step(
    func: &Function,
    consts: &[Option<i64>],
    def_info: &[DefInfo],
    (v, sc): (Value, i64),
    work: &mut Vec<(Value, i64)>,
) -> Result<bool, ()> {
    let Some((_, _, inst)) = def_info.get(v.index()).copied().flatten() else { return Ok(false) };
    match func.insts[inst.index()].op {
        Op::Convert(Convert::Bitcast, x) if func.ty(x) == Type::I64 => work.push((x, sc)),
        Op::Binary(Binary::Iadd, a, b) => work.extend([(a, sc), (b, sc)]),
        Op::Binary(Binary::Isub, a, b) => work.extend([(a, sc), (b, sc.checked_neg().ok_or(())?)]),
        Op::Binary(Binary::Ishl, a, amt) => {
            let Some(sh) = consts.get(amt.index()).copied().flatten().filter(|&s| (0..63).contains(&s)) else { return Ok(false) };
            work.push((a, sc.checked_mul(1_i64 << sh).ok_or(())?));
        }
        Op::Binary(Binary::Imul, a, b) => {
            let Some((inner, k)) = consts.get(b.index()).copied().flatten().map(|k| (a, k))
                .or_else(|| consts.get(a.index()).copied().flatten().map(|k| (b, k))) else { return Ok(false) };
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

fn canonical_addr(func: &Function, prov: &Provenance, consts: &[Option<i64>], def_info: &[DefInfo], addr: Value, off: i32) -> Option<(AddrKey, i64)> {
    let loc = prov.loc(addr, i64::from(off), 1);
    match loc.base {
        MemBase::Slot(slot) => return Some((AddrKey::Slot(slot), loc.offset)),
        MemBase::Global(sym) => return Some((AddrKey::Global(sym), loc.offset)),
        MemBase::Addr(_) => {}
    }
    let (mut const_off, mut terms, mut work, mut steps) = (i64::from(off), Vec::new(), vec![(addr, 1_i64)], 0_usize);
    while let Some((v, sc)) = work.pop() {
        if let Some(imm) = consts.get(v.index()).copied().flatten() {
            const_off = const_off.checked_add(sc.checked_mul(imm)?)?;
            continue;
        }
        if steps < MAX_ADDR_STEPS && func.ty(v) == Type::I64 && canon_addr_step(func, consts, def_info, (v, sc), &mut work).ok()? {
            steps += 1;
            continue;
        }
        terms.push((v.index() as u32, sc));
    }
    Some((merge_canon_terms(terms)?, const_off))
}

pub(super) fn pred_chain(cfg: &Cfg, blk: Block) -> Vec<Block> {
    let mut chain = vec![blk];
    let mut cur = blk;
    while chain.len() < MAX_PRED_CHAIN
        && let Some(pred) = cfg.single_pred(cur)
        && cfg.reachable.get(pred.index()).copied().unwrap_or(false)
        && !chain.contains(&pred)
    {
        chain.push(pred);
        cur = pred;
    }
    chain.reverse();
    chain
}

fn push_leaf_load(ctx: &PackCtx<'_>, chain: &[Block], val: Value, (shift_byte, expect_bytes): (i64, u32), (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>)) -> Option<()> {
    let (lblk, lpos, linst) = ctx.def_info.get(val.index()).copied().flatten()?;
    let blk_idx = chain.iter().position(|&b| b == lblk)?;
    let ty = ctx.func.ty(val);
    if ty.bytes() != expect_bytes {
        return None;
    }
    let (loc, raw, addr_key, canon_off) = match ctx.func.insts[linst.index()].op {
        Op::Load(addr, off) => {
            let (ak, coff) = canonical_addr(ctx.func, ctx.prov, ctx.consts, ctx.def_info, addr, off)?;
            (ctx.prov.loc(addr, i64::from(off), ty.bytes()), RawLoad::Addr(addr, off), ak, coff)
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
    leaves.push(Leaf { blk_idx, pos: lpos, loc, canon_off, raw, shift_byte, bytes: i64::from(expect_bytes), bswapped: false });
    Some(())
}

fn def_op<'a>(ctx: &'a PackCtx<'_>, v: Value) -> Option<&'a Op> {
    let (_, _, inst) = ctx.def_info.get(v.index()).copied().flatten()?;
    Some(&ctx.func.insts[inst.index()].op)
}

fn peel_shift_ext(ctx: &PackCtx<'_>, mut val: Value, (mut shift, mut avail): (i64, i64), touched: &mut Vec<Value>) -> Option<(Value, i64, i64, bool)> {
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
    if !load_ty.is_small_int() || !(min_bytes..=4).contains(&load_ty.bytes()) || i64::from(load_ty.bytes()) > avail {
        return None;
    }
    push_leaf_load(ctx, chain, cur, (shift, load_ty.bytes()), (leaves, touched, key))?;
    if bswapped {
        leaves.last_mut()?.bswapped = true;
    }
    Some(())
}

fn has_clobber(func: &Function, prov: &Provenance, chain: &[Block], leaves: &[Leaf], root_pos: usize, wide_loc: MemLoc) -> bool {
    let Some(min_leaf) = leaves.iter().min_by_key(|l| (l.blk_idx, l.pos)) else {
        return true;
    };
    let last_chain = chain.len() - 1;
    let check_inst = |inst: Inst| {
        let op = &func.insts[inst.index()].op;
        super::dom::clobbers_loc(func, (op, None), wide_loc, prov) || leaves.iter().any(|l| super::dom::clobbers_loc(func, (op, None), l.loc, prov))
    };
    for ci in min_leaf.blk_idx..=last_chain {
        let insts = &func.blocks[chain[ci].index()].insts;
        let lo = if ci == min_leaf.blk_idx { min_leaf.pos + 1 } else { 0 };
        let hi = if ci == last_chain { root_pos } else { insts.len() };
        if lo < hi && insts[lo..hi].iter().copied().any(check_inst) {
            return true;
        }
    }
    false
}

fn build_wide_load(ctx: &PackCtx<'_>, root_blk: Block, root_pos: usize, first: Leaf, wide_ty: Type) -> Option<Op> {
    match (first.loc.base, first.raw) {
        (MemBase::Slot(slot), _) => Some(Op::StackLoad(slot, i32::try_from(first.loc.offset).ok()?)),
        (_, RawLoad::Addr(addr, off)) if fits_mem_imm(wide_ty, i64::from(off)) => Some(Op::Load(addr, off)),
        (MemBase::Addr(base), _)
            if ctx.def_info.get(base.index()).copied().flatten().is_none_or(|(db, dp, _)| {
                ctx.cfg.dominates(db, root_blk) && (db != root_blk || dp < root_pos)
            }) =>
        {
            let off_i32 = i32::try_from(first.loc.offset).ok()?;
            fits_mem_imm(wide_ty, first.loc.offset).then_some(Op::Load(base, off_i32))
        }
        _ => None,
    }
}

fn contiguous_leaves(leaves: &mut [Leaf], expected_bytes: Option<i64>) -> Option<i64> {
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
    if expected_bytes.is_some_and(|want| want != total) {
        return None;
    }
    Some(min_off)
}

fn wide_mem_ty(total_w: i64) -> Type {
    if total_w == 2 { Type::I16 } else if total_w == 4 { Type::I32 } else { Type::I64 }
}

fn checked_wide_load(ctx: &PackCtx<'_>, chain: &[Block], root_pos: usize, leaves: &[Leaf], wide_ty: Type) -> Option<Op> {
    let first = leaves[0];
    let wide_op = build_wide_load(ctx, *chain.last()?, root_pos, first, wide_ty)?;
    let wide_loc = MemLoc::new(first.loc.root, first.loc.base, first.loc.offset, wide_ty.bytes());
    (!has_clobber(ctx.func, ctx.prov, chain, leaves, root_pos, wide_loc)).then_some(wide_op)
}

fn finish_pack_plan(ctx: &PackCtx<'_>, chain: &[Block], (root_pos, wide_ty, big_endian): (usize, Type, bool), leaves: &[Leaf], touched: Vec<Value>) -> Option<PackPlan> {
    let wide_op = checked_wide_load(ctx, chain, root_pos, leaves, wide_ty)?;
    Some(PackPlan { wide_op, wide_ty, out: PackOut::Load { big_endian }, touched })
}

fn build_pack(ctx: &PackCtx<'_>, chain: &[Block], (root_pos, root_val, res_ty): (usize, Value, Type)) -> Option<PackPlan> {
    let (mut leaves, mut touched, mut addr_key) = (Vec::new(), Vec::new(), None);
    let avail = i64::from(res_ty.bytes());
    collect_leaves(ctx, chain, (root_val, 0, avail), (&mut leaves, &mut touched, &mut addr_key))?;
    if !(2..=8).contains(&leaves.len()) {
        return None;
    }
    let min_off = contiguous_leaves(&mut leaves, None)?;
    let total_w: i64 = leaves.iter().map(|l| l.bytes).sum();
    if !matches!(total_w, 2 | 4 | 8) || total_w > avail || leaves.iter().any(|l| l.shift_byte + l.bytes > total_w) {
        return None;
    }
    let little = leaves.iter().all(|l| !l.bswapped && l.shift_byte == l.canon_off - min_off);
    let big = !little && leaves.iter().all(|l| (l.bytes == 1 || l.bswapped) && l.shift_byte == total_w - (l.canon_off - min_off) - l.bytes);
    if !little && !big {
        return None;
    }
    finish_pack_plan(ctx, chain, (root_pos, wide_mem_ty(total_w), big), &leaves, touched)
}

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
    c(b).map(|imm| (a, imm)).or_else(|| c(a).map(|imm| (b, imm)))
}

fn canon_const(raw: u64, ty: Type) -> i64 {
    if ty == Type::I16 { (raw as i16) as i64 } else if ty == Type::I32 { (raw as i32) as i64 } else { raw as i64 }
}

fn peel_uext(ctx: &PackCtx<'_>, mut v: Value, touched: &mut Vec<Value>) -> Value {
    while let Some(&Op::Convert(Convert::Uextend, inner)) = def_op(ctx, v) {
        touched.push(v);
        v = inner;
    }
    v
}

fn peel_masked_load(ctx: &PackCtx<'_>, mut v: Value, mut mask: u64, touched: &mut Vec<Value>) -> Option<(Value, u64)> {
    loop {
        v = peel_uext(ctx, v, touched);
        let Some(&Op::Binary(Binary::Band, a, b)) = def_op(ctx, v) else { break };
        let (inner, imm) = const_pair(ctx, a, b)?;
        touched.push(v);
        mask &= imm as u64;
        v = inner;
    }
    let ty = ctx.func.ty(v);
    (ty.is_small_int() && ty.bytes() <= 4).then_some((v, mask & (!0_u64 >> (64 - ty.bytes() * 8))))
}

fn push_cmp_masked(ctx: &PackCtx<'_>, chain: &[Block], val: Value, (init_m, c_bits): (u64, u64), acc: &mut CmpAcc) -> Option<()> {
    if acc.two_ptr == Some(true) {
        return None;
    }
    acc.two_ptr = Some(false);
    let (load_v, mask) = peel_masked_load(ctx, val, init_m, &mut acc.touched)?;
    if mask == 0 || (c_bits & !mask) != 0 {
        return None;
    }
    let idx = acc.masks.len() as i64;
    push_leaf_load(ctx, chain, load_v, (idx, ctx.func.ty(load_v).bytes()), (&mut acc.leaves_a, &mut acc.touched, &mut acc.key_a))?;
    acc.masks.push((mask, c_bits));
    Some(())
}

fn push_cmp_eq(ctx: &PackCtx<'_>, chain: &[Block], (a, b): (Value, Value), acc: &mut CmpAcc) -> Option<()> {
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
    let (la, lb) = (peel_uext(ctx, a, &mut acc.touched), peel_uext(ctx, b, &mut acc.touched));
    let ty = ctx.func.ty(la);
    if ty != ctx.func.ty(lb) || !ty.is_small_int() || ty.bytes() > 4 {
        return None;
    }
    let idx = acc.leaves_a.len() as i64;
    push_leaf_load(ctx, chain, la, (idx, ty.bytes()), (&mut acc.leaves_a, &mut acc.touched, &mut acc.key_a))?;
    push_leaf_load(ctx, chain, lb, (idx, ty.bytes()), (&mut acc.leaves_b, &mut acc.touched, &mut acc.key_b))
}

fn parse_bor_zero_leaf(ctx: &PackCtx<'_>, chain: &[Block], val: Value, acc: &mut CmpAcc) -> Option<()> {
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

fn collect_cmp_leaves(ctx: &PackCtx<'_>, chain: &[Block], val: Value, (join_op, leaf_cc): (Binary, Option<IntCC>), acc: &mut CmpAcc) -> Option<()> {
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

fn finish_two_ptr_cmp(ctx: &PackCtx<'_>, chain: &[Block], (root_pos, cc, wide_ty, min_off): (usize, IntCC, Type, i64), (wide_op, acc): (Op, CmpAcc)) -> Option<PackPlan> {
    if acc.leaves_b.len() != acc.leaves_a.len() {
        return None;
    }
    let pairs_b: Vec<Leaf> = acc.leaves_a.iter().map(|la| acc.leaves_b[la.shift_byte as usize]).collect();
    let min_b = pairs_b[0].canon_off;
    if !acc.leaves_a.iter().zip(&pairs_b).all(|(la, lb)| la.bytes == lb.bytes && lb.canon_off - min_b == la.canon_off - min_off) {
        return None;
    }
    let rhs_op = checked_wide_load(ctx, chain, root_pos, &pairs_b, wide_ty)?;
    Some(PackPlan { wide_op, wide_ty, out: PackOut::Cmp { cc, mask: None, rhs_const: None, rhs_op: Some(rhs_op) }, touched: acc.touched })
}

fn finish_cmp_pack(ctx: &PackCtx<'_>, chain: &[Block], (root_pos, cc): (usize, IntCC), mut acc: CmpAcc) -> Option<PackPlan> {
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
    let out = PackOut::Cmp { cc, mask, rhs_const: Some(canon_const(wide_val, wide_ty)), rhs_op: None };
    Some(PackPlan { wide_op, wide_ty, out, touched: acc.touched })
}

fn build_cmp_pack(ctx: &PackCtx<'_>, chain: &[Block], (root_pos, inst, res_ty): (usize, Inst, Type)) -> Option<PackPlan> {
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
    (acc.leaves_a.len() >= 2).then(|| finish_cmp_pack(ctx, chain, (root_pos, out_cc), acc)).flatten()
}

fn unwrap_bitcast_leaf(ctx: &PackCtx<'_>, mut val: Value, expect_bytes: u32, touched: &mut Vec<Value>) -> Value {
    for _ in 0..4 {
        let Some(&Op::Convert(Convert::Bitcast, inner)) = def_op(ctx, val) else { break };
        if ctx.func.ty(inner).bytes() != expect_bytes {
            break;
        }
        touched.push(val);
        val = inner;
    }
    val
}

fn collect_insert_leaves(ctx: &PackCtx<'_>, chain: &[Block], (val, vec_ty, byte_base): (Value, Type, i64), (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>)) -> Option<()> {
    let (lane, count) = vec_ty.vec_shape()?;
    let mut lanes = [None; 16];
    let (mut cur, mut filled) = (val, 0_u8);
    while filled < count {
        let (blk, _, inst) = ctx.def_info.get(cur.index()).copied().flatten()?;
        if !chain.contains(&blk) || ctx.func.ty(cur) != vec_ty {
            return None;
        }
        let Op::Insert(prev, scalar, idx) = ctx.func.insts[inst.index()].op else { return None };
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
    if ctx.def_info.get(cur.index()).copied().flatten().is_some_and(|(b, _, inst)| chain.contains(&b) && matches!(ctx.func.insts[inst.index()].op, Op::Const(_))) {
        touched.push(cur);
    }
    for (i, slot) in lanes[..usize::from(count)].iter().enumerate() {
        let leaf = unwrap_bitcast_leaf(ctx, (*slot)?, lane.bytes(), touched);
        push_leaf_load(ctx, chain, leaf, (byte_base + i as i64 * i64::from(lane.bytes()), lane.bytes()), (leaves, touched, key))?;
    }
    Some(())
}

fn collect_vec_source(ctx: &PackCtx<'_>, chain: &[Block], (val, byte_base, expect_bytes): (Value, i64, u32), out: (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>)) -> Option<()> {
    let (leaves, touched, key) = out;
    let cur = unwrap_bitcast_leaf(ctx, val, expect_bytes, touched);
    if push_leaf_load(ctx, chain, cur, (byte_base, expect_bytes), (leaves, touched, key)).is_some() {
        return Some(());
    }
    collect_insert_leaves(ctx, chain, (cur, ctx.func.ty(cur), byte_base), (leaves, touched, key))
}

fn build_vec_pack(ctx: &PackCtx<'_>, target: &crate::Target, chain: &[Block], (root_pos, root_val, res_ty): (usize, Value, Type)) -> Option<PackPlan> {
    if !res_ty.is_vector() || !res_ty.is_register_vec(target) || !matches!(res_ty.bytes(), 16 | 32) {
        return None;
    }
    let (mut leaves, mut touched, mut key) = (Vec::new(), Vec::new(), None);
    let inner = unwrap_bitcast_leaf(ctx, root_val, res_ty.bytes(), &mut touched);
    let (blk, _, inst) = ctx.def_info.get(inner.index()).copied().flatten()?;
    if !chain.contains(&blk) {
        return None;
    }
    match (res_ty.bytes(), &ctx.func.insts[inst.index()].op) {
        (16, Op::Insert(..)) => collect_insert_leaves(ctx, chain, (inner, ctx.func.ty(inner), 0), (&mut leaves, &mut touched, &mut key))?,
        (32, Op::TargetOp(TargetOpId::X64Concat128, args)) if args.as_slice().len() == 2 => {
            touched.push(inner);
            let [low, high] = [args.as_slice()[0], args.as_slice()[1]];
            collect_vec_source(ctx, chain, (low, 0, 16), (&mut leaves, &mut touched, &mut key))?;
            collect_vec_source(ctx, chain, (high, 16, 16), (&mut leaves, &mut touched, &mut key))?;
        }
        _ => return None,
    }
    let min_off = contiguous_leaves(&mut leaves, Some(i64::from(res_ty.bytes())))?;
    if !leaves.iter().all(|l| l.shift_byte == l.canon_off - min_off) {
        return None;
    }
    finish_pack_plan(ctx, chain, (root_pos, res_ty, false), &leaves, touched)
}

fn sweep_dead(func: &mut Function, blk_idx: usize, dead_pool: &[bool], uses: &mut [u32], def_info: &mut [DefInfo]) {
    let insts = std::mem::take(&mut func.blocks[blk_idx].insts);
    let mut live = Vec::with_capacity(insts.len());
    for &inst in insts.iter().rev() {
        let data = &func.insts[inst.index()];
        if let Some(res) = data.result
            && dead_pool.get(res.index()).copied().unwrap_or(false)
            && uses[res.index()] == 0
            && data.op.is_removable()
        {
            data.op.for_operands(|v| uses[v.index()] = uses[v.index()].saturating_sub(1));
            def_info[res.index()] = None;
            continue;
        }
        live.push(inst);
    }
    live.reverse();
    let blk = Block(blk_idx as u32);
    for (pos, &inst) in live.iter().enumerate() {
        if let Some(res) = func.insts[inst.index()].result {
            def_info[res.index()] = Some((blk, pos, inst));
        }
    }
    func.blocks[blk_idx].insts = live;
}

pub(super) fn has_vec_pack_cand(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        block.insts.iter().any(|&inst| {
            let data = &func.insts[inst.index()];
            data.result.is_some_and(|r| func.ty(r).is_vector())
                && matches!(data.op, Op::Insert(..) | Op::TargetOp(TargetOpId::X64Concat128, _))
        })
    })
}

struct PackTables {
    consts: Vec<Option<i64>>,
    def_info: Vec<DefInfo>,
    uses: Vec<u32>,
    dead_pool: Vec<bool>,
    touched_blk: Vec<bool>,
}

fn build_pack_tables(func: &Function, cfg: &Cfg) -> PackTables {
    let n = func.types.len();
    let (mut consts, mut def_info) = (vec![None; n], vec![None; n]);
    for &blk in &cfg.rpo {
        for (pos, &inst) in func.blocks[blk.index()].insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            def_info[res.index()] = Some((blk, pos, inst));
            if let Op::Const(imm) = data.op {
                consts[res.index()] = Some(imm);
            }
        }
    }
    PackTables { consts, def_info, uses: func.use_counts(), dead_pool: vec![false; n], touched_blk: vec![false; func.blocks.len()] }
}

fn build_scalar_pack(ctx: &PackCtx<'_>, chain: &[Block], (pos, inst, res, res_ty): (usize, Inst, Value, Type)) -> Option<PackPlan> {
    if matches!(ctx.func.insts[inst.index()].op, Op::Binary(Binary::Bor | Binary::Iadd, ..))
        && matches!(res_ty, Type::I16 | Type::I32 | Type::I64)
        && let Some(plan) = build_pack(ctx, chain, (pos, res, res_ty))
    {
        return Some(plan);
    }
    build_cmp_pack(ctx, chain, (pos, inst, res_ty))
}

fn plan_one_pack(ctx: &PackCtx<'_>, target: Option<&crate::Target>, chain: &[Block], uses: &[u32], (pos, inst): (usize, Inst)) -> Option<(Value, Type, PackPlan)> {
    let data = &ctx.func.insts[inst.index()];
    let res = data.result?;
    let res_ty = ctx.func.ty(res);
    let plan = match target {
        None if matches!(data.op, Op::Binary(Binary::Bor | Binary::Iadd | Binary::Band, ..) | Op::Icmp(..)) => {
            build_scalar_pack(ctx, chain, (pos, inst, res, res_ty))
        }
        Some(tgt)
            if matches!(data.op, Op::Insert(..) | Op::TargetOp(TargetOpId::X64Concat128, _))
                || matches!(data.op, Op::Convert(Convert::Bitcast, x) if uses[x.index()] == 1) =>
        {
            build_vec_pack(ctx, tgt, chain, (pos, res, res_ty))
        }
        _ => None,
    }?;
    Some((res, res_ty, plan))
}

fn plan_block_packs(ctx: &PackCtx<'_>, target: Option<&crate::Target>, blk: Block, uses: &[u32]) -> (Vec<Option<(Value, Type, PackPlan)>>, bool) {
    let chain = pred_chain(ctx.cfg, blk);
    let insts = &ctx.func.blocks[blk.index()].insts;
    let mut consumed = vec![false; ctx.func.types.len()];
    let mut plans: Vec<Option<(Value, Type, PackPlan)>> = (0..insts.len()).map(|_| None).collect();
    let mut any_plan = false;
    for (pos, &inst) in insts.iter().enumerate().rev() {
        let Some(res) = ctx.func.insts[inst.index()].result.filter(|r| !consumed[r.index()]) else { continue };
        let Some((_, res_ty, p)) = plan_one_pack(ctx, target, &chain, uses, (pos, inst)) else { continue };
        for &v in &p.touched {
            if v != res && uses.get(v.index()).copied().unwrap_or(0) <= 1 {
                consumed[v.index()] = true;
            }
        }
        plans[pos] = Some((res, res_ty, p));
        any_plan = true;
    }
    (plans, any_plan)
}

fn push_fresh_val(func: &mut Function, op: Op, ty: Type, out: &mut Vec<Inst>, t: &mut PackTables) -> Value {
    let val = func.value(ty);
    t.uses.push(1);
    t.def_info.push(None);
    t.dead_pool.push(false);
    t.consts.push(None);
    out.push(func.add(op, Some(val)));
    val
}

fn emit_cmp_inst(func: &mut Function, inst: Inst, res_ty: Type, plan: PackPlan, (out, t): (&mut Vec<Inst>, &mut PackTables)) {
    let PackOut::Cmp { cc, mask, rhs_const, rhs_op } = plan.out else { return };
    let mut lhs = push_fresh_val(func, plan.wide_op, plan.wide_ty, out, t);
    if let Some(m) = mask {
        let mv = push_fresh_val(func, Op::Const(m), plan.wide_ty, out, t);
        lhs = push_fresh_val(func, Op::Binary(Binary::Band, lhs, mv), plan.wide_ty, out, t);
    }
    let rhs = push_fresh_val(func, rhs_const.map(Op::Const).or(rhs_op).unwrap_or(Op::Const(0)), plan.wide_ty, out, t);
    let cmp = Op::Icmp(cc, lhs, rhs);
    func.insts[inst.index()].op = if res_ty == Type::I8 {
        cmp
    } else {
        let cv = push_fresh_val(func, cmp, Type::I8, out, t);
        Op::Convert(Convert::Uextend, cv)
    };
}

fn emit_packed_inst(func: &mut Function, inst: Inst, res_ty: Type, plan: PackPlan, (out, t): (&mut Vec<Inst>, &mut PackTables)) {
    let PackOut::Load { big_endian } = plan.out else {
        emit_cmp_inst(func, inst, res_ty, plan, (out, t));
        return;
    };
    if !big_endian && plan.wide_ty == res_ty {
        func.insts[inst.index()].op = plan.wide_op;
        return;
    }
    let wide_val = push_fresh_val(func, plan.wide_op, plan.wide_ty, out, t);
    func.insts[inst.index()].op = if big_endian && plan.wide_ty == res_ty {
        Op::Unary(Unary::Bswap, wide_val)
    } else if big_endian {
        let sw = push_fresh_val(func, Op::Unary(Unary::Bswap, wide_val), plan.wide_ty, out, t);
        Op::Convert(Convert::Uextend, sw)
    } else {
        Op::Convert(Convert::Uextend, wide_val)
    };
}

fn mark_dead_touched(touched: &[Value], res: Value, t: &mut PackTables) {
    for &v in touched {
        if v == res {
            continue;
        }
        t.dead_pool[v.index()] = true;
        if let Some((db, _, _)) = t.def_info[v.index()] {
            t.touched_blk[db.index()] = true;
        }
    }
}

fn apply_block_packs(func: &mut Function, b_idx: usize, mut plans: Vec<Option<(Value, Type, PackPlan)>>, t: &mut PackTables) -> bool {
    let insts = std::mem::take(&mut func.blocks[b_idx].insts);
    let mut out = Vec::with_capacity(insts.len());
    let mut changed = false;
    for (pos, &inst) in insts.iter().enumerate() {
        if let Some((res, res_ty, plan)) = plans[pos].take() {
            func.insts[inst.index()].op.for_operands(|v| t.uses[v.index()] = t.uses[v.index()].saturating_sub(1));
            mark_dead_touched(&plan.touched, res, t);
            emit_packed_inst(func, inst, res_ty, plan, (&mut out, t));
            changed = true;
        }
        out.push(inst);
    }
    func.blocks[b_idx].insts = out;
    t.touched_blk[b_idx] = true;
    sweep_dead(func, b_idx, &t.dead_pool, &mut t.uses, &mut t.def_info);
    changed
}

pub(super) fn coalesce_vec_rescan(func: &mut Function, cfg: &Cfg, target: &crate::Target) -> bool {
    if !has_vec_pack_cand(func) {
        return false;
    }
    let prov = Provenance::new(func, cfg);
    coalesce_loads(func, &prov, target)
}

fn run_pack_round(func: &mut Function, cfg: &Cfg, prov: &Provenance, target: Option<&crate::Target>, t: &mut PackTables) -> bool {
    let mut changed = false;
    for &blk in &cfg.rpo {
        let (plans, any_plan) = {
            let ctx = PackCtx { func, cfg, prov, consts: &t.consts, def_info: &t.def_info };
            plan_block_packs(&ctx, target, blk, &t.uses)
        };
        if any_plan {
            changed |= apply_block_packs(func, blk.index(), plans, t);
        }
    }
    if changed {
        for &blk in cfg.rpo.iter().rev() {
            if t.touched_blk[blk.index()] {
                sweep_dead(func, blk.index(), &t.dead_pool, &mut t.uses, &mut t.def_info);
            }
        }
    }
    changed
}

pub(super) fn coalesce_loads(func: &mut Function, prov: &Provenance, target: &crate::Target) -> bool {
    let cfg = func.cfg();
    let mut t = build_pack_tables(func, &cfg);
    let mut changed = run_pack_round(func, &cfg, prov, None, &mut t);
    if has_vec_pack_cand(func) {
        t.touched_blk.fill(false);
        changed |= run_pack_round(func, &cfg, prov, Some(target), &mut t);
    }
    changed
}
