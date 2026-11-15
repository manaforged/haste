use super::fits_mem_imm;
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, StackSlot, TargetOpId, Type, Unary, Value};
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
}

struct PackPlan {
    wide_op: Op,
    wide_ty: Type,
    big_endian: bool,
    touched: Vec<Value>,
}

type DefInfo = Option<(Block, usize, Inst)>;

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
    let mut const_off = i64::from(off);
    let mut terms: Vec<(u32, i64)> = Vec::new();
    let mut work = vec![(addr, 1_i64)];
    let mut steps = 0_usize;
    while let Some((v, sc)) = work.pop() {
        if let Some(imm) = consts.get(v.index()).copied().flatten() {
            const_off = const_off.checked_add(sc.checked_mul(imm)?)?;
            continue;
        }
        if steps < MAX_ADDR_STEPS
            && func.ty(v) == Type::I64
            && let Some((_, _, inst)) = def_info.get(v.index()).copied().flatten()
        {
            steps += 1;
            match func.insts[inst.index()].op {
                Op::Convert(Convert::Bitcast, x) if func.ty(x) == Type::I64 => {
                    work.push((x, sc));
                    continue;
                }
                Op::Binary(Binary::Iadd, a, b) => {
                    work.push((a, sc));
                    work.push((b, sc));
                    continue;
                }
                Op::Binary(Binary::Isub, a, b) => {
                    work.push((a, sc));
                    work.push((b, sc.checked_neg()?));
                    continue;
                }
                Op::Binary(Binary::Ishl, a, amt) => {
                    if let Some(sh) = consts.get(amt.index()).copied().flatten()
                        && (0..63).contains(&sh)
                    {
                        work.push((a, sc.checked_mul(1_i64 << sh)?));
                        continue;
                    }
                }
                Op::Binary(Binary::Imul, a, b) => {
                    if let Some(k) = consts.get(b.index()).copied().flatten() {
                        work.push((a, sc.checked_mul(k)?));
                        continue;
                    }
                    if let Some(k) = consts.get(a.index()).copied().flatten() {
                        work.push((b, sc.checked_mul(k)?));
                        continue;
                    }
                }
                _ => {}
            }
        }
        terms.push((v.index() as u32, sc));
    }
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
    Some((AddrKey::Expr(merged), const_off))
}

fn pred_chain(cfg: &Cfg, blk: Block) -> Vec<Block> {
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

fn push_leaf_load(
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
    leaves.push(Leaf { blk_idx, pos: lpos, loc, canon_off, raw, shift_byte, bytes: i64::from(expect_bytes) });
    Some(())
}

fn collect_leaves(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    val: Value,
    (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
    let (_, _, inst) = ctx.def_info.get(val.index()).copied().flatten()?;
    match ctx.func.insts[inst.index()].op {
        Op::Binary(Binary::Bor | Binary::Iadd, a, b) => {
            touched.push(val);
            collect_leaves(ctx, chain, a, (leaves, touched, key))?;
            collect_leaves(ctx, chain, b, (leaves, touched, key))
        }
        _ => {
            let mut cur = val;
            let mut shift_byte = 0_i64;
            let mut shift_ty_bytes: Option<i64> = None;
            loop {
                let Some((_, _, linst)) = ctx.def_info.get(cur.index()).copied().flatten() else { return None };
                match ctx.func.insts[linst.index()].op {
                    Op::Convert(Convert::Uextend, inner) => {
                        touched.push(cur);
                        cur = inner;
                    }
                    Op::Binary(Binary::Ishl, inner, amt) if shift_ty_bytes.is_none() => {
                        let imm = ctx.consts.get(amt.index()).copied().flatten()?;
                        if !(0..64).contains(&imm) || imm % 8 != 0 {
                            return None;
                        }
                        touched.push(cur);
                        shift_byte = imm / 8;
                        shift_ty_bytes = Some(i64::from(ctx.func.ty(cur).bytes()));
                        cur = inner;
                    }
                    _ => break,
                }
            }
            let load_ty = ctx.func.ty(cur);
            if !matches!(load_ty, Type::I8 | Type::I16 | Type::I32) || shift_ty_bytes.is_some_and(|st| shift_byte + i64::from(load_ty.bytes()) > st) {
                return None;
            }
            push_leaf_load(ctx, chain, cur, (shift_byte, load_ty.bytes()), (leaves, touched, key))
        }
    }
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
        super::dom::clobbers_loc(func, op, wide_loc, prov) || leaves.iter().any(|l| super::dom::clobbers_loc(func, op, l.loc, prov))
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

fn finish_pack_plan(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (root_pos, wide_ty, big_endian): (usize, Type, bool),
    leaves: &[Leaf],
    touched: Vec<Value>,
) -> Option<PackPlan> {
    let first = leaves[0];
    let wide_op = build_wide_load(ctx, *chain.last()?, root_pos, first, wide_ty)?;
    let wide_loc = MemLoc::new(first.loc.root, first.loc.base, first.loc.offset, wide_ty.bytes());
    if has_clobber(ctx.func, ctx.prov, chain, leaves, root_pos, wide_loc) {
        return None;
    }
    Some(PackPlan { wide_op, wide_ty, big_endian, touched })
}

fn build_pack(ctx: &PackCtx<'_>, chain: &[Block], (root_pos, root_val, res_ty): (usize, Value, Type)) -> Option<PackPlan> {
    let (mut leaves, mut touched, mut addr_key) = (Vec::new(), Vec::new(), None);
    collect_leaves(ctx, chain, root_val, (&mut leaves, &mut touched, &mut addr_key))?;
    if !(2..=8).contains(&leaves.len()) {
        return None;
    }
    let min_off = contiguous_leaves(&mut leaves, None)?;
    let total_w: i64 = leaves.iter().map(|l| l.bytes).sum();
    if !matches!(total_w, 2 | 4 | 8) || total_w > i64::from(res_ty.bytes()) || leaves.iter().any(|l| l.shift_byte + l.bytes > total_w) {
        return None;
    }
    let little = leaves.iter().all(|l| l.shift_byte == l.canon_off - min_off);
    let big = !little && leaves.iter().all(|l| l.bytes == 1 && l.shift_byte == total_w - (l.canon_off - min_off) - 1);
    if !little && !big {
        return None;
    }
    let wide_ty = match total_w {
        2 => Type::I16,
        4 => Type::I32,
        _ => Type::I64,
    };
    finish_pack_plan(ctx, chain, (root_pos, wide_ty, big), &leaves, touched)
}

fn unwrap_bitcast_leaf(ctx: &PackCtx<'_>, mut val: Value, expect_bytes: u32, touched: &mut Vec<Value>) -> Value {
    for _ in 0..4 {
        let Some((_, _, inst)) = ctx.def_info.get(val.index()).copied().flatten() else { break };
        if let Op::Convert(Convert::Bitcast, inner) = ctx.func.insts[inst.index()].op
            && ctx.func.ty(inner).bytes() == expect_bytes
        {
            touched.push(val);
            val = inner;
            continue;
        }
        break;
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

fn collect_vec_source(
    ctx: &PackCtx<'_>,
    chain: &[Block],
    (val, byte_base, expect_bytes): (Value, i64, u32),
    out: (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
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
    func.insts.iter().any(|data| {
        data.result.is_some_and(|r| func.ty(r).is_vector())
            && matches!(data.op, Op::Insert(..) | Op::TargetOp(TargetOpId::X64Concat128, _))
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

fn plan_one_pack(
    ctx: &PackCtx<'_>,
    target: Option<&crate::Target>,
    chain: &[Block],
    uses: &[u32],
    (pos, inst): (usize, Inst),
) -> Option<(Value, Type, PackPlan)> {
    let data = &ctx.func.insts[inst.index()];
    let res = data.result?;
    let res_ty = ctx.func.ty(res);
    let plan = match target {
        None if matches!(data.op, Op::Binary(Binary::Bor | Binary::Iadd, ..)) && matches!(res_ty, Type::I16 | Type::I32 | Type::I64) => {
            build_pack(ctx, chain, (pos, res, res_ty))
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

fn plan_block_packs(
    ctx: &PackCtx<'_>,
    target: Option<&crate::Target>,
    blk: Block,
    uses: &[u32],
) -> (Vec<Option<(Value, Type, PackPlan)>>, bool) {
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

fn emit_packed_inst(func: &mut Function, inst: Inst, res_ty: Type, plan: PackPlan, (out, t): (&mut Vec<Inst>, &mut PackTables)) {
    if !plan.big_endian && plan.wide_ty == res_ty {
        func.insts[inst.index()].op = plan.wide_op;
        return;
    }
    let wide_val = push_fresh_val(func, plan.wide_op, plan.wide_ty, out, t);
    let swapped = if plan.big_endian && plan.wide_ty == res_ty {
        Op::Unary(Unary::Bswap, wide_val)
    } else if plan.big_endian {
        let bswapped = push_fresh_val(func, Op::Unary(Unary::Bswap, wide_val), plan.wide_ty, out, t);
        Op::Convert(Convert::Uextend, bswapped)
    } else {
        Op::Convert(Convert::Uextend, wide_val)
    };
    func.insts[inst.index()].op = swapped;
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

fn apply_block_packs(
    func: &mut Function,
    b_idx: usize,
    mut plans: Vec<Option<(Value, Type, PackPlan)>>,
    t: &mut PackTables,
) -> bool {
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
