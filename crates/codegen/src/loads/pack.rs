use super::fits_mem_imm;
use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, StackSlot, Type, Unary, Value};
use crate::mem::{MemBase, MemLoc, Provenance};

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

fn collect_leaves(
    func: &Function,
    prov: &Provenance,
    consts: &[Option<i64>],
    def_info: &[DefInfo],
    chain: &[Block],
    val: Value,
    (leaves, touched, key): (&mut Vec<Leaf>, &mut Vec<Value>, &mut Option<AddrKey>),
) -> Option<()> {
    let (_, _, inst) = def_info.get(val.index()).copied().flatten()?;
    match func.insts[inst.index()].op {
        Op::Binary(Binary::Bor | Binary::Iadd, a, b) => {
            touched.push(val);
            collect_leaves(func, prov, consts, def_info, chain, a, (leaves, touched, key))?;
            collect_leaves(func, prov, consts, def_info, chain, b, (leaves, touched, key))
        }
        _ => {
            let mut cur = val;
            let mut shift_byte = 0_i64;
            let mut shift_ty_bytes: Option<i64> = None;
            loop {
                let Some((_, _, linst)) = def_info.get(cur.index()).copied().flatten() else { return None };
                match func.insts[linst.index()].op {
                    Op::Convert(Convert::Uextend, inner) => {
                        touched.push(cur);
                        cur = inner;
                    }
                    Op::Binary(Binary::Ishl, inner, amt) if shift_ty_bytes.is_none() => {
                        let imm = consts.get(amt.index()).copied().flatten()?;
                        if !(0..64).contains(&imm) || imm % 8 != 0 {
                            return None;
                        }
                        touched.push(cur);
                        shift_byte = imm / 8;
                        shift_ty_bytes = Some(i64::from(func.ty(cur).bytes()));
                        cur = inner;
                    }
                    _ => break,
                }
            }
            let (lblk, lpos, linst) = def_info.get(cur.index()).copied().flatten()?;
            let blk_idx = chain.iter().position(|&b| b == lblk)?;
            let load_ty = func.ty(cur);
            if !matches!(load_ty, Type::I8 | Type::I16) {
                return None;
            }
            let load_bytes = i64::from(load_ty.bytes());
            if shift_ty_bytes.is_some_and(|st| shift_byte + load_bytes > st) {
                return None;
            }
            let (loc, raw, addr_key, canon_off) = match func.insts[linst.index()].op {
                Op::Load(addr, off) => {
                    let (ak, coff) = canonical_addr(func, prov, consts, def_info, addr, off)?;
                    (prov.loc(addr, i64::from(off), load_ty.bytes()), RawLoad::Addr(addr, off), ak, coff)
                }
                Op::StackLoad(slot, off) => {
                    let loc = MemLoc::slot(slot, i64::from(off), load_ty.bytes());
                    (loc, RawLoad::Stack, AddrKey::Slot(slot), loc.offset)
                }
                _ => return None,
            };
            match key {
                None => *key = Some(addr_key),
                Some(expected) if *expected != addr_key => return None,
                _ => {}
            }
            touched.push(cur);
            leaves.push(Leaf { blk_idx, pos: lpos, loc, canon_off, raw, shift_byte, bytes: load_bytes });
            Some(())
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

fn build_pack(
    func: &Function,
    cfg: &Cfg,
    prov: &Provenance,
    consts: &[Option<i64>],
    def_info: &[DefInfo],
    chain: &[Block],
    (root_pos, root_val, res_ty): (usize, Value, Type),
) -> Option<PackPlan> {
    let mut leaves = Vec::new();
    let mut touched = Vec::new();
    let mut addr_key = None;
    collect_leaves(func, prov, consts, def_info, chain, root_val, (&mut leaves, &mut touched, &mut addr_key))?;
    if leaves.len() < 2 || leaves.len() > 8 {
        return None;
    }
    leaves.sort_by_key(|l| l.canon_off);
    let min_off = leaves[0].canon_off;
    let total_w: i64 = leaves.iter().map(|l| l.bytes).sum();
    if !matches!(total_w, 2 | 4 | 8) || total_w > i64::from(res_ty.bytes()) {
        return None;
    }
    let mut cursor = min_off;
    for l in &leaves {
        if l.canon_off != cursor || l.shift_byte + l.bytes > total_w {
            return None;
        }
        cursor += l.bytes;
    }
    let little = leaves.iter().all(|l| l.shift_byte == l.canon_off - min_off);
    let big = !little && leaves.iter().all(|l| l.shift_byte == total_w - (l.canon_off - min_off) - l.bytes);
    if !little && !big {
        return None;
    }
    let wide_ty = match total_w {
        2 => Type::I16,
        4 => Type::I32,
        _ => Type::I64,
    };
    let root_blk = *chain.last()?;
    let first = leaves[0];
    let wide_op = match (first.loc.base, first.raw) {
        (MemBase::Slot(slot), _) => Op::StackLoad(slot, i32::try_from(first.loc.offset).ok()?),
        (_, RawLoad::Addr(addr, off)) if fits_mem_imm(wide_ty, i64::from(off)) => Op::Load(addr, off),
        (MemBase::Addr(base), _)
            if def_info.get(base.index()).copied().flatten().is_none_or(|(db, dp, _)| {
                cfg.dominates(db, root_blk) && (db != root_blk || dp < root_pos)
            }) =>
        {
            let off_i32 = i32::try_from(first.loc.offset).ok()?;
            if !fits_mem_imm(wide_ty, first.loc.offset) {
                return None;
            }
            Op::Load(base, off_i32)
        }
        _ => return None,
    };
    let wide_loc = MemLoc::new(first.loc.root, first.loc.base, first.loc.offset, total_w as u32);
    if has_clobber(func, prov, chain, &leaves, root_pos, wide_loc) {
        return None;
    }
    Some(PackPlan { wide_op, wide_ty, big_endian: big, touched })
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

pub(super) fn coalesce_loads(func: &mut Function, prov: &Provenance) -> bool {
    let cfg = func.cfg();
    let mut consts = vec![None; func.types.len()];
    let mut def_info: Vec<DefInfo> = vec![None; func.types.len()];
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
    let mut uses = func.use_counts();
    let mut dead_pool = vec![false; func.types.len()];
    let mut touched_blk = vec![false; func.blocks.len()];
    let mut changed = false;
    for &blk in &cfg.rpo {
        let b_idx = blk.index();
        let chain = pred_chain(&cfg, blk);
        let insts = &func.blocks[b_idx].insts;
        let mut consumed = vec![false; func.types.len()];
        let mut plans: Vec<Option<(Value, Type, PackPlan)>> = (0..insts.len()).map(|_| None).collect();
        let mut any_plan = false;
        for (pos, &inst) in insts.iter().enumerate().rev() {
            let data = &func.insts[inst.index()];
            if let (Op::Binary(Binary::Bor | Binary::Iadd, ..), Some(res)) = (&data.op, data.result)
                && !consumed[res.index()]
            {
                let res_ty = func.ty(res);
                if matches!(res_ty, Type::I16 | Type::I32 | Type::I64)
                    && let Some(plan) = build_pack(func, &cfg, prov, &consts, &def_info, &chain, (pos, res, res_ty))
                {
                    for &v in &plan.touched {
                        if v != res {
                            consumed[v.index()] = true;
                        }
                    }
                    plans[pos] = Some((res, res_ty, plan));
                    any_plan = true;
                }
            }
        }
        if !any_plan {
            continue;
        }
        let insts = std::mem::take(&mut func.blocks[b_idx].insts);
        let mut out = Vec::with_capacity(insts.len());
        for (pos, &inst) in insts.iter().enumerate() {
            if let Some((res, res_ty, plan)) = plans[pos].take() {
                func.insts[inst.index()].op.for_operands(|v| uses[v.index()] = uses[v.index()].saturating_sub(1));
                for v in plan.touched {
                    if v != res {
                        dead_pool[v.index()] = true;
                        if let Some((db, _, _)) = def_info[v.index()] {
                            touched_blk[db.index()] = true;
                        }
                    }
                }
                if !plan.big_endian && plan.wide_ty == res_ty {
                    func.insts[inst.index()].op = plan.wide_op;
                } else {
                    let wide_val = func.value(plan.wide_ty);
                    uses.push(1);
                    def_info.push(None);
                    dead_pool.push(false);
                    consts.push(None);
                    out.push(func.add(plan.wide_op, Some(wide_val)));
                    let swapped = if plan.big_endian && plan.wide_ty == res_ty {
                        Op::Unary(Unary::Bswap, wide_val)
                    } else if plan.big_endian {
                        let bswapped = func.value(plan.wide_ty);
                        uses.push(1);
                        def_info.push(None);
                        dead_pool.push(false);
                        consts.push(None);
                        out.push(func.add(Op::Unary(Unary::Bswap, wide_val), Some(bswapped)));
                        Op::Convert(Convert::Uextend, bswapped)
                    } else {
                        Op::Convert(Convert::Uextend, wide_val)
                    };
                    func.insts[inst.index()].op = swapped;
                }
                changed = true;
            }
            out.push(inst);
        }
        func.blocks[b_idx].insts = out;
        touched_blk[b_idx] = true;
        sweep_dead(func, b_idx, &dead_pool, &mut uses, &mut def_info);
    }
    if changed {
        for &blk in cfg.rpo.iter().rev() {
            if touched_blk[blk.index()] {
                sweep_dead(func, blk.index(), &dead_pool, &mut uses, &mut def_info);
            }
        }
    }
    changed
}
