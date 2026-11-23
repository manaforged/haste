use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::fold::defs;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, Type, Unary, Value};
use crate::live::results;

mod addr;
mod reuse;

struct DistMatch {
    op: Binary,
    common: Value,
    x: Value,
    y: Value,
    kill: [Inst; 2],
}

struct ChMatch {
    e: Value,
    f: Value,
    g: Value,
    kill: [Inst; 3],
}

struct MajMatch {
    op: Binary,
    best: Value,
    u: Value,
    v: Value,
    common_old: Value,
    kill: [Inst; 3],
}

struct FactorState {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
    depth: Vec<u32>,
    inst_block: Vec<usize>,
    dead: Vec<bool>,
    changed: bool,
}

enum LoadGroup {
    One(LoadItem),
    Pair(LoadItem, LoadItem),
}

#[derive(Clone, Copy)]
struct LoadItem {
    pos: usize,
    inst: Inst,
    ptr: Value,
    off: i32,
    size: i32,
    first_use: usize,
}

impl LoadGroup {
    fn first_def(&self) -> usize {
        match self {
            Self::One(a) => a.pos,
            Self::Pair(a, b) => a.pos.min(b.pos),
        }
    }

    fn last_def(&self) -> usize {
        match self {
            Self::One(a) => a.pos,
            Self::Pair(a, b) => a.pos.max(b.pos),
        }
    }

    fn first_use(&self) -> usize {
        match self {
            Self::One(a) => a.first_use,
            Self::Pair(a, b) => a.first_use.min(b.first_use),
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Pair(..) => 2,
        }
    }

    fn push_insts(&self, out: &mut Vec<(Inst, usize)>) {
        match *self {
            Self::One(a) => out.push((a.inst, a.first_use)),
            Self::Pair(a, b) => {
                let u = a.first_use.min(b.first_use);
                out.push((a.inst, u));
                out.push((b.inst, u));
            }
        }
    }
}

fn count_operand_uses(func: &Function, inst: Inst, val: Value) -> u32 {
    let mut count = 0u32;
    func.insts[inst.index()].op.for_operands(|v| count += u32::from(v == val));
    count
}

fn uses_operand(func: &Function, inst: Inst, val: Value) -> bool {
    count_operand_uses(func, inst, val) > 0
}

fn first_use_in_block(func: &Function, insts: &[Inst], pos: usize, val: Value) -> Option<usize> {
    insts[pos + 1..].iter().position(|&i| uses_operand(func, i, val)).map(|d| pos + 1 + d)
}

fn band_operands(func: &Function, defs: &[Option<Inst>], uses: &[u32], val: Value) -> Option<(Inst, Value, Value)> {
    let inst = defs.get(val.index()).copied().flatten()?;
    if uses.get(val.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let Op::Binary(Binary::Band, a, b) = func.insts[inst.index()].op else { return None };
    Some((inst, a, b))
}

fn match_distributive(func: &Function, defs: &[Option<Inst>], uses: &[u32], op: &Op) -> Option<DistMatch> {
    let &Op::Binary(outer @ (Binary::Bxor | Binary::Bor), lhs, rhs) = op else { return None };
    let (li, l0, l1) = band_operands(func, defs, uses, lhs)?;
    let (ri, r0, r1) = band_operands(func, defs, uses, rhs)?;
    let (common, x, y) = if l0 == r0 {
        (l0, l1, r1)
    } else if l0 == r1 {
        (l0, l1, r0)
    } else if l1 == r0 {
        (l1, l0, r1)
    } else if l1 == r1 {
        (l1, l0, r0)
    } else {
        return None;
    };
    Some(DistMatch { op: outer, common, x, y, kill: [li, ri] })
}

fn find_bnot(func: &Function, defs: &[Option<Inst>], uses: &[u32], val: Value) -> Option<(Inst, Value)> {
    if uses.get(val.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let inst = defs.get(val.index()).copied().flatten()?;
    let Op::Unary(Unary::Bnot, inner) = func.insts[inst.index()].op else { return None };
    Some((inst, inner))
}

fn match_ch(func: &Function, defs: &[Option<Inst>], uses: &[u32], op: &Op) -> Option<ChMatch> {
    let &Op::Binary(Binary::Bxor, lhs, rhs) = op else { return None };
    let (li, l0, l1) = band_operands(func, defs, uses, lhs)?;
    let (ri, r0, r1) = band_operands(func, defs, uses, rhs)?;
    for ((a0, a1), (b0, b1)) in [((l0, l1), (r0, r1)), ((r0, r1), (l0, l1))] {
        for (e, f) in [(a0, a1), (a1, a0)] {
            for (not_c, g) in [(b0, b1), (b1, b0)] {
                if let Some((ni, inner)) = find_bnot(func, defs, uses, not_c)
                    && inner == e
                {
                    return Some(ChMatch { e, f, g, kill: [li, ri, ni] });
                }
            }
        }
    }
    None
}

fn inst_latency(op: &Op) -> u32 {
    match op {
        Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::StackAddr(..) | Op::FramePointer => 0,
        Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) | Op::Fma(..) | Op::Call { .. } | Op::TryCall { .. } => 3,
        Op::Binary(bin, ..) if matches!(bin, Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem | Binary::Fmul | Binary::Fdiv) => 3,
        _ => 1,
    }
}

fn fill_block_depths(func: &Function, bi: usize, depth: &mut Vec<u32>) {
    depth.clear();
    depth.resize(func.types.len(), 0);
    for &inst in &func.blocks[bi].insts {
        let data = &func.insts[inst.index()];
        let lat = inst_latency(&data.op);
        if lat == 0 { continue; }
        let mut mx = 0u32;
        data.op.for_operands(|v| mx = mx.max(depth[v.index()]));
        for r in results(data) { depth[r.index()] = mx + lat; }
    }
}

fn same_pair((a, b): (Value, Value), (c, d): (Value, Value)) -> bool {
    (a == c && b == d) || (a == d && b == c)
}

fn match_maj_side(func: &Function, fs: &FactorState, bi: usize, (outer, fac_side, pair_side): (Binary, Value, Value)) -> Option<MajMatch> {
    let (pair_i, p0, p1) = band_operands(func, &fs.defs, &fs.uses, pair_side)?;
    let (fac_i, f0, f1) = band_operands(func, &fs.defs, &fs.uses, fac_side)?;
    if p0 == p1 || fs.inst_block.get(pair_i.index()).copied() != Some(bi) || fs.inst_block.get(fac_i.index()).copied() != Some(bi) { return None; }
    for (common_old, inner) in [(f0, f1), (f1, f0)] {
        if common_old == p0 || common_old == p1 || fs.uses.get(inner.index()).copied().unwrap_or(0) != 1 { continue; }
        let Some(inner_i) = fs.defs.get(inner.index()).copied().flatten() else { continue };
        if fs.inst_block.get(inner_i.index()).copied() != Some(bi) { continue; }
        let Op::Binary(inner_op, x0, x1) = func.insts[inner_i.index()].op else { continue };
        if inner_op != outer || !same_pair((x0, x1), (p0, p1)) { continue; }
        let (best, other) = if fs.depth[p0.index()] >= fs.depth[p1.index()] { (p0, p1) } else { (p1, p0) };
        if fs.depth[best.index()] > fs.depth[common_old.index()] {
            return Some(MajMatch { op: outer, best, u: common_old, v: other, common_old, kill: [inner_i, fac_i, pair_i] });
        }
    }
    None
}

fn match_maj_refactor(func: &Function, fs: &FactorState, bi: usize, op: &Op) -> Option<MajMatch> {
    let &Op::Binary(outer @ (Binary::Bxor | Binary::Bor), lhs, rhs) = op else { return None };
    match_maj_side(func, fs, bi, (outer, lhs, rhs)).or_else(|| match_maj_side(func, fs, bi, (outer, rhs, lhs)))
}

fn add_use(uses: &mut Vec<u32>, val: Value, delta: i32) {
    if uses.len() <= val.index() {
        uses.resize(val.index() + 1, 0);
    }
    let next = (uses[val.index()] as i64 + i64::from(delta)).max(0);
    uses[val.index()] = next as u32;
}

fn set_bin_depth(depth: &mut [u32], val: Value, a: Value, b: Value) {
    if val.index() < depth.len() && a.index() < depth.len() && b.index() < depth.len() {
        depth[val.index()] = depth[a.index()].max(depth[b.index()]) + 1;
    }
}

fn record_factor_def(fs: &mut FactorState, func: &Function, bi: usize, (inst, val): (Inst, Value)) {
    fs.defs.resize(func.types.len(), None);
    fs.inst_block.resize(func.insts.len(), usize::MAX);
    fs.defs[val.index()] = Some(inst);
    fs.inst_block[inst.index()] = bi;
}

fn apply_distributive(func: &mut Function, block: Block, at: usize, m: DistMatch, fs: &mut FactorState) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let res = func.insts[inst.index()].result;
    let ty = func.ty(res.unwrap_or(m.common));
    let t_val = func.value(ty);
    let t_inst = func.add(Op::Binary(m.op, m.x, m.y), Some(t_val));
    func.insts[inst.index()].op = Op::Binary(Binary::Band, m.common, t_val);
    func.blocks[block.index()].insts.insert(at, t_inst);
    record_factor_def(fs, func, block.index(), (t_inst, t_val));
    if !fs.depth.is_empty() {
        fs.depth.resize(func.types.len(), 0);
        set_bin_depth(&mut fs.depth, t_val, m.x, m.y);
        if let Some(r) = res { set_bin_depth(&mut fs.depth, r, m.common, t_val); }
    }
    add_use(&mut fs.uses, m.common, -1);
    add_use(&mut fs.uses, t_val, 1);
    for k in m.kill { fs.dead[k.index()] = true; }
    fs.changed = true;
    1
}

fn apply_ch(func: &mut Function, block: Block, at: usize, m: ChMatch, fs: &mut FactorState) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let res = func.insts[inst.index()].result;
    let ty = func.ty(res.unwrap_or(m.e));
    let (t_val, m_val) = (func.value(ty), func.value(ty));
    let t_inst = func.add(Op::Binary(Binary::Bxor, m.f, m.g), Some(t_val));
    let m_inst = func.add(Op::Binary(Binary::Band, m.e, t_val), Some(m_val));
    func.insts[inst.index()].op = Op::Binary(Binary::Bxor, m.g, m_val);
    func.blocks[block.index()].insts.splice(at..at, [t_inst, m_inst]);
    record_factor_def(fs, func, block.index(), (t_inst, t_val));
    record_factor_def(fs, func, block.index(), (m_inst, m_val));
    if !fs.depth.is_empty() {
        fs.depth.resize(func.types.len(), 0);
        set_bin_depth(&mut fs.depth, t_val, m.f, m.g);
        set_bin_depth(&mut fs.depth, m_val, m.e, t_val);
        if let Some(r) = res { set_bin_depth(&mut fs.depth, r, m.g, m_val); }
    }
    add_use(&mut fs.uses, m.e, -1);
    add_use(&mut fs.uses, m.g, 1);
    add_use(&mut fs.uses, t_val, 1);
    add_use(&mut fs.uses, m_val, 1);
    for k in m.kill { fs.dead[k.index()] = true; }
    fs.changed = true;
    2
}

fn apply_maj_refactor(func: &mut Function, block: Block, at: usize, m: MajMatch, fs: &mut FactorState) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let res = func.insts[inst.index()].result;
    let ty = func.ty(res.unwrap_or(m.best));
    let (t_val, m_val, k_val) = (func.value(ty), func.value(ty), func.value(ty));
    let t_inst = func.add(Op::Binary(m.op, m.u, m.v), Some(t_val));
    let m_inst = func.add(Op::Binary(Binary::Band, m.best, t_val), Some(m_val));
    let k_inst = func.add(Op::Binary(Binary::Band, m.u, m.v), Some(k_val));
    func.insts[inst.index()].op = Op::Binary(m.op, m_val, k_val);
    func.blocks[block.index()].insts.splice(at..at, [t_inst, m_inst, k_inst]);
    record_factor_def(fs, func, block.index(), (t_inst, t_val));
    record_factor_def(fs, func, block.index(), (m_inst, m_val));
    record_factor_def(fs, func, block.index(), (k_inst, k_val));
    if !fs.depth.is_empty() {
        fs.depth.resize(func.types.len(), 0);
        set_bin_depth(&mut fs.depth, t_val, m.u, m.v);
        set_bin_depth(&mut fs.depth, m_val, m.best, t_val);
        set_bin_depth(&mut fs.depth, k_val, m.u, m.v);
        if let Some(r) = res { set_bin_depth(&mut fs.depth, r, m_val, k_val); }
    }
    add_use(&mut fs.uses, m.common_old, 1);
    add_use(&mut fs.uses, m.best, -1);
    add_use(&mut fs.uses, t_val, 1);
    add_use(&mut fs.uses, m_val, 1);
    add_use(&mut fs.uses, k_val, 1);
    for k in m.kill { fs.dead[k.index()] = true; }
    fs.changed = true;
    3
}

fn factor_block(func: &mut Function, bi: usize, fs: &mut FactorState, balance: bool) {
    let block = Block(bi as u32);
    if balance { fill_block_depths(func, bi, &mut fs.depth); }
    let mut at = 0;
    while at < func.blocks[bi].insts.len() {
        let inst = func.blocks[bi].insts[at];
        if fs.dead.get(inst.index()).copied().unwrap_or(false) { at += 1; continue; }
        let op = func.insts[inst.index()].op.clone();
        if let Some(m) = match_ch(func, &fs.defs, &fs.uses, &op) {
            fs.dead.resize(func.insts.len(), false);
            at += apply_ch(func, block, at, m, fs) + 1;
        } else if let Some(m) = match_distributive(func, &fs.defs, &fs.uses, &op) {
            fs.dead.resize(func.insts.len(), false);
            at += apply_distributive(func, block, at, m, fs) + 1;
        } else if balance && let Some(m) = match_maj_refactor(func, fs, bi, &op) {
            fs.dead.resize(func.insts.len(), false);
            at += apply_maj_refactor(func, block, at, m, fs) + 1;
        } else { at += 1; }
    }
}

fn factor_bitwise(func: &mut Function, balance: bool) {
    let mut inst_block = vec![usize::MAX; func.insts.len()];
    if balance {
        for (bi, block) in func.blocks.iter().enumerate() {
            for &inst in &block.insts { inst_block[inst.index()] = bi; }
        }
    }
    let mut fs = FactorState { defs: defs(func), uses: func.use_counts(), depth: Vec::new(), inst_block, dead: vec![false; func.insts.len()], changed: false };
    for bi in 0..func.blocks.len() {
        factor_block(func, bi, &mut fs, balance);
    }
    if !fs.changed {
        return;
    }
    for block in &mut func.blocks {
        block.insts.retain(|i| {
            let kill = fs.dead.get(i.index()).copied().unwrap_or(false);
            if kill {
                func.insts[i.index()].op = Op::Trap;
            }
            !kill
        });
    }
}

fn has_mem_or_trap(op: &Op) -> bool {
    crate::mem::is_hard_barrier(op) || matches!(op, Op::Store(..) | Op::StackStore(..) | Op::Copy(..) | Op::Move(..))
}

fn barrier_prefix(func: &Function, insts: &[Inst]) -> Vec<u32> {
    let mut pre = Vec::with_capacity(insts.len() + 1);
    pre.push(0);
    let mut acc = 0u32;
    for &inst in insts {
        acc += u32::from(has_mem_or_trap(&func.insts[inst.index()].op));
        pre.push(acc);
    }
    pre
}

fn collect_loads(func: &Function, insts: &[Inst]) -> Vec<LoadItem> {
    let mut out = Vec::new();
    for (pos, &inst) in insts.iter().enumerate() {
        let data = &func.insts[inst.index()];
        let (Op::Load(ptr, off), Some(res)) = (data.op.clone(), data.result) else { continue };
        let Some(first_use) = first_use_in_block(func, insts, pos, res) else { continue };
        out.push(LoadItem { pos, inst, ptr, off, size: func.ty(res).bytes() as i32, first_use });
    }
    out
}

fn group_ptr_loads(mut items: Vec<LoadItem>) -> Vec<LoadGroup> {
    items.sort_by_key(|l| (l.off, l.pos));
    let mut groups = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let a = items[i];
        if i + 1 < items.len() {
            let b = items[i + 1];
            if a.size == b.size && matches!(a.size, 4 | 8) && a.off.checked_add(a.size) == Some(b.off) {
                groups.push(LoadGroup::Pair(a, b));
                i += 2;
                continue;
            }
        }
        groups.push(LoadGroup::One(a));
        i += 1;
    }
    groups
}

fn ptr_live_at_use(func: &Function, insts: &[Inst], (ptr, first_use): (Value, usize), uses: &[u32], moving: &[bool]) -> bool {
    let block_uses: u32 = insts.iter().map(|&i| count_operand_uses(func, i, ptr)).sum();
    if uses.get(ptr.index()).copied().unwrap_or(0) > block_uses {
        return true;
    }
    insts[first_use..].iter().any(|&i| !moving[i.index()] && uses_operand(func, i, ptr))
}

fn select_ptr_moves(
    func: &Function,
    insts: &[Inst],
    barriers: &[u32],
    uses: &[u32],
    items: Vec<LoadItem>,
) -> Vec<(Inst, usize)> {
    let ptr = items[0].ptr;
    let groups = group_ptr_loads(items);
    let eligible: Vec<LoadGroup> = groups
        .into_iter()
        .filter(|g| {
            let fu = g.first_use();
            fu > g.last_def() + 16 && barriers[fu] - barriers[g.first_def() + 1] == 0
        })
        .collect();
    let cand_count: usize = eligible.iter().map(LoadGroup::count).sum();
    let mut moving = vec![false; func.insts.len()];
    let mut tmp = Vec::new();
    for g in &eligible {
        g.push_insts(&mut tmp);
    }
    for &(inst, _) in &tmp {
        moving[inst.index()] = true;
    }
    let mut out = Vec::new();
    for g in eligible {
        if cand_count >= 3 || ptr_live_at_use(func, insts, (ptr, g.first_use()), uses, &moving) {
            g.push_insts(&mut out);
        }
    }
    out
}

fn apply_load_moves(func: &mut Function, bi: usize, moves: Vec<(Inst, usize)>) {
    if moves.is_empty() {
        return;
    }
    let orig = func.blocks[bi].insts.clone();
    let mut moving = vec![false; func.insts.len()];
    for &(inst, _) in &moves {
        moving[inst.index()] = true;
    }
    let mut emitted = vec![false; func.insts.len()];
    let mut before: Vec<Vec<Inst>> = vec![Vec::new(); func.insts.len()];
    for (inst, first_use) in moves {
        let anchor = orig[first_use];
        if !moving[anchor.index()] {
            emitted[inst.index()] = true;
            before[anchor.index()].push(inst);
        }
    }
    let mut rebuilt = Vec::with_capacity(orig.len());
    for inst in orig {
        if emitted[inst.index()] {
            continue;
        }
        rebuilt.extend(before[inst.index()].iter().copied());
        rebuilt.push(inst);
    }
    func.blocks[bi].insts = rebuilt;
}

fn sink_loads_block(func: &mut Function, uses: &[u32], bi: usize) {
    let insts = func.blocks[bi].insts.clone();
    if insts.len() < 48 {
        return;
    }
    let loads = collect_loads(func, &insts);
    if loads.len() < 4 {
        return;
    }
    let barriers = barrier_prefix(func, &insts);
    let mut by_ptr: Vec<(Value, Vec<LoadItem>)> = Vec::new();
    for item in loads {
        if let Some((_, list)) = by_ptr.iter_mut().find(|(p, _)| *p == item.ptr) {
            list.push(item);
        } else {
            by_ptr.push((item.ptr, vec![item]));
        }
    }
    let mut moves = Vec::new();
    for (_, list) in by_ptr {
        if list.len() >= 4 {
            moves.extend(select_ptr_moves(func, &insts, &barriers, uses, list));
        }
    }
    apply_load_moves(func, bi, moves);
}

fn is_pure_sinkable(func: &Function, op: &Op, res: Value) -> bool {
    let ty = func.ty(res);
    if ty.is_vector() {
        return false;
    }
    match *op {
        Op::Binary(bin, ..) => !matches!(bin, Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem),
        Op::Unary(..) => true,
        Op::Convert(kind, x) => {
            !ty.is_float()
                && !func.ty(x).is_float()
                && !func.ty(x).is_vector()
                && matches!(kind, Convert::Uextend | Convert::Sextend | Convert::Ireduce | Convert::Bitcast)
        }
        _ => false,
    }
}

fn is_static_def(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    let Some(inst) = defs.get(val.index()).copied().flatten() else { return false };
    matches!(func.insts[inst.index()].op, Op::Const(_) | Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_))
}

fn operands_live_at(func: &Function, defs: &[Option<Inst>], tail: &[Inst], op: &Op) -> bool {
    let mut ok = true;
    op.for_operands(|v| {
        if !is_static_def(func, defs, v) {
            ok &= tail.iter().any(|&i| uses_operand(func, i, v));
        }
    });
    ok
}

fn sink_pure_block(func: &mut Function, defs: &[Option<Inst>], uses: &[u32], bi: usize) {
    let len = func.blocks[bi].insts.len();
    if len < 32 {
        return;
    }
    for at in (0..len.saturating_sub(1)).rev() {
        let inst = func.blocks[bi].insts[at];
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if uses.get(res.index()).copied().unwrap_or(0) != 1 || !is_pure_sinkable(func, &data.op, res) {
            continue;
        }
        let Some(u_idx) = first_use_in_block(func, &func.blocks[bi].insts, at, res) else { continue };
        if u_idx <= at + 4 || func.insts[func.blocks[bi].insts[u_idx].index()].op.is_terminator() {
            continue;
        }
        let op = data.op.clone();
        if !operands_live_at(func, defs, &func.blocks[bi].insts[u_idx..], &op) {
            continue;
        }
        let moved = func.blocks[bi].insts.remove(at);
        func.blocks[bi].insts.insert(u_idx - 1, moved);
    }
}

fn is_fits_paired(func: &Function, a: Inst, b: Inst) -> bool {
    let (f, s) = (&func.insts[a.index()], &func.insts[b.index()]);
    matches!((&f.op, &s.op, f.result), (Op::Binary(op, x, y), Op::Fits(chk, _, l, r), Some(sum)) if op == chk && (x, y) == (l, r) && func.ty(sum).is_small_int())
}

fn max_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) { (Some(x), Some(y)) => Some(x.max(y)), (Some(x), None) | (None, Some(x)) => Some(x), (None, None) => None }
}

fn small_iadd(func: &Function, inst: Inst, pinned: bool) -> Option<(Value, Value, Value)> {
    if pinned { return None; }
    let data = &func.insts[inst.index()];
    let (Op::Binary(Binary::Iadd, a, b), Some(res)) = (data.op.clone(), data.result) else { return None };
    func.ty(res).is_small_int().then_some((a, b, res))
}

fn add_scan(func: &Function, insts: &[Inst], uses: &[u32]) -> (Vec<bool>, Vec<Option<usize>>, Vec<usize>) {
    let n = insts.len();
    let pinned: Vec<bool> = (0..n).map(|p| p + 1 < n && is_fits_paired(func, insts[p], insts[p + 1])).collect();
    let mut def_pos = vec![None; func.types.len()];
    for (pos, &inst) in insts.iter().enumerate() {
        let p = if pinned[pos] { pos + 1 } else { pos };
        for r in results(&func.insts[inst.index()]) { def_pos[r.index()] = Some(p); }
    }
    let mut has_parent = vec![false; n];
    for (pos, &inst) in insts.iter().enumerate() {
        let Some((a, b, res)) = small_iadd(func, inst, pinned[pos]) else { continue };
        if a == b { continue; }
        for c in [a, b] {
            if let Some(cp) = def_pos[c.index()] && uses[c.index()] == 1 && func.ty(c) == func.ty(res)
                && small_iadd(func, insts[cp], pinned[cp]).is_some() { has_parent[cp] = true; }
        }
    }
    let roots = (0..n).filter(|&p| !has_parent[p] && small_iadd(func, insts[p], pinned[p]).is_some()).collect();
    (pinned, def_pos, roots)
}

fn flatten_add(func: &Function, (insts, def_pos, pinned, uses): (&[Inst], &[Option<usize>], &[bool], &[u32]), root_pos: usize) -> (Vec<Value>, Vec<Inst>) {
    let Some((a, b, root_res)) = small_iadd(func, insts[root_pos], pinned[root_pos]) else { return (Vec::new(), Vec::new()) };
    let ty = func.ty(root_res);
    let (mut leaves, mut interiors, mut stack) = (Vec::new(), Vec::new(), vec![b, a]);
    while let Some(v) = stack.pop() {
        if let Some(cp) = def_pos[v.index()] && uses[v.index()] == 1 && func.ty(v) == ty
            && let Some((x, y, _)) = small_iadd(func, insts[cp], pinned[cp]) && x != y
        {
            interiors.push(insts[cp]);
            stack.push(y);
            stack.push(x);
        } else { leaves.push(v); }
    }
    (leaves, interiors)
}

fn pop_pair(heap: &mut BinaryHeap<Reverse<(u32, usize, Value)>>, two_use: bool) -> ((u32, usize, Value), (u32, usize, Value)) {
    if two_use && heap.len() == 3 {
        let (Some(Reverse(e0)), Some(Reverse(e1)), Some(Reverse(e2))) = (heap.pop(), heap.pop(), heap.pop()) else {
            return ((0, 0, Value(0)), (0, 0, Value(0)));
        };
        if e0.0 + 1 < e1.0 && e1.0 == e2.0 {
            heap.push(Reverse(e0));
            return (e1, e2);
        }
        heap.push(Reverse(e2));
        return (e0, e1);
    }
    let (Some(Reverse(a)), Some(Reverse(b))) = (heap.pop(), heap.pop()) else { return ((0, 0, Value(0)), (0, 0, Value(0))); };
    (a, b)
}

fn plan_add_tree(
    func: &Function,
    (leaves, interiors, root_res, two_use): (&[Value], &[Inst], Value, bool),
    (def_pos, depth): (&[Option<usize>], &[u32]),
) -> Option<(Vec<(Inst, Value, Value, Option<usize>)>, Value, Value, u32)> {
    if leaves.len() < 3 || interiors.len() + 2 != leaves.len() { return None; }
    let mut heap: BinaryHeap<Reverse<(u32, usize, Value)>> = leaves.iter().enumerate().map(|(i, &v)| Reverse((depth[v.index()], i, v))).collect();
    let mut local_pos: Vec<(Value, Option<usize>)> = Vec::with_capacity(interiors.len());
    let get_pos = |v: Value, lp: &[(Value, Option<usize>)]| lp.iter().find(|&&(x, _)| x == v).map_or_else(|| def_pos[v.index()], |&(_, p)| p);
    let mut ops = Vec::with_capacity(interiors.len());
    let mut sum_d = 0u32;
    for &inst in interiors {
        let val = func.insts[inst.index()].result?;
        let ((da, ra, va), (db, rb, vb)) = pop_pair(&mut heap, two_use);
        let d_new = da.max(db) + 1;
        let p_new = max_opt(get_pos(va, &local_pos), get_pos(vb, &local_pos));
        ops.push((inst, va, vb, p_new));
        local_pos.push((val, p_new));
        sum_d = sum_d.saturating_add(d_new);
        heap.push(Reverse((d_new, ra.min(rb), val)));
    }
    let ((da, _, va), (db, _, vb)) = pop_pair(&mut heap, false);
    let new_root_d = da.max(db) + 1;
    let old_root_d = depth[root_res.index()];
    let old_int_sum: u32 = interiors.iter().filter_map(|&i| func.insts[i.index()].result).map(|v| depth[v.index()]).sum();
    if new_root_d > old_root_d || (new_root_d == old_root_d && sum_d >= old_int_sum) { return None; }
    Some((ops, va, vb, new_root_d))
}

fn balance_add_trees_block(func: &mut Function, uses: &[u32], depth: &mut Vec<u32>, bi: usize) {
    let orig = func.blocks[bi].insts.clone();
    if orig.len() < 8 { return; }
    fill_block_depths(func, bi, depth);
    let (pinned, def_pos, roots) = add_scan(func, &orig, uses);
    let (mut killed, mut at_start) = (vec![false; func.insts.len()], Vec::new());
    let mut after: Vec<Vec<Inst>> = vec![Vec::new(); orig.len()];
    let mut changed = false;
    for rp in roots {
        let root_inst = orig[rp];
        let Some(root_res) = func.insts[root_inst.index()].result else { continue };
        let (leaves, interiors) = flatten_add(func, (&orig, &def_pos, &pinned, uses), rp);
        let two_use = uses[root_res.index()] == 2;
        let Some((ops, ra, rb, _)) = plan_add_tree(func, (&leaves, &interiors, root_res, two_use), (&def_pos, depth)) else { continue };
        for &inst in &interiors { killed[inst.index()] = true; }
        for (inst, va, vb, p_new) in ops {
            func.insts[inst.index()].op = Op::Binary(Binary::Iadd, va, vb);
            if let Some(p) = p_new { after[p].push(inst); } else { at_start.push(inst); }
        }
        func.insts[root_inst.index()].op = Op::Binary(Binary::Iadd, ra, rb);
        changed = true;
    }
    if !changed { return; }
    let mut rebuilt = Vec::with_capacity(orig.len());
    rebuilt.extend(at_start);
    for (pos, &inst) in orig.iter().enumerate() {
        if !killed[inst.index()] { rebuilt.push(inst); }
        rebuilt.extend(after[pos].iter().copied());
    }
    func.blocks[bi].insts = rebuilt;
}

fn block_peak_gpr(func: &Function, insts: &[Inst], total_uses: &[u32]) -> usize {
    let mut in_bi = vec![0u32; func.types.len()];
    let mut tracked = vec![false; func.types.len()];
    for &inst in insts {
        let data = &func.insts[inst.index()];
        let is_stat = matches!(data.op, Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::StackAddr(..) | Op::FramePointer);
        for r in results(data) { tracked[r.index()] = func.ty(r).is_small_int() && !is_stat; }
        data.op.for_operands(|v| in_bi[v.index()] += 1);
    }
    let used_other: Vec<bool> = (0..in_bi.len()).map(|i| total_uses[i] > in_bi[i]).collect();
    let (mut live, mut peak) = (0usize, 0usize);
    for &inst in insts {
        let data = &func.insts[inst.index()];
        data.op.for_operands(|v| {
            in_bi[v.index()] -= 1;
            if tracked[v.index()] && in_bi[v.index()] == 0 && !used_other[v.index()] { live = live.saturating_sub(1); }
        });
        for r in results(data) {
            if tracked[r.index()] && (in_bi[r.index()] > 0 || used_other[r.index()]) { live += 1; peak = peak.max(live); }
        }
    }
    peak
}

fn find_two_use_bypass(func: &Function, (orig, def_pos, pinned): (&[Inst], &[Option<usize>], &[bool]), (uses, depth): (&[u32], &[u32]), (lhs, rhs, ty): (Value, Value, Type)) -> Option<(Value, Value, Value, Value)> {
    for (t1, d) in [(lhs, rhs), (rhs, lhs)] {
        if uses[t1.index()] != 2 || t1 == d { continue; }
        let Some(t1_pos) = def_pos[t1.index()] else { continue };
        let Some((x, y, _)) = small_iadd(func, orig[t1_pos], pinned[t1_pos]).filter(|_| func.ty(t1) == ty) else { continue };
        let (late, early) = if depth[x.index()] >= depth[y.index()] { (x, y) } else { (y, x) };
        if depth[late.index()] >= depth[early.index()] + 3 && depth[late.index()] >= depth[d.index()] + 2 {
            return Some((t1, d, late, early));
        }
    }
    None
}

fn bypass_two_use_adds_block(func: &mut Function, uses: &mut Vec<u32>, depth: &mut Vec<u32>, bi: usize) {
    let orig = func.blocks[bi].insts.clone();
    if orig.len() < 16 || block_peak_gpr(func, &orig, uses) >= 22 { return; }
    fill_block_depths(func, bi, depth);
    let (pinned, mut def_pos, _) = add_scan(func, &orig, uses);
    let (mut at_start, mut after, mut changed) = (Vec::new(), vec![Vec::new(); orig.len()], false);
    for (pos, &e_inst) in orig.iter().enumerate() {
        let Some((lhs, rhs, e_res)) = small_iadd(func, e_inst, pinned[pos]) else { continue };
        let ty = func.ty(e_res);
        let Some((t1, d, late, early)) = find_two_use_bypass(func, (&orig, &def_pos, &pinned), (uses, depth), (lhs, rhs, ty)) else { continue };
        let d_sum = func.value(ty);
        let sum_inst = func.add(Op::Binary(Binary::Iadd, early, d), Some(d_sum));
        let p_new = max_opt(def_pos[early.index()], def_pos[d.index()]);
        if let Some(p) = p_new { after[p].push(sum_inst); } else { at_start.push(sum_inst); }
        func.insts[e_inst.index()].op = Op::Binary(Binary::Iadd, late, d_sum);
        def_pos.resize(func.types.len(), None);
        def_pos[d_sum.index()] = p_new;
        depth.resize(func.types.len(), 0);
        set_bin_depth(depth, d_sum, early, d);
        set_bin_depth(depth, e_res, late, d_sum);
        add_use(uses, t1, -1);
        add_use(uses, early, 1);
        add_use(uses, late, 1);
        add_use(uses, d_sum, 1);
        changed = true;
    }
    if !changed { return; }
    let mut rebuilt = Vec::with_capacity(orig.len() + at_start.len());
    rebuilt.extend(at_start);
    for (pos, inst) in orig.into_iter().enumerate() { rebuilt.push(inst); rebuilt.extend(after[pos].iter().copied()); }
    func.blocks[bi].insts = rebuilt;
}

/// Factors single-use bitwise identities, balances addition trees by latency, sinks deferred loads and pure scalar ops, and splits shared address sums.
pub(crate) fn run(func: &mut Function, balance: bool) {
    factor_bitwise(func, balance);
    let mut uses = func.use_counts();
    if balance {
        let mut depth = Vec::new();
        for bi in 0..func.blocks.len() { balance_add_trees_block(func, &uses, &mut depth, bi); }
        for bi in 0..func.blocks.len() { bypass_two_use_adds_block(func, &mut uses, &mut depth, bi); }
    }
    let uses = if balance { func.use_counts() } else { uses };
    for bi in 0..func.blocks.len() {
        sink_loads_block(func, &uses, bi);
    }
    let live_defs = defs(func);
    for bi in 0..func.blocks.len() {
        sink_pure_block(func, &live_defs, &uses, bi);
    }
    reuse::after_last_read(func, &uses, &live_defs);
    addr::split_address_sums(func);
}
