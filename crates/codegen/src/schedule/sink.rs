use crate::fold::defs;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, Unary, Value};

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

struct FactorState {
    defs: Vec<Option<Inst>>,
    uses: Vec<u32>,
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

fn add_use(uses: &mut Vec<u32>, val: Value, delta: i32) {
    if uses.len() <= val.index() {
        uses.resize(val.index() + 1, 0);
    }
    let next = (uses[val.index()] as i64 + i64::from(delta)).max(0);
    uses[val.index()] = next as u32;
}

fn apply_distributive(func: &mut Function, block: Block, at: usize, m: DistMatch, fs: &mut FactorState) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let ty = func.ty(func.insts[inst.index()].result.unwrap_or(m.common));
    let t_val = func.value(ty);
    let t_inst = func.add(Op::Binary(m.op, m.x, m.y), Some(t_val));
    func.insts[inst.index()].op = Op::Binary(Binary::Band, m.common, t_val);
    func.blocks[block.index()].insts.insert(at, t_inst);
    fs.defs.resize(func.types.len(), None);
    fs.defs[t_val.index()] = Some(t_inst);
    add_use(&mut fs.uses, m.common, -1);
    add_use(&mut fs.uses, t_val, 1);
    for k in m.kill {
        fs.dead[k.index()] = true;
    }
    fs.changed = true;
    1
}

fn apply_ch(func: &mut Function, block: Block, at: usize, m: ChMatch, fs: &mut FactorState) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let ty = func.ty(func.insts[inst.index()].result.unwrap_or(m.e));
    let (t_val, m_val) = (func.value(ty), func.value(ty));
    let t_inst = func.add(Op::Binary(Binary::Bxor, m.f, m.g), Some(t_val));
    let m_inst = func.add(Op::Binary(Binary::Band, m.e, t_val), Some(m_val));
    func.insts[inst.index()].op = Op::Binary(Binary::Bxor, m.g, m_val);
    func.blocks[block.index()].insts.splice(at..at, [t_inst, m_inst]);
    fs.defs.resize(func.types.len(), None);
    fs.defs[t_val.index()] = Some(t_inst);
    fs.defs[m_val.index()] = Some(m_inst);
    add_use(&mut fs.uses, m.e, -1);
    add_use(&mut fs.uses, m.g, 1);
    add_use(&mut fs.uses, t_val, 1);
    add_use(&mut fs.uses, m_val, 1);
    for k in m.kill {
        fs.dead[k.index()] = true;
    }
    fs.changed = true;
    2
}

fn factor_block(func: &mut Function, bi: usize, fs: &mut FactorState) {
    let block = Block(bi as u32);
    let mut at = 0;
    while at < func.blocks[bi].insts.len() {
        let inst = func.blocks[bi].insts[at];
        if fs.dead.get(inst.index()).copied().unwrap_or(false) {
            at += 1;
            continue;
        }
        let op = func.insts[inst.index()].op.clone();
        if let Some(m) = match_ch(func, &fs.defs, &fs.uses, &op) {
            fs.dead.resize(func.insts.len(), false);
            at += apply_ch(func, block, at, m, fs) + 1;
        } else if let Some(m) = match_distributive(func, &fs.defs, &fs.uses, &op) {
            fs.dead.resize(func.insts.len(), false);
            at += apply_distributive(func, block, at, m, fs) + 1;
        } else {
            at += 1;
        }
    }
}

fn factor_bitwise(func: &mut Function) {
    let mut fs = FactorState {
        defs: defs(func),
        uses: func.use_counts(),
        dead: vec![false; func.insts.len()],
        changed: false,
    };
    for bi in 0..func.blocks.len() {
        factor_block(func, bi, &mut fs);
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
    matches!(
        op,
        Op::Store(..)
            | Op::StackStore(..)
            | Op::Copy(..)
            | Op::Move(..)
            | Op::Call { .. }
            | Op::TryCall { .. }
            | Op::AtomicLoad(..)
            | Op::AtomicStore(..)
            | Op::AtomicRmw(..)
            | Op::AtomicCas(..)
            | Op::Fence
            | Op::Trap
            | Op::Trapz(..)
            | Op::Assume(..)
            | Op::Binary(Binary::Sdiv | Binary::Udiv | Binary::Srem | Binary::Urem, ..)
    )
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

/// Factors single-use bitwise identities and sinks deferred loads and pure scalar ops to their uses.
pub(crate) fn run(func: &mut Function) {
    factor_bitwise(func);
    let uses = func.use_counts();
    for bi in 0..func.blocks.len() {
        sink_loads_block(func, &uses, bi);
    }
    let live_defs = defs(func);
    for bi in 0..func.blocks.len() {
        sink_pure_block(func, &live_defs, &uses, bi);
    }
}
