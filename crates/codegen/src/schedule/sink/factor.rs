use crate::fold::defs;
use crate::ir::{Binary, Block, Function, Inst, Op, Unary, Value};

use super::{add_use, fill_block_depths, set_bin_depth};

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

fn band_operands(
    func: &Function,
    defs: &[Option<Inst>],
    uses: &[u32],
    val: Value,
) -> Option<(Inst, Value, Value)> {
    let inst = defs.get(val.index()).copied().flatten()?;
    if uses.get(val.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let Op::Binary(Binary::Band, a, b) = func.insts[inst.index()].op else {
        return None;
    };
    Some((inst, a, b))
}

fn match_distributive(
    func: &Function,
    defs: &[Option<Inst>],
    uses: &[u32],
    op: &Op,
) -> Option<DistMatch> {
    let &Op::Binary(outer @ (Binary::Bxor | Binary::Bor), lhs, rhs) = op else {
        return None;
    };
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
    Some(DistMatch {
        op: outer,
        common,
        x,
        y,
        kill: [li, ri],
    })
}

fn find_bnot(
    func: &Function,
    defs: &[Option<Inst>],
    uses: &[u32],
    val: Value,
) -> Option<(Inst, Value)> {
    if uses.get(val.index()).copied().unwrap_or(0) != 1 {
        return None;
    }
    let inst = defs.get(val.index()).copied().flatten()?;
    let Op::Unary(Unary::Bnot, inner) = func.insts[inst.index()].op else {
        return None;
    };
    Some((inst, inner))
}

fn match_ch(func: &Function, defs: &[Option<Inst>], uses: &[u32], op: &Op) -> Option<ChMatch> {
    let &Op::Binary(Binary::Bxor, lhs, rhs) = op else {
        return None;
    };
    let (li, l0, l1) = band_operands(func, defs, uses, lhs)?;
    let (ri, r0, r1) = band_operands(func, defs, uses, rhs)?;
    for ((a0, a1), (b0, b1)) in [((l0, l1), (r0, r1)), ((r0, r1), (l0, l1))] {
        for (e, f) in [(a0, a1), (a1, a0)] {
            for (not_c, g) in [(b0, b1), (b1, b0)] {
                if let Some((ni, inner)) = find_bnot(func, defs, uses, not_c)
                    && inner == e
                {
                    return Some(ChMatch {
                        e,
                        f,
                        g,
                        kill: [li, ri, ni],
                    });
                }
            }
        }
    }
    None
}

fn same_pair((a, b): (Value, Value), (c, d): (Value, Value)) -> bool {
    (a == c && b == d) || (a == d && b == c)
}

fn match_maj_side(
    func: &Function,
    fs: &FactorState,
    bi: usize,
    (outer, fac_side, pair_side): (Binary, Value, Value),
) -> Option<MajMatch> {
    let (pair_i, p0, p1) = band_operands(func, &fs.defs, &fs.uses, pair_side)?;
    let (fac_i, f0, f1) = band_operands(func, &fs.defs, &fs.uses, fac_side)?;
    if p0 == p1
        || fs.inst_block.get(pair_i.index()).copied() != Some(bi)
        || fs.inst_block.get(fac_i.index()).copied() != Some(bi)
    {
        return None;
    }
    for (common_old, inner) in [(f0, f1), (f1, f0)] {
        if common_old == p0
            || common_old == p1
            || fs.uses.get(inner.index()).copied().unwrap_or(0) != 1
        {
            continue;
        }
        let Some(inner_i) = fs.defs.get(inner.index()).copied().flatten() else {
            continue;
        };
        if fs.inst_block.get(inner_i.index()).copied() != Some(bi) {
            continue;
        }
        let Op::Binary(inner_op, x0, x1) = func.insts[inner_i.index()].op else {
            continue;
        };
        if inner_op != outer || !same_pair((x0, x1), (p0, p1)) {
            continue;
        }
        let (best, other) = if fs.depth[p0.index()] >= fs.depth[p1.index()] {
            (p0, p1)
        } else {
            (p1, p0)
        };
        if fs.depth[best.index()] > fs.depth[common_old.index()] {
            return Some(MajMatch {
                op: outer,
                best,
                u: common_old,
                v: other,
                common_old,
                kill: [inner_i, fac_i, pair_i],
            });
        }
    }
    None
}

fn match_maj_refactor(func: &Function, fs: &FactorState, bi: usize, op: &Op) -> Option<MajMatch> {
    let &Op::Binary(outer @ (Binary::Bxor | Binary::Bor), lhs, rhs) = op else {
        return None;
    };
    match_maj_side(func, fs, bi, (outer, lhs, rhs))
        .or_else(|| match_maj_side(func, fs, bi, (outer, rhs, lhs)))
}

fn record_factor_def(fs: &mut FactorState, func: &Function, bi: usize, (inst, val): (Inst, Value)) {
    fs.defs.resize(func.types.len(), None);
    fs.inst_block.resize(func.insts.len(), usize::MAX);
    fs.defs[val.index()] = Some(inst);
    fs.inst_block[inst.index()] = bi;
}

fn apply_distributive(
    func: &mut Function,
    block: Block,
    at: usize,
    m: DistMatch,
    fs: &mut FactorState,
) -> usize {
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
        if let Some(r) = res {
            set_bin_depth(&mut fs.depth, r, m.common, t_val);
        }
    }
    add_use(&mut fs.uses, m.common, -1);
    add_use(&mut fs.uses, t_val, 1);
    for k in m.kill {
        fs.dead[k.index()] = true;
    }
    fs.changed = true;
    1
}

fn apply_ch(
    func: &mut Function,
    block: Block,
    at: usize,
    m: ChMatch,
    fs: &mut FactorState,
) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let res = func.insts[inst.index()].result;
    let ty = func.ty(res.unwrap_or(m.e));
    let (t_val, m_val) = (func.value(ty), func.value(ty));
    let t_inst = func.add(Op::Binary(Binary::Bxor, m.f, m.g), Some(t_val));
    let m_inst = func.add(Op::Binary(Binary::Band, m.e, t_val), Some(m_val));
    func.insts[inst.index()].op = Op::Binary(Binary::Bxor, m.g, m_val);
    func.blocks[block.index()]
        .insts
        .splice(at..at, [t_inst, m_inst]);
    record_factor_def(fs, func, block.index(), (t_inst, t_val));
    record_factor_def(fs, func, block.index(), (m_inst, m_val));
    if !fs.depth.is_empty() {
        fs.depth.resize(func.types.len(), 0);
        set_bin_depth(&mut fs.depth, t_val, m.f, m.g);
        set_bin_depth(&mut fs.depth, m_val, m.e, t_val);
        if let Some(r) = res {
            set_bin_depth(&mut fs.depth, r, m.g, m_val);
        }
    }
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

fn apply_maj_refactor(
    func: &mut Function,
    block: Block,
    at: usize,
    m: MajMatch,
    fs: &mut FactorState,
) -> usize {
    let inst = func.blocks[block.index()].insts[at];
    let res = func.insts[inst.index()].result;
    let ty = func.ty(res.unwrap_or(m.best));
    let (t_val, m_val, k_val) = (func.value(ty), func.value(ty), func.value(ty));
    let t_inst = func.add(Op::Binary(m.op, m.u, m.v), Some(t_val));
    let m_inst = func.add(Op::Binary(Binary::Band, m.best, t_val), Some(m_val));
    let k_inst = func.add(Op::Binary(Binary::Band, m.u, m.v), Some(k_val));
    func.insts[inst.index()].op = Op::Binary(m.op, m_val, k_val);
    func.blocks[block.index()]
        .insts
        .splice(at..at, [t_inst, m_inst, k_inst]);
    record_factor_def(fs, func, block.index(), (t_inst, t_val));
    record_factor_def(fs, func, block.index(), (m_inst, m_val));
    record_factor_def(fs, func, block.index(), (k_inst, k_val));
    if !fs.depth.is_empty() {
        fs.depth.resize(func.types.len(), 0);
        set_bin_depth(&mut fs.depth, t_val, m.u, m.v);
        set_bin_depth(&mut fs.depth, m_val, m.best, t_val);
        set_bin_depth(&mut fs.depth, k_val, m.u, m.v);
        if let Some(r) = res {
            set_bin_depth(&mut fs.depth, r, m_val, k_val);
        }
    }
    add_use(&mut fs.uses, m.common_old, 1);
    add_use(&mut fs.uses, m.best, -1);
    add_use(&mut fs.uses, t_val, 1);
    add_use(&mut fs.uses, m_val, 1);
    add_use(&mut fs.uses, k_val, 1);
    for k in m.kill {
        fs.dead[k.index()] = true;
    }
    fs.changed = true;
    3
}

fn factor_block(func: &mut Function, bi: usize, fs: &mut FactorState, balance: bool) {
    if !func.blocks[bi].insts.iter().any(|&i| {
        matches!(
            func.insts[i.index()].op,
            Op::Binary(Binary::Bor | Binary::Bxor, ..)
        )
    }) {
        return;
    }
    let block = Block(bi as u32);
    if balance {
        fill_block_depths(func, bi, &mut fs.depth);
    }
    let mut at = 0;
    while at < func.blocks[bi].insts.len() {
        let inst = func.blocks[bi].insts[at];
        if fs.dead.get(inst.index()).copied().unwrap_or(false)
            || !matches!(
                func.insts[inst.index()].op,
                Op::Binary(Binary::Bor | Binary::Bxor, ..)
            )
        {
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
        } else if balance && let Some(m) = match_maj_refactor(func, fs, bi, &op) {
            fs.dead.resize(func.insts.len(), false);
            at += apply_maj_refactor(func, block, at, m, fs) + 1;
        } else {
            at += 1;
        }
    }
}

pub(super) fn factor_bitwise(
    func: &mut Function,
    balance: bool,
) -> Option<(Vec<Option<Inst>>, Vec<u32>)> {
    if !func.blocks.iter().flat_map(|b| &b.insts).any(|&i| {
        matches!(
            func.insts[i.index()].op,
            Op::Binary(Binary::Bor | Binary::Bxor, ..)
        )
    }) {
        return None;
    }
    let inst_block = if balance {
        let mut map = vec![usize::MAX; func.insts.len()];
        for (bi, block) in func.blocks.iter().enumerate() {
            for &inst in &block.insts {
                map[inst.index()] = bi;
            }
        }
        map
    } else {
        Vec::new()
    };
    let mut fs = FactorState {
        defs: defs(func),
        uses: func.use_counts(),
        depth: Vec::new(),
        inst_block,
        dead: vec![false; func.insts.len()],
        changed: false,
    };
    for bi in 0..func.blocks.len() {
        factor_block(func, bi, &mut fs, balance);
    }
    if !fs.changed {
        return Some((fs.defs, fs.uses));
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
    None
}
