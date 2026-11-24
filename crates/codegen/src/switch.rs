mod bitmask;
mod chain;
mod emit;
mod lut;
mod member;
mod select;
mod tree;

use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::fold::{const_offset, extend};
use crate::ir::{Binary, Block, Domain, Function, Inst, IntCC, Op, Type, Value};
use bitmask::{build_runs, can_bitmask_runs, try_rewrite_expr};
use emit::{can_br_table, emit_node, forward_cached};

pub(super) struct Info {
    cfg: std::sync::Arc<Cfg>,
    pub(super) uses: Vec<u32>,
    consts: Vec<Option<i64>>,
    pub(super) def_block: Vec<Option<usize>>,
    pub(super) def_inst: Vec<Option<Inst>>,
}

#[derive(Clone, Copy)]
pub(super) enum TestKind {
    Cmp(IntCC, i64),
    SubCmp { lo: u64, cc: IntCC, imm: i64 },
}

pub(super) struct BlockTest {
    pub(super) val: Value,
    pub(super) kind: TestKind,
    pub(super) used: [Option<Inst>; 3],
}

#[derive(Clone, Copy)]
pub(super) struct Range {
    pub(super) lo: u64,
    pub(super) hi: u64,
    s_lo: i64,
    s_hi: i64,
    signed_ok: bool,
}

#[derive(Clone)]
pub(super) struct Edge {
    pub(super) raw: Block,
    pub(super) block: Block,
    pub(super) args: Vec<Value>,
    pub(super) synth_bool: Option<bool>,
}

#[derive(Clone)]
pub(super) struct Arm {
    pub(super) range: Range,
    pub(super) hit: Edge,
}

#[derive(Clone)]
pub(super) struct Run {
    pub(super) lo: u64,
    pub(super) hi: u64,
    pub(super) hit: Edge,
}

fn analyze(func: &Function) -> Info {
    let cfg = func.cfg();
    let uses = func.use_counts();
    let mut consts = vec![None; func.types.len()];
    let mut def_block = vec![None; func.types.len()];
    let mut def_inst = vec![None; func.types.len()];
    for (b, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                def_block[res.index()] = Some(b);
                def_inst[res.index()] = Some(inst);
                if let Op::Const(imm) = data.op {
                    consts[res.index()] = Some(imm);
                }
            }
        }
    }
    Info {
        cfg,
        uses,
        consts,
        def_block,
        def_inst,
    }
}

pub(super) fn make_range(ty: Type, lo: u64, hi: u64) -> Range {
    let (s_lo, s_hi) = (extend(lo as i64, ty, true), extend(hi as i64, ty, true));
    Range {
        lo,
        hi,
        s_lo,
        s_hi,
        signed_ok: s_lo <= s_hi,
    }
}

fn make_signed_range(ty: Type, s_lo: i64, s_hi: i64) -> Range {
    Range {
        lo: extend(s_lo, ty, false) as u64,
        hi: extend(s_hi, ty, false) as u64,
        s_lo,
        s_hi,
        signed_ok: true,
    }
}

fn match_sub_offset(
    func: &Function,
    info: &Info,
    val: Value,
    expected_block: Option<usize>,
) -> Option<(Value, u64, Option<Inst>)> {
    let inst = info.def_inst.get(val.index()).copied().flatten()?;
    let (src, delta) = const_offset(&func.insts[inst.index()].op, |c| {
        info.consts.get(c.index()).copied().flatten()
    })?;
    let owned = expected_block.is_none_or(|b| {
        info.def_block[val.index()] == Some(b) && info.uses.get(val.index()).copied() == Some(1)
    });
    let ty = func.ty(src);
    ty.is_small_int().then(|| {
        (
            src,
            extend(delta.wrapping_neg(), ty, false) as u64,
            owned.then_some(inst),
        )
    })
}

pub(super) fn match_test(
    func: &Function,
    info: &Info,
    cond: Value,
    expected_val: Option<Value>,
    expected_block: Option<usize>,
) -> Option<BlockTest> {
    if expected_block.is_some_and(|b| {
        info.def_block[cond.index()] != Some(b) || info.uses.get(cond.index()).copied() != Some(1)
    }) {
        return None;
    }
    let outer_inst = info.def_inst.get(cond.index()).copied().flatten()?;
    let (cmp_inst, xor_inst, invert) = match func.insts[outer_inst.index()].op {
        Op::Binary(Binary::Bxor, x, y) => {
            let inner = [(x, y), (y, x)].into_iter().find_map(|(v, o)| {
                let imm = info.consts.get(o.index()).copied().flatten()?;
                (extend(imm, Type::I8, false) == 1).then_some(v)
            })?;
            if expected_block.is_some_and(|b| {
                info.def_block[inner.index()] != Some(b)
                    || info.uses.get(inner.index()).copied() != Some(1)
            }) {
                return None;
            }
            (
                info.def_inst.get(inner.index()).copied().flatten()?,
                Some(outer_inst),
                true,
            )
        }
        _ => (outer_inst, None, false),
    };
    let Op::Icmp(raw_cc, a, b) = func.insts[cmp_inst.index()].op else {
        return None;
    };
    let (lhs, raw_cc, imm) = match (
        info.consts.get(a.index()).copied().flatten(),
        info.consts.get(b.index()).copied().flatten(),
    ) {
        (_, Some(imm)) => (a, raw_cc, imm),
        (Some(imm), None) => (b, raw_cc.swapped(), imm),
        _ => return None,
    };
    let cc = if invert { raw_cc.inverse() } else { raw_cc };
    let used_with = |diff: Option<Inst>| match xor_inst {
        Some(xi) => [Some(xi), Some(cmp_inst), diff],
        None => [Some(cmp_inst), diff, None],
    };
    if let Some((src, lo, diff_inst)) = match_sub_offset(func, info, lhs, expected_block)
        && expected_val.is_none_or(|ev| ev == src)
    {
        let kind = if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            TestKind::Cmp(cc, imm.wrapping_add(lo as i64))
        } else {
            TestKind::SubCmp { lo, cc, imm }
        };
        return Some(BlockTest {
            val: src,
            kind,
            used: used_with(diff_inst),
        });
    }
    if func.ty(lhs).is_small_int() && expected_val.is_none_or(|ev| ev == lhs) {
        return Some(BlockTest {
            val: lhs,
            kind: TestKind::Cmp(cc, imm),
            used: used_with(None),
        });
    }
    None
}

pub(super) fn range_of_test(ty: Type, kind: TestKind, want_true: bool) -> Option<Range> {
    let u_max = ty.unsigned_max();
    match kind {
        TestKind::SubCmp { lo, cc, imm } => {
            let r = range_of_test(ty, TestKind::Cmp(cc, imm), want_true)?;
            Some(make_range(
                ty,
                r.lo.wrapping_add(lo) & u_max,
                r.hi.wrapping_add(lo) & u_max,
            ))
        }
        TestKind::Cmp(cc, imm) => match if want_true { cc } else { cc.inverse() } {
            IntCC::Equal => {
                let (u, s) = (extend(imm, ty, false) as u64, extend(imm, ty, true));
                Some(Range {
                    lo: u,
                    hi: u,
                    s_lo: s,
                    s_hi: s,
                    signed_ok: true,
                })
            }
            IntCC::NotEqual => None,
            cc if !cc.is_signed() => {
                let u = extend(imm, ty, false) as u64;
                let (lo, hi) = match cc {
                    IntCC::UnsignedLessThan if u > 0 => (0, u - 1),
                    IntCC::UnsignedLessThanOrEqual => (0, u),
                    IntCC::UnsignedGreaterThan if u < u_max => (u + 1, u_max),
                    IntCC::UnsignedGreaterThanOrEqual => (u, u_max),
                    _ => return None,
                };
                Some(make_range(ty, lo, hi))
            }
            cc => {
                let (s_min, s_max) = if ty == Type::I64 {
                    (i64::MIN, i64::MAX)
                } else {
                    let h = 1_i64 << (ty.bits() - 1);
                    (-h, h - 1)
                };
                let s = extend(imm, ty, true);
                let (s_lo, s_hi) = match cc {
                    IntCC::SignedLessThan if s > s_min => (s_min, s - 1),
                    IntCC::SignedLessThanOrEqual => (s_min, s),
                    IntCC::SignedGreaterThan if s < s_max => (s + 1, s_max),
                    IntCC::SignedGreaterThanOrEqual => (s, s_max),
                    _ => return None,
                };
                Some(make_signed_range(ty, s_lo, s_hi))
            }
        },
    }
}

pub(super) fn intersect_ranges(ty: Type, r1: Range, r2: Range) -> Option<Range> {
    if r1.lo <= r1.hi && r2.lo <= r2.hi {
        let (lo, hi) = (r1.lo.max(r2.lo), r1.hi.min(r2.hi));
        return (lo <= hi).then(|| make_range(ty, lo, hi));
    }
    if r1.signed_ok && r2.signed_ok {
        let (s_lo, s_hi) = (r1.s_lo.max(r2.s_lo), r1.s_hi.min(r2.s_hi));
        return (s_lo <= s_hi).then(|| make_signed_range(ty, s_lo, s_hi));
    }
    None
}

pub(super) fn body_clean(
    func: &Function,
    info: &Info,
    block: usize,
    is_used: &dyn Fn(Inst) -> bool,
    jump_cond: Option<Value>,
) -> bool {
    let insts = &func.blocks[block].insts;
    let Some((&last, body)) = insts.split_last() else {
        return false;
    };
    let term_args = match &func.insts[last.index()].op {
        Op::Brif(_, _, t_args, _, f_args) => [t_args.as_slice(), f_args.as_slice()],
        Op::Jump(_, args) if jump_cond.is_some() => [args.as_slice(), &[]],
        _ => return false,
    };
    for &inst in body {
        let data = &func.insts[inst.index()];
        if matches!(data.op, Op::Const(_)) {
            continue;
        }
        let Some(res) = data.result.filter(|_| is_used(inst)) else {
            return false;
        };
        if jump_cond != Some(res) && (term_args[0].contains(&res) || term_args[1].contains(&res)) {
            return false;
        }
        let mut local = 0_u32;
        for &other in insts {
            func.insts[other.index()]
                .op
                .for_operands(|op| local += u32::from(op == res));
        }
        if local != info.uses[res.index()] {
            return false;
        }
    }
    true
}

pub(super) fn can_enter(
    func: &Function,
    info: &Info,
    (b0, cleared, chain): (usize, &[bool], &[usize]),
    cur: usize,
) -> bool {
    cur < cleared.len()
        && cur != b0
        && !cleared[cur]
        && !chain.contains(&cur)
        && func.entry != Some(Block(cur as u32))
        && info.cfg.pred_count(Block(cur as u32)) == 1
        && func.blocks[cur].params.is_empty()
}

fn const_bool(func: &Function, info: &Info, val: Value) -> Option<bool> {
    (func.ty(val) == Type::I8)
        .then(|| info.consts.get(val.index()).copied().flatten())
        .flatten()
        .and_then(|imm| match extend(imm, Type::I8, false) {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        })
}

pub(super) fn same_edge(func: &Function, info: &Info, a: &Edge, b: &Edge) -> bool {
    let len = |e: &Edge| {
        if e.synth_bool.is_some() {
            1
        } else {
            e.args.len()
        }
    };
    if a.block != b.block || len(a) != len(b) {
        return false;
    }
    match (a.synth_bool, b.synth_bool) {
        (Some(ab), Some(bb)) => ab == bb,
        (Some(ab), None) => b.args.first().and_then(|&v| const_bool(func, info, v)) == Some(ab),
        (None, Some(bb)) => a.args.first().and_then(|&v| const_bool(func, info, v)) == Some(bb),
        (None, None) => a.args.iter().zip(&b.args).all(|(&x, &y)| {
            x == y
                || (func.ty(x) == func.ty(y)
                    && info
                        .consts
                        .get(x.index())
                        .copied()
                        .flatten()
                        .is_some_and(|ix| {
                            info.consts
                                .get(y.index())
                                .copied()
                                .flatten()
                                .is_some_and(|iy| {
                                    extend(ix, func.ty(x), false) == extend(iy, func.ty(y), false)
                                })
                        }))
        }),
    }
}

pub(super) fn edge_of(func: &Function, block: Block, args: &[Value]) -> Edge {
    let resolved = if args.is_empty() {
        func.jump_target(block)
    } else {
        block
    };
    Edge {
        raw: block,
        block: resolved,
        args: args.to_vec(),
        synth_bool: None,
    }
}

pub(super) fn can_follow_into(
    func: &Function,
    info: &Info,
    ctx: (usize, &[bool], &[usize]),
    edge: &Edge,
) -> Option<(usize, Vec<usize>)> {
    if !edge.args.is_empty() || edge.synth_bool.is_some() {
        return None;
    }
    let (mut cur, mut hops) = (edge.raw, Vec::new());
    while cur != edge.block {
        if !can_enter(func, info, ctx, cur.index()) || hops.contains(&cur.index()) {
            return None;
        }
        hops.push(cur.index());
        let [only] = func.blocks[cur.index()].insts[..] else {
            return None;
        };
        let Op::Jump(next, _) = func.insts[only.index()].op else {
            return None;
        };
        cur = next;
    }
    can_enter(func, info, ctx, cur.index()).then_some((cur.index(), hops))
}

pub(super) fn range_fits(r: Range) -> bool {
    r.lo <= r.hi || (r.signed_ok && r.s_hi.abs_diff(r.s_lo) < 256)
}

pub(super) fn collect_hoisted_consts(func: &Function, info: &Info, chain: &[usize]) -> Vec<Inst> {
    let (mut consts, mut order): (HashMap<Value, (Inst, u32)>, Vec<Value>) =
        (HashMap::default(), Vec::new());
    for &b in chain {
        for &inst in &func.blocks[b].insts {
            let data = &func.insts[inst.index()];
            if matches!(data.op, Op::Const(_))
                && let Some(res) = data.result
                && !consts.contains_key(&res)
            {
                consts.insert(res, (inst, 0));
                order.push(res);
            }
        }
    }
    let mut bump = |v: &Value| {
        if let Some(entry) = consts.get_mut(v) {
            entry.1 += 1;
        }
    };
    for &b in chain {
        let Some((&last, body)) = func.blocks[b].insts.split_last() else {
            continue;
        };
        body.iter()
            .for_each(|&i| func.insts[i.index()].op.for_operands(|op| bump(&op)));
        match &func.insts[last.index()].op {
            Op::Brif(cond, ..) => bump(cond),
            Op::Jump(_, args) => args.iter().for_each(&mut bump),
            _ => {}
        }
    }
    order
        .into_iter()
        .filter(|res| consts[res].1 < info.uses[res.index()])
        .map(|res| consts[&res].0)
        .collect()
}

pub(super) fn push_before_last(func: &mut Function, block: usize, inst: Inst) {
    let insts = &mut func.blocks[block].insts;
    insts.insert(insts.len() - 1, inst);
}

pub(super) fn clean_dead_head_insts(func: &mut Function, info: &Info, b0: usize, used: &[Inst]) {
    for &inst in used {
        let Some(res) = func.insts[inst.index()].result else {
            break;
        };
        if info.def_block[res.index()] != Some(b0) || info.uses[res.index()] != 1 {
            break;
        }
        func.blocks[b0].insts.retain(|&i| i != inst);
        func.insts[inst.index()].op = Op::Trap;
    }
}

fn try_rewrite_br_table(
    func: &mut Function,
    info: &Info,
    (b0, last0): (usize, Inst),
    cleared: &mut [bool],
) -> bool {
    let Op::BrTable(idx, default_blk, ref targets) = func.insts[last0.index()].op else {
        return false;
    };
    let ty = func.ty(idx);
    let default_edge = edge_of(func, default_blk, &[]);
    let u_max = ty.unsigned_max();
    let mut arms = Vec::new();
    for (i, &t) in targets.iter().enumerate() {
        let u = i as u64;
        if u > u_max {
            break;
        }
        let hit = edge_of(func, t, &[]);
        if !same_edge(func, info, &hit, &default_edge) {
            arms.push(Arm {
                range: make_range(ty, u, u),
                hit,
            });
        }
    }
    let runs = build_runs(func, info, ty, &arms);
    if runs.is_empty() {
        func.insts[last0.index()].op = Op::Jump(default_edge.block, default_edge.args);
        cleared[b0] = true;
        return true;
    }
    if lut::try_emit_runs_lut(func, info, (b0, last0), idx, &runs, &default_edge, &[]) {
        cleared[b0] = true;
        return true;
    }
    if runs.len() > 2
        && !can_bitmask_runs(func, info, ty, &runs)
        && !(runs.len() <= 4 && can_br_table(&runs).is_none())
    {
        return false;
    }
    let mut cache = Vec::new();
    let default_target = forward_cached(func, &mut cache, default_edge.block, default_edge.args);
    emit_node(
        func,
        info,
        Some((b0, last0)),
        (idx, ty),
        &runs,
        (default_target, &mut cache),
    );
    cleared[b0] = true;
    true
}

pub(crate) fn run(func: &mut Function, target: &crate::Target) -> Domain {
    if func.coroutine.is_some() {
        return Domain::NONE;
    }
    let mut info = analyze(func);
    let initial_blocks = func.blocks.len();
    let mut cleared = vec![false; initial_blocks];
    let mut changed = false;
    let flag_chain = target.arch == crate::target::Arch::Arm64;
    for b0 in 0..initial_blocks {
        changed |= select::try_rewrite_select_trees(func, &info, b0);
    }
    if changed {
        info = analyze(func);
    }
    for b0 in 0..initial_blocks {
        if cleared[b0] {
            continue;
        }
        let Some(&last0) = func.blocks[b0].insts.last() else {
            continue;
        };
        if let Some(t) = tree::collect_tree(func, &info, b0, &cleared).filter(tree::Tree::is_forked)
            && tree::rewrite_tree(func, &info, (b0, last0), t, &mut cleared)
        {
            changed = true;
            continue;
        }
        if let Some(ch) = chain::collect_chain(func, &info, b0, &cleared)
            && chain::rewrite_chain(func, &info, (b0, last0), ch, (&mut cleared, flag_chain))
        {
            changed = true;
            continue;
        }
        if let Some(t) = tree::collect_tree(func, &info, b0, &cleared)
            && tree::rewrite_tree(func, &info, (b0, last0), t, &mut cleared)
        {
            changed = true;
        } else if try_rewrite_expr(func, &info, (b0, last0), &mut cleared)
            || try_rewrite_br_table(func, &info, (b0, last0), &mut cleared)
        {
            changed = true;
        }
    }
    if changed {
        func.bump_cfg();
        Domain::CFG | Domain::VALUES
    } else {
        Domain::NONE
    }
}
