mod bitmask;
mod emit;
mod lut;
mod select;
mod tree;

use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::cse::flip_int_cc;
use crate::fold::extend;
use crate::ir::{Binary, Block, Domain, Function, Inst, IntCC, Op, Type, Value};
use bitmask::{build_runs, can_bitmask_runs, try_rewrite_expr};
use emit::{can_br_table, emit_node, emit_sub_range, forward_cached, try_signed_br_table};

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
    SubCmp { lo: u64, cc: IntCC, bound: u64 },
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

struct ArmMatch {
    val: Value,
    arm: Arm,
    next: Edge,
    extra_blocks: Vec<usize>,
    used: [Option<Inst>; 3],
    terminates: bool,
}

struct Chain {
    val: Value,
    arms: Vec<Arm>,
    default_edge: Edge,
    chain_blocks: Vec<usize>,
    two_block_arms: u32,
    head_used: [Option<Inst>; 3],
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
    Info { cfg, uses, consts, def_block, def_inst }
}

pub(super) fn make_range(ty: Type, lo: u64, hi: u64) -> Range {
    let (s_lo, s_hi) = (extend(lo as i64, ty, true), extend(hi as i64, ty, true));
    Range { lo, hi, s_lo, s_hi, signed_ok: s_lo <= s_hi }
}

fn make_signed_range(ty: Type, s_lo: i64, s_hi: i64) -> Range {
    Range { lo: extend(s_lo, ty, false) as u64, hi: extend(s_hi, ty, false) as u64, s_lo, s_hi, signed_ok: true }
}

fn match_sub_offset(func: &Function, info: &Info, val: Value, expected_block: Option<usize>) -> Option<(Value, u64, Inst)> {
    let inst = info.def_inst.get(val.index()).copied().flatten()?;
    if expected_block.is_some_and(|b| info.def_block[val.index()] != Some(b) || info.uses.get(val.index()).copied() != Some(1)) {
        return None;
    }
    let (src, lo_raw) = match func.insts[inst.index()].op {
        Op::Binary(Binary::Isub, src, c) => (src, info.consts.get(c.index()).copied().flatten()?),
        Op::Binary(Binary::Iadd, src, c) => (src, info.consts.get(c.index()).copied().flatten()?.wrapping_neg()),
        _ => return None,
    };
    let ty = func.ty(src);
    ty.is_small_int().then(|| (src, extend(lo_raw, ty, false) as u64, inst))
}

pub(super) fn match_test(func: &Function, info: &Info, cond: Value, expected_val: Option<Value>, expected_block: Option<usize>) -> Option<BlockTest> {
    if expected_block.is_some_and(|b| info.def_block[cond.index()] != Some(b) || info.uses.get(cond.index()).copied() != Some(1)) {
        return None;
    }
    let outer_inst = info.def_inst.get(cond.index()).copied().flatten()?;
    let (cmp_inst, xor_inst, invert) = match func.insts[outer_inst.index()].op {
        Op::Binary(Binary::Bxor, x, y) => {
            let inner = [(x, y), (y, x)].into_iter().find_map(|(v, o)| {
                let imm = info.consts.get(o.index()).copied().flatten()?;
                (extend(imm, Type::I8, false) == 1).then_some(v)
            })?;
            if expected_block.is_some_and(|b| info.def_block[inner.index()] != Some(b) || info.uses.get(inner.index()).copied() != Some(1)) {
                return None;
            }
            (info.def_inst.get(inner.index()).copied().flatten()?, Some(outer_inst), true)
        }
        _ => (outer_inst, None, false),
    };
    let Op::Icmp(raw_cc, a, b) = func.insts[cmp_inst.index()].op else { return None };
    let (lhs, raw_cc, imm) = match (info.consts.get(a.index()).copied().flatten(), info.consts.get(b.index()).copied().flatten()) {
        (_, Some(imm)) => (a, raw_cc, imm),
        (Some(imm), None) => (b, flip_int_cc(raw_cc), imm),
        _ => return None,
    };
    let cc = if invert { raw_cc.inverse() } else { raw_cc };
    let used_with = |diff: Option<Inst>| match xor_inst {
        Some(xi) => [Some(xi), Some(cmp_inst), diff],
        None => [Some(cmp_inst), diff, None],
    };
    if !cc.is_signed()
        && !matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some((src, lo, diff_inst)) = match_sub_offset(func, info, lhs, expected_block)
        && expected_val.is_none_or(|ev| ev == src)
    {
        let bound = extend(imm, func.ty(src), false) as u64;
        return Some(BlockTest { val: src, kind: TestKind::SubCmp { lo, cc, bound }, used: used_with(Some(diff_inst)) });
    }
    if func.ty(lhs).is_small_int() && expected_val.is_none_or(|ev| ev == lhs) {
        return Some(BlockTest { val: lhs, kind: TestKind::Cmp(cc, imm), used: used_with(None) });
    }
    None
}

pub(super) fn range_of_test(ty: Type, kind: TestKind, want_true: bool) -> Option<Range> {
    let u_max = ty.unsigned_max();
    match kind {
        TestKind::SubCmp { lo, cc, bound } => {
            let span = match if want_true { cc } else { cc.inverse() } {
                IntCC::UnsignedLessThanOrEqual => bound,
                IntCC::UnsignedLessThan if bound > 0 => bound - 1,
                _ => return None,
            };
            (span < u_max && u128::from(lo) + u128::from(span) <= u128::from(u_max)).then(|| make_range(ty, lo, lo + span))
        }
        TestKind::Cmp(cc, imm) => match if want_true { cc } else { cc.inverse() } {
            IntCC::Equal => {
                let (u, s) = (extend(imm, ty, false) as u64, extend(imm, ty, true));
                Some(Range { lo: u, hi: u, s_lo: s, s_hi: s, signed_ok: true })
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
                let (s_min, s_max) = if ty == Type::I64 { (i64::MIN, i64::MAX) } else { let h = 1_i64 << (ty.bits() - 1); (-h, h - 1) };
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

pub(super) fn body_clean(func: &Function, info: &Info, block: usize, is_used: &dyn Fn(Inst) -> bool, jump_cond: Option<Value>) -> bool {
    let insts = &func.blocks[block].insts;
    let Some((&last, body)) = insts.split_last() else { return false };
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
        let Some(res) = data.result.filter(|_| is_used(inst)) else { return false };
        if jump_cond != Some(res) && (term_args[0].contains(&res) || term_args[1].contains(&res)) {
            return false;
        }
        let mut local = 0_u32;
        for &other in insts {
            func.insts[other.index()].op.for_operands(|op| local += u32::from(op == res));
        }
        if local != info.uses[res.index()] {
            return false;
        }
    }
    true
}

pub(super) fn can_enter(func: &Function, info: &Info, (b0, cleared, chain): (usize, &[bool], &[usize]), cur: usize) -> bool {
    cur < cleared.len()
        && cur != b0
        && !cleared[cur]
        && !chain.contains(&cur)
        && func.entry != Some(Block(cur as u32))
        && info.cfg.pred_count(Block(cur as u32)) == 1
        && func.blocks[cur].params.is_empty()
}

fn const_bool(func: &Function, info: &Info, val: Value) -> Option<bool> {
    (func.ty(val) == Type::I8).then(|| info.consts.get(val.index()).copied().flatten()).flatten().and_then(|imm| match extend(imm, Type::I8, false) {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    })
}

pub(super) fn same_edge(func: &Function, info: &Info, a: &Edge, b: &Edge) -> bool {
    let len = |e: &Edge| if e.synth_bool.is_some() { 1 } else { e.args.len() };
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
                    && info.consts.get(x.index()).copied().flatten().is_some_and(|ix| {
                        info.consts.get(y.index()).copied().flatten().is_some_and(|iy| extend(ix, func.ty(x), false) == extend(iy, func.ty(y), false))
                    }))
        }),
    }
}

pub(super) fn edge_of(func: &Function, block: Block, args: &[Value]) -> Edge {
    let resolved = if args.is_empty() { func.jump_target(block) } else { block };
    Edge { raw: block, block: resolved, args: args.to_vec(), synth_bool: None }
}

pub(super) fn can_follow_into(func: &Function, info: &Info, ctx: (usize, &[bool], &[usize]), edge: &Edge) -> Option<(usize, Vec<usize>)> {
    if !edge.args.is_empty() || edge.synth_bool.is_some() {
        return None;
    }
    let (mut cur, mut hops) = (edge.raw, Vec::new());
    while cur != edge.block {
        if !can_enter(func, info, ctx, cur.index()) || hops.contains(&cur.index()) {
            return None;
        }
        hops.push(cur.index());
        let [only] = func.blocks[cur.index()].insts[..] else { return None };
        let Op::Jump(next, _) = func.insts[only.index()].op else { return None };
        cur = next;
    }
    can_enter(func, info, ctx, cur.index()).then_some((cur.index(), hops))
}

fn match_two_block_range(
    func: &Function,
    info: &Info,
    ctx: (usize, &[bool], &[usize]),
    (v, ty, kind1, want_true1): (Value, Type, TestKind, bool),
    (hit_edge, fail_edge): (&Edge, &Edge),
) -> Option<(Arm, Vec<usize>)> {
    let (mid, mut extra) = can_follow_into(func, info, ctx, hit_edge)?;
    let r1 = range_of_test(ty, kind1, want_true1)?;
    let last = *func.blocks[mid].insts.last()?;
    let arm = match &func.insts[last.index()].op {
        Op::Brif(cond2, t2, t2_args, f2, f2_args) => {
            let test2 = match_test(func, info, *cond2, Some(v), Some(mid))?;
            if !body_clean(func, info, mid, &|i| test2.used.contains(&Some(i)), None) {
                return None;
            }
            let (t2_edge, f2_edge) = (edge_of(func, *t2, t2_args), edge_of(func, *f2, f2_args));
            let (want_true2, hit) = if same_edge(func, info, &f2_edge, fail_edge) {
                (true, t2_edge)
            } else if same_edge(func, info, &t2_edge, fail_edge) {
                (false, f2_edge)
            } else {
                return None;
            };
            Arm { range: intersect_ranges(ty, r1, range_of_test(ty, test2.kind, want_true2)?)?, hit }
        }
        Op::Jump(target, args) if args.len() == 1 && *target == fail_edge.block && fail_edge.args.len() == 1 => {
            let want_true2 = !const_bool(func, info, fail_edge.args[0])?;
            let cond2 = args[0];
            let test2 = (func.ty(cond2) == Type::I8).then(|| match_test(func, info, cond2, Some(v), Some(mid))).flatten()?;
            if !body_clean(func, info, mid, &|i| test2.used.contains(&Some(i)), Some(cond2)) {
                return None;
            }
            let range = intersect_ranges(ty, r1, range_of_test(ty, test2.kind, want_true2)?)?;
            Arm { range, hit: Edge { raw: *target, block: *target, args: Vec::new(), synth_bool: Some(want_true2) } }
        }
        _ => return None,
    };
    extra.push(mid);
    Some((arm, extra))
}

pub(super) fn range_fits(r: Range) -> bool {
    r.lo <= r.hi || (r.signed_ok && r.s_hi.abs_diff(r.s_lo) < 256)
}

fn pick_arm_direction(
    func: &Function,
    info: &Info,
    test: &BlockTest,
    (t_edge, f_edge, t_cont, f_cont): (Edge, Edge, bool, bool),
    (preferred_hit, terminates): (Option<&Edge>, bool),
) -> Option<ArmMatch> {
    let ty = func.ty(test.val);
    let t_range = range_of_test(ty, test.kind, true).filter(|&r| range_fits(r));
    let f_range = range_of_test(ty, test.kind, false).filter(|&r| range_fits(r));
    let pick_true = match (t_range, f_range) {
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (Some(_), Some(_)) if f_cont != t_cont => f_cont,
        (Some(tr), Some(fr)) => {
            let p_t = preferred_hit.is_some_and(|p| same_edge(func, info, p, &t_edge));
            let p_f = preferred_hit.is_some_and(|p| same_edge(func, info, p, &f_edge));
            let score = |r: Range| if r.lo <= r.hi { r.hi - r.lo } else { r.s_hi.abs_diff(r.s_lo) };
            if p_t != p_f { p_t } else { score(tr) <= score(fr) }
        }
        (None, None) => return None,
    };
    let range = if pick_true { t_range? } else { f_range? };
    let (hit, next) = if pick_true { (t_edge, f_edge) } else { (f_edge, t_edge) };
    Some(ArmMatch { val: test.val, arm: Arm { range, hit }, next, extra_blocks: Vec::new(), used: test.used, terminates })
}

fn match_arm(
    func: &Function,
    info: &Info,
    ctx @ (b0, _, _): (usize, &[bool], &[usize]),
    (b, expected_val, preferred_hit): (usize, Option<Value>, Option<&Edge>),
) -> Option<ArmMatch> {
    let is_head = b == b0;
    let last = *func.blocks[b].insts.last()?;
    match &func.insts[last.index()].op {
        Op::Brif(cond, t_blk, t_args, f_blk, f_args) => {
            let test = match_test(func, info, *cond, expected_val, (!is_head).then_some(b))?;
            if !is_head && !body_clean(func, info, b, &|i| test.used.contains(&Some(i)), None) {
                return None;
            }
            let ty = func.ty(test.val);
            let (t_edge, f_edge) = (edge_of(func, *t_blk, t_args), edge_of(func, *f_blk, f_args));
            if let TestKind::Cmp(cc @ (IntCC::Equal | IntCC::NotEqual), _) = test.kind {
                let want_true = cc == IntCC::Equal;
                let range = range_of_test(ty, test.kind, want_true)?;
                let (hit, next) = if want_true { (t_edge, f_edge) } else { (f_edge, t_edge) };
                return Some(ArmMatch { val: test.val, arm: Arm { range, hit }, next, extra_blocks: Vec::new(), used: test.used, terminates: false });
            }
            for (want, hit_e, next_e) in [(true, &t_edge, &f_edge), (false, &f_edge, &t_edge)] {
                if let Some((arm, extra_blocks)) = match_two_block_range(func, info, ctx, (test.val, ty, test.kind, want), (hit_e, next_e)) {
                    return Some(ArmMatch { val: test.val, arm, next: next_e.clone(), extra_blocks, used: test.used, terminates: false });
                }
            }
            let t_cont = can_follow_into(func, info, ctx, &t_edge).is_some_and(|(tb, _)| peek_chain(func, info, tb, test.val));
            let f_cont = can_follow_into(func, info, ctx, &f_edge).is_some_and(|(fb, _)| peek_chain(func, info, fb, test.val));
            pick_arm_direction(func, info, &test, (t_edge, f_edge, t_cont, f_cont), (preferred_hit, false))
        }
        Op::Jump(target, args) if !is_head && args.len() == 1 && func.blocks[target.index()].params.len() == 1 && func.ty(args[0]) == Type::I8 => {
            let cond = args[0];
            let test = match_test(func, info, cond, Some(expected_val?), Some(b))?;
            if !body_clean(func, info, b, &|i| test.used.contains(&Some(i)), Some(cond)) {
                return None;
            }
            let t_edge = Edge { raw: *target, block: *target, args: Vec::new(), synth_bool: Some(true) };
            let f_edge = Edge { raw: *target, block: *target, args: Vec::new(), synth_bool: Some(false) };
            pick_arm_direction(func, info, &test, (t_edge, f_edge, false, false), (preferred_hit, true))
        }
        _ => None,
    }
}

fn peek_chain(func: &Function, info: &Info, b: usize, v: Value) -> bool {
    let Some(&last) = func.blocks[b].insts.last() else { return false };
    match &func.insts[last.index()].op {
        Op::Brif(cond, ..) => match_test(func, info, *cond, Some(v), Some(b)).is_some_and(|t| body_clean(func, info, b, &|i| t.used.contains(&Some(i)), None)),
        Op::Jump(_, args) if args.len() == 1 => {
            func.ty(args[0]) == Type::I8
                && match_test(func, info, args[0], Some(v), Some(b)).is_some_and(|t| body_clean(func, info, b, &|i| t.used.contains(&Some(i)), Some(args[0])))
        }
        _ => false,
    }
}

pub(super) fn collect_hoisted_consts(func: &Function, info: &Info, chain: &[usize]) -> Vec<Inst> {
    let (mut consts, mut order): (HashMap<Value, (Inst, u32)>, Vec<Value>) = (HashMap::default(), Vec::new());
    for &b in chain {
        for &inst in &func.blocks[b].insts {
            let data = &func.insts[inst.index()];
            if matches!(data.op, Op::Const(_)) && let Some(res) = data.result && !consts.contains_key(&res) {
                consts.insert(res, (inst, 0));
                order.push(res);
            }
        }
    }
    let mut bump = |v: &Value| {
        if let Some(entry) = consts.get_mut(v) { entry.1 += 1; }
    };
    for &b in chain {
        let Some((&last, body)) = func.blocks[b].insts.split_last() else { continue };
        body.iter().for_each(|&i| func.insts[i.index()].op.for_operands(|op| bump(&op)));
        match &func.insts[last.index()].op {
            Op::Brif(cond, ..) => bump(cond),
            Op::Jump(_, args) => args.iter().for_each(&mut bump),
            _ => {}
        }
    }
    order.into_iter().filter(|res| consts[res].1 < info.uses[res.index()]).map(|res| consts[&res].0).collect()
}

pub(super) fn push_before_last(func: &mut Function, block: usize, inst: Inst) {
    let insts = &mut func.blocks[block].insts;
    insts.insert(insts.len() - 1, inst);
}

fn resolve_edge(func: &mut Function, b0: usize, bool_consts: &mut [Option<Value>; 2], edge: &mut Edge) {
    if let Some(b) = edge.synth_bool.take() {
        let val = *bool_consts[usize::from(b)].get_or_insert_with(|| {
            let v = func.value(Type::I8);
            let inst = func.add(Op::Const(i64::from(b)), Some(v));
            push_before_last(func, b0, inst);
            v
        });
        edge.args = vec![val];
    }
}

pub(super) fn clean_dead_head_insts(func: &mut Function, info: &Info, b0: usize, used: &[Inst]) {
    for &inst in used {
        let Some(res) = func.insts[inst.index()].result else { break };
        if info.def_block[res.index()] != Some(b0) || info.uses[res.index()] != 1 {
            break;
        }
        func.blocks[b0].insts.retain(|&i| i != inst);
        func.insts[inst.index()].op = Op::Trap;
    }
}

fn collect_chain(func: &Function, info: &Info, b0: usize, cleared: &[bool]) -> Option<Chain> {
    let head = match_arm(func, info, (b0, cleared, &[]), (b0, None, None))?;
    let v = head.val;
    let mut chain_blocks = head.extra_blocks;
    let mut two_block_arms = u32::from(!chain_blocks.is_empty());
    let (mut arms, mut default_edge) = (vec![head.arm], head.next);
    while !head.terminates && let Some((cur, hops)) = can_follow_into(func, info, (b0, cleared, &chain_blocks), &default_edge) {
        let Some(m) = match_arm(func, info, (b0, cleared, &chain_blocks), (cur, Some(v), Some(&arms[0].hit))) else { break };
        chain_blocks.extend(hops);
        chain_blocks.push(cur);
        if !m.extra_blocks.is_empty() {
            chain_blocks.extend(m.extra_blocks);
            two_block_arms += 1;
        }
        arms.push(m.arm);
        default_edge = m.next;
        if m.terminates {
            break;
        }
    }
    (!chain_blocks.is_empty()).then_some(Chain { val: v, arms, default_edge, chain_blocks, two_block_arms, head_used: head.used })
}

fn clear_blocks(func: &mut Function, blocks: &[usize], hoisted: &[Inst], cleared: &mut [bool]) {
    for &b in blocks {
        for inst in func.blocks[b].insts.drain(..) {
            if !hoisted.contains(&inst) {
                func.insts[inst.index()].op = Op::Trap;
            }
        }
        cleared[b] = true;
    }
}

fn rewrite_chain(func: &mut Function, info: &Info, (b0, last0): (usize, Inst), mut ch: Chain, cleared: &mut [bool]) -> bool {
    let ty = func.ty(ch.val);
    let single_mod = ch.arms.len() == 1 && ch.two_block_arms == 1 && ch.arms[0].range.lo > ch.arms[0].range.hi;
    let mut runs = if single_mod { Vec::new() } else { build_runs(func, info, ty, &ch.arms) };
    let head_dead: Vec<Inst> = ch.head_used.into_iter().flatten().collect();
    if !single_mod && lut::try_emit_runs_lut(func, info, (b0, last0), ch.val, &runs, &ch.default_edge, &head_dead) {
        let hoisted = collect_hoisted_consts(func, info, &ch.chain_blocks);
        hoisted.iter().for_each(|&inst| push_before_last(func, b0, inst));
        clear_blocks(func, &ch.chain_blocks, &hoisted, cleared);
        cleared[b0] = true;
        return true;
    }
    if !single_mod {
        let covered: u64 = runs.iter().map(|r| r.hi.saturating_sub(r.lo).saturating_add(1)).sum();
        if covered < 4 && ch.two_block_arms == 0 && runs.len() >= ch.arms.len() && !can_bitmask_runs(func, info, ty, &runs) {
            return false;
        }
    }
    let hoisted = collect_hoisted_consts(func, info, &ch.chain_blocks);
    hoisted.iter().for_each(|&inst| push_before_last(func, b0, inst));
    let mut bool_consts = [None; 2];
    ch.arms.iter_mut().for_each(|a| resolve_edge(func, b0, &mut bool_consts, &mut a.hit));
    runs.iter_mut().for_each(|r| resolve_edge(func, b0, &mut bool_consts, &mut r.hit));
    resolve_edge(func, b0, &mut bool_consts, &mut ch.default_edge);
    clean_dead_head_insts(func, info, b0, &head_dead);
    let mut cache = Vec::new();
    let default_blk = forward_cached(func, &mut cache, ch.default_edge.block, ch.default_edge.args);
    if single_mod {
        let r = ch.arms[0].range;
        emit_sub_range(func, Some((b0, last0)), (ch.val, ty), (r.s_lo, r.s_hi.abs_diff(r.s_lo)), (&ch.arms[0].hit, default_blk));
    } else if !try_signed_br_table(func, (b0, last0), (ch.val, ty), &ch.arms, (default_blk, &mut cache)) {
        emit_node(func, info, Some((b0, last0)), (ch.val, ty), &runs, (default_blk, &mut cache));
    }
    clear_blocks(func, &ch.chain_blocks, &hoisted, cleared);
    cleared[b0] = true;
    true
}

fn try_rewrite_br_table(func: &mut Function, info: &Info, (b0, last0): (usize, Inst), cleared: &mut [bool]) -> bool {
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
            arms.push(Arm { range: make_range(ty, u, u), hit });
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
    if runs.len() > 2 && !can_bitmask_runs(func, info, ty, &runs) && !(runs.len() <= 4 && can_br_table(&runs).is_none()) {
        return false;
    }
    let mut cache = Vec::new();
    let default_target = forward_cached(func, &mut cache, default_edge.block, default_edge.args);
    emit_node(func, info, Some((b0, last0)), (idx, ty), &runs, (default_target, &mut cache));
    cleared[b0] = true;
    true
}

pub(crate) fn run(func: &mut Function) -> Domain {
    if func.coroutine.is_some() { return Domain::NONE; }
    let mut info = analyze(func);
    let initial_blocks = func.blocks.len();
    let mut cleared = vec![false; initial_blocks];
    let mut changed = false;
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
        let Some(&last0) = func.blocks[b0].insts.last() else { continue };
        if let Some(t) = tree::collect_tree(func, &info, b0, &cleared).filter(tree::Tree::is_forked)
            && tree::rewrite_tree(func, &info, (b0, last0), t, &mut cleared)
        {
            changed = true;
            continue;
        }
        if let Some(ch) = collect_chain(func, &info, b0, &cleared)
            && rewrite_chain(func, &info, (b0, last0), ch, &mut cleared)
        {
            changed = true;
            continue;
        }
        if let Some(t) = tree::collect_tree(func, &info, b0, &cleared)
            && tree::rewrite_tree(func, &info, (b0, last0), t, &mut cleared)
        {
            changed = true;
        } else if try_rewrite_expr(func, &info, (b0, last0), &mut cleared) || try_rewrite_br_table(func, &info, (b0, last0), &mut cleared) {
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
