use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::fold::extend;
use crate::ir::{Binary, Block, BlockData, Convert, Function, Inst, IntCC, Op, Type, Value};

struct Info {
    cfg: Cfg,
    uses: Vec<u32>,
    consts: Vec<Option<i64>>,
    def_block: Vec<Option<usize>>,
    def_inst: Vec<Option<Inst>>,
}

#[derive(Clone, Copy)]
enum TestKind {
    Cmp(IntCC, i64),
    SubCmp { lo: u64, cc: IntCC, bound: u64 },
}

struct BlockTest {
    val: Value,
    kind: TestKind,
    used: [Option<Inst>; 2],
}

#[derive(Clone, Copy)]
struct Range {
    lo: u64,
    hi: u64,
    s_lo: i64,
    s_hi: i64,
    signed_ok: bool,
}

#[derive(Clone)]
struct Edge {
    raw: Block,
    block: Block,
    args: Vec<Value>,
    synth_bool: Option<bool>,
}

#[derive(Clone)]
struct Arm {
    range: Range,
    hit: Edge,
}

#[derive(Clone)]
struct Run {
    lo: u64,
    hi: u64,
    hit: Edge,
}

struct ArmMatch {
    val: Value,
    arm: Arm,
    next: Edge,
    extra_blocks: Vec<usize>,
    used: [Option<Inst>; 2],
    terminates: bool,
}

struct Chain {
    val: Value,
    arms: Vec<Arm>,
    default_edge: Edge,
    chain_blocks: Vec<usize>,
    two_block_arms: u32,
    head_used: [Option<Inst>; 2],
}

fn analyze(func: &Function) -> Info {
    let cfg = Cfg::new(func);
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

fn swap_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal | IntCC::NotEqual => cc,
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
    }
}

fn make_range(ty: Type, lo: u64, hi: u64) -> Range {
    let (s_lo, s_hi) = (extend(lo as i64, ty, true), extend(hi as i64, ty, true));
    Range { lo, hi, s_lo, s_hi, signed_ok: s_lo <= s_hi }
}

fn make_signed_range(ty: Type, s_lo: i64, s_hi: i64) -> Range {
    Range { lo: extend(s_lo, ty, false) as u64, hi: extend(s_hi, ty, false) as u64, s_lo, s_hi, signed_ok: true }
}

fn match_sub_offset(func: &Function, info: &Info, val: Value, expected_block: Option<usize>) -> Option<(Value, u64, Inst)> {
    let inst = info.def_inst.get(val.index()).copied().flatten()?;
    if expected_block.is_some_and(|b| info.def_block[val.index()] != Some(b)) || info.uses.get(val.index()).copied() != Some(1) {
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

fn match_test(func: &Function, info: &Info, cond: Value, expected_val: Option<Value>, expected_block: Option<usize>) -> Option<BlockTest> {
    if expected_block.is_some_and(|b| info.def_block[cond.index()] != Some(b) || info.uses.get(cond.index()).copied() != Some(1)) {
        return None;
    }
    let cond_inst = info.def_inst.get(cond.index()).copied().flatten()?;
    let Op::Icmp(cc, a, b) = func.insts[cond_inst.index()].op else { return None };
    let (lhs, cc, imm) = match (info.consts.get(a.index()).copied().flatten(), info.consts.get(b.index()).copied().flatten()) {
        (_, Some(imm)) => (a, cc, imm),
        (Some(imm), None) => (b, swap_cc(cc), imm),
        _ => return None,
    };
    if func.ty(lhs).is_small_int() && expected_val.is_none_or(|ev| ev == lhs) {
        return Some(BlockTest { val: lhs, kind: TestKind::Cmp(cc, imm), used: [Some(cond_inst), None] });
    }
    if !cc.is_signed()
        && !matches!(cc, IntCC::Equal | IntCC::NotEqual)
        && let Some((src, lo, diff_inst)) = match_sub_offset(func, info, lhs, expected_block)
        && expected_val.is_none_or(|ev| ev == src)
    {
        let bound = extend(imm, func.ty(src), false) as u64;
        return Some(BlockTest { val: src, kind: TestKind::SubCmp { lo, cc, bound }, used: [Some(cond_inst), Some(diff_inst)] });
    }
    None
}

fn range_of_test(ty: Type, kind: TestKind, want_true: bool) -> Option<Range> {
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

fn intersect_ranges(ty: Type, r1: Range, r2: Range) -> Option<Range> {
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

fn body_clean(func: &Function, info: &Info, block: usize, used: [Option<Inst>; 2], jump_cond: Option<Value>) -> bool {
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
        let Some(res) = data.result.filter(|_| used.contains(&Some(inst))) else { return false };
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

fn can_enter(func: &Function, info: &Info, (b0, cleared, chain): (usize, &[bool], &[usize]), cur: usize) -> bool {
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

fn same_edge(func: &Function, info: &Info, a: &Edge, b: &Edge) -> bool {
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

fn edge_of(func: &Function, block: Block, args: &[Value]) -> Edge {
    let resolved = if args.is_empty() { func.jump_target(block) } else { block };
    Edge { raw: block, block: resolved, args: args.to_vec(), synth_bool: None }
}

fn can_follow_into(func: &Function, info: &Info, ctx: (usize, &[bool], &[usize]), edge: &Edge) -> Option<(usize, Vec<usize>)> {
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
            if !body_clean(func, info, mid, test2.used, None) {
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
            if !body_clean(func, info, mid, test2.used, Some(cond2)) {
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

fn range_fits(r: Range) -> bool {
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
            if !is_head && !body_clean(func, info, b, test.used, None) {
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
            if !body_clean(func, info, b, test.used, Some(cond)) {
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
        Op::Brif(cond, ..) => match_test(func, info, *cond, Some(v), Some(b)).is_some_and(|t| body_clean(func, info, b, t.used, None)),
        Op::Jump(_, args) if args.len() == 1 => {
            func.ty(args[0]) == Type::I8 && match_test(func, info, args[0], Some(v), Some(b)).is_some_and(|t| body_clean(func, info, b, t.used, Some(args[0])))
        }
        _ => false,
    }
}

fn collect_hoisted_consts(func: &Function, info: &Info, chain: &[usize]) -> Vec<Inst> {
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

fn push_before_last(func: &mut Function, block: usize, inst: Inst) {
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

fn forward_cached(func: &mut Function, cache: &mut Vec<((Block, Vec<Value>), Block)>, target: Block, args: Vec<Value>) -> Block {
    let target = if args.is_empty() { func.jump_target(target) } else { target };
    if args.is_empty() && func.blocks[target.index()].params.is_empty() {
        return target;
    }
    if let Some(&(_, blk)) = cache.iter().find(|((t, a), _)| *t == target && *a == args) {
        return blk;
    }
    let block = Block(func.blocks.len() as u32);
    let inst = func.add(Op::Jump(target, args.clone()), None);
    func.blocks.push(BlockData { params: Vec::new(), insts: vec![inst] });
    cache.push(((target, args), block));
    block
}

fn insert_interval(runs: &mut Vec<Run>, mut cur: u64, hi: u64, hit: Edge) {
    let (mut added, mut done) = (Vec::new(), false);
    for r in runs.iter() {
        if r.hi < cur { continue; }
        if r.lo > hi { break; }
        if cur < r.lo { added.push(Run { lo: cur, hi: r.lo - 1, hit: hit.clone() }); }
        if r.hi >= hi { done = true; break; }
        cur = r.hi + 1;
    }
    if !done && cur <= hi {
        added.push(Run { lo: cur, hi, hit });
    }
    if !added.is_empty() {
        runs.extend(added);
        runs.sort_unstable_by_key(|r| r.lo);
    }
}

fn build_runs(func: &Function, info: &Info, ty: Type, arms: &[Arm]) -> Vec<Run> {
    let (mut runs, u_max) = (Vec::new(), ty.unsigned_max());
    for arm in arms {
        if arm.range.lo <= arm.range.hi {
            insert_interval(&mut runs, arm.range.lo, arm.range.hi, arm.hit.clone());
        } else {
            insert_interval(&mut runs, arm.range.lo, u_max, arm.hit.clone());
            insert_interval(&mut runs, 0, arm.range.hi, arm.hit.clone());
        }
    }
    let mut merged: Vec<Run> = Vec::with_capacity(runs.len());
    for r in runs {
        if let Some(prev) = merged.last_mut() && prev.hi < u64::MAX && prev.hi + 1 == r.lo && same_edge(func, info, &prev.hit, &r.hit) {
            prev.hi = r.hi;
        } else {
            merged.push(r);
        }
    }
    merged
}

fn push_node(func: &mut Function, at: Option<(usize, Inst)>, insts: Vec<Inst>) -> Block {
    match at {
        Some((b, last)) => {
            func.insts[last.index()].op = Op::Trap;
            let dst = &mut func.blocks[b].insts;
            dst.pop();
            dst.extend(insts);
            Block(b as u32)
        }
        None => {
            let b = Block(func.blocks.len() as u32);
            func.blocks.push(BlockData { params: Vec::new(), insts });
            b
        }
    }
}

fn finish_br_table(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), (min, signed): (i64, bool), (default_blk, table): (Block, Vec<Block>)) -> Block {
    let mut insts = Vec::with_capacity(4);
    let idx = if min == 0 && !signed {
        v
    } else {
        let v64 = if ty == Type::I64 {
            v
        } else {
            let conv = if signed { Convert::Sextend } else { Convert::Uextend };
            let res = func.value(Type::I64);
            insts.push(func.add(Op::Convert(conv, v), Some(res)));
            res
        };
        if min == 0 {
            v64
        } else {
            let (min_val, sub) = (func.value(Type::I64), func.value(Type::I64));
            insts.push(func.add(Op::Const(min), Some(min_val)));
            insts.push(func.add(Op::Binary(Binary::Isub, v64, min_val), Some(sub)));
            sub
        }
    };
    insts.push(func.add(Op::BrTable(idx, default_blk, table), None));
    push_node(func, at, insts)
}

fn try_signed_br_table(
    func: &mut Function,
    at: (usize, Inst),
    (v, ty): (Value, Type),
    arms: &[Arm],
    (default_blk, cache): (Block, &mut Vec<((Block, Vec<Value>), Block)>),
) -> bool {
    if arms.iter().any(|a| !a.range.signed_ok || a.range.s_hi.abs_diff(a.range.s_lo) >= 256) {
        return false;
    }
    let (mut s_min, mut s_max, mut u_min, mut u_max, mut wrap) = (i64::MAX, i64::MIN, u64::MAX, 0_u64, false);
    for a in arms {
        (s_min, s_max) = (s_min.min(a.range.s_lo), s_max.max(a.range.s_hi));
        if a.range.lo > a.range.hi { wrap = true; } else { (u_min, u_max) = (u_min.min(a.range.lo), u_max.max(a.range.hi)); }
    }
    let s_span = s_max.abs_diff(s_min);
    let Some(s_len) = usize::try_from(s_span).ok().and_then(|s| s.checked_add(1)).filter(|&l| l <= 256 && (wrap || s_span < u_max - u_min)) else {
        return false;
    };
    let (mut table, mut filled, mut covered) = (vec![default_blk; s_len], vec![false; s_len], 0_u64);
    for a in arms {
        let target = forward_cached(func, cache, a.hit.block, a.hit.args.clone());
        let (start, end) = ((a.range.s_lo as i128 - s_min as i128) as usize, (a.range.s_hi as i128 - s_min as i128) as usize);
        for idx in start..=end {
            if !std::mem::replace(&mut filled[idx], true) {
                table[idx] = target;
                covered += 1;
            }
        }
    }
    let (mut s_runs, mut prev) = (0_u32, None);
    for (&blk, &is_hit) in table.iter().zip(&filled) {
        if is_hit && prev != Some(blk) { s_runs += 1; prev = Some(blk); } else if !is_hit { prev = None; }
    }
    if covered < 4 || covered * 5 < (s_len as u64) * 2 || s_runs < 3 {
        return false;
    }
    finish_br_table(func, Some(at), (v, ty), (s_min, true), (default_blk, table));
    true
}

fn can_br_table(runs: &[Run]) -> Option<(u64, usize)> {
    if runs.len() < 3 {
        return None;
    }
    let (min_lo, max_hi) = (runs[0].lo, runs[runs.len() - 1].hi);
    let table_len = usize::try_from(max_hi - min_lo).ok().and_then(|s| s.checked_add(1))?;
    let covered: u64 = runs.iter().map(|r| r.hi - r.lo + 1).sum();
    (table_len <= 256 && covered >= 4 && covered * 5 >= (table_len as u64) * 2).then_some((min_lo, table_len))
}

fn find_cluster(runs: &[Run]) -> Option<(usize, usize)> {
    let mut best: Option<(u64, usize, usize)> = None;
    for i in 0..runs.len() {
        let (min_lo, mut covered) = (runs[i].lo, 0_u64);
        for j in (i + 1)..=runs.len() {
            let last = &runs[j - 1];
            covered += last.hi - last.lo + 1;
            let span = last.hi - min_lo;
            if span >= 256 {
                break;
            }
            if j - i >= 3 && covered >= 4 && covered * 5 >= (span + 1) * 2 && best.is_none_or(|(bc, bj, bi)| (covered, j - i) > (bc, bj - bi)) {
                best = Some((covered, j, i));
            }
        }
    }
    best.map(|(_, j, i)| (i, j))
}

fn emit_sub_range(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), (lo, span): (i64, u64), (hit, otherwise): (&Edge, Block)) -> Block {
    let (lo_c, diff, span_c, le) = (func.value(ty), func.value(ty), func.value(ty), func.value(Type::I8));
    let insts = vec![
        func.add(Op::Const(extend(lo, ty, true)), Some(lo_c)),
        func.add(Op::Binary(Binary::Isub, v, lo_c), Some(diff)),
        func.add(Op::Const(extend(span as i64, ty, true)), Some(span_c)),
        func.add(Op::Icmp(IntCC::UnsignedLessThanOrEqual, diff, span_c), Some(le)),
        func.add(Op::Brif(le, hit.block, hit.args.clone(), otherwise, Vec::new()), None),
    ];
    push_node(func, at, insts)
}

fn emit_run_check(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), run: &Run, otherwise: Block) -> Block {
    let u_max = ty.unsigned_max();
    if run.lo == 0 && run.hi == u_max {
        let jump = func.add(Op::Jump(run.hit.block, run.hit.args.clone()), None);
        return push_node(func, at, vec![jump]);
    }
    if run.lo > 0 && run.hi < u_max && run.lo != run.hi {
        return emit_sub_range(func, at, (v, ty), (run.lo as i64, run.hi - run.lo), (&run.hit, otherwise));
    }
    let (cc, imm) = match (run.lo == run.hi, run.lo == 0) {
        (true, _) => (IntCC::Equal, run.lo),
        (false, true) => (IntCC::UnsignedLessThanOrEqual, run.hi),
        (false, false) => (IntCC::UnsignedGreaterThanOrEqual, run.lo),
    };
    let (c, cond) = (func.value(ty), func.value(Type::I8));
    let insts = vec![
        func.add(Op::Const(extend(imm as i64, ty, true)), Some(c)),
        func.add(Op::Icmp(cc, v, c), Some(cond)),
        func.add(Op::Brif(cond, run.hit.block, run.hit.args.clone(), otherwise, Vec::new()), None),
    ];
    push_node(func, at, insts)
}

fn emit_node(func: &mut Function, at: Option<(usize, Inst)>, (v, ty): (Value, Type), runs: &[Run], (default_blk, cache): (Block, &mut Vec<((Block, Vec<Value>), Block)>)) -> Block {
    if let Some((min, table_len)) = can_br_table(runs) {
        let mut table = vec![default_blk; table_len];
        for r in runs {
            let target = forward_cached(func, cache, r.hit.block, r.hit.args.clone());
            table[(r.lo - min) as usize..=(r.hi - min) as usize].fill(target);
        }
        return finish_br_table(func, at, (v, ty), (min as i64, false), (default_blk, table));
    }
    if runs.len() <= 2 {
        let otherwise = if runs.len() == 2 { emit_node(func, None, (v, ty), &runs[1..], (default_blk, cache)) } else { default_blk };
        return emit_run_check(func, at, (v, ty), &runs[0], otherwise);
    }
    let mid = match find_cluster(runs) {
        Some((start, _)) if start > 0 => start,
        Some((_, end)) => end,
        None => runs.len() / 2,
    };
    let left = emit_node(func, None, (v, ty), &runs[..mid], (default_blk, cache));
    let right = emit_node(func, None, (v, ty), &runs[mid..], (default_blk, cache));
    let (pivot, lt) = (func.value(ty), func.value(Type::I8));
    let insts = vec![
        func.add(Op::Const(extend(runs[mid].lo as i64, ty, true)), Some(pivot)),
        func.add(Op::Icmp(IntCC::UnsignedLessThan, v, pivot), Some(lt)),
        func.add(Op::Brif(lt, left, Vec::new(), right, Vec::new()), None),
    ];
    push_node(func, at, insts)
}

fn clean_dead_head_insts(func: &mut Function, info: &Info, b0: usize, used: [Option<Inst>; 2]) {
    for inst in used.into_iter().flatten() {
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

fn rewrite_chain(func: &mut Function, info: &Info, (b0, last0): (usize, Inst), mut ch: Chain, cleared: &mut [bool]) -> bool {
    let ty = func.ty(ch.val);
    let single_mod = ch.arms.len() == 1 && ch.two_block_arms == 1 && ch.arms[0].range.lo > ch.arms[0].range.hi;
    let mut runs = if single_mod { Vec::new() } else { build_runs(func, info, ty, &ch.arms) };
    if !single_mod {
        let covered: u64 = runs.iter().map(|r| r.hi.saturating_sub(r.lo).saturating_add(1)).sum();
        if covered < 4 && ch.two_block_arms == 0 && runs.len() >= ch.arms.len() {
            return false;
        }
    }
    let hoisted = collect_hoisted_consts(func, info, &ch.chain_blocks);
    hoisted.iter().for_each(|&inst| push_before_last(func, b0, inst));
    let mut bool_consts = [None; 2];
    ch.arms.iter_mut().for_each(|a| resolve_edge(func, b0, &mut bool_consts, &mut a.hit));
    runs.iter_mut().for_each(|r| resolve_edge(func, b0, &mut bool_consts, &mut r.hit));
    resolve_edge(func, b0, &mut bool_consts, &mut ch.default_edge);
    clean_dead_head_insts(func, info, b0, ch.head_used);
    let mut cache = Vec::new();
    let default_blk = forward_cached(func, &mut cache, ch.default_edge.block, ch.default_edge.args);
    if single_mod {
        let r = ch.arms[0].range;
        emit_sub_range(func, Some((b0, last0)), (ch.val, ty), (r.s_lo, r.s_hi.abs_diff(r.s_lo)), (&ch.arms[0].hit, default_blk));
    } else if !try_signed_br_table(func, (b0, last0), (ch.val, ty), &ch.arms, (default_blk, &mut cache)) {
        emit_node(func, Some((b0, last0)), (ch.val, ty), &runs, (default_blk, &mut cache));
    }
    for b in ch.chain_blocks {
        for inst in func.blocks[b].insts.drain(..) {
            if !hoisted.contains(&inst) {
                func.insts[inst.index()].op = Op::Trap;
            }
        }
        cleared[b] = true;
    }
    cleared[b0] = true;
    true
}

pub(crate) fn run(func: &mut Function) -> bool {
    let info = analyze(func);
    let initial_blocks = func.blocks.len();
    let mut cleared = vec![false; initial_blocks];
    let mut changed = false;
    for b0 in 0..initial_blocks {
        if cleared[b0] {
            continue;
        }
        let Some(&last0) = func.blocks[b0].insts.last() else { continue };
        let Some(ch) = collect_chain(func, &info, b0, &cleared) else { continue };
        changed |= rewrite_chain(func, &info, (b0, last0), ch, &mut cleared);
    }
    changed
}
