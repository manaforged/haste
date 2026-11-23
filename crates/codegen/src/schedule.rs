mod branches;
mod copies;
mod products;
mod sink;

use std::time::Instant;

use crate::ir::{Binary, Block, Domain, Function, Inst, IntCC, Op, Type, Value};
use crate::{Effort, cse, fill, fold, ifconv, licm, loads, lsr, merge, promote, returns, rotate, switch, tailrec, thread, unroll};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Pass {
    Fill = 0,
    Fold = 1,
    Merge = 2,
    Promote = 3,
    Cse = 4,
    Loads = 5,
    Thread = 6,
    IfConv = 7,
    Rotate = 8,
    Unroll = 9,
    Licm = 10,
    Switch = 11,
    Returns = 12,
    TailRec = 13,
    Lsr = 14,
}

impl Pass {
    pub(crate) const ALL: &[Self] = &[
        Self::Fill,
        Self::Fold,
        Self::Merge,
        Self::Promote,
        Self::Cse,
        Self::Loads,
        Self::Thread,
        Self::IfConv,
        Self::Rotate,
        Self::Unroll,
        Self::Licm,
        Self::Switch,
        Self::Returns,
        Self::TailRec,
        Self::Lsr,
    ];

    pub(crate) const COUNT: usize = Self::ALL.len();

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Fill => "fill",
            Self::Fold => "fold",
            Self::Merge => "merge",
            Self::Promote => "promote",
            Self::Cse => "cse",
            Self::Loads => "loads",
            Self::Thread => "thread",
            Self::IfConv => "ifconv",
            Self::Rotate => "rotate",
            Self::Unroll => "unroll",
            Self::Licm => "licm",
            Self::Switch => "switch",
            Self::Returns => "returns",
            Self::TailRec => "tailrec",
            Self::Lsr => "lsr",
        }
    }
}

struct Rule {
    pass: Pass,
    reads: Domain,
}

struct Plan<'t> {
    moves: promote::Moves,
    effort: Effort,
    target: &'t crate::Target,
    rules: &'static [Rule],
    budget: usize,
    finish: &'static [Pass],
}

static INIT: &[Pass] = &[Pass::Fill, Pass::Fold, Pass::Merge];

static QUICK_RULES: &[Rule] = &[
    Rule { pass: Pass::Promote, reads: Domain::CFG.union(Domain::MEM).union(Domain::VALUES) },
    Rule { pass: Pass::Fold, reads: Domain::VALUES.union(Domain::CFG) },
    Rule { pass: Pass::Merge, reads: Domain::CFG.union(Domain::VALUES) },
];
static QUICK_FINISH: &[Pass] = &[Pass::Returns];

static FULL_RULES: &[Rule] = &[
    Rule { pass: Pass::Fill, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Promote, reads: Domain::CFG.union(Domain::MEM).union(Domain::VALUES) },
    Rule { pass: Pass::Fold, reads: Domain::VALUES.union(Domain::CFG) },
    Rule { pass: Pass::Merge, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Loads, reads: Domain::CFG.union(Domain::MEM).union(Domain::VALUES) },
    Rule { pass: Pass::Cse, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Thread, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::IfConv, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Rotate, reads: Domain::CFG.union(Domain::VALUES).union(Domain::MEM) },
    Rule { pass: Pass::Unroll, reads: Domain::CFG.union(Domain::VALUES).union(Domain::MEM) },
    Rule { pass: Pass::Licm, reads: Domain::CFG.union(Domain::VALUES).union(Domain::MEM) },
    Rule { pass: Pass::Lsr, reads: Domain::CFG.union(Domain::VALUES) },
];
static FULL_FINISH: &[Pass] = &[Pass::Switch, Pass::Returns, Pass::TailRec];
static INLINE_FINISH: &[Pass] = &[Pass::Returns];

fn run_pass(func: &mut Function, pass: Pass, moves: promote::Moves, effort: Effort, target: &crate::Target) -> Domain {
    let tracing = crate::trace::enabled();
    let (before_blocks, before_insts) = if tracing { (func.blocks.len(), func.size()) } else { (0, 0) };
    let start = tracing.then(Instant::now);
    let changed = match pass {
        Pass::Fill => fill::run(func),
        Pass::Fold => fold::run(func, effort),
        Pass::Merge => merge::run(func),
        Pass::Promote => {
            let did = promote::step(func, moves, effort);
            if !did.is_empty() && func.slots.len() > 64 {
                crate::ir::compact_slots(func);
            }
            did
        }
        Pass::Cse => cse::run(func),
        Pass::Loads => loads::run(func, moves == promote::Moves::Split, target),
        Pass::Thread => thread::run(func),
        Pass::IfConv => ifconv::run(func),
        Pass::Rotate => rotate::run(func),
        Pass::Unroll => unroll::run(func, target),
        Pass::Licm => licm::run(func),
        Pass::Lsr => lsr::run(func),
        Pass::Switch => switch::run(func, target),
        Pass::Returns => returns::run(func),
        Pass::TailRec => {
            let did = tailrec::run(func, target);
            if !did.is_empty() {
                fold::compact(func);
            }
            did
        }
    };
    if changed.contains(Domain::CFG) {
        func.bump_cfg();
    }
    if let Some(start) = start {
        crate::trace::after_pass(pass, before_blocks, before_insts, changed, start.elapsed(), func);
    }
    changed
}

struct PassState {
    fold_clean: bool,
    merge_clean: bool,
    loops_rev: Option<u32>,
    has_loops: bool,
}

impl PassState {
    fn new() -> Self {
        Self {
            fold_clean: false,
            merge_clean: false,
            loops_rev: None,
            has_loops: false,
        }
    }

    fn has_loops(&mut self, func: &Function) -> bool {
        if self.loops_rev != Some(func.cfg_rev) {
            self.loops_rev = Some(func.cfg_rev);
            self.has_loops = detect_loops(func);
        }
        self.has_loops
    }

    fn after_pass(&mut self, changed: Domain) {
        if !changed.is_empty() {
            self.fold_clean = false;
            self.merge_clean = false;
        }
    }
}

/// Returns whether any live instruction in `func.blocks` satisfies `pred`.
pub(crate) fn has_live_op(func: &Function, mut pred: impl FnMut(&Op) -> bool) -> bool {
    func.blocks.iter().any(|block| block.insts.iter().any(|&inst| pred(&func.insts[inst.index()].op)))
}

fn has_small_cycle(func: &Function, entry: Block) -> bool {
    let n = func.blocks.len();
    if entry.index() >= n {
        return false;
    }
    let mut state = [0_u8; 64];
    let mut stack = [0_u8; 64];
    let mut cursor = [0_usize; 64];
    state[entry.index()] = 1;
    stack[0] = entry.index() as u8;
    let mut sp = 1_usize;
    while sp > 0 {
        let b = stack[sp - 1] as usize;
        let idx = cursor[sp - 1];
        let Some(target) = func.successors(Block(b as u32)).nth(idx) else {
            state[b] = 2;
            sp -= 1;
            continue;
        };
        cursor[sp - 1] = idx + 1;
        let t = target.index();
        if t < n {
            if state[t] == 1 {
                return true;
            }
            if state[t] == 0 {
                state[t] = 1;
                stack[sp] = t as u8;
                cursor[sp] = 0;
                sp += 1;
            }
        }
    }
    false
}

fn detect_loops(func: &Function) -> bool {
    let Some(entry) = func.entry else { return false };
    if func.blocks.len() <= 1 {
        return func.successors(entry).any(|t| t == entry);
    }
    if func.blocks.len() <= 64 && !has_small_cycle(func, entry) {
        return false;
    }
    !func.cfg().loops().list.is_empty()
}

fn should_run(func: &Function, pass: Pass, state: &mut PassState) -> bool {
    let multi = func.blocks.len() > 1;
    match pass {
        Pass::Fold => !state.fold_clean,
        Pass::Merge => multi && !state.merge_clean,
        Pass::Thread | Pass::IfConv | Pass::Returns => multi,
        Pass::Switch => has_live_op(func, |op| matches!(op, Op::Select(..) | Op::Brif(..) | Op::BrTable(..))),
        Pass::Rotate | Pass::Unroll | Pass::Licm | Pass::Lsr => state.has_loops(func),
        Pass::Promote => !func.slots.is_empty(),
        Pass::Fill | Pass::Cse | Pass::Loads | Pass::TailRec => true,
    }
}

fn mark_dirty(rules: &[Rule], dirty: &mut [Domain; Pass::COUNT], source: Pass, changed: Domain) {
    for other in rules {
        if other.pass != source {
            dirty[other.pass as usize] |= changed;
        }
    }
    if source == Pass::Promote {
        dirty[source as usize] |= changed;
    }
}

fn run_step(func: &mut Function, plan: &Plan, state: &mut PassState, pass: Pass) -> Domain {
    if !should_run(func, pass, state) {
        return Domain::NONE;
    }
    let changed = run_pass(func, pass, plan.moves, plan.effort, plan.target);
    state.after_pass(changed);
    changed
}

fn run_fixpoint(func: &mut Function, plan: &Plan, state: &mut PassState) {
    let mut dirty = [Domain::NONE; Pass::ALL.len()];
    for rule in plan.rules {
        dirty[rule.pass as usize] = rule.reads;
    }
    let mut remaining = plan.budget;
    while remaining > 0 {
        let mut ran = false;
        for rule in plan.rules {
            if remaining == 0 {
                break;
            }
            let idx = rule.pass as usize;
            if (dirty[idx] & rule.reads).is_empty() {
                continue;
            }
            if rule.pass == Pass::Lsr && !dirty[Pass::Loads as usize].is_empty() {
                continue;
            }
            dirty[idx] = Domain::NONE;
            remaining -= 1;
            ran = true;
            let changed = run_step(func, plan, state, rule.pass);
            if !changed.is_empty() {
                mark_dirty(plan.rules, &mut dirty, rule.pass, changed);
            }
        }
        if !ran {
            break;
        }
    }
}

fn strip_assumes(func: &mut Function) -> bool {
    let Function { blocks, insts, .. } = func;
    let mut stripped = false;
    for block in blocks {
        block.insts.retain(|inst| {
            let assume = matches!(insts[inst.index()].op, Op::Assume(_));
            if assume {
                insts[inst.index()].op = Op::Trap;
            }
            stripped |= assume;
            !assume
        });
    }
    stripped
}

/// Whether `inst` computes a branch condition: a compare, or an `I8` and/or of conditions.
pub(crate) fn is_condition(func: &Function, inst: Inst) -> bool {
    let data = &func.insts[inst.index()];
    matches!(data.op, Op::Icmp(..) | Op::Fcmp(..))
        || (matches!(data.op, Op::Binary(Binary::Band | Binary::Bor, ..)) && data.result.is_some_and(|r| func.ty(r) == Type::I8))
}

fn has_sinkable_compare(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        if block.insts.len() < 3 {
            return false;
        }
        let Some(first_cmp) = block.insts[..block.insts.len() - 2]
            .iter()
            .position(|&inst| is_condition(func, inst))
        else {
            return false;
        };
        block.insts[first_cmp + 2..]
            .iter()
            .any(|&inst| matches!(func.insts[inst.index()].op, Op::Brif(..) | Op::Select(..)))
    })
}

/// Moves each single-use compare, or `I8` and/or of conditions, read by a `Brif` or `Select` of its own block to sit directly before that reader.
pub(crate) fn sink_compares(func: &mut Function) -> bool {
    if !has_sinkable_compare(func) {
        return false;
    }
    let uses = func.use_counts();
    let mut moved = false;
    let mut blocks = std::mem::take(&mut func.blocks);
    for block in &mut blocks {
        let mut at = 0;
        while at < block.insts.len() {
            let data = &func.insts[block.insts[at].index()];
            let Some(cond) = data.result.filter(|c| uses[c.index()] == 1 && is_condition(func, block.insts[at])) else {
                at += 1;
                continue;
            };
            let reads = |inst: &crate::ir::Inst| {
                let mut hit = false;
                func.insts[inst.index()].op.for_operands(|v| hit |= v == cond);
                hit
            };
            match block.insts[at + 1..].iter().position(reads) {
                Some(gap) if gap > 0 && matches!(func.insts[block.insts[at + 1 + gap].index()].op, Op::Brif(c, ..) | Op::Select(c, ..) if c == cond) => {
                    let inst = block.insts.remove(at);
                    block.insts.insert(at + gap, inst);
                    moved = true;
                }
                _ => at += 1,
            }
        }
    }
    func.blocks = blocks;
    moved
}

fn build_plan<'t>(moves: promote::Moves, target: &'t crate::Target) -> Plan<'t> {
    let effort = target.effort;
    match effort {
        Effort::Quick => Plan {
            moves: promote::Moves::Words,
            effort,
            target,
            rules: QUICK_RULES,
            budget: 6,
            finish: QUICK_FINISH,
        },
        Effort::Full if moves == promote::Moves::Guided => Plan {
            moves,
            effort,
            target,
            rules: FULL_RULES,
            budget: 32,
            finish: INLINE_FINISH,
        },
        Effort::Full => Plan {
            moves,
            effort,
            target,
            rules: FULL_RULES,
            budget: 32,
            finish: FULL_FINISH,
        },
    }
}

fn rerun_after_rewrite(func: &mut Function, plan: &Plan, state: &mut PassState) {
    state.fold_clean = false;
    state.merge_clean = false;
    run_fixpoint(func, plan, state);
}

fn run_late_lowering(func: &mut Function, plan: &Plan, state: &mut PassState) {
    if plan.moves == promote::Moves::Guided {
        return;
    }
    if (plan.effort == Effort::Full || func.elide_drops) && func.elide_dead_unwind_drops_post_init(true) {
        *state = PassState::new();
        run_fixpoint(func, plan, state);
    }
    if strip_assumes(func) {
        rerun_after_rewrite(func, plan, state);
    }
    if plan.effort == Effort::Full {
        let wide_changed = (plan.target.arch == crate::Arch::X64 && crate::native::divide_by_word(func)) | crate::pairs::split(func);
        if wide_changed {
            rerun_after_rewrite(func, plan, state);
        }
        if crate::slp::run(func) {
            rerun_after_rewrite(func, plan, state);
        }
    }
}

fn invert_unlikely_branches(func: &mut Function) {
    let cfg = func.cfg();
    let loops = cfg.loops();
    if loops.list.is_empty() {
        return;
    }
    let cold = crate::live::cold(func);
    let mut is_latch = vec![false; func.blocks.len()];
    loops.list.iter().flat_map(|each| &each.latches).for_each(|l| is_latch[l.index()] = true);
    let mut cands = Vec::new();
    for b in 0..func.blocks.len() {
        let block = Block(b as u32);
        let depth = loops.depth_of(block);
        if depth == 0 {
            continue;
        }
        let Some(&term) = func.blocks[b].insts.last() else { continue };
        let Op::Brif(cond, then_b, _, else_b, _) = func.insts[term.index()].op else { continue };
        if then_b != else_b
            && !cold[then_b.index()]
            && !cold[else_b.index()]
            && is_latch[then_b.index()] == is_latch[else_b.index()]
            && loops.depth_of(then_b) == loops.depth_of(else_b)
            && loops.depth_of(then_b) >= depth
        {
            cands.push((term, cond));
        }
    }
    drop(cfg);
    if cands.is_empty() {
        return;
    }
    let mut uses = vec![0_u32; func.types.len()];
    let mut defs = vec![None; func.types.len()];
    for &i in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[i.index()];
        data.op.for_operands(|v| uses[v.index()] += 1);
        if let Some(r) = data.result { defs[r.index()] = Some(i); }
    }
    let const_val = |func: &Function, v: Value| defs[v.index()].and_then(|inst| match func.insts[inst.index()].op { Op::Const(k) => Some(k), _ => None });
    let mut changed = false;
    for (term, cond) in cands {
        if uses[cond.index()] != 1 {
            continue;
        }
        let Some(cmp_inst) = defs[cond.index()] else { continue };
        let Op::Icmp(IntCC::Equal, x, y) = func.insts[cmp_inst.index()].op else { continue };
        if !matches!((const_val(func, x), const_val(func, y)), (Some(k), None) | (None, Some(k)) if k != 0 && k != 1) {
            continue;
        }
        func.insts[cmp_inst.index()].op = Op::Icmp(IntCC::NotEqual, x, y);
        if let Op::Brif(_, ref mut t, ref mut ta, ref mut e, ref mut ea) = func.insts[term.index()].op {
            std::mem::swap(t, e);
            std::mem::swap(ta, ea);
        }
        changed = true;
    }
    if changed {
        func.bump_cfg();
    }
}

fn run_backend_prep(func: &mut Function, plan: &Plan) {
    if plan.moves == promote::Moves::Guided {
        return;
    }
    if plan.effort == Effort::Full {
        if has_live_op(func, |op| matches!(op, Op::Binary(Binary::Isub | Binary::Bxor, ..))) {
            crate::algebra::compare_offsets(func);
        }
        sink::run(func, plan.moves == promote::Moves::Split);
    }
    let x64_full = plan.target.arch == crate::Arch::X64 && plan.effort == Effort::Full;
    if x64_full {
        branches::split_conditions(func);
    }
    sink_compares(func);
    if plan.effort == Effort::Full {
        invert_unlikely_branches(func);
    }
    if x64_full {
        products::adjoin(func);
        copies::pair(func);
    }
}

pub(crate) fn run(func: &mut Function, moves: promote::Moves, target: &crate::Target) {
    let plan = build_plan(moves, target);
    let mut state = PassState::new();
    run_step(func, &plan, &mut state, INIT[0]);
    let fold_init = run_step(func, &plan, &mut state, INIT[1]);
    let merge_init = run_step(func, &plan, &mut state, INIT[2]);
    state.fold_clean = fold_init.is_empty() && merge_init.is_empty();
    state.merge_clean = merge_init.is_empty();
    if moves != promote::Moves::Guided
        && (plan.effort == Effort::Full || func.elide_drops)
        && func.elide_dead_unwind_drops_post_init(false)
    {
        state = PassState::new();
    }
    if moves == promote::Moves::Split && plan.effort == Effort::Full && crate::loads::elide_early(func) {
        state.fold_clean = false;
        state.merge_clean = false;
    }
    run_fixpoint(func, &plan, &mut state);
    run_late_lowering(func, &plan, &mut state);
    for &pass in plan.finish {
        let changed = run_step(func, &plan, &mut state, pass);
        if pass == Pass::Switch && !changed.is_empty() && plan.effort == Effort::Full {
            run_step(func, &plan, &mut state, Pass::Fold);
            run_step(func, &plan, &mut state, Pass::Cse);
            run_step(func, &plan, &mut state, Pass::Merge);
            if state.has_loops(func) {
                run_step(func, &plan, &mut state, Pass::Licm);
            }
        }
    }
    run_backend_prep(func, &plan);
}
