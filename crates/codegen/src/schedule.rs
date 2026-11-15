mod sink;

use std::time::Instant;

use crate::ir::{Binary, Block, Domain, Function, Op};
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
static QUICK_FINISH: &[Pass] = &[Pass::Cse, Pass::Returns];

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

fn run_pass(func: &mut Function, pass: Pass, moves: promote::Moves, effort: Effort, target: &crate::Target) -> Domain {
    let tracing = crate::trace::enabled();
    let (before_blocks, before_insts) = if tracing { (func.blocks.len(), func.size()) } else { (0, 0) };
    let start = tracing.then(Instant::now);
    let changed = match pass {
        Pass::Fill => fill::run(func),
        Pass::Fold => fold::run(func, effort),
        Pass::Merge => merge::run(func),
        Pass::Promote => promote::step(func, moves),
        Pass::Cse => cse::run(func),
        Pass::Loads => loads::run(func, target),
        Pass::Thread => thread::run(func),
        Pass::IfConv => ifconv::run(func),
        Pass::Rotate => rotate::run(func),
        Pass::Unroll => unroll::run(func),
        Pass::Licm => licm::run(func),
        Pass::Lsr => lsr::run(func),
        Pass::Switch => switch::run(func),
        Pass::Returns => returns::run(func),
        Pass::TailRec => {
            let did = tailrec::run(func);
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

fn has_live_op(func: &Function, mut pred: impl FnMut(&Op) -> bool) -> bool {
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
        Pass::Fill | Pass::Promote | Pass::Cse | Pass::Loads | Pass::TailRec => true,
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

fn has_sinkable_compare(func: &Function) -> bool {
    func.blocks.iter().any(|block| {
        if block.insts.len() < 3 {
            return false;
        }
        let Some(first_cmp) = block.insts[..block.insts.len() - 2]
            .iter()
            .position(|&inst| matches!(func.insts[inst.index()].op, Op::Icmp(..) | Op::Fcmp(..)))
        else {
            return false;
        };
        block.insts[first_cmp + 2..]
            .iter()
            .any(|&inst| matches!(func.insts[inst.index()].op, Op::Brif(..) | Op::Select(..)))
    })
}

/// Moves each single-use compare read by a `Brif` or `Select` of its own block to sit directly before that reader.
pub(crate) fn sink_compares(func: &mut Function) {
    if !has_sinkable_compare(func) {
        return;
    }
    let uses = func.use_counts();
    for block in &mut func.blocks {
        let mut at = 0;
        while at < block.insts.len() {
            let data = &func.insts[block.insts[at].index()];
            let Some(cond) = data.result.filter(|c| uses[c.index()] == 1 && matches!(data.op, Op::Icmp(..) | Op::Fcmp(..))) else {
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
                }
                _ => at += 1,
            }
        }
    }
}

pub(crate) fn run(func: &mut Function, moves: promote::Moves, target: &crate::Target) {
    let effort = target.effort;
    let plan = match effort {
        Effort::Quick => Plan {
            moves: promote::Moves::Words,
            effort,
            target,
            rules: QUICK_RULES,
            budget: 6,
            finish: QUICK_FINISH,
        },
        Effort::Full => Plan {
            moves,
            effort,
            target,
            rules: FULL_RULES,
            budget: 32,
            finish: FULL_FINISH,
        },
    };
    let mut state = PassState::new();
    run_step(func, &plan, &mut state, INIT[0]);
    let fold_init = run_step(func, &plan, &mut state, INIT[1]);
    let merge_init = run_step(func, &plan, &mut state, INIT[2]);
    state.fold_clean = fold_init.is_empty() && merge_init.is_empty();
    state.merge_clean = merge_init.is_empty();
    run_fixpoint(func, &plan, &mut state);
    if moves != promote::Moves::Guided && strip_assumes(func) {
        state.fold_clean = false;
        state.merge_clean = false;
        run_fixpoint(func, &plan, &mut state);
    }
    for &pass in plan.finish {
        run_step(func, &plan, &mut state, pass);
    }
    if moves != promote::Moves::Guided && has_live_op(func, |op| matches!(op, Op::Binary(Binary::Isub | Binary::Bxor, ..))) {
        crate::algebra::compare_offsets(func);
    }
    if moves != promote::Moves::Guided {
        sink::run(func);
    }
    sink_compares(func);
}
