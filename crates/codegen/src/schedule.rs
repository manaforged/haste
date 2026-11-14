use std::time::Instant;

use crate::ir::{Domain, Function};
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

struct Plan {
    moves: promote::Moves,
    effort: Effort,
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

fn run_pass(func: &mut Function, pass: Pass, moves: promote::Moves, effort: Effort) -> Domain {
    let tracing = crate::trace::enabled();
    let (before_blocks, before_insts) = if tracing { (func.blocks.len(), func.size()) } else { (0, 0) };
    let start = tracing.then(Instant::now);
    let changed = match pass {
        Pass::Fill => fill::run(func),
        Pass::Fold => fold::run(func, effort),
        Pass::Merge => merge::run(func),
        Pass::Promote => promote::step(func, moves),
        Pass::Cse => cse::run(func),
        Pass::Loads => loads::run(func),
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

fn run_fixpoint(func: &mut Function, plan: &Plan) {
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
            let changed = run_pass(func, rule.pass, plan.moves, plan.effort);
            if !changed.is_empty() {
                for other in plan.rules {
                    if other.pass != rule.pass {
                        dirty[other.pass as usize] |= changed;
                    }
                }
                if rule.pass == Pass::Promote {
                    dirty[idx] |= changed;
                }
            }
        }
        if !ran {
            break;
        }
    }
}

pub(crate) fn run(func: &mut Function, moves: promote::Moves, effort: Effort) {
    let plan = match effort {
        Effort::Quick => Plan {
            moves: promote::Moves::Words,
            effort,
            rules: QUICK_RULES,
            budget: 6,
            finish: QUICK_FINISH,
        },
        Effort::Full => Plan {
            moves,
            effort,
            rules: FULL_RULES,
            budget: 32,
            finish: FULL_FINISH,
        },
    };
    for &pass in INIT {
        run_pass(func, pass, plan.moves, plan.effort);
    }
    run_fixpoint(func, &plan);
    for &pass in plan.finish {
        run_pass(func, pass, plan.moves, plan.effort);
    }
}
