use crate::ir::{Domain, Function};
use crate::{Effort, cse, fill, fold, ifconv, licm, loads, merge, promote, returns, switch, tailrec, thread, unroll};

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
    Unroll = 8,
    Licm = 9,
    Switch = 10,
    Returns = 11,
    TailRec = 12,
}

impl Pass {
    const ALL: &[Self] = &[
        Self::Fill,
        Self::Fold,
        Self::Merge,
        Self::Promote,
        Self::Cse,
        Self::Loads,
        Self::Thread,
        Self::IfConv,
        Self::Unroll,
        Self::Licm,
        Self::Switch,
        Self::Returns,
        Self::TailRec,
    ];
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
    Rule { pass: Pass::Promote, reads: Domain::CFG.union(Domain::MEM).union(Domain::VALUES) },
    Rule { pass: Pass::Fold, reads: Domain::VALUES.union(Domain::CFG) },
    Rule { pass: Pass::Merge, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Loads, reads: Domain::CFG.union(Domain::MEM).union(Domain::VALUES) },
    Rule { pass: Pass::Cse, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Thread, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::IfConv, reads: Domain::CFG.union(Domain::VALUES) },
    Rule { pass: Pass::Unroll, reads: Domain::CFG.union(Domain::VALUES).union(Domain::MEM) },
    Rule { pass: Pass::Licm, reads: Domain::CFG.union(Domain::VALUES).union(Domain::MEM) },
];
static FULL_FINISH: &[Pass] = &[Pass::Switch, Pass::Returns, Pass::TailRec];

fn run_pass(func: &mut Function, pass: Pass, moves: promote::Moves, effort: Effort) -> Domain {
    let changed = match pass {
        Pass::Fill => fill::run(func),
        Pass::Fold => fold::run(func, effort),
        Pass::Merge => merge::run(func),
        Pass::Promote => promote::step(func, moves),
        Pass::Cse => cse::run(func),
        Pass::Loads => loads::run(func),
        Pass::Thread => thread::run(func),
        Pass::IfConv => ifconv::run(func),
        Pass::Unroll => unroll::run(func),
        Pass::Licm => licm::run(func),
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
