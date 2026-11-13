use crate::ir::Function;
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

const PASS_COUNT: usize = 13;
const MAX_ROUNDS: usize = 6;

struct Rule {
    pass: Pass,
    max_runs: u8,
    enables: &'static [Pass],
}

struct Plan {
    moves: promote::Moves,
    effort: Effort,
    init: &'static [Pass],
    seed: &'static [Pass],
    rules: &'static [Rule],
    max_rounds: usize,
    finish: &'static [Pass],
}

static QUICK_INIT: &[Pass] = &[Pass::Fill, Pass::Fold, Pass::Merge];
static QUICK_SEED: &[Pass] = &[Pass::Promote];
static QUICK_RULES: &[Rule] = &[
    Rule { pass: Pass::Promote, max_runs: 2, enables: &[Pass::Promote, Pass::Fold, Pass::Merge] },
    Rule { pass: Pass::Fold, max_runs: 2, enables: &[] },
    Rule { pass: Pass::Merge, max_runs: 2, enables: &[] },
];
static QUICK_FINISH: &[Pass] = &[Pass::Cse, Pass::Returns];

static FULL_INIT: &[Pass] = &[Pass::Fill, Pass::Fold, Pass::Merge];
static FULL_SEED: &[Pass] = &[Pass::Promote, Pass::Loads, Pass::Cse, Pass::Thread, Pass::IfConv, Pass::Unroll, Pass::Licm];
static FULL_RULES: &[Rule] = &[
    Rule {
        pass: Pass::Promote,
        max_runs: 6,
        enables: &[Pass::Promote, Pass::Fold, Pass::Merge, Pass::Loads, Pass::Cse, Pass::Unroll, Pass::Licm],
    },
    Rule { pass: Pass::Fold, max_runs: 8, enables: &[Pass::Merge, Pass::Promote, Pass::Cse, Pass::Thread, Pass::IfConv, Pass::Unroll] },
    Rule { pass: Pass::Merge, max_runs: 8, enables: &[Pass::Fold, Pass::Promote, Pass::IfConv, Pass::Unroll] },
    Rule { pass: Pass::Loads, max_runs: 4, enables: &[Pass::Fold, Pass::Promote, Pass::Cse, Pass::Licm] },
    Rule { pass: Pass::Cse, max_runs: 4, enables: &[Pass::Fold, Pass::Loads] },
    Rule { pass: Pass::Thread, max_runs: 3, enables: &[Pass::Merge, Pass::Fold, Pass::Cse, Pass::Loads] },
    Rule { pass: Pass::IfConv, max_runs: 3, enables: &[Pass::Merge, Pass::Fold, Pass::Cse, Pass::Loads] },
    Rule { pass: Pass::Unroll, max_runs: 3, enables: &[Pass::Fold, Pass::Merge, Pass::Loads, Pass::Promote, Pass::Cse] },
    Rule { pass: Pass::Licm, max_runs: 3, enables: &[Pass::Loads, Pass::Fold, Pass::Cse] },
];
static FULL_FINISH: &[Pass] = &[Pass::Switch, Pass::Returns, Pass::TailRec];

fn run_pass(func: &mut Function, pass: Pass, moves: promote::Moves, effort: Effort) -> bool {
    match pass {
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
            if did {
                fold::compact(func);
            }
            did
        }
    }
}

fn run_fixpoint(func: &mut Function, plan: &Plan) {
    let mut active = [false; PASS_COUNT];
    let mut runs = [0_u8; PASS_COUNT];
    for &pass in plan.seed {
        active[pass as usize] = true;
    }
    for _ in 0..plan.max_rounds {
        let mut any = false;
        for rule in plan.rules {
            let idx = rule.pass as usize;
            if !active[idx] || runs[idx] >= rule.max_runs {
                continue;
            }
            active[idx] = false;
            runs[idx] += 1;
            if run_pass(func, rule.pass, plan.moves, plan.effort) {
                any = true;
                for &next in rule.enables {
                    active[next as usize] = true;
                }
            }
        }
        if !any {
            break;
        }
    }
}

pub(crate) fn run(func: &mut Function, moves: promote::Moves, effort: Effort) {
    let plan = match effort {
        Effort::Quick => Plan {
            moves: promote::Moves::Words,
            effort,
            init: QUICK_INIT,
            seed: QUICK_SEED,
            rules: QUICK_RULES,
            max_rounds: 2,
            finish: QUICK_FINISH,
        },
        Effort::Full => Plan {
            moves,
            effort,
            init: FULL_INIT,
            seed: FULL_SEED,
            rules: FULL_RULES,
            max_rounds: MAX_ROUNDS,
            finish: FULL_FINISH,
        },
    };
    for &pass in plan.init {
        run_pass(func, pass, plan.moves, plan.effort);
    }
    run_fixpoint(func, &plan);
    for &pass in plan.finish {
        run_pass(func, pass, plan.moves, plan.effort);
    }
}
