use crate::fold::extend;
use crate::ir::{Block, Function, Inst, IntCC, Op, Value};

/// The most equality tests one walk follows.
const MAX_WALK: usize = 32;

/// An equality test of a value against a constant that ends a block: the value, the constant, where it goes when equal and where it goes when not.
#[derive(Clone, Copy)]
struct Test {
    value: Value,
    constant: i64,
    equal: Block,
    other: Block,
}

/// Sends each edge out of the equal arm of a test into a later test of the same value straight to where those tests land for the arm's constant; returns whether it moved any edge.
pub(super) fn run(func: &mut Function) -> bool {
    let cfg = func.cfg();
    let mut defs = vec![None; func.types.len()];
    let mut users = vec![0_u32; func.types.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                defs[res.index()] = Some(inst);
            }
            data.op.for_operands(|v| users[v.index()] += 1);
        }
    }
    let tests: Vec<Option<Test>> = (0..func.blocks.len()).map(|b| test(func, &defs, &users, Block(b as u32))).collect();
    let mut moved = Vec::new();
    for (at, found) in tests.iter().enumerate() {
        let Some(Test { value, constant, equal, other }) = *found else { continue };
        if !cfg.reachable[at] || cfg.single_pred(equal) != Some(Block(at as u32)) || equal == other || tests[other.index()].is_none_or(|next| next.value != value) {
            continue;
        }
        let land = walk(func, &tests, other, value, constant);
        if land == other {
            continue;
        }
        for &pred in cfg.preds_of(other) {
            if pred.index() != at && cfg.dominates(equal, pred) {
                moved.push((pred, other, land));
            }
        }
    }
    let mut changed = false;
    for (pred, from, to) in moved {
        let Some(&term) = func.blocks[pred.index()].insts.last() else { continue };
        match &mut func.insts[term.index()].op {
            Op::Jump(target, args) if *target == from && args.is_empty() => *target = to,
            Op::Brif(_, then, then_args, other, other_args) => {
                if *then == from && then_args.is_empty() {
                    *then = to;
                }
                if *other == from && other_args.is_empty() {
                    *other = to;
                }
            }
            _ => continue,
        }
        changed = true;
    }
    if changed {
        func.bump_cfg();
    }
    changed
}

/// Where the tests from `block` on lead when `value` holds `constant`.
fn walk(func: &Function, tests: &[Option<Test>], mut block: Block, value: Value, constant: i64) -> Block {
    let ty = func.ty(value);
    for _ in 0..MAX_WALK {
        let Some(test) = tests[block.index()].filter(|test| test.value == value) else { break };
        block = if extend(test.constant, ty, false) == extend(constant, ty, false) { test.equal } else { test.other };
    }
    block
}

/// The test that ends `block` when the block has no parameters, holds only the comparison and constants used nowhere else, and branches without arguments.
fn test(func: &Function, defs: &[Option<Inst>], users: &[u32], block: Block) -> Option<Test> {
    let data = &func.blocks[block.index()];
    let (&last, body) = data.insts.split_last()?;
    let Op::Brif(cond, then, ref then_args, other, ref other_args) = func.insts[last.index()].op else { return None };
    if !data.params.is_empty() || !then_args.is_empty() || !other_args.is_empty() {
        return None;
    }
    let local = |inst: &Inst| func.insts[inst.index()].result.is_some_and(|res| users[res.index()] == 1);
    if !body.iter().all(|inst| local(inst) && matches!(func.insts[inst.index()].op, Op::Const(_) | Op::Icmp(..))) {
        return None;
    }
    let def = defs[cond.index()].filter(|inst| body.contains(inst))?;
    let Op::Icmp(cc @ (IntCC::Equal | IntCC::NotEqual), a, b) = func.insts[def.index()].op else { return None };
    let constant_of = |v: Value| match defs[v.index()].map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    let (value, constant) = match (constant_of(a), constant_of(b)) {
        (None, Some(imm)) => (a, imm),
        (Some(imm), None) => (b, imm),
        _ => return None,
    };
    let (equal, other) = if cc == IntCC::Equal { (then, other) } else { (other, then) };
    Some(Test { value, constant, equal, other })
}
