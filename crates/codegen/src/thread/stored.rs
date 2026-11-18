use crate::fold::holds;
use crate::ir::{Binary, Block, Function, Inst, IntCC, Op, Value};

/// A block that loads one word, compares it with a constant and branches on the result: the loaded address and offset, the test as `stored cc constant` or `constant cc stored`, and the two targets.
struct Reload {
    addr: Value,
    offset: i32,
    value: Value,
    cc: IntCC,
    constant: i64,
    stored_left: bool,
    then: Block,
    other: Block,
}

/// Sends each edge into a block that reloads, tests and branches on a word its predecessor just stored as a constant straight to the side that constant takes; returns whether it moved any edge.
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
    let constant = |v: Value| match defs[v.index()].map(|inst: Inst| &func.insts[inst.index()].op) {
        Some(&Op::Const(imm)) => Some(imm),
        _ => None,
    };
    let mut moved = Vec::new();
    for at in 0..func.blocks.len() {
        let block = Block(at as u32);
        if !cfg.reachable[at] || Some(block) == func.entry {
            continue;
        }
        let Some(reload) = reload(func, &defs, &users, block, &constant) else { continue };
        let ty = func.ty(reload.value);
        let readers = reader_blocks(func, block, reload.value);
        let held = |side: Block| cfg.single_pred(side) == Some(block) && readers.iter().all(|&at| cfg.dominates(side, at));
        for &pred in cfg.preds_of(block) {
            let Some(stored) = stored_const(func, pred, &reload).and_then(constant) else { continue };
            let decided = if reload.stored_left { holds(reload.cc, ty, stored, reload.constant) } else { holds(reload.cc, ty, reload.constant, stored) };
            let (to, skipped) = if decided { (reload.then, reload.other) } else { (reload.other, reload.then) };
            if readers.is_empty() || held(skipped) {
                moved.push((pred, block, to));
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

/// The reload `block` holds when it has no parameters, starts with a load read by the branch or one integer compare, subtract or xor with a constant that only the branch reads, holds nothing else but constants read only there, and branches without arguments.
fn reload(func: &Function, defs: &[Option<Inst>], users: &[u32], block: Block, constant: &impl Fn(Value) -> Option<i64>) -> Option<Reload> {
    let data = &func.blocks[block.index()];
    let (&last, body) = data.insts.split_last()?;
    let Op::Brif(cond, then, ref then_args, other, ref other_args) = func.insts[last.index()].op else { return None };
    if !data.params.is_empty() || !then_args.is_empty() || !other_args.is_empty() || then == other {
        return None;
    }
    let (&first, rest) = body.split_first()?;
    let Op::Load(addr, offset) = func.insts[first.index()].op else { return None };
    let value = func.insts[first.index()].result?;
    let test = defs[cond.index()].filter(|inst| rest.contains(inst) && users[cond.index()] == 1);
    let (cc, a, b) = match test.map(|inst| &func.insts[inst.index()].op) {
        _ if cond == value => (IntCC::NotEqual, value, None),
        Some(&Op::Icmp(cc, a, b)) => (cc, a, Some(b)),
        Some(&Op::Binary(Binary::Isub | Binary::Bxor, a, b)) => (IntCC::NotEqual, a, Some(b)),
        _ => return None,
    };
    let local = |inst: Inst| func.insts[inst.index()].result.is_some_and(|res| users[res.index()] == 1);
    if !rest.iter().all(|&inst| Some(inst) == test || (local(inst) && matches!(func.insts[inst.index()].op, Op::Const(_)))) {
        return None;
    }
    let (stored_left, other_side) = match b {
        None => (true, None),
        Some(b) if a == value => (true, Some(b)),
        Some(b) if b == value => (false, Some(a)),
        Some(_) => return None,
    };
    let constant = match other_side {
        None => 0,
        Some(side) => constant(side)?,
    };
    Some(Reload { addr, offset, value, cc, constant, stored_left, then, other })
}

/// The blocks other than `block` that read `value`.
fn reader_blocks(func: &Function, block: Block, value: Value) -> Vec<Block> {
    let mut readers = Vec::new();
    for (at, data) in func.blocks.iter().enumerate() {
        let here = Block(at as u32);
        if here != block && data.insts.iter().any(|inst| {
            let mut hit = false;
            func.insts[inst.index()].op.for_operands(|v| hit |= v == value);
            hit
        }) {
            readers.push(here);
        }
    }
    readers
}

/// The value the last store of `pred` writes at the reload's address and offset, when only pure instructions follow that store.
fn stored_const(func: &Function, pred: Block, reload: &Reload) -> Option<Value> {
    let (_, body) = func.blocks[pred.index()].insts.split_last()?;
    for &inst in body.iter().rev() {
        let op = &func.insts[inst.index()].op;
        match *op {
            Op::Store(value, addr, offset) if addr == reload.addr && offset == reload.offset => {
                return (func.ty(value) == func.ty(reload.value)).then_some(value);
            }
            _ if op.is_speculable() => {}
            _ => return None,
        }
    }
    None
}
