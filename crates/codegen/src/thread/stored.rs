use crate::fold::{extend, holds};
use crate::ir::{Binary, Block, Function, Inst, IntCC, Op, Type, Value};

use super::eval::{EvalState, ThreadCx, lookup_const};

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

fn unwrap_addr_offset(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    local: Option<(&[(Value, Op)], &[(Value, i64)])>,
    addr: Value,
    off: i32,
) -> (Value, i64) {
    let op = local
        .and_then(|(ops, _)| ops.iter().rev().find(|&&(r, _)| r == addr).map(|(_, op)| op))
        .or_else(|| defs.get(addr.index()).copied().flatten().map(|i| &func.insts[i.index()].op));
    let k = |v: Value| {
        local.map_or_else(|| consts.get(v.index()).copied().flatten(), |(_, lc)| lookup_const(lc, consts, v))
    };
    let Some(op) = op else { return (addr, i64::from(off)) };
    match *op {
        Op::Binary(Binary::Iadd, a, b) => {
            if let Some(c) = k(b).and_then(|c| i64::from(off).checked_add(c)) {
                return (a, c);
            }
            if let Some(c) = k(a).and_then(|c| i64::from(off).checked_add(c)) {
                return (b, c);
            }
        }
        Op::Binary(Binary::Isub, a, b) => {
            if let Some(c) = k(b).and_then(|c| i64::from(off).checked_sub(c)) {
                return (a, c);
            }
        }
        _ => {}
    }
    (addr, i64::from(off))
}

fn extract_const(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], val: Value, delta: i64, ty: Type) -> Option<i64> {
    let vty = func.ty(val);
    if vty.is_vector() {
        let inst = defs.get(val.index()).copied().flatten()?;
        return matches!(func.insts[inst.index()].op, Op::Const(0)).then_some(0);
    }
    if !vty.is_small_int() || !(0..8).contains(&delta) {
        return None;
    }
    let imm = consts.get(val.index()).copied().flatten()?;
    Some(extend(((imm as u64) >> (delta * 8)) as i64, ty, true))
}

/// Whether `block` contains a small-integer load that predecessor stores can resolve.
pub(super) fn body_has_load(func: &Function, block: Block) -> bool {
    let Some((_, body)) = func.blocks[block.index()].insts.split_last() else { return false };
    body.iter().any(|inst| matches!(func.insts[inst.index()].op, Op::Load(..) | Op::StackLoad(..)))
}

/// Whether `block` is a small pure-or-load dispatcher eligible for jump threading.
pub(super) fn steppable(func: &Function, block: Block, max_insts: usize) -> bool {
    let insts = &func.blocks[block.index()].insts;
    let Some((&last, body)) = insts.split_last() else {
        return false;
    };
    let mut loads = 0_u8;
    insts.len() <= max_insts
        && matches!(func.insts[last.index()].op, Op::Brif(..) | Op::Jump(..) | Op::BrTable(..))
        && body.iter().all(|inst| {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { return false };
            if data.op.is_speculable() {
                true
            } else if matches!(data.op, Op::Load(..) | Op::StackLoad(..)) && func.ty(res).is_small_int() && loads < 2 {
                loads += 1;
                true
            } else {
                false
            }
        })
}

enum StoreMatch {
    Hit(i64),
    Abort,
    Skip,
}

fn match_const_store(func: &Function, defs: &[Option<Inst>], consts: &[Option<i64>], (sval, s_lo): (Value, i64), (t_lo, t_hi, ty): (i64, i64, Type)) -> StoreMatch {
    let Some(s_hi) = s_lo.checked_add(i64::from(func.ty(sval).bytes())) else { return StoreMatch::Abort };
    if s_lo <= t_lo && t_hi <= s_hi {
        return match extract_const(func, defs, consts, sval, t_lo - s_lo, ty) {
            Some(c) => StoreMatch::Hit(c),
            None => StoreMatch::Abort,
        };
    }
    if s_lo < t_hi && t_lo < s_hi { StoreMatch::Abort } else { StoreMatch::Skip }
}

fn normalize_addr(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    local: Option<(&[(Value, Op)], &[(Value, i64)])>,
    raw_addr: Value,
    raw_off: i32,
) -> Option<(Result<Value, crate::ir::StackSlot>, i64)> {
    let (addr, lo) = unwrap_addr_offset(func, defs, consts, local, raw_addr, raw_off);
    let op = local
        .and_then(|(ops, _)| ops.iter().rev().find(|&&(r, _)| r == addr).map(|(_, op)| op))
        .or_else(|| defs.get(addr.index()).copied().flatten().map(|i| &func.insts[i.index()].op));
    if let Some(&Op::StackAddr(slot, base)) = op {
        return Some((Err(slot), lo.checked_add(i64::from(base))?));
    }
    Some((Ok(addr), lo))
}

/// Resolves a mapped `Load` or `StackLoad` of a small integer against trailing non-aliasing constant stores in `pred`.
pub(super) fn pred_load_const_raw(
    func: &Function,
    defs: &[Option<Inst>],
    consts: &[Option<i64>],
    local: Option<(&[(Value, Op)], &[(Value, i64)])>,
    pred: Block,
    op: &Op,
    ty: Type,
) -> Option<i64> {
    if !ty.is_small_int() {
        return None;
    }
    let (_, body) = func.blocks[pred.index()].insts.split_last()?;
    let sz = i64::from(ty.bytes());
    let (target_addr, t_lo) = match *op {
        Op::Load(raw_addr, raw_off) => normalize_addr(func, defs, consts, local, raw_addr, raw_off)?,
        Op::StackLoad(slot, raw_off) => (Err(slot), i64::from(raw_off)),
        _ => return None,
    };
    let t_hi = t_lo.checked_add(sz)?;
    for &inst in body.iter().rev() {
        let pop = &func.insts[inst.index()].op;
        let hit = match (target_addr, pop) {
            (Ok(t_addr), &Op::Store(sval, saddr, soff)) => {
                let (sb, s_lo) = normalize_addr(func, defs, consts, None, saddr, soff)?;
                if sb != Ok(t_addr) {
                    return None;
                }
                match_const_store(func, defs, consts, (sval, s_lo), (t_lo, t_hi, ty))
            }
            (Err(t_slot), &Op::StackStore(sval, sslot, soff)) => {
                if sslot != t_slot {
                    continue;
                }
                match_const_store(func, defs, consts, (sval, i64::from(soff)), (t_lo, t_hi, ty))
            }
            (Err(t_slot), &Op::Store(sval, saddr, soff)) => {
                let (sb, s_lo) = normalize_addr(func, defs, consts, None, saddr, soff)?;
                if sb != Err(t_slot) {
                    return None;
                }
                match_const_store(func, defs, consts, (sval, s_lo), (t_lo, t_hi, ty))
            }
            (_, Op::Load(..) | Op::StackLoad(..)) => StoreMatch::Skip,
            (_, o) if o.is_speculable() => StoreMatch::Skip,
            _ => return None,
        };
        match hit {
            StoreMatch::Hit(c) => return Some(c),
            StoreMatch::Abort => return None,
            StoreMatch::Skip => {}
        }
    }
    None
}

/// Resolves a mapped `Load` or `StackLoad` of a small integer against trailing non-aliasing constant stores in `pred`.
pub(super) fn pred_load_const(func: &Function, cx: &ThreadCx<'_>, state: &EvalState, pred: Block, op: &Op, ty: Type) -> Option<i64> {
    pred_load_const_raw(func, cx.defs, cx.consts, Some((state.local_ops.as_slice(), state.local_consts.as_slice())), pred, op, ty)
}

/// Sends each edge into a block that reloads, tests and branches on a word its predecessor just stored as a constant straight to the side that constant takes; returns whether it moved any edge.
pub(super) fn run(func: &mut Function) -> bool {
    let cfg = func.cfg();
    let (defs, users) = super::defs_and_uses(func);
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
    super::retarget_empty_preds(func, &moved)
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
    let (cc, a, b) = if cond == value {
        (IntCC::NotEqual, value, None)
    } else {
        let (cc, a, b) = test.and_then(|inst| crate::algebra::branch_cmp(&func.insts[inst.index()].op))?;
        (cc, a, Some(b))
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
