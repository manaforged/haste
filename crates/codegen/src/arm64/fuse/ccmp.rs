use super::super::asm::Reg;
use super::flag::{icmp_consumer, preserves_flags, try_cmp_preserved};
use super::{def_entry, local_single};
use crate::alloc::{Location, Plan};
use crate::ir::{Binary, Function, Inst, IntCC, Op, Value};

const MAX_CHAIN: usize = 8;

struct Leaf {
    pos: usize,
    val: Value,
    cc: IntCC,
    a: Value,
    b: Value,
}

fn plain_operands(plan: &Plan<Reg>, a: Value, b: Value) -> bool {
    !plan.elided[a.index()] && !plan.elided[b.index()]
}

fn safe_scratch_op(func: &Function, op: &Op) -> bool {
    matches!(op, Op::Const(_) | Op::StackAddr(..) | Op::FramePointer)
        || matches!(op, Op::Icmp(_, x, _) if func.ty(*x).is_small_int())
}

fn clobbers_target(plan: &Plan<Reg>, reg: Option<Reg>, slot_id: Option<u32>, r: Value) -> bool {
    if reg.is_some_and(|reg| plan.reg_of(r) == Some(reg)) {
        return true;
    }
    slot_id.is_some_and(|id| matches!(plan.location(r), Location::Slot) && plan.shared.get(r.index()).copied().flatten() == Some(id))
}

fn inst_clobbers_val(func: &Function, plan: &Plan<Reg>, chain: &[Value], (reg, slot_id): (Option<Reg>, Option<u32>), inst: Inst) -> bool {
    let data = &func.insts[inst.index()];
    if data.result.is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()] || chain.contains(&r)) {
        return false;
    }
    if reg.is_some_and(|r| (11..=15).contains(&r.0)) && !safe_scratch_op(func, &data.op) {
        return true;
    }
    if reg.is_some() && matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
        return true;
    }
    data.result.and_then(|r| plan.pairs[r.index()]).is_some_and(|p| clobbers_target(plan, reg, slot_id, p))
        || crate::live::results(data).any(|r| clobbers_target(plan, reg, slot_id, r))
}

fn val_preserved(func: &Function, plan: &Plan<Reg>, chain: &[Value], (b_idx, start, end): (usize, usize, usize), val: Value) -> bool {
    if plan.elided[val.index()] {
        return false;
    }
    if start + 1 >= end || plan.location(val).is_static() {
        return true;
    }
    let target = match plan.location(val) {
        Location::Reg(reg) => (Some(reg), None),
        Location::Slot => match plan.shared.get(val.index()).copied().flatten() {
            Some(id) => (None, Some(id)),
            None => return true,
        },
        _ => return false,
    };
    (start + 1..end).all(|pos| !inst_clobbers_val(func, plan, chain, target, func.blocks[b_idx].insts[pos]))
}

fn tail_preserved(func: &Function, plan: &Plan<Reg>, chain: &[Value], (b_idx, root_pos): (usize, usize), leaf: &Leaf) -> bool {
    func.ty(leaf.a).is_small_int()
        && plain_operands(plan, leaf.a, leaf.b)
        && val_preserved(func, plan, chain, (b_idx, leaf.pos, root_pos), leaf.a)
        && val_preserved(func, plan, chain, (b_idx, leaf.pos, root_pos), leaf.b)
}

fn already_elided_tst_ok(
    func: &Function,
    plan: &Plan<Reg>,
    pos_of: &[Option<(usize, usize, Inst)>],
    (b_idx, root_pos): (usize, usize),
    leaf: &Leaf,
) -> bool {
    if !matches!(leaf.cc, IntCC::Equal | IntCC::NotEqual) {
        return false;
    }
    let Some(m) = (plan.const_of(leaf.b) == Some(0))
        .then_some(leaf.a)
        .or_else(|| (plan.const_of(leaf.a) == Some(0)).then_some(leaf.b))
    else {
        return false;
    };
    if !plan.elided[m.index()] {
        return false;
    }
    let Some((mb, mpos, minst)) = def_entry(pos_of, m) else { return false };
    if mb != b_idx || mpos >= leaf.pos {
        return false;
    }
    let Op::Binary(Binary::Band, bx, by) = func.insts[minst.index()].op else { return false };
    super::bits::can_elide_tst_band(func, plan, pos_of, (b_idx, mpos, root_pos, None), (bx, by))
}

fn head_band(
    func: &Function,
    plan: &Plan<Reg>,
    info @ (_, pos_of): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, root_pos): (usize, usize),
    chain: &[Value],
    head: &Leaf,
) -> Option<Option<Value>> {
    if tail_preserved(func, plan, chain, (b_idx, root_pos), head)
        || already_elided_tst_ok(func, plan, pos_of, (b_idx, root_pos), head)
    {
        return Some(None);
    }
    try_cmp_preserved(func, plan, info, (b_idx, head.pos, root_pos, None), (head.cc, head.a, head.b, 1))
}

fn local_child(
    func: &Function,
    plan: &Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, limit): (usize, usize),
    val: Value,
) -> Option<(usize, Inst)> {
    (!plan.fused[val.index()] && !plan.is_static(val) && func.ty(val).is_small_int())
        .then(|| local_single(plan, info, b_idx, limit, val))
        .flatten()
}

fn as_leaf(func: &Function, pos: usize, val: Value, inst: Inst) -> Option<Leaf> {
    let Op::Icmp(cc, a, b) = func.insts[inst.index()].op else { return None };
    func.ty(a).is_small_int().then_some(Leaf { pos, val, cc, a, b })
}

fn collect_chain(
    func: &Function,
    plan: &Plan<Reg>,
    info: (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, x, y): (usize, usize, Value, Value),
) -> Option<(Vec<Value>, Leaf, Vec<Leaf>)> {
    let (mut cur_pos, mut cur_x, mut cur_y) = (pos, x, y);
    let (mut chain, mut tails) = (Vec::new(), Vec::new());
    while tails.len() + 1 < MAX_CHAIN {
        let (xp, xi) = local_child(func, plan, info, (b_idx, cur_pos), cur_x)?;
        let (yp, yi) = local_child(func, plan, info, (b_idx, cur_pos), cur_y)?;
        if let (Some(lx), Some(ly)) = (as_leaf(func, xp, cur_x, xi), as_leaf(func, yp, cur_y, yi)) {
            let (head, tail) = if plain_operands(plan, ly.a, ly.b) { (lx, ly) } else { (ly, lx) };
            chain.push(head.val);
            chain.push(tail.val);
            tails.push(tail);
            return Some((chain, head, tails));
        }
        let (next_val, next_pos, next_inst, tail) = match (as_leaf(func, xp, cur_x, xi), as_leaf(func, yp, cur_y, yi)) {
            (_, Some(ly)) if plain_operands(plan, ly.a, ly.b) => (cur_x, xp, xi, ly),
            (Some(lx), _) if plain_operands(plan, lx.a, lx.b) => (cur_y, yp, yi, lx),
            _ => return None,
        };
        let Op::Binary(Binary::Band | Binary::Bor, nx, ny) = func.insts[next_inst.index()].op else { return None };
        chain.push(tail.val);
        chain.push(next_val);
        tails.push(tail);
        (cur_pos, cur_x, cur_y) = (next_pos, nx, ny);
    }
    None
}

fn branch_or_trap_only(func: &Function, tail: &[Inst], res: Value) -> bool {
    for &ni in tail {
        let op = &func.insts[ni.index()].op;
        if matches!(op, Op::Brif(c, _, ta, _, ea) if *c == res && !ta.contains(&res) && !ea.contains(&res))
            || matches!(op, Op::Trapz(c) if *c == res)
        {
            return true;
        }
        if !preserves_flags(op) {
            return false;
        }
    }
    false
}

fn try_fuse_root(
    func: &Function,
    plan: &mut Plan<Reg>,
    info @ (uses, _): (&[u32], &[Option<(usize, usize, Inst)>]),
    (b_idx, pos, res): (usize, usize, Value),
    (op, x, y): (Binary, Value, Value),
) {
    if !matches!(op, Binary::Band | Binary::Bor) || plan.fused[res.index()] || plan.is_static(res) {
        return;
    }
    let tail = &func.blocks[b_idx].insts[pos + 1..];
    let already_elided = plan.elided[res.index()];
    if already_elided && (op != Binary::Band || uses[res.index()] != 1 || !branch_or_trap_only(func, tail, res)) {
        return;
    }
    let Some((chain, head, tails)) = collect_chain(func, plan, info, (b_idx, pos, x, y)) else { return };
    if !tails.iter().all(|t| tail_preserved(func, plan, &chain, (b_idx, pos), t)) {
        return;
    }
    let Some(band) = head_band(func, plan, info, (b_idx, pos), &chain, &head) else { return };
    let fuse_flags = already_elided || (uses[res.index()] == 1 && icmp_consumer(func, tail, res));
    for v in chain {
        plan.elided[v.index()] = true;
    }
    if let Some(m) = band {
        plan.elided[m.index()] = true;
    }
    plan.elided[res.index()] = false;
    plan.fused[res.index()] = fuse_flags;
}

pub(super) fn fuse_ccmp(func: &Function, plan: &mut Plan<Reg>, info: (&[u32], &[Option<(usize, usize, Inst)>])) {
    for b_idx in 0..func.blocks.len() {
        for pos in (0..func.blocks[b_idx].insts.len()).rev() {
            let data = &func.insts[func.blocks[b_idx].insts[pos].index()];
            let Some(res) = data.result else { continue };
            if let Op::Binary(op, x, y) = data.op {
                try_fuse_root(func, plan, info, (b_idx, pos, res), (op, x, y));
            }
        }
    }
}
