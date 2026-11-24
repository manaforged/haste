use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Op, StackSlot, Type, Value};

fn byte_mask(off: i64, bytes: u32) -> Option<u128> {
    if bytes == 0 {
        return Some(0);
    }
    let end = off.checked_add(i64::from(bytes))?;
    if off < 0 || end > 128 {
        return None;
    }
    let width = end - off;
    let bits = if width == 128 {
        u128::MAX
    } else {
        (1_u128 << width) - 1
    };
    Some(bits << off)
}

fn slot_read_mask(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
    op: &Op,
    res: Option<Value>,
    slot: StackSlot,
) -> Option<Option<u128>> {
    match op {
        Op::StackLoad(s, off) if *s == slot => {
            let bytes = func.ty(res?).bytes();
            Some(Some(byte_mask(i64::from(*off), bytes)?))
        }
        Op::Load(addr, off) => {
            let (s, base) = addrs[addr.index()]?;
            if s != slot {
                return Some(None);
            }
            let bytes = func.ty(res?).bytes();
            Some(Some(byte_mask(base.checked_add(i64::from(*off))?, bytes)?))
        }
        Op::Move(_, src, bytes) => match addrs[src.index()] {
            Some((s, base)) if s == slot => Some(Some(byte_mask(base, *bytes)?)),
            _ => Some(None),
        },
        Op::Copy(_, from, _) if addrs[from.index()].is_some_and(|(s, _)| s == slot) => None,
        _ => Some(None),
    }
}

fn slot_write_mask(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
    op: &Op,
    slot: StackSlot,
) -> Option<Option<u128>> {
    match op {
        Op::StackStore(val, s, off) if *s == slot => {
            let bytes = func.ty(*val).bytes();
            Some(Some(byte_mask(i64::from(*off), bytes)?))
        }
        Op::Store(val, addr, off) => {
            let (s, base) = addrs[addr.index()]?;
            if s != slot {
                return Some(None);
            }
            let bytes = func.ty(*val).bytes();
            Some(Some(byte_mask(base.checked_add(i64::from(*off))?, bytes)?))
        }
        Op::Move(dst, _, bytes) => match addrs[dst.index()] {
            Some((s, base)) if s == slot => Some(Some(byte_mask(base, *bytes)?)),
            _ => Some(None),
        },
        Op::Copy(to, _, _) if addrs[to.index()].is_some_and(|(s, _)| s == slot) => None,
        _ => Some(None),
    }
}

fn entry_written(cfg: &Cfg, outside_pred: &[bool], must_out: &[u128], blk: usize) -> u128 {
    if outside_pred[blk] {
        0
    } else {
        cfg.preds_of(Block(blk as u32))
            .iter()
            .fold(u128::MAX, |acc, p| acc & must_out[p.index()])
    }
}

fn loop_safe_writes(
    func: &Function,
    cfg: &Cfg,
    addrs: &[Option<(StackSlot, i64)>],
    slot: StackSlot,
    mem: &[bool],
    incoming: &[(usize, usize)],
) -> bool {
    if func.slots[slot.index()].size > 128 {
        return false;
    }
    let nblocks = func.blocks.len();
    let mut all_written = 0_u128;
    let mut outside_pred = vec![false; nblocks];
    for blk in 0..nblocks {
        if !mem[blk] {
            continue;
        }
        outside_pred[blk] = Some(Block(blk as u32)) == func.entry
            || cfg
                .preds_of(Block(blk as u32))
                .iter()
                .any(|p| !mem[p.index()]);
        for &inst in &func.blocks[blk].insts {
            let data = &func.insts[inst.index()];
            if slot_read_mask(func, addrs, &data.op, data.result, slot).is_none() {
                return false;
            }
            let Some(w) = slot_write_mask(func, addrs, &data.op, slot) else {
                return false;
            };
            if let Some(mask) = w {
                all_written |= mask;
            }
        }
    }
    let mut must_out = vec![u128::MAX; nblocks];
    let mut changed = true;
    while changed {
        changed = false;
        for blk in 0..nblocks {
            if !mem[blk] {
                continue;
            }
            let mut cur = entry_written(cfg, &outside_pred, &must_out, blk);
            for &inst in &func.blocks[blk].insts {
                let data = &func.insts[inst.index()];
                if let Some(Some(mask)) = slot_write_mask(func, addrs, &data.op, slot) {
                    cur |= mask;
                }
            }
            if cur != must_out[blk] {
                must_out[blk] = cur;
                changed = true;
            }
        }
    }
    for blk in 0..nblocks {
        if !mem[blk] {
            continue;
        }
        let mut cur = entry_written(cfg, &outside_pred, &must_out, blk);
        for &inst in &func.blocks[blk].insts {
            let data = &func.insts[inst.index()];
            if let Some(Some(rmask)) = slot_read_mask(func, addrs, &data.op, data.result, slot)
                && (rmask & !cur) != 0
            {
                return false;
            }
            if let Some(Some(wmask)) = slot_write_mask(func, addrs, &data.op, slot) {
                cur |= wmask;
            }
        }
    }
    incoming
        .iter()
        .all(|&(pred, _)| (all_written & !must_out[pred]) == 0)
}

fn collect_incoming_args(func: &Function, block: Block, pi: usize) -> Option<Vec<(usize, Value)>> {
    let mut incoming = Vec::new();
    for p_idx in 0..func.blocks.len() {
        let Some(last) = func.last(Block(p_idx as u32)) else {
            continue;
        };
        match last {
            Op::Jump(target, args) if *target == block => incoming.push((p_idx, args[pi])),
            Op::Brif(_, then, then_args, else_b, else_args) => {
                if *then == block {
                    incoming.push((p_idx, then_args[pi]));
                }
                if *else_b == block {
                    incoming.push((p_idx, else_args[pi]));
                }
            }
            Op::TryCall {
                normal, handler, ..
            } if *normal == block || *handler == block => {
                return None;
            }
            Op::BrTable(_, default, targets) if *default == block || targets.contains(&block) => {
                return None;
            }
            _ => {}
        }
    }
    (!incoming.is_empty()).then_some(incoming)
}

fn collect_edge_slots(
    addrs: &[Option<(StackSlot, i64)>],
    incoming: &[(usize, Value)],
) -> Option<(i32, Vec<StackSlot>, Vec<(usize, usize)>)> {
    let mut common_off: Option<i32> = None;
    let mut slots: Vec<StackSlot> = Vec::new();
    let mut edge_slots: Vec<(usize, usize)> = Vec::with_capacity(incoming.len());
    for &(pred, arg) in incoming {
        let (slot, off) = addrs[arg.index()]?;
        let off32 = i32::try_from(off).ok()?;
        match common_off {
            Some(prev) if prev != off32 => return None,
            None => common_off = Some(off32),
            _ => {}
        }
        let pos = slots.iter().position(|&s| s == slot).unwrap_or_else(|| {
            slots.push(slot);
            slots.len() - 1
        });
        edge_slots.push((pred, pos));
    }
    Some((common_off?, slots, edge_slots))
}

fn param_derived_values(func: &Function, param: Value) -> Vec<bool> {
    let mut derived = vec![false; func.types.len()];
    derived[param.index()] = true;
    loop {
        let mut grew = false;
        for blk in 0..func.blocks.len() {
            for &inst in &func.blocks[blk].insts {
                let data = &func.insts[inst.index()];
                let mut op_derived = false;
                data.op.for_operands(|v| op_derived |= derived[v.index()]);
                if op_derived
                    && let Some(r) = data.result
                    && !derived[r.index()]
                {
                    derived[r.index()] = true;
                    grew = true;
                }
                let mut mark_args = |t: Block, args: &[Value]| {
                    for (idx, &v) in args.iter().enumerate() {
                        if derived[v.index()]
                            && let Some(&tp) = func.blocks[t.index()].params.get(idx)
                            && !derived[tp.index()]
                        {
                            derived[tp.index()] = true;
                            grew = true;
                        }
                    }
                };
                match &data.op {
                    Op::Jump(t, args) => mark_args(*t, args),
                    Op::Brif(_, t1, a1, t2, a2) => {
                        mark_args(*t1, a1);
                        mark_args(*t2, a2);
                    }
                    _ => {}
                }
            }
        }
        if !grew {
            break;
        }
    }
    derived
}

fn args_escape_slots(
    addrs: &[Option<(StackSlot, i64)>],
    slots: &[StackSlot],
    block: Block,
    pi: usize,
    target: Block,
    args: &[Value],
) -> bool {
    args.iter().enumerate().any(|(idx, &v)| {
        addrs[v.index()].is_some_and(|(s, _)| slots.contains(&s)) && !(target == block && idx == pi)
    })
}

fn record_slot_op(
    addrs: &[Option<(StackSlot, i64)>],
    op: &Op,
    res: Option<Value>,
    (block, pi, blk): (Block, usize, usize),
    slots: &[StackSlot],
    mem_blocks: &mut [Vec<bool>],
) -> bool {
    let pos_of = |s: StackSlot| slots.iter().position(|&x| x == s);
    let in_slots = |v: Value| addrs[v.index()].and_then(|(s, _)| pos_of(s));
    let res_resolved = res.is_some_and(|r| addrs[r.index()].is_some());
    match op {
        Op::StackLoad(s, _) | Op::StackStore(_, s, _) => {
            if let Some(i) = pos_of(*s) {
                mem_blocks[i][blk] = true;
            }
            if let Op::StackStore(val, _, _) = op
                && in_slots(*val).is_some()
            {
                return false;
            }
        }
        Op::Load(addr, _) => {
            if let Some(i) = in_slots(*addr) {
                mem_blocks[i][blk] = true;
            }
        }
        Op::Store(val, addr, _) => {
            if in_slots(*val).is_some() {
                return false;
            }
            if let Some(i) = in_slots(*addr) {
                mem_blocks[i][blk] = true;
            }
        }
        Op::Move(dst, src, _) => {
            if let Some(i) = in_slots(*dst) {
                mem_blocks[i][blk] = true;
            }
            if let Some(i) = in_slots(*src) {
                mem_blocks[i][blk] = true;
            }
        }
        Op::Copy(to, from, len) => {
            if in_slots(*len).is_some() {
                return false;
            }
            if let Some(i) = in_slots(*to) {
                mem_blocks[i][blk] = true;
            }
            if let Some(i) = in_slots(*from) {
                mem_blocks[i][blk] = true;
            }
        }
        Op::Binary(Binary::Iadd | Binary::Isub, a, b_) if res_resolved => {
            if in_slots(*a).is_some() && in_slots(*b_).is_some() {
                return false;
            }
        }
        Op::Convert(Convert::Bitcast, _) if res_resolved => {}
        Op::Select(c, ..) if res_resolved => {
            if in_slots(*c).is_some() {
                return false;
            }
        }
        Op::Jump(target, args) => {
            if args_escape_slots(addrs, slots, block, pi, *target, args) {
                return false;
            }
        }
        Op::Brif(c, then, then_args, else_b, else_args) => {
            if in_slots(*c).is_some()
                || args_escape_slots(addrs, slots, block, pi, *then, then_args)
                || args_escape_slots(addrs, slots, block, pi, *else_b, else_args)
            {
                return false;
            }
        }
        _ => {
            let mut bad = false;
            op.for_operands(|v| {
                if in_slots(v).is_some() {
                    bad = true;
                }
            });
            if bad {
                return false;
            }
        }
    }
    true
}

fn collect_slot_mem_blocks(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
    (block, pi): (Block, usize),
    slots: &[StackSlot],
    edge_slots: &[(usize, usize)],
    derived: &[bool],
) -> Option<(Vec<Vec<bool>>, Vec<usize>)> {
    let nblocks = func.blocks.len();
    let mut mem_blocks = vec![vec![false; nblocks]; slots.len()];
    for &(pred, pos) in edge_slots {
        mem_blocks[pos][pred] = true;
    }
    let mut uses_of_p = Vec::new();
    for blk in 0..nblocks {
        for &inst in &func.blocks[blk].insts {
            let data = &func.insts[inst.index()];
            let mut uses_p = false;
            data.op.for_operands(|v| {
                if derived[v.index()] {
                    uses_p = true;
                }
            });
            if uses_p {
                uses_of_p.push(blk);
            }
            if !record_slot_op(
                addrs,
                &data.op,
                data.result,
                (block, pi, blk),
                slots,
                &mut mem_blocks,
            ) {
                return None;
            }
        }
    }
    Some((mem_blocks, uses_of_p))
}

fn param_live_blocks(cfg: &Cfg, nblocks: usize, b: usize, uses_of_p: &[usize]) -> Vec<bool> {
    let mut live_p = vec![false; nblocks];
    let mut stack = Vec::new();
    for &u in uses_of_p {
        if u != b && !live_p[u] {
            live_p[u] = true;
            stack.push(u);
        }
    }
    while let Some(cur) = stack.pop() {
        for &pred in cfg.preds_of(Block(cur as u32)) {
            let p_idx = pred.index();
            if p_idx != b && !live_p[p_idx] {
                live_p[p_idx] = true;
                stack.push(p_idx);
            }
        }
    }
    live_p
}

fn slots_mutually_unreachable(func: &Function, b: usize, mem_blocks: &[Vec<bool>]) -> bool {
    let nblocks = func.blocks.len();
    let k = mem_blocks.len();
    for i in 0..k {
        let mut seen = mem_blocks[i].clone();
        let mut work: Vec<usize> = (0..nblocks).filter(|&blk| seen[blk]).collect();
        while let Some(cur) = work.pop() {
            if cur == b {
                continue;
            }
            for succ in func.successors(Block(cur as u32)) {
                let si = succ.index();
                if (0..k).any(|j| j != i && mem_blocks[j][si]) {
                    return false;
                }
                if !seen[si] {
                    seen[si] = true;
                    work.push(si);
                }
            }
        }
    }
    true
}

fn reachable_after_block(func: &Function, block: Block) -> Vec<bool> {
    let mut reach = vec![false; func.blocks.len()];
    let mut work: Vec<usize> = func.successors(block).map(|s| s.index()).collect();
    for &si in &work {
        reach[si] = true;
    }
    while let Some(cur) = work.pop() {
        for succ in func.successors(Block(cur as u32)) {
            let si = succ.index();
            if !reach[si] {
                reach[si] = true;
                work.push(si);
            }
        }
    }
    reach
}

fn can_merge_slots(
    func: &Function,
    cfg: &Cfg,
    addrs: &[Option<(StackSlot, i64)>],
    b: usize,
    slots: &[StackSlot],
    edge_slots: &[(usize, usize)],
    mem_blocks: &[Vec<bool>],
    uses_of_p: &[usize],
) -> bool {
    let nblocks = func.blocks.len();
    let mut any_mem = vec![false; nblocks];
    for blk in 0..nblocks {
        let hits = mem_blocks.iter().filter(|m| m[blk]).count();
        if hits > 1 || (hits > 0 && blk == b) {
            return false;
        }
        any_mem[blk] = hits > 0;
    }
    let live_p = param_live_blocks(cfg, nblocks, b, uses_of_p);
    if (0..nblocks).any(|blk| live_p[blk] && any_mem[blk]) {
        return false;
    }
    if slots.iter().any(|&s| func.is_unwind_slot(s)) {
        return false;
    }
    if !slots_mutually_unreachable(func, b, mem_blocks) {
        return false;
    }
    let reach_b = reachable_after_block(func, Block(b as u32));
    for i in 0..slots.len() {
        if (0..nblocks).any(|blk| reach_b[blk] && mem_blocks[i][blk]) {
            let edges_i: Vec<(usize, usize)> = edge_slots
                .iter()
                .copied()
                .filter(|&(_, pos)| pos == i)
                .collect();
            if !loop_safe_writes(func, cfg, addrs, slots[i], &mem_blocks[i], &edges_i) {
                return false;
            }
        }
    }
    true
}

fn merge_slots(func: &mut Function, slots: &[StackSlot]) {
    let s0 = slots[0];
    func.slots[s0.index()].size = slots
        .iter()
        .fold(0, |m, s| m.max(func.slots[s.index()].size));
    func.slots[s0.index()].align_shift = slots
        .iter()
        .fold(0, |m, s| m.max(func.slots[s.index()].align_shift));
    func.slots[s0.index()].poll_local = slots.iter().all(|s| func.slots[s.index()].poll_local);
    for data in &mut func.insts {
        match &mut data.op {
            Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _)
                if slots[1..].contains(s) =>
            {
                *s = s0;
            }
            _ => {}
        }
    }
}

fn try_coalesce_param(
    func: &mut Function,
    cfg: &Cfg,
    addrs: &[Option<(StackSlot, i64)>],
    b: usize,
    pi: usize,
) -> bool {
    let block = Block(b as u32);
    if Some(block) == func.entry {
        return false;
    }
    let param = func.blocks[b].params[pi];
    if func.ty(param) != Type::I64 {
        return false;
    }
    let Some(incoming) = collect_incoming_args(func, block, pi) else {
        return false;
    };
    let Some((common_off, slots, edge_slots)) = collect_edge_slots(addrs, &incoming) else {
        return false;
    };
    if slots.len() >= 2 {
        let derived = param_derived_values(func, param);
        let Some((mem_blocks, uses_of_p)) =
            collect_slot_mem_blocks(func, addrs, (block, pi), &slots, &edge_slots, &derived)
        else {
            return false;
        };
        if !can_merge_slots(
            func,
            cfg,
            addrs,
            b,
            &slots,
            &edge_slots,
            &mem_blocks,
            &uses_of_p,
        ) {
            return false;
        }
        merge_slots(func, &slots);
    }
    let s0 = slots[0];
    func.blocks[b].params.remove(pi);
    for blk in 0..func.blocks.len() {
        func.for_edge_args_mut(Block(blk as u32), block, |args| {
            args.remove(pi);
        });
    }
    let addr_inst = func.add(Op::StackAddr(s0, common_off), Some(param));
    func.blocks[b].insts.insert(0, addr_inst);
    true
}

fn unresolved_params(
    func: &Function,
    addrs: &[Option<(StackSlot, i64)>],
) -> (Vec<usize>, Vec<bool>) {
    let mut starts = Vec::with_capacity(func.blocks.len());
    let mut total = 0;
    for block in &func.blocks {
        starts.push(total);
        total += block.params.len();
    }
    let mut unresolved = vec![false; total];
    let mut mark = |target: Block, args: &[Value]| {
        let count = func.blocks[target.index()].params.len();
        for (idx, v) in args.iter().enumerate().take(count) {
            unresolved[starts[target.index()] + idx] |= addrs[v.index()].is_none();
        }
    };
    for b in 0..func.blocks.len() {
        match func.last(Block(b as u32)) {
            Some(Op::Jump(target, args)) => mark(*target, args),
            Some(Op::Brif(_, then, then_args, else_b, else_args)) => {
                mark(*then, then_args);
                mark(*else_b, else_args);
            }
            _ => {}
        }
    }
    (starts, unresolved)
}

pub(super) fn coalesce_block_params(
    func: &mut Function,
    addrs: &[Option<(StackSlot, i64)>],
) -> bool {
    let (starts, unresolved) = unresolved_params(func, addrs);
    if unresolved.iter().all(|&u| u) {
        return false;
    }
    let cfg = func.cfg();
    for b in 0..func.blocks.len() {
        for pi in 0..func.blocks[b].params.len() {
            if !unresolved[starts[b] + pi] && try_coalesce_param(func, &cfg, addrs, b, pi) {
                return true;
            }
        }
    }
    false
}
