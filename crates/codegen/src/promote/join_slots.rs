use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, StackSlot, Type, Value};

fn byte_mask(off: i64, bytes: u32) -> Option<u128> {
    if bytes == 0 {
        return Some(0);
    }
    let end = off.checked_add(i64::from(bytes))?;
    if off < 0 || end > 128 {
        return None;
    }
    let width = end - off;
    let bits = if width == 128 { u128::MAX } else { (1_u128 << width) - 1 };
    Some(bits << off)
}

fn slot_read_mask(func: &Function, addrs: &[Option<(StackSlot, i64)>], op: &Op, res: Option<Value>, slot: StackSlot) -> Option<Option<u128>> {
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

fn slot_write_mask(func: &Function, addrs: &[Option<(StackSlot, i64)>], op: &Op, slot: StackSlot) -> Option<Option<u128>> {
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

fn loop_safe_writes(func: &Function, cfg: &Cfg, addrs: &[Option<(StackSlot, i64)>], slot: StackSlot, mem: &[bool], incoming: &[(usize, usize)]) -> bool {
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
        outside_pred[blk] = Some(Block(blk as u32)) == func.entry || cfg.preds_of(Block(blk as u32)).iter().any(|p| !mem[p.index()]);
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
            let mut cur = if outside_pred[blk] {
                0
            } else {
                cfg.preds_of(Block(blk as u32)).iter().fold(u128::MAX, |acc, p| acc & must_out[p.index()])
            };
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
        let mut cur = if outside_pred[blk] {
            0
        } else {
            cfg.preds_of(Block(blk as u32)).iter().fold(u128::MAX, |acc, p| acc & must_out[p.index()])
        };
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
    incoming.iter().all(|&(pred, _)| (all_written & !must_out[pred]) == 0)
}

struct SelectEdit {
    sel: Inst,
    cond: Value,
    sa: StackSlot,
    sb: StackSlot,
    loads: Vec<(Inst, Value, i32, i32)>,
}

fn fold_slot_selects(func: &mut Function, addrs: &[Option<(StackSlot, i64)>]) -> bool {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        func.insts[inst.index()].op.for_operands(|v| uses[v.index()] += 1);
    }
    let mut changed = false;
    for blk in 0..func.blocks.len() {
        let len = func.blocks[blk].insts.len();
        let mut edits = Vec::new();
        for pos in 0..len {
            let sel = func.blocks[blk].insts[pos];
            let data = &func.insts[sel.index()];
            let (Op::Select(cond, a, b), Some(r)) = (&data.op, data.result) else { continue };
            if addrs[r.index()].is_some() || uses[r.index()] == 0 || func.ty(*cond).is_vector() {
                continue;
            }
            let (Some((sa, oa)), Some((sb, ob))) = (addrs[a.index()], addrs[b.index()]) else { continue };
            let mut loads = Vec::new();
            let mut valid = true;
            for &later in &func.blocks[blk].insts[pos + 1..] {
                let ldata = &func.insts[later.index()];
                let Op::Load(addr, loff) = ldata.op else {
                    ldata.op.for_operands(|v| {
                        if v == r {
                            valid = false;
                        }
                    });
                    if !valid {
                        break;
                    }
                    continue;
                };
                if addr != r {
                    continue;
                }
                let Some(lres) = ldata.result else {
                    valid = false;
                    break;
                };
                let ty = func.ty(lres);
                if !(ty.is_small_int() || ty.is_float() || ty.is_vector() || ty == Type::I128) {
                    valid = false;
                    break;
                }
                let Some(off_a) = oa.checked_add(i64::from(loff)).and_then(|o| i32::try_from(o).ok()) else {
                    valid = false;
                    break;
                };
                let Some(off_b) = ob.checked_add(i64::from(loff)).and_then(|o| i32::try_from(o).ok()) else {
                    valid = false;
                    break;
                };
                let bytes = i64::from(ty.bytes());
                if off_a < 0 || i64::from(off_a) + bytes > i64::from(func.slots[sa.index()].size) || off_b < 0 || i64::from(off_b) + bytes > i64::from(func.slots[sb.index()].size) {
                    valid = false;
                    break;
                }
                loads.push((later, lres, off_a, off_b));
            }
            if valid && loads.len() as u32 == uses[r.index()] {
                edits.push(SelectEdit { sel, cond: *cond, sa, sb, loads });
            }
        }
        if edits.is_empty() {
            continue;
        }
        changed = true;
        let mut drops = vec![false; func.insts.len()];
        let mut rewrites: Vec<Option<(Value, StackSlot, i32, StackSlot, i32, Value)>> = vec![None; func.insts.len()];
        for edit in edits {
            drops[edit.sel.index()] = true;
            for (inst, res, off_a, off_b) in edit.loads {
                rewrites[inst.index()] = Some((edit.cond, edit.sa, off_a, edit.sb, off_b, res));
            }
        }
        let insts = std::mem::take(&mut func.blocks[blk].insts);
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            if drops[inst.index()] {
                continue;
            }
            if let Some((cond, sa, off_a, sb, off_b, res)) = rewrites[inst.index()] {
                let ty = func.ty(res);
                let va = func.value(ty);
                let vb = func.value(ty);
                let ia = func.add(Op::StackLoad(sa, off_a), Some(va));
                let ib = func.add(Op::StackLoad(sb, off_b), Some(vb));
                func.insts[inst.index()].op = Op::Select(cond, va, vb);
                out.push(ia);
                out.push(ib);
            }
            out.push(inst);
        }
        func.blocks[blk].insts = out;
    }
    changed
}

fn try_coalesce_param(func: &mut Function, cfg: &Cfg, addrs: &[Option<(StackSlot, i64)>], b: usize, pi: usize) -> bool {
    let block = Block(b as u32);
    if Some(block) == func.entry {
        return false;
    }
    let param = func.blocks[b].params[pi];
    if func.ty(param) != Type::I64 {
        return false;
    }
    let nblocks = func.blocks.len();
    let mut incoming: Vec<(usize, Value)> = Vec::new();
    for p_idx in 0..nblocks {
        let Some(last) = func.last(Block(p_idx as u32)) else { continue };
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
            Op::TryCall { normal, handler, .. } if *normal == block || *handler == block => return false,
            Op::BrTable(_, default, targets) if *default == block || targets.contains(&block) => return false,
            _ => {}
        }
    }
    if incoming.is_empty() {
        return false;
    }
    let mut common_off: Option<i32> = None;
    let mut slots: Vec<StackSlot> = Vec::new();
    let mut edge_slots: Vec<(usize, usize)> = Vec::with_capacity(incoming.len());
    for &(pred, arg) in &incoming {
        let Some((slot, off)) = addrs[arg.index()] else { return false };
        let Ok(off32) = i32::try_from(off) else { return false };
        match common_off {
            Some(prev) if prev != off32 => return false,
            None => common_off = Some(off32),
            _ => {}
        }
        let pos = slots.iter().position(|&s| s == slot).unwrap_or_else(|| {
            slots.push(slot);
            slots.len() - 1
        });
        edge_slots.push((pred, pos));
    }
    let Some(common_off) = common_off else { return false };
    let k = slots.len();
    if k >= 2 {
        let pos_of = |s: StackSlot| slots.iter().position(|&x| x == s);
        let in_slots = |v: Value| addrs[v.index()].and_then(|(s, _)| pos_of(s));
        let mut mem_blocks = vec![vec![false; nblocks]; k];
        for &(pred, pos) in &edge_slots {
            mem_blocks[pos][pred] = true;
        }
        let mut derived = vec![false; func.types.len()];
        derived[param.index()] = true;
        loop {
            let mut grew = false;
            for blk in 0..nblocks {
                for &inst in &func.blocks[blk].insts {
                    let data = &func.insts[inst.index()];
                    let mut op_derived = false;
                    data.op.for_operands(|v| op_derived |= derived[v.index()]);
                    if op_derived && let Some(r) = data.result && !derived[r.index()] {
                        derived[r.index()] = true;
                        grew = true;
                    }
                    let mut mark_args = |t: Block, args: &[Value]| {
                        for (idx, &v) in args.iter().enumerate() {
                            if derived[v.index()] && let Some(&tp) = func.blocks[t.index()].params.get(idx) && !derived[tp.index()] {
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
                match &data.op {
                    Op::StackLoad(s, _) | Op::StackStore(_, s, _) => {
                        if let Some(i) = pos_of(*s) {
                            mem_blocks[i][blk] = true;
                        }
                        if let Op::StackStore(val, _, _) = &data.op
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
                    Op::Binary(Binary::Iadd | Binary::Isub, a, b_) if data.result.is_some_and(|r| addrs[r.index()].is_some()) => {
                        if in_slots(*a).is_some() && in_slots(*b_).is_some() {
                            return false;
                        }
                    }
                    Op::Convert(Convert::Bitcast, _) if data.result.is_some_and(|r| addrs[r.index()].is_some()) => {}
                    Op::Select(c, ..) if data.result.is_some_and(|r| addrs[r.index()].is_some()) => {
                        if in_slots(*c).is_some() {
                            return false;
                        }
                    }
                    Op::Jump(target, args) => {
                        for (idx, &v) in args.iter().enumerate() {
                            if in_slots(v).is_some() && !(*target == block && idx == pi) {
                                return false;
                            }
                        }
                    }
                    Op::Brif(c, then, then_args, else_b, else_args) => {
                        if in_slots(*c).is_some() {
                            return false;
                        }
                        for (t, args) in [(*then, then_args), (*else_b, else_args)] {
                            for (idx, &v) in args.iter().enumerate() {
                                if in_slots(v).is_some() && !(t == block && idx == pi) {
                                    return false;
                                }
                            }
                        }
                    }
                    _ => {
                        let mut bad = false;
                        data.op.for_operands(|v| {
                            if in_slots(v).is_some() {
                                bad = true;
                            }
                        });
                        if bad {
                            return false;
                        }
                    }
                }
            }
        }
        let mut any_mem = vec![false; nblocks];
        for blk in 0..nblocks {
            let hits = mem_blocks.iter().filter(|m| m[blk]).count();
            if hits > 1 || (hits > 0 && blk == b) {
                return false;
            }
            any_mem[blk] = hits > 0;
        }
        let mut live_p = vec![false; nblocks];
        let mut stack = Vec::new();
        for u in uses_of_p {
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
        if (0..nblocks).any(|blk| live_p[blk] && any_mem[blk]) {
            return false;
        }
        if slots.iter().any(|&s| func.is_unwind_slot(s)) {
            return false;
        }
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
        let mut reach_b = vec![false; nblocks];
        let mut work: Vec<usize> = func.successors(block).map(|s| s.index()).collect();
        for &si in &work {
            reach_b[si] = true;
        }
        while let Some(cur) = work.pop() {
            for succ in func.successors(Block(cur as u32)) {
                let si = succ.index();
                if !reach_b[si] {
                    reach_b[si] = true;
                    work.push(si);
                }
            }
        }
        if slots.iter().any(|&s| func.is_unwind_slot(s)) {
            return false;
        }
        for i in 0..k {
            if (0..nblocks).any(|blk| reach_b[blk] && mem_blocks[i][blk]) {
                let edges_i: Vec<(usize, usize)> = edge_slots.iter().copied().filter(|&(_, pos)| pos == i).collect();
                if !loop_safe_writes(func, cfg, addrs, slots[i], &mem_blocks[i], &edges_i) {
                    return false;
                }
            }
        }
        let s0 = slots[0];
        func.slots[s0.index()].size = slots.iter().fold(0, |m, s| m.max(func.slots[s.index()].size));
        func.slots[s0.index()].align_shift = slots.iter().fold(0, |m, s| m.max(func.slots[s.index()].align_shift));
        for data in &mut func.insts {
            match &mut data.op {
                Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) if slots[1..].contains(s) => *s = s0,
                _ => {}
            }
        }
    }
    let s0 = slots[0];
    func.blocks[b].params.remove(pi);
    for blk in 0..nblocks {
        let Some(&last) = func.blocks[blk].insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, args) if *target == block => {
                args.remove(pi);
            }
            Op::Brif(_, then, then_args, else_b, else_args) => {
                if *then == block {
                    then_args.remove(pi);
                }
                if *else_b == block {
                    else_args.remove(pi);
                }
            }
            _ => {}
        }
    }
    let addr_inst = func.add(Op::StackAddr(s0, common_off), Some(param));
    func.blocks[b].insts.insert(0, addr_inst);
    true
}

fn unresolved_params(func: &Function, addrs: &[Option<(StackSlot, i64)>]) -> (Vec<usize>, Vec<bool>) {
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

fn coalesce_block_params(func: &mut Function, addrs: &[Option<(StackSlot, i64)>]) -> bool {
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

enum AddrOp {
    Add(Value, Value, bool),
    Bitcast(Value),
    Select(Value, Value),
}

/// Resolves slot-relative SSA address values across pointer arithmetic, bitcasts, and equal selects.
pub(super) fn resolve_addrs(func: &Function, consts: &[Option<i64>]) -> (Vec<Option<(StackSlot, i64)>>, Vec<(Inst, StackSlot)>) {
    let imm_of = |v: Value| consts.get(v.index()).copied().flatten();
    let (mut addrs, mut addr_insts) = func.slot_offsets(imm_of);
    let mut ops: Vec<(Inst, Value, AddrOp)> = Vec::new();
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        let Some(res) = data.result else { continue };
        if addrs[res.index()].is_some() {
            continue;
        }
        match data.op {
            Op::Binary(Binary::Iadd, x, c) => ops.push((*inst, res, AddrOp::Add(x, c, false))),
            Op::Binary(Binary::Isub, x, c) => ops.push((*inst, res, AddrOp::Add(x, c, true))),
            Op::Convert(Convert::Bitcast, x) if func.ty(res) == Type::I64 && func.ty(x) == Type::I64 => ops.push((*inst, res, AddrOp::Bitcast(x))),
            Op::Select(_, a, b) => ops.push((*inst, res, AddrOp::Select(a, b))),
            _ => {}
        }
    }
    loop {
        let before = ops.len();
        ops.retain(|(inst, res, op)| {
            let resolved = match *op {
                AddrOp::Add(x, c, sub) => {
                    let found = if sub {
                        addrs[x.index()].zip(imm_of(c).and_then(i64::checked_neg))
                    } else {
                        addrs[x.index()].zip(imm_of(c)).or_else(|| addrs[c.index()].zip(imm_of(x)))
                    };
                    found.and_then(|((slot, base), imm)| Some((slot, base.checked_add(imm)?)))
                }
                AddrOp::Bitcast(x) => addrs[x.index()],
                AddrOp::Select(a, b) => addrs[a.index()].filter(|pa| addrs[b.index()] == Some(*pa)),
            };
            let Some((slot, offset)) = resolved else { return true };
            addrs[res.index()] = Some((slot, offset));
            addr_insts.push((*inst, slot));
            false
        });
        if ops.len() == before {
            break;
        }
    }
    (addrs, addr_insts)
}

/// Returns which stack slots still have live address operands outside `dropped` and `addr_insts`.
pub(super) fn live_addr_slots(func: &Function, addrs: &[Option<(StackSlot, i64)>], addr_insts: &[(Inst, StackSlot)], dropped: &[bool]) -> Vec<bool> {
    let mut is_addr = vec![false; func.insts.len()];
    for &(inst, _) in addr_insts {
        is_addr[inst.index()] = true;
    }
    let mut live = vec![false; func.slots.len()];
    for block in &func.blocks {
        for &inst in &block.insts {
            if dropped[inst.index()] || is_addr[inst.index()] {
                continue;
            }
            func.insts[inst.index()].op.for_operands(|v| {
                if let Some((slot, _)) = addrs.get(v.index()).copied().flatten() {
                    live[slot.index()] = true;
                }
            });
        }
    }
    live
}

/// Coalesces disjoint arm-local slots at block joins and folds slot-select loads.
pub(super) fn run(func: &mut Function) -> bool {
    let mut changed = false;
    loop {
        let consts = super::subword::eval_consts(func);
        let (mut addrs, addr_insts) = resolve_addrs(func, &consts);
        if addr_insts.is_empty() {
            break;
        }
        let folded = fold_slot_selects(func, &addrs);
        if folded {
            addrs = resolve_addrs(func, &consts).0;
        }
        if !(folded | coalesce_block_params(func, &addrs)) {
            break;
        }
        changed = true;
    }
    changed
}
