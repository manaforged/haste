use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Function, Inst, Op, Value};
use crate::live::{BitMatrix, set_bits, solve_backward_liveness};
use crate::mem::Provenance;

struct SlotAnalysis {
    escaped: Vec<bool>,
    reads: BitMatrix,
    writes: BitMatrix,
    touches: BitMatrix,
    inst_touches: Vec<Vec<Vec<usize>>>,
}

fn collect_value_uses(func: &Function) -> Vec<Vec<(Block, Inst)>> {
    let mut uses = vec![Vec::new(); func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        let block_id = Block(b_idx as u32);
        for &inst in &block.insts {
            func.insts[inst.index()].op.for_operands(|value| uses[value.index()].push((block_id, inst)));
        }
    }
    uses
}

fn push_target_params(
    func: &Function,
    val: Value,
    (target, args): (Block, &[Value]),
    stamp: u32,
    queued: &mut [u32],
    work: &mut Vec<Value>,
) {
    for (pos, &arg) in args.iter().enumerate() {
        if arg == val
            && let Some(&param) = func.blocks[target.index()].params.get(pos)
            && std::mem::replace(&mut queued[param.index()], stamp) != stamp
        {
            work.push(param);
        }
    }
}

fn propagate_alias_uses(func: &Function, slot_alias: &[Vec<usize>], extra_uses: &mut [Vec<(Block, Inst)>]) {
    for (dst, srcs) in slot_alias.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        for (b_idx, block) in func.blocks.iter().enumerate() {
            let block_id = Block(b_idx as u32);
            for &inst in &block.insts {
                match func.insts[inst.index()].op {
                    Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) if s.index() == dst => {
                        for &src in srcs {
                            extra_uses[src].push((block_id, inst));
                        }
                    }
                    _ => {}
                }
            }
        }
        let dst_extra = extra_uses[dst].clone();
        for &src in srcs {
            extra_uses[src].extend(dst_extra.iter().copied());
        }
    }
}

fn step_addr_use(
    func: &Function,
    (slot, val, inst): (usize, Value, Inst),
    stamp: u32,
    (slot_alias, queued, work): (&mut [Vec<usize>], &mut [u32], &mut Vec<Value>),
) {
    let data = &func.insts[inst.index()];
    match &data.op {
        Op::Binary(Binary::Iadd | Binary::Isub, ..) | Op::Select(..) => {
            if let Some(res) = data.result
                && std::mem::replace(&mut queued[res.index()], stamp) != stamp
            {
                work.push(res);
            }
        }
        Op::StackStore(stored, dst, _) if *stored == val && !slot_alias[dst.index()].contains(&slot) => {
            slot_alias[dst.index()].push(slot);
        }
        Op::Jump(target, args) => push_target_params(func, val, (*target, args), stamp, queued, work),
        Op::Brif(_, t, t_args, f, f_args) => {
            push_target_params(func, val, (*t, t_args), stamp, queued, work);
            push_target_params(func, val, (*f, f_args), stamp, queued, work);
        }
        _ => {}
    }
}

type SlotTrace = (Vec<bool>, Vec<bool>, Vec<Vec<(Block, Inst)>>);

fn trace_stack_addrs(func: &Function, value_uses: &[Vec<(Block, Inst)>], is_spill: &[bool]) -> SlotTrace {
    let slots = func.slots.len();
    let escaped = Provenance::compute_with_spills(func, Some(is_spill)).escaped_slots();
    let mut taken = vec![false; slots];
    let mut extra_uses = vec![Vec::new(); slots];
    let mut addr_roots: Vec<Vec<usize>> = vec![Vec::new(); slots];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let Op::StackAddr(slot, _) = data.op {
            taken[slot.index()] = true;
            if let Some(res) = data.result {
                addr_roots[slot.index()].push(res.index());
            }
        }
    }
    let mut slot_alias: Vec<Vec<usize>> = vec![Vec::new(); slots];
    let mut queued = vec![0_u32; func.types.len()];
    let mut work = Vec::new();
    for (slot, roots) in addr_roots.iter().enumerate() {
        if roots.is_empty() {
            continue;
        }
        let stamp = (slot + 1) as u32;
        work.clear();
        for &root in roots {
            queued[root] = stamp;
            work.push(Value(root as u32));
        }
        while let Some(val) = work.pop() {
            for &(block, inst) in &value_uses[val.index()] {
                extra_uses[slot].push((block, inst));
                step_addr_use(func, (slot, val, inst), stamp, (&mut slot_alias, &mut queued, &mut work));
            }
        }
    }
    propagate_alias_uses(func, &slot_alias, &mut extra_uses);
    (taken, escaped, extra_uses)
}

fn record_slot_read(reads: &mut BitMatrix, writes: &BitMatrix, touches: &mut BitMatrix, b_idx: usize, slot: usize) {
    if !writes.get(b_idx, slot) {
        reads.set(b_idx, slot);
    }
    touches.set(b_idx, slot);
}

fn slot_effects(func: &Function, value_uses: &[Vec<(Block, Inst)>], is_spill: &[bool]) -> (Vec<bool>, SlotAnalysis) {
    let (slots, blocks) = (func.slots.len(), func.blocks.len());
    let (taken, escaped, extra_uses) = trace_stack_addrs(func, value_uses, is_spill);
    let mut reads = BitMatrix::new(blocks, slots);
    let mut writes = BitMatrix::new(blocks, slots);
    let mut touches = BitMatrix::new(blocks, slots);
    let mut inst_pos: HashMap<(usize, usize), usize> = HashMap::default();
    let mut inst_touches = vec![Vec::new(); blocks];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        inst_touches[b_idx] = vec![Vec::new(); block.insts.len()];
        for (pos, &inst) in block.insts.iter().enumerate() {
            inst_pos.insert((b_idx, inst.index()), pos);
            match func.insts[inst.index()].op {
                Op::StackAddr(slot, _) | Op::StackLoad(slot, _) => {
                    let s = slot.index();
                    record_slot_read(&mut reads, &writes, &mut touches, b_idx, s);
                    inst_touches[b_idx][pos].push(s);
                }
                Op::StackStore(value, slot, off) => {
                    let s = slot.index();
                    if off == 0 && func.types[value.index()].bytes() >= func.slots[s].size {
                        writes.set(b_idx, s);
                        touches.set(b_idx, s);
                    } else {
                        record_slot_read(&mut reads, &writes, &mut touches, b_idx, s);
                    }
                    inst_touches[b_idx][pos].push(s);
                }
                _ => {}
            }
        }
    }
    for (slot, list) in extra_uses.iter().enumerate() {
        for &(block, inst) in list {
            let b_idx = block.index();
            record_slot_read(&mut reads, &writes, &mut touches, b_idx, slot);
            if let Some(&pos) = inst_pos.get(&(b_idx, inst.index()))
                && !inst_touches[b_idx][pos].contains(&slot)
            {
                inst_touches[b_idx][pos].push(slot);
            }
        }
    }
    (taken, SlotAnalysis { escaped, reads, writes, touches, inst_touches })
}

fn written_before(func: &Function, resumes: &[Block], analysis: &SlotAnalysis, live_in: &BitMatrix) -> BitMatrix {
    let (slots, blocks) = (func.slots.len(), func.blocks.len());
    let mut before = BitMatrix::new(blocks, slots);
    let words = before.words();
    let mut scratch = vec![0_u64; words];
    for &resume in resumes {
        before.or_row(resume.index(), live_in.row(resume.index()));
    }
    let mut changed = words > 0;
    while changed {
        changed = false;
        for b in 0..blocks {
            let touches = analysis.touches.row(b);
            for (w, &bw) in before.row(b).iter().enumerate() {
                scratch[w] = bw | touches[w];
            }
            for next in func.successors(Block(b as u32)) {
                for (dst, &reached) in before.row_mut(next.index()).iter_mut().zip(&scratch) {
                    let added = reached & !*dst;
                    if added != 0 {
                        *dst |= added;
                        changed = true;
                    }
                }
            }
        }
    }
    before
}

fn add_clique(conflicts: &mut BitMatrix, mask: &[u64]) {
    for s in set_bits(mask) {
        conflicts.or_row(s, mask);
    }
}

fn block_conflicts(
    func: &Function,
    b_idx: usize,
    kept_mask: &[u64],
    (analysis, live_in, before): (&SlotAnalysis, &BitMatrix, &BitMatrix),
    conflicts: &mut BitMatrix,
) {
    let slots = func.slots.len();
    let words = conflicts.words();
    let mut first_touch = vec![usize::MAX; slots];
    for (pos, touched) in analysis.inst_touches[b_idx].iter().enumerate() {
        for &s in touched {
            if first_touch[s] == usize::MAX {
                first_touch[s] = pos;
            }
        }
    }
    let mut out_row = vec![0_u64; words];
    live_in.fill_succ_union(func, Block(b_idx as u32), &mut out_row);
    let (bef_row, tch_row) = (before.row(b_idx), analysis.touches.row(b_idx));
    let mut live: Vec<u64> = (0..words).map(|w| kept_mask[w] & out_row[w] & (bef_row[w] | tch_row[w])).collect();
    add_clique(conflicts, &live);
    let block = &func.blocks[b_idx];
    for (pos, &inst) in block.insts.iter().enumerate().rev() {
        for &s in &analysis.inst_touches[b_idx][pos] {
            if (kept_mask[s >> 6] >> (s & 63)) & 1 == 0 {
                continue;
            }
            conflicts.or_row(s, &live);
            let whole_kill = matches!(func.insts[inst.index()].op, Op::StackStore(v, slot, 0) if slot.index() == s && func.types[v.index()].bytes() >= func.slots[s].size);
            if whole_kill || (first_touch[s] == pos && !before.get(b_idx, s)) {
                live[s >> 6] &= !(1_u64 << (s & 63));
            } else {
                live[s >> 6] |= 1_u64 << (s & 63);
            }
        }
    }
    for (l, &b) in live.iter_mut().zip(bef_row) {
        *l &= b;
    }
    add_clique(conflicts, &live);
}

fn symmetrize_conflicts(conflicts: &mut BitMatrix, slots: usize) {
    for s in 0..slots {
        let mut row = conflicts.row(s).to_vec();
        row[s >> 6] &= !(1_u64 << (s & 63));
        conflicts.row_mut(s).copy_from_slice(&row);
        for t in set_bits(&row) {
            conflicts.set(t, s);
        }
    }
}

fn active_coroutine_blocks(func: &Function, cfg: &Cfg, resumes: &[Block]) -> Vec<bool> {
    let mut active = vec![false; func.blocks.len()];
    for &b in &cfg.rpo {
        active[b.index()] = true;
    }
    let mut stack: Vec<Block> = resumes.to_vec();
    while let Some(b) = stack.pop() {
        if !std::mem::replace(&mut active[b.index()], true) {
            stack.extend(func.successors(b));
        }
    }
    active
}

pub(super) fn framed(func: &Function, cfg: &Cfg, resumes: &[Block], is_spill: &[bool]) -> (Vec<bool>, BitMatrix, Vec<bool>) {
    let (slots, blocks) = (func.slots.len(), func.blocks.len());
    let value_uses = collect_value_uses(func);
    let (taken, analysis) = slot_effects(func, &value_uses, is_spill);
    let live_in = solve_backward_liveness(func, (0..blocks).map(|b| Block(b as u32)), &analysis.reads, &analysis.writes);
    let kept: Vec<bool> = (0..slots).map(|at| (taken[at] && analysis.escaped[at]) || resumes.iter().any(|resume| live_in.get(resume.index(), at))).collect();
    let before = written_before(func, resumes, &analysis, &live_in);
    let mut conflicts = BitMatrix::new(slots, slots);
    let mut kept_mask = vec![0_u64; conflicts.words()];
    for (s, &k) in kept.iter().enumerate() {
        if k {
            kept_mask[s >> 6] |= 1_u64 << (s & 63);
        }
    }
    for (s, (&k, &esc)) in kept.iter().zip(&analysis.escaped).enumerate() {
        if k && esc {
            conflicts.or_row(s, &kept_mask);
        }
    }
    let active = active_coroutine_blocks(func, cfg, resumes);
    for (b_idx, &is_active) in active.iter().enumerate() {
        if is_active {
            block_conflicts(func, b_idx, &kept_mask, (&analysis, &live_in, &before), &mut conflicts);
        }
    }
    symmetrize_conflicts(&mut conflicts, slots);
    (kept, conflicts, analysis.escaped)
}
