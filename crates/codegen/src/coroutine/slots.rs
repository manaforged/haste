use rustc_hash::FxHashMap as HashMap;

use crate::cfg::Cfg;
use crate::ir::{Binary, Block, Convert, Function, Inst, Op, Value};
use crate::live::{BitMatrix, set_bits, solve_backward_liveness};
use crate::mem::Provenance;

pub(super) struct FramedSlots {
    pub raw_of: Vec<usize>,
    pub kept: Vec<bool>,
    pub conflicts: BitMatrix,
    pub escaped: Vec<bool>,
}

struct SlotAnalysis {
    escaped: Vec<bool>,
    reads: BitMatrix,
    writes: BitMatrix,
    touches: BitMatrix,
    inst_touches: Vec<Vec<(u32, u32)>>,
}

fn dense_slot_map(func: &Function) -> (Vec<u32>, Vec<usize>) {
    let mut referenced = vec![false; func.slots.len()];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        match func.insts[inst.index()].op {
            Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) => {
                referenced[s.index()] = true
            }
            _ => {}
        }
    }
    let mut dense_of = vec![u32::MAX; func.slots.len()];
    let mut raw_of = Vec::new();
    for (raw, &used) in referenced.iter().enumerate() {
        if used {
            dense_of[raw] = raw_of.len() as u32;
            raw_of.push(raw);
        }
    }
    (dense_of, raw_of)
}

fn collect_value_uses(func: &Function) -> Vec<Vec<(Block, Inst)>> {
    let mut uses = vec![Vec::new(); func.types.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        let block_id = Block(b_idx as u32);
        for &inst in &block.insts {
            func.insts[inst.index()]
                .op
                .for_operands(|value| uses[value.index()].push((block_id, inst)));
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

fn propagate_alias_uses(
    func: &Function,
    dense_of: &[u32],
    slot_alias: &[Vec<usize>],
    extra_uses: &mut [Vec<(Block, Inst)>],
) {
    if !slot_alias.iter().any(|srcs| !srcs.is_empty()) {
        return;
    }
    for (b_idx, block) in func.blocks.iter().enumerate() {
        let block_id = Block(b_idx as u32);
        for &inst in &block.insts {
            let raw = match func.insts[inst.index()].op {
                Op::StackAddr(s, _) | Op::StackLoad(s, _) | Op::StackStore(_, s, _) => s.index(),
                _ => continue,
            };
            let dst = dense_of[raw] as usize;
            for &src in &slot_alias[dst] {
                extra_uses[src].push((block_id, inst));
            }
        }
    }
    for (dst, srcs) in slot_alias.iter().enumerate() {
        if srcs.is_empty() {
            continue;
        }
        let dst_extra = extra_uses[dst].clone();
        for &src in srcs {
            extra_uses[src].extend(dst_extra.iter().copied());
        }
    }
}

fn step_addr_use(
    func: &Function,
    dense_of: &[u32],
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
        Op::StackStore(stored, dst, _) if *stored == val => {
            let dst_d = dense_of[dst.index()] as usize;
            if !slot_alias[dst_d].contains(&slot) {
                slot_alias[dst_d].push(slot);
            }
        }
        Op::Jump(target, args) => {
            push_target_params(func, val, (*target, args), stamp, queued, work)
        }
        Op::Brif(_, t, t_args, f, f_args) => {
            push_target_params(func, val, (*t, t_args), stamp, queued, work);
            push_target_params(func, val, (*f, f_args), stamp, queued, work);
        }
        _ => {}
    }
}

type SlotTrace = (Vec<bool>, Vec<bool>, Vec<bool>, Vec<Vec<(Block, Inst)>>);

fn blocks_reaching_suspend(func: &Function, done: &[Block]) -> Vec<bool> {
    let n = func.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut can_suspend = vec![false; n];
    let mut stack = Vec::new();
    for b in 0..n {
        let blk = Block(b as u32);
        if matches!(func.last(blk), Some(Op::Return(..))) && !done.contains(&blk) {
            can_suspend[b] = true;
            stack.push(b);
        }
        for succ in func.successors(blk) {
            preds[succ.index()].push(b);
        }
    }
    while let Some(b) = stack.pop() {
        for &p in &preds[b] {
            if !std::mem::replace(&mut can_suspend[p], true) {
                stack.push(p);
            }
        }
    }
    can_suspend
}

fn slot_byte_mask(off: i32, bytes: u32, size: u32) -> Option<u128> {
    if size == 0 || size > 128 || off < 0 {
        return None;
    }
    let lo = off as u32;
    let hi = lo.saturating_add(bytes).min(size);
    if lo >= hi {
        return Some(0);
    }
    let width = hi - lo;
    let ones = if width == 128 {
        u128::MAX
    } else {
        (1_u128 << width) - 1
    };
    Some(ones << lo)
}

fn full_byte_mask(size: u32) -> u128 {
    if size == 0 {
        0
    } else if size >= 128 {
        u128::MAX
    } else {
        (1_u128 << size) - 1
    }
}

fn compute_read_masks(func: &Function, dense_of: &[u32], slots: usize) -> Vec<u128> {
    let mut masks = vec![0_u128; slots];
    for inst in func.blocks.iter().flat_map(|b| &b.insts) {
        let data = &func.insts[inst.index()];
        if let Op::StackLoad(slot, off) = data.op
            && let Some(res) = data.result
        {
            let s = dense_of[slot.index()] as usize;
            let size = func.slots[slot.index()].size;
            let bytes = func.types[res.index()].bytes();
            if let Some(bits) = slot_byte_mask(off, bytes, size) {
                masks[s] |= bits;
            } else {
                masks[s] = u128::MAX;
            }
        }
    }
    masks
}

fn trace_stack_addrs(
    func: &Function,
    (dense_of, raw_of): (&[u32], &[usize]),
    value_uses: &[Vec<(Block, Inst)>],
    (can_suspend, is_spill): (&[bool], &[bool]),
) -> SlotTrace {
    let slots = raw_of.len();
    let raw_escaped = Provenance::compute_with_spills(func, Some(is_spill)).addr_escaped_slots();
    let escaped: Vec<bool> = raw_of.iter().map(|&raw| raw_escaped[raw]).collect();
    let mut taken = vec![false; slots];
    let mut taken_can_suspend = vec![false; slots];
    let mut extra_uses = vec![Vec::new(); slots];
    let mut addr_roots: Vec<Vec<usize>> = vec![Vec::new(); slots];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Op::StackAddr(slot, _) = data.op {
                let d = dense_of[slot.index()] as usize;
                taken[d] = true;
                taken_can_suspend[d] |= can_suspend[b_idx];
                if let Some(res) = data.result {
                    addr_roots[d].push(res.index());
                }
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
                step_addr_use(
                    func,
                    dense_of,
                    (slot, val, inst),
                    stamp,
                    (&mut slot_alias, &mut queued, &mut work),
                );
            }
        }
    }
    propagate_alias_uses(func, dense_of, &slot_alias, &mut extra_uses);
    (taken, taken_can_suspend, escaped, extra_uses)
}

fn record_slot_read(
    reads: &mut BitMatrix,
    writes: &BitMatrix,
    touches: &mut BitMatrix,
    b_idx: usize,
    slot: usize,
) {
    if !writes.get(b_idx, slot) {
        reads.set(b_idx, slot);
    }
    touches.set(b_idx, slot);
}

fn build_inst_pos(func: &Function) -> HashMap<(usize, usize), u32> {
    let mut inst_pos = HashMap::default();
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for (pos, &inst) in block.insts.iter().enumerate() {
            inst_pos.insert((b_idx, inst.index()), pos as u32);
        }
    }
    inst_pos
}

fn push_if_unseen(v: Value, stamp: u32, seen: &mut [u32], work: &mut Vec<Value>) {
    if std::mem::replace(&mut seen[v.index()], stamp) != stamp {
        work.push(v);
    }
}

fn addr_use_escapes_poll(
    func: &Function,
    (dense_of, taken, live_at_resume, slot_loads): (&[u32], &[bool], &[bool], &[Vec<Value>]),
    (val, inst, stamp): (Value, Inst, u32),
    (seen_val, seen_holder, work): (&mut [u32], &mut [u32], &mut Vec<Value>),
) -> bool {
    let data = &func.insts[inst.index()];
    match &data.op {
        Op::Load(addr, _) | Op::AtomicLoad(addr, _) | Op::Trapz(addr) => *addr != val,
        Op::Store(stored, addr, _) | Op::AtomicStore(stored, addr, _) => {
            *addr != val || *stored == val
        }
        Op::AtomicRmw(_, _, addr, stored) => *addr != val || *stored == val,
        Op::AtomicCas(_, addr, exp, new) => *addr != val || *exp == val || *new == val,
        Op::Move(..) | Op::Icmp(..) => false,
        Op::Copy(_, _, len) => *len == val,
        Op::Binary(Binary::Iadd, a, b) => {
            if (*a == val) == (*b == val) {
                return true;
            }
            if let Some(res) = data.result {
                push_if_unseen(res, stamp, seen_val, work);
            }
            false
        }
        Op::Binary(Binary::Isub, a, b) => {
            if *a != val || *b == val {
                return true;
            }
            if let Some(res) = data.result {
                push_if_unseen(res, stamp, seen_val, work);
            }
            false
        }
        Op::Convert(Convert::Bitcast, a) => {
            if *a != val {
                return true;
            }
            if let Some(res) = data.result {
                push_if_unseen(res, stamp, seen_val, work);
            }
            false
        }
        Op::Select(cond, a, b) => {
            if *cond == val && *a != val && *b != val {
                return false;
            }
            if let Some(res) = data.result {
                push_if_unseen(res, stamp, seen_val, work);
            }
            false
        }
        Op::Jump(target, args) => {
            push_target_params(func, val, (*target, args), stamp, seen_val, work);
            false
        }
        Op::Brif(_, t, t_args, f, f_args) => {
            push_target_params(func, val, (*t, t_args), stamp, seen_val, work);
            push_target_params(func, val, (*f, f_args), stamp, seen_val, work);
            false
        }
        Op::StackStore(stored, dst, _) if *stored == val => {
            let dst_d = dense_of[dst.index()] as usize;
            if taken[dst_d] || live_at_resume[dst_d] || func.is_unwind_slot(*dst) {
                return true;
            }
            if std::mem::replace(&mut seen_holder[dst_d], stamp) != stamp {
                for &loaded in &slot_loads[dst_d] {
                    push_if_unseen(loaded, stamp, seen_val, work);
                }
            }
            false
        }
        _ => true,
    }
}

fn slot_addrs_escape_poll(
    func: &Function,
    (dense_of, raw_of): (&[u32], &[usize]),
    value_uses: &[Vec<(Block, Inst)>],
    (taken, live_at_resume, is_spill): (&[bool], &[bool], &[bool]),
) -> Vec<bool> {
    let slots = raw_of.len();
    let mut addr_roots: Vec<Vec<Value>> = vec![Vec::new(); slots];
    let mut slot_loads: Vec<Vec<Value>> = vec![Vec::new(); slots];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            match data.op {
                Op::StackAddr(slot, _) => addr_roots[dense_of[slot.index()] as usize].push(res),
                Op::StackLoad(slot, _) => slot_loads[dense_of[slot.index()] as usize].push(res),
                _ => {}
            }
        }
    }
    let mut escapes = vec![false; slots];
    let mut seen_val = vec![0_u32; func.types.len()];
    let mut seen_holder = vec![0_u32; slots];
    let mut work = Vec::new();
    for (s, roots) in addr_roots.iter().enumerate() {
        if roots.is_empty() || live_at_resume[s] {
            continue;
        }
        let stamp = (s + 1) as u32;
        work.clear();
        for &root in roots {
            push_if_unseen(root, stamp, &mut seen_val, &mut work);
        }
        while let Some(val) = work.pop() {
            if is_spill.get(val.index()).copied().unwrap_or(false) {
                escapes[s] = true;
                break;
            }
            for &(_, inst) in &value_uses[val.index()] {
                if addr_use_escapes_poll(
                    func,
                    (dense_of, taken, live_at_resume, &slot_loads),
                    (val, inst, stamp),
                    (&mut seen_val, &mut seen_holder, &mut work),
                ) {
                    escapes[s] = true;
                    break;
                }
            }
            if escapes[s] {
                break;
            }
        }
    }
    escapes
}

fn slot_effects(
    func: &Function,
    (dense_of, raw_of): (&[u32], &[usize]),
    value_uses: &[Vec<(Block, Inst)>],
    (can_suspend, is_spill): (&[bool], &[bool]),
) -> (Vec<bool>, Vec<bool>, SlotAnalysis) {
    let (slots, blocks) = (raw_of.len(), func.blocks.len());
    let (taken, taken_can_suspend, escaped, extra_uses) = trace_stack_addrs(
        func,
        (dense_of, raw_of),
        value_uses,
        (can_suspend, is_spill),
    );
    let read_masks = compute_read_masks(func, dense_of, slots);
    let mut reads = BitMatrix::new(blocks, slots);
    let mut writes = BitMatrix::new(blocks, slots);
    let mut touches = BitMatrix::new(blocks, slots);
    let inst_pos = extra_uses
        .iter()
        .any(|list| !list.is_empty())
        .then(|| build_inst_pos(func));
    let mut extra_min_pos: HashMap<(usize, usize), u32> = HashMap::default();
    if let Some(inst_pos) = &inst_pos {
        for (slot, list) in extra_uses.iter().enumerate() {
            for &(block, inst) in list {
                let b_idx = block.index();
                if let Some(&pos) = inst_pos.get(&(b_idx, inst.index())) {
                    extra_min_pos
                        .entry((b_idx, slot))
                        .and_modify(|cur| *cur = (*cur).min(pos))
                        .or_insert(pos);
                }
            }
        }
    }
    let mut inst_touches: Vec<Vec<(u32, u32)>> = vec![Vec::new(); blocks];
    let mut written: HashMap<usize, u128> = HashMap::default();
    for (b_idx, block) in func.blocks.iter().enumerate() {
        written.clear();
        for (pos, &inst) in block.insts.iter().enumerate() {
            match func.insts[inst.index()].op {
                Op::StackAddr(slot, _) | Op::StackLoad(slot, _) => {
                    let s = dense_of[slot.index()] as usize;
                    record_slot_read(&mut reads, &writes, &mut touches, b_idx, s);
                    inst_touches[b_idx].push((pos as u32, s as u32));
                }
                Op::StackStore(value, slot, off) => {
                    let s = dense_of[slot.index()] as usize;
                    if extra_min_pos
                        .get(&(b_idx, s))
                        .is_some_and(|&p| p < pos as u32)
                        && !writes.get(b_idx, s)
                    {
                        reads.set(b_idx, s);
                    }
                    let size = func.slots[slot.index()].size;
                    let bytes = func.types[value.index()].bytes();
                    if !writes.get(b_idx, s) && !reads.get(b_idx, s) {
                        let required = if taken[s] {
                            full_byte_mask(size)
                        } else {
                            read_masks[s]
                        };
                        if (off == 0 && bytes >= size)
                            || (!taken[s] && size <= 128 && required == 0)
                        {
                            writes.set(b_idx, s);
                        } else if let Some(bits) = slot_byte_mask(off, bytes, size) {
                            let mask = written.entry(s).or_insert(0);
                            *mask |= bits;
                            if (*mask & required) == required {
                                writes.set(b_idx, s);
                            }
                        } else {
                            reads.set(b_idx, s);
                        }
                    }
                    touches.set(b_idx, s);
                    inst_touches[b_idx].push((pos as u32, s as u32));
                }
                _ => {}
            }
        }
        for (s, mask) in written.drain() {
            if mask != 0 && !writes.get(b_idx, s) {
                reads.set(b_idx, s);
            }
        }
    }
    if let Some(inst_pos) = &inst_pos {
        for (slot, list) in extra_uses.iter().enumerate() {
            for &(block, inst) in list {
                let b_idx = block.index();
                record_slot_read(&mut reads, &writes, &mut touches, b_idx, slot);
                if let Some(&pos) = inst_pos.get(&(b_idx, inst.index())) {
                    inst_touches[b_idx].push((pos, slot as u32));
                }
            }
        }
    }
    for list in &mut inst_touches {
        list.sort_by_key(|&(pos, _)| pos);
        list.dedup();
    }
    (
        taken,
        taken_can_suspend,
        SlotAnalysis {
            escaped,
            reads,
            writes,
            touches,
            inst_touches,
        },
    )
}

fn written_before(
    func: &Function,
    slots: usize,
    resumes: &[Block],
    analysis: &SlotAnalysis,
    live_in: &BitMatrix,
) -> BitMatrix {
    let blocks = func.blocks.len();
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
    (func, raw_of): (&Function, &[usize]),
    b_idx: usize,
    kept_mask: &[u64],
    (analysis, live_in, before): (&SlotAnalysis, &BitMatrix, &BitMatrix),
    conflicts: &mut BitMatrix,
) {
    let slots = raw_of.len();
    let words = conflicts.words();
    let touches = &analysis.inst_touches[b_idx];
    let mut first_touch = vec![u32::MAX; slots];
    for &(pos, s) in touches {
        let su = s as usize;
        if first_touch[su] == u32::MAX {
            first_touch[su] = pos;
        }
    }
    let mut out_row = vec![0_u64; words];
    live_in.fill_succ_union(func, Block(b_idx as u32), &mut out_row);
    let (bef_row, tch_row) = (before.row(b_idx), analysis.touches.row(b_idx));
    let mut live: Vec<u64> = (0..words)
        .map(|w| kept_mask[w] & out_row[w] & (bef_row[w] | tch_row[w]))
        .collect();
    add_clique(conflicts, &live);
    let block = &func.blocks[b_idx];
    let mut end = touches.len();
    while end > 0 {
        let pos = touches[end - 1].0;
        let start = touches[..end].partition_point(|&(p, _)| p < pos);
        let inst = block.insts[pos as usize];
        for &(_, s_u32) in &touches[start..end] {
            let s = s_u32 as usize;
            if (kept_mask[s >> 6] >> (s & 63)) & 1 == 0 {
                continue;
            }
            conflicts.or_row(s, &live);
            let raw_s = raw_of[s];
            let whole_kill = matches!(func.insts[inst.index()].op, Op::StackStore(v, slot, 0) if slot.index() == raw_s && func.types[v.index()].bytes() >= func.slots[raw_s].size);
            if whole_kill || (first_touch[s] == pos && !before.get(b_idx, s)) {
                live[s >> 6] &= !(1_u64 << (s & 63));
            } else {
                live[s >> 6] |= 1_u64 << (s & 63);
            }
        }
        end = start;
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

pub(super) fn framed(
    func: &Function,
    cfg: &Cfg,
    (resumes, done): (&[Block], &[Block]),
    is_spill: &[bool],
) -> FramedSlots {
    let (dense_of, raw_of) = dense_slot_map(func);
    let (slots, blocks) = (raw_of.len(), func.blocks.len());
    if slots == 0 {
        return FramedSlots {
            raw_of,
            kept: Vec::new(),
            conflicts: BitMatrix::new(0, 0),
            escaped: Vec::new(),
        };
    }
    let can_suspend = blocks_reaching_suspend(func, done);
    let value_uses = collect_value_uses(func);
    let (taken, taken_can_suspend, analysis) = slot_effects(
        func,
        (&dense_of, &raw_of),
        &value_uses,
        (&can_suspend, is_spill),
    );
    let live_in = solve_backward_liveness(
        func,
        (0..blocks).map(|b| Block(b as u32)),
        &analysis.reads,
        &analysis.writes,
    );
    let live_at_resume: Vec<bool> = (0..slots)
        .map(|at| resumes.iter().any(|resume| live_in.get(resume.index(), at)))
        .collect();
    let escapes_poll = slot_addrs_escape_poll(
        func,
        (&dense_of, &raw_of),
        &value_uses,
        (&taken, &live_at_resume, is_spill),
    );
    let kept: Vec<bool> = (0..slots)
        .map(|at| {
            !func.slots[raw_of[at]].poll_local
                && (live_at_resume[at] || (taken_can_suspend[at] && escapes_poll[at]))
        })
        .collect();
    let before = written_before(func, slots, resumes, &analysis, &live_in);
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
            block_conflicts(
                (func, &raw_of),
                b_idx,
                &kept_mask,
                (&analysis, &live_in, &before),
                &mut conflicts,
            );
        }
    }
    symmetrize_conflicts(&mut conflicts, slots);
    FramedSlots {
        raw_of,
        kept,
        conflicts,
        escaped: analysis.escaped,
    }
}
