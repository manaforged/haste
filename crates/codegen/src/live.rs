use crate::ir::{Block, Function, Inst, InstData, Op, StackSlot, Value};

pub(crate) fn order(func: &Function) -> Vec<Block> {
    let Some(entry) = func.entry else { return Vec::new() };
    let mut order = crate::dom::reverse_postorder(func, entry);
    let mut placed = vec![false; func.blocks.len()];
    order.iter().for_each(|block| placed[block.index()] = true);
    order.retain(|block| *block == entry || !func.blocks[block.index()].insts.is_empty());
    order.extend((0..func.blocks.len() as u32).map(Block).filter(|block| !placed[block.index()] && !func.blocks[block.index()].insts.is_empty()));
    order
}

const TEST_LIMIT: usize = 4;

pub(crate) fn copied_tests(func: &Function, order: &[Block]) -> Vec<bool> {
    let rank = crate::dom::ranks(order, func.blocks.len());
    let mut copied = vec![false; func.insts.len()];
    for (at, &latch) in order.iter().enumerate() {
        let Some(&last) = func.blocks[latch.index()].insts.last() else { continue };
        let Op::Jump(head, _) = func.insts[last.index()].op else { continue };
        let insts = &func.blocks[head.index()].insts;
        let calls = insts.iter().any(|inst| matches!(func.insts[inst.index()].op, Op::Call { .. }));
        copied[last.index()] = rank[head.index()] < at && insts.len() <= TEST_LIMIT && !calls && matches!(func.last(head), Some(Op::Brif(..)));
    }
    copied
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SibBase {
    Value(Value),
    Frame(StackSlot),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Sib {
    pub base: SibBase,
    pub index: Value,
    pub shift: u8,
    pub disp: i32,
}

pub(crate) fn for_effective_operands(data: &InstData, sib: &[Option<Sib>], elided: &[bool], mut f: impl FnMut(Value)) {
    if data.result.is_some_and(|res| elided.get(res.index()) == Some(&true)) {
        return;
    }
    let visit_addr = |addr: Value, f: &mut dyn FnMut(Value)| match sib.get(addr.index()).copied().flatten() {
        Some(mode) => {
            if let SibBase::Value(base) = mode.base {
                f(base);
            }
            f(mode.index);
        }
        None => f(addr),
    };
    match &data.op {
        Op::Load(addr, _) => visit_addr(*addr, &mut f),
        Op::Store(value, addr, _) => {
            f(*value);
            visit_addr(*addr, &mut f);
        }
        op => op.for_operands(f),
    }
}

pub(crate) struct Live {
    pub ranges: Vec<Range>,
    pub slice: Vec<(u32, u32)>,
    pub defs: Vec<u32>,
    pub unwinds: Vec<u32>,
    pub calls: Vec<u32>,
    pub sites: Vec<(Inst, Block)>,
    pub trapped: Vec<bool>,
}

impl Live {
    pub(crate) fn ranges_of(&self, value: usize) -> &[Range] {
        let (off, len) = self.slice[value];
        &self.ranges[off as usize..(off + len) as usize]
    }

    pub(crate) fn bounds(&self, value: usize) -> Option<(u32, u32)> {
        let ranges = self.ranges_of(value);
        Some((ranges.first()?.start, ranges.last()?.end))
    }

    pub(crate) fn last_end(&self, value: usize) -> Option<u32> {
        self.ranges_of(value).last().map(|r| r.end)
    }

    pub(crate) fn total_len(&self, value: usize) -> u64 {
        self.ranges_of(value).iter().map(|r| u64::from(r.end - r.start) + 1).sum::<u64>().max(1)
    }
}

pub(crate) fn overlaps(a: &[Range], b: &[Range]) -> bool {
    let (Some(a_last), Some(b_last)) = (a.last(), b.last()) else { return false };
    if a_last.end < b[0].start || b_last.end < a[0].start {
        return false;
    }
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].end < b[j].start {
            i += 1;
        } else if b[j].end < a[i].start {
            j += 1;
        } else {
            return true;
        }
    }
    false
}

struct Sets {
    words: usize,
    bits: Vec<u64>,
}

impl Sets {
    fn new(blocks: usize, values: usize) -> Sets {
        let words = values.div_ceil(64);
        Sets { words, bits: vec![0; blocks * words] }
    }

    fn row(&self, block: usize) -> &[u64] {
        &self.bits[block * self.words..(block + 1) * self.words]
    }

    fn has(&self, block: usize, dense: usize) -> bool {
        self.bits[block * self.words + dense / 64] >> (dense % 64) & 1 == 1
    }

    fn insert(&mut self, block: usize, dense: usize) {
        self.bits[block * self.words + dense / 64] |= 1 << (dense % 64);
    }

    fn members(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        self.row(block).iter().enumerate().flat_map(|(word, bits)| (0..64).filter(move |bit| bits >> bit & 1 == 1).map(move |bit| word * 64 + bit))
    }
}

fn crossing(func: &Function, wanted: &[bool], sib: &[Option<Sib>], elided: &[bool]) -> (Vec<Option<usize>>, Vec<Value>) {
    let mut location = vec![usize::MAX; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        data.params.iter().for_each(|param| location[param.index()] = block);
        for inst in &data.insts {
            let inst_data = &func.insts[inst.index()];
            if !inst_data.result.is_some_and(|res| elided.get(res.index()) == Some(&true)) {
                results(inst_data).for_each(|result| location[result.index()] = block);
            }
        }
    }
    let mut dense = vec![None; func.types.len()];
    let mut values = Vec::new();
    for (block, data) in func.blocks.iter().enumerate() {
        for inst in &data.insts {
            for_effective_operands(&func.insts[inst.index()], sib, elided, |value| {
                if wanted[value.index()] && location[value.index()] != block && dense[value.index()].is_none() {
                    dense[value.index()] = Some(values.len());
                    values.push(value);
                }
            });
        }
    }
    (dense, values)
}

fn local_sets(func: &Function, dense: &[Option<usize>], count: usize, sib: &[Option<Sib>], elided: &[bool]) -> (Sets, Sets) {
    let blocks = func.blocks.len();
    let (mut uses, mut defs) = (Sets::new(blocks, count), Sets::new(blocks, count));
    for (block, data) in func.blocks.iter().enumerate() {
        for param in data.params.iter().filter_map(|param| dense[param.index()]) {
            defs.insert(block, param);
        }
        for inst in &data.insts {
            let inst = &func.insts[inst.index()];
            if inst.result.is_some_and(|res| elided.get(res.index()) == Some(&true)) {
                continue;
            }
            for_effective_operands(inst, sib, elided, |value| {
                if let Some(value) = dense[value.index()]
                    && !defs.has(block, value)
                {
                    uses.insert(block, value);
                }
            });
            for result in results(inst).filter_map(|result| dense[result.index()]) {
                defs.insert(block, result);
            }
        }
    }
    (uses, defs)
}

fn live_in(func: &Function, order: &[Block], dense: &[Option<usize>], count: usize, sib: &[Option<Sib>], elided: &[bool]) -> Sets {
    let (uses, defs) = local_sets(func, dense, count, sib, elided);
    let mut live = Sets::new(func.blocks.len(), count);
    let words = live.words;
    let mut out = vec![0_u64; words];
    let mut changed = words > 0;
    while changed {
        changed = false;
        for block in order.iter().rev().map(|block| block.index()) {
            out.iter_mut().for_each(|word| *word = 0);
            for next in func.successors(Block(block as u32)) {
                out.iter_mut().zip(live.row(next.index())).for_each(|(word, bits)| *word |= bits);
            }
            for word in 0..words {
                let new = uses.row(block)[word] | (out[word] & !defs.row(block)[word]);
                let at = block * words + word;
                if live.bits[at] != new {
                    live.bits[at] = new;
                    changed = true;
                }
            }
        }
    }
    live
}

fn push_range(out: &mut [Vec<Range>], value: usize, start: u32, end: u32) {
    if let Some(last) = out[value].last_mut()
        && last.end + 1 >= start
    {
        last.end = last.end.max(end);
    } else {
        out[value].push(Range { start, end });
    }
}

pub(crate) fn live(func: &Function, order: &[Block], copied: &[bool], wanted: &[bool], sib: &[Option<Sib>], elided: &[bool]) -> Live {
    let count = func.types.len();
    let (dense, crossing) = crossing(func, wanted, sib, elided);
    let live_in = live_in(func, order, &dense, crossing.len(), sib, elided);
    let mut per_val: Vec<Vec<Range>> = vec![Vec::new(); count];
    let mut open: Vec<Option<(u32, u32)>> = vec![None; count];
    let mut touched: Vec<usize> = Vec::new();
    let mut defs = vec![0; count];
    let (mut unwinds, mut calls, mut sites) = (Vec::new(), Vec::new(), Vec::new());
    let mut trapped = vec![false; count];
    let mut at = 0_u32;
    for &block in order {
        let index = block.index();
        let b_start = at;
        for member in live_in.members(index) {
            let v = crossing[member].index();
            open[v] = Some((b_start, b_start));
            touched.push(v);
        }
        for &param in &func.blocks[index].params {
            let p = param.index();
            defs[p] = b_start;
            if wanted[p] && open[p].is_none() {
                open[p] = Some((b_start, b_start));
                touched.push(p);
            }
        }
        let use_val = |value: Value, pos: u32, open: &mut [Option<(u32, u32)>], touched: &mut Vec<usize>| {
            let v = value.index();
            if !wanted[v] {
                return;
            }
            match open[v] {
                Some((s, e)) => open[v] = Some((s, e.max(pos))),
                None => {
                    open[v] = Some((b_start, pos));
                    touched.push(v);
                }
            }
        };
        let def_val = |result: Value, pos: u32, primary: bool, defs: &mut [u32], open: &mut [Option<(u32, u32)>], touched: &mut Vec<usize>, per_val: &mut [Vec<Range>]| {
            let r = result.index();
            if primary {
                defs[r] = pos;
            }
            if !wanted[r] {
                return;
            }
            if let Some((s, e)) = open[r] {
                if e < pos {
                    push_range(per_val, r, s, e);
                    open[r] = Some((pos, pos));
                } else {
                    open[r] = Some((s, e.max(pos)));
                }
            } else {
                open[r] = Some((pos, pos));
                touched.push(r);
            }
        };
        let mut copied_head = None;
        let insts = &func.blocks[index].insts;
        for (place, &inst) in insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            if data.result.is_some_and(|res| elided.get(res.index()) == Some(&true)) {
                continue;
            }
            if copied.get(inst.index()) == Some(&true)
                && let Op::Jump(head, args) = &data.op
            {
                at += 1;
                for &arg in args {
                    use_val(arg, at, &mut open, &mut touched);
                }
                for &param in &func.blocks[head.index()].params {
                    def_val(param, at, false, &mut defs, &mut open, &mut touched, &mut per_val);
                }
                for &h_inst in &func.blocks[head.index()].insts {
                    let h_data = &func.insts[h_inst.index()];
                    if h_data.result.is_some_and(|res| elided.get(res.index()) == Some(&true)) {
                        continue;
                    }
                    at += 1;
                    for_effective_operands(h_data, sib, elided, |v| use_val(v, at, &mut open, &mut touched));
                    at += 1;
                    results(h_data).for_each(|r| def_val(r, at, false, &mut defs, &mut open, &mut touched, &mut per_val));
                }
                copied_head = Some(*head);
                continue;
            }
            if matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
                at += 1;
                for_effective_operands(data, sib, elided, |v| use_val(v, at, &mut open, &mut touched));
                at += 1;
                results(data).for_each(|r| def_val(r, at, true, &mut defs, &mut open, &mut touched, &mut per_val));
                if matches!(data.op, Op::TryCall { .. }) {
                    unwinds.push(at);
                }
                let traps = insts.get(place + 1).is_some_and(|next| matches!(func.insts[next.index()].op, Op::Trap));
                if matches!(data.op, Op::Call { .. }) && traps {
                    for_effective_operands(data, sib, elided, |v| trapped[v.index()] = true);
                } else {
                    calls.push(at);
                    sites.push((inst, block));
                }
                continue;
            }
            at += 1;
            for_effective_operands(data, sib, elided, |v| use_val(v, at, &mut open, &mut touched));
            at += 1;
            results(data).for_each(|r| def_val(r, at, true, &mut defs, &mut open, &mut touched, &mut per_val));
        }
        at += 1;
        let b_end = at;
        let exit_block = copied_head.unwrap_or(Block(index as u32));
        for next in func.successors(exit_block) {
            for member in live_in.members(next.index()) {
                let v = crossing[member].index();
                match open[v] {
                    Some((s, _)) => open[v] = Some((s, b_end)),
                    None => {
                        open[v] = Some((b_start, b_end));
                        touched.push(v);
                    }
                }
            }
        }
        for v in touched.drain(..) {
            if let Some((s, e)) = open[v].take() {
                push_range(&mut per_val, v, s, e);
            }
        }
        at += 1;
    }
    let mut ranges = Vec::new();
    let mut slice = Vec::with_capacity(count);
    for list in per_val {
        let off = ranges.len() as u32;
        let len = list.len() as u32;
        ranges.extend(list);
        slice.push((off, len));
    }
    Live { ranges, slice, defs, unwinds, calls, sites, trapped }
}

pub(crate) fn results(data: &InstData) -> impl Iterator<Item = Value> + '_ {
    let calls: &[Value] = match &data.op {
        Op::Call { results, .. } => results,
        _ => &[],
    };
    data.result.into_iter().chain(calls.iter().copied())
}
