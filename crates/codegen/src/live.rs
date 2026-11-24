mod order;

pub(crate) use order::{
    cold, cold_and_detours, copied_tests, has_regular_call, order, quick_order,
};

use crate::ir::{Block, Function, Inst, InstData, Op, StackSlot, TargetOpId, Value};

/// Whether the instruction at `pos` of `insts` takes the carry out of the `adc` or `sbb` just before it on the same operands, so it reads only the flags.
pub(crate) fn carry_out_of_pair(func: &Function, insts: &[Inst], pos: usize) -> bool {
    let Some(prev) = pos.checked_sub(1) else {
        return false;
    };
    match (
        &func.insts[insts[prev].index()].op,
        &func.insts[insts[pos].index()].op,
    ) {
        (
            Op::TargetOp(TargetOpId::X64AddCarry, sum),
            Op::TargetOp(TargetOpId::X64AddCarryOut, out),
        )
        | (
            Op::TargetOp(TargetOpId::X64SubBorrow, sum),
            Op::TargetOp(TargetOpId::X64SubBorrowOut, out),
        ) => sum.as_slice() == out.as_slice(),
        _ => false,
    }
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

#[derive(Clone, Copy)]
pub(crate) struct Elided<'a> {
    pub sib: &'a [Option<Sib>],
    pub atomic_base: &'a [Option<Value>],
    pub elided: &'a [bool],
}

fn is_elided(data: &InstData, eff: Elided<'_>) -> bool {
    data.result
        .is_some_and(|res| eff.elided.get(res.index()) == Some(&true))
}

pub(crate) fn for_effective_operands(data: &InstData, eff: Elided<'_>, mut f: impl FnMut(Value)) {
    if is_elided(data, eff) {
        return;
    }
    let visit_mode = |mode: Sib, f: &mut dyn FnMut(Value)| {
        if let SibBase::Value(base) = mode.base {
            f(base);
        }
        f(mode.index);
    };
    if let Some(mode) = data
        .result
        .and_then(|res| eff.sib.get(res.index()).copied().flatten())
    {
        visit_mode(mode, &mut f);
        return;
    }
    let visit_addr = |addr: Value, f: &mut dyn FnMut(Value)| match eff
        .sib
        .get(addr.index())
        .copied()
        .flatten()
    {
        Some(mode) => visit_mode(mode, f),
        None => f(addr),
    };
    let visit_atomic = |addr: Value, f: &mut dyn FnMut(Value)| match eff
        .atomic_base
        .get(addr.index())
        .copied()
        .flatten()
    {
        Some(base) => f(base),
        None => f(addr),
    };
    match &data.op {
        Op::Load(addr, _) => visit_addr(*addr, &mut f),
        Op::Store(value, addr, _) => {
            f(*value);
            visit_addr(*addr, &mut f);
        }
        Op::AtomicLoad(addr, _) => visit_atomic(*addr, &mut f),
        Op::AtomicStore(value, addr, _) => {
            f(*value);
            visit_atomic(*addr, &mut f);
        }
        Op::AtomicRmw(_, _, addr, value) => {
            visit_atomic(*addr, &mut f);
            f(*value);
        }
        Op::AtomicCas(_, addr, expected, replacement) => {
            visit_atomic(*addr, &mut f);
            f(*expected);
            f(*replacement);
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
    pub preserved_calls: Vec<u32>,
    pub sites: Vec<(Inst, Block)>,
    pub spans: Vec<(u32, u32)>,
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
        self.ranges_of(value)
            .iter()
            .map(|r| u64::from(r.end - r.start) + 1)
            .sum::<u64>()
            .max(1)
    }
}

pub(crate) fn overlaps(a: &[Range], b: &[Range]) -> bool {
    let (Some(a_last), Some(b_last)) = (a.last(), b.last()) else {
        return false;
    };
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

/// Whether each range of `inner` lies inside one range of `outer`.
pub(crate) fn covers(outer: &[Range], inner: &[Range]) -> bool {
    inner.iter().all(|r| {
        let at = outer.partition_point(|o| o.end < r.start);
        outer
            .get(at)
            .is_some_and(|o| o.start <= r.start && r.end <= o.end)
    })
}

/// A dense row-major boolean matrix packed into 64-bit words.
pub(crate) struct BitMatrix {
    words: usize,
    bits: Vec<u64>,
}

impl BitMatrix {
    /// Allocates an all-zero matrix with `rows` rows and `cols` columns.
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        let words = cols.div_ceil(64);
        Self {
            words,
            bits: vec![0; rows * words],
        }
    }

    /// Returns the number of 64-bit words per row.
    pub(crate) const fn words(&self) -> usize {
        self.words
    }

    /// Returns whether bit `(row, col)` is set.
    pub(crate) fn get(&self, row: usize, col: usize) -> bool {
        (self.bits[row * self.words + (col >> 6)] >> (col & 63)) & 1 != 0
    }

    /// Sets bit `(row, col)`.
    pub(crate) fn set(&mut self, row: usize, col: usize) {
        self.bits[row * self.words + (col >> 6)] |= 1_u64 << (col & 63);
    }

    /// Returns the packed words of `row`.
    pub(crate) fn row(&self, row: usize) -> &[u64] {
        let start = row * self.words;
        &self.bits[start..start + self.words]
    }

    /// Returns a mutable slice of the packed words of `row`.
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [u64] {
        let start = row * self.words;
        &mut self.bits[start..start + self.words]
    }

    /// Unions `mask` into `row`.
    pub(crate) fn or_row(&mut self, row: usize, mask: &[u64]) {
        for (dst, &src) in self.row_mut(row).iter_mut().zip(mask) {
            *dst |= src;
        }
    }

    /// Writes the union of `func`'s successors of `block` into `out`.
    pub(crate) fn fill_succ_union(&self, func: &Function, block: Block, out: &mut [u64]) {
        out.fill(0);
        for succ in func.successors(block) {
            for (dst, &src) in out.iter_mut().zip(self.row(succ.index())) {
                *dst |= src;
            }
        }
    }

    /// Iterates over each set column index in `row`.
    #[inline]
    pub(crate) fn members(&self, row: usize) -> SetBits<'_> {
        set_bits(self.row(row))
    }
}

/// An iterator over the set bit indices of a `u64` word slice.
pub(crate) struct SetBits<'a> {
    rest: &'a [u64],
    base: usize,
    cur: u64,
}

impl Iterator for SetBits<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.cur == 0 {
            let (&word, tail) = self.rest.split_first()?;
            self.rest = tail;
            self.base += 64;
            self.cur = word;
        }
        let bit = self.cur.trailing_zeros() as usize;
        self.cur &= self.cur - 1;
        Some(self.base + bit)
    }
}

/// Returns an iterator over the set bit indices in `words`.
#[inline]
pub(crate) fn set_bits(words: &[u64]) -> SetBits<'_> {
    match words.split_first() {
        Some((&cur, rest)) => SetBits { rest, base: 0, cur },
        None => SetBits {
            rest: &[],
            base: 0,
            cur: 0,
        },
    }
}

/// Solves backward block liveness `in[b] = gen_sets[b] | (union_succ(in) & !kill_sets[b])` to a fixpoint.
pub(crate) fn solve_backward_liveness<I>(
    func: &Function,
    blocks: I,
    gen_sets: &BitMatrix,
    kill_sets: &BitMatrix,
) -> BitMatrix
where
    I: Clone + DoubleEndedIterator<Item = Block>,
{
    let words = gen_sets.words();
    let mut live = BitMatrix {
        words,
        bits: vec![0; func.blocks.len() * words],
    };
    let mut out = vec![0_u64; words];
    let mut changed = words > 0;
    while changed {
        changed = false;
        for block in blocks.clone().rev() {
            let idx = block.index();
            live.fill_succ_union(func, block, &mut out);
            let (gen_row, kill_row) = (gen_sets.row(idx), kill_sets.row(idx));
            for ((dst, &g), (&o, &k)) in live
                .row_mut(idx)
                .iter_mut()
                .zip(gen_row)
                .zip(out.iter().zip(kill_row))
            {
                let next = g | (o & !k);
                changed |= *dst != next;
                *dst = next;
            }
        }
    }
    live
}

fn crossing(func: &Function, wanted: &[bool], eff: Elided<'_>) -> (Vec<Option<usize>>, Vec<Value>) {
    let mut location = vec![usize::MAX; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        data.params
            .iter()
            .for_each(|param| location[param.index()] = block);
        for inst in &data.insts {
            let inst_data = &func.insts[inst.index()];
            if !is_elided(inst_data, eff) {
                results(inst_data).for_each(|result| location[result.index()] = block);
            }
        }
    }
    let mut dense = vec![None; func.types.len()];
    let mut values = Vec::new();
    for (block, data) in func.blocks.iter().enumerate() {
        for inst in &data.insts {
            for_effective_operands(&func.insts[inst.index()], eff, |value| {
                if wanted[value.index()]
                    && location[value.index()] != block
                    && dense[value.index()].is_none()
                {
                    dense[value.index()] = Some(values.len());
                    values.push(value);
                }
            });
        }
    }
    (dense, values)
}

fn local_sets(
    func: &Function,
    dense: &[Option<usize>],
    count: usize,
    eff: Elided<'_>,
) -> (BitMatrix, BitMatrix) {
    let blocks = func.blocks.len();
    let (mut uses, mut defs) = (BitMatrix::new(blocks, count), BitMatrix::new(blocks, count));
    for (block, data) in func.blocks.iter().enumerate() {
        for param in data.params.iter().filter_map(|param| dense[param.index()]) {
            defs.set(block, param);
        }
        for inst in &data.insts {
            let inst = &func.insts[inst.index()];
            if is_elided(inst, eff) {
                continue;
            }
            for_effective_operands(inst, eff, |value| {
                if let Some(value) = dense[value.index()]
                    && !defs.get(block, value)
                {
                    uses.set(block, value);
                }
            });
            for result in results(inst).filter_map(|result| dense[result.index()]) {
                defs.set(block, result);
            }
        }
    }
    (uses, defs)
}

fn live_in(
    func: &Function,
    order: &[Block],
    dense: &[Option<usize>],
    count: usize,
    eff: Elided<'_>,
) -> BitMatrix {
    let (uses, defs) = local_sets(func, dense, count, eff);
    solve_backward_liveness(func, order.iter().copied(), &uses, &defs)
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

struct LiveScan<'a> {
    wanted: &'a [bool],
    eff: Elided<'a>,
    per_val: Vec<Vec<Range>>,
    open: Vec<Option<(u32, u32)>>,
    touched: Vec<usize>,
    defs: Vec<u32>,
    unwinds: Vec<u32>,
    calls: Vec<u32>,
    preserved_calls: Vec<u32>,
    sites: Vec<(Inst, Block)>,
    at: u32,
}

impl<'a> LiveScan<'a> {
    fn new(count: usize, wanted: &'a [bool], eff: Elided<'a>) -> Self {
        Self {
            wanted,
            eff,
            per_val: vec![Vec::new(); count],
            open: vec![None; count],
            touched: Vec::new(),
            defs: vec![0; count],
            unwinds: Vec::new(),
            calls: Vec::new(),
            preserved_calls: Vec::new(),
            sites: Vec::new(),
            at: 0,
        }
    }

    fn use_val(&mut self, value: Value, b_start: u32) {
        let v = value.index();
        if !self.wanted[v] {
            return;
        }
        match self.open[v] {
            Some((s, e)) => self.open[v] = Some((s, e.max(self.at))),
            None => {
                self.open[v] = Some((b_start, self.at));
                self.touched.push(v);
            }
        }
    }

    fn def_val(&mut self, result: Value, primary: bool) {
        let r = result.index();
        if primary {
            self.defs[r] = self.at;
        }
        if !self.wanted[r] {
            return;
        }
        if let Some((s, e)) = self.open[r] {
            if e < self.at {
                push_range(&mut self.per_val, r, s, e);
                self.open[r] = Some((self.at, self.at));
            } else {
                self.open[r] = Some((s, e.max(self.at)));
            }
        } else {
            self.open[r] = Some((self.at, self.at));
            self.touched.push(r);
        }
    }

    fn scan_copied_head(&mut self, func: &Function, head: Block, args: &[Value], b_start: u32) {
        let eff = self.eff;
        self.at += 1;
        for &arg in args {
            self.use_val(arg, b_start);
        }
        self.at += 1;
        for &param in &func.blocks[head.index()].params {
            self.def_val(param, false);
        }
        for &h_inst in &func.blocks[head.index()].insts {
            let h_data = &func.insts[h_inst.index()];
            if is_elided(h_data, eff) {
                continue;
            }
            self.at += 1;
            for_effective_operands(h_data, eff, |v| self.use_val(v, b_start));
            self.at += 1;
            results(h_data).for_each(|r| self.def_val(r, false));
        }
    }

    fn scan_call(
        &mut self,
        func: &Function,
        block: Block,
        insts: &[Inst],
        place: usize,
        preserved_tc: &[bool],
        b_start: u32,
    ) {
        let inst = insts[place];
        let data = &func.insts[inst.index()];
        let eff = self.eff;
        self.at += 1;
        for_effective_operands(data, eff, |v| self.use_val(v, b_start));
        self.at += 1;
        results(data).for_each(|r| self.def_val(r, true));
        if matches!(data.op, Op::TryCall { .. }) {
            self.unwinds.push(self.at);
        }
        let traps = insts
            .get(place + 1)
            .is_some_and(|next| matches!(func.insts[next.index()].op, Op::Trap));
        if preserved_tc.get(inst.index()) == Some(&true) {
            self.preserved_calls.push(self.at);
        } else if !(traps && matches!(data.op, Op::Call { .. })) {
            self.calls.push(self.at);
            self.sites.push((inst, block));
        }
    }

    fn scan_insts(
        &mut self,
        func: &Function,
        block: Block,
        copied: &[bool],
        preserved_tc: &[bool],
        b_start: u32,
    ) -> Option<Block> {
        let eff = self.eff;
        let mut copied_head = None;
        let insts = &func.blocks[block.index()].insts;
        for (place, &inst) in insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            if is_elided(data, eff) {
                continue;
            }
            if copied.get(inst.index()) == Some(&true)
                && let Op::Jump(head, args) = &data.op
            {
                self.scan_copied_head(func, *head, args, b_start);
                copied_head = Some(*head);
                continue;
            }
            if matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
                self.scan_call(func, block, insts, place, preserved_tc, b_start);
                continue;
            }
            self.at += 1;
            let flags_only = carry_out_of_pair(func, insts, place)
                && func.insts[insts[place - 1].index()]
                    .result
                    .is_some_and(|sum| eff.elided.get(sum.index()) != Some(&true));
            if !flags_only {
                for_effective_operands(data, eff, |v| self.use_val(v, b_start));
            }
            self.at += 1;
            results(data).for_each(|r| self.def_val(r, true));
        }
        copied_head
    }
}

pub(crate) fn live(
    func: &Function,
    order: &[Block],
    copied: &[bool],
    wanted: &[bool],
    eff: Elided<'_>,
    preserved_tc: &[bool],
) -> Live {
    let count = func.types.len();
    let (dense, crossing) = crossing(func, wanted, eff);
    let live_in = live_in(func, order, &dense, crossing.len(), eff);
    let mut scan = LiveScan::new(count, wanted, eff);
    let mut spans = vec![(0, 0); func.blocks.len()];
    for &block in order {
        let index = block.index();
        let b_start = scan.at;
        for member in live_in.members(index) {
            let v = crossing[member].index();
            scan.open[v] = Some((b_start, b_start));
            scan.touched.push(v);
        }
        for &param in &func.blocks[index].params {
            let p = param.index();
            scan.defs[p] = b_start;
            if wanted[p] && scan.open[p].is_none() {
                scan.open[p] = Some((b_start, b_start));
                scan.touched.push(p);
            }
        }
        let copied_head = scan.scan_insts(func, block, copied, preserved_tc, b_start);
        scan.at += 1;
        let b_end = scan.at;
        spans[index] = (b_start, b_end);
        let exit_block = copied_head.unwrap_or(Block(index as u32));
        for next in func.successors(exit_block) {
            for member in live_in.members(next.index()) {
                let v = crossing[member].index();
                match scan.open[v] {
                    Some((s, _)) => scan.open[v] = Some((s, b_end)),
                    None => {
                        scan.open[v] = Some((b_start, b_end));
                        scan.touched.push(v);
                    }
                }
            }
        }
        for v in scan.touched.drain(..) {
            if let Some((s, e)) = scan.open[v].take() {
                push_range(&mut scan.per_val, v, s, e);
            }
        }
        scan.at += 1;
    }
    let mut ranges = Vec::new();
    let mut slice = Vec::with_capacity(count);
    for list in scan.per_val {
        let off = ranges.len() as u32;
        let len = list.len() as u32;
        ranges.extend(list);
        slice.push((off, len));
    }
    Live {
        ranges,
        slice,
        defs: scan.defs,
        unwinds: scan.unwinds,
        calls: scan.calls,
        preserved_calls: scan.preserved_calls,
        sites: scan.sites,
        spans,
    }
}

pub(crate) fn results(data: &InstData) -> impl Iterator<Item = Value> + '_ {
    let calls: &[Value] = match &data.op {
        Op::Call { results, .. } => results,
        _ => &[],
    };
    data.result.into_iter().chain(calls.iter().copied())
}
