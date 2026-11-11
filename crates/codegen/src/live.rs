use crate::ir::{Block, Function, InstData, Op, Value};

/// The blocks in the order the emitter lays them out: the entry, then every other block that has instructions.
pub(crate) fn order(func: &Function) -> Vec<Block> {
    let Some(entry) = func.entry else { return Vec::new() };
    let rest = (0..func.blocks.len() as u32).map(Block).filter(|block| *block != entry && !func.blocks[block.index()].insts.is_empty());
    std::iter::once(entry).chain(rest).collect()
}

/// The positions, in layout order, over which each wanted value is live, the positions of calls that can unwind into a handler, and the positions of every call.
pub(crate) struct Live {
    pub span: Vec<Option<(u32, u32)>>,
    pub unwinds: Vec<u32>,
    pub calls: Vec<u32>,
}

/// One bit per value that crosses blocks, for each block.
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

/// The wanted values that some block other than their defining block reads, as a dense index per value and the values in index order.
fn crossing(func: &Function, wanted: &[bool]) -> (Vec<Option<usize>>, Vec<Value>) {
    let mut location = vec![usize::MAX; func.types.len()];
    for (block, data) in func.blocks.iter().enumerate() {
        data.params.iter().for_each(|param| location[param.index()] = block);
        for inst in &data.insts {
            results(&func.insts[inst.index()]).for_each(|result| location[result.index()] = block);
        }
    }
    let mut dense = vec![None; func.types.len()];
    let mut values = Vec::new();
    for (block, data) in func.blocks.iter().enumerate() {
        for inst in &data.insts {
            func.insts[inst.index()].op.for_operands(|value| {
                if wanted[value.index()] && location[value.index()] != block && dense[value.index()].is_none() {
                    dense[value.index()] = Some(values.len());
                    values.push(value);
                }
            });
        }
    }
    (dense, values)
}

/// The crossing values each block reads before it defines them, and the crossing values it defines.
fn local_sets(func: &Function, dense: &[Option<usize>], count: usize) -> (Sets, Sets) {
    let blocks = func.blocks.len();
    let (mut uses, mut defs) = (Sets::new(blocks, count), Sets::new(blocks, count));
    for (block, data) in func.blocks.iter().enumerate() {
        for param in data.params.iter().filter_map(|param| dense[param.index()]) {
            defs.insert(block, param);
        }
        for inst in &data.insts {
            let inst = &func.insts[inst.index()];
            inst.op.for_operands(|value| {
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

/// The crossing values live into each block, by backward dataflow over the control flow graph.
fn live_in(func: &Function, order: &[Block], dense: &[Option<usize>], count: usize) -> Sets {
    let (uses, defs) = local_sets(func, dense, count);
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

fn widen(span: &mut Option<(u32, u32)>, at: u32) {
    *span = Some(span.map_or((at, at), |(start, end)| (start.min(at), end.max(at))));
}

/// The span over the layout order of each `wanted` value: its definition, its uses, and the bounds of the blocks it is live into or out of.
pub(crate) fn live(func: &Function, order: &[Block], wanted: &[bool]) -> Live {
    let (dense, crossing) = crossing(func, wanted);
    let live_in = live_in(func, order, &dense, crossing.len());
    let mut span = vec![None; func.types.len()];
    let (mut unwinds, mut calls) = (Vec::new(), Vec::new());
    let mut at = 0_u32;
    for &block in order {
        let index = block.index();
        live_in.members(index).for_each(|member| widen(&mut span[crossing[member].index()], at));
        func.blocks[index].params.iter().filter(|param| wanted[param.index()]).for_each(|param| widen(&mut span[param.index()], at));
        for inst in &func.blocks[index].insts {
            at += 1;
            let data = &func.insts[inst.index()];
            let mut mark = |value: Value| {
                if wanted[value.index()] {
                    widen(&mut span[value.index()], at);
                }
            };
            data.op.for_operands(&mut mark);
            results(data).for_each(mark);
            if matches!(data.op, Op::TryCall { .. }) {
                unwinds.push(at);
            }
            if matches!(data.op, Op::Call { .. } | Op::TryCall { .. }) {
                calls.push(at);
            }
        }
        for next in func.successors(Block(index as u32)) {
            live_in.members(next.index()).for_each(|member| widen(&mut span[crossing[member].index()], at));
        }
        at += 1;
    }
    Live { span, unwinds, calls }
}

/// The values an instruction defines: its result, or a call's results.
pub(crate) fn results(data: &InstData) -> impl Iterator<Item = Value> + '_ {
    let calls: &[Value] = match &data.op {
        Op::Call { results, .. } => results,
        _ => &[],
    };
    data.result.into_iter().chain(calls.iter().copied())
}
