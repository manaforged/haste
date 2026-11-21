use crate::ir::{Function, Inst, Value};
use crate::live::results;

use super::{is_fits_paired, is_pure_sinkable, is_static_def};

/// Where each value is read in one block: first and last position, read count, and whether the block defines it.
struct Reads {
    first: Vec<usize>,
    last: Vec<usize>,
    count: Vec<u32>,
    local: Vec<bool>,
    touched: Vec<Value>,
}

impl Reads {
    fn new(values: usize) -> Self {
        Self { first: vec![usize::MAX; values], last: vec![0; values], count: vec![0; values], local: vec![false; values], touched: Vec::new() }
    }

    fn fill(&mut self, func: &Function, block: usize) {
        let Self { first, last, count, local, touched } = self;
        for value in touched.drain(..) {
            let at = value.index();
            (first[at], last[at], count[at], local[at]) = (usize::MAX, 0, 0, false);
        }
        for &param in &func.blocks[block].params {
            local[param.index()] = true;
            touched.push(param);
        }
        for (pos, &inst) in func.blocks[block].insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|value| {
                let at = value.index();
                if count[at] == 0 {
                    first[at] = pos;
                    touched.push(value);
                }
                last[at] = pos;
                count[at] += 1;
            });
            for result in results(data) {
                local[result.index()] = true;
                touched.push(result);
            }
        }
    }

    fn dies_after(&self, uses: &[u32], value: Value, at: usize) -> bool {
        let index = value.index();
        self.local[index] && uses[index] == self.count[index] && self.last[index] > at
    }
}

/// For each value, the block parameters it is passed to on some edge.
fn feeds(func: &Function) -> Vec<Vec<Value>> {
    let mut feeds = vec![Vec::new(); func.types.len()];
    crate::alloc::walk::for_edge_copies(func, |_, _, param, arg| feeds[arg.index()].push(param));
    feeds
}

/// The position after which the pure op at `at` can move so that the block parameter it updates dies before the update is born.
fn target(func: &Function, (insts, uses, defs, params): (&[Inst], &[u32], &[Option<Inst>], &[Value]), (reads, feeds): (&Reads, &[Vec<Value>]), at: usize) -> Option<usize> {
    let inst = insts[at];
    let data = &func.insts[inst.index()];
    let result = data.result?;
    if !is_pure_sinkable(func, &data.op, result) || insts.get(at + 1).is_some_and(|&next| is_fits_paired(func, inst, next)) {
        return None;
    }
    let mut operands = Vec::with_capacity(2);
    data.op.for_operands(|value| operands.push(value));
    let dying = operands.iter().copied().find(|&value| params.contains(&value) && feeds[result.index()].contains(&value) && reads.dies_after(uses, value, at))?;
    let last = reads.last[dying.index()];
    if last + 1 >= insts.len() || last >= reads.first[result.index()] {
        return None;
    }
    let stays = |value: Value| {
        value == dying || is_static_def(func, defs, value) || !reads.local[value.index()] || uses[value.index()] > reads.count[value.index()] || reads.last[value.index()] > last
    };
    operands.iter().all(|&value| stays(value)).then_some(last)
}

/// Moves each pure scalar update of a block parameter, whose result an edge passes back to that parameter, past the parameter's last read in the block so both share one register.
pub(super) fn after_last_read(func: &mut Function, uses: &[u32], defs: &[Option<Inst>]) {
    let mut reads = Reads::new(func.types.len());
    let feeds = feeds(func);
    for block in 0..func.blocks.len() {
        if func.blocks[block].insts.len() < 3 {
            continue;
        }
        reads.fill(func, block);
        for at in (0..func.blocks[block].insts.len() - 1).rev() {
            let Some(last) = target(func, (&func.blocks[block].insts, uses, defs, &func.blocks[block].params), (&reads, &feeds), at) else { continue };
            let moved = func.blocks[block].insts.remove(at);
            func.blocks[block].insts.insert(last, moved);
            reads.fill(func, block);
        }
    }
}
