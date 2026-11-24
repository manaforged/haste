use rustc_hash::FxHashMap as HashMap;

use crate::ir::{Function, Op, Value};
use crate::live::Live;

use super::Location;

/// The deepest chain of same-block definitions followed from a branch condition.
const READ_DEPTH: usize = 8;

/// What a branch needs to know to run the moves of one edge before it: the registers live into each block and the locations each branch condition reads.
pub(crate) struct Hoist<R> {
    live_in: Vec<Vec<Location<R>>>,
    reads: HashMap<Value, Vec<Location<R>>>,
}

impl<R: Copy + PartialEq> Hoist<R> {
    /// An empty hoist table that declines all pre-branch edge-move hoisting.
    pub(super) fn empty() -> Self {
        Self {
            live_in: Vec::new(),
            reads: HashMap::default(),
        }
    }

    /// Collects, for each block, the register of every value live on entry (every register of a value with no live range counts everywhere), and for each branch condition the locations of the values its computation reads within its block.
    pub(super) fn new(func: &Function, live: &Live, locations: &[Location<R>]) -> Self {
        let mut starts: Vec<(u32, usize)> = live
            .spans
            .iter()
            .enumerate()
            .map(|(block, span)| (span.0, block))
            .collect();
        starts.sort_unstable();
        let mut live_in = vec![Vec::new(); func.blocks.len()];
        let mut everywhere = Vec::new();
        for (value, &loc) in locations.iter().enumerate() {
            let loc = match loc {
                Location::Reg(_) | Location::Float(_) => loc,
                Location::Offset(r, _) => Location::Reg(r),
                _ => continue,
            };
            let ranges = if value < live.slice.len() {
                live.ranges_of(value)
            } else {
                &[]
            };
            if ranges.is_empty() {
                everywhere.push(loc);
            }
            for range in ranges {
                let first = starts.partition_point(|&(start, _)| start < range.start);
                for &(_, block) in starts[first..]
                    .iter()
                    .take_while(|&&(start, _)| start <= range.end)
                {
                    live_in[block].push(loc);
                }
            }
        }
        live_in
            .iter_mut()
            .for_each(|regs| regs.extend_from_slice(&everywhere));
        let mut reads: HashMap<Value, Vec<Location<R>>> = HashMap::default();
        for block in &func.blocks {
            let Some(&last) = block.insts.last() else {
                continue;
            };
            let Op::Brif(cond, ..) = func.insts[last.index()].op else {
                continue;
            };
            let local: HashMap<Value, &Op> = block
                .insts
                .iter()
                .filter_map(|inst| {
                    func.insts[inst.index()]
                        .result
                        .map(|res| (res, &func.insts[inst.index()].op))
                })
                .collect();
            let mut stack = vec![(cond, 0)];
            let out = reads.entry(cond).or_default();
            while let Some((value, depth)) = stack.pop() {
                out.push(locations[value.index()]);
                if depth < READ_DEPTH
                    && let Some(op) = local.get(&value)
                {
                    op.for_operands(|operand| stack.push((operand, depth + 1)));
                }
            }
        }
        Self { live_in, reads }
    }

    fn clash(loc: Location<R>, other: Location<R>) -> bool {
        match (loc, other) {
            (
                Location::Reg(a) | Location::Offset(a, _),
                Location::Reg(b) | Location::Offset(b, _),
            ) => a == b,
            (Location::Float(a), Location::Float(b)) => a == b,
            _ => false,
        }
    }

    /// Whether writing the register `loc` leaves every location `condition` reads intact.
    pub(crate) fn reads_free(&self, loc: Location<R>, condition: Value) -> bool {
        self.reads
            .get(&condition)
            .is_some_and(|reads| !reads.iter().any(|&other| Self::clash(loc, other)))
    }

    /// Whether writing the register `loc` leaves every value live into `block` and every location `condition` reads intact.
    pub(crate) fn free(&self, loc: Location<R>, block: crate::ir::Block, condition: Value) -> bool {
        self.reads.get(&condition).is_some_and(|reads| {
            !self.live_in[block.index()]
                .iter()
                .chain(reads)
                .any(|&other| Self::clash(loc, other))
        })
    }
}
