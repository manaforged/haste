use crate::cfg::Cfg;
use crate::ir::{Function, Op};

use super::{MemBase, MemLoc, Provenance};

/// What a coroutine resume function knows about its own frame: how many leading bytes no outside code can address, which values may point into the frame, and which hold its exact start.
#[derive(Default)]
pub(super) struct FrameFacts {
    private: u32,
    derived: Vec<bool>,
    exact: Vec<bool>,
}

impl FrameFacts {
    pub(super) fn new(func: &Function, cfg: &Cfg) -> Self {
        if func.frame_private == 0 {
            return Self::default();
        }
        let count = func.types.len();
        let marked: Vec<bool> = (0..count).map(|v| func.is_frame_ptr(crate::ir::Value(v as u32))).collect();
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); count];
        for &block in &cfg.rpo {
            let Some(&last) = func.blocks[block.index()].insts.last() else { continue };
            func.insts[last.index()].op.for_edges(|target, args| {
                for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                    incoming[param.index()].push(arg.index());
                }
            });
        }
        let mut exact = marked.clone();
        for (param, args) in incoming.iter().enumerate() {
            if !args.is_empty() {
                exact[param] = true;
            }
        }
        let mut derived = marked.clone();
        for _ in 0..64 {
            let mut changed = false;
            for (param, args) in incoming.iter().enumerate() {
                if args.is_empty() || marked[param] {
                    continue;
                }
                if exact[param] && !args.iter().all(|&arg| exact[arg]) {
                    exact[param] = false;
                    changed = true;
                }
                if !derived[param] && args.iter().any(|&arg| derived[arg]) {
                    derived[param] = true;
                    changed = true;
                }
            }
            for inst in cfg.rpo.iter().flat_map(|block| &func.blocks[block.index()].insts) {
                let data = &func.insts[inst.index()];
                if matches!(data.op, Op::Load(..) | Op::StackLoad(..) | Op::Call { .. } | Op::TryCall { .. }) || super::is_atomic_barrier(&data.op) {
                    continue;
                }
                let mut from_frame = false;
                data.op.for_operands(|value| from_frame |= derived[value.index()]);
                if from_frame {
                    for result in crate::live::results(data) {
                        changed |= !std::mem::replace(&mut derived[result.index()], true);
                    }
                }
            }
            if !changed {
                return Self { private: func.frame_private, derived, exact };
            }
        }
        Self { private: func.frame_private, derived: vec![true; count], exact: vec![false; count] }
    }
}

impl Provenance {
    fn exact_frame(&self, loc: MemLoc) -> bool {
        matches!(loc.base, MemBase::Addr(v) if self.frame.exact.get(v.index()).copied().unwrap_or(false))
    }

    /// Whether `loc` lies in the private bytes of this coroutine's own frame, which only addresses derived from the frame start can reach.
    pub(crate) fn private_frame(&self, loc: MemLoc) -> bool {
        self.frame.private > 0 && loc.size != u32::MAX && self.exact_frame(loc) && loc.offset >= 0 && loc.offset + i64::from(loc.size) <= i64::from(self.frame.private)
    }

    /// Whether `op` takes an operand that may point into this coroutine's own frame.
    pub(crate) fn touches_frame(&self, op: &Op) -> bool {
        let mut touches = false;
        op.for_operands(|value| touches |= self.frame.derived.get(value.index()).copied().unwrap_or(false));
        touches
    }

    /// Whether `a` and `b` may overlap when one of them lies in the private frame bytes; `None` when neither does.
    pub(super) fn frame_alias(&self, a: MemLoc, b: MemLoc) -> Option<bool> {
        let (private, other) = match (self.private_frame(a), self.private_frame(b)) {
            (false, false) => return None,
            (true, _) => (a, b),
            (false, true) => (b, a),
        };
        Some(match other.base {
            MemBase::Addr(_) if self.exact_frame(other) => private.overlaps(other),
            MemBase::Addr(v) => self.frame.derived.get(v.index()).copied().unwrap_or(true),
            MemBase::Slot(slot) => !self.slot_unescaped(slot),
            MemBase::Global(_) => false,
        })
    }
}
