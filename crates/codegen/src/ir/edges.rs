use super::{Block, Function, IntCC, Op, Value};

impl IntCC {
    /// The comparison that holds exactly when this one does not.
    pub fn inverse(self) -> Self {
        match self {
            IntCC::Equal => IntCC::NotEqual,
            IntCC::NotEqual => IntCC::Equal,
            IntCC::SignedLessThan => IntCC::SignedGreaterThanOrEqual,
            IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThan,
            IntCC::SignedGreaterThan => IntCC::SignedLessThanOrEqual,
            IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThan,
            IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThanOrEqual,
            IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThan,
            IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThanOrEqual,
            IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThan,
        }
    }

    /// The comparison with its operands swapped: `a cc b` is `b (cc.swapped()) a`.
    pub fn swapped(self) -> Self {
        match self {
            IntCC::Equal | IntCC::NotEqual => self,
            IntCC::SignedLessThan => IntCC::SignedGreaterThan,
            IntCC::SignedGreaterThan => IntCC::SignedLessThan,
            IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
            IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
            IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
            IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
            IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
            IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        }
    }
}

impl Op {
    /// Every block this instruction can go to.
    pub(crate) fn targets(&self) -> impl Iterator<Item = Block> + '_ {
        let (first, second, slice) = match self {
            Op::Jump(block, _) => (Some(*block), None, &[][..]),
            Op::Brif(_, then, _, otherwise, _) => (Some(*then), Some(*otherwise), &[][..]),
            Op::BrTable(_, default, targets) => (Some(*default), None, &targets[..]),
            Op::TryCall {
                normal, handler, ..
            } => (Some(*normal), Some(*handler), &[][..]),
            _ => (None, None, &[][..]),
        };
        first.into_iter().chain(second).chain(slice.iter().copied())
    }

    /// Replaces every block this instruction can go to with `f` of it.
    pub(crate) fn map_targets(&mut self, mut f: impl FnMut(Block) -> Block) {
        match self {
            Op::Jump(block, _) => *block = f(*block),
            Op::Brif(_, then, _, otherwise, _) => {
                *then = f(*then);
                *otherwise = f(*otherwise);
            }
            Op::BrTable(_, default, targets) => {
                *default = f(*default);
                targets.iter_mut().for_each(|target| *target = f(*target));
            }
            Op::TryCall {
                normal, handler, ..
            } => {
                *normal = f(*normal);
                *handler = f(*handler);
            }
            _ => {}
        }
    }

    /// Calls `f` with each block this branch passes values to and the values it passes, once per edge.
    pub(crate) fn for_edges(&self, mut f: impl FnMut(Block, &[Value])) {
        match self {
            Op::Jump(target, args) => f(*target, args),
            Op::Brif(_, then, then_args, otherwise, otherwise_args) => {
                f(*then, then_args);
                f(*otherwise, otherwise_args);
            }
            _ => {}
        }
    }

    /// Calls `f` with a mutable reference to each branch target and its argument list, once per edge.
    pub(crate) fn for_edges_mut(&mut self, mut f: impl FnMut(&mut Block, &mut Vec<Value>)) {
        match self {
            Op::Jump(target, args) => f(target, args),
            Op::Brif(_, then, then_args, otherwise, otherwise_args) => {
                f(then, then_args);
                f(otherwise, otherwise_args);
            }
            _ => {}
        }
    }

    /// Redirects every argument-free branch edge targeting `from` to `to`; returns whether any edge changed.
    pub(crate) fn retarget_empty(&mut self, from: Block, to: Block) -> bool {
        let mut changed = false;
        self.for_edges_mut(|target, args| {
            if *target == from && args.is_empty() {
                *target = to;
                changed = true;
            }
        });
        changed
    }
}

impl Function {
    /// Calls `f` with the mutable argument list of each branch edge from `pred` to `target`.
    pub(crate) fn for_edge_args_mut(
        &mut self,
        pred: Block,
        target: Block,
        mut f: impl FnMut(&mut Vec<Value>),
    ) {
        let Some(&last) = self.blocks[pred.index()].insts.last() else {
            return;
        };
        self.insts[last.index()].op.for_edges_mut(|to, args| {
            if *to == target {
                f(args);
            }
        });
    }

    /// Appends `arg` to every branch edge from `pred` to `target`.
    pub(crate) fn push_edge_arg(&mut self, pred: Block, target: Block, arg: Value) {
        self.for_edge_args_mut(pred, target, |args| args.push(arg));
    }
}
