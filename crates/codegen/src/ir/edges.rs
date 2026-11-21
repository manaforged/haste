use super::{Block, Op, Value};

impl Op {
    /// Every block this instruction can go to.
    pub(crate) fn targets(&self) -> impl Iterator<Item = Block> + '_ {
        let (first, second, slice) = match self {
            Op::Jump(block, _) => (Some(*block), None, &[][..]),
            Op::Brif(_, then, _, otherwise, _) => (Some(*then), Some(*otherwise), &[][..]),
            Op::BrTable(_, default, targets) => (Some(*default), None, &targets[..]),
            Op::TryCall { normal, handler, .. } => (Some(*normal), Some(*handler), &[][..]),
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
            Op::TryCall { normal, handler, .. } => {
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
}
