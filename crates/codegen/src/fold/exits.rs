use super::{Fold, extend};
use crate::ir::{Block, Function, Inst, IntCC, Op, Value};

/// The values `block` returns when it holds only constants and a return.
fn exit(func: &Function, block: Block) -> Option<&[Value]> {
    let (&last, body) = func.blocks[block.index()].insts.split_last()?;
    let Op::Return(values) = &func.insts[last.index()].op else {
        return None;
    };
    body.iter()
        .all(|inst| matches!(func.insts[inst.index()].op, Op::Const(_)))
        .then_some(values.as_slice())
}

/// `value` with each parameter of `block` replaced by the argument `args` gives it.
fn bind(func: &Function, block: Block, args: &[Value], value: Value) -> Value {
    func.blocks[block.index()]
        .params
        .iter()
        .position(|&param| param == value)
        .map_or(value, |at| args[at])
}

impl Fold {
    /// Each value, constant and side of `condition` on which the value equals the constant: a zero condition when false, and a compare of a value with a constant.
    pub(super) fn equalities<'a>(
        &'a self,
        func: &'a Function,
        condition: Value,
    ) -> impl Iterator<Item = (Value, i64, bool)> + 'a {
        let compare = match self.def(func, condition) {
            Some(&Op::Icmp(cc, lhs, rhs)) => Some((cc, lhs, rhs)),
            _ => None,
        };
        let sides = compare
            .into_iter()
            .flat_map(|(cc, lhs, rhs)| [(cc, lhs, rhs, false), (cc, rhs, lhs, true)]);
        sides
            .filter_map(move |(cc, var, other, swapped)| {
                let imm = extend(self.known(other)?, func.ty(var), false);
                match (cc, swapped) {
                    (IntCC::Equal, _) => Some((var, imm, true)),
                    (IntCC::NotEqual, _) => Some((var, imm, false)),
                    (IntCC::UnsignedGreaterThan, false) | (IntCC::UnsignedLessThan, true)
                        if imm == 0 =>
                    {
                        Some((var, 0, false))
                    }
                    (IntCC::UnsignedGreaterThanOrEqual, false)
                    | (IntCC::UnsignedLessThanOrEqual, true)
                        if imm == 1 =>
                    {
                        Some((var, 0, false))
                    }
                    (IntCC::UnsignedLessThan, false) | (IntCC::UnsignedGreaterThan, true)
                        if imm == 1 =>
                    {
                        Some((var, 0, true))
                    }
                    (IntCC::UnsignedLessThanOrEqual, false)
                    | (IntCC::UnsignedGreaterThanOrEqual, true)
                        if imm == 0 =>
                    {
                        Some((var, 0, true))
                    }
                    _ => None,
                }
            })
            .chain([(condition, 0, false)])
    }

    /// Turns the branch `last` into a jump to one of its targets when both only return and return the same values once the edge to the other fixes a value the branch tests.
    pub(super) fn exits(&self, func: &mut Function, last: Inst) -> bool {
        let Some(jump) = self.exit_jump(func, last) else {
            return false;
        };
        func.insts[last.index()].op = jump;
        true
    }

    /// The jump that replaces the branch `last` for `exits`.
    fn exit_jump(&self, func: &Function, last: Inst) -> Option<Op> {
        let Op::Brif(condition, then, then_args, otherwise, else_args) =
            &func.insts[last.index()].op
        else {
            return None;
        };
        let [taken, fallen] = [
            (*then, then_args.as_slice()),
            (*otherwise, else_args.as_slice()),
        ]
        .map(|(block, args)| exit(func, block).map(|values| (values, block, args)));
        let (taken, fallen) = (taken?, fallen?);
        let agrees = |(var, imm, on_true): (Value, i64, bool)| {
            let (fixed, other) = if on_true {
                (taken, fallen)
            } else {
                (fallen, taken)
            };
            let ty = func.ty(var);
            let is_var = |v: Value| self.find(v) == self.find(var);
            let is_imm = |v: Value| {
                func.ty(v) == ty && self.known(v).is_some_and(|k| extend(k, ty, false) == imm)
            };
            let same = fixed.0.len() == other.0.len()
                && fixed.0.iter().zip(other.0).all(|(&a, &b)| {
                    let (a, b) = (
                        bind(func, fixed.1, fixed.2, a),
                        bind(func, other.1, other.2, b),
                    );
                    self.same(func.ty(a), a, b)
                        || (is_var(a) && is_imm(b))
                        || (is_imm(a) && is_var(b))
                });
            same.then_some(other)
        };
        let (_, target, args) = self.equalities(func, *condition).find_map(agrees)?;
        Some(Op::Jump(target, args.to_vec()))
    }
}
