use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Unary, Value};

/// `imm` of type `ty` extended to 64 bits, by its sign when `signed`.
pub(crate) fn extend(imm: i64, ty: Type, signed: bool) -> i64 {
    let bits = ty.bits();
    if bits >= 64 {
        return imm;
    }
    let shift = 64 - bits;
    match signed {
        true => imm << shift >> shift,
        false => ((imm as u64) << shift >> shift) as i64,
    }
}

/// Whether `cc` holds between the constants `x` and `y` of type `ty`.
pub(crate) fn holds(cc: IntCC, ty: Type, x: i64, y: i64) -> bool {
    let (x, y) = (extend(x, ty, cc.is_signed()), extend(y, ty, cc.is_signed()));
    match cc {
        IntCC::Equal => x == y,
        IntCC::NotEqual => x != y,
        IntCC::SignedLessThan => x < y,
        IntCC::SignedGreaterThanOrEqual => x >= y,
        IntCC::SignedGreaterThan => x > y,
        IntCC::SignedLessThanOrEqual => x <= y,
        IntCC::UnsignedLessThan => (x as u64) < y as u64,
        IntCC::UnsignedGreaterThanOrEqual => x as u64 >= y as u64,
        IntCC::UnsignedGreaterThan => x as u64 > y as u64,
        IntCC::UnsignedLessThanOrEqual => x as u64 <= y as u64,
    }
}

/// `a op b` on two constants, for the operations that wrap the same at every width.
pub(crate) fn constant(op: Binary, a: i64, b: i64) -> Option<i64> {
    match op {
        Binary::Iadd => Some(a.wrapping_add(b)),
        Binary::Isub => Some(a.wrapping_sub(b)),
        Binary::Imul => Some(a.wrapping_mul(b)),
        Binary::Band => Some(a & b),
        Binary::Bor => Some(a | b),
        Binary::Bxor => Some(a ^ b),
        _ => None,
    }
}

/// `a op b` on constants of type `ty`, with shift counts masked to the width as the encoders mask them, or `None` for a division that traps.
pub(crate) fn binary(op: Binary, ty: Type, a: i64, b: i64) -> Option<i64> {
    if let Some(value) = constant(op, a, b) {
        return Some(value);
    }
    let bits = ty.bits();
    let count = (b & i64::from(bits - 1)) as u32;
    let (sa, sb) = (extend(a, ty, true), extend(b, ty, true));
    let (ua, ub) = (extend(a, ty, false) as u64, extend(b, ty, false) as u64);
    match op {
        Binary::Ishl => Some(a << count),
        Binary::Ushr => Some((ua >> count) as i64),
        Binary::Sshr => Some(sa >> count),
        Binary::Rotl => Some(rotate(ty, ua, count)),
        Binary::Rotr => Some(rotate(ty, ua, bits - count)),
        Binary::Smin => Some(sa.min(sb)),
        Binary::Smax => Some(sa.max(sb)),
        Binary::Umulhi => Some((u128::from(ua) * u128::from(ub) >> bits) as i64),
        Binary::Smulhi => Some((i128::from(sa) * i128::from(sb) >> bits) as i64),
        Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => divide(op, ty, a, b),
        _ => None,
    }
}

/// The low bits of `value` of type `ty` rotated left by `left` modulo the width.
fn rotate(ty: Type, value: u64, left: u32) -> i64 {
    match ty {
        Type::I8 => i64::from((value as u8).rotate_left(left)),
        Type::I16 => i64::from((value as u16).rotate_left(left)),
        Type::I32 => i64::from((value as u32).rotate_left(left)),
        _ => value.rotate_left(left) as i64,
    }
}

/// `a / b` or `a % b` on constants of type `ty`, or `None` when `b` is zero or the operation is the signed minimum by -1, where the machine traps.
fn divide(op: Binary, ty: Type, a: i64, b: i64) -> Option<i64> {
    let signed = matches!(op, Binary::Sdiv | Binary::Srem);
    let (x, y) = (extend(a, ty, signed), extend(b, ty, signed));
    if y == 0 || (signed && y == -1 && x == -1_i64 << (ty.bits() - 1)) {
        return None;
    }
    Some(match op {
        Binary::Sdiv => x / y,
        Binary::Srem => x % y,
        Binary::Udiv => (x as u64 / y as u64) as i64,
        _ => (x as u64 % y as u64) as i64,
    })
}

/// `op` of the constant `a` of type `ty`, counting bits at that width.
fn unary(op: Unary, ty: Type, a: i64) -> Option<i64> {
    let bits = ty.bits();
    let unused = 64 - bits;
    let value = extend(a, ty, false) as u64;
    match op {
        Unary::Ineg => Some(a.wrapping_neg()),
        Unary::Bnot => Some(!a),
        Unary::Bswap => Some((value.swap_bytes() >> unused) as i64),
        Unary::Bitrev => Some((value.reverse_bits() >> unused) as i64),
        Unary::Clz => Some(i64::from(value.leading_zeros() - unused)),
        Unary::Ctz => Some(i64::from(value.trailing_zeros().min(bits))),
        Unary::Popcnt => Some(i64::from(value.count_ones())),
        _ => None,
    }
}

/// The constant that leaves the left operand of `op` unchanged when it stands on the right.
fn unit(op: Binary) -> Option<i64> {
    match op {
        Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor => Some(0),
        Binary::Imul => Some(1),
        Binary::Band => Some(-1),
        _ => None,
    }
}

/// Whether `x op y` on constants of type `ty` fits that type, read as signed when `signed`.
pub(crate) fn fits(op: Binary, signed: bool, ty: Type, (x, y): (i64, i64)) -> Option<bool> {
    let widen = |value: i64| if signed { i128::from(extend(value, ty, true)) } else { i128::from(extend(value, ty, false) as u64) };
    let (x, y) = (widen(x), widen(y));
    let result = match op {
        Binary::Iadd => x + y,
        Binary::Isub => x - y,
        Binary::Imul => x * y,
        _ => return None,
    };
    let bits = ty.bits();
    let (low, high) = if signed { (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1) } else { (0, (1_i128 << bits) - 1) };
    Some((low..=high).contains(&result))
}

/// `op` on small integers when `known` gives the constants it reads.
pub(crate) fn evaluate(func: &Function, op: &Op, known: &dyn Fn(Value) -> Option<i64>) -> Option<i64> {
    let (Op::Unary(_, x) | Op::Binary(_, x, _) | Op::Icmp(_, x, _) | Op::Convert(_, x) | Op::Fits(_, _, x, _)) = *op else {
        return if let Op::Const(imm) = *op { Some(imm) } else { None };
    };
    let ty = func.ty(x);
    if !ty.is_small_int() {
        return None;
    }
    let both = |x: Value, y: Value| Some((known(x)?, known(y)?));
    match *op {
        Op::Unary(op, x) => unary(op, ty, known(x)?),
        Op::Binary(op, x, y) => both(x, y).and_then(|(a, b)| binary(op, ty, a, b)),
        Op::Icmp(cc, x, y) => both(x, y).map(|(a, b)| i64::from(holds(cc, ty, a, b))),
        Op::Convert(Convert::Uextend | Convert::Ireduce, x) => Some(extend(known(x)?, ty, false)),
        Op::Convert(Convert::Sextend, x) => Some(extend(known(x)?, ty, true)),
        Op::Fits(op, signed, x, y) => both(x, y).and_then(|pair| fits(op, signed, ty, pair)).map(i64::from),
        _ => None,
    }
}

struct Fold {
    consts: Vec<Option<i64>>,
    subst: Vec<Value>,
    incoming: Vec<Option<Value>>,
    mixed: Vec<bool>,
    defs: Vec<Option<Inst>>,
}

impl Fold {
    fn find(&self, value: Value) -> Value {
        crate::merge::find(&self.subst, value)
    }

    fn known(&self, value: Value) -> Option<i64> {
        self.consts[self.find(value).index()]
    }

    fn both(&self, x: Value, y: Value) -> Option<(i64, i64)> {
        Some((self.known(x)?, self.known(y)?))
    }

    fn evaluate(&self, func: &Function, op: &Op) -> Option<i64> {
        evaluate(func, op, &|value| self.known(value)).or_else(|| match *op {
            Op::Binary(op, x, y) if func.ty(x).is_small_int() => self.absorbed(op, func.ty(x), x, y),
            _ => None,
        })
    }

    /// The constant `x op y` of type `ty` gives whatever one operand holds: a product, high product or mask with zero, or the difference or xor of a value with itself.
    fn absorbed(&self, op: Binary, ty: Type, x: Value, y: Value) -> Option<i64> {
        let zero = |value: Value| self.known(value).is_some_and(|imm| extend(imm, ty, false) == 0);
        match op {
            Binary::Imul | Binary::Umulhi | Binary::Smulhi | Binary::Band if zero(x) || zero(y) => Some(0),
            Binary::Isub | Binary::Bxor if self.find(x) == self.find(y) => Some(0),
            _ => None,
        }
    }

    /// The operation of the instruction that defines `value`, when one in a block does.
    fn def<'a>(&self, func: &'a Function, value: Value) -> Option<&'a Op> {
        self.defs[self.find(value).index()].map(|inst| &func.insts[inst.index()].op)
    }

    /// Whether `value` is 0 or 1 in its low byte because a comparison or overflow check defines it.
    fn boolean(&self, func: &Function, value: Value) -> bool {
        match self.def(func, value) {
            Some(Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..)) => true,
            Some(Op::Convert(Convert::Uextend, inner)) => matches!(self.def(func, *inner), Some(Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..))),
            _ => false,
        }
    }

    /// Whether `x` and `y` of type `ty` hold the same bits at that width.
    fn same(&self, ty: Type, x: Value, y: Value) -> bool {
        self.find(x) == self.find(y) || matches!(self.both(x, y), Some((a, b)) if extend(a, ty, false) == extend(b, ty, false))
    }

    /// The existing value the result of `op` always equals, when there is one.
    fn identity(&self, func: &Function, op: &Op) -> Option<Value> {
        match *op {
            Op::Select(condition, x, y) => self.known(condition).map(|known| if known != 0 { x } else { y }),
            Op::Binary(kind, x, y) if func.ty(x).is_small_int() => self.neutral(func, kind, func.ty(x), x, y).or_else(|| self.cancel(func, kind, x, y)),
            Op::Unary(kind @ (Unary::Ineg | Unary::Bnot), x) => match *self.def(func, x)? {
                Op::Unary(inner, value) if inner == kind => Some(value),
                _ => None,
            },
            Op::Icmp(cc, x, y) if self.boolean(func, x) => {
                let imm = extend(self.known(y)?, func.ty(x), false);
                matches!((cc, imm), (IntCC::NotEqual, 0) | (IntCC::Equal, 1)).then_some(x)
            }
            _ => None,
        }
    }

    /// `x` when `y` is the unit of `op` at type `ty` or a shift count that masks to zero, or `y` when `x` is the unit of a commuting `op`.
    fn neutral(&self, func: &Function, op: Binary, ty: Type, x: Value, y: Value) -> Option<Value> {
        let is = |value: Value, want: i64| self.known(value).is_some_and(|imm| extend(imm, ty, false) == extend(want, ty, false));
        if let Some(unit) = unit(op) {
            if is(y, unit) {
                return Some(x);
            }
            return (op != Binary::Isub && is(x, unit)).then_some(y);
        }
        let shift = matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr);
        if shift && self.known(y).is_some_and(|count| count & i64::from(ty.bits() - 1) == 0) {
            return Some(x);
        }
        if op == Binary::Band {
            if self.boolean(func, x) && self.known(y).is_some_and(|imm| imm & 1 == 1) {
                return Some(x);
            }
            if self.boolean(func, y) && self.known(x).is_some_and(|imm| imm & 1 == 1) {
                return Some(y);
            }
        }
        None
    }

    /// `a` when `x ^ y` is `(a ^ c) ^ c`, or `b` when `(a + b) - a` in either order of `Iadd`.
    fn cancel(&self, func: &Function, op: Binary, x: Value, y: Value) -> Option<Value> {
        let ty = func.ty(x);
        if op == Binary::Isub {
            return match *self.def(func, x)? {
                Op::Binary(Binary::Iadd, a, b) if self.same(ty, a, y) => Some(b),
                Op::Binary(Binary::Iadd, a, b) if self.same(ty, b, y) => Some(a),
                _ => None,
            };
        }
        if op != Binary::Bxor {
            return None;
        }
        [(x, y), (y, x)].into_iter().find_map(|(inner, other)| match *self.def(func, inner)? {
            Op::Binary(Binary::Bxor, a, b) if self.same(ty, b, other) => Some(a),
            Op::Binary(Binary::Bxor, a, b) if self.same(ty, a, other) => Some(b),
            _ => None,
        })
    }

    /// Finds, in one pass in block order, the small integers that are constants, wrapped to their type, and the results that equal an existing value.
    fn constants(&mut self, func: &Function) {
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result.filter(|result| self.subst[result.index()] == *result && self.consts[result.index()].is_none()) else { continue };
            let ty = func.ty(result);
            if ty.is_small_int()
                && let Some(imm) = self.evaluate(func, &data.op)
            {
                self.consts[result.index()] = Some(extend(imm, ty, true));
            } else if let Some(value) = self.identity(func, &data.op) {
                self.subst[result.index()] = value;
            }
        }
    }

    /// Turns the branches `decide` settles into jumps and drops traps that cannot fire.
    fn branches(&self, func: &mut Function) -> bool {
        let mut changed = false;
        for block in 0..func.blocks.len() {
            let before = func.blocks[block].insts.len();
            func.blocks[block].insts.retain(|inst| !matches!(func.insts[inst.index()].op, Op::Trapz(x) if self.known(x).is_some_and(|known| known != 0)));
            changed |= func.blocks[block].insts.len() != before;
            let Some(&last) = func.blocks[block].insts.last() else { continue };
            changed |= self.decide(&func.types, &mut func.insts[last.index()].op);
        }
        changed
    }

    /// Turns `op` into a jump when it branches on a constant or sends both edges to the same block with the same arguments.
    fn decide(&self, types: &[Type], op: &mut Op) -> bool {
        match op {
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let same = then == otherwise && then_args.len() == else_args.len() && then_args.iter().zip(else_args.iter()).all(|(a, b)| self.find(*a) == self.find(*b));
                let taken = match self.known(*condition) {
                    Some(known) => known != 0,
                    None if same => true,
                    None => return false,
                };
                let (target, args) = if taken { (*then, std::mem::take(then_args)) } else { (*otherwise, std::mem::take(else_args)) };
                *op = Op::Jump(target, args);
                true
            }
            Op::BrTable(index, default, targets) => {
                let target = if let Some(known) = self.known(*index) {
                    let i = extend(known, types[index.index()], false) as u64;
                    usize::try_from(i).ok().and_then(|at| targets.get(at).copied()).unwrap_or(*default)
                } else if targets.iter().all(|target| target == default) {
                    *default
                } else {
                    return false;
                };
                *op = Op::Jump(target, Vec::new());
                true
            }
            _ => false,
        }
    }

    /// Sends each edge that passes a constant to a block holding only a branch on that parameter straight to the target the branch takes.
    fn thread(&self, func: &mut Function, pinned: &[bool]) -> bool {
        let mut uses = use_counts(func);
        let mut changed = false;
        for block in 0..func.blocks.len() {
            let Some(&last) = func.blocks[block].insts.last() else { continue };
            let mut op = std::mem::replace(&mut func.insts[last.index()].op, Op::Trap);
            match &mut op {
                Op::Jump(target, args) => changed |= self.retarget(func, (&mut uses, pinned), target, args),
                Op::Brif(_, then, then_args, otherwise, else_args) => {
                    changed |= self.retarget(func, (&mut uses, pinned), then, then_args);
                    changed |= self.retarget(func, (&mut uses, pinned), otherwise, else_args);
                }
                _ => {}
            }
            func.insts[last.index()].op = op;
        }
        changed
    }

    /// Moves the edge to `target` with `args` along the blocks that only jump on, or only branch on a parameter the edge makes a constant, to the first block that does more, keeping `uses` counting the edge's new arguments.
    fn retarget(&self, func: &Function, (uses, pinned): (&mut [u32], &[bool]), target: &mut Block, args: &mut Vec<Value>) -> bool {
        let Some((first, first_values)) = self.step(func, (uses, pinned), *target, args) else { return false };
        if first == *target {
            return false;
        }
        let (mut at, mut values, mut seen) = (first, first_values, vec![*target, first]);
        while let Some((next, next_values)) = self.step(func, (uses, pinned), at, &values) {
            if seen.contains(&next) {
                return false;
            }
            seen.push(next);
            (at, values) = (next, next_values);
        }
        if !self.agrees(func, at, &values) {
            return false;
        }
        args.iter().for_each(|value| uses[value.index()] -= 1);
        values.iter().for_each(|value| uses[value.index()] += 1);
        (*target, *args) = (at, values);
        true
    }

    /// Whether the parameters of `block` that an earlier round replaced stand for the values `args` gives them.
    fn agrees(&self, func: &Function, block: Block, args: &[Value]) -> bool {
        func.blocks[block.index()].params.iter().zip(args).all(|(param, arg)| self.subst[param.index()] == *param || self.find(*arg) == self.find(*param))
    }

    /// Where an edge into `block` with `args` goes next when `block` holds only a jump, or a branch on a parameter `args` makes a constant or on a comparison of such a parameter with a constant, and reads its parameters nowhere else.
    fn step(&self, func: &Function, (uses, pinned): (&[u32], &[bool]), block: Block, args: &[Value]) -> Option<(Block, Vec<Value>)> {
        let data = &func.blocks[block.index()];
        let (compare, last) = match data.insts[..] {
            [only] => (None, only),
            [compare, last] => (Some(compare), last),
            _ => return None,
        };
        if !self.agrees(func, block, args) {
            return None;
        }
        let op = &func.insts[last.index()].op;
        let param = |value: &Value| data.params.iter().position(|param| param == value);
        let (next, next_args) = match op {
            Op::Jump(next, next_args) if compare.is_none() => (*next, next_args),
            Op::Brif(condition, then, then_args, otherwise, else_args) => match self.edge_condition(func, (compare, uses), *condition, &|value| param(&value).map(|at| args[at]))? {
                0 => (*otherwise, else_args),
                _ => (*then, then_args),
            },
            _ => return None,
        };
        let reads = |param: Value| {
            let mut count = 0_u32;
            compare.iter().map(|inst| &func.insts[inst.index()].op).chain([op]).for_each(|op| op.for_operands(|value| count += u32::from(value == param)));
            count
        };
        if pinned[next.index()] || data.params.iter().any(|param| uses[param.index()] != reads(*param)) {
            return None;
        }
        Some((next, next_args.iter().map(|value| param(value).map_or(*value, |at| args[at])).collect()))
    }

    /// The constant a branch on `condition` sees when the block's parameters hold what `arg` gives them: `condition` itself is such a parameter, or `compare` alone uses it and compares parameters or constants.
    fn edge_condition(&self, func: &Function, (compare, uses): (Option<Inst>, &[u32]), condition: Value, arg: &dyn Fn(Value) -> Option<Value>) -> Option<i64> {
        let Some(compare) = compare else { return self.known(arg(condition)?) };
        let data = &func.insts[compare.index()];
        let (Op::Icmp(cc, x, y), Some(result)) = (&data.op, data.result) else { return None };
        if result != condition || uses[result.index()] != 1 || !func.ty(*x).is_small_int() {
            return None;
        }
        let value = |operand: Value| arg(operand).map_or_else(|| self.known(operand), |given| self.known(given));
        Some(i64::from(holds(*cc, func.ty(*x), value(*x)?, value(*y)?)))
    }

    /// Replaces each block parameter that every reachable edge gives one value, other than itself.
    fn params(&mut self, func: &Function, reachable: &[bool], pinned: &[bool]) -> bool {
        for param in func.blocks.iter().flat_map(|block| &block.params) {
            self.incoming[param.index()] = None;
            self.mixed[param.index()] = false;
        }
        for (block, _) in reachable.iter().enumerate().filter(|(_, live)| **live) {
            for_edges(func, block, |target, args| {
                for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                    let arg = self.find(*arg);
                    match self.incoming[param.index()] {
                        _ if arg == *param => {}
                        None => self.incoming[param.index()] = Some(arg),
                        Some(seen) if seen != arg => self.mixed[param.index()] = true,
                        Some(_) => {}
                    }
                }
            });
        }
        let mut changed = false;
        for (_, data) in func.blocks.iter().enumerate().filter(|(block, _)| reachable[*block] && !pinned[*block]) {
            for param in &data.params {
                if let (Some(value), false) = (self.incoming[param.index()], self.mixed[param.index()])
                    && self.subst[param.index()] == *param
                {
                    self.subst[param.index()] = value;
                    changed = true;
                }
            }
        }
        changed
    }
}

/// Calls `f` with each block the terminator of `block` jumps to with arguments, and those arguments.
fn for_edges<'a>(func: &'a Function, block: usize, mut f: impl FnMut(Block, &'a [Value])) {
    let Some(last) = func.blocks[block].insts.last() else { return };
    match &func.insts[last.index()].op {
        Op::Jump(target, args) => f(*target, args),
        Op::Brif(_, then, then_args, otherwise, else_args) => {
            f(*then, then_args);
            f(*otherwise, else_args);
        }
        _ => {}
    }
}

/// The entry and every block a call enters, whose parameters are not edge arguments.
fn pinned(func: &Function, entry: Block) -> Vec<bool> {
    let mut pinned = vec![false; func.blocks.len()];
    pinned[entry.index()] = true;
    for block in 0..func.blocks.len() as u32 {
        match func.last(Block(block)) {
            Some(Op::TryCall { normal, handler, .. }) => {
                pinned[normal.index()] = true;
                pinned[handler.index()] = true;
            }
            Some(Op::BrTable(_, default, targets)) => {
                pinned[default.index()] = true;
                targets.iter().for_each(|target| pinned[target.index()] = true);
            }
            _ => {}
        }
    }
    pinned
}

/// Drops the parameters `fold` replaced, and their arguments on every edge.
fn drop_params(func: &mut Function, fold: &Fold) {
    let gone = |param: &Value| fold.subst[param.index()] != *param;
    let blocks = &func.blocks;
    let keep = |args: &mut Vec<Value>, target: Block| {
        let params = &blocks[target.index()].params;
        let mut index = 0;
        args.retain(|_| {
            index += 1;
            !params.get(index - 1).is_some_and(gone)
        });
    };
    for block in blocks {
        let Some(&last) = block.insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, args) => keep(args, *target),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                keep(then_args, *then);
                keep(else_args, *otherwise);
            }
            _ => {}
        }
    }
    for block in &mut func.blocks {
        block.params.retain(|param| !gone(param));
    }
}

fn pure(op: &Op) -> bool {
    matches!(op, Op::Const(_) | Op::SlotAddr(_) | Op::Symbol(_) | Op::Runtime(_) | Op::Unary(..) | Op::Binary(..) | Op::Convert(..) | Op::Icmp(..) | Op::Fits(..) | Op::Fcmp(..) | Op::ByteCmp(..) | Op::Splat(_) | Op::Movemask(_) | Op::Vector(..) | Op::Sha256Rounds(..) | Op::Select(..) | Op::StackAddr(..) | Op::Load(..) | Op::StackLoad(..) | Op::FramePointer)
}

/// How many times the instructions in blocks read each value.
fn use_counts(func: &Function) -> Vec<u32> {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        func.insts[inst.index()].op.for_operands(|value| uses[value.index()] += 1);
    }
    uses
}

/// The instruction in a block that defines each value.
pub(crate) fn defs(func: &Function) -> Vec<Option<Inst>> {
    let mut defs = vec![None; func.types.len()];
    for &inst in func.blocks.iter().flat_map(|block| &block.insts) {
        if let Some(result) = func.insts[inst.index()].result {
            defs[result.index()] = Some(inst);
        }
    }
    defs
}

/// Empties the blocks no path from `entry` reaches and returns which blocks a path reaches.
pub(crate) fn prune(func: &mut Function, entry: Block) -> Vec<bool> {
    let reachable = func.reachable(entry);
    for (block, live) in func.blocks.iter_mut().zip(&reachable) {
        if !live && !block.insts.is_empty() {
            block.insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
        }
    }
    reachable
}

/// Drops pure instructions whose results nothing reads, in one backward pass that also frees the operands they read.
fn dead(func: &mut Function) {
    let mut uses = use_counts(func);
    let mut keep = Vec::new();
    for block in func.blocks.iter_mut().rev() {
        keep.clear();
        keep.resize(block.insts.len(), true);
        for (at, inst) in block.insts.iter().enumerate().rev() {
            let data = &func.insts[inst.index()];
            if pure(&data.op) && data.result.is_some_and(|result| uses[result.index()] == 0) {
                keep[at] = false;
                data.op.for_operands(|value| uses[value.index()] -= 1);
            }
        }
        let mut at = 0;
        block.insts.retain(|_| {
            at += 1;
            keep[at - 1]
        });
    }
}

/// Folds constants, identities and branches on them, settles the compares, traps and branches the facts on each path decide when `effort` is full, empties blocks no path reaches, replaces block parameters every edge agrees on, drops dead pure instructions, and threads the edges through the blocks that leaves holding only a jump.
pub(crate) fn run(func: &mut Function, effort: crate::Effort) {
    let Some(entry) = func.entry else { return };
    let count = func.types.len();
    let defs = defs(func);
    let mut fold = Fold { consts: vec![None; count], subst: (0..count as u32).map(Value).collect(), incoming: vec![None; count], mixed: vec![false; count], defs };
    let pinned = pinned(func, entry);
    let mut reachable = Vec::new();
    loop {
        fold.constants(func);
        let settled = effort == crate::Effort::Full && reachable.is_empty() && crate::facts::settle(func, &fold.defs, &mut fold.consts);
        let branched = settled | fold.branches(func) | fold.thread(func, &pinned);
        if branched || reachable.is_empty() {
            reachable = prune(func, entry);
        }
        let replaced = fold.params(func, &reachable, &pinned);
        if !branched && !replaced {
            break;
        }
    }
    drop_params(func, &fold);
    for block in &func.blocks {
        for inst in &block.insts {
            func.insts[inst.index()].op.map_operands(|value| fold.find(value));
            let replacement = match func.insts[inst.index()].op {
                Op::Binary(Binary::Fmul, x, y) if is_float_two(func, &fold.defs, y) => Some(Op::Binary(Binary::Fadd, x, x)),
                Op::Binary(Binary::Fmul, x, y) if is_float_two(func, &fold.defs, x) => Some(Op::Binary(Binary::Fadd, y, y)),
                _ => func.insts[inst.index()].result.and_then(|result| fold.consts[result.index()]).map(Op::Const),
            };
            if let Some(op) = replacement {
                func.insts[inst.index()].op = op;
            }
        }
    }
    invert(func);
    dead(func);
    if fold.thread(func, &pinned) | fold.branches(func) {
        prune(func, entry);
        dead(func);
    }
    if devirt_vtables(func, &fold.defs) {
        dead(func);
    }
}

fn devirt_vtables(func: &mut Function, defs: &[Option<Inst>]) -> bool {
    if func.vtable_slots.is_empty() {
        return false;
    }
    let mut changed = false;
    for b in 0..func.blocks.len() {
        let mut i = 0;
        while i < func.blocks[b].insts.len() {
            let inst = func.blocks[b].insts[i];
            if let (Op::Load(addr, off), Some(res)) = (func.insts[inst.index()].op.clone(), func.insts[inst.index()].result)
                && func.ty(res) == Type::I64
                && let Some(def_inst) = defs.get(addr.index()).copied().flatten()
                && let Op::Symbol(vt_sym) = func.insts[def_inst.index()].op
                && let Some(&(_, _, slot_sym, low)) = func.vtable_slots.iter().find(|&&(v, o, _, _)| v == vt_sym && o == off)
            {
                let slot_val = func.value(Type::I64);
                let slot_op = if low { Op::SlotAddr(slot_sym) } else { Op::Symbol(slot_sym) };
                let slot_inst = func.add(slot_op, Some(slot_val));
                func.blocks[b].insts.insert(i, slot_inst);
                func.insts[inst.index()].op = Op::Load(slot_val, 0);
                changed = true;
                i += 2;
                continue;
            }
            i += 1;
        }
    }
    changed
}

fn is_float_two(func: &Function, defs: &[Option<Inst>], value: Value) -> bool {
    let Some(inst) = defs[value.index()] else { return false };
    let Op::Const(imm) = func.insts[inst.index()].op else { return false };
    match func.ty(value) {
        Type::F64 => imm as u64 == 0x4000_0000_0000_0000,
        Type::F32 => imm as u32 == 0x4000_0000,
        _ => false,
    }
}

/// The value `op` negates when the constants make it `v ^ 1`, `1 ^ v`, `v == 0` or `v != 1`.
fn negated(op: &Op, consts: &[Option<i64>]) -> Option<Value> {
    let is = |value: Value, want: i64| consts[value.index()] == Some(want);
    match *op {
        Op::Binary(Binary::Bxor, x, y) | Op::Icmp(IntCC::NotEqual, x, y) if is(y, 1) => Some(x),
        Op::Binary(Binary::Bxor, y, x) if is(y, 1) => Some(x),
        Op::Icmp(IntCC::Equal, x, y) if is(y, 0) => Some(x),
        _ => None,
    }
}

/// Rewrites a negation of a compare, `x ^ 1`, `x == 0` or `x != 1`, as the inverse compare, so a negated condition branches on the flags, and a branch on `x == 0` or `x != 0` as a branch on `x`.
fn invert(func: &mut Function) {
    let count = func.types.len();
    let (mut compares, mut consts, mut xors, mut bools) = (vec![None; count], vec![None; count], vec![None; count], vec![false; count]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        match (&data.op, data.result) {
            (&Op::Icmp(cc, x, y), Some(result)) => {
                compares[result.index()] = Some((cc, x, y));
                bools[result.index()] = true;
            }
            (&Op::Fcmp(..) | &Op::Fits(..), Some(result)) => bools[result.index()] = true,
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            (&Op::Binary(Binary::Bxor, x, y), Some(result)) => xors[result.index()] = Some((x, y)),
            (&Op::Binary(Binary::Band, x, y), Some(result)) => {
                if consts[x.index()] == Some(1) || consts[y.index()] == Some(1) {
                    bools[result.index()] = true;
                }
            }
            _ => {}
        }
    }
    let types = &func.types;
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &mut func.insts[inst.index()].op;
        if let Some((cc, a, b)) = negated(op, &consts).and_then(|value| compares[value.index()]) {
            *op = Op::Icmp(cc.inverse(), a, b);
            continue;
        }
        let Op::Brif(condition, then, then_args, otherwise, else_args) = op else { continue };
        if let Some((x, y)) = xors[condition.index()] {
            let inner = if consts[y.index()] == Some(1) {
                Some(x)
            } else if consts[x.index()] == Some(1) {
                Some(y)
            } else {
                None
            };
            if let Some(inner) = inner.filter(|v| bools[v.index()]) {
                *condition = inner;
                std::mem::swap(then, otherwise);
                std::mem::swap(then_args, else_args);
                continue;
            }
        }
        let Some((cc, x, y)) = compares[condition.index()] else { continue };
        if consts[y.index()] != Some(0) || !types[x.index()].is_small_int() {
            continue;
        }
        match cc {
            IntCC::NotEqual => *condition = x,
            IntCC::Equal => {
                *condition = x;
                std::mem::swap(then, otherwise);
                std::mem::swap(then_args, else_args);
            }
            _ => {}
        }
    }
}

pub(crate) use crate::thread::compact;
