use crate::ir::{Binary, Block, Convert, Function, IntCC, Op, SigRef, Type, Value};

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

fn small_int(ty: Type) -> bool {
    matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64)
}

struct Fold {
    consts: Vec<Option<i64>>,
    subst: Vec<Value>,
    incoming: Vec<Option<Value>>,
    mixed: Vec<bool>,
}

impl Fold {
    fn find(&self, mut value: Value) -> Value {
        while self.subst[value.index()] != value {
            value = self.subst[value.index()];
        }
        value
    }

    fn known(&self, value: Value) -> Option<i64> {
        self.consts[self.find(value).index()]
    }

    fn both(&self, x: Value, y: Value) -> Option<(i64, i64)> {
        Some((self.known(x)?, self.known(y)?))
    }

    fn evaluate(&self, func: &Function, op: &Op) -> Option<i64> {
        let (Op::Binary(_, x, _) | Op::Icmp(_, x, _) | Op::Convert(_, x) | Op::Fits(_, _, x, _)) = *op else {
            return if let Op::Const(imm) = *op { Some(imm) } else { None };
        };
        let ty = func.ty(x);
        if !small_int(ty) {
            return None;
        }
        match *op {
            Op::Binary(op, x, y) => self.both(x, y).and_then(|(a, b)| constant(op, a, b)),
            Op::Icmp(cc, x, y) => self.both(x, y).map(|(a, b)| i64::from(holds(cc, ty, a, b))),
            Op::Convert(Convert::Uextend | Convert::Ireduce, x) => Some(extend(self.known(x)?, ty, false)),
            Op::Convert(Convert::Sextend, x) => Some(extend(self.known(x)?, ty, true)),
            Op::Fits(op, signed, x, y) => self.both(x, y).and_then(|pair| fits(op, signed, ty, pair)).map(i64::from),
            _ => None,
        }
    }

    /// Finds, in one pass in block order, the small integers that are constants and the selects whose condition is one.
    fn constants(&mut self, func: &Function) {
        for inst in func.blocks.iter().flat_map(|block| &block.insts) {
            let data = &func.insts[inst.index()];
            let Some(result) = data.result.filter(|result| self.subst[result.index()] == *result) else { continue };
            if let Op::Select(condition, x, y) = data.op
                && let Some(known) = self.known(condition)
            {
                self.subst[result.index()] = if known != 0 { x } else { y };
                continue;
            }
            if self.consts[result.index()].is_none()
                && small_int(func.ty(result))
                && let Some(imm) = self.evaluate(func, &data.op)
            {
                self.consts[result.index()] = Some(imm);
            }
        }
    }

    /// Turns branches on constants into jumps and drops traps that cannot fire.
    fn branches(&self, func: &mut Function) -> bool {
        let mut changed = false;
        for block in 0..func.blocks.len() {
            let before = func.blocks[block].insts.len();
            func.blocks[block].insts.retain(|inst| !matches!(func.insts[inst.index()].op, Op::Trapz(x) if self.known(x).is_some_and(|known| known != 0)));
            changed |= func.blocks[block].insts.len() != before;
            let Some(&last) = func.blocks[block].insts.last() else { continue };
            let op = &mut func.insts[last.index()].op;
            if let Op::Brif(condition, then, then_args, otherwise, else_args) = op
                && let Some(known) = self.known(*condition)
            {
                let (target, args) = if known != 0 { (*then, std::mem::take(then_args)) } else { (*otherwise, std::mem::take(else_args)) };
                *op = Op::Jump(target, args);
                changed = true;
            }
        }
        changed
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
        let (mut at, mut values, mut seen) = (*target, args.clone(), vec![*target]);
        while let Some((next, next_values)) = self.step(func, (uses, pinned), at, &values) {
            if seen.contains(&next) {
                return false;
            }
            seen.push(next);
            (at, values) = (next, next_values);
        }
        if at == *target || !self.agrees(func, at, &values) {
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

    /// Where an edge into `block` with `args` goes next when `block` holds only a jump, or a branch on a parameter `args` makes a constant, and reads its parameters nowhere else.
    fn step(&self, func: &Function, (uses, pinned): (&[u32], &[bool]), block: Block, args: &[Value]) -> Option<(Block, Vec<Value>)> {
        let data = &func.blocks[block.index()];
        let [only] = data.insts[..] else { return None };
        if !self.agrees(func, block, args) {
            return None;
        }
        let op = &func.insts[only.index()].op;
        let param = |value: &Value| data.params.iter().position(|param| param == value);
        let (next, next_args) = match op {
            Op::Jump(next, next_args) => (*next, next_args),
            Op::Brif(condition, then, then_args, otherwise, else_args) => match self.known(args[param(condition)?])? {
                0 => (*otherwise, else_args),
                _ => (*then, then_args),
            },
            _ => return None,
        };
        let mut reads = vec![0_u32; data.params.len()];
        op.for_operands(|value| if let Some(at) = param(&value) { reads[at] += 1 });
        if pinned[next.index()] || data.params.iter().zip(&reads).any(|(param, read)| uses[param.index()] != *read) {
            return None;
        }
        Some((next, next_args.iter().map(|value| param(value).map_or(*value, |at| args[at])).collect()))
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

fn reachable(func: &Function, entry: Block) -> Vec<bool> {
    let mut seen = vec![false; func.blocks.len()];
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        if std::mem::replace(&mut seen[block.index()], true) {
            continue;
        }
        if let Some(last) = func.blocks[block.index()].insts.last() {
            stack.extend(func.insts[last.index()].op.targets());
        }
    }
    seen
}

/// The entry and every block a call enters, whose parameters are not edge arguments.
fn pinned(func: &Function, entry: Block) -> Vec<bool> {
    let mut pinned = vec![false; func.blocks.len()];
    pinned[entry.index()] = true;
    for block in &func.blocks {
        if let Some(Op::TryCall { normal, handler, .. }) = block.insts.last().map(|inst| &func.insts[inst.index()].op) {
            pinned[normal.index()] = true;
            pinned[handler.index()] = true;
        }
    }
    pinned
}

/// Drops the parameters `fold` replaced, and their arguments on every edge.
fn drop_params(func: &mut Function, fold: &Fold) {
    let gone: Vec<Vec<bool>> = func.blocks.iter().map(|block| block.params.iter().map(|param| fold.subst[param.index()] != *param).collect()).collect();
    let keep = |args: &mut Vec<Value>, target: Block| {
        let mut index = 0;
        args.retain(|_| {
            index += 1;
            !gone[target.index()].get(index - 1).copied().unwrap_or(false)
        });
    };
    for block in 0..func.blocks.len() {
        let Some(&last) = func.blocks[block].insts.last() else { continue };
        match &mut func.insts[last.index()].op {
            Op::Jump(target, args) => keep(args, *target),
            Op::Brif(_, then, then_args, otherwise, else_args) => {
                keep(then_args, *then);
                keep(else_args, *otherwise);
            }
            _ => {}
        }
    }
    for (block, gone) in func.blocks.iter_mut().zip(&gone) {
        let mut index = 0;
        block.params.retain(|_| {
            index += 1;
            !gone[index - 1]
        });
    }
}

fn pure(op: &Op) -> bool {
    matches!(op, Op::Const(_) | Op::Unary(..) | Op::Binary(..) | Op::Convert(..) | Op::Icmp(..) | Op::Fits(..) | Op::Fcmp(..) | Op::Select(..) | Op::StackAddr(..) | Op::Load(..) | Op::StackLoad(..) | Op::FramePointer)
}

/// Drops pure instructions whose results nothing reads, in one backward pass that also frees the operands they read.
/// How many times the instructions in blocks read each value.
fn use_counts(func: &Function) -> Vec<u32> {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        func.insts[inst.index()].op.for_operands(|value| uses[value.index()] += 1);
    }
    uses
}

fn dead(func: &mut Function) {
    let mut uses = use_counts(func);
    for block in func.blocks.iter_mut().rev() {
        let mut keep = vec![true; block.insts.len()];
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

/// Folds constants and branches on them, empties blocks no path reaches, replaces block parameters every edge agrees on, and drops dead pure instructions.
pub(crate) fn run(func: &mut Function) {
    let Some(entry) = func.entry else { return };
    let count = func.types.len();
    let mut fold = Fold { consts: vec![None; count], subst: (0..count as u32).map(Value).collect(), incoming: vec![None; count], mixed: vec![false; count] };
    let pinned = pinned(func, entry);
    let mut reachable = Vec::new();
    loop {
        fold.constants(func);
        let branched = fold.branches(func) | fold.thread(func, &pinned);
        if branched || reachable.is_empty() {
            reachable = self::reachable(func, entry);
            for (block, live) in func.blocks.iter_mut().zip(&reachable) {
                if !live && !block.insts.is_empty() {
                    block.insts.drain(..).for_each(|inst| func.insts[inst.index()].op = Op::Trap);
                }
            }
        }
        let replaced = fold.params(func, &reachable, &pinned);
        if !branched && !replaced {
            break;
        }
    }
    drop_params(func, &fold);
    for block in &func.blocks {
        for inst in &block.insts {
            let data = &mut func.insts[inst.index()];
            data.op.map_operands(|value| fold.find(value));
            if let Some(imm) = data.result.and_then(|result| fold.consts[result.index()]) {
                data.op = Op::Const(imm);
            }
        }
    }
    invert(func);
    dead(func);
}

/// Rewrites `x ^ 1` of a compare `x` as the inverse compare, so a negated condition branches on the flags, and a branch on `x == 0` or `x != 0` as a branch on `x`.
fn invert(func: &mut Function) {
    let count = func.types.len();
    let (mut compares, mut consts) = (vec![None; count], vec![None; count]);
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let data = &func.insts[inst.index()];
        match (&data.op, data.result) {
            (&Op::Icmp(cc, x, y), Some(result)) => compares[result.index()] = Some((cc, x, y)),
            (&Op::Const(imm), Some(result)) => consts[result.index()] = Some(imm),
            _ => {}
        }
    }
    let types = func.types.clone();
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        let op = &mut func.insts[inst.index()].op;
        match op {
            Op::Binary(Binary::Bxor, x, y) => {
                let compare = if consts[y.index()] == Some(1) { compares[x.index()] } else if consts[x.index()] == Some(1) { compares[y.index()] } else { None };
                if let Some((cc, a, b)) = compare {
                    *op = Op::Icmp(cc.inverse(), a, b);
                }
            }
            Op::Brif(condition, then, then_args, otherwise, else_args) => {
                let Some((cc, x, y)) = compares[condition.index()] else { continue };
                if consts[y.index()] != Some(0) || !small_int(types[x.index()]) {
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
            _ => {}
        }
    }
}

/// Renumbers the instructions and values the blocks hold and drops the rest, so a body kept or copied whole carries no dead entries.
pub(crate) fn compact(func: &mut Function) {
    if func.coroutine.is_some() {
        return;
    }
    let mut map: Vec<Option<Value>> = vec![None; func.types.len()];
    let mut types = Vec::new();
    let mut take = |value: Value| -> Value {
        *map[value.index()].get_or_insert_with(|| {
            types.push(func.types[value.index()]);
            Value(types.len() as u32 - 1)
        })
    };
    let live = func.blocks.iter().map(|block| block.insts.len()).sum();
    let mut insts = Vec::with_capacity(live);
    let mut blocks = std::mem::take(&mut func.blocks);
    for block in &mut blocks {
        block.params.iter_mut().for_each(|param| *param = take(*param));
        for inst in &mut block.insts {
            let slot = &mut func.insts[inst.index()];
            let mut data = crate::ir::InstData { op: std::mem::replace(&mut slot.op, Op::Trap), result: slot.result };
            data.result = data.result.map(&mut take);
            if let Op::Call { results, .. } = &mut data.op {
                results.iter_mut().for_each(|result| *result = take(*result));
            }
            data.op.map_operands(&mut take);
            insts.push(data);
            *inst = crate::ir::Inst(insts.len() as u32 - 1);
        }
    }
    func.blocks = blocks;
    func.insts = insts;
    types.shrink_to_fit();
    func.types = types;
    compact_blocks(func);
    compact_sigs(func);
}

/// Drops the empty blocks, which nothing reaches, renumbers the rest, and trims every list to its length.
fn compact_blocks(func: &mut Function) {
    let mut map: Vec<Option<Block>> = vec![None; func.blocks.len()];
    let mut blocks = Vec::new();
    for (index, mut block) in std::mem::take(&mut func.blocks).into_iter().enumerate() {
        if block.insts.is_empty() && func.entry != Some(Block(index as u32)) {
            continue;
        }
        block.params.shrink_to_fit();
        block.insts.shrink_to_fit();
        map[index] = Some(Block(blocks.len() as u32));
        blocks.push(block);
    }
    let renumber = |block: Block| map[block.index()].expect("a branch reaches only blocks that hold instructions");
    func.insts.iter_mut().for_each(|data| data.op.map_targets(renumber));
    func.entry = func.entry.map(renumber);
    func.blocks = blocks;
}

/// Drops the signatures no call uses and renumbers the rest.
fn compact_sigs(func: &mut Function) {
    let mut map: Vec<Option<SigRef>> = vec![None; func.sigs.len()];
    let mut sigs = Vec::new();
    for data in &mut func.insts {
        if let Op::Call { sig, .. } | Op::TryCall { sig, .. } = &mut data.op {
            *sig = *map[sig.index()].get_or_insert_with(|| {
                sigs.push(func.sigs[sig.index()].clone());
                SigRef(sigs.len() as u32 - 1)
            });
        }
    }
    func.sigs = sigs;
}
