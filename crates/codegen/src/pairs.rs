//! The split of 128-bit integers into pairs of 64-bit halves before register allocation: each 128-bit value gets its halves beside it, and the operations, compares, stores and jumps that read it take the halves, so loops that carry or combine 128-bit values keep them in registers.

use rustc_hash::FxHashMap as HashMap;

use crate::ir::{Binary, Block, Convert, Function, Inst, IntCC, Op, Type, Value};

type Pair = (Value, Value);

struct Splitter<'f> {
    func: &'f mut Function,
    defs: Vec<Option<Inst>>,
    halves: HashMap<Value, Pair>,
    alias: HashMap<Value, Value>,
    offsets: HashMap<Value, i128>,
    chained: Vec<bool>,
    split_at: Vec<Vec<bool>>,
    insts: Vec<Inst>,
}

/// Splits the 128-bit values of `func` into 64-bit halves; reports whether it changed the body.
pub(crate) fn split(func: &mut Function) -> bool {
    if !func.blocks.iter().flat_map(|b| &b.insts).any(|i| {
        func.insts[i.index()]
            .result
            .is_some_and(|r| func.ty(r) == Type::I128)
    }) {
        return false;
    }
    let cfg = func.cfg();
    if func
        .blocks
        .iter()
        .enumerate()
        .any(|(b, data)| !cfg.reachable[b] && !data.insts.is_empty())
    {
        return false;
    }
    let rpo = cfg.rpo.clone();
    let chained = chained(func);
    let mut splitter = Splitter {
        defs: crate::fold::defs(func),
        func,
        halves: HashMap::default(),
        alias: HashMap::default(),
        offsets: HashMap::default(),
        chained,
        split_at: Vec::new(),
        insts: Vec::new(),
    };
    splitter.split_params();
    for block in rpo {
        splitter.block(block);
    }
    true
}

/// Which 128-bit values feed only one 128-bit add or subtract in their own block, so a constant they carry can wait for it.
fn chained(func: &Function) -> Vec<bool> {
    let mut ok = vec![false; func.types.len()];
    let has_add = func.blocks.iter().flat_map(|b| &b.insts).any(|&inst| {
        let data = &func.insts[inst.index()];
        matches!(data.op, Op::Binary(Binary::Iadd | Binary::Isub, ..))
            && data.result.is_some_and(|r| func.ty(r) == Type::I128)
    });
    if !has_add {
        return ok;
    }
    let uses = func.use_counts();
    for block in &func.blocks {
        let local: Vec<Value> = block
            .insts
            .iter()
            .filter_map(|&inst| func.insts[inst.index()].result)
            .collect();
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if matches!(data.op, Op::Binary(Binary::Iadd | Binary::Isub, ..))
                && data.result.is_some_and(|r| func.ty(r) == Type::I128)
            {
                data.op
                    .for_operands(|v| ok[v.index()] = uses[v.index()] == 1 && local.contains(&v));
            }
        }
    }
    ok
}

/// Blocks whose parameters a split may widen: every jump into them names its arguments.
fn widenable(func: &Function) -> Vec<bool> {
    let mut ok = vec![true; func.blocks.len()];
    if let Some(entry) = func.entry {
        ok[entry.index()] = false;
    }
    for block in 0..func.blocks.len() {
        if let Some(op @ (Op::BrTable(..) | Op::TryCall { .. })) = func.last(Block(block as u32)) {
            op.targets().for_each(|target| ok[target.index()] = false);
        }
    }
    ok
}

impl Splitter<'_> {
    fn emit(&mut self, op: Op, ty: Type) -> Value {
        let result = self.func.value(ty);
        let inst = self.func.add(op, Some(result));
        self.insts.push(inst);
        self.defs.resize(self.func.types.len(), None);
        self.defs[result.index()] = Some(inst);
        result
    }

    fn effect(&mut self, op: Op) {
        let inst = self.func.add(op, None);
        self.insts.push(inst);
    }

    fn konst(&mut self, imm: i64) -> Value {
        self.emit(Op::Const(imm), Type::I64)
    }

    fn bin(&mut self, op: Binary, x: Value, y: Value) -> Value {
        match (op, self.word(x), self.word(y)) {
            (Binary::Iadd | Binary::Bor | Binary::Bxor, Some(0), _)
            | (Binary::Imul, _, Some(0)) => y,
            (Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Bxor, _, Some(0))
            | (Binary::Imul, Some(0), _) => x,
            _ => self.emit(Op::Binary(op, x, y), Type::I64),
        }
    }

    /// The value of 64-bit `v` when it is a constant.
    fn word(&self, v: Value) -> Option<i64> {
        match self.func.insts[self.defs.get(v.index()).copied().flatten()?.index()].op {
            Op::Const(imm) if self.func.ty(v) == Type::I64 => Some(imm),
            _ => None,
        }
    }

    fn bin_imm(&mut self, op: Binary, x: Value, imm: i64) -> Value {
        let k = self.konst(imm);
        self.bin(op, x, k)
    }

    fn widen(&mut self, flag: Value) -> Value {
        self.emit(Op::Convert(Convert::Uextend, flag), Type::I64)
    }

    /// The low 64 bits of `v` when it is a constant.
    fn const_low(&self, v: Value) -> Option<i64> {
        match self.func.insts[self.defs.get(v.index()).copied().flatten()?.index()].op {
            Op::Const(imm) => Some(imm),
            Op::Binary(Binary::Iconcat, lo, _)
            | Op::Convert(Convert::Uextend | Convert::Sextend, lo) => self.const_low(lo),
            _ => None,
        }
    }

    fn pair(&mut self, v: Value) -> Pair {
        match self.halves.get(&v) {
            Some(&pair) => pair,
            None => self.extract(v),
        }
    }

    fn extract(&mut self, v: Value) -> Pair {
        let lo = self.emit(Op::Convert(Convert::Ireduce, v), Type::I64);
        let k = self.konst(64);
        let shifted = self.emit(Op::Binary(Binary::Ushr, v, k), Type::I128);
        let hi = self.emit(Op::Convert(Convert::Ireduce, shifted), Type::I64);
        (lo, hi)
    }

    fn split_params(&mut self) {
        let ok = widenable(self.func);
        self.split_at = vec![Vec::new(); self.func.blocks.len()];
        for block in 0..self.func.blocks.len() {
            let params = self.func.blocks[block].params.clone();
            if !params.iter().any(|&p| self.func.ty(p) == Type::I128) {
                continue;
            }
            self.insts.clear();
            let mut kept = Vec::with_capacity(params.len() + 1);
            for &param in &params {
                if self.func.ty(param) != Type::I128 {
                    kept.push(param);
                } else if ok[block] {
                    let (lo, hi) = (self.func.value(Type::I64), self.func.value(Type::I64));
                    kept.extend([lo, hi]);
                    let joined = self
                        .func
                        .add(Op::Binary(Binary::Iconcat, lo, hi), Some(param));
                    self.insts.push(joined);
                    self.defs[param.index()] = Some(joined);
                    self.halves.insert(param, (lo, hi));
                } else {
                    kept.push(param);
                    let pair = self.extract(param);
                    self.halves.insert(param, pair);
                }
            }
            self.split_at[block] = params
                .iter()
                .map(|&p| ok[block] && self.func.ty(p) == Type::I128)
                .collect();
            self.func.blocks[block].params = kept;
            let body = std::mem::take(&mut self.func.blocks[block].insts);
            self.func.blocks[block].insts = std::mem::take(&mut self.insts)
                .into_iter()
                .chain(body)
                .collect();
        }
    }

    fn block(&mut self, block: Block) {
        let old = std::mem::take(&mut self.func.blocks[block.index()].insts);
        self.insts = Vec::with_capacity(old.len());
        for inst in old {
            let mut data = self.func.insts[inst.index()].clone();
            data.op
                .map_operands(|v| self.alias.get(&v).copied().unwrap_or(v));
            data.op = self.read(data.op, data.result);
            self.func.insts[inst.index()] = data.clone();
            self.insts.push(inst);
            if let Some(result) = data.result.filter(|&r| self.func.ty(r) == Type::I128) {
                let pair = self.expand(&data.op, result);
                self.halves.insert(result, pair);
            }
        }
        self.func.blocks[block.index()].insts = std::mem::take(&mut self.insts);
    }

    /// Rewrites an instruction that reads 128-bit operands to read their halves, emitting what it needs before it.
    fn read(&mut self, op: Op, result: Option<Value>) -> Op {
        match op {
            Op::Icmp(cc, a, b) if self.func.ty(a) == Type::I128 => self.compare(cc, a, b),
            Op::Convert(Convert::Ireduce, v) if self.func.ty(v) == Type::I128 => {
                let (lo, _) = self.pair(v);
                match result {
                    Some(r) if self.func.ty(r) == Type::I64 => {
                        self.alias.insert(r, lo);
                        Op::Convert(Convert::Ireduce, v)
                    }
                    _ => Op::Convert(Convert::Ireduce, lo),
                }
            }
            Op::Store(v, addr, at) if self.func.ty(v) == Type::I128 => {
                let (lo, hi) = self.pair(v);
                self.effect(Op::Store(lo, addr, at));
                Op::Store(hi, addr, at + 8)
            }
            Op::StackStore(v, slot, at) if self.func.ty(v) == Type::I128 => {
                let (lo, hi) = self.pair(v);
                self.effect(Op::StackStore(lo, slot, at));
                Op::StackStore(hi, slot, at + 8)
            }
            Op::Jump(target, args) => Op::Jump(target, self.args(target, args)),
            Op::Brif(c, then, then_args, other, other_args) => {
                let then_args = self.args(then, then_args);
                Op::Brif(c, then, then_args, other, self.args(other, other_args))
            }
            op => op,
        }
    }

    fn args(&mut self, target: Block, args: Vec<Value>) -> Vec<Value> {
        let split = self.split_at[target.index()].clone();
        if !split.contains(&true) {
            return args;
        }
        let mut out = Vec::with_capacity(args.len() + 1);
        for (index, arg) in args.into_iter().enumerate() {
            if split.get(index).copied().unwrap_or(false) {
                let (lo, hi) = self.pair(arg);
                out.extend([lo, hi]);
            } else {
                out.push(arg);
            }
        }
        out
    }

    fn compare(&mut self, cc: IntCC, a: Value, b: Value) -> Op {
        let ((al, ah), (bl, bh)) = (self.pair(a), self.pair(b));
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            let low = self.bin(Binary::Bxor, al, bl);
            let high = self.bin(Binary::Bxor, ah, bh);
            let any = self.bin(Binary::Bor, low, high);
            let zero = self.konst(0);
            return Op::Icmp(cc, any, zero);
        }
        let (low_cc, high_cc) = halves_cc(cc);
        let same = self.emit(Op::Icmp(IntCC::Equal, ah, bh), Type::I8);
        let low = self.emit(Op::Icmp(low_cc, al, bl), Type::I8);
        let high = self.emit(Op::Icmp(high_cc, ah, bh), Type::I8);
        Op::Select(same, low, high)
    }

    /// The halves of 128-bit `result` of `op`, computed from the halves of its operands where `op` has a 64-bit form.
    fn expand(&mut self, op: &Op, result: Value) -> Pair {
        match *op {
            Op::Const(imm) => (self.konst(imm), self.konst(imm >> 63)),
            Op::Binary(Binary::Iconcat, lo, hi) => (lo, hi),
            Op::Convert(Convert::Uextend, x) => (self.to_word(x, false), self.konst(0)),
            Op::Convert(Convert::Sextend, x) => {
                let lo = self.to_word(x, true);
                (lo, self.bin_imm(Binary::Sshr, lo, 63))
            }
            Op::Binary(op @ (Binary::Band | Binary::Bor | Binary::Bxor), a, b) => {
                let ((al, ah), (bl, bh)) = (self.pair(a), self.pair(b));
                (self.bin(op, al, bl), self.bin(op, ah, bh))
            }
            Op::Binary(op @ (Binary::Iadd | Binary::Isub), a, b) => {
                let ((a, ka), (b, kb)) = (self.terms(a), self.terms(b));
                let (pair, k) = if op == Binary::Iadd {
                    (self.add(a, b), ka.wrapping_add(kb))
                } else {
                    (self.sub(a, b), ka.wrapping_sub(kb))
                };
                if k != 0 && self.chained.get(result.index()).copied().unwrap_or(false) {
                    self.offsets.insert(result, k);
                    return pair;
                }
                self.settle(pair, k)
            }
            Op::Binary(Binary::Imul, a, b) => self.mul(a, b),
            Op::Binary(op @ (Binary::Ushr | Binary::Sshr | Binary::Ishl), a, k)
                if self.const_low(k).is_some() =>
            {
                let shift = (self.const_low(k).unwrap_or(0) & 127) as u32;
                let pair = self.pair(a);
                self.shift(op, pair, shift)
            }
            Op::Binary(op @ (Binary::Ushr | Binary::Sshr | Binary::Ishl), a, k) => {
                let amount = if self.func.ty(k) == Type::I128 {
                    self.pair(k).0
                } else {
                    self.to_word(k, false)
                };
                let pair = self.pair(a);
                self.shift_by(op, pair, amount)
            }
            Op::Select(c, a, b) => {
                let ((al, ah), (bl, bh)) = (self.pair(a), self.pair(b));
                (
                    self.emit(Op::Select(c, al, bl), Type::I64),
                    self.emit(Op::Select(c, ah, bh), Type::I64),
                )
            }
            Op::Load(addr, at) => (
                self.emit(Op::Load(addr, at), Type::I64),
                self.emit(Op::Load(addr, at + 8), Type::I64),
            ),
            Op::StackLoad(slot, at) => (
                self.emit(Op::StackLoad(slot, at), Type::I64),
                self.emit(Op::StackLoad(slot, at + 8), Type::I64),
            ),
            _ => self.extract(result),
        }
    }

    fn to_word(&mut self, x: Value, signed: bool) -> Value {
        if self.func.ty(x) == Type::I64 {
            return x;
        }
        self.emit(
            Op::Convert(
                if signed {
                    Convert::Sextend
                } else {
                    Convert::Uextend
                },
                x,
            ),
            Type::I64,
        )
    }

    /// The halves of `v` with its constant halves taken out, and the 128-bit constant that still has to be added to them.
    fn terms(&mut self, v: Value) -> (Pair, i128) {
        let mut k = self.offsets.remove(&v).unwrap_or(0);
        let (mut lo, mut hi) = self
            .halves
            .get(&v)
            .copied()
            .unwrap_or_else(|| self.extract(v));
        if let Some(c) = self.word(lo).filter(|&c| c != 0) {
            k = k.wrapping_add(i128::from(c as u64));
            lo = self.konst(0);
        }
        if let Some(c) = self.word(hi).filter(|&c| c != 0) {
            k = k.wrapping_add(i128::from(c) << 64);
            hi = self.konst(0);
        }
        ((lo, hi), k)
    }

    /// The halves of `pair` plus the 128-bit constant `k`.
    fn settle(&mut self, pair: Pair, k: i128) -> Pair {
        if k == 0 {
            return pair;
        }
        let c = (self.konst(k as i64), self.konst((k >> 64) as i64));
        self.add(pair, c)
    }

    fn add(&mut self, (al, ah): Pair, (bl, bh): Pair) -> Pair {
        let lo = self.bin(Binary::Iadd, al, bl);
        let high = self.bin(Binary::Iadd, ah, bh);
        if self.word(al) == Some(0) || self.word(bl) == Some(0) {
            return (lo, high);
        }
        let base = if self.word(bl).is_some() { bl } else { al };
        let carry = self.emit(Op::Icmp(IntCC::UnsignedLessThan, lo, base), Type::I8);
        let carry = self.widen(carry);
        (lo, self.bin(Binary::Iadd, high, carry))
    }

    fn sub(&mut self, (al, ah): Pair, (bl, bh): Pair) -> Pair {
        let lo = self.bin(Binary::Isub, al, bl);
        let high = self.bin(Binary::Isub, ah, bh);
        if self.word(bl) == Some(0) {
            return (lo, high);
        }
        let borrow = self.emit(Op::Icmp(IntCC::UnsignedLessThan, al, bl), Type::I8);
        let borrow = self.widen(borrow);
        (lo, self.bin(Binary::Isub, high, borrow))
    }

    fn mul(&mut self, a: Value, b: Value) -> Pair {
        let ((al, ah), (bl, bh)) = (self.pair(a), self.pair(b));
        let lo = self.bin(Binary::Imul, al, bl);
        let carry = self.bin(Binary::Umulhi, al, bl);
        let cross_a = self.bin(Binary::Imul, al, bh);
        let cross_b = self.bin(Binary::Imul, ah, bl);
        let cross = self.bin(Binary::Iadd, cross_a, cross_b);
        (lo, self.bin(Binary::Iadd, carry, cross))
    }

    /// The halves of `(lo, hi)` shifted by the low seven bits of the variable `amount`, with no 64-bit shift by 64 or more.
    fn shift_by(&mut self, op: Binary, (lo, hi): Pair, amount: Value) -> Pair {
        let small = self.bin_imm(Binary::Band, amount, 63);
        let wide = self.bin_imm(Binary::Band, amount, 64);
        let zero = self.konst(0);
        let big = self.emit(Op::Icmp(IntCC::NotEqual, wide, zero), Type::I8);
        let back = self.konst(63);
        let back = self.bin(Binary::Isub, back, small);
        if op == Binary::Ishl {
            let low = self.bin(Binary::Ishl, lo, small);
            let carried = self.bin_imm(Binary::Ushr, lo, 1);
            let carried = self.bin(Binary::Ushr, carried, back);
            let high = self.bin(Binary::Ishl, hi, small);
            let high = self.bin(Binary::Bor, high, carried);
            return (
                self.emit(Op::Select(big, zero, low), Type::I64),
                self.emit(Op::Select(big, low, high), Type::I64),
            );
        }
        let high = self.bin(op, hi, small);
        let carried = self.bin_imm(Binary::Ishl, hi, 1);
        let carried = self.bin(Binary::Ishl, carried, back);
        let low = self.bin(Binary::Ushr, lo, small);
        let low = self.bin(Binary::Bor, low, carried);
        let fill = if op == Binary::Sshr {
            self.bin_imm(Binary::Sshr, hi, 63)
        } else {
            zero
        };
        (
            self.emit(Op::Select(big, high, low), Type::I64),
            self.emit(Op::Select(big, fill, high), Type::I64),
        )
    }

    fn shift(&mut self, op: Binary, (lo, hi): Pair, shift: u32) -> Pair {
        let fill = |this: &mut Self| {
            if op == Binary::Sshr {
                this.bin_imm(Binary::Sshr, hi, 63)
            } else {
                this.konst(0)
            }
        };
        match (op, shift) {
            (_, 0) => (lo, hi),
            (Binary::Ishl, 64..) => {
                let high = if shift == 64 {
                    lo
                } else {
                    self.bin_imm(Binary::Ishl, lo, i64::from(shift - 64))
                };
                (self.konst(0), high)
            }
            (Binary::Ishl, _) => {
                let high = self.bin_imm(Binary::Ishl, hi, i64::from(shift));
                let carried = self.bin_imm(Binary::Ushr, lo, i64::from(64 - shift));
                (
                    self.bin_imm(Binary::Ishl, lo, i64::from(shift)),
                    self.bin(Binary::Bor, high, carried),
                )
            }
            (_, 64..) => {
                let low = if shift == 64 {
                    hi
                } else {
                    self.bin_imm(op, hi, i64::from(shift - 64))
                };
                (low, fill(self))
            }
            _ => {
                let low = self.bin_imm(Binary::Ushr, lo, i64::from(shift));
                let carried = self.bin_imm(Binary::Ishl, hi, i64::from(64 - shift));
                (
                    self.bin(Binary::Bor, low, carried),
                    self.bin_imm(op, hi, i64::from(shift)),
                )
            }
        }
    }
}

/// The compare of the low halves and the strict compare of the high halves that together decide `cc` on two 128-bit values.
fn halves_cc(cc: IntCC) -> (IntCC, IntCC) {
    match cc {
        IntCC::UnsignedLessThan | IntCC::SignedLessThan => (IntCC::UnsignedLessThan, strict(cc)),
        IntCC::UnsignedLessThanOrEqual | IntCC::SignedLessThanOrEqual => {
            (IntCC::UnsignedLessThanOrEqual, strict(cc))
        }
        IntCC::UnsignedGreaterThan | IntCC::SignedGreaterThan => {
            (IntCC::UnsignedGreaterThan, strict(cc))
        }
        _ => (IntCC::UnsignedGreaterThanOrEqual, strict(cc)),
    }
}

fn strict(cc: IntCC) -> IntCC {
    match cc {
        IntCC::UnsignedLessThan | IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedLessThan,
        IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual => IntCC::SignedLessThan,
        IntCC::UnsignedGreaterThan | IntCC::UnsignedGreaterThanOrEqual => {
            IntCC::UnsignedGreaterThan
        }
        _ => IntCC::SignedGreaterThan,
    }
}
