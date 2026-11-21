use super::Emit;
use crate::frame::Ext;
use super::asm::{Alu, Cond, Count, Group3, Reg, Rm, Shift, Size};
use crate::ir::{Binary, Function, IntCC, Op, TargetOpId, Type, Unary, Value};
use crate::wide::half_score;

/// The registers 128-bit rotates use as scratch besides rax, rcx and rdx.
pub(super) const WIDE_SCRATCH: [Reg; 3] = [Reg::R8, Reg::R9, Reg::R10];

fn imm_score((lo, hi): (Option<i64>, Option<i64>), zero_only: bool) -> u8 {
    half_score((lo, hi), zero_only) * 2 + u8::from(lo.is_some()) + u8::from(hi.is_some())
}

fn half((lo, hi): (Option<i64>, Option<i64>), high: bool) -> Option<i64> {
    if high { hi } else { lo }
}

/// The held-carry tag that says the carry flag equals the 0-or-1 value `v`.
pub(super) fn carry_tag(v: Value) -> usize {
    2 * v.index()
}

/// The held-carry tag that says the carry flag is the unsigned overflow of the add that produced `sum`.
pub(super) fn overflow_tag(sum: Value) -> usize {
    2 * sum.index() + 1
}

fn carry_op(op: &Op) -> Option<(TargetOpId, &[Value])> {
    match op {
        Op::TargetOp(id @ (TargetOpId::X64AddCarry | TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrow | TargetOpId::X64SubBorrowOut), args) if args.as_slice().len() == 3 => Some((*id, args.as_slice())),
        _ => None,
    }
}

/// Marks each carry out whose every use is the carry operand of the next carry pair in its block, with only register loads and stores between, so the carry flag hands it over and no register needs it.
pub(super) fn flag_carries(func: &Function, plan: &crate::alloc::Plan<Reg>) -> Vec<bool> {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        func.insts[inst.index()].op.for_operands(|value| uses[value.index()] += 1);
    }
    let held = |value: Value| !plan.elided[value.index()] && matches!(plan.locations[value.index()], crate::alloc::Location::Reg(_));
    let moved = |value: Value| !plan.elided[value.index()] && matches!(plan.locations[value.index()], crate::alloc::Location::Reg(_) | crate::alloc::Location::Slot);
    let defs = crate::fold::defs(func);
    let in_memory = |value: Value| {
        plan.elided[value.index()]
            && defs[value.index()].is_some_and(|inst| match func.insts[inst.index()].op {
                Op::Load(addr, _) => held(addr),
                Op::StackLoad(..) => true,
                _ => false,
            })
    };
    let mut flag = vec![false; func.types.len()];
    for block in &func.blocks {
        for (at, &inst) in block.insts.iter().enumerate() {
            let data = &func.insts[inst.index()];
            let (Some((TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrowOut, _)), Some(out)) = (carry_op(&data.op), data.result) else { continue };
            let mut consumers = 0;
            for &next in &block.insts[at + 1..] {
                let op = &func.insts[next.index()].op;
                if let Some((_, &[carry, a, b])) = carry_op(op)
                    && carry == out
                    && moved(a)
                    && (moved(b) || in_memory(b))
                {
                    consumers += 1;
                    continue;
                }
                let safe = match *op {
                    Op::Load(addr, _) => held(addr) && func.insts[next.index()].result.is_some_and(|r| moved(r) || in_memory(r)),
                    Op::Store(value, addr, _) => held(addr) && held(value),
                    _ => false,
                };
                if consumers > 0 || !safe {
                    break;
                }
            }
            flag[out.index()] = consumers > 0 && uses[out.index()] == consumers;
        }
    }
    flag
}

/// Marks each 32- or 64-bit sum that the next instruction compares against one of its own operands, and each difference whose operands a compare in its block tests, which read the carry or borrow of the add or subtract.
pub(super) fn overflow_reads(func: &Function) -> Vec<bool> {
    let mut read = vec![false; func.types.len()];
    let mut compared = std::collections::HashSet::new();
    for block in &func.blocks {
        compared.clear();
        for inst in &block.insts {
            if let Op::Icmp(_, x, y) = func.insts[inst.index()].op {
                compared.extend([(x, y), (y, x)]);
            }
        }
        for (at, data) in block.insts.iter().map(|inst| &func.insts[inst.index()]).enumerate() {
            let (Op::Binary(op @ (Binary::Iadd | Binary::Isub), a, b), Some(result)) = (&data.op, data.result) else { continue };
            let next = block.insts.get(at + 1).map(|inst| &func.insts[inst.index()].op);
            let sum_read = matches!(next, Some(&Op::Icmp(_, x, y)) if (x == result && (y == *a || y == *b)) || (y == result && (x == *a || x == *b)));
            read[result.index()] = matches!(func.ty(result), Type::I32 | Type::I64) && if *op == Binary::Iadd { sum_read } else { compared.contains(&(*a, *b)) };
        }
    }
    read
}

/// Marks each sum whose carry compare an `adc` or `sbb` absorbed, directly or through an absorbed widen, so the sum must set the carry flag.
pub(super) fn adc_sums(func: &Function, plan: &crate::alloc::Plan<Reg>, defs: &[Option<crate::ir::Inst>]) -> Vec<bool> {
    let mut marked = vec![false; func.types.len()];
    let op_of = |v: Value| defs[v.index()].map(|inst| &func.insts[inst.index()].op);
    let sum_of = |sum: Value, operand: Value| matches!(op_of(sum), Some(&Op::Binary(Binary::Iadd, a, b)) if a == operand || b == operand);
    for data in func.blocks.iter().flat_map(|block| &block.insts).map(|inst| &func.insts[inst.index()]) {
        let Op::Binary(Binary::Iadd | Binary::Isub, x, y) = data.op else { continue };
        for operand in [x, y].into_iter().filter(|v| plan.elided[v.index()]) {
            let compare = match op_of(operand) {
                Some(&Op::Convert(crate::ir::Convert::Uextend, inner)) if plan.elided[inner.index()] => inner,
                _ => operand,
            };
            if let Some(&Op::Icmp(_, a, b)) = op_of(compare) {
                for (sum, other) in [(a, b), (b, a)] {
                    if sum_of(sum, other) {
                        marked[sum.index()] = true;
                    }
                }
            }
        }
    }
    marked
}

fn icmp_order(cc: IntCC) -> (bool, Cond) {
    match cc {
        IntCC::SignedLessThan => (false, Cond::L),
        IntCC::SignedGreaterThanOrEqual => (false, Cond::Ge),
        IntCC::SignedGreaterThan => (true, Cond::L),
        IntCC::SignedLessThanOrEqual => (true, Cond::Ge),
        IntCC::UnsignedLessThan => (false, Cond::B),
        IntCC::UnsignedGreaterThanOrEqual => (false, Cond::Ae),
        IntCC::UnsignedGreaterThan => (true, Cond::B),
        _ => (true, Cond::Ae),
    }
}

impl Emit<'_> {
    /// Adds or subtracts `a` and `b` after the carry in `args[0]` by `adc` or `sbb` into `result`: the sum, or the carry out, which reads the flags of a sum of the same operands emitted just before.
    pub(super) fn carry_chain(&mut self, id: TargetOpId, args: [Value; 3], result: Value) -> Result<(), String> {
        let add = matches!(id, TargetOpId::X64AddCarry | TargetOpId::X64AddCarryOut);
        let out = matches!(id, TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrowOut);
        if !(out && self.carry_flags == Some((add, args))) {
            let [carry, a, b] = args;
            let held = |emit: &Self| emit.asm.held_carry() == Some(carry_tag(carry));
            let kept = held(self);
            if !kept {
                self.get(Reg::Rdx, carry, Ext::Zero);
            }
            let busy = [self.reg_of(b), self.reg_of(carry)];
            let target = Some(self.dst(result)).filter(|&reg| !out && !matches!(reg, Reg::Rdx | Reg::Rcx | Reg::Rsp) && !busy.contains(&Some(reg))).unwrap_or(Reg::Rax);
            self.get(target, a, Ext::Zero);
            let source: Rm = match self.elided_mem(b) {
                Some(mem) => mem.into(),
                None => self
                    .reg_of(b)
                    .filter(|&reg| !matches!(reg, Reg::Rdx | Reg::Rsp) && reg != target)
                    .unwrap_or_else(|| {
                        self.get(Reg::Rcx, b, Ext::Zero);
                        Reg::Rcx
                    })
                    .into(),
            };
            if !(kept && held(self)) {
                if self.flag_carry[carry.index()] {
                    return Err(format!("the carry flag lost the elided carry {carry:?}"));
                }
                if kept {
                    self.get(Reg::Rdx, carry, Ext::Zero);
                }
                self.asm.alu_imm_sized(Size::B, Alu::Add, Reg::Rdx, -1);
            }
            self.asm.alu(if add { Alu::Adc } else { Alu::Sbb }, target, source);
            self.clobber_regs(&[target, Reg::Rax, Reg::Rcx, Reg::Rdx]);
            if !out {
                self.carry_flags = Some((add, args));
                self.put(result, target);
            }
        }
        self.flags = None;
        if !out {
            return Ok(());
        }
        if self.flag_carry[result.index()] {
            self.asm.hold_carry(carry_tag(result));
            return Ok(());
        }
        let dst = self.dst(result);
        self.asm.setcc(Cond::B, dst);
        self.asm.load(Size::B, dst, dst);
        self.asm.hold_carry(carry_tag(result));
        self.put(result, dst);
        Ok(())
    }

    /// Multiplies in place for an add, or, or xor of the two elided halves of one 64-bit product, and returns rax, which holds the result.
    pub(super) fn folded_product(&mut self, op: Binary, (x, y): (Value, Value), result: Value) -> Option<Reg> {
        let alu = match op {
            Binary::Bxor => Alu::Xor,
            Binary::Bor => Alu::Or,
            Binary::Iadd => Alu::Add,
            _ => return None,
        };
        if !(self.plan.elided[x.index()] && self.plan.elided[y.index()]) {
            return None;
        }
        let (signed, a, b) = super::location::product_halves(self.func, &|v| self.defs[v.index()], (x, y))?;
        self.multiply_high(signed, a, b, 64);
        self.low_product = None;
        self.asm.alu(alu, Reg::Rax, Reg::Rdx);
        self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
        self.flags = self.plan.fused[result.index()].then_some((result, Cond::Ne));
        Some(Reg::Rax)
    }

    /// Adds or subtracts the carry `left <u right` (or `left != 0`), inverted when `invert`, to `other` by `adc` or `sbb`, which also computes `other` when it is a folded add or subtract.
    pub(super) fn carry_add(&mut self, op: Binary, ty: Type, dst: Reg, (left, other, right, invert): (Value, Value, Option<Value>, bool)) -> Result<(), String> {
        let size = if ty == Type::I32 { Size::D } else { Size::Q };
        let fold = if self.plan.elided[other.index()] { super::location::adc_fold(self.func, op, invert, other, |v| self.defs[v.index()]) } else { None };
        let ready = |emit: &Self, v: Value| !emit.plan.elided[v.index()] && matches!(emit.location(v), crate::alloc::Location::Reg(_) | crate::alloc::Location::Slot | crate::alloc::Location::Imm(_));
        let operands_ready = fold.map_or_else(|| ready(self, other), |(p, q)| ready(self, p) && ready(self, q));
        let held = operands_ready && right.is_some_and(|right| self.overflow_held(left, right));
        if !held {
            self.emit_cmp_pair(Size::of(self.ty(left).bytes()), left, right, false);
        }
        let (alu, imm) = match (op == Binary::Iadd, invert) {
            (true, false) => (Alu::Adc, 0),
            (false, false) => (Alu::Sbb, 0),
            (true, true) => (Alu::Sbb, -1),
            (false, true) => (Alu::Adc, -1),
        };
        let source = match fold {
            Some((p, q)) if p != q && self.reg_of(q) == Some(dst) => {
                self.asm.mov(Reg::Rax, dst);
                self.clobber_reg(Reg::Rax);
                self.raw(dst, p);
                Some(Rm::from(Reg::Rax))
            }
            Some((p, q)) => {
                self.raw(dst, p);
                self.imm_sized(q, size).is_none().then(|| self.operand(q))
            }
            None => {
                self.raw(dst, other);
                None
            }
        };
        if held && !right.is_some_and(|right| self.overflow_held(left, right)) {
            return Err("the carry flag was overwritten before an adc that reads it".to_owned());
        }
        match (source, fold) {
            (Some(source), _) => self.asm.alu_sized(size, alu, dst, source),
            (None, Some((_, q))) => self.asm.alu_imm_sized(size, alu, dst, self.imm_sized(q, size).unwrap_or_default()),
            (None, None) => self.asm.alu_imm_sized(size, alu, dst, imm),
        }
        Ok(())
    }

    /// Whether the carry flag still holds `left <u right`, from the add that produced `left` from `right` or the subtract of `right` from `left`.
    pub(super) fn overflow_held(&self, left: Value, right: Value) -> bool {
        let Some(source) = self.asm.held_carry().filter(|tag| tag % 2 == 1).map(|tag| Value((tag / 2) as u32)) else { return false };
        self.defs[source.index()].is_some_and(|inst| match self.func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, a, b) => source == left && (a == right || b == right),
            Op::Binary(Binary::Isub, a, b) => a == left && b == right,
            _ => false,
        })
    }

    /// The condition that reads the unsigned compare `x cc y` from a carry or borrow the carry flag still holds for these operands.
    pub(super) fn held_compare(&self, cc: IntCC, x: Value, y: Value) -> Option<Cond> {
        let (left, right, cond) = match cc {
            IntCC::UnsignedLessThan => (x, y, Cond::B),
            IntCC::UnsignedGreaterThan => (y, x, Cond::B),
            IntCC::UnsignedGreaterThanOrEqual => (x, y, Cond::Ae),
            IntCC::UnsignedLessThanOrEqual => (y, x, Cond::Ae),
            _ => return None,
        };
        self.overflow_held(left, right).then_some(cond)
    }

    fn const_halves(&self, v: Value) -> (Option<i64>, Option<i64>) {
        crate::wide::const_halves(self.func, &self.plan, &self.defs, v)
    }

    fn has_zero_high(&self, v: Value) -> bool {
        crate::wide::has_zero_high(self.func, &self.plan, &self.defs, v)
    }

    fn known_halves(&self, v: Value) -> (Option<i64>, Option<i64>) {
        crate::wide::known_halves(self.func, &self.plan, &self.defs, v)
    }

    fn shift_const(&self, y: Value) -> Option<u8> {
        crate::wide::shift_const(self.func, &self.plan, &self.defs, y)
    }

    fn load_half(&mut self, high: bool, reg: Reg, value: Value) {
        let half = if high { self.known_halves(value).1 } else { self.const_halves(value).0 };
        if let Some(imm) = half {
            if imm == 0 {
                self.asm.alu32(Alu::Xor, reg, reg);
            } else {
                self.asm.mov_imm(reg, imm);
            }
            self.clobber_reg(reg);
        } else if high {
            self.high(reg, value);
        } else {
            self.raw(reg, value);
        }
    }

    fn halves(&mut self, x: Value) {
        self.load_half(false, Reg::Rax, x);
        self.load_half(true, Reg::Rdx, x);
    }

    fn put_halves(&mut self, result: Value) {
        self.put(result, Reg::Rax);
        self.put_high(result, Reg::Rdx);
    }

    fn put_count(&mut self, result: Value) {
        self.put(result, Reg::Rax);
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        self.clobber_reg(Reg::Rax);
        self.put_high(result, Reg::Rax);
    }

    fn wide_neg(&mut self, x: Value) {
        if self.const_halves(x).0 == Some(0) {
            self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
            self.clobber_reg(Reg::Rax);
            self.load_half(true, Reg::Rdx, x);
            self.asm.group3(Group3::Neg, Reg::Rdx);
            return;
        }
        self.halves(x);
        self.asm.group3(Group3::Neg, Reg::Rax);
        self.asm.alu_imm(Alu::Adc, Reg::Rdx, 0);
        self.asm.group3(Group3::Neg, Reg::Rdx);
    }

    pub(super) fn wide_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        match op {
            Unary::Ineg => self.wide_neg(x),
            Unary::Bnot => {
                self.halves(x);
                self.asm.group3(Group3::Not, Reg::Rax);
                self.asm.group3(Group3::Not, Reg::Rdx);
            }
            Unary::Bswap => {
                self.load_half(true, Reg::Rax, x);
                self.load_half(false, Reg::Rdx, x);
                self.asm.bswap(Size::Q, Reg::Rax);
                self.asm.bswap(Size::Q, Reg::Rdx);
            }
            Unary::Bitrev => {
                self.load_half(false, Reg::R11, x);
                self.load_half(true, Reg::Rax, x);
                self.reverse_bits();
                self.put(result, Reg::Rax);
                self.asm.mov(Reg::Rax, Reg::R11);
                self.clobber_reg(Reg::Rax);
                self.reverse_bits();
                self.put_high(result, Reg::Rax);
                return Ok(());
            }
            Unary::Clz | Unary::Ctz | Unary::Popcnt => {
                self.wide_count(op, x);
                self.put_count(result);
                return Ok(());
            }
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.put_halves(result);
        Ok(())
    }

    fn wide_count(&mut self, op: Unary, x: Value) {
        let zero_hi = self.has_zero_high(x);
        if zero_hi && matches!(op, Unary::Popcnt | Unary::Clz) {
            self.load_half(false, Reg::Rcx, x);
            let kind = if op == Unary::Popcnt { Count::Popcnt } else { Count::Lzcnt };
            self.count(kind, Reg::Rax, Reg::Rcx);
            if op == Unary::Clz {
                self.asm.alu_imm(Alu::Add, Reg::Rax, 64);
            }
            return;
        }
        self.load_half(false, Reg::Rcx, x);
        self.load_half(true, Reg::Rdx, x);
        let (count, first, second) = match op {
            Unary::Clz => (Count::Lzcnt, Reg::Rdx, Reg::Rcx),
            Unary::Ctz => (Count::Tzcnt, Reg::Rcx, Reg::Rdx),
            _ => (Count::Popcnt, Reg::Rcx, Reg::Rdx),
        };
        self.count(count, Reg::Rax, first);
        self.count(count, second, second);
        if op == Unary::Popcnt {
            self.asm.alu(Alu::Add, Reg::Rax, second);
            return;
        }
        self.asm.alu_imm(Alu::Add, second, 64);
        self.asm.test(first, first);
        self.asm.cmov(Cond::E, Reg::Rax, second);
    }

    fn emit_pair_alu(&mut self, (low_op, high_op): (Alu, Alu), lhs: Value, (rhs, rhs_lo, rhs_hi): (Value, Option<i64>, Option<i64>)) {
        self.halves(lhs);
        let hi_imm32 = rhs_hi.and_then(|v| i32::try_from(v).ok());
        if let Some(imm) = rhs_hi.filter(|_| hi_imm32.is_none()) {
            self.asm.mov_imm(Reg::Rsi, imm);
            self.clobber_reg(Reg::Rsi);
        }
        self.alu_rhs_half(low_op, (false, Reg::Rax), (rhs, rhs_lo));
        if let Some(imm) = hi_imm32 {
            self.asm.alu_imm(high_op, Reg::Rdx, imm);
        } else if rhs_hi.is_some() {
            self.asm.alu(high_op, Reg::Rdx, Reg::Rsi);
        } else {
            self.asm.alu(high_op, Reg::Rdx, self.at(rhs).offset(8));
        }
    }

    fn wide_add_sub(&mut self, op: Binary, x: Value, y: Value) {
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        if op == Binary::Isub && matches!(cx, (Some(0), Some(0))) {
            return self.wide_neg(y);
        }
        let ((lhs, rhs), (rhs_lo, rhs_hi)) = if op == Binary::Iadd && imm_score(cx, false) > imm_score(cy, false) {
            ((y, x), cx)
        } else {
            ((x, y), cy)
        };
        if rhs_lo == Some(0) {
            self.load_half(false, Reg::Rax, lhs);
            self.load_half(true, Reg::Rdx, lhs);
            if rhs_hi != Some(0) {
                let plain = if op == Binary::Iadd { Alu::Add } else { Alu::Sub };
                self.alu_rhs_half(plain, (true, Reg::Rdx), (rhs, rhs_hi));
            }
            return;
        }
        let ops = if op == Binary::Iadd { (Alu::Add, Alu::Adc) } else { (Alu::Sub, Alu::Sbb) };
        self.emit_pair_alu(ops, lhs, (rhs, rhs_lo, rhs_hi));
    }

    pub(super) fn wide_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        match op {
            Binary::Iadd | Binary::Isub => self.wide_add_sub(op, x, y),
            Binary::Band | Binary::Bor | Binary::Bxor => self.wide_bitwise(op, x, y),
            Binary::Imul => self.wide_multiply(x, y),
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                if let Some(k) = self.shift_const(y) {
                    if k < 64 || op == Binary::Ishl {
                        self.load_half(false, Reg::Rax, x);
                    }
                    if k < 64 || op != Binary::Ishl {
                        self.load_half(true, Reg::Rdx, x);
                    }
                    self.wide_shift_const(op, k);
                } else {
                    self.halves(x);
                    self.raw(Reg::Rcx, y);
                    self.wide_shift(op, (Reg::Rax, Reg::Rdx), Reg::Rsi);
                }
            }
            Binary::Rotl | Binary::Rotr => self.wide_rotate(op == Binary::Rotl, x, y),
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.put_halves(result);
        Ok(())
    }

    fn wide_bitwise(&mut self, op: Binary, x: Value, y: Value) {
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        for (high, reg) in [(false, Reg::Rax), (true, Reg::Rdx)] {
            let (hx, hy) = (half(cx, high), half(cy, high));
            let (lhs, rhs, rhs_c) = if imm_score((hx, None), false) > imm_score((hy, None), false) { (y, x, hx) } else { (x, y, hy) };
            self.wide_bitwise_half((op, high), (lhs, rhs, rhs_c), reg);
        }
    }

    fn alu_rhs_half(&mut self, alu: Alu, (high, reg): (bool, Reg), (rhs, rhs_c): (Value, Option<i64>)) {
        if matches!(alu, Alu::Cmp) && rhs_c == Some(0) {
            self.asm.test(reg, reg);
        } else if let Some(imm) = rhs_c.and_then(|v| i32::try_from(v).ok()) {
            self.asm.alu_imm(alu, reg, imm);
        } else if let Some(imm) = rhs_c {
            self.asm.mov_imm(Reg::Rcx, imm);
            self.clobber_reg(Reg::Rcx);
            self.asm.alu(alu, reg, Reg::Rcx);
        } else {
            let mem = if high { self.at(rhs).offset(8) } else { self.at(rhs) };
            self.asm.alu(alu, reg, mem);
        }
    }

    fn wide_bitwise_half(&mut self, (op, high): (Binary, bool), (lhs, rhs, rhs_c): (Value, Value, Option<i64>), reg: Reg) {
        match (op, rhs_c) {
            (Binary::Band, Some(0)) => {
                self.asm.alu32(Alu::Xor, reg, reg);
                self.clobber_reg(reg);
                return;
            }
            (Binary::Bor, Some(-1)) => {
                self.asm.mov_imm(reg, -1);
                self.clobber_reg(reg);
                return;
            }
            (Binary::Bor | Binary::Bxor, Some(0)) | (Binary::Band, Some(-1)) => return self.load_half(high, reg, lhs),
            (Binary::Bxor, Some(-1)) => {
                self.load_half(high, reg, lhs);
                return self.asm.group3(Group3::Not, reg);
            }
            _ => {}
        }
        self.load_half(high, reg, lhs);
        let alu = match op {
            Binary::Band => Alu::And,
            Binary::Bor => Alu::Or,
            _ => Alu::Xor,
        };
        self.alu_rhs_half(alu, (high, reg), (rhs, rhs_c));
    }

    fn wide_shift_const(&mut self, op: Binary, k: u8) {
        if k == 0 {
            return;
        }
        if k >= 64 {
            let s = k - 64;
            match op {
                Binary::Ishl => {
                    self.asm.mov(Reg::Rdx, Reg::Rax);
                    if s > 0 {
                        self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rdx, s);
                    }
                    self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
                }
                _ => {
                    self.asm.mov(Reg::Rax, Reg::Rdx);
                    if s > 0 {
                        let kind = if op == Binary::Sshr { Shift::Sar } else { Shift::Shr };
                        self.asm.shift_imm(Size::Q, kind, Reg::Rax, s);
                    }
                    if op == Binary::Sshr {
                        self.asm.shift_imm(Size::Q, Shift::Sar, Reg::Rdx, 63);
                    } else {
                        self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
                    }
                }
            }
            return;
        }
        match op {
            Binary::Ishl => {
                self.asm.shld_imm(Size::Q, Reg::Rdx, Reg::Rax, k);
                self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, k);
            }
            _ => {
                self.asm.shrd_imm(Size::Q, Reg::Rax, Reg::Rdx, k);
                let kind = if op == Binary::Sshr { Shift::Sar } else { Shift::Shr };
                self.asm.shift_imm(Size::Q, kind, Reg::Rdx, k);
            }
        }
    }

    fn wide_mul_zero_lo(&mut self, a: Value, (b, b_hi): (Value, Option<i64>)) {
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        if b_hi == Some(0) {
            self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
            self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
            return;
        }
        self.load_half(false, Reg::Rdx, a);
        if b_hi == Some(-1) {
            self.asm.group3(Group3::Neg, Reg::Rdx);
        } else if let Some(imm) = b_hi.and_then(|v| i32::try_from(v).ok()) {
            if imm != 1 {
                self.asm.imul_imm(Reg::Rdx, Reg::Rdx, imm);
            }
        } else if let Some(imm) = b_hi {
            self.asm.mov_imm(Reg::Rcx, imm);
            self.asm.imul(Reg::Rdx, Reg::Rcx);
        } else {
            self.asm.imul(Reg::Rdx, self.at(b).offset(8));
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rdx, Reg::Rcx]);
    }

    fn mul_cross_term(&mut self, (a, a_high, a_c): (Value, bool, Option<i64>), (b, b_high, b_c): (Value, bool, Option<i64>)) {
        if a_c == Some(0) || b_c == Some(0) {
            return;
        }
        if matches!(b_c, Some(1 | -1)) {
            self.load_half(a_high, Reg::Rcx, a);
            self.asm.alu(if b_c == Some(1) { Alu::Add } else { Alu::Sub }, Reg::Rdx, Reg::Rcx);
            self.clobber_reg(Reg::Rcx);
            return;
        }
        if let Some(imm) = b_c.and_then(|v| i32::try_from(v).ok()) {
            if a_c.is_none() {
                let mem = if a_high { self.at(a).offset(8) } else { self.at(a) };
                self.asm.imul_imm(Reg::Rcx, mem, imm);
            } else {
                self.load_half(a_high, Reg::Rcx, a);
                self.asm.imul_imm(Reg::Rcx, Reg::Rcx, imm);
            }
        } else {
            self.load_half(a_high, Reg::Rcx, a);
            if let Some(imm) = b_c {
                self.asm.mov_imm(Reg::Rsi, imm);
                self.clobber_reg(Reg::Rsi);
                self.asm.imul(Reg::Rcx, Reg::Rsi);
            } else {
                let mem = if b_high { self.at(b).offset(8) } else { self.at(b) };
                self.asm.imul(Reg::Rcx, mem);
            }
        }
        self.asm.alu(Alu::Add, Reg::Rdx, Reg::Rcx);
        self.clobber_reg(Reg::Rcx);
    }

    fn wide_multiply(&mut self, x: Value, y: Value) {
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        if matches!(cx, (Some(0), Some(0))) || matches!(cy, (Some(0), Some(0))) {
            self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
            self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
            self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
            return;
        }
        if cx.0 == Some(0) {
            return self.wide_mul_zero_lo(y, (x, cx.1));
        }
        if cy.0 == Some(0) {
            return self.wide_mul_zero_lo(x, (y, cy.1));
        }
        let ((a, ca), (b, cb)) = if imm_score(cx, true) > imm_score(cy, true) { ((y, cy), (x, cx)) } else { ((x, cx), (y, cy)) };
        self.load_half(false, Reg::Rax, a);
        if cb.0.is_some() {
            self.load_half(false, Reg::Rcx, b);
            self.asm.group3(Group3::Mul, Reg::Rcx);
        } else {
            self.asm.group3(Group3::Mul, self.at(b));
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
        self.mul_cross_term((a, false, ca.0), (b, true, cb.1));
        self.mul_cross_term((a, true, ca.1), (b, false, cb.0));
    }

    /// Shifts the pair (low, high) by cl modulo 128, using `spare` for the fill word.
    fn wide_shift(&mut self, op: Binary, (low, high): (Reg, Reg), spare: Reg) {
        match op {
            Binary::Ishl => {
                self.asm.alu32(Alu::Xor, spare, spare);
                self.asm.shld(high, low);
                self.asm.shift(Size::Q, Shift::Shl, low);
                self.asm.test_imm8(Reg::Rcx, 64);
                self.asm.cmov(Cond::Ne, high, low);
                self.asm.cmov(Cond::Ne, low, spare);
            }
            Binary::Ushr => {
                self.asm.alu32(Alu::Xor, spare, spare);
                self.asm.shrd(low, high);
                self.asm.shift(Size::Q, Shift::Shr, high);
                self.asm.test_imm8(Reg::Rcx, 64);
                self.asm.cmov(Cond::Ne, low, high);
                self.asm.cmov(Cond::Ne, high, spare);
            }
            _ => {
                self.asm.mov(spare, high);
                self.asm.shift_imm(Size::Q, Shift::Sar, spare, 63);
                self.asm.shrd(low, high);
                self.asm.shift(Size::Q, Shift::Sar, high);
                self.asm.test_imm8(Reg::Rcx, 64);
                self.asm.cmov(Cond::Ne, low, high);
                self.asm.cmov(Cond::Ne, high, spare);
            }
        }
    }

    fn wide_rotate(&mut self, left: bool, x: Value, y: Value) {
        self.halves(x);
        if let Some(k) = self.shift_const(y) {
            let r = if left { (128 - k) & 127 } else { k };
            if r == 0 {
                return;
            }
            self.asm.mov(Reg::Rcx, Reg::Rax);
            if r < 64 {
                self.asm.shrd_imm(Size::Q, Reg::Rax, Reg::Rdx, r);
                self.asm.shrd_imm(Size::Q, Reg::Rdx, Reg::Rcx, r);
                return;
            }
            self.asm.mov(Reg::Rax, Reg::Rdx);
            if r > 64 {
                self.asm.shrd_imm(Size::Q, Reg::Rax, Reg::Rcx, r - 64);
                self.asm.shrd_imm(Size::Q, Reg::Rcx, Reg::Rdx, r - 64);
            }
            self.asm.mov(Reg::Rdx, Reg::Rcx);
            return;
        }
        let (first, second) = if left { (Binary::Ishl, Binary::Ushr) } else { (Binary::Ushr, Binary::Ishl) };
        let [low, high, spare] = WIDE_SCRATCH;
        self.asm.mov(low, Reg::Rax);
        self.asm.mov(high, Reg::Rdx);
        self.raw(Reg::Rcx, y);
        self.asm.alu_imm32(Alu::And, Reg::Rcx, 127);
        self.wide_shift(first, (Reg::Rax, Reg::Rdx), spare);
        self.asm.group3(Group3::Neg, Reg::Rcx);
        self.asm.alu_imm32(Alu::And, Reg::Rcx, 127);
        self.wide_shift(second, (low, high), spare);
        self.asm.alu(Alu::Or, Reg::Rax, low);
        self.asm.alu(Alu::Or, Reg::Rdx, high);
    }

    fn xor_icmp_half(&mut self, high: bool, reg: Reg, rhs: (Value, Option<i64>)) {
        if rhs.1 != Some(0) {
            self.alu_rhs_half(Alu::Xor, (high, reg), rhs);
        }
    }

    fn wide_icmp_eq(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        let ((lhs, lc), (rhs, rc)) = if imm_score(cx, true) > imm_score(cy, true) { ((y, cy), (x, cx)) } else { ((x, cx), (y, cy)) };
        let cond = if cc == IntCC::Equal { Cond::E } else { Cond::Ne };
        if lc.1.is_some() && lc.1 == rc.1 {
            self.load_half(false, Reg::Rax, lhs);
            self.alu_rhs_half(Alu::Cmp, (false, Reg::Rax), (rhs, rc.0));
            return cond;
        }
        if lc.0.is_some() && lc.0 == rc.0 {
            self.load_half(true, Reg::Rdx, lhs);
            self.alu_rhs_half(Alu::Cmp, (true, Reg::Rdx), (rhs, rc.1));
            return cond;
        }
        if matches!(rc, (Some(-1), Some(-1))) {
            self.load_half(false, Reg::Rax, lhs);
            self.alu_rhs_half(Alu::And, (true, Reg::Rax), (lhs, lc.1));
            self.asm.alu_imm(Alu::Cmp, Reg::Rax, -1);
            return cond;
        }
        let high = rc.0 == Some(0) && rc.1 != Some(0);
        let (reg, spare) = if high { (Reg::Rdx, Reg::Rax) } else { (Reg::Rax, Reg::Rdx) };
        self.load_half(high, reg, lhs);
        self.xor_icmp_half(high, reg, (rhs, half(rc, high)));
        if half(rc, !high) == Some(0) {
            self.alu_rhs_half(Alu::Or, (!high, reg), (lhs, half(lc, !high)));
        } else {
            self.load_half(!high, spare, lhs);
            self.xor_icmp_half(!high, spare, (rhs, half(rc, !high)));
            self.asm.alu(Alu::Or, reg, spare);
        }
        cond
    }

    pub(super) fn wide_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            return self.wide_icmp_eq(cc, x, y);
        }
        let (swap, cond) = icmp_order(cc);
        let (a, b) = if swap { (y, x) } else { (x, y) };
        let (ca, cb) = (self.known_halves(a), self.known_halves(b));
        if ca == (Some(0), Some(0)) && matches!(cond, Cond::B | Cond::Ae) {
            let eq = if cond == Cond::B { IntCC::NotEqual } else { IntCC::Equal };
            return self.wide_icmp_eq(eq, b, a);
        }
        if ca.1 == Some(0) && cb.1 == Some(0) {
            self.load_half(false, Reg::Rax, a);
            self.alu_rhs_half(Alu::Cmp, (false, Reg::Rax), (b, cb.0));
            return match cond {
                Cond::L => Cond::B,
                Cond::Ge => Cond::Ae,
                other => other,
            };
        }
        if cb.0 == Some(0) {
            self.load_half(true, Reg::Rdx, a);
            self.alu_rhs_half(Alu::Cmp, (true, Reg::Rdx), (b, cb.1));
            return cond;
        }
        self.emit_pair_alu((Alu::Cmp, Alu::Sbb), a, (b, cb.0, cb.1));
        cond
    }
}
