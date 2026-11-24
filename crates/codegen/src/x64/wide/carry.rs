use crate::alloc::{Location, Plan};
use crate::frame::Ext;
use crate::ir::{Binary, Convert, Function, Inst, IntCC, Op, TargetOpId, Type, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Cond, Reg, Rm, Size};

fn carry_tag(v: Value) -> usize {
    2 * v.index()
}

/// The held-carry tag that says the carry flag is the unsigned overflow of the add that produced `sum`.
pub(in crate::x64) fn overflow_tag(sum: Value) -> usize {
    2 * sum.index() + 1
}

fn carry_op(op: &Op) -> Option<(TargetOpId, &[Value])> {
    match op {
        Op::TargetOp(
            id @ (TargetOpId::X64AddCarry
            | TargetOpId::X64AddCarryOut
            | TargetOpId::X64SubBorrow
            | TargetOpId::X64SubBorrowOut),
            args,
        ) if args.as_slice().len() == 3 => Some((*id, args.as_slice())),
        _ => None,
    }
}

/// Marks each carry out whose every use is the carry operand of the next carry pair in its block, with only register loads and stores between, so the carry flag hands it over and no register needs it.
pub(in crate::x64) fn flag_carries(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
) -> Vec<bool> {
    let mut uses = vec![0_u32; func.types.len()];
    for inst in func.blocks.iter().flat_map(|block| &block.insts) {
        func.insts[inst.index()]
            .op
            .for_operands(|value| uses[value.index()] += 1);
    }
    let held = |value: Value| {
        !plan.elided[value.index()] && matches!(plan.locations[value.index()], Location::Reg(_))
    };
    let moved = |value: Value| {
        !plan.elided[value.index()]
            && matches!(
                plan.locations[value.index()],
                Location::Reg(_) | Location::Slot
            )
    };
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
            let (Some((TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrowOut, _)), Some(out)) =
                (carry_op(&data.op), data.result)
            else {
                continue;
            };
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
                    Op::Load(addr, _) => {
                        held(addr)
                            && func.insts[next.index()]
                                .result
                                .is_some_and(|r| moved(r) || in_memory(r))
                    }
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
pub(in crate::x64) fn overflow_reads(func: &Function) -> Vec<bool> {
    let mut read = vec![false; func.types.len()];
    let mut compared = std::collections::HashSet::new();
    for block in &func.blocks {
        compared.clear();
        for inst in &block.insts {
            if let Op::Icmp(_, x, y) = func.insts[inst.index()].op {
                compared.extend([(x, y), (y, x)]);
            }
        }
        for (at, data) in block
            .insts
            .iter()
            .map(|inst| &func.insts[inst.index()])
            .enumerate()
        {
            let (Op::Binary(op @ (Binary::Iadd | Binary::Isub), a, b), Some(result)) =
                (&data.op, data.result)
            else {
                continue;
            };
            let next = block
                .insts
                .get(at + 1)
                .map(|inst| &func.insts[inst.index()].op);
            let sum_read = matches!(
                next,
                Some(&Op::Icmp(_, x, y))
                    if (x == result && (y == *a || y == *b))
                        || (y == result && (x == *a || x == *b))
            );
            read[result.index()] = matches!(func.ty(result), Type::I32 | Type::I64)
                && if *op == Binary::Iadd {
                    sum_read
                } else {
                    compared.contains(&(*a, *b))
                };
        }
    }
    read
}

/// Marks each sum whose carry compare an `adc` or `sbb` absorbed, directly or through an absorbed widen, so the sum must set the carry flag.
pub(in crate::x64) fn adc_sums(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
) -> Vec<bool> {
    let mut marked = vec![false; func.types.len()];
    let op_of = |v: Value| defs[v.index()].map(|inst| &func.insts[inst.index()].op);
    let sum_of = |sum: Value, operand: Value| matches!(op_of(sum), Some(&Op::Binary(Binary::Iadd, a, b)) if a == operand || b == operand);
    for data in func
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .map(|inst| &func.insts[inst.index()])
    {
        let Op::Binary(Binary::Iadd | Binary::Isub, x, y) = data.op else {
            continue;
        };
        for operand in [x, y].into_iter().filter(|v| plan.elided[v.index()]) {
            let compare = match op_of(operand) {
                Some(&Op::Convert(Convert::Uextend, inner)) if plan.elided[inner.index()] => inner,
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

impl Emit<'_> {
    /// Adds or subtracts `a` and `b` after the carry in `args[0]` by `adc` or `sbb` into `result`: the sum, or the carry out, which reads the flags of a sum of the same operands emitted just before.
    pub(in crate::x64) fn carry_chain(
        &mut self,
        id: TargetOpId,
        args: [Value; 3],
        result: Value,
    ) -> Result<(), String> {
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
            let target = Some(self.dst(result))
                .filter(|&reg| {
                    !out && !matches!(reg, Reg::Rdx | Reg::Rcx | Reg::Rsp)
                        && !busy.contains(&Some(reg))
                })
                .unwrap_or(Reg::Rax);
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
            self.asm
                .alu(if add { Alu::Adc } else { Alu::Sbb }, target, source);
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
    pub(in crate::x64) fn folded_product(
        &mut self,
        op: Binary,
        (x, y): (Value, Value),
        result: Value,
    ) -> Option<Reg> {
        let alu = match op {
            Binary::Bxor => Alu::Xor,
            Binary::Bor => Alu::Or,
            Binary::Iadd => Alu::Add,
            _ => return None,
        };
        if !(self.plan.elided[x.index()] && self.plan.elided[y.index()]) {
            return None;
        }
        let (signed, a, b) =
            crate::x64::location::product_halves(self.func, &|v| self.defs[v.index()], (x, y))?;
        self.multiply_high(signed, a, b, 64);
        self.low_product = None;
        self.asm.alu(alu, Reg::Rax, Reg::Rdx);
        self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
        self.flags = self.plan.fused[result.index()].then_some((result, Cond::Ne));
        Some(Reg::Rax)
    }

    /// Adds or subtracts the carry `left <u right` (or `left != 0`), inverted when `invert`, to `other` by `adc` or `sbb`, which also computes `other` when it is a folded add or subtract.
    pub(in crate::x64) fn carry_add(
        &mut self,
        op: Binary,
        ty: Type,
        dst: Reg,
        (left, other, right, invert): (Value, Value, Option<Value>, bool),
    ) -> Result<(), String> {
        let size = if ty == Type::I32 { Size::D } else { Size::Q };
        let fold = if self.plan.elided[other.index()] {
            crate::x64::location::adc_fold(self.func, op, invert, other, |v| self.defs[v.index()])
        } else {
            None
        };
        let ready = |emit: &Self, v: Value| {
            !emit.plan.elided[v.index()]
                && matches!(
                    emit.location(v),
                    Location::Reg(_) | Location::Slot | Location::Imm(_)
                )
        };
        let operands_ready = fold.map_or_else(
            || ready(self, other),
            |(p, q)| ready(self, p) && ready(self, q),
        );
        let held = operands_ready && right.is_some_and(|right| self.overflow_held(left, right));
        if !held {
            self.emit_cmp_pair(Size::of(self.ty(left).bytes()), left, right, false, false);
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
            (None, Some((_, q))) => {
                self.asm
                    .alu_imm_sized(size, alu, dst, self.imm_sized(q, size).unwrap_or_default());
            }
            (None, None) => self.asm.alu_imm_sized(size, alu, dst, imm),
        }
        Ok(())
    }

    /// Whether the carry flag still holds `left <u right`, from the add that produced `left` from `right` or the subtract of `right` from `left`.
    pub(in crate::x64) fn overflow_held(&self, left: Value, right: Value) -> bool {
        let Some(source) = self
            .asm
            .held_carry()
            .filter(|tag| tag % 2 == 1)
            .map(|tag| Value((tag / 2) as u32))
        else {
            return false;
        };
        self.defs[source.index()].is_some_and(|inst| match self.func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, a, b) => source == left && (a == right || b == right),
            Op::Binary(Binary::Isub, a, b) => a == left && b == right,
            _ => false,
        })
    }

    /// The condition that reads the unsigned compare `x cc y` from a carry or borrow the carry flag still holds for these operands.
    pub(in crate::x64) fn held_compare(&self, cc: IntCC, x: Value, y: Value) -> Option<Cond> {
        let (left, right, cond) = match cc {
            IntCC::UnsignedLessThan => (x, y, Cond::B),
            IntCC::UnsignedGreaterThan => (y, x, Cond::B),
            IntCC::UnsignedGreaterThanOrEqual => (x, y, Cond::Ae),
            IntCC::UnsignedLessThanOrEqual => (y, x, Cond::Ae),
            _ => return None,
        };
        self.overflow_held(left, right).then_some(cond)
    }
}
