use crate::alloc::Location;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Cond, Group3, Reg, Size};

impl Emit<'_> {
    pub(in crate::x64) fn elided_band_cmp(&self, x: Value, y: Value) -> Option<(Value, Value)> {
        crate::emit::elided_band_cmp(self.func, &self.plan, &self.defs, (x, y), |v| {
            self.const_of(v)
                .is_some_and(|imm| crate::fold::extend(imm, self.ty(v), false) == 0)
        })
    }

    pub(in crate::x64) fn emit_cmp_pair(
        &mut self,
        size: Size,
        x: Value,
        y: Option<Value>,
        zero_test: bool,
        unsigned: bool,
    ) {
        let (size, imm) = match y {
            Some(yv) => match self.imm_sized(yv, size) {
                Some(imm) => (size, Some(imm)),
                None if unsigned && size == Size::Q && self.narrow_u32_cmp(x, yv) => {
                    (Size::D, self.const_of(yv).map(|c| c as u32 as i32))
                }
                None => (size, None),
            },
            None => (size, Some(1)),
        };
        if (matches!(self.location(x), Location::Slot) && self.cached_spill(x).is_none()
            || self.has_elided_mem(x))
            && let Some(imm) = imm
        {
            let rm = self.operand(x);
            self.asm.alu_imm_sized(size, Alu::Cmp, rm, imm);
            return;
        }
        let left = self
            .reg_of(x)
            .or_else(|| {
                y.filter(|&yv| imm.is_none() && self.imm_sized(yv, size).is_none())
                    .map_or_else(|| self.cached_spill(x), |yv| self.cached_beside(x, yv))
            })
            .unwrap_or_else(|| {
                self.raw(Reg::Rax, x);
                Reg::Rax
            });
        match y {
            Some(yv)
                if zero_test
                    && self
                        .const_of(yv)
                        .is_some_and(|c| crate::fold::extend(c, self.ty(yv), false) == 0) =>
            {
                self.asm.test_sized(size, left, left);
            }
            Some(_)
                if size == Size::D
                    && self.ty(x) == Type::I64
                    && let Some(imm) = imm =>
            {
                self.asm.alu_imm_sized(Size::D, Alu::Cmp, left, imm);
            }
            Some(yv) => self.alu_at(size, Alu::Cmp, left, yv),
            None => self.asm.alu_imm_sized(size, Alu::Cmp, left, 1),
        }
    }

    fn narrow_u32_cmp(&self, x: Value, y: Value) -> bool {
        let Some(c) = self.const_of(y) else {
            return false;
        };
        if u32::try_from(c as u64).is_err() || i32::try_from(c).is_ok() {
            return false;
        }
        let Some(inst) = self.defs[x.index()] else {
            return false;
        };
        match self.func.insts[inst.index()].op {
            Op::Convert(Convert::Uextend, a) => self.ty(a).bits() <= 32,
            Op::Binary(Binary::Ushr, _, sh) => self.const_of(sh).is_some_and(|k| (k & 63) >= 32),
            Op::Binary(Binary::Band, a, b) => [a, b].into_iter().any(|v| {
                self.const_of(v)
                    .is_some_and(|m| u32::try_from(m as u64).is_ok())
            }),
            _ => false,
        }
    }

    pub(in crate::x64) fn icmp(
        &mut self,
        cc: IntCC,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let live_flags = self.flags.take();
        if self.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() {
            self.invalidate_spills();
            let res = self.vector_icmp(cc, x, y, result);
            self.invalidate_spills();
            return res;
        }
        let flags = self.cmp_flags(cc, x, y, live_flags);
        if self.plan.fused[result.index()] {
            self.flags = Some((result, flags));
            return Ok(());
        }
        let dst = self.dst(result);
        self.asm.setcc(flags, dst);
        self.asm.load(Size::B, dst, dst);
        self.put(result, dst);
        if self.ty(x) != Type::I128 {
            self.flags = Some((result, flags));
        }
        Ok(())
    }

    /// Sets `result` to whether `x op y` fits its type, from the carry or overflow flag of the operation at that width; computes the paired operation's result with the same instruction.
    pub(in crate::x64) fn fits(
        &mut self,
        op: Binary,
        signed: bool,
        (x, y): (Value, Value),
        result: Value,
    ) -> Result<(), String> {
        self.flags = None;
        let pair = self.plan.pairs[result.index()];
        if self.is_static(result) && pair.is_none() {
            return Ok(());
        }
        let size = Size::of(self.ty(x).bytes());
        let reg = match op {
            Binary::Iadd | Binary::Isub => {
                let alu = if op == Binary::Iadd {
                    Alu::Add
                } else {
                    Alu::Sub
                };
                let reg = pair
                    .and_then(|sum| self.reg_of(sum))
                    .filter(|reg| Some(*reg) != self.reg_of(y))
                    .unwrap_or(Reg::Rax);
                self.raw(reg, x);
                self.alu_at(size, alu, reg, y);
                self.clobber_reg(reg);
                reg
            }
            Binary::Imul => {
                self.raw(Reg::Rax, x);
                self.raw(Reg::Rcx, y);
                self.asm.group3_sized(
                    size,
                    if signed { Group3::Imul } else { Group3::Mul },
                    Reg::Rcx,
                );
                self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
                Reg::Rax
            }
            other => return Err(format!("{other:?} has no overflow flag")),
        };
        if let Some(sum) = pair {
            self.put(sum, reg);
        }
        let ok = if signed { Cond::No } else { Cond::Ae };
        if self.plan.fused[result.index()] {
            self.flags = Some((result, ok));
            return Ok(());
        }
        let dst = self.dst(result);
        self.asm.setcc(ok, dst);
        self.asm.load(Size::B, dst, dst);
        self.put(result, dst);
        Ok(())
    }
}
