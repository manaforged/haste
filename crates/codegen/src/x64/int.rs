use super::asm::{Alu, Cond, Count, Group3, Mem, Reg, Shift, Size};
use super::Emit;
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};

/// The masks and shifts of the bit-parallel steps that reverse and count the bits of a word.
const BIT_STEPS: [(u64, u8); 3] = [(0x5555_5555_5555_5555, 1), (0x3333_3333_3333_3333, 2), (0x0F0F_0F0F_0F0F_0F0F, 4)];

pub(super) fn cond(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::E,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::L,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::G,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::B,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Ae,
        IntCC::UnsignedGreaterThan => Cond::A,
        IntCC::UnsignedLessThanOrEqual => Cond::Be,
    }
}

fn swap_cc(cc: IntCC) -> IntCC {
    match cc {
        IntCC::Equal => IntCC::Equal,
        IntCC::NotEqual => IntCC::NotEqual,
        IntCC::SignedLessThan => IntCC::SignedGreaterThan,
        IntCC::SignedGreaterThanOrEqual => IntCC::SignedLessThanOrEqual,
        IntCC::SignedGreaterThan => IntCC::SignedLessThan,
        IntCC::SignedLessThanOrEqual => IntCC::SignedGreaterThanOrEqual,
        IntCC::UnsignedLessThan => IntCC::UnsignedGreaterThan,
        IntCC::UnsignedGreaterThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        IntCC::UnsignedGreaterThan => IntCC::UnsignedLessThan,
        IntCC::UnsignedLessThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
    }
}

pub(super) fn alu_of(op: Binary) -> Option<Alu> {
    match op {
        Binary::Iadd => Some(Alu::Add),
        Binary::Isub => Some(Alu::Sub),
        Binary::Band => Some(Alu::And),
        Binary::Bor => Some(Alu::Or),
        Binary::Bxor => Some(Alu::Xor),
        _ => None,
    }
}

impl Emit<'_> {
    pub(super) fn unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if ty.is_vector() {
            return self.vector_unary(op, x, result);
        }
        if ty.is_float() {
            return self.float_unary(op, x, result);
        }
        if ty == Type::I128 {
            return self.wide_unary(op, x, result);
        }
        self.int_unary(op, x, ty.bits() as u8)?;
        self.put(result, Reg::Rax);
        Ok(())
    }

    fn int_unary(&mut self, op: Unary, x: Value, bits: u8) -> Result<(), String> {
        match op {
            Unary::Ineg | Unary::Bnot => {
                self.raw(Reg::Rax, x);
                self.asm.group3(if op == Unary::Ineg { Group3::Neg } else { Group3::Not }, Reg::Rax);
            }
            Unary::Bswap => self.swap_bytes(x, bits),
            Unary::Bitrev => {
                self.raw(Reg::Rax, x);
                self.reverse_bits();
                if bits < 64 {
                    self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rax, 64 - bits);
                }
            }
            Unary::Clz => {
                self.get(Reg::Rax, x, Ext::Zero);
                self.count(Count::Lzcnt, Reg::Rax, Reg::Rax);
                if bits < 64 {
                    self.asm.alu_imm(Alu::Sub, Reg::Rax, i32::from(64 - bits));
                }
            }
            Unary::Ctz => {
                self.raw(Reg::Rax, x);
                if bits < 64 {
                    self.asm.bts(Reg::Rax, bits);
                }
                self.count(Count::Tzcnt, Reg::Rax, Reg::Rax);
            }
            Unary::Popcnt => {
                self.get(Reg::Rax, x, Ext::Zero);
                self.count(Count::Popcnt, Reg::Rax, Reg::Rax);
            }
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    fn swap_bytes(&mut self, x: Value, bits: u8) {
        self.raw(Reg::Rax, x);
        match bits {
            16 => self.asm.shift_imm(Size::W, Shift::Rol, Reg::Rax, 8),
            32 => self.asm.bswap(Size::D, Reg::Rax),
            64 => self.asm.bswap(Size::Q, Reg::Rax),
            _ => {}
        }
    }

    /// `op` of the 64 bits of `src` into `dst`, with 64 leading or trailing zeros for zero; without the host instruction it uses r11 and rsi, where no value lives.
    pub(super) fn count(&mut self, op: Count, dst: Reg, src: Reg) {
        let fast = match op {
            Count::Lzcnt => self.features.lzcnt,
            Count::Tzcnt => self.features.bmi1,
            Count::Popcnt => self.features.popcnt,
        };
        if fast {
            return self.asm.count(op, dst, src);
        }
        let reverse = match op {
            Count::Popcnt => return self.popcount(dst, src),
            Count::Lzcnt => true,
            Count::Tzcnt => false,
        };
        self.asm.bit_scan(reverse, dst, src);
        self.asm.mov_imm(Reg::R11, if reverse { 127 } else { 64 });
        self.asm.cmov(Cond::E, dst, Reg::R11);
        if reverse {
            self.asm.alu_imm(Alu::Xor, dst, 63);
        }
    }

    /// Counts the set bits of `src` into `dst` by summing ever wider fields, using r11 and rsi.
    fn popcount(&mut self, dst: Reg, src: Reg) {
        let (part, mask) = (Reg::R11, Reg::Rsi);
        if dst != src {
            self.asm.mov(dst, src);
        }
        for (bits, shift) in BIT_STEPS {
            self.asm.mov_imm(mask, bits as i64);
            self.asm.mov(part, dst);
            self.asm.shift_imm(Size::Q, Shift::Shr, part, shift);
            self.asm.alu(Alu::And, part, mask);
            self.asm.alu(Alu::And, dst, mask);
            self.asm.alu(Alu::Add, dst, part);
        }
        self.asm.mov_imm(mask, 0x0101_0101_0101_0101);
        self.asm.imul(dst, mask);
        self.asm.shift_imm(Size::Q, Shift::Shr, dst, 56);
    }

    /// Reverses the 64 bits of rax in place, using rcx and rdx.
    pub(super) fn reverse_bits(&mut self) {
        for (mask, shift) in BIT_STEPS {
            self.asm.mov_imm(Reg::Rcx, mask as i64);
            self.asm.mov(Reg::Rdx, Reg::Rax);
            self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, shift);
            self.asm.alu(Alu::And, Reg::Rdx, Reg::Rcx);
            self.asm.alu(Alu::And, Reg::Rax, Reg::Rcx);
            self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, shift);
            self.asm.alu(Alu::Or, Reg::Rax, Reg::Rdx);
        }
        self.asm.bswap(Size::Q, Reg::Rax);
    }

    pub(super) fn binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if self.is_static(result) || self.plan.deferred[result.index()] {
            return Ok(());
        }
        if op == Binary::Iconcat {
            self.raw(Reg::Rax, x);
            self.put(result, Reg::Rax);
            self.raw(Reg::Rax, y);
            self.put_high(result, Reg::Rax);
            return Ok(());
        }
        if ty.is_vector() {
            return self.vector_binary(op, (x, y), result);
        }
        if ty.is_float() {
            return self.float_binary(op, x, y, result);
        }
        if ty == Type::I128 {
            return self.wide_binary(op, x, y, result);
        }
        let dst = self.dst(result);
        let reg = self.int_binary(op, (x, y), ty, dst, result)?;
        self.put(result, reg);
        Ok(())
    }

    /// Computes `x op y` into `dst` for the operations that can, or into rax, and returns the register that holds it.
    fn int_binary(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg, result: Value) -> Result<Reg, String> {
        if let Some(alu) = alu_of(op) {
            let size = if ty == Type::I32 { Size::D } else { Size::Q };
            let (x, y) = if !matches!(alu, Alu::Sub)
                && x != y
                && ((self.imm_sized(y, size).is_none() && self.reg_of(y) == Some(dst))
                    || (self.imm_sized(x, size).is_some() && self.imm_sized(y, size).is_none()))
            {
                (y, x)
            } else {
                (x, y)
            };
            if matches!(alu, Alu::And)
                && let Some(mask) = self.const_of(y)
                && let Some(narrow) = match mask {
                    0xFF => Some(Size::B),
                    0xFFFF => Some(Size::W),
                    0xFFFF_FFFF => Some(Size::D),
                    _ => None,
                }
            {
                let source = self.operand(x);
                self.asm.load(narrow, dst, source);
                return Ok(dst);
            }
            if matches!(alu, Alu::Add) && size == Size::Q {
                if let Some(rx) = self.reg_of(x).filter(|&rx| rx != dst) {
                    if let Some(imm) = self.imm_sized(y, size) {
                        self.asm.lea(dst, Mem::new(rx, imm));
                        return Ok(dst);
                    }
                    if let Some(ry) = self.reg_of(y) {
                        self.asm.lea(dst, Mem::sib(rx, ry, 0, 0));
                        return Ok(dst);
                    }
                }
                if let (Some(imm), Some(ry)) = (self.imm_sized(x, size), self.reg_of(y).filter(|&ry| ry != dst)) {
                    self.asm.lea(dst, Mem::new(ry, imm));
                    return Ok(dst);
                }
            }
            if self.imm_sized(y, size).is_none() && self.reg_of(y) == Some(dst) && x != y {
                self.asm.mov(Reg::Rax, dst);
                self.raw(dst, x);
                self.asm.alu_sized(size, alu, dst, Reg::Rax);
            } else {
                self.raw(dst, x);
                match self.imm_sized(y, size) {
                    Some(imm) => self.asm.alu_imm_sized(size, alu, dst, imm),
                    None => {
                        let source = self.operand(y);
                        self.asm.alu_sized(size, alu, dst, source);
                    }
                }
            }
            return Ok(dst);
        }
        match op {
            Binary::Imul => {
                self.multiply(dst, x, y);
                return Ok(dst);
            }
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => return Ok(dst),
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => self.divide(op, x, y),
            Binary::Umulhi | Binary::Smulhi => self.multiply_high(op == Binary::Smulhi, x, y, ty.bits() as u8),
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                let signed = matches!(op, Binary::Smin | Binary::Smax);
                self.get(Reg::Rax, x, Ext::of(signed));
                self.get(Reg::Rcx, y, Ext::of(signed));
                self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
                let take_y = match op {
                    Binary::Smin => Cond::G,
                    Binary::Smax => Cond::L,
                    Binary::Umin => Cond::A,
                    _ => Cond::B,
                };
                self.asm.cmov(take_y, Reg::Rax, Reg::Rcx);
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                self.raw(Reg::Rcx, y);
                self.raw(dst, x);
                self.emit_sat_op(op, Size::of(ty.bytes()), dst, Reg::Rcx);
                return Ok(dst);
            }
            Binary::AvgRound => {
                self.get(Reg::Rcx, y, Ext::Zero);
                self.get(dst, x, Ext::Zero);
                self.asm.mov(Reg::Rdx, dst);
                self.asm.alu(Alu::Or, dst, Reg::Rcx);
                self.asm.alu(Alu::Xor, Reg::Rdx, Reg::Rcx);
                self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rdx, 1);
                self.asm.alu(Alu::Sub, dst, Reg::Rdx);
                return Ok(dst);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr => return Ok(self.shift_by(op, (x, y), ty, dst, result)),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(Reg::Rax)
    }

    /// Multiplies `x` by `y` into `dst`, with a shift or an immediate when either factor is a constant.
    fn multiply(&mut self, dst: Reg, x: Value, y: Value) {
        let (x, imm) = match (self.imm_of(x), self.imm_of(y)) {
            (_, Some(imm)) => (x, imm),
            (Some(imm), None) => (y, imm),
            (None, None) => {
                let (x, y) = if self.reg_of(y) == Some(dst) && x != y { (y, x) } else { (x, y) };
                self.raw(dst, x);
                let source = self.operand(y);
                self.asm.imul(dst, source);
                return;
            }
        };
        if imm == 0 {
            self.asm.mov_imm(dst, 0);
            return;
        }
        if imm > 0 && imm.count_ones() == 1 {
            self.raw(dst, x);
            if imm > 1 {
                self.asm.shift_imm(Size::Q, Shift::Shl, dst, imm.trailing_zeros() as u8);
            }
            return;
        }
        let source = self.operand(x);
        self.asm.imul_imm(dst, source, imm);
    }

    /// Divides or takes the remainder by a power of two into `dst` with a shift or a mask, when `op` is unsigned and `y` is one.
    fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self.const_of(y).map(|imm| crate::fold::extend(imm, ty, false) as u64).filter(|d| d.is_power_of_two()) else {
            return false;
        };
        match op {
            Binary::Udiv => {
                self.get(dst, x, Ext::Zero);
                let shift = divisor.trailing_zeros() as u8;
                if shift != 0 {
                    self.asm.shift_imm(Size::Q, Shift::Shr, dst, shift);
                }
            }
            Binary::Urem => {
                let Ok(mask) = i32::try_from(divisor - 1) else { return false };
                self.raw(dst, x);
                self.asm.alu_imm(Alu::And, dst, mask);
            }
            _ => return false,
        }
        true
    }

    fn divide(&mut self, op: Binary, x: Value, y: Value) {
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let ext = Ext::of(signed);
        self.get(Reg::Rax, x, ext);
        self.get(Reg::Rcx, y, ext);
        if signed {
            self.asm.cqo();
            self.asm.group3(Group3::Idiv, Reg::Rcx);
        } else {
            self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
            self.asm.group3(Group3::Div, Reg::Rcx);
        }
        if matches!(op, Binary::Srem | Binary::Urem) {
            self.asm.mov(Reg::Rax, Reg::Rdx);
        }
    }

    fn multiply_high(&mut self, signed: bool, x: Value, y: Value, bits: u8) {
        let ext = Ext::of(signed);
        self.get(Reg::Rax, x, ext);
        self.get(Reg::Rcx, y, ext);
        if bits == 64 {
            self.asm.group3(if signed { Group3::Imul } else { Group3::Mul }, Reg::Rcx);
            self.asm.mov(Reg::Rax, Reg::Rdx);
        } else {
            self.asm.imul(Reg::Rax, Reg::Rcx);
            self.asm.shift_imm(Size::Q, if signed { Shift::Sar } else { Shift::Shr }, Reg::Rax, bits);
        }
    }

    fn unwrap_shift_mask(&self, y: Value, shift_res: Value, bits: u32) -> Value {
        if !matches!(bits, 32 | 64) {
            return y;
        }
        let Some(y_inst) = self.defs[y.index()] else { return y };
        let Op::Binary(Binary::Band, a, b) = self.func.insts[y_inst.index()].op else { return y };
        let mask = u64::from(bits - 1);
        let const_val = |v: Value| {
            self.const_of(v).or_else(|| {
                self.defs[v.index()].and_then(|i| match self.func.insts[i.index()].op {
                    Op::Const(imm) => Some(imm),
                    _ => None,
                })
            })
        };
        let Some(raw) = [(a, b), (b, a)].into_iter().find_map(|(raw, m)| {
            let imm = const_val(m)?;
            ((imm as u64 & mask) == mask).then_some(raw)
        }) else {
            return y;
        };
        if self.plan.elided[y.index()] {
            return raw;
        }
        let Some(r_raw) = self.reg_of(raw) else { return y };
        if self.plan.elided[raw.index()] || self.reg_of(y) == Some(r_raw) {
            return y;
        }
        let Some(shift_inst) = self.defs[shift_res.index()] else { return y };
        for block in &self.func.blocks {
            let Some(ypos) = block.insts.iter().position(|&i| i == y_inst) else { continue };
            let Some(spos) = block.insts.iter().position(|&i| i == shift_inst) else { return y };
            if ypos >= spos {
                return y;
            }
            let ok = block.insts[ypos + 1..spos].iter().all(|&inst| {
                let data = &self.func.insts[inst.index()];
                if data.result.is_some_and(|r| self.plan.elided[r.index()] || self.plan.deferred[r.index()]) {
                    return true;
                }
                !matches!(data.op, Op::Call { .. } | Op::TryCall { .. })
                    && !data.result.and_then(|r| self.plan.pairs[r.index()]).is_some_and(|p| self.reg_of(p) == Some(r_raw))
                    && !crate::live::results(data).any(|r| self.reg_of(r) == Some(r_raw))
            });
            return if ok { raw } else { y };
        }
        y
    }

    /// Shifts or rotates `x` by `y`: into `dst` by an immediate when `y` is a constant, else into rax by cl; returns the register that holds the result.
    fn shift_by(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg, result: Value) -> Reg {
        let (ext, shift) = match op {
            Binary::Ishl => (Ext::Zero, Shift::Shl),
            Binary::Ushr => (Ext::Zero, Shift::Shr),
            Binary::Sshr => (Ext::Sign, Shift::Sar),
            Binary::Rotl => (Ext::Zero, Shift::Rol),
            _ => (Ext::Zero, Shift::Ror),
        };
        let size = match op {
            Binary::Rotl | Binary::Rotr => Size::of(ty.bytes()),
            _ if ty == Type::I32 => Size::D,
            _ => Size::Q,
        };
        let need_ext = op != Binary::Ishl && ty != Type::I32;
        if let Some(imm) = self.const_of(y) {
            let count = (imm & i64::from(ty.bits() - 1)) as u8;
            if need_ext || (count == 0 && ty == Type::I32 && matches!(op, Binary::Ushr | Binary::Rotl | Binary::Rotr) && !self.plan.zero_ext[x.index()]) {
                self.get(dst, x, ext);
            } else {
                self.raw(dst, x);
            }
            if count != 0 {
                self.asm.shift_imm(size, shift, dst, count);
            }
            return dst;
        }
        let y = self.unwrap_shift_mask(y, result, ty.bits());
        if need_ext {
            self.get(Reg::Rax, x, ext);
        } else {
            self.raw(Reg::Rax, x);
        }
        self.raw(Reg::Rcx, y);
        if matches!(ty, Type::I8 | Type::I16) && !matches!(op, Binary::Rotl | Binary::Rotr) {
            self.asm.alu_imm32(Alu::And, Reg::Rcx, ty.bits() as i32 - 1);
        }
        self.asm.shift(size, shift, Reg::Rax);
        Reg::Rax
    }

    pub(super) fn icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        if self.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() {
            return self.vector_icmp(cc, x, y, result);
        }
        let flags = if self.ty(x) == Type::I128 {
            self.wide_icmp(cc, x, y)
        } else {
            let size = Size::of(self.ty(x).bytes());
            let (cc, x, y) = if self.reg_of(x).is_none() && (self.reg_of(y).is_some() || self.imm_sized(x, size).is_some()) {
                (swap_cc(cc), y, x)
            } else {
                (cc, x, y)
            };
            if matches!(self.location(x), Location::Slot)
                && self.cached_spill(x).is_none()
                && let Some(imm) = self.imm_sized(y, size)
            {
                self.asm.alu_imm_sized(size, Alu::Cmp, self.at(x), imm);
            } else {
                let left = self.reg_of(x).or_else(|| self.cached_spill(x)).unwrap_or_else(|| {
                    self.raw(Reg::Rax, x);
                    Reg::Rax
                });
                if matches!(cc, IntCC::Equal | IntCC::NotEqual)
                    && self.const_of(y).is_some_and(|imm| crate::fold::extend(imm, self.ty(y), false) == 0)
                {
                    self.asm.test_sized(size, left, left);
                } else {
                    self.alu_at(size, Alu::Cmp, left, y);
                }
            }
            cond(cc)
        };
        if self.plan.fused[result.index()] {
            self.flags = Some((result, flags));
            return Ok(());
        }
        let dst = self.dst(result);
        self.asm.setcc(flags, dst);
        self.asm.load(Size::B, dst, dst);
        self.put(result, dst);
        Ok(())
    }

    /// Sets `result` to whether `x op y` fits its type, from the carry or overflow flag of the operation at that width; computes the paired operation's result with the same instruction.
    pub(super) fn fits(&mut self, op: Binary, signed: bool, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let pair = self.plan.pairs[result.index()];
        if self.is_static(result) && pair.is_none() {
            return Ok(());
        }
        let size = Size::of(self.ty(x).bytes());
        let reg = match op {
            Binary::Iadd | Binary::Isub => {
                let alu = if op == Binary::Iadd { Alu::Add } else { Alu::Sub };
                let reg = pair.and_then(|sum| self.reg_of(sum)).filter(|reg| Some(*reg) != self.reg_of(y)).unwrap_or(Reg::Rax);
                self.raw(reg, x);
                self.alu_at(size, alu, reg, y);
                reg
            }
            Binary::Imul => {
                self.raw(Reg::Rax, x);
                self.raw(Reg::Rcx, y);
                self.asm.group3_sized(size, if signed { Group3::Imul } else { Group3::Mul }, Reg::Rcx);
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

    /// `dst op= y` on the low `size` bytes, with an immediate when `y` is a constant that fits one at that width.
    pub(super) fn alu_at(&mut self, size: Size, alu: Alu, dst: Reg, y: Value) {
        match self.imm_sized(y, size) {
            Some(imm) => self.asm.alu_imm_sized(size, alu, dst, imm),
            None => {
                let source = self.operand(y);
                self.asm.alu_sized(size, alu, dst, source);
            }
        }
    }

    pub(super) fn convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        if self.is_static(result) {
            return Ok(());
        }
        if self.ty(x).is_vector() || self.ty(result).is_vector() {
            return self.vector_convert(op, x, result);
        }
        let wide = self.ty(result) == Type::I128;
        match op {
            Convert::Uextend | Convert::Sextend => {
                let signed = op == Convert::Sextend;
                if !wide
                    && matches!(self.location(result), Location::Slot)
                    && let Some(rx) = self.reg_of(x).or_else(|| self.cached_spill(x))
                    && (self.ty(x).bytes() >= 8 || (!signed && self.plan.zero_ext[x.index()]))
                {
                    self.put(result, rx);
                    return Ok(());
                }
                let dst = self.dst(result);
                self.get(dst, x, Ext::of(signed));
                self.put(result, dst);
                if wide && signed {
                    self.asm.mov(Reg::Rcx, Reg::Rax);
                    self.asm.shift_imm(Size::Q, Shift::Sar, Reg::Rcx, 63);
                    self.put_high(result, Reg::Rcx);
                } else if wide {
                    self.asm.alu32(Alu::Xor, Reg::Rcx, Reg::Rcx);
                    self.put_high(result, Reg::Rcx);
                }
                Ok(())
            }
            Convert::Ireduce => {
                if matches!(self.location(result), Location::Slot)
                    && let Some(rx) = self.reg_of(x).or_else(|| self.cached_spill(x))
                {
                    self.put(result, rx);
                    return Ok(());
                }
                let dst = self.dst(result);
                self.raw(dst, x);
                self.put(result, dst);
                Ok(())
            }
            Convert::NarrowSat { signed } => {
                let dst = self.dst(result);
                self.get(dst, x, Ext::Sign);
                self.clamp_signed(dst, self.ty(result), signed);
                self.put(result, dst);
                Ok(())
            }
            _ => self.float_convert(op, x, result),
        }
    }

    pub(super) fn emit_sat_op(&mut self, op: Binary, size: Size, dst: Reg, src: Reg) {
        let (umax, smin, smax) = match size {
            Size::B => (0xFF_i64, 0x80_i64, 0x7F_i64),
            Size::W => (0xFFFF_i64, 0x8000_i64, 0x7FFF_i64),
            Size::D => (0xFFFF_FFFF_i64, 0x8000_0000_i64, 0x7FFF_FFFF_i64),
            Size::Q => (-1_i64, i64::MIN, i64::MAX),
        };
        match op {
            Binary::UaddSat => {
                self.asm.alu_sized(size, Alu::Add, dst, src);
                self.asm.mov_imm(Reg::Rdx, umax);
                self.asm.cmov(Cond::B, dst, Reg::Rdx);
            }
            Binary::UsubSat => {
                self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
                self.asm.alu_sized(size, Alu::Sub, dst, src);
                self.asm.cmov(Cond::B, dst, Reg::Rdx);
            }
            Binary::SaddSat | Binary::SsubSat => {
                let alu = if op == Binary::SaddSat { Alu::Add } else { Alu::Sub };
                self.asm.alu_sized(size, alu, dst, src);
                self.asm.mov_imm(Reg::Rdx, smin);
                self.asm.mov_imm(Reg::R11, smax);
                self.asm.cmov(Cond::S, Reg::Rdx, Reg::R11);
                self.asm.cmov(Cond::O, dst, Reg::Rdx);
            }
            _ => {}
        }
    }

    pub(super) fn clamp_signed(&mut self, reg: Reg, to: Type, signed: bool) {
        let bits = to.bits();
        if bits >= 64 {
            return;
        }
        let (lo, hi) = if signed {
            (-(1_i64 << (bits - 1)), (1_i64 << (bits - 1)) - 1)
        } else {
            (0, (1_i64 << bits) - 1)
        };
        self.asm.mov_imm(Reg::Rcx, lo);
        self.asm.alu(Alu::Cmp, reg, Reg::Rcx);
        self.asm.cmov(Cond::L, reg, Reg::Rcx);
        self.asm.mov_imm(Reg::Rcx, hi);
        self.asm.alu(Alu::Cmp, reg, Reg::Rcx);
        self.asm.cmov(Cond::G, reg, Reg::Rcx);
    }
}
