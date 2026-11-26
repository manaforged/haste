use super::Emit;
use super::asm::{Alu, Cond, Group3, Mem, Reg, Rm, Shift, Size};
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{AtomicRmwOp, Binary, Function, Op, Type, Value};

/// The masks and shifts of the bit-parallel steps that reverse and count the bits of a word.
const BIT_STEPS: [(u64, u8); 3] = [
    (0x5555_5555_5555_5555, 1),
    (0x3333_3333_3333_3333, 2),
    (0x0F0F_0F0F_0F0F_0F0F, 4),
];

/// Matches an atomic `Or`/`Xor` mask `1 << bit` whose `bit` is masked strictly inside `0..ty.bits()`, returning the x64 `lock bts`/`lock btc` opcode and `bit`.
pub(super) fn lock_bt_shift_bit(
    func: &Function,
    plan: &crate::alloc::Plan<Reg>,
    def: impl Fn(Value) -> Option<crate::ir::Inst>,
    op: AtomicRmwOp,
    value: Value,
) -> Option<(u8, Value)> {
    let opcode = match op {
        AtomicRmwOp::Or => 0xAB,
        AtomicRmwOp::Xor => 0xBB,
        _ => return None,
    };
    let ty = func.ty(value);
    if !matches!(ty, Type::I32 | Type::I64) {
        return None;
    }
    let Op::Binary(Binary::Ishl, one, bit) = func.insts[def(value)?.index()].op else {
        return None;
    };
    if func.ty(bit) != ty || plan.const_of(one) != Some(1) {
        return None;
    }
    let Op::Binary(Binary::Band, a, b) = func.insts[def(bit)?.index()].op else {
        return None;
    };
    let limit = i64::from(ty.bits());
    let bounded = [a, b]
        .into_iter()
        .any(|v| plan.const_of(v).is_some_and(|m| (0..limit).contains(&m)));
    bounded.then_some((opcode, bit))
}

/// Unwraps an elided constant-offset `Iadd`/`Isub` address feeding an atomic op into `(base, disp)`.
pub(super) fn atomic_addr_base(
    func: &Function,
    plan: &crate::alloc::Plan<Reg>,
    defs: &[Option<crate::ir::Inst>],
    addr: Value,
) -> Option<(Value, i32)> {
    if !plan.elided[addr.index()] || plan.sib[addr.index()].is_some() {
        return None;
    }
    let inst = defs[addr.index()]?;
    let (base, delta) = crate::alloc::iadd_sub_const(&func.insts[inst.index()].op, |v| {
        plan.const_or_def(func, |x| defs[x.index()], v)
    })?;
    i32::try_from(delta).ok().map(|disp| (base, disp))
}

impl Emit<'_> {
    fn atomic_offset(&self, addr: Value) -> (Value, i32) {
        atomic_addr_base(self.func, &self.plan, &self.defs, addr).unwrap_or((addr, 0))
    }

    /// The memory at `addr`: folds elided constant offsets and stack frames, using the register holding the base unless the operation writes it, else `scratch`.
    pub(super) fn atomic_target(&mut self, addr: Value, scratch: Reg, written: &[Reg]) -> Mem {
        let (base, disp) = self.atomic_offset(addr);
        if let Location::Frame(slot, off) = self.location(base)
            && let Some(total) = self.frame.slots[slot.index()]
                .checked_add(off)
                .and_then(|d| d.checked_add(disp))
        {
            return self.frame_mem(total);
        }
        if let Some(reg) = self
            .reg_of(base)
            .filter(|reg| !written.contains(reg) && *reg != Reg::Rsp)
        {
            return Mem::new(reg, disp);
        }
        self.raw(scratch, base);
        Mem::new(scratch, disp)
    }

    pub(super) fn lock_rmw(&mut self, op: AtomicRmwOp, size: Size, addr: Value, value: Value) {
        let ty = self.ty(value);
        let imm = self
            .const_of(value)
            .map(|c| crate::fold::extend(c, ty, false) as u64);
        let mask = ty.unsigned_max();
        let inc_dec = match (op, imm) {
            (AtomicRmwOp::Add, Some(1)) => Some(false),
            (AtomicRmwOp::Add, Some(u)) if u == mask => Some(true),
            (AtomicRmwOp::Sub, Some(1)) => Some(true),
            (AtomicRmwOp::Sub, Some(u)) if u == mask => Some(false),
            _ => None,
        };
        let bt_ext = match (op, size != Size::B, imm) {
            (AtomicRmwOp::Or, true, Some(u)) if u.count_ones() == 1 => {
                Some((5, u.trailing_zeros() as u8))
            }
            (AtomicRmwOp::And, true, Some(u)) if ((!u) & mask).count_ones() == 1 => {
                Some((6, ((!u) & mask).trailing_zeros() as u8))
            }
            (AtomicRmwOp::Xor, true, Some(u)) if u.count_ones() == 1 => {
                Some((7, u.trailing_zeros() as u8))
            }
            _ => None,
        };
        let alu = match op {
            AtomicRmwOp::Add => Alu::Add,
            AtomicRmwOp::Sub => Alu::Sub,
            AtomicRmwOp::Or => Alu::Or,
            AtomicRmwOp::And => Alu::And,
            _ => Alu::Xor,
        };
        if self.plan.elided[value.index()]
            && let Some((opcode, bit)) =
                lock_bt_shift_bit(self.func, &self.plan, |v| self.defs[v.index()], op, value)
        {
            if let Some(br) = self.reg_of(bit) {
                let target =
                    self.atomic_target(addr, if br == Reg::Rcx { Reg::Rax } else { Reg::Rcx }, &[]);
                self.asm.lock_bt_reg(size, opcode, target, br);
            } else {
                let target = self.atomic_target(addr, Reg::Rcx, &[Reg::Rax]);
                self.raw(Reg::Rax, bit);
                self.asm.lock_bt_reg(size, opcode, target, Reg::Rax);
            }
        } else if let Some(dec) = inc_dec {
            let target = self.atomic_target(addr, Reg::Rcx, &[]);
            self.asm.lock_inc_dec(size, dec, target);
        } else if let Some(imm32) = self.imm_sized(value, size) {
            let target = self.atomic_target(addr, Reg::Rcx, &[]);
            self.asm.lock_alu_imm(size, alu, target, imm32);
        } else if let Some((ext, bit)) = bt_ext {
            let target = self.atomic_target(addr, Reg::Rcx, &[]);
            self.asm.lock_bt_imm(size, ext, target, bit);
        } else if let Some(vr) = self.reg_of(value) {
            let target =
                self.atomic_target(addr, if vr == Reg::Rcx { Reg::Rax } else { Reg::Rcx }, &[]);
            self.asm.lock_alu_reg(size, alu, target, vr);
        } else {
            let target = self.atomic_target(addr, Reg::Rcx, &[Reg::Rax]);
            self.raw(Reg::Rax, value);
            self.asm.lock_alu_reg(size, alu, target, Reg::Rax);
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rcx]);
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
                let alu = if op == Binary::SaddSat {
                    Alu::Add
                } else {
                    Alu::Sub
                };
                self.asm.alu_sized(size, alu, dst, src);
                self.asm.mov_imm(Reg::Rdx, smin);
                self.asm.mov_imm(Reg::R11, smax);
                self.asm.cmov(Cond::S, Reg::Rdx, Reg::R11);
                self.asm.cmov(Cond::O, dst, Reg::Rdx);
            }
            _ => {}
        }
        self.clobber_regs(&[dst, Reg::Rdx, Reg::R11]);
    }

    pub(super) fn popcount(&mut self, dst: Reg, src: Rm) {
        let (part, mask) = (Reg::R11, Reg::Rsi);
        match src {
            Rm::Reg(r) if r == dst as u8 => {}
            _ => self.asm.load(Size::Q, dst, src),
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
        self.clobber_regs(&[dst, Reg::R11, Reg::Rsi]);
    }

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
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx]);
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
        self.clobber_regs(&[reg, Reg::Rcx]);
    }

    /// Divides or takes the remainder by a power of two into `dst` with a shift or a mask, when `op` is unsigned and `y` is one.
    pub(super) fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self
            .const_of(y)
            .map(|imm| crate::fold::extend(imm, ty, false) as u64)
            .filter(|d| d.is_power_of_two())
        else {
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
                let Ok(mask) = i32::try_from(divisor - 1) else {
                    return false;
                };
                self.raw(dst, x);
                self.asm.alu_imm(Alu::And, dst, mask);
            }
            _ => return false,
        }
        true
    }

    /// Divides `x` by `y` at 32-bit width for sub-64-bit operands and 64-bit otherwise, returning `Rdx` for remainders and `Rax` for quotients.
    pub(super) fn divide(&mut self, op: Binary, x: Value, y: Value) -> Reg {
        let ty = self.ty(x);
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let ext = Ext::of(signed);
        let narrow = ty.bytes() <= 4;
        self.get(Reg::Rax, x, ext);
        let divisor = self
            .reg_of(y)
            .filter(|&r| {
                !matches!(r, Reg::Rax | Reg::Rdx | Reg::Rsp)
                    && (ty.bytes() >= 4 || (!signed && self.plan.zero_ext[y.index()]))
            })
            .unwrap_or_else(|| {
                self.get(Reg::Rcx, y, ext);
                Reg::Rcx
            });
        self.asm.flags_overwritten();
        if signed {
            if narrow {
                self.asm.bytes(&[0x99]);
                self.asm.group3_sized(Size::D, Group3::Idiv, divisor);
            } else {
                self.asm.cqo();
                self.asm.group3(Group3::Idiv, divisor);
            }
        } else {
            self.asm.alu32(Alu::Xor, Reg::Rdx, Reg::Rdx);
            if narrow {
                self.asm.group3_sized(Size::D, Group3::Div, divisor);
            } else {
                self.asm.group3(Group3::Div, divisor);
            }
        }
        self.clobber_regs(&[Reg::Rax, Reg::Rdx]);
        if matches!(op, Binary::Srem | Binary::Urem) {
            Reg::Rdx
        } else {
            Reg::Rax
        }
    }

    /// Divides the 128-bit value `high:low` by `by` with one `div`, putting the quotient, or the remainder when `rem` is set, in `result`.
    pub(super) fn divide_wide(&mut self, rem: bool, [high, low, by]: [Value; 3], result: Value) {
        self.get(Reg::Rdx, high, Ext::Zero);
        self.get(Reg::Rax, low, Ext::Zero);
        self.get(Reg::Rcx, by, Ext::Zero);
        self.asm.group3(Group3::Div, Reg::Rcx);
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx]);
        self.flags = None;
        self.put(result, if rem { Reg::Rdx } else { Reg::Rax });
    }

    /// Computes the high half of `x * y` and returns its register; a 64-bit product leaves its low half in rax for the `Imul` that follows.
    pub(super) fn multiply_high(&mut self, signed: bool, x: Value, y: Value, bits: u8) -> Reg {
        let ext = Ext::of(signed);
        self.get(Reg::Rax, x, ext);
        if bits == 64 {
            let source = self
                .reg_of(y)
                .filter(|&reg| !matches!(reg, Reg::Rax | Reg::Rdx | Reg::Rsp))
                .unwrap_or_else(|| {
                    self.get(Reg::Rcx, y, ext);
                    Reg::Rcx
                });
            self.clobber_reg(Reg::Rax);
            self.asm
                .group3(if signed { Group3::Imul } else { Group3::Mul }, source);
            self.clobber_reg(Reg::Rdx);
            self.low_product = Some((x, y));
            return Reg::Rdx;
        }
        self.get(Reg::Rcx, y, ext);
        self.clobber_reg(Reg::Rax);
        self.asm.imul(Reg::Rax, Reg::Rcx);
        self.asm.shift_imm(
            Size::Q,
            if signed { Shift::Sar } else { Shift::Shr },
            Reg::Rax,
            bits,
        );
        Reg::Rax
    }
    pub(super) fn rmw(
        &mut self,
        op: AtomicRmwOp,
        size: Size,
        addr: Value,
        value: Value,
        dst: Option<Reg>,
    ) -> Reg {
        match op {
            AtomicRmwOp::Xchg | AtomicRmwOp::Add | AtomicRmwOp::Sub => {
                let target = self.atomic_target(addr, Reg::Rcx, &[Reg::Rax]);
                let out = dst
                    .filter(|&r| target.base != r && target.index.is_none_or(|(idx, _)| idx != r))
                    .unwrap_or(Reg::Rax);
                if op == AtomicRmwOp::Sub
                    && let Some(imm) = self.const_of(value)
                {
                    self.asm.mov_imm(out, imm.wrapping_neg());
                    self.asm.xadd(size, target, out);
                    return out;
                }
                self.raw(out, value);
                if op == AtomicRmwOp::Xchg {
                    self.asm.flags_overwritten();
                    self.asm.xchg(size, target, out);
                    return out;
                }
                if op == AtomicRmwOp::Sub {
                    self.asm.group3(Group3::Neg, out);
                }
                self.asm.xadd(size, target, out);
                out
            }
            _ => {
                self.rmw_loop(op, size, addr, value);
                Reg::Rax
            }
        }
    }

    fn rmw_loop(&mut self, op: AtomicRmwOp, size: Size, addr: Value, value: Value) {
        let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
        let imm_alu = match op {
            AtomicRmwOp::Or => self.imm_sized(value, size).map(|imm| (Alu::Or, imm)),
            AtomicRmwOp::Xor => self.imm_sized(value, size).map(|imm| (Alu::Xor, imm)),
            AtomicRmwOp::And => self.imm_sized(value, size).map(|imm| (Alu::And, imm)),
            _ => None,
        };
        let avoid: &[Reg] = if imm_alu.is_some() {
            &[Reg::Rax, Reg::Rcx]
        } else {
            &[Reg::Rax, Reg::Rcx, Reg::Rdx]
        };
        let target = self.atomic_target(addr, Reg::Rsi, avoid);
        if imm_alu.is_none() {
            self.get(Reg::Rdx, value, Ext::of(signed));
        }
        self.asm.load(size, Reg::Rax, target);
        let again = self.asm.label();
        self.asm.bind(again);
        if signed {
            self.asm.load_signed(size, Reg::Rcx, Reg::Rax);
        } else {
            self.asm.load(size, Reg::Rcx, Reg::Rax);
        }
        let choose = match op {
            AtomicRmwOp::Smin => Some(Cond::G),
            AtomicRmwOp::Smax => Some(Cond::L),
            AtomicRmwOp::Umin => Some(Cond::A),
            AtomicRmwOp::Umax => Some(Cond::B),
            _ => None,
        };
        match (choose, imm_alu, op) {
            (Some(cond), _, _) => {
                self.asm.alu(Alu::Cmp, Reg::Rcx, Reg::Rdx);
                self.asm.cmov(cond, Reg::Rcx, Reg::Rdx);
            }
            (None, Some((alu, imm)), _) => self.asm.alu_imm_sized(size, alu, Reg::Rcx, imm),
            (None, None, AtomicRmwOp::Or) => self.asm.alu(Alu::Or, Reg::Rcx, Reg::Rdx),
            (None, None, AtomicRmwOp::Xor) => self.asm.alu(Alu::Xor, Reg::Rcx, Reg::Rdx),
            (None, None, _) => {
                self.asm.alu(Alu::And, Reg::Rcx, Reg::Rdx);
                if op == AtomicRmwOp::Nand {
                    self.asm.group3(Group3::Not, Reg::Rcx);
                }
            }
        }
        self.asm.cmpxchg(size, target, Reg::Rcx);
        self.asm.jcc(Cond::Ne, again);
    }

    pub(super) fn emit_test_band(&mut self, x: Value, y: Value) -> Cond {
        let ty = self.ty(x);
        let size = Size::of(ty.bytes());
        let imm_pair = match (self.const_of(y), self.const_of(x)) {
            (Some(imm), _) => Some((x, crate::fold::extend(imm, ty, false))),
            (None, Some(imm)) => Some((y, crate::fold::extend(imm, ty, false))),
            (None, None) => None,
        };
        if let Some((val, imm)) = imm_pair {
            if imm == 1
                && matches!(ty, Type::I32 | Type::I64)
                && self.plan.elided[val.index()]
                && let Some(uinst) = self.defs[val.index()]
                && let Op::Binary(Binary::Ushr, word, k) = self.func.insts[uinst.index()].op
                && let Some(rw) = self.reg_of(word)
            {
                let rk = self.reg_of(k).unwrap_or_else(|| {
                    self.raw(Reg::Rax, k);
                    self.clobber_reg(Reg::Rax);
                    Reg::Rax
                });
                self.asm.bt_reg(size, rw, rk);
                return Cond::B;
            }
            let imm_sz = if ty != Type::I64 {
                Some((size, imm as i32))
            } else if let Ok(u) = u32::try_from(imm as u64) {
                Some((Size::D, u as i32))
            } else {
                i32::try_from(imm).ok().map(|s| (Size::Q, s))
            };
            if let Some((tsz, timm)) = imm_sz {
                let source = self.operand(val);
                self.asm.test_imm_sized(tsz, source, timm);
            } else {
                let source = if matches!(self.location(val), Location::Reg(_) | Location::Slot)
                    || self.has_elided_mem(val)
                {
                    self.operand(val)
                } else {
                    self.raw(Reg::Rax, val);
                    Reg::Rax.into()
                };
                if ty == Type::I64 && (imm as u64).is_power_of_two() {
                    self.asm
                        .bt_imm(Size::Q, 4, source, (imm as u64).trailing_zeros() as u8);
                    return Cond::B;
                } else if ty == Type::I64
                    && (imm as u64)
                        .checked_add(1)
                        .is_some_and(u64::is_power_of_two)
                {
                    let shift = (imm as u64).leading_zeros() as u8;
                    if !matches!(source, Rm::Reg(0)) {
                        self.asm.load(Size::Q, Reg::Rax, source);
                    }
                    if shift == 1 {
                        self.asm.alu_sized(Size::Q, Alu::Add, Reg::Rax, Reg::Rax);
                    } else {
                        self.asm.shift_imm(Size::Q, Shift::Shl, Reg::Rax, shift);
                    }
                    self.clobber_reg(Reg::Rax);
                } else {
                    let (source, wide) = self.spare(None, val, source);
                    self.asm.mov_imm(wide, imm);
                    self.clobber_reg(wide);
                    self.asm.test_sized(Size::Q, source, wide);
                }
            }
            return Cond::Ne;
        }
        if let Some(ry) = self.reg_of(y).or_else(|| self.cached_beside(y, x)) {
            let sx = self.operand(x);
            self.asm.test_sized(size, sx, ry);
        } else if let Some(rx) = self.reg_of(x).or_else(|| self.cached_beside(x, y)) {
            let sy = self.operand(y);
            self.asm.test_sized(size, sy, rx);
        } else {
            self.raw(Reg::Rax, x);
            let sy = self.operand(y);
            self.asm.test_sized(size, sy, Reg::Rax);
        }
        Cond::Ne
    }

    pub(super) fn emit_mem_rmw(&mut self, value: Value, to: Mem) {
        self.flags = None;
        let inst = self.defs[value.index()].expect("invariant: fuse_mem_rmw checked def");
        let Op::Binary(op, a, b) = self.func.insts[inst.index()].op else {
            return;
        };
        let other = if self.plan.elided[a.index()] { b } else { a };
        let Some(alu) = super::int::alu_of(op) else {
            return;
        };
        let size = Size::of(self.ty(value).bytes());
        if let Some(imm) = self.imm_sized(other, size) {
            self.asm.alu_imm_sized(size, alu, to, imm);
        } else {
            self.asm.alu_mem_reg(
                size,
                alu,
                to,
                self.reg_of(other)
                    .expect("invariant: fuse_mem_rmw checked reg"),
            );
        }
    }
}
