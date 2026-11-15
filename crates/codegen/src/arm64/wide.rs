use super::asm::{Alu, Bits, Cond, Reg, Size, Vreg, ZR};
use super::{A, C, D, Emit, S, T, V0, V1, WIDE_SCRATCH};
use crate::ir::{Binary, IntCC, Op, Unary, Value};

const E: Reg = WIDE_SCRATCH[0];
const F: Reg = WIDE_SCRATCH[1];

fn half_score((lo, hi): (Option<i64>, Option<i64>), zero_only: bool) -> u8 {
    let hit = |v: Option<i64>| matches!(v, Some(0)) || (!zero_only && matches!(v, Some(-1)));
    u8::from(hit(lo)) + u8::from(hit(hi))
}

impl Emit<'_> {
    fn const_val(&self, v: Value) -> Option<i64> {
        super::fuse::const_of(self.func, &self.plan, v, |x| self.defs[x.index()])
    }

    fn const_halves(&self, v: Value) -> (Option<i64>, Option<i64>) {
        if let Some(imm) = self.const_val(v) {
            return (Some(imm), Some(imm >> 63));
        }
        let Some(inst) = self.defs[v.index()] else { return (None, None) };
        match self.func.insts[inst.index()].op {
            Op::Binary(Binary::Iconcat, lo, hi) => (self.const_val(lo), self.const_val(hi)),
            _ => (None, None),
        }
    }

    fn shift_const(&self, y: Value) -> Option<u8> {
        self.const_halves(y).0.map(|imm| (imm & 127) as u8)
    }

    fn load_half(&mut self, high: bool, reg: Reg, value: Value) {
        if !high {
            self.raw(reg, value);
        } else if let Some(imm) = self.const_val(value) {
            self.asm.mov_imm(reg, imm >> 63);
        } else {
            self.high(reg, value);
        }
    }

    /// Loads the low half of `x` into `low` and the high half into `high`.
    fn halves(&mut self, x: Value, (low, high): (Reg, Reg)) {
        self.load_half(false, low, x);
        self.load_half(true, high, x);
    }

    fn put_halves(&mut self, result: Value, (low, high): (Reg, Reg)) {
        self.put(result, low);
        self.put_high(result, high);
    }

    pub(super) fn wide_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        self.halves(x, (A, D));
        match op {
            Unary::Ineg => {
                self.asm.alu(Alu::Subs, Size::X, A, ZR, A);
                self.asm.alu(Alu::Sbc, Size::X, D, ZR, D);
            }
            Unary::Bnot => {
                self.asm.alu(Alu::Orn, Size::X, A, ZR, A);
                self.asm.alu(Alu::Orn, Size::X, D, ZR, D);
            }
            Unary::Bswap | Unary::Bitrev => {
                let bits = if op == Unary::Bswap { Bits::Rev } else { Bits::Rbit };
                self.asm.bits(bits, C, D);
                self.asm.bits(bits, D, A);
                self.asm.mov(A, C);
            }
            Unary::Clz | Unary::Ctz => {
                let (first, second) = if op == Unary::Clz { (D, A) } else { (A, D) };
                if op == Unary::Ctz {
                    self.asm.bits(Bits::Rbit, A, A);
                    self.asm.bits(Bits::Rbit, D, D);
                }
                self.asm.bits(Bits::Clz, C, first);
                self.asm.bits(Bits::Clz, S, second);
                self.asm.add_imm(false, false, Size::X, S, S, 64);
                self.asm.add_imm(true, true, Size::X, ZR, first, 0);
                self.asm.csel(Cond::Eq, A, S, C);
                self.asm.mov(D, ZR);
            }
            Unary::Popcnt => {
                self.popcount(C, A);
                self.popcount(S, D);
                self.asm.alu(Alu::Add, Size::X, A, C, S);
                self.asm.mov(D, ZR);
            }
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.put_halves(result, (A, D));
        Ok(())
    }

    pub(super) fn wide_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        match op {
            Binary::Iadd | Binary::Isub => {
                let (low, high) = if op == Binary::Iadd { (Alu::Adds, Alu::Adc) } else { (Alu::Subs, Alu::Sbc) };
                self.halves(x, (A, D));
                self.halves(y, (C, S));
                self.asm.alu(low, Size::X, A, A, C);
                self.asm.alu(high, Size::X, D, D, S);
                self.put_halves(result, (A, D));
            }
            Binary::Band | Binary::Bor | Binary::Bxor => self.wide_bitwise(op, (x, y), result),
            Binary::Imul => {
                self.halves(x, (A, D));
                self.halves(y, (C, S));
                self.asm.alu(Alu::Umulh, Size::X, T, A, C);
                self.asm.madd(false, Size::X, T, (A, S), T);
                self.asm.madd(false, Size::X, T, (D, C), T);
                self.asm.madd(false, Size::X, A, (A, C), ZR);
                self.put_halves(result, (A, T));
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                let halves = if let Some(k) = self.shift_const(y) {
                    if k < 64 || op == Binary::Ishl {
                        self.load_half(false, A, x);
                    }
                    if k < 64 || op != Binary::Ishl {
                        self.load_half(true, D, x);
                    }
                    self.wide_shift_const(op, k)
                } else {
                    self.halves(x, (A, D));
                    self.raw(C, y);
                    self.wide_shift(op)
                };
                self.put_halves(result, halves);
            }
            Binary::Rotl | Binary::Rotr => {
                let halves = self.wide_rotate(op, x, y);
                self.put_halves(result, halves);
            }
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        Ok(())
    }

    fn wide_bitwise(&mut self, op: Binary, (x, y): (Value, Value), result: Value) {
        let (cx, cy) = (self.const_halves(x), self.const_halves(y));
        let ((lhs, rhs), (rhs_lo, rhs_hi)) = if half_score(cx, false) > half_score(cy, false) {
            ((y, x), cx)
        } else {
            ((x, y), cy)
        };
        let lo = self.wide_bitwise_half((op, false), (lhs, rhs, rhs_lo), (A, C));
        let hi = self.wide_bitwise_half((op, true), (lhs, rhs, rhs_hi), (D, S));
        self.put_halves(result, (lo, hi));
    }

    fn wide_bitwise_half(&mut self, (op, high): (Binary, bool), (lhs, rhs, rhs_c): (Value, Value, Option<i64>), (dst, tmp): (Reg, Reg)) -> Reg {
        match (op, rhs_c) {
            (Binary::Band, Some(0)) => return ZR,
            (Binary::Bor, Some(-1)) => {
                self.asm.mov_imm(dst, -1);
                return dst;
            }
            (Binary::Bor | Binary::Bxor, Some(0)) | (Binary::Band, Some(-1)) => {
                self.load_half(high, dst, lhs);
                return dst;
            }
            (Binary::Bxor, Some(-1)) => {
                self.load_half(high, dst, lhs);
                self.asm.alu(Alu::Orn, Size::X, dst, ZR, dst);
                return dst;
            }
            _ => {}
        }
        self.load_half(high, dst, lhs);
        self.load_half(high, tmp, rhs);
        let alu = match op {
            Binary::Band => Alu::And,
            Binary::Bor => Alu::Orr,
            _ => Alu::Eor,
        };
        self.asm.alu(alu, Size::X, dst, dst, tmp);
        dst
    }

    fn wide_rotate(&mut self, op: Binary, x: Value, y: Value) -> (Reg, Reg) {
        self.halves(x, (A, D));
        if let Some(k) = self.shift_const(y) {
            let r = if op == Binary::Rotl { (128 - k) & 127 } else { k };
            if r == 0 {
                return (A, D);
            }
            if r == 64 {
                return (D, A);
            }
            let (lo, hi, s) = if r < 64 { (A, D, r) } else { (D, A, r - 64) };
            self.asm.extr(Size::X, C, hi, lo, s);
            self.asm.extr(Size::X, S, lo, hi, s);
            return (C, S);
        }
        let (first, second) = if op == Binary::Rotl { (Binary::Ishl, Binary::Ushr) } else { (Binary::Ushr, Binary::Ishl) };
        self.raw(C, y);
        self.asm.fmov_in(true, V0, A);
        self.asm.fmov_in(true, V1, D);
        let (lo1, hi1) = self.wide_shift(first);
        self.asm.fmov_in(true, Vreg(2), lo1);
        self.asm.fmov_in(true, Vreg(3), hi1);
        self.asm.fmov_out(true, A, V0);
        self.asm.fmov_out(true, D, V1);
        self.asm.alu(Alu::Sub, Size::X, C, ZR, C);
        let (lo2, hi2) = self.wide_shift(second);
        self.asm.fmov_out(true, E, Vreg(2));
        self.asm.fmov_out(true, F, Vreg(3));
        self.asm.alu(Alu::Orr, Size::X, A, lo2, E);
        self.asm.alu(Alu::Orr, Size::X, D, hi2, F);
        (A, D)
    }

    fn wide_shift_const(&mut self, op: Binary, k: u8) -> (Reg, Reg) {
        if k == 0 {
            return (A, D);
        }
        if k >= 64 {
            let s = k - 64;
            return match op {
                Binary::Ishl => {
                    if s > 0 {
                        self.asm.lsl_imm(A, A, s);
                    }
                    (ZR, A)
                }
                Binary::Ushr => {
                    if s > 0 {
                        self.asm.lsr_imm(D, D, s);
                    }
                    (D, ZR)
                }
                _ => {
                    self.asm.asr_imm(C, D, 63);
                    if s > 0 {
                        self.asm.asr_imm(D, D, s);
                    }
                    (D, C)
                }
            };
        }
        match op {
            Binary::Ishl => {
                self.asm.extr(Size::X, D, D, A, 64 - k);
                self.asm.lsl_imm(A, A, k);
                (A, D)
            }
            _ => {
                self.asm.extr(Size::X, C, D, A, k);
                if op == Binary::Sshr {
                    self.asm.asr_imm(D, D, k);
                } else {
                    self.asm.lsr_imm(D, D, k);
                }
                (C, D)
            }
        }
    }

    /// Shifts the pair (A, D) by C modulo 128 and returns the registers that hold the result's halves.
    fn wide_shift(&mut self, op: Binary) -> (Reg, Reg) {
        self.asm.alu(Alu::Orn, Size::X, E, ZR, C);
        self.asm.lsr_imm(F, C, 6);
        self.asm.bitfield(false, F, F, (0, 0));
        let (near, far) = match op {
            Binary::Ishl => {
                self.asm.alu(Alu::Lslv, Size::X, T, D, C);
                self.asm.lsr_imm(S, A, 1);
                self.asm.alu(Alu::Lsrv, Size::X, S, S, E);
                self.asm.alu(Alu::Orr, Size::X, T, T, S);
                self.asm.alu(Alu::Lslv, Size::X, S, A, C);
                (S, T)
            }
            _ => {
                self.asm.alu(Alu::Lsrv, Size::X, T, A, C);
                self.asm.lsl_imm(S, D, 1);
                self.asm.alu(Alu::Lslv, Size::X, S, S, E);
                self.asm.alu(Alu::Orr, Size::X, T, T, S);
                self.asm.alu(if op == Binary::Sshr { Alu::Asrv } else { Alu::Lsrv }, Size::X, S, D, C);
                (T, S)
            }
        };
        let fill = match op {
            Binary::Sshr => {
                self.asm.asr_imm(E, D, 63);
                E
            }
            _ => ZR,
        };
        self.asm.add_imm(true, true, Size::X, ZR, F, 0);
        match op {
            Binary::Ishl => {
                self.asm.csel(Cond::Ne, D, near, far);
                self.asm.csel(Cond::Ne, A, fill, near);
            }
            _ => {
                self.asm.csel(Cond::Ne, A, far, near);
                self.asm.csel(Cond::Ne, D, fill, far);
            }
        }
        (A, D)
    }

    fn wide_icmp_eq(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        let (cx, cy) = (self.const_halves(x), self.const_halves(y));
        let ((lhs, rhs), (rhs_lo, rhs_hi)) = if half_score(cx, true) > half_score(cy, true) {
            ((y, x), cx)
        } else {
            ((x, y), cy)
        };
        self.halves(lhs, (A, D));
        if rhs_lo != Some(0) {
            self.load_half(false, C, rhs);
            self.asm.alu(Alu::Eor, Size::X, A, A, C);
        }
        if rhs_hi != Some(0) {
            self.load_half(true, S, rhs);
            self.asm.alu(Alu::Eor, Size::X, D, D, S);
        }
        self.asm.alu(Alu::Orr, Size::X, A, A, D);
        self.asm.add_imm(true, true, Size::X, ZR, A, 0);
        if cc == IntCC::Equal { Cond::Eq } else { Cond::Ne }
    }

    pub(super) fn wide_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            return self.wide_icmp_eq(cc, x, y);
        }
        let (swap, cond) = match cc {
            IntCC::SignedLessThan => (false, Cond::Lt),
            IntCC::SignedGreaterThanOrEqual => (false, Cond::Ge),
            IntCC::SignedGreaterThan => (true, Cond::Lt),
            IntCC::SignedLessThanOrEqual => (true, Cond::Ge),
            IntCC::UnsignedLessThan => (false, Cond::Lo),
            IntCC::UnsignedGreaterThanOrEqual => (false, Cond::Hs),
            IntCC::UnsignedGreaterThan => (true, Cond::Lo),
            _ => (true, Cond::Hs),
        };
        let (a, b) = if swap { (y, x) } else { (x, y) };
        self.halves(a, (A, D));
        self.halves(b, (C, S));
        self.asm.alu(Alu::Subs, Size::X, ZR, A, C);
        self.asm.alu(Alu::Sbcs, Size::X, ZR, D, S);
        cond
    }
}
