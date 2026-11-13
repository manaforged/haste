use super::asm::{Alu, Bits, Cond, Reg, Size, Vreg, ZR};
use super::int::alu_of;
use super::{A, C, D, Emit, S, T, V0, V1, WIDE_SCRATCH};
use crate::ir::{Binary, IntCC, Unary, Value};

const E: Reg = WIDE_SCRATCH[0];
const F: Reg = WIDE_SCRATCH[1];

impl Emit<'_> {
    /// Loads the low half of `x` into `low` and the high half into `high`.
    fn halves(&mut self, x: Value, (low, high): (Reg, Reg)) {
        self.raw(low, x);
        self.high(high, x);
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
        if let Some(alu) = alu_of(op) {
            let high = match alu {
                Alu::Add => Alu::Adc,
                Alu::Sub => Alu::Sbc,
                same => same,
            };
            let low = match alu {
                Alu::Add => Alu::Adds,
                Alu::Sub => Alu::Subs,
                same => same,
            };
            self.halves(x, (A, D));
            self.halves(y, (C, S));
            self.asm.alu(low, Size::X, A, A, C);
            self.asm.alu(high, Size::X, D, D, S);
            self.put_halves(result, (A, D));
            return Ok(());
        }
        match op {
            Binary::Imul => {
                self.halves(x, (A, D));
                self.halves(y, (C, S));
                self.asm.alu(Alu::Umulh, Size::X, T, A, C);
                self.asm.madd(false, T, (A, S), T);
                self.asm.madd(false, T, (D, C), T);
                self.asm.madd(false, A, (A, C), ZR);
                self.put_halves(result, (A, T));
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                self.halves(x, (A, D));
                self.raw(C, y);
                let halves = self.wide_shift(op);
                self.put_halves(result, halves);
            }
            Binary::Rotl | Binary::Rotr => {
                let (first, second) = if op == Binary::Rotl { (Binary::Ishl, Binary::Ushr) } else { (Binary::Ushr, Binary::Ishl) };
                self.halves(x, (A, D));
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
                self.put_halves(result, (A, D));
            }
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        Ok(())
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

    pub(super) fn wide_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            self.halves(x, (A, D));
            self.halves(y, (C, S));
            self.asm.alu(Alu::Eor, Size::X, A, A, C);
            self.asm.alu(Alu::Eor, Size::X, D, D, S);
            self.asm.alu(Alu::Orr, Size::X, A, A, D);
            self.asm.add_imm(true, true, Size::X, ZR, A, 0);
            return if cc == IntCC::Equal { Cond::Eq } else { Cond::Ne };
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
