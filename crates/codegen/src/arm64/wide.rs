use super::asm::{Alu, Bits, Cond, Reg, Size, Vreg, ZR};
use super::int::{add_imm_parts, encode_bitmask_imm};
use super::{A, C, D, Emit, S, T, V0, V1, WIDE_SCRATCH};
use crate::ir::{Binary, IntCC, Unary, Value};

const E: Reg = WIDE_SCRATCH[0];
const F: Reg = WIDE_SCRATCH[1];

use crate::wide::{WideCtx, half_score};

impl WideCtx<Reg> for Emit<'_> {
    fn wide_ctx(
        &self,
    ) -> (
        &crate::ir::Function,
        &crate::alloc::Plan<Reg>,
        &[Option<crate::ir::Inst>],
    ) {
        (self.func, &self.plan, &self.defs)
    }
}

impl Emit<'_> {
    fn load_half(&mut self, high: bool, reg: Reg, value: Value) {
        let half = if high {
            self.known_halves(value).1
        } else {
            self.const_halves(value).0
        };
        if let Some(imm) = half {
            self.asm.mov_imm(reg, imm);
            self.clobber_reg(reg);
        } else if high {
            self.high(reg, value);
        } else {
            self.raw(reg, value);
        }
    }

    fn load_half_reg(&mut self, high: bool, scratch: Reg, value: Value) -> Reg {
        let half = if high {
            self.known_halves(value).1
        } else {
            self.const_halves(value).0
        };
        if half == Some(0) {
            ZR
        } else {
            self.load_half(high, scratch, value);
            scratch
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

    fn wide_clz_ctz(&mut self, ctz: bool, x: Value, result: Value) {
        self.halves(x, (A, D));
        let (first, second) = if ctz { (A, D) } else { (D, A) };
        if ctz {
            self.asm.bits(Bits::Rbit, A, A);
            self.asm.bits(Bits::Rbit, D, D);
        }
        self.asm.bits(Bits::Clz, C, first);
        self.asm.bits(Bits::Clz, S, second);
        self.asm.add_imm(false, false, Size::X, S, S, 64);
        self.asm.add_imm(true, true, Size::X, ZR, first, 0);
        self.asm.csel(Cond::Eq, A, S, C);
        self.put_halves(result, (A, ZR));
    }

    pub(super) fn wide_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        match op {
            Unary::Ineg if self.const_halves(x).0 == Some(0) => {
                let d = self.load_half_reg(true, D, x);
                self.asm.alu(Alu::Sub, Size::X, D, ZR, d);
                self.put_halves(result, (ZR, D));
            }
            Unary::Ineg => {
                let (a, d) = (
                    self.load_half_reg(false, A, x),
                    self.load_half_reg(true, D, x),
                );
                self.asm.alu(Alu::Subs, Size::X, A, ZR, a);
                self.asm.alu(Alu::Sbc, Size::X, D, ZR, d);
                self.put_halves(result, (A, D));
            }
            Unary::Bnot => {
                let (a, d) = (
                    self.load_half_reg(false, A, x),
                    self.load_half_reg(true, D, x),
                );
                self.asm.alu(Alu::Orn, Size::X, A, ZR, a);
                self.asm.alu(Alu::Orn, Size::X, D, ZR, d);
                self.put_halves(result, (A, D));
            }
            Unary::Bswap | Unary::Bitrev => {
                self.halves(x, (A, D));
                let bits = if op == Unary::Bswap {
                    Bits::Rev
                } else {
                    Bits::Rbit
                };
                self.asm.bits(bits, C, D);
                self.asm.bits(bits, D, A);
                self.put_halves(result, (C, D));
            }
            Unary::Clz | Unary::Ctz => self.wide_clz_ctz(op == Unary::Ctz, x, result),
            Unary::Popcnt if self.has_zero_high(x) => {
                self.load_half(false, A, x);
                self.popcount(A, A);
                self.put_halves(result, (A, ZR));
            }
            Unary::Popcnt => {
                self.halves(x, (A, D));
                self.popcount(C, A);
                self.popcount(S, D);
                self.asm.alu(Alu::Add, Size::X, A, C, S);
                self.put_halves(result, (A, ZR));
            }
            other => return Err(format!("{other:?} of a 128-bit integer")),
        }
        self.invalidate_spills();
        Ok(())
    }

    pub(super) fn wide_binary(
        &mut self,
        op: Binary,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        match op {
            Binary::Iadd | Binary::Isub => self.wide_add_sub(op == Binary::Isub, x, y, result),
            Binary::Band | Binary::Bor | Binary::Bxor => self.wide_bitwise(op, (x, y), result),
            Binary::Imul => self.wide_mul(x, y, result),
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
        self.invalidate_spills();
        Ok(())
    }

    fn add_rhs_score(&self, v: Value) -> u8 {
        let c = self.known_halves(v);
        let lo_imm = u8::from(c.0.is_some_and(|i| add_imm_parts(false, i).is_some()));
        let hi_imm = u8::from(c.1.is_some_and(|i| add_imm_parts(false, i).is_some()));
        half_score(c, false) + lo_imm + hi_imm
    }

    fn wide_add_sub_zero_lo(
        &mut self,
        sub: bool,
        lhs: Value,
        (rhs, rhs_hi): (Value, Option<i64>),
        result: Value,
    ) {
        let a = self.load_half_reg(false, A, lhs);
        let d = if rhs_hi == Some(0) {
            self.load_half_reg(true, D, lhs)
        } else if let Some((eff_sub, imm)) = rhs_hi.and_then(|hi| add_imm_parts(sub, hi)) {
            self.load_half(true, D, lhs);
            self.asm.add_imm(eff_sub, false, Size::X, D, D, imm);
            D
        } else {
            let d_lhs = self.load_half_reg(true, D, lhs);
            let s_rhs = self.load_half_reg(true, S, rhs);
            self.asm.alu(
                if sub { Alu::Sub } else { Alu::Add },
                Size::X,
                D,
                d_lhs,
                s_rhs,
            );
            D
        };
        self.put_halves(result, (a, d));
    }

    fn wide_add_sub(&mut self, sub: bool, x: Value, y: Value, result: Value) {
        let (lhs, rhs) = if !sub && self.add_rhs_score(x) > self.add_rhs_score(y) {
            (y, x)
        } else {
            (x, y)
        };
        let (rhs_lo, rhs_hi) = self.known_halves(rhs);
        if rhs_lo == Some(0) {
            self.wide_add_sub_zero_lo(sub, lhs, (rhs, rhs_hi), result);
            return;
        }
        let a_lhs = self.load_half_reg(false, A, lhs);
        let d_lhs = self.load_half_reg(true, D, lhs);
        if a_lhs != ZR
            && let Some((eff_sub, imm)) = rhs_lo.and_then(|lo| add_imm_parts(sub, lo))
        {
            self.asm.add_imm(eff_sub, true, Size::X, A, a_lhs, imm);
        } else {
            let c_rhs = self.load_half_reg(false, C, rhs);
            self.asm.alu(
                if sub { Alu::Subs } else { Alu::Adds },
                Size::X,
                A,
                a_lhs,
                c_rhs,
            );
        }
        let (high_op, s_rhs) = match rhs_hi {
            Some(0) => (if sub { Alu::Sbc } else { Alu::Adc }, ZR),
            Some(-1) => (if sub { Alu::Adc } else { Alu::Sbc }, ZR),
            _ => {
                self.load_half(true, S, rhs);
                (if sub { Alu::Sbc } else { Alu::Adc }, S)
            }
        };
        self.asm.alu(high_op, Size::X, D, d_lhs, s_rhs);
        self.put_halves(result, (A, D));
    }

    fn wide_mul(&mut self, x: Value, y: Value, result: Value) {
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        if matches!(cx, (Some(0), Some(0))) || matches!(cy, (Some(0), Some(0))) {
            self.put_halves(result, (ZR, ZR));
            return;
        }
        if cx.0 == Some(0) || cy.0 == Some(0) {
            let (hi_src, lo_src) = if cx.0 == Some(0) { (x, y) } else { (y, x) };
            self.load_half(true, D, hi_src);
            self.load_half(false, C, lo_src);
            self.asm.madd(false, Size::X, T, (D, C), ZR);
            self.put_halves(result, (ZR, T));
            return;
        }
        let (x_zext, y_zext) = (cx.1 == Some(0), cy.1 == Some(0));
        let (a_val, b_val) = if x_zext && !y_zext { (y, x) } else { (x, y) };
        self.load_half(false, A, a_val);
        self.load_half(false, C, b_val);
        self.asm.alu(Alu::Umulh, Size::X, T, A, C);
        if !x_zext || !y_zext {
            self.load_half(true, D, a_val);
            self.asm.madd(false, Size::X, T, (D, C), T);
        }
        if !x_zext && !y_zext {
            self.load_half(true, S, b_val);
            self.asm.madd(false, Size::X, T, (A, S), T);
        }
        self.asm.madd(false, Size::X, A, (A, C), ZR);
        self.put_halves(result, (A, T));
    }

    fn wide_bitwise(&mut self, op: Binary, (x, y): (Value, Value), result: Value) {
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        let lo = self.wide_bitwise_half((op, false), (x, cx.0), (y, cy.0), (A, C));
        let hi = self.wide_bitwise_half((op, true), (x, cx.1), (y, cy.1), (D, S));
        self.put_halves(result, (lo, hi));
    }

    fn bitwise_score(c: Option<i64>) -> u8 {
        match c {
            Some(0 | -1) => 2,
            Some(v) if encode_bitmask_imm(v as u64, true).is_some() => 1,
            _ => 0,
        }
    }

    fn wide_bitwise_half(
        &mut self,
        (op, high): (Binary, bool),
        (x, cx): (Value, Option<i64>),
        (y, cy): (Value, Option<i64>),
        (dst, tmp): (Reg, Reg),
    ) -> Reg {
        let ((lhs, lhs_c), (rhs, rhs_c)) = if Self::bitwise_score(cx) > Self::bitwise_score(cy) {
            ((y, cy), (x, cx))
        } else {
            ((x, cx), (y, cy))
        };
        match (op, rhs_c) {
            (Binary::Band, Some(0)) => return ZR,
            (Binary::Bor, Some(-1)) => {
                self.asm.mov_imm(dst, -1);
                return dst;
            }
            (Binary::Bor | Binary::Bxor, Some(0)) | (Binary::Band, Some(-1)) => {
                return self.load_half_reg(high, dst, lhs);
            }
            (Binary::Bxor, Some(-1)) if lhs_c == Some(-1) => return ZR,
            (Binary::Bxor, Some(-1)) => {
                self.load_half(high, dst, lhs);
                self.asm.alu(Alu::Orn, Size::X, dst, ZR, dst);
                return dst;
            }
            _ => {}
        }
        let Some(alu) = super::int::alu_of(op) else {
            return dst;
        };
        if let Some(bm) = rhs_c.and_then(|v| encode_bitmask_imm(v as u64, true)) {
            self.load_half(high, dst, lhs);
            self.asm.alu_bitmask(alu, Size::X, bm, dst, dst);
            return dst;
        }
        self.load_half(high, dst, lhs);
        self.load_half(high, tmp, rhs);
        self.asm.alu(alu, Size::X, dst, dst, tmp);
        dst
    }

    fn wide_rotate(&mut self, op: Binary, x: Value, y: Value) -> (Reg, Reg) {
        self.halves(x, (A, D));
        if let Some(k) = self.shift_const(y) {
            let r = if op == Binary::Rotl {
                (128 - k) & 127
            } else {
                k
            };
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
        let (first, second) = if op == Binary::Rotl {
            (Binary::Ishl, Binary::Ushr)
        } else {
            (Binary::Ushr, Binary::Ishl)
        };
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
                self.asm.alu(
                    if op == Binary::Sshr {
                        Alu::Asrv
                    } else {
                        Alu::Lsrv
                    },
                    Size::X,
                    S,
                    D,
                    C,
                );
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

    fn eq_all_ones_other(
        (x, cx): (Value, (Option<i64>, Option<i64>)),
        (y, cy): (Value, (Option<i64>, Option<i64>)),
    ) -> Option<Value> {
        if cx == (Some(-1), Some(-1)) {
            Some(y)
        } else if cy == (Some(-1), Some(-1)) {
            Some(x)
        } else {
            None
        }
    }

    fn cmp64(&mut self, high: bool, a: Value, (b, b_imm): (Value, Option<i64>)) {
        let a_reg = self.load_half_reg(high, A, a);
        if a_reg != ZR
            && let Some((eff_sub, imm)) = b_imm.and_then(|v| add_imm_parts(true, v))
        {
            self.asm.add_imm(eff_sub, true, Size::X, ZR, a_reg, imm);
        } else {
            let b_reg = self.load_half_reg(high, C, b);
            self.asm.alu(Alu::Subs, Size::X, ZR, a_reg, b_reg);
        }
    }

    fn xor_half_diff(
        &mut self,
        high: bool,
        (x, cx): (Value, Option<i64>),
        (y, cy): (Value, Option<i64>),
        (dst, tmp): (Reg, Reg),
    ) {
        if cx == Some(0) {
            self.load_half(high, dst, y);
        } else if cy == Some(0) {
            self.load_half(high, dst, x);
        } else if let Some((eff_sub, imm)) = cx.and_then(|v| add_imm_parts(true, v)) {
            self.load_half(high, dst, y);
            self.asm.add_imm(eff_sub, false, Size::X, dst, dst, imm);
        } else if let Some((eff_sub, imm)) = cy.and_then(|v| add_imm_parts(true, v)) {
            self.load_half(high, dst, x);
            self.asm.add_imm(eff_sub, false, Size::X, dst, dst, imm);
        } else {
            self.load_half(high, dst, x);
            self.load_half(high, tmp, y);
            self.asm.alu(Alu::Eor, Size::X, dst, dst, tmp);
        }
    }

    fn wide_icmp_eq(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        let eq = if cc == IntCC::Equal {
            Cond::Eq
        } else {
            Cond::Ne
        };
        let (cx, cy) = (self.known_halves(x), self.known_halves(y));
        if let Some(val) = Self::eq_all_ones_other((x, cx), (y, cy)) {
            let (a, d) = (
                self.load_half_reg(false, A, val),
                self.load_half_reg(true, D, val),
            );
            self.asm.alu(Alu::And, Size::X, A, a, d);
            self.asm.add_imm(false, true, Size::X, ZR, A, 1);
            return eq;
        }
        let lo_same = cx.0.is_some() && cx.0 == cy.0;
        let hi_same = cx.1.is_some() && cx.1 == cy.1;
        if lo_same && hi_same {
            self.asm.alu(Alu::Subs, Size::X, ZR, ZR, ZR);
        } else if lo_same || hi_same {
            let (high, (hx, hy)) = if hi_same {
                (false, (cx.0, cy.0))
            } else {
                (true, (cx.1, cy.1))
            };
            let (lhs, rhs, r_imm) =
                if hy.is_some_and(|v| add_imm_parts(true, v).is_some()) || hx.is_none() {
                    (x, y, hy)
                } else {
                    (y, x, hx)
                };
            self.cmp64(high, lhs, (rhs, r_imm));
        } else {
            self.xor_half_diff(false, (x, cx.0), (y, cy.0), (A, C));
            self.xor_half_diff(true, (x, cx.1), (y, cy.1), (D, S));
            self.asm.alu(Alu::Orr, Size::X, A, A, D);
            self.asm.alu(Alu::Subs, Size::X, ZR, A, ZR);
        }
        eq
    }

    pub(super) fn wide_icmp(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        let cond = self.wide_icmp_inner(cc, x, y);
        self.invalidate_spills();
        cond
    }

    fn wide_icmp_inner(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            return self.wide_icmp_eq(cc, x, y);
        }
        let (swap, base_cc) = cc.ordered();
        let (a, b) = if swap { (y, x) } else { (x, y) };
        let (ca, cb) = (self.known_halves(a), self.known_halves(b));
        if ca.1 == Some(0) && cb.1 == Some(0) {
            self.cmp64(false, a, (b, cb.0));
            return super::int::cond(base_cc.as_unsigned());
        }
        let cond = super::int::cond(base_cc);
        if cb.0 == Some(0) {
            self.cmp64(true, a, (b, cb.1));
            return cond;
        }
        self.cmp64(false, a, (b, cb.0));
        let a_hi = self.load_half_reg(true, D, a);
        let b_hi = self.load_half_reg(true, S, b);
        self.asm.alu(Alu::Sbcs, Size::X, ZR, a_hi, b_hi);
        cond
    }
}
