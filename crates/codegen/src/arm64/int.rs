mod bits;
mod imm;

use super::asm::{Alu, Bits, Cond, Reg, Shift, Size, ZR};
use super::{A, C, D, Emit, IP, T, V0};
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};
pub(crate) use imm::{add_imm_parts, encode_bitmask_imm};

fn imm_within(imm: i64, bits: u32) -> bool {
    let mask = if bits >= 64 { u64::MAX } else { (1_u64 << bits) - 1 };
    imm >= 0 && (imm as u64) <= mask
}

pub(super) fn cond(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::Eq, IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::Lt, IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt, IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::Lo, IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi, IntCC::UnsignedLessThanOrEqual => Cond::Ls,
    }
}

pub(super) fn swap_cond(c: Cond) -> Cond {
    match c {
        Cond::Lt => Cond::Gt, Cond::Le => Cond::Ge, Cond::Gt => Cond::Lt, Cond::Ge => Cond::Le,
        Cond::Lo => Cond::Hi, Cond::Ls => Cond::Hs, Cond::Hi => Cond::Lo, Cond::Hs => Cond::Ls,
        other => other,
    }
}

pub(super) fn alu_of(op: Binary) -> Option<Alu> {
    match op {
        Binary::Iadd => Some(Alu::Add), Binary::Isub => Some(Alu::Sub),
        Binary::Band => Some(Alu::And), Binary::Bor => Some(Alu::Orr), Binary::Bxor => Some(Alu::Eor),
        _ => None,
    }
}

impl Emit<'_> {
    fn ushr_zext_within(&self, bits: u32, (a, amt): (Value, Value), expect_bits: u32, d: u8) -> bool {
        if bits <= 32 && bits <= expect_bits { return true; }
        if bits >= 32 && let Some(s) = self.plan.const_of(amt) && bits - ((s as u32) & (bits - 1)) <= expect_bits { return true; }
        self.is_zext_within(a, expect_bits, d)
    }

    fn ishl_zext_within(&self, bits: u32, (a, amt): (Value, Value), expect_bits: u32, d: u8) -> bool {
        let Some(s) = self.plan.const_of(amt) else { return bits == 32 && expect_bits >= 32; };
        let shift = (s as u32) & (bits - 1);
        expect_bits > shift && self.is_zext_within(a, expect_bits - shift, d)
    }

    fn bin_zext_within(&self, op: Binary, (a, b): (Value, Value), (bits, expect_bits, d): (u32, u32, u8)) -> bool {
        let w32 = bits == 32 && expect_bits >= 32;
        match op {
            Binary::Ushr => self.ushr_zext_within(bits, (a, b), expect_bits, d),
            Binary::Ishl => self.ishl_zext_within(bits, (a, b), expect_bits, d),
            Binary::Umin | Binary::Umax | Binary::Udiv | Binary::Urem | Binary::Umulhi | Binary::UaddSat | Binary::UsubSat | Binary::AvgRound => bits <= 32 && bits <= expect_bits,
            Binary::Iadd | Binary::Isub | Binary::Imul | Binary::Rotl | Binary::Rotr => w32,
            Binary::Bor | Binary::Bxor => w32 || (self.is_zext_within(a, expect_bits, d) && self.is_zext_within(b, expect_bits, d)),
            Binary::Band => w32 || self.is_zext_within(a, expect_bits, d) || self.is_zext_within(b, expect_bits, d),
            _ => false,
        }
    }

    fn unary_zext_within(&self, op: Unary, a: Value, (bits, expect_bits, d): (u32, u32, u8)) -> bool {
        match op {
            Unary::Clz | Unary::Ctz | Unary::Popcnt => expect_bits >= 7,
            Unary::Bswap if bits == 16 && expect_bits >= 16 => self.is_zext_within(a, 16, d),
            Unary::Bswap | Unary::Ineg | Unary::Bnot => bits == 32 && expect_bits >= 32,
            _ => false,
        }
    }

    fn op_zext_within(&self, op: &Op, ty: Type, expect_bits: u32, d: u8) -> bool {
        let bits = ty.bits();
        match *op {
            Op::Const(imm) => imm_within(imm, expect_bits),
            Op::Load(..) | Op::StackLoad(..) | Op::AtomicLoad(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) => bits <= expect_bits,
            Op::Icmp(..) | Op::Fcmp(..) | Op::Fits(..) => expect_bits >= 1,
            Op::Extract(..) => ty.is_small_int() && bits <= expect_bits,
            Op::Unary(u, a) => self.unary_zext_within(u, a, (bits, expect_bits, d)),
            Op::Binary(b, x, y) => self.bin_zext_within(b, (x, y), (bits, expect_bits, d)),
            Op::Select(_, a, b) => self.is_zext_within(a, expect_bits, d) && self.is_zext_within(b, expect_bits, d),
            Op::Convert(Convert::Uextend, inner) => self.ty(inner).bits() <= expect_bits || self.is_zext_within(inner, expect_bits, d),
            Op::Convert(Convert::Ireduce, inner) => self.is_zext_within(inner, expect_bits.min(bits), d),
            _ => false,
        }
    }

    fn is_zext_within(&self, val: Value, expect_bits: u32, depth: u8) -> bool {
        if matches!(self.plan.location(val), Location::Float(_)) { return false; }
        let ty = self.ty(val);
        if self.plan.zero_ext[val.index()] && ty.bits() <= expect_bits { return true; }
        if let Location::Imm(imm) = self.plan.location(val) { return imm_within(imm, expect_bits); }
        let Some(inst) = (depth > 0).then(|| self.defs[val.index()]).flatten() else { return false };
        self.op_zext_within(&self.func.insts[inst.index()].op, ty, expect_bits, depth - 1)
    }

    pub(super) fn is_zext_reg(&self, val: Value) -> bool {
        !self.plan.elided[val.index()] && matches!(self.plan.location(val), Location::Reg(_) | Location::Slot) && (self.plan.zero_ext[val.index()] || (self.defs[val.index()].is_some() && self.is_zext_within(val, self.ty(val).bits(), 4)))
    }

    pub(super) fn zext_src(&mut self, val: Value, scratch: Reg) -> Reg {
        if self.is_zext_reg(val) { self.bits(val, scratch) } else { self.src(val, scratch, Ext::Zero) }
    }

    pub(super) fn zext_pair(&mut self, x: Value, y: Value) -> (Reg, Reg) {
        match (self.is_zext_reg(x), self.is_zext_reg(y)) {
            (true, true) => self.bits_pair(x, y),
            (true, false) => (self.bits(x, A), self.src(y, C, Ext::Zero)),
            (false, true) => (self.src(x, A, Ext::Zero), self.bits(y, C)),
            (false, false) => self.src_pair(x, y, Ext::Zero),
        }
    }

    pub(super) fn zext_get(&mut self, dst: Reg, val: Value) {
        if self.is_zext_reg(val) { self.raw(dst, val); } else { self.get(dst, val, Ext::Zero); }
    }

    pub(super) fn get_opt(&mut self, dst: Reg, val: Value, signed: bool) {
        if signed { self.get(dst, val, Ext::Sign); } else { self.zext_get(dst, val); }
    }

    pub(super) fn src_opt(&mut self, val: Value, scratch: Reg, signed: bool) -> Reg {
        if signed { self.src(val, scratch, Ext::Sign) } else { self.zext_src(val, scratch) }
    }

    pub(super) fn pair_opt(&mut self, x: Value, y: Value, signed: bool) -> (Reg, Reg) {
        if signed { self.src_pair(x, y, Ext::Sign) } else { self.zext_pair(x, y) }
    }

    pub(super) fn tst_small_zero(&mut self, val: Value) -> bool {
        let bits = self.ty(val).bits();
        if bits >= 32 || self.is_zext_reg(val) || self.plan.elided[val.index()] { return false; }
        let Location::Reg(reg) = self.plan.location(val) else { return false };
        let Some(mask) = encode_bitmask_imm((1_u64 << bits) - 1, false) else { return false };
        self.asm.alu_bitmask(Alu::Ands, Size::W, mask, ZR, reg);
        true
    }

    pub(super) fn unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if ty.is_vector() { self.invalidate_spills(); return self.vec_unary(op, x, result); }
        if ty.is_float() { self.invalidate_spills(); return self.float_unary(op, x, result); }
        if ty == Type::I128 { self.invalidate_spills(); return self.wide_unary(op, x, result); }
        let dst = self.dst(result);
        self.int_unary(op, x, ty, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn bits32(&mut self, op: Bits, dst: Reg, src: Reg) {
        self.asm.emit_gp(dst, (op as u32 & !(1 << 31)) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    fn emit_shifted(&mut self, alu: Alu, size: Size, dst: Reg, lhs: Reg, (rhs, shift, imm): (Reg, Shift, u8)) {
        if imm == 0 { self.asm.alu(alu, size, dst, lhs, rhs); } else { self.asm.alu_shifted(alu, size, shift, imm, dst, lhs, rhs); }
    }

    fn int_unary(&mut self, op: Unary, x: Value, ty: Type, dst: Reg) -> Result<(), String> {
        let bits = ty.bits() as u8;
        let size = if bits <= 32 { Size::W } else { Size::X };
        match op {
            Unary::Ineg => {
                if matches!(bits, 32 | 64) && let Some((cc, a, b)) = self.cinc_operand(x) {
                    let cond = self.compare(cc, a, b);
                    self.asm.csinv(size, cond.invert(), dst, ZR, ZR);
                } else if matches!(bits, 32 | 64) && let Some((b, shift, imm)) = self.shifted_operand(Alu::Sub, ty, x) {
                    let right = self.bits(b, C);
                    self.emit_shifted(Alu::Sub, size, dst, ZR, (right, shift, imm));
                } else if matches!(bits, 32 | 64) && let Some((a, b)) = super::fuse::mneg_parts(self.func, &self.plan, ty, x, |v| self.defs[v.index()]) {
                    let (left, right) = self.bits_pair(a, b);
                    self.asm.madd(true, size, dst, (left, right), ZR);
                } else {
                    let reg = self.bits(x, A);
                    self.asm.alu(Alu::Sub, size, dst, ZR, reg);
                }
            }
            Unary::Bnot => {
                if let Some((a, b)) = super::fuse::eon_inner(self.func, &self.plan, x, |v| self.defs[v.index()]) {
                    let shifted = self.shifted_operand(Alu::Eon, ty, b).map(|s| (a, s)).or_else(|| self.shifted_operand(Alu::Eon, ty, a).map(|s| (b, s)));
                    if let Some((other, (rhs, shift, imm))) = shifted {
                        let (left, right) = self.bits_pair(self.peel_uextend(other), rhs);
                        self.emit_shifted(Alu::Eon, size, dst, left, (right, shift, imm));
                    } else {
                        let (left, right) = self.bits_pair(self.peel_uextend(a), self.peel_uextend(b));
                        self.asm.alu(Alu::Eon, size, dst, left, right);
                    }
                } else if let Some((b, shift, imm)) = self.shifted_operand(Alu::Orn, ty, x) {
                    let right = self.bits(b, C);
                    self.emit_shifted(Alu::Orn, size, dst, ZR, (right, shift, imm));
                } else {
                    let reg = self.bits(x, A);
                    self.asm.alu(Alu::Orn, size, dst, ZR, reg);
                }
            }
            Unary::Bswap => {
                let reg = self.bits(x, A);
                let op = match bits { 16 => Bits::Rev16, 32 => Bits::Rev32, 64 => Bits::Rev, _ => return Ok(self.asm.mov(dst, reg)) };
                self.asm.bits(op, dst, reg);
            }
            Unary::Bitrev => {
                let reg = self.bits(x, A);
                if bits == 32 { self.bits32(Bits::Rbit, dst, reg); } else { self.asm.bits(Bits::Rbit, dst, reg); if bits < 64 { self.asm.lsr_imm(dst, dst, 64 - bits); } }
            }
            Unary::Clz => if bits == 32 {
                let reg = self.bits(x, A);
                self.bits32(Bits::Clz, dst, reg);
            } else {
                let reg = self.zext_src(x, A);
                self.asm.bits(Bits::Clz, dst, reg);
                if bits < 64 { self.asm.add_imm(true, false, Size::X, dst, dst, u32::from(64 - bits)); }
            },
            Unary::Ctz => {
                let reg = self.bits(x, A);
                if bits == 32 {
                    self.bits32(Bits::Rbit, dst, reg);
                    self.bits32(Bits::Clz, dst, dst);
                } else if bits < 32 && let Some(mask) = encode_bitmask_imm(1_u64 << bits, false) {
                    self.asm.alu_bitmask(Alu::Orr, Size::W, mask, A, reg);
                    self.clobber_reg(A);
                    self.bits32(Bits::Rbit, dst, A);
                    self.bits32(Bits::Clz, dst, dst);
                } else {
                    let source = if bits < 64 { self.asm.mov_imm(C, 1 << bits); self.asm.alu(Alu::Orr, Size::X, A, reg, C); self.clobber_regs(&[A, C]); A } else { reg };
                    self.asm.bits(Bits::Rbit, dst, source);
                    self.asm.bits(Bits::Clz, dst, dst);
                }
            }
            Unary::Popcnt => { let reg = self.zext_src(x, A); self.popcount(dst, reg); self.clobber_vreg(V0); }
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    pub(super) fn popcount(&mut self, dst: Reg, src: Reg) {
        self.asm.fmov_to_float(V0, src);
        self.asm.popcount_bytes(V0, V0);
        self.asm.fmov_to_int(dst, V0);
    }

    pub(super) fn binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if self.plan.is_static(result) || self.plan.deferred[result.index()] { return Ok(()); }
        if op == Binary::Iconcat {
            self.raw(A, x); self.put(result, A); self.raw(A, y); self.put_high(result, A);
            return Ok(());
        }
        if ty.is_vector() { self.invalidate_spills(); return self.vec_binary(op, x, y, result); }
        if ty.is_float() { self.invalidate_spills(); return self.float_binary(op, x, y, result); }
        if ty == Type::I128 { self.invalidate_spills(); return self.wide_binary(op, x, y, result); }
        if matches!(op, Binary::Band | Binary::Bor) && self.try_emit_ccmp(op, (x, y), result) { return Ok(()); }
        let dst = self.dst(result);
        self.int_binary(op, (x, y), ty, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn add_imm_args(&self, sub: bool, value: Value, ty: Type) -> Option<(bool, u32)> {
        add_imm_parts(sub, crate::fold::extend(self.plan.const_of(value)?, ty, true))
    }

    fn add_imm_split(&self, sub: bool, value: Value, ty: Type) -> Option<(bool, u32, u32)> {
        let raw = self.plan.const_of(value)?;
        if super::asm::Asm::mov_imm_is_single(raw) { return None; }
        let simm = crate::fold::extend(raw, ty, true);
        let abs = simm.unsigned_abs();
        let (hi, lo) = ((abs & 0xFFF_000) as u32, (abs & 0xFFF) as u32);
        (abs < (1 << 24) && hi != 0 && lo != 0).then(|| (if simm < 0 { !sub } else { sub }, hi, lo))
    }

    fn emit_add_sub_imm(&mut self, sub: bool, (x, y): (Value, Value), (ty, size): (Type, Size), dst: Reg) -> bool {
        if let Some((eff_sub, imm)) = self.add_imm_args(sub, y, ty) {
            let left = self.bits(x, A); self.asm.add_imm(eff_sub, false, size, dst, left, imm); return true;
        }
        if !sub && let Some((eff_sub, imm)) = self.add_imm_args(false, x, ty) {
            let left = self.bits(y, A); self.asm.add_imm(eff_sub, false, size, dst, left, imm); return true;
        }
        if dst == ZR { return false; }
        let split = self.add_imm_split(sub, y, ty).map(|p| (x, p)).or_else(|| (!sub).then(|| self.add_imm_split(false, x, ty).map(|p| (y, p))).flatten());
        let Some((other, (eff_sub, hi, lo))) = split else { return false };
        let left = self.bits(other, A);
        if left == ZR { return false; }
        self.asm.add_imm(eff_sub, false, size, dst, left, hi);
        self.asm.add_imm(eff_sub, false, size, dst, dst, lo);
        true
    }

    fn peel_uextend(&self, val: Value) -> Value {
        if self.plan.elided[val.index()] && let Some(inst) = self.defs[val.index()] && let Op::Convert(Convert::Uextend, inner) = self.func.insts[inst.index()].op { return inner; }
        val
    }

    fn shifted_operand(&self, alu: Alu, ty: Type, val: Value) -> Option<(Value, Shift, u8)> {
        if !self.plan.elided[val.index()] { return None; }
        let (op, b, imm) = super::fuse::shift_parts(&self.func.insts[self.defs[val.index()]?.index()].op, &self.plan)?;
        let (shift, count) = super::fuse::alu_shift(matches!(alu, Alu::And | Alu::Orr | Alu::Eor | Alu::Orn | Alu::Eon), ty, op, imm)?;
        Some((self.peel_uextend(b), shift, count))
    }

    fn cinc_operand(&self, val: Value) -> Option<(IntCC, Value, Value)> {
        if !self.plan.elided[val.index()] { return None; }
        match self.func.insts[self.defs[val.index()]?.index()].op {
            Op::Icmp(cc, a, b) => Some((cc, a, b)),
            Op::Convert(Convert::Uextend, inner) if self.plan.elided[inner.index()] => match self.func.insts[self.defs[inner.index()]?.index()].op {
                Op::Icmp(cc, a, b) => Some((cc, a, b)),
                _ => None,
            },
            _ => None,
        }
    }

    fn const_imm(&self, val: Value) -> Option<i64> {
        super::fuse::const_of(self.func, &self.plan, val, |v| self.defs[v.index()])
    }

    fn emit_bit_insert(&mut self, ty: Type, dst: Reg, (a, b, lsb, width): (Value, Value, u8, u8)) {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let reg_bits: u8 = if size == Size::X { 64 } else { 32 };
        let b_src = self.peel_uextend(b);
        let rhs = if self.plan.reg_of(a) == Some(dst) { self.bits(b_src, C) } else if self.plan.reg_of(b_src) == Some(dst) {
            self.asm.mov(C, dst); self.clobber_reg(C); self.raw(dst, a); C
        } else { self.raw(dst, a); self.bits(b_src, C) };
        let base = if size == Size::X { 0xB340_0000 } else { 0x3300_0000 };
        self.asm.emit_gp(dst, base | u32::from((reg_bits - lsb) % reg_bits) << 16 | u32::from(width - 1) << 10 | u32::from(rhs.0) << 5 | u32::from(dst.0));
    }

    fn int_binary(&mut self, op: Binary, (x, y): (Value, Value), ty: Type, dst: Reg) -> Result<(), String> {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        if matches!(op, Binary::Band | Binary::Ishl | Binary::Ushr | Binary::Sshr)
            && let Some(bfx) = super::fuse::bfx_parts(self.func, &self.plan, ty, op, (x, y), &|v| self.defs[v.index()])
        { self.emit_bfm(bfx, ty, dst); return Ok(()); }
        if let Some(alu) = alu_of(op) {
            if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
                && self.plan.elided[x.index()] && self.plan.elided[y.index()]
                && let Some((hi, lo, kr)) = super::fuse::extr_parts(self.func, &self.plan, ty, (x, y), &|v| self.defs[v.index()])
            { let (l, r) = self.bits_pair(self.peel_uextend(hi), self.peel_uextend(lo)); self.asm.extr(size, dst, l, r, kr); return Ok(()); }
            if matches!(op, Binary::Bor | Binary::Bxor | Binary::Iadd)
                && let Some(bfi) = super::fuse::bfi_parts(self.func, &self.plan, ty, (x, y), &|v| self.defs[v.index()])
            { self.emit_bit_insert(ty, dst, bfi); return Ok(()); }
            if alu == Alu::Add && matches!(ty, Type::I32 | Type::I64)
                && let Some((other, (cc, a, b))) = self.cinc_operand(y).map(|cmp| (x, cmp)).or_else(|| self.cinc_operand(x).map(|cmp| (y, cmp)))
            { let cond = self.compare(cc, a, b); let reg = self.bits(other, A); self.asm.csinc(size, cond.invert(), dst, reg, reg); return Ok(()); }
            if alu == Alu::Sub && matches!(ty, Type::I32 | Type::I64) && self.const_imm(x) == Some(0)
                && let Some((cc, a, b)) = self.cinc_operand(y)
            { let cond = self.compare(cc, a, b); self.asm.csinv(size, cond.invert(), dst, ZR, ZR); return Ok(()); }
            if matches!(alu, Alu::Add | Alu::Sub) && matches!(ty, Type::I32 | Type::I64) {
                let ext = super::fuse::ext_parts(self.func, &self.plan, ty, y, |v| self.defs[v.index()]).map(|e| (x, e))
                    .or_else(|| (alu == Alu::Add).then(|| super::fuse::ext_parts(self.func, &self.plan, ty, x, |v| self.defs[v.index()]).map(|e| (y, e))).flatten());
                if let Some((other, (ext_op, shift, inner))) = ext {
                    let (left, right) = self.bits_pair(other, inner); self.asm.alu_extended(alu, size, ext_op, shift, dst, left, right); return Ok(());
                }
            }
            if let Some((not_alu, lhs, rhs, shifted)) = super::fuse::bnot_parts(self.func, &self.plan, ty, op, x, y, |v| self.defs[v.index()]) {
                let (left, right) = self.bits_pair(self.peel_uextend(lhs), self.peel_uextend(rhs));
                if let Some((shift, imm)) = shifted && imm != 0 { self.asm.alu_shifted(not_alu, size, shift, imm, dst, left, right); } else { self.asm.alu(not_alu, size, dst, left, right); }
                return Ok(());
            }
            let shifted = self.shifted_operand(alu, ty, y).map(|s| (x, s)).or_else(|| (alu != Alu::Sub).then(|| self.shifted_operand(alu, ty, x).map(|s| (y, s))).flatten());
            if let Some((other, (b, shift, imm))) = shifted {
                let (left, right) = if alu == Alu::Sub && self.const_imm(other) == Some(0) { (ZR, self.bits(b, C)) } else { self.bits_pair(self.peel_uextend(other), b) };
                self.emit_shifted(alu, size, dst, left, (right, shift, imm));
                return Ok(());
            }
            if matches!(alu, Alu::Add | Alu::Sub) && matches!(ty, Type::I32 | Type::I64)
                && let Some((sub, a, b, c)) = super::fuse::madd_parts(self.func, &self.plan, ty, op, x, y, |v| self.defs[v.index()])
            {
                let (left, right) = self.bits_pair(a, b);
                let acc = if self.plan.const_of(c) == Some(0) { ZR } else { let r = self.bits(c, IP); if r == IP { self.clobber_reg(IP); } r };
                self.asm.madd(sub, size, dst, (left, right), acc);
                return Ok(());
            }
            if matches!(alu, Alu::And | Alu::Orr | Alu::Eor) && dst != ZR {
                let imm_pair = match (self.const_imm(y), self.const_imm(x)) { (Some(imm), _) => Some((x, imm)), (None, Some(imm)) => Some((y, imm)), _ => None };
                if let Some((val, imm)) = imm_pair && let Some(bitmask) = encode_bitmask_imm(imm as u64, size == Size::X) {
                    let left = self.bits(val, A); self.asm.alu_bitmask(alu, size, bitmask, dst, left); return Ok(());
                }
            }
            if matches!(alu, Alu::Add | Alu::Sub) && self.emit_add_sub_imm(alu == Alu::Sub, (x, y), (ty, size), dst) { return Ok(()); }
            if alu == Alu::Sub && self.const_imm(x) == Some(0) { let right = self.bits(y, C); self.asm.alu(alu, size, dst, ZR, right); return Ok(()); }
            let (left, right) = self.bits_pair(x, y);
            self.asm.alu(alu, size, dst, left, right);
            return Ok(());
        }
        let bits = ty.bits() as u8;
        match op {
            Binary::Imul => self.multiply(dst, x, y),
            Binary::Udiv | Binary::Urem if self.divide_by_power(op, (x, y), dst) => {}
            Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem => self.divide(op, (x, y), dst),
            Binary::Umulhi | Binary::Smulhi => self.multiply_high(op == Binary::Smulhi, (x, y), bits, dst),
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                let signed = matches!(op, Binary::Smin | Binary::Smax);
                let left = self.src_opt(x, A, signed);
                let cond = match op { Binary::Smin => Cond::Lt, Binary::Smax => Cond::Gt, Binary::Umin => Cond::Lo, _ => Cond::Hi };
                let right = if self.plan.const_of(y) == Some(0) {
                    self.asm.add_imm(true, true, Size::X, ZR, left, 0);
                    ZR
                } else {
                    let r = self.src_opt(y, C, signed);
                    self.asm.alu(Alu::Subs, Size::X, ZR, left, r);
                    r
                };
                self.asm.csel(cond, dst, left, right);
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => self.sat_add_sub(op, (x, y), bits, dst),
            Binary::AvgRound => {
                let (left, right) = self.zext_pair(x, y);
                self.asm.alu(Alu::Orr, Size::X, D, left, right);
                self.asm.alu(Alu::Eor, Size::X, A, left, right);
                self.asm.alu_shifted(Alu::Sub, Size::X, Shift::Lsr, 1, dst, D, A);
                self.clobber_regs(&[A, D]);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => self.shift_by(op, (x, y), bits, dst),
            Binary::Rotl | Binary::Rotr => self.rotate(op == Binary::Rotl, (x, y), bits, dst),
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    fn clamp_range(&mut self, val: Reg, dst: Reg, lo: i64, hi: i64) {
        let low_src = if lo == 0 { ZR } else { self.asm.mov_imm(C, lo); C };
        self.asm.alu(Alu::Subs, Size::X, ZR, val, low_src);
        self.asm.csel(Cond::Lt, D, low_src, val);
        self.asm.mov_imm(C, hi);
        self.asm.alu(Alu::Subs, Size::X, ZR, D, C);
        self.asm.csel(Cond::Gt, dst, C, D);
        self.clobber_regs(&[C, D]);
    }

    fn sat_add_sub(&mut self, op: Binary, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let signed = matches!(op, Binary::SaddSat | Binary::SsubSat);
        let sub = matches!(op, Binary::SsubSat | Binary::UsubSat);
        if sub && !signed {
            let (left, right) = self.zext_pair(x, y);
            self.asm.alu(Alu::Subs, Size::X, D, left, right);
            self.asm.csel(Cond::Hs, dst, D, ZR);
            self.clobber_reg(D);
            return;
        }
        if bits < 64 {
            let (left, right) = self.pair_opt(x, y, signed);
            self.asm.alu(if sub { Alu::Sub } else { Alu::Add }, Size::X, D, left, right);
            if signed {
                let bound = 1i64 << (bits - 1);
                self.clamp_range(D, dst, -bound, bound - 1);
            } else {
                self.asm.mov_imm(C, (1i64 << bits) - 1);
                self.asm.alu(Alu::Subs, Size::X, ZR, D, C);
                self.asm.csel(Cond::Lo, dst, D, C);
                self.clobber_regs(&[C, D]);
            }
            return;
        }
        let (left, right) = self.bits_pair(x, y);
        self.asm.alu(if sub { Alu::Subs } else { Alu::Adds }, Size::X, D, left, right);
        if signed {
            self.asm.mov_imm(A, i64::MIN);
            self.asm.alu_shifted(Alu::Eor, Size::X, Shift::Asr, 63, C, A, D);
            self.asm.csel(Cond::Vs, dst, C, D);
        } else {
            self.asm.mov_imm(A, -1);
            self.asm.csel(Cond::Hs, dst, A, D);
        }
        self.clobber_regs(&[A, C, D]);
    }

    fn emit_mul_one(&mut self, size: Size, dst: Reg, val: Value, reg: Reg) {
        if size == Size::W && !self.is_zext_reg(val) { self.asm.extend(false, 4, dst, reg); } else { self.asm.mov(dst, reg); }
    }

    fn multiply(&mut self, dst: Reg, x: Value, y: Value) {
        let ty = self.ty(x);
        let size = if ty.bits() <= 32 { Size::W } else { Size::X };
        let lsl = |asm: &mut super::asm::Asm, d, s, k| {
            if size == Size::W { asm.shift32_imm(true, d, s, k) } else { asm.lsl_imm(d, s, k) }
        };
        let (value, raw_imm) = match (self.plan.const_of(x), self.plan.const_of(y)) {
            (_, Some(imm)) => (x, Some(imm)),
            (Some(imm), None) => (y, Some(imm)),
            (None, None) => (x, None),
        };
        if let Some(raw) = raw_imm {
            let uimm = crate::fold::extend(raw, ty, false) as u64;
            let simm = crate::fold::extend(raw, ty, true);
            if uimm == 0 { return self.asm.mov_imm(dst, 0); }
            let reg = self.bits(value, A);
            if uimm == 1 { return self.emit_mul_one(size, dst, value, reg); }
            if simm == -1 { return self.asm.alu(Alu::Sub, size, dst, ZR, reg); }
            if uimm.is_power_of_two() { return lsl(&mut self.asm, dst, reg, uimm.trailing_zeros() as u8); }
            if (uimm - 1).is_power_of_two() {
                return self.asm.alu_shifted(Alu::Add, size, Shift::Lsl, (uimm - 1).trailing_zeros() as u8, dst, reg, reg);
            }
            let max_shift = if size == Size::W { 31 } else { 63 };
            if let Some(next) = uimm.checked_add(1).filter(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift) {
                lsl(&mut self.asm, C, reg, next.trailing_zeros() as u8);
                self.clobber_reg(C);
                return self.asm.alu(Alu::Sub, size, dst, C, reg);
            }
            if let Some(next) = simm.checked_neg().filter(|&n| n > 2).and_then(|n| (n as u64).checked_add(1))
                .filter(|n| n.is_power_of_two() && n.trailing_zeros() as u8 <= max_shift)
            {
                return self.asm.alu_shifted(Alu::Sub, size, Shift::Lsl, next.trailing_zeros() as u8, dst, reg, reg);
            }
            let shift = uimm.trailing_zeros() as u8;
            let odd = uimm >> shift;
            if shift > 0 && matches!(odd, 3 | 5 | 9) {
                self.asm.alu_shifted(Alu::Add, size, Shift::Lsl, (odd - 1).trailing_zeros() as u8, dst, reg, reg);
                return lsl(&mut self.asm, dst, dst, shift);
            }
        }
        let (left, right) = self.bits_pair(x, y);
        self.asm.madd(false, size, dst, (left, right), ZR);
    }

    fn divide_by_power(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) -> bool {
        let ty = self.ty(y);
        let Some(divisor) = self.plan.const_of(y).map(|imm| crate::fold::extend(imm, ty, false)).filter(|imm| *imm > 0 && imm.count_ones() == 1) else { return false };
        if divisor == 1 && op == Binary::Urem { self.asm.mov_imm(dst, 0); return true; }
        if op == Binary::Udiv {
            let left = self.zext_src(x, A);
            if divisor == 1 { self.asm.mov(dst, left); } else { self.asm.lsr_imm(dst, left, divisor.trailing_zeros() as u8); }
            return true;
        }
        let left = self.bits(x, A);
        self.asm.mov_imm(C, divisor - 1);
        self.asm.alu(Alu::And, Size::X, dst, left, C);
        self.clobber_reg(C);
        true
    }

    fn divide(&mut self, op: Binary, (x, y): (Value, Value), dst: Reg) {
        let signed = matches!(op, Binary::Sdiv | Binary::Srem);
        let (left, right) = self.pair_opt(x, y, signed);
        let div = if signed { Alu::Sdiv } else { Alu::Udiv };
        if matches!(op, Binary::Sdiv | Binary::Udiv) { return self.asm.alu(div, Size::X, dst, left, right); }
        let size = if self.ty(x).bits() <= 32 { Size::W } else { Size::X };
        self.asm.alu(div, Size::X, IP, left, right);
        self.asm.madd(true, size, dst, (IP, right), left);
        self.clobber_reg(IP);
    }

    fn multiply_high(&mut self, signed: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let (left, right) = self.pair_opt(x, y, signed);
        if bits == 64 { return self.asm.alu(if signed { Alu::Smulh } else { Alu::Umulh }, Size::X, dst, left, right); }
        self.asm.madd(false, Size::X, dst, (left, right), ZR);
        if signed { self.asm.asr_imm(dst, dst, bits); } else { self.asm.lsr_imm(dst, dst, bits); }
    }

    fn unwrap_shift_mask(&self, y: Value, bits: u8) -> Value {
        if bits < 32 || !self.plan.elided[y.index()] { return y; }
        let Some(Op::Binary(Binary::Band, a, b)) = self.defs[y.index()].map(|i| self.func.insts[i.index()].op.clone()) else { return y };
        let mask = u64::from(bits - 1);
        [(a, b), (b, a)].into_iter().find_map(|(raw, m)| self.const_imm(m).filter(|&i| (i as u64 & mask) == mask).map(|_| raw)).unwrap_or(y)
    }

    fn shift_src(&mut self, op: Binary, x: Value) -> Reg {
        match op { Binary::Ishl => self.bits(x, A), Binary::Ushr => self.zext_src(x, A), _ => self.src(x, A, Ext::Sign) }
    }

    fn shift_by(&mut self, op: Binary, (x, y): (Value, Value), bits: u8, dst: Reg) {
        let alu = match op { Binary::Ishl => Alu::Lslv, Binary::Ushr => Alu::Lsrv, _ => Alu::Asrv };
        if let Some(count) = self.plan.const_of(y) {
            let count = (count as u8) & (bits - 1);
            if bits == 32 && op == Binary::Ushr && count != 0 { let value = self.bits(x, A); return self.asm.shift32_imm(false, dst, value, count); }
            let value = self.shift_src(op, x);
            if count == 0 { self.asm.mov(dst, value); return; }
            match alu { Alu::Lslv => self.asm.lsl_imm(dst, value, count), Alu::Lsrv => self.asm.lsr_imm(dst, value, count), _ => self.asm.asr_imm(dst, value, count) }
            return;
        }
        let y = self.unwrap_shift_mask(y, bits);
        if bits >= 32 {
            let (value, count) = self.bits_pair(x, y);
            return self.asm.alu(alu, if bits == 32 { Size::W } else { Size::X }, dst, value, count);
        }
        let (value, count) = (self.shift_src(op, x), self.bits(y, C));
        self.asm.mov_imm(D, i64::from(bits - 1));
        self.asm.alu(Alu::And, Size::X, C, count, D);
        self.asm.alu(alu, Size::X, dst, value, C);
        self.clobber_regs(&[C, D]);
    }

    fn rotate(&mut self, left: bool, (x, y): (Value, Value), bits: u8, dst: Reg) {
        if bits >= 32 && let Some(c) = self.plan.const_of(y) {
            let mask = u32::from(bits - 1);
            let c = (c as u32) & mask;
            let imm = if left { (u32::from(bits) - c) & mask } else { c } as u8;
            let value = self.bits(x, A);
            let size = Size::of(u32::from(bits) / 8);
            if imm == 0 { self.emit_mul_one(size, dst, x, value); } else { self.asm.ror_imm(size, dst, value, imm); }
            return;
        }
        let y = self.unwrap_shift_mask(y, bits);
        let count = self.bits(y, C);
        if left { self.asm.alu(Alu::Sub, Size::X, C, ZR, count); } else { self.asm.mov(C, count); }
        self.clobber_reg(C);
        if bits >= 32 {
            let value = self.bits(x, A);
            self.asm.alu(Alu::Rorv, Size::of(u32::from(bits) / 8), dst, value, C);
            return;
        }
        let value = self.zext_src(x, A);
        self.asm.mov_imm(D, i64::from(bits - 1));
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Lsrv, Size::X, T, value, C);
        self.asm.alu(Alu::Sub, Size::X, C, ZR, C);
        self.asm.alu(Alu::And, Size::X, C, C, D);
        self.asm.alu(Alu::Lslv, Size::X, dst, value, C);
        self.asm.alu(Alu::Orr, Size::X, dst, dst, T);
        self.clobber_regs(&[C, D, T]);
    }

    pub(super) fn elided_band_cmp(&self, x: Value, y: Value) -> Option<(Value, Value)> {
        let masked = (self.const_imm(y) == Some(0)).then_some(x).or_else(|| (self.const_imm(x) == Some(0)).then_some(y))?;
        if !self.plan.elided[masked.index()] { return None; }
        let Op::Binary(Binary::Band, a, b) = self.func.insts[self.defs[masked.index()]?.index()].op else { return None };
        Some((a, b))
    }

    pub(super) fn cmp_arg(&mut self, val: Value, scratch: Reg, bytes: u32, signed: bool) -> Reg {
        if bytes >= 4 { self.bits(val, scratch) } else { self.src_opt(val, scratch, signed) }
    }

    fn cmp_imm(&mut self, (lhs, rhs): (Value, Value), (bytes, size, cc): (u32, Size, IntCC)) -> bool {
        if self.plan.elided[lhs.index()] { return false; }
        let Some(raw) = self.plan.const_of(rhs) else { return false };
        let rhs_ty = self.ty(rhs);
        let parts = |signed: bool| add_imm_parts(true, crate::fold::extend(raw, rhs_ty, signed)).map(|p| (signed, p));
        let eq = matches!(cc, IntCC::Equal | IntCC::NotEqual);
        let Some((signed, (sub, imm))) = parts(cc.is_signed()).or_else(|| eq.then(|| parts(true)).flatten()) else { return false };
        if imm == 0 && eq && self.tst_small_zero(lhs) {
            self.last_cmp = None;
            return true;
        }
        let reg = self.cmp_arg(lhs, A, bytes, signed);
        self.asm.add_imm(sub, true, size, ZR, reg, imm);
        self.last_cmp = Some((size, lhs, rhs, signed));
        true
    }

    fn cmp_fused(&mut self, cc: IntCC, (x, y): (Value, Value), (x_ty, size, signed): (Type, Size, bool)) -> Option<Cond> {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) && let Some((a, b)) = super::fuse::cmn_sum_parts(self.func, &self.plan, (x, y), &|v| self.defs[v.index()]) {
            let (left, right) = self.bits_pair(a, b);
            self.asm.alu(Alu::Adds, size, ZR, left, right);
            self.last_cmp = None;
            return Some(cond(cc));
        }
        if matches!(x_ty, Type::I32 | Type::I64) && let Some((ext_op, shift, inner)) = super::fuse::ext_parts(self.func, &self.plan, x_ty, y, |v| self.defs[v.index()]) {
            let (left, right) = self.bits_pair(self.peel_uextend(x), inner);
            self.asm.alu_extended(Alu::Subs, size, ext_op, shift, ZR, left, right);
            self.last_cmp = Some((size, x, y, signed));
            return Some(cond(cc));
        }
        let bytes = x_ty.bytes();
        if self.cmp_imm((x, y), (bytes, size, cc)) { return Some(cond(cc)); }
        if self.cmp_imm((y, x), (bytes, size, cc)) { return Some(swap_cond(cond(cc))); }
        if let Some(c) = self.cmp_high_bits(cc, (x, y), size) { return Some(c); }
        if matches!(x_ty, Type::I32 | Type::I64) && let Some((ext_op, shift, inner)) = super::fuse::ext_parts(self.func, &self.plan, x_ty, x, |v| self.defs[v.index()]) {
            let (lhs, rhs) = self.bits_pair(self.peel_uextend(y), inner);
            self.asm.alu_extended(Alu::Subs, size, ext_op, shift, ZR, lhs, rhs);
            self.last_cmp = Some((size, y, x, signed));
            return Some(swap_cond(cond(cc)));
        }
        None
    }

    fn cmp_shift_cmn(&mut self, cc: IntCC, (x, y): (Value, Value), (x_ty, size, signed): (Type, Size, bool)) -> Option<Cond> {
        if !matches!(x_ty, Type::I32 | Type::I64) { return None; }
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) && let Some((other, (rhs, shift))) = super::fuse::cmn_parts(self.func, &self.plan, x_ty, y, &|v: Value| self.defs[v.index()]).map(|p| (x, p)).or_else(|| super::fuse::cmn_parts(self.func, &self.plan, x_ty, x, &|v: Value| self.defs[v.index()]).map(|p| (y, p))) {
            let (left, right) = self.bits_pair(other, rhs);
            match shift { Some((s, imm)) => self.emit_shifted(Alu::Adds, size, ZR, left, (right, s, imm)), None => self.asm.alu(Alu::Adds, size, ZR, left, right) }
            self.last_cmp = None;
            return Some(cond(cc));
        }
        if let Some((b, shift, imm)) = self.shifted_operand(Alu::Sub, x_ty, y) {
            let (left, right) = self.bits_pair(self.peel_uextend(x), b);
            self.emit_shifted(Alu::Subs, size, ZR, left, (right, shift, imm));
            self.last_cmp = Some((size, x, y, signed));
            return Some(cond(cc));
        }
        let (b, shift, imm) = self.shifted_operand(Alu::Sub, x_ty, x)?;
        let (lhs, rhs) = self.bits_pair(self.peel_uextend(y), b);
        self.emit_shifted(Alu::Subs, size, ZR, lhs, (rhs, shift, imm));
        self.last_cmp = Some((size, y, x, signed));
        Some(swap_cond(cond(cc)))
    }

    pub(super) fn compare(&mut self, cc: IntCC, x: Value, y: Value) -> Cond {
        if matches!(cc, IntCC::Equal | IntCC::NotEqual) && let Some((a, b)) = self.elided_band_cmp(x, y) {
            self.emit_tst(a, b);
            self.last_cmp = None;
            return cond(cc);
        }
        let x_ty = self.ty(x);
        let (bytes, signed) = (x_ty.bytes(), cc.is_signed());
        let size = if bytes >= 8 { Size::X } else { Size::W };
        if self.last_cmp == Some((size, x, y, signed)) { return cond(cc); }
        if self.last_cmp == Some((size, y, x, signed)) { return swap_cond(cond(cc)); }
        if let Some(c) = self.cmp_sload(cc, x, y).or_else(|| self.cmp_fused(cc, (x, y), (x_ty, size, signed))).or_else(|| self.cmp_shift_cmn(cc, (x, y), (x_ty, size, signed))) { return c; }
        let (left, right) = if bytes >= 4 { self.bits_pair(x, y) } else { self.pair_opt(x, y, signed) };
        self.asm.alu(Alu::Subs, size, ZR, left, right);
        self.last_cmp = Some((size, x, y, signed));
        cond(cc)
    }

    pub(super) fn icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        if self.plan.is_static(result) { return Ok(()); }
        if self.ty(x).is_vector() { self.invalidate_spills(); return self.vec_icmp(cc, x, y, result); }
        let flags = if self.ty(x) == Type::I128 { self.last_cmp = None; self.invalidate_spills(); self.wide_icmp(cc, x, y) } else { self.compare(cc, x, y) };
        self.set_flag(result, flags);
        Ok(())
    }

    pub(super) fn set_flag(&mut self, result: Value, flags: Cond) {
        self.flags = Some((result, flags));
        if self.plan.fused[result.index()] { return; }
        let dst = self.dst(result);
        self.asm.cset(flags, dst);
        self.put(result, dst);
    }

    pub(super) fn fits(&mut self, op: Binary, signed: bool, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let pair = self.plan.pairs[result.index()];
        if self.plan.is_static(result) && pair.is_none() { return Ok(()); }
        self.invalidate_spills();
        let bits = self.ty(x).bits() as u8;
        let reg = pair.and_then(|sum| self.plan.reg_of(sum)).unwrap_or(IP);
        let ok = match (op, bits) {
            (Binary::Iadd | Binary::Isub, 32 | 64) => {
                let (ty, sub) = (self.ty(x), op == Binary::Isub);
                let size = if bits == 64 { Size::X } else { Size::W };
                let imm_pair = self.add_imm_args(sub, y, ty).map(|p| (x, p)).or_else(|| (!sub).then(|| self.add_imm_args(false, x, ty).map(|p| (y, p))).flatten());
                if let Some((lhs, (eff_sub, imm))) = imm_pair {
                    let left = self.bits(lhs, A); self.asm.add_imm(eff_sub, true, size, reg, left, imm);
                } else {
                    let (left, right) = self.bits_pair(x, y); self.asm.alu(if sub { Alu::Subs } else { Alu::Adds }, size, reg, left, right);
                }
                match (signed, sub) { (true, _) => Cond::Vc, (false, false) => Cond::Lo, (false, true) => Cond::Hs }
            }
            (Binary::Iadd | Binary::Isub | Binary::Imul, _) if bits < 64 => {
                let (left, right) = self.pair_opt(x, y, signed);
                match op {
                    Binary::Iadd => self.asm.alu(Alu::Add, Size::X, reg, left, right),
                    Binary::Isub => self.asm.alu(Alu::Sub, Size::X, reg, left, right),
                    _ => self.asm.madd(false, Size::X, reg, (left, right), ZR),
                }
                self.asm.extend(signed, u32::from(bits) / 8, A, reg);
                self.asm.alu(Alu::Subs, Size::X, ZR, reg, A);
                Cond::Eq
            }
            (Binary::Imul, _) => {
                let (left, right) = self.bits_pair(x, y);
                self.asm.alu(if signed { Alu::Smulh } else { Alu::Umulh }, Size::X, T, left, right);
                self.asm.madd(false, Size::X, reg, (left, right), ZR);
                let expected = if signed { self.asm.asr_imm(A, reg, 63); A } else { ZR };
                self.asm.alu(Alu::Subs, Size::X, ZR, T, expected);
                Cond::Eq
            }
            (other, _) => return Err(format!("{other:?} has no overflow check")),
        };
        if let Some(sum) = pair { self.put(sum, reg); }
        self.set_flag(result, ok);
        Ok(())
    }

    pub(super) fn convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        if self.plan.is_static(result) { return Ok(()); }
        if self.ty(x).is_vector() || self.ty(result).is_vector() { self.invalidate_spills(); return self.vec_convert(op, x, result); }
        match op {
            Convert::Uextend if self.ty(result) != Type::I128 && self.is_zext_reg(x) => {
                let dst = self.dst(result); self.raw(dst, x); self.put(result, dst); Ok(())
            }
            Convert::Uextend | Convert::Sextend => self.emit_extend_conv(op, x, result),
            Convert::Ireduce => {
                let dst = self.dst(result);
                if self.plan.elided[x.index()] && let Some(inst) = self.defs[x.index()] && let Op::Binary(Binary::Ushr, val, amt) = self.func.insts[inst.index()].op {
                    self.shift_by(Binary::Ushr, (val, amt), self.ty(x).bits() as u8, dst);
                } else { self.raw(dst, x); }
                self.put(result, dst); Ok(())
            }
            Convert::NarrowSat { signed } => {
                let (src, dst, to_bits) = (self.src(x, A, Ext::Sign), self.dst(result), self.ty(result).bits());
                let (lo, hi) = if signed { (-(1i64 << (to_bits - 1)), (1i64 << (to_bits - 1)) - 1) } else { (0, (1i64 << to_bits) - 1) };
                self.clamp_range(src, dst, lo, hi); self.put(result, dst); Ok(())
            }
            _ => { self.invalidate_spills(); self.float_convert(op, x, result) }
        }
    }
}
