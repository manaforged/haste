mod arith;
mod bits;
mod cmp;
mod imm;

use super::asm::{Alu, Bits, Cond, Reg, Shift, Size, ZR};
use super::{A, C, Emit, V0};
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, Convert, IntCC, Op, Type, Unary, Value};
pub(crate) use imm::{add_imm_parts, add_imm_split_parts, encode_bitmask_imm};

pub(super) fn cond(cc: IntCC) -> Cond {
    match cc {
        IntCC::Equal => Cond::Eq,
        IntCC::NotEqual => Cond::Ne,
        IntCC::SignedLessThan => Cond::Lt,
        IntCC::SignedGreaterThanOrEqual => Cond::Ge,
        IntCC::SignedGreaterThan => Cond::Gt,
        IntCC::SignedLessThanOrEqual => Cond::Le,
        IntCC::UnsignedLessThan => Cond::Lo,
        IntCC::UnsignedGreaterThanOrEqual => Cond::Hs,
        IntCC::UnsignedGreaterThan => Cond::Hi,
        IntCC::UnsignedLessThanOrEqual => Cond::Ls,
    }
}

pub(super) fn alu_of(op: Binary) -> Option<Alu> {
    match op {
        Binary::Iadd => Some(Alu::Add),
        Binary::Isub => Some(Alu::Sub),
        Binary::Band => Some(Alu::And),
        Binary::Bor => Some(Alu::Orr),
        Binary::Bxor => Some(Alu::Eor),
        _ => None,
    }
}

impl Emit<'_> {
    pub(super) fn is_zext_reg(&self, val: Value) -> bool {
        !self.plan.elided[val.index()]
            && matches!(self.plan.location(val), Location::Reg(_) | Location::Slot)
            && self.plan.zero_ext[val.index()]
    }

    pub(super) fn zext_src(&mut self, val: Value, scratch: Reg) -> Reg {
        if self.is_zext_reg(val) {
            self.bits(val, scratch)
        } else {
            self.src(val, scratch, Ext::Zero)
        }
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
        if self.is_zext_reg(val) {
            self.raw(dst, val);
        } else {
            self.get(dst, val, Ext::Zero);
        }
    }

    pub(super) fn get_opt(&mut self, dst: Reg, val: Value, signed: bool) {
        if signed {
            self.get(dst, val, Ext::Sign);
        } else {
            self.zext_get(dst, val);
        }
    }

    pub(super) fn src_opt(&mut self, val: Value, scratch: Reg, signed: bool) -> Reg {
        if signed {
            self.src(val, scratch, Ext::Sign)
        } else {
            self.zext_src(val, scratch)
        }
    }

    pub(super) fn pair_opt(&mut self, x: Value, y: Value, signed: bool) -> (Reg, Reg) {
        if signed {
            self.src_pair(x, y, Ext::Sign)
        } else {
            self.zext_pair(x, y)
        }
    }

    pub(super) fn tst_small_zero(&mut self, val: Value) -> bool {
        let bits = self.ty(val).bits();
        if bits >= 32 || self.is_zext_reg(val) || self.plan.elided[val.index()] {
            return false;
        }
        let Location::Reg(reg) = self.plan.location(val) else {
            return false;
        };
        let Some(mask) = encode_bitmask_imm((1_u64 << bits) - 1, false) else {
            return false;
        };
        self.asm.alu_bitmask(Alu::Ands, Size::W, mask, ZR, reg);
        true
    }

    fn bits32(&mut self, op: Bits, dst: Reg, src: Reg) {
        self.asm.emit_gp(
            dst,
            (op as u32 & !(1 << 31)) | u32::from(src.0) << 5 | u32::from(dst.0),
        );
    }

    fn emit_shifted(
        &mut self,
        alu: Alu,
        size: Size,
        dst: Reg,
        lhs: Reg,
        (rhs, shift, imm): (Reg, Shift, u8),
    ) {
        if imm == 0 {
            self.asm.alu(alu, size, dst, lhs, rhs);
        } else {
            self.asm.alu_shifted(alu, size, shift, imm, dst, lhs, rhs);
        }
    }

    fn add_imm_args(&self, sub: bool, value: Value, ty: Type) -> Option<(bool, u32)> {
        add_imm_parts(
            sub,
            crate::fold::extend(self.plan.const_of(value)?, ty, true),
        )
    }

    fn add_imm_split(&self, sub: bool, value: Value, ty: Type) -> Option<(bool, u32, u32)> {
        let raw = self.plan.const_of(value)?;
        if super::asm::Asm::mov_imm_is_single(raw) {
            return None;
        }
        add_imm_split_parts(sub, crate::fold::extend(raw, ty, true))
    }

    fn emit_add_sub_imm(
        &mut self,
        sub: bool,
        (x, y): (Value, Value),
        (ty, size): (Type, Size),
        dst: Reg,
    ) -> bool {
        if let Some((eff_sub, imm)) = self.add_imm_args(sub, y, ty) {
            let left = self.bits(self.peel_uextend(x), A);
            self.asm.add_imm(eff_sub, false, size, dst, left, imm);
            return true;
        }
        if !sub && let Some((eff_sub, imm)) = self.add_imm_args(false, x, ty) {
            let left = self.bits(self.peel_uextend(y), A);
            self.asm.add_imm(eff_sub, false, size, dst, left, imm);
            return true;
        }
        if dst == ZR {
            return false;
        }
        let split = self.add_imm_split(sub, y, ty).map(|p| (x, p)).or_else(|| {
            (!sub)
                .then(|| self.add_imm_split(false, x, ty).map(|p| (y, p)))
                .flatten()
        });
        let Some((other, (eff_sub, hi, lo))) = split else {
            return false;
        };
        let left = self.bits(self.peel_uextend(other), A);
        if left == ZR {
            return false;
        }
        self.asm.add_imm(eff_sub, false, size, dst, left, hi);
        self.asm.add_imm(eff_sub, false, size, dst, dst, lo);
        true
    }

    fn peel_uextend(&self, val: Value) -> Value {
        if self.plan.elided[val.index()]
            && let Some(inst) = self.defs[val.index()]
            && let Op::Convert(Convert::Uextend, inner) = self.func.insts[inst.index()].op
        {
            return inner;
        }
        val
    }

    fn shifted_operand(&self, alu: Alu, ty: Type, val: Value) -> Option<(Value, Shift, u8)> {
        if !self.plan.elided[val.index()] {
            return None;
        }
        let (op, b, imm) = super::fuse::shift_parts(
            &self.func.insts[self.defs[val.index()]?.index()].op,
            &self.plan,
        )?;
        let (shift, count) = super::fuse::alu_shift(
            matches!(alu, Alu::And | Alu::Orr | Alu::Eor | Alu::Orn | Alu::Eon),
            ty,
            op,
            imm,
        )?;
        Some((self.peel_uextend(b), shift, count))
    }

    pub(super) fn cinc_operand(&self, val: Value) -> Option<(IntCC, Value, Value)> {
        super::fuse::cinc_operand(self.func, &self.plan, val, &|v| self.defs[v.index()])
    }

    fn const_imm(&self, val: Value) -> Option<i64> {
        super::fuse::const_of(self.func, &self.plan, val, |v| self.defs[v.index()])
    }

    pub(super) fn unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let ty = self.ty(x);
        if ty.is_vector() {
            self.invalidate_spills();
            return self.vec_unary(op, x, result);
        }
        if ty.is_float() {
            self.invalidate_spills();
            return self.float_unary(op, x, result);
        }
        if ty == Type::I128 {
            self.invalidate_spills();
            return self.wide_unary(op, x, result);
        }
        let dst = self.dst(result);
        self.int_unary(op, x, ty, dst)?;
        self.put(result, dst);
        Ok(())
    }

    fn int_unary(&mut self, op: Unary, x: Value, ty: Type, dst: Reg) -> Result<(), String> {
        let bits = ty.bits() as u8;
        let size = if bits <= 32 { Size::W } else { Size::X };
        match op {
            Unary::Ineg => {
                if matches!(bits, 32 | 64)
                    && let Some((cc, a, b)) = self.cinc_operand(x)
                {
                    let cond = self.compare(cc, a, b);
                    self.asm.csinv(size, cond.invert(), dst, ZR, ZR);
                } else if matches!(bits, 32 | 64)
                    && let Some((b, shift, imm)) = self.shifted_operand(Alu::Sub, ty, x)
                {
                    let right = self.bits(b, C);
                    self.emit_shifted(Alu::Sub, size, dst, ZR, (right, shift, imm));
                } else if matches!(bits, 32 | 64)
                    && let Some((a, b)) =
                        super::fuse::mneg_parts(self.func, &self.plan, ty, x, |v| {
                            self.defs[v.index()]
                        })
                {
                    let (left, right) = self.bits_pair(a, b);
                    self.asm.madd(true, size, dst, (left, right), ZR);
                } else {
                    let reg = self.bits(x, A);
                    self.asm.alu(Alu::Sub, size, dst, ZR, reg);
                }
            }
            Unary::Bnot => {
                if let Some((a, b)) =
                    super::fuse::eon_inner(self.func, &self.plan, x, |v| self.defs[v.index()])
                {
                    let shifted = self
                        .shifted_operand(Alu::Eon, ty, b)
                        .map(|s| (a, s))
                        .or_else(|| self.shifted_operand(Alu::Eon, ty, a).map(|s| (b, s)));
                    if let Some((other, (rhs, shift, imm))) = shifted {
                        let (left, right) = self.bits_pair(self.peel_uextend(other), rhs);
                        self.emit_shifted(Alu::Eon, size, dst, left, (right, shift, imm));
                    } else {
                        let (left, right) =
                            self.bits_pair(self.peel_uextend(a), self.peel_uextend(b));
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
                let op = match bits {
                    16 => Bits::Rev16,
                    32 => Bits::Rev32,
                    64 => Bits::Rev,
                    _ => return Ok(self.asm.mov(dst, reg)),
                };
                self.asm.bits(op, dst, reg);
            }
            Unary::Bitrev => {
                let reg = self.bits(x, A);
                if bits == 32 {
                    self.bits32(Bits::Rbit, dst, reg);
                } else {
                    self.asm.bits(Bits::Rbit, dst, reg);
                    if bits < 64 {
                        self.asm.lsr_imm(dst, dst, 64 - bits);
                    }
                }
            }
            Unary::Clz => {
                if bits == 32 {
                    let reg = self.bits(x, A);
                    self.bits32(Bits::Clz, dst, reg);
                } else {
                    let reg = self.zext_src(x, A);
                    self.asm.bits(Bits::Clz, dst, reg);
                    if bits < 64 {
                        self.asm
                            .add_imm(true, false, Size::X, dst, dst, u32::from(64 - bits));
                    }
                }
            }
            Unary::Ctz => {
                let reg = self.bits(x, A);
                if bits == 32 {
                    self.bits32(Bits::Rbit, dst, reg);
                    self.bits32(Bits::Clz, dst, dst);
                } else if bits < 32
                    && let Some(mask) = encode_bitmask_imm(1_u64 << bits, false)
                {
                    self.asm.alu_bitmask(Alu::Orr, Size::W, mask, A, reg);
                    self.clobber_reg(A);
                    self.bits32(Bits::Rbit, dst, A);
                    self.bits32(Bits::Clz, dst, dst);
                } else {
                    let source = if bits < 64 {
                        self.asm.mov_imm(C, 1 << bits);
                        self.asm.alu(Alu::Orr, Size::X, A, reg, C);
                        self.clobber_regs(&[A, C]);
                        A
                    } else {
                        reg
                    };
                    self.asm.bits(Bits::Rbit, dst, source);
                    self.asm.bits(Bits::Clz, dst, dst);
                }
            }
            Unary::Popcnt => {
                let reg = self.zext_src(x, A);
                self.popcount(dst, reg);
                self.clobber_vreg(V0);
            }
            other => return Err(format!("{other:?} of an integer")),
        }
        Ok(())
    }

    pub(super) fn popcount(&mut self, dst: Reg, src: Reg) {
        self.asm.fmov_to_float(V0, src);
        self.asm.popcount_bytes(V0, V0);
        self.asm.fmov_to_int(dst, V0);
    }
}

pub(super) const fn size_mask(size: Size) -> u64 {
    match size {
        Size::X => u64::MAX,
        _ => u32::MAX as u64,
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{Op, Type};
    use crate::{AbiParam, CallConv, Effort, Function, FunctionBuilder, IntCC, Signature, Target};

    fn arm64_full() -> Target {
        let mut target = *Target::find("aarch64-apple-darwin").expect("arm64 darwin target");
        target.effort = Effort::Full;
        target
    }

    fn arm64_words(code: crate::Code) -> Vec<u32> {
        code.bytes()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn band_folded_zero_does_not_materialize_zero_in_x9() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            b.switch_to_block(entry);
            let eight = b.ins().iconst(Type::I64, 8);
            let one = b.ins().iconst(Type::I64, 1);
            let eq = b.ins().icmp(IntCC::Equal, eight, one);
            let ext = b.ins().uextend(Type::I64, eq);
            let masked = b.ins().band(params[0], ext);
            b.ins().return_(&[masked]);
        }
        let has_band = func.blocks[0].insts.iter().any(|&i| {
            matches!(
                func.insts[i.index()].op,
                Op::Binary(crate::ir::Binary::Band, ..)
            )
        });
        assert!(has_band);
        let words = arm64_words(
            super::super::compile(&func, &arm64_full()).expect("arm64 band zero compiles"),
        );
        assert!(
            words.contains(&0xD280_0000),
            "expected movz x0, #0 in {words:#x?}"
        );
        assert!(
            !words.contains(&0xD280_0009),
            "must not load #0 into scratch x9: {words:#x?}"
        );
        assert!(
            !words.iter().any(|&w| (w & 0x7F20_0000) == 0x0A00_0000),
            "must not emit register AND when mask is known zero: {words:#x?}"
        );
    }

    #[test]
    fn tst_folded_zero_uses_xzr_without_scratch_imm() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            let then_b = b.create_block();
            let else_b = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            b.switch_to_block(entry);
            let eight = b.ins().iconst(Type::I64, 8);
            let one = b.ins().iconst(Type::I64, 1);
            let eq = b.ins().icmp(IntCC::Equal, eight, one);
            let ext = b.ins().uextend(Type::I64, eq);
            let masked = b.ins().band(params[0], ext);
            let zero_cmp = b.ins().icmp_imm(IntCC::Equal, masked, 0);
            b.ins().brif(zero_cmp, then_b, &[], else_b, &[]);

            b.switch_to_block(then_b);
            let a = b.ins().iconst(Type::I64, 7);
            b.ins().return_(&[a]);

            b.switch_to_block(else_b);
            let c = b.ins().iconst(Type::I64, 9);
            b.ins().return_(&[c]);
        }
        let words = arm64_words(
            super::super::compile(&func, &arm64_full()).expect("arm64 tst zero compiles"),
        );
        assert!(
            words.contains(&0xEA1F_03FF),
            "expected tst xzr, xzr in {words:#x?}"
        );
        assert!(
            !words.contains(&0xD280_000A),
            "must not load #0 into scratch x10 for tst: {words:#x?}"
        );
    }

    #[test]
    fn bor_zero_and_band_all_ones_avoid_scratch_mov() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let params = b.block_params(entry).to_vec();
            b.switch_to_block(entry);
            let eight = b.ins().iconst(Type::I64, 8);
            let one = b.ins().iconst(Type::I64, 1);
            let eq = b.ins().icmp(IntCC::Equal, eight, one);
            let zero = b.ins().uextend(Type::I64, eq);
            let ones = b.ins().iconst(Type::I64, -1);
            let same = b.ins().bor(params[1], zero);
            let still_same = b.ins().band(same, ones);
            b.ins().return_(&[still_same]);
        }
        let words = arm64_words(
            super::super::compile(&func, &arm64_full()).expect("arm64 identity bitwise compiles"),
        );
        assert!(
            !words.contains(&0xD280_0009) && !words.contains(&0x9280_0009),
            "must not materialize 0 or -1 in scratch x9: {words:#x?}"
        );
        assert!(
            !words.iter().any(|&w| (w & 0x7F20_0000) == 0x0A00_0000),
            "must not emit AND for all-ones mask: {words:#x?}"
        );
    }
}
