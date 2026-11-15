use super::super::asm::{Alu, Reg, Size, ZR};
use super::super::fuse::{Bfx, unwrap_elided_tst};
use super::{A, C, Emit, encode_bitmask_imm};
use crate::frame::Ext;
use crate::ir::{Convert, Type, Value};

impl Emit<'_> {
    pub(super) fn emit_bfm(&mut self, bfx: Bfx, ty: Type, dst: Reg) {
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let reg_bits: u8 = if size == Size::X { 64 } else { 32 };
        let reg = self.bits(self.peel_uextend(bfx.src), A);
        if !bfx.insert && bfx.lsb == 0 && bfx.width == reg_bits {
            if size == Size::W {
                self.asm.extend(false, 4, dst, reg);
            } else {
                self.asm.mov(dst, reg);
            }
            return;
        }
        let (immr, imms) = if bfx.insert {
            ((reg_bits - bfx.lsb) % reg_bits, bfx.width - 1)
        } else {
            (bfx.lsb, bfx.lsb + bfx.width - 1)
        };
        let base = match (bfx.signed, size == Size::X) {
            (false, false) => 0x5300_0000,
            (true, false) => 0x1300_0000,
            (false, true) => 0xD340_0000,
            (true, true) => 0x9340_0000,
        };
        self.asm.word(base | u32::from(immr) << 16 | u32::from(imms) << 10 | u32::from(reg.0) << 5 | u32::from(dst.0));
    }

    pub(in crate::arm64) fn emit_tst(&mut self, x: Value, y: Value) {
        let ty = self.ty(x);
        let bits = ty.bits();
        let size = if bits >= 64 { Size::X } else { Size::W };
        let ty_mask = if bits == 64 { u64::MAX } else { (1_u64 << bits) - 1 };
        let imm_pair = match (self.const_imm(y), self.const_imm(x)) {
            (Some(imm), _) => Some((x, (imm as u64) & ty_mask)),
            (None, Some(imm)) => Some((y, (imm as u64) & ty_mask)),
            (None, None) => None,
        };
        if let Some((raw_val, raw_imm)) = imm_pair
            && let Some((val, imm)) = unwrap_elided_tst(self.func, &self.plan, raw_val, raw_imm, &|v| self.defs[v.index()])
        {
            let left = self.bits(val, A);
            if let Some(bitmask) = encode_bitmask_imm(imm, size == Size::X) {
                self.asm.alu_bitmask(Alu::Ands, size, bitmask, ZR, left);
            } else {
                self.asm.mov_imm(C, imm as i64);
                self.asm.alu(Alu::Ands, size, ZR, left, C);
                self.clobber_reg(C);
            }
            return;
        }
        let left = if bits >= 32 { self.bits(x, A) } else { self.src(x, A, Ext::Zero) };
        let right = if bits >= 32 { self.bits(y, C) } else { self.src(y, C, Ext::Zero) };
        self.asm.alu(Alu::Ands, size, ZR, left, right);
    }

    pub(super) fn emit_extend_conv(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let dst_ty = self.ty(result);
        let dst = self.dst(result);
        if self.plan.elided[x.index()]
            && let Some(bfx) = super::super::fuse::conv_bfx_parts(self.func, &self.plan, (dst_ty, op), x, &|v| self.defs[v.index()])
        {
            self.emit_bfm(bfx, dst_ty, dst);
            self.put(result, dst);
            return Ok(());
        }
        let signed = op == Convert::Sextend;
        self.get(dst, x, Ext::of(signed));
        self.put(result, dst);
        if dst_ty == Type::I128 && signed {
            self.asm.asr_imm(C, dst, 63);
            self.put_high(result, C);
            self.clobber_reg(C);
        } else if dst_ty == Type::I128 {
            self.put_high(result, ZR);
        }
        Ok(())
    }
}
