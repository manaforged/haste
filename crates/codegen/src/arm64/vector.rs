use super::asm::{Alu, Reg, Size, Vreg, ZR};
use super::{A, C, D, Emit, IP, V0, V1};
use crate::frame::Ext;
use crate::ir::{Binary, ByteCC, Op, Type, Value, VectorOp};

impl Emit<'_> {
    pub(super) fn vector_inst(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<bool, String> {
        match *op {
            Op::Const(0) if self.ty(result()?) == Type::V128 => {
                let res = result()?;
                if !self.plan.is_static(res) {
                    let dst = self.fdst(res);
                    self.asm.veor_zero(dst);
                    self.fput(res, dst);
                }
            }
            Op::Binary(bin, x, y) if self.ty(x) == Type::V128 => {
                let res = result()?;
                if !self.plan.is_static(res) && !self.plan.deferred[res.index()] {
                    self.vector_binary(bin, (x, y), res)?;
                }
            }
            Op::Load(addr, offset) if self.ty(result()?) == Type::V128 => {
                let res = result()?;
                let dst = self.fdst(res);
                if let Some((base, index, shift)) = self.sib_indexed(addr, offset, 16) {
                    self.asm.vload_indexed(16, dst, base, index, shift);
                } else {
                    let from = self.address(addr, offset);
                    self.asm.vload(16, dst, from);
                }
                self.fput(res, dst);
            }
            Op::Store(val, addr, offset) if self.ty(val) == Type::V128 => {
                let src = self.fin(val, V0);
                if let Some((base, index, shift)) = self.sib_indexed(addr, offset, 16) {
                    self.asm.vstore_indexed(16, base, index, shift, src);
                } else {
                    let to = self.address(addr, offset);
                    self.asm.vstore(16, to, src);
                }
            }
            Op::StackLoad(slot, offset) if self.ty(result()?) == Type::V128 => {
                let res = result()?;
                let dst = self.fdst(res);
                self.asm.vload(16, dst, self.slot(slot, offset));
                self.fput(res, dst);
            }
            Op::StackStore(val, slot, offset) if self.ty(val) == Type::V128 => {
                let src = self.fin(val, V0);
                self.asm.vstore(16, self.slot(slot, offset), src);
            }
            Op::ByteCmp(cc, x, y) => {
                let res = result()?;
                let dst = self.fdst(res);
                let a = self.fin(x, V0);
                let b = self.fin(y, V1);
                self.asm.vcmp_byte(cc == ByteCC::Equal, dst, a, b);
                self.fput(res, dst);
            }
            Op::Vector(kind, x, y) => self.vector_op(kind, (x, y), result()?)?,
            Op::Splat(x) => {
                let res = result()?;
                let dst = self.fdst(res);
                let src = self.src(x, A, Ext::Zero);
                self.asm.vdup_byte(dst, src);
                self.fput(res, dst);
            }
            Op::Movemask(x) => {
                let res = result()?;
                let source = self.fin(x, V0);
                let dst = self.dst(res);
                self.movemask(dst, source);
                self.put(res, dst);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn vector_binary(&mut self, op: Binary, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let base = match op {
            Binary::Band => 0x4E20_1C00,
            Binary::Bor => 0x4EA0_1C00,
            Binary::Bxor => 0x6E20_1C00,
            _ => return Err(format!("{op:?} of a vector")),
        };
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        self.asm.vbit(base, dst, a, b);
        self.fput(result, dst);
        Ok(())
    }

    fn vector_op(&mut self, kind: VectorOp, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        match kind {
            VectorOp::ByteShuffle => self.asm.vtbl(dst, a, b),
            VectorOp::AddBytes => self.asm.varith(false, 0, dst, a, b),
            VectorOp::AddWords => self.asm.varith(false, 1, dst, a, b),
            VectorOp::AddDwords => self.asm.varith(false, 2, dst, a, b),
            VectorOp::AddQwords => self.asm.varith(false, 3, dst, a, b),
            VectorOp::SubBytes => self.asm.varith(true, 0, dst, a, b),
            VectorOp::SubWords => self.asm.varith(true, 1, dst, a, b),
            VectorOp::SubDwords => self.asm.varith(true, 2, dst, a, b),
            VectorOp::SubQwords => self.asm.varith(true, 3, dst, a, b),
            other => return Err(format!("unsupported ARM64 vector op {other:?}")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn movemask(&mut self, dst: Reg, source: Vreg) {
        self.asm.fmov_out(true, A, source);
        self.asm.umov_d1(C, source);
        self.asm.mov_imm(D, 0x0101_0101_0101_0101_u64 as i64);
        self.asm.mov_imm(IP, 0x0102_0408_1020_4080_u64 as i64);
        for reg in [A, C] {
            self.asm.lsr_imm(reg, reg, 7);
            self.asm.alu(Alu::And, Size::X, reg, reg, D);
            self.asm.madd(false, reg, (reg, IP), ZR);
            self.asm.lsr_imm(reg, reg, 56);
        }
        self.asm.lsl_imm(C, C, 8);
        self.asm.alu(Alu::Orr, Size::X, dst, A, C);
    }
}
