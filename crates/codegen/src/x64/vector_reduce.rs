use super::Emit;
use super::asm::{Alu, Cond, Mem, Reg, Size};
use super::float::{X0, X1, X2};
use super::vector::{vec_alu_opcode, vec_prefix};
use crate::ir::{Binary, IntCC, Lane, ReduceOp, Value};

impl Emit<'_> {
    pub(super) fn vector_reduce(
        &mut self,
        op: ReduceOp,
        v: Value,
        result: Value,
    ) -> Result<(), String> {
        let Some((lane, count)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in reduce", self.ty(v)));
        };
        if op == ReduceOp::Movemask {
            let dst = self.dst(result);
            let src = self.fin(v, X0);
            let full = match lane {
                Lane::I8 => {
                    self.asm.sse(&[0x66], &[0x0F, 0xD7], dst as u8, src);
                    16
                }
                Lane::I16 => {
                    self.asm.movaps(X0, src);
                    self.asm.sse(&[0x66], &[0x0F, 0x63], X0.0, X0);
                    self.asm.sse(&[0x66], &[0x0F, 0xD7], dst as u8, X0);
                    0
                }
                Lane::I32 | Lane::F32 => {
                    self.asm.sse(&[], &[0x0F, 0x50], dst as u8, src);
                    4
                }
                Lane::I64 | Lane::F64 => {
                    self.asm.sse(&[0x66], &[0x0F, 0x50], dst as u8, src);
                    2
                }
            };
            if count < full || full == 0 {
                self.asm.alu_imm32(Alu::And, dst, (1_i32 << count) - 1);
            }
            self.put(result, dst);
            return Ok(());
        }
        self.fget(X0, v);
        if lane == Lane::I64
            && matches!(
                op,
                ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax
            )
        {
            let dst = self.dst(result);
            self.asm.movaps(X1, X0);
            self.asm.pshift_imm(3, 8, X1, 8);
            self.asm.movq_out(dst, X0);
            self.asm.movq_out(Reg::Rcx, X1);
            self.asm.alu(Alu::Cmp, dst, Reg::Rcx);
            let cc = match op {
                ReduceOp::Smin => Cond::G,
                ReduceOp::Smax => Cond::L,
                ReduceOp::Umin => Cond::A,
                _ => Cond::B,
            };
            self.asm.cmov(cc, dst, Reg::Rcx);
            self.put(result, dst);
            return Ok(());
        }
        let mut active = count;
        while active > 1 {
            let half = active / 2;
            let shift_bytes = half * lane.bytes() as u8;
            self.asm.movaps(X1, X0);
            self.asm.pshift_imm(3, 8, X1, shift_bytes);
            self.reduce_step(op, lane)?;
            active = half;
        }
        if lane.is_float() {
            let dst = self.fdst(result);
            if dst != X0 {
                self.asm.movaps(dst, X0);
            }
            self.fput(result, dst);
        } else {
            let dst = self.dst(result);
            match lane {
                Lane::I8 | Lane::I16 => {
                    self.asm.movd_out(dst, X0);
                    self.asm.load(Size::of(lane.bytes()), dst, dst);
                }
                Lane::I32 => self.asm.movd_out(dst, X0),
                Lane::I64 => self.asm.movq_out(dst, X0),
                _ => {}
            }
            self.put(result, dst);
        }
        Ok(())
    }

    fn reduce_step(&mut self, op: ReduceOp, lane: Lane) -> Result<(), String> {
        match op {
            ReduceOp::Add if lane.is_float() => {
                self.asm.sse(vec_prefix(lane), &[0x0F, 0x58], X0.0, X1)
            }
            ReduceOp::Add => {
                let opc = vec_alu_opcode(Binary::Iadd, lane)
                    .ok_or_else(|| format!("reduce add {lane:?}"))?;
                self.asm.sse(&[0x66], &[0x0F, opc], X0.0, X1);
            }
            ReduceOp::And => self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1),
            ReduceOp::Or => self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X1),
            ReduceOp::Xor => self.asm.sse(&[0x66], &[0x0F, 0xEF], X0.0, X1),
            ReduceOp::Fmin | ReduceOp::Fmax => {
                let bin = if op == ReduceOp::Fmin {
                    Binary::Fmin
                } else {
                    Binary::Fmax
                };
                self.fminmax_regs(bin, lane, X2);
                self.asm.movaps(X0, X2);
            }
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax => {
                let (bin, cc) = match op {
                    ReduceOp::Smin => (Binary::Smin, IntCC::SignedLessThan),
                    ReduceOp::Smax => (Binary::Smax, IntCC::SignedGreaterThan),
                    ReduceOp::Umin => (Binary::Umin, IntCC::UnsignedLessThan),
                    _ => (Binary::Umax, IntCC::UnsignedGreaterThan),
                };
                if let Some(bytes) = self.iminmax_opc(bin, lane) {
                    self.asm.sse(&[0x66], bytes, X0.0, X1);
                } else {
                    self.reserve_scratch(32);
                    self.asm
                        .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
                    self.asm
                        .sse(&[0xF3], &[0x0F, 0x7F], X1.0, Mem::new(Reg::Rsp, 16));
                    self.emit_icmp_regs(cc, lane, X0, X1, X2);
                    self.asm
                        .sse(&[0xF3], &[0x0F, 0x6F], X0.0, Mem::new(Reg::Rsp, 0));
                    self.asm
                        .sse(&[0xF3], &[0x0F, 0x6F], X1.0, Mem::new(Reg::Rsp, 16));
                    self.release_scratch(32);
                    self.blend_regs(X2, X0, X1, X0);
                }
            }
            ReduceOp::Movemask => {}
        }
        Ok(())
    }
}
