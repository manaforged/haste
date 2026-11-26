mod scalarize;
mod shuffle;
mod target_op;

use super::Emit;
use super::asm::{Mem, Reg, Shift, Size, Xmm};
use super::float::{X0, X1, X2};
use crate::frame::Ext;
use crate::ir::{Lane, Op, Type, Value};

impl Emit<'_> {
    pub(super) fn vector_op(&mut self, op: &Op, result: Value) -> Result<(), String> {
        if let Some(done) = self.wide_op(op, result) {
            return done;
        }
        match op {
            Op::Shuffle(a, ..) if self.splat_insert(*a, result) => Ok(()),
            Op::Shuffle(a, b, mask) => self.vector_shuffle(*a, *b, mask, result),
            Op::Extract(v, lane) => self.vector_extract(*v, *lane, result),
            Op::Insert(v, x, lane) => self.vector_insert(*v, *x, *lane, result),
            Op::Reduce(rop, v) => self.vector_reduce(*rop, *v, result),
            Op::TargetOp(id, args) => self.vector_target_op(*id, args.as_slice(), result),
            other => Err(format!("unexpected op {other:?} in vector_op")),
        }
    }

    pub(super) fn emit_mov_xmm_gpr(&mut self, dst: Xmm, reg: Reg, wide: bool) {
        if self.features.avx && wide {
            let r = reg as u8;
            self.asm.bytes(&[
                0xC4,
                ((!dst.0 >> 3 & 1) << 7) | 0x41 | ((!r >> 3 & 1) << 5),
                0xF9,
                0x6E,
                0xC0 | ((dst.0 & 7) << 3) | (r & 7),
            ]);
        } else if self.features.avx {
            self.asm.vex_xmm(0b00001, 0b01, 0x6E, (dst, Xmm(0)), reg);
        } else if wide {
            self.asm.movq_in(dst, reg);
        } else {
            self.asm.movd_in(dst, reg);
        }
    }

    fn emit_vpinsrq(&mut self, dst: Xmm, src: Xmm, reg: Reg, idx: u8) {
        let r = reg as u8;
        self.asm.bytes(&[
            0xC4,
            ((!dst.0 >> 3 & 1) << 7) | 0x43 | ((!r >> 3 & 1) << 5),
            0x81 | ((!src.0 & 0xF) << 3),
            0x22,
            0xC0 | ((dst.0 & 7) << 3) | (r & 7),
            idx,
        ]);
    }

    /// `vpinsr{b,w,d,q}`: `src` with integer lane `idx` replaced by `reg`, into `dst`.
    pub(super) fn vpinsr(&mut self, lane: Lane, (dst, src): (Xmm, Xmm), reg: Reg, idx: u8) {
        match lane {
            Lane::I8 => self
                .asm
                .vex_xmm_imm(0b00011, 0b01, 0x20, (dst, src), reg, idx),
            Lane::I16 => self
                .asm
                .vex_xmm_imm(0b00001, 0b01, 0xC4, (dst, src), reg, idx),
            Lane::I32 => self
                .asm
                .vex_xmm_imm(0b00011, 0b01, 0x22, (dst, src), reg, idx),
            _ => self.emit_vpinsrq(dst, src, reg, idx),
        }
    }

    fn vector_extract(&mut self, v: Value, idx: u8, result: Value) -> Result<(), String> {
        let Some((lane, _)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in extract", self.ty(v)));
        };
        if lane.is_float() {
            let dst = self.fdst(result);
            let src = self.fin(v, if idx == 0 { dst } else { X0 });
            match (lane, idx) {
                (_, 0) => {
                    if dst != src {
                        self.asm.movaps(dst, src);
                    }
                }
                (Lane::F32, _) => self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, src, idx),
                (Lane::F64, _) => self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, src, 0xEE),
                _ => {}
            }
            self.fput(result, dst);
            return Ok(());
        }
        let dst = self.dst(result);
        let src = self.fin(v, X0);
        match (lane, idx) {
            (Lane::I16, _) => self
                .asm
                .sse_imm(&[0x66], &[0x0F, 0xC5], dst as u8, src, idx),
            (Lane::I8, 0) => {
                self.asm.movd_out(dst, src);
                self.asm.load(Size::B, dst, dst);
            }
            (Lane::I32, 0) => self.asm.movd_out(dst, src),
            (Lane::I64, 0) => self.asm.movq_out(dst, src),
            (Lane::I8, _) if self.features.sse41 => {
                self.asm
                    .sse_imm(&[0x66], &[0x0F, 0x3A, 0x14], src.0, dst, idx)
            }
            (Lane::I8, _) => {
                self.asm
                    .sse_imm(&[0x66], &[0x0F, 0xC5], dst as u8, src, idx / 2);
                if idx & 1 != 0 {
                    self.asm.shift_imm(Size::D, Shift::Shr, dst, 8);
                }
                self.asm.load(Size::B, dst, dst);
            }
            (Lane::I32, _) if self.features.sse41 => {
                self.asm
                    .sse_imm(&[0x66], &[0x0F, 0x3A, 0x16], src.0, dst, idx)
            }
            (Lane::I32, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X1.0, src, idx);
                self.asm.movd_out(dst, X1);
            }
            (Lane::I64, _) if self.features.sse41 => {
                self.asm
                    .sse_wide_imm(&[0x66], &[0x0F, 0x3A, 0x16], src.0, dst, idx)
            }
            (Lane::I64, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X1.0, src, 0xEE);
                self.asm.movq_out(dst, X1);
            }
            _ => unreachable!(),
        }
        self.put(result, dst);
        Ok(())
    }

    fn vector_insert(&mut self, v: Value, x: Value, idx: u8, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in insert", self.ty(v)));
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        if lane.is_float() {
            let [_, s] = self.fins(dst, true, [v, x], &[]);
            match (lane, idx) {
                (Lane::F32, 0) => self.asm.sse(&[0xF3], &[0x0F, 0x10], dst.0, s),
                (Lane::F32, _) if self.features.sse41 => {
                    self.asm
                        .sse_imm(&[0x66], &[0x0F, 0x3A, 0x21], dst.0, s, idx << 4);
                }
                (Lane::F32, _) => {
                    self.asm.movd_out(Reg::Rax, s);
                    self.insert_stack(lane, count, dst, idx);
                }
                (Lane::F64, 0) => self.asm.sse(&[0xF2], &[0x0F, 0x10], dst.0, s),
                (Lane::F64, _) => self.asm.sse(&[0x66], &[0x0F, 0x14], dst.0, s),
                _ => {}
            }
            self.fput(result, dst);
            return Ok(());
        }
        if matches!(lane, Lane::I8 | Lane::I16 | Lane::I32 | Lane::I64)
            && idx == 0
            && self.is_zero_vec(v)
        {
            if self.const_of(x) == Some(0) {
                self.zero_xmm(dst);
            } else {
                let reg = match self.reg_of(x) {
                    Some(r) if self.ty(x).bytes() >= 4 => r,
                    _ => {
                        self.get(Reg::Rax, x, Ext::Zero);
                        Reg::Rax
                    }
                };
                self.emit_mov_xmm_gpr(dst, reg, lane == Lane::I64);
            }
            self.fput(result, dst);
            return Ok(());
        }
        if self.features.avx && (lane == Lane::I16 || self.features.sse41) {
            let reg = self.reg_of(x).unwrap_or_else(|| {
                self.get(Reg::Rax, x, Ext::Zero);
                Reg::Rax
            });
            let sv = self.fin(v, dst);
            self.vpinsr(lane, (dst, sv), reg, idx);
            self.fput(result, dst);
            return Ok(());
        }
        self.fget(dst, v);
        self.get(Reg::Rax, x, Ext::Zero);
        match lane {
            Lane::I16 => self
                .asm
                .sse_imm(&[0x66], &[0x0F, 0xC4], dst.0, Reg::Rax, idx),
            Lane::I8 if self.features.sse41 => {
                self.asm
                    .sse_imm(&[0x66], &[0x0F, 0x3A, 0x20], dst.0, Reg::Rax, idx)
            }
            Lane::I32 if self.features.sse41 => {
                self.asm
                    .sse_imm(&[0x66], &[0x0F, 0x3A, 0x22], dst.0, Reg::Rax, idx)
            }
            Lane::I64 if self.features.sse41 => {
                self.asm
                    .sse_wide_imm(&[0x66], &[0x0F, 0x3A, 0x22], dst.0, Reg::Rax, idx)
            }
            _ => self.insert_stack(lane, count, dst, idx),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn insert_stack(&mut self, lane: Lane, count: u8, dst: Xmm, idx: u8) {
        self.reserve_scratch(16);
        self.asm
            .sse(&[0xF3], &[0x0F, 0x7F], dst.0, Mem::new(Reg::Rsp, 0));
        self.asm.store(
            Size::of(lane.bytes()),
            Mem::new(Reg::Rsp, i32::from(idx) * lane.bytes() as i32),
            Reg::Rax,
        );
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(16);
    }
}
