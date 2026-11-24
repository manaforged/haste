use crate::ir::{Binary, IntCC, Lane, Type, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Cond, Count, Group3, Mem, Reg, Shift, Size, Xmm};
use crate::x64::float::X0;
use crate::x64::int::cond;

impl Emit<'_> {
    fn spill_vec_pair(&mut self, ty: Type, x: Value, y: Value) {
        self.reserve_scratch(32);
        self.fget(X0, x);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 0), X0);
        self.fget(X0, y);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 16), X0);
    }

    pub(in crate::x64) fn scalarize_int_binary(
        &mut self,
        op: Binary,
        lane: Lane,
        count: u8,
        x: Value,
        y: Value,
        dst: Xmm,
    ) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        let signed_load = matches!(op, Binary::Smin | Binary::Smax | Binary::Sshr);
        self.spill_vec_pair(ty, x, y);
        for i in 0..i32::from(count) {
            let sx = Mem::new(Reg::Rsp, i * stride);
            let sy = Mem::new(Reg::Rsp, 16 + i * stride);
            if signed_load {
                self.asm.load_signed(size, Reg::Rax, sx);
                self.asm.load_signed(size, Reg::Rcx, sy);
            } else {
                self.asm.load(size, Reg::Rax, sx);
                self.asm.load(size, Reg::Rcx, sy);
            }
            match op {
                Binary::Imul => self.asm.imul(Reg::Rax, Reg::Rcx),
                Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                    self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
                    let cc = match op {
                        Binary::Smin => Cond::G,
                        Binary::Smax => Cond::L,
                        Binary::Umin => Cond::A,
                        _ => Cond::B,
                    };
                    self.asm.cmov(cc, Reg::Rax, Reg::Rcx);
                }
                Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                    self.asm
                        .alu_imm32(Alu::And, Reg::Rcx, (lane.bits() - 1) as i32);
                    let shift = match op {
                        Binary::Ishl => Shift::Shl,
                        Binary::Ushr => Shift::Shr,
                        _ => Shift::Sar,
                    };
                    self.asm.shift(size, shift, Reg::Rax);
                }
                Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                    self.emit_sat_op(op, size, Reg::Rax, Reg::Rcx)
                }
                _ => {}
            }
            self.asm.store(size, sx, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(32);
    }

    pub(in crate::x64) fn scalarize_int_clz(&mut self, lane: Lane, count: u8, x: Value, dst: Xmm) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        self.fget(X0, x);
        self.reserve_scratch(16);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 0), X0);
        for i in 0..i32::from(count) {
            let slot = Mem::new(Reg::Rsp, i * stride);
            if self.features.lzcnt && lane.bits() >= 32 {
                self.asm.count(Count::Lzcnt, size, Reg::Rax, slot);
            } else {
                self.asm.load(size, Reg::Rax, slot);
                self.count(Count::Lzcnt, Reg::Rax, Reg::Rax);
                if lane.bits() < 64 {
                    self.asm
                        .alu_imm(Alu::Sub, Reg::Rax, (64 - lane.bits()) as i32);
                }
            }
            self.asm.store(size, slot, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(16);
    }

    pub(in crate::x64) fn scalarize_int_icmp(
        &mut self,
        cc: IntCC,
        lane: Lane,
        count: u8,
        x: Value,
        y: Value,
        dst: Xmm,
    ) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        self.spill_vec_pair(ty, x, y);
        for i in 0..i32::from(count) {
            let sx = Mem::new(Reg::Rsp, i * stride);
            let sy = Mem::new(Reg::Rsp, 16 + i * stride);
            self.asm.load(Size::Q, Reg::Rax, sx);
            self.asm.load(Size::Q, Reg::Rcx, sy);
            self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
            self.asm.setcc(cond(cc), Reg::Rax);
            self.asm.load(Size::B, Reg::Rax, Reg::Rax);
            self.asm.group3(Group3::Neg, Reg::Rax);
            self.asm.store(Size::Q, sx, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(32);
    }
}
