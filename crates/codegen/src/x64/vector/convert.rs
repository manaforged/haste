use crate::ir::{Convert, Lane, Type, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Cond, Mem, Reg, Shift, Size, Xmm};
use crate::x64::float::{X0, X1, X2, scalar};

impl Emit<'_> {
    pub(in crate::x64) fn vector_convert(
        &mut self,
        op: Convert,
        x: Value,
        result: Value,
    ) -> Result<(), String> {
        let from_ty = self.ty(x);
        let to_ty = self.ty(result);
        if op == Convert::Bitcast {
            return self.float_convert(op, x, result);
        }
        if from_ty.is_wide_vec() || to_ty.is_wide_vec() {
            return Err(format!("{op:?} between {from_ty:?} and {to_ty:?}"));
        }
        let (Some((fl, fc)), Some((tl, tc))) = (from_ty.vec_shape(), to_ty.vec_shape()) else {
            return Err(format!("{op:?} between {from_ty:?} and {to_ty:?}"));
        };
        if fc != tc {
            return Err(format!("{op:?} lane count mismatch {fc} != {tc}"));
        }
        let dst = self.fdst(result);
        match (op, fl, tl) {
            (Convert::FromSint, Lane::I32, Lane::F32) => {
                let src = self.fin(x, dst);
                self.asm.sse(&[], &[0x0F, 0x5B], dst.0, src);
            }
            (Convert::Fpromote, Lane::F32, Lane::F64) => {
                let src = self.fin(x, dst);
                self.asm.sse(&[], &[0x0F, 0x5A], dst.0, src);
            }
            (Convert::Fdemote, Lane::F64, Lane::F32) => {
                let src = self.fin(x, dst);
                self.asm.sse(&[0x66], &[0x0F, 0x5A], dst.0, src);
            }
            (Convert::Uextend | Convert::Sextend, _, _)
                if self.features.sse41 && fl.is_int() && tl.is_int() =>
            {
                let code = match (fl, tl) {
                    (Lane::I8, Lane::I16) => 0x00,
                    (Lane::I8, Lane::I32) => 0x01,
                    (Lane::I8, Lane::I64) => 0x02,
                    (Lane::I16, Lane::I32) => 0x03,
                    (Lane::I16, Lane::I64) => 0x04,
                    (Lane::I32, Lane::I64) => 0x05,
                    _ => return Err(format!("{op:?} from {fl:?} to {tl:?}")),
                };
                let base = if op == Convert::Sextend { 0x20 } else { 0x30 };
                let src = self.fin(x, dst);
                self.asm
                    .sse(&[0x66], &[0x0F, 0x38, base + code], dst.0, src);
            }
            (Convert::NarrowSat { signed }, Lane::I16, Lane::I8) => {
                self.fget(dst, x);
                self.asm.sse(
                    &[0x66],
                    &[0x0F, if signed { 0x63 } else { 0x67 }],
                    dst.0,
                    dst,
                );
            }
            (Convert::NarrowSat { signed: true }, Lane::I32, Lane::I16) => {
                self.fget(dst, x);
                self.asm.sse(&[0x66], &[0x0F, 0x6B], dst.0, dst);
            }
            (Convert::NarrowSat { signed: false }, Lane::I32, Lane::I16) if self.features.sse41 => {
                self.fget(dst, x);
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x2B], dst.0, dst);
            }
            (Convert::NarrowSat { signed: false }, Lane::I32, Lane::I16) => {
                self.fget(dst, x);
                self.asm.movaps(X2, dst);
                self.zero_xmm(X1);
                self.asm.sse(&[0x66], &[0x0F, 0x66], X2.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xDB], dst.0, X2);
                self.splat_imm(Lane::I32, X1, 0x8000);
                self.asm.sse(&[0x66], &[0x0F, 0xFA], dst.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0x6B], dst.0, dst);
                self.splat_imm(Lane::I16, X1, 0x8000);
                self.asm.sse(&[0x66], &[0x0F, 0xEF], dst.0, X1);
            }
            (Convert::Ireduce, Lane::I16 | Lane::I32 | Lane::I64, _)
                if fl.bytes() == 2 * tl.bytes() =>
            {
                self.narrow(fl, x, dst)
            }
            _ => self.scalarize_convert(op, (fl, tl, fc), (from_ty, to_ty), x, dst)?,
        }
        self.fput(result, dst);
        Ok(())
    }

    fn scalarize_convert(
        &mut self,
        op: Convert,
        (fl, tl, count): (Lane, Lane, u8),
        (from_ty, to_ty): (Type, Type),
        x: Value,
        dst: Xmm,
    ) -> Result<(), String> {
        self.fget(X0, x);
        self.reserve_scratch(32);
        self.xmm_store(from_ty, Mem::new(Reg::Rsp, 16), X0);
        for i in 0..count {
            let src_slot = Mem::new(Reg::Rsp, 16 + i32::from(i) * fl.bytes() as i32);
            let dst_slot = Mem::new(Reg::Rsp, i32::from(i) * tl.bytes() as i32);
            match op {
                Convert::Uextend | Convert::Ireduce | Convert::Sextend => {
                    let sz = Size::of(fl.bytes());
                    if matches!(op, Convert::Sextend) {
                        self.asm.load_signed(sz, Reg::Rax, src_slot);
                    } else {
                        self.asm.load(sz, Reg::Rax, src_slot);
                    }
                    self.asm.store(Size::of(tl.bytes()), dst_slot, Reg::Rax);
                }
                Convert::FromSint => {
                    self.asm
                        .load_signed(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.asm
                        .sse_wide(scalar(tl.to_type()), 0x2A, X0.0, Reg::Rax);
                    self.xmm_store(tl.to_type(), dst_slot, X0);
                }
                Convert::FromUint => {
                    self.asm.load(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.asm
                        .sse_wide(scalar(tl.to_type()), 0x2A, X0.0, Reg::Rax);
                    if fl == Lane::I64 {
                        let (big, done) = (self.asm.label(), self.asm.label());
                        self.asm.test(Reg::Rax, Reg::Rax);
                        self.asm.jcc(Cond::S, big);
                        self.asm.jmp(done);
                        self.asm.bind(big);
                        self.asm.mov(Reg::Rcx, Reg::Rax);
                        self.asm.shift_imm(Size::Q, Shift::Shr, Reg::Rcx, 1);
                        self.asm.alu_imm32(Alu::And, Reg::Rax, 1);
                        self.asm.alu(Alu::Or, Reg::Rcx, Reg::Rax);
                        self.asm
                            .sse_wide(scalar(tl.to_type()), 0x2A, X0.0, Reg::Rcx);
                        self.asm
                            .sse(&[scalar(tl.to_type())], &[0x0F, 0x58], X0.0, X0);
                        self.asm.bind(done);
                    }
                    self.xmm_store(tl.to_type(), dst_slot, X0);
                }
                Convert::ToSintSat | Convert::ToUintSat => {
                    let fl_t = fl.to_type();
                    self.xmm_load(fl_t, X0, src_slot);
                    if fl_t == Type::F32 {
                        self.asm.sse(&[0xF3], &[0x0F, 0x5A], X0.0, X0);
                    }
                    let signed = op == Convert::ToSintSat;
                    if signed {
                        self.to_signed()
                    } else {
                        self.to_unsigned()
                    }
                    self.clamp(tl.to_type(), signed);
                    self.asm.store(Size::of(tl.bytes()), dst_slot, Reg::Rax);
                }
                Convert::NarrowSat { signed } => {
                    self.asm
                        .load_signed(Size::of(fl.bytes()), Reg::Rax, src_slot);
                    self.clamp_signed(Reg::Rax, tl.to_type(), signed);
                    self.asm.store(Size::of(tl.bytes()), dst_slot, Reg::Rax);
                }
                _ => {
                    self.release_scratch(32);
                    return Err(format!("unsupported vector convert {op:?} {fl:?}->{tl:?}"));
                }
            }
        }
        self.xmm_load(to_ty, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(32);
        Ok(())
    }
}
