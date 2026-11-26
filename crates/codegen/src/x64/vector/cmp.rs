use super::vec_prefix;
use crate::ir::{FloatCC, IntCC, Lane, Value};
use crate::x64::Emit;
use crate::x64::asm::Xmm;
use crate::x64::float::{X0, X1, X2};

impl Emit<'_> {
    pub(in crate::x64) fn vector_icmp(
        &mut self,
        cc: IntCC,
        x: Value,
        y: Value,
        result: Value,
    ) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in vector_icmp", self.ty(x)));
        };
        if self.ty(x).is_wide_vec() {
            return Err(format!("{cc:?} of 256-bit vector {lane:?}"));
        }
        let dst = self.xmm_of(result).unwrap_or(X2);
        let unsigned = cc.is_unsigned();
        if lane == Lane::I64 && !self.features.sse41 {
            self.scalarize_int_icmp(cc, lane, count, x, y, dst);
        } else if self.features.avx && !unsigned {
            let ax = self.fin(x, X0);
            let bx = self.fin(y, X1);
            self.emit_icmp_regs(cc, lane, ax, bx, dst);
        } else {
            self.fget(X0, x);
            self.fget(X1, y);
            self.emit_icmp_regs(cc, lane, X0, X1, dst);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(in crate::x64) fn emit_icmp_regs(
        &mut self,
        cc: IntCC,
        lane: Lane,
        a: Xmm,
        b: Xmm,
        dst: Xmm,
    ) {
        let unsigned = cc.is_unsigned();
        if unsigned {
            let sign = match lane {
                Lane::I8 => 0x80_i64,
                Lane::I16 => 0x8000_i64,
                Lane::I32 => 0x8000_0000_u32 as i64,
                _ => i64::MIN,
            };
            self.splat_imm(lane, dst, sign);
            self.asm.sse(&[0x66], &[0x0F, 0xEF], a.0, dst);
            self.asm.sse(&[0x66], &[0x0F, 0xEF], b.0, dst);
            self.clobber_xmms(&[a, b]);
        }
        let eq_opc: &[u8] = match lane {
            Lane::I8 => &[0x0F, 0x74],
            Lane::I16 => &[0x0F, 0x75],
            Lane::I32 => &[0x0F, 0x76],
            _ => &[0x0F, 0x38, 0x29],
        };
        let gt_opc: &[u8] = match lane {
            Lane::I8 => &[0x0F, 0x64],
            Lane::I16 => &[0x0F, 0x65],
            Lane::I32 => &[0x0F, 0x66],
            _ => &[0x0F, 0x38, 0x37],
        };
        let (equal, swap, invert) = cc.packed();
        let opc = if equal { eq_opc } else { gt_opc };
        let (left, right) = if swap { (b, a) } else { (a, b) };
        self.sse_or_vex(&[0x66], opc, (dst, left, right));
        if invert {
            self.sse_or_vex(&[0x66], &[0x0F, 0x76], (X1, X1, X1));
            self.sse_or_vex(&[0x66], &[0x0F, 0xEF], (dst, dst, X1));
        }
    }

    pub(in crate::x64) fn vector_fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let Some((lane, _)) = self.ty(x).vec_shape() else {
            return;
        };
        let (left, right, imm) = match cc {
            FloatCC::Equal => (x, y, 0),
            FloatCC::LessThan => (x, y, 1),
            FloatCC::LessThanOrEqual => (x, y, 2),
            FloatCC::NotEqual => (x, y, 4),
            FloatCC::GreaterThan => (y, x, 1),
            FloatCC::GreaterThanOrEqual => (y, x, 2),
        };
        let dst = self.fdst(result);
        if self.features.avx {
            let sx = self.fin(left, X0);
            let sy = self.fin(right, X1);
            let pp = if lane == Lane::F64 { 0b01 } else { 0b00 };
            self.asm.vex_xmm_imm(0b00001, pp, 0xC2, (dst, sx), sy, imm);
        } else {
            let [_, src] = self.fins(dst, true, [left, right], &[]);
            self.asm
                .sse_imm(vec_prefix(lane), &[0x0F, 0xC2], dst.0, src, imm);
        }
        self.fput(result, dst);
    }

    pub(in crate::x64) fn blend_regs(&mut self, mask: Xmm, a: Xmm, b: Xmm, dst: Xmm) {
        self.asm.movaps(X0, a);
        self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, mask);
        self.asm.sse(&[0x66], &[0x0F, 0xDF], mask.0, b);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], mask.0, X0);
        if dst != mask {
            self.asm.movaps(dst, mask);
        }
    }

    pub(in crate::x64) fn vector_select(&mut self, cond: Value, x: Value, y: Value, result: Value) {
        let dst = self.xmm_of(result).unwrap_or(X2);
        if self.features.avx {
            let [sy, mask, sx] = self.fins(dst, false, [y, cond, x], &[]);
            self.asm
                .vex_xmm_imm(0b00011, 0b01, 0x4C, (dst, sy), sx, mask.0 << 4);
        } else if self.features.sse41 {
            self.fget(X0, cond);
            let [_, src] = self.fins(dst, true, [y, x], &[]);
            self.asm.sse(&[0x66], &[0x0F, 0x38, 0x10], dst.0, src);
        } else {
            self.fget(X1, cond);
            self.fget(X0, x);
            self.fget(dst, y);
            self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1);
            self.asm.sse(&[0x66], &[0x0F, 0xDF], X1.0, dst);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], X1.0, X0);
            self.asm.movaps(dst, X1);
        }
        self.fput(result, dst);
    }
}
