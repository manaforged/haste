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
        let has_i64_cmp = if cc.packed().0 {
            self.features.sse41
        } else {
            self.features.sse42
        };
        if lane == Lane::I64 && !has_i64_cmp {
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

#[cfg(test)]
mod tests {
    use super::super::super::compile_with_features;
    use super::super::super::features::Features;
    use crate::ir::{AbiParam, CallConv, Function, IntCC, Lane, Signature, Type, Value};
    use crate::{Effort, FunctionBuilder, Target};

    fn x64_full() -> Target {
        let mut target = *Target::find("x86_64-unknown-linux-gnu").expect("x64 linux target");
        target.effort = Effort::Full;
        target
    }

    fn build_i64x2_fn(
        emit_op: impl FnOnce(crate::builder::Ins<'_>, Value, Value) -> Value,
    ) -> Function {
        let mut sig = Signature::new(CallConv::Native);
        sig.params = vec![
            AbiParam::new(Type::I64),
            AbiParam::new(Type::I64),
            AbiParam::new(Type::I64),
        ];
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let p = b.block_params(entry).to_vec();
            b.switch_to_block(entry);
            let vec_ty = Type::Vec {
                lane: Lane::I64,
                count: 2,
            };
            let va = b.ins().load(vec_ty, p[0], 0);
            let vb = b.ins().load(vec_ty, p[1], 0);
            let res = emit_op(b.ins(), va, vb);
            b.ins().store(res, p[2], 0);
            b.ins().return_(&[]);
        }
        func
    }

    #[test]
    fn i64x2_pcmpgtq_is_gated_on_sse42_not_sse41() {
        let target = x64_full();
        let sse41_only = Features {
            sse41: true,
            ..Features::default()
        };
        let with_sse42 = Features {
            sse41: true,
            sse42: true,
            ..Features::default()
        };

        let slt = build_i64x2_fn(|ins, a, b| ins.icmp(IntCC::SignedLessThan, a, b));
        let sgt = build_i64x2_fn(|ins, a, b| ins.icmp(IntCC::SignedGreaterThan, a, b));
        let smin = build_i64x2_fn(|ins, a, b| ins.smin(a, b));
        let smax = build_i64x2_fn(|ins, a, b| ins.smax(a, b));
        let eq = build_i64x2_fn(|ins, a, b| ins.icmp(IntCC::Equal, a, b));

        for (name, func) in [
            ("slt", &slt),
            ("sgt", &sgt),
            ("smin", &smin),
            ("smax", &smax),
        ] {
            let no_42 = compile_with_features(func, &target, sse41_only).expect("sse41 compiles");
            assert!(
                !no_42
                    .bytes()
                    .windows(4)
                    .any(|w| w == [0x66, 0x0F, 0x38, 0x37]),
                "{name} must not emit pcmpgtq without sse4.2: {:02x?}",
                no_42.bytes()
            );
            let yes_42 = compile_with_features(func, &target, with_sse42).expect("sse42 compiles");
            assert!(
                yes_42
                    .bytes()
                    .windows(4)
                    .any(|w| w == [0x66, 0x0F, 0x38, 0x37]),
                "{name} should emit pcmpgtq when sse4.2 is enabled: {:02x?}",
                yes_42.bytes()
            );
        }

        let eq_code = compile_with_features(&eq, &target, sse41_only).expect("eq compiles");
        assert!(
            eq_code
                .bytes()
                .windows(4)
                .any(|w| w == [0x66, 0x0F, 0x38, 0x29]),
            "pcmpeqq only requires sse4.1: {:02x?}",
            eq_code.bytes()
        );
    }

    #[cfg(all(unix, target_arch = "x86_64"))]
    #[test]
    fn i64x2_sse41_without_sse42_executes_correctly() {
        let target = x64_full();
        let sse41_only = Features {
            sse41: true,
            ..Features::default()
        };
        let mut mem = crate::Memory::default();
        let slt_code = compile_with_features(
            &build_i64x2_fn(|ins, a, b| ins.icmp(IntCC::SignedLessThan, a, b)),
            &target,
            sse41_only,
        )
        .expect("slt compiles");
        let sgt_code = compile_with_features(
            &build_i64x2_fn(|ins, a, b| ins.icmp(IntCC::SignedGreaterThan, a, b)),
            &target,
            sse41_only,
        )
        .expect("sgt compiles");
        let smin_code = compile_with_features(
            &build_i64x2_fn(|ins, a, b| ins.smin(a, b)),
            &target,
            sse41_only,
        )
        .expect("smin compiles");
        let smax_code = compile_with_features(
            &build_i64x2_fn(|ins, a, b| ins.smax(a, b)),
            &target,
            sse41_only,
        )
        .expect("smax compiles");

        let slt_addr = mem.write_patched(&slt_code, &[]).expect("map slt");
        let sgt_addr = mem.write_patched(&sgt_code, &[]).expect("map sgt");
        let smin_addr = mem.write_patched(&smin_code, &[]).expect("map smin");
        let smax_addr = mem.write_patched(&smax_code, &[]).expect("map smax");

        type VecBinFn = extern "C" fn(*const [i64; 2], *const [i64; 2], *mut [i64; 2]);
        // SAFETY: each compiled function has SysV x86-64 `CallConv::Native` signature `(*const [i64; 2], *const [i64; 2], *mut [i64; 2]) -> ()`.
        let (run_slt, run_sgt, run_smin, run_smax): (VecBinFn, VecBinFn, VecBinFn, VecBinFn) = unsafe {
            (
                std::mem::transmute(slt_addr),
                std::mem::transmute(sgt_addr),
                std::mem::transmute(smin_addr),
                std::mem::transmute(smax_addr),
            )
        };

        let cases = [
            ([i64::MIN, 42], [i64::MAX, -5]),
            ([7, -7], [7, -7]),
            ([-1, 0], [0, -1]),
        ];
        for (a, b) in cases {
            let mut out = [0_i64; 2];
            run_slt(&a, &b, &mut out);
            assert_eq!(
                out,
                [
                    if a[0] < b[0] { -1 } else { 0 },
                    if a[1] < b[1] { -1 } else { 0 }
                ]
            );
            run_sgt(&a, &b, &mut out);
            assert_eq!(
                out,
                [
                    if a[0] > b[0] { -1 } else { 0 },
                    if a[1] > b[1] { -1 } else { 0 }
                ]
            );
            run_smin(&a, &b, &mut out);
            assert_eq!(out, [a[0].min(b[0]), a[1].min(b[1])]);
            run_smax(&a, &b, &mut out);
            assert_eq!(out, [a[0].max(b[0]), a[1].max(b[1])]);
        }
    }
}
