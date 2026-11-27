use super::super::asm::{Alu, Cond, Reg, Shift, Size, Vreg, ZR};
use super::super::int::encode_bitmask_imm;
use super::super::vector::{V2, V3, VecSpec};
use super::super::{A, C, D, Emit, IP, V0};
use super::{NEON_AND, NEON_EOR, NEON_EXT, NEON_ORR, NEON_SMAX, NEON_SMIN, NEON_UMAX, NEON_UMIN};
use crate::ir::{Binary, Convert, Function, Inst, Lane, Op, ReduceOp, Unary, Value};

impl Emit<'_> {
    pub(in super::super) fn vec_reduce(
        &mut self,
        op: ReduceOp,
        vec: Value,
        result: Value,
    ) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(vec))?;
        let src = self.fin(vec, V0);
        if op == ReduceOp::Movemask {
            let dst = self.dst(result);
            self.vec_movemask(spec, dst, src, vec);
            self.put(result, dst);
            return Ok(());
        }
        if spec.is_1d() {
            return self.vec_extract(vec, 0, result);
        }
        match op {
            ReduceOp::And | ReduceOp::Or | ReduceOp::Xor => {
                self.vec_reduce_bitwise(op, spec, src, result)
            }
            ReduceOp::Add => self.vec_reduce_add(spec, src, result),
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax
                if spec.count == 2 =>
            {
                self.vec_reduce_minmax_pair(op, spec, src, result);
            }
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax => {
                self.vec_reduce_minmax(op, spec, src, result)
            }
            ReduceOp::Fmin | ReduceOp::Fmax => self.vec_reduce_float(op, spec, src, result),
            ReduceOp::Movemask => unreachable!(),
        }
        Ok(())
    }

    fn vec_reduce_add(&mut self, spec: VecSpec, src: Vreg, result: Value) {
        let tail = (spec.q << 30) | (spec.size << 22);
        if spec.count == 2 {
            self.asm.vbit(0x0E20_BC00 | tail, V2, src, src);
        } else {
            self.v_two(0x0E31_B800 | tail, V2, src);
        }
        let dst = self.dst(result);
        self.umov_gpr(spec.size, 0, dst, V2);
        self.put(result, dst);
    }

    fn vec_reduce_minmax(&mut self, op: ReduceOp, spec: VecSpec, src: Vreg, result: Value) {
        let base = match op {
            ReduceOp::Smax => 0x0E30_A800,
            ReduceOp::Smin => 0x0E31_A800,
            ReduceOp::Umax => 0x2E30_A800,
            _ => 0x2E31_A800,
        };
        self.v_two(base | (spec.q << 30) | (spec.size << 22), V2, src);
        let dst = self.dst(result);
        self.umov_gpr(spec.size, 0, dst, V2);
        self.put(result, dst);
    }

    fn vec_reduce_float(&mut self, op: ReduceOp, spec: VecSpec, src: Vreg, result: Value) {
        let dst = self.fdst(result);
        if spec.count == 2 {
            self.dup_scalar_elem(spec.size, 1, V2, src);
            let base = if op == ReduceOp::Fmin {
                0x1E20_7800
            } else {
                0x1E20_6800
            };
            self.asm.vbit(base | (spec.fsz() << 22), dst, src, V2);
        } else {
            let base = if op == ReduceOp::Fmin {
                0x6EB0_C800
            } else {
                0x6E30_C800
            };
            self.v_two(base, dst, src);
        }
        self.fput(result, dst);
    }

    fn vec_reduce_bitwise(&mut self, op: ReduceOp, spec: VecSpec, src: Vreg, result: Value) {
        let bit_op = match op {
            ReduceOp::And => NEON_AND,
            ReduceOp::Or => NEON_ORR,
            _ => NEON_EOR,
        } | (spec.q << 30);
        let mut cur = src;
        let mut stride = spec.lane.bytes() * u32::from(spec.count) / 2;
        while stride >= spec.lane.bytes() {
            self.asm
                .vbit(NEON_EXT | (spec.q << 30) | (stride << 11), V3, cur, cur);
            self.asm.vbit(bit_op, V2, cur, V3);
            cur = V2;
            stride /= 2;
        }
        let dst = self.dst(result);
        self.umov_gpr(spec.size, 0, dst, V2);
        self.put(result, dst);
    }

    fn vec_reduce_minmax_pair(&mut self, op: ReduceOp, spec: VecSpec, src: Vreg, result: Value) {
        if spec.lane == Lane::I64 {
            self.asm.fmov_out(true, A, src);
            self.asm.umov_d1(C, src);
            self.asm.alu(Alu::Subs, Size::X, ZR, A, C);
            let cond = match op {
                ReduceOp::Smin => Cond::Lt,
                ReduceOp::Smax => Cond::Gt,
                ReduceOp::Umin => Cond::Lo,
                _ => Cond::Hi,
            };
            let dst = self.dst(result);
            self.asm.csel(cond, dst, A, C);
            self.put(result, dst);
            return;
        }
        self.dup_scalar_elem(spec.size, 1, V2, src);
        let base = match op {
            ReduceOp::Smax => NEON_SMAX,
            ReduceOp::Smin => NEON_SMIN,
            ReduceOp::Umax => NEON_UMAX,
            _ => NEON_UMIN,
        };
        self.asm.vbit(base | (spec.size << 22), V2, src, V2);
        let dst = self.dst(result);
        self.umov_gpr(spec.size, 0, dst, V2);
        self.put(result, dst);
    }

    fn vec_movemask(&mut self, spec: VecSpec, dst: Reg, source: Vreg, val: Value) {
        match spec.lane {
            Lane::I8 | Lane::I16 => {
                if spec.lane == Lane::I8
                    && spec.q == 1
                    && is_byte_bool_mask(self.func, &self.defs, val, 4)
                    && self.movemask_shrn(dst, source)
                {
                    return;
                }
                self.movemask_mul(spec, dst, source);
            }
            Lane::I32 | Lane::F32 => self.movemask_i32(spec.count, dst, source),
            Lane::I64 | Lane::F64 => {
                if spec.count == 1 {
                    self.asm.fmov_out(true, dst, source);
                    self.asm.lsr_imm(dst, dst, 63);
                } else {
                    self.asm.fmov_out(true, A, source);
                    self.asm.umov_d1(C, source);
                    self.asm.lsr_imm(dst, A, 63);
                    self.asm.lsr_imm(C, C, 63);
                    self.asm
                        .alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 1, dst, dst, C);
                }
            }
        }
    }

    fn movemask_shrn(&mut self, dst: Reg, source: Vreg) -> bool {
        let (Some(m1), Some(m2), Some(m3), Some(m4)) = (
            encode_bitmask_imm(0x1111_1111_1111_1111, true),
            encode_bitmask_imm(0x0303_0303_0303_0303, true),
            encode_bitmask_imm(0x000F_000F_000F_000F, true),
            encode_bitmask_imm(0xFFFF, true),
        ) else {
            return false;
        };
        self.clobber_vreg(V2);
        self.v_two(0x0F0C_8400, V2, source);
        self.asm.fmov_out(true, A, V2);
        self.clobber_reg(A);
        self.asm.alu_bitmask(Alu::And, Size::X, m1, A, A);
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsr, 3, A, A, A);
        self.asm.alu_bitmask(Alu::And, Size::X, m2, A, A);
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsr, 6, A, A, A);
        self.asm.alu_bitmask(Alu::And, Size::X, m3, A, A);
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsr, 12, A, A, A);
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsr, 24, A, A, A);
        self.asm.alu_bitmask(Alu::And, Size::X, m4, dst, A);
        true
    }

    fn movemask_mul(&mut self, spec: VecSpec, dst: Reg, source: Vreg) {
        let (mask, mul, lsb_shift, top_shift, half_bits) = if spec.lane == Lane::I8 {
            (
                0x0101_0101_0101_0101_u64,
                0x0102_0408_1020_4080_u64,
                7_u8,
                56_u8,
                8_u8,
            )
        } else {
            (
                0x0001_0001_0001_0001_u64,
                0x0001_0002_0004_0008_u64,
                15_u8,
                48_u8,
                4_u8,
            )
        };
        self.asm.fmov_out(true, A, source);
        self.asm.mov_imm(D, mask as i64);
        self.asm.mov_imm(IP, mul as i64);
        if spec.q == 0 {
            self.asm.lsr_imm(A, A, lsb_shift);
            self.asm.alu(Alu::And, Size::X, A, A, D);
            self.asm.madd(false, Size::X, A, (A, IP), ZR);
            self.asm.lsr_imm(dst, A, top_shift);
            return;
        }
        self.asm.umov_d1(C, source);
        for reg in [A, C] {
            self.asm.lsr_imm(reg, reg, lsb_shift);
            self.asm.alu(Alu::And, Size::X, reg, reg, D);
            self.asm.madd(false, Size::X, reg, (reg, IP), ZR);
            self.asm.lsr_imm(reg, reg, top_shift);
        }
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsl, half_bits, dst, A, C);
    }

    fn movemask_i32(&mut self, count: u8, dst: Reg, source: Vreg) {
        self.asm.fmov_out(true, D, source);
        if count == 2 {
            self.asm.lsr_imm(C, D, 63);
            self.asm.bitfield(false, dst, D, (31, 31));
            self.asm
                .alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 1, dst, dst, C);
            return;
        }
        self.asm.umov_d1(C, source);
        self.asm.bitfield(false, dst, D, (31, 31));
        self.asm.lsr_imm(D, D, 63);
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 1, dst, dst, D);
        self.asm.bitfield(false, D, C, (31, 31));
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 2, dst, dst, D);
        self.asm.lsr_imm(C, C, 63);
        self.asm
            .alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 3, dst, dst, C);
    }
}

fn is_byte_bool_mask(func: &Function, defs: &[Option<Inst>], val: Value, depth: u8) -> bool {
    if depth == 0 {
        return false;
    }
    let Some(inst) = defs.get(val.index()).copied().flatten() else {
        return false;
    };
    match func.insts[inst.index()].op {
        Op::Icmp(..) | Op::Fcmp(..) => true,
        Op::Unary(Unary::Bnot, inner) => is_byte_bool_mask(func, defs, inner, depth - 1),
        Op::Binary(Binary::Band | Binary::Bor | Binary::Bxor, a, b) => {
            is_byte_bool_mask(func, defs, a, depth - 1)
                && is_byte_bool_mask(func, defs, b, depth - 1)
        }
        Op::Convert(Convert::Bitcast, inner)
            if func.ty(inner).is_vector() && func.ty(inner).bytes() == func.ty(val).bytes() =>
        {
            is_byte_bool_mask(func, defs, inner, depth - 1)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::ir::{IntCC, Lane, ReduceOp, Type};
    use crate::{
        AbiParam, CallConv, Effort, Function, FunctionBuilder, Signature, Target, compile,
    };

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn arm64_cmp_movemask_uses_shrn_without_madd() {
        let mut target = *Target::find("aarch64-apple-darwin").expect("arm64 target");
        target.effort = Effort::Full;
        let vty = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(vty));
        sig.params.push(AbiParam::new(vty));
        sig.returns.push(AbiParam::new(Type::I32));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let p = b.block_params(entry).to_vec();
            b.switch_to_block(entry);
            let eq = b.ins().icmp(IntCC::Equal, p[0], p[1]);
            let mask = b.ins().reduce(ReduceOp::Movemask, eq);
            b.ins().return_(&[mask]);
        }
        let code = compile(func, &target).expect("arm64 movemask compiles");
        let w = words(code.bytes());
        assert!(
            w.iter().any(|&insn| (insn & 0xFFFF_FC00) == 0x0F0C_8400),
            "expected shrn.8b in {w:#x?}"
        );
        assert!(
            !w.iter().any(|&insn| (insn & 0x7FE0_8000) == 0x1B00_0000),
            "unexpected madd/mul in {w:#x?}"
        );
    }

    #[test]
    fn arm64_arbitrary_byte_movemask_keeps_general_mul_path() {
        let mut target = *Target::find("aarch64-apple-darwin").expect("arm64 target");
        target.effort = Effort::Full;
        let vty = Type::Vec {
            lane: Lane::I8,
            count: 16,
        };
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(vty));
        sig.returns.push(AbiParam::new(Type::I32));
        let mut func = Function::new(sig);
        {
            let mut b = FunctionBuilder::new(&mut func);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            let p = b.block_params(entry)[0];
            b.switch_to_block(entry);
            let mask = b.ins().reduce(ReduceOp::Movemask, p);
            b.ins().return_(&[mask]);
        }
        let code = compile(func, &target).expect("arm64 general movemask compiles");
        let w = words(code.bytes());
        assert!(
            w.iter().any(|&insn| (insn & 0xFFE0_8000) == 0x9B00_0000),
            "expected 64-bit mul/madd for general byte input in {w:#x?}"
        );
    }
}
