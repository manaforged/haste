use super::asm::{Alu, Cond, Reg, Size, Vreg, ZR};
use super::vector::{V2, V3, VecSpec};
use super::{A, C, D, Emit, IP, V0, V1};
use crate::ir::{Lane, ReduceOp, ShuffleMask, TargetOpId, TinyArgs, Value};

#[derive(Clone, Copy)]
enum TargetKind {
    TwoInPlace,
    ThreeInPlace,
    ThreeReg,
    OneReg,
}

struct TargetDesc {
    id: TargetOpId,
    base: u32,
    kind: TargetKind,
    sized: bool,
}

const ARM64_TARGET_OPS: &[TargetDesc] = &[
    TargetDesc { id: TargetOpId::Arm64Umaxp, base: 0x2E20_A400, kind: TargetKind::ThreeReg, sized: true },
    TargetDesc { id: TargetOpId::Arm64Sqdmulh, base: 0x0E20_B400, kind: TargetKind::ThreeReg, sized: true },
    TargetDesc { id: TargetOpId::Arm64Sha256H, base: 0x5E00_4000, kind: TargetKind::ThreeInPlace, sized: false },
    TargetDesc { id: TargetOpId::Arm64Sha256H2, base: 0x5E00_5000, kind: TargetKind::ThreeInPlace, sized: false },
    TargetDesc { id: TargetOpId::Arm64Sha256Su1, base: 0x5E00_6000, kind: TargetKind::ThreeInPlace, sized: false },
    TargetDesc { id: TargetOpId::Arm64Sha256Su0, base: 0x5E28_2800, kind: TargetKind::TwoInPlace, sized: false },
    TargetDesc { id: TargetOpId::Arm64AesE, base: 0x4E28_4800, kind: TargetKind::TwoInPlace, sized: false },
    TargetDesc { id: TargetOpId::Arm64AesD, base: 0x4E28_5800, kind: TargetKind::TwoInPlace, sized: false },
    TargetDesc { id: TargetOpId::Arm64AesMc, base: 0x4E28_6800, kind: TargetKind::OneReg, sized: false },
    TargetDesc { id: TargetOpId::Arm64AesImc, base: 0x4E28_7800, kind: TargetKind::OneReg, sized: false },
];

impl Emit<'_> {
    pub(super) fn vec_shuffle(&mut self, a: Value, b: Value, mask: ShuffleMask, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(result))?;
        let dst = self.fdst(result);
        match mask {
            ShuffleMask::Dyn(idx) => {
                let vi = self.fin(idx, V2);
                if a == b {
                    let va = self.fin(a, V0);
                    self.asm.vbit(0x0E00_0000 | (spec.q << 30), dst, va, vi);
                    self.fput(result, dst);
                    return Ok(());
                }
                let va = self.fin(a, V0);
                let vb = self.fin(b, V1);
                if spec.q == 0 {
                    self.asm.vmov(V0, va);
                    self.ins_elem(V0, 3, 1, vb, 0);
                    self.asm.vbit(0x0E00_0000, dst, V0, vi);
                    self.fput(result, dst);
                } else {
                    self.asm.vmov(V0, va);
                    self.asm.vmov(V1, vb);
                    let out = self.vreg_of(result).unwrap_or(V3);
                    self.asm.vbit(0x4E00_2000, out, V0, vi);
                    self.fput(result, out);
                }
                Ok(())
            }
            ShuffleMask::Const(m) => self.vec_shuffle_const(spec, a, b, m, result),
        }
    }

    fn vec_shuffle_const(&mut self, spec: VecSpec, a: Value, b: Value, m: [u8; 16], result: Value) -> Result<(), String> {
        let n = usize::from(spec.count);
        let eb = spec.lane.bytes() as usize;
        let only_a = m[..n].iter().all(|&x| usize::from(x) < n);
        let only_b = m[..n].iter().all(|&x| usize::from(x) >= n);
        if only_a || only_b {
            let src_val = if only_a { a } else { b };
            let off = if only_a { 0 } else { spec.count };
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            if m[..n].iter().enumerate().all(|(i, &x)| usize::from(x - off) == i) {
                self.asm.vmov(dst, src);
            } else if m[..n].iter().all(|&x| x == m[0]) {
                self.dup_vec_elem(spec.q, spec.size, m[0] - off, dst, src);
            } else {
                self.load_byte_mask(n, eb, off, &m);
                self.asm.vbit(0x0E00_0000 | (spec.q << 30), dst, src, V2);
            }
            self.fput(result, dst);
            return Ok(());
        }
        let va = self.fin(a, V0);
        let vb = self.fin(b, V1);
        let k = usize::from(m[0]);
        if k < n && m[..n].iter().enumerate().all(|(i, &x)| usize::from(x) == k + i) {
            let dst = self.fdst(result);
            let byte_off = (k * eb) as u32;
            self.asm.vbit(0x2E00_0000 | (spec.q << 30) | (byte_off << 11), dst, va, vb);
            self.fput(result, dst);
            return Ok(());
        }
        self.load_byte_mask(n, eb, 0, &m);
        if spec.q == 0 {
            self.asm.vmov(V0, va);
            self.ins_elem(V0, 3, 1, vb, 0);
            let out = self.vreg_of(result).unwrap_or(V3);
            self.asm.vbit(0x0E00_0000, out, V0, V2);
            self.fput(result, out);
        } else {
            self.asm.vmov(V0, va);
            self.asm.vmov(V1, vb);
            let out = self.vreg_of(result).unwrap_or(V3);
            self.asm.vbit(0x4E00_2000, out, V0, V2);
            self.fput(result, out);
        }
        Ok(())
    }

    fn load_byte_mask(&mut self, n: usize, eb: usize, sub_lanes: u8, m: &[u8; 16]) {
        let mut bytes = [0u8; 16];
        for (i, &lane_idx) in m[..n].iter().enumerate() {
            let base = usize::from(lane_idx - sub_lanes) * eb;
            for k in 0..eb {
                bytes[i * eb + k] = (base + k) as u8;
            }
        }
        let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = bytes;
        let lo = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        self.asm.mov_imm(A, lo as i64);
        self.asm.fmov_in(true, V2, A);
        if n * eb == 16 {
            let hi = u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]);
            self.asm.mov_imm(C, hi as i64);
            self.ins_gpr(V2, 3, 1, C);
        }
    }

    pub(super) fn vec_extract(&mut self, vec: Value, idx: u8, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(vec))?;
        let src = self.fin(vec, V0);
        if spec.lane.is_float() {
            let dst = self.fdst(result);
            if idx == 0 {
                self.asm.fmov_reg(spec.lane == Lane::F64, dst, src);
            } else {
                self.dup_scalar_elem(spec.size, idx, dst, src);
            }
            self.fput(result, dst);
        } else {
            let dst = self.dst(result);
            self.umov_gpr(spec.size, idx, dst, src);
            self.put(result, dst);
        }
        Ok(())
    }

    pub(super) fn vec_insert(&mut self, vec: Value, val: Value, idx: u8, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(vec))?;
        let dst = self.fdst(result);
        let src = self.fin(vec, dst);
        if spec.lane.is_float() {
            let mut s = self.fin(val, V1);
            if s == dst && src != dst {
                self.asm.vmov(V1, s);
                s = V1;
            }
            self.asm.vmov(dst, src);
            self.ins_elem(dst, spec.size, idx, s, 0);
        } else {
            let s = self.bits(val, A);
            self.asm.vmov(dst, src);
            self.ins_gpr(dst, spec.size, idx, s);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn vec_reduce(&mut self, op: ReduceOp, vec: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(vec))?;
        let src = self.fin(vec, V0);
        if op == ReduceOp::Movemask {
            let dst = self.dst(result);
            self.vec_movemask(spec, dst, src);
            self.put(result, dst);
            return Ok(());
        }
        if spec.is_1d() {
            return self.vec_extract(vec, 0, result);
        }
        match op {
            ReduceOp::And | ReduceOp::Or | ReduceOp::Xor => {
                self.asm.vmov(V2, src);
                let mut stride = spec.lane.bytes() * u32::from(spec.count) / 2;
                let bit_op = match op {
                    ReduceOp::And => 0x0E20_1C00,
                    ReduceOp::Or => 0x0EA0_1C00,
                    _ => 0x2E20_1C00,
                } | (spec.q << 30);
                while stride >= spec.lane.bytes() {
                    self.asm.vbit(0x2E00_0000 | (spec.q << 30) | (stride << 11), V3, V2, V2);
                    self.asm.vbit(bit_op, V2, V2, V3);
                    stride /= 2;
                }
                let dst = self.dst(result);
                self.umov_gpr(spec.size, 0, dst, V2);
                self.put(result, dst);
            }
            ReduceOp::Add if spec.count == 2 => {
                self.asm.vbit(0x0E20_BC00 | (spec.q << 30) | (spec.size << 22), V2, src, src);
                let dst = self.dst(result);
                self.umov_gpr(spec.size, 0, dst, V2);
                self.put(result, dst);
            }
            ReduceOp::Add => {
                self.v_two(0x0E31_B800 | (spec.q << 30) | (spec.size << 22), V2, src);
                let dst = self.dst(result);
                self.umov_gpr(spec.size, 0, dst, V2);
                self.put(result, dst);
            }
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax if spec.count == 2 => {
                self.dup_scalar_elem(spec.size, 1, V2, src);
                if spec.lane == Lane::I64 {
                    self.asm.fmov_out(true, A, src);
                    self.asm.fmov_out(true, C, V2);
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
                } else {
                    let base = match op {
                        ReduceOp::Smax => 0x0E20_6400,
                        ReduceOp::Smin => 0x0E20_6C00,
                        ReduceOp::Umax => 0x2E20_6400,
                        _ => 0x2E20_6C00,
                    };
                    self.asm.vbit(base | (spec.size << 22), V2, src, V2);
                    let dst = self.dst(result);
                    self.umov_gpr(spec.size, 0, dst, V2);
                    self.put(result, dst);
                }
            }
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax => {
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
            ReduceOp::Fmin | ReduceOp::Fmax => {
                let dst = self.fdst(result);
                if spec.count == 2 {
                    self.dup_scalar_elem(spec.size, 1, V2, src);
                    let base = if op == ReduceOp::Fmin { 0x1E20_7800 } else { 0x1E20_6800 };
                    self.asm.vbit(base | (spec.fsz() << 22), dst, src, V2);
                } else {
                    let base = if op == ReduceOp::Fmin { 0x6EB0_C800 } else { 0x6E30_C800 };
                    self.v_two(base, dst, src);
                }
                self.fput(result, dst);
            }
            ReduceOp::Movemask => unreachable!(),
        }
        Ok(())
    }

    fn vec_movemask(&mut self, spec: VecSpec, dst: Reg, source: Vreg) {
        if spec.lane == Lane::I8 {
            self.asm.fmov_out(true, A, source);
            self.asm.mov_imm(D, 0x0101_0101_0101_0101_u64 as i64);
            self.asm.mov_imm(IP, 0x0102_0408_1020_4080_u64 as i64);
            if spec.count == 8 {
                self.asm.lsr_imm(A, A, 7);
                self.asm.alu(Alu::And, Size::X, A, A, D);
                self.asm.madd(false, A, (A, IP), ZR);
                self.asm.lsr_imm(dst, A, 56);
                return;
            }
            self.asm.umov_d1(C, source);
            for reg in [A, C] {
                self.asm.lsr_imm(reg, reg, 7);
                self.asm.alu(Alu::And, Size::X, reg, reg, D);
                self.asm.madd(false, reg, (reg, IP), ZR);
                self.asm.lsr_imm(reg, reg, 56);
            }
            self.asm.lsl_imm(C, C, 8);
            self.asm.alu(Alu::Orr, Size::X, dst, A, C);
            return;
        }
        self.asm.mov_imm(dst, 0);
        for i in 0..spec.count {
            match spec.lane {
                Lane::I16 => {
                    self.umov_gpr(1, i, C, source);
                    self.asm.lsr_imm(C, C, 15);
                }
                Lane::I32 | Lane::F32 => {
                    self.umov_gpr(2, i, C, source);
                    self.asm.lsr_imm(C, C, 31);
                }
                Lane::I64 | Lane::F64 => {
                    self.umov_gpr(3, i, C, source);
                    self.asm.lsr_imm(C, C, 63);
                }
                Lane::I8 => unreachable!(),
            }
            if i > 0 {
                self.asm.lsl_imm(C, C, i);
            }
            self.asm.alu(Alu::Orr, Size::X, dst, dst, C);
        }
    }

    pub(super) fn vec_target_op(&mut self, id: TargetOpId, args: TinyArgs, result: Value) -> Result<(), String> {
        let Some(desc) = ARM64_TARGET_OPS.iter().find(|d| d.id == id) else {
            return Err(format!("unsupported ARM64 target op {id:?}"));
        };
        let args = args.as_slice();
        let spec = VecSpec::of(self.ty(result))?;
        let base = if desc.sized { desc.base | (spec.q << 30) | (spec.size << 22) } else { desc.base };
        let dst = self.fdst(result);
        match desc.kind {
            TargetKind::OneReg => {
                let a = self.fin(args[0], V0);
                self.v_two(base, dst, a);
            }
            TargetKind::ThreeReg => {
                let a = self.fin(args[0], V0);
                let b = self.fin(args[1], V1);
                self.asm.vbit(base, dst, a, b);
            }
            TargetKind::TwoInPlace => {
                let a = self.fin(args[0], V0);
                let mut b = self.fin(args[1], V1);
                if b == dst && a != dst {
                    self.asm.vmov(V1, b);
                    b = V1;
                }
                self.asm.vmov(dst, a);
                self.v_two(base, dst, b);
            }
            TargetKind::ThreeInPlace => {
                let a = self.fin(args[0], V0);
                let mut b = self.fin(args[1], V1);
                let mut c = self.fin(args[2], V2);
                if dst != a {
                    if b == dst {
                        self.asm.vmov(V1, b);
                        b = V1;
                    }
                    if c == dst {
                        self.asm.vmov(V2, c);
                        c = V2;
                    }
                    self.asm.vmov(dst, a);
                }
                self.asm.vbit(base, dst, b, c);
            }
        }
        self.fput(result, dst);
        Ok(())
    }
}
