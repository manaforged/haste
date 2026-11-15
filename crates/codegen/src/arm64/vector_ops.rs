use super::asm::{Alu, Cond, Reg, Shift, Size, Vreg, ZR};
use super::vector::{V2, V3, VecSpec};
use super::{A, C, D, Emit, IP, V0, V1};
use crate::ir::{Lane, ReduceOp, ShuffleMask, TargetOpId, TinyArgs, Value};

#[derive(Clone, Copy)]
enum TargetKind {
    TwoInPlace,
    ThreeInPlace,
    OneReg,
}

struct TargetDesc {
    id: TargetOpId,
    base: u32,
    kind: TargetKind,
}

const ARM64_TARGET_OPS: &[TargetDesc] = &[
    TargetDesc { id: TargetOpId::Arm64Sha256H, base: 0x5E00_4000, kind: TargetKind::ThreeInPlace },
    TargetDesc { id: TargetOpId::Arm64Sha256H2, base: 0x5E00_5000, kind: TargetKind::ThreeInPlace },
    TargetDesc { id: TargetOpId::Arm64Sha256Su1, base: 0x5E00_6000, kind: TargetKind::ThreeInPlace },
    TargetDesc { id: TargetOpId::Arm64Sha256Su0, base: 0x5E28_2800, kind: TargetKind::TwoInPlace },
    TargetDesc { id: TargetOpId::Arm64AesE, base: 0x4E28_4800, kind: TargetKind::TwoInPlace },
    TargetDesc { id: TargetOpId::Arm64AesD, base: 0x4E28_5800, kind: TargetKind::TwoInPlace },
    TargetDesc { id: TargetOpId::Arm64AesMc, base: 0x4E28_6800, kind: TargetKind::OneReg },
    TargetDesc { id: TargetOpId::Arm64AesImc, base: 0x4E28_7800, kind: TargetKind::OneReg },
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
            ShuffleMask::Const(m) => self.vec_shuffle_const(spec, (a, b), m, result),
        }
    }

    fn vec_shuffle_const(&mut self, spec: VecSpec, (a, b): (Value, Value), m: [u8; 16], result: Value) -> Result<(), String> {
        let n = usize::from(spec.count);
        let eb = spec.lane.bytes() as usize;
        let total = n * eb;
        let mut bidx = [0x80_u8; 16];
        let (mut uses_a, mut uses_b, mut has_zero) = (false, false, false);
        for (i, &lane) in m[..n].iter().enumerate() {
            let x = usize::from(lane);
            let src_start = if x < n {
                uses_a = true;
                x * eb
            } else if x < 2 * n {
                uses_b = true;
                total + (x - n) * eb
            } else {
                has_zero = true;
                continue;
            };
            for k in 0..eb {
                bidx[i * eb + k] = (src_start + k) as u8;
            }
        }
        if !uses_a && !uses_b {
            let dst = self.fdst(result);
            self.asm.veor_zero(dst);
            self.fput(result, dst);
            return Ok(());
        }
        if uses_a ^ uses_b {
            self.shuffle_single_source((spec, eb, total), if uses_a { (a, 0) } else { (b, total as u8) }, has_zero, (&bidx, result));
        } else {
            self.shuffle_two_sources((spec, eb, total), (a, b), has_zero, (&bidx, result));
        }
        Ok(())
    }

    fn shuffle_single_source(
        &mut self,
        (spec, eb, total): (VecSpec, usize, usize),
        (src_val, src_off): (Value, u8),
        has_zero: bool,
        (bidx, result): (&[u8; 16], Value),
    ) {
        let mut ridx = [0x80_u8; 16];
        for j in 0..total {
            if bidx[j] != 0x80 {
                ridx[j] = bidx[j] - src_off;
            }
        }
        if !has_zero && self.shuffle_single_fast((spec, eb, total), src_val, &ridx, result) {
            return;
        }
        self.load_bytes_v2(&ridx, total == 16);
        let dst = self.fdst(result);
        let src = self.fin(src_val, V0);
        self.asm.vbit(0x0E00_0000 | (spec.q << 30), dst, src, V2);
        self.fput(result, dst);
    }

    fn shuffle_two_sources(&mut self, (spec, eb, total): (VecSpec, usize, usize), (a, b): (Value, Value), has_zero: bool, (bidx, result): (&[u8; 16], Value)) {
        let va = self.fin(a, V0);
        let vb = self.fin(b, V1);
        if !has_zero {
            let s = usize::from(bidx[0]);
            let ext_pair = if s < total && (0..total).all(|j| usize::from(bidx[j]) == s + j) {
                Some((va, vb, s))
            } else if (total..2 * total).contains(&s) && (0..total).all(|j| usize::from(bidx[j]) == (s + j) % (2 * total)) {
                Some((vb, va, s - total))
            } else {
                None
            };
            if let Some((vn, vm, byte_off)) = ext_pair {
                let dst = self.fdst(result);
                self.asm.vbit(0x2E00_0000 | (spec.q << 30) | ((byte_off as u32) << 11), dst, vn, vm);
                self.fput(result, dst);
                return;
            }
            if let Some((base, size, swap)) = match_permute(total, eb, bidx, false) {
                let dst = self.fdst(result);
                let (vn, vm) = if swap { (vb, va) } else { (va, vb) };
                self.asm.vbit(base | (spec.q << 30) | (size << 22), dst, vn, vm);
                self.fput(result, dst);
                return;
            }
        }
        self.load_bytes_v2(bidx, total == 16);
        self.asm.vmov(V0, va);
        let out = self.vreg_of(result).unwrap_or(V3);
        if spec.q == 0 {
            self.ins_elem(V0, 3, 1, vb, 0);
            self.asm.vbit(0x0E00_0000, out, V0, V2);
        } else {
            self.asm.vmov(V1, vb);
            self.asm.vbit(0x4E00_2000, out, V0, V2);
        }
        self.fput(result, out);
    }

    fn shuffle_single_fast(&mut self, (spec, eb, total): (VecSpec, usize, usize), src_val: Value, ridx: &[u8; 16], result: Value) -> bool {
        if (0..total).all(|j| usize::from(ridx[j]) == j) {
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.asm.vmov(dst, src);
            self.fput(result, dst);
            return true;
        }
        let r = usize::from(ridx[0]);
        if r < total && (0..total).all(|j| usize::from(ridx[j]) == (r + j) % total) {
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.asm.vbit(0x2E00_0000 | (spec.q << 30) | ((r as u32) << 11), dst, src, src);
            self.fput(result, dst);
            return true;
        }
        self.shuffle_dup_or_rev((spec, eb, total), src_val, ridx, result) || if let Some((base, size, _)) = match_permute(total, eb, ridx, true) {
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.asm.vbit(base | (spec.q << 30) | (size << 22), dst, src, src);
            self.fput(result, dst);
            true
        } else {
            false
        }
    }

    fn shuffle_dup_or_rev(&mut self, (spec, eb, total): (VecSpec, usize, usize), src_val: Value, ridx: &[u8; 16], result: Value) -> bool {
        for unit in [8_usize, 4, 2, 1] {
            if unit >= eb && unit <= total / 2 && usize::from(ridx[0]) % unit == 0 {
                let elem_idx = usize::from(ridx[0]) / unit;
                if (0..total).all(|j| usize::from(ridx[j]) == elem_idx * unit + (j % unit)) {
                    let dst = self.fdst(result);
                    let src = self.fin(src_val, V0);
                    self.dup_vec_elem(spec.q, unit.trailing_zeros(), elem_idx as u8, dst, src);
                    self.fput(result, dst);
                    return true;
                }
            }
        }
        const REVS: &[(usize, usize, u32)] = &[
            (8, 4, 0x0EA0_0800), (8, 2, 0x0E60_0800), (8, 1, 0x0E20_0800),
            (4, 2, 0x2E60_0800), (4, 1, 0x2E20_0800), (2, 1, 0x0E20_1800),
        ];
        for &(container, elem, base) in REVS {
            if elem >= eb && container <= total && (0..total).all(|j| usize::from(ridx[j]) == rev_byte(container, elem, j)) {
                let dst = self.fdst(result);
                let src = self.fin(src_val, V0);
                self.v_two(base | (spec.q << 30), dst, src);
                self.fput(result, dst);
                return true;
            }
        }
        if total == 16 && let Some(elem) = [4_usize, 2, 1].into_iter().find(|&e| e >= eb && (0..16).all(|j| usize::from(ridx[j]) == rev_byte(16, e, j))) {
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.v_two(0x0E20_0800 | (elem.trailing_zeros() << 22) | (spec.q << 30), dst, src);
            self.asm.vbit(0x6E00_0000 | (8 << 11), dst, dst, dst);
            self.fput(result, dst);
            return true;
        }
        false
    }

    fn load_bytes_v2(&mut self, bytes: &[u8; 16], full: bool) {
        let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = *bytes;
        let lo = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        let hi = u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]);
        if (!full || lo == hi) && let Some(word) = super::fuse::encode_neon_movi(u32::from(full), lo) {
            self.asm.word(word | u32::from(V2.0));
            return;
        }
        if full && lo == hi {
            self.asm.mov_imm(A, lo as i64);
            self.asm.word(0x4E08_0C00 | u32::from(A.0) << 5 | u32::from(V2.0));
            return;
        }
        if lo == 0 {
            self.asm.veor_zero(V2);
        } else {
            self.asm.mov_imm(A, lo as i64);
            self.asm.fmov_in(true, V2, A);
        }
        if full && hi != 0 {
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
            let s = if self.plan.const_of(val) == Some(0) { ZR } else { self.bits(val, A) };
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
            ReduceOp::And | ReduceOp::Or | ReduceOp::Xor => self.vec_reduce_bitwise(op, spec, src, result),
            ReduceOp::Add => self.vec_reduce_add(spec, src, result),
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax if spec.count == 2 => {
                self.vec_reduce_minmax_pair(op, spec, src, result);
            }
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax => self.vec_reduce_minmax(op, spec, src, result),
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
            let base = if op == ReduceOp::Fmin { 0x1E20_7800 } else { 0x1E20_6800 };
            self.asm.vbit(base | (spec.fsz() << 22), dst, src, V2);
        } else {
            let base = if op == ReduceOp::Fmin { 0x6EB0_C800 } else { 0x6E30_C800 };
            self.v_two(base, dst, src);
        }
        self.fput(result, dst);
    }

    fn vec_reduce_bitwise(&mut self, op: ReduceOp, spec: VecSpec, src: Vreg, result: Value) {
        let bit_op = match op {
            ReduceOp::And => 0x0E20_1C00,
            ReduceOp::Or => 0x0EA0_1C00,
            _ => 0x2E20_1C00,
        } | (spec.q << 30);
        let mut cur = src;
        let mut stride = spec.lane.bytes() * u32::from(spec.count) / 2;
        while stride >= spec.lane.bytes() {
            self.asm.vbit(0x2E00_0000 | (spec.q << 30) | (stride << 11), V3, cur, cur);
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

    fn vec_movemask(&mut self, spec: VecSpec, dst: Reg, source: Vreg) {
        match spec.lane {
            Lane::I8 | Lane::I16 => self.movemask_mul(spec, dst, source),
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
                    self.asm.alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 1, dst, dst, C);
                }
            }
        }
    }

    fn movemask_mul(&mut self, spec: VecSpec, dst: Reg, source: Vreg) {
        let (mask, mul, lsb_shift, top_shift, half_bits) = if spec.lane == Lane::I8 {
            (0x0101_0101_0101_0101_u64, 0x0102_0408_1020_4080_u64, 7_u8, 56_u8, 8_u8)
        } else {
            (0x0001_0001_0001_0001_u64, 0x0001_0002_0004_0008_u64, 15_u8, 48_u8, 4_u8)
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
        self.asm.alu_shifted(Alu::Orr, Size::X, Shift::Lsl, half_bits, dst, A, C);
    }

    fn movemask_i32(&mut self, count: u8, dst: Reg, source: Vreg) {
        self.asm.fmov_out(true, D, source);
        if count == 2 {
            self.asm.lsr_imm(C, D, 63);
            self.asm.bitfield(false, dst, D, (31, 31));
            self.asm.alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 1, dst, dst, C);
            return;
        }
        self.asm.umov_d1(C, source);
        self.asm.bitfield(false, dst, D, (31, 31));
        self.asm.lsr_imm(D, D, 63);
        self.asm.alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 1, dst, dst, D);
        self.asm.bitfield(false, D, C, (31, 31));
        self.asm.alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 2, dst, dst, D);
        self.asm.lsr_imm(C, C, 63);
        self.asm.alu_shifted(Alu::Orr, Size::X, Shift::Lsl, 3, dst, dst, C);
    }

    pub(super) fn vec_target_op(&mut self, id: TargetOpId, args: TinyArgs, result: Value) -> Result<(), String> {
        let Some(desc) = ARM64_TARGET_OPS.iter().find(|d| d.id == id) else {
            return Err(format!("unsupported ARM64 target op {id:?}"));
        };
        let args = args.as_slice();
        let base = desc.base;
        let dst = self.fdst(result);
        match desc.kind {
            TargetKind::OneReg => {
                let a = self.fin(args[0], V0);
                self.v_two(base, dst, a);
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

fn rev_byte(container: usize, elem: usize, j: usize) -> usize {
    let chunk_base = j & !(container - 1);
    let in_chunk = j & (container - 1);
    let reversed_elem = container - elem - (in_chunk & !(elem - 1));
    chunk_base + reversed_elem + (in_chunk & (elem - 1))
}

fn match_permute(total: usize, eb: usize, bidx: &[u8; 16], single: bool) -> Option<(u32, u32, bool)> {
    const OPS: &[(u32, fn(usize, usize) -> usize)] = &[
        (0x0E00_3800, |i, m| if i & 1 == 0 { i / 2 } else { m + i / 2 }),
        (0x0E00_7800, |i, m| if i & 1 == 0 { m / 2 + i / 2 } else { m + m / 2 + i / 2 }),
        (0x0E00_1800, |i, _| 2 * i),
        (0x0E00_5800, |i, _| 2 * i + 1),
        (0x0E00_2800, |i, m| if i & 1 == 0 { i } else { m + i - 1 }),
        (0x0E00_6800, |i, m| if i & 1 == 0 { i + 1 } else { m + i }),
    ];
    for unit in [8_usize, 4, 2, 1] {
        if unit < eb || unit > total / 2 {
            continue;
        }
        let m = total / unit;
        let mut elems = [0_usize; 16];
        let mut aligned = true;
        for i in 0..m {
            let base = usize::from(bidx[i * unit]);
            if base % unit != 0 || base >= 2 * total || !(1..unit).all(|k| usize::from(bidx[i * unit + k]) == base + k) {
                aligned = false;
                break;
            }
            elems[i] = base / unit;
        }
        if !aligned {
            continue;
        }
        let size = unit.trailing_zeros();
        for &(base, pat) in OPS {
            if single {
                if (0..m).all(|i| elems[i] == pat(i, m) % m) {
                    return Some((base, size, false));
                }
            } else if (0..m).all(|i| elems[i] == pat(i, m)) {
                return Some((base, size, false));
            } else if (0..m).all(|i| {
                let p = pat(i, m);
                elems[i] == if p < m { p + m } else { p - m }
            }) {
                return Some((base, size, true));
            }
        }
    }
    None
}
