use super::asm::{Alu, Cond, Reg, Size, Vreg, ZR};
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
            ShuffleMask::Const(m) => self.vec_shuffle_const(spec, a, b, m, result),
        }
    }

    fn vec_shuffle_const(&mut self, spec: VecSpec, a: Value, b: Value, m: [u8; 16], result: Value) -> Result<(), String> {
        let n = usize::from(spec.count);
        let eb = spec.lane.bytes() as usize;
        let total = n * eb;
        let mut bidx = [0x80_u8; 16];
        let mut uses_a = false;
        let mut uses_b = false;
        let mut has_zero = false;
        for (i, &lane) in m[..n].iter().enumerate() {
            let x = usize::from(lane);
            if x < n {
                uses_a = true;
                for k in 0..eb {
                    bidx[i * eb + k] = (x * eb + k) as u8;
                }
            } else if x < 2 * n {
                uses_b = true;
                let base = total + (x - n) * eb;
                for k in 0..eb {
                    bidx[i * eb + k] = (base + k) as u8;
                }
            } else {
                has_zero = true;
            }
        }
        if !uses_a && !uses_b {
            let dst = self.fdst(result);
            self.asm.veor_zero(dst);
            self.fput(result, dst);
            return Ok(());
        }
        if uses_a ^ uses_b {
            let src_val = if uses_a { a } else { b };
            let src_off = if uses_a { 0_u8 } else { total as u8 };
            let mut ridx = [0x80_u8; 16];
            for j in 0..total {
                if bidx[j] != 0x80 {
                    ridx[j] = bidx[j] - src_off;
                }
            }
            if !has_zero && self.shuffle_single_fast(spec, eb, total, src_val, &ridx, result) {
                return Ok(());
            }
            self.load_bytes_v2(&ridx, total == 16);
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.asm.vbit(0x0E00_0000 | (spec.q << 30), dst, src, V2);
            self.fput(result, dst);
            return Ok(());
        }
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
                return Ok(());
            }
            if let Some((base, size, swap)) = match_permute(total, eb, &bidx, false) {
                let dst = self.fdst(result);
                let (vn, vm) = if swap { (vb, va) } else { (va, vb) };
                self.asm.vbit(base | (spec.q << 30) | (size << 22), dst, vn, vm);
                self.fput(result, dst);
                return Ok(());
            }
        }
        self.load_bytes_v2(&bidx, total == 16);
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

    fn shuffle_single_fast(&mut self, spec: VecSpec, eb: usize, total: usize, src_val: Value, ridx: &[u8; 16], result: Value) -> bool {
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
            (8, 4, 0x0EA0_0800),
            (8, 2, 0x0E60_0800),
            (8, 1, 0x0E20_0800),
            (4, 2, 0x2E60_0800),
            (4, 1, 0x2E20_0800),
            (2, 1, 0x0E20_1800),
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
        if total == 16 {
            for elem in [4_usize, 2, 1] {
                if elem >= eb && (0..16).all(|j| usize::from(ridx[j]) == rev_byte(16, elem, j)) {
                    let dst = self.fdst(result);
                    let src = self.fin(src_val, V0);
                    self.v_two(0x0E20_0800 | (elem.trailing_zeros() << 22) | (spec.q << 30), dst, src);
                    self.asm.vbit(0x6E00_0000 | (8 << 11), dst, dst, dst);
                    self.fput(result, dst);
                    return true;
                }
            }
        }
        if let Some((base, size, _)) = match_permute(total, eb, ridx, true) {
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.asm.vbit(base | (spec.q << 30) | (size << 22), dst, src, src);
            self.fput(result, dst);
            return true;
        }
        false
    }

    fn load_bytes_v2(&mut self, bytes: &[u8; 16], full: bool) {
        let [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15] = *bytes;
        let lo = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        self.asm.mov_imm(A, lo as i64);
        self.asm.fmov_in(true, V2, A);
        if full {
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
                self.asm.madd(false, Size::X, A, (A, IP), ZR);
                self.asm.lsr_imm(dst, A, 56);
                return;
            }
            self.asm.umov_d1(C, source);
            for reg in [A, C] {
                self.asm.lsr_imm(reg, reg, 7);
                self.asm.alu(Alu::And, Size::X, reg, reg, D);
                self.asm.madd(false, Size::X, reg, (reg, IP), ZR);
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
