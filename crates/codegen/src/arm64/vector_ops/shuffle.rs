use super::super::asm::{Asm, ZR};
use super::super::vector::{V2, V3, VecSpec};
use super::super::{A, C, Emit, V0, V1};
use super::{NEON_EXT, NEON_REV16, NEON_REV32, NEON_REV64, can_emit_ext_pair_imm};
use crate::ir::{Op, ShuffleMask, Value};

impl Emit<'_> {
    pub(in super::super) fn vec_shuffle(
        &mut self,
        a: Value,
        b: Value,
        mask: ShuffleMask,
        result: Value,
    ) -> Result<(), String> {
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

    fn vec_shuffle_const(
        &mut self,
        spec: VecSpec,
        (a, b): (Value, Value),
        m: [u8; 16],
        result: Value,
    ) -> Result<(), String> {
        let n = usize::from(spec.count);
        let first = usize::from(m[0]);
        if spec.lane.is_int()
            && spec.count > 1
            && first < 2 * n
            && m[..n].iter().all(|&x| usize::from(x) == first)
        {
            let src = if first < n { a } else { b };
            if self.plan.elided[src.index()]
                && let Some(inst) = self.defs[src.index()]
                && let Op::Insert(_, scalar, idx) = self.func.insts[inst.index()].op
                && usize::from(idx) == first % n
            {
                let dst = self.fdst(result);
                let reg = if self.plan.const_of(scalar) == Some(0) {
                    ZR
                } else {
                    self.bits(scalar, A)
                };
                self.asm.word(
                    0x0E00_0C00
                        | (spec.q << 30)
                        | ((1_u32 << spec.size) << 16)
                        | u32::from(reg.0) << 5
                        | u32::from(dst.0),
                );
                self.fput(result, dst);
                return Ok(());
            }
        }
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
            self.shuffle_single_source(
                (spec, eb, total),
                if uses_a { (a, 0) } else { (b, total as u8) },
                has_zero,
                (&bidx, result),
            );
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
        let dst = self.fdst(result);
        let src = self.fin(src_val, V0);
        self.load_bytes_v2(&ridx, total == 16);
        self.asm.vbit(0x0E00_0000 | (spec.q << 30), dst, src, V2);
        self.fput(result, dst);
    }

    fn shuffle_two_sources(
        &mut self,
        (spec, eb, total): (VecSpec, usize, usize),
        (a, b): (Value, Value),
        has_zero: bool,
        (bidx, result): (&[u8; 16], Value),
    ) {
        if !has_zero && self.shuffle_two_fast((spec, eb, total), (a, b), bidx, result) {
            return;
        }
        if let Some((full, lo, hi)) = self.eval_vec_full(result)
            && full
            && ((lo == hi && Asm::mov_imm_is_single(lo as i64)) || can_emit_ext_pair_imm(lo, hi))
        {
            let dst = self.fdst(result);
            if !self.emit_neon_imm_only(dst, true, lo, hi)
                && !self.emit_ext_pair_imm(dst, V2, lo, hi)
            {
                self.emit_dup_i64(dst, lo);
            }
            self.fput(result, dst);
            return;
        }
        let va = self.fin(a, V0);
        let vb = self.fin(b, V1);
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

    fn shuffle_two_fast(
        &mut self,
        (spec, eb, total): (VecSpec, usize, usize),
        (a, b): (Value, Value),
        bidx: &[u8; 16],
        result: Value,
    ) -> bool {
        let s = usize::from(bidx[0]);
        let ext_pair = if s < total && (0..total).all(|j| usize::from(bidx[j]) == s + j) {
            Some((false, s))
        } else if (total..2 * total).contains(&s)
            && (0..total).all(|j| usize::from(bidx[j]) == (s + j) % (2 * total))
        {
            Some((true, s - total))
        } else {
            None
        };
        if let Some((swap, byte_off)) = ext_pair {
            let va = self.fin(a, V0);
            let vb = self.fin(b, V1);
            let (vn, vm) = if swap { (vb, va) } else { (va, vb) };
            let dst = self.fdst(result);
            self.asm.vbit(
                NEON_EXT | (spec.q << 30) | ((byte_off as u32) << 11),
                dst,
                vn,
                vm,
            );
            self.fput(result, dst);
            return true;
        }
        if let Some((base, size, swap)) = match_permute(total, eb, bidx, false) {
            let va = self.fin(a, V0);
            let vb = self.fin(b, V1);
            let (vn, vm) = if swap { (vb, va) } else { (va, vb) };
            let dst = self.fdst(result);
            self.asm
                .vbit(base | (spec.q << 30) | (size << 22), dst, vn, vm);
            self.fput(result, dst);
            return true;
        }
        let (src, off) = if self.is_vec_zero(b) {
            (a, 0)
        } else if self.is_vec_zero(a) {
            (b, total as u8)
        } else {
            return false;
        };
        let mut z_bidx = [0x80_u8; 16];
        let lo = usize::from(off);
        for j in 0..total {
            let idx = usize::from(bidx[j]);
            if idx >= lo && idx < lo + total {
                z_bidx[j] = bidx[j];
            }
        }
        self.shuffle_single_source((spec, eb, total), (src, off), true, (&z_bidx, result));
        true
    }

    fn shuffle_single_fast(
        &mut self,
        (spec, eb, total): (VecSpec, usize, usize),
        src_val: Value,
        ridx: &[u8; 16],
        result: Value,
    ) -> bool {
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
            self.asm.vbit(
                NEON_EXT | (spec.q << 30) | ((r as u32) << 11),
                dst,
                src,
                src,
            );
            self.fput(result, dst);
            return true;
        }
        self.shuffle_dup_or_rev((spec, eb, total), src_val, ridx, result)
            || if let Some((base, size, _)) = match_permute(total, eb, ridx, true) {
                let dst = self.fdst(result);
                let src = self.fin(src_val, V0);
                self.asm
                    .vbit(base | (spec.q << 30) | (size << 22), dst, src, src);
                self.fput(result, dst);
                true
            } else {
                false
            }
    }

    fn shuffle_dup_or_rev(
        &mut self,
        (spec, eb, total): (VecSpec, usize, usize),
        src_val: Value,
        ridx: &[u8; 16],
        result: Value,
    ) -> bool {
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
            (8, 1, NEON_REV64),
            (4, 2, 0x2E60_0800),
            (4, 1, NEON_REV32),
            (2, 1, NEON_REV16),
        ];
        for &(container, elem, base) in REVS {
            if elem >= eb
                && container <= total
                && (0..total).all(|j| usize::from(ridx[j]) == rev_byte(container, elem, j))
            {
                let dst = self.fdst(result);
                let src = self.fin(src_val, V0);
                self.v_two(base | (spec.q << 30), dst, src);
                self.fput(result, dst);
                return true;
            }
        }
        if total == 16
            && let Some(elem) = [4_usize, 2, 1]
                .into_iter()
                .find(|&e| e >= eb && (0..16).all(|j| usize::from(ridx[j]) == rev_byte(16, e, j)))
        {
            let dst = self.fdst(result);
            let src = self.fin(src_val, V0);
            self.v_two(
                NEON_REV64 | (elem.trailing_zeros() << 22) | (spec.q << 30),
                dst,
                src,
            );
            self.asm
                .vbit(NEON_EXT | (1 << 30) | (8 << 11), dst, dst, dst);
            self.fput(result, dst);
            return true;
        }
        false
    }

    fn load_bytes_v2(&mut self, bytes: &[u8; 16], full: bool) {
        self.clobber_vreg(V2);
        let [
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
        ] = *bytes;
        let lo = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
        let hi = u64::from_le_bytes([b8, b9, b10, b11, b12, b13, b14, b15]);
        if self.emit_neon_imm_only(V2, full, lo, hi) {
            return;
        }
        if full && lo == hi {
            self.emit_dup_i64(V2, lo);
            return;
        }
        if full && self.emit_ext_pair_imm(V2, V3, lo, hi) {
            return;
        }
        if !self.emit_neon_imm_only(V2, false, lo, 0) {
            self.asm.mov_imm(A, lo as i64);
            self.clobber_reg(A);
            self.asm.fmov_in(true, V2, A);
        }
        if full && hi != 0 {
            self.asm.mov_imm(C, hi as i64);
            self.clobber_reg(C);
            self.ins_gpr(V2, 3, 1, C);
        }
    }
}

fn rev_byte(container: usize, elem: usize, j: usize) -> usize {
    let chunk_base = j & !(container - 1);
    let in_chunk = j & (container - 1);
    let reversed_elem = container - elem - (in_chunk & !(elem - 1));
    chunk_base + reversed_elem + (in_chunk & (elem - 1))
}

fn match_permute(
    total: usize,
    eb: usize,
    bidx: &[u8; 16],
    single: bool,
) -> Option<(u32, u32, bool)> {
    const OPS: &[(u32, fn(usize, usize) -> usize)] = &[
        (
            0x0E00_3800,
            |i, m| if i & 1 == 0 { i / 2 } else { m + i / 2 },
        ),
        (0x0E00_7800, |i, m| {
            if i & 1 == 0 {
                m / 2 + i / 2
            } else {
                m + m / 2 + i / 2
            }
        }),
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
            if base % unit != 0
                || base >= 2 * total
                || !(1..unit).all(|k| usize::from(bidx[i * unit + k]) == base + k)
            {
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
