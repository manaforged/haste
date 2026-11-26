mod arith;
mod cmp;
mod convert;

use super::asm::{Reg, Vreg};
use super::vector_ops::NEON_BSL;
use super::{Emit, V1};
use crate::ir::{Binary, Convert, Lane, Op, ShuffleMask, Type, Value};

pub(super) const V2: Vreg = Vreg(2);
pub(super) const V3: Vreg = Vreg(3);

#[derive(Clone, Copy)]
pub(super) struct VecSpec {
    pub(super) lane: Lane,
    pub(super) count: u8,
    pub(super) q: u32,
    pub(super) size: u32,
}

impl VecSpec {
    pub(super) fn of(ty: Type) -> Result<Self, String> {
        let Type::Vec { lane, count } = ty else {
            return Err(format!("expected vector type, got {ty:?}"));
        };
        let q = u32::from(lane.bytes() * u32::from(count) == 16);
        let size = match lane {
            Lane::I8 => 0,
            Lane::I16 => 1,
            Lane::I32 | Lane::F32 => 2,
            Lane::I64 | Lane::F64 => 3,
        };
        Ok(Self {
            lane,
            count,
            q,
            size,
        })
    }

    pub(super) fn fsz(self) -> u32 {
        u32::from(self.lane == Lane::F64)
    }

    pub(super) fn is_1d(self) -> bool {
        self.count == 1
    }
}

impl Emit<'_> {
    pub(super) fn v_two(&mut self, base: u32, dst: Vreg, src: Vreg) {
        self.asm
            .word(base | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    fn v_not(&mut self, q: u32, dst: Vreg, src: Vreg) {
        self.v_two(0x2E20_5800 | (q << 30), dst, src);
    }

    fn v_neg(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        let base = if spec.is_1d() {
            0x7EE0_B800
        } else {
            0x2E20_B800 | (spec.q << 30) | (spec.size << 22)
        };
        self.v_two(base, dst, src);
    }

    pub(super) fn ins_gpr(&mut self, dst: Vreg, size: u32, idx: u8, src: Reg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        self.asm
            .word(0x4E00_1C00 | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn ins_elem(&mut self, dst: Vreg, size: u32, dst_idx: u8, src: Vreg, src_idx: u8) {
        let imm5 = (u32::from(dst_idx) << (size + 1)) | (1 << size);
        let imm4 = u32::from(src_idx) << size;
        self.asm.word(
            0x6E00_0400 | (imm5 << 16) | (imm4 << 11) | u32::from(src.0) << 5 | u32::from(dst.0),
        );
    }

    pub(super) fn umov_gpr(&mut self, size: u32, idx: u8, dst: Reg, src: Vreg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        let base = if size == 3 { 0x4E00_3C00 } else { 0x0E00_3C00 };
        self.asm.emit_gp(
            dst,
            base | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0),
        );
    }

    pub(super) fn dup_scalar_elem(&mut self, size: u32, idx: u8, dst: Vreg, src: Vreg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        self.asm
            .word(0x5E00_0400 | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn dup_vec_elem(&mut self, q: u32, size: u32, idx: u8, dst: Vreg, src: Vreg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        self.asm.word(
            0x0E00_0400 | (q << 30) | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0),
        );
    }

    fn bsl(&mut self, q: u32, mask: Vreg, mut x: Vreg, mut y: Vreg, dst: Vreg) {
        let out = if dst == mask {
            mask
        } else {
            if x == dst {
                self.asm.vmov(V1, x);
                self.clobber_vreg(V1);
                x = V1;
            }
            if y == dst {
                self.asm.vmov(V2, y);
                self.clobber_vreg(V2);
                y = V2;
            }
            self.asm.vmov(dst, mask);
            dst
        };
        self.asm.vbit(NEON_BSL | (q << 30), out, x, y);
    }

    pub(super) fn const_bits(&self, val: Value) -> Option<i64> {
        super::fuse::const_of(self.func, &self.plan, val, |v| self.defs[v.index()])
    }

    fn eval_shuffle_bytes(
        &self,
        (a, b, m): (Value, Value, [u8; 16]),
        spec: VecSpec,
        depth: u8,
    ) -> [Option<u8>; 16] {
        let (eb, count) = (spec.lane.bytes() as usize, usize::from(spec.count));
        let (ba, bb, mut out) = (
            self.eval_vec_bytes(a, depth),
            self.eval_vec_bytes(b, depth),
            [None; 16],
        );
        for (i, &sel) in m[..count].iter().enumerate() {
            let src = if sel < spec.count {
                Some(&ba[usize::from(sel) * eb..])
            } else if sel < 2 * spec.count {
                Some(&bb[usize::from(sel - spec.count) * eb..])
            } else {
                None
            };
            if let Some(s) = src {
                out[i * eb..(i + 1) * eb].copy_from_slice(&s[..eb]);
            }
        }
        out
    }

    fn eval_vec_bytes(&self, val: Value, depth: u8) -> [Option<u8>; 16] {
        let ty = self.ty(val);
        if let Some(imm) = self.const_bits(val) {
            if ty.is_vector() {
                return if imm == 0 { [Some(0); 16] } else { [None; 16] };
            }
            let mut out = [None; 16];
            if ty.is_small_int() || ty.is_float() {
                for (dst, &byte) in out[..ty.bytes() as usize]
                    .iter_mut()
                    .zip(&(imm as u64).to_le_bytes())
                {
                    *dst = Some(byte);
                }
            }
            return out;
        }
        let Some(inst) = (depth > 0).then(|| self.defs[val.index()]).flatten() else {
            return [None; 16];
        };
        match self.func.insts[inst.index()].op {
            Op::Convert(Convert::Bitcast, inner) if self.ty(inner).bytes() == ty.bytes() => {
                self.eval_vec_bytes(inner, depth - 1)
            }
            Op::Insert(vec, scalar, idx) => {
                let Ok(spec) = VecSpec::of(ty) else {
                    return [None; 16];
                };
                if idx >= spec.count {
                    return [None; 16];
                }
                let (mut out, sb, eb) = (
                    self.eval_vec_bytes(vec, depth - 1),
                    self.eval_vec_bytes(scalar, depth - 1),
                    spec.lane.bytes() as usize,
                );
                out[usize::from(idx) * eb..usize::from(idx) * eb + eb].copy_from_slice(&sb[..eb]);
                out
            }
            Op::Shuffle(a, b, ShuffleMask::Const(m)) => VecSpec::of(ty)
                .map_or([None; 16], |spec| {
                    self.eval_shuffle_bytes((a, b, m), spec, depth - 1)
                }),
            _ => [None; 16],
        }
    }

    pub(super) fn eval_vec_full(&self, val: Value) -> Option<(bool, u64, u64)> {
        let (bytes, total, mut raw) = (
            self.eval_vec_bytes(val, 5),
            self.ty(val).bytes() as usize,
            [0_u8; 16],
        );
        for (dst, &src) in raw[..total].iter_mut().zip(&bytes[..total]) {
            *dst = src?;
        }
        Some((
            total == 16,
            u64::from_le_bytes(raw[..8].try_into().ok()?),
            u64::from_le_bytes(raw[8..].try_into().ok()?),
        ))
    }

    pub(super) fn splat_const_lane(&self, val: Value) -> Option<i64> {
        let Ok(spec) = VecSpec::of(self.ty(val)) else {
            return None;
        };
        let (full, lo, hi) = self.eval_vec_full(val)?;
        let (eb, total, mut bytes, mut raw) = (
            spec.lane.bytes() as usize,
            if full { 16 } else { 8 },
            [0_u8; 16],
            [0_u8; 8],
        );
        bytes[..8].copy_from_slice(&lo.to_le_bytes());
        bytes[8..].copy_from_slice(&hi.to_le_bytes());
        let first = &bytes[..eb];
        if (1..total / eb).any(|i| &bytes[i * eb..(i + 1) * eb] != first) {
            return None;
        }
        raw[..eb].copy_from_slice(first);
        Some(u64::from_le_bytes(raw) as i64)
    }

    pub(super) fn is_vec_zero(&self, value: Value) -> bool {
        self.plan.const_of(value) == Some(0)
            || self
                .eval_vec_full(value)
                .is_some_and(|(_, lo, hi)| lo == 0 && hi == 0)
    }

    fn unwrap_load_addr(&self, addr: Value, off: i32) -> (Value, i32) {
        if self.plan.sib[addr.index()].is_none()
            && self.plan.elided[addr.index()]
            && let Some(inst) = self.defs[addr.index()]
            && let Op::Binary(Binary::Iadd, base, c) = self.func.insts[inst.index()].op
            && let Some(imm) =
                super::fuse::const_of(self.func, &self.plan, c, |v| self.defs[v.index()])
            && let Some(sum) = i64::from(off)
                .checked_add(imm)
                .and_then(|v| i32::try_from(v).ok())
        {
            return (base, sum);
        }
        (addr, off)
    }

    fn emit_vload_from(&mut self, lo: Value, dst: Vreg) -> bool {
        let Some(inst) = self.defs[lo.index()] else {
            return false;
        };
        match self.func.insts[inst.index()].op {
            Op::Load(addr, off) => {
                let (addr, off) = self.unwrap_load_addr(addr, off);
                if let Some((base, index, shift)) = self.sib_indexed(addr, off, 16) {
                    self.asm.vload_indexed(16, dst, base, index, shift);
                } else {
                    let mem = self.address(addr, off);
                    self.asm.vload(16, dst, mem);
                }
                true
            }
            Op::StackLoad(slot, off) => {
                let mem = self.slot(slot, off);
                self.asm.vload(16, dst, mem);
                true
            }
            _ => false,
        }
    }

    fn try_elided_insert_pair(
        &mut self,
        vec: Value,
        val: Value,
        idx: u8,
        result: Value,
    ) -> Result<bool, String> {
        if !self.plan.elided[vec.index()] {
            return Ok(false);
        }
        let Some(inst0) = self.defs[vec.index()] else {
            return Ok(false);
        };
        let Op::Insert(_, val_a, lane_a) = self.func.insts[inst0.index()].op else {
            return Ok(false);
        };
        let spec = VecSpec::of(self.ty(result))?;
        if spec.count != 2 || !matches!(spec.lane, Lane::I64 | Lane::F64) || lane_a + idx != 1 {
            return Ok(false);
        }
        let (lo, hi) = if lane_a == 0 {
            (val_a, val)
        } else {
            (val, val_a)
        };
        let dst = self.fdst(result);
        if self.plan.elided[lo.index()]
            && self.plan.elided[hi.index()]
            && self.emit_vload_from(lo, dst)
        {
            self.fput(result, dst);
            return Ok(true);
        }
        if spec.lane == Lane::F64 {
            if self.try_emit_const_vec_imm(result) {
                return Ok(true);
            }
            let mut s_hi = self.fin(hi, V1);
            if s_hi == dst {
                self.asm.vmov(V2, s_hi);
                self.clobber_vreg(V2);
                s_hi = V2;
            }
            let s_lo = self.fin(lo, dst);
            self.asm.vmov(dst, s_lo);
            self.ins_elem(dst, 3, 1, s_hi, 0);
        } else {
            self.emit_insert_i64_pair(dst, lo, hi);
        }
        self.fput(result, dst);
        Ok(true)
    }

    fn vec_insert_opt(
        &mut self,
        vec: Value,
        val: Value,
        idx: u8,
        result: Value,
    ) -> Result<(), String> {
        if self.try_elided_insert_pair(vec, val, idx, result)? {
            return Ok(());
        }
        let spec = VecSpec::of(self.ty(vec))?;
        if self.try_vec_insert_const(vec, val, idx, spec, result) {
            return Ok(());
        }
        self.vec_insert(vec, val, idx, result)
    }

    pub(super) fn vector_inst(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<bool, String> {
        match *op {
            Op::Const(0) if self.ty(result()?).is_vector() => {
                let res = result()?;
                if !self.plan.is_static(res) {
                    let dst = self.fdst(res);
                    self.asm.veor_zero(dst);
                    self.fput(res, dst);
                }
            }
            Op::Shuffle(a, b, mask) => {
                let res = result()?;
                if !self.try_emit_const_vec_imm(res) {
                    self.vec_shuffle(a, b, mask, res)?;
                }
            }
            Op::Extract(vec, idx) => self.vec_extract(vec, idx, result()?)?,
            Op::Insert(vec, val, idx) => self.vec_insert_opt(vec, val, idx, result()?)?,
            Op::Reduce(rop, vec) => self.vec_reduce(rop, vec, result()?)?,
            Op::TargetOp(id, args) => self.vec_target_op(id, args, result()?)?,
            _ => return Ok(false),
        }
        Ok(true)
    }
}
