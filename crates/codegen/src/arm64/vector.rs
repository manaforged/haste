use super::asm::{Cond, Fop, Funary, Reg, Size, Vreg, ZR};
use super::vector_ops::{
    NEON_AND, NEON_BSL, NEON_DUP_GPR_2D, NEON_EOR, NEON_ORR, NEON_REV16, NEON_REV32, NEON_REV64, NEON_SMAX, NEON_SMIN, NEON_SUB, NEON_SUB_1D, NEON_UMAX,
    NEON_UMIN, NEON_USHL, NEON_USHL_1D,
};
use super::{A, C, D, Emit, S, V0, V1};
use crate::cse::flip_int_cc;
use crate::ir::{Binary, Convert, FloatCC, IntCC, Lane, Op, ShuffleMask, Type, Unary, Value};

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
        Ok(Self { lane, count, q, size })
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
        self.asm.word(base | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    fn v_not(&mut self, q: u32, dst: Vreg, src: Vreg) {
        self.v_two(0x2E20_5800 | (q << 30), dst, src);
    }

    fn v_neg(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        let base = if spec.is_1d() { 0x7EE0_B800 } else { 0x2E20_B800 | (spec.q << 30) | (spec.size << 22) };
        self.v_two(base, dst, src);
    }

    pub(super) fn ins_gpr(&mut self, dst: Vreg, size: u32, idx: u8, src: Reg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        self.asm.word(0x4E00_1C00 | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn ins_elem(&mut self, dst: Vreg, size: u32, dst_idx: u8, src: Vreg, src_idx: u8) {
        let imm5 = (u32::from(dst_idx) << (size + 1)) | (1 << size);
        let imm4 = u32::from(src_idx) << size;
        self.asm.word(0x6E00_0400 | (imm5 << 16) | (imm4 << 11) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn umov_gpr(&mut self, size: u32, idx: u8, dst: Reg, src: Vreg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        let base = if size == 3 { 0x4E00_3C00 } else { 0x0E00_3C00 };
        self.asm.emit_gp(dst, base | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn dup_scalar_elem(&mut self, size: u32, idx: u8, dst: Vreg, src: Vreg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        self.asm.word(0x5E00_0400 | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub(super) fn dup_vec_elem(&mut self, q: u32, size: u32, idx: u8, dst: Vreg, src: Vreg) {
        let imm5 = (u32::from(idx) << (size + 1)) | (1 << size);
        self.asm.word(0x0E00_0400 | (q << 30) | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
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

    fn eval_shuffle_bytes(&self, (a, b, m): (Value, Value, [u8; 16]), spec: VecSpec, depth: u8) -> [Option<u8>; 16] {
        let (eb, count) = (spec.lane.bytes() as usize, usize::from(spec.count));
        let (ba, bb, mut out) = (self.eval_vec_bytes(a, depth), self.eval_vec_bytes(b, depth), [None; 16]);
        for (i, &sel) in m[..count].iter().enumerate() {
            let src = if sel < spec.count { Some(&ba[usize::from(sel) * eb..]) } else if sel < 2 * spec.count { Some(&bb[usize::from(sel - spec.count) * eb..]) } else { None };
            if let Some(s) = src { out[i * eb..(i + 1) * eb].copy_from_slice(&s[..eb]); }
        }
        out
    }

    fn eval_vec_bytes(&self, val: Value, depth: u8) -> [Option<u8>; 16] {
        let ty = self.ty(val);
        if let Some(imm) = self.const_bits(val) {
            if ty.is_vector() { return if imm == 0 { [Some(0); 16] } else { [None; 16] }; }
            let mut out = [None; 16];
            if ty.is_small_int() || ty.is_float() {
                for (dst, &byte) in out[..ty.bytes() as usize].iter_mut().zip(&(imm as u64).to_le_bytes()) { *dst = Some(byte); }
            }
            return out;
        }
        let Some(inst) = (depth > 0).then(|| self.defs[val.index()]).flatten() else { return [None; 16] };
        match self.func.insts[inst.index()].op {
            Op::Convert(Convert::Bitcast, inner) if self.ty(inner).bytes() == ty.bytes() => self.eval_vec_bytes(inner, depth - 1),
            Op::Insert(vec, scalar, idx) => {
                let Ok(spec) = VecSpec::of(ty) else { return [None; 16] };
                if idx >= spec.count { return [None; 16]; }
                let (mut out, sb, eb) = (self.eval_vec_bytes(vec, depth - 1), self.eval_vec_bytes(scalar, depth - 1), spec.lane.bytes() as usize);
                out[usize::from(idx) * eb..usize::from(idx) * eb + eb].copy_from_slice(&sb[..eb]);
                out
            }
            Op::Shuffle(a, b, ShuffleMask::Const(m)) => VecSpec::of(ty).map_or([None; 16], |spec| self.eval_shuffle_bytes((a, b, m), spec, depth - 1)),
            _ => [None; 16],
        }
    }

    pub(super) fn eval_vec_full(&self, val: Value) -> Option<(bool, u64, u64)> {
        let (bytes, total, mut raw) = (self.eval_vec_bytes(val, 5), self.ty(val).bytes() as usize, [0_u8; 16]);
        for (dst, &src) in raw[..total].iter_mut().zip(&bytes[..total]) { *dst = src?; }
        Some((total == 16, u64::from_le_bytes(raw[..8].try_into().ok()?), u64::from_le_bytes(raw[8..].try_into().ok()?)))
    }

    pub(super) fn splat_const_lane(&self, val: Value) -> Option<i64> {
        let Ok(spec) = VecSpec::of(self.ty(val)) else { return None };
        let (full, lo, hi) = self.eval_vec_full(val)?;
        let (eb, total, mut bytes, mut raw) = (spec.lane.bytes() as usize, if full { 16 } else { 8 }, [0_u8; 16], [0_u8; 8]);
        bytes[..8].copy_from_slice(&lo.to_le_bytes());
        bytes[8..].copy_from_slice(&hi.to_le_bytes());
        let first = &bytes[..eb];
        if (1..total / eb).any(|i| &bytes[i * eb..(i + 1) * eb] != first) { return None; }
        raw[..eb].copy_from_slice(first);
        Some(u64::from_le_bytes(raw) as i64)
    }

    fn unwrap_load_addr(&self, addr: Value, off: i32) -> (Value, i32) {
        if self.plan.sib[addr.index()].is_none()
            && self.plan.elided[addr.index()]
            && let Some(inst) = self.defs[addr.index()]
            && let Op::Binary(Binary::Iadd, base, c) = self.func.insts[inst.index()].op
            && let Some(imm) = super::fuse::const_of(self.func, &self.plan, c, |v| self.defs[v.index()])
            && let Some(sum) = i64::from(off).checked_add(imm).and_then(|v| i32::try_from(v).ok())
        {
            return (base, sum);
        }
        (addr, off)
    }

    fn emit_vload_from(&mut self, lo: Value, dst: Vreg) -> bool {
        let Some(inst) = self.defs[lo.index()] else { return false };
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

    fn try_elided_insert_pair(&mut self, vec: Value, val: Value, idx: u8, result: Value) -> Result<bool, String> {
        if !self.plan.elided[vec.index()] {
            return Ok(false);
        }
        let Some(inst0) = self.defs[vec.index()] else { return Ok(false) };
        let Op::Insert(_, val_a, lane_a) = self.func.insts[inst0.index()].op else { return Ok(false) };
        let spec = VecSpec::of(self.ty(result))?;
        if spec.count != 2 || !matches!(spec.lane, Lane::I64 | Lane::F64) || lane_a + idx != 1 {
            return Ok(false);
        }
        let (lo, hi) = if lane_a == 0 { (val_a, val) } else { (val, val_a) };
        let dst = self.fdst(result);
        if self.plan.elided[lo.index()] && self.plan.elided[hi.index()] && self.emit_vload_from(lo, dst) {
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

    fn vec_insert_opt(&mut self, vec: Value, val: Value, idx: u8, result: Value) -> Result<(), String> {
        if self.try_elided_insert_pair(vec, val, idx, result)? {
            return Ok(());
        }
        let spec = VecSpec::of(self.ty(vec))?;
        if self.try_vec_insert_const(vec, val, idx, spec, result) {
            return Ok(());
        }
        self.vec_insert(vec, val, idx, result)
    }

    pub(super) fn vector_inst(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<bool, String> {
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

    pub(super) fn vec_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let src = self.fin(x, V0);
        match op {
            Unary::Bnot => self.v_not(spec.q, dst, src),
            Unary::Ineg => self.v_neg(spec, dst, src),
            Unary::Bswap => self.vec_bswap(spec, dst, src),
            Unary::Bitrev => {
                self.v_two(0x2E60_5800 | (spec.q << 30), dst, src);
                if spec.lane != Lane::I8 {
                    self.vec_bswap(spec, dst, dst);
                }
            }
            Unary::Clz => self.vec_clz(spec, dst, src),
            Unary::Ctz => {
                if spec.lane == Lane::I64 {
                    self.vec_ctz_i64(spec, dst, src);
                } else {
                    self.v_two(0x2E60_5800 | (spec.q << 30), dst, src);
                    if spec.lane != Lane::I8 {
                        self.vec_bswap(spec, dst, dst);
                    }
                    self.v_two(0x2E20_4800 | (spec.q << 30) | (spec.size << 22), dst, dst);
                }
            }
            Unary::Popcnt => {
                self.v_two(0x0E20_5800 | (spec.q << 30), dst, src);
                for step in 0..spec.size {
                    self.v_two(0x2E20_2800 | (spec.q << 30) | (step << 22), dst, dst);
                }
            }
            Unary::Fneg | Unary::Fabs | Unary::Sqrt | Unary::Floor | Unary::Ceil | Unary::Trunc => {
                if spec.is_1d() {
                    let funary = match op {
                        Unary::Fneg => Funary::Neg,
                        Unary::Fabs => Funary::Abs,
                        Unary::Sqrt => Funary::Sqrt,
                        Unary::Floor => Funary::Floor,
                        Unary::Ceil => Funary::Ceil,
                        Unary::Trunc => Funary::Trunc,
                        _ => unreachable!(),
                    };
                    self.asm.funary(funary, true, dst, src);
                } else {
                    let base = match op {
                        Unary::Fabs => 0x0EA0_F800,
                        Unary::Fneg => 0x2EA0_F800,
                        Unary::Sqrt => 0x2EA1_F800,
                        Unary::Floor => 0x0EA1_8800,
                        Unary::Ceil => 0x0E21_9800,
                        Unary::Trunc => 0x0EA1_9800,
                        _ => unreachable!(),
                    };
                    self.v_two(base | (spec.q << 30) | (spec.fsz() << 22), dst, src);
                }
            }
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vec_bswap(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        let base = match spec.lane {
            Lane::I8 => {
                self.asm.vmov(dst, src);
                return;
            }
            Lane::I16 => NEON_REV16,
            Lane::I32 | Lane::F32 => NEON_REV32,
            Lane::I64 | Lane::F64 => NEON_REV64,
        };
        self.v_two(base | (spec.q << 30), dst, src);
    }

    fn vec_clz(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        if spec.lane != Lane::I64 {
            self.v_two(0x2E20_4800 | (spec.q << 30) | (spec.size << 22), dst, src);
        } else {
            self.vec_clz_or_ctz_i64(spec, false, dst, src);
        }
    }

    fn vec_ctz_i64(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        self.vec_clz_or_ctz_i64(spec, true, dst, src);
    }

    pub(super) fn vec_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        if matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr) && let Some(imm) = self.splat_const_lane(y) {
            self.vec_shift_imm(spec, op, imm, dst, a);
            self.fput(result, dst);
            return Ok(());
        }
        let b = self.fin(y, V1);
        match op {
            Binary::Band | Binary::Bor | Binary::Bxor => {
                let base = match op {
                    Binary::Band => NEON_AND,
                    Binary::Bor => NEON_ORR,
                    _ => NEON_EOR,
                };
                self.asm.vbit(base | (spec.q << 30), dst, a, b);
            }
            Binary::Iadd | Binary::Isub => {
                let base = match (spec.is_1d(), op == Binary::Isub) {
                    (true, false) => 0x5EE0_8400,
                    (true, true) => NEON_SUB_1D,
                    (false, false) => 0x0E20_8400 | (spec.q << 30) | (spec.size << 22),
                    (false, true) => NEON_SUB | (spec.q << 30) | (spec.size << 22),
                };
                self.asm.vbit(base, dst, a, b);
            }
            Binary::Imul if spec.lane != Lane::I64 => {
                self.asm.vbit(0x0E20_9C00 | (spec.q << 30) | (spec.size << 22), dst, a, b);
            }
            Binary::Imul => {
                self.asm.fmov_out(true, A, a);
                self.asm.fmov_out(true, C, b);
                self.asm.madd(false, Size::X, D, (A, C), ZR);
                if spec.count == 2 {
                    self.asm.umov_d1(A, a);
                    self.asm.umov_d1(C, b);
                    self.asm.madd(false, Size::X, S, (A, C), ZR);
                }
                self.clobber_regs(&[A, C, D, S]);
                self.asm.fmov_in(true, dst, D);
                if spec.count == 2 {
                    self.ins_gpr(dst, 3, 1, S);
                }
            }
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax if spec.lane != Lane::I64 => {
                let base = match op {
                    Binary::Smax => NEON_SMAX,
                    Binary::Smin => NEON_SMIN,
                    Binary::Umax => NEON_UMAX,
                    _ => NEON_UMIN,
                };
                self.asm.vbit(base | (spec.q << 30) | (spec.size << 22), dst, a, b);
            }
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                let signed = matches!(op, Binary::Smin | Binary::Smax);
                let cmp = match (spec.is_1d(), signed) {
                    (true, true) => 0x5EE0_3400,
                    (true, false) => 0x7EE0_3400,
                    (false, true) => 0x4EE0_3400,
                    (false, false) => 0x6EE0_3400,
                };
                let (t, f) = if matches!(op, Binary::Smax | Binary::Umax) { (a, b) } else { (b, a) };
                if dst != a && dst != b {
                    self.asm.vbit(cmp, dst, a, b);
                    self.asm.vbit(NEON_BSL | (spec.q << 30), dst, t, f);
                } else {
                    self.clobber_vreg(V2);
                    self.asm.vbit(cmp, V2, a, b);
                    self.asm.vbit(NEON_BSL | (spec.q << 30), V2, t, f);
                    self.asm.vmov(dst, V2);
                }
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                if op == Binary::Ishl {
                    let base = if spec.is_1d() { NEON_USHL_1D } else { NEON_USHL | (spec.q << 30) | (spec.size << 22) };
                    self.asm.vbit(base, dst, a, b);
                } else {
                    self.clobber_vreg(V2);
                    self.v_neg(spec, V2, b);
                    let base = match (spec.is_1d(), op == Binary::Sshr) {
                        (true, false) => NEON_USHL_1D,
                        (true, true) => 0x5EE0_4400,
                        (false, false) => NEON_USHL | (spec.q << 30) | (spec.size << 22),
                        (false, true) => 0x0E20_4400 | (spec.q << 30) | (spec.size << 22),
                    };
                    self.asm.vbit(base, dst, a, V2);
                }
            }
            Binary::Fcopysign => self.vec_fcopysign(spec, dst, a, b),
            Binary::Fadd | Binary::Fsub | Binary::Fmul | Binary::Fdiv | Binary::Fmin | Binary::Fmax if spec.is_1d() => match op {
                Binary::Fadd => self.asm.fop(Fop::Add, true, dst, a, b),
                Binary::Fsub => self.asm.fop(Fop::Sub, true, dst, a, b),
                Binary::Fmul => self.asm.fop(Fop::Mul, true, dst, a, b),
                Binary::Fdiv => self.asm.fop(Fop::Div, true, dst, a, b),
                Binary::Fmin => self.asm.vbit(0x1E60_7800, dst, a, b),
                Binary::Fmax => self.asm.vbit(0x1E60_6800, dst, a, b),
                _ => unreachable!(),
            },
            Binary::Fadd | Binary::Fsub | Binary::Fmul | Binary::Fdiv | Binary::Fmin | Binary::Fmax => {
                let base = match op {
                    Binary::Fadd => 0x0E20_D400,
                    Binary::Fsub => 0x0EA0_D400,
                    Binary::Fmul => 0x2E20_DC00,
                    Binary::Fdiv => 0x2E20_FC00,
                    Binary::Fmax => 0x0E20_C400,
                    Binary::Fmin => 0x0EA0_C400,
                    _ => unreachable!(),
                };
                self.asm.vbit(base | (spec.q << 30) | (spec.fsz() << 22), dst, a, b);
            }
            Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => {
                let base = match (spec.is_1d(), op) {
                    (true, Binary::SaddSat) => 0x5EE0_0C00,
                    (true, Binary::UaddSat) => 0x7EE0_0C00,
                    (true, Binary::SsubSat) => 0x5EE0_2C00,
                    (true, Binary::UsubSat) => 0x7EE0_2C00,
                    (false, Binary::SaddSat) => 0x0E20_0C00 | (spec.q << 30) | (spec.size << 22),
                    (false, Binary::UaddSat) => 0x2E20_0C00 | (spec.q << 30) | (spec.size << 22),
                    (false, Binary::SsubSat) => 0x0E20_2C00 | (spec.q << 30) | (spec.size << 22),
                    (false, Binary::UsubSat) => 0x2E20_2C00 | (spec.q << 30) | (spec.size << 22),
                    _ => unreachable!(),
                };
                self.asm.vbit(base, dst, a, b);
            }
            Binary::AvgRound if spec.lane != Lane::I64 => {
                self.asm.vbit(0x2E20_1400 | (spec.q << 30) | (spec.size << 22), dst, a, b);
            }
            Binary::AvgRound => {
                self.clobber_vreg(V2);
                self.clobber_vreg(V3);
                self.asm.vbit(NEON_ORR | (spec.q << 30), V2, a, b);
                self.asm.vbit(NEON_EOR | (spec.q << 30), V3, a, b);
                self.vec_shift_imm(spec, Binary::Ushr, 1, V3, V3);
                let sub = if spec.is_1d() { NEON_SUB_1D } else { NEON_SUB | (spec.q << 30) | (spec.size << 22) };
                self.asm.vbit(sub, dst, V2, V3);
            }
            other => return Err(format!("{other:?} of a vector")),
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn vec_fma(&mut self, a: Value, b: Value, c: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(a))?;
        let va = self.fin(a, V0);
        let vb = self.fin(b, V1);
        let vc = self.fin(c, V2);
        let dst = self.fdst(result);
        if spec.is_1d() {
            let word = 0x1F40_0000 | u32::from(vb.0) << 16 | u32::from(vc.0) << 10 | u32::from(va.0) << 5 | u32::from(dst.0);
            self.asm.word(word);
        } else {
            let acc = if dst == va || dst == vb { V3 } else { dst };
            if acc == V3 {
                self.clobber_vreg(V3);
            }
            self.asm.vmov(acc, vc);
            self.asm.vbit(0x0E20_CC00 | (spec.q << 30) | (spec.fsz() << 22), acc, va, vb);
            self.asm.vmov(dst, acc);
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn vec_shift_imm(&mut self, spec: VecSpec, op: Binary, imm: i64, dst: Vreg, src: Vreg) {
        let esize = spec.lane.bits();
        let shift = (imm as u32) & (esize - 1);
        if shift == 0 {
            self.asm.vmov(dst, src);
            return;
        }
        let (base, enc_imm) = match (op, spec.is_1d()) {
            (Binary::Ishl, true) => (0x5F00_5400, esize + shift),
            (Binary::Ishl, false) => (0x0F00_5400 | (spec.q << 30), esize + shift),
            (Binary::Sshr, true) => (0x5F00_0400, 2 * esize - shift),
            (Binary::Sshr, false) => (0x0F00_0400 | (spec.q << 30), 2 * esize - shift),
            (Binary::Ushr, true) => (0x7F00_0400, 2 * esize - shift),
            (Binary::Ushr, false) => (0x2F00_0400 | (spec.q << 30), 2 * esize - shift),
            _ => unreachable!(),
        };
        self.v_two(base | (enc_imm << 16), dst, src);
    }

    pub(super) fn is_vec_zero(&self, value: Value) -> bool {
        self.plan.const_of(value) == Some(0) || self.eval_vec_full(value).is_some_and(|(_, lo, hi)| lo == 0 && hi == 0)
    }

    pub(super) fn vec_icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let zero_cmp = if self.is_vec_zero(y) {
            Some((x, cc))
        } else if self.is_vec_zero(x) {
            Some((y, flip_int_cc(cc)))
        } else {
            None
        };
        if let Some((val, zcc)) = zero_cmp {
            match zcc {
                IntCC::UnsignedLessThan => self.asm.veor_zero(dst),
                IntCC::UnsignedGreaterThanOrEqual => self.asm.word(0x0F07_E7E0 | (spec.q << 30) | u32::from(dst.0)),
                IntCC::NotEqual | IntCC::UnsignedGreaterThan => {
                    let a = self.fin(val, V0);
                    let base = if spec.is_1d() { 0x5EE0_8C00 } else { 0x0E20_8C00 | (spec.q << 30) | (spec.size << 22) };
                    self.asm.vbit(base, dst, a, a);
                }
                _ => {
                    let a = self.fin(val, V0);
                    let base = match (spec.is_1d(), zcc) {
                        (true, IntCC::Equal | IntCC::UnsignedLessThanOrEqual) => 0x5EE0_9800,
                        (true, IntCC::SignedGreaterThan) => 0x5EE0_8800,
                        (true, IntCC::SignedGreaterThanOrEqual) => 0x7EE0_8800,
                        (true, IntCC::SignedLessThan) => 0x5EE0_A800,
                        (true, IntCC::SignedLessThanOrEqual) => 0x7EE0_9800,
                        (false, IntCC::Equal | IntCC::UnsignedLessThanOrEqual) => 0x0E20_9800,
                        (false, IntCC::SignedGreaterThan) => 0x0E20_8800,
                        (false, IntCC::SignedGreaterThanOrEqual) => 0x2E20_8800,
                        (false, IntCC::SignedLessThan) => 0x0E20_A800,
                        (false, IntCC::SignedLessThanOrEqual) => 0x2E20_9800,
                        _ => unreachable!(),
                    };
                    let word = if spec.is_1d() { base } else { base | (spec.q << 30) | (spec.size << 22) };
                    self.v_two(word, dst, a);
                }
            }
            self.fput(result, dst);
            return Ok(());
        }
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        let (eq_base, gt_base, ge_base, hi_base, hs_base) = if spec.is_1d() {
            (0x7EE0_8C00, 0x5EE0_3400, 0x5EE0_3C00, 0x7EE0_3400, 0x7EE0_3C00)
        } else {
            let tail = (spec.q << 30) | (spec.size << 22);
            (0x2E20_8C00 | tail, 0x0E20_3400 | tail, 0x0E20_3C00 | tail, 0x2E20_3400 | tail, 0x2E20_3C00 | tail)
        };
        match cc {
            IntCC::Equal => self.asm.vbit(eq_base, dst, a, b),
            IntCC::NotEqual => {
                self.asm.vbit(eq_base, dst, a, b);
                self.v_not(spec.q, dst, dst);
            }
            IntCC::SignedGreaterThan => self.asm.vbit(gt_base, dst, a, b),
            IntCC::SignedGreaterThanOrEqual => self.asm.vbit(ge_base, dst, a, b),
            IntCC::SignedLessThan => self.asm.vbit(gt_base, dst, b, a),
            IntCC::SignedLessThanOrEqual => self.asm.vbit(ge_base, dst, b, a),
            IntCC::UnsignedGreaterThan => self.asm.vbit(hi_base, dst, a, b),
            IntCC::UnsignedGreaterThanOrEqual => self.asm.vbit(hs_base, dst, a, b),
            IntCC::UnsignedLessThan => self.asm.vbit(hi_base, dst, b, a),
            IntCC::UnsignedLessThanOrEqual => self.asm.vbit(hs_base, dst, b, a),
        }
        self.fput(result, dst);
        Ok(())
    }

    pub(super) fn vec_fcmp(&mut self, cc: FloatCC, x: Value, y: Value, result: Value) {
        let Ok(spec) = VecSpec::of(self.ty(x)) else { return };
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        let (eq_base, ge_base, gt_base) = if spec.is_1d() {
            (0x5E60_E400, 0x7E60_E400, 0x7EE0_E400)
        } else {
            let tail = (spec.q << 30) | (spec.fsz() << 22);
            (0x0E20_E400 | tail, 0x2E20_E400 | tail, 0x2EA0_E400 | tail)
        };
        match cc {
            FloatCC::Equal => self.asm.vbit(eq_base, dst, a, b),
            FloatCC::NotEqual => {
                self.asm.vbit(eq_base, dst, a, b);
                self.v_not(spec.q, dst, dst);
            }
            FloatCC::GreaterThan => self.asm.vbit(gt_base, dst, a, b),
            FloatCC::GreaterThanOrEqual => self.asm.vbit(ge_base, dst, a, b),
            FloatCC::LessThan => self.asm.vbit(gt_base, dst, b, a),
            FloatCC::LessThanOrEqual => self.asm.vbit(ge_base, dst, b, a),
        }
        self.fput(result, dst);
    }

    pub(super) fn vec_select_mask(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        let q = u32::from(self.ty(result).bytes() == 16);
        let dst = self.fdst(result);
        let c = self.fin(condition, V0);
        let a = self.fin(x, V1);
        let b = self.fin(y, V2);
        if dst == a && dst != c {
            self.asm.vbit(0x2EE0_1C00 | (q << 30), dst, b, c);
        } else if dst == b && dst != c {
            self.asm.vbit(0x2EA0_1C00 | (q << 30), dst, a, c);
        } else {
            self.bsl(q, c, a, b, dst);
        }
        self.fput(result, dst);
    }

    pub(super) fn vec_select_cond(&mut self, cond: Cond, x: Value, y: Value, result: Value) {
        let q = u32::from(self.ty(result).bytes() == 16);
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        self.asm.csinv(Size::X, cond.invert(), A, ZR, ZR);
        self.clobber_reg(A);
        let target = if dst != a && dst != b { dst } else { V3 };
        if target == V3 {
            self.clobber_vreg(V3);
        }
        self.asm.word(NEON_DUP_GPR_2D | u32::from(A.0) << 5 | u32::from(target.0));
        self.asm.vbit(NEON_BSL | (q << 30), target, a, b);
        self.asm.vmov(dst, target);
        self.fput(result, dst);
    }

    pub(super) fn vec_convert(&mut self, op: Convert, x: Value, result: Value) -> Result<(), String> {
        let (from, to) = (self.ty(x), self.ty(result));
        if op == Convert::Bitcast {
            return self.vec_bitcast(from, to, x, result);
        }
        let fspec = VecSpec::of(from)?;
        let tspec = VecSpec::of(to)?;
        let dst = self.fdst(result);
        let src = self.fin(x, V0);
        match op {
            Convert::Fpromote => self.v_two(0x0E61_7800, dst, src),
            Convert::Fdemote => self.v_two(0x0E61_6800, dst, src),
            Convert::Sextend | Convert::Uextend => {
                let u = u32::from(op == Convert::Uextend);
                self.v_two(0x0F00_A400 | (u << 29) | ((1 << fspec.size) << 19), dst, src);
            }
            Convert::Ireduce => self.v_two(0x0E21_2800 | (tspec.size << 22), dst, src),
            Convert::NarrowSat { signed } => {
                let base = if signed { 0x0E21_4800 } else { 0x2E21_2800 };
                self.v_two(base | (tspec.size << 22), dst, src);
            }
            Convert::FromSint | Convert::FromUint => {
                let u = u32::from(op == Convert::FromUint);
                if fspec.size == tspec.size {
                    let base = if tspec.is_1d() {
                        0x5E61_D800 | (u << 29)
                    } else {
                        0x0E21_D800 | (tspec.q << 30) | (u << 29) | (tspec.fsz() << 22)
                    };
                    self.v_two(base, dst, src);
                } else if fspec.size < tspec.size {
                    self.v_two(0x0F00_A400 | (u << 29) | ((1 << fspec.size) << 19), dst, src);
                    self.v_two(0x0E21_D800 | (tspec.q << 30) | (u << 29) | (tspec.fsz() << 22), dst, dst);
                } else {
                    self.v_two(0x0E21_D800 | (fspec.q << 30) | (u << 29) | (1 << 22), dst, src);
                    self.v_two(0x0E61_6800, dst, dst);
                }
            }
            Convert::ToSintSat | Convert::ToUintSat => {
                let u = u32::from(op == Convert::ToUintSat);
                let base = if fspec.is_1d() {
                    0x5EE1_B800 | (u << 29)
                } else {
                    0x0EA1_B800 | (fspec.q << 30) | (u << 29) | (fspec.fsz() << 22)
                };
                self.v_two(base, dst, src);
            }
            other => return Err(format!("{other:?} from {from:?} to {to:?}")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vec_bitcast(&mut self, from: Type, to: Type, x: Value, result: Value) -> Result<(), String> {
        if from.bytes() != to.bytes() {
            return Err(format!("bitcast from {from:?} to {to:?}"));
        }
        if to.is_vector() {
            let dst = self.fdst(result);
            match from {
                Type::Vec { .. } | Type::F64 => {
                    let src = self.fin(x, dst);
                    self.asm.vmov(dst, src);
                }
                Type::I64 => self.emit_vec_from_i64(dst, x),
                Type::I128 => {
                    self.raw(A, x);
                    self.high(C, x);
                    self.asm.fmov_in(true, dst, A);
                    self.ins_gpr(dst, 3, 1, C);
                }
                _ => return Err(format!("bitcast from {from:?} to {to:?}")),
            }
            self.fput(result, dst);
            return Ok(());
        }
        let src = self.fin(x, V0);
        match to {
            Type::F64 => {
                let dst = self.fdst(result);
                self.asm.fmov_reg(true, dst, src);
                self.fput(result, dst);
            }
            Type::I64 => {
                let dst = self.dst(result);
                self.asm.fmov_out(true, dst, src);
                self.clobber_reg(dst);
                self.put(result, dst);
            }
            Type::I128 => {
                self.asm.fmov_out(true, A, src);
                self.clobber_reg(A);
                self.put(result, A);
                self.asm.umov_d1(A, src);
                self.clobber_reg(A);
                self.put_high(result, A);
            }
            _ => return Err(format!("bitcast from {from:?} to {to:?}")),
        }
        Ok(())
    }
}
