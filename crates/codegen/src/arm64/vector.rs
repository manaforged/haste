use super::asm::{Alu, Bits, Cond, Fop, Funary, Reg, Size, Vreg, ZR};
use super::{A, C, D, Emit, S, V0, V1};
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
        self.asm.word(base | (imm5 << 16) | u32::from(src.0) << 5 | u32::from(dst.0));
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
                x = V1;
            }
            if y == dst {
                self.asm.vmov(V2, y);
                y = V2;
            }
            self.asm.vmov(dst, mask);
            dst
        };
        self.asm.vbit(0x2E60_1C00 | (q << 30), out, x, y);
    }

    fn splat_const(&self, value: Value) -> Option<i64> {
        let inst = self.defs[value.index()]?;
        let mut cur = &self.func.insts[inst.index()].op;
        if let Op::Shuffle(a, b, ShuffleMask::Const(m)) = cur {
            let Type::Vec { count, .. } = self.ty(value) else { return None };
            let count = usize::from(count);
            let first = m[0];
            if !m[..count].iter().all(|&x| x == first) {
                return None;
            }
            let src = if usize::from(first) < count { *a } else { *b };
            cur = &self.func.insts[self.defs[src.index()]?.index()].op;
            if let Op::Insert(_, scalar, idx) = *cur
                && usize::from(idx) == usize::from(first) % count
            {
                return self.plan.const_of(scalar);
            }
            return None;
        }
        if let Op::Insert(_, scalar, 0) = *cur
            && matches!(self.ty(value), Type::Vec { count: 1, .. })
        {
            return self.plan.const_of(scalar);
        }
        None
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
            Op::Shuffle(a, b, mask) => self.vec_shuffle(a, b, mask, result()?)?,
            Op::Extract(vec, idx) => self.vec_extract(vec, idx, result()?)?,
            Op::Insert(vec, val, idx) => self.vec_insert(vec, val, idx, result()?)?,
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
            Lane::I16 => 0x0E20_1800,
            Lane::I32 | Lane::F32 => 0x2E20_0800,
            Lane::I64 | Lane::F64 => 0x0E20_0800,
        };
        self.v_two(base | (spec.q << 30), dst, src);
    }

    fn vec_clz(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        if spec.lane != Lane::I64 {
            self.v_two(0x2E20_4800 | (spec.q << 30) | (spec.size << 22), dst, src);
            return;
        }
        self.asm.fmov_out(true, A, src);
        self.asm.bits(Bits::Clz, A, A);
        if spec.count == 2 {
            self.asm.umov_d1(C, src);
            self.asm.bits(Bits::Clz, C, C);
        }
        self.asm.fmov_in(true, dst, A);
        if spec.count == 2 {
            self.ins_gpr(dst, 3, 1, C);
        }
    }

    fn vec_ctz_i64(&mut self, spec: VecSpec, dst: Vreg, src: Vreg) {
        self.asm.fmov_out(true, A, src);
        self.asm.bits(Bits::Rbit, A, A);
        self.asm.bits(Bits::Clz, A, A);
        if spec.count == 2 {
            self.asm.umov_d1(C, src);
            self.asm.bits(Bits::Rbit, C, C);
            self.asm.bits(Bits::Clz, C, C);
        }
        self.asm.fmov_in(true, dst, A);
        if spec.count == 2 {
            self.ins_gpr(dst, 3, 1, C);
        }
    }

    pub(super) fn vec_binary(&mut self, op: Binary, x: Value, y: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        match op {
            Binary::Band | Binary::Bor | Binary::Bxor => {
                let base = match op {
                    Binary::Band => 0x0E20_1C00,
                    Binary::Bor => 0x0EA0_1C00,
                    _ => 0x2E20_1C00,
                };
                self.asm.vbit(base | (spec.q << 30), dst, a, b);
            }
            Binary::Iadd | Binary::Isub => {
                let base = match (spec.is_1d(), op == Binary::Isub) {
                    (true, false) => 0x5EE0_8400,
                    (true, true) => 0x7EE0_8400,
                    (false, false) => 0x0E20_8400 | (spec.q << 30) | (spec.size << 22),
                    (false, true) => 0x2E20_8400 | (spec.q << 30) | (spec.size << 22),
                };
                self.asm.vbit(base, dst, a, b);
            }
            Binary::Imul if spec.lane != Lane::I64 => {
                self.asm.vbit(0x0E20_9C00 | (spec.q << 30) | (spec.size << 22), dst, a, b);
            }
            Binary::Imul => {
                self.asm.fmov_out(true, A, a);
                self.asm.fmov_out(true, C, b);
                self.asm.madd(false, D, (A, C), ZR);
                if spec.count == 2 {
                    self.asm.umov_d1(A, a);
                    self.asm.umov_d1(C, b);
                    self.asm.madd(false, S, (A, C), ZR);
                }
                self.asm.fmov_in(true, dst, D);
                if spec.count == 2 {
                    self.ins_gpr(dst, 3, 1, S);
                }
            }
            Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax if spec.lane != Lane::I64 => {
                let base = match op {
                    Binary::Smax => 0x0E20_6400,
                    Binary::Smin => 0x0E20_6C00,
                    Binary::Umax => 0x2E20_6400,
                    _ => 0x2E20_6C00,
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
                self.asm.vbit(cmp, V2, a, b);
                let (t, f) = if matches!(op, Binary::Smax | Binary::Umax) { (a, b) } else { (b, a) };
                self.asm.vbit(0x2E60_1C00 | (spec.q << 30), V2, t, f);
                self.asm.vmov(dst, V2);
            }
            Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                if self.plan.const_of(y) == Some(0) {
                    self.asm.vmov(dst, a);
                } else if let Some(imm) = self.splat_const(y) {
                    self.vec_shift_imm(spec, op, imm, dst, a);
                } else if op == Binary::Ishl {
                    let base = if spec.is_1d() { 0x7EE0_4400 } else { 0x2E20_4400 | (spec.q << 30) | (spec.size << 22) };
                    self.asm.vbit(base, dst, a, b);
                } else {
                    self.v_neg(spec, V2, b);
                    let base = match (spec.is_1d(), op == Binary::Sshr) {
                        (true, false) => 0x7EE0_4400,
                        (true, true) => 0x5EE0_4400,
                        (false, false) => 0x2E20_4400 | (spec.q << 30) | (spec.size << 22),
                        (false, true) => 0x0E20_4400 | (spec.q << 30) | (spec.size << 22),
                    };
                    self.asm.vbit(base, dst, a, V2);
                }
            }
            Binary::Fcopysign => {
                let sign = if spec.lane == Lane::F64 { i64::MIN } else { 0x8000_0000_u32 as i64 };
                self.asm.mov_imm(A, sign);
                let dup_imm5 = if spec.lane == Lane::F64 { 8 } else { 4 };
                self.asm.word(0x4E00_0C00 | (dup_imm5 << 16) | u32::from(A.0) << 5 | u32::from(V2.0));
                self.asm.vmov(V3, a);
                self.asm.vbit(0x2EA0_1C00 | (spec.q << 30), V3, b, V2);
                self.asm.vmov(dst, V3);
            }
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
            other => return Err(format!("{other:?} of a vector")),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vec_shift_imm(&mut self, spec: VecSpec, op: Binary, imm: i64, dst: Vreg, src: Vreg) {
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

    pub(super) fn vec_icmp(&mut self, cc: IntCC, x: Value, y: Value, result: Value) -> Result<(), String> {
        let spec = VecSpec::of(self.ty(x))?;
        let dst = self.fdst(result);
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
        let target = if dst == a || dst == b {
            self.asm.vmov(V3, c);
            self.bsl(q, V3, a, b, V3);
            self.asm.vmov(dst, V3);
            dst
        } else {
            self.bsl(q, c, a, b, dst);
            dst
        };
        self.fput(result, target);
    }

    pub(super) fn vec_select_cond(&mut self, cond: Cond, x: Value, y: Value, result: Value) {
        let q = u32::from(self.ty(result).bytes() == 16);
        let dst = self.fdst(result);
        let a = self.fin(x, V0);
        let b = self.fin(y, V1);
        self.asm.cset(cond, A);
        self.asm.alu(Alu::Sub, Size::X, A, ZR, A);
        self.asm.word(0x4E08_0C00 | u32::from(A.0) << 5 | u32::from(V3.0));
        self.bsl(q, V3, a, b, V3);
        self.asm.vmov(dst, V3);
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
                Type::I64 => {
                    let src = self.bits(x, A);
                    self.asm.fmov_in(true, dst, src);
                }
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
                self.put(result, dst);
            }
            Type::I128 => {
                self.asm.fmov_out(true, A, src);
                self.put(result, A);
                self.asm.umov_d1(A, src);
                self.put_high(result, A);
            }
            _ => return Err(format!("bitcast from {from:?} to {to:?}")),
        }
        Ok(())
    }
}
