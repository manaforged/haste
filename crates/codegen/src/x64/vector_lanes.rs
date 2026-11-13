use super::asm::{Alu, Cond, Group3, Mem, Reg, Shift, Size, Xmm};
use super::float::{X0, X1, X2};
use super::int::cond;
use super::vector::{vec_alu_opcode, vec_prefix};
use super::Emit;
use crate::frame::Ext;
use crate::ir::{Binary, IntCC, Lane, Op, ReduceOp, ShuffleMask, TargetOpId, Type, Value};

#[derive(Clone, Copy)]
enum TargetOpForm {
    Unary,
    Binary,
    Sha256Rnds2,
}

struct TargetOpSpec {
    id: TargetOpId,
    form: TargetOpForm,
    prefix: &'static [u8],
    opcode: &'static [u8],
}

const X64_TARGET_OPS: &[TargetOpSpec] = &[
    TargetOpSpec { id: TargetOpId::X64Sha1NextE, form: TargetOpForm::Binary, prefix: &[], opcode: &[0x0F, 0x38, 0xC8] },
    TargetOpSpec { id: TargetOpId::X64Sha1Msg1, form: TargetOpForm::Binary, prefix: &[], opcode: &[0x0F, 0x38, 0xC9] },
    TargetOpSpec { id: TargetOpId::X64Sha1Msg2, form: TargetOpForm::Binary, prefix: &[], opcode: &[0x0F, 0x38, 0xCA] },
    TargetOpSpec { id: TargetOpId::X64Sha256Rnds2, form: TargetOpForm::Sha256Rnds2, prefix: &[], opcode: &[0x0F, 0x38, 0xCB] },
    TargetOpSpec { id: TargetOpId::X64Sha256Msg1, form: TargetOpForm::Binary, prefix: &[], opcode: &[0x0F, 0x38, 0xCC] },
    TargetOpSpec { id: TargetOpId::X64Sha256Msg2, form: TargetOpForm::Binary, prefix: &[], opcode: &[0x0F, 0x38, 0xCD] },
    TargetOpSpec { id: TargetOpId::X64AesEnc, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0xDC] },
    TargetOpSpec { id: TargetOpId::X64AesEncLast, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0xDD] },
    TargetOpSpec { id: TargetOpId::X64AesDec, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0xDE] },
    TargetOpSpec { id: TargetOpId::X64AesDecLast, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0xDF] },
    TargetOpSpec { id: TargetOpId::X64AesImc, form: TargetOpForm::Unary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0xDB] },
    TargetOpSpec { id: TargetOpId::X64Pmaddwd, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0xF5] },
    TargetOpSpec { id: TargetOpId::X64Pmaddubsw, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0x04] },
    TargetOpSpec { id: TargetOpId::X64Psadbw, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0xF6] },
    TargetOpSpec { id: TargetOpId::X64Pavgb, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0xE0] },
    TargetOpSpec { id: TargetOpId::X64Pavgw, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0xE3] },
    TargetOpSpec { id: TargetOpId::X64Packsswb, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x63] },
    TargetOpSpec { id: TargetOpId::X64Packssdw, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x6B] },
    TargetOpSpec { id: TargetOpId::X64Packuswb, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x67] },
    TargetOpSpec { id: TargetOpId::X64Packusdw, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0x38, 0x2B] },
];

impl Emit<'_> {
    pub(super) fn vector_op(&mut self, op: &Op, result: Value) -> Result<(), String> {
        match op {
            Op::Shuffle(a, b, mask) => self.vector_shuffle(*a, *b, mask, result),
            Op::Extract(v, lane) => self.vector_extract(*v, *lane, result),
            Op::Insert(v, x, lane) => self.vector_insert(*v, *x, *lane, result),
            Op::Reduce(rop, v) => self.vector_reduce(*rop, *v, result),
            Op::TargetOp(id, args) => self.vector_target_op(*id, args.as_slice(), result),
            other => Err(format!("unexpected op {other:?} in vector_op")),
        }
    }

    fn load_bytes16(&mut self, dst: Xmm, scratch: Xmm, bytes: [u8; 16]) {
        let lo = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]) as i64;
        let hi = u64::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]]) as i64;
        self.asm.mov_imm(Reg::Rax, lo);
        self.asm.movq_in(dst, Reg::Rax);
        if hi != 0 {
            self.asm.mov_imm(Reg::Rax, hi);
            self.asm.movq_in(scratch, Reg::Rax);
            self.asm.sse(&[0x66], &[0x0F, 0x6C], dst.0, scratch);
        }
    }

    fn vector_shuffle(&mut self, a: Value, b: Value, mask: &ShuffleMask, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(a).vec_shape() else {
            return Err(format!("non-vector type {:?} in shuffle", self.ty(a)));
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        match mask {
            ShuffleMask::Const(indices) => {
                self.vector_shuffle_const(lane, count, a, b, indices, dst);
            }
            ShuffleMask::Dyn(mask_v) if self.features.ssse3 && lane == Lane::I8 => {
                self.vector_shuffle_dyn_ssse3(count, a, b, *mask_v, dst);
            }
            ShuffleMask::Dyn(mask_v) => {
                self.vector_shuffle_dyn_baseline(lane, count, a, b, *mask_v, dst);
            }
        }
        self.fput(result, dst);
        Ok(())
    }

    fn vector_shuffle_const(&mut self, lane: Lane, count: u8, a: Value, b: Value, indices: &[u8; 16], dst: Xmm) {
        let c = usize::from(count);
        let all_a = indices[..c].iter().all(|&i| i < count);
        let all_b = indices[..c].iter().all(|&i| i >= count && i < 2 * count);
        if all_a || all_b {
            let src_val = if all_a { a } else { b };
            let off = if all_a { 0 } else { count };
            let imm = match (lane.bytes(), count) {
                (4, 4) => Some(
                    ((indices[0] - off) & 3)
                        | (((indices[1] - off) & 3) << 2)
                        | (((indices[2] - off) & 3) << 4)
                        | (((indices[3] - off) & 3) << 6),
                ),
                (8, 2) => {
                    let l0 = (indices[0] - off) * 2;
                    let l1 = (indices[1] - off) * 2;
                    Some(l0 | ((l0 + 1) << 2) | (l1 << 4) | ((l1 + 1) << 6))
                }
                _ => None,
            };
            if let Some(imm) = imm {
                let src = self.fin(src_val, X0);
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, src, imm);
                return;
            }
            if lane.bytes() == 2 && count == 4 {
                let imm = ((indices[0] - off) & 3)
                    | (((indices[1] - off) & 3) << 2)
                    | (((indices[2] - off) & 3) << 4)
                    | (((indices[3] - off) & 3) << 6);
                let src = self.fin(src_val, X0);
                self.asm.sse_imm(&[0xF2], &[0x0F, 0x70], dst.0, src, imm);
                return;
            }
        }
        if self.features.ssse3 {
            let bytes = lane.bytes() as usize;
            let mut mask_a = [0x80_u8; 16];
            let mut mask_b = [0x80_u8; 16];
            let mut uses_a = false;
            let mut uses_b = false;
            for l in 0..c {
                let idx = indices[l];
                if idx < count {
                    uses_a = true;
                    for k in 0..bytes {
                        mask_a[l * bytes + k] = idx * bytes as u8 + k as u8;
                    }
                } else if idx < 2 * count {
                    uses_b = true;
                    let bi = idx - count;
                    for k in 0..bytes {
                        mask_b[l * bytes + k] = bi * bytes as u8 + k as u8;
                    }
                }
            }
            if !uses_a && !uses_b {
                self.zero_xmm(dst);
                return;
            }
            if uses_a {
                self.load_bytes16(X1, X2, mask_a);
                self.fget(X0, a);
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X0.0, X1);
            } else {
                self.zero_xmm(X0);
            }
            if uses_b {
                self.load_bytes16(X1, X2, mask_b);
                self.fget(X2, b);
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X2.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X2);
            }
            if dst != X0 {
                self.asm.movaps(dst, X0);
            }
            return;
        }
        let bytes = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 48);
        self.fget(X0, a);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
        self.fget(X0, b);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 16));
        for l in 0..c {
            let idx = indices[l];
            if idx < count {
                self.asm.load(size, Reg::Rax, Mem::new(Reg::Rsp, i32::from(idx) * bytes));
            } else if idx < 2 * count {
                self.asm.load(size, Reg::Rax, Mem::new(Reg::Rsp, 16 + i32::from(idx - count) * bytes));
            } else {
                self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
            }
            self.asm.store(size, Mem::new(Reg::Rsp, 32 + l as i32 * bytes), Reg::Rax);
        }
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 32));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 48);
    }

    fn vector_shuffle_dyn_ssse3(&mut self, count: u8, a: Value, b: Value, mask_v: Value, dst: Xmm) {
        self.fget(X0, mask_v);
        self.splat_imm(Lane::I8, X1, i64::from(count - 1));
        self.asm.movaps(X2, X0);
        self.asm.sse(&[0x66], &[0x0F, 0x64], X2.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], X2.0, X0);
        self.fget(X0, a);
        self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X0.0, X2);
        let save_stack = dst == X2 || self.xmm_of(mask_v) == Some(dst) || self.xmm_of(b) == Some(dst);
        if save_stack {
            self.asm.alu_imm(Alu::Sub, Reg::Rsp, 16);
            self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
        } else {
            self.asm.movaps(dst, X0);
        }
        self.fget(X0, mask_v);
        self.splat_imm(Lane::I8, X1, i64::from(count));
        self.asm.sse(&[0x66], &[0x0F, 0xF8], X0.0, X1);
        self.splat_imm(Lane::I8, X1, i64::from(count - 1));
        self.asm.movaps(X2, X0);
        self.asm.sse(&[0x66], &[0x0F, 0x64], X2.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], X2.0, X0);
        self.fget(X1, b);
        self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X1.0, X2);
        if save_stack {
            self.asm.sse(&[0xF3], &[0x0F, 0x6F], X0.0, Mem::new(Reg::Rsp, 0));
            self.asm.alu_imm(Alu::Add, Reg::Rsp, 16);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X1);
            self.asm.movaps(dst, X0);
        } else {
            self.asm.sse(&[0x66], &[0x0F, 0xEB], dst.0, X1);
        }
    }

    fn vector_shuffle_dyn_baseline(&mut self, lane: Lane, count: u8, a: Value, b: Value, mask_v: Value, dst: Xmm) {
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 64);
        self.fget(X0, a);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
        self.fget(X0, b);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 16));
        self.fget(X0, mask_v);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 32));
        self.asm.alu32(Alu::Xor, Reg::Rcx, Reg::Rcx);
        let (head, from_b, zero, store) = (self.asm.label(), self.asm.label(), self.asm.label(), self.asm.label());
        self.asm.bind(head);
        self.asm.load(Size::B, Reg::Rax, Mem::sib(Reg::Rsp, Reg::Rcx, 0, 32));
        self.asm.alu_imm32(Alu::Cmp, Reg::Rax, i32::from(count));
        self.asm.jcc(Cond::Ae, from_b);
        self.asm.load(Size::B, Reg::Rax, Mem::sib(Reg::Rsp, Reg::Rax, 0, 0));
        self.asm.jmp(store);
        self.asm.bind(from_b);
        self.asm.alu_imm32(Alu::Cmp, Reg::Rax, i32::from(count) * 2);
        self.asm.jcc(Cond::Ae, zero);
        self.asm.load(Size::B, Reg::Rax, Mem::sib(Reg::Rsp, Reg::Rax, 0, 16 - i32::from(count)));
        self.asm.jmp(store);
        self.asm.bind(zero);
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        self.asm.bind(store);
        self.asm.store(Size::B, Mem::sib(Reg::Rsp, Reg::Rcx, 0, 48), Reg::Rax);
        self.asm.alu_imm(Alu::Add, Reg::Rcx, 1);
        self.asm.alu_imm32(Alu::Cmp, Reg::Rcx, i32::from(count));
        self.asm.jcc(Cond::B, head);
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 48));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 64);
    }

    fn vector_extract(&mut self, v: Value, idx: u8, result: Value) -> Result<(), String> {
        let Some((lane, _)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in extract", self.ty(v)));
        };
        if lane.is_float() {
            let dst = self.fdst(result);
            let src = self.fin(v, if idx == 0 { dst } else { X0 });
            match (lane, idx) {
                (_, 0) if dst != src => self.asm.movaps(dst, src),
                (_, 0) => {}
                (Lane::F32, _) => self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, src, idx),
                (Lane::F64, _) => self.asm.sse_imm(&[0x66], &[0x0F, 0x70], dst.0, src, 0xEE),
                _ => {}
            }
            self.fput(result, dst);
            return Ok(());
        }
        let dst = self.dst(result);
        let src = self.fin(v, X0);
        match (lane, idx) {
            (Lane::I16, _) => self.asm.sse_imm(&[0x66], &[0x0F, 0xC5], dst as u8, src, idx),
            (Lane::I8, 0) => {
                self.asm.movd_out(dst, src);
                self.asm.load(Size::B, dst, dst);
            }
            (Lane::I32, 0) => self.asm.movd_out(dst, src),
            (Lane::I64, 0) => self.asm.movq_out(dst, src),
            (Lane::I8, _) if self.features.sse41 => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x14], src.0, dst, idx);
            }
            (Lane::I8, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0xC5], dst as u8, src, idx / 2);
                if idx & 1 != 0 {
                    self.asm.shift_imm(Size::D, Shift::Shr, dst, 8);
                }
                self.asm.load(Size::B, dst, dst);
            }
            (Lane::I32, _) if self.features.sse41 => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x16], src.0, dst, idx);
            }
            (Lane::I32, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X1.0, src, idx);
                self.asm.movd_out(dst, X1);
            }
            (Lane::I64, _) if self.features.sse41 => {
                self.asm.sse_wide_imm(&[0x66], &[0x0F, 0x3A, 0x16], src.0, dst, idx);
            }
            (Lane::I64, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X1.0, src, 0xEE);
                self.asm.movq_out(dst, X1);
            }
            _ => unreachable!(),
        }
        self.put(result, dst);
        Ok(())
    }

    fn vector_insert(&mut self, v: Value, x: Value, idx: u8, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in insert", self.ty(v)));
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        if lane.is_float() {
            let mut s = self.fin(x, X1);
            if s == dst {
                self.asm.movaps(X1, s);
                s = X1;
            }
            self.fget(dst, v);
            match (lane, idx) {
                (Lane::F32, 0) => self.asm.sse(&[0xF3], &[0x0F, 0x10], dst.0, s),
                (Lane::F32, _) if self.features.sse41 => {
                    self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x21], dst.0, s, idx << 4);
                }
                (Lane::F32, _) => {
                    self.asm.movd_out(Reg::Rax, s);
                    self.insert_stack(lane, count, dst, idx);
                }
                (Lane::F64, 0) => self.asm.sse(&[0xF2], &[0x0F, 0x10], dst.0, s),
                (Lane::F64, _) => self.asm.sse(&[0x66], &[0x0F, 0x14], dst.0, s),
                _ => {}
            }
            self.fput(result, dst);
            return Ok(());
        }
        self.fget(dst, v);
        self.get(Reg::Rax, x, Ext::Zero);
        match lane {
            Lane::I16 => self.asm.sse_imm(&[0x66], &[0x0F, 0xC4], dst.0, Reg::Rax, idx),
            Lane::I8 if self.features.sse41 => self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x20], dst.0, Reg::Rax, idx),
            Lane::I32 if self.features.sse41 => self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x22], dst.0, Reg::Rax, idx),
            Lane::I64 if self.features.sse41 => self.asm.sse_wide_imm(&[0x66], &[0x0F, 0x3A, 0x22], dst.0, Reg::Rax, idx),
            _ => self.insert_stack(lane, count, dst, idx),
        }
        self.fput(result, dst);
        Ok(())
    }

    fn insert_stack(&mut self, lane: Lane, count: u8, dst: Xmm, idx: u8) {
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 16);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], dst.0, Mem::new(Reg::Rsp, 0));
        self.asm.store(Size::of(lane.bytes()), Mem::new(Reg::Rsp, i32::from(idx) * lane.bytes() as i32), Reg::Rax);
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 0));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 16);
    }

    fn vector_reduce(&mut self, op: ReduceOp, v: Value, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in reduce", self.ty(v)));
        };
        if op == ReduceOp::Movemask {
            let dst = self.dst(result);
            let src = self.fin(v, X0);
            match lane {
                Lane::I8 => {
                    self.asm.sse(&[0x66], &[0x0F, 0xD7], dst as u8, src);
                    if count < 16 {
                        self.asm.alu_imm32(Alu::And, dst, (1_i32 << count) - 1);
                    }
                }
                Lane::I16 => {
                    self.asm.movaps(X0, src);
                    self.asm.sse(&[0x66], &[0x0F, 0x63], X0.0, X0);
                    self.asm.sse(&[0x66], &[0x0F, 0xD7], dst as u8, X0);
                    self.asm.alu_imm32(Alu::And, dst, (1_i32 << count) - 1);
                }
                Lane::I32 | Lane::F32 => {
                    self.asm.sse(&[], &[0x0F, 0x50], dst as u8, src);
                    if count < 4 {
                        self.asm.alu_imm32(Alu::And, dst, (1_i32 << count) - 1);
                    }
                }
                Lane::I64 | Lane::F64 => {
                    self.asm.sse(&[0x66], &[0x0F, 0x50], dst as u8, src);
                    if count < 2 {
                        self.asm.alu_imm32(Alu::And, dst, (1_i32 << count) - 1);
                    }
                }
            }
            self.put(result, dst);
            return Ok(());
        }
        self.fget(X0, v);
        if lane == Lane::I64 && matches!(op, ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax) {
            let dst = self.dst(result);
            self.asm.movaps(X1, X0);
            self.asm.pshift_imm(3, 8, X1, 8);
            self.asm.movq_out(dst, X0);
            self.asm.movq_out(Reg::Rcx, X1);
            self.asm.alu(Alu::Cmp, dst, Reg::Rcx);
            let cc = match op {
                ReduceOp::Smin => Cond::G,
                ReduceOp::Smax => Cond::L,
                ReduceOp::Umin => Cond::A,
                _ => Cond::B,
            };
            self.asm.cmov(cc, dst, Reg::Rcx);
            self.put(result, dst);
            return Ok(());
        }
        let mut active = count;
        while active > 1 {
            let half = active / 2;
            let shift_bytes = half * lane.bytes() as u8;
            self.asm.movaps(X1, X0);
            self.asm.pshift_imm(3, 8, X1, shift_bytes);
            self.reduce_step(op, lane)?;
            active = half;
        }
        if lane.is_float() {
            let dst = self.fdst(result);
            if dst != X0 {
                self.asm.movaps(dst, X0);
            }
            self.fput(result, dst);
        } else {
            let dst = self.dst(result);
            match lane {
                Lane::I8 => {
                    self.asm.movd_out(dst, X0);
                    self.asm.load(Size::B, dst, dst);
                }
                Lane::I16 => {
                    self.asm.movd_out(dst, X0);
                    self.asm.load(Size::W, dst, dst);
                }
                Lane::I32 => self.asm.movd_out(dst, X0),
                Lane::I64 => self.asm.movq_out(dst, X0),
                _ => {}
            }
            self.put(result, dst);
        }
        Ok(())
    }

    fn reduce_step(&mut self, op: ReduceOp, lane: Lane) -> Result<(), String> {
        match op {
            ReduceOp::Add if lane.is_float() => self.asm.sse(vec_prefix(lane), &[0x0F, 0x58], X0.0, X1),
            ReduceOp::Add => {
                let opc = vec_alu_opcode(Binary::Iadd, lane).ok_or_else(|| format!("reduce add {lane:?}"))?;
                self.asm.sse(&[0x66], &[0x0F, opc], X0.0, X1);
            }
            ReduceOp::And => self.asm.sse(&[0x66], &[0x0F, 0xDB], X0.0, X1),
            ReduceOp::Or => self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X1),
            ReduceOp::Xor => self.asm.sse(&[0x66], &[0x0F, 0xEF], X0.0, X1),
            ReduceOp::Fmin | ReduceOp::Fmax => {
                let bin = if op == ReduceOp::Fmin { Binary::Fmin } else { Binary::Fmax };
                self.fminmax_regs(bin, lane, X2);
                self.asm.movaps(X0, X2);
            }
            ReduceOp::Smin | ReduceOp::Smax | ReduceOp::Umin | ReduceOp::Umax => {
                let (bin, cc) = match op {
                    ReduceOp::Smin => (Binary::Smin, IntCC::SignedLessThan),
                    ReduceOp::Smax => (Binary::Smax, IntCC::SignedGreaterThan),
                    ReduceOp::Umin => (Binary::Umin, IntCC::UnsignedLessThan),
                    _ => (Binary::Umax, IntCC::UnsignedGreaterThan),
                };
                if let Some(bytes) = self.iminmax_opc(bin, lane) {
                    self.asm.sse(&[0x66], bytes, X0.0, X1);
                } else {
                    self.asm.alu_imm(Alu::Sub, Reg::Rsp, 32);
                    self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
                    self.asm.sse(&[0xF3], &[0x0F, 0x7F], X1.0, Mem::new(Reg::Rsp, 16));
                    self.emit_icmp_regs(cc, lane, X0, X1, X2);
                    self.asm.sse(&[0xF3], &[0x0F, 0x6F], X0.0, Mem::new(Reg::Rsp, 0));
                    self.asm.sse(&[0xF3], &[0x0F, 0x6F], X1.0, Mem::new(Reg::Rsp, 16));
                    self.asm.alu_imm(Alu::Add, Reg::Rsp, 32);
                    self.blend_regs(X2, X0, X1, X0);
                }
            }
            ReduceOp::Movemask => {}
        }
        Ok(())
    }

    fn vector_target_op(&mut self, id: TargetOpId, args: &[Value], result: Value) -> Result<(), String> {
        let spec = X64_TARGET_OPS.iter().find(|s| s.id == id).ok_or_else(|| format!("unsupported target op {id:?}"))?;
        match spec.form {
            TargetOpForm::Unary => {
                let [x] = *args else { return Err(format!("{id:?} expects 1 arg")); };
                let dst = self.fdst(result);
                let src = self.fin(x, X0);
                self.asm.sse(spec.prefix, spec.opcode, dst.0, src);
                self.fput(result, dst);
            }
            TargetOpForm::Binary => {
                let [x, y] = *args else { return Err(format!("{id:?} expects 2 args")); };
                self.packed_op(spec.prefix, spec.opcode, (x, y), result);
            }
            TargetOpForm::Sha256Rnds2 => {
                let [a, b, k] = *args else { return Err(format!("{id:?} expects 3 args")); };
                let dst = self.xmm_of(result).unwrap_or(X2);
                self.fget(dst, a);
                self.fget(X1, b);
                self.fget(X0, k);
                self.asm.sse(spec.prefix, spec.opcode, dst.0, X1);
                self.fput(result, dst);
            }
        }
        Ok(())
    }

    pub(super) fn scalarize_int_binary(&mut self, op: Binary, lane: Lane, count: u8, x: Value, y: Value, dst: Xmm) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        let signed_load = matches!(op, Binary::Smin | Binary::Smax | Binary::Sshr);
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 32);
        self.fget(X0, x);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 0), X0);
        self.fget(X0, y);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 16), X0);
        for i in 0..i32::from(count) {
            let sx = Mem::new(Reg::Rsp, i * stride);
            let sy = Mem::new(Reg::Rsp, 16 + i * stride);
            if signed_load {
                self.asm.load_signed(size, Reg::Rax, sx);
                self.asm.load_signed(size, Reg::Rcx, sy);
            } else {
                self.asm.load(size, Reg::Rax, sx);
                self.asm.load(size, Reg::Rcx, sy);
            }
            match op {
                Binary::Imul => self.asm.imul(Reg::Rax, Reg::Rcx),
                Binary::Smin | Binary::Smax | Binary::Umin | Binary::Umax => {
                    self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
                    let cc = match op {
                        Binary::Smin => Cond::G,
                        Binary::Smax => Cond::L,
                        Binary::Umin => Cond::A,
                        _ => Cond::B,
                    };
                    self.asm.cmov(cc, Reg::Rax, Reg::Rcx);
                }
                Binary::Ishl | Binary::Ushr | Binary::Sshr => {
                    self.asm.alu_imm32(Alu::And, Reg::Rcx, (lane.bits() - 1) as i32);
                    let shift = match op {
                        Binary::Ishl => Shift::Shl,
                        Binary::Ushr => Shift::Shr,
                        _ => Shift::Sar,
                    };
                    self.asm.shift(size, shift, Reg::Rax);
                }
                _ => {}
            }
            self.asm.store(size, sx, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 32);
    }

    pub(super) fn scalarize_int_icmp(&mut self, cc: IntCC, lane: Lane, count: u8, x: Value, y: Value, dst: Xmm) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, 32);
        self.fget(X0, x);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 0), X0);
        self.fget(X0, y);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 16), X0);
        for i in 0..i32::from(count) {
            let sx = Mem::new(Reg::Rsp, i * stride);
            let sy = Mem::new(Reg::Rsp, 16 + i * stride);
            self.asm.load(Size::Q, Reg::Rax, sx);
            self.asm.load(Size::Q, Reg::Rcx, sy);
            self.asm.alu(Alu::Cmp, Reg::Rax, Reg::Rcx);
            self.asm.setcc(cond(cc), Reg::Rax);
            self.asm.load(Size::B, Reg::Rax, Reg::Rax);
            self.asm.group3(Group3::Neg, Reg::Rax);
            self.asm.store(Size::Q, sx, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.asm.alu_imm(Alu::Add, Reg::Rsp, 32);
    }
}
