use super::asm::{Alu, Cond, Group3, Mem, Reg, Shift, Size, Vl, Xmm};
use super::float::{X0, X1, X2};
use super::int::cond;
use super::ymm::broadcast;
use super::Emit;
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Binary, IntCC, Lane, Op, ShuffleMask, TargetOpId, Type, Value};

#[derive(Clone, Copy)]
enum TargetOpForm {
    Unary,
    Binary,
    BinaryImm(u8),
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
    TargetOpSpec { id: TargetOpId::X64Psadbw, form: TargetOpForm::Binary, prefix: &[0x66], opcode: &[0x0F, 0xF6] },
    TargetOpSpec { id: TargetOpId::X64ClmulLowLow, form: TargetOpForm::BinaryImm(0x00), prefix: &[0x66], opcode: &[0x0F, 0x3A, 0x44] },
    TargetOpSpec { id: TargetOpId::X64ClmulHighLow, form: TargetOpForm::BinaryImm(0x01), prefix: &[0x66], opcode: &[0x0F, 0x3A, 0x44] },
    TargetOpSpec { id: TargetOpId::X64ClmulLowHigh, form: TargetOpForm::BinaryImm(0x10), prefix: &[0x66], opcode: &[0x0F, 0x3A, 0x44] },
    TargetOpSpec { id: TargetOpId::X64ClmulHighHigh, form: TargetOpForm::BinaryImm(0x11), prefix: &[0x66], opcode: &[0x0F, 0x3A, 0x44] },
];

impl Emit<'_> {
    pub(super) fn vector_op(&mut self, op: &Op, result: Value) -> Result<(), String> {
        if let Some(done) = self.wide_op(op, result) {
            return done;
        }
        match op {
            Op::Shuffle(a, ..) if self.splat_insert(*a, result) => Ok(()),
            Op::Shuffle(a, b, mask) => self.vector_shuffle(*a, *b, mask, result),
            Op::Extract(v, lane) => self.vector_extract(*v, *lane, result),
            Op::Insert(v, x, lane) => self.vector_insert(*v, *x, *lane, result),
            Op::Reduce(rop, v) => self.vector_reduce(*rop, *v, result),
            Op::TargetOp(id, args) => self.vector_target_op(*id, args.as_slice(), result),
            other => Err(format!("unexpected op {other:?} in vector_op")),
        }
    }

    pub(super) fn emit_mov_xmm_gpr(&mut self, dst: Xmm, reg: Reg, wide: bool) {
        if self.features.avx && wide {
            let r = reg as u8;
            self.asm.bytes(&[0xC4, ((!dst.0 >> 3 & 1) << 7) | 0x41 | ((!r >> 3 & 1) << 5), 0xF9, 0x6E, 0xC0 | ((dst.0 & 7) << 3) | (r & 7)]);
        } else if self.features.avx {
            self.asm.vex_xmm(0b00001, 0b01, 0x6E, (dst, Xmm(0)), reg);
        } else if wide {
            self.asm.movq_in(dst, reg);
        } else {
            self.asm.movd_in(dst, reg);
        }
    }

    fn emit_vpinsrq(&mut self, dst: Xmm, src: Xmm, reg: Reg, idx: u8) {
        let r = reg as u8;
        self.asm.bytes(&[0xC4, ((!dst.0 >> 3 & 1) << 7) | 0x43 | ((!r >> 3 & 1) << 5), 0x81 | ((!src.0 & 0xF) << 3), 0x22, 0xC0 | ((dst.0 & 7) << 3) | (r & 7), idx]);
    }

    /// `vpinsr{b,w,d,q}`: `src` with integer lane `idx` replaced by `reg`, into `dst`.
    pub(super) fn vpinsr(&mut self, lane: Lane, (dst, src): (Xmm, Xmm), reg: Reg, idx: u8) {
        match lane {
            Lane::I8 => self.asm.vex_xmm_imm(0b00011, 0b01, 0x20, (dst, src), reg, idx),
            Lane::I16 => self.asm.vex_xmm_imm(0b00001, 0b01, 0xC4, (dst, src), reg, idx),
            Lane::I32 => self.asm.vex_xmm_imm(0b00011, 0b01, 0x22, (dst, src), reg, idx),
            _ => self.emit_vpinsrq(dst, src, reg, idx),
        }
    }

    fn load_bytes16(&mut self, dst: Xmm, scratch: Xmm, bytes: [u8; 16]) {
        let lo = u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]) as i64;
        let hi = u64::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]]) as i64;
        if lo == 0 && hi == 0 {
            self.zero_xmm(dst);
            return;
        }
        self.asm.mov_imm(Reg::Rax, lo);
        self.emit_mov_xmm_gpr(dst, Reg::Rax, true);
        if hi == lo {
            self.sse_or_vex(&[0x66], &[0x0F, 0x6C], (dst, dst, dst));
        } else if hi != 0 {
            self.asm.mov_imm(Reg::Rax, hi);
            self.emit_mov_xmm_gpr(scratch, Reg::Rax, true);
            self.sse_or_vex(&[0x66], &[0x0F, 0x6C], (dst, dst, scratch));
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
        let mut norm = *indices;
        if a == b {
            for i in &mut norm[..c] {
                if *i >= count && *i < 2 * count {
                    *i -= count;
                }
            }
        }
        let indices = &norm;
        let all_a = indices[..c].iter().all(|&i| i < count);
        let all_b = indices[..c].iter().all(|&i| i >= count && i < 2 * count);
        if all_a || all_b {
            let src_val = if all_a { a } else { b };
            let off = if all_a { 0 } else { count };
            if self.features.avx2 && indices[..c].iter().all(|&i| i == off) {
                let src = self.fin(src_val, X0);
                self.asm.vex(Vl::X, broadcast(lane), (dst, Xmm(0)), src);
                return;
            }
            if (0..c).all(|l| usize::from(indices[l] - off) == l) {
                let src = self.fin(src_val, dst);
                if src != dst {
                    self.asm.movaps(dst, src);
                }
                return;
            }
            let imm = match (lane.bytes(), count) {
                (4, 4) => Some(((indices[0] - off) & 3) | (((indices[1] - off) & 3) << 2) | (((indices[2] - off) & 3) << 4) | (((indices[3] - off) & 3) << 6)),
                (8, 2) => {
                    let (l0, l1) = ((indices[0] - off) * 2, (indices[1] - off) * 2);
                    Some(l0 | ((l0 + 1) << 2) | (l1 << 4) | ((l1 + 1) << 6))
                }
                (2, 4) => Some(((indices[0] - off) & 3) | (((indices[1] - off) & 3) << 2) | (((indices[2] - off) & 3) << 4) | (((indices[3] - off) & 3) << 6)),
                _ => None,
            };
            if let Some(imm) = imm {
                let (pp_byte, pp_vex) = if lane.bytes() == 2 { (0xF2, 0b11) } else { (0x66, 0b01) };
                let src = self.fin(src_val, X0);
                if self.features.avx {
                    self.asm.vex_xmm_imm(0b00001, pp_vex, 0x70, (dst, Xmm(0)), src, imm);
                } else {
                    self.asm.sse_imm(&[pp_byte], &[0x0F, 0x70], dst.0, src, imm);
                }
                return;
            }
        }
        if lane.bytes() == 8 && count == 2 && matches!((indices[0], indices[1]), (0, 2) | (1, 3)) {
            self.vex_packed_regs(&[0x66], &[0x0F, if indices[0] == 0 { 0x6C } else { 0x6D }], (a, b), dst, false);
            return;
        }
        let bytes = lane.bytes() as usize;
        if bytes * c == 16 {
            let mut bidx = [0x80_u8; 16];
            for l in 0..c {
                let idx = indices[l];
                if idx < 2 * count {
                    let base = if idx < count { idx * bytes as u8 } else { 16 + (idx - count) * bytes as u8 };
                    for k in 0..bytes {
                        bidx[l * bytes + k] = base + k as u8;
                    }
                }
            }
            if self.features.ssse3 {
                let s = bidx[0];
                if (1..16).contains(&s) && (0..16_u8).all(|k| bidx[k as usize] == s + k) {
                    self.emit_two_vec_imm(0x0F, dst, b, a, s);
                    return;
                }
            }
            if self.features.sse41 {
                let mut imm = 0_u8;
                let mut ok = true;
                for w in 0..8_u8 {
                    let (lo, hi) = (bidx[usize::from(2 * w)], bidx[usize::from(2 * w + 1)]);
                    if lo == 16 + 2 * w && hi == 17 + 2 * w {
                        imm |= 1 << w;
                    } else if lo != 2 * w || hi != 2 * w + 1 {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    self.emit_two_vec_imm(0x0E, dst, a, b, imm);
                    return;
                }
            }
        }
        if self.features.ssse3 {
            let mut mask_a = [0x80_u8; 16];
            let mut mask_b = [0x80_u8; 16];
            let (mut uses_a, mut uses_b) = (false, false);
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
            if uses_a ^ uses_b {
                let (src_val, mask) = if uses_a { (a, mask_a) } else { (b, mask_b) };
                self.load_bytes16(X1, X0, mask);
                let src = self.fin(src_val, dst);
                self.sse_or_vex(&[0x66], &[0x0F, 0x38, 0x00], (dst, src, X1));
                self.clobber_xmms(&[X0, X1, dst]);
                return;
            }
            self.load_bytes16(X1, X2, mask_a);
            let sa = self.fin(a, X0);
            self.sse_or_vex(&[0x66], &[0x0F, 0x38, 0x00], (X0, sa, X1));
            self.clobber_xmms(&[X0, X1, X2]);
            self.load_bytes16(X1, X2, mask_b);
            if self.features.avx {
                let sb = self.fin(b, X2);
                self.asm.vex_xmm(0b00010, 0b01, 0x00, (X2, sb), X1);
                self.asm.vex_xmm(0b00001, 0b01, 0xEB, (dst, X0), X2);
            } else {
                self.fget(X2, b);
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X2.0, X1);
                self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X2);
                if dst != X0 {
                    self.asm.movaps(dst, X0);
                }
            }
            self.clobber_xmms(&[X0, X1, X2, dst]);
            return;
        }
        let bytes = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        self.reserve_scratch(48);
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
        self.release_scratch(48);
    }

    fn emit_two_vec_imm(&mut self, op: u8, dst: Xmm, stay: Value, src_val: Value, imm: u8) {
        if self.features.avx {
            let [s_stay] = self.fins(dst, false, [stay], &[src_val]);
            let src = self.foperand(src_val, X0);
            self.asm.vex_xmm_imm(0b00011, 0b01, op, (dst, s_stay), src, imm);
            self.clobber_xmms(&[X0, dst]);
            return;
        }
        let [_, src] = self.fins(dst, true, [stay, src_val], &[]);
        self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, op], dst.0, src, imm);
    }

    fn vector_shuffle_dyn_ssse3(&mut self, count: u8, a: Value, b: Value, mask_v: Value, dst: Xmm) {
        if a == b && count == 16 {
            if self.features.avx {
                let [src_a] = self.fins(dst, false, [a], &[mask_v]);
                let src_m = self.foperand(mask_v, X0);
                self.asm.vex_xmm(0b00010, 0b01, 0x00, (dst, src_a), src_m);
                self.clobber_xmm(X0);
            } else {
                self.packed_regs(&[0x66], &[0x0F, 0x38, 0x00], (a, mask_v), dst);
            }
            return;
        }
        if a == b && count == 8 {
            self.fget(X0, mask_v);
            self.splat_imm(Lane::I8, X1, 15);
            self.asm.sse(&[0x66], &[0x0F, 0x64], X0.0, X1);
            self.clobber_xmm(X0);
            self.splat_imm(Lane::I8, X1, -121);
            let m = self.fin(mask_v, X2);
            self.asm.sse(&[0x66], &[0x0F, 0xDB], X1.0, m);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X1);
            if self.features.avx {
                let sa = self.fin(a, dst);
                self.asm.vex_xmm(0b00010, 0b01, 0x00, (dst, sa), X0);
            } else {
                self.fget(dst, a);
                self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], dst.0, X0);
            }
            self.clobber_xmms(&[X0, X1, X2]);
            return;
        }
        if self.features.avx && dst != X2 && !self.lives(dst, &[mask_v, b]) {
            self.splat_imm(Lane::I8, X1, i64::from(count - 1));
            let m = self.fin(mask_v, X0);
            self.asm.vex_xmm(0b00001, 0b01, 0x64, (X2, m), X1);
            self.asm.vex_xmm(0b00001, 0b01, 0xEB, (X2, X2), m);
            let sa = self.fin(a, dst);
            self.asm.vex_xmm(0b00010, 0b01, 0x00, (dst, sa), X2);
            self.splat_imm(Lane::I8, X2, i64::from(count));
            self.asm.vex_xmm(0b00001, 0b01, 0xF8, (X2, m), X2);
            self.asm.vex_xmm(0b00001, 0b01, 0x64, (X0, X2), X1);
            self.clobber_xmm(X0);
            self.asm.vex_xmm(0b00001, 0b01, 0xEB, (X2, X2), X0);
            let sb = self.fin(b, X0);
            self.asm.vex_xmm(0b00010, 0b01, 0x00, (X0, sb), X2);
            self.asm.vex_xmm(0b00001, 0b01, 0xEB, (dst, dst), X0);
            self.clobber_xmms(&[X0, X1, X2, dst]);
            return;
        }
        self.fget(X0, mask_v);
        self.splat_imm(Lane::I8, X1, i64::from(count - 1));
        self.asm.movaps(X2, X0);
        self.asm.sse(&[0x66], &[0x0F, 0x64], X2.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], X2.0, X0);
        self.fget(X0, a);
        self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X0.0, X2);
        self.clobber_xmm(X0);
        let save_stack = dst == X2 || self.lives(dst, &[mask_v, b]);
        if save_stack {
            self.reserve_scratch(16);
            self.asm.sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
        } else {
            self.asm.movaps(dst, X0);
        }
        self.fget(X0, mask_v);
        self.splat_imm(Lane::I8, X1, i64::from(count));
        self.asm.sse(&[0x66], &[0x0F, 0xF8], X0.0, X1);
        self.clobber_xmm(X0);
        self.splat_imm(Lane::I8, X1, i64::from(count - 1));
        self.asm.movaps(X2, X0);
        self.asm.sse(&[0x66], &[0x0F, 0x64], X2.0, X1);
        self.asm.sse(&[0x66], &[0x0F, 0xEB], X2.0, X0);
        self.fget(X1, b);
        self.asm.sse(&[0x66], &[0x0F, 0x38, 0x00], X1.0, X2);
        if save_stack {
            self.asm.sse(&[0xF3], &[0x0F, 0x6F], X0.0, Mem::new(Reg::Rsp, 0));
            self.release_scratch(16);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X1);
            self.asm.movaps(dst, X0);
        } else {
            self.asm.sse(&[0x66], &[0x0F, 0xEB], dst.0, X1);
        }
        self.clobber_xmms(&[X0, X1, X2]);
    }

    fn vector_shuffle_dyn_baseline(&mut self, lane: Lane, count: u8, a: Value, b: Value, mask_v: Value, dst: Xmm) {
        self.reserve_scratch(64);
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
        self.release_scratch(64);
    }

    fn vector_extract(&mut self, v: Value, idx: u8, result: Value) -> Result<(), String> {
        let Some((lane, _)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in extract", self.ty(v)));
        };
        if lane.is_float() {
            let dst = self.fdst(result);
            let src = self.fin(v, if idx == 0 { dst } else { X0 });
            match (lane, idx) {
                (_, 0) => if dst != src { self.asm.movaps(dst, src); },
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
            (Lane::I8, _) if self.features.sse41 => self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x14], src.0, dst, idx),
            (Lane::I8, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0xC5], dst as u8, src, idx / 2);
                if idx & 1 != 0 { self.asm.shift_imm(Size::D, Shift::Shr, dst, 8); }
                self.asm.load(Size::B, dst, dst);
            }
            (Lane::I32, _) if self.features.sse41 => self.asm.sse_imm(&[0x66], &[0x0F, 0x3A, 0x16], src.0, dst, idx),
            (Lane::I32, _) => {
                self.asm.sse_imm(&[0x66], &[0x0F, 0x70], X1.0, src, idx);
                self.asm.movd_out(dst, X1);
            }
            (Lane::I64, _) if self.features.sse41 => self.asm.sse_wide_imm(&[0x66], &[0x0F, 0x3A, 0x16], src.0, dst, idx),
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
            let [_, s] = self.fins(dst, true, [v, x], &[]);
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
        if matches!(lane, Lane::I8 | Lane::I16 | Lane::I32 | Lane::I64) && idx == 0 && self.is_zero_vec(v) {
            if self.const_of(x) == Some(0) {
                self.zero_xmm(dst);
            } else {
                let reg = match self.reg_of(x) {
                    Some(r) if self.ty(x).bytes() >= 4 => r,
                    _ => {
                        self.get(Reg::Rax, x, Ext::Zero);
                        Reg::Rax
                    }
                };
                self.emit_mov_xmm_gpr(dst, reg, lane == Lane::I64);
            }
            self.fput(result, dst);
            return Ok(());
        }
        if self.features.avx && (lane == Lane::I16 || self.features.sse41) {
            let reg = self.reg_of(x).unwrap_or_else(|| {
                self.get(Reg::Rax, x, Ext::Zero);
                Reg::Rax
            });
            let sv = self.fin(v, dst);
            self.vpinsr(lane, (dst, sv), reg, idx);
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
        self.reserve_scratch(16);
        self.asm.sse(&[0xF3], &[0x0F, 0x7F], dst.0, Mem::new(Reg::Rsp, 0));
        self.asm.store(Size::of(lane.bytes()), Mem::new(Reg::Rsp, i32::from(idx) * lane.bytes() as i32), Reg::Rax);
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(16);
    }

    fn cached_fspill(&self, value: Value) -> Option<Xmm> {
        if !matches!(self.location(value), Location::Slot) {
            return None;
        }
        let disp = self.frame.values[value.index()];
        let ty = self.ty(value);
        self.spills.cached_fspill(disp, ty)
    }

    fn emit_target_binary(&mut self, spec: &TargetOpSpec, (a, b, k): (Value, Value, Option<Value>), imm: Option<u8>, result: Value) {
        let dst = if k.is_some() { self.xmm_of(result).unwrap_or(X2) } else { self.fdst(result) };
        let mut got_k = k.is_none();
        let mut got_a = false;
        if let Some(kv) = k && self.cached_fspill(kv).is_some() {
            self.fget(X0, kv);
            got_k = true;
        } else if self.cached_fspill(a).is_some() && (self.xmm_of(b) != Some(dst) || a == b) {
            self.fget(dst, a);
            got_a = true;
        }
        let mut src_b = match self.cached_fspill(b) {
            Some(cached) if cached != dst && (k.is_none() || cached != X0) => cached,
            _ => self.fin(b, X1),
        };
        if src_b == dst && a != b {
            self.asm.movaps(X1, src_b);
            self.clobber_xmm(X1);
            src_b = X1;
        }
        if let Some(kv) = k && !got_k { self.fget(X0, kv); }
        if !got_a { self.fget(dst, a); }
        match imm {
            Some(imm) => self.asm.sse_imm(spec.prefix, spec.opcode, dst.0, src_b, imm),
            None => self.asm.sse(spec.prefix, spec.opcode, dst.0, src_b),
        }
        self.clobber_xmm(dst);
        self.fput(result, dst);
    }

    fn vector_target_op(&mut self, id: TargetOpId, args: &[Value], result: Value) -> Result<(), String> {
        let spec = X64_TARGET_OPS.iter().find(|s| s.id == id).ok_or_else(|| format!("unsupported target op {id:?}"))?;
        match spec.form {
            TargetOpForm::Unary => {
                let [x] = *args else { return Err(format!("{id:?} expects 1 arg")); };
                let dst = self.fdst(result);
                let src = self.fin(x, X0);
                self.asm.sse(spec.prefix, spec.opcode, dst.0, src);
                self.clobber_xmm(dst);
                self.fput(result, dst);
            }
            TargetOpForm::Binary | TargetOpForm::BinaryImm(_) => {
                let [x, y] = *args else { return Err(format!("{id:?} expects 2 args")); };
                let imm = if let TargetOpForm::BinaryImm(imm) = spec.form { Some(imm) } else { None };
                self.emit_target_binary(spec, (x, y, None), imm, result);
            }
            TargetOpForm::Sha256Rnds2 => {
                let [a, b, k] = *args else { return Err(format!("{id:?} expects 3 args")); };
                self.emit_target_binary(spec, (a, b, Some(k)), None, result);
            }
        }
        Ok(())
    }

    fn spill_vec_pair(&mut self, ty: Type, x: Value, y: Value) {
        self.reserve_scratch(32);
        self.fget(X0, x);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 0), X0);
        self.fget(X0, y);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 16), X0);
    }

    pub(super) fn scalarize_int_binary(&mut self, op: Binary, lane: Lane, count: u8, x: Value, y: Value, dst: Xmm) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        let signed_load = matches!(op, Binary::Smin | Binary::Smax | Binary::Sshr);
        self.spill_vec_pair(ty, x, y);
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
                Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat => self.emit_sat_op(op, size, Reg::Rax, Reg::Rcx),
                _ => {}
            }
            self.asm.store(size, sx, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(32);
    }

    pub(super) fn scalarize_int_clz(&mut self, lane: Lane, count: u8, x: Value, dst: Xmm) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        self.fget(X0, x);
        self.reserve_scratch(16);
        self.xmm_store(ty, Mem::new(Reg::Rsp, 0), X0);
        for i in 0..i32::from(count) {
            let slot = Mem::new(Reg::Rsp, i * stride);
            if self.features.lzcnt && lane.bits() >= 32 {
                self.asm.count(super::asm::Count::Lzcnt, size, Reg::Rax, slot);
            } else {
                self.asm.load(size, Reg::Rax, slot);
                self.count(super::asm::Count::Lzcnt, Reg::Rax, Reg::Rax);
                if lane.bits() < 64 { self.asm.alu_imm(Alu::Sub, Reg::Rax, (64 - lane.bits()) as i32); }
            }
            self.asm.store(size, slot, Reg::Rax);
        }
        self.xmm_load(ty, dst, Mem::new(Reg::Rsp, 0));
        self.release_scratch(16);
    }

    pub(super) fn scalarize_int_icmp(&mut self, cc: IntCC, lane: Lane, count: u8, x: Value, y: Value, dst: Xmm) {
        let ty = Type::Vec { lane, count };
        let stride = lane.bytes() as i32;
        self.spill_vec_pair(ty, x, y);
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
        self.release_scratch(32);
    }
}
