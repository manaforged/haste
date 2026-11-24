use crate::ir::{Lane, ShuffleMask, Type, Value};
use crate::x64::Emit;
use crate::x64::asm::{Alu, Cond, Mem, Reg, Size, Vl, Xmm};
use crate::x64::float::{X0, X1, X2};
use crate::x64::ymm::broadcast;

fn pblendw_mask(bidx: &[u8; 16]) -> Option<u8> {
    let mut imm = 0_u8;
    for w in 0..8_u8 {
        let (lo, hi) = (bidx[usize::from(2 * w)], bidx[usize::from(2 * w + 1)]);
        if lo == 16 + 2 * w && hi == 17 + 2 * w {
            imm |= 1 << w;
        } else if lo != 2 * w || hi != 2 * w + 1 {
            return None;
        }
    }
    Some(imm)
}

impl Emit<'_> {
    fn load_bytes16(&mut self, dst: Xmm, scratch: Xmm, bytes: [u8; 16]) {
        let lo = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as i64;
        let hi = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]) as i64;
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

    pub(super) fn vector_shuffle(
        &mut self,
        a: Value,
        b: Value,
        mask: &ShuffleMask,
        result: Value,
    ) -> Result<(), String> {
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

    fn vector_shuffle_const(
        &mut self,
        lane: Lane,
        count: u8,
        a: Value,
        b: Value,
        indices: &[u8; 16],
        dst: Xmm,
    ) {
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
            if self.try_shuffle_single_source(lane, count, src_val, off, indices, dst) {
                return;
            }
        }
        if lane.bytes() == 8 && count == 2 && matches!((indices[0], indices[1]), (0, 2) | (1, 3)) {
            self.vex_packed_regs(
                &[0x66],
                &[0x0F, if indices[0] == 0 { 0x6C } else { 0x6D }],
                (a, b),
                dst,
                false,
            );
            return;
        }
        let bytes = lane.bytes() as usize;
        if bytes * c == 16 && self.try_shuffle_two_vec_16(bytes, count, (a, b), indices, dst) {
            return;
        }
        if self.features.ssse3 {
            self.vector_shuffle_const_ssse3(bytes, count, (a, b), indices, dst);
            return;
        }
        self.vector_shuffle_const_stack(lane, count, (a, b), indices, dst);
    }

    fn try_shuffle_single_source(
        &mut self,
        lane: Lane,
        count: u8,
        src_val: Value,
        off: u8,
        indices: &[u8; 16],
        dst: Xmm,
    ) -> bool {
        let c = usize::from(count);
        if self.features.avx2 && indices[..c].iter().all(|&i| i == off) {
            let src = self.fin(src_val, X0);
            self.asm.vex(Vl::X, broadcast(lane), (dst, Xmm(0)), src);
            return true;
        }
        if (0..c).all(|l| usize::from(indices[l] - off) == l) {
            let src = self.fin(src_val, dst);
            if src != dst {
                self.asm.movaps(dst, src);
            }
            return true;
        }
        let imm = match (lane.bytes(), count) {
            (4, 4) | (2, 4) => Some(
                ((indices[0] - off) & 3)
                    | (((indices[1] - off) & 3) << 2)
                    | (((indices[2] - off) & 3) << 4)
                    | (((indices[3] - off) & 3) << 6),
            ),
            (8, 2) => {
                let (l0, l1) = ((indices[0] - off) * 2, (indices[1] - off) * 2);
                Some(l0 | ((l0 + 1) << 2) | (l1 << 4) | ((l1 + 1) << 6))
            }
            _ => None,
        };
        if let Some(imm) = imm {
            let (pp_byte, pp_vex) = if lane.bytes() == 2 {
                (0xF2, 0b11)
            } else {
                (0x66, 0b01)
            };
            let src = self.fin(src_val, X0);
            if self.features.avx {
                self.asm
                    .vex_xmm_imm(0b00001, pp_vex, 0x70, (dst, Xmm(0)), src, imm);
            } else {
                self.asm.sse_imm(&[pp_byte], &[0x0F, 0x70], dst.0, src, imm);
            }
            return true;
        }
        false
    }

    fn try_shuffle_two_vec_16(
        &mut self,
        bytes: usize,
        count: u8,
        (a, b): (Value, Value),
        indices: &[u8; 16],
        dst: Xmm,
    ) -> bool {
        let c = usize::from(count);
        let mut bidx = [0x80_u8; 16];
        for l in 0..c {
            let idx = indices[l];
            if idx < 2 * count {
                let base = if idx < count {
                    idx * bytes as u8
                } else {
                    16 + (idx - count) * bytes as u8
                };
                for k in 0..bytes {
                    bidx[l * bytes + k] = base + k as u8;
                }
            }
        }
        if self.features.ssse3 {
            let s = bidx[0];
            if (1..16).contains(&s) && (0..16_u8).all(|k| bidx[k as usize] == s + k) {
                self.emit_two_vec_imm(0x0F, dst, b, a, s);
                return true;
            }
        }
        if self.features.sse41
            && let Some(imm) = pblendw_mask(&bidx)
        {
            self.emit_two_vec_imm(0x0E, dst, a, b, imm);
            return true;
        }
        false
    }

    fn vector_shuffle_const_ssse3(
        &mut self,
        bytes: usize,
        count: u8,
        (a, b): (Value, Value),
        indices: &[u8; 16],
        dst: Xmm,
    ) {
        let c = usize::from(count);
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
    }

    fn vector_shuffle_const_stack(
        &mut self,
        lane: Lane,
        count: u8,
        (a, b): (Value, Value),
        indices: &[u8; 16],
        dst: Xmm,
    ) {
        let c = usize::from(count);
        let bytes = lane.bytes() as i32;
        let size = Size::of(lane.bytes());
        self.reserve_scratch(48);
        self.fget(X0, a);
        self.asm
            .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
        self.fget(X0, b);
        self.asm
            .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 16));
        for l in 0..c {
            let idx = indices[l];
            if idx < count {
                self.asm
                    .load(size, Reg::Rax, Mem::new(Reg::Rsp, i32::from(idx) * bytes));
            } else if idx < 2 * count {
                self.asm.load(
                    size,
                    Reg::Rax,
                    Mem::new(Reg::Rsp, 16 + i32::from(idx - count) * bytes),
                );
            } else {
                self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
            }
            self.asm
                .store(size, Mem::new(Reg::Rsp, 32 + l as i32 * bytes), Reg::Rax);
        }
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 32));
        self.release_scratch(48);
    }

    fn emit_two_vec_imm(&mut self, op: u8, dst: Xmm, stay: Value, src_val: Value, imm: u8) {
        if self.features.avx {
            let [s_stay] = self.fins(dst, false, [stay], &[src_val]);
            let src = self.foperand(src_val, X0);
            self.asm
                .vex_xmm_imm(0b00011, 0b01, op, (dst, s_stay), src, imm);
            self.clobber_xmms(&[X0, dst]);
            return;
        }
        let [_, src] = self.fins(dst, true, [stay, src_val], &[]);
        self.asm
            .sse_imm(&[0x66], &[0x0F, 0x3A, op], dst.0, src, imm);
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
            self.asm
                .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
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
            self.asm
                .sse(&[0xF3], &[0x0F, 0x6F], X0.0, Mem::new(Reg::Rsp, 0));
            self.release_scratch(16);
            self.asm.sse(&[0x66], &[0x0F, 0xEB], X0.0, X1);
            self.asm.movaps(dst, X0);
        } else {
            self.asm.sse(&[0x66], &[0x0F, 0xEB], dst.0, X1);
        }
        self.clobber_xmms(&[X0, X1, X2]);
    }

    fn vector_shuffle_dyn_baseline(
        &mut self,
        lane: Lane,
        count: u8,
        a: Value,
        b: Value,
        mask_v: Value,
        dst: Xmm,
    ) {
        self.reserve_scratch(64);
        self.fget(X0, a);
        self.asm
            .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 0));
        self.fget(X0, b);
        self.asm
            .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 16));
        self.fget(X0, mask_v);
        self.asm
            .sse(&[0xF3], &[0x0F, 0x7F], X0.0, Mem::new(Reg::Rsp, 32));
        self.asm.alu32(Alu::Xor, Reg::Rcx, Reg::Rcx);
        let (head, from_b, zero, store) = (
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
            self.asm.label(),
        );
        self.asm.bind(head);
        self.asm
            .load(Size::B, Reg::Rax, Mem::sib(Reg::Rsp, Reg::Rcx, 0, 32));
        self.asm.alu_imm32(Alu::Cmp, Reg::Rax, i32::from(count));
        self.asm.jcc(Cond::Ae, from_b);
        self.asm
            .load(Size::B, Reg::Rax, Mem::sib(Reg::Rsp, Reg::Rax, 0, 0));
        self.asm.jmp(store);
        self.asm.bind(from_b);
        self.asm.alu_imm32(Alu::Cmp, Reg::Rax, i32::from(count) * 2);
        self.asm.jcc(Cond::Ae, zero);
        self.asm.load(
            Size::B,
            Reg::Rax,
            Mem::sib(Reg::Rsp, Reg::Rax, 0, 16 - i32::from(count)),
        );
        self.asm.jmp(store);
        self.asm.bind(zero);
        self.asm.alu32(Alu::Xor, Reg::Rax, Reg::Rax);
        self.asm.bind(store);
        self.asm
            .store(Size::B, Mem::sib(Reg::Rsp, Reg::Rcx, 0, 48), Reg::Rax);
        self.asm.alu_imm(Alu::Add, Reg::Rcx, 1);
        self.asm.alu_imm32(Alu::Cmp, Reg::Rcx, i32::from(count));
        self.asm.jcc(Cond::B, head);
        self.xmm_load(Type::Vec { lane, count }, dst, Mem::new(Reg::Rsp, 48));
        self.release_scratch(64);
    }
}
