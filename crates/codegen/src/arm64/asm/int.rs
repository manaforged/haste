use super::super::int::encode_bitmask_imm;
use super::{Alu, Asm, Bits, Cond, ExtOp, Reg, SP, Shift, Size, ZR, sized};

impl Asm {
    /// `op dst, a, b` on registers, where 31 names the zero register.
    pub fn alu(&mut self, op: Alu, size: Size, dst: Reg, a: Reg, b: Reg) {
        self.emit_gp(
            dst,
            sized(op as u32, size) | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0),
        );
    }

    pub fn alu_shifted(
        &mut self,
        op: Alu,
        size: Size,
        shift: Shift,
        imm: u8,
        dst: Reg,
        a: Reg,
        b: Reg,
    ) {
        self.emit_gp(
            dst,
            sized(op as u32, size)
                | (shift as u32) << 22
                | u32::from(b.0) << 16
                | u32::from(imm) << 10
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    pub fn alu_bitmask(
        &mut self,
        op: Alu,
        size: Size,
        (n, immr, imms): (u32, u32, u32),
        dst: Reg,
        a: Reg,
    ) {
        let base = match op {
            Alu::And => 0x9200_0000,
            Alu::Ands => 0xF200_0000,
            Alu::Orr => 0xB200_0000,
            Alu::Eor => 0xD200_0000,
            _ => return,
        };
        self.emit_gp(
            dst,
            sized(base, size)
                | n << 22
                | immr << 16
                | imms << 10
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    pub fn extr(&mut self, size: Size, dst: Reg, hi: Reg, lo: Reg, lsb: u8) {
        let base = if size == Size::X {
            0x93C0_0000
        } else {
            0x1380_0000
        };
        self.emit_gp(
            dst,
            base | u32::from(lo.0) << 16
                | u32::from(lsb) << 10
                | u32::from(hi.0) << 5
                | u32::from(dst.0),
        );
    }

    pub fn ror_imm(&mut self, size: Size, dst: Reg, src: Reg, imm: u8) {
        self.extr(size, dst, src, src, imm);
    }

    pub fn alu_extended(
        &mut self,
        alu: Alu,
        size: Size,
        ext: ExtOp,
        shift: u8,
        dst: Reg,
        a: Reg,
        b: Reg,
    ) {
        self.emit_gp(
            dst,
            sized((alu as u32) | (1 << 21), size)
                | u32::from(b.0) << 16
                | (ext as u32) << 13
                | u32::from(shift) << 10
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    /// `dst = a + b` where `a` or `dst` may be the stack pointer.
    pub fn add_extended(&mut self, sub: bool, dst: Reg, a: Reg, b: Reg) {
        self.alu_extended(
            if sub { Alu::Sub } else { Alu::Add },
            Size::X,
            ExtOp::Uxtx,
            0,
            dst,
            a,
            b,
        );
    }

    /// `dst = base + (index << shift)`.
    pub fn add_shifted(&mut self, dst: Reg, base: Reg, index: Reg, shift: u8) {
        self.emit_gp(
            dst,
            0x8B00_0000
                | u32::from(shift) << 10
                | u32::from(index.0) << 16
                | u32::from(base.0) << 5
                | u32::from(dst.0),
        );
    }

    /// `dst = a +/- imm` for `imm < 4096` or a multiple of 4096 below 2^24, where `a` or `dst` may be the stack pointer; sets the flags when `flags`.
    pub fn add_imm(&mut self, sub: bool, flags: bool, size: Size, dst: Reg, a: Reg, imm: u32) {
        if !flags
            && size == Size::X
            && dst == a
            && dst != SP
            && imm != 0
            && let Some(prev) = self.last_mem
            && prev.end_pos == self.pos()
            && prev.binds == self.binds
            && prev.disp == 0
            && prev.base == dst
            && (!prev.is_int || prev.reg != dst.0)
        {
            let delta = if sub { -i64::from(imm) } else { i64::from(imm) };
            if (-256..=255).contains(&delta) {
                self.code.truncate((prev.end_pos - 4) as usize);
                self.last_mem = None;
                self.emit_gp(
                    dst,
                    prev.base_op
                        | ((delta as u32) & 0x1FF) << 12
                        | 1 << 10
                        | u32::from(dst.0) << 5
                        | u32::from(prev.reg),
                );
                return;
            }
        }
        let (imm, shifted) = if imm >= 4096 {
            (imm >> 12, 1)
        } else {
            (imm, 0)
        };
        let base = 0x9100_0000 | u32::from(sub) << 30 | u32::from(flags) << 29 | shifted << 22;
        self.emit_gp(
            dst,
            sized(base, size) | (imm & 0xFFF) << 10 | u32::from(a.0) << 5 | u32::from(dst.0),
        );
    }

    /// `dst = a * b + c`, or `c - a * b` when `sub`, at `size`.
    pub fn madd(&mut self, sub: bool, size: Size, dst: Reg, (a, b): (Reg, Reg), c: Reg) {
        let base = if sub { 0x9B00_8000 } else { 0x9B00_0000 };
        self.emit_gp(
            dst,
            sized(base, size)
                | u32::from(b.0) << 16
                | u32::from(c.0) << 10
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    pub fn mov(&mut self, dst: Reg, src: Reg) {
        if dst != src {
            self.alu(Alu::Orr, Size::X, dst, ZR, src);
        }
    }

    /// `mov` that can read or write the stack pointer.
    pub fn mov_sp(&mut self, dst: Reg, src: Reg) {
        self.add_imm(false, false, Size::X, dst, src, 0);
    }

    fn half_counts(value: u64) -> (usize, usize) {
        let count = |target: u16| {
            (0..4)
                .filter(|&h| ((value >> (16 * h)) as u16) == target)
                .count()
        };
        (count(0xFFFF), count(0))
    }

    fn single_bitmask_imm(value: u64) -> Option<(Size, (u32, u32, u32))> {
        encode_bitmask_imm(value, true)
            .map(|b| (Size::X, b))
            .or_else(|| {
                ((value >> 32) == 0)
                    .then(|| encode_bitmask_imm(value, false).map(|b| (Size::W, b)))
                    .flatten()
            })
    }

    pub(in crate::arm64) fn mov_imm_is_single(imm: i64) -> bool {
        let value = imm as u64;
        let (ones, zeros) = Self::half_counts(value);
        ones >= 3 || zeros >= 3 || Self::single_bitmask_imm(value).is_some()
    }

    fn try_bitmask_half(value: u64, k: usize, cand: u16) -> Option<((u32, u32, u32), usize, u16)> {
        let (shift, mask) = (16 * k, 0xFFFF_u64 << (16 * k));
        let base = (value & !mask) | (u64::from(cand) << shift);
        let bitmask = encode_bitmask_imm(base, true)?;
        let hw = ((value >> shift) & 0xFFFF) as u16;
        (((base & !mask) | (u64::from(hw) << shift)) == value).then_some((bitmask, k, hw))
    }

    fn bitmask_movk_at(value: u64, h: [u16; 4], k: usize) -> Option<((u32, u32, u32), usize, u16)> {
        let fixed = [0, 0xFFFF, h[(k + 2) & 3], h[(k + 1) & 3]];
        if let Some(hit) = fixed
            .into_iter()
            .find_map(|c| Self::try_bitmask_half(value, k, c))
        {
            return Some(hit);
        }
        let (prev_top, next_bot) = (h[(k + 3) & 3] >> 15, h[(k + 1) & 3] & 1);
        (prev_top != next_bot)
            .then(|| {
                (1..16_u32).find_map(|w| {
                    let pat = (1_u16 << w) - 1;
                    Self::try_bitmask_half(value, k, if prev_top == 1 { pat } else { !pat })
                })
            })
            .flatten()
    }

    const MOVK: u32 = 0xF280_0000;

    fn try_emit_bitmask_imm(&mut self, dst: Reg, value: u64, ones: usize, zeros: usize) -> bool {
        if dst == ZR || ones >= 3 || zeros >= 3 {
            return false;
        }
        if let Some((size, bitmask)) = Self::single_bitmask_imm(value) {
            self.alu_bitmask(Alu::Orr, size, bitmask, dst, ZR);
            return true;
        }
        let h = [
            value as u16,
            (value >> 16) as u16,
            (value >> 32) as u16,
            (value >> 48) as u16,
        ];
        if ones < 2
            && zeros < 2
            && let Some((bitmask, k, hw)) = (0..4).find_map(|k| Self::bitmask_movk_at(value, h, k))
        {
            self.alu_bitmask(Alu::Orr, Size::X, bitmask, dst, ZR);
            self.word(Self::MOVK | ((k as u32) << 21) | (u32::from(hw) << 5) | u32::from(dst.0));
            return true;
        }
        false
    }

    pub fn mov_imm(&mut self, dst: Reg, imm: i64) {
        if dst != ZR && self.last_imm == Some((self.binds, dst, imm)) {
            return;
        }
        self.clobber_anchor(dst);
        let value = imm as u64;
        if value == 0 {
            self.word(0xD280_0000 | u32::from(dst.0));
        } else if value == u64::MAX {
            self.word(0x9280_0000 | u32::from(dst.0));
        } else {
            let (ones, zeros) = Self::half_counts(value);
            if !self.try_emit_bitmask_imm(dst, value, ones, zeros) {
                let (inverted, skip, mut first) =
                    (ones > zeros, if ones > zeros { 0xFFFF } else { 0 }, true);
                for half in 0..4 {
                    let bits = (value >> (16 * half)) as u16;
                    if bits == skip && !(first && half == 3) {
                        continue;
                    }
                    let shift = (half as u32) << 21;
                    let word = match (first, inverted) {
                        (true, true) => 0x9280_0000 | shift | u32::from(!bits) << 5,
                        (true, false) => 0xD280_0000 | shift | u32::from(bits) << 5,
                        (false, _) => Self::MOVK | shift | u32::from(bits) << 5,
                    };
                    self.word(word | u32::from(dst.0));
                    first = false;
                }
            }
        }
        if dst != ZR {
            self.last_imm = Some((self.binds, dst, imm));
        }
    }

    pub fn mov_symbol(&mut self, dst: Reg) -> u32 {
        self.clobber_anchor(dst);
        let at = self.pos();
        for base in [0xD280_0000, 0xF2A0_0000, 0xF2C0_0000, 0xF2E0_0000] {
            self.word(base | u32::from(dst.0));
        }
        at
    }

    pub fn adrp_add_reloc(&mut self, dst: Reg) -> u32 {
        self.clobber_anchor(dst);
        let at = self.pos();
        let rd = u32::from(dst.0);
        self.word(0x9000_0000 | rd);
        self.word(0x9100_0000 | (rd << 5) | rd);
        at
    }

    pub fn mov_fixed64(&mut self, dst: Reg, value: u64) {
        self.clobber_anchor(dst);
        let rd = u32::from(dst.0);
        for half in 0..4 {
            let bits = ((value >> (16 * half)) as u32) & 0xFFFF;
            let base = if half == 0 {
                0xD280_0000
            } else {
                Self::MOVK | ((half as u32) << 21)
            };
            self.word(base | (bits << 5) | rd);
        }
    }

    fn bitfield_word(&mut self, base: u32, dst: Reg, src: Reg, (immr, imms): (u8, u8)) {
        self.emit_gp(
            dst,
            base | u32::from(immr) << 16
                | u32::from(imms) << 10
                | u32::from(src.0) << 5
                | u32::from(dst.0),
        );
    }

    /// `ubfm`/`sbfm` with `immr` and `imms`: shifts and extensions by an immediate.
    pub fn bitfield(&mut self, signed: bool, dst: Reg, src: Reg, pair: (u8, u8)) {
        self.bitfield_word(
            if signed { 0x9340_0000 } else { 0xD340_0000 },
            dst,
            src,
            pair,
        );
    }

    pub fn lsl_imm(&mut self, dst: Reg, src: Reg, shift: u8) {
        self.bitfield(false, dst, src, ((64 - shift) % 64, 63 - shift));
    }

    pub fn shift32_imm(&mut self, left: bool, dst: Reg, src: Reg, shift: u8) {
        self.bitfield_word(
            0x5300_0000,
            dst,
            src,
            if left {
                ((32 - shift) % 32, 31 - shift)
            } else {
                (shift, 31)
            },
        );
    }

    pub fn lsr_imm(&mut self, dst: Reg, src: Reg, shift: u8) {
        self.bitfield(false, dst, src, (shift, 63));
    }

    pub fn asr_imm(&mut self, dst: Reg, src: Reg, shift: u8) {
        self.bitfield(true, dst, src, (shift, 63));
    }

    /// Extends the low `bytes` of `src` to 64 bits into `dst`.
    pub fn extend(&mut self, signed: bool, bytes: u32, dst: Reg, src: Reg) {
        match bytes {
            8.. => self.mov(dst, src),
            4 if !signed => {
                self.emit_gp(dst, 0x2A00_03E0 | u32::from(src.0) << 16 | u32::from(dst.0));
            }
            _ => self.bitfield(signed, dst, src, (0, (bytes * 8 - 1) as u8)),
        }
    }

    fn cond_select(&mut self, base: u32, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.emit_gp(
            dst,
            sized(base, size)
                | u32::from(b.0) << 16
                | (cond as u32) << 12
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    pub(crate) fn ccmp(
        &mut self,
        neg: bool,
        size: Size,
        imm: bool,
        (cond, rn, op2, nzcv): (Cond, Reg, u8, u8),
    ) {
        self.word(
            sized(if neg { 0xBA40_0000 } else { 0xFA40_0000 }, size)
                | u32::from(imm) << 11
                | u32::from(op2 & 0x1F) << 16
                | (cond as u32) << 12
                | u32::from(rn.0) << 5
                | u32::from(nzcv & 0xF),
        );
    }

    pub fn csel(&mut self, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.cond_select(0x9A80_0000, Size::X, cond, dst, a, b);
    }

    pub fn csinc(&mut self, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.cond_select(0x9A80_0400, size, cond, dst, a, b);
    }

    pub fn csinv(&mut self, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.cond_select(0xDA80_0000, size, cond, dst, a, b);
    }

    pub fn csneg(&mut self, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.cond_select(0xDA80_0400, size, cond, dst, a, b);
    }

    pub fn cset(&mut self, cond: Cond, dst: Reg) {
        self.emit_gp(
            dst,
            0x9A9F_07E0 | (cond.invert() as u32) << 12 | u32::from(dst.0),
        );
    }

    pub fn bits(&mut self, op: Bits, dst: Reg, src: Reg) {
        self.emit_gp(dst, op as u32 | u32::from(src.0) << 5 | u32::from(dst.0));
    }
}
