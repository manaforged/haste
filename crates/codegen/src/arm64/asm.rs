//! The A64 instruction encoder: one 32-bit word per instruction, with labels patched when they are bound.

mod atomic;
mod float;

use super::int::encode_bitmask_imm;

/// A general register `x0..x30`; 31 reads as the zero register or the stack pointer, by instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Reg(pub u8);

pub(crate) const SP: Reg = Reg(31);
pub(crate) const ZR: Reg = Reg(31);
pub(crate) const FP: Reg = Reg(29);
pub(crate) const LR: Reg = Reg(30);

/// A SIMD and floating-point register `v0..v31`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Vreg(pub u8);

/// A base register and a byte offset.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mem {
    pub base: Reg,
    pub disp: i32,
}

impl Mem {
    pub fn new(base: Reg, disp: i32) -> Self { Self { base, disp } }
    pub fn offset(self, by: i32) -> Self { Self { base: self.base, disp: self.disp + by } }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Shift { Lsl = 0, Lsr = 1, Asr = 2, Ror = 3 }

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExtOp { Uxtb = 0, Uxth = 1, Uxtw = 2, Uxtx = 3, Sxtb = 4, Sxth = 5, Sxtw = 6 }

pub(super) fn encode_sib(ext: ExtOp, shift: u8) -> u8 {
    let mode = match ext {
        ExtOp::Uxtw => 1,
        ExtOp::Sxtw => 2,
        _ => 0,
    };
    (mode << 3) | (shift & 0x7)
}

pub(super) fn decode_sib(packed: u8) -> (ExtOp, u8) {
    let ext = match packed >> 3 {
        1 => ExtOp::Uxtw,
        2 => ExtOp::Sxtw,
        _ => ExtOp::Uxtx,
    };
    (ext, packed & 0x7)
}

/// The width of a memory access or of an integer operation: 1, 2, 4 or 8 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Size { B, H, W, X }

impl Size {
    pub fn of(bytes: u32) -> Self {
        match bytes { 1 => Self::B, 2 => Self::H, 4 => Self::W, _ => Self::X }
    }

    fn log2(self) -> u32 { self as u32 }
}

/// A condition on the flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cond { Eq = 0, Ne = 1, Hs = 2, Lo = 3, Mi = 4, Pl = 5, Vs = 6, Vc = 7, Hi = 8, Ls = 9, Ge = 10, Lt = 11, Gt = 12, Le = 13 }

impl Cond {
    pub fn invert(self) -> Self {
        match self {
            Self::Eq => Self::Ne, Self::Ne => Self::Eq, Self::Hs => Self::Lo, Self::Lo => Self::Hs,
            Self::Mi => Self::Pl, Self::Pl => Self::Mi, Self::Vs => Self::Vc, Self::Vc => Self::Vs,
            Self::Hi => Self::Ls, Self::Ls => Self::Hi, Self::Ge => Self::Lt, Self::Lt => Self::Ge,
            Self::Gt => Self::Le, Self::Le => Self::Gt,
        }
    }
}

/// A logical or arithmetic operation on two registers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Alu {
    Add = 0x8B00_0000,
    Adds = 0xAB00_0000,
    Sub = 0xCB00_0000,
    Subs = 0xEB00_0000,
    And = 0x8A00_0000,
    Ands = 0xEA00_0000,
    Orr = 0xAA00_0000,
    Eor = 0xCA00_0000,
    Bic = 0x8A20_0000,
    Orn = 0xAA20_0000,
    Eon = 0xCA20_0000,
    Adc = 0x9A00_0000,
    Sbc = 0xDA00_0000,
    Sbcs = 0xFA00_0000,
    Lslv = 0x9AC0_2000,
    Lsrv = 0x9AC0_2400,
    Asrv = 0x9AC0_2800,
    Rorv = 0x9AC0_2C00,
    Udiv = 0x9AC0_0800,
    Sdiv = 0x9AC0_0C00,
    Smulh = 0x9B40_7C00,
    Umulh = 0x9BC0_7C00,
}

/// A one-register integer operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Bits {
    Clz = 0xDAC0_1000,
    Rbit = 0xDAC0_0000,
    Rev = 0xDAC0_0C00,
    Rev32 = 0x5AC0_0800,
    Rev16 = 0x5AC0_0400,
}

/// A floating-point operation; the encodings name the single-precision form, and double precision sets bit 22.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fop {
    Add = 0x1E20_2800,
    Sub = 0x1E20_3800,
    Mul = 0x1E20_0800,
    Div = 0x1E20_1800,
}

/// A one-register floating-point operation, single-precision encoding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Funary {
    Sqrt = 0x1E21_C000,
    Neg = 0x1E21_4000,
    Abs = 0x1E20_C000,
    Floor = 0x1E25_4000,
    Ceil = 0x1E24_C000,
    Trunc = 0x1E25_C000,
}

/// A conversion between an integer register and a floating-point register, single-precision and 32-bit encoding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Fconv {
    Scvtf = 0x1E22_0000,
    Ucvtf = 0x1E23_0000,
    Fcvtzs = 0x1E38_0000,
    Fcvtzu = 0x1E39_0000,
}

#[derive(Clone, Copy)]
pub(crate) struct Label(usize);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fixup {
    /// A 26-bit word offset: `b` and `bl`.
    Branch,
    /// A 19-bit word offset at bit 5: `b.cond`, `cbz`, `cbnz`.
    Cond,
    /// A 14-bit word offset at bit 5: `tbz`, `tbnz`.
    Tbz,
    /// A 21-bit byte offset: `adr`.
    Adr,
}

fn fixup_bits(kind: Fixup, bytes: i64) -> Option<u32> {
    let delta = bytes / 4;
    match kind {
        Fixup::Branch if (-(1 << 25)..1 << 25).contains(&delta) => Some((delta as u32) & 0x03FF_FFFF),
        Fixup::Cond if (-(1 << 18)..1 << 18).contains(&delta) => Some(((delta as u32) & 0x7_FFFF) << 5),
        Fixup::Tbz if (-(1 << 13)..1 << 13).contains(&delta) => Some(((delta as u32) & 0x3FFF) << 5),
        Fixup::Adr if (-(1 << 20)..1 << 20).contains(&bytes) => {
            let imm = bytes as u32;
            Some(((imm & 3) << 29) | (((imm >> 2) & 0x7_FFFF) << 5))
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PairKind { IntLoad(Size), IntStore(Size), FloatLoad(Size), FloatStore(Size) }

impl PairKind {
    fn step(self) -> i32 {
        if matches!(self, Self::IntLoad(Size::W) | Self::IntStore(Size::W) | Self::FloatLoad(Size::W) | Self::FloatStore(Size::W)) { 4 } else { 8 }
    }
}

#[derive(Clone, Copy)]
struct LastMem { end_pos: u32, binds: u32, pair: Option<PairKind>, base_op: u32, is_int: bool, reg: u8, base: Reg, disp: i32 }

#[derive(Default)]
pub(crate) struct Asm {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, usize, Fixup)>,
    table_fixups: Vec<(u32, usize, usize)>,
    binds: u32,
    last_mem: Option<LastMem>,
    last_anchor: Option<(u32, Reg, i64)>,
}

fn sized(base: u32, size: Size) -> u32 {
    if size == Size::X { base } else { base & !(1 << 31) }
}

impl Asm {
    pub fn pos(&self) -> u32 { self.code.len() as u32 }

    pub fn word(&mut self, word: u32) { self.code.extend(word.to_le_bytes()); }

    pub(super) fn clear_anchor(&mut self) { self.last_anchor = None; }

    pub(super) fn clobber_anchor(&mut self, reg: Reg) {
        if reg == Reg(17) || self.last_anchor.is_some_and(|(_, base, _)| base == reg) {
            self.last_anchor = None;
        }
    }

    pub(in crate::arm64) fn emit_gp(&mut self, dst: Reg, word: u32) {
        self.clobber_anchor(dst);
        self.word(word);
    }

    fn clear_word(&mut self, word: u32) {
        self.clear_anchor();
        self.word(word);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.pos());
        self.binds = self.binds.wrapping_add(1);
        self.clear_anchor();
    }

    fn jump(&mut self, word: u32, label: Label, kind: Fixup) {
        self.fixups.push((self.pos(), label.0, kind));
        self.word(word);
    }

    fn jump_cond(&mut self, kind: Fixup, word: u32, invert: u32, label: Label) {
        if let Some(target) = self.labels[label.0]
            && fixup_bits(kind, i64::from(target) - i64::from(self.pos())).is_none()
        {
            self.word(invert | (2 << 5));
            self.b(label);
            return;
        }
        self.jump(word, label, kind);
    }

    pub fn b(&mut self, label: Label) {
        self.clear_anchor();
        self.jump(0x1400_0000, label, Fixup::Branch);
    }

    pub fn bcond(&mut self, cond: Cond, label: Label) {
        self.jump_cond(Fixup::Cond, 0x5400_0000 | cond as u32, 0x5400_0000 | cond.invert() as u32, label);
    }

    /// Branches to `label` when `reg` is zero, or nonzero when `nonzero`.
    pub fn cbz(&mut self, nonzero: bool, size: Size, reg: Reg, label: Label) {
        let base = if nonzero { 0xB500_0000 } else { 0xB400_0000 };
        let word = sized(base, size) | u32::from(reg.0);
        self.jump_cond(Fixup::Cond, word, word ^ (1 << 24), label);
    }

    /// Branches to `label` when bit `bit` of `reg` is zero, or nonzero when `nonzero`.
    pub fn tbz(&mut self, nonzero: bool, reg: Reg, bit: u8, label: Label) {
        let base = if nonzero { 0x3700_0000 } else { 0x3600_0000 };
        let word = base | u32::from(bit >> 5) << 31 | u32::from(bit & 0x1F) << 19 | u32::from(reg.0);
        self.jump_cond(Fixup::Tbz, word, word ^ (1 << 24), label);
    }

    fn try_patch(&mut self, at: u32, kind: Fixup, target: u32) -> bool {
        let Some(bits) = fixup_bits(kind, i64::from(target) - i64::from(at)) else { return false };
        let at = at as usize;
        let mut word = u32::from_le_bytes([self.code[at], self.code[at + 1], self.code[at + 2], self.code[at + 3]]);
        word |= bits;
        self.code[at..at + 4].copy_from_slice(&word.to_le_bytes());
        true
    }

    fn emit_veneer(&mut self, veneers: &mut Vec<(usize, u32)>, label: usize) -> u32 {
        if let Some(&(_, pos)) = veneers.iter().find(|&&(l, _)| l == label) {
            return pos;
        }
        let pos = self.pos();
        self.b(Label(label));
        veneers.push((label, pos));
        pos
    }

    pub fn maybe_island(&mut self) -> Result<(), String> {
        let now = self.pos();
        let need_tbz = self.fixups.iter().any(|&(at, l, k)| k == Fixup::Tbz && self.labels[l].is_none() && now - at >= 16_384);
        let need_cond = self.fixups.iter().any(|&(at, l, k)| k == Fixup::Cond && self.labels[l].is_none() && now - at >= 524_288);
        if !need_tbz && !need_cond {
            return Ok(());
        }
        let over = self.label();
        self.b(over);
        let mut veneers = Vec::new();
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let flush = self.labels[label].is_none() && ((kind == Fixup::Tbz && (need_tbz || now - at >= 8_192)) || (kind == Fixup::Cond && need_cond));
            if !flush {
                self.fixups.push((at, label, kind));
            } else {
                let veneer = self.emit_veneer(&mut veneers, label);
                if !self.try_patch(at, kind, veneer) {
                    return Err("a branch out of range".to_owned());
                }
            }
        }
        self.bind(over);
        Ok(())
    }

    pub fn adr(&mut self, dst: Reg, label: Label) {
        self.clobber_anchor(dst);
        self.jump(0x1000_0000 | u32::from(dst.0), label, Fixup::Adr);
    }

    pub fn ldrsw_scaled(&mut self, dst: Reg, base: Reg, index: Reg) {
        self.emit_gp(dst, 0xB8A0_7800 | u32::from(index.0) << 16 | u32::from(base.0) << 5 | u32::from(dst.0));
    }

    pub fn table_entry(&mut self, base: Label, target: Label) {
        self.table_fixups.push((self.pos(), base.0, target.0));
        self.word(0);
    }

    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let target = self.labels[label].ok_or("a jump to a label that was never bound")?;
            if !self.try_patch(at, kind, target) {
                return Err("a branch out of range".to_owned());
            }
        }
        for (at, base, target) in std::mem::take(&mut self.table_fixups) {
            let base = self.labels[base].ok_or("a jump table without a base label")?;
            let target = self.labels[target].ok_or("a jump to a label that was never bound")?;
            let delta = i64::from(target) - i64::from(base);
            let at = at as usize;
            self.code[at..at + 4].copy_from_slice(&(delta as i32).to_le_bytes());
        }
        Ok(self.code)
    }

    /// `op dst, a, b` on registers, where 31 names the zero register.
    pub fn alu(&mut self, op: Alu, size: Size, dst: Reg, a: Reg, b: Reg) {
        self.emit_gp(dst, sized(op as u32, size) | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn alu_shifted(&mut self, op: Alu, size: Size, shift: Shift, imm: u8, dst: Reg, a: Reg, b: Reg) {
        self.emit_gp(dst, sized(op as u32, size) | (shift as u32) << 22 | u32::from(b.0) << 16 | u32::from(imm) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn alu_bitmask(&mut self, op: Alu, size: Size, (n, immr, imms): (u32, u32, u32), dst: Reg, a: Reg) {
        let base = match op {
            Alu::And => 0x9200_0000,
            Alu::Ands => 0xF200_0000,
            Alu::Orr => 0xB200_0000,
            Alu::Eor => 0xD200_0000,
            _ => return,
        };
        self.emit_gp(dst, sized(base, size) | n << 22 | immr << 16 | imms << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn extr(&mut self, size: Size, dst: Reg, hi: Reg, lo: Reg, lsb: u8) {
        let base = if size == Size::X { 0x93C0_0000 } else { 0x1380_0000 };
        self.emit_gp(dst, base | u32::from(lo.0) << 16 | u32::from(lsb) << 10 | u32::from(hi.0) << 5 | u32::from(dst.0));
    }

    pub fn ror_imm(&mut self, size: Size, dst: Reg, src: Reg, imm: u8) {
        self.extr(size, dst, src, src, imm);
    }

    pub fn alu_extended(&mut self, alu: Alu, size: Size, ext: ExtOp, shift: u8, dst: Reg, a: Reg, b: Reg) {
        self.emit_gp(dst, sized((alu as u32) | (1 << 21), size) | u32::from(b.0) << 16 | (ext as u32) << 13 | u32::from(shift) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    /// `dst = a + b` where `a` or `dst` may be the stack pointer.
    pub fn add_extended(&mut self, sub: bool, dst: Reg, a: Reg, b: Reg) {
        self.alu_extended(if sub { Alu::Sub } else { Alu::Add }, Size::X, ExtOp::Uxtx, 0, dst, a, b);
    }

    /// `dst = base + (index << shift)`.
    pub fn add_shifted(&mut self, dst: Reg, base: Reg, index: Reg, shift: u8) {
        self.emit_gp(dst, 0x8B00_0000 | u32::from(shift) << 10 | u32::from(index.0) << 16 | u32::from(base.0) << 5 | u32::from(dst.0));
    }

    /// `dst = a +/- imm` for `imm < 4096` or a multiple of 4096 below 2^24, where `a` or `dst` may be the stack pointer; sets the flags when `flags`.
    pub fn add_imm(&mut self, sub: bool, flags: bool, size: Size, dst: Reg, a: Reg, imm: u32) {
        if !flags && size == Size::X && dst == a && dst != SP && imm != 0
            && let Some(prev) = self.last_mem
            && prev.end_pos == self.pos() && prev.binds == self.binds && prev.disp == 0 && prev.base == dst
            && (!prev.is_int || prev.reg != dst.0)
        {
            let delta = if sub { -i64::from(imm) } else { i64::from(imm) };
            if (-256..=255).contains(&delta) {
                self.code.truncate((prev.end_pos - 4) as usize);
                self.last_mem = None;
                self.emit_gp(dst, prev.base_op | ((delta as u32) & 0x1FF) << 12 | 1 << 10 | u32::from(dst.0) << 5 | u32::from(prev.reg));
                return;
            }
        }
        let (imm, shifted) = if imm >= 4096 { (imm >> 12, 1) } else { (imm, 0) };
        let base = 0x9100_0000 | u32::from(sub) << 30 | u32::from(flags) << 29 | shifted << 22;
        self.emit_gp(dst, sized(base, size) | (imm & 0xFFF) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    /// `dst = a * b + c`, or `c - a * b` when `sub`, at `size`.
    pub fn madd(&mut self, sub: bool, size: Size, dst: Reg, (a, b): (Reg, Reg), c: Reg) {
        let base = if sub { 0x9B00_8000 } else { 0x9B00_0000 };
        self.emit_gp(dst, sized(base, size) | u32::from(b.0) << 16 | u32::from(c.0) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
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
        let count = |target: u16| (0..4).filter(|&h| ((value >> (16 * h)) as u16) == target).count();
        (count(0xFFFF), count(0))
    }

    fn single_bitmask_imm(value: u64) -> Option<(Size, (u32, u32, u32))> {
        encode_bitmask_imm(value, true).map(|b| (Size::X, b)).or_else(|| {
            ((value >> 32) == 0).then(|| encode_bitmask_imm(value, false).map(|b| (Size::W, b))).flatten()
        })
    }

    pub(super) fn mov_imm_is_single(imm: i64) -> bool {
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
        if let Some(hit) = fixed.into_iter().find_map(|c| Self::try_bitmask_half(value, k, c)) { return Some(hit); }
        let (prev_top, next_bot) = (h[(k + 3) & 3] >> 15, h[(k + 1) & 3] & 1);
        (prev_top != next_bot).then(|| (1..16_u32).find_map(|w| {
            let pat = (1_u16 << w) - 1;
            Self::try_bitmask_half(value, k, if prev_top == 1 { pat } else { !pat })
        })).flatten()
    }

    const MOVK: u32 = 0xF280_0000;
    fn try_emit_bitmask_imm(&mut self, dst: Reg, value: u64, ones: usize, zeros: usize) -> bool {
        if dst == ZR || ones >= 3 || zeros >= 3 { return false; }
        if let Some((size, bitmask)) = Self::single_bitmask_imm(value) {
            self.alu_bitmask(Alu::Orr, size, bitmask, dst, ZR);
            return true;
        }
        let h = [value as u16, (value >> 16) as u16, (value >> 32) as u16, (value >> 48) as u16];
        if ones < 2 && zeros < 2 && let Some((bitmask, k, hw)) = (0..4).find_map(|k| Self::bitmask_movk_at(value, h, k)) {
            self.alu_bitmask(Alu::Orr, Size::X, bitmask, dst, ZR);
            self.word(Self::MOVK | ((k as u32) << 21) | (u32::from(hw) << 5) | u32::from(dst.0));
            return true;
        }
        false
    }

    pub fn mov_imm(&mut self, dst: Reg, imm: i64) {
        self.clobber_anchor(dst);
        let value = imm as u64;
        let (ones, zeros) = Self::half_counts(value);
        if self.try_emit_bitmask_imm(dst, value, ones, zeros) { return; }
        let (inverted, skip, mut first) = (ones > zeros, if ones > zeros { 0xFFFF } else { 0 }, true);
        for half in 0..4 {
            let bits = (value >> (16 * half)) as u16;
            if bits == skip && !(first && half == 3) { continue; }
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
            let base = if half == 0 { 0xD280_0000 } else { Self::MOVK | ((half as u32) << 21) };
            self.word(base | (bits << 5) | rd);
        }
    }

    fn bitfield_word(&mut self, base: u32, dst: Reg, src: Reg, (immr, imms): (u8, u8)) {
        self.emit_gp(dst, base | u32::from(immr) << 16 | u32::from(imms) << 10 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    /// `ubfm`/`sbfm` with `immr` and `imms`: shifts and extensions by an immediate.
    pub fn bitfield(&mut self, signed: bool, dst: Reg, src: Reg, pair: (u8, u8)) {
        self.bitfield_word(if signed { 0x9340_0000 } else { 0xD340_0000 }, dst, src, pair);
    }

    pub fn lsl_imm(&mut self, dst: Reg, src: Reg, shift: u8) {
        self.bitfield(false, dst, src, ((64 - shift) % 64, 63 - shift));
    }

    pub fn shift32_imm(&mut self, left: bool, dst: Reg, src: Reg, shift: u8) {
        self.bitfield_word(0x5300_0000, dst, src, if left { ((32 - shift) % 32, 31 - shift) } else { (shift, 31) });
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
            4 if !signed => self.emit_gp(dst, 0x2A00_03E0 | u32::from(src.0) << 16 | u32::from(dst.0)),
            _ => self.bitfield(signed, dst, src, (0, (bytes * 8 - 1) as u8)),
        }
    }

    fn cond_select(&mut self, base: u32, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.emit_gp(dst, sized(base, size) | u32::from(b.0) << 16 | (cond as u32) << 12 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub(crate) fn ccmp(&mut self, neg: bool, size: Size, imm: bool, (cond, rn, op2, nzcv): (Cond, Reg, u8, u8)) { self.word(sized(if neg { 0xBA40_0000 } else { 0xFA40_0000 }, size) | u32::from(imm) << 11 | u32::from(op2 & 0x1F) << 16 | (cond as u32) << 12 | u32::from(rn.0) << 5 | u32::from(nzcv & 0xF)); }
    pub fn csel(&mut self, cond: Cond, dst: Reg, a: Reg, b: Reg) { self.cond_select(0x9A80_0000, Size::X, cond, dst, a, b); }

    pub fn csinc(&mut self, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) { self.cond_select(0x9A80_0400, size, cond, dst, a, b); }

    pub fn csinv(&mut self, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) { self.cond_select(0xDA80_0000, size, cond, dst, a, b); }

    pub fn csneg(&mut self, size: Size, cond: Cond, dst: Reg, a: Reg, b: Reg) { self.cond_select(0xDA80_0400, size, cond, dst, a, b); }

    pub fn cset(&mut self, cond: Cond, dst: Reg) {
        self.emit_gp(dst, 0x9A9F_07E0 | (cond.invert() as u32) << 12 | u32::from(dst.0));
    }

    pub fn bits(&mut self, op: Bits, dst: Reg, src: Reg) {
        self.emit_gp(dst, op as u32 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    fn load_base(size: Size, signed: bool) -> u32 {
        Self::store_base(size) | if signed && size != Size::X { 0x80_0000 } else { 0x40_0000 }
    }

    fn store_base(size: Size) -> u32 {
        [0x3800_0000, 0x7800_0000, 0xB800_0000, 0xF800_0000][size as usize]
    }

    fn pair_regs_ok(kind: PairKind, (r1, r2): (u8, u8), prev_reg: u8, base: Reg) -> bool {
        if matches!(kind, PairKind::IntLoad(_)) {
            return r1 != r2 && r1 != ZR.0 && r2 != ZR.0 && (base == SP || prev_reg != base.0);
        }
        !(matches!(kind, PairKind::FloatLoad(_)) && r1 == r2)
    }

    fn pair_match(prev: LastMem, (kind, reg): (PairKind, u8), mem: Mem, (pos, binds): (u32, u32)) -> Option<(u8, u8, i32)> {
        if prev.end_pos != pos || prev.binds != binds || prev.pair != Some(kind) || prev.base != mem.base {
            return None;
        }
        let step = kind.step();
        let (r1, r2, min_disp) = match mem.disp.checked_sub(prev.disp) {
            Some(d) if d == step => (prev.reg, reg, prev.disp),
            Some(d) if d == -step => (reg, prev.reg, mem.disp),
            _ => return None,
        };
        if min_disp.rem_euclid(step) != 0 || !(-64 * step..=63 * step).contains(&min_disp) {
            return None;
        }
        Self::pair_regs_ok(kind, (r1, r2), prev.reg, mem.base).then_some((r1, r2, min_disp))
    }

    fn pair_op(kind: PairKind) -> u32 {
        match kind {
            PairKind::IntLoad(s) => sized(0xA940_0000, s),
            PairKind::IntStore(s) => sized(0xA900_0000, s),
            PairKind::FloatLoad(s) => if s == Size::W { 0x2D40_0000 } else { 0x6D40_0000 },
            PairKind::FloatStore(s) => if s == Size::W { 0x2D00_0000 } else { 0x6D00_0000 },
        }
    }

    fn page_split(disp: i32, unit: i32) -> Option<(i64, u32)> {
        if disp % unit != 0 {
            return None;
        }
        let d = i64::from(disp);
        let rem = d.rem_euclid(4096);
        let page = d - rem;
        (page.unsigned_abs() < (1 << 24)).then_some((page, rem as u32))
    }

    fn anchor_mem(&mut self, scale: u32, mem: Mem) -> Mem {
        let unit = 1 << scale;
        if (mem.disp >= 0 && mem.disp % unit == 0 && mem.disp / unit < 4096) || (-256..256).contains(&mem.disp) || mem.base == Reg(17) {
            return mem;
        }
        let Some((page, rem)) = Self::page_split(mem.disp, unit) else { return mem };
        if self.last_anchor != Some((self.binds, mem.base, page)) {
            self.add_imm(page < 0, false, Size::X, Reg(17), mem.base, page.unsigned_abs() as u32);
            self.last_anchor = Some((self.binds, mem.base, page));
        }
        Mem::new(Reg(17), rem as i32)
    }

    fn access_pair(&mut self, pair: Option<(PairKind, u8)>, is_int: bool, base: u32, scale: u32, rt: u32, mem: Mem) {
        let mem = self.anchor_mem(scale, mem);
        if let Some((kind, reg)) = pair
            && let Some(prev) = self.last_mem
            && let Some((r1, r2, min_disp)) = Self::pair_match(prev, (kind, reg), mem, (self.pos(), self.binds))
        {
            self.code.truncate((prev.end_pos - 4) as usize);
            self.pair_offset(Self::pair_op(kind), kind.step(), r1, r2, mem.base, min_disp);
            if matches!(kind, PairKind::IntLoad(_)) {
                self.clobber_anchor(Reg(r1));
                self.clobber_anchor(Reg(r2));
            }
            self.last_mem = None;
            return;
        }
        let single = self.access(base, scale, rt, mem);
        self.last_mem = single.then(|| LastMem {
            end_pos: self.pos(), binds: self.binds, pair: pair.map(|(k, _)| k), base_op: base, is_int, reg: rt as u8, base: mem.base, disp: mem.disp,
        });
    }

    /// Loads `size` bytes at `mem` into `dst`, sign-extending to 64 bits when `signed`, else zero-extending.
    pub fn load(&mut self, size: Size, signed: bool, dst: Reg, mem: Mem) {
        let pair = (!signed && matches!(size, Size::W | Size::X)).then_some((PairKind::IntLoad(size), dst.0));
        self.access_pair(pair, true, Self::load_base(size, signed), size.log2(), u32::from(dst.0), mem);
        self.clobber_anchor(dst);
    }

    pub fn store(&mut self, size: Size, mem: Mem, src: Reg) {
        let pair = matches!(size, Size::W | Size::X).then_some((PairKind::IntStore(size), src.0));
        self.access_pair(pair, true, Self::store_base(size), size.log2(), u32::from(src.0), mem);
    }

    fn vpair(bytes: u32, reg: u8, load: bool) -> Option<(PairKind, u8)> {
        let size = match bytes { 4 => Size::W, 8 => Size::X, _ => return None };
        Some((if load { PairKind::FloatLoad(size) } else { PairKind::FloatStore(size) }, reg))
    }

    /// Loads a float or vector register of `bytes` 4, 8 or 16.
    pub fn vload(&mut self, bytes: u32, dst: Vreg, mem: Mem) {
        let (base, scale) = Self::vbase(bytes);
        self.access_pair(Self::vpair(bytes, dst.0, true), false, base | 0x40_0000, scale, u32::from(dst.0), mem);
    }

    pub fn vstore(&mut self, bytes: u32, mem: Mem, src: Vreg) {
        let (base, scale) = Self::vbase(bytes);
        self.access_pair(Self::vpair(bytes, src.0, false), false, base, scale, u32::from(src.0), mem);
    }

    /// Loads `size` bytes at `base + (index << shift)` into `dst`.
    pub fn load_indexed(&mut self, size: Size, signed: bool, dst: Reg, base: Reg, index: Reg, shift: u8) {
        self.access_indexed(Self::load_base(size, signed), u32::from(dst.0), base, index, shift);
        self.clobber_anchor(dst);
    }

    /// Stores `size` bytes from `src` to `base + (index << shift)`.
    pub fn store_indexed(&mut self, size: Size, base: Reg, index: Reg, shift: u8, src: Reg) {
        self.access_indexed(Self::store_base(size), u32::from(src.0), base, index, shift);
    }

    /// Loads a float register of `bytes` 4 or 8 at `base + (index << shift)`.
    pub fn vload_indexed(&mut self, bytes: u32, dst: Vreg, base: Reg, index: Reg, shift: u8) {
        self.access_indexed(Self::vbase(bytes).0 | 0x40_0000, u32::from(dst.0), base, index, shift);
    }

    /// Stores a float register of `bytes` 4 or 8 to `base + (index << shift)`.
    pub fn vstore_indexed(&mut self, bytes: u32, base: Reg, index: Reg, shift: u8, src: Vreg) {
        self.access_indexed(Self::vbase(bytes).0, u32::from(src.0), base, index, shift);
    }

    fn vbase(bytes: u32) -> (u32, u32) {
        match bytes { 4 => (0xBC00_0000, 2), 8 => (0xFC00_0000, 3), _ => (0x3C80_0000, 4) }
    }

    fn access_indexed(&mut self, base: u32, rt: u32, rn: Reg, rm: Reg, shift: u8) {
        let (ext, amt) = decode_sib(shift);
        self.word(base | 1 << 21 | u32::from(rm.0) << 16 | (ext as u32) << 13 | u32::from(amt > 0) << 12 | 0b10 << 10 | u32::from(rn.0) << 5 | rt);
    }

    /// One load or store from its unscaled-offset encoding `base`: the scaled form when the offset fits it, the unscaled form for small offsets, else through x17.
    fn access(&mut self, base: u32, scale: u32, rt: u32, mem: Mem) -> bool {
        let (disp, rn) = (mem.disp, u32::from(mem.base.0));
        let unit = 1 << scale;
        if disp >= 0 && disp % unit == 0 && disp / unit < 4096 {
            self.word(base | 1 << 24 | ((disp / unit) as u32) << 10 | rn << 5 | rt);
            true
        } else if (-256..256).contains(&disp) {
            self.word(base | ((disp as u32) & 0x1FF) << 12 | rn << 5 | rt);
            true
        } else if let Some((page, rem)) = Self::page_split(disp, unit) {
            self.add_imm(page < 0, false, Size::X, Reg(17), mem.base, page.unsigned_abs() as u32);
            self.word(base | 1 << 24 | (rem / (unit as u32)) << 10 | 17 << 5 | rt);
            false
        } else {
            self.mov_imm(Reg(17), i64::from(disp));
            self.access_indexed(base, rt, mem.base, Reg(17), 0);
            false
        }
    }

    pub(in crate::arm64) fn pair_offset(&mut self, base: u32, step: i32, a: u8, b: u8, addr: Reg, offset: i32) {
        self.word(base | (((offset / step) as u32) & 0x7F) << 15 | u32::from(b) << 10 | u32::from(addr.0) << 5 | u32::from(a));
    }

    pub fn stp_offset(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) { self.pair_offset(0xA900_0000, 8, a.0, b.0, base, offset); }

    pub fn ldp_offset(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.pair_offset(0xA940_0000, 8, a.0, b.0, base, offset);
        self.clobber_anchor(a);
        self.clobber_anchor(b);
    }

    pub fn fstp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) { self.pair_offset(0x6D00_0000, 8, a.0, b.0, base, offset); }
    pub fn fldp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) { self.pair_offset(0x6D40_0000, 8, a.0, b.0, base, offset); }
    pub fn qstp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) { self.pair_offset(0xAD00_0000, 16, a.0, b.0, base, offset); }
    pub fn qldp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) { self.pair_offset(0xAD40_0000, 16, a.0, b.0, base, offset); }

    pub fn stp_pre(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.word(0xA980_0000 | (((offset / 8) as u32) & 0x7F) << 15 | u32::from(b.0) << 10 | u32::from(base.0) << 5 | u32::from(a.0));
        self.clobber_anchor(base);
    }

    pub fn ldp_post(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.word(0xA8C0_0000 | (((offset / 8) as u32) & 0x7F) << 15 | u32::from(b.0) << 10 | u32::from(base.0) << 5 | u32::from(a.0));
        self.clobber_anchor(a);
        self.clobber_anchor(b);
        self.clobber_anchor(base);
    }

    pub fn blr(&mut self, reg: Reg) { self.clear_word(0xD63F_0000 | u32::from(reg.0) << 5); }

    pub fn bl_reloc(&mut self) -> u32 {
        self.clear_anchor();
        let at = self.pos();
        self.word(0x9400_0000);
        at
    }

    pub fn br(&mut self, reg: Reg) { self.clear_word(0xD61F_0000 | u32::from(reg.0) << 5); }

    pub fn ret(&mut self) { self.clear_word(0xD65F_03C0); }

    /// `brk #1`, the trap rustc emits for an abort.
    pub fn brk(&mut self) { self.clear_word(0xD420_0020); }
}
