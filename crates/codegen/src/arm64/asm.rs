//! The A64 instruction encoder: one 32-bit word per instruction, with labels patched when they are bound.

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
    pub fn new(base: Reg, disp: i32) -> Self {
        Self { base, disp }
    }

    pub fn offset(self, by: i32) -> Self {
        Self { base: self.base, disp: self.disp + by }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

/// The width of a memory access or of an integer operation: 1, 2, 4 or 8 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Size {
    B,
    H,
    W,
    X,
}

impl Size {
    pub fn of(bytes: u32) -> Self {
        match bytes {
            1 => Size::B,
            2 => Size::H,
            4 => Size::W,
            _ => Size::X,
        }
    }

    fn log2(self) -> u32 {
        self as u32
    }
}

/// A condition on the flags.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cond {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
}

impl Cond {
    pub fn invert(self) -> Self {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
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
    Orr = 0xAA00_0000,
    Eor = 0xCA00_0000,
    Bic = 0x8A20_0000,
    Orn = 0xAA20_0000,
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
enum PairKind { IntLoad, IntStore, FloatLoad, FloatStore }

#[derive(Clone, Copy)]
struct LastMem { end_pos: u32, binds: u32, kind: PairKind, reg: u8, base: Reg, disp: i32 }

#[derive(Default)]
pub(crate) struct Asm {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, usize, Fixup)>,
    table_fixups: Vec<(u32, usize, usize)>,
    binds: u32,
    last_mem: Option<LastMem>,
}

fn sized(base: u32, size: Size) -> u32 {
    if size == Size::X { base } else { base & !(1 << 31) }
}

impl Asm {
    pub fn pos(&self) -> u32 {
        self.code.len() as u32
    }

    pub fn stamp(&self) -> (usize, u32) {
        (self.code.len(), self.binds)
    }

    pub fn word(&mut self, word: u32) {
        self.code.extend(word.to_le_bytes());
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = Some(self.pos());
        self.binds = self.binds.wrapping_add(1);
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
        self.jump(0x1000_0000 | u32::from(dst.0), label, Fixup::Adr);
    }

    pub fn ldrsw_scaled(&mut self, dst: Reg, base: Reg, index: Reg) {
        self.word(0xB8A0_7800 | u32::from(index.0) << 16 | u32::from(base.0) << 5 | u32::from(dst.0));
    }

    pub fn table_entry(&mut self, base: Label, target: Label) {
        self.table_fixups.push((self.pos(), base.0, target.0));
        self.word(0);
    }

    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        let mut veneers = Vec::new();
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let target = self.labels[label].ok_or("a jump to a label that was never bound")?;
            if self.try_patch(at, kind, target) {
                continue;
            }
            if !matches!(kind, Fixup::Cond | Fixup::Tbz) {
                return Err("a branch out of range".to_owned());
            }
            let veneer = self.emit_veneer(&mut veneers, label);
            if !self.try_patch(at, kind, veneer) {
                return Err("a branch out of range".to_owned());
            }
        }
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
        self.word(sized(op as u32, size) | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn alu_shifted(&mut self, op: Alu, size: Size, shift: Shift, imm: u8, dst: Reg, a: Reg, b: Reg) {
        self.word(sized(op as u32, size) | (shift as u32) << 22 | u32::from(b.0) << 16 | u32::from(imm) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn alu_bitmask(&mut self, op: Alu, size: Size, (n, immr, imms): (u32, u32, u32), dst: Reg, a: Reg) {
        let base = match op {
            Alu::And => 0x9200_0000,
            Alu::Orr => 0xB200_0000,
            Alu::Eor => 0xD200_0000,
            _ => return,
        };
        self.word(sized(base, size) | n << 22 | immr << 16 | imms << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn ror_imm(&mut self, size: Size, dst: Reg, src: Reg, imm: u8) {
        let base = if size == Size::X { 0x93C0_0000 } else { 0x1380_0000 };
        let s = u32::from(src.0);
        self.word(base | s << 16 | u32::from(imm) << 10 | s << 5 | u32::from(dst.0));
    }

    /// `dst = a + b` where `a` or `dst` may be the stack pointer.
    pub fn add_extended(&mut self, sub: bool, dst: Reg, a: Reg, b: Reg) {
        let base = if sub { 0xCB20_6000 } else { 0x8B20_6000 };
        self.word(base | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    /// `dst = base + (index << shift)`.
    pub fn add_shifted(&mut self, dst: Reg, base: Reg, index: Reg, shift: u8) {
        self.word(0x8B00_0000 | u32::from(shift) << 10 | u32::from(index.0) << 16 | u32::from(base.0) << 5 | u32::from(dst.0));
    }

    /// `dst = a +/- imm` for `imm < 4096` or a multiple of 4096 below 2^24, where `a` or `dst` may be the stack pointer; sets the flags when `flags`.
    pub fn add_imm(&mut self, sub: bool, flags: bool, size: Size, dst: Reg, a: Reg, imm: u32) {
        let (imm, shifted) = if imm >= 4096 { (imm >> 12, 1) } else { (imm, 0) };
        let base = 0x9100_0000 | u32::from(sub) << 30 | u32::from(flags) << 29 | shifted << 22;
        self.word(sized(base, size) | (imm & 0xFFF) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    /// `dst = a * b + c`, or `c - a * b` when `sub`.
    pub fn madd(&mut self, sub: bool, dst: Reg, (a, b): (Reg, Reg), c: Reg) {
        let base = if sub { 0x9B00_8000 } else { 0x9B00_0000 };
        self.word(base | u32::from(b.0) << 16 | u32::from(c.0) << 10 | u32::from(a.0) << 5 | u32::from(dst.0));
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

    pub fn mov_imm(&mut self, dst: Reg, imm: i64) {
        let value = imm as u64;
        let halves = |word: u64| (0..4).map(move |half| (word >> (16 * half)) as u16);
        let ones = halves(value).filter(|half| *half == 0xFFFF).count();
        let zeros = halves(value).filter(|half| *half == 0).count();
        if zeros < 3 && ones < 3 && dst != ZR && let Some(bitmask) = encode_bitmask_imm(value, true) {
            self.alu_bitmask(Alu::Orr, Size::X, bitmask, dst, ZR);
            return;
        }
        let inverted = ones > zeros;
        let skip = if inverted { 0xFFFF } else { 0 };
        let mut first = true;
        for (half, bits) in halves(value).enumerate() {
            if bits == skip && !(first && half == 3) {
                continue;
            }
            let shift = (half as u32) << 21;
            let word = match (first, inverted) {
                (true, true) => 0x9280_0000 | shift | u32::from(!bits) << 5,
                (true, false) => 0xD280_0000 | shift | u32::from(bits) << 5,
                (false, _) => 0xF280_0000 | shift | u32::from(bits) << 5,
            };
            self.word(word | u32::from(dst.0));
            first = false;
        }
    }

    pub fn mov_symbol(&mut self, dst: Reg) -> u32 {
        let at = self.pos();
        let rd = u32::from(dst.0);
        self.word(0xD280_0000 | rd);
        self.word(0xF2A0_0000 | rd);
        self.word(0xF2C0_0000 | rd);
        self.word(0xF2E0_0000 | rd);
        at
    }

    pub fn mov_fixed64(&mut self, dst: Reg, value: u64) {
        let rd = u32::from(dst.0);
        for half in 0..4 {
            let bits = ((value >> (16 * half)) as u32) & 0xFFFF;
            let base = if half == 0 { 0xD280_0000 } else { 0xF280_0000 | ((half as u32) << 21) };
            self.word(base | (bits << 5) | rd);
        }
    }

    /// `ubfm`/`sbfm` with `immr` and `imms`: shifts and extensions by an immediate.
    pub fn bitfield(&mut self, signed: bool, dst: Reg, src: Reg, (immr, imms): (u8, u8)) {
        let base = if signed { 0x9340_0000 } else { 0xD340_0000 };
        self.word(base | u32::from(immr) << 16 | u32::from(imms) << 10 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn lsl_imm(&mut self, dst: Reg, src: Reg, shift: u8) {
        self.bitfield(false, dst, src, ((64 - shift) % 64, 63 - shift));
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
            4 if !signed => self.word(0x2A00_03E0 | u32::from(src.0) << 16 | u32::from(dst.0)),
            _ => self.bitfield(signed, dst, src, (0, (bytes * 8 - 1) as u8)),
        }
    }

    pub fn csel(&mut self, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.word(0x9A80_0000 | u32::from(b.0) << 16 | (cond as u32) << 12 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn csinc(&mut self, cond: Cond, dst: Reg, a: Reg, b: Reg) {
        self.word(0x9A80_0400 | u32::from(b.0) << 16 | (cond as u32) << 12 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn cset(&mut self, cond: Cond, dst: Reg) {
        self.word(0x9A9F_07E0 | (cond.invert() as u32) << 12 | u32::from(dst.0));
    }

    pub fn bits(&mut self, op: Bits, dst: Reg, src: Reg) {
        self.word(op as u32 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    fn load_base(size: Size, signed: bool) -> u32 {
        match (size, signed) {
            (Size::X, _) => 0xF840_0000,
            (Size::W, false) => 0xB840_0000,
            (Size::H, false) => 0x7840_0000,
            (Size::B, false) => 0x3840_0000,
            (Size::W, true) => 0xB880_0000,
            (Size::H, true) => 0x7880_0000,
            (Size::B, true) => 0x3880_0000,
        }
    }

    fn store_base(size: Size) -> u32 {
        [0x3800_0000, 0x7800_0000, 0xB800_0000, 0xF800_0000][size as usize]
    }

    fn access_pair(&mut self, pair: Option<(PairKind, u8)>, base: u32, scale: u32, rt: u32, mem: Mem) {
        if let Some((kind, reg)) = pair
            && let Some(prev) = self.last_mem
            && prev.end_pos == self.pos()
            && prev.binds == self.binds
            && prev.kind == kind
            && prev.base == mem.base
            && let Some((r1, r2, min_disp)) = match mem.disp.checked_sub(prev.disp) {
                Some(8) => Some((prev.reg, reg, prev.disp)),
                Some(-8) => Some((reg, prev.reg, mem.disp)),
                _ => None,
            }
            && min_disp % 8 == 0
            && (-512..=504).contains(&min_disp)
            && !(kind == PairKind::IntLoad && (r1 == r2 || r1 == ZR.0 || r2 == ZR.0 || (mem.base != SP && (prev.reg == mem.base.0 || reg == mem.base.0))))
            && !(kind == PairKind::FloatLoad && r1 == r2)
        {
            self.code.truncate((prev.end_pos - 4) as usize);
            match kind {
                PairKind::IntLoad => self.ldp_offset(Reg(r1), Reg(r2), mem.base, min_disp),
                PairKind::IntStore => self.stp_offset(Reg(r1), Reg(r2), mem.base, min_disp),
                PairKind::FloatLoad => self.fldp_offset(Vreg(r1), Vreg(r2), mem.base, min_disp),
                PairKind::FloatStore => self.fstp_offset(Vreg(r1), Vreg(r2), mem.base, min_disp),
            }
            self.last_mem = None;
            return;
        }
        let single = self.access(base, scale, rt, mem);
        self.last_mem = pair.filter(|_| single).map(|(kind, reg)| LastMem { end_pos: self.pos(), binds: self.binds, kind, reg, base: mem.base, disp: mem.disp });
    }

    /// Loads `size` bytes at `mem` into `dst`, sign-extending to 64 bits when `signed`, else zero-extending.
    pub fn load(&mut self, size: Size, signed: bool, dst: Reg, mem: Mem) {
        let pair = (size == Size::X).then_some((PairKind::IntLoad, dst.0));
        self.access_pair(pair, Self::load_base(size, signed), size.log2(), u32::from(dst.0), mem);
    }

    pub fn store(&mut self, size: Size, mem: Mem, src: Reg) {
        let pair = (size == Size::X).then_some((PairKind::IntStore, src.0));
        self.access_pair(pair, Self::store_base(size), size.log2(), u32::from(src.0), mem);
    }

    /// Loads a float or vector register of `bytes` 4, 8 or 16.
    pub fn vload(&mut self, bytes: u32, dst: Vreg, mem: Mem) {
        let (base, scale) = Self::vbase(bytes);
        let pair = (bytes == 8).then_some((PairKind::FloatLoad, dst.0));
        self.access_pair(pair, base | 0x40_0000, scale, u32::from(dst.0), mem);
    }

    pub fn vstore(&mut self, bytes: u32, mem: Mem, src: Vreg) {
        let (base, scale) = Self::vbase(bytes);
        let pair = (bytes == 8).then_some((PairKind::FloatStore, src.0));
        self.access_pair(pair, base, scale, u32::from(src.0), mem);
    }

    /// Loads `size` bytes at `base + (index << shift)` into `dst`.
    pub fn load_indexed(&mut self, size: Size, signed: bool, dst: Reg, base: Reg, index: Reg, shift: u8) {
        self.access_indexed(Self::load_base(size, signed), u32::from(dst.0), base, index, shift);
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
        match bytes {
            4 => (0xBC00_0000, 2),
            8 => (0xFC00_0000, 3),
            _ => (0x3C80_0000, 4),
        }
    }

    fn access_indexed(&mut self, base: u32, rt: u32, rn: Reg, rm: Reg, shift: u8) {
        self.word(base | 1 << 21 | u32::from(rm.0) << 16 | 0b011 << 13 | u32::from(shift > 0) << 12 | 0b10 << 10 | u32::from(rn.0) << 5 | rt);
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
        } else {
            self.mov_imm(Reg(17), i64::from(disp));
            self.access_indexed(base, rt, mem.base, Reg(17), 0);
            false
        }
    }

    fn pair_offset(&mut self, base: u32, a: u8, b: u8, addr: Reg, offset: i32) {
        self.word(base | (((offset / 8) as u32) & 0x7F) << 15 | u32::from(b) << 10 | u32::from(addr.0) << 5 | u32::from(a));
    }

    pub fn stp_offset(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.pair_offset(0xA900_0000, a.0, b.0, base, offset);
    }

    pub fn ldp_offset(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.pair_offset(0xA940_0000, a.0, b.0, base, offset);
    }

    pub fn fstp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) {
        self.pair_offset(0x6D00_0000, a.0, b.0, base, offset);
    }

    pub fn fldp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) {
        self.pair_offset(0x6D40_0000, a.0, b.0, base, offset);
    }

    pub fn stp_pre(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.word(0xA980_0000 | (((offset / 8) as u32) & 0x7F) << 15 | u32::from(b.0) << 10 | u32::from(base.0) << 5 | u32::from(a.0));
    }

    pub fn ldp_post(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.word(0xA8C0_0000 | (((offset / 8) as u32) & 0x7F) << 15 | u32::from(b.0) << 10 | u32::from(base.0) << 5 | u32::from(a.0));
    }

    pub fn blr(&mut self, reg: Reg) {
        self.word(0xD63F_0000 | u32::from(reg.0) << 5);
    }

    pub fn bl_reloc(&mut self) -> u32 {
        let at = self.pos();
        self.word(0x9400_0000);
        at
    }

    pub fn br(&mut self, reg: Reg) {
        self.word(0xD61F_0000 | u32::from(reg.0) << 5);
    }

    pub fn ret(&mut self) {
        self.word(0xD65F_03C0);
    }

    /// `brk #1`, the trap rustc emits for an abort.
    pub fn brk(&mut self) {
        self.word(0xD420_0020);
    }

    /// `dmb ish`: a full barrier between this core's memory accesses and other cores'.
    pub fn dmb(&mut self) {
        self.word(0xD503_3BBF);
    }

    /// `ldar`, or `ldaxr` when `exclusive`.
    pub fn load_acquire(&mut self, exclusive: bool, size: Size, dst: Reg, addr: Reg) {
        let base = if exclusive { 0x085F_FC00 } else { 0x08DF_FC00 };
        self.word(base | (size as u32) << 30 | u32::from(addr.0) << 5 | u32::from(dst.0));
    }

    /// `stlr`.
    pub fn store_release(&mut self, size: Size, addr: Reg, src: Reg) {
        self.word(0x089F_FC00 | (size as u32) << 30 | u32::from(addr.0) << 5 | u32::from(src.0));
    }

    /// `stlxr status, src, [addr]`: status is zero when the store took.
    pub fn store_exclusive(&mut self, size: Size, status: Reg, addr: Reg, src: Reg) {
        self.word(0x0800_FC00 | (size as u32) << 30 | u32::from(status.0) << 16 | u32::from(addr.0) << 5 | u32::from(src.0));
    }

    /// The double-precision form sets bit 22 of a single-precision encoding.
    fn precision(base: u32, double: bool) -> u32 {
        base | u32::from(double) << 22
    }

    pub fn fop(&mut self, op: Fop, double: bool, dst: Vreg, a: Vreg, b: Vreg) {
        self.word(Self::precision(op as u32, double) | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn funary(&mut self, op: Funary, double: bool, dst: Vreg, src: Vreg) {
        self.word(Self::precision(op as u32, double) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn fcmp(&mut self, double: bool, a: Vreg, b: Vreg) {
        self.word(Self::precision(0x1E20_2000, double) | u32::from(b.0) << 16 | u32::from(a.0) << 5);
    }

    /// `fcvt` from single to double precision, or back when `to_single`.
    pub fn fcvt(&mut self, to_single: bool, dst: Vreg, src: Vreg) {
        let word = if to_single { 0x1E62_4000 } else { 0x1E22_C000 };
        self.word(word | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    /// Converts between the integer register and the float register of a conversion, 64-bit integers when `wide`.
    pub fn fconv(&mut self, op: Fconv, double: bool, wide: bool, int: Reg, float: Vreg) {
        let base = Self::precision(op as u32, double) | u32::from(wide) << 31;
        let (dst, src) = match op {
            Fconv::Scvtf | Fconv::Ucvtf => (u32::from(float.0), u32::from(int.0)),
            Fconv::Fcvtzs | Fconv::Fcvtzu => (u32::from(int.0), u32::from(float.0)),
        };
        self.word(base | src << 5 | dst);
    }

    pub fn fmov_reg(&mut self, _double: bool, dst: Vreg, src: Vreg) {
        self.vmov(dst, src);
    }

    pub fn fcsel(&mut self, cond: Cond, double: bool, dst: Vreg, a: Vreg, b: Vreg) {
        self.word(Self::precision(0x1E20_0C00, double) | u32::from(b.0) << 16 | (cond as u32) << 12 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn fmov_in(&mut self, double: bool, dst: Vreg, src: Reg) {
        let base = Self::precision(0x1E27_0000, double) | u32::from(double) << 31;
        self.word(base | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn fmov_out(&mut self, double: bool, dst: Reg, src: Vreg) {
        let base = Self::precision(0x1E26_0000, double) | u32::from(double) << 31;
        self.word(base | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn fmov_to_float(&mut self, dst: Vreg, src: Reg) {
        self.fmov_in(true, dst, src);
    }

    pub fn fmov_to_int(&mut self, dst: Reg, src: Vreg) {
        self.fmov_out(true, dst, src);
    }

    /// Counts the set bits of each byte of `src` and adds them into the low byte of `dst`.
    pub fn popcount_bytes(&mut self, dst: Vreg, src: Vreg) {
        self.word(0x0E20_5800 | u32::from(src.0) << 5 | u32::from(dst.0));
        self.word(0x0E31_B800 | u32::from(dst.0) << 5 | u32::from(dst.0));
    }

    pub fn vmov(&mut self, dst: Vreg, src: Vreg) {
        if dst != src {
            let s = u32::from(src.0);
            self.word(0x4EA0_1C00 | s << 16 | s << 5 | u32::from(dst.0));
        }
    }

    pub fn veor_zero(&mut self, dst: Vreg) {
        let d = u32::from(dst.0);
        self.word(0x6E20_1C00 | d << 16 | d << 5 | d);
    }

    pub fn vbit(&mut self, base: u32, dst: Vreg, a: Vreg, b: Vreg) {
        self.word(base | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    pub fn umov_d1(&mut self, dst: Reg, src: Vreg) {
        self.word(0x4E18_3C00 | u32::from(src.0) << 5 | u32::from(dst.0));
    }
}
