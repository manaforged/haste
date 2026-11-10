//! The A64 instruction encoder: one 32-bit word per instruction, with labels patched when they are bound.

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

#[derive(Clone, Copy)]
enum Fixup {
    /// A 26-bit word offset: `b` and `bl`.
    Branch,
    /// A 19-bit word offset at bit 5: `b.cond`, `cbz`, `cbnz`.
    Cond,
}

#[derive(Default)]
pub(crate) struct Asm {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, usize, Fixup)>,
}

fn sized(base: u32, size: Size) -> u32 {
    if size == Size::X { base } else { base & !(1 << 31) }
}

impl Asm {
    pub fn pos(&self) -> u32 {
        self.code.len() as u32
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
    }

    fn jump(&mut self, word: u32, label: Label, kind: Fixup) {
        self.fixups.push((self.pos(), label.0, kind));
        self.word(word);
    }

    pub fn b(&mut self, label: Label) {
        self.jump(0x1400_0000, label, Fixup::Branch);
    }

    pub fn bcond(&mut self, cond: Cond, label: Label) {
        self.jump(0x5400_0000 | cond as u32, label, Fixup::Cond);
    }

    /// Branches to `label` when `reg` is zero, or nonzero when `nonzero`.
    pub fn cbz(&mut self, nonzero: bool, size: Size, reg: Reg, label: Label) {
        let base = if nonzero { 0xB500_0000 } else { 0xB400_0000 };
        self.jump(sized(base, size) | u32::from(reg.0), label, Fixup::Cond);
    }

    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let target = self.labels[label].ok_or("a jump to a label that was never bound")?;
            let delta = (i64::from(target) - i64::from(at)) / 4;
            let at = at as usize;
            let mut word = u32::from_le_bytes(self.code[at..at + 4].try_into().expect("four bytes"));
            word |= match kind {
                Fixup::Branch if (-(1 << 25)..1 << 25).contains(&delta) => (delta as u32) & 0x03FF_FFFF,
                Fixup::Cond if (-(1 << 18)..1 << 18).contains(&delta) => ((delta as u32) & 0x7_FFFF) << 5,
                _ => return Err("a branch out of range".to_owned()),
            };
            self.code[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        Ok(self.code)
    }

    /// `op dst, a, b` on registers, where 31 names the zero register.
    pub fn alu(&mut self, op: Alu, size: Size, dst: Reg, a: Reg, b: Reg) {
        self.word(sized(op as u32, size) | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
    }

    /// `dst = a + b` where `a` or `dst` may be the stack pointer.
    pub fn add_extended(&mut self, sub: bool, dst: Reg, a: Reg, b: Reg) {
        let base = if sub { 0xCB20_6000 } else { 0x8B20_6000 };
        self.word(base | u32::from(b.0) << 16 | u32::from(a.0) << 5 | u32::from(dst.0));
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

    pub fn cset(&mut self, cond: Cond, dst: Reg) {
        self.word(0x9A9F_07E0 | (cond.invert() as u32) << 12 | u32::from(dst.0));
    }

    pub fn bits(&mut self, op: Bits, dst: Reg, src: Reg) {
        self.word(op as u32 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    /// Loads `size` bytes at `mem` into `dst`, sign-extending to 64 bits when `signed`, else zero-extending.
    pub fn load(&mut self, size: Size, signed: bool, dst: Reg, mem: Mem) {
        let base = match (size, signed) {
            (Size::X, _) => 0xF840_0000,
            (Size::W, false) => 0xB840_0000,
            (Size::H, false) => 0x7840_0000,
            (Size::B, false) => 0x3840_0000,
            (Size::W, true) => 0xB880_0000,
            (Size::H, true) => 0x7880_0000,
            (Size::B, true) => 0x3880_0000,
        };
        self.access(base, size.log2(), u32::from(dst.0), mem);
    }

    pub fn store(&mut self, size: Size, mem: Mem, src: Reg) {
        let base = [0x3800_0000, 0x7800_0000, 0xB800_0000, 0xF800_0000][size as usize];
        self.access(base, size.log2(), u32::from(src.0), mem);
    }

    /// Loads a float or vector register of `bytes` 4, 8 or 16.
    pub fn vload(&mut self, bytes: u32, dst: Vreg, mem: Mem) {
        let (base, scale) = Self::vbase(bytes);
        self.access(base | 0x40_0000, scale, u32::from(dst.0), mem);
    }

    pub fn vstore(&mut self, bytes: u32, mem: Mem, src: Vreg) {
        let (base, scale) = Self::vbase(bytes);
        self.access(base, scale, u32::from(src.0), mem);
    }

    fn vbase(bytes: u32) -> (u32, u32) {
        match bytes {
            4 => (0xBC00_0000, 2),
            8 => (0xFC00_0000, 3),
            _ => (0x3C80_0000, 4),
        }
    }

    /// One load or store from its unscaled-offset encoding `base`: the scaled form when the offset fits it, the unscaled form for small offsets, else through x17.
    fn access(&mut self, base: u32, scale: u32, rt: u32, mem: Mem) {
        let (disp, rn) = (mem.disp, u32::from(mem.base.0));
        let unit = 1 << scale;
        if disp >= 0 && disp % unit == 0 && disp / unit < 4096 {
            self.word(base | 1 << 24 | ((disp / unit) as u32) << 10 | rn << 5 | rt);
        } else if (-256..256).contains(&disp) {
            self.word(base | ((disp as u32) & 0x1FF) << 12 | rn << 5 | rt);
        } else {
            self.mov_imm(Reg(17), i64::from(disp));
            self.word(base | 1 << 21 | 17 << 16 | 0b011 << 13 | 0b10 << 10 | rn << 5 | rt);
        }
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

    /// Moves the 64 bits of `src` into `dst`.
    pub fn fmov_to_float(&mut self, dst: Vreg, src: Reg) {
        self.word(0x9E67_0000 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn fmov_to_int(&mut self, dst: Reg, src: Vreg) {
        self.word(0x9E66_0000 | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    /// Counts the set bits of each byte of `src` and adds them into the low byte of `dst`.
    pub fn popcount_bytes(&mut self, dst: Vreg, src: Vreg) {
        self.word(0x0E20_5800 | u32::from(src.0) << 5 | u32::from(dst.0));
        self.word(0x0E31_B800 | u32::from(dst.0) << 5 | u32::from(dst.0));
    }
}
