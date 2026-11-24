mod atomic;
mod float;
mod int;
mod mem;

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
        Self {
            base: self.base,
            disp: self.disp + by,
        }
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

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExtOp {
    Uxtb = 0,
    Uxth = 1,
    Uxtw = 2,
    Uxtx = 3,
    Sxtb = 4,
    Sxth = 5,
    Sxtw = 6,
}

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
pub(crate) enum Size {
    B,
    H,
    W,
    X,
}

impl Size {
    pub fn of(bytes: u32) -> Self {
        match bytes {
            1 => Self::B,
            2 => Self::H,
            4 => Self::W,
            _ => Self::X,
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
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Hs => Self::Lo,
            Self::Lo => Self::Hs,
            Self::Mi => Self::Pl,
            Self::Pl => Self::Mi,
            Self::Vs => Self::Vc,
            Self::Vc => Self::Vs,
            Self::Hi => Self::Ls,
            Self::Ls => Self::Hi,
            Self::Ge => Self::Lt,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
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
    Branch,
    Cond,
    Tbz,
    Adr,
}

fn fixup_bits(kind: Fixup, bytes: i64) -> Option<u32> {
    let delta = bytes / 4;
    match kind {
        Fixup::Branch if (-(1 << 25)..1 << 25).contains(&delta) => {
            Some((delta as u32) & 0x03FF_FFFF)
        }
        Fixup::Cond if (-(1 << 18)..1 << 18).contains(&delta) => {
            Some(((delta as u32) & 0x7_FFFF) << 5)
        }
        Fixup::Tbz if (-(1 << 13)..1 << 13).contains(&delta) => {
            Some(((delta as u32) & 0x3FFF) << 5)
        }
        Fixup::Adr if (-(1 << 20)..1 << 20).contains(&bytes) => {
            let imm = bytes as u32;
            Some(((imm & 3) << 29) | (((imm >> 2) & 0x7_FFFF) << 5))
        }
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PairKind {
    IntLoad(Size),
    IntStore(Size),
    FloatLoad(Size),
    FloatStore(Size),
}

impl PairKind {
    fn step(self) -> i32 {
        if matches!(
            self,
            Self::IntLoad(Size::W)
                | Self::IntStore(Size::W)
                | Self::FloatLoad(Size::W)
                | Self::FloatStore(Size::W)
        ) {
            4
        } else {
            8
        }
    }
}

#[derive(Clone, Copy)]
struct LastMem {
    end_pos: u32,
    binds: u32,
    pair: Option<PairKind>,
    base_op: u32,
    is_int: bool,
    reg: u8,
    base: Reg,
    disp: i32,
}

#[derive(Default)]
pub(crate) struct Asm {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, usize, Fixup)>,
    table_fixups: Vec<(u32, usize, usize)>,
    binds: u32,
    last_mem: Option<LastMem>,
    last_anchor: Option<(u32, Reg, i64)>,
    last_imm: Option<(u32, Reg, i64)>,
}

fn sized(base: u32, size: Size) -> u32 {
    if size == Size::X {
        base
    } else {
        base & !(1 << 31)
    }
}

impl Asm {
    pub fn pos(&self) -> u32 {
        self.code.len() as u32
    }

    pub fn word(&mut self, word: u32) {
        self.code.extend(word.to_le_bytes());
    }

    pub(super) fn clear_anchor(&mut self) {
        self.last_anchor = None;
        self.last_imm = None;
    }

    pub(super) fn clobber_anchor(&mut self, reg: Reg) {
        if reg == Reg(17) || self.last_anchor.is_some_and(|(_, base, _)| base == reg) {
            self.last_anchor = None;
        }
        if self.last_imm.is_some_and(|(_, r, _)| r == reg) {
            self.last_imm = None;
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

    pub fn bl(&mut self, label: Label) {
        self.clear_anchor();
        self.jump(0x9400_0000, label, Fixup::Branch);
    }

    pub fn bcond(&mut self, cond: Cond, label: Label) {
        self.jump_cond(
            Fixup::Cond,
            0x5400_0000 | cond as u32,
            0x5400_0000 | cond.invert() as u32,
            label,
        );
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
        let word =
            base | u32::from(bit >> 5) << 31 | u32::from(bit & 0x1F) << 19 | u32::from(reg.0);
        self.jump_cond(Fixup::Tbz, word, word ^ (1 << 24), label);
    }

    fn try_patch(&mut self, at: u32, kind: Fixup, target: u32) -> bool {
        let Some(bits) = fixup_bits(kind, i64::from(target) - i64::from(at)) else {
            return false;
        };
        let at = at as usize;
        let mut word = u32::from_le_bytes([
            self.code[at],
            self.code[at + 1],
            self.code[at + 2],
            self.code[at + 3],
        ]);
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
        let need_tbz = self
            .fixups
            .iter()
            .any(|&(at, l, k)| k == Fixup::Tbz && self.labels[l].is_none() && now - at >= 16_384);
        let need_cond = self
            .fixups
            .iter()
            .any(|&(at, l, k)| k == Fixup::Cond && self.labels[l].is_none() && now - at >= 524_288);
        if !need_tbz && !need_cond {
            return Ok(());
        }
        let over = self.label();
        self.b(over);
        let mut veneers = Vec::new();
        for (at, label, kind) in std::mem::take(&mut self.fixups) {
            let flush = self.labels[label].is_none()
                && ((kind == Fixup::Tbz && (need_tbz || now - at >= 8_192))
                    || (kind == Fixup::Cond && need_cond));
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

    pub fn blr(&mut self, reg: Reg) {
        self.clear_word(0xD63F_0000 | u32::from(reg.0) << 5);
    }

    pub fn bl_reloc(&mut self) -> u32 {
        self.clear_anchor();
        let at = self.pos();
        self.word(0x9400_0000);
        at
    }

    pub fn br(&mut self, reg: Reg) {
        self.clear_word(0xD61F_0000 | u32::from(reg.0) << 5);
    }

    pub fn ret(&mut self) {
        self.clear_word(0xD65F_03C0);
    }

    /// `brk #1`, the trap rustc emits for an abort.
    pub fn brk(&mut self) {
        self.clear_word(0xD420_0020);
    }
}
