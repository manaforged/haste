mod int;
mod relax;
mod simd;

use crate::reloc::{RelocKind, StencilReloc};
use relax::{Disp, Fixup};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Reg {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Xmm(pub u8);

/// The vector length of a VEX instruction: the xmm or the ymm view of its registers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Vl {
    X = 0,
    Y = 0b100,
}

/// One VEX opcode: its opcode map, prefix class, W bit and opcode byte.
#[derive(Clone, Copy)]
pub(crate) struct VexOp {
    pub map: u8,
    pub pp: u8,
    pub w: bool,
    pub op: u8,
}

impl VexOp {
    /// The opcode `op` of map `map` and prefix class `pp`, with W clear.
    pub const fn new(map: u8, pp: u8, op: u8) -> Self {
        Self {
            map,
            pp,
            w: false,
            op,
        }
    }

    /// The opcode `op` of map `map` and prefix class `pp`, with W set.
    pub const fn wide(map: u8, pp: u8, op: u8) -> Self {
        Self {
            map,
            pp,
            w: true,
            op,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Mem {
    pub base: Reg,
    pub index: Option<(Reg, u8)>,
    pub disp: i32,
    /// The symbol this address is relative to, through `rip`, in place of `base`.
    pub rip: Option<u32>,
}

impl Mem {
    pub fn new(base: Reg, disp: i32) -> Self {
        Self {
            base,
            index: None,
            disp,
            rip: None,
        }
    }

    /// The address `disp` bytes past symbol `sym`, reached relative to `rip`.
    pub fn rip(sym: u32, disp: i32) -> Self {
        Self {
            base: Reg::Rbp,
            index: None,
            disp,
            rip: Some(sym),
        }
    }

    pub fn sib(base: Reg, index: Reg, shift: u8, disp: i32) -> Self {
        debug_assert!(index != Reg::Rsp && shift <= 3);
        Self {
            base,
            index: Some((index, shift)),
            disp,
            rip: None,
        }
    }

    pub fn offset(self, by: i32) -> Self {
        Self {
            disp: self.disp + by,
            ..self
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Rm {
    Reg(u8),
    Mem(Mem),
}

impl Rm {
    fn byte_rex(self) -> bool {
        matches!(self, Rm::Reg(4..=7))
    }

    fn index_base(self) -> (u8, u8) {
        match self {
            Rm::Reg(reg) => (0, reg),
            Rm::Mem(mem) => (mem.index.map_or(0, |(idx, _)| idx as u8), mem.base as u8),
        }
    }
}

impl From<Reg> for Rm {
    fn from(reg: Reg) -> Self {
        Rm::Reg(reg as u8)
    }
}

impl From<Xmm> for Rm {
    fn from(xmm: Xmm) -> Self {
        Rm::Reg(xmm.0)
    }
}

impl From<Mem> for Rm {
    fn from(mem: Mem) -> Self {
        Rm::Mem(mem)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Size {
    B,
    W,
    D,
    Q,
}

impl Size {
    pub fn of(bytes: u32) -> Self {
        match bytes {
            1 => Size::B,
            2 => Size::W,
            4 => Size::D,
            _ => Size::Q,
        }
    }

    fn load_opcode(self) -> (bool, &'static [u8]) {
        match self {
            Size::B => (false, &[0x0F, 0xB6]),
            Size::W => (false, &[0x0F, 0xB7]),
            Size::D => (false, &[0x8B]),
            Size::Q => (true, &[0x8B]),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Alu {
    Add = 0,
    Or = 1,
    Adc = 2,
    Sbb = 3,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

#[derive(Clone, Copy)]
pub(crate) enum Shift {
    Rol = 0,
    Ror = 1,
    Shl = 4,
    Shr = 5,
    Sar = 7,
}

#[derive(Clone, Copy)]
pub(crate) enum Group3 {
    Not = 2,
    Neg = 3,
    Mul = 4,
    Imul = 5,
    Div = 6,
    Idiv = 7,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Count {
    Popcnt = 0xB8,
    Tzcnt = 0xBC,
    Lzcnt = 0xBD,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Cond {
    O = 0,
    No = 1,
    B = 2,
    Ae = 3,
    E = 4,
    Ne = 5,
    Be = 6,
    A = 7,
    S = 8,
    Ns = 9,
    P = 0xA,
    Np = 0xB,
    L = 0xC,
    Ge = 0xD,
    Le = 0xE,
    G = 0xF,
}

const CONDS: [Cond; 16] = [
    Cond::O,
    Cond::No,
    Cond::B,
    Cond::Ae,
    Cond::E,
    Cond::Ne,
    Cond::Be,
    Cond::A,
    Cond::S,
    Cond::Ns,
    Cond::P,
    Cond::Np,
    Cond::L,
    Cond::Ge,
    Cond::Le,
    Cond::G,
];

impl Cond {
    /// The condition that holds exactly when this one does not; x86 encodes it in the low bit.
    pub fn invert(self) -> Self {
        CONDS[(self as usize) ^ 1]
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Label(u32);

#[derive(Default)]
pub(crate) struct Asm {
    pub code: Vec<u8>,
    pub avx: bool,
    labels: Vec<Option<u32>>,
    pub(super) fixups: Vec<Fixup>,
    pub(super) table_fixups: Vec<(u32, Label, Label)>,
    pub(super) tail: Option<super::imm::Tail>,
    pub(super) zeros: Option<super::imm::PendingZeros>,
    /// The offset and symbol of each `rip`-relative displacement, holding its addend.
    pub(super) rips: Vec<(u32, u32)>,
    pub(super) aligns: Vec<(u32, u8)>,
}

impl Asm {
    pub fn pos(&self) -> u32 {
        self.code.len() as u32
    }

    pub fn align(&mut self, max_pad: u8) {
        if max_pad == 0 {
            return;
        }
        let at = self.pos();
        if let Some(last) = self.aligns.last_mut()
            && last.0 == at
        {
            last.1 = last.1.max(max_pad);
        } else {
            self.aligns.push((at, max_pad));
        }
    }

    pub(super) fn rex(&mut self, w: bool, r: u8, x: u8, b: u8, force: bool) {
        let rex =
            0x40 | (u8::from(w) << 3) | ((r >> 3 & 1) << 2) | ((x >> 3 & 1) << 1) | (b >> 3 & 1);
        if rex != 0x40 || force {
            self.code.push(rex);
        }
    }

    fn modrm(&mut self, r: u8, rm: Rm) {
        let mem = match rm {
            Rm::Reg(reg) => return self.code.push(0xC0 | (r & 7) << 3 | (reg & 7)),
            Rm::Mem(mem) => mem,
        };
        if let Some(sym) = mem.rip {
            self.code.push((r & 7) << 3 | 5);
            self.rips.push((self.pos(), sym));
            return self.code.extend(mem.disp.to_le_bytes());
        }
        let base = mem.base as u8;
        let mode = if mem.disp == 0 && base & 7 != 5 {
            0
        } else if i8::try_from(mem.disp).is_ok() {
            1
        } else {
            2
        };
        match mem.index {
            Some((idx, shift)) => {
                self.code.push(mode << 6 | (r & 7) << 3 | 4);
                self.code
                    .push((shift & 3) << 6 | ((idx as u8) & 7) << 3 | (base & 7));
            }
            None => {
                self.code.push(mode << 6 | (r & 7) << 3 | (base & 7));
                if base & 7 == 4 {
                    self.code.push(0x24);
                }
            }
        }
        match mode {
            1 => self.code.push(mem.disp as i8 as u8),
            2 => self.code.extend(mem.disp.to_le_bytes()),
            _ => {}
        }
    }

    pub fn encode(&mut self, prefix: &[u8], w: bool, opcode: &[u8], r: u8, rm: Rm, force: bool) {
        self.code.extend_from_slice(prefix);
        let (x, b) = rm.index_base();
        self.rex(w, r, x, b, force);
        self.code.extend_from_slice(opcode);
        self.modrm(r, rm);
    }

    pub(super) fn sized(
        &mut self,
        lock: bool,
        size: Size,
        (byte, wide): (&[u8], &[u8]),
        r: u8,
        rm: Rm,
    ) {
        let prefix: &[u8] = match (lock, size == Size::W) {
            (true, true) => &[0xF0, 0x66],
            (true, false) => &[0xF0],
            (false, true) => &[0x66],
            (false, false) => &[],
        };
        let opcode = if size == Size::B { byte } else { wide };
        self.encode(
            prefix,
            size == Size::Q,
            opcode,
            r,
            rm,
            size == Size::B && (Rm::Reg(r).byte_rex() || rm.byte_rex()),
        );
    }

    pub fn push(&mut self, reg: Reg) {
        self.rex(false, 0, 0, reg as u8, false);
        self.code.push(0x50 + (reg as u8 & 7));
    }

    pub fn pop(&mut self, reg: Reg) {
        self.rex(false, 0, 0, reg as u8, false);
        self.code.push(0x58 + (reg as u8 & 7));
    }

    /// `call` to relocation symbol `sym`: PC-relative when `direct`, else through its call table slot at a 32-bit absolute address.
    pub fn call_symbol(&mut self, sym: u32, direct: bool) -> StencilReloc {
        self.branch_symbol(sym, direct, false)
    }

    /// `call` or `jmp` to relocation symbol `sym`: PC-relative when `direct`, else through its call table slot at a 32-bit absolute address.
    pub fn branch_symbol(&mut self, sym: u32, direct: bool, tail: bool) -> StencilReloc {
        if direct {
            return StencilReloc::symbol(self.branch_pcrel32(tail), sym, RelocKind::PcRel32);
        }
        StencilReloc::symbol(self.branch_slot_reloc(tail), sym, RelocKind::Abs32)
    }

    pub fn branch_absolute(&mut self, at: i32, tail: bool) {
        self.flags_overwritten();
        self.code
            .extend([0xFF, if tail { 0x24 } else { 0x14 }, 0x25]);
        self.code.extend(at.to_le_bytes());
    }

    pub fn branch_slot_reloc(&mut self, tail: bool) -> u32 {
        self.flags_overwritten();
        self.code
            .extend([0xFF, if tail { 0x24 } else { 0x14 }, 0x25]);
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn call_pcrel32(&mut self) -> u32 {
        self.branch_pcrel32(false)
    }

    pub fn branch_pcrel32(&mut self, tail: bool) -> u32 {
        self.flags_overwritten();
        self.code.push(if tail { 0xE9 } else { 0xE8 });
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn call(&mut self, reg: Reg) {
        self.flags_overwritten();
        self.encode(&[], false, &[0xFF], 2, reg.into(), false);
    }

    pub fn jmp_reg(&mut self, reg: Reg) {
        self.flags_overwritten();
        self.encode(&[], false, &[0xFF], 4, reg.into(), false);
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    pub fn bind(&mut self, label: Label) {
        self.tail = None;
        self.zeros = None;
        self.labels[label.0 as usize] = Some(self.pos());
    }

    fn fixup(&mut self, label: Label, disp: Disp) {
        self.fixups.push(Fixup {
            at: self.pos(),
            label,
            disp,
        });
        self.code.extend([0; 4]);
    }

    pub fn jmp(&mut self, label: Label) {
        self.flags_overwritten();
        self.code.push(0xE9);
        self.fixup(label, Disp::Jmp);
    }

    pub fn jcc(&mut self, cond: Cond, label: Label) {
        self.code.extend([0x0F, 0x80 + cond as u8]);
        self.fixup(label, Disp::Jcc);
    }

    fn rip_head(&mut self, wide: bool, op: &[u8], dst: Reg) -> u32 {
        let reg = dst as u8;
        self.rex(wide, reg, 0, 0, false);
        self.code.extend(op);
        self.code.push(0x05 | ((reg & 7) << 3));
        self.pos()
    }

    pub fn lea_rip(&mut self, dst: Reg, label: Label) {
        self.rip_head(true, &[0x8D], dst);
        self.fixup(label, Disp::Rip);
    }

    pub fn lea_rip_reloc(&mut self, dst: Reg) -> u32 {
        let at = self.rip_head(true, &[0x8D], dst);
        self.code.extend([0; 4]);
        at
    }

    /// Emits a RIP-relative load into `dst` with a 32-bit zero displacement and returns the displacement offset.
    pub fn load_rip_reloc(&mut self, size: Size, dst: Reg) -> u32 {
        let (wide, op) = size.load_opcode();
        let at = self.rip_head(wide, op, dst);
        self.code.extend([0; 4]);
        at
    }

    pub fn table_entry(&mut self, base: Label, target: Label) {
        self.table_fixups.push((self.pos(), base, target));
        self.code.extend([0; 4]);
    }
}
