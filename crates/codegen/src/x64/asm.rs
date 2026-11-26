mod relax;

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
        Self { map, pp, w: false, op }
    }

    /// The opcode `op` of map `map` and prefix class `pp`, with W set.
    pub const fn wide(map: u8, pp: u8, op: u8) -> Self {
        Self { map, pp, w: true, op }
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
        Self { base, index: None, disp, rip: None }
    }

    /// The address `disp` bytes past symbol `sym`, reached relative to `rip`.
    pub fn rip(sym: u32, disp: i32) -> Self {
        Self { base: Reg::Rbp, index: None, disp, rip: Some(sym) }
    }

    pub fn sib(base: Reg, index: Reg, shift: u8, disp: i32) -> Self {
        debug_assert!(index != Reg::Rsp && shift <= 3);
        Self { base, index: Some((index, shift)), disp, rip: None }
    }

    pub fn offset(self, by: i32) -> Self {
        Self { disp: self.disp + by, ..self }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Rm {
    Reg(u8),
    Mem(Mem),
}

impl Rm {
    /// True when a byte access to this operand names spl, bpl, sil or dil, which needs a REX prefix.
    fn byte_rex(self) -> bool {
        matches!(self, Rm::Reg(4..=7))
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
}

impl Asm {
    pub fn pos(&self) -> u32 {
        self.code.len() as u32
    }

    pub(super) fn rex(&mut self, w: bool, r: u8, x: u8, b: u8, force: bool) {
        let rex = 0x40 | (u8::from(w) << 3) | ((r >> 3 & 1) << 2) | ((x >> 3 & 1) << 1) | (b >> 3 & 1);
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
                self.code.push((shift & 3) << 6 | ((idx as u8) & 7) << 3 | (base & 7));
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
        let (x, b) = match rm {
            Rm::Reg(reg) => (0, reg),
            Rm::Mem(mem) => (mem.index.map_or(0, |(idx, _)| idx as u8), mem.base as u8),
        };
        self.rex(w, r, x, b, force);
        self.code.extend_from_slice(opcode);
        self.modrm(r, rm);
    }

    pub(super) fn sized(&mut self, lock: bool, size: Size, (byte, wide): (&[u8], &[u8]), r: u8, rm: Rm) {
        let prefix: &[u8] = match (lock, size == Size::W) {
            (true, true) => &[0xF0, 0x66],
            (true, false) => &[0xF0],
            (false, true) => &[0x66],
            (false, false) => &[],
        };
        let opcode = if size == Size::B { byte } else { wide };
        self.encode(prefix, size == Size::Q, opcode, r, rm, size == Size::B && (Rm::Reg(r).byte_rex() || rm.byte_rex()));
    }

    pub fn load(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) {
        let (w, opcode): (bool, &[u8]) = match size {
            Size::B => (false, &[0x0F, 0xB6]),
            Size::W => (false, &[0x0F, 0xB7]),
            Size::D => (false, &[0x8B]),
            Size::Q => (true, &[0x8B]),
        };
        let (start, src) = (self.pos(), src.into());
        self.encode(&[], w, opcode, dst as u8, src, size == Size::B && src.byte_rex());
        self.flags_kept(start);
    }

    pub fn load_signed(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) {
        let opcode: &[u8] = match size {
            Size::B => &[0x0F, 0xBE],
            Size::W => &[0x0F, 0xBF],
            Size::D => &[0x63],
            Size::Q => &[0x8B],
        };
        let start = self.pos();
        self.encode(&[], true, opcode, dst as u8, src.into(), false);
        self.flags_kept(start);
    }

    pub fn store(&mut self, size: Size, dst: Mem, src: Reg) {
        let start = self.pos();
        self.sized(false, size, (&[0x88], &[0x89]), src as u8, dst.into());
        self.flags_kept(start);
    }

    pub fn store_slot_reloc(&mut self, dst: Mem) -> u32 {
        self.encode(&[], true, &[0xC7], 0, dst.into(), false);
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn mov(&mut self, dst: Reg, src: Reg) {
        let start = self.pos();
        self.encode(&[], true, &[0x8B], dst as u8, src.into(), false);
        self.flags_kept(start);
    }

    pub fn mov_slot_reloc(&mut self, dst: Reg) -> u32 {
        let reg = dst as u8;
        self.rex(false, 0, 0, reg, false);
        self.code.push(0xB8 + (reg & 7));
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn mov_abs64(&mut self, dst: Reg) -> u32 {
        let reg = dst as u8;
        self.rex(true, 0, 0, reg, false);
        self.code.push(0xB8 + (reg & 7));
        let at = self.pos();
        self.code.extend(0_u64.to_le_bytes());
        at
    }

    pub fn mov_abs64_val(&mut self, dst: Reg, val: u64) {
        let reg = dst as u8;
        self.rex(true, 0, 0, reg, false);
        self.code.push(0xB8 + (reg & 7));
        self.code.extend(val.to_le_bytes());
    }

    pub fn alu(&mut self, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.alu_flags(op);
        self.encode(&[], true, &[op as u8 * 8 + 3], dst as u8, src.into(), false);
    }

    pub fn alu32(&mut self, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.alu_flags(op);
        self.encode(&[], false, &[op as u8 * 8 + 3], dst as u8, src.into(), false);
    }

    fn alu_imm_width(&mut self, lock: bool, w: bool, op: Alu, dst: Rm, imm: i32) {
        self.alu_flags(op);
        let prefix: &[u8] = if lock { &[0xF0] } else { &[] };
        if let Ok(small) = i8::try_from(imm) {
            self.encode(prefix, w, &[0x83], op as u8, dst, false);
            self.imm(&[small as u8]);
        } else {
            self.encode(prefix, w, &[0x81], op as u8, dst, false);
            self.imm(&imm.to_le_bytes());
        }
    }

    pub fn alu_imm(&mut self, op: Alu, dst: impl Into<Rm>, imm: i32) {
        self.alu_imm_width(false, true, op, dst.into(), imm);
    }

    pub fn alu_imm32(&mut self, op: Alu, dst: impl Into<Rm>, imm: i32) {
        self.alu_imm_width(false, false, op, dst.into(), imm);
    }

    pub(super) fn alu_imm_raw(&mut self, lock: bool, size: Size, op: Alu, dst: Rm, imm: i32) {
        self.alu_flags(op);
        match size {
            Size::B => {
                self.encode(if lock { &[0xF0] } else { &[] }, false, &[0x80], op as u8, dst, dst.byte_rex());
                self.imm(&[imm as u8]);
            }
            Size::W => {
                let prefix: &[u8] = if lock { &[0xF0, 0x66] } else { &[0x66] };
                match i8::try_from(imm as i16) {
                    Ok(small) => {
                        self.encode(prefix, false, &[0x83], op as u8, dst, false);
                        self.imm(&[small as u8]);
                    }
                    Err(_) => {
                        self.encode(prefix, false, &[0x81], op as u8, dst, false);
                        self.imm(&(imm as u16).to_le_bytes());
                    }
                }
            }
            Size::D => self.alu_imm_width(lock, false, op, dst, imm),
            Size::Q => self.alu_imm_width(lock, true, op, dst, imm),
        }
    }

    /// `dst op= imm` on the low `size` bytes of `dst`, with `imm` cut to that width.
    pub fn alu_imm_sized(&mut self, size: Size, op: Alu, dst: impl Into<Rm>, imm: i32) {
        self.alu_imm_raw(false, size, op, dst.into(), imm);
    }

    pub fn test(&mut self, a: Reg, b: Reg) {
        self.flags_overwritten();
        self.encode(&[], true, &[0x85], b as u8, a.into(), false);
    }

    /// `test` of the low `size` bytes of `a` and `b`.
    pub fn test_sized(&mut self, size: Size, a: impl Into<Rm>, b: Reg) {
        self.flags_overwritten();
        self.sized(false, size, (&[0x84], &[0x85]), b as u8, a.into());
    }

    pub fn test_imm8(&mut self, reg: Reg, imm: u8) {
        self.flags_overwritten();
        self.encode(&[], false, &[0xF6], 0, reg.into(), Rm::from(reg).byte_rex());
        self.code.push(imm);
    }

    /// `test` of the low `size` bytes of `rm` with `imm`.
    pub fn test_imm_sized(&mut self, size: Size, rm: impl Into<Rm>, imm: i32) {
        self.flags_overwritten();
        self.sized(false, size, (&[0xF6], &[0xF7]), 0, rm.into());
        self.imm_sized(size, imm);
    }

    pub(super) fn imul_sized(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) { self.encode(&[], size == Size::Q, &[0x0F, 0xAF], dst as u8, src.into(), false); }
    pub(super) fn imul(&mut self, dst: Reg, src: impl Into<Rm>) { self.imul_sized(Size::Q, dst, src); }
    /// `dst = src * imm`.
    pub fn imul_imm_sized(&mut self, size: Size, dst: Reg, src: impl Into<Rm>, imm: i32) {
        self.encode(&[], size == Size::Q, &[0x69], dst as u8, src.into(), false);
        self.imm(&imm.to_le_bytes());
    }
    /// `dst = src * imm`.
    pub fn imul_imm(&mut self, dst: Reg, src: impl Into<Rm>, imm: i32) { self.imul_imm_sized(Size::Q, dst, src, imm); }

    /// `op` on the low `size` bytes of `dst` and `src`.
    pub fn alu_sized(&mut self, size: Size, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.alu_flags(op);
        self.sized(false, size, (&[op as u8 * 8 + 2], &[op as u8 * 8 + 3]), dst as u8, src.into());
    }

    /// `dst op= src` on memory `dst` and register `src` at `size`.
    pub fn alu_mem_reg(&mut self, size: Size, op: Alu, dst: Mem, src: Reg) {
        self.alu_flags(op);
        self.sized(false, size, (&[op as u8 * 8], &[op as u8 * 8 + 1]), src as u8, dst.into());
    }

    /// The one-operand `op` on the low `size` bytes of `rm`.
    pub fn group3_sized(&mut self, size: Size, op: Group3, rm: impl Into<Rm>) { self.sized(false, size, (&[0xF6], &[0xF7]), op as u8, rm.into()); }
    pub(super) fn group3(&mut self, op: Group3, rm: impl Into<Rm>) { self.encode(&[], true, &[0xF7], op as u8, rm.into(), false); }
    pub(super) fn cqo(&mut self) { self.code.extend([0x48, 0x99]); }
    pub(super) fn shift(&mut self, size: Size, op: Shift, reg: Reg) { self.sized(false, size, (&[0xD2], &[0xD3]), op as u8, reg.into()); }

    pub fn shift_imm(&mut self, size: Size, op: Shift, reg: Reg, imm: u8) {
        self.flags_overwritten();
        self.sized(false, size, (&[0xC0], &[0xC1]), op as u8, reg.into());
        self.code.push(imm);
    }

    pub fn shld(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x0F, 0xA5], src as u8, dst.into(), false);
    }

    pub fn shld_imm(&mut self, size: Size, dst: impl Into<Rm>, src: Reg, imm: u8) {
        self.encode(&[], size == Size::Q, &[0x0F, 0xA4], src as u8, dst.into(), false);
        self.imm(&[imm]);
    }

    pub fn shrd(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x0F, 0xAD], src as u8, dst.into(), false);
    }

    pub fn shrd_imm(&mut self, size: Size, dst: impl Into<Rm>, src: Reg, imm: u8) {
        self.encode(&[], size == Size::Q, &[0x0F, 0xAC], src as u8, dst.into(), false);
        self.imm(&[imm]);
    }

    pub fn setcc(&mut self, cond: Cond, reg: Reg) {
        self.encode(&[], false, &[0x0F, 0x90 + cond as u8], 0, reg.into(), Rm::from(reg).byte_rex());
    }

    pub fn cmov(&mut self, cond: Cond, dst: Reg, src: impl Into<Rm>) {
        self.encode(&[], true, &[0x0F, 0x40 + cond as u8], dst as u8, src.into(), false);
    }

    pub fn bswap(&mut self, size: Size, reg: Reg) {
        self.rex(size == Size::Q, 0, 0, reg as u8, false);
        self.code.extend([0x0F, 0xC8 + (reg as u8 & 7)]);
    }

    pub fn count(&mut self, op: Count, size: Size, dst: Reg, src: impl Into<Rm>) {
        self.encode(&[0xF3], size == Size::Q, &[0x0F, op as u8], dst as u8, src.into(), false);
    }

    /// `bsr` when `reverse`, else `bsf`: the opcodes of lzcnt and tzcnt without the F3 prefix; a zero `src` sets ZF and leaves `dst` undefined.
    pub fn bit_scan(&mut self, reverse: bool, dst: Reg, src: impl Into<Rm>) {
        let op = if reverse { Count::Lzcnt } else { Count::Tzcnt };
        self.encode(&[], true, &[0x0F, op as u8], dst as u8, src.into(), false);
    }

    pub fn lea(&mut self, dst: Reg, src: Mem) {
        self.lea_sized(Size::Q, dst, src);
    }

    pub fn lea_sized(&mut self, size: Size, dst: Reg, src: Mem) {
        let start = self.pos();
        self.encode(&[], size == Size::Q, &[0x8D], dst as u8, src.into(), false);
        self.flags_kept(start);
    }

    pub fn sse(&mut self, prefix: &[u8], opcode: &[u8], r: u8, rm: impl Into<Rm>) {
        self.encode(prefix, false, opcode, r, rm.into(), false);
    }

    pub fn sse_wide(&mut self, prefix: u8, opcode: u8, r: u8, rm: impl Into<Rm>) {
        self.encode(&[prefix], true, &[0x0F, opcode], r, rm.into(), false);
    }

    pub fn sse_imm(&mut self, prefix: &[u8], opcode: &[u8], r: u8, rm: impl Into<Rm>, imm: u8) {
        self.encode(prefix, false, opcode, r, rm.into(), false);
        self.imm(&[imm]);
    }

    pub fn sse_wide_imm(&mut self, prefix: &[u8], opcode: &[u8], r: u8, rm: impl Into<Rm>, imm: u8) {
        self.encode(prefix, true, opcode, r, rm.into(), false);
        self.imm(&[imm]);
    }

    fn pshift_opc(lane_bytes: u32) -> u8 {
        match lane_bytes {
            2 => 0x71,
            4 => 0x72,
            _ => 0x73,
        }
    }

    pub fn pshift_imm(&mut self, ext: u8, lane_bytes: u32, dst: Xmm, imm: u8) {
        self.sse_imm(&[0x66], &[0x0F, Self::pshift_opc(lane_bytes)], ext, dst, imm);
    }

    pub fn vex_pshift_imm(&mut self, vl: Vl, ext: u8, lane_bytes: u32, dst: Xmm, src: Xmm, imm: u8) {
        self.vex_imm(vl, VexOp::new(0b00001, 0b01, Self::pshift_opc(lane_bytes)), (Xmm(ext), dst), src, imm);
    }

    pub fn movd_in(&mut self, xmm: Xmm, reg: Reg) {
        self.encode(&[0x66], false, &[0x0F, 0x6E], xmm.0, reg.into(), false);
    }

    pub fn movd_out(&mut self, reg: Reg, xmm: Xmm) {
        self.encode(&[0x66], false, &[0x0F, 0x7E], xmm.0, reg.into(), false);
    }

    fn vex3(&mut self, map: u8, w: bool, pp: u8, opcode: u8, (dst, src): (u8, u8), rm: Rm) {
        let (x, b) = match rm {
            Rm::Reg(reg) => (0, reg),
            Rm::Mem(mem) => (mem.index.map_or(0, |(idx, _)| idx as u8), mem.base as u8),
        };
        self.code.push(0xC4);
        self.code.push((!dst >> 3 & 1) << 7 | (!x >> 3 & 1) << 6 | (!b >> 3 & 1) << 5 | map);
        self.code.push((u8::from(w) << 7) | ((!src & 0xF) << 3) | pp);
        self.code.push(opcode);
        self.modrm(dst, rm);
    }

    /// The three-operand VEX form `dst = src op rm` of the scalar SSE operation `opcode`, where `prefix` is 0xF2 for doubles and 0xF3 for singles.
    pub fn vex_scalar(&mut self, prefix: u8, opcode: u8, (dst, src): (Xmm, Xmm), rm: impl Into<Rm>) {
        let pp = if prefix == 0xF2 { 0b11 } else { 0b10 };
        self.vex3(0b00001, false, pp, opcode, (dst.0, src.0), rm.into());
    }

    /// The three-operand VEX form `dst = src op rm` of `op` at vector length `vl`.
    pub fn vex(&mut self, vl: Vl, op: VexOp, (dst, src): (Xmm, Xmm), rm: impl Into<Rm>) {
        self.vex3(op.map, op.w, op.pp | vl as u8, op.op, (dst.0, src.0), rm.into());
    }

    /// The three-operand VEX immediate form `dst = src op rm, imm` of `op` at vector length `vl`.
    pub fn vex_imm(&mut self, vl: Vl, op: VexOp, regs: (Xmm, Xmm), rm: impl Into<Rm>, imm: u8) {
        self.vex(vl, op, regs, rm);
        self.imm(&[imm]);
    }

    /// The three-operand VEX.128 form `dst = src op rm` with opcode map `map` and prefix class `pp`.
    pub fn vex_xmm(&mut self, map: u8, pp: u8, opcode: u8, regs: (Xmm, Xmm), rm: impl Into<Rm>) {
        self.vex(Vl::X, VexOp::new(map, pp, opcode), regs, rm);
    }

    /// The three-operand VEX.128 immediate form `dst = src op rm, imm`.
    pub fn vex_xmm_imm(&mut self, map: u8, pp: u8, opcode: u8, regs: (Xmm, Xmm), rm: impl Into<Rm>, imm: u8) {
        self.vex_imm(Vl::X, VexOp::new(map, pp, opcode), regs, rm, imm);
    }

    /// `vfmadd213{ss,sd,ps,pd}`: computes `dst = src * dst + rm`.
    pub fn vex_fma(&mut self, vec: bool, f64: bool, (dst, src): (Xmm, Xmm), rm: impl Into<Rm>) {
        let opcode = if vec { 0xA8 } else { 0xA9 };
        self.vex3(0b00010, f64, 0b01, opcode, (dst.0, src.0), rm.into());
    }

    /// `shlx`/`shrx`/`sarx`: non-destructive 3-operand variable shift `dst = src op cnt`.
    pub fn shiftx(&mut self, size: Size, op: Shift, dst: Reg, src: impl Into<Rm>, cnt: Reg) {
        let pp = match op {
            Shift::Shl => 0b01,
            Shift::Sar => 0b10,
            _ => 0b11,
        };
        self.vex3(0b00010, size == Size::Q, pp, 0xF7, (dst as u8, cnt as u8), src.into());
    }

    /// `rorx`: non-destructive 3-operand rotate right `dst = ror(src, imm)`.
    pub fn rorx(&mut self, size: Size, dst: Reg, src: impl Into<Rm>, imm: u8) {
        self.vex3(0b00011, size == Size::Q, 0b11, 0xF0, (dst as u8, 0), src.into());
        self.imm(&[imm]);
    }

    /// `andn`: non-destructive 3-operand `dst = !not_src & src`.
    pub fn andn(&mut self, size: Size, dst: Reg, not_src: Reg, src: impl Into<Rm>) {
        self.vex3(0b00010, size == Size::Q, 0b00, 0xF2, (dst as u8, not_src as u8), src.into());
    }

    /// `bextr`: non-destructive 3-operand bitfield extract `dst = (src >> ctrl[7:0]) & ((1 << ctrl[15:8]) - 1)`.
    pub fn bextr(&mut self, size: Size, dst: Reg, src: impl Into<Rm>, ctrl: Reg) {
        self.vex3(0b00010, size == Size::Q, 0b00, 0xF7, (dst as u8, ctrl as u8), src.into());
    }

    /// `movaps`/`vmovaps dst, src`: copies the whole register, with no dependency on the old contents of `dst`.
    pub fn movaps(&mut self, dst: Xmm, src: Xmm) {
        if self.avx {
            self.vex_xmm(0b00001, 0b00, 0x28, (dst, Xmm(0)), src);
        } else {
            self.sse(&[], &[0x0F, 0x28], dst.0, src);
        }
    }

    /// `movq xmm, reg`.
    pub fn movq_in(&mut self, xmm: Xmm, reg: Reg) {
        self.encode(&[0x66], true, &[0x0F, 0x6E], xmm.0, reg.into(), false);
    }

    /// `movq reg, xmm`.
    pub fn movq_out(&mut self, reg: Reg, xmm: Xmm) {
        self.encode(&[0x66], true, &[0x0F, 0x7E], xmm.0, reg.into(), false);
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
        if direct {
            let at = self.call_pcrel32();
            return StencilReloc::symbol(at, sym, RelocKind::PcRel32);
        }
        let at = self.call_slot_reloc();
        StencilReloc::symbol(at, sym, RelocKind::Abs32)
    }

    /// `call` through the eight bytes at the 32-bit absolute address `at`.
    pub fn call_absolute(&mut self, at: i32) {
        self.flags_overwritten();
        self.code.extend([0xFF, 0x14, 0x25]);
        self.code.extend(at.to_le_bytes());
    }

    pub fn call_slot_reloc(&mut self) -> u32 {
        self.flags_overwritten();
        self.code.extend([0xFF, 0x14, 0x25]);
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn call_pcrel32(&mut self) -> u32 {
        self.flags_overwritten();
        self.code.push(0xE8);
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
        self.fixups.push(Fixup { at: self.pos(), label, disp });
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
        let (wide, op): (bool, &[u8]) = match size {
            Size::B => (false, &[0x0F, 0xB6]),
            Size::W => (false, &[0x0F, 0xB7]),
            Size::D => (false, &[0x8B]),
            Size::Q => (true, &[0x8B]),
        };
        let at = self.rip_head(wide, op, dst);
        self.code.extend([0; 4]);
        at
    }

    pub fn table_entry(&mut self, base: Label, target: Label) {
        self.table_fixups.push((self.pos(), base, target));
        self.code.extend([0; 4]);
    }
}
