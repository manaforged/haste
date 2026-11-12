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

#[derive(Clone, Copy, Debug)]
pub(crate) struct Mem {
    pub base: Reg,
    pub index: Option<(Reg, u8)>,
    pub disp: i32,
}

impl Mem {
    pub fn new(base: Reg, disp: i32) -> Self {
        Self { base, index: None, disp }
    }

    pub fn sib(base: Reg, index: Reg, shift: u8, disp: i32) -> Self {
        debug_assert!(index != Reg::Rsp && shift <= 3);
        Self { base, index: Some((index, shift)), disp }
    }

    pub fn offset(self, by: i32) -> Self {
        Self { base: self.base, index: self.index, disp: self.disp + by }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Rm {
    Reg(u8),
    Mem(Mem),
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

#[derive(Clone, Copy)]
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
    labels: Vec<Option<u32>>,
    fixups: Vec<(u32, Label)>,
}

impl Asm {
    pub fn pos(&self) -> u32 {
        self.code.len() as u32
    }

    fn rex(&mut self, w: bool, r: u8, x: u8, b: u8, force: bool) {
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

    fn sized(&mut self, lock: bool, size: Size, (byte, wide): (&[u8], &[u8]), r: u8, rm: Rm) {
        let prefix: &[u8] = match (lock, size == Size::W) {
            (true, true) => &[0xF0, 0x66],
            (true, false) => &[0xF0],
            (false, true) => &[0x66],
            (false, false) => &[],
        };
        let opcode = if size == Size::B { byte } else { wide };
        self.encode(prefix, size == Size::Q, opcode, r, rm, size == Size::B);
    }

    pub fn load(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) {
        let (w, opcode): (bool, &[u8]) = match size {
            Size::B => (false, &[0x0F, 0xB6]),
            Size::W => (false, &[0x0F, 0xB7]),
            Size::D => (false, &[0x8B]),
            Size::Q => (true, &[0x8B]),
        };
        self.encode(&[], w, opcode, dst as u8, src.into(), size == Size::B);
    }

    pub fn load_signed(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) {
        let opcode: &[u8] = match size {
            Size::B => &[0x0F, 0xBE],
            Size::W => &[0x0F, 0xBF],
            Size::D => &[0x63],
            Size::Q => &[0x8B],
        };
        self.encode(&[], true, opcode, dst as u8, src.into(), size == Size::B);
    }

    pub fn store(&mut self, size: Size, dst: Mem, src: Reg) {
        self.sized(false, size, (&[0x88], &[0x89]), src as u8, dst.into());
    }

    pub fn store_imm(&mut self, dst: Mem, imm: i32) {
        self.encode(&[], true, &[0xC7], 0, dst.into(), false);
        self.code.extend(imm.to_le_bytes());
    }

    /// Stores the low `size` bytes of `imm` to `dst`; a quadword takes `imm` sign-extended.
    pub fn store_imm_sized(&mut self, size: Size, dst: Mem, imm: i32) {
        self.sized(false, size, (&[0xC6], &[0xC7]), 0, dst.into());
        match size {
            Size::B => self.code.push(imm as u8),
            Size::W => self.code.extend((imm as i16).to_le_bytes()),
            _ => self.code.extend(imm.to_le_bytes()),
        }
    }

    pub fn store_slot_reloc(&mut self, dst: Mem) -> u32 {
        self.encode(&[], true, &[0xC7], 0, dst.into(), false);
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn mov(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x8B], dst as u8, src.into(), false);
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

    pub fn mov_imm(&mut self, dst: Reg, imm: i64) {
        let reg = dst as u8;
        if let Ok(small) = u32::try_from(imm) {
            self.rex(false, 0, 0, reg, false);
            self.code.push(0xB8 + (reg & 7));
            self.code.extend(small.to_le_bytes());
        } else if let Ok(signed) = i32::try_from(imm) {
            self.encode(&[], true, &[0xC7], 0, dst.into(), false);
            self.code.extend(signed.to_le_bytes());
        } else {
            self.rex(true, 0, 0, reg, false);
            self.code.push(0xB8 + (reg & 7));
            self.code.extend(imm.to_le_bytes());
        }
    }

    pub fn alu(&mut self, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.encode(&[], true, &[op as u8 * 8 + 3], dst as u8, src.into(), false);
    }

    pub fn alu32(&mut self, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.encode(&[], false, &[op as u8 * 8 + 3], dst as u8, src.into(), false);
    }

    fn alu_imm_width(&mut self, w: bool, op: Alu, dst: Reg, imm: i32) {
        if let Ok(small) = i8::try_from(imm) {
            self.encode(&[], w, &[0x83], op as u8, dst.into(), false);
            self.code.push(small as u8);
        } else {
            self.encode(&[], w, &[0x81], op as u8, dst.into(), false);
            self.code.extend(imm.to_le_bytes());
        }
    }

    pub fn alu_imm(&mut self, op: Alu, dst: Reg, imm: i32) {
        self.alu_imm_width(true, op, dst, imm);
    }

    pub fn alu_imm32(&mut self, op: Alu, dst: Reg, imm: i32) {
        self.alu_imm_width(false, op, dst, imm);
    }

    /// `dst op= imm` on the low `size` bytes of `dst`, with `imm` cut to that width.
    pub fn alu_imm_sized(&mut self, size: Size, op: Alu, dst: Reg, imm: i32) {
        match size {
            Size::B => {
                self.encode(&[], false, &[0x80], op as u8, dst.into(), true);
                self.code.push(imm as u8);
            }
            Size::W => match i8::try_from(imm as i16) {
                Ok(small) => {
                    self.encode(&[0x66], false, &[0x83], op as u8, dst.into(), false);
                    self.code.push(small as u8);
                }
                Err(_) => {
                    self.encode(&[0x66], false, &[0x81], op as u8, dst.into(), false);
                    self.code.extend((imm as u16).to_le_bytes());
                }
            },
            Size::D => self.alu_imm32(op, dst, imm),
            Size::Q => self.alu_imm(op, dst, imm),
        }
    }

    pub fn test(&mut self, a: Reg, b: Reg) {
        self.encode(&[], true, &[0x85], b as u8, a.into(), false);
    }

    /// `test` of the low `size` bytes of `a` and `b`.
    pub fn test_sized(&mut self, size: Size, a: Reg, b: Reg) {
        self.sized(false, size, (&[0x84], &[0x85]), b as u8, a.into());
    }

    pub fn test_imm8(&mut self, reg: Reg, imm: u8) {
        self.encode(&[], false, &[0xF6], 0, reg.into(), true);
        self.code.push(imm);
    }

    pub fn imul(&mut self, dst: Reg, src: impl Into<Rm>) {
        self.encode(&[], true, &[0x0F, 0xAF], dst as u8, src.into(), false);
    }

    /// `dst = src * imm`.
    pub fn imul_imm(&mut self, dst: Reg, src: impl Into<Rm>, imm: i32) {
        self.encode(&[], true, &[0x69], dst as u8, src.into(), false);
        self.code.extend(imm.to_le_bytes());
    }

    /// `op` on the low `size` bytes of `dst` and `src`.
    pub fn alu_sized(&mut self, size: Size, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.sized(false, size, (&[op as u8 * 8 + 2], &[op as u8 * 8 + 3]), dst as u8, src.into());
    }

    /// The one-operand `op` on the low `size` bytes of `rm`.
    pub fn group3_sized(&mut self, size: Size, op: Group3, rm: impl Into<Rm>) {
        self.sized(false, size, (&[0xF6], &[0xF7]), op as u8, rm.into());
    }

    pub fn group3(&mut self, op: Group3, rm: impl Into<Rm>) {
        self.encode(&[], true, &[0xF7], op as u8, rm.into(), false);
    }

    pub fn cqo(&mut self) {
        self.code.extend([0x48, 0x99]);
    }

    pub fn shift(&mut self, size: Size, op: Shift, reg: Reg) {
        self.sized(false, size, (&[0xD2], &[0xD3]), op as u8, reg.into());
    }

    pub fn shift_imm(&mut self, size: Size, op: Shift, reg: Reg, imm: u8) {
        self.sized(false, size, (&[0xC0], &[0xC1]), op as u8, reg.into());
        self.code.push(imm);
    }

    pub fn shld(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x0F, 0xA5], src as u8, dst.into(), false);
    }

    pub fn shrd(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x0F, 0xAD], src as u8, dst.into(), false);
    }

    pub fn bts(&mut self, reg: Reg, bit: u8) {
        self.encode(&[], true, &[0x0F, 0xBA], 5, reg.into(), false);
        self.code.push(bit);
    }

    pub fn setcc(&mut self, cond: Cond, reg: Reg) {
        self.encode(&[], false, &[0x0F, 0x90 + cond as u8], 0, reg.into(), true);
    }

    pub fn cmov(&mut self, cond: Cond, dst: Reg, src: impl Into<Rm>) {
        self.encode(&[], true, &[0x0F, 0x40 + cond as u8], dst as u8, src.into(), false);
    }

    pub fn bswap(&mut self, size: Size, reg: Reg) {
        self.rex(size == Size::Q, 0, 0, reg as u8, false);
        self.code.extend([0x0F, 0xC8 + (reg as u8 & 7)]);
    }

    pub fn count(&mut self, op: Count, dst: Reg, src: Reg) {
        self.encode(&[0xF3], true, &[0x0F, op as u8], dst as u8, src.into(), false);
    }

    /// `bsr` when `reverse`, else `bsf`: the opcodes of lzcnt and tzcnt without the F3 prefix; a zero `src` sets ZF and leaves `dst` undefined.
    pub fn bit_scan(&mut self, reverse: bool, dst: Reg, src: Reg) {
        let op = if reverse { Count::Lzcnt } else { Count::Tzcnt };
        self.encode(&[], true, &[0x0F, op as u8], dst as u8, src.into(), false);
    }

    pub fn lea(&mut self, dst: Reg, src: Mem) {
        self.encode(&[], true, &[0x8D], dst as u8, src.into(), false);
    }

    pub fn xchg(&mut self, size: Size, dst: Mem, src: Reg) {
        self.sized(false, size, (&[0x86], &[0x87]), src as u8, dst.into());
    }

    pub fn xadd(&mut self, size: Size, dst: Mem, src: Reg) {
        self.sized(true, size, (&[0x0F, 0xC0], &[0x0F, 0xC1]), src as u8, dst.into());
    }

    pub fn cmpxchg(&mut self, size: Size, dst: Mem, src: Reg) {
        self.sized(true, size, (&[0x0F, 0xB0], &[0x0F, 0xB1]), src as u8, dst.into());
    }

    pub fn sse(&mut self, prefix: &[u8], opcode: &[u8], r: u8, rm: impl Into<Rm>) {
        self.encode(prefix, false, opcode, r, rm.into(), false);
    }

    pub fn sse_wide(&mut self, prefix: u8, opcode: u8, r: u8, rm: impl Into<Rm>) {
        self.encode(&[prefix], true, &[0x0F, opcode], r, rm.into(), false);
    }

    /// The three-operand VEX form `dst = src op rm` of the scalar SSE operation `opcode`, where `prefix` is 0xF2 for doubles and 0xF3 for singles.
    pub fn vex_scalar(&mut self, prefix: u8, opcode: u8, (dst, src): (Xmm, Xmm), rm: impl Into<Rm>) {
        let rm = rm.into();
        let (x, b) = match rm {
            Rm::Reg(reg) => (0, reg),
            Rm::Mem(mem) => (mem.index.map_or(0, |(idx, _)| idx as u8), mem.base as u8),
        };
        let pp = if prefix == 0xF2 { 0b11 } else { 0b10 };
        self.code.push(0xC4);
        self.code.push((!dst.0 >> 3 & 1) << 7 | (!x >> 3 & 1) << 6 | (!b >> 3 & 1) << 5 | 0b00001);
        self.code.push((!src.0 & 0xF) << 3 | pp);
        self.code.push(opcode);
        self.modrm(dst.0, rm);
    }

    /// `movaps dst, src`: copies the whole register, with no dependency on the old contents of `dst`.
    pub fn movaps(&mut self, dst: Xmm, src: Xmm) {
        self.sse(&[], &[0x0F, 0x28], dst.0, src);
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

    /// `call` through the eight bytes at the 32-bit absolute address `at`.
    pub fn call_absolute(&mut self, at: i32) {
        self.code.extend([0xFF, 0x14, 0x25]);
        self.code.extend(at.to_le_bytes());
    }

    pub fn call_slot_reloc(&mut self) -> u32 {
        self.code.extend([0xFF, 0x14, 0x25]);
        let at = self.pos();
        self.code.extend(0_u32.to_le_bytes());
        at
    }

    pub fn call(&mut self, reg: Reg) {
        self.encode(&[], false, &[0xFF], 2, reg.into(), false);
    }

    pub fn jmp_reg(&mut self, reg: Reg) {
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
        self.labels[label.0 as usize] = Some(self.pos());
    }

    fn fixup(&mut self, label: Label) {
        self.fixups.push((self.pos(), label));
        self.code.extend([0; 4]);
    }

    pub fn jmp(&mut self, label: Label) {
        self.code.push(0xE9);
        self.fixup(label);
    }

    pub fn jcc(&mut self, cond: Cond, label: Label) {
        self.code.extend([0x0F, 0x80 + cond as u8]);
        self.fixup(label);
    }

    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        for (at, label) in std::mem::take(&mut self.fixups) {
            let target = self.labels[label.0 as usize].ok_or("a jump to a label that was never placed")?;
            let relative = target as i64 - (at as i64 + 4);
            let at = at as usize;
            self.code[at..at + 4].copy_from_slice(&(relative as i32).to_le_bytes());
        }
        Ok(self.code)
    }
}
