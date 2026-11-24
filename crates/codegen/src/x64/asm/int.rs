use super::{Alu, Asm, Cond, Count, Group3, Mem, Reg, Rm, Shift, Size};

impl Asm {
    pub fn load(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) {
        let (w, opcode) = size.load_opcode();
        let (start, src) = (self.pos(), src.into());
        self.encode(
            &[],
            w,
            opcode,
            dst as u8,
            src,
            size == Size::B && src.byte_rex(),
        );
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
        self.encode(
            &[],
            false,
            &[op as u8 * 8 + 3],
            dst as u8,
            src.into(),
            false,
        );
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

    pub(in crate::x64) fn alu_imm_raw(
        &mut self,
        lock: bool,
        size: Size,
        op: Alu,
        dst: Rm,
        imm: i32,
    ) {
        self.alu_flags(op);
        match size {
            Size::B => {
                self.encode(
                    if lock { &[0xF0] } else { &[] },
                    false,
                    &[0x80],
                    op as u8,
                    dst,
                    dst.byte_rex(),
                );
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

    pub(in crate::x64) fn imul_sized(&mut self, size: Size, dst: Reg, src: impl Into<Rm>) {
        self.encode(
            &[],
            size == Size::Q,
            &[0x0F, 0xAF],
            dst as u8,
            src.into(),
            false,
        );
    }

    pub(in crate::x64) fn imul(&mut self, dst: Reg, src: impl Into<Rm>) {
        self.imul_sized(Size::Q, dst, src);
    }

    /// `dst = src * imm`.
    pub fn imul_imm_sized(&mut self, size: Size, dst: Reg, src: impl Into<Rm>, imm: i32) {
        self.encode(&[], size == Size::Q, &[0x69], dst as u8, src.into(), false);
        self.imm(&imm.to_le_bytes());
    }

    /// `dst = src * imm`.
    pub fn imul_imm(&mut self, dst: Reg, src: impl Into<Rm>, imm: i32) {
        self.imul_imm_sized(Size::Q, dst, src, imm);
    }

    /// `op` on the low `size` bytes of `dst` and `src`.
    pub fn alu_sized(&mut self, size: Size, op: Alu, dst: Reg, src: impl Into<Rm>) {
        self.alu_flags(op);
        self.sized(
            false,
            size,
            (&[op as u8 * 8 + 2], &[op as u8 * 8 + 3]),
            dst as u8,
            src.into(),
        );
    }

    /// `dst op= src` on memory `dst` and register `src` at `size`.
    pub fn alu_mem_reg(&mut self, size: Size, op: Alu, dst: Mem, src: Reg) {
        self.alu_flags(op);
        self.sized(
            false,
            size,
            (&[op as u8 * 8], &[op as u8 * 8 + 1]),
            src as u8,
            dst.into(),
        );
    }

    /// The one-operand `op` on the low `size` bytes of `rm`.
    pub fn group3_sized(&mut self, size: Size, op: Group3, rm: impl Into<Rm>) {
        self.sized(false, size, (&[0xF6], &[0xF7]), op as u8, rm.into());
    }

    pub(in crate::x64) fn group3(&mut self, op: Group3, rm: impl Into<Rm>) {
        self.encode(&[], true, &[0xF7], op as u8, rm.into(), false);
    }

    pub(in crate::x64) fn cqo(&mut self) {
        self.code.extend([0x48, 0x99]);
    }

    pub(in crate::x64) fn shift(&mut self, size: Size, op: Shift, reg: Reg) {
        self.sized(false, size, (&[0xD2], &[0xD3]), op as u8, reg.into());
    }

    pub fn shift_imm(&mut self, size: Size, op: Shift, reg: Reg, imm: u8) {
        self.flags_overwritten();
        self.sized(false, size, (&[0xC0], &[0xC1]), op as u8, reg.into());
        self.code.push(imm);
    }

    pub fn shld(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x0F, 0xA5], src as u8, dst.into(), false);
    }

    pub fn shld_imm(&mut self, size: Size, dst: impl Into<Rm>, src: Reg, imm: u8) {
        self.encode(
            &[],
            size == Size::Q,
            &[0x0F, 0xA4],
            src as u8,
            dst.into(),
            false,
        );
        self.imm(&[imm]);
    }

    pub fn shrd(&mut self, dst: Reg, src: Reg) {
        self.encode(&[], true, &[0x0F, 0xAD], src as u8, dst.into(), false);
    }

    pub fn shrd_imm(&mut self, size: Size, dst: impl Into<Rm>, src: Reg, imm: u8) {
        self.encode(
            &[],
            size == Size::Q,
            &[0x0F, 0xAC],
            src as u8,
            dst.into(),
            false,
        );
        self.imm(&[imm]);
    }

    pub fn setcc(&mut self, cond: Cond, reg: Reg) {
        self.encode(
            &[],
            false,
            &[0x0F, 0x90 + cond as u8],
            0,
            reg.into(),
            Rm::from(reg).byte_rex(),
        );
    }

    pub fn cmov(&mut self, cond: Cond, dst: Reg, src: impl Into<Rm>) {
        self.encode(
            &[],
            true,
            &[0x0F, 0x40 + cond as u8],
            dst as u8,
            src.into(),
            false,
        );
    }

    pub fn bswap(&mut self, size: Size, reg: Reg) {
        self.rex(size == Size::Q, 0, 0, reg as u8, false);
        self.code.extend([0x0F, 0xC8 + (reg as u8 & 7)]);
    }

    pub fn count(&mut self, op: Count, size: Size, dst: Reg, src: impl Into<Rm>) {
        self.encode(
            &[0xF3],
            size == Size::Q,
            &[0x0F, op as u8],
            dst as u8,
            src.into(),
            false,
        );
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
}
