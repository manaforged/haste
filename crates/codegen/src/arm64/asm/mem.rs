use super::{Asm, LastMem, Mem, PairKind, Reg, SP, Size, Vreg, ZR, decode_sib, sized};

impl Asm {
    pub fn ldrsw_scaled(&mut self, dst: Reg, base: Reg, index: Reg) {
        self.emit_gp(
            dst,
            0xB8A0_7800 | u32::from(index.0) << 16 | u32::from(base.0) << 5 | u32::from(dst.0),
        );
    }

    fn load_base(size: Size, signed: bool) -> u32 {
        Self::store_base(size)
            | if signed && size != Size::X {
                0x80_0000
            } else {
                0x40_0000
            }
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

    fn pair_match(
        prev: LastMem,
        (kind, reg): (PairKind, u8),
        mem: Mem,
        (pos, binds): (u32, u32),
    ) -> Option<(u8, u8, i32)> {
        if prev.end_pos != pos
            || prev.binds != binds
            || prev.pair != Some(kind)
            || prev.base != mem.base
        {
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
            PairKind::FloatLoad(s) => {
                if s == Size::W {
                    0x2D40_0000
                } else {
                    0x6D40_0000
                }
            }
            PairKind::FloatStore(s) => {
                if s == Size::W {
                    0x2D00_0000
                } else {
                    0x6D00_0000
                }
            }
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
        if (mem.disp >= 0 && mem.disp % unit == 0 && mem.disp / unit < 4096)
            || (-256..256).contains(&mem.disp)
            || mem.base == Reg(17)
        {
            return mem;
        }
        let Some((page, rem)) = Self::page_split(mem.disp, unit) else {
            return mem;
        };
        if self.last_anchor != Some((self.binds, mem.base, page)) {
            self.add_imm(
                page < 0,
                false,
                Size::X,
                Reg(17),
                mem.base,
                page.unsigned_abs() as u32,
            );
            self.last_anchor = Some((self.binds, mem.base, page));
        }
        Mem::new(Reg(17), rem as i32)
    }

    fn access_pair(
        &mut self,
        pair: Option<(PairKind, u8)>,
        is_int: bool,
        base: u32,
        scale: u32,
        rt: u32,
        mem: Mem,
    ) {
        let mem = self.anchor_mem(scale, mem);
        if let Some((kind, reg)) = pair
            && let Some(prev) = self.last_mem
            && let Some((r1, r2, min_disp)) =
                Self::pair_match(prev, (kind, reg), mem, (self.pos(), self.binds))
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
            end_pos: self.pos(),
            binds: self.binds,
            pair: pair.map(|(k, _)| k),
            base_op: base,
            is_int,
            reg: rt as u8,
            base: mem.base,
            disp: mem.disp,
        });
    }

    /// Loads `size` bytes at `mem` into `dst`, sign-extending to 64 bits when `signed`, else zero-extending.
    pub fn load(&mut self, size: Size, signed: bool, dst: Reg, mem: Mem) {
        let pair = (!signed && matches!(size, Size::W | Size::X))
            .then_some((PairKind::IntLoad(size), dst.0));
        self.access_pair(
            pair,
            true,
            Self::load_base(size, signed),
            size.log2(),
            u32::from(dst.0),
            mem,
        );
        self.clobber_anchor(dst);
    }

    pub fn store(&mut self, size: Size, mem: Mem, src: Reg) {
        let pair = matches!(size, Size::W | Size::X).then_some((PairKind::IntStore(size), src.0));
        self.access_pair(
            pair,
            true,
            Self::store_base(size),
            size.log2(),
            u32::from(src.0),
            mem,
        );
    }

    fn vpair(bytes: u32, reg: u8, load: bool) -> Option<(PairKind, u8)> {
        let size = match bytes {
            4 => Size::W,
            8 => Size::X,
            _ => return None,
        };
        Some((
            if load {
                PairKind::FloatLoad(size)
            } else {
                PairKind::FloatStore(size)
            },
            reg,
        ))
    }

    /// Loads a float or vector register of `bytes` 4, 8 or 16.
    pub fn vload(&mut self, bytes: u32, dst: Vreg, mem: Mem) {
        let (base, scale) = Self::vbase(bytes);
        self.access_pair(
            Self::vpair(bytes, dst.0, true),
            false,
            base | 0x40_0000,
            scale,
            u32::from(dst.0),
            mem,
        );
    }

    pub fn vstore(&mut self, bytes: u32, mem: Mem, src: Vreg) {
        let (base, scale) = Self::vbase(bytes);
        self.access_pair(
            Self::vpair(bytes, src.0, false),
            false,
            base,
            scale,
            u32::from(src.0),
            mem,
        );
    }

    /// Loads `size` bytes at `base + (index << shift)` into `dst`.
    pub fn load_indexed(
        &mut self,
        size: Size,
        signed: bool,
        dst: Reg,
        base: Reg,
        index: Reg,
        shift: u8,
    ) {
        self.access_indexed(
            Self::load_base(size, signed),
            u32::from(dst.0),
            base,
            index,
            shift,
        );
        self.clobber_anchor(dst);
    }

    /// Stores `size` bytes from `src` to `base + (index << shift)`.
    pub fn store_indexed(&mut self, size: Size, base: Reg, index: Reg, shift: u8, src: Reg) {
        self.access_indexed(Self::store_base(size), u32::from(src.0), base, index, shift);
    }

    /// Loads a float register of `bytes` 4 or 8 at `base + (index << shift)`.
    pub fn vload_indexed(&mut self, bytes: u32, dst: Vreg, base: Reg, index: Reg, shift: u8) {
        self.access_indexed(
            Self::vbase(bytes).0 | 0x40_0000,
            u32::from(dst.0),
            base,
            index,
            shift,
        );
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
        let (ext, amt) = decode_sib(shift);
        self.word(
            base | 1 << 21
                | u32::from(rm.0) << 16
                | (ext as u32) << 13
                | u32::from(amt > 0) << 12
                | 0b10 << 10
                | u32::from(rn.0) << 5
                | rt,
        );
    }

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
            self.add_imm(
                page < 0,
                false,
                Size::X,
                Reg(17),
                mem.base,
                page.unsigned_abs() as u32,
            );
            self.word(base | 1 << 24 | (rem / (unit as u32)) << 10 | 17 << 5 | rt);
            false
        } else {
            self.mov_imm(Reg(17), i64::from(disp));
            self.access_indexed(base, rt, mem.base, Reg(17), 0);
            false
        }
    }

    pub(in crate::arm64) fn pair_offset(
        &mut self,
        base: u32,
        step: i32,
        a: u8,
        b: u8,
        addr: Reg,
        offset: i32,
    ) {
        self.word(
            base | (((offset / step) as u32) & 0x7F) << 15
                | u32::from(b) << 10
                | u32::from(addr.0) << 5
                | u32::from(a),
        );
    }

    pub fn stp_offset(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.pair_offset(0xA900_0000, 8, a.0, b.0, base, offset);
    }

    pub fn ldp_offset(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.pair_offset(0xA940_0000, 8, a.0, b.0, base, offset);
        self.clobber_anchor(a);
        self.clobber_anchor(b);
    }

    pub fn fstp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) {
        self.pair_offset(0x6D00_0000, 8, a.0, b.0, base, offset);
    }

    pub fn fldp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) {
        self.pair_offset(0x6D40_0000, 8, a.0, b.0, base, offset);
    }

    pub(in crate::arm64) fn qstp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) {
        self.pair_offset(0xAD00_0000, 16, a.0, b.0, base, offset);
    }

    pub(in crate::arm64) fn qldp_offset(&mut self, a: Vreg, b: Vreg, base: Reg, offset: i32) {
        self.pair_offset(0xAD40_0000, 16, a.0, b.0, base, offset);
    }

    pub fn stp_pre(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.word(
            0xA980_0000
                | (((offset / 8) as u32) & 0x7F) << 15
                | u32::from(b.0) << 10
                | u32::from(base.0) << 5
                | u32::from(a.0),
        );
        self.clobber_anchor(base);
    }

    pub fn ldp_post(&mut self, a: Reg, b: Reg, base: Reg, offset: i32) {
        self.word(
            0xA8C0_0000
                | (((offset / 8) as u32) & 0x7F) << 15
                | u32::from(b.0) << 10
                | u32::from(base.0) << 5
                | u32::from(a.0),
        );
        self.clobber_anchor(a);
        self.clobber_anchor(b);
        self.clobber_anchor(base);
    }
}
