use super::{Asm, Cond, Fconv, Fop, Funary, Reg, Vreg};

impl Asm {
    /// The double-precision form sets bit 22 of a single-precision encoding.
    fn precision(base: u32, double: bool) -> u32 {
        base | u32::from(double) << 22
    }

    pub fn fop(&mut self, op: Fop, double: bool, dst: Vreg, a: Vreg, b: Vreg) {
        self.word(
            Self::precision(op as u32, double)
                | u32::from(b.0) << 16
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    pub fn funary(&mut self, op: Funary, double: bool, dst: Vreg, src: Vreg) {
        self.word(Self::precision(op as u32, double) | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn fcmp(&mut self, double: bool, a: Vreg, b: Vreg) {
        self.word(
            Self::precision(0x1E20_2000, double) | u32::from(b.0) << 16 | u32::from(a.0) << 5,
        );
    }

    /// `fcvt` from single to double precision, or back when `to_single`.
    pub fn fcvt(&mut self, to_single: bool, dst: Vreg, src: Vreg) {
        let word = if to_single { 0x1E62_4000 } else { 0x1E22_C000 };
        self.word(word | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    /// Converts between the integer register and the float register of a conversion, 64-bit integers when `wide`.
    pub fn fconv(&mut self, op: Fconv, double: bool, wide: bool, int: Reg, float: Vreg) {
        let base = Self::precision(op as u32, double) | u32::from(wide) << 31;
        match op {
            Fconv::Scvtf | Fconv::Ucvtf => {
                self.word(base | u32::from(int.0) << 5 | u32::from(float.0))
            }
            Fconv::Fcvtzs | Fconv::Fcvtzu => {
                self.emit_gp(int, base | u32::from(float.0) << 5 | u32::from(int.0))
            }
        }
    }

    pub fn fmov_reg(&mut self, _double: bool, dst: Vreg, src: Vreg) {
        self.vmov(dst, src);
    }

    pub fn fcsel(&mut self, cond: Cond, double: bool, dst: Vreg, a: Vreg, b: Vreg) {
        self.word(
            Self::precision(0x1E20_0C00, double)
                | u32::from(b.0) << 16
                | (cond as u32) << 12
                | u32::from(a.0) << 5
                | u32::from(dst.0),
        );
    }

    pub fn fmov_in(&mut self, double: bool, dst: Vreg, src: Reg) {
        let base = Self::precision(0x1E27_0000, double) | u32::from(double) << 31;
        self.word(base | u32::from(src.0) << 5 | u32::from(dst.0));
    }

    pub fn fmov_out(&mut self, double: bool, dst: Reg, src: Vreg) {
        let base = Self::precision(0x1E26_0000, double) | u32::from(double) << 31;
        self.emit_gp(dst, base | u32::from(src.0) << 5 | u32::from(dst.0));
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
        self.emit_gp(dst, 0x4E18_3C00 | u32::from(src.0) << 5 | u32::from(dst.0));
    }
}
