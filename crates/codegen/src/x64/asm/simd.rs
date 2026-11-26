use super::{Asm, Reg, Rm, Shift, Size, VexOp, Vl, Xmm};

impl Asm {
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

    pub fn sse_wide_imm(
        &mut self,
        prefix: &[u8],
        opcode: &[u8],
        r: u8,
        rm: impl Into<Rm>,
        imm: u8,
    ) {
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
        self.sse_imm(
            &[0x66],
            &[0x0F, Self::pshift_opc(lane_bytes)],
            ext,
            dst,
            imm,
        );
    }

    pub fn vex_pshift_imm(
        &mut self,
        vl: Vl,
        ext: u8,
        lane_bytes: u32,
        dst: Xmm,
        src: Xmm,
        imm: u8,
    ) {
        self.vex_imm(
            vl,
            VexOp::new(0b00001, 0b01, Self::pshift_opc(lane_bytes)),
            (Xmm(ext), dst),
            src,
            imm,
        );
    }

    pub fn movd_in(&mut self, xmm: Xmm, reg: Reg) {
        self.encode(&[0x66], false, &[0x0F, 0x6E], xmm.0, reg.into(), false);
    }

    pub fn movd_out(&mut self, reg: Reg, xmm: Xmm) {
        self.encode(&[0x66], false, &[0x0F, 0x7E], xmm.0, reg.into(), false);
    }

    fn vex3(&mut self, map: u8, w: bool, pp: u8, opcode: u8, (dst, src): (u8, u8), rm: Rm) {
        let (x, b) = rm.index_base();
        self.code.push(0xC4);
        self.code
            .push((!dst >> 3 & 1) << 7 | (!x >> 3 & 1) << 6 | (!b >> 3 & 1) << 5 | map);
        self.code
            .push((u8::from(w) << 7) | ((!src & 0xF) << 3) | pp);
        self.code.push(opcode);
        self.modrm(dst, rm);
    }

    /// The three-operand VEX form `dst = src op rm` of the scalar SSE operation `opcode`, where `prefix` is 0xF2 for doubles and 0xF3 for singles.
    pub fn vex_scalar(
        &mut self,
        prefix: u8,
        opcode: u8,
        (dst, src): (Xmm, Xmm),
        rm: impl Into<Rm>,
    ) {
        let pp = if prefix == 0xF2 { 0b11 } else { 0b10 };
        self.vex3(0b00001, false, pp, opcode, (dst.0, src.0), rm.into());
    }

    /// The three-operand VEX form `dst = src op rm` of `op` at vector length `vl`.
    pub fn vex(&mut self, vl: Vl, op: VexOp, (dst, src): (Xmm, Xmm), rm: impl Into<Rm>) {
        self.vex3(
            op.map,
            op.w,
            op.pp | vl as u8,
            op.op,
            (dst.0, src.0),
            rm.into(),
        );
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
    pub fn vex_xmm_imm(
        &mut self,
        map: u8,
        pp: u8,
        opcode: u8,
        regs: (Xmm, Xmm),
        rm: impl Into<Rm>,
        imm: u8,
    ) {
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
        self.vex3(
            0b00010,
            size == Size::Q,
            pp,
            0xF7,
            (dst as u8, cnt as u8),
            src.into(),
        );
    }

    /// `rorx`: non-destructive 3-operand rotate right `dst = ror(src, imm)`.
    pub fn rorx(&mut self, size: Size, dst: Reg, src: impl Into<Rm>, imm: u8) {
        self.vex3(
            0b00011,
            size == Size::Q,
            0b11,
            0xF0,
            (dst as u8, 0),
            src.into(),
        );
        self.imm(&[imm]);
    }

    /// `andn`: non-destructive 3-operand `dst = !not_src & src`.
    pub fn andn(&mut self, size: Size, dst: Reg, not_src: Reg, src: impl Into<Rm>) {
        self.vex3(
            0b00010,
            size == Size::Q,
            0b00,
            0xF2,
            (dst as u8, not_src as u8),
            src.into(),
        );
    }

    /// `bextr`: non-destructive 3-operand bitfield extract `dst = (src >> ctrl[7:0]) & ((1 << ctrl[15:8]) - 1)`.
    pub fn bextr(&mut self, size: Size, dst: Reg, src: impl Into<Rm>, ctrl: Reg) {
        self.vex3(
            0b00010,
            size == Size::Q,
            0b00,
            0xF7,
            (dst as u8, ctrl as u8),
            src.into(),
        );
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
}
