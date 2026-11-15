use super::asm::{Alu, Asm, Mem, Reg, Rm, Size, Xmm};

/// The last instruction a later one may rewrite in place, valid while nothing was emitted, bound or recorded after it.
#[derive(Clone, Copy)]
pub(crate) struct Tail {
    start: u32,
    end: u32,
    marks: [usize; 2],
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    ZeroReg(Reg),
    ZeroRun { run: Mem, len: i32, last: Last },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Last {
    Imm,
    Movq,
    Movups,
}

impl Asm {
    fn marks(&self) -> [usize; 2] {
        [self.fixups.len(), self.table_fixups.len()]
    }

    fn take_tail(&mut self) -> Option<Tail> {
        let tail = self.tail.take()?;
        (tail.end == self.pos() && tail.marks == self.marks()).then_some(tail)
    }

    fn set_tail(&mut self, start: u32, kind: Kind) {
        self.tail = Some(Tail { start, end: self.pos(), marks: self.marks(), kind });
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

    /// Loads `imm` into `dst` without touching the flags; a zero becomes `xor` once the next instruction overwrites them.
    pub fn mov_imm(&mut self, dst: Reg, imm: i64) {
        let start = self.pos();
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
        if imm == 0 {
            self.set_tail(start, Kind::ZeroReg(dst));
        }
    }

    /// Marks that the next instruction writes every status flag and reads none, so a `mov r32, 0` just before it becomes `xor r32, r32`.
    pub fn flags_overwritten(&mut self) {
        if let Some(Tail { start, kind: Kind::ZeroReg(reg), .. }) = self.take_tail() {
            self.code.truncate(start as usize);
            self.encode(&[], false, &[0x33], reg as u8, reg.into(), false);
        }
    }

    /// Marks the flags overwritten before `op`, unless `op` reads the carry.
    pub fn alu_flags(&mut self, op: Alu) {
        if !matches!(op, Alu::Adc | Alu::Sbb) {
            self.flags_overwritten();
        }
    }

    fn xmm_op(&mut self, pp: u8, opcode: u8, (xmm, src): (Xmm, Xmm), rm: impl Into<Rm>) {
        if self.avx {
            return self.vex_xmm(1, pp, opcode, (xmm, src), rm);
        }
        let prefix: &[u8] = if pp == 1 { &[0x66] } else { &[] };
        self.sse(prefix, &[0x0F, opcode], xmm.0, rm);
    }

    /// Stores eight zero bytes to `dst`, joining adjacent zero stores into `movups` of `scratch` zeroed once by `xorps`; returns whether it wrote `scratch`.
    pub fn store_zero(&mut self, dst: Mem, scratch: Xmm) -> bool {
        let start = self.pos();
        let joined = self.take_tail().and_then(|tail| match tail.kind {
            Kind::ZeroRun { run, len, last }
                if run.base == dst.base && run.index == dst.index && run.disp.checked_add(len) == Some(dst.disp) =>
            {
                Some((tail.start, run, len, last))
            }
            _ => None,
        });
        let Some((last_start, run, len, last)) = joined else {
            self.store_imm_sized(Size::Q, dst, 0);
            self.set_tail(start, Kind::ZeroRun { run: dst, len: 8, last: Last::Imm });
            return false;
        };
        if last != Last::Movups {
            self.code.truncate(last_start as usize);
        }
        if last == Last::Imm {
            self.xmm_op(0, 0x57, (scratch, scratch), scratch);
        }
        let at = self.pos();
        let next = if last == Last::Movups {
            self.xmm_op(1, 0xD6, (scratch, Xmm(0)), dst);
            Last::Movq
        } else {
            self.xmm_op(0, 0x11, (scratch, Xmm(0)), dst.offset(-8));
            Last::Movups
        };
        self.set_tail(at, Kind::ZeroRun { run, len: len + 8, last: next });
        last == Last::Imm
    }
}
