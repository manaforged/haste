use super::asm::{Alu, Asm, Mem, Reg, Rm, Size, Xmm};

/// The most zero movs a tail holds before the oldest stay movs.
const PENDING_ZEROS: usize = 8;

/// The trailing instructions a later one may rewrite in place, valid while nothing was emitted, bound or recorded after them.
#[derive(Clone, Copy)]
pub(crate) struct Tail {
    end: u32,
    marks: [usize; 2],
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    Zeros { movs: [Zero; PENDING_ZEROS], len: usize },
    ZeroRun { start: u32, run: Mem, len: i32, last: Last },
    Carry { tag: usize },
}

/// A `mov r32, 0` at `start..end` that becomes `xor r32, r32` once the flags are dead.
#[derive(Clone, Copy)]
struct Zero {
    start: u32,
    end: u32,
    reg: Reg,
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

    fn set_tail(&mut self, kind: Kind) {
        self.tail = Some(Tail { end: self.pos(), marks: self.marks(), kind });
    }

    fn zeros_at(&mut self, at: u32) -> Option<&mut Tail> {
        let marks = self.marks();
        self.tail.as_mut().filter(|tail| tail.end == at && tail.marks == marks && matches!(tail.kind, Kind::Zeros { .. }))
    }

    /// Carries pending zero movs, or a held carry, past the one instruction emitted since `start`, which neither reads nor writes the flags.
    pub(super) fn flags_kept(&mut self, start: u32) {
        let end = self.pos();
        let marks = self.marks();
        if let Some(tail) = self.tail.as_mut().filter(|tail| tail.end == start && tail.marks == marks && matches!(tail.kind, Kind::Zeros { .. } | Kind::Carry { .. })) {
            tail.end = end;
        }
    }

    /// Notes that the carry flag holds the carry named `tag` here, until an instruction that may write the flags.
    pub fn hold_carry(&mut self, tag: usize) {
        self.set_tail(Kind::Carry { tag });
    }

    /// The carry the carry flag still holds here, if only flag-neutral instructions followed `hold_carry`.
    pub fn held_carry(&self) -> Option<usize> {
        self.tail.filter(|tail| tail.end == self.pos() && tail.marks == self.marks()).and_then(|tail| match tail.kind {
            Kind::Carry { tag } => Some(tag),
            _ => None,
        })
    }

    fn push_zero(&mut self, zero: Zero) {
        let marks = self.marks();
        if self.tail.is_some_and(|tail| tail.end == zero.start && tail.marks == marks && matches!(tail.kind, Kind::Carry { .. })) {
            self.flags_kept(zero.start);
            return;
        }
        if let Some(Tail { end, kind: Kind::Zeros { movs, len }, .. }) = self.zeros_at(zero.start)
            && *len < PENDING_ZEROS
        {
            movs[*len] = zero;
            *len += 1;
            *end = zero.end;
            return;
        }
        self.set_tail(Kind::Zeros { movs: [zero; PENDING_ZEROS], len: 1 });
    }

    /// Stores the low `size` bytes of `imm` to `dst`; a quadword takes `imm` sign-extended.
    pub fn store_imm_sized(&mut self, size: Size, dst: Mem, imm: i32) {
        let start = self.pos();
        self.sized(false, size, (&[0xC6], &[0xC7]), 0, dst.into());
        match size {
            Size::B => self.code.push(imm as u8),
            Size::W => self.code.extend((imm as i16).to_le_bytes()),
            _ => self.code.extend(imm.to_le_bytes()),
        }
        self.flags_kept(start);
    }

    /// Loads `imm` into `dst` without touching the flags; a zero becomes `xor` once the flags are known dead before any read.
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
            self.push_zero(Zero { start, end: self.pos(), reg: dst });
        } else {
            self.flags_kept(start);
        }
    }

    /// Marks the status flags dead here, read by nothing before their next write, so each pending `mov r32, 0` becomes `xor r32, r32`.
    pub fn flags_overwritten(&mut self) {
        let Some(Tail { kind: Kind::Zeros { movs, len }, .. }) = self.take_tail() else { return };
        for zero in movs[..len].iter().rev() {
            let at = self.pos();
            self.alu32(Alu::Xor, zero.reg, zero.reg);
            let xor = (self.pos() - at) as usize;
            self.code[zero.start as usize..].rotate_right(xor);
            self.code.drain(zero.start as usize + xor..zero.end as usize + xor);
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
            Kind::ZeroRun { start, run, len, last }
                if run.base == dst.base && run.index == dst.index && run.disp.checked_add(len) == Some(dst.disp) =>
            {
                Some((start, run, len, last))
            }
            _ => None,
        });
        let Some((last_start, run, len, last)) = joined else {
            self.store_imm_sized(Size::Q, dst, 0);
            self.set_tail(Kind::ZeroRun { start, run: dst, len: 8, last: Last::Imm });
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
        self.set_tail(Kind::ZeroRun { start: at, run, len: len + 8, last: next });
        last == Last::Imm
    }
}
