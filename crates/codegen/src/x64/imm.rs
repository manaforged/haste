use super::asm::{Alu, Asm, Mem, Reg, Rm, Size, Xmm};

/// The most zero movs a tail holds before the oldest stay movs.
const PENDING_ZEROS: usize = 8;

/// The trailing instructions a later one may rewrite in place, valid while nothing was emitted, bound or recorded after them.
#[derive(Clone, Copy)]
pub(crate) struct Tail {
    end: u32,
    marks: [usize; 3],
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    Zeros { movs: [Zero; PENDING_ZEROS], len: usize },
    ZeroRun { start: u32, run: Mem, len: i32, last: Last },
    Carry { tag: usize },
    Eq { a: u32, b: u32 },
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
    fn marks(&self) -> [usize; 3] {
        [self.fixups.len(), self.table_fixups.len(), self.rips.len()]
    }

    /// Appends the immediate `bytes` of the instruction just encoded, moving a `rip` displacement that ends here to count from the instruction's end.
    pub(super) fn imm(&mut self, bytes: &[u8]) {
        if let Some(&(at, _)) = self.rips.last()
            && at + 4 == self.pos()
        {
            let field = &mut self.code[at as usize..at as usize + 4];
            let addend = i32::from_le_bytes([field[0], field[1], field[2], field[3]]) - bytes.len() as i32;
            field.copy_from_slice(&addend.to_le_bytes());
        }
        self.code.extend_from_slice(bytes);
    }

    /// Appends `imm` cut to `size`, a quadword taking 32 bits sign-extended.
    pub(super) fn imm_sized(&mut self, size: Size, imm: i32) {
        match size {
            Size::B => self.imm(&[imm as u8]),
            Size::W => self.imm(&(imm as i16).to_le_bytes()),
            _ => self.imm(&imm.to_le_bytes()),
        }
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

    /// Carries pending zero movs, or a held carry or equality flag, past the one instruction emitted since `start`, which neither reads nor writes the flags.
    pub(super) fn flags_kept(&mut self, start: u32) {
        let end = self.pos();
        let marks = self.marks();
        if let Some(tail) = self.tail.as_mut().filter(|tail| {
            tail.end == start && tail.marks == marks && matches!(tail.kind, Kind::Zeros { .. } | Kind::Carry { .. } | Kind::Eq { .. })
        }) {
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

    /// Notes that the zero flag holds `a == b` here, until an instruction that may write the flags.
    pub fn hold_eq(&mut self, a: u32, b: u32) {
        self.set_tail(Kind::Eq { a, b });
    }

    /// The value pair the zero flag still compares here, if only flag-neutral instructions followed `hold_eq`.
    pub fn held_eq(&self) -> Option<(u32, u32)> {
        self.tail.filter(|tail| tail.end == self.pos() && tail.marks == self.marks()).and_then(|tail| match tail.kind {
            Kind::Eq { a, b } => Some((a, b)),
            _ => None,
        })
    }

    fn push_zero(&mut self, zero: Zero) {
        let marks = self.marks();
        if self.tail.is_some_and(|tail| tail.end == zero.start && tail.marks == marks && matches!(tail.kind, Kind::Carry { .. } | Kind::Eq { .. })) {
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
        self.imm_sized(size, imm);
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

    /// Atomic `xchg [dst], src` of width `size`.
    pub fn xchg(&mut self, size: Size, dst: Mem, src: Reg) {
        self.sized(false, size, (&[0x86], &[0x87]), src as u8, dst.into());
    }

    /// Atomic `lock xadd [dst], src` of width `size`.
    pub fn xadd(&mut self, size: Size, dst: Mem, src: Reg) {
        self.flags_overwritten();
        self.sized(true, size, (&[0x0F, 0xC0], &[0x0F, 0xC1]), src as u8, dst.into());
    }

    /// Atomic `lock cmpxchg [dst], src` of width `size`.
    pub fn cmpxchg(&mut self, size: Size, dst: Mem, src: Reg) {
        self.flags_overwritten();
        self.sized(true, size, (&[0x0F, 0xB0], &[0x0F, 0xB1]), src as u8, dst.into());
    }

    /// Atomic `lock op [dst], src` of width `size`.
    pub fn lock_alu_reg(&mut self, size: Size, op: Alu, dst: Mem, src: Reg) {
        self.alu_flags(op);
        self.sized(true, size, (&[op as u8 * 8], &[op as u8 * 8 + 1]), src as u8, dst.into());
    }

    /// Atomic `lock op [dst], imm` of width `size`.
    pub fn lock_alu_imm(&mut self, size: Size, op: Alu, dst: Mem, imm: i32) {
        self.alu_imm_raw(true, size, op, dst.into(), imm);
    }

    /// Atomic `lock inc/dec [dst]` of width `size`.
    pub fn lock_inc_dec(&mut self, size: Size, dec: bool, dst: Mem) {
        self.flags_overwritten();
        self.sized(true, size, (&[0xFE], &[0xFF]), u8::from(dec), dst.into());
    }

    /// Atomic `lock bts/btr/btc [dst], bit` (`ext` = 5/6/7) for `size != Size::B`.
    pub fn lock_bt_imm(&mut self, size: Size, ext: u8, dst: Mem, bit: u8) {
        debug_assert!(size != Size::B);
        self.flags_overwritten();
        self.sized(true, size, (&[], &[0x0F, 0xBA]), ext, dst.into());
        self.imm(&[bit]);
    }

    /// `bt rm, bit` on a 64-bit operand, copying bit `bit` into the carry flag.
    pub fn bt_imm(&mut self, rm: impl Into<Rm>, bit: u8) {
        self.flags_overwritten();
        self.encode(&[], true, &[0x0F, 0xBA], 4, rm.into(), false);
        self.imm(&[bit]);
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
        if dst.rip.is_some() {
            self.store_imm_sized(Size::Q, dst, 0);
            return false;
        }
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
