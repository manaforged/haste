use super::asm::{Mem, Reg, Rm, Size};
use super::{Emit, Ext};
use crate::alloc::Location;
use crate::ir::{Type, Value};

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    crate::fold::extend(imm, ty, matches!(ext, Ext::Sign))
}

impl Emit<'_> {
    fn location(&self, value: Value) -> Location<Reg> {
        self.plan.location(value)
    }

    /// Whether `value` needs no code where it is defined, because its location is a constant or a frame address.
    pub(super) fn is_static(&self, value: Value) -> bool {
        self.plan.is_static(value)
    }

    /// The constant `value` holds, when its location is one.
    pub(super) fn const_of(&self, value: Value) -> Option<i64> {
        self.plan.const_of(value)
    }

    pub(super) fn imm_of(&self, value: Value) -> Option<i32> {
        match self.location(value) {
            Location::Imm(imm) => i32::try_from(imm).ok(),
            _ => None,
        }
    }

    /// The constant `value` extended to 64 bits by `ext`, when it fits a sign-extended 32-bit immediate.
    pub(super) fn imm_extended(&self, value: Value, ext: Ext) -> Option<i32> {
        match self.location(value) {
            Location::Imm(imm) => i32::try_from(extend(imm, self.ty(value), ext)).ok(),
            _ => None,
        }
    }

    /// The register that holds `value`, when its location is one.
    pub(super) fn reg_of(&self, value: Value) -> Option<Reg> {
        self.plan.reg_of(value)
    }

    /// The register to compute `value` in: its location register, or rax.
    pub(super) fn dst(&self, value: Value) -> Reg {
        self.reg_of(value).unwrap_or(Reg::Rax)
    }

    /// The memory at `addr + offset`: relative to rbp for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return Mem::new(Reg::Rbp, disp);
        }
        self.raw(Reg::Rcx, addr);
        Mem::new(Reg::Rcx, offset)
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let source: Rm = match self.location(value) {
            Location::Imm(imm) => return self.asm.mov_imm(reg, extend(imm, ty, ext)),
            Location::Frame(..) => return self.raw(reg, value),
            Location::Reg(from) => from.into(),
            Location::Slot => self.at(value).into(),
        };
        match ext {
            _ if bytes >= 8 => self.asm.load(Size::Q, reg, source),
            Ext::Zero => self.asm.load(Size::of(bytes), reg, source),
            Ext::Sign => self.asm.load_signed(Size::of(bytes), reg, source),
        }
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Location::Reg(from) if from == reg => {}
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::Q, reg, at);
            }
        }
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        if let Location::Reg(location) = self.location(value) {
            if location != reg {
                self.asm.mov(location, reg);
            }
            return;
        }
        let at = self.at(value);
        self.asm.store(Size::Q, at, reg);
    }

    /// A register or memory operand that holds `value`, using r11 for a frame address.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Imm(_) => {
                let at = self.at(value);
                self.move_to(at, value);
                at.into()
            }
            Location::Frame(..) => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
            Location::Slot => self.at(value).into(),
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax or r11 as scratch.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.location(value) {
            Location::Imm(imm) => match i32::try_from(imm) {
                Ok(small) => self.asm.store_imm(to, small),
                Err(_) => {
                    self.asm.mov_imm(Reg::R11, imm);
                    self.asm.store(Size::Q, to, Reg::R11);
                }
            },
            Location::Frame(..) => {
                self.raw(Reg::R11, value);
                self.asm.store(Size::Q, to, Reg::R11);
            }
            Location::Reg(reg) => self.asm.store(Size::Q, to, reg),
            Location::Slot => self.copy_to(to, self.at(value), self.ty(value)),
        }
    }
}
