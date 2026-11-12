use super::asm::{Mem, Reg, Rm, Size, Xmm};
use super::float::scalar;
use super::{Emit, Ext};
use crate::alloc::Location;
use crate::ir::{Type, Value};
use crate::reloc::{RelocKind, StencilReloc};

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

    /// The float register that holds `value`, when its location is one.
    pub(super) fn xmm_of(&self, value: Value) -> Option<Xmm> {
        self.plan.float_of(value).map(Xmm)
    }

    /// The register to compute `value` in: its location register, or rax.
    pub(super) fn dst(&self, value: Value) -> Reg {
        self.reg_of(value).unwrap_or(Reg::Rax)
    }

    /// The float register to compute `value` in: its location register, or xmm0.
    pub(super) fn fdst(&self, value: Value) -> Xmm {
        self.xmm_of(value).unwrap_or(Xmm(0))
    }

    /// The memory at `addr + offset`: relative to rbp for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        self.address_in(addr, offset, Reg::Rcx)
    }

    /// The memory operand at `addr + offset`, loading `addr` into `scratch` when it is neither in a register nor a frame address.
    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg) -> Mem {
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return Mem::new(Reg::Rbp, disp);
        }
        self.raw(scratch, addr);
        Mem::new(scratch, offset)
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let source: Rm = match self.location(value) {
            Location::Imm(imm) => return self.asm.mov_imm(reg, extend(imm, ty, ext)),
            Location::SlotAddr(_) | Location::Frame(..) => return self.raw(reg, value),
            Location::Float(_) => {
                self.raw(reg, value);
                reg.into()
            }
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
            Location::SlotAddr(sym) => {
                let at = self.asm.mov_slot_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Location::Reg(from) if from == reg => {}
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Float(from) => self.asm.movq_out(reg, Xmm(from)),
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::Q, reg, at);
            }
        }
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        match self.location(value) {
            Location::Reg(location) if location == reg => {}
            Location::Reg(location) => self.asm.mov(location, reg),
            Location::Float(location) => self.asm.movq_in(Xmm(location), reg),
            _ => {
                let at = self.at(value);
                self.asm.store(Size::Q, at, reg);
            }
        }
    }

    /// Loads the float of type `ty` at `from` into `xmm`.
    pub(super) fn xmm_load(&mut self, ty: Type, xmm: Xmm, from: Mem) {
        self.asm.sse(&[scalar(ty)], &[0x0F, 0x10], xmm.0, from);
    }

    /// Stores the float of type `ty` in `xmm` to `to`.
    pub(super) fn xmm_store(&mut self, ty: Type, to: Mem, xmm: Xmm) {
        self.asm.sse(&[scalar(ty)], &[0x0F, 0x11], xmm.0, to);
    }

    /// Puts float `value` in `xmm`.
    pub(super) fn fget(&mut self, xmm: Xmm, value: Value) {
        match self.location(value) {
            Location::Float(from) if Xmm(from) == xmm => {}
            Location::Float(from) => self.asm.movaps(xmm, Xmm(from)),
            Location::Slot => self.xmm_load(self.ty(value), xmm, self.at(value)),
            _ => {
                self.raw(Reg::Rax, value);
                self.asm.movq_in(xmm, Reg::Rax);
            }
        }
    }

    /// Writes `xmm` into the location of float `value`.
    pub(super) fn fput(&mut self, value: Value, xmm: Xmm) {
        match self.location(value) {
            Location::Float(to) if Xmm(to) == xmm => {}
            Location::Float(to) => self.asm.movaps(Xmm(to), xmm),
            _ => self.xmm_store(self.ty(value), self.at(value), xmm),
        }
    }

    /// A float register that holds `value`: its location register, or `scratch` loaded with it.
    pub(super) fn fin(&mut self, value: Value, scratch: Xmm) -> Xmm {
        self.xmm_of(value).unwrap_or_else(|| {
            self.fget(scratch, value);
            scratch
        })
    }

    /// A float register or memory operand that holds `value`.
    pub(super) fn foperand(&self, value: Value) -> Rm {
        match self.xmm_of(value) {
            Some(xmm) => xmm.into(),
            None => self.at(value).into(),
        }
    }

    /// A register or memory operand that holds `value`, using r11 for a frame address or a float register.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Imm(_) | Location::SlotAddr(_) => {
                let at = self.at(value);
                self.move_to(at, value);
                at.into()
            }
            Location::Frame(..) | Location::Float(_) => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
            Location::Slot => self.at(value).into(),
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax or r11 as scratch; a float in a register writes the bytes of its type.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.location(value) {
            Location::Imm(imm) => match i32::try_from(imm) {
                Ok(small) => self.asm.store_imm(to, small),
                Err(_) => {
                    self.asm.mov_imm(Reg::R11, imm);
                    self.asm.store(Size::Q, to, Reg::R11);
                }
            },
            Location::SlotAddr(sym) => {
                let at = self.asm.store_slot_reloc(to);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Frame(..) => {
                self.raw(Reg::R11, value);
                self.asm.store(Size::Q, to, Reg::R11);
            }
            Location::Reg(reg) => self.asm.store(Size::Q, to, reg),
            Location::Float(xmm) => self.xmm_store(self.ty(value), to, Xmm(xmm)),
            Location::Slot => self.copy_to(to, self.at(value), self.ty(value)),
        }
    }
}
