use super::asm::{FP, Mem, Reg, Size};
use super::{A, C, Emit, Ext, IP};
use crate::alloc::Location;
use crate::ir::{StackSlot, Type, Value};

impl Emit<'_> {
    pub(super) fn at(&self, value: Value) -> Mem {
        Mem::new(FP, self.frame.values[value.index()])
    }

    pub(super) fn slot(&self, slot: StackSlot, offset: i32) -> Mem {
        Mem::new(FP, self.frame.slots[slot.index()] + offset)
    }

    /// The register to compute `value` in: its location register, or A.
    pub(super) fn dst(&self, value: Value) -> Reg {
        self.plan.reg_of(value).unwrap_or(A)
    }

    /// Puts the address `slot + offset` in `reg`.
    pub(super) fn slot_address(&mut self, reg: Reg, slot: StackSlot, offset: i32) {
        let disp = self.frame.slots[slot.index()] + offset;
        self.add_const(reg, FP, i64::from(disp));
    }

    /// `dst = base + imm`, with an immediate form when it fits one.
    pub(super) fn add_const(&mut self, dst: Reg, base: Reg, imm: i64) {
        match u32::try_from(imm.unsigned_abs()) {
            Ok(small) if small < 4096 => self.asm.add_imm(imm < 0, false, Size::X, dst, base, small),
            _ => {
                self.asm.mov_imm(IP, imm);
                self.asm.add_extended(false, dst, base, IP);
            }
        }
    }

    /// The memory at `addr + offset`: relative to the frame pointer for a frame address, to the location register of `addr`, or through C.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        if let Some(reg) = self.plan.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.plan.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return Mem::new(FP, disp);
        }
        self.raw(C, addr);
        Mem::new(C, offset)
    }

    /// Puts the 64 bits of `value` in `reg`; bits above its type's width are unspecified.
    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.plan.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::Frame(slot, offset) => self.slot_address(reg, slot, offset),
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::X, false, reg, at);
            }
        }
    }

    /// Puts `value` in `reg`, extended from its type's width to 64 bits by `ext`.
    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let signed = matches!(ext, Ext::Sign);
        match self.plan.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, crate::fold::extend(imm, ty, signed)),
            Location::Frame(..) => self.raw(reg, value),
            Location::Reg(from) => self.asm.extend(signed, bytes, reg, from),
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::of(bytes), signed && bytes < 8, reg, at);
            }
        }
    }

    /// A register that holds `value` extended by `ext`: its location register when no extension is needed, else `scratch` loaded with it.
    pub(super) fn src(&mut self, value: Value, scratch: Reg, ext: Ext) -> Reg {
        match self.plan.reg_of(value) {
            Some(reg) if self.ty(value).bytes() >= 8 => reg,
            _ => {
                self.get(scratch, value, ext);
                scratch
            }
        }
    }

    /// A register that holds the 64 bits of `value`, which may be unspecified above its type's width.
    pub(super) fn bits(&mut self, value: Value, scratch: Reg) -> Reg {
        match self.plan.reg_of(value) {
            Some(reg) => reg,
            None => {
                self.raw(scratch, value);
                scratch
            }
        }
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        if let Location::Reg(location) = self.plan.location(value) {
            self.asm.mov(location, reg);
            return;
        }
        let at = self.at(value);
        self.asm.store(Size::X, at, reg);
    }

    pub(super) fn high(&mut self, reg: Reg, value: Value) {
        let at = self.at(value).offset(8);
        self.asm.load(Size::X, false, reg, at);
    }

    pub(super) fn put_high(&mut self, value: Value, reg: Reg) {
        let at = self.at(value).offset(8);
        self.asm.store(Size::X, at, reg);
    }

    /// Loads the location of `value` from the bytes at `from`.
    pub(super) fn load_home(&mut self, value: Value, from: Mem) {
        match self.plan.location(value) {
            Location::Reg(reg) => self.asm.load(Size::X, false, reg, from),
            _ => self.copy_to(self.at(value), from, self.ty(value)),
        }
    }

    pub(super) fn copy_to(&mut self, to: Mem, from: Mem, ty: Type) {
        self.asm.load(Size::X, false, IP, from);
        self.asm.store(Size::X, to, IP);
        if ty == Type::I128 {
            self.asm.load(Size::X, false, IP, from.offset(8));
            self.asm.store(Size::X, to.offset(8), IP);
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using IP as scratch.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.plan.location(value) {
            Location::Imm(0) => self.asm.store(Size::X, to, super::ZR),
            Location::Imm(_) | Location::Frame(..) => {
                self.raw(IP, value);
                self.asm.store(Size::X, to, IP);
            }
            Location::Reg(reg) => self.asm.store(Size::X, to, reg),
            Location::Slot => self.copy_to(to, self.at(value), self.ty(value)),
        }
    }
}
