use super::asm::{FP, Mem, Reg, Size, Vreg};
use super::{A, C, D, Emit, IP, V0};
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{StackSlot, Type, Value};
use crate::live::SibBase;
use crate::reloc::{RelocKind, StencilReloc};

impl Emit<'_> {
    pub(super) fn frame_mem(&self, fp_disp: i32) -> Mem {
        if (-256..256).contains(&fp_disp) || !self.framed {
            return Mem::new(FP, fp_disp);
        }
        let sp_disp = i64::from(self.frame.size) + i64::from(self.sp_bias) + i64::from(fp_disp);
        match i32::try_from(sp_disp) {
            Ok(d) if (0..32760).contains(&d) => Mem::new(super::asm::SP, d),
            _ => Mem::new(FP, fp_disp),
        }
    }

    pub(super) fn at(&self, value: Value) -> Mem {
        self.frame_mem(self.frame.values[value.index()])
    }

    pub(super) fn slot(&self, slot: StackSlot, offset: i32) -> Mem {
        self.frame_mem(self.frame.slots[slot.index()] + offset)
    }

    pub(super) fn vreg_of(&self, value: Value) -> Option<Vreg> {
        self.plan.float_of(value).map(Vreg)
    }

    pub(super) fn dst(&self, value: Value) -> Reg {
        self.plan.reg_of(value).unwrap_or(A)
    }

    pub(super) fn fdst(&self, value: Value) -> Vreg {
        self.vreg_of(value).unwrap_or(V0)
    }

    pub(super) fn slot_address(&mut self, reg: Reg, slot: StackSlot, offset: i32) {
        let disp = self.frame.slots[slot.index()] + offset;
        self.add_const(reg, FP, i64::from(disp));
    }

    pub(super) fn add_const(&mut self, dst: Reg, base: Reg, imm: i64) {
        match u32::try_from(imm.unsigned_abs()) {
            Ok(small) if small < 4096 => self.asm.add_imm(imm < 0, false, Size::X, dst, base, small),
            _ => {
                self.asm.mov_imm(IP, imm);
                self.asm.add_extended(false, dst, base, IP);
            }
        }
    }

    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        self.address_in(addr, offset, C)
    }

    pub(super) fn sib_indexed(&mut self, addr: Value, offset: i32, bytes: u32) -> Option<(Reg, Reg, u8)> {
        let mode = self.plan.sib[addr.index()]?;
        let total_disp = match mode.base {
            SibBase::Frame(slot) => i64::from(self.frame.slots[slot.index()]) + i64::from(mode.disp) + i64::from(offset),
            SibBase::Value(_) => i64::from(mode.disp) + i64::from(offset),
        };
        if total_disp != 0 || (mode.shift != 0 && (1u32 << mode.shift) != bytes) {
            return None;
        }
        let base_reg = match mode.base {
            SibBase::Frame(_) => FP,
            SibBase::Value(base) => self.bits(base, C),
        };
        let idx_reg = self.bits(mode.index, IP);
        Some((base_reg, idx_reg, mode.shift))
    }

    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg) -> Mem {
        if let Some(mode) = self.plan.sib[addr.index()] {
            let (base_reg, disp) = match mode.base {
                SibBase::Frame(slot) => (FP, self.frame.slots[slot.index()] + mode.disp + offset),
                SibBase::Value(base) => (self.bits(base, scratch), mode.disp + offset),
            };
            let idx_reg = self.bits(mode.index, IP);
            self.asm.add_shifted(scratch, base_reg, idx_reg, mode.shift);
            return Mem::new(scratch, disp);
        }
        if let Some(reg) = self.plan.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.plan.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return self.frame_mem(disp);
        }
        self.raw(scratch, addr);
        Mem::new(scratch, offset)
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.plan.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::SlotAddr(sym) => {
                let at = self.asm.mov_symbol(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Arm64Movw64));
            }
            Location::Frame(slot, offset) => self.slot_address(reg, slot, offset),
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Float(from) => self.asm.fmov_out(self.ty(value) == Type::F64, reg, Vreg(from)),
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::X, false, reg, at);
            }
        }
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let signed = matches!(ext, Ext::Sign);
        match self.plan.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, crate::fold::extend(imm, ty, signed)),
            Location::Frame(..) | Location::SlotAddr(_) => self.raw(reg, value),
            Location::Reg(from) if !signed && self.plan.zero_ext[value.index()] => self.asm.mov(reg, from),
            Location::Reg(from) => self.asm.extend(signed, bytes, reg, from),
            Location::Float(from) => {
                self.asm.fmov_out(ty == Type::F64, reg, Vreg(from));
                self.asm.extend(signed, bytes, reg, reg);
            }
            Location::Slot => {
                let at = self.at(value);
                self.asm.load(Size::of(bytes), signed && bytes < 8, reg, at);
            }
        }
    }

    pub(super) fn src(&mut self, value: Value, scratch: Reg, ext: Ext) -> Reg {
        match self.plan.reg_of(value) {
            Some(reg) if self.ty(value).bytes() >= 8 || (matches!(ext, Ext::Zero) && self.plan.zero_ext[value.index()]) => reg,
            _ => {
                self.get(scratch, value, ext);
                scratch
            }
        }
    }

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
        match self.plan.location(value) {
            Location::Reg(location) => self.asm.mov(location, reg),
            Location::Float(location) => self.asm.fmov_in(self.ty(value) == Type::F64, Vreg(location), reg),
            _ => {
                let at = self.at(value);
                self.asm.store(Size::X, at, reg);
            }
        }
    }

    pub(super) fn fget(&mut self, reg: Vreg, value: Value) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Float(from) if ty == Type::V128 => self.asm.vmov(reg, Vreg(from)),
            Location::Float(from) => self.asm.fmov_reg(ty == Type::F64, reg, Vreg(from)),
            Location::Slot => self.asm.vload(ty.bytes(), reg, self.at(value)),
            _ => {
                self.raw(D, value);
                self.asm.fmov_in(ty == Type::F64, reg, D);
            }
        }
    }

    pub(super) fn fput(&mut self, value: Value, reg: Vreg) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Float(to) if ty == Type::V128 => self.asm.vmov(Vreg(to), reg),
            Location::Float(to) => self.asm.fmov_reg(ty == Type::F64, Vreg(to), reg),
            _ => self.asm.vstore(ty.bytes(), self.at(value), reg),
        }
    }

    pub(super) fn fin(&mut self, value: Value, scratch: Vreg) -> Vreg {
        self.vreg_of(value).unwrap_or_else(|| {
            self.fget(scratch, value);
            scratch
        })
    }

    pub(super) fn high(&mut self, reg: Reg, value: Value) {
        let at = self.at(value).offset(8);
        self.asm.load(Size::X, false, reg, at);
    }

    pub(super) fn put_high(&mut self, value: Value, reg: Reg) {
        let at = self.at(value).offset(8);
        self.asm.store(Size::X, at, reg);
    }

    pub(super) fn load_home(&mut self, value: Value, from: Mem) {
        match self.plan.location(value) {
            Location::Reg(reg) => self.asm.load(Size::X, false, reg, from),
            Location::Float(v) => self.asm.vload(self.ty(value).bytes(), Vreg(v), from),
            _ => self.copy_to(self.at(value), from, self.ty(value)),
        }
    }

    pub(super) fn copy_to(&mut self, to: Mem, from: Mem, ty: Type) {
        self.asm.load(Size::X, false, IP, from);
        self.asm.store(Size::X, to, IP);
        if ty.bytes() == 16 {
            self.asm.load(Size::X, false, IP, from.offset(8));
            self.asm.store(Size::X, to.offset(8), IP);
        }
    }

    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.plan.location(value) {
            Location::Imm(0) => self.asm.store(Size::X, to, super::ZR),
            Location::Imm(_) | Location::Frame(..) | Location::SlotAddr(_) => {
                self.raw(IP, value);
                self.asm.store(Size::X, to, IP);
            }
            Location::Reg(reg) => self.asm.store(Size::X, to, reg),
            Location::Float(v) => self.asm.vstore(self.ty(value).bytes(), to, Vreg(v)),
            Location::Slot => self.copy_to(to, self.at(value), self.ty(value)),
        }
    }
}
