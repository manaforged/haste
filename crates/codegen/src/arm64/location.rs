use super::asm::{FP, Mem, Reg, Size, Vreg};
use super::{A, C, Emit, IP, V0};
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
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?} in at()");
        self.frame_mem(disp)
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

    pub(super) fn invalidate_spills(&mut self) {
        self.last_spill = [None; 4];
        self.last_fspill = None;
    }

    pub(super) fn clobber_reg(&mut self, reg: Reg) {
        if reg == super::ZR {
            return;
        }
        for slot in &mut self.last_spill {
            if slot.is_some_and(|(_, r)| r == reg) {
                *slot = None;
            }
        }
    }

    pub(super) fn clobber_regs(&mut self, regs: &[Reg]) {
        for &reg in regs {
            self.clobber_reg(reg);
        }
    }

    pub(super) fn clobber_vreg(&mut self, reg: Vreg) {
        if self.last_fspill.is_some_and(|(_, r, _)| r == reg) {
            self.last_fspill = None;
        }
    }

    pub(super) fn invalidate_disp(&mut self, disp: i32) {
        for slot in &mut self.last_spill {
            if slot.is_some_and(|(d, _)| d == disp) {
                *slot = None;
            }
        }
        if self.last_fspill.is_some_and(|(d, _, _)| d == disp) {
            self.last_fspill = None;
        }
    }

    pub(super) fn record_spill(&mut self, disp: i32, reg: Reg) {
        self.clobber_reg(reg);
        let mut out = [Some((disp, reg)), None, None, None];
        let mut i = 1;
        for entry in self.last_spill.into_iter().flatten() {
            if entry != (disp, reg) && i < 4 {
                out[i] = Some(entry);
                i += 1;
            }
        }
        self.last_spill = out;
    }

    pub(super) fn write_spill(&mut self, disp: i32, reg: Reg) {
        self.invalidate_disp(disp);
        self.record_spill(disp, reg);
    }

    fn cached_spill_for(&self, disp: i32, preferred: Reg) -> Option<Reg> {
        let mut fallback = None;
        for (d, r) in self.last_spill.into_iter().flatten() {
            if d == disp {
                if r == preferred {
                    return Some(r);
                }
                if r != super::ZR || fallback.is_none() {
                    fallback = Some(r);
                }
            }
        }
        fallback
    }

    pub(super) fn slot_address(&mut self, reg: Reg, slot: StackSlot, offset: i32) {
        let disp = self.frame.slots[slot.index()] + offset;
        self.add_const(reg, FP, i64::from(disp));
    }

    pub(super) fn add_const(&mut self, dst: Reg, base: Reg, imm: i64) {
        match u32::try_from(imm.unsigned_abs()) {
            Ok(small) if small < 4096 => self.asm.add_imm(imm < 0, false, Size::X, dst, base, small),
            Ok(mid) if mid < (1 << 24) => {
                let (hi, lo) = (mid & !0xFFF, mid & 0xFFF);
                self.asm.add_imm(imm < 0, false, Size::X, dst, base, hi);
                if lo != 0 {
                    self.asm.add_imm(imm < 0, false, Size::X, dst, dst, lo);
                }
            }
            _ => {
                self.asm.mov_imm(IP, imm);
                self.asm.add_extended(false, dst, base, IP);
                self.clobber_reg(IP);
            }
        }
        self.clobber_reg(dst);
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
            self.clobber_reg(scratch);
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
            Location::Imm(imm) => {
                self.asm.mov_imm(reg, imm);
                self.clobber_reg(reg);
            }
            Location::SlotAddr(sym) | Location::Symbol(sym) if self.direct_calls => {
                let at = self.asm.adrp_add_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Arm64AdrpAdd));
                self.clobber_reg(reg);
            }
            Location::SlotAddr(sym) | Location::Symbol(sym) => {
                let at = self.asm.mov_symbol(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Arm64Movw64));
                self.clobber_reg(reg);
            }
            Location::Frame(slot, offset) => self.slot_address(reg, slot, offset),
            Location::Reg(from) => {
                if reg != from {
                    self.asm.mov(reg, from);
                    self.clobber_reg(reg);
                }
            }
            Location::Float(from) => {
                self.asm.fmov_out(self.ty(value).bytes() == 8, reg, Vreg(from));
                self.clobber_reg(reg);
            }
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in raw()");
                if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
                    if cached_reg != reg {
                        self.asm.mov(reg, cached_reg);
                        self.record_spill(disp, reg);
                    }
                } else {
                    let at = self.frame_mem(disp);
                    self.asm.load(Size::X, false, reg, at);
                    self.record_spill(disp, reg);
                }
            }
        }
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let signed = matches!(ext, Ext::Sign);
        match self.plan.location(value) {
            Location::Imm(imm) => {
                self.asm.mov_imm(reg, crate::fold::extend(imm, ty, signed));
                self.clobber_reg(reg);
            }
            Location::Frame(..) | Location::SlotAddr(_) | Location::Symbol(_) => self.raw(reg, value),
            Location::Reg(from) if !signed && self.plan.zero_ext[value.index()] => {
                if reg != from {
                    self.asm.mov(reg, from);
                    self.clobber_reg(reg);
                }
            }
            Location::Reg(from) => {
                self.asm.extend(signed, bytes, reg, from);
                self.clobber_reg(reg);
            }
            Location::Float(from) => {
                self.asm.fmov_out(ty.bytes() == 8, reg, Vreg(from));
                self.asm.extend(signed, bytes, reg, reg);
                self.clobber_reg(reg);
            }
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in get()");
                if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
                    if bytes >= 8 {
                        if cached_reg != reg {
                            self.asm.mov(reg, cached_reg);
                            self.record_spill(disp, reg);
                        }
                    } else {
                        if cached_reg == super::ZR {
                            self.asm.mov(reg, super::ZR);
                        } else {
                            self.asm.extend(signed, bytes, reg, cached_reg);
                        }
                        self.clobber_reg(reg);
                    }
                } else {
                    let at = self.frame_mem(disp);
                    self.asm.load(Size::of(bytes), signed && bytes < 8, reg, at);
                    if bytes >= 8 {
                        self.record_spill(disp, reg);
                    } else {
                        self.clobber_reg(reg);
                    }
                }
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
            Location::Reg(location) => {
                if location != reg {
                    self.asm.mov(location, reg);
                }
                self.clobber_reg(location);
            }
            Location::Float(location) => {
                self.clobber_vreg(Vreg(location));
                self.asm.fmov_in(self.ty(value).bytes() == 8, Vreg(location), reg);
            }
            _ => {
                let disp = self.frame.values[value.index()];
                let at = self.at(value);
                self.asm.store(Size::X, at, reg);
                if self.ty(value).bytes() <= 8 {
                    self.write_spill(disp, reg);
                } else {
                    self.invalidate_disp(disp);
                }
            }
        }
    }

    pub(super) fn fget(&mut self, reg: Vreg, value: Value) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Float(from) if Vreg(from) == reg => {}
            Location::Float(from) if ty.is_vector() => {
                self.asm.vmov(reg, Vreg(from));
                self.clobber_vreg(reg);
            }
            Location::Float(from) => {
                self.asm.fmov_reg(ty == Type::F64, reg, Vreg(from));
                self.clobber_vreg(reg);
            }
            Location::Imm(0) if ty.is_vector() => {
                self.asm.veor_zero(reg);
                self.clobber_vreg(reg);
            }
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in fget()");
                if let Some((cached_disp, cached_reg, cached_ty)) = self.last_fspill
                    && cached_disp == disp
                    && cached_ty == ty
                {
                    if cached_reg != reg {
                        if ty.is_vector() {
                            self.asm.vmov(reg, cached_reg);
                        } else {
                            self.asm.fmov_reg(ty == Type::F64, reg, cached_reg);
                        }
                        self.clobber_vreg(reg);
                    }
                } else {
                    let at = self.frame_mem(disp);
                    self.asm.vload(ty.bytes(), reg, at);
                    self.clobber_vreg(reg);
                    self.last_fspill = Some((disp, reg, ty));
                }
            }
            _ => {
                self.raw(IP, value);
                self.asm.fmov_in(ty.bytes() == 8, reg, IP);
                self.clobber_vreg(reg);
            }
        }
    }

    pub(super) fn fput(&mut self, value: Value, reg: Vreg) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Float(to) => {
                if Vreg(to) != reg {
                    if ty.is_vector() {
                        self.asm.vmov(Vreg(to), reg);
                    } else {
                        self.asm.fmov_reg(ty == Type::F64, Vreg(to), reg);
                    }
                }
                self.clobber_vreg(Vreg(to));
            }
            _ => {
                let disp = self.frame.values[value.index()];
                self.asm.vstore(ty.bytes(), self.at(value), reg);
                self.invalidate_disp(disp);
                self.last_fspill = Some((disp, reg, ty));
            }
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
        self.clobber_reg(reg);
    }

    pub(super) fn put_high(&mut self, value: Value, reg: Reg) {
        let disp = self.frame.values[value.index()];
        let at = self.at(value).offset(8);
        self.asm.store(Size::X, at, reg);
        self.invalidate_disp(disp);
    }

    pub(super) fn load_home(&mut self, value: Value, from: Mem) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Reg(reg) => {
                self.asm.load(Size::X, false, reg, from);
                self.clobber_reg(reg);
            }
            Location::Float(v) => {
                self.asm.vload(ty.bytes(), Vreg(v), from);
                self.clobber_vreg(Vreg(v));
            }
            _ => {
                let disp = self.frame.values[value.index()];
                self.copy_to(self.at(value), from, ty);
                if ty.bytes() <= 8 {
                    self.write_spill(disp, IP);
                } else {
                    self.invalidate_disp(disp);
                }
            }
        }
    }

    pub(super) fn copy_to(&mut self, to: Mem, from: Mem, ty: Type) {
        self.asm.load(Size::X, false, IP, from);
        self.asm.store(Size::X, to, IP);
        if ty.bytes() == 16 {
            self.asm.load(Size::X, false, IP, from.offset(8));
            self.asm.store(Size::X, to.offset(8), IP);
        }
        self.clobber_reg(IP);
    }

    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        self.move_to_slot(to, None, value);
    }

    pub(super) fn move_to_slot(&mut self, to: Mem, slot_disp: Option<i32>, value: Value) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Imm(0) if ty.bytes() == 16 => {
                self.asm.store(Size::X, to, super::ZR);
                self.asm.store(Size::X, to.offset(8), super::ZR);
                if let Some(d) = slot_disp {
                    self.invalidate_disp(d);
                }
            }
            Location::Imm(0) => {
                self.asm.store(Size::X, to, super::ZR);
                if let Some(d) = slot_disp {
                    self.write_spill(d, super::ZR);
                }
            }
            Location::Imm(_) | Location::Frame(..) | Location::SlotAddr(_) | Location::Symbol(_) => {
                self.raw(IP, value);
                self.asm.store(Size::X, to, IP);
                if let Some(d) = slot_disp {
                    if ty.bytes() <= 8 {
                        self.write_spill(d, IP);
                    } else {
                        self.invalidate_disp(d);
                    }
                }
            }
            Location::Reg(reg) => {
                self.asm.store(Size::X, to, reg);
                if let Some(d) = slot_disp {
                    if ty.bytes() <= 8 {
                        self.write_spill(d, reg);
                    } else {
                        self.invalidate_disp(d);
                    }
                }
            }
            Location::Float(v) => {
                self.asm.vstore(ty.bytes(), to, Vreg(v));
                if let Some(d) = slot_disp {
                    self.invalidate_disp(d);
                    self.last_fspill = Some((d, Vreg(v), ty));
                }
            }
            Location::Slot => {
                let src_disp = self.frame.values[value.index()];
                if ty.bytes() <= 8 && let Some(cached) = self.cached_spill_for(src_disp, super::ZR) {
                    self.asm.store(Size::X, to, cached);
                    if let Some(d) = slot_disp {
                        self.write_spill(d, cached);
                    }
                } else if let Some((cd, cr, ct)) = self.last_fspill
                    && cd == src_disp
                    && ct == ty
                {
                    self.asm.vstore(ty.bytes(), to, cr);
                    if let Some(d) = slot_disp {
                        self.invalidate_disp(d);
                        self.last_fspill = Some((d, cr, ct));
                    }
                } else {
                    self.copy_to(to, self.at(value), ty);
                    if let Some(d) = slot_disp {
                        if ty.bytes() <= 8 {
                            self.write_spill(d, IP);
                        } else {
                            self.invalidate_disp(d);
                        }
                    }
                }
            }
        }
    }
}
