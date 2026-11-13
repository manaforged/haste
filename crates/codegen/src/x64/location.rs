use super::asm::{Alu, Mem, Reg, Rm, Shift, Size, Xmm};
use super::float::scalar;
use super::Emit;
use crate::alloc::Location;
use crate::frame::Ext;
use crate::ir::{Type, Value};
use crate::live::SibBase;
use crate::reloc::{RelocKind, StencilReloc};

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    crate::fold::extend(imm, ty, matches!(ext, Ext::Sign))
}

impl Emit<'_> {
    pub(super) fn location(&self, value: Value) -> Location<Reg> {
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

    /// The constant `value` as a 32-bit immediate when `size` is narrower than 64 bits, or when it fits a sign-extended 32-bit immediate.
    pub(super) fn imm_sized(&self, value: Value, size: Size) -> Option<i32> {
        if size == Size::Q {
            self.imm_of(value)
        } else {
            self.const_of(value).map(|imm| imm as i32)
        }
    }

    pub(super) fn cached_spill(&self, value: Value) -> Option<Reg> {
        if !matches!(self.location(value), Location::Slot) {
            return None;
        }
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?}");
        let (cached_disp, cached_reg, stamp) = self.last_spill?;
        (cached_disp == disp && stamp == self.asm.stamp()).then_some(cached_reg)
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
        self.address_in(addr, offset, Reg::Rcx, Reg::R11)
    }

    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg, temp: Reg) -> Mem {
        if let Some(mode) = self.plan.sib[addr.index()] {
            return match mode.base {
                SibBase::Frame(slot) => {
                    let disp = self.frame.slots[slot.index()] + mode.disp + offset;
                    match self.reg_of(mode.index) {
                        Some(idx) => Mem::sib(Reg::Rbp, idx, mode.shift, disp),
                        None => {
                            self.raw(scratch, mode.index);
                            Mem::sib(Reg::Rbp, scratch, mode.shift, disp)
                        }
                    }
                }
                SibBase::Value(base) => {
                    let disp = mode.disp + offset;
                    match (self.reg_of(base), self.reg_of(mode.index)) {
                        (Some(b), Some(idx)) => Mem::sib(b, idx, mode.shift, disp),
                        (Some(b), None) => {
                            self.raw(scratch, mode.index);
                            Mem::sib(b, scratch, mode.shift, disp)
                        }
                        (None, Some(idx)) => {
                            self.raw(scratch, base);
                            Mem::sib(scratch, idx, mode.shift, disp)
                        }
                        (None, None) => {
                            self.raw(scratch, mode.index);
                            if mode.shift > 0 {
                                self.asm.shift_imm(Size::Q, Shift::Shl, scratch, mode.shift);
                            }
                            match self.location(base) {
                                Location::Slot => self.asm.alu(Alu::Add, scratch, self.at(base)),
                                Location::Imm(imm) if i32::try_from(imm).is_ok() => self.asm.alu_imm(Alu::Add, scratch, imm as i32),
                                Location::Frame(slot, off) => {
                                    self.asm.alu(Alu::Add, scratch, Reg::Rbp);
                                    self.asm.alu_imm(Alu::Add, scratch, self.frame.slots[slot.index()] + off);
                                }
                                _ => {
                                    self.raw(temp, base);
                                    self.asm.alu(Alu::Add, scratch, temp);
                                }
                            }
                            Mem::new(scratch, disp)
                        }
                    }
                }
            };
        }
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
        let full = bytes >= 8 || (!matches!(ext, Ext::Sign) && self.plan.zero_ext[value.index()]);
        let source: Rm = match self.location(value) {
            Location::Imm(imm) => return self.asm.mov_imm(reg, extend(imm, ty, ext)),
            Location::SlotAddr(_) | Location::Frame(..) => return self.raw(reg, value),
            Location::Float(from) => {
                self.asm.movq_out(reg, Xmm(from));
                if full {
                    return;
                }
                reg.into()
            }
            Location::Reg(from) => {
                if full {
                    if from != reg {
                        self.asm.mov(reg, from);
                    }
                    return;
                }
                from.into()
            }
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in get()");
                if let Some((cached_disp, cached_reg, stamp)) = self.last_spill
                    && cached_disp == disp
                    && stamp == self.asm.stamp()
                {
                    if full {
                        if cached_reg != reg {
                            self.asm.mov(reg, cached_reg);
                        }
                        self.last_spill = Some((disp, cached_reg, self.asm.stamp()));
                        return;
                    }
                    match ext {
                        Ext::Zero => self.asm.load(Size::of(bytes), reg, cached_reg),
                        Ext::Sign => self.asm.load_signed(Size::of(bytes), reg, cached_reg),
                    }
                    self.last_spill = if reg != cached_reg {
                        Some((disp, cached_reg, self.asm.stamp()))
                    } else {
                        None
                    };
                    return;
                }
                Mem::new(Reg::Rbp, disp).into()
            }
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
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in raw()");
                if let Some((cached_disp, cached_reg, stamp)) = self.last_spill
                    && cached_disp == disp
                    && stamp == self.asm.stamp()
                {
                    if cached_reg != reg {
                        self.asm.mov(reg, cached_reg);
                    }
                    self.last_spill = Some((disp, cached_reg, self.asm.stamp()));
                } else {
                    self.asm.load(Size::Q, reg, Mem::new(Reg::Rbp, disp));
                }
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
                if self.ty(value).bytes() <= 8 {
                    self.last_spill = Some((at.disp, reg, self.asm.stamp()));
                }
            }
        }
    }

    /// Loads the float or vector of type `ty` at `from` into `xmm`; an 8-byte vector by `movq` and a 16-byte vector by `movdqu`, which needs no alignment.
    pub(super) fn xmm_load(&mut self, ty: Type, xmm: Xmm, from: Mem) {
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 { (0xF3, 0x6F) } else { (0xF3, 0x7E) };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, from);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x10], xmm.0, from);
        }
    }

    /// Stores the float or vector of type `ty` in `xmm` to `to`.
    pub(super) fn xmm_store(&mut self, ty: Type, to: Mem, xmm: Xmm) {
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 { (0xF3, 0x7F) } else { (0x66, 0xD6) };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, to);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x11], xmm.0, to);
        }
    }

    /// Puts float `value` in `xmm`.
    pub(super) fn fget(&mut self, xmm: Xmm, value: Value) {
        let ty = self.ty(value);
        match self.location(value) {
            Location::Float(from) if Xmm(from) == xmm => {}
            Location::Float(from) => self.asm.movaps(xmm, Xmm(from)),
            Location::Imm(0) if ty.is_vector() => self.zero_xmm(xmm),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in fget()");
                if let Some((cached_disp, cached_reg, cached_ty, stamp)) = self.last_fspill
                    && cached_disp == disp
                    && cached_ty == ty
                    && stamp == self.asm.stamp()
                {
                    if cached_reg != xmm {
                        self.asm.movaps(xmm, cached_reg);
                    }
                    self.last_fspill = Some((disp, cached_reg, cached_ty, self.asm.stamp()));
                } else {
                    self.xmm_load(ty, xmm, Mem::new(Reg::Rbp, disp));
                }
            }
            _ => {
                self.raw(Reg::Rax, value);
                self.asm.movq_in(xmm, Reg::Rax);
            }
        }
    }

    /// Writes `xmm` into the location of float `value`.
    pub(super) fn fput(&mut self, value: Value, xmm: Xmm) {
        let ty = self.ty(value);
        match self.location(value) {
            Location::Float(to) if Xmm(to) == xmm => {}
            Location::Float(to) => self.asm.movaps(Xmm(to), xmm),
            _ => {
                let at = self.at(value);
                self.xmm_store(ty, at, xmm);
                self.last_fspill = Some((at.disp, xmm, ty, self.asm.stamp()));
            }
        }
    }

    /// A float register that holds `value`: its location register, or `scratch` loaded with it.
    pub(super) fn fin(&mut self, value: Value, scratch: Xmm) -> Xmm {
        self.xmm_of(value).unwrap_or_else(|| {
            self.fget(scratch, value);
            scratch
        })
    }

    /// A float register or memory operand that holds `value`, using `scratch` when `value` has no float home.
    pub(super) fn foperand(&mut self, value: Value, scratch: Xmm) -> Rm {
        match self.location(value) {
            Location::Float(xmm) => Xmm(xmm).into(),
            Location::Slot => self.at(value).into(),
            _ => {
                self.fget(scratch, value);
                scratch.into()
            }
        }
    }

    /// A register or memory operand that holds `value`, using r11 when `value` has no register or stack home.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Slot => self.cached_spill(value).map_or_else(|| Rm::from(self.at(value)), Rm::from),
            _ => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
        }
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax or r11 as scratch; a float in a register writes the bytes of its type.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        match self.location(value) {
            Location::Imm(0) if self.ty(value).is_vector() => {
                self.asm.store_imm(to, 0);
                if self.ty(value).bytes() == 16 {
                    self.asm.store_imm(to.offset(8), 0);
                }
            }
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
            Location::Reg(reg) => {
                self.asm.store(Size::Q, to, reg);
                if to.base == Reg::Rbp && to.index.is_none() && self.ty(value).bytes() <= 8 {
                    self.last_spill = Some((to.disp, reg, self.asm.stamp()));
                }
            }
            Location::Float(xmm) => {
                let ty = self.ty(value);
                self.xmm_store(ty, to, Xmm(xmm));
                if to.base == Reg::Rbp && to.index.is_none() {
                    self.last_fspill = Some((to.disp, Xmm(xmm), ty, self.asm.stamp()));
                }
            }
            Location::Slot => {
                let ty = self.ty(value);
                if ty.bytes() <= 8 && let Some(cached_reg) = self.cached_spill(value) {
                    self.asm.store(Size::Q, to, cached_reg);
                    if to.base == Reg::Rbp && to.index.is_none() {
                        self.last_spill = Some((to.disp, cached_reg, self.asm.stamp()));
                    }
                } else if let Some((cached_disp, cached_reg, cached_ty, stamp)) = self.last_fspill
                    && cached_disp == self.frame.values[value.index()]
                    && cached_ty == ty
                    && stamp == self.asm.stamp()
                {
                    self.xmm_store(ty, to, cached_reg);
                    if to.base == Reg::Rbp && to.index.is_none() {
                        self.last_fspill = Some((to.disp, cached_reg, cached_ty, self.asm.stamp()));
                    }
                } else {
                    self.copy_to(to, self.at(value), ty);
                }
            }
        }
    }
}
