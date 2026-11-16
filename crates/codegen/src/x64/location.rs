mod fuse;
mod patterns;
mod sib;

pub(super) use fuse::{fuse, narrows_exactly};
pub(super) use patterns::{LEAF, Root, adc_plan, bextr_parts, shld_parts, unwrap_shift_mask};
use patterns::odd_scale_source;

use super::asm::{Alu, Mem, Reg, Rm, Shift, Size, Xmm};
use super::float::{X2, scalar};
use super::Emit;
use crate::alloc::Location;
use crate::fold::fits_imm32;
use crate::frame::{Emitter, Ext};
use crate::ir::{Op, Type, Value};
use crate::live::{Sib, SibBase};
use crate::reloc::{RelocKind, StencilReloc};

fn extend(imm: i64, ty: Type, ext: Ext) -> i64 {
    crate::fold::extend(imm, ty, matches!(ext, Ext::Sign))
}

const ODD_LEA_TAG: i32 = 0x4000_0000;

fn odd_lea_key(x: Value, lea_scale: u8) -> Option<i32> {
    i32::try_from(x.index()).ok().filter(|&i| i < (1 << 28)).map(|i| ODD_LEA_TAG | (i << 2) | i32::from(lea_scale & 3))
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
        if self.plan.elided[value.index()] || !matches!(self.location(value), Location::Slot) {
            return None;
        }
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?}");
        self.cached_spill_for(disp, Reg::Rax)
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
        if self.plan.elided[value.index()] {
            return None;
        }
        self.plan.reg_of(value)
    }

    pub(super) fn has_elided_mem(&self, value: Value) -> bool {
        self.plan.elided[value.index()]
            && self.defs[value.index()].is_some_and(|inst| matches!(self.func.insts[inst.index()].op, Op::Load(..) | Op::StackLoad(..)))
    }

    pub(super) fn elided_mem(&mut self, value: Value) -> Option<Mem> {
        if !self.plan.elided[value.index()] {
            return None;
        }
        let inst = self.defs[value.index()]?;
        match self.func.insts[inst.index()].op {
            Op::Load(addr, off) => Some(self.address(addr, off)),
            Op::StackLoad(slot, off) => Some(self.slot(slot, off)),
            _ => None,
        }
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

    /// The frame memory at `disp` from where rbp would point, addressed from the frame base.
    pub(super) fn frame_mem(&self, disp: i32) -> Mem {
        Mem::new(self.base, disp + self.bias)
    }

    /// The displacement of `mem` from where rbp would point when it is a plain frame address.
    pub(super) fn frame_disp(&self, mem: Mem) -> Option<i32> {
        (mem.base == self.base && mem.index.is_none()).then(|| mem.disp - self.bias)
    }

    /// The memory at `addr + offset`: relative to the frame base for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        self.address_in(addr, offset, Reg::Rcx, Reg::R11)
    }

    fn load_odd_index(&mut self, dst: Reg, x: Value, lea_scale: u8) {
        let key = odd_lea_key(x, lea_scale);
        if let Some(cached) = key.and_then(|k| self.cached_spill_for(k, dst)) {
            if cached != dst {
                self.asm.mov(dst, cached);
                self.clobber_reg(dst);
            }
            return;
        }
        let rx = match self.reg_of(x).filter(|&r| r != Reg::Rsp && (self.ty(x).bytes() >= 8 || self.plan.zero_ext[x.index()])) {
            Some(r) => r,
            None => {
                self.get(dst, x, Ext::Zero);
                dst
            }
        };
        self.asm.lea(dst, Mem::sib(rx, rx, lea_scale, 0));
        match key {
            Some(k) if dst != Reg::R11 => self.record_spill(k, dst),
            _ => self.clobber_reg(dst),
        }
    }

    fn load_index(&mut self, dst: Reg, index: Value) {
        if let Some((x, lea_scale)) = odd_scale_source(self.func, &self.plan, |v| self.defs[v.index()], index) {
            self.load_odd_index(dst, x, lea_scale);
        } else {
            self.get(dst, index, Ext::Zero);
        }
    }

    fn sib_slow(&mut self, base: Value, mode: Sib, disp: i32, (scratch, temp): (Reg, Reg)) -> Mem {
        self.load_index(scratch, mode.index);
        if mode.shift > 0
            && !matches!(self.location(base), Location::Slot | Location::Frame(..))
            && !matches!(self.location(base), Location::Imm(imm) if fits_imm32(imm))
        {
            self.raw(temp, base);
            self.asm.lea(scratch, Mem::sib(temp, scratch, mode.shift, 0));
            self.clobber_reg(scratch);
            return Mem::new(scratch, disp);
        }
        self.clear_flags();
        if mode.shift > 0 {
            self.asm.shift_imm(Size::Q, Shift::Shl, scratch, mode.shift);
        }
        match self.location(base) {
            Location::Slot => self.asm.alu(Alu::Add, scratch, self.at(base)),
            Location::Imm(imm) if fits_imm32(imm) => self.asm.alu_imm(Alu::Add, scratch, imm as i32),
            Location::Frame(slot, off) => {
                self.asm.alu(Alu::Add, scratch, self.base);
                self.asm.alu_imm(Alu::Add, scratch, self.frame.slots[slot.index()] + off + self.bias);
            }
            _ => {
                self.raw(temp, base);
                self.asm.alu(Alu::Add, scratch, temp);
            }
        }
        self.clobber_reg(scratch);
        Mem::new(scratch, disp)
    }

    pub(super) fn sib_mem(&mut self, mode: Sib, offset: i32, scratch: Reg, temp: Reg) -> Mem {
        let idx_reg = self.reg_of(mode.index).filter(|_| self.ty(mode.index).bytes() >= 8 || self.plan.zero_ext[mode.index.index()]);
        match mode.base {
            SibBase::Frame(slot) => {
                let disp = self.frame.slots[slot.index()] + mode.disp + offset + self.bias;
                match idx_reg {
                    Some(idx) => Mem::sib(self.base, idx, mode.shift, disp),
                    None => {
                        self.load_index(scratch, mode.index);
                        Mem::sib(self.base, scratch, mode.shift, disp)
                    }
                }
            }
            SibBase::Value(base) => {
                let disp = mode.disp + offset;
                match (self.reg_of(base), idx_reg) {
                    (Some(b), Some(idx)) => Mem::sib(b, idx, mode.shift, disp),
                    (Some(b), None) => {
                        self.load_index(scratch, mode.index);
                        Mem::sib(b, scratch, mode.shift, disp)
                    }
                    (None, Some(idx)) => {
                        self.raw(scratch, base);
                        Mem::sib(scratch, idx, mode.shift, disp)
                    }
                    (None, None) => self.sib_slow(base, mode, disp, (scratch, temp)),
                }
            }
        }
    }

    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg, temp: Reg) -> Mem {
        if self.plan.elided[addr.index()]
            && let Some(mode) = self.plan.sib[addr.index()]
        {
            return self.sib_mem(mode, offset, scratch, temp);
        }
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Frame(slot, base) = self.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(base).and_then(|disp| disp.checked_add(offset))
        {
            return self.frame_mem(disp);
        }
        self.raw(scratch, addr);
        Mem::new(scratch, offset)
    }

    fn load_ext(&mut self, reg: Reg, source: impl Into<Rm>, bytes: u32, ext: Ext) {
        match ext {
            _ if bytes >= 8 => self.asm.load(Size::Q, reg, source),
            Ext::Zero => self.asm.load(Size::of(bytes), reg, source),
            Ext::Sign => self.asm.load_signed(Size::of(bytes), reg, source),
        }
        self.clobber_reg(reg);
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        if let Some(mem) = self.elided_mem(value) {
            self.load_ext(reg, mem, self.ty(value).bytes(), ext);
            return;
        }
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let full = bytes >= 8 || (!matches!(ext, Ext::Sign) && self.plan.zero_ext[value.index()]);
        match self.location(value) {
            Location::Imm(imm) => {
                self.asm.mov_imm(reg, extend(imm, ty, ext));
                self.clobber_reg(reg);
            }
            Location::SlotAddr(_) | Location::Symbol(_) | Location::Frame(..) => self.raw(reg, value),
            Location::Float(from) => {
                self.asm.movq_out(reg, Xmm(from));
                if full {
                    self.clobber_reg(reg);
                } else {
                    self.load_ext(reg, reg, bytes, ext);
                }
            }
            Location::Reg(from) if full => {
                if from != reg {
                    self.asm.mov(reg, from);
                    self.clobber_reg(reg);
                }
            }
            Location::Reg(from) => self.load_ext(reg, from, bytes, ext),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in get()");
                if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
                    if full {
                        if cached_reg != reg {
                            self.asm.mov(reg, cached_reg);
                        }
                        self.record_spill(disp, reg);
                    } else {
                        self.load_ext(reg, cached_reg, bytes, ext);
                    }
                    return;
                }
                self.load_ext(reg, self.frame_mem(disp), bytes, ext);
                if bytes >= 8 {
                    self.record_spill(disp, reg);
                }
            }
        }
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        if let Some(mem) = self.elided_mem(value) {
            self.asm.load(Size::of(self.ty(value).bytes()), reg, mem);
            self.clobber_reg(reg);
            return;
        }
        match self.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::SlotAddr(sym) if self.direct_calls => {
                let at = self.asm.lea_rip_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::X64RipRel32));
            }
            Location::SlotAddr(sym) => {
                let at = self.asm.mov_slot_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Symbol(sym) => {
                let at = self.asm.mov_abs64(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs64));
            }
            Location::Frame(slot, offset) => {
                let at = self.slot(slot, offset);
                self.asm.lea(reg, at);
            }
            Location::Reg(from) if from == reg => return,
            Location::Reg(from) => self.asm.mov(reg, from),
            Location::Float(from) => self.asm.movq_out(reg, Xmm(from)),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in raw()");
                if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
                    if cached_reg != reg {
                        self.asm.mov(reg, cached_reg);
                    }
                } else {
                    self.asm.load(Size::Q, reg, self.frame_mem(disp));
                }
                self.record_spill(disp, reg);
                return;
            }
        }
        self.clobber_reg(reg);
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        match self.location(value) {
            Location::Reg(location) => {
                if location != reg {
                    self.asm.mov(location, reg);
                }
                self.clobber_reg(location);
            }
            Location::Float(location) => {
                self.clobber_xmm(Xmm(location));
                self.asm.movq_in(Xmm(location), reg);
            }
            Location::Slot => self.store_reg_home(self.at(value), reg, self.ty(value).bytes()),
            _ => self.clobber_reg(reg),
        }
    }

    /// Loads the float or vector of type `ty` at `from` into `xmm`; an 8-byte vector by `movq` and a 16-byte vector by `movdqu`, which needs no alignment.
    pub(super) fn xmm_load(&mut self, ty: Type, xmm: Xmm, from: Mem) {
        if ty.is_wide_vec() {
            return self.ymm_load(xmm, from);
        }
        if self.features.avx {
            let (pp, opcode) = if ty.is_vector() {
                (0b10, if ty.bytes() == 16 { 0x6F } else { 0x7E })
            } else {
                (if ty == Type::F64 { 0b11 } else { 0b10 }, 0x10)
            };
            return self.asm.vex_xmm(0b00001, pp, opcode, (xmm, Xmm(0)), from);
        }
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 { (0xF3, 0x6F) } else { (0xF3, 0x7E) };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, from);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x10], xmm.0, from);
        }
    }

    /// Stores the float or vector of type `ty` in `xmm` to `to`.
    pub(super) fn xmm_store(&mut self, ty: Type, to: Mem, xmm: Xmm) {
        if ty.is_wide_vec() {
            return self.ymm_store(to, xmm);
        }
        if self.features.avx {
            let (pp, opcode) = if ty.is_vector() {
                if ty.bytes() == 16 { (0b10, 0x7F) } else { (0b01, 0xD6) }
            } else {
                (if ty == Type::F64 { 0b11 } else { 0b10 }, 0x11)
            };
            return self.asm.vex_xmm(0b00001, pp, opcode, (xmm, Xmm(0)), to);
        }
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
            Location::Float(from) => {
                self.fmov(ty, xmm, Xmm(from));
                self.clobber_xmm(xmm);
            }
            Location::Imm(0) if ty.is_vector() => self.zero_xmm(xmm),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in fget()");
                if let Some((cached_disp, cached_reg, cached_ty)) = self.last_fspill
                    && cached_disp == disp
                    && cached_ty == ty
                {
                    if cached_reg != xmm {
                        self.fmov(ty, xmm, cached_reg);
                        self.clobber_xmm(xmm);
                    }
                } else {
                    self.xmm_load(ty, xmm, self.frame_mem(disp));
                    self.clobber_xmm(xmm);
                    self.last_fspill = Some((disp, xmm, ty));
                }
            }
            _ => {
                self.raw(Reg::Rax, value);
                self.asm.movq_in(xmm, Reg::Rax);
                self.clobber_xmm(xmm);
            }
        }
    }

    /// Writes `xmm` into the location of float `value`.
    pub(super) fn fput(&mut self, value: Value, xmm: Xmm) {
        let ty = self.ty(value);
        match self.location(value) {
            Location::Float(to) => {
                if Xmm(to) != xmm {
                    self.fmov(ty, Xmm(to), xmm);
                }
                self.clobber_xmm(Xmm(to));
            }
            _ => self.store_xmm_home(self.at(value), xmm, ty),
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
        if let Some(mem) = self.elided_mem(value) {
            return mem.into();
        }
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Slot => self.cached_spill(value).map_or_else(|| Rm::from(self.at(value)), Rm::from),
            _ => {
                self.raw(Reg::R11, value);
                Reg::R11.into()
            }
        }
    }

    fn store_reg_home(&mut self, to: Mem, reg: Reg, bytes: u32) {
        self.asm.store(Size::Q, to, reg);
        if let Some(disp) = self.frame_disp(to) {
            if bytes <= 8 {
                self.write_spill(disp, reg);
            } else {
                self.invalidate_disp(disp);
            }
        }
    }

    /// Stores the float or vector of type `ty` in `xmm` to `to` in one store and caches it as the spill of a frame home.
    pub(super) fn store_xmm_home(&mut self, to: Mem, xmm: Xmm, ty: Type) {
        self.xmm_store(ty, to, xmm);
        if let Some(disp) = self.frame_disp(to) {
            self.invalidate_disp(disp);
            self.last_fspill = Some((disp, xmm, ty));
        }
    }

    /// Writes eight zero bytes to `to`, joined with the zero stores just before it through the scratch X2.
    pub(super) fn zero_to(&mut self, to: Mem) {
        if self.asm.store_zero(to, X2) {
            self.clobber_xmm(X2);
        }
    }

    /// Copies the eight bytes at `from` to `to` through rax.
    pub(super) fn copy_word(&mut self, to: Mem, from: Mem) {
        self.asm.load(Size::Q, Reg::Rax, from);
        self.asm.store(Size::Q, to, Reg::Rax);
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax as scratch; a float in a register writes the bytes of its type and a 16-byte vector goes through xmm2 in one store.
    pub(super) fn move_to(&mut self, to: Mem, value: Value) {
        let stack_home = self.frame_disp(to);
        let ty = self.ty(value);
        match self.location(value) {
            Location::Imm(0) if ty.is_vector() && ty.bytes() >= 16 => {
                self.zero_xmm(X2);
                return self.store_xmm_home(to, X2, ty);
            }
            Location::Imm(0) => self.zero_to(to),
            Location::Imm(imm) if fits_imm32(imm) => self.asm.store_imm_sized(Size::Q, to, imm as i32),
            Location::SlotAddr(sym) if !self.direct_calls => {
                let at = self.asm.store_slot_reloc(to);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Imm(_) | Location::Frame(..) | Location::SlotAddr(_) | Location::Symbol(_) => {
                self.raw(Reg::Rax, value);
                self.asm.store(Size::Q, to, Reg::Rax);
            }
            Location::Reg(reg) => return self.store_reg_home(to, reg, ty.bytes()),
            Location::Float(xmm) => return self.store_xmm_home(to, Xmm(xmm), ty),
            Location::Slot => {
                if ty.bytes() <= 8 && let Some(cached_reg) = self.cached_spill(value) {
                    return self.store_reg_home(to, cached_reg, ty.bytes());
                }
                if let Some((cached_disp, cached_reg, cached_ty)) = self.last_fspill
                    && cached_disp == self.frame.values[value.index()]
                    && cached_ty == ty
                {
                    return self.store_xmm_home(to, cached_reg, cached_ty);
                }
                return self.copy_to(to, self.at(value), ty, X2);
            }
        }
        if let Some(disp) = stack_home {
            self.invalidate_disp(disp);
            if ty.bytes() == 16 {
                self.invalidate_disp(disp + 8);
            }
        }
    }
}
