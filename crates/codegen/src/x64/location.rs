pub(super) mod fuse;
mod patterns;
mod sib;

pub(super) use fuse::{fuse, narrows_exactly};
pub(super) use patterns::{LEAF, Root, adc_fold, adc_plan, bextr_parts, product_halves, shld_parts, unwrap_shift_mask};
use patterns::odd_scale_source;

use super::asm::{Alu, Asm, Mem, Reg, Rm, Shift, Size, Xmm};
use super::float::{X2, scalar};
use super::Emit;
use crate::alloc::{Location, Plan};
use crate::fold::fits_imm32;
use crate::frame::{ArgStep, Emitter, Ext};
use crate::ir::{AtomicRmwOp, Function, Inst, IntCC, Op, ShuffleMask, Type, Value};
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
        if self.plan.elided[value.index()] {
            return None;
        }
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
        (mem.base == self.base && mem.index.is_none() && mem.rip.is_none()).then(|| mem.disp - self.bias)
    }

    /// The symbol of `value` when it is an image address that code reaches relative to `rip`.
    pub(super) fn image_sym(&self, value: Value) -> Option<u32> {
        match self.location(value) {
            Location::Symbol(sym) if self.direct_calls && self.func.is_image_addr(value) => Some(sym),
            _ => None,
        }
    }

    /// The memory at `addr + offset`: relative to the frame base for a frame address, to the location register of `addr`, or through rcx.
    pub(super) fn address(&mut self, addr: Value, offset: i32) -> Mem {
        self.address_in(addr, offset, Reg::Rcx, None)
    }

    /// Runs `emit` with the constant `base` in `temp`, or in r11 kept on the stack around it when no temporary is free.
    fn with_base(&mut self, temp: Option<Reg>, base: Value, emit: impl FnOnce(&mut Asm, Reg)) {
        let reg = temp.unwrap_or(Reg::R11);
        if temp.is_none() {
            self.asm.push(Reg::R11);
        }
        self.raw(reg, base);
        emit(&mut self.asm, reg);
        if temp.is_none() {
            self.asm.pop(Reg::R11);
        }
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

    fn sib_slow(&mut self, base: Value, mode: Sib, disp: i32, (scratch, temp): (Reg, Option<Reg>)) -> Mem {
        self.load_index(scratch, mode.index);
        if mode.shift > 0
            && !matches!(self.location(base), Location::Slot | Location::Frame(..))
            && !matches!(self.location(base), Location::Imm(imm) if fits_imm32(imm))
        {
            self.with_base(temp, base, |asm, reg| asm.lea(scratch, Mem::sib(reg, scratch, mode.shift, 0)));
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
            _ => self.with_base(temp, base, |asm, reg| asm.alu(Alu::Add, scratch, reg)),
        }
        self.clobber_reg(scratch);
        Mem::new(scratch, disp)
    }

    pub(super) fn sib_mem(&mut self, mode: Sib, offset: i32, scratch: Reg, temp: Option<Reg>) -> Mem {
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

    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg, temp: Option<Reg>) -> Mem {
        if self.plan.elided[addr.index()]
            && let Some(mode) = self.plan.sib[addr.index()]
        {
            return self.sib_mem(mode, offset, scratch, temp);
        }
        if let Some(reg) = self.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Some(sym) = self.image_sym(addr) {
            return Mem::rip(sym, offset);
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

    /// Loads `sym` into `reg` relative to `rip`.
    pub(super) fn emit_rip_symbol(&mut self, reg: Reg, sym: u32) {
        let at = self.asm.lea_rip_reloc(reg);
        self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::X64RipRel32));
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        if let Some(mem) = self.elided_mem(value) {
            self.asm.load(Size::of(self.ty(value).bytes()), reg, mem);
            self.clobber_reg(reg);
            return;
        }
        match self.location(value) {
            Location::Imm(imm) => self.asm.mov_imm(reg, imm),
            Location::SlotAddr(sym) if self.direct_calls => self.emit_rip_symbol(reg, sym),
            Location::SlotAddr(sym) => {
                let at = self.asm.mov_slot_reloc(reg);
                self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Symbol(sym) if self.direct_calls && self.func.is_image_addr(value) => self.emit_rip_symbol(reg, sym),
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
        if let Some(mem) = self.elided_mem(value) {
            self.xmm_load(ty, xmm, mem);
            self.clobber_xmm(xmm);
            return;
        }
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
        if let Some(mem) = self.elided_mem(value) {
            return mem.into();
        }
        match self.location(value) {
            Location::Float(xmm) => Xmm(xmm).into(),
            Location::Slot => self.at(value).into(),
            _ => {
                self.fget(scratch, value);
                scratch.into()
            }
        }
    }

    /// A register or memory operand that holds `value`, using rcx when `value` has no register or stack home.
    pub(super) fn operand(&mut self, value: Value) -> Rm {
        if let Some(mem) = self.elided_mem(value) {
            return mem.into();
        }
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Slot => self.cached_spill(value).map_or_else(|| Rm::from(self.at(value)), Rm::from),
            _ => {
                self.raw(Reg::Rcx, value);
                Reg::Rcx.into()
            }
        }
    }

    /// Whether reading `value` as an operand writes rcx: an elided load addresses through it, and a constant or address without a register or stack home lands in it.
    pub(super) fn reads_through_rcx(&self, value: Value) -> bool {
        self.has_elided_mem(value) || !matches!(self.location(value), Location::Reg(_) | Location::Slot)
    }

    /// `source` and the first of rax and rcx that neither it nor `dst` reads, loading `value` into `dst`, or rax, when `source` reads both.
    pub(super) fn spare(&mut self, dst: Option<Reg>, value: Value, source: Rm) -> (Rm, Reg) {
        let reads = |reg: Reg| match source {
            Rm::Reg(r) => r == reg as u8,
            Rm::Mem(mem) => mem.base == reg || mem.index.is_some_and(|(index, _)| index == reg),
        };
        if let Some(free) = [Reg::Rax, Reg::Rcx].into_iter().find(|&reg| Some(reg) != dst && !reads(reg)) {
            return (source, free);
        }
        let into = dst.unwrap_or(Reg::Rax);
        self.raw(into, value);
        (into.into(), if into == Reg::Rax { Reg::Rcx } else { Reg::Rax })
    }

    /// Loads `value` into rax and then runs `fill`, which loads `dst` without writing rax; when `dst` is rax, runs `fill` first and loads `value` into rcx. Returns the register that holds `value`.
    pub(super) fn beside(&mut self, dst: Reg, value: Value, fill: impl FnOnce(&mut Self)) -> Reg {
        if dst == Reg::Rax {
            fill(self);
            self.raw(Reg::Rcx, value);
            return Reg::Rcx;
        }
        self.raw(Reg::Rax, value);
        fill(self);
        Reg::Rax
    }

    /// Runs one step of a parallel move between integer registers, saving through `scratch`, or through the stack when no register is free.
    pub(super) fn int_step(&mut self, step: ArgStep<Reg>, scratch: Option<Reg>) {
        match (step, scratch) {
            (ArgStep::Move(src, dst), _) => self.asm.mov(dst, src),
            (ArgStep::Save(src), Some(reg)) => self.asm.mov(reg, src),
            (ArgStep::Save(src), None) => self.asm.push(src),
            (ArgStep::Restore(dst), Some(reg)) => self.asm.mov(dst, reg),
            (ArgStep::Restore(dst), None) => self.asm.pop(dst),
            (ArgStep::Load(val, dst), _) => self.raw(dst, val),
        }
        match step {
            ArgStep::Move(_, reg) | ArgStep::Restore(reg) => self.clobber_reg(reg),
            ArgStep::Save(_) => scratch.into_iter().for_each(|reg| self.clobber_reg(reg)),
            ArgStep::Load(..) => {}
        }
    }

    /// Holds `arg` in edge scratch `index` while a block's parameters move: rcx for the first, the top of the stack for the second.
    pub(super) fn save_edge(&mut self, index: usize, arg: Value) {
        if index == 0 {
            return self.raw(Reg::Rcx, arg);
        }
        self.raw(Reg::Rax, arg);
        self.asm.push(Reg::Rax);
        self.shift_base(8);
    }

    /// Writes edge scratch `index`, which `save_edge` filled, into `param`.
    pub(super) fn restore_edge(&mut self, index: usize, param: Value) {
        if index == 0 {
            return self.put(param, Reg::Rcx);
        }
        self.asm.pop(Reg::Rax);
        self.shift_base(-8);
        self.clobber_reg(Reg::Rax);
        self.put(param, Reg::Rax);
    }

    /// Keeps frame addresses right while `by` bytes sit pushed on a frame addressed from rsp.
    fn shift_base(&mut self, by: i32) {
        if self.base == Reg::Rsp {
            self.bias += by;
        }
    }

    /// Reserves `bytes` of scratch below rsp, keeping frame addresses right.
    pub(super) fn reserve_scratch(&mut self, bytes: i32) {
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, bytes);
        self.shift_base(bytes);
    }

    /// Releases scratch that `reserve_scratch` took.
    pub(super) fn release_scratch(&mut self, bytes: i32) {
        self.asm.alu_imm(Alu::Add, Reg::Rsp, bytes);
        self.shift_base(-bytes);
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

impl Emit<'_> {
    /// Broadcasts lane 0 of `v` into `result` straight from the scalar register when fuse elided `v` as an insert into zero; returns whether it did.
    pub(super) fn splat_insert(&mut self, v: Value, result: Value) -> bool {
        let Some(Op::Insert(_, scalar, 0)) = self.defs[v.index()].filter(|_| self.plan.elided[v.index()]).map(|inst| self.func.insts[inst.index()].op.clone()) else { return false };
        let Some((lane, _)) = self.ty(v).vec_shape() else { return false };
        let dst = self.xmm_of(result).unwrap_or(X2);
        let reg = self.reg_of(scalar).unwrap_or_else(|| {
            self.get(Reg::Rax, scalar, Ext::Zero);
            Reg::Rax
        });
        self.clobber_reg(Reg::Rax);
        self.emit_mov_xmm_gpr(dst, reg, lane == crate::ir::Lane::I64);
        self.asm.vex(super::asm::Vl::X, super::ymm::broadcast(lane), (dst, Xmm(0)), dst);
        self.fput(result, dst);
        true
    }
}

type Site = Option<(usize, usize, Inst)>;

/// Elides each integer lane-0 insert into a zero vector whose every use is one later lane-0 broadcast in its block, which then moves the scalar in from its own register.
pub(super) fn elide_broadcast_inserts(func: &Function, plan: &mut Plan<Reg>, defs: &[Option<Inst>], (uses, pos_of, use_of): (&[u32], &[Site], &[Site])) {
    let zero = |v: Value| defs[v.index()].is_some_and(|inst| matches!(func.insts[inst.index()].op, Op::Const(0)));
    for (v, def) in pos_of.iter().enumerate() {
        let Some((block, start, inst)) = *def else { continue };
        let Op::Insert(base, scalar, 0) = func.insts[inst.index()].op else { continue };
        let Some((lane, count)) = func.ty(base).vec_shape() else { continue };
        if base == scalar || !lane.is_int() || func.ty(base).bytes() != 16 || !zero(base) || plan.elided[v] { continue; }
        let Some((used_in, end, user)) = use_of[v] else { continue };
        let Op::Shuffle(a, b, ShuffleMask::Const(mask)) = func.insts[user.index()].op else { continue };
        let reads = u32::from(a.index() == v) + u32::from(b.index() == v);
        if used_in != block || a.index() != v || uses[v] != reads || mask[..usize::from(count)].iter().any(|&at| at != 0) { continue; }
        if fuse::preserved(func, plan, block, start, end, &[], scalar) {
            plan.elided[v] = true;
        }
    }
}

/// Marks each integer `AtomicRmw` whose result is unused (or a `Sub` consumed solely by an adjacent equality compare against its operand) as `fused` for a direct `lock` op.
pub(super) fn fuse_atomic_rmws(func: &Function, plan: &mut Plan<Reg>, (uses, pos_of, use_of): (&[u32], &[Site], &[Site])) {
    let eq_val = |plan: &Plan<Reg>, ty: Type, a: Value, b: Value| {
        a == b
            || plan.const_of(a).zip(plan.const_of(b)).is_some_and(|(c1, c2)| {
                crate::fold::extend(c1, ty, false) == crate::fold::extend(c2, ty, false)
            })
    };
    for (v, def) in pos_of.iter().enumerate() {
        let Some((block, start, inst)) = *def else { continue };
        let (res, ty) = (Value(v as u32), func.ty(Value(v as u32)));
        if !ty.is_small_int() || plan.elided[v] || plan.fused[v] {
            continue;
        }
        let expected = match func.insts[inst.index()].op {
            Op::AtomicRmw(rmw, _, _, value) => {
                if uses[v] == 0 && matches!(rmw, AtomicRmwOp::Add | AtomicRmwOp::Sub | AtomicRmwOp::Or | AtomicRmwOp::And | AtomicRmwOp::Xor | AtomicRmwOp::Xchg) {
                    plan.fused[v] = true;
                    continue;
                }
                if rmw != AtomicRmwOp::Sub || uses[v] != 1 { continue; }
                value
            }
            Op::AtomicCas(_, _, expected, _) => {
                if uses[v] == 0 {
                    plan.fused[v] = true;
                    continue;
                }
                if uses[v] != 1 { continue; }
                expected
            }
            _ => continue,
        };
        let Some((used_in, end, user)) = use_of[v] else { continue };
        let udata = &func.insts[user.index()];
        let zero_expected = plan.const_of(expected).is_some_and(|c| crate::fold::extend(c, ty, false) == 0);
        let ok_user = match udata.op {
            Op::Icmp(IntCC::Equal | IntCC::NotEqual, lhs, rhs) => {
                udata.result.is_some_and(|r| !plan.elided[r.index()] && !plan.is_static(r))
                    && ((lhs == res && eq_val(plan, ty, rhs, expected)) || (rhs == res && eq_val(plan, ty, lhs, expected)))
            }
            Op::Brif(cond, ..) | Op::Trapz(cond) => cond == res && zero_expected,
            _ => false,
        };
        if used_in != block || end <= start || !ok_user {
            continue;
        }
        let quiet = (start + 1..end).all(|p| {
            let mid = &func.insts[func.blocks[block].insts[p].index()];
            !matches!(mid.op, Op::Call { .. } | Op::TryCall { .. })
                && mid.result.is_some_and(|r| {
                    plan.elided[r.index()]
                        || (plan.is_static(r) && matches!(mid.op, Op::Const(_) | Op::StackAddr(..) | Op::Symbol(_) | Op::SlotAddr(_)))
                })
        });
        if quiet {
            plan.fused[v] = true;
        }
    }
}
