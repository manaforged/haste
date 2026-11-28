use super::asm::{Alu, Asm, ExtOp, FP, Mem, Reg, Size, Vreg, decode_sib};
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
        self.spills.clear();
        self.asm.clear_anchor();
    }

    pub(super) fn clobber_reg(&mut self, reg: Reg) {
        if reg == super::ZR {
            return;
        }
        self.asm.clobber_anchor(reg);
        self.spills.clobber_reg(reg);
    }

    pub(super) fn clobber_regs(&mut self, regs: &[Reg]) {
        regs.iter().for_each(|&reg| self.clobber_reg(reg));
    }

    pub(super) fn clobber_vreg(&mut self, reg: Vreg) {
        self.spills.clobber_freg(reg);
    }

    pub(super) fn invalidate_disp(&mut self, disp: i32) {
        self.spills.invalidate_disp(disp);
    }

    pub(super) fn resume_spills(&mut self, block: crate::ir::Block) {
        self.spills.resume(block);
    }

    pub(super) fn record_spill(&mut self, disp: i32, reg: Reg) {
        self.clobber_reg(reg);
        self.spills.cache.record(disp, reg);
    }

    pub(super) fn write_spill(&mut self, disp: i32, reg: Reg) {
        self.invalidate_disp(disp);
        self.record_spill(disp, reg);
    }

    fn cached_spill_for(&self, disp: i32, preferred: Reg) -> Option<Reg> {
        self.spills.cached_for(disp, preferred, Some(super::ZR))
    }

    pub(super) fn slot_address(&mut self, reg: Reg, slot: StackSlot, offset: i32) {
        let disp = self.frame.slots[slot.index()] + offset;
        self.add_const(reg, FP, i64::from(disp));
    }

    pub(super) fn add_const(&mut self, dst: Reg, base: Reg, imm: i64) {
        match u32::try_from(imm.unsigned_abs()) {
            Ok(small) if small < 4096 => {
                self.asm.add_imm(imm < 0, false, Size::X, dst, base, small)
            }
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

    pub(super) fn sib_indexed(
        &mut self,
        addr: Value,
        offset: i32,
        bytes: u32,
    ) -> Option<(Reg, Reg, u8)> {
        let mode = self.plan.sib[addr.index()]?;
        let (_, shift) = decode_sib(mode.shift);
        if shift != 0 && (1u32 << shift) != bytes {
            return None;
        }
        let disp = i64::from(mode.disp) + i64::from(offset);
        let base_reg = match mode.base {
            SibBase::Frame(slot) => {
                self.frame_base(disp + i64::from(self.frame.slots[slot.index()]))
            }
            SibBase::Value(base) if disp == 0 => self.bits(base, C),
            SibBase::Value(_) => return None,
        };
        let idx_reg = self.bits(mode.index, IP);
        Some((base_reg, idx_reg, mode.shift))
    }

    fn frame_base(&mut self, fp_disp: i64) -> Reg {
        if fp_disp == 0 {
            return FP;
        }
        self.add_const(C, FP, fp_disp);
        C
    }

    pub(super) fn address_in(&mut self, addr: Value, offset: i32, scratch: Reg) -> Mem {
        if let Some(mode) = self.plan.sib[addr.index()] {
            let (base_reg, disp) = match mode.base {
                SibBase::Frame(slot) => (FP, self.frame.slots[slot.index()] + mode.disp + offset),
                SibBase::Value(base) => (self.bits(base, scratch), mode.disp + offset),
            };
            let idx_reg = self.bits(mode.index, IP);
            let (ext, shift) = decode_sib(mode.shift);
            if ext == ExtOp::Uxtx {
                self.asm.add_shifted(scratch, base_reg, idx_reg, shift);
            } else {
                self.asm
                    .alu_extended(Alu::Add, Size::X, ext, shift, scratch, base_reg, idx_reg);
            }
            self.clobber_reg(scratch);
            return Mem::new(scratch, disp);
        }
        if let Some(reg) = self.plan.reg_of(addr) {
            return Mem::new(reg, offset);
        }
        if let Location::Offset(base, base_off) = self.plan.location(addr)
            && let Some(disp) = base_off.checked_add(offset)
        {
            return Mem::new(base, disp);
        }
        if let Location::Frame(slot, base) = self.plan.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()]
                .checked_add(base)
                .and_then(|disp| disp.checked_add(offset))
        {
            return self.frame_mem(disp);
        }
        self.raw(scratch, addr);
        Mem::new(scratch, offset)
    }

    fn cached_loop_const(&self, imm: i64) -> Option<Reg> {
        self.loop_consts
            .block_consts
            .get(self.cur_block.index())
            .and_then(|list| list.iter().find(|&&(k, _)| k == imm).map(|&(_, r)| r))
    }

    pub(super) fn raw(&mut self, reg: Reg, value: Value) {
        match self.plan.location(value) {
            Location::Imm(imm) => {
                if let Some(cached) = self.cached_loop_const(imm) {
                    if reg != cached {
                        self.asm.mov(reg, cached);
                        self.clobber_reg(reg);
                    }
                } else {
                    self.asm.mov_imm(reg, imm);
                    self.spills.clobber_reg(reg);
                }
            }
            Location::SlotAddr(sym) | Location::Symbol(sym) if self.direct_calls => {
                let at = self.asm.adrp_add_reloc(reg);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::Arm64AdrpAdd));
                self.clobber_reg(reg);
            }
            Location::SlotAddr(sym) | Location::Symbol(sym) => {
                let at = self.asm.mov_symbol(reg);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::Arm64Movw64));
                self.clobber_reg(reg);
            }
            Location::Frame(slot, offset) => self.slot_address(reg, slot, offset),
            Location::Offset(base, offset) => self.add_const(reg, base, i64::from(offset)),
            Location::Reg(from) => {
                if reg != from {
                    self.asm.mov(reg, from);
                    self.clobber_reg(reg);
                }
            }
            Location::Float(from) => {
                self.asm
                    .fmov_out(self.ty(value).bytes() == 8, reg, Vreg(from));
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
                } else if let Some(imm) = self
                    .const_bits(value)
                    .filter(|&imm| imm == 0 || Asm::mov_imm_is_single(imm))
                {
                    self.asm.mov_imm(reg, imm);
                    self.record_spill(disp, reg);
                } else {
                    let at = self.frame_mem(disp);
                    self.asm.load(Size::X, false, reg, at);
                    self.record_spill(disp, reg);
                }
            }
        }
    }

    fn get_slot(&mut self, reg: Reg, value: Value, (bytes, signed, full): (u32, bool, bool)) {
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?} in get()");
        if let Some(cached_reg) = self.cached_spill_for(disp, reg) {
            if full {
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
            if !signed || bytes >= 8 {
                self.record_spill(disp, reg);
            } else {
                self.clobber_reg(reg);
            }
        }
    }

    pub(super) fn get(&mut self, reg: Reg, value: Value, ext: Ext) {
        let ty = self.ty(value);
        let bytes = ty.bytes();
        let signed = matches!(ext, Ext::Sign);
        let full = bytes >= 8 || (!signed && self.plan.zero_ext[value.index()]);
        match self.plan.location(value) {
            Location::Imm(imm) => {
                let ext_imm = crate::fold::extend(imm, ty, signed);
                if let Some(cached) = self.cached_loop_const(ext_imm) {
                    if reg != cached {
                        self.asm.mov(reg, cached);
                        self.clobber_reg(reg);
                    }
                } else {
                    self.asm.mov_imm(reg, ext_imm);
                    self.spills.clobber_reg(reg);
                }
            }
            Location::Frame(..)
            | Location::SlotAddr(_)
            | Location::Symbol(_)
            | Location::Offset(..) => self.raw(reg, value),
            Location::Reg(from) if full => {
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
                if bytes < 4 || (bytes == 4 && signed) {
                    self.asm.extend(signed, bytes, reg, reg);
                }
                self.clobber_reg(reg);
            }
            Location::Slot => self.get_slot(reg, value, (bytes, signed, full)),
        }
    }

    fn is_scratch(reg: Reg) -> bool {
        matches!(
            reg,
            super::ZR | super::A | super::C | super::D | super::S | super::T | super::IP
        ) || reg.0 == 17
    }

    fn borrowable(cached: Reg, scratch: Reg) -> bool {
        !Self::is_scratch(cached)
            || cached == scratch
            || (cached != super::ZR
                && ((scratch == super::C && cached == super::A)
                    || (scratch == super::IP && (cached == super::A || cached == super::C))))
    }

    fn spill_borrow(&self, value: Value, scratch: Reg) -> Option<Reg> {
        let Location::Slot = self.plan.location(value) else {
            return None;
        };
        let disp = self.frame.values[value.index()];
        let mut fallback = None;
        for r in self.spills.cache.holders(disp) {
            if Self::borrowable(r, scratch) {
                if r == scratch {
                    return Some(r);
                }
                if !Self::is_scratch(r) || fallback.is_none() {
                    fallback = Some(r);
                }
            }
        }
        fallback
    }

    fn is_full(&self, value: Value, ext: Ext) -> bool {
        self.ty(value).bytes() >= 8
            || (matches!(ext, Ext::Zero) && self.plan.zero_ext[value.index()])
    }

    fn op_in_scratch(&self, value: Value, scratch: Reg, ext: Option<Ext>) -> bool {
        ext.is_none_or(|e| self.is_full(value, e))
            && self.spill_borrow(value, scratch) == Some(scratch)
    }

    fn op_ready(&self, value: Value, ext: Option<Ext>) -> bool {
        ext.is_none_or(|e| self.is_full(value, e))
            && (self.plan.reg_of(value).is_some() || self.spill_borrow(value, A).is_some())
    }

    fn resolve_op(&mut self, value: Value, scratch: Reg, ext: Option<Ext>) -> Reg {
        match ext {
            Some(e) => self.src(value, scratch, e),
            None => self.bits(value, scratch),
        }
    }

    fn resolve_pair(&mut self, x: Value, y: Value, ext: Option<Ext>) -> (Reg, Reg) {
        if x == y {
            let scratch = if self.op_ready(x, ext) || !self.op_in_scratch(x, C, ext) {
                A
            } else {
                C
            };
            let reg = self.resolve_op(x, scratch, ext);
            return (reg, reg);
        }
        let x_fast = self.op_ready(x, ext);
        if !x_fast && self.op_in_scratch(x, C, ext) {
            return (self.resolve_op(x, C, ext), self.resolve_op(y, A, ext));
        }
        if !x_fast && self.op_in_scratch(y, A, ext) {
            let right = self.resolve_op(y, A, ext);
            return (self.resolve_op(x, C, ext), right);
        }
        (self.resolve_op(x, A, ext), self.resolve_op(y, C, ext))
    }

    pub(super) fn bits_pair(&mut self, x: Value, y: Value) -> (Reg, Reg) {
        self.resolve_pair(x, y, None)
    }

    pub(super) fn src_pair(&mut self, x: Value, y: Value, ext: Ext) -> (Reg, Reg) {
        self.resolve_pair(x, y, Some(ext))
    }

    pub(super) fn src(&mut self, value: Value, scratch: Reg, ext: Ext) -> Reg {
        if self.is_full(value, ext)
            && let Some(reg) = self
                .plan
                .reg_of(value)
                .or_else(|| self.spill_borrow(value, scratch))
        {
            return reg;
        }
        if let Location::Imm(imm) = self.plan.location(value) {
            let ext_imm = crate::fold::extend(imm, self.ty(value), matches!(ext, Ext::Sign));
            if let Some(reg) = self.cached_loop_const(ext_imm) {
                return reg;
            }
        }
        self.get(scratch, value, ext);
        scratch
    }

    pub(super) fn bits(&mut self, value: Value, scratch: Reg) -> Reg {
        if let Some(reg) = self.plan.reg_of(value) {
            return reg;
        }
        if let Some(reg) = self.spill_borrow(value, scratch) {
            return reg;
        }
        if let Location::Imm(imm) = self.plan.location(value)
            && let Some(reg) = self.cached_loop_const(imm)
        {
            return reg;
        }
        self.raw(scratch, value);
        scratch
    }

    pub(super) fn put(&mut self, value: Value, reg: Reg) {
        if reg != super::ZR {
            self.asm.clobber_anchor(reg);
        }
        match self.plan.location(value) {
            Location::Reg(location) => {
                if location != reg {
                    self.asm.mov(location, reg);
                }
                self.clobber_reg(location);
            }
            Location::Float(location) => {
                self.clobber_vreg(Vreg(location));
                self.asm
                    .fmov_in(self.ty(value).bytes() == 8, Vreg(location), reg);
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

    fn fget_raw(&mut self, reg: Vreg, value: Value, ty: Type) {
        self.raw(IP, value);
        self.asm.fmov_in(ty.bytes() == 8, reg, IP);
        self.clobber_vreg(reg);
    }

    fn fget_slot(&mut self, reg: Vreg, value: Value, ty: Type) {
        let disp = self.frame.values[value.index()];
        debug_assert_ne!(disp, 0, "unhomed value {value:?} in fget()");
        if let Some(cached_reg) = self.spills.cached_fspill(disp, ty) {
            self.asm.vmov(reg, cached_reg);
            if cached_reg != reg {
                self.clobber_vreg(reg);
            }
            return;
        }
        if self
            .const_bits(value)
            .is_some_and(|imm| self.try_emit_float_imm_bits(ty, imm, reg))
        {
            self.spills.fspill = Some((disp, reg, ty));
            return;
        }
        if !ty.is_vector()
            && matches!(ty.bytes(), 4 | 8)
            && let Some(cached) = self.cached_spill_for(disp, super::ZR)
        {
            self.asm.fmov_in(ty.bytes() == 8, reg, cached);
            self.clobber_vreg(reg);
            self.spills.fspill = Some((disp, reg, ty));
            return;
        }
        let at = self.frame_mem(disp);
        self.asm.vload(ty.bytes(), reg, at);
        self.clobber_vreg(reg);
        self.spills.fspill = Some((disp, reg, ty));
    }

    pub(super) fn fget(&mut self, reg: Vreg, value: Value) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Float(from) if Vreg(from) == reg => {}
            Location::Float(from) => {
                self.asm.vmov(reg, Vreg(from));
                self.clobber_vreg(reg);
            }
            Location::Reg(from) => {
                self.asm.fmov_in(ty.bytes() == 8, reg, from);
                self.clobber_vreg(reg);
            }
            Location::Imm(imm) => {
                if !self.try_emit_float_imm_bits(ty, imm, reg) {
                    self.fget_raw(reg, value, ty);
                }
            }
            Location::Slot => self.fget_slot(reg, value, ty),
            _ => self.fget_raw(reg, value, ty),
        }
    }

    pub(super) fn fput(&mut self, value: Value, reg: Vreg) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Float(to) => {
                self.asm.vmov(Vreg(to), reg);
                self.clobber_vreg(Vreg(to));
            }
            _ => {
                let disp = self.frame.values[value.index()];
                self.asm.vstore(ty.bytes(), self.at(value), reg);
                self.spills.write_fspill(disp, reg, ty);
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
                self.finish_slot_store(Some(disp), ty, IP);
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

    fn finish_slot_store(&mut self, slot_disp: Option<i32>, ty: Type, reg: Reg) {
        if let Some(d) = slot_disp {
            if ty.bytes() <= 8 {
                self.write_spill(d, reg);
            } else {
                self.invalidate_disp(d);
            }
        }
    }

    pub(super) fn pair_disp_ok(disp: i32) -> bool {
        disp % 8 == 0 && (-512..=504).contains(&disp)
    }

    fn store_zero(&mut self, to: Mem, slot_disp: Option<i32>, ty: Type) {
        if ty.bytes() == 16 {
            if Self::pair_disp_ok(to.disp) {
                self.asm.stp_offset(super::ZR, super::ZR, to.base, to.disp);
            } else {
                self.asm.store(Size::X, to, super::ZR);
                self.asm.store(Size::X, to.offset(8), super::ZR);
            }
        } else {
            self.asm.store(Size::X, to, super::ZR);
        }
        self.finish_slot_store(slot_disp, ty, super::ZR);
    }

    fn move_slot_to_slot(&mut self, to: Mem, slot_disp: Option<i32>, value: Value, ty: Type) {
        let src_disp = self.frame.values[value.index()];
        if ty.bytes() <= 8
            && let Some(cached) = self.cached_spill_for(src_disp, super::ZR)
        {
            self.asm.store(Size::X, to, cached);
            if let Some(d) = slot_disp {
                self.write_spill(d, cached);
            }
        } else if let Some(cr) = self.spills.cached_fspill(src_disp, ty) {
            self.asm.vstore(ty.bytes(), to, cr);
            if let Some(d) = slot_disp {
                self.spills.write_fspill(d, cr, ty);
            }
        } else if self.const_bits(value) == Some(0) {
            self.store_zero(to, slot_disp, ty);
        } else {
            self.copy_to(to, self.at(value), ty);
            self.finish_slot_store(slot_disp, ty, IP);
        }
    }

    pub(super) fn move_to_slot(&mut self, to: Mem, slot_disp: Option<i32>, value: Value) {
        let ty = self.ty(value);
        match self.plan.location(value) {
            Location::Imm(0) => self.store_zero(to, slot_disp, ty),
            Location::Imm(_)
            | Location::Frame(..)
            | Location::SlotAddr(_)
            | Location::Symbol(_)
            | Location::Offset(..) => {
                self.raw(IP, value);
                self.asm.store(Size::X, to, IP);
                self.finish_slot_store(slot_disp, ty, IP);
            }
            Location::Reg(reg) => {
                self.asm.store(Size::X, to, reg);
                self.finish_slot_store(slot_disp, ty, reg);
            }
            Location::Float(v) => {
                self.asm.vstore(ty.bytes(), to, Vreg(v));
                if let Some(d) = slot_disp {
                    self.spills.write_fspill(d, Vreg(v), ty);
                }
            }
            Location::Slot => self.move_slot_to_slot(to, slot_disp, value, ty),
        }
    }
}
