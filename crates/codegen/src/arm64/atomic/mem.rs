use super::super::asm::{FP, Mem, Reg, Size, ZR};
use super::super::{A, C, D, Emit, IP, S, T, V0, V1};
use crate::ir::{AtomicOrd, Op, Type, Value};
use crate::reloc::{RelocKind, StencilReloc};

const COPY_SCRATCH: [Reg; 4] = [A, D, S, T];

impl Emit<'_> {
    pub(in crate::arm64) fn memory_op(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<(), String> {
        match op {
            Op::Const(0) => {
                let result = result()?;
                if !self.plan.is_static(result) {
                    self.put(result, ZR);
                }
            }
            Op::Const(imm) => {
                let result = result()?;
                if !self.plan.is_static(result) {
                    let dst = self.dst(result);
                    self.asm.mov_imm(dst, *imm);
                    self.put(result, dst);
                }
            }
            Op::SlotAddr(sym) | Op::Symbol(sym) => {
                let result = result()?;
                if !self.plan.is_static(result) && self.plan.through[result.index()].is_none() {
                    let dst = self.dst(result);
                    if self.direct_calls {
                        let at = self.asm.adrp_add_reloc(dst);
                        self.relocs
                            .push(StencilReloc::symbol(at, *sym, RelocKind::Arm64AdrpAdd));
                    } else {
                        let at = self.asm.mov_symbol(dst);
                        self.relocs
                            .push(StencilReloc::symbol(at, *sym, RelocKind::Arm64Movw64));
                    }
                    self.put(result, dst);
                }
            }
            Op::Runtime(rt) => {
                let result = result()?;
                if !self.plan.is_static(result) && self.plan.through[result.index()].is_none() {
                    let dst = self.dst(result);
                    let (at, kind) = if self.direct_calls {
                        (self.asm.adrp_add_reloc(dst), RelocKind::Arm64AdrpAdd)
                    } else {
                        (self.asm.mov_symbol(dst), RelocKind::Arm64Movw64)
                    };
                    self.relocs.push(StencilReloc::runtime(at, *rt, kind));
                    self.put(result, dst);
                }
            }
            Op::Select(condition, x, y) => self.select(*condition, *x, *y, result()?),
            _ => return self.access(op, result),
        }
        Ok(())
    }

    fn access(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<(), String> {
        match op {
            Op::Load(addr, offset) => self.load_at(result()?, *addr, *offset),
            Op::VolatileLoad(addr, offset) => {
                let from = self.address(*addr, *offset);
                self.read(result()?, from);
            }
            Op::Store(value, addr, offset) => self.store_at(*value, *addr, *offset),
            Op::VolatileStore(value, addr, offset) => {
                let to = self.address(*addr, *offset);
                self.write(*value, to);
            }
            Op::StackAddr(slot, offset) => {
                let result = result()?;
                if !self.plan.is_static(result) {
                    let dst = self.dst(result);
                    self.slot_address(dst, *slot, *offset);
                    self.put(result, dst);
                }
            }
            Op::StackLoad(slot, offset) => self.read(result()?, self.slot(*slot, *offset)),
            Op::StackStore(value, slot, offset) => self.write(*value, self.slot(*slot, *offset)),
            Op::Fence(ord) => match ord {
                AtomicOrd::Relaxed => {}
                AtomicOrd::Acquire => self.asm.dmb_ld(),
                AtomicOrd::Isb => self.asm.word(0xD503_3FDF),
                _ => self.asm.dmb(),
            },
            Op::FramePointer => self.put(result()?, FP),
            Op::Copy(to, from, bytes) => self.copy(*to, *from, *bytes)?,
            Op::Move(to, from, bytes) => self.move_bytes(*to, *from, *bytes),
            _ => return self.atomic(op, result),
        }
        Ok(())
    }

    fn load_gp(&mut self, dst: Reg, ty: Type, signed: bool, op: &Op) {
        let size = Size::of(ty.bytes());
        match *op {
            Op::Load(addr, off) => {
                if let Some((base, index, shift)) = self.sib_indexed(addr, off, ty.bytes()) {
                    self.asm.load_indexed(size, signed, dst, base, index, shift);
                } else {
                    let from = self.address(addr, off);
                    self.asm.load(size, signed, dst, from);
                }
            }
            Op::StackLoad(slot, off) => self.asm.load(size, signed, dst, self.slot(slot, off)),
            _ => {}
        }
    }

    pub(in crate::arm64) fn load_signed(&mut self, result: Value, src_ty: Type, op: &Op) {
        let dst = self.dst(result);
        self.load_gp(dst, src_ty, true, op);
        self.put(result, dst);
    }

    pub(in crate::arm64) fn load_signed_reg(&mut self, dst: Reg, src_ty: Type, op: &Op) {
        self.load_gp(dst, src_ty, true, op);
        self.clobber_reg(dst);
    }

    pub(super) fn load_at(&mut self, result: Value, addr: Value, offset: i32) {
        if self.plan.is_static(result) {
            return;
        }
        let ty = self.ty(result);
        if ty != Type::I128 && !ty.in_float_register() {
            let dst = self.dst(result);
            self.load_gp(dst, ty, false, &Op::Load(addr, offset));
            self.put(result, dst);
            return;
        }
        if ty != Type::I128
            && let Some((base, index, shift)) = self.sib_indexed(addr, offset, ty.bytes())
        {
            let dst = self.fdst(result);
            self.asm.vload_indexed(ty.bytes(), dst, base, index, shift);
            self.fput(result, dst);
            return;
        }
        let from = self.address(addr, offset);
        self.read(result, from);
    }

    pub(super) fn store_at(&mut self, value: Value, addr: Value, offset: i32) {
        let ty = self.ty(value);
        if ty != Type::I128
            && let Some((base, index, shift)) = self.sib_indexed(addr, offset, ty.bytes())
        {
            if ty.in_float_register() {
                let source = self.fin(value, V0);
                self.asm
                    .vstore_indexed(ty.bytes(), base, index, shift, source);
            } else {
                let source = self.store_src(value);
                self.asm
                    .store_indexed(Size::of(ty.bytes()), base, index, shift, source);
            }
            return;
        }
        let to = self.address(addr, offset);
        self.write(value, to);
    }

    fn move_bytes(&mut self, to: Value, from: Value, bytes: u32) {
        let source = self.address_in(from, 0, S);
        let target = self.address_in(to, 0, C);
        let qpair_ok = |d: i32| d % 16 == 0 && (-1024..=1008).contains(&d);
        let mut at = 0_u32;
        let mut used_v0 = false;
        let mut used_v1 = false;
        while at + 32 <= bytes
            && let (Some(sd), Some(td)) = (
                source.disp.checked_add(at as i32),
                target.disp.checked_add(at as i32),
            )
            && qpair_ok(sd)
            && qpair_ok(td)
        {
            self.asm.qldp_offset(V0, V1, source.base, sd);
            self.asm.qstp_offset(V0, V1, target.base, td);
            used_v0 = true;
            used_v1 = true;
            at += 32;
        }
        while at + 16 <= bytes {
            self.asm.vload(16, V1, source.offset(at as i32));
            self.asm.vstore(16, target.offset(at as i32), V1);
            used_v1 = true;
            at += 16;
        }
        for (ty, off) in crate::copy::words(bytes - at) {
            let off = (at + off) as i32;
            self.asm
                .load(Size::of(ty.bytes()), false, A, source.offset(off));
            self.asm.store(Size::of(ty.bytes()), target.offset(off), A);
        }
        if used_v0 {
            self.clobber_vreg(V0);
        }
        if used_v1 {
            self.clobber_vreg(V1);
        }
        self.clobber_regs(&[A, C, S]);
    }

    fn copy(&mut self, to: Value, from: Value, bytes: Value) -> Result<(), String> {
        let chunks = self
            .plan
            .const_of(bytes)
            .and_then(crate::copy::chunks)
            .ok_or("a copy of a size that is not a small constant")?;
        if chunks.is_empty() {
            return Ok(());
        }
        let source = self.address(from, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm
                .load(Size::of(width), false, reg, source.offset(offset as i32));
        }
        self.clobber_regs(&COPY_SCRATCH[..chunks.len()]);
        let target = self.address(to, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm
                .store(Size::of(width), target.offset(offset as i32), reg);
        }
        Ok(())
    }

    fn read(&mut self, result: Value, from: Mem) {
        if self.plan.is_static(result) {
            return;
        }
        let ty = self.ty(result);
        if ty.in_float_register() {
            let dst = self.fdst(result);
            self.asm.vload(ty.bytes(), dst, from);
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            let hi = if from.base == C { IP } else { C };
            if Self::pair_disp_ok(from.disp) {
                self.asm.ldp_offset(A, hi, from.base, from.disp);
            } else {
                self.asm.load(Size::X, false, hi, from.offset(8));
                self.asm.load(Size::X, false, A, from);
            }
            self.clobber_regs(&[A, hi]);
            self.put(result, A);
            self.put_high(result, hi);
            return;
        }
        let dst = self.dst(result);
        self.asm.load(Size::of(ty.bytes()), false, dst, from);
        self.put(result, dst);
    }

    fn write(&mut self, value: Value, to: Mem) {
        let ty = self.ty(value);
        if ty.in_float_register() {
            if self.vreg_of(value).is_none() {
                self.clobber_vreg(V0);
            }
            let source = self.fin(value, V0);
            self.asm.vstore(ty.bytes(), to, source);
            return;
        }
        if ty == Type::I128 {
            let (lo, hi) = if self.plan.const_of(value) == Some(0) {
                (ZR, ZR)
            } else {
                let hi = if to.base == C { IP } else { C };
                self.raw(A, value);
                self.high(hi, value);
                (A, hi)
            };
            if Self::pair_disp_ok(to.disp) {
                self.asm.stp_offset(lo, hi, to.base, to.disp);
            } else {
                self.asm.store(Size::X, to, lo);
                self.asm.store(Size::X, to.offset(8), hi);
            }
            return;
        }
        let source = self.store_src(value);
        self.asm.store(Size::of(ty.bytes()), to, source);
    }

    pub(super) fn store_src(&mut self, value: Value) -> Reg {
        match self.plan.const_of(value) {
            Some(0) => ZR,
            _ => {
                let source = super::super::fuse::store_source(self.func, &self.plan, value, |v| {
                    self.defs[v.index()]
                });
                self.bits(source, A)
            }
        }
    }
}
