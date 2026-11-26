mod select;

use super::Emit;
use super::asm::{Mem, Reg, Size, Xmm};
use super::float::X1;
use crate::alloc::Location;
use crate::frame::{Emitter, Ext};
use crate::ir::{
    AtomicRmwOp, Binary, Block, Function, Op, ShuffleMask, StackSlot, TargetOpId, Type, Unary,
    Value,
};
use crate::reloc::{RelocKind, StencilReloc};

const SEQ_CST_FENCE: [u8; 6] = [0xF0, 0x48, 0x83, 0x0C, 0x24, 0x00];

const COPY_SCRATCH: [Reg; 4] = [Reg::Rax, Reg::Rdx, Reg::R11, Reg::Rsi];

const VECTOR: u32 = 16;

const MOVUPS: ([u8; 2], [u8; 2]) = ([0x0F, 0x10], [0x0F, 0x11]);

/// Bitmask of `[Rsi, Rdx, R11]` scratch registers overwritten by `op` of `func`.
pub(super) fn scratch_mask(func: &Function, op: &Op) -> u8 {
    const RSI: u8 = 1;
    const RDX: u8 = 2;
    const R11: u8 = 4;
    const ALL: u8 = RSI | RDX | R11;
    match op {
        Op::Unary(_, x) | Op::Binary(_, x, _) | Op::Icmp(_, x, _) if func.ty(*x) == Type::I128 => {
            ALL
        }
        Op::Copy(..) | Op::Fits(Binary::Imul, ..) | Op::Binary(Binary::Fcopysign, ..) => ALL,
        Op::TargetOp(
            TargetOpId::X64AddCarry
            | TargetOpId::X64AddCarryOut
            | TargetOpId::X64SubBorrow
            | TargetOpId::X64SubBorrowOut
            | TargetOpId::X64DivWide
            | TargetOpId::X64RemWide,
            _,
        ) => ALL,
        Op::Move(..) | Op::AtomicCas(..) => R11,
        Op::AtomicRmw(AtomicRmwOp::Xchg | AtomicRmwOp::Add | AtomicRmwOp::Sub, ..) => 0,
        Op::AtomicRmw(..) => RSI | RDX,
        Op::Binary(
            Binary::Sdiv
            | Binary::Srem
            | Binary::Udiv
            | Binary::Urem
            | Binary::Smulhi
            | Binary::Umulhi
            | Binary::AvgRound
            | Binary::UaddSat
            | Binary::UsubSat,
            ..,
        ) => RDX,
        Op::Binary(Binary::SaddSat | Binary::SsubSat, ..) => RDX | R11,
        Op::Unary(Unary::Bitrev, _) => RDX,
        Op::Unary(Unary::Floor | Unary::Ceil | Unary::Trunc, x) if func.ty(*x).is_float() => RDX,
        Op::Unary(Unary::Popcnt, _) => RSI | R11,
        _ => 0,
    }
}

impl Emit<'_> {
    pub(super) fn invalidate_spills(&mut self) {
        self.spills.clear();
    }

    pub(super) fn clobber_call_spills(&mut self) {
        self.clobber_regs(&[
            Reg::Rax,
            Reg::Rcx,
            Reg::Rdx,
            Reg::Rsi,
            Reg::Rdi,
            Reg::R8,
            Reg::R9,
            Reg::R10,
            Reg::R11,
        ]);
        self.spills.clear_fspill();
    }

    pub(super) fn clobber_reg(&mut self, reg: Reg) {
        self.spills.clobber_reg(reg);
    }

    pub(super) fn clobber_regs(&mut self, regs: &[Reg]) {
        regs.iter().for_each(|&reg| self.clobber_reg(reg));
    }

    pub(super) fn clobber_xmm(&mut self, xmm: Xmm) {
        self.spills.clobber_freg(xmm);
    }

    pub(super) fn clobber_xmms(&mut self, xmms: &[Xmm]) {
        xmms.iter().for_each(|&xmm| self.clobber_xmm(xmm));
    }

    pub(super) fn invalidate_disp(&mut self, disp: i32) {
        self.spills.invalidate_disp(disp);
    }

    pub(super) fn resume_spills(&mut self, block: Block) {
        self.spills.resume(block);
    }

    pub(super) fn record_spill(&mut self, disp: i32, reg: Reg) {
        if reg != Reg::R11 {
            self.spills.record(disp, reg);
        }
    }

    pub(super) fn write_spill(&mut self, disp: i32, reg: Reg) {
        self.invalidate_disp(disp);
        self.record_spill(disp, reg);
    }

    pub(super) fn cached_spill_for(&self, disp: i32, preferred: Reg) -> Option<Reg> {
        self.spills.cached_for(disp, preferred, None)
    }

    pub(super) fn is_fused_cas(&self, v: Value) -> bool {
        self.plan.fused[v.index()]
            && self.defs[v.index()]
                .is_some_and(|i| matches!(self.func.insts[i.index()].op, Op::AtomicCas(..)))
    }

    pub(super) fn memory_op(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<(), String> {
        match op {
            Op::Const(_) if result().is_ok_and(|result| self.ty(result).is_vector()) => {
                self.vector_zero(result()?)
            }
            Op::Const(0) if result().is_ok_and(|result| self.ty(result).is_float()) => {
                let result = result()?;
                if !self.is_static(result) {
                    let dst = self.fdst(result);
                    self.zero_xmm(dst);
                    self.fput(result, dst);
                }
            }
            Op::Const(imm) => {
                let result = result()?;
                if !self.is_static(result) {
                    let dst = self.dst(result);
                    self.asm.mov_imm(dst, *imm);
                    self.put(result, dst);
                }
            }
            Op::SlotAddr(sym) => {
                let result = result()?;
                if !self.is_static(result) {
                    let dst = self.dst(result);
                    if self.direct_calls {
                        self.emit_rip_symbol(dst, *sym);
                    } else {
                        let at = self.asm.mov_slot_reloc(dst);
                        self.relocs
                            .push(StencilReloc::symbol(at, *sym, RelocKind::Abs32));
                    }
                    self.put(result, dst);
                }
            }
            Op::Symbol(sym) => {
                let result = result()?;
                if !self.is_static(result) && self.plan.through[result.index()].is_none() {
                    let dst = self.dst(result);
                    if self.direct_calls && self.func.is_image_addr(result) {
                        self.emit_rip_symbol(dst, *sym);
                    } else {
                        let at = self.asm.mov_abs64(dst);
                        self.relocs
                            .push(StencilReloc::symbol(at, *sym, RelocKind::Abs64));
                    }
                    self.put(result, dst);
                }
            }
            Op::Runtime(_) if self.plan.through[result()?.index()].is_some() => {}
            Op::Runtime(rt) => {
                let result = result()?;
                let dst = self.dst(result);
                let at = self.asm.mov_abs64(dst);
                self.relocs
                    .push(StencilReloc::runtime(at, *rt, RelocKind::Abs64));
                self.put(result, dst);
            }
            Op::Select(condition, x, y) => self.select(*condition, *x, *y, result()?),
            Op::Shuffle(..)
            | Op::Extract(..)
            | Op::Insert(..)
            | Op::Reduce(..)
            | Op::TargetOp(..) => {
                self.emit_vector_op(op, result()?)?;
            }
            _ => return self.access(op, result),
        }
        Ok(())
    }

    fn emit_vector_op(&mut self, op: &Op, result: Value) -> Result<(), String> {
        self.clear_flags();
        self.spills.clear_fspill();
        match op {
            Op::TargetOp(..) | Op::Extract(..) => self.vector_op(op, result),
            Op::Insert(..) => {
                self.vector_op(op, result)?;
                self.clobber_reg(Reg::Rax);
                Ok(())
            }
            Op::Shuffle(a, _, ShuffleMask::Const(_))
                if self.ty(*a).is_wide_vec() || self.features.ssse3 =>
            {
                self.vector_op(op, result)?;
                self.clobber_reg(Reg::Rax);
                Ok(())
            }
            _ => {
                self.invalidate_spills();
                self.vector_op(op, result)?;
                self.invalidate_spills();
                Ok(())
            }
        }
    }

    fn access(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<(), String> {
        match op {
            Op::Load(addr, offset) => self.load(*addr, *offset, result()?),
            Op::VolatileLoad(addr, offset) => {
                let from = self.address(*addr, *offset);
                self.read(result()?, from);
            }
            Op::Store(value, addr, offset) => {
                let to = self.address(*addr, *offset);
                if self.plan.elided[value.index()] && !self.has_elided_mem(*value) {
                    self.emit_mem_rmw(*value, to);
                } else {
                    self.write(*value, to);
                }
            }
            Op::VolatileStore(value, addr, offset) => {
                let to = self.address(*addr, *offset);
                self.write(*value, to);
            }
            Op::StackAddr(slot, offset) => {
                let result = result()?;
                if !self.is_static(result) {
                    let dst = self.dst(result);
                    self.asm.lea(dst, self.slot(*slot, *offset));
                    self.put(result, dst);
                }
            }
            Op::StackLoad(slot, offset) => self.read(result()?, self.slot(*slot, *offset)),
            Op::StackStore(value, slot, offset) => {
                let to = self.slot(*slot, *offset);
                if self.plan.elided[value.index()] && !self.has_elided_mem(*value) {
                    self.emit_mem_rmw(*value, to);
                } else {
                    self.write(*value, to);
                }
            }
            Op::Fence(ord) => {
                if ord.is_seq_cst() {
                    self.clear_flags();
                    self.asm.bytes(&SEQ_CST_FENCE);
                }
            }
            Op::FramePointer => self.put(result()?, Reg::Rbp),
            Op::Copy(to, from, bytes) => self.copy(*to, *from, *bytes)?,
            Op::Move(to, from, bytes) => self.move_bytes(*to, *from, *bytes),
            _ => return self.atomic(op, result),
        }
        Ok(())
    }

    fn try_load_rip(&mut self, addr: Value, result: Value) -> bool {
        if self.direct_calls
            && self.ty(result).is_small_int()
            && let Some(sym) = match self.location(addr) {
                Location::SlotAddr(s) => Some(s),
                Location::Symbol(s) if self.func.is_image_addr(addr) => Some(s),
                _ => None,
            }
        {
            let dst = self.dst(result);
            let at = self
                .asm
                .load_rip_reloc(Size::of(self.ty(result).bytes()), dst);
            self.relocs
                .push(StencilReloc::symbol(at, sym, RelocKind::X64RipRel32));
            self.put(result, dst);
            return true;
        }
        false
    }

    fn load(&mut self, addr: Value, offset: i32, result: Value) {
        if self.plan.through[result.index()].is_some()
            || (offset == 0 && self.try_load_rip(addr, result))
        {
            return;
        }
        let from = self.address(addr, offset);
        self.read(result, from);
    }

    fn copy(&mut self, to: Value, from: Value, bytes: Value) -> Result<(), String> {
        let chunks = self
            .const_of(bytes)
            .and_then(crate::copy::chunks)
            .ok_or("a copy of a size that is not a small constant")?;
        let source = self.address(from, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm
                .load(Size::of(width), reg, source.offset(offset as i32));
        }
        self.clobber_regs(&COPY_SCRATCH);
        let target = self.address(to, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm
                .store(Size::of(width), target.offset(offset as i32), reg);
        }
        Ok(())
    }

    fn move_bytes(&mut self, to: Value, from: Value, bytes: u32) {
        let source = self.address_in(from, 0, Reg::R11, Some(Reg::Rax));
        let target = self.address_in(to, 0, Reg::Rcx, Some(Reg::Rax));
        let mut at = 0_u32;
        if self.features.avx {
            while at + 32 <= bytes {
                self.ymm_load(X1, source.offset(at as i32));
                self.ymm_store(target.offset(at as i32), X1);
                at += 32;
            }
        }
        while at + VECTOR <= bytes {
            if self.features.avx {
                self.asm
                    .vex_xmm(0b00001, 0b00, 0x10, (X1, Xmm(0)), source.offset(at as i32));
                self.asm
                    .vex_xmm(0b00001, 0b00, 0x11, (X1, Xmm(0)), target.offset(at as i32));
            } else {
                self.asm.sse(&[], &MOVUPS.0, X1.0, source.offset(at as i32));
                self.asm.sse(&[], &MOVUPS.1, X1.0, target.offset(at as i32));
            }
            at += VECTOR;
        }
        for (ty, off) in crate::copy::words(bytes - at) {
            let off = (at + off) as i32;
            self.asm
                .load(Size::of(ty.bytes()), Reg::Rax, source.offset(off));
            self.asm
                .store(Size::of(ty.bytes()), target.offset(off), Reg::Rax);
        }
        self.clobber_regs(&[Reg::R11, Reg::Rcx, Reg::Rax]);
        self.clobber_xmm(X1);
    }

    fn atomic(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<(), String> {
        self.clear_flags();
        let ty = match op {
            Op::AtomicStore(value, _, _) => self.ty(*value),
            _ => self.ty(result()?),
        };
        if ty == Type::I128 {
            return Err("128-bit atomics".to_owned());
        }
        let size = Size::of(ty.bytes());
        match op {
            Op::AtomicLoad(addr, _) => {
                let res = result()?;
                if !self.try_load_rip(*addr, res) {
                    let target = self.atomic_target(*addr, Reg::Rcx, &[]);
                    self.read(res, target);
                }
            }
            Op::AtomicStore(value, addr, ord) => {
                if ord.is_seq_cst() {
                    let target = self.atomic_target(*addr, Reg::Rcx, &[Reg::Rax]);
                    self.raw(Reg::Rax, *value);
                    self.asm.flags_overwritten();
                    self.asm.xchg(size, target, Reg::Rax);
                    self.clobber_reg(Reg::Rax);
                } else {
                    let target = self.atomic_target(*addr, Reg::Rcx, &[]);
                    self.write(*value, target);
                }
            }
            Op::AtomicRmw(rmw, _, addr, value) => {
                let res = result()?;
                let out = if self.plan.fused[res.index()] && *rmw != AtomicRmwOp::Xchg {
                    self.lock_rmw(*rmw, size, *addr, *value);
                    Reg::Rax
                } else {
                    let out = self.rmw(*rmw, size, *addr, *value, self.reg_of(res));
                    if matches!(rmw, AtomicRmwOp::Xchg | AtomicRmwOp::Add | AtomicRmwOp::Sub) {
                        self.clobber_regs(&[out, Reg::Rcx]);
                    } else {
                        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi]);
                    }
                    out
                };
                if *rmw == AtomicRmwOp::Sub {
                    self.asm.hold_eq(res.index() as u32, value.index() as u32);
                }
                if !self.plan.fused[res.index()] {
                    self.put(res, out);
                }
            }
            Op::AtomicCas(_, addr, expected, replacement) => {
                let res = result()?;
                let (target, src) = match self
                    .reg_of(*replacement)
                    .filter(|&r| r != Reg::Rax && r != Reg::R11)
                {
                    Some(rep) => (self.atomic_target(*addr, Reg::R11, &[Reg::Rax, rep]), rep),
                    None => {
                        let t = self.atomic_target(*addr, Reg::R11, &[Reg::Rax, Reg::Rcx]);
                        self.raw(Reg::Rcx, *replacement);
                        (t, Reg::Rcx)
                    }
                };
                if size == Size::D {
                    self.get(Reg::Rax, *expected, Ext::Zero);
                } else {
                    self.raw(Reg::Rax, *expected);
                }
                self.asm.cmpxchg(size, target, src);
                self.asm
                    .hold_eq(res.index() as u32, expected.index() as u32);
                self.clobber_regs(if src == Reg::Rcx {
                    &[Reg::Rax, Reg::Rcx, Reg::R11]
                } else {
                    &[Reg::Rax, Reg::R11]
                });
                if !self.plan.fused[res.index()] {
                    self.put(res, Reg::Rax);
                }
            }
            other => return Err(format!("unexpected {other:?}")),
        }
        Ok(())
    }

    pub(super) fn slot(&self, slot: StackSlot, offset: i32) -> Mem {
        self.frame_mem(self.frame.slots[slot.index()] + offset)
    }

    fn read(&mut self, result: Value, from: Mem) {
        let ty = self.ty(result);
        if ty.in_float_register() {
            let dst = self.fdst(result);
            self.xmm_load(ty, dst, from);
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, from);
            self.put(result, Reg::Rax);
            self.asm.load(Size::Q, Reg::Rax, from.offset(8));
            self.put_high(result, Reg::Rax);
            self.clobber_reg(Reg::Rax);
            return;
        }
        let dst = self.dst(result);
        self.asm.load(Size::of(ty.bytes()), dst, from);
        self.put(result, dst);
    }

    fn write(&mut self, value: Value, to: Mem) {
        let ty = self.ty(value);
        if ty.in_float_register() {
            let source = self.fin(value, Xmm(0));
            self.xmm_store(ty, to, source);
            return;
        }
        let size = Size::of(ty.bytes());
        if let Some(imm) = self.const_of(value).and_then(|imm| {
            if ty.bytes() < 8 {
                Some(imm as i32)
            } else {
                i32::try_from(imm).ok()
            }
        }) {
            if imm == 0 && size == Size::Q {
                return self.zero_to(to);
            }
            self.asm.store_imm_sized(size, to, imm);
            return;
        }
        let source = self
            .reg_of(value)
            .or_else(|| {
                (ty != Type::I128)
                    .then(|| self.cached_spill(value))
                    .flatten()
            })
            .unwrap_or(Reg::Rax);
        self.raw(source, value);
        if ty == Type::I128 {
            self.asm.store(Size::Q, to, Reg::Rax);
            self.high(Reg::Rax, value);
            self.asm.store(Size::Q, to.offset(8), Reg::Rax);
            return;
        }
        self.asm.store(size, to, source);
    }
}
