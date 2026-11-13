use super::asm::{Alu, Cond, Group3, Mem, Reg, Size, Xmm};
use super::float::X1;
use super::{Emit, Ext};
use crate::ir::{AtomicRmwOp, Binary, Function, Op, StackSlot, Type, Unary, Value};
use crate::reloc::{RelocKind, StencilReloc};

const MFENCE: [u8; 3] = [0x0F, 0xAE, 0xF0];

/// Scratch registers an inline copy holds its chunks in; none of them forms an address.
const COPY_SCRATCH: [Reg; 4] = [Reg::Rax, Reg::Rdx, Reg::R11, Reg::Rsi];

/// The bytes a move copies through one xmm register at a time.
const VECTOR: u32 = 16;

/// The opcodes of `movups` from memory into an xmm register, and from an xmm register into memory.
const MOVUPS: ([u8; 2], [u8; 2]) = ([0x0F, 0x10], [0x0F, 0x11]);

/// Whether `op` of `func` uses rsi as scratch: a copy of a constant size, a compare-exchange or read-modify-write, a population count, or a 128-bit shift or rotate.
pub(super) fn uses_rsi(func: &Function, op: &Op) -> bool {
    match op {
        Op::Copy(..) | Op::AtomicCas(..) | Op::AtomicRmw(..) | Op::Unary(Unary::Popcnt, _) => true,
        Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr, x, _) => func.ty(*x) == Type::I128,
        _ => false,
    }
}

impl Emit<'_> {
    pub(super) fn memory_op(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Const(_) if result().is_ok_and(|result| self.ty(result) == Type::V128) => self.vector_zero(result()?),
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
                    let at = self.asm.mov_slot_reloc(dst);
                    self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Abs32));
                    self.put(result, dst);
                }
            }
            Op::Symbol(sym) => {
                let result = result()?;
                let dst = self.dst(result);
                let at = self.asm.mov_abs64(dst);
                self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Abs64));
                self.put(result, dst);
            }
            Op::Runtime(rt) => {
                let result = result()?;
                let dst = self.dst(result);
                let at = self.asm.mov_abs64(dst);
                self.relocs.push(StencilReloc::runtime(at, *rt, RelocKind::Abs64));
                self.put(result, dst);
            }
            Op::Select(condition, x, y) => self.select(*condition, *x, *y, result()?),
            Op::ByteCmp(..) | Op::Splat(_) | Op::Movemask(_) | Op::Vector(..) | Op::Sha256Rounds(..) => self.vector_op(op, result()?)?,
            _ => return self.access(op, result),
        }
        Ok(())
    }

    fn access(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Load(addr, offset) => self.load(*addr, *offset, result()?),
            Op::Store(value, addr, offset) => {
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
            Op::StackStore(value, slot, offset) => self.write(*value, self.slot(*slot, *offset)),
            Op::Fence => self.asm.bytes(&MFENCE),
            Op::FramePointer => self.put(result()?, Reg::Rbp),
            Op::Copy(to, from, bytes) => self.copy(*to, *from, *bytes)?,
            Op::Move(to, from, bytes) => self.move_bytes(*to, *from, *bytes),
            _ => return self.atomic(op, result),
        }
        Ok(())
    }

    /// Reads `result` from `addr + offset`, unless a call reads it in place.
    fn load(&mut self, addr: Value, offset: i32, result: Value) {
        if self.plan.through[result.index()].is_none() {
            let from = self.address(addr, offset);
            self.read(result, from);
        }
    }

    /// Copies a constant number of bytes inline: every chunk into its own scratch register first, so overlapping ranges copy correctly.
    fn copy(&mut self, to: Value, from: Value, bytes: Value) -> Result<(), String> {
        let chunks = self.const_of(bytes).and_then(crate::copy::chunks).ok_or("a copy of a size that is not a small constant")?;
        let source = self.address(from, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.load(Size::of(width), reg, source.offset(offset as i32));
        }
        let target = self.address(to, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.store(Size::of(width), target.offset(offset as i32), reg);
        }
        Ok(())
    }

    /// Copies `bytes` bytes between two ranges that are the same or disjoint: 16 at a time through xmm1, then word by word through rax; r11 and rcx hold addresses.
    fn move_bytes(&mut self, to: Value, from: Value, bytes: u32) {
        let source = self.address_in(from, 0, Reg::R11);
        let target = self.address_in(to, 0, Reg::Rcx);
        let whole = bytes / VECTOR * VECTOR;
        for at in (0..whole).step_by(VECTOR as usize) {
            self.asm.sse(&[], &MOVUPS.0, X1.0, source.offset(at as i32));
            self.asm.sse(&[], &MOVUPS.1, X1.0, target.offset(at as i32));
        }
        for (ty, at) in crate::copy::words(bytes - whole) {
            let at = (whole + at) as i32;
            self.asm.load(Size::of(ty.bytes()), Reg::Rax, source.offset(at));
            self.asm.store(Size::of(ty.bytes()), target.offset(at), Reg::Rax);
        }
    }

    fn atomic(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        let ty = match op {
            Op::AtomicStore(value, _) => self.ty(*value),
            _ => self.ty(result()?),
        };
        if ty == Type::I128 {
            return Err("128-bit atomics".to_owned());
        }
        let size = Size::of(ty.bytes());
        match op {
            Op::AtomicLoad(addr) => {
                self.raw(Reg::Rcx, *addr);
                self.read(result()?, Mem::new(Reg::Rcx, 0));
            }
            Op::AtomicStore(value, addr) => {
                self.raw(Reg::Rcx, *addr);
                self.raw(Reg::Rax, *value);
                self.asm.xchg(size, Mem::new(Reg::Rcx, 0), Reg::Rax);
            }
            Op::AtomicRmw(rmw, addr, value) => {
                self.rmw(*rmw, size, *addr, *value);
                self.put(result()?, Reg::Rax);
            }
            Op::AtomicCas(addr, expected, replacement) => {
                self.raw(Reg::Rsi, *addr);
                self.raw(Reg::Rax, *expected);
                self.raw(Reg::Rcx, *replacement);
                self.asm.cmpxchg(size, Mem::new(Reg::Rsi, 0), Reg::Rcx);
                self.put(result()?, Reg::Rax);
            }
            other => return Err(format!("unexpected {other:?}")),
        }
        Ok(())
    }

    pub(super) fn slot(&self, slot: StackSlot, offset: i32) -> Mem {
        Mem::new(Reg::Rbp, self.frame.slots[slot.index()] + offset)
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
        if let Some(imm) = self.const_of(value).and_then(|imm| if ty.bytes() < 8 { Some(imm as i32) } else { i32::try_from(imm).ok() }) {
            self.asm.store_imm_sized(size, to, imm);
            return;
        }
        let source = self.reg_of(value).unwrap_or(Reg::Rax);
        self.raw(source, value);
        if ty == Type::I128 {
            self.asm.store(Size::Q, to, Reg::Rax);
            self.high(Reg::Rax, value);
            self.asm.store(Size::Q, to.offset(8), Reg::Rax);
            return;
        }
        self.asm.store(size, to, source);
    }

    fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        self.get(Reg::Rcx, condition, Ext::Zero);
        self.asm.test(Reg::Rcx, Reg::Rcx);
        if self.ty(x) == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, self.at(y));
            self.asm.cmov(Cond::Ne, Reg::Rax, self.at(x));
            self.put(result, Reg::Rax);
            self.asm.load(Size::Q, Reg::Rax, self.at(y).offset(8));
            self.asm.cmov(Cond::Ne, Reg::Rax, self.at(x).offset(8));
            self.put_high(result, Reg::Rax);
            return;
        }
        self.raw(Reg::Rax, y);
        let source = self.operand(x);
        self.asm.cmov(Cond::Ne, Reg::Rax, source);
        self.put(result, Reg::Rax);
    }

    /// Leaves the old contents in rax.
    fn rmw(&mut self, op: AtomicRmwOp, size: Size, addr: Value, value: Value) {
        let target = Mem::new(Reg::Rcx, 0);
        match op {
            AtomicRmwOp::Xchg | AtomicRmwOp::Add | AtomicRmwOp::Sub => {
                self.raw(Reg::Rcx, addr);
                self.raw(Reg::Rax, value);
                if op == AtomicRmwOp::Xchg {
                    self.asm.xchg(size, target, Reg::Rax);
                    return;
                }
                if op == AtomicRmwOp::Sub {
                    self.asm.group3(Group3::Neg, Reg::Rax);
                }
                self.asm.xadd(size, target, Reg::Rax);
            }
            _ => self.rmw_loop(op, size, addr, value),
        }
    }

    fn rmw_loop(&mut self, op: AtomicRmwOp, size: Size, addr: Value, value: Value) {
        let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
        let target = Mem::new(Reg::Rsi, 0);
        self.raw(Reg::Rsi, addr);
        self.get(Reg::Rdx, value, if signed { Ext::Sign } else { Ext::Zero });
        self.asm.load(size, Reg::Rax, target);
        let again = self.asm.label();
        self.asm.bind(again);
        if signed {
            self.asm.load_signed(size, Reg::Rcx, Reg::Rax);
        } else {
            self.asm.load(size, Reg::Rcx, Reg::Rax);
        }
        let choose = match op {
            AtomicRmwOp::Smin => Some(Cond::G),
            AtomicRmwOp::Smax => Some(Cond::L),
            AtomicRmwOp::Umin => Some(Cond::A),
            AtomicRmwOp::Umax => Some(Cond::B),
            _ => None,
        };
        match (choose, op) {
            (Some(cond), _) => {
                self.asm.alu(Alu::Cmp, Reg::Rcx, Reg::Rdx);
                self.asm.cmov(cond, Reg::Rcx, Reg::Rdx);
            }
            (None, AtomicRmwOp::Or) => self.asm.alu(Alu::Or, Reg::Rcx, Reg::Rdx),
            (None, AtomicRmwOp::Xor) => self.asm.alu(Alu::Xor, Reg::Rcx, Reg::Rdx),
            (None, _) => {
                self.asm.alu(Alu::And, Reg::Rcx, Reg::Rdx);
                if op == AtomicRmwOp::Nand {
                    self.asm.group3(Group3::Not, Reg::Rcx);
                }
            }
        }
        self.asm.cmpxchg(size, target, Reg::Rcx);
        self.asm.jcc(Cond::Ne, again);
    }
}
