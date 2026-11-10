use super::asm::{Alu, Cond, Group3, Mem, Reg, Size};
use super::{Emit, Ext};
use crate::ir::{AtomicRmwOp, Op, StackSlot, Type, Value};

const MFENCE: [u8; 3] = [0x0F, 0xAE, 0xF0];

/// Scratch registers an inline copy holds its chunks in; none of them forms an address.
const COPY_SCRATCH: [Reg; 4] = [Reg::Rax, Reg::Rdx, Reg::R11, Reg::Rsi];

impl Emit<'_> {
    pub(super) fn memory_op(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Const(imm) => {
                let result = result()?;
                if !self.is_static(result) {
                    let dst = self.dst(result);
                    self.asm.mov_imm(dst, *imm);
                    self.put(result, dst);
                }
            }
            Op::Select(condition, x, y) => self.select(*condition, *x, *y, result()?),
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
        let source = self.reg_of(value).unwrap_or(Reg::Rax);
        self.raw(source, value);
        if ty == Type::I128 {
            self.asm.store(Size::Q, to, Reg::Rax);
            self.high(Reg::Rax, value);
            self.asm.store(Size::Q, to.offset(8), Reg::Rax);
            return;
        }
        self.asm.store(Size::of(ty.bytes()), to, source);
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
