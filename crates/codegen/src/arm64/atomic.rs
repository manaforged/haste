use super::asm::{Alu, Cond, FP, Mem, Reg, Size, ZR};
use super::{A, C, D, Emit, S, T};
use crate::frame::Ext;
use crate::ir::{AtomicRmwOp, Op, Type, Value};
use crate::reloc::{RelocKind, StencilReloc};

const COPY_SCRATCH: [Reg; 4] = [A, D, S, T];

impl Emit<'_> {
    pub(super) fn memory_op(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
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
                if !self.plan.is_static(result) {
                    let dst = self.dst(result);
                    let at = self.asm.mov_symbol(dst);
                    self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Arm64Movw64));
                    self.put(result, dst);
                }
            }
            Op::Runtime(rt) => {
                let result = result()?;
                let dst = self.dst(result);
                let at = self.asm.mov_symbol(dst);
                self.relocs.push(StencilReloc::runtime(at, *rt, RelocKind::Arm64Movw64));
                self.put(result, dst);
            }
            Op::Select(condition, x, y) => self.select(*condition, *x, *y, result()?),
            _ => return self.access(op, result),
        }
        Ok(())
    }

    fn access(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Load(addr, offset) => self.load_at(result()?, *addr, *offset),
            Op::Store(value, addr, offset) => self.store_at(*value, *addr, *offset),
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
            Op::Fence => self.asm.dmb(),
            Op::FramePointer => self.put(result()?, FP),
            Op::Copy(to, from, bytes) => self.copy(*to, *from, *bytes)?,
            Op::Move(to, from, bytes) => self.move_bytes(*to, *from, *bytes),
            _ => return self.atomic(op, result),
        }
        Ok(())
    }

    fn load_at(&mut self, result: Value, addr: Value, offset: i32) {
        let ty = self.ty(result);
        if ty != Type::I128
            && let Some((base, index, shift)) = self.sib_indexed(addr, offset, ty.bytes())
        {
            if ty.is_float() {
                let dst = self.fdst(result);
                self.asm.vload_indexed(ty.bytes(), dst, base, index, shift);
                self.fput(result, dst);
            } else {
                let dst = self.dst(result);
                self.asm.load_indexed(Size::of(ty.bytes()), false, dst, base, index, shift);
                self.put(result, dst);
            }
            return;
        }
        let from = self.address(addr, offset);
        self.read(result, from);
    }

    fn store_at(&mut self, value: Value, addr: Value, offset: i32) {
        let ty = self.ty(value);
        if ty != Type::I128
            && let Some((base, index, shift)) = self.sib_indexed(addr, offset, ty.bytes())
        {
            if ty.is_float() {
                let source = self.fin(value, super::V0);
                self.asm.vstore_indexed(ty.bytes(), base, index, shift, source);
            } else {
                let source = self.store_src(value);
                self.asm.store_indexed(Size::of(ty.bytes()), base, index, shift, source);
            }
            return;
        }
        let to = self.address(addr, offset);
        self.write(value, to);
    }

    fn move_bytes(&mut self, to: Value, from: Value, bytes: u32) {
        let source = self.address_in(from, 0, S);
        let target = self.address_in(to, 0, C);
        let whole = bytes / 16 * 16;
        for at in (0..whole).step_by(16) {
            self.asm.vload(16, super::V1, source.offset(at as i32));
            self.asm.vstore(16, target.offset(at as i32), super::V1);
        }
        for (ty, at) in crate::copy::words(bytes - whole) {
            let at = (whole + at) as i32;
            self.asm.load(Size::of(ty.bytes()), false, A, source.offset(at));
            self.asm.store(Size::of(ty.bytes()), target.offset(at), A);
        }
    }

    fn copy(&mut self, to: Value, from: Value, bytes: Value) -> Result<(), String> {
        let chunks = self.plan.const_of(bytes).and_then(crate::copy::chunks).ok_or("a copy of a size that is not a small constant")?;
        let source = self.address(from, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.load(Size::of(width), false, reg, source.offset(offset as i32));
        }
        let target = self.address(to, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.store(Size::of(width), target.offset(offset as i32), reg);
        }
        Ok(())
    }

    fn read(&mut self, result: Value, from: Mem) {
        let ty = self.ty(result);
        if ty.is_float() {
            let dst = self.fdst(result);
            self.asm.vload(ty.bytes(), dst, from);
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.asm.load(Size::X, false, A, from);
            self.put(result, A);
            self.asm.load(Size::X, false, A, from.offset(8));
            self.put_high(result, A);
            return;
        }
        let dst = self.dst(result);
        self.asm.load(Size::of(ty.bytes()), false, dst, from);
        self.put(result, dst);
    }

    fn write(&mut self, value: Value, to: Mem) {
        let ty = self.ty(value);
        if ty.is_float() {
            let source = self.fin(value, super::V0);
            self.asm.vstore(ty.bytes(), to, source);
            return;
        }
        if ty == Type::I128 {
            self.raw(A, value);
            self.asm.store(Size::X, to, A);
            self.high(A, value);
            self.asm.store(Size::X, to.offset(8), A);
            return;
        }
        let source = self.store_src(value);
        self.asm.store(Size::of(ty.bytes()), to, source);
    }

    fn store_src(&mut self, value: Value) -> Reg {
        match self.plan.const_of(value) {
            Some(0) => ZR,
            _ => self.bits(value, A),
        }
    }

    fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        let cond = if let Some((val, c)) = self.flags
            && val == condition
        {
            c
        } else if self.plan.elided[condition.index()]
            && let Some(inst) = self.defs[condition.index()]
            && let Op::Icmp(cc, a, b) = self.func.insts[inst.index()].op
        {
            let c = self.compare(cc, a, b);
            self.flags = Some((condition, c));
            c
        } else {
            self.flags = None;
            self.last_cmp = None;
            let flag = self.src(condition, C, Ext::Zero);
            self.asm.add_imm(true, true, Size::X, ZR, flag, 0);
            Cond::Ne
        };
        let ty = self.ty(x);
        if ty.is_float() {
            let (left, right) = (self.fin(x, super::V0), self.fin(y, super::V1));
            let dst = self.fdst(result);
            self.asm.fcsel(cond, ty == Type::F64, dst, left, right);
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.raw(A, x);
            self.raw(D, y);
            self.asm.csel(cond, A, A, D);
            self.put(result, A);
            self.high(A, x);
            self.high(D, y);
            self.asm.csel(cond, A, A, D);
            self.put_high(result, A);
            return;
        }
        let (left, right) = (self.bits(x, A), self.bits(y, D));
        let dst = self.dst(result);
        self.asm.csel(cond, dst, left, right);
        self.put(result, dst);
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
                let at = self.bits(*addr, S);
                let dst = self.dst(result()?);
                self.asm.load_acquire(false, size, dst, at);
                self.put(result()?, dst);
            }
            Op::AtomicStore(value, addr) => {
                let at = self.bits(*addr, S);
                let source = self.bits(*value, A);
                self.asm.store_release(size, at, source);
            }
            Op::AtomicRmw(rmw, addr, value) => {
                self.rmw(*rmw, size, *addr, *value);
                self.put(result()?, A);
            }
            Op::AtomicCas(addr, expected, replacement) => {
                self.cas(size, *addr, (*expected, *replacement));
                self.put(result()?, A);
            }
            other => return Err(format!("unexpected {other:?}")),
        }
        Ok(())
    }

    fn rmw(&mut self, op: AtomicRmwOp, size: Size, addr: Value, value: Value) {
        let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
        let bytes = [1, 2, 4, 8][size as usize];
        self.raw(S, addr);
        self.get(D, value, Ext::of(signed));
        let again = self.asm.label();
        self.asm.bind(again);
        self.asm.load_acquire(true, size, A, S);
        let old = if signed {
            self.asm.extend(true, bytes, C, A);
            C
        } else {
            A
        };
        let choose = match op {
            AtomicRmwOp::Smin => Some(Cond::Lt),
            AtomicRmwOp::Smax => Some(Cond::Gt),
            AtomicRmwOp::Umin => Some(Cond::Lo),
            AtomicRmwOp::Umax => Some(Cond::Hi),
            _ => None,
        };
        match (choose, op) {
            (Some(cond), _) => {
                self.asm.alu(Alu::Subs, Size::X, ZR, old, D);
                self.asm.csel(cond, C, old, D);
            }
            (None, AtomicRmwOp::Xchg) => self.asm.mov(C, D),
            (None, AtomicRmwOp::Add) => self.asm.alu(Alu::Add, Size::X, C, A, D),
            (None, AtomicRmwOp::Sub) => self.asm.alu(Alu::Sub, Size::X, C, A, D),
            (None, AtomicRmwOp::Or) => self.asm.alu(Alu::Orr, Size::X, C, A, D),
            (None, AtomicRmwOp::Xor) => self.asm.alu(Alu::Eor, Size::X, C, A, D),
            (None, AtomicRmwOp::And) => self.asm.alu(Alu::And, Size::X, C, A, D),
            (None, _) => {
                self.asm.alu(Alu::And, Size::X, C, A, D);
                self.asm.alu(Alu::Orn, Size::X, C, ZR, C);
            }
        }
        self.asm.store_exclusive(size, T, S, C);
        self.asm.cbz(true, Size::W, T, again);
    }

    fn cas(&mut self, size: Size, addr: Value, (expected, replacement): (Value, Value)) {
        let bytes = [1, 2, 4, 8][size as usize];
        self.raw(S, addr);
        self.get(D, expected, Ext::Zero);
        self.raw(C, replacement);
        let (again, done) = (self.asm.label(), self.asm.label());
        self.asm.bind(again);
        self.asm.load_acquire(true, size, A, S);
        let compare = if bytes < 8 { Size::W } else { Size::X };
        self.asm.alu(Alu::Subs, compare, ZR, A, D);
        self.asm.bcond(Cond::Ne, done);
        self.asm.store_exclusive(size, T, S, C);
        self.asm.cbz(true, Size::W, T, again);
        self.asm.bind(done);
    }
}
