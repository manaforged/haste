use super::asm::{Alu, Cond, FP, Mem, Reg, Size, ZR};
use super::{A, C, D, Emit, S, T};
use crate::frame::Ext;
use crate::ir::{AtomicRmwOp, Binary, Op, Type, Unary, Value};
use crate::reloc::{RelocKind, StencilReloc};

const COPY_SCRATCH: [Reg; 4] = [A, D, S, T];

#[derive(Clone, Copy)]
enum SelectOp {
    Inc,
    Inv,
    Neg,
}

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
                    if self.direct_calls {
                        let at = self.asm.adrp_add_reloc(dst);
                        self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Arm64AdrpAdd));
                    } else {
                        let at = self.asm.mov_symbol(dst);
                        self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Arm64Movw64));
                    }
                    self.put(result, dst);
                }
            }
            Op::Runtime(rt) => {
                let result = result()?;
                let dst = self.dst(result);
                let (at, kind) = if self.direct_calls {
                    (self.asm.adrp_add_reloc(dst), RelocKind::Arm64AdrpAdd)
                } else {
                    (self.asm.mov_symbol(dst), RelocKind::Arm64Movw64)
                };
                self.relocs.push(StencilReloc::runtime(at, *rt, kind));
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
            if ty.in_float_register() {
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
            if ty.in_float_register() {
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
        self.invalidate_spills();
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
        self.invalidate_spills();
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
        if ty.in_float_register() {
            let dst = self.fdst(result);
            self.asm.vload(ty.bytes(), dst, from);
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.invalidate_spills();
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
        if ty.in_float_register() {
            if self.vreg_of(value).is_none() {
                self.clobber_vreg(super::V0);
            }
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

    fn select_transform(&self, ty: Type, alt: Value, base: Value) -> Option<SelectOp> {
        if !matches!(ty, Type::I32 | Type::I64) {
            return None;
        }
        let op = &self.func.insts[self.defs[alt.index()]?.index()].op;
        let is_one = |v: Value| self.plan.const_of(v).is_some_and(|c| crate::fold::extend(c, ty, false) == 1);
        match *op {
            Op::Unary(Unary::Bnot, inner) if inner == base => Some(SelectOp::Inv),
            Op::Unary(Unary::Ineg, inner) if inner == base => Some(SelectOp::Neg),
            Op::Binary(Binary::Iadd, a, b) if (a == base && is_one(b)) || (b == base && is_one(a)) => Some(SelectOp::Inc),
            Op::Binary(Binary::Isub, z, inner) if inner == base && self.plan.const_of(z) == Some(0) => Some(SelectOp::Neg),
            _ => None,
        }
    }

    fn emit_select_op(&mut self, kind: SelectOp, size: Size, cond: Cond, dst: Reg, base: Reg) {
        match kind {
            SelectOp::Inc => self.asm.csinc(size, cond, dst, base, base),
            SelectOp::Inv => self.asm.csinv(size, cond, dst, base, base),
            SelectOp::Neg => self.asm.csneg(size, cond, dst, base, base),
        }
    }

    fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        if self.ty(condition).is_vector() {
            self.invalidate_spills();
            self.vec_select_mask(condition, x, y, result);
            return;
        }
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
        if ty.is_vector() {
            self.invalidate_spills();
            self.vec_select_cond(cond, x, y, result);
            return;
        }
        if ty.is_float() {
            let (left, right) = (self.fin(x, super::V0), self.fin(y, super::V1));
            let dst = self.fdst(result);
            self.asm.fcsel(cond, ty == Type::F64, dst, left, right);
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.invalidate_spills();
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
        let dst = self.dst(result);
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let word_ok = matches!(ty, Type::I32 | Type::I64);
        let uconst = |v: Value| self.plan.const_of(v).map(|imm| crate::fold::extend(imm, ty, false));
        let sconst = |v: Value| self.plan.const_of(v).map(|imm| crate::fold::extend(imm, ty, true));
        match (uconst(x), uconst(y)) {
            (Some(1), Some(0)) => self.asm.cset(cond, dst),
            (Some(0), Some(1)) => self.asm.cset(cond.invert(), dst),
            _ if word_ok && sconst(x) == Some(-1) && uconst(y) == Some(0) => self.asm.csinv(size, cond.invert(), dst, ZR, ZR),
            _ if word_ok && uconst(x) == Some(0) && sconst(y) == Some(-1) => self.asm.csinv(size, cond, dst, ZR, ZR),
            (kx, Some(1)) if word_ok => {
                let left = if kx == Some(0) { ZR } else { self.bits(x, A) };
                self.asm.csinc(size, cond, dst, left, ZR);
            }
            (Some(1), ky) if word_ok => {
                let right = if ky == Some(0) { ZR } else { self.bits(y, C) };
                self.asm.csinc(size, cond.invert(), dst, right, ZR);
            }
            (kx, _) if word_ok && sconst(y) == Some(-1) => {
                let left = if kx == Some(0) { ZR } else { self.bits(x, A) };
                self.asm.csinv(size, cond, dst, left, ZR);
            }
            (_, ky) if word_ok && sconst(x) == Some(-1) => {
                let right = if ky == Some(0) { ZR } else { self.bits(y, C) };
                self.asm.csinv(size, cond.invert(), dst, right, ZR);
            }
            (Some(kx), Some(ky)) if word_ok => {
                let (ux, uy) = (kx as u64, ky as u64);
                let mask = if size == Size::W { u32::MAX as u64 } else { u64::MAX };
                if ((ux.wrapping_add(1)) & mask) == uy {
                    let reg = self.bits(x, A);
                    self.asm.csinc(size, cond, dst, reg, reg);
                } else if ((uy.wrapping_add(1)) & mask) == ux {
                    let reg = self.bits(y, C);
                    self.asm.csinc(size, cond.invert(), dst, reg, reg);
                } else if (!ux & mask) == uy {
                    let reg = self.bits(x, A);
                    self.asm.csinv(size, cond, dst, reg, reg);
                } else if (ux.wrapping_neg() & mask) == uy {
                    let reg = self.bits(x, A);
                    self.asm.csneg(size, cond, dst, reg, reg);
                } else {
                    let left = if kx == 0 { ZR } else { self.bits(x, A) };
                    let right = if ky == 0 { ZR } else { self.bits(y, C) };
                    self.asm.csel(cond, dst, left, right);
                }
            }
            (kx, ky) => {
                if let Some(kind) = self.select_transform(ty, y, x) {
                    let reg = self.bits(x, A);
                    self.emit_select_op(kind, size, cond, dst, reg);
                } else if let Some(kind) = self.select_transform(ty, x, y) {
                    let reg = self.bits(y, C);
                    self.emit_select_op(kind, size, cond.invert(), dst, reg);
                } else {
                    let left = if kx == Some(0) { ZR } else { self.bits(x, A) };
                    let right = if ky == Some(0) { ZR } else { self.bits(y, C) };
                    self.asm.csel(cond, dst, left, right);
                }
            }
        }
        self.put(result, dst);
    }

    fn atomic(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        self.invalidate_spills();
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
