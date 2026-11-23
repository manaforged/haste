use super::asm::{Alu, Cond, FP, Mem, Reg, Size, ZR};
use super::{A, C, D, Emit, IP, S, T};
use crate::ir::{AtomicOrd, AtomicRmwOp, Binary, Convert, Function, IntCC, Lane, Op, ReduceOp, Type, Unary, Value};
use crate::reloc::{RelocKind, StencilReloc};

const COPY_SCRATCH: [Reg; 4] = [A, D, S, T];

fn scratch_mask_for(func: &Function, op: &Op, lse: bool) -> u8 {
    const ALL: u8 = 7;
    const S_BIT: u8 = 2;
    let mut wide = false;
    op.for_operands(|v| wide |= func.ty(v) == Type::I128);
    if wide {
        return ALL;
    }
    match op {
        Op::AtomicLoad(..) | Op::AtomicStore(..) => 0,
        Op::AtomicRmw(rmw, ..) if lse && *rmw != AtomicRmwOp::Nand => 0,
        Op::AtomicCas(..) if lse => 0,
        Op::Move(..) => S_BIT,
        Op::Copy(..) | Op::AtomicRmw(..) | Op::AtomicCas(..) => ALL,
        Op::Fits(Binary::Imul, _, x, _) if func.ty(*x) == Type::I64 => ALL,
        Op::Binary(Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr, x, _) if func.ty(*x).bits() < 32 => ALL,
        Op::Binary(Binary::AvgRound | Binary::SaddSat | Binary::UaddSat | Binary::SsubSat | Binary::UsubSat, ..) | Op::Convert(Convert::NarrowSat { .. }, _) | Op::Reduce(ReduceOp::Movemask, _) => ALL,
        Op::Binary(Binary::Imul, x, _) if matches!(func.ty(*x), Type::Vec { lane: Lane::I64, count: 2 | 4 }) => ALL,
        _ => 0,
    }
}

/// Bitmask of `[D, S, T]` scratch registers overwritten by `op` of `func` on LL/SC targets.
pub(super) fn scratch_mask(func: &Function, op: &Op) -> u8 {
    scratch_mask_for(func, op, false)
}

/// Bitmask of `[D, S, T]` scratch registers overwritten by `op` of `func` on ARMv8.1 LSE targets.
pub(super) fn scratch_mask_lse(func: &Function, op: &Op) -> u8 {
    scratch_mask_for(func, op, true)
}

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
                if !self.plan.is_static(result) && self.plan.through[result.index()].is_none() {
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

    pub(super) fn load_signed(&mut self, result: Value, src_ty: Type, op: &Op) {
        let dst = self.dst(result);
        self.load_gp(dst, src_ty, true, op);
        self.put(result, dst);
    }

    pub(super) fn load_signed_reg(&mut self, dst: Reg, src_ty: Type, op: &Op) {
        self.load_gp(dst, src_ty, true, op);
        self.clobber_reg(dst);
    }

    fn load_at(&mut self, result: Value, addr: Value, offset: i32) {
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
        let source = self.address_in(from, 0, S);
        let target = self.address_in(to, 0, C);
        let qpair_ok = |d: i32| d % 16 == 0 && (-1024..=1008).contains(&d);
        let mut at = 0_u32;
        let mut used_v0 = false;
        let mut used_v1 = false;
        while at + 32 <= bytes
            && let (Some(sd), Some(td)) = (source.disp.checked_add(at as i32), target.disp.checked_add(at as i32))
            && qpair_ok(sd)
            && qpair_ok(td)
        {
            self.asm.qldp_offset(super::V0, super::V1, source.base, sd);
            self.asm.qstp_offset(super::V0, super::V1, target.base, td);
            used_v0 = true;
            used_v1 = true;
            at += 32;
        }
        while at + 16 <= bytes {
            self.asm.vload(16, super::V1, source.offset(at as i32));
            self.asm.vstore(16, target.offset(at as i32), super::V1);
            used_v1 = true;
            at += 16;
        }
        for (ty, off) in crate::copy::words(bytes - at) {
            let off = (at + off) as i32;
            self.asm.load(Size::of(ty.bytes()), false, A, source.offset(off));
            self.asm.store(Size::of(ty.bytes()), target.offset(off), A);
        }
        if used_v0 {
            self.clobber_vreg(super::V0);
        }
        if used_v1 {
            self.clobber_vreg(super::V1);
        }
        self.clobber_regs(&[A, C, S]);
    }

    fn copy(&mut self, to: Value, from: Value, bytes: Value) -> Result<(), String> {
        let chunks = self.plan.const_of(bytes).and_then(crate::copy::chunks).ok_or("a copy of a size that is not a small constant")?;
        if chunks.is_empty() {
            return Ok(());
        }
        let source = self.address(from, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.load(Size::of(width), false, reg, source.offset(offset as i32));
        }
        self.clobber_regs(&COPY_SCRATCH[..chunks.len()]);
        let target = self.address(to, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.store(Size::of(width), target.offset(offset as i32), reg);
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
            let hi = if from.base == C { super::IP } else { C };
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
                self.clobber_vreg(super::V0);
            }
            let source = self.fin(value, super::V0);
            self.asm.vstore(ty.bytes(), to, source);
            return;
        }
        if ty == Type::I128 {
            let (lo, hi) = if self.plan.const_of(value) == Some(0) {
                (ZR, ZR)
            } else {
                let hi = if to.base == C { super::IP } else { C };
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

    fn store_src(&mut self, value: Value) -> Reg {
        match self.plan.const_of(value) {
            Some(0) => ZR,
            _ => {
                let source = super::fuse::store_source(self.func, &self.plan, value, |v| self.defs[v.index()]);
                self.bits(source, A)
            }
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

    fn elided_icmp_of(&self, condition: Value) -> Option<(IntCC, Value, Value)> {
        if !self.plan.elided[condition.index()] {
            return None;
        }
        match self.func.insts[self.defs[condition.index()]?.index()].op {
            Op::Icmp(cc, a, b) => Some((cc, a, b)),
            Op::Convert(Convert::Uextend, inner) if self.plan.elided[inner.index()] => {
                match self.func.insts[self.defs[inner.index()]?.index()].op {
                    Op::Icmp(cc, a, b) => Some((cc, a, b)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(super) fn preserves_flags(&self, op: &Op, result: Option<Value>) -> bool {
        matches!(*op, Op::Brif(..))
            || matches!(*op, Op::Select(cond, ..) | Op::Trapz(cond) if self.flags.is_some_and(|(v, _)| v == cond))
            || super::fuse::preserves_flags(self.func, &self.plan, op, result, true)
    }

    fn select_cond(&mut self, condition: Value) -> Cond {
        if let Some((val, c)) = self.flags
            && val == condition
        {
            return c;
        }
        if let Some((cc, a, b)) = self.elided_icmp_of(condition) {
            let c = self.compare(cc, a, b);
            self.flags = Some((condition, c));
            return c;
        }
        self.flags = None;
        self.last_cmp = None;
        if !self.tst_small_zero(condition) {
            let flag = self.zext_src(condition, C);
            let size = if self.ty(condition).bytes() <= 4 { Size::W } else { Size::X };
            self.asm.add_imm(true, true, size, ZR, flag, 0);
        }
        self.flags = Some((condition, Cond::Ne));
        Cond::Ne
    }

    fn select_csel(&mut self, cond: Cond, dst: Reg, (x, y): (Value, Value), (xz, yz): (bool, bool)) {
        let (left, right) = match (xz, yz) {
            (true, true) => (ZR, ZR),
            (true, false) => (ZR, self.bits(y, C)),
            (false, true) => (self.bits(x, A), ZR),
            (false, false) => self.bits_pair(x, y),
        };
        self.asm.csel(cond, dst, left, right);
    }

    fn select_const_pair(&mut self, cond: Cond, size: Size, dst: Reg, (x, y): (Value, Value), (kx, ky): (i64, i64)) {
        let (ux, uy) = (kx as u64, ky as u64);
        let mask = if size == Size::W { u32::MAX as u64 } else { u64::MAX };
        if ux == 0 || uy == 0 {
            self.select_csel(cond, dst, (x, y), (ux == 0, uy == 0));
            return;
        }
        let prefer_y = self.plan.reg_of(x).is_none() && (self.plan.reg_of(y).is_some() || super::asm::Asm::mov_imm_is_single(ky));
        let pick = |this: &mut Self| if prefer_y { (this.bits(y, C), cond.invert()) } else { (this.bits(x, A), cond) };
        if ((ux.wrapping_add(1)) & mask) == uy {
            let reg = self.bits(x, A);
            self.asm.csinc(size, cond, dst, reg, reg);
        } else if ((uy.wrapping_add(1)) & mask) == ux {
            let reg = self.bits(y, C);
            self.asm.csinc(size, cond.invert(), dst, reg, reg);
        } else if (!ux & mask) == uy {
            let (reg, c) = pick(self);
            self.asm.csinv(size, c, dst, reg, reg);
        } else if (ux.wrapping_neg() & mask) == uy {
            let (reg, c) = pick(self);
            self.asm.csneg(size, c, dst, reg, reg);
        } else {
            self.select_csel(cond, dst, (x, y), (false, false));
        }
    }

    fn select_int_pair(&mut self, cond: Cond, (ty, size): (Type, Size), dst: Reg, (x, y): (Value, Value), (kx, ky): (Option<i64>, Option<i64>)) {
        if let (Some(kx), Some(ky)) = (kx, ky)
            && matches!(ty, Type::I32 | Type::I64)
        {
            self.select_const_pair(cond, size, dst, (x, y), (kx, ky));
        } else if let Some(kind) = self.select_transform(ty, y, x) {
            let reg = self.bits(x, A);
            self.emit_select_op(kind, size, cond, dst, reg);
        } else if let Some(kind) = self.select_transform(ty, x, y) {
            let reg = self.bits(y, C);
            self.emit_select_op(kind, size, cond.invert(), dst, reg);
        } else {
            self.select_csel(cond, dst, (x, y), (kx == Some(0), ky == Some(0)));
        }
    }

    fn select_word_special(&mut self, (cond, size, ty): (Cond, Size, Type), dst: Reg, (x, y): (Value, Value)) -> bool {
        let uconst = |v: Value| self.plan.const_of(v).map(|imm| crate::fold::extend(imm, ty, false));
        let sconst = |v: Value| self.plan.const_of(v).map(|imm| crate::fold::extend(imm, ty, true));
        for ((keep, alt), c, scratch) in [((x, y), cond, A), ((y, x), cond.invert(), C)] {
            let k = uconst(keep);
            if sconst(alt) == Some(-1) {
                let reg = if k == Some(0) { ZR } else { self.bits(keep, scratch) };
                self.asm.csinv(size, c, dst, reg, ZR);
                return true;
            }
            if uconst(alt) == Some(1) {
                let reg = if k == Some(0) { ZR } else { self.bits(keep, scratch) };
                self.asm.csinc(size, c, dst, reg, ZR);
                return true;
            }
        }
        false
    }

    fn select_int(&mut self, cond: Cond, ty: Type, (x, y): (Value, Value), result: Value) {
        let dst = self.dst(result);
        let size = if ty == Type::I64 { Size::X } else { Size::W };
        let uconst = |v: Value| self.plan.const_of(v).map(|imm| crate::fold::extend(imm, ty, false));
        let (kx, ky) = (uconst(x), uconst(y));
        match (kx, ky) {
            (Some(1), Some(0)) => self.asm.cset(cond, dst),
            (Some(0), Some(1)) => self.asm.cset(cond.invert(), dst),
            _ if matches!(ty, Type::I32 | Type::I64) && self.select_word_special((cond, size, ty), dst, (x, y)) => {}
            _ => self.select_int_pair(cond, (ty, size), dst, (x, y), (kx, ky)),
        }
        self.put(result, dst);
    }

    fn select_i128(&mut self, cond: Cond, x: Value, y: Value, result: Value) {
        self.invalidate_spills();
        self.raw(A, x);
        self.raw(D, y);
        self.asm.csel(cond, A, A, D);
        self.put(result, A);
        self.high(A, x);
        self.high(D, y);
        self.asm.csel(cond, A, A, D);
        self.put_high(result, A);
    }

    fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        if self.ty(condition).is_vector() {
            self.invalidate_spills();
            self.vec_select_mask(condition, x, y, result);
            return;
        }
        let cond = self.select_cond(condition);
        let ty = self.ty(x);
        if ty.is_vector() {
            self.invalidate_spills();
            self.vec_select_cond(cond, x, y, result);
        } else if ty.is_float() {
            let (left, right) = (self.fin(x, super::V0), self.fin(y, super::V1));
            let dst = self.fdst(result);
            self.asm.fcsel(cond, ty == Type::F64, dst, left, right);
            self.fput(result, dst);
        } else if ty == Type::I128 {
            self.select_i128(cond, x, y, result);
        } else {
            self.select_int(cond, ty, (x, y), result);
        }
    }

    fn relaxed_atomic_target(&self, addr: Value) -> (Value, i32) {
        if !self.plan.elided[addr.index()] || self.plan.sib[addr.index()].is_some() {
            return (addr, 0);
        }
        let Some(inst) = self.defs[addr.index()] else { return (addr, 0) };
        let c = |v: Value| super::fuse::const_of(self.func, &self.plan, v, |x| self.defs[x.index()]);
        match self.func.insts[inst.index()].op {
            Op::Binary(crate::ir::Binary::Iadd, x, y) => {
                if let Some(imm) = c(y).and_then(|v| i32::try_from(v).ok()) { return (x, imm); }
                if let Some(imm) = c(x).and_then(|v| i32::try_from(v).ok()) { return (y, imm); }
            }
            Op::Binary(crate::ir::Binary::Isub, base, rhs) => {
                if let Some(imm) = c(rhs).and_then(|v| v.checked_neg()).and_then(|v| i32::try_from(v).ok()) { return (base, imm); }
            }
            _ => {}
        }
        (addr, 0)
    }

    fn atomic(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        let ty = match op {
            Op::AtomicStore(value, _, _) => self.ty(*value),
            _ => self.ty(result()?),
        };
        if ty == Type::I128 {
            return Err("128-bit atomics".to_owned());
        }
        let size = Size::of(ty.bytes());
        match op {
            Op::AtomicLoad(addr, ord) => {
                if ord.has_acquire() {
                    let at = self.bits(*addr, C);
                    let dst = self.dst(result()?);
                    self.asm.load_acquire(size, dst, at);
                    self.put(result()?, dst);
                } else {
                    let (base, off) = self.relaxed_atomic_target(*addr);
                    self.load_at(result()?, base, off);
                }
            }
            Op::AtomicStore(value, addr, ord) => {
                if ord.has_release() {
                    let at = self.bits(*addr, C);
                    let source = self.store_src(*value);
                    self.asm.store_release(size, at, source);
                } else {
                    let (base, off) = self.relaxed_atomic_target(*addr);
                    self.store_at(*value, base, off);
                }
            }
            Op::AtomicRmw(rmw, ord, addr, value) => {
                let res = result()?;
                if self.lse && *rmw != AtomicRmwOp::Nand {
                    self.lse_rmw(*rmw, *ord, size, *addr, *value, res);
                } else {
                    self.rmw(*rmw, *ord, size, *addr, *value);
                    self.clobber_regs(&[A, C, D, S, T]);
                    self.put(res, A);
                }
            }
            Op::AtomicCas(ord, addr, expected, replacement) => {
                let res = result()?;
                if self.lse {
                    self.lse_cas(*ord, size, *addr, (*expected, *replacement), res);
                } else {
                    self.cas(*ord, size, *addr, (*expected, *replacement));
                    self.clobber_regs(&[A, C, D, S, T]);
                    self.put(res, A);
                }
            }
            other => return Err(format!("unexpected {other:?}")),
        }
        Ok(())
    }

    fn lse_rmw(&mut self, op: AtomicRmwOp, ord: AtomicOrd, size: Size, addr: Value, value: Value, result: Value) {
        let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
        let at = self.bits(addr, C);
        let val = match (op, self.plan.const_of(value)) {
            (AtomicRmwOp::Sub, Some(imm)) => {
                self.asm.mov_imm(IP, imm.wrapping_neg());
                self.clobber_reg(IP);
                IP
            }
            (AtomicRmwOp::And, Some(imm)) => {
                self.asm.mov_imm(IP, !imm);
                self.clobber_reg(IP);
                IP
            }
            (AtomicRmwOp::Sub, None) => {
                let v = self.bits(value, IP);
                self.asm.alu(Alu::Sub, Size::X, IP, ZR, v);
                self.clobber_reg(IP);
                IP
            }
            (AtomicRmwOp::And, None) => {
                let v = self.bits(value, IP);
                self.asm.alu(Alu::Orn, Size::X, IP, ZR, v);
                self.clobber_reg(IP);
                IP
            }
            (_, Some(0)) => ZR,
            _ if signed => {
                self.get_opt(IP, value, true);
                IP
            }
            _ => self.bits(value, IP),
        };
        let dead = self.plan.fused[result.index()];
        let dst = if dead {
            if ord.has_acquire() { A } else { ZR }
        } else {
            self.dst(result)
        };
        self.asm.lse_rmw(op, ord, size, (at, val, dst));
        if dead {
            self.clobber_reg(dst);
        } else {
            self.put(result, dst);
        }
    }

    fn lse_cas(&mut self, ord: AtomicOrd, size: Size, addr: Value, (expected, replacement): (Value, Value), result: Value) {
        let at = self.bits(addr, C);
        let dst = self.dst(result);
        let rep_reg = self.plan.reg_of(replacement);
        let old = if dst == at || Some(dst) == rep_reg { A } else { dst };
        self.raw(old, expected);
        let rep = if self.plan.const_of(replacement) == Some(0) { ZR } else { self.bits(replacement, IP) };
        self.asm.lse_cas(ord, size, (at, old, rep));
        self.clobber_reg(old);
        self.put(result, old);
    }

    fn rmw(&mut self, op: AtomicRmwOp, ord: AtomicOrd, size: Size, addr: Value, value: Value) {
        let signed = matches!(op, AtomicRmwOp::Smin | AtomicRmwOp::Smax);
        let bytes = [1, 2, 4, 8][size as usize];
        self.raw(S, addr);
        self.get_opt(D, value, signed);
        let again = self.asm.label();
        self.asm.bind(again);
        self.asm.load_exclusive(ord.has_acquire(), size, A, S);
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
        self.asm.store_exclusive(ord.has_release(), size, T, S, C);
        self.asm.cbz(true, Size::W, T, again);
    }

    fn cas(&mut self, ord: AtomicOrd, size: Size, addr: Value, (expected, replacement): (Value, Value)) {
        let bytes = [1, 2, 4, 8][size as usize];
        self.raw(S, addr);
        self.zext_get(D, expected);
        self.raw(C, replacement);
        let (again, done) = (self.asm.label(), self.asm.label());
        self.asm.bind(again);
        self.asm.load_exclusive(ord.has_acquire(), size, A, S);
        let compare = if bytes < 8 { Size::W } else { Size::X };
        self.asm.alu(Alu::Subs, compare, ZR, A, D);
        self.asm.bcond(Cond::Ne, done);
        self.asm.store_exclusive(ord.has_release(), size, T, S, C);
        self.asm.cbz(true, Size::W, T, again);
        self.asm.bind(done);
    }
}
