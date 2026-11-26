use super::asm::{Alu, Cond, Mem, Reg, Size, Xmm};
use super::float::{X0, X1};
use super::location::LEAF;
use super::{Emit, int};
use crate::alloc::Location;
use crate::frame::{Emitter, Ext};
use crate::ir::{AtomicRmwOp, Binary, Block, Function, IntCC, Op, ShuffleMask, StackSlot, TargetOpId, Type, Unary, Value};
use crate::reloc::{RelocKind, StencilReloc};

const SEQ_CST_FENCE: [u8; 6] = [0xF0, 0x48, 0x83, 0x0C, 0x24, 0x00];

/// Scratch registers an inline copy holds its chunks in; none of them forms an address.
const COPY_SCRATCH: [Reg; 4] = [Reg::Rax, Reg::Rdx, Reg::R11, Reg::Rsi];

/// The bytes a move copies through one xmm register at a time.
const VECTOR: u32 = 16;

/// The opcodes of `movups` from memory into an xmm register, and from an xmm register into memory.
const MOVUPS: ([u8; 2], [u8; 2]) = ([0x0F, 0x10], [0x0F, 0x11]);

/// Bitmask of `[Rsi, Rdx, R11]` scratch registers overwritten by `op` of `func`.
pub(super) fn scratch_mask(func: &Function, op: &Op) -> u8 {
    const RSI: u8 = 1;
    const RDX: u8 = 2;
    const R11: u8 = 4;
    const ALL: u8 = RSI | RDX | R11;
    match op {
        Op::Unary(_, x) | Op::Binary(_, x, _) | Op::Icmp(_, x, _) if func.ty(*x) == Type::I128 => ALL,
        Op::Copy(..) | Op::Fits(Binary::Imul, ..) | Op::Binary(Binary::Fcopysign, ..) => ALL,
        Op::TargetOp(TargetOpId::X64AddCarry | TargetOpId::X64AddCarryOut | TargetOpId::X64SubBorrow | TargetOpId::X64SubBorrowOut | TargetOpId::X64DivWide | TargetOpId::X64RemWide, _) => ALL,
        Op::Move(..) | Op::AtomicCas(..) => R11,
        Op::AtomicRmw(AtomicRmwOp::Xchg | AtomicRmwOp::Add | AtomicRmwOp::Sub, ..) => 0,
        Op::AtomicRmw(..) => RSI | RDX,
        Op::Binary(Binary::Sdiv | Binary::Srem | Binary::Udiv | Binary::Urem | Binary::Smulhi | Binary::Umulhi | Binary::AvgRound | Binary::UaddSat | Binary::UsubSat, ..) => RDX,
        Op::Binary(Binary::SaddSat | Binary::SsubSat, ..) => RDX | R11,
        Op::Unary(Unary::Bitrev, _) => RDX,
        Op::Unary(Unary::Floor | Unary::Ceil | Unary::Trunc, x) if func.ty(*x).is_float() => RDX,
        Op::Unary(Unary::Popcnt, _) => RSI | R11,
        _ => 0,
    }
}

impl Emit<'_> {
    pub(super) fn invalidate_spills(&mut self) { self.spills.clear(); }

    pub(super) fn clobber_call_spills(&mut self) {
        self.clobber_regs(&[Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi, Reg::R8, Reg::R9, Reg::R10, Reg::R11]);
        self.spills.clear_fspill();
    }

    pub(super) fn clobber_reg(&mut self, reg: Reg) { self.spills.clobber_reg(reg); }

    pub(super) fn clobber_regs(&mut self, regs: &[Reg]) { regs.iter().for_each(|&reg| self.clobber_reg(reg)); }

    pub(super) fn clobber_xmm(&mut self, xmm: Xmm) { self.spills.clobber_freg(xmm); }

    pub(super) fn clobber_xmms(&mut self, xmms: &[Xmm]) { xmms.iter().for_each(|&xmm| self.clobber_xmm(xmm)); }

    pub(super) fn invalidate_disp(&mut self, disp: i32) { self.spills.invalidate_disp(disp); }

    pub(super) fn resume_spills(&mut self, block: Block) { self.spills.resume(block); }

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
        self.plan.fused[v.index()] && self.defs[v.index()].is_some_and(|i| matches!(self.func.insts[i.index()].op, Op::AtomicCas(..)))
    }

    pub(super) fn memory_op(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Const(_) if result().is_ok_and(|result| self.ty(result).is_vector()) => self.vector_zero(result()?),
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
                        self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Abs32));
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
                        self.relocs.push(StencilReloc::symbol(at, *sym, RelocKind::Abs64));
                    }
                    self.put(result, dst);
                }
            }
            Op::Runtime(_) if self.plan.through[result()?.index()].is_some() => {}
            Op::Runtime(rt) => {
                let result = result()?;
                let dst = self.dst(result);
                let at = self.asm.mov_abs64(dst);
                self.relocs.push(StencilReloc::runtime(at, *rt, RelocKind::Abs64));
                self.put(result, dst);
            }
            Op::Select(condition, x, y) => self.select(*condition, *x, *y, result()?),
            Op::Shuffle(..) | Op::Extract(..) | Op::Insert(..) | Op::Reduce(..) | Op::TargetOp(..) => {
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
            Op::Shuffle(a, _, ShuffleMask::Const(_)) if self.ty(*a).is_wide_vec() || self.features.ssse3 => {
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

    fn access(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
        match op {
            Op::Load(addr, offset) => self.load(*addr, *offset, result()?),
            Op::VolatileLoad(addr, offset) => { let from = self.address(*addr, *offset); self.read(result()?, from); }
            Op::Store(value, addr, offset) => {
                let to = self.address(*addr, *offset);
                if self.plan.elided[value.index()] && !self.has_elided_mem(*value) { self.emit_mem_rmw(*value, to); } else { self.write(*value, to); }
            }
            Op::VolatileStore(value, addr, offset) => { let to = self.address(*addr, *offset); self.write(*value, to); }
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
                if self.plan.elided[value.index()] && !self.has_elided_mem(*value) { self.emit_mem_rmw(*value, to); } else { self.write(*value, to); }
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
        if self.direct_calls && self.ty(result).is_small_int()
            && let Some(sym) = match self.location(addr) {
                Location::SlotAddr(s) => Some(s),
                Location::Symbol(s) if self.func.is_image_addr(addr) => Some(s),
                _ => None,
            }
        {
            let dst = self.dst(result);
            let at = self.asm.load_rip_reloc(Size::of(self.ty(result).bytes()), dst);
            self.relocs.push(StencilReloc::symbol(at, sym, RelocKind::X64RipRel32));
            self.put(result, dst);
            return true;
        }
        false
    }

    /// Reads `result` from `addr + offset`, unless a call reads it in place.
    fn load(&mut self, addr: Value, offset: i32, result: Value) {
        if self.plan.through[result.index()].is_some() || (offset == 0 && self.try_load_rip(addr, result)) { return; }
        let from = self.address(addr, offset);
        self.read(result, from);
    }

    /// Copies a constant number of bytes inline: every chunk into its own scratch register first, so overlapping ranges copy correctly.
    fn copy(&mut self, to: Value, from: Value, bytes: Value) -> Result<(), String> {
        let chunks = self.const_of(bytes).and_then(crate::copy::chunks).ok_or("a copy of a size that is not a small constant")?;
        let source = self.address(from, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.load(Size::of(width), reg, source.offset(offset as i32));
        }
        self.clobber_regs(&COPY_SCRATCH);
        let target = self.address(to, 0);
        for (&(width, offset), reg) in chunks.iter().zip(COPY_SCRATCH) {
            self.asm.store(Size::of(width), target.offset(offset as i32), reg);
        }
        Ok(())
    }

    /// Copies `bytes` bytes between two ranges that are the same or disjoint: 32 at a time under AVX, 16 through xmm1, then word by word through rax; r11 and rcx hold addresses.
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
                self.asm.vex_xmm(0b00001, 0b00, 0x10, (X1, Xmm(0)), source.offset(at as i32));
                self.asm.vex_xmm(0b00001, 0b00, 0x11, (X1, Xmm(0)), target.offset(at as i32));
            } else {
                self.asm.sse(&[], &MOVUPS.0, X1.0, source.offset(at as i32));
                self.asm.sse(&[], &MOVUPS.1, X1.0, target.offset(at as i32));
            }
            at += VECTOR;
        }
        for (ty, off) in crate::copy::words(bytes - at) {
            let off = (at + off) as i32;
            self.asm.load(Size::of(ty.bytes()), Reg::Rax, source.offset(off));
            self.asm.store(Size::of(ty.bytes()), target.offset(off), Reg::Rax);
        }
        self.clobber_regs(&[Reg::R11, Reg::Rcx, Reg::Rax]);
        self.clobber_xmm(X1);
    }

    fn atomic(&mut self, op: &Op, result: impl Fn() -> Result<Value, String>) -> Result<(), String> {
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
                let (target, src) = match self.reg_of(*replacement).filter(|&r| r != Reg::Rax && r != Reg::R11) {
                    Some(rep) => (self.atomic_target(*addr, Reg::R11, &[Reg::Rax, rep]), rep),
                    None => {
                        let t = self.atomic_target(*addr, Reg::R11, &[Reg::Rax, Reg::Rcx]);
                        self.raw(Reg::Rcx, *replacement);
                        (t, Reg::Rcx)
                    }
                };
                if size == Size::D { self.get(Reg::Rax, *expected, Ext::Zero); } else { self.raw(Reg::Rax, *expected); }
                self.asm.cmpxchg(size, target, src);
                self.asm.hold_eq(res.index() as u32, expected.index() as u32);
                self.clobber_regs(if src == Reg::Rcx { &[Reg::Rax, Reg::Rcx, Reg::R11] } else { &[Reg::Rax, Reg::R11] });
                if !self.plan.fused[res.index()] { self.put(res, Reg::Rax); }
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
        if let Some(imm) = self.const_of(value).and_then(|imm| if ty.bytes() < 8 { Some(imm as i32) } else { i32::try_from(imm).ok() }) {
            if imm == 0 && size == Size::Q {
                return self.zero_to(to);
            }
            self.asm.store_imm_sized(size, to, imm);
            return;
        }
        let source = self.reg_of(value).or_else(|| (ty != Type::I128).then(|| self.cached_spill(value)).flatten()).unwrap_or(Reg::Rax);
        self.raw(source, value);
        if ty == Type::I128 {
            self.asm.store(Size::Q, to, Reg::Rax);
            self.high(Reg::Rax, value);
            self.asm.store(Size::Q, to.offset(8), Reg::Rax);
            return;
        }
        self.asm.store(size, to, source);
    }

    fn root(&self, condition: Value) -> (Value, bool) {
        match self.roots[condition.index()] {
            Some((leaf, None, table)) => (leaf, table != LEAF[0]),
            _ => (condition, false),
        }
    }

    /// The flags condition that holds when `condition` is not zero, from the live flags of its compare or a new test.
    pub(super) fn truth(&mut self, condition: Value) -> Cond {
        let (root, flip) = self.root(condition);
        let cond = match self.flags.take() {
            Some((value, cond)) if value == root => cond,
            _ => self.test_cond(root),
        };
        if flip { cond.invert() } else { cond }
    }

    fn test_cond(&mut self, condition: Value) -> Cond {
        if self.plan.elided[condition.index()]
            && let Some(inst) = self.defs[condition.index()]
        {
            match self.func.insts[inst.index()].op {
                Op::Binary(Binary::Band, x, y) => return self.emit_test_band(x, y),
                Op::Convert(crate::ir::Convert::Ireduce, inner) if self.plan.elided[inner.index()] => {
                    if let Some(iinst) = self.defs[inner.index()]
                        && let Op::Binary(Binary::Band, x, y) = self.func.insts[iinst.index()].op
                    {
                        return self.emit_test_band(x, y);
                    }
                }
                Op::Icmp(cc, lhs, rhs) => return self.cmp_flags(cc, lhs, rhs, None),
                _ => {}
            }
        }
        if self.held_eq_zero(condition) {
            return Cond::Ne;
        }
        self.test_value(condition);
        Cond::Ne
    }

    fn held_eq_zero(&self, value: Value) -> bool {
        let Some((a, b)) = self.asm.held_eq() else { return false };
        let ty = self.ty(value);
        let is_zero = |u: u32| self.const_of(Value(u)).is_some_and(|c| crate::fold::extend(c, ty, false) == 0);
        (a == value.index() as u32 && is_zero(b)) || (b == value.index() as u32 && is_zero(a))
    }

    fn test_value(&mut self, value: Value) {
        let size = Size::of(self.ty(value).bytes());
        if self.ty(value).is_small_int() && (self.has_elided_mem(value) || (matches!(self.location(value), Location::Slot) && self.cached_spill(value).is_none())) {
            let rm = self.operand(value);
            self.asm.alu_imm_sized(size, Alu::Cmp, rm, 0);
            return;
        }
        let reg = self.reg_of(value).or_else(|| self.cached_spill(value)).unwrap_or_else(|| { self.get(Reg::Rax, value, Ext::Zero); Reg::Rax });
        self.asm.test_sized(size, reg, reg);
    }

    fn held_eq_compare(&self, cc: IntCC, x: Value, y: Value) -> Option<Cond> {
        if !matches!(cc, IntCC::Equal | IntCC::NotEqual) {
            return None;
        }
        let (a, b) = self.asm.held_eq()?;
        let ty = self.ty(x);
        let eq = |u: u32, v: Value| {
            u == v.index() as u32
                || self.const_of(Value(u)).zip(self.const_of(v)).is_some_and(|(c1, c2)| {
                    crate::fold::extend(c1, ty, false) == crate::fold::extend(c2, ty, false)
                })
        };
        ((eq(a, x) && eq(b, y)) || (eq(a, y) && eq(b, x))).then(|| int::cond(cc))
    }

    /// Compares `x` with `y` and returns the flags condition under which `cc` holds.
    pub(super) fn cmp_flags(&mut self, cc: IntCC, x: Value, y: Value, live_flags: Option<(Value, Cond)>) -> Cond {
        let is_zero = |v: Value| self.const_of(v).is_some_and(|imm| crate::fold::extend(imm, self.ty(v), false) == 0);
        let is_one = |v: Value| self.const_of(v).is_some_and(|imm| crate::fold::extend(imm, self.ty(v), false) == 1);
        let bool_one_var = |a: Value, b: Value| {
            if is_one(b) && self.plan.clean_bits[a.index()] <= 1 {
                Some(a)
            } else if is_one(a) && self.plan.clean_bits[b.index()] <= 1 {
                Some(b)
            } else {
                None
            }
        };
        if self.ty(x) == Type::I128 {
            self.invalidate_spills();
            let cond = self.wide_icmp(cc, x, y);
            self.invalidate_spills();
            cond
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && live_flags.is_some_and(|(val, fcc)| fcc == Cond::Ne && ((val == x && is_zero(y)) || (val == y && is_zero(x))))
        {
            int::cond(cc)
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some(bv) = bool_one_var(x, y)
            && live_flags.is_some_and(|(val, fcc)| fcc == Cond::Ne && val == bv)
        {
            if cc == IntCC::Equal { Cond::Ne } else { Cond::E }
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some(bv) = bool_one_var(x, y)
            && let Some(reg) = self.reg_of(bv).or_else(|| self.cached_spill(bv))
        {
            self.asm.test_sized(Size::of(self.ty(bv).bytes()), reg, reg);
            if cc == IntCC::Equal { Cond::Ne } else { Cond::E }
        } else if matches!(cc, IntCC::Equal | IntCC::NotEqual)
            && let Some((a, b)) = self.elided_band_cmp(x, y)
        {
            let nz = self.emit_test_band(a, b);
            if cc == IntCC::NotEqual { nz } else { nz.invert() }
        } else if let Some(cond) = self.held_eq_compare(cc, x, y).or_else(|| self.held_compare(cc, x, y)) {
            cond
        } else {
            let size = Size::of(self.ty(x).bytes());
            let (cc, x, y) = if self.reg_of(x).is_none() && (self.reg_of(y).is_some() || self.imm_sized(x, size).is_some()) {
                (crate::cse::flip_int_cc(cc), y, x)
            } else {
                (cc, x, y)
            };
            self.emit_cmp_pair(size, x, Some(y), matches!(cc, IntCC::Equal | IntCC::NotEqual) || cc.is_signed(), !cc.is_signed());
            int::cond(cc)
        }
    }

    fn select_chain(&mut self, leaves: [Value; 2], table: u8, (x, y): (Value, Value), result: Value) {
        let (key, first, other) = if table.count_ones() == 1 { (table.trailing_zeros(), x, y) } else { (table.trailing_ones(), y, x) };
        let dst = self.dst(result);
        for (at, leaf) in leaves.into_iter().enumerate() {
            let cond = self.test_cond(leaf);
            if at == 0 {
                self.raw(dst, first);
            }
            let source = self.operand(other);
            self.asm.cmov(if key >> at & 1 == 1 { cond.invert() } else { cond }, dst, source);
            self.clobber_reg(dst);
        }
        self.put(result, dst);
    }

    fn select_cond(&mut self, condition: Value, live_flags: Option<(Value, Cond)>) -> Cond {
        if let Some((value, cond)) = live_flags
            && value == condition
        {
            return cond;
        }
        if !self.plan.elided[condition.index()]
            && matches!(self.location(condition), Location::Slot)
            && self.cached_spill(condition).is_none()
        {
            self.asm.alu_imm_sized(Size::of(self.ty(condition).bytes()), Alu::Cmp, self.at(condition), 0);
            return Cond::Ne;
        }
        self.test_cond(condition)
    }

    fn select_copy(&mut self, value: Value, result: Value) {
        let ty = self.ty(result);
        if ty.in_float_register() {
            let dst = self.xmm_of(value).unwrap_or_else(|| {
                let dst = self.fdst(result);
                self.fget(dst, value);
                dst
            });
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, self.at(value));
            self.clobber_reg(Reg::Rax);
            self.put(result, Reg::Rax);
            self.asm.load(Size::Q, Reg::Rax, self.at(value).offset(8));
            self.put_high(result, Reg::Rax);
            self.clobber_reg(Reg::Rax);
            return;
        }
        if matches!(self.location(result), Location::Slot)
            && let Some(reg) = self.reg_of(value).or_else(|| self.cached_spill(value))
        {
            self.put(result, reg);
            return;
        }
        let dst = self.dst(result);
        self.raw(dst, value);
        self.put(result, dst);
    }

    fn select(&mut self, condition: Value, x: Value, y: Value, result: Value) {
        let live_flags = self.flags.take();
        if self.ty(condition).is_vector() {
            self.flags = live_flags;
            self.spills.clear_fspill();
            self.vector_select(condition, x, y, result);
            if !self.features.avx {
                self.clobber_xmms(&[X0, X1]);
            }
            return;
        }
        if x == y {
            self.flags = live_flags;
            self.select_copy(x, result);
            return;
        }
        if let Some(imm) = self.const_of(condition) {
            let known = crate::fold::extend(imm, self.ty(condition), false);
            self.flags = live_flags;
            self.select_copy(if known != 0 { x } else { y }, result);
            return;
        }
        if let Some((first, Some(second), table)) = self.roots[condition.index()] {
            return self.select_chain([first, second], table, (x, y), result);
        }
        let (condition, flip) = self.root(condition);
        let orig_cond = self.select_cond(condition, live_flags);
        let mut cond = if flip { orig_cond.invert() } else { orig_cond };
        let ty = self.ty(x);
        if ty.in_float_register() {
            let dst = self.fdst(result);
            let (mut x, mut y) = (x, y);
            if self.xmm_of(x) == Some(dst) {
                (x, y) = (y, x);
                cond = cond.invert();
            }
            let done = self.asm.label();
            self.flags = Some((condition, orig_cond));
            self.fget(dst, y);
            self.asm.jcc(cond.invert(), done);
            self.fget(dst, x);
            self.asm.bind(done);
            self.clobber_reg(Reg::Rax);
            self.spills.clear_fspill();
            self.fput(result, dst);
            return;
        }
        if ty == Type::I128 {
            self.asm.load(Size::Q, Reg::Rax, self.at(y));
            self.asm.cmov(cond, Reg::Rax, self.at(x));
            self.clobber_reg(Reg::Rax);
            self.put(result, Reg::Rax);
            self.asm.load(Size::Q, Reg::Rax, self.at(y).offset(8));
            self.asm.cmov(cond, Reg::Rax, self.at(x).offset(8));
            self.put_high(result, Reg::Rax);
            self.clobber_reg(Reg::Rax);
            self.flags = Some((condition, orig_cond));
            return;
        }
        let dst = self.dst(result);
        if self.select_const_pair((condition, orig_cond), cond, (x, y), (ty, dst, result)) {
            return;
        }
        let (mut x, mut y) = (x, y);
        if self.reg_of(x) == Some(dst) {
            (x, y) = (y, x);
            cond = cond.invert();
        }
        self.flags = Some((condition, orig_cond));
        self.raw(dst, y);
        let source = self.operand(x);
        self.asm.cmov(cond, dst, source);
        self.clobber_reg(dst);
        self.put(result, dst);
    }

    fn select_const_pair(
        &mut self,
        root: (Value, Cond),
        cond: Cond,
        values: (Value, Value),
        target: (Type, Reg, Value),
    ) -> bool {
        let (x, y) = values;
        let (ty, dst, result) = target;
        let (Some(xv), Some(yv)) = (self.const_of(x), self.const_of(y)) else {
            return false;
        };
        if !matches!(ty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
            return false;
        }
        let xu = crate::fold::extend(xv, ty, false) as u64;
        let yu = crate::fold::extend(yv, ty, false) as u64;
        if (xu == 1 && yu == 0) || (xu == 0 && yu == 1) {
            self.emit_setcc_byte(if xu == 1 { cond } else { cond.invert() }, dst);
        } else if !self.select_const_lea(cond, (xu, yu), target) {
            return false;
        }
        self.flags = Some(root);
        self.clobber_reg(dst);
        self.put(result, dst);
        true
    }

    fn select_const_lea(
        &mut self,
        cond: Cond,
        pair: (u64, u64),
        target: (Type, Reg, Value),
    ) -> bool {
        let (xu, yu) = pair;
        let (ty, dst, result) = target;
        if dst == Reg::Rsp {
            return false;
        }
        let mask = u64::MAX >> (64 - ty.bits());
        let subword_zext = matches!(ty, Type::I8 | Type::I16) && self.plan.zero_ext[result.index()];
        let Some((branch_cond, imm32, step)) = Self::match_select_lea(cond, (xu, yu), (ty, mask, subword_zext)) else {
            return false;
        };
        let size = if ty == Type::I64 { Size::Q } else { Size::D };
        let mem = match step {
            1 => Mem::new(dst, imm32),
            2 => Mem::sib(dst, dst, 0, imm32),
            3 => Mem::sib(dst, dst, 1, imm32),
            5 => Mem::sib(dst, dst, 2, imm32),
            _ => Mem::sib(dst, dst, 3, imm32),
        };
        self.emit_setcc_byte(branch_cond, dst);
        self.asm.lea_sized(size, dst, mem);
        true
    }

    fn match_select_lea(
        cond: Cond,
        pair: (u64, u64),
        info: (Type, u64, bool),
    ) -> Option<(Cond, i32, u64)> {
        let (xu, yu) = pair;
        let (ty, mask, subword_zext) = info;
        for step in [1_u64, 2, 3, 5, 9] {
            for (branch_cond, small, hi) in [(cond, yu, xu), (cond.invert(), xu, yu)] {
                if hi != (small.wrapping_add(step) & mask) {
                    continue;
                }
                if subword_zext && small.checked_add(step).is_none_or(|top| top > mask) {
                    continue;
                }
                let small_signed = if subword_zext { small as i64 } else { crate::fold::extend(small as i64, ty, true) };
                if let Ok(imm32) = i32::try_from(small_signed) {
                    return Some((branch_cond, imm32, step));
                }
            }
        }
        None
    }

    fn emit_setcc_byte(&mut self, cond: Cond, dst: Reg) {
        self.asm.setcc(cond, dst);
        self.asm.load(Size::B, dst, dst);
    }
}
