mod mem;
mod select;

use super::asm::{Alu, Cond, FP, Reg, Size, ZR};
use super::{A, C, D, Emit, IP, S, T};
use crate::alloc::Location;
use crate::ir::{
    AtomicOrd, AtomicRmwOp, Binary, Convert, Function, Lane, Op, ReduceOp, Type, Value,
};

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
        Op::Binary(
            Binary::Ishl | Binary::Ushr | Binary::Sshr | Binary::Rotl | Binary::Rotr,
            x,
            _,
        ) if func.ty(*x).bits() < 32 => ALL,
        Op::Binary(
            Binary::AvgRound
            | Binary::SaddSat
            | Binary::UaddSat
            | Binary::SsubSat
            | Binary::UsubSat,
            ..,
        )
        | Op::Convert(Convert::NarrowSat { .. }, _)
        | Op::Reduce(ReduceOp::Movemask, _) => ALL,
        Op::Binary(Binary::Imul, x, _)
            if matches!(
                func.ty(*x),
                Type::Vec {
                    lane: Lane::I64,
                    count: 2 | 4
                }
            ) =>
        {
            ALL
        }
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

impl Emit<'_> {
    fn relaxed_atomic_target(&self, addr: Value) -> (Value, i32) {
        if !self.plan.elided[addr.index()] || self.plan.sib[addr.index()].is_some() {
            return (addr, 0);
        }
        let Some(inst) = self.defs[addr.index()] else {
            return (addr, 0);
        };
        let c =
            |v: Value| super::fuse::const_of(self.func, &self.plan, v, |x| self.defs[x.index()]);
        match self.func.insts[inst.index()].op {
            Op::Binary(Binary::Iadd, x, y) => {
                if let Some(imm) = c(y).and_then(|v| i32::try_from(v).ok()) {
                    return (x, imm);
                }
                if let Some(imm) = c(x).and_then(|v| i32::try_from(v).ok()) {
                    return (y, imm);
                }
            }
            Op::Binary(Binary::Isub, base, rhs) => {
                if let Some(imm) = c(rhs)
                    .and_then(|v| v.checked_neg())
                    .and_then(|v| i32::try_from(v).ok())
                {
                    return (base, imm);
                }
            }
            _ => {}
        }
        (addr, 0)
    }

    fn rcpc2_target(&mut self, addr: Value) -> (Reg, i32) {
        if let Location::Offset(base, disp) = self.plan.location(addr)
            && (-256..=255).contains(&disp)
        {
            return (base, disp);
        }
        if let Location::Frame(slot, off) = self.plan.location(addr)
            && let Some(disp) = self.frame.slots[slot.index()].checked_add(off)
            && (-256..=255).contains(&disp)
        {
            return (FP, disp);
        }
        let (base, off) = self.relaxed_atomic_target(addr);
        if base != addr && (-256..=255).contains(&off) {
            return (self.bits(base, C), off);
        }
        (self.bits(addr, C), 0)
    }

    fn atomic(
        &mut self,
        op: &Op,
        result: impl Fn() -> Result<Value, String>,
    ) -> Result<(), String> {
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
                    let dst = self.dst(result()?);
                    if self.lse && *ord == AtomicOrd::Acquire {
                        let (base, disp) = self.rcpc2_target(*addr);
                        if disp != 0 {
                            self.asm.load_acquire_rcpc_off(size, dst, base, disp);
                        } else {
                            self.asm.load_acquire_rcpc(size, dst, base);
                        }
                    } else {
                        let at = self.bits(*addr, C);
                        self.asm.load_acquire(size, dst, at);
                    }
                    self.put(result()?, dst);
                } else {
                    let (base, off) = self.relaxed_atomic_target(*addr);
                    self.load_at(result()?, base, off);
                }
            }
            Op::AtomicStore(value, addr, ord) => {
                if ord.has_release() {
                    if self.lse && *ord == AtomicOrd::Release {
                        let (base, disp) = self.rcpc2_target(*addr);
                        let source = self.store_src(*value);
                        if disp != 0 {
                            self.asm.store_release_off(size, base, disp, source);
                        } else {
                            self.asm.store_release(size, base, source);
                        }
                    } else {
                        let at = self.bits(*addr, C);
                        let source = self.store_src(*value);
                        self.asm.store_release(size, at, source);
                    }
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

    fn lse_rmw(
        &mut self,
        op: AtomicRmwOp,
        ord: AtomicOrd,
        size: Size,
        addr: Value,
        value: Value,
        result: Value,
    ) {
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

    fn lse_cas(
        &mut self,
        ord: AtomicOrd,
        size: Size,
        addr: Value,
        (expected, replacement): (Value, Value),
        result: Value,
    ) {
        let at = self.bits(addr, C);
        let dst = self.dst(result);
        let rep_reg = self.plan.reg_of(replacement);
        let old = if dst == at || Some(dst) == rep_reg {
            A
        } else {
            dst
        };
        self.raw(old, expected);
        let rep = if self.plan.const_of(replacement) == Some(0) {
            ZR
        } else {
            self.bits(replacement, IP)
        };
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

    fn cas(
        &mut self,
        ord: AtomicOrd,
        size: Size,
        addr: Value,
        (expected, replacement): (Value, Value),
    ) {
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
