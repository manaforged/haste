use super::super::Emit;
use super::super::asm::{Alu, Mem, Reg, Rm, Size, Vl, Xmm};
use super::super::float::{X2, scalar};
use super::super::ymm::broadcast;
use crate::alloc::Location;
use crate::fold::fits_imm32;
use crate::frame::{ArgStep, Ext};
use crate::ir::{Lane, Op, Type, Value};
use crate::reloc::{RelocKind, StencilReloc};

impl Emit<'_> {
    /// Loads the float or vector of type `ty` at `from` into `xmm`; an 8-byte vector by `movq` and a 16-byte vector by `movdqu`, which needs no alignment.
    pub(in crate::x64) fn xmm_load(&mut self, ty: Type, xmm: Xmm, from: Mem) {
        if ty.is_wide_vec() {
            return self.ymm_load(xmm, from);
        }
        if self.features.avx {
            let (pp, opcode) = if ty.is_vector() {
                (0b10, if ty.bytes() == 16 { 0x6F } else { 0x7E })
            } else {
                (if ty == Type::F64 { 0b11 } else { 0b10 }, 0x10)
            };
            return self.asm.vex_xmm(0b00001, pp, opcode, (xmm, Xmm(0)), from);
        }
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 {
                (0xF3, 0x6F)
            } else {
                (0xF3, 0x7E)
            };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, from);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x10], xmm.0, from);
        }
    }

    /// Stores the float or vector of type `ty` in `xmm` to `to`.
    pub(in crate::x64) fn xmm_store(&mut self, ty: Type, to: Mem, xmm: Xmm) {
        if ty.is_wide_vec() {
            return self.ymm_store(to, xmm);
        }
        if self.features.avx {
            let (pp, opcode) = if ty.is_vector() {
                if ty.bytes() == 16 {
                    (0b10, 0x7F)
                } else {
                    (0b01, 0xD6)
                }
            } else {
                (if ty == Type::F64 { 0b11 } else { 0b10 }, 0x11)
            };
            return self.asm.vex_xmm(0b00001, pp, opcode, (xmm, Xmm(0)), to);
        }
        if ty.is_vector() {
            let (prefix, opcode) = if ty.bytes() == 16 {
                (0xF3, 0x7F)
            } else {
                (0x66, 0xD6)
            };
            self.asm.sse(&[prefix], &[0x0F, opcode], xmm.0, to);
        } else {
            self.asm.sse(&[scalar(ty)], &[0x0F, 0x11], xmm.0, to);
        }
    }

    /// Puts float `value` in `xmm`.
    pub(in crate::x64) fn fget(&mut self, xmm: Xmm, value: Value) {
        let ty = self.ty(value);
        if let Some(mem) = self.elided_mem(value) {
            self.xmm_load(ty, xmm, mem);
            self.clobber_xmm(xmm);
            return;
        }
        match self.location(value) {
            Location::Float(from) if Xmm(from) == xmm => {}
            Location::Float(from) => {
                self.fmov(ty, xmm, Xmm(from));
                self.clobber_xmm(xmm);
            }
            Location::Imm(0) if ty.is_vector() => self.zero_xmm(xmm),
            Location::Slot => {
                let disp = self.frame.values[value.index()];
                debug_assert_ne!(disp, 0, "unhomed value {value:?} in fget()");
                if let Some(cached_reg) = self.spills.cached_fspill(disp, ty) {
                    if cached_reg != xmm {
                        self.fmov(ty, xmm, cached_reg);
                        self.clobber_xmm(xmm);
                    }
                } else {
                    self.xmm_load(ty, xmm, self.frame_mem(disp));
                    self.clobber_xmm(xmm);
                    self.spills.fspill = Some((disp, xmm, ty));
                }
            }
            _ => {
                self.raw(Reg::Rax, value);
                self.asm.movq_in(xmm, Reg::Rax);
                self.clobber_xmm(xmm);
            }
        }
    }

    /// Writes `xmm` into the location of float `value`.
    pub(in crate::x64) fn fput(&mut self, value: Value, xmm: Xmm) {
        let ty = self.ty(value);
        match self.location(value) {
            Location::Float(to) => {
                if Xmm(to) != xmm {
                    self.fmov(ty, Xmm(to), xmm);
                }
                self.clobber_xmm(Xmm(to));
            }
            _ => self.store_xmm_home(self.at(value), xmm, ty),
        }
    }

    /// A float register that holds `value`: its location register, or `scratch` loaded with it.
    pub(in crate::x64) fn fin(&mut self, value: Value, scratch: Xmm) -> Xmm {
        self.xmm_of(value).unwrap_or_else(|| {
            self.fget(scratch, value);
            scratch
        })
    }

    /// A float register or memory operand that holds `value`, using `scratch` when `value` has no float home.
    pub(in crate::x64) fn foperand(&mut self, value: Value, scratch: Xmm) -> Rm {
        if let Some(mem) = self.elided_mem(value) {
            return mem.into();
        }
        match self.location(value) {
            Location::Float(xmm) => Xmm(xmm).into(),
            Location::Slot => self.at(value).into(),
            _ => {
                self.fget(scratch, value);
                scratch.into()
            }
        }
    }

    /// A register or memory operand that holds `value`, using rcx when `value` has no register or stack home.
    pub(in crate::x64) fn operand(&mut self, value: Value) -> Rm {
        if let Some(mem) = self.elided_mem(value) {
            return mem.into();
        }
        match self.location(value) {
            Location::Reg(reg) => reg.into(),
            Location::Slot => self
                .cached_spill(value)
                .map_or_else(|| Rm::from(self.at(value)), Rm::from),
            _ => {
                self.raw(Reg::Rcx, value);
                Reg::Rcx.into()
            }
        }
    }

    /// Whether reading `value` as an operand writes rcx: an elided load addresses through it, and a constant or address without a register or stack home lands in it.
    pub(in crate::x64) fn reads_through_rcx(&self, value: Value) -> bool {
        self.has_elided_mem(value)
            || !matches!(self.location(value), Location::Reg(_) | Location::Slot)
    }

    /// `source` and the first of rax and rcx that neither it nor `dst` reads, loading `value` into `dst`, or rax, when `source` reads both.
    pub(in crate::x64) fn spare(
        &mut self,
        dst: Option<Reg>,
        value: Value,
        source: Rm,
    ) -> (Rm, Reg) {
        let reads = |reg: Reg| match source {
            Rm::Reg(r) => r == reg as u8,
            Rm::Mem(mem) => mem.base == reg || mem.index.is_some_and(|(index, _)| index == reg),
        };
        if let Some(free) = [Reg::Rax, Reg::Rcx]
            .into_iter()
            .find(|&reg| Some(reg) != dst && !reads(reg))
        {
            return (source, free);
        }
        let into = dst.unwrap_or(Reg::Rax);
        self.raw(into, value);
        (
            into.into(),
            if into == Reg::Rax { Reg::Rcx } else { Reg::Rax },
        )
    }

    /// Loads `value` into rax and then runs `fill`, which loads `dst` without writing rax; when `dst` is rax, runs `fill` first and loads `value` into rcx. Returns the register that holds `value`.
    pub(in crate::x64) fn beside(
        &mut self,
        dst: Reg,
        value: Value,
        fill: impl FnOnce(&mut Self),
    ) -> Reg {
        if dst == Reg::Rax {
            fill(self);
            self.raw(Reg::Rcx, value);
            return Reg::Rcx;
        }
        self.raw(Reg::Rax, value);
        fill(self);
        Reg::Rax
    }

    /// Runs one step of a parallel move between integer registers, saving through `scratch`, or through the stack when no register is free.
    pub(in crate::x64) fn int_step(&mut self, step: ArgStep<Reg>, scratch: Option<Reg>) {
        match (step, scratch) {
            (ArgStep::Move(src, dst), _) => self.asm.mov(dst, src),
            (ArgStep::Save(src), Some(reg)) => self.asm.mov(reg, src),
            (ArgStep::Save(src), None) => self.asm.push(src),
            (ArgStep::Restore(dst), Some(reg)) => self.asm.mov(dst, reg),
            (ArgStep::Restore(dst), None) => self.asm.pop(dst),
            (ArgStep::Load(val, dst), _) => self.raw(dst, val),
        }
        match step {
            ArgStep::Move(_, reg) | ArgStep::Restore(reg) => self.clobber_reg(reg),
            ArgStep::Save(_) => scratch.into_iter().for_each(|reg| self.clobber_reg(reg)),
            ArgStep::Load(..) => {}
        }
    }

    /// Holds `arg` in edge scratch `index` while a block's parameters move: rcx for the first, the top of the stack for the second.
    pub(in crate::x64) fn save_edge(&mut self, index: usize, arg: Value) {
        if index == 0 {
            return self.raw(Reg::Rcx, arg);
        }
        self.raw(Reg::Rax, arg);
        self.asm.push(Reg::Rax);
        self.shift_base(8);
    }

    /// Writes edge scratch `index`, which `save_edge` filled, into `param`.
    pub(in crate::x64) fn restore_edge(&mut self, index: usize, param: Value) {
        if index == 0 {
            return self.put(param, Reg::Rcx);
        }
        self.asm.pop(Reg::Rax);
        self.shift_base(-8);
        self.clobber_reg(Reg::Rax);
        self.put(param, Reg::Rax);
    }

    fn shift_base(&mut self, by: i32) {
        if self.base == Reg::Rsp {
            self.bias += by;
        }
    }

    /// Reserves `bytes` of scratch below rsp, keeping frame addresses right.
    pub(in crate::x64) fn reserve_scratch(&mut self, bytes: i32) {
        self.asm.alu_imm(Alu::Sub, Reg::Rsp, bytes);
        self.shift_base(bytes);
    }

    /// Releases scratch that `reserve_scratch` took.
    pub(in crate::x64) fn release_scratch(&mut self, bytes: i32) {
        self.asm.alu_imm(Alu::Add, Reg::Rsp, bytes);
        self.shift_base(-bytes);
    }

    pub(super) fn store_reg_home(&mut self, to: Mem, reg: Reg, bytes: u32) {
        self.asm.store(Size::Q, to, reg);
        if let Some(disp) = self.frame_disp(to) {
            if bytes <= 8 {
                self.write_spill(disp, reg);
            } else {
                self.invalidate_disp(disp);
            }
        }
    }

    /// Stores the float or vector of type `ty` in `xmm` to `to` in one store and caches it as the spill of a frame home.
    pub(in crate::x64) fn store_xmm_home(&mut self, to: Mem, xmm: Xmm, ty: Type) {
        self.xmm_store(ty, to, xmm);
        if let Some(disp) = self.frame_disp(to) {
            self.spills.write_fspill(disp, xmm, ty);
        }
    }

    /// Writes eight zero bytes to `to`, joined with the zero stores just before it through the scratch X2.
    pub(in crate::x64) fn zero_to(&mut self, to: Mem) {
        if self.asm.store_zero(to, X2) {
            self.clobber_xmm(X2);
        }
    }

    /// Copies the eight bytes at `from` to `to` through rax.
    pub(in crate::x64) fn copy_word(&mut self, to: Mem, from: Mem) {
        self.asm.load(Size::Q, Reg::Rax, from);
        self.asm.store(Size::Q, to, Reg::Rax);
    }

    /// Writes the eight (or sixteen) bytes of `value` to `to`, using rax as scratch; a float in a register writes the bytes of its type and a 16-byte vector goes through xmm2 in one store.
    pub(in crate::x64) fn move_to(&mut self, to: Mem, value: Value) {
        let stack_home = self.frame_disp(to);
        let ty = self.ty(value);
        match self.location(value) {
            Location::Imm(0) if ty.is_vector() && ty.bytes() >= 16 => {
                self.zero_xmm(X2);
                return self.store_xmm_home(to, X2, ty);
            }
            Location::Imm(0) => self.zero_to(to),
            Location::Imm(imm) if fits_imm32(imm) => {
                self.asm.store_imm_sized(Size::Q, to, imm as i32)
            }
            Location::SlotAddr(sym) if !self.direct_calls => {
                let at = self.asm.store_slot_reloc(to);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::Abs32));
            }
            Location::Imm(_)
            | Location::Frame(..)
            | Location::SlotAddr(_)
            | Location::Symbol(_)
            | Location::Offset(..) => {
                self.raw(Reg::Rax, value);
                self.asm.store(Size::Q, to, Reg::Rax);
            }
            Location::Reg(reg) => return self.store_reg_home(to, reg, ty.bytes()),
            Location::Float(xmm) => return self.store_xmm_home(to, Xmm(xmm), ty),
            Location::Slot => {
                if ty.bytes() <= 8
                    && let Some(cached_reg) = self.cached_spill(value)
                {
                    return self.store_reg_home(to, cached_reg, ty.bytes());
                }
                if let Some(cached_reg) = self
                    .spills
                    .cached_fspill(self.frame.values[value.index()], ty)
                {
                    return self.store_xmm_home(to, cached_reg, ty);
                }
                return self.copy_to(to, self.at(value), ty, X2);
            }
        }
        if let Some(disp) = stack_home {
            self.invalidate_disp(disp);
            if ty.bytes() == 16 {
                self.invalidate_disp(disp + 8);
            }
        }
    }

    /// Broadcasts lane 0 of `v` into `result` straight from the scalar register when fuse elided `v` as an insert into zero; returns whether it did.
    pub(in crate::x64) fn splat_insert(&mut self, v: Value, result: Value) -> bool {
        let Some(Op::Insert(_, scalar, 0)) = self.defs[v.index()]
            .filter(|_| self.plan.elided[v.index()])
            .map(|inst| self.func.insts[inst.index()].op.clone())
        else {
            return false;
        };
        let Some((lane, _)) = self.ty(v).vec_shape() else {
            return false;
        };
        let dst = self.xmm_of(result).unwrap_or(X2);
        let reg = self.reg_of(scalar).unwrap_or_else(|| {
            self.get(Reg::Rax, scalar, Ext::Zero);
            Reg::Rax
        });
        self.clobber_reg(Reg::Rax);
        self.emit_mov_xmm_gpr(dst, reg, lane == Lane::I64);
        self.asm.vex(Vl::X, broadcast(lane), (dst, Xmm(0)), dst);
        self.fput(result, dst);
        true
    }
}
