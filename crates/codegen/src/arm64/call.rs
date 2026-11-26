use super::asm::{Asm, Mem, Reg, Size};
use super::frame::classify;
use super::{A, Abi, C, D, Emit, FP, IP, LR, S, SP, T};
use crate::alloc::CallTarget;
use crate::frame::{Loc, is_tail};
use crate::ir::{Inst, Op, SigRef, Value};
use crate::reloc::{RelocKind, RuntimeSymbol, StencilReloc};

impl Asm {
    fn emit_call_reloc(
        &mut self,
        direct: bool,
        load_slot: bool,
        tail: bool,
        scratch: Reg,
        build: impl FnOnce(u32, RelocKind) -> StencilReloc,
    ) -> StencilReloc {
        if direct {
            let at = if tail {
                self.b_reloc()
            } else {
                self.bl_reloc()
            };
            return build(at, RelocKind::Arm64Call26);
        }
        let at = self.mov_symbol(scratch);
        if load_slot {
            self.load(Size::X, false, scratch, Mem::new(scratch, 0));
        }
        if tail {
            self.br(scratch);
        } else {
            self.blr(scratch);
        }
        build(at, RelocKind::Arm64Movw64)
    }

    /// `bl` to relocation symbol `sym` when `direct`, else a load of its call table slot into `scratch` and a `blr` through it.
    pub(in crate::arm64) fn call_symbol(
        &mut self,
        sym: u32,
        direct: bool,
        scratch: Reg,
    ) -> StencilReloc {
        self.emit_call_reloc(direct, true, false, scratch, |at, kind| {
            StencilReloc::symbol(at, sym, kind)
        })
    }

    /// `bl` to direct symbol `sym` when `direct`, else `movz/movk` into `scratch` and `blr`.
    pub(in crate::arm64) fn call_direct_symbol(
        &mut self,
        sym: u32,
        direct: bool,
        scratch: Reg,
    ) -> StencilReloc {
        self.emit_call_reloc(direct, false, false, scratch, |at, kind| {
            StencilReloc::symbol(at, sym, kind)
        })
    }

    /// `bl` to runtime helper `rt` when `direct`, else `movz/movk` into `scratch` and `blr`.
    pub(in crate::arm64) fn call_runtime(
        &mut self,
        rt: RuntimeSymbol,
        direct: bool,
        scratch: Reg,
    ) -> StencilReloc {
        self.emit_call_reloc(direct, false, false, scratch, |at, kind| {
            StencilReloc::runtime(at, rt, kind)
        })
    }
}

impl Emit<'_> {
    fn emit_branch_target(&mut self, direct: Option<CallTarget>, target: Reg, tail: bool) {
        match direct {
            Some(CallTarget::Slot(sym)) => self.relocs.push(self.asm.emit_call_reloc(
                self.direct_calls,
                true,
                tail,
                T,
                |at, kind| StencilReloc::symbol(at, sym, kind),
            )),
            Some(CallTarget::Abs(at)) => {
                self.asm.mov_imm(T, i64::from(at));
                if tail {
                    self.asm.br(T);
                } else {
                    self.asm.blr(T);
                }
            }
            Some(CallTarget::Runtime(rt)) => self.relocs.push(self.asm.emit_call_reloc(
                self.direct_calls,
                false,
                tail,
                T,
                |at, kind| StencilReloc::runtime(at, rt, kind),
            )),
            Some(CallTarget::Symbol(sym)) => self.relocs.push(self.asm.emit_call_reloc(
                self.direct_calls,
                false,
                tail,
                T,
                |at, kind| StencilReloc::symbol(at, sym, kind),
            )),
            None if tail => self.asm.br(target),
            None => self.asm.blr(target),
        }
    }

    pub(super) fn arguments(
        &mut self,
        sig: SigRef,
        callee: Value,
        args: &[Value],
        is_try: bool,
    ) -> Result<Abi, String> {
        let sig_data = &self.func.sigs[sig.index()];
        let abi = classify(sig_data, self.packing)?;
        for ((arg, loc), param) in args.iter().zip(&abi.params).zip(&sig_data.params) {
            if let Loc::Stack(offset) = *loc {
                self.stack_argument(offset, param.purpose, *arg);
            }
        }
        let direct = self.plan.through[callee.index()];
        if !is_try {
            let inlined =
                self.thread_cell_cold(callee, args) || self.cached_frame_cold(callee, args);
            if inlined {
                self.clobber_regs(&[Reg(0), A, C, IP, Reg(17), LR]);
                self.spills.clear_fspill();
                self.asm.clear_anchor();
                return Ok(abi);
            }
        }
        let (target, extra) = if direct.is_none() {
            match self
                .plan
                .reg_of(callee)
                .filter(|&reg| !matches!(reg, A | C | IP | FP | LR | SP) && !abi.clobbers(reg))
            {
                Some(reg) => (reg, None),
                None => (T, Some((callee, T))),
            }
        } else {
            (T, None)
        };
        self.move_locs(args, &abi.params, extra);
        self.emit_branch_target(direct, target, false);
        self.clobber_call_spills();
        Ok(abi)
    }

    pub(super) fn emit_tail_call(&mut self, inst: Inst) -> Result<bool, String> {
        if !is_tail(&self.tail_calls, inst) {
            return Ok(false);
        }
        let Op::Call {
            sig, callee, args, ..
        } = &self.func.insts[inst.index()].op
        else {
            return Ok(false);
        };
        let abi = classify(&self.func.sigs[sig.index()], self.packing)?;
        let direct = self.plan.through[callee.index()];
        let (target, extra) = if direct.is_none() {
            match self.plan.reg_of(*callee).filter(|&reg| {
                matches!(reg, D | S | T | Reg(14) | Reg(15))
                    && !matches!(reg, A | C | IP | FP | LR | SP)
                    && !self.frame.saves.contains(&reg)
                    && !abi.clobbers(reg)
            }) {
                Some(reg) => (reg, None),
                None => (T, Some((*callee, T))),
            }
        } else {
            (T, None)
        };
        self.move_locs(args, &abi.params, extra);
        if !self.in_fast {
            self.epilogue_body();
        }
        self.emit_branch_target(direct, target, true);
        self.invalidate_spills();
        Ok(true)
    }
}
