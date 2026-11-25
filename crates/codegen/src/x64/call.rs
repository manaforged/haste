use super::asm::{Mem, Reg, Size};
use super::frame::classify;
use super::{Abi, Emit};
use crate::CallAbi;
use crate::alloc::CallTarget;
use crate::frame::{Loc, is_tail};
use crate::ir::{ArgumentPurpose, Inst, Op, SigRef, Value};
use crate::reloc::{RelocKind, StencilReloc};

impl Emit<'_> {
    fn copy_struct(&mut self, pointer: Value, offset: u32, size: u32) {
        self.raw(Reg::Rax, pointer);
        let mut done = 0;
        for (width, bytes) in [(Size::Q, 8), (Size::D, 4), (Size::W, 2), (Size::B, 1)] {
            while size - done >= bytes {
                self.asm
                    .load(width, Reg::Rcx, Mem::new(Reg::Rax, done as i32));
                self.asm
                    .store(width, Mem::new(Reg::Rsp, (offset + done) as i32), Reg::Rcx);
                done += bytes;
            }
        }
        self.clobber_reg(Reg::Rcx);
    }

    fn emit_branch_target(&mut self, direct: Option<CallTarget>, held: Option<Reg>, tail: bool) {
        let emit_reg = |asm: &mut super::asm::Asm, reg: Reg| {
            if tail {
                asm.jmp_reg(reg);
            } else {
                asm.call(reg);
            }
        };
        match direct {
            Some(CallTarget::Abs(at)) => self.asm.branch_absolute(at, tail),
            Some(CallTarget::Slot(sym)) => {
                let reloc = self.asm.branch_symbol(sym, self.direct_calls, tail);
                self.relocs.push(reloc);
            }
            Some(CallTarget::Runtime(rt)) if self.direct_calls => {
                let at = self.asm.branch_pcrel32(tail);
                self.relocs
                    .push(StencilReloc::runtime(at, rt, RelocKind::PcRel32));
            }
            Some(CallTarget::Runtime(rt)) => {
                let at = self.asm.mov_abs64(Reg::R11);
                self.relocs
                    .push(StencilReloc::runtime(at, rt, RelocKind::Abs64));
                emit_reg(&mut self.asm, Reg::R11);
            }
            Some(CallTarget::Symbol(sym)) if self.direct_calls => {
                let at = self.asm.branch_pcrel32(tail);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::PcRel32));
            }
            Some(CallTarget::Symbol(sym)) => {
                let at = self.asm.mov_abs64(Reg::R11);
                self.relocs
                    .push(StencilReloc::symbol(at, sym, RelocKind::Abs64));
                emit_reg(&mut self.asm, Reg::R11);
            }
            None => emit_reg(&mut self.asm, held.unwrap_or(Reg::R11)),
        }
    }

    pub(super) fn arguments(
        &mut self,
        sig: SigRef,
        callee: Value,
        args: &[Value],
        is_try: bool,
    ) -> Result<Abi, String> {
        let abi = classify(&self.func.sigs[sig.index()], self.abi_kind)?;
        for (arg, loc) in args.iter().zip(&abi.params) {
            match *loc {
                Loc::Stack(offset) => self.move_to(Mem::new(Reg::Rsp, offset as i32), *arg),
                Loc::Struct(offset, size) => self.copy_struct(*arg, offset, size),
                _ => {}
            }
        }
        let direct = self.plan.through[callee.index()];
        if !is_try && (self.thread_cell_cold(callee, args) || self.cached_frame_cold(callee, args))
        {
            self.clobber_call_spills();
            return Ok(abi);
        }
        let cell = match direct {
            Some(CallTarget::Symbol(lookup)) if self.func.is_thread_cell(callee) => args
                .first()
                .and_then(|&slot| self.thread_cell_hit(slot, lookup))
                .map(|hit| (lookup, hit)),
            _ => None,
        };
        if cell.is_some() {
            self.clobber_reg(Reg::Rax);
        }
        let held = self
            .reg_of(callee)
            .filter(|&reg| !matches!(reg, Reg::Rax | Reg::Rcx) && !abi.clobbers(reg));
        let (mut values, mut locs) = (args.to_vec(), abi.params.clone());
        if direct.is_none() && held.is_none() {
            values.push(callee);
            locs.push(Loc::Int(Reg::R11));
        }
        let float_count = self.move_locs(&values, &locs, Some(Reg::Rax));
        let func = self.func;
        self.leave_wide(&func.sigs[sig.index()].params);
        if self.abi_kind == CallAbi::SysV
            && self.func.sigs[sig.index()]
                .params
                .iter()
                .any(|param| param.purpose == ArgumentPurpose::Variadic)
        {
            self.asm.mov_imm(Reg::Rax, float_count as i64);
        }
        self.asm.flags_overwritten();
        if let Some((lookup, hit)) = cell {
            let at = self.asm.call_pcrel32();
            self.relocs
                .push(StencilReloc::symbol(at, lookup, RelocKind::PcRel32));
            self.asm.bind(hit);
            self.clobber_call_spills();
            return Ok(abi);
        }
        if self.inline_call(direct, args) {
            self.clobber_call_spills();
            return Ok(abi);
        }
        self.emit_branch_target(direct, held, false);
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
        let abi = classify(&self.func.sigs[sig.index()], self.abi_kind)?;
        let direct = self.plan.through[callee.index()];
        let held = self.reg_of(*callee).filter(|&reg| {
            !matches!(reg, Reg::Rax | Reg::Rcx | Reg::Rbp | Reg::Rsp)
                && !self.frame.saves.contains(&reg)
                && !abi.clobbers(reg)
        });
        let (mut values, mut locs) = (args.to_vec(), abi.params.clone());
        if direct.is_none() && held.is_none() {
            values.push(*callee);
            locs.push(Loc::Int(Reg::R11));
        }
        let float_count = self.move_locs(&values, &locs, Some(Reg::Rax));
        let func = self.func;
        self.leave_wide(&func.sigs[sig.index()].params);
        if self.abi_kind == CallAbi::SysV
            && func.sigs[sig.index()]
                .params
                .iter()
                .any(|p| p.purpose == ArgumentPurpose::Variadic)
        {
            self.asm.mov_imm(Reg::Rax, float_count as i64);
        }
        if !self.in_fast {
            self.epilogue_body();
        }
        self.emit_branch_target(direct, held, true);
        self.invalidate_spills();
        Ok(true)
    }
}
