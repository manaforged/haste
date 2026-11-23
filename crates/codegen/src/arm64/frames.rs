use super::asm::{Cond, Label, Mem, Reg, SP, Size};
use super::{A, C, Emit, IP, T};
use crate::ir::{Op, Value};
use crate::reloc::{FRAME_CACHE_DEPTH, RelocKind, RuntimeSymbol, StencilReloc};

pub(super) enum ColdCall {
    ThreadCell(Label, Label, u32, [Value; 3]),
    FrameAlloc(Label, Label, u32),
    FrameFree(Label, Label, u32),
}

impl Emit<'_> {
    fn thread_cell_sym(&self, slot: Value, lookup: u32) -> Option<u32> {
        if !self.tlv_cells {
            return None;
        }
        match self.defs[slot.index()].map(|inst| &self.func.insts[inst.index()].op) {
            Some(&Op::Symbol(slot_sym)) => Some(slot_sym),
            _ if self.plan.const_of(slot) == Some(0) => Some(lookup),
            _ => None,
        }
    }

    fn emit_tlv_base(&mut self) -> u32 {
        let at = self.asm.pos();
        self.asm.word(0x9000_0000);
        self.asm.word(0x9100_0000);
        self.asm.word(0xF940_0010);
        self.asm.blr(IP);
        at
    }

    pub(super) fn thread_cell_cold(&mut self, callee: Value, args: &[Value]) -> bool {
        if !self.tlv_cells {
            return false;
        }
        let Some((lookup, [slot, size, init])) = crate::alloc::cold_thread_cell(self.func, (callee, args), &self.plan.through, &self.plan.locations, &self.plan.elided) else {
            return false;
        };
        let Some(target_sym) = self.thread_cell_sym(slot, lookup) else { return false };
        let (cold, back) = (self.asm.label(), self.asm.label());
        let at = self.emit_tlv_base();
        self.asm.word(0xF940_0000);
        self.relocs.push(StencilReloc::symbol(at, target_sym, RelocKind::Arm64TlvCell));
        self.asm.cbz(false, Size::X, Reg(0), cold);
        self.asm.bind(back);
        self.cold_calls.push(ColdCall::ThreadCell(cold, back, lookup, [slot, size, init]));
        true
    }

    pub(super) fn cached_frame_cold(&mut self, callee: Value, args: &[Value]) -> bool {
        if !self.tlv_cells {
            return false;
        }
        let Some((rt, bytes)) = crate::alloc::cold_cached_frame((callee, args), &self.plan.through, &self.plan.locations) else {
            return false;
        };
        let (cold, back) = (self.asm.label(), self.asm.label());
        if rt == RuntimeSymbol::Malloc {
            let at = self.emit_tlv_base();
            self.asm.word(0x9100_000A | ((bytes & 0xFFF) << 10));
            self.relocs.push(StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell));
            self.asm.ldp_offset(Reg(0), A, C, 0);
            self.asm.cbz(false, Size::X, Reg(0), cold);
            self.asm.load(Size::X, false, Reg(17), Mem::new(Reg(0), 0));
            self.asm.add_imm(true, false, Size::X, A, A, 1);
            self.asm.stp_offset(Reg(17), A, C, 0);
            self.asm.bind(back);
            self.cold_calls.push(ColdCall::FrameAlloc(cold, back, bytes));
        } else {
            self.raw(A, args[0]);
            let at = self.emit_tlv_base();
            self.asm.word(0x9100_000A | ((bytes & 0xFFF) << 10));
            self.relocs.push(StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell));
            self.asm.ldp_offset(Reg(17), Reg(0), C, 0);
            self.asm.add_imm(true, true, Size::X, super::asm::ZR, Reg(0), u32::from(FRAME_CACHE_DEPTH));
            self.asm.bcond(Cond::Hs, cold);
            self.asm.store(Size::X, Mem::new(A, 0), Reg(17));
            self.asm.add_imm(false, false, Size::X, Reg(0), Reg(0), 1);
            self.asm.stp_offset(A, Reg(0), C, 0);
            self.asm.bind(back);
            self.cold_calls.push(ColdCall::FrameFree(cold, back, bytes));
        }
        true
    }

    fn emit_preserved_cold(&mut self, saved: &[Reg], back: Label, body: impl FnOnce(&mut Self)) {
        let alloc = (saved.len().div_ceil(2) * 16) as u32;
        if alloc > 0 {
            self.asm.add_imm(true, false, Size::X, SP, SP, alloc);
            let mut i = 0;
            while i + 1 < saved.len() {
                self.asm.stp_offset(saved[i], saved[i + 1], SP, (i * 8) as i32);
                i += 2;
            }
            if i < saved.len() {
                self.asm.store(Size::X, Mem::new(SP, (i * 8) as i32), saved[i]);
            }
        }
        body(self);
        if alloc > 0 {
            let mut i = 0;
            while i + 1 < saved.len() {
                self.asm.ldp_offset(saved[i], saved[i + 1], SP, (i * 8) as i32);
                i += 2;
            }
            if i < saved.len() {
                self.asm.load(Size::X, false, saved[i], Mem::new(SP, (i * 8) as i32));
            }
            self.asm.add_imm(false, false, Size::X, SP, SP, alloc);
        }
        self.asm.b(back);
    }

    pub(super) fn emit_cold_calls(&mut self) {
        self.invalidate_spills();
        let mut saved_leaf: Option<Vec<Reg>> = None;
        for call in std::mem::take(&mut self.cold_calls) {
            let saved = saved_leaf.get_or_insert_with(|| crate::alloc::used_preserved_leaf_regs(&super::REGS, &self.plan.locations, false)).clone();
            match call {
                ColdCall::ThreadCell(cold, back, lookup, [slot, size, init]) => {
                    self.asm.bind(cold);
                    self.emit_preserved_cold(&saved, back, |emit| {
                        emit.raw(Reg(0), slot);
                        emit.raw(Reg(1), size);
                        emit.raw(Reg(2), init);
                        let reloc = emit.asm.call_direct_symbol(lookup, true, T);
                        emit.relocs.push(reloc);
                    });
                }
                ColdCall::FrameAlloc(cold, back, bytes) => {
                    self.asm.bind(cold);
                    self.emit_preserved_cold(&saved, back, |emit| {
                        emit.asm.mov_imm(Reg(0), i64::from(bytes));
                        let reloc = emit.asm.call_runtime(RuntimeSymbol::Malloc, true, T);
                        emit.relocs.push(reloc);
                    });
                }
                ColdCall::FrameFree(cold, back, bytes) => {
                    self.asm.bind(cold);
                    self.emit_preserved_cold(&saved, back, |emit| {
                        emit.asm.mov(Reg(0), A);
                        emit.asm.mov_imm(Reg(1), i64::from(bytes));
                        let reloc = emit.asm.call_runtime(RuntimeSymbol::Free, true, T);
                        emit.relocs.push(reloc);
                    });
                }
            }
        }
    }
}
