use super::asm::{Cond, Label, Mem, Reg, SP, Size};
use super::{A, Emit, IP, T};
use crate::ir::{Op, Value};
use crate::reloc::{FRAME_CACHE_DEPTH, FRAME_CACHE_LIMIT, RelocKind, RuntimeSymbol, StencilReloc};

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

    pub(super) fn cached_frame_cold(&mut self, rt: RuntimeSymbol, args: &[Value]) -> bool {
        if !self.tlv_cells || !matches!(rt, RuntimeSymbol::Malloc | RuntimeSymbol::Free) {
            return false;
        }
        let Some(bytes) = args.last().and_then(|&b| self.plan.const_of(b)).filter(|&b| (0..i64::from(FRAME_CACHE_LIMIT)).contains(&b) && b % 16 == 0) else {
            return false;
        };
        let bytes = bytes as u32;
        let (cold, back) = (self.asm.label(), self.asm.label());
        if rt == RuntimeSymbol::Malloc {
            let at = self.emit_tlv_base();
            self.asm.word(0x9100_0002 | ((bytes & 0xFFF) << 10));
            self.relocs.push(StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell));
            self.asm.ldp_offset(Reg(0), A, Reg(2), 0);
            self.asm.cbz(false, Size::X, Reg(0), cold);
            self.asm.load(Size::X, false, Reg(8), Mem::new(Reg(0), 0));
            self.asm.add_imm(true, false, Size::X, A, A, 1);
            self.asm.stp_offset(Reg(8), A, Reg(2), 0);
            self.asm.bind(back);
            self.cold_calls.push(ColdCall::FrameAlloc(cold, back, bytes));
        } else {
            let Some(&ptr) = args.first() else { return false };
            self.raw(A, ptr);
            let at = self.emit_tlv_base();
            self.asm.word(0x9100_0002 | ((bytes & 0xFFF) << 10));
            self.relocs.push(StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell));
            self.asm.ldp_offset(Reg(11), Reg(8), Reg(2), 0);
            self.asm.add_imm(true, true, Size::X, super::asm::ZR, Reg(8), u32::from(FRAME_CACHE_DEPTH));
            self.asm.bcond(Cond::Hs, cold);
            self.asm.store(Size::X, Mem::new(A, 0), Reg(11));
            self.asm.add_imm(false, false, Size::X, Reg(8), Reg(8), 1);
            self.asm.stp_offset(A, Reg(8), Reg(2), 0);
            self.asm.bind(back);
            self.cold_calls.push(ColdCall::FrameFree(cold, back, bytes));
        }
        true
    }

    pub(super) fn emit_cold_calls(&mut self) {
        self.invalidate_spills();
        let mut saved_leaf: Option<Vec<Reg>> = None;
        for call in std::mem::take(&mut self.cold_calls) {
            match call {
                ColdCall::ThreadCell(cold, back, lookup, [slot, size, init]) => {
                    self.asm.bind(cold);
                    let saved = saved_leaf.get_or_insert_with(|| crate::alloc::used_preserved_leaf_regs(&super::REGS, &self.plan.locations, false));
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
                    self.raw(Reg(0), slot);
                    self.raw(Reg(1), size);
                    self.raw(Reg(2), init);
                    let reloc = self.asm.call_direct_symbol(lookup, true, T);
                    self.relocs.push(reloc);
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
                ColdCall::FrameAlloc(cold, back, bytes) => {
                    self.asm.bind(cold);
                    self.asm.mov_imm(Reg(0), i64::from(bytes));
                    let reloc = self.asm.call_runtime(RuntimeSymbol::Malloc, true, T);
                    self.relocs.push(reloc);
                    self.asm.b(back);
                }
                ColdCall::FrameFree(cold, back, bytes) => {
                    self.asm.bind(cold);
                    self.asm.mov(Reg(0), A);
                    self.asm.mov_imm(Reg(1), i64::from(bytes));
                    let reloc = self.asm.call_runtime(RuntimeSymbol::Free, true, T);
                    self.relocs.push(reloc);
                    self.asm.b(back);
                }
            }
        }
    }
}
