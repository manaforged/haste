use super::asm::{Label, Reg, Size};
use super::{Emit, IP, T};
use crate::ir::{Op, Value};
use crate::reloc::{RelocKind, StencilReloc};

pub(super) enum ColdCall {
    ThreadCell(Label, Label, u32, [Value; 3]),
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

    pub(super) fn thread_cell_cold(&mut self, lookup: u32, args: &[Value]) -> bool {
        let [slot, size, init] = *args else { return false };
        if !args.iter().all(|&a| self.plan.is_static(a)) {
            return false;
        }
        let Some(target_sym) = self.thread_cell_sym(slot, lookup) else { return false };
        let (cold, back) = (self.asm.label(), self.asm.label());
        let at = self.asm.pos();
        self.asm.word(0x9000_0000);
        self.asm.word(0x9100_0000);
        self.asm.word(0xF940_0010);
        self.asm.blr(IP);
        self.asm.word(0xF940_0000);
        self.relocs.push(StencilReloc::symbol(at, target_sym, RelocKind::Arm64TlvCell));
        self.asm.cbz(false, Size::X, Reg(0), cold);
        self.asm.bind(back);
        self.cold_calls.push(ColdCall::ThreadCell(cold, back, lookup, [slot, size, init]));
        true
    }

    pub(super) fn emit_cold_calls(&mut self) {
        self.invalidate_spills();
        for call in std::mem::take(&mut self.cold_calls) {
            let ColdCall::ThreadCell(cold, back, lookup, [slot, size, init]) = call;
            self.asm.bind(cold);
            self.raw(Reg(0), slot);
            self.raw(Reg(1), size);
            self.raw(Reg(2), init);
            let reloc = self.asm.call_direct_symbol(lookup, true, T);
            self.relocs.push(reloc);
            self.asm.b(back);
        }
    }
}
