use super::Emit;
use super::asm::{Asm, Cond, Label, Mem, Reg, Size};
use crate::alloc::CallTarget;
use crate::ir::{Op, Value};
use crate::reloc::{FRAME_CACHE_DEPTH, FRAME_CACHE_LIMIT, RelocKind, RuntimeSymbol, StencilReloc};

impl Asm {
    /// Encodes `opcode` with REX.W on the quadword `disp` bytes from the thread pointer, register field `r`, then `imm`; returns the offset of `disp`.
    fn fs_quad(&mut self, opcode: u8, r: u8, disp: i32, imm: Option<u8>) -> u32 {
        self.code.extend([0x64, 0x48 | ((r >> 3) & 1) << 2, opcode, (r & 7) << 3 | 4, 0x25]);
        let at = self.pos();
        self.code.extend(disp.to_le_bytes());
        self.code.extend(imm);
        at
    }
}

impl Emit<'_> {
    /// Emits the fast path of a runtime call the target can serve inline, with the call itself out of line; false when it has none.
    pub(super) fn inline_call(&mut self, direct: Option<CallTarget>, args: &[Value]) -> bool {
        match direct {
            Some(CallTarget::Runtime(rt)) => {
                let bytes = args.last().and_then(|&bytes| self.const_of(bytes));
                self.cached_frame(rt, bytes)
            }
            _ => false,
        }
    }

    /// Takes a cached frame of `bytes` for runtime `rt` from the thread's frame cache inline, calling the runtime only on a miss; false when the cache does not apply.
    pub(super) fn cached_frame(&mut self, rt: RuntimeSymbol, bytes: Option<i64>) -> bool {
        let Some(bytes) = bytes.filter(|&b| self.frame_cache && (0..i64::from(FRAME_CACHE_LIMIT)).contains(&b)) else { return false };
        if !matches!(rt, RuntimeSymbol::Malloc | RuntimeSymbol::Free) {
            return false;
        }
        let head = bytes as i32;
        let (cold, back) = (self.asm.label(), self.asm.label());
        if rt == RuntimeSymbol::Malloc {
            self.thread_reloc(rt, |asm| asm.fs_quad(0x8B, Reg::Rax as u8, head, None));
            self.asm.test(Reg::Rax, Reg::Rax);
            self.asm.jcc(Cond::E, cold);
            self.asm.load(Size::Q, Reg::Rcx, Mem::new(Reg::Rax, 0));
            self.thread_reloc(rt, |asm| asm.fs_quad(0x89, Reg::Rcx as u8, head, None));
            self.thread_reloc(rt, |asm| asm.fs_quad(0xFF, 1, head + 8, None));
        } else {
            self.thread_reloc(rt, |asm| asm.fs_quad(0x83, 7, head + 8, Some(FRAME_CACHE_DEPTH)));
            self.asm.jcc(Cond::Ae, cold);
            self.thread_reloc(rt, |asm| asm.fs_quad(0x8B, Reg::Rcx as u8, head, None));
            self.asm.store(Size::Q, Mem::new(Reg::Rdi, 0), Reg::Rcx);
            self.thread_reloc(rt, |asm| asm.fs_quad(0x89, Reg::Rdi as u8, head, None));
            self.thread_reloc(rt, |asm| asm.fs_quad(0xFF, 0, head + 8, None));
        }
        self.asm.bind(back);
        self.cold_calls.push((cold, back, StencilReloc::runtime(0, rt, RelocKind::PcRel32)));
        true
    }

    /// Reads the calling thread's cell of thread slot `slot` from the static TLS slot table into rax and jumps to the returned label when the cell is made, so only a miss runs the call that follows; `None` when the table does not apply.
    pub(super) fn thread_cell_hit(&mut self, slot: Value, lookup: u32) -> Option<Label> {
        if !self.frame_cache {
            return None;
        }
        let target_sym = match self.defs[slot.index()].map(|inst| &self.func.insts[inst.index()].op) {
            Some(&Op::Symbol(slot_sym)) => slot_sym,
            _ if self.const_of(slot) == Some(0) => lookup,
            _ => return None,
        };
        let hit = self.asm.label();
        let at = self.asm.fs_quad(0x8B, Reg::Rax as u8, 0, None);
        self.relocs.push(StencilReloc::symbol(at, target_sym, RelocKind::ThreadOff32));
        self.asm.test(Reg::Rax, Reg::Rax);
        self.asm.jcc(Cond::Ne, hit);
        Some(hit)
    }

    fn thread_reloc(&mut self, rt: RuntimeSymbol, encode: impl FnOnce(&mut Asm) -> u32) {
        let at = encode(&mut self.asm);
        self.relocs.push(StencilReloc::runtime(at, rt, RelocKind::ThreadOff32));
    }

    /// Emits the out-of-line runtime calls of cache misses, each returning to the code after its inline path.
    pub(super) fn emit_cold_calls(&mut self) {
        for (cold, back, target) in std::mem::take(&mut self.cold_calls) {
            self.asm.bind(cold);
            let offset = self.asm.call_pcrel32();
            self.relocs.push(StencilReloc { offset, ..target });
            self.asm.jmp(back);
        }
    }
}
