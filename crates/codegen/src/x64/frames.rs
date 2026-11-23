use super::Emit;
use super::asm::{Alu, Asm, Cond, Label, Mem, Reg, Size};
use crate::alloc::CallTarget;
use crate::CallAbi;
use crate::ir::{ArgumentPurpose, Op, Value};
use crate::reloc::{FRAME_CACHE_DEPTH, FRAME_CACHE_LIMIT, RelocKind, RuntimeSymbol, StencilReloc};

pub(super) enum ColdCall {
    Frame(Label, Label, StencilReloc),
    FramePreserved(Label, Label, RuntimeSymbol, u32, bool),
    ThreadCell(Label, Label, u32, [Value; 3], bool),
}

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
        self.cold_calls.push(ColdCall::Frame(cold, back, StencilReloc::runtime(0, rt, RelocKind::PcRel32)));
        true
    }

    fn thread_cell_sym(&self, slot: Value, lookup: u32) -> Option<u32> {
        if !self.frame_cache {
            return None;
        }
        match self.defs[slot.index()].map(|inst| &self.func.insts[inst.index()].op) {
            Some(&Op::Symbol(slot_sym)) => Some(slot_sym),
            _ if self.const_of(slot) == Some(0) => Some(lookup),
            _ => None,
        }
    }

    /// Reads the calling thread's cell of thread slot `slot` from the static TLS slot table into rax and jumps to the returned label when the cell is made, so only a miss runs the call that follows; `None` when the table does not apply.
    pub(super) fn thread_cell_hit(&mut self, slot: Value, lookup: u32) -> Option<Label> {
        let target_sym = self.thread_cell_sym(slot, lookup)?;
        let hit = self.asm.label();
        let at = self.asm.fs_quad(0x8B, Reg::Rax as u8, 0, None);
        self.relocs.push(StencilReloc::symbol(at, target_sym, RelocKind::ThreadOff32));
        self.asm.test(Reg::Rax, Reg::Rax);
        self.asm.jcc(Cond::Ne, hit);
        Some(hit)
    }

    /// Reads a thread cell inline with a straight-line fast path and a register-preserving out-of-line cold miss stub when all arguments are static.
    pub(super) fn thread_cell_cold(&mut self, callee: Value, args: &[Value]) -> bool {
        if self.abi_kind != CallAbi::SysV || !self.frame_cache {
            return false;
        }
        let Some((lookup, [slot, size, init])) = crate::alloc::cold_thread_cell(self.func, (callee, args), &self.plan.through, &self.plan.locations, &self.plan.elided) else {
            return false;
        };
        let Some(target_sym) = self.thread_cell_sym(slot, lookup) else { return false };
        let (cold, back) = (self.asm.label(), self.asm.label());
        let at = self.asm.fs_quad(0x8B, Reg::Rax as u8, 0, None);
        self.relocs.push(StencilReloc::symbol(at, target_sym, RelocKind::ThreadOff32));
        self.asm.test(Reg::Rax, Reg::Rax);
        self.asm.jcc(Cond::E, cold);
        self.asm.bind(back);
        self.cold_calls.push(ColdCall::ThreadCell(cold, back, lookup, [slot, size, init], !self.framed || self.in_fast));
        true
    }

    fn thread_reloc(&mut self, rt: RuntimeSymbol, encode: impl FnOnce(&mut Asm) -> u32) {
        let at = encode(&mut self.asm);
        self.relocs.push(StencilReloc::runtime(at, rt, RelocKind::ThreadOff32));
    }

    /// Takes or returns a cached coroutine frame inline with a register-preserving out-of-line cold miss stub.
    pub(super) fn cached_frame_cold(&mut self, callee: Value, args: &[Value]) -> bool {
        if self.abi_kind != CallAbi::SysV || !self.frame_cache {
            return false;
        }
        let Some((rt, bytes)) = crate::alloc::cold_cached_frame((callee, args), &self.plan.through, &self.plan.locations) else {
            return false;
        };
        let head = bytes as i32;
        let (cold, back) = (self.asm.label(), self.asm.label());
        self.asm.flags_overwritten();
        if rt == RuntimeSymbol::Malloc {
            self.thread_reloc(rt, |asm| asm.fs_quad(0x8B, Reg::Rax as u8, head, None));
            self.asm.test(Reg::Rax, Reg::Rax);
            self.asm.jcc(Cond::E, cold);
            self.asm.load(Size::Q, Reg::Rcx, Mem::new(Reg::Rax, 0));
            self.thread_reloc(rt, |asm| asm.fs_quad(0x89, Reg::Rcx as u8, head, None));
            self.thread_reloc(rt, |asm| asm.fs_quad(0xFF, 1, head + 8, None));
        } else {
            self.raw(Reg::Rax, args[0]);
            self.thread_reloc(rt, |asm| asm.fs_quad(0x83, 7, head + 8, Some(FRAME_CACHE_DEPTH)));
            self.asm.jcc(Cond::Ae, cold);
            self.thread_reloc(rt, |asm| asm.fs_quad(0x8B, Reg::Rcx as u8, head, None));
            self.asm.store(Size::Q, Mem::new(Reg::Rax, 0), Reg::Rcx);
            self.thread_reloc(rt, |asm| asm.fs_quad(0x89, Reg::Rax as u8, head, None));
            self.thread_reloc(rt, |asm| asm.fs_quad(0xFF, 0, head + 8, None));
        }
        self.asm.bind(back);
        self.cold_calls.push(ColdCall::FramePreserved(cold, back, rt, bytes, !self.framed || self.in_fast));
        true
    }

    fn emit_preserved_cold(&mut self, saved: &[Reg], back: Label, base_unaligned: bool, body: impl FnOnce(&mut Self)) {
        let needs_align = base_unaligned ^ (saved.len() % 2 == 1);
        for &r in saved { self.asm.push(r); }
        if needs_align { self.asm.alu_imm(Alu::Sub, Reg::Rsp, 8); }
        body(self);
        if needs_align { self.asm.alu_imm(Alu::Add, Reg::Rsp, 8); }
        for &r in saved.iter().rev() { self.asm.pop(r); }
        self.asm.jmp(back);
    }

    /// Emits the out-of-line runtime calls of cache misses, each returning to the code after its inline path.
    pub(super) fn emit_cold_calls(&mut self) {
        self.invalidate_spills();
        let mut saved_leaf: Option<Vec<Reg>> = None;
        for call in std::mem::take(&mut self.cold_calls) {
            match call {
                ColdCall::Frame(cold, back, target) => {
                    self.asm.bind(cold);
                    let offset = self.asm.call_pcrel32();
                    self.relocs.push(StencilReloc { offset, ..target });
                    self.asm.jmp(back);
                }
                ColdCall::FramePreserved(cold, back, rt, bytes, base_unaligned) => {
                    self.asm.bind(cold);
                    let saved = saved_leaf.get_or_insert_with(|| {
                        let has_dict = self.func.signature.params.iter().any(|p| p.purpose == ArgumentPurpose::Dict);
                        crate::alloc::used_preserved_leaf_regs(&super::REGS, &self.plan.locations, has_dict)
                    }).clone();
                    self.emit_preserved_cold(&saved, back, base_unaligned, |emit| {
                        if rt == RuntimeSymbol::Malloc {
                            emit.asm.mov_imm(Reg::Rdi, i64::from(bytes));
                        } else {
                            emit.asm.mov(Reg::Rdi, Reg::Rax);
                            emit.asm.mov_imm(Reg::Rsi, i64::from(bytes));
                        }
                        emit.asm.flags_overwritten();
                        let offset = emit.asm.call_pcrel32();
                        emit.relocs.push(StencilReloc::runtime(offset, rt, RelocKind::PcRel32));
                    });
                }
                ColdCall::ThreadCell(cold, back, lookup, [slot, size, init], base_unaligned) => {
                    self.asm.bind(cold);
                    let saved = saved_leaf.get_or_insert_with(|| {
                        let has_dict = self.func.signature.params.iter().any(|p| p.purpose == ArgumentPurpose::Dict);
                        crate::alloc::used_preserved_leaf_regs(&super::REGS, &self.plan.locations, has_dict)
                    }).clone();
                    self.emit_preserved_cold(&saved, back, base_unaligned, |emit| {
                        emit.raw(Reg::Rdi, slot);
                        emit.raw(Reg::Rsi, size);
                        emit.raw(Reg::Rdx, init);
                        emit.asm.flags_overwritten();
                        let offset = emit.asm.call_pcrel32();
                        emit.relocs.push(StencilReloc::symbol(offset, lookup, RelocKind::PcRel32));
                    });
                }
            }
        }
    }
}
