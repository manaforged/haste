use super::asm::{Asm, Cond, LR, Label, Mem, Reg, SP, Size, ZR};
use super::{A, C, Emit, IP, T};
use crate::alloc::Plan;
use crate::frame::Frame;
use crate::ir::{Function, Inst, Op, Value};
use crate::reloc::{FRAME_CACHE_DEPTH, RelocKind, RuntimeSymbol, StencilReloc};

const ARM64_NOP: u32 = 0xD503_201F;

#[derive(Clone, Copy)]
pub(super) enum TlvInitSym {
    Symbol(u32),
    Runtime(RuntimeSymbol),
}

pub(super) struct TlvCache {
    pub(super) slot: i32,
    helper: Label,
    init_sym: TlvInitSym,
    pub(super) dom_known: Vec<bool>,
    pub(super) live: bool,
    used: bool,
}

pub(super) enum ColdCall {
    ThreadCell(Label, Label, u32, [Value; 3]),
    FrameAlloc(Label, Label, u32),
    FrameFree(Label, Label, u32),
}

fn thread_cell_target(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    slot: Value,
    lookup: u32,
) -> Option<u32> {
    match defs[slot.index()].map(|inst| &func.insts[inst.index()].op) {
        Some(&Op::Symbol(slot_sym)) => Some(slot_sym),
        _ if plan.const_of(slot) == Some(0) => Some(lookup),
        _ => None,
    }
}

fn tlv_call_sym(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    callee: Value,
    args: &[Value],
) -> Option<TlvInitSym> {
    if let Some((lookup, [slot, ..])) = crate::alloc::cold_thread_cell(
        func,
        (callee, args),
        &plan.through,
        &plan.locations,
        &plan.elided,
    ) && let Some(sym) = thread_cell_target(func, plan, defs, slot, lookup)
    {
        return Some(TlvInitSym::Symbol(sym));
    }
    crate::alloc::cold_cached_frame((callee, args), &plan.through, &plan.locations)
        .map(|(rt, _)| TlvInitSym::Runtime(rt))
}

pub(super) fn plan_tlv_cache(
    func: &Function,
    plan: &Plan<Reg>,
    defs: &[Option<Inst>],
    frame: &mut Frame<Reg>,
    asm: &mut Asm,
) -> Option<TlvCache> {
    let mut total = 0_usize;
    let mut first_sym = None;
    let mut has_tlv = vec![false; func.blocks.len()];
    for (b_idx, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            if let Op::Call { callee, args, .. } = &func.insts[inst.index()].op
                && let Some(sym) = tlv_call_sym(func, plan, defs, *callee, args)
            {
                total += 1;
                has_tlv[b_idx] = true;
                first_sym.get_or_insert(sym);
            }
        }
    }
    if total < 2 {
        return None;
    }
    let cold = crate::live::cold(func);
    if !has_tlv.iter().enumerate().any(|(b, &has)| has && !cold[b]) {
        return None;
    }
    let cfg = func.cfg();
    let mut dom_known = vec![false; func.blocks.len()];
    for b in 0..func.blocks.len() {
        let mut cur = b;
        while let Some(d) = cfg.idom[cur] {
            let di = d.index();
            if di == cur {
                break;
            }
            if has_tlv[di] {
                dom_known[b] = true;
                break;
            }
            cur = di;
        }
    }
    let slot = frame.add_slot(8);
    let helper = asm.label();
    Some(TlvCache {
        slot,
        helper,
        init_sym: first_sym?,
        dom_known,
        live: false,
        used: false,
    })
}

impl Emit<'_> {
    fn thread_cell_sym(&self, slot: Value, lookup: u32) -> Option<u32> {
        if !self.tlv_cells {
            return None;
        }
        thread_cell_target(self.func, &self.plan, &self.defs, slot, lookup)
    }

    fn emit_tlv_base(&mut self) -> u32 {
        let at = self.asm.pos();
        self.asm.word(0x9000_0000);
        self.asm.word(0x9100_0000);
        self.asm.word(0xF940_0010);
        self.asm.blr(IP);
        at
    }

    fn emit_cached_tlv_base(&mut self) -> bool {
        let Some((slot, helper, known)) =
            self.tlv_cache.as_ref().map(|c| (c.slot, c.helper, c.live))
        else {
            return false;
        };
        let at = self.frame_mem(slot);
        self.asm.load(Size::X, false, Reg(0), at);
        if !known {
            let skip = self.asm.label();
            self.asm.cbz(true, Size::X, Reg(0), skip);
            self.asm.bl(helper);
            self.asm.bind(skip);
            if let Some(cache) = &mut self.tlv_cache {
                cache.live = true;
                cache.used = true;
            }
        }
        true
    }

    pub(super) fn thread_cell_cold(&mut self, callee: Value, args: &[Value]) -> bool {
        if !self.tlv_cells {
            return false;
        }
        let Some((lookup, [slot, size, init])) = crate::alloc::cold_thread_cell(
            self.func,
            (callee, args),
            &self.plan.through,
            &self.plan.locations,
            &self.plan.elided,
        ) else {
            return false;
        };
        let Some(target_sym) = self.thread_cell_sym(slot, lookup) else {
            return false;
        };
        let (cold, back) = (self.asm.label(), self.asm.label());
        if self.emit_cached_tlv_base() {
            let at = self.asm.pos();
            self.asm.word(0xF940_0000);
            self.relocs
                .push(StencilReloc::symbol(at, target_sym, RelocKind::ThreadOff32));
        } else {
            let at = self.emit_tlv_base();
            self.asm.word(0xF940_0000);
            self.relocs.push(StencilReloc::symbol(
                at,
                target_sym,
                RelocKind::Arm64TlvCell,
            ));
        }
        self.asm.cbz(false, Size::X, Reg(0), cold);
        self.asm.bind(back);
        self.cold_calls
            .push(ColdCall::ThreadCell(cold, back, lookup, [slot, size, init]));
        true
    }

    pub(super) fn cached_frame_cold(&mut self, callee: Value, args: &[Value]) -> bool {
        if !self.tlv_cells {
            return false;
        }
        let Some((rt, bytes)) = crate::alloc::cold_cached_frame(
            (callee, args),
            &self.plan.through,
            &self.plan.locations,
        ) else {
            return false;
        };
        let (cold, back) = (self.asm.label(), self.asm.label());
        let off_word = 0x9100_000A | ((bytes & 0xFFF) << 10);
        if rt == RuntimeSymbol::Malloc {
            if self.emit_cached_tlv_base() {
                let at = self.asm.pos();
                self.asm.word(off_word);
                self.relocs
                    .push(StencilReloc::runtime(at, rt, RelocKind::ThreadOff32));
            } else {
                let at = self.emit_tlv_base();
                self.asm.word(off_word);
                self.relocs
                    .push(StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell));
            }
            self.asm.ldp_offset(Reg(0), A, C, 0);
            self.asm.cbz(false, Size::X, Reg(0), cold);
            self.asm.load(Size::X, false, Reg(17), Mem::new(Reg(0), 0));
            self.asm.add_imm(true, false, Size::X, A, A, 1);
            self.asm.stp_offset(Reg(17), A, C, 0);
            self.asm.bind(back);
            self.cold_calls
                .push(ColdCall::FrameAlloc(cold, back, bytes));
        } else {
            self.raw(A, args[0]);
            if self.emit_cached_tlv_base() {
                let at = self.asm.pos();
                self.asm.word(off_word);
                self.relocs
                    .push(StencilReloc::runtime(at, rt, RelocKind::ThreadOff32));
            } else {
                let at = self.emit_tlv_base();
                self.asm.word(off_word);
                self.relocs
                    .push(StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell));
            }
            self.asm.ldp_offset(Reg(17), Reg(0), C, 0);
            self.asm.add_imm(
                true,
                true,
                Size::X,
                ZR,
                Reg(0),
                u32::from(FRAME_CACHE_DEPTH),
            );
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
                self.asm
                    .stp_offset(saved[i], saved[i + 1], SP, (i * 8) as i32);
                i += 2;
            }
            if i < saved.len() {
                self.asm
                    .store(Size::X, Mem::new(SP, (i * 8) as i32), saved[i]);
            }
        }
        body(self);
        if alloc > 0 {
            let mut i = 0;
            while i + 1 < saved.len() {
                self.asm
                    .ldp_offset(saved[i], saved[i + 1], SP, (i * 8) as i32);
                i += 2;
            }
            if i < saved.len() {
                self.asm
                    .load(Size::X, false, saved[i], Mem::new(SP, (i * 8) as i32));
            }
            self.asm.add_imm(false, false, Size::X, SP, SP, alloc);
        }
        self.asm.b(back);
    }

    pub(super) fn emit_cold_calls(&mut self) {
        self.invalidate_spills();
        if let Some((helper, init_sym, slot)) = self
            .tlv_cache
            .as_ref()
            .filter(|c| c.used)
            .map(|c| (c.helper, c.init_sym, c.slot))
        {
            self.asm.bind(helper);
            self.asm.stp_pre(LR, ZR, SP, -16);
            let at = self.emit_tlv_base();
            self.asm.word(ARM64_NOP);
            let reloc = match init_sym {
                TlvInitSym::Symbol(sym) => StencilReloc::symbol(at, sym, RelocKind::Arm64TlvCell),
                TlvInitSym::Runtime(rt) => StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell),
            };
            self.relocs.push(reloc);
            self.asm.ldp_post(LR, ZR, SP, 16);
            let slot_mem = self.frame_mem(slot);
            self.asm.store(Size::X, slot_mem, Reg(0));
            self.asm.ret();
        }
        let mut saved_leaf: Option<Vec<Reg>> = None;
        for call in std::mem::take(&mut self.cold_calls) {
            let saved = saved_leaf
                .get_or_insert_with(|| {
                    crate::alloc::used_preserved_leaf_regs(
                        &super::REGS,
                        &self.plan.locations,
                        false,
                    )
                })
                .clone();
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
