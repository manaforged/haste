use super::asm::{Asm, Cond, LR, Label, Mem, Reg, SP, Size, ZR};
use super::{A, C, Emit, IP, T};
use crate::alloc::Plan;
use crate::frame::Frame;
use crate::ir::{Function, Inst, Op, Value};
use crate::reloc::{FRAME_CACHE_DEPTH, RelocKind, RuntimeSymbol, StencilReloc};

pub(super) const ARM64_NOP: u32 = 0xD503_201F;

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
    pub(super) may_known: Vec<bool>,
    pub(super) needs_zero: bool,
    pub(super) live: bool,
    pub(super) may_live: bool,
    used: bool,
}

pub(super) enum ColdCall {
    ThreadCell(Label, Label, u32, [Value; 3]),
    FrameAlloc(Label, Label, u32),
    FrameFree(Label, Label, u32),
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
    ) && let Some(sym) = crate::alloc::thread_cell_target(func, plan, defs, slot, lookup)
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
    let mut may_known = vec![false; func.blocks.len()];
    let mut work: Vec<usize> = Vec::new();
    let mut push_succs = |b: usize, work: &mut Vec<usize>| {
        for succ in func.successors(crate::ir::Block(b as u32)) {
            let si = succ.index();
            if !may_known[si] {
                may_known[si] = true;
                work.push(si);
            }
        }
    };
    for b in 0..func.blocks.len() {
        if has_tlv[b] && cfg.reachable[b] {
            push_succs(b, &mut work);
        }
    }
    while let Some(u) = work.pop() {
        push_succs(u, &mut work);
    }
    let needs_zero = (0..func.blocks.len())
        .any(|b| has_tlv[b] && cfg.reachable[b] && !dom_known[b] && may_known[b]);
    let slot = frame.add_slot(8);
    let helper = asm.label();
    Some(TlvCache {
        slot,
        helper,
        init_sym: first_sym?,
        dom_known,
        may_known,
        needs_zero,
        live: false,
        may_live: false,
        used: false,
    })
}

impl Emit<'_> {
    fn thread_cell_sym(&self, slot: Value, lookup: u32) -> Option<u32> {
        self.tlv_cells
            .then(|| {
                crate::alloc::thread_cell_target(self.func, &self.plan, &self.defs, slot, lookup)
            })
            .flatten()
    }

    fn emit_tlv_base(&mut self) -> u32 {
        let at = self.asm.pos();
        self.asm.word(0x9000_0000);
        self.asm.word(0x9100_0000);
        self.asm.word(0xF940_0010);
        self.asm.blr(IP);
        at
    }

    fn emit_tlv_init_base(&mut self, init_sym: TlvInitSym) {
        let at = self.emit_tlv_base();
        self.asm.word(ARM64_NOP);
        let reloc = match init_sym {
            TlvInitSym::Symbol(sym) => StencilReloc::symbol(at, sym, RelocKind::Arm64TlvCell),
            TlvInitSym::Runtime(rt) => StencilReloc::runtime(at, rt, RelocKind::Arm64TlvCell),
        };
        self.relocs.push(reloc);
    }

    fn emit_cached_tlv_base(&mut self) -> bool {
        let Some((slot, helper, init_sym, known, may_live)) = self
            .tlv_cache
            .as_ref()
            .map(|c| (c.slot, c.helper, c.init_sym, c.live, c.may_live))
        else {
            return false;
        };
        let at = self.frame_mem(slot);
        if known {
            self.asm.load(Size::X, false, Reg(0), at);
        } else if !may_live {
            self.emit_tlv_init_base(init_sym);
            self.asm.store(Size::X, at, Reg(0));
            if let Some(cache) = &mut self.tlv_cache {
                cache.live = true;
                cache.may_live = true;
            }
        } else {
            self.asm.load(Size::X, false, Reg(0), at);
            let skip = self.asm.label();
            self.asm.cbz(true, Size::X, Reg(0), skip);
            self.asm.bl(helper);
            self.asm.bind(skip);
            if let Some(cache) = &mut self.tlv_cache {
                cache.live = true;
                cache.may_live = true;
                cache.used = true;
            }
        }
        true
    }

    fn emit_tlv_slot(&mut self, word: u32, build: impl FnOnce(u32, RelocKind) -> StencilReloc) {
        let (at, kind) = if self.emit_cached_tlv_base() {
            (self.asm.pos(), RelocKind::ThreadOff32)
        } else {
            (self.emit_tlv_base(), RelocKind::Arm64TlvCell)
        };
        self.asm.word(word);
        self.relocs.push(build(at, kind));
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
        self.emit_tlv_slot(0xF940_0000, |at, kind| {
            StencilReloc::symbol(at, target_sym, kind)
        });
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
            self.emit_tlv_slot(off_word, |at, kind| StencilReloc::runtime(at, rt, kind));
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
            self.emit_tlv_slot(off_word, |at, kind| StencilReloc::runtime(at, rt, kind));
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
            self.emit_tlv_init_base(init_sym);
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

pub(super) fn madd_shift_const(
    plan: &Plan<Reg>,
    ty: crate::ir::Type,
    a: Value,
    b: Value,
) -> Option<(Value, u8, u8)> {
    let bits = ty.bits() as u8;
    for (val, k) in [(a, b), (b, a)] {
        let Some(raw) = plan.const_of(k) else {
            continue;
        };
        let uimm = crate::fold::extend(raw, ty, false) as u64;
        if uimm == 0 {
            continue;
        }
        let shift = uimm.trailing_zeros() as u8;
        let odd = uimm >> shift;
        if shift > 0 && shift < bits && matches!(odd, 3 | 5 | 9) {
            return Some((val, (odd - 1).trailing_zeros() as u8, shift));
        }
    }
    None
}

fn empty_fallthrough(
    func: &Function,
    plan: &Plan<Reg>,
    frame: &Frame<Reg>,
    copied: &[bool],
    shrink: Option<&crate::frame::ShrinkWrap>,
    cur: crate::ir::Block,
    next: crate::ir::Block,
) -> bool {
    let insts = &func.blocks[cur.index()].insts;
    let Some((&term, body)) = insts.split_last() else {
        return false;
    };
    if copied[term.index()] {
        return false;
    }
    let Op::Jump(target, args) = &func.insts[term.index()].op else {
        return false;
    };
    if func.jump_target(*target) != next || plan.moves(func, frame, *target, args) {
        return false;
    }
    if shrink.is_some_and(|s| {
        s.fast[cur.index()] && (!s.fast[target.index()] || s.split_ret[next.index()])
    }) {
        return false;
    }
    body.iter().all(|&inst| {
        let d = &func.insts[inst.index()];
        !matches!(
            d.op,
            Op::Call { .. }
                | Op::TryCall { .. }
                | Op::Store(..)
                | Op::StackStore(..)
                | Op::Trap
                | Op::Trapz(..)
                | Op::AtomicRmw(..)
                | Op::AtomicCas(..)
                | Op::Fence(_)
        ) && d.result.is_some_and(|r| {
            plan.elided[r.index()] || plan.deferred[r.index()] || plan.is_static(r)
        })
    })
}

pub(super) fn shift_align_loops(
    func: &Function,
    plan: &Plan<Reg>,
    frame: &Frame<Reg>,
    order: &[crate::ir::Block],
    copied: &[bool],
    shrink: Option<&crate::frame::ShrinkWrap>,
    align_loops: &mut [bool],
) {
    for i in (0..order.len().saturating_sub(1)).rev() {
        let (cur, next) = (order[i], order[i + 1]);
        if align_loops[next.index()]
            && empty_fallthrough(func, plan, frame, copied, shrink, cur, next)
        {
            align_loops[next.index()] = false;
            align_loops[cur.index()] = true;
        }
    }
}

pub(super) fn loc_mask(loc: crate::alloc::Location<Reg>) -> u64 {
    match loc {
        crate::alloc::Location::Reg(r) | crate::alloc::Location::Offset(r, _) => 1_u64 << r.0,
        crate::alloc::Location::Float(f) => 1_u64 << (32 + f),
        _ => 0,
    }
}

pub(super) fn compute_live_in_regs(
    func: &Function,
    plan: &Plan<Reg>,
    order: &[crate::ir::Block],
) -> Vec<u64> {
    let blocks = func.blocks.len();
    let vals = func.types.len();
    let mut gen_sets = crate::live::BitMatrix::new(blocks, vals);
    let mut kill_sets = crate::live::BitMatrix::new(blocks, vals);
    for (b, block) in func.blocks.iter().enumerate() {
        for &p in &block.params {
            kill_sets.set(b, p.index());
        }
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            data.op.for_operands(|v| {
                if !kill_sets.get(b, v.index()) {
                    gen_sets.set(b, v.index());
                }
            });
            for res in crate::live::results(data) {
                kill_sets.set(b, res.index());
            }
        }
    }
    let live =
        crate::live::solve_backward_liveness(func, order.iter().copied(), &gen_sets, &kill_sets);
    (0..blocks)
        .map(|b| {
            let mut mask = 0_u64;
            for vi in live.members(b) {
                mask |= loc_mask(plan.locations[vi]);
            }
            for &p in &func.blocks[b].params {
                mask |= loc_mask(plan.locations[p.index()]);
            }
            mask
        })
        .collect()
}

pub(super) struct LoopConsts {
    pub(super) preheader_inits: Vec<Vec<(i64, Reg)>>,
    pub(super) block_consts: Vec<Vec<(i64, Reg)>>,
}

impl LoopConsts {
    pub(super) fn empty() -> Self {
        Self {
            preheader_inits: Vec::new(),
            block_consts: Vec::new(),
        }
    }
}

fn arm64_expensive_imm(imm: i64) -> bool {
    let u = imm as u64;
    let non_zero = (0..4).filter(|&s| ((u >> (s * 16)) & 0xFFFF) != 0).count();
    let non_ones = (0..4)
        .filter(|&s| ((u >> (s * 16)) & 0xFFFF) != 0xFFFF)
        .count();
    non_zero >= 2 && non_ones >= 2 && super::int::encode_bitmask_imm(u, true).is_none()
}

fn loop_preheader(
    func: &Function,
    cfg: &crate::cfg::Cfg,
    each: &crate::cfg::Loop,
    copied: &[bool],
    shrink: Option<&crate::frame::ShrinkWrap>,
) -> Option<crate::ir::Block> {
    if func.entry == Some(each.head) {
        return None;
    }
    let mut pre = None;
    for &p in cfg.preds_of(each.head) {
        if each.body[p.index()] {
            continue;
        }
        if pre.replace(p).is_some() {
            return None;
        }
    }
    let pre = pre?;
    if shrink.is_some_and(|s| s.fast[pre.index()]) {
        return None;
    }
    let &term = func.blocks[pre.index()].insts.last()?;
    if copied[term.index()] {
        return None;
    }
    let Op::Jump(target, _) = func.insts[term.index()].op else {
        return None;
    };
    (func.jump_target(target) == each.head).then_some(pre)
}

fn scan_loop_for_consts(
    func: &Function,
    plan: &Plan<Reg>,
    each: &crate::cfg::Loop,
    live_in_regs: &[u64],
) -> Option<(u32, Vec<i64>)> {
    let mut used_gp = 0_u32;
    let mut cands = Vec::new();
    for (b, &in_loop) in each.body.iter().enumerate() {
        if !in_loop {
            continue;
        }
        used_gp |= live_in_regs[b] as u32;
        for succ in func.successors(crate::ir::Block(b as u32)) {
            used_gp |= live_in_regs[succ.index()] as u32;
        }
        for &inst in &func.blocks[b].insts {
            let data = &func.insts[inst.index()];
            if matches!(
                data.op,
                Op::Call { .. }
                    | Op::TryCall { .. }
                    | Op::AtomicRmw(..)
                    | Op::AtomicCas(..)
                    | Op::Fence(_)
            ) {
                return None;
            }
            for res in crate::live::results(data) {
                used_gp |= loc_mask(plan.locations[res.index()]) as u32;
            }
            let skipped = data
                .result
                .is_some_and(|r| plan.elided[r.index()] || plan.deferred[r.index()]);
            data.op.for_operands(|v| {
                used_gp |= loc_mask(plan.locations[v.index()]) as u32;
                if !skipped
                    && let crate::alloc::Location::Imm(imm) = plan.locations[v.index()]
                    && arm64_expensive_imm(imm)
                    && !cands.contains(&imm)
                    && cands.len() < 2
                {
                    cands.push(imm);
                }
            });
        }
    }
    (!cands.is_empty()).then_some((used_gp, cands))
}

pub(super) fn plan_loop_consts(
    func: &Function,
    plan: &Plan<Reg>,
    frame: &Frame<Reg>,
    cfg: &crate::cfg::Cfg,
    loops: &crate::cfg::Loops,
    cold: &[bool],
    copied: &[bool],
    shrink: Option<&crate::frame::ShrinkWrap>,
    live_in_regs: &[u64],
) -> LoopConsts {
    let blocks = func.blocks.len();
    let mut out = LoopConsts {
        preheader_inits: vec![Vec::new(); blocks],
        block_consts: vec![Vec::new(); blocks],
    };
    let mut order_idx: Vec<usize> = (0..loops.list.len()).collect();
    order_idx.sort_by_key(|&i| loops.list[i].depth);
    for idx in order_idx {
        let each = &loops.list[idx];
        if cold[each.head.index()] {
            continue;
        }
        let Some(pre) = loop_preheader(func, cfg, each, copied, shrink) else {
            continue;
        };
        let Some((mut used_gp, cands)) = scan_loop_for_consts(func, plan, each, live_in_regs)
        else {
            continue;
        };
        for &(_, r) in &out.preheader_inits[pre.index()] {
            used_gp |= 1_u32 << r.0;
        }
        let mut already = Vec::new();
        for (b, &in_loop) in each.body.iter().enumerate() {
            if in_loop {
                for &(k, r) in &out.block_consts[b] {
                    used_gp |= 1_u32 << r.0;
                    if !already.contains(&k) {
                        already.push(k);
                    }
                }
            }
        }
        for imm in cands {
            if already.contains(&imm) {
                continue;
            }
            let free = frame
                .saves
                .iter()
                .copied()
                .chain((0..=8).map(Reg))
                .find(|r| (used_gp & (1_u32 << r.0)) == 0);
            let Some(reg) = free else {
                break;
            };
            used_gp |= 1_u32 << reg.0;
            out.preheader_inits[pre.index()].push((imm, reg));
            for (b, &in_loop) in each.body.iter().enumerate() {
                if in_loop {
                    out.block_consts[b].push((imm, reg));
                }
            }
        }
    }
    out
}
