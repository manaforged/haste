mod narrow;
mod sink;

use crate::ir::{Binary, Block, Function, Inst, Op, StackSlot, Value};
use narrow::{NarrowAnalysis, narrow_unwind_drops};
use sink::sink_entry_zeros;
pub(crate) use sink::sink_frontier;

/// One stack-allocated owned value dropped during panic unwinding when its 1-byte active flag is non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnwindDrop {
    pub local: StackSlot,
    pub local_off: i32,
    pub flag: StackSlot,
    pub drop_sym: u32,
    pub indirect: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CalleeKind {
    NoUnwind,
    MayUnwind,
}

fn def_op<'a>(func: &'a Function, defs: &[Option<Inst>], val: Value) -> Option<&'a Op> {
    defs.get(val.index())
        .copied()
        .flatten()
        .map(|i| &func.insts[i.index()].op)
}

fn collect_defs_and_addrs(
    func: &Function,
    defs: &mut [Option<Inst>],
    addr_slot: &mut [Option<(StackSlot, i32)>],
) {
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            if let Some(res) = data.result {
                defs[res.index()] = Some(inst);
                if let Op::StackAddr(slot, off) = data.op {
                    addr_slot[res.index()] = Some((slot, off));
                }
            }
        }
    }
}

pub(super) fn classify_callee(
    func: &Function,
    defs: &[Option<Inst>],
    callee: Value,
    args: &[Value],
    sym_may_unwind: &mut impl FnMut(u32) -> bool,
) -> CalleeKind {
    if func.is_thread_cell(callee) {
        return match args.get(2).and_then(|&init| def_op(func, defs, init)) {
            Some(&Op::Const(0)) => CalleeKind::NoUnwind,
            Some(&(Op::Symbol(sym) | Op::SlotAddr(sym))) if !sym_may_unwind(sym) => {
                CalleeKind::NoUnwind
            }
            _ => CalleeKind::MayUnwind,
        };
    }
    let sym = match def_op(func, defs, callee) {
        Some(&(Op::Symbol(sym) | Op::SlotAddr(sym))) => Some(sym),
        Some(&Op::Load(addr, 0)) => match def_op(func, defs, addr) {
            Some(&Op::SlotAddr(sym)) => Some(sym),
            _ => None,
        },
        Some(Op::Runtime(_)) => return CalleeKind::NoUnwind,
        _ => None,
    };
    match sym {
        Some(sym) if !sym_may_unwind(sym) => CalleeKind::NoUnwind,
        _ => CalleeKind::MayUnwind,
    }
}

pub(super) fn store_clears_active(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    match def_op(func, defs, val) {
        Some(&Op::Const(imm)) if (imm & 1) == 0 => true,
        Some(&Op::Binary(Binary::Band, a, b)) => {
            let clears =
                |v: Value| matches!(def_op(func, defs, v), Some(&Op::Const(imm)) if (imm & 1) == 0);
            clears(a) || clears(b)
        }
        _ => false,
    }
}

pub(super) enum FlagOp {
    MayUnwindCall,
    TryCall,
    Load,
    Clear,
    Set,
    Exit,
}

pub(super) fn classify_flag_op(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    flag: StackSlot,
    inst: Inst,
) -> Option<FlagOp> {
    match func.insts[inst.index()].op {
        Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => {
            Some(FlagOp::MayUnwindCall)
        }
        Op::TryCall { .. } => Some(FlagOp::TryCall),
        Op::StackLoad(s, _) if s == flag => Some(FlagOp::Load),
        Op::StackStore(v, s, 0) if s == flag => Some(if store_clears_active(func, defs, v) {
            FlagOp::Clear
        } else {
            FlagOp::Set
        }),
        Op::Return(..) | Op::Trap => Some(FlagOp::Exit),
        _ => None,
    }
}

impl Function {
    /// Alignment shift of stack slot `slot`.
    pub fn slot_align_shift(&self, slot: StackSlot) -> u8 {
        self.slots[slot.index()].align_shift
    }

    /// Registers a frame-table unwind drop entry on this function.
    pub fn add_unwind_drop(&mut self, drop: UnwindDrop) {
        self.unwind_drops.push(drop);
    }

    /// Frame-table unwind drop entries registered on this function.
    pub fn unwind_drops(&self) -> &[UnwindDrop] {
        &self.unwind_drops
    }

    pub(crate) fn is_unwind_slot(&self, slot: StackSlot) -> bool {
        self.unwind_drops
            .iter()
            .any(|u| u.local == slot || u.flag == slot)
    }

    pub(crate) fn import_unwind_drops(&mut self, callee: &Function, slot_base: u32, sym_base: u32) {
        self.keep_fp |= callee.keep_fp;
        self.elide_drops |= callee.elide_drops;
        for &sym in &callee.no_unwind_syms {
            self.no_unwind_syms.push(sym + sym_base);
        }
        for u in &callee.unwind_drops {
            self.unwind_drops.push(UnwindDrop {
                local: StackSlot(u.local.0 + slot_base),
                local_off: u.local_off,
                flag: StackSlot(u.flag.0 + slot_base),
                drop_sym: u.drop_sym + sym_base,
                indirect: u.indirect,
            });
        }
    }

    /// Zero-initializes every unwind-drop flag slot at the start of the entry block so early panics never read uninitialized stack bytes.
    pub fn zero_init_unwind_flags(&mut self) {
        let Some(entry) = self.entry else { return };
        if self.unwind_drops.is_empty() {
            return;
        }
        let mut flags: Vec<StackSlot> = self.unwind_drops.iter().map(|u| u.flag).collect();
        flags.sort_unstable_by_key(|s| s.0);
        flags.dedup();
        let zero = self.value(crate::Type::I8);
        let zinst = self.add(Op::Const(0), Some(zero));
        let mut init = Vec::with_capacity(1 + flags.len());
        init.push(zinst);
        for flag in flags {
            init.push(self.add(Op::StackStore(zero, flag, 0), None));
        }
        let rest = std::mem::take(&mut self.blocks[entry.index()].insts);
        init.extend(rest);
        self.blocks[entry.index()].insts = init;
    }

    /// Removes frame-table unwind drop entries whose flag can never be active across a potentially-unwinding call.
    pub fn elide_dead_unwind_drops(&mut self, sym_may_unwind: impl FnMut(u32) -> bool) -> bool {
        self.elide_dead_unwind_drops_inner(false, false, sym_may_unwind)
    }

    /// Post-optimization unwind-drop cleanup that preserves or re-emits entry zero-initialization for every surviving flag, optionally sinking entry zeros onto cold unwind-call frontiers.
    pub(crate) fn elide_dead_unwind_drops_post_init(&mut self, sink_zeros: bool) -> bool {
        if self.no_unwind_syms.is_empty() {
            return self.elide_dead_unwind_drops_inner(true, sink_zeros, |_| true);
        }
        let mut no_unwind = std::mem::take(&mut self.no_unwind_syms);
        no_unwind.sort_unstable();
        let changed = self.elide_dead_unwind_drops_inner(true, sink_zeros, |sym| {
            no_unwind.binary_search(&sym).is_err()
        });
        self.no_unwind_syms = no_unwind;
        changed
    }

    fn elide_dead_unwind_drops_inner(
        &mut self,
        post_init: bool,
        sink_zeros: bool,
        mut sym_may_unwind: impl FnMut(u32) -> bool,
    ) -> bool {
        let Some(entry) = self.entry else {
            return false;
        };
        if self.coroutine.is_some() || self.unwind_drops.is_empty() {
            return false;
        }
        let mut defs = vec![None; self.types.len()];
        let mut addr_slot = vec![None; self.types.len()];
        collect_defs_and_addrs(self, &mut defs, &mut addr_slot);
        let mut kinds = vec![CalleeKind::NoUnwind; self.insts.len()];
        let mut slot_escaped = vec![false; self.slots.len()];
        let mut slot_loaded = vec![false; self.slots.len()];
        let mut local_write_only = vec![true; self.slots.len()];
        let mut any_unwind_call = false;
        for block in &self.blocks {
            for &inst in &block.insts {
                let op = &self.insts[inst.index()].op;
                match op {
                    Op::Call { callee, args, .. } => {
                        let kind = classify_callee(self, &defs, *callee, args, &mut sym_may_unwind);
                        kinds[inst.index()] = kind;
                        any_unwind_call |= kind == CalleeKind::MayUnwind;
                    }
                    Op::TryCall { .. } => any_unwind_call = true,
                    Op::StackLoad(slot, _) if slot.index() < slot_loaded.len() => {
                        slot_loaded[slot.index()] = true;
                        local_write_only[slot.index()] = false;
                    }
                    _ => {}
                }
                op.for_operands(|v| {
                    if let Some((s, base)) = addr_slot[v.index()]
                        && s.index() < slot_escaped.len()
                    {
                        slot_escaped[s.index()] = true;
                        let pure_store = matches!(
                            *op,
                            Op::Store(val, addr, off)
                                if addr == v
                                    && val != v
                                    && base.checked_add(off).is_some_and(|total| crate::mem::in_range(
                                        i64::from(total),
                                        self.ty(val).bytes(),
                                        self.slots[s.index()].size,
                                    ))
                        );
                        if !pure_store {
                            local_write_only[s.index()] = false;
                        }
                    }
                });
            }
        }
        if !any_unwind_call {
            self.unwind_drops.clear();
            return true;
        }
        let before = self.unwind_drops.len();
        let drops = std::mem::take(&mut self.unwind_drops);
        self.unwind_drops = drops
            .into_iter()
            .filter(|u| {
                u.flag.index() >= slot_escaped.len()
                    || slot_escaped[u.flag.index()]
                    || unwind_flag_live_at_call(self, &defs, &kinds, entry, u.flag)
            })
            .collect();
        let folded = fold_known_flag_loads(
            self,
            entry,
            (&defs, &kinds, &slot_escaped, &mut slot_loaded),
        );
        let (mut narrowed, sunk_flag) = narrow_unwind_drops(
            self,
            entry,
            post_init,
            &NarrowAnalysis {
                defs: &defs,
                kinds: &kinds,
                addr_slot: &addr_slot,
                slot_escaped: &slot_escaped,
                slot_loaded: &slot_loaded,
                local_write_only: &local_write_only,
            },
        );
        if sunk_flag {
            defs.resize(self.types.len(), None);
            addr_slot.resize(self.types.len(), None);
            kinds.resize(self.insts.len(), CalleeKind::NoUnwind);
            defs.fill(None);
            addr_slot.fill(None);
            collect_defs_and_addrs(self, &mut defs, &mut addr_slot);
            for block in &self.blocks {
                for &inst in &block.insts {
                    if let Op::Call {
                        callee, ref args, ..
                    } = self.insts[inst.index()].op
                    {
                        kinds[inst.index()] =
                            classify_callee(self, &defs, callee, args, &mut sym_may_unwind);
                    }
                }
            }
            narrowed |= fold_known_flag_loads(
                self,
                entry,
                (&defs, &kinds, &slot_escaped, &mut slot_loaded),
            );
            narrowed |= narrow_unwind_drops(
                self,
                entry,
                post_init,
                &NarrowAnalysis {
                    defs: &defs,
                    kinds: &kinds,
                    addr_slot: &addr_slot,
                    slot_escaped: &slot_escaped,
                    slot_loaded: &slot_loaded,
                    local_write_only: &local_write_only,
                },
            )
            .0;
        }
        let sunk_zeros =
            sink_zeros && sink_entry_zeros(self, entry, &slot_escaped, &mut sym_may_unwind);
        self.unwind_drops.len() != before || folded || narrowed || sunk_zeros
    }
}

fn fold_known_flag_loads(
    func: &mut Function,
    entry: Block,
    (defs, kinds, slot_escaped, slot_loaded): (
        &[Option<Inst>],
        &[CalleeKind],
        &[bool],
        &mut [bool],
    ),
) -> bool {
    let nslots = func.slots.len();
    let mut is_flag = vec![false; nslots];
    let mut flags = Vec::new();
    for u in &func.unwind_drops {
        let fi = u.flag.index();
        if fi < nslots && !is_flag[fi] && !slot_escaped[fi] && slot_loaded[fi] {
            is_flag[fi] = true;
            flags.push(u.flag);
        }
    }
    if flags.is_empty() {
        return false;
    }
    let mut exact = vec![true; nslots];
    let mut loads: Vec<Vec<Inst>> = vec![Vec::new(); nslots];
    for block in &func.blocks {
        for &inst in &block.insts {
            let data = &func.insts[inst.index()];
            match data.op {
                Op::StackStore(v, slot, off) if slot.index() < nslots && is_flag[slot.index()] => {
                    if off != 0
                        || func.ty(v) != crate::Type::I8
                        || !matches!(def_op(func, defs, v), Some(&Op::Const(0 | 1)))
                    {
                        exact[slot.index()] = false;
                    }
                }
                Op::StackLoad(slot, off) if slot.index() < nslots && is_flag[slot.index()] => {
                    if off != 0 || !data.result.is_some_and(|r| func.ty(r) == crate::Type::I8) {
                        exact[slot.index()] = false;
                    } else {
                        loads[slot.index()].push(inst);
                    }
                }
                _ => {}
            }
        }
    }
    let mut changed = false;
    let mut load_state = vec![0u8; func.insts.len()];
    for flag in flags {
        let fi = flag.index();
        if !exact[fi] || loads[fi].is_empty() {
            continue;
        }
        for &inst in &loads[fi] {
            load_state[inst.index()] = 0;
        }
        walk_flag_cfg(func, defs, kinds, entry, flag, |ev| {
            if let FlagEvent::Load(inst, active) = ev {
                load_state[inst.index()] |= if active { 2 } else { 1 };
            }
            false
        });
        let mut remaining = false;
        for &inst in &loads[fi] {
            match load_state[inst.index()] {
                1 => {
                    func.insts[inst.index()].op = Op::Const(0);
                    changed = true;
                }
                2 => {
                    func.insts[inst.index()].op = Op::Const(1);
                    changed = true;
                }
                _ => remaining = true,
            }
        }
        if !remaining {
            slot_loaded[fi] = false;
        }
    }
    changed
}

enum FlagEvent {
    Call(Inst, bool),
    TryCall(bool),
    Clear(Inst, bool),
    Set(Inst, bool),
    Load(Inst, bool),
}

fn walk_flag_cfg(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    entry: Block,
    flag: StackSlot,
    mut on_event: impl FnMut(FlagEvent) -> bool,
) -> (bool, Vec<u8>, Vec<u8>) {
    let mut visited = vec![0u8; func.blocks.len()];
    let mut out_visited = vec![0u8; func.blocks.len()];
    visited[entry.index()] = 1;
    let mut work = vec![(entry, false)];
    while let Some((block, mut active)) = work.pop() {
        let mut stopped = false;
        for &inst in &func.blocks[block.index()].insts {
            match classify_flag_op(func, defs, kinds, flag, inst) {
                Some(FlagOp::MayUnwindCall) => {
                    if on_event(FlagEvent::Call(inst, active)) {
                        return (true, visited, out_visited);
                    }
                }
                Some(FlagOp::TryCall) => {
                    if on_event(FlagEvent::TryCall(active)) {
                        return (true, visited, out_visited);
                    }
                }
                Some(FlagOp::Load) => {
                    if on_event(FlagEvent::Load(inst, active)) {
                        return (true, visited, out_visited);
                    }
                }
                Some(FlagOp::Clear) => {
                    if on_event(FlagEvent::Clear(inst, active)) {
                        return (true, visited, out_visited);
                    }
                    active = false;
                }
                Some(FlagOp::Set) => {
                    if on_event(FlagEvent::Set(inst, active)) {
                        return (true, visited, out_visited);
                    }
                    active = true;
                }
                Some(FlagOp::Exit) => {
                    stopped = true;
                    break;
                }
                None => {}
            }
        }
        if stopped {
            continue;
        }
        let bit = if active { 2 } else { 1 };
        out_visited[block.index()] |= bit;
        if let Some(last) = func.last(block) {
            if let Op::Brif(cond, then, _, otherwise, _) = *last
                && let Some(&Op::Const(imm)) = def_op(func, defs, cond)
            {
                let target = if crate::fold::extend(imm, func.ty(cond), false) != 0 {
                    then
                } else {
                    otherwise
                };
                if visited[target.index()] & bit == 0 {
                    visited[target.index()] |= bit;
                    work.push((target, active));
                }
                continue;
            }
            for target in last.targets() {
                if visited[target.index()] & bit == 0 {
                    visited[target.index()] |= bit;
                    work.push((target, active));
                }
            }
        }
    }
    (false, visited, out_visited)
}

fn analyze_flag(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    entry: Block,
    flag: StackSlot,
) -> (
    Vec<Inst>,
    Vec<Inst>,
    Option<Vec<(Block, Inst)>>,
    Vec<u8>,
    Vec<u8>,
) {
    let mut call_state = vec![0u8; func.insts.len()];
    let mut clear_state = vec![0u8; func.insts.len()];
    let mut set_state = vec![0u8; func.insts.len()];
    let mut try_active = false;
    let bit = |a: bool| if a { 2u8 } else { 1u8 };
    let (_, visited, out_visited) = walk_flag_cfg(func, defs, kinds, entry, flag, |ev| {
        match ev {
            FlagEvent::Call(inst, active) => call_state[inst.index()] |= bit(active),
            FlagEvent::TryCall(active) => try_active |= active,
            FlagEvent::Clear(inst, active) => clear_state[inst.index()] |= bit(active),
            FlagEvent::Set(inst, active) => set_state[inst.index()] |= bit(active),
            FlagEvent::Load(..) => {}
        }
        false
    });
    let mut dead_zeros = Vec::new();
    let mut fresh_sets = Vec::new();
    let mut live = Vec::new();
    let mut exact = !try_active;
    for (bi, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            if let Op::StackStore(v, s, 0) = func.insts[inst.index()].op
                && s == flag
            {
                if store_clears_active(func, defs, v) {
                    if clear_state[inst.index()] <= 1 {
                        dead_zeros.push(inst);
                    }
                } else if set_state[inst.index()] <= 1 {
                    fresh_sets.push(inst);
                }
            }
            match call_state[inst.index()] {
                3 => exact = false,
                2 => live.push((Block(bi as u32), inst)),
                _ => {}
            }
        }
    }
    (
        dead_zeros,
        fresh_sets,
        exact.then_some(live),
        visited,
        out_visited,
    )
}

fn unwind_flag_live_at_call(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    entry: Block,
    flag: StackSlot,
) -> bool {
    walk_flag_cfg(func, defs, kinds, entry, flag, |ev| {
        matches!(ev, FlagEvent::Call(_, true) | FlagEvent::TryCall(true))
    })
    .0
}
