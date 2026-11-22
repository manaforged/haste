use crate::ir::{Binary, Block, Function, Inst, Op, StackSlot, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CalleeKind {
    DropPush,
    DropPop,
    NoUnwind,
    MayUnwind,
}

fn classify_callee(
    func: &Function,
    defs: &[Option<Inst>],
    callee: Value,
    is_drop_push: &mut impl FnMut(u32) -> bool,
    is_drop_pop: &mut impl FnMut(u32) -> bool,
    sym_may_unwind: &mut impl FnMut(u32) -> bool,
) -> CalleeKind {
    match defs.get(callee.index()).copied().flatten().map(|i| &func.insts[i.index()].op) {
        Some(&(Op::Symbol(sym) | Op::SlotAddr(sym))) => {
            if is_drop_push(sym) {
                CalleeKind::DropPush
            } else if is_drop_pop(sym) {
                CalleeKind::DropPop
            } else if sym_may_unwind(sym) {
                CalleeKind::MayUnwind
            } else {
                CalleeKind::NoUnwind
            }
        }
        Some(Op::Runtime(_)) => CalleeKind::NoUnwind,
        _ => CalleeKind::MayUnwind,
    }
}

fn store_clears_active(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    let Some(inst) = defs.get(val.index()).copied().flatten() else { return false };
    match func.insts[inst.index()].op {
        Op::Const(imm) if (imm & 1) == 0 => true,
        Op::Binary(Binary::Band, a, b) => {
            let clears = |v: Value| {
                defs.get(v.index())
                    .copied()
                    .flatten()
                    .is_some_and(|i| matches!(func.insts[i.index()].op, Op::Const(imm) if (imm & 1) == 0))
            };
            clears(a) || clears(b)
        }
        _ => false,
    }
}

fn init_active_before(func: &Function, defs: &[Option<Inst>], block: Block, push_pos: usize, slot: StackSlot) -> bool {
    for &inst in func.blocks[block.index()].insts[..push_pos].iter().rev() {
        match func.insts[inst.index()].op {
            Op::StackStore(v, s, 0) if s == slot => return !store_clears_active(func, defs, v),
            Op::Call { .. } | Op::TryCall { .. } => break,
            _ => {}
        }
    }
    true
}

fn slot_needs_unwind(
    func: &Function,
    defs: &[Option<Inst>],
    addr_slot: &[Option<StackSlot>],
    kinds: &[CalleeKind],
    slot: StackSlot,
    pushes: &[(Block, usize)],
) -> bool {
    let mut visited = vec![0u8; func.blocks.len()];
    let mut work: Vec<(Block, usize, bool)> = pushes
        .iter()
        .map(|&(b, pos)| (b, pos + 1, init_active_before(func, defs, b, pos, slot)))
        .collect();
    while let Some((block, start_pos, mut active)) = work.pop() {
        let insts = &func.blocks[block.index()].insts;
        let mut stopped = false;
        for &inst in &insts[start_pos..] {
            match &func.insts[inst.index()].op {
                Op::Call { args, .. } => match kinds[inst.index()] {
                    CalleeKind::DropPop if args.len() == 1 && addr_slot[args[0].index()] == Some(slot) => {
                        stopped = true;
                        break;
                    }
                    CalleeKind::MayUnwind if active => return true,
                    _ => {}
                },
                Op::TryCall { .. } if active => return true,
                &Op::StackStore(v, s, 0) if s == slot => {
                    active = !store_clears_active(func, defs, v);
                }
                Op::Return(..) => {
                    if active {
                        return true;
                    }
                    stopped = true;
                    break;
                }
                Op::Trap => {
                    stopped = true;
                    break;
                }
                _ => {}
            }
        }
        if stopped {
            continue;
        }
        let bit = if active { 2 } else { 1 };
        if let Some(last) = func.last(block) {
            for target in last.targets() {
                if visited[target.index()] & bit == 0 {
                    visited[target.index()] |= bit;
                    work.push((target, 0, active));
                }
            }
        }
    }
    false
}

impl Function {
    /// Removes `DropPush`/`DropPop` calls and unwind metadata stores for any 40-byte `DropNode` stack slot that can never be active across a potentially-unwinding call.
    pub fn elide_dead_drop_guards(
        &mut self,
        mut is_drop_push: impl FnMut(u32) -> bool,
        mut is_drop_pop: impl FnMut(u32) -> bool,
        mut sym_may_unwind: impl FnMut(u32) -> bool,
    ) -> bool {
        if self.entry.is_none() || self.coroutine.is_some() || self.slots.is_empty() {
            return false;
        }
        let mut defs = vec![None; self.types.len()];
        let mut addr_slot = vec![None; self.types.len()];
        let mut slot_ok: Vec<bool> = self.slots.iter().map(|s| s.size == 40).collect();
        for block in &self.blocks {
            for &inst in &block.insts {
                let data = &self.insts[inst.index()];
                if let Some(res) = data.result {
                    defs[res.index()] = Some(inst);
                    if let Op::StackAddr(slot, off) = data.op {
                        if off == 0 {
                            addr_slot[res.index()] = Some(slot);
                        } else {
                            slot_ok[slot.index()] = false;
                        }
                    }
                }
            }
        }
        let mut kinds = vec![CalleeKind::NoUnwind; self.insts.len()];
        let mut pushes: Vec<Vec<(Block, usize)>> = vec![Vec::new(); self.slots.len()];
        let mut any_push = false;
        for (b_idx, block) in self.blocks.iter().enumerate() {
            let b = Block(b_idx as u32);
            for (pos, &inst) in block.insts.iter().enumerate() {
                let op = &self.insts[inst.index()].op;
                match op {
                    Op::Call { callee, args, .. } => {
                        let kind = classify_callee(self, &defs, *callee, &mut is_drop_push, &mut is_drop_pop, &mut sym_may_unwind);
                        kinds[inst.index()] = kind;
                        let is_guard_call = matches!(kind, CalleeKind::DropPush | CalleeKind::DropPop)
                            && args.len() == 1
                            && addr_slot[args[0].index()].is_some();
                        if is_guard_call {
                            let slot = addr_slot[args[0].index()].unwrap();
                            if kind == CalleeKind::DropPush {
                                pushes[slot.index()].push((b, pos));
                                any_push = true;
                            }
                            if let Some(s) = addr_slot[callee.index()] {
                                slot_ok[s.index()] = false;
                            }
                        } else {
                            op.for_operands(|v| {
                                if let Some(s) = addr_slot[v.index()] {
                                    slot_ok[s.index()] = false;
                                }
                            });
                        }
                    }
                    &Op::StackLoad(s, off) if off != 0 => slot_ok[s.index()] = false,
                    &Op::StackStore(v, s, off) => {
                        if !matches!(off, 0 | 24 | 32) {
                            slot_ok[s.index()] = false;
                        }
                        if let Some(vs) = addr_slot[v.index()] {
                            slot_ok[vs.index()] = false;
                        }
                    }
                    _ => {
                        op.for_operands(|v| {
                            if let Some(s) = addr_slot[v.index()] {
                                slot_ok[s.index()] = false;
                            }
                        });
                    }
                }
            }
        }
        if !any_push {
            return false;
        }
        let mut dead = vec![false; self.slots.len()];
        let mut any_dead = false;
        for (idx, slot_pushes) in pushes.iter().enumerate() {
            if !slot_ok[idx] || slot_pushes.is_empty() {
                continue;
            }
            if !slot_needs_unwind(self, &defs, &addr_slot, &kinds, StackSlot(idx as u32), slot_pushes) {
                dead[idx] = true;
                any_dead = true;
            }
        }
        if !any_dead {
            return false;
        }
        for block in &mut self.blocks {
            block.insts.retain(|&inst| match &self.insts[inst.index()].op {
                Op::Call { args, .. }
                    if matches!(kinds[inst.index()], CalleeKind::DropPush | CalleeKind::DropPop)
                        && args.len() == 1
                        && addr_slot[args[0].index()].is_some_and(|s| dead[s.index()]) =>
                {
                    false
                }
                &Op::StackStore(_, s, 24 | 32) if dead[s.index()] => false,
                _ => true,
            });
        }
        true
    }
}
