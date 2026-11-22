use crate::ir::{Binary, Block, Function, Inst, Op, StackSlot, Value};

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
enum CalleeKind {
    DropPush,
    DropPop,
    NoUnwind,
    MayUnwind,
}

fn def_op<'a>(func: &'a Function, defs: &[Option<Inst>], val: Value) -> Option<&'a Op> {
    defs.get(val.index()).copied().flatten().map(|i| &func.insts[i.index()].op)
}

fn guard_call_slot(kind: CalleeKind, args: &[Value], addr_slot: &[Option<StackSlot>]) -> Option<StackSlot> {
    (matches!(kind, CalleeKind::DropPush | CalleeKind::DropPop) && args.len() == 1)
        .then(|| addr_slot[args[0].index()])
        .flatten()
}

fn classify_callee(
    func: &Function,
    defs: &[Option<Inst>],
    callee: Value,
    is_drop_push: &mut impl FnMut(u32) -> bool,
    is_drop_pop: &mut impl FnMut(u32) -> bool,
    sym_may_unwind: &mut impl FnMut(u32) -> bool,
) -> CalleeKind {
    let sym = match def_op(func, defs, callee) {
        Some(&(Op::Symbol(sym) | Op::SlotAddr(sym))) => Some(sym),
        Some(&Op::Load(addr, 0)) => match def_op(func, defs, addr) {
            Some(&Op::SlotAddr(sym)) => Some(sym),
            _ => None,
        },
        Some(Op::Runtime(_)) => return CalleeKind::NoUnwind,
        _ => None,
    };
    let Some(sym) = sym else {
        return CalleeKind::MayUnwind;
    };
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

fn store_clears_active(func: &Function, defs: &[Option<Inst>], val: Value) -> bool {
    match def_op(func, defs, val) {
        Some(&Op::Const(imm)) if (imm & 1) == 0 => true,
        Some(&Op::Binary(Binary::Band, a, b)) => {
            let clears = |v: Value| matches!(def_op(func, defs, v), Some(&Op::Const(imm)) if (imm & 1) == 0);
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
                Op::Call { args, .. } => {
                    let kind = kinds[inst.index()];
                    if kind == CalleeKind::DropPop && guard_call_slot(kind, args, addr_slot) == Some(slot) {
                        stopped = true;
                        break;
                    }
                    if kind == CalleeKind::MayUnwind && active {
                        return true;
                    }
                }
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
                        if let Some(slot) = guard_call_slot(kind, args, &addr_slot) {
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
                Op::Call { args, .. } if guard_call_slot(kinds[inst.index()], args, &addr_slot).is_some_and(|s| dead[s.index()]) => false,
                &Op::StackStore(_, s, 24 | 32) if dead[s.index()] => false,
                _ => true,
            });
        }
        true
    }

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
        self.unwind_drops.iter().any(|u| u.local == slot || u.flag == slot)
    }

    pub(crate) fn import_unwind_drops(&mut self, callee: &Function, slot_base: u32, sym_base: u32) {
        self.keep_fp |= callee.keep_fp;
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
    pub fn elide_dead_unwind_drops(&mut self, mut sym_may_unwind: impl FnMut(u32) -> bool) -> bool {
        let Some(entry) = self.entry else { return false };
        if self.coroutine.is_some() || self.unwind_drops.is_empty() {
            return false;
        }
        let mut defs = vec![None; self.types.len()];
        let mut addr_slot = vec![None; self.types.len()];
        for block in &self.blocks {
            for &inst in &block.insts {
                let data = &self.insts[inst.index()];
                if let Some(res) = data.result {
                    defs[res.index()] = Some(inst);
                    if let Op::StackAddr(slot, _) = data.op {
                        addr_slot[res.index()] = Some(slot);
                    }
                }
            }
        }
        let mut kinds = vec![CalleeKind::NoUnwind; self.insts.len()];
        let mut slot_escaped = vec![false; self.slots.len()];
        let mut slot_loaded = vec![false; self.slots.len()];
        let mut any_unwind_call = false;
        for block in &self.blocks {
            for &inst in &block.insts {
                let op = &self.insts[inst.index()].op;
                match op {
                    Op::Call { callee, .. } => {
                        let kind = classify_callee(self, &defs, *callee, &mut |_| false, &mut |_| false, &mut sym_may_unwind);
                        kinds[inst.index()] = kind;
                        any_unwind_call |= kind == CalleeKind::MayUnwind;
                    }
                    Op::TryCall { .. } => any_unwind_call = true,
                    Op::StackLoad(slot, _) if slot.index() < slot_loaded.len() => slot_loaded[slot.index()] = true,
                    _ => {}
                }
                op.for_operands(|v| {
                    if let Some(s) = addr_slot[v.index()]
                        && s.index() < slot_escaped.len()
                    {
                        slot_escaped[s.index()] = true;
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
        let narrowed = narrow_unwind_drops(self, entry, (&defs, &kinds, &slot_escaped, &slot_loaded));
        self.unwind_drops.len() != before || narrowed
    }
}

fn narrow_unwind_drops(
    func: &mut Function,
    entry: Block,
    (defs, kinds, slot_escaped, slot_loaded): (&[Option<Inst>], &[CalleeKind], &[bool], &[bool]),
) -> bool {
    let nslots = func.slots.len();
    let mut flag_uses = vec![0u32; nslots];
    let mut local_uses = vec![0u32; nslots];
    for u in &func.unwind_drops {
        if u.flag.index() < nslots { flag_uses[u.flag.index()] += 1; }
        if u.local.index() < nslots { local_uses[u.local.index()] += 1; }
    }
    let mut flag_stores: Vec<Vec<(Block, Inst, Option<bool>)>> = vec![Vec::new(); nslots];
    let mut local_stores: Vec<Vec<(Block, Inst, Option<Value>)>> = vec![Vec::new(); nslots];
    for (bi, block) in func.blocks.iter().enumerate() {
        let b = Block(bi as u32);
        for &inst in &block.insts {
            if let Op::StackStore(v, slot, off) = func.insts[inst.index()].op
                && slot.index() < nslots
            {
                if flag_uses[slot.index()] == 1 {
                    let bit = (off == 0).then(|| match def_op(func, defs, v) {
                        Some(&Op::Const(c)) => Some(c != 0),
                        _ => None,
                    }).flatten();
                    flag_stores[slot.index()].push((b, inst, bit));
                }
                if local_uses[slot.index()] == 1 {
                    local_stores[slot.index()].push((b, inst, (off == 0).then_some(v)));
                }
            }
        }
    }
    let cfg = func.cfg();
    let depths = &cfg.loops().depths;
    let mut remove = vec![false; func.insts.len()];
    let mut before_call: Vec<(Inst, Option<(StackSlot, Value)>, StackSlot)> = Vec::new();
    for u in &func.unwind_drops {
        let fi = u.flag.index();
        if fi >= nslots || flag_uses[fi] != 1 || slot_escaped[fi] || slot_loaded[fi] {
            continue;
        }
        let stores = &flag_stores[fi];
        if stores.is_empty() || stores.iter().any(|s| s.2.is_none()) {
            continue;
        }
        let arm_blocks: Vec<Block> = stores.iter().filter(|s| s.2 == Some(true)).map(|s| s.0).collect();
        if arm_blocks.is_empty() {
            continue;
        }
        let Some(live_calls) = exact_live_calls(func, defs, kinds, entry, u.flag) else { continue };
        if live_calls.is_empty() || live_calls.len() > 4 {
            continue;
        }
        let max_call_depth = live_calls.iter().map(|&(b, _)| depths[b.index()]).max().unwrap_or(0);
        let min_arm_depth = arm_blocks.iter().map(|&b| depths[b.index()]).min().unwrap_or(0);
        if max_call_depth > min_arm_depth
            || !arm_blocks.iter().any(|ab| !live_calls.iter().any(|&(cb, _)| cb == *ab))
        {
            continue;
        }
        for &(_, inst, _) in stores {
            remove[inst.index()] = true;
        }
        let mut sink_local = None;
        let li = u.local.index();
        if u.indirect
            && u.local_off == 0
            && live_calls.len() == 1
            && li < nslots
            && local_uses[li] == 1
            && !slot_escaped[li]
            && !slot_loaded[li]
            && local_stores[li].len() == 1
        {
            let (sb, sinst, pval) = local_stores[li][0];
            let cb = live_calls[0].0;
            if let Some(pv) = pval && sb != cb && cfg.dominates(sb, cb) {
                remove[sinst.index()] = true;
                sink_local = Some((u.local, pv));
            }
        }
        for &(_, cinst) in &live_calls {
            before_call.push((cinst, sink_local, u.flag));
        }
    }
    if before_call.is_empty() {
        return false;
    }
    let one = func.value(crate::Type::I8);
    let one_inst = func.add(Op::Const(1), Some(one));
    let zero = func.value(crate::Type::I8);
    let zero_inst = func.add(Op::Const(0), Some(zero));
    let mut per_call: Vec<Vec<(Option<(StackSlot, Value)>, StackSlot)>> = vec![Vec::new(); remove.len()];
    for (cinst, loc, flag) in before_call {
        per_call[cinst.index()].push((loc, flag));
    }
    for bi in 0..func.blocks.len() {
        let old = std::mem::take(&mut func.blocks[bi].insts);
        let mut new = Vec::with_capacity(old.len() + 4);
        if bi == entry.index() {
            new.push(one_inst);
            new.push(zero_inst);
        }
        for inst in old {
            if inst.index() < remove.len() && remove[inst.index()] {
                continue;
            }
            let hooks = if inst.index() < per_call.len() { &per_call[inst.index()] } else { &[][..] };
            for &(loc, flag) in hooks {
                if let Some((slot, pv)) = loc {
                    new.push(func.add(Op::StackStore(pv, slot, 0), None));
                }
                new.push(func.add(Op::StackStore(one, flag, 0), None));
            }
            new.push(inst);
            for &(_, flag) in hooks {
                new.push(func.add(Op::StackStore(zero, flag, 0), None));
            }
        }
        func.blocks[bi].insts = new;
    }
    true
}

fn exact_live_calls(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    entry: Block,
    flag: StackSlot,
) -> Option<Vec<(Block, Inst)>> {
    let mut visited = vec![0u8; func.blocks.len()];
    let mut call_state = vec![0u8; func.insts.len()];
    visited[entry.index()] = 1;
    let mut work = vec![(entry, false)];
    while let Some((block, mut active)) = work.pop() {
        let mut stopped = false;
        let bit = |a: bool| if a { 2u8 } else { 1u8 };
        for &inst in &func.blocks[block.index()].insts {
            match &func.insts[inst.index()].op {
                Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => {
                    call_state[inst.index()] |= bit(active);
                }
                Op::TryCall { .. } => {
                    if active { return None; }
                }
                &Op::StackStore(v, s, 0) if s == flag => active = !store_clears_active(func, defs, v),
                Op::Return(..) | Op::Trap => {
                    stopped = true;
                    break;
                }
                _ => {}
            }
        }
        if stopped { continue; }
        let b = bit(active);
        if let Some(last) = func.last(block) {
            for target in last.targets() {
                if visited[target.index()] & b == 0 {
                    visited[target.index()] |= b;
                    work.push((target, active));
                }
            }
        }
    }
    let mut live = Vec::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            match call_state[inst.index()] {
                3 => return None,
                2 => live.push((Block(bi as u32), inst)),
                _ => {}
            }
        }
    }
    Some(live)
}

fn unwind_flag_live_at_call(func: &Function, defs: &[Option<Inst>], kinds: &[CalleeKind], entry: Block, flag: StackSlot) -> bool {
    let mut visited = vec![0u8; func.blocks.len()];
    visited[entry.index()] = 1;
    let mut work = vec![(entry, false)];
    while let Some((block, mut active)) = work.pop() {
        let mut stopped = false;
        for &inst in &func.blocks[block.index()].insts {
            match &func.insts[inst.index()].op {
                Op::Call { .. } if active && kinds[inst.index()] == CalleeKind::MayUnwind => return true,
                Op::TryCall { .. } if active => return true,
                &Op::StackStore(v, s, 0) if s == flag => active = !store_clears_active(func, defs, v),
                Op::Return(..) | Op::Trap => {
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
                    work.push((target, active));
                }
            }
        }
    }
    false
}
