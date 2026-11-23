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
    NoUnwind,
    MayUnwind,
}

fn def_op<'a>(func: &'a Function, defs: &[Option<Inst>], val: Value) -> Option<&'a Op> {
    defs.get(val.index()).copied().flatten().map(|i| &func.insts[i.index()].op)
}

fn classify_callee(func: &Function, defs: &[Option<Inst>], callee: Value, sym_may_unwind: &mut impl FnMut(u32) -> bool) -> CalleeKind {
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
        self.unwind_drops.iter().any(|u| u.local == slot || u.flag == slot)
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
        self.elide_dead_unwind_drops_inner(false, sym_may_unwind)
    }

    /// Post-optimization unwind-drop cleanup that preserves or re-emits entry zero-initialization for every surviving flag.
    pub(crate) fn elide_dead_unwind_drops_post_init(&mut self) -> bool {
        if self.no_unwind_syms.is_empty() {
            return self.elide_dead_unwind_drops_inner(true, |_| true);
        }
        let mut no_unwind = std::mem::take(&mut self.no_unwind_syms);
        no_unwind.sort_unstable();
        let changed = self.elide_dead_unwind_drops_inner(true, |sym| no_unwind.binary_search(&sym).is_err());
        self.no_unwind_syms = no_unwind;
        changed
    }

    fn elide_dead_unwind_drops_inner(&mut self, post_init: bool, mut sym_may_unwind: impl FnMut(u32) -> bool) -> bool {
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
                    if let Op::StackAddr(slot, off) = data.op {
                        addr_slot[res.index()] = Some((slot, off));
                    }
                }
            }
        }
        let mut kinds = vec![CalleeKind::NoUnwind; self.insts.len()];
        let mut slot_escaped = vec![false; self.slots.len()];
        let mut slot_loaded = vec![false; self.slots.len()];
        let mut local_write_only = vec![true; self.slots.len()];
        let mut any_unwind_call = false;
        for block in &self.blocks {
            for &inst in &block.insts {
                let op = &self.insts[inst.index()].op;
                match op {
                    Op::Call { callee, .. } => {
                        let kind = classify_callee(self, &defs, *callee, &mut sym_may_unwind);
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
                                    && base.checked_add(off).is_some_and(|total| crate::mem::in_range(i64::from(total), self.ty(val).bytes(), self.slots[s.index()].size))
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
        let folded = fold_known_flag_loads(self, entry, (&defs, &kinds, &slot_escaped, &mut slot_loaded));
        let narrowed = narrow_unwind_drops(self, entry, post_init, (&defs, &kinds, &addr_slot, &slot_escaped, &slot_loaded, &local_write_only));
        self.unwind_drops.len() != before || folded || narrowed
    }
}

fn fold_known_flag_loads(
    func: &mut Function,
    entry: Block,
    (defs, kinds, slot_escaped, slot_loaded): (&[Option<Inst>], &[CalleeKind], &[bool], &mut [bool]),
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
                    if off != 0 || func.ty(v) != crate::Type::I8 || !matches!(def_op(func, defs, v), Some(&Op::Const(0 | 1))) {
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

fn blocks_reaching(cfg: &crate::cfg::Cfg, nblocks: usize, targets: &[(Block, Inst)]) -> Vec<bool> {
    let mut seen = vec![false; nblocks];
    let mut work: Vec<Block> = Vec::with_capacity(targets.len());
    for &(b, _) in targets {
        if !std::mem::replace(&mut seen[b.index()], true) {
            work.push(b);
        }
    }
    while let Some(b) = work.pop() {
        for &pred in cfg.preds_of(b) {
            if !std::mem::replace(&mut seen[pred.index()], true) {
                work.push(pred);
            }
        }
    }
    seen
}

fn entry_first_zeros(func: &Function, defs: &[Option<Inst>], entry: Block, nslots: usize) -> Vec<Option<Inst>> {
    let mut first = vec![None; nslots];
    let mut seen_call = false;
    for &inst in &func.blocks[entry.index()].insts {
        match func.insts[inst.index()].op {
            Op::Call { .. } | Op::TryCall { .. } => seen_call = true,
            Op::StackStore(v, slot, 0) if !seen_call && slot.index() < nslots && first[slot.index()].is_none() && store_clears_active(func, defs, v) => {
                first[slot.index()] = Some(inst);
            }
            _ => {}
        }
    }
    first
}

fn mark_dead_zeros(remove: &mut [bool], dead_zeros: &[Inst], keep_zero: Option<&[Option<Inst>]>, fi: usize) {
    let kept = keep_zero.and_then(|kz| kz[fi]);
    for &zinst in dead_zeros {
        if kept != Some(zinst) {
            remove[zinst.index()] = true;
        }
    }
}

fn narrow_unwind_drops(
    func: &mut Function,
    entry: Block,
    post_init: bool,
    (defs, kinds, addr_slot, slot_escaped, slot_loaded, local_write_only): (
        &[Option<Inst>],
        &[CalleeKind],
        &[Option<(StackSlot, i32)>],
        &[bool],
        &[bool],
        &[bool],
    ),
) -> bool {
    let nslots = func.slots.len();
    let keep_zero = post_init.then(|| entry_first_zeros(func, defs, entry, nslots));
    let mut entry_zero_flags: Vec<StackSlot> = Vec::new();
    let mut flag_uses = vec![0u32; nslots];
    let mut local_uses = vec![0u32; nslots];
    for u in &func.unwind_drops {
        if u.flag.index() < nslots { flag_uses[u.flag.index()] += 1; }
        if u.local.index() < nslots { local_uses[u.local.index()] += 1; }
    }
    let cfg = func.cfg();
    let depths = &cfg.loops().depths;
    let mut flag_stores: Vec<Vec<(Block, Inst, Option<bool>)>> = vec![Vec::new(); nslots];
    let mut local_stores: Vec<Vec<(Block, Inst, Value, i32)>> = vec![Vec::new(); nslots];
    for &b in &cfg.rpo {
        for &inst in &func.blocks[b.index()].insts {
            match func.insts[inst.index()].op {
                Op::StackStore(v, slot, off) if slot.index() < nslots => {
                    if flag_uses[slot.index()] == 1 {
                        let bit = (off == 0).then(|| match def_op(func, defs, v) {
                            Some(&Op::Const(c)) => Some(c != 0),
                            _ => None,
                        }).flatten();
                        flag_stores[slot.index()].push((b, inst, bit));
                    }
                    if local_uses[slot.index()] == 1 && local_write_only[slot.index()] {
                        local_stores[slot.index()].push((b, inst, v, off));
                    }
                }
                Op::Store(v, addr, off) => {
                    if let Some((slot, base)) = addr_slot[addr.index()]
                        && slot.index() < nslots
                        && local_uses[slot.index()] == 1
                        && local_write_only[slot.index()]
                        && let Some(total) = base.checked_add(off)
                        && crate::mem::in_range(i64::from(total), func.ty(v).bytes(), func.slots[slot.index()].size)
                    {
                        local_stores[slot.index()].push((b, inst, v, total));
                    }
                }
                _ => {}
            }
        }
    }
    let mut remove = vec![false; func.insts.len()];
    let mut before_call: Vec<(Inst, Vec<(StackSlot, Value, i32)>, Option<StackSlot>)> = Vec::new();
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
        let (dead_zeros, live_calls) = analyze_flag(func, defs, kinds, entry, u.flag);
        let narrow_calls = live_calls.filter(|lc| {
            !lc.is_empty()
                && lc.len() <= 4
                && lc.iter().map(|&(b, _)| depths[b.index()]).max().unwrap_or(0) <= arm_blocks.iter().map(|&b| depths[b.index()]).min().unwrap_or(0)
        });
        let Some(live_calls) = narrow_calls else {
            mark_dead_zeros(&mut remove, &dead_zeros, keep_zero.as_deref(), fi);
            continue;
        };
        let flag_needs_narrow = arm_blocks.iter().any(|ab| !live_calls.iter().any(|&(cb, _)| cb == *ab));
        let mut sink_local: Vec<(StackSlot, Value, i32)> = Vec::new();
        let li = u.local.index();
        if u.local != u.flag && li < nslots && local_uses[li] == 1 && local_write_only[li] && !local_stores[li].is_empty() {
            let reaches = blocks_reaching(&cfg, func.blocks.len(), &live_calls);
            let mut dom_stores = Vec::new();
            let mut ok = true;
            for &(sb, sinst, val, off) in &local_stores[li] {
                if live_calls.iter().all(|&(cb, _)| sb != cb && cfg.dominates(sb, cb)) {
                    dom_stores.push((sb, sinst, val, off));
                } else if reaches[sb.index()] {
                    ok = false;
                    break;
                }
            }
            let max_call_depth = live_calls.iter().map(|&(cb, _)| depths[cb.index()]).max().unwrap_or(0);
            let min_store_depth = dom_stores.iter().map(|&(sb, ..)| depths[sb.index()]).min().unwrap_or(0);
            if ok
                && !dom_stores.is_empty()
                && dom_stores.len() <= 4
                && dom_stores.len() * live_calls.len() <= 8
                && max_call_depth <= min_store_depth
            {
                for &(_, sinst, _, _) in &local_stores[li] {
                    remove[sinst.index()] = true;
                }
                for &(_, _, val, off) in &dom_stores {
                    sink_local.push((u.local, val, off));
                }
            }
        }
        if !flag_needs_narrow && sink_local.is_empty() {
            mark_dead_zeros(&mut remove, &dead_zeros, keep_zero.as_deref(), fi);
            continue;
        }
        if flag_needs_narrow {
            for &(_, inst, _) in stores {
                remove[inst.index()] = true;
            }
            if post_init {
                entry_zero_flags.push(u.flag);
            }
        }
        let flag_opt = flag_needs_narrow.then_some(u.flag);
        for &(_, cinst) in &live_calls {
            before_call.push((cinst, sink_local.clone(), flag_opt));
        }
    }
    if before_call.is_empty() && !remove.iter().any(|&r| r) {
        return false;
    }
    let consts = before_call.iter().any(|(_, _, f)| f.is_some()).then(|| {
        let one = func.value(crate::Type::I8);
        let one_inst = func.add(Op::Const(1), Some(one));
        let zero = func.value(crate::Type::I8);
        let zero_inst = func.add(Op::Const(0), Some(zero));
        (one, one_inst, zero, zero_inst)
    });
    let mut per_call: Vec<Vec<(Vec<(StackSlot, Value, i32)>, Option<StackSlot>)>> = vec![Vec::new(); remove.len()];
    for (cinst, locs, flag) in before_call {
        per_call[cinst.index()].push((locs, flag));
    }
    for bi in 0..func.blocks.len() {
        let old = std::mem::take(&mut func.blocks[bi].insts);
        let mut new = Vec::with_capacity(old.len() + 4);
        if bi == entry.index()
            && let Some((_, one_inst, zero, zero_inst)) = consts
        {
            new.push(one_inst);
            new.push(zero_inst);
            for &flag in &entry_zero_flags {
                new.push(func.add(Op::StackStore(zero, flag, 0), None));
            }
        }
        for inst in old {
            if inst.index() < remove.len() && remove[inst.index()] {
                continue;
            }
            let hooks = if inst.index() < per_call.len() { &per_call[inst.index()] } else { &[][..] };
            for (locs, flag) in hooks {
                for &(slot, pv, off) in locs {
                    new.push(func.add(Op::StackStore(pv, slot, off), None));
                }
                if let (Some(flag), Some((one, ..))) = (flag, consts) {
                    new.push(func.add(Op::StackStore(one, *flag, 0), None));
                }
            }
            new.push(inst);
            if let Some((_, _, zero, _)) = consts {
                for (_, flag) in hooks {
                    if let Some(flag) = flag {
                        new.push(func.add(Op::StackStore(zero, *flag, 0), None));
                    }
                }
            }
        }
        func.blocks[bi].insts = new;
    }
    true
}

enum FlagEvent {
    Call(Inst, bool),
    TryCall(bool),
    Clear(Inst, bool),
    Load(Inst, bool),
}

fn walk_flag_cfg(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    entry: Block,
    flag: StackSlot,
    mut on_event: impl FnMut(FlagEvent) -> bool,
) -> bool {
    let mut visited = vec![0u8; func.blocks.len()];
    visited[entry.index()] = 1;
    let mut work = vec![(entry, false)];
    while let Some((block, mut active)) = work.pop() {
        let mut stopped = false;
        for &inst in &func.blocks[block.index()].insts {
            match &func.insts[inst.index()].op {
                Op::Call { .. } if kinds[inst.index()] == CalleeKind::MayUnwind => {
                    if on_event(FlagEvent::Call(inst, active)) {
                        return true;
                    }
                }
                Op::TryCall { .. } => {
                    if on_event(FlagEvent::TryCall(active)) {
                        return true;
                    }
                }
                &Op::StackLoad(s, 0) if s == flag => {
                    if on_event(FlagEvent::Load(inst, active)) {
                        return true;
                    }
                }
                &Op::StackStore(v, s, 0) if s == flag => {
                    if store_clears_active(func, defs, v) {
                        if on_event(FlagEvent::Clear(inst, active)) {
                            return true;
                        }
                        active = false;
                    } else {
                        active = true;
                    }
                }
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
            if let Op::Brif(cond, then, _, otherwise, _) = *last
                && let Some(&Op::Const(imm)) = def_op(func, defs, cond)
            {
                let target = if crate::fold::extend(imm, func.ty(cond), false) != 0 { then } else { otherwise };
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
    false
}

fn analyze_flag(
    func: &Function,
    defs: &[Option<Inst>],
    kinds: &[CalleeKind],
    entry: Block,
    flag: StackSlot,
) -> (Vec<Inst>, Option<Vec<(Block, Inst)>>) {
    let mut call_state = vec![0u8; func.insts.len()];
    let mut clear_state = vec![0u8; func.insts.len()];
    let mut try_active = false;
    let bit = |a: bool| if a { 2u8 } else { 1u8 };
    walk_flag_cfg(func, defs, kinds, entry, flag, |ev| {
        match ev {
            FlagEvent::Call(inst, active) => call_state[inst.index()] |= bit(active),
            FlagEvent::TryCall(active) => try_active |= active,
            FlagEvent::Clear(inst, active) => clear_state[inst.index()] |= bit(active),
            FlagEvent::Load(..) => {}
        }
        false
    });
    let mut dead_zeros = Vec::new();
    let mut live = Vec::new();
    let mut exact = !try_active;
    for (bi, block) in func.blocks.iter().enumerate() {
        for &inst in &block.insts {
            if let Op::StackStore(v, s, 0) = func.insts[inst.index()].op
                && s == flag
                && store_clears_active(func, defs, v)
                && clear_state[inst.index()] <= 1
            {
                dead_zeros.push(inst);
            }
            match call_state[inst.index()] {
                3 => exact = false,
                2 => live.push((Block(bi as u32), inst)),
                _ => {}
            }
        }
    }
    (dead_zeros, exact.then_some(live))
}

fn unwind_flag_live_at_call(func: &Function, defs: &[Option<Inst>], kinds: &[CalleeKind], entry: Block, flag: StackSlot) -> bool {
    walk_flag_cfg(func, defs, kinds, entry, flag, |ev| {
        matches!(ev, FlagEvent::Call(_, true) | FlagEvent::TryCall(true))
    })
}
