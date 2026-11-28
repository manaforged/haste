use crate::cfg::Cfg;
use crate::ir::{AtomicOrd, Binary, Block, Convert, Function, Inst, Op, Value};
use crate::reloc::RuntimeSymbol;

use super::{MemBase, MemLoc, Provenance, may_alias_at};

const MAX_PRIVATE_BYTES: usize = 16384;
const MAX_FREE_DFS_DEPTH: usize = 8;
const MAX_FREE_DFS_STEPS: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MinOff {
    NonFrame,
    Unreached,
    AtLeast(i64),
    Unknown,
}

impl MinOff {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::NonFrame, x) | (x, Self::NonFrame) => x,
            (Self::Unreached, x) | (x, Self::Unreached) => x,
            (Self::AtLeast(a), Self::AtLeast(b)) => Self::AtLeast(a.min(b)),
            _ => Self::Unknown,
        }
    }

    fn shift(self, delta: Option<i64>, add: bool) -> Self {
        match (self, delta) {
            (Self::NonFrame, _) => Self::NonFrame,
            (Self::Unreached, _) => Self::Unreached,
            (Self::AtLeast(m), Some(c)) => {
                let next = if add {
                    m.checked_add(c)
                } else {
                    m.checked_sub(c)
                };
                next.map_or(Self::Unknown, Self::AtLeast)
            }
            _ => Self::Unknown,
        }
    }
}

/// What a coroutine resume function knows about its own frame: how many leading bytes no outside code can address, which values may point into the frame, and which hold its exact start.
#[derive(Default)]
pub(super) struct FrameFacts {
    private: u32,
    derived: Vec<bool>,
    exact: Vec<bool>,
    min_off: Vec<Option<i64>>,
}

fn derived_from_op(op: &Op, derived: &[bool]) -> bool {
    match *op {
        Op::Binary(Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Band, a, b) => {
            derived[a.index()] || derived[b.index()]
        }
        Op::Select(_, a, b) => derived[a.index()] || derived[b.index()],
        Op::Convert(Convert::Bitcast, a) => derived[a.index()],
        _ => false,
    }
}

fn op_frame_offset(op: &Op, offs: &[Option<i64>], consts: &[Option<i64>]) -> Option<i64> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => match (
            offs[a.index()],
            consts[b.index()],
            offs[b.index()],
            consts[a.index()],
        ) {
            (Some(o), Some(c), _, _) | (_, _, Some(o), Some(c)) => o.checked_add(c),
            _ => None,
        },
        Op::Binary(Binary::Isub, a, b) => offs[a.index()]?.checked_sub(consts[b.index()]?),
        Op::Convert(Convert::Bitcast, a) => offs[a.index()],
        Op::Select(_, a, b) if offs[a.index()].is_some() && offs[a.index()] == offs[b.index()] => {
            offs[a.index()]
        }
        _ => None,
    }
}

fn incoming_args(func: &Function, cfg: &Cfg) -> Vec<Vec<usize>> {
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); func.types.len()];
    for &block in &cfg.rpo {
        let Some(&last) = func.blocks[block.index()].insts.last() else {
            continue;
        };
        func.insts[last.index()].op.for_edges(|target, args| {
            for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                incoming[param.index()].push(arg.index());
            }
        });
    }
    incoming
}

fn op_min_off(op: &Op, state: &[MinOff], derived: &[bool], consts: &[Option<i64>]) -> MinOff {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => match (derived[a.index()], derived[b.index()]) {
            (true, false) => state[a.index()].shift(consts[b.index()], true),
            (false, true) => state[b.index()].shift(consts[a.index()], true),
            (false, false) => MinOff::NonFrame,
            (true, true) => MinOff::Unknown,
        },
        Op::Binary(Binary::Isub, a, b) => match (derived[a.index()], derived[b.index()]) {
            (true, false) => state[a.index()].shift(consts[b.index()], false),
            (false, false) => MinOff::NonFrame,
            _ => MinOff::Unknown,
        },
        Op::Convert(Convert::Bitcast, a) => state[a.index()],
        Op::Select(_, a, b) => state[a.index()].join(state[b.index()]),
        _ => MinOff::Unknown,
    }
}

fn compute_min_offsets(
    func: &Function,
    cfg: &Cfg,
    incoming: &[Vec<usize>],
    exact: &[bool],
    derived: &[bool],
) -> Vec<Option<i64>> {
    let count = func.types.len();
    let consts = func.const_values();
    let mut state: Vec<MinOff> = (0..count)
        .map(|v| {
            if exact[v] {
                MinOff::AtLeast(0)
            } else if derived[v] {
                MinOff::Unreached
            } else {
                MinOff::NonFrame
            }
        })
        .collect();
    for iter in 0..32 {
        let mut changed = false;
        for (param, args) in incoming.iter().enumerate() {
            if args.is_empty() || exact[param] || !derived[param] {
                continue;
            }
            let mut next = MinOff::NonFrame;
            for &arg in args {
                next = next.join(state[arg]);
            }
            if iter >= 4
                && let (MinOff::AtLeast(old), MinOff::AtLeast(cur)) = (state[param], next)
                && cur < old
            {
                next = MinOff::Unknown;
            }
            if next != state[param] {
                state[param] = next;
                changed = true;
            }
        }
        for inst in cfg
            .rpo
            .iter()
            .flat_map(|block| &func.blocks[block.index()].insts)
        {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            let r = res.index();
            if exact[r] || !derived[r] {
                continue;
            }
            let mut next = op_min_off(&data.op, &state, derived, &consts);
            if iter >= 4
                && let (MinOff::AtLeast(old), MinOff::AtLeast(cur)) = (state[r], next)
                && cur < old
            {
                next = MinOff::Unknown;
            }
            if next != state[r] {
                state[r] = next;
                changed = true;
            }
        }
        if !changed {
            return state
                .into_iter()
                .map(|m| match m {
                    MinOff::AtLeast(off) => Some(off),
                    _ => None,
                })
                .collect();
        }
    }
    vec![None; count]
}

impl FrameFacts {
    pub(super) fn outside_private(&self, v: Value, extra_off: i64) -> bool {
        !self.derived.get(v.index()).copied().unwrap_or(false)
            || self
                .min_off
                .get(v.index())
                .copied()
                .flatten()
                .and_then(|m| m.checked_add(extra_off))
                .is_some_and(|s| s >= i64::from(self.private))
    }

    pub(super) fn new(func: &Function, cfg: &Cfg) -> Self {
        if func.frame_private == 0 {
            return Self::default();
        }
        let count = func.types.len();
        let marked: Vec<bool> = (0..count)
            .map(|v| func.is_frame_ptr(Value(v as u32)))
            .collect();
        let incoming = incoming_args(func, cfg);
        let mut exact = marked.clone();
        for (param, args) in incoming.iter().enumerate() {
            if !args.is_empty() {
                exact[param] = true;
            }
        }
        let mut derived = marked.clone();
        for _ in 0..64 {
            let mut changed = false;
            for (param, args) in incoming.iter().enumerate() {
                if args.is_empty() || marked[param] {
                    continue;
                }
                if exact[param] && !args.iter().all(|&arg| exact[arg]) {
                    exact[param] = false;
                    changed = true;
                }
                if !derived[param] && args.iter().any(|&arg| derived[arg]) {
                    derived[param] = true;
                    changed = true;
                }
            }
            for inst in cfg
                .rpo
                .iter()
                .flat_map(|block| &func.blocks[block.index()].insts)
            {
                let data = &func.insts[inst.index()];
                if derived_from_op(&data.op, &derived) {
                    for result in crate::live::results(data) {
                        changed |= !std::mem::replace(&mut derived[result.index()], true);
                    }
                }
            }
            if !changed {
                let min_off = compute_min_offsets(func, cfg, &incoming, &exact, &derived);
                return Self {
                    private: func.frame_private,
                    derived,
                    exact,
                    min_off,
                };
            }
        }
        Self {
            private: func.frame_private,
            derived: vec![true; count],
            exact: vec![false; count],
            min_off: vec![None; count],
        }
    }
}

fn frame_offsets(func: &Function, cfg: &Cfg, prov: &Provenance) -> Vec<Option<i64>> {
    let count = func.types.len();
    let consts = func.const_values();
    let incoming = incoming_args(func, cfg);
    let mut offs: Vec<Option<i64>> = (0..count)
        .map(|v| {
            prov.frame
                .exact
                .get(v)
                .copied()
                .unwrap_or(false)
                .then_some(0)
        })
        .collect();
    for _ in 0..32 {
        let mut changed = false;
        for (param, args) in incoming.iter().enumerate() {
            if args.is_empty() || offs[param].is_some() {
                continue;
            }
            if let Some(first) = offs[args[0]]
                && args[1..].iter().all(|&a| offs[a] == Some(first))
            {
                offs[param] = Some(first);
                changed = true;
            }
        }
        for inst in cfg
            .rpo
            .iter()
            .flat_map(|block| &func.blocks[block.index()].insts)
        {
            let data = &func.insts[inst.index()];
            let Some(res) = data.result else { continue };
            if offs[res.index()].is_some() {
                continue;
            }
            if let Some(off) = op_frame_offset(&data.op, &offs, &consts) {
                offs[res.index()] = Some(off);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    offs
}

fn anchor_slots(func: &Function, cfg: &Cfg) -> Vec<bool> {
    let mut only_fp = vec![true; func.slots.len()];
    for inst in cfg
        .rpo
        .iter()
        .flat_map(|block| &func.blocks[block.index()].insts)
    {
        let data = &func.insts[inst.index()];
        match data.op {
            Op::StackLoad(slot, off) if slot.index() < only_fp.len() => {
                if off != 0 || !data.result.is_some_and(|r| func.is_frame_ptr(r)) {
                    only_fp[slot.index()] = false;
                }
            }
            Op::StackAddr(slot, _) if slot.index() < only_fp.len() => {
                only_fp[slot.index()] = false;
            }
            _ => {}
        }
    }
    only_fp
}

/// Drops stores into private coroutine frame bytes (`0..frame_private`) that no reachable load, move, or atomic in the function ever reads, and stores into allocations that every path frees before any read or call.
pub(crate) fn drop_unread_frame_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    if func.coroutine.is_some() {
        return false;
    }
    let mut dropped = drop_stores_before_free(func, cfg, prov);
    let private = func.frame_private as usize;
    if private == 0 || private > MAX_PRIVATE_BYTES {
        return dropped;
    }
    let offs = frame_offsets(func, cfg, prov);
    let derived = &prov.frame.derived;
    let is_derived = |v: Value| derived.get(v.index()).copied().unwrap_or(false);
    let is_private_ptr = |v: Value| !prov.frame.outside_private(v, 0);
    let anchors = anchor_slots(func, cfg);
    let mut read_bytes = vec![false; private];
    let mut mark_read = |start: i64, size: u32| {
        let Some(end) = start.checked_add(i64::from(size)) else {
            read_bytes.fill(true);
            return;
        };
        let lo = start.max(0).min(private as i64) as usize;
        let hi = end.max(0).min(private as i64) as usize;
        read_bytes[lo..hi].fill(true);
    };
    for &block in &cfg.rpo {
        for &inst in &func.blocks[block.index()].insts {
            let data = &func.insts[inst.index()];
            match &data.op {
                Op::Load(addr, off) => {
                    if let Some(base) = offs[addr.index()] {
                        let Some(bytes) = data.result.map(|r| func.ty(r).bytes()) else {
                            return false;
                        };
                        let Some(start) = base.checked_add(i64::from(*off)) else {
                            return false;
                        };
                        mark_read(start, bytes);
                    } else if !prov.frame.outside_private(*addr, i64::from(*off)) {
                        return false;
                    }
                }
                Op::VolatileLoad(..) | Op::VolatileStore(..) => return false,
                Op::AtomicLoad(addr, _)
                | Op::AtomicRmw(_, _, addr, _)
                | Op::AtomicCas(_, addr, _, _) => {
                    if let Some(base) = offs[addr.index()] {
                        let Some(bytes) = data.result.map(|r| func.ty(r).bytes()) else {
                            return false;
                        };
                        mark_read(base, bytes);
                    } else if is_private_ptr(*addr) {
                        return false;
                    }
                }
                Op::Store(val, addr, _) => {
                    if is_private_ptr(*val)
                        && !(func.is_frame_ptr(*val)
                            && offs[val.index()] == Some(0)
                            && !is_derived(*addr))
                    {
                        return false;
                    }
                }
                Op::StackStore(val, slot, off) => {
                    if is_private_ptr(*val)
                        && !(*off == 0
                            && func.is_frame_ptr(*val)
                            && anchors.get(slot.index()).copied().unwrap_or(false))
                    {
                        return false;
                    }
                }
                Op::Move(dst, src, bytes) => {
                    if is_private_ptr(*src) {
                        if offs[src.index()] == Some(0) && offs[dst.index()] == Some(0) {
                            continue;
                        }
                        let Some(base) = offs[src.index()] else {
                            return false;
                        };
                        mark_read(base, *bytes);
                    }
                }
                Op::Copy(_, src, len) => {
                    if is_private_ptr(*src) || is_private_ptr(*len) {
                        return false;
                    }
                }
                Op::Call { callee, args, .. } | Op::TryCall { callee, args, .. } => {
                    if prov.frame_runtime(*callee) == Some(RuntimeSymbol::Free) {
                        continue;
                    }
                    if is_private_ptr(*callee) || args.iter().any(|&a| is_private_ptr(a)) {
                        return false;
                    }
                }
                Op::Binary(Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Band, ..)
                | Op::Select(..)
                | Op::Convert(Convert::Bitcast, _)
                | Op::Icmp(..)
                | Op::Jump(..)
                | Op::Brif(..)
                | Op::Assume(_) => {}
                other => {
                    let mut bad = false;
                    other.for_operands(|v| bad |= is_private_ptr(v));
                    if bad {
                        return false;
                    }
                }
            }
        }
    }
    for &block in &cfg.rpo {
        let Function {
            blocks,
            insts,
            types,
            ..
        } = func;
        blocks[block.index()].insts.retain(|&inst| {
            let Op::Store(val, addr, off) = insts[inst.index()].op else {
                return true;
            };
            let Some(base) = offs[addr.index()] else {
                return true;
            };
            let Some(start) = base.checked_add(i64::from(off)) else {
                return true;
            };
            let bytes = types[val.index()].bytes();
            let Some(end) = start.checked_add(i64::from(bytes)) else {
                return true;
            };
            if start >= 0
                && end <= private as i64
                && read_bytes[start as usize..end as usize].iter().all(|&r| !r)
            {
                insts[inst.index()].op = Op::Trap;
                dropped = true;
                false
            } else {
                true
            }
        });
    }
    dropped
}

fn edge_track_ptr(func: &Function, target: Block, args: &[Value], ptrs: [Value; 2]) -> [Value; 2] {
    for (i, &arg) in args.iter().enumerate() {
        if (arg == ptrs[0] || arg == ptrs[1])
            && let Some(&param) = func.blocks[target.index()].params.get(i)
        {
            return [ptrs[0], param];
        }
    }
    ptrs
}

fn free_kills_ptrs(
    prov: &Provenance,
    consts: &[Option<i64>],
    (callee, args): (Value, &[Value]),
    ptrs: [Value; 2],
    span: (i64, i64),
) -> Option<bool> {
    if prov.frame_runtime(callee) != Some(RuntimeSymbol::Free) {
        return None;
    }
    let matches = args.first().is_some_and(|&a| a == ptrs[0] || a == ptrs[1]);
    let in_bounds = args
        .get(1)
        .and_then(|&sz| consts.get(sz.index()).copied().flatten())
        .is_some_and(|sz| span.0 >= 0 && span.1 <= sz);
    Some(matches && in_bounds)
}

fn loc_aliases_ptrs(
    prov: &Provenance,
    read: MemLoc,
    ptrs: [Value; 2],
    (off, size): (i64, u32),
    at: Inst,
) -> bool {
    let check = |p: Value| {
        let cur = MemLoc::new(
            prov.param_roots
                .get(p.index())
                .copied()
                .unwrap_or(super::MemRoot::Unknown),
            MemBase::Addr(p),
            off,
            size,
        );
        may_alias_at(read, cur, prov, Some(at))
    };
    check(ptrs[0]) || (ptrs[1] != ptrs[0] && check(ptrs[1]))
}

fn store_killed_by_free(
    func: &Function,
    prov: &Provenance,
    consts: &[Option<i64>],
    (block, start_pos, ptrs, off, size): (Block, usize, [Value; 2], i64, u32),
    (depth, steps, on_path, proved): (
        usize,
        &mut usize,
        &mut Vec<Block>,
        &mut Vec<(Block, [Value; 2])>,
    ),
) -> bool {
    let Some(end) = off.checked_add(i64::from(size)) else {
        return false;
    };
    if depth > MAX_FREE_DFS_DEPTH || *steps >= MAX_FREE_DFS_STEPS || on_path.contains(&block) {
        return false;
    }
    if start_pos == 0 && proved.contains(&(block, ptrs)) {
        return true;
    }
    *steps += 1;
    let insts = &func.blocks[block.index()].insts;
    for &inst in &insts[start_pos..] {
        let data = &func.insts[inst.index()];
        match &data.op {
            Op::Call { callee, args, .. } => {
                match free_kills_ptrs(prov, consts, (*callee, args), ptrs, (off, end)) {
                    Some(true) => {
                        if start_pos == 0 {
                            proved.push((block, ptrs));
                        }
                        return true;
                    }
                    Some(false) => {}
                    None => return false,
                }
            }
            Op::TryCall { .. }
            | Op::Return(..)
            | Op::Trap
            | Op::Trapz(_)
            | Op::BrTable(..)
            | Op::Fence(_)
            | Op::VolatileLoad(..)
            | Op::VolatileStore(..) => return false,
            Op::AtomicStore(_, _, ord) if !matches!(ord, AtomicOrd::Relaxed) => return false,
            Op::Load(a, o) => {
                let Some(res) = data.result else { return false };
                if loc_aliases_ptrs(
                    prov,
                    prov.loc(*a, i64::from(*o), func.ty(res).bytes()),
                    ptrs,
                    (off, size),
                    inst,
                ) {
                    return false;
                }
            }
            Op::AtomicLoad(a, _) | Op::AtomicRmw(_, _, a, _) | Op::AtomicCas(_, a, _, _) => {
                let Some(res) = data.result else { return false };
                if loc_aliases_ptrs(
                    prov,
                    prov.loc(*a, 0, func.ty(res).bytes()),
                    ptrs,
                    (off, size),
                    inst,
                ) {
                    return false;
                }
            }
            Op::Move(_, src, bytes) if *bytes > 0 => {
                if loc_aliases_ptrs(prov, prov.loc(*src, 0, *bytes), ptrs, (off, size), inst) {
                    return false;
                }
            }
            Op::Copy(_, src, _) => {
                if loc_aliases_ptrs(prov, prov.loc(*src, 0, u32::MAX), ptrs, (off, size), inst) {
                    return false;
                }
            }
            Op::Store(v, a, o) => {
                let wloc = prov.loc(*a, i64::from(*o), func.ty(*v).bytes());
                if (wloc.base == MemBase::Addr(ptrs[0]) || wloc.base == MemBase::Addr(ptrs[1]))
                    && wloc.offset <= off
                    && wloc
                        .offset
                        .checked_add(i64::from(wloc.size))
                        .is_some_and(|we| we >= end)
                {
                    if start_pos == 0 {
                        proved.push((block, ptrs));
                    }
                    return true;
                }
            }
            Op::Jump(target, args) => {
                let next_ptrs = edge_track_ptr(func, *target, args, ptrs);
                on_path.push(block);
                let ok = store_killed_by_free(
                    func,
                    prov,
                    consts,
                    (*target, 0, next_ptrs, off, size),
                    (depth + 1, steps, on_path, proved),
                );
                on_path.pop();
                if ok && start_pos == 0 {
                    proved.push((block, ptrs));
                }
                return ok;
            }
            Op::Brif(_, then_b, then_args, else_b, else_args) => {
                let then_ptrs = edge_track_ptr(func, *then_b, then_args, ptrs);
                let else_ptrs = edge_track_ptr(func, *else_b, else_args, ptrs);
                on_path.push(block);
                let ok = store_killed_by_free(
                    func,
                    prov,
                    consts,
                    (*then_b, 0, then_ptrs, off, size),
                    (depth + 1, steps, on_path, proved),
                ) && store_killed_by_free(
                    func,
                    prov,
                    consts,
                    (*else_b, 0, else_ptrs, off, size),
                    (depth + 1, steps, on_path, proved),
                );
                on_path.pop();
                if ok && start_pos == 0 {
                    proved.push((block, ptrs));
                }
                return ok;
            }
            _ => {}
        }
    }
    false
}

fn drop_stores_before_free(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    if !prov
        .frame_runtime
        .iter()
        .any(|&rt| rt == Some(RuntimeSymbol::Free))
    {
        return false;
    }
    let consts = func.const_values();
    let mut dead: Vec<(Block, Inst)> = Vec::new();
    let (mut on_path, mut proved) = (Vec::new(), Vec::new());
    for &block in &cfg.rpo {
        let insts = &func.blocks[block.index()].insts;
        for (pos, &inst) in insts.iter().enumerate() {
            let Op::Store(val, addr, off) = func.insts[inst.index()].op else {
                continue;
            };
            let loc = prov.loc(addr, i64::from(off), func.ty(val).bytes());
            let MemBase::Addr(fp) = loc.base else {
                continue;
            };
            if loc.offset < 0 || loc.size == 0 {
                continue;
            }
            on_path.clear();
            proved.clear();
            let mut steps = 0;
            if store_killed_by_free(
                func,
                prov,
                &consts,
                (block, pos + 1, [fp, fp], loc.offset, loc.size),
                (0, &mut steps, &mut on_path, &mut proved),
            ) {
                dead.push((block, inst));
            }
        }
    }
    if dead.is_empty() {
        return false;
    }
    for (block, inst) in dead {
        func.insts[inst.index()].op = Op::Trap;
        func.blocks[block.index()].insts.retain(|&i| i != inst);
    }
    true
}

impl Provenance {
    fn exact_frame(&self, loc: MemLoc) -> bool {
        matches!(loc.base, MemBase::Addr(v) if self.frame.exact.get(v.index()).copied().unwrap_or(false))
    }

    /// Whether `loc` lies in the private bytes of this coroutine's own frame, which only addresses derived from the frame start can reach.
    pub(crate) fn has_private_frame(&self) -> bool {
        self.frame.private > 0
    }

    pub(crate) fn private_frame(&self, loc: MemLoc) -> bool {
        self.frame.private > 0
            && loc.size != u32::MAX
            && self.exact_frame(loc)
            && loc.offset >= 0
            && loc.offset + i64::from(loc.size) <= i64::from(self.frame.private)
    }

    /// Whether `op` takes an operand that may point into this coroutine's own frame.
    pub(crate) fn touches_frame(&self, op: &Op) -> bool {
        let mut touches = false;
        op.for_operands(|value| {
            touches |= self
                .frame
                .derived
                .get(value.index())
                .copied()
                .unwrap_or(false)
        });
        touches
    }

    /// Whether `a` and `b` may overlap when one of them lies in the private frame bytes; `None` when neither does.
    pub(super) fn frame_alias(&self, a: MemLoc, b: MemLoc) -> Option<bool> {
        let (private, other) = match (self.private_frame(a), self.private_frame(b)) {
            (false, false) => return None,
            (true, _) => (a, b),
            (false, true) => (b, a),
        };
        Some(match other.base {
            MemBase::Addr(_) if self.exact_frame(other) => private.overlaps(other),
            MemBase::Addr(v) => !self.frame.outside_private(v, other.offset),
            MemBase::Slot(slot) => !self.slot_unescaped(slot),
            MemBase::Global(_) => false,
        })
    }
}
