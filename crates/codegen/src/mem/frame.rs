use crate::cfg::Cfg;
use crate::ir::{Binary, Convert, Function, Op, Value};
use crate::reloc::RuntimeSymbol;

use super::{MemBase, MemLoc, Provenance};

const MAX_PRIVATE_BYTES: usize = 16384;

/// What a coroutine resume function knows about its own frame: how many leading bytes no outside code can address, which values may point into the frame, and which hold its exact start.
#[derive(Default)]
pub(super) struct FrameFacts {
    private: u32,
    derived: Vec<bool>,
    exact: Vec<bool>,
}

fn derived_from_op(op: &Op, derived: &[bool]) -> bool {
    match *op {
        Op::Binary(Binary::Iadd | Binary::Isub | Binary::Bor | Binary::Band, a, b) => derived[a.index()] || derived[b.index()],
        Op::Select(_, a, b) => derived[a.index()] || derived[b.index()],
        Op::Convert(Convert::Bitcast, a) => derived[a.index()],
        _ => false,
    }
}

fn op_frame_offset(op: &Op, offs: &[Option<i64>], consts: &[Option<i64>]) -> Option<i64> {
    match *op {
        Op::Binary(Binary::Iadd, a, b) => match (offs[a.index()], consts[b.index()], offs[b.index()], consts[a.index()]) {
            (Some(o), Some(c), _, _) | (_, _, Some(o), Some(c)) => o.checked_add(c),
            _ => None,
        },
        Op::Binary(Binary::Isub, a, b) => offs[a.index()]?.checked_sub(consts[b.index()]?),
        Op::Convert(Convert::Bitcast, a) => offs[a.index()],
        Op::Select(_, a, b) if offs[a.index()].is_some() && offs[a.index()] == offs[b.index()] => offs[a.index()],
        _ => None,
    }
}

fn incoming_args(func: &Function, cfg: &Cfg) -> Vec<Vec<usize>> {
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); func.types.len()];
    for &block in &cfg.rpo {
        let Some(&last) = func.blocks[block.index()].insts.last() else { continue };
        func.insts[last.index()].op.for_edges(|target, args| {
            for (param, arg) in func.blocks[target.index()].params.iter().zip(args) {
                incoming[param.index()].push(arg.index());
            }
        });
    }
    incoming
}

impl FrameFacts {
    pub(super) fn new(func: &Function, cfg: &Cfg) -> Self {
        if func.frame_private == 0 {
            return Self::default();
        }
        let count = func.types.len();
        let marked: Vec<bool> = (0..count).map(|v| func.is_frame_ptr(Value(v as u32))).collect();
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
            for inst in cfg.rpo.iter().flat_map(|block| &func.blocks[block.index()].insts) {
                let data = &func.insts[inst.index()];
                if derived_from_op(&data.op, &derived) {
                    for result in crate::live::results(data) {
                        changed |= !std::mem::replace(&mut derived[result.index()], true);
                    }
                }
            }
            if !changed {
                return Self { private: func.frame_private, derived, exact };
            }
        }
        Self { private: func.frame_private, derived: vec![true; count], exact: vec![false; count] }
    }
}

fn frame_offsets(func: &Function, cfg: &Cfg, prov: &Provenance) -> Vec<Option<i64>> {
    let count = func.types.len();
    let consts = func.const_values();
    let incoming = incoming_args(func, cfg);
    let mut offs: Vec<Option<i64>> = (0..count).map(|v| prov.frame.exact.get(v).copied().unwrap_or(false).then_some(0)).collect();
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
        for inst in cfg.rpo.iter().flat_map(|block| &func.blocks[block.index()].insts) {
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
    for inst in cfg.rpo.iter().flat_map(|block| &func.blocks[block.index()].insts) {
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

/// Drops stores into private coroutine frame bytes (`0..frame_private`) that no reachable load, move, or atomic in the function ever reads.
pub(crate) fn drop_unread_frame_stores(func: &mut Function, cfg: &Cfg, prov: &Provenance) -> bool {
    let private = func.frame_private as usize;
    if private == 0 || private > MAX_PRIVATE_BYTES || func.coroutine.is_some() {
        return false;
    }
    let offs = frame_offsets(func, cfg, prov);
    let derived = &prov.frame.derived;
    let is_derived = |v: Value| derived.get(v.index()).copied().unwrap_or(false);
    let is_private_ptr = |v: Value| is_derived(v) && !offs[v.index()].is_some_and(|o| o >= private as i64);
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
                        let Some(bytes) = data.result.map(|r| func.ty(r).bytes()) else { return false };
                        let Some(start) = base.checked_add(i64::from(*off)) else { return false };
                        mark_read(start, bytes);
                    } else if is_derived(*addr) {
                        return false;
                    }
                }
                Op::AtomicLoad(addr, _) | Op::AtomicRmw(_, _, addr, _) | Op::AtomicCas(_, addr, _, _) => {
                    if let Some(base) = offs[addr.index()] {
                        let Some(bytes) = data.result.map(|r| func.ty(r).bytes()) else { return false };
                        mark_read(base, bytes);
                    } else if is_derived(*addr) {
                        return false;
                    }
                }
                Op::Store(val, addr, _) => {
                    if is_private_ptr(*val) && !(func.is_frame_ptr(*val) && offs[val.index()] == Some(0) && !is_derived(*addr)) {
                        return false;
                    }
                }
                Op::StackStore(val, slot, off) => {
                    if is_private_ptr(*val) && !(*off == 0 && func.is_frame_ptr(*val) && anchors.get(slot.index()).copied().unwrap_or(false)) {
                        return false;
                    }
                }
                Op::Move(dst, src, bytes) => {
                    if is_derived(*src) {
                        if offs[src.index()] == Some(0) && offs[dst.index()] == Some(0) {
                            continue;
                        }
                        let Some(base) = offs[src.index()] else { return false };
                        mark_read(base, *bytes);
                    }
                }
                Op::Copy(_, src, len) => {
                    if is_derived(*src) || is_private_ptr(*len) {
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
    let mut dropped = false;
    for &block in &cfg.rpo {
        let Function { blocks, insts, types, .. } = func;
        blocks[block.index()].insts.retain(|&inst| {
            let Op::Store(val, addr, off) = insts[inst.index()].op else { return true };
            let Some(base) = offs[addr.index()] else { return true };
            let Some(start) = base.checked_add(i64::from(off)) else { return true };
            let bytes = types[val.index()].bytes();
            let Some(end) = start.checked_add(i64::from(bytes)) else { return true };
            if start >= 0 && end <= private as i64 && read_bytes[start as usize..end as usize].iter().all(|&r| !r) {
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

impl Provenance {
    fn exact_frame(&self, loc: MemLoc) -> bool {
        matches!(loc.base, MemBase::Addr(v) if self.frame.exact.get(v.index()).copied().unwrap_or(false))
    }

    /// Whether `loc` lies in the private bytes of this coroutine's own frame, which only addresses derived from the frame start can reach.
    pub(crate) fn private_frame(&self, loc: MemLoc) -> bool {
        self.frame.private > 0 && loc.size != u32::MAX && self.exact_frame(loc) && loc.offset >= 0 && loc.offset + i64::from(loc.size) <= i64::from(self.frame.private)
    }

    /// Whether `op` takes an operand that may point into this coroutine's own frame.
    pub(crate) fn touches_frame(&self, op: &Op) -> bool {
        let mut touches = false;
        op.for_operands(|value| touches |= self.frame.derived.get(value.index()).copied().unwrap_or(false));
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
            MemBase::Addr(v) => self.frame.derived.get(v.index()).copied().unwrap_or(true),
            MemBase::Slot(slot) => !self.slot_unescaped(slot),
            MemBase::Global(_) => false,
        })
    }
}
