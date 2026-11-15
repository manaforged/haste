use crate::cfg::Cfg;
use crate::ir::{Block, Function, Inst, Op, Type, Value};
use crate::mem::{KnownLoads, MemBase, MemLoc, MemRoot, Provenance, clobbers, is_atomic_barrier, is_rodata, may_alias};
use crate::merge::find;

const MAX_MERGE_WALK: usize = 32;

/// Precomputed per-block write instructions and RPO ranks for cross-block load forwarding.
pub(super) struct BlockWrites<'a> {
    pub cfg: &'a Cfg,
    pub prov: &'a Provenance,
    rank: Vec<usize>,
    writes: Vec<Vec<Inst>>,
    visit: Vec<u32>,
    epoch: u32,
}

fn canonical_write(op: &Op, subst: &[Value]) -> Option<Op> {
    match *op {
        Op::Store(v, addr, off) => {
            let c = find(subst, addr);
            (c != addr).then_some(Op::Store(v, c, off))
        }
        Op::Move(dst, src, bytes) => {
            let c = find(subst, dst);
            (c != dst).then_some(Op::Move(c, src, bytes))
        }
        Op::Copy(dst, val, count) => {
            let c = find(subst, dst);
            (c != dst).then_some(Op::Copy(c, val, count))
        }
        _ => None,
    }
}

pub(super) fn record_base(bases: &mut Vec<(MemBase, MemRoot)>, loc: MemLoc) {
    if !bases.iter().any(|&(b, _)| b == loc.base) {
        bases.push((loc.base, loc.root));
    }
}

/// Whether `loc` lies in guaranteed-immutable read-only memory (`.rodata` or a `Freeze` `&T` param).
pub(super) fn loc_immutable(func: &Function, loc: MemLoc) -> bool {
    match loc.root {
        MemRoot::ReadonlyParam(_) => true,
        MemRoot::Global(_) => is_rodata(func, loc),
        _ => false,
    }
}

/// Whether `op` may overwrite `loc`, treating immutable `.rodata` and non-atomic `ReadonlyParam` accesses as un-clobbered by calls or alien stores.
pub(super) fn clobbers_loc(func: &Function, op: &Op, loc: MemLoc, prov: &Provenance) -> bool {
    if is_rodata(func, loc) {
        return false;
    }
    if matches!(loc.root, MemRoot::ReadonlyParam(_)) {
        return match *op {
            Op::Store(val, addr, off) => {
                let w = prov.loc(addr, i64::from(off), func.ty(val).bytes());
                w.root == loc.root && may_alias(w, loc, prov)
            }
            Op::Move(dst, _, bytes) => {
                let w = prov.loc(dst, 0, bytes);
                w.root == loc.root && may_alias(w, loc, prov)
            }
            Op::Copy(dst, ..) => prov.loc(dst, 0, u32::MAX).root == loc.root,
            _ => is_atomic_barrier(op),
        };
    }
    clobbers(func, op, loc, prov)
}

fn save_immutable_entries(
    func: &Function,
    known: &KnownLoads,
    bases: &[(MemBase, MemRoot)],
    keep: impl Fn(MemLoc) -> bool,
) -> Vec<(MemLoc, Type, Value)> {
    let mut saved = Vec::new();
    for &(base, root) in bases {
        if !loc_immutable(func, MemLoc::new(root, base, 0, 1)) {
            continue;
        }
        for &(off, ty, val) in known.entries_for(base) {
            let loc = MemLoc::new(root, base, off, ty.bytes());
            if keep(loc) {
                saved.push((loc, ty, val));
            }
        }
    }
    saved
}

pub(super) fn invalidate_preserving(func: &Function, known: &mut KnownLoads, bases: &[(MemBase, MemRoot)], op: &Op, prov: &Provenance) {
    let saved = save_immutable_entries(func, known, bases, |loc| !clobbers_loc(func, op, loc, prov));
    known.invalidate(func, op, prov);
    for (loc, ty, val) in saved {
        known.remember(loc, ty, val);
    }
}

pub(super) fn write_preserving(
    func: &Function,
    known: &mut KnownLoads,
    bases: &[(MemBase, MemRoot)],
    wloc: MemLoc,
    ty: Type,
    value: Value,
    prov: &Provenance,
) {
    let saved = save_immutable_entries(func, known, bases, |loc| {
        matches!(loc.root, MemRoot::Global(_)) || wloc.root != loc.root || !may_alias(wloc, loc, prov)
    });
    known.write(wloc, ty, value, prov);
    for (loc, ety, eval) in saved {
        known.remember(loc, ety, eval);
    }
}

impl<'a> BlockWrites<'a> {
    /// Collects potentially clobbering memory operations per block.
    pub(super) fn new(func: &Function, cfg: &'a Cfg, prov: &'a Provenance) -> Self {
        let mut writes = vec![Vec::new(); func.blocks.len()];
        for (b_idx, block) in func.blocks.iter().enumerate() {
            for &inst in &block.insts {
                if matches!(
                    func.insts[inst.index()].op,
                    Op::Store(..)
                        | Op::StackStore(..)
                        | Op::Copy(..)
                        | Op::Move(..)
                        | Op::AtomicLoad(_)
                        | Op::AtomicStore(..)
                        | Op::AtomicRmw(..)
                        | Op::AtomicCas(..)
                        | Op::Fence
                        | Op::Call { .. }
                        | Op::TryCall { .. }
                ) {
                    writes[b_idx].push(inst);
                }
            }
        }
        Self {
            cfg,
            prov,
            rank: Cfg::ranks(&cfg.rpo, func.blocks.len()),
            writes,
            visit: vec![0; func.blocks.len()],
            epoch: 0,
        }
    }

    /// Computes surviving `KnownLoads` and live `(base, root)` keys at entry to `block`.
    pub(super) fn entry_known(
        &mut self,
        func: &Function,
        subst: &[Value],
        known_out: &[KnownLoads],
        block_bases: &[Vec<(MemBase, MemRoot)>],
        block: Block,
    ) -> (KnownLoads, Vec<(MemBase, MemRoot)>) {
        if Some(block) == self.cfg.entry {
            return (KnownLoads::new(), Vec::new());
        }
        if let Some(pred) = self.cfg.single_pred(block) {
            return (known_out[pred.index()].clone(), block_bases[pred.index()].clone());
        }
        let block_rank = self.rank[block.index()];
        let preds = self.cfg.preds_of(block);
        let mut forward = true;
        let mut reach_count = 0_usize;
        let mut min_pred = None;
        for &pred in preds {
            if !self.cfg.reachable[pred.index()] {
                continue;
            }
            reach_count += 1;
            if self.rank[pred.index()] >= block_rank {
                forward = false;
                break;
            }
            if known_out[pred.index()].is_empty() {
                return (KnownLoads::new(), Vec::new());
            }
            if min_pred.is_none_or(|p: Block| block_bases[pred.index()].len() < block_bases[p.index()].len()) {
                min_pred = Some(pred);
            }
        }
        if forward {
            let Some(p0) = min_pred else { return (KnownLoads::new(), Vec::new()) };
            if reach_count == 1 {
                return (known_out[p0.index()].clone(), block_bases[p0.index()].clone());
            }
            let mut out = KnownLoads::new();
            let mut out_bases = Vec::new();
            for &(base, root) in &block_bases[p0.index()] {
                let mut kept_base = false;
                for &(off, ty, val) in known_out[p0.index()].entries_for(base) {
                    let all_match = preds
                        .iter()
                        .copied()
                        .filter(|&p| self.cfg.reachable[p.index()] && p != p0)
                        .all(|p| known_out[p.index()].get(base, off, ty) == Some(val));
                    if all_match {
                        out.remember(MemLoc::new(root, base, off, ty.bytes()), ty, val);
                        kept_base = true;
                    }
                }
                if kept_base {
                    out_bases.push((base, root));
                }
            }
            return (out, out_bases);
        }
        let Some(dom) = self.cfg.idom[block.index()] else {
            return (KnownLoads::new(), Vec::new());
        };
        let mut known = known_out[dom.index()].clone();
        if known.is_empty() {
            return (known, Vec::new());
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visit.fill(0);
            self.epoch = 1;
        }
        let epoch = self.epoch;
        let mut work = Vec::new();
        let mut walked = 0_usize;
        for &pred in preds {
            if self.cfg.reachable[pred.index()] && pred != dom && self.visit[pred.index()] != epoch {
                self.visit[pred.index()] = epoch;
                walked += 1;
                if walked > MAX_MERGE_WALK {
                    return (KnownLoads::new(), Vec::new());
                }
                work.push(pred);
            }
        }
        let dom_bases = &block_bases[dom.index()];
        while let Some(m) = work.pop() {
            for &w in &self.writes[m.index()] {
                let raw_op = &func.insts[w.index()].op;
                let canon = canonical_write(raw_op, subst);
                invalidate_preserving(func, &mut known, dom_bases, canon.as_ref().unwrap_or(raw_op), self.prov);
                if known.is_empty() {
                    return (known, Vec::new());
                }
            }
            for &pred in self.cfg.preds_of(m) {
                if self.cfg.reachable[pred.index()] && pred != dom && self.visit[pred.index()] != epoch {
                    self.visit[pred.index()] = epoch;
                    walked += 1;
                    if walked > MAX_MERGE_WALK {
                        return (KnownLoads::new(), Vec::new());
                    }
                    work.push(pred);
                }
            }
        }
        let out_bases = block_bases[dom.index()]
            .iter()
            .copied()
            .filter(|&(base, _)| !known.entries_for(base).is_empty())
            .collect();
        (known, out_bases)
    }
}
