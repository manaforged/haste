//! The stack slots whose values registers still hold after a spill or reload, shared by the backends' reload caches.

use crate::ir::{Block, Function, Op, Type};

/// Exact incoming edge count for each block in `order`, or `u32::MAX` when spill-cache carry is disabled for that block.
pub(crate) fn pred_totals(func: &Function, order: &[Block], copied: &[bool]) -> Vec<u32> {
    let n = func.blocks.len();
    let mut cloned = vec![false; n];
    for &from in order {
        if let Some(&last) = func.blocks[from.index()].insts.last()
            && copied.get(last.index()) == Some(&true)
            && let Op::Jump(head, _) = &func.insts[last.index()].op
        {
            cloned[head.index()] = true;
        }
    }
    let mut totals = vec![0_u32; n];
    if let Some(entry) = func.entry {
        totals[entry.index()] = u32::MAX;
    }
    for &from in order {
        let fi = from.index();
        if cloned[fi] {
            totals[fi] = u32::MAX;
        }
        let disable_succs = cloned[fi]
            || match func.last(from) {
                Some(Op::BrTable(..) | Op::TryCall { .. }) => true,
                Some(Op::Brif(_, then, _, otherwise, _)) => func.jump_target(*then) == func.jump_target(*otherwise),
                _ => false,
            };
        for next in func.successors(from) {
            let target = func.jump_target(next);
            let ti = target.index();
            if disable_succs || target != next {
                totals[ti] = u32::MAX;
            } else if totals[ti] != u32::MAX {
                totals[ti] += 1;
            }
        }
    }
    totals
}

/// Up to four slot displacements and the registers that hold them, newest first.
#[derive(Clone, Copy)]
pub(crate) struct SpillCache<R>([Option<(i32, R)>; 4]);

impl<R: Copy + PartialEq> SpillCache<R> {
    /// An empty cache.
    pub(crate) const fn new() -> Self {
        Self([None; 4])
    }

    /// Forgets every entry.
    pub(crate) fn clear(&mut self) {
        self.0 = [None; 4];
    }

    /// Keeps only `(disp, reg)` entries present in both `self` and `other`.
    pub(crate) fn intersect(&mut self, other: &Self) {
        for entry in &mut self.0 {
            if let Some(pair) = *entry
                && !other.0.contains(&Some(pair))
            {
                *entry = None;
            }
        }
    }

    /// Forgets what `reg` held.
    pub(crate) fn clobber(&mut self, reg: R) {
        for entry in &mut self.0 {
            if entry.is_some_and(|(_, r)| r == reg) {
                *entry = None;
            }
        }
    }

    /// Forgets every register that held the slot at `disp`.
    pub(crate) fn invalidate(&mut self, disp: i32) {
        for entry in &mut self.0 {
            if entry.is_some_and(|(d, _)| d == disp) {
                *entry = None;
            }
        }
    }

    /// Records that `reg` now holds the slot at `disp`, as the newest entry; the caller first forgets what `reg` held.
    pub(crate) fn record(&mut self, disp: i32, reg: R) {
        let mut out = [Some((disp, reg)), None, None, None];
        let mut i = 1;
        for entry in self.0.into_iter().flatten() {
            if entry != (disp, reg) && i < 4 {
                out[i] = Some(entry);
                i += 1;
            }
        }
        self.0 = out;
    }

    /// The registers that hold the slot at `disp`, newest first.
    pub(crate) fn holders(&self, disp: i32) -> impl Iterator<Item = R> + '_ {
        self.0.iter().flatten().filter(move |&&(d, _)| d == disp).map(|&(_, r)| r)
    }
}

/// Per-block integer and float spill reload cache with forward-join predecessor intersection.
pub(crate) struct SpillState<R, F> {
    pub(crate) cache: SpillCache<R>,
    pub(crate) fspill: Option<(i32, F, Type)>,
    pred_total: Vec<u32>,
    pred_seen: Vec<u32>,
    pred_cache: Vec<SpillCache<R>>,
}

impl<R: Copy + PartialEq, F: Copy + PartialEq> SpillState<R, F> {
    pub(crate) fn new(pred_total: Vec<u32>) -> Self {
        let n = pred_total.len();
        Self {
            cache: SpillCache::new(),
            fspill: None,
            pred_total,
            pred_seen: vec![0; n],
            pred_cache: vec![SpillCache::new(); n],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.cache.clear();
        self.fspill = None;
    }

    pub(crate) fn clear_fspill(&mut self) {
        self.fspill = None;
    }

    pub(crate) fn clobber_reg(&mut self, reg: R) {
        self.cache.clobber(reg);
    }

    pub(crate) fn clobber_freg(&mut self, reg: F) {
        if self.fspill.is_some_and(|(_, r, _)| r == reg) {
            self.fspill = None;
        }
    }

    pub(crate) fn invalidate_disp(&mut self, disp: i32) {
        self.cache.invalidate(disp);
        if self.fspill.is_some_and(|(d, _, _)| d == disp) {
            self.fspill = None;
        }
    }

    pub(crate) fn record_edge(&mut self, target: Block, cache: SpillCache<R>) {
        let bi = target.index();
        let total = self.pred_total.get(bi).copied().unwrap_or(u32::MAX);
        if total == u32::MAX || total == 0 {
            return;
        }
        if self.pred_seen[bi] == 0 {
            self.pred_cache[bi] = cache;
        } else {
            self.pred_cache[bi].intersect(&cache);
        }
        self.pred_seen[bi] += 1;
    }

    pub(crate) fn resume(&mut self, block: Block) {
        let bi = block.index();
        if let Some(&total) = self.pred_total.get(bi)
            && total != u32::MAX
            && total > 0
            && self.pred_seen[bi] == total
        {
            self.cache = self.pred_cache[bi];
        }
        if let Some(slot) = self.pred_total.get_mut(bi) {
            *slot = u32::MAX;
        }
    }

    pub(crate) fn record(&mut self, disp: i32, reg: R) {
        self.cache.clobber(reg);
        self.cache.record(disp, reg);
    }

    pub(crate) fn write_fspill(&mut self, disp: i32, reg: F, ty: Type) {
        self.invalidate_disp(disp);
        self.fspill = Some((disp, reg, ty));
    }

    pub(crate) fn cached_for(&self, disp: i32, preferred: R, deprioritize: Option<R>) -> Option<R> {
        let mut fallback = None;
        for r in self.cache.holders(disp) {
            if r == preferred {
                return Some(r);
            }
            if deprioritize != Some(r) || fallback.is_none() {
                fallback = Some(r);
            }
        }
        fallback
    }

    pub(crate) fn cached_fspill(&self, disp: i32, ty: Type) -> Option<F> {
        self.fspill.and_then(|(d, r, t)| (d == disp && t == ty).then_some(r))
    }
}
