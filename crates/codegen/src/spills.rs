//! The stack slots whose values registers still hold after a spill or reload, shared by the backends' reload caches.

use crate::ir::{Block, Function, Op};

/// Each block's predecessor when it has exactly one and that predecessor's body is never copied into another block's tail.
pub(crate) fn sole_preds(func: &Function, copied: &[bool]) -> Vec<Option<Block>> {
    let mut cloned = vec![false; func.blocks.len()];
    for data in &func.blocks {
        if let Some(&last) = data.insts.last()
            && copied.get(last.index()) == Some(&true)
            && let Op::Jump(head, _) = &func.insts[last.index()].op
        {
            cloned[head.index()] = true;
        }
    }
    let mut preds: Vec<(u32, Option<Block>)> = vec![(0, None); func.blocks.len()];
    for at in 0..func.blocks.len() {
        let from = Block(at as u32);
        for next in func.successors(from) {
            let entry = &mut preds[next.index()];
            entry.0 += 1;
            entry.1 = Some(from);
        }
    }
    preds.into_iter().map(|(count, from)| from.filter(|from| count == 1 && !cloned[from.index()])).collect()
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
