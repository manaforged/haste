//! How a wave of inference splits across cores, and the inference of a list of declared functions ahead of their first call.

use std::sync::atomic::AtomicUsize;

use super::{MAX_WORKER_BATCH, MIN_PARALLEL_WAVE, Reached, ReachedEntry, run_scoped};
use crate::sema::Context;

/// How many functions of a wave of `len` one worker takes at a time.
pub(super) fn wave_batch(len: usize) -> usize {
    (len / crate::jobs().max(1)).clamp(1, MAX_WORKER_BATCH)
}

/// How many workers infer a wave of `len` functions: one when the wave is too small to split.
pub(super) fn wave_workers(len: usize) -> usize {
    if len < MIN_PARALLEL_WAVE {
        return 1;
    }
    crate::jobs().max(1).min(len.div_ceil(wave_batch(len)))
}

/// Infers on every core the functions of `indices` that are declared without a body, keeps each body whose inference asked for nothing and deferred no constant evaluation, and returns the indexes it kept; the others stay declared, so `infer_declared` infers them on their first call.
pub fn infer_quiet(cx: &Context<'_>, reached: &mut Reached, indices: &[usize]) -> Vec<usize> {
    let wave: Vec<usize> = indices.iter().copied().filter(|&at| reached.is_untyped(at)).collect();
    if wave.is_empty() {
        return Vec::new();
    }
    let mut own = cx.fork();
    let forks = (1..wave_workers(wave.len())).map(|_| cx.fork()).collect();
    let next = AtomicUsize::new(0);
    let (mut items, joined) = run_scoped(&mut own, forks, reached, (&wave, &next, false));
    items.extend(joined.into_iter().flatten().flat_map(|(_, _, worker_items)| worker_items));
    items.sort_by_key(|item| item.pos);
    let mut kept = Vec::new();
    for item in items.into_iter().filter(|item| !item.asked && !item.deferred) {
        let at = wave[item.pos];
        reached.unlaid.extend(item.unlaid);
        reached.set_entry(at, ReachedEntry::Settled, item.typed);
        kept.push(at);
    }
    kept
}
