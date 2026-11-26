//! Relaxed counters for the on-disk stencil cache, reported on one line when `HASTE_MISSED` is set.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// One stencil cache event or accumulated quantity.
#[derive(Clone, Copy)]
pub(crate) enum Tally {
    Lookup,
    Hit,
    HitLowered,
    MissUnportable,
    MissUnkeyed,
    MissAbsent,
    MissPlan,
    MissOther,
    FreezeDeferred,
    FreezeSkipped,
    FreezePlanless,
    Offered,
    TrimStencils,
    TrimDropped,
    TrimBaseBytes,
    TrimNeedBytes,
    TrimKeptBytes,
    FootprintUs,
    LookupUs,
    LowerUs,
    MetaLookup,
    MetaHit,
    MetaMiss,
    MetaStored,
    MetaMismatch,
}

const TALLY_NAMES: [&str; 25] = [
    "lookup",
    "hit",
    "hit_lowered",
    "miss_unportable",
    "miss_unkeyed",
    "miss_absent",
    "miss_plan",
    "miss_other",
    "freeze_deferred",
    "freeze_skipped",
    "freeze_planless",
    "save_offered",
    "trim_stencils",
    "trim_dropped",
    "trim_base_bytes",
    "trim_need_bytes",
    "trim_kept_bytes",
    "footprint_us",
    "lookup_us",
    "miss_lower_us",
    "meta_lookup",
    "meta_hit",
    "meta_miss",
    "meta_stored",
    "meta_mismatch",
];
static TALLIES: [AtomicUsize; TALLY_NAMES.len()] =
    [const { AtomicUsize::new(0) }; TALLY_NAMES.len()];

/// Counts one `kind` event.
pub(crate) fn tally(kind: Tally) {
    add(kind, 1);
}

/// Adds `amount` to the `kind` counter.
pub(crate) fn add(kind: Tally, amount: usize) {
    TALLIES[kind as usize].fetch_add(amount, Ordering::Relaxed);
}

/// Adds the microseconds elapsed since `started` to the `kind` counter.
pub(crate) fn clock(kind: Tally, started: Instant) {
    add(kind, started.elapsed().as_micros() as usize);
}

/// Formats the stencil cache counters as one line, or nothing unless `HASTE_MISSED` is set.
pub(crate) fn report() -> String {
    if haste_codegen::missed_report().is_empty() {
        return String::new();
    }
    let mut out = String::from("stencil:");
    for (name, count) in TALLY_NAMES.iter().zip(&TALLIES) {
        out.push_str(&format!(" {name}={}", count.load(Ordering::Relaxed)));
    }
    out.push('\n');
    out
}
