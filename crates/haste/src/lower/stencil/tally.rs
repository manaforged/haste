use std::sync::atomic::{AtomicUsize, Ordering};

use haste_codegen::Function;

#[derive(Clone, Copy)]
pub(in crate::lower) enum Tally {
    Hit,
    Miss,
    Built,
    Raced,
    OpenCycle,
    Deferred,
    Devirt,
    Trim,
    Flush,
    Retry,
    Compact,
    Disk,
    Stale,
}

const NAMES: [&str; 13] = [
    "template_hit",
    "template_miss",
    "template_simplify",
    "template_raced",
    "unstored_cycle",
    "unstored_deferred",
    "devirt_simplify",
    "trim_partial",
    "trim_full",
    "compact_retry",
    "compact_fit",
    "template_disk",
    "template_stale",
];
static COUNTS: [AtomicUsize; NAMES.len()] = [const { AtomicUsize::new(0) }; NAMES.len()];

pub(super) fn tally(kind: Tally) {
    COUNTS[kind as usize].fetch_add(1, Ordering::Relaxed);
}

pub(in crate::lower) fn simplify_tallied(func: &mut Function, kind: Tally) {
    tally(kind);
    haste_codegen::simplify(func, &crate::host_target());
}

pub(super) fn report() -> String {
    if haste_codegen::missed_report().is_empty() {
        return String::new();
    }
    let mut out = String::from("inline:");
    for (name, count) in NAMES.iter().zip(&COUNTS) {
        out.push_str(&format!(" {name}={}", count.load(Ordering::Relaxed)));
    }
    out.push('\n');
    out
}
