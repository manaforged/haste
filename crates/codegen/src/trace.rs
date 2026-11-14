//! Zero-overhead-when-disabled IR pass tracing (`HASTE_TRACE`) and missed-optimization counters (`HASTE_MISSED`).

use std::cell::RefCell;
use std::fmt::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::ir::{Domain, Function};
use crate::schedule::Pass;

struct Config {
    trace: Option<String>,
    trace_ir: bool,
    missed: bool,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

fn config() -> &'static Config {
    CONFIG.get_or_init(|| {
        let trace = std::env::var("HASTE_TRACE").ok().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty() && s != "0");
        let trace_ir = std::env::var("HASTE_TRACE_IR").ok().is_some_and(|s| !s.is_empty() && s != "0");
        let missed = std::env::var("HASTE_MISSED").ok().is_some_and(|s| !s.is_empty() && s != "0");
        Config { trace, trace_ir, missed }
    })
}

struct Counters {
    functions: AtomicUsize,
    insts_before: AtomicUsize,
    insts_after: AtomicUsize,
    slots_after_promote: AtomicUsize,
    slots_after_schedule: AtomicUsize,
    pass_runs: [AtomicUsize; Pass::COUNT],
    pass_changes: [AtomicUsize; Pass::COUNT],
    pass_nanos: [AtomicU64; Pass::COUNT],
}

impl Counters {
    const fn new() -> Self {
        const ZERO: AtomicUsize = AtomicUsize::new(0);
        const ZERO_NANOS: AtomicU64 = AtomicU64::new(0);
        Self {
            functions: ZERO,
            insts_before: ZERO,
            insts_after: ZERO,
            slots_after_promote: ZERO,
            slots_after_schedule: ZERO,
            pass_runs: [ZERO; Pass::COUNT],
            pass_changes: [ZERO; Pass::COUNT],
            pass_nanos: [ZERO_NANOS; Pass::COUNT],
        }
    }
}

static COUNTERS: Counters = Counters::new();

thread_local! {
    static LABELS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct LabelGuard;

impl Drop for LabelGuard {
    fn drop(&mut self) {
        LABELS.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// Returns whether pass tracing (`HASTE_TRACE`) or missed-optimization counting (`HASTE_MISSED`) is enabled.
#[inline]
pub fn enabled() -> bool {
    let cfg = config();
    cfg.trace.is_some() || cfg.missed
}

/// Runs `f` with `label` pushed onto the thread-local trace label stack.
pub fn with_label<R>(label: &str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    LABELS.with(|stack| stack.borrow_mut().push(label.to_owned()));
    let _guard = LabelGuard;
    f()
}

fn current_label() -> Option<String> {
    LABELS.with(|stack| stack.borrow().last().cloned())
}

fn matches_glob(pattern: &str, text: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else { return true };
    if !pattern.contains('*') {
        return text.contains(pattern);
    }
    let Some(mut rest) = text.strip_prefix(first) else { return false };
    let trailing_wild = pattern.ends_with('*');
    let collected: Vec<&str> = parts.collect();
    for (idx, part) in collected.iter().enumerate() {
        let is_last = idx + 1 == collected.len();
        if is_last && !trailing_wild {
            return rest.ends_with(part);
        }
        let Some(pos) = rest.find(part) else { return false };
        rest = &rest[pos + part.len()..];
    }
    true
}

fn should_trace(label: Option<&str>) -> bool {
    let Some(pat) = config().trace.as_deref() else { return false };
    if matches!(pat, "1" | "*" | "all") {
        return true;
    }
    label.is_some_and(|l| matches_glob(pat, l))
}

fn format_domain(d: Domain) -> &'static str {
    match d.0 {
        0 => "none",
        1 => "values",
        2 => "cfg",
        3 => "cfg|values",
        4 => "mem",
        5 => "mem|values",
        6 => "cfg|mem",
        _ => "cfg|mem|values",
    }
}

pub(crate) fn on_compile_start(func: &Function) {
    if config().missed {
        COUNTERS.functions.fetch_add(1, Ordering::Relaxed);
        COUNTERS.insts_before.fetch_add(func.size(), Ordering::Relaxed);
    }
}

pub(crate) fn on_compile_end(func: &Function) {
    if config().missed {
        COUNTERS.insts_after.fetch_add(func.size(), Ordering::Relaxed);
        COUNTERS.slots_after_schedule.fetch_add(func.slots.len(), Ordering::Relaxed);
    }
    let label = current_label();
    if should_trace(label.as_deref()) {
        let name = label.as_deref().unwrap_or("<anon>");
        eprintln!(
            "[haste-trace] fn={name} final blocks={} insts={} slots={}",
            func.blocks.len(),
            func.size(),
            func.slots.len()
        );
        if config().trace_ir {
            eprintln!("{func}");
        }
    }
}

pub(crate) fn after_pass(pass: Pass, before_blocks: usize, before_insts: usize, changed: Domain, elapsed: Duration, func: &Function) {
    let cfg = config();
    if cfg.missed {
        let idx = pass as usize;
        COUNTERS.pass_runs[idx].fetch_add(1, Ordering::Relaxed);
        COUNTERS.pass_nanos[idx].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        if !changed.is_empty() {
            COUNTERS.pass_changes[idx].fetch_add(1, Ordering::Relaxed);
        }
        if pass == Pass::Promote {
            COUNTERS.slots_after_promote.fetch_add(func.slots.len(), Ordering::Relaxed);
        }
    }
    let label = current_label();
    if should_trace(label.as_deref()) && (!changed.is_empty() || cfg.trace.as_deref() == Some("all")) {
        let name = label.as_deref().unwrap_or("<anon>");
        eprintln!(
            "[haste-trace] fn={name} pass={} changed={} blocks={before_blocks}->{} insts={before_insts}->{} slots={}",
            pass.name(),
            format_domain(changed),
            func.blocks.len(),
            func.size(),
            func.slots.len()
        );
    }
}

/// Formats aggregated per-pass run, change and wall-time totals (slowest first) and stack-slot counters collected while `HASTE_MISSED=1`.
pub fn missed_report() -> String {
    let mut out = String::new();
    if !config().missed {
        return out;
    }
    let fns = COUNTERS.functions.load(Ordering::Relaxed);
    let ib = COUNTERS.insts_before.load(Ordering::Relaxed);
    let ia = COUNTERS.insts_after.load(Ordering::Relaxed);
    let sp = COUNTERS.slots_after_promote.load(Ordering::Relaxed);
    let sf = COUNTERS.slots_after_schedule.load(Ordering::Relaxed);
    let _ = writeln!(out, "functions={fns} insts={ib}->{ia} slots_after_promote={sp} slots_after_schedule={sf}");
    let mut rows: Vec<(&str, usize, usize, u64)> = Pass::ALL
        .iter()
        .map(|&pass| {
            let idx = pass as usize;
            let runs = COUNTERS.pass_runs[idx].load(Ordering::Relaxed);
            let changes = COUNTERS.pass_changes[idx].load(Ordering::Relaxed);
            (pass.name(), runs, changes, COUNTERS.pass_nanos[idx].load(Ordering::Relaxed))
        })
        .filter(|row| row.1 > 0)
        .collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.3));
    let total = rows.iter().fold(("total", 0, 0, 0), |sum, row| (sum.0, sum.1 + row.1, sum.2 + row.2, sum.3 + row.3));
    for (name, runs, changes, nanos) in rows.into_iter().chain([total]) {
        let _ = writeln!(out, "  pass={name:<8} runs={runs} changes={changes} ms={:.3}", nanos as f64 / 1e6);
    }
    out
}
