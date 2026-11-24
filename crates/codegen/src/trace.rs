//! Zero-overhead-when-disabled IR pass tracing (`HASTE_TRACE`, with `HASTE_TRACE_IR=1` for final IR or `=passes` for IR after each changing pass) and missed-optimization counters (`HASTE_MISSED`).

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
    trace_passes: bool,
    missed: bool,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

fn config() -> &'static Config {
    CONFIG.get_or_init(|| {
        let trace = std::env::var("HASTE_TRACE")
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty() && s != "0");
        let trace_ir = std::env::var("HASTE_TRACE_IR")
            .ok()
            .is_some_and(|s| !s.is_empty() && s != "0");
        let trace_passes = std::env::var("HASTE_TRACE_IR")
            .ok()
            .is_some_and(|s| s == "passes");
        let missed = std::env::var("HASTE_MISSED")
            .ok()
            .is_some_and(|s| !s.is_empty() && s != "0");
        Config {
            trace,
            trace_ir,
            trace_passes,
            missed,
        }
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
    let Some(first) = parts.next() else {
        return true;
    };
    if !pattern.contains('*') {
        return text.contains(pattern);
    }
    let Some(mut rest) = text.strip_prefix(first) else {
        return false;
    };
    let trailing_wild = pattern.ends_with('*');
    let collected: Vec<&str> = parts.collect();
    for (idx, part) in collected.iter().enumerate() {
        let is_last = idx + 1 == collected.len();
        if is_last && !trailing_wild {
            return rest.ends_with(part);
        }
        let Some(pos) = rest.find(part) else {
            return false;
        };
        rest = &rest[pos + part.len()..];
    }
    true
}

fn should_trace(label: Option<&str>) -> bool {
    let Some(pat) = config().trace.as_deref() else {
        return false;
    };
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
        COUNTERS
            .insts_before
            .fetch_add(func.size(), Ordering::Relaxed);
    }
}

/// Reports each integer register candidate of a traced function that register allocation left in a stack slot, and each that holds a register across a call: its uses, its live length, and how many calls its live range crosses.
pub(crate) fn on_alloc(
    func: &Function,
    live: &crate::live::Live,
    values: impl Iterator<Item = (usize, bool)>,
) {
    let label = current_label();
    if !should_trace(label.as_deref()) {
        return;
    }
    let mut uses = vec![0_u32; func.types.len()];
    func.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .for_each(|inst| {
            func.insts[inst.index()]
                .op
                .for_operands(|value| uses[value.index()] += 1)
        });
    let mut lines = String::new();
    let (mut count, mut crossing) = (0, 0);
    for (value, in_slot) in values {
        let ranges = live.ranges_of(value);
        let length: u32 = ranges.iter().map(|range| range.end - range.start).sum();
        let calls: usize = ranges
            .iter()
            .map(|range| {
                live.calls
                    .partition_point(|&at| at < range.end)
                    .saturating_sub(live.calls.partition_point(|&at| at <= range.start))
            })
            .sum();
        if !in_slot && calls == 0 {
            continue;
        }
        count += usize::from(in_slot);
        crossing += usize::from(in_slot && calls > 0);
        let _ = writeln!(
            lines,
            "  {} v{value} uses={} length={length} calls_crossed={calls}",
            if in_slot { "slot" } else { "reg " },
            uses[value]
        );
    }
    eprintln!(
        "[haste-trace] fn={} spilled={count} crossing_calls={crossing}\n{lines}",
        label.as_deref().unwrap_or("<anon>")
    );
}

/// Prints, for the traced function's hottest loops, the registers held by values live into the header but unused in the loop, and the spilled values the loop reads with their weighted uses there.
pub(crate) fn on_loops(
    func: &Function,
    live: &crate::live::Live,
    (loops, blocks): (&crate::cfg::Loops, &[u64]),
    ints: &[bool],
    in_slot: impl Fn(usize) -> bool,
) {
    let label = current_label();
    if !should_trace(label.as_deref()) {
        return;
    }
    let mut lines = String::new();
    let mut order: Vec<&crate::cfg::Loop> = loops.list.iter().collect();
    order
        .sort_by_key(|each| std::cmp::Reverse(blocks.get(each.head.index()).copied().unwrap_or(0)));
    for each in order.into_iter().take(30) {
        let mut used = vec![0_u64; func.types.len()];
        let mut defined = vec![false; func.types.len()];
        for (at, data) in func
            .blocks
            .iter()
            .enumerate()
            .filter(|&(at, _)| each.body.get(at).copied().unwrap_or(false))
        {
            for (slot, p) in data.params.iter().enumerate() {
                defined[p.index()] = at != each.head.index()
                    || !each
                        .latches
                        .iter()
                        .all(|&latch| latch_arg(func, latch, each.head, slot) == Some(*p));
            }
            for inst in &data.insts {
                let inst = &func.insts[inst.index()];
                inst.op
                    .for_operands(|v| used[v.index()] += blocks.get(at).copied().unwrap_or(1));
                if let Some(r) = inst.result {
                    defined[r.index()] = true;
                }
            }
        }
        let head = live.spans.get(each.head.index()).map_or(0, |span| span.0);
        let through = |v: usize| {
            live.ranges_of(v)
                .iter()
                .any(|r| r.start < head && r.end > head)
        };
        let idle = (0..func.types.len())
            .filter(|&v| ints[v] && !in_slot(v) && used[v] == 0 && through(v))
            .count();
        let any_slot = (0..func.types.len())
            .filter(|&v| in_slot(v) && used[v] > 0)
            .count();
        let mut reads: Vec<(u64, usize)> = (0..func.types.len())
            .filter(|&v| ints[v] && in_slot(v) && !defined[v] && used[v] > 0)
            .map(|v| (used[v], v))
            .collect();
        reads.sort_unstable_by(|a, b| b.cmp(a));
        let weight: u64 = reads.iter().map(|r| r.0).sum();
        let _ = writeln!(
            lines,
            "  loop b{} depth={} weight={} idle_regs={idle} any_slot={any_slot} slot_reads={} slot_weight={weight} top={:?}",
            each.head.index(),
            each.depth,
            blocks.get(each.head.index()).copied().unwrap_or(0),
            reads.len(),
            &reads[..reads.len().min(6)]
        );
    }
    eprintln!(
        "[haste-trace] fn={} loops\n{lines}",
        label.as_deref().unwrap_or("<anon>")
    );
}

fn latch_arg(
    func: &Function,
    latch: crate::ir::Block,
    head: crate::ir::Block,
    slot: usize,
) -> Option<crate::ir::Value> {
    match func.last(latch)? {
        crate::ir::Op::Jump(target, args) if *target == head => args.get(slot).copied(),
        crate::ir::Op::Brif(_, then, then_args, _, _) if *then == head => {
            then_args.get(slot).copied()
        }
        crate::ir::Op::Brif(_, _, _, otherwise, else_args) if *otherwise == head => {
            else_args.get(slot).copied()
        }
        _ => None,
    }
}

/// Prints each loop-header split of a spilled value in the traced function and whether its loop copy got a register.
pub(crate) fn on_split(
    made: &[(crate::ir::Value, crate::ir::Value, crate::ir::Block)],
    kept: &[(crate::ir::Block, crate::ir::Value)],
) {
    let label = current_label();
    if !should_trace(label.as_deref()) {
        return;
    }
    let mut lines = String::new();
    for &(_, origin, head) in made {
        let _ = writeln!(
            lines,
            "  split v{} at b{} {}",
            origin.index(),
            head.index(),
            if kept.contains(&(head, origin)) {
                "kept"
            } else {
                "dropped"
            }
        );
    }
    eprintln!(
        "[haste-trace] fn={} splits made={} kept={}\n{lines}",
        label.as_deref().unwrap_or("<anon>"),
        made.len(),
        kept.len()
    );
}

/// Prints the frame-relative displacement bands of value homes and of addressed stack slots, and the frame size, for the traced function.
pub(crate) fn on_frame(func: &Function, (values, slots): (&[i32], &[i32]), size: u32) {
    let label = current_label();
    if !should_trace(label.as_deref()) {
        return;
    }
    let band = |spans: &mut dyn Iterator<Item = (i32, u32)>| {
        spans
            .filter(|&(disp, _)| disp != 0)
            .fold(None, |acc: Option<(i32, i32)>, (disp, bytes)| {
                let (lo, hi) = (disp, disp + bytes.max(1) as i32);
                Some(acc.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))))
            })
    };
    let homes = band(
        &mut values
            .iter()
            .zip(&func.types)
            .map(|(&disp, ty)| (disp, ty.bytes())),
    );
    let addressed = band(
        &mut slots
            .iter()
            .zip(&func.slots)
            .map(|(&disp, slot)| (disp, slot.size)),
    );
    let show = |span: Option<(i32, i32)>| {
        span.map_or_else(|| "-".to_owned(), |(lo, hi)| format!("{lo}..{hi}"))
    };
    eprintln!(
        "[haste-trace] fn={} frame size={size} homes={} slots={}",
        label.as_deref().unwrap_or("<anon>"),
        show(homes),
        show(addressed)
    );
}

pub(crate) fn on_compile_end(func: &Function) {
    if config().missed {
        COUNTERS
            .insts_after
            .fetch_add(func.size(), Ordering::Relaxed);
        COUNTERS
            .slots_after_schedule
            .fetch_add(func.slots.len(), Ordering::Relaxed);
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

pub(crate) fn after_pass(
    pass: Pass,
    before_blocks: usize,
    before_insts: usize,
    changed: Domain,
    elapsed: Duration,
    func: &Function,
) {
    let cfg = config();
    if cfg.missed {
        let idx = pass as usize;
        COUNTERS.pass_runs[idx].fetch_add(1, Ordering::Relaxed);
        COUNTERS.pass_nanos[idx].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        if !changed.is_empty() {
            COUNTERS.pass_changes[idx].fetch_add(1, Ordering::Relaxed);
        }
        if pass == Pass::Promote {
            COUNTERS
                .slots_after_promote
                .fetch_add(func.slots.len(), Ordering::Relaxed);
        }
    }
    let label = current_label();
    if should_trace(label.as_deref())
        && (!changed.is_empty() || cfg.trace.as_deref() == Some("all"))
    {
        let name = label.as_deref().unwrap_or("<anon>");
        eprintln!(
            "[haste-trace] fn={name} pass={} changed={} blocks={before_blocks}->{} insts={before_insts}->{} slots={}",
            pass.name(),
            format_domain(changed),
            func.blocks.len(),
            func.size(),
            func.slots.len()
        );
        if cfg.trace_passes {
            eprintln!("{func}");
        }
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
    let _ = writeln!(
        out,
        "functions={fns} insts={ib}->{ia} slots_after_promote={sp} slots_after_schedule={sf}"
    );
    let mut rows: Vec<(&str, usize, usize, u64)> = Pass::ALL
        .iter()
        .map(|&pass| {
            let idx = pass as usize;
            let runs = COUNTERS.pass_runs[idx].load(Ordering::Relaxed);
            let changes = COUNTERS.pass_changes[idx].load(Ordering::Relaxed);
            (
                pass.name(),
                runs,
                changes,
                COUNTERS.pass_nanos[idx].load(Ordering::Relaxed),
            )
        })
        .filter(|row| row.1 > 0)
        .collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.3));
    let total = rows.iter().fold(("total", 0, 0, 0), |sum, row| {
        (sum.0, sum.1 + row.1, sum.2 + row.2, sum.3 + row.3)
    });
    for (name, runs, changes, nanos) in rows.into_iter().chain([total]) {
        let _ = writeln!(
            out,
            "  pass={name:<8} runs={runs} changes={changes} ms={:.3}",
            nanos as f64 / 1e6
        );
    }
    out
}

/// Prints the traced function's register-to-register edge copies, weighted by block weight: forced when argument and parameter overlap, blocked when another value holds the parameter's register during the argument, open otherwise.
pub(crate) fn on_edge_copies<R: PartialEq + Copy>(
    func: &Function,
    live: &crate::live::Live,
    blocks: &[u64],
    reg: impl Fn(usize) -> Option<R>,
) {
    let label = current_label();
    if !should_trace(label.as_deref()) {
        return;
    }
    let (mut forced, mut blocked, mut open, mut lines) = (0_u64, 0_u64, 0_u64, String::new());
    crate::alloc::walk::for_edge_copies(func, |src, target, param, arg| {
        let (Some(p), Some(a)) = (reg(param.index()), reg(arg.index())) else {
            return;
        };
        if p == a {
            return;
        }
        let weight = blocks[src.index()];
        let arg_ranges = live.ranges_of(arg.index());
        if crate::live::overlaps(live.ranges_of(param.index()), arg_ranges) {
            forced += weight;
        } else if (0..func.types.len()).any(|other| {
            other != param.index()
                && other != arg.index()
                && reg(other) == Some(p)
                && crate::live::overlaps(live.ranges_of(other), arg_ranges)
        }) {
            blocked += weight;
        } else {
            open += weight;
            let _ = writeln!(
                lines,
                "  open b{}->b{} v{} <- v{} weight={weight}",
                src.index(),
                target.index(),
                param.index(),
                arg.index()
            );
        }
    });
    eprintln!(
        "[haste-trace] fn={} edgecopies forced={forced} blocked={blocked} open={open}\n{lines}",
        label.as_deref().unwrap_or("<anon>")
    );
}
