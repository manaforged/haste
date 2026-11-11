mod calls;
mod digest;
mod footprint;
mod json;

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::Workspace;
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;

use digest::Digest;
use footprint::{InstanceRecord, analyze_all};

/// Wall-clock and process-plus-children CPU durations in nanoseconds.
#[derive(Clone, Copy, Default)]
pub struct Span {
    pub wall_ns: u64,
    pub cpu_ns: u64,
}

/// A start instant for measuring wall-clock and process-plus-children CPU duration.
#[derive(Clone, Copy)]
pub struct Clock {
    wall: Instant,
    cpu_ns: u64,
}

impl Clock {
    /// Captures the current wall-clock and process-plus-children CPU time.
    pub fn now() -> Self {
        Self { wall: Instant::now(), cpu_ns: process_cpu_ns() }
    }

    /// The wall-clock and CPU duration elapsed since this clock was captured.
    pub fn elapsed(self) -> Span {
        Span { wall_ns: self.wall.elapsed().as_nanos() as u64, cpu_ns: process_cpu_ns().saturating_sub(self.cpu_ns) }
    }
}

/// Measured durations of the build stages preceding the equivalence-class census.
#[derive(Clone, Copy, Default)]
pub struct Timings {
    pub graph: Span,
    pub outline: Span,
    pub reach: Span,
    pub codegen: Span,
}

/// Inputs to the build equivalence-class, hazard, origin, and code-size census.
pub struct CensusInput<'a> {
    pub workspace: &'a Workspace,
    pub reached: &'a Reached,
    pub code_sizes: &'a [u32],
    pub compile_times: &'a [u64],
    pub timings: Timings,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Local = 0,
    Pinned = 1,
    Sysroot = 2,
}

#[derive(Default)]
struct HazardCauses {
    niche_layout_mismatch: usize,
    field_offset_mismatch: usize,
    assoc_abi_mismatch: usize,
    specialization_mismatch: usize,
    reflection_mismatch: usize,
    other_footprint_mismatch: usize,
}

#[derive(Default)]
struct Histogram {
    le_128: usize,
    le_256: usize,
    le_512: usize,
    le_1024: usize,
    gt_1024: usize,
}

#[derive(Default)]
struct CodeSavings {
    compiled_functions: usize,
    total_code_bytes: u64,
    unique_footprint_code_bytes: u64,
    unique_alias_code_bytes: u64,
    dict_stencil_classes: usize,
    projected_stenciled_bytes: u64,
}

#[derive(Default)]
struct OriginReport {
    crates: usize,
    loaded_modules: usize,
    files: usize,
    source_bytes: u64,
    items: usize,
    expansions: usize,
    inferred_exprs: u64,
    n_reached: usize,
    n_inst: usize,
    n_fn: usize,
    n_generic_inst: usize,
    n_generic_fn: usize,
    n_foreign: usize,
    n_failed: usize,
    n_footprint_no_inline: usize,
    n_footprint: usize,
    n_alias: usize,
    n_naive_shape: usize,
    n_naive_shape_hazards: usize,
    savings: CodeSavings,
    codegen_worker_ns: u64,
    outline_span: Span,
    reach_span: Span,
    codegen_span: Span,
}

struct Report {
    n_reached: usize,
    n_inst: usize,
    n_fn: usize,
    n_generic_inst: usize,
    n_generic_fn: usize,
    n_compiled_inst: usize,
    n_foreign: usize,
    n_failed: usize,
    n_naive_shape: usize,
    n_naive_shape_hazards: usize,
    hazard_causes: HazardCauses,
    n_footprint_no_inline: usize,
    n_footprint: usize,
    n_alias: usize,
    n_generic_footprint_no_inline: usize,
    n_generic_footprint: usize,
    n_generic_alias: usize,
    histogram: Histogram,
    savings: CodeSavings,
    timings: Timings,
    census_span: Span,
    origins: [OriginReport; 3],
}

#[derive(Default)]
struct NaiveGroup {
    origin: usize,
    k_set: FxHashSet<Digest>,
    niche_set: FxHashSet<Digest>,
    offset_set: FxHashSet<Digest>,
    assoc_set: FxHashSet<Digest>,
    spec_set: FxHashSet<Digest>,
    reflection_set: FxHashSet<Digest>,
}

const DICT_MIN_BYTES: u64 = 512;
const DICT_MIN_FANOUT: u64 = 8;

#[derive(Default)]
struct StencilClass {
    rep_bytes: u64,
    d_slots: u64,
    alias_map: FxHashMap<Digest, u64>,
}

/// Computes the equivalence-class, hazard, origin, and code-size census and writes deterministic JSON to `path`.
pub fn write_stats(cx: &mut Context<'_>, input: CensusInput<'_>, path: &Path) -> Result<()> {
    let clock = Clock::now();
    let records = analyze_all(cx, input.reached);
    let mut report = aggregate(cx, &input, &records);
    report.census_span = clock.elapsed();
    let text = json::render_json(&report);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    Ok(())
}

fn process_cpu_ns() -> u64 {
    let mut total = 0u64;
    for who in [libc::RUSAGE_SELF, libc::RUSAGE_CHILDREN] {
        // SAFETY: rusage is plain data where all-zero bytes are valid, and getrusage writes only into usage.
        let usage = unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(who, &mut usage) == 0 { Some(usage) } else { None }
        };
        if let Some(u) = usage {
            let secs = (u.ru_utime.tv_sec as u64).saturating_add(u.ru_stime.tv_sec as u64);
            let usecs = (u.ru_utime.tv_usec as u64).saturating_add(u.ru_stime.tv_usec as u64);
            total = total.saturating_add(secs.saturating_mul(1_000_000_000)).saturating_add(usecs.saturating_mul(1_000));
        }
    }
    total
}

fn crate_origins(cx: &Context<'_>, workspace: &Workspace) -> Vec<Origin> {
    (0..cx.outline.roots.len())
        .map(|krate| {
            let pkg = cx.outline.package(workspace, krate);
            if pkg.sysroot {
                Origin::Sysroot
            } else if pkg.local() {
                Origin::Local
            } else {
                Origin::Pinned
            }
        })
        .collect()
}

fn fill_outline_origins(cx: &Context<'_>, origins_of: &[Origin], origins: &mut [OriginReport; 3]) {
    for (krate, tree) in cx.outline.crates.iter().enumerate() {
        let bucket = &mut origins[origins_of[krate] as usize];
        bucket.crates += 1;
        bucket.loaded_modules += tree.modules.iter().filter(|m| m.loaded()).count();
        bucket.expansions += tree.expansions;
        for file in &tree.files {
            if !file.expanded {
                bucket.files += 1;
                bucket.source_bytes = bucket.source_bytes.saturating_add(file.source.text.len() as u64);
            }
            bucket.items += file.items.items.len();
        }
    }
}

#[derive(Default)]
struct OriginSets {
    fn_set: FxHashSet<ItemRef>,
    gen_fn_set: FxHashSet<ItemRef>,
    fp_no_inline_set: FxHashSet<(ItemRef, Digest)>,
    fp_set: FxHashSet<(ItemRef, Digest)>,
    alias_set: FxHashSet<(ItemRef, Digest, Digest)>,
    stencil_map: FxHashMap<(ItemRef, Digest), StencilClass>,
}

#[derive(Default)]
struct Accum {
    origins: [OriginReport; 3],
    origin_sets: [OriginSets; 3],
    fn_set: FxHashSet<ItemRef>,
    gen_fn_set: FxHashSet<ItemRef>,
    fp_no_inline_set: FxHashSet<(ItemRef, Digest)>,
    fp_set: FxHashSet<(ItemRef, Digest)>,
    alias_set: FxHashSet<(ItemRef, Digest, Digest)>,
    gen_fp_no_inline_set: FxHashSet<(ItemRef, Digest)>,
    gen_fp_set: FxHashSet<(ItemRef, Digest)>,
    gen_alias_set: FxHashSet<(ItemRef, Digest, Digest)>,
    naive_map: FxHashMap<(ItemRef, Digest), NaiveGroup>,
    stencil_map: FxHashMap<(ItemRef, Digest), StencilClass>,
    histogram: Histogram,
    n_inst: usize,
    n_generic_inst: usize,
    n_compiled_inst: usize,
    n_foreign: usize,
    n_failed: usize,
}

impl Accum {
    fn ingest(&mut self, orig: usize, item: ItemRef, rec: &InstanceRecord, (code_len, worker_ns): (u32, u64)) {
        self.origins[orig].n_reached += 1;
        if !rec.valid {
            self.n_failed += 1;
            self.origins[orig].n_failed += 1;
            return;
        }
        self.record_valid(orig, item, rec);
        if rec.generic {
            self.record_generic(orig, item, rec);
        }
        self.origins[orig].codegen_worker_ns = self.origins[orig].codegen_worker_ns.saturating_add(worker_ns);
        if code_len > 0 {
            record_code_size(&mut self.histogram, &mut self.stencil_map, &mut self.origin_sets[orig].stencil_map, (item, rec, code_len));
            self.origins[orig].savings.compiled_functions += 1;
            self.origins[orig].savings.total_code_bytes = self.origins[orig].savings.total_code_bytes.saturating_add(u64::from(code_len));
        }
    }

    fn record_valid(&mut self, orig: usize, item: ItemRef, rec: &InstanceRecord) {
        self.n_inst += 1;
        self.origins[orig].n_inst += 1;
        self.origins[orig].inferred_exprs = self.origins[orig].inferred_exprs.saturating_add(rec.expr_count as u64);
        self.fn_set.insert(item);
        self.origin_sets[orig].fn_set.insert(item);
        self.fp_no_inline_set.insert((item, rec.k_no_inline));
        self.fp_set.insert((item, rec.k_inline));
        self.alias_set.insert((item, rec.k_inline, rec.d_inline));
        self.origin_sets[orig].fp_no_inline_set.insert((item, rec.k_no_inline));
        self.origin_sets[orig].fp_set.insert((item, rec.k_inline));
        self.origin_sets[orig].alias_set.insert((item, rec.k_inline, rec.d_inline));
        if rec.foreign {
            self.n_foreign += 1;
            self.origins[orig].n_foreign += 1;
        } else {
            self.n_compiled_inst += 1;
        }
    }

    fn record_generic(&mut self, orig: usize, item: ItemRef, rec: &InstanceRecord) {
        self.n_generic_inst += 1;
        self.origins[orig].n_generic_inst += 1;
        self.gen_fn_set.insert(item);
        self.origin_sets[orig].gen_fn_set.insert(item);
        self.gen_fp_no_inline_set.insert((item, rec.k_no_inline));
        self.gen_fp_set.insert((item, rec.k_inline));
        self.gen_alias_set.insert((item, rec.k_inline, rec.d_inline));
        if !rec.foreign {
            record_naive_member(self.naive_map.entry((item, rec.outer_shape)).or_default(), orig, rec);
        }
    }
}

fn aggregate(cx: &Context<'_>, input: &CensusInput<'_>, records: &[InstanceRecord]) -> Report {
    let origins_of = crate_origins(cx, input.workspace);
    let mut acc = Accum::default();
    fill_outline_origins(cx, &origins_of, &mut acc.origins);
    for (index, ((item, _), _)) in input.reached.list.iter().enumerate() {
        let orig = origins_of[item.krate as usize] as usize;
        let code_len = input.code_sizes.get(index).copied().unwrap_or(0);
        let worker_ns = input.compile_times.get(index).copied().unwrap_or(0);
        acc.ingest(orig, *item, &records[index], (code_len, worker_ns));
    }
    let (n_naive_shape, n_naive_shape_hazards, hazard_causes) = summarize_hazards(&acc.naive_map, &mut acc.origins);
    let mut savings = finalize_savings(&acc.stencil_map);
    savings.compiled_functions = acc.origins.iter().map(|o| o.savings.compiled_functions).sum();
    savings.total_code_bytes = acc.origins.iter().map(|o| o.savings.total_code_bytes).sum();
    finalize_origins(&mut acc.origins, &acc.origin_sets, input.timings);
    Report {
        n_reached: input.reached.list.len(),
        n_inst: acc.n_inst,
        n_fn: acc.fn_set.len(),
        n_generic_inst: acc.n_generic_inst,
        n_generic_fn: acc.gen_fn_set.len(),
        n_compiled_inst: acc.n_compiled_inst,
        n_foreign: acc.n_foreign,
        n_failed: acc.n_failed,
        n_naive_shape,
        n_naive_shape_hazards,
        hazard_causes,
        n_footprint_no_inline: acc.fp_no_inline_set.len(),
        n_footprint: acc.fp_set.len(),
        n_alias: acc.alias_set.len(),
        n_generic_footprint_no_inline: acc.gen_fp_no_inline_set.len(),
        n_generic_footprint: acc.gen_fp_set.len(),
        n_generic_alias: acc.gen_alias_set.len(),
        histogram: acc.histogram,
        savings,
        timings: input.timings,
        census_span: Span::default(),
        origins: acc.origins,
    }
}

fn record_naive_member(group: &mut NaiveGroup, orig: usize, rec: &InstanceRecord) {
    group.origin = orig;
    group.k_set.insert(rec.k_inline);
    group.niche_set.insert(rec.niche);
    group.offset_set.insert(rec.offset);
    group.assoc_set.insert(rec.assoc_abi);
    group.spec_set.insert(rec.spec_callee);
    group.reflection_set.insert(rec.reflection);
}

fn record_code_size(
    hist: &mut Histogram,
    global_map: &mut FxHashMap<(ItemRef, Digest), StencilClass>,
    origin_map: &mut FxHashMap<(ItemRef, Digest), StencilClass>,
    (item, rec, code_len): (ItemRef, &InstanceRecord, u32),
) {
    match code_len {
        1..=128 => hist.le_128 += 1,
        129..=256 => hist.le_256 += 1,
        257..=512 => hist.le_512 += 1,
        513..=1024 => hist.le_1024 += 1,
        _ => hist.gt_1024 += 1,
    }
    let bytes = u64::from(code_len);
    for map in [global_map, origin_map] {
        let entry = map.entry((item, rec.k_inline)).or_default();
        entry.rep_bytes = entry.rep_bytes.max(bytes);
        entry.d_slots = entry.d_slots.max(u64::from(rec.d_slots));
        let slot = entry.alias_map.entry(rec.d_inline).or_insert(0);
        *slot = (*slot).max(bytes);
    }
}

fn summarize_hazards(naive_map: &FxHashMap<(ItemRef, Digest), NaiveGroup>, origins: &mut [OriginReport; 3]) -> (usize, usize, HazardCauses) {
    let mut causes = HazardCauses::default();
    let mut hazards = 0;
    for group in naive_map.values() {
        origins[group.origin].n_naive_shape += 1;
        let niche = group.niche_set.len() > 1;
        let offset = group.offset_set.len() > 1;
        let assoc = group.assoc_set.len() > 1;
        let spec = group.spec_set.len() > 1;
        let reflection = group.reflection_set.len() > 1;
        let k_diff = group.k_set.len() > 1;
        if !k_diff && !spec && !reflection {
            continue;
        }
        hazards += 1;
        origins[group.origin].n_naive_shape_hazards += 1;
        causes.niche_layout_mismatch += usize::from(niche);
        causes.field_offset_mismatch += usize::from(offset);
        causes.assoc_abi_mismatch += usize::from(assoc);
        causes.specialization_mismatch += usize::from(spec);
        causes.reflection_mismatch += usize::from(reflection);
        causes.other_footprint_mismatch += usize::from(!niche && !offset && !assoc && !spec && !reflection);
    }
    (naive_map.len(), hazards, causes)
}

fn finalize_savings(stencil_map: &FxHashMap<(ItemRef, Digest), StencilClass>) -> CodeSavings {
    let mut out = CodeSavings::default();
    for class in stencil_map.values() {
        out.unique_footprint_code_bytes = out.unique_footprint_code_bytes.saturating_add(class.rep_bytes);
        let alias_bytes: u64 = class.alias_map.values().copied().sum();
        out.unique_alias_code_bytes = out.unique_alias_code_bytes.saturating_add(alias_bytes);
        let aliases = class.alias_map.len() as u64;
        if class.rep_bytes > DICT_MIN_BYTES && aliases >= DICT_MIN_FANOUT {
            let dict_cost = class.rep_bytes.saturating_add(aliases.saturating_mul(class.d_slots.saturating_mul(8).saturating_add(24)));
            if dict_cost < alias_bytes {
                out.dict_stencil_classes += 1;
                out.projected_stenciled_bytes = out.projected_stenciled_bytes.saturating_add(dict_cost);
                continue;
            }
        }
        out.projected_stenciled_bytes = out.projected_stenciled_bytes.saturating_add(alias_bytes);
    }
    out
}

fn split_span(span: Span, weights: [u64; 3]) -> [Span; 3] {
    let sum: u64 = weights.iter().sum();
    if sum == 0 {
        return [Span::default(); 3];
    }
    let scale = |total: u64, w: u64| ((u128::from(total) * u128::from(w)) / u128::from(sum)) as u64;
    weights.map(|w| Span { wall_ns: scale(span.wall_ns, w), cpu_ns: scale(span.cpu_ns, w) })
}

fn finalize_origins(origins: &mut [OriginReport; 3], sets: &[OriginSets; 3], timings: Timings) {
    let outline_w = [0, 1, 2].map(|i| (origins[i].items as u64).saturating_add((origins[i].expansions as u64) * 8).max(origins[i].source_bytes / 64));
    let reach_w = [0, 1, 2].map(|i| origins[i].inferred_exprs.max(origins[i].n_inst as u64));
    let codegen_w = [0, 1, 2].map(|i| origins[i].codegen_worker_ns.max(origins[i].savings.total_code_bytes));
    let outline_spans = split_span(timings.outline, outline_w);
    let reach_spans = split_span(timings.reach, reach_w);
    let codegen_spans = split_span(timings.codegen, codegen_w);
    for i in 0..3 {
        origins[i].n_fn = sets[i].fn_set.len();
        origins[i].n_generic_fn = sets[i].gen_fn_set.len();
        origins[i].n_footprint_no_inline = sets[i].fp_no_inline_set.len();
        origins[i].n_footprint = sets[i].fp_set.len();
        origins[i].n_alias = sets[i].alias_set.len();
        let saved = finalize_savings(&sets[i].stencil_map);
        origins[i].savings.unique_footprint_code_bytes = saved.unique_footprint_code_bytes;
        origins[i].savings.unique_alias_code_bytes = saved.unique_alias_code_bytes;
        origins[i].savings.dict_stencil_classes = saved.dict_stencil_classes;
        origins[i].savings.projected_stenciled_bytes = saved.projected_stenciled_bytes;
        origins[i].outline_span = outline_spans[i];
        origins[i].reach_span = reach_spans[i];
        origins[i].codegen_span = codegen_spans[i];
    }
}
