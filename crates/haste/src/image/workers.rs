//! Compile workers: scoped threads that each encode a fixed share of the reached functions, grouped by footprint into stencils, into thread-local code chunks merged into the image.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::lower::{self, FootprintId, RelocPlan, Slots, SymbolSource};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};
use crate::substrate::tally::{Tally, clock, tally};
use crate::sync::lock;
use haste_codegen::{Code, Memory};

use super::aot::{self, AotCode, AotFn, AotQueue, CachedAotTemplate, bucket_by_item, lpt_chunks};
use super::stencil::{DictConfig, DistinctInst, GroupWork, aot_relocs_valid, item_err, mode_dict_key, mode_stencil_key};
use super::global_asm::Foreign;
use super::{Image, ImageMode, Pass, resolve_host_symbol, weak};

const MIN_PARALLEL: usize = 64;
const AOT_MIN_PARALLEL: usize = 2;
const MAX_BUCKET: usize = 128;
const AOT_MAX_BUCKET: usize = 64;
const MAX_PARALLEL_WAVES: usize = 4;

/// What one worker hands back: its pass, the errors by index, the functions whose constant evaluations it deferred, and the requested evals.
type WorkerOutput = (Pass, Vec<(usize, String)>, Vec<usize>, Vec<(Instance, u32)>);

/// How a pass compiles its buckets: the dictionary configuration, and whether a bucket of one function compiles alone.
type Plan = (DictConfig, bool);

/// The functions of `work` grouped by item, in the order each item first appears.
pub(super) fn bucket_work(reached: &Reached, work: &[usize]) -> Vec<Vec<usize>> {
    bucket_by_item(work, |index| reached.list[index].0.0)
}

/// The chunks of `buckets` dealt into at most `jobs` shards of about equal function counts.
pub(super) fn lpt_shards(buckets: Vec<Vec<usize>>, jobs: usize) -> Vec<Vec<Vec<usize>>> {
    let chunks = lpt_chunks(buckets, MAX_BUCKET);
    let count = jobs.min(chunks.len());
    let mut loads = vec![0usize; count];
    let mut shards: Vec<Vec<Vec<usize>>> = (0..count).map(|_| Vec::new()).collect();
    for chunk in chunks {
        let part = (0..count).min_by_key(|&p| loads[p]).unwrap_or(0);
        loads[part] += chunk.len();
        shards[part].push(chunk);
    }
    shards
}

fn seed_aot_roots(reached: &Reached, roots: [usize; 2]) -> (Vec<AtomicBool>, Vec<usize>) {
    let claimed: Vec<AtomicBool> = (0..reached.list.len()).map(|_| AtomicBool::new(false)).collect();
    let mut seed = Vec::new();
    for root in roots {
        if root < claimed.len() && !claimed[root].swap(true, Ordering::AcqRel) {
            seed.push(root);
        }
    }
    seed.sort_unstable();
    (claimed, seed)
}

impl Image {
    /// Gives every reached static its cell before workers run, warming static layouts on `cx`.
    pub(super) fn place_statics(&self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) {
        for &index in work {
            let Ok(typed) = &reached.list[index].1 else {
                continue;
            };
            if !lower::is_static(cx, reached.list[index].0.0) {
                continue;
            }
            if let Some(size) = typed.ret.and_then(|ty| cx.layout(ty).ok()).map(|layout| layout.size.bytes() as u32) {
                self.cell(index, size);
            }
        }
    }

    fn persist_inline(&self, cfg: DictConfig) {
        if !self.backend.is_lazy_jit() && crate::prep::decided().is_none() {
            self.bodies.persist(mode_stencil_key(self.backend.mode(), 0), cfg.verify);
        }
    }

    /// Whether this image recorded inline template outcomes that the substrate does not hold yet.
    pub(crate) fn has_new_inline_meta(&self) -> bool {
        self.bodies.has_fresh_meta()
    }

    /// Appends the inline template outcomes this image recorded for sysroot items into `toolchain` and for dependency items into `deps`.
    pub(crate) fn export_inline_meta(&self, toolchain: Option<&mut crate::substrate::SubstrateBuilder>, deps: Option<&mut crate::substrate::SubstrateBuilder>) -> Result<()> {
        self.bodies.export_meta(toolchain, deps)
    }

    /// Compiles every function of `work` through stencils, split across scoped worker threads when there is enough of it, and returns the errors by index.
    pub(super) fn compile_all(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) -> Vec<(usize, String)> {
        if work.is_empty() {
            return Vec::new();
        }
        if self.backend.is_jit() {
            work.iter().for_each(|&index| self.slot(index).store(0, Ordering::Release));
        }
        let cfg = DictConfig::from_env();
        self.pass.stencils.cfg = cfg;
        self.persist_inline(cfg);
        let block_parallel = !self.pass.stencils.templates.is_empty() && !self.backend.is_aot();
        let plan = (cfg, !block_parallel && !cfg.verify && cfg.min_fanout > 1);
        let buckets = bucket_work(reached, work);
        let jobs = crate::jobs();
        let min_parallel = if self.backend.is_aot() { AOT_MIN_PARALLEL } else { MIN_PARALLEL };
        if jobs <= 1 || cfg.verify || work.len() < min_parallel || block_parallel {
            return self.compile_buckets(cx, reached, &buckets, plan);
        }
        let shards = lpt_shards(buckets, jobs);
        if shards.len() <= 1 {
            return self.compile_buckets(cx, reached, &shards.concat(), plan);
        }
        self.compile_parallel(cx, reached, &shards, plan)
    }

    fn compile_buckets(&mut self, cx: &mut Context<'_>, reached: &Reached, buckets: &[Vec<usize>], (cfg, allow_singleton): Plan) -> Vec<(usize, String)> {
        self.with_pass(|image, pass| {
            buckets
                .iter()
                .flat_map(|bucket| {
                    let out = image.compile_bucket(cx, reached, bucket, cfg, (true, allow_singleton), pass);
                    image.bodies.trim_transient();
                    out
                })
                .collect()
        })
    }

    fn compile_parallel(&mut self, cx: &mut Context<'_>, reached: &Reached, initial_shards: &[Vec<Vec<usize>>], plan: Plan) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        let mut current_shards = initial_shards.to_vec();
        let jobs = crate::jobs();
        let min_parallel = if self.backend.is_aot() { AOT_MIN_PARALLEL } else { MIN_PARALLEL };
        let mut deferred = Vec::new();
        for wave in 0..MAX_PARALLEL_WAVES {
            let forks: Vec<Context<'_>> = current_shards.iter().map(|_| cx.fork()).collect();
            let outputs = self.run_shards(forks, &current_shards, |fork, shard| self.worker_loop(fork, reached, shard, plan));
            deferred.clear();
            let mut eval_requests = Vec::new();
            for (shard, output) in current_shards.iter().zip(outputs) {
                match output {
                    Ok((pass, failed, later, reqs)) => {
                        self.pass.merge(pass);
                        errors.extend(failed);
                        deferred.extend(later);
                        eval_requests.extend(reqs);
                    }
                    Err(reason) => errors.extend(self.lost(cx, reached, shard, &reason)),
                }
            }
            if deferred.is_empty() || eval_requests.is_empty() || wave + 1 == MAX_PARALLEL_WAVES || deferred.len() < min_parallel {
                break;
            }
            prime_eval_requests(cx, eval_requests);
            let next_shards = lpt_shards(bucket_work(reached, &deferred), jobs);
            if next_shards.len() <= 1 {
                break;
            }
            current_shards = next_shards;
        }
        if !deferred.is_empty() {
            deferred.sort_unstable();
            let was_parallel = std::mem::replace(&mut cx.in_parallel, false);
            errors.extend(self.compile_buckets(cx, reached, &bucket_work(reached, &deferred), plan));
            cx.in_parallel = was_parallel;
        }
        errors.sort_by_key(|&(index, _)| index);
        errors
    }

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

/// Clears the slots of a worker that panicked, whose code chunks unmapped as it unwound, and gives each of its functions an error.
    fn lost(&self, cx: &Context<'_>, reached: &Reached, shard: &[Vec<usize>], reason: &str) -> Vec<(usize, String)> {
        let lost = shard.iter().flatten().map(|&index| {
            if self.backend.is_jit() {
                self.slot(index).store(0, Ordering::Release);
            }
            (index, format!("{}: compile worker panicked: {reason}", cx.name(reached.list[index].0.0)))
        });
        lost.collect()
    }

    /// Runs `work` on each shard in a scoped thread of its own with one of `forks`, returning each thread's value or its panic payload message.
    pub(super) fn run_shards<'a, T: Send>(&self, forks: Vec<Context<'a>>, shards: &[Vec<Vec<usize>>], work: impl Fn(Context<'a>, &[Vec<usize>]) -> T + Sync) -> Vec<Result<T, String>> {
        let work = &work;
        std::thread::scope(|scope| {
            let handles: Vec<_> = forks.into_iter().zip(shards).map(|(fork, shard)| scope.spawn(move || work(fork, shard))).collect();
            handles.into_iter().map(|handle| handle.join().map_err(Self::panic_message)).collect()
        })
    }

    /// Compiles each bucket of `shard` into a pass of this worker's own; a bucket whose lowering deferred a constant evaluation is handed back to compile after the workers join.
    fn worker_loop(&self, mut cx: Context<'_>, reached: &Reached, shard: &[Vec<usize>], (cfg, allow_singleton): Plan) -> WorkerOutput {
        let mut pass = Pass::default();
        pass.stencils.cfg = cfg;
        let (mut errors, mut deferred, mut eval_requests) = (Vec::new(), Vec::new(), Vec::new());
        for bucket in shard {
            self.compile_worker_bucket(&mut cx, reached, bucket, (cfg, allow_singleton), &mut pass, (&mut errors, &mut deferred, &mut eval_requests));
        }
        (pass, errors, deferred, eval_requests)
    }

    fn compile_worker_bucket(&self, cx: &mut Context<'_>, reached: &Reached, bucket: &[usize], (cfg, allow_singleton): Plan, pass: &mut Pass, (errors, deferred, eval_requests): (&mut Vec<(usize, String)>, &mut Vec<usize>, &mut Vec<(Instance, u32)>)) {
        let mark = pass.records.len();
        let frozen_mark = pass.frozen_stencils.len();
        let failed = self.compile_bucket(cx, reached, bucket, cfg, (false, allow_singleton), pass);
        let reqs = cx.take_deferred_evals();
        if reqs.is_empty() {
            errors.extend(failed);
        } else {
            pass.records.truncate(mark);
            pass.frozen_stencils.truncate(frozen_mark);
            deferred.extend_from_slice(bucket);
            eval_requests.extend(reqs);
        }
        self.bodies.trim_transient();
    }

    pub(super) fn build_aot_streaming(cx: &mut Context<'_>, reached: &Reached, roots: [usize; 2]) -> Result<(Image, FxHashMap<usize, String>)> {
        let mut image = Self::create(reached, ImageMode::Aot, (0, 0, 0))?;
        let cfg = DictConfig::from_env();
        image.pass.stencils.cfg = cfg;
        image.persist_inline(cfg);
        let plan = (cfg, !cfg.verify && cfg.min_fanout > 1);
        let (claimed, seed) = seed_aot_roots(reached, roots);
        let mut broken = FxHashMap::default();
        let mut init_errors = Vec::new();
        let work = image.prepare_work(cx, reached, &seed, (&mut init_errors, None));
        for (idx, err) in init_errors {
            broken.insert(idx, err);
        }
        let items: Box<[ItemRef]> = reached.list.iter().map(|((item, _), _)| *item).collect();
        let queue = AotQueue::new(claimed, items, AOT_MAX_BUCKET, lpt_chunks(bucket_work(reached, &work), AOT_MAX_BUCKET));
        if let Some(aot) = image.aot() {
            lock(aot).frontier = Some(Arc::clone(&queue));
        }
        let jobs = crate::jobs();
        let mut deferred = Vec::new();
        if jobs <= 1 || cfg.verify {
            image.drain_aot_serial(cx, reached, &queue, plan, &mut broken);
        } else {
            for wave in 0..MAX_PARALLEL_WAVES {
                let outputs = image.run_aot_workers(cx, reached, &queue, jobs, plan);
                deferred.clear();
                let mut eval_requests = Vec::new();
                for output in outputs {
                    match output {
                        Some((pass, failed, later, reqs)) => {
                            image.pass.merge(pass);
                            for (idx, err) in failed {
                                broken.insert(idx, err);
                            }
                            deferred.extend(later);
                            eval_requests.extend(reqs);
                        }
                        None => return Err(anyhow::anyhow!("AOT compile worker panicked")),
                    }
                }
                image.apply_records();
                if deferred.is_empty() || eval_requests.is_empty() {
                    break;
                }
                prime_eval_requests(cx, eval_requests);
                if wave + 1 == MAX_PARALLEL_WAVES || deferred.len() < AOT_MIN_PARALLEL {
                    break;
                }
                deferred.sort_unstable();
                queue.reseed_buckets(lpt_chunks(bucket_work(reached, &deferred), AOT_MAX_BUCKET));
            }
            if !deferred.is_empty() {
                deferred.sort_unstable();
                queue.reseed_buckets(lpt_chunks(bucket_work(reached, &deferred), AOT_MAX_BUCKET));
                let was_parallel = std::mem::replace(&mut cx.in_parallel, false);
                image.drain_aot_serial(cx, reached, &queue, plan, &mut broken);
                cx.in_parallel = was_parallel;
            }
        }
        if let Some(aot) = image.aot() {
            let mut state = lock(aot);
            state.frontier = None;
            state.stencils.clear();
        }
        image.bodies.reset();
        image.pass.stencils.clear();
        Ok((image, broken))
    }

    fn drain_aot_serial(&mut self, cx: &mut Context<'_>, reached: &Reached, queue: &AotQueue, plan: Plan, broken: &mut FxHashMap<usize, String>) {
        loop {
            let mut batch = queue.take_all_pending();
            if batch.is_empty() {
                break;
            }
            batch.sort_unstable();
            let mut batch_errors = Vec::new();
            let work = self.prepare_work(cx, reached, &batch, (&mut batch_errors, None));
            for (idx, err) in batch_errors {
                broken.insert(idx, err);
            }
            let buckets = lpt_chunks(bucket_work(reached, &work), AOT_MAX_BUCKET);
            for (idx, err) in self.compile_buckets(cx, reached, &buckets, plan) {
                broken.insert(idx, err);
            }
        }
        self.apply_records();
    }

    fn run_aot_workers(&self, cx: &Context<'_>, reached: &Reached, queue: &AotQueue, jobs: usize, plan: Plan) -> Vec<Option<WorkerOutput>> {
        let forks: Vec<Context<'_>> = (0..jobs).map(|_| cx.fork()).collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = forks.into_iter().map(|fork| scope.spawn(move || self.aot_worker_loop(fork, reached, queue, plan))).collect();
            handles.into_iter().map(|handle| handle.join().map_or(None, Some)).collect()
        })
    }

    fn aot_worker_loop(&self, mut cx: Context<'_>, reached: &Reached, queue: &AotQueue, (cfg, allow_singleton): Plan) -> WorkerOutput {
        let mut pass = Pass::default();
        pass.stencils.cfg = cfg;
        let (mut errors, mut deferred, mut eval_requests) = (Vec::new(), Vec::new(), Vec::new());
        while let Some((batch, _guard)) = queue.pop() {
            let mut batch_errors = Vec::new();
            let work = self.prepare_work(&mut cx, reached, &batch, (&mut batch_errors, None));
            errors.extend(batch_errors);
            if !work.is_empty() {
                self.compile_worker_bucket(&mut cx, reached, &work, (cfg, allow_singleton), &mut pass, (&mut errors, &mut deferred, &mut eval_requests));
            }
        }
        (pass, errors, deferred, eval_requests)
    }

    pub(super) fn prepare_work(&self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize], (errors, mut memory): (&mut Vec<(usize, String)>, Option<&mut Memory>)) -> Vec<usize> {
        self.bind_foreign_statics(cx, reached, indices);
        let mut work = Vec::new();
        let mut foreign_fns = Vec::new();
        for &index in indices {
            if reached.list[index].1.is_err() {
                continue;
            }
            let item = reached.list[index].0.0;
            if !cx.foreign(item) {
                work.push(index);
                continue;
            }
            match self.bind_foreign(cx, reached, index, memory.as_deref_mut()) {
                Ok(Foreign::Host(sym, address)) if !lower::is_static(cx, item) => {
                    if self.backend.is_jit() {
                        self.slot(index).store(address, Ordering::Release);
                    }
                    if self.aot().is_some() {
                        foreign_fns.push((index, sym));
                    }
                }
                Ok(_) => {}
                Err(_) if lower::is_static(cx, item) && weak(cx, item) => {}
                Err(error) => errors.push((index, error)),
            }
        }
        if !foreign_fns.is_empty()
            && let Some(aot) = self.aot()
        {
            let mut state = lock(aot);
            for (index, sym) in foreign_fns {
                state.foreign_fns.insert(index, sym);
            }
        }
        self.place_statics(cx, reached, &work);
        work
    }

    pub(super) fn bind_foreign_statics(&self, cx: &Context<'_>, reached: &Reached, indices: &[usize]) {
        let jit = self.backend.is_jit();
        let mut cells_batch = Vec::new();
        let mut aot_batch = Vec::new();
        for (index, ((item, _), typed)) in indices.iter().map(|&index| (index, &reached.list[index])) {
            if typed.is_err() || !cx.foreign(*item) || !lower::is_static(cx, *item) {
                continue;
            }
            let resolved = resolve_host_symbol(cx, *item, !self.backend.is_jit());
            let is_weak = weak(cx, *item);
            let host = match (&resolved, is_weak) {
                (Ok((host, _)), _) => Some(host.clone()),
                (Err(_), true) => None,
                (Err(_), false) => continue,
            };
            if jit {
                let cell = match (resolved, is_weak) {
                    (Ok((_, address)), true) => self.data(&address.to_le_bytes()),
                    (Err(_), true) => self.data(&0_usize.to_le_bytes()),
                    (Ok((_, address)), false) => address as i64,
                    (Err(_), false) => continue,
                };
                cells_batch.push((index, cell));
            }
            if self.aot().is_some() {
                aot_batch.push((index, aot::AotStatic::Foreign { weak: is_weak, host }));
            }
        }
        if !cells_batch.is_empty() {
            let mut cells = lock(&self.cells);
            for (index, cell) in cells_batch {
                cells.insert(index, cell);
            }
        }
        if !aot_batch.is_empty()
            && let Some(aot) = self.aot()
        {
            let mut state = lock(aot);
            for (index, entry) in aot_batch {
                state.statics.insert(index, entry);
            }
        }
    }

    pub(super) fn compile_partitioned(&self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], cfg: DictConfig, record_cache: bool, pass: &mut Pass) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        let mut order: Vec<(ItemRef, FootprintId)> = Vec::new();
        let mut by_fp: FxHashMap<(ItemRef, FootprintId), Vec<usize>> = FxHashMap::default();
        let footprint_clock = std::time::Instant::now();
        for &index in work {
            let item = reached.list[index].0.0;
            let footprint = match lower::eval_footprint(cx, reached, self, index) {
                Ok(fp) => fp,
                Err(err) => {
                    errors.push((index, item_err(cx, item, &err)));
                    continue;
                }
            };
            by_fp.entry((item, footprint)).or_insert_with(|| {
                order.push((item, footprint));
                Vec::new()
            }).push(index);
        }
        clock(Tally::FootprintUs, footprint_clock);
        let mut verify_hits = Vec::new();
        let aot_fast = self.backend.is_aot() && !cfg.verify;
        for (item, footprint) in order {
            let Some(indices) = by_fp.remove(&(item, footprint)) else { continue };
            if record_cache && let Some(reloc_plan) = pass.stencils.templates.get(&(item, footprint)).map(|t| Arc::clone(t.reloc_plan())) {
                for index in indices {
                    let relocs = match lower::eval_relocs(cx, reached, self, &reloc_plan, index) {
                        Ok(r) => r,
                        Err(err) => {
                            errors.push((index, item_err(cx, item, &err)));
                            continue;
                        }
                    };
                    let symbols = self.symbols(&relocs);
                    if let Some(hit) = self.reuse_cached((item, footprint), index, &symbols, self.backend.is_aot().then_some(&relocs), pass)
                        && cfg.verify
                    {
                        verify_hits.push((index, hit));
                    }
                }
                continue;
            }
            if aot_fast && let Some(cached) = self.cached_aot_template(item, footprint) {
                self.apply_cached_aot_template(cx, reached, item, &indices, &cached, pass, &mut errors);
                continue;
            }
            let Some((&rep, rest)) = indices.split_first() else { continue };
            let lookup_clock = std::time::Instant::now();
            let frozen = self.try_frozen_partitioned(cx, reached, item, footprint, rep, rest, &indices, cfg, pass);
            clock(Tally::LookupUs, lookup_clock);
            if frozen {
                continue;
            }
            let lower_clock = std::time::Instant::now();
            let lowered = lower::lower_stencil(cx, reached, self, rep);
            clock(Tally::LowerUs, lower_clock);
            let out = match lowered {
                Ok(out) => out,
                Err(err) => {
                    let msg = item_err(cx, item, &err);
                    errors.extend(indices.into_iter().map(|idx| (idx, msg.clone())));
                    continue;
                }
            };
            let distinct = self.group_instances(cx, reached, item, &out, rep, rest, aot_fast, &mut errors);
            let mut group = GroupWork { item, footprint, distinct, leader_out: Some(out) };
            if let Err(err) = self.compile_group(cx, reached, &mut group, cfg, record_cache, pass) {
                errors.extend(group.all_indices().map(|idx| (idx, err.clone())));
            }
            drop(group);
            self.bodies.trim_transient();
        }
        if cfg.verify {
            errors.extend(self.verify_records(cx, reached, &verify_hits));
        }
        errors
    }

    fn apply_cached_aot_template(&self, cx: &mut Context<'_>, reached: &Reached, item: ItemRef, indices: &[usize], cached: &CachedAotTemplate, pass: &mut Pass, errors: &mut Vec<(usize, String)>) {
        let mut evaluated = Vec::with_capacity(indices.len());
        for &index in indices {
            match lower::eval_relocs(cx, reached, self, cached.reloc_plan(), index) {
                Ok(relocs) => evaluated.push((index, relocs)),
                Err(err) => errors.push((index, item_err(cx, item, &err))),
            }
        }
        if !cx.deferred_evals.is_empty() {
            return;
        }
        self.commit_aot_groups(&evaluated, cached, 0, pass);
    }

    fn try_direct_frozen_slice(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        (item, footprint, rep): (ItemRef, FootprintId, usize),
        (rest, indices): (&[usize], &[usize]),
        slice: &crate::substrate::StencilSlice<'_>,
        pass: &mut Pass,
    ) -> Result<(), Tally> {
        if self.backend.is_check() {
            let len = slice.code().len() as u32;
            for &idx in indices {
                pass.record_size(idx, len, 0);
            }
            return Ok(());
        }
        if !self.backend.is_aot() {
            return Err(Tally::MissOther);
        }
        let Ok(code) = slice.to_code().map(Arc::new) else { return Err(Tally::MissOther) };
        if code.relocs().iter().all(|r| r.symbol_index().is_none()) && code.unwind_drops().is_empty() {
            let len = code.bytes().len() as u32;
            let aot_code = AotCode::from_code(&code);
            self.cache_aot_template(cx, reached, (item, footprint, rep), CachedAotTemplate::Direct { code: Arc::clone(&aot_code), reloc_plan: Arc::default() });
            self.record_aot(rep, AotFn::Direct { code: aot_code, relocs: Vec::new() }, rest);
            for &idx in indices {
                pass.record_size(idx, len, 0);
            }
            return Ok(());
        }
        let Some(reloc_plan) = self.decode_aot_reloc_plan(cx.outline, &cx.types, reached, false, slice.plan()).map(Arc::new) else {
            return Err(Tally::MissPlan);
        };
        if aot_relocs_valid(&code, reloc_plan.reloc_len())
            && self.try_commit_frozen_template(cx, reached, (item, footprint, rep), indices, CachedAotTemplate::Direct { code: AotCode::from_code(&code), reloc_plan }, 0, pass)
        {
            return Ok(());
        }
        Err(Tally::MissOther)
    }

    fn try_dict_frozen_slice(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        (item, footprint, rep): (ItemRef, FootprintId, usize),
        indices: &[usize],
        slice: &crate::substrate::StencilSlice<'_>,
        pass: &mut Pass,
    ) -> Result<(), Tally> {
        let Ok(code) = slice.to_code().map(Arc::new) else { return Err(Tally::MissOther) };
        let Some(reloc_plan) = self.decode_aot_reloc_plan(cx.outline, &cx.types, reached, false, slice.plan()).map(Arc::new) else {
            return Err(Tally::MissPlan);
        };
        let dict_len = code.bytes().len() as u32;
        if self.try_commit_frozen_template(cx, reached, (item, footprint, rep), indices, CachedAotTemplate::Dict { body: AotCode::from_code(&code), reloc_plan }, dict_len, pass) {
            Ok(())
        } else {
            Err(Tally::MissOther)
        }
    }

    fn try_frozen_partitioned(&self, cx: &mut Context<'_>, reached: &Reached, item: ItemRef, footprint: FootprintId, rep: usize, rest: &[usize], indices: &[usize], cfg: DictConfig, pass: &mut Pass) -> bool {
        if cfg.verify {
            return false;
        }
        tally(Tally::Lookup);
        let Some((_, direct_key, fp_bits)) = self.frozen_stencil_key(cx, reached, rep, footprint) else {
            tally(if footprint.is_portable() { Tally::MissUnkeyed } else { Tally::MissUnportable });
            return false;
        };
        let Some(substrate) = cx.store().substrate().cloned() else {
            tally(Tally::MissOther);
            return false;
        };
        let prefer_dict = self.backend.is_aot() && cfg.min_bytes == 0 && indices.len() >= cfg.min_fanout;
        let mut reason = Tally::MissAbsent;
        if let Some(slice) = (!prefer_dict).then(|| substrate.stencil(direct_key, fp_bits)).flatten() {
            match self.try_direct_frozen_slice(cx, reached, (item, footprint, rep), (rest, indices), &slice, pass) {
                Ok(()) => return { tally(Tally::Hit); true },
                Err(why) => reason = why,
            }
        }
        let dict_slice = (self.backend.is_aot() && indices.len() >= cfg.min_fanout)
            .then(|| substrate.stencil(mode_dict_key(direct_key), fp_bits))
            .flatten()
            .filter(|slice| prefer_dict || slice.code().len() > cfg.min_bytes);
        if let Some(slice) = dict_slice {
            match self.try_dict_frozen_slice(cx, reached, (item, footprint, rep), indices, &slice, pass) {
                Ok(()) => return { tally(Tally::Hit); true },
                Err(why) => reason = why,
            }
        }
        tally(reason);
        false
    }

    fn try_commit_frozen_template(&self, cx: &mut Context<'_>, reached: &Reached, key: (ItemRef, FootprintId, usize), indices: &[usize], cached: CachedAotTemplate, dict_len: u32, pass: &mut Pass) -> bool {
        let mut evaluated = Vec::with_capacity(indices.len());
        for &index in indices {
            let Ok(relocs) = lower::eval_relocs(cx, reached, self, cached.reloc_plan(), index) else {
                return !cx.deferred_evals.is_empty();
            };
            evaluated.push((index, relocs));
        }
        if !cx.deferred_evals.is_empty() {
            return true;
        }
        self.commit_aot_groups(&evaluated, &cached, dict_len, pass);
        self.cache_aot_template(cx, reached, key, cached);
        true
    }

    fn commit_aot_groups(&self, evaluated: &[(usize, Vec<SymbolSource>)], cached: &CachedAotTemplate, dict_len: u32, pass: &mut Pass) {
        let mut reloc_pos: FxHashMap<&[SymbolSource], usize> = FxHashMap::default();
        let mut groups: Vec<(usize, Vec<usize>, Vec<SymbolSource>)> = Vec::new();
        let code_len = match cached {
            CachedAotTemplate::Direct { code, .. } => code.bytes().len() as u32,
            CachedAotTemplate::Dict { .. } => 0,
        };
        for (i, (index, relocs)) in evaluated.iter().enumerate() {
            if let Some(&pos) = reloc_pos.get(relocs.as_slice()) {
                groups[pos].1.push(*index);
            } else {
                reloc_pos.insert(relocs.as_slice(), groups.len());
                groups.push((*index, Vec::new(), relocs.clone()));
            }
            let size = if i == 0 && dict_len > 0 { dict_len } else { code_len };
            pass.record_size(*index, size, 0);
        }
        for (leader, followers, relocs) in groups {
            self.register_aot_relocs(&relocs);
            let entry = match cached {
                CachedAotTemplate::Direct { code, .. } => AotFn::Direct { code: Arc::clone(code), relocs },
                CachedAotTemplate::Dict { body, .. } => AotFn::Dict { body: Arc::clone(body), relocs },
            };
            self.record_aot(leader, entry, &followers);
        }
    }

    pub(super) fn try_frozen_dict_group(&self, cx: &mut Context<'_>, reached: &Reached, group: &GroupWork, pass: &mut Pass) -> Option<Result<(), String>> {
        let rep = group.representative();
        let (_, direct_key, fp_bits) = self.frozen_stencil_key(cx, reached, rep, group.footprint)?;
        let slice = cx.store().substrate()?.stencil(mode_dict_key(direct_key), fp_bits)?;
        let code = Arc::new(slice.to_code().ok()?);
        let reloc_plan = Arc::new(self.decode_aot_reloc_plan(cx.outline, &cx.types, reached, false, slice.plan())?);
        let mut evaluated = Vec::with_capacity(group.distinct.len());
        for inst in &group.distinct {
            let relocs = lower::eval_relocs(cx, reached, self, &reloc_plan, inst.leader).ok()?;
            if !cx.deferred_evals.is_empty() {
                return Some(Ok(()));
            }
            evaluated.push((inst.leader, &inst.followers, relocs));
        }
        let body = AotCode::from_code(&code);
        for (leader, followers, relocs) in evaluated {
            self.register_aot_relocs(&relocs);
            self.record_aot(leader, AotFn::Dict { body: Arc::clone(&body), relocs }, followers);
            pass.record_size(leader, 0, 0);
        }
        pass.record_size(rep, code.bytes().len() as u32, 0);
        self.cache_aot_template(cx, reached, (group.item, group.footprint, rep), CachedAotTemplate::Dict { body, reloc_plan });
        tally(Tally::HitLowered);
        Some(Ok(()))
    }

    fn encode_frozen_plan(&self, cx: &Context<'_>, reached: &Reached, is_sysroot: bool, plan: Option<&RelocPlan>) -> Option<(bool, Vec<u8>)> {
        let p = plan?;
        if let Some(bytes) = self.encode_aot_reloc_plan(cx.outline, &cx.types, reached, is_sysroot, p) {
            return Some((is_sysroot, bytes));
        }
        is_sysroot
            .then(|| self.encode_aot_reloc_plan(cx.outline, &cx.types, reached, false, p))
            .flatten()
            .map(|bytes| (false, bytes))
    }

    pub(super) fn freeze_stencil(&self, cx: &Context<'_>, reached: &Reached, index: usize, fp: FootprintId, is_dict: bool, code: &Arc<Code>, plan: Option<&RelocPlan>, pass: &mut Pass) {
        let valid_relocs = if is_dict {
            code.relocs().iter().all(|r| r.as_runtime().is_some())
        } else {
            code.relocs().iter().all(|r| r.symbol_index().is_some() || (self.backend.is_aot() && r.as_runtime().is_some()))
        };
        if !cx.deferred_evals.is_empty() {
            return tally(Tally::FreezeDeferred);
        }
        let Some((is_sysroot, direct_key, fp_bits)) = valid_relocs.then(|| self.frozen_stencil_key(cx, reached, index, fp)).flatten() else {
            return tally(Tally::FreezeSkipped);
        };
        let needs_plan = is_dict || code.relocs().iter().any(|r| r.symbol_index().is_some());
        let (store_sysroot, plan_bytes) = if self.backend.is_aot() && needs_plan {
            let Some(pair) = self.encode_frozen_plan(cx, reached, is_sysroot, plan) else {
                return tally(if is_dict { Tally::FreezeSkipped } else { Tally::FreezePlanless });
            };
            pair
        } else {
            (is_sysroot, Vec::new())
        };
        let key = if is_dict { mode_dict_key(direct_key) } else { direct_key };
        pass.frozen_stencils.push(super::FrozenStencil { is_sysroot: store_sysroot, key, fp: fp_bits, code: Arc::clone(code), plan: plan_bytes });
    }

    fn group_instances(&self, cx: &mut Context<'_>, reached: &Reached, item: ItemRef, out: &lower::StencilOutput, rep: usize, rest: &[usize], aot_fast: bool, errors: &mut Vec<(usize, String)>) -> Vec<DistinctInst> {
        let mut sym_pos: FxHashMap<InstanceKey, usize> = FxHashMap::default();
        let mut distinct: Vec<DistinctInst> = Vec::new();
        let (rep_key, rep_inst) = self.build_inst(rep, out.relocs.clone(), aot_fast);
        sym_pos.insert(rep_key, 0);
        distinct.push(rep_inst);
        for &index in rest {
            let relocs = match lower::eval_relocs(cx, reached, self, &out.reloc_plan, index) {
                Ok(r) => r,
                Err(err) => {
                    errors.push((index, item_err(cx, item, &err)));
                    continue;
                }
            };
            let (key, inst) = self.build_inst(index, relocs, aot_fast);
            if let Some(&pos) = sym_pos.get(&key) {
                distinct[pos].followers.push(index);
            } else {
                sym_pos.insert(key, distinct.len());
                distinct.push(inst);
            }
        }
        distinct
    }

    fn build_inst(&self, index: usize, relocs: Vec<SymbolSource>, aot_fast: bool) -> (InstanceKey, DistinctInst) {
        if aot_fast {
            self.register_aot_relocs(&relocs);
            let inst = DistinctInst { leader: index, followers: Vec::new(), symbols: Vec::new(), relocs: Some(relocs.clone()) };
            (InstanceKey::Relocs(relocs), inst)
        } else {
            let symbols = self.symbols(&relocs);
            let key = InstanceKey::Syms(symbols.clone());
            let inst = DistinctInst { leader: index, followers: Vec::new(), symbols, relocs: self.backend.is_aot().then_some(relocs) };
            (key, inst)
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
enum InstanceKey {
    Relocs(Vec<SymbolSource>),
    Syms(Vec<u64>),
}

fn prime_eval_requests(cx: &mut Context<'_>, eval_requests: Vec<(Instance, u32)>) {
    let was_parallel = std::mem::replace(&mut cx.in_parallel, false);
    let mut seen = FxHashSet::default();
    for (inst, bits) in eval_requests {
        if seen.insert((inst, bits)) {
            crate::sema::evaluate::prime(cx, &inst, bits);
        }
    }
    cx.in_parallel = was_parallel;
}
