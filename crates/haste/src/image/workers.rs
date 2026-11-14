//! Compile workers: scoped threads that each encode a fixed share of the reached functions, grouped by footprint into stencils, into thread-local code chunks merged into the image.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::lower::{self, FootprintId, Slots, SymbolSource};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};
use crate::sync::lock;

use super::aot::{self, AotCode, AotFn, AotQueue, bucket_by_item, lpt_chunks};
use super::stencil::{DictConfig, DistinctInst, GroupWork, item_err};
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

fn bucket_work(reached: &Reached, work: &[usize]) -> Vec<Vec<usize>> {
    bucket_by_item(work, |index| reached.list[index].0.0)
}

fn lpt_shards(buckets: Vec<Vec<usize>>, jobs: usize) -> Vec<Vec<Vec<usize>>> {
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
            let outputs = self.run_workers(forks, reached, &current_shards, plan);
            deferred.clear();
            let mut eval_requests = Vec::new();
            for (shard, output) in current_shards.iter().zip(outputs) {
                match output {
                    Some((pass, failed, later, reqs)) => {
                        self.pass.merge(pass);
                        errors.extend(failed);
                        deferred.extend(later);
                        eval_requests.extend(reqs);
                    }
                    None => errors.extend(self.lost(cx, reached, shard)),
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

    /// Clears the slots of a worker that panicked, whose code chunks unmapped as it unwound, and gives each of its functions an error.
    fn lost(&self, cx: &Context<'_>, reached: &Reached, shard: &[Vec<usize>]) -> Vec<(usize, String)> {
        let lost = shard.iter().flatten().map(|&index| {
            if self.backend.is_jit() {
                self.slot(index).store(0, Ordering::Release);
            }
            (index, format!("{}: compile worker panicked", cx.name(reached.list[index].0.0)))
        });
        lost.collect()
    }

    fn run_workers<'a>(&self, forks: Vec<Context<'a>>, reached: &Reached, shards: &[Vec<Vec<usize>>], plan: Plan) -> Vec<Option<WorkerOutput>> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = forks.into_iter().zip(shards).map(|(fork, shard)| scope.spawn(move || self.worker_loop(fork, reached, shard, plan))).collect();
            handles.into_iter().map(|handle| handle.join().map_or(None, Some)).collect()
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
        let plan = (cfg, !cfg.verify && cfg.min_fanout > 1);
        let (claimed, seed) = seed_aot_roots(reached, roots);
        let mut broken = FxHashMap::default();
        let mut init_errors = Vec::new();
        let work = image.prepare_work(cx, reached, &seed, &mut init_errors);
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
            lock(aot).frontier = None;
        }
        image.bodies = lower::Bodies::default();
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
            let work = self.prepare_work(cx, reached, &batch, &mut batch_errors);
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
            let work = self.prepare_work(&mut cx, reached, &batch, &mut batch_errors);
            errors.extend(batch_errors);
            if !work.is_empty() {
                self.compile_worker_bucket(&mut cx, reached, &work, (cfg, allow_singleton), &mut pass, (&mut errors, &mut deferred, &mut eval_requests));
            }
        }
        (pass, errors, deferred, eval_requests)
    }

    pub(super) fn prepare_work(&self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize], errors: &mut Vec<(usize, String)>) -> Vec<usize> {
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
            match resolve_host_symbol(cx, item) {
                Ok((sym, address)) if !lower::is_static(cx, item) => {
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
            let resolved = resolve_host_symbol(cx, *item);
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
                    let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
                    if let Some(hit) = self.reuse_cached((item, footprint), index, &symbols, self.backend.is_aot().then_some(&relocs), pass)
                        && cfg.verify
                    {
                        verify_hits.push((index, hit));
                    }
                }
                continue;
            }
            let Some((&rep, rest)) = indices.split_first() else { continue };
            if !cfg.verify
                && let Some((stencil_key, fp_bits)) = self.frozen_stencil_key(cx, reached, rep, footprint)
                && let Some(substrate) = cx.store().substrate()
                && let Some(slice) = substrate.stencil(stencil_key, fp_bits)
            {
                if self.backend.is_check() {
                    let len = slice.code().len() as u32;
                    for idx in indices {
                        pass.record_size(idx, len, 0);
                    }
                    continue;
                }
                if self.backend.is_aot()
                    && slice.decode_relocs().is_ok_and(|relocs| relocs.iter().all(|r| r.symbol_index().is_none()))
                    && let Ok(code) = slice.to_code().map(Arc::new)
                {
                    let len = code.bytes().len() as u32;
                    self.record_aot(rep, AotFn::Direct { code: AotCode::from_code(&code), relocs: Vec::new() }, rest);
                    for idx in indices {
                        pass.record_size(idx, len, 0);
                    }
                    continue;
                }
            }
            let out = match lower::lower_stencil(cx, reached, self, rep) {
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
            let symbols: Vec<u64> = relocs.iter().map(|sym| self.symbol_value(sym)).collect();
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
