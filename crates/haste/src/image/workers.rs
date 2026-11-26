mod aot;
mod partition;

use std::sync::atomic::Ordering;

use anyhow::Result;
use haste_codegen::Memory;
use rustc_hash::FxHashSet;

use crate::lower::{self, Slots};
use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};
use crate::sync::lock;

use super::aot::{self as image_aot, bucket_by_item, lpt_chunks};
use super::global_asm::Foreign;
use super::stencil::{DictConfig, mode_stencil_key};
use super::{Image, Pass, resolve_host_symbol, weak};

const MIN_PARALLEL: usize = 64;
const AOT_MIN_PARALLEL: usize = 2;
const MAX_BUCKET: usize = 128;
const MAX_PARALLEL_WAVES: usize = 4;

type WorkerOutput = (Pass, Vec<(usize, String)>, Vec<usize>, Vec<(Instance, u32)>);

type Plan = (DictConfig, bool);

struct WorkerBatch {
    pass: Pass,
    errors: Vec<(usize, String)>,
    deferred: Vec<usize>,
    eval_requests: Vec<(Instance, u32)>,
}

impl WorkerBatch {
    fn new(cfg: DictConfig) -> Self {
        let mut pass = Pass::default();
        pass.stencils.cfg = cfg;
        Self {
            pass,
            errors: Vec::new(),
            deferred: Vec::new(),
            eval_requests: Vec::new(),
        }
    }

    fn into_output(self) -> WorkerOutput {
        (self.pass, self.errors, self.deferred, self.eval_requests)
    }
}

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
            if let Some(size) = typed
                .ret
                .and_then(|ty| cx.layout(ty).ok())
                .map(|layout| layout.size.bytes() as u32)
            {
                self.cell(index, size);
            }
        }
    }

    fn persist_inline(&self, cfg: DictConfig) {
        if !self.backend.is_lazy_jit() && crate::prep::decided().is_none() {
            self.bodies
                .persist(mode_stencil_key(self.backend.mode(), 0), cfg.verify);
        }
    }

    /// Whether this image recorded inline template outcomes that the substrate does not hold yet.
    pub(crate) fn has_new_inline_meta(&self) -> bool {
        self.bodies.has_fresh_meta()
    }

    /// Appends the inline template outcomes this image recorded for sysroot items into `toolchain` and for dependency items into `deps`.
    pub(crate) fn export_inline_meta(
        &self,
        toolchain: Option<&mut crate::substrate::SubstrateBuilder>,
        deps: Option<&mut crate::substrate::SubstrateBuilder>,
    ) -> Result<()> {
        self.bodies.export_meta(toolchain, deps)
    }

    /// Compiles every function of `work` through stencils, split across scoped worker threads when there is enough of it, and returns the errors by index.
    pub(super) fn compile_all(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        work: &[usize],
    ) -> Vec<(usize, String)> {
        if work.is_empty() {
            return Vec::new();
        }
        if self.backend.is_jit() {
            work.iter()
                .for_each(|&index| self.slot(index).store(0, Ordering::Release));
        }
        let cfg = DictConfig::from_env();
        self.pass.stencils.cfg = cfg;
        self.persist_inline(cfg);
        let block_parallel = !self.pass.stencils.templates.is_empty() && !self.backend.is_aot();
        let plan = (cfg, !block_parallel && !cfg.verify && cfg.min_fanout > 1);
        let buckets = bucket_work(reached, work);
        let jobs = crate::jobs();
        let min_parallel = if self.backend.is_aot() {
            AOT_MIN_PARALLEL
        } else {
            MIN_PARALLEL
        };
        if jobs <= 1 || cfg.verify || work.len() < min_parallel || block_parallel {
            return self.compile_buckets(cx, reached, &buckets, plan);
        }
        let shards = lpt_shards(buckets, jobs);
        if shards.len() <= 1 {
            return self.compile_buckets(cx, reached, &shards.concat(), plan);
        }
        self.compile_parallel(cx, reached, &shards, plan)
    }

    fn compile_buckets(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        buckets: &[Vec<usize>],
        (cfg, allow_singleton): Plan,
    ) -> Vec<(usize, String)> {
        self.with_pass(|image, pass| {
            buckets
                .iter()
                .flat_map(|bucket| {
                    let out = image.compile_bucket(
                        cx,
                        reached,
                        bucket,
                        cfg,
                        (true, allow_singleton),
                        pass,
                    );
                    image.bodies.trim_transient();
                    out
                })
                .collect()
        })
    }

    fn compile_parallel(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        initial_shards: &[Vec<Vec<usize>>],
        plan: Plan,
    ) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        let mut current_shards = initial_shards.to_vec();
        let jobs = crate::jobs();
        let min_parallel = if self.backend.is_aot() {
            AOT_MIN_PARALLEL
        } else {
            MIN_PARALLEL
        };
        let mut deferred = Vec::new();
        for wave in 0..MAX_PARALLEL_WAVES {
            let forks: Vec<Context<'_>> = current_shards.iter().map(|_| cx.fork()).collect();
            let outputs = self.run_shards(forks, &current_shards, |fork, shard| {
                self.worker_loop(fork, reached, shard, plan)
            });
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
            if deferred.is_empty()
                || eval_requests.is_empty()
                || wave + 1 == MAX_PARALLEL_WAVES
                || deferred.len() < min_parallel
            {
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
            errors.extend(self.compile_buckets(
                cx,
                reached,
                &bucket_work(reached, &deferred),
                plan,
            ));
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

    fn lost(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        shard: &[Vec<usize>],
        reason: &str,
    ) -> Vec<(usize, String)> {
        let lost = shard.iter().flatten().map(|&index| {
            if self.backend.is_jit() {
                self.slot(index).store(0, Ordering::Release);
            }
            (
                index,
                format!(
                    "{}: compile worker panicked: {reason}",
                    cx.name(reached.list[index].0.0)
                ),
            )
        });
        lost.collect()
    }

    /// Runs `work` on each shard in a scoped thread of its own with one of `forks`, returning each thread's value or its panic payload message.
    pub(super) fn run_shards<'a, T: Send>(
        &self,
        forks: Vec<Context<'a>>,
        shards: &[Vec<Vec<usize>>],
        work: impl Fn(Context<'a>, &[Vec<usize>]) -> T + Sync,
    ) -> Vec<Result<T, String>> {
        let work = &work;
        crate::sync::scope(|scope| {
            let handles: Vec<_> = forks
                .into_iter()
                .zip(shards)
                .map(|(fork, shard)| scope.spawn(move || work(fork, shard)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(Self::panic_message))
                .collect()
        })
    }

    fn worker_loop(
        &self,
        mut cx: Context<'_>,
        reached: &Reached,
        shard: &[Vec<usize>],
        plan: Plan,
    ) -> WorkerOutput {
        let mut batch = WorkerBatch::new(plan.0);
        for bucket in shard {
            self.compile_worker_bucket(&mut cx, reached, bucket, plan, &mut batch);
        }
        batch.into_output()
    }

    fn compile_worker_bucket(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        bucket: &[usize],
        (cfg, allow_singleton): Plan,
        batch: &mut WorkerBatch,
    ) {
        let mark = batch.pass.records.len();
        let frozen_mark = batch.pass.frozen_stencils.len();
        let failed = self.compile_bucket(
            cx,
            reached,
            bucket,
            cfg,
            (false, allow_singleton),
            &mut batch.pass,
        );
        let reqs = cx.take_deferred_evals();
        if reqs.is_empty() {
            batch.errors.extend(failed);
        } else {
            batch.pass.records.truncate(mark);
            batch.pass.frozen_stencils.truncate(frozen_mark);
            batch.deferred.extend_from_slice(bucket);
            batch.eval_requests.extend(reqs);
        }
        self.bodies.trim_transient();
    }

    pub(super) fn prepare_work(
        &self,
        cx: &mut Context<'_>,
        reached: &Reached,
        indices: &[usize],
        (errors, mut memory): (&mut Vec<(usize, String)>, Option<&mut Memory>),
    ) -> Vec<usize> {
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

    pub(super) fn bind_foreign_statics(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        indices: &[usize],
    ) {
        let jit = self.backend.is_jit();
        let mut cells_batch = Vec::new();
        let mut aot_batch = Vec::new();
        for (index, ((item, _), typed)) in
            indices.iter().map(|&index| (index, &reached.list[index]))
        {
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
                aot_batch.push((
                    index,
                    image_aot::AotStatic::Foreign {
                        weak: is_weak,
                        host,
                    },
                ));
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
