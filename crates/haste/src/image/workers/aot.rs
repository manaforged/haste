use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use rustc_hash::FxHashMap;

use crate::image::aot::{AotQueue, AotWork, collect_warm_callees, lpt_chunks};
use crate::image::stencil::DictConfig;
use crate::image::{Image, ImageMode};
use crate::lower::{self, Slots};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

use super::{
    AOT_MIN_PARALLEL, MAX_PARALLEL_WAVES, Plan, WorkerBatch, WorkerOutput, bucket_work,
    prime_eval_requests,
};

const AOT_MAX_BUCKET: usize = 64;

fn seed_aot_roots(reached: &Reached, roots: &[usize]) -> (Vec<AtomicBool>, Vec<usize>) {
    let claimed: Vec<AtomicBool> = (0..reached.list.len())
        .map(|_| AtomicBool::new(false))
        .collect();
    let mut seed = Vec::new();
    for &root in roots {
        if root < claimed.len() && !claimed[root].swap(true, Ordering::AcqRel) {
            seed.push(root);
        }
    }
    seed.sort_unstable();
    (claimed, seed)
}

impl Image {
    pub(in crate::image) fn build_aot_streaming(
        cx: &mut Context<'_>,
        reached: &Reached,
        roots: &[usize],
    ) -> Result<(Image, FxHashMap<usize, String>)> {
        let mut image = Self::create(reached, ImageMode::Aot, (0, 0, 0))?;
        let cfg = DictConfig::from_env();
        image.pass.stencils.cfg = cfg;
        image.persist_inline(cfg);
        let aot_cache = (!cfg.verify)
            .then(|| crate::image::aot::cache::AotCache::open(cx, reached, roots, cfg))
            .flatten()
            .map(Arc::new);
        if let Some(aot) = image.aot() {
            lock(aot).cache = aot_cache.clone();
        }
        let plan = (cfg, !cfg.verify && cfg.min_fanout > 1);
        let (claimed, seed) = seed_aot_roots(reached, roots);
        let mut broken = FxHashMap::default();
        let mut init_errors = Vec::new();
        let work = image.prepare_work(cx, reached, &seed, (&mut init_errors, None));
        for (idx, err) in init_errors {
            broken.insert(idx, err);
        }
        let items: Box<[ItemRef]> = reached.list.iter().map(|((item, _), _)| *item).collect();
        let queue = AotQueue::new(
            claimed,
            items,
            AOT_MAX_BUCKET,
            lpt_chunks(bucket_work(reached, &work), AOT_MAX_BUCKET),
        );
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
        let cache_to_flush = if let Some(aot) = image.aot() {
            let mut state = lock(aot);
            state.frontier = None;
            state.stencils.clear();
            state.cache.take()
        } else {
            None
        };
        drop(aot_cache);
        if let Some(cache) = cache_to_flush
            && let Ok(cache) = Arc::try_unwrap(cache)
            && let Some(aot) = image.aot()
        {
            let state = lock(aot);
            cache.flush(&state, cx, reached, broken.is_empty());
        }
        image.bodies.reset();
        image.pass.stencils.clear();
        Ok((image, broken))
    }

    fn drain_aot_serial(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        queue: &AotQueue,
        plan: Plan,
        broken: &mut FxHashMap<usize, String>,
    ) {
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

    fn run_aot_workers(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        queue: &AotQueue,
        jobs: usize,
        plan: Plan,
    ) -> Vec<Option<WorkerOutput>> {
        let forks: Vec<Context<'_>> = (0..jobs).map(|_| cx.fork()).collect();
        crate::sync::scope(|scope| {
            let handles: Vec<_> = forks
                .into_iter()
                .map(|fork| scope.spawn(move || self.aot_worker_loop(fork, reached, queue, plan)))
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().map_or(None, Some))
                .collect()
        })
    }

    fn aot_worker_loop(
        &self,
        mut cx: Context<'_>,
        reached: &Reached,
        queue: &AotQueue,
        plan: Plan,
    ) -> WorkerOutput {
        let cache = self.aot_cache();
        let has_bundle = cache.as_ref().is_some_and(|c| c.has_loaded_bundle());
        let mut out = WorkerBatch::new(plan.0);
        while let Some(item) = queue.pop() {
            match item {
                AotWork::Compile(batch, _guard) => {
                    if self.allow_inline() && !has_bundle && queue.want_warm() {
                        queue.push_warm(collect_warm_callees(&cx, reached, queue, &batch));
                    }
                    let mut batch_errors = Vec::new();
                    let work =
                        self.prepare_work(&mut cx, reached, &batch, (&mut batch_errors, None));
                    out.errors.extend(batch_errors);
                    if !work.is_empty() {
                        self.compile_worker_bucket(&mut cx, reached, &work, plan, &mut out);
                    }
                }
                AotWork::Warm(idx) => {
                    if let Some(cache) = &cache {
                        cache.note_warm();
                    }
                    lower::warm_inline(&mut cx, reached, self, idx);
                    let _ = cx.take_deferred_evals();
                    self.bodies.trim_transient();
                }
            }
        }
        out.into_output()
    }
}
