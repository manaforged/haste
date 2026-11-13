//! Compile workers: scoped threads that each encode a fixed share of the reached functions, grouped by footprint into stencils, into thread-local code chunks merged into the image.

use std::cmp::Reverse;
use std::sync::atomic::Ordering;

use rustc_hash::FxHashMap;

use crate::lower::{self, Slots};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::stencil::DictConfig;
use super::{Image, Pass};

const MIN_PARALLEL: usize = 64;
const MAX_BUCKET: usize = 1024;

/// What one worker hands back: its pass, the errors by index, and the functions whose constant evaluations it deferred.
type WorkerOutput = (Pass, Vec<(usize, String)>, Vec<usize>);

/// How a pass compiles its buckets: the dictionary configuration, and whether a bucket of one function compiles alone.
type Plan = (DictConfig, bool);

fn bucket_work(reached: &Reached, work: &[usize]) -> Vec<Vec<usize>> {
    let mut pos_map: FxHashMap<ItemRef, usize> = FxHashMap::default();
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for &index in work {
        let item = reached.list[index].0.0;
        if let Some(&pos) = pos_map.get(&item) {
            buckets[pos].push(index);
        } else {
            pos_map.insert(item, buckets.len());
            buckets.push(vec![index]);
        }
    }
    buckets
}

fn lpt_shards(buckets: Vec<Vec<usize>>, jobs: usize) -> Vec<Vec<Vec<usize>>> {
    let mut chunks: Vec<Vec<usize>> = buckets.into_iter().flat_map(|b| if b.len() <= MAX_BUCKET { vec![b] } else { b.chunks(MAX_BUCKET).map(|c| c.to_vec()).collect() }).collect();
    chunks.sort_by_key(|b| Reverse(b.len()));
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
        if self.mode == super::ImageMode::Jit {
            work.iter().for_each(|&index| self.slot(index).store(0, Ordering::Release));
        }
        let cfg = DictConfig::from_env();
        self.pass.stencils.cfg = cfg;
        let had_templates = !self.pass.stencils.templates.is_empty();
        let plan = (cfg, !had_templates && !cfg.verify && cfg.min_fanout > 1);
        let buckets = bucket_work(reached, work);
        let jobs = crate::jobs();
        if jobs <= 1 || cfg.verify || work.len() < MIN_PARALLEL || had_templates {
            return self.compile_buckets(cx, reached, &buckets, plan);
        }
        let shards = lpt_shards(buckets, jobs);
        if shards.len() <= 1 {
            return self.compile_buckets(cx, reached, &shards.concat(), plan);
        }
        self.compile_parallel(cx, reached, &shards, plan)
    }

    fn compile_buckets(&mut self, cx: &mut Context<'_>, reached: &Reached, buckets: &[Vec<usize>], (cfg, allow_singleton): Plan) -> Vec<(usize, String)> {
        self.with_pass(|image, pass| buckets.iter().flat_map(|bucket| image.compile_bucket(cx, reached, bucket, cfg, (true, allow_singleton), pass)).collect())
    }

    fn compile_parallel(&mut self, cx: &mut Context<'_>, reached: &Reached, shards: &[Vec<Vec<usize>>], plan: Plan) -> Vec<(usize, String)> {
        let forks: Vec<Context<'_>> = shards.iter().map(|_| cx.fork()).collect();
        let outputs = self.run_workers(forks, reached, shards, plan);
        let (mut errors, mut deferred) = (Vec::new(), Vec::new());
        for (shard, output) in shards.iter().zip(outputs) {
            match output {
                Some((pass, failed, later)) => {
                    self.pass.merge(pass);
                    errors.extend(failed);
                    deferred.extend(later);
                }
                None => errors.extend(self.lost(cx, reached, shard)),
            }
        }
        deferred.sort_unstable();
        let was_parallel = std::mem::replace(&mut cx.in_parallel, false);
        errors.extend(self.compile_buckets(cx, reached, &bucket_work(reached, &deferred), plan));
        cx.in_parallel = was_parallel;
        errors.sort_by_key(|&(index, _)| index);
        errors
    }

    /// Clears the slots of a worker that panicked, whose code chunks unmapped as it unwound, and gives each of its functions an error.
    fn lost(&self, cx: &Context<'_>, reached: &Reached, shard: &[Vec<usize>]) -> Vec<(usize, String)> {
        let lost = shard.iter().flatten().map(|&index| {
            if self.mode == super::ImageMode::Jit {
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
        let (mut errors, mut deferred) = (Vec::new(), Vec::new());
        for bucket in shard {
            let mark = pass.records.len();
            let frozen_mark = pass.frozen_stencils.len();
            let failed = self.compile_bucket(&mut cx, reached, bucket, cfg, (false, allow_singleton), &mut pass);
            if cx.take_deferred_evals().is_empty() {
                errors.extend(failed);
                continue;
            }
            pass.records.truncate(mark);
            pass.frozen_stencils.truncate(frozen_mark);
            deferred.extend_from_slice(bucket);
        }
        (pass, errors, deferred)
    }
}
