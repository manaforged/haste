//! Compile workers: forked copies of the compiler that each encode a share of the unique function footprints into the image's shared memory.

use std::cmp::Reverse;
use std::io::{Read, Write};
use std::sync::atomic::Ordering;

use rustc_hash::FxHashMap;

use crate::lower::{self, Slots};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::Image;
use super::stencil::DictConfig;

const MIN_PARALLEL: usize = 256;
const MAX_BUCKET: usize = 1024;

struct Worker {
    pid: libc::pid_t,
    report: std::fs::File,
    part: usize,
}

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
    let mut chunks: Vec<Vec<usize>> = buckets
        .into_iter()
        .flat_map(|b| if b.len() <= MAX_BUCKET { vec![b] } else { b.chunks(MAX_BUCKET).map(|c| c.to_vec()).collect() })
        .collect();
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
    pub(super) fn place_statics(&self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) {
        for &index in work {
            let Ok(typed) = &reached.list[index].1 else { continue };
            if !lower::is_static(cx, reached.list[index].0.0) {
                continue;
            }
            if let Some(size) = typed.ret.and_then(|ty| cx.layout(ty).ok()).map(|layout| layout.size.bytes() as u32) {
                self.cell(index, size);
            }
        }
    }

    pub(super) fn compile_all(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) -> Vec<(usize, String)> {
        if work.is_empty() {
            return Vec::new();
        }
        let cfg = DictConfig::from_env();
        self.stencils.cfg = cfg;
        let had_templates = !self.stencils.templates.is_empty();
        let allow_singleton = !had_templates && !cfg.verify && cfg.min_fanout > 1;
        let buckets = bucket_work(reached, work);
        let jobs = crate::jobs();
        if jobs <= 1 || cfg.verify || work.len() < MIN_PARALLEL || had_templates {
            return buckets
                .iter()
                .flat_map(|bucket| self.compile_bucket(cx, reached, bucket, cfg, (true, allow_singleton)))
                .collect();
        }
        let shards = lpt_shards(buckets, jobs);
        if shards.len() <= 1 {
            return shards
                .iter()
                .flatten()
                .flat_map(|bucket| self.compile_bucket(cx, reached, bucket, cfg, (true, allow_singleton)))
                .collect();
        }
        self.compile_parallel(cx, reached, work, &shards, (cfg, allow_singleton))
    }

    fn compile_parallel(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], shards: &[Vec<Vec<usize>>], (cfg, allow_singleton): (DictConfig, bool)) -> Vec<(usize, String)> {
        let fail_all = |error: String| -> Vec<(usize, String)> {
            work.iter().map(|&idx| (idx, error.clone())).collect()
        };
        if let Err(error) = std::io::stdout().flush() {
            return fail_all(format!("flushing stdout before compile workers fork: {error}"));
        }
        if let Err(error) = self.open_space() {
            return fail_all(format!("mapping memory for compile workers: {error}"));
        }
        work.iter().for_each(|&idx| self.slot(idx).store(0, Ordering::Release));
        let jobs = shards.len();
        let mut workers = Vec::new();
        for (part, shard) in shards.iter().enumerate() {
            match self.fork_worker(cx, reached, shard, (cfg, allow_singleton), (jobs, part)) {
                Ok(worker) => workers.push(worker),
                Err(error) => return fail_all(format!("compile worker {part}: {error}")),
            }
        }
        let errors = workers
            .into_iter()
            .flat_map(|worker| {
                let part = worker.part;
                self.collect(worker, &shards[part])
            })
            .collect();
        self.memory.close();
        self.shared.get_mut().close();
        self.state.get_mut().close();
        errors
    }

    fn open_space(&mut self) -> Result<(), haste_codegen::Error> {
        self.memory.reserve(super::CODE_BYTES)?;
        self.shared.get_mut().renew()?;
        self.state.get_mut().renew()
    }

    fn fork_worker(&mut self, cx: &mut Context<'_>, reached: &Reached, shard: &[Vec<usize>], (cfg, allow_singleton): (DictConfig, bool), (jobs, part): (usize, usize)) -> Result<Worker, String> {
        let (pid, report) = super::fork_piped(|report| self.work_share(cx, reached, shard, (cfg, allow_singleton), (jobs, part), report)).map_err(|error| format!("{error:#}"))?;
        Ok(Worker { pid, report, part })
    }

    fn work_share(&mut self, cx: &mut Context<'_>, reached: &Reached, shard: &[Vec<usize>], (cfg, allow_singleton): (DictConfig, bool), (jobs, part): (usize, usize), mut report: std::fs::File) -> std::io::Result<()> {
        self.memory.split(jobs, part);
        self.shared.get_mut().split(jobs, part);
        self.state.get_mut().split(jobs, part);
        let inherited = self.fixups.get_mut().len();
        let mut text = String::new();
        for bucket in shard {
            for (index, error) in self.compile_bucket(cx, reached, bucket, cfg, (false, allow_singleton)) {
                let clean = error.replace('\n', " ");
                text.push_str(&format!("E {index} {clean}\n"));
            }
        }
        for &index in shard.iter().flatten() {
            if index < self.sizes.len() && (self.sizes[index] > 0 || self.times[index] > 0) {
                text.push_str(&format!("S {index} {} {}\n", self.sizes[index], self.times[index]));
            }
        }
        for (address, index) in self.fixups.get_mut().iter().skip(inherited) {
            text.push_str(&format!("F {address} {index}\n"));
        }
        report.write_all(text.as_bytes())
    }

    fn collect(&mut self, mut worker: Worker, shard: &[Vec<usize>]) -> Vec<(usize, String)> {
        let mut text = String::new();
        let read = worker.report.read_to_string(&mut text);
        let status = super::wait(worker.pid);
        let mut errors = Vec::new();
        for line in text.lines() {
            let mut parts = line.splitn(3, ' ');
            match (parts.next(), parts.next().and_then(|index| index.parse::<usize>().ok()), parts.next()) {
                (Some("E"), Some(index), Some(error)) => errors.push((index, error.to_owned())),
                (Some("F"), Some(address), Some(index)) => self.fixups.get_mut().extend(index.parse().ok().map(|index| (address, index))),
                (Some("S"), Some(index), Some(rest)) if index < self.sizes.len() => {
                    let mut nums = rest.split_whitespace();
                    if let (Some(size), Some(ns)) = (nums.next().and_then(|s| s.parse().ok()), nums.next().and_then(|s| s.parse().ok())) {
                        self.sizes[index] = size;
                        self.times[index] = ns;
                    }
                }
                _ => {}
            }
        }
        if status != 0 || read.is_err() {
            let lost = shard.iter().flatten().copied().filter(|&idx| self.slot(idx).load(Ordering::Acquire) == 0);
            errors.extend(lost.map(|index| (index, format!("compile worker {} stopped with status {status}", worker.part))));
        }
        errors
    }
}
