//! Compile workers: scoped threads that pull batches of reached functions and encode them into thread-local code chunks merged into the image.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use haste_codegen::Memory;
use rustc_hash::FxHashSet;

use crate::lower::{self, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::Image;

const MIN_PARALLEL: usize = 64;
const BATCH: usize = 16;

type WorkerOutput = (Memory, Vec<(usize, String)>, Vec<usize>);

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

    /// Compiles every function of `work`, split across scoped worker threads when there is enough of it, and returns the errors by index.
    pub(super) fn compile_all(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) -> Vec<(usize, String)> {
        work.iter().for_each(|&index| self.slot(index).store(0, Ordering::Release));
        let jobs = crate::jobs().min(work.len().div_ceil(BATCH));
        if jobs <= 1 || work.len() < MIN_PARALLEL {
            return work.iter().filter_map(|&index| self.compile_into(cx, reached, index).err()).collect();
        }
        let forks: Vec<Context<'_>> = (0..jobs).map(|_| cx.fork()).collect();
        let next = AtomicUsize::new(0);
        let outputs = self.run_workers(forks, reached, work, &next);
        let (mut errors, mut deferred, mut panicked) = (Vec::new(), Vec::new(), false);
        for output in outputs {
            match output {
                Some((local, failed, later)) => {
                    self.memory.merge(local);
                    errors.extend(failed);
                    deferred.extend(later);
                }
                None => panicked = true,
            }
        }
        deferred.sort_unstable();
        let was_parallel = std::mem::replace(&mut cx.in_parallel, false);
        for index in deferred {
            if let Err(failed) = self.compile_into(cx, reached, index) {
                errors.push(failed);
            }
        }
        cx.in_parallel = was_parallel;
        if panicked {
            self.record_panicked(cx, reached, work, &mut errors);
        }
        errors.sort_by_key(|&(index, _)| index);
        errors
    }

    fn record_panicked(&self, cx: &Context<'_>, reached: &Reached, work: &[usize], errors: &mut Vec<(usize, String)>) {
        let recorded: FxHashSet<usize> = errors.iter().map(|&(index, _)| index).collect();
        let lost = work.iter().copied().filter(|&index| self.slot(index).load(Ordering::Acquire) == 0 && !recorded.contains(&index));
        errors.extend(lost.map(|index| (index, format!("{}: compile worker panicked", cx.name(reached.list[index].0.0)))));
    }

    fn compile_into(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<(), (usize, String)> {
        let result = self.compile_one(cx, reached, index);
        self.store_compiled(cx, reached, index, result)
    }

    fn store_compiled(&self, cx: &Context<'_>, reached: &Reached, index: usize, result: Result<usize>) -> Result<(), (usize, String)> {
        match result {
            Ok(address) => {
                self.slot(index).store(address, Ordering::Release);
                Ok(())
            }
            Err(error) => Err((index, format!("{}: {error:#}", cx.name(reached.list[index].0.0)))),
        }
    }

    fn run_workers<'a>(&self, forks: Vec<Context<'a>>, reached: &Reached, work: &[usize], next: &AtomicUsize) -> Vec<Option<WorkerOutput>> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = forks.into_iter().map(|fork| scope.spawn(move || self.worker_loop(fork, reached, work, next))).collect();
            handles.into_iter().map(|handle| handle.join().map_or(None, Some)).collect()
        })
    }

    fn worker_loop(&self, mut cx: Context<'_>, reached: &Reached, work: &[usize], next: &AtomicUsize) -> WorkerOutput {
        let (mut memory, mut errors, mut deferred) = (Memory::default(), Vec::new(), Vec::new());
        let batches = std::iter::from_fn(|| {
            let start = next.fetch_add(BATCH, Ordering::Relaxed);
            (start < work.len()).then(|| &work[start..(start + BATCH).min(work.len())])
        });
        for &index in batches.flatten() {
            let encoded = self.encode_one(&mut cx, reached, index);
            if !cx.take_deferred_evals().is_empty() {
                deferred.push(index);
                continue;
            }
            let result = encoded.and_then(|code| Ok(memory.write(code.bytes())?));
            if let Err(failed) = self.store_compiled(&cx, reached, index, result) {
                errors.push(failed);
            }
        }
        (memory, errors, deferred)
    }
}
