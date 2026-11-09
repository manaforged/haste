//! Compile workers: forked copies of the compiler that each encode a share of the reached functions into the image's shared memory.

use std::io::{Read, Write};
use std::os::fd::FromRawFd;
use std::sync::atomic::Ordering;

use crate::lower::{self, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::Image;

const MIN_PARALLEL: usize = 256;

/// The number of compile workers: `HASTE_JOBS`, or one per core.
fn jobs() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    std::env::var("HASTE_JOBS").ok().and_then(|text| text.parse().ok()).unwrap_or(cores).max(1)
}

/// One forked worker: its process, the read end of its report pipe, and its share of the work.
struct Worker {
    pid: libc::pid_t,
    report: std::fs::File,
    part: usize,
}

impl Image {
    /// Gives every reached static its cell before workers fork, so that each of them writes the same address.
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

    /// Compiles every function of `work`, split across forked workers when there is enough of it, and returns the errors by index.
    pub(super) fn compile_all(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize]) -> Vec<(usize, String)> {
        let jobs = jobs();
        if jobs == 1 || work.len() < MIN_PARALLEL {
            return work.iter().filter_map(|&index| self.compile_into(cx, reached, index).err().map(|error| (index, error))).collect();
        }
        if let Err(error) = std::io::stdout().flush() {
            return work.iter().map(|&index| (index, format!("flushing stdout before compile workers fork: {error}"))).collect();
        }
        let mut workers = Vec::new();
        for part in 0..jobs {
            match self.fork_worker(cx, reached, work, (jobs, part)) {
                Ok(worker) => workers.push(worker),
                Err(error) => return work.iter().map(|&index| (index, format!("compile worker {part}: {error}"))).collect(),
            }
        }
        let errors = workers.into_iter().flat_map(|worker| self.collect(worker, work, jobs)).collect();
        self.memory.close();
        self.shared.get_mut().close();
        errors
    }

    fn compile_into(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<(), String> {
        match self.compile_one(cx, reached, index) {
            Ok(address) => {
                self.slot(index).store(address, Ordering::Release);
                Ok(())
            }
            Err(error) => Err(format!("{}: {error:#}", cx.name(reached.list[index].0.0))),
        }
    }

    fn fork_worker(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], (jobs, part): (usize, usize)) -> Result<Worker, String> {
        let mut ends = [0; 2];
        // SAFETY: ends is a live two-element array for the new descriptors.
        if unsafe { libc::pipe(ends.as_mut_ptr()) } != 0 {
            return Err(format!("pipe: {}", std::io::Error::last_os_error()));
        }
        // SAFETY: no other thread runs at this point, so the child gets a consistent copy of every lock.
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(format!("fork: {}", std::io::Error::last_os_error()));
        }
        if pid == 0 {
            // SAFETY: ends[0] is the read end this child does not use, and ends[1] is the write end it owns from here on.
            let report = unsafe {
                libc::close(ends[0]);
                std::fs::File::from_raw_fd(ends[1])
            };
            let wrote = self.work_share(cx, reached, work, (jobs, part), report);
            // SAFETY: _exit ends the child without running the parent's destructors.
            unsafe { libc::_exit(i32::from(wrote.is_err())) }
        }
        // SAFETY: ends[1] is the write end that only the child uses, and ends[0] is the read end this process owns from here on.
        let report = unsafe {
            libc::close(ends[1]);
            std::fs::File::from_raw_fd(ends[0])
        };
        Ok(Worker { pid, report, part })
    }

    fn work_share(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], (jobs, part): (usize, usize), mut report: std::fs::File) -> std::io::Result<()> {
        self.memory.split(jobs, part);
        self.shared.get_mut().split(jobs, part);
        let mut text = String::new();
        for &index in work.iter().skip(part).step_by(jobs) {
            if let Err(error) = self.compile_into(cx, reached, index) {
                text.push_str(&format!("E {index} {}\n", error.replace('\n', " ")));
            }
        }
        for (address, index) in self.fixups.get_mut().iter() {
            text.push_str(&format!("F {address} {index}\n"));
        }
        report.write_all(text.as_bytes())
    }

    fn collect(&mut self, mut worker: Worker, work: &[usize], jobs: usize) -> Vec<(usize, String)> {
        let mut text = String::new();
        let read = worker.report.read_to_string(&mut text);
        let mut status = 0;
        // SAFETY: pid is a child this process forked, and status is a live local.
        unsafe { libc::waitpid(worker.pid, &mut status, 0) };
        let mut errors = Vec::new();
        for line in text.lines() {
            let mut parts = line.splitn(3, ' ');
            match (parts.next(), parts.next().and_then(|index| index.parse::<usize>().ok()), parts.next()) {
                (Some("E"), Some(index), Some(error)) => errors.push((index, error.to_owned())),
                (Some("F"), Some(address), Some(index)) => self.fixups.get_mut().extend(index.parse().ok().map(|index| (address, index))),
                _ => {}
            }
        }
        if status != 0 || read.is_err() {
            let lost = work.iter().skip(worker.part).step_by(jobs).filter(|&&index| self.slot(index).load(Ordering::Acquire) == 0);
            errors.extend(lost.map(|&index| (index, format!("compile worker {} stopped with status {status}", worker.part))));
        }
        errors
    }
}
