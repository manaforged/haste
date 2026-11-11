//! Compile workers: forked copies of the compiler that each encode a share of the reached functions into the image's shared memory.

use std::io::{Read, Write};
use std::sync::atomic::Ordering;

use crate::lower::{self, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;

use super::Image;

const MIN_PARALLEL: usize = 256;

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
        let jobs = crate::jobs();
        if jobs == 1 || work.len() < MIN_PARALLEL {
            return work.iter().filter_map(|&index| self.compile_into(cx, reached, index).err().map(|error| (index, error))).collect();
        }
        let fail_all = |error: String| -> Vec<(usize, String)> { work.iter().map(|&index| (index, error.clone())).collect() };
        if let Err(error) = std::io::stdout().flush() {
            return fail_all(format!("flushing stdout before compile workers fork: {error}"));
        }
        if let Err(error) = self.open_space() {
            return fail_all(format!("mapping memory for compile workers: {error}"));
        }
        work.iter().for_each(|&index| self.slot(index).store(0, Ordering::Release));
        let mut workers = Vec::new();
        for part in 0..jobs {
            match self.fork_worker(cx, reached, work, (jobs, part)) {
                Ok(worker) => workers.push(worker),
                Err(error) => return fail_all(format!("compile worker {part}: {error}")),
            }
        }
        let errors = workers.into_iter().flat_map(|worker| self.collect(worker, work, jobs)).collect();
        self.memory.close();
        self.shared.get_mut().close();
        self.state.get_mut().close();
        errors
    }

    /// Maps fresh code, shared and state space for the workers about to fork, since what earlier workers split stays closed.
    fn open_space(&mut self) -> Result<(), haste_codegen::Error> {
        self.memory.reserve(super::CODE_BYTES)?;
        self.shared.get_mut().renew()?;
        self.state.get_mut().renew()
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
        let (pid, report) = super::fork_piped(|report| self.work_share(cx, reached, work, (jobs, part), report)).map_err(|error| format!("{error:#}"))?;
        Ok(Worker { pid, report, part })
    }

    fn work_share(&mut self, cx: &mut Context<'_>, reached: &Reached, work: &[usize], (jobs, part): (usize, usize), mut report: std::fs::File) -> std::io::Result<()> {
        self.memory.split(jobs, part);
        self.shared.get_mut().split(jobs, part);
        self.state.get_mut().split(jobs, part);
        let inherited = self.fixups.get_mut().len();
        let mut text = String::new();
        for &index in work.iter().skip(part).step_by(jobs) {
            match self.compile_into(cx, reached, index) {
                Ok(()) => text.push_str(&format!("S {index} {} {}\n", self.sizes[index], self.times[index])),
                Err(error) => text.push_str(&format!("E {index} {}\n", error.replace('\n', " "))),
            }
        }
        for (address, index) in self.fixups.get_mut().iter().skip(inherited) {
            text.push_str(&format!("F {address} {index}\n"));
        }
        report.write_all(text.as_bytes())
    }

    fn collect(&mut self, mut worker: Worker, work: &[usize], jobs: usize) -> Vec<(usize, String)> {
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
            let lost = work.iter().skip(worker.part).step_by(jobs).filter(|&&index| self.slot(index).load(Ordering::Acquire) == 0);
            errors.extend(lost.map(|&index| (index, format!("compile worker {} stopped with status {status}", worker.part))));
        }
        errors
    }
}
