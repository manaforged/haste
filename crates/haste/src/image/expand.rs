//! Running procedural macros in forked children: each job calls an entry of haste's `proc_macro` crate on input text and sends back the output text, and each function compiles on its first call.

use std::io::Write;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};

use super::Image;
use super::lazy::{Answerer, Jit, Driver, Ended, serve};
use crate::macros::Compilation;

/// One macro call: the declared `proc_macro` entry function, the declared macro function it runs, the one or two input texts, and its group: the calls of one macro crate from one crate, which run in order in one child so the macro's statics carry from call to call as in one compiler process.
pub struct Job<'a> {
    pub entry: usize,
    pub run: usize,
    pub texts: Vec<&'a [u8]>,
    pub group: (usize, u32),
}

type One = extern "C" fn(usize, *const u8, usize, *mut usize);
type Two = extern "C" fn(usize, *const u8, usize, *const u8, usize, *mut usize);

/// Runs the jobs in children forked on `demand`, one child per group and at most one per core, answering what inference asks with `ask`, and gives each job's output text or why it has none, or `None` when it did not run; the jobs after a crash run in a new child.
pub fn run_macros(compilation: &mut Compilation, demand: &mut Jit, jobs: &[Job<'_>], ask: Answerer<'_>) -> Result<Vec<Option<Result<Vec<u8>, String>>>> {
    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut groups = rustc_hash::FxHashMap::default();
    for (at, job) in jobs.iter().enumerate() {
        let part = *groups.entry(job.group).or_insert_with(|| {
            parts.push(Vec::new());
            parts.len() - 1
        });
        parts[part].push(at);
    }
    let mut shares = Shares { jobs, next: vec![0; parts.len()], parts, running: Vec::new(), results: jobs.iter().map(|_| None).collect() };
    serve(compilation, ask, demand, &mut shares)?;
    Ok(shares.results)
}

/// The jobs of each group in order, the next job of each group, the child running each group, and each job's result.
struct Shares<'j, 'a> {
    jobs: &'j [Job<'a>],
    parts: Vec<Vec<usize>>,
    next: Vec<usize>,
    running: Vec<(usize, usize)>,
    results: Vec<Option<Result<Vec<u8>, String>>>,
}

impl Shares<'_, '_> {
    /// The indexes of the jobs of group `part` from its next job on.
    fn left(&self, part: usize) -> impl Iterator<Item = usize> {
        self.parts[part].iter().copied().skip(self.next[part])
    }
}

impl Driver for Shares<'_, '_> {
    fn start(&mut self, demand: &mut Jit) -> Result<()> {
        for part in 0..self.parts.len() {
            if self.running.len() >= crate::jobs() {
                break;
            }
            if self.running.iter().any(|&(_, running)| running == part) || self.left(part).next().is_none() {
                continue;
            }
            let todo: Vec<&Job<'_>> = self.left(part).map(|at| &self.jobs[at]).collect();
            let id = demand.fork(true, |image, out| out.map_or(1, |out| i32::from(image.macros_in_child(&todo, out).is_err())))?;
            self.running.push((id, part));
        }
        Ok(())
    }

    fn ended(&mut self, id: usize, ended: Ended) -> Result<()> {
        let Some(at) = self.running.iter().position(|&(running, _)| running == id) else { return Ok(()) };
        let (_, part) = self.running.swap_remove(at);
        let left: Vec<usize> = self.left(part).collect();
        let outputs = frames(&ended.bytes);
        let done = outputs.len();
        for (job, output) in left.iter().zip(outputs) {
            self.results[*job] = Some(Ok(output));
        }
        self.next[part] += done;
        if let Some(&job) = left.get(done) {
            if ended.status == 0 {
                bail!("the macro child exited before its last job");
            }
            self.results[job] = Some(Err(ended.failure.map_or_else(|| format!("the procedural macro crashed with status {}", ended.status), |(error, _)| error)));
            self.next[part] += 1;
        }
        Ok(())
    }
}

impl Image {
    fn macros_in_child(&self, jobs: &[&Job<'_>], mut out: std::fs::File) -> std::io::Result<()> {
        for job in jobs {
            let (entry, run) = (self.slot(job.entry).load(Ordering::Acquire), self.slot(job.run).load(Ordering::Acquire));
            let mut result = [0usize; 2];
            match job.texts[..] {
                [one] => {
                    // SAFETY: entry is the code of `__haste_expand1`, or a stub that compiles it, whose native signature is this one, and run is the code of a `fn(TokenStream) -> TokenStream`.
                    let call: One = unsafe { std::mem::transmute::<usize, One>(entry) };
                    call(run, one.as_ptr(), one.len(), result.as_mut_ptr());
                }
                [attr, item] => {
                    // SAFETY: entry is the code of `__haste_expand2`, or a stub that compiles it, whose native signature is this one, and run is the code of a `fn(TokenStream, TokenStream) -> TokenStream`.
                    let call: Two = unsafe { std::mem::transmute::<usize, Two>(entry) };
                    call(run, attr.as_ptr(), attr.len(), item.as_ptr(), item.len(), result.as_mut_ptr());
                }
                _ => return Err(std::io::Error::other("a macro job takes one or two texts")),
            }
            // SAFETY: the entry wrote the address and length of the output text it leaked into result.
            let text = unsafe { std::slice::from_raw_parts(result[0] as *const u8, result[1]) };
            out.write_all(&(text.len() as u64).to_le_bytes())?;
            out.write_all(text)?;
        }
        Ok(())
    }
}

fn frames(mut bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut found = Vec::new();
    while let Some((head, rest)) = bytes.split_first_chunk::<8>() {
        let len = u64::from_le_bytes(*head) as usize;
        if rest.len() < len {
            break;
        }
        found.push(rest[..len].to_vec());
        bytes = &rest[len..];
    }
    found
}
