//! Running procedural macros in a forked child: each job calls an entry of haste's `proc_macro` crate on input text and sends back the output text.

use std::io::{Read, Write};
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};

use super::Image;

/// One macro call: the reached `proc_macro` entry function, the reached macro function it runs, every function they reach, and the one or two input texts.
pub struct Job<'a> {
    pub entry: usize,
    pub run: usize,
    pub closure: Vec<usize>,
    pub texts: Vec<&'a [u8]>,
}

type One = extern "C" fn(usize, *const u8, usize, *mut usize);
type Two = extern "C" fn(usize, *const u8, usize, *const u8, usize, *mut usize);

impl Image {
    /// Runs each job in a forked child and gives its output text, or the status the child died with; the jobs after a crash run in a new child.
    pub fn run_macros(&self, jobs: &[Job<'_>]) -> Result<Vec<Result<Vec<u8>, String>>> {
        let mut results = Vec::new();
        while results.len() < jobs.len() {
            let (outputs, status) = self.macros_from(&jobs[results.len()..])?;
            results.extend(outputs.into_iter().map(Ok));
            if results.len() < jobs.len() {
                if status == 0 {
                    bail!("the macro child exited before its last job");
                }
                results.push(Err(format!("the procedural macro crashed with status {status}")));
            }
        }
        Ok(results)
    }

    fn macros_from(&self, jobs: &[Job<'_>]) -> Result<(Vec<Vec<u8>>, i32)> {
        std::io::stdout().flush()?;
        let (pid, mut input) = super::fork_piped(|out| self.macros_in_child(jobs, out))?;
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes)?;
        Ok((frames(&bytes), super::wait(pid)))
    }

    fn macros_in_child(&self, jobs: &[Job<'_>], mut out: std::fs::File) -> std::io::Result<()> {
        let mut filled = rustc_hash::FxHashSet::default();
        for job in jobs {
            self.fill_statics(&job.closure, &mut filled);
            let (entry, run) = (self.slot(job.entry).load(Ordering::Acquire), self.slot(job.run).load(Ordering::Acquire));
            let mut result = [0usize; 2];
            match job.texts[..] {
                [one] => {
                    // SAFETY: entry is the code of `__haste_expand1`, whose native signature is this one, and run is the code of a `fn(TokenStream) -> TokenStream`.
                    let call: One = unsafe { std::mem::transmute::<usize, One>(entry) };
                    call(run, one.as_ptr(), one.len(), result.as_mut_ptr());
                }
                [attr, item] => {
                    // SAFETY: entry is the code of `__haste_expand2`, whose native signature is this one, and run is the code of a `fn(TokenStream, TokenStream) -> TokenStream`.
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
