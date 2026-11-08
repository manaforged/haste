//! One process image: a call table with one entry per reached function, and the code it points at.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use haste_codegen::Memory;

use crate::lower::{self, Slots};
use crate::sema::Context;
use crate::sema::reach::Reached;

/// The call table and code memory of the running program.
pub struct Image {
    table: Box<[AtomicUsize]>,
    memory: Memory,
}

impl Slots for Image {
    fn entry(&self, index: usize) -> i64 {
        &self.table[index] as *const AtomicUsize as i64
    }
}

impl Image {
    /// An image with one table entry per reached function and no code yet.
    pub fn new(reached: &Reached) -> Image {
        Image { table: (0..reached.list.len()).map(|_| AtomicUsize::new(0)).collect(), memory: Memory::default() }
    }

    /// Lowers and encodes every reached function that type-checked, and returns the errors of those that did not compile, by index.
    pub fn compile(&mut self, cx: &mut Context<'_>, reached: &Reached) -> Vec<(usize, String)> {
        let mut errors = Vec::new();
        for index in 0..reached.list.len() {
            if reached.list[index].1.is_err() {
                continue;
            }
            match self.compile_one(cx, reached, index) {
                Ok(address) => self.table[index].store(address, Ordering::Release),
                Err(error) => errors.push((index, format!("{}: {error:#}", cx.name(reached.list[index].0.0)))),
            }
        }
        errors
    }

    fn compile_one(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<usize> {
        let func = lower::lower(cx, reached, self, index)?;
        let compiled = haste_codegen::compile(&func).map_err(|error| anyhow!(error))?;
        self.memory.write(&compiled.code).map_err(|error| anyhow!(error))
    }

    /// Runs reached function `index`, an `fn(i64) -> i64`, on each input in a forked child that prints `name input value` lines.
    pub fn run_case(&self, index: usize, name: &str, inputs: &[i64]) -> Result<i32> {
        let address = self.table[index].load(Ordering::Acquire);
        if address == 0 {
            bail!("{name} has no code");
        }
        // SAFETY: address is code that the encoder wrote for a native fn(i64) -> i64 signature.
        let case: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute::<usize, extern "C" fn(i64) -> i64>(address) };
        std::io::Write::flush(&mut std::io::stdout())?;
        // SAFETY: no other thread runs at this point, so the child gets a consistent copy of every lock.
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            bail!("fork failed: {}", std::io::Error::last_os_error());
        }
        if pid == 0 {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            let wrote = inputs.iter().try_for_each(|&input| writeln!(out, "{name} {input} {}", case(input)).and_then(|_| out.flush()));
            // SAFETY: _exit ends the child without running the parent's destructors.
            unsafe { libc::_exit(i32::from(wrote.is_err())) }
        }
        let mut status = 0;
        // SAFETY: pid is the child forked above, and status is a live local.
        unsafe { libc::waitpid(pid, &mut status, 0) };
        Ok(status)
    }
}
