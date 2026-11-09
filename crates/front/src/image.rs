//! One process image: a call table with one entry per reached function, the code it points at, and constant data.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, anyhow, bail};
use haste_codegen::Memory;

use crate::lower::{self, Slots, Word};
use crate::sema::Context;
use crate::sema::reach::Reached;

/// The call table and code memory of the running program.
pub struct Image {
    table: Box<[AtomicUsize]>,
    memory: Memory,
    data: RefCell<Vec<Box<[u128]>>>,
    tables: RefCell<HashMap<Vec<Word>, i64>>,
    fixups: RefCell<Vec<(usize, usize, usize)>>,
    cells: RefCell<HashMap<usize, i64>>,
    statics: rustc_hash::FxHashSet<usize>,
}

impl Slots for Image {
    fn entry(&self, index: usize) -> i64 {
        &self.table[index] as *const AtomicUsize as i64
    }

    fn data(&self, bytes: &[u8]) -> i64 {
        let mut copy: Box<[u128]> = vec![0; bytes.len().div_ceil(16)].into();
        // SAFETY: the buffer holds at least `bytes.len()` bytes and does not overlap `bytes`.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), copy.as_mut_ptr().cast::<u8>(), bytes.len()) };
        let address = copy.as_ptr() as i64;
        self.data.borrow_mut().push(copy);
        address
    }

    fn table(&self, words: &[Word]) -> i64 {
        if let Some(&address) = self.tables.borrow().get(words) {
            return address;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| (if let Word::Int(value) = *word { value } else { 0 }).to_le_bytes()).collect();
        let address = self.data(&bytes);
        let block = self.data.borrow().len() - 1;
        let codes = words.iter().enumerate().filter_map(|(at, word)| if let Word::Code(index) = *word { Some((block, at, index)) } else { None });
        self.fixups.borrow_mut().extend(codes);
        self.tables.borrow_mut().insert(words.to_vec(), address);
        address
    }

    fn cell(&self, index: usize, size: u32) -> i64 {
        if let Some(&address) = self.cells.borrow().get(&index) {
            return address;
        }
        let address = self.data(&vec![0; size as usize]);
        self.cells.borrow_mut().insert(index, address);
        address
    }
}

fn host_symbol(cx: &Context<'_>, item: crate::names::ItemRef) -> Result<usize, String> {
    let (src, entry) = cx.item(item);
    let name = cx.attr_value(item, "link_name").or_else(|| entry.name.map(|at| src.text(at)));
    name.and_then(crate::host::symbol).ok_or_else(|| format!("{}: no host symbol links to it", cx.name(item)))
}

impl Image {
    /// An image with one table entry per reached function and no code yet.
    pub fn new(reached: &Reached) -> Image {
        let table = (0..reached.list.len()).map(|_| AtomicUsize::new(0)).collect();
        Image { table, memory: Memory::default(), data: RefCell::default(), tables: RefCell::default(), fixups: RefCell::default(), cells: RefCell::default(), statics: Default::default() }
    }

    /// Lowers and encodes every reached function that type-checked, and returns the errors of those that did not compile, by index.
    pub fn compile(&mut self, cx: &mut Context<'_>, reached: &Reached) -> Vec<(usize, String)> {
        self.bind_foreign_statics(cx, reached);
        let mut errors = Vec::new();
        for index in 0..reached.list.len() {
            if reached.list[index].1.is_err() {
                continue;
            }
            let item = reached.list[index].0.0;
            if cx.foreign(item) {
                match host_symbol(cx, item) {
                    Ok(address) if !lower::is_static(cx, item) => self.table[index].store(address, Ordering::Release),
                    Ok(_) => {}
                    Err(error) => errors.push((index, error)),
                }
                continue;
            }
            match self.compile_one(cx, reached, index) {
                Ok(address) if lower::is_static(cx, item) => {
                    self.table[index].store(address, Ordering::Release);
                    self.statics.insert(index);
                }
                Ok(address) => self.table[index].store(address, Ordering::Release),
                Err(error) => errors.push((index, format!("{}: {error:#}", cx.name(reached.list[index].0.0)))),
            }
        }
        self.link();
        errors
    }

    fn bind_foreign_statics(&self, cx: &Context<'_>, reached: &Reached) {
        for (index, ((item, _), typed)) in reached.list.iter().enumerate() {
            if typed.is_err() || !cx.foreign(*item) || !lower::is_static(cx, *item) {
                continue;
            }
            let cell = match (host_symbol(cx, *item), cx.attr_value(*item, "linkage") == Some("extern_weak")) {
                (found, true) => self.data(&found.unwrap_or(0).to_le_bytes()),
                (Ok(address), false) => address as i64,
                (Err(_), false) => continue,
            };
            self.cells.borrow_mut().insert(index, cell);
        }
    }

    fn link(&mut self) {
        let data = self.data.get_mut();
        for &(block, at, index) in self.fixups.get_mut().iter() {
            let code = self.table[index].load(Ordering::Acquire) as u128;
            let shift = (at % 2) * 64;
            let cell = &mut data[block][at / 2];
            *cell = (*cell & !(u128::from(u64::MAX) << shift)) | (code << shift);
        }
    }

    fn compile_one(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<usize> {
        let func = lower::lower(cx, reached, self, index)?;
        let compiled = haste_codegen::compile(&func).map_err(|error| anyhow!(error))?;
        self.memory.write(&compiled.code).map_err(|error| anyhow!(error))
    }

    /// Runs reached function `index`, an `fn(i64) -> i64` that reaches the functions `closure`, on each input in a forked child that fills the statics it reaches and prints `name input value` lines.
    pub fn run_case(&self, (index, closure): (usize, &[usize]), name: &str, inputs: &[i64]) -> Result<i32> {
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
            self.fill_statics(closure);
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

    fn fill_statics(&self, closure: &[usize]) {
        for &index in closure.iter().filter(|index| self.statics.contains(index)) {
            let address = self.table[index].load(Ordering::Acquire);
            // SAFETY: address is code that the encoder wrote for a static's initializer, a native fn() that writes the static's cell.
            let fill: extern "C" fn() = unsafe { std::mem::transmute::<usize, extern "C" fn()>(address) };
            fill();
        }
    }
}
