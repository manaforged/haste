//! One process image: a call table with one entry per reached function, the code it points at, and constant data.

mod expand;
mod lazy;
mod script;
mod stub;
mod workers;

pub use expand::{Job, run_macros};
pub use lazy::{Answerer, Demand, Driver, Ended, entries, run_on_demand, serve};
pub use script::Script;

use std::cell::RefCell;
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap as HashMap;

use anyhow::{Result, anyhow, bail};
use haste_codegen::{Memory, Shared};

use crate::lower::{self, Slots, Word};
use crate::sema::Context;
use crate::sema::reach::Reached;

const SHARED_BYTES: usize = 4 << 30;
const STATE_BYTES: usize = 1 << 30;
const CODE_BYTES: usize = 1 << 30;
const TABLE_SLOTS: usize = 1 << 22;

/// The call table, code memory and data of the running program, in memory that compile workers forked from this process share, and the cells the program writes, in memory each run holds its own copy of.
pub struct Image {
    table: usize,
    capacity: usize,
    memory: Memory,
    stubs: Memory,
    shared: RefCell<Shared>,
    state: RefCell<Shared>,
    tables: RefCell<HashMap<Vec<Word>, i64>>,
    fixups: RefCell<Vec<(usize, usize)>>,
    cells: RefCell<HashMap<usize, i64>>,
    wanted: RefCell<Vec<crate::sema::reach::Instance>>,
}

impl Slots for Image {
    fn entry(&self, index: usize) -> i64 {
        self.slot(index) as *const AtomicUsize as i64
    }

    fn data(&self, bytes: &[u8]) -> i64 {
        let address = self.shared.borrow_mut().alloc(bytes.len()).unwrap_or_else(|error| panic!("{error}"));
        // SAFETY: alloc handed out `bytes.len()` writable bytes at address that nothing else uses, and they do not overlap `bytes`.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
        address as i64
    }

    fn table(&self, words: &[Word]) -> i64 {
        if let Some(&address) = self.tables.borrow().get(words) {
            return address;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|word| (if let Word::Int(value) = *word { value } else { 0 }).to_le_bytes()).collect();
        let address = self.data(&bytes);
        let codes = words.iter().enumerate().filter_map(|(at, word)| if let Word::Code(index) = *word { Some((address as usize + at * 8, index)) } else { None });
        self.fixups.borrow_mut().extend(codes);
        self.tables.borrow_mut().insert(words.to_vec(), address);
        address
    }

    fn zeroed(&self, len: usize) -> i64 {
        self.state.borrow_mut().alloc(len).unwrap_or_else(|error| panic!("{error}")) as i64
    }

    fn want(&self, instance: crate::sema::reach::Instance) {
        self.wanted.borrow_mut().push(instance);
    }

    fn cell(&self, index: usize, size: u32) -> i64 {
        if let Some(&address) = self.cells.borrow().get(&index) {
            return address;
        }
        let address = self.zeroed(size as usize + 1);
        self.cells.borrow_mut().insert(index, address);
        address
    }
}

fn host_symbol(cx: &Context<'_>, item: crate::names::ItemRef) -> Result<usize, String> {
    let (src, entry) = cx.item(item);
    let name = cx.attr_value(item, "link_name").or_else(|| entry.name.map(|at| src.text(at)));
    name.and_then(crate::host::symbol).ok_or_else(|| format!("{}: no host symbol links to it", cx.name(item)))
}

/// Whether the foreign item `item` links weakly, so a missing symbol reads as null.
fn weak(cx: &Context<'_>, item: crate::names::ItemRef) -> bool {
    cx.attr_value(item, "linkage") == Some("extern_weak")
}

impl Image {
    /// An image with one table entry per reached function and no code yet.
    pub fn new(reached: &Reached) -> Result<Image> {
        let mut shared = Shared::new(SHARED_BYTES).map_err(|error| anyhow!(error))?;
        let capacity = reached.list.len().max(TABLE_SLOTS);
        let table = shared.alloc(capacity * size_of::<AtomicUsize>()).map_err(|error| anyhow!(error))?;
        let memory = Memory::default();
        let state = Shared::private(STATE_BYTES).map_err(|error| anyhow!(error))?;
        Ok(Image { table, capacity, memory, stubs: Memory::default(), shared: RefCell::new(shared), state: RefCell::new(state), tables: RefCell::default(), fixups: RefCell::default(), cells: RefCell::default(), wanted: RefCell::default() })
    }

    fn slot(&self, index: usize) -> &AtomicUsize {
        assert!(index < self.capacity, "call table index {index} past its {} slots", self.capacity);
        // SAFETY: index is below capacity, and new allocated capacity zeroed, aligned AtomicUsize slots at table, in a mapping that lives as long as the process.
        unsafe { &*(self.table as *const AtomicUsize).add(index) }
    }

    /// An image of every function in `reached`, compiled, with why each one that did not compile failed.
    pub fn build(cx: &mut Context<'_>, reached: &Reached) -> Result<(Image, HashMap<usize, String>)> {
        let mut image = Image::new(reached)?;
        let all: Vec<usize> = (0..reached.list.len()).collect();
        let broken = image.compile(cx, reached, &all).into_iter().collect();
        Ok((image, broken))
    }

    /// Lowers and encodes the reached functions `indices` that type-checked, on every core when there are many, links them, and returns the errors of those that did not compile, by index.
    pub fn compile(&mut self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize]) -> Vec<(usize, String)> {
        self.bind_foreign_statics(cx, reached, indices);
        let mut errors = Vec::new();
        let mut work = Vec::new();
        for &index in indices {
            if reached.list[index].1.is_err() {
                continue;
            }
            let item = reached.list[index].0.0;
            if !cx.foreign(item) {
                work.push(index);
                continue;
            }
            match host_symbol(cx, item) {
                Ok(address) if !lower::is_static(cx, item) => self.slot(index).store(address, Ordering::Release),
                Ok(_) => {}
                Err(_) if lower::is_static(cx, item) && weak(cx, item) => {}
                Err(error) => errors.push((index, error)),
            }
        }
        self.place_statics(cx, reached, &work);
        errors.extend(self.compile_all(cx, reached, &work));
        self.link();
        errors
    }

    fn bind_foreign_statics(&self, cx: &Context<'_>, reached: &Reached, indices: &[usize]) {
        for (index, ((item, _), typed)) in indices.iter().map(|&index| (index, &reached.list[index])) {
            if typed.is_err() || !cx.foreign(*item) || !lower::is_static(cx, *item) {
                continue;
            }
            let cell = match (host_symbol(cx, *item), weak(cx, *item)) {
                (found, true) => self.data(&found.unwrap_or(0).to_le_bytes()),
                (Ok(address), false) => address as i64,
                (Err(_), false) => continue,
            };
            self.cells.borrow_mut().insert(index, cell);
        }
    }

    fn link(&mut self) {
        for &(address, index) in self.fixups.borrow().iter() {
            let code = self.slot(index).load(Ordering::Acquire) as u64;
            // SAFETY: address is a word of a constant table that data handed out, which stays mapped for the life of the process.
            unsafe { (address as *mut u64).write_unaligned(code) };
        }
    }

    fn compile_one(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<usize> {
        let func = lower::lower(cx, reached, self, index)?;
        let compiled = haste_codegen::compile(func).map_err(|error| anyhow!(error))?;
        self.memory.write(&compiled.code).map_err(|error| anyhow!(error))
    }

    /// Runs each case `(index, name)`, a reached `fn(i64) -> i64`, on each input and prints `name input value` lines, all in one forked child; a case that crashes gets its status and the cases after it run in a new child.
    pub fn run_cases(&self, cases: &[(usize, &str)], inputs: &[i64]) -> Result<Vec<i32>> {
        if let Some((_, name)) = cases.iter().find(|(index, _)| self.slot(*index).load(Ordering::Acquire) == 0) {
            bail!("{name} has no code");
        }
        let mut statuses = vec![0; cases.len()];
        let mut from = 0;
        while from < cases.len() {
            let (done, status) = self.run_from(&cases[from..], inputs)?;
            if status == 0 {
                break;
            }
            let crashed = (from + done).min(cases.len() - 1);
            statuses[crashed] = status;
            from = crashed + 1;
        }
        Ok(statuses)
    }

    fn run_from(&self, cases: &[(usize, &str)], inputs: &[i64]) -> Result<(usize, i32)> {
        std::io::Write::flush(&mut std::io::stdout())?;
        let (pid, mut progress) = fork_piped(|progress| self.run_in_child(cases, inputs, progress))?;
        let mut done = Vec::new();
        std::io::Read::read_to_end(&mut progress, &mut done)?;
        Ok((done.len(), wait(pid)))
    }

    fn run_in_child(&self, cases: &[(usize, &str)], inputs: &[i64], mut progress: std::fs::File) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        for (index, name) in cases {
            let address = self.slot(*index).load(Ordering::Acquire);
            // SAFETY: address is code that the encoder wrote for a native fn(i64) -> i64 signature, checked to be set before the fork.
            let case: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute::<usize, extern "C" fn(i64) -> i64>(address) };
            for &input in inputs {
                writeln!(out, "{name} {input} {}", case(input))?;
                out.flush()?;
            }
            progress.write_all(&[1])?;
        }
        Ok(())
    }

    /// Runs the program in a forked child the way the native entry point does: std's `lang_start` with `main`, `args` as argv and the default SIGPIPE handling, and returns the child's exit status.
    pub fn run_main(&self, entry: (usize, usize), args: &[String]) -> Result<i32> {
        let pid = self.fork_main(entry, args, || {})?;
        Ok(wait(pid))
    }

    /// Forks a child that runs `setup` and then the program from `lang_start` with `main` and `args`, and returns its pid.
    fn fork_main(&self, entry: (usize, usize), args: &[String], setup: impl FnOnce()) -> Result<libc::pid_t> {
        if self.slot(entry.0).load(Ordering::Acquire) == 0 || self.slot(entry.1).load(Ordering::Acquire) == 0 {
            bail!("`main` or `lang_start` has no code");
        }
        std::io::Write::flush(&mut std::io::stdout())?;
        // SAFETY: no other thread runs at this point, so the child gets a consistent copy of every lock.
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            bail!("fork failed: {}", std::io::Error::last_os_error());
        }
        if pid == 0 {
            setup();
            let code = self.enter_main(entry, args);
            // SAFETY: _exit ends the child without running the parent's destructors; lang_start flushed the program's own output.
            unsafe { libc::_exit(code) }
        }
        Ok(pid)
    }

    /// Runs the program the way the native entry point does: std's `lang_start` with `main`, `args` as argv and the default SIGPIPE handling; returns its exit status.
    fn enter_main(&self, (start, main): (usize, usize), args: &[String]) -> i32 {
        let (start, main) = (self.slot(start).load(Ordering::Acquire), self.slot(main).load(Ordering::Acquire));
        let Ok(owned) = args.iter().map(|arg| std::ffi::CString::new(arg.as_str())).collect::<Result<Vec<_>, _>>() else { return 101 };
        let mut argv: Vec<*const u8> = owned.iter().map(|arg| arg.as_ptr().cast()).collect();
        argv.push(std::ptr::null());
        // SAFETY: start is code that the encoder wrote, or a stub that compiles it, for `lang_start::<T>(fn() -> T, isize, *const *const u8, u8) -> isize`, and main for the `fn() -> T` it takes.
        let lang_start: extern "C" fn(usize, isize, *const *const u8, u8) -> isize = unsafe { std::mem::transmute::<usize, _>(start) };
        lang_start(main, owned.len() as isize, argv.as_ptr(), 0) as i32
    }
}

/// Waits for child `pid` and gives its exit status, or 128 plus the signal that ended it.
fn wait(pid: libc::pid_t) -> i32 {
    let mut status = 0;
    // SAFETY: pid is a child of this process, and status is a live local.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    if libc::WIFEXITED(status) { libc::WEXITSTATUS(status) } else { 128 + libc::WTERMSIG(status) }
}

/// Makes a fatal signal end this forked child at once with status 128 plus the signal, as `wait` reports it, so no core of the whole compiler is dumped.
pub(crate) fn quiet_crashes() {
    extern "C" fn end(signal: libc::c_int) {
        // SAFETY: _exit is async-signal-safe and ends the child without running the parent's destructors.
        unsafe { libc::_exit(128 + signal) }
    }
    for signal in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE, libc::SIGABRT] {
        // SAFETY: end is an async-signal-safe handler, and signal is a valid signal number.
        unsafe { libc::signal(signal, end as *const () as libc::sighandler_t) };
    }
}

/// Forks a child that runs `child` on the write end of a new pipe and exits with status 1 when it fails, and returns its pid and the read end.
pub(crate) fn fork_piped<E>(child: impl FnOnce(std::fs::File) -> Result<(), E>) -> Result<(libc::pid_t, std::fs::File)> {
    let mut ends = [0; 2];
    // SAFETY: ends is a live two-element array for the new descriptors.
    if unsafe { libc::pipe(ends.as_mut_ptr()) } != 0 {
        bail!("pipe failed: {}", std::io::Error::last_os_error());
    }
    // SAFETY: no other thread runs at this point, so the child gets a consistent copy of every lock.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        quiet_crashes();
        // SAFETY: ends[0] is the read end this child does not use, and ends[1] is the write end it owns from here on.
        let out = unsafe {
            libc::close(ends[0]);
            std::fs::File::from_raw_fd(ends[1])
        };
        let wrote = child(out);
        // SAFETY: _exit ends the child without running the parent's destructors.
        unsafe { libc::_exit(i32::from(wrote.is_err())) }
    }
    // SAFETY: ends[1] is the write end that only the child uses, and ends[0] is the read end this process owns from here on.
    let input = unsafe {
        libc::close(ends[1]);
        std::fs::File::from_raw_fd(ends[0])
    };
    Ok((pid, input))
}
