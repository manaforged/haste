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
    bodies: lower::Bodies,
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

    fn bodies(&self) -> &lower::Bodies {
        &self.bodies
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
        let shared = Shared::new(SHARED_BYTES).map_err(|error| anyhow!(error))?;
        let capacity = reached.list.len().max(TABLE_SLOTS);
        let table = Shared::low(capacity * size_of::<AtomicUsize>()).and_then(|mut low| low.alloc(capacity * size_of::<AtomicUsize>())).map_err(|error| anyhow!(error))?;
        let memory = Memory::default();
        let state = Shared::private(STATE_BYTES).map_err(|error| anyhow!(error))?;
        Ok(Image { table, capacity, memory, stubs: Memory::default(), shared: RefCell::new(shared), state: RefCell::new(state), tables: RefCell::default(), fixups: RefCell::default(), cells: RefCell::default(), wanted: RefCell::default(), bodies: lower::Bodies::default() })
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
        let lowered = self.bodies.body(index);
        let func = match lowered {
            Some(body) => std::rc::Rc::unwrap_or_clone(body),
            None => lower::lower(cx, reached, self, index)?,
        };
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
        if self.slot(entry.0).load(Ordering::Acquire) == 0 || self.slot(entry.1).load(Ordering::Acquire) == 0 {
            bail!("`main` or `lang_start` has no code");
        }
        Ok(wait(fork_child(|| self.enter_main(entry, args))?))
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

/// The state byte of a static whose initializer has written its value.
pub(crate) const FILLED: u8 = 1;
const EMPTY: u8 = 0;
const FILLING: u8 = 2;

/// What the host keeps for each thread of the program: its cells of `#[thread_local]` statics by initializer, and the statics it is filling.
#[derive(Default)]
struct ThreadState {
    cells: rustc_hash::FxHashMap<usize, usize>,
    filling: Vec<usize>,
}

thread_local! {
    static THREAD_STATE: std::cell::Cell<*mut ThreadState> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// The calling thread's state, made on first use and never freed, since the program's thread exit code may still read it after the host's own thread locals are gone.
fn thread_state() -> &'static mut ThreadState {
    THREAD_STATE.with(|state| {
        if state.get().is_null() {
            state.set(Box::into_raw(Box::default()));
        }
        // SAFETY: the state is this thread's own leaked allocation, never shared with another thread.
        unsafe { &mut *state.get() }
    })
}

/// Runs the initializer at `code` of the static whose state byte is at `flag`, unless another call filled it: a thread that finds another thread filling it waits, and the thread filling it reads it as it is.
pub(crate) extern "C" fn fill_once(flag: usize, code: usize) {
    // SAFETY: flag is the state byte after a static's value in a cell that lives as long as the program.
    let state = unsafe { &*(flag as *const std::sync::atomic::AtomicU8) };
    loop {
        match state.compare_exchange(EMPTY, FILLING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                thread_state().filling.push(flag);
                // SAFETY: code is the generated initializer of this static, a function that takes and returns nothing.
                let fill: extern "C" fn() = unsafe { std::mem::transmute::<usize, extern "C" fn()>(code) };
                fill();
                thread_state().filling.retain(|&at| at != flag);
                state.store(FILLED, Ordering::Release);
                return;
            }
            Err(FILLED) => return,
            Err(_) if thread_state().filling.contains(&flag) => return,
            Err(_) => std::thread::yield_now(),
        }
    }
}

/// The calling thread's cell of `size` zeroed bytes for the `#[thread_local]` static whose initializer is reached function `index`; cells are never freed, since the program's thread exit code may still read them.
pub(crate) extern "C" fn thread_cell(index: usize, size: usize) -> usize {
    *thread_state().cells.entry(index).or_insert_with(|| {
        let layout = std::alloc::Layout::from_size_align(size.max(1), 64).unwrap_or_else(|_| std::alloc::Layout::new::<u8>());
        // SAFETY: layout has a nonzero size.
        let cell = unsafe { std::alloc::alloc_zeroed(layout) };
        if cell.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        cell as usize
    })
}

/// Forks a child that dumps no core when a signal ends it, runs `child` and exits with the status it returns; returns the child's pid.
pub(crate) fn fork_child(child: impl FnOnce() -> i32) -> Result<libc::pid_t> {
    std::io::Write::flush(&mut std::io::stdout())?;
    // SAFETY: no other thread runs at this point, so the child gets a consistent copy of every lock.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        // SAFETY: PR_SET_DUMPABLE with 0 only clears this child's dumpable flag, so a crash dumps no core of the whole compiler.
        unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0) };
        let status = child();
        // SAFETY: _exit ends the child without running the parent's destructors.
        unsafe { libc::_exit(status) }
    }
    Ok(pid)
}

/// A new pipe whose ends close when the program executes another one.
pub(crate) fn pipe() -> Result<[i32; 2]> {
    let mut ends = [0; 2];
    // SAFETY: ends is a live two-element array for the new descriptors.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        bail!("pipe failed: {}", std::io::Error::last_os_error());
    }
    Ok(ends)
}

/// Forks a child that runs `child` on the write end of a new pipe and exits with status 1 when it fails, and returns its pid and the read end.
pub(crate) fn fork_piped<E>(child: impl FnOnce(std::fs::File) -> Result<(), E>) -> Result<(libc::pid_t, std::fs::File)> {
    let [read, write] = pipe()?;
    let pid = fork_child(|| {
        // SAFETY: read is the end this child does not use, and write is the end it owns from here on.
        let out = unsafe {
            libc::close(read);
            std::fs::File::from_raw_fd(write)
        };
        i32::from(child(out).is_err())
    })?;
    // SAFETY: write is the end that only the child uses, and read is the end this process owns from here on.
    let input = unsafe {
        libc::close(write);
        std::fs::File::from_raw_fd(read)
    };
    Ok((pid, input))
}
