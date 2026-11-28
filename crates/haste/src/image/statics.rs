use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticInit {
    Empty = 0,
    Filled = 1,
    Filling = 2,
    Poisoned = 3,
}

pub(crate) const fn static_flag_offset(size: u32) -> u32 {
    (size + 7) & !7
}

pub(crate) const fn static_cell_size(size: u32) -> u32 {
    static_flag_offset(size) + 8
}

struct FillGuard<'a> {
    state: &'a AtomicU64,
    flag: usize,
    done: bool,
}

impl Drop for FillGuard<'_> {
    fn drop(&mut self) {
        thread_state().filling.retain(|&at| at != self.flag);
        if !self.done {
            self.state
                .store(StaticInit::Poisoned as u64, Ordering::Release);
        }
    }
}

#[derive(Default)]
struct ThreadState {
    cells: Vec<usize>,
    filling: Vec<usize>,
}

const FAST_SLOTS: usize = 32;

static MAIN_TP: AtomicUsize = AtomicUsize::new(0);
static MAIN_CELLS: [AtomicUsize; FAST_SLOTS] = [const { AtomicUsize::new(0) }; FAST_SLOTS];

/// Clears the main-thread fast TLS cache after `fork` so a child never inherits parent image cells.
#[cfg(unix)]
pub(super) fn reset_after_fork() {
    MAIN_TP.store(0, Ordering::Relaxed);
    for cell in &MAIN_CELLS {
        cell.store(0, Ordering::Relaxed);
    }
    FAST_CELLS.with(|cells| {
        // SAFETY: called in a freshly forked single-threaded child before any JIT code runs.
        unsafe { *cells.get() = [0; FAST_SLOTS] };
    });
    THREAD_STATE.with(|state| state.set(std::ptr::null_mut()));
}

#[inline(always)]
fn current_tp() -> usize {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let tp: usize;
        // SAFETY: reading the read-only user thread-pointer register has no side effects.
        unsafe {
            std::arch::asm!("mrs {}, tpidrro_el0", out(reg) tp, options(nomem, nostack, preserves_flags));
        }
        tp & !7
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let tp: usize;
        // SAFETY: fs:[0] holds the current thread's self-pointer in the Linux x86-64 TLS ABI.
        unsafe {
            std::arch::asm!("mov {}, fs:[0]", out(reg) tp, options(nostack, readonly, preserves_flags));
        }
        tp
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    {
        0
    }
}

thread_local! {
    static FAST_CELLS: std::cell::UnsafeCell<[usize; FAST_SLOTS]> =
        const { std::cell::UnsafeCell::new([0; FAST_SLOTS]) };
    static THREAD_STATE: std::cell::Cell<*mut ThreadState> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

fn thread_state() -> &'static mut ThreadState {
    THREAD_STATE.with(|state| {
        if state.get().is_null() {
            state.set(Box::into_raw(Box::default()));
        }
        // SAFETY: the state is this thread's own leaked allocation, never shared with another thread.
        unsafe { &mut *state.get() }
    })
}

/// Runs the initializer at `code` of the static whose state byte is at `flag`, unless another call filled it.
pub(crate) extern "C" fn fill_once(flag: usize, code: usize) {
    // SAFETY: flag is the 8-byte aligned state word after a static's value in a cell that lives as long as the program.
    let state = unsafe { &*(flag as *const AtomicU64) };
    loop {
        match state.compare_exchange(
            StaticInit::Empty as u64,
            StaticInit::Filling as u64,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                thread_state().filling.push(flag);
                let mut guard = FillGuard {
                    state,
                    flag,
                    done: false,
                };
                // SAFETY: code is the generated initializer of this static, a function that takes and returns nothing.
                let fill: extern "C" fn() =
                    unsafe { std::mem::transmute::<usize, extern "C" fn()>(code) };
                fill();
                guard.done = true;
                state.store(StaticInit::Filled as u64, Ordering::Release);
                return;
            }
            Err(v) if v == StaticInit::Filled as u64 => return,
            Err(v) if v == StaticInit::Poisoned as u64 => panic!("static initializer panicked"),
            Err(_) if thread_state().filling.contains(&flag) => return,
            Err(_) => std::thread::yield_now(),
        }
    }
}

/// The calling thread's cell of `size` zeroed bytes for the `#[thread_local]` static with thread slot `slot`.
pub(crate) extern "C" fn thread_cell(slot: usize, size: usize, init_slot: usize) -> usize {
    let tp = current_tp();
    if slot < FAST_SLOTS && tp != 0 && tp == MAIN_TP.load(Ordering::Relaxed) {
        let cell = MAIN_CELLS[slot].load(Ordering::Relaxed);
        if cell != 0 {
            return cell;
        }
    }
    if slot < FAST_SLOTS {
        let cell = FAST_CELLS.with(|cells| {
            // SAFETY: FAST_CELLS is thread-local and only read here while no mutable reference exists.
            unsafe { (*cells.get())[slot] }
        });
        if cell != 0 {
            return cell;
        }
    }
    let cell = {
        let state = thread_state();
        if let Some(&cell) = state.cells.get(slot)
            && cell != 0
        {
            return cell;
        }
        new_cell(state, slot, size)
    };
    if slot < FAST_SLOTS {
        FAST_CELLS.with(|cells| {
            // SAFETY: FAST_CELLS is thread-local and no other reference to this slot is live.
            unsafe { (*cells.get())[slot] = cell };
        });
        if tp != 0 {
            let main = MAIN_TP.load(Ordering::Relaxed);
            if main == tp
                || (main == 0
                    && MAIN_TP
                        .compare_exchange(0, tp, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok())
            {
                MAIN_CELLS[slot].store(cell, Ordering::Release);
            }
        }
    }
    if init_slot != 0 {
        // SAFETY: init_slot points to the 8-byte function slot holding this static's zero-arg initializer.
        let code = unsafe { *(init_slot as *const usize) };
        // SAFETY: code is the entry address of the zero-arg extern "C" static initializer stored in init_slot.
        let fill: extern "C" fn() = unsafe { std::mem::transmute::<usize, extern "C" fn()>(code) };
        fill();
    }
    cell
}

#[cold]
fn new_cell(state: &mut ThreadState, slot: usize, size: usize) -> usize {
    let layout = std::alloc::Layout::from_size_align(size.max(1), 64)
        .unwrap_or_else(|_| std::alloc::Layout::new::<u8>());
    // SAFETY: layout has a nonzero size.
    let cell = unsafe { std::alloc::alloc_zeroed(layout) };
    if cell.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    if state.cells.len() <= slot {
        state.cells.resize(slot + 1, 0);
    }
    state.cells[slot] = cell as usize;
    cell as usize
}
