use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticInit {
    Empty = 0,
    Filled = 1,
    Filling = 2,
    Poisoned = 3,
}

struct FillGuard<'a> {
    state: &'a AtomicU8,
    flag: usize,
    done: bool,
}

impl Drop for FillGuard<'_> {
    fn drop(&mut self) {
        thread_state().filling.retain(|&at| at != self.flag);
        if !self.done {
            self.state
                .store(StaticInit::Poisoned as u8, Ordering::Release);
        }
    }
}

#[derive(Default)]
struct ThreadState {
    cells: Vec<usize>,
    filling: Vec<usize>,
}

const FAST_SLOTS: usize = 32;

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
    // SAFETY: flag is the state byte after a static's value in a cell that lives as long as the program.
    let state = unsafe { &*(flag as *const AtomicU8) };
    loop {
        match state.compare_exchange(
            StaticInit::Empty as u8,
            StaticInit::Filling as u8,
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
                state.store(StaticInit::Filled as u8, Ordering::Release);
                return;
            }
            Err(v) if v == StaticInit::Filled as u8 => return,
            Err(v) if v == StaticInit::Poisoned as u8 => panic!("static initializer panicked"),
            Err(_) if thread_state().filling.contains(&flag) => return,
            Err(_) => std::thread::yield_now(),
        }
    }
}

/// The calling thread's cell of `size` zeroed bytes for the `#[thread_local]` static with thread slot `slot`.
pub(crate) extern "C" fn thread_cell(slot: usize, size: usize, init_slot: usize) -> usize {
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
