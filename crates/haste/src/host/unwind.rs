//! Thread-local `setjmp`/`longjmp` panic unwinding for JIT execution (`std::panic::catch_unwind`).

use std::cell::Cell;

#[repr(C, align(16))]
struct JmpBuf([u64; 64]);

#[repr(C)]
struct UnwindFrame {
    prev: *mut UnwindFrame,
    env: JmpBuf,
    payload: *mut u8,
}

/// Two-word fat pointer matching Rust's `*mut (dyn Any + Send + 'static)` ABI on x86-64 and arm64.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FatPtr {
    pub data: usize,
    pub vtable: usize,
}

thread_local! {
    static CURRENT_FRAME: Cell<*mut UnwindFrame> = const { Cell::new(std::ptr::null_mut()) };
}

unsafe extern "C" {
    #[link_name = "_setjmp"]
    fn setjmp(env: *mut JmpBuf) -> libc::c_int;
    #[link_name = "_longjmp"]
    fn longjmp(env: *mut JmpBuf, val: libc::c_int) -> !;
}

/// Runs `try_fn(data)` inside a thread-local unwind frame; on panic, runs `catch_fn(data, payload)` and returns `1`, else returns `0`.
pub extern "C" fn catch_unwind(
    try_fn: extern "C" fn(*mut u8),
    data: *mut u8,
    catch_fn: extern "C" fn(*mut u8, *mut u8),
) -> i32 {
    let prev = CURRENT_FRAME.get();
    let mut frame = UnwindFrame { prev, env: JmpBuf([0; 64]), payload: std::ptr::null_mut() };
    CURRENT_FRAME.set(&raw mut frame);
    // SAFETY: frame.env is a live 512-byte 16-aligned jmp_buf on this stack frame.
    if unsafe { setjmp(&raw mut frame.env) } == 0 {
        try_fn(data);
        CURRENT_FRAME.set(prev);
        0
    } else {
        catch_fn(data, frame.payload);
        1
    }
}

/// Initiates unwinding of `payload` (`&mut dyn PanicPayload`) to the innermost active `catch_unwind` frame, or aborts when uncaught.
pub extern "C" fn start_panic(payload_data: usize, payload_vtable: usize) -> u32 {
    let frame = CURRENT_FRAME.get();
    if frame.is_null() {
        // SAFETY: uncaught panic terminates the process cleanly via abort.
        unsafe { libc::abort() };
    }
    // SAFETY: frame points to a live UnwindFrame on an enclosing catch_unwind stack frame on this thread.
    unsafe {
        CURRENT_FRAME.set((*frame).prev);
        let take_box_ptr = *(payload_vtable as *const usize).add(3);
        let take_box: extern "C" fn(usize) -> FatPtr = std::mem::transmute::<usize, _>(take_box_ptr);
        let boxed = take_box(payload_data);
        (*frame).payload = Box::into_raw(Box::new(boxed)).cast::<u8>();
        longjmp(&raw mut (*frame).env, 1)
    }
}

/// Extracts the boxed `dyn Any + Send + 'static` fat pointer saved by `start_panic`.
pub extern "C" fn panic_cleanup(payload: *mut u8) -> FatPtr {
    // SAFETY: payload was allocated with Box::into_raw(Box::new(boxed)) in start_panic.
    unsafe { *Box::from_raw(payload.cast::<FatPtr>()) }
}
