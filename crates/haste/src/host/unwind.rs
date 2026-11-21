//! Thread-local `setjmp`/`longjmp` panic unwinding with stack-allocated `Drop` cleanup guards (`std::panic::catch_unwind`).

use std::cell::Cell;

const DROP_ACTIVE: u8 = 1;
const DROP_LINKED: u8 = 2;

#[repr(C, align(16))]
struct JmpBuf([u64; 64]);

#[repr(C)]
struct UnwindFrame {
    prev: *mut UnwindFrame,
    drops: *mut DropNode,
    env: JmpBuf,
    payload: *mut u8,
}

/// Intrusive stack-allocated cleanup node linked onto the thread's unwind drop list while an owned value is live.
#[repr(C)]
pub struct DropNode {
    state: u8,
    prev: *mut DropNode,
    next: *mut DropNode,
    drop_fn: extern "C" fn(*mut u8),
    addr: *mut u8,
}

const _: () = assert!(std::mem::size_of::<DropNode>() == 40);
const _: () = assert!(std::mem::offset_of!(DropNode, state) == 0);
const _: () = assert!(std::mem::offset_of!(DropNode, prev) == 8);
const _: () = assert!(std::mem::offset_of!(DropNode, next) == 16);
const _: () = assert!(std::mem::offset_of!(DropNode, drop_fn) == 24);
const _: () = assert!(std::mem::offset_of!(DropNode, addr) == 32);

/// Two-word fat pointer matching Rust's `*mut (dyn Any + Send + 'static)` ABI on x86-64 and arm64.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FatPtr {
    pub data: usize,
    pub vtable: usize,
}

thread_local! {
    static CURRENT_FRAME: Cell<*mut UnwindFrame> = const { Cell::new(std::ptr::null_mut()) };
    static CURRENT_DROPS: Cell<*mut DropNode> = const { Cell::new(std::ptr::null_mut()) };
}

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "_setjmp"]
    fn setjmp(env: *mut JmpBuf) -> libc::c_int;
    #[link_name = "_longjmp"]
    fn longjmp(env: *mut JmpBuf, val: libc::c_int) -> !;
}

#[cfg(windows)]
unsafe extern "C" {
    #[link_name = "_setjmp"]
    fn win_setjmp(env: *mut JmpBuf, frame: *mut core::ffi::c_void) -> libc::c_int;
    fn longjmp(env: *mut JmpBuf, val: libc::c_int) -> !;
}

#[cfg(windows)]
unsafe fn setjmp(env: *mut JmpBuf) -> libc::c_int {
    // SAFETY: env is a live 16-aligned jmp_buf; passing null frame disables SEH stack unwind in _setjmp.
    unsafe { win_setjmp(env, std::ptr::null_mut()) }
}

/// Unlinks `node` from the thread-local cleanup list when linked; caller must ensure `node` is live.
unsafe fn unlink_drop(node: *mut DropNode) {
    // SAFETY: caller guarantees `node` is a valid, currently linked DropNode on this thread.
    unsafe {
        let prev = (*node).prev;
        let next = (*node).next;
        if !prev.is_null() {
            (*prev).next = next;
        }
        if !next.is_null() {
            (*next).prev = prev;
        } else {
            CURRENT_DROPS.set(prev);
        }
        (*node).prev = std::ptr::null_mut();
        (*node).next = std::ptr::null_mut();
        (*node).state &= !DROP_LINKED;
    }
}

/// Pushes a stack-allocated `DropNode` onto the current thread's unwind cleanup list.
pub extern "C" fn drop_push(node: *mut DropNode) {
    if node.is_null() {
        return;
    }
    // SAFETY: `node` points to a live 40-byte DropNode on the caller's stack frame.
    unsafe {
        if ((*node).state & DROP_LINKED) != 0 {
            unlink_drop(node);
        }
        let head = CURRENT_DROPS.get();
        (*node).prev = head;
        (*node).next = std::ptr::null_mut();
        (*node).state |= DROP_LINKED;
        if !head.is_null() {
            (*head).next = node;
        }
        CURRENT_DROPS.set(node);
    }
}

/// Pops `node` from the current thread's unwind cleanup list when linked.
pub extern "C" fn drop_pop(node: *mut DropNode) {
    if node.is_null() {
        return;
    }
    // SAFETY: `node` points to a live DropNode on the caller's stack frame with initialized state.
    unsafe {
        if ((*node).state & DROP_LINKED) != 0 {
            unlink_drop(node);
        }
    }
}

unsafe fn drain_drops(mark: *mut DropNode) {
    loop {
        let node = CURRENT_DROPS.get();
        if node.is_null() || node == mark {
            break;
        }
        // SAFETY: every node above `mark` lives on an active unwound stack frame above `catch_unwind`.
        unsafe {
            unlink_drop(node);
            let active = ((*node).state & DROP_ACTIVE) != 0;
            (*node).state &= !DROP_ACTIVE;
            if active {
                ((*node).drop_fn)((*node).addr);
            }
        }
    }
}

/// Runs `try_fn(data)` inside a thread-local unwind frame; on panic, drains unwound `Drop` guards, runs `catch_fn(data, payload)`, and returns `1`, else `0`.
pub extern "C" fn catch_unwind(
    try_fn: extern "C" fn(*mut u8),
    data: *mut u8,
    catch_fn: extern "C" fn(*mut u8, *mut u8),
) -> i32 {
    let prev = CURRENT_FRAME.get();
    let drops = CURRENT_DROPS.get();
    let mut frame = UnwindFrame { prev, drops, env: JmpBuf([0; 64]), payload: std::ptr::null_mut() };
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
        std::process::abort();
    }
    // SAFETY: frame points to a live UnwindFrame on an enclosing catch_unwind stack frame on this thread.
    unsafe {
        CURRENT_FRAME.set((*frame).prev);
        let take_box_ptr = *(payload_vtable as *const usize).add(3);
        let take_box: extern "C" fn(usize) -> FatPtr = std::mem::transmute::<usize, _>(take_box_ptr);
        let boxed = take_box(payload_data);
        (*frame).payload = Box::into_raw(Box::new(boxed)).cast::<u8>();
        drain_drops((*frame).drops);
        longjmp(&raw mut (*frame).env, 1)
    }
}

/// Extracts the boxed `dyn Any + Send + 'static` fat pointer saved by `start_panic`.
pub extern "C" fn panic_cleanup(payload: *mut u8) -> FatPtr {
    // SAFETY: payload was allocated with Box::into_raw(Box::new(boxed)) in start_panic.
    unsafe { *Box::from_raw(payload.cast::<FatPtr>()) }
}
