//! Thread-local `setjmp`/`longjmp` panic unwinding with stack-allocated `Drop` cleanup guards (`std::panic::catch_unwind`).

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use haste_codegen::CodeUnwindDrop;

const DROP_ACTIVE: u8 = 1;
const DROP_LINKED: u8 = 2;

#[repr(C, align(16))]
struct JmpBuf([u64; 64]);

#[repr(C)]
struct UnwindFrame {
    prev: *mut UnwindFrame,
    drops: *mut DropNode,
    stop_fp: usize,
    env: JmpBuf,
    payload: *mut u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct JitUnwindDrop {
    local_off: i32,
    flag_off: i32,
    drop_slot: usize,
    indirect: usize,
}

#[repr(C)]
struct SharedUnwindNode {
    next: usize,
    start: usize,
    end: usize,
    count: usize,
}

static JIT_UNWIND_HEAD: AtomicUsize = AtomicUsize::new(0);

/// Points this process's JIT unwind walker at the active image's shared unwind-list head.
pub(crate) fn set_jit_unwind_head(head: usize) {
    if head != 0 {
        JIT_UNWIND_HEAD.store(head, Ordering::Release);
    }
}

/// Clears the active JIT unwind-list head if it still matches `head`.
pub(crate) fn clear_jit_unwind_head(head: usize) {
    if head != 0 {
        let _ = JIT_UNWIND_HEAD.compare_exchange(head, 0, Ordering::AcqRel, Ordering::Relaxed);
    }
}

/// Registers frame-table unwind drops in shared image memory so forked JIT children see late-compiled functions.
pub(crate) fn register_shared_drops(
    head: usize,
    alloc: impl FnOnce(usize) -> Option<usize>,
    exec: usize,
    len: usize,
    drops: &[CodeUnwindDrop],
    symbols: &[u64],
) {
    if head == 0 || drops.is_empty() || len == 0 {
        return;
    }
    let resolved: Vec<JitUnwindDrop> = drops
        .iter()
        .filter_map(|d| {
            let slot = *symbols.get(d.drop_sym as usize)? as usize;
            (slot != 0).then_some(JitUnwindDrop { local_off: d.local_off, flag_off: d.flag_off, drop_slot: slot, indirect: usize::from(d.indirect) })
        })
        .collect();
    if resolved.is_empty() {
        return;
    }
    let header_bytes = size_of::<SharedUnwindNode>();
    let total_bytes = header_bytes + resolved.len() * size_of::<JitUnwindDrop>();
    let Some(mem) = alloc(total_bytes) else {
        return;
    };
    let head_atom = head as *const AtomicUsize;
    // SAFETY: `mem` is a freshly allocated, 8-aligned block in the image's shared arena; `head_atom` lives in the same arena.
    unsafe {
        let drops_dst = (mem + header_bytes) as *mut JitUnwindDrop;
        std::ptr::copy_nonoverlapping(resolved.as_ptr(), drops_dst, resolved.len());
        let node = mem as *mut SharedUnwindNode;
        (*node).start = exec;
        (*node).end = exec + len;
        (*node).count = resolved.len();
        let mut old = (*head_atom).load(Ordering::Acquire);
        loop {
            (*node).next = old;
            match (*head_atom).compare_exchange_weak(old, mem, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(cur) => old = cur,
            }
        }
    }
}

fn lookup_jit_drops(call_pc: usize) -> Option<&'static [JitUnwindDrop]> {
    let head = JIT_UNWIND_HEAD.load(Ordering::Acquire);
    if head == 0 {
        return None;
    }
    // SAFETY: `head` and every linked `SharedUnwindNode` live in the active image's immutable shared arena.
    unsafe {
        let mut cur = (*(head as *const AtomicUsize)).load(Ordering::Acquire);
        while cur != 0 {
            let node = &*(cur as *const SharedUnwindNode);
            if node.start <= call_pc && call_pc < node.end {
                let ptr = (cur + size_of::<SharedUnwindNode>()) as *const JitUnwindDrop;
                return Some(std::slice::from_raw_parts(ptr, node.count));
            }
            cur = node.next;
        }
    }
    None
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

unsafe fn drain_drops_until(mark: *mut DropNode, limit_addr: usize) {
    loop {
        let node = CURRENT_DROPS.get();
        if node.is_null() || node == mark || (limit_addr != 0 && (node as usize) >= limit_addr) {
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

unsafe fn run_jit_frame_drops(fp: usize, call_pc: usize) {
    let Some(drops) = lookup_jit_drops(call_pc) else {
        return;
    };
    for d in drops.iter().rev() {
        let flag_ptr = fp.wrapping_add_signed(d.flag_off as isize) as *mut u8;
        // SAFETY: `fp` is a live JIT frame whose `flag_off` and `local_off` come from codegen frame layout.
        unsafe {
            if *flag_ptr == 0 {
                continue;
            }
            *flag_ptr = 0;
            let slot_ptr = fp.wrapping_add_signed(d.local_off as isize);
            let arg = if d.indirect != 0 { *(slot_ptr as *const *mut u8) } else { slot_ptr as *mut u8 };
            let drop_addr = *(d.drop_slot as *const usize);
            if drop_addr != 0 {
                let drop_fn: extern "C" fn(*mut u8) = std::mem::transmute::<usize, _>(drop_addr);
                drop_fn(arg);
            }
        }
    }
}

unsafe fn drain_jit_frames(mut fp: usize, mut ret_pc: usize, stop_fp: usize, mark: *mut DropNode) {
    while fp != 0 && (stop_fp == 0 || fp < stop_fp) {
        // SAFETY: `mark` and `fp` bound active stack frames between the panic site and `catch_unwind`.
        unsafe {
            drain_drops_until(mark, fp);
            if ret_pc > 0 {
                run_jit_frame_drops(fp, ret_pc - 1);
            }
            let prev_fp = *(fp as *const usize);
            let prev_ret = *((fp + 8) as *const usize);
            if prev_fp <= fp {
                break;
            }
            fp = prev_fp;
            ret_pc = prev_ret;
        }
    }
    // SAFETY: drains any remaining linked drop guards above `mark`.
    unsafe { drain_drops_until(mark, 0) };
}

/// Runs `try_fn(data)` inside a thread-local unwind frame; on panic, drains unwound `Drop` guards, runs `catch_fn(data, payload)`, and returns `1`, else `0`.
pub extern "C" fn catch_unwind(
    try_fn: extern "C" fn(*mut u8),
    data: *mut u8,
    catch_fn: extern "C" fn(*mut u8, *mut u8),
) -> i32 {
    let prev = CURRENT_FRAME.get();
    let drops = CURRENT_DROPS.get();
    let mut frame = UnwindFrame { prev, drops, stop_fp: 0, env: JmpBuf([0; 64]), payload: std::ptr::null_mut() };
    frame.stop_fp = &raw const frame as usize;
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

extern "C" fn start_panic_inner(payload_data: usize, payload_vtable: usize, caller_fp: usize, caller_ret_pc: usize) -> u32 {
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
        drain_jit_frames(caller_fp, caller_ret_pc, (*frame).stop_fp, (*frame).drops);
        longjmp(&raw mut (*frame).env, 1)
    }
}

#[cfg(target_vendor = "apple")]
macro_rules! start_panic_label {
    () => {
        ".text\n.p2align 4\n.globl _haste_start_panic_tramp\n_haste_start_panic_tramp:"
    };
}

#[cfg(not(target_vendor = "apple"))]
macro_rules! start_panic_label {
    () => {
        ".text\n.p2align 4\n.globl haste_start_panic_tramp\nhaste_start_panic_tramp:"
    };
}

#[cfg(target_arch = "aarch64")]
std::arch::global_asm!(
    start_panic_label!(),
    "stp x29, x30, [sp, #-16]!",
    "mov x2, x29",
    "mov x3, x30",
    "bl {inner}",
    "ldp x29, x30, [sp], #16",
    "ret",
    inner = sym start_panic_inner,
);

#[cfg(all(target_arch = "x86_64", not(windows)))]
std::arch::global_asm!(
    start_panic_label!(),
    "push rbp",
    "mov rdx, rbp",
    "mov rcx, [rsp + 8]",
    "call {inner}",
    "pop rbp",
    "ret",
    inner = sym start_panic_inner,
);

#[cfg(all(target_arch = "x86_64", windows))]
std::arch::global_asm!(
    start_panic_label!(),
    "push rbp",
    "sub rsp, 32",
    "mov r8, rbp",
    "mov r9, [rsp + 40]",
    "call {inner}",
    "add rsp, 32",
    "pop rbp",
    "ret",
    inner = sym start_panic_inner,
);

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
unsafe extern "C" {
    fn haste_start_panic_tramp(payload_data: usize, payload_vtable: usize) -> u32;
}

/// Entry address of the panic unwinding trampoline that captures the caller's frame pointer and return PC.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub fn start_panic_addr() -> usize {
    haste_start_panic_tramp as *const () as usize
}

/// Entry address of the panic unwinding fallback on architectures without a frame-pointer trampoline.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub fn start_panic_addr() -> usize {
    extern "C" fn fallback(payload_data: usize, payload_vtable: usize) -> u32 {
        start_panic_inner(payload_data, payload_vtable, 0, 0)
    }
    fallback as *const () as usize
}

/// Extracts the boxed `dyn Any + Send + 'static` fat pointer saved by `start_panic`.
pub extern "C" fn panic_cleanup(payload: *mut u8) -> FatPtr {
    // SAFETY: payload was allocated with Box::into_raw(Box::new(boxed)) in start_panic.
    unsafe { *Box::from_raw(payload.cast::<FatPtr>()) }
}
