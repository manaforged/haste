//! Thread-local `setjmp`/`longjmp` panic unwinding with stack-allocated `Drop` cleanup guards (`std::panic::catch_unwind`).

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use haste_codegen::{CodeUnwindDrop, Site};

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
#[derive(Clone, Copy)]
struct JitUnwindSite {
    ret_pc: usize,
    handler_pc: usize,
    frame_offset: usize,
}

const SAVED_REG_COUNT: usize = 18;
type SavedRegs = [u64; SAVED_REG_COUNT];

#[repr(C)]
struct JitLand {
    handler_pc: usize,
    target_fp: usize,
    target_sp: usize,
    payload: *mut u8,
    regs: SavedRegs,
}

#[repr(C)]
struct SharedUnwindNode {
    next: usize,
    start: usize,
    end: usize,
    count: usize,
    site_count: usize,
    saved_regs: u32,
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
        JIT_UNWIND_HEAD
            .compare_exchange(head, 0, Ordering::AcqRel, Ordering::Relaxed)
            .ok();
    }
}

/// Registers frame-table unwind drops, `TryCall` landing pads, and callee-saved register masks in shared image memory so forked JIT children see late-compiled functions.
pub(crate) fn register_shared_drops(
    head: usize,
    alloc: impl FnOnce(usize) -> Option<usize>,
    (exec, len): (usize, usize),
    (drops, sites, saved_regs): (&[CodeUnwindDrop], &[Site], u32),
    symbols: &[u64],
) {
    if head == 0 || (drops.is_empty() && sites.is_empty() && saved_regs == 0) || len == 0 {
        return;
    }
    let resolved: Vec<JitUnwindDrop> = drops
        .iter()
        .filter_map(|d| {
            let slot = *symbols.get(d.drop_sym as usize)? as usize;
            (slot != 0).then_some(JitUnwindDrop {
                local_off: d.local_off,
                flag_off: d.flag_off,
                drop_slot: slot,
                indirect: usize::from(d.indirect),
            })
        })
        .collect();
    let resolved_sites: Vec<JitUnwindSite> = sites
        .iter()
        .map(|s| JitUnwindSite {
            ret_pc: exec + s.ret as usize,
            handler_pc: exec + s.handler as usize,
            frame_offset: s.frame_offset as usize,
        })
        .collect();
    if resolved.is_empty() && resolved_sites.is_empty() && saved_regs == 0 {
        return;
    }
    let header_bytes = size_of::<SharedUnwindNode>();
    let drops_bytes = resolved.len() * size_of::<JitUnwindDrop>();
    let sites_bytes = resolved_sites.len() * size_of::<JitUnwindSite>();
    let Some(mem) = alloc(header_bytes + drops_bytes + sites_bytes) else {
        return;
    };
    let head_atom = head as *const AtomicUsize;
    // SAFETY: `mem` is a freshly allocated, 8-aligned block in the image's shared arena; `head_atom` lives in the same arena.
    unsafe {
        let drops_dst = (mem + header_bytes) as *mut JitUnwindDrop;
        std::ptr::copy_nonoverlapping(resolved.as_ptr(), drops_dst, resolved.len());
        let sites_dst = (mem + header_bytes + drops_bytes) as *mut JitUnwindSite;
        std::ptr::copy_nonoverlapping(resolved_sites.as_ptr(), sites_dst, resolved_sites.len());
        let node = mem as *mut SharedUnwindNode;
        (*node).start = exec;
        (*node).end = exec + len;
        (*node).count = resolved.len();
        (*node).site_count = resolved_sites.len();
        (*node).saved_regs = saved_regs;
        let mut old = (*head_atom).load(Ordering::Acquire);
        loop {
            (*node).next = old;
            match (*head_atom).compare_exchange_weak(old, mem, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(cur) => old = cur,
            }
        }
    }
}

fn find_jit_node<T>(mut visit: impl FnMut(usize, &SharedUnwindNode) -> Option<T>) -> Option<T> {
    let head = JIT_UNWIND_HEAD.load(Ordering::Acquire);
    if head == 0 {
        return None;
    }
    // SAFETY: `head` and every linked `SharedUnwindNode` live in the active image's immutable shared arena.
    unsafe {
        let mut cur = (*(head as *const AtomicUsize)).load(Ordering::Acquire);
        while cur != 0 {
            let node = &*(cur as *const SharedUnwindNode);
            if let Some(hit) = visit(cur, node) {
                return Some(hit);
            }
            cur = node.next;
        }
    }
    None
}

fn lookup_jit_frame(call_pc: usize) -> Option<(&'static [JitUnwindDrop], u32)> {
    find_jit_node(|cur, node| {
        if !(node.start <= call_pc && call_pc < node.end) {
            return None;
        }
        let drops = if node.count > 0 {
            let ptr = (cur + size_of::<SharedUnwindNode>()) as *const JitUnwindDrop;
            // SAFETY: `node.count` `JitUnwindDrop` entries immediately follow `node` in the arena.
            unsafe { std::slice::from_raw_parts(ptr, node.count) }
        } else {
            &[]
        };
        Some((drops, node.saved_regs))
    })
}

fn lookup_jit_site(ret_pc: usize) -> Option<(usize, usize)> {
    find_jit_node(|cur, node| {
        if !(node.site_count > 0 && node.start <= ret_pc && ret_pc <= node.end) {
            return None;
        }
        let base = cur + size_of::<SharedUnwindNode>() + node.count * size_of::<JitUnwindDrop>();
        // SAFETY: `node.site_count` `JitUnwindSite` entries follow the drops slice in the arena.
        let slice =
            unsafe { std::slice::from_raw_parts(base as *const JitUnwindSite, node.site_count) };
        slice
            .iter()
            .find(|s| s.ret_pc == ret_pc)
            .map(|s| (s.handler_pc, s.frame_offset))
    })
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

/// Drains linked stack drop nodes above `mark` (and below `limit_addr` when non-zero).
///
/// # Safety
/// Every node on `CURRENT_DROPS` above `mark` must point to a live, initialized `DropNode` on an active unwound frame.
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

/// Runs registered frame-table drop entries and restores callee-saved registers for the JIT frame at `fp` and `call_pc`.
///
/// # Safety
/// `fp` must be a live JIT frame pointer matching the layout recorded for `call_pc`.
unsafe fn run_jit_frame_drops(fp: usize, call_pc: usize, regs: Option<&mut SavedRegs>) {
    let Some((drops, saved_regs)) = lookup_jit_frame(call_pc) else {
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
            let arg = if d.indirect != 0 {
                *(slot_ptr as *const *mut u8)
            } else {
                slot_ptr as *mut u8
            };
            let drop_addr = *(d.drop_slot as *const usize);
            if drop_addr != 0 {
                let drop_fn: extern "C" fn(*mut u8) = std::mem::transmute::<usize, _>(drop_addr);
                drop_fn(arg);
            }
        }
    }
    if let Some(regs) = regs {
        let mut mask = saved_regs;
        let mut bit = 0usize;
        let mut k = 0usize;
        while mask != 0 && bit < regs.len() {
            if (mask & 1) != 0 {
                k += 1;
                // SAFETY: prologue saved the `k`-th callee-saved register at `fp - k * 8`.
                regs[bit] = unsafe { *((fp - k * 8) as *const u64) };
            }
            mask >>= 1;
            bit += 1;
        }
    }
}

/// Walks active JIT frames looking for a `TryCall` handler or `stop_fp`, draining interleaved drops.
///
/// # Safety
/// `fp`, `stop_fp`, and `mark` must bound active frames on the current thread's stack.
unsafe fn unwind_jit_frames(
    (mut fp, mut ret_pc): (usize, usize),
    (stop_fp, mark): (usize, *mut DropNode),
    mut regs: Option<&mut SavedRegs>,
) -> Option<(usize, usize, usize)> {
    while fp != 0 && (stop_fp == 0 || fp < stop_fp) {
        // SAFETY: `fp` and `mark` bound active stack frames between the panic site and the catcher.
        unsafe {
            if ret_pc > 0
                && let Some((handler_pc, frame_offset)) = lookup_jit_site(ret_pc)
            {
                let target_sp = fp.wrapping_sub(frame_offset);
                drain_drops_until(std::ptr::null_mut(), target_sp);
                return Some((handler_pc, fp, target_sp));
            }
            drain_drops_until(mark, fp);
            if ret_pc > 0 {
                run_jit_frame_drops(fp, ret_pc - 1, regs.as_deref_mut());
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
    if stop_fp != 0 {
        // SAFETY: drains any remaining linked drop guards above `mark`.
        unsafe { drain_drops_until(mark, 0) };
    }
    None
}

/// Runs `try_fn(data)` inside a thread-local unwind frame; on panic, drains unwound `Drop` guards, runs `catch_fn(data, payload)`, and returns `1`, else `0`.
pub extern "C" fn catch_unwind(
    try_fn: extern "C" fn(*mut u8),
    data: *mut u8,
    catch_fn: extern "C" fn(*mut u8, *mut u8),
) -> i32 {
    let prev = CURRENT_FRAME.get();
    let drops = CURRENT_DROPS.get();
    let mut frame = UnwindFrame {
        prev,
        drops,
        stop_fp: 0,
        env: JmpBuf([0; 64]),
        payload: std::ptr::null_mut(),
    };
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

extern "C" fn start_panic_inner(
    payload_data: usize,
    payload_vtable: usize,
    caller_fp: usize,
    caller_ret_pc: usize,
    land_out: *mut JitLand,
) -> u32 {
    let frame = CURRENT_FRAME.get();
    // SAFETY: `payload_vtable` points to a valid `dyn BoxMeUp` vtable; `frame` (if non-null) is live on stack.
    unsafe {
        let take_box_ptr = *(payload_vtable as *const usize).add(3);
        let take_box: extern "C" fn(usize) -> FatPtr =
            std::mem::transmute::<usize, _>(take_box_ptr);
        let boxed = take_box(payload_data);
        let payload = Box::into_raw(Box::new(boxed)).cast::<u8>();
        let (stop_fp, mark) = if frame.is_null() {
            (0, std::ptr::null_mut())
        } else {
            ((*frame).stop_fp, (*frame).drops)
        };
        let regs = (!land_out.is_null()).then(|| &mut (*land_out).regs);
        if let Some((handler_pc, target_fp, target_sp)) =
            unwind_jit_frames((caller_fp, caller_ret_pc), (stop_fp, mark), regs)
            && !land_out.is_null()
        {
            (*land_out).handler_pc = handler_pc;
            (*land_out).target_fp = target_fp;
            (*land_out).target_sp = target_sp;
            (*land_out).payload = payload;
            return 0;
        }
        if frame.is_null() {
            std::process::abort();
        }
        CURRENT_FRAME.set((*frame).prev);
        (*frame).payload = payload;
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
    "sub sp, sp, #176",
    "stp x19, x20, [sp, #32]",
    "stp x21, x22, [sp, #48]",
    "stp x23, x24, [sp, #64]",
    "stp x25, x26, [sp, #80]",
    "stp x27, x28, [sp, #96]",
    "stp d8, d9, [sp, #112]",
    "stp d10, d11, [sp, #128]",
    "stp d12, d13, [sp, #144]",
    "stp d14, d15, [sp, #160]",
    "mov x2, x29",
    "mov x3, x30",
    "mov x4, sp",
    "str xzr, [sp]",
    "bl {inner}",
    "ldr x9, [sp]",
    "cbz x9, 2f",
    "ldr x29, [sp, #8]",
    "ldr x10, [sp, #16]",
    "ldr x0, [sp, #24]",
    "ldp x19, x20, [sp, #32]",
    "ldp x21, x22, [sp, #48]",
    "ldp x23, x24, [sp, #64]",
    "ldp x25, x26, [sp, #80]",
    "ldp x27, x28, [sp, #96]",
    "ldp d8, d9, [sp, #112]",
    "ldp d10, d11, [sp, #128]",
    "ldp d12, d13, [sp, #144]",
    "ldp d14, d15, [sp, #160]",
    "mov sp, x10",
    "br x9",
    "2:",
    "add sp, sp, #176",
    "ldp x29, x30, [sp], #16",
    "ret",
    inner = sym start_panic_inner,
);

#[cfg(all(target_arch = "x86_64", not(windows)))]
std::arch::global_asm!(
    start_panic_label!(),
    "push rbp",
    "sub rsp, 176",
    "mov [rsp + 32], rbx",
    "mov [rsp + 40], r12",
    "mov [rsp + 48], r13",
    "mov [rsp + 56], r14",
    "mov [rsp + 64], r15",
    "mov rdx, rbp",
    "mov rcx, [rsp + 184]",
    "mov r8, rsp",
    "mov qword ptr [rsp], 0",
    "call {inner}",
    "mov r11, [rsp]",
    "test r11, r11",
    "jz 2f",
    "mov rbp, [rsp + 8]",
    "mov rcx, [rsp + 16]",
    "mov rax, [rsp + 24]",
    "mov rbx, [rsp + 32]",
    "mov r12, [rsp + 40]",
    "mov r13, [rsp + 48]",
    "mov r14, [rsp + 56]",
    "mov r15, [rsp + 64]",
    "mov rsp, rcx",
    "jmp r11",
    "2:",
    "add rsp, 176",
    "pop rbp",
    "ret",
    inner = sym start_panic_inner,
);

#[cfg(all(target_arch = "x86_64", windows))]
std::arch::global_asm!(
    start_panic_label!(),
    "push rbp",
    "sub rsp, 48",
    "mov r8, rbp",
    "mov r9, [rsp + 56]",
    "mov qword ptr [rsp + 32], 0",
    "call {inner}",
    "add rsp, 48",
    "pop rbp",
    "ret",
    inner = sym start_panic_inner,
);

// SAFETY: `haste_start_panic_tramp` is defined by `global_asm!` above with the C ABI signature below.
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
        start_panic_inner(payload_data, payload_vtable, 0, 0, std::ptr::null_mut())
    }
    fallback as *const () as usize
}

/// Extracts the boxed `dyn Any + Send + 'static` fat pointer saved by `start_panic`.
pub extern "C" fn panic_cleanup(payload: *mut u8) -> FatPtr {
    // SAFETY: payload was allocated with Box::into_raw(Box::new(boxed)) in start_panic.
    unsafe { *Box::from_raw(payload.cast::<FatPtr>()) }
}
