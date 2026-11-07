use crate::fault::{Frame, frame_at};

/// The foreign symbol that std's panic runtime calls to start unwinding.
pub const RAISE_SYMBOL: &str = "_Unwind_RaiseException";

const PANIC_EXIT: i32 = 101;

fn stop(reason: &str) -> ! {
    eprintln!("unwind: {reason}");
    // SAFETY: abort ends the process and touches no Rust state.
    unsafe { libc::abort() }
}

/// Walks the frames above `frame` and resumes at the first handler with `exception` as its payload.
pub extern "C" fn raise(frame: usize, exception: usize) -> ! {
    let mut frame = frame;
    loop {
        if frame == 0 || frame % 8 != 0 {
            stop("the frame chain ended before a handler");
        }
        // SAFETY: generated code keeps frame pointers, so frame points at a saved (frame pointer, return address) pair.
        let (next, ret) = unsafe { (*(frame as *const usize), *((frame + 8) as *const usize)) };
        match frame_at(ret) {
            Frame::Handler { pc, frame_offset } => resume(pc, next - frame_offset, next, exception),
            Frame::Generated => frame = next,
            Frame::Native(name) => stop(&format!("a panic crossed a native frame in {name}")),
        }
    }
}

/// Ends the process when a panic reaches the host entry, as a panic out of `main` does.
pub extern "C" fn uncaught(_exception: usize) {
    eprintln!("unwind: a panic reached the engine entry");
    // SAFETY: _exit ends the process without running the parent's destructors.
    unsafe { libc::_exit(PANIC_EXIT) }
}

#[cfg(target_arch = "x86_64")]
fn resume(pc: usize, sp: usize, fp: usize, exception: usize) -> ! {
    // SAFETY: pc is a try_call handler whose frame has frame pointer fp and stack pointer sp at the call, and the handler expects the payload in rax and no preserved registers.
    unsafe {
        std::arch::asm!("mov rsp, {sp}", "mov rbp, {fp}", "jmp {pc}", sp = in(reg) sp, fp = in(reg) fp, pc = in(reg) pc, in("rax") exception, in("rdx") 0_usize, options(noreturn))
    }
}

#[cfg(target_arch = "aarch64")]
fn resume(pc: usize, sp: usize, fp: usize, exception: usize) -> ! {
    // SAFETY: pc is a try_call handler whose frame has frame pointer fp and stack pointer sp at the call, and the handler expects the payload in x0 and no preserved registers.
    unsafe {
        std::arch::asm!("mov sp, {sp}", "mov x29, {fp}", "br {pc}", sp = in(reg) sp, fp = in(reg) fp, pc = in(reg) pc, in("x0") exception, in("x1") 0_usize, options(noreturn))
    }
}
