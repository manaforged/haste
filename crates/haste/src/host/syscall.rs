//! The host's system call entry, which an `asm!` that runs one `syscall` instruction calls in its place.

const WORDS: usize = 7;

/// Runs the Linux system call whose number and six arguments `words` holds in the registers the kernel reads them from, and returns what the kernel leaves in `rax`.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub extern "C" fn run(words: &[usize; WORDS]) -> usize {
    let [number, a0, a1, a2, a3, a4, a5] = *words;
    let result;
    // SAFETY: compiled code calls this where its own `asm!` ran the same `syscall` on the same registers, so the call has the effects that code was written for.
    unsafe {
        std::arch::asm!("syscall", inlateout("rax") number => result, in("rdi") a0, in("rsi") a1, in("rdx") a2, in("r10") a3, in("r8") a4, in("r9") a5, lateout("rcx") _, lateout("r11") _, options(nostack))
    };
    result
}

/// Runs the Linux system call whose number and six arguments `words` holds in the registers the kernel reads them from, and returns what the kernel leaves in `x0`.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub extern "C" fn run(words: &[usize; WORDS]) -> usize {
    let [number, a0, a1, a2, a3, a4, a5] = *words;
    let result;
    // SAFETY: compiled code calls this where its own `asm!` ran the same `svc #0` on the same registers, so the call has the effects that code was written for.
    unsafe {
        std::arch::asm!("svc #0", in("x8") number, inlateout("x0") a0 => result, in("x1") a1, in("x2") a2, in("x3") a3, in("x4") a4, in("x5") a5, options(nostack))
    };
    result
}

/// Fails every system call with `ENOSYS` on a host whose system calls this entry does not make.
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
pub extern "C" fn run(_words: &[usize; WORDS]) -> usize {
    (libc::ENOSYS as usize).wrapping_neg()
}
