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

/// Fails every system call with `ENOSYS` on a host whose system calls this entry does not make.
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub extern "C" fn run(_words: &[usize; WORDS]) -> usize {
    (libc::ENOSYS as usize).wrapping_neg()
}
