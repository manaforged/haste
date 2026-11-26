//! Stubs that compile a function on its first call: each call table slot starts at a stub that asks the compiler process for its function's code and jumps to it.

#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::mem::ManuallyDrop;
#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::sync::Mutex;
#[cfg(unix)]
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

pub(super) const STUB_BYTES: usize = 64;
pub(super) const TIER_UP_BIT: u32 = 1 << 31;

static TABLE: AtomicUsize = AtomicUsize::new(0);
static STUBS_FROM: AtomicUsize = AtomicUsize::new(0);
static STUBS_TO: AtomicUsize = AtomicUsize::new(0);
static ASK: AtomicI32 = AtomicI32::new(-1);
static ANSWER: AtomicI32 = AtomicI32::new(-1);
static SHARED_LOCK: AtomicUsize = AtomicUsize::new(0);
static TALK: Mutex<()> = Mutex::new(());

/// The text section and global label of the shared stub entry, in the object format's syntax: Mach-O symbols take a leading underscore.
#[cfg(target_vendor = "apple")]
macro_rules! entry_label {
    () => {
        ".text\n.p2align 4\n.globl _haste_stub_entry\n_haste_stub_entry:"
    };
}
#[cfg(not(target_vendor = "apple"))]
macro_rules! entry_label {
    () => {
        ".text\n.p2align 4\n.globl haste_stub_entry\nhaste_stub_entry:"
    };
}

#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(
    entry_label!(),
    "push rbp",
    "mov rbp, rsp",
    "sub rsp, 192",
    "mov [rsp], rdi",
    "mov [rsp + 8], rsi",
    "mov [rsp + 16], rdx",
    "mov [rsp + 24], rcx",
    "mov [rsp + 32], r8",
    "mov [rsp + 40], r9",
    "mov [rsp + 48], rax",
    "mov [rsp + 56], r10",
    "movdqu [rsp + 64], xmm0",
    "movdqu [rsp + 80], xmm1",
    "movdqu [rsp + 96], xmm2",
    "movdqu [rsp + 112], xmm3",
    "movdqu [rsp + 128], xmm4",
    "movdqu [rsp + 144], xmm5",
    "movdqu [rsp + 160], xmm6",
    "movdqu [rsp + 176], xmm7",
    "mov rdi, r11",
    "call {resolve}",
    "mov r11, rax",
    "mov rdi, [rsp]",
    "mov rsi, [rsp + 8]",
    "mov rdx, [rsp + 16]",
    "mov rcx, [rsp + 24]",
    "mov r8, [rsp + 32]",
    "mov r9, [rsp + 40]",
    "mov rax, [rsp + 48]",
    "mov r10, [rsp + 56]",
    "movdqu xmm0, [rsp + 64]",
    "movdqu xmm1, [rsp + 80]",
    "movdqu xmm2, [rsp + 96]",
    "movdqu xmm3, [rsp + 112]",
    "movdqu xmm4, [rsp + 128]",
    "movdqu xmm5, [rsp + 144]",
    "movdqu xmm6, [rsp + 160]",
    "movdqu xmm7, [rsp + 176]",
    "mov rsp, rbp",
    "pop rbp",
    "jmp r11",
    resolve = sym resolve,
);

#[cfg(target_arch = "aarch64")]
std::arch::global_asm!(
    entry_label!(),
    "stp x29, x30, [sp, #-16]!",
    "mov x29, sp",
    "sub sp, sp, #208",
    "stp x0, x1, [sp]",
    "stp x2, x3, [sp, #16]",
    "stp x4, x5, [sp, #32]",
    "stp x6, x7, [sp, #48]",
    "stp x8, x9, [sp, #64]",
    "stp q0, q1, [sp, #80]",
    "stp q2, q3, [sp, #112]",
    "stp q4, q5, [sp, #144]",
    "stp q6, q7, [sp, #176]",
    "mov x0, x16",
    "bl {resolve}",
    "mov x16, x0",
    "ldp x0, x1, [sp]",
    "ldp x2, x3, [sp, #16]",
    "ldp x4, x5, [sp, #32]",
    "ldp x6, x7, [sp, #48]",
    "ldp x8, x9, [sp, #64]",
    "ldp q0, q1, [sp, #80]",
    "ldp q2, q3, [sp, #112]",
    "ldp q4, q5, [sp, #144]",
    "ldp q6, q7, [sp, #176]",
    "mov sp, x29",
    "ldp x29, x30, [sp], #16",
    "br x16",
    resolve = sym resolve,
);

unsafe extern "C" {
    fn haste_stub_entry();
}

pub(super) fn entry_addr() -> usize {
    haste_stub_entry as *const () as usize
}

/// The machine code of the stub of call table slot `index` at `slot`: when the slot no longer holds this stub it jumps to the code there, else it puts the index in r11 and jumps to the shared entry, which keeps every argument register; it clobbers only r10, r11 and the flags.
#[cfg(target_arch = "x86_64")]
pub(super) fn code(index: usize, slot: usize) -> Vec<u8> {
    let mut bytes = vec![0x49, 0xBB];
    bytes.extend((slot as u64).to_le_bytes());
    bytes.extend([0x4D, 0x8B, 0x1B, 0x4C, 0x8D, 0x15]);
    bytes.extend((-20i32).to_le_bytes());
    bytes.extend([0x4D, 0x39, 0xD3, 0x75, 0x13, 0x41, 0xBB]);
    bytes.extend((index as u32).to_le_bytes());
    bytes.extend([0x49, 0xBA]);
    bytes.extend((haste_stub_entry as *const () as usize as u64).to_le_bytes());
    bytes.extend([0x41, 0xFF, 0xE2, 0x41, 0xFF, 0xE3]);
    bytes
}

/// The machine code of the stub of call table slot `index` at `slot`: it loads the slot into x17 and, when that is not this stub, branches to it; else `movz`/`movk` put the index in x16 and it branches to the shared entry, which keeps every argument register; it clobbers only x16, x17 and the flags.
#[cfg(target_arch = "aarch64")]
pub(super) fn code(index: usize, slot: usize) -> Vec<u8> {
    let index = index as u32;
    let words = [
        0x5800_0151,
        0xF940_0231,
        0x10FF_FFD0,
        0xEB10_023F,
        0x5400_0081,
        0xD280_0010 | (index & 0xFFFF) << 5,
        0xF2A0_0010 | (index >> 16) << 5,
        0x5800_00B1,
        0xD61F_0220,
        0xD503_201F,
    ];
    let mut bytes: Vec<u8> = words
        .iter()
        .flat_map(|word: &u32| word.to_le_bytes())
        .collect();
    bytes.extend((slot as u64).to_le_bytes());
    bytes.extend((haste_stub_entry as *const () as usize as u64).to_le_bytes());
    bytes
}

/// Records, before the program forks, where the call table and the stubs are, the shared pipe lock, and which pipe ends reach the compiler.
pub(super) fn connect(
    table: usize,
    stubs: std::ops::Range<usize>,
    (ask, answer): (i32, i32),
    shared_lock: usize,
) {
    TABLE.store(table, Ordering::Release);
    STUBS_FROM.store(stubs.start, Ordering::Release);
    STUBS_TO.store(stubs.end, Ordering::Release);
    ASK.store(ask, Ordering::Release);
    ANSWER.store(answer, Ordering::Release);
    SHARED_LOCK.store(shared_lock, Ordering::Release);
}

#[cfg(unix)]
struct SharedPipeGuard(Option<&'static AtomicU32>);

#[cfg(unix)]
impl SharedPipeGuard {
    fn acquire() -> Self {
        let addr = SHARED_LOCK.load(Ordering::Acquire);
        if addr == 0 {
            return Self(None);
        }
        // SAFETY: `addr` points to a zero-initialized `AtomicU32` in the image's MAP_SHARED arena that lives for the child's lifetime.
        let lock = unsafe { &*(addr as *const AtomicU32) };
        // SAFETY: `getpid` has no preconditions.
        let raw_pid = unsafe { libc::getpid() };
        let pid = if raw_pid > 0 { raw_pid as u32 } else { 1 };
        let mut spins = 0u32;
        loop {
            match lock.compare_exchange_weak(0, pid, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Self(Some(lock)),
                Err(owner) => {
                    spins = spins.wrapping_add(1);
                    if spins & 0xFF == 0
                        && owner != 0
                        && owner != pid
                        // SAFETY: `owner` is a positive PID; signal 0 probes liveness without delivering a signal.
                        && unsafe { libc::kill(owner as libc::pid_t, 0) } != 0
                        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
                        && lock
                            .compare_exchange(owner, pid, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                    {
                        return Self(Some(lock));
                    }
                    if spins < 64 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                }
            }
        }
    }
}

#[cfg(unix)]
impl Drop for SharedPipeGuard {
    fn drop(&mut self) {
        if let Some(lock) = self.0 {
            lock.store(0, Ordering::Release);
        }
    }
}

/// The code that slot `index` holds when it is no longer a stub.
fn compiled(index: usize) -> Option<usize> {
    // SAFETY: TABLE is the call table, which holds a slot for every index a stub carries and stays mapped for the life of the process.
    let code = unsafe {
        (*(TABLE.load(Ordering::Acquire) as *const AtomicUsize).add(index)).load(Ordering::Acquire)
    };
    (!(STUBS_FROM.load(Ordering::Acquire)..STUBS_TO.load(Ordering::Acquire)).contains(&code))
        .then_some(code)
}

/// Requests the compiler for the code of slot `index`: writes the index and reads the code address, which is zero when the function has none.
#[cfg(unix)]
fn ask(index: usize) -> std::io::Result<usize> {
    // SAFETY: ASK and ANSWER are pipe ends this process holds for the life of the program; ManuallyDrop keeps them open.
    let (mut ask, mut answer) = unsafe {
        (
            ManuallyDrop::new(std::fs::File::from_raw_fd(ASK.load(Ordering::Acquire))),
            ManuallyDrop::new(std::fs::File::from_raw_fd(ANSWER.load(Ordering::Acquire))),
        )
    };
    ask.write_all(&(index as u64).to_le_bytes())?;
    let mut frame = [0u8; 16];
    loop {
        answer.read_exact(&mut frame)?;
        let reply_index = u64::from_le_bytes(frame[..8].try_into().unwrap_or([0; 8])) as usize;
        let code = u64::from_le_bytes(frame[8..].try_into().unwrap_or([0; 8])) as usize;
        if reply_index == index {
            return Ok(code);
        }
    }
}

#[cfg(not(unix))]
fn ask(_index: usize) -> std::io::Result<usize> {
    Err(std::io::Error::other("lazy JIT stubs require Unix"))
}

#[cfg(unix)]
fn errno_ptr() -> *mut libc::c_int {
    #[cfg(target_vendor = "apple")]
    // SAFETY: `__error` returns the current thread's writable `errno` slot.
    unsafe {
        libc::__error()
    }
    #[cfg(not(target_vendor = "apple"))]
    // SAFETY: `__errno_location` returns the current thread's writable `errno` slot.
    unsafe {
        libc::__errno_location()
    }
}

#[cfg(unix)]
struct ErrnoGuard(libc::c_int);

#[cfg(unix)]
impl ErrnoGuard {
    fn save() -> Self {
        // SAFETY: `errno_ptr()` points to this thread's valid `errno` slot.
        Self(unsafe { *errno_ptr() })
    }
}

#[cfg(unix)]
impl Drop for ErrnoGuard {
    fn drop(&mut self) {
        // SAFETY: `errno_ptr()` points to this thread's valid `errno` slot.
        unsafe { *errno_ptr() = self.0 };
    }
}

extern "C" fn resolve(index: usize) -> usize {
    #[cfg(unix)]
    let _errno = ErrnoGuard::save();
    let tier_up = index & (TIER_UP_BIT as usize) != 0;
    if !tier_up && let Some(code) = compiled(index) {
        return code;
    }
    let _talk = crate::sync::lock(&TALK);
    #[cfg(unix)]
    let _shared = SharedPipeGuard::acquire();
    if !tier_up && let Some(code) = compiled(index) {
        return code;
    }
    match ask(index) {
        Ok(code) if code != 0 => code,
        #[cfg(unix)]
        // SAFETY: _exit ends the program at once; the compiler printed why this function has no code.
        _ => unsafe { libc::_exit(101) },
        #[cfg(not(unix))]
        _ => std::process::exit(101),
    }
}
