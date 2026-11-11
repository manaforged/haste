//! Stubs that compile a function on its first call: each call table slot starts at a stub that asks the compiler process for its function's code and jumps to it.

use std::io::{Read, Write};
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

/// The bytes one stub takes in stub memory, at most.
pub(super) const STUB_BYTES: usize = 32;

static TABLE: AtomicUsize = AtomicUsize::new(0);
static STUBS_FROM: AtomicUsize = AtomicUsize::new(0);
static STUBS_TO: AtomicUsize = AtomicUsize::new(0);
static ASK: AtomicI32 = AtomicI32::new(-1);
static ANSWER: AtomicI32 = AtomicI32::new(-1);
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
    "str x8, [sp, #64]",
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
    "ldr x8, [sp, #64]",
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

/// The machine code of the stub of call table slot `index`: it puts the index in r11 and jumps to the shared entry, which keeps every argument register.
#[cfg(target_arch = "x86_64")]
pub(super) fn code(index: usize) -> Vec<u8> {
    let mut bytes = vec![0x41, 0xBB];
    bytes.extend((index as u32).to_le_bytes());
    bytes.extend([0x49, 0xBA]);
    bytes.extend((haste_stub_entry as *const () as usize as u64).to_le_bytes());
    bytes.extend([0x41, 0xFF, 0xE2]);
    bytes
}

/// The machine code of the stub of call table slot `index`: `movz`/`movk` put the index in x16, a literal load puts the shared entry in x17, and `br x17` jumps to it; the entry keeps every argument register.
#[cfg(target_arch = "aarch64")]
pub(super) fn code(index: usize) -> Vec<u8> {
    let index = index as u32;
    let words = [0xD280_0010 | (index & 0xFFFF) << 5, 0xF2A0_0010 | (index >> 16) << 5, 0x5800_0051, 0xD61F_0220];
    let mut bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    bytes.extend((haste_stub_entry as *const () as usize as u64).to_le_bytes());
    bytes
}

/// Records, before the program forks, where the call table and the stubs are and which pipe ends reach the compiler.
pub(super) fn connect(table: usize, stubs: std::ops::Range<usize>, (ask, answer): (i32, i32)) {
    TABLE.store(table, Ordering::Release);
    STUBS_FROM.store(stubs.start, Ordering::Release);
    STUBS_TO.store(stubs.end, Ordering::Release);
    ASK.store(ask, Ordering::Release);
    ANSWER.store(answer, Ordering::Release);
}

/// The code that slot `index` holds when it is no longer a stub.
fn compiled(index: usize) -> Option<usize> {
    // SAFETY: TABLE is the call table, which holds a slot for every index a stub carries and stays mapped for the life of the process.
    let code = unsafe { (*(TABLE.load(Ordering::Acquire) as *const AtomicUsize).add(index)).load(Ordering::Acquire) };
    (!(STUBS_FROM.load(Ordering::Acquire)..STUBS_TO.load(Ordering::Acquire)).contains(&code)).then_some(code)
}

/// Requests the compiler for the code of slot `index`: writes the index and reads the code address, which is zero when the function has none.
fn ask(index: usize) -> std::io::Result<usize> {
    // SAFETY: ASK and ANSWER are pipe ends this process holds for the life of the program; ManuallyDrop keeps them open.
    let (mut ask, mut answer) = unsafe { (ManuallyDrop::new(std::fs::File::from_raw_fd(ASK.load(Ordering::Acquire))), ManuallyDrop::new(std::fs::File::from_raw_fd(ANSWER.load(Ordering::Acquire)))) };
    ask.write_all(&(index as u64).to_le_bytes())?;
    let mut code = [0; 8];
    answer.read_exact(&mut code)?;
    Ok(u64::from_le_bytes(code) as usize)
}

extern "C" fn resolve(index: usize) -> usize {
    if let Some(code) = compiled(index) {
        return code;
    }
    let _talk = crate::sync::lock(&TALK);
    if let Some(code) = compiled(index) {
        return code;
    }
    match ask(index) {
        Ok(code) if code != 0 => code,
        // SAFETY: _exit ends the program at once; the compiler printed why this function has no code.
        _ => unsafe { libc::_exit(101) },
    }
}
