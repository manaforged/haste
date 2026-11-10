//! Stubs that compile a function on its first call: each call table slot starts at a stub that asks the compiler process for its function's code and jumps to it.

use std::io::{Read, Write};
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

/// The bytes one stub takes in stub memory.
pub(super) const STUB_BYTES: usize = 32;

static TABLE: AtomicUsize = AtomicUsize::new(0);
static STUBS_FROM: AtomicUsize = AtomicUsize::new(0);
static STUBS_TO: AtomicUsize = AtomicUsize::new(0);
static ASK: AtomicI32 = AtomicI32::new(-1);
static ANSWER: AtomicI32 = AtomicI32::new(-1);
static TALK: Mutex<()> = Mutex::new(());

std::arch::global_asm!(
    ".pushsection .text",
    ".globl haste_stub_entry",
    "haste_stub_entry:",
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
    ".popsection",
    resolve = sym resolve,
);

unsafe extern "C" {
    fn haste_stub_entry();
}

/// The machine code of the stub of call table slot `index`: it puts the index in r11 and jumps to the shared entry, which keeps every argument register.
pub(super) fn code(index: usize) -> [u8; 19] {
    let mut bytes = [0; 19];
    bytes[..2].copy_from_slice(&[0x41, 0xBB]);
    bytes[2..6].copy_from_slice(&(index as u32).to_le_bytes());
    bytes[6..8].copy_from_slice(&[0x49, 0xBA]);
    bytes[8..16].copy_from_slice(&(haste_stub_entry as *const () as usize as u64).to_le_bytes());
    bytes[16..].copy_from_slice(&[0x41, 0xFF, 0xE2]);
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
    let _talk = TALK.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(code) = compiled(index) {
        return code;
    }
    match ask(index) {
        Ok(code) if code != 0 => code,
        // SAFETY: _exit ends the program at once; the compiler printed why this function has no code.
        _ => unsafe { libc::_exit(101) },
    }
}
