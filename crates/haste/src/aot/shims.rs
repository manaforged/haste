use anyhow::{Result, bail};
use haste_codegen::{Arch, Os};

use super::wide::WideShim;
use super::{Buf, Plan, SectionAddrs};
use crate::host::cpu::HIDDEN;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(in crate::aot) enum Shim {
    Noop,
    Barrier,
    Cpuid,
    ExtendedState,
    Syscall,
    Alloc,
    AllocZeroed,
    Dealloc,
    Realloc,
    FillOnce,
    ThreadCell,
    FrameAlloc,
    FrameFree,
    CatchUnwind,
    StartPanic,
    PanicCleanup,
    DropPush,
    DropPop,
    UnwindWalk,
    Wide(WideShim),
}

const PTHREAD_DEPS: &[&str] = &["pthread_key_create", "pthread_getspecific", "pthread_setspecific", "calloc", "sched_yield"];
const WIN_TLS_DEPS: &[&str] = &["TlsAlloc", "TlsGetValue", "TlsSetValue", "calloc", "SwitchToThread"];

fn collect_deps(shim: Shim, tls: &[&'static str], mem: (&'static str, &'static str, &'static str), jmp: (&'static str, &'static str)) -> Vec<&'static str> {
    let (alloc_aligned, yield_fn, free_fn) = mem;
    match shim {
        Shim::Alloc => vec!["malloc", alloc_aligned],
        Shim::AllocZeroed => vec!["calloc", alloc_aligned, "memset"],
        Shim::Dealloc | Shim::FrameFree => vec![free_fn],
        Shim::FrameAlloc => vec!["malloc"],
        Shim::Realloc => vec!["realloc", "malloc", alloc_aligned, "memcpy", free_fn],
        Shim::FillOnce => vec![yield_fn],
        Shim::ThreadCell => tls.iter().copied().chain(["memcpy", "free", alloc_aligned, "memset"]).collect(),
        Shim::CatchUnwind => tls.iter().copied().chain([jmp.0]).collect(),
        Shim::StartPanic => tls.iter().copied().chain([jmp.1, "abort"]).collect(),
        Shim::DropPush | Shim::DropPop => tls.to_vec(),
        _ => Vec::new(),
    }
}

pub(super) fn deps(shim: Shim) -> Vec<&'static str> {
    collect_deps(shim, PTHREAD_DEPS, ("posix_memalign", "sched_yield", "free"), ("_setjmp", "_longjmp"))
}

pub(super) fn windows_deps(shim: Shim) -> Vec<&'static str> {
    match shim {
        Shim::Alloc => vec!["_aligned_malloc"],
        Shim::AllocZeroed => vec!["_aligned_malloc", "memset"],
        Shim::Dealloc => vec!["_aligned_free"],
        Shim::Realloc => vec!["_aligned_realloc"],
        Shim::FrameAlloc => vec!["malloc"],
        Shim::FrameFree => vec!["free"],
        _ => collect_deps(shim, WIN_TLS_DEPS, ("_aligned_malloc", "SwitchToThread", "free"), ("_setjmp", "longjmp")),
    }
}

pub(super) fn arm64_adrp(pc_va: u64, target_va: u64, reg: u32) -> u32 {
    let page_delta = ((target_va & !0xFFF) as i64 - (pc_va & !0xFFF) as i64) >> 12;
    0x9000_0000 | (((page_delta as u32) & 3) << 29) | ((((page_delta as u32) >> 2) & 0x7_FFFF) << 5) | reg
}

pub(super) fn emit_stub(arch: Arch, stub_va: u64, got_va: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.bytes(&[0xFF, 0x25]);
        out.u32(((got_va as i64 - (stub_va as i64 + 6)) as i32) as u32);
    } else if arch == Arch::Arm64 {
        let imm12 = ((got_va & 0xFFF) >> 3) as u32;
        out.words(&[arm64_adrp(stub_va, got_va, 16), 0xF940_0210 | (imm12 << 10), 0xD61F_0200]);
    }
}

pub(super) fn emit_arm64_adrp_add(pc_va: u64, target_va: u64, reg: u32, out: &mut Buf) {
    out.words(&[arm64_adrp(pc_va, target_va, reg), 0x9100_0000 | (((target_va & 0xFFF) as u32) << 10) | (reg << 5) | reg]);
}

pub(super) fn emit_arm64_veneer(pc_va: u64, target_va: u64, out: &mut Buf) {
    emit_arm64_adrp_add(pc_va, target_va, 16, out);
    out.u32(0xD61F_0200);
}

fn write_arm64_pc_load(out: &mut [u8], pc_va: u64, rd: u32, load_got: bool, target_va: u64) {
    let w0 = arm64_adrp(pc_va, target_va, rd);
    let w1 = if load_got {
        0xF940_0000 | ((((target_va & 0xFFF) >> 3) as u32) << 10) | (rd << 5) | rd
    } else {
        0x9100_0000 | (((target_va & 0xFFF) as u32) << 10) | (rd << 5) | rd
    };
    for (i, w) in [w0, w1].into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
}

pub(super) fn patch_arm64_pie(out: &mut [u8], pc_va: u64, load_got: bool, target_va: u64) -> Result<()> {
    let word0 = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    if (word0 & 0xFFE0_0000) != 0xD280_0000 {
        bail!("expected movz instruction at Arm64Movw64 relocation, found {word0:#010x}");
    }
    write_arm64_pc_load(&mut out[..8], pc_va, word0 & 0x1F, load_got, target_va);
    for i in 2..4 {
        out[i * 4..i * 4 + 4].copy_from_slice(&0xD503_201F_u32.to_le_bytes());
    }
    Ok(())
}

pub(super) fn patch_arm64_adrp_add(out: &mut [u8], pc_va: u64, load_got: bool, target_va: u64, is_word: bool) -> Result<()> {
    let word0 = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    if (word0 & 0x9F00_0000) != 0x9000_0000 {
        bail!("expected adrp instruction at Arm64AdrpAdd relocation, found {word0:#010x}");
    }
    let rd = word0 & 0x1F;
    if is_word {
        let Ok(val32) = u32::try_from(target_va) else {
            bail!("Arm64AdrpAdd Symbol::Word {target_va:#x} exceeds u32");
        };
        let w0 = 0xD280_0000 | ((val32 & 0xFFFF) << 5) | rd;
        let w1 = 0xF2A0_0000 | (((val32 >> 16) & 0xFFFF) << 5) | rd;
        out[..4].copy_from_slice(&w0.to_le_bytes());
        out[4..8].copy_from_slice(&w1.to_le_bytes());
    } else {
        write_arm64_pc_load(out, pc_va, rd, load_got, target_va);
    }
    Ok(())
}

pub(super) fn emit_main_trampoline(arch: Arch, tramp_va: u64, lang_start_va: u64, user_main_va: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.bytes(&[0x48, 0x89, 0xF2, 0x48, 0x63, 0xF7, 0x48, 0x8D, 0x3D]);
        out.u32(((user_main_va as i64 - (tramp_va as i64 + 13)) as i32) as u32);
        out.bytes(&[0x31, 0xC9, 0xE9]);
        out.u32(((lang_start_va as i64 - (tramp_va as i64 + 20)) as i32) as u32);
    } else if arch == Arch::Arm64 {
        out.words(&[0xAA01_03E2, 0x9340_7C01]);
        emit_arm64_adrp_add(tramp_va + 8, user_main_va, 0, out);
        let delta = ((lang_start_va as i64 - (tramp_va + 20) as i64) >> 2) as u32;
        out.words(&[0x5280_0003, 0x1400_0000 | (delta & 0x03FF_FFFF)]);
    }
}

pub(super) fn emit_elf_start(arch: Arch, start_va: u64, tramp_va: u64, libc_start_got: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.bytes(&[0x31, 0xED, 0x49, 0x89, 0xD1, 0x5E, 0x48, 0x89, 0xE2, 0x48, 0x83, 0xE4, 0xF0, 0x50, 0x54, 0x45, 0x31, 0xC0, 0x31, 0xC9, 0x48, 0x8D, 0x3D]);
        out.u32(((tramp_va as i64 - (start_va as i64 + 27)) as i32) as u32);
        out.bytes(&[0xFF, 0x15]);
        out.u32(((libc_start_got as i64 - (start_va as i64 + 33)) as i32) as u32);
        out.u8(0xF4);
    } else if arch == Arch::Arm64 {
        out.words(&[0xD280_001D, 0xD280_001E, 0xAA00_03E5, 0xF940_03E1, 0x9100_23E2, 0xD280_0003, 0xD280_0004, 0x9100_03E6]);
        emit_arm64_adrp_add(start_va + 32, tramp_va, 0, out);
        let imm12 = ((libc_start_got & 0xFFF) >> 3) as u32;
        out.words(&[arm64_adrp(start_va + 40, libc_start_got, 16), 0xF940_0210 | (imm12 << 10), 0xD63F_0200, 0xD420_0020]);
    }
}

pub(super) fn emit_rel_branch(arch: Arch, link: bool, from_va: u64, to_va: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.u8(if link { 0xE8 } else { 0xE9 });
        out.u32(((to_va as i64 - (from_va as i64 + 5)) as i32) as u32);
    } else if arch == Arch::Arm64 {
        out.u32((if link { 0x9400_0000 } else { 0x1400_0000 }) | (((to_va as i64 - from_va as i64) >> 2) as u32 & 0x03FF_FFFF));
    }
}

fn x64_fwd(op: u8, out: &mut Buf) -> usize {
    out.bytes(&[op, 0]);
    out.pos()
}

fn x64_patch8(after: usize, out: &mut Buf) {
    out.0[after - 1] = ((out.pos() as isize - after as isize) as i8) as u8;
}

fn x64_fwd32(prefix: &[u8], out: &mut Buf) -> usize {
    out.bytes(prefix);
    out.u32(0);
    out.pos()
}

fn x64_patch32(after: usize, out: &mut Buf) {
    out.patch_u32(after - 4, ((out.pos() as i64 - after as i64) as i32) as u32);
}

pub(super) fn arm64_fwd(word: u32, out: &mut Buf) -> usize {
    let at = out.pos();
    out.u32(word);
    at
}

fn arm64_patch(at: usize, target: usize, out: &mut Buf) {
    let delta = ((target as isize - at as isize) >> 2) as u32;
    let word = u32::from_le_bytes([out.0[at], out.0[at + 1], out.0[at + 2], out.0[at + 3]]);
    let patched = if (word >> 26) == 5 { (word & !0x03FF_FFFF) | (delta & 0x03FF_FFFF) } else { (word & !(0x7_FFFF << 5)) | ((delta & 0x7_FFFF) << 5) };
    out.patch_u32(at, patched);
}

pub(super) fn arm64_here(labels: &[usize], out: &mut Buf) {
    let target = out.pos();
    labels.iter().for_each(|&at| arm64_patch(at, target, out));
}

fn x64_here(labels: &[usize], out: &mut Buf) {
    labels.iter().for_each(|&after| x64_patch8(after, out));
}

fn x64_jcc(bytes: &[u8], op: u8, out: &mut Buf) -> usize {
    out.bytes(bytes);
    x64_fwd(op, out)
}

fn arm64_jcc(words: &[u32], word: u32, out: &mut Buf) -> usize {
    out.words(words);
    arm64_fwd(word, out)
}

pub(super) struct Emit<'a> {
    arch: Arch,
    pub(super) win: bool,
    va: u64,
    base: usize,
    stub_va: &'a dyn Fn(&str) -> Result<u64>,
}

impl<'a> Emit<'a> {
    pub(super) fn at(&self, out: &Buf) -> u64 { self.va + (out.pos() - self.base) as u64 }
    fn branch(&self, link: bool, name: &str, out: &mut Buf) -> Result<()> {
        emit_rel_branch(self.arch, link, self.at(out), (self.stub_va)(name)?, out);
        Ok(())
    }
    fn call(&self, name: &str, out: &mut Buf) -> Result<()> { self.branch(true, name, out) }
    pub(super) fn tail(&self, name: &str, out: &mut Buf) -> Result<()> { self.branch(false, name, out) }
    fn bytes_call(&self, b: &[u8], name: &str, out: &mut Buf) -> Result<()> { out.bytes(b); self.call(name, out) }
    fn bytes_tail(&self, b: &[u8], name: &str, out: &mut Buf) -> Result<()> { out.bytes(b); self.tail(name, out) }
    pub(super) fn words_call(&self, w: &[u32], name: &str, out: &mut Buf) -> Result<()> { out.words(w); self.call(name, out) }
    fn words_tail(&self, w: &[u32], name: &str, out: &mut Buf) -> Result<()> { out.words(w); self.tail(name, out) }
    fn pick(&self, w: &'static str, u: &'static str) -> &'static str { if self.win { w } else { u } }
}

fn emit_x64_tls_get(c: &Emit<'_>, key_va: u64, out: &mut Buf) -> Result<()> {
    out.bytes(&[0x4C, 0x8D, 0x2D]);
    out.u32(((key_va as i64 - (c.at(out) as i64 + 4)) as i32) as u32);
    let ready_fast = x64_jcc(&[0x41, 0x8B, 0x45, 0x00, 0x83, 0xF8, 0x02], 0x74, out);
    let init_loop = out.pos();
    let do_create = x64_jcc(&[0x31, 0xC0, 0xBA, 1, 0, 0, 0, 0xF0, 0x41, 0x0F, 0xB1, 0x55, 0x00], 0x74, out);
    let ready_wait = x64_jcc(&[0x83, 0xF8, 0x02], 0x74, out);
    c.call(c.pick("SwitchToThread", "sched_yield"), out)?;
    out.bytes(&[0xEB, ((init_loop as isize - (out.pos() as isize + 2)) as i8) as u8]);
    x64_patch8(do_create, out);
    if c.win {
        c.call("TlsAlloc", out)?;
        out.bytes(&[0x41, 0x89, 0x45, 0x08]);
    } else {
        c.bytes_call(&[0x49, 0x8D, 0x7D, 0x08, 0x31, 0xF6], "pthread_key_create", out)?;
    }
    out.bytes(&[0x41, 0xC7, 0x45, 0x00, 0x02, 0, 0, 0]);
    x64_here(&[ready_fast, ready_wait], out);
    let reg = if c.win { &[0x41, 0x8B, 0x4D, 0x08][..] } else { &[0x49, 0x8B, 0x7D, 0x08] };
    c.bytes_call(reg, c.pick("TlsGetValue", "pthread_getspecific"), out)
}

fn emit_x64_unwind_state(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if addrs.unwind_tls != 0 {
        super::tls::emit_x64_tls_unwind_state(addrs.unwind_tls, out);
        return Ok(());
    }
    emit_x64_tls_get(c, addrs.thread_key + 16, out)?;
    let hit = x64_jcc(&[0x48, 0x85, 0xC0], 0x75, out);
    let calloc_args = if c.win { &[0xB9, 1, 0, 0, 0, 0xBA, 0x10, 0, 0, 0][..] } else { &[0xBF, 1, 0, 0, 0, 0xBE, 0x10, 0, 0, 0] };
    c.bytes_call(calloc_args, "calloc", out)?;
    let fail = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
    let set_args = if c.win { &[0x48, 0x89, 0xC3, 0x41, 0x8B, 0x4D, 0x08, 0x48, 0x89, 0xDA][..] } else { &[0x48, 0x89, 0xC3, 0x49, 0x8B, 0x7D, 0x08, 0x48, 0x89, 0xDE] };
    c.bytes_call(set_args, c.pick("TlsSetValue", "pthread_setspecific"), out)?;
    out.bytes(&[0x48, 0x89, 0xD8]);
    x64_here(&[hit, fail], out);
    Ok(())
}

fn emit_x64_unlink_r12(out: &mut Buf) {
    let no_prev = x64_jcc(&[0x49, 0x8B, 0x4C, 0x24, 0x08, 0x49, 0x8B, 0x54, 0x24, 0x10, 0x48, 0x85, 0xC9], 0x74, out);
    out.bytes(&[0x48, 0x89, 0x51, 0x10]);
    x64_patch8(no_prev, out);
    let is_tail = x64_jcc(&[0x48, 0x85, 0xD2], 0x74, out);
    let after_next = x64_jcc(&[0x48, 0x89, 0x4A, 0x08], 0xEB, out);
    x64_patch8(is_tail, out);
    out.bytes(&[0x48, 0x89, 0x4B, 0x08]);
    x64_patch8(after_next, out);
    out.bytes(&[0x49, 0xC7, 0x44, 0x24, 0x08, 0, 0, 0, 0, 0x49, 0xC7, 0x44, 0x24, 0x10, 0, 0, 0, 0, 0x41, 0x80, 0x24, 0x24, 0xFD]);
}

fn emit_x64_drop_op(c: &Emit<'_>, push: bool, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    out.bytes(if c.win { &[0x48, 0x85, 0xC9, 0x75, 0x01, 0xC3] } else { &[0x48, 0x85, 0xFF, 0x75, 0x01, 0xC3] });
    if !push { out.bytes(if c.win { &[0xF6, 0x01, 0x02, 0x75, 0x01, 0xC3] } else { &[0xF6, 0x07, 0x02, 0x75, 0x01, 0xC3] }); }
    out.bytes(if c.win { &[0x53, 0x41, 0x54, 0x41, 0x55, 0x48, 0x83, 0xEC, 0x20, 0x49, 0x89, 0xCC] } else { &[0x53, 0x41, 0x54, 0x41, 0x55, 0x49, 0x89, 0xFC] });
    emit_x64_unwind_state(c, addrs, out)?;
    let no_state = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
    out.bytes(&[0x48, 0x89, 0xC3]);
    if push {
        let not_linked = x64_jcc(&[0x41, 0xF6, 0x04, 0x24, 0x02], 0x74, out);
        emit_x64_unlink_r12(out);
        x64_patch8(not_linked, out);
        let no_head = x64_jcc(&[0x48, 0x8B, 0x4B, 0x08, 0x49, 0x89, 0x4C, 0x24, 0x08, 0x49, 0xC7, 0x44, 0x24, 0x10, 0, 0, 0, 0, 0x41, 0x80, 0x0C, 0x24, 0x02, 0x48, 0x85, 0xC9], 0x74, out);
        out.bytes(&[0x4C, 0x89, 0x61, 0x10]);
        x64_patch8(no_head, out);
        out.bytes(&[0x4C, 0x89, 0x63, 0x08]);
    } else {
        emit_x64_unlink_r12(out);
    }
    x64_patch8(no_state, out);
    if c.win { out.bytes(&[0x48, 0x83, 0xC4, 0x20]); }
    out.bytes(&[0x41, 0x5D, 0x41, 0x5C, 0x5B, 0xC3]);
    Ok(())
}

fn emit_x64_catch_unwind(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    let d: u8 = if c.win { 0x20 } else { 0 };
    out.bytes(&[0x55, 0x48, 0x89, 0xE5, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x81, 0xEC]);
    out.u32(0x238 + u32::from(d));
    out.bytes(if c.win { &[0x49, 0x89, 0xCC, 0x49, 0x89, 0xD6, 0x4D, 0x89, 0xC7] } else { &[0x49, 0x89, 0xFC, 0x49, 0x89, 0xF6, 0x49, 0x89, 0xD7] });
    emit_x64_unwind_state(c, addrs, out)?;
    let no_state = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
    out.bytes(&[
        0x48, 0x89, 0xC3, 0x48, 0x8B, 0x03, 0x48, 0x89, 0x44, 0x24, d, 0x48, 0x8B, 0x43, 0x08, 0x48, 0x89, 0x44, 0x24, d + 8,
        0x48, 0xC7, 0x44, 0x24, d + 0x10, 0, 0, 0, 0, 0x48, 0xC7, 0x44, 0x24, d + 0x18, 0, 0, 0, 0, 0x48, 0x89, 0xAC, 0x24,
    ]);
    out.u32(super::unwind::STOP_FP_OFFSET + u32::from(d));
    out.bytes(&[0x48, 0x8D, 0x44, 0x24, d, 0x48, 0x89, 0x03]);
    let jmp_args = if c.win { &[0x48, 0x8D, 0x4C, 0x24, d + 0x20, 0x31, 0xD2][..] } else { &[0x48, 0x8D, 0x7C, 0x24, d + 0x20] };
    c.bytes_call(jmp_args, "_setjmp", out)?;
    let caught = x64_jcc(&[0x85, 0xC0], 0x75, out);
    let call_try: &[u8] = if c.win { &[0x4C, 0x89, 0xF1, 0x41, 0xFF, 0xD4] } else { &[0x4C, 0x89, 0xF7, 0x41, 0xFF, 0xD4] };
    out.bytes(call_try);
    let ok_zero = x64_jcc(&[0x48, 0x8B, 0x44, 0x24, d, 0x48, 0x89, 0x03], 0xEB, out);
    x64_patch8(no_state, out);
    out.bytes(call_try);
    x64_patch8(ok_zero, out);
    let done = x64_jcc(&[0x31, 0xC0], 0xEB, out);
    x64_patch8(caught, out);
    let seq = if c.win { [0x4C, 0x89, 0xF1, 0x48, 0x8D, 0x54, 0x24, d + 0x10] } else { [0x4C, 0x89, 0xF7, 0x48, 0x8D, 0x74, 0x24, d + 0x10] };
    out.bytes(&seq);
    out.bytes(&[0x41, 0xFF, 0xD7, 0xB8, 1, 0, 0, 0]);
    x64_patch8(done, out);
    out.bytes(&[0x48, 0x81, 0xC4]);
    out.u32(0x238 + u32::from(d));
    out.bytes(&[0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0x5D, 0xC3]);
    Ok(())
}

fn emit_x64_start_panic(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    out.bytes(&[0x55, 0x48, 0x89, 0xE5, 0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57]);
    out.bytes(if c.win { &[0x48, 0x83, 0xEC, 0x28, 0x49, 0x89, 0xCC, 0x49, 0x89, 0xD6] } else { &[0x48, 0x83, 0xEC, 0x08, 0x49, 0x89, 0xFC, 0x49, 0x89, 0xF6] });
    emit_x64_unwind_state(c, addrs, out)?;
    out.bytes(&[0x48, 0x85, 0xC0]);
    let no_state = x64_fwd32(&[0x0F, 0x84], out);
    out.bytes(&[0x48, 0x89, 0xC3, 0x4C, 0x8B, 0x3B, 0x4D, 0x85, 0xFF]);
    let no_frame = x64_fwd32(&[0x0F, 0x84], out);
    out.bytes(&[0x49, 0x8B, 0x07, 0x48, 0x89, 0x03, 0x49, 0x8B, 0x46, 0x18]);
    out.bytes(if c.win { &[0x4C, 0x89, 0xE1] } else { &[0x4C, 0x89, 0xE7] });
    out.bytes(&[0xFF, 0xD0, 0x49, 0x89, 0x47, 0x10, 0x49, 0x89, 0x57, 0x18, 0x4D, 0x8B, 0x77, 0x08]);
    let drain_loop = out.pos();
    let drain_null = x64_jcc(&[0x4C, 0x8B, 0x63, 0x08, 0x4D, 0x85, 0xE4], 0x74, out);
    let drain_mark = x64_jcc(&[0x4D, 0x39, 0xF4], 0x74, out);
    emit_x64_unlink_r12(out);
    let skip_inactive = x64_jcc(&[0x41, 0x0F, 0xB6, 0x04, 0x24, 0xA8, 0x01], 0x74, out);
    out.bytes(&[0x24, 0xFE, 0x41, 0x88, 0x04, 0x24, 0x49, 0x8B, 0x44, 0x24, 0x18]);
    out.bytes(if c.win { &[0x49, 0x8B, 0x4C, 0x24, 0x20, 0xFF, 0xD0] } else { &[0x49, 0x8B, 0x7C, 0x24, 0x20, 0xFF, 0xD0] });
    x64_patch8(skip_inactive, out);
    out.u8(0xE9);
    out.u32(((drain_loop as i64 - (out.pos() as i64 + 4)) as i32) as u32);
    x64_here(&[drain_null, drain_mark], out);
    if addrs.unwind_table != 0 { super::unwind::emit_x64_walk_call(c, addrs, out); }
    let longjmp_args = if c.win { &[0x49, 0x8D, 0x4F, 0x20, 0xBA, 1, 0, 0, 0][..] } else { &[0x49, 0x8D, 0x7F, 0x20, 0xBE, 1, 0, 0, 0] };
    c.bytes_call(longjmp_args, c.pick("longjmp", "_longjmp"), out)?;
    [no_state, no_frame].into_iter().for_each(|l| x64_patch32(l, out));
    c.call("abort", out)?;
    out.bytes(&[0x0F, 0x0B]);
    Ok(())
}

fn emit_x64_thread_cell(c: &Emit<'_>, key_va: u64, out: &mut Buf) -> Result<()> {
    out.bytes(&[0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x83, 0xEC, if c.win { 0x20 } else { 0x10 }]);
    out.bytes(if c.win { &[0x48, 0x89, 0xCB, 0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xD2, 0x48, 0x0F, 0x44, 0xD0, 0x49, 0x89, 0xD4] } else { &[0x48, 0x89, 0xFB, 0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xF6, 0x48, 0x0F, 0x44, 0xF0, 0x49, 0x89, 0xF4] });
    emit_x64_tls_get(c, key_va, out)?;
    let grow_null = x64_jcc(&[0x49, 0x89, 0xC6, 0x4D, 0x85, 0xF6], 0x74, out);
    let grow_cap = x64_jcc(&[0x49, 0x3B, 0x1E], 0x73, out);
    out.bytes(&[0x49, 0x8B, 0x44, 0xDE, 0x08, 0x48, 0x85, 0xC0]);
    let hit = x64_fwd32(&[0x0F, 0x85], out);
    let cell_skip = x64_fwd32(&[0xE9], out);
    x64_here(&[grow_null, grow_cap], out);
    out.bytes(&[0x31, 0xC0, 0x4D, 0x85, 0xF6, 0x74, 0x03, 0x49, 0x8B, 0x06, 0x48, 0x01, 0xC0, 0x4C, 0x8D, 0x7B, 0x08, 0x4C, 0x39, 0xF8, 0x4C, 0x0F, 0x47, 0xF8]);
    let calloc_args = if c.win { &[0x49, 0x8D, 0x4F, 1, 0xBA, 8, 0, 0, 0][..] } else { &[0x49, 0x8D, 0x7F, 1, 0xBE, 8, 0, 0, 0] };
    c.bytes_call(calloc_args, "calloc", out)?;
    let fail_calloc = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
    let no_copy = x64_jcc(&[0x4C, 0x89, 0x38, 0x49, 0x89, 0xC7, 0x4D, 0x85, 0xF6], 0x74, out);
    let memcpy_args = if c.win { &[0x49, 0x8D, 0x4F, 8, 0x49, 0x8D, 0x56, 8, 0x4D, 0x8B, 0x06, 0x49, 0xC1, 0xE0, 3][..] } else { &[0x49, 0x8D, 0x7F, 8, 0x49, 0x8D, 0x76, 8, 0x49, 0x8B, 0x16, 0x48, 0xC1, 0xE2, 3] };
    c.bytes_call(memcpy_args, "memcpy", out)?;
    c.bytes_call(if c.win { &[0x4C, 0x89, 0xF1] } else { &[0x4C, 0x89, 0xF7] }, "free", out)?;
    x64_patch8(no_copy, out);
    let set_args = if c.win { &[0x4D, 0x89, 0xFE, 0x41, 0x8B, 0x4D, 8, 0x4C, 0x89, 0xF2][..] } else { &[0x4D, 0x89, 0xFE, 0x49, 0x8B, 0x7D, 8, 0x4C, 0x89, 0xF6] };
    c.bytes_call(set_args, c.pick("TlsSetValue", "pthread_setspecific"), out)?;
    let fail_set = x64_jcc(&[0x85, 0xC0], if c.win { 0x74 } else { 0x75 }, out);
    x64_patch32(cell_skip, out);
    let fail_align = if c.win {
        c.bytes_call(&[0x4C, 0x89, 0xE1, 0xBA, 0x40, 0, 0, 0], "_aligned_malloc", out)?;
        let f = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
        c.bytes_call(&[0x48, 0x89, 0xC1, 0x31, 0xD2, 0x4D, 0x89, 0xE0], "memset", out)?;
        f
    } else {
        c.bytes_call(&[0x48, 0xC7, 0x04, 0x24, 0, 0, 0, 0, 0x48, 0x89, 0xE7, 0xBE, 0x40, 0, 0, 0, 0x4C, 0x89, 0xE2], "posix_memalign", out)?;
        let f = x64_jcc(&[0x85, 0xC0], 0x75, out);
        c.bytes_call(&[0x48, 0x8B, 0x3C, 0x24, 0x31, 0xF6, 0x4C, 0x89, 0xE2], "memset", out)?;
        f
    };
    let done_ok = x64_jcc(&[0x49, 0x89, 0x44, 0xDE, 0x08], 0xEB, out);
    x64_here(&[fail_calloc, fail_set, fail_align], out);
    out.bytes(&[0x31, 0xC0]);
    x64_patch32(hit, out);
    x64_patch8(done_ok, out);
    out.bytes(&[0x48, 0x83, 0xC4, if c.win { 0x20 } else { 0x10 }, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0xC3]);
    Ok(())
}

/// The x86-64 ELF thread cell: the cell pointer of slot `rdi` sits in the executable's static TLS block `tls` bytes below the thread pointer; a null slot gets `rsi` zeroed bytes aligned to 64.
fn emit_x64_tls_cell(c: &Emit<'_>, tls: u64, out: &mut Buf) -> Result<()> {
    let disp = (tls as i64).wrapping_neg() as i32 as u32;
    out.bytes(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0, 0, 0, 0, 0x48, 0x8B, 0x8C, 0xF8]);
    out.u32(disp);
    let slow = x64_jcc(&[0x48, 0x85, 0xC9], 0x74, out);
    out.bytes(&[0x48, 0x89, 0xC8, 0xC3]);
    x64_patch8(slow, out);
    out.bytes(&[0x53, 0x41, 0x54, 0x48, 0x83, 0xEC, 0x08, 0x48, 0x8D, 0x9C, 0xF8]);
    out.u32(disp);
    c.bytes_call(&[0x49, 0x89, 0xF4, 0x48, 0xC7, 0x04, 0x24, 0, 0, 0, 0, 0x48, 0x89, 0xE7, 0xBE, 0x40, 0, 0, 0, 0x4C, 0x89, 0xE2], "posix_memalign", out)?;
    let fail = x64_jcc(&[0x85, 0xC0], 0x75, out);
    c.bytes_call(&[0x48, 0x8B, 0x3C, 0x24, 0x31, 0xF6, 0x4C, 0x89, 0xE2], "memset", out)?;
    let done = x64_jcc(&[0x48, 0x89, 0x03], 0xEB, out);
    x64_patch8(fail, out);
    out.bytes(&[0x31, 0xC0]);
    x64_patch8(done, out);
    out.bytes(&[0x48, 0x83, 0xC4, 0x08, 0x41, 0x5C, 0x5B, 0xC3]);
    Ok(())
}

/// The x86-64 ELF coroutine frame cache: per thread, one free list of up to 64 frames for each 16-byte size class below 1024 bytes, `tls` bytes below the thread pointer, indexed by the size the caller passes as a 16-byte multiple; other sizes and misses go to `malloc` and `free`.
fn emit_x64_frame_cache(c: &Emit<'_>, alloc: bool, tls: u64, out: &mut Buf) -> Result<()> {
    let disp = (tls as i64).wrapping_neg() as i32 as u32;
    let (size_reg, sib) = if alloc { (0xFF, 0x3A) } else { (0xFE, 0x32) };
    out.bytes(&[0x48, 0x81, size_reg]);
    out.u32(haste_codegen::FRAME_CACHE_LIMIT);
    let big = x64_fwd(0x73, out);
    out.bytes(&[0x64, 0x48, 0x8B, 0x14, 0x25, 0, 0, 0, 0, 0x48, 0x8D, 0x94, sib]);
    out.u32(disp);
    let miss = if alloc {
        let m = x64_jcc(&[0x48, 0x8B, 0x02, 0x48, 0x85, 0xC0], 0x74, out);
        out.bytes(&[0x48, 0x8B, 0x08, 0x48, 0x89, 0x0A, 0x48, 0xFF, 0x4A, 0x08, 0xC3]);
        m
    } else {
        let full = x64_jcc(&[0x48, 0x83, 0x7A, 0x08, haste_codegen::FRAME_CACHE_DEPTH], 0x73, out);
        out.bytes(&[0x48, 0x8B, 0x0A, 0x48, 0x89, 0x0F, 0x48, 0x89, 0x3A, 0x48, 0xFF, 0x42, 0x08, 0xC3]);
        full
    };
    x64_here(&[big, miss], out);
    c.tail(if alloc { "malloc" } else { "free" }, out)
}

fn emit_arm64_tls_get(c: &Emit<'_>, key_va: u64, out: &mut Buf) -> Result<()> {
    emit_arm64_adrp_add(c.at(out), key_va, 21, out);
    let ready_fast = arm64_jcc(&[0x88DF_FEA8, 0x7100_091F], 0x5400_0000, out);
    out.u32(0x5280_002A);
    let init_loop = out.pos();
    out.u32(0x885F_FEA8);
    let already_init = arm64_fwd(0x3500_0008, out);
    out.u32(0x8809_FEAA);
    let retry_cas = arm64_fwd(0x3500_0009, out);
    arm64_patch(retry_cas, init_loop, out);
    if c.win {
        c.call("TlsAlloc", out)?;
        out.u32(0xB900_0AA0);
    } else {
        c.words_call(&[0x9100_22A0, 0xAA1F_03E1], "pthread_key_create", out)?;
    }
    let ready_create = arm64_jcc(&[0x5280_0048, 0x889F_FEA8], 0x1400_0000, out);
    arm64_here(&[already_init], out);
    let ready_wait = arm64_jcc(&[0x7100_091F], 0x5400_0000, out);
    c.call(c.pick("SwitchToThread", "sched_yield"), out)?;
    let back = arm64_fwd(0x1400_0000, out);
    arm64_patch(back, init_loop, out);
    arm64_here(&[ready_fast, ready_create, ready_wait], out);
    c.words_call(&[if c.win { 0xB940_0AA0 } else { 0xF940_06A0 }], c.pick("TlsGetValue", "pthread_getspecific"), out)
}

fn emit_arm64_unwind_state(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if addrs.tlv_desc != 0 {
        super::tls::emit_arm64_tlv_unwind_state(c, addrs.tlv_desc, addrs.unwind_tls, out);
        return Ok(());
    }
    emit_arm64_tls_get(c, addrs.thread_key + 16, out)?;
    let hit = arm64_fwd(0xB500_0000, out);
    c.words_call(&[0xD280_0020, 0xD280_0201], "calloc", out)?;
    let fail = arm64_fwd(0xB400_0000, out);
    c.words_call(&[0xAA00_03F3, if c.win { 0xB940_0AA0 } else { 0xF940_06A0 }, 0xAA13_03E1], c.pick("TlsSetValue", "pthread_setspecific"), out)?;
    out.u32(0xAA13_03E0);
    arm64_here(&[hit, fail], out);
    Ok(())
}

fn emit_arm64_unlink_x20(out: &mut Buf) {
    let no_prev = arm64_jcc(&[0xF940_0689, 0xF940_0A8A], 0xB400_0009, out);
    out.u32(0xF900_092A);
    arm64_here(&[no_prev], out);
    let is_tail = arm64_fwd(0xB400_000A, out);
    let after_next = arm64_jcc(&[0xF900_0549], 0x1400_0000, out);
    arm64_here(&[is_tail], out);
    out.u32(0xF900_0669);
    arm64_here(&[after_next], out);
    out.words(&[0xF900_069F, 0xF900_0A9F, 0x3940_0288, 0x121E_7908, 0x3900_0288]);
}

fn emit_arm64_drop_op(c: &Emit<'_>, push: bool, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    let ret_null = arm64_fwd(0xB400_0000, out);
    let ret_unlinked = if push { None } else { Some(arm64_jcc(&[0x3940_0008, 0x721F_011F], 0x5400_0000, out)) };
    out.words(&[0xA9BD_7BFD, 0xA901_53F3, 0xF900_13F5, 0xAA00_03F4]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    out.u32(0xAA00_03F3);
    if push {
        let not_linked = arm64_jcc(&[0x3940_0288, 0x721F_011F], 0x5400_0000, out);
        emit_arm64_unlink_x20(out);
        arm64_here(&[not_linked], out);
        let no_head = arm64_jcc(&[0xF940_0668, 0xF900_0688, 0xF900_0A9F, 0x3940_0289, 0x321F_0129, 0x3900_0289], 0xB400_0008, out);
        out.u32(0xF900_0914);
        arm64_here(&[no_head], out);
        out.u32(0xF900_0674);
    } else {
        emit_arm64_unlink_x20(out);
    }
    arm64_here(&[no_state], out);
    out.words(&[0xF940_13F5, 0xA941_53F3, 0xA8C3_7BFD]);
    arm64_here(&[ret_null], out);
    if let Some(lbl) = ret_unlinked { arm64_here(&[lbl], out); }
    out.u32(0xD65F_03C0);
    Ok(())
}

fn emit_arm64_catch_unwind(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    out.words(&[0xA9BC_7BFD, 0x9100_03FD, 0xA901_53F3, 0xA902_5BF5, 0xA903_63F7, 0xD108_C3FF, 0xAA00_03F4, 0xAA01_03F6, 0xAA02_03F7]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    out.words(&[0xAA00_03F3, 0xA940_2668, 0xA900_27E8, 0xA901_7FFF, 0xF900_03FD | ((super::unwind::STOP_FP_OFFSET / 8) << 10), 0x9100_03E8, 0xF900_0268, 0x9100_83E0]);
    if c.win { out.u32(0xAA1F_03E1); }
    c.call("_setjmp", out)?;
    let caught = arm64_fwd(0x3500_0000, out);
    let ok_zero = arm64_jcc(&[0xAA16_03E0, 0xD63F_0280, 0xF940_03E8, 0xF900_0268], 0x1400_0000, out);
    arm64_here(&[no_state], out);
    out.words(&[0xAA16_03E0, 0xD63F_0280]);
    arm64_here(&[ok_zero], out);
    let done = arm64_jcc(&[0x5280_0000], 0x1400_0000, out);
    arm64_here(&[caught], out);
    out.words(&[0xAA16_03E0, 0x9100_43E1, 0xD63F_02E0, 0x5280_0020]);
    arm64_here(&[done], out);
    out.words(&[0x9108_C3FF, 0xA943_63F7, 0xA942_5BF5, 0xA941_53F3, 0xA8C4_7BFD, 0xD65F_03C0]);
    Ok(())
}

fn emit_arm64_start_panic(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    out.words(&[0xA9BC_7BFD, 0x9100_03FD, 0xA901_53F3, 0xA902_5BF5, 0xA903_63F7, 0xAA00_03F4, 0xAA01_03F6]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    let no_frame = arm64_jcc(&[0xAA00_03F3, 0xF940_0277], 0xB400_0017, out);
    out.words(&[0xF940_02E8, 0xF900_0268, 0xF940_0EC8, 0xAA14_03E0, 0xD63F_0100, 0xA901_06E0, 0xF940_06F8]);
    let drain_loop = out.pos();
    out.u32(0xF940_0674);
    let drain_null = arm64_fwd(0xB400_0014, out);
    out.u32(0xEB18_029F);
    let drain_mark = arm64_fwd(0x5400_0000, out);
    emit_arm64_unlink_x20(out);
    let skip_inactive = arm64_jcc(&[0x7200_011F], 0x5400_0000, out);
    out.words(&[0x121F_7908, 0x3900_0288, 0xF940_0E88, 0xF940_1280, 0xD63F_0100]);
    arm64_here(&[skip_inactive], out);
    let back = arm64_fwd(0x1400_0000, out);
    arm64_patch(back, drain_loop, out);
    arm64_here(&[drain_null, drain_mark], out);
    if addrs.unwind_table != 0 { super::unwind::emit_arm64_walk_call(c, addrs, out); }
    c.words_call(&[0x9100_82E0, 0x5280_0021], c.pick("longjmp", "_longjmp"), out)?;
    arm64_here(&[no_state, no_frame], out);
    c.call("abort", out)?;
    out.u32(0xD420_0020);
    Ok(())
}

fn emit_arm64_thread_cell(c: &Emit<'_>, key_va: u64, out: &mut Buf) -> Result<()> {
    out.words(&[0xD101_03FF, 0xA900_7BFD, 0xA901_53F3, 0xA902_5BF5, 0xF900_1BF7, 0xAA00_03F3, 0xF100_003F, 0xD280_0029, 0x9A89_1034]);
    emit_arm64_tls_get(c, key_va, out)?;
    out.u32(0xAA00_03F6);
    let grow_null = arm64_fwd(0xB400_0016, out);
    let grow_cap = arm64_jcc(&[0xF940_02C8, 0xEB08_027F], 0x5400_0002, out);
    let hit = arm64_jcc(&[0x8B13_0EC8, 0xF940_0500], 0xB500_0000, out);
    let cell_skip = arm64_fwd(0x1400_0000, out);
    arm64_here(&[grow_null, grow_cap], out);
    c.words_call(&[0xD280_0008, 0xB400_0056, 0xF940_02C8, 0x8B08_0108, 0x9100_2277, 0xEB17_011F, 0x9A97_8117, 0x9100_06E0, 0xD280_0101], "calloc", out)?;
    let fail_calloc = arm64_fwd(0xB400_0000, out);
    let no_copy = arm64_jcc(&[0xF900_0017, 0xAA00_03F7], 0xB400_0016, out);
    c.words_call(&[0x9100_22E0, 0x9100_22C1, 0xF940_02C2, 0xD37D_F042], "memcpy", out)?;
    c.words_call(&[0xAA16_03E0], "free", out)?;
    arm64_here(&[no_copy], out);
    c.words_call(&[0xAA17_03F6, if c.win { 0xB940_0AA0 } else { 0xF940_06A0 }, 0xAA16_03E1], c.pick("TlsSetValue", "pthread_setspecific"), out)?;
    let fail_set = arm64_fwd(if c.win { 0x3400_0000 } else { 0x3500_0000 }, out);
    arm64_here(&[cell_skip], out);
    let fail_align = if c.win {
        c.words_call(&[0xAA14_03E0, 0xD280_0801], "_aligned_malloc", out)?;
        arm64_fwd(0xB400_0000, out)
    } else {
        c.words_call(&[0xF900_1FFF, 0x9100_E3E0, 0xD280_0801, 0xAA14_03E2], "posix_memalign", out)?;
        let f = arm64_fwd(0x3500_0000, out);
        out.u32(0xF940_1FE0);
        f
    };
    c.words_call(&[0x5280_0001, 0xAA14_03E2], "memset", out)?;
    out.words(&[0x8B13_0EC8, 0xF900_0500, 0x1400_0002]);
    arm64_here(&[fail_calloc, fail_set, fail_align], out);
    out.u32(0xD280_0000);
    arm64_here(&[hit], out);
    out.words(&[0xF940_1BF7, 0xA942_5BF5, 0xA941_53F3, 0xA940_7BFD, 0x9101_03FF, 0xD65F_03C0]);
    Ok(())
}

fn emit_shim_x64(os: Os, shim: Shim, va: u64, addrs: SectionAddrs, stub_va: &dyn Fn(&str) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let c = Emit { arch: Arch::X64, win: os == Os::Windows, va, base: out.pos(), stub_va };
    match shim {
        Shim::Noop | Shim::Barrier => out.u8(0xC3),
        Shim::Syscall if os == Os::Linux => out.bytes(&[
            0x48, 0x8B, 0x07, 0x48, 0x8B, 0x77, 0x10, 0x48, 0x8B, 0x57, 0x18, 0x4C, 0x8B, 0x57, 0x20, 0x4C, 0x8B, 0x47, 0x28, 0x4C, 0x8B, 0x4F, 0x30, 0x48, 0x8B, 0x7F, 0x08, 0x0F, 0x05, 0xC3,
        ]),
        Shim::Syscall => out.bytes(&[0x48, 0xC7, 0xC0, 0xDA, 0xFF, 0xFF, 0xFF, 0xC3]),
        Shim::Cpuid => {
            out.bytes(if c.win { &[0x53, 0x41, 0x89, 0xD1, 0x89, 0xD0, 0x45, 0x89, 0xC2, 0x49, 0x89, 0xC8, 0x44, 0x89, 0xD1, 0x0F, 0xA2] } else { &[0x53, 0x49, 0x89, 0xF8, 0x41, 0x89, 0xF1, 0x41, 0x89, 0xD2, 0x89, 0xF0, 0x89, 0xD1, 0x0F, 0xA2] });
            for (prefix, mask) in [
                (&[0x41, 0x83, 0xF9, 0x01, 0x75, 0x06, 0x81, 0xE1][..], !HIDDEN.leaf1_ecx),
                (&[0x41, 0x83, 0xF9, 0x07, 0x75, 0x2A, 0x45, 0x85, 0xD2, 0x75, 0x14, 0x81, 0xE3], !HIDDEN.leaf7_ebx),
                (&[0x81, 0xE1], !HIDDEN.leaf7_ecx),
                (&[0x81, 0xE2], !HIDDEN.leaf7_edx),
                (&[0xEB, 0x11, 0x41, 0x83, 0xFA, 0x01, 0x75, 0x0B, 0x25], !HIDDEN.leaf7_1_eax),
                (&[0x81, 0xE2], !HIDDEN.leaf7_1_edx),
                (&[0x41, 0x81, 0xF9, 0x01, 0x00, 0x00, 0x80, 0x75, 0x06, 0x81, 0xE1], !HIDDEN.extended_ecx),
            ] {
                out.bytes(prefix);
                out.u32(mask);
            }
            out.bytes(&[0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08, 0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]);
        }
        Shim::ExtendedState => out.bytes(&[
            0x53, 0xB8, 1, 0, 0, 0, 0x0F, 0xA2, 0xF7, 0xC1, 0, 0, 0, 8, 0x75, 4, 0x31, 0xC0, 0x5B, 0xC3,
            0x89, if c.win { 0xC9 } else { 0xF9 }, 0x0F, 0x01, 0xD0, 0x48, 0xC1, 0xE2, 0x20, 0x48, 0x09, 0xD0,
            0x85, if c.win { 0xC9 } else { 0xFF }, 0x75, 2, 0x24, !HIDDEN.xcr0 as u8, 0x5B, 0xC3,
        ]),
        Shim::Dealloc => c.tail(c.pick("_aligned_free", "free"), out)?,
        Shim::Alloc if c.win => c.bytes_tail(&[0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xC9, 0x48, 0x0F, 0x44, 0xC8, 0x48, 0x85, 0xD2, 0x48, 0x0F, 0x44, 0xD0], "_aligned_malloc", out)?,
        Shim::Alloc => {
            c.bytes_tail(&[0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xFF, 0x48, 0x0F, 0x44, 0xF8, 0x48, 0x83, 0xFE, 0x10, 0x77, 0x05], "malloc", out)?;
            c.bytes_call(&[0x48, 0x83, 0xEC, 0x18, 0x48, 0xC7, 0x04, 0x24, 0, 0, 0, 0, 0x48, 0x89, 0xFA, 0x48, 0x8D, 0x3C, 0x24], "posix_memalign", out)?;
            out.bytes(&[0x85, 0xC0, 0x48, 0x8B, 0x04, 0x24, 0x31, 0xD2, 0x48, 0x0F, 0x45, 0xC2, 0x48, 0x83, 0xC4, 0x18, 0xC3]);
        }
        Shim::AllocZeroed if c.win => {
            c.bytes_call(&[0x53, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x89, 0xCB, 0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xC9, 0x48, 0x0F, 0x44, 0xC8, 0x48, 0x85, 0xD2, 0x48, 0x0F, 0x44, 0xD0], "_aligned_malloc", out)?;
            let done = x64_jcc(&[0x48, 0x85, 0xC0], 0x74, out);
            c.bytes_call(&[0x48, 0x89, 0xC1, 0x31, 0xD2, 0x49, 0x89, 0xD8], "memset", out)?;
            x64_patch8(done, out);
            out.bytes(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]);
        }
        Shim::AllocZeroed => {
            c.bytes_tail(&[0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xFF, 0x48, 0x0F, 0x44, 0xF8, 0x48, 0x83, 0xFE, 0x10, 0x77, 0x0D, 0x48, 0x89, 0xFE, 0xBF, 1, 0, 0, 0], "calloc", out)?;
            c.bytes_call(&[0x53, 0x48, 0x83, 0xEC, 0x10, 0x48, 0x89, 0xFB, 0x48, 0x89, 0xFA, 0x48, 0xC7, 0x04, 0x24, 0, 0, 0, 0, 0x48, 0x8D, 0x3C, 0x24], "posix_memalign", out)?;
            c.bytes_call(&[0x85, 0xC0, 0x75, 0x10, 0x48, 0x8B, 0x3C, 0x24, 0x31, 0xF6, 0x48, 0x89, 0xDA], "memset", out)?;
            out.bytes(&[0xEB, 0x02, 0x31, 0xC0, 0x48, 0x83, 0xC4, 0x10, 0x5B, 0xC3]);
        }
        Shim::Realloc if c.win => c.bytes_tail(&[0xB8, 1, 0, 0, 0, 0x4D, 0x85, 0xC9, 0x4C, 0x0F, 0x44, 0xC8, 0x4C, 0x89, 0xCA, 0x4D, 0x85, 0xC0, 0x4C, 0x0F, 0x44, 0xC0], "_aligned_realloc", out)?,
        Shim::Realloc => {
            c.bytes_tail(&[0xB8, 1, 0, 0, 0, 0x48, 0x85, 0xC9, 0x48, 0x0F, 0x44, 0xC8, 0x48, 0x83, 0xFA, 0x10, 0x77, 0x08, 0x48, 0x89, 0xCE], "realloc", out)?;
            c.bytes_call(&[0x53, 0x41, 0x54, 0x41, 0x55, 0x48, 0x83, 0xEC, 0x10, 0x48, 0x89, 0xFB, 0x49, 0x89, 0xF4, 0x49, 0x89, 0xCD, 0x48, 0xC7, 0x04, 0x24, 0, 0, 0, 0, 0x48, 0x8D, 0x3C, 0x24, 0x48, 0x89, 0xD6, 0x48, 0x89, 0xCA], "posix_memalign", out)?;
            c.bytes_call(&[0x85, 0xC0, 0x75, 0x26, 0x48, 0x8B, 0x3C, 0x24, 0x48, 0x89, 0xDE, 0x4C, 0x89, 0xE2, 0x4D, 0x39, 0xEC, 0x49, 0x0F, 0x47, 0xD5], "memcpy", out)?;
            c.bytes_call(&[0x48, 0x89, 0xDF, 0x48, 0x89, 0xC3], "free", out)?;
            out.bytes(&[0x48, 0x89, 0xD8, 0xEB, 0x02, 0x31, 0xC0, 0x48, 0x83, 0xC4, 0x10, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0xC3]);
        }
        Shim::FillOnce if c.win => {
            out.bytes(&[0x53, 0x56, 0x48, 0x83, 0xEC, 0x28, 0x48, 0x89, 0xCB, 0x48, 0x89, 0xD6]);
            let retry = out.pos();
            let won = x64_jcc(&[0x31, 0xC0, 0xB2, 0x02, 0xF0, 0x0F, 0xB0, 0x13], 0x74, out);
            let done = x64_jcc(&[0x3C, 0x01], 0x74, out);
            c.call("SwitchToThread", out)?;
            out.bytes(&[0xEB, ((retry as isize - (out.pos() as isize + 2)) as i8) as u8]);
            x64_patch8(won, out);
            out.bytes(&[0xFF, 0xD6, 0xC6, 0x03, 0x01]);
            x64_patch8(done, out);
            out.bytes(&[0x48, 0x83, 0xC4, 0x28, 0x5E, 0x5B, 0xC3]);
        }
        Shim::FillOnce => {
            c.bytes_call(&[0x53, 0x41, 0x54, 0x48, 0x89, 0xFB, 0x49, 0x89, 0xF4, 0x31, 0xC0, 0xB2, 2, 0xF0, 0x0F, 0xB0, 0x13, 0x74, 0x0B, 0x3C, 1, 0x74, 0x0D], "sched_yield", out)?;
            out.bytes(&[0xEB, 0xEB, 0x41, 0xFF, 0xD4, 0xC6, 0x03, 1, 0x41, 0x5C, 0x5B, 0xC3]);
        }
        Shim::ThreadCell if os == Os::Linux => emit_x64_tls_cell(&c, addrs.thread_tls, out)?,
        Shim::ThreadCell => emit_x64_thread_cell(&c, addrs.thread_key, out)?,
        Shim::FrameAlloc | Shim::FrameFree if os == Os::Linux => emit_x64_frame_cache(&c, shim == Shim::FrameAlloc, addrs.frame_tls, out)?,
        Shim::FrameAlloc | Shim::FrameFree => c.tail(if shim == Shim::FrameAlloc { "malloc" } else { "free" }, out)?,
        Shim::CatchUnwind => emit_x64_catch_unwind(&c, addrs, out)?,
        Shim::StartPanic => emit_x64_start_panic(&c, addrs, out)?,
        Shim::PanicCleanup => out.bytes(if c.win { &[0x48, 0x8B, 0x01, 0x48, 0x8B, 0x51, 0x08, 0xC3] } else { &[0x48, 0x8B, 0x07, 0x48, 0x8B, 0x57, 0x08, 0xC3] }),
        Shim::DropPush | Shim::DropPop => emit_x64_drop_op(&c, shim == Shim::DropPush, addrs, out)?,
        Shim::UnwindWalk => super::unwind::emit_walk(Arch::X64, os, out)?,
        Shim::Wide(wide) => super::wide::emit(Arch::X64, os, wide, out)?,
    }
    Ok(())
}

fn emit_shim_arm64(os: Os, shim: Shim, va: u64, addrs: SectionAddrs, stub_va: &dyn Fn(&str) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let c = Emit { arch: Arch::Arm64, win: os == Os::Windows, va, base: out.pos(), stub_va };
    match shim {
        Shim::Noop => out.u32(0xD65F_03C0),
        Shim::Barrier => out.words(&[0xD503_3F9F, 0xD503_3FDF, 0xD65F_03C0]),
        Shim::Syscall => out.words(&[if os == Os::Darwin { 0x9280_09A0 } else { 0x9280_04A0 }, 0xD65F_03C0]),
        Shim::Cpuid => out.words(&[0xA900_7C1F, 0xD65F_03C0]),
        Shim::ExtendedState => out.words(&[0xD280_0000, 0xD65F_03C0]),
        Shim::Dealloc => c.tail(c.pick("_aligned_free", "free"), out)?,
        Shim::FrameAlloc | Shim::FrameFree if addrs.tlv_desc != 0 => super::tls::emit_arm64_tlv_frame_cache(&c, shim == Shim::FrameAlloc, addrs.tlv_desc, addrs.frame_tls, out)?,
        Shim::FrameAlloc | Shim::FrameFree => c.tail(if shim == Shim::FrameAlloc { "malloc" } else { "free" }, out)?,
        Shim::Alloc if c.win => c.words_tail(&[0xF100_001F, 0xD280_0029, 0x9A89_1000, 0xF100_003F, 0x9A89_1021], "_aligned_malloc", out)?,
        Shim::Alloc => {
            c.words_tail(&[0xF100_001F, 0xD280_0029, 0x9A89_1000, 0xF100_403F, 0x5400_0048], "malloc", out)?;
            c.words_call(&[0xD100_83FF, 0xA900_7BFD, 0xAA00_03E2, 0xF900_0BFF, 0x9100_43E0], "posix_memalign", out)?;
            out.words(&[0x7100_001F, 0xF940_0BE0, 0x9A9F_0000, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::AllocZeroed if c.win => {
            c.words_call(&[0xA9BE_7BFD, 0xF900_0BF3, 0xAA00_03F3, 0xF100_001F, 0xD280_0029, 0x9A89_1000, 0xF100_003F, 0x9A89_1021], "_aligned_malloc", out)?;
            let done = arm64_fwd(0xB400_0000, out);
            c.words_call(&[0x5280_0001, 0xAA13_03E2], "memset", out)?;
            arm64_patch(done, out.pos(), out);
            out.words(&[0xF940_0BF3, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::AllocZeroed => {
            c.words_tail(&[0xF100_001F, 0xD280_0029, 0x9A89_1000, 0xF100_403F, 0x5400_0068, 0xAA00_03E1, 0xD280_0020], "calloc", out)?;
            c.words_call(&[0xD100_83FF, 0xA900_7BFD, 0xF900_0BF3, 0xAA00_03F3, 0xAA00_03E2, 0xF900_0FFF, 0x9100_63E0], "posix_memalign", out)?;
            c.words_call(&[0x3500_00C0, 0xF940_0FE0, 0x5280_0001, 0xAA13_03E2], "memset", out)?;
            out.words(&[0x1400_0002, 0xD280_0000, 0xF940_0BF3, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::Realloc if c.win => c.words_tail(&[0xF100_007F, 0xD280_0029, 0x9A89_1061, 0xF100_005F, 0x9A89_1042], "_aligned_realloc", out)?,
        Shim::Realloc => {
            c.words_tail(&[0xF100_007F, 0xD280_0029, 0x9A89_1063, 0xF100_405F, 0x5400_0068, 0xAA03_03E1], "realloc", out)?;
            c.words_call(&[0xD100_C3FF, 0xA900_7BFD, 0xA901_53F3, 0xF900_13F5, 0xAA00_03F3, 0xEB03_003F, 0x9A83_3034, 0xAA02_03E1, 0xAA03_03E2, 0xF900_17FF, 0x9100_A3E0], "posix_memalign", out)?;
            c.words_call(&[0x3500_0140, 0xF940_17F5, 0xAA15_03E0, 0xAA13_03E1, 0xAA14_03E2], "memcpy", out)?;
            c.words_call(&[0xAA13_03E0], "free", out)?;
            out.words(&[0xAA15_03E0, 0x1400_0002, 0xD280_0000, 0xA941_53F3, 0xF940_13F5, 0xA8C3_7BFD, 0xD65F_03C0]);
        }
        Shim::FillOnce => {
            out.words(&[0xA9BE_7BFD, 0xA901_53F3, 0xAA00_03F3, 0xAA01_03F4, 0x5280_004A]);
            let retry = out.pos();
            c.words_call(&[0x085F_FE68, 0x3500_00E8, 0x0809_FE6A, 0x35FF_FFA9, 0xD63F_0280, 0x5280_0028, 0x089F_FE68, 0x1400_0005, 0x7100_051F, 0x5400_0060], c.pick("SwitchToThread", "sched_yield"), out)?;
            let back = ((retry as isize - out.pos() as isize) >> 2) as u32;
            out.words(&[0x1400_0000 | (back & 0x03FF_FFFF), 0xA941_53F3, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::ThreadCell if addrs.tlv_desc != 0 => super::tls::emit_arm64_tlv_thread_cell(&c, addrs.tlv_desc, addrs.thread_tls, out)?,
        Shim::ThreadCell => emit_arm64_thread_cell(&c, addrs.thread_key, out)?,
        Shim::CatchUnwind => emit_arm64_catch_unwind(&c, addrs, out)?,
        Shim::StartPanic => emit_arm64_start_panic(&c, addrs, out)?,
        Shim::PanicCleanup => out.words(&[0xA940_0400, 0xD65F_03C0]),
        Shim::DropPush | Shim::DropPop => emit_arm64_drop_op(&c, shim == Shim::DropPush, addrs, out)?,
        Shim::UnwindWalk => super::unwind::emit_walk(Arch::Arm64, os, out)?,
        Shim::Wide(wide) => super::wide::emit(Arch::Arm64, os, wide, out)?,
    }
    Ok(())
}

pub(super) fn emit_shims(arch: Arch, os: Os, plan: &Plan<'_>, base_va: u64, shims_off: usize, addrs: SectionAddrs, stub_va: &dyn Fn(&str) -> Result<u64>, mut offs: Option<&mut Vec<usize>>, out: &mut Buf) -> Result<()> {
    for &shim in &plan.shims {
        out.align(16);
        let at = shims_off + out.pos();
        if let Some(list) = offs.as_deref_mut() { list.push(at); }
        let emit = if arch == Arch::X64 { emit_shim_x64 } else { emit_shim_arm64 };
        emit(os, shim, base_va + at as u64, addrs, stub_va, out)?;
    }
    Ok(())
}

pub(super) fn plan_shims(arch: Arch, os: Os, plan: &Plan<'_>, base_va: u64, shims_off: usize, dummy: SectionAddrs) -> Result<(Vec<usize>, usize)> {
    let mut shim_offs = Vec::with_capacity(plan.shims.len());
    let mut probe = Buf::new();
    let zero_stub = |_: &str| Ok(base_va);
    emit_shims(arch, os, plan, base_va, shims_off, dummy, &zero_stub, Some(&mut shim_offs), &mut probe)?;
    Ok((shim_offs, (shims_off + probe.pos()).next_multiple_of(16)))
}
