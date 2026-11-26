use anyhow::{Result, bail};
use haste_codegen::{Arch, Os};

use super::wide::WideShim;
use super::{Buf, Plan, SectionAddrs};

mod arm64;
mod x64;

pub(super) use arm64::emit_arm64_unwind_state;
pub(super) use x64::emit_x64_unwind_state;

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

const PTHREAD_DEPS: &[&str] = &[
    "pthread_key_create",
    "pthread_getspecific",
    "pthread_setspecific",
    "calloc",
    "sched_yield",
];
pub(super) const KERNEL32_DLL: &str = "kernel32.dll";
pub(super) const UCRTBASE_DLL: &str = "ucrtbase.dll";
pub(super) const MSVCRT_DLL: &str = "msvcrt.dll";

const WIN_TLS_DEPS: &[(&str, &str)] = &[
    ("TlsAlloc", KERNEL32_DLL),
    ("TlsGetValue", KERNEL32_DLL),
    ("TlsSetValue", KERNEL32_DLL),
    ("calloc", UCRTBASE_DLL),
    ("SwitchToThread", KERNEL32_DLL),
];

fn collect_deps<T: Copy>(
    shim: Shim,
    tls: &[T],
    mem: (T, T, T),
    jmp: Option<(T, T)>,
    crt: impl Fn(&'static str) -> T,
) -> Vec<T> {
    let (alloc_aligned, yield_fn, free_fn) = mem;
    match shim {
        Shim::Alloc => vec![crt("malloc"), alloc_aligned],
        Shim::AllocZeroed => vec![crt("calloc"), alloc_aligned, crt("memset")],
        Shim::Dealloc | Shim::FrameFree => vec![free_fn],
        Shim::FrameAlloc => vec![crt("malloc")],
        Shim::Realloc => vec![
            crt("realloc"),
            crt("malloc"),
            alloc_aligned,
            crt("memcpy"),
            free_fn,
        ],
        Shim::FillOnce => vec![yield_fn],
        Shim::ThreadCell => tls
            .iter()
            .copied()
            .chain([crt("memcpy"), crt("free"), alloc_aligned, crt("memset")])
            .collect(),
        Shim::CatchUnwind => tls.iter().copied().chain(jmp.map(|j| j.0)).collect(),
        Shim::StartPanic => tls
            .iter()
            .copied()
            .chain(jmp.map(|j| j.1))
            .chain([crt("abort")])
            .collect(),
        Shim::DropPush | Shim::DropPop => tls.to_vec(),
        _ => Vec::new(),
    }
}

pub(super) fn deps(shim: Shim) -> Vec<&'static str> {
    collect_deps(
        shim,
        PTHREAD_DEPS,
        ("posix_memalign", "sched_yield", "free"),
        None,
        |s| s,
    )
}

pub(super) fn win_setjmp_name(arch: Arch) -> &'static str {
    if arch == Arch::Arm64 {
        "_setjmpex"
    } else {
        "_setjmp"
    }
}

pub(super) fn windows_crt_dll(name: &str) -> &'static str {
    match name {
        "atexit" | "_setjmp" | "_setjmpex" | "longjmp" => MSVCRT_DLL,
        _ => UCRTBASE_DLL,
    }
}

pub(super) fn windows_deps(arch: Arch, shim: Shim) -> Vec<(&'static str, &'static str)> {
    let crt = |s: &'static str| (s, windows_crt_dll(s));
    let setjmp = win_setjmp_name(arch);
    match shim {
        Shim::Alloc => vec![crt("_aligned_malloc")],
        Shim::AllocZeroed => vec![crt("_aligned_malloc"), crt("memset")],
        Shim::Dealloc => vec![crt("_aligned_free")],
        Shim::Realloc => vec![crt("_aligned_realloc")],
        Shim::FrameAlloc => vec![crt("malloc")],
        Shim::FrameFree => vec![crt("free")],
        _ => collect_deps(
            shim,
            WIN_TLS_DEPS,
            (
                crt("_aligned_malloc"),
                ("SwitchToThread", KERNEL32_DLL),
                crt("free"),
            ),
            Some((crt(setjmp), crt("longjmp"))),
            crt,
        ),
    }
}

pub(super) fn arm64_adrp(pc_va: u64, target_va: u64, reg: u32) -> u32 {
    let page_delta = ((target_va & !0xFFF) as i64 - (pc_va & !0xFFF) as i64) >> 12;
    0x9000_0000
        | (((page_delta as u32) & 3) << 29)
        | ((((page_delta as u32) >> 2) & 0x7_FFFF) << 5)
        | reg
}

pub(super) fn emit_stub(arch: Arch, stub_va: u64, got_va: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.bytes(&[0xFF, 0x25]);
        out.u32(((got_va as i64 - (stub_va as i64 + 6)) as i32) as u32);
    } else if arch == Arch::Arm64 {
        let imm12 = ((got_va & 0xFFF) >> 3) as u32;
        out.words(&[
            arm64_adrp(stub_va, got_va, 16),
            0xF940_0210 | (imm12 << 10),
            0xD61F_0200,
        ]);
    }
}

pub(super) fn emit_arm64_adrp_add(pc_va: u64, target_va: u64, reg: u32, out: &mut Buf) {
    out.words(&[
        arm64_adrp(pc_va, target_va, reg),
        0x9100_0000 | (((target_va & 0xFFF) as u32) << 10) | (reg << 5) | reg,
    ]);
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

pub(super) fn patch_arm64_pie(
    out: &mut [u8],
    pc_va: u64,
    load_got: bool,
    target_va: u64,
) -> Result<()> {
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

pub(super) fn patch_arm64_adrp_add(
    out: &mut [u8],
    pc_va: u64,
    load_got: bool,
    target_va: u64,
    is_word: bool,
) -> Result<()> {
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

pub(super) fn emit_main_trampoline(
    arch: Arch,
    tramp_va: u64,
    lang_start_va: u64,
    user_main_va: u64,
    out: &mut Buf,
) {
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

pub(super) fn emit_elf_start(
    arch: Arch,
    start_va: u64,
    tramp_va: u64,
    libc_start_got: u64,
    out: &mut Buf,
) {
    if arch == Arch::X64 {
        out.bytes(&[
            0x31, 0xED, 0x49, 0x89, 0xD1, 0x5E, 0x48, 0x89, 0xE2, 0x48, 0x83, 0xE4, 0xF0, 0x50,
            0x54, 0x45, 0x31, 0xC0, 0x31, 0xC9, 0x48, 0x8D, 0x3D,
        ]);
        out.u32(((tramp_va as i64 - (start_va as i64 + 27)) as i32) as u32);
        out.bytes(&[0xFF, 0x15]);
        out.u32(((libc_start_got as i64 - (start_va as i64 + 33)) as i32) as u32);
        out.u8(0xF4);
    } else if arch == Arch::Arm64 {
        out.words(&[
            0xD280_001D,
            0xD280_001E,
            0xAA00_03E5,
            0xF940_03E1,
            0x9100_23E2,
            0xD280_0003,
            0xD280_0004,
            0x9100_03E6,
        ]);
        emit_arm64_adrp_add(start_va + 32, tramp_va, 0, out);
        let imm12 = ((libc_start_got & 0xFFF) >> 3) as u32;
        out.words(&[
            arm64_adrp(start_va + 40, libc_start_got, 16),
            0xF940_0210 | (imm12 << 10),
            0xD63F_0200,
            0xD420_0020,
        ]);
    }
}

pub(super) fn emit_rel_branch(arch: Arch, link: bool, from_va: u64, to_va: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.u8(if link { 0xE8 } else { 0xE9 });
        out.u32(((to_va as i64 - (from_va as i64 + 5)) as i32) as u32);
    } else if arch == Arch::Arm64 {
        out.u32(
            (if link { 0x9400_0000 } else { 0x1400_0000 })
                | (((to_va as i64 - from_va as i64) >> 2) as u32 & 0x03FF_FFFF),
        );
    }
}

fn x64_fwd(op: u8, out: &mut Buf) -> usize {
    out.bytes(&[op, 0]);
    out.pos()
}

pub(super) fn x64_patch8(after: usize, out: &mut Buf) {
    out.0[after - 1] = ((out.pos() as isize - after as isize) as i8) as u8;
}

pub(super) fn x64_fwd32(prefix: &[u8], out: &mut Buf) -> usize {
    out.bytes(prefix);
    out.u32(0);
    out.pos()
}

pub(super) fn x64_patch32(after: usize, out: &mut Buf) {
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
    let patched = if (word >> 26) == 5 {
        (word & !0x03FF_FFFF) | (delta & 0x03FF_FFFF)
    } else {
        (word & !(0x7_FFFF << 5)) | ((delta & 0x7_FFFF) << 5)
    };
    out.patch_u32(at, patched);
}

pub(super) fn arm64_here(labels: &[usize], out: &mut Buf) {
    let target = out.pos();
    labels.iter().for_each(|&at| arm64_patch(at, target, out));
}

fn x64_here(labels: &[usize], out: &mut Buf) {
    labels.iter().for_each(|&after| x64_patch8(after, out));
}

pub(super) fn x64_jcc(bytes: &[u8], op: u8, out: &mut Buf) -> usize {
    out.bytes(bytes);
    x64_fwd(op, out)
}

pub(super) fn arm64_jcc(words: &[u32], word: u32, out: &mut Buf) -> usize {
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
    pub(super) fn at(&self, out: &Buf) -> u64 {
        self.va + (out.pos() - self.base) as u64
    }

    fn branch(&self, link: bool, name: &str, out: &mut Buf) -> Result<()> {
        emit_rel_branch(self.arch, link, self.at(out), (self.stub_va)(name)?, out);
        Ok(())
    }

    pub(super) fn call(&self, name: &str, out: &mut Buf) -> Result<()> {
        self.branch(true, name, out)
    }

    pub(super) fn tail(&self, name: &str, out: &mut Buf) -> Result<()> {
        self.branch(false, name, out)
    }

    pub(super) fn bytes_call(&self, b: &[u8], name: &str, out: &mut Buf) -> Result<()> {
        out.bytes(b);
        self.call(name, out)
    }

    fn bytes_tail(&self, b: &[u8], name: &str, out: &mut Buf) -> Result<()> {
        out.bytes(b);
        self.tail(name, out)
    }

    pub(super) fn words_call(&self, w: &[u32], name: &str, out: &mut Buf) -> Result<()> {
        out.words(w);
        self.call(name, out)
    }

    fn words_tail(&self, w: &[u32], name: &str, out: &mut Buf) -> Result<()> {
        out.words(w);
        self.tail(name, out)
    }

    fn pick(&self, w: &'static str, u: &'static str) -> &'static str {
        if self.win { w } else { u }
    }
}

pub(super) fn emit_shims(
    arch: Arch,
    os: Os,
    plan: &Plan<'_>,
    base_va: u64,
    shims_off: usize,
    addrs: SectionAddrs,
    stub_va: &dyn Fn(&str) -> Result<u64>,
    mut offs: Option<&mut Vec<usize>>,
    out: &mut Buf,
) -> Result<()> {
    for &shim in &plan.shims {
        out.align(16);
        let at = shims_off + out.pos();
        if let Some(list) = offs.as_deref_mut() {
            list.push(at);
        }
        let emit = if arch == Arch::X64 {
            x64::emit_shim_x64
        } else {
            arm64::emit_shim_arm64
        };
        emit(os, shim, base_va + at as u64, addrs, stub_va, out)?;
    }
    Ok(())
}

pub(super) fn plan_shims(
    arch: Arch,
    os: Os,
    plan: &Plan<'_>,
    base_va: u64,
    shims_off: usize,
    dummy: SectionAddrs,
) -> Result<(Vec<usize>, usize)> {
    let mut shim_offs = Vec::with_capacity(plan.shims.len());
    let mut probe = Buf::new();
    let zero_stub = |_: &str| Ok(base_va);
    emit_shims(
        arch,
        os,
        plan,
        base_va,
        shims_off,
        dummy,
        &zero_stub,
        Some(&mut shim_offs),
        &mut probe,
    )?;
    Ok((shim_offs, (shims_off + probe.pos()).next_multiple_of(16)))
}
