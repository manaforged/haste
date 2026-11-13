use anyhow::{Result, bail};
use haste_codegen::{Arch, Os};

use super::wide::WideShim;
use super::{Buf, Plan, SectionAddrs};
use crate::host::cpu::HIDDEN;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
    Wide(WideShim),
}

pub(super) fn deps(shim: Shim) -> &'static [&'static str] {
    match shim {
        Shim::Alloc => &["malloc", "posix_memalign"],
        Shim::AllocZeroed => &["calloc", "posix_memalign", "memset"],
        Shim::Dealloc => &["free"],
        Shim::Realloc => &["realloc", "malloc", "posix_memalign", "memcpy", "free"],
        Shim::FillOnce => &["sched_yield"],
        Shim::ThreadCell => &["pthread_key_create", "pthread_getspecific", "pthread_setspecific", "calloc", "memcpy", "free", "posix_memalign", "memset", "sched_yield"],
        _ => &[],
    }
}

fn arm64_adrp(pc_va: u64, target_va: u64, reg: u32) -> u32 {
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

pub(super) fn patch_arm64_pie(out: &mut [u8], pc_va: u64, load_got: bool, target_va: u64) -> Result<()> {
    let word0 = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    if (word0 & 0xFFE0_0000) != 0xD280_0000 {
        bail!("expected movz instruction at Arm64Movw64 relocation, found {word0:#010x}");
    }
    let rd = word0 & 0x1F;
    let w0 = arm64_adrp(pc_va, target_va, rd);
    let w1 = if load_got {
        0xF940_0000 | ((((target_va & 0xFFF) >> 3) as u32) << 10) | (rd << 5) | rd
    } else {
        0x9100_0000 | (((target_va & 0xFFF) as u32) << 10) | (rd << 5) | rd
    };
    for (i, w) in [w0, w1, 0xD503_201F, 0xD503_201F].into_iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
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

fn emit_rel_branch(arch: Arch, link: bool, from_va: u64, to_va: u64, out: &mut Buf) {
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

fn arm64_fwd(word: u32, out: &mut Buf) -> usize {
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

fn emit_x64_thread_cell(va: u64, key_va: u64, stub_va: &dyn Fn(&str) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let base = out.pos();
    let at = |out: &Buf| va + (out.pos() - base) as u64;
    out.bytes(&[0x53, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57, 0x48, 0x83, 0xEC, 0x10]);
    out.bytes(&[0x48, 0x89, 0xFB, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x48, 0x85, 0xF6, 0x48, 0x0F, 0x44, 0xF0, 0x49, 0x89, 0xF4, 0x4C, 0x8D, 0x2D]);
    out.u32(((key_va as i64 - (at(out) as i64 + 4)) as i32) as u32);
    out.bytes(&[0x41, 0x8B, 0x45, 0x00, 0x83, 0xF8, 0x02]);
    let ready_fast = x64_fwd(0x74, out);
    let init_loop = out.pos();
    out.bytes(&[0x31, 0xC0, 0xBA, 0x01, 0x00, 0x00, 0x00, 0xF0, 0x41, 0x0F, 0xB1, 0x55, 0x00]);
    let do_create = x64_fwd(0x74, out);
    out.bytes(&[0x83, 0xF8, 0x02]);
    let ready_wait = x64_fwd(0x74, out);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("sched_yield")?, out);
    out.bytes(&[0xEB, ((init_loop as isize - (out.pos() as isize + 2)) as i8) as u8]);
    x64_patch8(do_create, out);
    out.bytes(&[0x49, 0x8D, 0x7D, 0x08, 0x31, 0xF6]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("pthread_key_create")?, out);
    out.bytes(&[0x41, 0xC7, 0x45, 0x00, 0x02, 0x00, 0x00, 0x00]);
    x64_patch8(ready_fast, out);
    x64_patch8(ready_wait, out);
    out.bytes(&[0x49, 0x8B, 0x7D, 0x08]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("pthread_getspecific")?, out);
    out.bytes(&[0x49, 0x89, 0xC6, 0x4D, 0x85, 0xF6]);
    let grow_null = x64_fwd(0x74, out);
    out.bytes(&[0x49, 0x3B, 0x1E]);
    let grow_cap = x64_fwd(0x73, out);
    out.bytes(&[0x49, 0x8B, 0x44, 0xDE, 0x08, 0x48, 0x85, 0xC0]);
    let hit = x64_fwd32(&[0x0F, 0x85], out);
    let cell_skip = x64_fwd32(&[0xE9], out);
    x64_patch8(grow_null, out);
    x64_patch8(grow_cap, out);
    out.bytes(&[0x31, 0xC0, 0x4D, 0x85, 0xF6, 0x74, 0x03, 0x49, 0x8B, 0x06, 0x48, 0x01, 0xC0, 0x4C, 0x8D, 0x7B, 0x08, 0x4C, 0x39, 0xF8, 0x4C, 0x0F, 0x47, 0xF8, 0x49, 0x8D, 0x7F, 0x01, 0xBE, 0x08, 0x00, 0x00, 0x00]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("calloc")?, out);
    out.bytes(&[0x48, 0x85, 0xC0]);
    let fail_calloc = x64_fwd(0x74, out);
    out.bytes(&[0x4C, 0x89, 0x38, 0x49, 0x89, 0xC7, 0x4D, 0x85, 0xF6]);
    let no_copy = x64_fwd(0x74, out);
    out.bytes(&[0x49, 0x8D, 0x7F, 0x08, 0x49, 0x8D, 0x76, 0x08, 0x49, 0x8B, 0x16, 0x48, 0xC1, 0xE2, 0x03]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("memcpy")?, out);
    out.bytes(&[0x4C, 0x89, 0xF7]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("free")?, out);
    x64_patch8(no_copy, out);
    out.bytes(&[0x4D, 0x89, 0xFE, 0x49, 0x8B, 0x7D, 0x08, 0x4C, 0x89, 0xF6]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("pthread_setspecific")?, out);
    out.bytes(&[0x85, 0xC0]);
    let fail_set = x64_fwd(0x75, out);
    x64_patch32(cell_skip, out);
    out.bytes(&[0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x48, 0x89, 0xE7, 0xBE, 0x40, 0x00, 0x00, 0x00, 0x4C, 0x89, 0xE2]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("posix_memalign")?, out);
    out.bytes(&[0x85, 0xC0]);
    let fail_align = x64_fwd(0x75, out);
    out.bytes(&[0x48, 0x8B, 0x3C, 0x24, 0x31, 0xF6, 0x4C, 0x89, 0xE2]);
    emit_rel_branch(Arch::X64, true, at(out), stub_va("memset")?, out);
    out.bytes(&[0x49, 0x89, 0x44, 0xDE, 0x08]);
    let done_ok = x64_fwd(0xEB, out);
    x64_patch8(fail_calloc, out);
    x64_patch8(fail_set, out);
    x64_patch8(fail_align, out);
    out.bytes(&[0x31, 0xC0]);
    x64_patch32(hit, out);
    x64_patch8(done_ok, out);
    out.bytes(&[0x48, 0x83, 0xC4, 0x10, 0x41, 0x5F, 0x41, 0x5E, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0xC3]);
    Ok(())
}

fn emit_arm64_thread_cell(va: u64, key_va: u64, stub_va: &dyn Fn(&str) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let base = out.pos();
    let at = |out: &Buf| va + (out.pos() - base) as u64;
    out.words(&[0xD101_03FF, 0xA900_7BFD, 0xA901_53F3, 0xA902_5BF5, 0xF900_1BF7, 0xAA00_03F3, 0xF100_003F, 0xD280_0029, 0x9A89_1034]);
    emit_arm64_adrp_add(at(out), key_va, 21, out);
    out.words(&[0x88DF_FEA8, 0x7100_091F]);
    let ready_fast = arm64_fwd(0x5400_0000, out);
    out.u32(0x5280_002A);
    let init_loop = out.pos();
    out.u32(0x885F_FEA8);
    let already_init = arm64_fwd(0x3500_0008, out);
    out.u32(0x8809_FEAA);
    let retry_cas = arm64_fwd(0x3500_0009, out);
    arm64_patch(retry_cas, init_loop, out);
    out.words(&[0x9100_22A0, 0xAA1F_03E1]);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("pthread_key_create")?, out);
    out.words(&[0x5280_0048, 0x889F_FEA8]);
    let ready_create = arm64_fwd(0x1400_0000, out);
    arm64_patch(already_init, out.pos(), out);
    out.words(&[0x7100_091F]);
    let ready_wait = arm64_fwd(0x5400_0000, out);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("sched_yield")?, out);
    let back = arm64_fwd(0x1400_0000, out);
    arm64_patch(back, init_loop, out);
    arm64_patch(ready_fast, out.pos(), out);
    arm64_patch(ready_create, out.pos(), out);
    arm64_patch(ready_wait, out.pos(), out);
    out.u32(0xF940_06A0);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("pthread_getspecific")?, out);
    out.u32(0xAA00_03F6);
    let grow_null = arm64_fwd(0xB400_0016, out);
    out.words(&[0xF940_02C8, 0xEB08_027F]);
    let grow_cap = arm64_fwd(0x5400_0002, out);
    out.words(&[0x8B13_0EC8, 0xF940_0500]);
    let hit = arm64_fwd(0xB500_0000, out);
    let cell_skip = arm64_fwd(0x1400_0000, out);
    arm64_patch(grow_null, out.pos(), out);
    arm64_patch(grow_cap, out.pos(), out);
    out.words(&[0xD280_0008, 0xB400_0056, 0xF940_02C8, 0x8B08_0108, 0x9100_2277, 0xEB17_011F, 0x9A97_8117, 0x9100_06E0, 0xD280_0101]);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("calloc")?, out);
    let fail_calloc = arm64_fwd(0xB400_0000, out);
    out.words(&[0xF900_0017, 0xAA00_03F7]);
    let no_copy = arm64_fwd(0xB400_0016, out);
    out.words(&[0x9100_22E0, 0x9100_22C1, 0xF940_02C2, 0xD37D_F042]);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("memcpy")?, out);
    out.u32(0xAA16_03E0);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("free")?, out);
    arm64_patch(no_copy, out.pos(), out);
    out.words(&[0xAA17_03F6, 0xF940_06A0, 0xAA16_03E1]);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("pthread_setspecific")?, out);
    let fail_set = arm64_fwd(0x3500_0000, out);
    arm64_patch(cell_skip, out.pos(), out);
    out.words(&[0xF900_1FFF, 0x9100_E3E0, 0xD280_0801, 0xAA14_03E2]);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("posix_memalign")?, out);
    let fail_align = arm64_fwd(0x3500_0000, out);
    out.words(&[0xF940_1FE0, 0x5280_0001, 0xAA14_03E2]);
    emit_rel_branch(Arch::Arm64, true, at(out), stub_va("memset")?, out);
    out.words(&[0x8B13_0EC8, 0xF900_0500, 0x1400_0002]);
    arm64_patch(fail_calloc, out.pos(), out);
    arm64_patch(fail_set, out.pos(), out);
    arm64_patch(fail_align, out.pos(), out);
    out.u32(0xD280_0000);
    arm64_patch(hit, out.pos(), out);
    out.words(&[0xF940_1BF7, 0xA942_5BF5, 0xA941_53F3, 0xA940_7BFD, 0x9101_03FF, 0xD65F_03C0]);
    Ok(())
}

fn emit_shim_x64(os: Os, shim: Shim, va: u64, addrs: SectionAddrs, stub_va: &dyn Fn(&str) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let base = out.pos();
    let at = |out: &Buf| va + (out.pos() - base) as u64;
    match shim {
        Shim::Noop | Shim::Barrier => out.u8(0xC3),
        Shim::Syscall if os == Os::Linux => out.bytes(&[
            0x48, 0x8B, 0x07, 0x48, 0x8B, 0x77, 0x10, 0x48, 0x8B, 0x57, 0x18, 0x4C, 0x8B, 0x57, 0x20, 0x4C, 0x8B, 0x47, 0x28, 0x4C, 0x8B, 0x4F, 0x30, 0x48, 0x8B, 0x7F, 0x08, 0x0F, 0x05, 0xC3,
        ]),
        Shim::Syscall => out.bytes(&[0x48, 0xC7, 0xC0, 0xDA, 0xFF, 0xFF, 0xFF, 0xC3]),
        Shim::Cpuid => {
            out.bytes(&[0x53, 0x49, 0x89, 0xF8, 0x41, 0x89, 0xF1, 0x41, 0x89, 0xD2, 0x89, 0xF0, 0x89, 0xD1, 0x0F, 0xA2]);
            out.bytes(&[0x41, 0x83, 0xF9, 0x01, 0x75, 0x06, 0x81, 0xE1]);
            out.u32(!HIDDEN.leaf1_ecx);
            out.bytes(&[0x41, 0x83, 0xF9, 0x07, 0x75, 0x2A, 0x45, 0x85, 0xD2, 0x75, 0x14, 0x81, 0xE3]);
            out.u32(!HIDDEN.leaf7_ebx);
            out.bytes(&[0x81, 0xE1]);
            out.u32(!HIDDEN.leaf7_ecx);
            out.bytes(&[0x81, 0xE2]);
            out.u32(!HIDDEN.leaf7_edx);
            out.bytes(&[0xEB, 0x11, 0x41, 0x83, 0xFA, 0x01, 0x75, 0x0B, 0x25]);
            out.u32(!HIDDEN.leaf7_1_eax);
            out.bytes(&[0x81, 0xE2]);
            out.u32(!HIDDEN.leaf7_1_edx);
            out.bytes(&[0x41, 0x81, 0xF9, 0x01, 0x00, 0x00, 0x80, 0x75, 0x06, 0x81, 0xE1]);
            out.u32(!HIDDEN.extended_ecx);
            out.bytes(&[0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08, 0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]);
        }
        Shim::ExtendedState => {
            out.bytes(&[
                0x53, 0xB8, 0x01, 0x00, 0x00, 0x00, 0x0F, 0xA2, 0xF7, 0xC1, 0x00, 0x00, 0x00, 0x08, 0x75, 0x04, 0x31, 0xC0, 0x5B, 0xC3, 0x89, 0xF9, 0x0F, 0x01, 0xD0, 0x48, 0xC1, 0xE2, 0x20, 0x48,
                0x09, 0xD0, 0x85, 0xFF, 0x75, 0x02, 0x24, !HIDDEN.xcr0 as u8, 0x5B, 0xC3,
            ]);
        }
        Shim::Dealloc => emit_rel_branch(Arch::X64, false, va, stub_va("free")?, out),
        Shim::Alloc => {
            out.bytes(&[0xB8, 0x01, 0x00, 0x00, 0x00, 0x48, 0x85, 0xFF, 0x48, 0x0F, 0x44, 0xF8, 0x48, 0x83, 0xFE, 0x10, 0x77, 0x05]);
            emit_rel_branch(Arch::X64, false, at(out), stub_va("malloc")?, out);
            out.bytes(&[0x48, 0x83, 0xEC, 0x18, 0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x48, 0x89, 0xFA, 0x48, 0x8D, 0x3C, 0x24]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("posix_memalign")?, out);
            out.bytes(&[0x85, 0xC0, 0x48, 0x8B, 0x04, 0x24, 0x31, 0xD2, 0x48, 0x0F, 0x45, 0xC2, 0x48, 0x83, 0xC4, 0x18, 0xC3]);
        }
        Shim::AllocZeroed => {
            out.bytes(&[0xB8, 0x01, 0x00, 0x00, 0x00, 0x48, 0x85, 0xFF, 0x48, 0x0F, 0x44, 0xF8, 0x48, 0x83, 0xFE, 0x10, 0x77, 0x0D, 0x48, 0x89, 0xFE, 0xBF, 0x01, 0x00, 0x00, 0x00]);
            emit_rel_branch(Arch::X64, false, at(out), stub_va("calloc")?, out);
            out.bytes(&[0x53, 0x48, 0x83, 0xEC, 0x10, 0x48, 0x89, 0xFB, 0x48, 0x89, 0xFA, 0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x48, 0x8D, 0x3C, 0x24]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("posix_memalign")?, out);
            out.bytes(&[0x85, 0xC0, 0x75, 0x10, 0x48, 0x8B, 0x3C, 0x24, 0x31, 0xF6, 0x48, 0x89, 0xDA]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("memset")?, out);
            out.bytes(&[0xEB, 0x02, 0x31, 0xC0, 0x48, 0x83, 0xC4, 0x10, 0x5B, 0xC3]);
        }
        Shim::Realloc => {
            out.bytes(&[0xB8, 0x01, 0x00, 0x00, 0x00, 0x48, 0x85, 0xC9, 0x48, 0x0F, 0x44, 0xC8, 0x48, 0x83, 0xFA, 0x10, 0x77, 0x08, 0x48, 0x89, 0xCE]);
            emit_rel_branch(Arch::X64, false, at(out), stub_va("realloc")?, out);
            out.bytes(&[0x53, 0x41, 0x54, 0x41, 0x55, 0x48, 0x83, 0xEC, 0x10, 0x48, 0x89, 0xFB, 0x49, 0x89, 0xF4, 0x49, 0x89, 0xCD, 0x48, 0xC7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x48, 0x8D, 0x3C, 0x24, 0x48, 0x89, 0xD6, 0x48, 0x89, 0xCA]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("posix_memalign")?, out);
            out.bytes(&[0x85, 0xC0, 0x75, 0x26, 0x48, 0x8B, 0x3C, 0x24, 0x48, 0x89, 0xDE, 0x4C, 0x89, 0xE2, 0x4D, 0x39, 0xEC, 0x49, 0x0F, 0x47, 0xD5]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("memcpy")?, out);
            out.bytes(&[0x48, 0x89, 0xDF, 0x48, 0x89, 0xC3]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("free")?, out);
            out.bytes(&[0x48, 0x89, 0xD8, 0xEB, 0x02, 0x31, 0xC0, 0x48, 0x83, 0xC4, 0x10, 0x41, 0x5D, 0x41, 0x5C, 0x5B, 0xC3]);
        }
        Shim::FillOnce => {
            out.bytes(&[0x53, 0x41, 0x54, 0x48, 0x89, 0xFB, 0x49, 0x89, 0xF4, 0x31, 0xC0, 0xB2, 0x02, 0xF0, 0x0F, 0xB0, 0x13, 0x74, 0x0B, 0x3C, 0x01, 0x74, 0x0D]);
            emit_rel_branch(Arch::X64, true, at(out), stub_va("sched_yield")?, out);
            out.bytes(&[0xEB, 0xEB, 0x41, 0xFF, 0xD4, 0xC6, 0x03, 0x01, 0x41, 0x5C, 0x5B, 0xC3]);
        }
        Shim::ThreadCell => emit_x64_thread_cell(va, addrs.thread_key, stub_va, out)?,
        Shim::Wide(wide) => super::wide::emit(Arch::X64, os, wide, out)?,
    }
    Ok(())
}

fn emit_shim_arm64(os: Os, shim: Shim, va: u64, addrs: SectionAddrs, stub_va: &dyn Fn(&str) -> Result<u64>, out: &mut Buf) -> Result<()> {
    let base = out.pos();
    let at = |out: &Buf| va + (out.pos() - base) as u64;
    match shim {
        Shim::Noop => out.u32(0xD65F_03C0),
        Shim::Barrier => out.words(&[0xD503_3F9F, 0xD503_3FDF, 0xD65F_03C0]),
        Shim::Syscall => out.words(&[if os == Os::Darwin { 0x9280_09A0 } else { 0x9280_04A0 }, 0xD65F_03C0]),
        Shim::Cpuid => out.words(&[0xA900_7C1F, 0xD65F_03C0]),
        Shim::ExtendedState => out.words(&[0xD280_0000, 0xD65F_03C0]),
        Shim::Dealloc => emit_rel_branch(Arch::Arm64, false, va, stub_va("free")?, out),
        Shim::Alloc => {
            out.words(&[0xF100_001F, 0xD280_0029, 0x9A89_1000, 0xF100_403F, 0x5400_0048]);
            emit_rel_branch(Arch::Arm64, false, at(out), stub_va("malloc")?, out);
            out.words(&[0xD100_83FF, 0xA900_7BFD, 0xAA00_03E2, 0xF900_0BFF, 0x9100_43E0]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("posix_memalign")?, out);
            out.words(&[0x7100_001F, 0xF940_0BE0, 0x9A9F_0000, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::AllocZeroed => {
            out.words(&[0xF100_001F, 0xD280_0029, 0x9A89_1000, 0xF100_403F, 0x5400_0068, 0xAA00_03E1, 0xD280_0020]);
            emit_rel_branch(Arch::Arm64, false, at(out), stub_va("calloc")?, out);
            out.words(&[0xD100_83FF, 0xA900_7BFD, 0xF900_0BF3, 0xAA00_03F3, 0xAA00_03E2, 0xF900_0FFF, 0x9100_63E0]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("posix_memalign")?, out);
            out.words(&[0x3500_00C0, 0xF940_0FE0, 0x5280_0001, 0xAA13_03E2]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("memset")?, out);
            out.words(&[0x1400_0002, 0xD280_0000, 0xF940_0BF3, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::Realloc => {
            out.words(&[0xF100_007F, 0xD280_0029, 0x9A89_1063, 0xF100_405F, 0x5400_0068, 0xAA03_03E1]);
            emit_rel_branch(Arch::Arm64, false, at(out), stub_va("realloc")?, out);
            out.words(&[0xD100_C3FF, 0xA900_7BFD, 0xA901_53F3, 0xF900_13F5, 0xAA00_03F3, 0xEB03_003F, 0x9A83_3034, 0xAA02_03E1, 0xAA03_03E2, 0xF900_17FF, 0x9100_A3E0]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("posix_memalign")?, out);
            out.words(&[0x3500_0140, 0xF940_17F5, 0xAA15_03E0, 0xAA13_03E1, 0xAA14_03E2]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("memcpy")?, out);
            out.words(&[0xAA13_03E0]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("free")?, out);
            out.words(&[0xAA15_03E0, 0x1400_0002, 0xD280_0000, 0xA941_53F3, 0xF940_13F5, 0xA8C3_7BFD, 0xD65F_03C0]);
        }
        Shim::FillOnce => {
            out.words(&[0xA9BE_7BFD, 0xA901_53F3, 0xAA00_03F3, 0xAA01_03F4, 0x5280_004A]);
            let retry = out.pos();
            out.words(&[0x085F_FE68, 0x3500_00E8, 0x0809_FE6A, 0x35FF_FFA9, 0xD63F_0280, 0x5280_0028, 0x089F_FE68, 0x1400_0005, 0x7100_051F, 0x5400_0060]);
            emit_rel_branch(Arch::Arm64, true, at(out), stub_va("sched_yield")?, out);
            let back = ((retry as isize - out.pos() as isize) >> 2) as u32;
            out.words(&[0x1400_0000 | (back & 0x03FF_FFFF), 0xA941_53F3, 0xA8C2_7BFD, 0xD65F_03C0]);
        }
        Shim::ThreadCell => emit_arm64_thread_cell(va, addrs.thread_key, stub_va, out)?,
        Shim::Wide(wide) => super::wide::emit(Arch::Arm64, os, wide, out)?,
    }
    Ok(())
}

pub(super) fn emit_shims(arch: Arch, os: Os, plan: &Plan<'_>, base_va: u64, shims_off: usize, addrs: SectionAddrs, stub_va: &dyn Fn(&str) -> Result<u64>, mut offs: Option<&mut Vec<usize>>, out: &mut Buf) -> Result<()> {
    for &shim in &plan.shims {
        out.align(16);
        let at = shims_off + out.pos();
        if let Some(list) = offs.as_deref_mut() {
            list.push(at);
        }
        if arch == Arch::X64 {
            emit_shim_x64(os, shim, base_va + at as u64, addrs, stub_va, out)?;
        } else {
            emit_shim_arm64(os, shim, base_va + at as u64, addrs, stub_va, out)?;
        }
    }
    Ok(())
}

pub(super) fn plan_shims(arch: Arch, os: Os, plan: &Plan<'_>, base_va: u64, shims_off: usize) -> Result<(Vec<usize>, usize)> {
    let mut shim_offs = Vec::with_capacity(plan.shims.len());
    let mut probe = Buf::new();
    let zero_stub = |_: &str| Ok(base_va);
    let dummy = SectionAddrs { thread_key: base_va, ..SectionAddrs::default() };
    emit_shims(arch, os, plan, base_va, shims_off, dummy, &zero_stub, Some(&mut shim_offs), &mut probe)?;
    Ok((shim_offs, (shims_off + probe.pos()).next_multiple_of(16)))
}
