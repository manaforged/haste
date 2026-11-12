use anyhow::{Result, bail};
use haste_codegen::{Arch, Os};

use super::shims::{emit_elf_start, emit_main_trampoline, emit_shims, emit_stub, plan_shims};
use super::{Buf, ELF_BASE, Executable, PatchCtx, Plan, SectionAddrs, Symbol, patch_code, resolve_sym};

struct ElfTables {
    dynstr: Vec<u8>,
    needed: Vec<u32>,
    sym_offs: Vec<u32>,
    buckets: Vec<u32>,
    chains: Vec<u32>,
}

fn elf_hash(name: &[u8]) -> u32 {
    name.iter().fold(0u32, |mut h, &byte| {
        h = (h << 4).wrapping_add(u32::from(byte));
        let g = h & 0xF000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h & !g
    })
}

fn build_tables(plan: &Plan<'_>) -> ElfTables {
    let mut dynstr = vec![0u8];
    let mut push_str = |s: &str| {
        let off = dynstr.len() as u32;
        dynstr.extend_from_slice(s.as_bytes());
        dynstr.push(0);
        off
    };
    let mut needed = vec![push_str("libc.so.6")];
    plan.libs.iter().filter(|&&l| l != "libc.so.6").for_each(|l| needed.push(push_str(l)));
    let sym_offs: Vec<u32> = plan.imports.iter().map(|s| push_str(s)).collect();
    let nsyms = 1 + plan.imports.len();
    let nbucket = nsyms.next_power_of_two().max(4);
    let (mut buckets, mut chains) = (vec![0u32; nbucket], vec![0u32; nsyms]);
    for (i, name) in plan.imports.iter().enumerate() {
        let idx = (i + 1) as u32;
        let b = (elf_hash(name.as_bytes()) as usize) % nbucket;
        chains[idx as usize] = buckets[b];
        buckets[b] = idx;
    }
    ElfTables { dynstr, needed, sym_offs, buckets, chains }
}

fn write_phdr(out: &mut Buf, kind: u32, flags: u32, off: usize, vaddr: u64, filesz: usize, memsz: usize, align: usize) {
    out.words(&[kind, flags]);
    out.u64s(&[off as u64, vaddr, vaddr, filesz as u64, memsz as u64, align as u64]);
}

pub(super) fn emit(exe: &Executable, (lang_start, user_main): (u32, u32)) -> Result<Vec<u8>> {
    let arch = exe.target.arch;
    let plan = Plan::collect(exe, &["__libc_start_main"])?;
    let interp: &[u8] = match arch {
        Arch::X64 => b"/lib64/ld-linux-x86-64.so.2\0",
        Arch::Arm64 => b"/lib/ld-linux-aarch64.so.1\0",
        _ => bail!("unsupported ELF arch {arch:?}"),
    };
    let (page, stub_size, start_size, tramp_size) = if arch == Arch::Arm64 { (0x1_0000_usize, 12, 56, 24) } else { (0x1000_usize, 6, 34, 20) };
    let tables = build_tables(&plan);
    let nsyms = 1 + plan.imports.len();

    let interp_off = 64 + 5 * 56;
    let hash_off = (interp_off + interp.len()).next_multiple_of(8);
    let dynsym_off = (hash_off + (2 + tables.buckets.len() + nsyms) * 4).next_multiple_of(8);
    let dynstr_off = dynsym_off + nsyms * 24;
    let rela_off = (dynstr_off + tables.dynstr.len()).next_multiple_of(8);
    let rela_size = plan.imports.len() * 24;
    let rodata_off = (rela_off + rela_size).next_multiple_of(16);
    let dso_off = rodata_off + exe.rodata.len();
    let stubs_off = (dso_off + 8).next_multiple_of(16);
    let start_off = (stubs_off + plan.imports.len() * stub_size).next_multiple_of(16);
    let tramp_off = (start_off + start_size).next_multiple_of(16);
    let shims_off = (tramp_off + tramp_size).next_multiple_of(16);
    let (shim_offs, text_off) = plan_shims(arch, Os::Linux, &plan, ELF_BASE, shims_off)?;
    let rx_end = (text_off + exe.text.len()).next_multiple_of(16);

    let rw_off = rx_end;
    let rw_va = ELF_BASE + (rx_end + page) as u64;
    let got_off = (rw_off + exe.data.len()).next_multiple_of(8);
    let dyn_off = got_off + plan.imports.len() * 8;
    let dyn_size = (tables.needed.len() + 10) * 16;
    let rw_filesz = (dyn_off + dyn_size) - rw_off;
    let bss_in_rw = rw_filesz.next_multiple_of(16);
    let (key_off, bss_total) = plan.bss_layout(exe.bss);

    let bss_va = rw_va + bss_in_rw as u64;
    let addrs = SectionAddrs {
        text: ELF_BASE + text_off as u64,
        rodata: ELF_BASE + rodata_off as u64,
        data: rw_va,
        bss: bss_va,
        dso_handle: ELF_BASE + dso_off as u64,
        thread_key: bss_va + u64::from(key_off),
    };
    let stub_va = |name: &str| Ok(ELF_BASE + (stubs_off + plan.import_idx(name)? * stub_size) as u64);
    let got_va = |idx: usize| rw_va + ((got_off - rw_off) + idx * 8) as u64;
    let resolve = |sym: &Symbol| resolve_sym(sym, addrs, ELF_BASE, &shim_offs, &plan, &stub_va);

    let mut out = Buf::new();
    out.bytes(b"\x7FELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    out.u16(2);
    out.u16(if arch == Arch::X64 { 62 } else { 183 });
    out.u32(1);
    out.u64s(&[ELF_BASE + start_off as u64, 64, 0]);
    out.u32(0);
    [64_u16, 56, 5, 64, 0, 0].into_iter().for_each(|v| out.u16(v));
    write_phdr(&mut out, 6, 4, 64, ELF_BASE + 64, 5 * 56, 5 * 56, 8);
    write_phdr(&mut out, 3, 4, interp_off, ELF_BASE + interp_off as u64, interp.len(), interp.len(), 1);
    write_phdr(&mut out, 1, 5, 0, ELF_BASE, rx_end, rx_end, page);
    write_phdr(&mut out, 1, 6, rw_off, rw_va, rw_filesz, bss_in_rw + bss_total as usize, page);
    write_phdr(&mut out, 2, 6, dyn_off, rw_va + (dyn_off - rw_off) as u64, dyn_size, dyn_size, 8);

    out.bytes(interp);
    out.align(8);
    out.words(&[tables.buckets.len() as u32, nsyms as u32]);
    out.words(&tables.buckets);
    out.words(&tables.chains);
    out.align(8);
    out.bytes(&[0; 24]);
    tables.sym_offs.iter().for_each(|&off| { out.words(&[off, 0x12]); out.u64s(&[0, 0]); });
    out.bytes(&tables.dynstr);
    out.align(8);
    let r_type = if arch == Arch::X64 { 6_u64 } else { 1025_u64 };
    (0..plan.imports.len()).for_each(|i| out.u64s(&[got_va(i), ((i as u64 + 1) << 32) | r_type, 0]));

    out.align(16);
    out.bytes(&exe.rodata);
    out.bytes(&[0; 8]);
    for ptr in exe.pointers.iter().filter(|ptr| ptr.in_rodata) {
        out.patch_u64(rodata_off + ptr.offset as usize, resolve(&ptr.symbol)?);
    }
    out.align(16);
    (0..plan.imports.len()).for_each(|i| emit_stub(arch, ELF_BASE + (stubs_off + i * stub_size) as u64, got_va(i), &mut out));
    out.align(16);
    emit_elf_start(arch, ELF_BASE + start_off as u64, ELF_BASE + tramp_off as u64, got_va(plan.import_idx("__libc_start_main")?), &mut out);
    out.align(16);
    emit_main_trampoline(arch, ELF_BASE + tramp_off as u64, addrs.text + u64::from(lang_start), addrs.text + u64::from(user_main), &mut out);
    out.align(16);
    emit_shims(arch, Os::Linux, &plan, ELF_BASE, 0, addrs, &stub_va, None, &mut out)?;
    out.align(16);
    out.bytes(&exe.text);
    out.align(16);
    let resolve_got = |name: &str| Ok(got_va(plan.import_idx(name)?));
    let ctx = PatchCtx { text_base: text_off, text_va: addrs.text, pie_arm64: false, addrs, base_va: ELF_BASE, shim_offs: &shim_offs, plan: &plan, stub_va: &stub_va, got_va: &resolve_got };
    patch_code(&exe.relocs, &ctx, &mut out.0)?;

    out.bytes(&exe.data);
    for ptr in exe.pointers.iter().filter(|ptr| !ptr.in_rodata) {
        out.patch_u64(rw_off + ptr.offset as usize, resolve(&ptr.symbol)?);
    }
    out.0.resize(dyn_off, 0);
    tables.needed.iter().for_each(|&off| { out.i64(1); out.u64(u64::from(off)); });
    for (tag, val) in [(4_i64, ELF_BASE + hash_off as u64), (5, ELF_BASE + dynstr_off as u64), (6, ELF_BASE + dynsym_off as u64), (7, ELF_BASE + rela_off as u64), (8, rela_size as u64), (9, 24), (10, tables.dynstr.len() as u64), (11, 24), (30, 8), (0, 0)] {
        out.i64(tag);
        out.u64(val);
    }
    Ok(out.0)
}
