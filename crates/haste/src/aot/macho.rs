use anyhow::Result;
use haste_codegen::{Arch, Os};

use super::shims::{emit_main_trampoline, emit_shims, emit_stub, plan_shims};
use super::tls::TlsLayout;
use super::{Buf, Executable, MACH_BASE, MACH_PAGE, PatchCtx, Plan, SectionAddrs, Symbol, patch_code, plan_veneers, resolve_sym};

const LOCAL_IN_TEXT: u32 = 0x010E;
const MH_HAS_TLV_DESCRIPTORS: u32 = 0x0080_0000;
const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;
const S_THREAD_LOCAL_VARIABLES: u32 = 0x13;
const TLV_DESC_SIZE: usize = 24;

struct MachMeta<'a> {
    rebase_off: usize,
    rebase: &'a [u8],
    bind_off: usize,
    bind: &'a [u8],
    symtab_off: usize,
    indirect_off: usize,
    strtab_off: usize,
    strtab_len: usize,
    nimports: usize,
    nlocals: usize,
    tramp_off: usize,
    libs: &'a [String],
}

fn write_meta(out: &mut Buf, m: &MachMeta<'_>) {
    let rebase_sz = if m.rebase.is_empty() { 0 } else { m.rebase.len() as u32 };
    let rebase_off = if m.rebase.is_empty() { 0 } else { m.rebase_off as u32 };
    let bind_sz = if m.bind.is_empty() { 0 } else { m.bind.len() as u32 };
    let bind_off = if m.bind.is_empty() { 0 } else { m.bind_off as u32 };
    out.words(&[0x8000_0022, 48, rebase_off, rebase_sz, bind_off, bind_sz, 0, 0, 0, 0, 0, 0]);
    out.words(&[2, 24, m.symtab_off as u32, (m.nlocals + m.nimports) as u32, m.strtab_off as u32, m.strtab_len as u32]);
    out.words(&[11, 80, 0, m.nlocals as u32, m.nlocals as u32, 0, m.nlocals as u32, m.nimports as u32, 0, 0, 0, 0, 0, 0, m.indirect_off as u32, (m.nimports * 2) as u32, 0, 0, 0, 0]);
    let dylinker = b"/usr/lib/dyld\0";
    out.words(&[14, (12 + dylinker.len()).next_multiple_of(8) as u32, 12]);
    out.bytes(dylinker);
    out.align(8);
    out.words(&[0x32, 24, 1, 0x000E_0000, 0x000E_0000, 0, 0x8000_0028, 24]);
    out.u64s(&[m.tramp_off as u64, 0]);
    for lib in m.libs {
        let size = (24 + lib.len() + 1).next_multiple_of(8);
        out.words(&[12, size as u32, 24, 0, 0x0001_0000, 0x0001_0000]);
        out.bytes(lib.as_bytes());
        out.u8(0);
        out.align(8);
    }
}

fn build_dyld_info(exe: &Executable, plan: &Plan<'_>, data_in_seg: u64, rodata_in_seg: u64, tlv_in_seg: Option<u64>, single_lib: bool) -> (Vec<u8>, Vec<u8>) {
    let mut rebase = Buf::new();
    let mut ptrs: Vec<u64> = exe
        .pointers
        .iter()
        .filter(|p| !matches!(p.symbol, Symbol::Word(_)))
        .map(|p| if p.in_rodata { rodata_in_seg } else { data_in_seg } + u64::from(p.offset))
        .collect();
    ptrs.sort_unstable();
    ptrs.dedup();
    if !ptrs.is_empty() {
        rebase.u8(0x11);
        ptrs.into_iter().for_each(|off| { rebase.u8(0x22); rebase.uleb(off); rebase.u8(0x51); });
        rebase.u8(0);
    }
    let mut bind = Buf::new();
    if !plan.imports.is_empty() || tlv_in_seg.is_some() {
        bind.u8(if single_lib { 0x11 } else { 0x2E });
        bind.u8(0x51);
        for (i, name) in plan.imports.iter().enumerate() {
            bind.u8(0x40);
            bind.bytes(exe.target.symbol_prefix.as_bytes());
            bind.bytes(name.as_bytes());
            bind.u8(0);
            bind.u8(0x72);
            bind.uleb((i * 8) as u64);
            bind.u8(0x90);
        }
        if let Some(off) = tlv_in_seg {
            bind.u8(0x11);
            bind.u8(0x40);
            bind.bytes(exe.target.symbol_prefix.as_bytes());
            bind.bytes(b"_tlv_bootstrap");
            bind.u8(0);
            bind.u8(0x72);
            bind.uleb(off);
            bind.u8(0x90);
        }
        bind.u8(0);
    }
    (rebase.0, bind.0)
}

fn write_seg(out: &mut Buf, name: &str, vmaddr: u64, vmsize: u64, fileoff: u64, filesize: u64, prot: u32, nsects: u32) {
    out.words(&[0x19, 72 + nsects * 80]);
    out.fixed_name(name);
    out.u64s(&[vmaddr, vmsize, fileoff, filesize]);
    out.words(&[prot, prot, nsects, 0]);
}

fn write_sect(out: &mut Buf, sect: &str, seg: &str, addr: u64, size: u64, offset: u32, align: u32, flags: u32, r1: u32, r2: u32) {
    out.fixed_name(sect);
    out.fixed_name(seg);
    out.u64s(&[addr, size]);
    out.words(&[offset, align, 0, 0, flags, r1, r2, 0]);
}

pub(super) fn emit(exe: &Executable, (lang_start, user_main): (u32, u32)) -> Result<Vec<u8>> {
    let arch = exe.target.arch;
    let plan = Plan::collect(exe, &[])?;
    let (stub_size, tramp_size) = if arch == Arch::X64 { (6, 20) } else { (12, 24) };

    let mut libs = vec!["/usr/lib/libSystem.B.dylib".to_owned()];
    plan.libs.iter().filter(|&&l| !l.contains(".so")).for_each(|l| libs.push(format!("@rpath/lib{l}.dylib")));

    let tls = if arch == Arch::Arm64 { TlsLayout::compute(&plan, exe.thread_slots) } else { TlsLayout::default() };
    let has_tlv = tls.total > 0;
    let (nsects_text, nsects_data) = (if plan.imports.is_empty() { 1 } else { 2 }, (if plan.imports.is_empty() { 3 } else { 4 }) + if has_tlv { 2 } else { 0 });
    let dylib_cmds: usize = libs.iter().map(|l| (24 + l.len() + 1).next_multiple_of(8)).sum();
    let sizeofcmds = 72 + (72 + nsects_text * 80) + (72 + nsects_data * 80) + 72 + 48 + 24 + 80 + 32 + 24 + 24 + dylib_cmds;
    let tramp_off = (32 + sizeofcmds + 64).next_multiple_of(16);
    let shims_off = (tramp_off + tramp_size).next_multiple_of(16);
    let probe_addrs = SectionAddrs {
        thread_key: MACH_BASE,
        tlv_desc: if has_tlv { MACH_BASE } else { 0 },
        unwind_tls: tls.unwind_off,
        thread_tls: tls.cells_off,
        frame_tls: tls.frame_off,
        ..SectionAddrs::default()
    };
    let (shim_offs, head_off) = plan_shims(arch, Os::Darwin, &plan, MACH_BASE, shims_off, probe_addrs)?;
    let post_rel = exe.text.len().next_multiple_of(16);
    let veneers = plan_veneers(exe, &plan, MACH_BASE, head_off, &shim_offs, |pre, post| (head_off + pre + post_rel + post).next_multiple_of(16))?;
    let text_off = head_off + veneers.pre_len;
    let stubs_off = (text_off + post_rel + veneers.post_len).next_multiple_of(16);
    let text_seg_size = (stubs_off + plan.imports.len() * stub_size).next_multiple_of(MACH_PAGE as usize);

    let (data_seg_off, data_seg_va) = (text_seg_size, MACH_BASE + text_seg_size as u64);
    let data_off = (data_seg_off + plan.imports.len() * 8).next_multiple_of(16);
    let rodata_off = (data_off + exe.data.len()).next_multiple_of(16);
    let dso_off = rodata_off + exe.rodata.len();
    let tlv_vars_off = (dso_off + 8).next_multiple_of(8);
    let data_file_end = if has_tlv { tlv_vars_off + TLV_DESC_SIZE } else { dso_off + 8 };
    let bss_off = data_file_end.next_multiple_of(16);
    let (key_off, bss_total) = plan.bss_layout(exe.bss);
    let tbss_off = (bss_off + bss_total as usize).next_multiple_of(16);
    let data_seg_filesz = data_file_end.next_multiple_of(MACH_PAGE as usize) - data_seg_off;
    let data_seg_vmsz = (tbss_off + if has_tlv { tls.total as usize } else { 0 }).next_multiple_of(MACH_PAGE as usize) - data_seg_off;

    let bss_va = MACH_BASE + bss_off as u64;
    let addrs = SectionAddrs {
        text: MACH_BASE + text_off as u64,
        rodata: MACH_BASE + rodata_off as u64,
        data: MACH_BASE + data_off as u64,
        bss: bss_va,
        dso_handle: MACH_BASE + dso_off as u64,
        thread_key: bss_va + u64::from(key_off),
        tlv_desc: if has_tlv { MACH_BASE + tlv_vars_off as u64 } else { 0 },
        unwind_tls: tls.unwind_off,
        thread_tls: tls.cells_off,
        frame_tls: tls.frame_off,
    };
    let stub_va = |name: &str| Ok(MACH_BASE + (stubs_off + plan.import_idx(name)? * stub_size) as u64);
    let got_va = |idx: usize| MACH_BASE + (data_seg_off + idx * 8) as u64;
    let resolve = |sym: &Symbol| resolve_sym(sym, addrs, MACH_BASE, &shim_offs, &plan, &stub_va);

    let tlv_in_seg = has_tlv.then_some((tlv_vars_off - data_seg_off) as u64);
    let (rebase, bind) = build_dyld_info(exe, &plan, (data_off - data_seg_off) as u64, (rodata_off - data_seg_off) as u64, tlv_in_seg, libs.len() == 1);
    let mut strtab = vec![0u8];
    let sym_offs: Vec<u32> = plan.imports.iter().map(|n| { let o = strtab.len() as u32; strtab.extend_from_slice(exe.target.symbol_prefix.as_bytes()); strtab.extend_from_slice(n.as_bytes()); strtab.push(0); o }).collect();
    let fn_strx = exe.symbol_names(exe.target.symbol_prefix, &mut strtab);
    let nlocals = fn_strx.len();
    let linkedit_off = data_seg_off + data_seg_filesz;
    let linkedit_va = data_seg_va + data_seg_vmsz as u64;
    let bind_off = (linkedit_off + rebase.len()).next_multiple_of(8);
    let symtab_off = (bind_off + bind.len()).next_multiple_of(8);
    let indirect_off = symtab_off + (nlocals + plan.imports.len()) * 16;
    let strtab_off = indirect_off + plan.imports.len() * 8;
    let linkedit_filesz = (strtab_off + strtab.len()) - linkedit_off;

    let mut out = Buf::new();
    let mut flags = if libs.len() == 1 { 0x0020_0085 } else { 0x0020_0005 };
    if has_tlv {
        flags |= MH_HAS_TLV_DESCRIPTORS;
    }
    out.words(&[0xFEED_FACF, if arch == Arch::Arm64 { 0x0100_000C } else { 0x0100_0007 }, if arch == Arch::Arm64 { 0 } else { 3 }, 2, 10 + libs.len() as u32, sizeofcmds as u32, flags, 0]);
    write_seg(&mut out, "__PAGEZERO", 0, MACH_BASE, 0, 0, 0, 0);
    write_seg(&mut out, "__TEXT", MACH_BASE, text_seg_size as u64, 0, text_seg_size as u64, 5, nsects_text as u32);
    write_sect(&mut out, "__text", "__TEXT", MACH_BASE + tramp_off as u64, (stubs_off - tramp_off) as u64, tramp_off as u32, 4, 0x8000_0400, 0, 0);
    if !plan.imports.is_empty() {
        write_sect(&mut out, "__stubs", "__TEXT", MACH_BASE + stubs_off as u64, (plan.imports.len() * stub_size) as u64, stubs_off as u32, 4, 0x8000_0408, plan.imports.len() as u32, stub_size as u32);
    }
    write_seg(&mut out, "__DATA", data_seg_va, data_seg_vmsz as u64, data_seg_off as u64, data_seg_filesz as u64, 3, nsects_data as u32);
    if !plan.imports.is_empty() {
        write_sect(&mut out, "__got", "__DATA", data_seg_va, (plan.imports.len() * 8) as u64, data_seg_off as u32, 3, 6, 0, 0);
    }
    write_sect(&mut out, "__data", "__DATA", addrs.data, exe.data.len() as u64, data_off as u32, 4, 0, 0, 0);
    write_sect(&mut out, "__const", "__DATA", addrs.rodata, (exe.rodata.len() + 8) as u64, rodata_off as u32, 4, 0, 0, 0);
    if has_tlv {
        write_sect(&mut out, "__thread_vars", "__DATA", addrs.tlv_desc, TLV_DESC_SIZE as u64, tlv_vars_off as u32, 3, S_THREAD_LOCAL_VARIABLES, 0, 0);
    }
    write_sect(&mut out, "__bss", "__DATA", addrs.bss, u64::from(bss_total), 0, 4, 1, 0, 0);
    if has_tlv {
        write_sect(&mut out, "__thread_bss", "__DATA", MACH_BASE + tbss_off as u64, tls.total, 0, 4, S_THREAD_LOCAL_ZEROFILL, 0, 0);
    }
    write_seg(&mut out, "__LINKEDIT", linkedit_va, linkedit_filesz.next_multiple_of(MACH_PAGE as usize) as u64, linkedit_off as u64, linkedit_filesz as u64, 1, 0);
    write_meta(&mut out, &MachMeta { rebase_off: linkedit_off, rebase: &rebase, bind_off, bind: &bind, symtab_off, indirect_off, strtab_off, strtab_len: strtab.len(), nimports: plan.imports.len(), nlocals, tramp_off, libs: &libs });

    out.0.resize(tramp_off, 0);
    emit_main_trampoline(arch, MACH_BASE + tramp_off as u64, addrs.text + u64::from(lang_start), addrs.text + u64::from(user_main), &mut out);
    out.align(16);
    emit_shims(arch, Os::Darwin, &plan, MACH_BASE, 0, addrs, &stub_va, None, &mut out)?;
    out.align(16);
    veneers.emit_pre(&resolve, &mut out)?;
    out.bytes(&exe.text);
    let resolve_got = |name: &str| Ok(got_va(plan.import_idx(name)?));
    let ctx = PatchCtx { text_base: text_off, text_va: addrs.text, pie_arm64: arch == Arch::Arm64, addrs, base_va: MACH_BASE, shim_offs: &shim_offs, plan: &plan, stub_va: &stub_va, got_va: &resolve_got, veneers: &veneers };
    patch_code(&exe.relocs, &ctx, &mut out.0)?;

    out.align(16);
    veneers.emit_post(&resolve, &mut out)?;
    (0..plan.imports.len()).for_each(|i| emit_stub(arch, MACH_BASE + (stubs_off + i * stub_size) as u64, got_va(i), &mut out));
    out.0.resize(data_off, 0);
    out.bytes(&exe.data);
    out.0.resize(rodata_off, 0);
    out.bytes(&exe.rodata);
    out.bytes(&[0; 8]);
    for ptr in &exe.pointers {
        let off = if ptr.in_rodata { rodata_off } else { data_off } + ptr.offset as usize;
        out.patch_u64(off, resolve(&ptr.symbol)?);
    }
    out.0.resize(linkedit_off, 0);
    out.bytes(&rebase);
    out.align(8);
    out.bytes(&bind);
    out.align(8);
    for (&strx, (off, _)) in fn_strx.iter().zip(&exe.symbols) {
        out.words(&[strx, LOCAL_IN_TEXT]);
        out.u64(addrs.text + u64::from(*off));
    }
    sym_offs.iter().for_each(|&off| { out.words(&[off, 1]); out.u64(0); });
    for _ in 0..2 { (0..plan.imports.len()).for_each(|idx| out.u32((nlocals + idx) as u32)); }
    out.bytes(&strtab);
    Ok(out.0)
}
