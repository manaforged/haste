use std::borrow::Cow;

use anyhow::{Result, bail};
use haste_codegen::{Arch, Os};

use super::shims::{
    KERNEL32_DLL, UCRTBASE_DLL, arm64_adrp, emit_arm64_adrp_add, emit_shims, emit_stub, plan_shims,
    windows_crt_dll, windows_deps,
};
use super::{
    Buf, Executable, PatchCtx, Plan, SectionAddrs, Symbol, patch_code, plan_veneers, resolve_sym,
};

const PE_BASE: u64 = 0x0040_0000;
const SECT_ALIGN: usize = 0x1000;
const FILE_ALIGN: usize = 0x200;
const HEADERS_RAW: usize = 0x400;
const TEXT_RVA: usize = 0x1000;

const UNIX_ONLY_IMPORTS: &[&str] = &[
    "sched_yield",
    "pthread_key_create",
    "pthread_getspecific",
    "pthread_setspecific",
    "posix_memalign",
    "_longjmp",
];

struct PeImport<'a> {
    name: &'a str,
    dll: Cow<'static, str>,
}

fn normalize_pe_dll(raw: &str) -> Cow<'static, str> {
    let trimmed = raw.trim().to_ascii_lowercase();
    let stem = trimmed.strip_suffix(".lib").unwrap_or(&trimmed);
    match stem {
        "ucrt" | "ucrtbase" | "ucrtbase.dll" | "vcruntime" | "libcmt" => {
            Cow::Borrowed(UCRTBASE_DLL)
        }
        "kernel32" | "kernel32.dll" => Cow::Borrowed(KERNEL32_DLL),
        other if other.ends_with(".dll") || other.ends_with(".drv") || other.ends_with(".exe") => {
            Cow::Owned(other.to_owned())
        }
        other => Cow::Owned(format!("{other}.dll")),
    }
}

fn collect_pe_imports<'a>(arch: Arch, plan: &Plan<'a>) -> Vec<PeImport<'a>> {
    let mut imports: Vec<PeImport<'a>> = Vec::new();
    let mut push_fixed = |name: &'a str, dll: &'static str| {
        if !imports.iter().any(|imp| imp.name == name) {
            imports.push(PeImport {
                name,
                dll: Cow::Borrowed(dll),
            });
        }
    };
    push_fixed("ExitProcess", KERNEL32_DLL);
    for &shim in &plan.shims {
        for (dep, dll) in windows_deps(arch, shim) {
            push_fixed(dep, dll);
        }
    }
    for imp in &plan.imports {
        let link_lib = imp.link_lib.map(str::trim).filter(|s| !s.is_empty());
        if link_lib.is_none() && UNIX_ONLY_IMPORTS.contains(&imp.name) {
            continue;
        }
        if imports.iter().any(|existing| existing.name == imp.name) {
            continue;
        }
        let dll = link_lib.map_or_else(
            || Cow::Borrowed(windows_crt_dll(imp.name)),
            normalize_pe_dll,
        );
        imports.push(PeImport {
            name: imp.name,
            dll,
        });
    }
    imports
}

struct ImportGroup<'a> {
    dll: Cow<'static, str>,
    entries: Vec<(usize, &'a str)>,
}

struct IdataLayout<'a> {
    groups: Vec<ImportGroup<'a>>,
    iat_rva_of: Vec<u32>,
    iat_rva: u32,
    iat_size: u32,
    bytes: Vec<u8>,
}

fn build_idata<'a>(imports: &[PeImport<'a>], idata_rva: u32) -> IdataLayout<'a> {
    let mut groups: Vec<ImportGroup<'a>> = Vec::new();
    for (idx, imp) in imports.iter().enumerate() {
        if let Some(group) = groups.iter_mut().find(|g| g.dll == imp.dll) {
            group.entries.push((idx, imp.name));
        } else {
            groups.push(ImportGroup {
                dll: imp.dll.clone(),
                entries: vec![(idx, imp.name)],
            });
        }
    }

    let idt_size = (groups.len() + 1) * 20;
    let total_thunks: usize = groups.iter().map(|g| g.entries.len() + 1).sum();
    let iat_off = idt_size.next_multiple_of(8);
    let iat_size = total_thunks * 8;
    let ilt_off = iat_off + iat_size;
    let mut cursor = ilt_off + iat_size;

    let mut dll_rvas = Vec::with_capacity(groups.len());
    let mut hint_rvas: Vec<Vec<u32>> = Vec::with_capacity(groups.len());
    let mut iat_rva_of = vec![0u32; imports.len()];
    let mut group_iat_rvas = Vec::with_capacity(groups.len());
    let mut group_ilt_rvas = Vec::with_capacity(groups.len());

    let mut thunk_slot = 0usize;
    for group in &groups {
        group_iat_rvas.push(idata_rva + (iat_off + thunk_slot * 8) as u32);
        group_ilt_rvas.push(idata_rva + (ilt_off + thunk_slot * 8) as u32);
        let mut rvas = Vec::with_capacity(group.entries.len());
        for &(imp_idx, name) in &group.entries {
            iat_rva_of[imp_idx] = idata_rva + (iat_off + thunk_slot * 8) as u32;
            thunk_slot += 1;
            cursor = cursor.next_multiple_of(2);
            rvas.push(idata_rva + cursor as u32);
            cursor += 2 + name.len() + 1;
        }
        thunk_slot += 1;
        hint_rvas.push(rvas);
        dll_rvas.push(idata_rva + cursor as u32);
        cursor += group.dll.len() + 1;
    }

    let mut out = Buf::new();
    for (gi, _) in groups.iter().enumerate() {
        out.words(&[group_ilt_rvas[gi], 0, 0, dll_rvas[gi], group_iat_rvas[gi]]);
    }
    out.bytes(&[0; 20]);
    out.align(8);
    for rvas in &hint_rvas {
        for &rva in rvas {
            out.u64(u64::from(rva));
        }
        out.u64(0);
    }
    for rvas in &hint_rvas {
        for &rva in rvas {
            out.u64(u64::from(rva));
        }
        out.u64(0);
    }
    for group in &groups {
        for &(_, name) in &group.entries {
            out.align(2);
            out.u16(0);
            out.bytes(name.as_bytes());
            out.u8(0);
        }
        out.bytes(group.dll.as_bytes());
        out.u8(0);
    }

    IdataLayout {
        groups,
        iat_rva_of,
        iat_rva: idata_rva + iat_off as u32,
        iat_size: iat_size as u32,
        bytes: out.0,
    }
}

fn emit_pe_start(arch: Arch, start_va: u64, tramp_va: u64, exit_iat_va: u64, out: &mut Buf) {
    if arch == Arch::X64 {
        out.bytes(&[0x48, 0x83, 0xEC, 0x28, 0xE8]);
        out.u32(((tramp_va as i64 - (start_va as i64 + 9)) as i32) as u32);
        out.bytes(&[0x89, 0xC1, 0xFF, 0x15]);
        out.u32(((exit_iat_va as i64 - (start_va as i64 + 17)) as i32) as u32);
        out.bytes(&[0x0F, 0x0B]);
    } else if arch == Arch::Arm64 {
        let bl_delta = ((tramp_va as i64 - start_va as i64) >> 2) as u32;
        out.u32(0x9400_0000 | (bl_delta & 0x03FF_FFFF));
        let imm12 = ((exit_iat_va & 0xFFF) >> 3) as u32;
        out.words(&[
            arm64_adrp(start_va + 4, exit_iat_va, 16),
            0xF940_0210 | (imm12 << 10),
            0xD63F_0200,
            0xD420_0000,
        ]);
    }
}

fn emit_pe_trampoline(
    arch: Arch,
    tramp_va: u64,
    lang_start_va: u64,
    user_main_va: u64,
    out: &mut Buf,
) {
    if arch == Arch::X64 {
        out.bytes(&[0x48, 0x8D, 0x0D]);
        out.u32(((user_main_va as i64 - (tramp_va as i64 + 7)) as i32) as u32);
        out.bytes(&[0x31, 0xD2, 0x45, 0x31, 0xC0, 0x45, 0x31, 0xC9, 0xE9]);
        out.u32(((lang_start_va as i64 - (tramp_va as i64 + 20)) as i32) as u32);
    } else if arch == Arch::Arm64 {
        emit_arm64_adrp_add(tramp_va, user_main_va, 0, out);
        let delta = ((lang_start_va as i64 - (tramp_va + 20) as i64) >> 2) as u32;
        out.words(&[
            0xD280_0001,
            0xD280_0002,
            0xD280_0003,
            0x1400_0000 | (delta & 0x03FF_FFFF),
        ]);
    }
}

fn write_section_hdr(
    out: &mut Buf,
    name: &[u8; 8],
    vsize: u32,
    rva: u32,
    raw_size: u32,
    raw_off: u32,
    chars: u32,
) {
    out.bytes(name);
    out.words(&[vsize, rva, raw_size, raw_off, 0, 0, 0, chars]);
}

pub(super) fn emit(exe: &Executable, (lang_start, user_main): (u32, u32)) -> Result<Vec<u8>> {
    let arch = exe.target.arch;
    let (machine, stub_size, start_size, tramp_size) = match arch {
        Arch::X64 => (0x8664_u16, 6usize, 19usize, 20usize),
        Arch::Arm64 => (0xAA64_u16, 12usize, 20usize, 24usize),
    };

    let plan = Plan::collect(exe, &["ExitProcess"])?;
    let pe_imports = collect_pe_imports(arch, &plan);
    let pe_import_idx = |name: &str| pe_imports.iter().position(|imp| imp.name == name);
    let stubs_in_text = 0usize;
    let start_in_text = (stubs_in_text + pe_imports.len() * stub_size).next_multiple_of(16);
    let tramp_in_text = (start_in_text + start_size).next_multiple_of(16);
    let shims_in_text = (tramp_in_text + tramp_size).next_multiple_of(16);
    let text_base_va = PE_BASE + TEXT_RVA as u64;
    let probe_addrs = SectionAddrs {
        text: text_base_va,
        thread_key: text_base_va,
        unwind_table: if exe.unwind_table_off.is_some() {
            text_base_va
        } else {
            0
        },
        unwind_walk: text_base_va,
        ..SectionAddrs::default()
    };
    let (shim_offs, head_in_text) = plan_shims(
        arch,
        Os::Windows,
        &plan,
        text_base_va,
        shims_in_text,
        probe_addrs,
    )?;
    let post_rel = exe.text.len().next_multiple_of(16);
    let veneers = plan_veneers(
        exe,
        &plan,
        text_base_va,
        head_in_text,
        &shim_offs,
        |_, _| stubs_in_text,
    )?;
    let text_in_text = head_in_text + veneers.pre_len;
    let text_vsize = (text_in_text + post_rel + veneers.post_len).next_multiple_of(16);
    let text_raw_off = HEADERS_RAW;
    let text_raw_size = text_vsize.next_multiple_of(FILE_ALIGN);

    let rdata_rva = TEXT_RVA + text_vsize.next_multiple_of(SECT_ALIGN);
    let rdata_vsize = (exe.rodata.len() + 8).max(8);
    let rdata_raw_off = text_raw_off + text_raw_size;
    let rdata_raw_size = rdata_vsize.next_multiple_of(FILE_ALIGN);

    let data_rva = rdata_rva + rdata_vsize.next_multiple_of(SECT_ALIGN);
    let (thread_key_off, bss_total) = plan.bss_layout(exe.bss);
    let bss_in_data = exe.data.len().next_multiple_of(16);
    let data_vsize = (bss_in_data + bss_total as usize).max(16);
    let data_raw_off = rdata_raw_off + rdata_raw_size;
    let data_raw_size = exe.data.len().next_multiple_of(FILE_ALIGN);

    let idata_rva = data_rva + data_vsize.next_multiple_of(SECT_ALIGN);
    let idata = build_idata(&pe_imports, idata_rva as u32);
    let idata_vsize = idata.bytes.len().max(1);
    let idata_raw_off = data_raw_off + data_raw_size;
    let idata_raw_size = idata.bytes.len().next_multiple_of(FILE_ALIGN);
    let size_of_image = (idata_rva + idata_vsize.next_multiple_of(SECT_ALIGN)) as u32;

    let rodata_va = PE_BASE + rdata_rva as u64;
    let (unwind_table, unwind_walk) = plan.unwind_addrs(exe, rodata_va, text_base_va, &shim_offs);
    let addrs = SectionAddrs {
        text: text_base_va + text_in_text as u64,
        rodata: rodata_va,
        data: PE_BASE + data_rva as u64,
        bss: PE_BASE + (data_rva + bss_in_data) as u64,
        dso_handle: PE_BASE + (rdata_rva + exe.rodata.len()) as u64,
        thread_key: PE_BASE + (data_rva + bss_in_data + thread_key_off as usize) as u64,
        tlv_desc: 0,
        unwind_tls: 0,
        thread_tls: 0,
        frame_tls: 0,
        unwind_table,
        unwind_walk,
    };

    let stub_va = |name: &str| {
        let Some(idx) = pe_import_idx(name) else {
            bail!("missing PE import {name}")
        };
        Ok(text_base_va + (stubs_in_text + idx * stub_size) as u64)
    };
    let iat_va = |idx: usize| PE_BASE + u64::from(idata.iat_rva_of[idx]);
    let patch_stub_va = |name: &str| match pe_import_idx(name) {
        Some(idx) => Ok(text_base_va + (stubs_in_text + idx * stub_size) as u64),
        None if UNIX_ONLY_IMPORTS.contains(&name) => Ok(text_base_va),
        None => bail!("missing PE import {name}"),
    };
    let patch_iat_va = |name: &str| match pe_import_idx(name) {
        Some(idx) => Ok(iat_va(idx)),
        None if UNIX_ONLY_IMPORTS.contains(&name) => Ok(PE_BASE),
        None => bail!("missing PE import {name}"),
    };
    let resolve =
        |sym: &Symbol| resolve_sym(sym, addrs, text_base_va, &shim_offs, &plan, &patch_stub_va);

    let mut out = Buf::new();
    out.bytes(b"MZ");
    out.bytes(&[0; 58]);
    out.u32(0x40);
    out.bytes(b"PE\0\0");
    out.u16(machine);
    out.u16(4);
    out.words(&[0, 0, 0]);
    out.u16(240);
    out.u16(0x0022);

    out.u16(0x020B);
    out.bytes(&[0, 0]);
    out.words(&[
        text_raw_size as u32,
        (rdata_raw_size + data_raw_size + idata_raw_size) as u32,
        bss_total,
    ]);
    out.words(&[(TEXT_RVA + start_in_text) as u32, TEXT_RVA as u32]);
    out.u64(PE_BASE);
    out.words(&[SECT_ALIGN as u32, FILE_ALIGN as u32]);
    [6_u16, 0, 0, 0, 6, 0].into_iter().for_each(|v| out.u16(v));
    out.words(&[0, size_of_image, HEADERS_RAW as u32, 0]);
    out.u16(3);
    out.u16(0x0100);
    out.u64s(&[0x0010_0000, 0x1000, 0x0010_0000, 0x1000]);
    out.words(&[0, 16]);
    for dir in 0..16 {
        match dir {
            1 => out.words(&[idata_rva as u32, ((idata.groups.len() + 1) * 20) as u32]),
            12 => out.words(&[idata.iat_rva, idata.iat_size]),
            _ => out.words(&[0, 0]),
        }
    }

    write_section_hdr(
        &mut out,
        b".text\0\0\0",
        text_vsize as u32,
        TEXT_RVA as u32,
        text_raw_size as u32,
        text_raw_off as u32,
        0x6000_0020,
    );
    write_section_hdr(
        &mut out,
        b".rdata\0\0",
        rdata_vsize as u32,
        rdata_rva as u32,
        rdata_raw_size as u32,
        rdata_raw_off as u32,
        0x4000_0040,
    );
    write_section_hdr(
        &mut out,
        b".data\0\0\0",
        data_vsize as u32,
        data_rva as u32,
        data_raw_size as u32,
        data_raw_off as u32,
        0xC000_0040,
    );
    write_section_hdr(
        &mut out,
        b".idata\0\0",
        idata_vsize as u32,
        idata_rva as u32,
        idata_raw_size as u32,
        idata_raw_off as u32,
        0xC000_0040,
    );
    out.0.resize(text_raw_off, 0);

    (0..pe_imports.len()).for_each(|i| {
        emit_stub(
            arch,
            text_base_va + (stubs_in_text + i * stub_size) as u64,
            iat_va(i),
            &mut out,
        )
    });
    out.0.resize(text_raw_off + start_in_text, 0);
    let Some(exit_idx) = pe_import_idx("ExitProcess") else {
        bail!("missing PE import ExitProcess")
    };
    emit_pe_start(
        arch,
        text_base_va + start_in_text as u64,
        text_base_va + tramp_in_text as u64,
        iat_va(exit_idx),
        &mut out,
    );
    out.0.resize(text_raw_off + tramp_in_text, 0);
    emit_pe_trampoline(
        arch,
        text_base_va + tramp_in_text as u64,
        addrs.text + u64::from(lang_start),
        addrs.text + u64::from(user_main),
        &mut out,
    );
    out.0.resize(text_raw_off + shims_in_text, 0);
    emit_shims(
        arch,
        Os::Windows,
        &plan,
        text_base_va - text_raw_off as u64,
        0,
        addrs,
        &stub_va,
        None,
        &mut out,
    )?;
    out.0.resize(text_raw_off + head_in_text, 0);
    veneers.emit_pre(&resolve, &mut out)?;
    out.0.resize(text_raw_off + text_in_text, 0);
    out.bytes(&exe.text);
    out.align(16);
    veneers.emit_post(&resolve, &mut out)?;
    let ctx = PatchCtx {
        text_base: text_raw_off + text_in_text,
        text_va: addrs.text,
        pie_arm64: false,
        pie_x64: false,
        addrs,
        base_va: text_base_va,
        shim_offs: &shim_offs,
        plan: &plan,
        stub_va: &patch_stub_va,
        got_va: &patch_iat_va,
        veneers: &veneers,
    };
    patch_code(&exe.relocs, &ctx, &mut out.0)?;

    out.0.resize(rdata_raw_off, 0);
    out.bytes(&exe.rodata);
    out.bytes(&[0; 8]);
    for ptr in exe.pointers.iter().filter(|ptr| ptr.in_rodata) {
        out.patch_u64(rdata_raw_off + ptr.offset as usize, resolve(&ptr.symbol)?);
    }

    out.0.resize(data_raw_off, 0);
    out.bytes(&exe.data);
    for ptr in exe.pointers.iter().filter(|ptr| !ptr.in_rodata) {
        out.patch_u64(data_raw_off + ptr.offset as usize, resolve(&ptr.symbol)?);
    }

    out.0.resize(idata_raw_off, 0);
    out.bytes(&idata.bytes);
    out.0.resize(idata_raw_off + idata_raw_size, 0);
    Ok(out.0)
}
