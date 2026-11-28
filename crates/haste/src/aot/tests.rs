use std::path::PathBuf;

use haste_codegen::{Arch, Format, RelocKind, TARGETS, Target};

use super::{ExeSymbol, Executable, Symbol};
use crate::host::HostSymbol;
use crate::outline::source::file_key;
use crate::substrate::read_u32;

fn read_u16(bytes: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(bytes[off..off + 2].try_into().expect("u16 slice"))
}

fn read_u32_at(bytes: &[u8], off: usize) -> u32 {
    read_u32(bytes, off)
}

fn read_u64(bytes: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(bytes[off..off + 8].try_into().expect("u64 slice"))
}

fn read_cstr(bytes: &[u8], off: usize) -> &str {
    let end = bytes[off..]
        .iter()
        .position(|&b| b == 0)
        .map_or(bytes.len(), |n| off + n);
    std::str::from_utf8(&bytes[off..end]).expect("utf8 cstr")
}

fn minimal_executable_with_imports(
    target: &Target,
    imports: &[(&str, Option<&str>)],
    with_debug_line: bool,
) -> Executable {
    let mut exe = Executable::new(target);
    let (stub, call_off, kind) = match target.arch {
        Arch::X64 => (vec![0xE8, 0, 0, 0, 0, 0xC3], 1u32, RelocKind::PcRel32),
        Arch::Arm64 => (
            vec![0x00, 0x00, 0x00, 0x94, 0xC0, 0x03, 0x5F, 0xD6],
            0u32,
            RelocKind::Arm64Call26,
        ),
    };
    let lang_start = exe.function(&stub, 16, &[], &[]);
    let lines = if with_debug_line {
        let src_path = PathBuf::from("src/main.rs");
        let place = (u64::from(file_key(&src_path)) << 32) | 1;
        exe.sources(vec![src_path]);
        vec![(0u32, place)]
    } else {
        Vec::new()
    };
    let mut user_code = Vec::new();
    let mut call_sites = Vec::with_capacity(imports.len());
    for _ in imports {
        let site = user_code.len() as u32 + call_off;
        user_code.extend_from_slice(&stub);
        call_sites.push(site);
    }
    if user_code.is_empty() {
        user_code.extend_from_slice(&stub);
    }
    let user_main = exe.function(&user_code, 16, &[], &lines);
    let ro = exe.rodata(b"hello\0", 8);
    let data = exe.data(&[0u8; 8], 8);
    let _bss = exe.bss(16, 8);
    exe.entry(lang_start, user_main);
    for (site, &(name, lib)) in call_sites.into_iter().zip(imports) {
        exe.patch(
            user_main + site,
            kind,
            Symbol::Host(HostSymbol::Dynamic {
                name: name.into(),
                lib: lib.map(Into::into),
            }),
        );
    }
    exe.pointer(false, data, Symbol::Rodata(ro));
    exe.symbols(vec![
        ExeSymbol {
            offset: lang_start,
            name: "lang_start".into(),
            exported: false,
        },
        ExeSymbol {
            offset: user_main,
            name: "main".into(),
            exported: false,
        },
    ]);
    exe
}

fn minimal_executable(target: &Target, import: &str, with_debug_line: bool) -> Executable {
    minimal_executable_with_imports(target, &[(import, None)], with_debug_line)
}

fn parse_pe_idata(bytes: &[u8]) -> Vec<(String, Vec<String>)> {
    let pe_off = read_u32_at(bytes, 0x3C) as usize;
    let nsec = read_u16(bytes, pe_off + 6) as usize;
    let opt_size = read_u16(bytes, pe_off + 20) as usize;
    let opt_off = pe_off + 24;
    let import_dir_rva = read_u32_at(bytes, opt_off + 112 + 8);
    let sec_base = opt_off + opt_size;
    let mut idata_sec = None;
    for i in 0..nsec {
        let s = sec_base + i * 40;
        if read_cstr(bytes, s) == ".idata" {
            let rva = read_u32_at(bytes, s + 12);
            let raw_off = read_u32_at(bytes, s + 20) as usize;
            idata_sec = Some((rva, raw_off));
            break;
        }
    }
    let (idata_rva, idata_off) = idata_sec.expect(".idata section");
    let rva_to_off = |rva: u32| idata_off + (rva - idata_rva) as usize;
    let mut desc = rva_to_off(import_dir_rva);
    let mut dll_imports = Vec::new();
    loop {
        let ilt_rva = read_u32_at(bytes, desc);
        let name_rva = read_u32_at(bytes, desc + 12);
        if ilt_rva == 0 && name_rva == 0 {
            break;
        }
        let dll = read_cstr(bytes, rva_to_off(name_rva)).to_owned();
        let mut syms = Vec::new();
        let mut thunk = rva_to_off(ilt_rva);
        loop {
            let entry_val = read_u64(bytes, thunk);
            if entry_val == 0 {
                break;
            }
            syms.push(read_cstr(bytes, rva_to_off(entry_val as u32) + 2).to_owned());
            thunk += 8;
        }
        dll_imports.push((dll, syms));
        desc += 20;
    }
    dll_imports
}

fn elf_sections(bytes: &[u8]) -> Vec<(String, u64, usize, usize)> {
    let shoff = read_u64(bytes, 40) as usize;
    let shentsize = read_u16(bytes, 58) as usize;
    let shnum = read_u16(bytes, 60) as usize;
    let shstrndx = read_u16(bytes, 62) as usize;
    let str_hdr = shoff + shstrndx * shentsize;
    let str_off = read_u64(bytes, str_hdr + 24) as usize;
    (0..shnum)
        .map(|i| {
            let h = shoff + i * shentsize;
            let name = read_cstr(bytes, str_off + read_u32_at(bytes, h) as usize).to_owned();
            let addr = read_u64(bytes, h + 16);
            let off = read_u64(bytes, h + 24) as usize;
            let size = read_u64(bytes, h + 32) as usize;
            (name, addr, off, size)
        })
        .collect()
}

#[test]
fn encode_requires_entry_point() {
    let target = Target::find("x86_64-unknown-linux-gnu").expect("x64 linux");
    let exe = Executable::new(target);
    assert!(exe.encode().is_err());
}

#[test]
fn elf_structure_relocs_dynamic_table_and_debug_line_toggle() {
    for target in TARGETS.iter().filter(|t| t.format == Format::Elf) {
        let expected_machine = match target.arch {
            Arch::X64 => 0x3Eu16,
            Arch::Arm64 => 0xB7u16,
        };
        let bytes = minimal_executable(target, "puts", false)
            .encode()
            .expect("emit ELF");
        assert_eq!(&bytes[0..4], b"\x7fELF");
        assert_eq!(bytes[4], 2);
        assert_eq!(bytes[5], 1);
        assert_eq!(read_u16(&bytes, 16), 2);
        assert_eq!(read_u16(&bytes, 18), expected_machine);

        let entry = read_u64(&bytes, 24);
        let phoff = read_u64(&bytes, 32) as usize;
        let phentsize = read_u16(&bytes, 54) as usize;
        let phnum = read_u16(&bytes, 56) as usize;
        let mut ph_types = Vec::with_capacity(phnum);
        let mut entry_in_exec = false;
        for i in 0..phnum {
            let p = phoff + i * phentsize;
            let p_type = read_u32_at(&bytes, p);
            let p_flags = read_u32_at(&bytes, p + 4);
            let p_offset = read_u64(&bytes, p + 8) as usize;
            let p_vaddr = read_u64(&bytes, p + 16);
            let p_filesz = read_u64(&bytes, p + 32) as usize;
            let p_memsz = read_u64(&bytes, p + 40);
            assert!(p_offset + p_filesz <= bytes.len());
            if p_type == 1 && (p_flags & 1) != 0 && entry >= p_vaddr && entry < p_vaddr + p_memsz {
                entry_in_exec = true;
            }
            ph_types.push(p_type);
        }
        assert!(entry_in_exec, "{} entry not in exec PT_LOAD", target.triple);
        assert!(ph_types.contains(&1) && ph_types.contains(&2) && ph_types.contains(&3));

        let sections = elf_sections(&bytes);
        let (_, ro_va, _, _) = sections
            .iter()
            .find(|(n, ..)| n == ".rodata")
            .expect(".rodata section");
        let (_, _, data_off, _) = sections
            .iter()
            .find(|(n, ..)| n == ".data")
            .expect(".data section");
        assert_eq!(
            read_u64(&bytes, *data_off),
            *ro_va,
            "{} .data pointer relocation must target .rodata VA",
            target.triple
        );

        let (_, _, dynstr_off, dynstr_len) = sections
            .iter()
            .find(|(n, ..)| n == ".dynstr")
            .expect(".dynstr section");
        let (_, _, dyn_off, dyn_len) = sections
            .iter()
            .find(|(n, ..)| n == ".dynamic")
            .expect(".dynamic section");
        let mut needed = Vec::new();
        for pos in (*dyn_off..dyn_off + dyn_len).step_by(16) {
            if read_u64(&bytes, pos) == 1 {
                let str_off = read_u64(&bytes, pos + 8) as usize;
                needed.push(read_cstr(&bytes, dynstr_off + str_off).to_owned());
            }
        }
        assert!(needed.iter().any(|s| s == "libc.so.6"));
        let dynstr_slice = &bytes[*dynstr_off..dynstr_off + dynstr_len];
        assert!(dynstr_slice.windows(5).any(|w| w == b"puts\0"));
        assert!(!sections.iter().any(|(n, ..)| n == ".debug_line"));

        let dbg_bytes = minimal_executable(target, "puts", true)
            .encode()
            .expect("emit ELF with debug_line");
        let dbg_sections = elf_sections(&dbg_bytes);
        assert!(dbg_sections.iter().any(|(n, ..)| n == ".debug_line"));
    }
}

struct MachOView {
    cmds: Vec<u32>,
    text_range: std::ops::Range<usize>,
    entry: (usize, u64, usize),
    dyld_info: (u32, u32, usize, usize),
    symtab: (usize, usize, usize, usize),
    dysymtab: ([u32; 6], usize, usize),
    uuid: [u8; 16],
    func_starts: (usize, usize),
    dylibs: Vec<String>,
}

fn parse_macho(bytes: &[u8], expected_cpu: u32) -> MachOView {
    assert_eq!(
        (
            read_u32_at(bytes, 0),
            read_u32_at(bytes, 4),
            read_u32_at(bytes, 12)
        ),
        (0xFEED_FACF, expected_cpu, 2)
    );
    let (ncmds, sizeofcmds) = (
        read_u32_at(bytes, 16) as usize,
        read_u32_at(bytes, 20) as usize,
    );
    let (mut cursor, mut cmds, mut dylibs) = (32usize, Vec::with_capacity(ncmds), Vec::new());
    let (mut text_range, mut data_off, mut ro_va, mut entry_off) = (None, None, None, None);
    let (mut dyld_info, mut symtab, mut dysymtab, mut uuid, mut func_starts) =
        (None, None, None, None, None);
    for _ in 0..ncmds {
        let (cmd, cmdsize) = (
            read_u32_at(bytes, cursor),
            read_u32_at(bytes, cursor + 4) as usize,
        );
        match cmd {
            0x19 => {
                let (foff, fsz) = (
                    read_u64(bytes, cursor + 40) as usize,
                    read_u64(bytes, cursor + 48) as usize,
                );
                assert!(foff + fsz <= bytes.len());
                if read_cstr(bytes, cursor + 8) == "__TEXT" {
                    text_range = Some(foff..foff + fsz);
                }
                for s in 0..read_u32_at(bytes, cursor + 64) as usize {
                    let sh = cursor + 72 + s * 80;
                    match read_cstr(bytes, sh) {
                        "__data" => data_off = Some(read_u32_at(bytes, sh + 48) as usize),
                        "__const" => ro_va = Some(read_u64(bytes, sh + 32)),
                        _ => {}
                    }
                }
            }
            0x8000_0022 => {
                dyld_info = Some((
                    read_u32_at(bytes, cursor + 12),
                    read_u32_at(bytes, cursor + 20),
                    read_u32_at(bytes, cursor + 40) as usize,
                    read_u32_at(bytes, cursor + 44) as usize,
                ))
            }
            0x8000_0028 => entry_off = Some(read_u64(bytes, cursor + 8) as usize),
            0x1B => {
                assert_eq!(cmdsize, 24);
                uuid = Some(bytes[cursor + 8..cursor + 24].try_into().expect("uuid"));
            }
            0x26 => {
                assert_eq!(cmdsize, 16);
                func_starts = Some((
                    read_u32_at(bytes, cursor + 8) as usize,
                    read_u32_at(bytes, cursor + 12) as usize,
                ));
            }
            2 => {
                symtab = Some((
                    read_u32_at(bytes, cursor + 8) as usize,
                    read_u32_at(bytes, cursor + 12) as usize,
                    read_u32_at(bytes, cursor + 16) as usize,
                    read_u32_at(bytes, cursor + 20) as usize,
                ))
            }
            11 => {
                dysymtab = Some((
                    [0, 1, 2, 3, 4, 5].map(|i| read_u32_at(bytes, cursor + 8 + i * 4)),
                    read_u32_at(bytes, cursor + 56) as usize,
                    read_u32_at(bytes, cursor + 60) as usize,
                ))
            }
            12 => dylibs.push(
                read_cstr(bytes, cursor + read_u32_at(bytes, cursor + 8) as usize).to_owned(),
            ),
            _ => {}
        }
        cmds.push(cmd);
        cursor += cmdsize;
    }
    assert_eq!(cursor - 32, sizeofcmds);
    MachOView {
        cmds,
        text_range: text_range.expect("__TEXT"),
        entry: (
            data_off.expect("__data"),
            ro_va.expect("__const"),
            entry_off.expect("LC_MAIN"),
        ),
        dyld_info: dyld_info.expect("LC_DYLD_INFO_ONLY"),
        symtab: symtab.expect("LC_SYMTAB"),
        dysymtab: dysymtab.expect("LC_DYSYMTAB"),
        uuid: uuid.expect("LC_UUID"),
        func_starts: func_starts.expect("LC_FUNCTION_STARTS"),
        dylibs,
    }
}

#[test]
fn macho_structure_relocs_load_commands_entry_and_symtab_imports() {
    for target in TARGETS.iter().filter(|t| t.format == Format::MachO) {
        let cpu = if target.arch == Arch::Arm64 {
            0x0100_000C
        } else {
            0x0100_0007
        };
        let bytes = minimal_executable(target, "puts", false)
            .encode()
            .expect("emit Mach-O");
        let view = parse_macho(&bytes, cpu);
        assert!(view.text_range.contains(&view.entry.2));
        assert_eq!(read_u64(&bytes, view.entry.0), view.entry.1);
        assert!(view.dyld_info.0 > 0 && view.dyld_info.1 > 0);
        assert!(view.cmds.contains(&11) && view.cmds.contains(&14));
        assert!(
            view.dylibs
                .iter()
                .any(|d| d == "/usr/lib/libSystem.B.dylib")
        );
        assert!(view.uuid.iter().any(|&b| b != 0));
        let (fs_off, fs_len) = view.func_starts;
        assert!(fs_len >= 2 && fs_off + fs_len <= bytes.len());
        let starts = decode_func_starts(&bytes[fs_off..fs_off + fs_len]);
        assert!(
            starts.contains(&(view.entry.2 as u64))
                && starts
                    .iter()
                    .all(|&o| view.text_range.contains(&(o as usize)))
        );
        let strtab = &bytes[view.symtab.2..view.symtab.2 + view.symtab.3];
        assert!(
            strtab.windows(6).any(|w| w == b"_puts\0")
                && strtab.windows(6).any(|w| w == b"_main\0")
        );
    }
}

#[test]
fn pe_structure_relocs_sections_entry_and_idata_import_directory() {
    for target in TARGETS.iter().filter(|t| t.format == Format::Pe) {
        let expected_machine = match target.arch {
            Arch::X64 => 0x8664u16,
            Arch::Arm64 => 0xAA64u16,
        };
        let bytes =
            minimal_executable_with_imports(target, &[("GetLastError", Some("kernel32"))], false)
                .encode()
                .expect("emit PE32+");
        assert_eq!(&bytes[0..2], b"MZ");
        let pe_off = read_u32_at(&bytes, 0x3C) as usize;
        assert_eq!(&bytes[pe_off..pe_off + 4], b"PE\0\0");
        assert_eq!(read_u16(&bytes, pe_off + 4), expected_machine);
        let nsec = read_u16(&bytes, pe_off + 6) as usize;
        assert_eq!(nsec, 4);
        let opt_size = read_u16(&bytes, pe_off + 20) as usize;
        let opt_off = pe_off + 24;
        assert_eq!(read_u16(&bytes, opt_off), 0x020B);
        let entry_rva = read_u32_at(&bytes, opt_off + 16);
        let image_base = read_u64(&bytes, opt_off + 24);

        let sec_base = opt_off + opt_size;
        let mut sections = Vec::with_capacity(nsec);
        for i in 0..nsec {
            let s = sec_base + i * 40;
            let name = read_cstr(&bytes, s).to_owned();
            let vsize = read_u32_at(&bytes, s + 8);
            let rva = read_u32_at(&bytes, s + 12);
            let raw_size = read_u32_at(&bytes, s + 16) as usize;
            let raw_off = read_u32_at(&bytes, s + 20) as usize;
            assert!(raw_off + raw_size <= bytes.len());
            sections.push((name, rva, vsize, raw_off));
        }
        let text_sec = sections.iter().find(|(n, ..)| n == ".text").expect(".text");
        assert!(entry_rva >= text_sec.1 && entry_rva < text_sec.1 + text_sec.2);
        let rdata_sec = sections
            .iter()
            .find(|(n, ..)| n == ".rdata")
            .expect(".rdata");
        let data_sec = sections.iter().find(|(n, ..)| n == ".data").expect(".data");
        assert_eq!(
            read_u64(&bytes, data_sec.3),
            image_base + u64::from(rdata_sec.1),
            "{} .data pointer relocation must target .rdata VA",
            target.triple
        );
        let dll_imports = parse_pe_idata(&bytes);
        let (_, k32_syms) = dll_imports
            .iter()
            .find(|(dll, _)| dll == "kernel32.dll")
            .expect("kernel32.dll descriptor");
        assert!(k32_syms.iter().any(|s| s == "ExitProcess"));
        assert!(k32_syms.iter().any(|s| s == "GetLastError"));
    }
}

#[test]
fn pe_idata_maps_explicit_link_libraries_without_guessing_case() {
    let cases = [
        ("MessageBoxW", Some("user32"), "user32.dll"),
        ("wsprintfA", Some("USER32.dll"), "user32.dll"),
        ("RegOpenKeyExW", Some("advapi32"), "advapi32.dll"),
        ("advapi_lower", Some("advapi32.lib"), "advapi32.dll"),
        ("BCryptGenRandom", Some("bcrypt"), "bcrypt.dll"),
        ("bcrypt_lower", Some("bcrypt.dll"), "bcrypt.dll"),
        ("GetLastError", Some("kernel32"), "kernel32.dll"),
        ("kernel_lower", Some("kernel32.dll"), "kernel32.dll"),
        ("printf", Some("ucrtbase"), "ucrtbase.dll"),
        ("UppercaseUcrt", Some("ucrt"), "ucrtbase.dll"),
        ("OpenPrinterW", Some("winspool.drv"), "winspool.drv"),
    ];
    let imports: Vec<(&str, Option<&str>)> = cases.iter().map(|&(n, lib, _)| (n, lib)).collect();
    for target in TARGETS.iter().filter(|t| t.format == Format::Pe) {
        let bytes = minimal_executable_with_imports(target, &imports, false)
            .encode()
            .expect("emit PE32+");
        let dll_imports = parse_pe_idata(&bytes);
        for &(sym, _, expected_dll) in &cases {
            let found = dll_imports
                .iter()
                .find(|(dll, _)| dll == expected_dll)
                .unwrap_or_else(|| panic!("{} missing descriptor {expected_dll}", target.triple));
            assert!(
                found.1.iter().any(|s| s == sym),
                "{} expected {sym} in {expected_dll}, got {:?}",
                target.triple,
                dll_imports
            );
            for (other_dll, other_syms) in &dll_imports {
                if other_dll != expected_dll {
                    assert!(
                        !other_syms.iter().any(|s| s == sym),
                        "{} mapped {sym} to wrong DLL {other_dll}",
                        target.triple
                    );
                }
            }
        }
    }
}

#[test]
fn pe_end_to_end_compiler_emit_routes_std_and_explicit_link_attributes() {
    let base = std::env::current_exe()
        .expect("current test exe")
        .parent()
        .expect("test exe parent")
        .to_path_buf();
    let dir = base.join(format!("haste-pe-link-e2e-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&dir));
    std::fs::create_dir_all(dir.join("src")).expect("create fixture src");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"pe_link_e2e\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        dir.join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"pe_link_e2e\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(
        dir.join("src/main.rs"),
        r#"
#[link(name = "user32")]
extern "system" {
    fn MessageBoxW(hwnd: usize, text: *const u16, cap: *const u16, kind: u32) -> i32;
}

#[link(name = "advapi32")]
extern "system" {
    fn RegCloseKey(key: usize) -> i32;
}

#[cfg_attr(target_os = "windows", link(name = "bcrypt", kind = "raw-dylib"))]
extern "system" {
    fn BCryptGenRandom(alg: usize, buf: *mut u8, len: u32, flags: u32) -> i32;
}

#[link(name = "kernel32", cfg(windows))]
extern "system" {
    fn lowercase_k32_sym() -> u32;
}

#[link(name = "ucrtbase")]
extern "C" {
    fn UppercaseUcrtSym() -> i32;
}

fn main() {
    println!("hello windows");
    let _ = std::net::TcpStream::connect("127.0.0.1:0");
    let ptrs = [
        MessageBoxW as usize,
        RegCloseKey as usize,
        BCryptGenRandom as usize,
        lowercase_k32_sym as usize,
        UppercaseUcrtSym as usize,
    ];
    let mut acc = 0usize;
    for p in ptrs {
        acc ^= p;
    }
    std::hint::black_box(acc);
}
"#,
    )
    .expect("write main.rs");

    let out_bin = dir.join("pe-link-e2e.exe");
    let compiler = crate::Compiler::builder()
        .target("x86_64-pc-windows-msvc")
        .jobs(2)
        .build()
        .expect("build windows cross compiler");
    let emit_res = compiler.emit(&dir, &out_bin);
    let bytes = emit_res
        .as_ref()
        .ok()
        .map(|_| std::fs::read(&out_bin).expect("read PE32+"));
    drop(std::fs::remove_dir_all(&dir));
    emit_res.expect("emit Windows PE32+");
    let bytes = bytes.expect("PE32+ bytes");
    let dll_imports = parse_pe_idata(&bytes);

    let assert_exclusive = |expected_dll: &str, sym: &str| {
        let (_, syms) = dll_imports
            .iter()
            .find(|(dll, _)| dll == expected_dll)
            .unwrap_or_else(|| panic!("missing {expected_dll} in {dll_imports:?}"));
        assert!(
            syms.iter().any(|s| s == sym),
            "missing {sym} under {expected_dll} in {dll_imports:?}"
        );
        for (other_dll, other_syms) in &dll_imports {
            if other_dll != expected_dll {
                assert!(
                    !other_syms.iter().any(|s| s == sym),
                    "symbol {sym} leaked into wrong DLL {other_dll}"
                );
            }
        }
    };

    for (dll, sym) in [
        ("user32.dll", "MessageBoxW"),
        ("advapi32.dll", "RegCloseKey"),
        ("bcrypt.dll", "BCryptGenRandom"),
        ("kernel32.dll", "lowercase_k32_sym"),
        ("kernel32.dll", "ExitProcess"),
        ("ucrtbase.dll", "UppercaseUcrtSym"),
        ("ntdll.dll", "NtWriteFile"),
        ("ws2_32.dll", "WSAStartup"),
    ] {
        assert_exclusive(dll, sym);
    }
}

#[test]
fn pe_shims_patch_rip_relative_into_data_and_route_msvcrt_imports() {
    for target in TARGETS.iter().filter(|t| t.format == Format::Pe) {
        let mut exe = minimal_executable_with_imports(target, &[("atexit", None)], false);
        let (kind, call_off) = match target.arch {
            Arch::X64 => (RelocKind::PcRel32, 1u32),
            Arch::Arm64 => (RelocKind::Arm64Call26, 0u32),
        };
        let stub: &[u8] = match target.arch {
            Arch::X64 => &[0xE8, 0, 0, 0, 0, 0xC3],
            Arch::Arm64 => &[0x00, 0x00, 0x00, 0x94, 0xC0, 0x03, 0x5F, 0xD6],
        };
        let mut code = Vec::new();
        let mut sites = Vec::new();
        for _ in 0..3 {
            sites.push(code.len() as u32 + call_off);
            code.extend_from_slice(stub);
        }
        let caller = exe.function(&code, 16, &[], &[]);
        exe.patch(
            caller + sites[0],
            kind,
            Symbol::Host(HostSymbol::ThreadCell),
        );
        exe.patch(
            caller + sites[1],
            kind,
            Symbol::Host(HostSymbol::CatchUnwind),
        );
        exe.patch(
            caller + sites[2],
            kind,
            Symbol::Host(HostSymbol::StartPanic),
        );
        let bytes = exe.encode().expect("emit PE32+ with shims");

        let dll_imports = parse_pe_idata(&bytes);
        let msvcrt = dll_imports
            .iter()
            .find(|(dll, _)| dll == "msvcrt.dll")
            .unwrap_or_else(|| panic!("{} missing msvcrt.dll in {dll_imports:?}", target.triple));
        let setjmp_sym = if target.arch == Arch::Arm64 {
            "_setjmpex"
        } else {
            "_setjmp"
        };
        for sym in [setjmp_sym, "longjmp", "atexit"] {
            assert!(
                msvcrt.1.iter().any(|s| s == sym),
                "{} expected {sym} in msvcrt.dll, got {dll_imports:?}",
                target.triple
            );
            assert!(
                !dll_imports
                    .iter()
                    .any(|(d, syms)| d != "msvcrt.dll" && syms.iter().any(|s| s == sym)),
                "{} leaked {sym} outside msvcrt.dll",
                target.triple
            );
        }

        let pe_off = read_u32_at(&bytes, 0x3C) as usize;
        let nsec = read_u16(&bytes, pe_off + 6) as usize;
        let opt_size = read_u16(&bytes, pe_off + 20) as usize;
        let opt_off = pe_off + 24;
        let image_base = read_u64(&bytes, opt_off + 24);
        let sec_base = opt_off + opt_size;
        let mut text = (0u32, 0usize, 0usize);
        let mut data = (0u32, 0u32);
        for i in 0..nsec {
            let s = sec_base + i * 40;
            match read_cstr(&bytes, s) {
                ".text" => {
                    text = (
                        read_u32_at(&bytes, s + 12),
                        read_u32_at(&bytes, s + 20) as usize,
                        read_u32_at(&bytes, s + 8) as usize,
                    );
                }
                ".data" => data = (read_u32_at(&bytes, s + 12), read_u32_at(&bytes, s + 8)),
                _ => {}
            }
        }
        let data_start = image_base + u64::from(data.0);
        let data_end = data_start + u64::from(data.1);
        let text_bytes = &bytes[text.1..text.1 + text.2];
        if target.arch == Arch::X64 {
            let lea_r13 = text_bytes
                .windows(3)
                .position(|w| w == [0x4C, 0x8D, 0x2D])
                .expect("ThreadCell lea r13, [rip + disp32]");
            let disp = i32::from_le_bytes(
                text_bytes[lea_r13 + 3..lea_r13 + 7]
                    .try_into()
                    .expect("disp32"),
            );
            let target_va = (image_base + u64::from(text.0) + (lea_r13 + 7) as u64)
                .wrapping_add_signed(i64::from(disp));
            assert!(
                (data_start..data_end).contains(&target_va),
                "x64 ThreadCell thread_key VA 0x{target_va:x} outside .data [0x{data_start:x}..0x{data_end:x})"
            );
        } else {
            let words: Vec<u32> = text_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("word")))
                .collect();
            let at = words
                .windows(2)
                .position(|pair| {
                    (pair[0] & 0x9F00_001F) == 0x9000_0015 && (pair[1] & 0xFFC0_03FF) == 0x9100_02B5
                })
                .expect("ARM64 ThreadCell adrp x21 + add x21, x21, #imm12");
            let adrp = words[at];
            let add = words[at + 1];
            let imm_lo = (adrp >> 29) & 3;
            let imm_hi = (adrp >> 5) & 0x7_FFFF;
            let imm21 = (((imm_hi << 2) | imm_lo) as i32) << 11 >> 11;
            let pc_va = image_base + u64::from(text.0) + (at * 4) as u64;
            let page = (pc_va & !0xFFF).wrapping_add_signed(i64::from(imm21) << 12);
            let target_va = page + u64::from((add >> 10) & 0xFFF);
            assert!(
                (data_start..data_end).contains(&target_va),
                "arm64 ThreadCell thread_key VA 0x{target_va:x} outside .data [0x{data_start:x}..0x{data_end:x})"
            );
        }
    }
}

fn read_uleb(bytes: &[u8], pos: &mut usize) -> u64 {
    let (mut val, mut shift) = (0u64, 0u32);
    loop {
        let b = bytes[*pos];
        *pos += 1;
        val |= u64::from(b & 0x7F) << shift;
        if b & 0x80 == 0 {
            return val;
        }
        shift += 7;
    }
}

fn decode_func_starts(bytes: &[u8]) -> Vec<u64> {
    let (mut pos, mut cur, mut out) = (0usize, 0u64, Vec::new());
    while pos < bytes.len() {
        let delta = read_uleb(bytes, &mut pos);
        if delta == 0 {
            break;
        }
        cur += delta;
        out.push(cur);
    }
    out
}

mod exports;
mod macho_framework;
