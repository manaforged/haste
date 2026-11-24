use std::path::PathBuf;

use haste_codegen::{Arch, Format, RelocKind, TARGETS, Target};

use super::{Executable, Symbol};
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

fn minimal_executable(target: &Target, import: &str, with_debug_line: bool) -> Executable {
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
    let user_main = exe.function(&stub, 16, &[], &lines);
    let ro = exe.rodata(b"hello\0", 8);
    let data = exe.data(&[0u8; 8], 8);
    let _bss = exe.bss(16, 8);
    exe.entry(lang_start, user_main);
    exe.patch(
        user_main + call_off,
        kind,
        Symbol::Host(HostSymbol::Dynamic(import.into())),
    );
    exe.pointer(false, data, Symbol::Rodata(ro));
    exe.symbols(vec![
        (lang_start, "lang_start".to_owned()),
        (user_main, "main".to_owned()),
    ]);
    exe
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

#[test]
fn macho_structure_relocs_load_commands_entry_and_symtab_imports() {
    for target in TARGETS.iter().filter(|t| t.format == Format::MachO) {
        let expected_cpu = match target.arch {
            Arch::Arm64 => 0x0100_000Cu32,
            Arch::X64 => 0x0100_0007u32,
        };
        let bytes = minimal_executable(target, "puts", false)
            .encode()
            .expect("emit Mach-O");
        assert_eq!(read_u32_at(&bytes, 0), 0xFEED_FACF);
        assert_eq!(read_u32_at(&bytes, 4), expected_cpu);
        assert_eq!(read_u32_at(&bytes, 12), 2);
        let ncmds = read_u32_at(&bytes, 16) as usize;
        let sizeofcmds = read_u32_at(&bytes, 20) as usize;

        let mut cursor = 32usize;
        let mut cmds = Vec::with_capacity(ncmds);
        let mut text_file_range = None;
        let mut data_off = None;
        let mut ro_va = None;
        let mut entry_off = None;
        let mut symtab = None;
        let mut dyld_info = None;
        let mut dylibs = Vec::new();
        for _ in 0..ncmds {
            let cmd = read_u32_at(&bytes, cursor);
            let cmdsize = read_u32_at(&bytes, cursor + 4) as usize;
            match cmd {
                0x19 => {
                    let segname = read_cstr(&bytes, cursor + 8);
                    let fileoff = read_u64(&bytes, cursor + 40) as usize;
                    let filesize = read_u64(&bytes, cursor + 48) as usize;
                    assert!(fileoff + filesize <= bytes.len());
                    if segname == "__TEXT" {
                        text_file_range = Some(fileoff..fileoff + filesize);
                    }
                    let nsects = read_u32_at(&bytes, cursor + 64) as usize;
                    for s in 0..nsects {
                        let sh = cursor + 72 + s * 80;
                        let sname = read_cstr(&bytes, sh);
                        let saddr = read_u64(&bytes, sh + 32);
                        let soff = read_u32_at(&bytes, sh + 48) as usize;
                        if sname == "__data" {
                            data_off = Some(soff);
                        } else if sname == "__const" {
                            ro_va = Some(saddr);
                        }
                    }
                }
                0x8000_0022 => {
                    let rebase_size = read_u32_at(&bytes, cursor + 12);
                    let bind_size = read_u32_at(&bytes, cursor + 20);
                    dyld_info = Some((rebase_size, bind_size));
                }
                0x8000_0028 => entry_off = Some(read_u64(&bytes, cursor + 8) as usize),
                2 => {
                    let stroff = read_u32_at(&bytes, cursor + 16) as usize;
                    let strsize = read_u32_at(&bytes, cursor + 20) as usize;
                    symtab = Some((stroff, strsize));
                }
                12 => {
                    let name_off = read_u32_at(&bytes, cursor + 8) as usize;
                    dylibs.push(read_cstr(&bytes, cursor + name_off).to_owned());
                }
                _ => {}
            }
            cmds.push(cmd);
            cursor += cmdsize;
        }
        assert_eq!(cursor - 32, sizeofcmds);
        let text_range = text_file_range.expect("__TEXT segment");
        let entry = entry_off.expect("LC_MAIN entryoff");
        assert!(text_range.contains(&entry));
        assert_eq!(
            read_u64(&bytes, data_off.expect("__data section")),
            ro_va.expect("__const section"),
            "{} __data pointer relocation must target __const VA",
            target.triple
        );
        let (rebase_size, bind_size) = dyld_info.expect("LC_DYLD_INFO_ONLY");
        assert!(rebase_size > 0 && bind_size > 0);
        assert!(cmds.contains(&11) && cmds.contains(&14));
        assert!(dylibs.iter().any(|d| d == "/usr/lib/libSystem.B.dylib"));
        let (stroff, strsize) = symtab.expect("LC_SYMTAB");
        let strtab = &bytes[stroff..stroff + strsize];
        assert!(strtab.windows(6).any(|w| w == b"_puts\0"));
    }
}

#[test]
fn pe_structure_relocs_sections_entry_and_idata_import_directory() {
    for target in TARGETS.iter().filter(|t| t.format == Format::Pe) {
        let expected_machine = match target.arch {
            Arch::X64 => 0x8664u16,
            Arch::Arm64 => 0xAA64u16,
        };
        let bytes = minimal_executable(target, "GetLastError", false)
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
        let import_dir_rva = read_u32_at(&bytes, opt_off + 112 + 8);

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
        let idata_sec = sections
            .iter()
            .find(|(n, ..)| n == ".idata")
            .expect(".idata");
        let rva_to_off = |rva: u32| idata_sec.3 + (rva - idata_sec.1) as usize;

        let mut desc = rva_to_off(import_dir_rva);
        let mut dll_imports: Vec<(String, Vec<String>)> = Vec::new();
        loop {
            let ilt_rva = read_u32_at(&bytes, desc);
            let name_rva = read_u32_at(&bytes, desc + 12);
            if ilt_rva == 0 && name_rva == 0 {
                break;
            }
            let dll = read_cstr(&bytes, rva_to_off(name_rva)).to_owned();
            let mut syms = Vec::new();
            let mut thunk = rva_to_off(ilt_rva);
            loop {
                let entry_val = read_u64(&bytes, thunk);
                if entry_val == 0 {
                    break;
                }
                syms.push(read_cstr(&bytes, rva_to_off(entry_val as u32) + 2).to_owned());
                thunk += 8;
            }
            dll_imports.push((dll, syms));
            desc += 20;
        }
        let (_, k32_syms) = dll_imports
            .iter()
            .find(|(dll, _)| dll == "kernel32.dll")
            .expect("kernel32.dll descriptor");
        assert!(k32_syms.iter().any(|s| s == "ExitProcess"));
        assert!(k32_syms.iter().any(|s| s == "GetLastError"));
    }
}
