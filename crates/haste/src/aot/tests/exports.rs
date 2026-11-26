use haste_codegen::{Arch, Format, TARGETS};

use super::super::ExeSymbol;
use super::{
    elf_sections, minimal_executable, parse_macho, read_cstr, read_u16, read_u32_at, read_u64,
    read_uleb,
};

fn trie_lookup(trie: &[u8], mut sym: &str) -> Option<u64> {
    let mut off = 0usize;
    loop {
        let mut pos = off;
        let term = read_uleb(trie, &mut pos) as usize;
        if sym.is_empty() {
            return (term > 0).then(|| {
                let _ = read_uleb(trie, &mut pos);
                read_uleb(trie, &mut pos)
            });
        }
        pos += term + 1;
        let children = trie[pos - 1];
        off = (0..children).find_map(|_| {
            let label = read_cstr(trie, pos);
            pos += label.len() + 1;
            let child = read_uleb(trie, &mut pos) as usize;
            sym.strip_prefix(label).map(|rest| {
                sym = rest;
                child
            })
        })?;
    }
}

#[test]
fn macho_exports_no_mangle_in_symtab_dysymtab_export_trie_and_synth_main() {
    for target in TARGETS.iter().filter(|t| t.format == Format::MachO) {
        let cpu = if target.arch == Arch::Arm64 {
            0x0100_000C
        } else {
            0x0100_0007
        };
        let mut exe = minimal_executable(target, "puts", false);
        let (cb_off, alt_off) = (exe.symbols[0].offset, exe.symbols[1].offset);
        exe.symbols(vec![
            ExeSymbol {
                offset: cb_off,
                name: "rust_cb".into(),
                exported: true,
            },
            ExeSymbol {
                offset: alt_off,
                name: "rust_cb_alt".into(),
                exported: true,
            },
        ]);
        let bytes = exe.encode().expect("emit Mach-O exports");
        assert_eq!(bytes, exe.encode().expect("deterministic UUID"));
        let view = parse_macho(&bytes, cpu);
        let (symoff, nsyms, stroff, _) = view.symtab;
        assert_eq!((nsyms, view.dysymtab.0), (3, [0, 0, 0, 2, 2, 1]));
        let (ind_off, ind_len) = (view.dysymtab.1, view.dysymtab.2);
        assert_eq!(
            (0..ind_len)
                .map(|i| read_u32_at(&bytes, ind_off + i * 4))
                .collect::<Vec<_>>(),
            vec![2, 2]
        );
        let (n0, n1) = (symoff, symoff + 16);
        assert_eq!(
            (
                bytes[n0 + 4],
                bytes[n0 + 5],
                read_cstr(&bytes, stroff + read_u32_at(&bytes, n0) as usize)
            ),
            (0x0F, 1, "_rust_cb")
        );
        assert_eq!(
            (
                bytes[n1 + 4],
                bytes[n1 + 5],
                read_cstr(&bytes, stroff + read_u32_at(&bytes, n1) as usize)
            ),
            (0x0F, 1, "_rust_cb_alt")
        );
        let (v0, v1) = (read_u64(&bytes, n0 + 8), read_u64(&bytes, n1 + 8));
        assert_eq!(v1 - v0, u64::from(alt_off - cb_off));
        let trie = &bytes[view.dyld_info.2..view.dyld_info.2 + view.dyld_info.3];
        assert_eq!(
            (
                trie_lookup(trie, "_rust_cb"),
                trie_lookup(trie, "_rust_cb_alt")
            ),
            (Some(v0 - 0x1_0000_0000), Some(v1 - 0x1_0000_0000))
        );
        exe.symbols(vec![ExeSymbol {
            offset: cb_off,
            name: "p_dbg::main".into(),
            exported: false,
        }]);
        let dbg_bytes = exe.encode().expect("emit Mach-O with synth _main");
        let dbg = parse_macho(&dbg_bytes, cpu);
        let found_main = (0..dbg.symtab.1).find_map(|i| {
            let e = dbg.symtab.0 + i * 16;
            (read_cstr(
                &dbg_bytes,
                dbg.symtab.2 + read_u32_at(&dbg_bytes, e) as usize,
            ) == "_main")
                .then(|| (dbg_bytes[e + 4], read_u64(&dbg_bytes, e + 8)))
        });
        assert_eq!(found_main, Some((0x0E, 0x1_0000_0000 + dbg.entry.2 as u64)));
    }
}

#[test]
fn elf_exports_global_dynsym_and_hash_lookup() {
    for target in TARGETS.iter().filter(|t| t.format == Format::Elf) {
        let mut exe = minimal_executable(target, "puts", false);
        let cb_off = exe.symbols[1].offset;
        exe.symbols(vec![ExeSymbol {
            offset: cb_off,
            name: "rust_cb".into(),
            exported: true,
        }]);
        let bytes = exe.encode().expect("emit ELF exports");
        let sections = elf_sections(&bytes);
        let (_, _, dyn_off, dyn_len) = sections
            .iter()
            .find(|(n, ..)| n == ".dynamic")
            .expect(".dynamic");
        let (_, dynstr_va, dynstr_off, _) = sections
            .iter()
            .find(|(n, ..)| n == ".dynstr")
            .expect(".dynstr");
        let (mut hash_va, mut dynsym_va) = (0u64, 0u64);
        for pos in (*dyn_off..dyn_off + dyn_len).step_by(16) {
            match read_u64(&bytes, pos) {
                4 => hash_va = read_u64(&bytes, pos + 8),
                6 => dynsym_va = read_u64(&bytes, pos + 8),
                _ => {}
            }
        }
        let ro_base = dynstr_va - *dynstr_off as u64;
        let (hash_off, dynsym_off) = ((hash_va - ro_base) as usize, (dynsym_va - ro_base) as usize);
        let (nbucket, nchain) = (
            read_u32_at(&bytes, hash_off) as usize,
            read_u32_at(&bytes, hash_off + 4) as usize,
        );
        assert_eq!(nchain, 4);
        let mut idx = read_u32_at(
            &bytes,
            hash_off + 8 + (super::super::elf::elf_hash(b"rust_cb") as usize % nbucket) * 4,
        ) as usize;
        let mut matched = None;
        while idx != 0 {
            let s = dynsym_off + idx * 24;
            if read_cstr(&bytes, dynstr_off + read_u32_at(&bytes, s) as usize) == "rust_cb" {
                matched = Some((
                    bytes[s + 4],
                    read_u16(&bytes, s + 6),
                    read_u64(&bytes, s + 8),
                ));
                break;
            }
            idx = read_u32_at(&bytes, hash_off + 8 + nbucket * 4 + idx * 4) as usize;
        }
        let (_, text_va, _, text_sz) = sections.iter().find(|(n, ..)| n == ".text").expect(".text");
        let (info, shndx, val) = matched.expect("rust_cb in DT_HASH/.dynsym");
        assert_eq!((info, shndx), (0x12, 1));
        assert!(val >= *text_va && val < *text_va + *text_sz as u64);
    }
}
