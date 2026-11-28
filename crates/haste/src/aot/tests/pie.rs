use haste_codegen::{Arch, Format, RelocKind, TARGETS, Target};

use super::{
    Executable, Symbol, elf_sections, minimal_executable, read_u16, read_u32_at, read_u64,
};

#[test]
fn elf_pie_et_dyn_relro_flags_and_relative_relocs() {
    for target in TARGETS.iter().filter(|t| t.format == Format::Elf) {
        let mut exe = minimal_executable(target, "puts", false);
        let ro_ptr = exe.rodata(&[0u8; 8], 8);
        let word_slot = exe.data(&[0u8; 8], 8);
        let main_off = exe.symbols[1].offset;
        exe.pointer(true, ro_ptr, Symbol::Text(main_off));
        exe.pointer(false, word_slot, Symbol::Word(0x1234_5678));

        let bytes = exe.encode().expect("emit PIE ELF");
        assert_eq!(
            read_u16(&bytes, 16),
            3,
            "{}: e_type must be ET_DYN",
            target.triple
        );

        let phoff = read_u64(&bytes, 32) as usize;
        let phentsize = read_u16(&bytes, 54) as usize;
        let phnum = read_u16(&bytes, 56) as usize;
        let page = if target.arch == Arch::Arm64 {
            0x1_0000u64
        } else {
            0x1000u64
        };

        let mut relro = None;
        for i in 0..phnum {
            let p = phoff + i * phentsize;
            let p_type = read_u32_at(&bytes, p);
            if p_type == 0x6474_E552 {
                let p_vaddr = read_u64(&bytes, p + 16);
                let p_memsz = read_u64(&bytes, p + 40);
                relro = Some((p_vaddr, p_memsz));
            }
        }
        let (relro_va, relro_sz) = relro.expect("PT_GNU_RELRO header");
        assert_eq!(
            (relro_va + relro_sz) % page,
            0,
            "{}: PT_GNU_RELRO end must be page-aligned",
            target.triple
        );

        let sections = elf_sections(&bytes);
        let (_, text_va, _, _) = sections.iter().find(|(n, ..)| n == ".text").expect(".text");
        let (_, ro_va, ro_off, _) = sections
            .iter()
            .find(|(n, ..)| n == ".rodata")
            .expect(".rodata");
        let (_, data_va, _, _) = sections.iter().find(|(n, ..)| n == ".data").expect(".data");
        let (_, dyn_va, dyn_off, dyn_len) = sections
            .iter()
            .find(|(n, ..)| n == ".dynamic")
            .expect(".dynamic");

        assert!(
            *ro_va >= relro_va && *dyn_va < relro_va + relro_sz,
            "{}: RELRO must cover .rodata and .dynamic",
            target.triple
        );
        assert!(
            *data_va >= relro_va + relro_sz,
            "{}: writable .data must start at or after RELRO end",
            target.triple
        );

        let (mut rela_va, mut rela_sz, mut dt_flags, mut dt_flags_1) = (0u64, 0u64, 0u64, 0u64);
        for pos in (*dyn_off..dyn_off + dyn_len).step_by(16) {
            let tag = read_u64(&bytes, pos);
            let val = read_u64(&bytes, pos + 8);
            match tag {
                7 => rela_va = val,
                8 => rela_sz = val,
                30 => dt_flags = val,
                0x6FFF_FFFB => dt_flags_1 = val,
                _ => {}
            }
        }
        assert_ne!(
            dt_flags & 0x8,
            0,
            "{}: DT_FLAGS must set DF_BIND_NOW",
            target.triple
        );
        assert_eq!(
            dt_flags_1 & 0x0800_0001,
            0x0800_0001,
            "{}: DT_FLAGS_1 must set DF_1_PIE | DF_1_NOW",
            target.triple
        );

        let want_rel_type = if target.arch == Arch::X64 {
            8u64
        } else {
            1027u64
        };
        let mut relatives = Vec::new();
        for pos in (rela_va as usize..(rela_va + rela_sz) as usize).step_by(24) {
            let r_off = read_u64(&bytes, pos);
            let r_info = read_u64(&bytes, pos + 8);
            let r_addend = read_u64(&bytes, pos + 16);
            if (r_info & 0xFFFF_FFFF) == want_rel_type {
                relatives.push((r_off, r_addend));
            }
        }
        assert_eq!(
            relatives.len(),
            2,
            "{}: expected 2 R_*_RELATIVE entries (.data + .rodata, skipping Symbol::Word), got {relatives:?}",
            target.triple
        );
        assert!(
            relatives.contains(&(*data_va, *ro_va)),
            "{}: missing R_*_RELATIVE for .data -> .rodata",
            target.triple
        );
        let ro_ptr_va = *ro_va + u64::from(ro_ptr);
        let ro_ptr_val = read_u64(&bytes, *ro_off + ro_ptr as usize);
        assert!(
            relatives.contains(&(ro_ptr_va, ro_ptr_val)) && ro_ptr_val > *text_va,
            "{}: missing R_*_RELATIVE for .rodata -> .text",
            target.triple
        );
    }
}

#[test]
fn elf_x64_pie_rewrites_abs64_in_image_symbol_to_rip_lea() {
    let target = Target::find("x86_64-unknown-linux-gnu").expect("x64 linux");
    let mut exe = Executable::new(target);
    let stub = [0xC3u8];
    let lang_start = exe.function(&stub, 16, &[], &[]);
    let movabs = [0x48u8, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xC3];
    let user_main = exe.function(&movabs, 16, &[], &[]);
    let ro = exe.rodata(b"pie\0", 8);
    exe.entry(lang_start, user_main);
    exe.patch(user_main + 2, RelocKind::Abs64, Symbol::Rodata(ro));

    let bytes = exe.encode().expect("emit x64 PIE");
    let sections = elf_sections(&bytes);
    let (_, _, text_off, text_sz) = sections.iter().find(|(n, ..)| n == ".text").expect(".text");
    let text = &bytes[*text_off..text_off + text_sz];
    assert!(
        text.windows(3).any(|w| w == [0x48, 0x8D, 0x05]),
        "expected Abs64 in-image symbol to be rewritten to RIP-relative lea rax, [rip+disp32]"
    );
}
