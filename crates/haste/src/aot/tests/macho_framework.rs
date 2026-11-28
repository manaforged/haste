use super::*;

fn macho_bind_slice(bytes: &[u8]) -> &[u8] {
    let ncmds = read_u32_at(bytes, 16) as usize;
    let mut cursor = 32usize;
    for _ in 0..ncmds {
        let cmd = read_u32_at(bytes, cursor);
        let cmdsize = read_u32_at(bytes, cursor + 4) as usize;
        if cmd == 0x8000_0022 {
            let bind_off = read_u32_at(bytes, cursor + 16) as usize;
            let bind_len = read_u32_at(bytes, cursor + 20) as usize;
            return &bytes[bind_off..bind_off + bind_len];
        }
        cursor += cmdsize;
    }
    panic!("missing LC_DYLD_INFO_ONLY");
}

fn decode_dyld_binds(bind: &[u8]) -> Vec<(i64, String, u64)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut dylib_ord = 0i64;
    let mut sym_name = String::new();
    let mut offset = 0u64;
    while pos < bind.len() {
        let b = bind[pos];
        pos += 1;
        let opcode = b & 0xF0;
        let imm = b & 0x0F;
        match opcode {
            0x00 => break,
            0x10 => dylib_ord = i64::from(imm),
            0x20 => dylib_ord = read_uleb(bind, &mut pos) as i64,
            0x30 => {
                let signed = ((imm as i8) << 4) >> 4;
                dylib_ord = i64::from(signed);
            }
            0x40 => {
                let start = pos;
                while bind[pos] != 0 {
                    pos += 1;
                }
                sym_name = std::str::from_utf8(&bind[start..pos])
                    .expect("utf8 sym")
                    .to_owned();
                pos += 1;
            }
            0x50 => {}
            0x70 => offset = read_uleb(bind, &mut pos),
            0x90 => {
                out.push((dylib_ord, sym_name.clone(), offset));
                offset += 8;
            }
            other => panic!("unexpected dyld bind opcode 0x{other:02x}"),
        }
    }
    out
}

#[test]
fn macho_routes_frameworks_to_system_load_dylib_and_twolevel_ordinals() {
    for target in TARGETS.iter().filter(|t| t.format == Format::MachO) {
        let cpu = if target.arch == Arch::Arm64 {
            0x0100_000C
        } else {
            0x0100_0007
        };
        let bytes = minimal_executable_with_imports(
            target,
            &[
                ("puts", None),
                ("CFStringGetCStringPtr", Some("CoreFoundation")),
                ("CFTimeZoneCopySystem", Some("framework=CoreFoundation")),
                ("SecKeychainCopyDefault", Some("Security")),
            ],
            false,
        )
        .encode()
        .expect("emit Mach-O with frameworks");
        let view = parse_macho(&bytes, cpu);
        assert_eq!(
            view.dylibs,
            vec![
                "/usr/lib/libSystem.B.dylib",
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
                "/System/Library/Frameworks/Security.framework/Security",
            ]
        );
        let flags = read_u32_at(&bytes, 24);
        assert_ne!(
            flags & 0x80,
            0,
            "expected MH_TWOLEVEL for framework imports"
        );
        let binds = decode_dyld_binds(macho_bind_slice(&bytes));
        assert_eq!(
            binds,
            vec![
                (1, "_puts".to_owned(), 0),
                (2, "_CFStringGetCStringPtr".to_owned(), 8),
                (2, "_CFTimeZoneCopySystem".to_owned(), 16),
                (3, "_SecKeychainCopyDefault".to_owned(), 24),
            ]
        );
    }
}

#[test]
fn macho_extra_lib_native_bundle_falls_back_only_for_untagged_imports() {
    for target in TARGETS.iter().filter(|t| t.format == Format::MachO) {
        let cpu = if target.arch == Arch::Arm64 {
            0x0100_000C
        } else {
            0x0100_0007
        };
        let mut exe = minimal_executable_with_imports(
            target,
            &[
                ("custom_c_helper", None),
                ("CFTimeZoneGetName", Some("CoreFoundation")),
            ],
            false,
        );
        exe.extra_libs.push("/tmp/native/deadbeef.so".to_owned());
        let bytes = exe.encode().expect("emit Mach-O");
        let view = parse_macho(&bytes, cpu);
        assert_eq!(
            view.dylibs,
            vec![
                "/usr/lib/libSystem.B.dylib",
                "/tmp/native/deadbeef.so",
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
            ]
        );
        let flags = read_u32_at(&bytes, 24);
        assert_eq!(flags & 0x80, 0);
        let binds = decode_dyld_binds(macho_bind_slice(&bytes));
        assert_eq!(
            binds,
            vec![
                (-2, "_custom_c_helper".to_owned(), 0),
                (3, "_CFTimeZoneGetName".to_owned(), 8),
            ]
        );
    }
}
