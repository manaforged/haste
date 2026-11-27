use super::super::plan::Plan;
use super::super::shims::{Shim, emit_shims};
use super::super::{Buf, SectionAddrs};
use super::*;

fn start_panic_exe(target: &Target, panic_abort: bool, with_unwind_sites: bool) -> Executable {
    let mut exe = Executable::new(target);
    exe.panic_abort = panic_abort;
    let (stub, call_off, kind) = match target.arch {
        Arch::X64 => (&[0xE8u8, 0, 0, 0, 0, 0xC3][..], 1u32, RelocKind::PcRel32),
        Arch::Arm64 => (
            &[0x00u8, 0x00, 0x00, 0x94, 0xC0, 0x03, 0x5F, 0xD6][..],
            0u32,
            RelocKind::Arm64Call26,
        ),
    };
    let lang_start = exe.function(stub, 16, &[], &[]);
    let mut syms = vec![HostSymbol::StartPanic];
    if with_unwind_sites {
        syms.extend([
            HostSymbol::CatchUnwind,
            HostSymbol::DropPush,
            HostSymbol::DropPop,
        ]);
    }
    let mut code = Vec::with_capacity(stub.len() * syms.len());
    let mut sites = Vec::with_capacity(syms.len());
    for _ in &syms {
        sites.push(code.len() as u32 + call_off);
        code.extend_from_slice(stub);
    }
    let caller = exe.function(&code, 16, &[], &[]);
    exe.entry(lang_start, caller);
    for (site, sym) in sites.into_iter().zip(syms) {
        exe.patch(caller + site, kind, Symbol::Host(sym));
    }
    exe
}

#[test]
fn panic_abort_imports_abort_and_skips_unwind_walk_for_arm64_and_x64() {
    for target in TARGETS {
        let exe_abort = start_panic_exe(target, true, false);
        let plan_abort = Plan::collect(&exe_abort, &[]).expect("collect panic_abort plan");
        assert_eq!(
            plan_abort.shims,
            [Shim::StartPanicAbort],
            "{}: expected only StartPanicAbort shim",
            target.triple
        );
        assert!(!plan_abort.has_shim(Shim::StartPanic));
        assert!(!plan_abort.has_shim(Shim::UnwindWalk));
        assert!(!plan_abort.has_unwind());
        assert!(
            plan_abort.imports.iter().any(|imp| imp.name == "abort"),
            "{}: missing abort import in {:?}",
            target.triple,
            plan_abort.imports
        );
        for forbidden in [
            "pthread_key_create",
            "pthread_getspecific",
            "TlsAlloc",
            "TlsGetValue",
            "longjmp",
        ] {
            assert!(
                plan_abort.imports.iter().all(|imp| imp.name != forbidden),
                "{}: unexpected unwind import {forbidden}",
                target.triple
            );
        }

        let exe_with_drops = start_panic_exe(target, true, true);
        let plan_with_drops =
            Plan::collect(&exe_with_drops, &[]).expect("collect panic_abort plan with drop sites");
        assert!(plan_with_drops.has_shim(Shim::StartPanicAbort));
        assert!(!plan_with_drops.has_shim(Shim::StartPanic));
        assert!(!plan_with_drops.has_shim(Shim::UnwindWalk));

        let exe_unwind = start_panic_exe(target, false, false);
        let plan_unwind = Plan::collect(&exe_unwind, &[]).expect("collect panic_unwind plan");
        assert!(plan_unwind.has_shim(Shim::StartPanic));
        assert!(!plan_unwind.has_shim(Shim::StartPanicAbort));
        assert!(plan_unwind.has_shim(Shim::UnwindWalk));
        assert!(plan_unwind.has_unwind());
    }
}

#[test]
fn panic_abort_emits_direct_tail_branch_to_abort_on_arm64_and_x64() {
    for target in TARGETS {
        let exe_abort = start_panic_exe(target, true, false);
        let plan_abort = Plan::collect(&exe_abort, &[]).expect("collect panic_abort plan");
        let mut buf = Buf::new();
        let stub_va = |name: &str| {
            assert_eq!(
                name, "abort",
                "{}: unexpected shim import {name}",
                target.triple
            );
            Ok(0x1_1000)
        };
        emit_shims(
            target.arch,
            target.os,
            &plan_abort,
            0x1_0000,
            0,
            SectionAddrs::default(),
            &stub_va,
            None,
            &mut buf,
        )
        .expect("emit panic_abort shim");
        match target.arch {
            Arch::Arm64 => assert_eq!(
                buf.0,
                0x1400_0400_u32.to_le_bytes(),
                "{}: expected direct `b abort` without unwind walk",
                target.triple
            ),
            Arch::X64 => {
                let mut expected = vec![0xE9u8];
                expected.extend_from_slice(&0x0FFB_i32.to_le_bytes());
                assert_eq!(
                    buf.0, expected,
                    "{}: expected direct `jmp abort` without unwind walk",
                    target.triple
                );
            }
        }

        let bytes = exe_abort.encode().expect("encode panic_abort executable");
        assert!(!bytes.is_empty());
        if target.format == Format::Pe {
            let imports = parse_pe_idata(&bytes);
            assert!(
                imports
                    .iter()
                    .any(|(dll, syms)| dll == "ucrtbase.dll" && syms.iter().any(|s| s == "abort")),
                "{}: expected ucrtbase!abort in PE imports {imports:?}",
                target.triple
            );
            assert!(
                !imports
                    .iter()
                    .any(|(_, syms)| syms.iter().any(|s| s == "longjmp")),
                "{}: unexpected longjmp import in panic_abort PE {imports:?}",
                target.triple
            );
        }
    }
}
