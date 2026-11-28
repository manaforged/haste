use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::build::Build;
use crate::graph::{self, Targets};

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fresh_dir() -> TempDir {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("haste_refresh_{}_{id}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    TempDir(dir)
}

fn host_targets() -> Targets {
    let host = crate::graph::cfg::Target::of(&crate::host_target());
    Targets {
        target: host.clone(),
        host,
    }
}

fn load_workspace(root: &Path, targets: &Targets, sel: &crate::Selection) -> graph::Workspace {
    let mut ws = graph::load(root, targets, sel).expect("load workspace");
    let lib = graph::library().expect("sysroot library");
    ws.merge(graph::load_sysroot(&lib, targets).expect("load sysroot"));
    ws
}

fn write_two_crate_workspace(root: &Path) -> PathBuf {
    let core_dir = root.join("crates/core_lib/src");
    let app_dir = root.join("crates/app_bin/src");
    std::fs::create_dir_all(&core_dir).expect("mkdir core_lib");
    std::fs::create_dir_all(&app_dir).expect("mkdir app_bin");
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/core_lib\", \"crates/app_bin\"]\nresolver = \"2\"\n",
    )
    .expect("write workspace Cargo.toml");
    std::fs::write(
        root.join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"core_lib\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"app_bin\"\nversion = \"0.1.0\"\ndependencies = [\"core_lib\"]\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(
        root.join("crates/core_lib/Cargo.toml"),
        "[package]\nname = \"core_lib\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write core_lib Cargo.toml");
    std::fs::write(
        root.join("crates/app_bin/Cargo.toml"),
        "[package]\nname = \"app_bin\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[dependencies]\ncore_lib = { path = \"../core_lib\" }\n",
    )
    .expect("write app_bin Cargo.toml");
    let core_lib = core_dir.join("lib.rs");
    std::fs::write(
        &core_lib,
        "#[macro_export]\nmacro_rules! id_val { ($x:expr) => { $x }; }\nmacro_rules! def_helper { () => { pub fn inner_val() -> u64 { 10 } }; }\ndef_helper!();\npub fn helper() -> u64 { 40 + inner_val() }\npub fn main_helper() -> u64 { id_val!(helper() + 2) }\n",
    )
    .expect("write core_lib lib.rs");
    std::fs::write(
        app_dir.join("main.rs"),
        "fn main() { let _ = core_lib::main_helper(); }\n",
    )
    .expect("write app_bin main.rs");
    core_lib
}

#[test]
fn incremental_reload_keeps_dependents_and_handles_pubtype_and_error_recovery() {
    let dir = fresh_dir();
    let core_lib = write_two_crate_workspace(&dir.0);
    let targets = host_targets();
    let sel = crate::Selection {
        workspace: true,
        locked: true,
        offline: true,
        ..crate::selection()
    };
    let ws = load_workspace(&dir.0, &targets, &sel);
    let mut build = Build::load(&ws, &targets).expect("load build");
    build.compile_binary(&ws).expect("compile binary");
    assert!(
        build
            .binary
            .as_ref()
            .is_some_and(|b| b.reached.all_live_ok()),
        "initial live err"
    );

    let core_krate = build
        .compilation
        .outline
        .roots
        .iter()
        .position(|r| r.name == "core_lib")
        .expect("core_lib crate") as u32;
    let app_krate = build
        .compilation
        .outline
        .roots
        .iter()
        .position(|r| r.name == "app_bin")
        .expect("app_bin crate") as usize;
    let app_file0 = Arc::clone(&build.compilation.outline.crates[app_krate].files[0]);
    let base_reached = build.reached_count();
    assert!(base_reached > 0);

    let base_src = "#[macro_export]\nmacro_rules! id_val { ($x:expr) => { $x }; }\nmacro_rules! def_helper { () => { pub fn inner_val() -> u64 { 10 } }; }\ndef_helper!();\npub fn helper() -> u64 { 4000 + inner_val() }\npub fn main_helper() -> u64 { id_val!(helper() + 2) }\n";
    std::fs::write(&core_lib, base_src).expect("fnbody refresh");
    build.refresh(&ws, &targets).expect("refresh fnbody");
    assert!(Arc::ptr_eq(
        &build.compilation.outline.crates[app_krate].files[0],
        &app_file0
    ));
    assert_eq!(build.reached_count(), base_reached);
    assert!(
        build
            .binary
            .as_ref()
            .is_some_and(|b| b.broken.is_empty() && b.poly_failures.is_empty())
    );

    let with_a = format!("{base_src}pub struct BenchA(pub u64);\n");
    std::fs::write(&core_lib, &with_a).expect("append struct A");
    build.refresh(&ws, &targets).expect("refresh append A");
    assert!(Arc::ptr_eq(
        &build.compilation.outline.crates[app_krate].files[0],
        &app_file0
    ));
    assert_eq!(build.reached_count(), base_reached);

    let with_b = format!("{base_src}pub struct BenchB_1(pub u64);\n");
    std::fs::write(&core_lib, &with_b).expect("replace struct A -> B");
    build.refresh(&ws, &targets).expect("refresh replace B");
    assert!(Arc::ptr_eq(
        &build.compilation.outline.crates[app_krate].files[0],
        &app_file0
    ));
    assert_eq!(build.reached_count(), base_reached);

    let with_err = format!("{with_b}pub fn bench_err() -> u32 {{ true }}\n");
    std::fs::write(&core_lib, &with_err).expect("add type error");
    build.refresh(&ws, &targets).expect("refresh error");
    assert!(
        build
            .binary
            .as_ref()
            .is_some_and(|b| b.reached.list.iter().any(|(_, r)| r.is_err()))
    );

    std::fs::write(&core_lib, &with_b).expect("recover error");
    build.refresh(&ws, &targets).expect("refresh recover");
    assert!(Arc::ptr_eq(
        &build.compilation.outline.crates[app_krate].files[0],
        &app_file0
    ));
    assert_eq!(build.reached_count(), base_reached);
    let bin = build.binary.as_ref().expect("binary batch");
    assert!(bin.broken.is_empty(), "broken: {:?}", bin.broken);
    assert!(
        bin.poly_failures.is_empty(),
        "poly: {:?}",
        bin.poly_failures
    );
    assert!(bin.reached.all_live_ok(), "live err");

    let pre = super::snapshot_edited(
        &build.compilation.outline,
        &mut build.compilation.names,
        &[core_krate],
    );
    std::fs::write(
        &core_lib,
        "#[macro_export]\nmacro_rules! id_val { ($x:expr) => { $x }; }\nmacro_rules! def_helper { () => { pub fn inner_val() -> u64 { 10 } }; }\ndef_helper!();\npub fn helper() -> u64 { 99999 + inner_val() }\npub fn main_helper() -> u64 { id_val!(helper() + 2) }\n",
    )
    .expect("fnbody edit for reload_edited");
    let reloaded = build
        .compilation
        .outline
        .reload_edited(&[core_krate], std::slice::from_ref(&core_lib))
        .expect("direct outline reload");
    assert_eq!(reloaded, vec![core_krate as usize]);
    let resolved = super::resolve_with_dependents(
        &mut build.compilation,
        pre,
        reloaded,
        &[],
        std::slice::from_ref(&core_lib),
    )
    .expect("resolve_with_dependents")
    .expect("diff_reload result");
    assert!(
        resolved.2.is_some(),
        "unchanged expanded file must stay scoped"
    );
}

#[test]
fn refresh_fails_on_local_errors_and_catches_poly_fn_body_edits() {
    let dir = fresh_dir();
    let src_dir = dir.0.join("src");
    std::fs::create_dir_all(&src_dir).expect("mkdir src");
    std::fs::write(
        dir.0.join("Cargo.toml"),
        "[package]\nname = \"poly_pkg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        dir.0.join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"poly_pkg\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.lock");
    let lib_rs = src_dir.join("lib.rs");
    let ok_src = "pub fn keep<T>(x: T) -> T { x }\npub fn call_site() -> u64 { 1 }\n";
    std::fs::write(&lib_rs, ok_src).expect("write lib.rs");

    let targets = host_targets();
    let sel = crate::Selection {
        locked: true,
        offline: true,
        ..crate::selection()
    };
    let ws = load_workspace(&dir.0, &targets, &sel);
    let mut build = Build::load(&ws, &targets).expect("load build");
    build.compile_binary(&ws).expect("compile");
    assert_eq!(build.check(Duration::ZERO).expect("initial check"), 0);

    std::fs::write(&lib_rs, format!("{ok_src}compile_error!(\"boom\");\n"))
        .expect("write compile_error");
    let err = build
        .refresh(&ws, &targets)
        .expect_err("compile_error! must fail refresh");
    assert!(format!("{err:#}").contains("boom"), "err: {err:#}");

    std::fs::write(&lib_rs, ok_src).expect("recover compile_error");
    build.refresh(&ws, &targets).expect("recover refresh");
    build.compile_binary(&ws).expect("recompile");

    std::fs::write(&lib_rs, format!("{ok_src}mod missing_sub;\n")).expect("write mod missing");
    assert!(
        build.refresh(&ws, &targets).is_err(),
        "missing module must fail refresh"
    );

    std::fs::write(&lib_rs, ok_src).expect("recover mod missing");
    build.refresh(&ws, &targets).expect("recover mod refresh");
    build.compile_binary(&ws).expect("recompile after mod");

    let bad_poly =
        "pub fn keep<T>(x: T) -> T { let _: u32 = true; x }\npub fn call_site() -> u64 { 1 }\n";
    std::fs::write(&lib_rs, bad_poly).expect("break generic body");
    build.refresh(&ws, &targets).expect("refresh bad poly");
    assert!(
        build
            .binary
            .as_ref()
            .is_some_and(|b| !b.poly_failures.is_empty()),
        "poly body edit must record poly failure"
    );
    assert_ne!(build.check(Duration::ZERO).expect("check bad poly"), 0);

    std::fs::write(&lib_rs, ok_src).expect("fix generic body");
    build.refresh(&ws, &targets).expect("refresh fixed poly");
    assert!(
        build
            .binary
            .as_ref()
            .is_some_and(|b| b.poly_failures.is_empty()),
        "fixed poly body must clear poly_failures"
    );
    assert_eq!(build.check(Duration::ZERO).expect("check fixed poly"), 0);
}

#[test]
fn check_lib_skips_binary_target() {
    let dir = fresh_dir();
    let src_dir = dir.0.join("src");
    std::fs::create_dir_all(&src_dir).expect("mkdir src");
    std::fs::write(
        dir.0.join("Cargo.toml"),
        "[package]\nname = \"dual_pkg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        dir.0.join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"dual_pkg\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(src_dir.join("lib.rs"), "pub fn ok() -> u64 { 42 }\n").expect("write lib.rs");
    std::fs::write(
        src_dir.join("main.rs"),
        "fn main() { let _: u32 = true; }\n",
    )
    .expect("write main.rs");

    let cfg = std::sync::Arc::new(crate::config::Config {
        selection: crate::Selection {
            lib: true,
            locked: true,
            offline: true,
            ..crate::Selection::default()
        },
        ..crate::config::Config::for_target(haste_codegen::Target::host())
    });
    crate::with_config(cfg, || {
        let targets = host_targets();
        let ws = load_workspace(&dir.0, &targets, &crate::selection());
        let mut build = Build::load_oneshot(&ws, &targets).expect("load build");
        build.compile_binary_oneshot(&ws).expect("compile lib only");
        assert_eq!(build.check(Duration::ZERO).expect("check --lib"), 0);
    });
}

fn write_single_pkg(root: &Path, name: &str, files: &[(&str, &str)]) -> PathBuf {
    let src_dir = root.join("src");
    std::fs::create_dir_all(&src_dir).expect("mkdir src");
    std::fs::write(
        root.join("Cargo.toml"),
        format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n"),
    )
    .expect("write Cargo.toml");
    std::fs::write(
        root.join("Cargo.lock"),
        format!("version = 4\n\n[[package]]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
    )
    .expect("write Cargo.lock");
    for &(rel, body) in files {
        std::fs::write(src_dir.join(rel), body).expect("write source file");
    }
    src_dir
}

#[test]
fn refresh_catches_same_size_same_mtime_edit() {
    let dir = fresh_dir();
    let old_src = "fn main() {\n    let _: u32 = 123;\n}\n";
    let new_src = "fn main() {\n    let _: u32 = \"a\";\n}\n";
    assert_eq!(old_src.len(), new_src.len());
    let src_dir = write_single_pkg(&dir.0, "same_stamp_pkg", &[("main.rs", old_src)]);
    let main_rs = src_dir.join("main.rs");

    let targets = host_targets();
    let sel = crate::Selection {
        locked: true,
        offline: true,
        ..crate::selection()
    };
    let ws = load_workspace(&dir.0, &targets, &sel);
    let mut build = Build::load(&ws, &targets).expect("load build");
    build.compile_binary(&ws).expect("compile initial");
    assert_eq!(build.check(Duration::ZERO).expect("initial check"), 0);

    let orig_mtime = std::fs::metadata(&main_rs)
        .expect("main.rs metadata")
        .modified()
        .expect("main.rs mtime");
    let orig_stamp = crate::graph::store::file_stamp(&main_rs);

    std::fs::write(&main_rs, new_src).expect("overwrite main.rs");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&main_rs)
        .expect("open main.rs")
        .set_modified(orig_mtime)
        .expect("restore mtime");
    assert_eq!(
        std::fs::metadata(&main_rs)
            .expect("post metadata")
            .modified()
            .expect("post mtime"),
        orig_mtime
    );
    assert_ne!(
        crate::graph::store::file_stamp(&main_rs),
        orig_stamp,
        "content hash must distinguish same-size same-mtime edits"
    );

    build
        .refresh(&ws, &targets)
        .expect("refresh same-stamp edit");
    assert_ne!(build.check(Duration::ZERO).expect("check type mismatch"), 0);
}

fn setup_two_mod_build(root: &Path) -> (graph::Workspace, Targets, Build, PathBuf, PathBuf) {
    let src_dir = write_single_pkg(
        root,
        "coord_pkg",
        &[
            (
                "main.rs",
                "mod a;\nmod b;\nfn main() {\n    let _ = b::get();\n}\n",
            ),
            ("a.rs", "pub fn get() -> bool {\n    true\n}\n"),
            (
                "b.rs",
                "pub fn get() -> u32 {\n    u32::from(crate::a::get())\n}\n",
            ),
        ],
    );
    let targets = host_targets();
    let sel = crate::Selection {
        locked: true,
        offline: true,
        ..crate::selection()
    };
    let ws = load_workspace(root, &targets, &sel);
    let mut build = Build::load(&ws, &targets).expect("load build");
    build.compile_binary(&ws).expect("compile initial");
    assert_eq!(build.check(Duration::ZERO).expect("initial check"), 0);
    (
        ws,
        targets,
        build,
        src_dir.join("a.rs"),
        src_dir.join("b.rs"),
    )
}

#[test]
fn refresh_handles_coordinated_two_file_item_and_body_edits() {
    let new_a = "pub fn get() -> char {\n    'a'\n}\n";
    let new_b = "pub fn get() -> u32 {\n    crate::a::get() as u32\n}\n";

    let sim_dir = fresh_dir();
    let (ws, targets, mut sim_build, sim_a, sim_b) = setup_two_mod_build(&sim_dir.0);
    std::fs::write(&sim_a, new_a).expect("write a.rs");
    std::fs::write(&sim_b, new_b).expect("write b.rs");
    sim_build
        .refresh(&ws, &targets)
        .expect("refresh coordinated edit");
    assert_eq!(sim_build.check(Duration::ZERO).expect("check sim"), 0);
    assert!(
        sim_build
            .binary
            .as_ref()
            .is_some_and(|bin| bin.reached.all_live_ok()),
        "simultaneous coordinated edit left stale errors"
    );

    let ab_dir = fresh_dir();
    let (ab_ws, ab_targets, mut ab_build, ab_a, ab_b) = setup_two_mod_build(&ab_dir.0);
    std::fs::write(&ab_a, new_a).expect("write a first");
    ab_build.refresh(&ab_ws, &ab_targets).expect("refresh a");
    assert_eq!(ab_build.check(Duration::ZERO).expect("check after a"), 0);
    std::fs::write(&ab_b, new_b).expect("write b second");
    ab_build.refresh(&ab_ws, &ab_targets).expect("refresh b");
    assert_eq!(ab_build.check(Duration::ZERO).expect("check after b"), 0);
    assert!(
        ab_build
            .binary
            .as_ref()
            .is_some_and(|bin| bin.reached.all_live_ok()),
        "sequential a -> b edit left stale errors"
    );

    let ba_dir = fresh_dir();
    let (ba_ws, ba_targets, mut ba_build, ba_a, ba_b) = setup_two_mod_build(&ba_dir.0);
    std::fs::write(&ba_b, new_b).expect("write b first");
    ba_build.refresh(&ba_ws, &ba_targets).expect("refresh b");
    assert_eq!(ba_build.check(Duration::ZERO).expect("check after b"), 0);
    std::fs::write(&ba_a, new_a).expect("write a second");
    ba_build.refresh(&ba_ws, &ba_targets).expect("refresh a");
    assert_eq!(ba_build.check(Duration::ZERO).expect("check after a"), 0);
    assert!(
        ba_build
            .binary
            .as_ref()
            .is_some_and(|bin| bin.reached.all_live_ok()),
        "sequential b -> a edit left stale errors"
    );
}

#[test]
fn watch_reloader_accumulates_pending_edits_across_failed_reloads() {
    let dir = fresh_dir();
    let src_dir = write_single_pkg(
        &dir.0,
        "watch_pending_pkg",
        &[
            (
                "main.rs",
                "mod a;\nmod b;\nfn main() {\n    std::process::exit(a::val() * 10 + b::val());\n}\n",
            ),
            ("a.rs", "pub fn val() -> i32 {\n    1\n}\n"),
            ("b.rs", "pub fn val() -> i32 {\n    4\n}\n"),
        ],
    );
    let a_rs = src_dir.join("a.rs");
    let b_rs = src_dir.join("b.rs");

    let targets = host_targets();
    let sel = crate::Selection {
        locked: true,
        offline: true,
        ..crate::selection()
    };
    let ws = load_workspace(&dir.0, &targets, &sel);
    let mut build = Build::load_oneshot(&ws, &targets).expect("load build");
    let main = super::super::main_item(&ws, &build.compilation.outline).expect("main item");
    let file_slot = |name: &str| -> (u32, u32) {
        let idx = build.compilation.outline.crates[main.krate as usize]
            .files
            .iter()
            .position(|file| file.source.path.file_name().is_some_and(|f| f == name))
            .expect("source file slot");
        (main.krate, idx as u32)
    };
    let a_slot = file_slot("a.rs");
    let b_slot = file_slot("b.rs");

    let mut reloader = super::super::watch::WatchReloader::new(&ws, &build.compilation.outline);
    #[cfg(unix)]
    let mut program =
        crate::image::Program::new(&mut build.compilation, crate::sema::Cache::default(), main)
            .expect("create jit program");
    #[cfg(unix)]
    assert_eq!(
        program
            .run(&mut build.compilation, &[])
            .expect("initial run"),
        14
    );

    std::fs::write(
        &b_rs,
        "pub fn val() -> i32 {\n    let _: i32 = \"broken\";\n    4\n}\n",
    )
    .expect("break b.rs");
    assert!(matches!(
        reloader.step(&ws, &mut build.compilation, main),
        Err(crate::image::ReloadError::Compile(_))
    ));
    #[cfg(unix)]
    assert_eq!(
        program
            .run(&mut build.compilation, &[])
            .expect("still serving old code on failed B"),
        14
    );

    std::fs::write(&a_rs, "pub fn val() -> i32 {\n    2\n}\n").expect("edit a.rs to 2");
    assert!(matches!(
        reloader.step(&ws, &mut build.compilation, main),
        Err(crate::image::ReloadError::Compile(_))
    ));
    #[cfg(unix)]
    assert_eq!(
        program
            .run(&mut build.compilation, &[])
            .expect("still serving old code while B broken"),
        14
    );

    std::fs::write(&b_rs, "pub fn val() -> i32 {\n    5\n}\n").expect("fix b.rs to 5");
    let stale = reloader
        .step(&ws, &mut build.compilation, main)
        .ok()
        .expect("reload succeeds after B fixed");
    assert!(
        stale.contains(&a_slot) && stale.contains(&b_slot),
        "expected both a.rs and b.rs in pending stale list, got {stale:?}"
    );
    #[cfg(unix)]
    {
        program.forget(
            &build.compilation.outline,
            crate::sema::Edited::Files(&stale),
        );
        assert_eq!(
            program
                .run(&mut build.compilation, &[])
                .expect("post-fix run must serve 2 5"),
            25
        );
    }
}
