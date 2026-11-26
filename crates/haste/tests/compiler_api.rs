mod common;

use std::path::PathBuf;
use std::process::Command;

use common::FixtureDir;
use haste::{Compiler, ErrorKind};

#[test]
fn builder_validates_target_jobs_and_conflicting_flags() {
    let bad_target = Compiler::builder()
        .target("riscv64gc-unknown-linux-gnu")
        .build()
        .expect_err("unsupported target should fail");
    assert_eq!(bad_target.kind(), ErrorKind::Config);

    let bad_jobs = Compiler::builder()
        .jobs(0)
        .build()
        .expect_err("zero jobs should fail");
    assert_eq!(bad_jobs.kind(), ErrorKind::Config);

    let conflict_target = Compiler::builder()
        .bin("a")
        .example("b")
        .build()
        .expect_err("bin and example together should fail");
    assert_eq!(conflict_target.kind(), ErrorKind::Config);

    let host = haste::Target::host();
    assert_eq!(haste::Target::find(host.triple()), Some(host));
    assert!(haste::Target::supported().contains(&host));
    let _built = Compiler::builder()
        .target(host)
        .debug_info(true)
        .symbols(false)
        .build()
        .expect("explicit target and debug_info/symbols override build cleanly");
}

#[test]
fn cross_target_rejects_jit_run_and_test() {
    let host = haste::Target::host().triple();
    let cross = if host == "x86_64-unknown-linux-gnu" {
        "aarch64-unknown-linux-gnu"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    let compiler = Compiler::builder()
        .target(cross)
        .build()
        .expect("cross compiler builds");

    let run_err = compiler
        .run(".", &[])
        .expect_err("cross-target JIT run must fail");
    assert_eq!(run_err.kind(), ErrorKind::Config);

    let test_err = compiler
        .test(".", &[])
        .expect_err("cross-target JIT test must fail");
    assert_eq!(test_err.kind(), ErrorKind::Config);
}

#[test]
fn missing_workspace_reports_io_error() {
    let missing = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("haste-nonexistent-{}", std::process::id()));
    let err = Compiler::new()
        .check(&missing)
        .expect_err("missing directory must fail");
    assert_eq!(err.kind(), ErrorKind::Io);
    assert!(!err.to_string().is_empty());
}

#[test]
fn compiler_api_checks_runs_tests_and_emits_standalone_executable() {
    let dir = FixtureDir::write_crate(
        "api",
        "e2e",
        "api_fixture",
        r#"
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("exit42") {
        std::process::exit(42);
    }
    println!("api-ok:{:?}", args);
}

#[test]
fn sanity_math() {
    assert_eq!(6 * 7, 42);
}
"#,
    );

    let compiler = Compiler::builder().jobs(2).build().expect("valid compiler");
    compiler.check(dir.path()).expect("check succeeds");
    assert_eq!(
        compiler
            .run(dir.path(), &["exit42".to_owned()])
            .expect("jit run succeeds"),
        42
    );
    assert_eq!(
        compiler
            .run(dir.path(), &["ok".to_owned()])
            .expect("second jit run succeeds"),
        0
    );
    assert_eq!(
        compiler.test(dir.path(), &[]).expect("test run succeeds"),
        0
    );

    let bin = dir
        .path()
        .join(format!("api_fixture_aot{}", std::env::consts::EXE_SUFFIX));
    compiler.emit(dir.path(), &bin).expect("aot emit succeeds");
    let out = Command::new(&bin)
        .arg("hello")
        .output()
        .expect("run emitted binary");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(stdout.contains("api-ok:"));
    assert!(stdout.contains("\"hello\""));
}

#[test]
fn cli_run_isolates_argv_and_test_prints_panic_stderr() {
    let dir = FixtureDir::write_crate(
        "cli",
        "argv_panic",
        "cli_argv_panic",
        r#"
fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("ARGS={}", args[1..].join("|"));
}

#[test]
fn failing_assertion_reports_message() {
    assert_eq!(2 + 2, 5, "custom_panic_marker_8472");
}
"#,
    );
    let haste = env!("CARGO_BIN_EXE_haste");

    let run_out = Command::new(haste)
        .args(["run"])
        .arg(dir.path())
        .args(["--release", "--", "--release", "alpha", "beta"])
        .output()
        .expect("run haste cli");
    assert!(
        run_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run_out.stderr)
    );
    let stdout = String::from_utf8(run_out.stdout).expect("utf8 stdout");
    assert_eq!(stdout.trim(), "ARGS=--release|alpha|beta");

    let test_out = Command::new(haste)
        .args(["test"])
        .arg(dir.path())
        .output()
        .expect("run haste test cli");
    assert!(!test_out.status.success());
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&test_out.stdout),
        String::from_utf8_lossy(&test_out.stderr)
    );
    assert!(
        combined.contains("custom_panic_marker_8472"),
        "expected panic message in test output, got:\n{combined}"
    );
}

#[cfg(unix)]
fn run_serve_cmd(haste: &str, verb: &str, dir: &std::path::Path) -> std::process::Output {
    Command::new(haste)
        .args(["serve", verb])
        .arg(dir)
        .output()
        .expect("run haste serve control")
}

#[cfg(unix)]
#[test]
fn cli_daemon_is_opt_in_and_serve_status_stop_controls_lifecycle() {
    let dir = FixtureDir::write_crate(
        "cli",
        "daemon",
        "cli_daemon_probe",
        r#"
fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("DAEMON_ARGS={}", args[1..].join(","));
}
"#,
    );
    let haste = env!("CARGO_BIN_EXE_haste");

    let plain = Command::new(haste)
        .args(["build"])
        .arg(dir.path())
        .output()
        .expect("plain build");
    assert!(plain.status.success());

    let status_before = run_serve_cmd(haste, "status", dir.path());
    assert!(status_before.status.success());
    let before_text = String::from_utf8_lossy(&status_before.stdout);
    assert!(
        before_text.contains("no active haste daemons"),
        "plain build should not spawn daemon, got: {before_text}"
    );

    struct StopGuard<'a>(&'a str, &'a std::path::Path);
    impl Drop for StopGuard<'_> {
        fn drop(&mut self) {
            drop(run_serve_cmd(self.0, "stop", self.1));
        }
    }
    let _guard = StopGuard(haste, dir.path());

    let daemon_run = Command::new(haste)
        .env("HASTE_SERVE_IDLE_SECS", "30")
        .args(["run", "--daemon"])
        .arg(dir.path())
        .args(["--", "one", "two"])
        .output()
        .expect("daemon run");
    assert!(
        daemon_run.status.success(),
        "daemon run failed: {}",
        String::from_utf8_lossy(&daemon_run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&daemon_run.stdout).trim(),
        "DAEMON_ARGS=one,two"
    );

    let mut active_text = String::new();
    for _ in 0..300 {
        let status_active = run_serve_cmd(haste, "status", dir.path());
        assert!(status_active.status.success());
        active_text = String::from_utf8_lossy(&status_active.stdout).into_owned();
        if active_text.contains("pid=") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(
        active_text.contains("pid="),
        "expected running daemon, got: {active_text}"
    );

    let served_run = Command::new(haste)
        .args(["run", "--daemon"])
        .arg(dir.path())
        .args(["--", "served_alpha", "served_beta"])
        .output()
        .expect("served daemon run");
    assert!(
        served_run.status.success(),
        "served daemon run failed: {}",
        String::from_utf8_lossy(&served_run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&served_run.stdout).trim(),
        "DAEMON_ARGS=served_alpha,served_beta"
    );

    let stop = run_serve_cmd(haste, "stop", dir.path());
    assert!(stop.status.success());
    assert!(String::from_utf8_lossy(&stop.stdout).contains("stopped pid="));

    let status_after = run_serve_cmd(haste, "status", dir.path());
    assert!(status_after.status.success());
    let after_text = String::from_utf8_lossy(&status_after.stdout);
    assert!(
        after_text.contains("no active haste daemons"),
        "daemon should be stopped, got: {after_text}"
    );
}

#[cfg(unix)]
fn assert_aborts_jit_and_aot(haste: &str, name: &str, pkg: &str, source: &str) {
    use std::os::unix::process::ExitStatusExt;

    let dir = FixtureDir::write_crate("haste-rt", name, pkg, source);
    let jit = Command::new(haste)
        .args(["run"])
        .arg(dir.path())
        .output()
        .expect("jit run");
    assert!(
        !jit.status.success(),
        "{name} JIT unexpectedly exited 0: {}",
        String::from_utf8_lossy(&jit.stdout)
    );
    let jit_out = String::from_utf8_lossy(&jit.stdout);
    assert!(
        jit_out.contains("before") && !jit_out.contains("survived"),
        "{name} JIT unexpected stdout: {jit_out}"
    );
    assert!(
        jit.status.signal() == Some(libc::SIGABRT)
            || jit.status.code() == Some(128 + libc::SIGABRT),
        "{name} JIT expected SIGABRT, got {:?}",
        jit.status
    );

    let bin = dir.path().join("out_bin");
    let build = Command::new(haste)
        .args(["--release", "build", "-o"])
        .arg(&bin)
        .arg(dir.path())
        .output()
        .expect("aot build");
    assert!(
        build.status.success(),
        "{name} AOT build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let aot = Command::new(&bin).output().expect("aot run");
    assert!(
        !aot.status.success(),
        "{name} AOT unexpectedly exited 0: {}",
        String::from_utf8_lossy(&aot.stdout)
    );
    let aot_out = String::from_utf8_lossy(&aot.stdout);
    assert!(
        aot_out.contains("before") && !aot_out.contains("survived"),
        "{name} AOT unexpected stdout: {aot_out}"
    );
    assert_eq!(
        aot.status.signal(),
        Some(libc::SIGABRT),
        "{name} AOT expected SIGABRT, got {:?}",
        aot.status
    );
}

#[cfg(unix)]
#[test]
fn runtime_unwind_abort_and_lazy_jit_close_fds() {
    let haste = env!("CARGO_BIN_EXE_haste");
    assert_aborts_jit_and_aot(
        haste,
        "extern-c-panic",
        "extern_c_panic",
        r#"
use std::panic::catch_unwind;
extern "C" fn cb() { panic!("in C abi"); }
fn main() {
    println!("before");
    let r = catch_unwind(|| cb());
    println!("survived {}", r.is_err());
}
"#,
    );
    assert_aborts_jit_and_aot(
        haste,
        "double-panic",
        "double_panic",
        r#"
use std::panic::catch_unwind;
struct Bomb;
impl Drop for Bomb { fn drop(&mut self) { panic!("second"); } }
fn main() {
    println!("before");
    let r = catch_unwind(|| { let _b = Bomb; panic!("first"); });
    println!("survived {}", r.is_err());
}
"#,
    );

    assert_aborts_jit_and_aot(
        haste,
        "extern-c-tls-panic",
        "extern_c_tls_panic",
        r#"
use std::panic::catch_unwind;
thread_local! { static TL: u64 = panic!("tls init in extern C"); }
#[inline(never)] fn slow() -> u64 { std::hint::black_box(7) }
extern "C" fn cb(flag: bool) -> u64 {
    let x = TL.with(|v| *v);
    if flag { x + slow() } else { x }
}
fn main() {
    println!("before");
    let r = catch_unwind(|| cb(std::hint::black_box(false)));
    println!("survived {}", r.is_err());
}
"#,
    );

    let cunwind = FixtureDir::write_crate(
        "haste-rt",
        "extern-cunwind-ok",
        "extern_cunwind_ok",
        r#"
use std::panic::catch_unwind;
extern "C-unwind" fn cb() { panic!("in C-unwind"); }
fn main() {
    let r = catch_unwind(|| cb());
    println!("caught {}", r.is_err());
}
"#,
    );
    let cu_jit = Command::new(haste)
        .args(["run"])
        .arg(cunwind.path())
        .output()
        .expect("cunwind jit run");
    assert!(
        cu_jit.status.success(),
        "C-unwind JIT failed: {}",
        String::from_utf8_lossy(&cu_jit.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&cu_jit.stdout).trim(),
        "caught true"
    );

    let dict_drop = FixtureDir::write_crate(
        "haste-rt",
        "dict-drop-rest",
        "dict_drop_rest",
        r#"
struct D(u32);
impl Drop for D {
    fn drop(&mut self) {
        println!("drop {}", self.0);
        if self.0 == 0 { panic!("d0"); }
    }
}
fn run_arr<T>(a: [T; 3]) { drop(a); }
fn main() {
    let r = std::panic::catch_unwind(|| run_arr([D(0), D(1), D(2)]));
    println!("caught {}", r.is_err());
}
"#,
    );
    let dd_out = Command::new(haste)
        .env("HASTE_DICT_MIN_BYTES", "0")
        .args(["run"])
        .arg(dict_drop.path())
        .output()
        .expect("dict drop-rest run");
    assert!(
        dd_out.status.success(),
        "dict drop-rest failed: {}",
        String::from_utf8_lossy(&dd_out.stderr)
    );
    let dd_stdout = String::from_utf8_lossy(&dd_out.stdout);
    assert!(
        dd_stdout.contains("drop 0")
            && dd_stdout.contains("drop 1")
            && dd_stdout.contains("drop 2")
            && dd_stdout.contains("caught true"),
        "dict drop-rest skipped elements: {dd_stdout}"
    );

    let dir = FixtureDir::write_crate(
        "haste-rt",
        "close-fds",
        "close_fds",
        r#"
extern "C" { fn close(fd: i32) -> i32; }
#[inline(never)] fn late(x: u32) -> u32 { x + 1 }
fn main() {
    let mut fd = 3;
    while fd < 256 {
        unsafe { close(fd); }
        fd += 1;
    }
    println!("{}", late(41));
}
"#,
    );
    let out = Command::new(haste)
        .args(["run"])
        .arg(dir.path())
        .output()
        .expect("lazy jit close fds run");
    assert!(
        out.status.success(),
        "lazy JIT failed after closing fds 3..256: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "42");
}

#[test]
fn reports_spanned_diagnostics_for_call_op_index_and_pat_errors() {
    let haste = env!("CARGO_BIN_EXE_haste");
    let cases = [
        (
            "diag-call-arity",
            "diag_call_arity",
            "fn f(a: i32, b: i32) -> i32 { a + b }\nfn main() { let _ = f(1); }\n",
            "error[type]:",
            "call takes 2 arguments, 1 given",
        ),
        (
            "diag-not-callable",
            "diag_not_callable",
            "fn main() { let x = 5i32; let _ = x(); }\n",
            "error[type]:",
            "calling `i32` is not supported",
        ),
        (
            "diag-bad-index",
            "diag_bad_index",
            "fn main() { let x = 5i32; let _ = x[0]; }\n",
            "error[type]:",
            "cannot index",
        ),
        (
            "diag-bad-op",
            "diag_bad_op",
            "struct S;\nfn main() { let _ = S + S; }\n",
            "error[type]:",
            "unsupported operator",
        ),
        (
            "diag-pat-field",
            "diag_pat_field",
            "struct S { a: i32 }\nfn main() { let S { nope } = S { a: 1 }; let _ = nope; }\n",
            "error[name]:",
            "no field `nope` on `S`",
        ),
        (
            "diag-or-pat",
            "diag_or_pat",
            "enum E { A(i32), B }\nfn main() { match E::B { E::A(x) | E::B => { let _ = x; } } }\n",
            "error[name]:",
            "bind different names",
        ),
        (
            "diag-deferred-shift",
            "diag_deferred_shift",
            "fn make<T>() -> T { loop {} }\nfn main() { let x = make(); let _ = x << true; let _: i32 = x; }\n",
            "error[type]:",
            "unsupported shift",
        ),
        (
            "diag-named-variant-pat",
            "diag_named_variant_pat",
            "enum E { A { x: i32 }, B }\nfn main() { match E::B { E::A => {} E::B => {} } }\n",
            "error[type]:",
            "expected struct pattern with named fields",
        ),
    ];
    for (dir_name, pkg, src, kind, msg) in cases {
        let dir = FixtureDir::write_crate("haste-diag", dir_name, pkg, src);
        let out = Command::new(haste)
            .args(["build", "--jobs", "1"])
            .arg(dir.path())
            .output()
            .expect("run haste build");
        assert!(!out.status.success(), "{dir_name} unexpectedly compiled");
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            combined.contains(kind)
                && combined.contains(msg)
                && combined.contains("--> ")
                && combined.contains('^'),
            "{dir_name}: expected spanned diagnostic ({kind} / {msg}), got:\n{combined}"
        );
    }
}

#[test]
fn compiler_check_and_emit_return_diagnostics_with_source_location() {
    let dir = FixtureDir::write_crate(
        "api-diag",
        "check_emit",
        "api_diag_check_emit",
        "fn main() {\n    let value: i32 = \"bad\";\n}\n",
    );
    let compiler = Compiler::builder().jobs(1).build().expect("valid compiler");

    let check_err = compiler
        .check(dir.path())
        .expect_err("type error must fail check");
    assert_eq!(check_err.kind(), ErrorKind::Compile);
    let check_msg = check_err.to_string();
    assert!(
        check_msg.contains("error[type]:")
            && check_msg.contains("src/main.rs:2:")
            && check_msg != "compilation failed",
        "expected formatted check diagnostic with source location, got:\n{check_msg}"
    );

    let bin = dir
        .path()
        .join(format!("api_diag_out{}", std::env::consts::EXE_SUFFIX));
    let emit_err = compiler
        .emit(dir.path(), &bin)
        .expect_err("type error must fail emit");
    assert_eq!(emit_err.kind(), ErrorKind::Compile);
    let emit_msg = emit_err.to_string();
    assert!(
        emit_msg.contains("error[type]:") && emit_msg.contains("src/main.rs:2:"),
        "expected formatted emit diagnostic with source location, got:\n{emit_msg}"
    );
}

#[test]
fn compiler_test_filters_match_cli_behavior() {
    let dir = FixtureDir::write_crate(
        "api-filter",
        "test_filter",
        "api_test_filter",
        r#"
unsafe extern "C" {
    fn haste_nonexistent_test_symbol_49123();
}

#[test]
fn selected_passing_test() {
    assert_eq!(2 + 2, 4);
}

#[test]
fn unselected_broken_test() {
    unsafe {
        haste_nonexistent_test_symbol_49123();
    }
}
"#,
    );
    let haste = env!("CARGO_BIN_EXE_haste");
    let cli_out = Command::new(haste)
        .args(["test"])
        .arg(dir.path())
        .args(["--", "selected_passing_test"])
        .output()
        .expect("run haste test cli");
    assert!(
        cli_out.status.success(),
        "cli filtered test failed: {}",
        String::from_utf8_lossy(&cli_out.stderr)
    );

    let compiler = Compiler::builder().jobs(1).build().expect("valid compiler");
    let status = compiler
        .test(dir.path(), &["selected_passing_test".to_owned()])
        .expect("filtered Compiler::test must skip unselected broken test");
    assert_eq!(status, 0);

    let err = compiler
        .test(dir.path(), &["unselected_broken_test".to_owned()])
        .expect_err("selecting broken test must return Compile error");
    assert_eq!(err.kind(), ErrorKind::Compile);
    let msg = err.to_string();
    assert!(
        msg.contains("haste_nonexistent_test_symbol_49123") && msg.contains("src/main.rs:"),
        "expected formatted compile diagnostic with source location, got:\n{msg}"
    );
}

fn write_worktree_test_fixture(name: &str) -> FixtureDir {
    let dir = FixtureDir::new("tmpdir-wt", name);
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"shared_wt_pkg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        dir.path().join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"shared_wt_pkg\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(dir.path().join("src/lib.rs"), "").expect("write src/lib.rs");
    std::fs::create_dir_all(dir.path().join("tests")).expect("create tests dir");
    std::fs::write(
        dir.path().join("tests/shared_it.rs"),
        "#[test]\nfn prints_target_tmpdir() {\n    println!(\"TARGET_TMPDIR={}\", env!(\"CARGO_TARGET_TMPDIR\"));\n}\n",
    )
    .expect("write tests/shared_it.rs");
    dir
}

fn read_target_tmpdir(haste: &str, dir: &std::path::Path) -> PathBuf {
    let out = Command::new(haste)
        .args(["test"])
        .arg(dir)
        .args(["--", "--show-output"])
        .output()
        .expect("run haste test");
    assert!(
        out.status.success(),
        "haste test failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("TARGET_TMPDIR="))
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("missing TARGET_TMPDIR line in:\n{stdout}"))
}

#[test]
fn integration_test_target_tmpdir_is_isolated_per_worktree() {
    let wt_a = write_worktree_test_fixture("wt_a");
    let wt_b = write_worktree_test_fixture("wt_b");
    let haste = env!("CARGO_BIN_EXE_haste");

    let tmp_a1 = read_target_tmpdir(haste, wt_a.path());
    let tmp_b = read_target_tmpdir(haste, wt_b.path());
    let tmp_a2 = read_target_tmpdir(haste, wt_a.path());

    assert_ne!(
        tmp_a1, tmp_b,
        "two worktrees must not share CARGO_TARGET_TMPDIR"
    );
    assert_eq!(
        tmp_a1, tmp_a2,
        "CARGO_TARGET_TMPDIR must be deterministic per worktree"
    );
    assert!(tmp_a1.is_dir() && tmp_b.is_dir());
    let expected_root = std::env::temp_dir().join("haste-target-tmp");
    assert!(tmp_a1.starts_with(&expected_root));
    assert!(tmp_b.starts_with(&expected_root));

    drop(std::fs::remove_dir_all(&tmp_a1));
    drop(std::fs::remove_dir_all(&tmp_b));
}

#[test]
fn env_macro_change_invalidates_oneshot_and_daemon_executables() {
    let dir = FixtureDir::write_crate(
        "env-inval",
        "app",
        "env_inval_app",
        "fn main() { println!(\"{}:{}\", env!(\"BUILD_LABEL\"), option_env!(\"OPT_LABEL\").unwrap_or(\"none\")); }\n",
    );
    let haste = env!("CARGO_BIN_EXE_haste");
    let bin = dir
        .path()
        .join(format!("env_out{}", std::env::consts::EXE_SUFFIX));

    let run_build = |daemon: bool, label: &str, opt: Option<&str>| -> String {
        let mut cmd = Command::new(haste);
        cmd.args(["build", "-o"]).arg(&bin);
        if daemon {
            cmd.arg("--daemon");
        }
        cmd.arg(dir.path()).env("BUILD_LABEL", label);
        match opt {
            Some(v) => cmd.env("OPT_LABEL", v),
            None => cmd.env_remove("OPT_LABEL"),
        };
        let out = cmd.output().expect("run haste build");
        assert!(
            out.status.success(),
            "build failed (daemon={daemon}, label={label}):\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let ran = Command::new(&bin).output().expect("run built binary");
        assert!(ran.status.success(), "built binary failed");
        String::from_utf8_lossy(&ran.stdout).trim().to_owned()
    };

    assert_eq!(run_build(false, "alpha", None), "alpha:none");
    assert_eq!(run_build(false, "beta", Some("set")), "beta:set");
    assert_eq!(run_build(false, "beta", None), "beta:none");

    #[cfg(unix)]
    {
        assert_eq!(run_build(true, "gamma", None), "gamma:none");
        assert_eq!(run_build(true, "delta", Some("on")), "delta:on");
        assert_eq!(run_build(true, "epsilon", None), "epsilon:none");
        let _ = Command::new(haste)
            .args(["serve", "stop"])
            .arg(dir.path())
            .output();
    }
}
