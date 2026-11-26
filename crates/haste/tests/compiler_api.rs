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
