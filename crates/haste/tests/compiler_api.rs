use std::path::{Path, PathBuf};
use std::process::Command;

use haste::{Compiler, ErrorKind};

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new(name: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("haste-api-{name}-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&dir));
        std::fs::create_dir_all(dir.join("src")).expect("create fixture src");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

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
    let dir = FixtureDir::new("e2e");
    std::fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"api_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        dir.path().join("Cargo.lock"),
        "version = 4\n\n[[package]]\nname = \"api_fixture\"\nversion = \"0.1.0\"\n",
    )
    .expect("write Cargo.lock");
    std::fs::write(
        dir.path().join("src/main.rs"),
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
    )
    .expect("write main.rs");

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
