mod common;

use std::process::Command;

use common::FixtureDir;

#[test]
fn release_lazy_jit_drops_batch_exit0_and_cargo_pkg_meta() {
    let haste = env!("CARGO_BIN_EXE_haste");

    let lazy_deep = FixtureDir::write_crate(
        "haste-rt",
        "lazy-deep",
        "lazy_deep",
        r#"
struct D;
impl Drop for D { fn drop(&mut self) { println!("DROPPED"); } }
#[inline(never)] fn work(do_panic: bool) { let _d = D; if do_panic { panic!("boom"); } }
#[cold] #[inline(never)] fn hop1() { hop2(); }
#[cold] #[inline(never)] fn hop2() { hop3(); }
#[cold] #[inline(never)] fn hop3() { let _ = std::panic::catch_unwind(|| work(true)); }
fn main() { work(false); hop1(); println!("DONE"); }
"#,
    );
    let out = Command::new(haste)
        .args(["--release", "run"])
        .arg(lazy_deep.path())
        .output()
        .expect("release lazy jit run");
    assert!(
        out.status.success(),
        "release lazy jit failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout.matches("DROPPED").count(),
        2,
        "release lazy JIT must drop locals during late-discovered catch_unwind: {stdout}"
    );

    let batch = FixtureDir::write_crate(
        "haste-rt",
        "batch-exit0",
        "batch_exit0",
        r#"
#[test]
fn a_exits_zero() {
    std::process::exit(0);
}
#[test]
fn b_must_still_fail() {
    panic!("second test ran");
}
"#,
    );
    let bout = Command::new(haste)
        .args(["test", "--jobs", "1"])
        .arg(batch.path())
        .output()
        .expect("haste test batch exit0");
    assert!(
        !bout.status.success(),
        "test after std::process::exit(0) must still run and fail; stdout={} stderr={}",
        String::from_utf8_lossy(&bout.stdout),
        String::from_utf8_lossy(&bout.stderr)
    );

    let meta = FixtureDir::write_crate(
        "haste-rt",
        "pkg-meta",
        "pkg_meta",
        r#"
fn main() {
    println!("{}|{}", env!("CARGO_PKG_DESCRIPTION"), env!("CARGO_PKG_AUTHORS"));
}
"#,
    );
    std::fs::write(
        meta.path().join("Cargo.toml"),
        "[package]\nname = \"pkg_meta\"\nversion = \"0.1.0\"\nedition = \"2021\"\ndescription = \"fast compiler\"\nauthors = [\"Alice\", \"Bob\"]\n",
    )
    .expect("write Cargo.toml");
    let mout = Command::new(haste)
        .args(["run"])
        .arg(meta.path())
        .output()
        .expect("pkg_meta run");
    assert!(
        mout.status.success(),
        "pkg_meta failed: {}",
        String::from_utf8_lossy(&mout.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&mout.stdout).trim(),
        "fast compiler|Alice:Bob"
    );
}

#[test]
fn handles_float_int_literals_deferred_unsize_static_mut_guard_drops_and_large_arrays() {
    let haste = env!("CARGO_BIN_EXE_haste");
    let dir = FixtureDir::write_crate(
        "haste-reg",
        "ui-regressions",
        "ui_regressions",
        r#"
use std::ops::Add;
use std::ptr::NonNull;
use std::sync::Mutex;

fn add_t<T: Add>(a: T, b: T) -> <T as Add>::Output { a + b }
const fn dangling_slice<T>() -> NonNull<[T]> { NonNull::<[T; 1]>::dangling() }
const C: NonNull<[i32]> = dangling_slice();

fn bare() {}
fn alt() {}
struct Wrap<F: FnOnce()>(F);
static mut CLOSURES: &'static mut [Wrap<fn()>] = &mut [Wrap(bare as fn()), Wrap(bare as fn())];
static BIG_ZST: [(); usize::MAX] = [(); usize::MAX];

enum Forward {
    X = Forward::Y as isize * 2,
    Y = 9,
}

macro_rules! nest {
    ($cur:ident $prev:ty) => {
        #[allow(dead_code)]
        enum $cur { E, A($prev), B($prev), C($prev), D($prev) }
    };
}
nest!(N0 ()); nest!(N1 N0); nest!(N2 N1); nest!(N3 N2); nest!(N4 N3);
nest!(N5 N4); nest!(N6 N5); nest!(N7 N6); nest!(N8 N7); nest!(N9 N8);
nest!(N10 N9); nest!(N11 N10); nest!(N12 N11); nest!(N13 N12); nest!(N14 N13);
nest!(N15 N14); nest!(N16 N15); nest!(N17 N16); nest!(N18 N17); nest!(N19 N18);
nest!(N20 N19);

static LOOP_STATIC: i32 = {
    let mut s = 0;
    let mut i = 0;
    while i < 3 {
        let r = &mut 0;
        *r += 1;
        s += *r;
        i += 1;
    }
    s
};
static mut LOOP_STATIC_MUT: i32 = {
    let mut s = 0;
    let mut i = 0;
    while i < 3 {
        let r = &mut 0;
        *r += 1;
        s += *r;
        i += 1;
    }
    s
};

static DROP_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct CountDrop(u32);
impl Drop for CountDrop {
    fn drop(&mut self) {
        DROP_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[allow(dead_code)]
fn partial_move_large(arr: [String; 1_000_000]) -> String {
    let [first, ..] = arr;
    first
}

fn partial_move_65() {
    let first = {
        let arr: [CountDrop; 65] = std::array::from_fn(|i| CountDrop(i as u32));
        let [first, ..] = arr;
        first
    };
    assert_eq!(first.0, 0);
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::SeqCst), 64);
    drop(first);
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::SeqCst), 65);
}

static ZST_DROPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct ZstDrop;
impl Drop for ZstDrop {
    fn drop(&mut self) {
        ZST_DROPS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

fn partial_move_zst() {
    let x = {
        let arr = [ZstDrop, ZstDrop, ZstDrop];
        let [_, x, _] = arr;
        x
    };
    assert_eq!(ZST_DROPS.load(std::sync::atomic::Ordering::SeqCst), 2);
    drop(x);
    assert_eq!(ZST_DROPS.load(std::sync::atomic::Ordering::SeqCst), 3);
}

fn diverging_guard_no_drop() {
    let before = DROP_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let o: Option<CountDrop> = None;
    match o {
        Some(s) if return => drop(s),
        _ => {}
    }
    assert_eq!(DROP_COUNT.load(std::sync::atomic::Ordering::SeqCst), before);
    let live = Some(CountDrop(7));
    match (live, false) {
        (Some(_s), true) if return => {}
        _ => {}
    }
    assert_eq!(
        DROP_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        before + 1
    );
}

fn main() {
    assert_eq!(add_t::<f32>(100f32, 200f32), 300.0f32);
    assert_eq!(1f64 as i32, 1);
    assert_eq!(LOOP_STATIC, 3);
    assert_eq!(unsafe { LOOP_STATIC_MUT }, 3);
    assert_eq!(C.as_ptr(), NonNull::<[i32; 1]>::dangling().as_ptr() as *mut _);
    assert_eq!(C.as_ptr().len(), 1);
    unsafe {
        for c in &mut *CLOSURES {
            let Wrap(ref mut f) = *c;
            (*f)();
            *f = alt;
            (*f)();
        }
    }
    assert_eq!(BIG_ZST.len(), usize::MAX);
    assert_eq!(Forward::X as isize, 18);
    assert_eq!(Forward::Y as isize, 9);
    let m = Some(Mutex::new(true));
    match m {
        Some(ref z) if *z.lock().unwrap() => assert!(*z.lock().unwrap()),
        _ => panic!("guard failed"),
    }
    partial_move_65();
    partial_move_zst();
    diverging_guard_no_drop();
    let _ = N20::E;
    println!("UI_REG_OK");
}
"#,
    );
    let jit = Command::new(haste)
        .args(["run"])
        .arg(dir.path())
        .output()
        .expect("ui regressions jit");
    assert!(
        jit.status.success() && String::from_utf8_lossy(&jit.stdout).trim() == "UI_REG_OK",
        "ui regressions JIT failed: stdout={} stderr={}",
        String::from_utf8_lossy(&jit.stdout),
        String::from_utf8_lossy(&jit.stderr)
    );
    let bin = dir.path().join("reg_aot_bin");
    let aot_b = Command::new(haste)
        .args(["--release", "build", "-o"])
        .arg(&bin)
        .arg(dir.path())
        .output()
        .expect("ui regressions aot build");
    assert!(
        aot_b.status.success(),
        "ui regressions AOT build failed: {}",
        String::from_utf8_lossy(&aot_b.stderr)
    );
    let aot_r = Command::new(&bin).output().expect("ui regressions aot run");
    assert!(
        aot_r.status.success() && String::from_utf8_lossy(&aot_r.stdout).trim() == "UI_REG_OK",
        "ui regressions AOT run failed: stdout={} stderr={}",
        String::from_utf8_lossy(&aot_r.stdout),
        String::from_utf8_lossy(&aot_r.stderr)
    );

    for (dir_name, pkg, src) in [
        (
            "bad-bin-float",
            "bad_bin_float",
            "fn main() { let _ = 0b1f32; }\n",
        ),
        (
            "bad-cross-enum-cycle",
            "bad_cross_enum_cycle",
            "enum A { X = B::Y as isize }\nenum B { Y = A::X as isize }\nfn main() { let _ = A::X as isize; }\n",
        ),
        (
            "bad-discr-overflow",
            "bad_discr_overflow",
            "#[repr(u8)]\nenum U { A = 255, B }\nfn main() { let _ = U::B as u8; }\n",
        ),
    ] {
        let bad_dir = FixtureDir::write_crate("haste-reg", dir_name, pkg, src);
        let bad_out = Command::new(haste)
            .args(["build", "--jobs", "1"])
            .arg(bad_dir.path())
            .output()
            .expect("bad regression build");
        assert!(
            !bad_out.status.success(),
            "{dir_name} must be rejected, stdout={} stderr={}",
            String::from_utf8_lossy(&bad_out.stdout),
            String::from_utf8_lossy(&bad_out.stderr)
        );
    }
}

#[test]
fn run_test_and_aot_reject_uncalled_errors_and_cli_help_is_per_subcommand() {
    let haste = env!("CARGO_BIN_EXE_haste");

    let global_help = Command::new(haste)
        .arg("--help")
        .output()
        .expect("global help");
    let build_help = Command::new(haste)
        .args(["build", "--help"])
        .output()
        .expect("build help");
    let run_help = Command::new(haste)
        .args(["run", "--help"])
        .output()
        .expect("run help");
    let test_help = Command::new(haste)
        .args(["test", "--help"])
        .output()
        .expect("test help");
    let version = Command::new(haste)
        .arg("--version")
        .output()
        .expect("version");
    assert!(
        global_help.status.success()
            && build_help.status.success()
            && run_help.status.success()
            && test_help.status.success()
            && version.status.success()
    );
    let g = String::from_utf8_lossy(&global_help.stdout);
    let b = String::from_utf8_lossy(&build_help.stdout);
    let r = String::from_utf8_lossy(&run_help.stdout);
    let t = String::from_utf8_lossy(&test_help.stdout);
    assert!(g.contains("Commands:") && b.contains("Usage: haste build") && b != g);
    assert!(r.contains("Usage: haste run") && r.contains("--watch") && r != b);
    assert!(t.contains("Usage: haste test") && t.contains("--ignored") && t != r);
    assert!(String::from_utf8_lossy(&version.stdout).starts_with("haste "));

    let broken = FixtureDir::write_crate(
        "haste-reg",
        "uncalled-errors",
        "uncalled_errors",
        r#"
#[allow(dead_code)]
fn bad_type() -> i32 {
    "not an int"
}

#[allow(dead_code)]
fn bad_move() {
    let s = String::from("hi");
    drop(s);
    println!("{s}");
}

fn main() {
    println!("SHOULD_NOT_RUN");
}

#[test]
fn passing_test() {
    println!("TEST_SHOULD_NOT_RUN");
}
"#,
    );

    let run_out = Command::new(haste)
        .args(["run"])
        .arg(broken.path())
        .output()
        .expect("haste run broken");
    assert!(
        !run_out.status.success(),
        "haste run must refuse code with uncalled errors; stdout={} stderr={}",
        String::from_utf8_lossy(&run_out.stdout),
        String::from_utf8_lossy(&run_out.stderr)
    );
    assert!(!String::from_utf8_lossy(&run_out.stdout).contains("SHOULD_NOT_RUN"));

    let test_out = Command::new(haste)
        .args(["test"])
        .arg(broken.path())
        .output()
        .expect("haste test broken");
    assert!(
        !test_out.status.success(),
        "haste test must refuse code with uncalled errors; stdout={} stderr={}",
        String::from_utf8_lossy(&test_out.stdout),
        String::from_utf8_lossy(&test_out.stderr)
    );
    assert!(!String::from_utf8_lossy(&test_out.stdout).contains("TEST_SHOULD_NOT_RUN"));

    let aot_bin = broken.path().join("should_not_exist");
    let aot_out = Command::new(haste)
        .args(["build", "-o"])
        .arg(&aot_bin)
        .arg(broken.path())
        .output()
        .expect("haste build -o broken");
    assert!(
        !aot_out.status.success() && !aot_bin.exists(),
        "haste build -o must refuse code with uncalled errors; stdout={} stderr={}",
        String::from_utf8_lossy(&aot_out.stdout),
        String::from_utf8_lossy(&aot_out.stderr)
    );
}

#[cfg(unix)]
#[test]
fn cold_cache_dep_feature_fetches_in_workspace_dir() {
    use std::os::unix::fs::PermissionsExt;

    let haste = env!("CARGO_BIN_EXE_haste");
    let proj = FixtureDir::write_crate("haste-reg", "cold-fetch", "cold_fetch", "fn main() {}\n");
    std::fs::write(
        proj.path().join("Cargo.toml"),
        "[package]\nname = \"cold_fetch\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nfake_dep = \"0.1.0\"\n",
    )
    .expect("write Cargo.toml");
    std::fs::write(
        proj.path().join("Cargo.lock"),
        "version = 3\n\n[[package]]\nname = \"cold_fetch\"\nversion = \"0.1.0\"\ndependencies = [\n \"fake_dep\",\n]\n\n[[package]]\nname = \"fake_dep\"\nversion = \"0.1.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n",
    )
    .expect("write Cargo.lock");

    let cargo_home = proj.path().join("empty-cargo-home");
    std::fs::create_dir_all(&cargo_home).expect("create cargo home");
    let expected_cwd = std::fs::canonicalize(proj.path()).expect("canonical proj");
    let wrapper = proj.path().join("fake-cargo.sh");
    let script = format!(
        "#!/bin/sh\nset -eu\nif [ \"$(pwd -P)\" != \"{}\" ]; then\n  echo \"wrong cwd: $(pwd -P)\" >&2\n  exit 42\nfi\ndir=\"$CARGO_HOME/registry/src/test-reg/fake_dep-0.1.0\"\nmkdir -p \"$dir/src\"\nprintf '[package]\\nname = \"fake_dep\"\\nversion = \"0.1.0\"\\nedition = \"2021\"\\n\\n[features]\\nextra = []\\n' > \"$dir/Cargo.toml\"\nprintf 'pub fn ok() {{}}\\n' > \"$dir/src/lib.rs\"\n",
        expected_cwd.display()
    );
    std::fs::write(&wrapper, script).expect("write fake cargo");
    let mut perms = std::fs::metadata(&wrapper).expect("meta").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&wrapper, perms).expect("chmod");

    let out = Command::new(haste)
        .current_dir(std::env::temp_dir())
        .env("CARGO", &wrapper)
        .env("CARGO_HOME", &cargo_home)
        .args(["-F", "fake_dep/extra", "debug", "graph"])
        .arg(proj.path())
        .output()
        .expect("run haste debug graph");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "cold-cache -F dep/feature must fetch in project cwd; stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("fake_dep 0.1.0|extra"),
        "expected fetched dep feature in graph output, got:\n{stdout}"
    );

    let vend = FixtureDir::write_crate(
        "haste-reg",
        "cargo-vendor-offline",
        "vendor_app",
        "fn main() { println!(\"{}\", vdep::answer()); }\n",
    );
    std::fs::write(
        vend.path().join("Cargo.toml"),
        "[package]\nname = \"vendor_app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nvdep = \"1.2.3\"\n",
    )
    .expect("write vendor Cargo.toml");
    std::fs::write(
        vend.path().join("Cargo.lock"),
        "version = 3\n\n[[package]]\nname = \"vendor_app\"\nversion = \"0.1.0\"\ndependencies = [\n \"vdep\",\n]\n\n[[package]]\nname = \"vdep\"\nversion = \"1.2.3\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n",
    )
    .expect("write vendor Cargo.lock");
    std::fs::create_dir_all(vend.path().join(".cargo")).expect("mkdir .cargo");
    std::fs::write(
        vend.path().join(".cargo/config.toml"),
        "[source.crates-io]\nreplace-with = \"vendored-sources\"\n\n[source.vendored-sources]\ndirectory = \"third_party/vendor\"\n",
    )
    .expect("write .cargo/config.toml");
    let vdep_dir = vend.path().join("third_party/vendor/vdep");
    std::fs::create_dir_all(vdep_dir.join("src")).expect("mkdir vdep");
    std::fs::write(
        vdep_dir.join("Cargo.toml"),
        "[package]\nname = \"vdep\"\nversion = \"1.2.3\"\nedition = \"2021\"\n",
    )
    .expect("write vdep Cargo.toml");
    std::fs::write(
        vdep_dir.join("src/lib.rs"),
        "pub fn answer() -> u32 { 42 }\n",
    )
    .expect("write vdep lib.rs");

    let empty_home = vend.path().join("empty-home");
    std::fs::create_dir_all(&empty_home).expect("mkdir empty-home");
    let vout = Command::new(haste)
        .env("CARGO_HOME", &empty_home)
        .args(["run"])
        .arg(vend.path())
        .output()
        .expect("run vendored crate offline");
    assert!(
        vout.status.success() && String::from_utf8_lossy(&vout.stdout).trim() == "42",
        "ordinary cargo vendor (unversioned dir + .cargo/config.toml directory) must work offline; stdout={} stderr={}",
        String::from_utf8_lossy(&vout.stdout),
        String::from_utf8_lossy(&vout.stderr)
    );
}
