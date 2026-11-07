use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicUsize;

use anyhow::{Result, bail};
use base_db::{Crate, all_crates};
use cfg::CfgAtom;
use hir_def::FunctionId;
use hir_def::signatures::FunctionSignature;
use hir_ty::db::HirDatabase;
use ide_db::RootDatabase;

use crate::{backend, patch};

const SCRIPT_CRATE: &str = "build-script-build";
const ENTRY: &str = "main";
const OUT_ROOT: &str = "target/engine-scripts";
const LIMIT_SECONDS: u32 = 120;
const RESULT_WORDS: usize = 64;
const SHOWN_LINES: usize = 4;
const SHOWN_FRAMES: usize = 8;

type Table = &'static [AtomicUsize];

struct Script {
    label: String,
    function: FunctionId,
    env: BTreeMap<String, String>,
    manifest: PathBuf,
    real_out: Option<PathBuf>,
}

enum Verdict {
    Agree,
    Differ(Vec<String>),
    Failed(String),
    NoRecord,
}

fn envs_of(db: &dyn HirDatabase, krate: Crate) -> BTreeMap<String, String> {
    krate.env(db).into_iter().map(|(key, value)| (key.clone(), value.clone())).collect()
}

fn package_key(env: &BTreeMap<String, String>) -> Option<(String, String)> {
    Some((env.get("CARGO_PKG_NAME")?.clone(), env.get("CARGO_PKG_VERSION")?.clone()))
}

fn is_script(db: &dyn HirDatabase, krate: Crate) -> bool {
    krate.extra_data(db).display_name.as_ref().is_some_and(|name| name.canonical_name().as_str() == SCRIPT_CRATE)
}

fn feature_envs(db: &dyn HirDatabase, krate: Crate, into: &mut BTreeMap<String, String>) {
    for atom in krate.cfg_options(db) {
        if let CfgAtom::KeyValue { key, value } = atom
            && key.as_str() == "feature"
        {
            let name = value.as_str().to_uppercase().replace('-', "_");
            into.insert(format!("CARGO_FEATURE_{name}"), "1".to_owned());
        }
    }
}

fn rustc_text(args: &[&str]) -> Result<String> {
    Ok(String::from_utf8(Command::new("rustc").args(args).output()?.stdout)?)
}

fn target_envs() -> Result<BTreeMap<String, String>> {
    let mut env = BTreeMap::new();
    let version = rustc_text(&["-vV"])?;
    let Some(host) = version.lines().find_map(|line| line.strip_prefix("host: ")) else {
        bail!("rustc -vV printed no host");
    };
    for (key, value) in [("TARGET", host), ("HOST", host), ("PROFILE", "debug"), ("OPT_LEVEL", "0"), ("DEBUG", "true"), ("RUSTC", "rustc"), ("NUM_JOBS", "1")] {
        env.insert(key.to_owned(), value.to_owned());
    }
    let mut cfgs: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in rustc_text(&["--print", "cfg"])?.lines() {
        let (key, value) = line.split_once('=').unwrap_or((line, "\"\""));
        cfgs.entry(key.to_uppercase()).or_default().push(value.trim_matches('"').to_owned());
    }
    for (key, values) in cfgs {
        env.insert(format!("CARGO_CFG_{key}"), values.join(","));
    }
    Ok(env)
}

fn main_of(db: &dyn HirDatabase, krate: Crate) -> Option<FunctionId> {
    crate::free_functions_of(db, krate).into_iter().find(|id| FunctionSignature::of(db, *id).name.as_str() == ENTRY)
}

fn scripts(db: &dyn HirDatabase, shared: &BTreeMap<String, String>) -> Vec<Script> {
    let mut libs = BTreeMap::new();
    for krate in all_crates(db).iter().copied().filter(|krate| !is_script(db, *krate)) {
        if let Some(key) = package_key(&envs_of(db, krate)) {
            libs.insert(key, krate);
        }
    }
    let mut found = Vec::new();
    for krate in all_crates(db).iter().copied().filter(|krate| is_script(db, *krate)) {
        let mut env = envs_of(db, krate);
        let Some(key) = package_key(&env) else { continue };
        let Some(function) = main_of(db, krate) else { continue };
        let lib = libs.get(&key).copied();
        let real_out = lib.and_then(|lib| envs_of(db, lib).get("OUT_DIR").map(PathBuf::from));
        if let Some(lib) = lib {
            feature_envs(db, lib, &mut env);
        }
        let manifest = PathBuf::from(env.get("CARGO_MANIFEST_DIR").cloned().unwrap_or_default());
        env.extend(shared.clone());
        found.push(Script { label: format!("{} {}", key.0, key.1), function, env, manifest, real_out });
    }
    found
}

fn run_main(table: Table, slot: usize) {
    let mut result = [0_u64; RESULT_WORDS];
    let argument = 0_i64;
    let code = table[slot].load(std::sync::atomic::Ordering::Acquire);
    // SAFETY: the slot holds code taking a return pointer and an argument pointer; main reads neither past these live locals.
    let main: extern "C" fn(*mut u64, *const i64) = unsafe { std::mem::transmute(code) };
    main(result.as_mut_ptr(), &argument);
}

fn child(table: Table, slot: usize, script: &Script, out: &Path, write: std::io::PipeWriter) -> ! {
    use std::os::fd::AsRawFd;
    // SAFETY: alarm only arms a timer for this forked child.
    unsafe { libc::alarm(LIMIT_SECONDS) };
    crate::fault::install();
    // SAFETY: these calls change only this forked child's descriptors and environment, and it has one thread.
    unsafe {
        libc::dup2(write.as_raw_fd(), libc::STDOUT_FILENO);
        libc::dup2(write.as_raw_fd(), libc::STDERR_FILENO);
        for (key, value) in &script.env {
            std::env::set_var(key, value);
        }
        std::env::set_var("OUT_DIR", out);
    }
    let moved = std::env::set_current_dir(&script.manifest);
    if moved.is_ok() {
        run_main(table, slot);
    }
    // SAFETY: _exit ends the forked copy without running destructors that belong to the parent's state.
    unsafe { libc::_exit(i32::from(moved.is_err())) }
}

fn fork_script(table: Table, slot: usize, script: &Script, out: &Path) -> Result<(String, i32)> {
    let (mut read, write) = std::io::pipe()?;
    // SAFETY: the parent holds no engine lock here, and the child only uses state that the fork copied.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        drop(read);
        child(table, slot, script, out, write);
    }
    drop(write);
    let mut bytes = Vec::new();
    read.read_to_end(&mut bytes)?;
    let mut status = 0;
    // SAFETY: pid is the child forked above, and status is a live local.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    Ok((String::from_utf8_lossy(&bytes).into_owned(), status))
}

fn directives(text: &str) -> BTreeSet<String> {
    text.lines().filter(|line| line.starts_with("cargo:")).map(str::to_owned).collect()
}

fn compare(output: &str, status: i32, out: &Path, real_out: Option<&Path>) -> Verdict {
    if status != 0 {
        let notes: Vec<&str> = output.lines().filter(|line| !line.starts_with("cargo:")).collect();
        let (frames, others): (Vec<&str>, Vec<&str>) = notes.iter().partition(|line| line.starts_with("  at"));
        let shown = others.iter().rev().take(SHOWN_LINES).rev().chain(frames.iter().take(SHOWN_FRAMES));
        let tail = shown.copied().collect::<Vec<_>>().join(" | ");
        return Verdict::Failed(format!("status {status:#x}: {tail}"));
    }
    let Some(real_out) = real_out else { return Verdict::NoRecord };
    let Some(record) = real_out.parent().map(|build| build.join("output")) else { return Verdict::NoRecord };
    let Ok(real) = std::fs::read_to_string(record) else { return Verdict::NoRecord };
    let real = directives(&real.replace(&*real_out.to_string_lossy(), &out.to_string_lossy()));
    let ours = directives(output);
    if real == ours {
        return Verdict::Agree;
    }
    let extra = ours.difference(&real).map(|line| format!("+{line}"));
    Verdict::Differ(extra.chain(real.difference(&ours).map(|line| format!("-{line}"))).collect())
}

fn run_one(table: Table, slot: usize, script: &Script, root: &Path) -> Result<Verdict> {
    let out = root.join(script.label.replace(' ', "-"));
    std::fs::create_dir_all(&out)?;
    let (output, status) = fork_script(table, slot, script, &out)?;
    Ok(compare(&output, status, &out, script.real_out.as_deref()))
}

#[derive(Default)]
struct Tally {
    agree: usize,
    differ: usize,
    failed: usize,
    no_record: usize,
}

impl Tally {
    fn add(&mut self, label: &str, verdict: Verdict) {
        match verdict {
            Verdict::Agree => self.agree += 1,
            Verdict::NoRecord => self.no_record += 1,
            Verdict::Differ(lines) => {
                self.differ += 1;
                println!("script_differ {label}: {lines:?}");
            }
            Verdict::Failed(reason) => {
                self.failed += 1;
                println!("script_failed {label}: {reason}");
            }
        }
    }
}

pub fn run(db: RootDatabase, project: &Path) -> Result<()> {
    let shared = target_envs()?;
    patch::start(db, false, false)?;
    let found = patch::with_engine(|db, _| scripts(db, &shared));
    let root = project.join(OUT_ROOT);
    let mut tally = Tally::default();
    for script in &found {
        let entered = patch::with_engine(|db, backend| backend.entry(db, script.function).map(|(slot, _)| (slot, backend.table())));
        let verdict = match entered {
            Ok((slot, table)) => run_one(table, slot, script, &root)?,
            Err(reason) => Verdict::Failed(reason),
        };
        tally.add(&script.label, verdict);
    }
    println!(
        "scripts total={} agree={} differ={} failed={} no_record={}",
        found.len(),
        tally.agree,
        tally.differ,
        tally.failed,
        tally.no_record
    );
    Ok(())
}
