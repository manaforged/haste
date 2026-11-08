pub mod census;
mod store;

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
use load_cargo::SourceRootConfig;
use paths::AbsPathBuf;

use crate::patch;

const SCRIPT_CRATE: &str = "build-script-build";
const ENTRY: &str = "main";
const OUT_ENV: &str = "OUT_DIR";
const LINKS_ENV: &str = "CARGO_MANIFEST_LINKS";
const LIMIT_SECONDS: u32 = 120;
const RESULT_WORDS: usize = 64;
const SHOWN_LINES: usize = 4;
const SHOWN_FRAMES: usize = 8;
const RESERVED_KEYS: &[&str] = &[
    "rerun-if-changed",
    "rerun-if-env-changed",
    "rustc-link-lib",
    "rustc-link-search",
    "rustc-flags",
    "rustc-cdylib-link-arg",
    "rustc-bin-link-arg",
    "rustc-link-arg",
    "rustc-link-arg-bin",
    "rustc-link-arg-bins",
    "rustc-link-arg-tests",
    "rustc-link-arg-examples",
    "rustc-link-arg-benches",
    "rustc-check-cfg",
    "warning",
    "error",
];

type Table = &'static [AtomicUsize];
type Units = BTreeMap<String, Vec<Crate>>;
type Exports = BTreeMap<String, Vec<(String, String)>>;

struct Script {
    label: String,
    version: String,
    krate: Crate,
    registry: bool,
    env: BTreeMap<String, String>,
    manifest: PathBuf,
    name: String,
    out: PathBuf,
    out_text: String,
}

enum Directive<'a> {
    Cfg(&'a str),
    Env(&'a str, &'a str),
    Metadata(&'a str, &'a str),
}

struct Forked {
    pid: libc::pid_t,
    read: std::io::PipeReader,
}

/// How a build reaches each build script's output.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scripts {
    /// Runs every script and keeps nothing.
    Live,
    /// Uses the saved output under the project when its key matches, and runs the rest.
    Saved,
    /// Runs every script and saves its output under the project.
    Generate,
}

struct Loaded<'a> {
    root: &'a Path,
    mode: Scripts,
    release: String,
    vfs: &'a mut vfs::Vfs,
    roots: &'a SourceRootConfig,
    shared: &'a BTreeMap<String, String>,
    exports: Exports,
}

/// What a build script did.
pub enum Run {
    Done(String),
    Failed(String),
}

/// One build script run and the out dir of its unit.
pub struct Outcome {
    pub label: String,
    pub name: String,
    pub out: PathBuf,
    pub run: Run,
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

fn target_envs() -> Result<(BTreeMap<String, String>, String)> {
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
    Ok((env, version))
}

fn main_of(db: &dyn HirDatabase, krate: Crate) -> Option<FunctionId> {
    crate::free_functions_of(db, krate).into_iter().find(|id| FunctionSignature::of(db, *id).name.as_str() == ENTRY)
}

fn dep_envs(db: &dyn HirDatabase, members: &[Crate], exports: &Exports) -> Vec<(String, String)> {
    let libs = members.iter().copied().filter(|krate| !is_script(db, *krate));
    let deps = libs.flat_map(|krate| krate.data(db).dependencies.iter().map(|dep| dep.crate_id));
    let outs = deps.filter_map(|dep| dep.env(db).get(OUT_ENV));
    outs.filter_map(|out| exports.get(&out)).flatten().cloned().collect()
}

fn script_of(db: &dyn HirDatabase, members: &[Crate], shared: &BTreeMap<String, String>, exports: &Exports) -> Option<Result<Script, String>> {
    let krate = members.iter().copied().find(|krate| is_script(db, *krate))?;
    let mut env = envs_of(db, krate);
    Some((|| {
        let (name, version) = package_key(&env).ok_or("the script has no CARGO_PKG_NAME or CARGO_PKG_VERSION")?;
        let manifest = PathBuf::from(env.get("CARGO_MANIFEST_DIR").ok_or_else(|| format!("{name} {version}: no CARGO_MANIFEST_DIR"))?);
        let out_text = env.get(OUT_ENV).ok_or_else(|| format!("{name} {version}: no {OUT_ENV}"))?.clone();
        feature_envs(db, krate, &mut env);
        env.extend(shared.clone());
        env.extend(dep_envs(db, members, exports));
        let registry = !krate.data(db).origin.is_local();
        Ok(Script { label: format!("{name} {version}"), version, krate, registry, env, manifest, name, out: PathBuf::from(&out_text), out_text })
    })())
}

#[derive(Clone, Copy)]
struct Entry {
    main: usize,
    report: Option<usize>,
}

fn run_main(table: Table, entry: Entry) -> i32 {
    use std::sync::atomic::Ordering::Acquire;
    let mut result = [0_u64; RESULT_WORDS];
    let argument = 0_i64;
    crate::backend::enter(table[entry.main].load(Acquire), result.as_mut_ptr(), &argument);
    let Some(report) = entry.report else {
        return 0;
    };
    let mut code = [0_u64; RESULT_WORDS];
    crate::backend::enter(table[report].load(Acquire), code.as_mut_ptr(), result.as_ptr().cast());
    i32::from(code[0] as u8)
}

fn child(table: Table, entry: Entry, script: &Script, write: std::io::PipeWriter) -> ! {
    use std::os::fd::AsRawFd;
    // SAFETY: alarm only arms a timer for this forked child.
    unsafe { libc::alarm(LIMIT_SECONDS) };
    crate::fault::install();
    crate::fault::write_perf_map();
    // SAFETY: these calls change only this forked child's descriptors and environment, and it has one thread.
    unsafe {
        libc::dup2(write.as_raw_fd(), libc::STDOUT_FILENO);
        libc::dup2(write.as_raw_fd(), libc::STDERR_FILENO);
        for (key, value) in &script.env {
            std::env::set_var(key, value);
        }
    }
    let status = match std::env::set_current_dir(&script.manifest) {
        Ok(()) => run_main(table, entry),
        Err(_) => 1,
    };
    // SAFETY: _exit ends the forked copy without running destructors that belong to the parent's state.
    unsafe { libc::_exit(status) }
}

fn fork_script(table: Table, entry: Entry, script: &Script) -> Result<Forked> {
    let (read, write) = std::io::pipe()?;
    // SAFETY: the parent holds no engine lock and runs no other thread here, and the child only uses state that the fork copied.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        drop(read);
        child(table, entry, script, write);
    }
    drop(write);
    Ok(Forked { pid, read })
}

fn collect_forked(forked: Forked) -> Result<(String, i32)> {
    let Forked { pid, mut read } = forked;
    let mut bytes = Vec::new();
    let outcome = read.read_to_end(&mut bytes);
    drop(read);
    let mut status = 0;
    // SAFETY: pid is the child forked for this pipe, and status is a live local.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    outcome?;
    Ok((String::from_utf8_lossy(&bytes).into_owned(), status))
}

fn directives(text: &str) -> BTreeSet<String> {
    text.lines().filter(|line| line.starts_with("cargo:")).map(str::to_owned).collect()
}

fn run_of(output: String, status: i32) -> Run {
    if status == 0 {
        return Run::Done(output);
    }
    let notes: Vec<&str> = output.lines().filter(|line| !line.starts_with("cargo:")).collect();
    let (frames, others): (Vec<&str>, Vec<&str>) = notes.iter().partition(|line| line.starts_with("  at"));
    let shown = others.iter().rev().take(SHOWN_LINES).rev().chain(frames.iter().take(SHOWN_FRAMES));
    let tail = shown.copied().collect::<Vec<_>>().join(" | ");
    Run::Failed(format!("status {status:#x}: {tail}"))
}

fn start_batch<'a>(scripts: impl Iterator<Item = &'a Script>) -> Vec<Result<Forked, String>> {
    let mut started = Vec::new();
    for script in scripts {
        let entered = patch::with_engine(|db, backend| {
            let function = main_of(db, script.krate).ok_or_else(|| format!("{}: the script has no main", script.label))?;
            let (main, _) = backend.entry(db, function)?;
            let report = backend.report(db, function).map_err(|why| format!("main's result: {why}"))?;
            Ok::<_, String>((Entry { main, report }, backend.table()))
        });
        started.push(match entered {
            Ok((entry, table)) => std::fs::create_dir_all(&script.out)
                .map_err(anyhow::Error::from)
                .and_then(|()| fork_script(table, entry, script))
                .map_err(|error| format!("could not start: {error}")),
            Err(reason) => Err(reason),
        });
    }
    started
}

fn collect_batch(started: Vec<Result<Forked, String>>) -> Result<Vec<Run>> {
    std::thread::scope(|scope| {
        let readers: Vec<_> = started.into_iter().map(|forked| scope.spawn(move || forked.map(collect_forked))).collect();
        readers.into_iter().map(|reader| finish_run(reader.join().expect("reader thread"))).collect()
    })
}

fn finish_run(joined: Result<Result<(String, i32)>, String>) -> Result<Run> {
    match joined {
        Ok(Ok((output, status))) => Ok(run_of(output, status)),
        Ok(Err(error)) => Ok(Run::Failed(format!("reading the script output: {error}"))),
        Err(reason) => Ok(Run::Failed(reason)),
    }
}

fn parse_directive(line: &str) -> Option<Directive<'_>> {
    let (rest, modern) = match line.strip_prefix("cargo::") {
        Some(rest) => (rest, true),
        None => (line.strip_prefix("cargo:")?, false),
    };
    let (key, value) = rest.split_once('=')?;
    match key {
        "rustc-cfg" => Some(Directive::Cfg(value)),
        "rustc-env" => value.split_once('=').map(|(name, text)| Directive::Env(name, text)),
        "metadata" if modern => value.split_once('=').map(|(name, text)| Directive::Metadata(name, text)),
        _ if modern || RESERVED_KEYS.contains(&key) => None,
        _ => Some(Directive::Metadata(key, value)),
    }
}

fn add_files_under(dir: &Path, into: &mut Vec<(AbsPathBuf, Vec<u8>)>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            add_files_under(&path, into)?;
        } else {
            into.push((AbsPathBuf::assert_utf8(path.clone()), std::fs::read(&path)?));
        }
    }
    Ok(())
}

fn export_key(links: &str, key: &str) -> String {
    format!("DEP_{links}_{key}").to_uppercase().replace('-', "_")
}

fn apply_output(state: &mut Loaded<'_>, members: &[Crate], script: &Script, output: &str, files: Vec<(AbsPathBuf, Vec<u8>)>) -> Result<()> {
    let mut cfgs = Vec::new();
    let mut envs = Vec::new();
    let mut exports = Vec::new();
    for directive in output.lines().filter_map(parse_directive) {
        match directive {
            Directive::Cfg(text) => cfgs.push(project_model::parse_cfg(text).map_err(anyhow::Error::msg)?),
            Directive::Env(name, text) => envs.push((name.to_owned(), text.to_owned())),
            Directive::Metadata(name, text) => exports.push((name.to_owned(), text.to_owned())),
        }
    }
    if let Some(links) = script.env.get(LINKS_ENV) {
        let keyed = exports.into_iter().map(|(key, text)| (export_key(links, &key), text));
        state.exports.insert(script.out_text.clone(), keyed.collect());
    }
    patch::with_database(|db| {
        let built: Vec<Crate> = members.iter().copied().filter(|krate| !is_script(db, *krate)).collect();
        for krate in built {
            krate.add_script_output(db, &cfgs, &envs);
        }
        if !files.is_empty() {
            load_cargo::add_files(db, state.vfs, state.roots, files);
        }
    });
    Ok(())
}

fn macro_units(units: &Units) -> BTreeSet<String> {
    patch::with_engine(|db, _| {
        let mut reach = BTreeSet::new();
        let mut stack: Vec<Crate> = all_crates(db).iter().copied().filter(|krate| krate.data(db).target_kind.is_proc_macro()).collect();
        while let Some(krate) = stack.pop() {
            if reach.insert(krate) {
                stack.extend(krate.data(db).dependencies.iter().map(|dep| dep.crate_id));
            }
        }
        units.iter().filter(|(_, members)| members.iter().any(|member| reach.contains(member))).map(|(out, _)| out.clone()).collect()
    })
}

fn units_of(db: &dyn HirDatabase) -> Units {
    let mut units = Units::new();
    for krate in all_crates(db).iter().copied() {
        if let Some(out) = krate.env(db).get(OUT_ENV) {
            units.entry(out).or_default().push(krate);
        }
    }
    units
}

fn visit(db: &dyn HirDatabase, units: &Units, krate: Crate, levels: &mut BTreeMap<Crate, usize>) -> usize {
    if let Some(level) = levels.get(&krate) {
        return *level;
    }
    let out = krate.env(db).get(OUT_ENV);
    let members = out.as_ref().and_then(|out| units.get(out)).map_or(std::slice::from_ref(&krate), Vec::as_slice);
    levels.extend(members.iter().map(|member| (*member, 0)));
    let mut deepest = 0;
    for member in members {
        for dep in &member.data(db).dependencies {
            deepest = deepest.max(visit(db, units, dep.crate_id, levels));
        }
    }
    let level = deepest + usize::from(out.is_some());
    levels.extend(members.iter().map(|member| (*member, level)));
    level
}

fn layers_of(db: &dyn HirDatabase, units: &Units) -> Vec<Vec<String>> {
    let mut levels = BTreeMap::new();
    for krate in all_crates(db).iter().copied() {
        visit(db, units, krate, &mut levels);
    }
    let mut layers: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (out, members) in units {
        layers.entry(levels[&members[0]]).or_default().push(out.clone());
    }
    layers.into_values().collect()
}

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get())
}

fn binaries() -> Vec<Crate> {
    patch::with_engine(|db, _| crate::local_crates(db).into_iter().filter(|krate| !is_script(db, *krate)).collect())
}

fn run_outs(state: &mut Loaded<'_>, units: &Units, outs: &[String], warm: bool, outcomes: &mut Vec<Outcome>) -> Result<()> {
    let scripts = patch::with_engine(|db, _| {
        let found = outs.iter().filter_map(|out| script_of(db, &units[out], state.shared, &state.exports));
        let (scripts, skipped): (Vec<_>, Vec<_>) = found.partition(Result::is_ok);
        for reason in skipped.into_iter().filter_map(Result::err) {
            println!("script_failed {reason}");
        }
        scripts.into_iter().filter_map(Result::ok).collect::<Vec<_>>()
    });
    let mut live = Vec::new();
    for script in scripts {
        let place = place_of(state, &script);
        match (state.mode == Scripts::Saved && script.registry).then(|| store::load(&place)).flatten() {
            Some(saved) => {
                store::write(&saved.files)?;
                apply_output(state, &units[&script.out_text], &script, &saved.stdout, saved.files)?;
                outcomes.push(Outcome { label: script.label, name: script.name, out: script.out, run: Run::Done(saved.stdout) });
            }
            None => live.push((script, place)),
        }
    }
    let started = start_batch(live.iter().map(|(script, _)| script));
    let warmer = warm.then(|| crate::warm::Warmer::start(binaries(), cores()));
    let runs = collect_batch(started);
    if let Some(warmer) = warmer {
        warmer.stop();
    }
    let runs = runs?;
    for ((script, place), run) in live.into_iter().zip(runs) {
        finish_live(state, &units[&script.out_text], &script, &place, &run)?;
        outcomes.push(Outcome { label: script.label, name: script.name, out: script.out, run });
    }
    Ok(())
}

fn finish_live(state: &mut Loaded<'_>, members: &[Crate], script: &Script, place: &store::Place, run: &Run) -> Result<()> {
    match run {
        Run::Done(output) => {
            let mut files = Vec::new();
            add_files_under(&script.out, &mut files)?;
            if state.mode == Scripts::Generate && script.registry {
                store::save(place, output, &files)?;
            }
            apply_output(state, members, script, output, files)
        }
        Run::Failed(reason) => {
            println!("script_failed {}: {reason}", script.label);
            Ok(())
        }
    }
}

const KEY_PREFIXES: &[&str] = &["CARGO_FEATURE_", "CARGO_CFG_", "DEP_", "TARGET"];

fn place_of(state: &Loaded<'_>, script: &Script) -> store::Place {
    let keyed = script.env.iter().filter(|(key, _)| KEY_PREFIXES.iter().any(|prefix| key.starts_with(prefix)));
    let mut key: String = keyed.map(|(key, value)| format!("{key}={value}\n")).collect();
    key.push_str(&state.release);
    store::Place::new(state.root, &script.name, &script.version, key, &script.out)
}

/// Runs the local binary's main in this process and returns its exit code.
pub fn run_binary(db: RootDatabase) -> Result<i32> {
    let warmed = std::time::Instant::now();
    let binaries: Vec<Crate> = crate::local_crates(&db).into_iter().filter(|krate| !is_script(&db, *krate)).collect();
    crate::warm::warm(&db, &binaries, cores());
    eprintln!("warm {:.2}s", warmed.elapsed().as_secs_f64());
    patch::start(db, false, false)?;
    let entered = patch::with_engine(|db, backend| {
        let binaries = crate::local_crates(db).into_iter().filter(|krate| !is_script(db, *krate));
        let function = binaries.filter_map(|krate| main_of(db, krate)).next().ok_or("no local crate has a main")?;
        let (main, _) = backend.entry(db, function)?;
        let report = backend.report(db, function).map_err(|why| format!("main's result: {why}"))?;
        Ok::<_, String>((Entry { main, report }, backend.table()))
    });
    let (entry, table) = entered.map_err(anyhow::Error::msg)?;
    let code = run_main(table, entry);
    let (compiled, slots) = patch::with_engine(|_, backend| (backend.compiled_total(), backend.slot_count()));
    let (replayed, live) = crate::macros::replay::counts();
    eprintln!("compiled {compiled} of {slots} functions, replayed {replayed} expansions, expanded {live} live");
    Ok(code)
}

/// Expands every crate's item macros and saves the recorded expansions.
pub fn expand_all(db: RootDatabase) -> Result<usize> {
    patch::start(db, false, false)?;
    patch::with_engine(|db, _| {
        for krate in all_crates(db).iter().copied() {
            hir_def::nameres::crate_def_map(db, krate);
        }
    });
    crate::macros::replay::save()
}

/// Runs the build scripts layer by layer in parallel and applies each output to its unit.
pub fn run_all(db: RootDatabase, vfs: &mut vfs::Vfs, roots: &SourceRootConfig, root: &Path, mode: Scripts) -> Result<(RootDatabase, Vec<Outcome>)> {
    let (shared, release) = target_envs()?;
    patch::start(db, false, false)?;
    let (units, layers) = patch::with_engine(|db, _| {
        let units = units_of(db);
        let layers = layers_of(db, &units);
        (units, layers)
    });
    let macro_units = macro_units(&units);
    let chunks: Vec<&[String]> = layers.iter().flat_map(|layer| layer.chunks(cores())).collect();
    let mut state = Loaded { root, mode, release, vfs, roots, shared: &shared, exports: BTreeMap::new() };
    let mut outcomes = Vec::new();
    for (index, outs) in chunks.iter().enumerate() {
        let warm = chunks[index..].iter().flat_map(|chunk| chunk.iter()).all(|out| !macro_units.contains(out));
        run_outs(&mut state, &units, outs, warm, &mut outcomes)?;
        if outs.iter().any(|out| macro_units.contains(out)) {
            crate::macros::refresh();
        }
    }
    Ok((patch::stop(), outcomes))
}
