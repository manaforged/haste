use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use base_db::SourceDatabase;
use hir_ty::db::HirDatabase;
use hir_def::hir::generics::GenericParams;
use hir_def::signatures::FunctionSignature;
use hir_ty::attach_db;
use ide_db::RootDatabase;
use paths::AbsPathBuf;
use vfs::{Vfs, VfsPath};

use crate::backend::Backend;

pub(crate) struct Edit {
    pub(crate) label: String,
    path: String,
    find: String,
    replace: String,
}

pub struct Engine {
    db: RootDatabase,
    backend: std::sync::Arc<Backend>,
}

// SAFETY: the engine is only reached through ENGINE's mutex, so one thread uses the database and JIT module at a time.
unsafe impl Send for Engine {}

impl Engine {
    pub fn parts(&mut self) -> (&RootDatabase, &std::sync::Arc<Backend>) {
        (&self.db, &self.backend)
    }
}

pub static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

pub(crate) fn start(db: RootDatabase, dump: bool, verbose: bool) -> Result<()> {
    let backend = Backend::new(dump, verbose)?;
    *lock() = Some(Engine { db, backend: std::sync::Arc::new(backend) });
    crate::backend::watch_forks();
    crate::macros::refresh();
    Ok(())
}

pub(crate) fn stop() -> RootDatabase {
    lock().take().expect("engine is running").db
}

pub(crate) fn with_database<T>(work: impl FnOnce(&mut RootDatabase) -> T) -> T {
    crate::backend::quiesce();
    work(&mut lock().as_mut().expect("engine is running").db)
}

/// A handle on the running engine's database for work beside the engine; a database change cancels it.
pub(crate) fn snapshot() -> RootDatabase {
    lock().as_ref().expect("engine is running").db.clone()
}

fn lock() -> std::sync::MutexGuard<'static, Option<Engine>> {
    ENGINE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn with_engine<T>(work: impl FnOnce(&dyn HirDatabase, &Backend) -> T) -> T {
    let mut guard = lock();
    let engine = guard.as_mut().expect("engine is running");
    let (db, backend) = engine.parts();
    let db: &dyn HirDatabase = db;
    attach_db(db, || work(db, backend))
}

pub(crate) fn launch(entry: &str) -> Result<(&'static [AtomicUsize], usize)> {
    let (slot, callable, table) = with_engine(|db, backend| {
        let func = crate::find(db, entry).map_err(|e| e.to_string())?;
        let (slot, callable) = backend.entry(db, func)?;
        Ok::<_, String>((slot, callable, backend.table()))
    })
    .map_err(|reason| anyhow::anyhow!(reason))?;
    if !callable {
        bail!("{entry} is not (i64) -> i64");
    }
    Ok((table, slot))
}

pub fn run(mut db: RootDatabase, vfs: &Vfs, root: &Path, entry: &str, edits: &Path, dump: bool) -> Result<()> {
    let edits = read_edits(edits)?;
    let mut texts = HashMap::new();
    let (seeds, edits): (Vec<&Edit>, Vec<&Edit>) = edits.iter().partition(|edit| edit.label == "seed");
    for seed in seeds {
        apply(&mut db, vfs, root, &mut texts, seed)?;
    }
    start(db, dump, true)?;
    let started = Instant::now();
    let (table, slot) = launch(entry)?;
    let first = crate::backend::call(table, slot, 20);
    println!("first call: total_us={} entry(20)={first} compiled={}", started.elapsed().as_micros(), compiled());
    if dump {
        return Ok(());
    }
    let caller = Caller::start(table, slot);
    for edit in edits {
        let started = Instant::now();
        let outcome = apply_edit(vfs, root, &mut texts, edit)?;
        let refresh_us = started.elapsed().as_micros();
        match outcome {
            Ok(names) => {
                let call_start = Instant::now();
                let value = crate::backend::call(table, slot, 20);
                println!(
                    "{}: refresh_us={refresh_us} first_call_us={} recompiled=[{}] entry(20)={value} caller_calls={}",
                    edit.label,
                    call_start.elapsed().as_micros(),
                    names.join(" "),
                    caller.calls()
                );
            }
            Err(reason) => println!(
                "{}: refresh_us={refresh_us} rejected, old code keeps running: {reason} caller_calls={}",
                edit.label,
                caller.calls()
            ),
        }
    }
    println!("compiled_total={}", compiled());
    caller.stop();
    Ok(())
}

fn compiled() -> usize {
    with_engine(|_, backend| backend.compiled_total())
}

pub fn coverage(db: RootDatabase, crate_name: &str) -> Result<()> {
    start(db, false, false)?;
    static PANIC: Mutex<String> = Mutex::new(String::new());
    std::panic::set_hook(Box::new(|info| {
        let place = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let text = info.payload().downcast_ref::<String>().cloned().unwrap_or_default();
        let text = if text.is_empty() { info.payload().downcast_ref::<&str>().map(|it| (*it).to_owned()).unwrap_or_default() } else { text };
        let shown: String = text.chars().take(80).collect();
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        let frames: Vec<&str> = trace.lines().filter(|line| line.contains("haste::")).filter_map(|line| line.split("haste::").nth(1)).take(3).collect();
        *PANIC.lock().expect("panic note") = format!("{place} {shown} {frames:?}");
    }));
    let started = Instant::now();
    let (attempted, generic, entry_reasons, stubs, compiled) = with_engine(|db, backend| {
        let mut entry_reasons: HashMap<String, usize> = HashMap::new();
        let (mut attempted, mut generic) = (0, 0);
        for func in crate::free_functions(db, crate_name) {
            if GenericParams::of(db, func.into()).len() != 0 {
                generic += 1;
                continue;
            }
            attempted += 1;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (slot, _) = backend.entry(db, func)?;
                backend.compile(db, slot);
                Ok::<_, String>(())
            }));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(reason)) => *entry_reasons.entry(bucket(&reason)).or_default() += 1,
                Err(_) => {
                    let place = PANIC.lock().expect("panic note").clone();
                    *entry_reasons.entry(format!("panic at {place}")).or_default() += 1;
                }
            }
        }
        let walk = Instant::now();
        while let Some(slot) = backend.next_uncompiled() {
            if walk.elapsed().as_secs() >= WALK_SECONDS {
                println!("walk stopped at the time limit");
                break;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| backend.compile(db, slot)));
            if result.is_err() {
                backend.mark_failed(slot);
                let place = PANIC.lock().expect("panic note").clone();
                *entry_reasons.entry(format!("panic at {place}")).or_default() += 1;
            }
        }
        println!("walk {:.2}s", walk.elapsed().as_secs_f64());
        (attempted, generic, entry_reasons, backend.take_stubs(), backend.compiled_total())
    });
    let mut reasons: HashMap<String, usize> = HashMap::new();
    let mut samples: HashMap<String, Vec<String>> = HashMap::new();
    for (name, reason) in &stubs {
        let key = bucket(reason);
        *reasons.entry(key.clone()).or_default() += 1;
        let list = samples.entry(key).or_default();
        if list.len() < SAMPLES {
            list.push(format!("{name}: {reason}"));
        }
    }
    let mut sample_rows: Vec<_> = samples.into_iter().collect();
    sample_rows.sort();
    for (key, list) in sample_rows {
        for row in list {
            println!("sample {key} :: {}", row.chars().take(520).collect::<String>());
        }
    }
    println!(
        "attempted {attempted} generic_skipped {generic} finished {compiled} trapped {} {:.2}s",
        stubs.len(),
        started.elapsed().as_secs_f64()
    );
    print_ranked("entry", entry_reasons);
    print_ranked("trap", reasons);
    Ok(())
}

fn print_ranked(kind: &str, reasons: HashMap<String, usize>) {
    let mut ranked: Vec<_> = reasons.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, count) in ranked.into_iter().take(30) {
        println!("{kind} {count} {reason}");
    }
}

const WALK_SECONDS: u64 = 1200;
const SAMPLES: usize = 6;
const CHECK_INPUTS: [i64; 5] = [-5, 0, 1, 3, 20];
const CASE_SECONDS: u32 = 120;
const CASE_PREFIX: &str = "case_";

pub const PROGRAM_STACK: usize = 1 << 30;

unsafe extern "C" {
    fn alarm(seconds: u32) -> u32;
}

pub fn check(db: RootDatabase, skip: &[String]) -> Result<()> {
    start(db, false, true)?;
    let (cases, table) = with_engine(|db, backend| {
        let mut cases = Vec::new();
        for krate in crate::local_crates(db) {
            for func in crate::free_functions_of(db, krate) {
                let name = FunctionSignature::of(db, func).name.as_str().to_owned();
                if name.starts_with(CASE_PREFIX) && !skip.contains(&name) {
                    cases.push((name, backend.entry(db, func)));
                }
            }
        }
        (cases, backend.table())
    });
    let mut ran = 0;
    for (name, entry) in cases {
        let (slot, callable) = entry.map_err(|reason| anyhow::anyhow!("{name}: {reason}"))?;
        if !callable {
            bail!("{name} is not (i64) -> i64");
        }
        for input in CHECK_INPUTS {
            run_case(table, &name, slot, input);
            ran += 1;
        }
    }
    println!("summary ran={ran}");
    Ok(())
}

fn run_case(table: &'static [AtomicUsize], name: &str, slot: usize, input: i64) {
    println!("running {name} {input}");
    // SAFETY: alarm takes no pointers; a case that outlives it ends the process, and the last running line names it.
    unsafe { alarm(CASE_SECONDS) };
    println!("{name} {input} {}", crate::backend::call(table, slot, input));
    // SAFETY: as above; zero cancels the timer.
    unsafe { alarm(0) };
}

fn bucket(reason: &str) -> String {
    reason.chars().take(34).collect()
}

pub(crate) fn read_edits(path: &Path) -> Result<Vec<Edit>> {
    let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    let mut edits = Vec::new();
    for line in text.lines().filter(|line| !line.is_empty()) {
        let parts: Vec<&str> = line.split('|').collect();
        let [label, path, find, replace] = parts[..] else {
            bail!("bad edit line: {line}");
        };
        edits.push(Edit {
            label: label.to_owned(),
            path: path.to_owned(),
            find: find.replace("\\n", "\n"),
            replace: replace.replace("\\n", "\n"),
        });
    }
    Ok(edits)
}

pub(crate) fn apply_edit(
    vfs: &Vfs,
    root: &Path,
    texts: &mut HashMap<String, String>,
    edit: &Edit,
) -> Result<std::result::Result<Vec<String>, String>> {
    let mut guard = lock();
    let engine = guard.as_mut().expect("engine is running");
    crate::backend::quiesce();
    apply(&mut engine.db, vfs, root, texts, edit)?;
    let (db, backend) = engine.parts();
    let db: &dyn HirDatabase = db;
    Ok(attach_db(db, || backend.refresh(db)))
}

fn apply(
    db: &mut RootDatabase,
    vfs: &Vfs,
    root: &Path,
    texts: &mut HashMap<String, String>,
    edit: &Edit,
) -> Result<()> {
    let full = root.join(&edit.path);
    if !texts.contains_key(&edit.path) {
        texts.insert(edit.path.clone(), std::fs::read_to_string(&full)?);
    }
    let text = texts.get_mut(&edit.path).context("edit text")?;
    if !text.contains(&edit.find) {
        bail!("edit {}: text not found: {}", edit.label, edit.find);
    }
    *text = text.replacen(&edit.find, &edit.replace, 1);
    let abs = AbsPathBuf::assert_utf8(std::fs::canonicalize(full)?);
    let (file, _) = vfs
        .file_id(&VfsPath::from(abs))
        .with_context(|| format!("{} is not in the loaded project", edit.path))?;
    db.set_file_text(file, text);
    Ok(())
}

struct Caller {
    stop: Arc<AtomicBool>,
    calls: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Caller {
    fn start(table: &'static [AtomicUsize], slot: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicU64::new(0));
        let thread = {
            let (stop, calls) = (stop.clone(), calls.clone());
            let body = move || {
                crate::fault::install();
                while !stop.load(Ordering::Relaxed) {
                    std::hint::black_box(crate::backend::call(table, slot, 20));
                    calls.fetch_add(1, Ordering::Relaxed);
                }
            };
            Some(std::thread::Builder::new().stack_size(PROGRAM_STACK).spawn(body).expect("caller thread"))
        };
        Self { stop, calls, thread }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("caller thread");
        }
        println!("caller_total={}", self.calls());
    }
}
