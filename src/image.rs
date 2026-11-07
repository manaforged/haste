use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use ide_db::RootDatabase;
use vfs::Vfs;

use crate::backend::{self, call};
use crate::patch;

const REPEATS: usize = 3;
const INPUT: i64 = 20;
const WORD: usize = std::mem::size_of::<u64>();

struct Outcome {
    wall_us: u128,
    program_us: u64,
    value: i64,
    order: Vec<usize>,
}

fn words(bytes: &[u8]) -> Vec<u64> {
    bytes.chunks_exact(WORD).map(|chunk| u64::from_le_bytes(chunk.try_into().expect("word"))).collect()
}

fn child(table: &'static [std::sync::atomic::AtomicUsize], slot: usize, mut report: std::io::PipeWriter) -> ! {
    backend::forget_compiler();
    let started = Instant::now();
    let value = call(table, slot, INPUT);
    let program_us = started.elapsed().as_micros() as u64;
    let order = patch::with_engine(|_, backend| backend.take_order());
    let mut bytes = Vec::with_capacity((order.len() + 2) * WORD);
    bytes.extend_from_slice(&program_us.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
    bytes.extend(order.iter().flat_map(|slot| (*slot as u64).to_le_bytes()));
    let wrote = report.write_all(&bytes);
    // SAFETY: _exit ends the forked copy without running destructors that belong to the parent's state.
    unsafe { libc::_exit(i32::from(wrote.is_err())) }
}

fn fork_run(table: &'static [std::sync::atomic::AtomicUsize], slot: usize) -> Result<Outcome> {
    let (mut read, write) = std::io::pipe()?;
    let started = Instant::now();
    // SAFETY: the parent holds no engine lock here, and the child only uses state that the fork copied.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        drop(read);
        child(table, slot, write);
    }
    drop(write);
    let mut bytes = Vec::new();
    read.read_to_end(&mut bytes)?;
    let mut status = 0;
    // SAFETY: pid is the child forked above, and status is a live local.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    let wall_us = started.elapsed().as_micros();
    let found = words(&bytes);
    let [program_us, value, order @ ..] = found.as_slice() else {
        bail!("program ended without a result, status {status:#x}");
    };
    Ok(Outcome {
        wall_us,
        program_us: *program_us,
        value: *value as i64,
        order: order.iter().map(|slot| *slot as usize).collect(),
    })
}

fn settle(label: &str, table: &'static [std::sync::atomic::AtomicUsize], slot: usize, refresh_us: u128) -> Result<()> {
    let outcome = fork_run(table, slot)?;
    let started = Instant::now();
    patch::with_engine(|db, backend| backend.replay(db, &outcome.order));
    println!(
        "{label}: refresh_us={refresh_us} wall_us={} program_us={} compiled_in_child={} replay_us={} value={}",
        outcome.wall_us,
        outcome.program_us,
        outcome.order.len(),
        started.elapsed().as_micros(),
        outcome.value
    );
    Ok(())
}

pub fn run(db: RootDatabase, vfs: &Vfs, root: &Path, entry: &str, edits: &Path) -> Result<()> {
    let edits = patch::read_edits(edits)?;
    patch::start(db, false, false)?;
    let (slot, table) = patch::with_engine(|db, backend| {
        let func = crate::find(db, entry).map_err(|e| e.to_string())?;
        let (slot, _) = backend.entry(db, func)?;
        Ok::<_, String>((slot, backend.table()))
    })
    .map_err(|reason| anyhow::anyhow!(reason))?;
    for repeat in 0..REPEATS {
        settle(&format!("repeat{repeat}"), table, slot, 0)?;
    }
    let mut texts = HashMap::new();
    for edit in &edits {
        let started = Instant::now();
        let outcome = patch::apply_edit(vfs, root, &mut texts, edit)?;
        let refresh_us = started.elapsed().as_micros();
        if let Err(reason) = outcome {
            println!("{}: rejected, image keeps the old code: {reason}", edit.label);
            continue;
        }
        settle(&edit.label, table, slot, refresh_us)?;
    }
    Ok(())
}
