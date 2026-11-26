//! Daemon registry, `haste serve status`, and `haste serve stop` lifecycle management.

use std::path::{Path, PathBuf};

use anyhow::Result;

#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::FromRawFd;

/// Control byte sent over a daemon socket to check liveness without running a request.
#[cfg(unix)]
pub(super) const PING_BYTE: u8 = 0xFE;

/// Control byte sent over a daemon socket asking the server to shut down cleanly.
#[cfg(unix)]
pub(super) const STOP_BYTE: u8 = 0xFF;

/// One registered resident daemon entry persisted under `$HASTE_HOME/daemons/`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Record {
    pub key: String,
    pub pid: u32,
    pub socket: String,
    pub root: PathBuf,
    pub record_path: PathBuf,
}

fn registry_dir() -> PathBuf {
    crate::graph::store::home().join("daemons")
}

fn record_path(key: &str) -> PathBuf {
    registry_dir().join(format!("{key}.pid"))
}

/// RAII guard that registers a live daemon record and deletes it when the server exits.
#[cfg(unix)]
pub(super) struct Registration {
    path: PathBuf,
}

#[cfg(unix)]
impl Registration {
    pub(super) fn create(key: &str, root: &Path, socket: &str) -> Result<Self> {
        let dir = registry_dir();
        std::fs::create_dir_all(&dir)?;
        let path = record_path(key);
        let body = format!("{}\n{}\n{}\n", std::process::id(), socket, root.display());
        std::fs::write(&path, body)?;
        Ok(Self { path })
    }
}

#[cfg(unix)]
impl Drop for Registration {
    fn drop(&mut self) {
        drop(std::fs::remove_file(&self.path));
    }
}

fn parse_record(path: PathBuf, text: &str) -> Option<Record> {
    let mut lines = text.lines();
    let pid: u32 = lines.next()?.trim().parse().ok()?;
    let socket = lines.next()?.to_owned();
    let root = PathBuf::from(lines.next()?);
    let key = path.file_stem()?.to_str()?.to_owned();
    Some(Record {
        key,
        pid,
        socket,
        root,
        record_path: path,
    })
}

fn load_records() -> Vec<Record> {
    let Ok(entries) = std::fs::read_dir(registry_dir()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("pid") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Some(rec) = parse_record(path, &text) {
            out.push(rec);
        }
    }
    out.sort_by(|a, b| a.root.cmp(&b.root).then_with(|| a.key.cmp(&b.key)));
    out
}

#[cfg(unix)]
fn send_control(socket_name: &str, byte: u8) -> bool {
    let Ok(Some(fd)) = super::unix_ipc::connect_name(socket_name) else {
        return false;
    };
    // SAFETY: fd is a connected AF_UNIX socket owned by this File.
    let mut conn = unsafe { std::fs::File::from_raw_fd(fd) };
    if conn.write_all(&[byte]).is_err() {
        return false;
    }
    let mut ack = [0u8; 4];
    conn.read_exact(&mut ack).is_ok()
}

#[cfg(not(unix))]
fn send_control(_socket_name: &str, _byte: u8) -> bool {
    false
}

fn cleanup_stale(rec: &Record) {
    drop(std::fs::remove_file(&rec.record_path));
    if !cfg!(target_os = "linux") && !rec.socket.is_empty() {
        drop(std::fs::remove_file(&rec.socket));
    }
}

fn matches_filter(rec: &Record, filter: Option<&Path>) -> bool {
    let Some(target) = filter else {
        return true;
    };
    let a = std::fs::canonicalize(&rec.root).unwrap_or_else(|_| rec.root.clone());
    let b = std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    a == b
}

/// Lists active daemons verified over their Unix sockets (optionally filtered to `filter`), pruning stale records.
pub(super) fn list_active(filter: Option<&Path>) -> Vec<Record> {
    let mut active = Vec::new();
    for rec in load_records() {
        if !send_control(&rec.socket, PING_BYTE) {
            cleanup_stale(&rec);
            continue;
        }
        if matches_filter(&rec, filter) {
            active.push(rec);
        }
    }
    active
}

/// Prints active `haste serve` daemons (optionally filtered to `filter`).
pub fn status(filter: Option<&Path>) -> Result<i32> {
    let active = list_active(filter);
    if active.is_empty() {
        println!("no active haste daemons");
        return Ok(0);
    }
    for rec in active {
        println!(
            "pid={} key={} root={}",
            rec.pid,
            rec.key,
            rec.root.display()
        );
    }
    Ok(0)
}

#[cfg(unix)]
fn stop_record(rec: &Record) -> bool {
    if !send_control(&rec.socket, STOP_BYTE) {
        cleanup_stale(rec);
        return false;
    }
    for _ in 0..50 {
        if !send_control(&rec.socket, PING_BYTE) {
            cleanup_stale(rec);
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    false
}

#[cfg(not(unix))]
fn stop_record(_rec: &Record) -> bool {
    false
}

/// Stops active `haste serve` daemons matching `filter` (or all when `filter` is `None`).
pub fn stop(filter: Option<&Path>) -> Result<i32> {
    let active = list_active(filter);
    if active.is_empty() {
        println!("no active haste daemons to stop");
        return Ok(0);
    }
    let mut stopped = 0usize;
    for rec in &active {
        if stop_record(rec) {
            println!("stopped pid={} root={}", rec.pid, rec.root.display());
            stopped += 1;
        } else {
            eprintln!("failed to stop pid={} root={}", rec.pid, rec.root.display());
        }
    }
    Ok(if stopped == active.len() { 0 } else { 1 })
}
