//! The source files that `run --watch` reloads on: one descriptor that turns readable when any of them changes, from kqueue vnode events on macOS and inotify on Linux.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};

/// Whether `--watch` was given.
static WANTED: AtomicBool = AtomicBool::new(false);

/// How long, in milliseconds, the files must stay quiet after a change before it counts, so an editor's write and rename count once.
const QUIET_MS: libc::c_int = 30;

/// The most events one read takes from the queue while it drains.
const DRAIN_EVENTS: usize = 64;

/// Makes `run` watch the workspace's source files and reload the function bodies an edit changes in the running program.
pub fn enable() {
    WANTED.store(true, Ordering::Relaxed);
}

/// Whether `run` watches the workspace's source files.
pub fn enabled() -> bool {
    WANTED.load(Ordering::Relaxed)
}

/// A queue of change events for a list of files.
pub struct Watcher {
    /// The paths watched, opened again after each change.
    paths: Vec<PathBuf>,
    /// The kqueue or inotify descriptor that turns readable on a change.
    queue: OwnedFd,
    /// The open files whose vnodes the kqueue watches; empty on Linux.
    files: Vec<OwnedFd>,
}

impl Watcher {
    /// Watches each file of `paths` that exists now.
    pub fn new(paths: Vec<PathBuf>) -> Result<Watcher> {
        let (queue, files) = open(&paths)?;
        Ok(Watcher { paths, queue, files })
    }

    /// The descriptor to poll for a change.
    pub fn fd(&self) -> RawFd {
        self.queue.as_raw_fd()
    }

    /// Waits until the files stay quiet for `QUIET_MS`, then watches them again from their paths, as a save that renames a new file over an old one ends the watch on the old one.
    pub fn settle(&mut self) -> Result<()> {
        loop {
            drain(&self.queue);
            let mut fd = libc::pollfd { fd: self.queue.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: fd is one live pollfd entry.
            match unsafe { libc::poll(&mut fd, 1, QUIET_MS) } {
                0 => break,
                ready if ready < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted => bail!("poll failed: {}", std::io::Error::last_os_error()),
                _ => {}
            }
        }
        self.files.clear();
        (self.queue, self.files) = open(&self.paths)?;
        Ok(())
    }
}

/// The path as a C string, or none when it holds a NUL byte.
fn c_path(path: &std::path::Path) -> Option<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(path.as_os_str().as_bytes()).ok()
}

/// A kqueue that reports writes, deletes, renames and attribute changes of each file of `paths` that opens, with those open files.
#[cfg(target_os = "macos")]
fn open(paths: &[PathBuf]) -> Result<(OwnedFd, Vec<OwnedFd>)> {
    // SAFETY: kqueue takes no arguments and returns a new descriptor or -1.
    let raw = unsafe { libc::kqueue() };
    if raw < 0 {
        bail!("kqueue: {}", std::io::Error::last_os_error());
    }
    // SAFETY: raw is the new kqueue, which this OwnedFd owns from here on.
    let queue = unsafe { OwnedFd::from_raw_fd(raw) };
    let (mut files, mut changes) = (Vec::new(), Vec::new());
    for path in paths.iter().filter_map(|path| c_path(path)) {
        // SAFETY: path is a NUL-terminated C string, and O_EVTONLY opens the file only for events.
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_EVTONLY | libc::O_CLOEXEC) };
        if fd < 0 {
            continue;
        }
        changes.push(libc::kevent {
            ident: fd as libc::uintptr_t,
            filter: libc::EVFILT_VNODE,
            flags: libc::EV_ADD | libc::EV_CLEAR,
            fflags: libc::NOTE_WRITE | libc::NOTE_EXTEND | libc::NOTE_DELETE | libc::NOTE_RENAME | libc::NOTE_ATTRIB,
            data: 0,
            udata: std::ptr::null_mut(),
        });
        // SAFETY: fd is the file just opened, which this OwnedFd owns from here on.
        files.push(unsafe { OwnedFd::from_raw_fd(fd) });
    }
    // SAFETY: changes is a live array of changes.len() kevent entries, and the call reads no events back.
    if unsafe { libc::kevent(raw, changes.as_ptr(), changes.len() as libc::c_int, std::ptr::null_mut(), 0, std::ptr::null()) } < 0 {
        bail!("kevent: {}", std::io::Error::last_os_error());
    }
    Ok((queue, files))
}

/// Takes every event the kqueue holds, so it reads as quiet until the next change.
#[cfg(target_os = "macos")]
fn drain(queue: &OwnedFd) {
    // SAFETY: kevent is plain data, and all zero bytes, with a null udata, are a valid value.
    let mut events: [libc::kevent; DRAIN_EVENTS] = unsafe { std::mem::zeroed() };
    let now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    loop {
        // SAFETY: events is a live array of DRAIN_EVENTS kevent entries, and now is a zero timeout, so the call does not block.
        let taken = unsafe { libc::kevent(queue.as_raw_fd(), std::ptr::null(), 0, events.as_mut_ptr(), DRAIN_EVENTS as libc::c_int, &now) };
        if taken < DRAIN_EVENTS as libc::c_int {
            break;
        }
    }
}

/// An inotify descriptor that reports writes, attribute changes, deletes and moves of each file of `paths` that exists.
#[cfg(target_os = "linux")]
fn open(paths: &[PathBuf]) -> Result<(OwnedFd, Vec<OwnedFd>)> {
    // SAFETY: inotify_init1 takes plain flags and returns a new descriptor or -1.
    let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
    if raw < 0 {
        bail!("inotify_init1: {}", std::io::Error::last_os_error());
    }
    // SAFETY: raw is the new inotify descriptor, which this OwnedFd owns from here on.
    let queue = unsafe { OwnedFd::from_raw_fd(raw) };
    let mask = libc::IN_MODIFY | libc::IN_CLOSE_WRITE | libc::IN_ATTRIB | libc::IN_DELETE_SELF | libc::IN_MOVE_SELF;
    for path in paths.iter().filter_map(|path| c_path(path)) {
        // SAFETY: raw is a live inotify descriptor and path is a NUL-terminated C string; a file that is gone just fails.
        unsafe { libc::inotify_add_watch(raw, path.as_ptr(), mask) };
    }
    Ok((queue, Vec::new()))
}

/// Reads every event the inotify descriptor holds, so it reads as quiet until the next change.
#[cfg(target_os = "linux")]
fn drain(queue: &OwnedFd) {
    let mut buffer = [0_u8; DRAIN_EVENTS * std::mem::size_of::<libc::inotify_event>()];
    // SAFETY: buffer is live for buffer.len() bytes, and the descriptor does not block, so the loop ends when it is empty.
    while unsafe { libc::read(queue.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) } > 0 {}
}

/// No file watching on this system.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn open(_: &[PathBuf]) -> Result<(OwnedFd, Vec<OwnedFd>)> {
    bail!("--watch needs macOS or Linux")
}

/// No file watching on this system.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn drain(_: &OwnedFd) {}
