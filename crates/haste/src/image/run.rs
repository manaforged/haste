//! Child-process execution and fork/pipe helpers for running JIT code and `#[test]` harnesses.

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};

use super::Image;

const PANIC_MARKER: &[u8] = b"HPNC";

fn invoke_unit_test(address: usize) -> bool {
    extern "C" fn run_fn(ptr: *mut u8) {
        // SAFETY: `ptr` is the native `fn()` code address checked non-zero before invoking.
        let test_fn: extern "C" fn() =
            unsafe { std::mem::transmute::<usize, extern "C" fn()>(ptr as usize) };
        test_fn();
        std::hint::black_box(());
    }
    extern "C" fn caught(_data: *mut u8, _payload: *mut u8) {}
    crate::host::unwind::catch_unwind(run_fn, address as *mut u8, caught) != 0
}

#[cfg(unix)]
unsafe fn dup_above_stdio(fd: i32) -> i32 {
    if fd > libc::STDERR_FILENO {
        return fd;
    }
    // SAFETY: `fd` is a live descriptor in this child process; `F_DUPFD_CLOEXEC` duplicates it to `>= 3`.
    let moved = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, libc::STDERR_FILENO + 1) };
    if moved >= 0 {
        // SAFETY: `fd` was duplicated to `moved` above, so closing `fd` releases only the old slot.
        unsafe { libc::close(fd) };
        moved
    } else {
        fd
    }
}

#[cfg(unix)]
unsafe fn redirect_stdio(mut out_fd: i32, mut err_fd: i32) {
    // SAFETY: `out_fd` and `err_fd` are live distinct pipe write descriptors owned by the child.
    unsafe {
        if out_fd == libc::STDERR_FILENO {
            out_fd = dup_above_stdio(out_fd);
        }
        if err_fd == libc::STDOUT_FILENO {
            err_fd = dup_above_stdio(err_fd);
        }
        if out_fd != libc::STDOUT_FILENO {
            libc::dup2(out_fd, libc::STDOUT_FILENO);
            libc::close(out_fd);
        }
        if err_fd != libc::STDERR_FILENO {
            libc::dup2(err_fd, libc::STDERR_FILENO);
            libc::close(err_fd);
        }
    }
}

#[cfg(unix)]
fn drain_pipes(pipes: &mut [(std::io::PipeReader, &mut Vec<u8>)]) -> std::io::Result<()> {
    use std::io::Read;
    let mut active: Vec<usize> = (0..pipes.len()).collect();
    let mut buf = [0u8; 4096];
    while !active.is_empty() {
        let mut fds: Vec<libc::pollfd> = active
            .iter()
            .map(|&idx| libc::pollfd {
                fd: pipes[idx].0.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            })
            .collect();
        // SAFETY: `fds` points to `fds.len()` initialized `pollfd` structs holding live pipe descriptors.
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) };
        if ready < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        let mut next = Vec::with_capacity(active.len());
        for (pos, pfd) in fds.iter().enumerate() {
            let idx = active[pos];
            if pfd.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) == 0 {
                next.push(idx);
                continue;
            }
            match pipes[idx].0.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    pipes[idx].1.extend_from_slice(&buf[..n]);
                    next.push(idx);
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => next.push(idx),
                Err(err) => return Err(err),
            }
        }
        active = next;
    }
    Ok(())
}

#[cfg(unix)]
struct TestChildPipes {
    panic_r: i32,
    panic_w: i32,
    stdio: Option<((i32, i32), (i32, i32))>,
    sentinel: Option<[u8; crate::host::unwind::PANIC_SENTINEL_LEN]>,
}

#[cfg(unix)]
fn panic_sentinel() -> [u8; crate::host::unwind::PANIC_SENTINEL_LEN] {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::AtomicU64;
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut hi_hasher = RandomState::new().build_hasher();
    hi_hasher.write_u64(seq);
    let mut lo_hasher = RandomState::new().build_hasher();
    lo_hasher.write_u64(seq ^ 0x9e37_79b9_7f4a_7c15);
    let (hi, lo) = (hi_hasher.finish(), lo_hasher.finish());
    let mut out = [0u8; crate::host::unwind::PANIC_SENTINEL_LEN];
    out[..6].copy_from_slice(b"\x00HPNC:");
    let hex = b"0123456789abcdef";
    for i in 0..16 {
        let shift = (15 - i) * 4;
        out[6 + i] = hex[((hi >> shift) & 0xf) as usize];
        out[22 + i] = hex[((lo >> shift) & 0xf) as usize];
    }
    out[38] = b':';
    out[39] = 0;
    out
}

#[cfg(unix)]
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(unix)]
fn rfind_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

#[cfg(unix)]
fn extract_panic_msg(stderr: &[u8], sentinel: &[u8]) -> Option<String> {
    let end = rfind_subslice(stderr, sentinel)?;
    let before = &stderr[..end];
    let start = rfind_subslice(before, sentinel).map_or(0, |pos| pos + sentinel.len());
    let text = String::from_utf8_lossy(&before[start..]);
    crate::cli::panic_payload(&text).map(str::to_owned)
}

#[cfg(unix)]
fn strip_sentinel(buf: &mut Vec<u8>, sentinel: &[u8]) {
    if sentinel.is_empty() || find_subslice(buf, sentinel).is_none() {
        return;
    }
    let mut cleaned = Vec::with_capacity(buf.len());
    let mut rest = buf.as_slice();
    while let Some(pos) = find_subslice(rest, sentinel) {
        cleaned.extend_from_slice(&rest[..pos]);
        rest = &rest[pos + sentinel.len()..];
    }
    cleaned.extend_from_slice(rest);
    *buf = cleaned;
}

#[cfg(unix)]
fn combine_test_output(mut stdout: Vec<u8>, stderr: &[u8]) -> String {
    if !stdout.is_empty()
        && !stderr.is_empty()
        && !stdout.ends_with(b"\n")
        && !stderr.starts_with(b"\n")
    {
        stdout.push(b'\n');
    }
    stdout.extend_from_slice(stderr);
    String::from_utf8_lossy(&stdout).into_owned()
}

impl Image {
    #[cfg(unix)]
    fn run_test_child(
        &self,
        (index, start, address): (usize, Option<usize>, usize),
        (name, env): (&str, &[(String, String)]),
        pipes: TestChildPipes,
    ) -> i32 {
        // SAFETY: read ends belong only to the parent; write ends belong to this child until moved or closed.
        let mut panic_file = unsafe {
            libc::close(pipes.panic_r);
            if let Some(((or, _), (er, _))) = pipes.stdio {
                libc::close(or);
                libc::close(er);
            }
            let pw = dup_above_stdio(pipes.panic_w);
            std::fs::File::from_raw_fd(pw)
        };
        for (key, value) in env {
            // SAFETY: single-threaded forked child process before invoking test function.
            unsafe { std::env::set_var(key, value) };
        }
        crate::host::unwind::set_jit_unwind_head(self.unwind_head);
        if let Some(sentinel) = pipes.sentinel {
            crate::host::unwind::set_panic_stderr_sentinel(sentinel);
        }
        if let Some(((_, ow), (_, ew))) = pipes.stdio {
            // SAFETY: `ow` and `ew` are live pipe write descriptors belonging to this child.
            unsafe { redirect_stdio(ow, ew) };
        }
        let (panicked, status) = if let Some(start) = start {
            let status = self.enter_main((start, index), &[name.to_owned()]);
            (status == 101, status)
        } else if invoke_unit_test(address) {
            (true, 101)
        } else {
            (false, 0)
        };
        if panicked {
            drop(std::io::Write::write_all(&mut panic_file, PANIC_MARKER));
        }
        drop(panic_file);
        status
    }

    /// Runs each case `(index, name)`, a reached `fn(i64) -> i64`, on each input and prints `name input value` lines, all in one forked child; a case that crashes gets its status and the cases after it run in a new child.
    #[cfg(feature = "diff-harness")]
    pub fn run_cases(&self, cases: &[(usize, &str)], inputs: &[i64]) -> Result<Vec<i32>> {
        self.run_batch(cases, false, |_, address, name| {
            use std::io::Write;
            let mut out = std::io::stdout().lock();
            // SAFETY: address is code that the encoder wrote for a native fn(i64) -> i64 signature, checked to be set before the fork.
            let case: extern "C" fn(i64) -> i64 =
                unsafe { std::mem::transmute::<usize, extern "C" fn(i64) -> i64>(address) };
            for &input in inputs {
                writeln!(out, "{name} {input} {}", case(input))?;
                out.flush()?;
            }
            Ok(())
        })
    }

    /// Runs a single `#[test]` function in a forked child, capturing stdout and stderr unless `nocapture` is true.
    #[cfg(unix)]
    pub fn run_test_captured(
        &self,
        index: usize,
        start: Option<usize>,
        name: &str,
        env: &[(String, String)],
        nocapture: bool,
    ) -> Result<crate::cli::TestRun> {
        let address = self.slot(index).load(Ordering::Acquire);
        if address == 0 {
            bail!("{name} has no code");
        }
        if let Some(s) = start
            && self.slot(s).load(Ordering::Acquire) == 0
        {
            bail!("{name} has no start wrapper (slot {s})");
        }
        let (panic_read, panic_write) = std::io::pipe()?;
        let stdio = (!nocapture)
            .then(|| Ok::<_, std::io::Error>((std::io::pipe()?, std::io::pipe()?)))
            .transpose()?;
        let sentinel = (!nocapture).then(panic_sentinel);
        let pipes = TestChildPipes {
            panic_r: panic_read.as_raw_fd(),
            panic_w: panic_write.as_raw_fd(),
            stdio: stdio.as_ref().map(|((or, ow), (er, ew))| {
                (
                    (or.as_raw_fd(), ow.as_raw_fd()),
                    (er.as_raw_fd(), ew.as_raw_fd()),
                )
            }),
            sentinel,
        };
        let pid = fork_child(|| self.run_test_child((index, start, address), (name, env), pipes))?;
        drop(panic_write);
        let (mut panic_buf, mut stdout_buf, mut stderr_buf) = (Vec::new(), Vec::new(), Vec::new());
        if let Some(((out_read, out_write), (err_read, err_write))) = stdio {
            drop(out_write);
            drop(err_write);
            drain_pipes(&mut [
                (panic_read, &mut panic_buf),
                (out_read, &mut stdout_buf),
                (err_read, &mut stderr_buf),
            ])?;
        } else {
            drain_pipes(&mut [(panic_read, &mut panic_buf)])?;
        }
        let raw_status = wait(pid);
        let panicked = panic_buf == PANIC_MARKER;
        let panic_msg = match (panicked, &sentinel) {
            (true, Some(marker)) => extract_panic_msg(&stderr_buf, marker),
            _ => None,
        };
        if let Some(marker) = &sentinel {
            strip_sentinel(&mut stderr_buf, marker);
        }
        let status = if panicked {
            crate::cli::TEST_PANICKED_STATUS
        } else {
            raw_status
        };
        let combined = combine_test_output(stdout_buf, &stderr_buf);
        Ok(crate::cli::TestRun {
            status,
            output: (!combined.is_empty()).then_some(combined),
            panic_msg,
        })
    }

    /// Runs a single `#[test]` function in-process on non-Unix platforms.
    #[cfg(not(unix))]
    pub fn run_test_captured(
        &self,
        index: usize,
        start: Option<usize>,
        name: &str,
        _env: &[(String, String)],
        _nocapture: bool,
    ) -> Result<crate::cli::TestRun> {
        let address = self.slot(index).load(Ordering::Acquire);
        if address == 0 {
            bail!("{name} has no code");
        }
        if let Some(s) = start {
            if self.slot(s).load(Ordering::Acquire) == 0 {
                bail!("{name} has no start wrapper (slot {s})");
            }
            let status = self.enter_main((s, index), &[name.to_owned()]);
            let status = if status == 101 {
                crate::cli::TEST_PANICKED_STATUS
            } else {
                status
            };
            return Ok(crate::cli::TestRun {
                status,
                output: None,
                panic_msg: None,
            });
        }
        let status = if invoke_unit_test(address) {
            crate::cli::TEST_PANICKED_STATUS
        } else {
            0
        };
        Ok(crate::cli::TestRun {
            status,
            output: None,
            panic_msg: None,
        })
    }

    #[cfg(feature = "diff-harness")]
    fn run_batch(
        &self,
        items: &[(usize, &str)],
        quiet_stderr: bool,
        invoke: impl Fn(usize, usize, &str) -> std::io::Result<()>,
    ) -> Result<Vec<i32>> {
        if let Some((_, name)) = items
            .iter()
            .find(|(index, _)| self.slot(*index).load(Ordering::Acquire) == 0)
        {
            bail!("{name} has no code");
        }
        let mut statuses = vec![0; items.len()];
        let mut from = 0;
        while from < items.len() {
            let (done, status) = self.run_from(from, &items[from..], quiet_stderr, &invoke)?;
            if status == 0 && done >= items.len() - from {
                break;
            }
            let crashed = (from + done).min(items.len() - 1);
            statuses[crashed] = status;
            from = crashed + 1;
        }
        Ok(statuses)
    }

    #[cfg(all(unix, feature = "diff-harness"))]
    fn run_from(
        &self,
        offset: usize,
        items: &[(usize, &str)],
        quiet_stderr: bool,
        invoke: &impl Fn(usize, usize, &str) -> std::io::Result<()>,
    ) -> Result<(usize, i32)> {
        let (pid, mut progress) = fork_piped(|progress| {
            if quiet_stderr {
                silence_stderr();
            }
            self.run_in_child(offset, items, invoke, progress)
        })?;
        let mut done = Vec::new();
        std::io::Read::read_to_end(&mut progress, &mut done)?;
        Ok((done.len(), wait(pid)))
    }

    #[cfg(all(not(unix), feature = "diff-harness"))]
    fn run_from(
        &self,
        offset: usize,
        items: &[(usize, &str)],
        _quiet_stderr: bool,
        invoke: &impl Fn(usize, usize, &str) -> std::io::Result<()>,
    ) -> Result<(usize, i32)> {
        for (i, &(index, name)) in items.iter().enumerate() {
            let address = self.slot(index).load(Ordering::Acquire);
            if invoke(offset + i, address, name).is_err() {
                return Ok((i, 101));
            }
        }
        Ok((items.len(), 0))
    }

    #[cfg(all(unix, feature = "diff-harness"))]
    fn run_in_child(
        &self,
        offset: usize,
        items: &[(usize, &str)],
        invoke: &impl Fn(usize, usize, &str) -> std::io::Result<()>,
        mut progress: std::fs::File,
    ) -> std::io::Result<()> {
        use std::io::Write;
        crate::host::unwind::set_jit_unwind_head(self.unwind_head);
        for (i, &(index, name)) in items.iter().enumerate() {
            let address = self.slot(index).load(Ordering::Acquire);
            invoke(offset + i, address, name)?;
            progress.write_all(&[1])?;
        }
        Ok(())
    }

    /// Runs the program the way the native entry point does: std's `lang_start` with `main`, `args` as argv and the default SIGPIPE handling; returns its exit status.
    pub(super) fn enter_main(&self, (start, main): (usize, usize), args: &[String]) -> i32 {
        crate::host::unwind::set_jit_unwind_head(self.unwind_head);
        let (start, main) = (
            self.slot(start).load(Ordering::Acquire),
            self.slot(main).load(Ordering::Acquire),
        );
        let Ok(owned) = args
            .iter()
            .map(|arg| std::ffi::CString::new(arg.as_str()))
            .collect::<Result<Vec<_>, _>>()
        else {
            return 101;
        };
        let mut argv: Vec<*const u8> = owned.iter().map(|arg| arg.as_ptr().cast()).collect();
        argv.push(std::ptr::null());
        #[cfg(target_vendor = "apple")]
        let _args_guard = crate::host::apple_args::Guard::install(
            owned.len() as libc::c_int,
            argv.as_mut_ptr().cast(),
        );
        // SAFETY: start is code that the encoder wrote, or a stub that compiles it, for `lang_start::<T>(fn() -> T, isize, *const *const u8, u8) -> isize`, and main for the `fn() -> T` it takes.
        let lang_start: extern "C" fn(usize, isize, *const *const u8, u8) -> isize =
            unsafe { std::mem::transmute::<usize, _>(start) };
        let status = lang_start(main, owned.len() as isize, argv.as_ptr(), 0) as i32;
        crate::prep::finish();
        status
    }
}

#[cfg(all(unix, feature = "diff-harness"))]
fn silence_stderr() {
    // SAFETY: c"/dev/null" is a static NUL-terminated path, and O_WRONLY | O_CLOEXEC takes no mode argument.
    let null = unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
    if null >= 0 && null != libc::STDERR_FILENO {
        // SAFETY: null is a valid open descriptor distinct from STDERR_FILENO; dup2 redirects child stderr and close releases the duplicate.
        unsafe {
            libc::dup2(null, libc::STDERR_FILENO);
            libc::close(null);
        }
    }
}

/// Waits for child `pid` and gives its exit status, or 128 plus the signal that ended it.
#[cfg(unix)]
pub(crate) fn wait(pid: libc::pid_t) -> i32 {
    let mut status = 0;
    // SAFETY: pid is a child of this process, and status is a live local.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        128 + libc::WTERMSIG(status)
    }
}

/// Whether a child that ended with `status` was stopped from outside, by a hangup, an interrupt, a kill or a termination request, so its failure says nothing about its input.
#[cfg(unix)]
pub(crate) fn interrupted(status: i32) -> bool {
    [libc::SIGHUP, libc::SIGINT, libc::SIGKILL, libc::SIGTERM]
        .iter()
        .any(|&signal| status == 128 + signal)
}

/// Whether a child that ended with `status` was stopped from outside.
#[cfg(not(unix))]
pub(crate) fn interrupted(_status: i32) -> bool {
    false
}

/// Forks a child that dumps no core when a signal ends it, runs `child` and exits with the status it returns; returns the child's pid.
#[cfg(unix)]
pub(crate) fn fork_child(child: impl FnOnce() -> i32) -> Result<libc::pid_t> {
    std::io::Write::flush(&mut std::io::stdout())?;
    // SAFETY: stdout is flushed, no active scope jobs run during fork (any worker pool threads are parked idle and forked children bypass the pool via `std::process::id() != self.pid`), and the child exits via `_exit`.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        bail!("fork failed: {}", std::io::Error::last_os_error());
    }
    if pid == 0 {
        super::statics::reset_after_fork();
        #[cfg(target_os = "linux")]
        // SAFETY: PR_SET_DUMPABLE with 0 only clears this child's dumpable flag, so a crash dumps no core of the whole compiler.
        unsafe {
            libc::prctl(libc::PR_SET_DUMPABLE, 0)
        };
        let status = child();
        // SAFETY: _exit ends the child without running the parent's destructors.
        unsafe { libc::_exit(status) }
    }
    Ok(pid)
}

/// A new pipe whose ends close when the program executes another one.
#[cfg(unix)]
pub(crate) fn pipe() -> Result<[i32; 2]> {
    use std::os::fd::IntoRawFd;
    let (read, write) = std::io::pipe()?;
    Ok([read.into_raw_fd(), write.into_raw_fd()])
}

/// Marks `fd` to close when the program executes another one.
#[cfg(unix)]
pub(crate) fn close_on_exec(fd: i32) -> Result<i32> {
    // SAFETY: fcntl with F_SETFD only changes the flags of a descriptor this process holds.
    if unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
        bail!("fcntl: {}", std::io::Error::last_os_error());
    }
    Ok(fd)
}

/// Forks a child that runs `child` on the write end of a new pipe and exits with status 1 when it fails, and returns its pid and the read end.
#[cfg(all(unix, feature = "diff-harness"))]
pub(crate) fn fork_piped<E>(
    child: impl FnOnce(std::fs::File) -> Result<(), E>,
) -> Result<(libc::pid_t, std::fs::File)> {
    let [read, write] = pipe()?;
    let pid = fork_child(|| {
        // SAFETY: read is the end this child does not use, and write is the end it owns from here on.
        let out = unsafe {
            libc::close(read);
            std::fs::File::from_raw_fd(write)
        };
        i32::from(child(out).is_err())
    })?;
    // SAFETY: write is the end that only the child uses, and read is the end this process owns from here on.
    let input = unsafe {
        libc::close(write);
        std::fs::File::from_raw_fd(read)
    };
    Ok((pid, input))
}
