//! Child-process execution and fork/pipe helpers for running JIT code and `#[test]` harnesses.

#[cfg(unix)]
use std::os::fd::FromRawFd;
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};

use super::Image;

impl Image {
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

    /// Runs a single `#[test]` function in a forked child with stderr captured, returning `(exit_status, stderr)`.
    #[cfg(unix)]
    pub fn run_test_captured(
        &self,
        index: usize,
        start: Option<usize>,
        name: &str,
    ) -> Result<(i32, String)> {
        let address = self.slot(index).load(Ordering::Acquire);
        if address == 0 {
            bail!("{name} has no code");
        }
        if let Some(s) = start
            && self.slot(s).load(Ordering::Acquire) == 0
        {
            bail!("{name} has no start wrapper (slot {s})");
        }
        let (pid, mut stderr_pipe) = fork_piped(|writer| {
            crate::host::unwind::set_jit_unwind_head(self.unwind_head);
            let raw = std::os::fd::AsRawFd::as_raw_fd(&writer);
            if raw != libc::STDERR_FILENO {
                // SAFETY: raw is a live pipe write fd distinct from STDERR_FILENO; dup2 redirects child stderr to it before running the test.
                unsafe {
                    libc::dup2(raw, libc::STDERR_FILENO);
                }
                drop(writer);
            } else {
                std::mem::forget(writer);
            }
            if let Some(start) = start {
                let status = self.enter_main((start, index), &[name.to_owned()]);
                if status != 0 {
                    std::process::exit(status);
                }
            } else {
                // SAFETY: address is code that the encoder wrote for a native fn() signature, checked to be set before the fork.
                let test_fn: extern "C" fn() =
                    unsafe { std::mem::transmute::<usize, extern "C" fn()>(address) };
                test_fn();
            }
            Ok::<(), std::io::Error>(())
        })?;
        let mut stderr = Vec::new();
        std::io::Read::read_to_end(&mut stderr_pipe, &mut stderr)?;
        Ok((wait(pid), String::from_utf8_lossy(&stderr).into_owned()))
    }

    /// Runs a single `#[test]` function in-process on non-Unix platforms, returning `(exit_status, stderr)`.
    #[cfg(not(unix))]
    pub fn run_test_captured(
        &self,
        index: usize,
        start: Option<usize>,
        name: &str,
    ) -> Result<(i32, String)> {
        let address = self.slot(index).load(Ordering::Acquire);
        if address == 0 {
            bail!("{name} has no code");
        }
        if let Some(s) = start {
            if self.slot(s).load(Ordering::Acquire) == 0 {
                bail!("{name} has no start wrapper (slot {s})");
            }
            return Ok((
                self.enter_main((s, index), &[name.to_owned()]),
                String::new(),
            ));
        }
        extern "C" fn run_fn(ptr: *mut u8) {
            // SAFETY: ptr is the native fn() code address checked non-zero above.
            let test_fn: extern "C" fn() =
                unsafe { std::mem::transmute::<usize, extern "C" fn()>(ptr as usize) };
            test_fn();
        }
        extern "C" fn caught(_data: *mut u8, _payload: *mut u8) {}
        let panicked = crate::host::unwind::catch_unwind(run_fn, address as *mut u8, caught) != 0;
        Ok((if panicked { 101 } else { 0 }, String::new()))
    }

    /// Runs each `#[test]` function `(index, start_index, name)` in a forked child and returns each test's exit status (0 for pass, non-zero for panic/Err).
    pub fn run_tests(&self, tests: &[(usize, Option<usize>, &str)]) -> Result<Vec<i32>> {
        let pairs: Vec<(usize, &str)> = tests.iter().map(|&(idx, _, name)| (idx, name)).collect();
        if let Some((_, Some(start), name)) = tests
            .iter()
            .find(|(_, start, _)| start.is_some_and(|s| self.slot(s).load(Ordering::Acquire) == 0))
        {
            bail!("{name} has no start wrapper (slot {start})");
        }
        let starts: Vec<Option<usize>> = tests.iter().map(|&(_, start, _)| start).collect();
        self.run_batch(&pairs, true, |pos, address, name| {
            if let Some(start) = starts[pos] {
                let status = self.enter_main((start, pairs[pos].0), &[name.to_owned()]);
                if status != 0 {
                    std::process::exit(status);
                }
            } else {
                // SAFETY: address is code that the encoder wrote for a native fn() signature, checked to be set before the fork.
                let test_fn: extern "C" fn() =
                    unsafe { std::mem::transmute::<usize, extern "C" fn()>(address) };
                test_fn();
            }
            Ok(())
        })
    }

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

    #[cfg(unix)]
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

    #[cfg(not(unix))]
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

    #[cfg(unix)]
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

#[cfg(unix)]
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
#[cfg(unix)]
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
