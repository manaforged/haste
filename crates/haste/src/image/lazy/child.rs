#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

use anyhow::{Result, bail};

use crate::image::Image;
#[cfg(unix)]
use crate::image::{stub, wait};
use crate::names::ItemRef;

use super::{Ended, Jit};

#[cfg(unix)]
pub(super) struct Child {
    pid: libc::pid_t,
    asks: Option<std::fs::File>,
    answers: std::fs::File,
    out: Option<std::fs::File>,
    bytes: Vec<u8>,
    pub(super) failure: Option<(String, ItemRef)>,
}

#[cfg(not(unix))]
pub(super) struct Child {
    bytes: Vec<u8>,
    pub(super) failure: Option<(String, ItemRef)>,
}

#[cfg(unix)]
fn high_fd_base() -> i32 {
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `rl` is a valid stack-allocated `rlimit`.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) } != 0 {
        return 512;
    }
    if rl.rlim_cur < 1024 {
        let target = rl.rlim_max.min(1024);
        if target > rl.rlim_cur {
            rl.rlim_cur = target;
            // SAFETY: raising soft `RLIMIT_NOFILE` within `rlim_max` in the forked child.
            let _ = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rl) };
        }
    }
    (rl.rlim_cur.min(1024) as i32 - 32).max(256)
}

#[cfg(unix)]
fn relocate_fd(fd: i32, min_fd: i32) -> i32 {
    // SAFETY: `fd` is an open pipe fd owned by the forked child; `F_DUPFD_CLOEXEC` duplicates it above low user fds.
    let moved = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, min_fd) };
    if moved >= 0 {
        // SAFETY: `moved` succeeded, so the original low `fd` is no longer needed.
        unsafe { libc::close(fd) };
        moved
    } else {
        fd
    }
}

impl Jit {
    /// Forks a child that runs `body` on the image, with an output pipe when `piped`, and exits with the status `body` returns; returns the child's id.
    #[cfg(unix)]
    pub fn fork(
        &mut self,
        piped: bool,
        body: impl FnOnce(&Image, Option<std::fs::File>) -> i32,
    ) -> Result<usize> {
        let Some(stubs) = self.image.stubs.span() else {
            bail!("no stub memory");
        };
        let (ask, answer) = (crate::image::pipe()?, crate::image::pipe()?);
        let out = if piped {
            Some(crate::image::pipe()?)
        } else {
            None
        };
        let image = &self.image;
        let pid = crate::image::fork_child(|| {
            // SAFETY: the child closes the ends that only the compiler uses and relocates its own pipe ends above low user fds.
            let (child_ask, child_answer, out) = unsafe {
                libc::close(ask[0]);
                libc::close(answer[1]);
                let min_fd = high_fd_base();
                let child_ask = relocate_fd(ask[1], min_fd);
                let child_answer = relocate_fd(answer[0], min_fd);
                let out = out.map(|[read, write]| {
                    libc::close(read);
                    std::fs::File::from_raw_fd(relocate_fd(write, min_fd))
                });
                (child_ask, child_answer, out)
            };
            stub::connect(image.table, stubs, (child_ask, child_answer));
            crate::host::unwind::set_jit_unwind_head(image.unwind_head);
            body(image, out)
        })?;
        // SAFETY: the parent owns ask[0], answer[1] and the read end of the output pipe from here on, and closes the ends that only the child uses.
        let child = unsafe {
            libc::close(ask[1]);
            libc::close(answer[0]);
            let out = out.map(|[read, write]| {
                libc::close(write);
                std::fs::File::from_raw_fd(read)
            });
            Child {
                pid,
                asks: Some(std::fs::File::from_raw_fd(ask[0])),
                answers: std::fs::File::from_raw_fd(answer[1]),
                out,
                bytes: Vec::new(),
                failure: None,
            }
        };
        self.children.push(Some(child));
        Ok(self.children.len() - 1)
    }

    /// Forks a child that runs `body` on the image, with an output pipe when `piped`, and exits with the status `body` returns; returns the child's id.
    #[cfg(not(unix))]
    pub fn fork(
        &mut self,
        _piped: bool,
        _body: impl FnOnce(&Image, Option<std::fs::File>) -> i32,
    ) -> Result<usize> {
        bail!("lazy forked JIT execution requires Unix; use `haste build -o` on Windows")
    }

    pub(super) fn kill_children(&mut self) {
        self.pending.clear();
        #[cfg(unix)]
        for child in self.children.drain(..).flatten() {
            // SAFETY: pid is a child this process forked and has not reaped, so the signal reaches only that child.
            unsafe { libc::kill(child.pid, libc::SIGKILL) };
            wait(child.pid);
        }
        #[cfg(not(unix))]
        self.children.clear();
    }

    #[cfg(unix)]
    pub(super) fn ready(&mut self) -> Result<Vec<(usize, bool)>> {
        let mut polled = Vec::new();
        let mut fds = Vec::new();
        for (id, child) in self
            .children
            .iter()
            .enumerate()
            .filter_map(|(id, child)| Some((id, child.as_ref()?)))
        {
            for (file, asking) in [(child.asks.as_ref(), true), (child.out.as_ref(), false)] {
                if let Some(file) = file {
                    polled.push((id, asking));
                    fds.push(libc::pollfd {
                        fd: file.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    });
                }
            }
        }
        if let Some(watcher) = &self.watcher {
            fds.push(libc::pollfd {
                fd: watcher.fd(),
                events: libc::POLLIN,
                revents: 0,
            });
        }
        // SAFETY: fds is a live array of fds.len() pollfd entries.
        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                return Ok(Vec::new());
            }
            bail!("poll failed: {error}");
        }
        self.edited |= self.watcher.is_some() && fds.last().is_some_and(|fd| fd.revents != 0);
        Ok(polled
            .into_iter()
            .zip(fds)
            .filter(|(_, fd)| fd.revents != 0)
            .map(|(ready, _)| ready)
            .collect())
    }

    #[cfg(not(unix))]
    pub(super) fn ready(&mut self) -> Result<Vec<(usize, bool)>> {
        Ok(Vec::new())
    }

    #[cfg(unix)]
    pub(super) fn drain(&mut self, id: usize) {
        let Some(child) = self.children[id].as_mut() else {
            return;
        };
        let mut buffer = [0; 1 << 16];
        match child.out.as_mut().map(|out| out.read(&mut buffer)) {
            Some(Ok(read)) if read > 0 => child.bytes.extend_from_slice(&buffer[..read]),
            _ => child.out = None,
        }
    }

    #[cfg(not(unix))]
    pub(super) fn drain(&mut self, _id: usize) {}

    #[cfg(unix)]
    pub(super) fn read_ask(&mut self, id: usize) -> Option<usize> {
        let child = self.children[id].as_mut()?;
        let mut index = [0; 8];
        match child.asks.as_mut()?.read_exact(&mut index) {
            Ok(()) => Some(u64::from_le_bytes(index) as usize),
            Err(_) => {
                child.asks = None;
                None
            }
        }
    }

    #[cfg(not(unix))]
    pub(super) fn read_ask(&mut self, _id: usize) -> Option<usize> {
        None
    }

    #[cfg(unix)]
    pub(super) fn reply(&mut self, id: usize, code: usize) {
        let Some(child) = self.children[id].as_mut() else {
            return;
        };
        if child
            .answers
            .write_all(&(code as u64).to_le_bytes())
            .is_err()
        {
            child.asks = None;
        }
    }

    #[cfg(not(unix))]
    pub(super) fn reply(&mut self, _id: usize, _code: usize) {}

    #[cfg(unix)]
    pub(super) fn reap(&mut self) -> Vec<(usize, Ended)> {
        let mut ended = Vec::new();
        for (id, slot) in self.children.iter_mut().enumerate() {
            if slot
                .as_ref()
                .is_some_and(|child| child.asks.is_none() && child.out.is_none())
            {
                let child = slot.take().unwrap_or_else(|| unreachable!());
                ended.push((
                    id,
                    Ended {
                        status: wait(child.pid),
                        bytes: child.bytes,
                        failure: child.failure,
                    },
                ));
            }
        }
        ended
    }

    #[cfg(not(unix))]
    pub(super) fn reap(&mut self) -> Vec<(usize, Ended)> {
        Vec::new()
    }
}
