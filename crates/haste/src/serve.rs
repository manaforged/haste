//! A resident build of one workspace that answers `build`, `test` and `run` requests over an abstract Unix socket, which leaves no file behind.

use std::io::{Read, Write};
use std::os::fd::{FromRawFd, RawFd};
use std::time::Instant;

use anyhow::{Result, bail};
use crate::graph::{self, Targets};

use crate::build::Build;

/// What a client asks the server to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    /// Bring the build up to date.
    Build,
    /// Bring the build up to date and run every case.
    Test,
    /// Bring the build up to date and run the binary's `main`.
    Run,
}

const MODES: [(Request, &str); 3] = [(Request::Build, "build"), (Request::Test, "test"), (Request::Run, "run")];

impl Request {
    fn byte(self) -> u8 {
        self as u8
    }

    /// The request a command-line mode names, if it names one.
    pub fn parse_mode(mode: &str) -> Option<Request> {
        MODES.into_iter().find(|(_, name)| *name == mode).map(|(request, _)| request)
    }

    fn parse(byte: u8) -> Option<Request> {
        MODES.into_iter().find(|(request, _)| request.byte() == byte).map(|(request, _)| request)
    }
}

fn address(root: &std::path::Path) -> (libc::sockaddr_un, libc::socklen_t) {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    root.hash(&mut hasher);
    let name = format!("haste/{:016x}", hasher.finish());
    // SAFETY: sockaddr_un is plain data, and all zero bytes are a valid value.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, byte) in addr.sun_path[1..].iter_mut().zip(name.bytes()) {
        *slot = byte as libc::c_char;
    }
    let len = std::mem::offset_of!(libc::sockaddr_un, sun_path) + 1 + name.len();
    (addr, len as libc::socklen_t)
}

fn socket() -> Result<RawFd> {
    // SAFETY: socket takes plain integers and returns a new descriptor or -1.
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        bail!("socket: {}", std::io::Error::last_os_error());
    }
    Ok(fd)
}

/// Sends `request` and `args` with this process's stdout and stderr to the server of `root`, waits for it, and returns its status, or none when no server runs.
pub fn ask(root: &std::path::Path, request: Request, args: &[String]) -> Result<Option<i32>> {
    let fd = socket()?;
    let (addr, len) = address(root);
    // SAFETY: addr is a live sockaddr_un of length len, and fd is the socket made above.
    if unsafe { libc::connect(fd, (&raw const addr).cast(), len) } != 0 {
        // SAFETY: fd is the socket made above, which nothing else holds.
        unsafe { libc::close(fd) };
        return Ok(None);
    }
    // SAFETY: fd is a connected socket that this File owns from here on.
    let mut conn = unsafe { std::fs::File::from_raw_fd(fd) };
    std::io::stdout().flush()?;
    send(fd, request.byte(), &[1, 2])?;
    write_args(&mut conn, args)?;
    let mut status = [0; 4];
    conn.read_exact(&mut status)?;
    Ok(Some(i32::from_le_bytes(status)))
}

/// Builds `workspace` once, then answers requests until the process is killed.
pub fn serve(workspace: &graph::Workspace, targets: &Targets, root: &std::path::Path) -> Result<()> {
    let listener = listen(root)?;
    let started = Instant::now();
    let mut build = Some(Build::cold(workspace, targets)?);
    eprintln!("serving {} after {:.1} ms", root.display(), started.elapsed().as_secs_f64() * 1e3);
    loop {
        // SAFETY: listener is a listening socket; accept returns a new descriptor or -1.
        let fd = unsafe { libc::accept4(listener, std::ptr::null_mut(), std::ptr::null_mut(), libc::SOCK_CLOEXEC) };
        if fd < 0 {
            continue;
        }
        // SAFETY: fd is the accepted connection, which this File owns from here on.
        let mut conn = unsafe { std::fs::File::from_raw_fd(fd) };
        let status = match receive(fd) {
            Ok((byte, fds)) => match read_args(&mut conn) {
                Ok(args) => answer(&mut build, (workspace, targets), Request::parse(byte).map(|request| (request, args)), &fds),
                Err(_) => 1,
            },
            Err(_) => 1,
        };
        let _ = conn.write_all(&status.to_le_bytes());
    }
}

fn listen(root: &std::path::Path) -> Result<RawFd> {
    let fd = socket()?;
    let (addr, len) = address(root);
    // SAFETY: addr is a live sockaddr_un of length len, and fd is the socket made above.
    if unsafe { libc::bind(fd, (&raw const addr).cast(), len) } != 0 || unsafe { libc::listen(fd, 16) } != 0 {
        bail!("a build server for {} runs already: {}", root.display(), std::io::Error::last_os_error());
    }
    Ok(fd)
}

fn write_args(conn: &mut std::fs::File, args: &[String]) -> Result<()> {
    let mut bytes = (args.len() as u32).to_le_bytes().to_vec();
    for arg in args {
        bytes.extend((arg.len() as u32).to_le_bytes());
        bytes.extend(arg.as_bytes());
    }
    Ok(conn.write_all(&bytes)?)
}

fn read_args(conn: &mut std::fs::File) -> Result<Vec<String>> {
    let mut word = [0; 4];
    let mut next = |conn: &mut std::fs::File| -> Result<u32> {
        conn.read_exact(&mut word)?;
        Ok(u32::from_le_bytes(word))
    };
    let count = next(conn)?;
    let mut args = Vec::new();
    for _ in 0..count {
        let mut arg = vec![0; next(conn)? as usize];
        conn.read_exact(&mut arg)?;
        args.push(String::from_utf8(arg)?);
    }
    Ok(args)
}

fn answer(build: &mut Option<Build>, (workspace, targets): (&graph::Workspace, &Targets), request: Option<(Request, Vec<String>)>, fds: &[RawFd]) -> i32 {
    let [out, err] = fds else {
        close_all(fds);
        return 1;
    };
    let saved = redirect(*out, *err);
    let started = Instant::now();
    let done = match request {
        Some((request, args)) => current(build.take(), workspace, targets).and_then(|fresh| finish(build.insert(fresh), request, &args, started)),
        None => Err(anyhow::anyhow!("unknown request")),
    };
    if let Err(error) = &done {
        eprintln!("error: {error:#}");
    }
    restore(saved, fds);
    done.unwrap_or(1)
}

/// Prints how long building took since `started`, runs every case for `cases` and `main` with `args` for `run`, and returns the exit status.
pub fn finish(build: &mut Build, request: Request, args: &[String], started: Instant) -> Result<i32> {
    let built = started.elapsed();
    match request {
        Request::Test => build.report(started, built).map(|()| 0),
        Request::Build => {
            eprintln!("built in {:.1} ms", built.as_secs_f64() * 1e3);
            Ok(0)
        }
        Request::Run => {
            eprintln!("built in {:.1} ms", built.as_secs_f64() * 1e3);
            build.run(args)
        }
    }
}

fn redirect(out: RawFd, err: RawFd) -> [RawFd; 2] {
    // SAFETY: dup and dup2 take descriptors this process holds; 1 and 2 stay open throughout.
    unsafe {
        let saved = [libc::dup(1), libc::dup(2)];
        libc::dup2(out, 1);
        libc::dup2(err, 2);
        saved
    }
}

fn restore(saved: [RawFd; 2], received: &[RawFd]) {
    let _ = (std::io::stdout().flush(), std::io::stderr().flush());
    // SAFETY: saved holds the duplicates of this process's own stdout and stderr.
    unsafe {
        libc::dup2(saved[0], 1);
        libc::dup2(saved[1], 2);
    }
    close_all(&saved);
    close_all(received);
}

fn close_all(fds: &[RawFd]) {
    // SAFETY: fds are descriptors this process owns and nothing else uses after this call.
    fds.iter().for_each(|&fd| _ = unsafe { libc::close(fd) });
}

/// Returns `build` brought up to date with the member files, or a build from nothing when there is none or the refresh fails, so a failed refresh never leaves a half-updated build behind.
fn current(build: Option<Build>, workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
    match build {
        Some(mut build) => build.refresh(workspace, targets).map(|()| build),
        None => Build::cold(workspace, targets),
    }
}

fn send(fd: RawFd, byte: u8, fds: &[RawFd]) -> Result<()> {
    let mut data = [byte];
    let mut iov = libc::iovec { iov_base: data.as_mut_ptr().cast(), iov_len: 1 };
    let bytes = std::mem::size_of_val(fds);
    // SAFETY: CMSG_SPACE only computes a size.
    let mut control = vec![0_u8; unsafe { libc::CMSG_SPACE(bytes as u32) } as usize];
    // SAFETY: msghdr is plain data, and all zero bytes are a valid value.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    (msg.msg_iov, msg.msg_iovlen) = (&raw mut iov, 1);
    (msg.msg_control, msg.msg_controllen) = (control.as_mut_ptr().cast(), control.len());
    // SAFETY: msg points at live iov and control buffers sized by CMSG_SPACE for fds, so the first header and its data fit in control.
    let sent = unsafe {
        let header = libc::CMSG_FIRSTHDR(&msg);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(bytes as u32) as usize;
        std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(header).cast::<RawFd>(), fds.len());
        libc::sendmsg(fd, &msg, 0)
    };
    if sent != 1 {
        bail!("sendmsg: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

fn receive(fd: RawFd) -> Result<(u8, Vec<RawFd>)> {
    let mut data = [0_u8];
    let mut iov = libc::iovec { iov_base: data.as_mut_ptr().cast(), iov_len: 1 };
    // SAFETY: CMSG_SPACE only computes a size.
    let mut control = vec![0_u8; unsafe { libc::CMSG_SPACE(2 * size_of::<RawFd>() as u32) } as usize];
    // SAFETY: msghdr is plain data, and all zero bytes are a valid value.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    (msg.msg_iov, msg.msg_iovlen) = (&raw mut iov, 1);
    (msg.msg_control, msg.msg_controllen) = (control.as_mut_ptr().cast(), control.len());
    // SAFETY: msg points at live iov and control buffers for recvmsg to fill.
    if unsafe { libc::recvmsg(fd, &mut msg, libc::MSG_CMSG_CLOEXEC) } != 1 {
        bail!("recvmsg: {}", std::io::Error::last_os_error());
    }
    let mut fds = Vec::new();
    // SAFETY: recvmsg filled msg; each header that CMSG_FIRSTHDR and CMSG_NXTHDR yield lies inside control with cmsg_len bytes.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let count = ((*header).cmsg_len - libc::CMSG_LEN(0) as usize) / size_of::<RawFd>();
                let data = libc::CMSG_DATA(header).cast::<RawFd>();
                fds.extend((0..count).map(|at| data.add(at).read_unaligned()));
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    Ok((data[0], fds))
}
