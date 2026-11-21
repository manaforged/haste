//! A resident build of one workspace that answers `build`, `test` and `run` requests over a Unix socket: an abstract one on Linux, which leaves no file behind, else one in the temporary directory.

#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::path::Path;
#[cfg(unix)]
use std::path::PathBuf;
use std::time::Instant;

use crate::graph::{self, Targets};
use anyhow::{Result, bail};

use crate::build::Build;
#[cfg(unix)]
use crate::graph::store::{FileStamp, file_stamp, watched_stamps};
#[cfg(unix)]
use crate::outline::tree::Profile;

/// The workspace a server answers for: its root, targets and graph, and the stamps of the lock file, local manifests, build scripts and watched inputs it was loaded from.
#[cfg(unix)]
struct Served<'a> {
    root: &'a Path,
    targets: &'a Targets,
    workspace: &'a mut graph::Workspace,
    manifests: Vec<(PathBuf, FileStamp)>,
    watched: (Vec<(PathBuf, u64, i128)>, Vec<(String, Option<std::ffi::OsString>)>),
}

#[cfg(unix)]
impl Served<'_> {
    fn manifests(workspace: &graph::Workspace, root: &Path) -> Vec<(PathBuf, FileStamp)> {
        graph::manifest_paths(workspace, root)
            .into_iter()
            .map(|path| {
                let now = file_stamp(&path);
                (path, now)
            })
            .collect()
    }

    /// Loads the workspace again, running changed build scripts, when the lock file, a local manifest, a local build script or a watched input changed since it loaded; returns whether it did.
    fn reload(&mut self) -> Result<bool> {
        let manifests_ok = self.manifests.iter().all(|(path, kept)| file_stamp(path) == *kept);
        if manifests_ok && watched_stamps(self.workspace) == self.watched {
            return Ok(false);
        }
        *self.workspace = crate::cli::load_workspace(self.root, self.targets)?;
        self.manifests = Served::manifests(self.workspace, self.root);
        self.watched = watched_stamps(self.workspace);
        Ok(true)
    }
}

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

/// The command-line names of server-handled requests.
pub const MODES: [(Request, &str); 3] = [(Request::Build, "build"), (Request::Test, "test"), (Request::Run, "run")];

impl Request {
    #[cfg(unix)]
    fn byte(self) -> u8 {
        self as u8
    }

    /// The request a command-line mode names, if it names one.
    pub fn parse_mode(mode: &str) -> Option<Request> {
        MODES.into_iter().find(|(_, name)| *name == mode).map(|(request, _)| request)
    }

    #[cfg(unix)]
    fn parse(byte: u8) -> Option<Request> {
        MODES.into_iter().find(|(request, _)| request.byte() == byte).map(|(request, _)| request)
    }
}

/// Builds what `request` needs, prints how long that took since `started`, then prints what failed for `build`, runs every case for `test` or `main` with `args` for `run`, and returns the exit status.
pub fn finish(build: &mut Build, workspace: &graph::Workspace, (request, out, args): (Request, Option<&Path>, &[String]), started: Instant) -> Result<i32> {
    if let (Request::Build, Some(path)) = (request, out) {
        return build.emit_executable(workspace, path, false);
    }
    match request {
        Request::Test => build.compile_cases(workspace)?,
        Request::Build => build.compile_binary(workspace)?,
        Request::Run => {}
    }
    let built = started.elapsed();
    match request {
        Request::Test => build.report(started, built, args.get(1..).unwrap_or(&[])),
        Request::Build => build.check(built),
        Request::Run => build.run(workspace, args),
    }
}

#[cfg(not(unix))]
/// Sends `request`, optional `out` path, and `args` to the server of `root`, or returns `Ok(None)` when no server runs.
pub fn ask(_root: &std::path::Path, _request: Request, _out: Option<&Path>, _args: &[String]) -> Result<Option<i32>> {
    Ok(None)
}

#[cfg(not(unix))]
/// Loads `workspace` once, then answers requests until the process is killed.
pub fn serve(_workspace: &mut graph::Workspace, _targets: &Targets, _root: &std::path::Path) -> Result<()> {
    bail!("haste serve requires Unix domain sockets")
}

#[cfg(not(unix))]
/// Spawns a detached idle-timed background server for `root` when supported.
pub fn spawn(_root: &std::path::Path, _request: Request, _out: Option<&Path>) {}

#[cfg(unix)]
pub use unix_ipc::{ask, serve, spawn};

#[cfg(unix)]
mod unix_ipc {
use super::*;

fn address(root: &std::path::Path) -> (libc::sockaddr_un, libc::socklen_t) {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    root.hash(&mut hasher);
    crate::graph::store::home().hash(&mut hasher);
    crate::graph::store::program().hash(&mut hasher);
    (crate::profile() == Profile::Release).hash(&mut hasher);
    crate::prep::prepared().hash(&mut hasher);
    crate::selection().hash(&mut hasher);
    crate::test_mode().hash(&mut hasher);
    std::env::var_os("HASTE_LIBRARY").hash(&mut hasher);
    let abstract_name = cfg!(target_os = "linux");
    let name = match abstract_name {
        true => format!("haste/{:016x}", hasher.finish()),
        false => std::env::temp_dir().join(format!("haste-{:016x}.sock", hasher.finish())).to_string_lossy().into_owned(),
    };
    // SAFETY: sockaddr_un is plain data, and all zero bytes are a valid value.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let skip = usize::from(abstract_name);
    let room = addr.sun_path.len() - 1;
    for (slot, byte) in addr.sun_path[skip..room].iter_mut().zip(name.bytes()) {
        *slot = byte as libc::c_char;
    }
    let len = std::mem::offset_of!(libc::sockaddr_un, sun_path) + skip + name.len().min(room - skip);
    (addr, len as libc::socklen_t)
}

fn socket() -> Result<RawFd> {
    // SAFETY: socket takes plain integers and returns a new descriptor or -1.
    let raw = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0) };
    if raw < 0 {
        bail!("socket: {}", std::io::Error::last_os_error());
    }
    if raw >= 3 {
        return crate::image::close_on_exec(raw);
    }
    // SAFETY: raw is a valid descriptor below 3; F_DUPFD_CLOEXEC duplicates it at >= 3 and close releases raw.
    let moved = unsafe {
        let duped = libc::fcntl(raw, libc::F_DUPFD_CLOEXEC, 3);
        libc::close(raw);
        duped
    };
    if moved < 0 {
        bail!("fcntl: {}", std::io::Error::last_os_error());
    }
    Ok(moved)
}

fn client_stdio() -> Result<([RawFd; 3], Vec<RawFd>)> {
    let mut passed = STDIO;
    let mut opened = Vec::new();
    for (slot, &fd) in passed.iter_mut().zip(&STDIO) {
        // SAFETY: fcntl with F_GETFD inspects whether fd is open in this process without side effects.
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0 {
            continue;
        }
        let flags = (if fd == 0 { libc::O_RDONLY } else { libc::O_WRONLY }) | libc::O_CLOEXEC;
        // SAFETY: c"/dev/null" is a static NUL-terminated path, and flags takes no mode argument.
        let null = unsafe { libc::open(c"/dev/null".as_ptr(), flags) };
        if null < 0 {
            close_all(&opened);
            bail!("open /dev/null: {}", std::io::Error::last_os_error());
        }
        *slot = null;
        opened.push(null);
    }
    Ok((passed, opened))
}

/// A socket connected to the server of `root`, or none when no server answers.
fn connect(root: &std::path::Path) -> Result<Option<RawFd>> {
    let fd = socket()?;
    let (addr, len) = address(root);
    // SAFETY: addr is a live sockaddr_un of length len, and fd is the socket made above.
    if unsafe { libc::connect(fd, (&raw const addr).cast(), len) } != 0 || !same_user(fd) {
        close_all(&[fd]);
        return Ok(None);
    }
    Ok(Some(fd))
}

/// Sends `request`, optional `out` path, and `args` with this process's stdout and stderr to the server of `root`, waits for it, and returns its status, or none when no server runs.
pub fn ask(root: &std::path::Path, request: Request, out: Option<&Path>, args: &[String]) -> Result<Option<i32>> {
    let Some(fd) = connect(root)? else {
        return Ok(None);
    };
    // SAFETY: fd is a connected socket that this File owns from here on.
    let mut conn = unsafe { std::fs::File::from_raw_fd(fd) };
    std::io::stdout().flush()?;
    let (stdio, opened) = client_stdio()?;
    let sent = send(fd, request.byte(), &stdio);
    close_all(&opened);
    sent?;
    write_payload(&mut conn, out, args)?;
    let mut status = [0; 4];
    conn.read_exact(&mut status)?;
    Ok(Some(i32::from_le_bytes(status)))
}

/// Spawns a detached background `serve` daemon for `root` with an idle timeout.
pub fn spawn(root: &std::path::Path, request: Request, out: Option<&Path>) {
    use std::os::unix::process::CommandExt as _;
    use std::process::{Command, Stdio};
    let Ok(exe) = std::env::current_exe() else { return };
    let mut cmd = Command::new(exe);
    cmd.arg("serve").arg(root);
    let sel = crate::selection();
    for (flag, on) in [
        ("--release", crate::profile() == Profile::Release),
        ("--prepared", crate::prep::prepared()),
        ("--all-features", sel.all_features),
        ("--no-default-features", sel.no_default_features),
        ("--ignored", sel.ignored_only),
        ("--include-ignored", sel.include_ignored),
    ] {
        if on {
            cmd.arg(flag);
        }
    }
    for (flag, val) in [("-p", sel.package.as_deref()), ("--bin", sel.bin.as_deref()), ("--example", sel.example.as_deref())] {
        if let Some(val) = val {
            cmd.arg(flag).arg(val);
        }
    }
    if !sel.features.is_empty() {
        cmd.arg("-F").arg(sel.features.join(","));
    }
    if std::env::var_os("HASTE_SERVE_IDLE_SECS").is_none() {
        cmd.env("HASTE_SERVE_IDLE_SECS", "120");
    }
    cmd.env("HASTE_NO_DAEMON", "1");
    match (request, out) {
        (Request::Build, None) => drop(cmd.env("HASTE_SERVE_WARM", "build")),
        (Request::Test, _) => drop(cmd.env("HASTE_SERVE_TEST", "1").env("HASTE_SERVE_WARM", "test")),
        _ => {}
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    // SAFETY: setsid takes no pointers and detaches the child into a new session before exec.
    unsafe {
        cmd.pre_exec(|| if libc::setsid() < 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) });
    }
    drop(cmd.spawn());
}

fn idle_timeout_ms() -> Option<i32> {
    let secs: u64 = std::env::var_os("HASTE_SERVE_IDLE_SECS")?.to_str()?.parse().ok()?;
    Some(secs.saturating_mul(1000).min(i32::MAX as u64) as i32)
}

fn wait_connection(listener: RawFd, timeout_ms: i32) -> Result<bool> {
    loop {
        let mut pfd = libc::pollfd { fd: listener, events: libc::POLLIN, revents: 0 };
        // SAFETY: pfd is one live pollfd entry for the listening socket.
        let rc = unsafe { libc::poll(&raw mut pfd, 1, timeout_ms) };
        if rc > 0 {
            return Ok(true);
        }
        if rc == 0 {
            return Ok(false);
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            bail!("poll: {err}");
        }
    }
}

fn unlink_addr(addr: &libc::sockaddr_un) {
    if !cfg!(target_os = "linux") {
        // SAFETY: sun_path is NUL-terminated, since address leaves its last byte zero.
        unsafe { libc::unlink(addr.sun_path.as_ptr()) };
    }
}

/// Loads `workspace` once, then answers requests until the process is killed or `HASTE_SERVE_IDLE_SECS` elapses, loading it again when its lock file or a local manifest changes.
pub fn serve(workspace: &mut graph::Workspace, targets: &Targets, root: &std::path::Path) -> Result<()> {
    let listener = listen(root)?;
    let started = Instant::now();
    let mut build = Some(Build::load(workspace, targets)?);
    if let Some(ref mut warm) = build {
        match std::env::var_os("HASTE_SERVE_WARM").as_deref().and_then(|v| v.to_str()) {
            Some("build") => drop(warm.compile_binary(workspace)),
            Some("test") => drop(warm.compile_cases(workspace)),
            _ => {}
        }
    }
    let manifests = Served::manifests(workspace, root);
    let watched = watched_stamps(workspace);
    let mut served = Served { root, targets, workspace, manifests, watched };
    let idle_ms = idle_timeout_ms();
    eprintln!("serving {} after {:.1} ms", root.display(), started.elapsed().as_secs_f64() * 1e3);
    loop {
        if let Some(timeout_ms) = idle_ms {
            match wait_connection(listener, timeout_ms) {
                Ok(true) => {}
                Ok(false) => {
                    unlink_addr(&address(root).0);
                    close_all(&[listener]);
                    return Ok(());
                }
                Err(err) => {
                    unlink_addr(&address(root).0);
                    close_all(&[listener]);
                    return Err(err);
                }
            }
        }
        // SAFETY: listener is a listening socket; accept returns a new descriptor or -1.
        let fd = unsafe { libc::accept(listener, std::ptr::null_mut(), std::ptr::null_mut()) };
        if fd < 0 || crate::image::close_on_exec(fd).is_err() {
            continue;
        }
        // SAFETY: fd is the accepted connection, which this File owns from here on.
        let mut conn = unsafe { std::fs::File::from_raw_fd(fd) };
        if !same_user(fd) {
            continue;
        }
        let status = match receive(fd) {
            Ok((byte, fds)) => match read_payload(&mut conn) {
                Ok((out, args)) => answer(&mut build, &mut served, Request::parse(byte).map(|request| (request, out, args)), &fds),
                Err(_) => 1,
            },
            Err(_) => 1,
        };
        drop(conn.write_all(&status.to_le_bytes()));
    }
}

#[cfg(target_os = "linux")]
fn same_user(fd: RawFd) -> bool {
    // SAFETY: ucred is plain data, and all zero bytes are a valid value.
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: fd is an accepted AF_UNIX socket, and cred/len point to live storage of size len.
    let rc = unsafe { libc::getsockopt(fd, libc::SOL_SOCKET, libc::SO_PEERCRED, (&raw mut cred).cast(), &raw mut len) };
    // SAFETY: geteuid takes no pointers and reads the process's effective user id.
    rc == 0 && len as usize == std::mem::size_of::<libc::ucred>() && cred.uid == unsafe { libc::geteuid() }
}

#[cfg(target_os = "macos")]
fn same_user(fd: RawFd) -> bool {
    let (mut uid, mut gid) = (0 as libc::uid_t, 0 as libc::gid_t);
    // SAFETY: fd is an accepted AF_UNIX socket, and uid/gid point to live storage.
    let rc = unsafe { libc::getpeereid(fd, &raw mut uid, &raw mut gid) };
    // SAFETY: geteuid takes no pointers and reads the process's effective user id.
    rc == 0 && uid == unsafe { libc::geteuid() }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn same_user(_fd: RawFd) -> bool {
    false
}

fn listen(root: &std::path::Path) -> Result<RawFd> {
    let fd = socket()?;
    let (addr, len) = address(root);
    let answered = connect(root)?.inspect(|fd| close_all(&[*fd])).is_some();
    if !answered {
        unlink_addr(&addr);
    }
    // SAFETY: addr is a live sockaddr_un of length len, and fd is the socket made above.
    if unsafe { libc::bind(fd, (&raw const addr).cast(), len) } != 0 {
        bail!("a build server for {} runs already: {}", root.display(), std::io::Error::last_os_error());
    }
    if !cfg!(target_os = "linux") {
        // SAFETY: sun_path is NUL-terminated and was just bound above.
        unsafe { libc::chmod(addr.sun_path.as_ptr(), 0o600) };
    }
    // SAFETY: fd is a bound AF_UNIX socket.
    if unsafe { libc::listen(fd, 16) } != 0 {
        bail!("a build server for {} runs already: {}", root.display(), std::io::Error::last_os_error());
    }
    Ok(fd)
}

fn write_payload(conn: &mut std::fs::File, out: Option<&Path>, args: &[String]) -> Result<()> {
    use std::os::unix::ffi::OsStrExt as _;
    let out_bytes = out.map_or(&[][..], |p| p.as_os_str().as_bytes());
    let mut bytes = (out_bytes.len() as u32).to_le_bytes().to_vec();
    bytes.extend_from_slice(out_bytes);
    bytes.extend((args.len() as u32).to_le_bytes());
    for arg in args {
        bytes.extend((arg.len() as u32).to_le_bytes());
        bytes.extend(arg.as_bytes());
    }
    Ok(conn.write_all(&bytes)?)
}

const MAX_PATH_BYTES: usize = 64 * 1024;
const MAX_ARGS: u32 = 4096;
const MAX_ARG_BYTES: usize = 1024 * 1024;

fn read_payload(conn: &mut std::fs::File) -> Result<(Option<PathBuf>, Vec<String>)> {
    use std::os::unix::ffi::OsStringExt as _;
    let mut word = [0; 4];
    let mut next = |conn: &mut std::fs::File| -> Result<u32> {
        conn.read_exact(&mut word)?;
        Ok(u32::from_le_bytes(word))
    };
    let out_len = next(conn)? as usize;
    if out_len > MAX_PATH_BYTES {
        bail!("output path exceeds limit");
    }
    let out = if out_len == 0 {
        None
    } else {
        let mut raw = vec![0; out_len];
        conn.read_exact(&mut raw)?;
        Some(PathBuf::from(std::ffi::OsString::from_vec(raw)))
    };
    let count = next(conn)?;
    if count > MAX_ARGS {
        bail!("argument count exceeds limit");
    }
    let mut args = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let arg_len = next(conn)? as usize;
        if arg_len > MAX_ARG_BYTES {
            bail!("argument exceeds limit");
        }
        let mut arg = vec![0; arg_len];
        conn.read_exact(&mut arg)?;
        args.push(String::from_utf8(arg)?);
    }
    Ok((out, args))
}

/// The standard input, output and error that a client hands the server for the program it runs.
const STDIO: [RawFd; 3] = [0, 1, 2];

fn answer(build: &mut Option<Build>, served: &mut Served<'_>, request: Option<(Request, Option<PathBuf>, Vec<String>)>, fds: &[RawFd]) -> i32 {
    let Ok(&client) = <&[RawFd; 3]>::try_from(fds) else {
        close_all(fds);
        return 1;
    };
    let saved = redirect(client);
    let started = Instant::now();
    let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match request {
        Some((request, out, args)) => served.reload().and_then(|reloaded| {
            let kept = build.take().filter(|_| !reloaded);
            let (workspace, targets) = (&*served.workspace, served.targets);
            current(kept, workspace, targets).and_then(|fresh| finish(build.insert(fresh), workspace, (request, out.as_deref(), &args), started))
        }),
        None => Err(anyhow::anyhow!("unknown request")),
    }));
    let done = answered.unwrap_or_else(|_| {
        *build = None;
        Err(anyhow::anyhow!("the request panicked; the next request loads the workspace again"))
    });
    if let Err(error) = &done {
        drop(writeln!(std::io::stderr(), "error: {error:#}"));
    }
    restore(saved, fds);
    done.unwrap_or(1)
}

fn redirect(client: [RawFd; 3]) -> [RawFd; 3] {
    // SAFETY: fcntl and dup2 take descriptors this process holds; 0, 1 and 2 stay open throughout.
    STDIO.map(|own| unsafe {
        let saved = libc::fcntl(own, libc::F_DUPFD_CLOEXEC, 3);
        libc::dup2(client[own as usize], own);
        saved
    })
}

fn restore(saved: [RawFd; 3], received: &[RawFd]) {
    drop(std::io::stdout().flush());
    drop(std::io::stderr().flush());
    // SAFETY: saved holds the duplicates of this process's own standard descriptors.
    for (&own, kept) in STDIO.iter().zip(saved) {
        unsafe {
            libc::dup2(kept, own);
        }
    }
    close_all(&saved);
    close_all(received);
}

fn close_all(fds: &[RawFd]) {
    // SAFETY: fds are descriptors this process owns and nothing else uses after this call.
    for &fd in fds {
        unsafe {
            libc::close(fd);
        }
    }
}

/// Returns `build` brought up to date with the member files, or a build from nothing when there is none or the refresh fails, so a failed refresh never leaves a half-updated build behind.
fn current(build: Option<Build>, workspace: &graph::Workspace, targets: &Targets) -> Result<Build> {
    match build {
        Some(mut build) => build.refresh(workspace, targets).map(|()| build),
        None => Build::load(workspace, targets),
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
    (msg.msg_control, msg.msg_controllen) = (control.as_mut_ptr().cast(), control.len() as _);
    // SAFETY: msg points at live iov and control buffers sized by CMSG_SPACE for fds, so the first header and its data fit in control.
    let sent = unsafe {
        let header = libc::CMSG_FIRSTHDR(&msg);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(bytes as u32) as _;
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
    let mut control = vec![0_u8; unsafe { libc::CMSG_SPACE(std::mem::size_of_val(&STDIO) as u32) } as usize];
    // SAFETY: msghdr is plain data, and all zero bytes are a valid value.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    (msg.msg_iov, msg.msg_iovlen) = (&raw mut iov, 1);
    (msg.msg_control, msg.msg_controllen) = (control.as_mut_ptr().cast(), control.len() as _);
    // SAFETY: msg points at live iov and control buffers for recvmsg to fill.
    if unsafe { libc::recvmsg(fd, &mut msg, 0) } != 1 {
        bail!("recvmsg: {}", std::io::Error::last_os_error());
    }
    let mut fds = Vec::new();
    // SAFETY: recvmsg filled msg; each header that CMSG_FIRSTHDR and CMSG_NXTHDR yield lies inside control with cmsg_len bytes.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let count = ((*header).cmsg_len as usize - libc::CMSG_LEN(0) as usize) / size_of::<RawFd>();
                let data = libc::CMSG_DATA(header).cast::<RawFd>();
                fds.extend((0..count).map(|at| data.add(at).read_unaligned()));
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    fds.iter().try_for_each(|&fd| crate::image::close_on_exec(fd).map(drop))?;
    Ok((data[0], fds))
}
}
