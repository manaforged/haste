//! Wire serialization and descriptor framing for daemon request payloads (output path, args, and client environment).

use std::io::{Read as _, Write as _};
use std::os::fd::RawFd;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rustc_hash::FxHashMap;

const MAX_PATH_BYTES: usize = 64 * 1024;
const MAX_ARGS: u32 = 4096;
const MAX_ARG_BYTES: usize = 1024 * 1024;
const MAX_ENVS: u32 = 8192;

pub(super) struct Payload {
    pub(super) out: Option<PathBuf>,
    pub(super) args: Vec<String>,
    pub(super) env: FxHashMap<String, String>,
}

pub(super) fn write_payload(
    conn: &mut std::fs::File,
    out: Option<&Path>,
    args: &[String],
) -> Result<()> {
    use std::os::unix::ffi::OsStrExt as _;
    let out_bytes = out.map_or(&[][..], |p| p.as_os_str().as_bytes());
    let mut bytes = (out_bytes.len() as u32).to_le_bytes().to_vec();
    bytes.extend_from_slice(out_bytes);
    bytes.extend((args.len() as u32).to_le_bytes());
    for arg in args {
        bytes.extend((arg.len() as u32).to_le_bytes());
        bytes.extend(arg.as_bytes());
    }
    let envs: Vec<(String, String)> = std::env::vars().collect();
    bytes.extend((envs.len().min(MAX_ENVS as usize) as u32).to_le_bytes());
    for (k, v) in envs.into_iter().take(MAX_ENVS as usize) {
        bytes.extend((k.len() as u32).to_le_bytes());
        bytes.extend(k.as_bytes());
        bytes.extend((v.len() as u32).to_le_bytes());
        bytes.extend(v.as_bytes());
    }
    Ok(conn.write_all(&bytes)?)
}

fn read_u32(conn: &mut std::fs::File) -> Result<u32> {
    let mut word = [0; 4];
    conn.read_exact(&mut word)?;
    Ok(u32::from_le_bytes(word))
}

fn read_string(conn: &mut std::fs::File, limit: usize, what: &str) -> Result<String> {
    let len = read_u32(conn)? as usize;
    if len > limit {
        bail!("{what} exceeds limit");
    }
    let mut raw = vec![0; len];
    conn.read_exact(&mut raw)?;
    Ok(String::from_utf8(raw)?)
}

pub(super) fn read_payload(conn: &mut std::fs::File) -> Result<Payload> {
    use std::os::unix::ffi::OsStringExt as _;
    let out_len = read_u32(conn)? as usize;
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
    let count = read_u32(conn)?;
    if count > MAX_ARGS {
        bail!("argument count exceeds limit");
    }
    let mut args = Vec::with_capacity(count as usize);
    for _ in 0..count {
        args.push(read_string(conn, MAX_ARG_BYTES, "argument")?);
    }
    let env_count = read_u32(conn)?;
    if env_count > MAX_ENVS {
        bail!("environment count exceeds limit");
    }
    let mut env = FxHashMap::with_capacity_and_hasher(env_count as usize, Default::default());
    for _ in 0..env_count {
        let k = read_string(conn, MAX_ARG_BYTES, "env key")?;
        let v = read_string(conn, MAX_ARG_BYTES, "env value")?;
        env.insert(k, v);
    }
    Ok(Payload { out, args, env })
}

pub(super) const STDIO: [RawFd; 3] = [0, 1, 2];

pub(super) fn socket_name_in(temp_dir: &Path, uid: libc::uid_t, key: u64) -> String {
    use std::os::unix::ffi::OsStrExt as _;
    let dir_name = format!("haste-{uid}");
    let file_name = format!("{key:016x}.sock");
    let primary = temp_dir.join(&dir_name).join(&file_name);
    if primary.as_os_str().as_bytes().len() < 100 {
        primary.to_string_lossy().into_owned()
    } else {
        Path::new("/tmp")
            .join(&dir_name)
            .join(&file_name)
            .to_string_lossy()
            .into_owned()
    }
}

pub(super) fn socket_name_for(key: u64) -> String {
    // SAFETY: geteuid takes no arguments and reads the process's effective user id.
    let uid = unsafe { libc::geteuid() };
    if cfg!(target_os = "linux") {
        format!("haste/{uid}/{key:016x}")
    } else {
        socket_name_in(&std::env::temp_dir(), uid, key)
    }
}

pub(super) fn sockaddr_from_name(name: &str) -> Result<(libc::sockaddr_un, libc::socklen_t)> {
    let abstract_name = cfg!(target_os = "linux");
    // SAFETY: sockaddr_un is plain data, and all zero bytes are a valid value.
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let skip = usize::from(abstract_name);
    let max_bytes = addr.sun_path.len() - 1 - skip;
    if !abstract_name && name.as_bytes().contains(&0) {
        bail!("unix socket path contains NUL byte: {name}");
    }
    if name.len() > max_bytes {
        bail!(
            "unix socket path `{name}` exceeds {} bytes ({})",
            max_bytes,
            name.len()
        );
    }
    for (slot, byte) in addr.sun_path[skip..skip + name.len()]
        .iter_mut()
        .zip(name.bytes())
    {
        *slot = byte as libc::c_char;
    }
    let len = std::mem::offset_of!(libc::sockaddr_un, sun_path) + skip + name.len();
    Ok((addr, len as libc::socklen_t))
}

pub(super) fn connect_name(name: &str) -> Result<Option<RawFd>> {
    let (addr, len) = sockaddr_from_name(name)?;
    connect_addr(&addr, len)
}

pub(super) fn connect_addr(
    addr: &libc::sockaddr_un,
    len: libc::socklen_t,
) -> Result<Option<RawFd>> {
    let fd = socket()?;
    // SAFETY: addr is a live sockaddr_un of length len, and fd is the socket made above.
    if unsafe { libc::connect(fd, (&raw const *addr).cast(), len) } != 0 || !same_user(fd) {
        close_all(&[fd]);
        return Ok(None);
    }
    Ok(Some(fd))
}

pub(super) fn socket() -> Result<RawFd> {
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

pub(super) fn client_stdio() -> Result<([RawFd; 3], Vec<RawFd>)> {
    let mut passed = STDIO;
    let mut opened = Vec::new();
    for (slot, &fd) in passed.iter_mut().zip(&STDIO) {
        // SAFETY: fcntl with F_GETFD inspects whether fd is open in this process without side effects.
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0 {
            continue;
        }
        let flags = (if fd == 0 {
            libc::O_RDONLY
        } else {
            libc::O_WRONLY
        }) | libc::O_CLOEXEC;
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

pub(super) fn wait_connection(listener: RawFd, timeout_ms: i32) -> Result<bool> {
    loop {
        let mut pfd = libc::pollfd {
            fd: listener,
            events: libc::POLLIN,
            revents: 0,
        };
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

pub(super) fn unlink_addr(addr: &libc::sockaddr_un) {
    if !cfg!(target_os = "linux") {
        // SAFETY: sun_path is NUL-terminated, since sockaddr_from_name leaves its last byte zero.
        unsafe { libc::unlink(addr.sun_path.as_ptr()) };
    }
}

#[cfg(target_os = "linux")]
pub(super) fn same_user(fd: RawFd) -> bool {
    // SAFETY: ucred is plain data, and all zero bytes are a valid value.
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: fd is an accepted AF_UNIX socket, and cred/len point to live storage of size len.
    let rc = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&raw mut cred).cast(),
            &raw mut len,
        )
    };
    // SAFETY: geteuid takes no pointers and reads the process's effective user id.
    rc == 0
        && len as usize == std::mem::size_of::<libc::ucred>()
        && cred.uid == unsafe { libc::geteuid() }
}

#[cfg(target_os = "macos")]
pub(super) fn same_user(fd: RawFd) -> bool {
    let (mut uid, mut gid) = (0 as libc::uid_t, 0 as libc::gid_t);
    // SAFETY: fd is an accepted AF_UNIX socket, and uid/gid point to live storage.
    let rc = unsafe { libc::getpeereid(fd, &raw mut uid, &raw mut gid) };
    // SAFETY: geteuid takes no pointers and reads the process's effective user id.
    rc == 0 && uid == unsafe { libc::geteuid() }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn same_user(_fd: RawFd) -> bool {
    false
}

pub(super) fn ensure_socket_dir(sock_name: &str) -> Result<()> {
    if cfg!(target_os = "linux") {
        return Ok(());
    }
    use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _, PermissionsExt as _};
    let Some(dir) = Path::new(sock_name).parent() else {
        bail!("socket path `{sock_name}` has no parent directory");
    };
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)?;
    let meta = std::fs::symlink_metadata(dir)?;
    if !meta.is_dir() {
        bail!("socket directory {} is not a directory", dir.display());
    }
    // SAFETY: geteuid takes no arguments and reads the process's effective user id.
    let uid = unsafe { libc::geteuid() };
    if meta.uid() != uid {
        bail!(
            "socket directory {} is not owned by uid {uid}",
            dir.display()
        );
    }
    if meta.mode() & 0o077 != 0 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

pub(super) fn redirect(client: [RawFd; 3]) -> [RawFd; 3] {
    // SAFETY: fcntl and dup2 take descriptors this process holds; 0, 1 and 2 stay open throughout.
    STDIO.map(|own| unsafe {
        let saved = libc::fcntl(own, libc::F_DUPFD_CLOEXEC, 3);
        libc::dup2(client[own as usize], own);
        saved
    })
}

pub(super) fn restore(saved: [RawFd; 3], received: &[RawFd]) {
    drop(std::io::stdout().flush());
    drop(std::io::stderr().flush());
    for (&own, kept) in STDIO.iter().zip(saved) {
        // SAFETY: saved holds the duplicates of this process's own standard descriptors.
        unsafe {
            libc::dup2(kept, own);
        }
    }
    close_all(&saved);
    close_all(received);
}

pub(super) fn close_all(fds: &[RawFd]) {
    for &fd in fds {
        // SAFETY: fds are descriptors this process owns and nothing else uses after this call.
        unsafe {
            libc::close(fd);
        }
    }
}

pub(super) fn send(fd: RawFd, byte: u8, fds: &[RawFd]) -> Result<()> {
    let mut data = [byte];
    let mut iov = libc::iovec {
        iov_base: data.as_mut_ptr().cast(),
        iov_len: 1,
    };
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
        std::ptr::copy_nonoverlapping(
            fds.as_ptr(),
            libc::CMSG_DATA(header).cast::<RawFd>(),
            fds.len(),
        );
        libc::sendmsg(fd, &msg, 0)
    };
    if sent != 1 {
        bail!("sendmsg: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

pub(super) fn receive(fd: RawFd) -> Result<(u8, Vec<RawFd>)> {
    let mut data = [0_u8];
    let mut iov = libc::iovec {
        iov_base: data.as_mut_ptr().cast(),
        iov_len: 1,
    };
    // SAFETY: CMSG_SPACE only computes a size.
    let mut control =
        vec![0_u8; unsafe { libc::CMSG_SPACE(std::mem::size_of_val(&STDIO) as u32) } as usize];
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
                let count = ((*header).cmsg_len as usize)
                    .saturating_sub(libc::CMSG_LEN(0) as usize)
                    / size_of::<RawFd>();
                let data = libc::CMSG_DATA(header).cast::<RawFd>();
                fds.extend((0..count).map(|at| data.add(at).read_unaligned()));
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    fds.iter()
        .try_for_each(|&fd| crate::image::close_on_exec(fd).map(drop))?;
    Ok((data[0], fds))
}
