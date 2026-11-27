//! A resident build of one workspace that answers `build`, `test` and `run` requests over a Unix socket: an abstract one on Linux, which leaves no file behind, else one in the temporary directory.

mod control;
#[cfg(unix)]
mod proto;

pub use control::{status, stop};

#[cfg(unix)]
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::graph::{self, Targets};
use anyhow::{Result, bail};

use crate::build::Build;
use crate::graph::store::{FileStamp, file_stamp, watched_stamps};
#[cfg(unix)]
use crate::outline::tree::Profile;

/// Snapshot of workspace manifests, target discovery, build-script watches, and rustflags environment.
pub(crate) struct WorkspaceWatch {
    root: PathBuf,
    manifests: Vec<(PathBuf, FileStamp)>,
    target_stamp: u64,
    rustflags_env: (Option<String>, Option<String>),
    watched: (
        Vec<(PathBuf, u64, i128)>,
        Vec<(String, Option<std::ffi::OsString>)>,
    ),
}

impl WorkspaceWatch {
    fn rustflags_snapshot() -> (Option<String>, Option<String>) {
        (
            crate::process_env_var("RUSTFLAGS"),
            crate::process_env_var("CARGO_ENCODED_RUSTFLAGS"),
        )
    }

    fn manifest_stamps(workspace: &graph::Workspace, root: &Path) -> Vec<(PathBuf, FileStamp)> {
        graph::manifest_paths(workspace, root)
            .into_iter()
            .map(|path| {
                let now = file_stamp(&path);
                (path, now)
            })
            .collect()
    }

    fn local_script_snapshot(
        workspace: &graph::Workspace,
    ) -> Vec<(
        usize,
        Option<std::sync::Arc<graph::script::Output>>,
        Vec<(PathBuf, u64, i128)>,
    )> {
        workspace
            .graph
            .units
            .iter()
            .enumerate()
            .filter_map(|(unit, entry)| {
                let pkg = &workspace.packages[entry.package];
                if !pkg.local() || pkg.manifest.build.is_none() {
                    return None;
                }
                let mut out_stamps = Vec::new();
                let out_dir = graph::store::dir(workspace, unit).join("out");
                if out_dir.exists() {
                    graph::store::walk(&out_dir, &mut out_stamps);
                }
                Some((unit, entry.script.clone(), out_stamps))
            })
            .collect()
    }

    /// Captures all workspace, build-script, and environment freshness inputs for `workspace` at `root`.
    pub(crate) fn capture(workspace: &graph::Workspace, root: &Path) -> Self {
        Self {
            root: root.to_owned(),
            manifests: Self::manifest_stamps(workspace, root),
            target_stamp: crate::substrate::noop::target_discovery_stamp(workspace),
            rustflags_env: Self::rustflags_snapshot(),
            watched: watched_stamps(workspace),
        }
    }

    /// Returns whether every compiler-recorded `env!`/`option_env!` read on `build` still matches.
    pub(crate) fn env_reads_fresh(build: &Build) -> bool {
        !build
            .env_reads()
            .into_iter()
            .any(|(name, prev)| crate::process_env_var(&name) != prev)
    }

    /// Returns whether manifests, discovered targets, and rustflags environment match the captured snapshot.
    fn workspace_inputs_fresh(&self, workspace: &graph::Workspace) -> bool {
        self.manifests
            .iter()
            .all(|(path, kept)| file_stamp(path) == *kept)
            && crate::substrate::noop::target_discovery_stamp(workspace) == self.target_stamp
            && Self::rustflags_snapshot() == self.rustflags_env
    }

    /// Returns whether manifests, discovered targets, build-script watches, rustflags, and `build` env reads are all unchanged.
    pub(crate) fn is_fresh(&self, workspace: &graph::Workspace, build: Option<&Build>) -> bool {
        self.workspace_inputs_fresh(workspace)
            && watched_stamps(workspace) == self.watched
            && build.is_none_or(Self::env_reads_fresh)
    }

    /// Refreshes `workspace` when manifests, targets, rustflags, or build-script watches changed; returns whether cached build state is invalidated.
    pub(crate) fn refresh_workspace(
        &mut self,
        workspace: &mut graph::Workspace,
        targets: &Targets,
    ) -> Result<bool> {
        if self.workspace_inputs_fresh(workspace) {
            if watched_stamps(workspace) == self.watched {
                return Ok(false);
            }
            let before = Self::local_script_snapshot(workspace);
            for &(unit, _, _) in &before {
                workspace.graph.units[unit].script = None;
            }
            crate::scripts::settle(workspace, targets)?;
            self.watched = watched_stamps(workspace);
            let after = Self::local_script_snapshot(workspace);
            return Ok(before != after);
        }
        *workspace = crate::cli::load_workspace(&self.root, targets)?;
        *self = Self::capture(workspace, &self.root);
        Ok(true)
    }
}

/// The workspace a server answers for: its targets, graph, and captured watch state.
#[cfg(unix)]
struct Served<'a> {
    targets: &'a Targets,
    workspace: &'a mut graph::Workspace,
    watch: WorkspaceWatch,
}

#[cfg(unix)]
impl Served<'_> {
    /// Loads the workspace again, running changed build scripts, when the lock file, a local manifest, a local build script, a target directory, or a watched input changed since it loaded; returns whether it did.
    fn reload(&mut self) -> Result<bool> {
        self.watch.refresh_workspace(self.workspace, self.targets)
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
pub const MODES: [(Request, &str); 3] = [
    (Request::Build, "build"),
    (Request::Test, "test"),
    (Request::Run, "run"),
];

impl Request {
    #[cfg(unix)]
    fn byte(self) -> u8 {
        self as u8
    }

    /// The request a command-line mode names, if it names one.
    pub fn parse_mode(mode: &str) -> Option<Request> {
        MODES
            .into_iter()
            .find(|(_, name)| *name == mode)
            .map(|(request, _)| request)
    }

    #[cfg(unix)]
    fn parse(byte: u8) -> Option<Request> {
        MODES
            .into_iter()
            .find(|(request, _)| request.byte() == byte)
            .map(|(request, _)| request)
    }
}

/// Builds what `request` needs, prints how long that took since `started`, then prints what failed for `build`, runs every case for `test` or `main` with `args` for `run`, and returns the exit status.
pub fn finish(
    build: &mut Build,
    workspace: &graph::Workspace,
    (request, out, args): (Request, Option<&Path>, &[String]),
    started: Instant,
) -> Result<i32> {
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
pub fn ask(
    _root: &std::path::Path,
    _request: Request,
    _out: Option<&Path>,
    _args: &[String],
) -> Result<Option<i32>> {
    Ok(None)
}

#[cfg(not(unix))]
/// Loads `workspace` once, then answers requests until the process is killed.
pub fn serve(
    _workspace: &mut graph::Workspace,
    _targets: &Targets,
    _root: &std::path::Path,
) -> Result<()> {
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

    fn socket_key_for(root: &std::path::Path, cfg: &crate::config::Config) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        root.hash(&mut hasher);
        crate::graph::store::home().hash(&mut hasher);
        crate::graph::store::program().hash(&mut hasher);
        cfg.hash_cache_key(&mut hasher);
        hasher.finish()
    }

    pub(super) use proto::connect_name;

    fn address(
        root: &std::path::Path,
    ) -> Result<(String, String, libc::sockaddr_un, libc::socklen_t)> {
        let key = socket_key_for(root, &crate::config::Config::current());
        let key_hex = format!("{key:016x}");
        let name = proto::socket_name_for(key);
        let (addr, len) = proto::sockaddr_from_name(&name)?;
        Ok((key_hex, name, addr, len))
    }

    /// A socket connected to the server of `root`, or none when no server answers.
    fn connect(root: &std::path::Path) -> Result<Option<RawFd>> {
        let (_, _, addr, len) = address(root)?;
        proto::connect_addr(&addr, len)
    }

    /// Sends `request`, optional `out` path, and `args` with this process's stdout and stderr to the server of `root`, waits for it, and returns its status, or none when no server runs.
    pub fn ask(
        root: &std::path::Path,
        request: Request,
        out: Option<&Path>,
        args: &[String],
    ) -> Result<Option<i32>> {
        let Some(fd) = connect(root)? else {
            return Ok(None);
        };
        // SAFETY: fd is a connected socket that this File owns from here on.
        let mut conn = unsafe { std::fs::File::from_raw_fd(fd) };
        std::io::stdout().flush()?;
        let (stdio, opened) = proto::client_stdio()?;
        let sent = proto::send(fd, request.byte(), &stdio);
        proto::close_all(&opened);
        sent?;
        let mut ack = [0u8; 1];
        match conn.read_exact(&mut ack) {
            Ok(()) if ack[0] == 0 => return Ok(None),
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(err) => return Err(err.into()),
        }
        proto::write_payload(&mut conn, out, args)?;
        let mut status = [0; 4];
        conn.read_exact(&mut status)?;
        Ok(Some(i32::from_le_bytes(status)))
    }

    fn daemon_args(
        sel: &crate::Selection,
        release: bool,
        prepared: bool,
        jobs: usize,
    ) -> Vec<String> {
        let mut args = Vec::new();
        for (flag, on) in [
            ("--release", release),
            ("--prepared", prepared),
            ("--workspace", sel.workspace),
            ("--tests", sel.tests),
            ("--all-targets", sel.all_targets),
            ("--all-features", sel.all_features),
            ("--no-default-features", sel.no_default_features),
            ("--lib", sel.lib),
            ("--bins", sel.bins),
            ("--locked", sel.locked),
            ("--offline", sel.offline),
            ("-q", sel.quiet),
            ("--ignored", sel.ignored_only),
            ("--include-ignored", sel.include_ignored),
        ] {
            if on {
                args.push(flag.to_owned());
            }
        }
        let features = (!sel.features.is_empty()).then(|| sel.features.join(","));
        let message_format = match sel.message_format {
            crate::MessageFormat::Human => None,
            crate::MessageFormat::Short => Some("short"),
            crate::MessageFormat::Json => Some("json"),
            crate::MessageFormat::JsonRenderDiagnostics => Some("json-render-diagnostics"),
        };
        let jobs = jobs.to_string();
        for (flag, vals) in [
            ("-p", &sel.packages),
            ("--bin", &sel.bin),
            ("--example", &sel.example),
            ("--test", &sel.test),
        ] {
            for val in vals {
                args.extend([flag.to_owned(), val.clone()]);
            }
        }
        for (flag, val) in [
            ("-F", features.as_deref()),
            ("--message-format", message_format),
            ("-j", Some(jobs.as_str())),
        ]
        .into_iter()
        .filter_map(|(f, v)| v.map(|s| (f, s)))
        {
            args.extend([flag.to_owned(), val.to_owned()]);
        }
        args
    }

    /// Spawns a detached background `serve` daemon for `root` with an idle timeout.
    pub fn spawn(root: &std::path::Path, request: Request, out: Option<&Path>) {
        use std::os::unix::process::CommandExt as _;
        use std::process::{Command, Stdio};
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let mut cmd = Command::new(exe);
        cmd.arg("serve").arg(root);
        let sel = crate::selection();
        cmd.args(daemon_args(
            &sel,
            crate::profile() == Profile::Release,
            crate::prep::prepared(),
            crate::jobs(),
        ));
        if std::env::var_os("HASTE_SERVE_IDLE_SECS").is_none() {
            cmd.env("HASTE_SERVE_IDLE_SECS", "900");
        }
        cmd.env("HASTE_NO_DAEMON", "1");
        match (request, out) {
            (Request::Build, None) => drop(cmd.env("HASTE_SERVE_WARM", "build")),
            (Request::Test, _) => drop(
                cmd.env("HASTE_SERVE_TEST", "1")
                    .env("HASTE_SERVE_WARM", "test"),
            ),
            _ => {}
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        // SAFETY: setsid takes no pointers and detaches the child into a new session before exec.
        unsafe {
            cmd.pre_exec(|| {
                (libc::setsid() >= 0)
                    .then_some(())
                    .ok_or_else(std::io::Error::last_os_error)
            });
        }
        if let Ok(mut child) = cmd.spawn()
            && let Ok((_, sock, _, _)) = address(root)
        {
            for _ in 0..200 {
                if control::ping(&sock) || child.try_wait().ok().flatten().is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }

    fn idle_timeout_ms() -> Option<i32> {
        let secs: u64 = std::env::var_os("HASTE_SERVE_IDLE_SECS")?
            .to_str()?
            .parse()
            .ok()?;
        Some(secs.saturating_mul(1000).min(i32::MAX as u64) as i32)
    }

    struct SocketGuard {
        fd: RawFd,
        addr: libc::sockaddr_un,
    }

    impl Drop for SocketGuard {
        fn drop(&mut self) {
            proto::unlink_addr(&self.addr);
            proto::close_all(&[self.fd]);
        }
    }

    struct QueuedRequest {
        conn: std::fs::File,
        byte: u8,
        fds: Vec<RawFd>,
    }

    struct AcceptState<'a> {
        in_flight: &'a std::sync::atomic::AtomicUsize,
        running_program: &'a std::sync::atomic::AtomicBool,
        socket: &'a std::sync::Mutex<Option<SocketGuard>>,
        reg: &'a std::sync::Mutex<Option<control::Registration>>,
    }

    fn handle_accepted(
        mut conn: std::fs::File,
        fd: RawFd,
        req_tx: &std::sync::mpsc::Sender<QueuedRequest>,
        state: &AcceptState<'_>,
    ) -> bool {
        use std::sync::atomic::Ordering;
        if !proto::same_user(fd) {
            return false;
        }
        match proto::receive(fd) {
            Ok((control::PING_BYTE, fds)) => {
                proto::close_all(&fds);
                drop(conn.write_all(&0_i32.to_le_bytes()));
                false
            }
            Ok((control::STOP_BYTE, fds)) => {
                proto::close_all(&fds);
                drop(conn.write_all(&0_i32.to_le_bytes()));
                drop(conn);
                *crate::sync::lock(state.reg) = None;
                *crate::sync::lock(state.socket) = None;
                if state.in_flight.load(Ordering::Acquire) > 0 {
                    std::process::exit(0);
                }
                true
            }
            Ok((byte, fds)) => {
                if state.running_program.load(Ordering::Acquire) {
                    proto::close_all(&fds);
                    drop(conn.write_all(&[0]));
                    return false;
                }
                if conn.write_all(&[1]).is_err() {
                    proto::close_all(&fds);
                    return false;
                }
                let is_run = byte == Request::Run.byte();
                if is_run {
                    state.running_program.store(true, Ordering::Release);
                }
                state.in_flight.fetch_add(1, Ordering::AcqRel);
                if let Err(err) = req_tx.send(QueuedRequest { conn, byte, fds }) {
                    proto::close_all(&err.0.fds);
                    if is_run {
                        state.running_program.store(false, Ordering::Release);
                    }
                    state.in_flight.fetch_sub(1, Ordering::AcqRel);
                    return true;
                }
                false
            }
            Err(_) => {
                drop(conn.write_all(&1_i32.to_le_bytes()));
                false
            }
        }
    }

    fn accept_loop(
        listener: RawFd,
        idle_ms: Option<i32>,
        req_tx: std::sync::mpsc::Sender<QueuedRequest>,
        state: &AcceptState<'_>,
    ) -> Result<()> {
        use std::sync::atomic::Ordering;
        loop {
            if let Some(timeout_ms) = idle_ms
                && !proto::wait_connection(listener, timeout_ms)?
            {
                if state.in_flight.load(Ordering::Acquire) > 0 {
                    continue;
                }
                return Ok(());
            }
            // SAFETY: listener is a listening socket; accept returns a new descriptor or -1.
            let fd = unsafe { libc::accept(listener, std::ptr::null_mut(), std::ptr::null_mut()) };
            if fd < 0 {
                continue;
            }
            if crate::image::close_on_exec(fd).is_err() {
                proto::close_all(&[fd]);
                continue;
            }
            // SAFETY: fd is the accepted connection, which this File owns from here on.
            let conn = unsafe { std::fs::File::from_raw_fd(fd) };
            if handle_accepted(conn, fd, &req_tx, state) {
                return Ok(());
            }
        }
    }

    fn process_request(
        mut req: QueuedRequest,
        build: &mut Option<Build>,
        served: &mut Served<'_>,
        in_flight: &std::sync::atomic::AtomicUsize,
        running_program: &std::sync::atomic::AtomicBool,
    ) {
        struct Flight<'a> {
            in_flight: &'a std::sync::atomic::AtomicUsize,
            running_program: &'a std::sync::atomic::AtomicBool,
            is_run: bool,
        }
        impl Drop for Flight<'_> {
            fn drop(&mut self) {
                use std::sync::atomic::Ordering;
                if self.is_run {
                    self.running_program.store(false, Ordering::Release);
                }
                self.in_flight.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _flight = Flight {
            in_flight,
            running_program,
            is_run: req.byte == Request::Run.byte(),
        };
        let status = match proto::read_payload(&mut req.conn) {
            Ok(payload) => answer(
                build,
                served,
                Request::parse(req.byte).map(|request| (request, payload)),
                &req.fds,
            ),
            Err(_) => {
                proto::close_all(&req.fds);
                1
            }
        };
        drop(req.conn.write_all(&status.to_le_bytes()));
    }

    /// Loads `workspace` once, then answers requests until stopped or `HASTE_SERVE_IDLE_SECS` elapses, loading it again when its lock file or a local manifest changes.
    pub fn serve(
        workspace: &mut graph::Workspace,
        targets: &Targets,
        root: &std::path::Path,
    ) -> Result<()> {
        let (key_hex, sock_name, addr, len) = address(root)?;
        let listener = listen(root, &sock_name, &addr, len)?;
        let socket = std::sync::Mutex::new(Some(SocketGuard { fd: listener, addr }));
        let reg = std::sync::Mutex::new(Some(control::Registration::create(
            &key_hex, root, &sock_name,
        )?));
        let started = Instant::now();
        let watch = WorkspaceWatch::capture(workspace, root);
        let mut build = Some(Build::load(workspace, targets)?);
        if let Some(ref mut warm) = build {
            match std::env::var_os("HASTE_SERVE_WARM")
                .as_deref()
                .and_then(|v| v.to_str())
            {
                Some("build") => drop(warm.compile_binary(workspace)),
                Some("test") => drop(warm.compile_cases(workspace)),
                _ => {}
            }
        }
        let mut served = Served {
            targets,
            workspace,
            watch,
        };
        let idle_ms = idle_timeout_ms();
        eprintln!(
            "serving {} after {:.1} ms",
            root.display(),
            started.elapsed().as_secs_f64() * 1e3
        );
        let in_flight = std::sync::atomic::AtomicUsize::new(0);
        let running_program = std::sync::atomic::AtomicBool::new(false);
        let state = AcceptState {
            in_flight: &in_flight,
            running_program: &running_program,
            socket: &socket,
            reg: &reg,
        };
        std::thread::scope(|scope| {
            let (req_tx, req_rx) = std::sync::mpsc::channel();
            let accept_handle = scope.spawn(|| accept_loop(listener, idle_ms, req_tx, &state));
            while let Ok(req) = req_rx.recv() {
                process_request(req, &mut build, &mut served, &in_flight, &running_program);
            }
            match accept_handle.join() {
                Ok(res) => res,
                Err(_) => bail!("daemon accept loop panicked"),
            }
        })
    }

    fn listen(
        root: &std::path::Path,
        sock_name: &str,
        addr: &libc::sockaddr_un,
        len: libc::socklen_t,
    ) -> Result<RawFd> {
        let answered = proto::connect_addr(addr, len)?
            .inspect(|fd| proto::close_all(&[*fd]))
            .is_some();
        if answered {
            bail!("a build server for {} runs already", root.display());
        }
        proto::ensure_socket_dir(sock_name)?;
        proto::unlink_addr(addr);
        let fd = proto::socket()?;
        // SAFETY: addr is a live sockaddr_un of length len, and fd is the socket made above.
        if unsafe { libc::bind(fd, (&raw const *addr).cast(), len) } != 0 {
            let err = std::io::Error::last_os_error();
            proto::close_all(&[fd]);
            bail!("a build server for {} runs already: {err}", root.display());
        }
        if !cfg!(target_os = "linux") {
            // SAFETY: sun_path is NUL-terminated and was just bound above.
            unsafe { libc::chmod(addr.sun_path.as_ptr(), 0o600) };
        }
        // SAFETY: fd is a bound AF_UNIX socket.
        if unsafe { libc::listen(fd, 16) } != 0 {
            let err = std::io::Error::last_os_error();
            proto::close_all(&[fd]);
            bail!("a build server for {} runs already: {err}", root.display());
        }
        Ok(fd)
    }

    fn answer(
        build: &mut Option<Build>,
        served: &mut Served<'_>,
        request: Option<(Request, proto::Payload)>,
        fds: &[RawFd],
    ) -> i32 {
        let Ok(&client) = <&[RawFd; 3]>::try_from(fds) else {
            proto::close_all(fds);
            return 1;
        };
        let saved = proto::redirect(client);
        let started = Instant::now();
        let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match request {
            Some((request, payload)) => {
                let cenv = Some(std::sync::Arc::new(payload.env));
                crate::with_client_env(cenv, || {
                    served.reload().and_then(|reloaded| {
                        let env_stale = build
                            .as_ref()
                            .is_some_and(|b| !WorkspaceWatch::env_reads_fresh(b));
                        let kept = build.take().filter(|_| !reloaded && !env_stale);
                        let (workspace, targets) = (&*served.workspace, served.targets);
                        current(kept, workspace, targets).and_then(|fresh| {
                            finish(
                                build.insert(fresh),
                                workspace,
                                (request, payload.out.as_deref(), &payload.args),
                                started,
                            )
                        })
                    })
                })
            }
            None => Err(anyhow::anyhow!("unknown request")),
        }));
        let done = answered.unwrap_or_else(|_| {
            *build = None;
            Err(anyhow::anyhow!(
                "the request panicked; the next request loads the workspace again"
            ))
        });
        if let Err(error) = &done
            && !crate::build::emit_early_json_error(error)
        {
            drop(writeln!(std::io::stderr(), "error: {error:#}"));
        }
        proto::restore(saved, fds);
        done.unwrap_or(1)
    }

    /// Returns `build` brought up to date with the member files, or a build from nothing when there is none or the refresh fails, so a failed refresh never leaves a half-updated build behind.
    fn current(
        build: Option<Build>,
        workspace: &graph::Workspace,
        targets: &Targets,
    ) -> Result<Build> {
        match build {
            Some(mut build) => build.refresh(workspace, targets).map(|()| build),
            None => Build::load(workspace, targets),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn socket_key_changes_for_behavioral_env_toggles() {
            let root = Path::new("/tmp/sample-ws");
            let base = crate::config::Config::for_target(haste_codegen::Target::host());
            let k_base = socket_key_for(root, &base);
            let mut variants: [_; 7] = std::array::from_fn(|_| base.clone());
            variants[0].debug_info = !base.debug_info;
            variants[1].symbols = !base.symbols;
            variants[2].x64_baseline = !base.x64_baseline;
            variants[3].fn_align = Some(128);
            variants[4].use_substrate = !base.use_substrate;
            variants[5].verify_passes = !base.verify_passes;
            variants[6].verify_stencil = !base.verify_stencil;
            for variant in &variants {
                assert_ne!(socket_key_for(root, variant), k_base);
            }
        }

        #[test]
        fn daemon_args_forwards_all_selection_flags_and_jobs() {
            let sel = crate::Selection {
                packages: vec!["pkg".to_owned(), "pkg2".to_owned()],
                test: vec!["payments".to_owned(), "receipts".to_owned()],
                workspace: true,
                all_targets: true,
                tests: true,
                features: vec!["a".to_owned(), "b".to_owned()],
                locked: true,
                offline: true,
                quiet: true,
                message_format: crate::MessageFormat::Json,
                ..crate::Selection::default()
            };
            let args = daemon_args(&sel, true, true, 6);
            for expected in
                "--release --prepared --workspace --tests --all-targets --locked --offline -q"
                    .split_whitespace()
            {
                assert!(args.iter().any(|a| a == expected), "missing {expected}");
            }
            for pair in [
                ["-p", "pkg"],
                ["-p", "pkg2"],
                ["--test", "payments"],
                ["--test", "receipts"],
                ["-F", "a,b"],
                ["--message-format", "json"],
                ["-j", "6"],
            ] {
                assert!(args.windows(2).any(|w| w == pair), "missing {pair:?}");
            }
        }
    }
}
