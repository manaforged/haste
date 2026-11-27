//! Build scripts kept as executables between runs, with a pre-outline manifest so warm reruns exec cached binaries directly without loading `sysroot` or `outline`.

use std::fs::File;
use std::hash::{BuildHasher as _, Hash as _, Hasher as _};
#[cfg(unix)]
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use rustc_hash::{FxHashMap as HashMap, FxHasher};

use crate::config::Config;
use crate::graph::{Side, Targets, Workspace, store};
use crate::image::Script;
use crate::macros::Compilation;
use crate::outline::Outline;
use crate::outline::tree::Profile;
use crate::substrate::{Backing, CrateHashes};

const EXECUTABLE: &str = "script";
const KEY: &str = "script.key";
const FAST: &str = "script.fast";
const FAST_MAGIC: [u8; 8] = *b"HSTSCR01";
const FAST_VERSION: u32 = 2;
pub(super) const KEY_MAGIC: [u8; 8] = *b"HSTKEY01";
pub(super) const KEY_VERSION: u32 = 1;
const WATCH_HEADER_LEN: usize = 28;

/// What the store holds for a script whose crate hashes to its key: its executable, the mark that it does not compile ahead of time, or nothing.
pub(super) enum Kept {
    Executable(PathBuf),
    Failed,
    Missing,
}

/// Runs `f` with `Profile::Debug`, `debug_info = false`, and `symbols = false` so build-script compilation skips Release SSA optimization and inlining overhead.
pub(super) fn with_script_profile<R>(f: impl FnOnce() -> R) -> R {
    let cfg = crate::config();
    if cfg.profile == Profile::Debug && !cfg.debug_info && !cfg.symbols {
        return f();
    }
    let mut script_cfg = (*cfg).clone();
    script_cfg.profile = Profile::Debug;
    script_cfg.debug_info = false;
    script_cfg.symbols = false;
    crate::with_config(Arc::new(script_cfg), f)
}

/// Checks whether `unit`'s cached build-script executable is current from `script.fast` alone, without parsing `build.rs` or loading `sysroot`/`outline`.
pub(super) fn fast_kept(
    workspace: &Workspace,
    targets: &Targets,
    unit: usize,
    dir: &Path,
) -> Option<PathBuf> {
    let executable = dir.join(EXECUTABLE);
    if !executable.is_file() {
        return None;
    }
    let expected = fast_meta_hash(workspace, targets, unit);
    validate_watch_manifest(&dir.join(FAST), FAST_MAGIC, FAST_VERSION, expected)
        .then_some(executable)
}

/// Computes the pre-outline metadata key for `unit`'s compiled build script under the active `Config` and host `targets`.
pub(super) fn fast_meta_hash(workspace: &Workspace, targets: &Targets, unit: usize) -> u64 {
    let script_host = haste_codegen::Target::host().effort(Profile::Debug.into());
    let tc_key = crate::substrate::toolchain_key(workspace, &script_host);
    let t_hash = crate::substrate::target_hash(&script_host);

    let mut closure: Vec<usize> = crate::outline::build_closure(workspace, &[unit])
        .into_iter()
        .collect();
    if !closure.contains(&unit) {
        closure.push(unit);
    }
    closure.sort_unstable();

    let mut hasher = FxHasher::default();
    FAST_MAGIC.hash(&mut hasher);
    FAST_VERSION.hash(&mut hasher);
    store::program().hash(&mut hasher);
    tc_key.hash(&mut hasher);
    t_hash.hash(&mut hasher);
    (
        &targets.host.triple,
        &targets.host.names,
        &targets.host.pairs,
        &targets.host.internal,
    )
        .hash(&mut hasher);
    (crate::profile() == Profile::Release).hash(&mut hasher);
    workspace.rustflags.hash(&mut hasher);
    workspace.config_env.hash(&mut hasher);
    (
        workspace.profile_dev.debug_assertions,
        workspace.profile_dev.overflow_checks,
        workspace.profile_release.debug_assertions,
        workspace.profile_release.overflow_checks,
    )
        .hash(&mut hasher);
    closure.len().hash(&mut hasher);
    for &u in &closure {
        let Some(entry) = workspace.graph.units.get(u) else {
            continue;
        };
        let Some(pkg) = workspace.packages.get(entry.package) else {
            continue;
        };
        pkg.locked.name.hash(&mut hasher);
        pkg.locked.version.to_string().hash(&mut hasher);
        pkg.locked.source.hash(&mut hasher);
        pkg.manifest.dir.hash(&mut hasher);
        pkg.manifest.edition.hash(&mut hasher);
        pkg.manifest.lib_name.hash(&mut hasher);
        pkg.manifest.build.hash(&mut hasher);
        pkg.manifest.links.hash(&mut hasher);
        (entry.side == Side::Host).hash(&mut hasher);
        entry.features.hash(&mut hasher);
        entry.deps.len().hash(&mut hasher);
        for (dep_name, dep_u, kind) in &entry.deps {
            let dep_pkg_name = workspace
                .graph
                .units
                .get(*dep_u)
                .and_then(|du| workspace.packages.get(du.package))
                .map_or("", |dp| dp.locked.name.as_str());
            dep_name.hash(&mut hasher);
            dep_pkg_name.hash(&mut hasher);
            (*kind as u8).hash(&mut hasher);
        }
        if u != unit
            && let Some(out) = &entry.script
        {
            (&out.names, &out.pairs, &out.vars).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Writes `script.fast` for `unit` given `meta_hash`, the exact source/included paths, and compile-time env reads.
pub(super) fn write_fast_manifest(
    dir: &Path,
    meta_hash: u64,
    paths: &[PathBuf],
    envs: &[(String, Option<u64>)],
) -> std::io::Result<()> {
    let Some(bytes) = encode_watch_manifest(FAST_MAGIC, FAST_VERSION, meta_hash, paths, envs)
    else {
        return Ok(());
    };
    store::write_new_bytes(&dir.join(FAST), &bytes)
}

#[cfg(test)]
pub(super) fn hash_fast_env(name: &str, val: &str) -> u64 {
    store::hash_env_val(&FAST_MAGIC, name, val)
}

pub(super) fn encode_watch_manifest(
    magic: [u8; 8],
    version: u32,
    key: u64,
    paths: &[PathBuf],
    envs: &[(String, Option<u64>)],
) -> Option<Vec<u8>> {
    let file_count = u32::try_from(paths.len()).ok()?;
    let env_count = u32::try_from(envs.len()).ok()?;
    let stamps = store::file_stamps(paths);
    let mut out = Vec::with_capacity(WATCH_HEADER_LEN + paths.len() * 88 + envs.len() * 32);
    out.extend_from_slice(&magic);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&key.to_le_bytes());
    out.extend_from_slice(&file_count.to_le_bytes());
    out.extend_from_slice(&env_count.to_le_bytes());
    for (path, stamp) in paths.iter().zip(stamps) {
        write_path_ref(&mut out, path)?;
        match stamp {
            Some((len, nanos)) => {
                out.push(1);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&nanos.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    for (name, val) in envs {
        let bytes = name.as_bytes();
        let len = u32::try_from(bytes.len()).ok()?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bytes);
        match val {
            Some(hash) => {
                out.push(1);
                out.extend_from_slice(&hash.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    Some(out)
}

pub(super) fn validate_watch_manifest(
    manifest_path: &Path,
    magic: [u8; 8],
    version: u32,
    expected_key: u64,
) -> bool {
    let Ok(backing) = Backing::open(manifest_path, WATCH_HEADER_LEN) else {
        return false;
    };
    let data = backing.as_slice();
    if data.len() < WATCH_HEADER_LEN
        || data[0..8] != magic
        || read_u32(data, 8) != version
        || read_u64(data, 12) != expected_key
    {
        return false;
    }
    let file_count = read_u32(data, 20) as usize;
    let env_count = read_u32(data, 24) as usize;
    let mut pos = WATCH_HEADER_LEN;
    for _ in 0..file_count {
        let Some(raw_len) = read_bytes(data, &mut pos, 4) else {
            return false;
        };
        let len = read_u32(raw_len, 0) as usize;
        let Some(raw_path) = read_bytes(data, &mut pos, len) else {
            return false;
        };
        let Some(present) = read_bytes(data, &mut pos, 1).and_then(|b| b.first().copied()) else {
            return false;
        };
        let expected = match present {
            0 => None,
            1 => {
                let Some(raw) = read_bytes(data, &mut pos, 24) else {
                    return false;
                };
                Some((read_u64(raw, 0), read_i128(raw, 8)))
            }
            _ => return false,
        };
        if !stamp_matches_raw(raw_path, expected) {
            return false;
        }
    }
    for _ in 0..env_count {
        let Some(raw_len) = read_bytes(data, &mut pos, 4) else {
            return false;
        };
        let len = read_u32(raw_len, 0) as usize;
        let Some(raw_name) = read_bytes(data, &mut pos, len) else {
            return false;
        };
        let Ok(name) = std::str::from_utf8(raw_name) else {
            return false;
        };
        let Some(present) = read_bytes(data, &mut pos, 1).and_then(|b| b.first().copied()) else {
            return false;
        };
        let expected = match present {
            0 => None,
            1 => {
                let Some(raw) = read_bytes(data, &mut pos, 8) else {
                    return false;
                };
                Some(read_u64(raw, 0))
            }
            _ => return false,
        };
        let actual = crate::process_env_var(name)
            .as_deref()
            .map(|v| store::hash_env_val(&magic, name, v));
        if actual != expected {
            return false;
        }
    }
    pos == data.len()
}

fn closure_crates(outline: &Outline, krate: usize) -> Vec<usize> {
    let mut seen = vec![false; outline.crates.len()];
    let mut stack = vec![krate];
    let mut found = Vec::new();
    while let Some(at) = stack.pop() {
        if std::mem::replace(&mut seen[at], true) {
            continue;
        }
        found.push(at);
        stack.extend(outline.deps[at].iter().map(|(_, dep)| *dep));
    }
    found
}

fn collect_closure_files(outline: &Outline, krate: usize, include_sources: bool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for at in closure_crates(outline, krate) {
        if let Some(tree) = outline.crates.get(at)
            && !tree.env.fixed
        {
            if include_sources {
                for entry in &tree.files {
                    if !entry.expanded {
                        paths.push(entry.source.path.clone());
                    }
                }
            }
            for (inc, _) in outline.included(at) {
                paths.push(inc);
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn collect_closure_envs(
    outline: &Outline,
    krate: usize,
    magic: &[u8; 8],
) -> Vec<(String, Option<u64>)> {
    let mut merged = std::collections::BTreeMap::new();
    for at in closure_crates(outline, krate) {
        if let Some(tree) = outline.crates.get(at) {
            let reads = crate::sync::lock(&tree.env.env_reads);
            for (name, val) in reads.iter() {
                let hashed = val.as_deref().map(|v| store::hash_env_val(magic, name, v));
                merged.insert(name.clone(), hashed);
            }
        }
    }
    merged.into_iter().collect()
}

/// The key of each script unit's executable: the hash of its crate and of every crate that crate reaches, under this toolchain, target, profile and compiler.
pub(super) struct Binaries {
    keys: HashMap<usize, u64>,
    dirs: HashMap<usize, PathBuf>,
    outer_cfg: Arc<Config>,
    outer_targets: Targets,
}

impl Binaries {
    /// The keys of the scripts `scripts`, each a unit and its crate in `outline`; none while no script of them ran before, as a first run keeps nothing.
    pub(super) fn new(
        workspace: &Workspace,
        outline: &Outline,
        scripts: &[(usize, usize)],
        dirs: HashMap<usize, PathBuf>,
        outer_cfg: Arc<Config>,
        outer_targets: Targets,
    ) -> Binaries {
        if !scripts
            .iter()
            .any(|&(unit, _)| dirs.get(&unit).is_some_and(|dir| dir_ran_before(dir)))
        {
            return Binaries {
                keys: HashMap::default(),
                dirs,
                outer_cfg,
                outer_targets,
            };
        }
        let target = crate::host_target();
        let hashes = CrateHashes::compute(
            outline,
            crate::substrate::toolchain_key(workspace, &target),
            crate::substrate::target_hash(&target),
        );
        Binaries {
            keys: scripts
                .iter()
                .map(|&(unit, krate)| (unit, closure_key(outline, &hashes, krate)))
                .collect(),
            dirs,
            outer_cfg,
            outer_targets,
        }
    }

    fn unit_dir(&self, workspace: &Workspace, unit: usize) -> PathBuf {
        self.dirs.get(&unit).cloned().unwrap_or_else(|| {
            crate::with_config(Arc::clone(&self.outer_cfg), || store::dir(workspace, unit))
        })
    }

    /// What the store holds for the script of `unit` under its key; the caller holds the unit's lock.
    pub(super) fn kept(&self, workspace: &Workspace, unit: usize) -> Kept {
        let dir = self.unit_dir(workspace, unit);
        let Some(&key) = self.keys.get(&unit) else {
            return Kept::Missing;
        };
        if !validate_watch_manifest(&dir.join(KEY), KEY_MAGIC, KEY_VERSION, key) {
            return Kept::Missing;
        }
        let executable = dir.join(EXECUTABLE);
        if executable.is_file() {
            Kept::Executable(executable)
        } else {
            Kept::Failed
        }
    }

    /// Compiles the scripts `scripts`, each a unit and the crate of its `main`, ahead of time into the store in a detached process that outlives this one on Linux, or fast Debug AOT elsewhere; the keeper lets go of this process's unit locks `locks`.
    pub(super) fn keep_detached(
        &self,
        workspace: &Workspace,
        compilation: &mut Compilation,
        scripts: &[(usize, usize)],
        locks: &mut HashMap<usize, store::Lock>,
    ) -> anyhow::Result<()> {
        if scripts.is_empty() {
            return Ok(());
        }
        #[cfg(not(target_os = "linux"))]
        {
            locks.clear();
            for &(unit, krate) in scripts {
                drop(self.keep(workspace, compilation, unit, krate));
            }
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            let child = crate::image::fork_child(|| {
                locks.clear();
                let keeper = crate::image::fork_child(|| {
                    detach();
                    let kept = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        scripts
                            .iter()
                            .filter(|&&(unit, krate)| {
                                self.keep(workspace, compilation, unit, krate).is_err()
                            })
                            .count()
                    }));
                    i32::from(!matches!(kept, Ok(0)))
                });
                i32::from(keeper.is_err())
            })?;
            match crate::image::wait(child) {
                0 => Ok(()),
                status => anyhow::bail!("the keeper did not start: status {status}"),
            }
        }
    }

    /// Compiles the script of `unit`, whose `main` is in crate `krate`, ahead of time into the store under its key, or marks that it does not compile, holding the unit's lock.
    fn keep(
        &self,
        workspace: &Workspace,
        compilation: &mut Compilation,
        unit: usize,
        krate: usize,
    ) -> std::io::Result<()> {
        let Some(&key) = self.keys.get(&unit) else {
            return Ok(());
        };
        let dir = self.unit_dir(workspace, unit);
        let _lock =
            crate::with_config(Arc::clone(&self.outer_cfg), || store::lock(workspace, unit))?;
        let executable = dir.join(EXECUTABLE);
        store::remove(&dir.join(KEY))?;
        store::remove(&dir.join(FAST))?;
        store::remove(&executable)?;
        let staged = store::fresh(&executable);
        let built = compilation
            .outline
            .main(krate)
            .ok_or_else(|| anyhow::anyhow!("no `fn main`"))
            .and_then(|main| crate::build::emit_main(workspace, compilation, main, &staged));
        match built {
            Ok(()) => {
                std::fs::rename(&staged, &executable)?;
                let meta = crate::with_config(Arc::clone(&self.outer_cfg), || {
                    fast_meta_hash(workspace, &self.outer_targets, unit)
                });
                let files = collect_closure_files(&compilation.outline, krate, true);
                let fast_envs = collect_closure_envs(&compilation.outline, krate, &FAST_MAGIC);
                drop(write_fast_manifest(&dir, meta, &files, &fast_envs));
            }
            Err(_) => store::remove(&staged)?,
        }
        let included = collect_closure_files(&compilation.outline, krate, false);
        let key_envs = collect_closure_envs(&compilation.outline, krate, &KEY_MAGIC);
        match encode_watch_manifest(KEY_MAGIC, KEY_VERSION, key, &included, &key_envs) {
            Some(bytes) => store::write_new_bytes(&dir.join(KEY), &bytes),
            None => Ok(()),
        }
    }
}

/// Detaches this forked keeper from the session and from every pipe and socket it shares with its parent, so neither a terminal signal nor a reader waiting for end of file ties the two together.
#[cfg(target_os = "linux")]
fn detach() {
    let shared: Vec<i32> = std::fs::read_dir("/dev/fd")
        .map(|dir| {
            dir.filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    for fd in shared.into_iter().filter(|&fd| fd > 2) {
        // SAFETY: fstat writes only into the zeroed local `stat`; a descriptor that is not open fails without effect.
        let mode = unsafe {
            let mut stat = std::mem::zeroed::<libc::stat>();
            if libc::fstat(fd, &mut stat) != 0 {
                continue;
            }
            stat.st_mode & libc::S_IFMT
        };
        if mode == libc::S_IFIFO || mode == libc::S_IFSOCK {
            // SAFETY: fd is a pipe or socket this keeper inherited and never uses; closing it drops only this process's reference.
            unsafe { libc::close(fd) };
        }
    }
    // SAFETY: setsid, alarm, open and dup2 take only the static NUL-terminated path and descriptors of this process; 0, 1 and 2 then point at /dev/null.
    unsafe {
        libc::setsid();
        libc::alarm(60);
        let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
        if null >= 0 {
            for fd in 0..3 {
                libc::dup2(null, fd);
            }
            if null > 2 {
                libc::close(null);
            }
        }
    }
}

/// Drops the executable and fast manifest of `unit`'s script after it failed, so later runs compile the script on demand and give their own verdict; the caller holds the unit's lock.
pub(super) fn distrust(workspace: &Workspace, unit: usize) {
    let dir = store::dir(workspace, unit);
    for file in [EXECUTABLE, FAST] {
        if let Err(error) = store::remove(&dir.join(file)) {
            eprintln!(
                "cannot drop cached build script artifact of {}: {error}",
                super::package_name(workspace, unit)
            );
        }
    }
}

fn dir_ran_before(dir: &Path) -> bool {
    dir.join(super::OUT).is_dir()
}

/// Whether the script of `unit` ran before, so its `OUT_DIR` is there and a change it watches is what runs it again.
pub(super) fn ran_before(workspace: &Workspace, unit: usize) -> bool {
    dir_ran_before(&store::dir(workspace, unit))
}

fn configure_command(
    executable: &Path,
    script: &Script,
    client_env: Option<&rustc_hash::FxHashMap<String, String>>,
    log: File,
) -> Command {
    let mut cmd = Command::new(executable);
    cmd.current_dir(&script.dir).stderr(log);
    if let Some(map) = client_env {
        cmd.env_clear();
        cmd.envs(map.iter());
    }
    cmd.envs(script.vars.iter().map(|(key, value)| (key, value)));
    #[cfg(unix)]
    {
        cmd.arg0("build-script-build");
    }
    cmd
}

/// Runs `script` from `executable` with `out` as its standard output, as the image runs a script's `main`; returns the exit status when it cannot start.
pub(super) fn exec(executable: &Path, script: &Script, out: File) -> i32 {
    let Ok(log) = File::create(&script.log) else {
        return 101;
    };
    let mut cmd = configure_command(executable, script, None, log);
    cmd.stdout(out);
    #[cfg(unix)]
    {
        let error = cmd.exec();
        eprintln!("cannot start {}: {error}", executable.display());
        127
    }
    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(status) => status.code().unwrap_or(1),
            Err(error) => {
                eprintln!("cannot start {}: {error}", executable.display());
                127
            }
        }
    }
}

/// Runs `script` directly from `executable` via `Command` (without `fork` or `Jit`) and captures `(exit_code, stdout_bytes)`.
pub(super) fn exec_capture(
    executable: &Path,
    script: &Script,
    client_env: Option<&rustc_hash::FxHashMap<String, String>>,
) -> Result<(i32, Vec<u8>), String> {
    let log = File::create(&script.log)
        .map_err(|error| format!("cannot create {}: {error}", script.log.display()))?;
    let mut cmd = configure_command(executable, script, client_env, log);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped());
    let output = cmd
        .output()
        .map_err(|error| format!("cannot start {}: {error}", executable.display()))?;
    Ok((output.status.code().unwrap_or(1), output.stdout))
}

/// The hash of crate `krate` and of every crate it reaches in `outline`, since a fixed crate's hash leaves out its build dependencies.
fn closure_key(outline: &Outline, hashes: &CrateHashes, krate: usize) -> u64 {
    let mut found: Vec<Option<u64>> = closure_crates(outline, krate)
        .into_iter()
        .map(|at| hashes.get(at))
        .collect();
    found.sort_unstable();
    rustc_hash::FxBuildHasher.hash_one((hashes.get(krate), found))
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(len)?;
    let slice = data.get(*pos..end)?;
    *pos = end;
    Some(slice)
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[offset..offset + 4]);
    u32::from_le_bytes(buf)
}

fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[offset..offset + 8]);
    u64::from_le_bytes(buf)
}

fn read_i128(data: &[u8], offset: usize) -> i128 {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&data[offset..offset + 16]);
    i128::from_le_bytes(buf)
}

fn write_path_ref(out: &mut Vec<u8>, path: &Path) -> Option<()> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt as _;
        path.as_os_str().as_bytes()
    };
    #[cfg(not(unix))]
    let bytes = path.to_str()?.as_bytes();
    let len = u32::try_from(bytes.len()).ok()?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Some(())
}

fn stamp_matches_raw(raw_path: &[u8], expected: store::FileStamp) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        store::file_stamp(Path::new(std::ffi::OsStr::from_bytes(raw_path))) == expected
    }
    #[cfg(not(unix))]
    {
        let Ok(text) = std::str::from_utf8(raw_path) else {
            return false;
        };
        store::file_stamp(Path::new(text)) == expected
    }
}
