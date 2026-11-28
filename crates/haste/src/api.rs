//! Embeddable Rust API for checking, running, testing, and emitting executables with Haste.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::build::Build;
use crate::config::Config;
use crate::graph::store::FileStamp;
use crate::graph::{Targets, Workspace, cfg};
use crate::outline::tree::Profile;

/// A supported compilation target triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target(&'static haste_codegen::Target);

impl Target {
    /// Returns the host machine's native compilation target.
    pub fn host() -> Self {
        Self(haste_codegen::Target::host())
    }

    /// Finds a supported target by its Rust target triple.
    pub fn find(triple: &str) -> Option<Self> {
        haste_codegen::Target::find(triple).map(Self)
    }

    /// Returns all supported compilation targets.
    pub fn supported() -> Vec<Self> {
        haste_codegen::TARGETS.iter().map(Self).collect()
    }

    /// Returns the Rust target triple string for this target.
    pub fn triple(self) -> &'static str {
        self.0.triple
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.triple)
    }
}

impl AsRef<str> for Target {
    fn as_ref(&self) -> &str {
        self.0.triple
    }
}

impl From<Target> for String {
    fn from(target: Target) -> Self {
        target.0.triple.to_owned()
    }
}

/// Category of failure returned by `Compiler` operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Invalid builder option, conflicting selection, or unsupported target triple.
    Config,
    /// Workspace loading, type checking, borrow checking, lowering, or executable emission failed.
    Compile,
    /// JIT execution or test runner failed.
    Runtime,
    /// Filesystem or process I/O failed.
    Io,
}

/// An error returned by `CompilerBuilder::build` or `Compiler` operations.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn from_anyhow(kind: ErrorKind, err: anyhow::Error) -> Self {
        let kind = if err.chain().any(|cause| cause.is::<std::io::Error>()) {
            ErrorKind::Io
        } else {
            kind
        };
        Self {
            kind,
            message: format!("{err:#}"),
        }
    }

    fn compile(err: anyhow::Error) -> Self {
        Self::from_anyhow(ErrorKind::Compile, err)
    }

    fn runtime(err: anyhow::Error) -> Self {
        Self::from_anyhow(ErrorKind::Runtime, err)
    }

    /// The category of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// Configures and validates a `Compiler` instance.
#[must_use]
#[derive(Clone, Debug)]
pub struct CompilerBuilder {
    config: Config,
    target_name: Option<String>,
    jobs_override: Option<usize>,
    debug_info_override: Option<bool>,
    symbols_override: Option<bool>,
}

impl Default for CompilerBuilder {
    fn default() -> Self {
        Self {
            config: Config::from_env(),
            target_name: None,
            jobs_override: None,
            debug_info_override: None,
            symbols_override: None,
        }
    }
}

impl CompilerBuilder {
    /// Enables or disables the release optimization profile.
    pub fn release(mut self, release: bool) -> Self {
        self.config.profile = if release {
            Profile::Release
        } else {
            Profile::Debug
        };
        self
    }

    /// Sets the target triple for ahead-of-time executable emission.
    pub fn target(mut self, triple: impl Into<String>) -> Self {
        self.target_name = Some(triple.into());
        self
    }

    /// Sets the maximum worker thread count (must be at least 1).
    pub fn jobs(mut self, jobs: usize) -> Self {
        self.jobs_override = Some(jobs);
        self
    }

    /// Selects a specific workspace package.
    pub fn package(mut self, package: impl Into<String>) -> Self {
        self.config.selection.packages.push(package.into());
        self
    }

    /// Selects a specific binary target.
    pub fn bin(mut self, bin: impl Into<String>) -> Self {
        self.config.selection.bin = Some(bin.into());
        self
    }

    /// Selects a specific example target.
    pub fn example(mut self, example: impl Into<String>) -> Self {
        self.config.selection.example = Some(example.into());
        self
    }

    /// Enables one Cargo feature on the selected workspace packages.
    pub fn feature(mut self, feature: impl Into<String>) -> Self {
        self.config.selection.features.push(feature.into());
        self
    }

    /// Enables all Cargo features on the selected workspace packages.
    pub fn all_features(mut self, enabled: bool) -> Self {
        self.config.selection.all_features = enabled;
        self
    }

    /// Disables the `default` Cargo feature on the selected workspace packages.
    pub fn no_default_features(mut self, disabled: bool) -> Self {
        self.config.selection.no_default_features = disabled;
        self
    }

    /// Controls DWARF `.debug_line` emission in standalone AOT executables.
    pub fn debug_info(mut self, enabled: bool) -> Self {
        self.debug_info_override = Some(enabled);
        self
    }

    /// Controls symbol-table emission in standalone AOT executables.
    pub fn symbols(mut self, enabled: bool) -> Self {
        self.symbols_override = Some(enabled);
        self
    }

    /// Forces baseline SSE2 x86-64 code generation instead of host CPU feature detection.
    pub fn x64_baseline(mut self, baseline: bool) -> Self {
        self.config.x64_baseline = baseline;
        self
    }

    /// Enables or disables persistent sysroot and workspace substrate caching.
    pub fn substrate(mut self, enabled: bool) -> Self {
        self.config.use_substrate = enabled;
        self
    }

    /// Validates all settings and constructs a `Compiler`.
    pub fn build(mut self) -> Result<Compiler, Error> {
        if let Some(jobs) = self.jobs_override {
            if jobs == 0 {
                return Err(Error::new(ErrorKind::Config, "jobs must be at least 1"));
            }
            self.config.jobs = jobs;
        }
        if self.config.selection.bin.is_some() && self.config.selection.example.is_some() {
            return Err(Error::new(
                ErrorKind::Config,
                "cannot select both `bin` and `example` targets",
            ));
        }
        if let Some(triple) = self.target_name {
            self.config.target = crate::config::find_target(&triple)
                .map_err(|err| Error::new(ErrorKind::Config, format!("{err:#}")))?;
        }
        self.config.refresh_debug_flags();
        if let Some(debug_info) = self.debug_info_override {
            self.config.debug_info = debug_info;
            if debug_info && self.symbols_override.is_none() {
                self.config.symbols = true;
            }
        }
        if let Some(symbols) = self.symbols_override {
            self.config.symbols = symbols;
        }
        Ok(Compiler {
            config: Arc::new(self.config),
        })
    }
}

/// An isolated Haste compiler instance configured without process-global state.
#[derive(Clone, Debug)]
pub struct Compiler {
    config: Arc<Config>,
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Compiler>();
    assert_send_sync::<CompilerBuilder>();
    assert_send_sync::<Error>();
};

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    /// Creates a `Compiler` using default host settings and the current environment.
    pub fn new() -> Self {
        Self {
            config: Arc::new(Config::from_env()),
        }
    }

    /// Returns a builder for customizing target, profile, features, and codegen flags.
    pub fn builder() -> CompilerBuilder {
        CompilerBuilder::default()
    }

    fn run_scoped<R>(
        &self,
        test_mode: bool,
        f: impl FnOnce() -> Result<R, Error>,
    ) -> Result<R, Error> {
        crate::sema::evaluate::install(crate::image::evaluator);
        let cfg = if self.config.test_mode == test_mode {
            Arc::clone(&self.config)
        } else {
            let mut cloned = (*self.config).clone();
            cloned.test_mode = test_mode;
            Arc::new(cloned)
        };
        crate::with_config(cfg, f)
    }

    fn require_host_target(op: &str) -> Result<(), Error> {
        let active = crate::compile_target().triple;
        if active != haste_codegen::Target::host().triple {
            return Err(Error::new(
                ErrorKind::Config,
                format!("cannot JIT-{op} cross-target `{active}`; use `emit` instead"),
            ));
        }
        Ok(())
    }

    fn resolve_root(path: &Path) -> Result<PathBuf, Error> {
        let root = std::path::absolute(path)
            .map_err(|err| Error::from_anyhow(ErrorKind::Io, err.into()))?;
        std::fs::metadata(&root).map_err(|err| {
            Error::from_anyhow(
                ErrorKind::Io,
                anyhow::Error::new(err).context(format!("accessing {}", root.display())),
            )
        })?;
        Ok(root)
    }

    fn targets() -> Targets {
        Targets {
            target: cfg::Target::of(crate::compile_target()),
            host: cfg::Target::host(),
        }
    }

    fn load_build(path: &Path) -> Result<(Workspace, Build, PathBuf), Error> {
        let root = Self::resolve_root(path)?;
        let targets = Self::targets();
        let workspace = crate::cli::load_workspace(&root, &targets).map_err(Error::compile)?;
        let build = Build::load(&workspace, &targets).map_err(Error::compile)?;
        Ok((workspace, build, root))
    }

    fn batch_compile_error(diagnostics: Option<String>, fallback: &str) -> Error {
        Error::new(
            ErrorKind::Compile,
            diagnostics.unwrap_or_else(|| fallback.to_owned()),
        )
    }

    fn run_check(workspace: &Workspace, build: &mut Build, oneshot: bool) -> Result<(), Error> {
        let started = Instant::now();
        if oneshot {
            build.compile_binary_oneshot(workspace)
        } else {
            build.compile_binary(workspace)
        }
        .map_err(Error::compile)?;
        let status = build.check(started.elapsed()).map_err(Error::compile)?;
        if status != 0 {
            return Err(Self::batch_compile_error(
                build.binary_diagnostics(),
                "compilation failed",
            ));
        }
        Ok(())
    }

    fn run_emit(
        workspace: &Workspace,
        build: &mut Build,
        out: &Path,
        drop_finals: bool,
    ) -> Result<(), Error> {
        if let Err(err) = build.emit_executable(workspace, out, drop_finals) {
            return Err(build.binary_diagnostics().map_or_else(
                || Error::compile(err),
                |msg| Error::new(ErrorKind::Compile, msg),
            ));
        }
        Ok(())
    }

    fn run_jit(
        workspace: &Workspace,
        build: &mut Build,
        root: &Path,
        args: &[String],
    ) -> Result<i32, Error> {
        let argv: Vec<String> = std::iter::once(crate::cli::program_name(root))
            .chain(args.iter().cloned())
            .collect();
        build.run(workspace, &argv).map_err(Error::runtime)
    }

    fn run_tests(
        workspace: &Workspace,
        build: &mut Build,
        filters: &[String],
    ) -> Result<i32, Error> {
        let started = Instant::now();
        build
            .compile_cases_filtered(workspace, filters)
            .map_err(Error::compile)?;
        if let Some(msg) = build.cases_diagnostics() {
            return Err(Error::new(ErrorKind::Compile, msg));
        }
        build
            .report(started, started.elapsed(), filters)
            .map_err(Error::runtime)
    }

    /// Opens an incremental compilation session keeping a warm in-process build for `path`.
    pub fn session(&self, path: impl AsRef<Path>) -> Result<Session, Error> {
        let root = Self::resolve_root(path.as_ref())?;
        let (targets, regular) = self.run_scoped(false, || {
            let targets = Self::targets();
            let regular = LoadedBuild::load(&root, &targets, None)?;
            Ok((targets, regular))
        })?;
        Ok(Session {
            compiler: self.clone(),
            root,
            targets,
            regular: Some(regular),
            binary_is_aot: false,
            tests: None,
        })
    }

    /// Type-checks, borrow-checks, and lowers reached functions in the workspace at `path`.
    pub fn check(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        self.run_scoped(false, || {
            let (workspace, mut build, _) = Self::load_build(path)?;
            Self::run_check(&workspace, &mut build, true)
        })
    }

    /// Compiles the binary at `path` and writes a standalone native AOT executable to `out`.
    pub fn emit(&self, path: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<(), Error> {
        let (path, out) = (path.as_ref(), out.as_ref());
        self.run_scoped(false, || {
            let (workspace, mut build, _) = Self::load_build(path)?;
            Self::run_emit(&workspace, &mut build, out, true)
        })
    }

    /// Runs the binary's `main` at `path` via JIT with `args` (after program name) and returns its exit status.
    pub fn run(&self, path: impl AsRef<Path>, args: &[String]) -> Result<i32, Error> {
        let path = path.as_ref();
        self.run_scoped(false, || {
            Self::require_host_target("run")?;
            let (workspace, mut build, root) = Self::load_build(path)?;
            Self::run_jit(&workspace, &mut build, &root, args)
        })
    }

    /// Runs `#[test]` functions in the workspace at `path` matching `filters` and returns 0 when all pass, else 101.
    pub fn test(&self, path: impl AsRef<Path>, filters: &[String]) -> Result<i32, Error> {
        let path = path.as_ref();
        self.run_scoped(true, || {
            Self::require_host_target("test")?;
            let (workspace, mut build, _) = Self::load_build(path)?;
            Self::run_tests(&workspace, &mut build, filters)
        })
    }
}

struct LoadedBuild {
    workspace: Workspace,
    build: Build,
    manifests: Vec<(PathBuf, FileStamp)>,
    watch_stamps: Vec<FileStamp>,
    test_filters: Option<Vec<String>>,
}

impl LoadedBuild {
    fn manifest_stamps(workspace: &Workspace, root: &Path) -> Vec<(PathBuf, FileStamp)> {
        crate::graph::manifest_paths(workspace, root)
            .into_iter()
            .map(|path| {
                let stamp = crate::graph::store::file_stamp(&path);
                (path, stamp)
            })
            .collect()
    }

    fn capture_watch_stamps(
        workspace: &Workspace,
        build: &Build,
        test_filters: Option<&[String]>,
    ) -> Vec<FileStamp> {
        if test_filters.is_some_and(|filters| !filters.is_empty()) {
            crate::graph::store::file_stamps(&build.watch_paths(workspace))
        } else {
            Vec::new()
        }
    }

    fn load(
        root: &Path,
        targets: &Targets,
        test_filters: Option<Vec<String>>,
    ) -> Result<Self, Error> {
        let workspace = crate::cli::load_workspace(root, targets).map_err(Error::compile)?;
        let build = Build::load(&workspace, targets).map_err(Error::compile)?;
        let manifests = Self::manifest_stamps(&workspace, root);
        let watch_stamps = Self::capture_watch_stamps(&workspace, &build, test_filters.as_deref());
        Ok(Self {
            workspace,
            build,
            manifests,
            watch_stamps,
            test_filters,
        })
    }

    fn is_stale(&self, next_filters: Option<&[String]>) -> bool {
        let manifests_stale = self
            .manifests
            .iter()
            .any(|(path, kept)| crate::graph::store::file_stamp(path) != *kept);
        let env_stale = self
            .build
            .env_reads()
            .into_iter()
            .any(|(name, prev)| crate::process_env_var(&name) != prev);
        let filters_stale =
            next_filters.is_some_and(|filters| self.test_filters.as_deref() != Some(filters));
        let filtered_sources_stale = next_filters.is_some_and(|filters| {
            !filters.is_empty()
                && self.watch_stamps
                    != Self::capture_watch_stamps(&self.workspace, &self.build, Some(filters))
        });
        manifests_stale || env_stale || filters_stale || filtered_sources_stale
    }

    fn ensure<'a>(
        slot: &'a mut Option<Self>,
        root: &Path,
        targets: &Targets,
        next_filters: Option<&[String]>,
    ) -> Result<&'a mut Self, Error> {
        let needs_reload = slot
            .as_ref()
            .is_none_or(|loaded| loaded.is_stale(next_filters));
        if needs_reload {
            *slot = Some(Self::load(
                root,
                targets,
                next_filters.map(<[String]>::to_vec),
            )?);
        } else if let Some(loaded) = slot.as_mut() {
            loaded
                .build
                .refresh(&loaded.workspace, targets)
                .map_err(Error::compile)?;
        }
        slot.as_mut()
            .ok_or_else(|| Error::new(ErrorKind::Compile, "session build missing"))
    }
}

/// An incremental compilation session keeping a warm in-process build across edits.
pub struct Session {
    compiler: Compiler,
    root: PathBuf,
    targets: Targets,
    regular: Option<LoadedBuild>,
    binary_is_aot: bool,
    tests: Option<LoadedBuild>,
}

impl Session {
    /// Refreshes changed files and type-checks, borrow-checks, and lowers reached functions.
    pub fn check(&mut self) -> Result<(), Error> {
        let compiler = self.compiler.clone();
        compiler.run_scoped(false, || {
            if self.binary_is_aot {
                self.regular = None;
                self.binary_is_aot = false;
            }
            let loaded = LoadedBuild::ensure(&mut self.regular, &self.root, &self.targets, None)?;
            Compiler::run_check(&loaded.workspace, &mut loaded.build, false)
        })
    }

    /// Refreshes changed files and writes a standalone native AOT executable to `out`.
    pub fn emit(&mut self, out: impl AsRef<Path>) -> Result<(), Error> {
        let out = out.as_ref();
        let compiler = self.compiler.clone();
        compiler.run_scoped(false, || {
            let loaded = LoadedBuild::ensure(&mut self.regular, &self.root, &self.targets, None)?;
            self.binary_is_aot = true;
            Compiler::run_emit(&loaded.workspace, &mut loaded.build, out, false)
        })
    }

    /// Refreshes changed files and runs the binary's `main` via JIT with `args`.
    pub fn run(&mut self, args: &[String]) -> Result<i32, Error> {
        let compiler = self.compiler.clone();
        compiler.run_scoped(false, || {
            Compiler::require_host_target("run")?;
            let loaded = LoadedBuild::ensure(&mut self.regular, &self.root, &self.targets, None)?;
            Compiler::run_jit(&loaded.workspace, &mut loaded.build, &self.root, args)
        })
    }

    /// Refreshes changed files and runs `#[test]` functions matching `filters`.
    pub fn test(&mut self, filters: &[String]) -> Result<i32, Error> {
        let compiler = self.compiler.clone();
        compiler.run_scoped(true, || {
            Compiler::require_host_target("test")?;
            let loaded =
                LoadedBuild::ensure(&mut self.tests, &self.root, &self.targets, Some(filters))?;
            Compiler::run_tests(&loaded.workspace, &mut loaded.build, filters)
        })
    }
}
