//! Embeddable Rust API for checking, running, testing, and emitting executables with Haste.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::build::Build;
use crate::config::Config;
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
        self.config.selection.package = Some(package.into());
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

    fn load_build(path: &Path) -> Result<(Workspace, Build, PathBuf), Error> {
        let root = std::path::absolute(path)
            .map_err(|err| Error::from_anyhow(ErrorKind::Io, err.into()))?;
        std::fs::metadata(&root).map_err(|err| {
            Error::from_anyhow(
                ErrorKind::Io,
                anyhow::Error::new(err).context(format!("accessing {}", root.display())),
            )
        })?;
        let targets = Targets {
            target: cfg::Target::of(crate::compile_target()),
            host: cfg::Target::host(),
        };
        let workspace = crate::cli::load_workspace(&root, &targets).map_err(Error::compile)?;
        let build = Build::load(&workspace, &targets).map_err(Error::compile)?;
        Ok((workspace, build, root))
    }

    /// Type-checks, borrow-checks, and lowers reached functions in the workspace at `path`.
    pub fn check(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        self.run_scoped(false, || {
            let started = Instant::now();
            let (workspace, mut build, _) = Self::load_build(path)?;
            build
                .compile_binary_oneshot(&workspace)
                .map_err(Error::compile)?;
            let status = build.check(started.elapsed()).map_err(Error::compile)?;
            if status != 0 {
                return Err(Error::new(ErrorKind::Compile, "compilation failed"));
            }
            Ok(())
        })
    }

    /// Compiles the binary at `path` and writes a standalone native AOT executable to `out`.
    pub fn emit(&self, path: impl AsRef<Path>, out: impl AsRef<Path>) -> Result<(), Error> {
        let (path, out) = (path.as_ref(), out.as_ref());
        self.run_scoped(false, || {
            let (workspace, mut build, _) = Self::load_build(path)?;
            build
                .emit_executable(&workspace, out, true)
                .map_err(Error::compile)?;
            Ok(())
        })
    }

    /// Runs the binary's `main` at `path` via JIT with `args` (after program name) and returns its exit status.
    pub fn run(&self, path: impl AsRef<Path>, args: &[String]) -> Result<i32, Error> {
        let path = path.as_ref();
        self.run_scoped(false, || {
            Self::require_host_target("run")?;
            let (workspace, mut build, root) = Self::load_build(path)?;
            let argv: Vec<String> = std::iter::once(crate::cli::program_name(&root))
                .chain(args.iter().cloned())
                .collect();
            build.run(&workspace, &argv).map_err(Error::runtime)
        })
    }

    /// Runs `#[test]` functions in the workspace at `path` matching `filters` and returns 0 when all pass, else 101.
    pub fn test(&self, path: impl AsRef<Path>, filters: &[String]) -> Result<i32, Error> {
        let path = path.as_ref();
        self.run_scoped(true, || {
            Self::require_host_target("test")?;
            let started = Instant::now();
            let (workspace, mut build, _) = Self::load_build(path)?;
            build.compile_cases(&workspace).map_err(Error::compile)?;
            build
                .report(started, started.elapsed(), filters)
                .map_err(Error::runtime)
        })
    }
}
