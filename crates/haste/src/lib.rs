//! Haste: the build graph, name resolution, type inference, lowering, and the JIT that runs the program.

#![cfg_attr(not(unix), allow(dead_code))]

#[cfg(not(any(
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "macos"),
    all(target_arch = "x86_64", target_os = "windows"),
    all(target_arch = "aarch64", target_os = "windows"),
)))]
compile_error!("haste currently supports x86_64/aarch64 Linux, macOS, and Windows hosts");

mod aot;
mod api;
mod body;
mod build;
mod cli;
pub(crate) mod config;
pub(crate) mod diag;
mod dump;
mod graph;
mod host;
mod image;
mod lower;
mod macros;
mod names;
mod outline;
mod prep;
mod scripts;
mod sema;
mod serve;
mod substrate;
mod sync;

use std::cell::{Cell, RefCell};
use std::sync::Arc;

/// How deep macro expansion, autoderef, trait selection and nested item frames go before failing, rustc's default `recursion_limit`.
pub(crate) const RECURSION_LIMIT: usize = 128;

/// Environment variable naming an explicit Rust standard library `library/` source directory.
pub(crate) const LIBRARY_ENV: &str = "HASTE_LIBRARY";

pub use api::{Compiler, CompilerBuilder, Error, ErrorKind, Target};
pub use cli::main;
pub use diag::{Diagnostic, DiagnosticKind, Label, Span};

/// Whether an optional environment variable value is enabled (non-empty and not `"0"`).
fn parse_env_flag(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether environment variable `name` is set to a non-empty value other than `"0"`.
pub(crate) fn env_flag(name: &str) -> bool {
    parse_env_flag(std::env::var_os(name).as_deref())
}

/// Whether an optional environment variable value is enabled, falling back to `default` when unset.
fn parse_env_flag_or(value: Option<&std::ffi::OsStr>, default: bool) -> bool {
    match value {
        Some(v) => parse_env_flag(Some(v)),
        None => default,
    }
}

/// Returns `env_flag(name)` when `name` is set in the environment, else `default`.
pub(crate) fn env_flag_or(name: &str, default: bool) -> bool {
    parse_env_flag_or(std::env::var_os(name).as_deref(), default)
}

/// Diagnostic output format selected on the command line.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) enum MessageFormat {
    #[default]
    Human,
    Short,
    Json,
    JsonRenderDiagnostics,
}

/// Package, binary, and feature selection from the command line or builder.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct Selection {
    pub package: Option<String>,
    pub bin: Option<String>,
    pub example: Option<String>,
    pub test: Option<String>,
    pub lib: bool,
    pub bins: bool,
    pub workspace: bool,
    pub all_targets: bool,
    pub tests: bool,
    pub features: Vec<String>,
    pub all_features: bool,
    pub no_default_features: bool,
    pub ignored_only: bool,
    pub include_ignored: bool,
    pub locked: bool,
    pub offline: bool,
    pub quiet: bool,
    pub message_format: MessageFormat,
}

thread_local! {
    static ACTIVE_CONFIG: RefCell<Option<Arc<config::Config>>> = const { RefCell::new(None) };
    static HOST_SCOPE: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` with a borrowed reference to this thread's active `Config` without cloning the `Arc`.
pub(crate) fn with_cfg<R>(f: impl FnOnce(&config::Config) -> R) -> R {
    ACTIVE_CONFIG.with(|slot| {
        if slot.borrow().is_none() {
            *slot.borrow_mut() = Some(Arc::new(config::Config::from_env()));
        }
        let guard = slot.borrow();
        let cfg = guard
            .as_ref()
            .expect("invariant: ACTIVE_CONFIG initialized above");
        f(cfg)
    })
}

/// Returns a cloned handle to the active build `Config` for propagating across worker threads.
pub(crate) fn config() -> Arc<config::Config> {
    with_cfg(|_| {
        ACTIVE_CONFIG.with(|slot| {
            Arc::clone(
                slot.borrow()
                    .as_ref()
                    .expect("invariant: ACTIVE_CONFIG initialized above"),
            )
        })
    })
}

/// Runs `f` with `cfg` installed as the active build configuration for this thread and any `sync::scope` workers it spawns.
pub(crate) fn with_config<R>(cfg: Arc<config::Config>, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<Arc<config::Config>>);
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE_CONFIG.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }
    let prev = ACTIVE_CONFIG.with(|slot| slot.borrow_mut().replace(cfg));
    let _guard = Guard(prev);
    f()
}

type ClientEnvMap = Arc<rustc_hash::FxHashMap<String, String>>;

thread_local! {
    static CLIENT_ENV: std::cell::RefCell<Option<ClientEnvMap>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn client_env() -> Option<ClientEnvMap> {
    CLIENT_ENV.with(|slot| slot.borrow().clone())
}

/// Runs `f` with `env` installed as the request-scoped client environment for `env!`/`option_env!` lookups.
pub(crate) fn with_client_env<R>(env: Option<ClientEnvMap>, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<ClientEnvMap>);
    impl Drop for Guard {
        fn drop(&mut self) {
            CLIENT_ENV.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }
    let prev = CLIENT_ENV.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), env));
    let _guard = Guard(prev);
    f()
}

/// Reads an environment variable from the active request's client environment when set, else the process environment.
pub(crate) fn process_env_var(name: &str) -> Option<String> {
    CLIENT_ENV.with(|slot| {
        if let Some(map) = slot.borrow().as_ref() {
            map.get(name).cloned()
        } else {
            std::env::var(name).ok()
        }
    })
}

/// How many processes or threads haste runs at once.
fn jobs() -> usize {
    with_cfg(|c| c.jobs)
}

/// The build profile (`Debug` or `Release`).
pub(crate) fn profile() -> outline::tree::Profile {
    with_cfg(|c| c.profile)
}

/// Whether this thread is currently inside `with_host_target`.
pub(crate) fn in_host_target() -> bool {
    HOST_SCOPE.with(Cell::get) > 0
}

/// The target being compiled for: the host target inside a host build-script/proc-macro scope on this thread, else the active config's target.
pub(crate) fn compile_target() -> &'static haste_codegen::Target {
    if in_host_target() {
        return haste_codegen::Target::host();
    }
    with_cfg(|c| c.target)
}

/// Runs `f` with `compile_target()` pinned to the host target on this thread (for build scripts and proc macros).
pub(crate) fn with_host_target<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            HOST_SCOPE.with(|depth| depth.set(depth.get() - 1));
        }
    }
    HOST_SCOPE.with(|depth| depth.set(depth.get() + 1));
    let _guard = Guard;
    f()
}

/// The compilation target with the code generator's effort set by the build profile.
pub(crate) fn host_target() -> haste_codegen::Target {
    let target = compile_target();
    target.effort(with_cfg(|c| c.profile).into())
}

/// The package, binary, and feature selection for the active run.
pub(crate) fn selection() -> Selection {
    with_cfg(|c| c.selection.clone())
}

/// Whether the active invocation is running tests (enabling `#[cfg(test)]` and member dev-dependencies).
pub(crate) fn test_mode() -> bool {
    with_cfg(|c| c.test_mode)
}

/// Whether `--daemon` opted into spawning a resident background `haste serve` daemon.
pub(crate) fn daemon() -> bool {
    with_cfg(|c| c.daemon)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn env_flag_treats_unset_empty_and_zero_as_false() {
        assert!(!parse_env_flag(None));
        assert!(!parse_env_flag(Some(OsStr::new(""))));
        assert!(!parse_env_flag(Some(OsStr::new("0"))));
        assert!(parse_env_flag(Some(OsStr::new("1"))));
        assert!(parse_env_flag(Some(OsStr::new("true"))));
        assert!(parse_env_flag_or(None, true));
        assert!(!parse_env_flag_or(None, false));
        assert!(!parse_env_flag_or(Some(OsStr::new("0")), true));
        assert!(parse_env_flag_or(Some(OsStr::new("1")), false));
    }

    #[test]
    fn host_target_scope_is_thread_local_and_propagates_to_sync_scope() {
        let entered = Arc::new(Barrier::new(2));
        let checked = Arc::new(Barrier::new(2));
        let (e2, c2) = (Arc::clone(&entered), Arc::clone(&checked));
        let other = std::thread::spawn(move || {
            e2.wait();
            let seen = in_host_target();
            c2.wait();
            seen
        });
        let (inside, worker_seen) = with_host_target(|| {
            entered.wait();
            let worker = crate::sync::scope(|s| s.spawn(in_host_target).join().expect("worker"));
            let here = in_host_target();
            checked.wait();
            (here, worker)
        });
        let outside = other.join().expect("peer thread completed");
        assert!(inside);
        assert!(worker_seen);
        assert!(!outside);
        assert!(!in_host_target());
    }

    #[test]
    fn concurrent_configs_stay_isolated_across_threads_and_workers() {
        let host = haste_codegen::Target::host();
        let cross = haste_codegen::TARGETS
            .iter()
            .find(|t| t.triple != host.triple)
            .expect("non-host target exists");
        let cfg_a = Arc::new(config::Config {
            jobs: 2,
            profile: outline::tree::Profile::Debug,
            target: host,
            selection: Selection {
                features: vec!["alpha".to_owned()],
                ..Selection::default()
            },
            ..config::Config::for_target(host)
        });
        let cfg_b = Arc::new(config::Config {
            jobs: 7,
            profile: outline::tree::Profile::Release,
            target: cross,
            selection: Selection {
                features: vec!["beta".to_owned()],
                ..Selection::default()
            },
            ..config::Config::for_target(cross)
        });
        let ready = Arc::new(Barrier::new(2));
        let done = Arc::new(Barrier::new(2));
        let (r2, d2, b2) = (Arc::clone(&ready), Arc::clone(&done), Arc::clone(&cfg_b));
        let peer = std::thread::spawn(move || {
            with_config(b2, || {
                r2.wait();
                let worker = crate::sync::scope(|s| {
                    s.spawn(|| {
                        (
                            jobs(),
                            profile(),
                            compile_target().triple,
                            selection().features,
                        )
                    })
                    .join()
                    .expect("worker b")
                });
                d2.wait();
                worker
            })
        });
        let seen_a = with_config(cfg_a, || {
            ready.wait();
            let worker = crate::sync::scope(|s| {
                s.spawn(|| {
                    (
                        jobs(),
                        profile(),
                        compile_target().triple,
                        selection().features,
                    )
                })
                .join()
                .expect("worker a")
            });
            done.wait();
            worker
        });
        let seen_b = peer.join().expect("peer thread");
        assert_eq!(
            seen_a,
            (
                2,
                outline::tree::Profile::Debug,
                host.triple,
                vec!["alpha".to_owned()]
            )
        );
        assert_eq!(
            seen_b,
            (
                7,
                outline::tree::Profile::Release,
                cross.triple,
                vec!["beta".to_owned()]
            )
        );
    }
}
