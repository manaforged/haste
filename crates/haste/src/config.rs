//! Immutable build configuration owned per invocation and propagated across worker threads.

use std::ffi::OsString;
use std::hash::{Hash, Hasher};

use haste_codegen::Target;

use crate::Selection;
use crate::outline::tree::Profile;

/// Environment variable setting function alignment in bytes for standalone AOT executables.
pub(crate) const FN_ALIGN_ENV: &str = "HASTE_FN_ALIGN";

/// Resolved compilation configuration from CLI flags or library builder and `HASTE_*` environment variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    pub jobs: usize,
    pub profile: Profile,
    pub target: &'static Target,
    pub selection: Selection,
    pub test_mode: bool,
    pub no_run: bool,
    pub daemon: bool,
    pub prepared: bool,
    pub debug_info: bool,
    pub symbols: bool,
    pub x64_baseline: bool,
    pub fn_align: Option<u32>,
    pub use_substrate: bool,
    pub verify_passes: bool,
    pub verify_stencil: bool,
    pub library: Option<OsString>,
}

impl Default for Config {
    fn default() -> Self {
        Self::from_env()
    }
}

impl Config {
    /// Builds a default configuration populated from the current process environment.
    pub(crate) fn from_env() -> Self {
        let jobs = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let mut cfg = Self {
            jobs,
            profile: Profile::default(),
            target: Target::host(),
            selection: Selection::default(),
            test_mode: false,
            no_run: false,
            daemon: false,
            prepared: false,
            debug_info: false,
            symbols: false,
            x64_baseline: crate::env_flag("HASTE_X64_BASELINE"),
            fn_align: parse_fn_align(std::env::var(FN_ALIGN_ENV).ok().as_deref()),
            use_substrate: !crate::env_flag("HASTE_NO_SUBSTRATE"),
            verify_passes: crate::env_flag("HASTE_VERIFY_PASSES"),
            verify_stencil: crate::env_flag("HASTE_VERIFY_STENCIL"),
            library: std::env::var_os(crate::LIBRARY_ENV),
        };
        cfg.refresh_debug_flags();
        cfg
    }

    /// Recomputes `debug_info` and `symbols` from `profile` and environment toggles.
    pub(crate) fn refresh_debug_flags(&mut self) {
        self.debug_info = crate::env_flag_or("HASTE_DEBUG_INFO", self.profile != Profile::Release);
        self.symbols = crate::env_flag_or("HASTE_SYMBOLS", self.debug_info);
    }

    /// Returns the active process configuration snapshot for the current thread.
    pub(crate) fn current() -> Self {
        crate::with_cfg(Clone::clone)
    }

    /// Hashes every build-affecting field into `state` (excluding concurrency `jobs` and `daemon`).
    pub(crate) fn hash_cache_key(&self, state: &mut impl Hasher) {
        (self.profile == Profile::Release).hash(state);
        self.target.triple.hash(state);
        self.selection.hash(state);
        self.test_mode.hash(state);
        self.no_run.hash(state);
        self.prepared.hash(state);
        self.debug_info.hash(state);
        self.symbols.hash(state);
        self.x64_baseline.hash(state);
        self.fn_align.hash(state);
        self.use_substrate.hash(state);
        self.verify_passes.hash(state);
        self.verify_stencil.hash(state);
        self.library.hash(state);
    }

    /// Returns a deterministic test configuration targeting `target`.
    #[cfg(test)]
    pub(crate) fn for_target(target: &'static Target) -> Self {
        Self {
            jobs: 1,
            profile: Profile::Debug,
            target,
            selection: Selection::default(),
            test_mode: false,
            no_run: false,
            daemon: false,
            prepared: false,
            debug_info: true,
            symbols: true,
            x64_baseline: false,
            fn_align: None,
            use_substrate: true,
            verify_passes: false,
            verify_stencil: false,
            library: None,
        }
    }
}

/// Parses a power-of-two function alignment (`>= 16`) from `HASTE_FN_ALIGN`.
pub(crate) fn parse_fn_align(raw: Option<&str>) -> Option<u32> {
    raw.and_then(|v| v.parse().ok())
        .filter(|a: &u32| a.is_power_of_two() && *a >= 16)
}

/// Resolves a target triple against the supported catalog, formatting supported triples on error.
pub(crate) fn find_target(triple: &str) -> anyhow::Result<&'static Target> {
    Target::find(triple).ok_or_else(|| {
        let known: Vec<&str> = haste_codegen::TARGETS.iter().map(|t| t.triple).collect();
        anyhow::anyhow!(
            "unknown target `{triple}`; supported targets: {}",
            known.join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use rustc_hash::FxHasher;

    use super::*;

    fn key_of(cfg: &Config) -> u64 {
        let mut h = FxHasher::default();
        cfg.hash_cache_key(&mut h);
        h.finish()
    }

    #[test]
    fn cache_key_distinguishes_every_behavioral_toggle() {
        let base = Config::for_target(Target::host());
        let base_key = key_of(&base);
        let variants: Vec<Config> = vec![
            Config {
                profile: Profile::Release,
                ..base.clone()
            },
            Config {
                debug_info: !base.debug_info,
                ..base.clone()
            },
            Config {
                symbols: !base.symbols,
                ..base.clone()
            },
            Config {
                x64_baseline: !base.x64_baseline,
                ..base.clone()
            },
            Config {
                fn_align: Some(32),
                ..base.clone()
            },
            Config {
                use_substrate: !base.use_substrate,
                ..base.clone()
            },
            Config {
                verify_passes: !base.verify_passes,
                ..base.clone()
            },
            Config {
                verify_stencil: !base.verify_stencil,
                ..base.clone()
            },
            Config {
                selection: Selection {
                    all_features: true,
                    ..Selection::default()
                },
                ..base.clone()
            },
        ];
        for variant in variants {
            assert_ne!(key_of(&variant), base_key);
        }
        let different_jobs = Config { jobs: 99, ..base };
        assert_eq!(key_of(&different_jobs), base_key);
    }
}
