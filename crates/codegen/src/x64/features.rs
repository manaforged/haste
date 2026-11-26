use std::cell::Cell;
use std::sync::OnceLock;

/// The environment variable that, set to `1`, makes the host report no optional features.
const BASELINE: &str = "HASTE_X64_BASELINE";

thread_local! {
    static BASELINE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Replaces this thread's `x64_baseline` override, returning the previous value.
pub(crate) fn replace_baseline_override(value: Option<bool>) -> Option<bool> {
    BASELINE_OVERRIDE.with(|slot| slot.replace(value))
}

/// The instruction-set extensions past the x86-64 baseline that emitted code may use.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Features {
    pub lzcnt: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub popcnt: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
}

impl Features {
    /// The host CPU's features, detected once; `HASTE_X64_BASELINE=1` or an active baseline scope reports none.
    pub fn host() -> Self {
        static DETECTED: OnceLock<Features> = OnceLock::new();
        let baseline = BASELINE_OVERRIDE.with(Cell::get).unwrap_or_else(|| {
            static ENV_BASELINE: OnceLock<bool> = OnceLock::new();
            *ENV_BASELINE.get_or_init(|| crate::env_flag(BASELINE))
        });
        if baseline {
            Self::default()
        } else {
            *DETECTED.get_or_init(detect)
        }
    }

    /// Optional x86-64 features enabled for `target`: host CPU features when compiling natively, else baseline SSE2.
    pub fn for_target(target: &crate::Target) -> Self {
        if target.triple == crate::Target::host().triple {
            Self::host()
        } else {
            Self::default()
        }
    }
}

/// The optional features the CPU reports through CPUID.
#[cfg(target_arch = "x86_64")]
fn detect() -> Features {
    Features {
        lzcnt: std::arch::is_x86_feature_detected!("lzcnt"),
        bmi1: std::arch::is_x86_feature_detected!("bmi1"),
        bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        popcnt: std::arch::is_x86_feature_detected!("popcnt"),
        ssse3: std::arch::is_x86_feature_detected!("ssse3"),
        sse41: std::arch::is_x86_feature_detected!("sse4.1"),
        avx: std::arch::is_x86_feature_detected!("avx"),
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
    }
}

/// No optional features: on another host the encoder targets the baseline.
#[cfg(not(target_arch = "x86_64"))]
fn detect() -> Features {
    Features::default()
}

#[cfg(test)]
mod tests {
    use super::Features;
    use crate::{Arch, TARGETS, Target};

    #[test]
    fn cross_x64_target_uses_baseline_features() {
        let host = Target::host();
        let cross = TARGETS
            .iter()
            .find(|t| t.arch == Arch::X64 && t.triple != host.triple)
            .expect("catalog has a non-host x86_64 target");
        assert_eq!(Features::for_target(cross), Features::default());
    }

    #[test]
    fn unimplemented_targets_are_absent() {
        assert!(Target::find("riscv64gc-unknown-linux-gnu").is_none());
        assert!(Target::find("wasm32-unknown-unknown").is_none());
        assert!(Target::find("wasm32-wasip1").is_none());
    }

    #[test]
    fn baseline_scope_forces_sse2_and_restores_previous_state() {
        let before = Features::host();
        {
            let _outer = crate::ConfigScope::enter(true, false);
            assert_eq!(Features::host(), Features::default());
            assert_eq!(Features::for_target(Target::host()), Features::default());
            assert!(!crate::verify::verify_passes_enabled());
            {
                let _inner = crate::ConfigScope::enter(false, true);
                assert_eq!(Features::host(), super::detect());
                assert!(crate::verify::verify_passes_enabled());
            }
            assert_eq!(Features::host(), Features::default());
            assert!(!crate::verify::verify_passes_enabled());
        }
        assert_eq!(Features::host(), before);
    }
}
