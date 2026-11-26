use std::sync::OnceLock;

/// The environment variable that, set to `1`, makes the host report no optional features.
const BASELINE: &str = "HASTE_X64_BASELINE";

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
    /// The host CPU's features, detected once; `HASTE_X64_BASELINE=1` reports none.
    pub fn host() -> Self {
        static HOST: OnceLock<Features> = OnceLock::new();
        *HOST.get_or_init(|| {
            if crate::env_flag(BASELINE) {
                Self::default()
            } else {
                detect()
            }
        })
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
}
