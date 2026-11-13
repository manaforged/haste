use std::sync::OnceLock;

/// The environment variable that, set to `1`, makes the host report no optional features.
const BASELINE: &str = "HASTE_X64_BASELINE";

/// The instruction-set extensions past the x86-64 baseline that emitted code may use.
#[derive(Clone, Copy, Default)]
pub(super) struct Features {
    pub lzcnt: bool,
    pub bmi1: bool,
    pub popcnt: bool,
    pub sse41: bool,
    pub avx: bool,
}

impl Features {
    /// The host CPU's features, detected once; `HASTE_X64_BASELINE=1` reports none.
    pub fn host() -> Self {
        static HOST: OnceLock<Features> = OnceLock::new();
        *HOST.get_or_init(|| if std::env::var_os(BASELINE).is_some_and(|value| value == "1") { Self::default() } else { detect() })
    }
}

/// The optional features the CPU reports through CPUID.
#[cfg(target_arch = "x86_64")]
fn detect() -> Features {
    Features {
        lzcnt: std::arch::is_x86_feature_detected!("lzcnt"),
        bmi1: std::arch::is_x86_feature_detected!("bmi1"),
        popcnt: std::arch::is_x86_feature_detected!("popcnt"),
        sse41: std::arch::is_x86_feature_detected!("sse4.1"),
        avx: std::arch::is_x86_feature_detected!("avx"),
    }
}

/// No optional features: on another host the encoder targets the baseline.
#[cfg(not(target_arch = "x86_64"))]
fn detect() -> Features {
    Features::default()
}
