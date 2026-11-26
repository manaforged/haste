use std::sync::Mutex;

use super::aot;

/// How `Image::build` emits code: live JIT pages, AOT object relocations, or check-only verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageMode {
    Jit,
    Aot,
    Check,
}

pub(crate) enum ImageBackend {
    Check,
    EagerJit { fast: bool },
    LazyJit { fast: bool, optimize: bool },
    Aot(Mutex<aot::AotState>),
}

impl ImageBackend {
    pub(super) fn from_mode(mode: ImageMode, reached_len: usize) -> Self {
        match mode {
            ImageMode::Check => Self::Check,
            ImageMode::Jit => Self::EagerJit { fast: false },
            ImageMode::Aot => Self::Aot(aot::AotState::new(reached_len)),
        }
    }

    pub(crate) fn mode(&self) -> ImageMode {
        match self {
            Self::Check => ImageMode::Check,
            Self::EagerJit { .. } | Self::LazyJit { .. } => ImageMode::Jit,
            Self::Aot(_) => ImageMode::Aot,
        }
    }

    pub(crate) fn is_jit(&self) -> bool {
        matches!(self, Self::EagerJit { .. } | Self::LazyJit { .. })
    }

    pub(crate) fn is_lazy_jit(&self) -> bool {
        matches!(self, Self::LazyJit { .. })
    }

    pub(crate) fn is_fast_jit(&self) -> bool {
        matches!(
            self,
            Self::EagerJit { fast: true }
                | Self::LazyJit {
                    fast: true,
                    optimize: false
                }
        )
    }

    pub(crate) fn is_optimized_jit(&self) -> bool {
        matches!(self, Self::LazyJit { optimize: true, .. })
    }

    pub(crate) fn is_aot(&self) -> bool {
        matches!(self, Self::Aot(_))
    }

    pub(crate) fn is_check(&self) -> bool {
        matches!(self, Self::Check)
    }

    pub(crate) fn stencil_tag(&self) -> u64 {
        match self {
            Self::Aot(_) => 0xA070_A070_A070_A070,
            Self::LazyJit { fast, optimize } => {
                let base = if *optimize {
                    0x1A27_0000_1A27_0000
                } else if *fast || crate::profile() != crate::outline::tree::Profile::Release {
                    0x1A27_FA57_1A27_FA57
                } else {
                    0x1A27_0000_1A27_0000
                };
                if crate::in_host_target() {
                    base ^ 0x4057_0000_4057_0000
                } else {
                    base
                }
            }
            Self::Check | Self::EagerJit { .. } => 0,
        }
    }

    pub(crate) fn aot(&self) -> Option<&Mutex<aot::AotState>> {
        match self {
            Self::Aot(state) => Some(state),
            _ => None,
        }
    }
}
