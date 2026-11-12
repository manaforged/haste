/// A CPU instruction set, each with one encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    X64,
    Arm64,
    Wasm32,
    Riscv64,
}

/// An operating system family, which decides the host layer and the C library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Os {
    Linux,
    Darwin,
    Windows,
    Wasi,
    None,
}

/// A platform C calling convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallAbi {
    SysV,
    Aapcs64,
    AppleArm64,
    Win64,
    Wasm,
    Lp64d,
}

/// An executable file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Elf,
    MachO,
    Pe,
    Wasm,
}

/// How much work the code generator spends on each function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Effort {
    /// The passes that pay for themselves when code runs briefly after an edit: the dev profile.
    #[default]
    Quick,
    /// Every pass: the release profile.
    Full,
}

/// One compilation target: everything that differs between triples, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub triple: &'static str,
    pub arch: Arch,
    pub os: Os,
    pub abi: CallAbi,
    pub format: Format,
    /// The LLVM data layout, as `rustc --print target-spec-json` reports it.
    pub data_layout: &'static str,
    pub ptr_bytes: u8,
    /// What the object format puts before a C symbol name.
    pub symbol_prefix: &'static str,
    /// `rustc --print cfg` without `debug_assertions`, then `[internal]` and the cfgs only `-Z unstable-options` shows.
    pub cfg: &'static str,
    /// How much work the code generator spends on each function.
    pub effort: Effort,
    pub direct_calls: bool,
}

macro_rules! target {
    ($triple:literal, $arch:ident, $os:ident, $abi:ident, $format:ident, $ptr:literal, $prefix:literal, $layout:literal) => {
        Target {
            triple: $triple,
            arch: Arch::$arch,
            os: Os::$os,
            abi: CallAbi::$abi,
            format: Format::$format,
            data_layout: $layout,
            ptr_bytes: $ptr,
            symbol_prefix: $prefix,
            cfg: include_str!(concat!("targets/", $triple, ".cfg")),
            effort: Effort::Quick,
            direct_calls: false,
        }
    };
}

/// Every target Haste knows, from `rustc 1.99.1 --print target-spec-json` and `--print cfg`.
pub const TARGETS: &[Target] = &[
    target!("x86_64-unknown-linux-gnu", X64, Linux, SysV, Elf, 8, "", "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"),
    target!("aarch64-unknown-linux-gnu", Arm64, Linux, Aapcs64, Elf, 8, "", "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"),
    target!("aarch64-apple-darwin", Arm64, Darwin, AppleArm64, MachO, 8, "_", "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"),
    target!("x86_64-apple-darwin", X64, Darwin, SysV, MachO, 8, "_", "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"),
    target!("x86_64-pc-windows-msvc", X64, Windows, Win64, Pe, 8, "", "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"),
    target!("aarch64-pc-windows-msvc", Arm64, Windows, Aapcs64, Pe, 8, "", "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"),
    target!("wasm32-unknown-unknown", Wasm32, None, Wasm, Wasm, 4, "", "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"),
    target!("wasm32-wasip1", Wasm32, Wasi, Wasm, Wasm, 4, "", "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"),
    target!("riscv64gc-unknown-linux-gnu", Riscv64, Linux, Lp64d, Elf, 8, "", "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"),
];

/// The catalog index of the machine this program runs on; another host does not compile.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
const HOST: usize = 0;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HOST: usize = 1;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const HOST: usize = 2;
#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
const HOST: usize = 3;
#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
const HOST: usize = 4;
#[cfg(all(target_arch = "aarch64", target_os = "windows"))]
const HOST: usize = 5;
#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
const HOST: usize = 8;

impl Target {
    /// The target named by `triple`.
    pub fn find(triple: &str) -> Option<&'static Target> {
        TARGETS.iter().find(|target| target.triple == triple)
    }

    /// The target of the machine this program runs on.
    pub fn host() -> &'static Target {
        &TARGETS[HOST]
    }

    /// This target with the code generator spending `effort` on each function.
    #[must_use]
    pub const fn effort(mut self, effort: Effort) -> Self {
        self.effort = effort;
        self
    }

    /// This target with direct PC-relative calls enabled when `direct_calls` is true.
    #[must_use]
    pub const fn direct_calls(mut self, direct_calls: bool) -> Self {
        self.direct_calls = direct_calls;
        self
    }
}
