# haste — Agent Onboarding Guide

`haste` is a standalone Rust compiler, in-memory JIT runner, and zero-linker AOT executable builder (`crates/haste`, `crates/codegen`).

## Install

Follow the canonical install and prerequisites instructions in [README.md#install-and-quickstart](README.md#install-and-quickstart) (do **not** install `haste` from crates.io).

## Daily commands

- `haste build [PATH]` / `haste check [PATH]` — one-shot check (`check` is an alias for `build`; re-checks local crate with dep summaries on disk; `2.35 s` median leaf or public-type edit on 340 crates vs `1.95 s` / `2.21 s` `cargo check`).
- `haste build --daemon [PATH]` — Unix resident daemon (`8.4–9.0 ms` small crates; `22.3 ms` median leaf edit / `87x` vs `cargo check`, `4.65 s` cold start, `3.00 s` public-type edit on 340 crates; `900 s` idle timeout).
- `haste::Compiler` (library API) — retains configuration, loads a fresh build per `check`/`test`/`emit` call (no resident daemon; do not assume daemon edit latency).
- `haste run [PATH] [-- ARGS...]` — check and run in memory via JIT (Unix; add `--watch` to hot-reload changed functions in place).
- `haste test [PATH] [-- FILTER...]` — check and run `#[test]` and `#[tokio::test]` suites in memory (Unix). Always put `--` before test filters (`haste test -- my_test` or `haste test -- --list`); a bare positional argument is treated as `PATH`.
- `haste --release build -o <OUT> [--target <TRIPLE>] [PATH]` — emit a standalone pure-Rust ELF, Mach-O, or PE32+ executable without an external cross-linker. Crates linking static C archives (`lib*.a`) link a host-local `$HASTE_HOME/native/` shared helper via `CC -shared` on native Unix and error on cross-target builds.
- `haste serve status [PATH]` / `haste serve stop [PATH | --all]` — inspect or stop opt-in background daemons.

## Guardrails for agents using haste

- Keep `cargo check`, `cargo clippy`, or `rustc` as your final CI/merge gate: `haste` emits compiler errors only (no warnings or lints such as `unused_variables` or `dead_code`), and does not yet support nightly `#![feature(...)]`, full Polonius lifetimes, custom `harness = false` test runners, doctests (`--doc`), or cross-target static C archives (`lib*.a`) in `haste build -o`.
- In `--release` AOT builds (`haste --release build -o`), bounds, divide-by-zero, and signed `MIN / -1` division-overflow checks lower to hardware traps unless `catch_unwind` or `std::thread::{spawn, spawn_unchecked, spawn_scoped, scope}` is reached (`haste --release test` and JIT always keep unwinding enabled).
- Under JIT (`haste run` and `haste test`), `std::env::current_exe()` resolves to the `haste` binary rather than a standalone test binary, so self-spawning tests (`Command::new(current_exe())`) fail alongside `harness = false` runners.
- Target and Cargo selection supports `-p <PKG>`, `--workspace`, `--bin <NAME>`, `--bins`, `--lib`, `--example <NAME>`, `--test <NAME>`, `--tests`, `--all-targets`, `--no-run` (for `test`), `--manifest-path <PATH>`, `--profile <dev|debug|release>`, `--locked`, `--offline`, `--frozen`, `-j <N>`, `-q`, `--color=<auto|always|never>`, `--message-format <human|short|json|json-render-diagnostics>`, feature flags, `RUSTFLAGS` / `CARGO_ENCODED_RUSTFLAGS`, and `.cargo/config.toml` `[build].rustflags` / `[env]` (not `--doc`).
- `haste test` captures stdout/stderr on a single isolated run per test (`-- --nocapture`, `-- --show-output`, `-- --list`, `-- --exact`, `-- --skip`).
- Daemons are keyed per workspace path and auto-exit after 900 s (15 min) idle (`HASTE_SERVE_IDLE_SECS`; foreground `haste serve` runs until stopped unless `HASTE_SERVE_IDLE_SECS` is set); stop worktree daemons (`haste serve stop --all`) before removing worktrees, or set `HASTE_NO_DAEMON=1` to bypass an active daemon.
- `haste clean` stops active daemons and removes the global `$HASTE_HOME` cache (including the daemon registry), not a per-project `target/` directory.
- Format and platform limits: Linux ELF binaries are non-PIE and target glibc with DWARF `.debug_line` on Linux only; Mach-O emits `LC_UUID`/`LC_FUNCTION_STARTS`/symtab without `.debug_line` (Linux→macOS ARM64 cross-builds are unsigned until `codesign`ed on macOS); Windows PE32+ is experimental, non-PIE, omits `.pdata`/PDBs, and imports CRT shims from `ucrtbase.dll` and `msvcrt.dll`; JIT `run`/`test`, `--daemon`, and `--watch` require Unix; on macOS, Hardened Runtime or App Sandbox restrictions can block JIT `mach_vm_remap` pages.
- Run `haste <command> --help` for exact flags.

## Repo layout & verification

- `crates/haste/` — CLI, Cargo graph/vendor/fetch, parser/outline, macro expander, type/borrow checker, IR lowering, JIT, AOT ELF/Mach-O/PE32+ emitters, resident daemon, and `Compiler` library API.
- `crates/codegen/` — SSA IR, optimization passes, register allocator, and x86-64 / ARM64 encoders.
- `scripts/check.sh` — differential JIT + AOT tests vs `rustc` (`tests/diff`, `tests/async`, `tests/proc-macro`).
- `scripts/probes.sh` — real-world crates.io compatibility probes (`serde_json`, `clap`, `regex`, `toml`, `winnow`, workspaces).
- `scripts/bench.sh` — build-speed and AOT compute runtime benchmarks vs `rustc`.
- `scripts/rustc-ui.sh` — upstream `rust-lang/rust` `tests/ui/` borrowck and run-pass harness.
