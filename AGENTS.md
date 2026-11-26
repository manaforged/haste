# haste — Agent Onboarding Guide

`haste` is a standalone Rust compiler, in-memory JIT runner, and zero-linker AOT executable builder (`crates/haste`, `crates/codegen`).

## Install

Follow the canonical install and prerequisites instructions in [README.md#install-and-quickstart](README.md#install-and-quickstart) (do **not** install `haste` from crates.io).

## Daily commands

- `haste build [PATH]` — type-check and borrow-check the workspace (one-shot; add `--daemon` on Unix for `8–9 ms` incremental edits).
- `haste run [PATH] [-- ARGS...]` — check and run in memory via JIT (Unix; add `--watch` to hot-reload changed functions in place).
- `haste test [PATH] [-- FILTER...]` — check and run `#[test]` and `#[tokio::test]` suites in memory (Unix). Always put `--` before test filters (`haste test -- my_test` or `haste test -- --list`); a bare positional argument is treated as `PATH`.
- `haste --release build -o <OUT> [--target <TRIPLE>] [PATH]` — emit a standalone ELF, Mach-O, or PE32+ executable without an external cross-linker.
- `haste serve status [PATH]` / `haste serve stop [PATH | --all]` — inspect or stop opt-in background daemons.

## Guardrails for agents using haste

- Keep `cargo check` or `rustc` as your final CI/merge gate: `haste` does not yet support nightly `#![feature(...)]`, full Polonius lifetimes, custom `harness = false` test runners, or cross-target static C archives (`lib*.a`) in `haste build -o`.
- Target selection supports `-p <PKG>`, `--bin <NAME>`, `--bins`, `--lib`, `--example <NAME>`, `--test <NAME>`, `--workspace`, and feature flags (not `--doc`).
- `haste test` captures stdout/stderr on a single isolated run per test (`-- --nocapture`, `-- --show-output`, `-- --list`, `-- --exact`, `-- --skip`).
- Daemons are keyed per workspace path and auto-exit after 900 s (15 min) idle (`HASTE_SERVE_IDLE_SECS`); stop worktree daemons (`haste serve stop --all`) before removing worktrees, or set `HASTE_NO_DAEMON=1` to bypass an active daemon.
- `haste clean` removes the global `$HASTE_HOME` cache (including the daemon registry), not a per-project `target/` directory.
- On Windows, use `haste build -o` (JIT `run`/`test`, `--daemon`, and `--watch` require Unix). On macOS, Hardened Runtime or App Sandbox restrictions can block JIT `mach_vm_remap` pages.
- Run `haste <command> --help` for exact flags.

## Repo layout & verification

- `crates/haste/` — CLI, Cargo graph/vendor/fetch, parser/outline, macro expander, type/borrow checker, IR lowering, JIT, AOT ELF/Mach-O/PE32+ emitters, resident daemon, and `Compiler` library API.
- `crates/codegen/` — SSA IR, optimization passes, register allocator, and x86-64 / ARM64 encoders.
- `scripts/check.sh` — differential JIT + AOT tests vs `rustc` (`tests/diff`, `tests/async`, `tests/proc-macro`).
- `scripts/probes.sh` — real-world crates.io compatibility probes (`serde_json`, `clap`, `regex`, `toml`, `winnow`, workspaces).
- `scripts/bench.sh` — build-speed and AOT compute runtime benchmarks vs `rustc`.
- `scripts/rustc-ui.sh` — upstream `rust-lang/rust` `tests/ui/` borrowck and run-pass harness.
