# haste — Agent Onboarding Guide

`haste` is a standalone Rust compiler, in-memory JIT runner, and zero-linker AOT executable builder (`crates/haste`, `crates/codegen`).

## Install

Do **not** install `haste` from crates.io (that is an unrelated benchmarking crate). Install from git:

```bash
rustup component add rust-src
cargo install --git https://github.com/manaforged/haste haste
```

Prerequisites: Rust 1.88+ with `rust-src`, plus a C compiler and `make` on macOS and Linux x86-64 GNU.

## Daily commands

- `haste build [PATH]` — type-check and borrow-check the workspace (one-shot; add `--daemon` on Unix for `8–9 ms` incremental edits).
- `haste run [PATH] [-- ARGS...]` — check and run in memory via JIT (Unix; add `--watch` to hot-reload changed functions in place).
- `haste test [PATH] [-- FILTER...]` — check and run `#[test]` and `#[tokio::test]` suites in memory (Unix).
- `haste --release build -o <OUT> [--target <TRIPLE>] [PATH]` — emit a standalone ELF, Mach-O, or PE32+ executable without an external cross-linker.
- `haste serve status [PATH]` / `haste serve stop [PATH | --all]` — inspect or stop opt-in background daemons.

## Guardrails for agents using haste

- Keep `cargo check` or `rustc` as your final CI/merge gate: `haste` does not yet support nightly `#![feature(...)]`, full Polonius lifetimes, custom `harness = false` test runners, or static C archives (`lib*.a`) in `haste build -o`.
- On Windows, use `haste build -o` (JIT `run`/`test`, `--daemon`, and `--watch` require Unix).
- Run `haste <command> --help` for exact flags.

## Repo layout & verification

- `crates/haste/` — CLI, Cargo graph/vendor/fetch, parser/outline, macro expander, type/borrow checker, IR lowering, JIT, AOT ELF/Mach-O/PE32+ emitters, resident daemon, and `Compiler` library API.
- `crates/codegen/` — SSA IR, optimization passes, register allocator, and x86-64 / ARM64 encoders.
- `scripts/check.sh` — differential JIT + AOT tests vs `rustc` (`tests/diff`, `tests/async`, `tests/proc-macro`).
- `scripts/probes.sh` — real-world crates.io compatibility probes (`serde_json`, `clap`, `regex`, `toml`, `winnow`, workspaces).
- `scripts/bench.sh` — build-speed and AOT compute runtime benchmarks vs `rustc`.
- `scripts/rustc-ui.sh` — upstream `rust-lang/rust` `tests/ui/` borrowck and run-pass harness.
