# haste

`haste` is an experimental Rust compiler, in-memory JIT runner, and zero-linker AOT builder for pure-Rust crates. Shared build artifacts live under `~/.cache/haste` (`HASTE_HOME`) instead of per-project `target/` directories, and `haste` targets fast local dev loops (`version = "0.0.0"`, no tagged release yet).

## Try live hot-reload

Start the bundled TCP server in [`examples/live-server`](examples/live-server/src/main.rs) on Unix, send a request, then edit `handle()` and save:

```bash
haste run --watch examples/live-server
curl http://127.0.0.1:8080
```

`haste` recompiles only the changed functions and patches them into the same PID and open listener socket. On macOS ARM64, a body edit measured `85.3 ms` save-to-first-changed-response; a compile error keeps serving the previous code and recovers in `135.3 ms` once fixed, while a structural edit keeps serving and prints a restart-required notice.

## Install and quickstart

You need Rust 1.88 or newer with `rust-src`, plus a C compiler and `make` on macOS and Linux x86-64 GNU (to build `tikv-jemallocator` from source).

> [!WARNING]
> Do not run `cargo install haste`. The `haste` name on crates.io is an unrelated benchmarking crate. Install from this repository with `--git`:

```bash
rustup component add rust-src
cargo install --git https://github.com/manaforged/haste --locked haste
haste --version # prints: haste 0.0.0 (<commit>)
```

Try it on a fresh crate (the first run per profile indexes `rust-src` once):

```bash
cargo new --bin haste-demo
haste run haste-demo -- --your-arg                 # Unix: run in memory via JIT
haste test haste-demo                              # Unix: run #[test] and #[tokio::test]
haste build --daemon haste-demo                    # Unix: warm resident daemon (900 s idle)
haste --release build -o ./haste-demo-bin haste-demo
haste --release build --target x86_64-unknown-linux-gnu -o ./haste-demo-linux haste-demo
```

If `Cargo.lock` or locked crates are missing, `haste` runs `cargo generate-lockfile` or `cargo fetch --locked` automatically unless `--offline` is set. It honors `cargo vendor`, verifies `.cargo-checksum.json` hashes, and applies `RUSTFLAGS` and `.cargo/config.toml`. Inspect or stop background daemons with `haste serve status` and `haste serve stop`.

## Benchmarks

Resident-daemon edits (`haste build --daemon`) keep workspace state warm in `haste serve`, while one-shot `haste build` caches only dependency summaries on disk (`7.1–8.4 ms` vs `81.8–130 ms` on bundled `tests/diff`). Full tables, methodology, upstream `rust-lang/rust` UI results (`292 / 344` borrowck, `3,820 / 4,168` pass), and crate compatibility live in [BENCHMARKS.md](BENCHMARKS.md).

| Benchmark | `haste` | `cargo` / `rustc` | Comparison |
| --- | ---: | ---: | --- |
| `tests/diff` edit, daemon (`macOS ARM64` / `Linux x86-64`) | 7.1 ms / 8.4 ms | 139 ms / 73.3 ms | 19.5x / 8.8x faster |
| `tests/diff` edit, one-shot (`macOS ARM64` / `Linux x86-64`) | 81.8 ms / 130 ms | 140 ms / 80.4 ms | 1.7x faster / 0.6x |
| 340-crate workspace leaf edit (daemon vs one-shot) | 22.3 ms / 2.35 s | 1.95 s | 87x faster / 2.35 s |
| 340-crate workspace public-type edit (daemon vs one-shot) | 3.00 s / 2.35 s | 2.21 s | 3.00 s / 2.35 s |
| `--release` AOT compute (5 kernels vs `rustc -O3`) | — | — | `0.68x–1.05x` (`<1.00x` = faster) |

## Embed as a library

Add `haste = { git = "https://github.com/manaforged/haste" }` in `Cargo.toml` (not from crates.io). `Compiler::session` keeps a warm in-process build for incremental checks, while `Compiler::check` and `Compiler::emit` load a fresh build per call without a resident daemon:

```rust
use haste::Compiler;

let compiler = Compiler::builder().jobs(8).build()?;
let mut session = compiler.session("./my-app")?;
session.check()?;
session.emit("./out-bin")?;
```

Both the CLI and `Compiler` execute crate `build.rs` scripts and procedural macros unsandboxed with the invoking user's privileges; untrusted repositories require isolated workers without credentials ([SECURITY.md](SECURITY.md)).

## Limits

- `haste` emits compiler errors only (no `unused_variables` or `dead_code` warnings) and does not yet support full Polonius lifetimes, some `for<'a>` edge cases, or nightly `#![feature(...)]`. Keep `cargo check` and `cargo clippy` in CI; external dependencies and `std` are checked lazily along reached paths.
- Pure-Rust AOT binaries (`build -o`) are zero-linker and standalone. Crates linking static C archives (`lib*.a`, such as `rusqlite` `bundled`) compile a host-local `$HASTE_HOME/native/` shared helper via `CC -shared` on native Unix and error on cross-target `build -o`.
- `haste test` runs `#[test]` and `#[tokio::test]` functions and skips doctests (`--doc`) and custom `harness = false` runners. Under JIT (`run` and `test`), `std::env::current_exe()` resolves to `haste`, so self-spawning tests fail.
- JIT `run`/`test`, `serve`, `--daemon`, and `--watch` require Unix (`prep` / `--prepared` are Linux x86-64 only). Windows PE32+ is experimental AOT-only (no `.pdata` or PDBs, imports `ucrtbase.dll` and `msvcrt.dll`), Linux→macOS cross-builds require `codesign --force --sign -` on macOS, and DWARF `.debug_line` is Linux ELF only.
- In `--release` AOT builds (`build -o`), bounds, divide-by-zero, and signed `MIN / -1` division-overflow checks lower to hardware traps unless `catch_unwind`, `std::panic::{set_hook, update_hook}`, or `std::thread::{spawn, spawn_unchecked, spawn_scoped, scope}` is reached (`--release test` and JIT always unwind).

## Commands

| Command | Does |
| --- | --- |
| `run [PATH] [--] [ARGS...]` | Check and run a binary or `--example` via JIT (Unix; add `--watch` for live hot-reload). |
| `test [PATH] [--] [FILTER...]` | Check and run `#[test]` and `#[tokio::test]` functions in memory (Unix). |
| `build [-o OUT] [PATH]` / `check [PATH]` | Check workspace (`check` aliases `build`), or emit a standalone executable with `-o`. |
| `serve [PATH]` / `status` / `stop` | Run, inspect, or stop resident `haste serve` daemons (Unix). |
| `prep [PATH]` | Pre-compile dependencies for `--prepared` (Linux x86-64 only). |
| `clean [PATH \| --all]` | Stop the workspace daemon and clear its caches (or stop all daemons and remove `$HASTE_HOME` with `--all`). |
| `debug QUERY PATH [CRATE]` | Print one compiler stage or dump machine code (`debug code PATH DIR`). |

Run `haste <command> --help` for per-command flags, and see [BENCHMARKS.md](BENCHMARKS.md#environment-variables) and [ARCHITECTURE.md](ARCHITECTURE.md#internal-debugging-and-tuning-variables) for environment variables.

## Contributing and license

Bug reports are welcome in [issues](https://github.com/manaforged/haste/issues); pull requests are not accepted yet. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [LICENSE-MIT](LICENSE-MIT).
