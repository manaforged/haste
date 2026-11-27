# haste

haste is an experimental Rust compiler, in-memory JIT runner, and pure-Rust zero-linker AOT builder with no LLVM in the compiler binary and no per-project `target/` directory (cached artifacts live under `~/.cache/haste`).

- **Live hot-reload in a running process (Unix):** edit a function in [`examples/live-server`](examples/live-server/src/main.rs), save, and hit the same TCP server again — same PID, updated response in under 100 ms (`haste run --watch examples/live-server`).
- **In-memory execution and tests (Unix):** run binaries (`haste run`) and `#[test]` / `#[tokio::test]` suites (`haste test`) without invoking an external linker.
- **Sub-10 ms resident-daemon edits on bundled `tests/diff`:** `7.1–8.4 ms` on single-line edits with `--daemon` (`8.8x–19.5x` vs `cargo check`); unchanged no-ops exit in `0.5–2.4 ms`.
- **Pure-Rust zero-linker AOT & cross-compilation:** emit native ELF, Mach-O, and PE32+ executables directly for pure-Rust crates (`haste --release build --target ... -o`), with `--release` compute kernels at `0.68x–1.05x` of `rustc -O3` across 5 benchmarks. See [Limits](#limits) for C archive, debug-info, and trap caveats.
- **Built for fast local dev loops; keep `rustc` in CI:** experimental (`version = "0.0.0"`, no tagged release yet). See [Crate compatibility](#crate-compatibility), [Limits](#limits), and [ARCHITECTURE.md](ARCHITECTURE.md).

## Install and quickstart

Prerequisites: Rust 1.88 or newer with `rust-src`, plus a C compiler and `make` on macOS and Linux x86-64 GNU (to build `tikv-jemallocator` from source).

> [!WARNING]
> Do not run `cargo install haste`. The `haste` name on crates.io is an unrelated benchmarking crate. Install from this repository with `--git`:

```bash
rustup component add rust-src
cargo install --git https://github.com/manaforged/haste --locked haste
haste --version # prints: haste 0.0.0 (<commit>) when built from a git checkout
```

Teams can pin a specific revision (`--rev <SHA>` or `--tag <TAG>`) and roll back by reinstalling a known-good revision.

Try it on a fresh crate from any directory (the first run per profile indexes `rust-src` once and logs progress to stderr):

```bash
cargo new --bin haste-demo

# Unix: run in memory via JIT (prints: Hello, world!)
haste run haste-demo -- --your-arg

# Unix: run #[test] and #[tokio::test] suites in memory
haste test haste-demo

# Windows (or any platform): emit and run a standalone native AOT executable
haste --release build -o ./haste-demo-bin haste-demo
./haste-demo-bin
```

If a workspace has no `Cargo.lock`, `haste` runs `cargo generate-lockfile` automatically. If locked registry or git crates are missing locally, `haste` runs `cargo fetch --locked` automatically unless `--offline`, `CARGO_NET_OFFLINE=true`, or `[net] offline = true` is set. Dependency lookup matches `Cargo.lock` `source` URLs against `.cargo/config.toml` `[source.*]` / `replace-with` entries, honors `cargo vendor` offline, verifies `.cargo-checksum.json` `"package"` hashes fail-closed against `Cargo.lock` (without re-hashing individual unpacked files on disk), and applies `RUSTFLAGS` / `CARGO_ENCODED_RUSTFLAGS` plus `.cargo/config.toml` `[build].rustflags` and `[env]`.

### Three incremental workflows: one-shot, daemon, and library

`haste` has three distinct execution models with different incremental latency:

1. **One-shot CLI (`haste build`):** runs a fresh process per invocation, re-checks the local crate, and caches only dependency summaries on disk. On the bundled `tests/diff` fixture (`144` test cases), single-line edits take `81.8–130 ms` (`0.6x–1.7x` vs `cargo check`) and unchanged no-ops take `0.5–2.4 ms`. On an unbundled 340-crate workspace (macOS ARM64, 6 CPUs), leaf edits take `2.35 s` median (`2.33–2.49 s`) and public-type edits take `2.35 s` (`2.33–2.40 s`), compared to `1.95 s` (leaf) and `2.21 s` (public-type) for `cargo check`.
2. **Resident daemon (`haste build --daemon`, Unix):** keeps workspace and dependency state warm in a background `haste serve` process (`900 s` idle timeout, configurable via `HASTE_SERVE_IDLE_SECS`). On the bundled `tests/diff` fixture, single-line edits take `7.1–8.4 ms` (`8.8x–19.5x` vs `cargo check`). On the unbundled 340-crate workspace, daemon cold start takes `4.65 s`, leaf edits drop to `22.3 ms` median (`22.1–37.3 ms`, `87x` faster than `cargo check` at `1.95 s`), while cross-crate public-type edits take `3.00 s` (`2.94–3.09 s`, slower than `cargo check` at `2.21 s`). Plain `haste` commands never spawn a daemon, though they reuse an already-running daemon for the same workspace unless `HASTE_NO_DAEMON=1` is set.
3. **Embeddable library (`haste::Compiler`):** retains configuration across calls, but loads a fresh build on every `check`, `test`, or `emit` call. It does not run a resident daemon, so embedders should not assume `--daemon` edit latency.

Inspect or stop background daemons with `serve status` and `serve stop` (`PATH` is optional):

```bash
haste build --daemon haste-demo
haste serve status haste-demo
haste serve stop haste-demo
```

### Live hot-reload

Try it right away on the bundled [`examples/live-server`](examples/live-server/src/main.rs) TCP server (Unix):

```bash
# Terminal 1: start the server with live function patching
haste run --watch examples/live-server

# Terminal 2: send a request, edit `handle()` in examples/live-server/src/main.rs, save, and repeat
curl http://127.0.0.1:8080
```

When you save a file, `haste` recompiles only the affected functions and patches them in memory on their next call while keeping the same PID and open listener socket. On macOS ARM64, save-to-first-changed-response measured `85.3 ms` on a body edit; a compile error keeps serving the previous code and recovers in `135.3 ms` once fixed, while a structural edit keeps serving and prints a restart-required notice.

### Zero-linker cross-compilation

Emit a Linux x86-64 ELF binary from macOS (or macOS/Windows executables from Linux) without installing a cross-linker:

```bash
haste --release build --target x86_64-unknown-linux-gnu -o ./haste-demo-linux haste-demo
```

Pure-Rust executables are zero-linker and standalone. Linux ELF binaries are non-PIE and target glibc, Windows PE32+ binaries are non-PIE and import CRT shims from `ucrtbase.dll` and `msvcrt.dll`, and Linux→macOS cross-builds (both ARM64 and x86-64) are unsigned until `codesign`ed on macOS (`codesign --force --sign -`). Crates linking native static C archives (`lib*.a`) work on native Unix via a host-local shared helper under `$HASTE_HOME/native/`, not across `--target` triples.

## Benchmarks

Run `bash scripts/bench.sh` to reproduce build-speed and AOT runtime numbers on your machine (pass `--long` for 4x compute iterations). By default, `scripts/bench.sh` runs with `HASTE_BENCH_COLD=1`, which isolates `HASTE_HOME` and reports two distinct cold rows on `tests/diff` against a fresh `cargo` target directory: **Empty `HASTE_HOME`** (first command after install, including one-time stdlib indexing) and **First project build, stdlib indexed** (first build of a new project after stdlib indexing on `examples/hello`). Set `HASTE_BENCH_COLD=0` to reuse your existing `$HASTE_HOME` and print `Warm cache` rows instead.

### Build speed (`tests/diff`, bundled 144-case fixture, `HASTE_BENCH_COLD=1`)

Resident-daemon edits (`--daemon`) compare a warm background `haste serve` process against one-shot `cargo check` (using `rustup which cargo` so PATH shims do not skew the baseline). `Empty HASTE_HOME` includes one-time stdlib indexing (`915–966 ms` for `haste build` vs `200–249 ms` for `cargo check`); once stdlib is indexed, the first build of a new project drops to `82.0–136 ms` (`1.5x–3.0x` faster than `cargo check`).

| Benchmark | Platform | haste | cargo | Speedup |
| --- | --- | ---: | ---: | ---: |
| Single-line edit (`haste build --daemon` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 7.1 ms | 139 ms | 19.5x |
| Single-line edit (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 81.8 ms | 140 ms | 1.7x |
| Single-line edit (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 198 ms | 269 ms | 1.4x |
| Unchanged no-op (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 2.4 ms | 81.3 ms | 34.3x |
| Empty `HASTE_HOME` (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 915 ms | 249 ms | 0.3x |
| Empty `HASTE_HOME` (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 768 ms | 550 ms | 0.7x |
| First project build, stdlib indexed (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 82.0 ms | 249 ms | 3.0x |
| First project build, stdlib indexed (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 258 ms | 550 ms | 2.1x |
| Single-line edit (`haste build --daemon` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 8.4 ms | 73.3 ms | 8.8x |
| Single-line edit (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 130 ms | 80.4 ms | 0.6x |
| Single-line edit (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 307 ms | 145 ms | 0.5x |
| Unchanged no-op (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 0.5 ms | 27.4 ms | 52.1x |
| Empty `HASTE_HOME` (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 966 ms | 200 ms | 0.2x |
| Empty `HASTE_HOME` (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 1.4 s | 534 ms | 0.4x |
| First project build, stdlib indexed (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 136 ms | 200 ms | 1.5x |
| First project build, stdlib indexed (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 575 ms | 534 ms | 0.9x |

On a larger private multi-crate workspace (not bundled; `340` crates, `45` build scripts, `17,521` functions, macOS ARM64, 6 CPUs):

- **Cold check (empty `HASTE_HOME`):** `18.7 s` (`159 MiB` shared cache, `+6–18 MiB` per additional workspace; checks reached functions for compiler errors only, with no warnings/lints) vs `34.6 s` for cold `cargo check --workspace -j 6` (which checks all items and warnings).
- **Daemon cold start (`haste build --daemon`):** `4.65 s` (`900 s` idle timeout).
- **Leaf edit:** `22.3 ms` median (`22.1–37.3 ms`, `87x` faster than `cargo check` at `1.95 s`) with `haste build --daemon`, vs `2.35 s` median (`2.33–2.49 s`) for one-shot `haste build`.
- **Cross-crate public-type edit:** `2.35 s` (`2.33–2.40 s`) for one-shot `haste build` and `3.00 s` (`2.94–3.09 s`) with `haste build --daemon`, vs `2.21 s` for `cargo check`.

### AOT codegen runtime (`haste --release` vs `rustc`, baseline-subtracted median of 5, warmed)

The compute tables below show baseline-subtracted `net` times (`scripts/bench.sh` prints both `net` and `raw` times as `<net> (raw <raw>)`), subtracting each compiler's warmed empty-main startup baseline (median of 7) and verifying bit-identical outputs across `haste --release`, stock `rustc -C opt-level=3`, and matched-flag `rustc -C opt-level=3 -C target-cpu=native -C panic=abort`. In `--release` AOT builds (`haste --release build -o`), bounds, divide-by-zero, and signed `MIN / -1` division-overflow checks lower to hardware traps unless `catch_unwind` or `std::thread::{spawn, spawn_unchecked, spawn_scoped, scope}` is reached (while `haste --release test` and JIT always keep unwinding enabled, and integer `+`/`-`/`*` wrap in `--release`).

#### Linux x86-64 (`rustc 1.99.1`, empty-main baseline: 0.2 ms across all three)

| Workload | `haste --release` (net) | `rustc -O3` (net) | vs `-O3` | `rustc -O3 native+abort` (net) | vs matched | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (2M blocks) | 229 ms | 289 ms | **0.79x** | 253 ms | **0.90x** | exact match (`2980548810864065682`) |
| `sieve` (20x 4M primes) | 44.7 ms | 65.3 ms | **0.68x** | 51.6 ms | **0.87x** | exact match (`10840522878760`) |
| `sort` (8x 500k quicksort + bsearch) | 186 ms | 188 ms | **0.99x** | 189 ms | **0.99x** | exact match (`1823227901928002215`) |
| `scan` (1200x JSON byte scanner) | 66.9 ms | 76.8 ms | **0.87x** | 77.3 ms | **0.87x** | exact match (`2400205310288`) |
| `fp_iter` (4096x f64 generic iterator) | 164 ms | 161 ms | 1.02x | 164 ms | **1.00x** | exact match (`4733336957083238967`) |

#### macOS ARM64 (`rustc 1.98.1`, empty-main baseline: `haste` 1.6 ms, `-O3` 3.8 ms, matched 3.2 ms)

| Workload | `haste --release` (net) | `rustc -O3` (net) | vs `-O3` | `rustc -O3 native+abort` (net) | vs matched | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (2M blocks) | 282 ms | 290 ms | **0.97x** | 291 ms | **0.97x** | exact match (`2980548810864065682`) |
| `sieve` (20x 4M primes) | 61.8 ms | 81.1 ms | **0.76x** | 81.6 ms | **0.76x** | exact match (`10840522878760`) |
| `sort` (8x 500k quicksort + bsearch) | 180 ms | 182 ms | **0.99x** | 181 ms | **1.00x** | exact match (`1823227901928002215`) |
| `scan` (1200x JSON byte scanner) | 106 ms | 101 ms | 1.05x | 104 ms | 1.02x | exact match (`2400205310288`) |
| `fp_iter` (4096x f64 generic iterator) | 135 ms | 162 ms | **0.83x** | 163 ms | **0.83x** | exact match (`4733336957083238967`) |

### Upstream `rust-lang/rust` UI suite (`scripts/rustc-ui.sh`, macOS ARM64)

- `scripts/rustc-ui.sh --borrowck`: **292 / 344 (84.9%)** of the filtered `Ran` subset emit `error[borrow]:` (**276 / 344**, `80.2%`, match a `src/{main,lib}.rs` span line against an expected `//~ ERROR` annotation, checking first match).
- `scripts/rustc-ui.sh --all` (pass suite, JIT mode): **3,820 / 4,168 (91.7%)** of the filtered `Ran` subset pass after skipping files with `#![feature(`, `aux-build`, `compile-flags`, `revisions`, `only-`/`ignore-`, `edition:2015`, external `extern crate`, `//@ check-pass`/`//@ build-pass` files without `fn main(` (unless marked `//@ run-pass`), or failures under `rustc --edition=2021`. This count includes `//@ check-pass` and `//@ build-pass` build checks containing `fn main(` alongside `//@ run-pass` tests, and was run in JIT mode (`--aot` is not enabled by default).

## Crate compatibility

`scripts/probes.sh` cross-checks the probes under `tests/probes/` against `rustc` in both JIT and `--release` AOT modes (`tests/probes/so` is JIT-only). The bundled `tests/probes/` suite covers `serde`/`serde_json`, `clap`, `regex`, `toml`, `winnow`, multi-crate workspaces, and local proc-macros (`tests/probes/pc`), while `rusqlite`, `#[tokio::test]`, `prost`, `uuid`, and `chrono` were verified in external test crates rather than bundled in `tests/probes/`:

| Crate or workload | Status | Notes |
| --- | --- | --- |
| Pure-Rust multi-crate workspaces (`[workspace.dependencies]`, path crates) | Supported (JIT + AOT) | `tests/probes/ws` |
| `serde` (`derive`) + `serde_json` (`raw_value`), `toml` (`0.8`) | Supported (JIT + AOT) | `tests/probes/sj`, `tests/probes/fp`, `tests/probes/ws` |
| `clap` (`4.x`, `derive`), `regex` (`1.x`), `winnow` (`0.7`) | Supported (JIT + AOT) | `tests/probes/cd`, `tests/probes/rx`, `tests/probes/wn` |
| `#[tokio::test]` (`current_thread`, `multi_thread`, `LocalSet`, `start_paused`) | Supported (JIT) | Verified in external 9-test Tokio suite (not bundled in `tests/probes/`); `harness = false` unsupported |
| Procedural macros (`derive`, attribute, function-like) | Supported (JIT + AOT) | `tests/proc-macro`, `tests/probes/pc` |
| `rusqlite` (`0.32`, `bundled` C `libsqlite3.a`) | Supported (JIT + host-local Unix AOT) | Verified in external test crate (not bundled in `tests/probes/`); links cached `$HASTE_HOME/native/*.so` shared helper via `CC -shared` |
| `prost` (`0.13`), `uuid` (`1.x`), `chrono` (`0.4`) | Supported (JIT + AOT) | Verified in external test crates (not bundled in `tests/probes/`) |
| Crates bundling static C archives (`lib*.a`, e.g. `rusqlite` `bundled`, `tikv-jemalloc-sys`) | JIT + host-local Unix AOT (`build -o`); cross-target AOT errors | Host-local `$HASTE_HOME/native/` shared helper (not standalone); see [Limits](#limits) |
| `reqwest` / `openssl-sys` | Untested | Not covered in `tests/probes` |
| Nightly `#![feature(...)]` crates | Unsupported | Stable syntax and stdlib only |

## Embedding haste as a library

`haste` exposes an embeddable API (`Compiler`, `CompilerBuilder`, `Target`, `Error`, `ErrorKind`) with explicit per-build configuration and no process-global state. `Compiler` retains configuration across calls, but loads a fresh build per `check`, `test`, or `emit` call. It does not run a resident daemon, so embedders should not assume `--daemon` edit latency.

Because `Compiler::check` (along with `test`, `emit`, and the CLI) executes crate `build.rs` scripts and procedural macros unsandboxed with the invoking user's privileges, untrusted or customer-submitted repositories require isolated, resource-limited workers without service credentials.

Add it as a git dependency in `Cargo.toml` (`haste = { git = "https://github.com/manaforged/haste" }`, not from crates.io):

```rust
use haste::Compiler;

let compiler = Compiler::builder()
    .target("x86_64-unknown-linux-gnu")
    .release(true)
    .jobs(8)
    .build()?;

compiler.check("./my-app")?;
compiler.emit("./my-app", "./out-bin")?;
```

## Limits

- **Errors only (no warnings or lints):** `haste` emits compiler errors only (`error[borrow]`, `error[type]`, and related error categories) and does not emit warnings or lints such as `unused_variables` or `dead_code`. Keep `cargo check` and `cargo clippy` for lints and final release gates.
- **Incomplete borrow-checker and nightly coverage:** Full Polonius borrow checking, some upstream `rust-lang/rust` `tests/ui/borrowck` and higher-ranked (`for<'a>`) lifetime edge cases, and nightly `#![feature(...)]` attributes are not yet supported. Keep `cargo check` or `rustc` in CI as your merge and release gate.
- **External dependencies stay lazy:** `haste` type-checks and borrow-checks every concrete and generic function in the selected member crate before running or emitting code, while external crates and `std` are checked only along reached paths.
- **Static C archives in AOT (`build -o`):** Pure-Rust executables are zero-linker and standalone. Crates that link native static C archives (`lib*.a`, such as `rusqlite` `bundled`) compile a cached shared helper via `CC -shared` under `$HASTE_HOME/native/` on native Unix (`haste build .` checks `haste`'s own 42-package workspace including `tikv-jemalloc-sys`, 0 failed functions). Those binaries depend on that host-local shared helper rather than being portable standalone executables, and cross-target `haste build -o --target <TRIPLE>` rejects static C archives rather than emitting an incomplete binary.
- **Test harnesses, doctests, and proc-macro edge cases:** `haste test` runs `#[test]` and `#[tokio::test]` functions (`fn()` and `fn() -> Result<(), E>`), skipping doctests (`--doc` is unsupported) and custom `harness = false` runners; procedural macros relying on rustc-private span or hygiene internals can fail. Under JIT (`haste run` and `haste test`), `std::env::current_exe()` resolves to the `haste` binary rather than a standalone target or test binary, so self-spawning tests (`Command::new(current_exe())`) fail under JIT; use `haste build -o` or `cargo test` for those.
- **Unsandboxed `build.rs` and proc-macro execution:** Both the CLI and `Compiler::check` execute `build.rs` scripts and procedural macros with the invoking user's privileges. Untrusted or customer-submitted repositories require isolated, resource-limited workers without service credentials. See [SECURITY.md](SECURITY.md).
- **Platform and executable format limits:** Supported hosts and AOT `--target` triples are `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`, and `aarch64-pc-windows-msvc`. Linux ELF binaries are non-PIE and target glibc; Linux→macOS cross-builds (both ARM64 and x86-64) are unsigned until `codesign`ed on macOS (`codesign --force --sign -`); Windows PE32+ emission is experimental, non-PIE, and imports CRT shims from `ucrtbase.dll` and `msvcrt.dll`. JIT `run`/`test`, `serve`, `--daemon`, and `--watch` require Unix; `prep` / `--prepared` are Linux x86-64 only (using `RUSTC_BOOTSTRAP=1`). On macOS, Hardened Runtime or App Sandbox restrictions can block dual-mapped `mach_vm_remap` JIT pages.
- **Debug info, unwinding, and traps:** `haste` emits DWARF `.debug_line` (`HASTE_DEBUG_INFO`) on Linux ELF only (no `.debug_info` variable/type DIEs). Mach-O emits `LC_UUID`, `LC_FUNCTION_STARTS`, and a symbol table without DWARF `.debug_line`; Windows PE32+ omits `.pdata` unwind tables and PDBs. In `--release` AOT builds (`haste --release build -o`), bounds, divide-by-zero, and signed `MIN / -1` division-overflow panics lower to hardware traps unless `catch_unwind` or `std::thread::{spawn, spawn_unchecked, spawn_scoped, scope}` is reached (while `haste --release test` and JIT always keep unwinding enabled, and integer `+`/`-`/`*` wrap in `--release`). Panicking across `extern "C"` aborts (`extern "C-unwind"` unwinds).

## Commands, options, and environment

```text
haste [OPTIONS] build|check|test|run|serve|prep [PATH] [OPTIONS] [--] [ARGS...]
haste serve status [PATH]
haste serve stop [PATH | --all]
haste clean
haste [-j|--jobs N] debug graph|outline|lazy|typeck|scope|items|expand|failures PATH [CRATE]
haste [--release] debug code PATH DIR
haste -V|--version | -h|--help
```

`PATH` defaults to `.`. Options are accepted before or after `PATH` up to `--` (place options before `debug` for `haste debug`); everything after `--` passes verbatim to the target program's `std::env::args()` or test filter. Cargo-compatible flags include `--workspace`, `-p`/`--package`, `--manifest-path`, `--tests`, `--all-targets`, `--profile <dev|debug|release>`, `--locked`, `--offline`, `--message-format=<human|short|json|json-render-diagnostics>`, `-q`/`--quiet`, `-j`/`--jobs <N>`, and feature flags, along with `RUSTFLAGS` / `CARGO_ENCODED_RUSTFLAGS` and `.cargo/config.toml` `[build].rustflags` / `[env]`. Run `haste <command> --help` for the full per-command flag list.

Before running or emitting code, `haste` type-checks and borrow-checks the full selected member crate (including uncalled and generic functions) with multi-label diagnostics.

| Command | Does |
| --- | --- |
| `run [PATH] [--] [ARGS...]` | Checks the member crate, then runs the binary or `--example` `main` via JIT (Unix) with isolated `ARGS`; pass `--watch` for live hot-reload. |
| `test [PATH] [--] [FILTER...]` | Checks the member crates, then runs `#[test]` and `#[tokio::test]` functions (unit and `tests/*.rs`, `#[ignore]`, `#[should_panic]`, captured panic stderr); skips doctests. |
| `build [-o PATH] [PATH]` / `check [PATH]` | Checks the member crate(s) and reached code (`check` is an alias for `build`), or emits a native executable when `-o PATH` is given. |
| `serve [PATH]` | Runs a foreground resident daemon for `PATH`; use `serve status [PATH]` and `serve stop [PATH \| --all]` to inspect or stop daemons. |
| `prep [PATH]` | Pre-compiles dependencies for `--prepared` (Linux x86-64 only). |
| `clean` | Stops active daemons and removes the local haste cache directory (`$HASTE_HOME`, `$XDG_CACHE_HOME/haste`, or `~/.cache/haste`). |
| `debug QUERY PATH [CRATE]` | Prints one compiler stage or dumps machine code (`debug code PATH DIR`). |

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` (`%LOCALAPPDATA%\haste` on Windows) | Cache directory for stdlib index, substrates, native helpers, `build.rs` outputs, and daemon registry. |
| `HASTE_LIBRARY` | Active `rustc --print sysroot` `rust-src` library, else newest `rustup` toolchain with `rust-src` | Explicit path to the Rust standard library `library/` directory. |
| `HASTE_NO_DAEMON` | Unset (`0`) | Set to `1` to bypass running `haste serve` daemons. |
| `HASTE_SERVE_IDLE_SECS` | `900` (`--daemon`); unset (`haste serve`) | Idle timeout in seconds for `--daemon` workers; foreground `haste serve` runs until stopped unless set. |
| `HASTE_NO_NOOP` | Unset (`0`) | Set to `1` to disable the unchanged-workspace skip check on one-shot `haste build`. |
| `HASTE_X64_BASELINE` | Unset (`0`) | Set to `1` on x86-64 to disable host CPU feature detection beyond baseline SSE2. |
| `HASTE_FN_ALIGN` | `16` (loop-free) / `64` (with loops) | Uniform function alignment override in bytes (power of two, `>= 16`) for AOT binaries. |
| `HASTE_DEBUG_INFO` | Enabled in debug builds | Set to `1` to emit DWARF `.debug_line` in `--release` Linux ELF binaries, or `0` to disable in debug builds. |
| `HASTE_SYMBOLS` | Enabled when debug info is on | Set to `1` to keep the symbol table in `--release` AOT binaries without `.debug_line`, or `0` to strip it. |

See [ARCHITECTURE.md](ARCHITECTURE.md#internal-debugging-and-tuning-variables) for compiler-internal trace, verification, and stencil tuning variables.

## Verify it yourself

To build from a local checkout and run the verification suites:

```bash
cargo build --release -p haste
./target/release/haste --version
./target/release/haste run examples/hello -- --your-arg

cargo test --locked -p haste-codegen                                                 # codegen unit tests
cargo test --locked -p haste --lib                                                   # compiler unit tests
cargo test --locked -p haste --test compiler_api --test soundness --test regressions # integration & soundness tests
bash scripts/check.sh                                                                # JIT + AOT differential suite vs rustc
bash scripts/check.sh tests/async                                                    # async differential suite vs rustc
bash scripts/check.sh tests/proc-macro                                               # proc-macro differential suite vs rustc
bash scripts/probes.sh                                                               # real-world crate probes vs rustc
bash scripts/bench.sh                                                                # build-speed and AOT runtime benchmarks
bash scripts/rustc-ui.sh                                                             # upstream rust-lang/rust tests/ui/ suite
```

## Contributing

Bug reports are welcome in [issues](https://github.com/manaforged/haste/issues). Pull requests are not accepted yet. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
