# Benchmarks, compatibility, and verification

This document holds the full benchmark tables, compatibility matrix, environment variables, and verification commands for `haste`. Run `bash scripts/bench.sh` to reproduce build-speed and AOT runtime numbers on your machine (pass `--long` for 4x compute iterations).

By default, `scripts/bench.sh` runs with `HASTE_BENCH_COLD=1`, which isolates `HASTE_HOME` and reports two distinct cold rows on `tests/diff` against a fresh `cargo` target directory: **Empty `HASTE_HOME`** (first command after install, including one-time stdlib indexing) and **First project build, stdlib indexed** (first build of a new project after stdlib indexing on `examples/hello`). Set `HASTE_BENCH_COLD=0` to reuse your existing `$HASTE_HOME` and print `Warm cache` rows instead.

## Build speed (`tests/diff`, bundled 144-case fixture, `HASTE_BENCH_COLD=1`)

Resident-daemon edits (`--daemon`) compare a warm background `haste serve` process against one-shot `cargo check` (using `rustup which cargo` so PATH shims do not skew the baseline). `Empty HASTE_HOME` includes one-time stdlib indexing (`915–966 ms` for `haste build` vs `200–249 ms` for `cargo check`); once stdlib is indexed, the first build of a new project drops to `82.0–136 ms` (`1.5x–3.0x` faster than `cargo check`).

| Benchmark | Platform | haste | cargo | Speedup (higher = faster) |
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

## Multi-crate workspace (unbundled 340-crate workspace)

Measured on a private multi-crate workspace (not bundled; `340` crates, `45` build scripts, `17,521` functions, macOS ARM64, 6 CPUs):

- Cold check (empty `HASTE_HOME`): `18.7 s` (`159 MiB` shared cache, `+6–18 MiB` per additional workspace; checks reached functions for compiler errors only, with no warnings or lints) vs `34.6 s` for cold `cargo check --workspace -j 6` (which checks all items and warnings).
- Daemon cold start (`haste build --daemon`): `4.65 s` (`900 s` idle timeout).
- Leaf edit: `22.3 ms` median (`22.1–37.3 ms`, `87x` faster than `cargo check` at `1.95 s`) with `haste build --daemon`, vs `2.35 s` median (`2.33–2.49 s`) for one-shot `haste build`.
- Cross-crate public-type edit: `2.35 s` (`2.33–2.40 s`) for one-shot `haste build` and `3.00 s` (`2.94–3.09 s`) with `haste build --daemon`, vs `2.21 s` for `cargo check`.

## Live hot-reload (`examples/live-server`, macOS ARM64)

- Body edit in `handle()`: `85.3 ms` save-to-first-changed-response, keeping the same PID and open listener socket.
- Compile error while watching: keeps serving the previous code and recovers in `135.3 ms` once fixed.
- Structural edit while watching: keeps serving the previous code and prints a restart-required notice.

## AOT codegen runtime (`haste --release` vs `rustc`, baseline-subtracted median of 5, warmed)

The compute tables below show baseline-subtracted `net` times (`scripts/bench.sh` prints both `net` and `raw` times as `<net> (raw <raw>)`), subtracting each compiler's warmed empty-main startup baseline (median of 7) and verifying bit-identical outputs across `haste --release`, stock `rustc -C opt-level=3`, and matched-flag `rustc -C opt-level=3 -C target-cpu=native -C panic=abort`.

In `--release` AOT builds (`haste --release build -o`), bounds, divide-by-zero, and signed `MIN / -1` division-overflow checks lower to hardware traps unless `catch_unwind` or `std::thread::{spawn, spawn_unchecked, spawn_scoped, scope}` is reached (`haste --release test` and JIT always keep unwinding enabled, and integer `+`/`-`/`*` wrap in `--release`).

### Linux x86-64 (`rustc 1.99.1`, empty-main baseline: 0.2 ms across all three)

| Workload | `haste --release` (net) | `rustc -O3` (net) | `haste / rustc -O3` (`<1.00x` = faster) | `rustc -O3 native+abort` (net) | `haste / matched` (`<1.00x` = faster) | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (2M blocks) | 229 ms | 289 ms | **0.79x** | 253 ms | **0.90x** | exact match (`2980548810864065682`) |
| `sieve` (20x 4M primes) | 44.7 ms | 65.3 ms | **0.68x** | 51.6 ms | **0.87x** | exact match (`10840522878760`) |
| `sort` (8x 500k quicksort + bsearch) | 186 ms | 188 ms | **0.99x** | 189 ms | **0.99x** | exact match (`1823227901928002215`) |
| `scan` (1200x JSON byte scanner) | 66.9 ms | 76.8 ms | **0.87x** | 77.3 ms | **0.87x** | exact match (`2400205310288`) |
| `fp_iter` (4096x f64 generic iterator) | 164 ms | 161 ms | 1.02x | 164 ms | **1.00x** | exact match (`4733336957083238967`) |

### macOS ARM64 (`rustc 1.98.1`, empty-main baseline: `haste` 1.6 ms, `-O3` 3.8 ms, matched 3.2 ms)

| Workload | `haste --release` (net) | `rustc -O3` (net) | `haste / rustc -O3` (`<1.00x` = faster) | `rustc -O3 native+abort` (net) | `haste / matched` (`<1.00x` = faster) | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (2M blocks) | 282 ms | 290 ms | **0.97x** | 291 ms | **0.97x** | exact match (`2980548810864065682`) |
| `sieve` (20x 4M primes) | 61.8 ms | 81.1 ms | **0.76x** | 81.6 ms | **0.76x** | exact match (`10840522878760`) |
| `sort` (8x 500k quicksort + bsearch) | 180 ms | 182 ms | **0.99x** | 181 ms | **1.00x** | exact match (`1823227901928002215`) |
| `scan` (1200x JSON byte scanner) | 106 ms | 101 ms | 1.05x | 104 ms | 1.02x | exact match (`2400205310288`) |
| `fp_iter` (4096x f64 generic iterator) | 135 ms | 162 ms | **0.83x** | 163 ms | **0.83x** | exact match (`4733336957083238967`) |

## Upstream `rust-lang/rust` UI suite (`scripts/rustc-ui.sh`, macOS ARM64)

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
| Crates bundling static C archives (`lib*.a`, e.g. `rusqlite` `bundled`, `tikv-jemalloc-sys`) | JIT + host-local Unix AOT (`build -o`); cross-target AOT errors | Host-local `$HASTE_HOME/native/` shared helper (not standalone); see [README.md#limits](README.md#limits) |
| `reqwest` / `openssl-sys` | Untested | Not covered in `tests/probes` |
| Nightly `#![feature(...)]` crates | Unsupported | Stable syntax and stdlib only |

## Environment variables

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
