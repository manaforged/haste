# haste

haste is a clean-room Rust compiler, JIT engine, and standalone AOT compiler for your inner loop—built from scratch with its own frontend, borrow checker, SSA optimizer, and native x86-64 and ARM64 backends, with **no LLVM** and **no `target/` directory**.

Use **haste** for instant local edit-run-test cycles, live hot-reloading, and zero-linker cross-compilation; keep `cargo check` / `rustc` in CI for full Polonius lifetime verification and production releases.

- **Instant inner loop:** Single-line edit rebuilds in **12–13 ms** on `tests/diff` (**8x–37x faster than `cargo`**) and **0.5–6.0 ms** no-op checks—with zero multi-GB `target/` directories.
- **Live hot-reloading:** `haste run --watch` watches workspace source files and patches changed function bodies directly into the running process.
- **Zero-linker 6-target AOT cross-compilation:** `haste --release build -o ./app --target TRIPLE` emits standalone ELF, Mach-O, and PE32+ executables for Linux, macOS, and Windows (x86-64 & ARM64) with no Docker and no external cross-linker.
- **Self-hosting:** Compiles its own 86-package, ~178k-function workspace (including `serde`, `serde_derive`, `toml`, `syn`, `tracing`, `anyhow`, `indexmap`, `memchr`) in **10.3–10.8 s** (`0` failed; `13.1 s` end-to-end vs `23.9 s` for cold `cargo build` on macOS ARM64).
- **Not a toy backend:** Matches or beats `rustc -C opt-level=3` on **3 of 4** AOT compute kernels on both x86-64 (`sha256`, `sieve`, `scan`) and ARM64 (`sha256` **0.78x**, `scan` **0.98x**, `sort` **1.00x**, `scripts/bench.sh`, bit-identical outputs), and passes **620/620** differential tests, **90/90** async tests, **15/15** proc-macro tests, **17/17** crate probes (`jit+aot`), and **259/259 (100.0%)** single-file edition-2021 `rust-lang/rust` `tests/ui/` `run-pass` tests (`jit+aot`).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the compiler pipeline and module tour.

## Quickstart (60 seconds)

Requires Rust 1.88+ with the `rust-src` component:

```bash
rustup component add rust-src
cargo build --release -p haste

# 1. Run a crate via JIT (no target/ directory created)
./target/release/haste run examples/hello

# 2. Hot-reload edits live into a running process
./target/release/haste run --watch examples/hello

# 3. Emit a standalone native AOT executable
./target/release/haste --release build -o ./hello examples/hello

# 4. Zero-linker cross-compile a Linux x86-64 ELF from macOS (or vice versa)
./target/release/haste --release build --target x86_64-unknown-linux-gnu -o ./hello-linux examples/hello
```

If a workspace has no `Cargo.lock`, haste runs `cargo generate-lockfile` automatically on first use. Build scripts (`build.rs`), declarative and procedural macros, `const fn`, `async`/`await`, generics, trait objects, SIMD, and C FFI all run inside haste. Settled standard-library summaries, build script outputs, macro expansions, and compiled workspace stencils stay in `~/.cache/haste`, keyed by their inputs, so repeated builds rebuild only edited items and skip unchanged workspaces.

haste is experimental. There is no tagged release yet, and the command line can change.

## Commands

```text
haste [-p PKG] [--bin NAME | --example NAME] [-F FEATURES] [--all-features] [--no-default-features] [--target TRIPLE] [--jobs N] [--release] [--prepared] [--prep-wait MS] [--perf-map] [--watch] [--stats-json PATH] [-o PATH] build|test|run|serve|prep [PATH] [--] [ARGS...]
haste clean
haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand|failures PATH [CRATE]
haste [--release] debug code PATH DIR
haste -V|--version | -h|--help
```

`PATH` defaults to `.` for `build`, `test`, `run`, `serve`, and `prep`.

| Command | Does |
| --- | --- |
| `run [PATH] [--] [ARGS...]` | Runs the workspace binary or `--example` `main` with `ARGS`. |
| `test [PATH] [--] [FILTER...]` | Runs unit and integration (`tests/*.rs`) `#[test]` functions (with `#[cfg(test)]`, `[dev-dependencies]`, `#[ignore]`, and `#[should_panic]`), or `fn(i64) -> i64` `case_*` harness functions when no `#[test]` functions are present. |
| `build [-o PATH] [PATH]` | Type-checks the workspace and lowers reached code, or writes a standalone executable when `-o PATH` is given. |
| `serve [PATH]` | Keeps the workspace loaded in memory and answers `build`, `test`, and `run` requests for `PATH`, rebuilding only what edited files affect. |
| `prep [PATH]` | Pre-compiles dependencies that `main` reaches into the prepared library cache for `--prepared` (**x86-64 Linux only**). |
| `clean` | Removes the haste cache directory (`$HASTE_HOME`, `$XDG_CACHE_HOME/haste`, or `~/.cache/haste`). |
| `debug QUERY PATH [CRATE]` | Prints one compiler stage: unit graph, module outline, lazy reach, type checking, scopes, items, macro expansions, failures, or emitted machine code (`code PATH DIR`). |

### Options

- `-V`, `--version`: Prints the compiler version and exits.
- `-h`, `--help`: Prints command usage and exits.
- `-p PKG`, `--package PKG`: Selects a specific workspace package to build or run.
- `--bin NAME`: Selects which binary target (`src/main.rs`, `src/bin/*.rs`, or `[[bin]]`) to build or run.
- `--example NAME`: Selects which example target (`examples/*.rs` or `[[example]]`) to build or run (with `[dev-dependencies]`).
- `-F FEATURES`, `--features FEATURES`: Comma- or space-separated list of Cargo features to enable (repeatable).
- `--all-features`: Enables all features on the selected workspace packages.
- `--no-default-features`: Disables the `default` feature on the selected workspace packages.
- `--ignored`: Runs only `#[ignore]` tests (`haste test` only).
- `--include-ignored`: Runs both normal and `#[ignore]` tests (`haste test` only).
- `--target TRIPLE`: Cross-compiles a standalone AOT executable (`haste build -o`) for any supported target triple (`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`) without an external cross-linker.
- `--jobs N`: Sets the worker thread or process count. Defaults to the cores available to the process.
- `--release`: Enables full codegen optimization and release `cfg` settings.
- `-o PATH`, `--output PATH`: Writes a standalone native executable (`haste build` only).
- `--watch`: Watches workspace source files and hot-reloads changed function bodies into the running process (`haste run` only).
- `--stats-json PATH`: Writes stage timing and reachability statistics as JSON (`haste build` only).
- `--prepared`: Links pre-compiled dependency code produced by `haste prep` (**x86-64 Linux only**).
- `--prep-wait MS`: Waits up to `MS` milliseconds for a running `prep` job before compiling directly (**x86-64 Linux only**).
- `--perf-map`: Writes `/tmp/perf-<pid>.map` symbol entries for JIT profiling.

## Requirements and environment

- Hosts: Linux (`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`), macOS (`aarch64-apple-darwin`, `x86_64-apple-darwin`), and Windows (`x86_64-pc-windows-msvc`, `x86_64-pc-windows-gnu`, `aarch64-pc-windows-msvc`; Windows runtime execution is experimental and `serve`/`--watch` require Unix).
- AOT cross-compilation (`--target TRIPLE`): standalone ELF, Mach-O, and PE32+ executables for all six x86-64 and ARM64 Linux, macOS, and Windows targets.
- Rust 1.88 or newer with the `rust-src` component installed (`rustup component add rust-src`).
- On macOS, `codesign` (included with macOS / Xcode Command Line Tools) for ad-hoc signing standalone AOT executables.
- A C compiler (`$CC`, defaulting to `cc`) when building workspaces that link native C libraries.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` | Cache directory for build script outputs, macro expansions, standard library summaries, and compiled stencils. |
| `HASTE_LIBRARY` | Active `rustc --print sysroot` `rust-src` library, else newest `rustup` toolchain with `rust-src` | Path to the Rust standard library `library/` source tree. |
| `HASTE_NO_SUBSTRATE` | Unset (`0`) | Set to `1` to bypass the cached standard library and workspace substrate. |
| `HASTE_NO_NOOP` | Unset (`0`) | Set to `1` to disable the unchanged-workspace skip check on one-shot `haste build`. |
| `HASTE_NO_DAEMON` | Unset (`0`) | Set to `1` to disable automatic background `haste serve` reuse on Unix. |
| `HASTE_X64_BASELINE` | Unset (`0`) | Set to `1` on x86-64 to disable host ISA feature detection beyond baseline SSE2. |
| `HASTE_FN_ALIGN` | `64` | Function alignment in bytes (power of two, `>= 16`) for standalone AOT executables. |
| `HASTE_DEBUG_INFO` | Unset (enabled in debug builds) | Set to emit DWARF debug info in `--release` AOT executables. |
| `HASTE_SYMBOLS` | Unset (enabled when debug info is on) | Set to `1` to keep the symbol table in `--release` AOT executables without full debug info. |
| `HASTE_DICT_MIN_BYTES` | `512` | Minimum generic body size in bytes before dictionary lowering is considered. |
| `HASTE_DICT_MIN_FANOUT` | `8` | Minimum monomorphization fanout before sharing a dictionary body. |
| `HASTE_JIT_TIER_UP` | `64` | Call count threshold before a hot dictionary function specializes in JIT mode (`0` disables tier-up). |
| `HASTE_VERIFY_STENCIL` | Unset (`0`) | Set to `1` to verify stencil instantiations against concrete lowering. |
| `HASTE_TRACE` | Unset (`0`) | Function substring filter (`1` matches all) for printing codegen pass summaries. |
| `HASTE_TRACE_IR` | Unset (`0`) | Set to `1` with `HASTE_TRACE` to dump IR after each codegen pass, or `passes` for pass-by-pass summaries. |
| `HASTE_MISSED` | Unset (`0`) | Set to `1` to print aggregated codegen pass timings and stencil cache counters. |

## Benchmarks

Run `bash scripts/bench.sh` to reproduce the build-speed and AOT runtime tables on your machine. Every workload verifies bit-identical output against `rustc -C opt-level=3` before timing.

### Build speed (vs stock `cargo`)

| Workload | Platform | haste | cargo | Speedup |
| --- | --- | ---: | ---: | ---: |
| `tests/diff` (124 fns) — single-line edit rebuild | macOS ARM64 | 13.1 ms | 491 ms | 37.6x |
| `tests/diff` (124 fns) — single-line edit rebuild | Linux x86-64 | 12.4 ms | 102 ms | 8.2x |
| `tests/diff` (124 fns) — unchanged no-op | Linux x86-64 | 0.5 ms | 4.8 ms | 9.8x |
| Self-host `haste` (86 pkgs, 178k fns) — cold debug build | macOS ARM64 | 13.1 s | 23.9 s | 1.8x |
| Self-host `haste` (86 pkgs, 178k fns) — unchanged no-op | macOS ARM64 | 6.0 ms | 36.4 ms | 6.0x |

### AOT codegen runtime (`haste --release` vs `rustc -C opt-level=3`, median of 5)

| Workload | macOS ARM64 (`haste` / `rustc -O3`) | Linux x86-64 (`haste` / `rustc -O3`) | Output |
| --- | ---: | ---: | --- |
| `sha256` (250k blocks) | 57.1 ms / 73.0 ms (**0.78x**) | 32.1 ms / 36.0 ms (**0.89x**) | exact match (`7333157476866902348`) |
| `sieve` (4M primes) | 7.0 ms / 6.7 ms (**1.06x**) | 3.6 ms / 3.9 ms (**0.93x**) | exact match (`544501644261`) |
| `sort` (500k quicksort + bsearch) | 29.1 ms / 29.0 ms (**1.00x**) | 23.4 ms / 23.0 ms (**1.02x**) | exact match (`8647866321337317866`) |
| `scan` (JSON-like byte scanner) | 7.5 ms / 7.6 ms (**0.98x**) | 2.6 ms / 2.8 ms (**0.95x**) | exact match (`80005120000`) |

## Verify it yourself

`tests/diff`, `tests/async`, and `tests/proc-macro` hold `case_*` suites, and `tests/probes` holds standalone crates. `scripts/check.sh` compares `cargo run` against both `haste test` (JIT) and `haste --release build -o` (standalone AOT).

```bash
bash scripts/check.sh                   # 620/620 differential cases (jit+aot) vs rustc
bash scripts/check.sh tests/async       # 90/90 async cases (jit+aot)
bash scripts/check.sh tests/proc-macro  # 15/15 proc-macro cases (jit+aot)
bash scripts/probes.sh                  # 17/17 standalone crate probes (jit+aot) vs rustc
bash scripts/bench.sh                   # Build speed & AOT runtime benchmarks vs rustc -O3
bash scripts/rustc-ui.sh                # Official rust-lang/rust tests/ui/ conformance suite
```

On `rust-lang/rust` `tests/ui/` (filtered to single-file edition-2021 tests without `#![feature]`/`aux-build`/`compile-flags`/`revisions`), `scripts/rustc-ui.sh` passes **259/259 (100.0%)** `run-pass` tests (`jit+aot`, `0` failed, `0` crashed) and **172/341 (50.4%, `0` crashes)** negative `tests/ui/borrowck/` tests (the remaining accepted cases are largely uncalled functions in `fn main() {}` binaries or full-Polonius/HRTB lifetime checks).

The scripts run `cargo`, or the command in `$CARGO` or `~/.config/haste/cargo`.

## Limits

- **Windows runtime:** Windows host runtime execution and PE32+ binaries built via `--target x86_64-pc-windows-msvc` / `aarch64-pc-windows-msvc` are experimental, and `serve`/`--watch` require Unix.
- **Full Polonius & HRTB lifetimes:** haste checks ownership, use-after-move (including across branches and loops), binding mutability, disjoint struct fields, NLL-style borrow aliasing, and dangling local reference returns on reached workspace functions, though full Polonius and higher-ranked lifetime (`for<'a>`) inference are not yet implemented; run `cargo check` or `rustc` in CI for complete lifetime verification.
- **`catch_unwind` and `Drop` unwinding:** In debug builds, `haste test`, and any build using `std::panic::catch_unwind` or threads, out-of-bounds indexing and divide-by-zero panic with a message and run `Drop` destructors in both JIT and standalone AOT (x86-64 and ARM64); in `--release` builds without `catch_unwind`, bounds checks and divide-by-zero emit hardware traps for zero-overhead codegen, uncaught panics print their message and abort, and unwinding across C FFI frames is not supported.
- **Full `.debug_info`:** Standalone AOT executables emit symbol tables (`HASTE_SYMBOLS`) and DWARF `.debug_line` tables (`HASTE_DEBUG_INFO`) for backtraces and profiling, not full `.debug_info` variable/type DIEs.
- **Test harnesses:** `haste test` runs synchronous `fn()` and `fn() -> Result<(), E>` (`Termination`) `#[test]` functions (not `#[tokio::test]` macro suites that depend on custom harnesses).
- **Prepared mode:** `haste prep` and `--prepared` are x86-64 Linux only and compile an internal `rustc_private` driver with `RUSTC_BOOTSTRAP=1`, which is tied to the installed `rustc` version.
- **macOS Hardened Runtime:** On macOS, JIT execution uses dual-mapped `mach_vm_remap` pages (writable + executable aliases), which may be blocked inside a Hardened Runtime or App Sandbox.
- **Feature coverage:** Some Rust features and crates.io edge cases do not compile yet. If any reached function fails to compile, `haste build` exits non-zero.

## Contributing

Bug reports are welcome as issues. Pull requests are not accepted yet.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
