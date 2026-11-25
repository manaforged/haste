# haste

haste is a standalone Rust compiler and JIT runner that runs, tests, and builds Rust crates with no LLVM dependency and no `target/` directory.

Run binaries and `#[test]` suites in memory on Unix (`haste run`, `haste test`), hot-reload edited functions into a running process (`haste run --watch`), or emit native ELF, Mach-O, and PE32+ executables directly without an external cross-linker (`haste build -o`). Plain `haste` commands run one-shot and leave no background daemon.

haste is experimental (`version = "0.0.0"` in `Cargo.toml`, no tagged release yet). Install it from this repository with `--git`, not `cargo install haste` (which installs an unrelated benchmarking crate on crates.io). See [ARCHITECTURE.md](ARCHITECTURE.md) for the compiler pipeline, SSA optimization passes, register allocator, memory-mapped cache layers, and module layout.

## Install and quickstart

Prerequisites: Rust 1.88 or newer with the `rust-src` component, plus a C compiler and `make` on macOS and Linux x86-64 GNU (required to build `tikv-jemallocator` from source).

```bash
rustup component add rust-src
cargo install --git https://github.com/manaforged/haste haste
haste --version # prints: haste 0.0.0 (<commit>)
```

Try it on a standalone crate from any directory (the first run for a toolchain performs a one-time index of `rust-src` and logs `haste: indexing standard library (one-time)...` to stderr):

```bash
cargo new --bin haste-demo

# Unix: run in memory via JIT (prints: Hello, world!)
haste run haste-demo -- --your-arg

# Windows (or any platform): emit and run a standalone native AOT executable
haste --release build -o ./haste-demo-bin haste-demo
./haste-demo-bin

# Cross-compile a Linux x86-64 ELF binary from macOS (or vice versa) without a cross-linker
haste --release build --target x86_64-unknown-linux-gnu -o ./haste-demo-linux haste-demo
```

To opt in to the resident workspace daemon on Unix for sub-20 ms edit checks (`8.4–9.0 ms` in `tests/diff`, `6.2x–7.4x` faster than `cargo check`), pass `--daemon` and pass the same workspace path to `serve status` and `serve stop`:

```bash
haste build --daemon haste-demo
haste serve status haste-demo
haste serve stop haste-demo
```

To watch and hot-reload edited functions on Unix, run `haste run --watch <path>` on a long-running crate (such as a loop or server); `haste` patches changed functions on their next call in the running process.

To build from a local checkout instead of `cargo install --git`:

```bash
cargo build --release -p haste
./target/release/haste --version
./target/release/haste run examples/hello -- --your-arg
```

If a workspace has no `Cargo.lock`, `haste` runs `cargo generate-lockfile` automatically. If locked registry or git crates are missing locally, `haste` runs `cargo fetch` automatically. Build scripts (`build.rs`), declarative and procedural macros, `const fn`, `async`/`await`, generics, trait objects, SIMD, and C FFI run inside `haste`, with cached outputs stored under `~/.cache/haste`.

## Crate compatibility

`scripts/probes.sh` cross-checks real crates and language features against `rustc` in both JIT and `--release` AOT modes:

| Crate or workload | Status | Source / notes |
| --- | --- | --- |
| Pure-Rust multi-crate workspaces (`[workspace.dependencies]`, path crates) | Supported (JIT + AOT) | `tests/probes/ws` |
| `serde` (`derive`) + `serde_json` (including `raw_value`) | Supported (JIT + AOT) | `tests/probes/sj`, `tests/probes/fp`, `tests/probes/ws` |
| `toml` (`0.8`) | Supported (JIT + AOT) | `tests/probes/fp` |
| `clap` (`4.x`, `derive`) | Supported (JIT + AOT) | `tests/probes/cd` |
| `regex` (`1.x`) | Supported (JIT + AOT) | `tests/probes/rx` |
| `winnow` (`0.7`) | Supported (JIT + AOT) | `tests/probes/wn` |
| Procedural macros (`derive`, attribute, function-like) | Supported (JIT + AOT) | `tests/proc-macro`, `tests/probes/pc` (some rustc-internal span/hygiene edge cases fail) |
| Crates bundling native C static archives (`lib*.a`, e.g. `ring`, `tikv-jemalloc-sys`) | Unsupported in AOT | `haste build -o` emits dynamic system/C imports only; `haste build .` checks `haste`'s 42-package workspace (0 failed functions), and JIT loads `build.rs` archives via a generated shared library |
| `reqwest` / `openssl-sys` | Untested | Not covered in `tests/probes` |
| `#[tokio::test]` / custom test harnesses | Unsupported | `haste test` runs synchronous `fn()` and `fn() -> Result<(), E>` tests only |
| Nightly `#![feature(...)]` crates | Unsupported | Stable syntax and stdlib only |

## Embedding haste as a library

`haste` exposes an embeddable API (`Compiler`, `CompilerBuilder`, `Target`, `Error`, `ErrorKind`, `Diagnostic`, `DiagnosticKind`, `Label`, `Span`) with explicit per-build configuration and no process-global configuration state:

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

## Commands and options

```text
haste [OPTIONS] build|test|run|serve|prep [PATH] [OPTIONS] [--] [ARGS...]
haste serve status [PATH]
haste serve stop [PATH | --all]
haste clean
haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand|failures PATH [CRATE]
haste [--release] debug code PATH DIR
haste -V|--version | -h|--help
```

`PATH` defaults to `.` for `build`, `test`, `run`, `serve`, and `prep`. Options are accepted before or after `PATH` up to `--` for `build`, `test`, `run`, `serve`, and `prep` (place options before `debug` for `haste debug`); everything after `--` passes verbatim to the target program's `std::env::args()` or test filter.

`haste build`, `haste run`, `haste test`, and `haste build -o` all type-check and borrow-check the full selected member crate (including uncalled functions and polymorphic items) before running or emitting an executable. Checked rules include ownership, moves across branches and loops, binding mutability, disjoint struct-field borrows, NLL aliasing, non-`'static` borrow escapes, named-lifetime return subtyping, structural `Send`/`Sync` auto traits, match exhaustiveness, module and field privacy, concrete and generic trait bounds, and checked `const` overflow, reported with spanned multi-label diagnostics.

| Command | Does |
| --- | --- |
| `run [PATH] [--] [ARGS...]` | Checks the member crate, then runs the binary or `--example` `main` via JIT with isolated `ARGS`. |
| `test [PATH] [--] [FILTER...]` | Checks the member crates, then runs unit and integration (`tests/*.rs`) `#[test]` functions (with `#[cfg(test)]`, `[dev-dependencies]`, `#[ignore]`, `#[should_panic]`, and captured panic stderr on failure). |
| `build [-o PATH] [PATH]` | Checks the member crate(s) and reached code, or writes a standalone executable when `-o PATH` is given. |
| `serve [PATH]` | Runs a foreground resident daemon for `PATH`, rebuilding only what edited files affect. |
| `serve status [PATH]` | Lists active opt-in `haste serve` daemons (optionally filtered to `PATH`). |
| `serve stop [PATH \| --all]` | Stops active `haste serve` daemons (for `PATH` or `--all`) and removes their Unix sockets. |
| `prep [PATH]` | Pre-compiles dependencies for `--prepared` (`x86_64-unknown-linux-gnu` hosts only). |
| `clean` | Removes the local haste cache directory (`$HASTE_HOME`, `$XDG_CACHE_HOME/haste`, or `~/.cache/haste`). |
| `debug QUERY PATH [CRATE]` | Prints one compiler stage (`graph`, `outline`, `lazy`, `typeck`, `scope`, `items`, `expand`, `failures`) or writes machine code (`code PATH DIR`). |

### Options

- `-V`, `--version`: Prints the compiler version and short commit hash, then exits.
- `-h`, `--help`: Prints global or subcommand usage (`haste <command> --help`) and exits.
- `-p PKG`, `--package PKG`: Selects a workspace package to build or run.
- `--bin NAME`: Selects a binary target (`src/main.rs`, `src/bin/*.rs`, or `[[bin]]`).
- `--example NAME`: Selects an example target (`examples/*.rs` or `[[example]]`, with `[dev-dependencies]`).
- `-F FEATURES`, `--features FEATURES`: Comma- or space-separated list of Cargo features to enable (repeatable).
- `--all-features`: Enables all features on the selected workspace packages.
- `--no-default-features`: Disables the `default` feature on the selected workspace packages.
- `--ignored`: Runs only `#[ignore]` tests (`haste test` only).
- `--include-ignored`: Runs both normal and `#[ignore]` tests (`haste test` only).
- `--daemon`: Opts in to reusing or spawning a background `haste serve` daemon on Unix (plain commands run one-shot).
- `--target TRIPLE`: Cross-compiles a standalone AOT executable (`haste build -o`) for any supported target triple without an external cross-linker; cross-compiled x86-64 binaries default to baseline SSE2 (see [Requirements and environment](#requirements-and-environment)).
- `--jobs N`: Sets the worker thread count (defaults to available cores).
- `--release`: Enables full codegen optimization and release `cfg` settings.
- `-o PATH`, `--output PATH`: Writes a standalone native executable (`haste build` only).
- `--watch`: Watches workspace source files and hot-reloads changed function bodies into the running process (`haste run` only).
- `--stats-json PATH`: Writes stage timing and reachability census data as JSON (`haste build` only).
- `--prepared`: Links pre-compiled dependency code produced by `haste prep` (`x86_64-unknown-linux-gnu` hosts only).
- `--prep-wait MS`: Waits up to `MS` milliseconds for a background `prep` job before compiling directly (`x86_64-unknown-linux-gnu` hosts only).
- `--perf-map`: Writes `/tmp/perf-<pid>.map` symbol entries for JIT profiling.

## Requirements and environment

- Supported host platforms and standalone AOT `--target` triples: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`, and `aarch64-pc-windows-msvc` (Windows runtime execution and PE32+ binaries are experimental; `serve`, `--daemon`, and `--watch` require Unix).
- Rust 1.88 or newer with `rust-src` installed (`rustup component add rust-src`).
- On macOS, `codesign` (included with macOS / Xcode Command Line Tools) for ad-hoc signing emitted Mach-O executables.
- A C compiler (`$CC`, defaulting to `cc`) when running JIT workspaces whose `build.rs` compiles or links native C libraries.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` (`%LOCALAPPDATA%\haste` on Windows) | Cache directory for stdlib index, substrates, `build.rs` outputs, and daemon registry. |
| `HASTE_LIBRARY` | Active `rustc --print sysroot` `rust-src` library, else newest `rustup` toolchain with `rust-src` | Explicit path to the Rust standard library `library/` directory. |
| `HASTE_NO_DAEMON` | Unset (`0`) | Set to `1` to bypass running `haste serve` daemons. |
| `HASTE_NO_NOOP` | Unset (`0`) | Set to `1` to disable the unchanged-workspace skip check on one-shot `haste build`. |
| `HASTE_X64_BASELINE` | Unset (`0`) | Set to `1` on x86-64 to disable host CPU feature detection beyond baseline SSE2. |
| `HASTE_FN_ALIGN` | `16` (loop-free) / `64` (with loops) | Uniform function alignment override in bytes (power of two, `>= 16`) for AOT binaries. |
| `HASTE_DEBUG_INFO` | Enabled in debug builds | Set to `1` to emit DWARF `.debug_line` in `--release` AOT binaries, or `0` to disable in debug builds. |
| `HASTE_SYMBOLS` | Enabled when debug info is on | Set to `1` to keep the symbol table in `--release` AOT binaries without `.debug_line`, or `0` to strip it. |

See [ARCHITECTURE.md](ARCHITECTURE.md#internal-debugging-and-tuning-variables) for compiler-internal trace, verification, and stencil tuning variables.

## Benchmarks

Run `bash scripts/bench.sh` to reproduce the build-speed and AOT runtime numbers on your machine (pass `--long` for 4x compute iterations). Compute benchmarks subtract each compiler's warmed empty-main startup baseline (median of 7) and compare bit-identical outputs across `haste --release`, stock `rustc -C opt-level=3`, and matched-flag `rustc -C opt-level=3 -C target-cpu=native -C panic=abort` (1 warmup run, median of 5).

### Build speed (`tests/diff`, recorded at 136 fns)

`scripts/bench.sh` warms the one-time standard-library index on `examples/hello` before timing `tests/diff`, so the uncached `tests/diff` rows measure a warm toolchain cache rather than a cold first run.

| Benchmark | Platform | haste | cargo | Speedup |
| --- | --- | ---: | ---: | ---: |
| Single-line edit (`haste build --daemon` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 8.4 ms | 62.2 ms | 7.4x |
| Single-line edit (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 66.3 ms | 62.2 ms | 0.9x |
| Single-line edit (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 161 ms | 185 ms | 1.1x |
| Unchanged no-op (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 2.3 ms | 12.1 ms | 5.3x |
| Unchanged no-op (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 2.3 ms | 12.4 ms | 5.4x |
| Warm cache (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 75.7 ms | 227 ms | 3.0x |
| Warm cache (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 226 ms | 508 ms | 2.3x |
| Single-line edit (`haste build --daemon` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 9.0 ms | 55.7 ms | 6.2x |
| Single-line edit (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 94.2 ms | 55.7 ms | 0.6x |
| Single-line edit (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 183 ms | 110 ms | 0.6x |
| Unchanged no-op (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 0.7 ms | 7.1 ms | 10.1x |
| Unchanged no-op (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 0.7 ms | 7.1 ms | 11.0x |
| Warm cache (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 113 ms | 202 ms | 1.8x |
| Warm cache (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 345 ms | 506 ms | 1.5x |

### AOT codegen runtime (`haste --release` vs `rustc`, baseline-subtracted median of 5, warmed)

#### Linux x86-64 (`rustc 1.99.1`, empty-main baseline: 0.2 ms across all three)

| Workload | `haste --release` (net) | `rustc -O3` (net) | vs `-O3` | `rustc -O3 native+abort` (net) | vs matched | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (2M blocks) | 229 ms | 289 ms | **0.79x** | 253 ms | **0.90x** | exact match (`2980548810864065682`) |
| `sieve` (20x 4M primes) | 44.7 ms | 65.3 ms | **0.68x** | 51.6 ms | **0.87x** | exact match (`10840522878760`) |
| `sort` (8x 500k quicksort + bsearch) | 186 ms | 188 ms | **0.99x** | 189 ms | **0.99x** | exact match (`1823227901928002215`) |
| `scan` (1200x JSON byte scanner) | 66.9 ms | 76.8 ms | **0.87x** | 77.3 ms | **0.87x** | exact match (`2400205310288`) |
| `fp_iter` (4096x f64 generic iterator) | 166 ms | 161 ms | 1.03x | 161 ms | 1.03x | exact match (`4733336957083238967`) |

#### macOS ARM64 (`rustc 1.98.1`, empty-main baseline: `haste` 1.6 ms, `-O3` 3.8 ms, matched 3.1 ms)

| Workload | `haste --release` (net) | `rustc -O3` (net) | vs `-O3` | `rustc -O3 native+abort` (net) | vs matched | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (2M blocks) | 282 ms | 290 ms | **0.97x** | 291 ms | **0.97x** | exact match (`2980548810864065682`) |
| `sieve` (20x 4M primes) | 61.2 ms | 81.0 ms | **0.76x** | 80.9 ms | **0.76x** | exact match (`10840522878760`) |
| `sort` (8x 500k quicksort + bsearch) | 180 ms | 182 ms | **0.99x** | 181 ms | **0.99x** | exact match (`1823227901928002215`) |
| `scan` (1200x JSON byte scanner) | 110 ms | 104 ms | 1.05x | 107 ms | 1.02x | exact match (`2400205310288`) |
| `fp_iter` (4096x f64 generic iterator) | 135 ms | 162 ms | **0.83x** | 163 ms | **0.83x** | exact match (`4733336957083238967`) |

## Verify it yourself

`tests/diff`, `tests/async`, and `tests/proc-macro` hold `case_*` suites (built with `--features diff-harness`), and `tests/probes` holds standalone crate probes. `scripts/check.sh` compares `cargo run` against both `haste test` (JIT) and `haste --release build -o` (standalone AOT).

```bash
cargo test --locked -p haste-codegen                                                 # 42 unit tests
cargo test --locked -p haste --lib                                                   # 30 unit tests on macOS, 29 on Linux, 28 on Windows
cargo test --locked -p haste --test compiler_api --test soundness --test regressions # 33 integration tests on Unix (8 compiler_api, 21 soundness, 4 regressions; 30 on Windows)
bash scripts/check.sh                                                                # 144 differential functions (x 9 inputs = 1,296 checks), JIT + AOT vs rustc
bash scripts/check.sh tests/async                                                    # 19 async functions (x 9 inputs = 171 checks), JIT + AOT vs rustc
bash scripts/check.sh tests/proc-macro                                               # 3 proc-macro functions (x 9 inputs = 27 checks), JIT + AOT vs rustc
bash scripts/probes.sh                                                               # 21 standalone crate probes (20 JIT + --release AOT, plus `so` JIT-only)
bash scripts/bench.sh                                                                # Build speed and 5-kernel AOT runtime benchmarks vs rustc
bash scripts/rustc-ui.sh                                                             # Official rust-lang/rust tests/ui/ suite (--borrowck requires error[borrow]:)
```

`scripts/rustc-ui.sh --borrowck` requires an explicit `error[borrow]:` diagnostic to count a negative test as `PASS`, reporting unrelated front-end errors separately as `OtherErr`. The scripts build `haste` using `$CARGO` or `~/.config/haste/cargo` (defaulting to `cargo`), while `scripts/bench.sh` invokes `cargo` directly for timed `cargo check` and `cargo build` measurements.

## Limits

- **Incomplete borrow-checker and nightly coverage:** Full Polonius borrow checking, some `tests/ui/borrowck` and higher-ranked (`for<'a>`) lifetime edge cases, and nightly `#![feature(...)]` attributes are not yet supported. Keep `cargo check` or `rustc` in CI for complete toolchain verification across unreached dependency code, nightly attributes, and Polonius/HRTB edge cases.
- **External dependencies stay lazy:** `haste` checks every concrete and generic function in the selected member crate before running or emitting code, while external crates and `std` are only parsed and inferred along reached paths.
- **Static C archives in standalone AOT:** `haste build .` type-checks and lowers `haste`'s own multi-crate workspace (42 lockfile packages, including `tikv-jemalloc-sys`, with 0 failed functions), whereas `haste build -o` does not yet link native static C archives (`lib*.a`, such as `tikv-jemalloc-sys` or `ring`).
- **Test harnesses and proc-macro edge cases:** `haste test` runs synchronous `fn()` and `fn() -> Result<(), E>` tests, not custom harnesses or `#[tokio::test]`, and procedural macros that rely on rustc-private span or hygiene internals can fail.
- **Platform, unwinding, and debug info caveats:** Windows runtime execution and emitted PE32+ executables are experimental (`serve`, `--daemon`, and `--watch` require Unix). `haste prep` and `--prepared` are x86-64 Linux only and compile an internal `rustc_private` driver with `RUSTC_BOOTSTRAP=1`. Standalone AOT executables emit symbol tables (`HASTE_SYMBOLS`) and DWARF `.debug_line` (`HASTE_DEBUG_INFO`), not full `.debug_info` variable/type DIEs. In `--release` builds without `catch_unwind` or threads, out-of-bounds indexing and divide-by-zero emit hardware traps instead of unwinding, and panicking across `extern "C"` frames aborts (`extern "C-unwind"` unwinds). On macOS, Hardened Runtime or App Sandbox restrictions can block dual-mapped `mach_vm_remap` JIT pages.

## Contributing

Bug reports are welcome in [issues](https://github.com/manaforged/haste/issues). Pull requests are not accepted yet. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
