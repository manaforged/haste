# haste

haste is a standalone Rust compiler, in-memory JIT runner, and zero-linker AOT builder with no LLVM in the compiler binary and no per-project `target/` directory (cached artifacts live under `~/.cache/haste`).

- **Sub-10 ms incremental edit checks:** `8.4–9.0 ms` on single-line edits with `--daemon` (`6.2x–7.4x` faster than one-shot `cargo check`); unchanged no-ops exit in `0.7–2.3 ms`.
- **In-memory execution, tests, and live hot-reload (Unix):** run binaries (`haste run`), unit/integration and `#[tokio::test]` suites (`haste test`), or patch edited functions into a running process (`haste run --watch`).
- **Zero-linker cross-compilation:** emit native ELF, Mach-O, and PE32+ executables directly (`haste --release build --target ... -o`), with `--release` compute kernels matching or beating `rustc -O3` (`0.68x–1.05x`).
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

If a workspace has no `Cargo.lock`, `haste` runs `cargo generate-lockfile` automatically. If locked registry or git crates are missing locally, `haste` runs `cargo fetch --locked` automatically unless `CARGO_NET_OFFLINE=true` or `[net] offline = true` is set. Dependency lookup matches `Cargo.lock` `source` URLs against `.cargo/config.toml` `[source.*]` / `replace-with` entries, honors `cargo vendor` offline, and verifies `.cargo-checksum.json` package hashes when present.

### Incremental daemon

Plain `haste` commands never spawn a background process, though they reuse an already-running daemon for the same workspace unless `HASTE_NO_DAEMON=1` is set. Opt in to a background daemon on Unix with `--daemon` for `8–9 ms` incremental edit checks (120 s idle timeout, configurable via `HASTE_SERVE_IDLE_SECS`), and inspect or stop daemons with `serve status` and `serve stop` (`PATH` is optional):

```bash
haste build --daemon haste-demo
haste serve status haste-demo
haste serve stop haste-demo
```

### Live hot-reload

Run `haste run --watch <path>` on a long-running Unix process (such as a loop or server). When you save a file, `haste` recompiles only the affected functions and patches them in memory on their next call.

### Zero-linker cross-compilation

Emit a Linux x86-64 ELF binary from macOS (or macOS/Windows executables from Linux) without installing a cross-linker:

```bash
haste --release build --target x86_64-unknown-linux-gnu -o ./haste-demo-linux haste-demo
```

## Benchmarks

Run `bash scripts/bench.sh` to reproduce build-speed and AOT runtime numbers on your machine (pass `--long` for 4x compute iterations).

### Build speed (`tests/diff`, warm toolchain cache)

Resident-daemon edits (`--daemon`) compare a warm background `haste serve` process against one-shot `cargo check`; one-shot `haste build` pays process startup (`0.6x–0.9x` on small edits), while unchanged no-ops exit in `0.7–2.3 ms` (`5.3x–11.0x`).

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

Compute benchmarks subtract each compiler's warmed empty-main startup baseline (median of 7) and verify bit-identical outputs across `haste --release`, stock `rustc -C opt-level=3`, and matched-flag `rustc -C opt-level=3 -C target-cpu=native -C panic=abort`.

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

- `scripts/rustc-ui.sh --borrowck`: **292 / 344 (84.9%)** emit `error[borrow]:` (**276 / 344**, `80.2%`, match the exact `//~` error line).
- `scripts/rustc-ui.sh --all` (`run-pass`): **3,820 / 4,168 (91.7%)** pass.

## Crate compatibility

`scripts/probes.sh` cross-checks real crates against `rustc` in both JIT and `--release` AOT modes:

| Crate or workload | Status | Notes |
| --- | --- | --- |
| Pure-Rust multi-crate workspaces (`[workspace.dependencies]`, path crates) | Supported (JIT + AOT) | `tests/probes/ws` |
| `serde` (`derive`) + `serde_json` (`raw_value`), `toml` (`0.8`) | Supported (JIT + AOT) | `tests/probes/sj`, `tests/probes/fp`, `tests/probes/ws` |
| `clap` (`4.x`, `derive`), `regex` (`1.x`), `winnow` (`0.7`) | Supported (JIT + AOT) | `tests/probes/cd`, `tests/probes/rx`, `tests/probes/wn` |
| `#[tokio::test]` (`current_thread`, `multi_thread`, `LocalSet`, `start_paused`) | Supported (JIT) | Verified on 9-test Tokio suite; `harness = false` unsupported |
| Procedural macros (`derive`, attribute, function-like) | Supported (JIT + AOT) | `tests/proc-macro`, `tests/probes/pc` |
| `rusqlite` (`0.32`, `bundled` C `libsqlite3.a`) | Supported (JIT + native Unix AOT) | Live-verified with `build.rs` static archive |
| `prost` (`0.13`), `uuid` (`1.x`), `chrono` (`0.4`) | Supported (JIT + AOT) | Live-verified |
| Crates bundling static C archives (`lib*.a`, e.g. `rusqlite` `bundled`, `tikv-jemalloc-sys`) | JIT + native Unix AOT (`build -o`); cross-target AOT errors | See [Limits](#limits) |
| `reqwest` / `openssl-sys` | Untested | Not covered in `tests/probes` |
| Nightly `#![feature(...)]` crates | Unsupported | Stable syntax and stdlib only |

## Embedding haste as a library

`haste` exposes an embeddable API (`Compiler`, `CompilerBuilder`, `Target`, `Error`, `ErrorKind`) with explicit per-build configuration and no process-global state. Add it as a git dependency in `Cargo.toml` (`haste = { git = "https://github.com/manaforged/haste" }`, not from crates.io):

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

- **Incomplete borrow-checker and nightly coverage:** Full Polonius borrow checking, some upstream `rust-lang/rust` `tests/ui/borrowck` and higher-ranked (`for<'a>`) lifetime edge cases, and nightly `#![feature(...)]` attributes are not yet supported. Keep `cargo check` or `rustc` in CI as your merge and release gate.
- **External dependencies stay lazy:** `haste` type-checks and borrow-checks every concrete and generic function in the selected member crate before running or emitting code, while external crates and `std` are checked only along reached paths.
- **Static C archives in standalone AOT:** `haste build .` checks `haste`'s own 42-package workspace (including `tikv-jemalloc-sys`, 0 failed functions), and both JIT and native Unix `haste build -o` link `build.rs` static C archives (`lib*.a`, such as `rusqlite` `bundled`) via a cached shared bundle. Cross-target `haste build -o --target <TRIPLE>` rejects static C archives rather than emitting an incomplete binary.
- **Test harnesses and proc-macro edge cases:** `haste test` runs `#[test]` and `#[tokio::test]` functions (`fn()` and `fn() -> Result<(), E>`), not custom `harness = false` runners, and procedural macros relying on rustc-private span or hygiene internals can fail.
- **Platform support:** Supported hosts and AOT `--target` triples are `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`, and `aarch64-pc-windows-msvc`. Windows PE32+ emission is experimental; JIT `run`/`test`, `serve`, `--daemon`, and `--watch` require Unix; `prep` / `--prepared` are Linux x86-64 only (using `RUSTC_BOOTSTRAP=1`). On macOS, Hardened Runtime or App Sandbox restrictions can block dual-mapped `mach_vm_remap` JIT pages.
- **Debug info, unwinding, and traps:** AOT binaries emit symbol tables (`HASTE_SYMBOLS`) and DWARF `.debug_line` (`HASTE_DEBUG_INFO`), not full `.debug_info` variable/type DIEs. In `--release` builds without `catch_unwind` or threads, out-of-bounds indexing and divide-by-zero emit hardware traps instead of unwinding; panicking across `extern "C"` aborts (`extern "C-unwind"` unwinds).

## Commands, options, and environment

```text
haste [OPTIONS] build|test|run|serve|prep [PATH] [OPTIONS] [--] [ARGS...]
haste serve status [PATH]
haste serve stop [PATH | --all]
haste clean
haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand|failures PATH [CRATE]
haste [--release] debug code PATH DIR
haste -V|--version | -h|--help
```

`PATH` defaults to `.`. Options are accepted before or after `PATH` up to `--` (place options before `debug` for `haste debug`); everything after `--` passes verbatim to the target program's `std::env::args()` or test filter. Run `haste <command> --help` for the full per-command flag list.

Before running or emitting code, `haste` type-checks and borrow-checks the full selected member crate (including uncalled and generic functions) with multi-label diagnostics.

| Command | Does |
| --- | --- |
| `run [PATH] [--] [ARGS...]` | Checks the member crate, then runs the binary or `--example` `main` via JIT (Unix) with isolated `ARGS`; pass `--watch` for live hot-reload. |
| `test [PATH] [--] [FILTER...]` | Checks the member crates, then runs `#[test]` and `#[tokio::test]` functions (unit and `tests/*.rs`, `#[ignore]`, `#[should_panic]`, captured panic stderr). |
| `build [-o PATH] [PATH]` | Checks the member crate(s) and reached code, or emits a standalone native executable when `-o PATH` is given. |
| `serve [PATH]` | Runs a foreground resident daemon for `PATH`; use `serve status [PATH]` and `serve stop [PATH \| --all]` to inspect or stop daemons. |
| `prep [PATH]` | Pre-compiles dependencies for `--prepared` (Linux x86-64 only). |
| `clean` | Removes the local haste cache directory (`$HASTE_HOME`, `$XDG_CACHE_HOME/haste`, or `~/.cache/haste`). |
| `debug QUERY PATH [CRATE]` | Prints one compiler stage or dumps machine code (`debug code PATH DIR`). |

### Environment variables

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
