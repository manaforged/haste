# haste

haste is a standalone Rust compiler, JIT engine, and zero-linker AOT emitter built around its own parser, name resolver, type/trait solver, borrow checker, SSA optimizer, register allocator, and native x86-64 and ARM64 backends—with **no LLVM** and **no `target/` directory** (reusing only `ra-ap-rustc_lexer`, `ra-ap-rustc_abi`, and `ra-ap-rustc_index` for lexing and layout primitives, alongside `serde`/`toml`/`semver`, `anyhow`, `rustc-hash`, `libc`, and `tikv-jemallocator`).

Use **haste** for fast local edit-run-test cycles, live hot-reloading, embeddable library builds, and zero-linker cross-compilation; keep `cargo check` / `rustc` in CI for full whole-crate type-checking, Polonius/HRTB lifetime verification, and production releases.

- **Fast inner loop:** Single-line edit checks in **8.4–9.0 ms** with opt-in `--daemon` (**6.2x–7.4x faster than `cargo check`**) and **0.7–2.3 ms** unchanged no-op checks—with zero `target/` directory.
- **Live hot-reloading:** `haste run --watch` watches workspace source files and patches changed function bodies directly into the running process.
- **Zero-linker 6-target AOT cross-compilation:** `haste --release build -o ./app --target TRIPLE` emits standalone ELF, Mach-O, and PE32+ executables for Linux, macOS, and Windows (x86-64 and ARM64) with no external cross-linker.
- **Spanned diagnostics and baseline safety checks:** Reports file, line, column, source snippets, and secondary labels while enforcing ownership/NLL borrow checking, non-`'static` borrow escapes, structural `Send`/`Sync` auto traits, match exhaustiveness, and module/field privacy on reached code.
- **Competitive AOT codegen:** Across five standalone AOT compute kernels (`scripts/bench.sh`, bit-identical outputs), beats or matches both `rustc -C opt-level=3` and matched-flag `rustc -C opt-level=3 -C target-cpu=native -C panic=abort` on **3 of 5** workloads on both x86-64 and ARM64 (`sha256` **0.80x–0.83x**, `sieve` **0.88x–0.91x**, `sort` **0.99x–1.05x**, `scan` **0.94x–1.09x**, `fp_iter` **1.11x–1.16x** vs matched flags).
- **Verified across layered suites:** Passes **49 unit tests** on macOS / **48** on Linux (`haste-codegen` passes/verifier/wire/targets + `haste` AOT/borrow/diagnostics/CLI), **12 integration tests** (`Compiler` API, CLI/daemon lifecycle, and soundness/borrowck), **136 differential functions** (× 9 inputs = **1,224** checks), **19 async functions** (× 9 = **171** checks), **3 proc-macro functions** (× 9 = **27** checks), and **21 standalone crate probes** (including `regex`, `clap` derive, `serde_json`, `toml`, `winnow`, and a 3-crate workspace; 20 JIT + `--release` AOT plus `so` JIT-only) cross-checked against `rustc` with `HASTE_VERIFY_PASSES=1`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the compiler pipeline and module tour.

## Quickstart (60 seconds)

Requires Rust 1.88+ with the `rust-src` component:

```bash
rustup component add rust-src
cargo build --release -p haste

# 1. Run a crate via JIT (one-shot; no target/ directory or background daemon created)
./target/release/haste run examples/hello -- --your-arg

# 2. Hot-reload edits live into a running process
./target/release/haste run --watch examples/hello

# 3. Emit a standalone native AOT executable
./target/release/haste --release build -o ./hello examples/hello

# 4. Zero-linker cross-compile a Linux x86-64 ELF from macOS (or vice versa)
./target/release/haste --release build --target x86_64-unknown-linux-gnu -o ./hello-linux examples/hello

# 5. Opt in to a resident workspace daemon for <15 ms edit checks
./target/release/haste build --daemon examples/hello
./target/release/haste serve status
./target/release/haste serve stop
```

If a workspace has no `Cargo.lock`, haste runs `cargo generate-lockfile` automatically on first use. Build scripts (`build.rs`), declarative and procedural macros, `const fn`, `async`/`await`, generics, trait objects, SIMD, and C FFI run inside haste. Settled standard-library summaries, build script outputs, macro expansions, and compiled workspace stencils stay in `~/.cache/haste`, keyed by their inputs and build configuration.

haste is experimental (`version = "0.0.0"`, `publish = false`). There is no tagged release yet, and the command line and library API can change.

## Library API

`haste` exposes a small embeddable builder API (`Compiler`, `CompilerBuilder`, `Target`, `Error`, `ErrorKind`) with explicit per-build configuration and no process-global configuration state:

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

## Commands

```text
haste [OPTIONS] build|test|run|serve|prep [PATH] [OPTIONS] [--] [ARGS...]
haste serve status [PATH]
haste serve stop [PATH | --all]
haste clean
haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand|failures PATH [CRATE]
haste [--release] debug code PATH DIR
haste -V|--version | -h|--help
```

`PATH` defaults to `.` for `build`, `test`, `run`, `serve`, and `prep`. Haste options are accepted before or after `PATH` up to `--`; everything after `--` passes verbatim to the target program's `std::env::args()` or test filter.

| Command | Does |
| --- | --- |
| `run [PATH] [--] [ARGS...]` | Runs the workspace binary or `--example` `main` with isolated user `ARGS`. |
| `test [PATH] [--] [FILTER...]` | Runs unit and integration (`tests/*.rs`) `#[test]` functions (with `#[cfg(test)]`, `[dev-dependencies]`, `#[ignore]`, `#[should_panic]`, and captured panic stderr on failure). |
| `build [-o PATH] [PATH]` | Type-checks and lowers reached functions in the workspace, or writes a standalone executable when `-o PATH` is given. |
| `serve [PATH]` | Runs a foreground resident daemon for `PATH`, rebuilding only what edited files affect. |
| `serve status [PATH]` | Lists active opt-in `haste serve` daemons (optionally filtered to `PATH`). |
| `serve stop [PATH \| --all]` | Cleanly stops active `haste serve` daemons (for `PATH` or `--all`) and removes their Unix sockets. |
| `prep [PATH]` | Pre-compiles dependencies that `main` reaches into the prepared library cache for `--prepared` (**x86-64 Linux only**). |
| `clean` | Removes the haste cache directory (`$HASTE_HOME`, `$XDG_CACHE_HOME/haste`, or `~/.cache/haste`). |
| `debug QUERY PATH [CRATE]` | Prints one compiler stage: unit graph, module outline, lazy reach, type checking, scopes, items, macro expansions, grouped failures, or emitted machine code (`code PATH DIR`). |

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
- `--daemon`: Opts in to reusing or spawning a background `haste serve` daemon on Unix (plain invocations run one-shot and leave no background process behind).
- `--target TRIPLE`: Cross-compiles a standalone AOT executable (`haste build -o`) for any supported target triple (`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`) without an external cross-linker. Cross-compiled x86-64 targets default to baseline SSE2.
- `--jobs N`: Sets the worker thread count. Defaults to the cores available to the process.
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
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` | Cache directory for build script outputs, macro expansions, standard library summaries, daemon registry, and compiled stencils. |
| `HASTE_LIBRARY` | Active `rustc --print sysroot` `rust-src` library, else newest `rustup` toolchain with `rust-src` | Path to the Rust standard library `library/` source tree. |
| `HASTE_NO_SUBSTRATE` | Unset (`0`) | Set to `1` to bypass the cached standard library and workspace substrate. |
| `HASTE_NO_NOOP` | Unset (`0`) | Set to `1` to disable the unchanged-workspace skip check on one-shot `haste build`. |
| `HASTE_X64_BASELINE` | Unset (`0`) | Set to `1` on x86-64 to disable host ISA feature detection beyond baseline SSE2. |
| `HASTE_FN_ALIGN` | `16` (loop-free) / `64` (with loops) | Uniform function alignment override in bytes (power of two, `>= 16`) for standalone AOT executables. |
| `HASTE_DEBUG_INFO` | Unset (enabled in debug builds) | Set to `1` to emit DWARF `.debug_line` in `--release` AOT executables, or `0` to disable it in debug builds. |
| `HASTE_SYMBOLS` | Unset (enabled when debug info is on) | Set to `1` to keep the symbol table in `--release` AOT executables without `.debug_line`, or `0` to strip it. |
| `HASTE_DICT_MIN_BYTES` | `512` | Minimum generic body size in bytes before dictionary lowering is considered. |
| `HASTE_DICT_MIN_FANOUT` | `8` | Minimum monomorphization fanout before sharing a dictionary body. |
| `HASTE_JIT_TIER_UP` | `64` | Call count threshold before a hot dictionary function specializes in JIT mode (`0` disables tier-up). |
| `HASTE_VERIFY_STENCIL` | Unset (`0`) | Set to `1` to verify stencil instantiations against concrete lowering. |
| `HASTE_VERIFY_PASSES` | Unset (`0`) | Set to `1` to run the full SSA dominance and CFG IR verifier after every mutating codegen pass. |
| `HASTE_TRACE` | Unset (`0`) | Function substring filter (`1` matches all) for printing codegen pass summaries. |
| `HASTE_TRACE_IR` | Unset (`0`) | Set to `1` with `HASTE_TRACE` to dump IR after each codegen pass, or `passes` for pass-by-pass summaries. |
| `HASTE_MISSED` | Unset (`0`) | Set to `1` to print aggregated codegen pass timings and stencil cache counters. |

## Benchmarks

Run `bash scripts/bench.sh` to reproduce the build-speed and AOT runtime tables on your machine. Every compute workload verifies bit-identical output across `haste --release`, stock `rustc -C opt-level=3`, and matched-flag `rustc -C opt-level=3 -C target-cpu=native -C panic=abort` (warmed 1x before timing).

### Build speed (`tests/diff`, 136 fns)

| Benchmark | Platform | haste | cargo | Speedup |
| --- | --- | ---: | ---: | ---: |
| Single-line edit (`haste build --daemon` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 8.4 ms | 62.2 ms | 7.4x |
| Single-line edit (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 66.3 ms | 62.2 ms | 0.9x |
| Single-line edit (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 161 ms | 185 ms | 1.1x |
| Unchanged no-op (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 2.3 ms | 12.1 ms | 5.3x |
| Unchanged no-op (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 2.3 ms | 12.4 ms | 5.4x |
| Cold debug (`haste build` vs `cargo check`) | macOS ARM64 (`rustc 1.98.1`) | 75.7 ms | 227 ms | 3.0x |
| Cold debug (`haste build -o` vs `cargo build`) | macOS ARM64 (`rustc 1.98.1`) | 226 ms | 508 ms | 2.3x |
| Single-line edit (`haste build --daemon` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 9.0 ms | 55.7 ms | 6.2x |
| Single-line edit (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 94.2 ms | 55.7 ms | 0.6x |
| Single-line edit (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 183 ms | 110 ms | 0.6x |
| Unchanged no-op (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 0.7 ms | 7.1 ms | 10.1x |
| Unchanged no-op (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 0.7 ms | 7.1 ms | 11.0x |
| Cold debug (`haste build` vs `cargo check`) | Linux x86-64 (`rustc 1.99.1`) | 113 ms | 202 ms | 1.8x |
| Cold debug (`haste build -o` vs `cargo build`) | Linux x86-64 (`rustc 1.99.1`) | 345 ms | 506 ms | 1.5x |

### AOT codegen runtime (`haste --release` vs `rustc`, median of 5, warmed)

#### Linux x86-64 (`rustc 1.99.1`)

| Workload | `haste --release` | `rustc -O3` | vs `-O3` | `rustc -O3 native+abort` | vs matched | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (250k blocks) | 26.3 ms | 36.1 ms | **0.73x** | 31.8 ms | **0.83x** | exact match (`7333157476866902348`) |
| `sieve` (4M primes) | 3.3 ms | 4.5 ms | **0.73x** | 3.8 ms | **0.88x** | exact match (`544501644261`) |
| `sort` (500k quicksort + bsearch) | 24.0 ms | 24.3 ms | **0.99x** | 24.4 ms | **0.99x** | exact match (`8647866321337317866`) |
| `scan` (JSON-like byte scanner) | 3.3 ms | 3.0 ms | 1.08x | 3.0 ms | 1.09x | exact match (`80005120000`) |
| `fp_iter` (f64 generic iterator) | 3.3 ms | 2.9 ms | 1.12x | 2.9 ms | 1.11x | exact match (`4685835832532186613`) |

#### macOS ARM64 (`rustc 1.98.1`)

| Workload | `haste --release` | `rustc -O3` | vs `-O3` | `rustc -O3 native+abort` | vs matched | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sha256` (250k blocks) | 46.3 ms | 54.6 ms | **0.85x** | 58.0 ms | **0.80x** | exact match (`7333157476866902348`) |
| `sieve` (4M primes) | 10.2 ms | 10.9 ms | **0.93x** | 11.2 ms | **0.91x** | exact match (`544501644261`) |
| `sort` (500k quicksort + bsearch) | 30.2 ms | 29.4 ms | 1.02x | 28.9 ms | 1.05x | exact match (`8647866321337317866`) |
| `scan` (JSON-like byte scanner) | 9.7 ms | 10.3 ms | **0.94x** | 10.3 ms | **0.94x** | exact match (`80005120000`) |
| `fp_iter` (f64 generic iterator) | 10.5 ms | 9.0 ms | 1.16x | 9.1 ms | 1.16x | exact match (`4685835832532186613`) |

## Verify it yourself

`tests/diff`, `tests/async`, and `tests/proc-macro` hold `case_*` suites (built with `--features diff-harness`, keeping default `haste` builds free of internal test fixtures), and `tests/probes` holds standalone crates. `scripts/check.sh` compares `cargo run` against both `haste test` (JIT) and `haste --release build -o` (standalone AOT).

```bash
cargo test --locked --workspace --lib                       # Unit tests (49 macOS / 48 Linux)
cargo test --locked -p haste --test compiler_api --test soundness  # Integration tests (API, CLI/daemon, soundness, borrowck)
bash scripts/check.sh                                       # Differential suite (jit+aot vs rustc)
bash scripts/check.sh tests/async                           # Async suite (jit+aot)
bash scripts/check.sh tests/proc-macro                      # Proc-macro suite (jit+aot)
bash scripts/probes.sh                                      # 21 standalone crate probes (20 jit+aot, `so` jit-only)
bash scripts/bench.sh                                       # Build speed & 5-kernel matched-flag AOT benchmarks vs rustc
bash scripts/rustc-ui.sh                                    # Official rust-lang/rust tests/ui/ conformance suite (--borrowck requires error[borrow]:)
```

`scripts/rustc-ui.sh --borrowck` requires an explicit `error[borrow]:` diagnostic to count a negative test as `PASS`, reporting non-borrow parse/type rejections separately as `OtherErr` so true borrow-checker coverage is never inflated by unrelated front-end errors.

The scripts build `haste` using `$CARGO` or `~/.config/haste/cargo` (defaulting to `cargo`); `scripts/bench.sh` invokes `cargo` directly for timed `cargo check` / `cargo build` measurements.

## Limits

- **Lazy reachability (unreached code is not checked):** haste discovers and compiles only functions reachable from the selected entry point (`main` or `#[test]`). Uncalled functions, uninstantiated generic paths, and unreachable modules are not parsed into ASTs, type-checked, or borrow-checked—run `cargo check` in CI to check every item in the workspace.
- **Self-hosting status:** haste's own workspace (`42` lockfile packages; `86` packages once merged with the Rust sysroot) reaches **182,265 functions** in ~6.3 s on macOS ARM64 (`rustc 1.98.1`), with **32 reached functions** currently failing (ARM64 floating-point register operands in `global_asm!`, `core::mem::SizedTypeProperties`, and two borrow-checker edge cases); on Linux, `tikv-jemalloc-sys`'s `build.rs` fails under JIT. Full 0-error self-hosting is in progress.
- **Full Polonius and HRTB lifetimes:** haste enforces ownership, use-after-move (across branches and loops), binding mutability, disjoint struct fields, NLL-style borrow aliasing, non-`'static` borrow escapes, structural `Send`/`Sync` auto traits, match exhaustiveness, and module/field privacy on reached functions. Full Polonius and higher-ranked lifetime (`for<'a>`) inference are not yet implemented; run `cargo check` or `rustc` in CI for complete lifetime verification.
- **Windows runtime:** Windows host runtime execution and PE32+ binaries built via `--target x86_64-pc-windows-msvc` / `aarch64-pc-windows-msvc` are experimental, and `serve`/`--watch` require Unix.
- **`catch_unwind` and `Drop` unwinding:** In debug builds, `haste test`, and any build using `std::panic::catch_unwind` or threads, out-of-bounds indexing and divide-by-zero panic with a message and run `Drop` destructors in both JIT and standalone AOT (x86-64 and ARM64); in `--release` builds without `catch_unwind`, bounds checks and divide-by-zero emit hardware traps for zero-overhead codegen, uncaught panics print their message and abort, and unwinding across C FFI frames is not supported.
- **Full `.debug_info`:** Standalone AOT executables emit symbol tables (`HASTE_SYMBOLS`) and DWARF `.debug_line` tables (`HASTE_DEBUG_INFO`) for backtraces and profiling, not full `.debug_info` variable/type DIEs.
- **Test harnesses:** `haste test` runs synchronous `fn()` and `fn() -> Result<(), E>` (`Termination`) `#[test]` functions (not `#[tokio::test]` macro suites that depend on custom harnesses).
- **Prepared mode:** `haste prep` and `--prepared` are x86-64 Linux only and compile an internal `rustc_private` driver with `RUSTC_BOOTSTRAP=1`, which is tied to the installed `rustc` version.
- **macOS Hardened Runtime:** On macOS, JIT execution uses dual-mapped `mach_vm_remap` pages (writable + executable aliases), which may be blocked inside a Hardened Runtime or App Sandbox.
- **Feature coverage:** Some Rust features and crates.io edge cases do not compile yet. If any reached function fails to compile, `haste build` exits non-zero.

## References and design influences

haste builds its own frontend, optimizer, register allocator, and native machine-code backends on established compiler algorithms from the literature (reusing `ra-ap-rustc_{lexer,abi,index}` for Rust lexing and ABI layout primitives).

### Established techniques and prior art

- **Iterative dominance (`crates/codegen/src/cfg.rs`):** Cooper, Harvey, & Kennedy (2001), *"A Simple, Fast Dominance Algorithm"*.
- **Block-parameter SSA and on-demand promotion (`crates/codegen/src/ir.rs`, `promote.rs`):** Block-parameter SSA form (Kelsey 1995; Appel 1998) with demand-driven SSA construction in the spirit of Braun, Buchwald, Hack, Leißa, Mallon, & Zwinkau (2013), *"Simple and Efficient Construction of Static Single Assignment Form"* (CC 2013).
- **Linear scan register allocation and out-of-SSA copy resolution (`crates/codegen/src/park.rs`, `alloc.rs`, `frame.rs`):** Poletto & Sarkar (1999), *"Linear Scan Register Allocation"* (TOPLAS); Wimmer & Franz (2010), *"Linear Scan Register Allocation on SSA Form"* (CGO 2010); Boissinot, Darte, Rastello, Dupont de Dinechin, & Guillon (2009), *"Revisiting Out-of-SSA Translation for Correctness, Code Quality, and Efficiency"* (CGO 2009).
- **Shrink-wrapping (`crates/codegen/src/frame.rs`):** Fred C. Chow (1988), *"Minimizing Register Usage Penalty at Procedure Calls"* (PLDI 1988).
- **Division by invariant integers via multiplication (`crates/codegen/src/algebra.rs`):** Torbjörn Granlund & Peter L. Montgomery (1994), *"Division by Invariant Integers using Multiplication"* (PLDI 1994).
- **Superword-level parallelism (`crates/codegen/src/slp.rs`):** Samuel Larsen & Saman Amarasinghe (2000), *"Exploiting Superword Level Parallelism with Multimedia Instruction Sets"* (PLDI 2000).
- **Dictionary passing and stencil instantiation (`crates/haste/src/lower/stencil.rs`, `image/stencil_jit.rs`):** Hall, Hammond, Peyton Jones, & Wadler (1996), *"Type Classes in Haskell"* (TOPLAS); Xu & Kjolstad (2021), *"Copy-and-Patch Compilation"* (OOPSLA 2021); Swift's hybrid dictionary/specialization model.
- **Identical Code Folding (`crates/haste/src/image/aot/icf.rs`):** Tallam, Coutant, Taylor, Li, & Demetriou (2010), *"Safe ICF: Pointer Safe and Unwinding Aware Identical Code Folding in the Gold Linker"* (GCC Developers Summit 2010).
- **Non-Lexical Lifetimes (`crates/haste/src/sema/borrow.rs`):** Rust RFC 2094 (*Non-Lexical Lifetimes*).
- **Match exhaustiveness (`crates/haste/src/sema/infer/matching/`):** Luc Maranget (2007), *"Warnings for Pattern Matching"* (JFP 2007).

### Distinctive architectural techniques in haste

- **Whole-workspace wave-based lazy monomorphization and frozen sysroot substrate (`sema/reach/`, `substrate/sysroot.rs`):** Parallel `cfg`-pruning outline scans, a binary-serialized settled sysroot cache (`HSTSYS01`) that eliminates stdlib re-parsing on cold CLI invocations, and wave-based reachability keyed by 16-byte `Instance = (ItemRef, Substs)` so unreached stdlib and dependency bodies are never parsed into ASTs or type-checked—with no `.rlib` artifacts and no `target/` directory.
- **Structural 128-bit `FootprintId` generic stencils with live JIT tier-up (`lower/stencil.rs`, `image/stencil_jit.rs`):** Hashes each generic body's lowered layout, ABI, inlining, and unwind footprint into a 128-bit digest to share compiled stencils or dictionary bodies across monomorphizations with identical machine-level footprints (cross-checked via `HASTE_VERIFY_STENCIL=1`), while hot-tiering hot dictionary instances to concrete specialized machine code in JIT after `HASTE_JIT_TIER_UP` calls.
- **Call-site register parking paired with loop-header live-range splitting (`park.rs`, `loopargs.rs`):** Balances keeping call-crossing values in caller-saved registers with targeted spill/reload parking around crossed calls against function-wide callee-saved pressure, paired with hoisting loop-invariant reads into explicit loop-header block parameters so inner loops allocate registers independently of outer call pressure.
- **Domain-dirty fixed-point pass scheduler and zero-linker 6-target native emitters (`schedule.rs`, `aot/`):** A single lightweight Rust codegen crate switching between `QUICK_RULES` (JIT/edit loop) and `FULL_RULES` (`--release`) via 3-bit `Domain` (`CFG | MEM | VALUES`) dirty tracking, emitting standalone ELF, Mach-O (with native `__thread_vars` TLV descriptors and per-thread lock-free frame caches), and PE32+ binaries directly with no external linker or LLVM.

## Contributing

Bug reports are welcome as issues. Pull requests are not accepted yet.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
