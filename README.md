# haste

haste is a clean-room Rust compiler and execution engine with its own frontend,
SSA optimizer, and native x86-64 and Apple Silicon arm64 code generator.

It reads `Cargo.lock`, resolves names and types directly from source, and
either JIT-compiles reached functions on first execution (`haste run`) or emits
a standalone native executable (`haste --release build -o`) without LLVM or a
`target/` directory.

```bash
cargo build --release -p haste
./target/release/haste --version
./target/release/haste run examples/hello
./target/release/haste --release build -o ./hello examples/hello
```

If a workspace has no `Cargo.lock`, haste runs `cargo generate-lockfile`
automatically on first use. Build scripts (`build.rs`), declarative and
procedural macros, `const fn`, `async`/`await`, generics, trait objects, SIMD,
and C FFI all run inside haste. Unchanged build script outputs, macro
expansions, standard library summaries, and compiled workspace stencils stay in
`~/.cache/haste`, keyed by their inputs, so repeated builds rebuild only edited
items and skip unchanged workspaces.

haste is experimental. There is no tagged release yet, and the command line can change.

## Commands

```text
haste [-p PKG] [--bin NAME | --example NAME] [-F FEATURES] [--all-features] [--no-default-features] [--jobs N] [--release] [--prepared] [--prep-wait MS] [--perf-map] [--watch] [--stats-json PATH] [-o PATH] build|test|run|serve|prep [PATH] [--] [ARGS...]
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
- `--jobs N`: Sets the worker thread or process count. Defaults to the cores available to the process.
- `--release`: Enables full codegen optimization and release `cfg` settings.
- `-o PATH`, `--output PATH`: Writes a standalone native executable (`haste build` only).
- `--watch`: Watches workspace source files and hot-reloads changed function bodies into the running process (`haste run` only).
- `--stats-json PATH`: Writes stage timing and reachability statistics as JSON (`haste build` only).
- `--prepared`: Links pre-compiled dependency code produced by `haste prep` (**x86-64 Linux only**).
- `--prep-wait MS`: Waits up to `MS` milliseconds for a running `prep` job before compiling directly (**x86-64 Linux only**).
- `--perf-map`: Writes `/tmp/perf-<pid>.map` symbol entries for JIT profiling.

## Requirements and environment

- Linux on x86-64 (`x86_64-unknown-linux-gnu`) or macOS on Apple Silicon (`aarch64-apple-darwin`).
- Rust 1.88 or newer with the `rust-src` component installed (`rustup component add rust-src`).
- On macOS, `codesign` (included with macOS / Xcode Command Line Tools) for ad-hoc signing standalone AOT executables.
- A C compiler (`$CC`, defaulting to `cc`) when building workspaces that link native C libraries.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` | Cache directory for build script outputs, macro expansions, standard library summaries, and compiled stencils. |
| `HASTE_LIBRARY` | Active `rustc --print sysroot` `rust-src` library, else newest `rustup` toolchain with `rust-src` | Path to the Rust standard library `library/` source tree. |
| `HASTE_NO_SUBSTRATE` | Unset (`0`) | Set to `1` to bypass the cached standard library and workspace substrate. |
| `HASTE_NO_NOOP` | Unset (`0`) | Set to `1` to disable the unchanged-workspace skip check on one-shot `haste build`. |
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

## Check against rustc

`tests/diff`, `tests/async`, and `tests/proc-macro` hold `case_*` suites, and
`tests/probes` holds standalone crates. `scripts/check.sh` compares `cargo run`
against both `haste test` (JIT) and `haste --release build -o` (standalone AOT).

```bash
bash scripts/check.sh
bash scripts/check.sh tests/async
bash scripts/check.sh tests/proc-macro
bash scripts/probes.sh
```

The scripts run `cargo`, or the command in `$CARGO` or `~/.config/haste/cargo`.

## Limits

- haste compiles only for the host (`x86_64-unknown-linux-gnu` or `aarch64-apple-darwin`). Windows and cross-compilation are not supported.
- haste performs name resolution, type inference, trait selection, and move/drop tracking, but does not run a borrow checker yet. Use `cargo check` or `rustc` in CI to verify lifetime and aliasing rules.
- `haste test` currently runs synchronous `fn()` `#[test]` functions (not `Result`-returning or `#[tokio::test]` functions yet).
- `haste prep` and `--prepared` are x86-64 Linux only and compile an internal `rustc_private` driver with `RUSTC_BOOTSTRAP=1`, which is tied to the installed `rustc` version.
- On macOS, JIT execution uses dual-mapped `mach_vm_remap` pages (writable + executable aliases), which may be blocked inside a Hardened Runtime or App Sandbox.
- Some Rust features do not compile yet. If any reached function fails to compile, `haste build` exits non-zero.
- Panics print their message and abort instead of unwinding the stack.

## Contributing

Bug reports are welcome as issues. Pull requests are not accepted yet.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
