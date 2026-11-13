# haste

haste is a Rust compiler and execution engine for running, testing, and
building workspaces without a `target/` build directory.

You run a crate directly from source, or emit a standalone native executable
when you want a binary on disk. haste reads `Cargo.lock`, resolves names and
types itself, and compiles reached functions to x86-64 or Apple Silicon arm64
machine code.

```bash
cargo build --release -p haste
./target/release/haste run examples/hello
./target/release/haste --release build -o /tmp/hello examples/hello
```

If a workspace has no `Cargo.lock`, run `cargo generate-lockfile` in it first.
`haste run` compiles functions the first time execution reaches them, and build
scripts and procedural macros run inside haste. Unchanged build script outputs,
macro expansions, and standard library summaries stay in `~/.cache/haste`,
keyed by their inputs.

haste is experimental. There is no release yet, and the command line can change.

## Commands

```text
haste [--jobs N] [--release] [--prepared] [--prep-wait MS] [--perf-map] [--stats-json PATH] [-o PATH] build|test|run|serve|prep PATH [--] [ARGS...]
haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand|failures PATH [CRATE]
haste [--release] debug code PATH DIR
```

| Command | Does |
| --- | --- |
| `run PATH [--] [ARGS...]` | Runs the workspace binary's `main` with `ARGS`. |
| `test PATH` | Runs each `fn(i64) -> i64` `case_*` function in the workspace members. |
| `build [-o PATH] PATH` | Type-checks the workspace and lowers reached code, or writes a standalone executable when `-o PATH` is given. |
| `serve PATH` | Keeps the workspace loaded in memory and answers `build`, `test`, and `run` requests for `PATH`, rebuilding only what edited files affect. |
| `prep PATH` | Pre-compiles dependencies that `main` reaches into the prepared library cache for `--prepared`. |
| `debug QUERY PATH [CRATE]` | Prints one compiler stage: unit graph, module outline, lazy reach, type checking, scopes, items, macro expansions, failures, or emitted machine code (`code PATH DIR`). |

### Options

- `--jobs N`: Sets the worker thread or process count. Defaults to the cores available to the process.
- `--release`: Enables full codegen optimization and release `cfg` settings.
- `-o PATH`, `--output PATH`: Writes a standalone native executable (`haste build` only).
- `--stats-json PATH`: Writes stage timing and reachability statistics as JSON (`haste build` only).
- `--prepared`: Links pre-compiled dependency code produced by `haste prep`.
- `--prep-wait MS`: Waits up to `MS` milliseconds for a running `prep` job before compiling directly.
- `--perf-map`: Writes `/tmp/perf-<pid>.map` symbol entries for JIT profiling.

## Requirements and environment

- Linux on x86-64 (`x86_64-unknown-linux-gnu`) or macOS on Apple Silicon (`aarch64-apple-darwin`).
- A stable Rust toolchain from `rustup` with the `rust-src` component installed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` | Cache directory for build script outputs, macro expansions, and standard library summaries. |
| `HASTE_LIBRARY` | Newest `stable-*` `rust-src` `library` directory under `$RUSTUP_HOME` | Path to the Rust standard library `library/` source tree. |
| `HASTE_NO_SUBSTRATE` | Unset (`0`) | Set to `1` to bypass the cached standard library substrate. |
| `HASTE_X64_BASELINE` | Unset (`0`) | Set to `1` on x86-64 to disable host ISA feature detection beyond baseline SSE2. |
| `HASTE_DICT_MIN_BYTES` | `2048` | Minimum generic body size in bytes before dictionary lowering is considered. |
| `HASTE_DICT_MIN_FANOUT` | `2` (JIT) or `8` (AOT) | Minimum monomorphization fanout before sharing a dictionary body. |
| `HASTE_JIT_TIER_UP` | `64` | Call count threshold before a hot dictionary function specializes in JIT mode (`0` disables tier-up). |
| `HASTE_VERIFY_STENCIL` | Unset (`0`) | Set to `1` to verify stencil instantiations against concrete lowering. |

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
- Some Rust features do not compile yet. If any reached function fails to compile, `haste build` exits non-zero.
- Panics print their message and abort instead of unwinding the stack.

## Contributing

Bug reports are welcome as issues. Pull requests are not accepted yet.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
