# haste

haste runs a Rust project without building it first. It reads the
workspace from `Cargo.lock`, resolves names and types itself, and
compiles each function to x86-64 machine code the first time the program
calls it. Nothing is written to `target/`.

```bash
cargo build --release -p haste
./target/release/haste run examples/hello
```

The project needs a `Cargo.lock`; run `cargo generate-lockfile` in it
first if it has none. `haste run` starts `main`. Functions that the run
never reaches are never compiled, and build scripts and procedural macros run inside haste.
Results that do not change between runs, such as build script output and
macro expansions, are kept in `~/.cache/haste`, keyed by their inputs.

Status: experimental. No release yet, and the command line can change.

## Commands

```text
haste [--jobs N] build|test|run|serve PATH [ARGS...]
haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand PATH [CRATE]
```

| Command | Does |
| --- | --- |
| `run PATH [ARGS...]` | Runs the workspace binary's `main` with `ARGS`. |
| `test PATH` | Runs each `case_*` function of the workspace members. |
| `build PATH` | Type-checks and compiles every function the binary reaches. |
| `serve PATH` | Keeps the workspace loaded and answers `build`, `test` and `run` for `PATH` from other haste processes. An edited file is read again and only what it changes is rebuilt. |
| `debug QUERY PATH` | Prints what one stage found: the unit graph, the module outline, scopes, items, macro expansions or type errors. |

`--jobs N` sets how many processes or threads haste uses. The default is
the number of cores the process may run on.

## Requirements

- Linux on x86-64 with glibc.
- A stable Rust toolchain from rustup with the `rust-src` component.
  haste reads the standard library from that source tree. Set
  `HASTE_LIBRARY` to point at another `library` directory.

| Variable | Default | Use |
| --- | --- | --- |
| `HASTE_HOME` | `$XDG_CACHE_HOME/haste`, else `~/.cache/haste` | The store for build script results and macro expansions. |
| `HASTE_LIBRARY` | The newest `stable-*` toolchain under `$RUSTUP_HOME` | The standard library source. |

## Check it against rustc

`tests/diff` holds a crate of `case_*` functions. `scripts/check.sh` runs
it once with cargo and once with `haste test`, and fails when an output
differs.

```bash
bash scripts/check.sh
bash scripts/check.sh tests/async
bash scripts/check.sh tests/proc-macro
bash scripts/probes.sh
```

The scripts run `cargo`, or the command in `$CARGO` or in
`~/.config/haste/cargo`.

## Limits

- Only the `x86_64-unknown-linux-gnu` target. There is no macOS, Windows
  or ARM support.
- Some Rust code does not compile yet. 400 of the 425 checks in
  `tests/diff` agree with rustc; haste gives no result for the other 25.
  128-bit division and float-to-`i128` casts are among the gaps.
- `haste build` does not write a binary.
- The first `serve` load of a large workspace takes about a minute.

## Contributing

Bug reports are welcome as issues. Pull requests are not accepted yet.

## License

MIT. See [LICENSE-MIT](LICENSE-MIT).
