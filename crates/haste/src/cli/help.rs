use super::args::Command;
use crate::serve::Request;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HelpTopic {
    Build,
    Run,
    Test,
    Serve,
    Prep,
    Clean,
    Debug,
}

impl HelpTopic {
    pub(super) fn from_command(cmd: Command) -> Self {
        match cmd {
            Command::Request(Request::Build) => Self::Build,
            Command::Request(Request::Run) => Self::Run,
            Command::Request(Request::Test) => Self::Test,
            Command::Serve(_) => Self::Serve,
            Command::Prep => Self::Prep,
            Command::Clean => Self::Clean,
            Command::Debug(_) => Self::Debug,
        }
    }
}

pub(super) fn version_string() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let sha = option_env!("HASTE_GIT_SHA").unwrap_or("");
    if sha.is_empty() {
        format!("haste {version}")
    } else {
        format!("haste {version} ({sha})")
    }
}

pub(super) fn help_text(topic: Option<HelpTopic>) -> &'static str {
    match topic {
        None => GLOBAL_HELP,
        Some(HelpTopic::Build) => BUILD_HELP,
        Some(HelpTopic::Run) => RUN_HELP,
        Some(HelpTopic::Test) => TEST_HELP,
        Some(HelpTopic::Serve) => SERVE_HELP,
        Some(HelpTopic::Prep) => PREP_HELP,
        Some(HelpTopic::Clean) => CLEAN_HELP,
        Some(HelpTopic::Debug) => DEBUG_HELP,
    }
}

macro_rules! target_options {
    () => {
        "  -p, --package <PKG>       Select a package in the workspace
      --bin <NAME>          Select a binary target
      --example <NAME>      Select an example target
  -F, --features <LIST>     Space- or comma-separated list of features to activate
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers"
    };
}

const GLOBAL_HELP: &str = concat!(
    "Standalone Rust compiler, JIT runner, and zero-linker AOT builder.

Usage: haste [OPTIONS] <COMMAND> [PATH] [-- ARGS...]
       (PATH defaults to `.`)

Commands:
  build   Type-check, borrow-check, and optionally emit a native executable (-o)
  run     Compile and run a binary or example target in memory (Unix JIT)
  test    Compile and run #[test] and #[tokio::test] suites (Unix JIT)
  serve   Run a foreground resident compiler daemon or inspect/stop daemons
  prep    Precompile dependencies with rustc for --prepared (Linux x86_64 only)
  clean   Remove the local haste cache directory
  debug   Inspect the workspace graph, outline, types, expansions, or machine code

Common options:
",
    target_options!(),
    "
      --release             Build with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat runs (Unix)
  -h, --help                Print help (use `haste <command> --help` for details)
  -V, --version             Print version and commit",
);

const BUILD_HELP: &str = concat!(
    "Usage: haste build [OPTIONS] [PATH]

Type-check, borrow-check, and compile a crate or workspace.
Pass -o PATH to emit a standalone native executable.

Options:
  -o, --output <PATH>       Emit a standalone native executable at PATH
      --target <TRIPLE>     Cross-compile target triple (requires -o)
      --release             Build with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat edits (Unix)
",
    target_options!(),
    "
      --stats-json <PATH>   Write compilation timing and code-size stats as JSON
  -h, --help                Print help",
);

const RUN_HELP: &str = concat!(
    "Usage: haste run [OPTIONS] [PATH] [-- ARGS...]

Check the crate, then compile and execute its binary or example target in memory (Unix).
Arguments after `--` are passed to the program as argv.

Options:
      --release             Run with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat runs (Unix)
      --watch               Hot-reload changed functions into the running process (Unix)
      --prepared            Use precompiled rustc dependency code (Linux x86_64 only)
      --prep-wait <MS>      Wait up to MS for background prep (Linux x86_64 only)
      --perf-map            Write /tmp/perf-<pid>.map symbols for JIT profiling
",
    target_options!(),
    "
  -h, --help                Print help",
);

const TEST_HELP: &str = "\
Usage: haste test [OPTIONS] [PATH] [-- FILTER...]

Check the crate, then compile and run its #[test] and #[tokio::test] suite (Unix).
Filter arguments (such as substring patterns, --exact, --skip NAME) follow `--`.

Options:
      --ignored             Run only tests marked #[ignore]
      --include-ignored     Run both normal and #[ignore] tests
      --release             Run tests with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat test runs (Unix)
  -p, --package <PKG>       Select a package in the workspace
  -F, --features <LIST>     Space- or comma-separated list of features to activate
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";

const SERVE_HELP: &str = concat!(
    "Usage: haste serve [OPTIONS] [PATH]
       haste serve status [PATH]
       haste serve stop [PATH | --all]

Run a foreground resident compiler daemon for fast repeat edits, or inspect/stop running daemons (Unix).
Pass `--daemon` to `build`, `run`, or `test` to spawn one in the background.

Subcommands:
  status [PATH]             List running daemons (optionally filtered by workspace path)
  stop [PATH | --all]       Stop the daemon for PATH (default `.`) or all daemons (--all)

Options:
      --all                 With `serve stop`, stop every running haste daemon
      --release             Serve release-profile builds
      --prepared            Use precompiled dependency code (Linux x86_64 only)
",
    target_options!(),
    "
  -h, --help                Print help",
);

const PREP_HELP: &str = concat!(
    "Usage: haste prep [OPTIONS] [PATH]

Precompile workspace dependencies with rustc for `haste --release --prepared run` (Linux x86_64 only).

Options:
",
    target_options!(),
    "
  -h, --help                Print help",
);

const CLEAN_HELP: &str = "\
Usage: haste clean

Remove the local haste cache directory ($HASTE_HOME or platform cache dir).

Options:
  -h, --help                Print help";

const DEBUG_HELP: &str = "\
Usage: haste debug <QUERY> <PATH> [CRATE]
       haste [--release] debug code <PATH> <DIR>

Inspect internal compiler phases for a workspace or single file.

Queries:
  graph     Print resolved package and unit dependency graph
  outline   Print crate module tree and outline counts
  lazy      Inspect lazy crate loading
  typeck    Run type-checking diagnostics
  scope     Inspect name resolution scopes
  items     Print parsed items of a source file
  expand    Print macro-expanded source of a file
  failures  Group and print compilation failures
  code      Dump lowered/compiled machine code into DIR

Options:
      --release             Compile release code (with `debug code`)
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";
