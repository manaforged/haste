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

const GLOBAL_HELP: &str = "\
Usage: haste [OPTIONS] <COMMAND> [PATH] [-- ARGS...]

Commands:
  build   Type-check, borrow-check, and optionally emit a native executable (-o)
  run     Compile and run a binary or example target directly in memory
  test    Compile and run #[test] functions in the workspace
  serve   Run or manage a background compiler daemon for fast repeat edits
  prep    Precompile dependencies with rustc for --prepared release runs
  clean   Remove the local haste cache directory
  debug   Inspect the workspace graph, outline, types, expansions, or machine code

Common options:
  -p, --package <PKG>       Select a package in the workspace
      --bin <NAME>          Select a binary target
      --example <NAME>      Select an example target
  -F, --features <LIST>     Space or comma separated list of features to activate
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --release             Build with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat runs
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help (use `haste <command> --help` for details)
  -V, --version             Print version and commit";

const BUILD_HELP: &str = "\
Usage: haste build [OPTIONS] [PATH]

Type-check, borrow-check, and compile a crate or workspace. Pass -o PATH to emit a standalone executable.

Options:
  -o, --output <PATH>       Emit a standalone native executable at PATH
      --target <TRIPLE>     Cross-compile target triple (requires -o)
      --release             Build with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat edits
  -p, --package <PKG>       Select a package in the workspace
      --bin <NAME>          Select a binary target
      --example <NAME>      Select an example target
  -F, --features <LIST>     Activate specific features
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers
      --stats-json <PATH>   Write compilation timing and code-size stats as JSON
  -h, --help                Print help";

const RUN_HELP: &str = "\
Usage: haste run [OPTIONS] [PATH] [-- ARGS...]

Check the crate, then compile and execute its binary or example target in memory.
Arguments after `--` are passed to the program as argv.

Options:
      --release             Run with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat runs
      --watch               Re-run automatically when source files change
      --prepared            Use precompiled rustc dependency code in --release mode
      --prep-wait <MS>      Milliseconds to wait for background prep before running
      --perf-map            Write /tmp/perf-<pid>.map symbols for JIT profiling
  -p, --package <PKG>       Select a package in the workspace
      --bin <NAME>          Select a binary target
      --example <NAME>      Select an example target
  -F, --features <LIST>     Activate specific features
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";

const TEST_HELP: &str = "\
Usage: haste test [OPTIONS] [PATH] [-- FILTER...]

Check the crate, then compile and run its #[test] suite.
Filter arguments (such as substring patterns, --exact, --skip NAME) follow `--`.

Options:
      --ignored             Run only tests marked #[ignore]
      --include-ignored     Run both normal and #[ignore] tests
      --release             Run tests with optimizations enabled
      --daemon              Start or reuse a background daemon for repeat test runs
  -p, --package <PKG>       Select a package in the workspace
  -F, --features <LIST>     Activate specific features
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";

const SERVE_HELP: &str = "\
Usage: haste serve [OPTIONS] [PATH]
       haste serve status [PATH]
       haste serve stop [PATH | --all]

Run a persistent background compiler daemon for fast incremental edits, or inspect/stop running daemons.

Subcommands:
  status [PATH]             List running daemons (optionally filtered by workspace path)
  stop [PATH | --all]       Stop the daemon for PATH (default `.`) or all daemons (--all)

Options:
      --all                 With `serve stop`, stop every running haste daemon
      --release             Serve release-profile builds
      --prepared            Use precompiled dependency code in --release mode
  -p, --package <PKG>       Select a package in the workspace
      --bin <NAME>          Select a binary target
      --example <NAME>      Select an example target
  -F, --features <LIST>     Activate specific features
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";

const PREP_HELP: &str = "\
Usage: haste prep [OPTIONS] [PATH]

Precompile workspace dependencies with rustc so `haste --release --prepared run` can call them directly.

Options:
  -p, --package <PKG>       Select a package in the workspace
      --bin <NAME>          Select a binary target
      --example <NAME>      Select an example target
  -F, --features <LIST>     Activate specific features
      --all-features        Activate all available features
      --no-default-features Do not activate the `default` feature
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";

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
      --release             Compile with optimizations (for `debug code`)
      --jobs <N>            Number of parallel compiler workers
  -h, --help                Print help";
