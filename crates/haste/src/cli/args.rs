use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::outline;
use crate::serve::Request;

use super::help::{HelpTopic, help_text, version_string};

const TOP_COMMANDS: [(Command, &str); 3] = [
    (Command::Serve(ServeAction::Run), "serve"),
    (Command::Prep, "prep"),
    (Command::Clean, "clean"),
];

pub(super) fn usage() -> String {
    let modes: Vec<&str> = crate::serve::MODES
        .into_iter()
        .map(|(_, name)| name)
        .chain(
            TOP_COMMANDS
                .into_iter()
                .filter(|(cmd, _)| !matches!(cmd, Command::Clean))
                .map(|(_, name)| name),
        )
        .collect();
    let queries: Vec<&str> = QUERIES
        .into_iter()
        .filter(|(q, _)| *q != Query::Code)
        .map(|(_, name)| name)
        .collect();
    format!(
        "usage: haste [-p PKG] [--bin NAME | --example NAME] [-F FEATURES] [--all-features] [--no-default-features] [--ignored] [--include-ignored] [--target TRIPLE] [--jobs N] [--release] [--daemon] [--prepared] [--prep-wait MS] [--perf-map] [--watch] [--stats-json PATH] [-o PATH] {} [PATH] [--] [ARGS...]\n       haste serve status [PATH]\n       haste serve stop [PATH | --all]\n       haste clean\n       haste [--jobs N] debug {} PATH [CRATE]\n       haste [--release] debug code PATH DIR\n       haste -V|--version | -h|--help",
        modes.join("|"),
        queries.join("|"),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ServeAction {
    Run,
    Status,
    Stop { all: bool },
}

#[derive(Clone, Copy)]
pub(super) enum Command {
    Request(Request),
    Serve(ServeAction),
    Prep,
    Clean,
    Debug(Query),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Query {
    Graph,
    Outline,
    Lazy,
    Typeck,
    Scope,
    Items,
    Expand,
    Failures,
    Code,
}

const QUERIES: [(Query, &str); 9] = [
    (Query::Graph, "graph"),
    (Query::Outline, "outline"),
    (Query::Lazy, "lazy"),
    (Query::Typeck, "typeck"),
    (Query::Scope, "scope"),
    (Query::Items, "items"),
    (Query::Expand, "expand"),
    (Query::Failures, "failures"),
    (Query::Code, "code"),
];

fn next_str(args: &mut impl Iterator<Item = std::ffi::OsString>) -> Result<String> {
    let raw = args.next().with_context(usage)?;
    raw.into_string()
        .map_err(|bad| anyhow::anyhow!("non-UTF-8 argument: {}", bad.to_string_lossy()))
}

fn parse(
    args: &mut std::iter::Peekable<impl Iterator<Item = std::ffi::OsString>>,
) -> Result<std::result::Result<Command, Meta>> {
    let word = next_str(args)?;
    if word == "serve" {
        if let Some(meta) = args
            .peek()
            .and_then(|raw| raw.to_str())
            .and_then(Meta::from_flag)
        {
            args.next();
            return Ok(Err(meta.with_command(Command::Serve(ServeAction::Run))));
        }
        return Ok(match args.peek().and_then(|raw| raw.to_str()) {
            Some("status") => {
                args.next();
                Ok(Command::Serve(ServeAction::Status))
            }
            Some("stop") => {
                args.next();
                Ok(Command::Serve(ServeAction::Stop { all: false }))
            }
            _ => Ok(Command::Serve(ServeAction::Run)),
        });
    }
    if let Some((cmd, _)) = TOP_COMMANDS.into_iter().find(|(_, name)| *name == word) {
        return Ok(Ok(cmd));
    }
    if word == "debug" {
        if let Some(meta) = args
            .peek()
            .and_then(|raw| raw.to_str())
            .and_then(Meta::from_flag)
        {
            args.next();
            return Ok(Err(meta.with_command(Command::Debug(Query::Graph))));
        }
        let name = next_str(args)?;
        return QUERIES
            .into_iter()
            .find(|(_, query)| *query == name)
            .map(|(query, _)| Ok(Command::Debug(query)))
            .with_context(usage);
    }
    Request::parse_mode(&word)
        .map(|req| Ok(Command::Request(req)))
        .ok_or_else(|| {
            if word == "check" {
                anyhow::anyhow!(
                    "unknown command `check` (did you mean `haste build`?); run `haste --help` for usage"
                )
            } else {
                anyhow::anyhow!("unknown command `{word}`; run `haste --help` for usage")
            }
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Meta {
    Version,
    Help(Option<HelpTopic>),
}

impl Meta {
    pub(super) fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "-V" | "--version" => Some(Self::Version),
            "-h" | "--help" => Some(Self::Help(None)),
            _ => None,
        }
    }

    fn with_command(self, cmd: Command) -> Self {
        match self {
            Self::Help(None) => Self::Help(Some(HelpTopic::from_command(cmd))),
            other => other,
        }
    }
}

#[derive(Default)]
struct ParsedOpts {
    selection: crate::Selection,
    stats: Option<PathBuf>,
    out: Option<PathBuf>,
    jobs: Option<usize>,
    release: bool,
    daemon: bool,
    all_daemons: bool,
    prepared: bool,
    prep_wait: Option<std::time::Duration>,
    perf_map: bool,
    watch: bool,
    target: Option<&'static haste_codegen::Target>,
}

fn consume_options<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    opts: &mut ParsedOpts,
) -> Result<Option<Meta>> {
    while let Some(flag) = args.peek().and_then(|raw| raw.to_str()) {
        if let Some(meta) = Meta::from_flag(flag) {
            args.next();
            return Ok(Some(meta));
        }
        match flag {
            "-p" | "--package" => {
                args.next();
                opts.selection.package =
                    Some(next_str(args).context("--package takes a package name")?);
            }
            "--bin" => {
                args.next();
                opts.selection.bin = Some(next_str(args).context("--bin takes a binary name")?);
            }
            "--example" => {
                args.next();
                opts.selection.example =
                    Some(next_str(args).context("--example takes an example name")?);
            }
            "-F" | "--features" => {
                args.next();
                let list = next_str(args).context("--features takes a feature list")?;
                opts.selection.features.extend(
                    list.split([',', ' '])
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                );
            }
            "--all-features" => {
                args.next();
                opts.selection.all_features = true;
            }
            "--workspace" => {
                args.next();
            }
            "--no-default-features" => {
                args.next();
                opts.selection.no_default_features = true;
            }
            "--ignored" => {
                args.next();
                opts.selection.ignored_only = true;
            }
            "--include-ignored" => {
                args.next();
                opts.selection.include_ignored = true;
            }
            "--jobs" => {
                args.next();
                let count: usize = next_str(args)?
                    .parse()
                    .context("--jobs takes a positive number")?;
                opts.jobs = Some(count.max(1));
            }
            "--release" => {
                args.next();
                opts.release = true;
            }
            "--daemon" => {
                args.next();
                opts.daemon = true;
            }
            "--all" => {
                args.next();
                opts.all_daemons = true;
            }
            "--prepared" => {
                args.next();
                opts.prepared = true;
            }
            "--prep-wait" => {
                args.next();
                let millis: u64 = next_str(args)?
                    .parse()
                    .context("--prep-wait takes a number of milliseconds")?;
                opts.prep_wait = Some(std::time::Duration::from_millis(millis));
            }
            "--perf-map" => {
                args.next();
                opts.perf_map = true;
            }
            "--watch" => {
                args.next();
                opts.watch = true;
            }
            "--stats-json" => {
                args.next();
                let value = args.next().context("--stats-json takes a file path")?;
                opts.stats = Some(PathBuf::from(value));
            }
            "--target" => {
                args.next();
                let triple = next_str(args)?;
                opts.target = Some(crate::config::find_target(&triple)?);
            }
            "-o" | "--output" => {
                args.next();
                let value = args.next().context("-o takes a file path")?;
                opts.out = Some(PathBuf::from(value));
            }
            _ if flag == "--" => break,
            _ if flag.starts_with('-') && flag != "-" => {
                anyhow::bail!("unknown option `{flag}`\n{}", usage())
            }
            _ => break,
        }
    }
    Ok(None)
}

pub(super) type ParsedInvocation = (
    Command,
    PathBuf,
    Vec<String>,
    Option<PathBuf>,
    Option<PathBuf>,
    crate::config::Config,
);

fn parse_args_raw(
    raw: impl IntoIterator<Item = std::ffi::OsString>,
) -> Result<std::result::Result<(ParsedInvocation, ParsedOpts), Meta>> {
    let mut args = raw.into_iter().peekable();
    let mut opts = ParsedOpts::default();
    if let Some(meta) = consume_options(&mut args, &mut opts)? {
        return Ok(Err(meta));
    }
    let command = match parse(&mut args)? {
        Ok(cmd) => cmd,
        Err(meta) => return Ok(Err(meta)),
    };
    if let Some(meta) = consume_options(&mut args, &mut opts)? {
        return Ok(Err(meta.with_command(command)));
    }
    if matches!(command, Command::Clean) {
        if args.peek().is_some() {
            anyhow::bail!("{}", usage());
        }
        let cfg = crate::config::Config::from_env();
        return Ok(Ok((
            (command, PathBuf::new(), Vec::new(), None, None, cfg),
            opts,
        )));
    }
    let path = match command {
        Command::Serve(ServeAction::Status | ServeAction::Stop { .. })
        | Command::Request(_)
        | Command::Serve(ServeAction::Run)
        | Command::Prep => {
            let default_path = if matches!(
                command,
                Command::Serve(ServeAction::Status | ServeAction::Stop { .. })
            ) {
                ""
            } else {
                "."
            };
            match args.peek().and_then(|raw| raw.to_str()) {
                None | Some("--") => PathBuf::from(default_path),
                Some(_) => PathBuf::from(args.next().with_context(usage)?),
            }
        }
        _ => PathBuf::from(args.next().with_context(usage)?),
    };
    if !matches!(command, Command::Debug(_))
        && let Some(meta) = consume_options(&mut args, &mut opts)?
    {
        return Ok(Err(meta.with_command(command)));
    }
    let command = match command {
        Command::Serve(ServeAction::Stop { .. }) => Command::Serve(ServeAction::Stop {
            all: opts.all_daemons,
        }),
        other => {
            if opts.all_daemons {
                anyhow::bail!("--all is only supported with `haste serve stop`");
            }
            other
        }
    };
    if !matches!(command, Command::Request(Request::Build))
        && (opts.stats.is_some() || opts.out.is_some())
    {
        anyhow::bail!("--stats-json and -o are only supported with `haste build`");
    }
    let effective_target = opts.target.unwrap_or_else(haste_codegen::Target::host);
    if effective_target.triple != haste_codegen::Target::host().triple
        && !(matches!(command, Command::Request(Request::Build)) && opts.out.is_some())
    {
        anyhow::bail!(
            "--target {} is only supported with `haste build -o`",
            effective_target.triple
        );
    }
    if opts.watch && !matches!(command, Command::Request(Request::Run)) {
        anyhow::bail!("--watch is only supported with `haste run`");
    }
    let rest = args
        .map(|raw| {
            raw.into_string()
                .map_err(|bad| anyhow::anyhow!("non-UTF-8 argument: {}", bad.to_string_lossy()))
        })
        .collect::<Result<Vec<String>>>()?;
    let rest = match rest.split_first() {
        Some((first, tail)) if first == "--" => tail.to_vec(),
        _ => rest,
    };
    let stats_json = opts.stats.clone();
    let out = opts.out.clone();
    let mut cfg = crate::config::Config::from_env();
    if let Some(count) = opts.jobs {
        cfg.jobs = count;
    }
    if opts.release {
        cfg.profile = outline::tree::Profile::Release;
    }
    cfg.refresh_debug_flags();
    cfg.daemon = opts.daemon;
    cfg.prepared = opts.prepared;
    if let Some(target) = opts.target {
        cfg.target = target;
    }
    cfg.selection = opts.selection.clone();
    cfg.test_mode = matches!(command, Command::Request(Request::Test))
        || (matches!(command, Command::Serve(ServeAction::Run))
            && crate::env_flag("HASTE_SERVE_TEST"));
    Ok(Ok(((command, path, rest, stats_json, out, cfg), opts)))
}

pub(super) fn parse_invocation() -> Result<std::result::Result<ParsedInvocation, Meta>> {
    let (inv, opts) = match parse_args_raw(std::env::args_os().skip(1))? {
        Ok(parsed) => parsed,
        Err(meta) => return Ok(Err(meta)),
    };
    if opts.prepared {
        crate::prep::set_prepared();
    }
    if let Some(wait) = opts.prep_wait {
        crate::prep::set_wait(wait);
    }
    if opts.perf_map {
        crate::image::perf::enable();
    }
    if opts.watch {
        crate::image::watch::enable();
    }
    Ok(Ok(inv))
}

pub(super) fn print_meta(meta: Meta) {
    match meta {
        Meta::Version => println!("{}", version_string()),
        Meta::Help(topic) => println!("{}", help_text(topic)),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;

    fn parse_strs(
        args: &[&str],
    ) -> Result<std::result::Result<(ParsedInvocation, ParsedOpts), Meta>> {
        parse_args_raw(args.iter().map(OsString::from))
    }

    #[test]
    fn run_parses_haste_flags_after_path_until_double_dash() {
        let ((cmd, path, rest, _, _, cfg), opts) =
            parse_strs(&["run", "examples/hello", "--release", "--", "--release", "x"])
                .expect("valid invocation")
                .expect("not meta");
        assert!(matches!(cmd, Command::Request(Request::Run)));
        assert_eq!(path, PathBuf::from("examples/hello"));
        assert!(opts.release);
        assert_eq!(cfg.profile, outline::tree::Profile::Release);
        assert_eq!(rest, vec!["--release", "x"]);
    }

    #[test]
    fn run_passes_flags_after_double_dash_to_program_only() {
        let ((_, _, rest, _, _, cfg), opts) =
            parse_strs(&["run", "examples/hello", "--", "--release"])
                .expect("valid invocation")
                .expect("not meta");
        assert!(!opts.release);
        assert_eq!(cfg.profile, outline::tree::Profile::Debug);
        assert_eq!(rest, vec!["--release"]);
    }

    #[test]
    fn run_rejects_unknown_flag_after_path_without_double_dash() {
        assert!(parse_strs(&["run", "examples/hello", "--unknown-flag"]).is_err());
    }

    #[test]
    fn rejects_removed_unimplemented_targets() {
        assert!(
            parse_strs(&[
                "build",
                "--target",
                "riscv64gc-unknown-linux-gnu",
                "-o",
                "out"
            ])
            .is_err()
        );
        assert!(parse_strs(&["build", "--target", "wasm32-unknown-unknown", "-o", "out"]).is_err());
    }

    #[test]
    fn parses_daemon_flag_and_serve_control_subcommands() {
        let ((cmd, _, _, _, _, cfg), opts) = parse_strs(&["build", "--daemon", "examples/hello"])
            .expect("valid")
            .expect("not meta");
        assert!(matches!(cmd, Command::Request(Request::Build)));
        assert!(opts.daemon && cfg.daemon);

        let ((cmd, path, _, _, _, _), _) = parse_strs(&["serve", "status"])
            .expect("valid")
            .expect("not meta");
        assert!(matches!(cmd, Command::Serve(ServeAction::Status)));
        assert!(path.as_os_str().is_empty());

        let ((cmd, path, _, _, _, _), _) = parse_strs(&["serve", "stop", "--all"])
            .expect("valid")
            .expect("not meta");
        assert!(matches!(
            cmd,
            Command::Serve(ServeAction::Stop { all: true })
        ));
        assert!(path.as_os_str().is_empty());

        let ((cmd, path, _, _, _, _), _) = parse_strs(&["serve", "stop", "examples/hello"])
            .expect("valid")
            .expect("not meta");
        assert!(matches!(
            cmd,
            Command::Serve(ServeAction::Stop { all: false })
        ));
        assert_eq!(path, PathBuf::from("examples/hello"));

        assert!(parse_strs(&["build", "--all", "examples/hello"]).is_err());
    }

    #[test]
    fn subcommand_help_routes_to_topic_and_version_formats_sha() {
        assert_eq!(
            parse_strs(&["--help"]).unwrap().err(),
            Some(Meta::Help(None))
        );
        assert_eq!(
            parse_strs(&["build", "--help"]).unwrap().err(),
            Some(Meta::Help(Some(HelpTopic::Build)))
        );
        assert_eq!(
            parse_strs(&["run", "-h"]).unwrap().err(),
            Some(Meta::Help(Some(HelpTopic::Run)))
        );
        assert_eq!(
            parse_strs(&["test", "--help"]).unwrap().err(),
            Some(Meta::Help(Some(HelpTopic::Test)))
        );
        assert_eq!(
            parse_strs(&["serve", "--help"]).unwrap().err(),
            Some(Meta::Help(Some(HelpTopic::Serve)))
        );
        assert_eq!(
            parse_strs(&["debug", "--help"]).unwrap().err(),
            Some(Meta::Help(Some(HelpTopic::Debug)))
        );
        assert_ne!(help_text(Some(HelpTopic::Build)), help_text(None));
        assert_ne!(
            help_text(Some(HelpTopic::Run)),
            help_text(Some(HelpTopic::Test))
        );
        assert!(version_string().starts_with("haste "));
    }
}
