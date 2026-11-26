use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::outline;
use crate::serve::Request;

const TOP_COMMANDS: [(Command, &str); 3] = [
    (Command::Serve, "serve"),
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
        "usage: haste [-p PKG] [--bin NAME | --example NAME] [-F FEATURES] [--all-features] [--no-default-features] [--target TRIPLE] [--jobs N] [--release] [--prepared] [--prep-wait MS] [--perf-map] [--watch] [--stats-json PATH] [-o PATH] {} [PATH] [--] [ARGS...]\n       haste clean\n       haste [--jobs N] debug {} PATH [CRATE]\n       haste [--release] debug code PATH DIR\n       haste -V|--version | -h|--help",
        modes.join("|"),
        queries.join("|"),
    )
}

#[derive(Clone, Copy)]
pub(super) enum Command {
    Request(Request),
    Serve,
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

fn parse(args: &mut impl Iterator<Item = std::ffi::OsString>) -> Result<Command> {
    let word = next_str(args)?;
    if let Some((cmd, _)) = TOP_COMMANDS.into_iter().find(|(_, name)| *name == word) {
        return Ok(cmd);
    }
    if word == "debug" {
        let name = next_str(args)?;
        return QUERIES
            .into_iter()
            .find(|(_, query)| *query == name)
            .map(|(query, _)| Command::Debug(query))
            .with_context(usage);
    }
    Request::parse_mode(&word)
        .map(Command::Request)
        .with_context(usage)
}

#[derive(Clone, Copy)]
pub(super) enum Meta {
    Version,
    Help,
}

impl Meta {
    pub(super) fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "-V" | "--version" => Some(Self::Version),
            "-h" | "--help" => Some(Self::Help),
            _ => None,
        }
    }
}

fn consume_options<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    selection: &mut crate::Selection,
    stats: &mut Option<PathBuf>,
    out: &mut Option<PathBuf>,
) -> Result<Option<Meta>> {
    while let Some(flag) = args.peek().and_then(|raw| raw.to_str()) {
        if let Some(meta) = Meta::from_flag(flag) {
            args.next();
            return Ok(Some(meta));
        }
        match flag {
            "-p" | "--package" => {
                args.next();
                selection.package = Some(next_str(args).context("--package takes a package name")?);
            }
            "--bin" => {
                args.next();
                selection.bin = Some(next_str(args).context("--bin takes a binary name")?);
            }
            "--example" => {
                args.next();
                selection.example =
                    Some(next_str(args).context("--example takes an example name")?);
            }
            "-F" | "--features" => {
                args.next();
                let list = next_str(args).context("--features takes a feature list")?;
                selection.features.extend(
                    list.split([',', ' '])
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                );
            }
            "--all-features" => {
                args.next();
                selection.all_features = true;
            }
            "--no-default-features" => {
                args.next();
                selection.no_default_features = true;
            }
            "--ignored" => {
                args.next();
                selection.ignored_only = true;
            }
            "--include-ignored" => {
                args.next();
                selection.include_ignored = true;
            }
            "--jobs" => {
                args.next();
                let count: usize = next_str(args)?
                    .parse()
                    .context("--jobs takes a positive number")?;
                crate::set_jobs(count.max(1));
            }
            "--release" => {
                args.next();
                crate::set_profile(outline::tree::Profile::Release);
            }
            "--prepared" => {
                args.next();
                crate::prep::set_prepared();
            }
            "--prep-wait" => {
                args.next();
                let millis: u64 = next_str(args)?
                    .parse()
                    .context("--prep-wait takes a number of milliseconds")?;
                crate::prep::set_wait(std::time::Duration::from_millis(millis));
            }
            "--perf-map" => {
                args.next();
                crate::image::perf::enable();
            }
            "--watch" => {
                args.next();
                crate::image::watch::enable();
            }
            "--stats-json" => {
                args.next();
                let value = args.next().context("--stats-json takes a file path")?;
                *stats = Some(PathBuf::from(value));
            }
            "--target" => {
                args.next();
                let triple = next_str(args)?;
                let target = haste_codegen::Target::find(&triple).with_context(|| {
                    let known: Vec<&str> =
                        haste_codegen::TARGETS.iter().map(|t| t.triple).collect();
                    format!(
                        "unknown target `{triple}`; supported targets: {}",
                        known.join(", ")
                    )
                })?;
                crate::set_target(target);
            }
            "-o" | "--output" => {
                args.next();
                let value = args.next().context("-o takes a file path")?;
                *out = Some(PathBuf::from(value));
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
);

pub(super) fn parse_invocation() -> Result<std::result::Result<ParsedInvocation, Meta>> {
    let mut args = std::env::args_os().skip(1).peekable();
    let mut selection = crate::Selection::default();
    let mut stats_json = None;
    let mut out = None;
    if let Some(meta) = consume_options(&mut args, &mut selection, &mut stats_json, &mut out)? {
        return Ok(Err(meta));
    }
    let command = parse(&mut args)?;
    if let Some(meta) = consume_options(&mut args, &mut selection, &mut stats_json, &mut out)? {
        return Ok(Err(meta));
    }
    if matches!(command, Command::Clean) {
        if args.peek().is_some() {
            anyhow::bail!("{}", usage());
        }
        return Ok(Ok((command, PathBuf::new(), Vec::new(), None, None)));
    }
    let path = if matches!(
        command,
        Command::Request(_) | Command::Serve | Command::Prep
    ) {
        match args.peek().and_then(|raw| raw.to_str()) {
            None | Some("--") => PathBuf::from("."),
            Some(_) => PathBuf::from(args.next().with_context(usage)?),
        }
    } else {
        PathBuf::from(args.next().with_context(usage)?)
    };
    if !matches!(command, Command::Request(Request::Run) | Command::Debug(_))
        && let Some(meta) = consume_options(&mut args, &mut selection, &mut stats_json, &mut out)?
    {
        return Ok(Err(meta));
    }
    crate::set_selection(selection);
    if matches!(command, Command::Request(Request::Test))
        || (matches!(command, Command::Serve)
            && std::env::var_os("HASTE_SERVE_TEST").is_some_and(|v| v == "1"))
    {
        crate::set_test_mode();
    }
    if !matches!(command, Command::Request(Request::Build))
        && (stats_json.is_some() || out.is_some())
    {
        anyhow::bail!("--stats-json and -o are only supported with `haste build`");
    }
    if crate::compile_target().triple != haste_codegen::Target::host().triple
        && !(matches!(command, Command::Request(Request::Build)) && out.is_some())
    {
        anyhow::bail!(
            "--target {} is only supported with `haste build -o`",
            crate::compile_target().triple
        );
    }
    if crate::image::watch::enabled() && !matches!(command, Command::Request(Request::Run)) {
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
    Ok(Ok((command, path, rest, stats_json, out)))
}

pub(super) fn print_meta(meta: Meta) {
    match meta {
        Meta::Version => println!("haste {}", env!("CARGO_PKG_VERSION")),
        Meta::Help => println!("{}", usage()),
    }
}
