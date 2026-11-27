use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::outline;
use crate::serve::Request;

use super::help::{HelpTopic, help_text, version_string};

const TOP_COMMANDS: [(Command, &str); 3] = [
    (Command::Serve(ServeAction::Run), "serve"),
    (Command::Prep, "prep"),
    (Command::Clean { all: false }, "clean"),
];

pub(super) fn usage() -> String {
    let modes: Vec<&str> = crate::serve::MODES
        .into_iter()
        .map(|(_, name)| name)
        .chain(std::iter::once("check"))
        .chain(
            TOP_COMMANDS
                .into_iter()
                .filter(|(cmd, _)| !matches!(cmd, Command::Clean { .. }))
                .map(|(_, name)| name),
        )
        .collect();
    let queries: Vec<&str> = QUERIES
        .into_iter()
        .filter(|(q, _)| *q != Query::Code)
        .map(|(_, name)| name)
        .collect();
    format!(
        "usage: haste [-p PKG] [--workspace] [--lib] [--bins] [--bin NAME | --example NAME] [--tests] [--all-targets] [-F FEATURES] [--all-features] [--no-default-features] [--manifest-path PATH] [--locked] [--offline] [--frozen] [--color WHEN] [--no-run] [--ignored] [--include-ignored] [--target TRIPLE] [-j|--jobs N] [--release | --profile NAME] [--message-format FMT] [-q|--quiet] [--daemon] [--prepared] [--prep-wait MS] [--perf-map] [--watch] [--stats-json PATH] [-o PATH] {} [PATH] [--] [ARGS...]\n       haste serve status [PATH]\n       haste serve stop [PATH | --all]\n       haste clean [PATH | --all]\n       haste [-j|--jobs N] debug {} PATH [CRATE]\n       haste [--release] debug code PATH DIR\n       haste -V|--version | -h|--help",
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
    Clean { all: bool },
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
    opts: &mut ParsedOpts,
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
    if word == "check" {
        opts.check_alias = true;
        return Ok(Ok(Command::Request(Request::Build)));
    }
    Request::parse_mode(&word)
        .map(|req| Ok(Command::Request(req)))
        .ok_or_else(|| anyhow::anyhow!("unknown command `{word}`; run `haste --help` for usage"))
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
    manifest_path: Option<PathBuf>,
    stats: Option<PathBuf>,
    out: Option<PathBuf>,
    jobs: Option<usize>,
    release: bool,
    profile: Option<outline::tree::Profile>,
    daemon: bool,
    all_daemons: bool,
    prepared: bool,
    prep_wait: Option<std::time::Duration>,
    perf_map: bool,
    watch: bool,
    target: Option<&'static haste_codegen::Target>,
    check_alias: bool,
    no_run: bool,
}

fn split_flag(flag: &str) -> (&str, Option<&str>) {
    if let Some((name, val)) = flag.split_once('=') {
        (name, Some(val))
    } else if let Some(rest) = flag.strip_prefix("-j")
        && !rest.is_empty()
        && rest.bytes().all(|b| b.is_ascii_digit())
    {
        ("-j", Some(rest))
    } else {
        (flag, None)
    }
}

fn take_str<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    eq_val: Option<&str>,
    ctx: &str,
) -> Result<String> {
    args.next();
    match eq_val {
        Some(val) => Ok(val.to_owned()),
        None => next_str(args).with_context(|| ctx.to_owned()),
    }
}

fn take_non_empty<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    eq_val: Option<&str>,
    ctx: &str,
) -> Result<String> {
    let val = take_str(args, eq_val, ctx)?;
    if val.is_empty() {
        anyhow::bail!("{ctx}");
    }
    Ok(val)
}

fn take_os<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    eq_val: Option<&str>,
    ctx: &str,
) -> Result<std::ffi::OsString> {
    args.next();
    match eq_val {
        Some(val) if !val.is_empty() => Ok(std::ffi::OsString::from(val)),
        Some(_) => anyhow::bail!("{ctx}"),
        None => args.next().with_context(|| ctx.to_owned()),
    }
}

fn parse_profile(name: &str) -> Result<outline::tree::Profile> {
    match name {
        "release" => Ok(outline::tree::Profile::Release),
        "dev" | "debug" => Ok(outline::tree::Profile::Debug),
        _ => anyhow::bail!("unknown profile `{name}`; expected `dev`, `debug`, or `release`"),
    }
}

fn parse_message_format(fmt: &str) -> Result<crate::MessageFormat> {
    match fmt {
        "human" => Ok(crate::MessageFormat::Human),
        "short" => Ok(crate::MessageFormat::Short),
        "json" => Ok(crate::MessageFormat::Json),
        "json-render-diagnostics" => Ok(crate::MessageFormat::JsonRenderDiagnostics),
        _ => anyhow::bail!(
            "unknown message format `{fmt}`; expected `human`, `short`, `json`, or `json-render-diagnostics`"
        ),
    }
}

fn parse_color(when: &str) -> Result<()> {
    match when {
        "auto" | "always" | "never" => Ok(()),
        _ => anyhow::bail!("unknown color option `{when}`; expected `auto`, `always`, or `never`"),
    }
}

fn manifest_workspace_dir(manifest: &std::path::Path) -> PathBuf {
    match manifest.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn apply_bool_flag(opts: &mut ParsedOpts, flag: &str) -> bool {
    match flag {
        "--lib" => opts.selection.lib = true,
        "--bins" => opts.selection.bins = true,
        "--tests" => opts.selection.tests = true,
        "--all-targets" => opts.selection.all_targets = true,
        "--all-features" => opts.selection.all_features = true,
        "--workspace" => opts.selection.workspace = true,
        "--no-default-features" => opts.selection.no_default_features = true,
        "--locked" => opts.selection.locked = true,
        "--offline" => opts.selection.offline = true,
        "--frozen" => {
            opts.selection.locked = true;
            opts.selection.offline = true;
        }
        "--no-run" => opts.no_run = true,
        "-q" | "--quiet" => opts.selection.quiet = true,
        "--ignored" => opts.selection.ignored_only = true,
        "--include-ignored" => opts.selection.include_ignored = true,
        "--release" => {
            opts.release = true;
            opts.profile = Some(outline::tree::Profile::Release);
        }
        "--daemon" => opts.daemon = true,
        "--all" => opts.all_daemons = true,
        "--prepared" => opts.prepared = true,
        "--perf-map" => opts.perf_map = true,
        "--watch" => opts.watch = true,
        _ => return false,
    }
    true
}

fn apply_value_flag<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    opts: &mut ParsedOpts,
    name: &str,
    eq_val: Option<&str>,
) -> Result<bool> {
    match name {
        "-p" | "--package" | "--bin" | "--example" | "--test" => {
            let (dest, msg) = match name {
                "-p" | "--package" => (
                    &mut opts.selection.packages,
                    "--package takes a package name",
                ),
                "--bin" => (&mut opts.selection.bin, "--bin takes a binary name"),
                "--example" => (
                    &mut opts.selection.example,
                    "--example takes an example name",
                ),
                _ => (&mut opts.selection.test, "--test takes a test target name"),
            };
            dest.push(take_non_empty(args, eq_val, msg)?);
        }
        "-F" | "--features" => {
            let list = take_str(args, eq_val, "--features takes a feature list")?;
            opts.selection.features.extend(
                list.split([',', ' '])
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
            );
        }
        "--manifest-path" => {
            let value = take_os(args, eq_val, "--manifest-path takes a file path")?;
            opts.manifest_path = Some(PathBuf::from(value));
        }
        "-j" | "--jobs" => {
            let raw = take_non_empty(args, eq_val, "--jobs takes a positive number")?;
            let count: usize = raw.parse().context("--jobs takes a positive number")?;
            opts.jobs = Some(count.max(1));
        }
        "--profile" => {
            let prof_name = take_non_empty(args, eq_val, "--profile takes a profile name")?;
            let prof = parse_profile(&prof_name)?;
            opts.profile = Some(prof);
            opts.release = prof == outline::tree::Profile::Release;
        }
        "--message-format" => {
            let fmt = take_non_empty(args, eq_val, "--message-format takes a format name")?;
            opts.selection.message_format = parse_message_format(&fmt)?;
        }
        "--color" => {
            let when = take_non_empty(args, eq_val, "--color takes `auto`, `always`, or `never`")?;
            parse_color(&when)?;
        }
        "--prep-wait" => {
            let raw = take_non_empty(args, eq_val, "--prep-wait takes a number of milliseconds")?;
            let millis: u64 = raw
                .parse()
                .context("--prep-wait takes a number of milliseconds")?;
            opts.prep_wait = Some(std::time::Duration::from_millis(millis));
        }
        "--stats-json" => {
            let value = take_os(args, eq_val, "--stats-json takes a file path")?;
            opts.stats = Some(PathBuf::from(value));
        }
        "--target" => {
            let triple = take_non_empty(args, eq_val, "--target takes a target triple")?;
            opts.target = Some(crate::config::find_target(&triple)?);
        }
        "-o" | "--output" => {
            let value = take_os(args, eq_val, "-o takes a file path")?;
            opts.out = Some(PathBuf::from(value));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn consume_options<I: Iterator<Item = std::ffi::OsString>>(
    args: &mut std::iter::Peekable<I>,
    opts: &mut ParsedOpts,
) -> Result<Option<Meta>> {
    while let Some(flag) = args.peek().and_then(|raw| raw.to_str()).map(str::to_owned) {
        if let Some(meta) = Meta::from_flag(&flag) {
            args.next();
            return Ok(Some(meta));
        }
        if flag == "--" || !flag.starts_with('-') || flag == "-" {
            break;
        }
        let (name, eq_val) = split_flag(&flag);
        if apply_value_flag(args, opts, name, eq_val)? {
            continue;
        }
        if eq_val.is_none() && apply_bool_flag(opts, &flag) {
            args.next();
            continue;
        }
        anyhow::bail!("unknown option `{flag}`\n{}", usage());
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

fn take_path<I: Iterator<Item = std::ffi::OsString>>(
    command: Command,
    args: &mut std::iter::Peekable<I>,
    has_manifest: bool,
) -> Result<(PathBuf, bool)> {
    match command {
        Command::Serve(ServeAction::Status | ServeAction::Stop { .. })
        | Command::Clean { .. }
        | Command::Request(_)
        | Command::Serve(ServeAction::Run)
        | Command::Prep => {
            let default_path = if matches!(
                command,
                Command::Serve(ServeAction::Status | ServeAction::Stop { .. })
                    | Command::Clean { .. }
            ) {
                ""
            } else {
                "."
            };
            match args.peek().and_then(|raw| raw.to_str()) {
                None | Some("--") => Ok((PathBuf::from(default_path), false)),
                Some(_) => Ok((PathBuf::from(args.next().with_context(usage)?), true)),
            }
        }
        Command::Debug(_) if has_manifest && args.peek().is_none() => Ok((PathBuf::new(), false)),
        _ => Ok((PathBuf::from(args.next().with_context(usage)?), true)),
    }
}

fn validate_command_opts(command: Command, opts: &ParsedOpts) -> Result<Command> {
    let command = match command {
        Command::Serve(ServeAction::Stop { .. }) => Command::Serve(ServeAction::Stop {
            all: opts.all_daemons,
        }),
        Command::Clean { .. } => Command::Clean {
            all: opts.all_daemons,
        },
        other => {
            if opts.all_daemons {
                anyhow::bail!("--all is only supported with `haste serve stop` and `haste clean`");
            }
            other
        }
    };
    if opts.check_alias && opts.out.is_some() {
        anyhow::bail!("cannot use -o/--output with 'haste check'; use 'haste build -o'");
    }
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
    if (matches!(command, Command::Request(Request::Run))
        || (matches!(command, Command::Request(Request::Build)) && opts.out.is_some()))
        && (opts.selection.bin.len() > 1
            || opts.selection.example.len() > 1
            || (!opts.selection.bin.is_empty() && !opts.selection.example.is_empty()))
    {
        anyhow::bail!("cannot select multiple binary or example targets when running or linking");
    }
    if opts.watch && !matches!(command, Command::Request(Request::Run)) {
        anyhow::bail!("--watch is only supported with `haste run`");
    }
    if opts.no_run && !matches!(command, Command::Request(Request::Test)) {
        anyhow::bail!("--no-run is only supported with `haste test`");
    }
    Ok(command)
}

fn build_config(command: Command, opts: &ParsedOpts) -> crate::config::Config {
    let mut cfg = crate::config::Config::from_env();
    if let Some(count) = opts.jobs {
        cfg.jobs = count;
    }
    if let Some(profile) = opts.profile {
        cfg.profile = profile;
    } else if opts.release {
        cfg.profile = outline::tree::Profile::Release;
    }
    cfg.refresh_debug_flags();
    cfg.daemon = opts.daemon;
    cfg.prepared = opts.prepared;
    if let Some(target) = opts.target {
        cfg.target = target;
    }
    cfg.selection = opts.selection.clone();
    cfg.no_run = opts.no_run;
    cfg.test_mode = matches!(command, Command::Request(Request::Test))
        || opts.selection.tests
        || !opts.selection.test.is_empty()
        || opts.selection.all_targets
        || (matches!(command, Command::Serve(ServeAction::Run))
            && crate::env_flag("HASTE_SERVE_TEST"));
    cfg
}

fn collect_rest(args: impl Iterator<Item = std::ffi::OsString>) -> Result<Vec<String>> {
    let rest = args
        .map(|raw| {
            raw.into_string()
                .map_err(|bad| anyhow::anyhow!("non-UTF-8 argument: {}", bad.to_string_lossy()))
        })
        .collect::<Result<Vec<String>>>()?;
    Ok(match rest.split_first() {
        Some((first, tail)) if first == "--" => tail.to_vec(),
        _ => rest,
    })
}

fn parse_args_raw(
    raw: impl IntoIterator<Item = std::ffi::OsString>,
) -> Result<std::result::Result<(ParsedInvocation, ParsedOpts), Meta>> {
    let mut args = raw.into_iter().peekable();
    let mut opts = ParsedOpts::default();
    if let Some(meta) = consume_options(&mut args, &mut opts)? {
        return Ok(Err(meta));
    }
    let command = match parse(&mut args, &mut opts)? {
        Ok(cmd) => cmd,
        Err(meta) => return Ok(Err(meta)),
    };
    if let Some(meta) = consume_options(&mut args, &mut opts)? {
        return Ok(Err(meta.with_command(command)));
    }
    let (path, explicit_path) = take_path(command, &mut args, opts.manifest_path.is_some())?;
    if !matches!(command, Command::Debug(_))
        && let Some(meta) = consume_options(&mut args, &mut opts)?
    {
        return Ok(Err(meta.with_command(command)));
    }
    let path = match &opts.manifest_path {
        Some(_) if explicit_path => {
            anyhow::bail!("cannot pass both a positional path and `--manifest-path`")
        }
        Some(manifest) => manifest_workspace_dir(manifest),
        None => path,
    };
    if matches!(command, Command::Clean { .. })
        && opts.all_daemons
        && (explicit_path || opts.manifest_path.is_some())
    {
        anyhow::bail!("cannot pass both a path and `--all`");
    }
    let command = validate_command_opts(command, &opts)?;
    let rest = collect_rest(args)?;
    let (stats_json, out) = (opts.stats.clone(), opts.out.clone());
    let cfg = build_config(command, &opts);
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
            parse_strs(&["--help"]).expect("valid help").err(),
            Some(Meta::Help(None))
        );
        assert_eq!(
            parse_strs(&["build", "--help"]).expect("valid help").err(),
            Some(Meta::Help(Some(HelpTopic::Build)))
        );
        assert_eq!(
            parse_strs(&["run", "-h"]).expect("valid help").err(),
            Some(Meta::Help(Some(HelpTopic::Run)))
        );
        assert_eq!(
            parse_strs(&["test", "--help"]).expect("valid help").err(),
            Some(Meta::Help(Some(HelpTopic::Test)))
        );
        assert_eq!(
            parse_strs(&["serve", "--help"]).expect("valid help").err(),
            Some(Meta::Help(Some(HelpTopic::Serve)))
        );
        assert_eq!(
            parse_strs(&["debug", "--help"]).expect("valid help").err(),
            Some(Meta::Help(Some(HelpTopic::Debug)))
        );
        assert_ne!(help_text(Some(HelpTopic::Build)), help_text(None));
        assert_ne!(
            help_text(Some(HelpTopic::Run)),
            help_text(Some(HelpTopic::Test))
        );
        assert!(version_string().starts_with("haste "));
    }

    #[test]
    fn check_alias_rejects_output_flag() {
        for args in [
            &["check", "-o", "out"][..],
            &["check", "--output", "out"],
            &["check", "examples/hello", "-o=out"],
            &["-o", "out", "check"],
        ] {
            let err = parse_strs(args).err().expect("check with -o must error");
            assert!(
                err.to_string()
                    .contains("cannot use -o/--output with 'haste check'; use 'haste build -o'"),
                "unexpected error for {args:?}: {err:#}"
            );
        }
        let ((cmd, _, _, _, out, _), opts) = parse_strs(&["check", "examples/hello"])
            .expect("plain check succeeds")
            .expect("not meta");
        assert!(matches!(cmd, Command::Request(Request::Build)));
        assert!(opts.check_alias);
        assert!(out.is_none());
    }

    #[test]
    fn parses_cargo_muscle_memory_flags() {
        let ((_, _, _, _, _, cfg), _) = parse_strs(&["build", "--frozen", "--color=always"])
            .expect("valid")
            .expect("not meta");
        assert!(cfg.selection.locked && cfg.selection.offline);
        assert!(parse_strs(&["build", "--color", "never"]).is_ok());
        assert!(parse_strs(&["build", "--color=bogus"]).is_err());
        let ((_, _, _, _, _, cfg), _) = parse_strs(&["test", "--no-run"])
            .expect("valid")
            .expect("not meta");
        assert!(cfg.no_run);
        assert!(parse_strs(&["build", "--no-run"]).is_err());
        let u = usage();
        assert!(u.contains("--lib") && u.contains("--bins") && u.contains("--frozen"));
    }

    #[test]
    fn parses_repeated_target_and_package_flags_and_enables_test_mode() {
        let cfg = |a: &[&str]| parse_strs(a).expect("valid").expect("not meta").0.5;
        assert_eq!(
            cfg(&["check", "-p", "a", "-p", "b", "--package=c"])
                .selection
                .packages,
            vec!["a", "b", "c"]
        );
        let check_test = cfg(&["check", "--test", "payments"]);
        assert!(check_test.test_mode && check_test.selection.test == vec!["payments"]);
        let multi = cfg(&[
            "test", "--test", "payments", "--test", "receipts", "--bin", "a", "--bin=b",
        ]);
        assert_eq!(multi.selection.test, vec!["payments", "receipts"]);
        assert_eq!(multi.selection.bin, vec!["a", "b"]);
        assert!(parse_strs(&["run", "--bin", "a", "--bin", "b"]).is_err());
    }

    #[test]
    fn clean_parses_workspace_path_and_all_flag() {
        let clean = |a: &[&str]| {
            let ((cmd, path, _, _, _, _), _) = parse_strs(a).expect("valid").expect("not meta");
            (cmd, path)
        };
        let (cmd, path) = clean(&["clean"]);
        assert!(matches!(cmd, Command::Clean { all: false }) && path.as_os_str().is_empty());
        let (cmd, path) = clean(&["clean", "examples/hello"]);
        assert!(
            matches!(cmd, Command::Clean { all: false }) && path == PathBuf::from("examples/hello")
        );
        let (cmd, path) = clean(&["clean", "--all"]);
        assert!(matches!(cmd, Command::Clean { all: true }) && path.as_os_str().is_empty());
        assert!(parse_strs(&["clean", "--all", "examples/hello"]).is_err());
    }
}
