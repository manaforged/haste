mod args;
mod debug;

use std::time::Instant;

use anyhow::{Context, Result};

use crate::graph::{self, Targets, cfg};
use crate::names::ItemRef;
use crate::serve::Request;
use crate::{outline, sema};

use args::{Command, Meta, Query, ServeAction, parse_invocation, print_meta};
use debug::{print_expanded, print_items, report_query};

/// Metadata for a discovered `#[test]` function.
#[derive(Clone)]
pub(crate) struct TestItem {
    pub item: ItemRef,
    pub ignored: bool,
    pub should_panic: bool,
    pub expected: Option<String>,
}

fn item_enabled(tree: &outline::tree::Crate, file_at: usize, at: usize) -> bool {
    let items = &tree.files[file_at].items.items;
    let mut cur = Some(at);
    while let Some(idx) = cur {
        let item = &items[idx];
        if !tree.env.enabled(&item.attrs) {
            return false;
        }
        if idx != at && !matches!(item.kind, outline::item::ItemKind::Mod) {
            return false;
        }
        cur = item.parent.map(|p| p as usize);
    }
    true
}

fn member_fns<'a>(
    workspace: &'a graph::Workspace,
    outline: &'a outline::Outline,
) -> impl Iterator<
    Item = (
        ItemRef,
        &'a outline::tree::Crate,
        &'a outline::tree::File,
        &'a outline::item::Item,
    ),
> + 'a {
    outline.members(workspace).flat_map(move |krate| {
        let tree = &outline.crates[krate];
        tree.files
            .iter()
            .enumerate()
            .flat_map(move |(file_at, file)| {
                file.items
                    .items
                    .iter()
                    .enumerate()
                    .filter_map(move |(at, item)| {
                        (item.kind == outline::item::ItemKind::Fn
                            && item.body.is_some()
                            && item_enabled(tree, file_at, at))
                        .then_some((
                            ItemRef {
                                krate: krate as u32,
                                file: file_at as u32,
                                item: at as u32,
                            },
                            tree,
                            file.as_ref(),
                            item,
                        ))
                    })
            })
    })
}

pub(crate) fn test_items(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
) -> Vec<TestItem> {
    member_fns(workspace, outline)
        .filter(|(_, tree, file, item)| {
            outline::attr::has_attr(&file.source, &item.attrs, &tree.env, "test")
        })
        .map(|(item_ref, tree, file, item)| {
            let should_panic =
                outline::attr::has_attr(&file.source, &item.attrs, &tree.env, "should_panic");
            let expected = should_panic
                .then(|| outline::attr::should_panic_expected(&file.source, &item.attrs, &tree.env))
                .flatten();
            TestItem {
                item: item_ref,
                ignored: outline::attr::has_attr(&file.source, &item.attrs, &tree.env, "ignore"),
                should_panic,
                expected,
            }
        })
        .collect()
}

fn panic_payload(stderr: &str) -> Option<&str> {
    let mut offset = 0;
    for line in stderr.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.starts_with("thread '")
            && trimmed.contains(" panicked at ")
            && trimmed.ends_with(':')
        {
            let rest = &stderr[offset + line.len()..];
            let body = rest.split("\nnote: run with ").next().unwrap_or(rest);
            let body = body.split("\nstack backtrace:").next().unwrap_or(body);
            return Some(body.trim_end_matches(['\r', '\n']));
        }
        offset += line.len();
    }
    None
}

/// Formats and prints standard Rust test output and returns 0 when all tests pass, else 101.
pub(crate) fn print_test_report(
    entries: &[(&str, &TestItem, Option<Result<(i32, Option<String>), &str>>)],
    elapsed: std::time::Duration,
) -> i32 {
    println!(
        "\nrunning {} test{}",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" }
    );
    let (mut passed, mut failed, mut ignored) = (0usize, 0usize, 0usize);
    let mut details: Vec<(&str, String)> = Vec::new();
    for &(name, meta, ref outcome) in entries {
        let Some(outcome) = outcome else {
            println!("test {name} ... ignored");
            ignored += 1;
            continue;
        };
        match outcome {
            Ok((status, stderr)) => {
                let expected_ok = match (&meta.expected, stderr) {
                    (Some(exp), Some(err)) => {
                        panic_payload(err).is_some_and(|msg| msg.contains(exp.as_str()))
                    }
                    (Some(_), None) => false,
                    (None, _) => true,
                };
                let ok = if meta.should_panic {
                    *status != 0 && expected_ok
                } else {
                    *status == 0
                };
                if ok {
                    println!("test {name} ... ok");
                    passed += 1;
                } else {
                    let msg = if meta.should_panic && *status == 0 {
                        "note: test did not panic as expected".to_owned()
                    } else if let Some(exp) = &meta.expected {
                        format!(
                            "note: panic did not contain expected string `{exp}`\n{}",
                            stderr.as_deref().unwrap_or("").trim()
                        )
                    } else if let Some(out) =
                        stderr.as_deref().map(str::trim).filter(|s| !s.is_empty())
                    {
                        format!("{out}\n(exited with status {status})")
                    } else {
                        format!("exited with status {status}")
                    };
                    println!("test {name} ... FAILED");
                    details.push((name, msg));
                    failed += 1;
                }
            }
            Err(err) => {
                println!("test {name} ... FAILED");
                details.push((name, (*err).to_owned()));
                failed += 1;
            }
        }
    }
    if !details.is_empty() {
        println!("\nfailures:\n");
        for (name, msg) in &details {
            println!("---- {name} ----\n{msg}\n");
        }
    }
    let verdict = if failed == 0 { "ok" } else { "FAILED" };
    println!(
        "\ntest result: {verdict}. {passed} passed; {failed} failed; {ignored} ignored; finished in {:.2}s\n",
        elapsed.as_secs_f64()
    );
    if failed == 0 { 0 } else { 101 }
}

pub(crate) fn lib_fn_items(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
) -> Vec<ItemRef> {
    member_fns(workspace, outline)
        .map(|(item_ref, _, _, _)| item_ref)
        .collect()
}

pub(crate) fn case_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    member_fns(workspace, outline)
        .filter(|(_, _, file, item)| {
            item.name
                .is_some_and(|name| file.source.text(name).starts_with("case_"))
        })
        .map(|(item_ref, _, _, _)| item_ref)
        .collect()
}

pub(crate) type Failures = std::collections::BTreeMap<String, Vec<String>>;

pub(crate) fn print_failures(failures: &Failures) {
    for (error, items) in failures {
        let names: Vec<&str> = items
            .iter()
            .take(3)
            .filter_map(|item| item.split(':').next())
            .collect();
        eprintln!(
            "{:>4} {error} [{}]\n       {}",
            items.len(),
            names.join(" "),
            items[0].chars().take(300).collect::<String>()
        );
    }
}

pub(crate) fn error_kind(text: &str) -> String {
    let text = text.split_once(": ").map_or(text, |(_, rest)| rest);
    text.split([' ', '(', '`'])
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Each group of `failures`, largest first, as a count and kind line followed by one indented line per error.
pub(crate) fn failure_lines(failures: &Failures) -> Vec<String> {
    let mut groups: Vec<(&String, &Vec<String>)> = failures.iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));
    groups
        .into_iter()
        .flat_map(|(kind, errors)| {
            std::iter::once(format!("{} {kind}", errors.len()))
                .chain(errors.iter().map(|error| format!("    {error}")))
        })
        .collect()
}

fn report_cases(
    root: &std::path::Path,
    (workspace, targets): (&graph::Workspace, &Targets),
    (request, args): (Request, &[String]),
    stats: Option<(&std::path::Path, sema::census::Span)>,
    out: Option<&std::path::Path>,
) -> Result<i32> {
    if request == Request::Run {
        return crate::build::run(workspace, targets, args);
    }
    let started = Instant::now();
    let mut build = crate::build::Build::load(workspace, targets)?;
    let result = match (request, out) {
        (Request::Build, Some(path)) => build.emit_executable(workspace, path, stats.is_none()),
        (Request::Build, None) if stats.is_none() => build
            .compile_binary_oneshot(workspace)
            .and_then(|()| build.check(started.elapsed())),
        _ => crate::serve::finish(&mut build, workspace, (request, out, args), started),
    };
    if let Some((path, graph_span)) = stats {
        build.write_stats(workspace, graph_span, path)?;
    }
    eprint!("{}", haste_codegen::missed_report());
    eprint!("{}", crate::lower::Bodies::inline_report());
    eprint!("{}", crate::substrate::tally::report());
    if request == Request::Build && crate::substrate::noop::active(stats.map(|(p, _)| p)) {
        match &result {
            Ok(0) => {
                let watch_paths = build.watch_paths();
                crate::substrate::noop::record(
                    root,
                    workspace,
                    &watch_paths,
                    out,
                    build.reached_count(),
                );
            }
            _ => crate::substrate::noop::invalidate(root, out),
        }
    }
    std::mem::forget(build);
    result
}

fn clean_cache() -> Result<()> {
    let home = crate::graph::store::home();
    match std::fs::remove_dir_all(&home) {
        Ok(()) => eprintln!("removed {}", home.display()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("cache {} is already empty", home.display())
        }
        Err(err) => return Err(err).with_context(|| format!("removing {}", home.display())),
    }
    Ok(())
}

fn can_serve(command: Command, stats_json: Option<&std::path::Path>) -> bool {
    matches!(command, Command::Request(_))
        && stats_json.is_none()
        && !crate::image::watch::enabled()
        && crate::compile_target().triple == haste_codegen::Target::host().triple
        && !crate::env_flag("HASTE_NO_DAEMON")
}

fn try_serve(
    command: Command,
    root: &std::path::Path,
    stats_json: Option<&std::path::Path>,
    out: Option<&std::path::Path>,
    argv: &[String],
) -> Result<Option<i32>> {
    if !can_serve(command, stats_json) {
        return Ok(None);
    }
    match command {
        Command::Request(request) => crate::serve::ask(root, request, out, argv),
        _ => Ok(None),
    }
}

/// Runs the haste command line.
pub fn main() -> Result<()> {
    if let Some(meta) = std::env::args_os()
        .nth(1)
        .and_then(|a| a.to_str().and_then(Meta::from_flag))
    {
        return Ok(print_meta(meta));
    }
    crate::sema::evaluate::install(crate::image::evaluator);
    let (command, path, rest, stats_json, out, cfg) = match parse_invocation()? {
        Ok(parsed) => parsed,
        Err(meta) => return Ok(print_meta(meta)),
    };
    crate::with_config(std::sync::Arc::new(cfg), || {
        run_parsed(command, path, rest, stats_json, out)
    })
}

fn run_parsed(
    command: Command,
    path: std::path::PathBuf,
    rest: Vec<String>,
    stats_json: Option<std::path::PathBuf>,
    out: Option<std::path::PathBuf>,
) -> Result<()> {
    match command {
        Command::Clean => return clean_cache(),
        Command::Debug(Query::Items) => return print_items(&path),
        Command::Debug(Query::Expand) => return print_expanded(&path),
        Command::Serve(ServeAction::Status) => {
            let filter = (!path.as_os_str().is_empty())
                .then(|| std::path::absolute(&path))
                .transpose()?;
            return exit_status(crate::serve::status(filter.as_deref())?);
        }
        Command::Serve(ServeAction::Stop { all }) => {
            let filter = if all {
                None
            } else if path.as_os_str().is_empty() {
                Some(std::path::absolute(std::path::Path::new("."))?)
            } else {
                Some(std::path::absolute(&path)?)
            };
            return exit_status(crate::serve::stop(filter.as_deref())?);
        }
        _ => {}
    }
    let root = std::path::absolute(path)?;
    let abs_out = out.as_deref().map(std::path::absolute).transpose()?;
    let argv: Vec<String> = std::iter::once(program_name(&root)).chain(rest).collect();
    if let Some(status) = try_serve(
        command,
        &root,
        stats_json.as_deref(),
        abs_out.as_deref(),
        &argv,
    )? {
        std::process::exit(status);
    }
    let status = report(
        command,
        &root,
        &argv,
        stats_json.as_deref(),
        abs_out.as_deref(),
    )?;
    if status == 0
        && cfg!(unix)
        && crate::daemon()
        && can_serve(command, stats_json.as_deref())
        && let Command::Request(request) = command
    {
        crate::serve::spawn(&root, request, abs_out.as_deref());
    }
    exit_status(status)
}

fn exit_status(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        code => std::process::exit(code),
    }
}

fn report_prep(workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
    let started = Instant::now();
    let counts = crate::build::prep(workspace, targets)?;
    println!(
        "{counts} in {:.1} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}

pub(crate) fn program_name(root: &std::path::Path) -> String {
    root.file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// Loads the workspace at `root` with the standard library merged in and the output of every build script settled.
pub(crate) fn load_workspace(
    root: &std::path::Path,
    targets: &Targets,
) -> Result<graph::Workspace> {
    let mut workspace = graph::load(root, targets, &crate::selection())?;
    workspace.merge(graph::load_sysroot(&graph::library()?, targets)?);
    crate::scripts::settle(&mut workspace, targets)?;
    Ok(workspace)
}

fn report(
    command: Command,
    root: &std::path::Path,
    argv: &[String],
    stats_json: Option<&std::path::Path>,
    out: Option<&std::path::Path>,
) -> Result<i32> {
    let started = Instant::now();
    if matches!(command, Command::Request(Request::Build))
        && crate::substrate::noop::active(stats_json)
        && let Some(reached) = crate::substrate::noop::try_hit(root, out)
    {
        let ms = started.elapsed().as_secs_f64() * 1e3;
        match out {
            Some(path) => eprintln!(
                "built {} ({reached} functions reached, {ms:.1} ms)",
                path.display()
            ),
            None => eprintln!("{reached} functions reached, 0 failed, built in {ms:.1} ms"),
        }
        return Ok(0);
    }
    let targets = Targets {
        target: cfg::Target::of(crate::compile_target()),
        host: cfg::Target::host(),
    };
    let graph_clock = sema::census::Clock::now();
    let mut workspace = match command {
        Command::Debug(Query::Graph) => graph::load(root, &targets, &crate::selection())?,
        _ => load_workspace(root, &targets)?,
    };
    let graph_span = graph_clock.elapsed();
    let loaded = started.elapsed();
    let status = match command {
        Command::Request(request) => report_cases(
            root,
            (&workspace, &targets),
            (request, argv),
            stats_json.map(|path| (path, graph_span)),
            out,
        )?,
        Command::Serve(ServeAction::Run) => {
            crate::serve::serve(&mut workspace, &targets, root).map(|()| 0)?
        }
        Command::Serve(_) => 0,
        Command::Prep => report_prep(&workspace, &targets).map(|()| 0)?,
        Command::Clean => 0,
        Command::Debug(query) => report_query(query, (&workspace, &targets), argv).map(|()| 0)?,
    };
    if !matches!(command, Command::Request(Request::Run)) {
        eprintln!(
            "graph {} packages {} units in {:.1} ms",
            workspace.packages.len(),
            workspace.graph.units.len(),
            loaded.as_secs_f64() * 1e3
        );
    }
    Ok(status)
}
