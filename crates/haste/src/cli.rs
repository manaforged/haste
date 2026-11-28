mod args;
mod debug;
mod help;

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
    let file = &tree.files[file_at];
    let items = &file.items.items;
    if !tree.env.test && is_test_only_fn(tree, file, &items[at]) {
        return false;
    }
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

fn crate_items<'a>(
    krate: usize,
    tree: &'a outline::tree::Crate,
    include_consts: bool,
) -> impl Iterator<
    Item = (
        ItemRef,
        &'a outline::tree::Crate,
        &'a outline::tree::File,
        &'a outline::item::Item,
    ),
> + 'a {
    tree.files
        .iter()
        .enumerate()
        .flat_map(move |(file_at, file)| {
            file.items
                .items
                .iter()
                .enumerate()
                .filter_map(move |(at, item)| {
                    let match_kind = (item.kind == outline::item::ItemKind::Fn
                        && item.body.is_some())
                        || (include_consts && item.kind == outline::item::ItemKind::Const);
                    (match_kind && item_enabled(tree, file_at, at)).then_some((
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
}

fn crate_fns<'a>(
    krate: usize,
    tree: &'a outline::tree::Crate,
) -> impl Iterator<
    Item = (
        ItemRef,
        &'a outline::tree::Crate,
        &'a outline::tree::File,
        &'a outline::item::Item,
    ),
> + 'a {
    crate_items(krate, tree, false)
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
    outline
        .members(workspace)
        .flat_map(move |krate| crate_fns(krate, &outline.crates[krate]))
}

fn target_name_matches(root_name: &str, selected: &str) -> bool {
    root_name == selected || root_name.replace('-', "_") == selected.replace('-', "_")
}

fn test_target_selected(outline: &outline::Outline, krate: usize, sel: &crate::Selection) -> bool {
    let test_name = sel.test.as_deref();
    let bin_name = sel.bin.as_deref();
    if !(sel.lib || sel.bins || test_name.is_some() || bin_name.is_some()) {
        return true;
    }
    let root = &outline.roots[krate];
    let is_lib_root = root.role == outline::Role::Lib
        || (root.role == outline::Role::Test
            && outline.roots.iter().any(|r| {
                r.role == outline::Role::Lib && r.unit == root.unit && r.path == root.path
            }));
    (sel.lib && is_lib_root)
        || (sel.bins && root.role == outline::Role::Bin)
        || bin_name
            .is_some_and(|n| root.role == outline::Role::Bin && target_name_matches(&root.name, n))
        || test_name.is_some_and(|n| {
            root.role == outline::Role::Test && !is_lib_root && target_name_matches(&root.name, n)
        })
}

fn item_coord(
    tree: &outline::tree::Crate,
    file: &outline::tree::File,
    item: &outline::item::Item,
) -> Option<(u32, u32)> {
    let fidx = tree
        .files
        .iter()
        .position(|f| std::ptr::eq(f.as_ref(), file))? as u32;
    let idx = file
        .items
        .items
        .iter()
        .position(|it| std::ptr::eq(it, item))? as u32;
    Some((fidx, idx))
}

fn is_test_fn(tree: &outline::tree::Crate, item_ref: ItemRef) -> bool {
    tree.resolved_tests
        .contains(&(item_ref.file, item_ref.item))
}

pub(crate) fn is_test_only_fn(
    tree: &outline::tree::Crate,
    file: &outline::tree::File,
    item: &outline::item::Item,
) -> bool {
    item.kind == outline::item::ItemKind::Fn
        && item_coord(tree, file, item).is_some_and(|coord| {
            tree.consumed.contains(&coord) || tree.resolved_tests.contains(&coord)
        })
}

/// Counts enabled `#[test]` functions across loaded member test crates matching target selection.
pub(crate) fn outline_test_count(outline: &outline::Outline) -> usize {
    let sel = crate::selection();
    (0..outline.crates.len())
        .filter(|&krate| {
            outline.crates[krate].env.test && test_target_selected(outline, krate, &sel)
        })
        .flat_map(|krate| crate_fns(krate, &outline.crates[krate]))
        .filter(|(item_ref, tree, _, _)| is_test_fn(tree, *item_ref))
        .count()
}

pub(crate) fn test_items(
    workspace: &graph::Workspace,
    outline: &outline::Outline,
) -> Vec<TestItem> {
    let sel = crate::selection();
    member_fns(workspace, outline)
        .filter(|(item_ref, tree, _, _)| {
            tree.env.test
                && test_target_selected(outline, item_ref.krate as usize, &sel)
                && is_test_fn(tree, *item_ref)
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

/// Internal `run_test_captured` exit status indicating a test unwound via a Rust panic.
pub(crate) const TEST_PANICKED_STATUS: i32 = -101;

/// Outcome of running a single `#[test]` function.
pub(crate) struct TestRun {
    pub status: i32,
    pub output: Option<String>,
    pub panic_msg: Option<String>,
}

#[cfg(unix)]
fn is_panic_header(line: &str) -> bool {
    let Some((head, loc)) = line.rsplit_once(" panicked at ") else {
        return false;
    };
    let Some(after_prefix) = head.strip_prefix("thread '") else {
        return false;
    };
    let valid_thread = after_prefix.ends_with('\'')
        || after_prefix.rsplit_once("' (").is_some_and(|(_, tid)| {
            tid.strip_suffix(')')
                .is_some_and(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_alphanumeric()))
        });
    if !valid_thread {
        return false;
    }
    let Some(loc) = loc.strip_suffix(':') else {
        return false;
    };
    let mut parts = loc.rsplitn(3, ':');
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(col), Some(line_num), Some(file))
            if !file.is_empty()
                && !line_num.is_empty()
                && !col.is_empty()
                && line_num.bytes().all(|b| b.is_ascii_digit())
                && col.bytes().all(|b| b.is_ascii_digit())
    )
}

#[cfg(unix)]
pub(crate) fn panic_payload(stderr: &str) -> Option<&str> {
    let mut last_rest = None;
    let mut offset = 0;
    for line in stderr.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if is_panic_header(trimmed) {
            last_rest = Some(&stderr[offset + line.len()..]);
        }
        offset += line.len();
    }
    let rest = last_rest?;
    let body = rest.split("\nnote: run with ").next().unwrap_or(rest);
    let body = body.split("\nstack backtrace:").next().unwrap_or(body);
    Some(body.trim_end_matches(['\r', '\n']))
}

fn should_panic_non_panic_msg(status: i32, stderr: Option<&str>) -> String {
    let note = if status == 0 {
        "note: test did not panic as expected".to_owned()
    } else {
        format!("note: test exited with status {status} instead of panicking")
    };
    match stderr.map(str::trim).filter(|s| !s.is_empty()) {
        Some(out) => format!("{note}\n{out}"),
        None => note,
    }
}

/// Formats and prints standard Rust test output and returns 0 when all tests pass, else 101.
pub(crate) fn print_test_report(
    entries: &[(&str, &TestItem, Option<Result<TestRun, &str>>)],
    elapsed: std::time::Duration,
    show_output: bool,
    filtered_out: usize,
) -> i32 {
    println!(
        "\nrunning {} test{}",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" }
    );
    let (mut passed, mut failed, mut ignored) = (0usize, 0usize, 0usize);
    let mut details: Vec<(&str, String)> = Vec::new();
    let mut successes: Vec<(&str, &str)> = Vec::new();
    for &(name, meta, ref outcome) in entries {
        let Some(outcome) = outcome else {
            println!("test {name} ... ignored");
            ignored += 1;
            continue;
        };
        match outcome {
            Ok(run) => {
                let panicked = run.status == TEST_PANICKED_STATUS;
                let expected_ok = match (&meta.expected, &run.panic_msg) {
                    (Some(exp), Some(msg)) => msg.contains(exp.as_str()),
                    (Some(_), None) => false,
                    (None, _) => true,
                };
                let ok = if meta.should_panic {
                    panicked && expected_ok
                } else {
                    run.status == 0
                };
                if ok {
                    println!("test {name} ... ok");
                    if show_output
                        && let Some(out) = run
                            .output
                            .as_deref()
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                    {
                        successes.push((name, out));
                    }
                    passed += 1;
                } else {
                    let msg = if meta.should_panic && !panicked {
                        should_panic_non_panic_msg(run.status, run.output.as_deref())
                    } else if let Some(exp) = &meta.expected {
                        format!(
                            "note: panic did not contain expected string `{exp}`\n{}",
                            run.output.as_deref().unwrap_or("").trim()
                        )
                    } else if let Some(out) = run
                        .output
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                    {
                        out.to_owned()
                    } else if panicked {
                        "test panicked".to_owned()
                    } else {
                        format!("exited with status {}", run.status)
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
    if !successes.is_empty() {
        println!("\nsuccesses:\n");
        for (name, out) in &successes {
            println!("---- {name} output ----\n{out}\n");
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
        "\ntest result: {verdict}. {passed} passed; {failed} failed; {ignored} ignored; 0 measured; {filtered_out} filtered out; finished in {:.2}s\n",
        elapsed.as_secs_f64()
    );
    if failed == 0 { 0 } else { 101 }
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
    let line = text.lines().next().unwrap_or(text);
    let text = line.split_once(": ").map_or(line, |(_, rest)| rest);
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
    let mut build = crate::build::Build::load_oneshot(workspace, targets)?;
    let result = match (request, out) {
        (Request::Build, Some(path)) => build.emit_executable(workspace, path, stats.is_none()),
        (Request::Build, None) if stats.is_none() => build
            .compile_binary_oneshot(workspace)
            .and_then(|()| build.check(started.elapsed())),
        (Request::Test, None) => {
            let filters = args.get(1..).unwrap_or(&[]);
            build
                .compile_cases_filtered(workspace, filters)
                .and_then(|()| build.report(started, started.elapsed(), filters))
        }
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
                let watch_paths = build.watch_paths(workspace);
                let env_reads: Vec<String> = build
                    .env_reads()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                crate::substrate::noop::record(
                    root,
                    workspace,
                    &watch_paths,
                    &env_reads,
                    out,
                    build.reached_count(),
                );
                std::mem::forget((watch_paths, env_reads));
            }
            _ => crate::substrate::noop::invalidate(root, out),
        }
    }
    std::mem::forget(build);
    result
}

fn clean_cache() -> Result<()> {
    if crate::serve::stop(None)? != 0 {
        anyhow::bail!("failed to stop active haste daemons");
    }
    let home = crate::graph::store::home();
    let quiet = crate::selection().quiet;
    match std::fs::remove_dir_all(&home) {
        Ok(()) => {
            if !quiet {
                eprintln!("removed {}", home.display());
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            if !quiet {
                eprintln!("cache {} is already empty", home.display());
            }
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
        match run_parsed(command, path, rest, stats_json, out) {
            Ok(()) => Ok(()),
            Err(err) => {
                if crate::build::emit_early_json_error(&err) {
                    std::process::exit(1);
                }
                Err(err)
            }
        }
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
    if !crate::selection().quiet {
        println!(
            "{counts} in {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
    }
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
    graph::snapshot::load_or_build(root, targets)
}

fn report(
    command: Command,
    root: &std::path::Path,
    argv: &[String],
    stats_json: Option<&std::path::Path>,
    out: Option<&std::path::Path>,
) -> Result<i32> {
    let started = Instant::now();
    if let Some(path) = out {
        if path.is_dir() {
            anyhow::bail!("output path `{}` is a directory", path.display());
        }
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory `{}`", parent.display()))?;
        }
    }
    if matches!(command, Command::Request(Request::Build))
        && crate::substrate::noop::active(stats_json)
        && let Some(reached) = crate::substrate::noop::try_hit(root, out)
    {
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let sel = crate::selection();
        if !sel.quiet && sel.message_format != crate::MessageFormat::Json {
            match out {
                Some(path) => eprintln!(
                    "built {} ({reached} functions reached, {ms:.1} ms)",
                    path.display()
                ),
                None => eprintln!("{reached} functions reached, 0 failed, built in {ms:.1} ms"),
            }
        }
        crate::diag::print_build_finished(true);
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
    if matches!(command, Command::Debug(Query::Graph)) {
        eprintln!(
            "graph {} packages {} units in {:.1} ms",
            workspace.packages.len(),
            workspace.graph.units.len(),
            loaded.as_secs_f64() * 1e3
        );
    }
    if matches!(command, Command::Request(_)) {
        std::mem::forget(workspace);
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_outline() -> outline::Outline {
        outline::Outline {
            roots: vec![
                outline::Root {
                    unit: 0,
                    name: "my-bin".to_owned(),
                    path: std::path::PathBuf::from("src/bin/my-bin.rs"),
                    role: outline::Role::Bin,
                    edition: "2021".to_owned(),
                    proc_macro: false,
                },
                outline::Root {
                    unit: 0,
                    name: "my-test".to_owned(),
                    path: std::path::PathBuf::from("tests/my-test.rs"),
                    role: outline::Role::Test,
                    edition: "2021".to_owned(),
                    proc_macro: false,
                },
            ],
            crates: Vec::new(),
            deps: Vec::new(),
            cache: outline::tree::Files::default(),
            procs: rustc_hash::FxHashMap::default(),
            pins: Vec::new(),
        }
    }

    #[test]
    fn test_target_selected_matches_hyphenated_bin_and_test_names() {
        let outline = test_outline();
        let by_bin_hyphen = crate::Selection {
            bin: Some("my-bin".to_owned()),
            ..crate::Selection::default()
        };
        let by_bin_under = crate::Selection {
            bin: Some("my_bin".to_owned()),
            ..crate::Selection::default()
        };
        assert!(test_target_selected(&outline, 0, &by_bin_hyphen));
        assert!(test_target_selected(&outline, 0, &by_bin_under));
        assert!(!test_target_selected(&outline, 1, &by_bin_hyphen));

        let by_test_hyphen = crate::Selection {
            test: Some("my-test".to_owned()),
            ..crate::Selection::default()
        };
        let by_test_under = crate::Selection {
            test: Some("my_test".to_owned()),
            ..crate::Selection::default()
        };
        assert!(test_target_selected(&outline, 1, &by_test_hyphen));
        assert!(test_target_selected(&outline, 1, &by_test_under));
        assert!(!test_target_selected(&outline, 0, &by_test_hyphen));
    }
}
