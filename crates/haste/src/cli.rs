use std::path::PathBuf;
use std::time::Instant;

use crate::graph::{self, Targets, cfg};
use crate::names::{ItemRef, Names};
use crate::serve::Request;
use crate::{outline, sema};
use anyhow::{Context, Result};

fn print_graph(workspace: &graph::Workspace) {
    for line in workspace.unit_lines() {
        println!("{line}");
    }
}

fn print_outline(workspace: &graph::Workspace, outline: &outline::Outline) {
    for (root, krate) in outline.roots.iter().zip(&outline.crates) {
        let unit = &workspace.graph.units[root.unit];
        let version = &workspace.packages[unit.package].locked.version;
        let read = krate.files.iter().filter(|file| !file.expanded).map(|file| &file.source.path).chain(&krate.gated);
        let mut files: Vec<String> = read.map(|path| path.display().to_string()).collect();
        files.sort();
        files.dedup();
        for file in files {
            println!("{:?} {} {version} {file}", unit.side, root.name);
        }
    }
}

fn print_items(path: &std::path::Path) -> Result<()> {
    let file = outline::tree::Files::default().file(path)?;
    for item in &file.items.items {
        let name = item.name.map_or("", |at| file.source.text(at));
        println!("{:?} {:?} {name} parent={:?} attrs={:?}", item.kind, item.vis, item.parent, item.attrs);
    }
    Ok(())
}

fn print_expanded(path: &std::path::Path) -> Result<()> {
    let mut outline = outline::Outline::file(path, cfg::Target::host());
    outline.load_all()?;
    let krate = &outline.crates[0];
    for file in krate.files.iter().filter(|file| file.expanded) {
        println!("{}", file.source.text);
    }
    for error in &krate.errors {
        eprintln!("error: {error}");
    }
    Ok(())
}

fn summary(outline: &outline::Outline) -> String {
    let items: usize = outline.crates.iter().flat_map(|krate| &krate.files).map(|file| file.items.items.len()).sum();
    let loaded: usize = outline.crates.iter().map(|krate| krate.modules.iter().filter(|module| module.loaded()).count()).sum();
    let expansions: usize = outline.crates.iter().map(|krate| krate.expansions).sum();
    let files = outline.crates.iter().flat_map(|krate| &krate.files).filter(|file| !file.expanded).count();
    format!("{} crates {files} files {loaded} modules {items} items {expansions} expansions", outline.crates.len())
}

fn report_outline(workspace: &graph::Workspace, targets: &Targets, threads: usize) -> Result<()> {
    let started = Instant::now();
    let compilation = crate::macros::load_eager(workspace, targets, &[])?;
    let (outline, names) = (&compilation.outline, &compilation.names);
    let elapsed = started.elapsed();
    for error in outline.crates.iter().flat_map(|krate| &krate.errors) {
        eprintln!("error: {error}");
    }
    let named = outline.roots.iter().zip(&outline.crates).flat_map(|(root, krate)| krate.unresolved().into_iter().map(move |name| format!("{}:{name}", root.name)));
    let mut unresolved: Vec<String> = named.collect();
    unresolved.sort();
    unresolved.dedup();
    eprintln!("unresolved macros {unresolved:?}");
    let foreign: usize = outline.crates.iter().map(|krate| krate.foreign.len()).sum();
    eprintln!("procedural macro calls {foreign}");
    eprintln!("outline {} in {:.1} ms", summary(outline), elapsed.as_secs_f64() * 1e3);
    report_names(outline, names, started);
    print_outline(workspace, outline);
    report_bodies(outline, threads);
    Ok(())
}

fn report_lazy(workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
    let started = Instant::now();
    let compilation = crate::macros::load_lazy(workspace, targets)?;
    eprintln!("lazy {} cached {} in {:.1} ms", summary(&compilation.outline), compilation.outline.cache.len(), started.elapsed().as_secs_f64() * 1e3);
    report_names(&compilation.outline, &compilation.names, started);
    Ok(())
}

pub(crate) fn case_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    let mut found = Vec::new();
    for krate in outline.members(workspace) {
        let tree = &outline.crates[krate];
        for (file_at, file) in tree.files.iter().enumerate() {
            for (at, item) in file.items.items.iter().enumerate() {
                let named = item.name.is_some_and(|name| file.source.text(name).starts_with("case_"));
                if named && item.kind == outline::item::ItemKind::Fn {
                    found.push(ItemRef { krate: krate as u32, file: file_at as u32, item: at as u32 });
                }
            }
        }
    }
    found
}

pub(crate) type Failures = std::collections::BTreeMap<String, Vec<String>>;

fn typeck_pass(cx: &mut sema::Context<'_>, cases: &[ItemRef]) -> Failures {
    let mut failures = Failures::new();
    for &item in cases {
        if let Err(error) = sema::infer::infer(cx, item, &[]) {
            let text = format!("{error:#}");
            let kind = text.split([' ', '(', '`']).take(3).collect::<Vec<_>>().join(" ");
            failures.entry(kind).or_default().push(cx.name(item).to_owned());
        }
    }
    failures
}

/// Loads and resolves what a sema pass asked for, and returns the store for the next pass and whether the outline changed.
fn report_typeck(workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
    let started = Instant::now();
    let mut compilation = crate::macros::load_lazy(workspace, targets)?;
    let load = started.elapsed();
    let cases = case_items(workspace, &compilation.outline);
    let mut passes = 0;
    let failures = crate::macros::resolve_sema(&mut compilation, (sema::Cache::default(), crate::macros::load), |mut cx, last| {
        passes += 1;
        let failures = typeck_pass(&mut cx, &cases);
        let settled = cx.resolved();
        let released = cx.release();
        Ok(if settled || last { Ok(failures) } else { Err(released) })
    })?;
    let failed: usize = failures.values().map(Vec::len).sum();
    print_failures(&failures);
    let ms = |at: std::time::Duration| at.as_secs_f64() * 1e3;
    eprintln!("typeck {} cases, {} passed, {failed} failed, {passes} passes, {}, load {:.1} ms, total {:.1} ms", cases.len(), cases.len() - failed, summary(&compilation.outline), ms(load), ms(started.elapsed()));
    Ok(())
}

fn report_cases(workspace: &graph::Workspace, targets: &Targets, request: Request, args: &[String]) -> Result<i32> {
    if request == Request::Run {
        return crate::build::run(workspace, targets, args);
    }
    let started = Instant::now();
    let mut build = crate::build::Build::load(workspace, targets)?;
    let result = crate::serve::finish(&mut build, workspace, (request, args), started);
    std::mem::forget(build);
    result
}

pub(crate) fn print_failures(failures: &Failures) {
    for (error, items) in failures {
        let names: Vec<&str> = items.iter().take(3).filter_map(|item| item.split(':').next()).collect();
        eprintln!("{:>4} {error} [{}]\n       {}", items.len(), names.join(" "), items[0].chars().take(300).collect::<String>());
    }
}

pub(crate) fn error_kind(text: &str) -> String {
    let text = text.split_once(": ").map_or(text, |(_, rest)| rest);
    text.split([' ', '(', '`']).take(4).collect::<Vec<_>>().join(" ")
}

fn parse_bodies(tree: &outline::tree::Crate) -> (usize, Vec<String>) {
    let mut parsed = 0;
    let mut errors = Vec::new();
    for file in &tree.files {
        for item in &file.items.items {
            let Some(open) = item.body.filter(|_| item.kind == outline::item::ItemKind::Fn) else {
                continue;
            };
            match crate::body::parse(&file.source, open, crate::body::Config::default()) {
                Ok(_) => parsed += 1,
                Err(error) => errors.push(format!("{error:#}")),
            }
        }
    }
    (parsed, errors)
}

fn report_bodies(outline: &outline::Outline, threads: usize) {
    let started = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some(tree) = outline.crates.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                    let result = parse_bodies(tree);
                    crate::sync::lock(&results).push(result);
                }
            });
        }
    });
    let results = crate::sync::into_inner(results);
    let parsed: usize = results.iter().map(|(parsed, _)| parsed).sum();
    let errors: Vec<&String> = results.iter().flat_map(|(_, errors)| errors).collect();
    for error in &errors {
        eprintln!("body error: {error}");
    }
    eprintln!("bodies {parsed} parsed, {} failed in {:.1} ms", errors.len(), started.elapsed().as_secs_f64() * 1e3);
}

fn report_names(outline: &outline::Outline, names: &Names, started: Instant) {
    let elapsed = started.elapsed();
    let mut total = 0;
    for (index, krate) in names.crates.iter().enumerate() {
        total += krate.unresolved.len();
        let tree = &outline.crates[index];
        for entry in krate.unresolved.iter().take(3) {
            let file = &tree.files[entry.file as usize];
            let import = &file.items.imports[entry.import as usize];
            let path: Vec<&str> = import.path.iter().map(|at| file.source.text(*at)).collect();
            eprintln!("unresolved {} in {}: {}", outline.roots[index].name, file.source.path.display(), path.join("::"));
        }
    }
    let bindings: usize = names.crates.iter().flat_map(|krate| &krate.scopes).map(|scope| scope.names.len()).sum();
    eprintln!("names {bindings} names, {total} unresolved imports in {:.1} ms", elapsed.as_secs_f64() * 1e3);
}

/// Prints the root scope of each crate named `name`: every name with a flag per namespace, types, values and macros.
fn print_scope(outline: &outline::Outline, names: &Names, name: &str) {
    for (index, krate) in names.crates.iter().enumerate().filter(|(index, _)| outline.roots[*index].name == name) {
        let mut keys: Vec<String> =
            krate.scopes[0].names.iter().map(|(name, entry)| format!("{}:{}{}{}", names.symbols.text(*name), u8::from(entry.types.is_some()), u8::from(entry.values.is_some()), u8::from(entry.macros.is_some()))).collect();
        keys.sort();
        println!("scope {index} {}: {}", outline.roots[index].name, keys.join(" "));
    }
}

const USAGE: &str = "usage: haste [--jobs N] build|test|run|serve PATH [ARGS...]\n       haste [--jobs N] debug graph|outline|lazy|typeck|scope|items|expand PATH [CRATE]";

/// What the command line asks for.
#[derive(Clone, Copy)]
enum Command {
    /// Build, test or run the workspace, through its server when one runs.
    Request(Request),
    /// Keep a build of the workspace in memory and answer requests.
    Serve,
    /// Print what one stage found.
    Debug(Query),
}

/// What a debug command prints.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Query {
    Graph,
    Outline,
    Lazy,
    Typeck,
    Scope,
    Items,
    Expand,
}

const QUERIES: [(Query, &str); 7] = [(Query::Graph, "graph"), (Query::Outline, "outline"), (Query::Lazy, "lazy"), (Query::Typeck, "typeck"), (Query::Scope, "scope"), (Query::Items, "items"), (Query::Expand, "expand")];

fn parse(args: &mut impl Iterator<Item = String>) -> Result<Command> {
    let word = args.next().context(USAGE)?;
    match word.as_str() {
        "serve" => Ok(Command::Serve),
        "debug" => {
            let name = args.next().context(USAGE)?;
            QUERIES.into_iter().find(|(_, query)| *query == name).map(|(query, _)| Command::Debug(query)).context(USAGE)
        }
        _ => Request::parse_mode(&word).map(Command::Request).context(USAGE),
    }
}

/// Takes a leading `--jobs N` off `args` and sets the job count from it.
fn take_jobs(args: &mut Vec<String>) -> Result<()> {
    if args.first().map(String::as_str) != Some("--jobs") {
        return Ok(());
    }
    let count: usize = args.get(1).context(USAGE)?.parse().context("--jobs takes a positive number")?;
    crate::set_jobs(count.max(1));
    args.drain(..2);
    Ok(())
}

/// The glibc malloc settings haste runs with: transparent huge pages for the heap, and freed memory kept for reuse instead of returned to the kernel and faulted in again.
const TUNABLES: &str = "glibc.malloc.hugetlb=1:glibc.malloc.mmap_threshold=4294967296:glibc.malloc.trim_threshold=4294967296:glibc.malloc.top_pad=268435456";

/// Starts haste again with `TUNABLES` when its environment sets no tunables, since glibc reads them only at startup; the started process takes them out of its environment, so the programs it runs do not see them.
fn tune_malloc() {
    if !cfg!(all(target_os = "linux", target_env = "gnu")) {
        return;
    }
    match std::env::var_os("GLIBC_TUNABLES") {
        // SAFETY: no other thread runs yet, so nothing reads the environment while it changes.
        Some(set) if set == TUNABLES => unsafe { std::env::remove_var("GLIBC_TUNABLES") },
        Some(_) => {}
        None => {
            use std::os::unix::process::CommandExt;
            let mut args = std::env::args_os();
            let arg0 = args.next().unwrap_or_default();
            let error = std::process::Command::new("/proc/self/exe").arg0(arg0).args(args).env("GLIBC_TUNABLES", TUNABLES).exec();
            eprintln!("haste: running without malloc tunables: {error}");
        }
    }
}

/// Runs the haste command line.
pub fn main() -> Result<()> {
    tune_malloc();
    crate::sema::evaluate::install(crate::image::evaluator);
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    take_jobs(&mut args)?;
    let mut args = args.into_iter();
    let command = parse(&mut args)?;
    let path = PathBuf::from(args.next().context(USAGE)?);
    match command {
        Command::Debug(Query::Items) => return print_items(&path),
        Command::Debug(Query::Expand) => return print_expanded(&path),
        _ => {}
    }
    let root = std::path::absolute(path)?;
    let argv: Vec<String> = std::iter::once(program_name(&root)).chain(args).collect();
    if let Command::Request(request) = command
        && let Some(status) = crate::serve::ask(&root, request, &argv)?
    {
        std::process::exit(status);
    }
    match report(command, &root, &argv)? {
        0 => Ok(()),
        status => std::process::exit(status),
    }
}

fn report_query(query: Query, (workspace, targets): (&graph::Workspace, &Targets), argv: &[String]) -> Result<()> {
    match query {
        Query::Graph => Ok(print_graph(workspace)),
        Query::Outline => report_outline(workspace, targets, crate::jobs()),
        Query::Lazy => report_lazy(workspace, targets),
        Query::Typeck => report_typeck(workspace, targets),
        Query::Scope => {
            let name = argv.get(1).context(USAGE)?;
            let compilation = crate::macros::load_eager(workspace, targets, &[])?;
            print_scope(&compilation.outline, &compilation.names, name);
            Ok(())
        }
        Query::Items | Query::Expand => Ok(()),
    }
}

fn program_name(root: &std::path::Path) -> String {
    root.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// Loads the workspace at `root` with the standard library merged in and the output of every build script settled.
pub(crate) fn load_workspace(root: &std::path::Path, targets: &Targets) -> Result<graph::Workspace> {
    let mut workspace = graph::load(root, targets)?;
    workspace.merge(graph::load_sysroot(&graph::library()?, targets)?);
    crate::scripts::settle(&mut workspace, targets)?;
    Ok(workspace)
}

fn report(command: Command, root: &std::path::Path, argv: &[String]) -> Result<i32> {
    let targets = Targets { target: cfg::Target::host(), host: cfg::Target::host() };
    let started = Instant::now();
    let mut workspace = match command {
        Command::Debug(Query::Graph) => graph::load(root, &targets)?,
        _ => load_workspace(root, &targets)?,
    };
    let loaded = started.elapsed();
    let status = match command {
        Command::Request(request) => report_cases(&workspace, &targets, request, argv)?,
        Command::Serve => crate::serve::serve(&mut workspace, &targets, root).map(|()| 0)?,
        Command::Debug(query) => report_query(query, (&workspace, &targets), argv).map(|()| 0)?,
    };
    eprintln!("graph {} packages {} units in {:.1} ms", workspace.packages.len(), workspace.graph.units.len(), loaded.as_secs_f64() * 1e3);
    Ok(status)
}
