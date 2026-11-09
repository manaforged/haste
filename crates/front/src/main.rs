use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use haste_front::graph::{self, Targets, cfg};
use haste_front::names::{ItemRef, Names};
use haste_front::{outline, sema};

mod cases;
mod serve;

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
    let file = outline::tree::Cache::default().file(path)?;
    for item in &file.items.items {
        let name = item.name.map_or("", |at| file.source.text(at));
        println!("{:?} {:?} {name} parent={:?} attrs={:?}", item.kind, item.vis, item.parent, item.attrs);
    }
    Ok(())
}

fn print_expanded(path: &std::path::Path) -> Result<()> {
    let mut outline = outline::Outline::file(path, cfg::Target::x86_64_linux_gnu());
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
    let (outline, names) = eager_front(workspace, targets)?;
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
    eprintln!("outline {} in {:.1} ms", summary(&outline), elapsed.as_secs_f64() * 1e3);
    report_names(&outline, &names, started);
    print_outline(workspace, &outline);
    report_bodies(&outline, threads);
    Ok(())
}

/// Loads every module of every crate and resolves every import.
fn eager_front(workspace: &graph::Workspace, targets: &Targets) -> Result<(outline::Outline, Names)> {
    let mut outline = outline::new(workspace, targets);
    outline.load_all()?;
    let mut names = Names::new(workspace, &outline);
    names.crates.iter_mut().for_each(|krate| krate.eager = true);
    names.settle(&mut outline)?;
    Ok((outline, names))
}

/// Loads the member crates' roots and settles names, reading only what their lookups reach.
fn demand_front(workspace: &graph::Workspace, targets: &Targets) -> Result<(outline::Outline, Names)> {
    let mut outline = outline::new(workspace, targets);
    let members: Vec<usize> = (0..outline.roots.len()).filter(|at| is_member(workspace, &outline.roots[*at])).collect();
    for &krate in &members {
        outline.load(krate, 0)?;
    }
    let mut names = Names::new(workspace, &outline);
    members.iter().for_each(|krate| names.crates[*krate].eager = true);
    names.settle(&mut outline)?;
    Ok((outline, names))
}

fn is_member(workspace: &graph::Workspace, root: &outline::Root) -> bool {
    workspace.graph.members.contains(&workspace.graph.units[root.unit].package)
}

fn report_demand(workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
    let started = Instant::now();
    let (outline, names) = demand_front(workspace, targets)?;
    eprintln!("demand {} cached {} in {:.1} ms", summary(&outline), outline.cache.len(), started.elapsed().as_secs_f64() * 1e3);
    report_names(&outline, &names, started);
    Ok(())
}

fn case_items(workspace: &graph::Workspace, outline: &outline::Outline) -> Vec<ItemRef> {
    let mut found = Vec::new();
    for (krate, (_, tree)) in outline.roots.iter().zip(&outline.crates).enumerate().filter(|(_, (root, _))| is_member(workspace, root)) {
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

type Failures = std::collections::BTreeMap<String, Vec<String>>;

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
fn answer(names: &mut Names, outline: &mut outline::Outline, (mut store, asks): (sema::Store, haste_front::names::path::Asks)) -> Result<(sema::Store, bool)> {
    let more = names.answer(outline, asks)?;
    store.forget_lookups();
    Ok((store, more))
}

fn report_typeck(workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
    let started = Instant::now();
    let (mut outline, mut names) = demand_front(workspace, targets)?;
    let front = started.elapsed();
    let cases = case_items(workspace, &outline);
    let (mut store, mut passes) = (sema::Store::default(), 0);
    let failures = loop {
        passes += 1;
        let mut cx = sema::Context::new(&outline, &names, store);
        let failures = typeck_pass(&mut cx, &cases);
        let released = cx.release();
        let more;
        (store, more) = answer(&mut names, &mut outline, released)?;
        if !more {
            break failures;
        }
    };
    let failed: usize = failures.values().map(Vec::len).sum();
    print_failures(&failures);
    let ms = |at: std::time::Duration| at.as_secs_f64() * 1e3;
    eprintln!("typeck {} cases, {} passed, {failed} failed, {passes} passes, {}, front {:.1} ms, total {:.1} ms", cases.len(), cases.len() - failed, summary(&outline), ms(front), ms(started.elapsed()));
    Ok(())
}

fn report_cases(workspace: &graph::Workspace, targets: &Targets, request: serve::Request) -> Result<()> {
    let started = Instant::now();
    let build = cases::Build::cold(workspace, targets)?;
    let result = serve::finish(&build, request, started);
    std::mem::forget(build);
    result
}

fn print_failures(failures: &Failures) {
    for (error, items) in failures {
        let names: Vec<&str> = items.iter().take(3).filter_map(|item| item.split(':').next()).collect();
        eprintln!("{:>4} {error} [{}]\n       {}", items.len(), names.join(" "), items[0].chars().take(300).collect::<String>());
    }
}

fn error_kind(text: &str) -> String {
    let text = text.split_once(": ").map_or(text, |(_, rest)| rest);
    text.split([' ', '(', '`']).take(4).collect::<Vec<_>>().join(" ")
}

fn parse_bodies(tree: &outline::tree::Crate) -> (usize, Vec<String>) {
    let mut parsed = 0;
    let mut errors = Vec::new();
    for file in &tree.files {
        for item in &file.items.items {
            let Some(open) = item.body.filter(|_| item.kind == outline::item::ItemKind::Fn) else { continue };
            match haste_front::body::parse(&file.source, open, haste_front::body::Config::default()) {
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
                    results.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(result);
                }
            });
        }
    });
    let results = results.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner);
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
    if let Ok(dump) = std::env::var("HASTE_DUMP") {
        for (index, krate) in names.crates.iter().enumerate().filter(|(index, _)| outline.roots[*index].name == dump) {
            let mut keys: Vec<String> = krate.scopes[0].names.iter().map(|(name, entry)| format!("{}:{}{}{}", names.symbols.text(*name), u8::from(entry.types.is_some()), u8::from(entry.values.is_some()), u8::from(entry.macros.is_some()))).collect();
            keys.sort();
            eprintln!("dump {index} {}: {}", outline.roots[index].name, keys.join(" "));
        }
    }
    let bindings: usize = names.crates.iter().flat_map(|krate| &krate.scopes).map(|scope| scope.names.len()).sum();
    eprintln!("names {bindings} names, {total} unresolved imports in {:.1} ms", elapsed.as_secs_f64() * 1e3);
}

const USAGE: &str = "usage: haste-front graph|outline|demand|typeck|cases|build|serve|items|expand PATH";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().context(USAGE)?;
    let path = PathBuf::from(args.next().context(USAGE)?);
    match mode.as_str() {
        "items" => print_items(&path),
        "expand" => print_expanded(&path),
        _ => {
            let root = std::path::absolute(path)?;
            if let Some(status) = serve::Request::parse_mode(&mode).map(|request| serve::ask(&root, request)).transpose()?.flatten() {
                std::process::exit(status);
            }
            report(&mode, &root)
        }
    }
}

fn report(mode: &str, root: &std::path::Path) -> Result<()> {
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let targets = Targets { target: cfg::Target::x86_64_linux_gnu(), host: cfg::Target::x86_64_linux_gnu() };
    let started = Instant::now();
    let mut workspace = graph::load(root, &targets)?;
    if mode != "graph" {
        workspace.merge(graph::load_sysroot(&graph::library()?, &targets)?);
    }
    let loaded = started.elapsed();
    match (mode, serve::Request::parse_mode(mode)) {
        (_, Some(request)) => report_cases(&workspace, &targets, request),
        ("graph", _) => Ok(print_graph(&workspace)),
        ("outline", _) => report_outline(&workspace, &targets, threads),
        ("demand", _) => report_demand(&workspace, &targets),
        ("typeck", _) => report_typeck(&workspace, &targets),
        ("serve", _) => serve::serve(&workspace, &targets, root),
        (other, _) => Err(anyhow::anyhow!("unknown mode {other}")),
    }?;
    eprintln!("graph {} packages {} units in {:.1} ms", workspace.packages.len(), workspace.graph.units.len(), loaded.as_secs_f64() * 1e3);
    Ok(())
}
