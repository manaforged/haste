use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use haste_front::graph::{self, Targets, cfg};
use haste_front::outline;

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
    let mut outline = outline::new(workspace, targets);
    outline.load_all()?;
    let elapsed = started.elapsed();
    let started = Instant::now();
    let mut names = haste_front::names::Names::new(workspace, &outline);
    names.crates.iter_mut().for_each(|krate| krate.eager = true);
    names.settle(&mut outline)?;
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

fn report_demand(workspace: &graph::Workspace, targets: &Targets) -> Result<()> {
    let started = Instant::now();
    let mut outline = outline::new(workspace, targets);
    let members: Vec<usize> = (0..outline.roots.len()).filter(|at| workspace.graph.members.contains(&workspace.graph.units[outline.roots[*at].unit].package)).collect();
    for &krate in &members {
        outline.load(krate, 0)?;
    }
    let mut names = haste_front::names::Names::new(workspace, &outline);
    members.iter().for_each(|krate| names.crates[*krate].eager = true);
    names.settle(&mut outline)?;
    eprintln!("demand {} in {:.1} ms", summary(&outline), started.elapsed().as_secs_f64() * 1e3);
    report_names(&outline, &names, started);
    Ok(())
}

fn parse_bodies(tree: &outline::tree::Crate) -> (usize, Vec<String>) {
    let mut parsed = 0;
    let mut errors = Vec::new();
    for file in &tree.files {
        for item in &file.items.items {
            let Some(open) = item.body.filter(|_| item.kind == outline::item::ItemKind::Fn) else { continue };
            match haste_front::body::parse(&file.source, open) {
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
                    results.lock().map(|mut results| results.push(result)).ok();
                }
            });
        }
    });
    let results = results.into_inner().unwrap_or_default();
    let parsed: usize = results.iter().map(|(parsed, _)| parsed).sum();
    let errors: Vec<&String> = results.iter().flat_map(|(_, errors)| errors).collect();
    for error in &errors {
        eprintln!("body error: {error}");
    }
    eprintln!("bodies {parsed} parsed, {} failed in {:.1} ms", errors.len(), started.elapsed().as_secs_f64() * 1e3);
}

fn report_names(outline: &outline::Outline, names: &haste_front::names::Names, started: Instant) {
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

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().context("usage: haste-front graph|outline|items PATH")?;
    match mode.as_str() {
        "items" => return print_items(&PathBuf::from(args.next().context("usage: haste-front items FILE")?)),
        "expand" => return print_expanded(&PathBuf::from(args.next().context("usage: haste-front expand FILE")?)),
        _ => {}
    }
    let root = std::path::absolute(PathBuf::from(args.next().context("usage: haste-front graph|outline PATH")?))?;
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let targets = Targets { target: cfg::Target::x86_64_linux_gnu(), host: cfg::Target::x86_64_linux_gnu() };
    let started = Instant::now();
    let mut workspace = graph::load(&root, &targets)?;
    if mode != "graph" {
        workspace.merge(graph::load_sysroot(&graph::library()?, &targets)?);
    }
    let loaded = started.elapsed();
    match mode.as_str() {
        "graph" => print_graph(&workspace),
        "outline" => report_outline(&workspace, &targets, threads)?,
        "demand" => report_demand(&workspace, &targets)?,
        other => bail!("unknown mode {other}"),
    }
    eprintln!("graph {} packages {} units in {:.1} ms", workspace.packages.len(), workspace.graph.units.len(), loaded.as_secs_f64() * 1e3);
    Ok(())
}
