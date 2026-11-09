//! Procedural macros in the image: the calls that name resolution binds are built into one image with haste's `proc_macro` crate, run in a forked child, and their output read back into the outline.

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::image::{Image, Job};
use crate::names::procs::{Bound, Site};
use crate::names::path::Asks;
use crate::names::{ItemRef, Names};
use crate::outline::Outline;
use crate::outline::item::{ItemKind, ProcKind};
use crate::outline::source::Source;
use crate::outline::tree::PROC_MACRO_ROOT;
use crate::graph::{Targets, Workspace};
use crate::sema::reach::{Instance, Reached, reach};
use crate::sema::{Context, Store};

const ENTRIES: [&str; 2] = ["__haste_expand1", "__haste_expand2"];

/// Loads every module of every crate, with the build scripts of the units `scripts`, resolves every import and expands every procedural macro call.
pub fn front(workspace: &Workspace, targets: &Targets, scripts: &[usize]) -> Result<(Outline, Names)> {
    let mut outline = crate::outline::new(workspace, targets, scripts);
    outline.load_all()?;
    let mut names = Names::new(workspace, &outline);
    names.crates.iter_mut().for_each(|krate| krate.eager = true);
    settle(&mut outline, &mut names)?;
    Ok((outline, names))
}

/// Reaches from the instances `roots` gives, answering what each pass asks for until a pass settles or nothing new can load, then hands that pass's context and what it reached to `finish`.
pub fn reach_settled<R>(outline: &mut Outline, names: &mut Names, roots: impl Fn(&mut Context<'_>) -> Vec<Instance>, finish: impl FnOnce(Context<'_>, Reached) -> Result<R>) -> Result<R> {
    reach_answered(outline, names, (roots, answer), finish)
}

/// Reaches from `roots` as `reach_settled` does, answering each pass's asks with `ask`, which says whether anything new loaded.
fn reach_answered<R>(outline: &mut Outline, names: &mut Names, (roots, ask): (impl Fn(&mut Context<'_>) -> Vec<Instance>, impl Fn(&mut Outline, &mut Names, Asks) -> Result<bool>), finish: impl FnOnce(Context<'_>, Reached) -> Result<R>) -> Result<R> {
    let (mut store, mut last) = (Store::default(), false);
    loop {
        let mut cx = Context::new(outline, names, store);
        let from = roots(&mut cx);
        let reached = reach(&mut cx, &from);
        if last || cx.settled() {
            return finish(cx, reached);
        }
        let (mut kept, asks) = cx.release();
        last = !ask(outline, names, asks)?;
        kept.forget_lookups();
        store = kept;
    }
}

/// Settles names, then expands every procedural macro call they bind and settles again, until no call is left to bind.
pub fn settle(outline: &mut Outline, names: &mut Names) -> Result<()> {
    loop {
        names.settle(outline)?;
        let bound = names.bind_procs(outline);
        if bound.is_empty() {
            return Ok(());
        }
        let inputs: Vec<Vec<String>> = bound.iter().map(inputs).collect();
        let calls: Vec<(ItemRef, ProcKind)> = bound.iter().map(|call| (call.run, call.kind)).collect();
        let outputs = run(outline, names, &calls, &inputs)?;
        apply(outline, names, &bound, outputs)?;
    }
}

/// Answers what a Sema pass asked for: runs the procedural macros its bodies call and keeps their output, loads and resolves the names it asked for, and expands the item macros that brings in; returns whether anything was new.
pub fn answer(outline: &mut Outline, names: &mut Names, asks: Asks) -> Result<bool> {
    let mut procs = asks.procs.take();
    procs.sort_unstable();
    procs.dedup();
    procs.retain(|key| !outline.procs.contains_key(key));
    if !procs.is_empty() {
        let calls: Vec<(ItemRef, ProcKind)> = procs.iter().map(|(run, _)| (*run, ProcKind::Bang)).collect();
        let inputs: Vec<Vec<String>> = procs.iter().map(|(_, text)| vec![text.clone()]).collect();
        let outputs = run(outline, names, &calls, &inputs)?;
        outline.procs.extend(procs.iter().cloned().zip(outputs));
    }
    let more = names.answer(outline, asks)?;
    settle(outline, names)?;
    Ok(more || !procs.is_empty())
}

/// Adds each call's output to its module, takes out the items that attribute macros replaced, and has names resolve the crates that held them again.
fn apply(outline: &mut Outline, names: &mut Names, bound: &[Bound], outputs: Vec<Result<String, String>>) -> Result<()> {
    let mut by_crate: HashMap<usize, Vec<(usize, Source, String)>> = HashMap::default();
    let mut stale = Vec::new();
    for (call, output) in bound.iter().zip(outputs) {
        let name = macro_name(outline, call.run);
        let parsed = output.and_then(|text| Source::parse(site_path(&call.site), text).map_err(|error| format!("{error:#}")));
        match parsed {
            Ok(source) => by_crate.entry(call.krate).or_default().push((call.site.module(), source, name)),
            Err(error) => {
                outline.crates[call.krate].errors.push(format!("{name}: {error}"));
                continue;
            }
        }
        if let Site::Attribute(held) = &call.site {
            outline.consume(call.krate, &held.file, held.at);
            stale.extend(outline.dependents(call.krate));
        }
    }
    for (krate, outputs) in by_crate {
        outline.insert(krate, outputs)?;
    }
    stale.sort_unstable();
    stale.dedup();
    stale.into_iter().for_each(|krate| names.reset(krate));
    Ok(())
}

fn site_path(site: &Site) -> std::path::PathBuf {
    match site {
        Site::Bang(call) => call.file.source.path.clone(),
        Site::Derive(held) | Site::Attribute(held) => held.file.source.path.clone(),
    }
}

fn macro_name(outline: &Outline, run: ItemRef) -> String {
    let file = &outline.crates[run.krate as usize].files[run.file as usize];
    let item = &file.items.items[run.item as usize];
    item.name.map_or_else(String::new, |name| file.source.text(name).to_owned())
}

/// The texts a call runs on: the tokens between a bang call's delimiters; the item under a derive without its `#[derive]` attributes; or an attribute's arguments and the item without that attribute.
fn inputs(call: &Bound) -> Vec<String> {
    match &call.site {
        Site::Bang(waiting) => {
            let item = &waiting.file.items.items[waiting.at as usize];
            vec![item.body.map_or_else(String::new, |open| inside(&waiting.file.source, open))]
        }
        Site::Derive(held) => {
            let src = &held.file.source;
            vec![item_text(src, &held.file.items.items[held.at as usize], &|path| src.text(path) != "derive")]
        }
        Site::Attribute(held) => {
            let src = &held.file.source;
            let after = held.path.1 + 1;
            let args = if matches!(src.text(after), "(" | "[" | "{") { inside(src, after) } else { String::new() };
            vec![args, item_text(src, &held.file.items.items[held.at as usize], &|path| path != held.path.0)]
        }
    }
}

/// The tokens between the delimiter that opens at `open` and its pair.
fn inside(src: &Source, open: u32) -> String {
    let close = src.pair(open);
    if close > open + 1 { src.span(open + 1, close - 1).to_owned() } else { String::new() }
}

/// The text of `item` with the outer attributes whose path starts at a token that `keep` rejects left out.
fn item_text(src: &Source, item: &crate::outline::item::Item, keep: &dyn Fn(u32) -> bool) -> String {
    let mut parts = Vec::new();
    let mut at = item.lead;
    while at < item.start && src.text(at) == "#" {
        let close = src.pair(at + 1);
        if keep(at + 2) {
            parts.push(src.span(at, close));
        }
        at = close + 1;
    }
    parts.push(src.span(at, item.end - 1));
    parts.join(" ")
}

/// The two entry functions of haste's `proc_macro` crate.
fn entries(outline: &Outline) -> Result<[ItemRef; 2]> {
    let Some(krate) = outline.roots.iter().position(|root| root.path.as_os_str() == PROC_MACRO_ROOT) else { bail!("no `proc_macro` crate in the graph") };
    let tree = &outline.crates[krate];
    let Some(file) = tree.files.iter().position(|file| !file.expanded && file.source.path.as_os_str() == PROC_MACRO_ROOT) else { bail!("`proc_macro` is not read") };
    let items = &tree.files[file].items.items;
    let find = |name: &str| items.iter().position(|item| item.kind == ItemKind::Fn && item.parent.is_none() && item.name.is_some_and(|at| tree.files[file].source.text(at) == name));
    let [Some(one), Some(two)] = ENTRIES.map(find) else { bail!("`proc_macro` has no entry functions") };
    Ok([one, two].map(|item| ItemRef { krate: krate as u32, file: file as u32, item: item as u32 }))
}

fn run(outline: &mut Outline, names: &mut Names, bound: &[(ItemRef, ProcKind)], inputs: &[Vec<String>]) -> Result<Vec<Result<String, String>>> {
    let pair = entries(outline)?;
    let mut roots: Vec<Instance> = pair.iter().map(|item| (*item, Vec::new())).collect();
    roots.extend(bound.iter().map(|(run, _)| (*run, Vec::new())));
    roots.dedup();
    let ask = |outline: &mut Outline, names: &mut Names, asks: Asks| names.answer(outline, asks);
    reach_answered(outline, names, (|_: &mut Context<'_>| roots.clone(), ask), |mut cx, reached| execute(&mut cx, &reached, (pair, bound), inputs))
}

fn execute(cx: &mut Context<'_>, reached: &Reached, (pair, bound): ([ItemRef; 2], &[(ItemRef, ProcKind)]), inputs: &[Vec<String>]) -> Result<Vec<Result<String, String>>> {
    let (image, broken) = Image::build(cx, reached)?;
    let mut results: Vec<Result<String, String>> = Vec::new();
    let mut jobs = Vec::new();
    for (&(call, kind), texts) in bound.iter().zip(inputs) {
        let entry = reached.index[&(pair[usize::from(kind == ProcKind::Attribute)], Vec::new())];
        let run = reached.index[&(call, Vec::new())];
        let closure = reached.closure(&[entry, run]);
        let failure = reached.failure(&closure, &broken).map(|(_, error)| error);
        match failure {
            Some(error) => results.push(Err(error.clone())),
            None => {
                results.push(Ok(String::new()));
                jobs.push(Job { entry, run, closure, texts: texts.iter().map(|text| text.as_bytes()).collect() });
            }
        }
    }
    let mut ran = image.run_macros(&jobs)?.into_iter();
    for result in results.iter_mut().filter(|result| result.is_ok()) {
        *result = ran.next().unwrap_or_else(|| Err("no output".to_owned())).and_then(|bytes| String::from_utf8(bytes).map_err(|error| error.to_string()));
    }
    Ok(results)
}
