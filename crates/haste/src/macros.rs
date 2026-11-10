//! Procedural macros in the image: the calls that name resolution binds are built into one image with haste's `proc_macro` crate, run in a forked child, and their output read back into the outline.

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::image::{Jit, Job, run_macros};
use crate::names::procs::{Bound, Site};
use crate::names::path::Requests;
use crate::names::{ItemRef, Names};
use crate::outline::Outline;
use crate::outline::item::{ItemKind, ProcKind};
use crate::outline::source::Source;
use crate::outline::tree::PROC_MACRO_ROOT;
use crate::graph::{Targets, Workspace, store};
use crate::sema::reach::{Instance, Reached, reach};
use crate::sema::{Context, Cache};

const ENTRIES: [&str; 2] = ["__haste_expand1", "__haste_expand2"];

/// One build's compilation: its outline, the names resolved in it, and the macro expander, which keeps its code and inference from one expansion pass to the next.
pub struct Compilation {
    pub outline: Outline,
    pub names: Names,
    expander: Option<Jit>,
}

impl Compilation {
    fn new(workspace: &Workspace, outline: Outline) -> Compilation {
        let names = Names::new(workspace, &outline);
        Compilation { outline, names, expander: None }
    }

    /// Drops the macro expander once loading settles, so that a later expansion compiles the macros again from the current outline.
    pub fn settle(&mut self) {
        self.expander = None;
    }
}

/// Loads every module of every crate, with the build scripts of the units `scripts`, resolves every import and expands every procedural macro call, then drops the macro expander.
pub fn load_eager(workspace: &Workspace, targets: &Targets, scripts: &[usize]) -> Result<Compilation> {
    let mut outline = crate::outline::new(workspace, targets, scripts);
    outline.load_all()?;
    let mut compilation = Compilation::new(workspace, outline);
    compilation.names.crates.iter_mut().for_each(|krate| krate.eager = true);
    resolve(&mut compilation)?;
    Ok(compilation)
}

/// Loads the member crates' roots, resolves their names and expands their procedural macro calls, reading of other crates only what those lookups reach, then drops the macro expander.
pub fn load_lazy(workspace: &Workspace, targets: &Targets) -> Result<Compilation> {
    let mut outline = crate::outline::new(workspace, targets, &[]);
    let members: Vec<usize> = (0..outline.roots.len()).filter(|at| crate::cli::is_member(workspace, &outline.roots[*at])).collect();
    for &krate in &members {
        outline.load(krate, 0)?;
    }
    let mut compilation = Compilation::new(workspace, outline);
    members.iter().for_each(|krate| compilation.names.crates[*krate].eager = true);
    resolve(&mut compilation)?;
    Ok(compilation)
}

/// Reaches from the instances `roots` gives, answering what each pass asks for until a pass settles or nothing new can load, then hands that pass's context and what it reached to `finish`.
pub fn reach_resolved<R>(compilation: &mut Compilation, roots: impl Fn(&mut Context<'_>) -> Vec<Instance>, finish: impl FnOnce(Context<'_>, Reached) -> Result<R>) -> Result<R> {
    reach_with(compilation, (roots, load), finish)
}

/// Reaches from `roots` as `reach_resolved` does, answering each pass's asks with `ask`, which says whether anything new loaded.
fn reach_with<R>(compilation: &mut Compilation, (roots, ask): (impl Fn(&mut Context<'_>) -> Vec<Instance>, impl Fn(&mut Compilation, Requests) -> Result<bool>), finish: impl FnOnce(Context<'_>, Reached) -> Result<R>) -> Result<R> {
    let mut finish = Some(finish);
    resolve_sema(compilation, (Cache::default(), ask), |mut cx, last| {
        let from = roots(&mut cx);
        let reached = reach(&mut cx, &from);
        match finish.take_if(|_| last || cx.resolved()) {
            Some(finish) => finish(cx, reached).map(Ok),
            None => Ok(Err(cx.release())),
        }
    })
}

/// Runs `pass` on a context over `store`, answering with `ask` what it asked for, until it gives a result: `pass` returns its result, or the released context to answer and run again, and learns whether the last answer loaded nothing new. Drops the macro expander at the end.
pub fn resolve_sema<R>(compilation: &mut Compilation, asking: (Cache, impl Fn(&mut Compilation, Requests) -> Result<bool>), pass: impl FnMut(Context<'_>, bool) -> Result<Result<R, (Cache, Requests)>>) -> Result<R> {
    let done = answer_passes(compilation, asking, pass);
    compilation.settle();
    done
}

fn answer_passes<R>(compilation: &mut Compilation, (mut store, ask): (Cache, impl Fn(&mut Compilation, Requests) -> Result<bool>), mut pass: impl FnMut(Context<'_>, bool) -> Result<Result<R, (Cache, Requests)>>) -> Result<R> {
    let mut last = false;
    loop {
        let (mut kept, asks) = match pass(Context::new(&compilation.outline, &compilation.names, store), last)? {
            Ok(done) => return Ok(done),
            Err(released) => released,
        };
        last = !ask(compilation, asks)?;
        kept.forget_lookups();
        store = kept;
    }
}

/// Settles names, then expands every procedural macro call they bind and settles again, until no call is left to bind, then drops the macro expander.
pub fn resolve(compilation: &mut Compilation) -> Result<()> {
    let done = compilation.names.resolve(&mut compilation.outline).and_then(|()| expand_bound(compilation));
    compilation.settle();
    done
}

/// Expands every procedural macro call that resolved names bind and resolves again, loading what the binding asked for, until no call is left to bind.
fn expand_bound(compilation: &mut Compilation) -> Result<()> {
    loop {
        let (bound, asks) = compilation.names.bind_procs(&mut compilation.outline);
        if bound.is_empty() {
            if asks.count() == 0 || !compilation.names.load(&mut compilation.outline, asks)? {
                return Ok(());
            }
            continue;
        }
        let inputs: Vec<Vec<String>> = bound.iter().map(inputs).collect();
        let calls: Vec<Call> = bound.iter().map(|call| (call.run, call.kind, call.krate)).collect();
        let outputs = run(compilation, &calls, &inputs)?;
        apply(compilation, &bound, outputs)?;
        compilation.names.resolve(&mut compilation.outline)?;
    }
}

/// Answers what a Sema pass asked for: runs the procedural macros its bodies call and keeps their output, loads and resolves the names it asked for, and expands the item macros that brings in; returns whether anything was new.
pub fn load(compilation: &mut Compilation, asks: Requests) -> Result<bool> {
    let mut procs = asks.procs.take();
    procs.sort_unstable();
    procs.dedup();
    procs.retain(|key| !compilation.outline.procs.contains_key(key));
    if !procs.is_empty() {
        let calls: Vec<Call> = procs.iter().map(|(run, _)| (*run, ProcKind::Bang, usize::MAX)).collect();
        let inputs: Vec<Vec<String>> = procs.iter().map(|(_, text)| vec![text.clone()]).collect();
        let outputs = run(compilation, &calls, &inputs)?;
        compilation.outline.procs.extend(procs.iter().cloned().zip(outputs));
    }
    let more = compilation.names.load(&mut compilation.outline, asks)?;
    expand_bound(compilation)?;
    Ok(more || !procs.is_empty())
}

/// Adds each call's output to its module, takes out the items that attribute macros replaced, and has names resolve the crates that held them again.
fn apply(compilation: &mut Compilation, bound: &[Bound], outputs: Vec<Result<String, String>>) -> Result<()> {
    let Compilation { outline, names, .. } = compilation;
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

/// The two entry functions of haste's `proc_macro` crate, reading its root first if it is only declared.
fn entries(outline: &mut Outline) -> Result<[ItemRef; 2]> {
    let Some(krate) = outline.roots.iter().position(|root| root.path.as_os_str() == PROC_MACRO_ROOT) else { bail!("no `proc_macro` crate in the graph") };
    outline.load(krate, 0)?;
    let tree = &outline.crates[krate];
    let Some(file) = tree.files.iter().position(|file| !file.expanded && file.source.path.as_os_str() == PROC_MACRO_ROOT) else { bail!("`proc_macro` is not read") };
    let items = &tree.files[file].items.items;
    let find = |name: &str| items.iter().position(|item| item.kind == ItemKind::Fn && item.parent.is_none() && item.name.is_some_and(|at| tree.files[file].source.text(at) == name));
    let [Some(one), Some(two)] = ENTRIES.map(find) else { bail!("`proc_macro` has no entry functions") };
    Ok([one, two].map(|item| ItemRef { krate: krate as u32, file: file as u32, item: item as u32 }))
}

/// A procedural macro call: the macro function, its kind, and the crate that calls it, `usize::MAX` for a call in a function body.
type Call = (ItemRef, ProcKind, usize);

/// The output of each call `bound` on its `inputs`, or why it failed: the result the store kept for a macro of a pinned crate, else from running it, kept when it ran.
fn run(compilation: &mut Compilation, bound: &[Call], inputs: &[Vec<String>]) -> Result<Vec<Result<String, String>>> {
    let outline = &compilation.outline;
    let keys: Vec<Option<u64>> = bound.iter().zip(inputs).map(|(&(call, ..), texts)| Some(store::expansion_key(outline.pins[call.krate as usize]?, &macro_name(outline, call), texts))).collect();
    let mut results: Vec<Option<Result<String, String>>> = keys.iter().map(|key| key.and_then(store::kept_expansion)).collect();
    let missing: Vec<usize> = (0..bound.len()).filter(|&at| results[at].is_none()).collect();
    if !missing.is_empty() {
        let calls: Vec<Call> = missing.iter().map(|&at| bound[at]).collect();
        let texts: Vec<Vec<String>> = missing.iter().map(|&at| inputs[at].clone()).collect();
        for (at, result) in missing.into_iter().zip(expand(compilation, &calls, &texts)?) {
            if let (Some(ran), Some(key)) = (&result, keys[at])
                && let Err(error) = store::keep_expansion(key, ran)
            {
                eprintln!("cannot keep the output of {}: {error}", macro_name(&compilation.outline, bound[at].0));
            }
            results[at] = result;
        }
    }
    Ok(results.into_iter().map(|result| result.unwrap_or_else(|| Err("no output".to_owned()))).collect())
}

/// Runs each call `bound` on its `inputs` in children that compile each function on its first call; `None` for a call that did not run. The macro expander keeps its code and inference for the next pass, unless a pass is already using it.
fn expand(compilation: &mut Compilation, bound: &[Call], inputs: &[Vec<String>]) -> Result<Vec<Option<Result<String, String>>>> {
    let pair = entries(&mut compilation.outline)?;
    let mut demand = match compilation.expander.take() {
        Some(kept) => kept,
        None => Jit::new(Cache::default())?,
    };
    let entry = pair.map(|item| demand.declare((item, Vec::new())));
    let jobs: Vec<Job<'_>> = bound.iter().zip(inputs).map(|(&(call, kind, caller), texts)| Job { entry: entry[usize::from(kind == ProcKind::Attribute)], run: demand.declare((call, Vec::new())), texts: texts.iter().map(|text| text.as_bytes()).collect(), group: (caller, call.krate) }).collect();
    let ask = |compilation: &mut Compilation, asks: Requests| compilation.names.load(&mut compilation.outline, asks);
    let ran = run_macros(compilation, &mut demand, &jobs, &ask);
    drop(jobs);
    compilation.expander.get_or_insert(demand);
    let ran = ran?;
    Ok(ran.into_iter().map(|result| result.map(|result| result.and_then(|bytes| String::from_utf8(bytes).map_err(|error| error.to_string())))).collect())
}
