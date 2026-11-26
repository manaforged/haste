//! Procedural macros in the image: the calls that name resolution binds are built into one image with haste's `proc_macro` crate, run in a forked child, and their output read back into the outline.

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Targets, Workspace, store};
use crate::image::{Jit, Job, run_macros};
use crate::names::path::Requests;
use crate::names::procs::{Bound, Site};
use crate::names::{ItemRef, Names};
use crate::outline::Outline;
use crate::outline::attr::{self, split_commas};
use crate::outline::item::{Attr, ItemKind, ProcKind};
use crate::outline::source::Source;
use crate::outline::tree::{PROC_MACRO_ROOT, Profile};
use crate::sema::reach::{Instance, Reached, reach_wave};
use crate::sema::{Answer, Cache, Context};

const ENTRIES: [&str; 2] = ["__haste_expand1", "__haste_expand2"];

/// One build's compilation: its outline, the names resolved in it, and the persistent macro expander.
pub struct Compilation {
    pub outline: Outline,
    pub names: Names,
    expander: Option<Expander>,
    substrate: Option<std::sync::Arc<crate::substrate::Substrate>>,
}

struct Expander {
    jit: Jit,
    roots: Vec<usize>,
    stamp: u64,
}

impl Expander {
    fn new(outline: &Outline, jit: Jit, mut roots: Vec<usize>) -> Expander {
        roots.sort_unstable();
        roots.dedup();
        let stamp = macro_graph_stamp(outline, &roots);
        Expander { jit, roots, stamp }
    }

    fn matches(&self, outline: &Outline) -> bool {
        macro_graph_stamp(outline, &self.roots) == self.stamp
    }
}

impl Compilation {
    /// Drops the macro expander only when the macro crate graph changed, keeping compiled macro code across settle phases.
    pub fn settle(&mut self) {
        if self
            .expander
            .as_ref()
            .is_some_and(|kept| !kept.matches(&self.outline))
        {
            self.expander = None;
        }
    }
}

fn load_with_sysroot(
    workspace: &Workspace,
    targets: &Targets,
    scripts: &[usize],
    eager_all: bool,
) -> Result<Compilation> {
    let mut outline = crate::outline::new(workspace, targets, scripts);
    let cached = crate::substrate::sysroot::try_load(workspace, targets, &mut outline);
    let was_miss = cached.is_none();
    let show_sysroot = was_miss
        && crate::substrate::enabled()
        && crate::scripts::sysroot_scripts_settled(workspace);
    let sysroot_started = show_sysroot.then(std::time::Instant::now);
    if show_sysroot {
        let mode = if crate::profile() == Profile::Release {
            "release"
        } else {
            "debug"
        };
        eprintln!("haste: indexing standard library ({mode}, one-time)...");
    }
    if !eager_all {
        if was_miss {
            crate::substrate::preload_fixed_files(workspace, &targets.target.triple, &outline);
        } else {
            crate::substrate::preload_dep_files(workspace, &targets.target.triple, &outline);
        }
    }
    let selected: Vec<bool> = (0..outline.crates.len())
        .map(|krate| {
            let sysroot = outline.is_sysroot(krate as u32);
            if eager_all {
                !sysroot || was_miss
            } else {
                (was_miss && sysroot)
                    || workspace
                        .graph
                        .members
                        .contains(&workspace.graph.units[outline.roots[krate].unit].package)
            }
        })
        .collect();
    outline.load_selected(&selected)?;
    let names = Names::new_with(workspace, &outline, cached);
    let has_proc_macro = outline
        .roots
        .iter()
        .any(|root| root.path.as_os_str() == PROC_MACRO_ROOT);
    let macro_slot = has_proc_macro
        .then(|| {
            crate::with_host_target(|| crate::build::substrate::open_batch_substrate(workspace))
        })
        .flatten();
    let substrate = macro_slot
        .as_ref()
        .and_then(crate::build::substrate::SubstrateSlot::top);
    let mut compilation = Compilation {
        outline,
        names,
        expander: None,
        substrate,
    };
    for (krate, &on) in selected.iter().enumerate() {
        if on {
            compilation.names.crates[krate].set_eager();
        }
    }
    resolve(&mut compilation)?;
    if let (Some(slot), Some(expander)) = (macro_slot, &compilation.expander) {
        crate::with_host_target(|| {
            crate::build::substrate::save_jit_substrate(
                &compilation.outline,
                expander.jit.store(),
                expander.jit.image(),
                slot,
            )
        });
    }
    if was_miss && crate::scripts::sysroot_scripts_settled(workspace) {
        crate::substrate::sysroot::save(workspace, targets, &compilation);
        if let Some(started) = sysroot_started {
            let secs = started.elapsed().as_secs_f64();
            eprintln!("haste: standard library cached in {secs:.1}s");
        }
    }
    let errors = compilation.outline.local_errors(workspace);
    if !errors.is_empty() {
        anyhow::bail!("{}", errors.join("\n"));
    }
    crate::host::native::load(&compilation.outline)?;
    Ok(compilation)
}

/// Loads every module of every crate, with the build scripts of the units `scripts`, resolves every import and expands every procedural macro call, then drops the macro expander.
pub fn load_eager(
    workspace: &Workspace,
    targets: &Targets,
    scripts: &[usize],
) -> Result<Compilation> {
    load_with_sysroot(workspace, targets, scripts, true)
}

/// Loads the member crates' roots, resolves their names and expands their procedural macro calls, reading of other crates only what those lookups reach, then drops the macro expander.
pub fn load_lazy(workspace: &Workspace, targets: &Targets) -> Result<Compilation> {
    load_with_sysroot(workspace, targets, &[], false)
}

/// Reaches from the instances `roots` gives, answering what each pass asks for until a pass settles or nothing new can load, then hands that pass's context and what it reached to `finish`.
pub fn reach_resolved<R>(
    compilation: &mut Compilation,
    roots: impl Fn(&mut Context<'_>) -> Vec<Instance>,
    finish: impl FnOnce(Context<'_>, Reached) -> Result<R>,
) -> Result<R> {
    reach_with(compilation, Cache::default(), (roots, load), finish)
}

/// Reaches from `roots` using an existing `cache`, answering what each pass asks for until a pass settles or nothing new can load.
pub fn reach_resolved_with_cache<R>(
    compilation: &mut Compilation,
    cache: Cache,
    roots: impl Fn(&mut Context<'_>) -> Vec<Instance>,
    finish: impl FnOnce(Context<'_>, Reached) -> Result<R>,
) -> Result<R> {
    reach_with(compilation, cache, (roots, load), finish)
}

/// Reaches from `roots` as `reach_resolved` does, answering each pass's asks with `ask`, which says whether anything new loaded.
fn reach_with<R>(
    compilation: &mut Compilation,
    cache: Cache,
    (roots, ask): (
        impl Fn(&mut Context<'_>) -> Vec<Instance>,
        impl Fn(&mut Compilation, Requests) -> Result<Answer>,
    ),
    finish: impl FnOnce(Context<'_>, Reached) -> Result<R>,
) -> Result<R> {
    let mut finish = Some(finish);
    let mut reached = Reached::default();
    resolve_sema(compilation, (cache, ask), |mut cx, last| {
        let from = roots(&mut cx);
        reach_wave(&mut cx, &mut reached, &from, last);
        match finish.take_if(|_| last || cx.resolved()) {
            Some(finish) => finish(cx, std::mem::take(&mut reached)).map(Ok),
            None => Ok(Err(cx.release())),
        }
    })
}

/// Runs `pass` on a context over `store`, answering with `ask` what it asked for, until it gives a result: `pass` returns its result, or the released context to answer and run again, and learns whether the last answer loaded nothing new. Drops the macro expander at the end.
pub fn resolve_sema<R>(
    compilation: &mut Compilation,
    asking: (Cache, impl Fn(&mut Compilation, Requests) -> Result<Answer>),
    pass: impl FnMut(Context<'_>, bool) -> Result<Result<R, (Cache, Requests)>>,
) -> Result<R> {
    let done = answer_passes(compilation, asking, pass);
    compilation.settle();
    done
}

/// Answers the asks a pass released with `ask` and drops from the store it released what the answer made stale.
pub fn answer(
    compilation: &mut Compilation,
    ask: impl Fn(&mut Compilation, Requests) -> Result<Answer>,
    (mut kept, asks): (Cache, Requests),
) -> (Cache, Result<Answer>) {
    let answer = ask(compilation, asks);
    if let Ok(answer) = &answer {
        kept.forget_after(&compilation.outline, *answer);
    }
    (kept, answer)
}

fn answer_passes<R>(
    compilation: &mut Compilation,
    (mut store, ask): (Cache, impl Fn(&mut Compilation, Requests) -> Result<Answer>),
    mut pass: impl FnMut(Context<'_>, bool) -> Result<Result<R, (Cache, Requests)>>,
) -> Result<R> {
    let mut last = false;
    loop {
        let (kept, asks) = match pass(
            Context::new(&compilation.outline, &compilation.names, store),
            last,
        )? {
            Ok(done) => return Ok(done),
            Err(released) => released,
        };
        let (kept, answer) = answer(compilation, &ask, (kept, asks));
        last = answer? == Answer::Nothing;
        store = kept;
    }
}

/// Settles names, then expands every procedural macro call they bind and settles again, until no call is left to bind, then drops the macro expander.
pub fn resolve(compilation: &mut Compilation) -> Result<()> {
    let done = compilation
        .names
        .resolve(&mut compilation.outline)
        .and_then(|()| expand_bound(compilation).map(drop));
    compilation.settle();
    done
}

/// Expands every procedural macro call that resolved names bind and resolves again, loading what the binding asked for, until no call is left to bind; returns whether it expanded or loaded anything.
fn expand_bound(compilation: &mut Compilation) -> Result<bool> {
    let mut changed = false;
    loop {
        let (bound, asks) = compilation.names.bind_procs(&mut compilation.outline);
        if bound.is_empty() {
            if asks.count() == 0 || !compilation.names.load(&mut compilation.outline, asks)? {
                return Ok(changed);
            }
            changed = true;
            continue;
        }
        changed = true;
        let inputs: Vec<Vec<String>> = bound
            .iter()
            .map(|call| inputs(&compilation.outline, call))
            .collect();
        let calls: Vec<Call> = bound
            .iter()
            .map(|call| (call.run, call.kind, call.krate))
            .collect();
        let outputs = run(compilation, &calls, &inputs)?;
        apply(compilation, &bound, outputs)?;
        compilation.names.resolve(&mut compilation.outline)?;
    }
}

/// Answers what a Sema pass asked for: runs the procedural macros its bodies call and keeps their output, loads and resolves the names it asked for, and expands the item macros that brings in; returns what was new.
pub fn load(compilation: &mut Compilation, asks: Requests) -> Result<Answer> {
    let mut procs = asks.procs.take();
    procs.sort_unstable();
    procs.dedup();
    procs.retain(|key| !compilation.outline.procs.contains_key(key));
    if !procs.is_empty() {
        let calls: Vec<Call> = procs
            .iter()
            .map(|(run, _)| {
                let file = &compilation.outline.crates[run.krate as usize].files[run.file as usize];
                let kind = crate::names::collect::proc_kind(
                    &file.source,
                    &file.items.items[run.item as usize],
                )
                .map_or(ProcKind::Bang, |(_, kind)| kind);
                (*run, kind, usize::MAX)
            })
            .collect();
        let inputs: Vec<Vec<String>> = procs.iter().map(|(_, text)| vec![text.clone()]).collect();
        let outputs = run(compilation, &calls, &inputs)?;
        compilation
            .outline
            .procs
            .extend(procs.iter().cloned().zip(outputs));
    }
    let more = compilation.names.load(&mut compilation.outline, asks)?;
    let expanded = expand_bound(compilation)?;
    Ok(match (more, expanded, procs.is_empty()) {
        (true, _, _) | (false, true, _) => Answer::Names,
        (false, false, false) => Answer::Bodies,
        (false, false, true) => Answer::Nothing,
    })
}

/// Adds each call's output to its module, parsing outputs in parallel before sequential AST insertion, takes out replaced items, and resets stale crates.
fn apply(
    compilation: &mut Compilation,
    bound: &[Bound],
    outputs: Vec<Result<String, String>>,
) -> Result<()> {
    let Compilation { outline, names, .. } = compilation;
    let to_parse: Vec<(std::path::PathBuf, Result<String, String>)> = bound
        .iter()
        .zip(outputs)
        .map(|(call, output)| (site_path(&call.site), output))
        .collect();
    let parsed = crate::sync::parallel_map(to_parse, |(path, output)| {
        output.and_then(|text| Source::parse(path, text).map_err(|error| format!("{error:#}")))
    });
    let mut by_crate: HashMap<usize, Vec<(usize, Source, String)>> = HashMap::default();
    let mut stale = Vec::new();
    for (call, parsed) in bound.iter().zip(parsed) {
        let name = macro_name(outline, call.run);
        match parsed {
            Ok(source) => {
                by_crate
                    .entry(call.krate)
                    .or_default()
                    .push((call.site.module(), source, name))
            }
            Err(error) => {
                outline.crates[call.krate]
                    .errors
                    .push(format!("{name}: {error}"));
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
    item.name
        .map_or_else(String::new, |name| file.source.text(name).to_owned())
}

/// The texts a call runs on: the tokens between a bang call's delimiters; the item under a derive without its `#[derive]` attributes; or an attribute's arguments and the item without that attribute.
fn inputs(outline: &Outline, call: &Bound) -> Vec<String> {
    match &call.site {
        Site::Bang(waiting) => {
            let item = &waiting.file.items.items[waiting.at as usize];
            vec![item.body.map_or_else(String::new, |open| {
                bang_input(
                    &outline.deps[call.krate],
                    call.krate,
                    &waiting.file.source,
                    open,
                )
            })]
        }
        Site::Derive(held) => vec![derive_input(
            &held.file.source,
            &held.file.items.items[held.at as usize],
            &outline.crates[call.krate].env,
        )],
        Site::Attribute(held) => {
            let src = &held.file.source;
            let after = held.path.1 + 1;
            let args = if matches!(src.text(after), "(" | "[" | "{") {
                inside(src, after)
            } else {
                String::new()
            };
            vec![
                args,
                item_text(
                    src,
                    &held.file.items.items[held.at as usize],
                    &|path| path != held.path.0,
                    None,
                ),
            ]
        }
    }
}

/// The text of `item` as a procedural derive macro receives it: without its `#[derive]` attributes or cfg-disabled fields and variants.
pub(crate) fn derive_input(
    src: &Source,
    item: &crate::outline::item::Item,
    env: &crate::outline::tree::Env,
) -> String {
    item_text(
        src,
        item,
        &|path| !crate::outline::attr::is_derive_at(src, path, src.pair(path - 1)),
        Some(&|attrs| env.enabled(attrs)),
    )
}

/// The tokens between the delimiter that opens at `open` and its pair, as a function-like macro of crate `krate`, whose dependencies are `deps`, receives them: with each `$crate` a `macro_rules!` wrote spelled as a path to its crate.
pub(crate) fn bang_input(
    deps: &[(String, usize)],
    krate: usize,
    src: &Source,
    open: u32,
) -> String {
    let close = src.pair(open);
    let (mut parts, mut from) = (Vec::new(), open + 1);
    for at in open + 1..close {
        let Some(path) = src
            .dollar(at)
            .filter(|_| at >= from)
            .and_then(|named| crate_path(deps, krate, named as usize))
        else {
            continue;
        };
        if at > from {
            parts.push(src.span(from, at - 1).to_owned());
        }
        parts.push(path);
        from = at + 2;
    }
    if close > from {
        parts.push(src.span(from, close - 1).to_owned());
    }
    parts.join(" ")
}

/// The path that names crate `named` from crate `krate`, whose dependencies are `deps`.
fn crate_path(deps: &[(String, usize)], krate: usize, named: usize) -> Option<String> {
    if named == krate {
        return Some("crate".to_owned());
    }
    deps.iter()
        .find(|(_, dep)| *dep == named)
        .map(|(name, _)| format!("::{name}"))
}

/// The tokens between the delimiter that opens at `open` and its pair.
fn inside(src: &Source, open: u32) -> String {
    let close = src.pair(open);
    if close > open + 1 {
        src.span(open + 1, close - 1).to_owned()
    } else {
        String::new()
    }
}

/// The text of `item` with the outer attributes whose path starts at a token that `keep` rejects left out, and, as a derive sees it, the fields or variants whose attributes `enabled` rejects.
fn item_text(
    src: &Source,
    item: &crate::outline::item::Item,
    keep: &dyn Fn(u32) -> bool,
    enabled: Option<&dyn Fn(&[Attr]) -> bool>,
) -> String {
    let mut parts = Vec::new();
    let mut at = item.lead;
    while at < item.start && src.text(at) == "#" {
        let close = src.pair(at + 1);
        if keep(at + 2) {
            parts.push(src.span(at, close).to_owned());
        }
        at = close + 1;
    }
    match item
        .body
        .zip(enabled)
        .and_then(|(open, enabled)| Some((open, configured(src, open, enabled)?)))
    {
        Some((open, members)) => parts.extend([
            src.span(at, open).to_owned(),
            members,
            src.span(src.pair(open), item.end - 1).to_owned(),
        ]),
        None => parts.push(src.span(at, item.end - 1).to_owned()),
    }
    parts.join(" ")
}

/// The members inside the delimiter that opens at `open` whose attributes `enabled` accepts, or none when it accepts them all.
fn configured(src: &Source, open: u32, enabled: &dyn Fn(&[Attr]) -> bool) -> Option<String> {
    let (mut kept, mut dropped) = (Vec::new(), false);
    for (from, to) in split_commas(src, open + 1, src.pair(open))
        .into_iter()
        .filter(|(from, to)| from < to)
    {
        let mut attrs = Vec::new();
        attr::attrs(src, from, false, &mut attrs).ok()?;
        if enabled(&attrs) {
            kept.push(src.spelled((from, to)))
        } else {
            dropped = true
        }
    }
    dropped.then(|| kept.join(" , "))
}

/// The two entry functions of haste's `proc_macro` crate, reading its root first if it is only declared.
fn entries(outline: &mut Outline) -> Result<[ItemRef; 2]> {
    let Some(krate) = outline
        .roots
        .iter()
        .position(|root| root.path.as_os_str() == PROC_MACRO_ROOT)
    else {
        bail!("no `proc_macro` crate in the graph")
    };
    outline.load(krate, 0)?;
    let tree = &outline.crates[krate];
    let Some(file) = tree
        .files
        .iter()
        .position(|file| !file.expanded && file.source.path.as_os_str() == PROC_MACRO_ROOT)
    else {
        bail!("`proc_macro` is not read")
    };
    let items = &tree.files[file].items.items;
    let find = |name: &str| {
        items.iter().position(|item| {
            item.kind == ItemKind::Fn
                && item.parent.is_none()
                && item
                    .name
                    .is_some_and(|at| tree.files[file].source.text(at) == name)
        })
    };
    let [Some(one), Some(two)] = ENTRIES.map(find) else {
        bail!("`proc_macro` has no entry functions")
    };
    Ok([one, two].map(|item| ItemRef {
        krate: krate as u32,
        file: file as u32,
        item: item as u32,
    }))
}

/// A procedural macro call: the macro function, its kind, and the crate that calls it, `usize::MAX` for a call in a function body.
type Call = (ItemRef, ProcKind, usize);

/// The output of each call `bound` on its `inputs`, or why it failed: the result the store kept for a macro of a pinned crate, else from running it, kept when it ran.
fn run(
    compilation: &mut Compilation,
    bound: &[Call],
    inputs: &[Vec<String>],
) -> Result<Vec<Result<String, String>>> {
    let outline = &compilation.outline;
    let keys: Vec<Option<store::Expansion>> = bound
        .iter()
        .zip(inputs)
        .map(|(&(call, _, caller), texts)| {
            let macro_pin = outline.pins[call.krate as usize]?;
            let caller_pin = if caller == usize::MAX {
                0
            } else if let Some(pin) = outline.pins[caller] {
                pin
            } else {
                use std::hash::BuildHasher as _;
                rustc_hash::FxBuildHasher
                    .hash_one((0x6c6f_6361_6c5f_7069_u64, caller_vars(outline, caller)))
            };
            Some(store::Expansion::new(
                (macro_pin, caller_pin),
                &macro_name(outline, call),
                texts,
            ))
        })
        .collect();
    let mut results: Vec<Option<Result<String, String>>> = vec![None; keys.len()];
    let mut checks: Vec<(usize, &store::Expansion)> = Vec::new();
    for (at, key) in keys.iter().enumerate() {
        if let Some(key) = key {
            checks.push((at, key));
        }
    }
    for (at, kept) in crate::sync::parallel_map(checks, |(at, key)| (at, key.kept())) {
        results[at] = kept;
    }
    let missing: Vec<usize> = (0..bound.len())
        .filter(|&at| results[at].is_none())
        .collect();
    if !missing.is_empty() {
        let calls: Vec<Call> = missing.iter().map(|&at| bound[at]).collect();
        let texts: Vec<Vec<String>> = missing.iter().map(|&at| inputs[at].clone()).collect();
        for (at, result) in missing
            .into_iter()
            .zip(expand(compilation, &calls, &texts)?)
        {
            if let (Some(ran), Some(key)) = (&result, &keys[at])
                && let Err(error) = key.keep(ran)
            {
                eprintln!(
                    "cannot keep the output of {}: {error}",
                    macro_name(&compilation.outline, bound[at].0)
                );
            }
            results[at] = result;
        }
    }
    Ok(results
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("no output".to_owned())))
        .collect())
}

fn caller_vars(outline: &Outline, caller: usize) -> Vec<(String, String)> {
    let Some(tree) = outline.crates.get(caller) else {
        return Vec::new();
    };
    let env = &tree.env;
    let mut vars = Vec::with_capacity(env.vars.len() + 2);
    vars.push((
        "CARGO_CRATE_NAME".to_owned(),
        outline.roots[caller].name.clone(),
    ));
    vars.extend(env.vars.iter().cloned());
    if let Some(script) = &env.script {
        vars.extend(script.vars.iter().cloned());
    }
    vars
}

fn dep_closure(outline: &Outline, roots: &[usize]) -> (Vec<bool>, Vec<usize>) {
    let n = outline.crates.len();
    let mut seen = vec![false; n];
    let mut queue = Vec::new();
    for &root in roots {
        if root < n && !seen[root] {
            seen[root] = true;
            queue.push(root);
        }
    }
    let mut head = 0;
    while head < queue.len() {
        let krate = queue[head];
        head += 1;
        for &(_, dep) in &outline.deps[krate] {
            if dep < n && !seen[dep] {
                seen[dep] = true;
                queue.push(dep);
            }
        }
    }
    (seen, queue)
}

fn preload_macro_crates(compilation: &mut Compilation, roots: &[usize]) -> Result<()> {
    let (seen, _) = dep_closure(&compilation.outline, roots);
    let selected: Vec<bool> = seen
        .iter()
        .enumerate()
        .map(|(k, &on)| on && compilation.outline.crates[k].partial())
        .collect();
    if !selected.iter().any(|&on| on) {
        return Ok(());
    }
    compilation.outline.load_selected(&selected)?;
    for (krate, &on) in selected.iter().enumerate() {
        if on {
            compilation.names.crates[krate].set_eager();
        }
    }
    compilation.names.resolve(&mut compilation.outline)
}

/// Runs each call `bound` on its `inputs` in children that compile each function on its first call; `None` for a call that did not run. The macro expander keeps its code and inference while its macro crate graph is unchanged.
fn expand(
    compilation: &mut Compilation,
    bound: &[Call],
    inputs: &[Vec<String>],
) -> Result<Vec<Option<Result<String, String>>>> {
    crate::with_host_target(|| {
        let pair = entries(&mut compilation.outline)?;
        let mut preload_roots = Vec::with_capacity(bound.len() + 1);
        preload_roots.push(pair[0].krate as usize);
        preload_roots.extend(bound.iter().map(|(call, ..)| call.krate as usize));
        preload_macro_crates(compilation, &preload_roots)?;
        let (mut demand, mut roots) = match compilation.expander.take() {
            Some(kept) if kept.matches(&compilation.outline) => (kept.jit, kept.roots),
            _ => {
                let mut cache = Cache::default();
                if let Some(sub) = &compilation.substrate {
                    cache.set_substrate(Some(std::sync::Arc::clone(sub)));
                }
                (Jit::new(cache)?, Vec::new())
            }
        };
        let mut env_by_caller: HashMap<usize, Vec<(String, String)>> = HashMap::default();
        for &(_, _, caller) in bound {
            env_by_caller
                .entry(caller)
                .or_insert_with(|| caller_vars(&compilation.outline, caller));
        }
        let entry = pair.map(|item| demand.declare((item, crate::sema::substs::Substs::EMPTY)));
        let jobs: Vec<Job<'_>> = bound
            .iter()
            .zip(inputs)
            .map(|(&(call, kind, caller), texts)| Job {
                entry: entry[usize::from(kind == ProcKind::Attribute)],
                run: demand.declare((call, crate::sema::substs::Substs::EMPTY)),
                texts: texts.iter().map(|text| text.as_bytes()).collect(),
                vars: env_by_caller.get(&caller).map_or(&[], Vec::as_slice),
                group: (caller, call.krate),
            })
            .collect();
        let ask = |compilation: &mut Compilation, asks: Requests| {
            Ok(if compilation.names.load(&mut compilation.outline, asks)? {
                Answer::Names
            } else {
                Answer::Nothing
            })
        };
        let ran = run_macros(compilation, &mut demand, &jobs, &ask);
        drop(jobs);
        let ran = ran?;
        roots.push(pair[0].krate as usize);
        roots.extend(bound.iter().map(|(call, ..)| call.krate as usize));
        compilation.expander = Some(Expander::new(&compilation.outline, demand, roots));
        Ok(ran
            .into_iter()
            .map(|result| {
                result.map(|result| {
                    result.and_then(|bytes| {
                        String::from_utf8(bytes).map_err(|error| error.to_string())
                    })
                })
            })
            .collect())
    })
}

fn macro_graph_stamp(outline: &Outline, roots: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};

    let (_, mut queue) = dep_closure(outline, roots);
    queue.sort_unstable();
    let mut hasher = rustc_hash::FxHasher::default();
    outline.crates.len().hash(&mut hasher);
    for krate in queue {
        krate.hash(&mut hasher);
        outline.deps[krate].hash(&mut hasher);
        let holder = &outline.crates[krate];
        let e = holder.extent();
        let meta = (e.files, e.impls, e.impl_bodies, e.consumed, e.partial);
        (meta, holder.expansions).hash(&mut hasher);
        for file in &holder.files {
            let ptr = std::sync::Arc::as_ptr(file) as usize;
            (ptr, file.items.items.len(), file.source.text.len()).hash(&mut hasher);
        }
    }
    hasher.finish()
}
