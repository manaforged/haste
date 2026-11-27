//! Procedural macros in the image: the calls that name resolution binds are built into one image with haste's `proc_macro` crate, run in a forked child, and their output read back into the outline.

mod expand;
#[cfg(test)]
mod tests;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::graph::{Targets, Workspace};
use crate::image::Jit;
use crate::names::path::Requests;
use crate::names::procs::{Bound, Site};
use crate::names::{ItemRef, Names};
use crate::outline::Outline;
use crate::outline::attr::{self, split_commas};
use crate::outline::item::{Attr, ProcKind};
use crate::outline::source::Source;
use crate::outline::tree::{PROC_MACRO_ROOT, Profile};
use crate::sema::reach::{Instance, Reached, reach_wave};
use crate::sema::{Answer, Cache, Context};

use expand::{Call, dep_closure, infer_caller, macro_graph_stamp, run};

const ENTRIES: [&str; 2] = ["__haste_expand1", "__haste_expand2"];

/// One build's compilation: its outline, the names resolved in it, and the persistent macro expander.
pub struct Compilation {
    pub outline: Outline,
    pub names: Names,
    expander: Option<Expander>,
    substrate: Option<std::sync::Arc<crate::substrate::Substrate>>,
    sysroot_heads: Option<std::sync::Arc<HashMap<u32, std::sync::Arc<crate::sema::impls::Index>>>>,
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
    #[cfg(test)]
    pub(crate) fn for_test(outline: Outline, names: Names) -> Self {
        Self {
            outline,
            names,
            expander: None,
            substrate: None,
            sysroot_heads: None,
        }
    }

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

fn merge_heads_arc<'a>(
    dst: &mut Option<std::sync::Arc<HashMap<u32, std::sync::Arc<crate::sema::impls::Index>>>>,
    src: impl Iterator<Item = (&'a u32, &'a std::sync::Arc<crate::sema::impls::Index>)>,
) {
    let mut iter = src.peekable();
    if iter.peek().is_none() {
        return;
    }
    let map = dst.get_or_insert_with(|| std::sync::Arc::new(HashMap::default()));
    let inner = std::sync::Arc::make_mut(map);
    for (&k, v) in iter {
        inner.insert(k, std::sync::Arc::clone(v));
    }
}

fn load_with_sysroot(
    workspace: &Workspace,
    targets: &Targets,
    scripts: &[usize],
    eager_all: bool,
) -> Result<Compilation> {
    let mut outline = crate::outline::new(workspace, targets, scripts);
    let sys_key =
        crate::substrate::enabled().then(|| crate::substrate::sysroot_key(workspace, targets));
    let cached = crate::substrate::sysroot::load_cached_crates(
        workspace,
        targets,
        &mut outline,
        sys_key,
        scripts.is_empty(),
    );
    let was_miss = !cached.sysroot_hit;
    let deps_miss = !cached.deps_hit;
    let local_hits = cached.local_hits;
    let local_keys = cached.local_keys;
    let cached_names = cached.symbols.map(|syms| (syms, cached.names));
    let mut sysroot_heads = (!cached.heads.is_empty()).then(|| std::sync::Arc::new(cached.heads));
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
    if !eager_all && deps_miss {
        if was_miss {
            crate::substrate::preload_fixed_files(workspace, &targets.target.triple, &outline);
        } else {
            crate::substrate::preload_dep_files(workspace, &targets.target.triple, &outline);
        }
    }
    let cache_deps = deps_miss && crate::substrate::deps_index::can_cache(&outline);
    let selected: Vec<bool> = (0..outline.crates.len())
        .map(|krate| {
            let sysroot = outline.is_sysroot(krate as u32);
            let pinned_dep = outline.is_fixed(krate as u32) && !sysroot;
            let local_hit = local_hits.get(krate).copied().unwrap_or(false);
            if eager_all {
                (!sysroot || was_miss) && (!pinned_dep || deps_miss) && !local_hit
            } else {
                (was_miss && sysroot)
                    || (cache_deps && pinned_dep)
                    || (!local_hit
                        && workspace
                            .graph
                            .members
                            .contains(&workspace.graph.units[outline.roots[krate].unit].package))
            }
        })
        .collect();
    outline.load_selected(&selected)?;
    let names = Names::new_with(workspace, &outline, cached_names);
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
        sysroot_heads: sysroot_heads.take(),
    };
    for (krate, _) in selected.iter().enumerate().filter(|(_, on)| **on) {
        compilation.names.crates[krate].set_eager();
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
    if let Some(key) = sys_key {
        if was_miss
            && crate::scripts::sysroot_scripts_settled(workspace)
            && let Some(heads) = crate::substrate::sysroot::save(workspace, &compilation, key)
        {
            merge_heads_arc(&mut compilation.sysroot_heads, heads.iter());
            if let Some(started) = sysroot_started {
                eprintln!(
                    "haste: standard library cached in {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
        }
        if cache_deps
            && let Some(heads) = crate::substrate::deps_index::save(workspace, &compilation, key)
        {
            merge_heads_arc(&mut compilation.sysroot_heads, heads.iter());
        }
        if scripts.is_empty() {
            let heads = crate::substrate::local_index::save(
                workspace,
                targets,
                &compilation,
                key,
                &local_hits,
                &local_keys,
            );
            merge_heads_arc(&mut compilation.sysroot_heads, heads.iter());
        }
    }
    let errors = compilation.outline.local_errors(workspace);
    if !errors.is_empty() {
        anyhow::bail!("{}", errors.join("\n"));
    }
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
    store.set_sysroot_heads_if_unset(&compilation.sysroot_heads);
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
    if let Some(heads) = &mut compilation.sysroot_heads {
        let stale = heads.keys().any(|&k| {
            !compilation.outline.is_fixed(k)
                && !compilation
                    .names
                    .crates
                    .get(k as usize)
                    .is_some_and(|c| c.settled)
        });
        if stale {
            let map = std::sync::Arc::make_mut(heads);
            map.retain(|&k, _| {
                compilation.outline.is_fixed(k)
                    || compilation
                        .names
                        .crates
                        .get(k as usize)
                        .is_some_and(|c| c.settled)
            });
            if map.is_empty() {
                compilation.sysroot_heads = None;
            }
        }
    }
    let done = compilation
        .names
        .resolve(&mut compilation.outline)
        .and_then(|()| expand_bound(compilation).map(drop));
    compilation.settle();
    done
}

/// Expands every procedural macro call that resolved names bind and resolves again, loading what the binding asked for, until no call is left to bind; returns whether it expanded or loaded anything.
fn expand_bound(compilation: &mut Compilation) -> Result<bool> {
    expand_bound_scoped(compilation, None)
}

fn push_site(outline: &mut Outline, krate: usize, site: Site) {
    let holder = &mut outline.crates[krate];
    match site {
        Site::Bang(call) => holder.foreign.push(call),
        Site::Derive(held) => holder.derived.push(held),
        Site::Attribute(held) => holder.attributed.push(held),
    }
}

fn bind_procs_in(compilation: &mut Compilation, filter: Option<&[bool]>) -> (Vec<Bound>, Requests) {
    let Some(seen) = filter else {
        return compilation.names.bind_procs(&mut compilation.outline);
    };
    let mut saved = Vec::new();
    for (k, _) in seen.iter().enumerate().filter(|(_, on)| !**on) {
        let h = &mut compilation.outline.crates[k];
        if !h.foreign.is_empty() || !h.derived.is_empty() || !h.attributed.is_empty() {
            saved.push((
                k,
                std::mem::take(&mut h.foreign),
                std::mem::take(&mut h.derived),
                std::mem::take(&mut h.attributed),
            ));
        }
    }
    let result = compilation.names.bind_procs(&mut compilation.outline);
    for (k, foreign, derived, attributed) in saved {
        let h = &mut compilation.outline.crates[k];
        h.foreign.extend(foreign);
        h.derived.extend(derived);
        h.attributed.extend(attributed);
    }
    result
}

fn split_ready_bound(outline: &mut Outline, bound: Vec<Bound>) -> Vec<Bound> {
    if bound.len() <= 1 {
        return bound;
    }
    let mut deps_by_macro: HashMap<usize, Vec<bool>> = HashMap::default();
    let (mut ready, mut deferred) = (Vec::with_capacity(bound.len()), Vec::new());
    let caller_crates: Vec<usize> = bound.iter().map(|c| c.krate).collect();
    for call in bound {
        let deps = deps_by_macro
            .entry(call.run.krate as usize)
            .or_insert_with(|| dep_closure(outline, &[call.run.krate as usize]).0);
        if caller_crates
            .iter()
            .any(|&k| deps.get(k).copied().unwrap_or(false))
        {
            deferred.push(call);
        } else {
            ready.push(call);
        }
    }
    if ready.is_empty() {
        return deferred;
    }
    for call in deferred {
        push_site(outline, call.krate, call.site);
    }
    ready
}

fn expand_bound_scoped(compilation: &mut Compilation, filter: Option<&[bool]>) -> Result<bool> {
    let mut changed = false;
    loop {
        let (bound, asks) = bind_procs_in(compilation, filter);
        if bound.is_empty() {
            if asks.count() == 0 || !compilation.names.load(&mut compilation.outline, asks)? {
                return Ok(changed);
            }
            changed = true;
            continue;
        }
        let bound = split_ready_bound(&mut compilation.outline, bound);
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
            .map(|(run, text)| {
                let file = &compilation.outline.crates[run.krate as usize].files[run.file as usize];
                let kind = crate::names::collect::proc_kind(
                    &file.source,
                    &file.items.items[run.item as usize],
                )
                .map_or(ProcKind::Bang, |(_, kind)| kind);
                let caller = infer_caller(&compilation.outline, *run, kind, text);
                (*run, kind, caller)
            })
            .collect();
        let inputs: Vec<Vec<String>> = procs.iter().map(|(_, text)| vec![text.clone()]).collect();
        let outputs = run(compilation, &calls, &inputs)?;
        compilation
            .outline
            .procs
            .extend(procs.iter().cloned().zip(outputs));
    }
    let mut unsettled = false;
    for krate in asks
        .names
        .borrow()
        .iter()
        .map(|(at, _)| at.krate as usize)
        .chain(asks.scopes.borrow().iter().map(|at| at.krate as usize))
    {
        if let Some(entry) = compilation.names.crates.get_mut(krate)
            && entry.settled
            && entry.pending.iter().any(|p| p.stamp.is_none())
        {
            entry.settled = false;
            unsettled = true;
        }
    }
    let more = compilation.names.load(&mut compilation.outline, asks)?;
    let expanded = expand_bound(compilation)?;
    Ok(match (more || unsettled, expanded, procs.is_empty()) {
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
    let to_parse: Vec<_> = bound
        .iter()
        .zip(outputs)
        .map(|(c, o)| (site_path(&c.site), o))
        .collect();
    let parsed = crate::sync::parallel_map(to_parse, |(path, output)| {
        output.and_then(|text| Source::parse(path, text).map_err(|error| format!("{error:#}")))
    });
    let (mut by_crate, mut stale): (HashMap<usize, Vec<(usize, Source, String)>>, Vec<usize>) =
        (HashMap::default(), Vec::new());
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
        Site::Derive(held) => vec![derive_input_in(
            &held.file.source,
            &held.file.items.items[held.at as usize],
            &outline.crates[call.krate].env,
            Some((&outline.deps[call.krate], call.krate)),
        )],
        Site::Attribute(held) => {
            let src = &held.file.source;
            let scope = Some((outline.deps[call.krate].as_slice(), call.krate));
            let after = held.path.1 + 1;
            let args = if matches!(src.text(after), "(" | "[" | "{") {
                render_tokens(scope, src, after + 1, src.pair(after))
            } else {
                String::new()
            };
            vec![
                args,
                item_text(
                    src,
                    &held.file.items.items[held.at as usize],
                    scope,
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
    derive_input_in(src, item, env, None)
}

fn derive_input_in(
    src: &Source,
    item: &crate::outline::item::Item,
    env: &crate::outline::tree::Env,
    scope: Option<(&[(String, usize)], usize)>,
) -> String {
    item_text(
        src,
        item,
        scope,
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
    render_tokens(Some((deps, krate)), src, open + 1, src.pair(open))
}

fn push_raw_span(parts: &mut Vec<String>, src: &Source, from: u32, to: u32) {
    if from >= to {
        return;
    }
    if from < src.offset && to > src.offset {
        parts.push(src.span(from, src.offset - 1).to_owned());
        parts.push(src.span(src.offset, to - 1).to_owned());
    } else {
        parts.push(src.span(from, to - 1).to_owned());
    }
}

fn render_tokens(
    scope: Option<(&[(String, usize)], usize)>,
    src: &Source,
    from: u32,
    to: u32,
) -> String {
    let mut parts = Vec::new();
    let Some((deps, krate)) = scope else {
        push_raw_span(&mut parts, src, from, to);
        return parts.join(" ");
    };
    let mut cursor = from;
    for at in from..to {
        let Some(path) = src
            .dollar(at)
            .filter(|_| at >= cursor && at + 1 < to)
            .and_then(|named| crate_path(deps, krate, named as usize))
        else {
            continue;
        };
        push_raw_span(&mut parts, src, cursor, at);
        parts.push(path);
        cursor = at + 2;
    }
    push_raw_span(&mut parts, src, cursor, to);
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

/// The text of `item` with the outer attributes whose path starts at a token that `keep` rejects left out, and, as a derive sees it, the fields or variants whose attributes `enabled` rejects.
fn item_text(
    src: &Source,
    item: &crate::outline::item::Item,
    scope: Option<(&[(String, usize)], usize)>,
    keep: &dyn Fn(u32) -> bool,
    enabled: Option<&dyn Fn(&[Attr]) -> bool>,
) -> String {
    let mut parts = Vec::new();
    let mut at = item.lead;
    while at < item.start && src.text(at) == "#" {
        let close = src.pair(at + 1);
        if keep(at + 2) {
            parts.push(render_tokens(scope, src, at, close + 1));
        }
        at = close + 1;
    }
    match item
        .body
        .zip(enabled)
        .and_then(|(open, enabled)| Some((open, configured(src, scope, open, enabled)?)))
    {
        Some((open, members)) => parts.extend([
            render_tokens(scope, src, at, open + 1),
            members,
            render_tokens(scope, src, src.pair(open), item.end),
        ]),
        None => parts.push(render_tokens(scope, src, at, item.end)),
    }
    parts.join(" ")
}

/// The members inside the delimiter that opens at `open` whose attributes `enabled` accepts, or none when it accepts them all.
fn configured(
    src: &Source,
    scope: Option<(&[(String, usize)], usize)>,
    open: u32,
    enabled: &dyn Fn(&[Attr]) -> bool,
) -> Option<String> {
    let (mut kept, mut dropped) = (Vec::new(), false);
    for (from, to) in split_commas(src, open + 1, src.pair(open))
        .into_iter()
        .filter(|(from, to)| from < to)
    {
        let mut attrs = Vec::new();
        attr::attrs(src, from, false, &mut attrs).ok()?;
        if enabled(&attrs) {
            kept.push(if scope.is_some() {
                render_tokens(scope, src, from, to)
            } else {
                src.spelled((from, to))
            });
        } else {
            dropped = true
        }
    }
    dropped.then(|| kept.join(" , "))
}
