//! Procedural macro execution, expansion caching, dependency preloading, and graph stamping.

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::store;
use crate::image::{Jit, Job, run_macros};
use crate::names::ItemRef;
use crate::names::path::Requests;
use crate::outline::Outline;
use crate::outline::item::{ItemKind, ProcKind};
use crate::outline::tree::PROC_MACRO_ROOT;
use crate::sema::{Answer, Cache};

use super::{Compilation, ENTRIES, Expander, expand_bound_scoped, macro_name};

/// A procedural macro call: the macro function, its kind, and the crate that calls it, `usize::MAX` for a call in a function body.
pub(super) type Call = (ItemRef, ProcKind, usize);

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

/// The output of each call `bound` on its `inputs`, or why it failed: the result the store kept for a macro of a pinned crate, else from running it, kept when it ran.
pub(super) fn run(
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
    let checks: Vec<_> = keys
        .iter()
        .enumerate()
        .filter_map(|(at, k)| Some((at, k.as_ref()?)))
        .collect();
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

pub(super) fn dep_closure(outline: &Outline, roots: &[usize]) -> (Vec<bool>, Vec<usize>) {
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
    if selected.iter().any(|&on| on) {
        compilation.outline.load_selected(&selected)?;
        for (krate, _) in selected.iter().enumerate().filter(|(_, on)| **on) {
            compilation.names.crates[krate].set_eager();
        }
        compilation.names.resolve(&mut compilation.outline)?;
    }
    let has_pending = seen.iter().enumerate().any(|(k, &on)| {
        on && {
            let h = &compilation.outline.crates[k];
            !h.foreign.is_empty() || !h.derived.is_empty() || !h.attributed.is_empty()
        }
    });
    if has_pending && expand_bound_scoped(compilation, Some(&seen))? {
        compilation.expander = None;
    }
    Ok(())
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

pub(super) fn macro_graph_stamp(outline: &Outline, roots: &[usize]) -> u64 {
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
