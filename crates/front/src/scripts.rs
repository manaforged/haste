//! Build scripts in the image: each unit's script is built from source like any crate, run in a forked child, and what it prints decides its package's cfgs; scripts whose dependencies still wait on other scripts run in a later round.

use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::graph::manifest::Kind;
use crate::graph::script::{self, Output};
use crate::graph::{Targets, Workspace};
use crate::image::{Image, Script};
use crate::outline::Role;
use crate::sema::reach::{Instance, Reached, main_roots};
use crate::sema::Context;

/// Gives every unit with a build script that script's output: the kept one while the files it watches are unchanged, else from running it; a script that does not build or fails stays without output, so its package's unknown cfgs stay open.
pub fn settle(workspace: &mut Workspace, targets: &Targets) -> Result<()> {
    let mut failed = HashSet::default();
    loop {
        let pending = pending(workspace, &failed);
        let ready: Vec<usize> = pending.iter().copied().filter(|unit| !waits(workspace, *unit, &pending)).collect();
        if ready.is_empty() {
            return Ok(());
        }
        let (mut progressed, mut unbuilt) = (false, Vec::new());
        for (unit, result) in run(workspace, targets, &ready)? {
            match result {
                Ok(output) => {
                    workspace.graph.units[unit].script = Some(Arc::new(output));
                    progressed = true;
                }
                Err(Failure::Build(error)) => unbuilt.push((unit, error)),
                Err(Failure::Run(error)) => fail(workspace, unit, &error, &mut failed),
            }
        }
        if !progressed {
            unbuilt.iter().for_each(|(unit, error)| fail(workspace, *unit, error, &mut failed));
            return Ok(());
        }
    }
}

/// Why a script has no output: it did not build, which other scripts' output may change, or it ran and failed.
enum Failure {
    Build(String),
    Run(String),
}

/// Reports that `unit`'s script failed and keeps the failure, so later runs skip the script until it or this program changes.
fn fail(workspace: &Workspace, unit: usize, error: &str, failed: &mut HashSet<usize>) {
    eprintln!("build script of {}: {error}", workspace.packages[workspace.graph.units[unit].package].locked.name);
    let dir = script::dir(workspace, unit);
    if let Err(write) = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(dir.join("failed"), error)) {
        eprintln!("cannot keep that failure in {}: {write}", dir.display());
    }
    failed.insert(unit);
}

/// The units with a build script and no output yet, after taking the kept outputs that are still current.
fn pending(workspace: &mut Workspace, failed: &HashSet<usize>) -> Vec<usize> {
    let mut found = Vec::new();
    for unit in 0..workspace.graph.units.len() {
        let entry = &workspace.graph.units[unit];
        if entry.script.is_some() || failed.contains(&unit) || workspace.packages[entry.package].manifest.build.is_none() || script::failed_before(workspace, unit) {
            continue;
        }
        match script::kept(workspace, unit) {
            Some(output) => workspace.graph.units[unit].script = Some(Arc::new(output)),
            None => found.push(unit),
        }
    }
    found
}

/// Whether the script of `unit` needs the metadata of a dependency's script that has not run.
fn waits(workspace: &Workspace, unit: usize, pending: &[usize]) -> bool {
    let deps = workspace.graph.units[unit].deps.iter().filter(|(_, _, kind)| *kind == Kind::Normal);
    deps.map(|(_, dep, _)| *dep).any(|dep| dep != unit && pending.contains(&dep) && workspace.packages[workspace.graph.units[dep].package].manifest.links.is_some())
}

fn metadata(workspace: &Workspace, unit: usize) -> Vec<(String, String)> {
    let deps = workspace.graph.units[unit].deps.iter().filter(|(_, _, kind)| *kind == Kind::Normal).map(|(_, dep, _)| &workspace.graph.units[*dep]);
    let linked = deps.filter_map(|dep| Some((workspace.packages[dep.package].manifest.links.as_deref()?, dep.script.as_deref()?)));
    linked.flat_map(|(links, output)| script::dep_vars(links, output)).collect()
}

type Outcome = Vec<(usize, Result<Output, Failure>)>;

/// Builds the scripts of the units `ready` into one image, runs each one that built, and keeps what each printed.
fn run(workspace: &Workspace, targets: &Targets, ready: &[usize]) -> Result<Outcome> {
    let (mut outline, mut names) = crate::macros::front(workspace, targets, ready)?;
    let scripts: Vec<(usize, usize)> = outline.roots.iter().enumerate().filter(|(_, root)| root.role == Role::Script).map(|(krate, root)| (root.unit, krate)).collect();
    let mains: Vec<Option<crate::names::ItemRef>> = scripts.iter().map(|(_, krate)| outline.main(*krate)).collect();
    let entries = |cx: &mut Context<'_>| -> Vec<Result<[Instance; 2], String>> {
        mains.iter().map(|main| main.ok_or_else(|| "no `fn main`".to_owned()).and_then(|item| main_roots(cx, item).map_err(|error| format!("{error:#}")))).collect()
    };
    let roots = |cx: &mut Context<'_>| entries(cx).into_iter().flatten().flatten().collect();
    let finish = |mut cx: Context<'_>, reached: Reached| {
        let entries = entries(&mut cx);
        let mut image = Image::new(&reached)?;
        let all: Vec<usize> = (0..reached.list.len()).collect();
        let broken: rustc_hash::FxHashMap<usize, String> = image.compile(&mut cx, &reached, &all).into_iter().collect();
        execute(workspace, targets, (&image, &reached, &broken), scripts.iter().map(|(unit, _)| *unit).zip(entries).collect())
    };
    crate::macros::reach_settled(&mut outline, &mut names, roots, finish)
}

fn execute(workspace: &Workspace, targets: &Targets, (image, reached, broken): (&Image, &Reached, &rustc_hash::FxHashMap<usize, String>), entries: Vec<(usize, Result<[Instance; 2], String>)>) -> Result<Outcome> {
    let mut outcome: Outcome = Vec::new();
    let mut jobs = Vec::new();
    for (unit, entry) in entries {
        let job = entry.and_then(|[start, main]| job(workspace, targets, unit, (reached.index[&start], reached.index[&main]), (reached, broken)));
        match job {
            Ok(job) => jobs.push((unit, job)),
            Err(error) => outcome.push((unit, Err(Failure::Build(error)))),
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let (units, scripts): (Vec<usize>, Vec<Script>) = jobs.into_iter().unzip();
    let results = image.run_scripts(&scripts, threads)?;
    for ((unit, job), (status, text)) in units.into_iter().zip(&scripts).zip(results) {
        let dir = job.log.parent().map(std::path::Path::to_owned).unwrap_or_default();
        let result = match status {
            0 => std::fs::write(dir.join("output"), &text).map(|()| Output::parse(&String::from_utf8_lossy(&text))).map_err(|error| Failure::Run(format!("cannot keep its output: {error}"))),
            _ => Err(Failure::Run(format!("exited with status {status}; its errors are in {}", job.log.display()))),
        };
        outcome.push((unit, result));
    }
    Ok(outcome)
}

fn job(workspace: &Workspace, targets: &Targets, unit: usize, entry: (usize, usize), (reached, broken): (&Reached, &rustc_hash::FxHashMap<usize, String>)) -> Result<Script, String> {
    let mut seen = HashSet::default();
    let closure: Vec<usize> = reached.closure(entry.0).into_iter().chain(reached.closure(entry.1)).filter(|at| seen.insert(*at)).collect();
    if let Some(error) = closure.iter().find_map(|at| reached.list[*at].1.as_ref().err().or_else(|| broken.get(at))) {
        return Err(format!("does not build: {error}"));
    }
    let dir = script::dir(workspace, unit);
    std::fs::create_dir_all(dir.join("out")).map_err(|error| format!("cannot make {}: {error}", dir.display()))?;
    let vars = script::script_vars(workspace, targets, unit, &metadata(workspace, unit));
    let package = &workspace.packages[workspace.graph.units[unit].package].manifest;
    Ok(Script { entry, closure, vars, dir: package.dir.clone(), log: dir.join("stderr") })
}
