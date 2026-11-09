//! Build scripts in the image: every unit's script is built from source like any crate into one image, run in a forked child, and what it prints decides its package's cfgs; scripts that read a `links` dependency's metadata run in a later wave.

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

/// Gives every unit with a build script the output of running that script in this run: all of them build into one image, then run in waves so a script that reads a `links` dependency's metadata runs after it; a script that does not build or fails stays without output, so its package's unknown cfgs stay open.
pub fn settle(workspace: &mut Workspace, targets: &Targets) -> Result<()> {
    let mut failed = HashSet::default();
    loop {
        let pending = pending(workspace, &failed);
        if pending.is_empty() {
            return Ok(());
        }
        let outcome = run(workspace, targets, &pending)?;
        let mut errors = Vec::new();
        for (unit, result) in outcome {
            match result {
                Ok(output) => workspace.graph.units[unit].script = Some(output),
                Err(failure) => errors.push((unit, failure)),
            }
        }
        for (unit, failure) in errors {
            let ran = failure.broken.is_some_and(|at| pending.contains(&at) && workspace.graph.units[at].script.is_some());
            if !ran {
                eprintln!("build script of {}: {}", workspace.packages[workspace.graph.units[unit].package].locked.name, failure.error);
                failed.insert(unit);
            }
        }
    }
}

/// Why a script has no output, with the unit of the crate whose function did not build, when one did not: a script whose broken crate had its own script run since builds again, since that script's cfgs may fix it.
struct Failure {
    error: String,
    broken: Option<usize>,
}

impl From<String> for Failure {
    fn from(error: String) -> Failure {
        Failure { error, broken: None }
    }
}

/// The units with a build script that has neither run nor failed in this run.
fn pending(workspace: &Workspace, failed: &HashSet<usize>) -> Vec<usize> {
    let units = workspace.graph.units.iter().enumerate();
    units.filter(|(unit, entry)| entry.script.is_none() && !failed.contains(unit) && workspace.packages[entry.package].manifest.build.is_some()).map(|(unit, _)| unit).collect()
}

/// Whether the script of `unit` needs the metadata of a dependency's script that has not run.
fn waits(workspace: &Workspace, unit: usize, left: &[usize]) -> bool {
    let deps = workspace.graph.units[unit].deps.iter().filter(|(_, _, kind)| *kind == Kind::Normal);
    deps.map(|(_, dep, _)| *dep).any(|dep| dep != unit && left.contains(&dep) && workspace.packages[workspace.graph.units[dep].package].manifest.links.is_some())
}

/// The `DEP_<links>_<key>` variables that the `links` dependencies of `unit` printed, from `ran` in this run.
fn metadata(workspace: &Workspace, unit: usize, ran: &Ran) -> Vec<(String, String)> {
    let deps = workspace.graph.units[unit].deps.iter().filter(|(_, _, kind)| *kind == Kind::Normal).map(|(_, dep, _)| *dep);
    let linked = deps.filter_map(|dep| Some((workspace.packages[workspace.graph.units[dep].package].manifest.links.as_deref()?, ran.get(&dep)?)));
    linked.flat_map(|(links, output)| script::dep_vars(links, output)).collect()
}

type Ran = rustc_hash::FxHashMap<usize, Arc<Output>>;
type Outcome = Vec<(usize, Result<Arc<Output>, Failure>)>;
type Built<'r> = (&'r Image, &'r Reached, &'r rustc_hash::FxHashMap<usize, String>, &'r [usize]);

/// Builds the scripts of the units `units` into one image and runs each one that built.
fn run(workspace: &Workspace, targets: &Targets, units: &[usize]) -> Result<Outcome> {
    let (mut outline, mut names) = crate::macros::front(workspace, targets, units)?;
    let scripts: Vec<(usize, usize)> = outline.roots.iter().enumerate().filter(|(_, root)| root.role == Role::Script).map(|(krate, root)| (root.unit, krate)).collect();
    let mains: Vec<Option<crate::names::ItemRef>> = scripts.iter().map(|(_, krate)| outline.main(*krate)).collect();
    let entries = |cx: &mut Context<'_>| -> Vec<Result<[Instance; 2], String>> {
        mains.iter().map(|main| main.ok_or_else(|| "no `fn main`".to_owned()).and_then(|item| main_roots(cx, item).map_err(|error| format!("{error:#}")))).collect()
    };
    let roots = |cx: &mut Context<'_>| entries(cx).into_iter().flatten().flatten().collect();
    let finish = |mut cx: Context<'_>, reached: Reached| {
        let entries = entries(&mut cx);
        let (image, broken) = Image::build(&mut cx, &reached)?;
        let owners: Vec<usize> = cx.outline.roots.iter().map(|root| root.unit).collect();
        waves(workspace, targets, (&image, &reached, &broken, &owners), scripts.iter().map(|(unit, _)| *unit).zip(entries).collect())
    };
    crate::macros::reach_settled(&mut outline, &mut names, roots, finish)
}

/// Runs the built scripts in waves: each wave holds the scripts whose `links` dependencies already ran or failed.
fn waves(workspace: &Workspace, targets: &Targets, built: Built<'_>, mut left: Vec<(usize, Result<[Instance; 2], String>)>) -> Result<Outcome> {
    let (mut outcome, mut ran) = (Outcome::new(), Ran::default());
    while !left.is_empty() {
        let units: Vec<usize> = left.iter().map(|(unit, _)| *unit).collect();
        let (mut wave, rest): (Vec<_>, Vec<_>) = left.into_iter().partition(|(unit, _)| !waits(workspace, *unit, &units));
        left = if wave.is_empty() { std::mem::replace(&mut wave, rest) } else { rest };
        for (unit, result) in execute(workspace, targets, built, wave, &ran)? {
            if let Ok(output) = &result {
                ran.insert(unit, output.clone());
            }
            outcome.push((unit, result));
        }
    }
    Ok(outcome)
}

fn execute(workspace: &Workspace, targets: &Targets, built: Built<'_>, entries: Vec<(usize, Result<[Instance; 2], String>)>, ran: &Ran) -> Result<Outcome> {
    let mut outcome: Outcome = Vec::new();
    let mut jobs = Vec::new();
    for (unit, entry) in entries {
        let job = entry.map_err(Failure::from).and_then(|[start, main]| job(workspace, targets, unit, (built.1.index[&start], built.1.index[&main]), built, ran));
        match job {
            Ok(job) => jobs.push((unit, job)),
            Err(failure) => outcome.push((unit, Err(failure))),
        }
    }
    let threads = crate::cores();
    let (units, scripts): (Vec<usize>, Vec<Script>) = jobs.into_iter().unzip();
    let results = built.0.run_scripts(&scripts, threads)?;
    for ((unit, job), (status, text)) in units.into_iter().zip(&scripts).zip(results) {
        let result = match status {
            0 => String::from_utf8(text).map(|text| Arc::new(Output::parse(&text))).map_err(|_| Failure::from("printed output that is not UTF-8".to_owned())),
            _ => Err(Failure::from(format!("exited with status {status}; its errors are in {}", job.log.display()))),
        };
        outcome.push((unit, result));
    }
    Ok(outcome)
}

fn job(workspace: &Workspace, targets: &Targets, unit: usize, entry: (usize, usize), (_, reached, broken, owners): Built<'_>, ran: &Ran) -> Result<Script, Failure> {
    let closure = reached.closure(&[entry.0, entry.1]);
    if let Some((at, error)) = reached.failure(&closure, broken) {
        let broken = owners.get(reached.list[at].0.0.krate as usize).copied();
        return Err(Failure { error: format!("does not build: {error}"), broken });
    }
    let dir = script::dir(workspace, unit);
    std::fs::create_dir_all(dir.join("out")).map_err(|error| Failure::from(format!("cannot make {}: {error}", dir.display())))?;
    let vars = script::script_vars(workspace, targets, unit, &metadata(workspace, unit, ran));
    let package = &workspace.packages[workspace.graph.units[unit].package].manifest;
    Ok(Script { entry, closure, vars, dir: package.dir.clone(), log: dir.join("stderr") })
}
