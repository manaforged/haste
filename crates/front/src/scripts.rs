//! Build scripts in the image: every unit's script is built from source like any crate into one image, run in a forked child, and what it prints decides its package's cfgs; scripts that read a `links` dependency's metadata run in a later wave.

use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashSet as HashSet;

use crate::graph::manifest::Kind;
use crate::graph::script::{self, Output};
use crate::graph::{Targets, Workspace, store};
use crate::image::{self, Demand, Driver, Ended, Script};
use crate::outline::Role;

/// Gives every unit with a build script its output: the stored one while it is current, else from running the script. The scripts to run build into one image, then run in waves so a script that reads a `links` dependency's metadata runs after it; a script that does not build or fails stays without output, so its package's unknown cfgs stay open.
pub fn settle(workspace: &mut Workspace, targets: &Targets) -> Result<()> {
    let mut failed = HashSet::default();
    loop {
        take_kept(workspace, targets, &mut failed);
        let mut missing = pending(workspace, &failed);
        missing.sort_by_cached_key(|&unit| store::dir(workspace, unit));
        let _held = missing.iter().map(|&unit| store::lock(workspace, unit)).collect::<std::io::Result<Vec<_>>>()?;
        take_kept(workspace, targets, &mut failed);
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
                give_up(workspace, targets, unit, &failure.error, &mut failed);
            }
        }
    }
}

/// Reports that `unit`'s script did not build or run, keeps that failure for later runs, and builds its package with the script's cfgs open.
fn give_up(workspace: &Workspace, targets: &Targets, unit: usize, error: &str, failed: &mut HashSet<usize>) {
    report(workspace, unit, error, failed);
    if let Err(write) = store::keep_failure(workspace, unit, &vars(workspace, targets, unit), error) {
        eprintln!("cannot keep that failure: {write}");
    }
}

fn report(workspace: &Workspace, unit: usize, error: &str, failed: &mut HashSet<usize>) {
    eprintln!("build script of {}: {error}", workspace.packages[workspace.graph.units[unit].package].locked.name);
    failed.insert(unit);
}

/// The variables the script of `unit` runs with, given the outputs that units already have.
fn vars(workspace: &Workspace, targets: &Targets, unit: usize) -> Vec<(String, String)> {
    script::script_vars(workspace, targets, unit, &metadata(workspace, unit, &known(workspace)))
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

/// Takes the stored result of each pending script that is still current, a script whose `links` dependency is pending only once that dependency has its result.
fn take_kept(workspace: &mut Workspace, targets: &Targets, failed: &mut HashSet<usize>) {
    loop {
        let pending = pending(workspace, failed);
        let ready = pending.iter().copied().filter(|&unit| !waits(workspace, unit, &pending));
        let found: Vec<(usize, Result<Output, String>)> = ready.filter_map(|unit| store::kept(workspace, unit, &vars(workspace, targets, unit)).map(|kept| (unit, kept))).collect();
        if found.is_empty() {
            return;
        }
        for (unit, kept) in found {
            match kept {
                Ok(output) => workspace.graph.units[unit].script = Some(Arc::new(output)),
                Err(error) => report(workspace, unit, &error, failed),
            }
        }
    }
}

/// The script outputs that units already have.
fn known(workspace: &Workspace) -> Ran {
    let units = workspace.graph.units.iter().enumerate();
    units.filter_map(|(unit, entry)| Some((unit, entry.script.clone()?))).collect()
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

/// Builds and runs the scripts of the units `units` in children that compile each function on its first call.
fn run(workspace: &Workspace, targets: &Targets, units: &[usize]) -> Result<Outcome> {
    let (mut outline, mut names) = crate::macros::front(workspace, targets, units)?;
    let scripts: Vec<(usize, usize)> = outline.roots.iter().enumerate().filter(|(_, root)| root.role == Role::Script).map(|(krate, root)| (root.unit, krate)).collect();
    let mains: Vec<Option<crate::names::ItemRef>> = scripts.iter().map(|(_, krate)| outline.main(*krate)).collect();
    let (entries, store) = image::entries(&mut outline, &mut names, &mains)?;
    let mut demand = Demand::new()?;
    let left = scripts.iter().zip(entries).map(|(&(unit, _), entry)| (unit, entry.map(|pair| pair.map(|root| demand.declare(root))))).collect();
    let owners = outline.roots.iter().map(|root| root.unit).collect();
    let mut driver = Runs { workspace, targets, owners, left, running: Vec::new(), ran: known(workspace), outcome: Outcome::new() };
    image::serve(&mut outline, &mut names, (store, &crate::macros::answer), &mut demand, &mut driver)?;
    std::mem::forget((outline, names, demand));
    Ok(driver.outcome)
}

/// The scripts left to run with their declared entries, the ones running by child, the outputs known so far and each script's outcome.
struct Runs<'w> {
    workspace: &'w Workspace,
    targets: &'w Targets,
    owners: Vec<usize>,
    left: Vec<(usize, Result<[usize; 2], String>)>,
    running: Vec<(usize, usize, Script)>,
    ran: Ran,
    outcome: Outcome,
}

impl Runs<'_> {
    /// The next script that can start: one whose `links` dependencies ran or failed, else any when every one left waits.
    fn next(&mut self) -> Option<(usize, Result<[usize; 2], String>)> {
        let waiting: Vec<usize> = self.left.iter().map(|(unit, _)| *unit).chain(self.running.iter().map(|(_, unit, _)| *unit)).collect();
        let ready = self.left.iter().position(|(unit, _)| !waits(self.workspace, *unit, &waiting));
        let at = ready.or_else(|| self.running.is_empty().then_some(0)).filter(|_| !self.left.is_empty())?;
        Some(self.left.remove(at))
    }
}

impl Driver for Runs<'_> {
    fn start(&mut self, demand: &mut Demand) -> Result<()> {
        while self.running.len() < crate::cores() {
            let Some((unit, entry)) = self.next() else { return Ok(()) };
            let script = entry.map_err(Failure::from).and_then(|[start, main]| job(self.workspace, self.targets, unit, (start, main), &self.ran));
            match script {
                Ok(script) => {
                    let id = demand.fork(true, |image, out| out.map_or(101, |out| image.enter_script(&script, out)))?;
                    self.running.push((id, unit, script));
                }
                Err(failure) => self.outcome.push((unit, Err(failure))),
            }
        }
        Ok(())
    }

    fn ended(&mut self, id: usize, ended: Ended) -> Result<()> {
        let Some(at) = self.running.iter().position(|(running, ..)| *running == id) else { return Ok(()) };
        let (_, unit, script) = self.running.swap_remove(at);
        let result = match (ended.status, ended.failure) {
            (0, _) => String::from_utf8(ended.bytes).map(|text| finished(self.workspace, unit, &script, &text)).map_err(|_| Failure::from("printed output that is not UTF-8".to_owned())),
            (_, Some((error, item))) => Err(Failure { error: format!("does not build: {error}"), broken: self.owners.get(item.krate as usize).copied() }),
            (status, None) => Err(Failure::from(format!("exited with status {status}; its errors are in {}", script.log.display()))),
        };
        if let Ok(output) = &result {
            self.ran.insert(unit, output.clone());
        }
        self.outcome.push((unit, result));
        Ok(())
    }
}

/// The output of a script that exited cleanly, kept in the store for later runs.
fn finished(workspace: &Workspace, unit: usize, job: &Script, text: &str) -> Arc<Output> {
    if let Err(error) = store::keep(workspace, unit, &job.vars, text) {
        eprintln!("cannot keep the output of {}'s build script: {error}", workspace.packages[workspace.graph.units[unit].package].locked.name);
    }
    Arc::new(Output::parse(text))
}

fn job(workspace: &Workspace, targets: &Targets, unit: usize, entry: (usize, usize), ran: &Ran) -> Result<Script, Failure> {
    let dir = store::dir(workspace, unit);
    let made = store::forget(workspace, unit).and_then(|()| std::fs::create_dir_all(dir.join("out")));
    made.map_err(|error| Failure::from(format!("cannot make {}: {error}", dir.display())))?;
    let vars = script::script_vars(workspace, targets, unit, &metadata(workspace, unit, ran));
    let package = &workspace.packages[workspace.graph.units[unit].package].manifest;
    Ok(Script { entry, vars, dir: package.dir.clone(), log: dir.join("stderr") })
}
