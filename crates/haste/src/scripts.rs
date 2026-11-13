//! Build scripts in the image: every unit's script is built from source like any crate into one image, run in a forked child, and what it prints decides its package's cfgs; scripts that read a `links` dependency's metadata wait only on their prerequisites.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph::manifest::Kind;
use crate::graph::script::{self, Output};
use crate::graph::{Targets, Workspace, store};
use crate::image::{self, Driver, Ended, Jit, Script};
use crate::outline::Role;

/// Gives every unit with a build script its output: the stored one while it is current, else from running the script.
pub fn settle(workspace: &mut Workspace, targets: &Targets) -> Result<()> {
    let mut failed = HashSet::default();
    let mut dag = ScriptDag::new(workspace);
    let mut ran = known(workspace);
    let mut kept_queue = dag.initial_ready(workspace);
    drain_kept(workspace, targets, &mut dag, &mut kept_queue, &mut ran, &mut failed, None);
    loop {
        let mut ready = dag.ready_build_units(workspace, &failed);
        if ready.is_empty() {
            return Ok(());
        }
        ready.sort_by_cached_key(|&unit| store::dir(workspace, unit));
        let mut locks = HashMap::default();
        for &unit in &ready {
            locks.insert(unit, store::lock(workspace, unit)?);
        }
        for &unit in &ready {
            if dag.links_rem[unit] == 0 {
                kept_queue.push_back(unit);
            }
        }
        drain_kept(workspace, targets, &mut dag, &mut kept_queue, &mut ran, &mut failed, Some(&mut locks));
        let mut wave: Vec<usize> = ready.into_iter().filter(|unit| workspace.graph.units[*unit].script.is_none() && !failed.contains(unit)).collect();
        if wave.is_empty() {
            continue;
        }
        wave.sort_by_key(|&unit| std::cmp::Reverse((dag.build_users[unit].len(), dag.links_users[unit].len())));
        let outcome = run(workspace, targets, &wave, &dag, &mut locks, &mut ran)?;
        drop(locks);
        finish_wave(workspace, targets, &wave, outcome, &mut dag, &mut ran, &mut failed, &mut kept_queue);
    }
}

struct ScriptDag {
    build_deps: Vec<Vec<usize>>,
    build_users: Vec<Vec<usize>>,
    links_users: Vec<Vec<usize>>,
    build_rem: Vec<usize>,
    links_rem: Vec<usize>,
    script_units: Vec<usize>,
}

impl ScriptDag {
    fn new(workspace: &Workspace) -> Self {
        let units = &workspace.graph.units;
        let n = units.len();
        let is_script: Vec<bool> = units.iter().map(|entry| workspace.packages[entry.package].manifest.build.is_some()).collect();
        let script_units: Vec<usize> = (0..n).filter(|&u| is_script[u] && units[u].script.is_none()).collect();
        let mut build_deps = vec![Vec::new(); n];
        let mut build_users = vec![Vec::new(); n];
        let mut links_users = vec![Vec::new(); n];
        let mut build_rem = vec![0; n];
        let mut links_rem = vec![0; n];
        for &unit in &script_units {
            let closure = crate::outline::build_closure(workspace, &[unit]);
            for dep in closure {
                if dep != unit && is_script[dep] && units[dep].script.is_none() {
                    build_deps[unit].push(dep);
                    build_users[dep].push(unit);
                    build_rem[unit] += 1;
                }
            }
            let mut seen_links = HashSet::default();
            for &(_, dep, kind) in &units[unit].deps {
                if kind == Kind::Normal
                    && dep != unit
                    && is_script[dep]
                    && units[dep].script.is_none()
                    && workspace.packages[units[dep].package].manifest.links.is_some()
                    && seen_links.insert(dep)
                {
                    links_users[dep].push(unit);
                    links_rem[unit] += 1;
                }
            }
        }
        Self { build_deps, build_users, links_users, build_rem, links_rem, script_units }
    }

    fn initial_ready(&self, workspace: &Workspace) -> VecDeque<usize> {
        let mut ready = VecDeque::new();
        for &unit in &self.script_units {
            if workspace.graph.units[unit].script.is_none() && self.build_rem[unit] == 0 && self.links_rem[unit] == 0 {
                ready.push_back(unit);
            }
        }
        ready
    }

    fn ready_build_units(&self, workspace: &Workspace, failed: &HashSet<usize>) -> Vec<usize> {
        let mut pending = Vec::new();
        let mut ready = Vec::new();
        for &unit in &self.script_units {
            if workspace.graph.units[unit].script.is_none() && !failed.contains(&unit) {
                pending.push(unit);
                if self.build_rem[unit] == 0 {
                    ready.push(unit);
                }
            }
        }
        if ready.is_empty() { pending } else { ready }
    }

    fn finish_unit(&mut self, workspace: &Workspace, failed: &HashSet<usize>, unit: usize, kept_queue: &mut VecDeque<usize>) {
        for &user in &self.build_users[unit] {
            self.build_rem[user] = self.build_rem[user].saturating_sub(1);
            if self.build_rem[user] == 0 && self.links_rem[user] == 0 && workspace.graph.units[user].script.is_none() && !failed.contains(&user) {
                kept_queue.push_back(user);
            }
        }
        for &user in &self.links_users[unit] {
            self.links_rem[user] = self.links_rem[user].saturating_sub(1);
            if self.build_rem[user] == 0 && self.links_rem[user] == 0 && workspace.graph.units[user].script.is_none() && !failed.contains(&user) {
                kept_queue.push_back(user);
            }
        }
    }
}

fn drain_kept(
    workspace: &mut Workspace,
    targets: &Targets,
    dag: &mut ScriptDag,
    kept_queue: &mut VecDeque<usize>,
    ran: &mut Ran,
    failed: &mut HashSet<usize>,
    mut locks: Option<&mut HashMap<usize, store::Lock>>,
) {
    while let Some(unit) = kept_queue.pop_front() {
        if workspace.graph.units[unit].script.is_some() || failed.contains(&unit) {
            continue;
        }
        if let Some(locks) = locks.as_deref()
            && !locks.contains_key(&unit)
        {
            continue;
        }
        let unit_vars = vars(workspace, targets, unit, ran);
        let Some(kept) = store::kept(workspace, unit, &unit_vars) else { continue };
        match kept {
            Ok(output) => {
                let output = Arc::new(output);
                workspace.graph.units[unit].script = Some(output.clone());
                ran.insert(unit, output);
            }
            Err(error) => report(workspace, unit, &error, failed),
        }
        if let Some(locks) = locks.as_deref_mut() {
            locks.remove(&unit);
        }
        dag.finish_unit(workspace, failed, unit, kept_queue);
    }
}

fn finish_wave(
    workspace: &mut Workspace,
    targets: &Targets,
    wave: &[usize],
    outcome: Outcome,
    dag: &mut ScriptDag,
    ran: &mut Ran,
    failed: &mut HashSet<usize>,
    kept_queue: &mut VecDeque<usize>,
) {
    let mut wave_set = HashSet::default();
    for &unit in wave {
        wave_set.insert(unit);
    }
    let mut errors = Vec::new();
    for (unit, result) in outcome {
        match result {
            Ok(output) => {
                workspace.graph.units[unit].script = Some(output.clone());
                ran.insert(unit, output);
                dag.finish_unit(workspace, failed, unit, kept_queue);
            }
            Err(failure) => errors.push((unit, failure)),
        }
    }
    for (unit, failure) in errors {
        let retried = failure.broken.is_some_and(|at| wave_set.contains(&at) && workspace.graph.units[at].script.is_some());
        if !retried {
            give_up(workspace, targets, unit, &failure, ran, failed);
            dag.finish_unit(workspace, failed, unit, kept_queue);
        }
    }
}

/// Reports that `unit`'s script did not build or run, keeps that failure for later runs unless something outside stopped the script, and builds its package with the script's cfgs open.
fn give_up(workspace: &Workspace, targets: &Targets, unit: usize, failure: &Failure, ran: &Ran, failed: &mut HashSet<usize>) {
    let error = &failure.error;
    report(workspace, unit, error, failed);
    if !failure.interrupted
        && let Err(write) = store::keep_failure(workspace, unit, &vars(workspace, targets, unit, ran), error)
    {
        eprintln!("cannot keep that failure: {write}");
    }
}

fn report(workspace: &Workspace, unit: usize, error: &str, failed: &mut HashSet<usize>) {
    eprintln!("build script of {}: {error}", workspace.packages[workspace.graph.units[unit].package].locked.name);
    failed.insert(unit);
}

/// The variables the script of `unit` runs with, given the outputs that units already have.
fn vars(workspace: &Workspace, targets: &Targets, unit: usize, ran: &Ran) -> Vec<(String, String)> {
    script::script_vars(workspace, targets, unit, &metadata(workspace, unit, ran))
}

/// Why a script has no output, with the unit of the crate whose function did not build, when one did not: a script whose broken crate had its own script run since builds again, since that script's cfgs may fix it.
struct Failure {
    error: String,
    broken: Option<usize>,
    interrupted: bool,
}

impl From<String> for Failure {
    fn from(error: String) -> Failure {
        Failure { error, broken: None, interrupted: false }
    }
}

/// The script outputs that units already have.
fn known(workspace: &Workspace) -> Ran {
    let units = workspace.graph.units.iter().enumerate();
    units.filter_map(|(unit, entry)| Some((unit, entry.script.clone()?))).collect()
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
fn run(workspace: &Workspace, targets: &Targets, units: &[usize], dag: &ScriptDag, locks: &mut HashMap<usize, store::Lock>, ran: &mut Ran) -> Result<Outcome> {
    let mut compilation = crate::macros::load_eager(workspace, targets, units)?;
    let outline = &compilation.outline;
    let scripts: Vec<(usize, usize)> = outline.roots.iter().enumerate().filter(|(_, root)| root.role == Role::Script).map(|(krate, root)| (root.unit, krate)).collect();
    let mains: Vec<Option<crate::names::ItemRef>> = scripts.iter().map(|(_, krate)| outline.main(*krate)).collect();
    let owners = outline.roots.iter().map(|root| root.unit).collect();
    let (entries, store) = image::entries(&mut compilation, (crate::sema::Cache::default(), &mains))?;
    let mut demand = Jit::new(store)?;
    demand.set_fast(true);
    let mut unit_set = HashSet::default();
    for &(unit, _) in &scripts {
        unit_set.insert(unit);
    }
    let mut entries_map = HashMap::default();
    let mut ready = VecDeque::new();
    let mut waiting = Vec::new();
    let mut wave_links_rem = HashMap::default();
    let mut active_build_rem = HashMap::default();
    for (&(unit, _), entry) in scripts.iter().zip(entries) {
        let declared = entry.map(|pair| pair.map(|root| demand.declare(root)));
        entries_map.insert(unit, declared);
        let lrem = dag.links_rem[unit];
        let brem = dag.build_deps[unit].iter().filter(|dep| unit_set.contains(dep)).count();
        wave_links_rem.insert(unit, lrem);
        active_build_rem.insert(unit, brem);
        if lrem == 0 && brem == 0 {
            ready.push_back(unit);
        } else {
            waiting.push(unit);
        }
    }
    let mut driver = Runs {
        workspace,
        targets,
        owners,
        dag,
        locks,
        entries: entries_map,
        ready,
        waiting,
        wave_links_rem,
        active_build_rem,
        running: Vec::new(),
        ran: ran.clone(),
        outcome: Outcome::new(),
    };
    image::serve(&mut compilation, &crate::macros::load, &mut demand, &mut driver)?;
    Ok(driver.outcome)
}

struct Runs<'w, 'l> {
    workspace: &'w Workspace,
    targets: &'w Targets,
    owners: Vec<usize>,
    dag: &'w ScriptDag,
    locks: &'l mut HashMap<usize, store::Lock>,
    entries: HashMap<usize, Result<[usize; 2], String>>,
    ready: VecDeque<usize>,
    waiting: Vec<usize>,
    wave_links_rem: HashMap<usize, usize>,
    active_build_rem: HashMap<usize, usize>,
    running: Vec<(usize, usize, Script)>,
    ran: Ran,
    outcome: Outcome,
}

impl Runs<'_, '_> {
    fn next(&mut self) -> Option<(usize, Result<[usize; 2], String>)> {
        let unit = self.ready.pop_front().or_else(|| self.running.is_empty().then(|| (!self.waiting.is_empty()).then(|| self.waiting.remove(0))).flatten())?;
        let entry = self.entries.remove(&unit)?;
        Some((unit, entry))
    }

    fn complete_unit(&mut self, unit: usize) {
        self.locks.remove(&unit);
        for &user in &self.dag.links_users[unit] {
            if let Some(rem) = self.wave_links_rem.get_mut(&user) {
                *rem = rem.saturating_sub(1);
                self.try_wake(user);
            }
        }
        for &user in &self.dag.build_users[unit] {
            if let Some(rem) = self.active_build_rem.get_mut(&user) {
                *rem = rem.saturating_sub(1);
                self.try_wake(user);
            }
        }
    }

    fn try_wake(&mut self, user: usize) {
        if self.wave_links_rem.get(&user).copied() == Some(0)
            && self.active_build_rem.get(&user).copied() == Some(0)
            && let Some(pos) = self.waiting.iter().position(|&u| u == user)
        {
            self.waiting.swap_remove(pos);
            self.ready.push_back(user);
        }
    }
}

impl Driver for Runs<'_, '_> {
    fn start(&mut self, demand: &mut Jit) -> Result<()> {
        while self.running.len() < crate::jobs() {
            let Some((unit, entry)) = self.next() else {
                return Ok(());
            };
            let script = entry.map_err(Failure::from).and_then(|[start, main]| job(self.workspace, self.targets, unit, (start, main), &self.ran));
            match script {
                Ok(script) => {
                    let id = demand.fork(true, |image, out| out.map_or(101, |out| image.enter_script(&script, out)))?;
                    self.running.push((id, unit, script));
                }
                Err(failure) => {
                    self.complete_unit(unit);
                    self.outcome.push((unit, Err(failure)));
                }
            }
        }
        Ok(())
    }

    fn ended(&mut self, id: usize, ended: Ended) -> Result<()> {
        let Some(at) = self.running.iter().position(|(running, ..)| *running == id) else {
            return Ok(());
        };
        let (_, unit, script) = self.running.swap_remove(at);
        let result = match (ended.status, ended.failure) {
            (0, _) => String::from_utf8(ended.bytes).map(|text| finished(self.workspace, unit, &script, &text)).map_err(|_| Failure::from("printed output that is not UTF-8".to_owned())),
            (_, Some((error, item))) => Err(Failure { error: format!("does not build: {error}"), broken: self.owners.get(item.krate as usize).copied(), interrupted: false }),
            (status, None) => Err(Failure { interrupted: crate::image::interrupted(status), ..Failure::from(format!("exited with status {status}; its errors are in {}", script.log.display())) }),
        };
        if let Ok(output) = &result {
            self.ran.insert(unit, output.clone());
        }
        self.complete_unit(unit);
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
