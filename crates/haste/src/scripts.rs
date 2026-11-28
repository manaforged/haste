//! Build scripts in the image: every unit's script is built from source like any crate into one image, run in a forked child, and what it prints decides its package's cfgs; scripts that read a `links` dependency's metadata wait only on their prerequisites.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph::manifest::Kind;
use crate::graph::script::{self, ChildEnv, Output, apply_client_env, prepare_client_env};
use crate::graph::{Targets, Workspace, store};
use crate::image::{self, Driver, Ended, Jit, Script};
use crate::outline::Role;

mod binary;
mod fast;
#[cfg(test)]
mod tests;

use binary::{Binaries, Kept};
use fast::WaveQueue;

const OUT: &str = "out";

/// Whether every sysroot unit that has a build script already has its settled script output.
pub(crate) fn sysroot_scripts_settled(workspace: &Workspace) -> bool {
    workspace.graph.units.iter().all(|unit| {
        let pkg = &workspace.packages[unit.package];
        !pkg.sysroot || pkg.manifest.build.is_none() || unit.script.is_some()
    })
}

/// Gives every unit with a build script its output: the stored one while it is current, else from running the script.
pub fn settle(workspace: &mut Workspace, targets: &Targets) -> Result<()> {
    crate::with_host_target(|| settle_inner(workspace, targets))
}

fn attach_rustflag_cfgs(workspace: &mut Workspace, targets: &Targets, after_scripts: bool) {
    let (names, pairs) = script::rustflag_cfgs(&workspace.rustflags);
    if names.is_empty() && pairs.is_empty() {
        return;
    }
    for unit in 0..workspace.graph.units.len() {
        if !script::rustflags_apply(workspace, targets, unit) {
            continue;
        }
        let has_build = workspace.packages[workspace.graph.units[unit].package]
            .manifest
            .build
            .is_some();
        let entry = &mut workspace.graph.units[unit];
        if after_scripts && has_build {
            if let Some(output) = &mut entry.script {
                let out = Arc::make_mut(output);
                out.names.extend(names.iter().cloned());
                out.pairs.extend(pairs.iter().cloned());
            }
        } else if !after_scripts && !has_build {
            entry.script = Some(Arc::new(Output {
                names: names.clone(),
                pairs: pairs.clone(),
                ..Output::default()
            }));
        }
    }
}

struct SettleState {
    dag: ScriptDag,
    ran: Ran,
    failed: HashSet<usize>,
    fatal_errors: Vec<String>,
    kept_queue: VecDeque<usize>,
}

fn settle_inner(workspace: &mut Workspace, targets: &Targets) -> Result<()> {
    attach_rustflag_cfgs(workspace, targets, false);
    let dag = ScriptDag::new(workspace);
    let kept_queue = dag.initial_ready(workspace);
    let mut state = SettleState {
        dag,
        ran: known(workspace),
        failed: HashSet::default(),
        fatal_errors: Vec::new(),
        kept_queue,
    };
    drain_kept(workspace, targets, &mut state, None);
    anyhow::ensure!(
        state.fatal_errors.is_empty(),
        "{}",
        state.fatal_errors.join("\n")
    );
    loop {
        let mut ready = state.dag.ready_build_units(workspace, &state.failed);
        if ready.is_empty() {
            attach_rustflag_cfgs(workspace, targets, true);
            return Ok(());
        }
        ready.sort_by_cached_key(|&unit| store::dir(workspace, unit));
        let mut locks = HashMap::default();
        for &unit in &ready {
            locks.insert(unit, store::lock(workspace, unit)?);
            if state.dag.links_rem[unit] == 0 {
                state.kept_queue.push_back(unit);
            }
        }
        drain_kept(workspace, targets, &mut state, Some(&mut locks));
        anyhow::ensure!(
            state.fatal_errors.is_empty(),
            "{}",
            state.fatal_errors.join("\n")
        );
        let mut wave: Vec<usize> = ready
            .into_iter()
            .filter(|unit| {
                workspace.graph.units[*unit].script.is_none() && !state.failed.contains(unit)
            })
            .collect();
        if wave.is_empty() {
            continue;
        }
        wave.sort_by_key(|&unit| {
            std::cmp::Reverse((
                state.dag.build_users[unit].len(),
                state.dag.links_users[unit].len(),
            ))
        });
        let outcome = run(
            workspace,
            targets,
            &wave,
            &state.dag,
            &mut locks,
            &mut state.ran,
        )?;
        drop(locks);
        finish_wave(workspace, &wave, outcome, &mut state);
        anyhow::ensure!(
            state.fatal_errors.is_empty(),
            "{}",
            state.fatal_errors.join("\n")
        );
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
        let is_script: Vec<bool> = units
            .iter()
            .map(|entry| workspace.packages[entry.package].manifest.build.is_some())
            .collect();
        let script_units: Vec<usize> = (0..n)
            .filter(|&u| is_script[u] && units[u].script.is_none())
            .collect();
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
                    && workspace.packages[units[dep].package]
                        .manifest
                        .links
                        .is_some()
                    && seen_links.insert(dep)
                {
                    links_users[dep].push(unit);
                    links_rem[unit] += 1;
                }
            }
        }
        Self {
            build_deps,
            build_users,
            links_users,
            build_rem,
            links_rem,
            script_units,
        }
    }

    fn initial_ready(&self, workspace: &Workspace) -> VecDeque<usize> {
        let mut ready = VecDeque::new();
        for &unit in &self.script_units {
            if workspace.graph.units[unit].script.is_none()
                && self.build_rem[unit] == 0
                && self.links_rem[unit] == 0
            {
                ready.push_back(unit);
            }
        }
        ready
    }

    fn ready_build_units(&self, workspace: &Workspace, failed: &HashSet<usize>) -> Vec<usize> {
        let pending: Vec<usize> = self
            .script_units
            .iter()
            .copied()
            .filter(|&u| workspace.graph.units[u].script.is_none() && !failed.contains(&u))
            .collect();
        let mut ready: HashSet<usize> = pending
            .iter()
            .copied()
            .filter(|&u| self.build_rem[u] == 0)
            .collect();
        loop {
            let blocked: HashSet<usize> = pending
                .iter()
                .filter(|d| !ready.contains(d))
                .flat_map(|&d| self.links_users[d].iter().copied())
                .collect();
            let before = ready.len();
            ready.retain(|u| !blocked.contains(u));
            if ready.len() == before {
                break;
            }
        }
        let ordered: Vec<usize> = pending
            .iter()
            .copied()
            .filter(|u| ready.contains(u))
            .collect();
        if ordered.is_empty() { pending } else { ordered }
    }

    fn finish_unit(
        &mut self,
        workspace: &Workspace,
        failed: &HashSet<usize>,
        unit: usize,
        kept_queue: &mut VecDeque<usize>,
    ) {
        let mut wake = |this: &Self, user: usize| {
            if this.build_rem[user] == 0
                && this.links_rem[user] == 0
                && workspace.graph.units[user].script.is_none()
                && !failed.contains(&user)
            {
                kept_queue.push_back(user);
            }
        };
        for &user in &self.build_users[unit] {
            self.build_rem[user] = self.build_rem[user].saturating_sub(1);
            wake(self, user);
        }
        for &user in &self.links_users[unit] {
            self.links_rem[user] = self.links_rem[user].saturating_sub(1);
            wake(self, user);
        }
    }
}

fn drain_kept(
    workspace: &mut Workspace,
    targets: &Targets,
    state: &mut SettleState,
    mut locks: Option<&mut HashMap<usize, store::Lock>>,
) {
    while let Some(unit) = state.kept_queue.pop_front() {
        if workspace.graph.units[unit].script.is_some()
            || state.failed.contains(&unit)
            || locks.as_deref().is_some_and(|l| !l.contains_key(&unit))
        {
            continue;
        }
        let unit_vars = vars(workspace, targets, unit, &state.ran);
        let Some(kept) = store::kept(workspace, unit, &unit_vars) else {
            continue;
        };
        match kept {
            Ok(output) if !output.errors.is_empty() => record_fatal(
                workspace,
                unit,
                Failure {
                    error: output.errors.join("\n"),
                    broken: None,
                    interrupted: false,
                    fatal: true,
                },
                state,
            ),
            Ok(output) => {
                let output = Arc::new(output);
                workspace.graph.units[unit].script = Some(output.clone());
                state.ran.insert(unit, output);
            }
            Err(error) => record_fatal(workspace, unit, Failure::from(error), state),
        }
        if let Some(locks) = locks.as_deref_mut() {
            locks.remove(&unit);
        }
        state
            .dag
            .finish_unit(workspace, &state.failed, unit, &mut state.kept_queue);
    }
}

fn record_fatal(workspace: &Workspace, unit: usize, failure: Failure, state: &mut SettleState) {
    give_up(workspace, unit, &failure, state);
    if failure.fatal {
        state.fatal_errors.push(format!(
            "build script of `{}` failed: {}",
            package_name(workspace, unit),
            failure.error
        ));
    }
}

fn finish_wave(
    workspace: &mut Workspace,
    wave: &[usize],
    outcome: Outcome,
    state: &mut SettleState,
) {
    let wave_set: HashSet<usize> = wave.iter().copied().collect();
    let mut errors = Vec::new();
    for (unit, result) in outcome {
        match result {
            Ok(output) => {
                workspace.graph.units[unit].script = Some(output.clone());
                state.ran.insert(unit, output);
                state
                    .dag
                    .finish_unit(workspace, &state.failed, unit, &mut state.kept_queue);
            }
            Err(failure) => errors.push((unit, failure)),
        }
    }
    for (unit, failure) in errors {
        let retried = failure
            .broken
            .is_some_and(|at| wave_set.contains(&at) && workspace.graph.units[at].script.is_some());
        if !retried {
            record_fatal(workspace, unit, failure, state);
            state
                .dag
                .finish_unit(workspace, &state.failed, unit, &mut state.kept_queue);
        }
    }
}

/// Reports that `unit`'s script did not build or run, clears any stale cache entry so a retry runs fresh, and builds its package with the script's cfgs open.
fn give_up(workspace: &Workspace, unit: usize, failure: &Failure, state: &mut SettleState) {
    report(workspace, unit, &failure.error, &mut state.failed);
    if !failure.interrupted
        && let Err(write) = store::clear_failure(workspace, unit)
    {
        eprintln!("cannot clear stale build script state: {write}");
    }
}

fn report(workspace: &Workspace, unit: usize, error: &str, failed: &mut HashSet<usize>) {
    eprintln!("build script of {}: {error}", package_name(workspace, unit));
    failed.insert(unit);
}

/// The variables the script of `unit` runs with, given the outputs that units already have.
fn vars(workspace: &Workspace, targets: &Targets, unit: usize, ran: &Ran) -> Vec<(String, String)> {
    script::script_vars(workspace, targets, unit, &metadata(workspace, unit, ran))
}

/// Why a script has no output, with the unit of the crate whose function did not build, when one did not: a script whose broken crate had its own script run since builds again, since that script's cfgs may fix it.
#[derive(Debug)]
struct Failure {
    error: String,
    broken: Option<usize>,
    interrupted: bool,
    fatal: bool,
}

impl From<String> for Failure {
    fn from(error: String) -> Failure {
        Failure {
            error,
            broken: None,
            interrupted: false,
            fatal: true,
        }
    }
}

/// The script outputs that units already have.
fn known(workspace: &Workspace) -> Ran {
    let units = workspace.graph.units.iter().enumerate();
    units
        .filter_map(|(unit, entry)| Some((unit, entry.script.clone()?)))
        .collect()
}

/// The `DEP_<links>_<key>` variables that the `links` dependencies of `unit` printed, from `ran` in this run.
fn metadata(workspace: &Workspace, unit: usize, ran: &Ran) -> Vec<(String, String)> {
    let deps = workspace.graph.units[unit]
        .deps
        .iter()
        .filter(|(_, _, kind)| *kind == Kind::Normal)
        .map(|(_, dep, _)| *dep);
    let linked = deps.filter_map(|dep| {
        Some((
            workspace.packages[workspace.graph.units[dep].package]
                .manifest
                .links
                .as_deref()?,
            ran.get(&dep)?,
        ))
    });
    linked
        .flat_map(|(links, output)| script::dep_vars(links, output))
        .collect()
}

/// The name of the package of `unit`.
fn package_name(workspace: &Workspace, unit: usize) -> &str {
    &workspace.packages[workspace.graph.units[unit].package]
        .locked
        .name
}

type Ran = rustc_hash::FxHashMap<usize, Arc<Output>>;
type Outcome = Vec<(usize, Result<Arc<Output>, Failure>)>;

/// How a script starts: its declared `lang_start` and `main` in children that compile each function on its first call, or its kept executable.
enum Start {
    Jit([usize; 2]),
    Executable(std::path::PathBuf),
}

/// Builds and runs the scripts of the units `units`, each from its kept executable directly before parsing/expanding when its `script.fast` manifest holds, else in children that compile each function on its first call under fast script profile and keep the executable for later runs.
fn run(
    workspace: &Workspace,
    targets: &Targets,
    units: &[usize],
    dag: &ScriptDag,
    locks: &mut HashMap<usize, store::Lock>,
    ran: &mut Ran,
) -> Result<Outcome> {
    let mut unit_dirs = HashMap::default();
    let mut fast_exec = HashMap::default();
    let mut slow_units = Vec::new();
    for &unit in units {
        let dir = store::dir(workspace, unit);
        if let Some(path) = binary::fast_kept(workspace, targets, unit, &dir) {
            fast_exec.insert(unit, path);
        } else {
            slow_units.push(unit);
        }
        unit_dirs.insert(unit, dir);
    }
    if slow_units.is_empty() {
        return fast::run_executables(workspace, targets, dag, units, &fast_exec, locks, ran);
    }

    let outer_cfg = crate::config();
    let outer_targets = Targets {
        target: targets.target.clone(),
        host: targets.host.clone(),
    };
    let host_targets = Targets {
        target: targets.host.clone(),
        host: targets.host.clone(),
    };
    binary::with_script_profile(|| {
        let mut compilation = crate::macros::load_eager(workspace, &host_targets, &slow_units)?;
        let outline = &compilation.outline;
        let scripts: Vec<(usize, usize)> = outline
            .roots
            .iter()
            .enumerate()
            .filter(|(_, root)| root.role == Role::Script)
            .map(|(krate, root)| (root.unit, krate))
            .collect();
        let binaries = Binaries::new(
            workspace,
            outline,
            &scripts,
            unit_dirs,
            Arc::clone(&outer_cfg),
            Targets {
                target: outer_targets.target.clone(),
                host: outer_targets.host.clone(),
            },
        );
        let kept: Vec<Kept> = scripts
            .iter()
            .map(|&(unit, _)| binaries.kept(workspace, unit))
            .collect();
        let compiled: Vec<(usize, usize)> = scripts
            .iter()
            .zip(&kept)
            .filter(|(_, kept)| !matches!(kept, Kept::Executable(_)))
            .map(|(script, _)| *script)
            .collect();
        if !compiled.is_empty() {
            let user_scripts = compiled
                .iter()
                .filter(|&&(unit, _)| {
                    !workspace.packages[workspace.graph.units[unit].package].sysroot
                })
                .count();
            if user_scripts > 0 {
                eprintln!(
                    "haste: running {user_scripts} build script{}...",
                    if user_scripts == 1 { "" } else { "s" }
                );
            }
        }
        let mains: Vec<Option<crate::names::ItemRef>> = compiled
            .iter()
            .map(|(_, krate)| outline.main(*krate))
            .collect();
        let owners = outline.roots.iter().map(|root| root.unit).collect();
        let (slot, cache) = if sysroot_scripts_settled(workspace) {
            crate::build::substrate::cache_with_toolchain_substrate(workspace)
        } else {
            (None, crate::sema::Cache::default())
        };
        let (entries, store) = image::entries(&mut compilation, (cache, &mains))?;
        let mut demand = Jit::new(store)?;
        let has_build_deps = compiled.iter().any(|&(unit, _)| {
            workspace.graph.units[unit]
                .deps
                .iter()
                .any(|&(_, _, kind)| kind == Kind::Build)
        });
        if has_build_deps {
            demand.set_optimize(true);
        } else {
            demand.set_fast(true);
        }
        demand.set_ahead(!has_build_deps);
        let mut entries_map: HashMap<usize, Result<Start, String>> = HashMap::default();
        for (&(unit, _), entry) in compiled.iter().zip(entries) {
            entries_map.insert(
                unit,
                entry.map(|pair| Start::Jit(pair.map(|root| demand.declare(root)))),
            );
        }
        let missing: HashSet<usize> = scripts
            .iter()
            .zip(&kept)
            .filter(|(_, kept)| matches!(kept, Kept::Missing))
            .map(|(&(unit, _), _)| unit)
            .collect();
        for (&(unit, _), kept) in scripts.iter().zip(kept) {
            if let Kept::Executable(path) = kept {
                entries_map.insert(unit, Ok(Start::Executable(path)));
            }
        }
        for (unit, path) in fast_exec {
            entries_map.insert(unit, Ok(Start::Executable(path)));
        }
        let mut driver = Runs {
            workspace,
            targets: &outer_targets,
            outer_cfg,
            owners,
            dag,
            locks: &mut *locks,
            entries: entries_map,
            queue: WaveQueue::new(dag, units),
            running: Vec::new(),
            ran: ran.clone(),
            outcome: Outcome::new(),
            missing,
            reran: HashSet::default(),
            executables: HashSet::default(),
            client_env: prepare_client_env(),
        };
        image::serve(
            &mut compilation,
            &crate::macros::load,
            &mut demand,
            &mut driver,
        )?;
        if let Some(slot) = slot {
            crate::build::substrate::save_jit_substrate(
                &compilation.outline,
                demand.store(),
                demand.image(),
                slot,
            );
        }
        let Runs { outcome, reran, .. } = driver;
        let rerun: Vec<(usize, usize)> = scripts
            .into_iter()
            .filter(|(unit, _)| {
                reran.contains(unit)
                    && outcome
                        .iter()
                        .any(|(at, result)| at == unit && result.is_ok())
            })
            .collect();
        if let Err(error) = binaries.keep_detached(workspace, &mut compilation, &rerun, locks) {
            eprintln!("cannot keep the executables of build scripts: {error}");
        }
        Ok(outcome)
    })
}

struct Runs<'w, 'l> {
    workspace: &'w Workspace,
    targets: &'w Targets,
    outer_cfg: Arc<crate::config::Config>,
    owners: Vec<usize>,
    dag: &'w ScriptDag,
    locks: &'l mut HashMap<usize, store::Lock>,
    entries: HashMap<usize, Result<Start, String>>,
    queue: WaveQueue,
    running: Vec<(usize, usize, Script)>,
    ran: Ran,
    outcome: Outcome,
    missing: HashSet<usize>,
    reran: HashSet<usize>,
    executables: HashSet<usize>,
    client_env: Option<ChildEnv>,
}

impl Runs<'_, '_> {
    fn next(&mut self) -> Option<(usize, Result<Start, String>)> {
        let unit = self.queue.pop_ready(self.running.is_empty())?;
        let entry = self.entries.remove(&unit)?;
        Some((unit, entry))
    }

    fn complete_unit(&mut self, unit: usize) {
        self.locks.remove(&unit);
        self.queue.complete(self.dag, unit);
    }
}

impl Driver for Runs<'_, '_> {
    fn start(&mut self, demand: &mut Jit) -> Result<()> {
        while self.running.len() < crate::jobs() {
            let Some((unit, entry)) = self.next() else {
                return Ok(());
            };
            let was_ran_before = crate::with_config(Arc::clone(&self.outer_cfg), || {
                binary::ran_before(self.workspace, unit)
            });
            if self.missing.contains(&unit) && was_ran_before {
                self.reran.insert(unit);
            }
            let script = entry.map_err(Failure::from).and_then(|start| {
                let s = crate::with_config(Arc::clone(&self.outer_cfg), || {
                    job(self.workspace, self.targets, unit, &start, &self.ran)
                })?;
                Ok((s, start))
            });
            match script {
                Ok((script, start)) => {
                    let env_patch = self.client_env.as_ref();
                    let id = demand.fork(true, |image, out| {
                        if let Some((unset, set)) = env_patch {
                            apply_client_env(unset, set);
                        }
                        match (&start, out) {
                            (Start::Jit(_), Some(out)) => image.enter_script(&script, out),
                            (Start::Executable(path), Some(out)) => {
                                binary::exec(path, &script, out)
                            }
                            (_, None) => 101,
                        }
                    })?;
                    if matches!(start, Start::Executable(_)) {
                        self.executables.insert(unit);
                    }
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
            (0, _) => String::from_utf8(ended.bytes)
                .map_err(|error| {
                    Failure::from(format!("printed output that is not UTF-8: {error}"))
                })
                .and_then(|text| {
                    crate::with_config(Arc::clone(&self.outer_cfg), || {
                        finished(self.workspace, unit, &script, &text)
                    })
                }),
            (_, Some((error, item))) => Err(Failure {
                error: format!("does not build: {error}"),
                broken: self.owners.get(item.krate as usize).copied(),
                interrupted: false,
                fatal: true,
            }),
            (status, None) => {
                if self.executables.contains(&unit) {
                    crate::with_config(Arc::clone(&self.outer_cfg), || {
                        binary::distrust(self.workspace, unit);
                    });
                }
                Err(Failure {
                    interrupted: crate::image::interrupted(status),
                    ..Failure::from(format!(
                        "exited with status {status}; its errors are in {}",
                        script.log.display()
                    ))
                })
            }
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
fn finished(
    workspace: &Workspace,
    unit: usize,
    job: &Script,
    text: &str,
) -> Result<Arc<Output>, Failure> {
    let output = Output::parse(text);
    if !output.errors.is_empty() {
        return Err(Failure {
            error: output.errors.join("\n"),
            broken: None,
            interrupted: false,
            fatal: true,
        });
    }
    if let Err(error) = store::keep(workspace, unit, &job.vars, text) {
        eprintln!(
            "cannot keep the output of {}'s build script: {error}",
            package_name(workspace, unit)
        );
    }
    Ok(Arc::new(output))
}

fn job(
    workspace: &Workspace,
    targets: &Targets,
    unit: usize,
    start: &Start,
    ran: &Ran,
) -> Result<Script, Failure> {
    let entry = match start {
        Start::Jit([start, main]) => (*start, *main),
        Start::Executable(_) => (0, 0),
    };
    let dir = store::dir(workspace, unit);
    let made = store::forget(workspace, unit).and_then(|()| std::fs::create_dir_all(dir.join(OUT)));
    made.map_err(|error| Failure::from(format!("cannot make {}: {error}", dir.display())))?;
    let vars = script::script_vars(workspace, targets, unit, &metadata(workspace, unit, ran));
    let package = &workspace.packages[workspace.graph.units[unit].package].manifest;
    Ok(Script {
        entry,
        vars,
        dir: package.dir.clone(),
        log: dir.join("stderr"),
    })
}
