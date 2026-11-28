//! Direct parallel execution and DAG queue for already-cached build script executables.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::graph::{Targets, Workspace, store};
use crate::image::Script;

use super::{Failure, Outcome, Ran, ScriptDag, Start, binary};

pub(super) struct WaveQueue {
    ready: VecDeque<usize>,
    waiting: Vec<usize>,
    wave_links_rem: HashMap<usize, usize>,
    active_build_rem: HashMap<usize, usize>,
}

impl WaveQueue {
    pub(super) fn new(dag: &ScriptDag, units: &[usize]) -> Self {
        let unit_set: HashSet<usize> = units.iter().copied().collect();
        let mut ready = VecDeque::new();
        let mut waiting = Vec::new();
        let mut wave_links_rem = HashMap::default();
        let mut active_build_rem = HashMap::default();
        for &unit in units {
            let lrem = dag.links_rem[unit];
            let brem = dag.build_deps[unit]
                .iter()
                .filter(|dep| unit_set.contains(dep))
                .count();
            wave_links_rem.insert(unit, lrem);
            active_build_rem.insert(unit, brem);
            if lrem == 0 && brem == 0 {
                ready.push_back(unit);
            } else {
                waiting.push(unit);
            }
        }
        Self {
            ready,
            waiting,
            wave_links_rem,
            active_build_rem,
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        !self.ready.is_empty() || !self.waiting.is_empty()
    }

    pub(super) fn pop_ready(&mut self, running_empty: bool) -> Option<usize> {
        self.ready.pop_front().or_else(|| {
            running_empty
                .then(|| (!self.waiting.is_empty()).then(|| self.waiting.remove(0)))
                .flatten()
        })
    }

    pub(super) fn complete(&mut self, dag: &ScriptDag, unit: usize) {
        for &user in &dag.links_users[unit] {
            if let Some(rem) = self.wave_links_rem.get_mut(&user) {
                *rem = rem.saturating_sub(1);
                self.try_wake(user);
            }
        }
        for &user in &dag.build_users[unit] {
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

pub(super) fn run_executables(
    workspace: &Workspace,
    targets: &Targets,
    dag: &ScriptDag,
    units: &[usize],
    executables: &HashMap<usize, PathBuf>,
    locks: &mut HashMap<usize, store::Lock>,
    ran: &mut Ran,
) -> Result<Outcome> {
    let client_env = crate::client_env();
    let mut queue = WaveQueue::new(dag, units);
    let mut outcome = Outcome::new();
    let max_jobs = crate::jobs().max(1);

    while queue.has_pending() {
        let mut batch: Vec<(usize, PathBuf, Script)> = Vec::new();
        while batch.len() < max_jobs {
            let Some(unit) = queue.pop_ready(batch.is_empty()) else {
                break;
            };
            let Some(exe) = executables.get(&unit) else {
                locks.remove(&unit);
                queue.complete(dag, unit);
                outcome.push((
                    unit,
                    Err(Failure::from("missing script executable".to_owned())),
                ));
                continue;
            };
            let start = Start::Executable(exe.clone());
            match super::job(workspace, targets, unit, &start, ran) {
                Ok(script) => batch.push((unit, exe.clone(), script)),
                Err(failure) => {
                    locks.remove(&unit);
                    queue.complete(dag, unit);
                    outcome.push((unit, Err(failure)));
                }
            }
        }
        if batch.is_empty() {
            continue;
        }
        let client_ref = client_env.as_deref();
        let results: Vec<(usize, Script, Result<(i32, Vec<u8>), String>)> = if batch.len() == 1 {
            batch
                .into_iter()
                .map(|(unit, exe, script)| {
                    let res = binary::exec_capture(&exe, &script, client_ref);
                    (unit, script, res)
                })
                .collect()
        } else {
            crate::sync::scope(|scope| {
                let handles: Vec<_> = batch
                    .into_iter()
                    .map(|(unit, exe, script)| {
                        scope.spawn(move || {
                            let res = binary::exec_capture(&exe, &script, client_ref);
                            (unit, script, res)
                        })
                    })
                    .collect();
                let mut out = Vec::with_capacity(handles.len());
                for handle in handles {
                    match handle.join() {
                        Ok(item) => out.push(item),
                        Err(payload) => std::panic::resume_unwind(payload),
                    }
                }
                out
            })
        };

        for (unit, script, exec_result) in results {
            let res = match exec_result {
                Ok((0, stdout)) => String::from_utf8(stdout)
                    .map_err(|error| {
                        Failure::from(format!("printed output that is not UTF-8: {error}"))
                    })
                    .and_then(|text| super::finished(workspace, unit, &script, &text)),
                Ok((status, _)) => {
                    binary::distrust(workspace, unit);
                    Err(Failure {
                        interrupted: true,
                        ..Failure::from(format!(
                            "exited with status {status}; its errors are in {}",
                            script.log.display()
                        ))
                    })
                }
                Err(error) => {
                    binary::distrust(workspace, unit);
                    Err(Failure {
                        interrupted: true,
                        ..Failure::from(error)
                    })
                }
            };
            if let Ok(output) = &res {
                ran.insert(unit, Arc::clone(output));
            }
            locks.remove(&unit);
            queue.complete(dag, unit);
            outcome.push((unit, res));
        }
    }
    Ok(outcome)
}
