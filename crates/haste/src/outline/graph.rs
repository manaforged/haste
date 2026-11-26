use rustc_hash::FxHashMap as HashMap;
use std::path::{Path, PathBuf};

use crate::graph::manifest::Kind;
use crate::graph::{Targets, Workspace};

use super::item::Vis;
use super::tree::{Crate, Decl, Env, Files, Module, PROC_MACRO_ROOT, State};
use super::{Outline, Role, Root};

/// The library and member binary targets of every unit; with build scripts `scripts`, those scripts and only the libraries they build from.
pub fn roots(workspace: &Workspace, scripts: &[usize]) -> Vec<Root> {
    let mut roots = Vec::new();
    let needed = (!scripts.is_empty()).then(|| script_closure(workspace, scripts));
    for (unit, entry) in workspace.graph.units.iter().enumerate() {
        let manifest = &workspace.packages[entry.package].manifest;
        if let Some(lib) = manifest
            .lib
            .as_ref()
            .filter(|_| needed.as_ref().is_none_or(|needed| needed.contains(&unit)))
        {
            let name = manifest
                .lib_name
                .as_ref()
                .unwrap_or(&manifest.name)
                .replace('-', "_");
            let supplied = workspace.packages[entry.package].sysroot && name == "proc_macro";
            let path = if supplied {
                PathBuf::from(PROC_MACRO_ROOT)
            } else {
                lib.clone()
            };
            roots.push(Root {
                unit,
                name,
                path,
                role: Role::Lib,
                edition: manifest.edition.clone(),
                proc_macro: manifest.proc_macro,
            });
        }
        if needed.is_none() && workspace.graph.members.contains(&entry.package) {
            let bins = manifest.bins.iter().map(|(name, path)| Root {
                unit,
                name: name.clone(),
                path: path.clone(),
                role: Role::Bin,
                edition: manifest.edition.clone(),
                proc_macro: false,
            });
            roots.extend(bins);
            let sel = crate::selection();
            if sel.all_targets || sel.example.is_some() {
                let examples = manifest
                    .examples
                    .iter()
                    .filter(|(name, _)| sel.example.as_deref().is_none_or(|wanted| name == wanted))
                    .map(|(name, path)| Root {
                        unit,
                        name: name.clone(),
                        path: path.clone(),
                        role: Role::Example,
                        edition: manifest.edition.clone(),
                        proc_macro: false,
                    });
                roots.extend(examples);
            }
            if sel.all_targets {
                let benches = manifest.benches.iter().map(|(name, path)| Root {
                    unit,
                    name: name.clone(),
                    path: path.clone(),
                    role: Role::Bench,
                    edition: manifest.edition.clone(),
                    proc_macro: false,
                });
                roots.extend(benches);
            }
            if crate::test_mode()
                && !sel.lib
                && !sel.bins
                && sel.bin.is_none()
                && sel.example.is_none()
            {
                let tests = manifest
                    .tests
                    .iter()
                    .filter(|(name, _)| sel.test.as_deref().is_none_or(|want| name == want))
                    .map(|(name, path)| Root {
                        unit,
                        name: name.clone(),
                        path: path.clone(),
                        role: Role::Test,
                        edition: manifest.edition.clone(),
                        proc_macro: false,
                    });
                roots.extend(tests);
            }
        }
        if let Some(path) = manifest.build.as_ref().filter(|_| scripts.contains(&unit)) {
            roots.push(Root {
                unit,
                name: "build_script_build".to_owned(),
                path: path.clone(),
                role: Role::Script,
                edition: manifest.edition.clone(),
                proc_macro: false,
            });
        }
    }
    roots
}

/// The units that the build dependencies of `scripts` reach through normal dependencies.
pub(crate) fn build_closure(
    workspace: &Workspace,
    scripts: &[usize],
) -> rustc_hash::FxHashSet<usize> {
    let units = &workspace.graph.units;
    let builds = scripts.iter().flat_map(|&unit| {
        units[unit]
            .deps
            .iter()
            .filter(|(_, _, kind)| *kind == Kind::Build)
            .map(|(_, dep, _)| *dep)
    });
    normal_closure(units, builds)
}

/// The units whose libraries the build scripts of `scripts` build from: the sysroot and every library their build dependencies reach.
pub(crate) fn script_closure(
    workspace: &Workspace,
    scripts: &[usize],
) -> rustc_hash::FxHashSet<usize> {
    let units = &workspace.graph.units;
    let sysroot = (0..units.len()).filter(|&unit| workspace.packages[units[unit].package].sysroot);
    let builds = scripts.iter().flat_map(|&unit| {
        units[unit]
            .deps
            .iter()
            .filter(|(_, _, kind)| *kind == Kind::Build)
            .map(|(_, dep, _)| *dep)
    });
    normal_closure(units, sysroot.chain(builds))
}

fn normal_closure(
    units: &[crate::graph::Unit],
    roots: impl IntoIterator<Item = usize>,
) -> rustc_hash::FxHashSet<usize> {
    let mut left: Vec<usize> = roots.into_iter().collect();
    let mut needed = rustc_hash::FxHashSet::default();
    while let Some(unit) = left.pop() {
        if needed.insert(unit) {
            left.extend(
                units[unit]
                    .deps
                    .iter()
                    .filter(|(_, _, kind)| *kind == Kind::Normal)
                    .map(|(_, dep, _)| *dep),
            );
        }
    }
    needed
}

const IMPLICIT: [&str; 4] = ["std", "core", "alloc", "proc_macro"];

fn implicit(workspace: &Workspace, roots: &[Root]) -> Vec<(String, usize)> {
    let named = |name: &str| {
        let found = roots.iter().position(|root| {
            root.role == Role::Lib
                && workspace.packages[workspace.graph.units[root.unit].package].sysroot
                && root.name == name
        });
        found.map(|at| (name.to_owned(), at))
    };
    IMPLICIT.iter().filter_map(|name| named(name)).collect()
}

fn dependencies(workspace: &Workspace, roots: &[Root]) -> Vec<Vec<(String, usize)>> {
    let libs: HashMap<usize, usize> = roots
        .iter()
        .enumerate()
        .filter(|(_, root)| root.role == Role::Lib)
        .map(|(index, root)| (root.unit, index))
        .collect();
    let implicit = implicit(workspace, roots);
    roots
        .iter()
        .map(|root| {
            let unit = &workspace.graph.units[root.unit];
            let package = &workspace.packages[unit.package];
            let include_dev = (crate::test_mode()
                && workspace.graph.members.contains(&unit.package))
                || matches!(root.role, Role::Example | Role::Test | Role::Bench);
            let normal = unit.deps.iter().filter(|(_, _, kind)| {
                if root.role == Role::Script {
                    *kind == Kind::Build
                } else {
                    *kind == Kind::Normal || (include_dev && *kind == Kind::Dev)
                }
            });
            let extern_name = |key: &str, dep: usize, lib: usize| {
                let renamed = workspace.packages[workspace.graph.units[dep].package]
                    .manifest
                    .name
                    != key;
                if renamed {
                    key.replace('-', "_")
                } else {
                    roots[lib].name.clone()
                }
            };
            let mut deps: Vec<(String, usize)> = normal
                .filter_map(|(key, dep, _)| {
                    libs.get(dep).map(|&lib| (extern_name(key, *dep, lib), lib))
                })
                .collect();
            if matches!(
                root.role,
                Role::Bin | Role::Example | Role::Test | Role::Bench
            ) && let Some(&lib) = libs.get(&root.unit)
            {
                deps.push((roots[lib].name.clone(), lib));
            }
            if !package.sysroot || root.role == Role::Script {
                deps.extend(implicit.iter().cloned());
            }
            deps
        })
        .collect()
}

/// Creates the outline of every root, with the build scripts of the units `scripts`, with only the root modules declared; `load` reads modules as lookups need them.
pub fn new(workspace: &Workspace, targets: &Targets, scripts: &[usize]) -> Outline {
    let mut roots = roots(workspace, scripts);
    let mut deps = dependencies(workspace, &roots);
    let mut is_test_dep = vec![false; roots.len()];
    if crate::test_mode() {
        let adj: Vec<Vec<usize>> = deps
            .iter()
            .map(|list| list.iter().map(|&(_, dep)| dep).collect())
            .collect();
        let test_roots = roots
            .iter()
            .enumerate()
            .filter(|(_, root)| root.role == Role::Test)
            .map(|(at, _)| at);
        dependents_of(&adj, &mut is_test_dep, test_roots);
        let extra: Vec<(Root, Vec<(String, usize)>)> = roots
            .iter()
            .enumerate()
            .filter(|&(at, root)| {
                is_test_dep[at]
                    && root.role == Role::Lib
                    && workspace
                        .graph
                        .members
                        .contains(&workspace.graph.units[root.unit].package)
            })
            .map(|(at, root)| {
                (
                    Root {
                        unit: root.unit,
                        name: root.name.clone(),
                        path: root.path.clone(),
                        role: Role::Test,
                        edition: root.edition.clone(),
                        proc_macro: root.proc_macro,
                    },
                    deps[at].clone(),
                )
            })
            .collect();
        for (root, root_deps) in extra {
            roots.push(root);
            deps.push(root_deps);
            is_test_dep.push(false);
        }
    }
    let crates = roots
        .iter()
        .enumerate()
        .map(|(at, root)| {
            let package = workspace.graph.units[root.unit].package;
            let is_member = workspace.graph.members.contains(&package);
            let test = crate::test_mode()
                && is_member
                && (matches!(root.role, Role::Test | Role::Bench)
                    || (matches!(root.role, Role::Lib | Role::Bin) && !is_test_dep[at]));
            root_crate(workspace, targets, root, test)
        })
        .collect();
    let pins = pins(workspace, &roots);
    Outline {
        roots,
        crates,
        deps,
        cache: Files::default(),
        procs: HashMap::default(),
        pins,
    }
}

fn pins(workspace: &Workspace, roots: &[Root]) -> Vec<Option<u64>> {
    use std::hash::BuildHasher as _;
    let units = &workspace.graph.units;
    let mut memo = vec![None; units.len()];
    let sysroot: Vec<u64> = (0..units.len())
        .filter(|&u| workspace.packages[units[u].package].sysroot)
        .map(|u| unit_pin(workspace, u, &[], &mut memo))
        .collect();
    let base = [rustc_hash::FxBuildHasher.hash_one(&sysroot)];
    roots
        .iter()
        .map(|root| {
            workspace.packages[units[root.unit].package]
                .locked
                .source
                .is_some()
                .then(|| {
                    rustc_hash::FxBuildHasher
                        .hash_one((unit_pin(workspace, root.unit, &base, &mut memo), &root.path))
                })
        })
        .collect()
}

fn unit_pin(workspace: &Workspace, unit: usize, extra: &[u64], memo: &mut [Option<u64>]) -> u64 {
    use std::hash::BuildHasher as _;
    if let Some(found) = memo[unit] {
        return found;
    }
    memo[unit] = Some(0);
    let entry = &workspace.graph.units[unit];
    let deps: Vec<(&str, u64)> = entry
        .deps
        .iter()
        .filter(|(_, _, kind)| *kind == Kind::Normal)
        .map(|(key, dep, _)| (key.as_str(), unit_pin(workspace, *dep, extra, memo)))
        .collect();
    let locked = &workspace.packages[entry.package].locked;
    let cfgs = entry
        .script
        .as_deref()
        .map(crate::graph::script::Output::key_parts);
    let hash = rustc_hash::FxBuildHasher.hash_one((
        &locked.name,
        locked.version.to_string(),
        &locked.source,
        &entry.features,
        entry.side == crate::graph::Side::Host,
        cfgs,
        extra,
        deps,
    ));
    memo[unit] = Some(hash);
    hash
}

fn target_tmpdir(workspace: &Workspace, root: &Root) -> PathBuf {
    use std::hash::BuildHasher as _;
    let unit = &workspace.graph.units[root.unit];
    let manifest_dir = &workspace.packages[unit.package].manifest.dir;
    let scope = rustc_hash::FxBuildHasher.hash_one((
        manifest_dir,
        root.unit,
        unit.package,
        &root.name,
        &root.path,
    ));
    std::env::temp_dir()
        .join("haste-target-tmp")
        .join(format!("{}-{scope:016x}", root.name))
}

fn root_crate(workspace: &Workspace, targets: &Targets, root: &Root, test: bool) -> Crate {
    let unit = &workspace.graph.units[root.unit];
    let script = root.role == Role::Script;
    let known = script
        || unit.script.is_some()
        || workspace.packages[unit.package].manifest.build.is_none();
    let side = if script {
        crate::graph::Side::Host
    } else {
        unit.side
    };
    let mut vars = if script {
        crate::graph::script::package_vars(workspace, root.unit)
    } else {
        crate::graph::script::crate_vars(workspace, root.unit)
    };
    let mut errors = Vec::new();
    if root.role == Role::Test {
        let tmp = target_tmpdir(workspace, root);
        if let Err(err) = std::fs::create_dir_all(&tmp) {
            errors.push(format!(
                "creating CARGO_TARGET_TMPDIR `{}`: {err}",
                tmp.display()
            ));
        }
        vars.push(("CARGO_TARGET_TMPDIR".to_owned(), tmp.display().to_string()));
    }
    let env = Env {
        target: targets.of(side).clone(),
        features: unit.features.clone(),
        script: unit.script.clone().filter(|_| !script),
        known,
        vars,
        release: workspace.packages[unit.package].sysroot,
        fixed: !workspace.packages[unit.package].local(),
        test,
        env_reads: Default::default(),
    };
    Crate {
        env,
        modules: vec![root_module(&root.path)],
        errors,
        ..Crate::default()
    }
}

pub(super) fn root_module(path: &Path) -> Module {
    let dir = path.parent().map(PathBuf::from).unwrap_or_default();
    let decl = Decl::File {
        candidates: vec![path.to_owned()],
        by_attr: true,
    };
    Module {
        file: usize::MAX,
        item: None,
        parent: None,
        name: String::new(),
        dir,
        parts: Vec::new(),
        vis: Vis::Pub,
        macro_use: false,
        state: State::Declared(decl),
        textual: None,
    }
}

/// The crates `from` and the crates that depend on one of them, by the direct `users` of each crate, that `seen` does not hold yet; marks them in `seen`.
pub fn dependents_of(
    users: &[Vec<usize>],
    seen: &mut [bool],
    from: impl IntoIterator<Item = usize>,
) -> Vec<usize> {
    let mut found: Vec<usize> = from
        .into_iter()
        .filter(|&krate| !std::mem::replace(&mut seen[krate], true))
        .collect();
    let mut at = 0;
    while let Some(&next) = found.get(at) {
        for &user in &users[next] {
            if !std::mem::replace(&mut seen[user], true) {
                found.push(user);
            }
        }
        at += 1;
    }
    found
}
