use std::collections::{BTreeMap, BTreeSet, VecDeque};

use anyhow::{Result, bail};

use super::manifest::{Dep, Kind};
use super::{Graph, Package, Side, Unit};

/// One step of feature resolution on a unit.
enum Step {
    Feature(String),
    Dep(String),
}

struct Resolver<'a> {
    packages: &'a [Package],
    targets: &'a super::Targets,
    members: &'a [usize],
    unify_sides: bool,
    units: Vec<Unit>,
    by_key: BTreeMap<(usize, Side), usize>,
    enabled: Vec<BTreeSet<String>>,
    want: Vec<BTreeMap<String, Vec<String>>>,
    queue: VecDeque<(usize, Step)>,
}

fn implicit(package: &Package, key: &str) -> bool {
    let manifest = &package.manifest;
    let optional = manifest
        .deps
        .iter()
        .any(|dep| dep.optional && dep.key == key);
    let explicit = manifest
        .features
        .values()
        .flatten()
        .any(|value| value.strip_prefix("dep:") == Some(key));
    optional && !explicit
}

fn has_feature(package: &Package, name: &str) -> bool {
    package.manifest.features.contains_key(name) || implicit(package, name)
}

fn unknown_feature(packages: &[Package], members: &[usize], feature: &str) -> anyhow::Error {
    if members.len() == 1 {
        anyhow::anyhow!(
            "package `{}` does not have feature `{feature}`",
            packages[members[0]].manifest.name
        )
    } else {
        anyhow::anyhow!("none of the selected packages have feature `{feature}`")
    }
}

fn validate_features(packages: &[Package], members: &[usize], features: &[&str]) -> Result<()> {
    for &feature in features {
        if let Some((pkg_or_dep, sub)) = feature.split_once('/') {
            if let Some(&member) = members
                .iter()
                .find(|&&m| packages[m].manifest.name == pkg_or_dep)
            {
                if !has_feature(&packages[member], sub) {
                    bail!(
                        "package `{}` does not have feature `{sub}`",
                        packages[member].manifest.name
                    );
                }
                continue;
            }
            let targets: Vec<usize> = members
                .iter()
                .filter_map(|&m| packages[m].resolved.get(pkg_or_dep).copied())
                .collect();
            if targets.is_empty() {
                return Err(unknown_feature(packages, members, feature));
            }
            if !targets
                .iter()
                .any(|&target| has_feature(&packages[target], sub))
            {
                bail!(
                    "package `{}` does not have feature `{sub}`",
                    packages[targets[0]].manifest.name
                );
            }
        } else if !members.iter().any(|&m| has_feature(&packages[m], feature)) {
            return Err(unknown_feature(packages, members, feature));
        }
    }
    Ok(())
}

impl Resolver<'_> {
    fn sibling(&self, package: usize, side: Side) -> Option<usize> {
        let other = match side {
            Side::Target => Side::Host,
            Side::Host => Side::Target,
        };
        self.by_key.get(&(package, other)).copied()
    }

    fn unit(&mut self, package: usize, side: Side) -> usize {
        if let Some(unit) = self.by_key.get(&(package, side)) {
            return *unit;
        }
        let unit = self.units.len();
        self.units.push(Unit {
            package,
            side,
            features: BTreeSet::new(),
            deps: Vec::new(),
            script: None,
        });
        self.by_key.insert((package, side), unit);
        self.enabled.push(BTreeSet::new());
        self.want.push(BTreeMap::new());
        let required: Vec<String> = self
            .applicable(unit)
            .filter(|dep| !dep.optional)
            .map(|dep| dep.key.clone())
            .collect();
        self.queue
            .extend(required.into_iter().map(|key| (unit, Step::Dep(key))));
        if self.unify_sides
            && let Some(other) = self.sibling(package, side)
        {
            let existing: Vec<String> = self.units[other].features.iter().cloned().collect();
            self.queue
                .extend(existing.into_iter().map(|feat| (unit, Step::Feature(feat))));
        }
        unit
    }

    fn applicable(&self, unit: usize) -> impl Iterator<Item = &Dep> + '_ {
        let Unit { package, side, .. } = self.units[unit];
        let target = self.targets.of(side);
        let manifest = &self.packages[package].manifest;
        let dev = (crate::test_mode() || !crate::selection().example.is_empty())
            && self.members.contains(&package);
        let used = move |dep: &&Dep| match dep.kind {
            Kind::Normal => true,
            Kind::Build => manifest.build.is_some(),
            Kind::Dev => dev,
        };
        let host_target = &self.targets.host;
        let is_std = manifest.name == "std";
        manifest.deps.iter().filter(used).filter(move |dep| {
            dep.platform.as_deref().is_none_or(|key| {
                target.matches_key(key).unwrap_or(false)
                    || (is_std && host_target.matches_key(key).unwrap_or(false))
            })
        })
    }

    fn feature(&mut self, unit: usize, name: String) {
        let package = &self.packages[self.units[unit].package];
        let known = package.manifest.features.get(&name).cloned();
        if known.is_none() && !implicit(package, &name) {
            return;
        }
        if !self.units[unit].features.insert(name.clone()) {
            return;
        }
        if self.unify_sides {
            let Unit { package, side, .. } = self.units[unit];
            if let Some(other) = self.sibling(package, side)
                && !self.units[other].features.contains(&name)
            {
                self.queue.push_back((other, Step::Feature(name.clone())));
            }
        }
        match known {
            Some(values) => values.into_iter().for_each(|value| self.value(unit, value)),
            None => self.queue.push_back((unit, Step::Dep(name))),
        }
    }

    fn value(&mut self, unit: usize, value: String) {
        if let Some(key) = value.strip_prefix("dep:") {
            return self.queue.push_back((unit, Step::Dep(key.to_owned())));
        }
        let Some((key, feature)) = value.split_once('/') else {
            return self.queue.push_back((unit, Step::Feature(value)));
        };
        let (key, weak) = key
            .strip_suffix('?')
            .map_or((key, false), |key| (key, true));
        self.want[unit]
            .entry(key.to_owned())
            .or_default()
            .push(feature.to_owned());
        if weak {
            return self.flush(unit, key);
        }
        if implicit(&self.packages[self.units[unit].package], key) {
            self.queue.push_back((unit, Step::Feature(key.to_owned())));
        }
        self.queue.push_back((unit, Step::Dep(key.to_owned())));
        self.flush(unit, key);
    }

    fn flush(&mut self, unit: usize, key: &str) {
        if !self.enabled[unit].contains(key) {
            return;
        }
        let wanted = self.want[unit].get(key).cloned().unwrap_or_default();
        let children: Vec<usize> = self.units[unit]
            .deps
            .iter()
            .filter(|(dep_key, _, _)| dep_key == key)
            .map(|(_, child, _)| *child)
            .collect();
        for child in children {
            self.queue.extend(
                wanted
                    .iter()
                    .map(|feature| (child, Step::Feature(feature.clone()))),
            );
        }
    }

    fn dep(&mut self, unit: usize, key: String) {
        if !self.enabled[unit].insert(key.clone()) {
            return;
        }
        let side = self.units[unit].side;
        let package = self.units[unit].package;
        let matches: Vec<Dep> = self
            .applicable(unit)
            .filter(|dep| dep.key == key)
            .cloned()
            .collect();
        for dep in matches {
            let Some(target) = self.packages[package].resolved.get(&dep.key).copied() else {
                continue;
            };
            let host = side == Side::Host
                || dep.kind == Kind::Build
                || self.packages[target].manifest.proc_macro;
            let child = self.unit(target, if host { Side::Host } else { Side::Target });
            self.units[unit]
                .deps
                .push((dep.key.clone(), child, dep.kind));
            if dep.default_features {
                self.queue
                    .push_back((child, Step::Feature("default".to_owned())));
            }
            self.queue.extend(
                dep.features
                    .iter()
                    .map(|feature| (child, Step::Feature(feature.clone()))),
            );
        }
        self.flush(unit, &key);
    }

    fn run(&mut self) {
        while let Some((unit, step)) = self.queue.pop_front() {
            match step {
                Step::Feature(name) => self.feature(unit, name),
                Step::Dep(key) => self.dep(unit, key),
            }
        }
    }
}

/// Resolves the features of every unit reachable from `members`, using Cargo resolver 1 feature unification across target and host when configured or implied by edition/workspace, and resolver 2 isolation otherwise.
pub fn resolve(
    packages: &[Package],
    targets: &super::Targets,
    members: &[usize],
    features: &[&str],
    (default_features, all_features, workspace, top): (
        bool,
        bool,
        Option<&toml::Table>,
        &std::path::Path,
    ),
) -> Result<Graph> {
    validate_features(packages, members, features)?;
    let unify_sides = !super::manifest::uses_resolver_v2(workspace, top, packages, members);
    let mut resolver = Resolver {
        packages,
        targets,
        members,
        unify_sides,
        units: Vec::new(),
        by_key: BTreeMap::new(),
        enabled: Vec::new(),
        want: Vec::new(),
        queue: VecDeque::new(),
    };
    for member in members {
        let unit = resolver.unit(*member, Side::Target);
        if default_features {
            resolver
                .queue
                .push_back((unit, Step::Feature("default".to_owned())));
        }
        let package = &packages[*member];
        if all_features {
            resolver.queue.extend(
                package
                    .manifest
                    .features
                    .keys()
                    .cloned()
                    .map(|name| (unit, Step::Feature(name))),
            );
            resolver.queue.extend(
                package
                    .manifest
                    .deps
                    .iter()
                    .filter(|dep| implicit(package, &dep.key))
                    .map(|dep| (unit, Step::Feature(dep.key.clone()))),
            );
        }
        for &feature in features {
            if let Some((pkg_or_dep, sub)) = feature.split_once('/') {
                if package.manifest.name == pkg_or_dep {
                    resolver
                        .queue
                        .push_back((unit, Step::Feature(sub.to_owned())));
                } else if package
                    .manifest
                    .deps
                    .iter()
                    .any(|dep| dep.key == pkg_or_dep)
                {
                    resolver.value(unit, feature.to_owned());
                }
            } else {
                resolver
                    .queue
                    .push_back((unit, Step::Feature(feature.to_owned())));
            }
        }
    }
    resolver.run();
    Ok(Graph {
        units: resolver.units,
        members: members.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use crate::graph::cfg::Target;
    use crate::graph::lock::Locked;
    use crate::graph::manifest::Manifest;

    fn pkg(name: &str, edition: &str, resolver: Option<u8>, manifest: Manifest) -> Package {
        Package {
            locked: Locked {
                name: name.to_owned(),
                version: semver::Version::new(1, 0, 0),
                source: None,
                checksum: None,
                dependencies: Vec::new(),
            },
            manifest: Manifest {
                name: name.to_owned(),
                edition: edition.to_owned(),
                resolver,
                ..manifest
            },
            resolved: BTreeMap::new(),
            sysroot: false,
        }
    }

    fn dep(key: &str, kind: Kind, default_features: bool, features: &[&str]) -> Dep {
        Dep {
            key: key.to_owned(),
            package: key.to_owned(),
            req: None,
            path: None,
            optional: false,
            default_features,
            features: features.iter().map(|s| (*s).to_owned()).collect(),
            kind,
            platform: None,
        }
    }

    fn clap_packages(app_edition: &str, app_resolver: Option<u8>) -> Vec<Package> {
        let mut builder = pkg("clap_builder", "2021", None, Manifest::default());
        builder
            .manifest
            .features
            .insert("default".to_owned(), Vec::new());
        builder
            .manifest
            .features
            .insert("cargo".to_owned(), Vec::new());

        let mut clap = pkg("clap", "2021", None, Manifest::default());
        clap.manifest
            .features
            .insert("default".to_owned(), Vec::new());
        clap.manifest
            .features
            .insert("cargo".to_owned(), vec!["clap_builder/cargo".to_owned()]);
        clap.manifest
            .deps
            .push(dep("clap_builder", Kind::Normal, true, &[]));
        clap.resolved.insert("clap_builder".to_owned(), 2);

        let mut app = pkg(
            "app",
            app_edition,
            app_resolver,
            Manifest {
                build: Some(PathBuf::from("build.rs")),
                ..Manifest::default()
            },
        );
        app.manifest
            .deps
            .push(dep("clap", Kind::Normal, false, &["cargo"]));
        app.manifest.deps.push(dep("clap", Kind::Build, true, &[]));
        app.resolved.insert("clap".to_owned(), 1);

        vec![app, clap, builder]
    }

    fn unit_features<'a>(graph: &'a Graph, package: usize, side: Side) -> &'a BTreeSet<String> {
        &graph
            .units
            .iter()
            .find(|u| u.package == package && u.side == side)
            .expect("unit exists")
            .features
    }

    fn test_targets() -> super::super::Targets {
        let host = Target::of(&crate::host_target());
        super::super::Targets {
            target: host.clone(),
            host,
        }
    }

    #[test]
    fn resolver_unifies_host_and_target_in_v1_and_isolates_in_v2() {
        let targets = test_targets();
        let root_top = std::path::Path::new("");
        let virtual_top = std::path::Path::new("/virtual-workspace");

        let pkgs_2018 = clap_packages("2018", None);
        let g1 = resolve(
            &pkgs_2018,
            &targets,
            &[0],
            &[],
            (true, false, None, root_top),
        )
        .expect("resolve edition 2018");
        assert!(unit_features(&g1, 1, Side::Host).contains("cargo"));
        assert!(unit_features(&g1, 2, Side::Host).contains("cargo"));
        assert!(unit_features(&g1, 1, Side::Target).contains("default"));

        let pkgs_2021 = clap_packages("2021", None);
        let g2 = resolve(
            &pkgs_2021,
            &targets,
            &[0],
            &[],
            (true, false, None, root_top),
        )
        .expect("resolve edition 2021");
        assert!(unit_features(&g2, 1, Side::Target).contains("cargo"));
        assert!(!unit_features(&g2, 1, Side::Host).contains("cargo"));
        assert!(!unit_features(&g2, 2, Side::Host).contains("cargo"));

        let ws_no_resolver = toml::Table::new();
        let g_ws_virtual = resolve(
            &pkgs_2021,
            &targets,
            &[0],
            &[],
            (true, false, Some(&ws_no_resolver), virtual_top),
        )
        .expect("resolve virtual workspace");
        assert!(unit_features(&g_ws_virtual, 1, Side::Host).contains("cargo"));

        let g_ws_root_2021 = resolve(
            &pkgs_2021,
            &targets,
            &[0],
            &[],
            (true, false, Some(&ws_no_resolver), root_top),
        )
        .expect("resolve non-virtual edition 2021 workspace");
        assert!(!unit_features(&g_ws_root_2021, 1, Side::Host).contains("cargo"));

        let mut ws_v2 = toml::Table::new();
        ws_v2.insert("resolver".to_owned(), toml::Value::String("2".to_owned()));
        let g_ws2 = resolve(
            &pkgs_2021,
            &targets,
            &[0],
            &[],
            (true, false, Some(&ws_v2), virtual_top),
        )
        .expect("resolve v2 workspace");
        assert!(!unit_features(&g_ws2, 1, Side::Host).contains("cargo"));

        let mut ws_v3 = toml::Table::new();
        ws_v3.insert("resolver".to_owned(), toml::Value::String("3".to_owned()));
        let g_ws3 = resolve(
            &pkgs_2021,
            &targets,
            &[0],
            &[],
            (true, false, Some(&ws_v3), virtual_top),
        )
        .expect("resolve v3 workspace");
        assert!(!unit_features(&g_ws3, 1, Side::Host).contains("cargo"));

        let pkgs_explicit_v1 = clap_packages("2021", Some(1));
        let g_pkg1 = resolve(
            &pkgs_explicit_v1,
            &targets,
            &[0],
            &[],
            (true, false, None, root_top),
        )
        .expect("resolve explicit resolver 1");
        assert!(unit_features(&g_pkg1, 2, Side::Host).contains("cargo"));
    }
}
