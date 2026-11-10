use std::collections::{BTreeMap, BTreeSet, VecDeque};

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
    units: Vec<Unit>,
    by_key: BTreeMap<(usize, Side), usize>,
    enabled: Vec<BTreeSet<String>>,
    want: Vec<BTreeMap<String, Vec<String>>>,
    queue: VecDeque<(usize, Step)>,
}

fn implicit(package: &Package, key: &str) -> bool {
    let manifest = &package.manifest;
    let optional = manifest.deps.iter().any(|dep| dep.optional && dep.key == key);
    let explicit = manifest.features.values().flatten().any(|value| value.strip_prefix("dep:") == Some(key));
    optional && !explicit
}

impl Resolver<'_> {
    fn unit(&mut self, package: usize, side: Side) -> usize {
        if let Some(unit) = self.by_key.get(&(package, side)) {
            return *unit;
        }
        let unit = self.units.len();
        self.units.push(Unit { package, side, features: BTreeSet::new(), deps: Vec::new(), script: None });
        self.by_key.insert((package, side), unit);
        self.enabled.push(BTreeSet::new());
        self.want.push(BTreeMap::new());
        let required: Vec<String> = self.applicable(unit).filter(|dep| !dep.optional).map(|dep| dep.key.clone()).collect();
        self.queue.extend(required.into_iter().map(|key| (unit, Step::Dep(key))));
        unit
    }

    fn applicable(&self, unit: usize) -> impl Iterator<Item = &Dep> + '_ {
        let Unit { package, side, .. } = self.units[unit];
        let target = self.targets.of(side);
        let manifest = &self.packages[package].manifest;
        let used = move |dep: &&Dep| match dep.kind {
            Kind::Normal => true,
            Kind::Build => manifest.build.is_some(),
            Kind::Dev => false,
        };
        manifest.deps.iter().filter(used).filter(move |dep| dep.platform.as_deref().is_none_or(|key| target.matches_key(key).unwrap_or(false)))
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
        let (key, weak) = key.strip_suffix('?').map_or((key, false), |key| (key, true));
        self.want[unit].entry(key.to_owned()).or_default().push(feature.to_owned());
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
        let children: Vec<usize> = self.units[unit].deps.iter().filter(|(dep_key, _, _)| dep_key == key).map(|(_, child, _)| *child).collect();
        for child in children {
            self.queue.extend(wanted.iter().map(|feature| (child, Step::Feature(feature.clone()))));
        }
    }

    fn dep(&mut self, unit: usize, key: String) {
        if !self.enabled[unit].insert(key.clone()) {
            return;
        }
        let side = self.units[unit].side;
        let package = self.units[unit].package;
        let matches: Vec<Dep> = self.applicable(unit).filter(|dep| dep.key == key).cloned().collect();
        for dep in matches {
            let Some(target) = self.packages[package].resolved.get(&dep.key).copied() else { continue };
            let host = side == Side::Host || dep.kind == Kind::Build || self.packages[target].manifest.proc_macro;
            let child = self.unit(target, if host { Side::Host } else { Side::Target });
            self.units[unit].deps.push((dep.key.clone(), child, dep.kind));
            if dep.default_features {
                self.queue.push_back((child, Step::Feature("default".to_owned())));
            }
            self.queue.extend(dep.features.iter().map(|feature| (child, Step::Feature(feature.clone()))));
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

/// Resolves the features of every unit reachable from `members`, with their default features, as Cargo's resolver 2 does for `normal` and `build` edges.
pub fn resolve(packages: &[Package], targets: &super::Targets, members: &[usize], features: &[&str]) -> Graph {
    let mut resolver = Resolver {
        packages,
        targets,
        units: Vec::new(),
        by_key: BTreeMap::new(),
        enabled: Vec::new(),
        want: Vec::new(),
        queue: VecDeque::new(),
    };
    for member in members {
        let unit = resolver.unit(*member, Side::Target);
        resolver.queue.push_back((unit, Step::Feature("default".to_owned())));
        resolver.queue.extend(features.iter().map(|feature| (unit, Step::Feature((*feature).to_owned()))));
    }
    resolver.run();
    Graph { units: resolver.units, members: members.to_vec() }
}
