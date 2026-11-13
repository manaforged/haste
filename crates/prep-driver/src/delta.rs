use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::files::{self, DELTA, FIELD, LIBRARY, UNITS};
use crate::link::{self, Against};
use crate::record::MANIFEST;

/// Links the units under `root` rebuilt since the library in `base` was linked into a second library in `dir`, beside links to that library, its manifest and its unit list, and writes the second library's manifest, which replaces the records of those units and of the units that are gone.
pub fn link(root: &Path, dir: &Path, base: &Path) -> Result<()> {
    let listed = listed(&base.join(UNITS))?;
    for file in [LIBRARY.to_string(), files::manifest(LIBRARY), UNITS.to_string()] {
        fs::hard_link(base.join(&file), dir.join(&file))?;
    }
    let all = link::units(root)?;
    let mut rebuilt = Vec::new();
    for unit in &all {
        if listed.get(name(unit)) != Some(&stamp(unit)?) {
            rebuilt.push(unit.clone());
        }
    }
    let mut replaces: Vec<String> = rebuilt.iter().map(|unit| name(unit).to_string()).collect();
    replaces.extend(listed.into_keys().filter(|unit| !all.iter().any(|dir| name(dir) == unit)));
    if replaces.is_empty() {
        return Ok(());
    }
    let inputs = link::inputs(&rebuilt, &all)?;
    let out = dir.join(DELTA);
    link::link(&inputs, &out, Against::Base(dir))?;
    let counts = link::merge(&inputs, &out, (&dir.join(files::manifest(DELTA)), &replaces))?;
    link::report(&inputs, &out, counts)
}

/// Removes from `dir` what a link may have written there, so a link of every unit starts from nothing.
pub fn clear(dir: &Path) -> Result<()> {
    for file in [LIBRARY.to_string(), files::manifest(LIBRARY), UNITS.to_string(), DELTA.to_string(), files::manifest(DELTA)] {
        let path = dir.join(file);
        if path.symlink_metadata().is_ok() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Writes the name of each unit in `dirs` with the stamp of its manifest to `out`, one unit per line.
pub fn write_units(dirs: &[PathBuf], out: &Path) -> Result<()> {
    let mut text = String::new();
    for dir in dirs {
        text.push_str(&format!("{}{FIELD}{}\n", name(dir), stamp(dir)?));
    }
    fs::write(out, text)?;
    Ok(())
}

/// The units the list at `path` names, each with the stamp of its manifest.
fn listed(path: &Path) -> Result<HashMap<String, String>> {
    let text = fs::read_to_string(path)?;
    Ok(text.lines().filter_map(|line| line.split_once(FIELD)).map(|(unit, stamp)| (unit.to_string(), stamp.to_string())).collect())
}

/// The stamp of the manifest of the unit in `dir`, which every rebuild of the unit changes.
fn stamp(dir: &Path) -> Result<String> {
    let path = dir.join(MANIFEST);
    files::stamp(&path).ok_or_else(|| Error(format!("reading {}", path.display())))
}

fn name(dir: &Path) -> &str {
    dir.file_name().and_then(|name| name.to_str()).unwrap_or_default()
}
