use std::path::{Path, PathBuf};

use anyhow::Result;
use paths::AbsPathBuf;

const STORE: &str = "haste-generated";
const OUT_MARK: &str = "${OUT_DIR}";
const KEY_FILE: &str = "key";
const STDOUT_FILE: &str = "stdout";
const FILES_DIR: &str = "out";

/// A build script's saved stdout and the files it wrote, placed under its unit's out dir.
pub struct Recorded {
    pub stdout: String,
    pub files: Vec<(AbsPathBuf, Vec<u8>)>,
}

/// Where one package's script output is saved, and the key that must match to use it.
pub struct Place {
    dir: PathBuf,
    key: String,
    out: PathBuf,
}

impl Place {
    /// Names the saved output of package `name` at `version` under the project `root`.
    pub fn new(root: &Path, name: &str, version: &str, key: String, out: &Path) -> Self {
        Place { dir: root.join(STORE).join(format!("{name}-{version}")), key, out: out.to_owned() }
    }
}

fn swap(bytes: Vec<u8>, from: &str, to: &str) -> Vec<u8> {
    let (from, to) = (from.as_bytes(), to.as_bytes());
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at..].starts_with(from) {
            out.extend_from_slice(to);
            at += from.len();
        } else {
            out.push(bytes[at]);
            at += 1;
        }
    }
    out
}

/// Writes saved files to their out dir on disk, so code that reads the out dir finds them.
pub fn write(files: &[(AbsPathBuf, Vec<u8>)]) -> Result<()> {
    for (path, bytes) in files {
        let path = Path::new(path.as_str());
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

fn walk(base: &Path, dir: &Path, into: &mut Vec<(PathBuf, Vec<u8>)>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(base, &path, into)?;
        } else {
            into.push((path.strip_prefix(base)?.to_owned(), std::fs::read(&path)?));
        }
    }
    Ok(())
}

/// Reads the saved output of a script when its key matches, with paths moved to `out`.
pub fn load(place: &Place) -> Option<Recorded> {
    let (dir, out) = (&place.dir, &place.out);
    if std::fs::read_to_string(dir.join(KEY_FILE)).ok()? != place.key {
        return None;
    }
    let out_text = out.to_string_lossy();
    let stdout = std::fs::read_to_string(dir.join(STDOUT_FILE)).ok()?.replace(OUT_MARK, &out_text);
    let mut found = Vec::new();
    let files_dir = dir.join(FILES_DIR);
    if files_dir.is_dir() {
        walk(&files_dir, &files_dir, &mut found).ok()?;
    }
    let files = found.into_iter().map(|(relative, bytes)| (AbsPathBuf::assert_utf8(out.join(relative)), swap(bytes, OUT_MARK, &out_text)));
    Some(Recorded { stdout, files: files.collect() })
}

/// Saves a script's stdout and files as package source, with the out dir path replaced by a mark.
pub fn save(place: &Place, stdout: &str, files: &[(AbsPathBuf, Vec<u8>)]) -> Result<()> {
    let (dir, out) = (&place.dir, &place.out);
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir)?;
    let out_text = out.to_string_lossy();
    std::fs::write(dir.join(KEY_FILE), &place.key)?;
    let directives: String = stdout.lines().filter(|line| line.starts_with("cargo:")).map(|line| format!("{line}\n")).collect();
    std::fs::write(dir.join(STDOUT_FILE), directives.replace(&*out_text, OUT_MARK))?;
    for (path, bytes) in files {
        let to = dir.join(FILES_DIR).join(Path::new(path.as_str()).strip_prefix(out)?);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(to, swap(bytes.clone(), &out_text, OUT_MARK))?;
    }
    Ok(())
}
