use std::path::{Path, PathBuf};

use rustc_hash::FxHashSet as HashSet;
use toml::{Table, Value};

fn is_glob_pattern(pattern: &str) -> bool {
    pattern.bytes().any(|b| matches!(b, b'*' | b'?' | b'['))
}

fn match_char_class(p: &[char], pi: usize, ch: char) -> Option<(usize, bool)> {
    let mut i = pi + 1;
    let negated = matches!(p.get(i), Some('!'));
    if negated {
        i += 1;
    }
    let start = i;
    let mut matched = false;
    while i < p.len() {
        if p[i] == ']' && i > start {
            return Some((i + 1, if negated { !matched } else { matched }));
        }
        if i + 2 < p.len() && p[i + 1] == '-' && p[i + 2] != ']' {
            if ch >= p[i] && ch <= p[i + 2] {
                matched = true;
            }
            i += 3;
        } else {
            if ch == p[i] {
                matched = true;
            }
            i += 1;
        }
    }
    None
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut match_ti) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            match_ti = ti;
            pi += 1;
            continue;
        }
        let step = if pi < p.len() && p[pi] == '?' {
            Some(pi + 1)
        } else if pi < p.len() && p[pi] == '[' {
            match match_char_class(&p, pi, t[ti]) {
                Some((next_pi, true)) => Some(next_pi),
                Some((_, false)) => None,
                None if t[ti] == '[' => Some(pi + 1),
                None => None,
            }
        } else if pi < p.len() && p[pi] == t[ti] {
            Some(pi + 1)
        } else {
            None
        };
        if let Some(next_pi) = step {
            pi = next_pi;
            ti += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            match_ti += 1;
            ti = match_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn recursive_subdirs(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = HashSet::default();
    let mut stack: Vec<PathBuf> = dirs.into_iter().rev().collect();
    while let Some(dir) = stack.pop() {
        if !dir.is_dir() {
            continue;
        }
        let key = std::fs::canonicalize(&dir).unwrap_or_else(|_| super::normalize(&dir));
        if !seen.insert(key) {
            continue;
        }
        let mut children = super::super::cache::subdirs(&dir).unwrap_or_default();
        children.sort();
        for child in children.into_iter().rev() {
            stack.push(child);
        }
        out.push(dir);
    }
    out
}

fn advance_pattern_segment(dirs: Vec<PathBuf>, seg: &str) -> Vec<PathBuf> {
    if seg == "**" {
        return recursive_subdirs(dirs);
    }
    let mut next = Vec::new();
    if is_glob_pattern(seg) {
        for dir in dirs {
            for sub in super::super::cache::subdirs(&dir).unwrap_or_default() {
                if sub
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| wildcard_match(seg, name))
                {
                    next.push(sub);
                }
            }
        }
    } else {
        for dir in dirs {
            let candidate = dir.join(seg);
            if candidate.is_dir() {
                next.push(candidate);
            }
        }
    }
    next
}

fn expand_pattern(top: &Path, pattern: &str) -> Vec<PathBuf> {
    let trimmed = pattern.trim_end_matches('/');
    if !is_glob_pattern(trimmed) {
        let path = if trimmed.is_empty() || trimmed == "." {
            top.to_owned()
        } else {
            top.join(trimmed)
        };
        return vec![path];
    }
    let mut dirs = vec![top.to_owned()];
    for seg in trimmed.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        dirs = advance_pattern_segment(dirs, seg);
    }
    dirs
}

/// Workspace member directories listed under `key`, honoring globs and `[workspace].exclude`.
pub(crate) fn workspace_member_dirs(
    top: &Path,
    workspace: Option<&Table>,
    key: &str,
) -> Vec<PathBuf> {
    let top_norm = super::normalize(top);
    let excluded: Vec<PathBuf> = workspace
        .and_then(|ws| ws.get("exclude"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .flat_map(|pat| expand_pattern(top, pat))
        .map(|dir| super::normalize(&dir))
        .collect();
    let is_excluded = |dir: &Path| {
        let norm = super::normalize(dir);
        excluded
            .iter()
            .any(|ex| norm == *ex || (*ex != top_norm && norm.starts_with(ex)))
    };
    let listed = workspace
        .and_then(|ws| ws.get(key))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    let mut out = Vec::new();
    for member in listed {
        let is_glob = is_glob_pattern(member);
        for dir in expand_pattern(top, member) {
            if is_glob && !dir.join(super::super::MANIFEST_FILE).is_file() {
                continue;
            }
            if !is_excluded(&dir) && !out.contains(&dir) {
                out.push(dir);
            }
        }
    }
    out
}

/// Parent and intermediate directories of workspace `members` / `default-members` glob entries.
pub(crate) fn member_glob_dirs(top: &Path) -> Vec<PathBuf> {
    let Ok(Some(ws)) = super::workspace_of(top) else {
        return Vec::new();
    };
    let mut watched = Vec::new();
    for key in ["members", "default-members"] {
        let listed = ws
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str);
        for member in listed {
            let trimmed = member.trim_end_matches('/');
            if !is_glob_pattern(trimmed) {
                continue;
            }
            let mut dirs = vec![top.to_owned()];
            let mut seen_glob = false;
            for seg in trimmed.split('/') {
                if seg.is_empty() || seg == "." {
                    continue;
                }
                if is_glob_pattern(seg) {
                    seen_glob = true;
                }
                if seen_glob {
                    watched.extend(dirs.iter().filter(|dir| dir.is_dir()).cloned());
                }
                dirs = advance_pattern_segment(dirs, seg);
                if seg == "**" {
                    watched.extend(dirs.iter().filter(|dir| dir.is_dir()).cloned());
                }
            }
        }
    }
    watched.sort();
    watched.dedup();
    watched
}
