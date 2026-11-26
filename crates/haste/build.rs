use std::path::{Path, PathBuf};

fn resolve_git_dirs(manifest_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let git_dir = manifest_dir.ancestors().take(4).find_map(|dir| {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            Some(dot_git)
        } else if dot_git.is_file() {
            let text = std::fs::read_to_string(&dot_git).ok()?;
            let rel = text.trim().strip_prefix("gitdir:")?.trim();
            let resolved = dir.join(rel);
            resolved.is_dir().then_some(resolved)
        } else {
            None
        }
    })?;
    let common_dir = std::fs::read_to_string(git_dir.join("commondir"))
        .ok()
        .map(|rel| git_dir.join(rel.trim()))
        .filter(|dir| dir.is_dir())
        .unwrap_or_else(|| git_dir.clone());
    Some((git_dir, common_dir))
}

fn read_git_sha(manifest_dir: &Path) -> Option<String> {
    let (git_dir, common_dir) = resolve_git_dirs(manifest_dir)?;
    let head_path = git_dir.join("HEAD");
    println!("cargo:rerun-if-changed={}", head_path.display());
    let head = std::fs::read_to_string(&head_path).ok()?;
    let head = head.trim();
    let sha = if let Some(ref_name) = head.strip_prefix("ref: ") {
        let ref_path = common_dir.join(ref_name);
        println!("cargo:rerun-if-changed={}", ref_path.display());
        if ref_path.is_file() {
            std::fs::read_to_string(&ref_path).ok()?.trim().to_owned()
        } else {
            let packed = common_dir.join("packed-refs");
            if packed.is_file() {
                println!("cargo:rerun-if-changed={}", packed.display());
            }
            let text = std::fs::read_to_string(&packed).ok()?;
            text.lines()
                .filter(|line| !line.starts_with('#') && !line.starts_with('^'))
                .find_map(|line| {
                    let (hash, name) = line.split_once(' ')?;
                    (name.trim() == ref_name).then(|| hash.trim().to_owned())
                })?
        }
    } else {
        head.to_owned()
    };
    (sha.len() >= 7).then(|| sha[..8.min(sha.len())].to_owned())
}

fn main() {
    println!("cargo:rerun-if-env-changed=HASTE_GIT_SHA");
    let sha = std::env::var("HASTE_GIT_SHA")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let dir = std::env::var_os("CARGO_MANIFEST_DIR")?;
            read_git_sha(Path::new(&dir))
        })
        .unwrap_or_default();
    println!("cargo:rustc-env=HASTE_GIT_SHA={sha}");
}
