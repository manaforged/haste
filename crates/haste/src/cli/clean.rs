use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub(super) fn run(all: bool, path: &Path) -> Result<()> {
    if all {
        return clean_all();
    }
    clean_workspace(path)
}

fn clean_all() -> Result<()> {
    if crate::serve::stop(None)? != 0 {
        anyhow::bail!("failed to stop active haste daemons");
    }
    let home = crate::graph::store::home();
    let quiet = crate::selection().quiet;
    match std::fs::remove_dir_all(&home) {
        Ok(()) => {
            if !quiet {
                eprintln!("removed {}", home.display());
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            if !quiet {
                eprintln!("cache {} is already empty", home.display());
            }
        }
        Err(err) => return Err(err).with_context(|| format!("removing {}", home.display())),
    }
    Ok(())
}

fn clean_workspace(path: &Path) -> Result<()> {
    let target = if path.as_os_str().is_empty() {
        Path::new(".")
    } else {
        path
    };
    let abs = std::path::absolute(target)?;
    let canon = std::fs::canonicalize(&abs).unwrap_or_else(|_| abs.clone());
    let ws_root = manifest_root_of(&canon);
    let stop_target = if ws_root.join("Cargo.toml").is_file() {
        &ws_root
    } else {
        &canon
    };
    if crate::serve::stop(Some(stop_target))? != 0 {
        anyhow::bail!("failed to stop active haste daemon");
    }
    let mut roots = vec![canon, abs];
    if !roots.contains(&ws_root) {
        roots.push(ws_root.clone());
    }
    let home = crate::graph::store::home();
    let removed = remove_workspace_caches(&home, &roots);
    if !crate::selection().quiet {
        if removed == 0 {
            eprintln!("cache for {} is already clean", ws_root.display());
        } else {
            eprintln!(
                "removed {removed} cached file{} for {}",
                if removed == 1 { "" } else { "s" },
                ws_root.display()
            );
        }
    }
    Ok(())
}

fn remove_workspace_caches(home: &Path, roots: &[PathBuf]) -> usize {
    let mut removed = 0usize;
    removed += remove_matching_in(&home.join("noop"), "stamp", |data| {
        crate::substrate::noop::matches_workspace(data, roots)
    });
    removed += remove_matching_in(&home.join("graph"), "ws", |data| {
        crate::graph::snapshot::matches_workspace(data, roots)
    });
    removed += remove_matching_in(&home.join("local"), "check", |data| {
        crate::substrate::local::check_matches_workspace(data, roots)
    });
    removed += remove_matching_in(&home.join("local"), "files", |data| {
        crate::substrate::local::files_matches_workspace(data, roots)
    });
    let lidx_dir = home.join("local-index");
    if let Ok(entries) = std::fs::read_dir(&lidx_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("meta") {
                continue;
            }
            let Ok(data) = std::fs::read(&path) else {
                continue;
            };
            if let Some(content_key) =
                crate::substrate::local_index::matches_workspace(&data, roots)
            {
                if std::fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
                let bin = lidx_dir.join(format!("{content_key:016x}.bin"));
                if std::fs::remove_file(bin).is_ok() {
                    removed += 1;
                }
            }
        }
    }
    removed
}

fn remove_matching_in(dir: &Path, ext: &str, matches: impl Fn(&[u8]) -> bool) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some(ext) {
            continue;
        }
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        if matches(&data) && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

fn manifest_root_of(canon: &Path) -> PathBuf {
    let start = if canon.is_file() {
        canon.parent().unwrap_or(canon)
    } else {
        canon
    };
    start
        .ancestors()
        .find(|dir| dir.join("Cargo.toml").is_file())
        .unwrap_or(start)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::snapshot::encode_test_snapshot;
    use crate::substrate::local::{encode_test_check, encode_test_files};
    use crate::substrate::local_index::encode_test_meta;
    use crate::substrate::noop::encode_test_stamp;

    #[test]
    fn workspace_clean_removes_only_matching_workspace_caches() {
        let seq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root =
            std::env::temp_dir().join(format!("haste-clean-test-{}-{seq}", std::process::id()));
        drop(std::fs::remove_dir_all(&root));
        let home = root.join("home");
        let ws_a = root.join("ws_a");
        let ws_b = root.join("ws_b");
        for dir in [
            &home.join("noop"),
            &home.join("graph"),
            &home.join("local"),
            &home.join("local-index"),
            &ws_a,
            &ws_b,
        ] {
            std::fs::create_dir_all(dir).expect("mkdir");
        }
        let a_stamp = home.join("noop/aaaa.stamp");
        let b_stamp = home.join("noop/bbbb.stamp");
        let a_ws = home.join("graph/aaaa.ws");
        let b_ws = home.join("graph/bbbb.ws");
        let a_check = home.join("local/aaaa.check");
        let b_check = home.join("local/bbbb.check");
        let a_files = home.join("local/aaaa.files");
        let b_files = home.join("local/bbbb.files");
        let a_meta = home.join("local-index/aaaa.meta");
        let a_bin = home.join("local-index/00000000000000aa.bin");
        let b_meta = home.join("local-index/bbbb.meta");
        let b_bin = home.join("local-index/00000000000000bb.bin");

        std::fs::write(&a_stamp, encode_test_stamp(&ws_a.join("Cargo.toml"))).expect("write");
        std::fs::write(&b_stamp, encode_test_stamp(&ws_b.join("Cargo.toml"))).expect("write");
        std::fs::write(&a_ws, encode_test_snapshot(&ws_a.join("Cargo.toml"))).expect("write");
        std::fs::write(&b_ws, encode_test_snapshot(&ws_b.join("Cargo.toml"))).expect("write");
        std::fs::write(&a_check, encode_test_check(&ws_a.join("src/msg.txt"))).expect("write");
        std::fs::write(&b_check, encode_test_check(&ws_b.join("src/msg.txt"))).expect("write");
        std::fs::write(&a_files, encode_test_files(&ws_a.join("src/msg.txt"))).expect("write");
        std::fs::write(&b_files, encode_test_files(&ws_b.join("src/msg.txt"))).expect("write");
        std::fs::write(&a_meta, encode_test_meta(0xaa, &ws_a.join("src/lib.rs"))).expect("write");
        std::fs::write(&a_bin, b"bin-a").expect("write");
        std::fs::write(&b_meta, encode_test_meta(0xbb, &ws_b.join("src/lib.rs"))).expect("write");
        std::fs::write(&b_bin, b"bin-b").expect("write");

        let canon_a = std::fs::canonicalize(&ws_a).expect("canon");
        let removed = remove_workspace_caches(&home, &[canon_a, ws_a]);
        assert_eq!(removed, 6);
        assert!(!a_stamp.exists() && !a_ws.exists() && !a_check.exists() && !a_files.exists());
        assert!(!a_meta.exists() && !a_bin.exists());
        assert!(b_stamp.exists() && b_ws.exists() && b_check.exists() && b_files.exists());
        assert!(b_meta.exists() && b_bin.exists());

        drop(std::fs::remove_dir_all(&root));
    }
}
