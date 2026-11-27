use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const NOOP_MAGIC: [u8; 8] = *b"HSTNOP03";
const NOOP_HEADER_LEN: usize = 52;
const WS_MAGIC: [u8; 8] = *b"HSTWKS01";
const WS_HEADER_LEN: usize = 28;
const LOCAL_MAGIC: [u8; 8] = *b"HSTLCHK2";
const FILES_MAGIC: [u8; 8] = *b"HSTLFIL2";
const LOCAL_CHECK_HEADER: usize = 24;
const LOCAL_FILES_HEADER: usize = 28;
const LOCAL_META_MAGIC: [u8; 8] = *b"HSTLMT01";
const LOCAL_META_HEADER: usize = 28;

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
        noop_matches_workspace(data, roots)
    });
    removed += remove_matching_in(&home.join("graph"), "ws", |data| {
        ws_matches_workspace(data, roots)
    });
    removed += remove_matching_in(&home.join("local"), "check", |data| {
        local_check_matches_workspace(data, roots)
    });
    removed += remove_matching_in(&home.join("local"), "files", |data| {
        local_files_matches_workspace(data, roots)
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
            if let Some(content_key) = local_index_meta_match(&data, roots) {
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

fn path_in_workspace(candidate: &Path, roots: &[PathBuf]) -> bool {
    let matches_any = |p: &Path| roots.iter().any(|r| p == r || p.starts_with(r));
    if matches_any(candidate) {
        return true;
    }
    std::fs::canonicalize(candidate).is_ok_and(|c| matches_any(&c))
}

fn read_u32(data: &[u8], at: usize) -> Option<u32> {
    let slice = data.get(at..at + 4)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Some(u32::from_le_bytes(buf))
}

fn read_u64(data: &[u8], at: usize) -> Option<u64> {
    let slice = data.get(at..at + 8)?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Some(u64::from_le_bytes(buf))
}

fn take_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(len)?;
    let slice = data.get(*pos..end)?;
    *pos = end;
    Some(slice)
}

fn take_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let slice = take_bytes(data, pos, 4)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Some(u32::from_le_bytes(buf))
}

fn skip_len_bytes(data: &[u8], pos: &mut usize) -> Option<()> {
    let len = take_u32(data, pos)? as usize;
    take_bytes(data, pos, len)?;
    Some(())
}

fn read_path(data: &[u8], pos: &mut usize) -> Option<PathBuf> {
    let len = take_u32(data, pos)? as usize;
    let bytes = take_bytes(data, pos, len)?;
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt as _;
        Some(PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec())))
    }
    #[cfg(not(unix))]
    {
        let text = std::str::from_utf8(bytes).ok()?;
        Some(PathBuf::from(text))
    }
}

fn skip_opt_tag(data: &[u8], pos: &mut usize, some_bytes: usize) -> Option<()> {
    match *take_bytes(data, pos, 1)?.first()? {
        0 => Some(()),
        1 => {
            take_bytes(data, pos, some_bytes)?;
            Some(())
        }
        _ => None,
    }
}

fn noop_matches_workspace(data: &[u8], roots: &[PathBuf]) -> bool {
    if data.len() < NOOP_HEADER_LEN || data[..8] != NOOP_MAGIC {
        return false;
    }
    let Some(total_count) = read_u32(data, 40) else {
        return false;
    };
    let Some(dir_count) = read_u32(data, 44) else {
        return false;
    };
    let Some(env_count) = read_u32(data, 48) else {
        return false;
    };
    let mut pos = NOOP_HEADER_LEN;
    for _ in 0..env_count {
        if skip_len_bytes(data, &mut pos).is_none() || skip_opt_tag(data, &mut pos, 8).is_none() {
            return false;
        }
    }
    for _ in 0..total_count {
        let Some(p) = read_path(data, &mut pos) else {
            return false;
        };
        if skip_opt_tag(data, &mut pos, 24).is_none() {
            return false;
        }
        if path_in_workspace(&p, roots) {
            return true;
        }
    }
    for _ in 0..dir_count {
        let Some(d) = read_path(data, &mut pos) else {
            return false;
        };
        if path_in_workspace(&d, roots) {
            return true;
        }
    }
    false
}

fn ws_matches_workspace(data: &[u8], roots: &[PathBuf]) -> bool {
    if data.len() < WS_HEADER_LEN + 8 || data[..8] != WS_MAGIC {
        return false;
    }
    let mut pos = WS_HEADER_LEN;
    let Some(file_count) = take_u32(data, &mut pos) else {
        return false;
    };
    for _ in 0..file_count {
        let Some(p) = read_path(data, &mut pos) else {
            return false;
        };
        if skip_opt_tag(data, &mut pos, 24).is_none() {
            return false;
        }
        if path_in_workspace(&p, roots) {
            return true;
        }
    }
    let Some(env_count) = take_u32(data, &mut pos) else {
        return false;
    };
    for _ in 0..env_count {
        if skip_len_bytes(data, &mut pos).is_none() || skip_opt_tag(data, &mut pos, 8).is_none() {
            return false;
        }
    }
    for _ in 0..2 {
        let Some(count) = take_u32(data, &mut pos) else {
            return false;
        };
        for _ in 0..count {
            let Some(p) = read_path(data, &mut pos) else {
                return false;
            };
            if path_in_workspace(&p, roots) {
                return true;
            }
        }
    }
    false
}

fn scan_included_paths(data: &[u8], mut pos: usize, inc_count: u32, roots: &[PathBuf]) -> bool {
    for _ in 0..inc_count {
        let Some(p) = read_path(data, &mut pos) else {
            return false;
        };
        if skip_opt_tag(data, &mut pos, 24).is_none() {
            return false;
        }
        if path_in_workspace(&p, roots) {
            return true;
        }
    }
    false
}

fn local_check_matches_workspace(data: &[u8], roots: &[PathBuf]) -> bool {
    if data.len() < LOCAL_CHECK_HEADER || data[..8] != LOCAL_MAGIC {
        return false;
    }
    let Some(inc_count) = read_u32(data, 16) else {
        return false;
    };
    scan_included_paths(data, LOCAL_CHECK_HEADER, inc_count, roots)
}

fn local_files_matches_workspace(data: &[u8], roots: &[PathBuf]) -> bool {
    if data.len() < LOCAL_FILES_HEADER || data[..8] != FILES_MAGIC {
        return false;
    }
    let Some(file_count) = read_u32(data, 16) else {
        return false;
    };
    let Some(inc_count) = read_u32(data, 20) else {
        return false;
    };
    let Some(pos) = LOCAL_FILES_HEADER.checked_add((file_count as usize).saturating_mul(16)) else {
        return false;
    };
    scan_included_paths(data, pos, inc_count, roots)
}

fn local_index_meta_match(data: &[u8], roots: &[PathBuf]) -> Option<u64> {
    if data.len() < LOCAL_META_HEADER + 8 || data[..8] != LOCAL_META_MAGIC {
        return None;
    }
    let content_key = read_u64(data, 20)?;
    let mut pos = LOCAL_META_HEADER;
    let dep_count = take_u32(data, &mut pos)? as usize;
    take_bytes(data, &mut pos, dep_count.checked_mul(16)?)?;
    let watch_count = take_u32(data, &mut pos)?;
    for _ in 0..watch_count {
        let p = read_path(data, &mut pos)?;
        skip_opt_tag(data, &mut pos, 24)?;
        if path_in_workspace(&p, roots) {
            return Some(content_key);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_len_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }

    fn write_path(out: &mut Vec<u8>, path: &Path) {
        #[cfg(unix)]
        let bytes = {
            use std::os::unix::ffi::OsStrExt as _;
            path.as_os_str().as_bytes()
        };
        #[cfg(not(unix))]
        let bytes = path.to_string_lossy().as_bytes().to_vec();
        write_len_bytes(out, &bytes);
    }

    fn fake_noop(path: &Path) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&NOOP_MAGIC);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        write_path(&mut out, path);
        out.push(0);
        out
    }

    fn fake_ws(path: &Path) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&WS_MAGIC);
        out.extend_from_slice(&6u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        write_path(&mut out, path);
        out.push(0);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&WS_MAGIC);
        out
    }

    fn fake_local_check(included: &Path) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&LOCAL_MAGIC);
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        write_path(&mut out, included);
        out.push(0);
        out
    }

    fn fake_local_meta(content_key: u64, watched: &Path) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&LOCAL_META_MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&content_key.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        write_path(&mut out, watched);
        out.push(0);
        out.extend_from_slice(&LOCAL_META_MAGIC);
        out
    }

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
        let a_meta = home.join("local-index/aaaa.meta");
        let a_bin = home.join("local-index/00000000000000aa.bin");
        let b_meta = home.join("local-index/bbbb.meta");
        let b_bin = home.join("local-index/00000000000000bb.bin");

        std::fs::write(&a_stamp, fake_noop(&ws_a.join("Cargo.toml"))).expect("write");
        std::fs::write(&b_stamp, fake_noop(&ws_b.join("Cargo.toml"))).expect("write");
        std::fs::write(&a_ws, fake_ws(&ws_a.join("Cargo.toml"))).expect("write");
        std::fs::write(&b_ws, fake_ws(&ws_b.join("Cargo.toml"))).expect("write");
        std::fs::write(&a_check, fake_local_check(&ws_a.join("src/msg.txt"))).expect("write");
        std::fs::write(&b_check, fake_local_check(&ws_b.join("src/msg.txt"))).expect("write");
        std::fs::write(&a_meta, fake_local_meta(0xaa, &ws_a.join("src/lib.rs"))).expect("write");
        std::fs::write(&a_bin, b"bin-a").expect("write");
        std::fs::write(&b_meta, fake_local_meta(0xbb, &ws_b.join("src/lib.rs"))).expect("write");
        std::fs::write(&b_bin, b"bin-b").expect("write");

        let canon_a = std::fs::canonicalize(&ws_a).expect("canon");
        let removed = remove_workspace_caches(&home, &[canon_a, ws_a]);
        assert_eq!(removed, 5);
        assert!(!a_stamp.exists() && !a_ws.exists() && !a_check.exists());
        assert!(!a_meta.exists() && !a_bin.exists());
        assert!(b_stamp.exists() && b_ws.exists() && b_check.exists());
        assert!(b_meta.exists() && b_bin.exists());

        drop(std::fs::remove_dir_all(&root));
    }
}
