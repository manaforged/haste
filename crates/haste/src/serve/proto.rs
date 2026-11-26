//! Wire serialization for daemon request payloads (output path, args, and client environment).

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use rustc_hash::FxHashMap;

const MAX_PATH_BYTES: usize = 64 * 1024;
const MAX_ARGS: u32 = 4096;
const MAX_ARG_BYTES: usize = 1024 * 1024;
const MAX_ENVS: u32 = 8192;

pub(super) struct Payload {
    pub(super) out: Option<PathBuf>,
    pub(super) args: Vec<String>,
    pub(super) env: FxHashMap<String, String>,
}

pub(super) fn write_payload(
    conn: &mut std::fs::File,
    out: Option<&Path>,
    args: &[String],
) -> Result<()> {
    use std::os::unix::ffi::OsStrExt as _;
    let out_bytes = out.map_or(&[][..], |p| p.as_os_str().as_bytes());
    let mut bytes = (out_bytes.len() as u32).to_le_bytes().to_vec();
    bytes.extend_from_slice(out_bytes);
    bytes.extend((args.len() as u32).to_le_bytes());
    for arg in args {
        bytes.extend((arg.len() as u32).to_le_bytes());
        bytes.extend(arg.as_bytes());
    }
    let envs: Vec<(String, String)> = std::env::vars().collect();
    bytes.extend((envs.len().min(MAX_ENVS as usize) as u32).to_le_bytes());
    for (k, v) in envs.into_iter().take(MAX_ENVS as usize) {
        bytes.extend((k.len() as u32).to_le_bytes());
        bytes.extend(k.as_bytes());
        bytes.extend((v.len() as u32).to_le_bytes());
        bytes.extend(v.as_bytes());
    }
    Ok(conn.write_all(&bytes)?)
}

fn read_u32(conn: &mut std::fs::File) -> Result<u32> {
    let mut word = [0; 4];
    conn.read_exact(&mut word)?;
    Ok(u32::from_le_bytes(word))
}

fn read_string(conn: &mut std::fs::File, limit: usize, what: &str) -> Result<String> {
    let len = read_u32(conn)? as usize;
    if len > limit {
        bail!("{what} exceeds limit");
    }
    let mut raw = vec![0; len];
    conn.read_exact(&mut raw)?;
    Ok(String::from_utf8(raw)?)
}

pub(super) fn read_payload(conn: &mut std::fs::File) -> Result<Payload> {
    use std::os::unix::ffi::OsStringExt as _;
    let out_len = read_u32(conn)? as usize;
    if out_len > MAX_PATH_BYTES {
        bail!("output path exceeds limit");
    }
    let out = if out_len == 0 {
        None
    } else {
        let mut raw = vec![0; out_len];
        conn.read_exact(&mut raw)?;
        Some(PathBuf::from(std::ffi::OsString::from_vec(raw)))
    };
    let count = read_u32(conn)?;
    if count > MAX_ARGS {
        bail!("argument count exceeds limit");
    }
    let mut args = Vec::with_capacity(count as usize);
    for _ in 0..count {
        args.push(read_string(conn, MAX_ARG_BYTES, "argument")?);
    }
    let env_count = read_u32(conn)?;
    if env_count > MAX_ENVS {
        bail!("environment count exceeds limit");
    }
    let mut env = FxHashMap::with_capacity_and_hasher(env_count as usize, Default::default());
    for _ in 0..env_count {
        let k = read_string(conn, MAX_ARG_BYTES, "env key")?;
        let v = read_string(conn, MAX_ARG_BYTES, "env value")?;
        env.insert(k, v);
    }
    Ok(Payload { out, args, env })
}
