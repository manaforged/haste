use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use crate::sema::summary::codec::{Cursor, read_str, write_str, write_u32};

use super::super::lock::Locked;
use super::super::manifest::{Dep, Kind, Manifest, PackageMeta, TargetSpec};
use super::super::script::Output;
use super::super::{Package, Side, Unit};

pub(super) fn write_path(out: &mut Vec<u8>, path: &Path) -> Option<()> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt as _;
        path.as_os_str().as_bytes()
    };
    #[cfg(not(unix))]
    let bytes = path.to_str()?.as_bytes();
    write_u32(out, u32::try_from(bytes.len()).ok()?);
    out.extend_from_slice(bytes);
    Some(())
}

pub(super) fn read_path(cur: &mut Cursor<'_>) -> Result<PathBuf> {
    let len = cur.count()?;
    let bytes = cur.take(len)?;
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt as _;
        Ok(PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec())))
    }
    #[cfg(not(unix))]
    {
        let text = std::str::from_utf8(bytes).context("invalid utf-8 path")?;
        Ok(PathBuf::from(text))
    }
}

fn write_opt_str(out: &mut Vec<u8>, val: Option<&str>) -> Option<()> {
    match val {
        Some(s) => {
            out.push(1);
            write_str(out, s)?;
        }
        None => out.push(0),
    }
    Some(())
}

fn read_opt_str(cur: &mut Cursor<'_>) -> Result<Option<String>> {
    match cur.u8()? {
        0 => Ok(None),
        1 => Ok(Some(read_str(cur)?.to_owned())),
        _ => bail!("invalid option string tag"),
    }
}

pub(super) fn write_opt_path(out: &mut Vec<u8>, val: Option<&Path>) -> Option<()> {
    match val {
        Some(p) => {
            out.push(1);
            write_path(out, p)?;
        }
        None => out.push(0),
    }
    Some(())
}

pub(super) fn read_opt_path(cur: &mut Cursor<'_>) -> Result<Option<PathBuf>> {
    match cur.u8()? {
        0 => Ok(None),
        1 => Ok(Some(read_path(cur)?)),
        _ => bail!("invalid option path tag"),
    }
}

pub(super) fn write_file_stamps(
    out: &mut Vec<u8>,
    files: &[(PathBuf, Option<(u64, i128)>)],
) -> Option<()> {
    write_u32(out, u32::try_from(files.len()).ok()?);
    for (path, stamp) in files {
        write_path(out, path)?;
        match stamp {
            Some((len, nanos)) => {
                out.push(1);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&nanos.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    Some(())
}

pub(super) fn read_file_stamps(
    cur: &mut Cursor<'_>,
) -> Result<Vec<(PathBuf, Option<(u64, i128)>)>> {
    let count = cur.count()?;
    let mut out = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let path = read_path(cur)?;
        let stamp = match cur.u8()? {
            0 => None,
            1 => Some((cur.u64()?, cur.u128()? as i128)),
            _ => bail!("invalid file stamp tag"),
        };
        out.push((path, stamp));
    }
    Ok(out)
}

pub(super) fn write_env_hashes(out: &mut Vec<u8>, envs: &[(String, Option<u64>)]) -> Option<()> {
    write_u32(out, u32::try_from(envs.len()).ok()?);
    for (name, val) in envs {
        write_str(out, name)?;
        match val {
            Some(hash) => {
                out.push(1);
                out.extend_from_slice(&hash.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    Some(())
}

pub(super) fn read_env_hashes(cur: &mut Cursor<'_>) -> Result<Vec<(String, Option<u64>)>> {
    let count = cur.count()?;
    let mut out = Vec::with_capacity(count.min(512));
    for _ in 0..count {
        let name = read_str(cur)?.to_owned();
        let hash = match cur.u8()? {
            0 => None,
            1 => Some(cur.u64()?),
            _ => bail!("invalid env hash tag"),
        };
        out.push((name, hash));
    }
    Ok(out)
}

pub(super) fn write_strs(out: &mut Vec<u8>, items: &[String]) -> Option<()> {
    write_u32(out, u32::try_from(items.len()).ok()?);
    for s in items {
        write_str(out, s)?;
    }
    Some(())
}

pub(super) fn read_strs(cur: &mut Cursor<'_>) -> Result<Vec<String>> {
    let n = cur.count()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_str(cur)?.to_owned());
    }
    Ok(out)
}

fn write_str_pairs(out: &mut Vec<u8>, items: &[(String, String)]) -> Option<()> {
    write_u32(out, u32::try_from(items.len()).ok()?);
    for (k, v) in items {
        write_str(out, k)?;
        write_str(out, v)?;
    }
    Some(())
}

fn read_str_pairs(cur: &mut Cursor<'_>) -> Result<Vec<(String, String)>> {
    let n = cur.count()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push((read_str(cur)?.to_owned(), read_str(cur)?.to_owned()));
    }
    Ok(out)
}

pub(super) fn write_opt_bool(out: &mut Vec<u8>, val: Option<bool>) {
    out.push(match val {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    });
}

pub(super) fn read_opt_bool(cur: &mut Cursor<'_>) -> Result<Option<bool>> {
    match cur.u8()? {
        0 => Ok(None),
        1 => Ok(Some(false)),
        2 => Ok(Some(true)),
        _ => bail!("invalid opt bool tag"),
    }
}

fn kind_tag(kind: Kind) -> u8 {
    match kind {
        Kind::Normal => 0,
        Kind::Build => 1,
        Kind::Dev => 2,
    }
}

fn kind_from_tag(tag: u8) -> Result<Kind> {
    match tag {
        0 => Ok(Kind::Normal),
        1 => Ok(Kind::Build),
        2 => Ok(Kind::Dev),
        _ => bail!("invalid dep kind tag"),
    }
}

fn write_targets(out: &mut Vec<u8>, specs: &[TargetSpec]) -> Option<()> {
    write_u32(out, u32::try_from(specs.len()).ok()?);
    for spec in specs {
        write_str(out, &spec.name)?;
        write_path(out, &spec.path)?;
        write_strs(out, &spec.required_features)?;
    }
    Some(())
}

fn read_targets(cur: &mut Cursor<'_>) -> Result<Vec<TargetSpec>> {
    let n = cur.count()?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(TargetSpec {
            name: read_str(cur)?.to_owned(),
            path: read_path(cur)?,
            required_features: read_strs(cur)?,
        });
    }
    Ok(out)
}

pub(super) fn write_package(out: &mut Vec<u8>, pkg: &Package) -> Option<()> {
    write_str(out, &pkg.locked.name)?;
    write_str(out, &pkg.locked.version.to_string())?;
    write_opt_str(out, pkg.locked.source.as_deref())?;
    write_opt_str(out, pkg.locked.checksum.as_deref())?;
    write_strs(out, &pkg.locked.dependencies)?;

    let m = &pkg.manifest;
    write_path(out, &m.dir)?;
    write_str(out, &m.name)?;
    let mver = m.version.as_ref().map(ToString::to_string);
    write_opt_str(out, mver.as_deref())?;
    write_str(out, &m.edition)?;
    for s in [
        &m.meta.authors,
        &m.meta.description,
        &m.meta.homepage,
        &m.meta.repository,
        &m.meta.license,
        &m.meta.license_file,
        &m.meta.readme,
        &m.meta.rust_version,
    ] {
        write_str(out, s)?;
    }
    write_opt_str(out, m.links.as_deref())?;
    write_opt_str(out, m.default_run.as_deref())?;
    write_opt_path(out, m.build.as_deref())?;
    write_opt_path(out, m.lib.as_deref())?;
    write_opt_str(out, m.lib_name.as_deref())?;
    out.push(u8::from(m.proc_macro));
    write_targets(out, &m.bins)?;
    write_targets(out, &m.examples)?;
    write_targets(out, &m.tests)?;
    write_targets(out, &m.benches)?;
    write_u32(out, u32::try_from(m.features.len()).ok()?);
    for (k, v) in &m.features {
        write_str(out, k)?;
        write_strs(out, v)?;
    }
    write_u32(out, u32::try_from(m.deps.len()).ok()?);
    for d in &m.deps {
        write_str(out, &d.key)?;
        write_str(out, &d.package)?;
        let req = d.req.as_ref().map(ToString::to_string);
        write_opt_str(out, req.as_deref())?;
        write_opt_path(out, d.path.as_deref())?;
        out.push(u8::from(d.optional));
        out.push(u8::from(d.default_features));
        write_strs(out, &d.features)?;
        out.push(kind_tag(d.kind));
        write_opt_str(out, d.platform.as_deref())?;
    }
    write_u32(out, u32::try_from(pkg.resolved.len()).ok()?);
    for (k, &v) in &pkg.resolved {
        write_str(out, k)?;
        write_u32(out, u32::try_from(v).ok()?);
    }
    out.push(u8::from(pkg.sysroot));
    Some(())
}

pub(super) fn read_package(cur: &mut Cursor<'_>) -> Result<Package> {
    let locked = Locked {
        name: read_str(cur)?.to_owned(),
        version: semver::Version::parse(read_str(cur)?).context("invalid locked version")?,
        source: read_opt_str(cur)?,
        checksum: read_opt_str(cur)?,
        dependencies: read_strs(cur)?,
    };
    let dir = read_path(cur)?;
    let name = read_str(cur)?.to_owned();
    let version = read_opt_str(cur)?
        .map(|s| semver::Version::parse(&s))
        .transpose()
        .context("invalid manifest version")?;
    let edition = read_str(cur)?.to_owned();
    let meta = PackageMeta {
        authors: read_str(cur)?.to_owned(),
        description: read_str(cur)?.to_owned(),
        homepage: read_str(cur)?.to_owned(),
        repository: read_str(cur)?.to_owned(),
        license: read_str(cur)?.to_owned(),
        license_file: read_str(cur)?.to_owned(),
        readme: read_str(cur)?.to_owned(),
        rust_version: read_str(cur)?.to_owned(),
    };
    let links = read_opt_str(cur)?;
    let default_run = read_opt_str(cur)?;
    let build = read_opt_path(cur)?;
    let lib = read_opt_path(cur)?;
    let lib_name = read_opt_str(cur)?;
    let proc_macro = cur.u8()? != 0;
    let bins = read_targets(cur)?;
    let examples = read_targets(cur)?;
    let tests = read_targets(cur)?;
    let benches = read_targets(cur)?;
    let feat_len = cur.count()?;
    let mut features = BTreeMap::new();
    for _ in 0..feat_len {
        let k = read_str(cur)?.to_owned();
        let v = read_strs(cur)?;
        features.insert(k, v);
    }
    let dep_len = cur.count()?;
    let mut deps = Vec::with_capacity(dep_len);
    for _ in 0..dep_len {
        let key = read_str(cur)?.to_owned();
        let package = read_str(cur)?.to_owned();
        let req = read_opt_str(cur)?
            .map(|s| semver::VersionReq::parse(&s))
            .transpose()
            .context("invalid dep req")?;
        let path = read_opt_path(cur)?;
        let optional = cur.u8()? != 0;
        let default_features = cur.u8()? != 0;
        let dfeatures = read_strs(cur)?;
        let kind = kind_from_tag(cur.u8()?)?;
        let platform = read_opt_str(cur)?;
        deps.push(Dep {
            key,
            package,
            req,
            path,
            optional,
            default_features,
            features: dfeatures,
            kind,
            platform,
        });
    }
    let manifest = Manifest {
        dir,
        name,
        version,
        edition,
        meta,
        links,
        default_run,
        build,
        lib,
        lib_name,
        proc_macro,
        bins,
        examples,
        tests,
        benches,
        features,
        deps,
    };
    let res_len = cur.count()?;
    let mut resolved = BTreeMap::new();
    for _ in 0..res_len {
        resolved.insert(read_str(cur)?.to_owned(), cur.u32()? as usize);
    }
    let sysroot = cur.u8()? != 0;
    Ok(Package {
        locked,
        manifest,
        resolved,
        sysroot,
    })
}

fn write_output(out: &mut Vec<u8>, s: &Output) -> Option<()> {
    write_u32(out, u32::try_from(s.names.len()).ok()?);
    for n in &s.names {
        write_str(out, n)?;
    }
    write_u32(out, u32::try_from(s.pairs.len()).ok()?);
    for (k, v) in &s.pairs {
        write_str(out, k)?;
        write_str(out, v)?;
    }
    write_str_pairs(out, &s.vars)?;
    write_strs(out, &s.libs)?;
    write_strs(out, &s.search)?;
    write_str_pairs(out, &s.metadata)?;
    write_strs(out, &s.watched)?;
    write_strs(out, &s.watched_vars)?;
    write_strs(out, &s.errors)?;
    Some(())
}

fn read_output(cur: &mut Cursor<'_>) -> Result<Output> {
    let n_len = cur.count()?;
    let mut names = BTreeSet::new();
    for _ in 0..n_len {
        names.insert(read_str(cur)?.to_owned());
    }
    let p_len = cur.count()?;
    let mut pairs = BTreeSet::new();
    for _ in 0..p_len {
        pairs.insert((read_str(cur)?.to_owned(), read_str(cur)?.to_owned()));
    }
    Ok(Output {
        names,
        pairs,
        vars: read_str_pairs(cur)?,
        libs: read_strs(cur)?,
        search: read_strs(cur)?,
        metadata: read_str_pairs(cur)?,
        watched: read_strs(cur)?,
        watched_vars: read_strs(cur)?,
        errors: read_strs(cur)?,
    })
}

pub(super) fn write_unit(out: &mut Vec<u8>, unit: &Unit) -> Option<()> {
    write_u32(out, u32::try_from(unit.package).ok()?);
    out.push(u8::from(unit.side == Side::Host));
    write_u32(out, u32::try_from(unit.features.len()).ok()?);
    for f in &unit.features {
        write_str(out, f)?;
    }
    write_u32(out, u32::try_from(unit.deps.len()).ok()?);
    for (name, dep, kind) in &unit.deps {
        write_str(out, name)?;
        write_u32(out, u32::try_from(*dep).ok()?);
        out.push(kind_tag(*kind));
    }
    match &unit.script {
        Some(s) => {
            out.push(1);
            write_output(out, s)?;
        }
        None => out.push(0),
    }
    Some(())
}

pub(super) fn read_unit(cur: &mut Cursor<'_>) -> Result<Unit> {
    let package = cur.u32()? as usize;
    let side = if cur.u8()? != 0 {
        Side::Host
    } else {
        Side::Target
    };
    let feat_len = cur.count()?;
    let mut features = BTreeSet::new();
    for _ in 0..feat_len {
        features.insert(read_str(cur)?.to_owned());
    }
    let dep_len = cur.count()?;
    let mut deps = Vec::with_capacity(dep_len);
    for _ in 0..dep_len {
        let name = read_str(cur)?.to_owned();
        let dep = cur.u32()? as usize;
        let kind = kind_from_tag(cur.u8()?)?;
        deps.push((name, dep, kind));
    }
    let script = if cur.u8()? != 0 {
        Some(Arc::new(read_output(cur)?))
    } else {
        None
    };
    Ok(Unit {
        package,
        side,
        features,
        deps,
        script,
    })
}
