use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::outline::tree::{
    Crate as OutlineCrate, Decl, File, Held, Impl, Macro, MemberBlock, Module, State, Textual,
    Waiting,
};
use crate::sema::summary::codec::write_u32;

use super::tokens::{
    Reader, read_vis, u32_le, write_opt_u32, write_path, write_paths, write_str, write_usize,
    write_vis,
};
use super::wire::{Tables, opt_u32_le};

fn write_waiting<'a>(out: &mut Vec<u8>, w: &'a Waiting, tables: &mut Tables<'a>) -> Option<()> {
    write_u32(out, tables.file(&w.file));
    write_usize(out, w.owner)?;
    write_u32(out, w.at);
    let scope = w.scope.as_ref().map(|s| tables.textual(s));
    write_opt_u32(out, scope);
    Some(())
}

#[inline(always)]
fn decode_waiting(b: &[u8], files: &[Arc<File>], textuals: &[Arc<Textual>]) -> Result<Waiting> {
    let file = Arc::clone(
        files
            .get(u32_le(b, 0) as usize)
            .context("waiting file out of bounds")?,
    );
    let owner = u32_le(b, 4) as usize;
    let at = u32_le(b, 8);
    let scope_raw = u32_le(b, 12);
    let scope = if scope_raw != u32::MAX {
        Some(Arc::clone(
            textuals
                .get(scope_raw as usize)
                .context("waiting scope out of bounds")?,
        ))
    } else {
        None
    };
    Ok(Waiting {
        file,
        owner,
        at,
        scope,
    })
}

fn write_held<'a>(out: &mut Vec<u8>, h: &'a Held, tables: &mut Tables<'a>) -> Option<()> {
    write_u32(out, tables.file(&h.file));
    write_usize(out, h.owner)?;
    write_u32(out, h.at);
    write_u32(out, h.path.0);
    write_u32(out, h.path.1);
    Some(())
}

fn write_impls(out: &mut Vec<u8>, impls: &[Impl]) -> Option<()> {
    write_usize(out, impls.len())?;
    for block in impls {
        write_usize(out, block.file)?;
        write_u32(out, block.at);
        write_usize(out, block.module)?;
        write_usize(out, block.parts.len())?;
        for &part in &block.parts {
            write_usize(out, part)?;
        }
    }
    Some(())
}

fn read_impls(reader: &mut Reader<'_>) -> Result<Vec<Impl>> {
    let count = reader.count()?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut impls = Vec::with_capacity(count);
    for _ in 0..count {
        let b = reader.take_array::<12>()?;
        let parts = reader.u32s_map(|v| v as usize)?;
        impls.push(Impl {
            file: u32_le(&b, 0) as usize,
            at: u32_le(&b, 4),
            module: u32_le(&b, 8) as usize,
            parts,
        });
    }
    Ok(impls)
}

fn write_local_krates(out: &mut Vec<u8>, krates: &[usize], tables: &Tables<'_>) -> Option<()> {
    write_usize(out, krates.len())?;
    for &dep in krates {
        write_u32(out, tables.local_krate(dep)?);
    }
    Some(())
}

fn read_remapped_krates(
    reader: &mut Reader<'_>,
    remap: &[u32],
    err: &'static str,
) -> Result<Vec<usize>> {
    let count = reader.count()?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let bytes = reader.take(count.checked_mul(4).context("crate list overflow")?)?;
    let mut out = Vec::with_capacity(count);
    for c in bytes.chunks_exact(4) {
        out.push(*remap.get(u32_le(c, 0) as usize).context(err)? as usize);
    }
    Ok(out)
}

fn write_waitings<'a>(
    out: &mut Vec<u8>,
    waitings: &'a [Waiting],
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_usize(out, waitings.len())?;
    for w in waitings {
        write_waiting(out, w, tables)?;
    }
    Some(())
}

fn write_helds<'a>(out: &mut Vec<u8>, helds: &'a [Held], tables: &mut Tables<'a>) -> Option<()> {
    write_usize(out, helds.len())?;
    for h in helds {
        write_held(out, h, tables)?;
    }
    Some(())
}

fn write_crate_files<'a>(
    out: &mut Vec<u8>,
    files: &'a [Arc<File>],
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_usize(out, files.len())?;
    for file in files {
        write_u32(out, tables.file(file));
    }
    Some(())
}

fn write_module<'a>(out: &mut Vec<u8>, module: &'a Module, tables: &mut Tables<'a>) -> Option<()> {
    write_opt_u32(
        out,
        (module.file != usize::MAX).then_some(module.file as u32),
    );
    write_opt_u32(out, module.item);
    write_opt_u32(out, module.parent.map(|p| p as u32));
    write_str(out, &module.name)?;
    write_path(out, &module.dir)?;
    write_usize(out, module.parts.len())?;
    for &part in &module.parts {
        write_usize(out, part)?;
    }
    write_vis(out, module.vis);
    out.push(u8::from(module.macro_use));
    match &module.state {
        State::Declared(Decl::Inline { file, at }) => {
            out.push(0);
            write_usize(out, *file)?;
            write_u32(out, *at);
        }
        State::Declared(Decl::File {
            candidates,
            by_attr,
        }) => {
            out.push(1);
            write_paths(out, candidates)?;
            out.push(u8::from(*by_attr));
        }
        State::Loaded => out.push(2),
        State::Missing => out.push(3),
        State::Gated => out.push(4),
    }
    let textual = module.textual.as_ref().map(|t| tables.textual(t));
    write_opt_u32(out, textual);
    Some(())
}

fn write_modules<'a>(
    out: &mut Vec<u8>,
    modules: &'a [Module],
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_usize(out, modules.len())?;
    for module in modules {
        write_module(out, module, tables)?;
    }
    Some(())
}

fn write_strings(out: &mut Vec<u8>, items: &[String]) -> Option<()> {
    write_usize(out, items.len())?;
    for e in items {
        write_str(out, e)?;
    }
    Some(())
}

fn write_exports<'a>(
    out: &mut Vec<u8>,
    exports: &'a HashMap<String, Arc<Macro>>,
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_usize(out, exports.len())?;
    for (name, mac) in exports {
        write_str(out, name)?;
        write_u32(out, tables.mac(mac));
    }
    Some(())
}

fn write_defined<'a>(
    out: &mut Vec<u8>,
    defined: &'a HashMap<(u32, u32), Arc<Macro>>,
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_usize(out, defined.len())?;
    for (&(file, item), mac) in defined {
        write_u32(out, file);
        write_u32(out, item);
        write_u32(out, tables.mac(mac));
    }
    Some(())
}

fn write_impl_bodies(out: &mut Vec<u8>, bodies: &[(usize, u32)]) -> Option<()> {
    write_usize(out, bodies.len())?;
    for &(file, at) in bodies {
        write_usize(out, file)?;
        write_u32(out, at);
    }
    Some(())
}

fn write_members<'a>(
    out: &mut Vec<u8>,
    members: &'a [(MemberBlock, Waiting)],
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_usize(out, members.len())?;
    for (block, w) in members {
        match *block {
            MemberBlock::Impl(idx) => {
                out.push(0);
                write_usize(out, idx)?;
            }
            MemberBlock::Trait(idx) => {
                out.push(1);
                write_usize(out, idx)?;
            }
        }
        write_waiting(out, w, tables)?;
    }
    Some(())
}

fn write_coords(out: &mut Vec<u8>, coords: &HashSet<(u32, u32)>) -> Option<()> {
    write_usize(out, coords.len())?;
    for &(file, item) in coords {
        write_u32(out, file);
        write_u32(out, item);
    }
    Some(())
}

pub(super) fn write_outline_crate<'a>(
    out: &mut Vec<u8>,
    krate: &'a OutlineCrate,
    tables: &mut Tables<'a>,
) -> Option<()> {
    write_crate_files(out, &krate.files, tables)?;
    write_modules(out, &krate.modules, tables)?;
    write_paths(out, &krate.missing)?;
    write_paths(out, &krate.gated)?;
    write_strings(out, &krate.errors)?;
    write_exports(out, &krate.exports, tables)?;
    write_defined(out, &krate.defined, tables)?;
    write_waitings(out, &krate.deferred, tables)?;
    write_impls(out, &krate.impls)?;
    write_impls(out, &krate.traits)?;
    write_impl_bodies(out, &krate.impl_bodies)?;
    write_members(out, &krate.members, tables)?;
    write_local_krates(out, &krate.fallback, tables)?;
    write_waitings(out, &krate.foreign, tables)?;
    write_waitings(out, &krate.global_asm, tables)?;
    write_helds(out, &krate.derived, tables)?;
    write_helds(out, &krate.attributed, tables)?;
    write_coords(out, &krate.consumed)?;
    write_coords(out, &krate.resolved_tests)?;
    write_local_krates(out, &krate.macro_use, tables)?;
    write_usize(out, krate.link_libs.len())?;
    for lib in &krate.link_libs {
        write_str(out, lib)?;
    }
    write_usize(out, krate.expansions)?;
    Some(())
}

fn read_crate_files(reader: &mut Reader<'_>, files: &[Arc<File>]) -> Result<Vec<Arc<File>>> {
    let file_count = reader.count()?;
    if file_count == 0 {
        return Ok(Vec::new());
    }
    let file_bytes = reader.take(file_count.checked_mul(4).context("crate files overflow")?)?;
    let mut crate_files = Vec::with_capacity(file_count);
    for c in file_bytes.chunks_exact(4) {
        crate_files.push(Arc::clone(
            files
                .get(u32_le(c, 0) as usize)
                .context("crate file out of bounds")?,
        ));
    }
    Ok(crate_files)
}

fn read_module(reader: &mut Reader<'_>, textuals: &[Arc<Textual>]) -> Result<Module> {
    let b = reader.take_array::<12>()?;
    let file = opt_u32_le(&b, 0).map_or(usize::MAX, |v| v as usize);
    let item = opt_u32_le(&b, 4);
    let parent = opt_u32_le(&b, 8).map(|v| v as usize);
    let name = reader.str()?.to_owned();
    let dir = reader.path()?;
    let parts = reader.u32s_map(|v| v as usize)?;
    let vis = read_vis(reader)?;
    let macro_use = reader.u8()? != 0;
    let state = match reader.u8()? {
        0 => State::Declared(Decl::Inline {
            file: reader.u32()? as usize,
            at: reader.u32()?,
        }),
        1 => State::Declared(Decl::File {
            candidates: reader.paths()?,
            by_attr: reader.u8()? != 0,
        }),
        2 => State::Loaded,
        3 => State::Missing,
        4 => State::Gated,
        _ => bail!("invalid module state tag"),
    };
    let textual = match reader.opt_u32()? {
        Some(id) => Some(Arc::clone(
            textuals
                .get(id as usize)
                .context("module textual out of bounds")?,
        )),
        None => None,
    };
    Ok(Module {
        file,
        item,
        parent,
        name,
        dir,
        parts,
        vis,
        macro_use,
        state,
        textual,
    })
}

fn read_modules(reader: &mut Reader<'_>, textuals: &[Arc<Textual>]) -> Result<Vec<Module>> {
    let mod_count = reader.count()?;
    if mod_count == 0 {
        return Ok(Vec::new());
    }
    let mut modules = Vec::with_capacity(mod_count);
    for _ in 0..mod_count {
        modules.push(read_module(reader, textuals)?);
    }
    Ok(modules)
}

fn read_strings(reader: &mut Reader<'_>) -> Result<Vec<String>> {
    let len = reader.count()?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        items.push(reader.str()?.to_owned());
    }
    Ok(items)
}

fn read_exports(
    reader: &mut Reader<'_>,
    macros: &[Arc<Macro>],
) -> Result<HashMap<String, Arc<Macro>>> {
    let export_len = reader.count()?;
    if export_len == 0 {
        return Ok(HashMap::default());
    }
    let mut exports = HashMap::with_capacity_and_hasher(export_len, Default::default());
    for _ in 0..export_len {
        let name = reader.str()?.to_owned();
        let mac = Arc::clone(
            macros
                .get(reader.u32()? as usize)
                .context("export macro out of bounds")?,
        );
        exports.insert(name, mac);
    }
    Ok(exports)
}

fn read_defined(
    reader: &mut Reader<'_>,
    macros: &[Arc<Macro>],
) -> Result<HashMap<(u32, u32), Arc<Macro>>> {
    let def_len = reader.count()?;
    if def_len == 0 {
        return Ok(HashMap::default());
    }
    let def_bytes = reader.take(def_len.checked_mul(12).context("defined macros overflow")?)?;
    let mut defined = HashMap::with_capacity_and_hasher(def_len, Default::default());
    for c in def_bytes.chunks_exact(12) {
        let key = (u32_le(c, 0), u32_le(c, 4));
        let mac = Arc::clone(
            macros
                .get(u32_le(c, 8) as usize)
                .context("defined macro out of bounds")?,
        );
        defined.insert(key, mac);
    }
    Ok(defined)
}

fn read_impl_bodies(reader: &mut Reader<'_>) -> Result<Vec<(usize, u32)>> {
    let body_len = reader.count()?;
    if body_len == 0 {
        return Ok(Vec::new());
    }
    let body_bytes = reader.take(body_len.checked_mul(8).context("impl bodies overflow")?)?;
    Ok(body_bytes
        .chunks_exact(8)
        .map(|c| (u32_le(c, 0) as usize, u32_le(c, 4)))
        .collect())
}

fn read_members(
    reader: &mut Reader<'_>,
    files: &[Arc<File>],
    textuals: &[Arc<Textual>],
) -> Result<Vec<(MemberBlock, Waiting)>> {
    let member_len = reader.count()?;
    if member_len == 0 {
        return Ok(Vec::new());
    }
    let member_bytes = reader.take(member_len.checked_mul(21).context("members overflow")?)?;
    let mut members = Vec::with_capacity(member_len);
    for c in member_bytes.chunks_exact(21) {
        let idx = u32_le(c, 1) as usize;
        let block = match c[0] {
            0 => MemberBlock::Impl(idx),
            1 => MemberBlock::Trait(idx),
            _ => bail!("invalid member block tag"),
        };
        members.push((block, decode_waiting(&c[5..], files, textuals)?));
    }
    Ok(members)
}

fn read_coords(reader: &mut Reader<'_>) -> Result<HashSet<(u32, u32)>> {
    let count = reader.count()?;
    if count == 0 {
        return Ok(HashSet::default());
    }
    let bytes = reader.take(count.checked_mul(8).context("coords overflow")?)?;
    let mut coords = HashSet::with_capacity_and_hasher(count, Default::default());
    for c in bytes.chunks_exact(8) {
        coords.insert((u32_le(c, 0), u32_le(c, 4)));
    }
    Ok(coords)
}

pub(super) fn read_outline_crate(
    reader: &mut Reader<'_>,
    files: &[Arc<File>],
    macros: &[Arc<Macro>],
    textuals: &[Arc<Textual>],
    remap: &[u32],
) -> Result<OutlineCrate> {
    let crate_files = read_crate_files(reader, files)?;
    let modules = read_modules(reader, textuals)?;
    let missing = reader.paths()?;
    let gated = reader.paths()?;
    let errors = read_strings(reader)?;
    let exports = read_exports(reader, macros)?;
    let defined = read_defined(reader, macros)?;
    let deferred = read_waitings(reader, files, textuals)?;
    let impls = read_impls(reader)?;
    let traits = read_impls(reader)?;
    let impl_bodies = read_impl_bodies(reader)?;
    let members = read_members(reader, files, textuals)?;
    let fallback = read_remapped_krates(reader, remap, "fallback crate out of bounds")?;
    let foreign = read_waitings(reader, files, textuals)?;
    let global_asm = read_waitings(reader, files, textuals)?;
    let derived = read_helds(reader, files)?;
    let attributed = read_helds(reader, files)?;
    let consumed = read_coords(reader)?;
    let resolved_tests = read_coords(reader)?;
    let macro_use = read_remapped_krates(reader, remap, "macro_use crate out of bounds")?;
    let link_lib_count = reader.count()?;
    let mut link_libs = Vec::with_capacity(link_lib_count);
    for _ in 0..link_lib_count {
        link_libs.push(reader.str()?.to_owned());
    }
    let expansions = reader.u32()? as usize;
    Ok(OutlineCrate {
        env: Default::default(),
        files: crate_files,
        modules,
        missing,
        gated,
        errors,
        exports,
        defined,
        deferred,
        impls,
        traits,
        impl_bodies,
        members,
        fallback,
        foreign,
        global_asm,
        derived,
        attributed,
        consumed,
        resolved_tests,
        macro_use,
        link_libs,
        expansions,
        cached_included: None,
    })
}

fn read_waitings(
    reader: &mut Reader<'_>,
    files: &[Arc<File>],
    textuals: &[Arc<Textual>],
) -> Result<Vec<Waiting>> {
    let count = reader.count()?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let bytes = reader.take(count.checked_mul(16).context("waitings overflow")?)?;
    let mut out = Vec::with_capacity(count);
    for c in bytes.chunks_exact(16) {
        out.push(decode_waiting(c, files, textuals)?);
    }
    Ok(out)
}

fn read_helds(reader: &mut Reader<'_>, files: &[Arc<File>]) -> Result<Vec<Held>> {
    let count = reader.count()?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let bytes = reader.take(count.checked_mul(20).context("helds overflow")?)?;
    let mut out = Vec::with_capacity(count);
    for c in bytes.chunks_exact(20) {
        let file = Arc::clone(
            files
                .get(u32_le(c, 0) as usize)
                .context("held file out of bounds")?,
        );
        out.push(Held {
            file,
            owner: u32_le(c, 4) as usize,
            at: u32_le(c, 8),
            path: (u32_le(c, 12), u32_le(c, 16)),
        });
    }
    Ok(out)
}
