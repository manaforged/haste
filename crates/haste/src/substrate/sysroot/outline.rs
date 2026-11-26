use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::outline::tree::{
    Crate as OutlineCrate, Decl, File, Held, Impl, Macro, MemberBlock, Module, State, Textual,
    Waiting,
};
use crate::sema::summary::codec::{Cursor, write_u32};

use super::tokens::{
    read_opt_u32, read_path, read_paths, read_str, read_u32s_map, read_vis, u32_le, write_opt_u32,
    write_path, write_paths, write_str, write_usize, write_vis,
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

fn read_impls(cur: &mut Cursor<'_>) -> Result<Vec<Impl>> {
    let count = cur.count()?;
    let mut impls = Vec::with_capacity(count);
    for _ in 0..count {
        let b = cur.take(12)?;
        let parts = read_u32s_map(cur, |v| v as usize)?;
        impls.push(Impl {
            file: u32_le(b, 0) as usize,
            at: u32_le(b, 4),
            module: u32_le(b, 8) as usize,
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
    cur: &mut Cursor<'_>,
    remap: &[u32],
    err: &'static str,
) -> Result<Vec<usize>> {
    let count = cur.count()?;
    let bytes = cur.take(count.checked_mul(4).context("crate list overflow")?)?;
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

fn write_consumed(out: &mut Vec<u8>, consumed: &HashSet<(u32, u32)>) -> Option<()> {
    write_usize(out, consumed.len())?;
    for &(file, item) in consumed {
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
    write_consumed(out, &krate.consumed)?;
    write_local_krates(out, &krate.macro_use, tables)?;
    write_usize(out, krate.expansions)?;
    Some(())
}

fn read_crate_files(cur: &mut Cursor<'_>, files: &[Arc<File>]) -> Result<Vec<Arc<File>>> {
    let file_count = cur.count()?;
    let file_bytes = cur.take(file_count.checked_mul(4).context("crate files overflow")?)?;
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

fn read_module(cur: &mut Cursor<'_>, textuals: &[Arc<Textual>]) -> Result<Module> {
    let b = cur.take(12)?;
    let file = opt_u32_le(b, 0).map_or(usize::MAX, |v| v as usize);
    let item = opt_u32_le(b, 4);
    let parent = opt_u32_le(b, 8).map(|v| v as usize);
    let name = read_str(cur)?.to_owned();
    let dir = read_path(cur)?;
    let parts = read_u32s_map(cur, |v| v as usize)?;
    let vis = read_vis(cur)?;
    let macro_use = cur.u8()? != 0;
    let state = match cur.u8()? {
        0 => State::Declared(Decl::Inline {
            file: cur.u32()? as usize,
            at: cur.u32()?,
        }),
        1 => State::Declared(Decl::File {
            candidates: read_paths(cur)?,
            by_attr: cur.u8()? != 0,
        }),
        2 => State::Loaded,
        3 => State::Missing,
        4 => State::Gated,
        _ => bail!("invalid module state tag"),
    };
    let textual = match read_opt_u32(cur)? {
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

fn read_modules(cur: &mut Cursor<'_>, textuals: &[Arc<Textual>]) -> Result<Vec<Module>> {
    let mod_count = cur.count()?;
    let mut modules = Vec::with_capacity(mod_count);
    for _ in 0..mod_count {
        modules.push(read_module(cur, textuals)?);
    }
    Ok(modules)
}

fn read_strings(cur: &mut Cursor<'_>) -> Result<Vec<String>> {
    let len = cur.count()?;
    let mut items = Vec::with_capacity(len);
    for _ in 0..len {
        items.push(read_str(cur)?.to_owned());
    }
    Ok(items)
}

fn read_exports(
    cur: &mut Cursor<'_>,
    macros: &[Arc<Macro>],
) -> Result<HashMap<String, Arc<Macro>>> {
    let export_len = cur.count()?;
    let mut exports = HashMap::with_capacity_and_hasher(export_len, Default::default());
    for _ in 0..export_len {
        let name = read_str(cur)?.to_owned();
        let mac = Arc::clone(
            macros
                .get(cur.u32()? as usize)
                .context("export macro out of bounds")?,
        );
        exports.insert(name, mac);
    }
    Ok(exports)
}

fn read_defined(
    cur: &mut Cursor<'_>,
    macros: &[Arc<Macro>],
) -> Result<HashMap<(u32, u32), Arc<Macro>>> {
    let def_len = cur.count()?;
    let def_bytes = cur.take(def_len.checked_mul(12).context("defined macros overflow")?)?;
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

fn read_impl_bodies(cur: &mut Cursor<'_>) -> Result<Vec<(usize, u32)>> {
    let body_len = cur.count()?;
    let body_bytes = cur.take(body_len.checked_mul(8).context("impl bodies overflow")?)?;
    Ok(body_bytes
        .chunks_exact(8)
        .map(|c| (u32_le(c, 0) as usize, u32_le(c, 4)))
        .collect())
}

fn read_members(
    cur: &mut Cursor<'_>,
    files: &[Arc<File>],
    textuals: &[Arc<Textual>],
) -> Result<Vec<(MemberBlock, Waiting)>> {
    let member_len = cur.count()?;
    let member_bytes = cur.take(member_len.checked_mul(21).context("members overflow")?)?;
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

fn read_consumed(cur: &mut Cursor<'_>) -> Result<HashSet<(u32, u32)>> {
    let consumed_len = cur.count()?;
    let consumed_bytes = cur.take(consumed_len.checked_mul(8).context("consumed overflow")?)?;
    let mut consumed = HashSet::with_capacity_and_hasher(consumed_len, Default::default());
    for c in consumed_bytes.chunks_exact(8) {
        consumed.insert((u32_le(c, 0), u32_le(c, 4)));
    }
    Ok(consumed)
}

pub(super) fn read_outline_crate(
    cur: &mut Cursor<'_>,
    files: &[Arc<File>],
    macros: &[Arc<Macro>],
    textuals: &[Arc<Textual>],
    remap: &[u32],
) -> Result<OutlineCrate> {
    let crate_files = read_crate_files(cur, files)?;
    let modules = read_modules(cur, textuals)?;
    let missing = read_paths(cur)?;
    let gated = read_paths(cur)?;
    let errors = read_strings(cur)?;
    let exports = read_exports(cur, macros)?;
    let defined = read_defined(cur, macros)?;
    let deferred = read_waitings(cur, files, textuals)?;
    let impls = read_impls(cur)?;
    let traits = read_impls(cur)?;
    let impl_bodies = read_impl_bodies(cur)?;
    let members = read_members(cur, files, textuals)?;
    let fallback = read_remapped_krates(cur, remap, "fallback crate out of bounds")?;
    let foreign = read_waitings(cur, files, textuals)?;
    let global_asm = read_waitings(cur, files, textuals)?;
    let derived = read_helds(cur, files)?;
    let attributed = read_helds(cur, files)?;
    let consumed = read_consumed(cur)?;
    let macro_use = read_remapped_krates(cur, remap, "macro_use crate out of bounds")?;
    let expansions = cur.u32()? as usize;
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
        macro_use,
        expansions,
    })
}

fn read_waitings(
    cur: &mut Cursor<'_>,
    files: &[Arc<File>],
    textuals: &[Arc<Textual>],
) -> Result<Vec<Waiting>> {
    let count = cur.count()?;
    let bytes = cur.take(count.checked_mul(16).context("waitings overflow")?)?;
    let mut out = Vec::with_capacity(count);
    for c in bytes.chunks_exact(16) {
        out.push(decode_waiting(c, files, textuals)?);
    }
    Ok(out)
}

fn read_helds(cur: &mut Cursor<'_>, files: &[Arc<File>]) -> Result<Vec<Held>> {
    let count = cur.count()?;
    let bytes = cur.take(count.checked_mul(20).context("helds overflow")?)?;
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
