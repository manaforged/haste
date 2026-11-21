use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::names::{Binding, Crate as NamesCrate, Def, ItemRef, ModRef, Ns, Origin, PRIMITIVES, PerNs, Scope, Sym, Unresolved, collect::Pending};
use crate::sema::summary::codec::{Cursor, write_u32};

use super::tokens::{read_opt_u32, read_str, read_vis, write_opt_u32, write_str, write_usize, write_vis};
use super::wire::Tables;

const ORIGIN_KINDS: [Origin; 3] = [Origin::Item, Origin::Import, Origin::Glob];

fn write_def(out: &mut Vec<u8>, def: Def, tables: &Tables<'_>) -> Option<()> {
    match def {
        Def::Module(m) => { out.push(0); write_u32(out, tables.local_krate(m.krate as usize)?); write_u32(out, m.module); }
        Def::Item(i) => { out.push(1); write_u32(out, tables.local_krate(i.krate as usize)?); write_u32(out, i.file); write_u32(out, i.item); }
        Def::Variant(i, v) => { out.push(2); write_u32(out, tables.local_krate(i.krate as usize)?); write_u32(out, i.file); write_u32(out, i.item); write_u32(out, v); }
        Def::Primitive(p) => { out.push(3); out.push(PRIMITIVES.iter().position(|&(_, x)| x == p)? as u8); }
    }
    Some(())
}

fn read_def(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Def> {
    let map_krate = |k: u32| remap.get(k as usize).copied().context("def crate out of bounds");
    Ok(match cur.u8()? {
        0 => Def::Module(ModRef { krate: map_krate(cur.u32()?)?, module: cur.u32()? }),
        1 => Def::Item(ItemRef { krate: map_krate(cur.u32()?)?, file: cur.u32()?, item: cur.u32()? }),
        2 => Def::Variant(ItemRef { krate: map_krate(cur.u32()?)?, file: cur.u32()?, item: cur.u32()? }, cur.u32()?),
        3 => Def::Primitive(PRIMITIVES.get(cur.u8()? as usize).map(|&(_, p)| p).context("invalid primitive tag")?),
        _ => bail!("invalid def tag"),
    })
}

fn write_binding(out: &mut Vec<u8>, b: Binding, tables: &Tables<'_>) -> Option<()> {
    write_def(out, b.def, tables)?;
    write_vis(out, b.vis);
    out.push(ORIGIN_KINDS.iter().position(|&o| o == b.origin)? as u8);
    Some(())
}

fn read_binding(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Binding> {
    Ok(Binding {
        def: read_def(cur, remap)?,
        vis: read_vis(cur)?,
        origin: ORIGIN_KINDS.get(cur.u8()? as usize).copied().context("invalid origin tag")?,
    })
}

pub(super) fn write_names_crate(out: &mut Vec<u8>, krate: &NamesCrate, tables: &mut Tables<'_>) -> Option<()> {
    write_usize(out, krate.scopes.len())?;
    for scope in &krate.scopes {
        write_usize(out, scope.names.len())?;
        for (&sym, per) in &scope.names {
            write_u32(out, tables.sym(sym));
            let mask = u8::from(per.types.is_some()) | (u8::from(per.values.is_some()) << 1) | (u8::from(per.macros.is_some()) << 2);
            out.push(mask);
            for ns in [Ns::Types, Ns::Values, Ns::Macros] {
                if let Some(binding) = per.get(ns) {
                    write_binding(out, binding, tables)?;
                }
            }
        }
        write_usize(out, scope.log.len())?;
        for &sym in &scope.log {
            write_u32(out, tables.sym(sym));
        }
    }
    match krate.prelude {
        Some(m) => { out.push(1); write_u32(out, tables.local_krate(m.krate as usize)?); write_u32(out, m.module); }
        None => out.push(0),
    }
    write_usize(out, krate.unresolved.len())?;
    for u in &krate.unresolved {
        write_u32(out, u.file);
        write_u32(out, u.import);
    }
    write_usize(out, krate.externs.len())?;
    for &(sym, def) in &krate.externs {
        write_u32(out, tables.sym(sym));
        write_def(out, def, tables)?;
    }
    write_usize(out, krate.deps.len())?;
    for &(sym, dep) in &krate.deps {
        write_u32(out, tables.sym(sym));
        write_u32(out, tables.local_krate(dep)?);
    }
    out.push(u8::from(krate.edition_2015));
    write_str(out, &krate.edition)?;
    write_usize(out, krate.collected.len())?;
    for &c in &krate.collected {
        write_opt_u32(out, c.map(|v| v as u32));
    }
    write_usize(out, krate.pending.len())?;
    for p in &krate.pending {
        write_u32(out, p.module);
        write_u32(out, p.file);
        write_u32(out, p.import);
        write_usize(out, p.segments.len())?;
        for &seg in &p.segments {
            write_u32(out, tables.sym(seg));
        }
        out.push(u8::from(p.absolute));
        let dollar = match p.dollar {
            Some(d) => Some(tables.local_krate(d as usize)?),
            None => None,
        };
        write_opt_u32(out, dollar);
        write_opt_u32(out, p.name.map(|s| tables.sym(s)));
        write_vis(out, p.vis);
        match p.stamp {
            Some((def, ver)) => { out.push(1); write_def(out, def, tables)?; write_u32(out, ver); }
            None => out.push(0),
        }
        out.push(u8::from(p.active));
    }
    write_usize(out, krate.enums.len())?;
    for (&(file, item), variants) in &krate.enums {
        write_u32(out, file);
        write_u32(out, item);
        write_usize(out, variants.len())?;
        for &v in variants {
            write_u32(out, tables.sym(v));
        }
    }
    out.push(u8::from(krate.eager));
    Some(())
}

pub(super) fn read_names_crate(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<NamesCrate> {
    let scope_len = cur.count()?;
    let mut scopes = Vec::with_capacity(scope_len);
    for _ in 0..scope_len {
        let name_len = cur.count()?;
        let mut names = HashMap::with_capacity_and_hasher(name_len, Default::default());
        for _ in 0..name_len {
            let sym = Sym(cur.u32()?);
            let mask = cur.u8()?;
            let types = if mask & 1 != 0 { Some(read_binding(cur, remap)?) } else { None };
            let values = if mask & 2 != 0 { Some(read_binding(cur, remap)?) } else { None };
            let macros = if mask & 4 != 0 { Some(read_binding(cur, remap)?) } else { None };
            names.insert(sym, PerNs { types, values, macros });
        }
        let log_len = cur.count()?;
        let mut log = Vec::with_capacity(log_len);
        for _ in 0..log_len {
            log.push(Sym(cur.u32()?));
        }
        scopes.push(Scope { names, log });
    }
    let prelude = if cur.u8()? != 0 {
        let krate = *remap.get(cur.u32()? as usize).context("prelude crate out of bounds")?;
        Some(ModRef { krate, module: cur.u32()? })
    } else {
        None
    };
    let unres_len = cur.count()?;
    let mut unresolved = Vec::with_capacity(unres_len);
    for _ in 0..unres_len {
        unresolved.push(Unresolved { file: cur.u32()?, import: cur.u32()? });
    }
    let ext_len = cur.count()?;
    let mut externs = Vec::with_capacity(ext_len);
    for _ in 0..ext_len {
        externs.push((Sym(cur.u32()?), read_def(cur, remap)?));
    }
    let dep_len = cur.count()?;
    let mut deps = Vec::with_capacity(dep_len);
    for _ in 0..dep_len {
        let sym = Sym(cur.u32()?);
        let dep = *remap.get(cur.u32()? as usize).context("names dep crate out of bounds")? as usize;
        deps.push((sym, dep));
    }
    let edition_2015 = cur.u8()? != 0;
    let edition = read_str(cur)?.to_owned();
    let coll_len = cur.count()?;
    let mut collected = Vec::with_capacity(coll_len);
    for _ in 0..coll_len {
        collected.push(read_opt_u32(cur)?.map(|v| v as usize));
    }
    let pend_len = cur.count()?;
    let mut pending = Vec::with_capacity(pend_len);
    for _ in 0..pend_len {
        let module = cur.u32()?;
        let file = cur.u32()?;
        let import = cur.u32()?;
        let seg_len = cur.count()?;
        let mut segments = Vec::with_capacity(seg_len);
        for _ in 0..seg_len {
            segments.push(Sym(cur.u32()?));
        }
        let absolute = cur.u8()? != 0;
        let dollar = match read_opt_u32(cur)? {
            Some(d) => Some(*remap.get(d as usize).context("pending dollar crate out of bounds")?),
            None => None,
        };
        let name = read_opt_u32(cur)?.map(Sym);
        let vis = read_vis(cur)?;
        let stamp = if cur.u8()? != 0 { Some((read_def(cur, remap)?, cur.u32()?)) } else { None };
        let active = cur.u8()? != 0;
        pending.push(Pending { module, file, import, segments: segments.into_boxed_slice(), absolute, dollar, name, vis, stamp, active });
    }
    let enum_len = cur.count()?;
    let mut enums = HashMap::with_capacity_and_hasher(enum_len, Default::default());
    for _ in 0..enum_len {
        let key = (cur.u32()?, cur.u32()?);
        let v_len = cur.count()?;
        let mut variants = Vec::with_capacity(v_len);
        for _ in 0..v_len {
            variants.push(Sym(cur.u32()?));
        }
        enums.insert(key, variants.into_boxed_slice());
    }
    let eager = cur.u8()? != 0;
    let mut out = NamesCrate::default();
    out.scopes = scopes;
    out.prelude = prelude;
    out.unresolved = unresolved;
    out.externs = externs;
    out.deps = deps;
    out.edition_2015 = edition_2015;
    out.edition = edition;
    out.collected = collected;
    out.pending = pending;
    out.enums = enums;
    out.eager = eager;
    out.settled = true;
    out.rebuild_indices();
    Ok(out)
}
