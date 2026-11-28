use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::names::{
    Binding, Crate as NamesCrate, Def, ItemRef, ModRef, Ns, Origin, PRIMITIVES, PerNs, Scope, Sym,
    UNDERSCORE, Unresolved, collect::Pending,
};
use crate::sema::summary::codec::write_u32;

use super::tokens::{
    Reader, read_vis, u32_le, vis_from_tag, vis_tag, write_opt_u32, write_str, write_usize,
    write_vis,
};
use super::wire::{FIXED_SYM_COUNT, Tables, opt_u32_le};

#[inline(always)]
fn map_sym(raw: u32, sym_remap: &[Sym]) -> Result<Sym> {
    if raw < FIXED_SYM_COUNT || sym_remap.is_empty() {
        Ok(Sym(raw))
    } else {
        sym_remap
            .get((raw - FIXED_SYM_COUNT) as usize)
            .copied()
            .context("symbol id out of bounds")
    }
}

fn read_syms(reader: &mut Reader<'_>, sym_remap: &[Sym]) -> Result<Box<[Sym]>> {
    let count = reader.count()?;
    if count == 0 {
        return Ok(Vec::new().into_boxed_slice());
    }
    let byte_len = count.checked_mul(4).context("u32 slice overflow")?;
    let bytes = reader.take(byte_len)?;
    let mut out = Vec::with_capacity(count);
    if sym_remap.is_empty() {
        for c in bytes.chunks_exact(4) {
            out.push(Sym(u32_le(c, 0)));
        }
    } else {
        for c in bytes.chunks_exact(4) {
            out.push(map_sym(u32_le(c, 0), sym_remap)?);
        }
    }
    Ok(out.into_boxed_slice())
}

const ORIGIN_KINDS: [Origin; 3] = [Origin::Item, Origin::Import, Origin::Glob];

fn def_tag(def: Def) -> u8 {
    match def {
        Def::Module(_) => 0,
        Def::Item(_) => 1,
        Def::Variant(_, _) => 2,
        Def::Primitive(_) => 3,
    }
}

fn write_def_payload(out: &mut Vec<u8>, def: Def, tables: &Tables<'_>) -> Option<()> {
    match def {
        Def::Module(m) => {
            write_u32(out, tables.local_krate(m.krate as usize)?);
            write_u32(out, m.module);
        }
        Def::Item(i) => {
            write_u32(out, tables.local_krate(i.krate as usize)?);
            write_u32(out, i.file);
            write_u32(out, i.item);
        }
        Def::Variant(i, v) => {
            write_u32(out, tables.local_krate(i.krate as usize)?);
            write_u32(out, i.file);
            write_u32(out, i.item);
            write_u32(out, v);
        }
        Def::Primitive(p) => {
            out.push(PRIMITIVES.iter().position(|&(_, x)| x == p)? as u8);
        }
    }
    Some(())
}

#[inline(always)]
fn map_krate(k: u32, remap: &[u32]) -> Result<u32> {
    remap
        .get(k as usize)
        .copied()
        .context("def crate out of bounds")
}

#[inline(always)]
fn read_def_with_tag(reader: &mut Reader<'_>, tag: u8, remap: &[u32]) -> Result<Def> {
    Ok(match tag {
        0 => {
            let [k0, k1, k2, k3, m0, m1, m2, m3] = reader.take_array::<8>()?;
            Def::Module(ModRef {
                krate: map_krate(u32::from_le_bytes([k0, k1, k2, k3]), remap)?,
                module: u32::from_le_bytes([m0, m1, m2, m3]),
            })
        }
        1 => {
            let [k0, k1, k2, k3, f0, f1, f2, f3, i0, i1, i2, i3] = reader.take_array::<12>()?;
            Def::Item(ItemRef {
                krate: map_krate(u32::from_le_bytes([k0, k1, k2, k3]), remap)?,
                file: u32::from_le_bytes([f0, f1, f2, f3]),
                item: u32::from_le_bytes([i0, i1, i2, i3]),
            })
        }
        2 => {
            let [
                k0,
                k1,
                k2,
                k3,
                f0,
                f1,
                f2,
                f3,
                i0,
                i1,
                i2,
                i3,
                v0,
                v1,
                v2,
                v3,
            ] = reader.take_array::<16>()?;
            Def::Variant(
                ItemRef {
                    krate: map_krate(u32::from_le_bytes([k0, k1, k2, k3]), remap)?,
                    file: u32::from_le_bytes([f0, f1, f2, f3]),
                    item: u32::from_le_bytes([i0, i1, i2, i3]),
                },
                u32::from_le_bytes([v0, v1, v2, v3]),
            )
        }
        3 => Def::Primitive(
            PRIMITIVES
                .get(reader.u8()? as usize)
                .map(|&(_, p)| p)
                .context("invalid primitive tag")?,
        ),
        _ => bail!("invalid def tag"),
    })
}

fn write_def(out: &mut Vec<u8>, def: Def, tables: &Tables<'_>) -> Option<()> {
    out.push(def_tag(def));
    write_def_payload(out, def, tables)
}

#[inline(always)]
fn read_def(reader: &mut Reader<'_>, remap: &[u32]) -> Result<Def> {
    let tag = reader.u8()?;
    read_def_with_tag(reader, tag, remap)
}

fn write_binding(out: &mut Vec<u8>, b: Binding, tables: &Tables<'_>) -> Option<()> {
    let dtag = def_tag(b.def);
    let (vtag, vis_at) = vis_tag(b.vis);
    let otag = ORIGIN_KINDS.iter().position(|&o| o == b.origin)? as u8;
    out.push(dtag | (vtag << 2) | (otag << 5));
    if vtag == 5 {
        write_u32(out, vis_at);
    }
    write_def_payload(out, b.def, tables)
}

#[inline(always)]
fn read_binding(reader: &mut Reader<'_>, remap: &[u32]) -> Result<Binding> {
    let hdr = reader.u8()?;
    let dtag = hdr & 3;
    let vtag = (hdr >> 2) & 7;
    let otag = (hdr >> 5) & 3;
    let vis_at = if vtag == 5 { reader.u32()? } else { 0 };
    let vis = vis_from_tag(vtag, vis_at)?;
    let origin = match otag {
        0 => Origin::Item,
        1 => Origin::Import,
        2 => Origin::Glob,
        _ => bail!("invalid origin tag"),
    };
    let def = read_def_with_tag(reader, dtag, remap)?;
    Ok(Binding { def, vis, origin })
}

pub(super) fn write_names_crate(
    out: &mut Vec<u8>,
    krate: &NamesCrate,
    tables: &mut Tables<'_>,
) -> Option<()> {
    write_usize(out, krate.scopes.len())?;
    let mut scope_ends = Vec::with_capacity(krate.scopes.len());
    let mut scope_buf = Vec::new();
    for scope in &krate.scopes {
        write_usize(&mut scope_buf, scope.names.len())?;
        for (&sym, per) in &scope.names {
            write_u32(&mut scope_buf, tables.sym(sym));
            let mask = u8::from(per.types.is_some())
                | (u8::from(per.values.is_some()) << 1)
                | (u8::from(per.macros.is_some()) << 2);
            scope_buf.push(mask);
            for ns in [Ns::Types, Ns::Values, Ns::Macros] {
                if let Some(binding) = per.get(ns) {
                    write_binding(&mut scope_buf, binding, tables)?;
                }
            }
        }
        scope_ends.push(u32::try_from(scope_buf.len()).ok()?);
    }
    for end in scope_ends {
        write_u32(out, end);
    }
    write_usize(out, scope_buf.len())?;
    out.extend_from_slice(&scope_buf);

    match krate.prelude {
        Some(m) => {
            out.push(1);
            write_u32(out, tables.local_krate(m.krate as usize)?);
            write_u32(out, m.module);
        }
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
    let kept_pending: Vec<&Pending> = krate
        .pending
        .iter()
        .filter(|p| p.name.is_none() || p.name == Some(UNDERSCORE))
        .collect();
    write_usize(out, kept_pending.len())?;
    for p in kept_pending {
        write_u32(out, p.module);
        write_u32(out, p.file);
        write_u32(out, p.import);
        let dollar = match p.dollar {
            Some(d) => Some(tables.local_krate(d as usize)?),
            None => None,
        };
        write_opt_u32(out, dollar);
        write_opt_u32(out, p.name.map(|s| tables.sym(s)));
        match p.stamp {
            Some((def, ver)) => {
                out.push(1);
                write_def(out, def, tables)?;
                write_u32(out, ver);
            }
            None => {
                out.push(0);
                write_usize(out, p.segments.len())?;
                for &seg in &p.segments {
                    write_u32(out, tables.sym(seg));
                }
            }
        }
        out.push(u8::from(p.absolute));
        write_vis(out, p.vis);
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

fn decode_scope(slice: &[u8], remap: &[u32], sym_remap: &[Sym]) -> Result<Scope> {
    let mut reader = Reader::new(slice);
    let name_len = reader.u32()? as usize;
    if name_len == 0 {
        if !reader.is_empty() {
            bail!("trailing bytes in sysroot scope");
        }
        return Ok(Scope {
            names: HashMap::default(),
            log: Vec::new(),
        });
    }
    let mut names = HashMap::with_capacity_and_hasher(name_len, Default::default());
    for _ in 0..name_len {
        let [s0, s1, s2, s3, mask] = reader.take_array::<5>()?;
        let sym = map_sym(u32::from_le_bytes([s0, s1, s2, s3]), sym_remap)?;
        let types = if mask & 1 != 0 {
            Some(read_binding(&mut reader, remap)?)
        } else {
            None
        };
        let values = if mask & 2 != 0 {
            Some(read_binding(&mut reader, remap)?)
        } else {
            None
        };
        let macros = if mask & 4 != 0 {
            Some(read_binding(&mut reader, remap)?)
        } else {
            None
        };
        names.insert(
            sym,
            PerNs {
                types,
                values,
                macros,
            },
        );
    }
    if !reader.is_empty() {
        bail!("trailing bytes in sysroot scope");
    }
    Ok(Scope {
        names,
        log: Vec::new(),
    })
}

pub(super) struct ScopeTable<'a> {
    pub end_bytes: &'a [u8],
    pub scope_bytes: &'a [u8],
    pub count: usize,
    pub total_bytes: usize,
}

pub(super) fn split_names_crate_scopes<'a>(reader: &mut Reader<'a>) -> Result<ScopeTable<'a>> {
    let scope_len = reader.count()?;
    let end_bytes = reader.take(scope_len.checked_mul(4).context("scope table overflow")?)?;
    let scope_byte_len = reader.count()?;
    let scope_bytes = reader.take(scope_byte_len)?;
    let mut prev = 0usize;
    for c in end_bytes.chunks_exact(4) {
        let end = u32_le(c, 0) as usize;
        if end < prev || end > scope_byte_len {
            bail!("non-monotonic sysroot scope offset");
        }
        prev = end;
    }
    if prev != scope_byte_len {
        bail!("sysroot scope table length mismatch");
    }
    Ok(ScopeTable {
        end_bytes,
        scope_bytes,
        count: scope_len,
        total_bytes: scope_byte_len,
    })
}

pub(super) fn decode_scope_range(
    table: &ScopeTable<'_>,
    start: usize,
    end: usize,
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<Vec<Scope>> {
    if start >= end {
        return Ok(Vec::new());
    }
    let mut prev = if start == 0 {
        0
    } else {
        u32_le(table.end_bytes, (start - 1) * 4) as usize
    };
    let mut scopes = Vec::with_capacity(end - start);
    for c in table.end_bytes[start * 4..end * 4].chunks_exact(4) {
        let end_off = u32_le(c, 0) as usize;
        scopes.push(decode_scope(
            &table.scope_bytes[prev..end_off],
            remap,
            sym_remap,
        )?);
        prev = end_off;
    }
    Ok(scopes)
}

fn read_scopes(
    reader: &mut Reader<'_>,
    remap: &[u32],
    sym_remap: &[Sym],
    parallel: bool,
) -> Result<Vec<Scope>> {
    let table = split_names_crate_scopes(reader)?;
    if parallel && table.count >= 16 && table.total_bytes >= 8 * 1024 && crate::jobs() > 1 {
        let mut slices = Vec::with_capacity(table.count);
        let mut prev = 0usize;
        for c in table.end_bytes.chunks_exact(4) {
            let end_off = u32_le(c, 0) as usize;
            slices.push(&table.scope_bytes[prev..end_off]);
            prev = end_off;
        }
        crate::sync::parallel_map(slices, |s| decode_scope(s, remap, sym_remap))
            .into_iter()
            .collect::<Result<Vec<_>>>()
    } else {
        decode_scope_range(&table, 0, table.count, remap, sym_remap)
    }
}

fn read_prelude(reader: &mut Reader<'_>, remap: &[u32]) -> Result<Option<ModRef>> {
    if reader.u8()? == 0 {
        return Ok(None);
    }
    let [k0, k1, k2, k3, m0, m1, m2, m3] = reader.take_array::<8>()?;
    let krate = map_krate(u32::from_le_bytes([k0, k1, k2, k3]), remap)?;
    Ok(Some(ModRef {
        krate,
        module: u32::from_le_bytes([m0, m1, m2, m3]),
    }))
}

fn read_unresolved(reader: &mut Reader<'_>) -> Result<Vec<Unresolved>> {
    let unres_len = reader.count()?;
    if unres_len == 0 {
        return Ok(Vec::new());
    }
    let unres_bytes = reader.take(unres_len.checked_mul(8).context("unresolved overflow")?)?;
    Ok(unres_bytes
        .chunks_exact(8)
        .map(|c| Unresolved {
            file: u32_le(c, 0),
            import: u32_le(c, 4),
        })
        .collect())
}

fn read_externs(
    reader: &mut Reader<'_>,
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<Vec<(Sym, Def)>> {
    let ext_len = reader.count()?;
    if ext_len == 0 {
        return Ok(Vec::new());
    }
    let mut externs = Vec::with_capacity(ext_len);
    for _ in 0..ext_len {
        externs.push((map_sym(reader.u32()?, sym_remap)?, read_def(reader, remap)?));
    }
    Ok(externs)
}

fn read_deps(
    reader: &mut Reader<'_>,
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<Vec<(Sym, usize)>> {
    let dep_len = reader.count()?;
    if dep_len == 0 {
        return Ok(Vec::new());
    }
    let dep_bytes = reader.take(dep_len.checked_mul(8).context("names deps overflow")?)?;
    let mut deps = Vec::with_capacity(dep_len);
    for c in dep_bytes.chunks_exact(8) {
        let sym = map_sym(u32_le(c, 0), sym_remap)?;
        let dep = *remap
            .get(u32_le(c, 4) as usize)
            .context("names dep crate out of bounds")? as usize;
        deps.push((sym, dep));
    }
    Ok(deps)
}

type DecodedPending = (
    Vec<Pending>,
    HashMap<(u32, Sym), Vec<usize>>,
    HashMap<u32, Vec<usize>>,
);

fn read_pending(
    reader: &mut Reader<'_>,
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<DecodedPending> {
    let pend_len = reader.count()?;
    if pend_len == 0 {
        return Ok((Vec::new(), HashMap::default(), HashMap::default()));
    }
    let mut pending = Vec::with_capacity(pend_len);
    let mut named: HashMap<(u32, Sym), Vec<usize>> = HashMap::default();
    let mut globs: HashMap<u32, Vec<usize>> = HashMap::default();
    for index in 0..pend_len {
        let b = reader.take_array::<20>()?;
        let module = u32_le(&b, 0);
        let file = u32_le(&b, 4);
        let import = u32_le(&b, 8);
        let dollar = match opt_u32_le(&b, 12) {
            Some(d) => Some(
                *remap
                    .get(d as usize)
                    .context("pending dollar crate out of bounds")?,
            ),
            None => None,
        };
        let name = opt_u32_le(&b, 16)
            .map(|id| map_sym(id, sym_remap))
            .transpose()?;
        let (stamp, segments) = if reader.u8()? != 0 {
            let def = read_def(reader, remap)?;
            let ver = reader.u32()?;
            (Some((def, ver)), Box::default())
        } else {
            (None, read_syms(reader, sym_remap)?)
        };
        let absolute = reader.u8()? != 0;
        let vis = read_vis(reader)?;
        let active = reader.u8()? != 0;
        match name {
            Some(sym) => named.entry((module, sym)).or_default().push(index),
            None => globs.entry(module).or_default().push(index),
        }
        pending.push(Pending {
            module,
            file,
            import,
            segments,
            absolute,
            dollar,
            name,
            vis,
            stamp,
            guard: Default::default(),
            active,
        });
    }
    Ok((pending, named, globs))
}

fn read_enums(
    reader: &mut Reader<'_>,
    sym_remap: &[Sym],
) -> Result<HashMap<(u32, u32), Box<[Sym]>>> {
    let enum_len = reader.count()?;
    if enum_len == 0 {
        return Ok(HashMap::default());
    }
    let mut enums = HashMap::with_capacity_and_hasher(enum_len, Default::default());
    for _ in 0..enum_len {
        let [f0, f1, f2, f3, i0, i1, i2, i3] = reader.take_array::<8>()?;
        let key = (
            u32::from_le_bytes([f0, f1, f2, f3]),
            u32::from_le_bytes([i0, i1, i2, i3]),
        );
        let variants = read_syms(reader, sym_remap)?;
        enums.insert(key, variants);
    }
    Ok(enums)
}

pub(super) fn read_names_crate_body(
    reader: &mut Reader<'_>,
    scopes: Vec<Scope>,
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<NamesCrate> {
    let prelude = read_prelude(reader, remap)?;
    let unresolved = read_unresolved(reader)?;
    let externs = read_externs(reader, remap, sym_remap)?;
    let deps = read_deps(reader, remap, sym_remap)?;
    let edition_2015 = reader.u8()? != 0;
    let edition = reader.str()?.to_owned();
    let collected = reader.u32s_map(|v| (v != u32::MAX).then_some(v as usize))?;
    let (pending, named, globs) = read_pending(reader, remap, sym_remap)?;
    let enums = read_enums(reader, sym_remap)?;
    let eager = reader.u8()? != 0;
    let mut out = NamesCrate::default();
    out.scopes = scopes;
    out.prelude = prelude;
    out.unresolved = unresolved;
    out.externs = externs;
    out.deps = deps;
    out.edition_2015 = edition_2015;
    out.edition = edition;
    out.collected = collected;
    out.set_pending_with_indices(pending, named, globs);
    out.enums = enums;
    out.eager = eager;
    out.settled = true;
    Ok(out)
}

pub(super) fn read_names_crate(
    reader: &mut Reader<'_>,
    remap: &[u32],
    sym_remap: &[Sym],
    parallel_scopes: bool,
) -> Result<NamesCrate> {
    let scopes = read_scopes(reader, remap, sym_remap, parallel_scopes)?;
    read_names_crate_body(reader, scopes, remap, sym_remap)
}
