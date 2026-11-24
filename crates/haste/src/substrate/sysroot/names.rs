use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::names::{
    Binding, Crate as NamesCrate, Def, ItemRef, ModRef, Ns, Origin, PRIMITIVES, PerNs, Scope, Sym,
    UNDERSCORE, Unresolved, collect::Pending,
};
use crate::sema::summary::codec::{Cursor, write_u32};

use super::tokens::{
    read_str, read_u32s_map, read_vis, u32_le, vis_from_tag, vis_tag, write_opt_u32, write_str,
    write_usize, write_vis,
};
use super::wire::Tables;

const ORIGIN_KINDS: [Origin; 3] = [Origin::Item, Origin::Import, Origin::Glob];
const DEF_BYTES: [usize; 4] = [8, 12, 16, 1];

struct SliceReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    #[inline]
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("slice overflow")?;
        let slice = self.data.get(self.pos..end).context("unexpected eof")?;
        self.pos = end;
        Ok(slice)
    }

    #[inline]
    fn u8(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).context("unexpected eof")?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32_le(s, 0))
    }
}

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

#[inline]
fn decode_def_bytes(tag: u8, b: &[u8], remap: &[u32]) -> Result<Def> {
    let map_krate = |k: u32| {
        remap
            .get(k as usize)
            .copied()
            .context("def crate out of bounds")
    };
    Ok(match tag {
        0 => Def::Module(ModRef {
            krate: map_krate(u32_le(b, 0))?,
            module: u32_le(b, 4),
        }),
        1 => Def::Item(ItemRef {
            krate: map_krate(u32_le(b, 0))?,
            file: u32_le(b, 4),
            item: u32_le(b, 8),
        }),
        2 => Def::Variant(
            ItemRef {
                krate: map_krate(u32_le(b, 0))?,
                file: u32_le(b, 4),
                item: u32_le(b, 8),
            },
            u32_le(b, 12),
        ),
        3 => Def::Primitive(
            PRIMITIVES
                .get(b[0] as usize)
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

fn read_def(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Def> {
    let tag = cur.u8()?;
    let len = *DEF_BYTES.get(tag as usize).context("invalid def tag")?;
    decode_def_bytes(tag, cur.take(len)?, remap)
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

#[inline]
fn read_binding(reader: &mut SliceReader<'_>, remap: &[u32]) -> Result<Binding> {
    let hdr = reader.u8()?;
    let dtag = hdr & 3;
    let vtag = (hdr >> 2) & 7;
    let otag = (hdr >> 5) & 3;
    let vis_len = if vtag == 5 { 4 } else { 0 };
    let def_len = DEF_BYTES[dtag as usize];
    let b = reader.take(vis_len + def_len)?;
    let vis_at = if vis_len != 0 { u32_le(b, 0) } else { 0 };
    let vis = vis_from_tag(vtag, vis_at)?;
    let origin = ORIGIN_KINDS
        .get(otag as usize)
        .copied()
        .context("invalid origin tag")?;
    let def = decode_def_bytes(dtag, &b[vis_len..], remap)?;
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
        write_usize(out, p.segments.len())?;
        for &seg in &p.segments {
            write_u32(out, tables.sym(seg));
        }
        out.push(u8::from(p.absolute));
        write_vis(out, p.vis);
        match p.stamp {
            Some((def, ver)) => {
                out.push(1);
                write_def(out, def, tables)?;
                write_u32(out, ver);
            }
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

fn decode_scope(slice: &[u8], remap: &[u32]) -> Result<Scope> {
    let mut reader = SliceReader::new(slice);
    let name_len = reader.u32()? as usize;
    let mut names = HashMap::with_capacity_and_hasher(name_len, Default::default());
    for _ in 0..name_len {
        let b = reader.take(5)?;
        let sym = Sym(u32_le(b, 0));
        let mask = b[4];
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
    if reader.pos != slice.len() {
        bail!("trailing bytes in sysroot scope");
    }
    Ok(Scope {
        names,
        log: Vec::new(),
    })
}

fn read_scopes(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<Scope>> {
    let scope_len = cur.count()?;
    let end_bytes = cur.take(scope_len.checked_mul(4).context("scope table overflow")?)?;
    let mut ranges = Vec::with_capacity(scope_len);
    let mut prev = 0usize;
    for c in end_bytes.chunks_exact(4) {
        let end = u32_le(c, 0) as usize;
        if end < prev {
            bail!("non-monotonic sysroot scope offset");
        }
        ranges.push((prev, end));
        prev = end;
    }
    let scope_byte_len = cur.count()?;
    if prev != scope_byte_len {
        bail!("sysroot scope table length mismatch");
    }
    let scope_bytes = cur.take(scope_byte_len)?;
    if scope_len >= 64 && scope_byte_len >= 64 * 1024 && crate::jobs() > 1 {
        let slices: Vec<&[u8]> = ranges
            .into_iter()
            .map(|(s, e)| &scope_bytes[s..e])
            .collect();
        crate::sync::parallel_map(slices, |s| decode_scope(s, remap))
            .into_iter()
            .collect::<Result<Vec<_>>>()
    } else {
        ranges
            .into_iter()
            .map(|(s, e)| decode_scope(&scope_bytes[s..e], remap))
            .collect::<Result<Vec<_>>>()
    }
}

fn read_prelude(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Option<ModRef>> {
    if cur.u8()? == 0 {
        return Ok(None);
    }
    let b = cur.take(8)?;
    let krate = *remap
        .get(u32_le(b, 0) as usize)
        .context("prelude crate out of bounds")?;
    Ok(Some(ModRef {
        krate,
        module: u32_le(b, 4),
    }))
}

fn read_unresolved(cur: &mut Cursor<'_>) -> Result<Vec<Unresolved>> {
    let unres_len = cur.count()?;
    let unres_bytes = cur.take(unres_len.checked_mul(8).context("unresolved overflow")?)?;
    Ok(unres_bytes
        .chunks_exact(8)
        .map(|c| Unresolved {
            file: u32_le(c, 0),
            import: u32_le(c, 4),
        })
        .collect())
}

fn read_externs(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<(Sym, Def)>> {
    let ext_len = cur.count()?;
    let mut externs = Vec::with_capacity(ext_len);
    for _ in 0..ext_len {
        externs.push((Sym(cur.u32()?), read_def(cur, remap)?));
    }
    Ok(externs)
}

fn read_deps(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<(Sym, usize)>> {
    let dep_len = cur.count()?;
    let dep_bytes = cur.take(dep_len.checked_mul(8).context("names deps overflow")?)?;
    let mut deps = Vec::with_capacity(dep_len);
    for c in dep_bytes.chunks_exact(8) {
        let sym = Sym(u32_le(c, 0));
        let dep = *remap
            .get(u32_le(c, 4) as usize)
            .context("names dep crate out of bounds")? as usize;
        deps.push((sym, dep));
    }
    Ok(deps)
}

fn read_pending(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<Pending>> {
    let pend_len = cur.count()?;
    let mut pending = Vec::with_capacity(pend_len);
    let opt_at = |b: &[u8], at: usize| {
        let v = u32_le(b, at);
        (v != u32::MAX).then_some(v)
    };
    for _ in 0..pend_len {
        let b = cur.take(20)?;
        let module = u32_le(b, 0);
        let file = u32_le(b, 4);
        let import = u32_le(b, 8);
        let dollar = match opt_at(b, 12) {
            Some(d) => Some(
                *remap
                    .get(d as usize)
                    .context("pending dollar crate out of bounds")?,
            ),
            None => None,
        };
        let name = opt_at(b, 16).map(Sym);
        let segments = read_u32s_map(cur, Sym)?.into_boxed_slice();
        let absolute = cur.u8()? != 0;
        let vis = read_vis(cur)?;
        let stamp = if cur.u8()? != 0 {
            Some((read_def(cur, remap)?, cur.u32()?))
        } else {
            None
        };
        let active = cur.u8()? != 0;
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
    Ok(pending)
}

fn read_enums(cur: &mut Cursor<'_>) -> Result<HashMap<(u32, u32), Box<[Sym]>>> {
    let enum_len = cur.count()?;
    let mut enums = HashMap::with_capacity_and_hasher(enum_len, Default::default());
    for _ in 0..enum_len {
        let b = cur.take(8)?;
        let key = (u32_le(b, 0), u32_le(b, 4));
        let variants = read_u32s_map(cur, Sym)?.into_boxed_slice();
        enums.insert(key, variants);
    }
    Ok(enums)
}

pub(super) fn read_names_crate(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<NamesCrate> {
    let scopes = read_scopes(cur, remap)?;
    let prelude = read_prelude(cur, remap)?;
    let unresolved = read_unresolved(cur)?;
    let externs = read_externs(cur, remap)?;
    let deps = read_deps(cur, remap)?;
    let edition_2015 = cur.u8()? != 0;
    let edition = read_str(cur)?.to_owned();
    let collected = read_u32s_map(cur, |v| (v != u32::MAX).then_some(v as usize))?;
    let pending = read_pending(cur, remap)?;
    let enums = read_enums(cur)?;
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
