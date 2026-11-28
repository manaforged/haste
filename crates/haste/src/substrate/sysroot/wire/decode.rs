use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::names::{Crate as NamesCrate, Sym, Symbols};
use crate::outline::Outline;
use crate::outline::source::{DiskSource, Outputs, Source, SourceText, SourceTokens};
use crate::outline::tree::{Crate as OutlineCrate, File, Macro, Textual};
use crate::sema::impls::Index as ImplIndex;
use crate::sema::summary::codec::Cursor;
use crate::substrate::Backing;

use super::super::impls::read_impl_index;
use super::super::names::read_names_crate;
use super::super::outline::read_outline_crate;
use super::super::tokens::{
    Reader, frag_from_tag, read_items, read_marks_reader, read_origins, read_rules, read_str,
    u32_le,
};
use super::super::{EnvReads, HEADER_LEN, match_crate_list_wire};
use super::batch::decode_snapshot_parallel;
use super::{FIXED_SYM_COUNT, opt_u32_le};

pub(in crate::substrate) type DecodedSnapshot = (
    Vec<(usize, OutlineCrate)>,
    Option<Symbols>,
    Vec<(usize, NamesCrate)>,
    HashMap<u32, Arc<ImplIndex>>,
    Vec<EnvReads>,
);

pub(in crate::substrate) struct DecodeSpec<'a> {
    pub magic: [u8; 8],
    pub version: u32,
    pub key: u64,
    pub cached_crates: &'a [usize],
    pub ref_crates: &'a [usize],
    pub scope_bit: u32,
}

pub(super) struct WireSections<'a> {
    pub sym_count: usize,
    pub sym_ends: &'a [u8],
    pub sym_blob: &'a [u8],
    pub file_slices: Vec<(usize, &'a [u8], Option<u32>)>,
    pub macro_slices: Vec<&'a [u8]>,
    pub textual_raw: Vec<(&'a str, u32, Option<u32>)>,
    pub outline_slices: Vec<&'a [u8]>,
    pub name_slices: Vec<&'a [u8]>,
    pub impl_slices: Vec<&'a [u8]>,
}

fn read_slices<'a>(reader: &mut Reader<'a>, count: usize) -> Result<Vec<&'a [u8]>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let table_bytes = count.checked_mul(4).context("slice table overflow")?;
    let table = reader.take(table_bytes)?;
    let last_end = u32_le(table, (count - 1) * 4) as usize;
    let blob = reader.take(last_end)?;
    let mut out = Vec::with_capacity(count);
    let mut start = 0usize;
    for c in table.chunks_exact(4) {
        let end = u32_le(c, 0) as usize;
        if end < start || end > blob.len() {
            bail!("non-monotonic sysroot slice offset");
        }
        out.push(&blob[start..end]);
        start = end;
    }
    Ok(out)
}

fn read_file_slices<'a>(
    reader: &mut Reader<'a>,
    data_len: usize,
    count: usize,
) -> Result<Vec<(usize, &'a [u8], Option<u32>)>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let table_bytes = count.checked_mul(8).context("file table overflow")?;
    let table = reader.take(table_bytes)?;
    let last_end = u32_le(table, (count - 1) * 8) as usize;
    let blob_off = data_len - reader.len();
    let blob = reader.take(last_end)?;
    let mut out = Vec::with_capacity(count);
    let mut start = 0usize;
    for c in table.chunks_exact(8) {
        let end = u32_le(c, 0) as usize;
        if end < start || end > blob.len() {
            bail!("non-monotonic sysroot file offset");
        }
        out.push((blob_off + start, &blob[start..end], opt_u32_le(c, 4)));
        start = end;
    }
    Ok(out)
}

fn decode_symbols_with(
    sym_ends: &[u8],
    sym_blob: &[u8],
    existing: Option<&Symbols>,
) -> Result<(Option<Symbols>, Vec<Sym>)> {
    let text = std::str::from_utf8(sym_blob).context("invalid snapshot symbol utf-8")?;
    let owned = existing.is_none().then(Symbols::default);
    let syms = existing.or(owned.as_ref()).context("missing symbols")?;
    let mut sym_remap = if existing.is_some() {
        Vec::with_capacity(sym_ends.len() / 4)
    } else {
        Vec::new()
    };
    let mut prev = 0usize;
    for (idx, c) in sym_ends.chunks_exact(4).enumerate() {
        let end = u32_le(c, 0) as usize;
        if end < prev || end > text.len() || !text.is_char_boundary(end) {
            bail!("invalid snapshot symbol boundary");
        }
        let interned = syms.intern(&text[prev..end]);
        if existing.is_some() {
            sym_remap.push(interned);
        } else {
            let expected = FIXED_SYM_COUNT + idx as u32;
            if interned.0 != expected {
                bail!("snapshot symbol table mismatch");
            }
        }
        prev = end;
    }
    if prev != text.len() {
        bail!("trailing bytes in snapshot symbol blob");
    }
    Ok((owned, sym_remap))
}

fn decode_files_seq(
    backing: &Arc<Backing>,
    file_slices: Vec<(usize, &[u8], Option<u32>)>,
    remap: &[u32],
    scope_bit: u32,
) -> Result<Vec<Arc<File>>> {
    let mut files: Vec<Arc<File>> = Vec::with_capacity(file_slices.len());
    for (idx, (file_off, slice, base_id)) in file_slices.into_iter().enumerate() {
        let base = match base_id {
            Some(id) => {
                if id as usize >= idx {
                    bail!("invalid snapshot file base dependency");
                }
                Some(Arc::clone(
                    files.get(id as usize).context("base file out of bounds")?,
                ))
            }
            None => None,
        };
        let mut freader = Reader::new(slice);
        let file = read_file(
            &mut freader,
            slice.len(),
            backing,
            file_off,
            base,
            remap,
            scope_bit,
        )?;
        if !freader.is_empty() {
            bail!("trailing bytes in snapshot file");
        }
        files.push(Arc::new(file));
    }
    Ok(files)
}

fn decode_macros_seq(
    macro_slices: Vec<&[u8]>,
    files: &[Arc<File>],
    remap: &[u32],
) -> Result<Vec<Arc<Macro>>> {
    let mut macros = Vec::with_capacity(macro_slices.len());
    for slice in macro_slices {
        let mut mreader = Reader::new(slice);
        let file = Arc::clone(
            files
                .get(mreader.u32()? as usize)
                .context("macro file out of bounds")?,
        );
        let krate = *remap
            .get(mreader.u32()? as usize)
            .context("macro crate out of bounds")? as usize;
        let rules = read_rules(&mut mreader, remap)?;
        if !mreader.is_empty() {
            bail!("trailing bytes in snapshot macro");
        }
        macros.push(Arc::new(Macro { file, rules, krate }));
    }
    Ok(macros)
}

fn decode_crates_seq(
    outline_slices: Vec<&[u8]>,
    name_slices: Vec<&[u8]>,
    impl_slices: Vec<&[u8]>,
    files: &[Arc<File>],
    macros: &[Arc<Macro>],
    textuals: &[Arc<Textual>],
    local_crates: &[usize],
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<(
    Vec<(usize, OutlineCrate)>,
    Vec<(usize, NamesCrate)>,
    HashMap<u32, Arc<ImplIndex>>,
)> {
    let crate_count = local_crates.len();
    let parallel_scopes = crate_count < 4;
    let mut outline_crates = Vec::with_capacity(crate_count);
    for (&k, slice) in local_crates.iter().zip(outline_slices) {
        let mut creader = Reader::new(slice);
        let krate = read_outline_crate(&mut creader, files, macros, textuals, remap)?;
        if !creader.is_empty() {
            bail!("trailing bytes in snapshot outline crate");
        }
        outline_crates.push((k, krate));
    }
    let mut name_crates = Vec::with_capacity(crate_count);
    for (&k, slice) in local_crates.iter().zip(name_slices) {
        let mut creader = Reader::new(slice);
        let krate = read_names_crate(&mut creader, remap, sym_remap, parallel_scopes)?;
        if !creader.is_empty() {
            bail!("trailing bytes in snapshot names crate");
        }
        name_crates.push((k, krate));
    }
    let mut heads = HashMap::default();
    for (&k, slice) in local_crates.iter().zip(impl_slices) {
        let mut creader = Reader::new(slice);
        let index = read_impl_index(&mut creader, k as u32, remap)?;
        if !creader.is_empty() {
            bail!("trailing bytes in snapshot impl index");
        }
        if let Some(index) = index {
            heads.insert(k as u32, index);
        }
    }
    Ok((outline_crates, name_crates, heads))
}

fn read_env_reads_list(cur: &mut Cursor<'_>) -> Result<Vec<EnvReads>> {
    let crate_count = cur.count()?;
    let mut all = Vec::with_capacity(crate_count);
    for _ in 0..crate_count {
        let len = cur.count()?;
        let mut reads = Vec::with_capacity(len);
        for _ in 0..len {
            let name = read_str(cur)?.to_owned();
            let val = match cur.u8()? {
                0 => None,
                1 => Some(read_str(cur)?.to_owned()),
                _ => bail!("invalid env read tag"),
            };
            reads.push((name, val));
        }
        all.push(reads);
    }
    Ok(all)
}

pub(super) fn read_header_and_match(
    cur: &mut Cursor<'_>,
    data: &[u8],
    spec: &DecodeSpec<'_>,
    workspace: &Workspace,
    outline: &Outline,
) -> Result<Option<Vec<EnvReads>>> {
    if cur.take(8)? != spec.magic {
        bail!("invalid snapshot cache magic");
    }
    if cur.u32()? != spec.version {
        return Ok(None);
    }
    let cached_count = cur.count()?;
    let ref_count = cur.count()?;
    if cached_count != spec.cached_crates.len()
        || ref_count != spec.ref_crates.len()
        || spec.ref_crates.get(..cached_count) != Some(spec.cached_crates)
    {
        return Ok(None);
    }
    if cur.u64()? != spec.key {
        return Ok(None);
    }
    let expected_len = cur.u64()?;
    if expected_len != data.len() as u64
        || data.len() < HEADER_LEN + spec.magic.len()
        || &data[data.len() - spec.magic.len()..] != spec.magic
    {
        return Ok(None);
    }
    if !match_crate_list_wire(cur, workspace, outline, spec.ref_crates)? {
        return Ok(None);
    }
    let env_reads = read_env_reads_list(cur)?;
    for reads in &env_reads {
        for (name, expected) in reads {
            if crate::process_env_var(name) != *expected {
                return Ok(None);
            }
        }
    }
    Ok(Some(env_reads))
}

pub(super) fn read_sections<'a>(
    reader: &mut Reader<'a>,
    data_len: usize,
    crate_count: usize,
    magic: [u8; 8],
) -> Result<WireSections<'a>> {
    let sym_count = reader.count()?;
    let sym_blob_len = reader.count()?;
    let sym_ends = reader.take(sym_count.checked_mul(4).context("symbol ends overflow")?)?;
    let sym_blob = reader.take(sym_blob_len)?;
    let file_count = reader.count()?;
    let file_slices = read_file_slices(reader, data_len, file_count)?;
    let macro_count = reader.count()?;
    let macro_slices = read_slices(reader, macro_count)?;
    let textual_count = reader.count()?;
    let mut textual_raw = Vec::with_capacity(textual_count);
    for _ in 0..textual_count {
        textual_raw.push((reader.str()?, reader.u32()?, reader.opt_u32()?));
    }
    let outline_slices = read_slices(reader, crate_count)?;
    let name_slices = read_slices(reader, crate_count)?;
    let impl_slices = read_slices(reader, crate_count)?;
    if reader.take(magic.len())? != magic || !reader.is_empty() {
        bail!("trailing bytes in snapshot cache");
    }
    Ok(WireSections {
        sym_count,
        sym_ends,
        sym_blob,
        file_slices,
        macro_slices,
        textual_raw,
        outline_slices,
        name_slices,
        impl_slices,
    })
}

pub(super) fn decode_textuals(
    textual_raw: &[(&str, u32, Option<u32>)],
    macros: &[Arc<Macro>],
) -> Result<Vec<Arc<Textual>>> {
    let mut textuals: Vec<Arc<Textual>> = Vec::with_capacity(textual_raw.len());
    for &(name, def_id, next_id) in textual_raw {
        let def = Arc::clone(
            macros
                .get(def_id as usize)
                .context("textual macro out of bounds")?,
        );
        let next = match next_id {
            Some(id) => Some(Arc::clone(
                textuals
                    .get(id as usize)
                    .context("textual next out of bounds")?,
            )),
            None => None,
        };
        textuals.push(Arc::new(Textual {
            name: name.to_owned(),
            def,
            next,
        }));
    }
    Ok(textuals)
}

pub(in super::super) fn decode_snapshot(
    backing: &Arc<Backing>,
    spec: &DecodeSpec<'_>,
    existing_symbols: Option<&Symbols>,
    workspace: &Workspace,
    outline: &Outline,
) -> Result<Option<DecodedSnapshot>> {
    let data = backing.as_slice();
    let mut cur = Cursor::new(data);
    let Some(env_reads) = read_header_and_match(&mut cur, data, spec, workspace, outline)? else {
        return Ok(None);
    };
    let remap: Vec<u32> = spec.ref_crates.iter().map(|&k| k as u32).collect();
    let mut sreader = Reader::new(&data[cur.pos..]);
    let sections = read_sections(
        &mut sreader,
        data.len(),
        spec.cached_crates.len(),
        spec.magic,
    )?;
    if crate::jobs() > 1 && sections.sym_count >= 2048 {
        let owned = existing_symbols.is_none().then(Symbols::default);
        let syms = existing_symbols
            .or(owned.as_ref())
            .context("missing symbols")?;
        let (outline_crates, _, name_crates, heads, env_reads) = decode_snapshot_parallel(
            backing,
            spec.cached_crates,
            &remap,
            spec.scope_bit,
            env_reads,
            sections,
            syms,
        )?;
        return Ok(Some((outline_crates, owned, name_crates, heads, env_reads)));
    }
    let (owned_symbols, sym_remap) =
        decode_symbols_with(sections.sym_ends, sections.sym_blob, existing_symbols)?;
    let files = decode_files_seq(backing, sections.file_slices, &remap, spec.scope_bit)?;
    let macros = decode_macros_seq(sections.macro_slices, &files, &remap)?;
    let textuals = decode_textuals(&sections.textual_raw, &macros)?;
    let (outline_crates, name_crates, heads) = decode_crates_seq(
        sections.outline_slices,
        sections.name_slices,
        sections.impl_slices,
        &files,
        &macros,
        &textuals,
        spec.cached_crates,
        &remap,
        &sym_remap,
    )?;
    Ok(Some((
        outline_crates,
        owned_symbols,
        name_crates,
        heads,
        env_reads,
    )))
}

pub(super) fn read_file(
    reader: &mut Reader<'_>,
    slice_len: usize,
    backing: &Arc<Backing>,
    file_off: usize,
    base: Option<Arc<File>>,
    remap: &[u32],
    scope_bit: u32,
) -> Result<File> {
    let path = reader.path()?;
    let text = match reader.u8()? {
        1 => {
            let text_len = reader.u32()? as usize;
            let tail = Box::<str>::from(reader.str()?);
            let disk = Arc::new(DiskSource::new(path.clone(), text_len, tail));
            SourceText::disk(disk)
        }
        0 => {
            let text_len = reader.count()?;
            let text_start = file_off + (slice_len - reader.len());
            reader.take(text_len)?;
            let text_end = file_off + (slice_len - reader.len());
            SourceText::mapped_trusted(Arc::clone(backing), text_start, text_end)?
        }
        _ => bail!("invalid sysroot file storage tag"),
    };
    let tok_count = reader.u32()? as usize;
    let tok_bytes = reader.count()?;
    let tok_start = file_off + (slice_len - reader.len());
    reader.take(tok_bytes)?;
    let tok_end = file_off + (slice_len - reader.len());
    let tokens = SourceTokens::compact(Arc::clone(backing), tok_start, tok_end, tok_count)?;
    let marks = read_marks_reader(reader, remap, scope_bit)?;
    let opaque_len = reader.count()?;
    let opaque = if opaque_len == 0 {
        Vec::new()
    } else {
        let opaque_bytes = reader.take(opaque_len.checked_mul(9).context("opaque overflow")?)?;
        let mut opaque = Vec::with_capacity(opaque_len);
        for c in opaque_bytes.chunks_exact(9) {
            opaque.push((u32_le(c, 0), u32_le(c, 4), frag_from_tag(c[8])?));
        }
        opaque
    };
    let offset = reader.u32()?;
    let origins = read_origins(reader)?;
    let items = read_items(reader)?;
    let expanded = reader.u8()? != 0;
    let expand_origin = reader.u64()?;
    let source = Source {
        path,
        text,
        tokens,
        marks,
        opaque,
        base,
        offset,
        outputs: Outputs::default(),
        origins,
        line_starts: std::sync::OnceLock::new(),
    };
    let mut file = File::new(source, items, expanded);
    file.expand_origin = expand_origin;
    Ok(file)
}
