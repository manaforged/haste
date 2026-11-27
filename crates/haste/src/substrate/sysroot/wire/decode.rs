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
    frag_from_tag, read_items, read_marks, read_opt_u32, read_origins, read_path, read_rules,
    read_str, u32_le,
};
use super::super::{CrateIdentity, EnvReads, HEADER_LEN, match_crate_list};
use super::{FIXED_SYM_COUNT, opt_u32_le};

type DecodedCrates = (
    Vec<(usize, OutlineCrate)>,
    Vec<(usize, NamesCrate)>,
    HashMap<u32, Arc<ImplIndex>>,
);

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

struct WireSections<'a> {
    sym_count: usize,
    sym_ends: &'a [u8],
    sym_blob: &'a [u8],
    file_slices: Vec<(usize, &'a [u8], Option<u32>)>,
    macro_slices: Vec<&'a [u8]>,
    textual_raw: Vec<(String, u32, Option<u32>)>,
    outline_slices: Vec<&'a [u8]>,
    name_slices: Vec<&'a [u8]>,
    impl_slices: Vec<&'a [u8]>,
}

fn read_slices<'a>(cur: &mut Cursor<'a>, count: usize) -> Result<Vec<&'a [u8]>> {
    let table_bytes = count.checked_mul(4).context("slice table overflow")?;
    let table = cur.take(table_bytes)?;
    let mut ranges = Vec::with_capacity(count);
    let mut prev = 0usize;
    for c in table.chunks_exact(4) {
        let end = u32_le(c, 0) as usize;
        if end < prev {
            bail!("non-monotonic sysroot slice offset");
        }
        ranges.push((prev, end));
        prev = end;
    }
    let blob = cur.take(prev)?;
    Ok(ranges.into_iter().map(|(s, e)| &blob[s..e]).collect())
}

fn read_file_slices<'a>(
    cur: &mut Cursor<'a>,
    count: usize,
) -> Result<Vec<(usize, &'a [u8], Option<u32>)>> {
    let table_bytes = count.checked_mul(8).context("file table overflow")?;
    let table = cur.take(table_bytes)?;
    let mut entries = Vec::with_capacity(count);
    let mut prev = 0usize;
    for c in table.chunks_exact(8) {
        let end = u32_le(c, 0) as usize;
        if end < prev {
            bail!("non-monotonic sysroot file offset");
        }
        entries.push((prev, end, opt_u32_le(c, 4)));
        prev = end;
    }
    let blob_off = cur.pos;
    let blob = cur.take(prev)?;
    Ok(entries
        .into_iter()
        .map(|(s, e, base)| (blob_off + s, &blob[s..e], base))
        .collect())
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

fn decode_files(
    backing: &Arc<Backing>,
    file_slices: Vec<(usize, &[u8], Option<u32>)>,
    remap: &[u32],
    scope_bit: u32,
) -> Result<Vec<Arc<File>>> {
    let file_count = file_slices.len();
    let mut depths = vec![0usize; file_count];
    let mut max_depth = 0usize;
    for (idx, &(_, _, base)) in file_slices.iter().enumerate() {
        if let Some(base_id) = base {
            let base_idx = base_id as usize;
            if base_idx >= idx {
                bail!("invalid snapshot file base dependency");
            }
            let d = depths[base_idx] + 1;
            depths[idx] = d;
            max_depth = max_depth.max(d);
        }
    }
    let mut levels: Vec<Vec<(usize, usize, &[u8], Option<u32>)>> =
        (0..=max_depth).map(|_| Vec::new()).collect();
    for (idx, (file_off, slice, base)) in file_slices.into_iter().enumerate() {
        levels[depths[idx]].push((idx, file_off, slice, base));
    }
    let mut file_slots: Vec<Option<Arc<File>>> = vec![None; file_count];
    for mut level in levels {
        if level.is_empty() {
            continue;
        }
        level.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
        let slots_ref: &[Option<Arc<File>>] = &file_slots;
        let decode_one = |(idx, file_off, slice, base_id): (usize, usize, &[u8], Option<u32>)| -> Result<(usize, Arc<File>)> {
            let base = match base_id {
                Some(id) => Some(Arc::clone(
                    slots_ref[id as usize]
                        .as_ref()
                        .context("base file out of bounds")?,
                )),
                None => None,
            };
            let mut fcur = Cursor::new(slice);
            let file = read_file(&mut fcur, backing, file_off, base, remap, scope_bit)?;
            if fcur.pos != slice.len() {
                bail!("trailing bytes in snapshot file");
            }
            Ok((idx, Arc::new(file)))
        };
        let total_bytes: usize = level.iter().map(|(_, _, s, _)| s.len()).sum();
        let decoded = if level.len() >= 4 || (level.len() >= 2 && total_bytes >= 32 * 1024) {
            crate::sync::parallel_map(level, decode_one)
        } else {
            level.into_iter().map(decode_one).collect()
        };
        for item in decoded {
            let (idx, file) = item?;
            file_slots[idx] = Some(file);
        }
    }
    file_slots
        .into_iter()
        .map(|f| f.context("missing decoded file slot"))
        .collect()
}

fn decode_macros(
    macro_slices: Vec<&[u8]>,
    files: &[Arc<File>],
    remap: &[u32],
) -> Result<Vec<Arc<Macro>>> {
    let mut tasks: Vec<(usize, &[u8])> = macro_slices.into_iter().enumerate().collect();
    tasks.sort_unstable_by(|a, b| b.1.len().cmp(&a.1.len()));
    let decode_one = |(idx, slice): (usize, &[u8])| -> Result<(usize, Arc<Macro>)> {
        let mut mcur = Cursor::new(slice);
        let file = Arc::clone(
            files
                .get(mcur.u32()? as usize)
                .context("macro file out of bounds")?,
        );
        let krate = *remap
            .get(mcur.u32()? as usize)
            .context("macro crate out of bounds")? as usize;
        let rules = read_rules(&mut mcur, remap)?;
        if mcur.pos != slice.len() {
            bail!("trailing bytes in snapshot macro");
        }
        Ok((idx, Arc::new(Macro { file, rules, krate })))
    };
    let count = tasks.len();
    let decoded = if count >= 16 {
        crate::sync::parallel_map(tasks, decode_one)
    } else {
        tasks.into_iter().map(decode_one).collect()
    };
    let mut slots: Vec<Option<Arc<Macro>>> = vec![None; count];
    for item in decoded {
        let (idx, mac) = item?;
        slots[idx] = Some(mac);
    }
    slots
        .into_iter()
        .map(|m| m.context("missing decoded macro slot"))
        .collect()
}

fn decode_crates(
    outline_slices: Vec<&[u8]>,
    name_slices: Vec<&[u8]>,
    impl_slices: Vec<&[u8]>,
    files: &[Arc<File>],
    macros: &[Arc<Macro>],
    textuals: &[Arc<Textual>],
    local_crates: &[usize],
    remap: &[u32],
    sym_remap: &[Sym],
) -> Result<DecodedCrates> {
    enum CrateTask<'a> {
        Outline(usize, &'a [u8]),
        Names(usize, &'a [u8]),
        Impls(usize, &'a [u8]),
    }
    enum CrateOut {
        Outline(usize, OutlineCrate),
        Names(usize, NamesCrate),
        Impls(usize, Option<Arc<ImplIndex>>),
    }
    let crate_count = local_crates.len();
    let mut tasks = Vec::with_capacity(crate_count * 3);
    for (pos, slice) in outline_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Outline(pos, slice));
    }
    for (pos, slice) in name_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Names(pos, slice));
    }
    for (pos, slice) in impl_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Impls(pos, slice));
    }
    tasks.sort_unstable_by(|a, b| {
        let la = match a {
            CrateTask::Outline(_, s) | CrateTask::Names(_, s) | CrateTask::Impls(_, s) => s.len(),
        };
        let lb = match b {
            CrateTask::Outline(_, s) | CrateTask::Names(_, s) | CrateTask::Impls(_, s) => s.len(),
        };
        lb.cmp(&la)
    });
    let decoded = crate::sync::parallel_map(tasks, |task| -> Result<CrateOut> {
        match task {
            CrateTask::Outline(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_outline_crate(&mut ccur, files, macros, textuals, remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot outline crate");
                }
                Ok(CrateOut::Outline(pos, krate))
            }
            CrateTask::Names(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_names_crate(&mut ccur, remap, sym_remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot names crate");
                }
                Ok(CrateOut::Names(pos, krate))
            }
            CrateTask::Impls(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let index = read_impl_index(&mut ccur, local_crates[pos] as u32, remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot impl index");
                }
                Ok(CrateOut::Impls(pos, index))
            }
        }
    });
    let mut outline_slots: Vec<Option<OutlineCrate>> = (0..crate_count).map(|_| None).collect();
    let mut name_slots: Vec<Option<NamesCrate>> = (0..crate_count).map(|_| None).collect();
    let mut heads = HashMap::default();
    for item in decoded {
        match item? {
            CrateOut::Outline(pos, krate) => outline_slots[pos] = Some(krate),
            CrateOut::Names(pos, krate) => name_slots[pos] = Some(krate),
            CrateOut::Impls(pos, Some(index)) => {
                heads.insert(local_crates[pos] as u32, index);
            }
            CrateOut::Impls(_, None) => {}
        }
    }
    let outline_crates = local_crates
        .iter()
        .zip(outline_slots)
        .map(|(&k, s)| Ok((k, s.context("missing decoded outline crate")?)))
        .collect::<Result<Vec<_>>>()?;
    let name_crates = local_crates
        .iter()
        .zip(name_slots)
        .map(|(&k, s)| Ok((k, s.context("missing decoded names crate")?)))
        .collect::<Result<Vec<_>>>()?;
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

fn read_header_and_match(
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
    let mut identities = Vec::with_capacity(ref_count);
    for _ in 0..ref_count {
        identities.push(read_identity(cur)?);
    }
    if !match_crate_list(&identities, workspace, outline, spec.ref_crates) {
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

fn read_sections<'a>(
    cur: &mut Cursor<'a>,
    data_len: usize,
    crate_count: usize,
    magic: [u8; 8],
) -> Result<WireSections<'a>> {
    let sym_count = cur.count()?;
    let sym_blob_len = cur.count()?;
    let sym_ends = cur.take(sym_count.checked_mul(4).context("symbol ends overflow")?)?;
    let sym_blob = cur.take(sym_blob_len)?;
    let file_count = cur.count()?;
    let file_slices = read_file_slices(cur, file_count)?;
    let macro_count = cur.count()?;
    let macro_slices = read_slices(cur, macro_count)?;
    let textual_count = cur.count()?;
    let mut textual_raw = Vec::with_capacity(textual_count);
    for _ in 0..textual_count {
        textual_raw.push((read_str(cur)?.to_owned(), cur.u32()?, read_opt_u32(cur)?));
    }
    let outline_slices = read_slices(cur, crate_count)?;
    let name_slices = read_slices(cur, crate_count)?;
    let impl_slices = read_slices(cur, crate_count)?;
    if cur.take(magic.len())? != magic || cur.pos != data_len {
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

fn decode_textuals(
    textual_raw: Vec<(String, u32, Option<u32>)>,
    macros: &[Arc<Macro>],
) -> Result<Vec<Arc<Textual>>> {
    let mut textuals: Vec<Arc<Textual>> = Vec::with_capacity(textual_raw.len());
    for (name, def_id, next_id) in textual_raw {
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
        textuals.push(Arc::new(Textual { name, def, next }));
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
    let WireSections {
        sym_count,
        sym_ends,
        sym_blob,
        file_slices,
        macro_slices,
        textual_raw,
        outline_slices,
        name_slices,
        impl_slices,
    } = read_sections(&mut cur, data.len(), spec.cached_crates.len(), spec.magic)?;

    let decode_front = || -> Result<(Vec<Arc<File>>, Vec<Arc<Macro>>, Vec<Arc<Textual>>)> {
        let files = decode_files(backing, file_slices, &remap, spec.scope_bit)?;
        let macros = decode_macros(macro_slices, &files, &remap)?;
        let textuals = decode_textuals(textual_raw, &macros)?;
        Ok((files, macros, textuals))
    };

    let ((owned_symbols, sym_remap), (files, macros, textuals)) =
        if crate::jobs() > 1 && sym_count >= 256 {
            crate::sync::scope(|scope| -> Result<_> {
                let sym_handle =
                    scope.spawn(|| decode_symbols_with(sym_ends, sym_blob, existing_symbols));
                let front_res = decode_front();
                let syms_res = match sym_handle.join() {
                    Ok(res) => res,
                    Err(payload) => std::panic::resume_unwind(payload),
                };
                Ok((syms_res?, front_res?))
            })?
        } else {
            (
                decode_symbols_with(sym_ends, sym_blob, existing_symbols)?,
                decode_front()?,
            )
        };

    let (outline_crates, name_crates, heads) = decode_crates(
        outline_slices,
        name_slices,
        impl_slices,
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

fn read_identity(cur: &mut Cursor<'_>) -> Result<CrateIdentity> {
    let unit_name = read_str(cur)?.to_owned();
    let pkg_name = read_str(cur)?.to_owned();
    let pkg_version = read_str(cur)?.to_owned();
    let side_host = cur.u8()? != 0;
    let role = cur.u8()?;
    let feat_len = cur.count()?;
    let mut features = Vec::with_capacity(feat_len);
    for _ in 0..feat_len {
        features.push(read_str(cur)?.to_owned());
    }
    let dep_len = cur.count()?;
    let mut deps = Vec::with_capacity(dep_len);
    for _ in 0..dep_len {
        deps.push((read_str(cur)?.to_owned(), cur.u32()?));
    }
    Ok(CrateIdentity {
        unit_name,
        pkg_name,
        pkg_version,
        side_host,
        role,
        features,
        deps,
    })
}

fn read_file(
    cur: &mut Cursor<'_>,
    backing: &Arc<Backing>,
    file_off: usize,
    base: Option<Arc<File>>,
    remap: &[u32],
    scope_bit: u32,
) -> Result<File> {
    let path = read_path(cur)?;
    let text = match cur.u8()? {
        1 => {
            let text_len = cur.u32()? as usize;
            let tail = read_str(cur)?.to_owned().into_boxed_str();
            let disk = Arc::new(DiskSource::new(path.clone(), text_len, tail));
            SourceText::disk(disk)
        }
        0 => {
            let text_len = cur.count()?;
            let text_start = file_off + cur.pos;
            cur.take(text_len)?;
            let text_end = file_off + cur.pos;
            SourceText::mapped_trusted(Arc::clone(backing), text_start, text_end)?
        }
        _ => bail!("invalid sysroot file storage tag"),
    };
    let tok_count = cur.u32()? as usize;
    let tok_bytes = cur.count()?;
    let tok_start = file_off + cur.pos;
    cur.take(tok_bytes)?;
    let tok_end = file_off + cur.pos;
    let tokens = SourceTokens::compact(Arc::clone(backing), tok_start, tok_end, tok_count)?;
    let marks = read_marks(cur, remap, scope_bit)?;
    let opaque_len = cur.count()?;
    let opaque_bytes = cur.take(opaque_len.checked_mul(9).context("opaque overflow")?)?;
    let mut opaque = Vec::with_capacity(opaque_len);
    for c in opaque_bytes.chunks_exact(9) {
        opaque.push((u32_le(c, 0), u32_le(c, 4), frag_from_tag(c[8])?));
    }
    let offset = cur.u32()?;
    let origins = read_origins(cur)?;
    let items = read_items(cur)?;
    let expanded = cur.u8()? != 0;
    let expand_origin = cur.u64()?;
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
