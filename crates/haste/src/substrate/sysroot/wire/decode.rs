use std::sync::Arc;

use anyhow::{Context as _, Result, bail};

use crate::graph::Workspace;
use crate::names::{Crate as NamesCrate, Symbols};
use crate::outline::Outline;
use crate::outline::source::{DiskSource, Outputs, Source, SourceText, SourceTokens};
use crate::outline::tree::{Crate as OutlineCrate, File, Macro, Textual};
use crate::sema::summary::codec::Cursor;
use crate::substrate::Backing;

use super::super::names::read_names_crate;
use super::super::outline::read_outline_crate;
use super::super::tokens::{
    frag_from_tag, read_items, read_marks, read_opt_u32, read_origins, read_path, read_rules,
    read_str, u32_le,
};
use super::super::{CrateIdentity, HEADER_LEN, SYSROOT_MAGIC, SYSROOT_VERSION, match_crates};
use super::{FIXED_SYM_COUNT, opt_u32_le};

type DecodedCrates = (Vec<(usize, OutlineCrate)>, Vec<(usize, NamesCrate)>);

struct WireSections<'a> {
    sym_count: usize,
    sym_ends: &'a [u8],
    sym_blob: &'a [u8],
    file_slices: Vec<(usize, &'a [u8], Option<u32>)>,
    macro_slices: Vec<&'a [u8]>,
    textual_raw: Vec<(String, u32, Option<u32>)>,
    outline_slices: Vec<&'a [u8]>,
    name_slices: Vec<&'a [u8]>,
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

fn decode_symbols(sym_ends: &[u8], sym_blob: &[u8]) -> Result<Symbols> {
    let text = std::str::from_utf8(sym_blob).context("invalid sysroot symbol utf-8")?;
    let symbols = Symbols::default();
    let mut prev = 0usize;
    for (idx, c) in sym_ends.chunks_exact(4).enumerate() {
        let end = u32_le(c, 0) as usize;
        if end < prev || end > text.len() || !text.is_char_boundary(end) {
            bail!("invalid sysroot symbol boundary");
        }
        let expected = FIXED_SYM_COUNT + idx as u32;
        if symbols.intern(&text[prev..end]).0 != expected {
            bail!("sysroot symbol table mismatch");
        }
        prev = end;
    }
    if prev != text.len() {
        bail!("trailing bytes in sysroot symbol blob");
    }
    Ok(symbols)
}

fn decode_files(
    backing: &Arc<Backing>,
    file_slices: Vec<(usize, &[u8], Option<u32>)>,
    remap: &[u32],
) -> Result<Vec<Arc<File>>> {
    let file_count = file_slices.len();
    let mut depths = vec![0usize; file_count];
    let mut max_depth = 0usize;
    for (idx, &(_, _, base)) in file_slices.iter().enumerate() {
        if let Some(base_id) = base {
            let base_idx = base_id as usize;
            if base_idx >= idx {
                bail!("invalid sysroot file base dependency");
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
            let file = read_file(&mut fcur, backing, file_off, base, remap)?;
            if fcur.pos != slice.len() {
                bail!("trailing bytes in sysroot file");
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
    Ok(file_slots
        .into_iter()
        .map(|f| f.expect("every file index decoded"))
        .collect())
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
            bail!("trailing bytes in sysroot macro");
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
    Ok(slots
        .into_iter()
        .map(|m| m.expect("every macro decoded"))
        .collect())
}

fn decode_crates(
    outline_slices: Vec<&[u8]>,
    name_slices: Vec<&[u8]>,
    files: &[Arc<File>],
    macros: &[Arc<Macro>],
    textuals: &[Arc<Textual>],
    local_crates: &[usize],
    remap: &[u32],
) -> Result<DecodedCrates> {
    enum CrateTask<'a> {
        Outline(usize, &'a [u8]),
        Names(usize, &'a [u8]),
    }
    enum CrateOut {
        Outline(usize, OutlineCrate),
        Names(usize, NamesCrate),
    }
    let crate_count = local_crates.len();
    let mut tasks = Vec::with_capacity(crate_count * 2);
    for (pos, slice) in outline_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Outline(pos, slice));
    }
    for (pos, slice) in name_slices.into_iter().enumerate() {
        tasks.push(CrateTask::Names(pos, slice));
    }
    tasks.sort_unstable_by(|a, b| {
        let la = match a {
            CrateTask::Outline(_, s) | CrateTask::Names(_, s) => s.len(),
        };
        let lb = match b {
            CrateTask::Outline(_, s) | CrateTask::Names(_, s) => s.len(),
        };
        lb.cmp(&la)
    });
    let decoded = crate::sync::parallel_map(tasks, |task| -> Result<CrateOut> {
        match task {
            CrateTask::Outline(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_outline_crate(&mut ccur, files, macros, textuals, remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in sysroot outline crate");
                }
                Ok(CrateOut::Outline(pos, krate))
            }
            CrateTask::Names(pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_names_crate(&mut ccur, remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in sysroot names crate");
                }
                Ok(CrateOut::Names(pos, krate))
            }
        }
    });
    let mut outline_slots: Vec<Option<OutlineCrate>> = (0..crate_count).map(|_| None).collect();
    let mut name_slots: Vec<Option<NamesCrate>> = (0..crate_count).map(|_| None).collect();
    for item in decoded {
        match item? {
            CrateOut::Outline(pos, krate) => outline_slots[pos] = Some(krate),
            CrateOut::Names(pos, krate) => name_slots[pos] = Some(krate),
        }
    }
    let outline_crates = local_crates
        .iter()
        .zip(outline_slots)
        .map(|(&k, s)| (k, s.expect("every outline crate decoded")))
        .collect();
    let name_crates = local_crates
        .iter()
        .zip(name_slots)
        .map(|(&k, s)| (k, s.expect("every names crate decoded")))
        .collect();
    Ok((outline_crates, name_crates))
}

fn read_header_and_match(
    cur: &mut Cursor<'_>,
    data: &[u8],
    expected_key: u64,
    workspace: &Workspace,
    outline: &Outline,
) -> Result<Option<Vec<usize>>> {
    if cur.take(8)? != SYSROOT_MAGIC {
        bail!("invalid sysroot cache magic");
    }
    if cur.u32()? != SYSROOT_VERSION {
        return Ok(None);
    }
    let crate_count = cur.count()?;
    if cur.u64()? != expected_key {
        return Ok(None);
    }
    let expected_len = cur.u64()?;
    if expected_len != data.len() as u64
        || data.len() < HEADER_LEN + SYSROOT_MAGIC.len()
        || &data[data.len() - SYSROOT_MAGIC.len()..] != SYSROOT_MAGIC
    {
        return Ok(None);
    }
    let mut identities = Vec::with_capacity(crate_count);
    for _ in 0..crate_count {
        identities.push(read_identity(cur)?);
    }
    Ok(match_crates(&identities, workspace, outline))
}

fn read_sections<'a>(
    cur: &mut Cursor<'a>,
    data_len: usize,
    crate_count: usize,
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
    if cur.take(SYSROOT_MAGIC.len())? != SYSROOT_MAGIC || cur.pos != data_len {
        bail!("trailing bytes in sysroot cache");
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

pub(in super::super) fn decode_sysroot(
    backing: &Arc<Backing>,
    expected_key: u64,
    workspace: &Workspace,
    outline: &Outline,
) -> Result<
    Option<(
        Vec<(usize, OutlineCrate)>,
        Symbols,
        Vec<(usize, NamesCrate)>,
    )>,
> {
    let data = backing.as_slice();
    let mut cur = Cursor::new(data);
    let Some(local_crates) =
        read_header_and_match(&mut cur, data, expected_key, workspace, outline)?
    else {
        return Ok(None);
    };
    let remap: Vec<u32> = local_crates.iter().map(|&k| k as u32).collect();
    let WireSections {
        sym_count,
        sym_ends,
        sym_blob,
        file_slices,
        macro_slices,
        textual_raw,
        outline_slices,
        name_slices,
    } = read_sections(&mut cur, data.len(), local_crates.len())?;

    let decode_rest = || -> Result<DecodedCrates> {
        let files = decode_files(backing, file_slices, &remap)?;
        let macros = decode_macros(macro_slices, &files, &remap)?;
        let textuals = decode_textuals(textual_raw, &macros)?;
        decode_crates(
            outline_slices,
            name_slices,
            &files,
            &macros,
            &textuals,
            &local_crates,
            &remap,
        )
    };

    let (symbols, (outline_crates, name_crates)) = if crate::jobs() > 1 && sym_count >= 256 {
        crate::sync::scope(|scope| -> Result<(Symbols, DecodedCrates)> {
            let sym_handle = scope.spawn(|| decode_symbols(sym_ends, sym_blob));
            let crates_res = decode_rest();
            let syms_res = match sym_handle.join() {
                Ok(res) => res,
                Err(payload) => std::panic::resume_unwind(payload),
            };
            Ok((syms_res?, crates_res?))
        })?
    } else {
        (decode_symbols(sym_ends, sym_blob)?, decode_rest()?)
    };

    Ok(Some((outline_crates, symbols, name_crates)))
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
) -> Result<File> {
    let path = read_path(cur)?;
    let (text, tokens) = match cur.u8()? {
        1 => {
            let text_len = cur.count()?;
            let tok_count = cur.count()?;
            let disk = Arc::new(DiskSource::new(path.clone(), text_len, tok_count));
            (
                SourceText::disk(Arc::clone(&disk)),
                SourceTokens::disk(disk),
            )
        }
        0 => {
            let text_len = cur.count()?;
            let text_start = file_off + cur.pos;
            cur.take(text_len)?;
            let text_end = file_off + cur.pos;
            let text = SourceText::mapped_trusted(Arc::clone(backing), text_start, text_end)?;
            let tok_count = cur.count()?;
            let tok_bytes = cur.count()?;
            let tok_start = file_off + cur.pos;
            cur.take(tok_bytes)?;
            let tok_end = file_off + cur.pos;
            let tokens = SourceTokens::compact(Arc::clone(backing), tok_start, tok_end, tok_count)?;
            (text, tokens)
        }
        _ => bail!("invalid sysroot file storage tag"),
    };
    let marks = read_marks(cur, remap)?;
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
