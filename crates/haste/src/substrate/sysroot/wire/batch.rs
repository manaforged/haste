use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::names::{Crate as NamesCrate, Scope, Sym, Symbols};
use crate::outline::Outline;
use crate::outline::tree::{Crate as OutlineCrate, File, Macro};
use crate::sema::impls::Index as ImplIndex;
use crate::sema::summary::codec::Cursor;
use crate::substrate::Backing;

use super::super::EnvReads;
use super::super::impls::read_impl_index;
use super::super::names::{decode_scope_slice, read_names_crate_body, split_names_crate_scopes};
use super::super::outline::read_outline_crate;
use super::super::tokens::{read_rules, u32_le};
use super::decode::{
    DecodeSpec, DecodedSnapshot, WireSections, decode_textuals, read_file, read_header_and_match,
    read_sections,
};

pub(crate) struct PreparedWire {
    pub backing: Arc<Backing>,
    pub magic: [u8; 8],
    pub cached_crates: Vec<usize>,
    pub remap: Vec<u32>,
    pub scope_bit: u32,
    pub env_reads: Vec<EnvReads>,
    pub sections_off: usize,
}

pub(in crate::substrate) fn prepare_wire(
    backing: Arc<Backing>,
    spec: &DecodeSpec<'_>,
    workspace: &Workspace,
    outline: &Outline,
) -> Result<Option<PreparedWire>> {
    let data = backing.as_slice();
    let mut cur = Cursor::new(data);
    let Some(env_reads) = read_header_and_match(&mut cur, data, spec, workspace, outline)? else {
        return Ok(None);
    };
    let sections_off = cur.pos;
    let remap: Vec<u32> = spec.ref_crates.iter().map(|&k| k as u32).collect();
    Ok(Some(PreparedWire {
        backing,
        magic: spec.magic,
        cached_crates: spec.cached_crates.to_vec(),
        remap,
        scope_bit: spec.scope_bit,
        env_reads,
        sections_off,
    }))
}

struct SymChunk<'a> {
    item_idx: usize,
    chunk_idx: usize,
    base_off: usize,
    ends: &'a [u8],
    bytes: &'a [u8],
}

fn plan_sym_chunks<'a>(
    sections: &[WireSections<'a>],
    workers: usize,
) -> Result<(Vec<SymChunk<'a>>, Vec<usize>)> {
    let total_syms: usize = sections.iter().map(|s| s.sym_count).sum();
    if total_syms == 0 {
        for sec in sections {
            if !sec.sym_blob.is_empty() {
                bail!("trailing bytes in snapshot symbol blob");
            }
        }
        return Ok((Vec::new(), vec![0; sections.len()]));
    }
    let chunk_syms = if total_syms >= 4096 {
        total_syms.div_ceil(workers * 2).max(256)
    } else {
        total_syms.div_ceil(workers).max(256)
    };
    let mut chunks = Vec::new();
    let mut chunk_counts = vec![0usize; sections.len()];
    for (item_idx, sec) in sections.iter().enumerate() {
        if sec.sym_count == 0 {
            if !sec.sym_blob.is_empty() {
                bail!("trailing bytes in snapshot symbol blob");
            }
            continue;
        }
        let last_end = u32_le(sec.sym_ends, (sec.sym_count - 1) * 4) as usize;
        if last_end != sec.sym_blob.len() {
            bail!("trailing bytes in snapshot symbol blob");
        }
        let mut start = 0usize;
        let mut prev = 0usize;
        let mut cidx = 0usize;
        while start < sec.sym_count {
            let end = (start + chunk_syms).min(sec.sym_count);
            let next_prev = u32_le(sec.sym_ends, (end - 1) * 4) as usize;
            if next_prev < prev || next_prev > sec.sym_blob.len() {
                bail!("invalid snapshot symbol boundary");
            }
            chunks.push(SymChunk {
                item_idx,
                chunk_idx: cidx,
                base_off: prev,
                ends: &sec.sym_ends[start * 4..end * 4],
                bytes: &sec.sym_blob[prev..next_prev],
            });
            prev = next_prev;
            start = end;
            cidx += 1;
        }
        chunk_counts[item_idx] = cidx;
    }
    Ok((chunks, chunk_counts))
}

fn decode_sym_chunk(chunk: SymChunk<'_>, syms: &Symbols) -> Result<(usize, usize, Vec<Sym>)> {
    let text = std::str::from_utf8(chunk.bytes).context("invalid snapshot symbol utf-8")?;
    let mut out = Vec::with_capacity(chunk.ends.len() / 4);
    let mut p = 0usize;
    for c in chunk.ends.chunks_exact(4) {
        let abs_e = u32_le(c, 0) as usize;
        let e = abs_e
            .checked_sub(chunk.base_off)
            .context("invalid snapshot symbol boundary")?;
        if e < p || e > text.len() || !text.is_char_boundary(e) {
            bail!("invalid snapshot symbol boundary");
        }
        out.push(syms.intern(&text[p..e]));
        p = e;
    }
    if p != text.len() {
        bail!("invalid snapshot symbol boundary");
    }
    Ok((chunk.item_idx, chunk.chunk_idx, out))
}

struct BatchInput<'a> {
    backing: &'a Arc<Backing>,
    cached_crates: &'a [usize],
    remap: &'a [u32],
    scope_bit: u32,
    env_reads: Vec<EnvReads>,
}

fn decode_files_and_syms<'a>(
    items: &[BatchInput<'a>],
    sections: &[WireSections<'a>],
    syms: &Symbols,
) -> Result<(Vec<Vec<Arc<File>>>, Vec<Vec<Sym>>)> {
    let mut all_depths = Vec::with_capacity(items.len());
    let mut max_depth = 0usize;
    let mut file_slots: Vec<Vec<Option<Arc<File>>>> = Vec::with_capacity(items.len());
    for sec in sections {
        let count = sec.file_slices.len();
        let mut depths = vec![0usize; count];
        for (idx, &(_, _, base)) in sec.file_slices.iter().enumerate() {
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
        all_depths.push(depths);
        file_slots.push(vec![None; count]);
    }

    enum Stage1Task<'a> {
        File {
            item_idx: usize,
            file_idx: usize,
            file_off: usize,
            slice: &'a [u8],
        },
        Sym(SymChunk<'a>),
    }
    enum Stage1Out {
        File(usize, usize, Arc<File>),
        Sym(usize, usize, Vec<Sym>),
    }

    let workers = crate::jobs().clamp(1, 8);
    let (sym_chunks, chunk_counts) = plan_sym_chunks(sections, workers)?;
    let mut levels: Vec<Vec<(usize, usize, usize, &[u8], Option<u32>)>> =
        (0..=max_depth).map(|_| Vec::new()).collect();
    let mut stage1 = Vec::with_capacity(sym_chunks.len());
    for chunk in sym_chunks {
        stage1.push(Stage1Task::Sym(chunk));
    }
    for (item_idx, sec) in sections.iter().enumerate() {
        for (file_idx, &(file_off, slice, base)) in sec.file_slices.iter().enumerate() {
            let depth = all_depths[item_idx][file_idx];
            if depth == 0 {
                stage1.push(Stage1Task::File {
                    item_idx,
                    file_idx,
                    file_off,
                    slice,
                });
            } else {
                levels[depth].push((item_idx, file_idx, file_off, slice, base));
            }
        }
    }
    stage1.sort_unstable_by(|a, b| {
        let wa = match a {
            Stage1Task::File { slice, .. } => slice.len(),
            Stage1Task::Sym(chunk) => chunk.bytes.len(),
        };
        let wb = match b {
            Stage1Task::File { slice, .. } => slice.len(),
            Stage1Task::Sym(chunk) => chunk.bytes.len(),
        };
        wb.cmp(&wa)
    });
    let run_stage1 = |task: Stage1Task<'_>| -> Result<Stage1Out> {
        match task {
            Stage1Task::File {
                item_idx,
                file_idx,
                file_off,
                slice,
            } => {
                let mut fcur = Cursor::new(slice);
                let item = &items[item_idx];
                let file = read_file(
                    &mut fcur,
                    item.backing,
                    file_off,
                    None,
                    item.remap,
                    item.scope_bit,
                )?;
                if fcur.pos != slice.len() {
                    bail!("trailing bytes in snapshot file");
                }
                Ok(Stage1Out::File(item_idx, file_idx, Arc::new(file)))
            }
            Stage1Task::Sym(chunk) => {
                let (item_idx, cidx, vec) = decode_sym_chunk(chunk, syms)?;
                Ok(Stage1Out::Sym(item_idx, cidx, vec))
            }
        }
    };
    let stage1_out = if stage1.len() >= 2 && crate::jobs() > 1 {
        crate::sync::parallel_map(stage1, run_stage1)
    } else {
        stage1.into_iter().map(run_stage1).collect()
    };
    let mut per_item_syms: Vec<Vec<Option<Vec<Sym>>>> =
        chunk_counts.into_iter().map(|n| vec![None; n]).collect();
    for res in stage1_out {
        match res? {
            Stage1Out::File(item_idx, file_idx, file) => {
                file_slots[item_idx][file_idx] = Some(file);
            }
            Stage1Out::Sym(item_idx, cidx, vec) => {
                per_item_syms[item_idx][cidx] = Some(vec);
            }
        }
    }
    let mut sym_remaps = Vec::with_capacity(sections.len());
    for (item_idx, slots) in per_item_syms.into_iter().enumerate() {
        let mut flat = Vec::with_capacity(sections[item_idx].sym_count);
        for slot in slots {
            flat.extend(slot.context("missing decoded symbol chunk")?);
        }
        sym_remaps.push(flat);
    }

    for mut level in levels.into_iter().skip(1) {
        if level.is_empty() {
            continue;
        }
        level.sort_unstable_by(|a, b| b.3.len().cmp(&a.3.len()));
        let slots_ref: &[Vec<Option<Arc<File>>>] = &file_slots;
        let decode_one = |(item_idx, file_idx, file_off, slice, base_id): (
            usize,
            usize,
            usize,
            &[u8],
            Option<u32>,
        )|
         -> Result<(usize, usize, Arc<File>)> {
            let base = match base_id {
                Some(id) => Some(Arc::clone(
                    slots_ref[item_idx][id as usize]
                        .as_ref()
                        .context("base file out of bounds")?,
                )),
                None => None,
            };
            let mut fcur = Cursor::new(slice);
            let item = &items[item_idx];
            let file = read_file(
                &mut fcur,
                item.backing,
                file_off,
                base,
                item.remap,
                item.scope_bit,
            )?;
            if fcur.pos != slice.len() {
                bail!("trailing bytes in snapshot file");
            }
            Ok((item_idx, file_idx, Arc::new(file)))
        };
        let total_bytes: usize = level.iter().map(|(_, _, _, s, _)| s.len()).sum();
        let decoded = if level.len() >= 4 || (level.len() >= 2 && total_bytes >= 32 * 1024) {
            crate::sync::parallel_map(level, decode_one)
        } else {
            level.into_iter().map(decode_one).collect()
        };
        for res in decoded {
            let (item_idx, file_idx, file) = res?;
            file_slots[item_idx][file_idx] = Some(file);
        }
    }
    let all_files = file_slots
        .into_iter()
        .map(|slots| {
            slots
                .into_iter()
                .map(|f| f.context("missing decoded file slot"))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((all_files, sym_remaps))
}

fn decode_macros_batch(
    items: &[BatchInput<'_>],
    sections: &[WireSections<'_>],
    all_files: &[Vec<Arc<File>>],
) -> Result<Vec<Vec<Arc<Macro>>>> {
    let total: usize = sections.iter().map(|s| s.macro_slices.len()).sum();
    let mut tasks = Vec::with_capacity(total);
    let mut macro_slots: Vec<Vec<Option<Arc<Macro>>>> = Vec::with_capacity(items.len());
    for (item_idx, sec) in sections.iter().enumerate() {
        macro_slots.push(vec![None; sec.macro_slices.len()]);
        for (mac_idx, &slice) in sec.macro_slices.iter().enumerate() {
            tasks.push((item_idx, mac_idx, slice));
        }
    }
    tasks.sort_unstable_by(|a, b| b.2.len().cmp(&a.2.len()));
    let decode_one =
        |(item_idx, mac_idx, slice): (usize, usize, &[u8])| -> Result<(usize, usize, Arc<Macro>)> {
            let mut mcur = Cursor::new(slice);
            let file = Arc::clone(
                all_files[item_idx]
                    .get(mcur.u32()? as usize)
                    .context("macro file out of bounds")?,
            );
            let remap = items[item_idx].remap;
            let krate = *remap
                .get(mcur.u32()? as usize)
                .context("macro crate out of bounds")? as usize;
            let rules = read_rules(&mut mcur, remap)?;
            if mcur.pos != slice.len() {
                bail!("trailing bytes in snapshot macro");
            }
            Ok((item_idx, mac_idx, Arc::new(Macro { file, rules, krate })))
        };
    let decoded = if tasks.len() >= 4 && crate::jobs() > 1 {
        crate::sync::parallel_map(tasks, decode_one)
    } else {
        tasks.into_iter().map(decode_one).collect()
    };
    for res in decoded {
        let (item_idx, mac_idx, mac) = res?;
        macro_slots[item_idx][mac_idx] = Some(mac);
    }
    macro_slots
        .into_iter()
        .map(|slots| {
            slots
                .into_iter()
                .map(|m| m.context("missing decoded macro slot"))
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

fn chunk_scope_slices<'a>(slices: Vec<&'a [u8]>) -> Vec<(Vec<&'a [u8]>, usize)> {
    const TARGET_BYTES: usize = 8 * 1024;
    const MAX_SCOPES: usize = 64;
    let mut chunks = Vec::new();
    let mut cur = Vec::new();
    let mut cur_bytes = 0usize;
    for s in slices {
        cur_bytes += s.len();
        cur.push(s);
        if cur_bytes >= TARGET_BYTES || cur.len() >= MAX_SCOPES {
            chunks.push((std::mem::take(&mut cur), cur_bytes));
            cur_bytes = 0;
        }
    }
    if !cur.is_empty() {
        chunks.push((cur, cur_bytes));
    }
    chunks
}

fn decode_batch_sections<'a>(
    items: Vec<BatchInput<'a>>,
    mut sections: Vec<WireSections<'a>>,
    syms: &Symbols,
) -> Result<Vec<DecodedSnapshot>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let (all_files, sym_remaps) = decode_files_and_syms(&items, &sections, syms)?;
    let all_macros = decode_macros_batch(&items, &sections, &all_files)?;
    let mut all_textuals = Vec::with_capacity(items.len());
    for (sec, macros) in sections.iter_mut().zip(&all_macros) {
        let raw = std::mem::take(&mut sec.textual_raw);
        all_textuals.push(decode_textuals(raw, macros)?);
    }

    enum Task<'a> {
        Outline(usize, usize, &'a [u8]),
        Impls(usize, usize, &'a [u8]),
        NamesFull(usize, usize, Vec<&'a [u8]>, &'a [u8], usize),
        NamesBody(usize, usize, &'a [u8]),
        ScopeChunk(usize, usize, usize, Vec<&'a [u8]>, usize),
    }
    enum Out {
        Outline(usize, usize, OutlineCrate),
        Impls(usize, usize, Option<Arc<ImplIndex>>),
        Names(usize, usize, NamesCrate),
        ScopeChunk(usize, usize, usize, Vec<Scope>),
    }

    let total_crates: usize = items.iter().map(|i| i.cached_crates.len()).sum();
    let mut tasks = Vec::with_capacity(total_crates * 3);
    let mut outline_slots: Vec<Vec<Option<OutlineCrate>>> = Vec::with_capacity(items.len());
    let mut name_slots: Vec<Vec<Option<NamesCrate>>> = Vec::with_capacity(items.len());
    let mut scope_chunks: Vec<Vec<Vec<Option<Vec<Scope>>>>> = Vec::with_capacity(items.len());
    let mut heads_all: Vec<HashMap<u32, Arc<ImplIndex>>> =
        (0..items.len()).map(|_| HashMap::default()).collect();

    for (item_idx, sec) in sections.iter().enumerate() {
        let count = items[item_idx].cached_crates.len();
        outline_slots.push((0..count).map(|_| None).collect());
        name_slots.push((0..count).map(|_| None).collect());
        let mut crate_chunks = vec![Vec::new(); count];
        for (pos, &slice) in sec.outline_slices.iter().enumerate() {
            tasks.push(Task::Outline(item_idx, pos, slice));
        }
        for (pos, &slice) in sec.impl_slices.iter().enumerate() {
            tasks.push(Task::Impls(item_idx, pos, slice));
        }
        for (pos, &slice) in sec.name_slices.iter().enumerate() {
            let mut ncur = Cursor::new(slice);
            let split = split_names_crate_scopes(&mut ncur)?;
            let body_slice = &slice[ncur.pos..];
            if split.total_bytes >= 8 * 1024 && split.slices.len() >= 2 && crate::jobs() > 1 {
                let chunks = chunk_scope_slices(split.slices);
                crate_chunks[pos] = vec![None; chunks.len()];
                for (cidx, (chunk_slices, chunk_bytes)) in chunks.into_iter().enumerate() {
                    tasks.push(Task::ScopeChunk(
                        item_idx,
                        pos,
                        cidx,
                        chunk_slices,
                        chunk_bytes,
                    ));
                }
                tasks.push(Task::NamesBody(item_idx, pos, body_slice));
            } else {
                tasks.push(Task::NamesFull(
                    item_idx,
                    pos,
                    split.slices,
                    body_slice,
                    split.total_bytes + body_slice.len(),
                ));
            }
        }
        scope_chunks.push(crate_chunks);
    }
    tasks.sort_unstable_by(|a, b| {
        let wa = match a {
            Task::Outline(_, _, s) | Task::Impls(_, _, s) | Task::NamesBody(_, _, s) => s.len(),
            Task::NamesFull(_, _, _, _, w) | Task::ScopeChunk(_, _, _, _, w) => *w,
        };
        let wb = match b {
            Task::Outline(_, _, s) | Task::Impls(_, _, s) | Task::NamesBody(_, _, s) => s.len(),
            Task::NamesFull(_, _, _, _, w) | Task::ScopeChunk(_, _, _, _, w) => *w,
        };
        wb.cmp(&wa)
    });
    let decoded = crate::sync::parallel_map(tasks, |task| -> Result<Out> {
        match task {
            Task::Outline(item_idx, pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_outline_crate(
                    &mut ccur,
                    &all_files[item_idx],
                    &all_macros[item_idx],
                    &all_textuals[item_idx],
                    items[item_idx].remap,
                )?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot outline crate");
                }
                Ok(Out::Outline(item_idx, pos, krate))
            }
            Task::Impls(item_idx, pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate_id = items[item_idx].cached_crates[pos] as u32;
                let index = read_impl_index(&mut ccur, krate_id, items[item_idx].remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot impl index");
                }
                Ok(Out::Impls(item_idx, pos, index))
            }
            Task::NamesFull(item_idx, pos, slices, body_slice, _) => {
                let scopes =
                    decode_scope_slice(&slices, items[item_idx].remap, &sym_remaps[item_idx])?;
                let mut ccur = Cursor::new(body_slice);
                let krate = read_names_crate_body(
                    &mut ccur,
                    scopes,
                    items[item_idx].remap,
                    &sym_remaps[item_idx],
                )?;
                if ccur.pos != body_slice.len() {
                    bail!("trailing bytes in snapshot names crate");
                }
                Ok(Out::Names(item_idx, pos, krate))
            }
            Task::NamesBody(item_idx, pos, body_slice) => {
                let mut ccur = Cursor::new(body_slice);
                let krate = read_names_crate_body(
                    &mut ccur,
                    Vec::new(),
                    items[item_idx].remap,
                    &sym_remaps[item_idx],
                )?;
                if ccur.pos != body_slice.len() {
                    bail!("trailing bytes in snapshot names crate");
                }
                Ok(Out::Names(item_idx, pos, krate))
            }
            Task::ScopeChunk(item_idx, pos, cidx, slices, _) => {
                let scopes =
                    decode_scope_slice(&slices, items[item_idx].remap, &sym_remaps[item_idx])?;
                Ok(Out::ScopeChunk(item_idx, pos, cidx, scopes))
            }
        }
    });
    for res in decoded {
        match res? {
            Out::Outline(item_idx, pos, krate) => outline_slots[item_idx][pos] = Some(krate),
            Out::Impls(item_idx, pos, Some(index)) => {
                let krate_id = items[item_idx].cached_crates[pos] as u32;
                heads_all[item_idx].insert(krate_id, index);
            }
            Out::Impls(_, _, None) => {}
            Out::Names(item_idx, pos, krate) => name_slots[item_idx][pos] = Some(krate),
            Out::ScopeChunk(item_idx, pos, cidx, scopes) => {
                scope_chunks[item_idx][pos][cidx] = Some(scopes);
            }
        }
    }
    for (item_idx, crate_chunks) in scope_chunks.into_iter().enumerate() {
        for (pos, chunks) in crate_chunks.into_iter().enumerate() {
            if chunks.is_empty() {
                continue;
            }
            let total: usize = chunks.iter().map(|c| c.as_ref().map_or(0, Vec::len)).sum();
            let mut scopes = Vec::with_capacity(total);
            for chunk in chunks {
                scopes.extend(chunk.context("missing decoded scope chunk")?);
            }
            let krate = name_slots[item_idx][pos]
                .as_mut()
                .context("missing decoded names crate")?;
            krate.scopes = scopes;
        }
    }

    let mut out = Vec::with_capacity(items.len());
    for (((item, oslots), nslots), heads) in items
        .into_iter()
        .zip(outline_slots)
        .zip(name_slots)
        .zip(heads_all)
    {
        let outline_crates = item
            .cached_crates
            .iter()
            .zip(oslots)
            .map(|(&k, s)| Ok((k, s.context("missing decoded outline crate")?)))
            .collect::<Result<Vec<_>>>()?;
        let name_crates = item
            .cached_crates
            .iter()
            .zip(nslots)
            .map(|(&k, s)| Ok((k, s.context("missing decoded names crate")?)))
            .collect::<Result<Vec<_>>>()?;
        out.push((outline_crates, None, name_crates, heads, item.env_reads));
    }
    Ok(out)
}

pub(crate) fn decode_prepared_batch(
    mut items: Vec<PreparedWire>,
    existing_symbols: &Symbols,
) -> Result<Vec<DecodedSnapshot>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let env_reads_all: Vec<Vec<EnvReads>> = items
        .iter_mut()
        .map(|item| std::mem::take(&mut item.env_reads))
        .collect();
    let mut sections = Vec::with_capacity(items.len());
    for item in &items {
        let data = item.backing.as_slice();
        let mut cur = Cursor::new(data);
        cur.pos = item.sections_off;
        sections.push(read_sections(
            &mut cur,
            data.len(),
            item.cached_crates.len(),
            item.magic,
        )?);
    }
    let inputs: Vec<BatchInput<'_>> = items
        .iter()
        .zip(env_reads_all)
        .map(|(item, env_reads)| BatchInput {
            backing: &item.backing,
            cached_crates: &item.cached_crates,
            remap: &item.remap,
            scope_bit: item.scope_bit,
            env_reads,
        })
        .collect();
    decode_batch_sections(inputs, sections, existing_symbols)
}

pub(super) fn decode_snapshot_parallel(
    backing: &Arc<Backing>,
    cached_crates: &[usize],
    remap: &[u32],
    scope_bit: u32,
    env_reads: Vec<EnvReads>,
    sections: WireSections<'_>,
    syms: &Symbols,
) -> Result<DecodedSnapshot> {
    let inputs = vec![BatchInput {
        backing,
        cached_crates,
        remap,
        scope_bit,
        env_reads,
    }];
    let mut decoded = decode_batch_sections(inputs, vec![sections], syms)?;
    decoded.pop().context("missing decoded snapshot")
}
