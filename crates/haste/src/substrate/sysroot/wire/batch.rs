use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::names::{Crate as NamesCrate, Sym, Symbols};
use crate::outline::Outline;
use crate::outline::tree::{Crate as OutlineCrate, File, Macro};
use crate::sema::impls::Index as ImplIndex;
use crate::sema::summary::codec::Cursor;
use crate::substrate::Backing;

use super::super::EnvReads;
use super::super::impls::read_impl_index;
use super::super::names::read_names_crate;
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

fn decode_files_batch(
    items: &[PreparedWire],
    sections: &[WireSections<'_>],
) -> Result<Vec<Vec<Arc<File>>>> {
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
    let mut levels: Vec<Vec<(usize, usize, usize, &[u8], Option<u32>)>> =
        (0..=max_depth).map(|_| Vec::new()).collect();
    for (item_idx, sec) in sections.iter().enumerate() {
        for (file_idx, &(file_off, slice, base)) in sec.file_slices.iter().enumerate() {
            levels[all_depths[item_idx][file_idx]]
                .push((item_idx, file_idx, file_off, slice, base));
        }
    }
    for mut level in levels {
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
                &item.backing,
                file_off,
                base,
                &item.remap,
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
    file_slots
        .into_iter()
        .map(|slots| {
            slots
                .into_iter()
                .map(|f| f.context("missing decoded file slot"))
                .collect::<Result<Vec<_>>>()
        })
        .collect()
}

fn decode_macros_batch(
    items: &[PreparedWire],
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
            let remap = &items[item_idx].remap;
            let krate = *remap
                .get(mcur.u32()? as usize)
                .context("macro crate out of bounds")? as usize;
            let rules = read_rules(&mut mcur, remap)?;
            if mcur.pos != slice.len() {
                bail!("trailing bytes in snapshot macro");
            }
            Ok((item_idx, mac_idx, Arc::new(Macro { file, rules, krate })))
        };
    let decoded = if tasks.len() >= 16 {
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

fn decode_symbols_batch(sections: &[WireSections<'_>], syms: &Symbols) -> Result<Vec<Vec<Sym>>> {
    let total_syms: usize = sections.iter().map(|s| s.sym_count).sum();
    if total_syms == 0 {
        for sec in sections {
            if !sec.sym_blob.is_empty() {
                bail!("trailing bytes in snapshot symbol blob");
            }
        }
        return Ok(vec![Vec::new(); sections.len()]);
    }
    let workers = crate::jobs().clamp(1, 8);
    let chunk_syms = total_syms.div_ceil(workers).max(256);
    let mut chunks = Vec::new();
    let mut chunk_counts = vec![0usize; sections.len()];
    for (item_idx, sec) in sections.iter().enumerate() {
        let text = std::str::from_utf8(sec.sym_blob).context("invalid snapshot symbol utf-8")?;
        if sec.sym_count == 0 {
            if !text.is_empty() {
                bail!("trailing bytes in snapshot symbol blob");
            }
            continue;
        }
        let last_end = u32_le(sec.sym_ends, (sec.sym_count - 1) * 4) as usize;
        if last_end != text.len() {
            bail!("trailing bytes in snapshot symbol blob");
        }
        let mut start = 0usize;
        let mut prev = 0usize;
        let mut cidx = 0usize;
        while start < sec.sym_count {
            let end = (start + chunk_syms).min(sec.sym_count);
            let next_prev = u32_le(sec.sym_ends, (end - 1) * 4) as usize;
            if next_prev < prev || next_prev > text.len() || !text.is_char_boundary(next_prev) {
                bail!("invalid snapshot symbol boundary");
            }
            chunks.push((
                item_idx,
                cidx,
                prev,
                &sec.sym_ends[start * 4..end * 4],
                text,
            ));
            prev = next_prev;
            start = end;
            cidx += 1;
        }
        chunk_counts[item_idx] = cidx;
    }
    let decode_chunk = |(item_idx, cidx, init_prev, ends, text): (
        usize,
        usize,
        usize,
        &[u8],
        &str,
    )|
     -> Result<(usize, usize, Vec<Sym>)> {
        let mut out = Vec::with_capacity(ends.len() / 4);
        let mut p = init_prev;
        for c in ends.chunks_exact(4) {
            let e = u32_le(c, 0) as usize;
            if e < p || e > text.len() || !text.is_char_boundary(e) {
                bail!("invalid snapshot symbol boundary");
            }
            out.push(syms.intern(&text[p..e]));
            p = e;
        }
        Ok((item_idx, cidx, out))
    };
    let parts = if crate::jobs() > 1 && total_syms >= 256 {
        crate::sync::parallel_map(chunks, decode_chunk)
    } else {
        chunks.into_iter().map(decode_chunk).collect()
    };
    let mut per_item: Vec<Vec<Option<Vec<Sym>>>> =
        chunk_counts.into_iter().map(|n| vec![None; n]).collect();
    for res in parts {
        let (item_idx, cidx, vec) = res?;
        per_item[item_idx][cidx] = Some(vec);
    }
    let mut sym_remaps = Vec::with_capacity(sections.len());
    for (item_idx, slots) in per_item.into_iter().enumerate() {
        let mut flat = Vec::with_capacity(sections[item_idx].sym_count);
        for slot in slots {
            flat.extend(slot.context("missing decoded symbol chunk")?);
        }
        sym_remaps.push(flat);
    }
    Ok(sym_remaps)
}

pub(crate) fn decode_prepared_batch(
    items: Vec<PreparedWire>,
    existing_symbols: &Symbols,
) -> Result<Vec<DecodedSnapshot>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
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
    let all_files = decode_files_batch(&items, &sections)?;
    let all_macros = decode_macros_batch(&items, &sections, &all_files)?;
    let mut all_textuals = Vec::with_capacity(items.len());
    for (sec, macros) in sections.iter_mut().zip(&all_macros) {
        let raw = std::mem::take(&mut sec.textual_raw);
        all_textuals.push(decode_textuals(raw, macros)?);
    }
    let all_sym_remaps = decode_symbols_batch(&sections, existing_symbols)?;

    enum Task<'a> {
        Outline(usize, usize, &'a [u8]),
        Names(usize, usize, &'a [u8]),
        Impls(usize, usize, &'a [u8]),
    }
    enum Out {
        Outline(usize, usize, OutlineCrate),
        Names(usize, usize, NamesCrate),
        Impls(usize, usize, Option<Arc<ImplIndex>>),
    }

    let total_crates: usize = items.iter().map(|i| i.cached_crates.len()).sum();
    let parallel_scopes = total_crates < 4;
    let mut tasks = Vec::with_capacity(total_crates * 3);
    let mut outline_slots: Vec<Vec<Option<OutlineCrate>>> = Vec::with_capacity(items.len());
    let mut name_slots: Vec<Vec<Option<NamesCrate>>> = Vec::with_capacity(items.len());
    let mut heads_all: Vec<HashMap<u32, Arc<ImplIndex>>> =
        (0..items.len()).map(|_| HashMap::default()).collect();

    for (item_idx, sec) in sections.iter().enumerate() {
        let count = items[item_idx].cached_crates.len();
        outline_slots.push((0..count).map(|_| None).collect());
        name_slots.push((0..count).map(|_| None).collect());
        for (pos, &slice) in sec.outline_slices.iter().enumerate() {
            tasks.push(Task::Outline(item_idx, pos, slice));
        }
        for (pos, &slice) in sec.name_slices.iter().enumerate() {
            tasks.push(Task::Names(item_idx, pos, slice));
        }
        for (pos, &slice) in sec.impl_slices.iter().enumerate() {
            tasks.push(Task::Impls(item_idx, pos, slice));
        }
    }
    tasks.sort_unstable_by(|a, b| {
        let la = match a {
            Task::Outline(_, _, s) | Task::Names(_, _, s) | Task::Impls(_, _, s) => s.len(),
        };
        let lb = match b {
            Task::Outline(_, _, s) | Task::Names(_, _, s) | Task::Impls(_, _, s) => s.len(),
        };
        lb.cmp(&la)
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
                    &items[item_idx].remap,
                )?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot outline crate");
                }
                Ok(Out::Outline(item_idx, pos, krate))
            }
            Task::Names(item_idx, pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate = read_names_crate(
                    &mut ccur,
                    &items[item_idx].remap,
                    &all_sym_remaps[item_idx],
                    parallel_scopes,
                )?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot names crate");
                }
                Ok(Out::Names(item_idx, pos, krate))
            }
            Task::Impls(item_idx, pos, slice) => {
                let mut ccur = Cursor::new(slice);
                let krate_id = items[item_idx].cached_crates[pos] as u32;
                let index = read_impl_index(&mut ccur, krate_id, &items[item_idx].remap)?;
                if ccur.pos != slice.len() {
                    bail!("trailing bytes in snapshot impl index");
                }
                Ok(Out::Impls(item_idx, pos, index))
            }
        }
    });
    for res in decoded {
        match res? {
            Out::Outline(item_idx, pos, krate) => outline_slots[item_idx][pos] = Some(krate),
            Out::Names(item_idx, pos, krate) => name_slots[item_idx][pos] = Some(krate),
            Out::Impls(item_idx, pos, Some(index)) => {
                let krate_id = items[item_idx].cached_crates[pos] as u32;
                heads_all[item_idx].insert(krate_id, index);
            }
            Out::Impls(_, _, None) => {}
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
