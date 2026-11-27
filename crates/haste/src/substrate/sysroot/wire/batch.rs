use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{Context as _, Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::graph::Workspace;
use crate::names::Symbols;
use crate::outline::Outline;
use crate::sema::summary::codec::Cursor;
use crate::substrate::Backing;

use super::super::EnvReads;
use super::super::names::split_names_crate_scopes;
use super::super::tokens::{Reader, u32_le};
use super::dag::{
    BatchInput, DagCtx, DagTask, ItemState, chunk_scope_ranges, plan_sym_chunks, worker_loop,
};
use super::decode::{
    DecodeSpec, DecodedSnapshot, WireSections, read_header_and_match, read_sections,
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

fn decode_batch_sections<'a>(
    inputs: Vec<BatchInput<'a>>,
    sections: Vec<WireSections<'a>>,
    syms: &Symbols,
) -> Result<Vec<DecodedSnapshot>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let sym_workers = crate::jobs().clamp(1, 16);
    let mut all_sym_chunks = plan_sym_chunks(&sections, sym_workers)?;
    let mut states = Vec::with_capacity(inputs.len());
    let mut initial = Vec::new();
    let mut total_tasks = 0usize;

    for (item_idx, (input, sec)) in inputs.iter().zip(sections).enumerate() {
        let file_count = sec.file_slices.len();
        let macro_count = sec.macro_slices.len();
        let crate_count = input.cached_crates.len();

        let mut file_children = vec![Vec::new(); file_count];
        let mut depth0_files = Vec::new();
        for (file_idx, &(_, slice, base)) in sec.file_slices.iter().enumerate() {
            if let Some(base_id) = base {
                let base_idx = base_id as usize;
                if base_idx >= file_idx {
                    bail!("invalid snapshot file base dependency");
                }
                file_children[base_idx].push((file_idx, slice.len()));
            } else {
                depth0_files.push((file_idx, slice.len()));
            }
        }
        let mut file_macros = vec![Vec::new(); file_count];
        for (mac_idx, &slice) in sec.macro_slices.iter().enumerate() {
            if slice.len() < 4 {
                bail!("unexpected eof");
            }
            let file_id = u32_le(slice, 0) as usize;
            if file_id >= file_count {
                bail!("macro file out of bounds");
            }
            file_macros[file_id].push((mac_idx, slice.len()));
        }

        let item_sym_chunks = std::mem::take(&mut all_sym_chunks[item_idx]);
        let sym_chunk_count = item_sym_chunks.len();
        let sym_chunk_slots: Vec<_> = (0..sym_chunk_count).map(|_| Mutex::new(None)).collect();
        let sym_remap = OnceLock::new();
        if sym_chunk_count == 0 {
            let _ = sym_remap.set(Vec::new());
        } else {
            for (chunk_idx, chunk) in item_sym_chunks.iter().enumerate() {
                initial.push(DagTask::SymChunk {
                    item_idx,
                    chunk_idx,
                    weight: chunk.bytes.len(),
                });
            }
        }

        for (file_idx, weight) in depth0_files {
            initial.push(DagTask::File {
                item_idx,
                file_idx,
                base_id: None,
                weight,
            });
        }
        for (pos, &slice) in sec.impl_slices.iter().enumerate() {
            initial.push(DagTask::Impls {
                item_idx,
                pos,
                weight: slice.len(),
            });
        }

        let all_files = OnceLock::new();
        if file_count == 0 {
            let _ = all_files.set(Vec::new());
        }
        let all_macros = OnceLock::new();
        if macro_count == 0 {
            let _ = all_macros.set(Vec::new());
        }
        let textual_slot = OnceLock::new();
        let has_textuals = !sec.textual_raw.is_empty();
        if !has_textuals {
            let _ = textual_slot.set(Vec::new());
        } else if macro_count == 0 {
            initial.push(DagTask::Textuals {
                item_idx,
                weight: sec.textual_raw.len() * 24,
            });
        }

        let mut outline_tasks: Vec<DagTask> = sec
            .outline_slices
            .iter()
            .enumerate()
            .map(|(pos, &slice)| DagTask::Outline {
                item_idx,
                pos,
                weight: slice.len(),
            })
            .collect();
        outline_tasks.sort_unstable_by_key(|t| t.weight());

        let outline_deps =
            usize::from(file_count > 0) + usize::from(macro_count > 0 || has_textuals);
        if outline_deps == 0 {
            initial.extend(outline_tasks.iter().copied());
        }

        let mut name_tables = Vec::with_capacity(sec.name_slices.len());
        let mut scope_chunks = Vec::with_capacity(sec.name_slices.len());
        let mut pending_names = Vec::new();
        for (pos, &slice) in sec.name_slices.iter().enumerate() {
            let mut nreader = Reader::new(slice);
            let table = split_names_crate_scopes(&mut nreader)?;
            let body_slice = &slice[slice.len() - nreader.len()..];
            if table.total_bytes >= 8 * 1024 && table.count >= 2 && crate::jobs() > 1 {
                let ranges = chunk_scope_ranges(&table);
                scope_chunks.push((0..ranges.len()).map(|_| OnceLock::new()).collect());
                for (cidx, (start, end, weight)) in ranges.into_iter().enumerate() {
                    pending_names.push(DagTask::ScopeChunk {
                        item_idx,
                        pos,
                        cidx,
                        start,
                        end,
                        weight,
                    });
                }
                pending_names.push(DagTask::NamesBody {
                    item_idx,
                    pos,
                    weight: body_slice.len(),
                });
            } else {
                scope_chunks.push(Vec::new());
                pending_names.push(DagTask::NamesFull {
                    item_idx,
                    pos,
                    weight: table.total_bytes + body_slice.len(),
                });
            }
            name_tables.push((table, body_slice));
        }
        pending_names.sort_unstable_by_key(|t| t.weight());
        let name_task_count = pending_names.len();
        if sym_chunk_count == 0 {
            initial.extend(pending_names.iter().copied());
            pending_names.clear();
        }

        total_tasks += sym_chunk_count
            + file_count
            + macro_count
            + usize::from(has_textuals)
            + crate_count * 2
            + name_task_count;

        states.push(ItemState {
            sec,
            file_slots: (0..file_count).map(|_| OnceLock::new()).collect(),
            all_files,
            file_children,
            file_macros,
            files_rem: AtomicUsize::new(file_count),
            macro_slots: (0..macro_count).map(|_| OnceLock::new()).collect(),
            all_macros,
            macros_rem: AtomicUsize::new(macro_count),
            textual_slot,
            outline_deps: AtomicUsize::new(outline_deps),
            sym_chunks: item_sym_chunks,
            sym_chunk_slots,
            sym_rem: AtomicUsize::new(sym_chunk_count),
            sym_remap,
            name_tables,
            pending_names,
            outline_tasks,
            outline_slots: (0..crate_count).map(|_| OnceLock::new()).collect(),
            name_slots: (0..crate_count).map(|_| OnceLock::new()).collect(),
            impl_slots: (0..crate_count).map(|_| OnceLock::new()).collect(),
            scope_chunks,
        });
    }

    initial.sort_unstable_by(|a, b| b.weight().cmp(&a.weight()));
    let ctx = DagCtx {
        inputs: &inputs,
        states,
        syms,
        initial,
        next_initial: AtomicUsize::new(0),
        queue: Mutex::new(Vec::new()),
        ready_cv: Condvar::new(),
        tasks_remaining: AtomicUsize::new(total_tasks),
        abort: AtomicBool::new(false),
        first_err: Mutex::new(None),
    };
    if total_tasks > 0 {
        let workers = crate::jobs().min(total_tasks);
        if workers <= 1 {
            worker_loop(&ctx);
        } else {
            crate::sync::scope(|scope| {
                let handles: Vec<_> = (0..workers - 1)
                    .map(|_| scope.spawn(|| worker_loop(&ctx)))
                    .collect();
                worker_loop(&ctx);
                for handle in handles {
                    if let Err(payload) = handle.join() {
                        std::panic::resume_unwind(payload);
                    }
                }
            });
        }
    }
    if let Some(err) = crate::sync::into_inner(ctx.first_err) {
        return Err(err);
    }

    let mut out = Vec::with_capacity(inputs.len());
    for (input, state) in inputs.iter().zip(ctx.states) {
        let outline_crates = input
            .cached_crates
            .iter()
            .zip(state.outline_slots)
            .map(|(&k, slot)| {
                Ok((
                    k,
                    slot.into_inner().context("missing decoded outline crate")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut name_crates = Vec::with_capacity(input.cached_crates.len());
        for ((&k, nslot), chunks) in input
            .cached_crates
            .iter()
            .zip(state.name_slots)
            .zip(state.scope_chunks)
        {
            let mut krate = nslot.into_inner().context("missing decoded names crate")?;
            if !chunks.is_empty() {
                let mut decoded_chunks = Vec::with_capacity(chunks.len());
                let mut total_scopes = 0usize;
                for cslot in chunks {
                    let scopes = cslot.into_inner().context("missing decoded scope chunk")?;
                    total_scopes += scopes.len();
                    decoded_chunks.push(scopes);
                }
                let mut scopes = Vec::with_capacity(total_scopes);
                for chunk in decoded_chunks {
                    scopes.extend(chunk);
                }
                krate.scopes = scopes;
            }
            name_crates.push((k, krate));
        }

        let mut heads = HashMap::default();
        for (&k, islot) in input.cached_crates.iter().zip(state.impl_slots) {
            if let Some(index) = islot.into_inner().context("missing decoded impl index")? {
                heads.insert(k as u32, index);
            }
        }
        out.push((
            outline_crates,
            None,
            name_crates,
            heads,
            input.env_reads.clone(),
        ));
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
        let mut sreader = Reader::new(&data[item.sections_off..]);
        sections.push(read_sections(
            &mut sreader,
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
