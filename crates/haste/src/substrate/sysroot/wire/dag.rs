use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{Context as _, Result, bail};

use crate::names::{Crate as NamesCrate, Scope, Sym, Symbols};
use crate::outline::tree::{Crate as OutlineCrate, File, Macro, Textual};
use crate::sema::impls::Index as ImplIndex;
use crate::substrate::Backing;

use super::super::EnvReads;
use super::super::impls::read_impl_index;
use super::super::names::{ScopeTable, decode_scope_range, read_names_crate_body};
use super::super::outline::read_outline_crate;
use super::super::tokens::{Reader, read_rules, u32_le};
use super::decode::{WireSections, decode_textuals, read_file};

pub(super) struct SymChunk<'a> {
    pub chunk_idx: usize,
    pub base_off: usize,
    pub ends: &'a [u8],
    pub bytes: &'a [u8],
}

pub(super) fn plan_sym_chunks<'a>(
    sections: &[WireSections<'a>],
    workers: usize,
) -> Result<Vec<Vec<SymChunk<'a>>>> {
    let total_syms: usize = sections.iter().map(|s| s.sym_count).sum();
    if total_syms == 0 {
        for sec in sections {
            if !sec.sym_blob.is_empty() {
                bail!("trailing bytes in snapshot symbol blob");
            }
        }
        return Ok(sections.iter().map(|_| Vec::new()).collect());
    }
    let chunk_syms = if total_syms >= 4096 {
        total_syms.div_ceil(workers * 2).max(256)
    } else {
        total_syms.div_ceil(workers).max(256)
    };
    let mut all_chunks = Vec::with_capacity(sections.len());
    for sec in sections {
        if sec.sym_count == 0 {
            if !sec.sym_blob.is_empty() {
                bail!("trailing bytes in snapshot symbol blob");
            }
            all_chunks.push(Vec::new());
            continue;
        }
        let last_end = u32_le(sec.sym_ends, (sec.sym_count - 1) * 4) as usize;
        if last_end != sec.sym_blob.len() {
            bail!("trailing bytes in snapshot symbol blob");
        }
        let mut chunks = Vec::with_capacity(sec.sym_count.div_ceil(chunk_syms));
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
                chunk_idx: cidx,
                base_off: prev,
                ends: &sec.sym_ends[start * 4..end * 4],
                bytes: &sec.sym_blob[prev..next_prev],
            });
            prev = next_prev;
            start = end;
            cidx += 1;
        }
        all_chunks.push(chunks);
    }
    Ok(all_chunks)
}

fn decode_sym_chunk(chunk: &SymChunk<'_>, syms: &Symbols) -> Result<Vec<Sym>> {
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
    Ok(out)
}

pub(super) fn chunk_scope_ranges(table: &ScopeTable<'_>) -> Vec<(usize, usize, usize)> {
    const TARGET_BYTES: usize = 8 * 1024;
    const MAX_SCOPES: usize = 64;
    let mut chunks = Vec::new();
    let mut chunk_start = 0usize;
    let mut chunk_base_off = 0usize;
    for (idx, c) in table.end_bytes.chunks_exact(4).enumerate() {
        let end_off = u32_le(c, 0) as usize;
        let cur_bytes = end_off - chunk_base_off;
        let cur_len = idx + 1 - chunk_start;
        if cur_bytes >= TARGET_BYTES || cur_len >= MAX_SCOPES {
            chunks.push((chunk_start, idx + 1, cur_bytes));
            chunk_start = idx + 1;
            chunk_base_off = end_off;
        }
    }
    if chunk_start < table.count {
        chunks.push((chunk_start, table.count, table.total_bytes - chunk_base_off));
    }
    chunks
}

pub(super) struct BatchInput<'a> {
    pub backing: &'a Arc<Backing>,
    pub cached_crates: &'a [usize],
    pub remap: &'a [u32],
    pub scope_bit: u32,
    pub env_reads: Vec<EnvReads>,
}

#[derive(Clone, Copy)]
pub(super) enum DagTask {
    SymChunk {
        item_idx: usize,
        chunk_idx: usize,
        weight: usize,
    },
    File {
        item_idx: usize,
        file_idx: usize,
        base_id: Option<u32>,
        weight: usize,
    },
    Macro {
        item_idx: usize,
        mac_idx: usize,
        weight: usize,
    },
    Textuals {
        item_idx: usize,
        weight: usize,
    },
    Impls {
        item_idx: usize,
        pos: usize,
        weight: usize,
    },
    Outline {
        item_idx: usize,
        pos: usize,
        weight: usize,
    },
    NamesFull {
        item_idx: usize,
        pos: usize,
        weight: usize,
    },
    NamesBody {
        item_idx: usize,
        pos: usize,
        weight: usize,
    },
    ScopeChunk {
        item_idx: usize,
        pos: usize,
        cidx: usize,
        start: usize,
        end: usize,
        weight: usize,
    },
}

impl DagTask {
    #[inline]
    pub(super) fn weight(&self) -> usize {
        match *self {
            DagTask::SymChunk { weight, .. }
            | DagTask::File { weight, .. }
            | DagTask::Macro { weight, .. }
            | DagTask::Textuals { weight, .. }
            | DagTask::Impls { weight, .. }
            | DagTask::Outline { weight, .. }
            | DagTask::NamesFull { weight, .. }
            | DagTask::NamesBody { weight, .. }
            | DagTask::ScopeChunk { weight, .. } => weight,
        }
    }
}

pub(super) struct ItemState<'a> {
    pub sec: WireSections<'a>,
    pub file_slots: Vec<OnceLock<Arc<File>>>,
    pub all_files: OnceLock<Vec<Arc<File>>>,
    pub file_children: Vec<Vec<(usize, usize)>>,
    pub file_macros: Vec<Vec<(usize, usize)>>,
    pub files_rem: AtomicUsize,
    pub macro_slots: Vec<OnceLock<Arc<Macro>>>,
    pub all_macros: OnceLock<Vec<Arc<Macro>>>,
    pub macros_rem: AtomicUsize,
    pub textual_slot: OnceLock<Vec<Arc<Textual>>>,
    pub outline_deps: AtomicUsize,
    pub sym_chunks: Vec<SymChunk<'a>>,
    pub sym_chunk_slots: Vec<Mutex<Option<Vec<Sym>>>>,
    pub sym_rem: AtomicUsize,
    pub sym_remap: OnceLock<Vec<Sym>>,
    pub name_tables: Vec<(ScopeTable<'a>, &'a [u8])>,
    pub pending_names: Vec<DagTask>,
    pub outline_tasks: Vec<DagTask>,
    pub outline_slots: Vec<OnceLock<OutlineCrate>>,
    pub name_slots: Vec<OnceLock<NamesCrate>>,
    pub impl_slots: Vec<OnceLock<Option<Arc<ImplIndex>>>>,
    pub scope_chunks: Vec<Vec<OnceLock<Vec<Scope>>>>,
}

pub(super) struct DagCtx<'a, 'b> {
    pub inputs: &'b [BatchInput<'a>],
    pub states: Vec<ItemState<'a>>,
    pub syms: &'b Symbols,
    pub initial: Vec<DagTask>,
    pub next_initial: AtomicUsize,
    pub queue: Mutex<Vec<DagTask>>,
    pub ready_cv: Condvar,
    pub tasks_remaining: AtomicUsize,
    pub abort: AtomicBool,
    pub first_err: Mutex<Option<anyhow::Error>>,
}

fn push_unlocked(ctx: &DagCtx<'_, '_>, tasks: impl IntoIterator<Item = DagTask>) {
    let mut queue = crate::sync::lock(&ctx.queue);
    let before = queue.len();
    queue.extend(tasks);
    let added = queue.len() - before;
    if added == 1 {
        ctx.ready_cv.notify_one();
    } else if added > 1 {
        ctx.ready_cv.notify_all();
    }
}

fn run_dag_task(ctx: &DagCtx<'_, '_>, task: DagTask) -> Result<()> {
    match task {
        DagTask::SymChunk {
            item_idx,
            chunk_idx,
            ..
        } => {
            let state = &ctx.states[item_idx];
            let chunk = &state.sym_chunks[chunk_idx];
            let syms = decode_sym_chunk(chunk, ctx.syms)?;
            *crate::sync::lock(&state.sym_chunk_slots[chunk.chunk_idx]) = Some(syms);
            if state.sym_rem.fetch_sub(1, Ordering::AcqRel) == 1 {
                let mut flat = Vec::with_capacity(state.sec.sym_count);
                for slot in &state.sym_chunk_slots {
                    if let Some(chunk_syms) = crate::sync::lock(slot).take() {
                        flat.extend(chunk_syms);
                    } else {
                        bail!("missing decoded symbol chunk");
                    }
                }
                let _ = state.sym_remap.set(flat);
                if !state.pending_names.is_empty() {
                    push_unlocked(ctx, state.pending_names.iter().copied());
                }
            }
        }
        DagTask::File {
            item_idx,
            file_idx,
            base_id,
            ..
        } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let (file_off, slice, _) = state.sec.file_slices[file_idx];
            let base = match base_id {
                Some(id) => Some(Arc::clone(
                    state.file_slots[id as usize]
                        .get()
                        .context("base file out of bounds")?,
                )),
                None => None,
            };
            let mut freader = Reader::new(slice);
            let file = read_file(
                &mut freader,
                slice.len(),
                input.backing,
                file_off,
                base,
                input.remap,
                input.scope_bit,
            )?;
            if !freader.is_empty() {
                bail!("trailing bytes in snapshot file");
            }
            let _ = state.file_slots[file_idx].set(Arc::new(file));
            let children = &state.file_children[file_idx];
            let macros = &state.file_macros[file_idx];
            let last_file = state.files_rem.fetch_sub(1, Ordering::AcqRel) == 1;
            let unlock_outlines = if last_file {
                let mut files = Vec::with_capacity(state.file_slots.len());
                for slot in &state.file_slots {
                    files.push(Arc::clone(slot.get().context("missing decoded file slot")?));
                }
                let _ = state.all_files.set(files);
                !state.outline_tasks.is_empty()
                    && state.outline_deps.fetch_sub(1, Ordering::AcqRel) == 1
            } else {
                false
            };
            if !children.is_empty() || !macros.is_empty() || unlock_outlines {
                let mut unlocked = Vec::with_capacity(
                    children.len()
                        + macros.len()
                        + if unlock_outlines {
                            state.outline_tasks.len()
                        } else {
                            0
                        },
                );
                for &(child_idx, weight) in children {
                    unlocked.push(DagTask::File {
                        item_idx,
                        file_idx: child_idx,
                        base_id: Some(file_idx as u32),
                        weight,
                    });
                }
                for &(mac_idx, weight) in macros {
                    unlocked.push(DagTask::Macro {
                        item_idx,
                        mac_idx,
                        weight,
                    });
                }
                if unlock_outlines {
                    unlocked.extend(state.outline_tasks.iter().copied());
                }
                unlocked.sort_unstable_by_key(|t| t.weight());
                push_unlocked(ctx, unlocked);
            }
        }
        DagTask::Macro {
            item_idx, mac_idx, ..
        } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let slice = state.sec.macro_slices[mac_idx];
            let mut mreader = Reader::new(slice);
            let file = Arc::clone(
                state.file_slots[mreader.u32()? as usize]
                    .get()
                    .context("macro file out of bounds")?,
            );
            let krate = *input
                .remap
                .get(mreader.u32()? as usize)
                .context("macro crate out of bounds")? as usize;
            let rules = read_rules(&mut mreader, input.remap)?;
            if !mreader.is_empty() {
                bail!("trailing bytes in snapshot macro");
            }
            let _ = state.macro_slots[mac_idx].set(Arc::new(Macro { file, rules, krate }));
            if state.macros_rem.fetch_sub(1, Ordering::AcqRel) == 1 {
                let mut macros = Vec::with_capacity(state.macro_slots.len());
                for slot in &state.macro_slots {
                    macros.push(Arc::clone(
                        slot.get().context("missing decoded macro slot")?,
                    ));
                }
                let _ = state.all_macros.set(macros);
                if state.sec.textual_raw.is_empty() {
                    let _ = state.textual_slot.set(Vec::new());
                    if !state.outline_tasks.is_empty()
                        && state.outline_deps.fetch_sub(1, Ordering::AcqRel) == 1
                    {
                        push_unlocked(ctx, state.outline_tasks.iter().copied());
                    }
                } else {
                    push_unlocked(
                        ctx,
                        std::iter::once(DagTask::Textuals {
                            item_idx,
                            weight: state.sec.textual_raw.len() * 24,
                        }),
                    );
                }
            }
        }
        DagTask::Textuals { item_idx, .. } => {
            let state = &ctx.states[item_idx];
            let macros = state.all_macros.get().context("missing decoded macros")?;
            let textuals = decode_textuals(&state.sec.textual_raw, macros)?;
            let _ = state.textual_slot.set(textuals);
            if !state.outline_tasks.is_empty()
                && state.outline_deps.fetch_sub(1, Ordering::AcqRel) == 1
            {
                push_unlocked(ctx, state.outline_tasks.iter().copied());
            }
        }
        DagTask::Impls { item_idx, pos, .. } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let slice = state.sec.impl_slices[pos];
            let mut creader = Reader::new(slice);
            let krate_id = input.cached_crates[pos] as u32;
            let index = read_impl_index(&mut creader, krate_id, input.remap)?;
            if !creader.is_empty() {
                bail!("trailing bytes in snapshot impl index");
            }
            let _ = state.impl_slots[pos].set(index);
        }
        DagTask::Outline { item_idx, pos, .. } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let slice = state.sec.outline_slices[pos];
            let files = state.all_files.get().context("missing decoded files")?;
            let macros = state.all_macros.get().context("missing decoded macros")?;
            let textuals = state
                .textual_slot
                .get()
                .context("missing decoded textuals")?;
            let mut creader = Reader::new(slice);
            let krate = read_outline_crate(&mut creader, files, macros, textuals, input.remap)?;
            if !creader.is_empty() {
                bail!("trailing bytes in snapshot outline crate");
            }
            let _ = state.outline_slots[pos].set(krate);
        }
        DagTask::NamesFull { item_idx, pos, .. } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let sym_remap = state
                .sym_remap
                .get()
                .context("missing decoded symbol remap")?;
            let (ref table, body_slice) = state.name_tables[pos];
            let scopes = decode_scope_range(table, 0, table.count, input.remap, sym_remap)?;
            let mut creader = Reader::new(body_slice);
            let krate = read_names_crate_body(&mut creader, scopes, input.remap, sym_remap)?;
            if !creader.is_empty() {
                bail!("trailing bytes in snapshot names crate");
            }
            let _ = state.name_slots[pos].set(krate);
        }
        DagTask::NamesBody { item_idx, pos, .. } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let sym_remap = state
                .sym_remap
                .get()
                .context("missing decoded symbol remap")?;
            let (_, body_slice) = state.name_tables[pos];
            let mut creader = Reader::new(body_slice);
            let krate = read_names_crate_body(&mut creader, Vec::new(), input.remap, sym_remap)?;
            if !creader.is_empty() {
                bail!("trailing bytes in snapshot names crate");
            }
            let _ = state.name_slots[pos].set(krate);
        }
        DagTask::ScopeChunk {
            item_idx,
            pos,
            cidx,
            start,
            end,
            ..
        } => {
            let state = &ctx.states[item_idx];
            let input = &ctx.inputs[item_idx];
            let sym_remap = state
                .sym_remap
                .get()
                .context("missing decoded symbol remap")?;
            let (ref table, _) = state.name_tables[pos];
            let scopes = decode_scope_range(table, start, end, input.remap, sym_remap)?;
            let _ = state.scope_chunks[pos][cidx].set(scopes);
        }
    }
    Ok(())
}

pub(super) fn worker_loop(ctx: &DagCtx<'_, '_>) {
    loop {
        if ctx.abort.load(Ordering::Relaxed) {
            return;
        }
        let next = {
            let idx = ctx.next_initial.fetch_add(1, Ordering::Relaxed);
            if let Some(&task) = ctx.initial.get(idx) {
                Some(task)
            } else {
                let mut queue = crate::sync::lock(&ctx.queue);
                loop {
                    if let Some(task) = queue.pop() {
                        break Some(task);
                    }
                    if ctx.tasks_remaining.load(Ordering::Acquire) == 0
                        || ctx.abort.load(Ordering::Relaxed)
                    {
                        break None;
                    }
                    queue = ctx
                        .ready_cv
                        .wait(queue)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
        };
        let Some(task) = next else {
            return;
        };
        if let Err(err) = run_dag_task(ctx, task) {
            {
                let mut err_guard = crate::sync::lock(&ctx.first_err);
                if err_guard.is_none() {
                    *err_guard = Some(err);
                }
            }
            ctx.abort.store(true, Ordering::Release);
            let _guard = crate::sync::lock(&ctx.queue);
            ctx.ready_cv.notify_all();
            return;
        }
        if ctx.tasks_remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = crate::sync::lock(&ctx.queue);
            ctx.ready_cv.notify_all();
            return;
        }
    }
}
