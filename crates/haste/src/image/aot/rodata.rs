use std::cmp::Reverse;
use std::sync::Arc;

use anyhow::{Context, Result};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::aot::{Executable, Symbol, UNWIND_FN_REC_BYTES, UNWIND_REC12_BYTES};
use crate::lower::{RodataTarget, SymbolSource};

use super::{AotCode, AotFn, AotReach, AotState};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RodataRef {
    Rodata(u32),
    Fn(usize),
}

type RodataKey<'a> = (&'a [u8], Vec<(u32, RodataRef)>);

const NGRAM_BUCKET_CAP: usize = 16;
const SMALL_SCAN_LIMIT: usize = 65_536;

fn rodata_align(len: usize, has_relocs: bool) -> u32 {
    if len == 0 {
        return 16;
    }
    let natural = 1u32 << len.trailing_zeros().min(4);
    if has_relocs { natural.max(8) } else { natural }
}

fn push_ngram(ngrams: &mut HashMap<[u8; 4], Vec<u32>>, key: [u8; 4], pos: u32) {
    let bucket = ngrams.entry(key).or_default();
    if bucket.len() < NGRAM_BUCKET_CAP {
        bucket.push(pos);
    }
}

fn index_blob_ngrams(ngrams: &mut HashMap<[u8; 4], Vec<u32>>, base: u32, bytes: &[u8]) {
    if bytes.len() < 4 {
        return;
    }
    let last = bytes.len() - 4;
    if bytes.len() <= 4096 {
        for off in 0..=last {
            let key = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
            push_ngram(ngrams, key, base + off as u32);
        }
        return;
    }
    for off in 0..512.min(last + 1) {
        let key = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        push_ngram(ngrams, key, base + off as u32);
    }
    let tail_start = last.saturating_sub(511);
    for off in tail_start..=last {
        let key = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        push_ngram(ngrams, key, base + off as u32);
    }
    let mut off = ((base as usize).next_multiple_of(16) - base as usize).min(last);
    let mut indexed = 0usize;
    while off <= last && indexed < 256 {
        let key = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
        push_ngram(ngrams, key, base + off as u32);
        off += 16;
        indexed += 1;
    }
}

fn find_small_substring(haystack: &[u8], needle: &[u8], align: u32) -> Option<u32> {
    if haystack.len() < needle.len() {
        return None;
    }
    let limit = haystack.len().min(SMALL_SCAN_LIMIT) - needle.len();
    let step = align as usize;
    let mut pos = 0usize;
    while pos <= limit {
        if &haystack[pos..pos + needle.len()] == needle {
            return Some(pos as u32);
        }
        pos += step;
    }
    None
}

fn find_indexed_substring(
    haystack: &[u8],
    needle: &[u8],
    align: u32,
    ngrams: &HashMap<[u8; 4], Vec<u32>>,
) -> Option<u32> {
    let key = [needle[0], needle[1], needle[2], needle[3]];
    let cands = ngrams.get(&key)?;
    for &pos in cands {
        if pos % align != 0 {
            continue;
        }
        let start = pos as usize;
        if haystack.get(start..start + needle.len()) == Some(needle) {
            return Some(pos);
        }
    }
    None
}

fn place_plain_rodata<'a>(
    state: &'a AotState,
    live: &AotReach,
    rodata_offs: &mut [Option<u32>],
    exe: &mut Executable,
) {
    let mut plain: Vec<usize> = live
        .rodata_order
        .iter()
        .copied()
        .filter(|&id| state.rodata.get(id).is_some_and(|(_, r)| r.is_empty()))
        .collect();
    plain.sort_unstable_by_key(|&id| (Reverse(state.rodata[id].0.len()), id));

    let mut exact: HashMap<&'a [u8], u32> = HashMap::default();
    let mut ngrams: HashMap<[u8; 4], Vec<u32>> = HashMap::default();
    for id in plain {
        let bytes = state.rodata[id].0.as_slice();
        if bytes.is_empty() {
            rodata_offs[id] = Some(0);
            continue;
        }
        if let Some(&hit) = exact.get(bytes) {
            rodata_offs[id] = Some(hit);
            continue;
        }
        let align = rodata_align(bytes.len(), false);
        let found = if bytes.len() < 4 {
            find_small_substring(exe.rodata_bytes(), bytes, align)
        } else {
            find_indexed_substring(exe.rodata_bytes(), bytes, align, &ngrams)
        };
        let off = if let Some(pos) = found {
            pos
        } else {
            let fresh = exe.rodata(bytes, align);
            index_blob_ngrams(&mut ngrams, fresh, bytes);
            fresh
        };
        exact.insert(bytes, off);
        rodata_offs[id] = Some(off);
    }
}

fn emit_rodata_relocs(
    base: u32,
    resolved: &[(u32, RodataRef)],
    exe: &mut Executable,
    fn_relocs: &mut Vec<(u32, usize)>,
) {
    for &(reloc_off, target) in resolved {
        match target {
            RodataRef::Rodata(child_off) => {
                exe.pointer(true, base + reloc_off, Symbol::Rodata(child_off))
            }
            RodataRef::Fn(fn_idx) => fn_relocs.push((base + reloc_off, fn_idx)),
        }
    }
}

fn place_rodata_node<'a>(
    id: usize,
    state: &'a AotState,
    rodata_offs: &mut [Option<u32>],
    visiting: &mut [bool],
    seen: &mut HashMap<RodataKey<'a>, u32>,
    fn_relocs: &mut Vec<(u32, usize)>,
    exe: &mut Executable,
) -> Result<u32> {
    if let Some(off) = rodata_offs[id] {
        return Ok(off);
    }
    let entry = state.rodata.get(id).context("missing rodata entry")?;
    let align = rodata_align(entry.0.len(), !entry.1.is_empty());
    if visiting[id] {
        let off = exe.rodata(&entry.0, align);
        rodata_offs[id] = Some(off);
        return Ok(off);
    }
    visiting[id] = true;
    let mut resolved = Vec::with_capacity(entry.1.len());
    for reloc in &entry.1 {
        let target = match reloc.target {
            RodataTarget::Rodata(child) => RodataRef::Rodata(place_rodata_node(
                child.0 as usize,
                state,
                rodata_offs,
                visiting,
                seen,
                fn_relocs,
                exe,
            )?),
            RodataTarget::Fn(fn_idx) => RodataRef::Fn(fn_idx),
        };
        resolved.push((reloc.offset, target));
    }
    visiting[id] = false;
    if let Some(off) = rodata_offs[id] {
        emit_rodata_relocs(off, &resolved, exe, fn_relocs);
        return Ok(off);
    }
    let off = match seen.entry((entry.0.as_slice(), resolved)) {
        std::collections::hash_map::Entry::Occupied(hit) => *hit.get(),
        std::collections::hash_map::Entry::Vacant(slot) => {
            let fresh = exe.rodata(&entry.0, align);
            emit_rodata_relocs(fresh, &slot.key().1, exe, fn_relocs);
            slot.insert(fresh);
            fresh
        }
    };
    rodata_offs[id] = Some(off);
    Ok(off)
}

pub(super) fn emit_rodata(
    state: &AotState,
    live: &AotReach,
    exe: &mut Executable,
) -> Result<(Vec<Option<u32>>, Vec<(u32, usize)>)> {
    let mut rodata_offs = vec![None; state.rodata.len()];
    place_plain_rodata(state, live, &mut rodata_offs, exe);
    let mut visiting = vec![false; state.rodata.len()];
    let mut seen: HashMap<RodataKey<'_>, u32> = HashMap::default();
    let mut fn_relocs = Vec::new();
    for &id in &live.rodata_order {
        place_rodata_node(
            id,
            state,
            &mut rodata_offs,
            &mut visiting,
            &mut seen,
            &mut fn_relocs,
            exe,
        )?;
    }
    Ok((rodata_offs, fn_relocs))
}

pub(super) fn emit_unwind_table(
    state: &AotState,
    live: &AotReach,
    (roots, canon): (&[usize], &[usize]),
    (fn_offs, dict_bodies): (&[Option<u32>], &HashMap<usize, u32>),
    exe: &mut Executable,
) {
    let mut seen: HashSet<usize> = HashSet::default();
    let mut seen_dict: HashSet<usize> = HashSet::default();
    let mut fns: Vec<(u32, u32, u32, Vec<(i32, i32, u32)>)> = Vec::new();
    let mut sites: Vec<(u32, u32, u32)> = Vec::new();
    let push_fn = |base_off: u32,
                   code: &AotCode,
                   drops: Vec<(i32, i32, u32)>,
                   sites: &mut Vec<(u32, u32, u32)>,
                   fns: &mut Vec<(u32, u32, u32, Vec<(i32, i32, u32)>)>| {
        for s in code.sites() {
            sites.push((base_off + s.ret, base_off + s.handler, s.frame_offset));
        }
        if !drops.is_empty() || code.saved_regs() != 0 {
            fns.push((
                base_off,
                base_off + code.bytes().len() as u32,
                code.saved_regs(),
                drops,
            ));
        }
    };
    for &idx in &live.order {
        let leader = canon[roots[idx]];
        if !seen.insert(leader) {
            continue;
        }
        match &state.functions[leader] {
            Some(AotFn::Direct { code, relocs }) => {
                let Some(fn_start) = fn_offs.get(leader).copied().flatten() else {
                    continue;
                };
                let mut drops = Vec::with_capacity(code.unwind_drops().len());
                for d in code.unwind_drops() {
                    if let Some(&SymbolSource::Slot(callee_idx)) = relocs.get(d.drop_sym as usize)
                        && let Some(drop_off) = fn_offs.get(callee_idx).copied().flatten()
                    {
                        let arg = u32::from(d.arg.as_u8());
                        debug_assert_eq!(drop_off & 3, 0);
                        debug_assert!(arg < 4);
                        drops.push((d.local_off, d.flag_off, drop_off | arg));
                    }
                }
                push_fn(fn_start, code, drops, &mut sites, &mut fns);
            }
            Some(AotFn::Dict { body, .. }) => {
                let ptr = Arc::as_ptr(body) as usize;
                if seen_dict.insert(ptr)
                    && let Some(&body_off) = dict_bodies.get(&ptr)
                {
                    push_fn(body_off, body, Vec::new(), &mut sites, &mut fns);
                }
            }
            _ => {}
        }
    }
    if fns.is_empty() && sites.is_empty() {
        return;
    }
    fns.sort_unstable_by_key(|&(start, _, _, _)| start);
    sites.sort_unstable_by_key(|&(ret, _, _)| ret);

    let mut drop_pool: Vec<(i32, i32, u32)> = Vec::new();
    let mut drop_seen: HashMap<Vec<(i32, i32, u32)>, u32> = HashMap::default();
    let mut fn_recs: Vec<[u32; 5]> = Vec::with_capacity(fns.len());
    for (start, end, saved_regs, drops) in fns {
        let count = drops.len() as u32;
        let cursor = if count == 0 {
            0
        } else if let Some(&existing) = drop_seen.get(&drops) {
            existing
        } else {
            let at = drop_pool.len() as u32;
            drop_pool.extend_from_slice(&drops);
            drop_seen.insert(drops, at);
            at
        };
        fn_recs.push([start, end, cursor, count, saved_regs]);
    }

    let sites_off = 24 + fn_recs.len() * UNWIND_FN_REC_BYTES + drop_pool.len() * UNWIND_REC12_BYTES;
    let mut buf = Vec::with_capacity(sites_off + sites.len() * UNWIND_REC12_BYTES);
    buf.extend_from_slice(&(fn_recs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(sites.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(sites_off as u64).to_le_bytes());
    for rec in fn_recs {
        for word in rec {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }
    for (local_off, flag_off, drop_word) in drop_pool {
        for word in [local_off as u32, flag_off as u32, drop_word] {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }
    for (ret, handler, frame_offset) in sites {
        for word in [ret, handler, frame_offset] {
            buf.extend_from_slice(&word.to_le_bytes());
        }
    }
    exe.unwind_table_off = Some(exe.rodata(&buf, 8));
}
