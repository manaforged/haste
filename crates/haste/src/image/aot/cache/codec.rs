use std::hash::{Hash as _, Hasher as _};
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use haste_codegen::{CodeUnwindDrop, RelocKind, Site, StencilReloc, UnwindArg};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::graph::store::{file_stamp, write_new_with};
use crate::image::aot::AotCode;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::edit::hash_outside_bodies;
use crate::sema::substs::Substs;
use crate::sema::summary::codec::{Cursor, write_u32, write_u64};
use crate::sema::ty::{Kind, Ty, Types};
use crate::substrate::{encode_relocs, encode_sites, hash_crate_env, sysroot_item_key};
use crate::sync::lock;

use super::relocs::{hash_str, item_check_key, stable_file_key};
use super::tables::BundleSections;
use super::{AotConfigKey, CACHE_MAGIC, CACHE_VERSION, HEADER_LEN, RECORD_LEN};

const CODE_HEADER_LEN: usize = 28;

const RELOC_KINDS: [RelocKind; 9] = [
    RelocKind::Abs32,
    RelocKind::Abs64,
    RelocKind::Arm64Movw64,
    RelocKind::PcRel32,
    RelocKind::Arm64Call26,
    RelocKind::Arm64AdrpAdd,
    RelocKind::X64RipRel32,
    RelocKind::ThreadOff32,
    RelocKind::Arm64TlvCell,
];

pub(super) struct LoadedHeader {
    pub(super) items_off: usize,
    pub(super) items_count: usize,
    pub(super) tys_off: usize,
    pub(super) tys_count: usize,
    pub(super) tys_len: usize,
    pub(super) rodatas_off: usize,
    pub(super) rodatas_count: usize,
    pub(super) vtables_off: usize,
    pub(super) vtables_count: usize,
    pub(super) codes_off: usize,
    pub(super) codes_count: usize,
    pub(super) records_off: usize,
    pub(super) records_count: usize,
    pub(super) blob_off: usize,
    pub(super) blob_len: usize,
}

impl LoadedHeader {
    pub(super) fn empty() -> Self {
        Self {
            items_off: HEADER_LEN,
            items_count: 0,
            tys_off: HEADER_LEN,
            tys_count: 0,
            tys_len: 0,
            rodatas_off: HEADER_LEN,
            rodatas_count: 0,
            vtables_off: HEADER_LEN,
            vtables_count: 0,
            codes_off: HEADER_LEN,
            codes_count: 0,
            records_off: HEADER_LEN,
            records_count: 0,
            blob_off: HEADER_LEN,
            blob_len: 0,
        }
    }
}

pub(in crate::image) struct CacheInstanceKey {
    pub(super) key_hi: u64,
    pub(super) key_lo: u64,
    pub(super) item_key: u64,
    pub(super) substs_hash: u64,
    pub(super) stamps_digest: u64,
    pub(super) flags: u32,
}

#[inline]
pub(super) fn read_u32_le(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

#[inline]
pub(super) fn read_u64_le(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        data[at],
        data[at + 1],
        data[at + 2],
        data[at + 3],
        data[at + 4],
        data[at + 5],
        data[at + 6],
        data[at + 7],
    ])
}

pub(super) fn write_bundle(
    path: &Path,
    config: AotConfigKey,
    sec: &BundleSections,
) -> std::io::Result<()> {
    let items_off = HEADER_LEN as u32;
    let tys_off = items_off + sec.items_bytes.len() as u32;
    let rodatas_off = tys_off + sec.tys_bytes.len() as u32;
    let vtables_off = rodatas_off + (sec.rodatas.len() * 16) as u32;
    let codes_off = vtables_off + (sec.vtables.len() * 8) as u32;
    let records_off = codes_off + (sec.codes.len() * 8) as u32;
    let blob_off = records_off + (sec.records.len() * RECORD_LEN) as u32;
    let blob_len = sec.blob.len() as u32;
    write_new_with(path, |writer| {
        writer.write_all(&CACHE_MAGIC)?;
        writer.write_all(&CACHE_VERSION.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        writer.write_all(&config.config_digest.to_le_bytes())?;
        writer.write_all(&config.target_hash.to_le_bytes())?;
        writer.write_all(&config.flags_bits.to_le_bytes())?;
        writer.write_all(&config.triple_hash.to_le_bytes())?;
        for word in [
            items_off,
            sec.items_count,
            tys_off,
            sec.tys_count,
            sec.tys_bytes.len() as u32,
            rodatas_off,
            sec.rodatas.len() as u32,
            vtables_off,
            sec.vtables.len() as u32,
            codes_off,
            sec.codes.len() as u32,
            records_off,
            sec.records.len() as u32,
            blob_off,
            blob_len,
            0,
        ] {
            writer.write_all(&word.to_le_bytes())?;
        }
        writer.write_all(&sec.items_bytes)?;
        writer.write_all(&sec.tys_bytes)?;
        for r in &sec.rodatas {
            for word in [r.data_off, r.data_len, r.reloc_off, r.reloc_count] {
                writer.write_all(&word.to_le_bytes())?;
            }
        }
        for v in &sec.vtables {
            writer.write_all(&v.entry_off.to_le_bytes())?;
            writer.write_all(&v.entry_count.to_le_bytes())?;
        }
        for c in &sec.codes {
            writer.write_all(&c.code_off.to_le_bytes())?;
            writer.write_all(&c.code_len.to_le_bytes())?;
        }
        for &(hi, lo, off, len) in &sec.records {
            writer.write_all(&lo.to_le_bytes())?;
            writer.write_all(&hi.to_le_bytes())?;
            writer.write_all(&off.to_le_bytes())?;
            writer.write_all(&len.to_le_bytes())?;
        }
        writer.write_all(&sec.blob)
    })
}

pub(super) fn parse_bundle_header(data: &[u8], config: AotConfigKey) -> Option<LoadedHeader> {
    if data.len() < HEADER_LEN || data[0..8] != CACHE_MAGIC {
        return None;
    }
    if read_u32_le(data, 8) != CACHE_VERSION {
        return None;
    }
    if read_u64_le(data, 16) != config.config_digest
        || read_u64_le(data, 24) != config.target_hash
        || read_u64_le(data, 32) != config.flags_bits
        || read_u64_le(data, 40) != config.triple_hash
    {
        return None;
    }
    let items_off = read_u32_le(data, 48) as usize;
    let items_count = read_u32_le(data, 52) as usize;
    let tys_off = read_u32_le(data, 56) as usize;
    let tys_count = read_u32_le(data, 60) as usize;
    let tys_len = read_u32_le(data, 64) as usize;
    let rodatas_off = read_u32_le(data, 68) as usize;
    let rodatas_count = read_u32_le(data, 72) as usize;
    let vtables_off = read_u32_le(data, 76) as usize;
    let vtables_count = read_u32_le(data, 80) as usize;
    let codes_off = read_u32_le(data, 84) as usize;
    let codes_count = read_u32_le(data, 88) as usize;
    let records_off = read_u32_le(data, 92) as usize;
    let records_count = read_u32_le(data, 96) as usize;
    let blob_off = read_u32_le(data, 100) as usize;
    let blob_len = read_u32_le(data, 104) as usize;
    let check_range =
        |off: usize, len: usize| off.checked_add(len).filter(|&end| end <= data.len());
    check_range(items_off, items_count.checked_mul(24)?)?;
    check_range(tys_off, tys_len)?;
    check_range(rodatas_off, rodatas_count.checked_mul(16)?)?;
    check_range(vtables_off, vtables_count.checked_mul(8)?)?;
    check_range(codes_off, codes_count.checked_mul(8)?)?;
    check_range(records_off, records_count.checked_mul(RECORD_LEN)?)?;
    check_range(blob_off, blob_len)?;
    Some(LoadedHeader {
        items_off,
        items_count,
        tys_off,
        tys_count,
        tys_len,
        rodatas_off,
        rodatas_count,
        vtables_off,
        vtables_count,
        codes_off,
        codes_count,
        records_off,
        records_count,
        blob_off,
        blob_len,
    })
}

pub(super) fn find_record(
    data: &[u8],
    base: usize,
    count: usize,
    target: (u64, u64),
) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = count;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let rec = base + mid * RECORD_LEN;
        let key_lo = read_u64_le(data, rec);
        let key_hi = read_u64_le(data, rec + 8);
        match (key_hi, key_lo).cmp(&target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(mid),
        }
    }
    None
}

pub(super) fn encode_code_blob(code: &AotCode) -> Option<Vec<u8>> {
    let mut sites = Vec::new();
    encode_sites(code.sites(), &mut sites);
    let mut relocs = Vec::new();
    encode_relocs(code.relocs(), &mut relocs);
    let mut loops = Vec::with_capacity(code.loops().len() * 8);
    for r in code.loops() {
        write_u32(&mut loops, r.start);
        write_u32(&mut loops, r.end);
    }
    let mut lines = Vec::with_capacity(code.lines().len() * 12);
    for &(off, loc) in code.lines() {
        write_u32(&mut lines, off);
        write_u64(&mut lines, loc);
    }
    let mut drops = Vec::with_capacity(code.unwind_drops().len() * 16);
    for d in code.unwind_drops() {
        write_u32(&mut drops, d.local_off as u32);
        write_u32(&mut drops, d.flag_off as u32);
        write_u32(&mut drops, d.drop_sym);
        write_u32(&mut drops, u32::from(d.arg.as_u8()));
    }
    let bytes = code.bytes();
    let total = CODE_HEADER_LEN
        .checked_add(bytes.len())?
        .checked_add(sites.len())?
        .checked_add(relocs.len())?
        .checked_add(loops.len())?
        .checked_add(lines.len())?
        .checked_add(drops.len())?;
    let mut out = Vec::with_capacity(total);
    write_u32(&mut out, code.saved_regs());
    write_u32(&mut out, u32::try_from(bytes.len()).ok()?);
    write_u32(&mut out, u32::try_from(sites.len()).ok()?);
    write_u32(&mut out, u32::try_from(relocs.len()).ok()?);
    write_u32(&mut out, u32::try_from(loops.len()).ok()?);
    write_u32(&mut out, u32::try_from(lines.len()).ok()?);
    write_u32(&mut out, u32::try_from(drops.len()).ok()?);
    out.extend_from_slice(bytes);
    out.extend_from_slice(&sites);
    out.extend_from_slice(&relocs);
    out.extend_from_slice(&loops);
    out.extend_from_slice(&lines);
    out.extend_from_slice(&drops);
    Some(out)
}

pub(super) fn decode_code_blob(blob: &[u8]) -> Option<Arc<AotCode>> {
    if blob.len() < CODE_HEADER_LEN {
        return None;
    }
    let saved_regs = read_u32_le(blob, 0);
    let bytes_len = read_u32_le(blob, 4) as usize;
    let sites_len = read_u32_le(blob, 8) as usize;
    let relocs_len = read_u32_le(blob, 12) as usize;
    let loops_len = read_u32_le(blob, 16) as usize;
    let lines_len = read_u32_le(blob, 20) as usize;
    let drops_len = read_u32_le(blob, 24) as usize;
    let mut cur = Cursor::new(blob.get(CODE_HEADER_LEN..)?);
    let bytes = cur.take(bytes_len).ok()?.to_vec();
    let sites = decode_sites(cur.take(sites_len).ok()?)?;
    let relocs = decode_relocs(cur.take(relocs_len).ok()?)?;
    let loops = decode_loops(cur.take(loops_len).ok()?)?;
    let lines = decode_lines(cur.take(lines_len).ok()?)?;
    let unwind_drops = decode_unwind_drops(cur.take(drops_len).ok()?)?;
    if CODE_HEADER_LEN + cur.pos != blob.len() {
        return None;
    }
    Some(Arc::new(AotCode {
        bytes,
        sites,
        relocs,
        loops,
        lines,
        unwind_drops,
        saved_regs,
    }))
}

fn decode_sites(raw: &[u8]) -> Option<Vec<Site>> {
    if raw.len() % 12 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len() / 12);
    for at in (0..raw.len()).step_by(12) {
        out.push(Site {
            ret: read_u32_le(raw, at),
            handler: read_u32_le(raw, at + 4),
            frame_offset: read_u32_le(raw, at + 8),
        });
    }
    Some(out)
}

fn decode_relocs(raw: &[u8]) -> Option<Vec<StencilReloc>> {
    if raw.len() % 12 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len() / 12);
    for at in (0..raw.len()).step_by(12) {
        let kind = RELOC_KINDS.get(raw[at + 8] as usize).copied()?;
        out.push(StencilReloc {
            offset: read_u32_le(raw, at),
            symbol: read_u32_le(raw, at + 4),
            kind,
        });
    }
    Some(out)
}

fn decode_loops(raw: &[u8]) -> Option<Vec<std::ops::Range<u32>>> {
    if raw.len() % 8 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len() / 8);
    for at in (0..raw.len()).step_by(8) {
        out.push(read_u32_le(raw, at)..read_u32_le(raw, at + 4));
    }
    Some(out)
}

fn decode_lines(raw: &[u8]) -> Option<Vec<(u32, u64)>> {
    if raw.len() % 12 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len() / 12);
    for at in (0..raw.len()).step_by(12) {
        out.push((read_u32_le(raw, at), read_u64_le(raw, at + 4)));
    }
    Some(out)
}

fn decode_unwind_drops(raw: &[u8]) -> Option<Vec<CodeUnwindDrop>> {
    if raw.len() % 16 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(raw.len() / 16);
    for at in (0..raw.len()).step_by(16) {
        let arg = u8::try_from(read_u32_le(raw, at + 12))
            .ok()
            .and_then(UnwindArg::from_u8)?;
        out.push(CodeUnwindDrop {
            local_off: read_u32_le(raw, at) as i32,
            flag_off: read_u32_le(raw, at + 4) as i32,
            drop_sym: read_u32_le(raw, at + 8),
            arg,
        });
    }
    Some(out)
}

pub(super) struct Stamps {
    iface: Vec<u64>,
    files: Vec<Vec<u64>>,
}

impl Stamps {
    pub(super) fn compute(outline: &Outline, tc_key: u64, target_key: u64) -> Self {
        let n = outline.crates.len();
        let mut iface = vec![0u64; n];
        let mut files = vec![Vec::new(); n];
        let mut state = vec![0u8; n];
        for idx in 0..n {
            resolve_crate_stamps(
                outline, tc_key, target_key, idx, &mut iface, &mut files, &mut state,
            );
        }
        Self { iface, files }
    }

    pub(super) fn iface(&self, krate: u32) -> Option<u64> {
        self.iface.get(krate as usize).copied()
    }

    pub(super) fn file(&self, krate: u32, file: u32) -> Option<u64> {
        let per = self.files.get(krate as usize)?;
        if per.is_empty() {
            self.iface(krate)
        } else {
            per.get(file as usize).copied()
        }
    }
}

fn resolve_crate_stamps(
    outline: &Outline,
    tc_key: u64,
    target_key: u64,
    idx: usize,
    iface: &mut [u64],
    files: &mut [Vec<u64>],
    state: &mut [u8],
) -> u64 {
    if state[idx] == 2 {
        return iface[idx];
    }
    if state[idx] == 1 {
        return 0;
    }
    state[idx] = 1;
    let mut h = FxHasher::default();
    hash_crate_env(outline, idx, tc_key, target_key, &mut h);
    let mut local_raw = Vec::new();
    if let (Some(tree), Some(root)) = (outline.crates.get(idx), outline.roots.get(idx)) {
        if tree.env.fixed {
            outline.pins.get(idx).copied().flatten().hash(&mut h);
            (&root.path, file_stamp(&root.path)).hash(&mut h);
            tree.env.features.hash(&mut h);
            tree.env.vars.hash(&mut h);
            if let Some(script) = &tree.env.script {
                script.key_parts().hash(&mut h);
            }
        } else {
            tree.env.features.hash(&mut h);
            tree.env.vars.hash(&mut h);
            if let Some(script) = &tree.env.script {
                script.key_parts().hash(&mut h);
            }
            let base_dir = root.path.parent().unwrap_or(Path::new(""));
            let mut plain_iface = Vec::new();
            let mut expanded_iface = Vec::new();
            local_raw.reserve(tree.files.len());
            for f in &tree.files {
                let rel = f
                    .source
                    .path
                    .strip_prefix(base_dir)
                    .unwrap_or(&f.source.path);
                let text_hash = hash_str(&f.source.text);
                if f.expanded {
                    let mut eh = FxHasher::default();
                    (rel, f.expand_origin, f.source.offset, text_hash).hash(&mut eh);
                    let exp = eh.finish();
                    expanded_iface.push((rel, f.expand_origin, f.source.offset, text_hash));
                    local_raw.push((rel, exp));
                } else {
                    let mut oh = FxHasher::default();
                    hash_outside_bodies(f, &mut oh);
                    let out_hash = oh.finish();
                    plain_iface.push((rel, out_hash));
                    local_raw.push((rel, text_hash));
                }
            }
            plain_iface.sort_unstable();
            plain_iface.hash(&mut h);
            expanded_iface.sort_unstable();
            expanded_iface.hash(&mut h);
            let mut inc: Vec<_> = outline
                .included(idx)
                .into_iter()
                .map(|(p, _)| {
                    let s = file_stamp(&p);
                    (p, s)
                })
                .collect();
            inc.sort_unstable();
            inc.dedup();
            inc.hash(&mut h);
            let mut env_reads: Vec<(String, Option<String>)> = {
                let guard = lock(&tree.env.env_reads);
                guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            };
            env_reads.sort();
            env_reads.hash(&mut h);
        }
    }
    let mut deps: Vec<(&str, u64)> = outline
        .deps
        .get(idx)
        .into_iter()
        .flatten()
        .map(|(name, dep)| {
            (
                name.as_str(),
                resolve_crate_stamps(outline, tc_key, target_key, *dep, iface, files, state),
            )
        })
        .collect();
    deps.sort_unstable_by(|a, b| a.0.cmp(b.0));
    deps.hash(&mut h);
    let digest = h.finish();
    iface[idx] = digest;
    if !local_raw.is_empty() {
        files[idx] = local_raw
            .into_iter()
            .map(|(rel, body_hash)| {
                let mut fh = FxHasher::default();
                (digest, rel, body_hash).hash(&mut fh);
                fh.finish()
            })
            .collect();
    }
    state[idx] = 2;
    digest
}

pub(super) fn stable_entry_item_key(
    outline: &Outline,
    stamps: &Stamps,
    item: ItemRef,
) -> Option<u64> {
    if outline.is_fixed(item.krate) {
        return sysroot_item_key(outline, item);
    }
    let (base, sub) = match item.outer() {
        Some((outer, sub)) => (outer, Some(sub)),
        None => (item, None),
    };
    let file_digest = stable_file_key(outline, base.krate, base.file)?;
    let base_key = item_check_key(outline, stamps, file_digest, base)?;
    let mut h = FxHasher::default();
    base_key.hash(&mut h);
    sub.hash(&mut h);
    Some(h.finish())
}

fn collect_substs_crates(types: &Types, substs: Substs, out: &mut Vec<u32>) {
    let mut seen = HashSet::default();
    let mut stack: Vec<Ty> = types.substs(substs).to_vec();
    while let Some(ty) = stack.pop() {
        if ty == Ty::NONE || !seen.insert(ty) {
            continue;
        }
        match *types.kind(ty) {
            Kind::Adt(item, s) | Kind::FnDef(item, s) | Kind::Closure(item, s, _) => {
                out.push(item.krate);
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Dyn(item, s, m, a) => {
                out.push(item.krate);
                for x in types.items(m).iter().chain(types.items(a)) {
                    out.push(x.krate);
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Proj(id) => {
                let p = types.proj(id);
                out.push(p.of.krate);
                out.push(p.assoc.krate);
                stack.push(p.self_ty);
                stack.extend_from_slice(types.substs(p.args));
                stack.extend_from_slice(types.substs(p.own));
            }
            ref other => {
                for child in other.children(types) {
                    stack.push(child);
                }
            }
        }
    }
}

pub(super) fn touched_stamps_digest(
    types: &Types,
    stamps: &Stamps,
    item: ItemRef,
    substs: Substs,
) -> Option<u64> {
    let mut krates = vec![item.krate];
    if !substs.is_empty() {
        collect_substs_crates(types, substs, &mut krates);
        krates.sort_unstable();
        krates.dedup();
    }
    let own_file_stamp = stamps.file(item.krate, item.body_owner().file)?;
    let mut h = FxHasher::default();
    own_file_stamp.hash(&mut h);
    krates.len().hash(&mut h);
    for k in krates {
        stamps.iface(k)?.hash(&mut h);
    }
    Some(h.finish())
}

pub(super) fn stable_substs_hash(
    outline: &Outline,
    types: &Types,
    stamps: &Stamps,
    substs: Substs,
    memo: &mut HashMap<Ty, u64>,
) -> Option<u64> {
    let mut h = FxHasher::default();
    let slice = types.substs(substs);
    slice.len().hash(&mut h);
    for &ty in slice {
        stable_ty_hash(outline, types, stamps, ty, memo)?.hash(&mut h);
    }
    Some(h.finish())
}

fn stable_ty_hash(
    outline: &Outline,
    types: &Types,
    stamps: &Stamps,
    ty: Ty,
    memo: &mut HashMap<Ty, u64>,
) -> Option<u64> {
    if ty == Ty::NONE {
        return Some(0);
    }
    if let Some(&cached) = memo.get(&ty) {
        return Some(cached);
    }
    let mut h = FxHasher::default();
    match *types.kind(ty) {
        Kind::Prim(p) => (1u8, p).hash(&mut h),
        Kind::Never => 2u8.hash(&mut h),
        Kind::Tuple(s) => {
            3u8.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
        }
        Kind::Ref(m, i) => {
            4u8.hash(&mut h);
            m.hash(&mut h);
            stable_ty_hash(outline, types, stamps, i, memo)?.hash(&mut h);
        }
        Kind::Ptr(m, i) => {
            5u8.hash(&mut h);
            m.hash(&mut h);
            stable_ty_hash(outline, types, stamps, i, memo)?.hash(&mut h);
        }
        Kind::Array(e, l) => {
            6u8.hash(&mut h);
            stable_ty_hash(outline, types, stamps, e, memo)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, l, memo)?.hash(&mut h);
        }
        Kind::Slice(e) => {
            7u8.hash(&mut h);
            stable_ty_hash(outline, types, stamps, e, memo)?.hash(&mut h);
        }
        Kind::Adt(i, s) => {
            8u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, i)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
        }
        Kind::FnDef(i, s) => {
            9u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, i)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
        }
        Kind::FnPtr(p, r) => {
            10u8.hash(&mut h);
            stable_substs_hash(outline, types, stamps, p, memo)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, r, memo)?.hash(&mut h);
        }
        Kind::UnsafeFnPtr(p, r) => {
            11u8.hash(&mut h);
            stable_substs_hash(outline, types, stamps, p, memo)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, r, memo)?.hash(&mut h);
        }
        Kind::Closure(p, s, at) => {
            12u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, p)?.hash(&mut h);
            at.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
        }
        Kind::Dyn(o, s, m, a) => {
            13u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, o)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
            for &x in types.items(m).iter().chain(types.items(a)) {
                stable_entry_item_key(outline, stamps, x)?.hash(&mut h);
            }
        }
        Kind::Pat(b, range) => {
            let (lo, hi) = types.pat(range);
            (14u8, lo, hi).hash(&mut h);
            stable_ty_hash(outline, types, stamps, b, memo)?.hash(&mut h);
        }
        Kind::Value(v) => (15u8, v).hash(&mut h),
        Kind::Param(p) => (16u8, p).hash(&mut h),
        Kind::Proj(id) => {
            let d = types.proj(id);
            17u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, d.of)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, d.self_ty, memo)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, d.args, memo)?.hash(&mut h);
            stable_entry_item_key(outline, stamps, d.assoc)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, d.own, memo)?.hash(&mut h);
        }
        Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return None,
    }
    let val = h.finish();
    memo.insert(ty, val);
    Some(val)
}

pub(super) fn hash_entry_128(
    config_digest: u64,
    item_key: u64,
    substs_hash: u64,
    stamps_digest: u64,
    flags: u32,
) -> (u64, u64) {
    let mut hi = FxHasher::default();
    (config_digest, item_key, substs_hash, stamps_digest, flags).hash(&mut hi);
    let mut lo = FxHasher::default();
    (
        0x9E37_79B9_7F4A_7C15_u64,
        flags,
        stamps_digest,
        substs_hash,
        item_key,
        config_digest,
    )
        .hash(&mut lo);
    (hi.finish(), lo.finish())
}

pub(super) fn prune_stale_bundles(dir: &Path, keep: &Path, keep_others: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut others: Vec<_> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path == keep || path.extension().and_then(|e| e.to_str()) != Some("aot") {
                return None;
            }
            let meta = entry.metadata().ok()?;
            meta.is_file().then(|| {
                (
                    meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                    path,
                )
            })
        })
        .collect();
    others.sort_by(|a, b| b.0.cmp(&a.0));
    others
        .into_iter()
        .skip(keep_others)
        .for_each(|(_, p)| drop(std::fs::remove_file(p)));
}
