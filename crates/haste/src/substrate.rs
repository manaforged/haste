//! Toolchain- and dependency-scoped mmap-able substrate for frozen summaries, stencils, and symbols.

mod backing;
mod consumers;
mod inline;
mod keys;
mod merge;
pub mod noop;
pub mod relocs;
pub(crate) mod sysroot;
pub(crate) mod tally;
mod trim;
mod wire;

pub(crate) use backing::Backing;
pub(super) use backing::{bytes_to_path, path_to_bytes};
pub use inline::{InlineKey, InlineShape};
pub use keys::{
    CrateHashes, deps_key, deps_path, enabled, file_key, layer_item_key, preload_dep_files,
    preload_fixed_files, save_fixed_files, sysroot_cache_path, sysroot_item_key, sysroot_key,
    sysroot_only_item_key, target_hash, toolchain_key, toolchain_path, workspace_key,
    workspace_path,
};
pub use wire::{StencilSlice, encode_relocs, encode_sites, patch_stencil};
use wire::{encode_loops, encode_unwind_drops};
pub(super) use wire::{read_u32, read_u64};

use std::cmp::Ordering;
use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use anyhow::{Context as _, Result, bail};
use haste_codegen::Code;

use consumers::{CONSUMER_REC_LEN, Consumer};

const MAGIC: [u8; 8] = *b"HSTSUB01";
const VERSION: u32 = 30;
const HEADER_LEN: usize = 72;
const SUMMARY_REC_LEN: usize = 16;
const STENCIL_REC_LEN: usize = 80;
const STENCIL_BLOB_AT: [usize; 6] = [24, 32, 40, 48, 56, 64];
const STENCIL_LAST_USED_AT: usize = 72;
const STENCIL_SAVED_REGS_AT: usize = 76;
const INSTANCE_REC_LEN: usize = 24;

#[derive(Clone, Copy, Debug)]
struct Header {
    toolchain_hash: u64,
    target_hash: u64,
    summary_count: usize,
    stencil_count: usize,
    instance_count: usize,
    summaries_off: usize,
    stencils_off: usize,
    instances_off: usize,
    blob_off: usize,
    blob_len: usize,
    generation: u32,
    consumer_count: usize,
    consumers_off: usize,
}

/// Mmap-backed frozen substrate storing summaries, stencils, and concrete instances.
pub struct Substrate {
    backing: Backing,
    header: Header,
    parent: Option<Arc<Substrate>>,
    used: Box<[AtomicBool]>,
}

impl fmt::Debug for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Substrate")
            .field(
                "toolchain_hash",
                &format_args!("{:016x}", self.header.toolchain_hash),
            )
            .field(
                "target_hash",
                &format_args!("{:016x}", self.header.target_hash),
            )
            .field("summaries", &self.header.summary_count)
            .field("stencils", &self.header.stencil_count)
            .field("instances", &self.header.instance_count)
            .field("has_parent", &self.parent.is_some())
            .finish()
    }
}

impl Substrate {
    /// Opens and memory-maps a substrate file from `path`, validating its header and section bounds.
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_backing(Backing::open(path, HEADER_LEN)?)
    }

    /// Opens the substrate for `toolchain_hash` from the toolchain cache directory.
    pub fn open_toolchain(toolchain_hash: u64) -> Result<Self> {
        let substrate = Self::open(&toolchain_path(toolchain_hash))?;
        if substrate.header.toolchain_hash != toolchain_hash {
            bail!("toolchain hash mismatch");
        }
        Ok(substrate)
    }

    fn open_scoped(path: &Path, scope_hash: u64, target_hash: u64) -> Result<Self> {
        let substrate = Self::open(path)?;
        if substrate.header.toolchain_hash != scope_hash
            || substrate.header.target_hash != target_hash
        {
            bail!("scoped substrate hash mismatch");
        }
        Ok(substrate)
    }

    /// Opens the dependency substrate for `(deps_hash, target_hash)` from the dependency cache directory.
    pub fn open_deps(deps_hash: u64, target_hash: u64) -> Result<Self> {
        Self::open_scoped(&deps_path(deps_hash), deps_hash, target_hash)
    }

    /// Opens the local workspace substrate for `(ws_hash, target_hash)` from the workspace cache directory.
    pub fn open_workspace(ws_hash: u64, target_hash: u64) -> Result<Self> {
        Self::open_scoped(&workspace_path(ws_hash), ws_hash, target_hash)
    }

    /// Chains `parent` as a fallback layer consulted after this substrate.
    pub fn with_parent(mut self, parent: Arc<Substrate>) -> Self {
        self.parent = Some(parent);
        self
    }

    fn from_backing(backing: Backing) -> Result<Self> {
        let header = parse_and_validate(backing.as_slice())?;
        let used = (0..header.stencil_count)
            .map(|_| AtomicBool::new(false))
            .collect();
        Ok(Self {
            backing,
            header,
            parent: None,
            used,
        })
    }

    /// Starts building a new frozen substrate.
    pub fn builder() -> SubstrateBuilder {
        SubstrateBuilder::default()
    }

    /// Toolchain or dependency scope fingerprint this substrate was compiled for.
    pub fn toolchain_hash(&self) -> u64 {
        self.header.toolchain_hash
    }

    /// Target triple fingerprint this substrate was compiled for.
    pub fn target_hash(&self) -> u64 {
        self.header.target_hash
    }

    /// Looks up a frozen `PolySummary` payload by `item_hash` in `O(log N)`.
    pub fn summary(&self, item_hash: u64) -> Option<&[u8]> {
        self.local_summary(item_hash)
            .or_else(|| self.parent.as_deref().and_then(|p| p.summary(item_hash)))
    }

    fn local_summary(&self, item_hash: u64) -> Option<&[u8]> {
        let data = self.backing.as_slice();
        let idx = binary_search_index(self.header.summary_count, |i| {
            let rec = self.header.summaries_off + i * SUMMARY_REC_LEN;
            read_u64(data, rec).cmp(&item_hash)
        })?;
        let rec = self.header.summaries_off + idx * SUMMARY_REC_LEN;
        let off = read_u32(data, rec + 8) as usize;
        let len = read_u32(data, rec + 12) as usize;
        self.blob_slice(off, len)
    }

    /// Looks up a frozen `Effort::Full` stencil by `(item_hash, footprint)` in `O(log N)`.
    pub fn stencil(&self, item_hash: u64, footprint: u128) -> Option<StencilSlice<'_>> {
        self.local_stencil(item_hash, footprint).or_else(|| {
            self.parent
                .as_deref()
                .and_then(|p| p.stencil(item_hash, footprint))
        })
    }

    fn local_stencil(&self, item_hash: u64, footprint: u128) -> Option<StencilSlice<'_>> {
        let data = self.backing.as_slice();
        let fp_lo = footprint as u64;
        let fp_hi = (footprint >> 64) as u64;
        let idx = binary_search_index(self.header.stencil_count, |i| {
            let rec = self.header.stencils_off + i * STENCIL_REC_LEN;
            let k_item = read_u64(data, rec);
            let k_lo = read_u64(data, rec + 8);
            let k_hi = read_u64(data, rec + 16);
            (k_item, k_hi, k_lo).cmp(&(item_hash, fp_hi, fp_lo))
        })?;
        self.used[idx].store(true, AtomicOrdering::Relaxed);
        self.stencil_at(self.header.stencils_off + idx * STENCIL_REC_LEN)
    }

    fn stencil_at(&self, rec: usize) -> Option<StencilSlice<'_>> {
        let data = self.backing.as_slice();
        let blob = |at: usize| {
            self.blob_slice(
                read_u32(data, rec + at) as usize,
                read_u32(data, rec + at + 4) as usize,
            )
        };
        let [code, sites, relocs, plan, loops, unwind_drops] = STENCIL_BLOB_AT;
        Some(StencilSlice {
            code: blob(code)?,
            sites: blob(sites)?,
            relocs: blob(relocs)?,
            plan: blob(plan)?,
            loops: blob(loops)?,
            unwind_drops: blob(unwind_drops)?,
            saved_regs: read_u32(data, rec + STENCIL_SAVED_REGS_AT),
        })
    }

    /// Iterates frozen concrete instance payloads matching `(item_hash, substs_hash)` across this layer and its ancestors.
    pub fn instance_candidates(
        &self,
        item_hash: u64,
        substs_hash: u64,
    ) -> impl Iterator<Item = &[u8]> + '_ {
        std::iter::successors(Some(self), |s| s.parent.as_deref())
            .flat_map(move |s| s.local_instance_candidates(item_hash, substs_hash))
    }

    fn local_instance_candidates(
        &self,
        item_hash: u64,
        substs_hash: u64,
    ) -> impl Iterator<Item = &[u8]> + '_ {
        let data = self.backing.as_slice();
        let count = self.header.instance_count;
        let base = self.header.instances_off;
        let start = lower_bound_index(count, |i| {
            let rec = base + i * INSTANCE_REC_LEN;
            (read_u64(data, rec), read_u64(data, rec + 8)).cmp(&(item_hash, substs_hash))
        });
        (start..count)
            .map(move |i| base + i * INSTANCE_REC_LEN)
            .take_while(move |&rec| {
                read_u64(data, rec) == item_hash && read_u64(data, rec + 8) == substs_hash
            })
            .filter_map(move |rec| {
                self.blob_slice(
                    read_u32(data, rec + 16) as usize,
                    read_u32(data, rec + 20) as usize,
                )
            })
    }

    fn blob_slice(&self, off: usize, len: usize) -> Option<&[u8]> {
        let end = off.checked_add(len)?;
        if end > self.header.blob_len {
            return None;
        }
        let abs_start = self.header.blob_off + off;
        let abs_end = abs_start + len;
        self.backing.as_slice().get(abs_start..abs_end)
    }
}

struct PendingSummary {
    item_hash: u64,
    off: u32,
    len: u32,
}

struct PendingStencil {
    item_hash: u64,
    fp_lo: u64,
    fp_hi: u64,
    code_off: u32,
    code_len: u32,
    sites_off: u32,
    sites_len: u32,
    relocs_off: u32,
    relocs_len: u32,
    plan_off: u32,
    plan_len: u32,
    loops_off: u32,
    loops_len: u32,
    unwind_off: u32,
    unwind_len: u32,
    last_used: Option<u32>,
    saved_regs: u32,
}

struct PendingInstance {
    item_hash: u64,
    substs_hash: u64,
    off: u32,
    len: u32,
}

/// Builder that packs sorted summaries, stencils, and concrete instances into a `Substrate`.
#[derive(Default)]
#[must_use]
pub struct SubstrateBuilder {
    toolchain_hash: u64,
    target_hash: u64,
    summaries: Vec<PendingSummary>,
    stencils: Vec<PendingStencil>,
    instances: Vec<PendingInstance>,
    blob: Vec<u8>,
    generation: u32,
    consumer: (u64, u32),
    consumers: Vec<Consumer>,
}

impl fmt::Debug for SubstrateBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubstrateBuilder")
            .field(
                "toolchain_hash",
                &format_args!("{:016x}", self.toolchain_hash),
            )
            .field("target_hash", &format_args!("{:016x}", self.target_hash))
            .field("summaries", &self.summaries.len())
            .field("stencils", &self.stencils.len())
            .field("instances", &self.instances.len())
            .field("blob_bytes", &self.blob.len())
            .finish()
    }
}

impl SubstrateBuilder {
    /// Sets the toolchain or dependency scope fingerprint.
    pub fn toolchain_hash(mut self, hash: u64) -> Self {
        self.toolchain_hash = hash;
        self
    }

    /// Sets the target triple fingerprint.
    pub fn target_hash(mut self, hash: u64) -> Self {
        self.target_hash = hash;
        self
    }

    /// Names the build that is saving this layer, as its workspace key and batch kind tag.
    pub fn consumer(mut self, workspace: u64, kind: u32) -> Self {
        self.consumer = (workspace, kind);
        self
    }

    fn current_consumer(&self) -> Consumer {
        Consumer {
            workspace: self.consumer.0,
            kind: self.consumer.1,
            generation: self.generation,
        }
    }

    /// Returns whether no summaries, stencils, or concrete instances have been added yet.
    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty() && self.stencils.is_empty() && self.instances.is_empty()
    }

    /// Appends a serialized `PolySummary` entry.
    pub fn add_summary(&mut self, item_hash: u64, bytes: &[u8]) -> Result<()> {
        let (off, len) = self.push_blob(bytes)?;
        self.summaries.push(PendingSummary {
            item_hash,
            off,
            len,
        });
        Ok(())
    }

    /// Appends a serialized concrete instance entry.
    pub fn add_instance(&mut self, item_hash: u64, substs_hash: u64, bytes: &[u8]) -> Result<()> {
        let (off, len) = self.push_blob(bytes)?;
        self.instances.push(PendingInstance {
            item_hash,
            substs_hash,
            off,
            len,
        });
        Ok(())
    }

    fn push_stencil(
        &mut self,
        item_hash: u64,
        footprint: u128,
        slice: StencilSlice<'_>,
        last_used: Option<u32>,
    ) -> Result<()> {
        let (code_off, code_len) = self.push_blob(slice.code)?;
        let (sites_off, sites_len) = self.push_blob(slice.sites)?;
        let (relocs_off, relocs_len) = self.push_blob(slice.relocs)?;
        let (plan_off, plan_len) = self.push_blob(slice.plan)?;
        let (loops_off, loops_len) = self.push_blob(slice.loops)?;
        let (unwind_off, unwind_len) = self.push_blob(slice.unwind_drops)?;
        self.stencils.push(PendingStencil {
            item_hash,
            fp_lo: footprint as u64,
            fp_hi: (footprint >> 64) as u64,
            code_off,
            code_len,
            sites_off,
            sites_len,
            relocs_off,
            relocs_len,
            plan_off,
            plan_len,
            loops_off,
            loops_len,
            unwind_off,
            unwind_len,
            last_used,
            saved_regs: slice.saved_regs,
        });
        Ok(())
    }

    /// Appends a compiled `haste_codegen::Code` stencil together with serialized portable `RelocPlan` bytes.
    pub fn add_code_stencil_with_plan(
        &mut self,
        item_hash: u64,
        footprint: u128,
        code: &Code,
        plan: &[u8],
    ) -> Result<()> {
        tally::tally(tally::Tally::Offered);
        let mut sites = Vec::new();
        encode_sites(code.sites(), &mut sites);
        let mut relocs = Vec::new();
        encode_relocs(code.relocs(), &mut relocs);
        let mut loops = Vec::new();
        encode_loops(code.loops(), &mut loops);
        let mut unwind_drops = Vec::new();
        encode_unwind_drops(code.unwind_drops(), &mut unwind_drops);
        self.push_stencil(
            item_hash,
            footprint,
            StencilSlice {
                code: code.bytes(),
                sites: &sites,
                relocs: &relocs,
                plan,
                loops: &loops,
                unwind_drops: &unwind_drops,
                saved_regs: code.saved_regs(),
            },
            None,
        )
    }

    /// Shrinks the builder's backing payload buffer to fit its current length.
    pub fn shrink_blob(&mut self) {
        self.blob.shrink_to_fit();
    }

    fn push_blob(&mut self, bytes: &[u8]) -> Result<(u32, u32)> {
        push_aligned_blob(&mut self.blob, bytes)
    }
}

fn aligned8(len: usize) -> usize {
    (len + 7) & !7
}

fn push_aligned_blob(blob: &mut Vec<u8>, bytes: &[u8]) -> Result<(u32, u32)> {
    let off = u32::try_from(blob.len()).context("substrate blob exceeds 4 GiB")?;
    let len = u32::try_from(bytes.len()).context("substrate entry exceeds 4 GiB")?;
    blob.extend_from_slice(bytes);
    blob.resize(aligned8(blob.len()), 0);
    Ok((off, len))
}

fn parse_and_validate(data: &[u8]) -> Result<Header> {
    if data.len() < HEADER_LEN {
        bail!("truncated substrate header");
    }
    if data[0..8] != MAGIC {
        bail!("invalid substrate magic");
    }
    let version = read_u32(data, 8);
    if version != VERSION {
        bail!("unsupported substrate version {version}");
    }
    let generation = read_u32(data, 12);
    let toolchain_hash = read_u64(data, 16);
    let target_hash = read_u64(data, 24);
    let summary_count = usize::try_from(read_u64(data, 32)).context("summary count overflow")?;
    let stencil_count = usize::try_from(read_u64(data, 40)).context("stencil count overflow")?;
    let instance_count = usize::try_from(read_u64(data, 48)).context("instance count overflow")?;
    let blob_len = usize::try_from(read_u64(data, 56)).context("blob length overflow")?;
    let consumer_count = usize::try_from(read_u64(data, 64)).context("consumer count overflow")?;

    let summaries_off = HEADER_LEN;
    let summaries_bytes = summary_count
        .checked_mul(SUMMARY_REC_LEN)
        .context("summary table overflow")?;
    let stencils_off = summaries_off
        .checked_add(summaries_bytes)
        .context("stencil offset overflow")?;
    let stencils_bytes = stencil_count
        .checked_mul(STENCIL_REC_LEN)
        .context("stencil table overflow")?;
    let instances_off = stencils_off
        .checked_add(stencils_bytes)
        .context("instance offset overflow")?;
    let instances_bytes = instance_count
        .checked_mul(INSTANCE_REC_LEN)
        .context("instance table overflow")?;
    let consumers_off = instances_off
        .checked_add(instances_bytes)
        .context("consumer offset overflow")?;
    let consumers_bytes = consumer_count
        .checked_mul(CONSUMER_REC_LEN)
        .context("consumer table overflow")?;
    let blob_off = consumers_off
        .checked_add(consumers_bytes)
        .context("blob offset overflow")?;
    let total = blob_off
        .checked_add(blob_len)
        .context("total length overflow")?;
    if data.len() < total {
        bail!("truncated substrate payload");
    }
    Ok(Header {
        toolchain_hash,
        target_hash,
        summary_count,
        stencil_count,
        instance_count,
        summaries_off,
        stencils_off,
        instances_off,
        blob_off,
        blob_len,
        generation,
        consumer_count,
        consumers_off,
    })
}

fn lower_bound_index(len: usize, mut cmp: impl FnMut(usize) -> Ordering) -> usize {
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cmp(mid) == Ordering::Less {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn binary_search_index(len: usize, mut cmp: impl FnMut(usize) -> Ordering) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match cmp(mid) {
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid,
            Ordering::Equal => return Some(mid),
        }
    }
    None
}

fn read_substrate_toolchain_hash(path: &Path) -> Option<u64> {
    let mut file = File::open(path).ok()?;
    let mut buf = [0u8; HEADER_LEN];
    io::Read::read_exact(&mut file, &mut buf).ok()?;
    (buf[0..8] == MAGIC).then(|| read_u64(&buf, 16))
}

fn prune_dir_by_hash(dir: &Path, active_toolchain_hash: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("substrate")
            && read_substrate_toolchain_hash(&path).is_some_and(|h| h != active_toolchain_hash)
        {
            drop(std::fs::remove_file(path));
        }
    }
}

fn prune_superseded(toolchain_hash: u64) {
    if let Some(dir) = toolchain_path(toolchain_hash).parent() {
        prune_dir_by_hash(dir, toolchain_hash);
    }
}

pub(super) fn prune_stale(
    dir: &Path,
    keep: &Path,
    keep_others: usize,
    keep_file: impl Fn(&Path) -> bool,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut others: Vec<_> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path == keep || !keep_file(&path) {
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

fn prune_stale_substrates(dir: &Path, keep: &Path, keep_others: usize) {
    prune_stale(dir, keep, keep_others, |p| {
        p.extension().and_then(|e| e.to_str()) == Some("substrate")
    });
}
