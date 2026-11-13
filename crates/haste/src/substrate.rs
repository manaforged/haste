//! Toolchain-scoped mmap-able substrate for frozen sysroot summaries, stencils, and symbols.

use std::cmp::Ordering;
use std::fmt;
use std::fs::File;
use std::hash::{Hash as _, Hasher as _};
use std::io::{self, Write as _};
use std::os::fd::AsRawFd as _;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use anyhow::{Context as _, Result, anyhow, bail};
use haste_codegen::{Code, RelocKind, Site, StencilReloc, Target};
use rustc_hash::FxHasher;

const MAGIC: [u8; 8] = *b"HSTSUB01";
const VERSION: u32 = 3;
const HEADER_LEN: usize = 64;
const SUMMARY_REC_LEN: usize = 16;
const STENCIL_REC_LEN: usize = 48;
const INSTANCE_REC_LEN: usize = 24;
const SITE_WIRE_LEN: usize = 12;
const RELOC_WIRE_LEN: usize = 12;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Zero-copy view of a pre-compiled `Effort::Full` stencil in a substrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilSlice<'a> {
    code: &'a [u8],
    sites: &'a [u8],
    relocs: &'a [u8],
}

impl<'a> StencilSlice<'a> {
    /// Encoded machine code bytes.
    pub fn code(&self) -> &'a [u8] {
        self.code
    }

    /// Decodes the unwind site table into `Site` records.
    pub fn decode_sites(&self) -> Result<Vec<Site>> {
        if self.sites.len() % SITE_WIRE_LEN != 0 {
            bail!("misaligned site table");
        }
        let mut out = Vec::with_capacity(self.sites.len() / SITE_WIRE_LEN);
        for at in (0..self.sites.len()).step_by(SITE_WIRE_LEN) {
            out.push(Site {
                ret: read_u32(self.sites, at),
                handler: read_u32(self.sites, at + 4),
                frame_offset: read_u32(self.sites, at + 8),
            });
        }
        Ok(out)
    }

    /// Decodes the relocation descriptor table into `StencilReloc` records.
    pub fn decode_relocs(&self) -> Result<Vec<StencilReloc>> {
        if self.relocs.len() % RELOC_WIRE_LEN != 0 {
            bail!("misaligned reloc table");
        }
        let mut out = Vec::with_capacity(self.relocs.len() / RELOC_WIRE_LEN);
        for at in (0..self.relocs.len()).step_by(RELOC_WIRE_LEN) {
            let kind = decode_reloc_kind(self.relocs[at + 8])?;
            out.push(StencilReloc {
                offset: read_u32(self.relocs, at),
                symbol: read_u32(self.relocs, at + 4),
                kind,
            });
        }
        Ok(out)
    }
}

enum Backing {
    Owned(Vec<u8>),
    Mmap { ptr: NonNull<u8>, len: usize },
}

// SAFETY: `Backing::Mmap` points to a read-only immutable mapping owned exclusively by `Backing`.
unsafe impl Send for Backing {}
// SAFETY: `Backing::Mmap` is immutable for its entire lifetime and safe to read concurrently.
unsafe impl Sync for Backing {}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Mmap { ptr, len } = *self {
            // SAFETY: `ptr` and `len` were returned by a successful `libc::mmap` call in `Substrate::open`.
            unsafe {
                if libc::munmap(ptr.as_ptr().cast(), len) != 0 {}
            }
        }
    }
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Owned(buf) => buf.as_slice(),
            // SAFETY: `ptr` is valid for `len` bytes for the lifetime of `&self` and never mutated.
            Backing::Mmap { ptr, len } => unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) },
        }
    }
}

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
}

/// Mmap-backed frozen sysroot substrate storing summaries, stencils, and concrete instances.
pub struct Substrate {
    backing: Backing,
    header: Header,
}

impl fmt::Debug for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Substrate")
            .field("toolchain_hash", &format_args!("{:016x}", self.header.toolchain_hash))
            .field("target_hash", &format_args!("{:016x}", self.header.target_hash))
            .field("summaries", &self.header.summary_count)
            .field("stencils", &self.header.stencil_count)
            .field("instances", &self.header.instance_count)
            .finish()
    }
}

/// Returns the toolchain-scoped substrate path under `$HASTE_HOME` / `$XDG_CACHE_HOME/haste` / `$HOME/.cache/haste`.
pub fn toolchain_path(toolchain_hash: u64) -> PathBuf {
    crate::graph::store::home().join("toolchain").join(format!("{toolchain_hash:016x}.substrate"))
}

/// Computes a workspace-invariant key for a sysroot item, or `None` for non-sysroot items.
pub fn sysroot_item_key(outline: &crate::outline::Outline, item: crate::names::ItemRef) -> Option<u64> {
    let krate = outline.crates.get(item.krate as usize)?;
    if !krate.env.release || !krate.env.fixed {
        return None;
    }
    let root = outline.roots.get(item.krate as usize)?;
    let file = krate.files.get(item.file as usize)?;
    if file.expanded && file.expand_origin == 0 {
        return None;
    }
    let (base_item, sub_idx) = match item.outer() {
        Some((outer, sub)) => (outer.item, Some(sub)),
        None => (item.item, None),
    };
    let entry = file.items.items.get(base_item as usize)?;
    let base_dir = root.path.parent().unwrap_or(Path::new(""));
    let rel_path = file.source.path.strip_prefix(base_dir).unwrap_or(&file.source.path);
    let mut hasher = FxHasher::default();
    root.name.hash(&mut hasher);
    krate.env.target.triple.hash(&mut hasher);
    rel_path.hash(&mut hasher);
    file.expanded.hash(&mut hasher);
    if file.expanded {
        file.expand_origin.hash(&mut hasher);
        file.source.offset.hash(&mut hasher);
        file.source.text.hash(&mut hasher);
    }
    item.item.hash(&mut hasher);
    sub_idx.hash(&mut hasher);
    entry.start.hash(&mut hasher);
    entry.kind.hash(&mut hasher);
    if let Some(name_tok) = entry.name {
        file.source.text(name_tok).hash(&mut hasher);
    }
    Some(hasher.finish())
}

/// Returns whether substrate caching is enabled (disabled when `HASTE_NO_SUBSTRATE=1`).
pub fn enabled() -> bool {
    std::env::var_os("HASTE_NO_SUBSTRATE").is_none_or(|v| v != "1")
}

/// Computes a 64-bit fingerprint for `target` and the running compiler binary.
pub fn target_hash(target: &Target) -> u64 {
    let mut hasher = FxHasher::default();
    target.triple.hash(&mut hasher);
    target.effort.hash(&mut hasher);
    if let Ok(exe) = std::env::current_exe()
        && let Ok(meta) = std::fs::metadata(exe)
    {
        meta.len().hash(&mut hasher);
        if let Ok(modified) = meta.modified()
            && let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            dur.as_nanos().hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Patches a frozen `StencilSlice` against `symbols` into a fresh byte buffer, or fails if any relocation is runtime-only.
pub fn patch_stencil(slice: &StencilSlice<'_>, symbols: &[u64]) -> Result<Vec<u8>> {
    let relocs = slice.decode_relocs()?;
    let mut buf = slice.code().to_vec();
    for reloc in relocs {
        let Some(sym_idx) = reloc.symbol_index() else { bail!("frozen stencil has runtime relocation") };
        let value = *symbols.get(sym_idx).context("frozen stencil symbol index out of bounds")?;
        let off = reloc.offset as usize;
        let end = buf.len();
        haste_codegen::patch_reloc(reloc.kind, value, buf.get_mut(off..end).context("frozen stencil reloc offset out of bounds")?)
            .map_err(|err| anyhow!("{err}"))?;
    }
    Ok(buf)
}

fn hash_file_meta(path: &Path, hasher: &mut FxHasher) {
    path.hash(hasher);
    if let Ok(meta) = std::fs::metadata(path) {
        meta.len().hash(hasher);
        if let Ok(modified) = meta.modified()
            && let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            dur.as_nanos().hash(hasher);
        }
    }
}

/// Computes a workspace-invariant toolchain key from sysroot packages, target, and compiler binary.
pub fn toolchain_key(workspace: &crate::graph::Workspace, target: &Target) -> u64 {
    let mut sysroot_units: Vec<(&str, crate::graph::Side, usize)> = workspace
        .graph
        .units
        .iter()
        .enumerate()
        .filter(|(_, unit)| workspace.packages[unit.package].sysroot)
        .map(|(idx, unit)| (workspace.packages[unit.package].locked.name.as_str(), unit.side, idx))
        .collect();
    sysroot_units.sort_unstable_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    let mut hasher = FxHasher::default();
    target.triple.hash(&mut hasher);
    target.effort.hash(&mut hasher);
    crate::graph::store::program().hash(&mut hasher);
    if let Ok(exe) = std::env::current_exe() {
        hash_file_meta(&exe, &mut hasher);
    }
    MAGIC.hash(&mut hasher);
    VERSION.hash(&mut hasher);
    for (_, _, idx) in sysroot_units {
        let unit = &workspace.graph.units[idx];
        let pkg = &workspace.packages[unit.package];
        let locked = &pkg.locked;
        locked.name.hash(&mut hasher);
        locked.version.to_string().hash(&mut hasher);
        locked.source.hash(&mut hasher);
        matches!(unit.side, crate::graph::Side::Host).hash(&mut hasher);
        if let Some(lib) = &pkg.manifest.lib {
            hash_file_meta(lib, &mut hasher);
        }
        for feat in &unit.features {
            feat.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Encodes `sites` into compact fixed-width little-endian bytes.
pub fn encode_sites(sites: &[Site], out: &mut Vec<u8>) {
    out.reserve(sites.len() * SITE_WIRE_LEN);
    for site in sites {
        out.extend_from_slice(&site.ret.to_le_bytes());
        out.extend_from_slice(&site.handler.to_le_bytes());
        out.extend_from_slice(&site.frame_offset.to_le_bytes());
    }
}

/// Encodes `relocs` into compact fixed-width little-endian bytes.
pub fn encode_relocs(relocs: &[StencilReloc], out: &mut Vec<u8>) {
    out.reserve(relocs.len() * RELOC_WIRE_LEN);
    for reloc in relocs {
        out.extend_from_slice(&reloc.offset.to_le_bytes());
        out.extend_from_slice(&reloc.symbol.to_le_bytes());
        out.push(encode_reloc_kind(reloc.kind));
        out.extend_from_slice(&[0u8; 3]);
    }
}

impl Substrate {
    /// Opens and memory-maps a substrate file from `path`, validating its header and section bounds.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let meta = file.metadata().with_context(|| format!("stat {}", path.display()))?;
        let len_u64 = meta.len();
        if len_u64 < HEADER_LEN as u64 {
            bail!("substrate file smaller than header");
        }
        let len = usize::try_from(len_u64).context("substrate file exceeds address space")?;
        // SAFETY: `file` is open for reading and `len > 0`; `MAP_SHARED` creates a read-only mapping owned by `Backing`.
        let raw = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if raw == libc::MAP_FAILED {
            return Err(io::Error::last_os_error()).with_context(|| format!("mmap {}", path.display()));
        }
        let ptr = NonNull::new(raw.cast::<u8>()).ok_or_else(|| anyhow!("mmap returned null pointer"))?;
        let backing = Backing::Mmap { ptr, len };
        Self::from_backing(backing)
    }

    /// Opens the substrate for `toolchain_hash` from the toolchain cache directory.
    pub fn open_toolchain(toolchain_hash: u64) -> Result<Self> {
        let substrate = Self::open(&toolchain_path(toolchain_hash))?;
        if substrate.header.toolchain_hash != toolchain_hash {
            bail!("toolchain hash mismatch");
        }
        Ok(substrate)
    }

    /// Constructs a substrate from an in-memory byte buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Backing::Owned(bytes))
    }

    fn from_backing(backing: Backing) -> Result<Self> {
        let header = parse_and_validate(backing.as_slice())?;
        Ok(Self { backing, header })
    }

    /// Starts building a new frozen substrate.
    pub fn builder() -> SubstrateBuilder {
        SubstrateBuilder::default()
    }

    /// Toolchain fingerprint this substrate was compiled for.
    pub fn toolchain_hash(&self) -> u64 {
        self.header.toolchain_hash
    }

    /// Target triple fingerprint this substrate was compiled for.
    pub fn target_hash(&self) -> u64 {
        self.header.target_hash
    }

    /// Number of frozen `PolySummary` entries.
    pub fn summary_count(&self) -> usize {
        self.header.summary_count
    }

    /// Number of frozen `Effort::Full` stencil entries.
    pub fn stencil_count(&self) -> usize {
        self.header.stencil_count
    }

    /// Looks up a frozen `PolySummary` payload by `item_hash` in `O(log N)`.
    pub fn summary(&self, item_hash: u64) -> Option<&[u8]> {
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
        let rec = self.header.stencils_off + idx * STENCIL_REC_LEN;
        let code = self.blob_slice(read_u32(data, rec + 24) as usize, read_u32(data, rec + 28) as usize)?;
        let sites = self.blob_slice(read_u32(data, rec + 32) as usize, read_u32(data, rec + 36) as usize)?;
        let relocs = self.blob_slice(read_u32(data, rec + 40) as usize, read_u32(data, rec + 44) as usize)?;
        Some(StencilSlice { code, sites, relocs })
    }

    /// Iterates frozen concrete sysroot instance payloads matching `(item_hash, substs_hash)` in `O(log N)`.
    pub fn instance_candidates(&self, item_hash: u64, substs_hash: u64) -> impl Iterator<Item = &[u8]> + '_ {
        let data = self.backing.as_slice();
        let count = self.header.instance_count;
        let base = self.header.instances_off;
        let start = lower_bound_index(count, |i| {
            let rec = base + i * INSTANCE_REC_LEN;
            (read_u64(data, rec), read_u64(data, rec + 8)).cmp(&(item_hash, substs_hash))
        });
        (start..count)
            .map(move |i| base + i * INSTANCE_REC_LEN)
            .take_while(move |&rec| read_u64(data, rec) == item_hash && read_u64(data, rec + 8) == substs_hash)
            .filter_map(move |rec| self.blob_slice(read_u32(data, rec + 16) as usize, read_u32(data, rec + 20) as usize))
    }

    /// Atomically writes the substrate to `path` via a sibling temporary file and rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let seq = TMP_SEQ.fetch_add(1, AtomicOrdering::Relaxed);
        let tmp = path.with_extension(format!("substrate.tmp.{}.{seq}", std::process::id()));
        let write_res = (|| -> io::Result<()> {
            let mut file = File::create(&tmp)?;
            file.write_all(self.backing.as_slice())?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(err) = write_res {
            drop(std::fs::remove_file(&tmp));
            return Err(err).with_context(|| format!("write {}", tmp.display()));
        }
        std::fs::rename(&tmp, path).inspect_err(|_| drop(std::fs::remove_file(&tmp))).with_context(|| format!("rename {}", path.display()))
    }

    /// Atomically saves the substrate into the toolchain cache directory.
    pub fn save_toolchain(&self) -> Result<PathBuf> {
        let path = toolchain_path(self.header.toolchain_hash);
        self.save(&path)?;
        Ok(path)
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
}

struct PendingInstance {
    item_hash: u64,
    substs_hash: u64,
    off: u32,
    len: u32,
}

/// Builder that packs sorted sysroot summaries, stencils, and concrete instances into a `Substrate`.
#[derive(Default)]
#[must_use]
pub struct SubstrateBuilder {
    toolchain_hash: u64,
    target_hash: u64,
    summaries: Vec<PendingSummary>,
    stencils: Vec<PendingStencil>,
    instances: Vec<PendingInstance>,
    blob: Vec<u8>,
}

impl fmt::Debug for SubstrateBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubstrateBuilder")
            .field("toolchain_hash", &format_args!("{:016x}", self.toolchain_hash))
            .field("target_hash", &format_args!("{:016x}", self.target_hash))
            .field("summaries", &self.summaries.len())
            .field("stencils", &self.stencils.len())
            .field("instances", &self.instances.len())
            .field("blob_bytes", &self.blob.len())
            .finish()
    }
}

impl SubstrateBuilder {
    /// Sets the toolchain fingerprint.
    pub fn toolchain_hash(mut self, hash: u64) -> Self {
        self.toolchain_hash = hash;
        self
    }

    /// Sets the target triple fingerprint.
    pub fn target_hash(mut self, hash: u64) -> Self {
        self.target_hash = hash;
        self
    }

    /// Appends all summaries and stencils from an existing `Substrate` into this builder.
    pub fn merge_from(&mut self, existing: &Substrate) -> Result<()> {
        let data = existing.backing.as_slice();
        for i in 0..existing.header.summary_count {
            let rec = existing.header.summaries_off + i * SUMMARY_REC_LEN;
            let item_hash = read_u64(data, rec);
            let off = read_u32(data, rec + 8) as usize;
            let len = read_u32(data, rec + 12) as usize;
            let bytes = existing.blob_slice(off, len).context("corrupt existing summary slice")?;
            self.add_summary(item_hash, bytes)?;
        }
        for i in 0..existing.header.stencil_count {
            let rec = existing.header.stencils_off + i * STENCIL_REC_LEN;
            let item_hash = read_u64(data, rec);
            let fp_lo = read_u64(data, rec + 8);
            let fp_hi = read_u64(data, rec + 16);
            let footprint = u128::from(fp_lo) | (u128::from(fp_hi) << 64);
            let code = existing.blob_slice(read_u32(data, rec + 24) as usize, read_u32(data, rec + 28) as usize).context("corrupt existing stencil code")?;
            let sites = existing.blob_slice(read_u32(data, rec + 32) as usize, read_u32(data, rec + 36) as usize).context("corrupt existing stencil sites")?;
            let relocs = existing.blob_slice(read_u32(data, rec + 40) as usize, read_u32(data, rec + 44) as usize).context("corrupt existing stencil relocs")?;
            self.add_stencil(item_hash, footprint, code, sites, relocs)?;
        }
        for i in 0..existing.header.instance_count {
            let rec = existing.header.instances_off + i * INSTANCE_REC_LEN;
            let item_hash = read_u64(data, rec);
            let substs_hash = read_u64(data, rec + 8);
            let off = read_u32(data, rec + 16) as usize;
            let len = read_u32(data, rec + 20) as usize;
            let bytes = existing.blob_slice(off, len).context("corrupt existing instance slice")?;
            self.add_instance(item_hash, substs_hash, bytes)?;
        }
        Ok(())
    }

    /// Appends a serialized `PolySummary` entry.
    pub fn add_summary(&mut self, item_hash: u64, bytes: &[u8]) -> Result<()> {
        let (off, len) = self.push_blob(bytes)?;
        self.summaries.push(PendingSummary { item_hash, off, len });
        Ok(())
    }

    /// Appends a serialized concrete sysroot instance entry.
    pub fn add_instance(&mut self, item_hash: u64, substs_hash: u64, bytes: &[u8]) -> Result<()> {
        let (off, len) = self.push_blob(bytes)?;
        self.instances.push(PendingInstance { item_hash, substs_hash, off, len });
        Ok(())
    }

    /// Appends a pre-compiled `Effort::Full` stencil entry.
    pub fn add_stencil(&mut self, item_hash: u64, footprint: u128, code: &[u8], sites: &[u8], relocs: &[u8]) -> Result<()> {
        let (code_off, code_len) = self.push_blob(code)?;
        let (sites_off, sites_len) = self.push_blob(sites)?;
        let (relocs_off, relocs_len) = self.push_blob(relocs)?;
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
        });
        Ok(())
    }

    /// Appends a compiled `haste_codegen::Code` stencil by encoding its bytes, sites, and relocations.
    pub fn add_code_stencil(&mut self, item_hash: u64, footprint: u128, code: &Code) -> Result<()> {
        let mut sites = Vec::new();
        encode_sites(code.sites(), &mut sites);
        let mut relocs = Vec::new();
        encode_relocs(code.relocs(), &mut relocs);
        self.add_stencil(item_hash, footprint, code.bytes(), &sites, &relocs)
    }

    fn push_blob(&mut self, bytes: &[u8]) -> Result<(u32, u32)> {
        let off = u32::try_from(self.blob.len()).context("substrate blob exceeds 4 GiB")?;
        let len = u32::try_from(bytes.len()).context("substrate entry exceeds 4 GiB")?;
        self.blob.extend_from_slice(bytes);
        let pad = (8 - (self.blob.len() & 7)) & 7;
        self.blob.resize(self.blob.len() + pad, 0);
        Ok((off, len))
    }

    /// Sorts all index tables, serializes the binary image, and returns a validated `Substrate`.
    pub fn build(mut self) -> Result<Substrate> {
        self.summaries.sort_by_key(|entry| entry.item_hash);
        self.summaries.dedup_by_key(|entry| entry.item_hash);
        self.stencils.sort_by_key(|entry| (entry.item_hash, entry.fp_hi, entry.fp_lo));
        self.stencils.dedup_by_key(|entry| (entry.item_hash, entry.fp_hi, entry.fp_lo));
        self.instances.sort_by(|a, b| {
            (a.item_hash, a.substs_hash)
                .cmp(&(b.item_hash, b.substs_hash))
                .then_with(|| {
                    let ab = &self.blob[a.off as usize..a.off as usize + a.len as usize];
                    let bb = &self.blob[b.off as usize..b.off as usize + b.len as usize];
                    ab.cmp(bb)
                })
        });
        self.instances.dedup_by(|a, b| {
            a.item_hash == b.item_hash
                && a.substs_hash == b.substs_hash
                && self.blob[a.off as usize..a.off as usize + a.len as usize] == self.blob[b.off as usize..b.off as usize + b.len as usize]
        });

        let summaries_off = HEADER_LEN;
        let stencils_off = summaries_off + self.summaries.len() * SUMMARY_REC_LEN;
        let instances_off = stencils_off + self.stencils.len() * STENCIL_REC_LEN;
        let blob_off = instances_off + self.instances.len() * INSTANCE_REC_LEN;
        let total = blob_off + self.blob.len();

        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.toolchain_hash.to_le_bytes());
        out.extend_from_slice(&self.target_hash.to_le_bytes());
        out.extend_from_slice(&(self.summaries.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.stencils.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.instances.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.blob.len() as u64).to_le_bytes());

        for entry in &self.summaries {
            out.extend_from_slice(&entry.item_hash.to_le_bytes());
            out.extend_from_slice(&entry.off.to_le_bytes());
            out.extend_from_slice(&entry.len.to_le_bytes());
        }
        for entry in &self.stencils {
            out.extend_from_slice(&entry.item_hash.to_le_bytes());
            out.extend_from_slice(&entry.fp_lo.to_le_bytes());
            out.extend_from_slice(&entry.fp_hi.to_le_bytes());
            out.extend_from_slice(&entry.code_off.to_le_bytes());
            out.extend_from_slice(&entry.code_len.to_le_bytes());
            out.extend_from_slice(&entry.sites_off.to_le_bytes());
            out.extend_from_slice(&entry.sites_len.to_le_bytes());
            out.extend_from_slice(&entry.relocs_off.to_le_bytes());
            out.extend_from_slice(&entry.relocs_len.to_le_bytes());
        }
        for entry in &self.instances {
            out.extend_from_slice(&entry.item_hash.to_le_bytes());
            out.extend_from_slice(&entry.substs_hash.to_le_bytes());
            out.extend_from_slice(&entry.off.to_le_bytes());
            out.extend_from_slice(&entry.len.to_le_bytes());
        }
        out.extend_from_slice(&self.blob);
        Substrate::from_bytes(out)
    }
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
    let toolchain_hash = read_u64(data, 16);
    let target_hash = read_u64(data, 24);
    let summary_count = usize::try_from(read_u64(data, 32)).context("summary count overflow")?;
    let stencil_count = usize::try_from(read_u64(data, 40)).context("stencil count overflow")?;
    let instance_count = usize::try_from(read_u64(data, 48)).context("instance count overflow")?;
    let blob_len = usize::try_from(read_u64(data, 56)).context("blob length overflow")?;

    let summaries_off = HEADER_LEN;
    let summaries_bytes = summary_count.checked_mul(SUMMARY_REC_LEN).context("summary table overflow")?;
    let stencils_off = summaries_off.checked_add(summaries_bytes).context("stencil offset overflow")?;
    let stencils_bytes = stencil_count.checked_mul(STENCIL_REC_LEN).context("stencil table overflow")?;
    let instances_off = stencils_off.checked_add(stencils_bytes).context("instance offset overflow")?;
    let instances_bytes = instance_count.checked_mul(INSTANCE_REC_LEN).context("instance table overflow")?;
    let blob_off = instances_off.checked_add(instances_bytes).context("blob offset overflow")?;
    let total = blob_off.checked_add(blob_len).context("total length overflow")?;
    if data.len() < total {
        bail!("truncated substrate payload");
    }
    Ok(Header { toolchain_hash, target_hash, summary_count, stencil_count, instance_count, summaries_off, stencils_off, instances_off, blob_off, blob_len })
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

fn read_u32(data: &[u8], at: usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[at..at + 4]);
    u32::from_le_bytes(bytes)
}

fn read_u64(data: &[u8], at: usize) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[at..at + 8]);
    u64::from_le_bytes(bytes)
}

fn encode_reloc_kind(kind: RelocKind) -> u8 {
    match kind {
        RelocKind::Abs32 => 0,
        RelocKind::Abs64 => 1,
        RelocKind::Arm64Movw64 => 2,
        RelocKind::PcRel32 => 3,
        RelocKind::Arm64Call26 => 4,
    }
}

fn decode_reloc_kind(raw: u8) -> Result<RelocKind> {
    match raw {
        0 => Ok(RelocKind::Abs32),
        1 => Ok(RelocKind::Abs64),
        2 => Ok(RelocKind::Arm64Movw64),
        3 => Ok(RelocKind::PcRel32),
        4 => Ok(RelocKind::Arm64Call26),
        _ => bail!("unknown relocation kind"),
    }
}
