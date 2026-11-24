//! Merge and streaming serialization of substrate builders.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use anyhow::{Context as _, Result};
use rustc_hash::FxHashSet;

use super::consumers::{CONSUMER_REC_LEN, Consumer, KEPT_WORKSPACES};
use super::{
    HEADER_LEN, INSTANCE_REC_LEN, MAGIC, STENCIL_LAST_USED_AT, STENCIL_REC_LEN, SUMMARY_REC_LEN,
    Substrate, SubstrateBuilder, VERSION, aligned8, deps_path, prune_stale_substrates,
    prune_superseded, read_u32, read_u64, toolchain_path, workspace_path,
};

fn existing_needed_bytes(
    existing: &Substrate,
    seen_summaries: &FxHashSet<u64>,
    seen_stencils: &FxHashSet<(u64, u64, u64)>,
) -> usize {
    let data = existing.backing.as_slice();
    let mut needed = 0usize;
    for i in 0..existing.header.summary_count {
        let rec = existing.header.summaries_off + i * SUMMARY_REC_LEN;
        if !seen_summaries.contains(&read_u64(data, rec)) {
            needed += aligned8(read_u32(data, rec + 12) as usize);
        }
    }
    for i in 0..existing.header.stencil_count {
        let rec = existing.header.stencils_off + i * STENCIL_REC_LEN;
        let key = (
            read_u64(data, rec),
            read_u64(data, rec + 16),
            read_u64(data, rec + 8),
        );
        if !seen_stencils.contains(&key)
            && let Some(slice) = existing.stencil_at(rec)
        {
            needed += aligned8(slice.code.len())
                + aligned8(slice.sites.len())
                + aligned8(slice.relocs.len())
                + aligned8(slice.plan.len())
                + aligned8(slice.loops.len())
                + aligned8(slice.unwind_drops.len());
        }
    }
    for i in 0..existing.header.instance_count {
        let rec = existing.header.instances_off + i * INSTANCE_REC_LEN;
        needed += aligned8(read_u32(data, rec + 20) as usize);
    }
    needed
}

impl SubstrateBuilder {
    /// Appends non-superseded entries from `existing`'s own backing, stamping stencils this build looked up as used now.
    pub fn merge_from(&mut self, existing: &Substrate) -> Result<()> {
        let data = existing.backing.as_slice();
        self.generation = self
            .generation
            .max(existing.header.generation.wrapping_add(1));
        self.consumers = (0..existing.header.consumer_count)
            .map(|i| Consumer::read(data, existing.header.consumers_off + i * CONSUMER_REC_LEN))
            .collect();
        let seen_summaries: FxHashSet<u64> = self.summaries.iter().map(|s| s.item_hash).collect();
        let seen_stencils: FxHashSet<(u64, u64, u64)> = self
            .stencils
            .iter()
            .map(|s| (s.item_hash, s.fp_hi, s.fp_lo))
            .collect();
        self.blob.reserve_exact(existing_needed_bytes(
            existing,
            &seen_summaries,
            &seen_stencils,
        ));
        self.summaries.reserve(existing.header.summary_count);
        self.stencils.reserve(existing.header.stencil_count);
        self.instances.reserve(existing.header.instance_count);
        for i in 0..existing.header.summary_count {
            let rec = existing.header.summaries_off + i * SUMMARY_REC_LEN;
            let item_hash = read_u64(data, rec);
            if seen_summaries.contains(&item_hash) {
                continue;
            }
            let bytes = existing
                .blob_slice(
                    read_u32(data, rec + 8) as usize,
                    read_u32(data, rec + 12) as usize,
                )
                .context("corrupt existing summary slice")?;
            self.add_summary(item_hash, bytes)?;
        }
        for i in 0..existing.header.stencil_count {
            let rec = existing.header.stencils_off + i * STENCIL_REC_LEN;
            let (item_hash, fp_lo, fp_hi) = (
                read_u64(data, rec),
                read_u64(data, rec + 8),
                read_u64(data, rec + 16),
            );
            if seen_stencils.contains(&(item_hash, fp_hi, fp_lo)) {
                continue;
            }
            let footprint = u128::from(fp_lo) | (u128::from(fp_hi) << 64);
            let last_used = (!existing.used[i].load(Ordering::Relaxed))
                .then(|| read_u32(data, rec + STENCIL_LAST_USED_AT));
            self.push_stencil(
                item_hash,
                footprint,
                existing
                    .stencil_at(rec)
                    .context("corrupt existing stencil")?,
                last_used,
            )?;
        }
        for i in 0..existing.header.instance_count {
            let rec = existing.header.instances_off + i * INSTANCE_REC_LEN;
            let (item_hash, substs_hash) = (read_u64(data, rec), read_u64(data, rec + 8));
            let bytes = existing
                .blob_slice(
                    read_u32(data, rec + 16) as usize,
                    read_u32(data, rec + 20) as usize,
                )
                .context("corrupt existing instance slice")?;
            self.add_instance(item_hash, substs_hash, bytes)?;
        }
        Ok(())
    }

    pub(super) fn prepare(&mut self) -> Result<usize> {
        self.summaries.sort_by_key(|entry| entry.item_hash);
        self.summaries.dedup_by_key(|entry| entry.item_hash);
        self.stencils
            .sort_by_key(|entry| (entry.item_hash, entry.fp_hi, entry.fp_lo));
        self.stencils
            .dedup_by_key(|entry| (entry.item_hash, entry.fp_hi, entry.fp_lo));
        let blob = self.blob.as_slice();
        self.instances.sort_by(|a, b| {
            (a.item_hash, a.substs_hash)
                .cmp(&(b.item_hash, b.substs_hash))
                .then_with(|| {
                    blob[a.off as usize..a.off as usize + a.len as usize]
                        .cmp(&blob[b.off as usize..b.off as usize + b.len as usize])
                })
        });
        self.instances.dedup_by(|a, b| {
            a.item_hash == b.item_hash
                && a.substs_hash == b.substs_hash
                && blob[a.off as usize..a.off as usize + a.len as usize]
                    == blob[b.off as usize..b.off as usize + b.len as usize]
        });
        self.evict_unneeded_stencils()?;
        let tables_end = HEADER_LEN
            + self.summaries.len() * SUMMARY_REC_LEN
            + self.stencils.len() * STENCIL_REC_LEN
            + self.instances.len() * INSTANCE_REC_LEN
            + self.consumers.len() * CONSUMER_REC_LEN;
        Ok(tables_end + self.blob.len())
    }

    pub(super) fn encode_tables(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.generation.to_le_bytes());
        for word in [
            self.toolchain_hash,
            self.target_hash,
            self.summaries.len() as u64,
            self.stencils.len() as u64,
            self.instances.len() as u64,
            self.blob.len() as u64,
            self.consumers.len() as u64,
        ] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        for entry in &self.summaries {
            out.extend_from_slice(&entry.item_hash.to_le_bytes());
            out.extend_from_slice(&entry.off.to_le_bytes());
            out.extend_from_slice(&entry.len.to_le_bytes());
        }
        for entry in &self.stencils {
            for word in [entry.item_hash, entry.fp_lo, entry.fp_hi] {
                out.extend_from_slice(&word.to_le_bytes());
            }
            for field in [
                entry.code_off,
                entry.code_len,
                entry.sites_off,
                entry.sites_len,
                entry.relocs_off,
                entry.relocs_len,
                entry.plan_off,
                entry.plan_len,
                entry.loops_off,
                entry.loops_len,
                entry.unwind_off,
                entry.unwind_len,
                entry.last_used.unwrap_or(self.generation),
                entry.saved_regs,
            ] {
                out.extend_from_slice(&field.to_le_bytes());
            }
        }
        for entry in &self.instances {
            out.extend_from_slice(&entry.item_hash.to_le_bytes());
            out.extend_from_slice(&entry.substs_hash.to_le_bytes());
            out.extend_from_slice(&entry.off.to_le_bytes());
            out.extend_from_slice(&entry.len.to_le_bytes());
        }
        self.consumers.iter().for_each(|c| c.write(out));
    }

    fn save_streaming(mut self, path: &Path) -> Result<bool> {
        let total = self.prepare()?;
        if self.is_empty() {
            return Ok(false);
        }
        self.blob.shrink_to_fit();
        let mut tables = Vec::with_capacity(total - self.blob.len());
        self.encode_tables(&mut tables);
        crate::graph::store::write_new_with(path, |writer| {
            writer.write_all(&tables)?;
            writer.write_all(&self.blob)
        })
        .with_context(|| format!("writing {}", path.display()))?;
        Ok(true)
    }

    /// Streams this builder into the toolchain cache directory and prunes superseded substrates.
    pub fn save_toolchain(self) -> Result<Option<PathBuf>> {
        let hash = self.toolchain_hash;
        let path = toolchain_path(hash);
        if !self.save_streaming(&path)? {
            return Ok(None);
        }
        prune_superseded(hash);
        Ok(Some(path))
    }

    /// Streams this builder into the dependency cache directory.
    pub fn save_deps(self) -> Result<Option<PathBuf>> {
        let path = deps_path(self.toolchain_hash);
        self.save_scoped(path, 16)
    }

    /// Streams this builder into the workspace cache directory.
    pub fn save_workspace(self) -> Result<Option<PathBuf>> {
        let path = workspace_path(self.toolchain_hash);
        self.save_scoped(path, KEPT_WORKSPACES - 1)
    }

    fn save_scoped(self, path: PathBuf, keep_others: usize) -> Result<Option<PathBuf>> {
        if !self.save_streaming(&path)? {
            return Ok(None);
        }
        if let Some(dir) = path.parent() {
            prune_stale_substrates(dir, &path, keep_others);
        }
        Ok(Some(path))
    }
}
