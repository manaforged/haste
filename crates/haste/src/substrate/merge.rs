//! Merge of an existing substrate layer into a builder, carrying use generations and live consumers.

use anyhow::{Context as _, Result};
use std::sync::atomic::Ordering;

use super::consumers::{CONSUMER_REC_LEN, Consumer};
use super::{INSTANCE_REC_LEN, STENCIL_LAST_USED_AT, STENCIL_REC_LEN, SUMMARY_REC_LEN, Substrate, SubstrateBuilder, read_u32, read_u64};

impl SubstrateBuilder {
    /// Appends all entries from `existing`'s own backing, stamping stencils this build looked up as used now.
    pub fn merge_from(&mut self, existing: &Substrate) -> Result<()> {
        let data = existing.backing.as_slice();
        self.generation = self.generation.max(existing.header.generation.wrapping_add(1));
        self.consumers = (0..existing.header.consumer_count).map(|i| Consumer::read(data, existing.header.consumers_off + i * CONSUMER_REC_LEN)).collect();
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
            let last_used = (!existing.used[i].load(Ordering::Relaxed)).then(|| read_u32(data, rec + STENCIL_LAST_USED_AT));
            self.push_stencil(item_hash, footprint, existing.stencil_at(rec).context("corrupt existing stencil")?, last_used)?;
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
}
