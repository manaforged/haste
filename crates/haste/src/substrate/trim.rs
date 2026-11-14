//! Size trim that evicts stencils until a capped substrate fits its byte limit.

use std::cmp::Reverse;

use anyhow::Result;

use super::tally::{Tally, add, tally};
use super::{HEADER_LEN, INSTANCE_REC_LEN, PendingStencil, STENCIL_REC_LEN, SUMMARY_REC_LEN, SubstrateBuilder, aligned8, push_aligned_blob};

impl PendingStencil {
    fn cost(&self) -> usize {
        STENCIL_REC_LEN + aligned8(self.code_len as usize) + aligned8(self.sites_len as usize) + aligned8(self.relocs_len as usize) + aligned8(self.plan_len as usize) + aligned8(self.loops_len as usize)
    }
}

impl SubstrateBuilder {
    pub(super) fn trim_stencils_to_limit(&mut self, limit: usize) -> Result<()> {
        let base_table = HEADER_LEN + self.summaries.len() * SUMMARY_REC_LEN + self.instances.len() * INSTANCE_REC_LEN;
        let current = base_table + self.stencils.len() * STENCIL_REC_LEN + self.blob.len();
        let base = base_table
            + self.summaries.iter().map(|s| aligned8(s.len as usize)).sum::<usize>()
            + self.instances.iter().map(|i| aligned8(i.len as usize)).sum::<usize>();
        let need = base + self.stencils.iter().map(PendingStencil::cost).sum::<usize>();
        add(Tally::TrimStencils, self.stencils.len());
        add(Tally::TrimBaseBytes, base);
        add(Tally::TrimNeedBytes, need);
        if current <= limit {
            add(Tally::TrimKeptBytes, need);
            return Ok(());
        }
        let mut used = base;
        self.stencils.sort_by_key(|s| (Reverse(s.code_len), Reverse(s.plan_len), s.item_hash));
        let mut kept = Vec::with_capacity(self.stencils.len());
        for s in self.stencils.drain(..) {
            let cost = s.cost();
            if used.saturating_add(cost) > limit {
                tally(Tally::TrimDropped);
                continue;
            }
            used += cost;
            kept.push(s);
        }
        add(Tally::TrimKeptBytes, used);
        kept.sort_by_key(|entry| (entry.item_hash, entry.fp_hi, entry.fp_lo));
        self.stencils = kept;
        let old = std::mem::take(&mut self.blob);
        for s in &mut self.summaries {
            (s.off, s.len) = push_aligned_blob(&mut self.blob, &old[s.off as usize..s.off as usize + s.len as usize])?;
        }
        for s in &mut self.stencils {
            (s.code_off, s.code_len) = push_aligned_blob(&mut self.blob, &old[s.code_off as usize..s.code_off as usize + s.code_len as usize])?;
            (s.sites_off, s.sites_len) = push_aligned_blob(&mut self.blob, &old[s.sites_off as usize..s.sites_off as usize + s.sites_len as usize])?;
            (s.relocs_off, s.relocs_len) = push_aligned_blob(&mut self.blob, &old[s.relocs_off as usize..s.relocs_off as usize + s.relocs_len as usize])?;
            (s.plan_off, s.plan_len) = push_aligned_blob(&mut self.blob, &old[s.plan_off as usize..s.plan_off as usize + s.plan_len as usize])?;
            (s.loops_off, s.loops_len) = push_aligned_blob(&mut self.blob, &old[s.loops_off as usize..s.loops_off as usize + s.loops_len as usize])?;
        }
        for i in &mut self.instances {
            (i.off, i.len) = push_aligned_blob(&mut self.blob, &old[i.off as usize..i.off as usize + i.len as usize])?;
        }
        Ok(())
    }
}
