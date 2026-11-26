//! Eviction that keeps every stencil a live consumer still needs and drops the rest.

use anyhow::Result;

use super::consumers::retain_live;
use super::tally::{Tally, add};
use super::{PendingStencil, STENCIL_REC_LEN, SubstrateBuilder, aligned8, push_aligned_blob};

impl PendingStencil {
    fn cost(&self) -> usize {
        STENCIL_REC_LEN
            + aligned8(self.code_len as usize)
            + aligned8(self.sites_len as usize)
            + aligned8(self.relocs_len as usize)
            + aligned8(self.plan_len as usize)
            + aligned8(self.loops_len as usize)
            + aligned8(self.unwind_len as usize)
    }
}

impl SubstrateBuilder {
    pub(super) fn evict_unneeded_stencils(&mut self) -> Result<()> {
        let current = self.current_consumer();
        let floor = retain_live(&mut self.consumers, current);
        let need = self
            .stencils
            .iter()
            .map(PendingStencil::cost)
            .sum::<usize>();
        add(Tally::TrimStencils, self.stencils.len());
        add(Tally::TrimNeedBytes, need);
        add(
            Tally::TrimBaseBytes,
            self.summaries
                .iter()
                .map(|s| aligned8(s.len as usize))
                .sum::<usize>()
                + self
                    .instances
                    .iter()
                    .map(|i| aligned8(i.len as usize))
                    .sum::<usize>(),
        );
        let before = self.stencils.len();
        self.stencils
            .retain(|s| s.last_used.is_none_or(|g| g >= floor));
        add(Tally::TrimDropped, before - self.stencils.len());
        add(
            Tally::TrimKeptBytes,
            self.stencils
                .iter()
                .map(PendingStencil::cost)
                .sum::<usize>(),
        );
        if self.stencils.len() == before {
            return Ok(());
        }
        let old = std::mem::take(&mut self.blob);
        let mut repack = |off: &mut u32, len: &mut u32| -> Result<()> {
            (*off, *len) = push_aligned_blob(
                &mut self.blob,
                &old[*off as usize..*off as usize + *len as usize],
            )?;
            Ok(())
        };
        for s in &mut self.summaries {
            repack(&mut s.off, &mut s.len)?;
        }
        for s in &mut self.stencils {
            repack(&mut s.code_off, &mut s.code_len)?;
            repack(&mut s.sites_off, &mut s.sites_len)?;
            repack(&mut s.relocs_off, &mut s.relocs_len)?;
            repack(&mut s.plan_off, &mut s.plan_len)?;
            repack(&mut s.loops_off, &mut s.loops_len)?;
            repack(&mut s.unwind_off, &mut s.unwind_len)?;
        }
        for i in &mut self.instances {
            repack(&mut i.off, &mut i.len)?;
        }
        Ok(())
    }
}
