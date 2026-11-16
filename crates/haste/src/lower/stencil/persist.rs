use std::sync::Mutex;

use anyhow::Result;

use crate::sema::Context;
use crate::substrate::tally::{Tally, tally};
use crate::substrate::{InlineKey, InlineShape, SubstrateBuilder, layer_item_key};
use crate::sync::lock;

use super::{Bodies, TemplateKey};

pub(super) struct Persist {
    mode: u64,
    verify: bool,
    fresh: Mutex<Vec<(bool, InlineKey, InlineShape)>>,
}

pub(super) struct Disk<'b> {
    persist: &'b Persist,
    sysroot: bool,
    key: InlineKey,
    found: Option<InlineShape>,
}

impl Disk<'_> {
    pub(super) fn hit(&self) -> Option<InlineShape> {
        if self.found.is_some() {
            tally(Tally::MetaHit);
        }
        self.found.filter(|_| !self.persist.verify)
    }

    pub(super) fn settle(self, built: InlineShape, stored: bool, served: bool) {
        match self.found {
            Some(found) if (stored || served) && found != built => {
                tally(Tally::MetaMismatch);
                assert!(!self.persist.verify, "inline metadata of {:?} is {found:?} in the substrate and {built:?} when built", self.key);
            }
            None if stored => {
                tally(Tally::MetaStored);
                lock(&self.persist.fresh).push((self.sysroot, self.key, built));
            }
            _ => {}
        }
    }
}

impl Bodies {
    /// Starts reading and recording inline template outcomes in the substrate under item hashes XORed with `mode`; with `verify`, every template is still built and a differing recorded outcome panics.
    pub(crate) fn persist(&self, mode: u64, verify: bool) {
        self.persist.get_or_init(|| Persist { mode, verify, fresh: Mutex::default() });
    }

    /// Drops every cached body, footprint plan and cycle entry and keeps the inline template outcomes recorded for the substrate.
    pub(crate) fn reset(&mut self) {
        let persist = std::mem::take(&mut self.persist);
        *self = Self { persist, ..Self::default() };
    }

    pub(super) fn disk(&self, cx: &Context<'_>, key: &TemplateKey) -> Option<Disk<'_>> {
        let persist = self.persist.get()?;
        if !key.footprint.is_portable() {
            return None;
        }
        let (sysroot, item) = layer_item_key(cx.outline, key.item)?;
        let key = InlineKey { item: item ^ persist.mode, footprint: key.footprint.0, limit: u32::try_from(key.limit).ok()?, low: key.low, sole: key.sole };
        let found = cx.store().substrate().and_then(|substrate| substrate.inline_meta(&key));
        tally(Tally::MetaLookup);
        if found.is_none() {
            tally(Tally::MetaMiss);
        }
        Some(Disk { persist, sysroot, key, found })
    }

    /// Whether this build recorded inline template outcomes that the substrate does not hold yet.
    pub(crate) fn has_fresh_meta(&self) -> bool {
        self.persist.get().is_some_and(|persist| !lock(&persist.fresh).is_empty())
    }

    /// Appends the inline template outcomes this build recorded for sysroot items into `toolchain` and for dependency items into `deps`.
    pub(crate) fn export_meta(&self, mut toolchain: Option<&mut SubstrateBuilder>, mut deps: Option<&mut SubstrateBuilder>) -> Result<()> {
        let Some(persist) = self.persist.get() else { return Ok(()) };
        for &(sysroot, key, shape) in lock(&persist.fresh).iter() {
            let target = if sysroot { toolchain.as_deref_mut() } else { deps.as_deref_mut() };
            if let Some(builder) = target {
                builder.add_inline_meta(&key, shape)?;
            }
        }
        Ok(())
    }
}
