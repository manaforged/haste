use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::image::stencil::DictConfig;
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::tree::Profile;
use crate::substrate::target_hash;

use super::{CACHE_MAGIC, CACHE_VERSION, Stamps, hash_str, stable_entry_item_key};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct AotConfigKey {
    pub(super) target_hash: u64,
    pub(super) triple_hash: u64,
    pub(super) flags_bits: u64,
    pub(super) global_alloc_stamp: u64,
    pub(super) dict_cfg: (usize, usize),
    pub(super) config_digest: u64,
}

impl AotConfigKey {
    pub(super) fn compute(
        outline: &Outline,
        stamps: &Stamps,
        unwind_enabled: bool,
        global_alloc: Option<ItemRef>,
        dict: DictConfig,
    ) -> Self {
        let target = crate::host_target().direct_calls(true);
        let t_hash = target_hash(&target);
        let triple_hash = hash_str(target.triple);
        let release = crate::profile() == Profile::Release;
        let (x64_baseline, debug_info) = crate::with_cfg(|c| (c.x64_baseline, c.debug_info));
        let in_host = crate::in_host_target();
        let allow_inline = release;
        let low_entries = true;
        let flags_bits = u64::from(target.wide_vectors())
            | (u64::from(release) << 1)
            | (u64::from(unwind_enabled) << 2)
            | (u64::from(global_alloc.is_some()) << 3)
            | (u64::from(low_entries) << 4)
            | (u64::from(allow_inline) << 5)
            | (u64::from(x64_baseline) << 6)
            | (u64::from(in_host) << 7)
            | (u64::from(debug_info) << 8);
        let global_alloc_stamp = global_alloc
            .and_then(|item| {
                let k = stable_entry_item_key(outline, stamps, item)?;
                let mut h = FxHasher::default();
                k.hash(&mut h);
                stamps.iface(item.krate).hash(&mut h);
                Some(h.finish())
            })
            .unwrap_or(0);
        let dict_cfg = (dict.min_bytes, dict.min_fanout);
        let mut h = FxHasher::default();
        CACHE_MAGIC.hash(&mut h);
        CACHE_VERSION.hash(&mut h);
        t_hash.hash(&mut h);
        triple_hash.hash(&mut h);
        flags_bits.hash(&mut h);
        global_alloc_stamp.hash(&mut h);
        dict_cfg.hash(&mut h);
        Self {
            target_hash: t_hash,
            triple_hash,
            flags_bits,
            global_alloc_stamp,
            dict_cfg,
            config_digest: h.finish(),
        }
    }
}
