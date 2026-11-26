use crate::image::Image;
use crate::image::aot::CachedAotTemplate;
use crate::lower::{
    FootprintId, RodataId, RodataReloc, RodataTarget, Slots, SymbolSource, VtableEntry, VtableId,
    Word,
};
use crate::names::ItemRef;
use crate::sema::Context;
use crate::sema::reach::Reached;
use crate::sync::lock;

impl Image {
    fn alloc_rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        if let Some(aot) = self.aot() {
            return lock(aot).alloc_rodata(bytes, relocs);
        }
        if self.backend.is_check() {
            return RodataId(0);
        }
        let address = self.data(bytes) as usize as u64;
        for reloc in relocs {
            self.apply_rodata_reloc(address, *reloc);
        }
        let mut addrs = lock(&self.rodata_addrs);
        let id = RodataId(addrs.len() as u32);
        addrs.push(address);
        if self.backend.is_lazy_jit() {
            lock(&self.rodata_entries).push((bytes.to_vec(), relocs.to_vec()));
        }
        id
    }

    pub(in crate::image) fn intern_rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
        if let Some(aot) = self.aot() {
            return lock(aot).intern_rodata(bytes, relocs);
        }
        if self.backend.is_check() {
            return RodataId(0);
        }
        let key = (bytes.to_vec(), relocs.to_vec());
        let mut map = lock(&self.rodata_map);
        if let Some(&id) = map.get(&key) {
            return id;
        }
        let id = self.alloc_rodata(bytes, relocs);
        map.insert(key, id);
        id
    }

    pub(in crate::image) fn intern_static_rodata(&self, index: usize, bytes: &[u8]) -> RodataId {
        if self.backend.is_check() {
            return RodataId(0);
        }
        let mut map = lock(&self.static_rodata);
        if let Some(&id) = map.get(&index) {
            return id;
        }
        let id = self.alloc_rodata(bytes, &[]);
        map.insert(index, id);
        id
    }

    fn apply_rodata_reloc(&self, base: u64, reloc: RodataReloc) {
        let at = (base + u64::from(reloc.offset)) as usize;
        match reloc.target {
            RodataTarget::Rodata(target) => {
                let ptr = lock(&self.rodata_addrs)[target.0 as usize];
                // SAFETY: at..at + 8 lies within the freshly allocated rodata blob in shared memory.
                unsafe { (at as *mut u64).write_unaligned(ptr) };
            }
            RodataTarget::Fn(index) => self.register_code_fixup(index, at),
        }
    }

    pub(in crate::image) fn intern_vtable(&self, entries: &[VtableEntry]) -> VtableId {
        if let Some(aot) = self.aot() {
            return lock(aot).intern_vtable(entries);
        }
        let mut map = lock(&self.vtable_map);
        if let Some(&id) = map.get(entries) {
            return id;
        }
        let address = if self.backend.is_jit() {
            let words: Vec<Word> = entries
                .iter()
                .map(|entry| match *entry {
                    VtableEntry::Word(word) => word,
                    VtableEntry::Super(sub) => Word::Int(lock(&self.vtable_addrs)[sub.0 as usize]),
                })
                .collect();
            self.table(&words) as usize as u64
        } else {
            0
        };
        let mut addrs = lock(&self.vtable_addrs);
        let id = VtableId(addrs.len() as u32);
        addrs.push(address);
        lock(&self.vtable_entries).push(entries.to_vec());
        map.insert(entries.to_vec(), id);
        id
    }

    pub(in crate::image) fn promoted_cell(&self, owner: usize, seq: u32, size: u32) -> u64 {
        if let Some(aot) = self.aot() {
            let mut state = lock(aot);
            if !state.promoted_seen.contains_key(&(owner, seq)) {
                let at = state.promoted.len();
                state.promoted_seen.insert((owner, seq), at);
                state.promoted.push(((owner, seq), size));
            }
            return 0;
        }
        if self.backend.is_check() {
            return 0;
        }
        let mut cells = lock(&self.promoted_cells);
        if let Some(&addr) = cells.get(&(owner, seq)) {
            return addr;
        }
        let addr = self.zeroed(size as usize) as usize as u64;
        cells.insert((owner, seq), addr);
        addr
    }

    pub(in crate::image) fn register_aot_relocs(&self, relocs: &[SymbolSource]) {
        for sym in relocs {
            match *sym {
                SymbolSource::StaticCell { index, size } => {
                    let _ = self.cell(index, size);
                }
                SymbolSource::PromotedCell { owner, seq, size } => {
                    let _ = self.promoted_cell(owner, seq, size);
                }
                SymbolSource::ThreadSlot(index) => {
                    let _ = self.thread_slot(index);
                }
                _ => {}
            }
        }
    }

    pub(in crate::image) fn symbol_value(&self, sym: &SymbolSource) -> u64 {
        if !self.backend.is_jit() {
            return match *sym {
                SymbolSource::Slot(index) => index as u64,
                SymbolSource::Vtable(id) => u64::from(id.0),
                SymbolSource::Rodata(id) => u64::from(id.0),
                SymbolSource::StaticCell { index, size } => {
                    self.cell(index, size);
                    index as u64
                }
                SymbolSource::PromotedCell { owner, seq, size } => {
                    self.promoted_cell(owner, seq, size);
                    (u64::from(seq) << 32) | (owner as u64 & 0xffff_ffff)
                }
                SymbolSource::Host(ref host) => host.host_addr().unwrap_or(0) as u64,
                SymbolSource::Word(word) => word,
                SymbolSource::ThreadSlot(index) => self.thread_slot(index),
            };
        }
        match *sym {
            SymbolSource::Slot(index) => self.entry(index) as usize as u64,
            SymbolSource::Vtable(id) => lock(&self.vtable_addrs)[id.0 as usize],
            SymbolSource::Rodata(id) => lock(&self.rodata_addrs)[id.0 as usize],
            SymbolSource::StaticCell { index, size } => self.cell(index, size) as usize as u64,
            SymbolSource::PromotedCell { owner, seq, size } => self.promoted_cell(owner, seq, size),
            SymbolSource::Host(ref host) => host.host_addr().unwrap_or(0) as u64,
            SymbolSource::Word(word) => word,
            SymbolSource::ThreadSlot(index) => self.thread_slot(index),
        }
    }

    pub(in crate::image) fn reset_caches(&mut self) {
        self.bodies.clear();
        self.pass.stencils.clear();
        if let Some(aot) = self.aot() {
            lock(aot).stencils.clear();
        }
    }

    pub(in crate::image) fn cached_aot_template(
        &self,
        item: ItemRef,
        fp: FootprintId,
    ) -> Option<CachedAotTemplate> {
        self.aot()
            .and_then(|aot| lock(aot).stencils.get(&(item, fp)).cloned())
    }

    pub(in crate::image) fn cache_aot_template(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        (item, fp, rep): (ItemRef, FootprintId, usize),
        template: CachedAotTemplate,
    ) {
        if self.backend.is_aot()
            && cx.deferred_evals.is_empty()
            && !reached.list[rep].0.1.is_empty()
            && let Some(aot) = self.aot()
        {
            lock(aot).stencils.entry((item, fp)).or_insert(template);
        }
    }
}
