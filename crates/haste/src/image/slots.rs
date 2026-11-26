use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Image, ImageBackend, aot};
use crate::lower::{self, Slots, Word};
use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};
use crate::sync::lock;

impl Slots for Image {
    fn entry(&self, index: usize) -> i64 {
        if !self.backend.is_jit() {
            return index as i64;
        }
        self.slot(index) as *const AtomicUsize as i64
    }

    fn data(&self, bytes: &[u8]) -> i64 {
        if !self.backend.is_jit() {
            return 0;
        }
        let address = lock(&self.shared)
            .alloc(bytes.len())
            .unwrap_or_else(|error| panic!("{error}"));
        // SAFETY: alloc handed out `bytes.len()` writable bytes at address that nothing else uses, and they do not overlap `bytes`.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
        address as i64
    }

    fn table(&self, words: &[Word]) -> i64 {
        if !self.backend.is_jit() {
            return 0;
        }
        let mut tables = lock(&self.tables);
        if let Some(&address) = tables.get(words) {
            return address;
        }
        let value = |word: &Word| match *word {
            Word::Int(value) => value,
            Word::Code(index) => self.slot(index).load(Ordering::Acquire) as u64,
        };
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|word| value(word).to_le_bytes())
            .collect();
        let address = self.data(&bytes);
        for (at, word) in words.iter().enumerate() {
            if let Word::Code(index) = *word {
                self.register_code_fixup(index, address as usize + at * 8);
            }
        }
        tables.insert(words.to_vec(), address);
        address
    }

    fn zeroed(&self, len: usize) -> i64 {
        if !self.backend.is_jit() {
            return 0;
        }
        lock(&self.state)
            .alloc(len)
            .unwrap_or_else(|error| panic!("{error}")) as i64
    }

    fn want(&self, instance: Instance) {
        if self.backend.is_lazy_jit() {
            lock(&self.wanted).push((std::thread::current().id(), instance));
        }
    }

    fn wanted_len(&self) -> usize {
        if !self.backend.is_lazy_jit() {
            return 0;
        }
        let me = std::thread::current().id();
        lock(&self.wanted)
            .iter()
            .filter(|(thread, _)| *thread == me)
            .count()
    }

    fn bodies(&self) -> &lower::Bodies {
        &self.bodies
    }

    fn allow_inline(&self) -> bool {
        self.backend.is_optimized_jit()
            || (crate::profile() == crate::outline::tree::Profile::Release
                && match &self.backend {
                    ImageBackend::Check => false,
                    ImageBackend::EagerJit { fast } | ImageBackend::LazyJit { fast, .. } => !fast,
                    ImageBackend::Aot(_) => true,
                })
    }

    fn cell(&self, index: usize, size: u32) -> i64 {
        if let Some(aot) = self.aot() {
            lock(aot)
                .statics
                .entry(index)
                .or_insert(aot::AotStatic::Local(size));
            return 0;
        }
        if self.backend.is_check() {
            return 0;
        }
        let mut cells = lock(&self.cells);
        if let Some(&address) = cells.get(&index) {
            return address;
        }
        let address = self.zeroed(size as usize + 1);
        cells.insert(index, address);
        address
    }

    fn low_entries(&self) -> bool {
        self.backend.is_aot() || cfg!(target_arch = "aarch64") || self._low.is_low()
    }

    fn rodata(&self, bytes: &[u8], relocs: &[lower::RodataReloc]) -> lower::RodataId {
        self.intern_rodata(bytes, relocs)
    }

    fn static_rodata(&self, index: usize, bytes: &[u8]) -> lower::RodataId {
        self.intern_static_rodata(index, bytes)
    }

    fn vtable_id(&self, entries: &[lower::VtableEntry]) -> lower::VtableId {
        self.intern_vtable(entries)
    }

    fn vtable_entries(&self, id: lower::VtableId) -> Option<Vec<lower::VtableEntry>> {
        if let Some(aot) = self.aot() {
            return lock(aot).vtables.get(id.0 as usize).cloned();
        }
        lock(&self.vtable_entries).get(id.0 as usize).cloned()
    }

    fn resolve_symbol(&self, sym: &lower::SymbolSource) -> u64 {
        self.symbol_value(sym)
    }

    fn encode_plan(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        is_sysroot: bool,
        plan: &lower::RelocPlan,
    ) -> Option<Vec<u8>> {
        self.encode_aot_reloc_plan(cx.outline, &cx.types, reached, is_sysroot, plan)
    }

    fn decode_plan(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        is_sysroot: bool,
        bytes: &[u8],
    ) -> Option<lower::RelocPlan> {
        self.decode_aot_reloc_plan(cx.outline, &cx.types, reached, is_sysroot, bytes)
    }
}
