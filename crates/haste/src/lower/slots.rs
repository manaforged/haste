use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};

use super::reloc::RelocPlan;
use super::stencil::{Bodies, RodataId, RodataReloc, SymbolSource, VtableEntry, VtableId};

/// Where lowered code finds what it refers to: call table entries and constant data.
pub trait Slots {
    /// The address of the table entry that holds the code address of reached function `index`.
    fn entry(&self, index: usize) -> i64;
    /// The address of a 16-byte-aligned copy of `bytes` that lives as long as the program.
    fn data(&self, bytes: &[u8]) -> i64;
    /// The address of `len` zeroed, 16-byte-aligned bytes that each run of the program writes for itself.
    fn zeroed(&self, len: usize) -> i64;
    /// The address of a table of `words` that lives as long as the program, each `Word::Code` holding a code address once every function is compiled.
    fn table(&self, words: &[Word]) -> i64;
    /// The address of the zeroed, 16-byte-aligned cell of `size` bytes and one more that holds the value of the static whose initializer is reached function `index`, then whether it is filled.
    fn cell(&self, index: usize, size: u32) -> i64;
    /// Notes that lowering used `instance`, which was not reached, so a caller that reaches functions on demand can reach it and lower again.
    fn want(&self, _instance: Instance) {}
    /// The count of unreached function instances recorded so far.
    fn wanted_len(&self) -> usize {
        0
    }
    /// The bodies that calls copy inline.
    fn bodies(&self) -> &Bodies;
    /// Whether call table entries live in the low 2 GiB address range.
    fn low_entries(&self) -> bool {
        false
    }
    /// Interns a read-only blob with optional intra-rodata relocations and returns its symbol id.
    fn rodata(&self, bytes: &[u8], _relocs: &[RodataReloc]) -> RodataId {
        self.data(bytes);
        RodataId(0)
    }
    /// Interns read-only bytes for immutable static `index` without deduplicating across distinct statics.
    fn static_rodata(&self, _index: usize, bytes: &[u8]) -> RodataId {
        self.rodata(bytes, &[])
    }
    /// Whether calls may inline bodies through this image.
    fn allow_inline(&self) -> bool {
        true
    }
    /// Interns a vtable layout and returns its symbol id.
    fn vtable_id(&self, _entries: &[VtableEntry]) -> VtableId {
        VtableId(0)
    }
    /// Returns the entries of an interned vtable `id`, if available.
    fn vtable_entries(&self, _id: VtableId) -> Option<Vec<VtableEntry>> {
        None
    }
    /// Encodes `plan` into substrate bytes, toolchain items only when `is_sysroot`, or `None` when this image keeps no plans across builds.
    fn encode_plan(
        &self,
        _cx: &Context<'_>,
        _reached: &Reached,
        _is_sysroot: bool,
        _plan: &RelocPlan,
    ) -> Option<Vec<u8>> {
        None
    }
    /// Decodes plan bytes that `encode_plan` wrote, or `None` when they do not resolve in this image.
    fn decode_plan(
        &self,
        _cx: &Context<'_>,
        _reached: &Reached,
        _is_sysroot: bool,
        _bytes: &[u8],
    ) -> Option<RelocPlan> {
        None
    }
    /// Resolves a symbolic relocation entry to its concrete 64-bit value in this image.
    fn resolve_symbol(&self, sym: &SymbolSource) -> u64 {
        match *sym {
            SymbolSource::Slot(index) => self.entry(index) as u64,
            SymbolSource::StaticCell { index, size } => self.cell(index, size) as u64,
            SymbolSource::PromotedCell { size, .. } => self.zeroed(size as usize) as u64,
            SymbolSource::Host(ref host) => host.host_addr().unwrap_or(0) as u64,
            SymbolSource::Word(word) => word,
            SymbolSource::ThreadSlot(index) => index as u64,
            SymbolSource::Vtable(_) | SymbolSource::Rodata(_) => 0,
        }
    }
}

/// One machine word of a constant table: a number, or the code address of reached function `index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Word {
    Int(u64),
    Code(usize),
}
