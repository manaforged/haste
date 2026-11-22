//! One process image: a call table with one entry per reached function, the code it points at, and constant data.

mod aot;
mod expand;
mod global_asm;
mod lazy;
pub mod perf;
mod run;
mod script;
mod stencil;
mod stencil_jit;
pub(crate) use stencil::{DICT_MIN_BYTES, DICT_MIN_FANOUT, verify_stencil};
mod stub;
pub mod watch;
mod workers;

pub use expand::{Job, run_macros};
pub use lazy::{Driver, Ended, Jit, Program, entries, evaluator, run_on_demand, serve};
#[cfg(unix)]
pub(crate) use run::{close_on_exec, fork_child, pipe, wait};
pub(crate) use run::interrupted;
pub use script::Script;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashMap as HashMap;

use anyhow::Result;
use haste_codegen::{Arena, Memory};

use crate::host::HostSymbol;
use crate::lower::{self, Slots, Word};
use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};
use crate::sync::lock;

const SHARED_BYTES: usize = 4 << 30;
const STATE_BYTES: usize = 1 << 30;
const CODE_BYTES: usize = 1 << 30;
const TABLE_SLOTS: usize = 1 << 22;

/// How `Image::build` emits code: live JIT pages, AOT object relocations, or check-only verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageMode {
    Jit,
    Aot,
    Check,
}

pub(super) enum ImageBackend {
    Check,
    EagerJit { fast: bool },
    LazyJit { fast: bool },
    Aot(Mutex<aot::AotState>),
}

impl ImageBackend {
    fn from_mode(mode: ImageMode, reached_len: usize) -> Self {
        match mode {
            ImageMode::Check => Self::Check,
            ImageMode::Jit => Self::EagerJit { fast: false },
            ImageMode::Aot => Self::Aot(aot::AotState::new(reached_len)),
        }
    }

    pub(super) fn mode(&self) -> ImageMode {
        match self {
            Self::Check => ImageMode::Check,
            Self::EagerJit { .. } | Self::LazyJit { .. } => ImageMode::Jit,
            Self::Aot(_) => ImageMode::Aot,
        }
    }

    pub(super) fn is_jit(&self) -> bool {
        matches!(self, Self::EagerJit { .. } | Self::LazyJit { .. })
    }

    pub(super) fn is_lazy_jit(&self) -> bool {
        matches!(self, Self::LazyJit { .. })
    }

    pub(super) fn is_fast_jit(&self) -> bool {
        matches!(self, Self::EagerJit { fast: true } | Self::LazyJit { fast: true })
    }

    pub(super) fn is_aot(&self) -> bool {
        matches!(self, Self::Aot(_))
    }

    pub(super) fn is_check(&self) -> bool {
        matches!(self, Self::Check)
    }

    pub(super) fn aot(&self) -> Option<&Mutex<aot::AotState>> {
        match self {
            Self::Aot(state) => Some(state),
            _ => None,
        }
    }
}

struct OptArena(Option<Arena>);

impl OptArena {
    fn alloc(&mut self, len: usize) -> std::result::Result<usize, haste_codegen::Error> {
        match &mut self.0 {
            Some(arena) => arena.alloc(len),
            None => Ok(0),
        }
    }

    fn is_low(&self) -> bool {
        self.0.as_ref().is_some_and(Arena::is_low)
    }
}

/// The call table, code memory and data of the running program, in memory shared with children forked from this process, and the cells the program writes, in memory each run holds its own copy of.
pub struct Image {
    table: usize,
    _low: OptArena,
    capacity: usize,
    pass: Pass,
    sizes: Vec<u32>,
    times: Vec<u64>,
    stubs: Memory,
    shared: Mutex<OptArena>,
    unwind_head: usize,
    state: Mutex<OptArena>,
    tables: Mutex<HashMap<Vec<Word>, i64>>,
    fixups: Mutex<HashMap<usize, Vec<usize>>>,
    registered: AtomicUsize,
    cells: Mutex<HashMap<usize, i64>>,
    rodata_map: Mutex<HashMap<(Vec<u8>, Vec<lower::RodataReloc>), lower::RodataId>>,
    static_rodata: Mutex<HashMap<usize, lower::RodataId>>,
    rodata_addrs: Mutex<Vec<u64>>,
    vtable_map: Mutex<HashMap<Vec<lower::VtableEntry>, lower::VtableId>>,
    vtable_addrs: Mutex<Vec<u64>>,
    vtable_entries: Mutex<Vec<Vec<lower::VtableEntry>>>,
    promoted_cells: Mutex<HashMap<(usize, u32), u64>>,
    thread_slots: Mutex<HashMap<usize, u64>>,
    wanted: Mutex<Vec<(std::thread::ThreadId, Instance)>>,
    bodies: lower::Bodies,
    backend: ImageBackend,
    file_index: std::sync::OnceLock<crate::sema::summary::codec::FileIndex>,
}

pub(super) struct FrozenStencil {
    pub is_sysroot: bool,
    pub key: u64,
    pub fp: u128,
    pub code: std::sync::Arc<haste_codegen::Code>,
    pub plan: Vec<u8>,
}

/// What one compile pass writes: its code memory, its stencil cache, sparse `(index, size, ns)` records, and what workers made ahead for a lazy batch, by index.
#[derive(Default)]
pub(super) struct Pass {
    memory: Memory,
    stencils: stencil::StencilState,
    records: Vec<(usize, u32, u64)>,
    frozen_stencils: Vec<FrozenStencil>,
    ready: HashMap<usize, stencil_jit::Ready>,
}

impl Pass {
    /// Adds `size` code bytes and `ns` nanoseconds of compile time to reached function `index`.
    pub(super) fn record_size(&mut self, index: usize, size: u32, ns: u64) {
        if size > 0 || ns > 0 {
            self.records.push((index, size, ns));
        }
    }

    /// Moves the code chunks, sparse compile records, and stencil entries of a worker's pass into this one.
    fn merge(&mut self, other: Pass) {
        self.memory.merge(other.memory);
        self.records.extend(other.records);
        self.frozen_stencils.extend(other.frozen_stencils);
        for (key, template) in other.stencils.templates {
            self.stencils.templates.entry(key).or_insert(template);
        }
        for (key, alias) in other.stencils.aliases {
            self.stencils.aliases.entry(key).or_insert(alias);
        }
    }
}

impl Image {
    /// Returns whether this image compiled any new sysroot stencils not yet in a substrate.
    pub(crate) fn has_new_frozen_stencils(&self) -> bool {
        !self.pass.frozen_stencils.is_empty()
    }

    /// Appends all newly compiled sysroot `Effort::Full` stencils from this image into `toolchain` and dependency stencils into `deps`.
    pub(crate) fn export_sysroot_stencils(
        &self,
        mut toolchain: Option<&mut crate::substrate::SubstrateBuilder>,
        mut deps: Option<&mut crate::substrate::SubstrateBuilder>,
    ) -> Result<()> {
        for stencil in &self.pass.frozen_stencils {
            let target = if stencil.is_sysroot { toolchain.as_deref_mut() } else { deps.as_deref_mut() };
            if let Some(builder) = target {
                builder.add_code_stencil_with_plan(stencil.key, stencil.fp, &stencil.code, &stencil.plan)?;
            }
        }
        Ok(())
    }

    pub(super) fn aot_rodata_entry(&self, id: lower::RodataId) -> Option<(Vec<u8>, Vec<lower::RodataReloc>)> {
        let aot = self.aot()?;
        lock(aot).rodata.get(id.0 as usize).cloned()
    }

    pub(super) fn encode_aot_reloc_plan(&self, outline: &crate::outline::Outline, types: &crate::sema::ty::Types, reached: &Reached, is_sysroot: bool, plan: &lower::RelocPlan) -> Option<Vec<u8>> {
        if !self.backend.is_aot() {
            return None;
        }
        let key_of = if is_sysroot { crate::substrate::sysroot_only_item_key } else { crate::substrate::sysroot_item_key };
        crate::substrate::relocs::encode_reloc_plan(outline, types, reached, plan, !is_sysroot, key_of, |id| self.aot_rodata_entry(id), |id| self.vtable_entries(id))
    }

    pub(super) fn decode_aot_reloc_plan(&self, outline: &crate::outline::Outline, types: &crate::sema::ty::Types, reached: &Reached, is_sysroot: bool, bytes: &[u8]) -> Option<lower::RelocPlan> {
        if !self.backend.is_aot() || bytes.is_empty() {
            return None;
        }
        let index = self.file_index.get_or_init(|| crate::sema::summary::codec::FileIndex::build(outline));
        crate::substrate::relocs::decode_reloc_plan(outline, types, reached, self, index, is_sysroot, bytes).ok().flatten()
    }

    pub(super) fn register_code_fixup(&self, index: usize, at: usize) {
        let mut fixups = lock(&self.fixups);
        fixups.entry(index).or_default().push(at);
        self.registered.fetch_add(1, Ordering::SeqCst);
        patch(&[at], self.slot(index).load(Ordering::SeqCst));
    }
}

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
        let address = lock(&self.shared).alloc(bytes.len()).unwrap_or_else(|error| panic!("{error}"));
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
        let bytes: Vec<u8> = words.iter().flat_map(|word| value(word).to_le_bytes()).collect();
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
        lock(&self.state).alloc(len).unwrap_or_else(|error| panic!("{error}")) as i64
    }

    fn want(&self, instance: Instance) {
        if self.backend.is_lazy_jit() {
            lock(&self.wanted).push((std::thread::current().id(), instance));
        }
    }

    fn wanted_len(&self) -> usize {
        let me = std::thread::current().id();
        lock(&self.wanted).iter().filter(|(thread, _)| *thread == me).count()
    }

    fn bodies(&self) -> &lower::Bodies {
        &self.bodies
    }

    fn allow_inline(&self) -> bool {
        match &self.backend {
            ImageBackend::Check => false,
            ImageBackend::EagerJit { fast } => !fast,
            ImageBackend::LazyJit { fast } => !fast && crate::profile() == crate::outline::tree::Profile::Release,
            ImageBackend::Aot(_) => true,
        }
    }

    fn cell(&self, index: usize, size: u32) -> i64 {
        if let Some(aot) = self.aot() {
            lock(aot).statics.entry(index).or_insert(aot::AotStatic::Local(size));
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

    fn encode_plan(&self, cx: &Context<'_>, reached: &Reached, is_sysroot: bool, plan: &lower::RelocPlan) -> Option<Vec<u8>> {
        self.encode_aot_reloc_plan(cx.outline, &cx.types, reached, is_sysroot, plan)
    }

    fn decode_plan(&self, cx: &Context<'_>, reached: &Reached, is_sysroot: bool, bytes: &[u8]) -> Option<lower::RelocPlan> {
        self.decode_aot_reloc_plan(cx.outline, &cx.types, reached, is_sysroot, bytes)
    }
}

pub(super) fn resolve_host_symbol(cx: &Context<'_>, item: crate::names::ItemRef, aot: bool) -> Result<(HostSymbol, usize), String> {
    let name = cx.symbol_name(item);
    let resolved = name.as_deref().and_then(|n| HostSymbol::resolve_for(n, aot)).and_then(|sym| {
        if aot {
            Some((sym, 0))
        } else {
            sym.host_addr().map(|addr| (sym, addr))
        }
    });
    resolved.ok_or_else(|| format!("{}: no host symbol links to it", cx.name(item)))
}

/// Whether the foreign item `item` links weakly, so a missing symbol reads as null.
pub(super) fn weak(cx: &Context<'_>, item: crate::names::ItemRef) -> bool {
    cx.attr_value(item, "linkage") == Some("extern_weak")
}

impl Image {
    /// An image with one table entry per reached function and no code yet.
    pub fn new(reached: &Reached) -> Result<Image> {
        Self::create(reached, ImageMode::Jit, (SHARED_BYTES, STATE_BYTES, TABLE_SLOTS))
    }

    pub(super) fn create(reached: &Reached, mode: ImageMode, (shared_bytes, state_bytes, table_slots): (usize, usize, usize)) -> Result<Image> {
        let (shared, unwind_head, low, table, state, capacity) = if mode == ImageMode::Jit {
            let capacity = reached.list.len().max(table_slots);
            let mut low = Arena::low(capacity * size_of::<AtomicUsize>())?;
            let table = low.alloc(capacity * size_of::<AtomicUsize>())?;
            let mut shared = Arena::shared(shared_bytes)?;
            let unwind_head = shared.alloc(size_of::<AtomicUsize>())?;
            crate::host::unwind::set_jit_unwind_head(unwind_head);
            (OptArena(Some(shared)), unwind_head, OptArena(Some(low)), table, OptArena(Some(Arena::private(state_bytes)?)), capacity)
        } else {
            (OptArena(None), 0, OptArena(None), 0, OptArena(None), 0)
        };
        Ok(Image {
            table,
            _low: low,
            capacity,
            pass: Pass::default(),
            sizes: vec![0; reached.list.len()],
            times: vec![0; reached.list.len()],
            stubs: Memory::default(),
            shared: Mutex::new(shared),
            unwind_head,
            state: Mutex::new(state),
            tables: Mutex::default(),
            fixups: Mutex::default(),
            registered: AtomicUsize::new(0),
            cells: Mutex::default(),
            rodata_map: Mutex::default(),
            static_rodata: Mutex::default(),
            rodata_addrs: Mutex::default(),
            vtable_map: Mutex::default(),
            vtable_addrs: Mutex::default(),
            vtable_entries: Mutex::default(),
            promoted_cells: Mutex::default(),
            thread_slots: Mutex::default(),
            wanted: Mutex::default(),
            bodies: lower::Bodies::default(),
            backend: ImageBackend::from_mode(mode, reached.list.len()),
            file_index: std::sync::OnceLock::new(),
        })
    }

    pub(super) fn aot(&self) -> Option<&Mutex<aot::AotState>> {
        self.backend.aot()
    }

    pub(super) fn compile_target(&self) -> haste_codegen::Target {
        let base = if self.backend.is_fast_jit() {
            haste_codegen::Target::host().effort(haste_codegen::Effort::Quick)
        } else {
            crate::host_target()
        };
        base.direct_calls(self.backend.is_aot())
    }

    pub(super) fn compile_target_for(&self, cx: &Context<'_>, item: crate::names::ItemRef) -> haste_codegen::Target {
        let base = self.compile_target();
        if cx.attr_value(item, "cold").is_some() { base.effort(haste_codegen::Effort::Quick) } else { base }
    }

    /// The encoded machine-code byte length of each reached function, or zero when not compiled.
    pub fn code_sizes(&self) -> &[u32] {
        &self.sizes
    }

    /// The lowering and encoding wall time in nanoseconds of each reached function, or zero when not compiled.
    pub fn compile_times(&self) -> &[u64] {
        &self.times
    }

    pub(super) fn apply_records(&mut self) {
        for (index, size, ns) in self.pass.records.drain(..) {
            if index >= self.sizes.len() {
                self.sizes.resize(index + 1, 0);
                self.times.resize(index + 1, 0);
            }
            self.sizes[index] = self.sizes[index].saturating_add(size);
            self.times[index] = self.times[index].saturating_add(ns);
        }
    }

    /// Takes every unreached instance that lowering noted, and gives those this thread noted.
    pub(super) fn take_wanted(&self) -> Vec<Instance> {
        let me = std::thread::current().id();
        std::mem::take(&mut *lock(&self.wanted)).into_iter().filter(|(thread, _)| *thread == me).map(|(_, instance)| instance).collect()
    }

    /// Runs `run` with this image's pass taken out, so that `run` lowers through the shared image while it writes into the pass.
    pub(super) fn with_pass<R>(&mut self, run: impl FnOnce(&Image, &mut Pass) -> R) -> R {
        let mut pass = std::mem::take(&mut self.pass);
        let result = run(self, &mut pass);
        self.pass = pass;
        result
    }

    /// The dense number of the per-thread cell of the `#[thread_local]` static that reached function `index` initializes, given out in order of first use and kept for the life of the image.
    fn thread_slot(&self, index: usize) -> u64 {
        let mut slots = lock(&self.thread_slots);
        let next = slots.len() as u64 + 1;
        *slots.entry(index).or_insert(next)
    }

    fn slot(&self, index: usize) -> &AtomicUsize {
        assert!(index < self.capacity, "call table index {index} past its {} slots", self.capacity);
        // SAFETY: index is below capacity, and new allocated capacity zeroed, aligned AtomicUsize slots at table, in a mapping that lives as long as the process.
        unsafe { &*(self.table as *const AtomicUsize).add(index) }
    }

    /// An image of every function in `reached`, compiled under `mode`, with why each one that did not compile failed.
    pub(crate) fn build(cx: &mut Context<'_>, reached: &Reached, mode: ImageMode) -> Result<(Image, HashMap<usize, String>)> {
        let mut image = Self::create(reached, mode, (SHARED_BYTES, STATE_BYTES, reached.list.len().max(1)))?;
        let all: Vec<usize> = match mode {
            ImageMode::Check => (0..reached.list.len()).filter(|&idx| !cx.outline.is_fixed(reached.list[idx].0.0.krate)).collect(),
            ImageMode::Jit | ImageMode::Aot => (0..reached.list.len()).collect(),
        };
        let broken = image.compile(cx, reached, &all).into_iter().collect();
        image.bodies = lower::Bodies::default();
        Ok((image, broken))
    }

    pub(crate) fn build_aot(cx: &mut Context<'_>, reached: &Reached, roots: [usize; 2]) -> Result<(Image, HashMap<usize, String>)> {
        Self::build_aot_streaming(cx, reached, roots)
    }

    /// Lowers and encodes the reached functions `indices` that type-checked, on every core when there are many, links them, and returns the errors of those that did not compile, by index.
    pub fn compile(&mut self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize]) -> Vec<(usize, String)> {
        let reset = !indices.is_empty() && indices.len() < reached.list.len();
        self.compile_inner(cx, reached, indices, reset)
    }

    fn compile_inner(&mut self, cx: &mut Context<'_>, reached: &Reached, indices: &[usize], reset: bool) -> Vec<(usize, String)> {
        if self.sizes.len() < reached.list.len() {
            self.sizes.resize(reached.list.len(), 0);
            self.times.resize(reached.list.len(), 0);
        }
        if let Some(aot) = self.aot() {
            lock(aot).ensure_len(reached.list.len());
        }
        if reset {
            self.reset_caches();
        }
        let mut errors = Vec::new();
        let work = self.with_pass(|image, pass| image.prepare_work(cx, reached, indices, (&mut errors, Some(&mut pass.memory))));
        errors.extend(self.compile_all(cx, reached, &work));
        self.apply_records();
        self.link();
        errors
    }

    fn link(&self) {
        if !self.backend.is_jit() {
            return;
        }
        for (&index, addresses) in lock(&self.fixups).iter() {
            patch(addresses, self.slot(index).load(Ordering::Acquire));
        }
    }

    /// Points slot `index` at `code` and rewrites every constant table word that names it, so neither a call through the slot nor one through a table reaches its stub again.
    fn publish(&self, index: usize, code: usize) {
        if !self.backend.is_jit() {
            return;
        }
        self.slot(index).store(code, Ordering::SeqCst);
        if self.registered.load(Ordering::SeqCst) == 0 {
            return;
        }
        if let Some(addresses) = lock(&self.fixups).get(&index) {
            patch(addresses, code);
        }
    }

    fn compile_one(&mut self, cx: &mut Context<'_>, reached: &Reached, index: usize) -> Result<usize> {
        self.with_pass(|image, pass| image.compile_jit_one(cx, reached, index, pass))
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        crate::host::unwind::clear_jit_unwind_head(self.unwind_head);
    }
}

/// Writes `code` into each constant table word at `addresses`.
fn patch(addresses: &[usize], code: usize) {
    for &address in addresses {
        // SAFETY: address is an 8-aligned word of a constant table that data handed out, which stays mapped for the life of the process; an aligned word store is single-copy atomic, so a thread that reads it sees the old entry or the new one, and both run the same function.
        unsafe { (*(address as *const AtomicUsize)).store(code, Ordering::Release) };
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticInit {
    Empty = 0,
    Filled = 1,
    Filling = 2,
    Poisoned = 3,
}

struct FillGuard<'a> {
    state: &'a std::sync::atomic::AtomicU8,
    flag: usize,
    done: bool,
}

impl Drop for FillGuard<'_> {
    fn drop(&mut self) {
        thread_state().filling.retain(|&at| at != self.flag);
        if !self.done {
            self.state.store(StaticInit::Poisoned as u8, Ordering::Release);
        }
    }
}

/// What the host keeps for each thread of the program: its cells of `#[thread_local]` statics by thread slot, zero for one not made yet, and the statics it is filling.
#[derive(Default)]
struct ThreadState {
    cells: Vec<usize>,
    filling: Vec<usize>,
}

thread_local! {
    static THREAD_STATE: std::cell::Cell<*mut ThreadState> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// The calling thread's state, made on first use and never freed, since the program's thread exit code may still read it after the host's own thread locals are gone.
fn thread_state() -> &'static mut ThreadState {
    THREAD_STATE.with(|state| {
        if state.get().is_null() {
            state.set(Box::into_raw(Box::default()));
        }
        // SAFETY: the state is this thread's own leaked allocation, never shared with another thread.
        unsafe { &mut *state.get() }
    })
}

/// Runs the initializer at `code` of the static whose state byte is at `flag`, unless another call filled it: a thread that finds another thread filling it waits, and the thread filling it reads it as it is.
pub(crate) extern "C" fn fill_once(flag: usize, code: usize) {
    // SAFETY: flag is the state byte after a static's value in a cell that lives as long as the program.
    let state = unsafe { &*(flag as *const std::sync::atomic::AtomicU8) };
    loop {
        match state.compare_exchange(StaticInit::Empty as u8, StaticInit::Filling as u8, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                thread_state().filling.push(flag);
                let mut guard = FillGuard { state, flag, done: false };
                // SAFETY: code is the generated initializer of this static, a function that takes and returns nothing.
                let fill: extern "C" fn() = unsafe { std::mem::transmute::<usize, extern "C" fn()>(code) };
                fill();
                guard.done = true;
                state.store(StaticInit::Filled as u8, Ordering::Release);
                return;
            }
            Err(v) if v == StaticInit::Filled as u8 => return,
            Err(v) if v == StaticInit::Poisoned as u8 => panic!("static initializer panicked"),
            Err(_) if thread_state().filling.contains(&flag) => return,
            Err(_) => std::thread::yield_now(),
        }
    }
}

/// The calling thread's cell of `size` zeroed bytes for the `#[thread_local]` static with thread slot `slot`, running `*init_slot` once on first allocation when non-zero.
pub(crate) extern "C" fn thread_cell(slot: usize, size: usize, init_slot: usize) -> usize {
    let cell = {
        let state = thread_state();
        if let Some(&cell) = state.cells.get(slot)
            && cell != 0
        {
            return cell;
        }
        new_cell(state, slot, size)
    };
    if init_slot != 0 {
        // SAFETY: init_slot points to the 8-byte function slot holding this static's zero-arg initializer.
        let code = unsafe { *(init_slot as *const usize) };
        let fill: extern "C" fn() = unsafe { std::mem::transmute::<usize, extern "C" fn()>(code) };
        fill();
    }
    cell
}

/// Makes the calling thread's cell of `size` zeroed bytes for thread slot `slot`.
#[cold]
fn new_cell(state: &mut ThreadState, slot: usize, size: usize) -> usize {
    let layout = std::alloc::Layout::from_size_align(size.max(1), 64).unwrap_or_else(|_| std::alloc::Layout::new::<u8>());
    // SAFETY: layout has a nonzero size.
    let cell = unsafe { std::alloc::alloc_zeroed(layout) };
    if cell.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    if state.cells.len() <= slot {
        state.cells.resize(slot + 1, 0);
    }
    state.cells[slot] = cell as usize;
    cell as usize
}
