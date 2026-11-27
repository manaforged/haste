mod aot;
mod backend;
mod expand;
mod global_asm;
mod lazy;
pub mod perf;
mod run;
mod script;
mod slots;
mod statics;
mod stencil;
mod stencil_jit;
pub(crate) use stencil::{DICT_MIN_BYTES, DICT_MIN_FANOUT, verify_stencil};
mod stub;
pub mod watch;
mod workers;

pub(crate) use aot::cache as aot_cache;
pub(super) use backend::ImageBackend;
pub use backend::ImageMode;
pub use expand::{Job, run_macros};
pub use lazy::{Driver, Ended, Jit, Program, ReloadError, entries, evaluator, serve};
pub(crate) use run::interrupted;
#[cfg(unix)]
pub(crate) use run::{close_on_exec, fork_child, pipe, wait};
pub use script::Script;
pub(crate) use statics::{StaticInit, fill_once, thread_cell};

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use haste_codegen::{Arena, Memory};
use rustc_hash::FxHashMap as HashMap;

use crate::host::HostSymbol;
use crate::lower::{self, Slots, Word};
use crate::sema::Context;
use crate::sema::reach::{Instance, Reached};
use crate::sync::lock;

const SHARED_BYTES: usize = 4 << 30;
const STATE_BYTES: usize = 1 << 30;
const CODE_BYTES: usize = 1 << 30;
const TABLE_SLOTS: usize = 1 << 22;

struct OptArena(Option<Arena>);

impl OptArena {
    fn alloc(&mut self, len: usize) -> std::result::Result<usize, haste_codegen::Error> {
        match &mut self.0 {
            Some(arena) => arena.alloc(len),
            None => Ok(0),
        }
    }

    fn seal(&mut self) {
        if let Some(arena) = &mut self.0 {
            arena.seal();
        }
    }

    fn is_low(&self) -> bool {
        self.0.as_ref().is_some_and(Arena::is_low)
    }
}

/// The call table, code memory and data of the running program, in shared and per-run memory.
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
    stub_lock: usize,
    state: Mutex<OptArena>,
    tables: Mutex<HashMap<Vec<Word>, i64>>,
    fixups: Mutex<HashMap<usize, Vec<usize>>>,
    registered: AtomicUsize,
    cells: Mutex<HashMap<usize, i64>>,
    rodata_map: Mutex<HashMap<(Vec<u8>, Vec<lower::RodataReloc>), lower::RodataId>>,
    static_rodata: Mutex<HashMap<usize, lower::RodataId>>,
    rodata_addrs: Mutex<Vec<u64>>,
    rodata_entries: Mutex<Vec<(Vec<u8>, Vec<lower::RodataReloc>)>>,
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

#[derive(Default)]
pub(super) struct Pass {
    memory: Memory,
    stencils: stencil::StencilState,
    records: Vec<(usize, u32, u64)>,
    frozen_stencils: Vec<FrozenStencil>,
    ready: HashMap<usize, stencil_jit::Ready>,
}

impl Pass {
    pub(super) fn record_size(&mut self, index: usize, size: u32, ns: u64) {
        if size > 0 || ns > 0 {
            self.records.push((index, size, ns));
        }
    }

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
            let target = if stencil.is_sysroot {
                toolchain.as_deref_mut()
            } else {
                deps.as_deref_mut()
            };
            if let Some(builder) = target {
                builder.add_code_stencil_with_plan(
                    stencil.key,
                    stencil.fp,
                    &stencil.code,
                    &stencil.plan,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn rodata_entry(
        &self,
        id: lower::RodataId,
    ) -> Option<(Vec<u8>, Vec<lower::RodataReloc>)> {
        if let Some(aot) = self.aot() {
            return lock(aot).rodata.get(id.0 as usize).cloned();
        }
        lock(&self.rodata_entries).get(id.0 as usize).cloned()
    }

    pub(super) fn encode_aot_reloc_plan(
        &self,
        outline: &crate::outline::Outline,
        types: &crate::sema::ty::Types,
        reached: &Reached,
        is_sysroot: bool,
        plan: &lower::RelocPlan,
    ) -> Option<Vec<u8>> {
        if !self.backend.is_aot() && !self.backend.is_lazy_jit() {
            return None;
        }
        let key_of = if is_sysroot {
            crate::substrate::sysroot_only_item_key
        } else {
            crate::substrate::sysroot_item_key
        };
        crate::substrate::relocs::encode_reloc_plan(
            outline,
            types,
            reached,
            plan,
            !is_sysroot,
            key_of,
            |id| self.rodata_entry(id),
            |id| self.vtable_entries(id),
        )
    }

    pub(super) fn decode_aot_reloc_plan(
        &self,
        outline: &crate::outline::Outline,
        types: &crate::sema::ty::Types,
        reached: &Reached,
        is_sysroot: bool,
        bytes: &[u8],
    ) -> Option<lower::RelocPlan> {
        if !self.backend.is_aot() || bytes.is_empty() {
            return None;
        }
        let index = self
            .file_index
            .get_or_init(|| crate::sema::summary::codec::FileIndex::build(outline));
        crate::substrate::relocs::decode_reloc_plan(
            outline, types, reached, self, index, is_sysroot, bytes,
        )
        .ok()
        .flatten()
    }

    pub(super) fn decode_lazy_reloc_plan(
        &self,
        cx: &Context<'_>,
        reached: &Reached,
        bytes: &[u8],
    ) -> Option<lower::RelocPlan> {
        if bytes.is_empty() {
            return None;
        }
        let index = cx.store().file_index(cx.outline);
        crate::substrate::relocs::decode_reloc_plan(
            cx.outline, &cx.types, reached, self, index, false, bytes,
        )
        .ok()
        .flatten()
    }

    pub(super) fn register_code_fixup(&self, index: usize, at: usize) {
        let mut fixups = lock(&self.fixups);
        fixups.entry(index).or_default().push(at);
        self.registered.fetch_add(1, Ordering::SeqCst);
        patch(&[at], self.slot(index).load(Ordering::SeqCst));
    }

    /// An image with one table entry per reached function and no code yet.
    pub fn new(reached: &Reached) -> Result<Image> {
        Self::create(
            reached,
            ImageMode::Jit,
            (SHARED_BYTES, STATE_BYTES, TABLE_SLOTS),
        )
    }

    pub(super) fn create(
        reached: &Reached,
        mode: ImageMode,
        (shared_bytes, state_bytes, table_slots): (usize, usize, usize),
    ) -> Result<Image> {
        let (shared, unwind_head, stub_lock, low, table, state, capacity) =
            if mode == ImageMode::Jit {
                let capacity = reached.list.len().max(table_slots);
                let mut low = Arena::low(capacity * size_of::<AtomicUsize>())?;
                let table = low.alloc(capacity * size_of::<AtomicUsize>())?;
                let mut shared = Arena::shared(shared_bytes)?;
                let unwind_head = shared.alloc(size_of::<AtomicUsize>())?;
                let stub_lock = shared.alloc(size_of::<std::sync::atomic::AtomicU32>())?;
                crate::host::unwind::set_jit_unwind_head(unwind_head);
                (
                    OptArena(Some(shared)),
                    unwind_head,
                    stub_lock,
                    OptArena(Some(low)),
                    table,
                    OptArena(Some(Arena::private(state_bytes)?)),
                    capacity,
                )
            } else {
                (OptArena(None), 0, 0, OptArena(None), 0, OptArena(None), 0)
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
            stub_lock,
            state: Mutex::new(state),
            tables: Mutex::default(),
            fixups: Mutex::default(),
            registered: AtomicUsize::new(0),
            cells: Mutex::default(),
            rodata_map: Mutex::default(),
            static_rodata: Mutex::default(),
            rodata_addrs: Mutex::default(),
            rodata_entries: Mutex::default(),
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
        let base = if self.backend.is_fast_jit() || self.backend.is_optimized_jit() {
            haste_codegen::Target::host().effort(haste_codegen::Effort::Quick)
        } else {
            crate::host_target()
        };
        base.direct_calls(self.backend.is_aot())
    }

    pub(super) fn compile_target_for(
        &self,
        cx: &Context<'_>,
        item: crate::names::ItemRef,
    ) -> haste_codegen::Target {
        let base = self.compile_target();
        if cx.has_attr(item, "cold") {
            base.effort(haste_codegen::Effort::Quick)
        } else {
            base
        }
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

    pub(super) fn take_wanted(&self) -> Vec<Instance> {
        let me = std::thread::current().id();
        std::mem::take(&mut *lock(&self.wanted))
            .into_iter()
            .filter(|(thread, _)| *thread == me)
            .map(|(_, instance)| instance)
            .collect()
    }

    pub(super) fn with_pass<R>(&mut self, run: impl FnOnce(&Image, &mut Pass) -> R) -> R {
        let mut pass = std::mem::take(&mut self.pass);
        let result = run(self, &mut pass);
        self.pass = pass;
        result
    }

    fn thread_slot(&self, index: usize) -> u64 {
        let mut slots = lock(&self.thread_slots);
        let next = slots.len() as u64 + 1;
        *slots.entry(index).or_insert(next)
    }

    fn slot(&self, index: usize) -> &AtomicUsize {
        assert!(
            index < self.capacity,
            "call table index {index} past its {} slots",
            self.capacity
        );
        // SAFETY: index is below capacity, and new allocated capacity zeroed, aligned AtomicUsize slots at table, in a mapping that lives as long as the process.
        unsafe { &*(self.table as *const AtomicUsize).add(index) }
    }

    /// An image of every function in `reached`, compiled under `mode`, with why each one that did not compile failed.
    pub(crate) fn build(
        cx: &mut Context<'_>,
        reached: &Reached,
        mode: ImageMode,
    ) -> Result<(Image, HashMap<usize, String>)> {
        if mode != ImageMode::Check {
            crate::host::native::load(cx.outline)?;
        }
        let mut image = Self::create(
            reached,
            mode,
            (SHARED_BYTES, STATE_BYTES, reached.list.len().max(1)),
        )?;
        let all: Vec<usize> = match mode {
            ImageMode::Check => (0..reached.list.len())
                .filter(|&idx| !cx.outline.is_fixed(reached.list[idx].0.0.krate))
                .collect(),
            ImageMode::Jit | ImageMode::Aot => (0..reached.list.len()).collect(),
        };
        let broken = image.compile(cx, reached, &all).into_iter().collect();
        image.bodies = lower::Bodies::default();
        Ok((image, broken))
    }

    pub(crate) fn build_aot(
        cx: &mut Context<'_>,
        reached: &Reached,
        roots: &[usize],
    ) -> Result<(Image, HashMap<usize, String>)> {
        if crate::compile_target() == haste_codegen::Target::host() {
            crate::host::native::load(cx.outline)?;
        }
        Self::build_aot_streaming(cx, reached, roots)
    }

    /// Lowers and encodes the reached functions `indices` that type-checked, on every core when there are many, links them, and returns the errors of those that did not compile, by index.
    pub fn compile(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        indices: &[usize],
    ) -> Vec<(usize, String)> {
        let reset = !indices.is_empty() && indices.len() < reached.list.len();
        self.compile_inner(cx, reached, indices, reset)
    }

    fn compile_inner(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        indices: &[usize],
        reset: bool,
    ) -> Vec<(usize, String)> {
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
        let work = self.with_pass(|image, pass| {
            image.prepare_work(cx, reached, indices, (&mut errors, Some(&mut pass.memory)))
        });
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

    pub(super) fn is_fast_lazy(&self) -> bool {
        matches!(
            self.backend,
            ImageBackend::LazyJit { fast: true, .. } | ImageBackend::LazyJit { optimize: true, .. }
        )
    }

    fn compile_one(
        &mut self,
        cx: &mut Context<'_>,
        reached: &Reached,
        index: usize,
    ) -> Result<(usize, bool)> {
        self.with_pass(|image, pass| image.compile_jit_one(cx, reached, index, pass))
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        crate::host::unwind::clear_jit_unwind_head(self.unwind_head);
    }
}

pub(super) fn resolve_host_symbol(
    cx: &Context<'_>,
    item: crate::names::ItemRef,
    aot: bool,
) -> Result<(HostSymbol, usize), String> {
    let name = cx.symbol_name(item);
    let resolved = name
        .as_deref()
        .and_then(|n| resolve_symbol_with_fallback(cx, item, n, aot))
        .and_then(|sym| {
            if aot {
                Some((sym, 0))
            } else {
                sym.host_addr().map(|addr| (sym, addr))
            }
        });
    resolved.ok_or_else(|| format!("{}: no host symbol links to it", cx.name(item)))
}

fn resolve_symbol_with_fallback(
    cx: &Context<'_>,
    item: crate::names::ItemRef,
    name: &str,
    aot: bool,
) -> Option<HostSymbol> {
    if let Some(lib) = cx.link_lib(item) {
        return HostSymbol::resolve_for_lib(name, Some(&lib), aot);
    }
    let libs = cx.crate_link_libs(item.krate);
    if libs.is_empty() {
        return HostSymbol::resolve_for_lib(name, None, aot);
    }
    // The live process is no oracle on Apple: a framework that an earlier lookup opened would look global.
    #[cfg(target_vendor = "apple")]
    let global = in_apple_libsystem(name);
    #[cfg(not(target_vendor = "apple"))]
    let global = HostSymbol::resolve_for_lib(name, None, false).is_some();
    if global {
        return HostSymbol::resolve_for_lib(name, None, aot);
    }
    for lib in libs {
        if HostSymbol::resolve_for_lib(name, Some(lib), false).is_some() {
            return HostSymbol::resolve_for_lib(name, Some(lib), aot);
        }
    }
    if aot && let Some(lib) = libs.first() {
        return HostSymbol::resolve_for_lib(name, Some(lib), true);
    }
    None
}

#[cfg(target_vendor = "apple")]
/// Whether the macOS system C runtime exports `name`, independent of what this process has loaded.
fn in_apple_libsystem(name: &str) -> bool {
    let Ok(cname) = std::ffi::CString::new(name) else {
        return false;
    };
    static LIBSYSTEM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let handle = *LIBSYSTEM.get_or_init(|| {
        // SAFETY: static NUL-terminated path to the macOS system C runtime library.
        unsafe {
            libc::dlopen(
                c"/usr/lib/libSystem.B.dylib".as_ptr(),
                libc::RTLD_LAZY | libc::RTLD_LOCAL,
            ) as usize
        }
    }) as *mut core::ffi::c_void;
    if handle.is_null() {
        return false;
    }
    // SAFETY: `handle` is an open dlopen handle and `cname` is NUL-terminated.
    unsafe { !libc::dlsym(handle, cname.as_ptr()).is_null() }
}

pub(super) fn weak(cx: &Context<'_>, item: crate::names::ItemRef) -> bool {
    cx.attr_value(item, "linkage") == Some("extern_weak")
}

fn patch(addresses: &[usize], code: usize) {
    for &address in addresses {
        // SAFETY: address is an 8-aligned word of a constant table that data handed out, which stays mapped for the life of the process; an aligned word store is single-copy atomic, so a thread that reads it sees the old entry or the new one, and both run the same function.
        unsafe { (*(address as *const AtomicUsize)).store(code, Ordering::Release) };
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap as HashMap;

    use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
    use crate::outline::build::{Loader, Others};
    use crate::outline::tree::{Crate, Decl, Env, Files, Module, State};
    use crate::outline::{Outline, Role, Root};
    use crate::sema::{Cache, Context};

    #[test]
    fn crate_level_link_framework_fallback_collects_and_resolves() {
        let dir = std::env::temp_dir().join(format!("haste_link_fallback_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        let lib_path = dir.join("lib.rs");
        let str_path = dir.join("string.rs");
        std::fs::write(
            &lib_path,
            concat!(
                "#[cfg_attr(all(any(target_os = \"macos\", target_os = \"linux\"), feature = \"link\"), link(name = \"CoreFoundation\", kind = \"framework\"))]\n",
                "extern \"C\" {}\n",
                "#[cfg(FALSE)]\n",
                "#[link(name = \"DisabledLib\", kind = \"framework\")]\n",
                "extern \"C\" {}\n",
                "pub mod string;\n",
            ),
        )
        .expect("write lib.rs");
        std::fs::write(
            &str_path,
            "unsafe extern \"C\" { pub fn CFStringGetCStringPtr(s: *const u8, enc: u32) -> *const u8; pub fn free(ptr: *mut u8); }\n",
        )
        .expect("write string.rs");

        let mut features = std::collections::BTreeSet::new();
        features.insert("link".to_owned());
        let env = Env {
            target: crate::graph::cfg::Target::host(),
            features,
            known: true,
            ..Env::default()
        };
        let files = Files::default();
        let mut tree = Crate {
            env,
            modules: vec![Module {
                file: usize::MAX,
                item: None,
                parent: None,
                name: String::new(),
                dir: dir.clone(),
                parts: Vec::new(),
                vis: crate::outline::item::Vis::Pub,
                macro_use: false,
                state: State::Declared(Decl::File {
                    candidates: vec![lib_path.clone()],
                    by_attr: true,
                }),
                textual: None,
            }],
            ..Crate::default()
        };
        Loader::new(0, &files, Others::Crates(&[]), &[], &mut tree)
            .load_module(0)
            .expect("load root");
        Loader::new(0, &files, Others::Crates(&[]), &[], &mut tree)
            .load_module(1)
            .expect("load string");

        assert_eq!(tree.link_libs, vec!["framework=CoreFoundation".to_owned()]);

        let outline = Outline {
            roots: vec![Root {
                unit: 0,
                name: "core_foundation_sys".to_owned(),
                path: lib_path,
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            }],
            crates: vec![tree],
            deps: vec![Vec::new()],
            cache: files,
            procs: HashMap::default(),
            pins: vec![None],
        };
        let mut names = Names {
            crates: vec![NamesCrate::default()],
            symbols: Symbols::default(),
        };
        names.crates[0].eager = true;
        let cx = Context::new(&outline, &names, Cache::default());
        assert_eq!(
            cx.crate_link_libs(0),
            &["framework=CoreFoundation".to_owned()]
        );

        let free_item = ItemRef {
            krate: 0,
            file: 1,
            item: 2,
        };
        assert_eq!(cx.link_lib(free_item), None);
        assert!(matches!(
            super::resolve_host_symbol(&cx, free_item, true),
            Ok((crate::host::HostSymbol::Dynamic { lib: None, .. }, 0))
        ));

        #[cfg(target_vendor = "apple")]
        {
            let cf_item = ItemRef {
                krate: 0,
                file: 1,
                item: 1,
            };
            assert_eq!(cx.link_lib(cf_item), None);
            assert!(matches!(
                super::resolve_host_symbol(&cx, cf_item, true),
                Ok((
                    crate::host::HostSymbol::Dynamic {
                        lib: Some(ref lib),
                        ..
                    },
                    0
                )) if lib.as_ref() == "framework=CoreFoundation"
            ));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
