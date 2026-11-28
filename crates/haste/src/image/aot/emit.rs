use std::sync::Arc;

use anyhow::{Context, Result, bail};
use haste_codegen::{RelocKind, Target};
use rustc_hash::FxHashMap as HashMap;

use crate::aot::{ExeSymbol, Executable, Symbol};
use crate::host::HostSymbol;
use crate::lower::{SymbolSource, VtableEntry, Word};
use crate::sync::lock;

use super::super::Image;
use super::{AotCode, AotFn, AotReach, AotState, AotStatic, icf, rodata, vdce};

const FN_ALIGN: u32 = 16;

fn place(exe: &mut Executable, code: &AotCode, debug_info: bool, fn_align: Option<u32>) -> u32 {
    let align = fn_align.unwrap_or(if code.loops().is_empty() { 4 } else { FN_ALIGN });
    exe.function(
        code.bytes(),
        align,
        code.loops(),
        if debug_info { code.lines() } else { &[] },
    )
}

struct SymTable<'a> {
    slot_offs: &'a HashMap<usize, u32>,
    rodata_offs: &'a [Option<u32>],
    vtable_offs: &'a [Option<u32>],
    static_syms: &'a HashMap<usize, Symbol>,
    promoted_syms: &'a HashMap<(usize, u32), Symbol>,
    thread_slots: &'a HashMap<usize, u64>,
}

impl SymTable<'_> {
    fn translate(&self, sym: &SymbolSource) -> Result<Symbol> {
        Ok(match sym {
            SymbolSource::Slot(index) => {
                Symbol::Rodata(*self.slot_offs.get(index).context("missing slot offset")?)
            }
            SymbolSource::Vtable(id) => Symbol::Rodata(
                self.vtable_offs
                    .get(id.0 as usize)
                    .copied()
                    .flatten()
                    .context("missing vtable offset")?,
            ),
            SymbolSource::Rodata(id) => Symbol::Rodata(
                self.rodata_offs
                    .get(id.0 as usize)
                    .copied()
                    .flatten()
                    .context("missing rodata offset")?,
            ),
            SymbolSource::StaticCell { index, .. } => self
                .static_syms
                .get(index)
                .cloned()
                .context("missing static cell")?,
            SymbolSource::PromotedCell { owner, seq, .. } => self
                .promoted_syms
                .get(&(*owner, *seq))
                .cloned()
                .context("missing promoted cell")?,
            SymbolSource::Host(host) => Symbol::Host(host.clone()),
            SymbolSource::Word(word) => Symbol::Word(*word),
            SymbolSource::ThreadSlot(index) => Symbol::Word(
                self.thread_slots
                    .get(index)
                    .copied()
                    .unwrap_or(*index as u64),
            ),
        })
    }
}

fn emit_dict_bodies(
    state: &AotState,
    order: &[usize],
    (debug_info, fn_align): (bool, Option<u32>),
    exe: &mut Executable,
) -> Result<HashMap<usize, u32>> {
    let mut dict_bodies: HashMap<usize, u32> = HashMap::default();
    for &idx in order {
        let Some(AotFn::Dict { body, .. }) = &state.functions[idx] else {
            continue;
        };
        let key = Arc::as_ptr(body) as usize;
        if dict_bodies.contains_key(&key) {
            continue;
        }
        let off = place(exe, body, debug_info, fn_align);
        for reloc in body.relocs() {
            let rt = reloc
                .as_runtime()
                .context("dict body contains non-runtime relocation")?;
            exe.patch(off + reloc.offset, reloc.kind, Symbol::Runtime(rt));
        }
        dict_bodies.insert(key, off);
    }
    Ok(dict_bodies)
}

fn slot_target(
    fn_offs: &[Option<u32>],
    foreign_fns: &HashMap<usize, HostSymbol>,
    idx: usize,
) -> Symbol {
    if let Some(off) = fn_offs.get(idx).copied().flatten() {
        Symbol::Text(off)
    } else {
        icf::foreign_sym(foreign_fns, idx)
    }
}

fn place_dict_fn(
    target: &Target,
    syms: &SymTable<'_>,
    dict_bodies: &HashMap<usize, u32>,
    (body, relocs): (&Arc<AotCode>, &[SymbolSource]),
    exe: &mut Executable,
) -> Result<u32> {
    let dict_off = exe.rodata(&vec![0u8; relocs.len() * 8], 8);
    for (i, reloc) in relocs.iter().enumerate() {
        exe.pointer(true, dict_off + (i * 8) as u32, syms.translate(reloc)?);
    }
    let stub = target.dict_entry_stub(0, 1usize << 40, 0, None)?;
    let [(dict_rel, dict_kind), (body_rel, body_kind)] = target.dict_stub_relocs()?;
    let off = exe.text(&stub);
    let body_off = dict_bodies[&(Arc::as_ptr(body) as usize)];
    exe.patch(off + dict_rel, dict_kind, Symbol::Rodata(dict_off));
    exe.patch(off + body_rel, body_kind, Symbol::Text(body_off));
    Ok(off)
}

fn patch_pending_relocs(
    state: &AotState,
    syms: &SymTable<'_>,
    fn_offs: &[Option<u32>],
    pending: Vec<(u32, &Arc<AotCode>, &[SymbolSource])>,
    exe: &mut Executable,
) -> Result<()> {
    for (off, code, relocs) in pending {
        for reloc in code.relocs() {
            let target_sym = if let Some(rt) = reloc.as_runtime() {
                Symbol::Runtime(rt)
            } else if matches!(reloc.kind, RelocKind::Arm64Call26 | RelocKind::PcRel32)
                && let SymbolSource::Slot(callee_idx) = relocs[reloc.symbol as usize]
            {
                slot_target(fn_offs, &state.foreign_fns, callee_idx)
            } else {
                syms.translate(&relocs[reloc.symbol as usize])?
            };
            exe.patch(off + reloc.offset, reloc.kind, target_sym);
        }
    }
    Ok(())
}

fn emit_functions(
    state: &AotState,
    live: &AotReach,
    (target, debug_info, fn_align): (&Target, bool, Option<u32>),
    (syms, dict_bodies): (&SymTable<'_>, &HashMap<usize, u32>),
    (roots, canon): (&[usize], &[usize]),
    exe: &mut Executable,
) -> Result<Vec<Option<u32>>> {
    let mut fn_offs = vec![None; state.functions.len()];
    let mut pending = Vec::new();
    for &idx in &live.order {
        let leader = canon[roots[idx]];
        if fn_offs[leader].is_none() {
            fn_offs[leader] = match &state.functions[leader] {
                Some(AotFn::Direct { code, relocs }) => {
                    let off = place(exe, code, debug_info, fn_align);
                    pending.push((off, code, relocs.as_slice()));
                    Some(off)
                }
                Some(AotFn::Dict { body, relocs }) => Some(place_dict_fn(
                    target,
                    syms,
                    dict_bodies,
                    (body, relocs),
                    exe,
                )?),
                Some(AotFn::Alias(_)) | None => None,
            };
        }
        fn_offs[idx] = fn_offs[leader];
    }
    patch_pending_relocs(state, syms, &fn_offs, pending, exe)?;
    Ok(fn_offs)
}

fn emit_static_syms(
    state: &AotState,
    live: &AotReach,
    exe: &mut Executable,
) -> (HashMap<usize, Symbol>, HashMap<(usize, u32), Symbol>) {
    let mut static_syms = HashMap::default();
    for &index in &live.statics.list {
        let Some(entry) = state.statics.get(&index) else {
            continue;
        };
        let sym = match entry {
            AotStatic::Local(size) => {
                Symbol::Bss(exe.bss(crate::image::static_cell_size(*size), 16))
            }
            AotStatic::Foreign {
                weak: false,
                host: Some(host),
            } => Symbol::Host(host.clone()),
            AotStatic::Foreign {
                weak: false,
                host: None,
            } => Symbol::Data(exe.data(&[0; 8], 8)),
            AotStatic::Foreign { weak: true, host } => {
                let off = exe.data(&[0; 8], 8);
                if let Some(host) = host {
                    exe.pointer(false, off, Symbol::Host(host.clone()));
                }
                Symbol::Data(off)
            }
        };
        static_syms.insert(index, sym);
    }
    let mut promoted_syms = HashMap::default();
    for &key in &live.promoted.list {
        let Some(&at) = state.promoted_seen.get(&key) else {
            continue;
        };
        promoted_syms.insert(key, Symbol::Bss(exe.bss(state.promoted[at].1, 16)));
    }
    (static_syms, promoted_syms)
}

fn patch_vtables(
    state: &AotState,
    vmask: &vdce::VtableMask,
    (vtable_offs, fn_offs): (&[Option<u32>], &[Option<u32>]),
    unique: &[(u32, usize)],
    exe: &mut Executable,
) -> Result<()> {
    for &(base, id) in unique {
        for (i, entry) in state.vtables[id].iter().enumerate() {
            if !vmask.slot_live(id, i) {
                exe.pointer(true, base + (i * 8) as u32, Symbol::Word(0));
                continue;
            }
            let target_sym = match *entry {
                VtableEntry::Word(Word::Int(val)) => Symbol::Word(val),
                VtableEntry::Word(Word::Code(fn_idx)) => {
                    slot_target(fn_offs, &state.foreign_fns, fn_idx)
                }
                VtableEntry::Super(super_id) => Symbol::Rodata(
                    vtable_offs
                        .get(super_id.0 as usize)
                        .copied()
                        .flatten()
                        .context("missing super vtable")?,
                ),
            };
            exe.pointer(true, base + (i * 8) as u32, target_sym);
        }
    }
    Ok(())
}

impl Image {
    /// Freezes a compiled AOT image into a standalone executable starting at `(lang_start, user_main)`.
    pub(crate) fn to_aot(
        &self,
        target: &Target,
        (lang_start, user_main): (usize, usize),
        cx: &mut crate::sema::Context<'_>,
        reached: &crate::sema::reach::Reached,
    ) -> Result<Executable> {
        let mut state = lock(self.aot().context("image has no AOT recording")?);
        repair_foreign_host_libs(cx, reached, &mut state);
        let (debug_info, symbols, fn_align) =
            crate::with_cfg(|c| (c.debug_info, c.symbols, c.fn_align));
        let mut aot_roots = vec![lang_start, user_main];
        let mut pre_exports: HashMap<usize, String> = HashMap::default();
        for (idx, (instance, _)) in reached.list.iter().enumerate() {
            if state.functions.get(idx).is_some_and(Option::is_some)
                && let Some(name) = export_name(cx, *instance)
            {
                pre_exports.insert(idx, name);
                if idx != lang_start && idx != user_main {
                    aot_roots.push(idx);
                }
            }
        }
        let panic_abort = user_panic_abort(cx, reached, user_main);
        let (live, vmask) = vdce::VtableMask::converge(cx, reached, &state.vtables, |vmask| {
            let live = AotReach::collect(&state, &aot_roots, vmask, !panic_abort);
            let counts = (live.order.len(), live.vtable_order.len());
            let fns = live.fns.clone();
            (live, fns, counts)
        });
        let mut exe = Executable::new(target);
        exe.panic_abort = panic_abort;
        if target.format == haste_codegen::Format::MachO {
            collect_macho_frameworks(cx, reached, &live, &mut exe);
        }
        let thread_slots: HashMap<usize, u64> = live
            .thread_slots
            .list
            .iter()
            .enumerate()
            .map(|(rank, &index)| (index, (rank + 1) as u64))
            .collect();
        exe.thread_slots = live.thread_slots.list.len() as u32 + 1;
        let (rodata_offs, rodata_fns) = rodata::emit_rodata(&state, &live, &mut exe)?;
        let (static_syms, promoted_syms) = emit_static_syms(&state, &live, &mut exe);
        let pre = icf::PreSyms {
            rodata_offs: &rodata_offs,
            static_syms: &static_syms,
            promoted_syms: &promoted_syms,
            thread_slots: &thread_slots,
        };
        let roots = icf::alias_roots(&state.functions);
        let canon = icf::fold_functions(&state, &live, &vmask, &pre, &roots)?;
        let (slot_offs, unique_slots) =
            icf::alloc_slots(&state, &live.slots.list, (&roots, &canon), &mut exe);
        let (vtable_offs, unique_vtables) = icf::alloc_vtables(
            &state,
            &live.vtable_order,
            &vmask,
            (&roots, &canon),
            &mut exe,
        )?;
        let syms = SymTable {
            slot_offs: &slot_offs,
            rodata_offs: &rodata_offs,
            vtable_offs: &vtable_offs,
            static_syms: &static_syms,
            promoted_syms: &promoted_syms,
            thread_slots: &thread_slots,
        };
        let dict_bodies = emit_dict_bodies(&state, &live.order, (debug_info, fn_align), &mut exe)?;
        exe.text.reserve(live.order.len().saturating_mul(64));
        exe.relocs.reserve(live.order.len().saturating_mul(4));
        exe.fn_starts.reserve(live.order.len());
        let fn_offs = emit_functions(
            &state,
            &live,
            (target, debug_info, fn_align),
            (&syms, &dict_bodies),
            (&roots, &canon),
            &mut exe,
        )?;
        if !exe.panic_abort {
            rodata::emit_unwind_table(
                &state,
                &live,
                (&roots, &canon),
                (&fn_offs, &dict_bodies),
                &mut exe,
            );
        }
        patch_vtables(
            &state,
            &vmask,
            (&vtable_offs, &fn_offs),
            &unique_vtables,
            &mut exe,
        )?;
        for (slot_off, target_idx) in unique_slots.into_iter().chain(rodata_fns) {
            exe.pointer(
                true,
                slot_off,
                slot_target(&fn_offs, &state.foreign_fns, target_idx),
            );
        }
        let Some(start_off) = fn_offs.get(lang_start).copied().flatten() else {
            bail!("`std::rt::lang_start` failed to compile");
        };
        let Some(main_off) = fn_offs.get(user_main).copied().flatten() else {
            bail!("`main` failed to compile");
        };
        exe.entry(start_off, main_off);
        if debug_info {
            exe.sources(
                cx.outline
                    .crates
                    .iter()
                    .flat_map(|krate| krate.files.iter().map(|file| file.source.path.clone()))
                    .collect(),
            );
        }
        let syms = collect_exe_symbols(&live, &fn_offs, symbols, &pre_exports, cx, reached);
        if !syms.is_empty() {
            exe.symbols(syms);
        }
        Ok(exe)
    }
}

fn export_name(
    cx: &crate::sema::Context<'_>,
    (item, substs): crate::sema::reach::Instance,
) -> Option<String> {
    if cx
        .types
        .substs(substs)
        .first()
        .is_some_and(|&ty| matches!(cx.types.kind(ty), crate::sema::ty::Kind::Closure(..)))
    {
        return None;
    }
    let (src, entry) = cx.item(item);
    if entry.attrs.is_empty() {
        return None;
    }
    let env = &cx.outline.crates[item.krate as usize].env;
    if let Some(name) = env.find_attr(&entry.attrs, crate::outline::tree::Truth::Yes, &|attr| {
        if let crate::outline::item::Attr::Other(at) = *attr
            && src.text(at) == "export_name"
            && src.kind(at + 1) == rustc_lexer::TokenKind::Eq
        {
            return src
                .literal_text(at + 2)
                .ok()
                .flatten()
                .filter(|s| !s.is_empty());
        }
        None
    }) {
        return Some(name);
    }
    (cx.has_attr(item, "no_mangle") && entry.name.is_some()).then(|| cx.plain_name(item).to_owned())
}

fn collect_exe_symbols(
    live: &AotReach,
    fn_offs: &[Option<u32>],
    include_locals: bool,
    pre_exports: &HashMap<usize, String>,
    cx: &mut crate::sema::Context<'_>,
    reached: &crate::sema::reach::Reached,
) -> Vec<ExeSymbol> {
    let mut out = Vec::new();
    let mut seen_exports = rustc_hash::FxHashSet::default();
    let mut local_at_off: HashMap<u32, usize> = HashMap::default();
    for &index in &live.order {
        let Some(off) = fn_offs[index] else {
            continue;
        };
        if let Some(name) = pre_exports.get(&index) {
            if seen_exports.insert(name.clone()) {
                out.push(ExeSymbol {
                    offset: off,
                    name: name.clone(),
                    exported: true,
                });
            }
            continue;
        }
        if include_locals {
            local_at_off.entry(off).or_insert(index);
        }
    }
    if include_locals {
        for (off, index) in local_at_off {
            out.push(ExeSymbol {
                offset: off,
                name: reached.list.get(index).map_or_else(
                    || format!("fn{index}"),
                    |(instance, _)| cx.symbol_text(*instance),
                ),
                exported: false,
            });
        }
    }
    out
}

fn rustflags_panic_abort() -> Option<bool> {
    let flags: Vec<String> = if let Some(raw) = crate::process_env_var("CARGO_ENCODED_RUSTFLAGS") {
        if raw.is_empty() {
            Vec::new()
        } else {
            raw.split('\x1f').map(str::to_owned).collect()
        }
    } else if let Some(raw) = crate::process_env_var("RUSTFLAGS") {
        raw.split_whitespace().map(str::to_owned).collect()
    } else {
        Vec::new()
    };
    let mut abort = None;
    let mut expect_c_arg = false;
    for flag in &flags {
        let opt = if expect_c_arg {
            expect_c_arg = false;
            Some(flag.as_str())
        } else if flag == "-C" || flag == "--codegen" {
            expect_c_arg = true;
            None
        } else {
            flag.strip_prefix("-C")
                .or_else(|| flag.strip_prefix("--codegen="))
        };
        if let Some(opt) = opt {
            match opt {
                "panic=abort" => abort = Some(true),
                "panic=unwind" => abort = Some(false),
                _ => {}
            }
        }
    }
    abort
}

fn user_panic_abort(
    cx: &crate::sema::Context<'_>,
    reached: &crate::sema::reach::Reached,
    user_main: usize,
) -> bool {
    if let Some(abort) = rustflags_panic_abort() {
        return abort;
    }
    let release =
        crate::config::Config::current().profile == crate::outline::tree::Profile::Release;
    let main_krate = reached
        .list
        .get(user_main)
        .map(|((item, _), _)| item.krate as usize);
    let manifest_dir = main_krate
        .into_iter()
        .chain(0..cx.outline.crates.len())
        .filter(|&k| !cx.outline.crates[k].env.fixed)
        .find_map(|k| {
            cx.outline.crates[k]
                .env
                .vars
                .iter()
                .find(|(key, _)| key == "CARGO_MANIFEST_DIR")
                .map(|(_, val)| std::path::Path::new(val.as_str()))
        });
    manifest_dir.is_some_and(|dir| {
        crate::graph::manifest::profile_panic_for(dir, release).as_deref() == Some("abort")
    })
}

fn crate_fallback_lib(
    cx: &crate::sema::Context<'_>,
    item: crate::names::ItemRef,
    name: &str,
) -> Option<String> {
    if let Some(explicit) = cx.link_lib(item) {
        return Some(explicit);
    }
    let libs = cx.crate_link_libs(item.krate);
    if libs.is_empty() {
        return None;
    }
    #[cfg(target_vendor = "apple")]
    if crate::image::in_apple_libsystem(name) {
        return None;
    }
    #[cfg(not(target_vendor = "apple"))]
    if HostSymbol::resolve_for_lib(name, None, false).is_some() {
        return None;
    }
    libs.iter()
        .find(|lib| HostSymbol::resolve_for_lib(name, Some(lib), false).is_some())
        .cloned()
        .or_else(|| libs.first().cloned())
}

fn repair_foreign_host_libs(
    cx: &crate::sema::Context<'_>,
    reached: &crate::sema::reach::Reached,
    state: &mut AotState,
) {
    for (&idx, sym) in &mut state.foreign_fns {
        if let HostSymbol::Dynamic { name, lib } = sym
            && lib.is_none()
            && let Some(((item, _), _)) = reached.list.get(idx)
            && let Some(found) = crate_fallback_lib(cx, *item, name)
        {
            *lib = Some(found.into_boxed_str());
        }
    }
    for (&idx, entry) in &mut state.statics {
        if let AotStatic::Foreign {
            host: Some(HostSymbol::Dynamic { name, lib }),
            ..
        } = entry
            && lib.is_none()
            && let Some(((item, _), _)) = reached.list.get(idx)
            && let Some(found) = crate_fallback_lib(cx, *item, name)
        {
            *lib = Some(found.into_boxed_str());
        }
    }
}

fn push_macho_framework_path(exe: &mut Executable, raw: &str) {
    if let Some(path) = crate::aot::Plan::macho_framework_path(raw)
        && !exe.extra_libs.contains(&path)
    {
        exe.extra_libs.push(path);
    }
}

fn collect_macho_frameworks(
    cx: &crate::sema::Context<'_>,
    reached: &crate::sema::reach::Reached,
    live: &AotReach,
    exe: &mut Executable,
) {
    let mut seen_crates = vec![false; cx.outline.crates.len()];
    let mut queue = Vec::new();
    for &idx in &live.order {
        if let Some(((item, _), _)) = reached.list.get(idx) {
            let k = item.krate as usize;
            if let Some(slot) = seen_crates.get_mut(k)
                && !*slot
            {
                *slot = true;
                queue.push(k);
            }
        }
    }
    while let Some(k) = queue.pop() {
        if let Some(script) = &cx.outline.crates[k].env.script {
            for (kind, name) in script.link_libs() {
                if kind == crate::graph::script::LinkKind::Framework {
                    let fw = name.trim().trim_end_matches(".framework");
                    if !fw.is_empty() {
                        push_macho_framework_path(exe, &format!("framework={fw}"));
                    }
                }
            }
        }
        for raw in cx.crate_link_libs(k as u32) {
            push_macho_framework_path(exe, raw);
        }
        for &(_, dep) in &cx.outline.deps[k] {
            if let Some(slot) = seen_crates.get_mut(dep)
                && !*slot
            {
                *slot = true;
                queue.push(dep);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
    use crate::outline::tree::{Crate, File, Files};
    use crate::outline::{Outline, Role, Root};
    use crate::sema::reach::Reached;
    use crate::sema::substs::Substs;
    use crate::sema::{Cache, Context};

    #[test]
    fn collect_macho_frameworks_and_repair_foreign_libs_use_crate_link_libs() {
        let src = crate::outline::source::Source::parse(
            std::path::PathBuf::from("lib.rs"),
            "extern \"C\" { fn CFTimeZoneCopySystem(); fn free(); }".to_owned(),
        )
        .expect("parse source");
        let items = crate::outline::scan::scan(&src).expect("scan source");
        let tree = Crate {
            files: vec![std::sync::Arc::new(File::new(src, items, false))],
            link_libs: vec!["framework=CoreFoundation".to_owned()],
            ..Crate::default()
        };
        let outline = Outline {
            roots: vec![Root {
                unit: 0,
                name: "core_foundation_sys".to_owned(),
                path: std::path::PathBuf::from("lib.rs"),
                role: Role::Lib,
                edition: "2021".to_owned(),
                proc_macro: false,
            }],
            crates: vec![tree],
            deps: vec![Vec::new()],
            cache: Files::default(),
            procs: HashMap::default(),
            pins: vec![None],
        };
        let names = Names {
            crates: vec![NamesCrate::default()],
            symbols: Symbols::default(),
        };
        let cx = Context::new(&outline, &names, Cache::default());
        let mut reached = Reached::default();
        let cf_item = ItemRef {
            krate: 0,
            file: 0,
            item: 0,
        };
        let free_item = ItemRef {
            krate: 0,
            file: 0,
            item: 1,
        };
        reached
            .list
            .push(((cf_item, Substs::EMPTY), Err(String::new())));
        reached
            .list
            .push(((free_item, Substs::EMPTY), Err(String::new())));

        let mut state = AotState::new(2).into_inner().expect("aot state");
        state.foreign_fns.insert(
            0,
            HostSymbol::Dynamic {
                name: "CFTimeZoneCopySystem".into(),
                lib: None,
            },
        );
        state.foreign_fns.insert(
            1,
            HostSymbol::Dynamic {
                name: "free".into(),
                lib: None,
            },
        );
        repair_foreign_host_libs(&cx, &reached, &mut state);

        assert_eq!(
            state.foreign_fns.get(&0),
            Some(&HostSymbol::Dynamic {
                name: "CFTimeZoneCopySystem".into(),
                lib: Some("framework=CoreFoundation".into()),
            })
        );
        #[cfg(target_vendor = "apple")]
        assert_eq!(
            state.foreign_fns.get(&1),
            Some(&HostSymbol::Dynamic {
                name: "free".into(),
                lib: None,
            })
        );

        let mut live = AotReach::new(2, &[0]);
        live.order.push(0);
        let mut exe = Executable::new(haste_codegen::Target::host());
        collect_macho_frameworks(&cx, &reached, &live, &mut exe);
        assert_eq!(
            exe.extra_libs,
            vec!["/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"]
        );
    }
}
