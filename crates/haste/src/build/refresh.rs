use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::macros::Compilation;
use crate::names::{Binding, Def, ItemRef, ModRef, Names, PerNs, Sym};
use crate::outline::Outline;
use crate::outline::edit::{
    attrs_tokens_eq, files_identical, has_critical_mark, is_global_kind, item_tokens_diff,
};
use crate::outline::item::{ItemKind, Vis};
use crate::outline::tree::File;
use crate::sema::reach::ReloadScope;

pub(super) use super::watch::{refresh_edits, update_stamps};

struct PreMod {
    file: usize,
    item: Option<u32>,
    parent: Option<usize>,
    vis: Vis,
    macro_use: bool,
    parts: Vec<usize>,
}

struct PreCrate {
    krate: u32,
    files: Vec<Arc<File>>,
    consumed: HashSet<(u32, u32)>,
    modules: Vec<PreMod>,
    impls: Vec<(usize, u32, usize, Vec<usize>)>,
    traits: Vec<(usize, u32, usize, Vec<usize>)>,
    macro_use: Vec<usize>,
    errors: Vec<String>,
    exports_sig: Vec<(String, usize, usize, String)>,
    prelude: Option<ModRef>,
    externs: Vec<(Sym, Def)>,
    scopes: Vec<crate::names::Scope>,
}

pub(super) struct PreReload(Vec<PreCrate>);

fn crate_exports_sig(tree: &crate::outline::tree::Crate) -> Vec<(String, usize, usize, String)> {
    let mut sig: Vec<_> = tree
        .exports
        .iter()
        .map(|(name, def)| {
            let expanded = if def.file.expanded {
                def.file.source.text.as_str().to_owned()
            } else {
                String::new()
            };
            (name.clone(), def.krate, def.rules.len(), expanded)
        })
        .collect();
    sig.sort();
    sig
}

fn snapshot_crate(outline: &Outline, names: &mut Names, krate: u32) -> PreCrate {
    let tree = &outline.crates[krate as usize];
    let ncrate = &mut names.crates[krate as usize];
    PreCrate {
        krate,
        files: tree.files.clone(),
        consumed: tree.consumed.clone(),
        modules: tree
            .modules
            .iter()
            .map(|m| PreMod {
                file: m.file,
                item: m.item,
                parent: m.parent,
                vis: m.vis,
                macro_use: m.macro_use,
                parts: m.parts.clone(),
            })
            .collect(),
        impls: tree
            .impls
            .iter()
            .map(|i| (i.file, i.at, i.module, i.parts.clone()))
            .collect(),
        traits: tree
            .traits
            .iter()
            .map(|t| (t.file, t.at, t.module, t.parts.clone()))
            .collect(),
        macro_use: tree.macro_use.clone(),
        errors: tree.errors.clone(),
        exports_sig: crate_exports_sig(tree),
        prelude: ncrate.prelude,
        externs: ncrate.externs.clone(),
        scopes: std::mem::take(&mut ncrate.scopes),
    }
}

pub(super) fn all_dependents(outline: &Outline, crates: &[u32]) -> Vec<u32> {
    let mut seen = vec![false; outline.deps.len()];
    let mut out: Vec<u32> = crate::outline::dependents_of(
        &outline.users(),
        &mut seen,
        crates.iter().map(|&k| k as usize),
    )
    .into_iter()
    .map(|at| at as u32)
    .collect();
    out.sort_unstable();
    out.dedup();
    out
}

pub(super) fn snapshot_edited(
    outline: &Outline,
    names: &mut Names,
    edited: &[u32],
) -> (Vec<u32>, PreReload) {
    let mut crates = edited.to_vec();
    crates.sort_unstable();
    crates.dedup();
    let entries = crates
        .iter()
        .map(|&k| snapshot_crate(outline, names, k))
        .collect();
    (crates, PreReload(entries))
}

fn downstream_names_to_reset(
    pre: &PreReload,
    edited: &[u32],
    outline: &Outline,
    names: &Names,
) -> Vec<u32> {
    let mut full_cascade = false;
    let mut added_syms: Vec<&str> = Vec::new();
    for (pos, &krate) in edited.iter().enumerate() {
        let old = &pre.0[pos];
        let new_tree = &outline.crates[krate as usize];
        let new_nc = &names.crates[krate as usize];
        if old.prelude != new_nc.prelude
            || old.externs != new_nc.externs
            || old.modules.len() != new_tree.modules.len()
            || old.scopes.len() != new_nc.scopes.len()
        {
            full_cascade = true;
            break;
        }
        for (old_scope, new_scope) in old.scopes.iter().zip(&new_nc.scopes) {
            if old_scope.names.iter().any(|(sym, &old_per)| {
                !new_scope
                    .names
                    .get(sym)
                    .is_some_and(|&new_per| per_ns_eq(old_per, new_per))
            }) {
                full_cascade = true;
                break;
            }
            for (&sym, &new_per) in &new_scope.names {
                if !old_scope.names.contains_key(&sym) {
                    if binding_is_trait_or_macro(outline, new_per) {
                        full_cascade = true;
                        break;
                    }
                    added_syms.push(names.symbols.text(sym));
                }
            }
            if full_cascade {
                break;
            }
        }
        if full_cascade {
            break;
        }
    }
    let deps = all_dependents(outline, edited);
    if full_cascade {
        return deps.into_iter().filter(|k| !edited.contains(k)).collect();
    }
    if added_syms.is_empty() {
        return Vec::new();
    }
    added_syms.sort_unstable();
    added_syms.dedup();
    let matched: Vec<u32> = deps
        .into_iter()
        .filter(|&k| {
            !edited.contains(&k)
                && !outline.is_fixed(k)
                && outline.crates[k as usize].files.iter().any(|f| {
                    added_syms
                        .iter()
                        .any(|sym| f.source.text.as_str().contains(sym))
                })
        })
        .collect();
    if matched.is_empty() {
        return Vec::new();
    }
    all_dependents(outline, &matched)
        .into_iter()
        .filter(|k| !edited.contains(k))
        .collect()
}

pub(super) fn resolve_with_dependents(
    compilation: &mut Compilation,
    (edited, mut pre): (Vec<u32>, PreReload),
    reloaded: Vec<usize>,
    changed_paths: &[PathBuf],
) -> Result<Option<(Vec<u32>, Vec<u32>, Option<ReloadScope>)>> {
    let mut crates: Vec<u32> = reloaded.into_iter().map(|at| at as u32).collect();
    crates.sort_unstable();
    crates.dedup();
    crates
        .iter()
        .for_each(|&k| compilation.names.reset(k as usize));
    crate::macros::resolve(compilation)?;
    if crates != edited {
        let deps = all_dependents(&compilation.outline, &crates);
        return Ok(Some((crates, deps, None)));
    }
    let exports_changed = crates.iter().enumerate().any(|(pos, &k)| {
        crate_exports_sig(&compilation.outline.crates[k as usize]) != pre.0[pos].exports_sig
    });
    if exports_changed {
        let all = compilation
            .outline
            .reload_dependents(&edited, changed_paths)?;
        let mut all_crates: Vec<u32> = all.into_iter().map(|at| at as u32).collect();
        all_crates.sort_unstable();
        all_crates.dedup();
        for &k in &all_crates {
            if !edited.contains(&k) {
                compilation.names.reset(k as usize);
            }
        }
        crate::macros::resolve(compilation)?;
        let deps = all_dependents(&compilation.outline, &all_crates);
        return Ok(Some((all_crates, deps, None)));
    }
    let extra = downstream_names_to_reset(&pre, &crates, &compilation.outline, &compilation.names);
    if !extra.is_empty() {
        for &k in &extra {
            pre.0.push(snapshot_crate(
                &compilation.outline,
                &mut compilation.names,
                k,
            ));
        }
        crates.extend(extra.iter().copied());
        extra
            .iter()
            .for_each(|&k| compilation.names.reset(k as usize));
        crate::macros::resolve(compilation)?;
        pre.0.sort_by_key(|entry| entry.krate);
        crates.sort_unstable();
        crates.dedup();
    }
    let dependents = all_dependents(&compilation.outline, &crates);
    let scoped = diff_reload(
        pre,
        &crates,
        &dependents,
        &compilation.outline,
        &compilation.names,
    );
    Ok(Some((crates, dependents, scoped)))
}

fn binding_eq(a: Option<Binding>, b: Option<Binding>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x.def == y.def && x.vis == y.vis && x.origin == y.origin,
        _ => false,
    }
}

fn per_ns_eq(a: PerNs, b: PerNs) -> bool {
    binding_eq(a.types, b.types) && binding_eq(a.values, b.values) && binding_eq(a.macros, b.macros)
}

fn binding_is_trait_or_macro(outline: &Outline, per: PerNs) -> bool {
    [per.types, per.values, per.macros]
        .into_iter()
        .flatten()
        .any(|b| match b.def {
            Def::Item(item) if item.outer().is_none() => outline
                .crates
                .get(item.krate as usize)
                .and_then(|c| c.files.get(item.file as usize))
                .and_then(|f| f.items.items.get(item.item as usize))
                .is_some_and(|e| {
                    matches!(
                        e.kind,
                        ItemKind::Trait
                            | ItemKind::TraitAlias
                            | ItemKind::MacroRules
                            | ItemKind::Macro
                    )
                }),
            _ => false,
        })
}

fn file_consumed_eq(old: &HashSet<(u32, u32)>, new: &HashSet<(u32, u32)>, file: u32) -> bool {
    let old_cnt = old.iter().filter(|&&(f, _)| f == file).count();
    let new_cnt = new.iter().filter(|&&(f, _)| f == file).count();
    old_cnt == new_cnt
        && old
            .iter()
            .filter(|&&(f, _)| f == file)
            .all(|k| new.contains(k))
}

fn mods_match(old: &[PreMod], new: &[crate::outline::tree::Module]) -> bool {
    old.len() == new.len()
        && old.iter().zip(new).all(|(a, b)| {
            a.file == b.file
                && a.item == b.item
                && a.parent == b.parent
                && a.vis == b.vis
                && a.macro_use == b.macro_use
                && a.parts == b.parts
        })
}

pub(super) fn diff_reload(
    pre: PreReload,
    crates: &[u32],
    dependents: &[u32],
    outline: &Outline,
    names: &Names,
) -> Option<ReloadScope> {
    if pre.0.len() != crates.len() {
        return None;
    }
    let mut dirty_crates = Vec::new();
    let mut dirty_files = Vec::new();
    let mut edited_files: HashSet<(u32, u32)> = HashSet::default();
    let mut clean_items_in_edited: Vec<ItemRef> = Vec::new();
    let mut old_dirty_items = HashSet::default();
    let mut all_dirty_items = HashSet::default();
    let mut dirty_modules = HashSet::default();
    let mut trait_dirty_modules = Vec::new();
    let mut syms_by_mod: Vec<(ModRef, Vec<&str>)> = Vec::new();

    for (pos, &krate) in crates.iter().enumerate() {
        let old = &pre.0[pos];
        let new_tree = &outline.crates[krate as usize];
        let new_nc = &names.crates[krate as usize];
        if old.prelude != new_nc.prelude
            || old.externs != new_nc.externs
            || old.macro_use != new_tree.macro_use
            || !new_tree.errors.is_empty()
            || old.errors != new_tree.errors
            || old.files.len() != new_tree.files.len()
            || !mods_match(&old.modules, &new_tree.modules)
            || old.scopes.len() != new_nc.scopes.len()
            || old.impls.len() != new_tree.impls.len()
            || old.traits.len() != new_tree.traits.len()
        {
            return None;
        }
        let impls_eq = old
            .impls
            .iter()
            .zip(&new_tree.impls)
            .all(|(a, b)| a.0 == b.file && a.1 == b.at && a.2 == b.module && a.3 == b.parts);
        let traits_eq = old
            .traits
            .iter()
            .zip(&new_tree.traits)
            .all(|(a, b)| a.0 == b.file && a.1 == b.at && a.2 == b.module && a.3 == b.parts);
        if !impls_eq || !traits_eq {
            return None;
        }

        let mut crate_dirty = false;
        for (file_idx, (old_f, new_f)) in old.files.iter().zip(&new_tree.files).enumerate() {
            let f_u32 = file_idx as u32;
            if Arc::ptr_eq(old_f, new_f) || files_identical(old_f, new_f) {
                if !file_consumed_eq(&old.consumed, &new_tree.consumed, f_u32) {
                    crate_dirty = true;
                    for &(cf, ci) in old.consumed.iter().chain(&new_tree.consumed) {
                        if cf == f_u32
                            && old.consumed.contains(&(cf, ci))
                                != new_tree.consumed.contains(&(cf, ci))
                        {
                            let iref = ItemRef {
                                krate,
                                file: f_u32,
                                item: ci,
                            };
                            old_dirty_items.insert(iref);
                            all_dirty_items.insert(iref);
                        }
                    }
                }
                continue;
            }
            crate_dirty = true;
            if old_f.expanded
                || new_f.expanded
                || old_f.expand_origin != new_f.expand_origin
                || old_f.source.path != new_f.source.path
                || !attrs_tokens_eq(
                    &old_f.source,
                    &old_f.items.inner,
                    &new_f.source,
                    &new_f.items.inner,
                )
            {
                return None;
            }
            edited_files.insert((krate, f_u32));
            let old_items = &old_f.items.items;
            let new_items = &new_f.items.items;
            let min_len = old_items.len().min(new_items.len());
            for idx in 0..min_len {
                let (oi, ni) = (&old_items[idx], &new_items[idx]);
                let (mut sig_changed, body_changed) = item_tokens_diff(old_f, oi, new_f, ni);
                let consumed_diff = old.consumed.contains(&(f_u32, idx as u32))
                    != new_tree.consumed.contains(&(f_u32, idx as u32));
                sig_changed |= consumed_diff;
                let iref = ItemRef {
                    krate,
                    file: f_u32,
                    item: idx as u32,
                };
                if sig_changed {
                    if is_global_kind(oi.kind)
                        || is_global_kind(ni.kind)
                        || has_critical_mark(old_f, idx)
                        || has_critical_mark(new_f, idx)
                    {
                        return None;
                    }
                    if let Some(parent) = oi.parent {
                        let pk = old_items[parent as usize].kind;
                        if matches!(pk, ItemKind::Impl | ItemKind::Trait) {
                            return None;
                        }
                    }
                    old_dirty_items.insert(iref);
                    all_dirty_items.insert(iref);
                } else if body_changed {
                    all_dirty_items.insert(iref);
                } else {
                    clean_items_in_edited.push(iref);
                }
            }
            for (idx, oi) in old_items.iter().enumerate().skip(min_len) {
                if is_global_kind(oi.kind) || has_critical_mark(old_f, idx) {
                    return None;
                }
                let iref = ItemRef {
                    krate,
                    file: f_u32,
                    item: idx as u32,
                };
                old_dirty_items.insert(iref);
                all_dirty_items.insert(iref);
            }
            for (idx, ni) in new_items.iter().enumerate().skip(min_len) {
                if is_global_kind(ni.kind) || has_critical_mark(new_f, idx) {
                    return None;
                }
                all_dirty_items.insert(ItemRef {
                    krate,
                    file: f_u32,
                    item: idx as u32,
                });
            }
        }
        if crate_dirty {
            dirty_crates.push(krate);
        }
        for (mod_idx, (old_scope, new_scope)) in old.scopes.iter().zip(&new_nc.scopes).enumerate() {
            let mref = ModRef {
                krate,
                module: mod_idx as u32,
            };
            let mut mod_trait_dirty = false;
            let mut mod_syms = Vec::new();
            let mut any_scope_diff = false;
            for (&sym, &old_per) in &old_scope.names {
                let changed = !new_scope
                    .names
                    .get(&sym)
                    .is_some_and(|&new_per| per_ns_eq(old_per, new_per));
                if changed {
                    any_scope_diff = true;
                    if binding_is_trait_or_macro(outline, old_per)
                        || new_scope
                            .names
                            .get(&sym)
                            .is_some_and(|&np| binding_is_trait_or_macro(outline, np))
                    {
                        mod_trait_dirty = true;
                        break;
                    }
                    mod_syms.push(names.symbols.text(sym));
                }
            }
            if !mod_trait_dirty {
                for (&sym, &new_per) in &new_scope.names {
                    if !old_scope.names.contains_key(&sym) {
                        any_scope_diff = true;
                        if binding_is_trait_or_macro(outline, new_per) {
                            mod_trait_dirty = true;
                            break;
                        }
                        mod_syms.push(names.symbols.text(sym));
                    }
                }
            }
            if any_scope_diff {
                dirty_modules.insert(mref);
            }
            if mod_trait_dirty {
                trait_dirty_modules.push(mref);
            } else if !mod_syms.is_empty() {
                syms_by_mod.push((mref, mod_syms));
            }
        }
    }

    if !syms_by_mod.is_empty() {
        let mut syms_by_file: HashMap<(u32, u32), Vec<&str>> = HashMap::default();
        for (mref, syms) in syms_by_mod {
            let m = &outline.crates[mref.krate as usize].modules[mref.module as usize];
            if m.file != usize::MAX {
                syms_by_file
                    .entry((mref.krate, m.file as u32))
                    .or_default()
                    .extend(syms.iter().copied());
            }
            for &part in &m.parts {
                syms_by_file
                    .entry((mref.krate, part as u32))
                    .or_default()
                    .extend(syms.iter().copied());
            }
        }
        for ((krate, f_idx), mut syms) in syms_by_file {
            syms.sort_unstable();
            syms.dedup();
            let key = (krate, f_idx);
            if edited_files.contains(&key) {
                for &iref in &clean_items_in_edited {
                    if (iref.krate, iref.file) == key {
                        let file = &outline.crates[krate as usize].files[f_idx as usize];
                        let entry = &file.items.items[iref.item as usize];
                        if (entry.lead..entry.end).any(|at| syms.contains(&file.source.text(at))) {
                            old_dirty_items.insert(iref);
                            all_dirty_items.insert(iref);
                        }
                    }
                }
            } else {
                let text = outline.crates[krate as usize].files[f_idx as usize]
                    .source
                    .text
                    .as_str();
                if syms.iter().any(|sym| text.contains(sym)) {
                    dirty_files.push(key);
                }
            }
        }
    }

    for mref in trait_dirty_modules {
        let m = &outline.crates[mref.krate as usize].modules[mref.module as usize];
        if m.file != usize::MAX {
            dirty_files.push((mref.krate, m.file as u32));
        }
        for &part in &m.parts {
            dirty_files.push((mref.krate, part as u32));
        }
    }
    dirty_files.sort_unstable();
    dirty_files.dedup();
    for &(krate, f_idx) in &dirty_files {
        if !dirty_crates.contains(&krate) {
            dirty_crates.push(krate);
        }
        for (mod_idx, m) in outline.crates[krate as usize].modules.iter().enumerate() {
            if m.file == f_idx as usize || m.parts.contains(&(f_idx as usize)) {
                dirty_modules.insert(ModRef {
                    krate,
                    module: mod_idx as u32,
                });
            }
        }
    }
    dirty_crates.sort_unstable();

    Some(ReloadScope {
        crates: crates.to_vec(),
        dependents: dependents.to_vec(),
        dirty_crates,
        dirty_files,
        old_dirty_items,
        all_dirty_items,
        dirty_modules,
    })
}

#[cfg(test)]
mod tests;
