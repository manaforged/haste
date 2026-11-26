use anyhow::Result;

use super::build::{self, Loader, Others};
use super::item::ItemKind;
use super::mbe;
use super::source::Source;
use super::tree::{Crate, Files};

pub(super) fn resolve_macro_includes(
    krate: usize,
    cache: &Files,
    others: Others<'_>,
    deps: &[(String, usize)],
    holder: &mut Crate,
    module: usize,
) -> Result<()> {
    const BAD_INCLUDE: &str = "include!: a path that is not a literal, `concat!` or `env!`";
    if !holder.errors.iter().any(|err| err == BAD_INCLUDE) {
        return Ok(());
    }
    let mut done = rustc_hash::FxHashSet::default();
    loop {
        let entry = &holder.modules[module];
        if !entry.loaded() {
            return Ok(());
        }
        let slices: Vec<(usize, Option<u32>)> = std::iter::once((entry.file, entry.item))
            .chain(entry.parts.iter().map(|&p| (p, None)))
            .collect();
        let mut resolved = Vec::new();
        for (fidx, owner) in slices {
            let file = holder.files[fidx].clone();
            for (at, item) in file.items.items.iter().enumerate() {
                if item.parent != owner
                    || item.kind != ItemKind::MacroCall
                    || !holder.env.enabled(&item.attrs)
                    || holder.consumed.contains(&(fidx as u32, at as u32))
                {
                    continue;
                }
                let (Some(name), Some(open)) = (item.name, item.body) else {
                    continue;
                };
                if crate::names::plain(file.source.text(name)) != "include"
                    || !done.insert((fidx, at))
                {
                    continue;
                }
                let range = (open + 1, file.source.pair(open));
                if build::path_text(&file.source, range, &holder.env).is_none()
                    && let Some(path) = macro_path_text(krate, holder, &file.source, range, 0)
                {
                    resolved.push((file.source.beside(&path), path));
                }
            }
        }
        if resolved.is_empty() {
            return Ok(());
        }
        for (full, _) in resolved {
            if let Some(pos) = holder.errors.iter().position(|err| err == BAD_INCLUDE) {
                holder.errors.remove(pos);
            }
            match Source::read(&full) {
                Ok(src) => Loader::new(krate, cache, others, deps, holder)
                    .insert(module, src, "include")?,
                Err(err) => holder.errors.push(format!("include!: {err:#}")),
            }
        }
    }
}

fn macro_path_text(
    krate: usize,
    holder: &Crate,
    src: &Source,
    (from, to): (u32, u32),
    depth: usize,
) -> Option<String> {
    if depth >= crate::RECURSION_LIMIT {
        return None;
    }
    if let Some(plain) = build::path_text(src, (from, to), &holder.env) {
        return Some(plain);
    }
    if src.text(from) == "(" && src.pair(from) == to - 1 {
        return macro_path_text(krate, holder, src, (from + 1, to - 1), depth + 1);
    }
    if to < from + 3 || src.text(from + 1) != "!" || src.pair(from + 2) != to - 1 {
        return None;
    }
    let name = crate::names::plain(src.text(from));
    if name == "concat" {
        return src
            .args(from + 2)
            .into_iter()
            .map(|r| macro_path_text(krate, holder, src, r, depth + 1))
            .collect();
    }
    let def = holder.exports.get(name).or_else(|| {
        holder.defined.iter().find_map(|(&(f, i), mac)| {
            let fsrc = &holder.files.get(f as usize)?.source;
            let ntok = holder.files[f as usize].items.items.get(i as usize)?.name?;
            (crate::names::plain(fsrc.text(ntok)) == name).then_some(mac)
        })
    })?;
    let expanded = mbe::expand(
        &def.file.source,
        &def.rules,
        src,
        (from + 2, krate as u32),
        None,
    )
    .ok()?;
    macro_path_text(krate, holder, &expanded, (0, expanded.len()), depth + 1)
}
