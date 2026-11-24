use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;

use crate::body::{self as syntax, Body, sig};
use crate::names::collect::proc_kind;
use crate::names::{ItemRef, ModRef, plain};
use crate::outline::item::{Item, ItemKind, Items, ProcKind};
use crate::outline::source::{Source, Tokens};
use crate::outline::tree::{Expansion, File};
use crate::outline::{derive, scan};

use super::Context;

mod builtins;
mod scope;
mod writer;

use builtins::Site;
use writer::{Run, Writer, call_path, defined_expand};

fn body_range(file: &File, item: u32) -> Option<(u32, u32)> {
    let entry = &file.items.items[item as usize];
    match (entry.kind, entry.name) {
        (ItemKind::Const | ItemKind::Static, Some(name)) => {
            Some((sig::initializer(&file.source, name, entry.end)?, entry.end))
        }
        _ => entry.body.map(|open| (open, file.source.pair(open) + 1)),
    }
}

impl<'a> Context<'a> {
    /// The tokens that the macro `item` writes in module `home` for the call whose group opens at `open` of `src`, expanded once per call and kept with `src`.
    pub fn macro_output(
        &mut self,
        home: ModRef,
        src: &'a Source,
        item: ItemRef,
        open: u32,
    ) -> Result<Option<&'a Source>> {
        let key = (home.krate, home.module, open);
        if let Some(found) = src.outputs.get(key)? {
            return Ok(Some(found));
        }
        if let Some(output) = defined_expand(self, item, src, open) {
            return src.outputs.keep(key, output?).map(Some);
        }
        let Some(builtin) = self.builtin(item) else {
            return Ok(None);
        };
        let writer = Writer::new(home, 0);
        let start = open
            .checked_sub(1)
            .and_then(|bang| call_path(src, bang, 0))
            .map_or(open, |(start, _)| start);
        let run = Run {
            src,
            from: 0,
            shift: 0,
            depth: 0,
            call: (src, start),
        };
        let mut out = Tokens::default();
        if !writer.builtin(
            self,
            run,
            Site {
                builtin,
                krate: item.krate,
                start,
                open,
            },
            &mut out,
        )? {
            return Ok(None);
        }
        src.outputs
            .keep(key, out.finish(src.path.clone())?)
            .map(Some)
    }

    pub(super) fn template_text(
        &mut self,
        home: ModRef,
        src: &Source,
        range: (u32, u32),
    ) -> Result<Option<String>> {
        Writer::new(home, 0).literal(
            self,
            Run {
                src,
                from: 0,
                shift: 0,
                depth: 0,
                call: (src, range.0),
            },
            range,
        )
    }

    pub(super) fn parse_signature(&mut self, item: ItemRef) -> Result<sig::Signature> {
        let calls = item
            .declaring_body()
            .and_then(|outer| self.expansion(outer))
            .map(|found| &found.calls);
        let config = self.body_config(item, calls);
        let (src, entry) = self.item(item);
        match (entry.kind, entry.name) {
            (ItemKind::Const | ItemKind::Static, Some(name)) => {
                sig::const_signature(src, name, entry.end, config)
            }
            _ => sig::signature(src, entry.start, entry.body.unwrap_or(entry.end), config),
        }
    }

    /// The expansion of the body of `item`, or of the body that declares it, once a pass read that body and found macro calls or items in it.
    pub fn expansion(&self, item: ItemRef) -> Option<&'a Expansion> {
        let item = item.body_owner();
        self.outline.crates[item.krate as usize].files[item.file as usize]
            .expansion(item.krate, item.item)
    }

    /// The source that the body of `item` is parsed from: its file, with the expansions of its macro calls when it has some.
    pub fn body_src(&self, item: ItemRef) -> &'a Source {
        self.expansion(item)
            .map_or(&self.file(item).source, |found| &found.file.source)
    }

    /// The traits that the body of `root` declares, at any depth.
    pub fn body_declared_traits(&self, root: ItemRef) -> Vec<ItemRef> {
        let items = self
            .expansion(root)
            .map_or(&[][..], |found| &found.file.items.items[..]);
        items
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind == ItemKind::Trait)
            .map(|(at, _)| root.inner(at as u32))
            .collect()
    }

    fn body_config<'c>(
        &self,
        item: ItemRef,
        calls: Option<&'c HashMap<u32, u32>>,
    ) -> syntax::Config<'c>
    where
        'a: 'c,
    {
        syntax::Config {
            calls,
            env: Some(&self.outline.crates[item.krate as usize].env),
        }
    }

    pub(super) fn parse_body(&mut self, item: ItemRef) -> Result<Body> {
        let file = self.file(item);
        if item.outer().is_some() || file.body_read(item.krate, item.item) {
            let found = self.expansion(item);
            return super::body_scope::parse_in(
                self.body_src(item),
                self.body_config(item, found.map(|found| &found.calls)),
                self.item(item).1,
            );
        }
        let Some((from, to)) = body_range(file, item.item) else {
            return super::body_scope::parse_in(
                &file.source,
                self.body_config(item, None),
                self.item(item).1,
            );
        };
        let home = self.module_of(item)?;
        let before = self.asks.count();
        let mut writer = Writer::new(home, file.source.len());
        writer.scan(
            self,
            Run {
                src: &file.source,
                from,
                shift: 0,
                depth: 0,
                call: (&file.source, from),
            },
            to,
        )?;
        if self.asks.count() > before {
            bail!(
                "the macros that {} calls are not resolved yet",
                self.name(item)
            );
        }
        let base = self.outline.crates[item.krate as usize].files[item.file as usize].clone();
        let source = std::mem::take(&mut writer.out).finish_over(base)?;
        let body = super::body_scope::parse_in(
            &source,
            self.body_config(item, Some(&writer.calls)),
            self.item(item).1,
        )?;
        let (items, mut scopes, item_scopes) = super::body_scope::declared(
            &source,
            self.body_config(item, Some(&writer.calls)),
            &body,
        )?;
        let (source, items, item_scopes) = self.body_derives(
            item,
            before,
            &mut writer,
            (source, items, &mut scopes, item_scopes),
        )?;
        let calls = writer.calls;
        let found = (!calls.is_empty() || !items.items.is_empty()).then(|| {
            Box::new(Expansion {
                file: File::new(source, items, true),
                calls,
                scopes: scopes.into(),
                item_scopes: item_scopes.into(),
            })
        });
        self.file(item).set_expansion(item.krate, item.item, found);
        Ok(body)
    }
}

impl Context<'_> {
    fn body_derives(
        &mut self,
        item: ItemRef,
        before: usize,
        writer: &mut Writer,
        (source, mut items, scopes, mut item_scopes): (
            Source,
            Items,
            &mut Vec<Option<u32>>,
            Vec<u32>,
        ),
    ) -> Result<(Source, Items, Vec<u32>)> {
        let derived = self.collect_derives(item.krate, writer, &source, &items, &item_scopes)?;
        if self.asks.count() > before {
            bail!(
                "the macros that {} calls are not resolved yet",
                self.name(item)
            );
        }
        let Some(base) = source.base.clone().filter(|_| !derived.is_empty()) else {
            return Ok((source, items, item_scopes));
        };
        writer.offset = source.offset;
        writer.out.copy_range(&source, source.offset, source.len());
        let (mut starts, mut chunks) = (Vec::new(), Vec::new());
        for (expanded, parent, scope) in derived {
            let chunk = writer.offset + writer.out.len();
            let mut at = 0;
            while at < expanded.len() {
                starts.push((chunk + at, parent, scope));
                at = scan::item_end(&expanded, at)?;
            }
            writer.out.copy_range(&expanded, 0, expanded.len());
            writer.out.space();
            chunks.push((chunk, expanded));
        }
        for (chunk, expanded) in &chunks {
            writer.scan(
                self,
                Run {
                    src: expanded,
                    from: 0,
                    shift: *chunk,
                    depth: 1,
                    call: (expanded, 0),
                },
                expanded.len(),
            )?;
        }
        if self.asks.count() > before {
            bail!(
                "the macros that {} calls are not resolved yet",
                self.name(item)
            );
        }
        let source = std::mem::take(&mut writer.out).finish_over(base)?;
        super::body_scope::scan_nested(
            &source,
            self.body_config(item, Some(&writer.calls)),
            starts,
            &mut items,
            scopes,
            &mut item_scopes,
        )?;
        Ok((source, items, item_scopes))
    }

    fn collect_derives(
        &mut self,
        krate: u32,
        writer: &Writer,
        source: &Source,
        items: &Items,
        item_scopes: &[u32],
    ) -> Result<Vec<(Source, Option<u32>, u32)>> {
        let mut derived = Vec::new();
        for (at, entry) in items.items.iter().enumerate() {
            if !matches!(
                entry.kind,
                ItemKind::Struct | ItemKind::Enum | ItemKind::Union
            ) {
                continue;
            }
            let (builtin, foreign) = {
                let (env, deps) = (
                    &self.outline.crates[krate as usize].env,
                    &self.outline.deps[krate as usize][..],
                );
                derive::derived(source, entry, env, (krate as usize, deps))
            };
            if let Some(expanded) = builtin {
                derived.push((expanded?, entry.parent, item_scopes[at]));
            }
            for path in foreign {
                let run = Run {
                    src: source,
                    from: 0,
                    shift: 0,
                    depth: 0,
                    call: (source, path.0),
                };
                let found = writer.find_macro(self, run, path, plain(source.text(path.1)))?;
                if let Some(expanded) = self.proc_derive(krate, found, source, entry)? {
                    derived.push((expanded, entry.parent, item_scopes[at]));
                }
            }
        }
        Ok(derived)
    }

    fn proc_derive(
        &mut self,
        krate: u32,
        found: Option<ItemRef>,
        source: &Source,
        entry: &Item,
    ) -> Result<Option<Source>> {
        let Some(run) = found.filter(|run| run.outer().is_none()) else {
            return Ok(None);
        };
        let (def_src, def_item) = self.item(run);
        let Some((_, ProcKind::Derive)) = proc_kind(def_src, def_item) else {
            return Ok(None);
        };
        let key = (
            run,
            crate::macros::derive_input(source, entry, &self.outline.crates[krate as usize].env),
        );
        match self.outline.procs.get(&key) {
            Some(Ok(text)) => Source::parse(source.path.clone(), text.clone()).map(Some),
            Some(Err(error)) => bail!("{}: {error}", self.name(run)),
            None => {
                self.asks.procs.borrow_mut().push(key);
                Ok(None)
            }
        }
    }
}

/// The items that the item statement from `start` up to `end` declares in `items`, with the functions of an extern block it is.
pub fn stmt_items(items: &[Item], (start, end): (u32, u32)) -> impl Iterator<Item = usize> + '_ {
    let in_block = |entry: &Item| {
        entry.parent.is_none_or(|parent| {
            matches!(
                items[parent as usize].kind,
                ItemKind::Fn | ItemKind::Const | ItemKind::Static
            )
        })
    };
    let top = items
        .iter()
        .position(|entry| in_block(entry) && entry.start >= start && entry.start < end);
    let block = top.filter(|&at| items[at].kind == ItemKind::ExternBlock);
    let children = items
        .iter()
        .enumerate()
        .filter(move |(_, entry)| block.is_some() && entry.parent == block.map(|at| at as u32))
        .map(|(at, _)| at);
    top.into_iter().chain(children)
}
