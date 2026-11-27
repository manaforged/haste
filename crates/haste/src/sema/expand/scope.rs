use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::body::cursor::is_name;
use crate::names::collect::extern_crate_root;
use crate::names::resolve::macro_at;
use crate::names::{ItemRef, plain};
use crate::outline::rules::{self, Rule};
use crate::outline::source::{Source, Tokens};
use crate::sema::Context;

use super::writer::{Local, Run, Writer, def_site_macro};

fn use_entries(
    src: &Source,
    prefix: &[u32],
    (from, to): (u32, u32),
    found: &mut Vec<(Vec<u32>, u32)>,
    globs: &mut Vec<Vec<u32>>,
) -> bool {
    let mut at = from;
    while at < to {
        let mut end = at;
        while end < to && src.kind(end) != TokenKind::Comma {
            end = if src.kind(end) == TokenKind::OpenBrace {
                src.pair(end) + 1
            } else {
                end + 1
            };
        }
        if end > at && !use_entry(src, prefix, (at, end), found, globs) {
            return false;
        }
        at = end + 1;
    }
    true
}

fn is_use_path_token(src: &Source, at: u32) -> bool {
    is_name(src.kind(at))
        || src.kind(at) == TokenKind::Colon
        || (src.kind(at) == TokenKind::Dollar && src.dollar(at).is_some())
}

fn use_entry(
    src: &Source,
    prefix: &[u32],
    (from, to): (u32, u32),
    found: &mut Vec<(Vec<u32>, u32)>,
    globs: &mut Vec<Vec<u32>>,
) -> bool {
    let pathlike = |range: std::ops::Range<u32>| range.clone().all(|at| is_use_path_token(src, at));
    let joined = |end: u32| {
        prefix
            .iter()
            .copied()
            .chain(from..end)
            .collect::<Vec<u32>>()
    };
    if src.kind(to - 1) == TokenKind::CloseBrace {
        let open = src.pair(to - 1);
        return pathlike(from..open)
            && use_entries(src, &joined(open), (open + 1, to - 1), found, globs);
    }
    if src.text(to - 1) == "*" {
        if to == from + 1 {
            globs.push(prefix.to_vec());
            return true;
        }
        if to > from + 3
            && src.kind(to - 2) == TokenKind::Colon
            && src.kind(to - 3) == TokenKind::Colon
            && is_name(src.kind(to - 4))
            && pathlike(from..to - 3)
        {
            globs.push(joined(to - 1));
            return true;
        }
        return false;
    }
    let (end, alias) = if to >= from + 3 && src.text(to - 2) == "as" {
        (to - 2, to - 1)
    } else {
        (to, to - 1)
    };
    if !pathlike(from..end) || !is_name(src.kind(end - 1)) || !is_name(src.kind(alias)) {
        return false;
    }
    found.push((joined(end), alias));
    true
}

impl Writer {
    pub(super) fn note_use(&mut self, src: &Source, at: u32) -> Result<Option<u32>> {
        let mut end = at + 1;
        while src.kind(end) != TokenKind::Semi {
            if src.text(end) == "*"
                && src.kind(end + 1) == TokenKind::Semi
                && end >= at + 4
                && src.kind(end - 1) == TokenKind::Colon
                && src.kind(end - 2) == TokenKind::Colon
            {
                let mut copy = Tokens::default();
                (at + 1..end).for_each(|token| copy.copy(src, token));
                self.globs.push(copy.finish(src.path.clone())?);
                return Ok(Some(end + 2));
            }
            if src.kind(end) == TokenKind::OpenBrace
                && end >= at + 3
                && src.kind(end - 1) == TokenKind::Colon
            {
                return self.note_use_group(src, (at + 1, end));
            }
            if !is_use_path_token(src, end) {
                return Ok(None);
            }
            end += 1;
        }
        let (last, alias) = if end >= at + 4 && src.text(end - 2) == "as" {
            (end - 3, end - 1)
        } else {
            (end - 1, end - 1)
        };
        if last <= at || !is_name(src.kind(last)) {
            return Ok(None);
        }
        let mut copy = Tokens::default();
        (at + 1..=last).for_each(|token| copy.copy(src, token));
        self.uses.push((
            plain(src.text(alias)).to_owned(),
            copy.finish(src.path.clone())?,
        ));
        Ok(Some(end + 1))
    }

    fn note_use_group(&mut self, src: &Source, (from, open): (u32, u32)) -> Result<Option<u32>> {
        let close = src.pair(open);
        if src.kind(close + 1) != TokenKind::Semi {
            return Ok(None);
        }
        let mut found = Vec::new();
        let mut globs = Vec::new();
        if !use_entries(
            src,
            &(from..open).collect::<Vec<_>>(),
            (open + 1, close),
            &mut found,
            &mut globs,
        ) {
            return Ok(None);
        }
        for path in globs {
            let mut copy = Tokens::default();
            path.into_iter().for_each(|token| copy.copy(src, token));
            self.globs.push(copy.finish(src.path.clone())?);
        }
        for (path, alias) in found {
            let mut copy = Tokens::default();
            path.into_iter().for_each(|token| copy.copy(src, token));
            self.uses.push((
                plain(src.text(alias)).to_owned(),
                copy.finish(src.path.clone())?,
            ));
        }
        Ok(Some(close + 2))
    }

    pub(super) fn define(&mut self, src: &Source, name: u32) -> Result<()> {
        self.local(src, (name, src.pair(name + 1)), rules::parse)
    }

    pub(super) fn define_macro(&mut self, src: &Source, name: u32) -> Result<u32> {
        let open = name + 1;
        let last = if src.kind(open) == TokenKind::OpenParen {
            src.pair(src.pair(open) + 1)
        } else {
            src.pair(open)
        };
        self.local(src, (name, last), rules::parse_macro)?;
        Ok(last + 1)
    }

    fn local(
        &mut self,
        src: &Source,
        (name, last): (u32, u32),
        parse: fn(&Source, u32) -> Result<Vec<Rule>>,
    ) -> Result<()> {
        let mut copy = Tokens::default();
        (name + 1..=last).for_each(|at| copy.copy(src, at));
        let source = copy.finish(src.path.clone())?;
        let rules = parse(&source, 0)?;
        self.locals.push(Local {
            name: plain(src.text(name)).to_owned(),
            source,
            rules,
        });
        Ok(())
    }

    pub(super) fn note_extern(&mut self, cx: &Context<'_>, src: &Source, at: u32) -> Option<u32> {
        let orig = plain(src.text(at + 2));
        let (alias, end) = match (src.text(at + 3), src.kind(at + 3)) {
            (_, TokenKind::Semi) => (orig, at + 4),
            ("as", _) if is_name(src.kind(at + 4)) && src.kind(at + 5) == TokenKind::Semi => {
                (plain(src.text(at + 4)), at + 6)
            }
            _ => return None,
        };
        if alias != "_"
            && let Some(root) = extern_crate_root(
                orig,
                self.home.krate,
                &cx.outline.deps[self.home.krate as usize],
            )
        {
            self.externs.push((alias.to_owned(), root));
        }
        Some(end)
    }

    fn resolve_macro(&self, cx: &Context<'_>, src: &Source, path: (u32, u32)) -> Option<ItemRef> {
        if path.0 + 3 <= path.1
            && is_name(src.kind(path.0))
            && src.kind(path.0 + 1) == TokenKind::Colon
        {
            let first = plain(src.text(path.0));
            if let Some(&(_, root)) = self.externs.iter().rev().find(|(alias, _)| alias == first) {
                return macro_at(&cx.paths(root.krate), src, (path.0 + 3, path.1), root);
            }
        }
        macro_at(&cx.paths(self.home.krate), src, path, self.home)
    }

    fn resolve_glob(
        &self,
        cx: &Context<'_>,
        src: &Source,
        path: (u32, u32),
    ) -> Result<Option<ItemRef>> {
        for prefix in self.globs.iter().rev() {
            if let Some(found) = self.resolve_joined(cx, prefix, src, path)? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    fn resolve_joined(
        &self,
        cx: &Context<'_>,
        prefix: &Source,
        src: &Source,
        (from, to): (u32, u32),
    ) -> Result<Option<ItemRef>> {
        let mut full = Tokens::default();
        full.copy_range(prefix, 0, prefix.len());
        (from..=to).for_each(|token| full.copy(src, token));
        let joined = full.finish(src.path.clone())?;
        Ok(self.resolve_macro(cx, &joined, (0, joined.len() - 1)))
    }

    fn resolve_use_source(
        &self,
        cx: &Context<'_>,
        src: &Source,
        depth: u8,
    ) -> Result<Option<ItemRef>> {
        let last = src.len() - 1;
        if let Some(found) = self.resolve_macro(cx, src, (0, last)) {
            return Ok(Some(found));
        }
        if depth < 8
            && last > 0
            && is_name(src.kind(0))
            && src.kind(1) == TokenKind::Colon
            && src.dollar(0).is_none()
        {
            let first = plain(src.text(0));
            if let Some((_, prefix)) = self.uses.iter().rev().find(|(alias, _)| alias == first) {
                let mut full = Tokens::default();
                full.copy_range(prefix, 0, prefix.len());
                (1..=last).for_each(|token| full.copy(src, token));
                let joined = full.finish(src.path.clone())?;
                return self.resolve_use_source(cx, &joined, depth + 1);
            }
        }
        Ok(None)
    }

    pub(super) fn find_macro(
        &self,
        cx: &mut Context<'_>,
        run: Run<'_>,
        path: (u32, u32),
        name: &str,
    ) -> Result<Option<ItemRef>> {
        if let Some((_, source)) = self
            .uses
            .iter()
            .rev()
            .find(|(alias, _)| path.0 == path.1 && *alias == name)
        {
            return self.resolve_use_source(cx, source, 0);
        }
        let first = plain(run.src.text(path.0));
        if path.0 < path.1
            && let Some((_, source)) = self.uses.iter().rev().find(|(alias, _)| *alias == first)
            && let Some(found) = self.resolve_joined(cx, source, run.src, (path.0 + 1, path.1))?
        {
            return Ok(Some(found));
        }
        match self
            .resolve_macro(cx, run.src, path)
            .or_else(|| def_site_macro(cx, run.src, path))
        {
            Some(found) => Ok(Some(found)),
            None => self.resolve_glob(cx, run.src, path),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rustc_hash::FxHashMap as HashMap;

    use crate::names::{Crate as NamesCrate, ItemRef, Names, Symbols};
    use crate::outline::tree::{Crate, Decl, Files, Module, State};
    use crate::outline::{Outline, Role, Root};
    use crate::sema::{Cache, Context};

    fn declared_crate(path: PathBuf) -> Crate {
        Crate {
            modules: vec![Module {
                file: usize::MAX,
                item: None,
                parent: None,
                name: String::new(),
                dir: PathBuf::new(),
                parts: Vec::new(),
                vis: crate::outline::item::Vis::Pub,
                macro_use: false,
                state: State::Declared(Decl::File {
                    candidates: vec![path],
                    by_attr: true,
                }),
                textual: None,
            }],
            ..Crate::default()
        }
    }

    #[test]
    fn grouped_block_use_with_glob_resolves_exported_and_glob_macros() {
        let dir = std::env::temp_dir().join(format!("haste_grouped_use_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        let dep_path = dir.join("bugreport.rs");
        let app_path = dir.join("app.rs");
        std::fs::write(
            &dep_path,
            concat!(
                "#[macro_export] macro_rules! bugreport { () => { 40u32 }; }\n",
                "#[macro_export] macro_rules! col_m { () => { 2u32 }; }\n",
                "pub mod collector { pub use crate::col_m; }\n",
                "pub mod format { pub struct Markdown; }\n",
            ),
        )
        .expect("write bugreport.rs");
        std::fs::write(
            &app_path,
            concat!(
                "fn invoke_bugreport() -> u32 {\n",
                "    use bugreport::{bugreport, collector::*, format::Markdown};\n",
                "    bugreport!() + col_m!()\n",
                "}\n",
            ),
        )
        .expect("write app.rs");

        let app_deps = vec![("bugreport".to_owned(), 0usize)];
        let mut outline = Outline {
            roots: vec![
                Root {
                    unit: 0,
                    name: "bugreport".to_owned(),
                    path: dep_path.clone(),
                    role: Role::Lib,
                    edition: "2021".to_owned(),
                    proc_macro: false,
                },
                Root {
                    unit: 1,
                    name: "app".to_owned(),
                    path: app_path.clone(),
                    role: Role::Bin,
                    edition: "2021".to_owned(),
                    proc_macro: false,
                },
            ],
            crates: vec![declared_crate(dep_path), declared_crate(app_path)],
            deps: vec![Vec::new(), app_deps],
            cache: Files::default(),
            procs: HashMap::default(),
            pins: vec![None, None],
        };
        outline.load_all().expect("load outline");

        let syms = Symbols::default();
        let dep_sym = syms.intern("bugreport");
        let mut dep_names = NamesCrate::default();
        dep_names.edition = "2021".to_owned();
        dep_names.eager = true;
        let mut app_names = NamesCrate::default();
        app_names.edition = "2021".to_owned();
        app_names.deps = vec![(dep_sym, 0)];
        app_names.eager = true;

        let mut names = Names {
            crates: vec![dep_names, app_names],
            symbols: syms,
        };
        names.resolve(&mut outline).expect("resolve names");

        let mut cx = Context::new(&outline, &names, Cache::default());
        let fn_item = ItemRef {
            krate: 1,
            file: 0,
            item: 0,
        };
        let body = cx.body(fn_item).expect("expand body with grouped use");
        assert!(
            body.exprs
                .iter()
                .all(|expr| !matches!(expr, crate::body::Expr::Macro { .. })),
            "macros imported via grouped use with wildcard sibling must expand"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
