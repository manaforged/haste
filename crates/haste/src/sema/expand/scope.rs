use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::body::cursor::is_name;
use crate::names::collect::extern_crate_root;
use crate::names::resolve::macro_at;
use crate::names::{ItemRef, plain};
use crate::outline::rules::{self, Rule};
use crate::outline::source::{Source, Tokens};
use crate::sema::Context;

use super::writer::{Local, Run, Writer};

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
    let path = if end == from + 1 && plain(src.text(from)) == "self" && prefix.len() >= 2 {
        prefix[..prefix.len() - 2].to_vec()
    } else {
        joined(end)
    };
    found.push((path, alias));
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
                && (end == at + 1 || (end >= at + 3 && src.kind(end - 1) == TokenKind::Colon))
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

    pub(super) fn note_extern(
        &mut self,
        cx: &Context<'_>,
        src: &Source,
        at: u32,
        macro_use: bool,
    ) -> Option<u32> {
        let orig = plain(src.text(at + 2));
        let (alias, end) = match (src.text(at + 3), src.kind(at + 3)) {
            (_, TokenKind::Semi) => (orig, at + 4),
            ("as", _) if is_name(src.kind(at + 4)) && src.kind(at + 5) == TokenKind::Semi => {
                (plain(src.text(at + 4)), at + 6)
            }
            _ => return None,
        };
        if let Some(root) = extern_crate_root(
            orig,
            self.home.krate,
            &cx.outline.deps[self.home.krate as usize],
        ) {
            if macro_use && !self.macro_uses.contains(&root) {
                self.macro_uses.push(root);
            }
            if alias != "_" {
                self.externs.push((alias.to_owned(), root));
            }
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
            if let Some(found) = self.resolve_joined(cx, prefix, src, path, 0)? {
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
        depth: u8,
    ) -> Result<Option<ItemRef>> {
        let mut full = Tokens::default();
        full.copy_range(prefix, 0, prefix.len());
        (from..=to).for_each(|token| full.copy(src, token));
        let joined = full.finish(src.path.clone())?;
        self.resolve_use_source(cx, &joined, depth)
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
            for (_, prefix) in self.uses.iter().rev().filter(|(alias, _)| alias == first) {
                if let Some(found) = self.resolve_joined(cx, prefix, src, (1, last), depth + 1)? {
                    return Ok(Some(found));
                }
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
        if path.0 == path.1 {
            for (_, source) in self.uses.iter().rev().filter(|(alias, _)| *alias == name) {
                if let Some(found) = self.resolve_use_source(cx, source, 0)? {
                    return Ok(Some(found));
                }
            }
        } else {
            let first = plain(run.src.text(path.0));
            for (_, source) in self.uses.iter().rev().filter(|(alias, _)| *alias == first) {
                if let Some(found) =
                    self.resolve_joined(cx, source, run.src, (path.0 + 1, path.1), 0)?
                {
                    return Ok(Some(found));
                }
            }
        }
        match self
            .resolve_macro(cx, run.src, path)
            .or_else(|| self.def_site_macro(cx, run.src, path))
        {
            Some(found) => Ok(Some(found)),
            None => {
                if let Some(found) = self.resolve_glob(cx, run.src, path)? {
                    return Ok(Some(found));
                }
                for &root in self.macro_uses.iter().rev() {
                    if let Some(found) = macro_at(&cx.paths(root.krate), run.src, path, root) {
                        return Ok(Some(found));
                    }
                }
                Ok(None)
            }
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

    fn expand_two_crates(
        tag: &str,
        dep_src: &str,
        app_src: &str,
    ) -> std::sync::Arc<crate::body::Body> {
        expand_two_crates_at(tag, dep_src, app_src, 0)
    }

    fn expand_two_crates_at(
        tag: &str,
        dep_src: &str,
        app_src: &str,
        item: u32,
    ) -> std::sync::Arc<crate::body::Body> {
        let dir = std::env::temp_dir().join(format!("haste_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        let dep_path = dir.join("dep.rs");
        let app_path = dir.join("app.rs");
        std::fs::write(&dep_path, dep_src).expect("write dep.rs");
        std::fs::write(&app_path, app_src).expect("write app.rs");

        let app_deps = vec![("dep".to_owned(), 0usize)];
        let mut outline = Outline {
            roots: vec![
                Root {
                    unit: 0,
                    name: "dep".to_owned(),
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
        let dep_sym = syms.intern("dep");
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
            item,
        };
        let body = cx.body(fn_item).expect("expand body");
        let _ = std::fs::remove_dir_all(&dir);
        body
    }

    fn assert_all_macros_expanded(body: &crate::body::Body, msg: &str) {
        assert!(
            body.exprs
                .iter()
                .all(|expr| !matches!(expr, crate::body::Expr::Macro { .. })),
            "{msg}"
        );
    }

    #[test]
    fn grouped_block_use_with_glob_resolves_exported_and_glob_macros() {
        let body = expand_two_crates(
            "grouped_use",
            concat!(
                "#[macro_export] macro_rules! bugreport { () => { 40u32 }; }\n",
                "#[macro_export] macro_rules! col_m { () => { 2u32 }; }\n",
                "pub mod collector { pub use crate::col_m; }\n",
                "pub mod format { pub struct Markdown; }\n",
            ),
            concat!(
                "fn invoke_bugreport() -> u32 {\n",
                "    use dep::{bugreport, collector::*, format::Markdown};\n",
                "    bugreport!() + col_m!()\n",
                "}\n",
            ),
        );
        assert_all_macros_expanded(
            &body,
            "macros imported via grouped use with wildcard sibling must expand",
        );
    }

    #[test]
    fn local_non_macro_use_does_not_shadow_macro() {
        let body = expand_two_crates(
            "non_macro_use_shadow",
            concat!(
                "pub mod env {}\n",
                "#[macro_export] macro_rules! env { () => { 40u32 }; }\n",
                "#[macro_export] macro_rules! inner_m { () => { 2u32 }; }\n",
                "pub struct sub;\n",
                "pub mod submod { pub use crate::inner_m as m; }\n",
            ),
            concat!(
                "fn run() -> u32 {\n",
                "    use dep::*;\n",
                "    use dep::env;\n",
                "    use dep::submod as s;\n",
                "    use dep::sub as s;\n",
                "    env!() + s::m!()\n",
                "}\n",
            ),
        );
        assert_all_macros_expanded(
            &body,
            "local non-macro `use` must not shadow macro resolution",
        );
    }

    #[test]
    fn def_site_multi_segment_macro_expands_in_caller() {
        let body = expand_two_crates(
            "def_site_multi_seg",
            concat!(
                "pub mod inner { pub macro m() { 42u32 } }\n",
                "pub macro outer() { inner::m!() }\n",
            ),
            concat!("fn run() -> u32 {\n", "    dep::outer!()\n", "}\n",),
        );
        assert_all_macros_expanded(
            &body,
            "multi-segment def-site path inside `pub macro` must expand",
        );
    }

    #[test]
    fn grouped_braced_use_and_self_alias_expand_macros() {
        let body = expand_two_crates(
            "grouped_braced_self",
            concat!(
                "#[macro_export] macro_rules! top { () => { 2u32 }; }\n",
                "#[macro_export] macro_rules! leaf { () => { 40u32 }; }\n",
                "pub mod sub { pub use crate::leaf as m; }\n",
            ),
            concat!(
                "fn run() -> u32 {\n",
                "    use {dep::top, dep::sub::{self as s}};\n",
                "    top!() + s::m!()\n",
                "}\n",
            ),
        );
        assert_all_macros_expanded(
            &body,
            "braced root `use` and `{self as alias}` must expand macros",
        );
    }

    #[test]
    fn local_inner_macros_argument_macro_expands_without_caller_import() {
        let body = expand_two_crates(
            "local_inner_arg_macro",
            concat!(
                "#[macro_export]\n",
                "macro_rules! relative_eq { ($l:expr, $r:expr) => { $l == $r }; }\n",
                "#[macro_export]\n",
                "macro_rules! __assert_approx { ($eq:ident, $l:expr, $r:expr) => { if !($eq!($l, $r)) {} }; }\n",
                "#[macro_export(local_inner_macros)]\n",
                "macro_rules! assert_relative_eq { ($l:expr, $r:expr) => { __assert_approx!(relative_eq, $l, $r) }; }\n",
            ),
            concat!(
                "fn run() {\n",
                "    use dep::assert_relative_eq;\n",
                "    assert_relative_eq!(1u32, 1u32);\n",
                "}\n",
            ),
        );
        assert_all_macros_expanded(
            &body,
            "macro ident emitted by #[macro_export(local_inner_macros)] must expand",
        );
    }

    #[test]
    fn macro_use_extern_crate_with_cfg_attr_and_local_extern_expands() {
        let dep_src =
            "#[macro_export] macro_rules! relative_eq { ($l:expr, $r:expr) => { $l == $r }; }\n";
        let root_body = expand_two_crates_at(
            "macro_use_cfg_attr",
            dep_src,
            concat!(
                "#[cfg_attr(all(), macro_use)]\n",
                "pub extern crate dep;\n",
                "fn run() -> bool {\n",
                "    relative_eq!(1u32, 1u32)\n",
                "}\n",
            ),
            1,
        );
        assert_all_macros_expanded(
            &root_body,
            "#[cfg_attr(..., macro_use)] pub extern crate must import macros",
        );

        let local_body = expand_two_crates(
            "macro_use_local_extern",
            dep_src,
            concat!(
                "fn run() -> bool {\n",
                "    #[macro_use]\n",
                "    extern crate dep;\n",
                "    relative_eq!(1u32, 1u32)\n",
                "}\n",
            ),
        );
        assert_all_macros_expanded(
            &local_body,
            "function-local #[macro_use] extern crate must import macros",
        );
    }
}
