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
        if end > at && !use_entry(src, prefix, (at, end), found) {
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
        return pathlike(from..open) && use_entries(src, &joined(open), (open + 1, to - 1), found);
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
        if !use_entries(
            src,
            &(from..open).collect::<Vec<_>>(),
            (open + 1, close),
            &mut found,
        ) {
            return Ok(None);
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
