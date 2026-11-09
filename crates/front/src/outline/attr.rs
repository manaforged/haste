use anyhow::{Context, Result};
use rustc_lexer::TokenKind;

use super::item::Attr;
use super::source::{Source, opens};
use crate::graph::cfg;

/// The top-level comma-separated ranges of tokens in `from..to`.
pub fn split_commas(src: &Source, from: u32, to: u32) -> Vec<(u32, u32)> {
    let mut parts = Vec::new();
    let (mut start, mut at) = (from, from);
    while at < to {
        match src.kind(at) {
            TokenKind::Comma => {
                parts.push((start, at));
                start = at + 1;
            }
            kind if opens(kind) => at = src.pair(at),
            _ => {}
        }
        at += 1;
    }
    if start < to {
        parts.push((start, to));
    }
    parts
}

fn unquote(text: &str) -> String {
    text.trim_start_matches('r').trim_matches('#').trim_matches('"').to_owned()
}

fn joined(src: &Source, from: u32, to: u32) -> String {
    (from..to).map(|at| src.text(at)).collect::<Vec<_>>().join(" ")
}

fn cfg_of(src: &Source, open: u32) -> Result<cfg::Cfg> {
    let close = src.pair(open);
    if close == open + 1 {
        return Ok(cfg::Cfg::All(Vec::new()));
    }
    let text = joined(src, open + 1, close);
    cfg::parse(&text).with_context(|| format!("{}: cfg({text})", src.path.display()))
}

fn cfg_attr(src: &Source, open: u32) -> Result<Attr> {
    let parts = split_commas(src, open + 1, src.pair(open));
    let Some(&(first, last)) = parts.first() else { return Ok(Attr::Other(open)) };
    let text = joined(src, first, last);
    let predicate = cfg::parse(&text).with_context(|| format!("{}: cfg_attr({text}, ..)", src.path.display()))?;
    let inner = parts[1..].iter().map(|&(from, to)| content(src, from, to)).collect::<Result<Vec<_>>>()?;
    Ok(Attr::CfgAttr(predicate, inner))
}

fn path_end(src: &Source, mut at: u32, to: u32) -> u32 {
    while at < to && matches!(src.kind(at), TokenKind::Ident | TokenKind::RawIdent | TokenKind::Colon) {
        at += 1;
    }
    at
}

const FLAGS: [(&str, Attr); 5] = [
    ("macro_use", Attr::MacroUse),
    ("macro_export", Attr::MacroExport),
    ("no_std", Attr::NoStd),
    ("no_core", Attr::NoCore),
    ("no_implicit_prelude", Attr::NoImplicitPrelude),
];

/// Parses the tokens `from..to` inside `#[...]`.
pub fn content(src: &Source, from: u32, to: u32) -> Result<Attr> {
    if from >= to {
        return Ok(Attr::Other(from));
    }
    let after = path_end(src, from, to);
    let next = src.kind(after);
    let path = src.span(from, after.saturating_sub(1).max(from));
    Ok(match (path, next) {
        ("unsafe", TokenKind::OpenParen) => content(src, after + 1, src.pair(after))?,
        ("cfg", TokenKind::OpenParen) => Attr::Cfg(cfg_of(src, after)?),
        ("cfg_attr", TokenKind::OpenParen) => cfg_attr(src, after)?,
        ("path", TokenKind::Eq) if after + 1 < to => Attr::Path(unquote(src.text(after + 1))),
        ("derive" | "derive_const", TokenKind::OpenParen) => Attr::Derive(after),
        _ => FLAGS.iter().find(|(name, _)| *name == path).map_or(Attr::Other(from), |(_, attr)| attr.clone()),
    })
}

/// Reads the attributes starting at `at`, outer (`#[..]`) or inner (`#![..]`), and returns the token after them.
pub fn attrs(src: &Source, mut at: u32, inner: bool, into: &mut Vec<Attr>) -> Result<u32> {
    loop {
        let bang = src.kind(at + 1) == TokenKind::Bang;
        if src.kind(at) != TokenKind::Pound || bang != inner {
            return Ok(at);
        }
        let open = at + 1 + u32::from(bang);
        if src.kind(open) != TokenKind::OpenBracket {
            return Ok(at);
        }
        let close = src.pair(open);
        into.push(content(src, open + 1, close)?);
        at = close + 1;
    }
}
