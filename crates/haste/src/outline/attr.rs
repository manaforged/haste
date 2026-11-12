use anyhow::{Context, Result};
use rustc_lexer::TokenKind;

use super::item::{Attr, mark};
use super::source::{Source, opens};
use super::tree::{Env, Truth};
use crate::graph::cfg;

/// The top-level comma-separated ranges of tokens in `from..to`, where the generic arguments of a `::<...>` turbofish nest like a group.
pub fn split_commas(src: &Source, from: u32, to: u32) -> Vec<(u32, u32)> {
    let mut parts = Vec::new();
    let (mut start, mut at, mut angles) = (from, from, 0u32);
    while at < to {
        match src.kind(at) {
            TokenKind::Comma if angles == 0 => {
                parts.push((start, at));
                start = at + 1;
            }
            TokenKind::Lt if angles > 0 || turbofish(src, at) => angles += 1,
            TokenKind::Gt if angles > 0 && !(src.kind(at - 1) == TokenKind::Minus && src.joint(at - 1)) => angles -= 1,
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

/// The first token of each word of every `#[repr(..)]` among `attrs`, written directly or through a `cfg_attr` that holds under `env`.
pub fn repr_words(src: &Source, attrs: &[Attr], env: &Env) -> Vec<u32> {
    let mut words = Vec::new();
    for attr in attrs {
        match attr {
            Attr::CfgAttr(cfg, inner) if env.eval(cfg) == Truth::Yes => words.extend(repr_words(src, inner, env)),
            &Attr::Other(at) if src.text(at) == "repr" && src.kind(at + 1) == TokenKind::OpenParen => words.extend(split_commas(src, at + 2, src.pair(at + 1)).into_iter().map(|(from, _)| from)),
            _ => {}
        }
    }
    words
}

fn turbofish(src: &Source, at: u32) -> bool {
    at >= 2 && src.kind(at - 1) == TokenKind::Colon && src.kind(at - 2) == TokenKind::Colon && src.joint(at - 2)
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
    let Some(&(first, last)) = parts.first() else {
        return Ok(Attr::Other(open));
    };
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

const FLAGS: [(&str, Attr); 4] = [("macro_use", Attr::MacroUse), ("no_std", Attr::NoStd), ("no_core", Attr::NoCore), ("no_implicit_prelude", Attr::NoImplicitPrelude)];

/// Whether the attribute tokens `from..to` start with `derive(...)` or `derive_const(...)`, bare or under a standard prelude path.
pub fn is_derive_at(src: &Source, from: u32, to: u32) -> bool {
    let end = path_end(src, from, to);
    if end >= to || src.kind(end) != TokenKind::OpenParen {
        return false;
    }
    let mut at = from;
    if end - at >= 2 && src.kind(at) == TokenKind::Colon && src.kind(at + 1) == TokenKind::Colon {
        at += 2;
    }
    match end - at {
        1 => matches!(src.text(at), "derive" | "derive_const"),
        10 => matches!(src.text(at), "core" | "std") && src.text(at + 3) == "prelude" && matches!(src.text(at + 6), "v1" | "rust_2015" | "rust_2018" | "rust_2021" | "rust_2024") && matches!(src.text(at + 9), "derive" | "derive_const"),
        _ => false,
    }
}

/// Parses the tokens `from..to` inside `#[...]`.
pub fn content(src: &Source, from: u32, to: u32) -> Result<Attr> {
    if from >= to {
        return Ok(Attr::Other(from));
    }
    let after = path_end(src, from, to);
    let next = src.kind(after);
    if is_derive_at(src, from, to) {
        return Ok(Attr::Derive(after));
    }
    let path = src.span(from, after.saturating_sub(1).max(from));
    Ok(match (path, next) {
        ("unsafe", TokenKind::OpenParen) => content(src, after + 1, src.pair(after))?,
        ("cfg", TokenKind::OpenParen) => Attr::Cfg(cfg_of(src, after)?),
        ("cfg_attr", TokenKind::OpenParen) => cfg_attr(src, after)?,
        ("path", TokenKind::Eq) if after + 1 < to => Attr::Path(unquote(src.text(after + 1))),
        ("macro_export", next) => Attr::MacroExport(next == TokenKind::OpenParen && joined(src, after + 1, src.pair(after)).contains("local_inner_macros")),
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

/// The bits of the `MARKED` attributes that `attrs` name, directly or inside a `cfg_attr` whatever its condition.
pub fn marks(src: &Source, attrs: &[Attr]) -> u8 {
    attrs.iter().fold(0, |bits, attr| {
        bits | match attr {
            Attr::Other(at) => mark(src.text(*at)),
            Attr::CfgAttr(_, inner) => marks(src, inner),
            _ => 0,
        }
    })
}
