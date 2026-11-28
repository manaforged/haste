use anyhow::{Context, Result};
use rustc_lexer::TokenKind;

use super::item::{Attr, mark};
use super::source::{Source, opens};
use super::tree::{Env, Truth};
use crate::graph::cfg;

/// The top-level comma-separated ranges of tokens in `from..to`, where the generic arguments of a `::<...>` turbofish and a qualified path `<...>` that starts a part nest like a group.
pub fn split_commas(src: &Source, from: u32, to: u32) -> Vec<(u32, u32)> {
    let mut parts = Vec::new();
    let (mut start, mut at, mut angles) = (from, from, 0u32);
    while at < to {
        match src.kind(at) {
            TokenKind::Comma if angles == 0 => {
                parts.push((start, at));
                start = at + 1;
            }
            TokenKind::Lt if angles > 0 || at == start || turbofish(src, at) => angles += 1,
            TokenKind::Gt
                if angles > 0 && !(src.kind(at - 1) == TokenKind::Minus && src.joint(at - 1)) =>
            {
                angles -= 1
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

/// Returns the library name declared by an active `#[link(name = "...")]` attribute (prefixing `framework=` for `kind = "framework"`, and ignoring `kind = "static"` or failing `cfg(...)` clauses).
pub fn link_name(src: &Source, attrs: &[Attr], env: &Env) -> Option<String> {
    for attr in attrs {
        match attr {
            Attr::CfgAttr(cfg, inner) if env.eval(cfg) == Truth::Yes => {
                if let Some(found) = link_name(src, inner, env) {
                    return Some(found);
                }
            }
            &Attr::Other(at) if attr_is_named(src, at, "link") => {
                let end = path_end(src, at, src.len());
                if src.kind(end) != TokenKind::OpenParen {
                    continue;
                }
                let close = src.pair(end);
                let mut name = None;
                let mut is_framework = false;
                let mut skip = false;
                for (from, to) in split_commas(src, end + 1, close) {
                    if from >= to {
                        continue;
                    }
                    let key = src.text(from);
                    if key == "cfg" && from + 1 < to && src.kind(from + 1) == TokenKind::OpenParen {
                        if cfg_of(src, from + 1)
                            .ok()
                            .is_none_or(|c| env.eval(&c) != Truth::Yes)
                        {
                            skip = true;
                            break;
                        }
                    } else if from + 2 < to && src.kind(from + 1) == TokenKind::Eq {
                        let val = src
                            .literal_text(from + 2)
                            .ok()
                            .flatten()
                            .unwrap_or_else(|| unquote(src.text(from + 2)));
                        match key {
                            "name" => {
                                let trimmed = val.trim();
                                if !trimmed.is_empty() {
                                    name = Some(trimmed.to_owned());
                                }
                            }
                            "kind" => match val.as_str() {
                                "dylib" | "raw-dylib" => {}
                                "framework" => is_framework = true,
                                _ => {
                                    skip = true;
                                    break;
                                }
                            },
                            _ => {}
                        }
                    }
                }
                if !skip && let Some(found) = name {
                    return Some(if is_framework {
                        format!("framework={found}")
                    } else {
                        found
                    });
                }
            }
            _ => {}
        }
    }
    None
}

/// Returns the `expected` string of `#[should_panic(expected = "...")]` or `#[should_panic = "..."]`, if present.
pub fn should_panic_expected(src: &Source, attrs: &[Attr], env: &Env) -> Option<String> {
    for attr in attrs {
        match attr {
            Attr::CfgAttr(cfg, inner) if env.eval(cfg) == Truth::Yes => {
                if let Some(found) = should_panic_expected(src, inner, env) {
                    return Some(found);
                }
            }
            &Attr::Other(at) if attr_is_named(src, at, "should_panic") => {
                let mut end = at;
                while matches!(
                    src.kind(end),
                    TokenKind::Ident | TokenKind::RawIdent | TokenKind::Colon
                ) {
                    end += 1;
                }
                if src.kind(end) == TokenKind::Eq {
                    return Some(unquote(src.text(end + 1)));
                }
                if src.kind(end) == TokenKind::OpenParen {
                    let close = src.pair(end);
                    for (from, to) in split_commas(src, end + 1, close) {
                        if from + 2 < to
                            && src.text(from) == "expected"
                            && src.kind(from + 1) == TokenKind::Eq
                        {
                            return Some(unquote(src.text(from + 2)));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `attrs` contains an attribute named `wanted` (bare or under a standard prelude path), directly or via an active `cfg_attr`.
pub fn has_attr(src: &Source, attrs: &[Attr], env: &Env, wanted: &str) -> bool {
    attrs.iter().any(|attr| match attr {
        Attr::CfgAttr(cfg, inner) if env.eval(cfg) == Truth::Yes => {
            has_attr(src, inner, env, wanted)
        }
        Attr::Test => wanted == "test",
        Attr::Bench => wanted == "bench",
        &Attr::Other(at) => attr_is_named(src, at, wanted),
        _ => false,
    })
}

fn attr_is_named(src: &Source, from: u32, wanted: &str) -> bool {
    let mut end = from;
    while matches!(
        src.kind(end),
        TokenKind::Ident | TokenKind::RawIdent | TokenKind::Colon
    ) {
        end += 1;
    }
    let mut at = from;
    if end - at >= 2 && src.kind(at) == TokenKind::Colon && src.kind(at + 1) == TokenKind::Colon {
        at += 2;
    }
    match end - at {
        1 => src.text(at) == wanted,
        10 => {
            matches!(src.text(at), "core" | "std")
                && src.text(at + 3) == "prelude"
                && matches!(
                    src.text(at + 6),
                    "v1" | "rust_2015" | "rust_2018" | "rust_2021" | "rust_2024"
                )
                && src.text(at + 9) == wanted
        }
        _ => false,
    }
}

/// The first token of each word of every `#[repr(..)]` among `attrs`, written directly or through a `cfg_attr` that holds under `env`.
pub fn repr_words(src: &Source, attrs: &[Attr], env: &Env) -> Vec<u32> {
    let mut words = Vec::new();
    for attr in attrs {
        match attr {
            Attr::CfgAttr(cfg, inner) if env.eval(cfg) == Truth::Yes => {
                words.extend(repr_words(src, inner, env))
            }
            &Attr::Other(at)
                if src.text(at) == "repr" && src.kind(at + 1) == TokenKind::OpenParen =>
            {
                words.extend(
                    split_commas(src, at + 2, src.pair(at + 1))
                        .into_iter()
                        .map(|(from, _)| from),
                )
            }
            _ => {}
        }
    }
    words
}

fn turbofish(src: &Source, at: u32) -> bool {
    at >= 2
        && src.kind(at - 1) == TokenKind::Colon
        && src.kind(at - 2) == TokenKind::Colon
        && src.joint(at - 2)
}

fn unquote(text: &str) -> String {
    crate::body::literal::string_bytes(text)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_else(|| text.to_owned())
}

fn joined(src: &Source, from: u32, to: u32) -> String {
    (from..to)
        .map(|at| src.text(at))
        .collect::<Vec<_>>()
        .join(" ")
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
    let predicate = cfg::parse(&text)
        .with_context(|| format!("{}: cfg_attr({text}, ..)", src.path.display()))?;
    let inner = parts[1..]
        .iter()
        .map(|&(from, to)| content(src, from, to))
        .collect::<Result<Vec<_>>>()?;
    Ok(Attr::CfgAttr(predicate, inner))
}

fn path_end(src: &Source, mut at: u32, to: u32) -> u32 {
    while at < to
        && matches!(
            src.kind(at),
            TokenKind::Ident | TokenKind::RawIdent | TokenKind::Colon
        )
    {
        at += 1;
    }
    at
}

const FLAGS: [(&str, Attr); 4] = [
    ("macro_use", Attr::MacroUse),
    ("no_std", Attr::NoStd),
    ("no_core", Attr::NoCore),
    ("no_implicit_prelude", Attr::NoImplicitPrelude),
];

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
        10 => {
            matches!(src.text(at), "core" | "std")
                && src.text(at + 3) == "prelude"
                && matches!(
                    src.text(at + 6),
                    "v1" | "rust_2015" | "rust_2018" | "rust_2021" | "rust_2024"
                )
                && matches!(src.text(at + 9), "derive" | "derive_const")
        }
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
        ("macro_export", next) => Attr::MacroExport(
            next == TokenKind::OpenParen
                && joined(src, after + 1, src.pair(after)).contains("local_inner_macros"),
        ),
        _ => FLAGS
            .iter()
            .find(|(name, _)| *name == path)
            .map_or(Attr::Other(from), |(_, attr)| attr.clone()),
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

/// The bits of the marked attributes that `attrs` name, directly or inside a `cfg_attr` whatever its condition.
pub fn marks(src: &Source, attrs: &[Attr]) -> u16 {
    let inline_bit = mark("inline");
    let mut bits = 0u16;
    let mut has_cfg = false;
    for attr in attrs {
        match attr {
            Attr::Other(at) => {
                let b = mark(src.text(*at));
                bits |= b;
                if b == inline_bit && src.kind(*at + 1) == TokenKind::OpenParen {
                    bits |= super::item::MARK_INLINE_SPEC;
                }
            }
            Attr::CfgAttr(_, inner) => {
                has_cfg = true;
                bits |= marks(src, inner);
            }
            _ => {}
        }
    }
    if bits != 0 && has_cfg {
        bits |= super::item::MARK_CFG_ATTR;
    }
    bits
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use haste_codegen::Target;

    use super::*;
    use crate::outline::scan::scan;

    fn parse_first_link(src_text: &str, triple: &str) -> Option<String> {
        let src = Source::parse(PathBuf::from("test.rs"), src_text.to_owned()).expect("lex");
        let items = scan(&src).expect("scan");
        let target = crate::graph::cfg::Target::of(Target::find(triple).expect("target"));
        let env = Env {
            target,
            features: std::collections::BTreeSet::new(),
            script: None,
            known: true,
            vars: Vec::new(),
            release: false,
            fixed: false,
            fixed_files: Vec::new(),
            test: false,
            profile_override: Default::default(),
            env_reads: Default::default(),
        };
        link_name(&src, &items.items[0].attrs, &env)
    }

    #[test]
    fn link_name_handles_raw_dylib_cfg_attr_cfg_clause_static_and_empty() {
        let win = "x86_64-pc-windows-msvc";
        let lin = "x86_64-unknown-linux-gnu";
        assert_eq!(
            parse_first_link(r#"#[link(name = "user32")] extern "system" {}"#, win).as_deref(),
            Some("user32")
        );
        assert_eq!(
            parse_first_link(
                r#"#[cfg_attr(target_os = "windows", link(name = "bcrypt", kind = "raw-dylib"))] extern "system" {}"#,
                win,
            )
            .as_deref(),
            Some("bcrypt")
        );
        assert_eq!(
            parse_first_link(
                r#"#[cfg_attr(target_os = "windows", link(name = "bcrypt", kind = "raw-dylib"))] extern "system" {}"#,
                lin,
            ),
            None
        );
        assert_eq!(
            parse_first_link(
                r#"#[link(name = "foo", kind = "static")] #[link(name = "advapi32")] extern "system" {}"#,
                win,
            )
            .as_deref(),
            Some("advapi32")
        );
        assert_eq!(
            parse_first_link(
                r#"#[link(name = "unix_only", cfg(unix))] #[link(name = "win_only", cfg(windows))] extern "C" {}"#,
                win,
            )
            .as_deref(),
            Some("win_only")
        );
        assert_eq!(
            parse_first_link(r#"#[link(name = "   ")] extern "C" {}"#, win),
            None
        );
        let mac = "aarch64-apple-darwin";
        assert_eq!(
            parse_first_link(
                r#"#[link(name = "CoreFoundation", kind = "framework")] extern "C" {}"#,
                mac,
            )
            .as_deref(),
            Some("framework=CoreFoundation")
        );
        assert_eq!(
            parse_first_link(
                r#"#[cfg_attr(target_os = "macos", link(kind = "framework", name = "Security"))] extern "C" {}"#,
                mac,
            )
            .as_deref(),
            Some("framework=Security")
        );
        assert_eq!(
            parse_first_link(
                r#"#[cfg_attr(target_os = "macos", link(kind = "framework", name = "Security"))] extern "C" {}"#,
                lin,
            ),
            None
        );
    }
}
