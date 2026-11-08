use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::body::literal::{char_value, string_bytes};
use crate::names::ItemRef;
use crate::outline::item::Attr;
use crate::outline::source::{Source, Tokens};

use super::Context;

/// A macro whose expansion the compiler writes instead of a definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    Concat,
    Stringify,
    Line,
    Column,
    File,
    Cfg,
    Assert,
    FormatArgs,
    Panic,
    Unreachable,
    PatternType,
}

const BUILTINS: [(&str, Builtin); 12] = [
    ("concat", Builtin::Concat),
    ("stringify", Builtin::Stringify),
    ("line", Builtin::Line),
    ("column", Builtin::Column),
    ("file", Builtin::File),
    ("cfg", Builtin::Cfg),
    ("assert", Builtin::Assert),
    ("format_args", Builtin::FormatArgs),
    ("const_format_args", Builtin::FormatArgs),
    ("panic", Builtin::Panic),
    ("unreachable", Builtin::Unreachable),
    ("pattern_type", Builtin::PatternType),
];

impl Builtin {
    /// The builtin that the `#[rustc_builtin_macro]` definition named `name` stands for.
    pub fn named(name: &str) -> Option<Builtin> {
        BUILTINS.iter().find(|(known, _)| *known == name).map(|(_, builtin)| *builtin)
    }
}

impl Context<'_> {
    /// The builtin that the macro `item` is, when its definition is `#[rustc_builtin_macro]`.
    pub fn builtin(&self, item: ItemRef) -> Option<Builtin> {
        let (src, entry) = self.item(item);
        let marked = entry.attrs.iter().any(|attr| matches!(attr, Attr::Other(at) if src.text(*at) == "rustc_builtin_macro"));
        Builtin::named(src.text(entry.name?)).filter(|_| marked)
    }
}

/// The token ranges of the comma-separated arguments in the group that opens at `open`.
pub fn args(src: &Source, open: u32) -> Vec<(u32, u32)> {
    let close = src.pair(open);
    let (mut found, mut from, mut at) = (Vec::new(), open + 1, open + 1);
    while at < close {
        match src.kind(at) {
            TokenKind::Comma => {
                found.push((from, at));
                from = at + 1;
                at += 1;
            }
            TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => at = src.pair(at) + 1,
            _ => at += 1,
        }
    }
    if from < close {
        found.push((from, close));
    }
    found
}

/// The text a literal token or `true` and `false` adds to `concat!`, or none for any other token.
pub fn literal_text(src: &Source, at: u32) -> Result<Option<String>> {
    let text = src.text(at);
    Ok(match src.kind(at) {
        TokenKind::Literal { kind: rustc_lexer::LiteralKind::Str { .. } | rustc_lexer::LiteralKind::RawStr { .. }, .. } => Some(String::from_utf8(string_bytes(text)?)?),
        TokenKind::Literal { kind: rustc_lexer::LiteralKind::Char { .. }, .. } => Some(char_value(text)?.to_string()),
        TokenKind::Literal { kind: rustc_lexer::LiteralKind::Int { .. } | rustc_lexer::LiteralKind::Float { .. }, suffix_start } => Some(text[..suffix_start as usize].to_owned()),
        TokenKind::Ident if matches!(text, "true" | "false") => Some(text.to_owned()),
        _ => None,
    })
}

/// The tokens from `from` up to `to` as `stringify!` spells them.
pub fn spelled(src: &Source, (from, to): (u32, u32)) -> String {
    let mut text = String::new();
    for at in from..to {
        text.push_str(src.text(at));
        if at + 1 < to && !src.joint(at) {
            text.push(' ');
        }
    }
    text
}

/// Writes `$crate` of crate `krate` and then `rest`.
pub fn crate_path(out: &mut Tokens, krate: u32, rest: &str) {
    out.dollar_crate(krate);
    out.lexed(rest);
}

/// The expansion of `assert!` with the arguments `args` of the call in `src`, for a macro of crate `krate`.
pub fn assert(src: &Source, args: &[(u32, u32)], krate: u32, out: &mut Tokens) {
    let Some(&(from, to)) = args.first() else { return };
    out.lexed("if ! (");
    (from..to).for_each(|at| out.copy(src, at));
    out.lexed(") {");
    match args.get(1) {
        None => {
            crate_path(out, krate, "::panicking::panic");
            out.lexed(&format!("({:?})", format!("assertion failed: {}", spelled(src, (from, to)))));
        }
        Some(&(rest, _)) => {
            crate_path(out, krate, "::panic::panic_2021!");
            out.lexed("(");
            (rest..args.last().map_or(rest, |last| last.1)).for_each(|at| out.copy(src, at));
            out.lexed(")");
        }
    }
    out.lexed("}");
}

/// The expansion of `format_args!` whose whole template is `template` with no placeholder, or none when it has one.
pub fn format_literal(template: &str, krate: u32, out: &mut Tokens) -> bool {
    let plain = template.replace("{{", "").replace("}}", "");
    if plain.contains(['{', '}']) {
        return false;
    }
    crate_path(out, krate, "::fmt::Arguments::from_str");
    out.lexed(&format!("({:?})", template.replace("{{", "{").replace("}}", "}")));
    true
}
