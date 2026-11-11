use rustc_lexer::TokenKind;

use crate::body::{Fragment, fragment_end};

use super::rules::Frag;
use super::scan;
use super::source::{Source, opens};

fn is_name(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Ident | TokenKind::RawIdent)
}

const OPERATORS: [&str; 24] = [
    "<<=", ">>=", "...", "..=", "::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "^=", "&=", "|=", "<<", ">>", "..",
];

fn glued(src: &Source, at: u32, operator: &str) -> bool {
    operator.char_indices().all(|(index, ch)| {
        let token = at + index as u32;
        let last = index + 1 == operator.len();
        token < src.len() && src.text(token).len() == 1 && src.text(token).starts_with(ch) && (last || src.joint(token))
    })
}

/// The end of the token tree that starts at `at`.
pub(super) fn tree(src: &Source, at: u32) -> u32 {
    if opens(src.kind(at)) {
        return src.pair(at) + 1;
    }
    if !src.joint(at) {
        return at + 1;
    }
    let operator = OPERATORS.iter().find(|operator| glued(src, at, operator));
    at + operator.map_or(1, |operator| operator.len() as u32)
}

fn is_pair(src: &Source, at: u32, first: TokenKind, second: TokenKind) -> bool {
    src.kind(at) == first && src.kind(at + 1) == second && src.joint(at)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stop {
    Expr,
    Ty,
    Pat,
    PatParam,
}

fn is_punct(kind: TokenKind) -> bool {
    !matches!(kind, TokenKind::Ident | TokenKind::RawIdent | TokenKind::Literal { .. } | TokenKind::Lifetime { .. })
}

fn single_eq(src: &Source, at: u32) -> bool {
    let glued_before = at > 0 && src.joint(at - 1) && is_punct(src.kind(at - 1)) && !opens(src.kind(at - 1));
    let glued_after = src.joint(at) && matches!(src.kind(at + 1), TokenKind::Eq | TokenKind::Gt);
    src.kind(at) == TokenKind::Eq && !glued_before && !glued_after
}

fn ty_stops(src: &Source, at: u32, first: bool) -> bool {
    let kind = src.kind(at);
    let word = if is_name(kind) { src.text(at) } else { "" };
    let block = kind == TokenKind::OpenBrace || (!first && kind == TokenKind::OpenBracket && ends_type(src, at));
    single_eq(src, at) || kind == TokenKind::Or || single_colon(src, at) || matches!(word, "as" | "where") || block
}

fn pat_stops(src: &Source, at: u32, param: bool) -> bool {
    let kind = src.kind(at);
    let word = if is_name(kind) { src.text(at) } else { "" };
    single_eq(src, at) || (param && kind == TokenKind::Or) || matches!(word, "if" | "in")
}

fn stops(src: &Source, at: u32, stop: Stop, first: bool) -> bool {
    let common = is_pair(src, at, TokenKind::Eq, TokenKind::Gt) || matches!(src.kind(at), TokenKind::Comma | TokenKind::Semi);
    common
        || match stop {
            Stop::Expr => false,
            Stop::Ty => ty_stops(src, at, first),
            Stop::Pat => pat_stops(src, at, false),
            Stop::PatParam => pat_stops(src, at, true),
        }
}

fn ends_type(src: &Source, at: u32) -> bool {
    let Some(before) = at.checked_sub(1) else { return false };
    match src.kind(before) {
        TokenKind::Ident | TokenKind::RawIdent => !matches!(src.text(before), "mut" | "dyn" | "impl" | "const"),
        TokenKind::Gt | TokenKind::CloseParen | TokenKind::CloseBracket => true,
        _ => false,
    }
}

fn single_colon(src: &Source, at: u32) -> bool {
    let before = at > 0 && src.kind(at - 1) == TokenKind::Colon && src.joint(at - 1);
    let after = src.kind(at + 1) == TokenKind::Colon && src.joint(at);
    src.kind(at) == TokenKind::Colon && !before && !after
}

fn angle(src: &Source, at: u32, angles: u32) -> Option<u32> {
    let arrow = at > 0 && matches!(src.kind(at - 1), TokenKind::Minus | TokenKind::Eq) && src.joint(at - 1);
    match src.kind(at) {
        TokenKind::Lt => Some(angles + 1),
        TokenKind::Gt if arrow => Some(angles),
        TokenKind::Gt => angles.checked_sub(1),
        _ => Some(angles),
    }
}

fn until(src: &Source, mut at: u32, end: u32, stop: Stop) -> u32 {
    let start = at;
    let mut angles = 0_u32;
    while at < end {
        if angles == 0 && stops(src, at, stop, at == start) {
            break;
        }
        if stop == Stop::Ty {
            let Some(next) = angle(src, at, angles) else { break };
            angles = next;
        }
        at = tree(src, at);
    }
    at
}

fn literal(src: &Source, at: u32) -> Option<u32> {
    let at = if src.kind(at) == TokenKind::Minus { at + 1 } else { at };
    let word = is_name(src.kind(at)) && matches!(src.text(at), "true" | "false");
    (matches!(src.kind(at), TokenKind::Literal { .. }) || word).then_some(at + 1)
}

const PREFIX_WORDS: [&str; 20] = [
    "if", "match", "while", "for", "loop", "let", "move", "async", "unsafe", "return", "break", "mut", "const", "dyn", "ref", "box", "yield", "static", "impl", "where",
];

fn closure_params(src: &Source, at: u32, end: u32) -> u32 {
    if src.kind(at + 1) == TokenKind::Or && src.joint(at) {
        return at + 2;
    }
    let mut next = at + 1;
    while next < end && src.kind(next) != TokenKind::Or {
        next = tree(src, next);
    }
    next + 1
}

fn after_operand(src: &Source, at: u32) -> Option<(u32, bool)> {
    let kind = src.kind(at);
    if is_name(kind) {
        return match src.text(at) {
            "as" | "else" | "in" => Some((at + 1, false)),
            _ => None,
        };
    }
    match kind {
        TokenKind::Literal { .. } | TokenKind::Lifetime { .. } => None,
        kind if opens(kind) => Some((src.pair(at) + 1, true)),
        TokenKind::Question => Some((at + 1, true)),
        TokenKind::Bang if is_open_at(src, at + 1) => Some((src.pair(at + 1) + 1, true)),
        _ => Some((at + 1, false)),
    }
}

fn is_open_at(src: &Source, at: u32) -> bool {
    opens(src.kind(at))
}

fn turbofish(src: &Source, at: u32, end: u32) -> Option<u32> {
    let after_path = at >= 2 && src.kind(at - 1) == TokenKind::Colon && src.kind(at - 2) == TokenKind::Colon && src.joint(at - 2);
    if src.kind(at) != TokenKind::Lt || !after_path {
        return None;
    }
    let (mut next, mut angles) = (at, 0_u32);
    while next < end {
        angles = angle(src, next, angles)?;
        next = tree(src, next);
        if angles == 0 {
            return Some(next);
        }
    }
    None
}

fn before_operand(src: &Source, at: u32, end: u32) -> (u32, bool) {
    if let Some(next) = turbofish(src, at, end) {
        return (next, true);
    }
    let kind = src.kind(at);
    if is_name(kind) {
        return (at + 1, !PREFIX_WORDS.contains(&src.text(at)));
    }
    match kind {
        TokenKind::Literal { .. } | TokenKind::Lifetime { .. } => (at + 1, true),
        kind if opens(kind) => (src.pair(at) + 1, true),
        TokenKind::Or => (closure_params(src, at, end), false),
        _ => (at + 1, false),
    }
}

fn expr(src: &Source, mut at: u32, end: u32) -> u32 {
    let mut operand = false;
    while at < end && !stops(src, at, Stop::Expr, false) {
        let step = if operand { after_operand(src, at) } else { Some(before_operand(src, at, end)) };
        let Some((next, done)) = step else { break };
        at = next;
        operand = done;
    }
    at
}

fn vis(src: &Source, at: u32) -> u32 {
    if !is_name(src.kind(at)) {
        return at;
    }
    match src.text(at) {
        "pub" if src.kind(at + 1) == TokenKind::OpenParen => src.pair(at + 1) + 1,
        "pub" | "crate" => at + 1,
        _ => at,
    }
}

fn path(src: &Source, mut at: u32, end: u32) -> u32 {
    while at < end && (is_name(src.kind(at)) || src.kind(at) == TokenKind::Colon || src.kind(at) == TokenKind::Dollar) {
        at += 1;
    }
    at
}

fn meta(src: &Source, at: u32, end: u32) -> u32 {
    let after = path(src, at, end);
    match src.kind(after) {
        kind if opens(kind) && after < end => src.pair(after) + 1,
        TokenKind::Eq if after < end => expr(src, after + 1, end),
        _ => after,
    }
}

fn short(src: &Source, frag: Frag, at: u32) -> Option<u32> {
    let kind = src.kind(at);
    match frag {
        Frag::Tt => Some(tree(src, at)),
        Frag::Ident => (is_name(kind) && src.text(at) != "_").then_some(at + 1),
        Frag::Lifetime => matches!(kind, TokenKind::Lifetime { .. }).then_some(at + 1),
        Frag::Literal => literal(src, at),
        Frag::Block => (kind == TokenKind::OpenBrace).then(|| src.pair(at) + 1),
        _ => Some(vis(src, at)),
    }
}

fn parsed(frag: Frag) -> Option<Fragment> {
    Some(match frag {
        Frag::Ty => Fragment::Ty,
        Frag::Path => Fragment::Path,
        Frag::Pat => Fragment::Pat,
        Frag::PatParam => Fragment::PatParam,
        Frag::Expr => Fragment::Expr,
        _ => return None,
    })
}

fn long(src: &Source, frag: Frag, at: u32, end: u32) -> Option<u32> {
    if let Some(found) = parsed(frag).and_then(|fragment| fragment_end(src, fragment, at, end)) {
        return Some(found);
    }
    match frag {
        Frag::Item => scan::item_end(src, at).ok(),
        Frag::Meta => Some(meta(src, at, end)),
        Frag::Expr | Frag::Stmt => Some(expr(src, at, end)),
        Frag::Ty | Frag::Path => Some(until(src, at, end, Stop::Ty)),
        Frag::Pat => Some(until(src, at, end, Stop::Pat)),
        _ => Some(until(src, at, end, Stop::PatParam)),
    }
}

/// Whether a fragment of kind `frag` is found by looking at its first token or tree, without parsing.
pub(super) fn is_short(frag: Frag) -> bool {
    matches!(frag, Frag::Tt | Frag::Ident | Frag::Lifetime | Frag::Literal | Frag::Block | Frag::Vis)
}

/// The end of the fragment of kind `frag` that starts at `at`, or `None` when it cannot start there.
pub fn extent(src: &Source, frag: Frag, at: u32, end: u32) -> Option<u32> {
    if frag == Frag::Vis {
        return Some(vis(src, at));
    }
    if at >= end {
        return None;
    }
    let to = if is_short(frag) { short(src, frag, at)? } else { long(src, frag, at, end)? };
    (to > at && to <= end).then_some(to)
}
