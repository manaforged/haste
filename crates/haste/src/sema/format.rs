use anyhow::{Result, bail};

use crate::outline::source::{Mark, Source, Tokens};

use super::builtin::crate_path;

/// The `fmt::rt::Argument` constructor that each formatting trait spelling uses, with the debug-hex flag it adds.
const TRAITS: [(&str, &str, u32); 11] = [
    ("", "new_display", 0),
    ("?", "new_debug", 0),
    ("x?", "new_debug", 1 << 25),
    ("X?", "new_debug", 1 << 26),
    ("x", "new_lower_hex", 0),
    ("X", "new_upper_hex", 0),
    ("o", "new_octal", 0),
    ("b", "new_binary", 0),
    ("e", "new_lower_exp", 0),
    ("E", "new_upper_exp", 0),
    ("p", "new_pointer", 0),
];

const DEFAULT_FLAGS: u32 = 3 << 29 | ' ' as u32;
const WIDTH_SET: u32 = 1 << 27;
const PRECISION_SET: u32 = 1 << 28;

/// Which argument a placeholder or a count reads.
#[derive(Clone, PartialEq, Eq)]
enum Arg {
    Next,
    Index(usize),
    Name(String),
}

/// A width or precision: absent, written in the template, or read from an argument.
#[derive(Clone, PartialEq, Eq)]
enum Count {
    Unset,
    Literal(u16),
    Arg(Arg),
}

/// One `{...}` of a template.
struct Hole {
    arg: Arg,
    constructor: &'static str,
    flags: u32,
    width: Count,
    precision: Count,
}

/// One part of a template.
enum Piece {
    Text(String),
    Hole(Hole),
}

/// Where the value of an argument of the expansion comes from: a positional argument, a named one, or a variable that the template names.
#[derive(Clone, PartialEq, Eq)]
enum Origin {
    Positional(usize),
    Named(usize),
    Captured(String),
}

/// The arguments of one `format_args!` call after its template: positional token ranges, then `name = value` ones, and the mark of the template, which the names it captures take.
pub struct Call<'s> {
    pub src: &'s Source,
    pub mark: Option<Mark>,
    pub positional: Vec<(u32, u32)>,
    pub named: Vec<(&'s str, (u32, u32))>,
}

impl<'s> Call<'s> {
    /// Splits the arguments `args` of a call in `src` whose template starts at token `template` into positional and named ones.
    pub fn new(src: &'s Source, template: u32, args: &[(u32, u32)]) -> Call<'s> {
        let mut call = Call { src, mark: src.mark(template), positional: Vec::new(), named: Vec::new() };
        for &(from, to) in args {
            let assigns = to > from + 2 && src.text(from + 1) == "=" && !(src.joint(from + 1) && src.text(from + 2) == "=");
            match assigns && src.kind(from) == rustc_lexer::TokenKind::Ident {
                true => call.named.push((src.text(from), (from + 2, to))),
                false => call.positional.push((from, to)),
            }
        }
        call
    }
}

/// Writes the expansion of `format_args!` with the template `template` and the arguments `call`, for the macro of crate `krate`.
pub fn expand(template: &str, call: &Call<'_>, krate: u32, out: &mut Tokens) -> Result<()> {
    let pieces = parse(template)?;
    let mut sources = Vec::new();
    let mut next = 0;
    let mut values = Vec::new();
    let mut counts = Vec::new();
    for piece in &pieces {
        let Piece::Hole(hole) = piece else { continue };
        for count in [&hole.precision, &hole.width] {
            if let Count::Arg(arg) = count {
                counts.push(source_index(&mut sources, call, arg, &mut next)?);
            }
        }
        values.push((hole.constructor, source_index(&mut sources, call, &hole.arg, &mut next)?));
    }
    let bytes = encode(&pieces, values.len());
    write_call(out, krate, (&bytes, &sources, call), (&values, &counts))
}

fn source_index(sources: &mut Vec<Origin>, call: &Call<'_>, arg: &Arg, next: &mut usize) -> Result<usize> {
    let source = match arg {
        Arg::Next => {
            *next += 1;
            Origin::Positional(*next - 1)
        }
        Arg::Index(index) => Origin::Positional(*index),
        Arg::Name(name) => match call.named.iter().position(|(named, _)| named == name) {
            Some(at) => Origin::Named(at),
            None => Origin::Captured(name.clone()),
        },
    };
    if let Origin::Positional(index) = source
        && index >= call.positional.len()
    {
        bail!("format_args! has no argument {index}");
    }
    Ok(match sources.iter().position(|known| *known == source) {
        Some(at) => at,
        None => {
            sources.push(source);
            sources.len() - 1
        }
    })
}

fn write_call(out: &mut Tokens, krate: u32, (bytes, sources, call): (&[u8], &[Origin], &Call<'_>), (values, counts): (&[(&str, usize)], &[usize])) -> Result<()> {
    out.lexed("match (");
    for source in sources {
        out.lexed("&");
        let (from, to) = match source {
            Origin::Positional(index) => call.positional[*index],
            Origin::Named(index) => call.named[*index].1,
            Origin::Captured(name) => {
                out.ident(name, call.mark);
                out.lexed(",");
                continue;
            }
        };
        out.lexed("(");
        out.copy_range(call.src, from, to);
        out.lexed("),");
    }
    out.lexed(") { args => unsafe {");
    crate_path(out, krate, "::fmt::Arguments::new");
    out.lexed(&format!("({}, &[", byte_string(bytes)));
    for &(constructor, at) in values {
        crate_path(out, krate, &format!("::fmt::rt::Argument::{constructor}(args.{at}),"));
    }
    for &at in counts {
        crate_path(out, krate, &format!("::fmt::rt::Argument::from_usize(args.{at}),"));
    }
    out.lexed("]) } }");
    Ok(())
}

fn byte_string(bytes: &[u8]) -> String {
    let mut text = String::from("b\"");
    for &byte in bytes {
        match byte {
            b' '..=b'~' if byte != b'"' && byte != b'\\' => text.push(byte as char),
            _ => text.push_str(&format!("\\x{byte:02x}")),
        }
    }
    text.push('"');
    text
}

fn encode(pieces: &[Piece], holes: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut count_at = holes;
    for piece in pieces {
        match piece {
            Piece::Text(text) => encode_text(&mut bytes, text),
            Piece::Hole(hole) => encode_hole(&mut bytes, hole, &mut count_at),
        }
    }
    bytes.push(0);
    bytes
}

fn encode_text(bytes: &mut Vec<u8>, mut text: &str) {
    while !text.is_empty() {
        let mut take = text.len().min(u16::MAX as usize);
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        let (head, rest) = text.split_at(take);
        if head.len() < 0x80 {
            bytes.push(head.len() as u8);
        } else {
            bytes.push(0x80);
            bytes.extend_from_slice(&(head.len() as u16).to_le_bytes());
        }
        bytes.extend_from_slice(head.as_bytes());
        text = rest;
    }
}

fn encode_hole(bytes: &mut Vec<u8>, hole: &Hole, count_at: &mut usize) {
    let mut fields = Vec::new();
    let mut lead = 0xC0_u8;
    let mut flags = hole.flags;
    let precision = count_value(&hole.precision, count_at);
    let width = count_value(&hole.width, count_at);
    flags |= if width.is_some() { WIDTH_SET } else { 0 } | if precision.is_some() { PRECISION_SET } else { 0 };
    if flags != DEFAULT_FLAGS {
        lead |= 1;
        fields.extend_from_slice(&flags.to_le_bytes());
    }
    if let Some((value, indirect)) = width {
        lead |= 2 | if indirect { 16 } else { 0 };
        fields.extend_from_slice(&value.to_le_bytes());
    }
    if let Some((value, indirect)) = precision {
        lead |= 4 | if indirect { 32 } else { 0 };
        fields.extend_from_slice(&value.to_le_bytes());
    }
    bytes.push(lead);
    bytes.extend_from_slice(&fields);
}

fn count_value(count: &Count, count_at: &mut usize) -> Option<(u16, bool)> {
    match count {
        Count::Unset => None,
        Count::Literal(value) => Some((*value, false)),
        Count::Arg(_) => {
            *count_at += 1;
            Some(((*count_at - 1) as u16, true))
        }
    }
}

fn parse(template: &str) -> Result<Vec<Piece>> {
    let mut pieces = Vec::new();
    let mut text = String::new();
    let mut chars = template.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        match (c, chars.peek().map(|&(_, next)| next)) {
            ('{', Some('{')) | ('}', Some('}')) => {
                chars.next();
                text.push(c);
            }
            ('{', _) => {
                let Some(end) = template[at..].find('}') else { bail!("a format template with an unclosed `{{`") };
                pieces.push(Piece::Text(std::mem::take(&mut text)));
                pieces.push(Piece::Hole(hole(&template[at + 1..at + end])?));
                while chars.peek().is_some_and(|&(next, _)| next <= at + end) {
                    chars.next();
                }
            }
            ('}', _) => bail!("a format template with an unopened `}}`"),
            _ => text.push(c),
        }
    }
    pieces.push(Piece::Text(text));
    pieces.retain(|piece| !matches!(piece, Piece::Text(text) if text.is_empty()));
    Ok(pieces)
}

fn hole(inside: &str) -> Result<Hole> {
    let (arg, spec) = inside.split_once(':').unwrap_or((inside, ""));
    let mut hole = Hole { arg: arg_ref(arg.trim()), constructor: "new_display", flags: DEFAULT_FLAGS, width: Count::Unset, precision: Count::Unset };
    let rest = fill_align(spec, &mut hole.flags);
    let rest = sign_flags(rest, &mut hole.flags);
    let (width, rest) = count(rest);
    hole.width = width;
    let rest = match rest.strip_prefix('.') {
        Some(after) if after.starts_with('*') => {
            hole.precision = Count::Arg(Arg::Next);
            &after[1..]
        }
        Some(after) => {
            let (precision, rest) = count(after);
            hole.precision = precision;
            rest
        }
        None => rest,
    };
    let Some(&(_, constructor, flag)) = TRAITS.iter().find(|(spelling, ..)| *spelling == rest) else { bail!("the format trait `{rest}` is not known") };
    (hole.constructor, hole.flags) = (constructor, hole.flags | flag);
    Ok(hole)
}

fn arg_ref(text: &str) -> Arg {
    match text {
        "" => Arg::Next,
        _ => text.parse().map_or_else(|_| Arg::Name(text.to_string()), Arg::Index),
    }
}

fn fill_align<'t>(spec: &'t str, flags: &mut u32) -> &'t str {
    let align = |c: char| ['<', '>', '^'].iter().position(|&known| known == c).map(|at| at as u32);
    let mut chars = spec.chars();
    let (first, second) = (chars.next(), chars.next());
    let (fill, align_at, skip) = match (first, second.and_then(align), first.and_then(align)) {
        (Some(fill), Some(found), _) => (fill, Some(found), fill.len_utf8() + 1),
        (_, _, Some(found)) => (' ', Some(found), 1),
        _ => (' ', None, 0),
    };
    if let Some(found) = align_at {
        *flags = (*flags & !(3 << 29) & !0x1F_FFFF) | found << 29 | fill as u32;
    }
    &spec[skip..]
}

fn sign_flags<'t>(mut spec: &'t str, flags: &mut u32) -> &'t str {
    for (mark, bit) in [('+', 21), ('-', 22), ('#', 23)] {
        if let Some(rest) = spec.strip_prefix(mark) {
            (*flags, spec) = (*flags | 1 << bit, rest);
        }
    }
    if spec.starts_with('0') && !spec[1..].starts_with('$') {
        (*flags, spec) = (*flags | 1 << 24, &spec[1..]);
    }
    spec
}

fn count(spec: &str) -> (Count, &str) {
    let digits = spec.find(|c: char| !c.is_ascii_digit()).unwrap_or(spec.len());
    let word = match digits {
        0 => spec.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(spec.len()),
        _ => digits,
    };
    let (text, rest) = spec.split_at(word);
    match (text, rest.strip_prefix('$')) {
        ("", _) => (Count::Unset, spec),
        (text, Some(rest)) => (Count::Arg(arg_ref(text)), rest),
        (text, None) if digits > 0 => (text.parse().map_or(Count::Unset, Count::Literal), rest),
        _ => (Count::Unset, spec),
    }
}
