use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use rustc_lexer::{Base, DocStyle, LiteralKind, TokenKind};

use crate::graph::cfg::Cfg;
use crate::outline::item::{Attr, Import, Item, ItemKind, Items, Leaf, Vis};
use crate::outline::rules::{Frag, Meta, Op, Piece, Rule};
use crate::outline::source::{Mark, Origins, Token};
use crate::sema::summary::codec::{Cursor, write_u32, write_u32s};
use crate::substrate::{bytes_to_path, path_to_bytes};

const BASE_KINDS: [Base; 4] = [Base::Binary, Base::Octal, Base::Decimal, Base::Hexadecimal];
const FRAG_KINDS: [Frag; 14] = [
    Frag::Ident, Frag::Path, Frag::Expr, Frag::Ty, Frag::Pat, Frag::PatParam, Frag::Stmt,
    Frag::Block, Frag::Item, Frag::Meta, Frag::Tt, Frag::Literal, Frag::Lifetime, Frag::Vis,
];
const OP_KINDS: [Op; 3] = [Op::Many, Op::Some, Op::Maybe];
const ITEM_KINDS: [ItemKind; 17] = [
    ItemKind::Mod, ItemKind::Use, ItemKind::ExternCrate, ItemKind::Fn, ItemKind::Struct, ItemKind::Enum,
    ItemKind::Union, ItemKind::Trait, ItemKind::TraitAlias, ItemKind::Impl, ItemKind::Type, ItemKind::Const,
    ItemKind::Static, ItemKind::MacroRules, ItemKind::Macro, ItemKind::MacroCall, ItemKind::ExternBlock,
];
const SIMPLE_TOKENS: [TokenKind; 38] = [
    TokenKind::Whitespace, TokenKind::Ident, TokenKind::InvalidIdent, TokenKind::RawIdent, TokenKind::ForcedKeywordIdent,
    TokenKind::UnknownPrefix, TokenKind::UnknownPrefixLifetime, TokenKind::RawLifetime, TokenKind::GuardedStrPrefix,
    TokenKind::Semi, TokenKind::Comma, TokenKind::Dot,
    TokenKind::At, TokenKind::Pound, TokenKind::Tilde, TokenKind::Question, TokenKind::Colon, TokenKind::Dollar,
    TokenKind::Eq, TokenKind::Bang, TokenKind::Lt, TokenKind::Gt, TokenKind::Minus, TokenKind::And, TokenKind::Or,
    TokenKind::Plus, TokenKind::Star, TokenKind::Slash, TokenKind::Caret, TokenKind::Percent, TokenKind::Unknown, TokenKind::Eof,
    TokenKind::OpenParen, TokenKind::CloseParen, TokenKind::OpenBrace, TokenKind::CloseBrace, TokenKind::OpenBracket, TokenKind::CloseBracket,
];

#[inline]
pub(super) fn u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

pub(super) fn read_u32s_map<T>(cur: &mut Cursor<'_>, mut f: impl FnMut(u32) -> T) -> Result<Vec<T>> {
    let count = cur.count()?;
    let byte_len = count.checked_mul(4).context("u32 slice overflow")?;
    let bytes = cur.take(byte_len)?;
    Ok(bytes.chunks_exact(4).map(|c| f(u32_le(c, 0))).collect())
}

pub(super) fn read_paths(cur: &mut Cursor<'_>) -> Result<Vec<PathBuf>> {
    let count = cur.count()?;
    let mut paths = Vec::with_capacity(count);
    for _ in 0..count {
        paths.push(read_path(cur)?);
    }
    Ok(paths)
}

pub(super) fn write_paths(out: &mut Vec<u8>, paths: &[PathBuf]) -> Option<()> {
    write_usize(out, paths.len())?;
    for path in paths {
        write_path(out, path)?;
    }
    Some(())
}

pub(super) fn write_str(out: &mut Vec<u8>, s: &str) -> Option<()> {
    write_u32(out, u32::try_from(s.len()).ok()?);
    out.extend_from_slice(s.as_bytes());
    Some(())
}

pub(super) fn read_str<'a>(cur: &mut Cursor<'a>) -> Result<&'a str> {
    let len = cur.count()?;
    std::str::from_utf8(cur.take(len)?).context("invalid utf-8 string")
}

pub(super) fn write_path(out: &mut Vec<u8>, path: &std::path::Path) -> Option<()> {
    let bytes = path_to_bytes(path)?;
    write_u32(out, u32::try_from(bytes.len()).ok()?);
    out.extend_from_slice(&bytes);
    Some(())
}

pub(super) fn read_path(cur: &mut Cursor<'_>) -> Result<PathBuf> {
    let len = cur.count()?;
    bytes_to_path(cur.take(len)?).context("invalid path bytes")
}

pub(super) fn write_opt_u32(out: &mut Vec<u8>, val: Option<u32>) {
    write_u32(out, val.unwrap_or(u32::MAX));
}

pub(super) fn read_opt_u32(cur: &mut Cursor<'_>) -> Result<Option<u32>> {
    let v = cur.u32()?;
    Ok((v != u32::MAX).then_some(v))
}

pub(super) fn write_usize(out: &mut Vec<u8>, val: usize) -> Option<()> {
    write_u32(out, u32::try_from(val).ok()?);
    Some(())
}

pub(super) fn vis_tag(vis: Vis) -> (u8, u32) {
    match vis {
        Vis::Private => (0, 0),
        Vis::Pub => (1, 0),
        Vis::Crate => (2, 0),
        Vis::Super => (3, 0),
        Vis::SelfMod => (4, 0),
        Vis::In(at) => (5, at),
    }
}

pub(super) fn vis_from_tag(tag: u8, at: u32) -> Result<Vis> {
    Ok(match tag {
        0 => Vis::Private,
        1 => Vis::Pub,
        2 => Vis::Crate,
        3 => Vis::Super,
        4 => Vis::SelfMod,
        5 => Vis::In(at),
        _ => bail!("invalid vis tag"),
    })
}

fn doc_bits(doc: Option<DocStyle>) -> u16 {
    match doc {
        None => 0,
        Some(DocStyle::Outer) => 1,
        Some(DocStyle::Inner) => 2,
    }
}

fn doc_from_bits(bits: u16) -> Result<Option<DocStyle>> {
    Ok(match bits {
        0 => None,
        1 => Some(DocStyle::Outer),
        2 => Some(DocStyle::Inner),
        _ => bail!("invalid doc style tag"),
    })
}

fn base_bits(base: Base) -> u16 {
    BASE_KINDS.iter().position(|&b| b == base).unwrap_or(0) as u16
}

fn encode_token(kind: TokenKind, pair: u32) -> Option<(u16, u32)> {
    let simple = match kind {
        TokenKind::Whitespace => 0,
        TokenKind::Ident => 1,
        TokenKind::InvalidIdent => 2,
        TokenKind::RawIdent => 3,
        TokenKind::ForcedKeywordIdent => 4,
        TokenKind::UnknownPrefix => 5,
        TokenKind::UnknownPrefixLifetime => 6,
        TokenKind::RawLifetime => 7,
        TokenKind::GuardedStrPrefix => 8,
        TokenKind::Semi => 9,
        TokenKind::Comma => 10,
        TokenKind::Dot => 11,
        TokenKind::At => 12,
        TokenKind::Pound => 13,
        TokenKind::Tilde => 14,
        TokenKind::Question => 15,
        TokenKind::Colon => 16,
        TokenKind::Dollar => 17,
        TokenKind::Eq => 18,
        TokenKind::Bang => 19,
        TokenKind::Lt => 20,
        TokenKind::Gt => 21,
        TokenKind::Minus => 22,
        TokenKind::And => 23,
        TokenKind::Or => 24,
        TokenKind::Plus => 25,
        TokenKind::Star => 26,
        TokenKind::Slash => 27,
        TokenKind::Caret => 28,
        TokenKind::Percent => 29,
        TokenKind::Unknown => 30,
        TokenKind::Eof => 31,
        TokenKind::OpenParen => 32,
        TokenKind::CloseParen => 33,
        TokenKind::OpenBrace => 34,
        TokenKind::CloseBrace => 35,
        TokenKind::OpenBracket => 36,
        TokenKind::CloseBracket => 37,
        TokenKind::LineComment { doc_style } => return Some((38 | (doc_bits(doc_style) << 6), u32::MAX)),
        TokenKind::BlockComment { doc_style, terminated } => {
            return Some((39 | ((doc_bits(doc_style) | (u16::from(terminated) << 2)) << 6), u32::MAX));
        }
        TokenKind::Frontmatter { has_invalid_preceding_whitespace, invalid_infostring } => {
            let flags = u16::from(has_invalid_preceding_whitespace) | (u16::from(invalid_infostring) << 1);
            return Some((40 | (flags << 6), u32::MAX));
        }
        TokenKind::Lifetime { starts_with_number } => return Some((41 | (u16::from(starts_with_number) << 6), u32::MAX)),
        TokenKind::Literal { kind, suffix_start } => {
            let (tag, payload) = match kind {
                LiteralKind::Int { base, empty_int } => (42, base_bits(base) | (u16::from(empty_int) << 2)),
                LiteralKind::Float { base, empty_exponent } => (43, base_bits(base) | (u16::from(empty_exponent) << 2)),
                LiteralKind::Char { terminated } => (44, u16::from(terminated)),
                LiteralKind::Byte { terminated } => (45, u16::from(terminated)),
                LiteralKind::Str { terminated } => (46, u16::from(terminated)),
                LiteralKind::ByteStr { terminated } => (47, u16::from(terminated)),
                LiteralKind::CStr { terminated } => (48, u16::from(terminated)),
                LiteralKind::RawStr { n_hashes } => (49, n_hashes.map_or(256, u16::from)),
                LiteralKind::RawByteStr { n_hashes } => (50, n_hashes.map_or(256, u16::from)),
                LiteralKind::RawCStr { n_hashes } => (51, n_hashes.map_or(256, u16::from)),
            };
            return Some((tag | (payload << 6), suffix_start));
        }
    };
    Some((simple, pair))
}

fn raw_hashes(payload: u16) -> Result<Option<u8>> {
    if payload > 256 {
        bail!("invalid raw literal hash count");
    }
    Ok((payload < 256).then_some(payload as u8))
}

fn decode_complex_token(tag: usize, payload: u16, extra: u32) -> Result<TokenKind> {
    Ok(match tag {
        38 => TokenKind::LineComment { doc_style: doc_from_bits(payload)? },
        39 => TokenKind::BlockComment { doc_style: doc_from_bits(payload & 3)?, terminated: payload & 4 != 0 },
        40 => TokenKind::Frontmatter { has_invalid_preceding_whitespace: payload & 1 != 0, invalid_infostring: payload & 2 != 0 },
        41 => TokenKind::Lifetime { starts_with_number: payload != 0 },
        42..=51 => {
            let kind = match tag {
                42 => LiteralKind::Int { base: BASE_KINDS[(payload & 3) as usize], empty_int: payload & 4 != 0 },
                43 => LiteralKind::Float { base: BASE_KINDS[(payload & 3) as usize], empty_exponent: payload & 4 != 0 },
                44 => LiteralKind::Char { terminated: payload != 0 },
                45 => LiteralKind::Byte { terminated: payload != 0 },
                46 => LiteralKind::Str { terminated: payload != 0 },
                47 => LiteralKind::ByteStr { terminated: payload != 0 },
                48 => LiteralKind::CStr { terminated: payload != 0 },
                49 => LiteralKind::RawStr { n_hashes: raw_hashes(payload)? },
                50 => LiteralKind::RawByteStr { n_hashes: raw_hashes(payload)? },
                51 => LiteralKind::RawCStr { n_hashes: raw_hashes(payload)? },
                _ => unreachable!(),
            };
            TokenKind::Literal { kind, suffix_start: extra }
        }
        _ => bail!("invalid token kind tag"),
    })
}

pub(crate) const TOKEN_WIRE_LEN: usize = 14;

pub(super) fn write_tokens(out: &mut Vec<u8>, tokens: &[Token]) -> Option<()> {
    write_usize(out, tokens.len())?;
    out.reserve(tokens.len() * TOKEN_WIRE_LEN);
    for &token in tokens {
        let (len, pair) = token.raw();
        let (code, extra) = encode_token(token.kind, pair)?;
        out.extend_from_slice(&code.to_le_bytes());
        write_u32(out, token.start);
        write_u32(out, len);
        write_u32(out, extra);
    }
    Some(())
}

/// Decodes one wire-encoded token from `bytes[offset..offset + TOKEN_WIRE_LEN]`.
#[inline]
pub(crate) fn decode_token_at(bytes: &[u8], offset: usize) -> Token {
    let c = &bytes[offset..offset + TOKEN_WIRE_LEN];
    let code = u16::from_le_bytes([c[0], c[1]]);
    let tag = (code & 0x3f) as usize;
    let start = u32_le(c, 2);
    let len = u32_le(c, 6);
    let extra = u32_le(c, 10);
    if tag < SIMPLE_TOKENS.len() {
        Token::from_raw(SIMPLE_TOKENS[tag], start, len, extra)
    } else {
        let kind = decode_complex_token(tag, code >> 6, extra).expect("valid sysroot token kind");
        Token::from_raw(kind, start, len, u32::MAX)
    }
}

/// Reads `(start, raw_len)` of one wire-encoded token without decoding its `TokenKind`.
#[inline]
pub(crate) fn token_raw_span_at(bytes: &[u8], offset: usize) -> (u32, u32) {
    let c = &bytes[offset..offset + 10];
    (u32_le(c, 2), u32_le(c, 6))
}

/// Decodes `count` wire-encoded tokens from `bytes`.
pub(crate) fn decode_tokens_slice(bytes: &[u8], count: usize) -> Result<Vec<Token>> {
    let expected = count.checked_mul(TOKEN_WIRE_LEN).context("token stream overflow")?;
    if bytes.len() != expected {
        bail!("token stream byte length mismatch");
    }
    let mut tokens = Vec::with_capacity(count);
    for c in bytes.chunks_exact(TOKEN_WIRE_LEN) {
        let code = u16::from_le_bytes([c[0], c[1]]);
        let tag = (code & 0x3f) as usize;
        let start = u32_le(c, 2);
        let len = u32_le(c, 6);
        let extra = u32_le(c, 10);
        if tag < SIMPLE_TOKENS.len() {
            tokens.push(Token::from_raw(SIMPLE_TOKENS[tag], start, len, extra));
        } else {
            let kind = decode_complex_token(tag, code >> 6, extra)?;
            tokens.push(Token::from_raw(kind, start, len, u32::MAX));
        }
    }
    Ok(tokens)
}

pub(super) fn frag_tag(frag: Frag) -> u8 {
    FRAG_KINDS.iter().position(|&f| f == frag).unwrap_or(0) as u8
}

pub(super) fn frag_from_tag(tag: u8) -> Result<Frag> {
    FRAG_KINDS.get(tag as usize).copied().context("invalid fragment tag")
}

pub(super) fn write_frag(out: &mut Vec<u8>, frag: Frag) {
    out.push(frag_tag(frag));
}

fn read_frag(cur: &mut Cursor<'_>) -> Result<Frag> {
    frag_from_tag(cur.u8()?)
}

pub(super) fn write_marks(out: &mut Vec<u8>, marks: &[(u32, Mark)], to_local: &[Option<u32>]) -> Option<()> {
    write_usize(out, marks.len())?;
    let len_at = out.len();
    write_u32(out, 0);
    let body_start = out.len();
    out.reserve(marks.len() * 13);
    for &(at, mark) in marks {
        write_u32(out, at);
        match mark {
            Mark::Crate(k) => {
                out.push(0);
                write_u32(out, (*to_local.get(k as usize)?)?);
            }
            Mark::Home { krate, file, item } => {
                out.push(1);
                write_u32(out, (*to_local.get(krate as usize)?)?);
                write_u32(out, file);
                write_u32(out, item);
            }
            Mark::Scope { id, parent } => {
                out.push(2);
                write_u32(out, id);
                write_u32(out, parent);
            }
        }
    }
    let byte_len = u32::try_from(out.len() - body_start).ok()?;
    out[len_at..len_at + 4].copy_from_slice(&byte_len.to_le_bytes());
    Some(())
}

pub(super) fn read_marks(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<(u32, Mark)>> {
    const CACHED_SCOPE_BIT: u32 = 1 << 31;
    let map_scope = |s: u32| if s == 0 { 0 } else { s | CACHED_SCOPE_BIT };
    let count = cur.count()?;
    let byte_len = cur.count()?;
    let data = cur.take(byte_len)?;
    if count.saturating_mul(9) > data.len() {
        bail!("mark count exceeds byte length");
    }
    let mut marks = Vec::with_capacity(count);
    let mut pos = 0usize;
    while marks.len() < count && pos + 17 <= data.len() {
        let b = &data[pos..pos + 17];
        let at = u32_le(b, 0);
        let mark = match b[4] {
            0 => {
                pos += 9;
                Mark::Crate(*remap.get(u32_le(b, 5) as usize).context("mark crate out of bounds")?)
            }
            1 => {
                pos += 17;
                Mark::Home {
                    krate: *remap.get(u32_le(b, 5) as usize).context("mark home crate out of bounds")?,
                    file: u32_le(b, 9),
                    item: u32_le(b, 13),
                }
            }
            2 => {
                pos += 13;
                Mark::Scope { id: map_scope(u32_le(b, 5)), parent: map_scope(u32_le(b, 9)) }
            }
            _ => bail!("invalid mark tag"),
        };
        marks.push((at, mark));
    }
    let mut tail = Cursor::new(&data[pos..]);
    while marks.len() < count {
        let at = tail.u32()?;
        let mark = match tail.u8()? {
            0 => Mark::Crate(*remap.get(tail.u32()? as usize).context("mark crate out of bounds")?),
            1 => Mark::Home {
                krate: *remap.get(tail.u32()? as usize).context("mark home crate out of bounds")?,
                file: tail.u32()?,
                item: tail.u32()?,
            },
            2 => Mark::Scope { id: map_scope(tail.u32()?), parent: map_scope(tail.u32()?) },
            _ => bail!("invalid mark tag"),
        };
        marks.push((at, mark));
    }
    if pos + tail.pos != data.len() {
        bail!("trailing bytes in sysroot marks");
    }
    Ok(marks)
}

pub(super) fn write_origins(out: &mut Vec<u8>, origins: &Origins) -> Option<()> {
    match origins {
        Origins::Read => out.push(0),
        Origins::Copied(pairs) => {
            out.push(1);
            write_usize(out, pairs.len())?;
            for &(a, b) in pairs { write_u32(out, a); write_u32(out, b); }
        }
    }
    Some(())
}

pub(super) fn read_origins(cur: &mut Cursor<'_>) -> Result<Origins> {
    Ok(match cur.u8()? {
        0 => Origins::Read,
        1 => {
            let count = cur.count()?;
            let bytes = cur.take(count.checked_mul(8).context("origins overflow")?)?;
            Origins::Copied(bytes.chunks_exact(8).map(|c| (u32_le(c, 0), u32_le(c, 4))).collect())
        }
        _ => bail!("invalid origins tag"),
    })
}

pub(super) fn write_vis(out: &mut Vec<u8>, vis: Vis) {
    let (tag, at) = vis_tag(vis);
    out.push(tag);
    if tag == 5 {
        write_u32(out, at);
    }
}

pub(super) fn read_vis(cur: &mut Cursor<'_>) -> Result<Vis> {
    let tag = cur.u8()?;
    let at = if tag == 5 { cur.u32()? } else { 0 };
    vis_from_tag(tag, at)
}

fn write_cfg(out: &mut Vec<u8>, cfg: &Cfg) -> Option<()> {
    match cfg {
        Cfg::Name(name) => { out.push(0); write_str(out, name)?; }
        Cfg::KeyValue(k, v) => { out.push(1); write_str(out, k)?; write_str(out, v)?; }
        Cfg::All(list) => {
            out.push(2);
            write_usize(out, list.len())?;
            for item in list { write_cfg(out, item)?; }
        }
        Cfg::Any(list) => {
            out.push(3);
            write_usize(out, list.len())?;
            for item in list { write_cfg(out, item)?; }
        }
        Cfg::Not(inner) => { out.push(4); write_cfg(out, inner)?; }
    }
    Some(())
}

fn read_cfg(cur: &mut Cursor<'_>) -> Result<Cfg> {
    Ok(match cur.u8()? {
        0 => Cfg::Name(read_str(cur)?.to_owned()),
        1 => Cfg::KeyValue(read_str(cur)?.to_owned(), read_str(cur)?.to_owned()),
        2 => {
            let len = cur.count()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len { list.push(read_cfg(cur)?); }
            Cfg::All(list)
        }
        3 => {
            let len = cur.count()?;
            let mut list = Vec::with_capacity(len);
            for _ in 0..len { list.push(read_cfg(cur)?); }
            Cfg::Any(list)
        }
        4 => Cfg::Not(Box::new(read_cfg(cur)?)),
        _ => bail!("invalid cfg tag"),
    })
}

fn write_attrs(out: &mut Vec<u8>, attrs: &[Attr]) -> Option<()> {
    write_usize(out, attrs.len())?;
    for attr in attrs {
        match attr {
            Attr::Cfg(cfg) => { out.push(0); write_cfg(out, cfg)?; }
            Attr::CfgAttr(cfg, inner) => { out.push(1); write_cfg(out, cfg)?; write_attrs(out, inner)?; }
            Attr::Path(p) => { out.push(2); write_str(out, p)?; }
            Attr::MacroUse => out.push(3),
            Attr::MacroExport(local) => { out.push(4); out.push(u8::from(*local)); }
            Attr::Derive(at) => { out.push(5); write_u32(out, *at); }
            Attr::NoStd => out.push(6),
            Attr::NoCore => out.push(7),
            Attr::NoImplicitPrelude => out.push(8),
            Attr::Other(at) => { out.push(9); write_u32(out, *at); }
        }
    }
    Some(())
}

fn read_attrs(cur: &mut Cursor<'_>) -> Result<Vec<Attr>> {
    let len = cur.count()?;
    let mut attrs = Vec::with_capacity(len);
    for _ in 0..len {
        attrs.push(match cur.u8()? {
            0 => Attr::Cfg(read_cfg(cur)?),
            1 => Attr::CfgAttr(read_cfg(cur)?, read_attrs(cur)?),
            2 => Attr::Path(read_str(cur)?.to_owned()),
            3 => Attr::MacroUse,
            4 => Attr::MacroExport(cur.u8()? != 0),
            5 => Attr::Derive(cur.u32()?),
            6 => Attr::NoStd,
            7 => Attr::NoCore,
            8 => Attr::NoImplicitPrelude,
            9 => Attr::Other(cur.u32()?),
            _ => bail!("invalid attr tag"),
        });
    }
    Ok(attrs)
}

pub(super) fn write_items(out: &mut Vec<u8>, items: &Items) -> Option<()> {
    write_attrs(out, &items.inner)?;
    write_usize(out, items.items.len())?;
    out.reserve(items.items.len() * 30);
    for item in &items.items {
        let (vtag, vis_at) = vis_tag(item.vis);
        let has_attrs = u8::from(!item.attrs.is_empty()) << 7;
        out.push(ITEM_KINDS.iter().position(|&k| k == item.kind)? as u8);
        out.push(vtag | has_attrs);
        write_opt_u32(out, item.name);
        write_u32(out, item.lead);
        write_u32(out, item.start);
        write_u32(out, item.end);
        write_opt_u32(out, item.body);
        write_opt_u32(out, item.parent);
        write_u32(out, vis_at);
    }
    for item in &items.items {
        if !item.attrs.is_empty() {
            write_attrs(out, &item.attrs)?;
        }
    }
    write_usize(out, items.imports.len())?;
    let total_paths: usize = items.imports.iter().map(|i| i.path.len()).sum();
    write_usize(out, total_paths)?;
    out.reserve(items.imports.len() * 13 + total_paths * 4);
    for import in &items.imports {
        let (leaf_tag, alias) = match import.leaf {
            Leaf::Name(alias) => (0u8, alias.unwrap_or(u32::MAX)),
            Leaf::Underscore => (1u8, u32::MAX),
            Leaf::Glob => (2u8, u32::MAX),
        };
        out.push(leaf_tag | (u8::from(import.absolute) << 2));
        write_u32(out, import.item);
        write_u32(out, alias);
        write_usize(out, import.path.len())?;
    }
    for import in &items.imports {
        for &seg in &import.path {
            write_u32(out, seg);
        }
    }
    write_usize(out, items.marks.len())?;
    out.reserve(items.marks.len() * 6);
    for &(at, bits) in &items.marks {
        write_u32(out, at);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Some(())
}

pub(super) fn read_items(cur: &mut Cursor<'_>) -> Result<Items> {
    let inner = read_attrs(cur)?;
    let item_count = cur.count()?;
    let item_bytes = cur.take(item_count.checked_mul(30).context("items overflow")?)?;
    let mut items = Vec::with_capacity(item_count);
    let mut attr_indices = Vec::new();
    let opt_at = |b: &[u8], at: usize| {
        let v = u32_le(b, at);
        (v != u32::MAX).then_some(v)
    };
    for (idx, b) in item_bytes.chunks_exact(30).enumerate() {
        let kind = ITEM_KINDS.get(b[0] as usize).copied().context("invalid item kind tag")?;
        let flags = b[1];
        let vis = vis_from_tag(flags & 0x7f, u32_le(b, 26))?;
        if flags & 0x80 != 0 {
            attr_indices.push(idx);
        }
        items.push(Item {
            kind,
            name: opt_at(b, 2),
            vis,
            attrs: Vec::new(),
            lead: u32_le(b, 6),
            start: u32_le(b, 10),
            end: u32_le(b, 14),
            body: opt_at(b, 18),
            parent: opt_at(b, 22),
        });
    }
    for idx in attr_indices {
        items[idx].attrs = read_attrs(cur)?;
    }
    let import_count = cur.count()?;
    let total_paths = cur.count()?;
    let import_bytes = cur.take(import_count.checked_mul(13).context("imports overflow")?)?;
    let path_bytes = cur.take(total_paths.checked_mul(4).context("import paths overflow")?)?;
    let mut imports = Vec::with_capacity(import_count);
    let mut path_pos = 0usize;
    for b in import_bytes.chunks_exact(13) {
        let flags = b[0];
        let item = u32_le(b, 1);
        let alias = u32_le(b, 5);
        let path_len = u32_le(b, 9) as usize;
        let path_byte_len = path_len.checked_mul(4).context("import path length overflow")?;
        let path_end = path_pos.checked_add(path_byte_len).context("import path overflow")?;
        let seg_slice = path_bytes.get(path_pos..path_end).context("import path out of bounds")?;
        path_pos = path_end;
        let path = seg_slice.chunks_exact(4).map(|p| u32_le(p, 0)).collect();
        let leaf = match flags & 3 {
            0 => Leaf::Name((alias != u32::MAX).then_some(alias)),
            1 => Leaf::Underscore,
            2 => Leaf::Glob,
            _ => bail!("invalid import leaf tag"),
        };
        imports.push(Import { item, absolute: flags & 4 != 0, path, leaf });
    }
    if path_pos != path_bytes.len() {
        bail!("trailing bytes in import paths");
    }
    let mark_count = cur.count()?;
    let mark_bytes = cur.take(mark_count.checked_mul(6).context("item marks overflow")?)?;
    let marks = mark_bytes.chunks_exact(6).map(|c| (u32_le(c, 0), u16::from_le_bytes([c[4], c[5]]))).collect();
    Ok(Items { inner, items, imports, marks })
}

pub(super) fn write_rules(out: &mut Vec<u8>, rules: &[Rule], to_local: &[Option<u32>]) -> Option<()> {
    write_usize(out, rules.len())?;
    for rule in rules {
        write_pieces(out, &rule.matcher, to_local)?;
        write_pieces(out, &rule.body, to_local)?;
        write_usize(out, rule.slots)?;
    }
    Some(())
}

pub(super) fn read_rules(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<Rule>> {
    let count = cur.count()?;
    let mut rules = Vec::with_capacity(count);
    for _ in 0..count {
        rules.push(Rule {
            matcher: read_pieces(cur, remap)?,
            body: read_pieces(cur, remap)?,
            slots: cur.u32()? as usize,
        });
    }
    Ok(rules)
}

fn write_pieces(out: &mut Vec<u8>, pieces: &[Piece], to_local: &[Option<u32>]) -> Option<()> {
    write_usize(out, pieces.len())?;
    for piece in pieces {
        match piece {
            Piece::Token(t) => { out.push(0); write_u32(out, *t); }
            Piece::Group(open, inner) => { out.push(1); write_u32(out, *open); write_pieces(out, inner, to_local)?; }
            Piece::Var(at, frag, slot) => {
                out.push(2);
                write_u32(out, *at);
                match frag { Some(f) => { out.push(1); write_frag(out, *f); } None => out.push(0) }
                write_opt_u32(out, *slot);
            }
            Piece::Crate(krate) => {
                out.push(3);
                let mapped = match krate {
                    Some(k) => Some((*to_local.get(*k as usize)?)?),
                    None => None,
                };
                write_opt_u32(out, mapped);
            }
            Piece::CratePath => out.push(4),
            Piece::Meta(meta) => { out.push(5); write_meta(out, meta)?; }
            Piece::Repeat(inner, sep, op, slots) => {
                out.push(6);
                write_pieces(out, inner, to_local)?;
                match sep { Some((a, b)) => { out.push(1); write_u32(out, *a); write_u32(out, *b); } None => out.push(0) }
                out.push(OP_KINDS.iter().position(|&o| o == *op)? as u8);
                write_u32s(out, slots);
            }
        }
    }
    Some(())
}

fn read_pieces(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Vec<Piece>> {
    let count = cur.count()?;
    let mut pieces = Vec::with_capacity(count);
    for _ in 0..count {
        pieces.push(match cur.u8()? {
            0 => Piece::Token(cur.u32()?),
            1 => Piece::Group(cur.u32()?, read_pieces(cur, remap)?),
            2 => {
                let at = cur.u32()?;
                let frag = if cur.u8()? != 0 { Some(read_frag(cur)?) } else { None };
                let slot = read_opt_u32(cur)?;
                Piece::Var(at, frag, slot)
            }
            3 => {
                let mapped = match read_opt_u32(cur)? {
                    Some(k) => Some(*remap.get(k as usize).context("macro crate out of bounds")?),
                    None => None,
                };
                Piece::Crate(mapped)
            }
            4 => Piece::CratePath,
            5 => Piece::Meta(read_meta(cur)?),
            6 => {
                let inner = read_pieces(cur, remap)?;
                let sep = if cur.u8()? != 0 { Some((cur.u32()?, cur.u32()?)) } else { None };
                let op = OP_KINDS.get(cur.u8()? as usize).copied().context("invalid repeat op")?;
                let slots = read_u32s_map(cur, |v| v)?.into_boxed_slice();
                Piece::Repeat(inner, sep, op, slots)
            }
            _ => bail!("invalid piece tag"),
        });
    }
    Ok(pieces)
}

fn write_meta(out: &mut Vec<u8>, meta: &Meta) -> Option<()> {
    match meta {
        Meta::Ignore(slot) => { out.push(0); write_opt_u32(out, *slot); }
        Meta::Index(depth) => { out.push(1); write_usize(out, *depth)?; }
        Meta::Len(depth) => { out.push(2); write_usize(out, *depth)?; }
        Meta::Count(slot) => { out.push(3); write_opt_u32(out, *slot); }
        Meta::Concat(parts) => {
            out.push(4);
            write_usize(out, parts.len())?;
            for &(slot, token) in parts { write_opt_u32(out, slot); write_u32(out, token); }
        }
    }
    Some(())
}

fn read_meta(cur: &mut Cursor<'_>) -> Result<Meta> {
    Ok(match cur.u8()? {
        0 => Meta::Ignore(read_opt_u32(cur)?),
        1 => Meta::Index(cur.u32()? as usize),
        2 => Meta::Len(cur.u32()? as usize),
        3 => Meta::Count(read_opt_u32(cur)?),
        4 => {
            let count = cur.count()?;
            let mut parts = Vec::with_capacity(count);
            for _ in 0..count { parts.push((read_opt_u32(cur)?, cur.u32()?)); }
            Meta::Concat(parts)
        }
        _ => bail!("invalid meta tag"),
    })
}
