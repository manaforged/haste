mod items;
mod rules;

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use rustc_lexer::{Base, DocStyle, LiteralKind, TokenKind};

use crate::outline::item::Vis;
use crate::outline::rules::Frag;
use crate::outline::source::{Mark, Origins, Token};
use crate::sema::summary::codec::write_u32;
pub(super) use crate::sema::summary::codec::{read_str, write_str};
use crate::substrate::{bytes_to_path, path_to_bytes};

pub(super) use items::{read_items, write_items};
pub(super) use rules::{read_rules, write_rules};

const BASE_KINDS: [Base; 4] = [Base::Binary, Base::Octal, Base::Decimal, Base::Hexadecimal];
const FRAG_KINDS: [Frag; 14] = [
    Frag::Ident,
    Frag::Path,
    Frag::Expr,
    Frag::Ty,
    Frag::Pat,
    Frag::PatParam,
    Frag::Stmt,
    Frag::Block,
    Frag::Item,
    Frag::Meta,
    Frag::Tt,
    Frag::Literal,
    Frag::Lifetime,
    Frag::Vis,
];
const SIMPLE_TOKENS: [TokenKind; 38] = [
    TokenKind::Whitespace,
    TokenKind::Ident,
    TokenKind::InvalidIdent,
    TokenKind::RawIdent,
    TokenKind::ForcedKeywordIdent,
    TokenKind::UnknownPrefix,
    TokenKind::UnknownPrefixLifetime,
    TokenKind::RawLifetime,
    TokenKind::GuardedStrPrefix,
    TokenKind::Semi,
    TokenKind::Comma,
    TokenKind::Dot,
    TokenKind::At,
    TokenKind::Pound,
    TokenKind::Tilde,
    TokenKind::Question,
    TokenKind::Colon,
    TokenKind::Dollar,
    TokenKind::Eq,
    TokenKind::Bang,
    TokenKind::Lt,
    TokenKind::Gt,
    TokenKind::Minus,
    TokenKind::And,
    TokenKind::Or,
    TokenKind::Plus,
    TokenKind::Star,
    TokenKind::Slash,
    TokenKind::Caret,
    TokenKind::Percent,
    TokenKind::Unknown,
    TokenKind::Eof,
    TokenKind::OpenParen,
    TokenKind::CloseParen,
    TokenKind::OpenBrace,
    TokenKind::CloseBrace,
    TokenKind::OpenBracket,
    TokenKind::CloseBracket,
];

pub(super) use crate::substrate::read_u32 as u32_le;

pub(super) struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    #[inline(always)]
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self(data)
    }

    #[inline(always)]
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub(super) fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let (head, tail) = self.0.split_first_chunk::<N>().context("unexpected eof")?;
        self.0 = tail;
        Ok(*head)
    }

    #[inline(always)]
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(n).context("unexpected eof")?;
        self.0 = tail;
        Ok(head)
    }

    #[inline(always)]
    pub(super) fn u8(&mut self) -> Result<u8> {
        let (&b, tail) = self.0.split_first().context("unexpected eof")?;
        self.0 = tail;
        Ok(b)
    }

    #[inline(always)]
    pub(super) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take_array::<4>()?))
    }

    #[inline(always)]
    pub(super) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take_array::<8>()?))
    }

    #[inline(always)]
    pub(super) fn opt_u32(&mut self) -> Result<Option<u32>> {
        let v = self.u32()?;
        Ok((v != u32::MAX).then_some(v))
    }

    #[inline(always)]
    pub(super) fn count(&mut self) -> Result<usize> {
        let n = self.u32()? as usize;
        if n > self.0.len() {
            bail!("count exceeds remaining bytes");
        }
        Ok(n)
    }

    #[inline(always)]
    pub(super) fn str(&mut self) -> Result<&'a str> {
        let len = self.count()?;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).context("invalid utf-8")
    }

    #[inline(always)]
    pub(super) fn path(&mut self) -> Result<PathBuf> {
        let len = self.count()?;
        if len == 0 {
            return Ok(PathBuf::new());
        }
        bytes_to_path(self.take(len)?).context("invalid path bytes")
    }

    #[inline(always)]
    pub(super) fn paths(&mut self) -> Result<Vec<PathBuf>> {
        let count = self.count()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut paths = Vec::with_capacity(count);
        for _ in 0..count {
            paths.push(self.path()?);
        }
        Ok(paths)
    }

    #[inline(always)]
    pub(super) fn u32s_map<T>(&mut self, mut f: impl FnMut(u32) -> T) -> Result<Vec<T>> {
        let count = self.count()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let byte_len = count.checked_mul(4).context("u32 slice overflow")?;
        let bytes = self.take(byte_len)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect())
    }
}

pub(super) fn write_paths(out: &mut Vec<u8>, paths: &[PathBuf]) -> Option<()> {
    write_usize(out, paths.len())?;
    for path in paths {
        write_path(out, path)?;
    }
    Some(())
}

pub(super) fn write_path(out: &mut Vec<u8>, path: &std::path::Path) -> Option<()> {
    let bytes = path_to_bytes(path)?;
    write_u32(out, u32::try_from(bytes.len()).ok()?);
    out.extend_from_slice(&bytes);
    Some(())
}

pub(super) fn write_opt_u32(out: &mut Vec<u8>, val: Option<u32>) {
    write_u32(out, val.unwrap_or(u32::MAX));
}

pub(super) fn write_usize(out: &mut Vec<u8>, val: usize) -> Option<()> {
    write_u32(out, u32::try_from(val).ok()?);
    Some(())
}

#[inline(always)]
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

#[inline(always)]
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

#[inline]
fn encode_punct(kind: TokenKind) -> Option<u16> {
    Some(match kind {
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
        _ => return None,
    })
}

#[inline]
fn encode_literal(kind: LiteralKind) -> (u16, u16) {
    match kind {
        LiteralKind::Int { base, empty_int } => (42, base_bits(base) | (u16::from(empty_int) << 2)),
        LiteralKind::Float {
            base,
            empty_exponent,
        } => (43, base_bits(base) | (u16::from(empty_exponent) << 2)),
        LiteralKind::Char { terminated } => (44, u16::from(terminated)),
        LiteralKind::Byte { terminated } => (45, u16::from(terminated)),
        LiteralKind::Str { terminated } => (46, u16::from(terminated)),
        LiteralKind::ByteStr { terminated } => (47, u16::from(terminated)),
        LiteralKind::CStr { terminated } => (48, u16::from(terminated)),
        LiteralKind::RawStr { n_hashes } => (49, n_hashes.map_or(256, u16::from)),
        LiteralKind::RawByteStr { n_hashes } => (50, n_hashes.map_or(256, u16::from)),
        LiteralKind::RawCStr { n_hashes } => (51, n_hashes.map_or(256, u16::from)),
    }
}

#[inline]
fn encode_token(kind: TokenKind, pair: u32) -> Option<(u16, u32)> {
    if let Some(tag) = encode_punct(kind) {
        return Some((tag, pair));
    }
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
        TokenKind::Semi
        | TokenKind::Comma
        | TokenKind::Dot
        | TokenKind::At
        | TokenKind::Pound
        | TokenKind::Tilde
        | TokenKind::Question
        | TokenKind::Colon
        | TokenKind::Dollar
        | TokenKind::Eq
        | TokenKind::Bang
        | TokenKind::Lt
        | TokenKind::Gt
        | TokenKind::Minus
        | TokenKind::And
        | TokenKind::Or
        | TokenKind::Plus
        | TokenKind::Star
        | TokenKind::Slash
        | TokenKind::Caret
        | TokenKind::Percent => unreachable!(),
        TokenKind::Unknown => 30,
        TokenKind::Eof => 31,
        TokenKind::OpenParen => 32,
        TokenKind::CloseParen => 33,
        TokenKind::OpenBrace => 34,
        TokenKind::CloseBrace => 35,
        TokenKind::OpenBracket => 36,
        TokenKind::CloseBracket => 37,
        TokenKind::LineComment { doc_style } => {
            return Some((38 | (doc_bits(doc_style) << 6), u32::MAX));
        }
        TokenKind::BlockComment {
            doc_style,
            terminated,
        } => {
            return Some((
                39 | ((doc_bits(doc_style) | (u16::from(terminated) << 2)) << 6),
                u32::MAX,
            ));
        }
        TokenKind::Frontmatter {
            has_invalid_preceding_whitespace,
            invalid_infostring,
        } => {
            let flags =
                u16::from(has_invalid_preceding_whitespace) | (u16::from(invalid_infostring) << 1);
            return Some((40 | (flags << 6), u32::MAX));
        }
        TokenKind::Lifetime { starts_with_number } => {
            return Some((41 | (u16::from(starts_with_number) << 6), u32::MAX));
        }
        TokenKind::Literal { kind, suffix_start } => {
            let (tag, payload) = encode_literal(kind);
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
        38 => TokenKind::LineComment {
            doc_style: doc_from_bits(payload)?,
        },
        39 => TokenKind::BlockComment {
            doc_style: doc_from_bits(payload & 3)?,
            terminated: payload & 4 != 0,
        },
        40 => TokenKind::Frontmatter {
            has_invalid_preceding_whitespace: payload & 1 != 0,
            invalid_infostring: payload & 2 != 0,
        },
        41 => TokenKind::Lifetime {
            starts_with_number: payload != 0,
        },
        42..=51 => {
            let kind = match tag {
                42 => LiteralKind::Int {
                    base: BASE_KINDS[(payload & 3) as usize],
                    empty_int: payload & 4 != 0,
                },
                43 => LiteralKind::Float {
                    base: BASE_KINDS[(payload & 3) as usize],
                    empty_exponent: payload & 4 != 0,
                },
                44 => LiteralKind::Char {
                    terminated: payload != 0,
                },
                45 => LiteralKind::Byte {
                    terminated: payload != 0,
                },
                46 => LiteralKind::Str {
                    terminated: payload != 0,
                },
                47 => LiteralKind::ByteStr {
                    terminated: payload != 0,
                },
                48 => LiteralKind::CStr {
                    terminated: payload != 0,
                },
                49 => LiteralKind::RawStr {
                    n_hashes: raw_hashes(payload)?,
                },
                50 => LiteralKind::RawByteStr {
                    n_hashes: raw_hashes(payload)?,
                },
                51 => LiteralKind::RawCStr {
                    n_hashes: raw_hashes(payload)?,
                },
                _ => unreachable!(),
            };
            TokenKind::Literal {
                kind,
                suffix_start: extra,
            }
        }
        _ => bail!("invalid token kind tag"),
    })
}

#[inline]
fn write_varint(out: &mut Vec<u8>, mut val: u32) {
    while val >= 0x80 {
        out.push((val as u8) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

#[inline(always)]
fn read_varint(reader: &mut Reader<'_>) -> Result<u32> {
    read_varint_slice(&mut reader.0)
}

pub(super) fn write_compact_tokens(out: &mut Vec<u8>, tokens: &[Token]) -> Option<()> {
    write_usize(out, tokens.len())?;
    let len_at = out.len();
    write_u32(out, 0);
    let body_start = out.len();
    out.reserve(tokens.len() * 3);
    let mut prev_end = 0u32;
    let mut prev_joint = true;
    for (idx, &token) in tokens.iter().enumerate() {
        let (raw_len, pair) = token.raw();
        let joint = token.joint();
        let len = raw_len & !(1u32 << 31);
        let (code, extra) = encode_token(token.kind, pair)?;
        let tag = (code & 0x3f) as u8;
        let expected_start = prev_end.wrapping_add(u32::from(!prev_joint && idx > 0));
        let non_default_start = token.start != expected_start;
        out.push(tag | (u8::from(joint) << 6) | (u8::from(non_default_start) << 7));
        if tag >= SIMPLE_TOKENS.len() as u8 {
            write_varint(out, u32::from(code >> 6));
            if tag >= 42 {
                write_varint(out, extra);
            }
        }
        if non_default_start {
            write_varint(out, token.start);
        }
        write_varint(out, len);
        prev_end = token.start.wrapping_add(len);
        prev_joint = joint;
    }
    let byte_len = u32::try_from(out.len() - body_start).ok()?;
    out[len_at..len_at + 4].copy_from_slice(&byte_len.to_le_bytes());
    Some(())
}

/// Decodes `count` compact variable-length tokens from `bytes` and restores delimiter pairs.
pub(crate) fn decode_compact_tokens(bytes: &[u8], count: usize) -> Result<Vec<Token>> {
    let mut reader = Reader::new(bytes);
    let mut tokens = Vec::with_capacity(count);
    let mut prev_end = 0u32;
    let mut prev_joint = true;
    for idx in 0..count {
        let header = reader.u8()?;
        let tag = (header & 0x3f) as usize;
        let joint = header & 0x40 != 0;
        let non_default_start = header & 0x80 != 0;
        let kind = if tag < SIMPLE_TOKENS.len() {
            SIMPLE_TOKENS[tag]
        } else {
            let payload =
                u16::try_from(read_varint(&mut reader)?).context("token payload overflow")?;
            let extra = if tag >= 42 {
                read_varint(&mut reader)?
            } else {
                u32::MAX
            };
            decode_complex_token(tag, payload, extra)?
        };
        let start = if non_default_start {
            read_varint(&mut reader)?
        } else {
            prev_end.wrapping_add(u32::from(!prev_joint && idx > 0))
        };
        let len = read_varint(&mut reader)?;
        let raw_len = len | (u32::from(joint) << 31);
        tokens.push(Token::from_raw(kind, start, raw_len, u32::MAX));
        prev_end = start.wrapping_add(len);
        prev_joint = joint;
    }
    if !reader.is_empty() {
        bail!("trailing bytes in compact token stream");
    }
    crate::outline::source::pair_up_tokens(&mut tokens, std::path::Path::new("<sysroot>"))?;
    Ok(tokens)
}

pub(super) fn frag_tag(frag: Frag) -> u8 {
    FRAG_KINDS.iter().position(|&f| f == frag).unwrap_or(0) as u8
}

#[inline(always)]
pub(super) fn frag_from_tag(tag: u8) -> Result<Frag> {
    FRAG_KINDS
        .get(tag as usize)
        .copied()
        .context("invalid fragment tag")
}

pub(super) fn write_frag(out: &mut Vec<u8>, frag: Frag) {
    out.push(frag_tag(frag));
}

#[inline(always)]
pub(super) fn read_frag(reader: &mut Reader<'_>) -> Result<Frag> {
    frag_from_tag(reader.u8()?)
}

pub(in crate::substrate) fn write_marks(
    out: &mut Vec<u8>,
    marks: &[(u32, Mark)],
    to_local: &[Option<u32>],
) -> Option<()> {
    write_usize(out, marks.len())?;
    let len_at = out.len();
    write_u32(out, 0);
    let body_start = out.len();
    out.reserve(marks.len() * 3);
    let mut prev_at = 0u32;
    let mut prev_mark: Option<Mark> = None;
    for &(at, mark) in marks {
        let delta = at.wrapping_sub(prev_at);
        let mapped = match mark {
            Mark::Crate(k) => Mark::Crate((*to_local.get(k as usize)?)?),
            Mark::Home { krate, file, item } => Mark::Home {
                krate: (*to_local.get(krate as usize)?)?,
                file,
                item,
            },
            Mark::Scope { id, parent } => Mark::Scope { id, parent },
        };
        if prev_mark == Some(mapped) {
            out.push(3);
            write_varint(out, delta);
        } else {
            match mapped {
                Mark::Crate(k) => {
                    out.push(0);
                    write_varint(out, delta);
                    write_varint(out, k);
                }
                Mark::Home { krate, file, item } => {
                    out.push(1);
                    write_varint(out, delta);
                    write_varint(out, krate);
                    write_varint(out, file);
                    write_varint(out, item);
                }
                Mark::Scope { id, parent } => {
                    out.push(2);
                    write_varint(out, delta);
                    write_varint(out, id);
                    write_varint(out, parent);
                }
            }
            prev_mark = Some(mapped);
        }
        prev_at = at;
    }
    let byte_len = u32::try_from(out.len() - body_start).ok()?;
    out[len_at..len_at + 4].copy_from_slice(&byte_len.to_le_bytes());
    Some(())
}

#[inline(always)]
fn read_varint_slice(data: &mut &[u8]) -> Result<u32> {
    let (&b0, mut rest) = data.split_first().context("unexpected eof")?;
    if b0 < 0x80 {
        *data = rest;
        return Ok(u32::from(b0));
    }
    let mut val = u32::from(b0 & 0x7f);
    let mut shift = 7u32;
    loop {
        let (&byte, next) = rest.split_first().context("unexpected eof")?;
        rest = next;
        val |= u32::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            *data = rest;
            return Ok(val);
        }
        shift += 7;
        if shift >= 35 {
            bail!("varint overflow");
        }
    }
}

pub(crate) fn decode_marks_body(
    count: usize,
    mut rem: &[u8],
    remap: &[u32],
    scope_bit: u32,
) -> Result<Vec<(u32, Mark)>> {
    const SCOPE_BITS_MASK: u32 = (1 << 31) | (1 << 30) | (1 << 29);
    let map_scope = |s: u32| {
        if s == 0 || (s & SCOPE_BITS_MASK) != 0 {
            s
        } else {
            s | scope_bit
        }
    };
    if count == 0 {
        if !rem.is_empty() {
            bail!("trailing bytes in sysroot marks");
        }
        return Ok(Vec::new());
    }
    let mut marks = Vec::with_capacity(count);
    let mut prev_at = 0u32;
    let mut prev_mark: Option<Mark> = None;
    for _ in 0..count {
        let (&tag, next) = rem.split_first().context("unexpected eof")?;
        rem = next;
        let at = prev_at.wrapping_add(read_varint_slice(&mut rem)?);
        let mark = match tag {
            3 => prev_mark.context("repeat mark without previous mark")?,
            0 => Mark::Crate(
                *remap
                    .get(read_varint_slice(&mut rem)? as usize)
                    .context("mark crate out of bounds")?,
            ),
            1 => Mark::Home {
                krate: *remap
                    .get(read_varint_slice(&mut rem)? as usize)
                    .context("mark home crate out of bounds")?,
                file: read_varint_slice(&mut rem)?,
                item: read_varint_slice(&mut rem)?,
            },
            2 => Mark::Scope {
                id: map_scope(read_varint_slice(&mut rem)?),
                parent: map_scope(read_varint_slice(&mut rem)?),
            },
            _ => bail!("invalid mark tag"),
        };
        prev_at = at;
        prev_mark = Some(mark);
        marks.push((at, mark));
    }
    if !rem.is_empty() {
        bail!("trailing bytes in sysroot marks");
    }
    Ok(marks)
}

#[cfg(test)]
pub(super) fn read_marks_reader(
    reader: &mut Reader<'_>,
    remap: &[u32],
    scope_bit: u32,
) -> Result<Vec<(u32, Mark)>> {
    let count = reader.count()?;
    let byte_len = reader.count()?;
    let data = reader.take(byte_len)?;
    decode_marks_body(count, data, remap, scope_bit)
}

#[cfg(test)]
pub(in crate::substrate) fn read_marks(
    data: &[u8],
    remap: &[u32],
    scope_bit: u32,
) -> Result<Vec<(u32, Mark)>> {
    let mut reader = Reader::new(data);
    read_marks_reader(&mut reader, remap, scope_bit)
}

pub(super) fn write_origins(out: &mut Vec<u8>, origins: &Origins) -> Option<()> {
    match origins {
        Origins::Read => out.push(0),
        Origins::Copied(pairs) => {
            out.push(1);
            write_usize(out, pairs.len())?;
            for &(a, b) in pairs {
                write_u32(out, a);
                write_u32(out, b);
            }
        }
    }
    Some(())
}

pub(super) fn read_origins(reader: &mut Reader<'_>) -> Result<Origins> {
    Ok(match reader.u8()? {
        0 => Origins::Read,
        1 => {
            let count = reader.count()?;
            let bytes = reader.take(count.checked_mul(8).context("origins overflow")?)?;
            Origins::Copied(
                bytes
                    .chunks_exact(8)
                    .map(|c| {
                        (
                            u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                            u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                        )
                    })
                    .collect(),
            )
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

#[inline(always)]
pub(super) fn read_vis(reader: &mut Reader<'_>) -> Result<Vis> {
    let tag = reader.u8()?;
    let at = if tag == 5 { reader.u32()? } else { 0 };
    vis_from_tag(tag, at)
}
