use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use rustc_lexer::{Base, DocStyle, LiteralKind, TokenKind};

use crate::graph::cfg::Cfg;
use crate::outline::item::{Attr, Import, Item, ItemKind, Items, Leaf, Vis};
use crate::outline::rules::{Frag, Meta, Op, Piece, Rule};
use crate::outline::source::{Mark, Origins, Token, opens};
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
    TokenKind::Semi, TokenKind::Comma, TokenKind::Dot, TokenKind::OpenParen, TokenKind::CloseParen,
    TokenKind::OpenBrace, TokenKind::CloseBrace, TokenKind::OpenBracket, TokenKind::CloseBracket,
    TokenKind::At, TokenKind::Pound, TokenKind::Tilde, TokenKind::Question, TokenKind::Colon, TokenKind::Dollar,
    TokenKind::Eq, TokenKind::Bang, TokenKind::Lt, TokenKind::Gt, TokenKind::Minus, TokenKind::And, TokenKind::Or,
    TokenKind::Plus, TokenKind::Star, TokenKind::Slash, TokenKind::Caret, TokenKind::Percent, TokenKind::Unknown, TokenKind::Eof,
];

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

fn write_opt_doc(out: &mut Vec<u8>, doc: Option<DocStyle>) {
    out.push(match doc {
        None => 0,
        Some(DocStyle::Outer) => 1,
        Some(DocStyle::Inner) => 2,
    });
}

fn read_opt_doc(cur: &mut Cursor<'_>) -> Result<Option<DocStyle>> {
    Ok(match cur.u8()? {
        0 => None,
        1 => Some(DocStyle::Outer),
        2 => Some(DocStyle::Inner),
        _ => bail!("invalid doc style tag"),
    })
}

fn write_base(out: &mut Vec<u8>, base: Base) {
    out.push(BASE_KINDS.iter().position(|&b| b == base).unwrap_or(0) as u8);
}

fn read_base(cur: &mut Cursor<'_>) -> Result<Base> {
    BASE_KINDS.get(cur.u8()? as usize).copied().context("invalid literal base")
}

fn write_opt_u8(out: &mut Vec<u8>, val: Option<u8>) {
    match val {
        Some(v) => { out.push(1); out.push(v); }
        None => out.push(0),
    }
}

fn read_opt_u8(cur: &mut Cursor<'_>) -> Result<Option<u8>> {
    Ok(if cur.u8()? != 0 { Some(cur.u8()?) } else { None })
}

fn write_literal_kind(out: &mut Vec<u8>, kind: LiteralKind) {
    match kind {
        LiteralKind::Int { base, empty_int } => { out.push(0); write_base(out, base); out.push(u8::from(empty_int)); }
        LiteralKind::Float { base, empty_exponent } => { out.push(1); write_base(out, base); out.push(u8::from(empty_exponent)); }
        LiteralKind::Char { terminated } => { out.push(2); out.push(u8::from(terminated)); }
        LiteralKind::Byte { terminated } => { out.push(3); out.push(u8::from(terminated)); }
        LiteralKind::Str { terminated } => { out.push(4); out.push(u8::from(terminated)); }
        LiteralKind::ByteStr { terminated } => { out.push(5); out.push(u8::from(terminated)); }
        LiteralKind::CStr { terminated } => { out.push(6); out.push(u8::from(terminated)); }
        LiteralKind::RawStr { n_hashes } => { out.push(7); write_opt_u8(out, n_hashes); }
        LiteralKind::RawByteStr { n_hashes } => { out.push(8); write_opt_u8(out, n_hashes); }
        LiteralKind::RawCStr { n_hashes } => { out.push(9); write_opt_u8(out, n_hashes); }
    }
}

fn read_literal_kind(cur: &mut Cursor<'_>) -> Result<LiteralKind> {
    Ok(match cur.u8()? {
        0 => LiteralKind::Int { base: read_base(cur)?, empty_int: cur.u8()? != 0 },
        1 => LiteralKind::Float { base: read_base(cur)?, empty_exponent: cur.u8()? != 0 },
        2 => LiteralKind::Char { terminated: cur.u8()? != 0 },
        3 => LiteralKind::Byte { terminated: cur.u8()? != 0 },
        4 => LiteralKind::Str { terminated: cur.u8()? != 0 },
        5 => LiteralKind::ByteStr { terminated: cur.u8()? != 0 },
        6 => LiteralKind::CStr { terminated: cur.u8()? != 0 },
        7 => LiteralKind::RawStr { n_hashes: read_opt_u8(cur)? },
        8 => LiteralKind::RawByteStr { n_hashes: read_opt_u8(cur)? },
        9 => LiteralKind::RawCStr { n_hashes: read_opt_u8(cur)? },
        _ => bail!("invalid literal kind tag"),
    })
}

pub(super) fn write_token(out: &mut Vec<u8>, token: Token) {
    if let Some(pos) = SIMPLE_TOKENS.iter().position(|&k| k == token.kind) {
        out.push(pos as u8);
    } else {
        match token.kind {
            TokenKind::LineComment { doc_style } => { out.push(38); write_opt_doc(out, doc_style); }
            TokenKind::BlockComment { doc_style, terminated } => { out.push(39); write_opt_doc(out, doc_style); out.push(u8::from(terminated)); }
            TokenKind::Frontmatter { has_invalid_preceding_whitespace, invalid_infostring } => {
                out.push(40);
                out.push(u8::from(has_invalid_preceding_whitespace) | (u8::from(invalid_infostring) << 1));
            }
            TokenKind::Lifetime { starts_with_number } => { out.push(41); out.push(u8::from(starts_with_number)); }
            TokenKind::Literal { kind, suffix_start } => { out.push(42); write_literal_kind(out, kind); write_u32(out, suffix_start); }
            _ => {}
        }
    }
    let (len, pair) = token.raw();
    write_u32(out, token.start);
    write_u32(out, len);
    if opens(token.kind) || closes(token.kind) {
        write_u32(out, pair);
    }
}

fn closes(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::CloseParen | TokenKind::CloseBrace | TokenKind::CloseBracket)
}

pub(super) fn read_token(cur: &mut Cursor<'_>) -> Result<Token> {
    let tag = cur.u8()?;
    let kind = if let Some(&simple) = SIMPLE_TOKENS.get(tag as usize) {
        simple
    } else {
        match tag {
            38 => TokenKind::LineComment { doc_style: read_opt_doc(cur)? },
            39 => TokenKind::BlockComment { doc_style: read_opt_doc(cur)?, terminated: cur.u8()? != 0 },
            40 => {
                let flags = cur.u8()?;
                TokenKind::Frontmatter { has_invalid_preceding_whitespace: flags & 1 != 0, invalid_infostring: flags & 2 != 0 }
            }
            41 => TokenKind::Lifetime { starts_with_number: cur.u8()? != 0 },
            42 => TokenKind::Literal { kind: read_literal_kind(cur)?, suffix_start: cur.u32()? },
            _ => bail!("invalid token kind tag"),
        }
    };
    let start = cur.u32()?;
    let len = cur.u32()?;
    let pair = if opens(kind) || closes(kind) { cur.u32()? } else { u32::MAX };
    Ok(Token::from_raw(kind, start, len, pair))
}

pub(super) fn write_frag(out: &mut Vec<u8>, frag: Frag) {
    out.push(FRAG_KINDS.iter().position(|&f| f == frag).unwrap_or(0) as u8);
}

pub(super) fn read_frag(cur: &mut Cursor<'_>) -> Result<Frag> {
    FRAG_KINDS.get(cur.u8()? as usize).copied().context("invalid fragment tag")
}

pub(super) fn write_mark(out: &mut Vec<u8>, mark: Mark, to_local: &[Option<u32>]) -> Option<()> {
    match mark {
        Mark::Crate(k) => { out.push(0); write_u32(out, (*to_local.get(k as usize)?)?); }
        Mark::Home { krate, file, item } => {
            out.push(1);
            write_u32(out, (*to_local.get(krate as usize)?)?);
            write_u32(out, file);
            write_u32(out, item);
        }
        Mark::Scope { id, parent } => { out.push(2); write_u32(out, id); write_u32(out, parent); }
    }
    Some(())
}

pub(super) fn read_mark(cur: &mut Cursor<'_>, remap: &[u32]) -> Result<Mark> {
    const CACHED_SCOPE_BIT: u32 = 1 << 31;
    let map_scope = |s: u32| if s == 0 { 0 } else { s | CACHED_SCOPE_BIT };
    Ok(match cur.u8()? {
        0 => Mark::Crate(*remap.get(cur.u32()? as usize).context("mark crate out of bounds")?),
        1 => Mark::Home {
            krate: *remap.get(cur.u32()? as usize).context("mark home crate out of bounds")?,
            file: cur.u32()?,
            item: cur.u32()?,
        },
        2 => Mark::Scope { id: map_scope(cur.u32()?), parent: map_scope(cur.u32()?) },
        _ => bail!("invalid mark tag"),
    })
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
            let mut pairs = Vec::with_capacity(count);
            for _ in 0..count { pairs.push((cur.u32()?, cur.u32()?)); }
            Origins::Copied(pairs)
        }
        _ => bail!("invalid origins tag"),
    })
}

pub(super) fn write_vis(out: &mut Vec<u8>, vis: Vis) {
    match vis {
        Vis::Private => out.push(0),
        Vis::Pub => out.push(1),
        Vis::Crate => out.push(2),
        Vis::Super => out.push(3),
        Vis::SelfMod => out.push(4),
        Vis::In(at) => { out.push(5); write_u32(out, at); }
    }
}

pub(super) fn read_vis(cur: &mut Cursor<'_>) -> Result<Vis> {
    Ok(match cur.u8()? {
        0 => Vis::Private,
        1 => Vis::Pub,
        2 => Vis::Crate,
        3 => Vis::Super,
        4 => Vis::SelfMod,
        5 => Vis::In(cur.u32()?),
        _ => bail!("invalid vis tag"),
    })
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
    for item in &items.items {
        out.push(ITEM_KINDS.iter().position(|&k| k == item.kind)? as u8);
        write_opt_u32(out, item.name);
        write_vis(out, item.vis);
        write_attrs(out, &item.attrs)?;
        write_u32(out, item.lead);
        write_u32(out, item.start);
        write_u32(out, item.end);
        write_opt_u32(out, item.body);
        write_opt_u32(out, item.parent);
    }
    write_usize(out, items.imports.len())?;
    for import in &items.imports {
        write_u32(out, import.item);
        out.push(u8::from(import.absolute));
        write_u32s(out, &import.path);
        match import.leaf {
            Leaf::Name(alias) => { out.push(0); write_opt_u32(out, alias); }
            Leaf::Underscore => out.push(1),
            Leaf::Glob => out.push(2),
        }
    }
    write_usize(out, items.marks.len())?;
    for &(at, bits) in &items.marks {
        write_u32(out, at);
        out.push(bits);
    }
    Some(())
}

pub(super) fn read_items(cur: &mut Cursor<'_>) -> Result<Items> {
    let inner = read_attrs(cur)?;
    let item_count = cur.count()?;
    let mut items = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let kind = ITEM_KINDS.get(cur.u8()? as usize).copied().context("invalid item kind tag")?;
        items.push(Item {
            kind,
            name: read_opt_u32(cur)?,
            vis: read_vis(cur)?,
            attrs: read_attrs(cur)?,
            lead: cur.u32()?,
            start: cur.u32()?,
            end: cur.u32()?,
            body: read_opt_u32(cur)?,
            parent: read_opt_u32(cur)?,
        });
    }
    let import_count = cur.count()?;
    let mut imports = Vec::with_capacity(import_count);
    for _ in 0..import_count {
        let item = cur.u32()?;
        let absolute = cur.u8()? != 0;
        let path = cur.u32s()?;
        let leaf = match cur.u8()? {
            0 => Leaf::Name(read_opt_u32(cur)?),
            1 => Leaf::Underscore,
            2 => Leaf::Glob,
            _ => bail!("invalid import leaf tag"),
        };
        imports.push(Import { item, absolute, path, leaf });
    }
    let mark_count = cur.count()?;
    let mut marks = Vec::with_capacity(mark_count);
    for _ in 0..mark_count {
        marks.push((cur.u32()?, cur.u8()?));
    }
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
                let slots = cur.u32s()?.into_boxed_slice();
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
