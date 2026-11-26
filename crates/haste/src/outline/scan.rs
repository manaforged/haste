use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;
use rustc_lexer::TokenKind;

use super::attr;
use super::item::{Attr, Import, Item, ItemKind, Items, Leaf, Vis};
use super::source::{Source, opens};
use crate::diag::{Diagnostic, DiagnosticKind, Span};

struct Scanner<'a> {
    src: &'a Source,
    out: Items,
    calls: Option<&'a HashMap<u32, u32>>,
}

struct Head {
    kind: ItemKind,
    name: Option<u32>,
    body: Option<u32>,
    end: u32,
}

fn is_name(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Ident | TokenKind::RawIdent)
}

const QUALIFIERS: [&str; 7] = ["default", "async", "unsafe", "safe", "auto", "gen", "final"];
const BODIED: [(&str, ItemKind); 4] = [
    ("mod", ItemKind::Mod),
    ("fn", ItemKind::Fn),
    ("struct", ItemKind::Struct),
    ("enum", ItemKind::Enum),
];
const AFTER_CONST: [&str; 7] = ["fn", "async", "unsafe", "extern", "safe", "impl", "trait"];

/// Scans the items of a whole file.
pub fn scan(src: &Source) -> Result<Items> {
    let mut scanner = Scanner {
        src,
        out: Items::default(),
        calls: None,
    };
    let mut inner = Vec::new();
    let at = attr::attrs(src, 0, true, &mut inner)?;
    scanner.out.inner = inner;
    scanner.items(at, src.len(), None)?;
    Ok(scanner.out)
}

/// Adds to `items` the items declared in a body, each starting at its token under the item whose body declares it, reading a module's macro calls through their expansions in `calls`.
pub fn body_items(
    src: &Source,
    starts: impl IntoIterator<Item = (u32, Option<u32>)>,
    (items, calls): (Items, Option<&HashMap<u32, u32>>),
) -> Result<Items> {
    let mut scanner = Scanner {
        src,
        out: items,
        calls,
    };
    for (at, owner) in starts {
        scanner.item(at, owner)?;
    }
    Ok(scanner.out)
}

/// The token after the item that starts at `at`.
pub fn item_end(src: &Source, at: u32) -> Result<u32> {
    Scanner {
        src,
        out: Items::default(),
        calls: None,
    }
    .item(at, None)
}

impl Scanner<'_> {
    fn word(&self, at: u32) -> &str {
        if is_name(self.src.kind(at)) {
            self.src.text(at)
        } else {
            ""
        }
    }

    fn is_path_sep(&self, at: u32) -> bool {
        self.src.kind(at) == TokenKind::Colon
            && self.src.kind(at + 1) == TokenKind::Colon
            && self.src.joint(at)
    }

    fn items(&mut self, mut at: u32, to: u32, parent: Option<u32>) -> Result<()> {
        while at < to {
            at = self.item(at, parent)?;
        }
        Ok(())
    }

    fn item(&mut self, at: u32, parent: Option<u32>) -> Result<u32> {
        if self.src.kind(at) == TokenKind::Semi {
            return Ok(at + 1);
        }
        let mut attrs = Vec::new();
        let lead = at;
        let at = attr::attrs(self.src, at, false, &mut attrs)?;
        let (vis, at) = self.vis(at);
        let start = at;
        let head = self.head(self.qualifiers(at))?;
        let index = self.out.items.len() as u32;
        if let Some(open) = head.body {
            self.inner_attrs(open, head.kind, &mut attrs)?;
        }
        let bits = attr::marks(self.src, &attrs);
        if bits != 0 {
            self.out.marks.push((index, bits));
        }
        self.out.items.push(Item {
            kind: head.kind,
            name: head.name,
            vis,
            attrs,
            lead,
            start,
            end: head.end,
            body: head.body,
            parent,
        });
        self.children(index, &head)?;
        if let (ItemKind::MacroCall, Some(_), Some(open)) = (head.kind, parent, head.body)
            && let Some(&block) = self.calls.and_then(|calls| calls.get(&open))
        {
            self.items(block + 1, self.src.pair(block), parent)?;
        }
        if let (ItemKind::Mod, None) = (head.kind, head.body)
            && let Some(&block) = self.calls.and_then(|calls| calls.get(&(head.end - 1)))
        {
            self.items(block + 1, self.src.pair(block), Some(index))?;
        }
        Ok(head.end)
    }

    fn inner_attrs(&mut self, open: u32, kind: ItemKind, into: &mut Vec<Attr>) -> Result<()> {
        if matches!(
            kind,
            ItemKind::Mod | ItemKind::Impl | ItemKind::Trait | ItemKind::ExternBlock
        ) {
            attr::attrs(self.src, open + 1, true, into)?;
        }
        Ok(())
    }

    fn children(&mut self, index: u32, head: &Head) -> Result<()> {
        let Some(open) = head.body else { return Ok(()) };
        if !matches!(
            head.kind,
            ItemKind::Mod | ItemKind::Impl | ItemKind::Trait | ItemKind::ExternBlock
        ) {
            return Ok(());
        }
        let mut skipped = Vec::new();
        let first = attr::attrs(self.src, open + 1, true, &mut skipped)?;
        self.items(first, self.src.pair(open), Some(index))
    }

    fn vis(&self, at: u32) -> (Vis, u32) {
        match self.word(at) {
            "pub" if self.src.kind(at + 1) == TokenKind::OpenParen => self.restricted(at + 1),
            "pub" => (Vis::Pub, at + 1),
            "crate" if !self.is_path_sep(at + 1) => (Vis::Crate, at + 1),
            _ => (Vis::Private, at),
        }
    }

    fn restricted(&self, open: u32) -> (Vis, u32) {
        let close = self.src.pair(open);
        let vis = match self.word(open + 1) {
            "crate" if close == open + 2 => Vis::Crate,
            "super" if close == open + 2 => Vis::Super,
            "self" if close == open + 2 => Vis::SelfMod,
            "in" => Vis::In(open + 2),
            _ => return (Vis::Pub, open),
        };
        (vis, close + 1)
    }

    fn qualifiers(&self, mut at: u32) -> u32 {
        loop {
            let word = self.word(at);
            let next = self.word(at + 1);
            let qualifier = QUALIFIERS.contains(&word)
                && is_name(self.src.kind(at + 1))
                && !matches!(next, "" | "!");
            let constant = word == "const" && AFTER_CONST.contains(&next);
            let restricted = word == "impl"
                && self.src.kind(at + 1) == TokenKind::OpenParen
                && is_name(self.src.kind(self.src.pair(at + 1) + 1));
            if restricted {
                at = self.src.pair(at + 1) + 1;
                continue;
            }
            if !(qualifier || constant) {
                return at;
            }
            at += 1;
        }
    }

    fn head(&mut self, at: u32) -> Result<Head> {
        let word = self.word(at);
        if let Some(&(_, kind)) = BODIED.iter().find(|(name, _)| *name == word) {
            let (body, end) = self.to_body(at + 2);
            return Ok(Head {
                kind,
                name: Some(at + 1),
                body,
                end,
            });
        }
        let next = self.src.kind(at + 1);
        match word {
            "use" => self.use_item(at),
            "extern" => self.extern_item(at),
            "union" if is_name(next) => self.bodied(ItemKind::Union, at),
            "trait" => self.trait_item(at),
            "impl" => {
                let (body, end) = self.to_body(at + 1);
                Ok(Head {
                    kind: ItemKind::Impl,
                    name: None,
                    body,
                    end,
                })
            }
            "type" | "const" => {
                let kind = if word == "type" {
                    ItemKind::Type
                } else {
                    ItemKind::Const
                };
                Ok(Head {
                    kind,
                    name: Some(at + 1),
                    body: None,
                    end: self.to_semi(at + 2),
                })
            }
            "static" => self.static_item(at),
            "macro_rules" if next == TokenKind::Bang => self.macro_rules(at),
            "macro" if is_name(next) => self.macro_item(at),
            _ => self.macro_call(at),
        }
    }

    fn bodied(&self, kind: ItemKind, at: u32) -> Result<Head> {
        let (body, end) = self.to_body(at + 2);
        Ok(Head {
            kind,
            name: Some(at + 1),
            body,
            end,
        })
    }

    fn is_arrow(&self, at: u32) -> bool {
        at.checked_sub(1).is_some_and(|before| {
            matches!(self.src.kind(before), TokenKind::Minus | TokenKind::Eq)
                && self.src.joint(before)
        })
    }

    fn to_body(&self, mut at: u32) -> (Option<u32>, u32) {
        let mut angles = 0_u32;
        loop {
            match self.src.kind(at) {
                TokenKind::OpenBrace if angles == 0 => return (Some(at), self.src.pair(at) + 1),
                TokenKind::Semi | TokenKind::Eof => return (None, at + 1),
                kind if opens(kind) => at = self.src.pair(at),
                TokenKind::Lt => angles += 1,
                TokenKind::Gt if !self.is_arrow(at) => angles = angles.saturating_sub(1),
                _ => {}
            }
            at += 1;
        }
    }

    fn to_semi(&self, mut at: u32) -> u32 {
        loop {
            match self.src.kind(at) {
                TokenKind::Semi | TokenKind::Eof => return at + 1,
                kind if opens(kind) => at = self.src.pair(at) + 1,
                _ => at += 1,
            }
        }
    }

    fn trait_item(&self, at: u32) -> Result<Head> {
        let (body, end) = self.to_body(at + 2);
        let alias =
            body.is_none() && (at + 2..end).any(|token| self.src.kind(token) == TokenKind::Eq);
        let kind = if alias {
            ItemKind::TraitAlias
        } else {
            ItemKind::Trait
        };
        Ok(Head {
            kind,
            name: Some(at + 1),
            body,
            end,
        })
    }

    fn static_item(&self, at: u32) -> Result<Head> {
        let name = if self.word(at + 1) == "mut" {
            at + 2
        } else {
            at + 1
        };
        Ok(Head {
            kind: ItemKind::Static,
            name: Some(name),
            body: None,
            end: self.to_semi(name + 1),
        })
    }

    fn extern_item(&mut self, at: u32) -> Result<Head> {
        if self.word(at + 1) == "crate" {
            let name = if self.word(at + 3) == "as" {
                at + 4
            } else {
                at + 2
            };
            return Ok(Head {
                kind: ItemKind::ExternCrate,
                name: Some(name),
                body: None,
                end: self.to_semi(at + 2),
            });
        }
        let after = if matches!(self.src.kind(at + 1), TokenKind::Literal { .. }) {
            at + 2
        } else {
            at + 1
        };
        if self.src.kind(after) == TokenKind::OpenBrace {
            return Ok(Head {
                kind: ItemKind::ExternBlock,
                name: None,
                body: Some(after),
                end: self.src.pair(after) + 1,
            });
        }
        self.head(self.qualifiers(after))
    }

    fn delimited_end(&self, open: u32) -> Result<u32> {
        if !opens(self.src.kind(open)) {
            bail!(
                "{}: expected `(`, `[` or `{{` at byte {} after a macro name",
                self.src.path.display(),
                self.src.byte(open)
            );
        }
        let close = self.src.pair(open);
        let braced = self.src.kind(open) == TokenKind::OpenBrace;
        Ok(if !braced && self.src.kind(close + 1) == TokenKind::Semi {
            close + 2
        } else {
            close + 1
        })
    }

    fn macro_rules(&self, at: u32) -> Result<Head> {
        let open = at + 3;
        Ok(Head {
            kind: ItemKind::MacroRules,
            name: Some(at + 2),
            body: Some(open),
            end: self.delimited_end(open)?,
        })
    }

    fn macro_item(&self, at: u32) -> Result<Head> {
        let mut last = at + 2;
        if self.src.kind(last) == TokenKind::OpenParen
            && self.src.kind(self.src.pair(last) + 1) == TokenKind::OpenBrace
        {
            last = self.src.pair(last) + 1;
        }
        Ok(Head {
            kind: ItemKind::Macro,
            name: Some(at + 1),
            body: Some(at + 2),
            end: self.delimited_end(last)?,
        })
    }

    fn macro_call(&self, start: u32) -> Result<Head> {
        let mut at = start;
        if self.src.kind(at) == TokenKind::Dollar && self.word(at + 1) == "crate" {
            at += 1;
        }
        if self.is_path_sep(at) {
            at += 2;
        }
        while is_name(self.src.kind(at)) && self.is_path_sep(at + 1) {
            at += 3;
        }
        let bang = at + 1;
        if !is_name(self.src.kind(at)) || self.src.kind(bang) != TokenKind::Bang {
            bail!(
                "{}: unexpected {:?} at byte {} in item position",
                self.src.path.display(),
                self.src.text(start),
                self.src.byte(start)
            );
        }
        let open = if is_name(self.src.kind(bang + 1)) {
            bang + 2
        } else {
            bang + 1
        };
        Ok(Head {
            kind: ItemKind::MacroCall,
            name: Some(at),
            body: Some(open),
            end: self.delimited_end(open)?,
        })
    }

    fn use_item(&mut self, at: u32) -> Result<Head> {
        let item = self.out.items.len() as u32;
        let end = self.to_semi(at + 1);
        self.tree(at + 1, &[], false, item)?;
        Ok(Head {
            kind: ItemKind::Use,
            name: None,
            body: None,
            end,
        })
    }

    fn tree(&mut self, mut at: u32, prefix: &[u32], mut absolute: bool, item: u32) -> Result<u32> {
        let mut path = prefix.to_vec();
        if path.is_empty() && self.is_path_sep(at) {
            absolute = true;
            at += 2;
        }
        if path.is_empty() && self.src.kind(at) == TokenKind::Dollar {
            at += 1;
        }
        loop {
            match self.src.kind(at) {
                TokenKind::Star => {
                    self.out.imports.push(Import {
                        item,
                        absolute,
                        path,
                        leaf: Leaf::Glob,
                    });
                    return Ok(at + 1);
                }
                TokenKind::OpenBrace => return self.group(at, &path, absolute, item),
                TokenKind::Ident | TokenKind::RawIdent if self.is_path_sep(at + 1) => {
                    path.push(at);
                    at += 3;
                }
                TokenKind::Ident | TokenKind::RawIdent => {
                    return Ok(self.leaf(at, path, absolute, item));
                }
                _ => bail!(
                    "{}: bad use tree at byte {}",
                    self.src.path.display(),
                    self.src.byte(at)
                ),
            }
        }
    }

    fn group(&mut self, open: u32, path: &[u32], absolute: bool, item: u32) -> Result<u32> {
        let close = self.src.pair(open);
        let mut at = open + 1;
        while at < close {
            at = self.tree(at, path, absolute, item)?;
            if self.src.kind(at) == TokenKind::Comma {
                at += 1;
            }
        }
        Ok(close + 1)
    }

    fn leaf(&mut self, name: u32, mut path: Vec<u32>, absolute: bool, item: u32) -> u32 {
        path.push(name);
        let (leaf, next) = match self.word(name + 1) {
            "as" if self.word(name + 2) == "_" => (Leaf::Underscore, name + 3),
            "as" => (Leaf::Name(Some(name + 2)), name + 3),
            _ => (Leaf::Name(None), name + 1),
        };
        self.out.imports.push(Import {
            item,
            absolute,
            path,
            leaf,
        });
        next
    }
}

const DISALLOWED_BEFORE_GEN_BLOCK: [&str; 25] = [
    "struct", "enum", "union", "trait", "mod", "impl", "for", "if", "while", "match", "let",
    "type", "const", "static", "use", "extern", "pub", "crate", "super", "self", "in", "where",
    "as", "unsafe", "async",
];

const NON_MACRO_KW: [&str; 11] = [
    "if", "else", "while", "for", "loop", "match", "return", "break", "in", "yield", "where",
];

pub(crate) fn syntax_error(src: &Source, start: u32, end: u32, msg: &str, label: &str) -> String {
    let span = Span::from_tokens(src, start, end);
    let mut diag = if span.is_some() {
        Diagnostic::new(DiagnosticKind::Syntax, msg).with_span(span)
    } else {
        Diagnostic::new(
            DiagnosticKind::Syntax,
            format!("{}: {msg}", src.path.display()),
        )
    };
    if !label.is_empty() && diag.span().is_some() {
        diag = diag.with_primary_label(label);
    }
    diag.to_string()
}

pub(crate) fn is_reserved_ident_2024(src: &Source, tok: u32) -> bool {
    src.kind(tok) == TokenKind::Ident && src.text(tok) == "gen"
}

fn gen_error(src: &Source, tok: u32) -> String {
    syntax_error(
        src,
        tok,
        tok + 1,
        "expected identifier, found reserved keyword `gen`",
        "expected identifier, found reserved keyword",
    )
}

fn check_tok_2024(src: &Source, tok: u32, unmarked_only: bool, errors: &mut Vec<String>) {
    if (!unmarked_only || src.mark(tok).is_none()) && is_reserved_ident_2024(src, tok) {
        errors.push(gen_error(src, tok));
    }
}

fn attr_open_at(src: &Source, at: u32) -> Option<u32> {
    if src.kind(at) != TokenKind::Pound {
        return None;
    }
    if src.kind(at + 1) == TokenKind::OpenBracket {
        Some(at + 1)
    } else if src.kind(at + 1) == TokenKind::Bang && src.kind(at + 2) == TokenKind::OpenBracket {
        Some(at + 2)
    } else {
        None
    }
}

fn delimited_after(src: &Source, open: u32) -> u32 {
    let close = src.pair(open);
    if src.kind(open) != TokenKind::OpenBrace && src.kind(close + 1) == TokenKind::Semi {
        close + 2
    } else {
        close + 1
    }
}

fn macro_skip_end(src: &Source, at: u32, n: u32) -> Option<(Option<u32>, u32)> {
    if src.kind(at) == TokenKind::Ident
        && src.text(at) == "macro_rules"
        && src.kind(at + 1) == TokenKind::Bang
        && at + 3 < n
        && opens(src.kind(at + 3))
    {
        return Some((Some(at + 2), delimited_after(src, at + 3)));
    }
    if src.kind(at) == TokenKind::Ident
        && src.text(at) == "macro"
        && is_name(src.kind(at + 1))
        && at + 2 < n
    {
        let body = if src.kind(at + 2) == TokenKind::OpenBrace {
            Some(at + 2)
        } else if src.kind(at + 2) == TokenKind::OpenParen
            && src.kind(src.pair(at + 2) + 1) == TokenKind::OpenBrace
        {
            Some(src.pair(at + 2) + 1)
        } else {
            None
        };
        if let Some(open) = body {
            return Some((Some(at + 1), src.pair(open) + 1));
        }
    }
    if is_name(src.kind(at))
        && !(src.kind(at) == TokenKind::Ident && NON_MACRO_KW.contains(&src.text(at)))
        && src.kind(at + 1) == TokenKind::Bang
        && at + 2 < n
        && opens(src.kind(at + 2))
    {
        return Some((Some(at), delimited_after(src, at + 2)));
    }
    None
}

fn opens_gen_block(src: &Source, at: u32) -> bool {
    let followed_by_brace = src.kind(at + 1) == TokenKind::OpenBrace
        || (src.kind(at + 1) == TokenKind::Ident
            && src.text(at + 1) == "move"
            && src.kind(at + 2) == TokenKind::OpenBrace);
    if !followed_by_brace {
        return false;
    }
    let Some(prev) = at.checked_sub(1) else {
        return false;
    };
    if matches!(src.kind(prev), TokenKind::Dot | TokenKind::Colon) {
        return false;
    }
    !(src.kind(prev) == TokenKind::Ident && DISALLOWED_BEFORE_GEN_BLOCK.contains(&src.text(prev)))
}

fn check_extern_block_2024(src: &Source, at: u32, errors: &mut Vec<String>) {
    let after = if matches!(src.kind(at + 1), TokenKind::Literal { .. }) {
        at + 2
    } else {
        at + 1
    };
    let prev_unsafe = at
        .checked_sub(1)
        .is_some_and(|p| src.kind(p) == TokenKind::Ident && src.text(p) == "unsafe");
    if !prev_unsafe && src.kind(after) == TokenKind::OpenBrace {
        errors.push(syntax_error(
            src,
            at,
            at + 1,
            "extern blocks must be unsafe",
            "extern block without `unsafe`",
        ));
    }
}

fn update_enum_braces(
    src: &Source,
    at: u32,
    pending_enum: &mut bool,
    enum_braces: &mut Vec<(u32, bool)>,
) {
    let kind = src.kind(at);
    if kind == TokenKind::Ident
        && src.text(at) == "enum"
        && !at
            .checked_sub(1)
            .is_some_and(|prev| matches!(src.kind(prev), TokenKind::Dot | TokenKind::Colon))
    {
        *pending_enum = true;
    } else if kind == TokenKind::Semi {
        *pending_enum = false;
    } else if kind == TokenKind::OpenBrace {
        let prev_eq = at
            .checked_sub(1)
            .is_some_and(|prev| src.kind(prev) == TokenKind::Eq);
        let is_enum = *pending_enum && !prev_eq;
        if is_enum {
            *pending_enum = false;
        }
        enum_braces.push((src.pair(at), is_enum));
    }
}

pub(crate) fn check_edition_2024_source(
    src: &Source,
    unmarked_only: bool,
    on_attr: &mut impl FnMut(u32, u32, &mut Vec<String>),
    errors: &mut Vec<String>,
) {
    let mut at = 0_u32;
    let n = src.len();
    let mut pending_enum = false;
    let mut enum_braces: Vec<(u32, bool)> = Vec::new();
    while at < n {
        while enum_braces.last().is_some_and(|&(close, _)| at >= close) {
            enum_braces.pop();
        }
        if let Some(open) = attr_open_at(src, at) {
            let close = src.pair(open);
            on_attr(open + 1, close, errors);
            let mut p = open + 1;
            while p < close
                && matches!(
                    src.kind(p),
                    TokenKind::Ident | TokenKind::RawIdent | TokenKind::Colon
                )
            {
                check_tok_2024(src, p, unmarked_only, errors);
                p += 1;
            }
            at = close + 1;
            continue;
        }
        if let Some((name_tok, next)) = macro_skip_end(src, at, n) {
            if let Some(tok) = name_tok {
                check_tok_2024(src, tok, unmarked_only, errors);
            }
            pending_enum = false;
            at = next;
            continue;
        }
        update_enum_braces(src, at, &mut pending_enum, &mut enum_braces);
        if !unmarked_only || src.mark(at).is_none() {
            if src.kind(at) == TokenKind::Ident && src.text(at) == "extern" {
                check_extern_block_2024(src, at, errors);
            }
            if is_reserved_ident_2024(src, at)
                && (enum_braces.last().is_some_and(|&(_, is_enum)| is_enum)
                    || !opens_gen_block(src, at))
            {
                errors.push(gen_error(src, at));
            }
        }
        at += 1;
    }
}
