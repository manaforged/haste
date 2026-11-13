use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;
use rustc_lexer::TokenKind;

use super::attr;
use super::item::{Attr, Import, Item, ItemKind, Items, Leaf, Vis};
use super::source::{Source, opens};

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
const BODIED: [(&str, ItemKind); 4] = [("mod", ItemKind::Mod), ("fn", ItemKind::Fn), ("struct", ItemKind::Struct), ("enum", ItemKind::Enum)];
const AFTER_CONST: [&str; 7] = ["fn", "async", "unsafe", "extern", "safe", "impl", "trait"];

/// Scans the items of a whole file.
pub fn scan(src: &Source) -> Result<Items> {
    let mut scanner = Scanner { src, out: Items::default(), calls: None };
    let mut inner = Vec::new();
    let at = attr::attrs(src, 0, true, &mut inner)?;
    scanner.out.inner = inner;
    scanner.items(at, src.len(), None)?;
    Ok(scanner.out)
}

/// Adds to `items` the items declared in a body, each starting at its token under the item whose body declares it, reading a module's macro calls through their expansions in `calls`.
pub fn body_items(src: &Source, starts: impl IntoIterator<Item = (u32, Option<u32>)>, (items, calls): (Items, Option<&HashMap<u32, u32>>)) -> Result<Items> {
    let mut scanner = Scanner { src, out: items, calls };
    for (at, owner) in starts {
        scanner.item(at, owner)?;
    }
    Ok(scanner.out)
}

/// The token after the item that starts at `at`.
pub fn item_end(src: &Source, at: u32) -> Result<u32> {
    Scanner { src, out: Items::default(), calls: None }.item(at, None)
}

impl Scanner<'_> {
    fn word(&self, at: u32) -> &str {
        if is_name(self.src.kind(at)) { self.src.text(at) } else { "" }
    }

    fn is_path_sep(&self, at: u32) -> bool {
        self.src.kind(at) == TokenKind::Colon && self.src.kind(at + 1) == TokenKind::Colon && self.src.joint(at)
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
        self.out.items.push(Item { kind: head.kind, name: head.name, vis, attrs, lead, start, end: head.end, body: head.body, parent });
        self.children(index, &head)?;
        if let (ItemKind::MacroCall, Some(_), Some(open)) = (head.kind, parent, head.body)
            && let Some(&block) = self.calls.and_then(|calls| calls.get(&open))
        {
            self.items(block + 1, self.src.pair(block), parent)?;
        }
        Ok(head.end)
    }

    fn inner_attrs(&mut self, open: u32, kind: ItemKind, into: &mut Vec<Attr>) -> Result<()> {
        if matches!(kind, ItemKind::Mod | ItemKind::Impl | ItemKind::Trait | ItemKind::ExternBlock) {
            attr::attrs(self.src, open + 1, true, into)?;
        }
        Ok(())
    }

    fn children(&mut self, index: u32, head: &Head) -> Result<()> {
        let Some(open) = head.body else { return Ok(()) };
        if !matches!(head.kind, ItemKind::Mod | ItemKind::Impl | ItemKind::Trait | ItemKind::ExternBlock) {
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
            let qualifier = QUALIFIERS.contains(&word) && is_name(self.src.kind(at + 1)) && !matches!(next, "" | "!");
            let constant = word == "const" && AFTER_CONST.contains(&next);
            let restricted = word == "impl" && self.src.kind(at + 1) == TokenKind::OpenParen && is_name(self.src.kind(self.src.pair(at + 1) + 1));
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
            return Ok(Head { kind, name: Some(at + 1), body, end });
        }
        let next = self.src.kind(at + 1);
        match word {
            "use" => self.use_item(at),
            "extern" => self.extern_item(at),
            "union" if is_name(next) => self.bodied(ItemKind::Union, at),
            "trait" => self.trait_item(at),
            "impl" => {
                let (body, end) = self.to_body(at + 1);
                Ok(Head { kind: ItemKind::Impl, name: None, body, end })
            }
            "type" | "const" => {
                let kind = if word == "type" { ItemKind::Type } else { ItemKind::Const };
                Ok(Head { kind, name: Some(at + 1), body: None, end: self.to_semi(at + 2) })
            }
            "static" => self.static_item(at),
            "macro_rules" if next == TokenKind::Bang => self.macro_rules(at),
            "macro" if is_name(next) => self.macro_item(at),
            _ => self.macro_call(at),
        }
    }

    fn bodied(&self, kind: ItemKind, at: u32) -> Result<Head> {
        let (body, end) = self.to_body(at + 2);
        Ok(Head { kind, name: Some(at + 1), body, end })
    }

    fn is_arrow(&self, at: u32) -> bool {
        at.checked_sub(1).is_some_and(|before| matches!(self.src.kind(before), TokenKind::Minus | TokenKind::Eq) && self.src.joint(before))
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
        let alias = body.is_none() && (at + 2..end).any(|token| self.src.kind(token) == TokenKind::Eq);
        let kind = if alias { ItemKind::TraitAlias } else { ItemKind::Trait };
        Ok(Head { kind, name: Some(at + 1), body, end })
    }

    fn static_item(&self, at: u32) -> Result<Head> {
        let name = if self.word(at + 1) == "mut" { at + 2 } else { at + 1 };
        Ok(Head { kind: ItemKind::Static, name: Some(name), body: None, end: self.to_semi(name + 1) })
    }

    fn extern_item(&mut self, at: u32) -> Result<Head> {
        if self.word(at + 1) == "crate" {
            let name = if self.word(at + 3) == "as" { at + 4 } else { at + 2 };
            return Ok(Head { kind: ItemKind::ExternCrate, name: Some(name), body: None, end: self.to_semi(at + 2) });
        }
        let after = if matches!(self.src.kind(at + 1), TokenKind::Literal { .. }) { at + 2 } else { at + 1 };
        if self.src.kind(after) == TokenKind::OpenBrace {
            return Ok(Head { kind: ItemKind::ExternBlock, name: None, body: Some(after), end: self.src.pair(after) + 1 });
        }
        self.head(self.qualifiers(after))
    }

    fn delimited_end(&self, open: u32) -> Result<u32> {
        if !opens(self.src.kind(open)) {
            bail!("{}: expected `(`, `[` or `{{` at byte {} after a macro name", self.src.path.display(), self.src.byte(open));
        }
        let close = self.src.pair(open);
        let braced = self.src.kind(open) == TokenKind::OpenBrace;
        Ok(if !braced && self.src.kind(close + 1) == TokenKind::Semi { close + 2 } else { close + 1 })
    }

    fn macro_rules(&self, at: u32) -> Result<Head> {
        let open = at + 3;
        Ok(Head { kind: ItemKind::MacroRules, name: Some(at + 2), body: Some(open), end: self.delimited_end(open)? })
    }

    fn macro_item(&self, at: u32) -> Result<Head> {
        let mut last = at + 2;
        if self.src.kind(last) == TokenKind::OpenParen && self.src.kind(self.src.pair(last) + 1) == TokenKind::OpenBrace {
            last = self.src.pair(last) + 1;
        }
        Ok(Head { kind: ItemKind::Macro, name: Some(at + 1), body: Some(at + 2), end: self.src.pair(last) + 1 })
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
            bail!("{}: unexpected {:?} at byte {} in item position", self.src.path.display(), self.src.text(start), self.src.byte(start));
        }
        let open = if is_name(self.src.kind(bang + 1)) { bang + 2 } else { bang + 1 };
        Ok(Head { kind: ItemKind::MacroCall, name: Some(at), body: Some(open), end: self.delimited_end(open)? })
    }

    fn use_item(&mut self, at: u32) -> Result<Head> {
        let item = self.out.items.len() as u32;
        let end = self.to_semi(at + 1);
        self.tree(at + 1, &[], false, item)?;
        Ok(Head { kind: ItemKind::Use, name: None, body: None, end })
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
                    self.out.imports.push(Import { item, absolute, path, leaf: Leaf::Glob });
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
                _ => bail!("{}: bad use tree at byte {}", self.src.path.display(), self.src.byte(at)),
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
        self.out.imports.push(Import { item, absolute, path, leaf });
        next
    }
}
