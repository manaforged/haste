use anyhow::{Result, bail};
use rustc_lexer::TokenKind;

use crate::outline::source::Source;

use super::{Body, ExprId, List, PatId, PathId, TyId};

/// A position in a token range, with the body being built from it.
pub struct Cursor<'a> {
    pub src: &'a Source,
    pub at: u32,
    pub end: u32,
    pub body: Body,
    pub no_struct: bool,
}

/// Whether `kind` is an identifier.
pub fn is_name(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Ident | TokenKind::RawIdent)
}

const RESERVED: [&str; 38] = [
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where", "while", "async", "await", "dyn",
];

impl<'a> Cursor<'a> {
    pub fn new(src: &'a Source, at: u32, end: u32) -> Cursor<'a> {
        Cursor { src, at, end: end.min(src.len()), body: Body::default(), no_struct: false }
    }

    pub fn kind_at(&self, at: u32) -> TokenKind {
        if at < self.end { self.src.kind(at) } else { TokenKind::Eof }
    }

    pub fn kind(&self) -> TokenKind {
        self.kind_at(self.at)
    }

    pub fn done(&self) -> bool {
        self.at >= self.end
    }

    pub fn word_at(&self, at: u32) -> &'a str {
        if is_name(self.kind_at(at)) { self.src.text(at) } else { "" }
    }

    pub fn word(&self) -> &'a str {
        self.word_at(self.at)
    }

    pub fn is_word(&self, word: &str) -> bool {
        self.kind() == TokenKind::Ident && self.src.text(self.at) == word
    }

    pub fn eat_word(&mut self, word: &str) -> bool {
        let found = self.is_word(word);
        self.at += u32::from(found);
        found
    }

    pub fn ident(&self) -> bool {
        match self.kind() {
            TokenKind::RawIdent => true,
            TokenKind::Ident => !RESERVED.contains(&self.src.text(self.at)),
            _ => false,
        }
    }

    pub fn punct_at(&self, at: u32, text: &str) -> bool {
        let mut at = at;
        let count = text.len() as u32;
        for (index, ch) in text.char_indices() {
            if at >= self.end || !is_punct(self.src.kind(at)) || self.src.text(at) != &text[index..index + ch.len_utf8()] {
                return false;
            }
            if (index as u32) + 1 < count && !self.src.joint(at) {
                return false;
            }
            at += 1;
        }
        true
    }

    pub fn punct(&self, text: &str) -> bool {
        self.punct_at(self.at, text)
    }

    pub fn eat(&mut self, text: &str) -> bool {
        let found = self.punct(text);
        if found {
            self.at += text.len() as u32;
        }
        found
    }

    pub fn expect(&mut self, text: &str) -> Result<()> {
        if self.eat(text) {
            return Ok(());
        }
        self.fail(&format!("expected `{text}`"))
    }

    pub fn expect_kind(&mut self, kind: TokenKind) -> Result<u32> {
        if self.kind() != kind {
            return self.fail(&format!("expected {kind:?}"));
        }
        self.at += 1;
        Ok(self.at - 1)
    }

    pub fn name(&mut self) -> Result<u32> {
        if !is_name(self.kind()) {
            return self.fail("expected a name");
        }
        self.at += 1;
        Ok(self.at - 1)
    }

    pub fn fail<T>(&self, what: &str) -> Result<T> {
        let near = if self.at < self.end { self.src.text(self.at).to_owned() } else { "end".to_owned() };
        let line = self.src.tokens.get(self.at.min(self.src.len().saturating_sub(1)) as usize).map_or(0, |token| self.src.text[..token.start as usize].matches('\n').count() + 1);
        bail!("{}:{line}: {what}, found `{near}`", self.src.path.display())
    }

    pub fn finish(&self) -> Result<()> {
        if self.done() { Ok(()) } else { self.fail("expected the end") }
    }

    pub fn group(&mut self) -> u32 {
        let open = self.at;
        self.at = self.src.pair(open) + 1;
        open
    }

    pub fn inside<T>(&mut self, open: u32, run: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let (end, no_struct) = (self.end, self.no_struct);
        self.end = self.src.pair(open);
        self.at = open + 1;
        self.no_struct = false;
        let out = run(self);
        let out = out.and_then(|value| self.finish().map(|()| value));
        self.at = self.end + 1;
        self.end = end;
        self.no_struct = no_struct;
        out
    }

    pub fn list(&mut self, ids: &[u32]) -> List {
        let start = self.body.ids.len() as u32;
        self.body.ids.extend_from_slice(ids);
        List { start, len: ids.len() as u32 }
    }

    pub fn comma_list(&mut self, open: u32, mut item: impl FnMut(&mut Self) -> Result<u32>) -> Result<List> {
        let ids = self.inside(open, |cursor| {
            let mut ids = Vec::new();
            while !cursor.done() {
                ids.push(item(cursor)?);
                if !cursor.eat(",") {
                    break;
                }
            }
            Ok(ids)
        })?;
        Ok(self.list(&ids))
    }

    pub fn push_expr(&mut self, expr: super::Expr) -> ExprId {
        self.body.exprs.push(expr);
        ExprId(self.body.exprs.len() as u32 - 1)
    }

    pub fn push_pat(&mut self, pat: super::Pat) -> PatId {
        self.body.pats.push(pat);
        PatId(self.body.pats.len() as u32 - 1)
    }

    pub fn push_ty(&mut self, ty: super::Ty) -> TyId {
        self.body.tys.push(ty);
        TyId(self.body.tys.len() as u32 - 1)
    }

    pub fn push_path(&mut self, path: super::Path) -> PathId {
        self.body.paths.push(path);
        PathId(self.body.paths.len() as u32 - 1)
    }

    pub fn skip_attrs(&mut self) -> Result<Option<(u32, u32)>> {
        let start = self.at;
        while self.punct("#") {
            self.at += 1;
            self.eat("!");
            if self.kind() != TokenKind::OpenBracket {
                return self.fail("expected an attribute");
            }
            self.group();
        }
        Ok((self.at > start).then_some((start, self.at)))
    }

    pub fn lifetime(&mut self) -> Option<u32> {
        if matches!(self.kind(), TokenKind::Lifetime { .. } | TokenKind::RawLifetime) {
            self.at += 1;
            return Some(self.at - 1);
        }
        None
    }
}

fn is_punct(kind: TokenKind) -> bool {
    use TokenKind::*;
    matches!(kind, Semi | Comma | Dot | At | Pound | Tilde | Question | Colon | Dollar | Eq | Bang | Lt | Gt | Minus | And | Or | Plus | Star | Slash | Caret | Percent)
}
