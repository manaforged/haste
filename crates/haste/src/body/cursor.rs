use anyhow::Result;
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
    pub config: super::Config<'a>,
    /// The block that the last macro expansion parsed to, which a statement macro splices into its enclosing block.
    pub expansion: Option<ExprId>,
    /// The `expr;` statement that the last expansion held alone, which an expression takes as its value and a statement macro splices with its semicolon.
    pub expansion_stmt: Option<u32>,
}

/// Whether `kind` is an identifier.
pub fn is_name(kind: TokenKind) -> bool {
    matches!(kind, TokenKind::Ident | TokenKind::RawIdent)
}

pub const RESERVED: [&str; 38] = [
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn", "for",
    "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
    "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
    "while", "async", "await", "dyn",
];

/// The keywords that can lead a path.
pub const PATH_KEYWORDS: [&str; 4] = ["crate", "self", "Self", "super"];

impl<'a> Cursor<'a> {
    pub fn new(src: &'a Source, at: u32, end: u32) -> Cursor<'a> {
        Cursor {
            src,
            at,
            end: end.min(src.len()),
            body: Body::default(),
            no_struct: false,
            config: super::Config::default(),
            expansion: None,
            expansion_stmt: None,
        }
    }

    /// Whether the outer attributes at `attrs` hold a `#[cfg]` that the environment turns off.
    pub fn configured_out(&self, attrs: Option<(u32, u32)>) -> bool {
        let (Some(env), Some((from, _))) = (self.config.env, attrs) else {
            return false;
        };
        let mut parsed = Vec::new();
        crate::outline::attr::attrs(self.src, from, false, &mut parsed).is_ok()
            && !env.enabled(&parsed)
    }

    #[inline]
    pub fn kind_at(&self, at: u32) -> TokenKind {
        if at < self.end {
            self.src.kind(at)
        } else {
            TokenKind::Eof
        }
    }

    #[inline]
    pub fn kind(&self) -> TokenKind {
        self.kind_at(self.at)
    }

    #[inline]
    pub fn done(&self) -> bool {
        self.at >= self.end
    }

    pub fn word_at(&self, at: u32) -> &'a str {
        if is_name(self.kind_at(at)) {
            self.src.text(at)
        } else {
            ""
        }
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

    #[inline]
    pub fn punct_at(&self, at: u32, text: &str) -> bool {
        punct_in(self.src, (at, self.end), text)
    }

    #[inline]
    pub fn punct(&self, text: &str) -> bool {
        self.punct_at(self.at, text)
    }

    #[inline]
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
        let near = if self.at < self.end {
            self.src.text(self.at).to_owned()
        } else {
            "end".to_owned()
        };
        let tok = if self.src.is_empty() {
            None
        } else {
            Some(self.at.min(self.src.len() - 1))
        };
        let span = tok.and_then(|t| crate::diag::Span::from_token(self.src, t));
        let diag = crate::diag::Diagnostic::new(
            crate::diag::DiagnosticKind::Syntax,
            format!("{what}, found `{near}`"),
        )
        .with_span(span)
        .with_primary_label(what);
        Err(diag.into())
    }

    pub fn finish(&self) -> Result<()> {
        if self.done() {
            Ok(())
        } else {
            self.fail("expected the end")
        }
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
        List {
            start,
            len: ids.len() as u32,
        }
    }

    pub fn comma_list(
        &mut self,
        open: u32,
        mut item: impl FnMut(&mut Self) -> Result<u32>,
    ) -> Result<List> {
        self.kept_list(open, |cursor| item(cursor).map(Some))
    }

    /// The comma-separated items inside the group at `open`, keeping those that `item` returns.
    pub fn kept_list(
        &mut self,
        open: u32,
        mut item: impl FnMut(&mut Self) -> Result<Option<u32>>,
    ) -> Result<List> {
        let ids = self.inside(open, |cursor| {
            let mut ids = Vec::new();
            while !cursor.done() {
                ids.extend(item(cursor)?);
                if !cursor.eat(",") {
                    break;
                }
            }
            Ok(ids)
        })?;
        Ok(self.list(&ids))
    }

    pub fn push_expr(&mut self, start: u32, expr: super::Expr) -> ExprId {
        self.body.exprs.push(expr);
        self.body.expr_starts.push(start);
        self.body.expr_ends.push(self.at);
        ExprId(self.body.exprs.len() as u32 - 1)
    }

    pub fn expr_start_tok(&self, id: ExprId) -> u32 {
        self.body.expr_span(id).map_or(self.at, |(s, _)| s)
    }

    pub fn push_pat(&mut self, start: u32, pat: super::Pat) -> PatId {
        self.body.pats.push(pat);
        self.body.pat_spans.push((start, self.at));
        PatId(self.body.pats.len() as u32 - 1)
    }

    pub fn pat_start_tok(&self, id: PatId) -> u32 {
        self.body.pat_span(id).map_or(self.at, |(s, _)| s)
    }

    /// Remaps macro-synthesized expression and pattern spans parsed since `(first_expr, first_pat)` to the invocation token `start`.
    pub(super) fn remap_expansion_spans(
        &mut self,
        first_expr: usize,
        first_pat: usize,
        start: u32,
    ) {
        for idx in first_expr..self.body.expr_ends.len() {
            let (s, e) = (self.body.expr_starts[idx], self.body.expr_ends[idx]);
            if self.src.place(s) == 0 || self.src.place(e.saturating_sub(1)) == 0 {
                self.body.expr_starts[idx] = start;
                self.body.expr_ends[idx] = start + 1;
            }
        }
        for slot in &mut self.body.pat_spans[first_pat..] {
            if self.src.place(slot.0) == 0 || self.src.place(slot.1.saturating_sub(1)) == 0 {
                *slot = (start, start + 1);
            }
        }
    }

    pub fn push_ty(&mut self, ty: super::Ty) -> TyId {
        self.body.tys.push(ty);
        TyId(self.body.tys.len() as u32 - 1)
    }

    pub fn push_path(&mut self, path: super::Path) -> PathId {
        self.body.paths.push(path);
        PathId(self.body.paths.len() as u32 - 1)
    }

    /// The fields of the brace group at the cursor that the configuration keeps, each read by `field` with its attributes, and whether a `..` ends them, after which `rest` reads what follows.
    pub fn struct_fields(
        &mut self,
        mut field: impl FnMut(&mut Self, Option<(u32, u32)>) -> Result<u32>,
        rest: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<(Vec<u32>, bool)> {
        let open = self.group();
        self.inside(open, |cursor| {
            let mut fields = Vec::new();
            while !cursor.done() {
                let attrs = cursor.skip_attrs()?;
                if cursor.eat("..") {
                    rest(cursor)?;
                    return Ok((fields, true));
                }
                let at = field(cursor, attrs)?;
                if !cursor.configured_out(attrs) {
                    fields.push(at);
                }
                if !cursor.eat(",") {
                    break;
                }
            }
            Ok((fields, false))
        })
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
        if matches!(
            self.kind(),
            TokenKind::Lifetime { .. } | TokenKind::RawLifetime
        ) {
            self.at += 1;
            return Some(self.at - 1);
        }
        None
    }
}

const PUNCT: [(u8, TokenKind); 21] = [
    (b';', TokenKind::Semi),
    (b',', TokenKind::Comma),
    (b'.', TokenKind::Dot),
    (b'@', TokenKind::At),
    (b'#', TokenKind::Pound),
    (b'~', TokenKind::Tilde),
    (b'?', TokenKind::Question),
    (b':', TokenKind::Colon),
    (b'$', TokenKind::Dollar),
    (b'=', TokenKind::Eq),
    (b'!', TokenKind::Bang),
    (b'<', TokenKind::Lt),
    (b'>', TokenKind::Gt),
    (b'-', TokenKind::Minus),
    (b'&', TokenKind::And),
    (b'|', TokenKind::Or),
    (b'+', TokenKind::Plus),
    (b'*', TokenKind::Star),
    (b'/', TokenKind::Slash),
    (b'^', TokenKind::Caret),
    (b'%', TokenKind::Percent),
];

const PUNCT_KINDS: [Option<TokenKind>; 128] = {
    let mut table = [None; 128];
    let mut at = 0;
    while at < PUNCT.len() {
        table[PUNCT[at].0 as usize] = Some(PUNCT[at].1);
        at += 1;
    }
    table
};

/// Whether the joined punctuation tokens of `src` from `at` spell `text` and end before `end`.
#[inline]
pub(crate) fn punct_in(src: &Source, (at, end): (u32, u32), text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() == 1 {
        return at < end && Some(src.kind(at)) == punct_kind(bytes[0]);
    }
    let last = at + bytes.len() as u32 - 1;
    last < end
        && bytes.iter().zip(at..).all(|(&byte, here)| {
            src.token_at(here).is_some_and(|tok| {
                Some(tok.kind) == punct_kind(byte) && (here == last || tok.joint())
            })
        })
}

#[inline]
fn punct_kind(byte: u8) -> Option<TokenKind> {
    PUNCT_KINDS.get(byte as usize).copied().flatten()
}
