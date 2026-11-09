use anyhow::Result;
use rustc_lexer::TokenKind;

use super::cursor::Cursor;
use super::path::Style;
use crate::outline::source::opens;
use super::{Expr, FieldPat, Lit, Pat, PatId, UnOp};

impl Cursor<'_> {
    /// Parses a pattern, with `|` alternatives.
    pub fn pat(&mut self) -> Result<PatId> {
        self.eat("|");
        let first = self.pat_no_alt()?;
        if !self.alt_ahead() {
            return Ok(first);
        }
        let mut alternatives = vec![first.0];
        while self.alt_ahead() {
            self.at += 1;
            alternatives.push(self.pat_no_alt()?.0);
        }
        let list = self.list(&alternatives);
        Ok(self.push_pat(Pat::Or(list)))
    }

    fn alt_ahead(&self) -> bool {
        self.punct("|") && !self.punct("||")
    }

    /// Parses a pattern without top-level `|`.
    pub fn pat_no_alt(&mut self) -> Result<PatId> {
        let pat = self.pat_start()?;
        self.range_tail(pat)
    }

    fn range_tail(&mut self, pat: PatId) -> Result<PatId> {
        let inclusive = self.punct("..=") || self.punct("...");
        if !(inclusive || self.punct("..")) {
            return Ok(pat);
        }
        let from = match self.body.pats[pat.0 as usize] {
            Pat::Lit(from) => from,
            Pat::Path(path) => self.push_expr(Expr::Path(path)),
            _ => return Ok(pat),
        };
        self.at += if inclusive { 3 } else { 2 };
        let to = if self.range_end_ahead() { Some(self.range_bound()?) } else { None };
        Ok(self.push_pat(Pat::Range { from: Some(from), to, inclusive }))
    }

    fn range_end_ahead(&self) -> bool {
        matches!(self.kind(), TokenKind::Literal { .. }) || self.punct("-") || self.ident() || self.punct("::") || self.punct("<") || self.is_word("crate") || self.is_word("self") || self.is_word("Self") || self.is_word("super")
    }

    fn range_bound(&mut self) -> Result<super::ExprId> {
        if matches!(self.kind(), TokenKind::Literal { .. }) || self.punct("-") {
            return self.lit_expr();
        }
        let path = self.path(Style::Expr)?;
        Ok(self.push_expr(Expr::Path(path)))
    }

    fn lit_expr(&mut self) -> Result<super::ExprId> {
        let negative = self.eat("-");
        let lit = self.literal()?;
        Ok(if negative { self.push_expr(Expr::Unary(UnOp::Neg, lit)) } else { lit })
    }

    fn pat_start(&mut self) -> Result<PatId> {
        match self.kind() {
            TokenKind::OpenParen => return self.tuple_pat(),
            TokenKind::OpenBracket => {
                let open = self.at;
                self.group();
                let list = self.comma_list(open, |cursor| Ok(cursor.pat()?.0))?;
                return Ok(self.push_pat(Pat::Slice(list)));
            }
            TokenKind::Literal { .. } => {
                let lit = self.literal()?;
                return Ok(self.push_pat(Pat::Lit(lit)));
            }
            _ => {}
        }
        self.punct_pat()
    }

    fn punct_pat(&mut self) -> Result<PatId> {
        if self.punct("-") {
            let lit = self.lit_expr()?;
            return Ok(self.push_pat(Pat::Lit(lit)));
        }
        if self.punct("&") {
            self.at += 1;
            let mutable = self.eat_word("mut");
            let inner = self.pat_no_alt()?;
            return Ok(self.push_pat(Pat::Ref { mutable, pat: inner }));
        }
        if self.punct("..=") || self.punct("...") {
            self.at += 3;
            let to = self.range_bound()?;
            return Ok(self.push_pat(Pat::Range { from: None, to: Some(to), inclusive: true }));
        }
        if self.eat("..") {
            if self.range_end_ahead() {
                let to = self.range_bound()?;
                return Ok(self.push_pat(Pat::Range { from: None, to: Some(to), inclusive: false }));
            }
            return Ok(self.push_pat(Pat::Rest));
        }
        if self.eat("!") {
            return Ok(self.push_pat(Pat::Never));
        }
        self.word_pat()
    }

    fn word_pat(&mut self) -> Result<PatId> {
        match self.word() {
            "_" => {
                self.at += 1;
                return Ok(self.push_pat(Pat::Wild));
            }
            "true" | "false" => {
                let lit = self.literal()?;
                return Ok(self.push_pat(Pat::Lit(lit)));
            }
            "box" | "const" => {
                if let Some(pat) = self.keyword_pat()? {
                    return Ok(pat);
                }
            }
            "ref" | "mut" => return self.bind_pat(),
            _ => {}
        }
        if self.ident() && !self.path_continues() {
            return self.bind_pat();
        }
        self.path_pat()
    }

    fn keyword_pat(&mut self) -> Result<Option<PatId>> {
        let next = self.at + 1;
        if self.is_word("const") && self.kind_at(next) == TokenKind::OpenBrace {
            self.at += 1;
            let block = self.block_expr(None, super::BlockKind::Const)?;
            return Ok(Some(self.push_pat(Pat::Lit(block))));
        }
        let boxed = self.is_word("box") && (!self.word_at(next).is_empty() || matches!(self.kind_at(next), TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::Literal { .. }) || self.punct_at(next, "&"));
        if !boxed {
            return Ok(None);
        }
        self.at += 1;
        let inner = self.pat_no_alt()?;
        Ok(Some(self.push_pat(Pat::Box(inner))))
    }

    fn path_continues(&self) -> bool {
        let next = self.at + 1;
        self.punct_at(next, "::") || self.punct_at(next, "!") || matches!(self.kind_at(next), TokenKind::OpenParen | TokenKind::OpenBrace) || self.punct_at(next, "<")
    }

    fn bind_pat(&mut self) -> Result<PatId> {
        let by_ref = self.eat_word("ref");
        let mutable = self.eat_word("mut");
        let name = self.name()?;
        let sub = if self.punct("@") {
            self.at += 1;
            Some(self.pat_no_alt()?)
        } else {
            None
        };
        Ok(self.push_pat(Pat::Bind { name, by_ref, mutable, sub }))
    }

    fn path_pat(&mut self) -> Result<PatId> {
        let path = self.path(Style::Expr)?;
        if self.punct("!") && opens(self.kind_at(self.at + 1)) {
            self.at += 1;
            let open = self.group();
            return Ok(self.push_pat(Pat::Macro { path, open }));
        }
        match self.kind() {
            TokenKind::OpenParen => {
                let open = self.group();
                let list = self.comma_list(open, |cursor| Ok(cursor.pat()?.0))?;
                Ok(self.push_pat(Pat::TupleStruct(path, list)))
            }
            TokenKind::OpenBrace => self.struct_pat(path),
            _ => Ok(self.push_pat(Pat::Path(path))),
        }
    }

    fn struct_pat(&mut self, path: super::PathId) -> Result<PatId> {
        let open = self.group();
        let mut rest = false;
        let fields = self.inside(open, |cursor| {
            let mut fields = Vec::new();
            while !cursor.done() {
                cursor.skip_attrs()?;
                if cursor.eat("..") {
                    rest = true;
                    break;
                }
                fields.push(cursor.field_pat()?);
                if !cursor.eat(",") {
                    break;
                }
            }
            Ok(fields)
        })?;
        let fields = self.list(&fields);
        Ok(self.push_pat(Pat::Struct { path, fields, rest }))
    }

    fn field_pat(&mut self) -> Result<u32> {
        let shorthand = !(matches!(self.kind(), TokenKind::Literal { .. } | TokenKind::Ident | TokenKind::RawIdent) && self.punct_at(self.at + 1, ":") && !self.punct_at(self.at + 1, "::"));
        let field = if shorthand {
            let pat = self.bind_pat()?;
            let Pat::Bind { name, .. } = self.body.pats[pat.0 as usize] else { unreachable!() };
            FieldPat { name, pat }
        } else {
            let name = self.at;
            self.at += 2;
            FieldPat { name, pat: self.pat()? }
        };
        self.body.field_pats.push(field);
        Ok(self.body.field_pats.len() as u32 - 1)
    }

    fn tuple_pat(&mut self) -> Result<PatId> {
        let open = self.group();
        let (pats, trailing) = self.inside(open, |cursor| {
            let mut pats = Vec::new();
            let mut trailing = false;
            while !cursor.done() {
                pats.push(cursor.pat()?.0);
                trailing = cursor.eat(",");
                if !trailing {
                    break;
                }
            }
            Ok((pats, trailing))
        })?;
        if pats.len() == 1 && !trailing && !matches!(self.body.pats[pats[0] as usize], Pat::Rest) {
            return Ok(self.push_pat(Pat::Paren(PatId(pats[0]))));
        }
        let list = self.list(&pats);
        Ok(self.push_pat(Pat::Tuple(list)))
    }

    /// Parses one literal token, or `true` or `false`.
    pub fn literal(&mut self) -> Result<super::ExprId> {
        let at = self.at;
        let lit = match self.kind() {
            TokenKind::Literal { kind, .. } => lit_kind(kind),
            TokenKind::Ident if self.src.text(at) == "true" => Lit::Bool(true),
            TokenKind::Ident if self.src.text(at) == "false" => Lit::Bool(false),
            _ => return self.fail("expected a literal"),
        };
        self.at += 1;
        Ok(self.push_expr(Expr::Lit(lit, at)))
    }
}

fn lit_kind(kind: rustc_lexer::LiteralKind) -> Lit {
    use rustc_lexer::LiteralKind as Kind;
    match kind {
        Kind::Int { .. } => Lit::Int,
        Kind::Float { .. } => Lit::Float,
        Kind::Char { .. } => Lit::Char,
        Kind::Byte { .. } => Lit::Byte,
        Kind::Str { .. } | Kind::RawStr { .. } => Lit::Str,
        Kind::ByteStr { .. } | Kind::RawByteStr { .. } => Lit::ByteStr,
        Kind::CStr { .. } | Kind::RawCStr { .. } => Lit::CStr,
    }
}
