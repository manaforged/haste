use anyhow::Result;
use rustc_lexer::TokenKind;

use super::cursor::{Cursor, is_name};
use super::{Args, Bound, GenericArg, Path, PathId, Segment, Ty};

/// Where a path is written, which decides how generic arguments open.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Expr,
    Type,
}

impl Cursor<'_> {
    /// Parses a path written in `style`.
    pub fn path(&mut self, style: Style) -> Result<PathId> {
        if self.punct("$") && self.word_at(self.at + 1) == "crate" {
            self.at += 1;
        }
        let qself = if self.punct("<") { Some(self.qself()?) } else { None };
        let global = qself.is_none() && self.eat("::");
        let mut segments = Vec::new();
        if qself.is_some() {
            self.expect("::")?;
        }
        loop {
            segments.push(self.segment(style)?);
            if !(self.punct("::") && is_name(self.kind_at(self.at + 2))) {
                break;
            }
            self.at += 2;
        }
        let segments = self.list(&segments);
        Ok(self.push_path(Path { qself, global, segments }))
    }

    fn qself(&mut self) -> Result<(super::TyId, Option<PathId>)> {
        self.expect("<")?;
        let ty = self.ty()?;
        let as_trait = if self.eat_word("as") { Some(self.path(Style::Type)?) } else { None };
        self.expect(">")?;
        Ok((ty, as_trait))
    }

    fn segment(&mut self, style: Style) -> Result<u32> {
        let name = self.name()?;
        let args = self.segment_args(style)?;
        self.body.segments.push(Segment { name, args });
        Ok(self.body.segments.len() as u32 - 1)
    }

    fn segment_args(&mut self, style: Style) -> Result<Option<Args>> {
        if self.punct("::") && self.punct_at(self.at + 2, "<") {
            self.at += 2;
            return Ok(Some(self.angle()?));
        }
        if style == Style::Expr {
            return Ok(None);
        }
        if self.punct("<") && !self.punct("<-") && !self.punct("<=") {
            return Ok(Some(self.angle()?));
        }
        if self.punct("::") && self.kind_at(self.at + 2) == TokenKind::OpenParen {
            self.at += 2;
        }
        if self.kind() == TokenKind::OpenParen {
            return Ok(Some(self.paren_args()?));
        }
        Ok(None)
    }

    fn paren_args(&mut self) -> Result<Args> {
        let open = self.at;
        let dots = self.punct_at(open + 1, "..") && self.src.pair(open) == open + 3;
        if dots {
            self.group();
            return Ok(Args::ReturnType);
        }
        self.group();
        let inputs = self.comma_list(open, |cursor| Ok(cursor.ty()?.0))?;
        let output = if self.eat("->") { Some(self.ty_no_bounds()?) } else { None };
        Ok(Args::Paren { inputs, output })
    }

    /// Parses `<...>` generic arguments.
    pub fn angle(&mut self) -> Result<Args> {
        self.expect("<")?;
        let mut args = Vec::new();
        while !self.punct(">") {
            args.push(self.generic_arg()?);
            if !self.eat(",") {
                break;
            }
        }
        self.expect(">")?;
        let list = self.list(&args);
        Ok(Args::Angle(list))
    }

    fn generic_arg(&mut self) -> Result<u32> {
        let arg = if let Some(lifetime) = self.lifetime() {
            GenericArg::Lifetime(lifetime)
        } else if self.kind() == TokenKind::OpenBrace || matches!(self.kind(), TokenKind::Literal { .. }) || self.punct("-") || self.is_word("true") || self.is_word("false") {
            GenericArg::Const(self.const_arg()?)
        } else if is_name(self.kind()) && self.binding_ahead() {
            self.binding()?
        } else {
            GenericArg::Ty(self.ty()?)
        };
        self.body.args.push(arg);
        Ok(self.body.args.len() as u32 - 1)
    }

    fn const_arg(&mut self) -> Result<super::ExprId> {
        if self.kind() == TokenKind::OpenBrace {
            return self.block_expr(None, super::BlockKind::Plain);
        }
        self.unary_operand()
    }

    fn binding_ahead(&self) -> bool {
        let next = self.at + 1;
        if self.punct_at(next, "<") {
            let mut depth = 0;
            let mut at = next;
            while at < self.end {
                if self.punct_at(at, "<") {
                    depth += 1;
                } else if self.punct_at(at, ">") && !self.punct_at(at.saturating_sub(1), "-") {
                    depth -= 1;
                    if depth == 0 {
                        return self.assoc_after(at + 1);
                    }
                } else if matches!(self.kind_at(at), TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace) {
                    at = self.src.pair(at);
                }
                at += 1;
            }
            return false;
        }
        self.assoc_after(next)
    }

    fn assoc_after(&self, at: u32) -> bool {
        (self.punct_at(at, "=") && !self.punct_at(at, "==")) || (self.punct_at(at, ":") && !self.punct_at(at, "::"))
    }

    fn binding(&mut self) -> Result<GenericArg> {
        let name = self.name()?;
        let args = if self.punct("<") { Some(self.angle()?) } else { None };
        if self.eat(":") {
            let bounds = self.bounds()?;
            return Ok(GenericArg::Constraint { name, bounds });
        }
        self.expect("=")?;
        let ty = if self.kind() == TokenKind::OpenBrace || matches!(self.kind(), TokenKind::Literal { .. }) {
            let value = self.const_arg()?;
            self.push_ty(Ty::Const(value))
        } else {
            self.ty()?
        };
        Ok(GenericArg::Binding { name, args, ty })
    }

    /// Parses a `+`-separated list of bounds.
    pub fn bounds(&mut self) -> Result<super::List> {
        let mut bounds = Vec::new();
        while self.bound_start() {
            bounds.push(self.bound()?);
            if !self.eat("+") {
                break;
            }
        }
        Ok(self.list(&bounds))
    }

    fn bound_start(&self) -> bool {
        matches!(self.kind(), TokenKind::Lifetime { .. } | TokenKind::OpenParen | TokenKind::OpenBracket) || self.punct("?") || self.punct("~") || self.punct("::") || self.punct("<") || self.punct("$") || is_name(self.kind())
    }

    fn bound(&mut self) -> Result<u32> {
        let bound = if let Some(lifetime) = self.lifetime() {
            Bound::Lifetime(lifetime)
        } else if self.kind() == TokenKind::OpenParen {
            let open = self.at;
            let inner = self.inside(open, |cursor| {
                let list = cursor.bounds()?;
                Ok(cursor.body.ids[list.start as usize])
            })?;
            return Ok(inner);
        } else if self.is_word("use") && self.punct_at(self.at + 1, "<") {
            let at = self.at;
            self.at += 1;
            self.angle()?;
            Bound::Use(at)
        } else {
            self.trait_bound()?
        };
        self.body.bounds.push(bound);
        Ok(self.body.bounds.len() as u32 - 1)
    }

    fn trait_bound(&mut self) -> Result<Bound> {
        let mut maybe = false;
        let mut constness = false;
        loop {
            if self.eat("?") {
                maybe = true;
            } else if self.eat("~") {
                self.eat_word("const");
                constness = true;
            } else if self.kind() == TokenKind::OpenBracket {
                self.group();
                constness = true;
            } else if self.is_word("const") || self.is_word("async") || self.is_word("unsafe") {
                constness |= self.is_word("const");
                self.at += 1;
            } else if self.is_word("for") && self.punct_at(self.at + 1, "<") {
                self.at += 1;
                self.angle()?;
            } else {
                break;
            }
        }
        if self.eat("!") {
            maybe = true;
        }
        let path = self.path(Style::Type)?;
        Ok(Bound::Trait { path, maybe, constness })
    }
}
