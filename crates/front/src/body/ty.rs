use anyhow::Result;
use rustc_lexer::TokenKind;

use super::cursor::Cursor;
use super::path::Style;
use super::{Ty, TyId};

impl Cursor<'_> {
    /// Parses a type, with `+` bounds after a bare trait path.
    pub fn ty(&mut self) -> Result<TyId> {
        self.ty_no_bounds()
    }

    /// Parses a type that stops before `+`.
    pub fn ty_no_bounds(&mut self) -> Result<TyId> {
        match self.kind() {
            TokenKind::OpenParen => return self.paren_ty(),
            TokenKind::OpenBracket => return self.bracket_ty(),
            _ => {}
        }
        if self.eat("!") {
            return Ok(self.push_ty(Ty::Never));
        }
        if self.punct("&") {
            return self.ref_ty();
        }
        if self.eat("*") {
            let mutable = self.eat_word("mut");
            if !mutable {
                self.eat_word("const");
            }
            let ty = self.ty_no_bounds()?;
            return Ok(self.push_ty(Ty::Ptr { mutable, ty }));
        }
        if self.word() == "_" {
            self.at += 1;
            return Ok(self.push_ty(Ty::Infer));
        }
        self.word_ty()
    }

    fn word_ty(&mut self) -> Result<TyId> {
        match self.word() {
            "impl" => {
                let at = self.at;
                self.at += 1;
                let bounds = self.bounds()?;
                Ok(self.push_ty(Ty::ImplTrait(at, bounds)))
            }
            "dyn" => {
                self.at += 1;
                let bounds = self.bounds()?;
                Ok(self.push_ty(Ty::DynTrait(bounds)))
            }
            "fn" | "unsafe" | "extern" | "safe" => self.fn_ty(),
            "for" if self.punct_at(self.at + 1, "<") => {
                self.at += 1;
                self.angle()?;
                self.ty_no_bounds()
            }
            _ if self.punct("?") || matches!(self.kind(), TokenKind::Lifetime { .. }) => {
                let bounds = self.bounds()?;
                Ok(self.push_ty(Ty::DynTrait(bounds)))
            }
            _ => self.path_ty(),
        }
    }

    fn path_ty(&mut self) -> Result<TyId> {
        let path = self.path(Style::Type)?;
        if self.punct("!") && matches!(self.kind_at(self.at + 1), TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace) {
            self.at += 1;
            let open = self.group();
            return Ok(self.push_ty(Ty::Macro { path, open }));
        }
        let ty = self.push_ty(Ty::Path(path));
        if self.is_word("is") {
            self.at += 1;
            let pat = self.pat_no_alt()?;
            return Ok(self.push_ty(Ty::Pat(ty, pat)));
        }
        Ok(ty)
    }

    fn ref_ty(&mut self) -> Result<TyId> {
        let double = self.punct("&&");
        self.at += 1;
        let lifetime = self.lifetime();
        let mutable = self.eat_word("mut");
        let ty = if double { self.ref_ty_inner()? } else { self.ty_no_bounds()? };
        let outer = self.push_ty(Ty::Ref { lifetime, mutable, ty });
        Ok(outer)
    }

    fn ref_ty_inner(&mut self) -> Result<TyId> {
        let lifetime = self.lifetime();
        let mutable = self.eat_word("mut");
        let inner = self.ty_no_bounds()?;
        Ok(self.push_ty(Ty::Ref { lifetime, mutable, ty: inner }))
    }

    fn paren_ty(&mut self) -> Result<TyId> {
        let open = self.at;
        let (types, trailing) = self.inside(open, |cursor| {
            let mut types = Vec::new();
            let mut trailing = false;
            while !cursor.done() {
                types.push(cursor.ty()?.0);
                trailing = cursor.eat(",");
                if !trailing {
                    break;
                }
            }
            Ok((types, trailing))
        })?;
        if types.len() == 1 && !trailing {
            let inner = TyId(types[0]);
            if self.punct("+") {
                self.at += 1;
                self.bounds()?;
            }
            return Ok(self.push_ty(Ty::Paren(inner)));
        }
        let list = self.list(&types);
        Ok(self.push_ty(Ty::Tuple(list)))
    }

    fn bracket_ty(&mut self) -> Result<TyId> {
        let open = self.at;
        let (element, len) = self.inside(open, |cursor| {
            let element = cursor.ty()?;
            let len = if cursor.eat(";") { Some(cursor.expr()?) } else { None };
            Ok((element, len))
        })?;
        Ok(self.push_ty(match len {
            Some(len) => Ty::Array(element, len),
            None => Ty::Slice(element),
        }))
    }

    fn fn_ty(&mut self) -> Result<TyId> {
        self.fn_qualifiers()?;
        let open = self.expect_kind(TokenKind::OpenParen)?;
        self.at = open;
        let mut variadic = false;
        let params = self.comma_list(open, |cursor| cursor.fn_param(&mut variadic))?;
        let ret = if self.eat("->") { Some(self.ty_no_bounds()?) } else { None };
        Ok(self.push_ty(Ty::Fn { params, ret, variadic }))
    }

    fn fn_qualifiers(&mut self) -> Result<()> {
        while !self.eat_word("fn") {
            if self.eat_word("extern") {
                if matches!(self.kind(), TokenKind::Literal { .. }) {
                    self.at += 1;
                }
            } else if !(self.eat_word("unsafe") || self.eat_word("safe")) {
                return self.fail("expected `fn`");
            }
        }
        Ok(())
    }

    fn fn_param(&mut self, variadic: &mut bool) -> Result<u32> {
        self.skip_attrs()?;
        if self.eat("...") {
            *variadic = true;
            return Ok(self.push_ty(Ty::Infer).0);
        }
        let named = (self.ident() || self.word() == "_") && self.punct_at(self.at + 1, ":") && !self.punct_at(self.at + 1, "::");
        if named {
            self.at += 2;
        }
        Ok(self.ty()?.0)
    }
}
