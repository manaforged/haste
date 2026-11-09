use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::outline::source::{Source, opens};

use super::cursor::Cursor;
use super::{Body, List, PatId, TyId};

/// What kind of generic parameter a declaration has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenericKind {
    Type,
    Lifetime,
    Const(TyId),
}

/// One generic parameter: its name token, kind, bounds and default.
#[derive(Clone, Copy, Debug)]
pub struct Generic {
    pub name: u32,
    pub kind: GenericKind,
    pub bounds: List,
    pub default: Option<TyId>,
    /// The tokens of the parameter and its bounds, before any default.
    pub span: (u32, u32),
}

/// One where-clause predicate: a type or lifetime and its bounds.
#[derive(Clone, Copy, Debug)]
pub struct Predicate {
    pub ty: Option<TyId>,
    pub bounds: List,
}

/// How a method takes `self`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfKind {
    Value,
    Ref,
    RefMut,
    Typed(TyId),
}

/// A function header: generics, parameters, return type and where clause, in its own arena.
#[derive(Debug, Default)]
pub struct Signature {
    pub body: Body,
    pub name: u32,
    pub generics: Vec<Generic>,
    pub receiver: Option<(SelfKind, bool)>,
    pub params: Vec<(PatId, TyId)>,
    pub ret: Option<TyId>,
    pub predicates: Vec<Predicate>,
    pub variadic: bool,
    /// Whether the return type holds `impl Trait`, which the body's value decides.
    pub opaque: bool,
    /// Whether the function is `async`, so it returns a future of its declared return type.
    pub is_async: bool,
    /// The first token of each parameter's pattern.
    pub param_starts: Vec<u32>,
}

/// Generic parameters and a where clause, for items other than functions.
#[derive(Debug, Default)]
pub struct Generics {
    pub params: Vec<Generic>,
    pub predicates: Vec<Predicate>,
    /// The supertrait bounds of a trait, after its `:`.
    pub supers: List,
}

impl Cursor<'_> {
    /// Parses `<...>` generic parameters when they open here.
    pub fn generic_params(&mut self) -> Result<Vec<Generic>> {
        let mut params = Vec::new();
        if !self.eat("<") {
            return Ok(params);
        }
        while !self.punct(">") {
            self.skip_attrs()?;
            params.push(self.generic_param()?);
            if !self.eat(",") {
                break;
            }
        }
        self.expect(">")?;
        Ok(params)
    }

    fn generic_param(&mut self) -> Result<Generic> {
        let start = self.at;
        if let Some(name) = self.lifetime() {
            let bounds = if self.eat(":") { self.bounds()? } else { List::default() };
            return Ok(Generic { name, kind: GenericKind::Lifetime, bounds, default: None, span: (start, self.at) });
        }
        if self.eat_word("const") {
            let name = self.name()?;
            self.expect(":")?;
            let ty = self.ty()?;
            let span = (start, self.at);
            let default = if self.eat("=") { Some(self.const_default()?) } else { None };
            return Ok(Generic { name, kind: GenericKind::Const(ty), bounds: List::default(), default, span });
        }
        let name = self.name()?;
        let bounds = if self.eat(":") { self.bounds()? } else { List::default() };
        let span = (start, self.at);
        let default = if self.eat("=") { Some(self.ty()?) } else { None };
        Ok(Generic { name, kind: GenericKind::Type, bounds, default, span })
    }

    fn const_default(&mut self) -> Result<TyId> {
        let value = if self.kind() == TokenKind::OpenBrace { self.block_expr(None, super::BlockKind::Plain)? } else { self.unary_operand()? };
        Ok(self.push_ty(super::Ty::Const(value)))
    }

    /// Parses a `where` clause when one starts here.
    pub fn where_clause(&mut self) -> Result<Vec<Predicate>> {
        let mut predicates = Vec::new();
        if !self.eat_word("where") {
            return Ok(predicates);
        }
        while !self.done() && self.kind() != TokenKind::OpenBrace && !self.punct(";") && !self.punct("=") {
            predicates.push(self.predicate()?);
            if !self.eat(",") {
                break;
            }
        }
        Ok(predicates)
    }

    fn predicate(&mut self) -> Result<Predicate> {
        if self.lifetime().is_some() {
            self.expect(":")?;
            return Ok(Predicate { ty: None, bounds: self.bounds()? });
        }
        if self.is_word("for") && self.punct_at(self.at + 1, "<") {
            self.at += 1;
            self.angle()?;
        }
        let ty = self.ty()?;
        self.expect(":")?;
        Ok(Predicate { ty: Some(ty), bounds: self.bounds()? })
    }

    fn receiver(&mut self) -> Result<Option<(SelfKind, bool)>> {
        let at = self.at;
        let reference = self.punct("&");
        if reference {
            self.at += 1;
            self.lifetime();
        }
        let mutable = self.eat_word("mut");
        if !self.is_word("self") || self.punct_at(self.at + 1, "::") {
            self.at = at;
            return Ok(None);
        }
        self.at += 1;
        let kind = match (reference, mutable) {
            (true, true) => SelfKind::RefMut,
            (true, false) => SelfKind::Ref,
            _ if self.eat(":") => SelfKind::Typed(self.ty()?),
            _ => SelfKind::Value,
        };
        Ok(Some((kind, mutable && !reference)))
    }

    fn fn_params(&mut self, sig: &mut Signature) -> Result<()> {
        let open = self.expect_kind(TokenKind::OpenParen)?;
        self.at = open;
        let mut params = Vec::new();
        let mut receiver = None;
        let mut variadic = false;
        let mut starts = Vec::new();
        self.comma_list(open, |cursor| {
            cursor.skip_attrs()?;
            if receiver.is_none() && params.is_empty() && let Some(found) = cursor.receiver()? {
                receiver = Some(found);
                return Ok(0);
            }
            if cursor.eat("...") {
                variadic = true;
                return Ok(0);
            }
            starts.push(cursor.at);
            let pat = cursor.pat_no_alt()?;
            cursor.expect(":")?;
            params.push((pat, cursor.ty()?));
            Ok(0)
        })?;
        sig.params = params;
        sig.receiver = receiver;
        sig.variadic = variadic;
        sig.param_starts = starts;
        Ok(())
    }
}

/// Parses the header of the function item whose tokens run from `from` to `to` in `src`.
pub fn signature(src: &Source, from: u32, to: u32) -> Result<Signature> {
    let mut cursor = Cursor::new(src, from, to);
    let mut is_async = false;
    while !cursor.done() && !cursor.is_word("fn") {
        is_async |= cursor.is_word("async");
        cursor.at += if cursor.kind() == TokenKind::OpenParen { cursor.src.pair(cursor.at) + 1 - cursor.at } else { 1 };
    }
    cursor.at += 1;
    let mut sig = Signature { name: cursor.name()?, ..Signature::default() };
    sig.generics = cursor.generic_params()?;
    let first = cursor.body.tys.len();
    cursor.fn_params(&mut sig)?;
    for ty in &cursor.body.tys[first..] {
        if let super::Ty::ImplTrait(name, bounds) = *ty {
            sig.generics.push(Generic { name, kind: GenericKind::Type, bounds, default: None, span: (name, name + 1) });
        }
    }
    let ret_first = cursor.body.tys.len();
    sig.ret = if cursor.eat("->") { Some(cursor.ty()?) } else { None };
    sig.opaque = is_async || cursor.body.tys[ret_first..].iter().any(|ty| matches!(ty, super::Ty::ImplTrait(..)));
    sig.is_async = is_async;
    sig.predicates = cursor.where_clause()?;
    sig.body = cursor.body;
    Ok(sig)
}

/// Parses the generics and where clause of an item whose generics open at token `at`, parsing no further than `to`.
pub fn generics(src: &Source, at: u32, to: u32) -> Result<(Body, Generics)> {
    let mut cursor = Cursor::new(src, at, to);
    let params = cursor.generic_params()?;
    let supers = if cursor.eat(":") { cursor.bounds()? } else { List::default() };
    let predicates = cursor.where_clause()?;
    Ok((cursor.body, Generics { params, predicates, supers }))
}

/// An impl header: generics, the trait it implements if any, the self type and the where clause, in its own arena.
#[derive(Debug)]
pub struct ImplHeader {
    pub body: Body,
    pub generics: Vec<Generic>,
    pub negative: bool,
    pub trait_ref: Option<TyId>,
    pub self_ty: TyId,
    pub predicates: Vec<Predicate>,
}

/// Parses the header of the impl item whose tokens run from `from` to `to` in `src`.
pub fn impl_header(src: &Source, from: u32, to: u32) -> Result<ImplHeader> {
    let mut cursor = Cursor::new(src, from, to);
    while !cursor.done() && !cursor.is_word("impl") {
        cursor.at += 1;
    }
    cursor.at += 1;
    let generics = cursor.generic_params()?;
    cursor.eat_word("const");
    let negative = cursor.eat("!");
    let first = cursor.ty_no_bounds()?;
    let (trait_ref, self_ty) = if cursor.eat_word("for") { (Some(first), cursor.ty_no_bounds()?) } else { (None, first) };
    let predicates = cursor.where_clause()?;
    Ok(ImplHeader { body: cursor.body, generics, negative, trait_ref, self_ty, predicates })
}

/// Parses the type of the constant item whose name is token `name`, as the return type of a header with no parameters.
pub fn const_signature(src: &Source, name: u32, to: u32) -> Result<Signature> {
    let mut cursor = Cursor::new(src, name + 1, to);
    cursor.expect(":")?;
    let ret = cursor.ty()?;
    Ok(Signature { body: cursor.body, name, ret: Some(ret), ..Signature::default() })
}

/// The token after the `=` of the constant or static item whose name is token `name`, searching no further than `to`.
pub fn initializer(src: &Source, name: u32, to: u32) -> Option<u32> {
    let mut at = name + 1;
    while at < to {
        match src.kind(at) {
            TokenKind::Eq => return Some(at + 1),
            kind if opens(kind) => at = src.pair(at) + 1,
            _ => at += 1,
        }
    }
    None
}
