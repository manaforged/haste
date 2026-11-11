use anyhow::Result;
use rustc_lexer::TokenKind;

use crate::outline::source::Source;

use super::cursor::Cursor;
use super::sig::Generics;
use super::{Body, ExprId, TyId};

/// How widely a struct or union field is visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldVis {
    Public,
    Crate,
    Private,
}

/// One field: its name token, or none in a tuple variant, its type, its visibility, and where its attributes start.
#[derive(Clone, Copy, Debug)]
pub struct FieldDef {
    pub name: Option<u32>,
    pub ty: TyId,
    pub vis: FieldVis,
    pub attrs: Option<u32>,
}

/// How a variant holds its fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Unit,
    Tuple,
    Named,
}

/// One variant of an enum, or the single variant of a struct or union.
#[derive(Clone, Debug)]
pub struct VariantDef {
    pub name: u32,
    pub shape: Shape,
    pub fields: Vec<FieldDef>,
    pub discriminant: Option<ExprId>,
    pub attrs: Option<u32>,
    pub index: u32,
}

/// A struct, union or enum definition: its generics and variants, in its own arena.
#[derive(Debug, Default)]
pub struct AdtDef {
    pub body: Body,
    pub generics: Generics,
    pub variants: Vec<VariantDef>,
    /// The tokens of each `where` clause, after the keyword.
    pub wheres: Vec<(u32, u32)>,
}

impl AdtDef {
    /// Drops the variants and fields whose attributes `enabled` rejects, as `#[cfg]` configures them out.
    pub fn configure(&mut self, src: &Source, enabled: &dyn Fn(&[crate::outline::item::Attr]) -> bool) {
        let kept = |at: Option<u32>| {
            let mut attrs = Vec::new();
            at.is_none_or(|at| crate::outline::attr::attrs(src, at, false, &mut attrs).is_ok() && enabled(&attrs))
        };
        self.variants.retain(|variant| kept(variant.attrs));
        for variant in &mut self.variants {
            variant.fields.retain(|field| kept(field.attrs));
        }
    }

    /// The position among the variants that are configured in of the variant whose declaration index is `index`.
    pub fn position(&self, index: u32) -> Option<usize> {
        self.variants.iter().position(|variant| variant.index == index)
    }
}

/// Parses the struct, union or enum whose name is token `name`, reading no further than `to`.
pub fn adt(src: &Source, name: u32, to: u32, is_enum: bool) -> Result<AdtDef> {
    let mut cursor = Cursor::new(src, name + 1, to);
    let params = cursor.generic_params()?;
    let mut wheres = Vec::new();
    let mut predicates = cursor.where_span(&mut wheres)?;
    let variants = if is_enum {
        let open = cursor.expect_kind(TokenKind::OpenBrace)?;
        let mut variants = Vec::new();
        cursor.comma_list(open, |cursor| {
            let index = variants.len() as u32;
            variants.push(VariantDef { index, ..cursor.variant()? });
            Ok(0)
        })?;
        variants
    } else {
        let (shape, fields) = cursor.fields()?;
        predicates.extend(cursor.where_span(&mut wheres)?);
        vec![VariantDef { name, shape, fields, discriminant: None, attrs: None, index: 0 }]
    };
    Ok(AdtDef { body: cursor.body, generics: Generics { params, predicates, ..Default::default() }, variants, wheres })
}

impl Cursor<'_> {
    fn where_span(&mut self, into: &mut Vec<(u32, u32)>) -> Result<Vec<super::sig::Predicate>> {
        let start = self.at + 1;
        let predicates = self.where_clause()?;
        if self.at > start {
            into.push((start, self.at));
        }
        Ok(predicates)
    }

    fn variant(&mut self) -> Result<VariantDef> {
        let attrs = self.skip_attrs()?.map(|(start, _)| start);
        self.visibility();
        let name = self.name()?;
        let (shape, fields) = self.fields()?;
        let discriminant = if self.eat("=") { Some(self.expr()?) } else { None };
        Ok(VariantDef { name, shape, fields, discriminant, attrs, index: 0 })
    }

    fn fields(&mut self) -> Result<(Shape, Vec<FieldDef>)> {
        let shape = match self.kind() {
            TokenKind::OpenParen => Shape::Tuple,
            TokenKind::OpenBrace => Shape::Named,
            _ => return Ok((Shape::Unit, Vec::new())),
        };
        let open = self.at;
        let mut fields = Vec::new();
        self.comma_list(open, |cursor| {
            let attrs = cursor.skip_attrs()?.map(|(start, _)| start);
            let vis = cursor.visibility();
            let name = if shape == Shape::Named {
                let name = cursor.name()?;
                cursor.expect(":")?;
                Some(name)
            } else {
                None
            };
            fields.push(FieldDef { name, ty: cursor.ty()?, vis, attrs });
            Ok(0)
        })?;
        Ok((shape, fields))
    }

    fn visibility(&mut self) -> FieldVis {
        if !self.eat_word("pub") {
            return FieldVis::Private;
        }
        if self.kind() != TokenKind::OpenParen {
            return FieldVis::Public;
        }
        let close = self.src.pair(self.at);
        let is_self = close == self.at + 2 && self.src.text(self.at + 1) == "self";
        self.group();
        if is_self { FieldVis::Private } else { FieldVis::Crate }
    }
}
