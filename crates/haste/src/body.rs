//! Function bodies parsed into an arena of expressions, patterns and types, over the outline's tokens.

pub mod adt;
pub(crate) mod cursor;
mod expr;
pub mod literal;
mod pat;
pub mod sig;
mod path;
mod stmt;
mod syntax;
mod ty;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::outline::source::Source;

use cursor::Cursor;

pub use syntax::*;

impl Body {
    /// The expression `id`.
    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id.0 as usize]
    }

    /// Whether the resolution of call callee `id` names the function the call runs, which is false for a method call whose result is called.
    pub fn names_callee(&self, id: ExprId) -> bool {
        !matches!(self.expr(id), Expr::Method { .. })
    }

    /// The pattern `id`.
    pub fn pat(&self, id: PatId) -> &Pat {
        &self.pats[id.0 as usize]
    }

    /// Whether the parameter pattern `id` is a plain name or `_`, which binds the argument without taking it apart.
    pub fn plain_param(&self, id: PatId) -> bool {
        matches!(self.pat(id), Pat::Bind { sub: None, by_ref: false, .. } | Pat::Wild)
    }

    /// The type `id`.
    pub fn ty(&self, id: TyId) -> &Ty {
        &self.tys[id.0 as usize]
    }

    /// The path `id`.
    pub fn path(&self, id: PathId) -> &Path {
        &self.paths[id.0 as usize]
    }

    /// The ids in `list`.
    pub fn list(&self, list: List) -> &[u32] {
        &self.ids[list.start as usize..(list.start + list.len) as usize]
    }
}

/// What a body is parsed with besides its tokens: the block each macro call became, and the environment that `#[cfg]` attributes read.
#[derive(Clone, Copy, Default)]
pub struct Config<'a> {
    pub calls: Option<&'a HashMap<u32, u32>>,
    pub env: Option<&'a crate::outline::tree::Env>,
}

/// Parses the block that opens at token `open` of `src` with `config`, leaving out the statements, arms, fields and arguments that `#[cfg]` turns off.
/// Whether a `break` or `continue` with `label` jumps to the loop or labeled block `frame`, given as its own label and whether it is a block; one with no label skips labeled blocks.
pub fn jumps_to(src: &Source, (own, block): (Option<u32>, bool), label: Option<u32>) -> bool {
    match label {
        Some(wanted) => own.is_some_and(|own| src.text(own) == src.text(wanted)),
        None => !block,
    }
}

pub fn parse(src: &Source, open: u32, config: Config<'_>) -> Result<Body> {
    let mut cursor = Cursor::new(src, open, src.pair(open) + 1);
    cursor.config = config;
    let root = cursor.block_expr(None, BlockKind::Plain)?;
    cursor.body.root = Some(root);
    Ok(cursor.body)
}

/// Adds to `body` the parameter patterns that start at tokens `starts` of `src`.
pub fn parse_params(src: &Source, body: Body, starts: &[u32]) -> Result<Body> {
    let mut cursor = Cursor::new(src, 0, src.len());
    cursor.body = body;
    for &start in starts {
        cursor.at = start;
        let pat = cursor.pat_no_alt()?;
        cursor.body.fn_params.push(pat);
    }
    Ok(cursor.body)
}

/// Parses the tokens from `from` up to `to` of `src` as one expression, with expanded macro calls as in [`parse`].
pub fn parse_expr(src: &Source, from: u32, to: u32, config: Config<'_>) -> Result<Body> {
    let mut cursor = Cursor::new(src, from, to);
    cursor.config = config;
    let root = cursor.expr()?;
    cursor.finish()?;
    cursor.body.root = Some(root);
    Ok(cursor.body)
}

/// Parses the tokens from `from` up to `to` of `src` as one type.
pub fn parse_ty(src: &Source, from: u32, to: u32) -> Result<(Body, TyId)> {
    let mut cursor = Cursor::new(src, from, to);
    let ty = cursor.ty()?;
    cursor.finish()?;
    Ok((cursor.body, ty))
}

/// A macro fragment kind that the body parser measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fragment {
    Ty,
    Path,
    Pat,
    PatParam,
    Expr,
}

/// The end of the `fragment` that starts at token `at` of `src`, parsing no further than `end`.
pub fn fragment_end(src: &Source, fragment: Fragment, at: u32, end: u32) -> Option<u32> {
    let mut cursor = Cursor::new(src, at, end);
    let parsed = match fragment {
        Fragment::Ty => cursor.ty().is_ok(),
        Fragment::Path => cursor.path(path::Style::Type).is_ok(),
        Fragment::Pat => cursor.pat().is_ok(),
        Fragment::PatParam => cursor.pat_no_alt().is_ok(),
        Fragment::Expr => cursor.expr().is_ok(),
    };
    (parsed && cursor.at > at).then_some(cursor.at)
}
