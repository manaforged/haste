//! Function bodies parsed into an arena of expressions, patterns and types, over the outline's tokens.

pub mod adt;
pub(crate) mod cursor;
mod expr;
pub mod literal;
mod pat;
pub mod sig;
mod path;
mod stmt;
mod ty;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::outline::source::Source;

use cursor::Cursor;

/// An expression in a body's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExprId(pub u32);

/// A pattern in a body's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PatId(pub u32);

/// A type in a body's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TyId(pub u32);

/// A path in a body's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PathId(pub u32);

/// A run of ids in one of a body's list pools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct List {
    pub start: u32,
    pub len: u32,
}

/// A literal, by the kind of its token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lit {
    Int,
    Float,
    Char,
    Byte,
    Str,
    ByteStr,
    CStr,
    Bool(bool),
}

/// A unary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
    Deref,
}

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
}

/// One arm of a `match`.
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub pat: PatId,
    pub guard: Option<ExprId>,
    pub body: ExprId,
}

/// One field of a struct literal: its name or index token and its value.
#[derive(Clone, Copy, Debug)]
pub struct FieldInit {
    pub name: u32,
    pub value: ExprId,
}

/// One closure parameter.
#[derive(Clone, Copy, Debug)]
pub struct Param {
    pub pat: PatId,
    pub ty: Option<TyId>,
}

/// An expression.
#[derive(Clone, Copy, Debug)]
pub enum Expr {
    Lit(Lit, u32),
    Path(PathId),
    Unary(UnOp, ExprId),
    Binary(BinOp, ExprId, ExprId),
    Assign(ExprId, ExprId),
    AssignOp(BinOp, ExprId, ExprId),
    Call(ExprId, List),
    Method { receiver: ExprId, name: u32, args: Option<List>, params: List },
    Field(ExprId, u32),
    TupleField(ExprId, u32),
    Index(ExprId, ExprId),
    Cast(ExprId, TyId),
    Ref { mutable: bool, raw: bool, expr: ExprId },
    Block { stmts: List, tail: Option<ExprId>, label: Option<u32>, kind: BlockKind },
    If(ExprId, ExprId, Option<ExprId>),
    Let(PatId, ExprId),
    While { label: Option<u32>, cond: ExprId, body: ExprId },
    Loop { label: Option<u32>, body: ExprId },
    For { label: Option<u32>, pat: PatId, iter: ExprId, body: ExprId },
    Match(ExprId, List),
    Closure { params: List, ret: Option<TyId>, body: ExprId, capture: bool },
    Tuple(List),
    Array(List),
    Repeat(ExprId, ExprId),
    Struct { path: PathId, fields: List, base: Option<ExprId>, rest: bool },
    Range { from: Option<ExprId>, to: Option<ExprId>, inclusive: bool },
    Try(ExprId),
    Await(ExprId),
    Return(Option<ExprId>),
    Break(Option<u32>, Option<ExprId>),
    Continue(Option<u32>),
    Yield(Option<ExprId>),
    Become(ExprId),
    Macro { path: PathId, open: u32 },
    Paren(ExprId),
    Underscore,
    FormatArgs(u32),
}

/// What kind of block a block expression is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Plain,
    Unsafe,
    Const,
    Async,
    Gen,
    Try,
}

/// A statement.
#[derive(Clone, Copy, Debug)]
pub enum Stmt {
    Let { pat: PatId, ty: Option<TyId>, init: Option<ExprId>, otherwise: Option<ExprId> },
    Expr(ExprId, bool),
    Item(u32, u32),
    Macro { path: PathId, open: u32, semi: bool },
    Empty,
}

/// A pattern.
#[derive(Clone, Copy, Debug)]
pub enum Pat {
    Wild,
    Rest,
    Bind { name: u32, by_ref: bool, mutable: bool, sub: Option<PatId> },
    Lit(ExprId),
    Range { from: Option<ExprId>, to: Option<ExprId>, inclusive: bool },
    Path(PathId),
    TupleStruct(PathId, List),
    Struct { path: PathId, fields: List, rest: bool },
    Tuple(List),
    Slice(List),
    Ref { mutable: bool, pat: PatId },
    Box(PatId),
    Or(List),
    Macro { path: PathId, open: u32 },
    Paren(PatId),
    Never,
    Deref(PatId),
}

/// One field of a struct pattern.
#[derive(Clone, Copy, Debug)]
pub struct FieldPat {
    pub name: u32,
    pub pat: PatId,
}

/// A type.
#[derive(Clone, Copy, Debug)]
pub enum Ty {
    Path(PathId),
    Ref { lifetime: Option<u32>, mutable: bool, ty: TyId },
    Ptr { mutable: bool, ty: TyId },
    Slice(TyId),
    Array(TyId, ExprId),
    Tuple(List),
    Fn { params: List, ret: Option<TyId>, variadic: bool },
    ImplTrait(u32, List),
    DynTrait(List),
    Never,
    Infer,
    Macro { path: PathId, open: u32 },
    Paren(TyId),
    Pat(TyId, PatId),
    Const(ExprId),
}

/// A bound in `impl`, `dyn` or a where clause: a trait path with its modifiers and `for<..>` lifetimes, or a lifetime.
#[derive(Clone, Copy, Debug)]
pub enum Bound {
    Trait { path: PathId, maybe: bool, constness: bool },
    Lifetime(u32),
    Use(u32),
}

/// One segment of a path, with its generic arguments.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub name: u32,
    pub args: Option<Args>,
}

/// The generic arguments of a path segment.
#[derive(Clone, Copy, Debug)]
pub enum Args {
    Angle(List),
    Paren { inputs: List, output: Option<TyId> },
    ReturnType,
}

/// One generic argument.
#[derive(Clone, Copy, Debug)]
pub enum GenericArg {
    Ty(TyId),
    Lifetime(u32),
    Const(ExprId),
    Binding { name: u32, args: Option<Args>, ty: TyId },
    Constraint { name: u32, bounds: List },
}

/// A path: an optional `<T as Trait>` qualifier, a leading `::`, and its segments.
#[derive(Clone, Copy, Debug)]
pub struct Path {
    pub qself: Option<(TyId, Option<PathId>)>,
    pub global: bool,
    pub segments: List,
}

/// A node that carries outer attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    Stmt(u32),
    Arm(u32),
    Field(u32),
    Expr(ExprId),
}

/// The parsed body of one function, with its arenas and list pools.
#[derive(Debug, Default)]
pub struct Body {
    pub exprs: Vec<Expr>,
    pub pats: Vec<Pat>,
    pub tys: Vec<Ty>,
    pub paths: Vec<Path>,
    pub stmts: Vec<Stmt>,
    pub segments: Vec<Segment>,
    pub args: Vec<GenericArg>,
    pub bounds: Vec<Bound>,
    pub arms: Vec<Arm>,
    pub fields: Vec<FieldInit>,
    pub field_pats: Vec<FieldPat>,
    pub params: Vec<Param>,
    pub ids: Vec<u32>,
    pub attrs: Vec<(Node, u32, u32)>,
    pub root: Option<ExprId>,
    /// The function's parameter patterns, parsed into this arena when one of them is more than a name.
    pub fn_params: Vec<PatId>,
}

impl Body {
    /// The expression `id`.
    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs[id.0 as usize]
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
