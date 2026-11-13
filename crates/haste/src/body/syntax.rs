#![expect(dead_code, reason = "the syntax tree records every part of the syntax it parses; sema reads the parts it supports and prints the rest in errors")]

/// An expression in a body's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(pub u32);

/// A pattern in a body's arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    OffsetOf { ty: TyId, fields: List },
}

/// What kind of block a block expression is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Plain,
    /// The output of a macro call in expression position, whose temporaries belong to the enclosing statement as if the call were its output.
    Expansion,
    Unsafe,
    Const,
    Async,
    /// An `async move` block, which captures by value.
    AsyncMove,
    Gen,
    Try,
}

/// A statement.
#[derive(Clone, Copy, Debug)]
pub enum Stmt {
    Let { pat: PatId, ty: Option<TyId>, init: Option<ExprId>, otherwise: Option<ExprId>, extend: bool },
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
