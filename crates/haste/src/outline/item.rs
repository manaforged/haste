use crate::graph::cfg::Cfg;

/// What an item declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ItemKind {
    Mod,
    Use,
    ExternCrate,
    Fn,
    Struct,
    Enum,
    Union,
    Trait,
    TraitAlias,
    Impl,
    Type,
    Const,
    Static,
    MacroRules,
    Macro,
    MacroCall,
    ExternBlock,
}

/// What a procedural macro function is invoked as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcKind {
    Bang,
    Attribute,
    Derive,
}

/// Each procedural macro attribute with the kind of macro it makes its function.
pub const PROC_MACROS: [(&str, ProcKind); 3] = [("proc_macro", ProcKind::Bang), ("proc_macro_attribute", ProcKind::Attribute), ("proc_macro_derive", ProcKind::Derive)];

/// An item's visibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vis {
    Private,
    Pub,
    Crate,
    Super,
    SelfMod,
    In(u32),
}

/// An attribute the compiler acts on. Others are kept as `Other` with the token of their path.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Cfg(Cfg),
    CfgAttr(Cfg, Vec<Attr>),
    Path(String),
    MacroUse,
    MacroExport(bool),
    Derive(u32),
    NoStd,
    NoCore,
    NoImplicitPrelude,
    Other(u32),
}

/// One item, as token positions in its file.
#[derive(Clone, Debug)]
pub struct Item {
    pub kind: ItemKind,
    pub name: Option<u32>,
    pub vis: Vis,
    pub attrs: Vec<Attr>,
    pub lead: u32,
    pub start: u32,
    pub end: u32,
    pub body: Option<u32>,
    pub parent: Option<u32>,
}

/// The last segment of a use path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf {
    Name(Option<u32>),
    Underscore,
    Glob,
}

/// One flattened import of a use tree: `path` names tokens, and `item` is the `use` item.
#[derive(Clone, Debug)]
pub struct Import {
    pub item: u32,
    pub absolute: bool,
    pub path: Vec<u32>,
    pub leaf: Leaf,
}

/// The items of one file.
#[derive(Clone, Debug, Default)]
pub struct Items {
    pub inner: Vec<Attr>,
    pub items: Vec<Item>,
    pub imports: Vec<Import>,
}
