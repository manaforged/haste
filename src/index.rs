pub mod headers;
pub mod oracle;
mod scan;

pub use scan::{scan, scan_all};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Head {
    Name(String),
    Ref,
    Ptr,
    Slice,
    Array,
    Tuple,
    Fn,
    Dyn,
    Never,
    Param,
    Unknown,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Header {
    pub trait_name: String,
    pub head: Head,
}

#[derive(Clone, Debug)]
pub struct Alias {
    pub name: String,
    pub head: Head,
}

#[derive(Clone, Debug)]
pub struct MacroUse {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MacroFacts {
    pub name: String,
    pub declares_alias: bool,
    pub calls: Vec<MacroUse>,
}

#[derive(Default)]
pub struct Scanned {
    pub entries: Vec<Entry>,
    pub headers: Vec<Header>,
    pub aliases: Vec<Alias>,
    pub macros: Vec<MacroFacts>,
    pub uses: Vec<MacroUse>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Kind {
    Const,
    Enum,
    ExternBlock,
    ExternCrate,
    Function,
    Impl,
    Macro2,
    MacroCall,
    MacroRules,
    Mod,
    Static,
    Struct,
    Trait,
    TypeAlias,
    Union,
    Use,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Entry {
    pub path: String,
    pub kind: Kind,
    pub name: String,
}
