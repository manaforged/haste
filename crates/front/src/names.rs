pub mod collect;
pub mod path;
pub mod resolve;
mod want;

use std::cell::RefCell;

use rustc_hash::FxHashMap;

use crate::outline::item::Vis;

/// An interned name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(pub u32);

/// The interned names of an outline. Each name text is stored once for the life of the process.
#[derive(Debug)]
pub struct Symbols {
    map: RefCell<FxHashMap<&'static str, Sym>>,
    names: RefCell<Vec<&'static str>>,
}

/// `crate`, interned first.
pub const CRATE: Sym = Sym(0);
/// `self`.
pub const SELF: Sym = Sym(1);
/// `super`.
pub const SUPER: Sym = Sym(2);
/// `_`.
pub const UNDERSCORE: Sym = Sym(3);
const FIXED: [&str; 4] = ["crate", "self", "super", "_"];

impl Default for Symbols {
    fn default() -> Symbols {
        let symbols = Symbols { map: RefCell::default(), names: RefCell::default() };
        FIXED.iter().chain(PRIMITIVES.iter().map(|(name, _)| name)).for_each(|name| {
            symbols.intern(name);
        });
        symbols
    }
}

impl Symbols {
    /// The primitive type that `sym` names.
    pub fn primitive(sym: Sym) -> Option<Primitive> {
        let index = (sym.0 as usize).checked_sub(FIXED.len())?;
        PRIMITIVES.get(index).map(|(_, primitive)| *primitive)
    }

    /// The symbol of `name`, adding it when it is new.
    pub fn intern(&self, name: &str) -> Sym {
        if let Some(found) = self.map.borrow().get(name) {
            return *found;
        }
        let text: &'static str = Box::leak(name.into());
        let sym = Sym(self.names.borrow().len() as u32);
        self.names.borrow_mut().push(text);
        self.map.borrow_mut().insert(text, sym);
        sym
    }

    /// The symbol of `name` if any scope holds it.
    pub fn get(&self, name: &str) -> Option<Sym> {
        self.map.borrow().get(name).copied()
    }

    /// The text of `sym`.
    pub fn text(&self, sym: Sym) -> &'static str {
        self.names.borrow()[sym.0 as usize]
    }
}

/// A module of one crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModRef {
    pub krate: u32,
    pub module: u32,
}

/// An item of one crate, by file and item index in the crate's outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ItemRef {
    pub krate: u32,
    pub file: u32,
    pub item: u32,
}

/// What a name refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Def {
    Module(ModRef),
    Item(ItemRef),
    Variant(ItemRef, u32),
    Primitive(Primitive),
}

/// A built-in type that paths can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primitive {
    Bool,
    Char,
    Str,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F16,
    F32,
    F64,
    F128,
}

const PRIMITIVES: [(&str, Primitive); 19] = [
    ("bool", Primitive::Bool),
    ("char", Primitive::Char),
    ("str", Primitive::Str),
    ("i8", Primitive::I8),
    ("i16", Primitive::I16),
    ("i32", Primitive::I32),
    ("i64", Primitive::I64),
    ("i128", Primitive::I128),
    ("isize", Primitive::Isize),
    ("u8", Primitive::U8),
    ("u16", Primitive::U16),
    ("u32", Primitive::U32),
    ("u64", Primitive::U64),
    ("u128", Primitive::U128),
    ("usize", Primitive::Usize),
    ("f16", Primitive::F16),
    ("f32", Primitive::F32),
    ("f64", Primitive::F64),
    ("f128", Primitive::F128),
];

impl Primitive {
    /// The primitive type spelled `name`.
    pub fn named(name: &str) -> Option<Primitive> {
        PRIMITIVES.iter().find(|(spelled, _)| *spelled == name).map(|(_, primitive)| *primitive)
    }
}

/// How a name entered a scope. Items and explicit imports shadow glob imports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Item,
    Import,
    Glob,
}

/// One name in one namespace of a scope.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub def: Def,
    pub vis: Vis,
    pub origin: Origin,
}

/// A namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ns {
    Types,
    Values,
    Macros,
}

/// The bindings of one name in the three namespaces.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerNs {
    pub types: Option<Binding>,
    pub values: Option<Binding>,
    pub macros: Option<Binding>,
}

impl PerNs {
    /// The binding in `ns`.
    pub fn get(&self, ns: Ns) -> Option<Binding> {
        match ns {
            Ns::Types => self.types,
            Ns::Values => self.values,
            Ns::Macros => self.macros,
        }
    }

    fn slot(&mut self, ns: Ns) -> &mut Option<Binding> {
        match ns {
            Ns::Types => &mut self.types,
            Ns::Values => &mut self.values,
            Ns::Macros => &mut self.macros,
        }
    }

    /// Adds `binding` in `ns` unless a binding that shadows it is there; returns whether the scope changed.
    pub fn insert(&mut self, ns: Ns, binding: Binding) -> bool {
        let slot = self.slot(ns);
        match slot {
            Some(old) if old.def == binding.def && old.origin == binding.origin => false,
            Some(old) if old.origin == Origin::Glob && binding.origin != Origin::Glob => {
                *slot = Some(binding);
                true
            }
            Some(_) => false,
            None => {
                *slot = Some(binding);
                true
            }
        }
    }
}

/// The names declared in or imported into one module.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    pub names: FxHashMap<Sym, PerNs>,
    pub log: Vec<Sym>,
}

impl Scope {
    /// Adds `binding` for `name` in `ns` and logs the name when the scope changed; returns whether it changed.
    pub fn bind(&mut self, name: Sym, ns: Ns, binding: Binding) -> bool {
        let changed = self.names.entry(name).or_default().insert(ns, binding);
        if changed {
            self.log.push(name);
        }
        changed
    }

    /// The number of changes so far, which globs compare to copy only what is new.
    pub fn version(&self) -> u32 {
        self.log.len() as u32
    }
}

/// One `use` that resolved to nothing.
#[derive(Clone, Debug)]
pub struct Unresolved {
    pub module: u32,
    pub file: u32,
    pub import: u32,
}

/// The resolved scopes of one crate.
#[derive(Clone, Debug, Default)]
pub struct Crate {
    pub scopes: Vec<Scope>,
    pub prelude: Option<ModRef>,
    pub unresolved: Vec<Unresolved>,
    pub externs: Vec<(Sym, Def)>,
    pub deps: Vec<(Sym, usize)>,
    pub edition_2015: bool,
    pub edition: String,
    pub collected: Vec<Option<usize>>,
    pub pending: Vec<collect::Pending>,
    pub eager: bool,
    wanted: FxHashMap<u32, rustc_hash::FxHashSet<Sym>>,
    named: FxHashMap<(u32, Sym), Vec<usize>>,
    globs: FxHashMap<u32, Vec<usize>>,
}

impl Crate {
    /// Whether a lookup of `name` in `module` already made its imports resolve.
    pub fn wants(&self, module: u32, name: Sym) -> bool {
        self.wanted.get(&module).is_some_and(|names| names.contains(&name))
    }
}

/// The resolved scopes of every crate in an outline.
pub struct Names {
    pub crates: Vec<Crate>,
    pub symbols: Symbols,
}

impl Names {
    /// Path resolution in crate `krate` of `outline`, over the finished scopes.
    pub fn paths<'a>(&'a self, outline: &'a crate::outline::Outline, krate: u32) -> path::Paths<'a> {
        let entry = &self.crates[krate as usize];
        path::Paths {
            krate,
            trees: &outline.crates,
            done: &self.crates,
            scopes: &entry.scopes,
            deps: &entry.deps,
            edition_2015: entry.edition_2015,
            prelude: entry.prelude,
            externs: &entry.externs,
            symbols: &self.symbols,
            demands: None,
        }
    }
}

/// A name without its `r#` prefix.
pub fn plain(name: &str) -> &str {
    name.strip_prefix("r#").unwrap_or(name)
}
