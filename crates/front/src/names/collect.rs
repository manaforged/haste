use crate::outline::item::{Attr, Item, ItemKind, Leaf, PROC_MACROS, ProcKind, Vis};
use crate::outline::tree::{Crate as Tree, Env};

use super::{Binding, Def, ItemRef, ModRef, Ns, Origin, SELF, Scope, Sym, Symbols, UNDERSCORE, plain};

/// One `use` waiting to resolve: where it is, its interned path, the name it binds (none for a glob) and the source it last read.
#[derive(Clone, Debug)]
pub struct Pending {
    pub module: u32,
    pub file: u32,
    pub import: u32,
    pub segments: Box<[Sym]>,
    pub absolute: bool,
    pub dollar: Option<u32>,
    pub name: Option<Sym>,
    pub vis: Vis,
    pub stamp: Option<(Def, u32)>,
    pub active: bool,
}

/// The items of `module`: (file, item) pairs from its own file and from the macro expansions spliced into it.
pub fn items_of(tree: &Tree, module: usize) -> Vec<(u32, u32)> {
    let entry = &tree.modules[module];
    if !entry.loaded() {
        return Vec::new();
    }
    let files: Vec<(u32, Option<u32>)> = std::iter::once((entry.file as u32, entry.item)).chain(entry.parts.iter().map(|&part| (part as u32, None))).collect();
    items_in(tree, &files)
}

fn namespaces(item: &Item) -> &'static [Ns] {
    match item.kind {
        ItemKind::Struct if item.body.is_none() => &[Ns::Types, Ns::Values],
        ItemKind::Struct | ItemKind::Enum | ItemKind::Union | ItemKind::Trait | ItemKind::TraitAlias | ItemKind::Type => &[Ns::Types],
        ItemKind::Fn | ItemKind::Const | ItemKind::Static => &[Ns::Values],
        ItemKind::MacroRules | ItemKind::Macro => &[Ns::Macros],
        _ => &[],
    }
}

fn extern_crate(tree: &Tree, (file, at): (u32, u32), krate: u32, deps: &[(String, usize)]) -> Option<Def> {
    let source = &tree.files[file as usize].source;
    let item = &tree.files[file as usize].items.items[at as usize];
    let named = plain(source.text(item.start + 2));
    if named == "self" {
        return Some(Def::Module(ModRef { krate, module: 0 }));
    }
    let (_, dep) = deps.iter().find(|(dep, _)| dep == named)?;
    Some(Def::Module(ModRef { krate: *dep as u32, module: 0 }))
}

/// What a crate's collection needs besides its tree: its index, its cfg values and its dependencies.
pub struct Context<'a> {
    pub krate: u32,
    pub env: &'a Env,
    pub deps: &'a [(String, usize)],
    pub symbols: &'a Symbols,
    pub children: &'a [Vec<usize>],
}

/// Collects the items of module `index` from the files `files` (its own file and expansion parts), binds its child modules, and lists its imports.
pub fn collect_module(tree: &Tree, index: usize, files: &[(u32, Option<u32>)], context: &Context<'_>, scopes: &mut [Scope]) -> Vec<Pending> {
    let krate = context.krate;
    let mut pending = Vec::new();
    for &child in &context.children[index] {
        let module = &tree.modules[child];
        let binding = Binding { def: Def::Module(ModRef { krate, module: child as u32 }), vis: module.vis, origin: Origin::Item };
        scopes[index].bind(context.symbols.intern(plain(&module.name)), Ns::Types, binding);
    }
    for (file, at) in items_in(tree, files) {
        let source = &tree.files[file as usize];
        let item = &source.items.items[at as usize];
        if !context.env.enabled(&item.attrs) || tree.consumed.contains(&(file, at)) {
            continue;
        }
        if item.kind == ItemKind::Use {
            let imports = source.items.imports.iter().enumerate().filter(|(_, import)| import.item == at);
            pending.extend(imports.map(|(import, _)| pending_import(source, (file, import as u32), index, context.symbols)));
            continue;
        }
        bind_item(tree, (file, at), index, context, scopes);
    }
    pending
}

/// An import's path, whether it starts at the crate root, the `$crate` it starts from, and the name it binds, or none for a glob.
pub type ImportParts = (Box<[Sym]>, bool, Option<u32>, Option<Sym>);

/// The path and bound name of import `import` of `file`.
pub fn import_parts(file: &crate::outline::tree::File, import: u32, symbols: &Symbols) -> ImportParts {
    let entry = &file.items.imports[import as usize];
    let src = &file.source;
    let segments: Box<[Sym]> = entry.path.iter().map(|at| symbols.intern(plain(src.text(*at)))).collect();
    let dollar = entry.path.first().and_then(|first| first.checked_sub(1)).and_then(|at| src.dollar(at));
    let last = segments.last().copied();
    let named = if last == Some(SELF) { segments.iter().rev().nth(1).copied().or(last) } else { last };
    let name = match entry.leaf {
        Leaf::Glob => None,
        Leaf::Underscore => Some(UNDERSCORE),
        Leaf::Name(alias) => alias.map(|token| symbols.intern(plain(src.text(token)))).or(named),
    };
    (segments, entry.absolute, dollar, name)
}

fn pending_import(file: &crate::outline::tree::File, (index, import): (u32, u32), module: usize, symbols: &Symbols) -> Pending {
    let (segments, absolute, dollar, name) = import_parts(file, import, symbols);
    let vis = file.items.items[file.items.imports[import as usize].item as usize].vis;
    Pending { module: module as u32, file: index, import, segments, absolute, dollar, name, vis, stamp: None, active: false }
}

fn items_in(tree: &Tree, files: &[(u32, Option<u32>)]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for &(file, owner) in files {
        let items = &tree.files[file as usize].items.items;
        for (at, item) in items.iter().enumerate() {
            let in_block = item.parent.is_some_and(|parent| items[parent as usize].kind == ItemKind::ExternBlock && items[parent as usize].parent == owner);
            if item.parent == owner || in_block {
                out.push((file, at as u32));
            }
        }
    }
    out
}

fn bind_item(tree: &Tree, (file, at): (u32, u32), index: usize, context: &Context<'_>, scopes: &mut [Scope]) {
    let source = &tree.files[file as usize];
    let item = &source.items.items[at as usize];
    let Some(name) = item.name else { return };
    let text = plain(source.source.text(name));
    if text == "_" {
        return;
    }
    let name = context.symbols.intern(text);
    let def = Def::Item(ItemRef { krate: context.krate, file, item: at });
    let vis = if item.kind == ItemKind::MacroRules { Vis::Crate } else { item.vis };
    if item.kind == ItemKind::ExternCrate {
        if let Some(root) = extern_crate(tree, (file, at), context.krate, context.deps) {
            scopes[index].bind(name, Ns::Types, Binding { def: root, vis, origin: Origin::Item });
        }
        return;
    }
    let public = Binding { def, vis: Vis::Pub, origin: Origin::Item };
    if item.kind == ItemKind::Fn && let Some(exported) = proc_macro_name(&source.source, item, text) {
        scopes[0].bind(context.symbols.intern(exported), Ns::Macros, public);
    }
    if item.attrs.iter().any(|attr| *attr == Attr::MacroExport) {
        scopes[0].bind(name, Ns::Macros, public);
    }
    for ns in namespaces(item) {
        scopes[index].bind(name, *ns, Binding { def, vis, origin: Origin::Item });
    }
}

/// Copies the macros of `#[macro_use]` modules up into their parents, children first.
pub fn lift_macro_use(tree: &Tree, scopes: &mut [Scope]) {
    for (index, module) in tree.modules.iter().enumerate().rev() {
        let Some(parent) = module.parent.filter(|_| module.macro_use && module.loaded()) else { continue };
        let lifted: Vec<(Sym, Binding)> = scopes[index].names.iter().filter_map(|(name, entry)| Some((*name, entry.macros.filter(|binding| binding.origin == Origin::Item)?))).collect();
        for (name, binding) in lifted {
            scopes[parent].bind(name, Ns::Macros, binding);
        }
    }
}

fn proc_macro_name<'a>(source: &'a crate::outline::source::Source, item: &Item, name: &'a str) -> Option<&'a str> {
    let (at, kind) = proc_kind(source, item)?;
    Some(if kind == ProcKind::Derive { plain(source.text(at + 2)) } else { name })
}

/// The procedural macro that the function `item` of `source` defines, with the token of its attribute.
pub fn proc_kind(source: &crate::outline::source::Source, item: &Item) -> Option<(u32, ProcKind)> {
    item.attrs.iter().find_map(|attr| match attr {
        Attr::Other(at) => PROC_MACROS.iter().find(|(name, _)| *name == source.text(*at)).map(|&(_, kind)| (*at, kind)),
        _ => None,
    })
}

/// The names that root-level `extern crate` items add to the extern prelude: root-scope items that name a crate root.
pub fn externs(root: &Scope) -> Vec<(Sym, Def)> {
    let crate_root = |binding: &Binding| binding.origin == Origin::Item && matches!(binding.def, Def::Module(found) if found.module == 0);
    root.names.iter().filter_map(|(name, entry)| Some((*name, entry.types.filter(crate_root)?.def))).collect()
}
