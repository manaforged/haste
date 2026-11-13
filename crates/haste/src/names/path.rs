use std::cell::RefCell;

use rustc_lexer::TokenKind;

use crate::outline::attr::split_commas;
use crate::outline::item::{ItemKind, Vis};
use crate::outline::tree::Crate as Tree;

use super::{Binding, CRATE, Crate, Def, ItemRef, ModRef, Origin, PerNs, SELF, SUPER, Scope, Sym, Symbols, UNDERSCORE, plain};

/// What lookups asked for that is not there yet: modules to read, names whose imports must resolve, crates to read whole, and modules whose every import must resolve.
#[derive(Debug, Default)]
pub struct Requests {
    pub modules: RefCell<Vec<ModRef>>,
    pub names: RefCell<Vec<(ModRef, Sym)>>,
    pub crates: RefCell<Vec<u32>>,
    pub scopes: RefCell<Vec<ModRef>>,
    pub procs: RefCell<Vec<(ItemRef, String)>>,
}

impl Requests {
    /// How many lookups have asked for something so far.
    pub fn count(&self) -> usize {
        self.modules.borrow().len() + self.names.borrow().len() + self.crates.borrow().len() + self.scopes.borrow().len() + self.procs.borrow().len()
    }

    /// Where each list of asks ends now, to take what later lookups add with `since`.
    pub fn mark(&self) -> [usize; 5] {
        [self.modules.borrow().len(), self.names.borrow().len(), self.crates.borrow().len(), self.scopes.borrow().len(), self.procs.borrow().len()]
    }

    /// The asks added after `mark`.
    pub fn since(&self, mark: [usize; 5]) -> Requests {
        Requests {
            modules: RefCell::new(self.modules.borrow()[mark[0]..].to_vec()),
            names: RefCell::new(self.names.borrow()[mark[1]..].to_vec()),
            crates: RefCell::new(self.crates.borrow()[mark[2]..].to_vec()),
            scopes: RefCell::new(self.scopes.borrow()[mark[3]..].to_vec()),
            procs: RefCell::new(self.procs.borrow()[mark[4]..].to_vec()),
        }
    }

    /// Requests again for everything that `other` asked for, so that a lookup reading a result built from those asks counts as asking too.
    pub fn repeat(&self, other: &Requests) {
        self.modules.borrow_mut().extend(other.modules.borrow().iter().copied());
        self.names.borrow_mut().extend(other.names.borrow().iter().copied());
        self.crates.borrow_mut().extend(other.crates.borrow().iter().copied());
        self.scopes.borrow_mut().extend(other.scopes.borrow().iter().copied());
        self.procs.borrow_mut().extend(other.procs.borrow().iter().cloned());
    }
}

/// What path resolution in one crate can see.
pub struct Paths<'a> {
    pub krate: u32,
    pub trees: &'a [Tree],
    pub done: &'a [Crate],
    pub scopes: &'a [Scope],
    pub deps: &'a [(Sym, usize)],
    pub edition_2015: bool,
    pub prelude: Option<ModRef>,
    pub externs: &'a [(Sym, Def)],
    pub symbols: &'a Symbols,
    pub demands: Option<&'a Requests>,
}

/// A path to resolve: its interned segments, whether it starts with `::`, and the crate `$crate` names.
#[derive(Clone, Copy, Debug)]
pub struct Path<'a> {
    pub segments: &'a [Sym],
    pub absolute: bool,
    pub dollar: Option<u32>,
}

fn module(krate: u32, module: u32) -> Def {
    Def::Module(ModRef { krate, module })
}

impl<'a> Paths<'a> {
    /// The scope of `module`, in this crate or a finished one.
    pub fn scope(&self, module: ModRef) -> Option<&Scope> {
        let entry = self.trees.get(module.krate as usize)?.modules.get(module.module as usize)?;
        if !entry.loaded() {
            if let (Some(demands), crate::outline::tree::State::Declared(_)) = (self.demands, &entry.state) {
                demands.modules.borrow_mut().push(module);
            }
            return None;
        }
        if module.krate == self.krate {
            return self.scopes.get(module.module as usize);
        }
        self.done.get(module.krate as usize)?.scopes.get(module.module as usize)
    }

    /// The parent of `module`.
    pub fn parent(&self, module: ModRef) -> Option<ModRef> {
        let parent = self.trees.get(module.krate as usize)?.modules.get(module.module as usize)?.parent?;
        Some(ModRef { krate: module.krate, module: parent as u32 })
    }

    /// Whether `module` is `ancestor` or inside it.
    pub fn within(&self, module: ModRef, ancestor: ModRef) -> bool {
        let mut at = Some(module);
        while let Some(current) = at {
            if current == ancestor {
                return true;
            }
            at = self.parent(current);
        }
        false
    }

    fn extern_crate(&self, name: Sym) -> Option<Def> {
        if let Some((_, def)) = self.externs.iter().find(|(alias, _)| *alias == name) {
            return Some(*def);
        }
        let (_, root) = self.deps.iter().find(|(dep, _)| *dep == name)?;
        Some(module(*root as u32, 0))
    }

    fn variants(&self, item: ItemRef) -> Vec<(&'a str, u32)> {
        let Some(file) = self.trees.get(item.krate as usize).and_then(|tree| tree.files.get(item.file as usize)) else {
            return Vec::new();
        };
        let Some(open) = file.items.items[item.item as usize].body else {
            return Vec::new();
        };
        variant_names(&file.source, open).into_iter().enumerate().map(|(index, name)| (name, index as u32)).collect()
    }

    fn item_kind(&self, item: ItemRef) -> Option<ItemKind> {
        Some(self.trees.get(item.krate as usize)?.files.get(item.file as usize)?.items.items.get(item.item as usize)?.kind)
    }

    /// The bindings of `name` inside `def`: a module's scope or an enum's variants.
    pub fn member(&self, def: Def, name: Sym) -> PerNs {
        match def {
            Def::Module(at) => {
                let found = self.scope(at).and_then(|scope| scope.names.get(&name).copied());
                if let Some(asks) = self.demands {
                    let entry = self.done.get(at.krate as usize);
                    if !entry.is_some_and(|krate| krate.wants(at.module, name)) {
                        asks.names.borrow_mut().push((at, name));
                    } else if found.is_none()
                        && let Some(krate) = entry
                        && self.demand_pending_import(krate, at, name)
                    {
                        asks.names.borrow_mut().push((at, name));
                    }
                }
                if found.is_none() {
                    self.ask_crate(at);
                }
                found.unwrap_or_default()
            }
            Def::Item(item) if self.item_kind(item) == Some(ItemKind::Enum) => {
                let text = self.symbols.text(name);
                let found = self.variants(item).into_iter().find(|(variant, _)| *variant == text);
                let binding = found.map(|(_, index)| Binding { def: Def::Variant(item, index), vis: Vis::Pub, origin: Origin::Item });
                PerNs { types: binding, values: binding, macros: None }
            }
            _ => PerNs::default(),
        }
    }

    /// Asks to load every module of the partial crate whose root is `root`, when this lookup records asks.
    pub(super) fn ask_crate(&self, root: ModRef) {
        if let Some(demands) = self.demands
            && root.module == 0
            && root.krate != self.krate
            && self.trees.get(root.krate as usize).is_some_and(|tree| tree.partial())
        {
            demands.crates.borrow_mut().push(root.krate);
        }
    }

    fn demand_pending_import(&self, krate: &super::Crate, at: ModRef, name: Sym) -> bool {
        let named = krate.named.get(&(at.module, name)).into_iter().flatten();
        let globs = krate.globs.get(&at.module).into_iter().flatten();
        let mut any = false;
        for &index in named.chain(globs) {
            let pending = &krate.pending[index];
            if pending.stamp.is_some() {
                continue;
            }
            any = true;
            if at.krate == self.krate
                && let Some(&first) = pending.segments.first()
                && let Some(Def::Module(ext)) = self.extern_crate(first)
            {
                self.ask_crate(ext);
            }
        }
        any
    }

    /// Every variant of the enum `item`, as bindings.
    pub fn all_variants(&self, item: ItemRef) -> Vec<(Sym, Binding)> {
        let variants = self.variants(item).into_iter();
        variants.map(|(name, index)| (self.symbols.intern(name), Binding { def: Def::Variant(item, index), vis: Vis::Pub, origin: Origin::Glob })).collect()
    }

    fn keyword(&self, word: Sym, at: ModRef, dollar: Option<u32>) -> Option<Option<Def>> {
        match word {
            CRATE => Some(Some(module(dollar.unwrap_or(self.krate), 0))),
            SELF => Some(Some(Def::Module(at))),
            SUPER => Some(self.parent(at).map(Def::Module)),
            _ => None,
        }
    }

    fn textual(&self, sym: Sym, at: ModRef) -> Option<Binding> {
        let mut current = self.parent(at);
        while let Some(module) = current {
            let found = self.scope(module).and_then(|scope| scope.names.get(&sym)).and_then(|entry| entry.macros);
            if let Some(binding) = found.filter(|binding| binding.origin == Origin::Item) {
                return Some(binding);
            }
            current = self.parent(module);
        }
        None
    }

    /// The traits in scope in module `at`: declared or imported there, imported as `_`, or in the prelude; asks for every import to resolve first.
    pub fn traits_in(&self, at: ModRef) -> Vec<ItemRef> {
        let mut found = Vec::new();
        for module in std::iter::once(at).chain(self.prelude) {
            self.traits_of(module, &mut found);
        }
        found
    }

    /// Adds to `found` the traits that module `module` declares or imports.
    pub fn traits_of(&self, module: ModRef, found: &mut Vec<ItemRef>) {
        let Some(entry) = self.done.get(module.krate as usize) else {
            return;
        };
        if !entry.resolves_all(module.module)
            && let Some(demands) = self.demands
        {
            demands.scopes.borrow_mut().push(module);
        }
        let named = self.scope(module).into_iter().flat_map(|scope| scope.names.values()).filter_map(|names| names.types.map(|binding| binding.def));
        let hidden = entry.pending.iter().filter(|pending| pending.module == module.module && pending.name == Some(UNDERSCORE)).filter_map(|pending| pending.stamp.map(|(def, _)| def));
        let mut defs: Vec<Def> = named.chain(hidden).collect();
        self.glob_hidden(module, (module, &mut vec![module]), &mut defs);
        for def in defs {
            if let Def::Item(item) = def
                && self.trees[item.krate as usize].files[item.file as usize].items.items[item.item as usize].kind == ItemKind::Trait
                && !found.contains(&item)
            {
                found.push(item);
            }
        }
    }

    /// Adds to `found` what the `use … as _` imports of the modules that `module` glob-imports bind, through further globs, when visible to `importer`.
    fn glob_hidden(&self, module: ModRef, (importer, seen): (ModRef, &mut Vec<ModRef>), found: &mut Vec<Def>) {
        let Some(entry) = self.done.get(module.krate as usize) else {
            return;
        };
        let globs = entry.globs.get(&module.module).into_iter().flatten();
        let targets: Vec<ModRef> = globs
            .filter_map(|&index| match entry.pending[index].stamp {
                Some((Def::Module(target), _)) => Some(target),
                _ => None,
            })
            .collect();
        for target in targets {
            if seen.contains(&target) {
                continue;
            }
            seen.push(target);
            let Some(other) = self.done.get(target.krate as usize) else {
                continue;
            };
            if !other.resolves_all(target.module)
                && let Some(demands) = self.demands
            {
                demands.scopes.borrow_mut().push(target);
            }
            let hidden = other.pending.iter().filter(|pending| pending.module == target.module && pending.name == Some(UNDERSCORE) && super::resolve::visible(self, pending.vis, target, importer));
            found.extend(hidden.filter_map(|pending| pending.stamp.map(|(def, _)| def)));
            self.glob_hidden(target, (importer, seen), found);
        }
    }

    /// The bindings of the first segment `word` of a path used in module `at`, merged across namespaces from the local scope, the extern prelude and the std prelude.
    pub fn first(&self, word: Sym, at: ModRef, path: &Path<'_>) -> PerNs {
        let item = |def: Def| Binding { def, vis: Vis::Pub, origin: Origin::Item };
        if let Some(def) = self.keyword(word, at, path.dollar) {
            return PerNs { types: def.map(item), values: None, macros: None };
        }
        let external = self.extern_crate(word).map(item);
        if path.absolute && self.edition_2015 {
            let root = self.member(module(self.krate, 0), word);
            return PerNs { types: root.types.or(external), ..root };
        }
        if path.absolute {
            return PerNs { types: external, values: None, macros: None };
        }
        let local = self.member(Def::Module(at), word);
        let prelude = self.prelude.map(|prelude| self.member(Def::Module(prelude), word)).unwrap_or_default();
        PerNs {
            types: local.types.or(external).or(prelude.types).or_else(|| Symbols::primitive(word).map(|found| item(Def::Primitive(found)))),
            values: local.values.or(prelude.values),
            macros: local.macros.or_else(|| self.textual(word, at)).or(prelude.macros),
        }
    }

    pub(super) fn demand_macro_module(&self, submod: ModRef) {
        if self.scope(submod).is_some() {
            self.ask_crate(ModRef { krate: submod.krate, module: 0 });
        }
    }

    pub(super) fn demand_macro_in(&self, root: ModRef, found: PerNs) {
        match found.types {
            Some(Binding { def: Def::Module(submod), .. }) if submod.krate == root.krate => self.demand_macro_module(submod),
            _ => self.ask_crate(root),
        }
    }

    /// The macro `word` that a crate this crate names in `#[macro_use] extern crate` exports at its root.
    pub fn macro_use(&self, word: Sym) -> Option<Binding> {
        self.trees[self.krate as usize].macro_use.iter().find_map(|&krate| {
            let root = ModRef { krate: krate as u32, module: 0 };
            let found = self.member(Def::Module(root), word);
            if found.macros.is_none() {
                self.demand_macro_in(root, found);
            }
            found.macros
        })
    }

    /// The path that a `use` declaration of `path` names: in edition 2015 it starts at the crate root unless it starts with a keyword.
    pub fn use_path<'p>(&self, path: Path<'p>) -> Path<'p> {
        let keyword = path.segments.first().is_some_and(|word| [CRATE, SELF, SUPER].contains(word));
        Path { absolute: path.absolute || (self.edition_2015 && !keyword), ..path }
    }

    /// Resolves every segment of `path` in the type namespace, from module `at`.
    pub fn resolve(&self, path: &Path<'_>, at: ModRef) -> Option<Def> {
        let (first, rest) = path.segments.split_first()?;
        let mut def = self.first(*first, at, path).types?.def;
        for segment in rest {
            def = match (*segment, def) {
                (SUPER, Def::Module(current)) => Def::Module(self.parent(current)?),
                _ => self.member(def, *segment).types?.def,
            };
        }
        Some(def)
    }
}

/// The names of the variants of the enum whose body opens at token `open` of `src`, in order.
pub fn variant_names(src: &crate::outline::source::Source, open: u32) -> Vec<&str> {
    let parts = split_commas(src, open + 1, src.pair(open));
    let names = parts.into_iter().filter_map(|(mut at, to)| {
        while at < to && src.kind(at) == TokenKind::Pound {
            at = src.pair(at + 1) + 1;
        }
        (at < to && matches!(src.kind(at), TokenKind::Ident | TokenKind::RawIdent)).then(|| plain(src.text(at)))
    });
    names.collect()
}
