use std::cell::RefCell;

use rustc_lexer::TokenKind;

use crate::outline::attr::split_commas;
use crate::outline::item::{ItemKind, Vis};
use crate::outline::tree::Crate as Tree;

use super::{Binding, CRATE, Crate, Def, ItemRef, ModRef, Origin, PerNs, SELF, SUPER, Scope, Sym, Symbols, UNDERSCORE, plain};

/// What lookups asked for that is not there yet: modules to read, names whose imports must resolve, crates to read whole, and modules whose every import must resolve.
#[derive(Debug, Default)]
pub struct Asks {
    pub modules: RefCell<Vec<ModRef>>,
    pub names: RefCell<Vec<(ModRef, Sym)>>,
    pub crates: RefCell<Vec<u32>>,
    pub scopes: RefCell<Vec<ModRef>>,
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
    pub demands: Option<&'a Asks>,
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
        let Some(file) = self.trees.get(item.krate as usize).and_then(|tree| tree.files.get(item.file as usize)) else { return Vec::new() };
        let Some(open) = file.items.items[item.item as usize].body else { return Vec::new() };
        let src = &file.source;
        let parts = split_commas(src, open + 1, src.pair(open));
        let names = parts.into_iter().filter_map(|(mut at, to)| {
            while at < to && src.kind(at) == TokenKind::Pound {
                at = src.pair(at + 1) + 1;
            }
            (at < to && matches!(src.kind(at), TokenKind::Ident | TokenKind::RawIdent)).then(|| plain(src.text(at)))
        });
        names.enumerate().map(|(index, name)| (name, index as u32)).collect()
    }

    fn item_kind(&self, item: ItemRef) -> Option<ItemKind> {
        Some(self.trees.get(item.krate as usize)?.files.get(item.file as usize)?.items.items.get(item.item as usize)?.kind)
    }

    /// The bindings of `name` inside `def`: a module's scope or an enum's variants.
    pub fn member(&self, def: Def, name: Sym) -> PerNs {
        match def {
            Def::Module(at) => {
                if let Some(asks) = self.demands && !self.done.get(at.krate as usize).is_some_and(|krate| krate.wants(at.module, name)) {
                    asks.names.borrow_mut().push((at, name));
                }
                self.scope(at).and_then(|scope| scope.names.get(&name).copied()).unwrap_or_default()
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
            let Some(entry) = self.done.get(module.krate as usize) else { continue };
            if !entry.whole.contains(&module.module) && let Some(demands) = self.demands {
                demands.scopes.borrow_mut().push(module);
            }
            let named = self.scope(module).into_iter().flat_map(|scope| scope.names.values()).filter_map(|names| names.types.map(|binding| binding.def));
            let hidden = entry.pending.iter().filter(|pending| pending.module == module.module && pending.name == Some(UNDERSCORE)).filter_map(|pending| pending.stamp.map(|(def, _)| def));
            for def in named.chain(hidden) {
                if let Def::Item(item) = def && self.trees[item.krate as usize].files[item.file as usize].items.items[item.item as usize].kind == ItemKind::Trait && !found.contains(&item) {
                    found.push(item);
                }
            }
        }
        found
    }

    /// The bindings of the first segment `word` of a path used in module `at`, merged across namespaces from the local scope, the extern prelude and the std prelude.
    pub fn first(&self, word: Sym, at: ModRef, path: &Path<'_>) -> PerNs {
        let item = |def: Def| Binding { def, vis: Vis::Pub, origin: Origin::Item };
        if let Some(def) = self.keyword(word, at, path.dollar) {
            return PerNs { types: def.map(item), values: None, macros: None };
        }
        let external = self.extern_crate(word).map(item);
        if path.absolute {
            return if self.edition_2015 { self.member(module(self.krate, 0), word) } else { PerNs { types: external, values: None, macros: None } };
        }
        let home = if self.edition_2015 { ModRef { krate: self.krate, module: 0 } } else { at };
        let local = self.member(Def::Module(home), word);
        let prelude = self.prelude.map(|prelude| self.member(Def::Module(prelude), word)).unwrap_or_default();
        PerNs {
            types: local.types.or(external).or(prelude.types).or_else(|| Symbols::primitive(word).map(|found| item(Def::Primitive(found)))),
            values: local.values.or(prelude.values),
            macros: local.macros.or_else(|| self.textual(word, at)).or(prelude.macros),
        }
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
