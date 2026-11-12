use std::sync::Arc;

use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::sig::{self, Generics};
use crate::names::{ItemRef, ModRef};
use crate::outline::item::{ItemKind, mark};

use super::{Context, read_lock, write_lock};

/// The generic parameters and where clause of a trait, in their own arena.
pub type TraitGenerics = (Body, Generics);

/// The lang items of the standard crates by name, built from the items whose attributes name `lang` or `panic_handler`.
pub struct Langs {
    /// Each lang item by its `#[lang]` value, and std's `#[panic_handler]` function by `PANIC_HANDLER`.
    table: HashMap<String, ItemRef>,
    /// The standard crates that were read only in part when the table was built, which each lookup asks to load.
    partial: Vec<u32>,
    /// The standard crates whose files the table was built from.
    pub(super) crates: Vec<u32>,
}

/// The lang-table key of the function `#[panic_handler]` marks, which no `#[lang]` value can spell.
pub(crate) const PANIC_HANDLER: &str = "#[panic_handler]";

impl<'a> Context<'a> {
    /// The generic parameters and where clause of trait `item`.
    pub fn trait_generics(&mut self, item: ItemRef) -> Result<Arc<TraitGenerics>> {
        if let Some(found) = self.store.lookups.trait_defs.get(&item) {
            return Ok(found);
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("trait has no name"))?;
        let parsed = Arc::new(sig::generics(src, name + 1, entry.body.unwrap_or(entry.end))?);
        self.store.lookups.trait_defs.insert(item, Arc::clone(&parsed));
        Ok(parsed)
    }

    /// The trait whose member `item` is.
    pub fn owner_trait(&self, item: ItemRef) -> Option<ItemRef> {
        self.parent_item(item).filter(|parent| self.item(*parent).1.kind == ItemKind::Trait)
    }

    /// The member of trait `of` named `name`.
    pub fn trait_member(&self, of: ItemRef, name: &str) -> Option<ItemRef> {
        let members = self.trait_members.get_or_insert_with(of, || {
            let mut map = HashMap::default();
            for member in self.children(of) {
                map.entry(self.plain_name(member)).or_insert(member);
            }
            Arc::new(map)
        });
        members.get(name).copied()
    }

    /// The methods of trait `of` and then of its supertraits in vtable order: each one that takes `self`, has no type parameters of its own and no `Self: Sized` bound, and none for the rest or for an unfixed generic supertrait.
    pub fn vtable_methods(&mut self, of: ItemRef) -> Result<Vec<Option<ItemRef>>> {
        let traits = super::infer::detached(self, of, |infer| infer.vtable_traits(of))?;
        let mut methods = Vec::new();
        for (owner, fixed) in traits {
            for member in self.children(owner) {
                if self.item(member).1.kind != ItemKind::Fn {
                    continue;
                }
                let sig = self.signature(member)?;
                let callable = fixed && sig.receiver.is_some() && super::infer::type_params(&sig.generics).next().is_none() && !super::infer::detached(self, member, |infer| infer.needs_sized_self(member))?;
                methods.push(callable.then_some(member));
            }
        }
        Ok(methods)
    }

    /// The supertraits of trait `of` that a trait object of it can become, in vtable order: each one it implies that has no type parameters.
    pub fn vtable_supers(&mut self, of: ItemRef) -> Result<Vec<ItemRef>> {
        let traits = super::infer::detached(self, of, |infer| infer.implied_traits(vec![of]))?;
        let mut supers = Vec::new();
        for owner in traits.into_iter().filter(|&owner| owner != of) {
            if !self.is_generic_trait(owner)? {
                supers.push(owner);
            }
        }
        Ok(supers)
    }

    fn is_generic_trait(&mut self, of: ItemRef) -> Result<bool> {
        Ok(super::infer::type_params(&self.trait_generics(of)?.1.params).next().is_some())
    }

    /// The traits in scope in module `home`.
    pub fn traits_in_scope(&self, home: ModRef) -> Vec<ItemRef> {
        if let Some(known) = self.store.lookups.scope_traits.get(&home) {
            return known.to_vec();
        }
        let before = self.quiet_mark();
        let found = self.paths(home.krate).traits_in(home);
        if self.quiet_mark() == before {
            self.store.lookups.scope_traits.insert(home, Arc::from(found.as_slice()));
        }
        found
    }

    /// Whether impl `block` marks a member `default`, so that a more specific impl may override it.
    pub fn specializable(&self, block: crate::sema::impls::ImplRef) -> bool {
        self.members(block).into_iter().any(|member| {
            let (src, entry) = self.item(member);
            entry.name.is_some_and(|name| (entry.start..name).any(|at| src.text(at) == "default"))
        })
    }

    /// Whether trait `item` is an auto trait, which every type implements unless an impl says otherwise.
    pub fn is_auto_trait(&self, item: ItemRef) -> bool {
        let (src, entry) = self.item(item);
        entry.name.is_some_and(|name| (entry.start..name).any(|at| src.text(at) == "auto"))
    }

    /// The function that a call to `item` runs: std's `#[panic_handler]` for core's `panic_impl` declaration, as the linker binds it, else `item`.
    pub fn linked(&mut self, item: ItemRef) -> ItemRef {
        if !self.foreign(item) || self.lang(item) != Some("panic_impl") {
            return item;
        }
        self.lang_item(PANIC_HANDLER).unwrap_or(item)
    }

    /// The item that `#[lang = "name"]` marks in the standard crates.
    pub fn lang_item(&mut self, name: &str) -> Option<ItemRef> {
        if let Some(langs) = read_lock(&self.store.lookups.state.langs).as_ref() {
            self.asks.crates.borrow_mut().extend(langs.partial.iter().copied());
            return langs.table.get(name).copied();
        }
        let built = self.lang_table();
        let mut guard = write_lock(&self.store.lookups.state.langs);
        let langs = guard.get_or_insert(built);
        self.asks.crates.borrow_mut().extend(langs.partial.iter().copied());
        langs.table.get(name).copied()
    }

    fn lang_table(&self) -> Langs {
        let (mut table, mut partial, crates) = (HashMap::default(), Vec::new(), self.primitive_holders());
        let wanted = mark("lang") | mark("panic_handler");
        for &krate in &crates {
            let tree = &self.outline.crates[krate as usize];
            let std = self.outline.roots[krate as usize].name == "std";
            if tree.partial() {
                partial.push(krate);
            }
            for (file, entry) in tree.files.iter().enumerate() {
                for &(at, _) in entry.items.marks.iter().filter(|(_, bits)| bits & wanted != 0) {
                    let item = ItemRef { krate, file: file as u32, item: at };
                    if let Some(lang) = self.lang(item) {
                        table.insert(lang.to_owned(), item);
                    } else if std && entry.items.items[at as usize].kind == ItemKind::Fn && self.attr_at(item, "panic_handler").is_some() {
                        table.insert(PANIC_HANDLER.to_owned(), item);
                    }
                }
            }
        }
        Langs { table, partial, crates }
    }
}
