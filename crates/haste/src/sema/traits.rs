use std::rc::Rc;

use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::sig::{self, Generics};
use crate::names::{ItemRef, ModRef};
use crate::outline::item::ItemKind;

use super::Context;

/// The generic parameters and where clause of a trait, in their own arena.
pub type TraitGenerics = (Body, Generics);

impl<'a> Context<'a> {
    /// The generic parameters and where clause of trait `item`.
    pub fn trait_generics(&mut self, item: ItemRef) -> Result<Rc<TraitGenerics>> {
        if let Some(found) = self.store.lookups.trait_defs.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("trait has no name"))?;
        let parsed = Rc::new(sig::generics(src, name + 1, entry.body.unwrap_or(entry.end))?);
        self.store.lookups.trait_defs.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The trait whose member `item` is.
    pub fn owner_trait(&self, item: ItemRef) -> Option<ItemRef> {
        self.parent_item(item).filter(|parent| self.item(*parent).1.kind == ItemKind::Trait)
    }

    /// The member of trait `of` named `name`.
    pub fn trait_member(&self, of: ItemRef, name: &str) -> Option<ItemRef> {
        let known = self.trait_members.borrow().get(&of).cloned();
        let members = known.unwrap_or_else(|| {
            let mut members = HashMap::default();
            for member in self.children(of) {
                members.entry(self.name(member)).or_insert(member);
            }
            let members = Rc::new(members);
            self.trait_members.borrow_mut().insert(of, members.clone());
            members
        });
        members.get(name).copied()
    }

    /// The methods of trait `of` and then of its supertraits in vtable order: each one that takes `self`, has no type parameters of its own and no `Self: Sized` bound, and none for the rest or for a generic supertrait.
    pub fn vtable_methods(&mut self, of: ItemRef) -> Result<Vec<Option<ItemRef>>> {
        let traits = super::infer::detached(self, of, |infer| infer.implied_traits(vec![of]))?;
        let mut methods = Vec::new();
        for owner in traits {
            let open = owner != of && self.is_generic_trait(owner)?;
            for member in self.children(owner).filter(|member| self.item(*member).1.kind == ItemKind::Fn).collect::<Vec<_>>() {
                let sig = self.signature(member)?;
                let callable = !open && sig.receiver.is_some() && super::infer::type_params(&sig.generics).next().is_none() && !super::infer::detached(self, member, |infer| infer.needs_sized_self(member))?;
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
        if let Some(known) = self.store.lookups.scope_traits.borrow().get(&home) {
            return known.to_vec();
        }
        let found = self.paths(home.krate).traits_in(home);
        self.store.lookups.scope_traits.borrow_mut().insert(home, found.as_slice().into());
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

    /// The item that `#[lang = "name"]` marks in the standard crates.
    pub fn lang_item(&mut self, name: &str) -> Option<ItemRef> {
        if let Some((table, partial)) = &self.store.lookups.langs {
            self.asks.crates.borrow_mut().extend(partial.iter().copied());
            return table.get(name).copied();
        }
        let (mut table, mut partial) = (HashMap::default(), Vec::new());
        for krate in self.primitive_holders() {
            let tree = &self.outline.crates[krate as usize];
            if tree.partial() {
                partial.push(krate);
            }
            for (file, items) in tree.files.iter().map(|file| &file.items.items).enumerate() {
                for at in 0..items.len() {
                    let item = ItemRef { krate, file: file as u32, item: at as u32 };
                    if let Some(lang) = self.lang(item) {
                        table.insert(lang.to_owned(), item);
                    }
                }
            }
        }
        let found = table.get(name).copied();
        self.asks.crates.borrow_mut().extend(partial.iter().copied());
        self.store.lookups.langs = Some((Rc::new(table), partial.into()));
        found
    }
}
