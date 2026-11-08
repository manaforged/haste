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

impl Context<'_> {
    /// The generic parameters and where clause of trait `item`.
    pub fn trait_generics(&mut self, item: ItemRef) -> Result<Rc<TraitGenerics>> {
        if let Some(found) = self.trait_defs.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("trait has no name"))?;
        let parsed = Rc::new(sig::generics(src, name + 1, entry.body.unwrap_or(entry.end))?);
        self.trait_defs.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The trait whose member `item` is.
    pub fn owner_trait(&self, item: ItemRef) -> Option<ItemRef> {
        if item.outer().is_some() {
            return None;
        }
        let items = &self.file(item).items.items;
        let parent = items[item.item as usize].parent?;
        (items[parent as usize].kind == ItemKind::Trait).then_some(ItemRef { item: parent, ..item })
    }

    /// The member of trait `of` named `name`.
    pub fn trait_member(&self, of: ItemRef, name: &str) -> Option<ItemRef> {
        if of.outer().is_some() {
            return None;
        }
        let items = &self.file(of).items.items;
        let members = (0..items.len() as u32).filter(|&at| items[at as usize].parent == Some(of.item));
        members.map(|at| ItemRef { item: at, ..of }).find(|member| self.name(*member) == name)
    }

    /// The traits in scope in module `home`.
    pub fn traits_in_scope(&self, home: ModRef) -> Vec<ItemRef> {
        self.paths(home.krate).traits_in(home)
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
        if let Some(table) = &self.langs {
            return table.get(name).copied();
        }
        let mut table = HashMap::default();
        for krate in self.primitive_holders() {
            let tree = &self.outline.crates[krate as usize];
            if tree.partial() {
                self.asks.crates.borrow_mut().push(krate);
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
        self.langs = Some(Rc::new(table));
        found
    }
}
