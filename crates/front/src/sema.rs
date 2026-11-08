//! Semantic analysis of reached function bodies at concrete types, lowered to `haste-codegen` SSA.

pub mod infer;
pub mod layout;
pub mod reach;
pub mod ty;

use rustc_hash::FxHashMap as HashMap;
use std::rc::Rc;

use anyhow::{Context as _, Result, anyhow};

use crate::body::sig::{self, Signature};
use crate::body::{self as syntax, Body};
use crate::names::collect::items_of;
use crate::names::path::{Asks, Paths};
use crate::names::{ItemRef, ModRef, Names};
use crate::outline::Outline;
use crate::outline::item::{Item, ItemKind};
use crate::outline::source::Source;

use ty::Types;

/// What Sema keeps between passes over a growing outline: interned types and parsed headers and bodies.
#[derive(Default)]
pub struct Store {
    types: Types,
    signatures: HashMap<ItemRef, Rc<Signature>>,
    bodies: HashMap<ItemRef, Rc<Body>>,
    generics: HashMap<ItemRef, Rc<(Body, sig::Generics)>>,
    layouts: HashMap<ty::Ty, Rc<layout::Layout>>,
}

/// Everything Sema reads in one pass, with its caches and the lookups that found nothing loaded yet.
pub struct Context<'a> {
    pub outline: &'a Outline,
    pub names: &'a Names,
    pub types: Types,
    pub asks: Asks,
    modules: HashMap<u32, Rc<HashMap<(u32, u32), u32>>>,
    signatures: HashMap<ItemRef, Rc<Signature>>,
    bodies: HashMap<ItemRef, Rc<Body>>,
    generics: HashMap<ItemRef, Rc<(Body, sig::Generics)>>,
    layouts: HashMap<ty::Ty, Rc<layout::Layout>>,
}

impl<'a> Context<'a> {
    /// A pass over an outline and its resolved names that reuses `store`.
    pub fn new(outline: &'a Outline, names: &'a Names, store: Store) -> Context<'a> {
        let Store { types, signatures, bodies, generics, layouts } = store;
        Context { outline, names, types, asks: Asks::default(), modules: HashMap::default(), signatures, bodies, generics, layouts }
    }

    /// Ends the pass: the store for the next one, and what lookups asked the front end to load or resolve.
    pub fn release(self) -> (Store, Asks) {
        (Store { types: self.types, signatures: self.signatures, bodies: self.bodies, generics: self.generics, layouts: self.layouts }, self.asks)
    }

    /// The source file and outline entry of `item`.
    pub fn item(&self, item: ItemRef) -> (&'a Source, &'a Item) {
        let file = &self.outline.crates[item.krate as usize].files[item.file as usize];
        (&file.source, &file.items.items[item.item as usize])
    }

    /// The outline file that holds `item`.
    pub fn file(&self, item: ItemRef) -> &'a crate::outline::tree::File {
        &self.outline.crates[item.krate as usize].files[item.file as usize]
    }

    /// Path resolution inside crate `krate`, recording what it finds not loaded yet.
    pub fn paths(&self, krate: u32) -> Paths<'_> {
        Paths { demands: Some(&self.asks), ..self.names.paths(self.outline, krate) }
    }

    /// The module that holds `item`, through any impl, trait or extern block around it.
    pub fn module_of(&mut self, item: ItemRef) -> Result<ModRef> {
        let map = self.modules_of(item.krate);
        let (_, mut entry) = self.item(item);
        let mut top = item.item;
        let file = &self.outline.crates[item.krate as usize].files[item.file as usize];
        while let Some(parent) = entry.parent.filter(|parent| file.items.items[*parent as usize].kind != ItemKind::Mod) {
            top = parent;
            entry = &file.items.items[parent as usize];
        }
        let module = map.get(&(item.file, top)).copied().with_context(|| format!("no module holds item {item:?}"))?;
        Ok(ModRef { krate: item.krate, module })
    }

    fn modules_of(&mut self, krate: u32) -> Rc<HashMap<(u32, u32), u32>> {
        let tree = &self.outline.crates[krate as usize];
        self.modules
            .entry(krate)
            .or_insert_with(|| {
                let pairs = (0..tree.modules.len()).flat_map(|module| items_of(tree, module).into_iter().map(move |pair| (pair, module as u32)));
                Rc::new(pairs.collect())
            })
            .clone()
    }

    /// The parsed header of the function `item`.
    pub fn signature(&mut self, item: ItemRef) -> Result<Rc<Signature>> {
        if let Some(found) = self.signatures.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let parsed = Rc::new(sig::signature(src, entry.start, entry.body.unwrap_or(entry.end))?);
        self.signatures.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The parsed body of the function `item`.
    pub fn body(&mut self, item: ItemRef) -> Result<Rc<Body>> {
        if let Some(found) = self.bodies.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let open = entry.body.ok_or_else(|| anyhow!("function {} has no body", src.text(entry.name.unwrap_or(entry.start))))?;
        let parsed = Rc::new(syntax::parse(src, open)?);
        self.bodies.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The parsed generics and where clause of the struct, enum or union `item`.
    pub fn adt_generics(&mut self, item: ItemRef) -> Result<Rc<(Body, sig::Generics)>> {
        if let Some(found) = self.generics.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("type item has no name"))?;
        let parsed = Rc::new(sig::generics(src, name + 1, entry.body.unwrap_or(entry.end))?);
        self.generics.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The name of `item`, for messages.
    pub fn name(&self, item: ItemRef) -> &'a str {
        let (src, entry) = self.item(item);
        entry.name.map_or("<anonymous>", |name| src.text(name))
    }
}
