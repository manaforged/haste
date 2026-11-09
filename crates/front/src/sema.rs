//! Semantic analysis of reached function bodies at concrete types, lowered to `haste-codegen` SSA.

pub mod adt;
mod builtin;
mod expand;
mod format;
pub mod impls;
pub mod infer;
pub mod intrinsic;
pub mod layout;
pub mod reach;
pub mod traits;
pub mod ty;

use rustc_hash::FxHashMap as HashMap;
use std::rc::Rc;

use anyhow::{Context as _, Result, anyhow, bail};

use crate::body::sig::{self, Signature};
use crate::body::adt::{AdtDef, adt as parse_adt};
use crate::body::Body;
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
    adts: HashMap<ItemRef, Rc<AdtDef>>,
    layouts: HashMap<ty::Ty, Rc<layout::Layout>>,
    tails: HashMap<ty::Ty, ty::Ty>,
    headers: HashMap<impls::ImplRef, Rc<sig::ImplHeader>>,
    finals: HashMap<reach::Instance, infer::Typed>,
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
    adts: HashMap<ItemRef, Rc<AdtDef>>,
    layouts: HashMap<ty::Ty, Rc<layout::Layout>>,
    headers: HashMap<impls::ImplRef, Rc<sig::ImplHeader>>,
    heads: HashMap<impls::Holder, Rc<impls::Index>>,
    owners: HashMap<u32, Rc<HashMap<(u32, Option<u32>), u32>>>,
    trait_defs: HashMap<ItemRef, Rc<traits::TraitGenerics>>,
    discriminants: HashMap<ItemRef, Rc<[i128]>>,
    langs: Option<(Rc<HashMap<String, ItemRef>>, Rc<[u32]>)>,
    drops: HashMap<ty::Ty, bool>,
    closures: HashMap<ty::Ty, Rc<infer::ClosureInfo>>,
    finals: HashMap<reach::Instance, infer::Typed>,
    tails: HashMap<ty::Ty, ty::Ty>,
    reveals: HashMap<reach::Instance, Option<ty::Ty>>,
}

impl<'a> Context<'a> {
    /// A pass over an outline and its resolved names that reuses `store`.
    pub fn new(outline: &'a Outline, names: &'a Names, store: Store) -> Context<'a> {
        let Store { types, signatures, bodies, adts, layouts, tails, headers, finals } = store;
        let (modules, heads, owners) = (HashMap::default(), HashMap::default(), HashMap::default());
        let (trait_defs, langs, drops) = (HashMap::default(), None, HashMap::default());
        Context { outline, names, types, asks: Asks::default(), modules, signatures, bodies, adts, layouts, headers, heads, owners, trait_defs, discriminants: HashMap::default(), langs, drops, closures: HashMap::default(), finals, reveals: HashMap::default(), tails }
    }

    /// Ends the pass: the store for the next one, and what lookups asked the front end to load or resolve.
    pub fn release(self) -> (Store, Asks) {
        (Store { types: self.types, signatures: self.signatures, bodies: self.bodies, adts: self.adts, layouts: self.layouts, tails: self.tails, headers: self.headers, finals: self.finals }, self.asks)
    }

    /// The source file and outline entry of `item`.
    pub fn item(&self, item: ItemRef) -> (&'a Source, &'a Item) {
        let file = self.file(item);
        let index = item.outer().map_or(item.item, |(_, index)| index);
        (&file.source, &file.items.items[index as usize])
    }

    /// Whether `item` is a function declared in an `extern` block: a signature with no body.
    pub fn foreign(&self, item: ItemRef) -> bool {
        let (_, entry) = self.item(item);
        let items = &self.file(item).items.items;
        item.outer().is_none() && entry.kind == ItemKind::Fn && entry.body.is_none() && entry.parent.is_some_and(|parent| items[parent as usize].kind == ItemKind::ExternBlock)
    }

    /// What the body that defines the closure type `ty` inferred for it.
    pub fn closure(&self, ty: ty::Ty) -> Option<Rc<infer::ClosureInfo>> {
        self.closures.get(&ty).cloned()
    }

    /// Records what the body that defines the closure type `ty` inferred for it.
    pub fn set_closure(&mut self, ty: ty::Ty, info: infer::ClosureInfo) {
        self.closures.insert(ty, Rc::new(info));
    }

    /// The tuple of the captures of the closure type `ty`, which is how the closure is laid out and dropped.
    pub fn closure_env(&self, ty: ty::Ty) -> Option<ty::Ty> {
        self.closures.get(&ty).map(|info| info.env)
    }

    /// Whether the code of the closure type `ty` takes the address of its environment first, which only a closure that captures something does.
    pub fn closure_takes_env(&self, ty: ty::Ty) -> bool {
        self.closures.get(&ty).is_some_and(|info| !info.closure.captures.is_empty())
    }

    /// The type that `ty` owns when it is a `Box`.
    pub fn boxed(&self, ty: ty::Ty) -> Option<ty::Ty> {
        match self.types.kind(ty) {
            ty::Kind::Adt(item, args) if self.lang(*item) == Some("owned_box") => args.first().copied(),
            _ => None,
        }
    }

    /// Whether a call of the closure type `ty` consumes the closure, because its body moves a captured value out.
    pub fn closure_consumes(&self, ty: ty::Ty) -> bool {
        self.closures.get(&ty).is_some_and(|info| info.closure.consumes)
    }

    /// `FnOnce::call_once`, which every call of a closure value is an instance of.
    pub fn call_once(&mut self) -> Option<ItemRef> {
        let of = self.lang_item("fn_once")?;
        self.trait_member(of, "call_once")
    }

    /// The outline file that holds `item`, or the expansion of the body that declares it.
    pub fn file(&self, item: ItemRef) -> &'a crate::outline::tree::File {
        match item.outer() {
            Some((outer, _)) => &self.expansion(outer).expect("a body that declares items is read").file,
            None => &self.outline.crates[item.krate as usize].files[item.file as usize],
        }
    }

    /// Path resolution inside crate `krate`, recording what it finds not loaded yet.
    pub fn paths(&self, krate: u32) -> Paths<'_> {
        Paths { demands: Some(&self.asks), ..self.names.paths(self.outline, krate) }
    }

    /// The module that holds `item`, through any impl, trait or extern block around it.
    pub fn module_of(&mut self, item: ItemRef) -> Result<ModRef> {
        if let Some((outer, _)) = item.outer() {
            return self.module_of(outer);
        }
        if let Some(block) = self.owner_impl(item) {
            return self.impl_module(block);
        }
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
        let parsed = match (entry.kind, entry.name) {
            (ItemKind::Const | ItemKind::Static, Some(name)) => sig::const_signature(src, name, entry.end)?,
            _ => sig::signature(src, entry.start, entry.body.unwrap_or(entry.end))?,
        };
        let parsed = Rc::new(parsed);
        self.signatures.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The parsed body of the function `item`.
    pub fn body(&mut self, item: ItemRef) -> Result<Rc<Body>> {
        if let Some(found) = self.bodies.get(&item) {
            return Ok(found.clone());
        }
        let mut parsed = self.parse_body(item)?;
        let sig = self.signature(item)?;
        if sig.params.iter().any(|&(pat, _)| !sig.body.plain_param(pat)) {
            parsed = crate::body::parse_params(self.body_src(item), parsed, &sig.param_starts)?;
        }
        let parsed = Rc::new(parsed);
        self.bodies.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The position of the variant with declaration index `index` among the configured variants of the enum `item`.
    pub fn variant_position(&mut self, item: ItemRef, index: u32) -> Result<usize> {
        match self.adt_def(item)?.position(index) {
            Some(at) => Ok(at),
            None => bail!("variant {index} of {} is configured out", self.name(item)),
        }
    }

    /// The parsed definition of the struct, enum or union `item`.
    pub fn adt_def(&mut self, item: ItemRef) -> Result<Rc<AdtDef>> {
        if let Some(found) = self.adts.get(&item) {
            return Ok(found.clone());
        }
        let (src, entry) = self.item(item);
        let name = entry.name.ok_or_else(|| anyhow!("type item has no name"))?;
        let mut parsed = parse_adt(src, name, entry.end, entry.kind == ItemKind::Enum)?;
        let env = &self.outline.crates[item.krate as usize].env;
        parsed.configure(src, &|attrs| env.enabled(attrs));
        let parsed = Rc::new(parsed);
        self.adts.insert(item, parsed.clone());
        Ok(parsed)
    }

    /// The type parameters, arena and target type of the type alias `item`.
    pub fn alias_parts(&self, item: ItemRef) -> Result<(Vec<sig::Generic>, Body, crate::body::TyId)> {
        let (src, entry) = self.item(item);
        let Some(at) = entry.name else { bail!("type alias with no name") };
        let Some(eq) = (at + 1..entry.end).find(|&token| src.kind(token) == rustc_lexer::TokenKind::Eq) else { bail!("type `{}` has no `=`", src.text(at)) };
        let params = match src.text(at + 1) {
            "<" => sig::generics(src, at + 1, eq)?.1.params.into_iter().filter(|generic| generic.kind != sig::GenericKind::Lifetime).collect(),
            _ => Vec::new(),
        };
        let (arena, ty) = crate::body::parse_ty(src, eq + 1, entry.end - 1)?;
        Ok((params, arena, ty))
    }

    /// The name of `item`, for messages.
    pub fn name(&self, item: ItemRef) -> &'a str {
        let (src, entry) = self.item(item);
        entry.name.map_or("<anonymous>", |name| src.text(name))
    }
}
