use std::rc::Rc;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::body::sig::{self, ImplHeader};
use crate::body::{self as syntax, Body};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef, Ns, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::outline::tree::File;

use super::Context;
use super::body_scope::{BodyPath, outside};
use super::infer::segments;
use super::ty::{Kind, Mut};

/// One impl block: one that a module holds, by its index among its crate's impls, or one that a function body declares, by its item there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImplRef {
    Module { krate: u32, index: u32 },
    Body(ItemRef),
}

impl ImplRef {
    /// The function whose body declares the impl, when a body does.
    pub fn root(self) -> Option<ItemRef> {
        match self {
            ImplRef::Module { .. } => None,
            ImplRef::Body(item) => item.declaring_body(),
        }
    }

    /// The impl's item in the body that declares it, when a body does.
    pub fn item(self) -> Option<ItemRef> {
        match self {
            ImplRef::Module { .. } => None,
            ImplRef::Body(item) => Some(item),
        }
    }
}

/// What holds a set of impls: the modules of a crate, or the body of a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Holder {
    Crate(u32),
    Body(ItemRef),
}

/// The outer shape of a type, by which its inherent impls are found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Head {
    Prim(Primitive),
    Adt(ItemRef),
    Ref(Mut),
    Ptr(Mut),
    Slice,
    Array,
    Tuple,
    Never,
    Dyn(ItemRef),
}

/// The head of a type of shape `kind`, or none for a type with no impls of its own.
pub fn head(kind: &Kind) -> Option<Head> {
    Some(match kind {
        Kind::Prim(primitive) => Head::Prim(*primitive),
        Kind::Adt(item, _) => Head::Adt(*item),
        Kind::Ref(mutability, _) => Head::Ref(*mutability),
        Kind::Ptr(mutability, _) => Head::Ptr(*mutability),
        Kind::Slice(_) => Head::Slice,
        Kind::Array(..) => Head::Array,
        Kind::Tuple(_) => Head::Tuple,
        Kind::Never => Head::Never,
        Kind::Dyn(principal, ..) => Head::Dyn(*principal),
        _ => return None,
    })
}

/// The impls of one crate: inherent ones by the head of their self type, trait ones by their trait.
#[derive(Default)]
pub struct Index {
    inherent: HashMap<Head, Vec<ImplRef>>,
    traits: HashMap<ItemRef, Vec<(ImplRef, Option<Head>)>>,
    asked: crate::names::path::Requests,
}

const PRIMITIVE_HOLDERS: [&str; 3] = ["core", "alloc", "std"];

impl Index {
    /// Whether building the index asked for nothing, so it holds every impl its holder has.
    pub fn complete(&self) -> bool {
        self.asked.count() == 0
    }
}

impl<'a> Context<'a> {
    /// The parsed header of impl `block`.
    pub fn impl_header(&mut self, block: ImplRef) -> Result<Rc<ImplHeader>> {
        if let Some(found) = self.store.headers.get(&block) {
            return Ok(found.clone());
        }
        let (file, item) = (self.impl_file(block), self.impl_entry(block));
        let parsed = Rc::new(sig::impl_header(&file.source, item.start, item.body.unwrap_or(item.end))?);
        self.store.headers.insert(block, parsed.clone());
        Ok(parsed)
    }

    fn impl_entry(&self, block: ImplRef) -> &'a crate::outline::item::Item {
        let at = match block {
            ImplRef::Module { krate, index } => self.outline.crates[krate as usize].impls[index as usize].at,
            ImplRef::Body(item) => item.outer().map_or(item.item, |(_, at)| at),
        };
        &self.impl_file(block).items.items[at as usize]
    }

    /// Whether impl `block` is a `#[rustc_reservation_impl]`, which selection never picks.
    pub fn reserved(&self, block: ImplRef) -> bool {
        let src = &self.impl_file(block).source;
        self.impl_entry(block).attrs.iter().any(|attr| matches!(attr, crate::outline::item::Attr::Other(at) if src.text(*at) == "rustc_reservation_impl"))
    }

    /// The outline file that holds the header of impl `block`.
    pub fn impl_file(&self, block: ImplRef) -> &'a File {
        match block {
            ImplRef::Module { krate, index } => {
                let tree = &self.outline.crates[krate as usize];
                &tree.files[tree.impls[index as usize].file]
            }
            ImplRef::Body(item) => self.file(item),
        }
    }

    /// The module around impl `block`.
    pub fn impl_module(&mut self, block: ImplRef) -> Result<ModRef> {
        match block {
            ImplRef::Module { krate, index } => Ok(ModRef { krate, module: self.outline.crates[krate as usize].impls[index as usize].module as u32 }),
            ImplRef::Body(item) => self.module_of(item),
        }
    }

    /// The functions, constants and types of impl `block`, its own and those its macro calls expand to.
    pub fn members(&self, block: ImplRef) -> Vec<ItemRef> {
        let (krate, index) = match block {
            ImplRef::Module { krate, index } => (krate, index),
            ImplRef::Body(item) => return self.body_members(item),
        };
        let tree = &self.outline.crates[krate as usize];
        let entry = &tree.impls[index as usize];
        let parts = std::iter::once((entry.file, Some(entry.at))).chain(entry.parts.iter().map(|&part| (part, None)));
        let mut found = Vec::new();
        for (file, owner) in parts {
            let items = &tree.files[file].items.items;
            for &at in self.kids_of(items, owner).iter() {
                let item = &items[at as usize];
                if matches!(item.kind, ItemKind::Fn | ItemKind::Const | ItemKind::Type) && tree.env.enabled(&item.attrs) {
                    found.push(ItemRef { krate, file: file as u32, item: at });
                }
            }
        }
        found
    }

    fn body_members(&self, block: ItemRef) -> Vec<ItemRef> {
        self.children(block).filter(|member| matches!(self.item(*member).1.kind, ItemKind::Fn | ItemKind::Const | ItemKind::Type)).collect()
    }

    /// The member of impl `block` named `name`.
    pub fn member(&self, block: ImplRef, name: &str) -> Option<ItemRef> {
        let known = self.impl_members.borrow().get(&block).cloned();
        let members = known.unwrap_or_else(|| {
            let mut members = HashMap::default();
            for member in self.members(block) {
                members.entry(self.name(member)).or_insert(member);
            }
            let members = Rc::new(members);
            self.impl_members.borrow_mut().insert(block, members.clone());
            members
        });
        members.get(name).copied()
    }

    /// The impl block whose member `item` is.
    pub fn owner_impl(&mut self, item: ItemRef) -> Option<ImplRef> {
        if item.declaring_body().is_some() {
            return self.parent_item(item).filter(|parent| self.item(*parent).1.kind == ItemKind::Impl).map(ImplRef::Body);
        }
        let owners = self.owners_of(item.krate);
        let file = self.file(item);
        let key = match file.items.items[item.item as usize].parent {
            Some(parent) if file.items.items[parent as usize].kind == ItemKind::Impl => (item.file, Some(parent)),
            Some(_) => return None,
            None => (item.file, None),
        };
        owners.get(&key).map(|&index| ImplRef::Module { krate: item.krate, index })
    }

    fn owners_of(&mut self, krate: u32) -> Rc<HashMap<(u32, Option<u32>), u32>> {
        let tree = &self.outline.crates[krate as usize];
        self.store.lookups.owners
            .entry(krate)
            .or_insert_with(|| {
                let mut map = HashMap::default();
                for (index, entry) in tree.impls.iter().enumerate() {
                    map.insert((entry.file as u32, Some(entry.at)), index as u32);
                    map.extend(entry.parts.iter().map(|&part| ((part as u32, None), index as u32)));
                }
                Rc::new(map)
            })
            .clone()
    }

    /// The inherent impls of types with head `head`, asking for every module of the crates that may hold one.
    pub fn inherent(&mut self, head: Head) -> Vec<ImplRef> {
        let standard = self.primitive_holders();
        let holders = match head {
            Head::Adt(item) | Head::Dyn(item) if !standard.contains(&item.krate) => vec![item.krate],
            _ => standard,
        };
        let mut found = Vec::new();
        for holder in self.holders_with_body(&holders, (Some(head), None)) {
            found.extend(self.index_of(holder).inherent.get(&head).into_iter().flatten().copied());
        }
        found
    }

    fn holders_with_body(&self, crates: &[u32], (head, of): (Option<Head>, Option<ItemRef>)) -> Vec<Holder> {
        let mut holders: Vec<Holder> = crates.iter().map(|&krate| Holder::Crate(krate)).collect();
        let local_self = match head {
            Some(Head::Adt(item)) => item.declaring_body(),
            _ => None,
        };
        for root in [local_self, of.and_then(ItemRef::declaring_body)].into_iter().flatten() {
            let in_crate_index = crates.contains(&root.krate) && self.outline.crates[root.krate as usize].impl_bodies.iter().any(|&(file, item)| file as u32 == root.file && item == root.item);
            if !in_crate_index && !holders.contains(&Holder::Body(root)) {
                holders.push(Holder::Body(root));
            }
        }
        holders
    }

    /// The impls of trait `of` in crates `holders` whose self type may have head `head`.
    pub fn trait_impls(&mut self, of: ItemRef, head: Option<Head>, holders: &[u32]) -> Vec<ImplRef> {
        let mut found = Vec::new();
        for holder in self.holders_with_body(holders, (head, Some(of))) {
            let index = self.index_of(holder);
            let blocks = index.traits.get(&of).into_iter().flatten();
            found.extend(blocks.filter(|(_, own)| own.is_none() || head.is_none() || *own == head).map(|(block, _)| *block));
        }
        found
    }

    /// The trait that impl `block` implements, or none for an inherent impl.
    pub fn impl_trait(&mut self, block: ImplRef) -> Option<ItemRef> {
        if let Some(&known) = self.store.lookups.impl_traits.get(&block) {
            return known;
        }
        let found = self.impl_trait_here(block);
        self.store.lookups.impl_traits.insert(block, found);
        found
    }

    fn impl_trait_here(&mut self, block: ImplRef) -> Option<ItemRef> {
        let header = self.impl_header(block).ok()?;
        let syntax::Ty::Path(path) = *header.body.ty(header.trait_ref?) else { return None };
        let src = &self.impl_file(block).source;
        let home = self.impl_module(block).ok()?;
        let (segments, absolute, dollar) = segments(&self.names.symbols, src, &header.body, path);
        match self.resolve_in(block.item(), &NamePath { segments: &segments, absolute, dollar }, home)? {
            Def::Item(item) if self.item(item).1.kind == ItemKind::Trait => Some(item),
            _ => None,
        }
    }

    /// The item that `path`, written at the item `at`, names from `home`, first among the items that the body declaring `at` holds.
    fn resolve_in(&self, at: Option<ItemRef>, path: &NamePath<'_>, home: ModRef) -> Option<Def> {
        let anchor = at.and_then(ItemRef::outer).map(|(root, from)| (root, Some(from)));
        let paths = self.paths(home.krate);
        match self.local_path(anchor, path, Ns::Types) {
            Some(BodyPath::Item(item)) => Some(Def::Item(item)),
            Some(BodyPath::Variant(_)) => None,
            Some(BodyPath::Outside(skip)) => paths.resolve(&outside(path, skip), home),
            None => anchor.and_then(|(root, _)| self.body_use_path((root, home), path, at.map(|at| self.item(at).1.start))).or_else(|| paths.resolve(path, home)),
        }
    }

    /// The standard crates, which hold the impls and lang items of primitive types.
    pub fn primitive_holders(&self) -> Vec<u32> {
        let roots = self.outline.roots.iter().enumerate();
        roots.filter(|(_, root)| root.role == crate::outline::Role::Lib && PRIMITIVE_HOLDERS.contains(&root.name.as_str())).map(|(krate, _)| krate as u32).collect()
    }

    fn index_of(&mut self, holder: Holder) -> Rc<Index> {
        if let Holder::Crate(krate) = holder
            && self.outline.crates[krate as usize].partial()
        {
            self.asks.crates.borrow_mut().push(krate);
        }
        if let Some(found) = self.store.lookups.heads.get(&holder) {
            self.asks.repeat(&found.asked);
            return found.clone();
        }
        if let Holder::Body(root) = holder
            && self.expansion(root).is_none()
        {
            return Rc::default();
        }
        let blocks = match holder {
            Holder::Crate(krate) => self.crate_impls(krate),
            Holder::Body(root) => self.body_impls(root),
        };
        let mark = self.asks.mark();
        let mut index = Index::default();
        for block in blocks {
            self.index_block(&mut index, block);
        }
        index.asked = self.asks.since(mark);
        let index = Rc::new(index);
        self.store.lookups.heads.insert(holder, index.clone());
        index
    }

    /// The impls of crate `krate`: those of its modules, then those in the bodies of its module-level `const _` items and functions.
    fn crate_impls(&mut self, krate: u32) -> Vec<ImplRef> {
        let tree = &self.outline.crates[krate as usize];
        let mut blocks: Vec<ImplRef> = (0..tree.impls.len() as u32).map(|index| ImplRef::Module { krate, index }).collect();
        for &(file, item) in &tree.impl_bodies {
            let root = ItemRef { krate, file: file as u32, item };
            if self.parse_body(root).is_ok() {
                blocks.extend(self.body_impls(root));
            }
        }
        blocks
    }

    fn body_impls(&self, root: ItemRef) -> Vec<ImplRef> {
        let Some(found) = self.expansion(root) else { return Vec::new() };
        let items = &found.file.items.items;
        let blocks = items.iter().enumerate().filter(|(_, item)| item.kind == ItemKind::Impl);
        blocks.map(|(at, _)| ImplRef::Body(root.inner(at as u32))).collect()
    }

    fn index_block(&mut self, index: &mut Index, block: ImplRef) {
        let Ok(header) = self.impl_header(block) else { return };
        if header.negative {
            return;
        }
        let src = &self.impl_file(block).source;
        let Ok(home) = self.impl_module(block) else { return };
        let head = if generic_self(src, &header) { None } else { self.syntax_head(src, &header.body, header.self_ty, (home, block.item())) };
        match (header.trait_ref.is_some(), head) {
            (false, Some(head)) => index.inherent.entry(head).or_default().push(block),
            (false, None) => {}
            (true, _) => {
                if let Some(of) = self.impl_trait(block) {
                    index.traits.entry(of).or_default().push((block, head));
                }
            }
        }
    }

    fn syntax_head(&mut self, src: &Source, arena: &Body, ty: syntax::TyId, home: (ModRef, Option<ItemRef>)) -> Option<Head> {
        Some(match *arena.ty(ty) {
            syntax::Ty::Paren(inner) => return self.syntax_head(src, arena, inner, home),
            syntax::Ty::Ref { mutable, .. } => Head::Ref(if mutable { Mut::Mut } else { Mut::Not }),
            syntax::Ty::Ptr { mutable, .. } => Head::Ptr(if mutable { Mut::Mut } else { Mut::Not }),
            syntax::Ty::Slice(_) => Head::Slice,
            syntax::Ty::Array(..) => Head::Array,
            syntax::Ty::Tuple(_) => Head::Tuple,
            syntax::Ty::Never => Head::Never,
            syntax::Ty::Path(path) => return self.path_head(src, arena, path, home),
            syntax::Ty::DynTrait(bounds) => return self.dyn_head(src, arena, bounds, home),
            _ => return None,
        })
    }

    fn dyn_head(&mut self, src: &Source, arena: &Body, bounds: crate::body::List, (home, root): (ModRef, Option<ItemRef>)) -> Option<Head> {
        arena.list(bounds).iter().find_map(|&at| {
            let crate::body::Bound::Trait { path, .. } = arena.bounds[at as usize] else { return None };
            let (segments, absolute, dollar) = segments(&self.names.symbols, src, arena, path);
            match self.resolve_in(root, &NamePath { segments: &segments, absolute, dollar }, home)? {
                Def::Item(item) if self.item(item).1.kind == ItemKind::Trait && !self.is_auto_trait(item) => Some(Head::Dyn(item)),
                _ => None,
            }
        })
    }

    fn path_head(&mut self, src: &Source, arena: &Body, path: syntax::PathId, (home, root): (ModRef, Option<ItemRef>)) -> Option<Head> {
        let (segments, absolute, dollar) = segments(&self.names.symbols, src, arena, path);
        match self.resolve_in(root, &NamePath { segments: &segments, absolute, dollar }, home)? {
            Def::Primitive(primitive) => Some(Head::Prim(primitive)),
            Def::Item(item) if matches!(self.item(item).1.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union) => Some(Head::Adt(item)),
            Def::Item(item) if self.item(item).1.kind == ItemKind::Type => {
                let alias = self.alias_def(item).ok()?;
                let home = self.module_of(item).ok()?;
                self.syntax_head(self.item(item).0, &alias.arena, alias.ty, (home, Some(item)))
            }
            _ => None,
        }
    }
}

fn generic_self(src: &Source, header: &ImplHeader) -> bool {
    let syntax::Ty::Path(path) = *header.body.ty(header.self_ty) else { return false };
    let segments = header.body.list(header.body.path(path).segments);
    let [only] = segments else { return false };
    let name = src.text(header.body.segments[*only as usize].name);
    header.generics.iter().any(|generic| src.text(generic.name) == name)
}
