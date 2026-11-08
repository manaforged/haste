use std::rc::Rc;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::body::sig::{self, ImplHeader};
use crate::body::{self as syntax, Body};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef, Primitive};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::outline::tree::File;

use super::Context;
use super::infer::segments;
use super::ty::{Kind, Mut};

/// One impl block: its crate and its index among the crate's impls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImplRef {
    pub krate: u32,
    pub index: u32,
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
        _ => return None,
    })
}

/// The impls of one crate: inherent ones by the head of their self type, trait ones by their trait.
#[derive(Default)]
pub struct Index {
    inherent: HashMap<Head, Vec<ImplRef>>,
    traits: HashMap<ItemRef, Vec<(ImplRef, Option<Head>)>>,
}

const PRIMITIVE_HOLDERS: [&str; 3] = ["core", "alloc", "std"];

impl<'a> Context<'a> {
    /// The parsed header of impl `block`.
    pub fn impl_header(&mut self, block: ImplRef) -> Result<Rc<ImplHeader>> {
        if let Some(found) = self.headers.get(&block) {
            return Ok(found.clone());
        }
        let file = self.impl_file(block);
        let entry = &self.outline.crates[block.krate as usize].impls[block.index as usize];
        let item = &file.items.items[entry.at as usize];
        let parsed = Rc::new(sig::impl_header(&file.source, item.start, item.body.unwrap_or(item.end))?);
        self.headers.insert(block, parsed.clone());
        Ok(parsed)
    }

    /// The outline file that holds the header of impl `block`.
    pub fn impl_file(&self, block: ImplRef) -> &'a File {
        let tree = &self.outline.crates[block.krate as usize];
        &tree.files[tree.impls[block.index as usize].file]
    }

    /// The module around impl `block`.
    pub fn impl_module(&self, block: ImplRef) -> ModRef {
        ModRef { krate: block.krate, module: self.outline.crates[block.krate as usize].impls[block.index as usize].module as u32 }
    }

    /// The functions, constants and types of impl `block`, its own and those its macro calls expand to.
    pub fn members(&self, block: ImplRef) -> Vec<ItemRef> {
        let tree = &self.outline.crates[block.krate as usize];
        let entry = &tree.impls[block.index as usize];
        let parts = std::iter::once((entry.file, Some(entry.at))).chain(entry.parts.iter().map(|&part| (part, None)));
        let mut found = Vec::new();
        for (file, owner) in parts {
            for (at, item) in tree.files[file].items.items.iter().enumerate() {
                if item.parent == owner && matches!(item.kind, ItemKind::Fn | ItemKind::Const | ItemKind::Type) && tree.env.enabled(&item.attrs) {
                    found.push(ItemRef { krate: block.krate, file: file as u32, item: at as u32 });
                }
            }
        }
        found
    }

    /// The member of impl `block` named `name`.
    pub fn member(&self, block: ImplRef, name: &str) -> Option<ItemRef> {
        self.members(block).into_iter().find(|member| self.name(*member) == name)
    }

    /// The impl block whose member `item` is.
    pub fn owner_impl(&mut self, item: ItemRef) -> Option<ImplRef> {
        if item.outer().is_some() {
            return None;
        }
        let owners = self.owners_of(item.krate);
        let file = self.file(item);
        let key = match file.items.items[item.item as usize].parent {
            Some(parent) if file.items.items[parent as usize].kind == ItemKind::Impl => (item.file, Some(parent)),
            Some(_) => return None,
            None => (item.file, None),
        };
        owners.get(&key).map(|&index| ImplRef { krate: item.krate, index })
    }

    fn owners_of(&mut self, krate: u32) -> Rc<HashMap<(u32, Option<u32>), u32>> {
        let tree = &self.outline.crates[krate as usize];
        self.owners
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
        let holders = match head {
            Head::Adt(item) => vec![item.krate],
            _ => self.primitive_holders(),
        };
        let mut found = Vec::new();
        for krate in holders {
            found.extend(self.index_of(krate).inherent.get(&head).into_iter().flatten().copied());
        }
        found
    }

    /// The impls of trait `of` in crates `holders` whose self type may have head `head`.
    pub fn trait_impls(&mut self, of: ItemRef, head: Option<Head>, holders: &[u32]) -> Vec<ImplRef> {
        let mut found = Vec::new();
        for &krate in holders {
            let index = self.index_of(krate);
            let blocks = index.traits.get(&of).into_iter().flatten();
            found.extend(blocks.filter(|(_, own)| own.is_none() || head.is_none() || *own == head).map(|(block, _)| *block));
        }
        found
    }

    /// The trait that impl `block` implements, or none for an inherent impl.
    pub fn impl_trait(&mut self, block: ImplRef) -> Option<ItemRef> {
        let header = self.impl_header(block).ok()?;
        let syntax::Ty::Path(path) = *header.body.ty(header.trait_ref?) else { return None };
        let src = &self.impl_file(block).source;
        let home = self.impl_module(block);
        let (segments, absolute, dollar) = segments(&self.names.symbols, src, &header.body, path);
        match self.paths(home.krate).resolve(&NamePath { segments: &segments, absolute, dollar }, home)? {
            Def::Item(item) if self.item(item).1.kind == ItemKind::Trait => Some(item),
            _ => None,
        }
    }

    /// The standard crates, which hold the impls and lang items of primitive types.
    pub fn primitive_holders(&self) -> Vec<u32> {
        let roots = self.outline.roots.iter().enumerate();
        roots.filter(|(_, root)| root.lib && PRIMITIVE_HOLDERS.contains(&root.name.as_str())).map(|(krate, _)| krate as u32).collect()
    }

    fn index_of(&mut self, krate: u32) -> Rc<Index> {
        let tree = &self.outline.crates[krate as usize];
        if tree.partial() {
            self.asks.crates.borrow_mut().push(krate);
        }
        if let Some(found) = self.heads.get(&krate) {
            return found.clone();
        }
        let mut index = Index::default();
        for at in 0..tree.impls.len() as u32 {
            let block = ImplRef { krate, index: at };
            let Ok(header) = self.impl_header(block) else { continue };
            if header.negative {
                continue;
            }
            let src = &self.impl_file(block).source;
            let head = if generic_self(src, &header) { None } else { self.syntax_head(src, &header.body, header.self_ty, self.impl_module(block)) };
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
        let index = Rc::new(index);
        self.heads.insert(krate, index.clone());
        index
    }

    fn syntax_head(&self, src: &Source, arena: &Body, ty: syntax::TyId, home: ModRef) -> Option<Head> {
        Some(match *arena.ty(ty) {
            syntax::Ty::Paren(inner) => return self.syntax_head(src, arena, inner, home),
            syntax::Ty::Ref { mutable, .. } => Head::Ref(if mutable { Mut::Mut } else { Mut::Not }),
            syntax::Ty::Ptr { mutable, .. } => Head::Ptr(if mutable { Mut::Mut } else { Mut::Not }),
            syntax::Ty::Slice(_) => Head::Slice,
            syntax::Ty::Array(..) => Head::Array,
            syntax::Ty::Tuple(_) => Head::Tuple,
            syntax::Ty::Never => Head::Never,
            syntax::Ty::Path(path) => return self.path_head(src, arena, path, home),
            _ => return None,
        })
    }

    fn path_head(&self, src: &Source, arena: &Body, path: syntax::PathId, home: ModRef) -> Option<Head> {
        let (segments, absolute, dollar) = segments(&self.names.symbols, src, arena, path);
        match self.paths(home.krate).resolve(&NamePath { segments: &segments, absolute, dollar }, home)? {
            Def::Primitive(primitive) => Some(Head::Prim(primitive)),
            Def::Item(item) if matches!(self.item(item).1.kind, ItemKind::Struct | ItemKind::Enum | ItemKind::Union) => Some(Head::Adt(item)),
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
