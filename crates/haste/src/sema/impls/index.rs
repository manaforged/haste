use rustc_hash::FxHashMap as HashMap;

use crate::body::sig::ImplHeader;
use crate::body::{self as syntax, Body};
use crate::names::path::Path as NamePath;
use crate::names::{Def, ItemRef, ModRef};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;

use super::super::Context;
use super::super::infer::segments;
use super::super::ty::Mut;
use super::{Head, ImplRef, impl_file_entry};

#[derive(Clone, Default)]
pub(super) struct Asked {
    modules: Vec<crate::names::ModRef>,
    names: Vec<(crate::names::ModRef, crate::names::Sym)>,
    crates: Vec<u32>,
    scopes: Vec<crate::names::ModRef>,
    procs: Vec<(ItemRef, String)>,
}

impl Asked {
    pub(super) fn from_requests(requests: crate::names::path::Requests) -> Self {
        let mut modules = requests.modules.into_inner();
        modules.sort_unstable_by_key(|m| (m.krate, m.module));
        modules.dedup();
        let mut names = requests.names.into_inner();
        names.sort_unstable_by_key(|&(m, s)| (m.krate, m.module, s.0));
        names.dedup();
        let mut crates = requests.crates.into_inner();
        crates.sort_unstable();
        crates.dedup();
        let mut scopes = requests.scopes.into_inner();
        scopes.sort_unstable_by_key(|m| (m.krate, m.module));
        scopes.dedup();
        let mut procs = requests.procs.into_inner();
        procs.sort_unstable_by(|a, b| {
            (a.0.krate, a.0.file, a.0.item, a.1.as_str()).cmp(&(
                b.0.krate,
                b.0.file,
                b.0.item,
                b.1.as_str(),
            ))
        });
        procs.dedup();
        Self {
            modules,
            names,
            crates,
            scopes,
            procs,
        }
    }

    pub(super) fn clear(&mut self) {
        self.modules.clear();
        self.names.clear();
        self.crates.clear();
        self.scopes.clear();
        self.procs.clear();
    }

    pub(super) fn is_empty(&self) -> bool {
        self.modules.is_empty()
            && self.names.is_empty()
            && self.crates.is_empty()
            && self.scopes.is_empty()
            && self.procs.is_empty()
    }

    #[inline]
    pub(super) fn replay(&self, into: &crate::names::path::Requests) {
        if self.is_empty() {
            return;
        }
        if !self.modules.is_empty() {
            into.modules
                .borrow_mut()
                .extend(self.modules.iter().copied());
        }
        if !self.names.is_empty() {
            into.names.borrow_mut().extend(self.names.iter().copied());
        }
        if !self.crates.is_empty() {
            into.crates.borrow_mut().extend(self.crates.iter().copied());
        }
        if !self.scopes.is_empty() {
            into.scopes.borrow_mut().extend(self.scopes.iter().copied());
        }
        if !self.procs.is_empty() {
            into.procs.borrow_mut().extend(self.procs.iter().cloned());
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct TraitBucket {
    pub(crate) all: Vec<ImplRef>,
    pub(crate) blanket: Vec<ImplRef>,
    pub(crate) by_head: HashMap<Head, Vec<ImplRef>>,
}

/// The impls of one crate: inherent ones by the head of their self type, trait ones by their trait.
#[derive(Clone, Default)]
pub struct Index {
    pub(crate) inherent: HashMap<Head, Vec<ImplRef>>,
    pub(crate) traits: HashMap<ItemRef, TraitBucket>,
    pub(crate) negative: HashMap<ItemRef, Vec<Head>>,
    pub(super) asked: Asked,
    pub(super) settled_crate: bool,
    pub(super) incomplete: bool,
}

impl Index {
    /// Whether building the index asked for nothing, so it holds every impl its holder has.
    pub fn complete(&self) -> bool {
        self.asked.is_empty()
    }

    pub(in crate::sema) fn reusable_after_names(&self) -> bool {
        self.settled_crate && !self.incomplete
    }

    pub(in crate::sema) fn clear_asked(&mut self) {
        self.asked.clear();
    }

    pub(super) fn merge_from(&mut self, other: Index) {
        self.incomplete |= other.incomplete;
        for (head, blocks) in other.inherent {
            self.inherent.entry(head).or_default().extend(blocks);
        }
        for (of, bucket) in other.traits {
            let dst = self.traits.entry(of).or_default();
            dst.all.extend(bucket.all);
            dst.blanket.extend(bucket.blanket);
            for (head, blocks) in bucket.by_head {
                dst.by_head.entry(head).or_default().extend(blocks);
            }
        }
        for (of, heads) in other.negative {
            self.negative.entry(of).or_default().extend(heads);
        }
    }

    pub(crate) fn from_parts(
        inherent: HashMap<Head, Vec<ImplRef>>,
        traits: HashMap<ItemRef, TraitBucket>,
        negative: HashMap<ItemRef, Vec<Head>>,
    ) -> Self {
        Self {
            inherent,
            traits,
            negative,
            asked: Asked::default(),
            settled_crate: true,
            incomplete: false,
        }
    }
}

impl<'a> Context<'a> {
    pub(super) fn index_block(&mut self, index: &mut Index, block: ImplRef) {
        let Ok(header) = self.impl_header(block) else {
            index.incomplete = true;
            return;
        };
        let (file, entry) = impl_file_entry(self.outline, block);
        let src = &file.source;
        let Ok(home) = self.impl_module(block) else {
            index.incomplete = true;
            return;
        };
        let asks_before = self.asks.count();
        let head = if generic_self(src, &header) {
            None
        } else {
            let found = self.syntax_head(src, &header.body, header.self_ty, (home, block.item()));
            if found.is_none() && self.asks.count() > asks_before {
                index.incomplete = true;
            }
            found
        };
        if header.negative {
            if let Some(of) = self.impl_trait_with(block, &header, src, home) {
                if let Some(head) = head {
                    index.negative.entry(of).or_default().push(head);
                }
            } else if header.trait_ref.is_some() {
                index.incomplete = true;
            }
            return;
        }
        match (header.trait_ref.is_some(), head) {
            (false, Some(head)) => index.inherent.entry(head).or_default().push(block),
            (false, None) => {}
            (true, _) => {
                let reserved = entry.attrs.iter().any(|attr| {
                    matches!(attr, crate::outline::item::Attr::Other(at) if src.text(*at) == "rustc_reservation_impl")
                });
                if !reserved {
                    if let Some(of) = self.impl_trait_with(block, &header, src, home) {
                        let bucket = index.traits.entry(of).or_default();
                        bucket.all.push(block);
                        match head {
                            Some(head) => bucket.by_head.entry(head).or_default().push(block),
                            None => bucket.blanket.push(block),
                        }
                    } else {
                        index.incomplete = true;
                    }
                }
            }
        }
    }

    fn syntax_head(
        &mut self,
        src: &Source,
        arena: &Body,
        ty: syntax::TyId,
        home: (ModRef, Option<ItemRef>),
    ) -> Option<Head> {
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

    fn dyn_head(
        &mut self,
        src: &Source,
        arena: &Body,
        bounds: crate::body::List,
        (home, root): (ModRef, Option<ItemRef>),
    ) -> Option<Head> {
        arena.list(bounds).iter().find_map(|&at| {
            let crate::body::Bound::Trait { path, .. } = arena.bounds[at as usize] else {
                return None;
            };
            let (segments, absolute, dollar) = segments(&self.names.symbols, src, arena, path);
            match self.resolve_in(
                root,
                &NamePath {
                    segments: &segments,
                    absolute,
                    dollar,
                },
                home,
            )? {
                Def::Item(item)
                    if self.item(item).1.kind == ItemKind::Trait && !self.is_auto_trait(item) =>
                {
                    Some(Head::Dyn(item))
                }
                _ => None,
            }
        })
    }

    fn path_head(
        &mut self,
        src: &Source,
        arena: &Body,
        path: syntax::PathId,
        (home, root): (ModRef, Option<ItemRef>),
    ) -> Option<Head> {
        let (segments, absolute, dollar) = segments(&self.names.symbols, src, arena, path);
        match self.resolve_in(
            root,
            &NamePath {
                segments: &segments,
                absolute,
                dollar,
            },
            home,
        )? {
            Def::Primitive(primitive) => Some(Head::Prim(primitive)),
            Def::Item(item)
                if matches!(
                    self.item(item).1.kind,
                    ItemKind::Struct | ItemKind::Enum | ItemKind::Union
                ) || self.foreign(item) =>
            {
                Some(Head::Adt(item))
            }
            Def::Item(item) if self.item(item).1.kind == ItemKind::Type => {
                let alias = self.alias_def(item).ok()?;
                let home = self.module_of(item).ok()?;
                self.syntax_head(
                    self.item(item).0,
                    &alias.arena,
                    alias.ty,
                    (home, Some(item)),
                )
            }
            _ => None,
        }
    }
}

fn generic_self(src: &Source, header: &ImplHeader) -> bool {
    let syntax::Ty::Path(path) = *header.body.ty(header.self_ty) else {
        return false;
    };
    let segments = header.body.list(header.body.path(path).segments);
    let [only] = segments else { return false };
    let name = src.text(header.body.segments[*only as usize].name);
    header
        .generics
        .iter()
        .any(|generic| src.text(generic.name) == name)
}
