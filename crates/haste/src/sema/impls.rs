use std::sync::Arc;

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::body::sig::{self, ImplHeader};
use crate::body::{self as syntax};
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

mod index;

use index::Asked;
pub use index::Index;
pub(crate) use index::TraitBucket;

const PRIMITIVE_HOLDERS: [&str; 3] = ["core", "alloc", "std"];

fn impl_file_entry(
    outline: &crate::outline::Outline,
    block: ImplRef,
) -> (&File, &crate::outline::item::Item) {
    match block {
        ImplRef::Module { krate, index } => {
            let tree = &outline.crates[krate as usize];
            let entry = &tree.impls[index as usize];
            let file = &tree.files[entry.file];
            (file, &file.items.items[entry.at as usize])
        }
        ImplRef::Body(item) => {
            let file = match item.outer() {
                Some((outer, _)) => {
                    let owner = outer.body_owner();
                    &outline.crates[owner.krate as usize].files[owner.file as usize]
                        .expansion(owner.krate, owner.item)
                        .expect("a body that declares items is read")
                        .file
                }
                None => &outline.crates[item.krate as usize].files[item.file as usize],
            };
            let at = item.outer().map_or(item.item, |(_, at)| at);
            (file, &file.items.items[at as usize])
        }
    }
}

impl<'a> Context<'a> {
    /// The last path segment of the self type of impl `block`, or `None` when it is no path.
    pub fn impl_self_name(&mut self, block: ImplRef) -> Option<String> {
        let header = self.impl_header(block).ok()?;
        let syntax::Ty::Path(path) = *header.body.ty(header.self_ty) else {
            return None;
        };
        let &last = header.body.list(header.body.path(path).segments).last()?;
        Some(
            self.impl_file(block)
                .source
                .text(header.body.segments[last as usize].name)
                .to_owned(),
        )
    }

    fn ensure_impl_header(&mut self, block: ImplRef) -> Result<&Arc<ImplHeader>> {
        use std::collections::hash_map::Entry;
        match self.headers_local.entry(block) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(slot) => {
                let found = if let Some(found) = self.store.headers.get(&block) {
                    found
                } else {
                    let (file, item) = impl_file_entry(self.outline, block);
                    let parsed = Arc::new(sig::impl_header(
                        &file.source,
                        item.start,
                        item.body.unwrap_or(item.end),
                    )?);
                    self.store.headers.insert(block, Arc::clone(&parsed));
                    parsed
                };
                Ok(slot.insert(found))
            }
        }
    }

    /// The parsed header of impl `block`.
    pub fn impl_header(&mut self, block: ImplRef) -> Result<Arc<ImplHeader>> {
        if let Some(found) = self.headers_local.get(&block) {
            return Ok(Arc::clone(found));
        }
        Ok(Arc::clone(self.ensure_impl_header(block)?))
    }

    /// Borrows the parsed header of impl `block` from the per-context cache without cloning its `Arc`.
    pub fn impl_header_ref(&mut self, block: ImplRef) -> Result<&ImplHeader> {
        Ok(self.ensure_impl_header(block)?)
    }

    /// The outline file that holds the header of impl `block`.
    pub fn impl_file(&self, block: ImplRef) -> &'a File {
        impl_file_entry(self.outline, block).0
    }

    /// The module around impl `block`.
    pub fn impl_module(&mut self, block: ImplRef) -> Result<ModRef> {
        match block {
            ImplRef::Module { krate, index } => Ok(ModRef {
                krate,
                module: self.outline.crates[krate as usize].impls[index as usize].module as u32,
            }),
            ImplRef::Body(item) => self.module_of(item),
        }
    }

    /// The functions, constants and types of impl `block`, its own and those its macro calls expand to.
    pub fn members(&self, block: ImplRef) -> Vec<ItemRef> {
        let items = match block {
            ImplRef::Module { krate, index } => self.block_parts(
                krate,
                &self.outline.crates[krate as usize].impls[index as usize],
            ),
            ImplRef::Body(item) => self.children(item),
        };
        items
            .into_iter()
            .filter(|&member| {
                matches!(
                    self.item(member).1.kind,
                    ItemKind::Fn | ItemKind::Const | ItemKind::Type
                )
            })
            .collect()
    }

    /// The member of impl `block` named `name`.
    pub fn member(&self, block: ImplRef, name: &str) -> Option<ItemRef> {
        if let Some(members) = self.impl_members_local.borrow().get(&block) {
            return members.get(name).copied();
        }
        let members = self.impl_members.get_or_insert_with(block, || {
            let mut map = HashMap::default();
            for member in self.members(block) {
                map.entry(self.plain_name(member)).or_insert(member);
            }
            Arc::new(map)
        });
        let found = members.get(name).copied();
        self.impl_members_local.borrow_mut().insert(block, members);
        found
    }

    /// The impl block whose member `item` is.
    pub fn owner_impl(&mut self, item: ItemRef) -> Option<ImplRef> {
        if item.declaring_body().is_some() {
            return self
                .parent_item(item)
                .filter(|parent| self.item(*parent).1.kind == ItemKind::Impl)
                .map(ImplRef::Body);
        }
        let file = self.file(item);
        let key = match file.items.items[item.item as usize].parent {
            Some(parent) if file.items.items[parent as usize].kind == ItemKind::Impl => {
                (item.file, Some(parent))
            }
            Some(_) => return None,
            None if !file.expanded => return None,
            None => (item.file, None),
        };
        self.owners_of(item.krate)
            .get(&key)
            .map(|&index| ImplRef::Module {
                krate: item.krate,
                index,
            })
    }

    fn owners_of(&mut self, krate: u32) -> &HashMap<(u32, Option<u32>), u32> {
        let tree = &self.outline.crates[krate as usize];
        let store = &self.store.lookups.owners;
        self.crate_owners[krate as usize].get_or_insert_with(|| {
            store.get_or_insert_with(krate, || {
                let mut map = HashMap::default();
                for (index, entry) in tree.impls.iter().enumerate() {
                    map.insert((entry.file as u32, Some(entry.at)), index as u32);
                    map.extend(
                        entry
                            .parts
                            .iter()
                            .map(|&part| ((part as u32, None), index as u32)),
                    );
                }
                Arc::new(map)
            })
        })
    }

    /// Whether code in module `from` may name the inherent member `item`, as the member's visibility allows.
    pub fn member_visible(&mut self, item: ItemRef, from: ModRef) -> Result<bool> {
        let vis = self.item(item).1.vis;
        if vis == crate::outline::item::Vis::Pub {
            return Ok(true);
        }
        let home = self.module_of(item)?;
        Ok(crate::names::resolve::visible(
            &self.paths(from.krate),
            vis,
            home,
            from,
        ))
    }

    /// The inherent impls of types with head `head`, asking for every module of the crates that may hold one.
    pub fn inherent(&mut self, head: Head) -> Vec<ImplRef> {
        let standard = self.primitive_holders();
        let one;
        let holders: &[u32] = match head {
            Head::Adt(item) | Head::Dyn(item) if !standard.contains(&item.krate) => {
                one = [item.krate];
                &one
            }
            _ => &standard,
        };
        let mut found = Vec::new();
        self.any_holder(holders, Some(head), None, |index| {
            found.extend(index.inherent.get(&head).into_iter().flatten().copied());
            false
        });
        found
    }

    fn extra_body_holders(
        &self,
        crates: &[u32],
        head: Option<Head>,
        of: Option<ItemRef>,
    ) -> [Option<ItemRef>; 2] {
        let local_self = match head {
            Some(Head::Adt(item)) => item.declaring_body(),
            _ => None,
        };
        let local_trait = of.and_then(ItemRef::declaring_body);
        if local_self.is_none() && local_trait.is_none() {
            return [None, None];
        }
        let mut out = [None, None];
        let mut len = 0;
        for root in [local_self, local_trait].into_iter().flatten() {
            let in_crate_index = crates.contains(&root.krate)
                && self.outline.crates[root.krate as usize]
                    .impl_bodies
                    .iter()
                    .any(|&(file, item)| file as u32 == root.file && item == root.item);
            if !in_crate_index && !out[..len].contains(&Some(root)) {
                out[len] = Some(root);
                len += 1;
            }
        }
        out
    }

    fn any_holder(
        &mut self,
        crates: &[u32],
        head: Option<Head>,
        of: Option<ItemRef>,
        mut visit: impl FnMut(&Index) -> bool,
    ) -> bool {
        for &krate in crates {
            if self.with_crate_index(krate, &mut visit) {
                return true;
            }
        }
        for root in self
            .extra_body_holders(crates, head, of)
            .into_iter()
            .flatten()
        {
            let index = self.index_of(Holder::Body(root));
            if visit(&index) {
                return true;
            }
        }
        false
    }

    fn with_crate_index(&mut self, krate: u32, visit: &mut impl FnMut(&Index) -> bool) -> bool {
        let k = krate as usize;
        if let Some(index) = self.crate_heads[k].as_deref() {
            if !self.crate_heads_replayed[k] {
                index.asked.replay(&self.asks);
                self.crate_heads_replayed[k] = true;
            } else if !index.asked.is_empty() {
                self.asks.bump_taint();
            }
            return visit(index);
        }
        let index = self.index_of(Holder::Crate(krate));
        let hit = visit(&index);
        self.crate_heads[k] = Some(index);
        hit
    }

    /// Whether any impl of trait `of` in crates `holders` may match a self type with head `head`.
    pub fn has_trait_impl_for_head(&mut self, of: ItemRef, head: Head, holders: &[u32]) -> bool {
        self.has_trait_impl_for_head_inner(of, head, holders, true)
    }

    /// Whether any non-blanket impl of trait `of` in crates `holders` targets self head `head`.
    pub fn has_specific_trait_impl_for_head(
        &mut self,
        of: ItemRef,
        head: Head,
        holders: &[u32],
    ) -> bool {
        self.has_trait_impl_for_head_inner(of, head, holders, false)
    }

    fn has_trait_impl_for_head_inner(
        &mut self,
        of: ItemRef,
        head: Head,
        holders: &[u32],
        include_blanket: bool,
    ) -> bool {
        self.any_holder(holders, Some(head), Some(of), |index| {
            index.traits.get(&of).is_some_and(|bucket| {
                (include_blanket && !bucket.blanket.is_empty())
                    || bucket
                        .by_head
                        .get(&head)
                        .is_some_and(|matching| !matching.is_empty())
            })
        })
    }

    /// The impls of trait `of` in crates `holders` whose self type may have head `head`.
    pub fn trait_impls(
        &mut self,
        of: ItemRef,
        head: Option<Head>,
        holders: &[u32],
    ) -> Vec<ImplRef> {
        let mut found = Vec::new();
        self.any_holder(holders, head, Some(of), |index| {
            if let Some(bucket) = index.traits.get(&of) {
                match head {
                    None => found.extend_from_slice(&bucket.all),
                    Some(head) => {
                        if let Some(matching) = bucket.by_head.get(&head) {
                            found.extend_from_slice(matching);
                        }
                        found.extend_from_slice(&bucket.blanket);
                    }
                }
            }
            false
        });
        found
    }

    /// Whether crates `holders` write a negative impl of trait `of` for a self type with head `head`.
    pub fn negative_impl(&mut self, of: ItemRef, head: Head, holders: &[u32]) -> bool {
        self.any_holder(holders, Some(head), Some(of), |index| {
            index
                .negative
                .get(&of)
                .is_some_and(|heads| heads.contains(&head))
        })
    }

    /// The trait that impl `block` implements, or none for an inherent impl.
    pub fn impl_trait(&mut self, block: ImplRef) -> Option<ItemRef> {
        if let Some(known) = self.store.lookups.impl_traits.get(&block) {
            return known;
        }
        let header = self.impl_header(block).ok()?;
        let src = &self.impl_file(block).source;
        let home = self.impl_module(block).ok()?;
        self.impl_trait_with(block, &header, src, home)
    }

    fn impl_trait_with(
        &mut self,
        block: ImplRef,
        header: &ImplHeader,
        src: &Source,
        home: ModRef,
    ) -> Option<ItemRef> {
        if let Some(known) = self.store.lookups.impl_traits.get(&block) {
            return known;
        }
        let before = self.quiet_mark();
        let found = header.trait_ref.and_then(|trait_ref| {
            let syntax::Ty::Path(path) = *header.body.ty(trait_ref) else {
                return None;
            };
            let (segments, absolute, dollar) =
                segments(&self.names.symbols, src, &header.body, path);
            match self.resolve_in(
                block.item(),
                &NamePath {
                    segments: &segments,
                    absolute,
                    dollar,
                },
                home,
            )? {
                Def::Item(item) if self.item(item).1.kind == ItemKind::Trait => Some(item),
                _ => None,
            }
        });
        if self.quiet_mark() == before {
            self.store.lookups.impl_traits.insert(block, found);
        }
        found
    }

    /// The item that `path`, written at the item `at`, names from `home`, first among the items that the body declaring `at` holds.
    fn resolve_in(&self, at: Option<ItemRef>, path: &NamePath<'_>, home: ModRef) -> Option<Def> {
        let paths = self.paths(home.krate);
        let Some((root, from)) = at.and_then(ItemRef::outer) else {
            return paths.resolve(path, home);
        };
        match self.local_path(home, Some((root, Some(from))), path, Ns::Types) {
            Some(BodyPath::Def(def)) => Some(def),
            Some(BodyPath::Variant(_)) => None,
            Some(BodyPath::Outside(skip)) => paths.resolve(&outside(path, skip), home),
            None => self
                .body_use_path((root, home), path, Some(from))
                .or_else(|| paths.resolve(path, home)),
        }
    }

    /// The standard crates, which hold the impls and lang items of primitive types.
    pub fn primitive_holders(&self) -> Arc<[u32]> {
        if let Some(found) = super::read_lock(&self.store.lookups.state.primitives).as_ref() {
            return Arc::clone(found);
        }
        let roots = self.outline.roots.iter().enumerate();
        let found: Arc<[u32]> = roots
            .filter(|(_, root)| {
                root.role == crate::outline::Role::Lib
                    && PRIMITIVE_HOLDERS.contains(&root.name.as_str())
            })
            .map(|(krate, _)| krate as u32)
            .collect();
        *super::write_lock(&self.store.lookups.state.primitives) = Some(Arc::clone(&found));
        found
    }

    fn with_fresh_replay<R>(&mut self, build: impl FnOnce(&mut Self) -> R) -> (R, Asked) {
        let saved = std::mem::replace(
            &mut self.crate_heads_replayed,
            vec![false; self.outline.crates.len()],
        );
        let mark = self.asks.mark();
        let out = build(self);
        let asked = Asked::from_requests(self.asks.since(mark));
        for (dst, prev) in self.crate_heads_replayed.iter_mut().zip(saved) {
            *dst |= prev;
        }
        (out, asked)
    }

    /// Indexes only the module-level impls of `krate` (excluding `impl_bodies`) for sysroot caching.
    pub(crate) fn index_module_impls(&mut self, krate: u32) -> Index {
        let (mut index, asked) = self.with_fresh_replay(|this| {
            let count = this.outline.crates[krate as usize].impls.len() as u32;
            let blocks: Vec<ImplRef> = (0..count)
                .map(|index| ImplRef::Module { krate, index })
                .collect();
            let mut index = Index::default();
            this.index_blocks(&mut index, blocks);
            index
        });
        index.settled_crate = !self.outline.crates[krate as usize].unsettled();
        index.asked = asked;
        index
    }

    fn index_of(&mut self, holder: Holder) -> Arc<Index> {
        if let Some(found) = self.store.lookups.heads.get(&holder) {
            if let Holder::Crate(k) = holder {
                if !self.crate_heads_replayed[k as usize] {
                    found.asked.replay(&self.asks);
                    self.crate_heads_replayed[k as usize] = true;
                } else if !found.asked.is_empty() {
                    self.asks.bump_taint();
                }
            } else {
                found.asked.replay(&self.asks);
            }
            return found;
        }
        if let Holder::Body(root) = holder
            && self.expansion(root).is_none()
        {
            return Arc::default();
        }
        if let Holder::Crate(krate) = holder
            && let Some(cached) = self
                .store
                .lookups
                .sysroot_heads
                .as_ref()
                .and_then(|map| map.get(&krate))
                .cloned()
        {
            let tree = &self.outline.crates[krate as usize];
            if tree.impl_bodies.is_empty() && !tree.unsettled() {
                return cached;
            }
            let (mut index, asked) = self.with_fresh_replay(|this| {
                if this.outline.crates[krate as usize].unsettled() {
                    this.asks.crates.borrow_mut().push(krate);
                }
                let mut index = (*cached).clone();
                let blocks = this.crate_body_impls(krate);
                this.index_blocks(&mut index, blocks);
                index
            });
            index.settled_crate = !self.outline.crates[krate as usize].unsettled();
            index.asked = asked;
            let index = Arc::new(index);
            self.store.lookups.heads.insert(holder, Arc::clone(&index));
            return index;
        }
        let settled_crate = match holder {
            Holder::Crate(krate) => !self.outline.crates[krate as usize].unsettled(),
            Holder::Body(_) => false,
        };
        let (mut index, asked) = self.with_fresh_replay(|this| {
            if let Holder::Crate(krate) = holder
                && this.outline.crates[krate as usize].unsettled()
            {
                this.asks.crates.borrow_mut().push(krate);
            }
            let blocks = match holder {
                Holder::Crate(krate) => this.crate_impls(krate),
                Holder::Body(root) => this.body_impls(root),
            };
            let mut index = Index::default();
            this.index_blocks(&mut index, blocks);
            index
        });
        index.settled_crate = settled_crate;
        index.asked = asked;
        let index = Arc::new(index);
        self.store.lookups.heads.insert(holder, Arc::clone(&index));
        index
    }

    fn index_blocks(&mut self, index: &mut Index, blocks: Vec<ImplRef>) {
        let workers = crate::jobs().min(blocks.len() / 32);
        if !self.in_parallel && workers > 1 {
            let chunk_size = blocks.len().div_ceil(workers);
            let chunks: Vec<(Context<'_>, Vec<ImplRef>)> = blocks
                .chunks(chunk_size)
                .map(|chunk| (self.fork(), chunk.to_vec()))
                .collect();
            let results = crate::sync::parallel_map(chunks, |(mut worker, chunk)| {
                let mut partial = Index::default();
                for block in chunk {
                    worker.index_block(&mut partial, block);
                }
                (partial, worker.asks)
            });
            for (partial, asks) in results {
                index.merge_from(partial);
                self.asks.extend_parts(asks.into_parts());
            }
        } else {
            for block in blocks {
                self.index_block(index, block);
            }
        }
    }

    /// The impls of crate `krate`: those of its modules, then those in the bodies of its module-level `const _` items and functions.
    fn crate_impls(&mut self, krate: u32) -> Vec<ImplRef> {
        let count = self.outline.crates[krate as usize].impls.len() as u32;
        let mut blocks: Vec<ImplRef> = (0..count)
            .map(|index| ImplRef::Module { krate, index })
            .collect();
        blocks.extend(self.crate_body_impls(krate));
        blocks
    }

    fn crate_body_impls(&mut self, krate: u32) -> Vec<ImplRef> {
        let bodies = self.outline.crates[krate as usize].impl_bodies.clone();
        let mut blocks = Vec::new();
        for (file, item) in bodies {
            let root = ItemRef {
                krate,
                file: file as u32,
                item,
            };
            if self.file(root).body_read(krate, item) || self.parse_body(root).is_ok() {
                blocks.extend(self.body_impls(root));
            }
        }
        blocks
    }

    fn body_impls(&self, root: ItemRef) -> Vec<ImplRef> {
        let Some(found) = self.expansion(root) else {
            return Vec::new();
        };
        let items = &found.file.items.items;
        let blocks = items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.kind == ItemKind::Impl);
        blocks
            .map(|(at, _)| ImplRef::Body(root.inner(at as u32)))
            .collect()
    }
}
