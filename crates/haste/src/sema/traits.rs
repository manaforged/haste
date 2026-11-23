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
    pub(super) crates: Arc<[u32]>,
    hot: [Option<ItemRef>; 10],
}

impl Langs {
    #[inline]
    fn get(&self, name: &str) -> Option<ItemRef> {
        let idx = match name {
            "drop" => 0,
            "copy" => 1,
            "sized" => 2,
            "fn_once" => 3,
            "fn_mut" => 4,
            "fn" => 5,
            "drop_glue" => 6,
            "future_trait" => 7,
            "owned_box" => 8,
            "manually_drop" => 9,
            _ => return self.table.get(name).copied(),
        };
        self.hot[idx]
    }
}

/// Which allocator shim function (`__rust_alloc`, `__rust_alloc_zeroed`, `__rust_dealloc`, `__rust_realloc`) dispatches to `#[global_allocator]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalAllocOp {
    Alloc,
    AllocZeroed,
    Dealloc,
    Realloc,
}

impl GlobalAllocOp {
    /// Classifies a foreign allocator shim symbol name into its `GlobalAlloc` operation.
    pub fn from_symbol(name: &str) -> Option<Self> {
        match name {
            "__rust_alloc" => Some(Self::Alloc),
            "__rust_alloc_zeroed" => Some(Self::AllocZeroed),
            "__rust_dealloc" => Some(Self::Dealloc),
            "__rust_realloc" => Some(Self::Realloc),
            _ => None,
        }
    }

    /// The `core::alloc::GlobalAlloc` trait method name for this operation.
    pub const fn method_name(self) -> &'static str {
        match self {
            Self::Alloc => "alloc",
            Self::AllocZeroed => "alloc_zeroed",
            Self::Dealloc => "dealloc",
            Self::Realloc => "realloc",
        }
    }
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
        if let Some(members) = self.trait_members_local.borrow().get(&of) {
            return members.get(name).copied();
        }
        let members = self.trait_members.get_or_insert_with(of, || {
            let mut map = HashMap::default();
            for member in self.children(of) {
                map.entry(self.plain_name(member)).or_insert(member);
            }
            Arc::new(map)
        });
        let found = members.get(name).copied();
        self.trait_members_local.borrow_mut().insert(of, members);
        found
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

    /// The traits in scope in module `home`, returned as a shared slice from a per-context cache.
    pub fn traits_in_scope(&self, home: ModRef) -> Arc<[ItemRef]> {
        if let Some(known) = self.scope_traits_local.borrow().get(&home) {
            return Arc::clone(known);
        }
        if let Some(known) = self.store.lookups.scope_traits.get(&home) {
            self.scope_traits_local.borrow_mut().insert(home, Arc::clone(&known));
            return known;
        }
        let before = self.quiet_mark();
        let found: Arc<[ItemRef]> = self.paths(home.krate).traits_in(home).into();
        if self.quiet_mark() == before {
            self.store.lookups.scope_traits.insert(home, Arc::clone(&found));
            self.scope_traits_local.borrow_mut().insert(home, Arc::clone(&found));
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

    /// The `#[global_allocator]` static item in the crate graph, when one is defined.
    pub fn global_allocator(&self) -> Option<ItemRef> {
        if let Some(cached) = *read_lock(&self.store.lookups.state.global_allocator) {
            return cached;
        }
        let wanted = mark("global_allocator");
        let mut found = None;
        for (krate, tree) in self.outline.crates.iter().enumerate() {
            if self.outline.is_sysroot(krate as u32) {
                continue;
            }
            for (file, entry) in tree.files.iter().enumerate() {
                for &(at, _) in entry.items.marks.iter().filter(|(_, bits)| bits & wanted != 0) {
                    let meta = &entry.items.items[at as usize];
                    if meta.kind != ItemKind::Static || !tree.env.enabled(&meta.attrs) {
                        continue;
                    }
                    let item = ItemRef { krate: krate as u32, file: file as u32, item: at };
                    if self.has_attr(item, "global_allocator") {
                        found = Some(item);
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        *write_lock(&self.store.lookups.state.global_allocator) = Some(found);
        found
    }

    /// The `(op, static_item)` pair when `item` is one of the four foreign `__rust_*alloc*` shims and a `#[global_allocator]` is installed.
    pub fn global_alloc_shim(&self, item: ItemRef) -> Option<(GlobalAllocOp, ItemRef)> {
        if !self.is_extern_fn(item) {
            return None;
        }
        let op = GlobalAllocOp::from_symbol(self.symbol_name(item)?.as_str())?;
        Some((op, self.global_allocator()?))
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
        let langs = match &self.langs {
            Some(langs) => langs,
            None => {
                let loaded = if let Some(langs) = read_lock(&self.store.lookups.state.langs).as_ref() {
                    Arc::clone(langs)
                } else {
                    let built = Arc::new(self.lang_table());
                    let mut guard = write_lock(&self.store.lookups.state.langs);
                    Arc::clone(guard.get_or_insert(built))
                };
                self.langs.insert(loaded)
            }
        };
        if !langs.partial.is_empty() {
            self.asks.crates.borrow_mut().extend(langs.partial.iter().copied());
        }
        langs.get(name)
    }

    fn lang_table(&self) -> Langs {
        let (mut table, mut partial, crates) = (HashMap::default(), Vec::new(), self.primitive_holders());
        let wanted = mark("lang") | mark("panic_handler");
        for &krate in crates.iter() {
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
                    } else if std && entry.items.items[at as usize].kind == ItemKind::Fn && self.has_attr(item, "panic_handler") {
                        table.insert(PANIC_HANDLER.to_owned(), item);
                    }
                }
            }
        }
        let hot = [
            table.get("drop").copied(),
            table.get("copy").copied(),
            table.get("sized").copied(),
            table.get("fn_once").copied(),
            table.get("fn_mut").copied(),
            table.get("fn").copied(),
            table.get("drop_glue").copied(),
            table.get("future_trait").copied(),
            table.get("owned_box").copied(),
            table.get("manually_drop").copied(),
        ];
        Langs { table, partial, crates, hot }
    }
}
