use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::adt::AdtDef;
use crate::body::sig::{self, Signature};
use crate::names::{ItemRef, ModRef};
use crate::outline::Outline;

use super::table::{self, Table};
use super::ty::{self, Types};
use super::{impls, infer, reach, summary, traits, write_lock};

pub(super) type TypeScope = (ModRef, Option<ItemRef>, Option<ItemRef>, bool, Option<u32>);

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct SegKey {
    inline: [u32; 4],
    len: u8,
    overflow: Option<Arc<[crate::names::Sym]>>,
}

impl SegKey {
    pub(super) fn from_slice(segments: &[crate::names::Sym]) -> Self {
        let mut inline = [0; 4];
        if segments.len() <= inline.len() {
            for (slot, sym) in inline.iter_mut().zip(segments) {
                *slot = sym.0;
            }
            return Self { inline, len: segments.len() as u8, overflow: None };
        }
        Self { inline, len: u8::MAX, overflow: Some(Arc::from(segments)) }
    }
}

#[derive(Default)]
pub(super) struct RunnerSlot(std::sync::Mutex<Option<Box<dyn super::evaluate::Evaluate>>>);

impl RunnerSlot {
    pub(super) fn take(&self) -> Option<Box<dyn super::evaluate::Evaluate>> { crate::sync::lock(&self.0).take() }
    pub(super) fn put(&self, runner: Box<dyn super::evaluate::Evaluate>) {
        let mut guard = crate::sync::lock(&self.0);
        if guard.is_none() {
            *guard = Some(runner);
        }
    }
    pub(super) fn clear(&self) { *crate::sync::lock(&self.0) = None; }
}

/// What Sema keeps between passes over a growing outline: interned types and parsed headers and bodies.
#[derive(Clone, Default)]
pub struct Cache {
    pub(super) types: Types,
    pub(super) signatures: Table<ItemRef, Arc<Signature>>,
    pub(super) bodies: Table<ItemRef, Arc<Body>>,
    pub(super) adts: Table<ItemRef, Arc<AdtDef>>,
    pub(super) layouts: Table<ty::Ty, Arc<super::layout::Layout>>,
    pub(super) tails: Table<ty::Ty, ty::Ty>,
    pub(super) headers: Table<impls::ImplRef, Arc<sig::ImplHeader>>,
    pub(super) finals: Table<reach::Instance, Arc<infer::Typed>>,
    pub(super) closures: Table<ty::Ty, Arc<infer::ClosureInfo>>,
    pub(super) evaluated: Table<reach::Instance, u128>,
    pub(super) runner: Arc<RunnerSlot>,
    pub(super) lookups: Lookups,
}

#[derive(Default)]
pub(super) struct OutlineState {
    pub(super) langs: RwLock<Option<traits::Langs>>,
    pub(super) loaded: RwLock<Vec<[usize; 5]>>,
    pub(super) primitives: RwLock<Option<Arc<[u32]>>>,
}

#[derive(Clone, Default)]
pub(super) struct Lookups {
    pub(super) modules: Table<u32, Arc<HashMap<(u32, u32), u32>>>,
    pub(super) heads: Table<impls::Holder, Arc<impls::Index>>,
    pub(super) owners: Table<u32, Arc<HashMap<(u32, Option<u32>), u32>>>,
    pub(super) trait_defs: Table<ItemRef, Arc<traits::TraitGenerics>>,
    pub(super) discriminants: Table<ItemRef, Arc<[i128]>>,
    pub(super) state: Arc<OutlineState>,
    pub(super) drops: Table<ty::Ty, bool>,
    pub(super) glues: Table<ty::Ty, Option<infer::Res>>,
    pub(super) fields: Table<ty::Ty, infer::Fields>,
    pub(super) forms: Table<(infer::Form, usize), Option<infer::Fields>>,
    pub(super) reveals: table::Flight<reach::Instance, ty::Ty>,
    pub(super) futures: Table<ty::Ty, ty::Ty>,
    pub(super) scope_traits: Table<ModRef, Arc<[ItemRef]>>,
    pub(super) type_defs: Table<(TypeScope, SegKey), crate::names::Def>,
    pub(super) impl_traits: Table<impls::ImplRef, Option<ItemRef>>,
    pub(super) selections: Table<(ItemRef, ty::Ty, super::substs::Substs), Option<(impls::ImplRef, super::substs::Substs)>>,
    pub(super) projections: Table<(ItemRef, ty::Ty, super::substs::Substs, crate::names::Sym, super::substs::Substs), Option<ty::Ty>>,
    pub(super) trait_calls: Table<(ItemRef, super::substs::Substs), infer::Res>,
    pub(super) assoc_types: Table<(ItemRef, ty::Ty, super::substs::Substs, ItemRef, super::substs::Substs), ty::Ty>,
    pub(super) summaries: table::Flight<ItemRef, summary::SummaryKind>,
    pub(super) has_new_sysroot_summaries: Arc<AtomicBool>,
    pub(super) has_new_sysroot_instances: Arc<AtomicBool>,
    pub(super) substrate: Option<Arc<crate::substrate::Substrate>>,
    pub(super) sysroot_index: Arc<std::sync::OnceLock<Arc<summary::codec::SysrootIndex>>>,
    pub(super) skip_warm_sysroot_bodies: bool,
}

/// What answering the asks of a pass added: nothing new to infer with, the output of procedural macros that function bodies call, or loaded files, resolved names and expanded items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    Nothing,
    Bodies,
    Names,
}

impl Lookups {
    fn after_pass(&mut self, outline: &Outline, answer: Answer) -> Lookups {
        if !matches!(answer, Answer::Names) {
            return std::mem::take(self);
        }
        let heads = self.heads.retained(|holder, index| matches!(holder, impls::Holder::Crate(_)) && index.complete());
        let selections = self.selections.retained(|_, found| matches!(found, Some((impls::ImplRef::Module { .. }, _))));
        let projections = self.projections.retained(|_, found| found.is_some());
        Lookups {
            modules: std::mem::take(&mut self.modules),
            heads,
            owners: std::mem::take(&mut self.owners),
            trait_defs: std::mem::take(&mut self.trait_defs),
            discriminants: std::mem::take(&mut self.discriminants),
            state: Arc::clone(&self.state),
            drops: std::mem::take(&mut self.drops),
            glues: std::mem::take(&mut self.glues),
            fields: std::mem::take(&mut self.fields),
            forms: self.forms.retained(|_, found| found.is_some()),
            futures: std::mem::take(&mut self.futures),
            scope_traits: self.scope_traits.retained(|home, _| outline.is_sysroot(home.krate)),
            impl_traits: std::mem::take(&mut self.impl_traits),
            selections,
            projections,
            summaries: std::mem::take(&mut self.summaries),
            has_new_sysroot_summaries: Arc::clone(&self.has_new_sysroot_summaries),
            has_new_sysroot_instances: Arc::clone(&self.has_new_sysroot_instances),
            substrate: self.substrate.clone(),
            sysroot_index: Arc::clone(&self.sysroot_index),
            skip_warm_sysroot_bodies: self.skip_warm_sysroot_bodies,
            ..Lookups::default()
        }
    }

    pub(super) fn check(&mut self, outline: &Outline) {
        let loaded: Vec<[usize; 5]> = outline.crates.iter().map(crate::outline::tree::Crate::extent).collect();
        let mut guard = write_lock(&self.state.loaded);
        if *guard != loaded {
            let same_len = guard.len() == loaded.len();
            let reset_sysroot = !same_len
                || guard
                    .iter()
                    .zip(&loaded)
                    .enumerate()
                    .any(|(k, (a, b))| a != b && outline.is_sysroot(k as u32));
            if same_len {
                self.modules.retain(|k, _| guard.get(*k as usize) == loaded.get(*k as usize));
                self.owners.retain(|k, _| guard.get(*k as usize) == loaded.get(*k as usize));
                self.scope_traits.retain(|home, _| guard.get(home.krate as usize) == loaded.get(home.krate as usize));
                self.heads.retain(|holder, _| match *holder {
                    impls::Holder::Crate(k) => guard.get(k as usize) == loaded.get(k as usize),
                    impls::Holder::Body(item) => guard.get(item.krate as usize) == loaded.get(item.krate as usize),
                });
                self.summaries.retain_keys(|item| guard.get(item.krate as usize) == loaded.get(item.krate as usize));
                if guard.iter().zip(&loaded).any(|(a, b)| a[3] != b[3]) {
                    self.discriminants.retain(|item, _| guard.get(item.krate as usize) == loaded.get(item.krate as usize));
                    self.glues.clear();
                }
            } else {
                self.modules.clear();
                self.owners.clear();
                self.scope_traits.clear();
                self.heads.clear();
                self.discriminants.clear();
                self.summaries = table::Flight::default();
                self.glues.clear();
            }
            if reset_sysroot {
                *write_lock(&self.state.langs) = None;
                *write_lock(&self.state.primitives) = None;
                self.sysroot_index = Arc::new(std::sync::OnceLock::new());
            }
            self.trait_calls.clear();
            self.assoc_types.clear();
            *guard = loaded;
        }
    }
}

impl Cache {
    /// Attaches a frozen sysroot substrate to serve pre-computed `PolySummary` entries.
    pub fn set_substrate(&mut self, substrate: Option<Arc<crate::substrate::Substrate>>) {
        self.lookups.substrate = substrate;
        self.lookups.sysroot_index = Arc::new(std::sync::OnceLock::new());
    }

    /// Controls whether warm `.substrate` sysroot lookups eagerly parse bodies for macro expansion.
    pub fn set_warm_sysroot_bodies(&mut self, warm: bool) {
        self.lookups.skip_warm_sysroot_bodies = !warm;
    }

    /// Returns whether any new sysroot `PolySummary` or concrete sysroot instance was built in this session.
    pub fn has_new_sysroot_summaries(&self) -> bool {
        self.lookups.has_new_sysroot_summaries.load(std::sync::atomic::Ordering::Relaxed)
            || self.lookups.has_new_sysroot_instances.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the attached frozen sysroot substrate, if any.
    pub fn substrate(&self) -> Option<&Arc<crate::substrate::Substrate>> {
        self.lookups.substrate.as_ref()
    }

    /// Encodes all settled sysroot `PolySummary` and concrete sysroot instance entries into `builder`.
    pub fn export_sysroot_summaries(&self, outline: &Outline, builder: &mut crate::substrate::SubstrateBuilder) -> Result<()> {
        let mut err = None;
        self.lookups.summaries.for_each_ready(|&item, kind| {
            if err.is_some() {
                return;
            }
            if !outline.is_sysroot(item.krate) {
                return;
            }
            if let summary::SummaryKind::Poly(summary) = kind
                && let Some(key) = crate::substrate::sysroot_item_key(outline, item)
                && let Ok(Some(bytes)) = summary::codec::encode_summary(outline, &self.types, summary, crate::substrate::sysroot_item_key)
                && let Err(e) = builder.add_summary(key, &bytes)
            {
                err = Some(e);
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        summary::frozen::export_sysroot_instances(outline, &self.types, &self.finals, builder)
    }

    /// Drops the cached lookups, which no longer hold once the outline loads more files or resolves more names.
    pub fn forget_lookups(&mut self) {
        self.lookups = Lookups {
            substrate: self.lookups.substrate.clone(),
            sysroot_index: Arc::clone(&self.lookups.sysroot_index),
            skip_warm_sysroot_bodies: self.lookups.skip_warm_sysroot_bodies,
            ..Lookups::default()
        };
    }

    /// Drops the cached lookups that `answer` may make stale, keeping settled summaries and unchanged crate indexes.
    pub fn forget_after(&mut self, outline: &Outline, answer: Answer) {
        self.lookups = self.lookups.after_pass(outline, answer);
    }

    /// Drops what the store keeps about the items that `edited` holds, and about the instances, types and lookups that name one of them.
    pub fn forget(&mut self, outline: &Outline, edited: Edited<'_>) {
        let mine = |item: ItemRef| edited.holds(item.krate, item.file);
        let block_kept = |block: &impls::ImplRef| match *block {
            impls::ImplRef::Module { krate, index } => {
                outline.crates[krate as usize].impls.get(index as usize).is_some_and(|entry| !edited.holds(krate, entry.file as u32) && !entry.parts.iter().any(|&part| edited.holds(krate, part as u32)))
            }
            impls::ImplRef::Body(item) => !mine(item),
        };
        self.signatures.retain(|item, _| !mine(*item));
        self.bodies.retain(|item, _| !mine(*item));
        self.adts.retain(|item, _| !mine(*item));
        self.headers.retain(|block, _| block_kept(block));
        let mut naming = Naming { types: &self.types, mine: &mine, memo: HashMap::default() };
        self.finals.retain(|instance, _| !naming.touches(instance));
        self.layouts.retain(|ty, _| !naming.names(*ty));
        self.tails.retain(|ty, _| !naming.names(*ty));
        self.closures.retain(|ty, _| !naming.names(*ty));
        self.evaluated.retain(|instance, _| !naming.touches(instance));
        self.runner.clear();
        self.lookups.forget_edited(edited, &mut naming, &mine, &block_kept);
    }

    /// Drops what the store keeps about what `edited` holds and returns the indexes of the instances in `list` that it touched.
    pub fn invalidate<'l>(&mut self, outline: &Outline, list: impl Iterator<Item = &'l reach::Instance>, edited: Edited<'_>) -> Vec<usize> {
        let touched = self.touched(list, edited);
        self.forget(outline, edited);
        touched
    }

    /// The indexes of the instances in `list` that are functions `edited` holds or take a type that names one of its items.
    pub fn touched<'l>(&self, list: impl Iterator<Item = &'l reach::Instance>, edited: Edited<'_>) -> Vec<usize> {
        let mine = |item: ItemRef| edited.holds(item.krate, item.file);
        let mut naming = Naming { types: &self.types, mine: &mine, memo: HashMap::default() };
        list.enumerate().filter(|(_, instance)| naming.touches(instance)).map(|(at, _)| at).collect()
    }
}

impl Lookups {
    fn forget_edited<F: Fn(ItemRef) -> bool>(&mut self, edited: Edited<'_>, naming: &mut Naming<'_, F>, mine: &F, block_kept: &dyn Fn(&impls::ImplRef) -> bool) {
        self.drops.retain(|ty, _| !naming.names(*ty));
        self.glues.retain(|ty, _| !naming.names(*ty));
        self.fields.retain(|ty, _| !naming.names(*ty));
        self.forms.retain(|(form, _), found| {
            let own = match *form {
                infer::Form::Adt(item) | infer::Form::Fn(item) => !mine(item),
                infer::Form::Impl(block) => block_kept(&block),
            };
            own && found.iter().flat_map(|types| types.iter().flatten()).all(|&ty| !naming.names(ty))
        });
        self.reveals.retain(|instance| !naming.touches(instance));
        self.futures.retain(|ty, _| !naming.names(*ty));
        self.impl_traits.retain(|block, _| block_kept(block));
        self.heads.retain(|holder, _| match *holder {
            impls::Holder::Crate(krate) => !edited.touches_crate(krate),
            impls::Holder::Body(item) => !mine(item),
        });
        self.modules.retain(|krate, _| !edited.touches_crate(*krate));
        self.owners.retain(|krate, _| !edited.touches_crate(*krate));
        self.scope_traits.retain(|home, _| !edited.touches_crate(home.krate));
        self.type_defs.retain(|key, _| !edited.touches_crate(key.0.0.krate));
        self.trait_defs.retain(|item, _| !mine(*item));
        self.selections.clear();
        self.projections.clear();
        self.trait_calls.clear();
        self.assoc_types.clear();
        self.summaries = table::Flight::default();
        self.discriminants.retain(|item, _| !mine(*item));
        let mut langs = write_lock(&self.state.langs);
        if langs.as_ref().is_some_and(|found| found.crates.iter().any(|&krate| edited.touches_crate(krate))) {
            *langs = None;
        }
    }
}

/// What an edit changed: function bodies in some `(crate, file)` slots, or whole crates read again.
#[derive(Clone, Copy, Debug)]
pub enum Edited<'e> {
    /// Function bodies in these `(crate, file)` slots.
    Files(&'e [(u32, u32)]),
    /// Every file of these crates.
    Crates(&'e [u32]),
}

impl Edited<'_> {
    fn holds(self, krate: u32, file: u32) -> bool {
        match self {
            Edited::Files(files) => files.contains(&(krate, file)),
            Edited::Crates(crates) => crates.contains(&krate),
        }
    }

    fn touches_crate(self, krate: u32) -> bool {
        match self {
            Edited::Files(files) => files.iter().any(|&(owner, _)| owner == krate),
            Edited::Crates(crates) => crates.contains(&krate),
        }
    }
}

pub(super) struct Naming<'t, F> {
    pub(super) types: &'t Types,
    pub(super) mine: &'t F,
    pub(super) memo: HashMap<ty::Ty, bool>,
}

impl<F: Fn(ItemRef) -> bool> Naming<'_, F> {
    pub(super) fn touches(&mut self, (item, substs): &reach::Instance) -> bool {
        (self.mine)(*item) || self.types.substs(*substs).iter().any(|ty| self.names(*ty))
    }

    fn names(&mut self, ty: ty::Ty) -> bool {
        if let Some(&known) = self.memo.get(&ty) {
            return known;
        }
        let kind = self.types.kind(ty);
        let own = match kind {
            ty::Kind::Adt(item, _) | ty::Kind::FnDef(item, _) | ty::Kind::Closure(item, ..) | ty::Kind::Dyn(item, ..) => (self.mine)(*item),
            ty::Kind::Proj(id) => {
                let def = self.types.proj(*id);
                (self.mine)(def.of) || (self.mine)(def.assoc)
            }
            _ => false,
        };
        let inner: Vec<ty::Ty> = kind.children(self.types).collect();
        let found = own || inner.into_iter().any(|inner| self.names(inner));
        self.memo.insert(ty, found);
        found
    }
}
