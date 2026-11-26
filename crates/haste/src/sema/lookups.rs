use std::sync::atomic::{AtomicBool, Ordering};
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

#[derive(Clone)]
pub(super) enum ActiveAttrs {
    None,
    One(u32),
    Two([u32; 2]),
    Many(Arc<[u32]>),
}

impl ActiveAttrs {
    pub(super) fn from_vec(attrs: Vec<u32>) -> Self {
        match attrs.as_slice() {
            [] => Self::None,
            &[a] => Self::One(a),
            &[a, b] => Self::Two([a, b]),
            _ => Self::Many(attrs.into()),
        }
    }

    pub(super) fn as_slice(&self) -> &[u32] {
        match self {
            Self::None => &[],
            Self::One(a) => std::slice::from_ref(a),
            Self::Two(pair) => pair,
            Self::Many(many) => many,
        }
    }
}

pub(super) struct LangMap {
    pub(super) map: HashMap<ItemRef, Box<str>>,
    bloom: [u64; 64],
    crates_lo: u64,
    crates_hi: u64,
    any_high: bool,
}

impl LangMap {
    #[inline]
    fn bloom_bits(item: ItemRef) -> (usize, u64, usize, u64) {
        let h1 = item.file.wrapping_mul(0x9E37_79B1) ^ item.item.wrapping_mul(0x85EB_CA77);
        let h2 = item.file.wrapping_mul(0xC2B2_AE35) ^ item.item.wrapping_mul(0x27D4_EB2F);
        let b1 = ((h1 ^ (h1 >> 16)) as usize) & 4095;
        let b2 = ((h2 ^ (h2 >> 16)) as usize) & 4095;
        (b1 >> 6, 1u64 << (b1 & 63), b2 >> 6, 1u64 << (b2 & 63))
    }

    pub(super) fn from_map(map: HashMap<ItemRef, Box<str>>) -> Self {
        let (mut crates_lo, mut crates_hi, mut any_high) = (0u64, 0u64, false);
        let mut bloom = [0u64; 64];
        for &item in map.keys() {
            if item.krate < 64 {
                crates_lo |= 1u64 << item.krate;
            } else if item.krate < 128 {
                crates_hi |= 1u64 << (item.krate - 64);
            } else {
                any_high = true;
            }
            let (w1, m1, w2, m2) = Self::bloom_bits(item);
            bloom[w1] |= m1;
            bloom[w2] |= m2;
        }
        Self { map, bloom, crates_lo, crates_hi, any_high }
    }

    #[inline]
    pub(super) fn may_contain(&self, item: ItemRef) -> bool {
        let has_krate = if item.krate < 64 {
            self.crates_lo & (1u64 << item.krate) != 0
        } else if item.krate < 128 {
            self.crates_hi & (1u64 << (item.krate - 64)) != 0
        } else {
            self.any_high
        };
        if !has_krate {
            return false;
        }
        let (w1, m1, w2, m2) = Self::bloom_bits(item);
        (self.bloom[w1] & m1) != 0 && (self.bloom[w2] & m2) != 0
    }
}

#[derive(Default)]
pub(super) struct OutlineState {
    pub(super) langs: RwLock<Option<Arc<traits::Langs>>>,
    pub(super) global_allocator: RwLock<Option<Option<ItemRef>>>,
    pub(super) loaded: RwLock<Vec<crate::outline::tree::CrateExtent>>,
    pub(super) primitives: RwLock<Option<Arc<[u32]>>>,
}

#[derive(Clone, Default)]
pub(super) struct Lookups {
    pub(super) modules: Table<u32, Arc<HashMap<(u32, u32), u32>>>,
    pub(super) heads: Table<impls::Holder, Arc<impls::Index>>,
    pub(super) owners: Table<u32, Arc<HashMap<(u32, Option<u32>), u32>>>,
    pub(super) trait_defs: Table<ItemRef, Arc<traits::TraitGenerics>>,
    pub(super) discriminants: Table<ItemRef, Arc<[i128]>>,
    pub(super) reprs: Table<ItemRef, rustc_abi::ReprOptions>,
    pub(super) attrs: Table<ItemRef, ActiveAttrs>,
    pub(super) state: Arc<OutlineState>,
    pub(super) drops: Table<ty::Ty, bool>,
    pub(super) own_drops: Table<ty::Ty, bool>,
    pub(super) unpins: Table<ty::Ty, (bool, u32)>,
    pub(super) freezes: Table<ty::Ty, (bool, u32)>,
    pub(super) glues: Table<ty::Ty, Option<infer::Res>>,
    pub(super) fields: Table<ty::Ty, infer::Fields>,
    pub(super) forms: Table<(infer::Form, usize), Option<infer::Fields>>,
    pub(super) reveals: table::Flight<reach::Instance, ty::Ty>,
    pub(super) provisional_rets: Table<reach::Instance, ty::Ty>,
    pub(super) futures: Table<ty::Ty, ty::Ty>,
    pub(super) scope_traits: Table<ModRef, Arc<[ItemRef]>>,
    pub(super) type_defs: Table<(TypeScope, SegKey), crate::names::Def>,
    pub(super) impl_traits: Table<impls::ImplRef, Option<ItemRef>>,
    pub(super) selections: Table<(ItemRef, ty::Ty, super::substs::Substs), Option<(impls::ImplRef, super::substs::Substs)>>,
    pub(super) projections: Table<(ItemRef, ty::Ty, super::substs::Substs, crate::names::Sym, super::substs::Substs), Option<ty::Ty>>,
    pub(super) trait_calls: Table<(ItemRef, super::substs::Substs), infer::Res>,
    pub(super) assoc_types: Table<(ItemRef, ty::Ty, super::substs::Substs, ItemRef, super::substs::Substs), ty::Ty>,
    pub(super) summaries: table::Flight<ItemRef, summary::SummaryKind>,
    pub(super) verdicts: Table<ItemRef, ()>,
    pub(super) fresh_summaries: Table<ItemRef, (bool, u64, Box<[u8]>)>,
    pub(super) fresh_instances: Table<reach::Instance, (bool, u64, u64, Box<[u8]>)>,
    pub(super) has_new_sysroot_summaries: Arc<AtomicBool>,
    pub(super) has_new_sysroot_instances: Arc<AtomicBool>,
    pub(super) substrate: Option<Arc<crate::substrate::Substrate>>,
    pub(super) file_index: Arc<std::sync::OnceLock<Arc<summary::codec::FileIndex>>>,
    pub(super) lang_items: Arc<std::sync::OnceLock<Arc<LangMap>>>,
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
    pub(super) fn check(&mut self, outline: &Outline) {
        self.reconcile(outline, None);
    }

    fn reset_after_names(&mut self, outline: &Outline) {
        self.heads.retain(|holder, index| matches!(holder, impls::Holder::Crate(_)) && index.complete());
        self.selections.retain(|_, found| matches!(found, Some((impls::ImplRef::Module { .. }, _))));
        self.projections.retain(|_, found| found.is_some());
        self.forms.retain(|_, found| found.is_some());
        self.scope_traits.retain(|home, _| outline.is_fixed(home.krate));
        self.reveals.clear();
        self.provisional_rets.clear();
        self.type_defs.clear();
        self.trait_calls.clear();
        self.assoc_types.clear();
    }

    pub(super) fn reconcile(&mut self, outline: &Outline, answer: Option<Answer>) {
        if matches!(answer, Some(Answer::Names)) {
            self.reset_after_names(outline);
        }
        let loaded: Vec<crate::outline::tree::CrateExtent> = outline.crates.iter().map(crate::outline::tree::Crate::extent).collect();
        let mut guard = write_lock(&self.state.loaded);
        if *guard == loaded {
            return;
        }
        let same_len = guard.len() == loaded.len();
        let extent_changed = |pred: fn(&Outline, u32) -> bool| {
            !same_len
                || guard
                    .iter()
                    .zip(&loaded)
                    .enumerate()
                    .any(|(k, (a, b))| a != b && pred(outline, k as u32))
        };
        let reset_sysroot = extent_changed(Outline::is_sysroot);
        let reset_fixed = reset_sysroot || extent_changed(Outline::is_fixed);
        if same_len {
            self.modules.retain(|k, _| guard.get(*k as usize) == loaded.get(*k as usize));
            self.owners.retain(|k, _| guard.get(*k as usize) == loaded.get(*k as usize));
            self.scope_traits.retain(|home, _| guard.get(home.krate as usize) == loaded.get(home.krate as usize));
            self.heads.retain(|holder, _| match *holder {
                impls::Holder::Crate(k) => guard.get(k as usize) == loaded.get(k as usize),
                impls::Holder::Body(item) => guard.get(item.krate as usize) == loaded.get(item.krate as usize),
            });
            self.summaries.retain_keys(|item| guard.get(item.krate as usize) == loaded.get(item.krate as usize));
            if guard.iter().zip(&loaded).any(|(a, b)| a.consumed != b.consumed) {
                self.discriminants.retain(|item, _| guard.get(item.krate as usize) == loaded.get(item.krate as usize));
                self.glues.clear();
            }
        } else {
            self.modules.clear();
            self.owners.clear();
            self.scope_traits.clear();
            self.heads.clear();
            self.discriminants.clear();
            self.reprs.clear();
            self.attrs.clear();
            self.summaries = table::Flight::default();
            self.glues.clear();
        }
        if self.lang_items.get().is_some() && lang_files_changed(outline, &guard, &loaded, same_len) {
            self.lang_items = Arc::new(std::sync::OnceLock::new());
        }
        if reset_sysroot {
            *write_lock(&self.state.langs) = None;
            *write_lock(&self.state.primitives) = None;
        }
        if reset_fixed {
            self.file_index = Arc::new(std::sync::OnceLock::new());
            self.fresh_summaries.clear();
            self.fresh_instances.clear();
            self.has_new_sysroot_summaries.store(false, Ordering::Relaxed);
            self.has_new_sysroot_instances.store(false, Ordering::Relaxed);
        }
        *write_lock(&self.state.global_allocator) = None;
        self.trait_calls.clear();
        self.assoc_types.clear();
        self.type_defs.clear();
        *guard = loaded;
    }
}

fn lang_files_changed(
    outline: &Outline,
    prev: &[crate::outline::tree::CrateExtent],
    next: &[crate::outline::tree::CrateExtent],
    same_len: bool,
) -> bool {
    if !same_len {
        return true;
    }
    let lang_bit = crate::outline::item::mark("lang");
    prev.iter().zip(next).enumerate().any(|(k, (a, b))| {
        if a.files == b.files {
            return false;
        }
        b.files < a.files
            || outline.crates[k].files[a.files..]
                .iter()
                .any(|f| f.items.marks.iter().any(|&(_, bits)| bits & lang_bit != 0))
    })
}

impl Cache {
    /// Returns the reconciled file index for `outline`, building it lazily on first use.
    pub(crate) fn file_index(&self, outline: &Outline) -> &summary::codec::FileIndex {
        self.lookups.file_index.get_or_init(|| Arc::new(summary::codec::FileIndex::build(outline)))
    }

    /// Attaches a frozen sysroot substrate to serve pre-computed `PolySummary` entries.
    pub fn set_substrate(&mut self, substrate: Option<Arc<crate::substrate::Substrate>>) {
        self.lookups.substrate = substrate;
        self.lookups.file_index = Arc::new(std::sync::OnceLock::new());
    }

    /// Controls whether warm `.substrate` sysroot lookups eagerly parse bodies for macro expansion.
    pub fn set_warm_sysroot_bodies(&mut self, warm: bool) {
        self.lookups.skip_warm_sysroot_bodies = !warm;
    }

    /// Returns whether any new sysroot `PolySummary` or concrete sysroot instance was built in this session.
    pub fn has_new_sysroot_summaries(&self) -> bool {
        self.lookups.has_new_sysroot_summaries.load(Ordering::Relaxed)
            || self.lookups.has_new_sysroot_instances.load(Ordering::Relaxed)
    }

    /// Whether each type is structurally `Unpin`, with the depth its answer looked below it.
    pub(crate) fn unpins(&self) -> &Table<ty::Ty, (bool, u32)> {
        &self.lookups.unpins
    }

    /// Whether each type is structurally `Freeze`, with the depth its answer looked below it.
    pub(crate) fn freezes(&self) -> &Table<ty::Ty, (bool, u32)> {
        &self.lookups.freezes
    }

    /// Returns the attached frozen sysroot substrate, if any.
    pub fn substrate(&self) -> Option<&Arc<crate::substrate::Substrate>> {
        self.lookups.substrate.as_ref()
    }

    /// Drains newly built fixed `PolySummary` and concrete instance entries into `toolchain` and `deps` builders.
    pub fn export_sysroot_summaries(
        &self,
        _outline: &Outline,
        mut toolchain: Option<&mut crate::substrate::SubstrateBuilder>,
        mut deps: Option<&mut crate::substrate::SubstrateBuilder>,
    ) -> Result<()> {
        let mut err = None;
        self.lookups.fresh_summaries.retain(|_, (is_sysroot, key, bytes)| {
            if err.is_some() {
                return true;
            }
            let target = if *is_sysroot { toolchain.as_deref_mut() } else { deps.as_deref_mut() };
            if let Some(builder) = target
                && let Err(e) = builder.add_summary(*key, bytes)
            {
                err = Some(e);
                return true;
            }
            false
        });
        if let Some(e) = err {
            return Err(e);
        }
        self.lookups.has_new_sysroot_summaries.store(false, Ordering::Relaxed);
        summary::frozen::export_sysroot_instances(&self.lookups.fresh_instances, &self.lookups.has_new_sysroot_instances, toolchain, deps)
    }

    /// Drops the cached lookups, which no longer hold once the outline loads more files or resolves more names.
    pub fn forget_lookups(&mut self) {
        self.lookups = Lookups {
            substrate: self.lookups.substrate.clone(),
            file_index: Arc::clone(&self.lookups.file_index),
            skip_warm_sysroot_bodies: self.lookups.skip_warm_sysroot_bodies,
            ..Lookups::default()
        };
    }

    /// Drops the cached lookups that `answer` may make stale, keeping settled summaries and unchanged crate indexes.
    pub fn forget_after(&mut self, outline: &Outline, answer: Answer) {
        self.lookups.reconcile(outline, Some(answer));
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
        self.own_drops.retain(|ty, _| !naming.names(*ty));
        self.unpins.retain(|ty, _| !naming.names(*ty));
        self.freezes.retain(|ty, _| !naming.names(*ty));
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
        self.provisional_rets.retain(|instance, _| !naming.touches(instance));
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
        *write_lock(&self.state.global_allocator) = None;
        self.trait_calls.clear();
        self.assoc_types.clear();
        self.summaries = table::Flight::default();
        self.discriminants.retain(|item, _| !mine(*item));
        self.reprs.retain(|item, _| !mine(*item));
        self.attrs.retain(|item, _| !mine(*item));
        if self.lang_items.get().is_some() {
            self.lang_items = Arc::new(std::sync::OnceLock::new());
        }
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
