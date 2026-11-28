mod naming;

pub(super) use naming::Naming;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use rustc_hash::FxHashMap as HashMap;

use crate::body::Body;
use crate::body::adt::AdtDef;
use crate::body::sig::{self, Signature};
use crate::names::{ItemRef, ModRef};
use crate::outline::Outline;

use super::substs::Substs;
use super::table::{self, Table};
use super::ty::{self, Types};
use super::{impls, infer, reach, read_lock, summary, traits, write_lock};

pub(super) type TypeScope = (ModRef, Option<ItemRef>, Option<ItemRef>, u8, Option<u32>);

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct SegKey {
    inline: [u32; 4],
    len: u8,
    overflow: Option<Arc<[crate::names::Sym]>>,
}

impl SegKey {
    pub(super) fn from_slice(segments: &[crate::names::Sym]) -> Self {
        let mut inline = [0; 4];
        let (len, overflow) = if segments.len() <= inline.len() {
            for (slot, sym) in inline.iter_mut().zip(segments) {
                *slot = sym.0;
            }
            (segments.len() as u8, None)
        } else {
            (u8::MAX, Some(Arc::from(segments)))
        };
        Self {
            inline,
            len,
            overflow,
        }
    }

    pub(super) fn push_syms(&self, out: &mut Vec<crate::names::Sym>) {
        match &self.overflow {
            Some(many) => out.extend_from_slice(many),
            None => out.extend(
                self.inline[..usize::from(self.len)]
                    .iter()
                    .copied()
                    .map(crate::names::Sym),
            ),
        }
    }
}

#[derive(Default)]
pub(super) struct RunnerSlot(std::sync::Mutex<Option<Box<dyn super::evaluate::Evaluate>>>);

impl RunnerSlot {
    pub(super) fn take(&self) -> Option<Box<dyn super::evaluate::Evaluate>> {
        crate::sync::lock(&self.0).take()
    }
    pub(super) fn put(&self, runner: Box<dyn super::evaluate::Evaluate>) {
        let mut guard = crate::sync::lock(&self.0);
        if guard.is_none() {
            *guard = Some(runner);
        }
    }
    pub(super) fn clear(&self) {
        *crate::sync::lock(&self.0) = None;
    }
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
    bloom: [u64; 256],
    crates_lo: u64,
    crates_hi: u64,
    any_high: bool,
}

impl LangMap {
    #[inline]
    fn bloom_bits(item: ItemRef) -> (usize, u64, usize, u64) {
        let h1 = item.file.wrapping_mul(0x9E37_79B1) ^ item.item.wrapping_mul(0x85EB_CA77);
        let h2 = item.file.wrapping_mul(0xC2B2_AE35) ^ item.item.wrapping_mul(0x27D4_EB2F);
        let b1 = ((h1 ^ (h1 >> 16)) as usize) & 16383;
        let b2 = ((h2 ^ (h2 >> 16)) as usize) & 16383;
        (b1 >> 6, 1u64 << (b1 & 63), b2 >> 6, 1u64 << (b2 & 63))
    }

    pub(super) fn from_map(map: HashMap<ItemRef, Box<str>>) -> Self {
        let (mut crates_lo, mut crates_hi, mut any_high, mut bloom) =
            (0u64, 0u64, false, [0u64; 256]);
        for &item in map.keys() {
            match item.krate {
                k if k < 64 => crates_lo |= 1u64 << k,
                k if k < 128 => crates_hi |= 1u64 << (k - 64),
                _ => any_high = true,
            }
            let (w1, m1, w2, m2) = Self::bloom_bits(item);
            bloom[w1] |= m1;
            bloom[w2] |= m2;
        }
        Self {
            map,
            bloom,
            crates_lo,
            crates_hi,
            any_high,
        }
    }

    #[inline]
    pub(super) fn may_contain(&self, item: ItemRef) -> bool {
        let has_krate = match item.krate {
            k if k < 64 => self.crates_lo & (1u64 << k) != 0,
            k if k < 128 => self.crates_hi & (1u64 << (k - 64)) != 0,
            _ => self.any_high,
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
    pub(super) sysroot_heads: Option<Arc<HashMap<u32, Arc<impls::Index>>>>,
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
    pub(super) trait_autos: Table<ItemRef, bool>,
    pub(super) type_defs: Table<(TypeScope, SegKey), crate::names::Def>,
    pub(super) impl_traits: Table<impls::ImplRef, Option<ItemRef>>,
    pub(super) selections: Table<(ItemRef, ty::Ty, Substs), Option<(impls::ImplRef, Substs)>>,
    pub(super) auto_traits: Table<(ItemRef, ty::Ty), bool>,
    pub(super) checked_impls: table::Flight<impls::ImplRef, bool>,
    pub(super) projections:
        Table<(ItemRef, ty::Ty, Substs, crate::names::Sym, Substs), Option<ty::Ty>>,
    pub(super) trait_calls: Table<(ItemRef, Substs), infer::Res>,
    pub(super) assoc_types: Table<(ItemRef, ty::Ty, Substs, ItemRef, Substs), ty::Ty>,
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
    pub(super) reach_memo_enabled: bool,
    pub(super) reach_memo: Option<Arc<reach::memo::ReachMemoCache>>,
    pub(super) warm: Arc<summary::WarmState>,
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

    fn reset_after_names(&mut self, outline: &Outline, fixed_stable: bool) {
        let fixed_item = |i: ItemRef| fixed_stable && outline.is_fixed(i.krate);
        self.selections.retain(|_, found| {
            matches!(found, Some((impls::ImplRef::Module { .. }, _)))
                || matches!(found, Some((impls::ImplRef::Body(item), _)) if fixed_item(*item))
        });
        self.impl_traits.retain(|block, found| {
            found.is_some()
                && match *block {
                    impls::ImplRef::Module { .. } => true,
                    impls::ImplRef::Body(item) => fixed_item(item),
                }
        });
        self.projections.retain(|_, found| found.is_some());
        self.forms.retain(|_, found| found.is_some());
        self.scope_traits
            .retain(|home, _| outline.is_fixed(home.krate));
        self.reveals
            .retain(|&(item, substs)| fixed_item(item) && substs.is_empty());
        self.type_defs.retain(|&(sc, _), _| {
            fixed_stable
                && outline.is_fixed(sc.0.krate)
                && sc.1.is_none_or(|i| outline.is_fixed(i.krate))
                && sc.2.is_none_or(|i| outline.is_fixed(i.krate))
        });
        self.trait_calls
            .retain(|&(item, substs), _| fixed_item(item) && substs.is_empty());
        self.provisional_rets.clear();
        self.assoc_types.clear();
        self.auto_traits.clear();
        self.checked_impls.retain_entries(|_, &settled| settled);
    }

    pub(super) fn reconcile(&mut self, outline: &Outline, answer: Option<Answer>) {
        let after_names = matches!(answer, Some(Answer::Names));
        let after_bodies = matches!(answer, Some(Answer::Bodies));
        if !after_names && !after_bodies {
            let cur = read_lock(&self.state.loaded);
            if cur.len() == outline.crates.len()
                && cur
                    .iter()
                    .zip(&outline.crates)
                    .all(|(a, c)| *a == c.extent())
            {
                return;
            }
        }
        let loaded: Vec<_> = outline
            .crates
            .iter()
            .map(crate::outline::tree::Crate::extent)
            .collect();
        let state = Arc::clone(&self.state);
        let mut guard = write_lock(&state.loaded);
        if *guard == loaded {
            if after_names {
                self.reset_after_names(outline, true);
            } else if after_bodies {
                self.selections.retain(|_, found| found.is_some());
            }
            if after_names || after_bodies {
                retain_heads(&mut self.heads, outline, true, |_| true);
            }
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
        if after_names {
            self.reset_after_names(outline, !reset_fixed);
        } else if after_bodies {
            self.selections.retain(|_, found| found.is_some());
        }
        let lang_bit = crate::outline::item::mark("lang");
        if self.lang_items.get().is_some()
            && (!same_len
                || guard.iter().zip(&loaded).enumerate().any(|(k, (a, b))| {
                    a.files != b.files
                        && (b.files < a.files
                            || slice_has_mark(&outline.crates[k].files[a.files..], lang_bit))
                }))
        {
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
            self.has_new_sysroot_summaries
                .store(false, Ordering::Relaxed);
            self.has_new_sysroot_instances
                .store(false, Ordering::Relaxed);
        }
        if same_len {
            let same_krate = |k: u32| guard.get(k as usize) == loaded.get(k as usize);
            self.modules.retain(|k, _| same_krate(*k));
            self.owners.retain(|k, _| same_krate(*k));
            self.scope_traits.retain(|home, _| same_krate(home.krate));
            self.trait_autos.retain(|item, _| same_krate(item.krate));
            retain_heads(
                &mut self.heads,
                outline,
                after_names || after_bodies,
                same_krate,
            );
            self.summaries.retain_keys(|item| same_krate(item.krate));
            if guard
                .iter()
                .zip(&loaded)
                .any(|(a, b)| a.consumed != b.consumed)
            {
                self.discriminants.retain(|item, _| same_krate(item.krate));
                self.glues.clear();
            }
            if let Some(heads) = &mut self.sysroot_heads
                && heads.keys().any(|&k| !same_krate(k))
            {
                let map = Arc::make_mut(heads);
                map.retain(|&k, _| same_krate(k));
                if map.is_empty() {
                    self.sysroot_heads = None;
                }
            }
            self.checked_impls.retain_entries(|b, &s| {
                s && same_krate(match *b {
                    impls::ImplRef::Module { krate, .. } => krate,
                    impls::ImplRef::Body(item) => item.krate,
                })
            });
        } else {
            self.modules.clear();
            self.owners.clear();
            self.scope_traits.clear();
            self.trait_autos.clear();
            self.heads.clear();
            self.discriminants.clear();
            self.reprs.clear();
            self.attrs.clear();
            self.summaries.clear();
            self.glues.clear();
            self.checked_impls.clear();
            self.sysroot_heads = None;
        }
        *write_lock(&self.state.global_allocator) = None;
        self.trait_calls.clear();
        self.assoc_types.clear();
        self.type_defs.clear();
        self.auto_traits.clear();
        *guard = loaded;
    }
}

fn slice_has_mark(files: &[Arc<crate::outline::tree::File>], bit: u16) -> bool {
    files
        .iter()
        .any(|f| f.items.marks.iter().any(|&(_, bits)| bits & bit != 0))
}

fn retain_heads(
    heads: &mut Table<impls::Holder, Arc<impls::Index>>,
    outline: &Outline,
    after_names: bool,
    same_krate: impl Fn(u32) -> bool,
) {
    heads.retain(|holder, index| {
        let k = match *holder {
            impls::Holder::Crate(k) => k,
            impls::Holder::Body(item) => return !after_names && same_krate(item.krate),
        };
        same_krate(k)
            && (!after_names
                || index.complete()
                || (outline.is_fixed(k) && index.reusable_after_names() && {
                    Arc::make_mut(index).clear_asked();
                    true
                }))
    });
}

impl Cache {
    /// Returns the reconciled file index for `outline`, building it lazily on first use.
    pub(crate) fn file_index(&self, outline: &Outline) -> &summary::codec::FileIndex {
        self.lookups
            .file_index
            .get_or_init(|| Arc::new(summary::codec::FileIndex::build(outline)))
    }

    /// Attaches pre-built sysroot module impl indices if none are attached yet.
    pub(crate) fn set_sysroot_heads_if_unset(
        &mut self,
        heads: &Option<Arc<HashMap<u32, Arc<impls::Index>>>>,
    ) {
        if self.lookups.sysroot_heads.is_none() {
            self.lookups.sysroot_heads = heads.clone();
        }
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

    pub(crate) fn set_reach_memo_enabled(&mut self, enabled: bool) {
        self.lookups.reach_memo_enabled = enabled;
        if !enabled {
            self.lookups.reach_memo = None;
        }
    }

    /// Whether warm `.substrate` sysroot lookups eagerly parse bodies for lowering.
    pub fn warm_sysroot_bodies(&self) -> bool {
        !self.lookups.skip_warm_sysroot_bodies
    }

    /// Returns the one-shot warm check snapshot state attached to this cache.
    pub(crate) fn warm_state(&self) -> &summary::WarmState {
        &self.lookups.warm
    }

    /// Returns whether any new sysroot `PolySummary` or concrete sysroot instance was built in this session.
    pub fn has_new_sysroot_summaries(&self) -> bool {
        let l = &self.lookups;
        l.has_new_sysroot_summaries.load(Ordering::Relaxed)
            || l.has_new_sysroot_instances.load(Ordering::Relaxed)
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

    pub(crate) fn reach_memo(&self) -> Option<&Arc<reach::memo::ReachMemoCache>> {
        self.lookups.reach_memo.as_ref()
    }

    /// Drains newly built fixed `PolySummary` and concrete instance entries into `toolchain` and `deps` builders.
    pub fn export_sysroot_summaries(
        &self,
        _outline: &Outline,
        mut toolchain: Option<&mut crate::substrate::SubstrateBuilder>,
        mut deps: Option<&mut crate::substrate::SubstrateBuilder>,
    ) -> Result<()> {
        let mut err = None;
        self.lookups
            .fresh_summaries
            .retain(|_, (is_sysroot, key, bytes)| {
                if err.is_some() {
                    return true;
                }
                let target = if *is_sysroot {
                    toolchain.as_deref_mut()
                } else {
                    deps.as_deref_mut()
                };
                err = target.and_then(|b| b.add_summary(*key, bytes).err());
                err.is_some()
            });
        if let Some(e) = err {
            return Err(e);
        }
        self.lookups
            .has_new_sysroot_summaries
            .store(false, Ordering::Relaxed);
        summary::frozen::export_sysroot_instances(
            &self.lookups.fresh_instances,
            &self.lookups.has_new_sysroot_instances,
            toolchain,
            deps,
        )
    }

    /// Drops the cached lookups, which no longer hold once the outline loads more files or resolves more names.
    pub fn forget_lookups(&mut self) {
        self.lookups = Lookups {
            substrate: self.lookups.substrate.clone(),
            file_index: Arc::clone(&self.lookups.file_index),
            skip_warm_sysroot_bodies: self.lookups.skip_warm_sysroot_bodies,
            reach_memo_enabled: self.lookups.reach_memo_enabled,
            ..Lookups::default()
        };
    }

    /// Drops the cached lookups that `answer` may make stale, keeping settled summaries and unchanged crate indexes.
    pub fn forget_after(&mut self, outline: &Outline, answer: Answer) {
        self.lookups.reconcile(outline, Some(answer));
    }

    /// Drops what the store keeps about the items that `edited` holds, and about the instances, types and lookups that name one of them.
    pub fn forget(&mut self, outline: &Outline, edited: Edited<'_>) {
        self.invalidate(outline, std::iter::empty(), edited);
    }

    /// Drops what the store keeps about what `edited` holds and returns the indexes of the instances in `list` that it touched.
    pub fn invalidate<'l>(
        &mut self,
        outline: &Outline,
        list: impl Iterator<Item = &'l reach::Instance>,
        edited: Edited<'_>,
    ) -> Vec<usize> {
        let mine = |item: ItemRef| edited.holds(item.krate, item.file);
        let block_kept = |block: &impls::ImplRef| match *block {
            impls::ImplRef::Module { krate, index } => outline.crates[krate as usize]
                .impls
                .get(index as usize)
                .is_some_and(|e| {
                    !edited.holds(krate, e.file as u32)
                        && !e.parts.iter().any(|&p| edited.holds(krate, p as u32))
                }),
            impls::ImplRef::Body(item) => !mine(item),
        };
        let mut naming = Naming::new(&self.types, &mine);
        let touched: Vec<usize> = list
            .enumerate()
            .filter_map(|(i, inst)| naming.touches(inst).then_some(i))
            .collect();
        macro_rules! retain_each {
            ($pred:expr; $($t:expr),+ $(,)?) => { $($t.retain($pred);)+ };
        }
        retain_each!(
            |item, _| !mine(*item);
            self.signatures, self.bodies, self.adts, self.lookups.trait_autos,
            self.lookups.trait_defs, self.lookups.discriminants, self.lookups.reprs, self.lookups.attrs
        );
        self.headers.retain(|block, _| block_kept(block));
        retain_each!(|instance, _| !naming.touches(instance); self.finals, self.evaluated, self.lookups.provisional_rets);
        retain_each!(
            |ty, _| !naming.names(*ty);
            self.layouts, self.tails, self.closures, self.lookups.drops, self.lookups.own_drops,
            self.lookups.unpins, self.lookups.freezes, self.lookups.glues, self.lookups.fields, self.lookups.futures
        );
        self.runner.clear();
        self.lookups
            .forget_edited(outline, edited, &mut naming, &block_kept);
        touched
    }
}

fn has_mark(outline: &Outline, crates: &[u32], name: &str) -> bool {
    let bit = crate::outline::item::mark(name);
    crates
        .iter()
        .any(|&k| slice_has_mark(&outline.crates[k as usize].files, bit))
}

impl Lookups {
    fn forget_edited<F: Fn(ItemRef) -> bool>(
        &mut self,
        outline: &Outline,
        edited: Edited<'_>,
        naming: &mut Naming<'_, F>,
        block_kept: &dyn Fn(&impls::ImplRef) -> bool,
    ) {
        let mine = naming.mine;
        self.forms.retain(|(form, _), found| {
            let own = match *form {
                infer::Form::Adt(i) | infer::Form::Fn(i) => !mine(i),
                infer::Form::Impl(b) => block_kept(&b),
            };
            own && found
                .iter()
                .flat_map(|t| t.iter().flatten())
                .all(|&ty| !naming.names(ty))
        });
        self.reveals.retain(|instance| !naming.touches(instance));
        self.impl_traits.retain(|block, _| block_kept(block));
        self.heads.retain(|holder, _| match *holder {
            impls::Holder::Crate(krate) => !edited.touches_crate(krate),
            impls::Holder::Body(item) => !mine(item),
        });
        let keep_crate = |k: u32| !edited.touches_crate(k);
        self.modules.retain(|k, _| keep_crate(*k));
        self.owners.retain(|k, _| keep_crate(*k));
        if let Some(heads) = &mut self.sysroot_heads
            && heads.keys().any(|&k| !keep_crate(k))
        {
            let map = std::sync::Arc::make_mut(heads);
            map.retain(|&k, _| keep_crate(k));
            if map.is_empty() {
                self.sysroot_heads = None;
            }
        }
        self.scope_traits.retain(|home, _| keep_crate(home.krate));
        self.type_defs.retain(|key, _| keep_crate(key.0.0.krate));
        self.selections.retain(|(tr, ty, s), found| {
            naming.trait_kept(*tr, *ty, *s)
                && found
                    .as_ref()
                    .is_none_or(|(b, a)| block_kept(b) && !naming.substs(*a))
        });
        self.auto_traits
            .retain(|(tr, ty), _| naming.trait_kept(*tr, *ty, Substs::EMPTY));
        self.checked_impls.retain_keys(|block| block_kept(block));
        self.projections.retain(|(tr, ty, s, _, gat), found| {
            naming.trait_kept(*tr, *ty, *s)
                && !naming.substs(*gat)
                && found.is_none_or(|t| !naming.names(t))
        });
        self.trait_calls.retain(|(i, s), r| {
            naming.fn_kept(*i, *s)
                && match *r {
                    infer::Res::Local(_) => true,
                    infer::Res::Fn(ri, rs) | infer::Res::Ctor(ri, _, rs) => naming.fn_kept(ri, rs),
                    infer::Res::Const(ty) => !naming.names(ty),
                }
        });
        self.assoc_types.retain(|(tr, ty, s, a, gat), res| {
            naming.trait_kept(*tr, *ty, *s)
                && !mine(*a)
                && !naming.substs(*gat)
                && !naming.names(*res)
        });
        self.summaries.retain_keys(|item| !mine(*item));
        let mut alloc = write_lock(&self.state.global_allocator);
        if matches!(*alloc, Some(Some(i)) if mine(i))
            || matches!((*alloc, edited), (Some(None), Edited::Crates(c)) if has_mark(outline, c, "global_allocator"))
        {
            *alloc = None;
        }
        drop(alloc);
        if let Edited::Crates(crates) = edited {
            if self.lang_items.get().is_some_and(|l| {
                l.map.keys().any(|i| crates.contains(&i.krate)) || has_mark(outline, crates, "lang")
            }) {
                self.lang_items = Arc::new(std::sync::OnceLock::new());
            }
            let mut guard = write_lock(&self.state.loaded);
            if guard.len() == outline.crates.len() {
                for &k in crates {
                    guard[k as usize] = outline.crates[k as usize].extent();
                }
            }
        }
        let mut langs = write_lock(&self.state.langs);
        if langs
            .as_ref()
            .is_some_and(|f| f.crates.iter().any(|&k| edited.touches_crate(k)))
        {
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
