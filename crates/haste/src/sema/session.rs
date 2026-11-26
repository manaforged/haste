use rustc_hash::FxHashMap as HashMap;

use crate::names::path::Requests;
use crate::names::{ItemRef, Names};
use crate::outline::Outline;

use super::lookups::{SegKey, TypeScope};
use super::table::Table;
use super::{Cache, Context, impls, reach, substs, table, ty};

impl<'a> Context<'a> {
    /// Returns the shared inference cache attached to this context.
    pub(crate) fn store(&self) -> &Cache {
        &self.store
    }

    /// Drops cached reach-only tables to cut RSS before one-shot codegen while keeping caches codegen reads.
    pub(crate) fn compact_for_oneshot_codegen(&mut self) {
        self.store.finals.clear();
        self.store.lookups.summaries = table::Flight::default();
        self.store.lookups.selections.clear();
        self.store.lookups.auto_traits.clear();
        self.store.lookups.projections.clear();
        self.store.lookups.trait_calls.clear();
        self.store.lookups.assoc_types.clear();
        self.store.lookups.type_defs.clear();
        self.type_defs_local.borrow_mut().clear();
        self.type_defs_single.borrow_mut().clear();
        self.selections_local.borrow_mut().clear();
        self.scope_traits_local.borrow_mut().clear();
        self.store.lookups.scope_traits.clear();
        self.store.lookups.reveals = table::Flight::default();
        self.reveals_local.clear();
        self.store.lookups.futures.clear();
        self.store.lookups.forms.clear();
        self.aliases.clear();
        self.eval_errors.clear();
    }

    /// A pass over an outline and its resolved names that reuses `store`.
    pub fn new(outline: &'a Outline, names: &'a Names, mut store: Cache) -> Context<'a> {
        let types = store.types.clone();
        store.lookups.check(outline);
        let (kids, trait_members, impl_members, aliases, asks) = Default::default();
        Context {
            outline,
            names,
            types,
            asks,
            in_parallel: false,
            deferred_evals: Vec::new(),
            store,
            kids,
            trait_members,
            impl_members,
            aliases,
            eval_errors: Table::default(),
            eval_stack: Vec::new(),
            drop_stack: Vec::new(),
            drop_cycle_at: usize::MAX,
            discr_visiting: Vec::new(),
            form_stack: Vec::new(),
            global_asm: None,
            langs: None,
            lang_map: std::cell::OnceCell::new(),
            crate_owners: vec![None; outline.crates.len()],
            crate_modules: vec![None; outline.crates.len()],
            crate_heads: vec![None; outline.crates.len()],
            headers_local: HashMap::default(),
            trait_members_local: std::cell::RefCell::new(HashMap::default()),
            impl_members_local: std::cell::RefCell::new(HashMap::default()),
            intrinsics_local: std::cell::RefCell::new(HashMap::default()),
            type_defs_local: std::cell::RefCell::new(HashMap::default()),
            type_defs_single: std::cell::RefCell::new(HashMap::default()),
            selections_local: std::cell::RefCell::new(HashMap::default()),
            scope_traits_local: std::cell::RefCell::new(HashMap::default()),
            glues_local: HashMap::default(),
            drops_local: HashMap::default(),
            reveals_local: HashMap::default(),
            ty_facts: HashMap::default(),
        }
    }

    /// A sibling context over the same outline, names, types and caches, with its own empty request list so a worker thread can run without contending on asks.
    pub fn fork(&self) -> Context<'a> {
        let crates = self.outline.crates.len();
        Context {
            outline: self.outline,
            names: self.names,
            types: self.types.clone(),
            asks: Requests::default(),
            in_parallel: true,
            deferred_evals: Vec::new(),
            store: self.store.clone(),
            kids: self.kids.clone(),
            trait_members: self.trait_members.clone(),
            impl_members: self.impl_members.clone(),
            aliases: self.aliases.clone(),
            eval_errors: self.eval_errors.clone(),
            eval_stack: Vec::new(),
            drop_stack: Vec::new(),
            drop_cycle_at: usize::MAX,
            discr_visiting: Vec::new(),
            form_stack: Vec::new(),
            global_asm: self.global_asm.clone(),
            langs: self.langs.clone(),
            lang_map: std::cell::OnceCell::new(),
            crate_owners: vec![None; crates],
            crate_modules: vec![None; crates],
            crate_heads: vec![None; crates],
            headers_local: HashMap::default(),
            trait_members_local: std::cell::RefCell::new(HashMap::default()),
            impl_members_local: std::cell::RefCell::new(HashMap::default()),
            intrinsics_local: std::cell::RefCell::new(HashMap::default()),
            type_defs_local: std::cell::RefCell::new(HashMap::default()),
            type_defs_single: std::cell::RefCell::new(HashMap::default()),
            selections_local: std::cell::RefCell::new(HashMap::default()),
            scope_traits_local: std::cell::RefCell::new(HashMap::default()),
            glues_local: HashMap::default(),
            drops_local: HashMap::default(),
            reveals_local: HashMap::default(),
            ty_facts: HashMap::default(),
        }
    }

    /// Takes the constant evaluations that a parallel pass deferred until worker threads joined.
    pub fn take_deferred_evals(&mut self) -> Vec<(reach::Instance, u32)> {
        std::mem::take(&mut self.deferred_evals)
    }

    /// The lengths of `asks` and `deferred_evals`, unchanged only when a lookup neither asked for unloaded names nor deferred a constant evaluation.
    pub(crate) fn quiet_mark(&self) -> (usize, usize) {
        (self.asks.count(), self.deferred_evals.len())
    }

    /// Whether no lookup in this pass asked for anything to load or resolve, so another pass would find the same.
    pub fn resolved(&self) -> bool {
        self.asks.count() == 0
    }

    /// Ends the pass: the store for the next one, and what lookups asked to load or resolve.
    pub fn release(mut self) -> (Cache, Requests) {
        self.store.types = self.types;
        (self.store, self.asks)
    }

    /// The cached type-namespace definition for `segments` in `scope`.
    pub fn type_def(
        &self,
        scope: TypeScope,
        segments: &[crate::names::Sym],
    ) -> Option<crate::names::Def> {
        if let (&[sym], (home, None, None, false, None)) = (segments, scope) {
            if let Some(&found) = self.type_defs_single.borrow().get(&(home, sym)) {
                return Some(found);
            }
            let key = (scope, SegKey::from_slice(segments));
            let found = self.store.lookups.type_defs.get(&key)?;
            self.type_defs_single
                .borrow_mut()
                .insert((home, sym), found);
            return Some(found);
        }
        let key = (scope, SegKey::from_slice(segments));
        if let Some(&found) = self.type_defs_local.borrow().get(&key) {
            return Some(found);
        }
        let found = self.store.lookups.type_defs.get(&key)?;
        self.type_defs_local.borrow_mut().insert(key, found);
        Some(found)
    }

    /// Caches `def` as the type-namespace definition for `segments` in `scope`.
    pub fn set_type_def(
        &self,
        scope: TypeScope,
        segments: &[crate::names::Sym],
        def: crate::names::Def,
    ) {
        let key = (scope, SegKey::from_slice(segments));
        self.store.lookups.type_defs.insert(key.clone(), def);
        if let (&[sym], (home, None, None, false, None)) = (segments, scope) {
            self.type_defs_single.borrow_mut().insert((home, sym), def);
        } else {
            self.type_defs_local.borrow_mut().insert(key, def);
        }
    }

    /// The cached closed trait selection for `key`, backed by a per-context cache over the shared table.
    pub(crate) fn selection(
        &self,
        key: &(ItemRef, ty::Ty, substs::Substs),
    ) -> Option<Option<(impls::ImplRef, substs::Substs)>> {
        if let Some(&found) = self.selections_local.borrow().get(key) {
            return Some(found);
        }
        let found = self.store.lookups.selections.get(key)?;
        self.selections_local.borrow_mut().insert(*key, found);
        Some(found)
    }

    /// Caches `picked` as the closed trait selection for `key`.
    pub(crate) fn set_selection(
        &self,
        key: (ItemRef, ty::Ty, substs::Substs),
        picked: Option<(impls::ImplRef, substs::Substs)>,
    ) {
        self.store.lookups.selections.insert(key, picked);
        self.selections_local.borrow_mut().insert(key, picked);
    }

    /// The cached ground `Send`/`Sync` auto-trait outcome for `key`.
    pub(crate) fn auto_trait(&self, key: &(ItemRef, ty::Ty)) -> Option<bool> {
        self.store.lookups.auto_traits.get(key)
    }

    /// Caches `holds` as the ground `Send`/`Sync` auto-trait outcome for `key`.
    pub(crate) fn set_auto_trait(&self, key: (ItemRef, ty::Ty), holds: bool) {
        self.store.lookups.auto_traits.insert(key, holds);
    }
}
