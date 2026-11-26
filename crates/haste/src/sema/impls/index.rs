use rustc_hash::FxHashMap as HashMap;

use crate::names::ItemRef;

use super::{Head, ImplRef};

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
        Self {
            modules: requests.modules.into_inner(),
            names: requests.names.into_inner(),
            crates: requests.crates.into_inner(),
            scopes: requests.scopes.into_inner(),
            procs: requests.procs.into_inner(),
        }
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
}

impl Index {
    /// Whether building the index asked for nothing, so it holds every impl its holder has.
    pub fn complete(&self) -> bool {
        self.asked.is_empty()
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
        }
    }
}
