use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;

use hir_def::builtin_derive::BuiltinDeriveImplMethod as Method;
use hir_def::{FunctionId, StaticId};
use hir_ty::db::InternedClosureId;
use hir_ty::next_solver::{StoredGenericArgs, StoredTy};
use hir_ty::traits::StoredParamEnvAndCrate;

use crate::lower::native::NativeSig;

pub(super) const TABLE_SLOTS: usize = 1 << 20;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Func(FunctionId, StoredGenericArgs),
    Closure(InternedClosureId<'static>, StoredGenericArgs),
    Glue(StoredTy),
    Thunk(Box<Key>, bool),
    Derive(Method, FunctionId, StoredGenericArgs),
    Const(hir_def::ConstId, StoredGenericArgs),
    Shim(StoredTy),
    Static(hir_def::StaticId),
    Native(Box<Key>, NativeSig),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sig {
    pub params: usize,
    pub ret: bool,
    pub by_value: u64,
}

impl Sig {
    /// A signature that passes every parameter by address.
    pub const fn new(params: usize, ret: bool) -> Self {
        Sig { params, ret, by_value: 0 }
    }

    /// Whether parameter `index` arrives as its 8-byte value instead of its address.
    pub fn passes_value(self, index: usize) -> bool {
        index < 64 && self.by_value & (1 << index) != 0
    }
}

pub(super) struct SlotInfo {
    pub(super) key: Key,
    pub(super) sig: Sig,
    pub(super) env: StoredParamEnvAndCrate,
}

pub struct Slots {
    pub(super) table: &'static [AtomicUsize],
    pub(super) by_key: HashMap<(Key, Sig), usize>,
    pub(super) info: Vec<SlotInfo>,
    pub(super) pending: Vec<usize>,
    pub native_words: Vec<(usize, usize)>,
    pub statics: HashMap<StaticId, usize>,
    pub lazy_statics: HashMap<StaticId, (usize, usize)>,
    pub thread_locals: HashMap<StaticId, usize>,
    pub vtables: HashMap<(StoredTy, StoredTy), i64>,
    pub notes: Vec<String>,
}

impl Slots {
    pub fn functions(&self) -> Vec<FunctionId> {
        let plain = |info: &SlotInfo| match &info.key {
            Key::Func(function, _) | Key::Derive(_, function, _) => Some(*function),
            _ => None,
        };
        self.info.iter().filter_map(plain).collect()
    }

    pub fn sig_of(&self, slot: usize) -> Sig {
        self.info[slot].sig
    }

    pub fn get(&mut self, key: Key, sig: Sig, env: StoredParamEnvAndCrate) -> Result<usize, String> {
        if let Some(slot) = self.by_key.get(&(key.clone(), sig)) {
            return Ok(*slot);
        }
        let slot = self.info.len();
        if slot == TABLE_SLOTS {
            return Err("call table is full".to_owned());
        }
        self.info.push(SlotInfo { key: key.clone(), sig, env });
        self.by_key.insert((key, sig), slot);
        self.pending.push(slot);
        Ok(slot)
    }

    pub fn address(&self, slot: usize) -> i64 {
        &self.table[slot] as *const AtomicUsize as i64
    }
}
