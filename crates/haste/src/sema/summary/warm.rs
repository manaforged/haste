use std::hash::{Hash as _, Hasher as _};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use rustc_hash::FxHasher;

use crate::body::sig::ImplHeader;
use crate::names::{Def, ItemRef, ModRef, Names};
use crate::outline::{Outline, Role};
use crate::sema::impls::{Holder, ImplRef, Index as ImplIndex};
use crate::sema::infer::{Fields, Form};
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, Types};

use super::SummaryKind;
use super::codec::{Cursor, read_str, read_vec, write_opt, write_str, write_u32, write_u64};
mod count;
mod flights;
use super::warm_decode::decode_payload;
use super::warm_encode::encode_payload;
use super::warm_expr::write_slice;
use count::{count_fixed_entries, raw_entries_at_least};

const WARM_MAGIC: [u8; 8] = *b"HSTWRM01";
const WARM_VERSION: u32 = 6;
pub(crate) const WARM_MIN_LEN: usize = 28;
pub(super) const WARM_DEF_SHARDS: usize = 4;
pub(super) const WARM_HDR_SHARDS: usize = 8;
pub(super) const WARM_SUM_SHARDS: usize = 4;
pub(super) const WARM_NON_SHARD_SECTIONS: usize = 8;
pub(super) const WARM_SECTIONS: usize =
    WARM_NON_SHARD_SECTIONS + WARM_DEF_SHARDS + WARM_HDR_SHARDS + WARM_SUM_SHARDS;
const MAX_WARM_PAYLOAD_BYTES: usize = 128 * 1024 * 1024;

fn debug_warm() -> bool {
    crate::env_flag("HASTE_DEBUG_WARM")
        || std::env::var_os("HASTE_DEBUG").is_some_and(|v| v == "warm" || v == "1")
}
const ROLES: [Role; 6] = [
    Role::Lib,
    Role::Bin,
    Role::Example,
    Role::Test,
    Role::Bench,
    Role::Script,
];

fn role_tag(role: Role) -> u8 {
    ROLES.iter().position(|&r| r == role).unwrap_or(0) as u8
}

/// Deterministic workspace and toolchain cache key for one-shot cold check look-ahead snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WarmKey {
    pub toolchain_key: u64,
    pub deps_key: Option<u64>,
    pub target_key: u64,
    pub workspace_key: Option<u64>,
    pub release: bool,
}

impl WarmKey {
    pub(crate) fn fingerprint(self) -> u64 {
        let mut h = FxHasher::default();
        WARM_MAGIC.hash(&mut h);
        WARM_VERSION.hash(&mut h);
        self.toolchain_key.hash(&mut h);
        self.deps_key.hash(&mut h);
        self.target_key.hash(&mut h);
        self.workspace_key.hash(&mut h);
        self.release.hash(&mut h);
        h.finish()
    }

    pub(crate) fn path(self) -> PathBuf {
        crate::graph::store::home()
            .join("reach")
            .join(format!("{:016x}.warm", self.fingerprint()))
    }
}

#[derive(Clone, Copy)]
pub(super) struct WarmCounts {
    pub fixed_entries: u32,
}

#[derive(Default)]
pub(in crate::sema) struct WarmTimings {
    pub payload_bytes: usize,
    pub item_count: usize,
    pub ty_count: usize,
    pub sym_count: usize,
    pub prelude_us: u64,
    pub sections_us: u64,
    pub section_bytes: [u32; WARM_SECTIONS],
}

pub(in crate::sema) struct WarmFlights {
    pub reveals: Vec<(crate::sema::reach::Instance, Ty)>,
    pub checked_impls: Vec<(ImplRef, bool)>,
    pub summaries: Vec<(ItemRef, SummaryKind)>,
    pub body_heads: Vec<(Holder, Arc<ImplIndex>)>,
    pub body_headers: Vec<(ImplRef, Arc<ImplHeader>)>,
    pub body_forms: Vec<((Form, usize), Option<Fields>)>,
    pub body_impl_traits: Vec<(ImplRef, Option<ItemRef>)>,
    pub body_selections: Vec<((ItemRef, Ty, Substs), Option<(ImplRef, Substs)>)>,
    pub timings: WarmTimings,
}

enum WarmSource {
    Backing(crate::substrate::Backing),
    #[cfg(test)]
    Bytes(Vec<u8>),
}

impl WarmSource {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Backing(b) => b.as_slice(),
            #[cfg(test)]
            Self::Bytes(v) => v,
        }
    }
}

#[derive(Clone)]
struct PreExtent {
    files: u32,
    modules: u32,
    impls: u32,
    impl_bodies: u32,
}

#[derive(Default)]
struct WarmInner {
    key: Option<WarmKey>,
    pending: Option<WarmSource>,
    pre_extents: Option<Vec<PreExtent>>,
    loaded_counts: Option<WarmCounts>,
    attempted_save: bool,
}

/// Shared one-shot warm check snapshot state held by [`super::super::lookups::Lookups`].
#[derive(Default)]
pub struct WarmState {
    inner: Mutex<WarmInner>,
}

impl WarmState {
    /// Attaches `key` and any persisted warm snapshot backing for this build cache.
    pub(crate) fn set(&self, key: WarmKey, backing: Option<crate::substrate::Backing>) {
        let mut guard = crate::sync::lock(&self.inner);
        guard.key = Some(key);
        guard.pending = backing.map(WarmSource::Backing);
        guard.pre_extents = None;
        guard.loaded_counts = None;
        guard.attempted_save = false;
    }

    pub(crate) fn toolchain_key(&self) -> Option<u64> {
        crate::sync::lock(&self.inner).key.map(|k| k.toolchain_key)
    }

    #[cfg(test)]
    pub(super) fn set_bytes_for_test(&self, key: WarmKey, bytes: Vec<u8>) {
        let mut guard = crate::sync::lock(&self.inner);
        guard.key = Some(key);
        guard.pending = Some(WarmSource::Bytes(bytes));
        guard.pre_extents = None;
        guard.loaded_counts = None;
        guard.attempted_save = false;
    }
}

pub(super) fn is_settled_fixed(outline: &Outline, krate: u32) -> bool {
    outline.is_sysroot(krate)
        || outline
            .pins
            .get(krate as usize)
            .is_some_and(Option::is_some)
}

pub(super) struct WarmFilter<'a> {
    outline: &'a Outline,
    pre: &'a [PreExtent],
}

impl<'a> WarmFilter<'a> {
    pub(super) fn krate(&self, krate: u32) -> bool {
        (krate as usize) < self.pre.len() && is_settled_fixed(self.outline, krate)
    }

    pub(super) fn pre_files(&self, krate: u32) -> u32 {
        self.pre.get(krate as usize).map_or(0, |p| p.files)
    }

    pub(super) fn file(&self, krate: u32, file: u32) -> bool {
        self.krate(krate) && file < self.pre_files(krate)
    }

    pub(super) fn item(&self, item: ItemRef) -> bool {
        item.outer().is_none() && self.file(item.krate, item.file)
    }

    pub(super) fn mod_ref(&self, m: ModRef) -> bool {
        self.krate(m.krate) && m.module < self.pre[m.krate as usize].modules
    }

    pub(super) fn holder(&self, holder: crate::sema::impls::Holder) -> bool {
        match holder {
            crate::sema::impls::Holder::Crate(k) => self.krate(k),
            crate::sema::impls::Holder::Body(root) => self.item(root),
        }
    }

    pub(super) fn impl_ref(&self, block: ImplRef) -> bool {
        match block {
            ImplRef::Module { krate, index } => {
                self.krate(krate) && index < self.pre[krate as usize].impls
            }
            ImplRef::Body(item) => item.declaring_body().is_some_and(|root| self.item(root)),
        }
    }

    pub(super) fn def(&self, def: &Def) -> bool {
        match *def {
            Def::Module(m) => self.mod_ref(m),
            Def::Item(i) | Def::Variant(i, _) => self.item(i),
            Def::Primitive(_) => true,
        }
    }

    pub(super) fn substs(&self, types: &Types, s: Substs, memo: &mut Vec<u8>) -> bool {
        types.substs(s).iter().all(|&ty| self.ty(types, ty, memo))
    }

    pub(super) fn ty(&self, types: &Types, ty: Ty, memo: &mut Vec<u8>) -> bool {
        if ty == Ty::NONE {
            return true;
        }
        if types.has_var_flag(ty) {
            return false;
        }
        let idx = ty.0 as usize;
        if let Some(&v) = memo.get(idx)
            && v != 0
        {
            return v == 2;
        }
        if idx >= memo.len() {
            memo.resize(idx + 1, 0);
        }
        memo[idx] = 1;
        let ok = match types.kind(ty) {
            Kind::Prim(_) | Kind::Never | Kind::Value(_) | Kind::Param(_) => true,
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => false,
            Kind::Ref(_, inner)
            | Kind::Ptr(_, inner)
            | Kind::Slice(inner)
            | Kind::Pat(inner, _) => self.ty(types, *inner, memo),
            Kind::Array(elem, len) => self.ty(types, *elem, memo) && self.ty(types, *len, memo),
            Kind::Tuple(s) => self.substs(types, *s, memo),
            Kind::FnPtr(p, r) | Kind::UnsafeFnPtr(p, r) => {
                self.substs(types, *p, memo) && self.ty(types, *r, memo)
            }
            Kind::Adt(i, s) | Kind::FnDef(i, s) | Kind::Closure(i, s, _) => {
                self.item(*i) && self.substs(types, *s, memo)
            }
            Kind::Dyn(o, s, m, a) => {
                self.item(*o)
                    && self.substs(types, *s, memo)
                    && types.items(*m).iter().all(|&i| self.item(i))
                    && types.items(*a).iter().all(|&i| self.item(i))
            }
            Kind::Proj(id) => {
                let d = types.proj(*id);
                self.item(d.of)
                    && self.item(d.assoc)
                    && self.ty(types, d.self_ty, memo)
                    && self.substs(types, d.args, memo)
                    && self.substs(types, d.own, memo)
            }
        };
        memo[idx] = if ok { 2 } else { 1 };
        ok
    }
}

fn snapshot_extents(outline: &Outline) -> Vec<PreExtent> {
    outline
        .crates
        .iter()
        .map(|tree| PreExtent {
            files: tree.files.len() as u32,
            modules: tree.modules.len() as u32,
            impls: tree.impls.len() as u32,
            impl_bodies: tree.impl_bodies.len() as u32,
        })
        .collect()
}

fn local_crates_have_lang_marks(outline: &Outline) -> bool {
    let lang_bit = crate::outline::item::mark("lang");
    outline.crates.iter().enumerate().any(|(k, tree)| {
        !is_settled_fixed(outline, k as u32)
            && tree
                .files
                .iter()
                .any(|f| f.items.marks.iter().any(|&(_, bits)| bits & lang_bit != 0))
    })
}

fn validate_stamps(cur: &mut Cursor<'_>, outline: &Outline, expected_key: u64) -> Result<bool> {
    if cur.take(8)? != WARM_MAGIC || cur.u32()? != WARM_VERSION || cur.u64()? != expected_key {
        return Ok(false);
    }
    for _ in 0..cur.count()? {
        let k = cur.u32()? as usize;
        let is_sysroot = cur.u8()? != 0;
        let has_pin = cur.u8()? != 0;
        let pin = if has_pin { Some(cur.u64()?) } else { None };
        let name = read_str(cur)?;
        let role = cur.u8()?;
        let pre_files = cur.u32()?;
        let pre_modules = cur.u32()?;
        let pre_impls = cur.u32()?;
        let pre_impl_bodies = cur.u32()?;
        let file_stamps = read_vec(cur, |c| Ok((c.u32()?, c.u32()?, c.u32()?)))?;
        let Some(tree) = outline.crates.get(k) else {
            return Ok(false);
        };
        let root = &outline.roots[k];
        if !is_settled_fixed(outline, k as u32)
            || outline.is_sysroot(k as u32) != is_sysroot
            || outline.pins[k] != pin
            || root.name != name
            || role_tag(root.role) != role
            || tree.files.len() as u32 != pre_files
            || tree.modules.len() as u32 != pre_modules
            || tree.impls.len() as u32 != pre_impls
            || tree.impl_bodies.len() as u32 != pre_impl_bodies
        {
            return Ok(false);
        }
        for (f, src_len, item_len) in file_stamps {
            let Some(file) = tree.files.get(f as usize) else {
                return Ok(false);
            };
            if file.source.len() != src_len || file.items.items.len() as u32 != item_len {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn write_stamps(
    out: &mut Vec<u8>,
    outline: &Outline,
    pre: &[PreExtent],
    items: &[ItemRef],
    key_hash: u64,
) -> Option<()> {
    let mut files_by_crate: Vec<Vec<u32>> = vec![Vec::new(); pre.len()];
    for item in items {
        if let Some(bucket) = files_by_crate.get_mut(item.krate as usize) {
            bucket.push(item.file);
        }
    }
    for bucket in &mut files_by_crate {
        bucket.sort_unstable();
        bucket.dedup();
    }
    out.extend_from_slice(&WARM_MAGIC);
    write_u32(out, WARM_VERSION);
    write_u64(out, key_hash);
    let fixed_crates: Vec<usize> = (0..pre.len())
        .filter(|&k| is_settled_fixed(outline, k as u32))
        .collect();
    write_u32(out, fixed_crates.len() as u32);
    for k in fixed_crates {
        let ext = &pre[k];
        let root = &outline.roots[k];
        let tree = &outline.crates[k];
        write_u32(out, k as u32);
        out.push(u8::from(outline.is_sysroot(k as u32)));
        write_opt(out, outline.pins[k].as_ref(), |o, p| {
            write_u64(o, *p);
            Some(())
        })?;
        write_str(out, &root.name)?;
        out.push(role_tag(root.role));
        write_u32(out, ext.files);
        write_u32(out, ext.modules);
        write_u32(out, ext.impls);
        write_u32(out, ext.impl_bodies);
        let used: Vec<(u32, u32, u32)> = files_by_crate[k]
            .iter()
            .filter_map(|&f| {
                let file = tree.files.get(f as usize)?;
                Some((f, file.source.len(), file.items.items.len() as u32))
            })
            .collect();
        write_slice(out, &used, |o, &(f, sl, il)| {
            write_u32(o, f);
            write_u32(o, sl);
            write_u32(o, il);
        });
    }
    Some(())
}

fn decode_warm_blob(
    bytes: &[u8],
    key_hash: u64,
    outline: &Outline,
    names: &Names,
    store: &mut crate::sema::Cache,
) -> Result<Option<(WarmCounts, WarmFlights)>> {
    if bytes.len() > MAX_WARM_PAYLOAD_BYTES {
        bail!("warm cache payload exceeds size bound");
    }
    let mut cur = Cursor::new(bytes);
    if !validate_stamps(&mut cur, outline, key_hash)? {
        return Ok(None);
    }
    let fixed_entries = cur.u32()?;
    let allow_lang_map = !local_crates_have_lang_marks(outline);
    let flights = decode_payload(
        &bytes[cur.pos..],
        store,
        outline,
        &names.symbols,
        allow_lang_map,
        &|k| is_settled_fixed(outline, k),
    )?;
    Ok(Some((WarmCounts { fixed_entries }, flights)))
}

fn encode_warm_blob(
    key_hash: u64,
    outline: &Outline,
    names: &Names,
    store: &crate::sema::Cache,
    pre: &[PreExtent],
    raw_count: Option<u32>,
) -> Option<Vec<u8>> {
    let filter = WarmFilter { outline, pre };
    let (payload, items, encoded_count) = encode_payload(store, outline, &names.symbols, &filter)?;
    if payload.len() > MAX_WARM_PAYLOAD_BYTES {
        return None;
    }
    let mut out = Vec::with_capacity(pre.len() * 48 + payload.len() + 64);
    write_stamps(&mut out, outline, pre, &items, key_hash)?;
    write_u32(&mut out, raw_count.unwrap_or(encoded_count));
    out.extend_from_slice(&payload);
    Some(out)
}

/// Preloads any persisted one-shot check snapshot into `store` and returns `Flight` entries to register on the new context.
pub(in crate::sema) fn warm_preload(
    outline: &Outline,
    names: &Names,
    store: &mut crate::sema::Cache,
) -> Option<WarmFlights> {
    let warm = std::sync::Arc::clone(&store.lookups.warm);
    let (key, pending) = {
        let mut guard = crate::sync::lock(&warm.inner);
        if guard.pre_extents.is_none() {
            guard.pre_extents = Some(snapshot_extents(outline));
        }
        (guard.key?, guard.pending.take()?)
    };
    let Ok(Some((counts, flights))) =
        decode_warm_blob(pending.as_slice(), key.fingerprint(), outline, names, store)
    else {
        return None;
    };
    let extents = outline
        .crates
        .iter()
        .map(crate::outline::tree::Crate::extent)
        .collect();
    *crate::sema::write_lock(&store.lookups.state.loaded) = extents;
    crate::sync::lock(&warm.inner).loaded_counts = Some(counts);
    Some(flights)
}

pub(in crate::sema) use flights::warm_populate_flights;

/// Saves newly populated settled-fixed entries to the one-shot check snapshot when entries grew since load.
pub(crate) fn save_warm_if_grown(cx: &crate::sema::Context<'_>) {
    if !crate::substrate::enabled() {
        return;
    }
    let store = cx.store();
    let warm = &store.lookups.warm;
    let (key, pre, loaded_counts) = {
        let mut guard = crate::sync::lock(&warm.inner);
        if guard.attempted_save {
            return;
        }
        guard.attempted_save = true;
        let Some(key) = guard.key else {
            return;
        };
        let pre = guard
            .pre_extents
            .get_or_insert_with(|| snapshot_extents(cx.outline))
            .clone();
        (key, pre, guard.loaded_counts)
    };
    let raw_count = if let Some(prev) = loaded_counts {
        let threshold = prev
            .fixed_entries
            .saturating_add((prev.fixed_entries / 8).max(128));
        if !raw_entries_at_least(store, threshold) {
            return;
        }
        let filter = WarmFilter {
            outline: cx.outline,
            pre: &pre,
        };
        let count = count_fixed_entries(store, &filter);
        if count < threshold {
            return;
        }
        Some(count)
    } else {
        None
    };
    let Some(blob) = encode_warm_blob(
        key.fingerprint(),
        cx.outline,
        cx.names,
        store,
        &pre,
        raw_count,
    ) else {
        return;
    };
    let _ = crate::graph::store::write_new_bytes(&key.path(), &blob);
}

#[cfg(test)]
mod tests;
