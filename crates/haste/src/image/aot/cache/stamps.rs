use std::hash::{Hash as _, Hasher as _};
use std::path::Path;
use std::sync::OnceLock;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::graph::store::file_stamp;
use crate::image::aot::{AotCode, AotFn};
use crate::names::ItemRef;
use crate::outline::Outline;
use crate::outline::edit::hash_outside_bodies;
use crate::sema::reach::Reached;
use crate::sema::substs::Substs;
use crate::sema::ty::{Kind, Ty, Types};
use crate::substrate::{hash_crate_env, sysroot_item_key};
use crate::sync::lock;

use super::relocs::{hash_str, item_check_key, stable_file_key};
use crate::lower::SymbolSource;

pub(crate) struct Stamps {
    iface: Vec<u64>,
    files: Vec<Vec<u64>>,
    file_keys: Vec<OnceLock<Vec<Option<u64>>>>,
}

impl Stamps {
    pub(crate) fn compute(outline: &Outline, tc_key: u64, target_key: u64) -> Self {
        let n = outline.crates.len();
        let mut iface = vec![0u64; n];
        let mut files = vec![Vec::new(); n];
        let mut state = vec![0u8; n];
        for idx in 0..n {
            resolve_crate_stamps(
                outline, tc_key, target_key, idx, &mut iface, &mut files, &mut state,
            );
        }
        let file_keys = (0..n).map(|_| OnceLock::new()).collect();
        Self {
            iface,
            files,
            file_keys,
        }
    }

    pub(crate) fn iface(&self, krate: u32) -> Option<u64> {
        self.iface.get(krate as usize).copied()
    }

    pub(in crate::image::aot::cache) fn file(&self, krate: u32, file: u32) -> Option<u64> {
        let per = self.files.get(krate as usize)?;
        if per.is_empty() {
            self.iface(krate)
        } else {
            per.get(file as usize).copied()
        }
    }

    pub(in crate::image::aot::cache) fn file_key(
        &self,
        outline: &Outline,
        krate: u32,
        file: u32,
    ) -> Option<u64> {
        let slot = self.file_keys.get(krate as usize)?;
        let keys = slot.get_or_init(|| {
            let count = outline
                .crates
                .get(krate as usize)
                .map_or(0, |c| c.files.len());
            (0..count as u32)
                .map(|f| stable_file_key(outline, krate, f))
                .collect()
        });
        keys.get(file as usize).copied().flatten()
    }
}

fn resolve_crate_stamps(
    outline: &Outline,
    tc_key: u64,
    target_key: u64,
    idx: usize,
    iface: &mut [u64],
    files: &mut [Vec<u64>],
    state: &mut [u8],
) -> u64 {
    if state[idx] == 2 {
        return iface[idx];
    }
    if state[idx] == 1 {
        return 0;
    }
    state[idx] = 1;
    let mut h = FxHasher::default();
    hash_crate_env(outline, idx, tc_key, target_key, &mut h);
    let mut local_raw = Vec::new();
    if let (Some(tree), Some(root)) = (outline.crates.get(idx), outline.roots.get(idx)) {
        if tree.env.fixed {
            outline.pins.get(idx).copied().flatten().hash(&mut h);
            (&root.path, file_stamp(&root.path)).hash(&mut h);
            tree.env.features.hash(&mut h);
            tree.env.vars.hash(&mut h);
            if let Some(script) = &tree.env.script {
                script.key_parts().hash(&mut h);
            }
        } else {
            tree.env.features.hash(&mut h);
            tree.env.vars.hash(&mut h);
            if let Some(script) = &tree.env.script {
                script.key_parts().hash(&mut h);
            }
            let base_dir = root.path.parent().unwrap_or(Path::new(""));
            let mut plain_iface = Vec::new();
            let mut expanded_iface = Vec::new();
            local_raw.reserve(tree.files.len());
            for f in &tree.files {
                let rel = f
                    .source
                    .path
                    .strip_prefix(base_dir)
                    .unwrap_or(&f.source.path);
                let text_hash = hash_str(&f.source.text);
                if f.expanded {
                    let mut eh = FxHasher::default();
                    (rel, f.expand_origin, f.source.offset, text_hash).hash(&mut eh);
                    let exp = eh.finish();
                    expanded_iface.push((rel, f.expand_origin, f.source.offset, text_hash));
                    local_raw.push((rel, exp));
                } else {
                    let mut oh = FxHasher::default();
                    hash_outside_bodies(f, &mut oh);
                    let out_hash = oh.finish();
                    plain_iface.push((rel, out_hash));
                    local_raw.push((rel, text_hash));
                }
            }
            plain_iface.sort_unstable();
            plain_iface.hash(&mut h);
            expanded_iface.sort_unstable();
            expanded_iface.hash(&mut h);
            let mut inc: Vec<_> = outline
                .included(idx)
                .into_iter()
                .map(|(p, _)| {
                    let s = file_stamp(&p);
                    (p, s)
                })
                .collect();
            inc.sort_unstable();
            inc.dedup();
            inc.hash(&mut h);
            let mut env_reads: Vec<(String, Option<String>)> = {
                let guard = lock(&tree.env.env_reads);
                guard.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
            };
            env_reads.sort();
            env_reads.hash(&mut h);
        }
    }
    let mut deps: Vec<(&str, u64)> = outline
        .deps
        .get(idx)
        .into_iter()
        .flatten()
        .map(|(name, dep)| {
            (
                name.as_str(),
                resolve_crate_stamps(outline, tc_key, target_key, *dep, iface, files, state),
            )
        })
        .collect();
    deps.sort_unstable_by(|a, b| a.0.cmp(b.0));
    deps.hash(&mut h);
    let digest = h.finish();
    iface[idx] = digest;
    if !local_raw.is_empty() {
        files[idx] = local_raw
            .into_iter()
            .map(|(rel, body_hash)| {
                let mut fh = FxHasher::default();
                (digest, rel, body_hash).hash(&mut fh);
                fh.finish()
            })
            .collect();
    }
    state[idx] = 2;
    digest
}

pub(crate) fn stable_entry_item_key(
    outline: &Outline,
    stamps: &Stamps,
    item: ItemRef,
) -> Option<u64> {
    if outline.is_fixed(item.krate) {
        return sysroot_item_key(outline, item);
    }
    let (base, sub) = match item.outer() {
        Some((outer, sub)) => (outer, Some(sub)),
        None => (item, None),
    };
    let file_digest = stamps.file_key(outline, base.krate, base.file)?;
    let base_key = item_check_key(outline, stamps, file_digest, base)?;
    let mut h = FxHasher::default();
    base_key.hash(&mut h);
    sub.hash(&mut h);
    Some(h.finish())
}

fn push_unique_krate(out: &mut Vec<u32>, krate: u32) {
    if !out.contains(&krate) {
        out.push(krate);
    }
}

fn collect_substs_crates(types: &Types, substs: Substs, out: &mut Vec<u32>) {
    let slice = types.substs(substs);
    let mut stack: Vec<Ty> = Vec::with_capacity(slice.len());
    stack.extend_from_slice(slice);
    let mut small_seen: Vec<Ty> = Vec::with_capacity(slice.len().min(32));
    let mut large_seen: Option<HashSet<Ty>> = None;
    while let Some(ty) = stack.pop() {
        if ty == Ty::NONE {
            continue;
        }
        if let Some(set) = large_seen.as_mut() {
            if !set.insert(ty) {
                continue;
            }
        } else if small_seen.contains(&ty) {
            continue;
        } else if small_seen.len() < 32 {
            small_seen.push(ty);
        } else {
            let mut set: HashSet<Ty> = small_seen.drain(..).collect();
            set.insert(ty);
            large_seen = Some(set);
        }
        match *types.kind(ty) {
            Kind::Adt(item, s) | Kind::FnDef(item, s) | Kind::Closure(item, s, _) => {
                push_unique_krate(out, item.krate);
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Dyn(item, s, m, a) => {
                push_unique_krate(out, item.krate);
                for x in types.items(m).iter().chain(types.items(a)) {
                    push_unique_krate(out, x.krate);
                }
                stack.extend_from_slice(types.substs(s));
            }
            Kind::Proj(id) => {
                let p = types.proj(id);
                push_unique_krate(out, p.of.krate);
                push_unique_krate(out, p.assoc.krate);
                stack.push(p.self_ty);
                stack.extend_from_slice(types.substs(p.args));
                stack.extend_from_slice(types.substs(p.own));
            }
            ref other => {
                for child in other.children(types) {
                    stack.push(child);
                }
            }
        }
    }
}

pub(crate) fn touched_stamps_digest(
    types: &Types,
    stamps: &Stamps,
    item: ItemRef,
    substs: Substs,
) -> Option<u64> {
    let mut krates = Vec::with_capacity(4);
    krates.push(item.krate);
    if !substs.is_empty() {
        collect_substs_crates(types, substs, &mut krates);
        if krates.len() > 2 {
            krates.sort_unstable();
        } else if krates.len() == 2 && krates[0] > krates[1] {
            krates.swap(0, 1);
        }
    }
    let own_file_stamp = stamps.file(item.krate, item.body_owner().file)?;
    let mut h = FxHasher::default();
    own_file_stamp.hash(&mut h);
    krates.len().hash(&mut h);
    for k in krates {
        stamps.iface(k)?.hash(&mut h);
    }
    Some(h.finish())
}

pub(crate) fn stable_substs_hash(
    outline: &Outline,
    types: &Types,
    stamps: &Stamps,
    substs: Substs,
    memo: &mut HashMap<Ty, u64>,
) -> Option<u64> {
    let mut h = FxHasher::default();
    let slice = types.substs(substs);
    slice.len().hash(&mut h);
    for &ty in slice {
        stable_ty_hash(outline, types, stamps, ty, memo)?.hash(&mut h);
    }
    Some(h.finish())
}

fn stable_ty_hash(
    outline: &Outline,
    types: &Types,
    stamps: &Stamps,
    ty: Ty,
    memo: &mut HashMap<Ty, u64>,
) -> Option<u64> {
    if ty == Ty::NONE {
        return Some(0);
    }
    if let Some(&cached) = memo.get(&ty) {
        return Some(cached);
    }
    let mut h = FxHasher::default();
    let leaf = match *types.kind(ty) {
        Kind::Prim(p) => {
            (1u8, p).hash(&mut h);
            true
        }
        Kind::Never => {
            2u8.hash(&mut h);
            true
        }
        Kind::Value(v) => {
            (15u8, v).hash(&mut h);
            true
        }
        Kind::Param(p) => {
            (16u8, p).hash(&mut h);
            true
        }
        Kind::Tuple(s) => {
            3u8.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
            false
        }
        Kind::Ref(m, i) => {
            4u8.hash(&mut h);
            m.hash(&mut h);
            stable_ty_hash(outline, types, stamps, i, memo)?.hash(&mut h);
            false
        }
        Kind::Ptr(m, i) => {
            5u8.hash(&mut h);
            m.hash(&mut h);
            stable_ty_hash(outline, types, stamps, i, memo)?.hash(&mut h);
            false
        }
        Kind::Array(e, l) => {
            6u8.hash(&mut h);
            stable_ty_hash(outline, types, stamps, e, memo)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, l, memo)?.hash(&mut h);
            false
        }
        Kind::Slice(e) => {
            7u8.hash(&mut h);
            stable_ty_hash(outline, types, stamps, e, memo)?.hash(&mut h);
            false
        }
        Kind::Adt(i, s) => {
            8u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, i)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
            false
        }
        Kind::FnDef(i, s) => {
            9u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, i)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
            false
        }
        Kind::FnPtr(p, r) => {
            10u8.hash(&mut h);
            stable_substs_hash(outline, types, stamps, p, memo)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, r, memo)?.hash(&mut h);
            false
        }
        Kind::UnsafeFnPtr(p, r) => {
            11u8.hash(&mut h);
            stable_substs_hash(outline, types, stamps, p, memo)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, r, memo)?.hash(&mut h);
            false
        }
        Kind::Closure(p, s, at) => {
            12u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, p)?.hash(&mut h);
            at.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
            false
        }
        Kind::Dyn(o, s, m, a) => {
            13u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, o)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, s, memo)?.hash(&mut h);
            for &x in types.items(m).iter().chain(types.items(a)) {
                stable_entry_item_key(outline, stamps, x)?.hash(&mut h);
            }
            false
        }
        Kind::Pat(b, range) => {
            let (lo, hi) = types.pat(range);
            (14u8, lo, hi).hash(&mut h);
            stable_ty_hash(outline, types, stamps, b, memo)?.hash(&mut h);
            false
        }
        Kind::Proj(id) => {
            let d = types.proj(id);
            17u8.hash(&mut h);
            stable_entry_item_key(outline, stamps, d.of)?.hash(&mut h);
            stable_ty_hash(outline, types, stamps, d.self_ty, memo)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, d.args, memo)?.hash(&mut h);
            stable_entry_item_key(outline, stamps, d.assoc)?.hash(&mut h);
            stable_substs_hash(outline, types, stamps, d.own, memo)?.hash(&mut h);
            false
        }
        Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return None,
    };
    let val = h.finish();
    if !leaf {
        memo.insert(ty, val);
    }
    Some(val)
}

pub(crate) fn hash_entry_128(
    config_digest: u64,
    item_key: u64,
    substs_hash: u64,
    stamps_digest: u64,
    flags: u32,
) -> (u64, u64) {
    let mut hi = FxHasher::default();
    (config_digest, item_key, substs_hash, stamps_digest, flags).hash(&mut hi);
    let mut lo = FxHasher::default();
    (
        0x9E37_79B9_7F4A_7C15_u64,
        flags,
        stamps_digest,
        substs_hash,
        item_key,
        config_digest,
    )
        .hash(&mut lo);
    (hi.finish(), lo.finish())
}

pub(in crate::image::aot::cache) fn should_skip_aot_flush(
    aot_loaded: bool,
    hits: usize,
    misses: usize,
    aot_records: usize,
) -> bool {
    aot_loaded && hits > 0 && misses < 128 && misses.saturating_mul(16) < aot_records
}

pub(in crate::image::aot::cache) fn resolve_aot_fn(
    functions: &[Option<AotFn>],
    mut idx: usize,
) -> Option<(usize, bool, &std::sync::Arc<AotCode>, &[SymbolSource])> {
    for _ in 0..functions.len() {
        match functions.get(idx).and_then(Option::as_ref)? {
            AotFn::Alias(next) if *next < functions.len() => idx = *next,
            AotFn::Direct { code, relocs } => return Some((idx, false, code, relocs)),
            AotFn::Dict { body, relocs } => return Some((idx, true, body, relocs)),
            AotFn::Alias(_) => return None,
        }
    }
    None
}

pub(in crate::image::aot::cache) fn compute_bundle_key(
    outline: &Outline,
    reached: &Reached,
    roots: &[usize],
    config_digest: u64,
) -> u64 {
    let mut locals: Vec<(&str, u8, &Path)> = outline
        .crates
        .iter()
        .zip(&outline.roots)
        .filter(|(c, _)| !c.env.fixed)
        .map(|(_, r)| (r.name.as_str(), r.role as u8, r.path.as_path()))
        .collect();
    locals.sort_unstable();
    let mut root_tags: Vec<(&str, &Path, u32, u32)> = roots
        .iter()
        .take(2)
        .filter_map(|&idx| {
            let (item, _) = reached.list.get(idx)?.0;
            let r = outline.roots.get(item.krate as usize)?;
            Some((r.name.as_str(), r.path.as_path(), item.file, item.item))
        })
        .collect();
    root_tags.sort_unstable();
    let mut h = FxHasher::default();
    config_digest.hash(&mut h);
    locals.hash(&mut h);
    root_tags.hash(&mut h);
    h.finish()
}
