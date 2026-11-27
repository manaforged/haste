//! Persistent disk cache for settled, unchanged local library crate outlines, macros, names, and impl indices.

mod meta;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::graph::store;
use crate::graph::{Targets, Workspace};
use crate::macros::Compilation;
use crate::names::{Crate as NamesCrate, Symbols};
use crate::outline::Outline;
use crate::sema::impls::Index as ImplIndex;

use meta::{
    LOCAL_INDEX_MAGIC, LOCAL_INDEX_VERSION, LocalMeta, collect_watched_files, compute_content_key,
    crate_ranks, crate_refs, decode_meta, encode_meta, local_lib_topo_order, structural_key,
    target_fingerprint, validate_env_reads, validate_watched,
};

use super::sysroot::{
    DecodeSpec, HEADER_LEN, PreparedWire, SnapshotSpec, apply_decoded_crates, collect_env_reads,
    decode_prepared_batch, decode_snapshot, encode_snapshot, prepare_wire,
};

const KEPT_LOCAL_INDEX: usize = 64;
pub(crate) const LOCAL_SCOPE_BIT: u32 = 1 << 29;

/// Cached local library symbols, resolved crate scopes, module impl indices, and hit tracking.
pub(crate) struct LoadedLocal {
    pub symbols: Option<Symbols>,
    pub names: Vec<(usize, NamesCrate)>,
    pub heads: HashMap<u32, Arc<ImplIndex>>,
    pub hits: Vec<bool>,
    pub keys: Vec<Option<(u64, u64)>>,
}

/// Validated, memory-mapped local library snapshot ready for batched decoding once `Symbols` is available.
pub(crate) struct PreparedLocal {
    pub krate: usize,
    pub structural_key: u64,
    pub content_key: u64,
    pub needs_sysroot: bool,
    pub needs_deps: bool,
    pub meta_path: PathBuf,
    pub bin_path: PathBuf,
    pub included: Vec<(PathBuf, PathBuf)>,
    pub wire: PreparedWire,
}

/// Returns whether the frozen local library crate snapshot cache is enabled.
pub(crate) fn enabled() -> bool {
    super::enabled() && !crate::env_flag("HASTE_NO_LOCAL_INDEX")
}

fn cache_dir() -> PathBuf {
    store::home().join("local-index")
}

fn bin_path(content_key: u64) -> PathBuf {
    cache_dir().join(format!("{content_key:016x}.bin"))
}

fn meta_path(structural_key: u64) -> PathBuf {
    cache_dir().join(format!("{structural_key:016x}.meta"))
}

fn is_ext(path: &Path, ext: &str) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some(ext)
}

fn local_scope_bit(slot: usize) -> u32 {
    LOCAL_SCOPE_BIT | (((slot as u32) & 0xff) << 21)
}

/// Validates file stamps and memory-maps settled local library snapshots without requiring `Symbols`.
pub(crate) fn prepare(
    workspace: &Workspace,
    targets: &Targets,
    outline: &Outline,
    sys_key: u64,
    parallel_stat: bool,
) -> Vec<PreparedLocal> {
    if !enabled() {
        return Vec::new();
    }
    let th = target_fingerprint(targets);
    let ranks = crate_ranks(workspace, outline);
    let n = outline.crates.len();
    let mut keys: Vec<Option<(u64, u64)>> = vec![None; n];
    let mut pinned_stamps: Vec<Option<store::FileStamp>> = vec![None; n];
    let mut prepared = Vec::new();
    let mut slot = 0usize;
    for krate in local_lib_topo_order(outline, &ranks) {
        let Some(refs) = crate_refs(outline, &ranks, krate) else {
            continue;
        };
        let Some(expected_deps) = refs
            .local_deps
            .iter()
            .map(|&d| keys[d])
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(sk) = structural_key(
            workspace,
            targets,
            outline,
            krate,
            &refs,
            sys_key,
            th,
            &mut pinned_stamps,
        ) else {
            continue;
        };
        let mpath = meta_path(sk);
        let Ok(meta_bytes) = std::fs::read(&mpath) else {
            continue;
        };
        let Some(meta) = decode_meta(&meta_bytes, sk) else {
            drop(std::fs::remove_file(&mpath));
            continue;
        };
        if meta.dep_keys != expected_deps
            || !validate_env_reads(&meta.env_reads)
            || !validate_watched(&meta.watched, parallel_stat)
            || compute_content_key(
                sk,
                &meta.watched,
                &meta.included,
                &meta.env_reads,
                &meta.dep_keys,
            ) != meta.content_key
        {
            continue;
        }
        let bpath = bin_path(meta.content_key);
        let Ok(backing) = super::Backing::open(&bpath, HEADER_LEN).map(Arc::new) else {
            drop(std::fs::remove_file(&mpath));
            continue;
        };
        let cached = [krate];
        let spec = DecodeSpec {
            magic: LOCAL_INDEX_MAGIC,
            version: LOCAL_INDEX_VERSION,
            key: meta.content_key,
            cached_crates: &cached,
            ref_crates: &refs.ref_crates,
            scope_bit: local_scope_bit(slot),
        };
        match prepare_wire(backing, &spec, workspace, outline) {
            Ok(Some(wire)) => {
                keys[krate] = Some((sk, meta.content_key));
                prepared.push(PreparedLocal {
                    krate,
                    structural_key: sk,
                    content_key: meta.content_key,
                    needs_sysroot: !refs.sysroot_deps.is_empty(),
                    needs_deps: !refs.pinned_deps.is_empty(),
                    meta_path: mpath,
                    bin_path: bpath,
                    included: meta.included,
                    wire,
                });
                slot += 1;
            }
            Ok(None) | Err(_) => {
                drop(std::fs::remove_file(&mpath));
                drop(std::fs::remove_file(&bpath));
            }
        }
    }
    prepared
}

/// Filters `prepared` in topological order against `sysroot_hit` and `deps_hit`, keeping only crates whose dependencies all hit.
pub(crate) fn filter_prepared(
    outline: &Outline,
    prepared: Vec<PreparedLocal>,
    sysroot_hit: bool,
    deps_hit: bool,
) -> Vec<PreparedLocal> {
    let mut ready = vec![false; outline.crates.len()];
    let mut out = Vec::with_capacity(prepared.len());
    for item in prepared {
        if (item.needs_sysroot && !sysroot_hit)
            || (item.needs_deps && !deps_hit)
            || outline.deps[item.krate]
                .iter()
                .any(|&(_, dep)| !outline.is_fixed(dep as u32) && !ready[dep])
        {
            continue;
        }
        ready[item.krate] = true;
        out.push(item);
    }
    out
}

/// Attempts to load cached settled local library crates into `outline`, interning symbols into `existing_symbols` if present.
pub(crate) fn try_load(
    workspace: &Workspace,
    targets: &Targets,
    outline: &mut Outline,
    existing_symbols: Option<&Symbols>,
    sys_key: u64,
    sysroot_hit: bool,
    deps_hit: bool,
) -> LoadedLocal {
    let n = outline.crates.len();
    let mut loaded = LoadedLocal {
        symbols: None,
        names: Vec::new(),
        heads: HashMap::default(),
        hits: vec![false; n],
        keys: vec![None; n],
    };
    let prepared = filter_prepared(
        outline,
        prepare(workspace, targets, outline, sys_key, true),
        sysroot_hit,
        deps_hit,
    );
    if prepared.is_empty() {
        return loaded;
    }
    let mut iter = prepared.into_iter();
    if existing_symbols.is_none()
        && let Some(first) = iter.next()
    {
        let ranks = crate_ranks(workspace, outline);
        if let Some(refs) = crate_refs(outline, &ranks, first.krate) {
            let cached = [first.krate];
            let spec = DecodeSpec {
                magic: LOCAL_INDEX_MAGIC,
                version: LOCAL_INDEX_VERSION,
                key: first.content_key,
                cached_crates: &cached,
                ref_crates: &refs.ref_crates,
                scope_bit: first.wire.scope_bit,
            };
            match decode_snapshot(&first.wire.backing, &spec, None, workspace, outline) {
                Ok(Some((crates, owned_syms, names, heads, env_reads))) => {
                    loaded.symbols = owned_syms;
                    apply_decoded_crates(outline, crates, env_reads);
                    outline.crates[first.krate].cached_included = Some(first.included);
                    loaded.names.extend(names);
                    loaded.heads.extend(heads);
                    loaded.hits[first.krate] = true;
                    loaded.keys[first.krate] = Some((first.structural_key, first.content_key));
                }
                Ok(None) | Err(_) => {
                    drop(std::fs::remove_file(&first.meta_path));
                    drop(std::fs::remove_file(&first.bin_path));
                    return loaded;
                }
            }
        }
    }
    let rest: Vec<PreparedLocal> = iter.collect();
    if rest.is_empty() {
        return loaded;
    }
    let Some(syms) = existing_symbols.or(loaded.symbols.as_ref()) else {
        return loaded;
    };
    let mut meta_info = Vec::with_capacity(rest.len());
    let mut wires = Vec::with_capacity(rest.len());
    for p in rest {
        meta_info.push((
            p.krate,
            p.structural_key,
            p.content_key,
            p.meta_path,
            p.bin_path,
            p.included,
        ));
        wires.push(p.wire);
    }
    match decode_prepared_batch(wires, syms) {
        Ok(decoded) => {
            for ((krate, sk, ck, _, _, included), (crates, _, names, heads, env_reads)) in
                meta_info.into_iter().zip(decoded)
            {
                apply_decoded_crates(outline, crates, env_reads);
                outline.crates[krate].cached_included = Some(included);
                loaded.names.extend(names);
                loaded.heads.extend(heads);
                loaded.hits[krate] = true;
                loaded.keys[krate] = Some((sk, ck));
            }
        }
        Err(_) => {
            for (_, _, _, mpath, bpath, _) in meta_info {
                drop(std::fs::remove_file(&mpath));
                drop(std::fs::remove_file(&bpath));
            }
        }
    }
    loaded
}

/// Atomically persists settled local library crates from `compilation` to `$HASTE_HOME/local-index/` and returns their impl indices.
pub(crate) fn save(
    workspace: &Workspace,
    targets: &Targets,
    compilation: &Compilation,
    sys_key: u64,
    hits: &[bool],
    hit_keys: &[Option<(u64, u64)>],
) -> HashMap<u32, Arc<ImplIndex>> {
    let mut all_heads = HashMap::default();
    if !enabled() {
        return all_heads;
    }
    let outline = &compilation.outline;
    let th = target_fingerprint(targets);
    let ranks = crate_ranks(workspace, outline);
    let mut keys: Vec<Option<(u64, u64)>> = hit_keys.to_vec();
    keys.resize(outline.crates.len(), None);
    let mut pinned_stamps: Vec<Option<store::FileStamp>> = vec![None; outline.crates.len()];
    let (mut last_bin, mut last_meta) = (None, None);

    for krate in local_lib_topo_order(outline, &ranks) {
        if hits.get(krate).copied().unwrap_or(false) {
            continue;
        }
        let tree = &outline.crates[krate];
        if tree.partial()
            || tree.unsettled()
            || !tree.errors.is_empty()
            || !compilation.names.crates.get(krate).is_some_and(|c| c.eager)
        {
            continue;
        }
        let Some(refs) = crate_refs(outline, &ranks, krate) else {
            continue;
        };
        let deps_settled = refs
            .sysroot_deps
            .iter()
            .chain(&refs.pinned_deps)
            .all(|&d| !outline.crates[d].partial() && !outline.crates[d].unsettled());
        let Some(dep_keys) = deps_settled
            .then(|| {
                refs.local_deps
                    .iter()
                    .map(|&d| keys[d])
                    .collect::<Option<Vec<_>>>()
            })
            .flatten()
        else {
            continue;
        };
        let Some(sk) = structural_key(
            workspace,
            targets,
            outline,
            krate,
            &refs,
            sys_key,
            th,
            &mut pinned_stamps,
        ) else {
            continue;
        };
        let (watched, included) = collect_watched_files(outline, krate);
        let env_reads_vec = collect_env_reads(outline, &[krate]);
        let content_key = compute_content_key(
            sk,
            &watched,
            &included,
            env_reads_vec.first().map_or(&[][..], Vec::as_slice),
            &dep_keys,
        );
        let cached = [krate];
        let spec = SnapshotSpec {
            magic: LOCAL_INDEX_MAGIC,
            version: LOCAL_INDEX_VERSION,
            key: content_key,
            cached_crates: &cached,
            ref_crates: &refs.ref_crates,
            env_reads: &env_reads_vec,
        };
        let Some((bytes, heads)) = encode_snapshot(&spec, workspace, compilation) else {
            continue;
        };
        all_heads.extend(heads);
        let bpath = bin_path(content_key);
        if let Err(error) = super::backing::write_raw(&bpath, &bytes) {
            eprintln!("haste: cannot save local index cache: {error:#}");
            continue;
        }
        let meta = LocalMeta {
            content_key,
            dep_keys,
            watched,
            included,
            env_reads: env_reads_vec.into_iter().next().unwrap_or_default(),
        };
        let Some(meta_bytes) = encode_meta(sk, &meta) else {
            continue;
        };
        let mpath = meta_path(sk);
        if let Err(error) = super::backing::write_raw(&mpath, &meta_bytes) {
            eprintln!("haste: cannot save local index meta: {error:#}");
            continue;
        }
        keys[krate] = Some((sk, content_key));
        last_bin = Some(bpath);
        last_meta = Some(mpath);
    }

    let dir = cache_dir();
    if let Some(keep) = last_bin {
        super::prune_stale(&dir, &keep, KEPT_LOCAL_INDEX, |p| is_ext(p, "bin"));
    }
    if let Some(keep) = last_meta {
        super::prune_stale(&dir, &keep, KEPT_LOCAL_INDEX, |p| is_ext(p, "meta"));
    }
    all_heads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::Role;
    use crate::sema::summary::codec::Cursor;

    fn temp_dir(tag: &str) -> PathBuf {
        let seq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "haste-local-idx-{tag}-{}-{seq}",
            std::process::id()
        ));
        drop(std::fs::remove_dir_all(&dir));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir.canonicalize().unwrap_or(dir)
    }

    fn with_local_env<R>(home: &Path, disable_local: bool, f: impl FnOnce() -> R) -> R {
        let _guard = crate::sync::lock(&crate::substrate::TEST_HOME_LOCK);
        let prev_home = std::env::var_os("HASTE_HOME");
        let prev_no_local = std::env::var_os("HASTE_NO_LOCAL_INDEX");
        // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
        unsafe {
            std::env::set_var("HASTE_HOME", home);
            if disable_local {
                std::env::set_var("HASTE_NO_LOCAL_INDEX", "1");
            } else {
                std::env::remove_var("HASTE_NO_LOCAL_INDEX");
            }
        }
        let out = f();
        // SAFETY: serialized by `TEST_HOME_LOCK` within unit tests.
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HASTE_HOME", v),
                None => std::env::remove_var("HASTE_HOME"),
            }
            match prev_no_local {
                Some(v) => std::env::set_var("HASTE_NO_LOCAL_INDEX", v),
                None => std::env::remove_var("HASTE_NO_LOCAL_INDEX"),
            }
        }
        out
    }

    fn test_targets() -> Targets {
        let host = crate::graph::cfg::Target::of(&crate::host_target());
        Targets {
            target: host.clone(),
            host,
        }
    }

    fn write_test_workspace(root: &Path) {
        std::fs::create_dir_all(root.join("crates/lib_a/src")).expect("mkdir lib_a");
        std::fs::create_dir_all(root.join("crates/app/src")).expect("mkdir app");
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/lib_a\", \"crates/app\"]\nresolver = \"2\"\n",
        )
        .expect("write Cargo.toml");
        std::fs::write(
            root.join("Cargo.lock"),
            "version = 4\n\n[[package]]\nname = \"lib_a\"\nversion = \"0.1.0\"\n\n[[package]]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = [\"lib_a\"]\n",
        )
        .expect("write Cargo.lock");
        std::fs::write(
            root.join("crates/lib_a/Cargo.toml"),
            "[package]\nname = \"lib_a\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("write lib_a Cargo.toml");
        std::fs::write(
            root.join("crates/lib_a/src/lib.rs"),
            "pub mod extra;\npub const MSG: &str = include_str!(\"msg.txt\");\n#[macro_export]\nmacro_rules! make_val {\n    () => { { let local_var = 41usize; local_var + 1 } };\n}\nmacro_rules! def_helper {\n    () => { pub fn helper() -> usize { let v = 41usize; v + 1 } };\n}\ndef_helper!();\npub fn base_val() -> usize { make_val!() + helper() + extra::ext() }\n",
        )
        .expect("write lib.rs");
        std::fs::write(
            root.join("crates/lib_a/src/extra.rs"),
            "pub fn ext() -> usize { 7 }\n",
        )
        .expect("write extra.rs");
        std::fs::write(root.join("crates/lib_a/src/msg.txt"), "hello").expect("write msg.txt");
        std::fs::write(
            root.join("crates/app/Cargo.toml"),
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n[dependencies]\nlib_a = { path = \"../lib_a\" }\n",
        )
        .expect("write app Cargo.toml");
        std::fs::write(
            root.join("crates/app/src/main.rs"),
            "fn main() { let _ = lib_a::base_val(); }\n",
        )
        .expect("write main.rs");
    }

    fn build_comp(ws: &Workspace, targets: &Targets) -> Compilation {
        let mut outline = crate::outline::new(ws, targets, &[]);
        let selected = vec![true; outline.crates.len()];
        outline.load_selected(&selected).expect("load_selected");
        let mut names = crate::names::Names::new_with(ws, &outline, None);
        for k in 0..selected.len() {
            names.crates[k].set_eager();
        }
        names.resolve(&mut outline).expect("resolve");
        Compilation::for_test(outline, names)
    }

    #[test]
    fn meta_round_trip_and_key_mismatch() {
        let meta = LocalMeta {
            content_key: 0x1234_5678_9abc_def0,
            dep_keys: vec![(11, 22)],
            watched: vec![
                (PathBuf::from("/tmp/a.rs"), Some((10, 20))),
                (PathBuf::from("/tmp/missing.rs"), None),
            ],
            included: vec![(PathBuf::from("/tmp/msg.txt"), PathBuf::from("/tmp/a.rs"))],
            env_reads: vec![("K".to_owned(), Some("V".to_owned()))],
        };
        let bytes = encode_meta(0xdead_beef, &meta).expect("encode meta");
        assert!(decode_meta(&bytes, 0xcafe_babe).is_none());
        let decoded = decode_meta(&bytes, 0xdead_beef).expect("decode meta");
        assert_eq!(decoded.content_key, meta.content_key);
        assert_eq!(decoded.dep_keys, meta.dep_keys);
        assert_eq!(decoded.watched, meta.watched);
        assert_eq!(decoded.included, meta.included);
        assert_eq!(decoded.env_reads, meta.env_reads);
    }

    #[test]
    fn scope_bit_isolation_preserves_sysroot_and_deps_and_tags_local() {
        use crate::outline::source::Mark;
        let mut out = Vec::new();
        let marks = vec![
            (0, Mark::Scope { id: 1, parent: 0 }),
            (
                1,
                Mark::Scope {
                    id: 2,
                    parent: (1 << 31) | 7,
                },
            ),
            (
                2,
                Mark::Scope {
                    id: 3,
                    parent: (1 << 30) | 9,
                },
            ),
            (
                3,
                Mark::Scope {
                    id: 4,
                    parent: LOCAL_SCOPE_BIT | 11,
                },
            ),
        ];
        super::super::sysroot::tokens::write_marks(&mut out, &marks, &[Some(0)]).expect("write");
        let mut cur = Cursor::new(&out);
        let decoded = super::super::sysroot::tokens::read_marks(&mut cur, &[0], LOCAL_SCOPE_BIT)
            .expect("read");
        assert_eq!(
            decoded[0].1,
            Mark::Scope {
                id: LOCAL_SCOPE_BIT | 1,
                parent: 0,
            }
        );
        assert_eq!(
            decoded[1].1,
            Mark::Scope {
                id: LOCAL_SCOPE_BIT | 2,
                parent: (1 << 31) | 7,
            }
        );
        assert_eq!(
            decoded[2].1,
            Mark::Scope {
                id: LOCAL_SCOPE_BIT | 3,
                parent: (1 << 30) | 9,
            }
        );
        assert_eq!(
            decoded[3].1,
            Mark::Scope {
                id: LOCAL_SCOPE_BIT | 4,
                parent: LOCAL_SCOPE_BIT | 11,
            }
        );
        assert_eq!(LOCAL_SCOPE_BIT & ((1 << 31) | (1 << 30)), 0);
    }

    #[test]
    fn file_stamp_invalidation_and_main_edit_cache_hit() {
        let root = temp_dir("ws");
        let home = temp_dir("home");
        write_test_workspace(&root);

        with_local_env(&home, false, || {
            let targets = test_targets();
            let ws = crate::graph::load(&root, &targets, &Default::default()).expect("load ws");
            let sys_key = super::super::sysroot_key(&ws, &targets);
            let comp1 = build_comp(&ws, &targets);
            let lib_krate = comp1
                .outline
                .roots
                .iter()
                .position(|r| r.name == "lib_a" && r.role == Role::Lib)
                .expect("lib_a");
            let empty_hits = vec![false; comp1.outline.crates.len()];
            let empty_keys = vec![None; comp1.outline.crates.len()];
            let heads = save(&ws, &targets, &comp1, sys_key, &empty_hits, &empty_keys);
            assert!(heads.contains_key(&(lib_krate as u32)));

            std::fs::write(
                root.join("crates/app/src/main.rs"),
                "fn main() { let x = lib_a::base_val(); let _ = x + 1; }\n",
            )
            .expect("edit main.rs");

            let mut outline2 = crate::outline::new(&ws, &targets, &[]);
            let loaded2 = try_load(&ws, &targets, &mut outline2, None, sys_key, true, true);
            assert!(loaded2.hits[lib_krate]);
            assert!(!outline2.crates[lib_krate].partial());
            let expected_included = vec![(
                root.join("crates/lib_a/src/msg.txt"),
                root.join("crates/lib_a/src/lib.rs"),
            )];
            assert_eq!(
                outline2.crates[lib_krate].cached_included.as_deref(),
                Some(expected_included.as_slice())
            );
            assert_eq!(outline2.included(lib_krate), expected_included);
            let has_local_scope = outline2.crates[lib_krate].files.iter().any(|f| {
                f.source.marks.iter().any(|&(_, m)| {
                    matches!(
                        m,
                        crate::outline::source::Mark::Scope { id, .. }
                        if (id & LOCAL_SCOPE_BIT) != 0 && (id & ((1 << 31) | (1 << 30))) == 0
                    )
                })
            });
            assert!(has_local_scope);

            std::fs::write(
                root.join("crates/lib_a/src/extra.rs"),
                "pub fn ext() -> usize { 99 }\n",
            )
            .expect("edit extra.rs");
            let mut outline3 = crate::outline::new(&ws, &targets, &[]);
            assert!(
                !try_load(&ws, &targets, &mut outline3, None, sys_key, true, true).hits[lib_krate]
            );

            std::fs::write(
                root.join("crates/lib_a/src/extra.rs"),
                "pub fn ext() -> usize { 7 }\n",
            )
            .expect("restore extra.rs");
            let comp4 = build_comp(&ws, &targets);
            save(&ws, &targets, &comp4, sys_key, &empty_hits, &empty_keys);

            std::fs::write(root.join("crates/lib_a/src/msg.txt"), "changed include")
                .expect("edit msg.txt");
            let mut outline5 = crate::outline::new(&ws, &targets, &[]);
            assert!(
                !try_load(&ws, &targets, &mut outline5, None, sys_key, true, true).hits[lib_krate]
            );
        });

        with_local_env(&home, true, || {
            assert!(!enabled());
        });

        drop(std::fs::remove_dir_all(&root));
        drop(std::fs::remove_dir_all(&home));
    }
}
