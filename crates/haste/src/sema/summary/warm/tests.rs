use std::path::PathBuf;
use std::sync::Arc;

use rustc_hash::FxHashMap as HashMap;

use crate::names::{ItemRef, ModRef, Primitive};
use crate::outline::source::Source;
use crate::outline::tree::{Crate, File};
use crate::outline::{Outline, Role, Root};
use crate::sema::impls::ImplRef;
use crate::sema::ty::Kind;
use crate::sema::{Cache, Context};

use super::*;

fn test_outline() -> (Outline, Names) {
    let src = Source::parse(PathBuf::from("/sys/core.rs"), "pub struct A;".to_owned())
        .expect("valid source");
    let items = crate::outline::scan::scan(&src).expect("valid scan");
    let file = Arc::new(File::new(src, items, false));
    let exp_src =
        Source::parse(PathBuf::from("/sys/core.rs"), "impl A {}".to_owned()).expect("exp");
    let exp_items = crate::outline::scan::scan(&exp_src).expect("exp scan");
    file.set_expansion(
        0,
        0,
        Some(Box::new(crate::outline::tree::Expansion {
            file: File::new(exp_src, exp_items, true),
            calls: HashMap::default(),
            scopes: Box::default(),
            item_scopes: Box::default(),
        })),
    );
    let mut krate = Crate::default();
    krate.env.fixed = true;
    krate.env.release = true;
    krate.files.push(file);
    krate.impls.push(crate::outline::tree::Impl {
        file: 0,
        module: 0,
        at: 0,
        parts: Vec::new(),
    });
    let outline = Outline {
        roots: vec![Root {
            unit: 0,
            name: "core".to_owned(),
            path: PathBuf::from("/sys/core.rs"),
            edition: "2021".to_owned(),
            role: Role::Lib,
            proc_macro: false,
        }],
        crates: vec![krate],
        deps: vec![Vec::new()],
        cache: Default::default(),
        procs: HashMap::default(),
        pins: vec![None],
    };
    (outline, Names::default())
}

#[test]
fn warm_blob_round_trips_and_rejects_mismatch_or_corruption() {
    let (outline, names) = test_outline();
    let store = Cache::default();
    let item = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let ty_a = store.types.intern(Kind::Adt(item, Substs::EMPTY));
    let ty_u8 = store.types.prim(Primitive::U8);
    let mut cx0 = Context::new(&outline, &names, store);
    let body_impl = ImplRef::Body(item.inner(0));
    let unready_body = ImplRef::Body(item.inner(99));
    let checked = cx0.store().lookups.checked_impls.clone();
    let _ = checked.run_cached(&mut cx0, body_impl, |_| Ok(true), |_| Ok(false));
    let _ = checked.run_cached(&mut cx0, unready_body, |_| Ok(true), |_| Ok(false));
    let _ = cx0.store().lookups.reveals.clone().run_cached(
        &mut cx0,
        (item, Substs::EMPTY),
        |_| Ok(ty_a),
        |_| Ok(Ty::NONE),
    );
    let store = cx0.release().0;
    store
        .lookups
        .provisional_rets
        .insert((item, Substs::EMPTY), ty_u8);
    store.lookups.selections.insert(
        (item, ty_a, Substs::EMPTY),
        Some((ImplRef::Module { krate: 0, index: 0 }, Substs::EMPTY)),
    );
    store.lookups.drops.insert(ty_a, true);
    store.lookups.futures.insert(ty_a, ty_u8);
    store.lookups.scope_traits.insert(
        ModRef {
            krate: 0,
            module: 0,
        },
        Arc::from([item]),
    );
    let mut lang_map = HashMap::default();
    lang_map.insert(item, "drop".to_owned().into_boxed_str());
    let _ = store
        .lookups
        .lang_items
        .set(Arc::new(crate::sema::lookups::LangMap::from_map(lang_map)));

    let key = WarmKey {
        toolchain_key: 11,
        deps_key: Some(22),
        target_key: 33,
        workspace_key: Some(44),
        release: false,
    };
    let pre = snapshot_extents(&outline);
    let raw_count = count_fixed_entries(
        &store,
        &WarmFilter {
            outline: &outline,
            pre: &pre,
        },
    );
    let blob = encode_warm_blob(key.fingerprint(), &outline, &names, &store, &pre, raw_count)
        .expect("warm blob encodes");

    let loaded = Cache::default();
    loaded.lookups.warm.set_bytes_for_test(key, blob.clone());
    let cx = Context::new(&outline, &names, loaded);
    let restored_ty = cx.types.intern(Kind::Adt(item, Substs::EMPTY));
    assert!(
        cx.selection(&(item, restored_ty, Substs::EMPTY))
            .flatten()
            .is_some()
    );
    assert_eq!(
        cx.store().lookups.checked_impls.get_ready(&body_impl),
        Some(true)
    );
    assert_eq!(
        cx.store().lookups.checked_impls.get_ready(&unready_body),
        None
    );
    assert_eq!(
        cx.store().lookups.reveals.get_ready(&(item, Substs::EMPTY)),
        Some(restored_ty)
    );
    assert_eq!(
        cx.store()
            .lookups
            .provisional_rets
            .get(&(item, Substs::EMPTY)),
        Some(ty_u8)
    );
    assert_eq!(cx.lang(item), Some("drop"));
    let post_count = count_fixed_entries(
        cx.store(),
        &WarmFilter {
            outline: &outline,
            pre: &pre,
        },
    );
    assert_eq!(post_count, raw_count - 1);

    let wrong_key = WarmKey {
        deps_key: Some(99),
        ..key
    };
    let mut missed = Cache::default();
    assert!(
        decode_warm_blob(
            &blob,
            wrong_key.fingerprint(),
            &outline,
            &names,
            &mut missed
        )
        .expect("valid header check")
        .is_none()
    );

    let mut corrupt = blob;
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xff;
    let mut bad = Cache::default();
    assert!(decode_warm_blob(&corrupt, key.fingerprint(), &outline, &names, &mut bad).is_err());
}
