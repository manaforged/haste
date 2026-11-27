use rustc_hash::FxHashSet as HashSet;

use super::places::PlaceStep;
use super::propagate::{BorrowSource, unwrap_container_loans};
use super::{Loan, State, prune_dead_loans};

#[test]
fn prune_dead_loans_drops_guard_only_binding_and_keeps_arm_live_binding() {
    let mut state = State::new(vec![true, true, true]);
    let f = 0;
    let guard_only = 1;
    let live_in_arm = 2;
    state.loans.push(Loan::new(
        f,
        Vec::new(),
        Some(guard_only),
        false,
        false,
        None,
    ));
    state.loans.push(Loan::new(
        f,
        Vec::new(),
        Some(live_in_arm),
        false,
        false,
        None,
    ));

    let guard_live = HashSet::from_iter([live_in_arm]);
    prune_dead_loans(&mut state, &guard_live);

    assert_eq!(state.loans.len(), 1);
    assert_eq!(state.loans[0].holder, Some(live_in_arm));

    let empty_live = HashSet::default();
    prune_dead_loans(&mut state, &empty_live);
    assert!(state.loans.is_empty());
}

#[test]
fn unwrap_container_loans_drops_temps_and_unwraps_intermediate_local_slices() {
    let mut state = State::new(vec![true, true, true]);
    let raw_line = 0;
    let local_vec = 1;
    let input_vec = 2;
    state.loans.push(Loan::new(
        raw_line,
        vec![PlaceStep::Deref],
        Some(local_vec),
        false,
        true,
        None,
    ));
    state.loans.push(Loan::new(
        local_vec,
        Vec::new(),
        Some(input_vec),
        false,
        false,
        None,
    ));

    let sources = vec![
        BorrowSource::temp(1, false, false),
        BorrowSource {
            target: local_vec,
            steps: Vec::new(),
            mutable: false,
            through_ref: false,
            env_only: false,
            origin_only: false,
        },
    ];
    let out = unwrap_container_loans(sources, &state, |s| {
        s.target == local_vec || super::checks::is_temp_target(s.target).is_some()
    });

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].target, raw_line);
    assert!(out[0].through_ref);
}

#[test]
fn deprecated_safe_2024_requires_unsafe_only_in_edition_2024() {
    use std::path::PathBuf;
    use std::sync::Arc;

    use rustc_hash::FxHashMap as HashMap;

    use crate::body::{Expr, ExprId};
    use crate::names::{Crate as NamesCrate, ItemRef, Names};
    use crate::outline::item::Vis;
    use crate::outline::scan::scan;
    use crate::outline::source::Source;
    use crate::outline::tree::{Crate, File, Module, State};
    use crate::outline::{Outline, Role, Root};
    use crate::sema::infer::{BodyTables, LazyTables, Res, Typed};
    use crate::sema::substs::{PackedMap, Substs};
    use crate::sema::{Cache, Context};

    let check_edition = |edition: &str| {
        let src = Source::parse(
            PathBuf::from("lib.rs"),
            "#[rustc_deprecated_safe_2024(audit_that = \"test\")]\npub unsafe fn set_var() {}\npub fn caller() { set_var(); }\n"
                .to_owned(),
        )
        .expect("parse source");
        let items = scan(&src).expect("scan source");
        let krate = Crate {
            files: vec![Arc::new(File::new(src, items, false))],
            modules: vec![Module {
                file: 0,
                item: None,
                parent: None,
                name: String::new(),
                dir: PathBuf::new(),
                parts: Vec::new(),
                vis: Vis::Pub,
                macro_use: false,
                state: State::Loaded,
                textual: None,
            }],
            ..Crate::default()
        };
        let outline = Outline {
            roots: vec![Root {
                unit: 0,
                name: "k".to_owned(),
                path: PathBuf::from("lib.rs"),
                edition: edition.to_owned(),
                role: Role::Lib,
                proc_macro: false,
            }],
            crates: vec![krate],
            deps: vec![Vec::new()],
            cache: Default::default(),
            procs: HashMap::default(),
            pins: vec![None],
        };
        let mut names = Names::default();
        let mut nc = NamesCrate::default();
        nc.edition = edition.to_owned();
        nc.edition_2015 = edition == "2015";
        names.crates.push(nc);
        let mut cx = Context::new(&outline, &names, Cache::default());
        let callee = ItemRef {
            krate: 0,
            file: 0,
            item: 0,
        };
        let caller = ItemRef {
            krate: 0,
            file: 0,
            item: 1,
        };
        let callee_sig = cx.signature(callee).expect("callee sig");
        assert!(callee_sig.is_unsafe);
        assert!(callee_sig.deprecated_safe_2024);
        let caller_sig = cx.signature(caller).expect("caller sig");
        let caller_body = cx.body(caller).expect("caller body");
        let call_expr = caller_body
            .exprs
            .iter()
            .enumerate()
            .find_map(|(i, e)| matches!(e, Expr::Call { .. }).then_some(ExprId(i as u32)))
            .expect("call expr");
        let tables = BodyTables {
            exprs: vec![cx.types.unit(); caller_body.exprs.len()].into_boxed_slice(),
            res: PackedMap::pack([(call_expr, Res::Fn(callee, Substs::EMPTY))]),
            ..BodyTables::default()
        };
        let typed = Typed {
            tables: LazyTables::Eager(Arc::new(tables)),
            ret: Some(cx.types.unit()),
            ..Typed::default()
        };
        let src = &outline.crates[caller.krate as usize].files[caller.file as usize].source;
        super::check(&mut cx, caller, (&caller_body, &caller_sig), src, &typed)
    };

    assert!(check_edition("2021").is_ok());
    assert!(check_edition("2024").is_err());
}
