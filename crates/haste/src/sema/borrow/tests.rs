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

fn test_outline(code: &str, edition: &str) -> (crate::outline::Outline, crate::names::Names) {
    use std::path::PathBuf;
    use std::sync::Arc;

    use rustc_hash::FxHashMap as HashMap;

    use crate::names::{Crate as NamesCrate, Names};
    use crate::outline::item::Vis;
    use crate::outline::scan::scan;
    use crate::outline::source::Source;
    use crate::outline::tree::{Crate, File, Module, State as TreeState};
    use crate::outline::{Outline, Role, Root};

    let src = Source::parse(PathBuf::from("lib.rs"), code.to_owned()).expect("parse source");
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
            state: TreeState::Loaded,
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
    (outline, names)
}

fn find_item_in(outline: &crate::outline::Outline, name: &str) -> crate::names::ItemRef {
    let file = &outline.crates[0].files[0];
    let idx = file
        .items
        .items
        .iter()
        .position(|it| it.name.is_some_and(|n| file.source.text(n) == name))
        .expect("item present");
    crate::names::ItemRef {
        krate: 0,
        file: 0,
        item: idx as u32,
    }
}

#[test]
fn deprecated_safe_2024_requires_unsafe_only_in_edition_2024() {
    use std::sync::Arc;

    use crate::body::{Expr, ExprId};
    use crate::names::ItemRef;
    use crate::sema::infer::{BodyTables, LazyTables, Res, Typed};
    use crate::sema::substs::{PackedMap, Substs};
    use crate::sema::{Cache, Context};

    let check_edition = |edition: &str| {
        let (outline, names) = test_outline(
            "#[rustc_deprecated_safe_2024(audit_that = \"test\")]\npub unsafe fn set_var() {}\npub fn caller() { set_var(); }\n",
            edition,
        );
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

#[test]
fn compute_param_lifetimes_resolves_named_type_param_and_assoc_item_bounds() {
    use std::sync::Arc;

    use crate::names::ItemRef;
    use crate::sema::infer::{LazyTables, Local, Typed};
    use crate::sema::{Cache, Context};

    let (outline, names) = test_outline(
        "pub fn f<'a, S, It>(input: Vec<(S, &'a str)>, it: &mut It, raw: S) where It: Iterator<Item = Vec<(&'a str, u8)>> {}\n",
        "2021",
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let item = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let sig = cx.signature(item).expect("sig");
    let unit = cx.types.unit();
    let locals = vec![Local { ty: unit }; 3];
    let typed = Typed {
        tables: LazyTables::Eager(Arc::default()),
        locals,
        params: vec![0, 1, 2],
        ..Typed::default()
    };
    let fn_src = &outline.crates[0].files[0].source;
    let info = super::calls::compute_param_lifetimes(&mut cx, item, &sig, fn_src, &typed, (3, &[]));
    assert!(!info.has_elided_lt[0]);
    assert_eq!(info.named_lts[0], vec!["'a".to_owned()]);
    assert!(!info.inner_has_elided_lt[1]);
    assert_eq!(info.inner_named_lts[1], vec!["'a".to_owned()]);
    assert!(info.has_elided_lt[2]);
    assert!(info.named_lts[2].is_empty());
}

#[test]
fn tuple_sources_keep_zipped_elements_isolated() {
    use super::propagate::TupleSources;

    let leaf = |target| TupleSources::Leaf {
        sources: vec![BorrowSource {
            target,
            steps: Vec::new(),
            mutable: false,
            through_ref: false,
            env_only: false,
            origin_only: false,
        }],
        inner_ref: None,
    };
    let zipped = TupleSources::Tuple(vec![
        TupleSources::Tuple(vec![leaf(10), leaf(20)]),
        leaf(30),
    ]);
    let TupleSources::Tuple(outer) = &zipped else {
        panic!("expected outer tuple");
    };
    let TupleSources::Tuple(inner) = &outer[0] else {
        panic!("expected inner tuple");
    };
    let mut style_srcs = Vec::new();
    inner[1].flatten_into(&mut style_srcs);
    assert_eq!(style_srcs.len(), 1);
    assert_eq!(style_srcs[0].target, 20);
}

#[test]
fn call_out_fresh_ref_peels_temp_borrow_while_keeping_inner_local_borrow() {
    use std::sync::Arc;

    use crate::body::{Expr, ExprId, Stmt};
    use crate::names::Primitive;
    use crate::sema::infer::{BodyTables, LazyTables, Local, Res, SyntaxTables, Typed};
    use crate::sema::substs::PackedMap;
    use crate::sema::ty::{Kind, Mut};
    use crate::sema::{Cache, Context};

    let code = concat!(
        "pub struct Sty { pub v: u8 }\n",
        "pub struct Opt<T> { pub v: T }\n",
        "pub struct Wrap<'a, S> { pub r: &'a u8, pub s: S }\n",
        "pub fn make_opt<T>(v: T) -> Opt<T> { Opt { v } }\n",
        "pub fn take<'a, S>(dst: &mut Wrap<'a, S>, x: &Opt<S>) {}\n",
        "pub fn ok_call<'a>(dst: &mut Wrap<'a, Sty>, s: Sty) { take(dst, &make_opt(s)); }\n",
        "pub fn err_call<'a>(dst: &mut Wrap<'a, &'a u8>) { let local = 0u8; take(dst, &make_opt(&local)); }\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let sty = find_item_in(&outline, "Sty");
    let opt = find_item_in(&outline, "Opt");
    let wrap = find_item_in(&outline, "Wrap");
    let make_opt = find_item_in(&outline, "make_opt");
    let take = find_item_in(&outline, "take");
    let ok_call = find_item_in(&outline, "ok_call");
    let err_call = find_item_in(&outline, "err_call");

    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let ref_u8 = cx.types.intern(Kind::Ref(Mut::Not, u8_ty));
    let sty_ty = cx.types.adt(sty, &[]);
    let opt_sty = cx.types.adt(opt, &[sty_ty]);
    let ref_opt_sty = cx.types.intern(Kind::Ref(Mut::Not, opt_sty));
    let wrap_sty = cx.types.adt(wrap, &[sty_ty]);
    let mut_wrap_sty = cx.types.intern(Kind::Ref(Mut::Mut, wrap_sty));
    let opt_ref_u8 = cx.types.adt(opt, &[ref_u8]);
    let ref_opt_ref_u8 = cx.types.intern(Kind::Ref(Mut::Not, opt_ref_u8));
    let wrap_ref_u8 = cx.types.adt(wrap, &[ref_u8]);
    let mut_wrap_ref_u8 = cx.types.intern(Kind::Ref(Mut::Mut, wrap_ref_u8));
    let unit = cx.types.unit();
    let substs_sty = cx.types.intern_substs(&[sty_ty]);
    let substs_ref_u8 = cx.types.intern_substs(&[ref_u8]);

    let src = &outline.crates[0].files[0].source;
    let path_name = |body: &crate::body::Body, path_id| {
        let segs = body.list(body.path(path_id).segments);
        segs.last()
            .map(|&s| src.text(body.segments[s as usize].name))
    };

    let ok_sig = cx.signature(ok_call).expect("ok sig");
    let ok_body = cx.body(ok_call).expect("ok body");
    let mut ok_exprs = vec![unit; ok_body.exprs.len()];
    let mut ok_res = Vec::new();
    for (i, e) in ok_body.exprs.iter().enumerate() {
        let id = ExprId(i as u32);
        match *e {
            Expr::Path(pid) => match path_name(&ok_body, pid) {
                Some("dst") => {
                    ok_exprs[i] = mut_wrap_sty;
                    ok_res.push((id, Res::Local(0)));
                }
                Some("s") => {
                    ok_exprs[i] = sty_ty;
                    ok_res.push((id, Res::Local(1)));
                }
                _ => {}
            },
            Expr::Ref { .. } => ok_exprs[i] = ref_opt_sty,
            Expr::Call(callee, _) => {
                if let Expr::Path(pid) = *ok_body.expr(callee) {
                    match path_name(&ok_body, pid) {
                        Some("make_opt") => {
                            ok_exprs[i] = opt_sty;
                            ok_res.push((id, Res::Fn(make_opt, substs_sty)));
                        }
                        Some("take") => ok_res.push((id, Res::Fn(take, substs_sty))),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let ok_syntax = SyntaxTables::default();
    let ok_tables = BodyTables {
        syntax: Arc::new(ok_syntax),
        exprs: ok_exprs.into_boxed_slice(),
        res: PackedMap::pack(ok_res),
        ..BodyTables::default()
    };
    let ok_typed = Typed {
        tables: LazyTables::Eager(Arc::new(ok_tables)),
        locals: vec![Local { ty: mut_wrap_sty }, Local { ty: sty_ty }],
        params: vec![0, 1],
        ret: Some(unit),
        ..Typed::default()
    };
    assert!(super::check(&mut cx, ok_call, (&ok_body, &ok_sig), src, &ok_typed).is_ok());

    let err_sig = cx.signature(err_call).expect("err sig");
    let err_body = cx.body(err_call).expect("err body");
    let local_pat = err_body
        .stmts
        .iter()
        .find_map(|s| match *s {
            Stmt::Let { pat, .. } => Some(pat),
            _ => None,
        })
        .expect("let local");
    let mut err_exprs = vec![unit; err_body.exprs.len()];
    let mut err_res = Vec::new();
    for (i, e) in err_body.exprs.iter().enumerate() {
        let id = ExprId(i as u32);
        match *e {
            Expr::Lit(..) => err_exprs[i] = u8_ty,
            Expr::Path(pid) => match path_name(&err_body, pid) {
                Some("dst") => {
                    err_exprs[i] = mut_wrap_ref_u8;
                    err_res.push((id, Res::Local(0)));
                }
                Some("local") => {
                    err_exprs[i] = u8_ty;
                    err_res.push((id, Res::Local(1)));
                }
                _ => {}
            },
            Expr::Ref { expr: inner, .. } => {
                err_exprs[i] = if matches!(*err_body.expr(inner), Expr::Path(..)) {
                    ref_u8
                } else {
                    ref_opt_ref_u8
                };
            }
            Expr::Call(callee, _) => {
                if let Expr::Path(pid) = *err_body.expr(callee) {
                    match path_name(&err_body, pid) {
                        Some("make_opt") => {
                            err_exprs[i] = opt_ref_u8;
                            err_res.push((id, Res::Fn(make_opt, substs_ref_u8)));
                        }
                        Some("take") => err_res.push((id, Res::Fn(take, substs_ref_u8))),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let err_syntax = SyntaxTables {
        binds: PackedMap::pack([(local_pat, 1)]),
        ..SyntaxTables::default()
    };
    let err_tables = BodyTables {
        syntax: Arc::new(err_syntax),
        exprs: err_exprs.into_boxed_slice(),
        res: PackedMap::pack(err_res),
        ..BodyTables::default()
    };
    let err_typed = Typed {
        tables: LazyTables::Eager(Arc::new(err_tables)),
        locals: vec![
            Local {
                ty: mut_wrap_ref_u8,
            },
            Local { ty: u8_ty },
        ],
        params: vec![0],
        ret: Some(unit),
        ..Typed::default()
    };
    assert!(super::check(&mut cx, err_call, (&err_body, &err_sig), src, &err_typed).is_err());
}

#[test]
fn value_receiver_borrow_free_copy_type_skips_container_loan() {
    use std::sync::Arc;

    use crate::body::{Expr, ExprId};
    use crate::names::Primitive;
    use crate::sema::infer::{Adjust, BodyTables, LazyTables, Local, Res, SyntaxTables, Typed};
    use crate::sema::substs::{PackedMap, Substs};
    use crate::sema::ty::{Kind, Mut};
    use crate::sema::{Cache, Context};

    let code = concat!(
        "pub struct Ansi<'a> { pub s: &'a u8 }\n",
        "pub trait Paint { fn paint<'a>(self, s: &'a u8) -> Ansi<'a>; }\n",
        "impl Paint for u8 { fn paint<'a>(self, s: &'a u8) -> Ansi<'a> { Ansi { s } } }\n",
        "pub fn render<'a>(styles: &[u8], x: &'a u8) -> Ansi<'a> { styles[0].paint(x) }\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let ansi = find_item_in(&outline, "Ansi");
    let paint = find_item_in(&outline, "paint");
    let render = find_item_in(&outline, "render");

    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let usize_ty = cx.types.prim(Primitive::Usize);
    let ref_u8 = cx.types.intern(Kind::Ref(Mut::Not, u8_ty));
    let slice_u8 = cx.types.intern(Kind::Slice(u8_ty));
    let ref_slice_style = cx.types.intern(Kind::Ref(Mut::Not, slice_u8));
    let ansi_ty = cx.types.adt(ansi, &[]);

    let src = &outline.crates[0].files[0].source;
    let sig = cx.signature(render).expect("render sig");
    let body = cx.body(render).expect("render body");
    let mut exprs = vec![cx.types.unit(); body.exprs.len()];
    let mut res = Vec::new();
    let mut adjusts = Vec::new();
    for (i, e) in body.exprs.iter().enumerate() {
        let id = ExprId(i as u32);
        match *e {
            Expr::Lit(..) => exprs[i] = usize_ty,
            Expr::Index(..) => exprs[i] = ref_u8,
            Expr::Method { .. } => {
                exprs[i] = ansi_ty;
                res.push((id, Res::Fn(paint, Substs::EMPTY)));
                adjusts.push((
                    id,
                    Adjust {
                        derefs: 1,
                        autoref: None,
                    },
                ));
            }
            Expr::Path(pid) => {
                let seg = *body.list(body.path(pid).segments).last().expect("path seg");
                match src.text(body.segments[seg as usize].name) {
                    "styles" => {
                        exprs[i] = ref_slice_style;
                        res.push((id, Res::Local(0)));
                    }
                    "x" => {
                        exprs[i] = ref_u8;
                        res.push((id, Res::Local(1)));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let syntax = SyntaxTables {
        adjusts: PackedMap::pack(adjusts),
        ..SyntaxTables::default()
    };
    let tables = BodyTables {
        syntax: Arc::new(syntax),
        exprs: exprs.into_boxed_slice(),
        res: PackedMap::pack(res),
        ..BodyTables::default()
    };
    let typed = Typed {
        tables: LazyTables::Eager(Arc::new(tables)),
        locals: vec![
            Local {
                ty: ref_slice_style,
            },
            Local { ty: ref_u8 },
        ],
        params: vec![0, 1],
        ret: Some(ansi_ty),
        ..Typed::default()
    };
    super::check(&mut cx, render, (&body, &sig), src, &typed).expect("render ok");
}

#[test]
fn unwrap_container_loans_strips_copied_vec_and_preserves_inner_local_borrow() {
    let mut state = State::new(vec![true, true, true, true]);
    let caller_line = 0;
    let local_vec = 1;
    let local_str = 2;
    let mixed_vec = 3;
    state.loans.push(Loan::new(
        caller_line,
        Vec::new(),
        Some(local_vec),
        false,
        true,
        None,
    ));
    state.loans.push(Loan::new(
        local_str,
        Vec::new(),
        Some(mixed_vec),
        false,
        false,
        None,
    ));

    let clean_srcs = vec![BorrowSource {
        target: local_vec,
        steps: vec![PlaceStep::Deref],
        mutable: false,
        through_ref: false,
        env_only: false,
        origin_only: false,
    }];
    let clean_out = unwrap_container_loans(clean_srcs, &state, |s| s.target == local_vec);
    assert_eq!(clean_out.len(), 1);
    assert_eq!(clean_out[0].target, caller_line);

    let dangling_srcs = vec![BorrowSource {
        target: mixed_vec,
        steps: vec![PlaceStep::Deref],
        mutable: false,
        through_ref: false,
        env_only: false,
        origin_only: false,
    }];
    let dangling_out = unwrap_container_loans(dangling_srcs, &state, |s| s.target == mixed_vec);
    assert_eq!(dangling_out.len(), 1);
    assert_eq!(dangling_out[0].target, local_str);
}
