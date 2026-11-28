use super::{find_item_in, test_outline};
use crate::body::{Expr, ExprId};
use crate::names::Primitive;
use crate::sema::infer::{Adjust, BodyTables, LazyTables, Local, Res, SyntaxTables, Typed};
use crate::sema::substs::{PackedMap, Substs};
use crate::sema::ty::{Kind, Mut};
use crate::sema::{Cache, Context};
use std::sync::Arc;

#[test]
fn deref_shared_method_borrows_instead_of_moving_owned_pin_wrapper() {
    let code = concat!(
        "pub struct Pin<P> { pub pointer: P }\n",
        "impl<P> Pin<P> {\n",
        "    pub fn deref(&self) -> &P { loop {} }\n",
        "    pub fn get_mut(self) -> P { self.pointer }\n",
        "}\n",
        "pub struct SliceWrap { pub len: u8 }\n",
        "impl SliceWrap {\n",
        "    pub fn len(&self) -> usize { 0 }\n",
        "    pub fn split_at(&self, mid: usize) -> (&SliceWrap, &SliceWrap) { (self, self) }\n",
        "}\n",
        "pub fn poll_read(this: Pin<&mut &SliceWrap>) -> &SliceWrap {\n",
        "    let amt = this.len();\n",
        "    let (_a, b) = this.split_at(amt);\n",
        "    let _ = this.get_mut();\n",
        "    b\n",
        "}\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let (pin, slice_wrap) = (
        find_item_in(&outline, "Pin"),
        find_item_in(&outline, "SliceWrap"),
    );
    let (deref_fn, get_mut_fn, len_fn, split_fn, poll_fn) = (
        find_item_in(&outline, "deref"),
        find_item_in(&outline, "get_mut"),
        find_item_in(&outline, "len"),
        find_item_in(&outline, "split_at"),
        find_item_in(&outline, "poll_read"),
    );

    let mut cx = Context::new(&outline, &names, Cache::default());
    let usize_ty = cx.types.prim(Primitive::Usize);
    let unit = cx.types.unit();
    let wrap_ty = cx.types.adt(slice_wrap, &[]);
    let ref_wrap = cx.types.intern(Kind::Ref(Mut::Not, wrap_ty));
    let mut_ref_wrap = cx.types.intern(Kind::Ref(Mut::Mut, ref_wrap));
    let pin_ty = cx.types.adt(pin, &[mut_ref_wrap]);
    let split_ret = cx.types.tuple(&[ref_wrap, ref_wrap]);
    let substs_pin = cx.types.intern_substs(&[mut_ref_wrap]);
    let src = &outline.crates[0].files[0].source;

    let sig = cx.signature(poll_fn).expect("poll sig");
    let body = cx.body(poll_fn).expect("poll body");
    let mut binds = Vec::new();
    let mut next_local = 1u32;
    let mut amt_local = 0u32;
    let mut b_local = 0u32;
    for stmt in &body.stmts {
        if let crate::body::Stmt::Let { pat, .. } = *stmt {
            match *body.pat(pat) {
                crate::body::Pat::Tuple(list) => {
                    let ids = body.list(list);
                    binds.push((crate::body::PatId(ids[0]), next_local));
                    next_local += 1;
                    b_local = next_local;
                    binds.push((crate::body::PatId(ids[1]), next_local));
                    next_local += 1;
                }
                crate::body::Pat::Wild => {}
                _ => {
                    amt_local = next_local;
                    binds.push((pat, next_local));
                    next_local += 1;
                }
            }
        }
    }

    let mut exprs = vec![unit; body.exprs.len()];
    let mut res = Vec::new();
    let mut adjusts = Vec::new();
    let mut derefs = Vec::new();
    for (i, e) in body.exprs.iter().enumerate() {
        let id = ExprId(i as u32);
        match *e {
            Expr::Method { name, .. } => match src.text(name) {
                "len" => {
                    exprs[i] = usize_ty;
                    res.push((id, Res::Fn(len_fn, Substs::EMPTY)));
                    adjusts.push((
                        id,
                        Adjust {
                            derefs: 3,
                            autoref: None,
                        },
                    ));
                    derefs.push(((id, 0), Res::Fn(deref_fn, substs_pin)));
                }
                "split_at" => {
                    exprs[i] = split_ret;
                    res.push((id, Res::Fn(split_fn, Substs::EMPTY)));
                    adjusts.push((
                        id,
                        Adjust {
                            derefs: 3,
                            autoref: None,
                        },
                    ));
                    derefs.push(((id, 0), Res::Fn(deref_fn, substs_pin)));
                }
                "get_mut" => {
                    exprs[i] = mut_ref_wrap;
                    res.push((id, Res::Fn(get_mut_fn, substs_pin)));
                }
                _ => {}
            },
            Expr::Path(pid) => {
                let segs = body.list(body.path(pid).segments);
                match segs
                    .last()
                    .map(|&s| src.text(body.segments[s as usize].name))
                {
                    Some("this") => {
                        exprs[i] = pin_ty;
                        res.push((id, Res::Local(0)));
                    }
                    Some("amt") => {
                        exprs[i] = usize_ty;
                        res.push((id, Res::Local(amt_local)));
                    }
                    Some("b") => {
                        exprs[i] = ref_wrap;
                        res.push((id, Res::Local(b_local)));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let syntax = SyntaxTables {
        binds: PackedMap::pack(binds),
        adjusts: PackedMap::pack(adjusts),
        ..SyntaxTables::default()
    };
    let tables = BodyTables {
        syntax: Arc::new(syntax),
        exprs: exprs.into_boxed_slice(),
        res: PackedMap::pack(res),
        derefs: PackedMap::pack(derefs),
        ..BodyTables::default()
    };
    let mut locals = vec![Local { ty: unit }; next_local as usize];
    locals[0] = Local { ty: pin_ty };
    locals[amt_local as usize] = Local { ty: usize_ty };
    locals[(b_local - 1) as usize] = Local { ty: ref_wrap };
    locals[b_local as usize] = Local { ty: ref_wrap };
    let typed = Typed {
        tables: LazyTables::Eager(Arc::new(tables)),
        locals,
        params: vec![0],
        ret: Some(ref_wrap),
        ..Typed::default()
    };
    assert!(super::super::check(&mut cx, poll_fn, (&body, &sig), src, &typed).is_ok());
}

#[test]
fn self_typed_param_inherits_impl_named_lifetimes_for_return() {
    let code = concat!(
        "pub struct Guard<'a, T> { pub v: &'a T }\n",
        "pub struct Mapped<'a> { pub s: &'a u8 }\n",
        "pub struct Res<T, E> { pub ok: T, pub err: E }\n",
        "pub fn wrap_err<T, E>(e: E) -> Res<T, E> { loop {} }\n",
        "impl<'a, T> Guard<'a, T> {\n",
        "    pub fn try_map(mut this: Self) -> Res<Mapped<'a>, Self> { wrap_err(this) }\n",
        "    pub fn bad_map<'b>(this: Self) -> Guard<'b, T> { this }\n",
        "}\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let (guard, mapped, res_adt, wrap_err, try_map, bad_map) = (
        find_item_in(&outline, "Guard"),
        find_item_in(&outline, "Mapped"),
        find_item_in(&outline, "Res"),
        find_item_in(&outline, "wrap_err"),
        find_item_in(&outline, "try_map"),
        find_item_in(&outline, "bad_map"),
    );

    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let guard_ty = cx.types.adt(guard, &[u8_ty]);
    let mapped_ty = cx.types.adt(mapped, &[]);
    let ret_ty = cx.types.adt(res_adt, &[mapped_ty, guard_ty]);
    let substs_err = cx.types.intern_substs(&[mapped_ty, guard_ty]);
    let src = &outline.crates[0].files[0].source;

    let try_sig = cx.signature(try_map).expect("try sig");
    let try_body = cx.body(try_map).expect("try body");
    let mut try_exprs = vec![cx.types.unit(); try_body.exprs.len()];
    let mut try_res = Vec::new();
    for (i, e) in try_body.exprs.iter().enumerate() {
        let id = ExprId(i as u32);
        match *e {
            Expr::Call(..) => {
                try_exprs[i] = ret_ty;
                try_res.push((id, Res::Fn(wrap_err, substs_err)));
            }
            Expr::Path(pid) => {
                let segs = try_body.list(try_body.path(pid).segments);
                if segs
                    .last()
                    .is_some_and(|&s| src.text(try_body.segments[s as usize].name) == "this")
                {
                    try_exprs[i] = guard_ty;
                    try_res.push((id, Res::Local(0)));
                }
            }
            _ => {}
        }
    }
    let try_typed = Typed {
        tables: LazyTables::Eager(Arc::new(BodyTables {
            syntax: Arc::new(SyntaxTables {
                ..SyntaxTables::default()
            }),
            exprs: try_exprs.into_boxed_slice(),
            res: PackedMap::pack(try_res),
            ..BodyTables::default()
        })),
        locals: vec![Local { ty: guard_ty }],
        params: vec![0],
        ret: Some(ret_ty),
        ..Typed::default()
    };
    let info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        try_map,
        &try_sig,
        src,
        &try_typed,
        (1, &[]),
    );
    assert!(!info.has_elided_lt[0]);
    assert_eq!(info.named_lts[0], vec!["'a".to_owned()]);
    assert!(super::super::check(&mut cx, try_map, (&try_body, &try_sig), src, &try_typed).is_ok());

    let bad_sig = cx.signature(bad_map).expect("bad sig");
    let bad_body = cx.body(bad_map).expect("bad body");
    let mut bad_exprs = vec![cx.types.unit(); bad_body.exprs.len()];
    let mut bad_res = Vec::new();
    for (i, e) in bad_body.exprs.iter().enumerate() {
        let id = ExprId(i as u32);
        if let Expr::Path(pid) = *e {
            let segs = bad_body.list(bad_body.path(pid).segments);
            if segs
                .last()
                .is_some_and(|&s| src.text(bad_body.segments[s as usize].name) == "this")
            {
                bad_exprs[i] = guard_ty;
                bad_res.push((id, Res::Local(0)));
            }
        }
    }
    let bad_typed = Typed {
        tables: LazyTables::Eager(Arc::new(BodyTables {
            syntax: Arc::new(SyntaxTables {
                ..SyntaxTables::default()
            }),
            exprs: bad_exprs.into_boxed_slice(),
            res: PackedMap::pack(bad_res),
            ..BodyTables::default()
        })),
        locals: vec![Local { ty: guard_ty }],
        params: vec![0],
        ret: Some(guard_ty),
        ..Typed::default()
    };
    let bad_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        bad_map,
        &bad_sig,
        src,
        &bad_typed,
        (1, &[]),
    );
    assert!(!bad_info.has_elided_lt[0]);
    assert_eq!(bad_info.named_lts[0], vec!["'a".to_owned()]);
    assert!(super::super::check(&mut cx, bad_map, (&bad_body, &bad_sig), src, &bad_typed).is_err());
}

#[test]
fn mut_ref_deref_behind_mut_ref_reborrows_and_behind_shared_ref_rejects() {
    let code = concat!(
        "pub struct Proj<'a> { pub reader: &'a mut &'a mut u8 }\n",
        "pub fn wrap<P>(pointer: P) -> P { pointer }\n",
        "pub fn ok<'a>(me: &'a mut Proj<'a>) -> &'a mut u8 { wrap(*me.reader) }\n",
        "pub fn bad<'a>(me: &'a Proj<'a>) -> &'a mut u8 { wrap(*me.reader) }\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let (proj, wrap_fn, ok_fn, bad_fn) = (
        find_item_in(&outline, "Proj"),
        find_item_in(&outline, "wrap"),
        find_item_in(&outline, "ok"),
        find_item_in(&outline, "bad"),
    );

    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let mut_u8 = cx.types.intern(Kind::Ref(Mut::Mut, u8_ty));
    let mut_mut_u8 = cx.types.intern(Kind::Ref(Mut::Mut, mut_u8));
    let proj_ty = cx.types.adt(proj, &[]);
    let mut_proj = cx.types.intern(Kind::Ref(Mut::Mut, proj_ty));
    let ref_proj = cx.types.intern(Kind::Ref(Mut::Not, proj_ty));
    let substs_mut_u8 = cx.types.intern_substs(&[mut_u8]);
    let src = &outline.crates[0].files[0].source;

    let run_case = |cx: &mut Context<'_>, fn_item, me_ty| {
        let sig = cx.signature(fn_item).expect("sig");
        let body = cx.body(fn_item).expect("body");
        let mut exprs = vec![cx.types.unit(); body.exprs.len()];
        let mut res = Vec::new();
        for (i, e) in body.exprs.iter().enumerate() {
            let id = ExprId(i as u32);
            match *e {
                Expr::Call(callee, ..) => {
                    exprs[i] = mut_u8;
                    res.push((id, Res::Fn(wrap_fn, substs_mut_u8)));
                    res.push((callee, Res::Fn(wrap_fn, substs_mut_u8)));
                }
                Expr::Field(..) => exprs[i] = mut_mut_u8,
                Expr::Unary(..) => exprs[i] = mut_u8,
                Expr::Path(pid) => {
                    let segs = body.list(body.path(pid).segments);
                    if segs
                        .last()
                        .is_some_and(|&s| src.text(body.segments[s as usize].name) == "me")
                    {
                        exprs[i] = me_ty;
                        res.push((id, Res::Local(0)));
                    }
                }
                _ => {}
            }
        }
        let typed = Typed {
            tables: LazyTables::Eager(Arc::new(BodyTables {
                syntax: Arc::new(SyntaxTables {
                    ..SyntaxTables::default()
                }),
                exprs: exprs.into_boxed_slice(),
                res: PackedMap::pack(res),
                ..BodyTables::default()
            })),
            locals: vec![Local { ty: me_ty }],
            params: vec![0],
            ret: Some(mut_u8),
            ..Typed::default()
        };
        super::super::check(cx, fn_item, (&body, &sig), src, &typed)
    };

    run_case(&mut cx, ok_fn, mut_proj).expect("mut_ref behind mut_ref should reborrow");
    assert!(run_case(&mut cx, bad_fn, ref_proj).is_err());
}
