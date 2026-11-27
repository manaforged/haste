use std::sync::Arc;

use super::{find_item_in, test_outline};
use crate::body::{Body, Expr, ExprId, PathId, Stmt};
use crate::names::{ItemRef, Primitive};
use crate::outline::source::Source;
use crate::sema::infer::{
    BodyTables, By, Capture, Closure, LazyTables, Local, Res, SyntaxTables, Typed,
};
use crate::sema::substs::{PackedMap, Substs};
use crate::sema::ty::{Kind, Mut, Ty};
use crate::sema::{Cache, Context};

fn check_case(
    cx: &mut Context<'_>,
    fn_item: ItemRef,
    build: impl FnOnce(&mut Context<'_>, &Body, &Source) -> Typed,
) -> anyhow::Result<()> {
    let sig = cx.signature(fn_item).expect("sig");
    let body = cx.body(fn_item).expect("body");
    let src = &cx.file(fn_item).source;
    let typed = build(cx, &body, src);
    super::super::check(cx, fn_item, (&body, &sig), src, &typed)
}

fn path_last<'a>(body: &Body, src: &'a Source, pid: PathId) -> Option<&'a str> {
    body.list(body.path(pid).segments)
        .last()
        .map(|&s| src.text(body.segments[s as usize].name))
}

fn build_typed(
    exprs: Vec<Ty>,
    res: Vec<(ExprId, Res)>,
    binds: Vec<(crate::body::PatId, u32)>,
    closures: Vec<(ExprId, Closure)>,
    locals: Vec<Local>,
    params: Vec<u32>,
    ret: Option<Ty>,
) -> Typed {
    Typed {
        tables: LazyTables::Eager(Arc::new(BodyTables {
            syntax: Arc::new(SyntaxTables {
                binds: PackedMap::pack(binds),
                ..SyntaxTables::default()
            }),
            exprs: exprs.into_boxed_slice(),
            res: PackedMap::pack(res),
            closures: PackedMap::pack(closures),
            ..BodyTables::default()
        })),
        locals,
        params,
        ret,
        ..Typed::default()
    }
}

#[test]
fn ctor_shared_reborrow_and_tuple_projection_cases() {
    let code = concat!(
        "pub struct ReadHalf<'a>(pub &'a u8);\n",
        "pub struct WriteHalf<'a>(pub &'a u8);\n",
        "pub struct MutHalf<'a>(pub &'a mut u8);\n",
        "pub fn ok_split(stream: &mut u8) -> (ReadHalf<'_>, WriteHalf<'_>) {\n",
        "    (ReadHalf(stream), WriteHalf(stream))\n",
        "}\n",
        "pub fn bad_split(stream: &mut u8) -> (MutHalf<'_>, WriteHalf<'_>) {\n",
        "    (MutHalf(stream), WriteHalf(stream))\n",
        "}\n",
        "pub struct Holder<'a> { pub inner: &'a mut u8 }\n",
        "impl<'a> Holder<'a> {\n",
        "    pub fn ok_ref(&self) -> &u8 { self.inner }\n",
        "    pub fn bad_mut(&self) -> &mut u8 { self.inner }\n",
        "}\n",
        "pub struct Pair<'a> { pub rx: &'a mut u8, pub w: u8 }\n",
        "pub fn ok_pair(me: Pair<'_>) -> (&mut u8, &u8) { (me.rx, &me.w) }\n",
        "pub fn bad_pair(me: Pair<'_>) -> (&mut u8, &mut u8) { (me.rx, me.rx) }\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let (read_h, write_h, mut_h, holder, pair) = (
        find_item_in(&outline, "ReadHalf"),
        find_item_in(&outline, "WriteHalf"),
        find_item_in(&outline, "MutHalf"),
        find_item_in(&outline, "Holder"),
        find_item_in(&outline, "Pair"),
    );
    let (ok_split, bad_split, ok_ref, bad_mut, ok_pair, bad_pair) = (
        find_item_in(&outline, "ok_split"),
        find_item_in(&outline, "bad_split"),
        find_item_in(&outline, "ok_ref"),
        find_item_in(&outline, "bad_mut"),
        find_item_in(&outline, "ok_pair"),
        find_item_in(&outline, "bad_pair"),
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let ref_u8 = cx.types.intern(Kind::Ref(Mut::Not, u8_ty));
    let mut_u8 = cx.types.intern(Kind::Ref(Mut::Mut, u8_ty));
    let (read_ty, write_ty, mut_ty) = (
        cx.types.adt(read_h, &[]),
        cx.types.adt(write_h, &[]),
        cx.types.adt(mut_h, &[]),
    );
    let ok_split_ret = cx.types.tuple(&[read_ty, write_ty]);
    let bad_split_ret = cx.types.tuple(&[mut_ty, write_ty]);
    let ref_holder = cx
        .types
        .intern(Kind::Ref(Mut::Not, cx.types.adt(holder, &[])));
    let pair_ty = cx
        .types
        .intern(Kind::Ref(Mut::Mut, cx.types.adt(pair, &[])));
    let ok_pair_ret = cx.types.tuple(&[mut_u8, ref_u8]);
    let bad_pair_ret = cx.types.tuple(&[mut_u8, mut_u8]);

    let run_split = |cx: &mut Context<'_>, f, ret_ty| {
        check_case(cx, f, |_, body, src| {
            let mut exprs = vec![ret_ty; body.exprs.len()];
            let mut res = Vec::new();
            for (i, e) in body.exprs.iter().enumerate() {
                let id = ExprId(i as u32);
                match *e {
                    Expr::Tuple(..) => exprs[i] = ret_ty,
                    Expr::Call(callee, ..) => {
                        if let Expr::Path(pid) = *body.expr(callee) {
                            let (ty, ctor) = match path_last(body, src, pid) {
                                Some("ReadHalf") => (read_ty, read_h),
                                Some("WriteHalf") => (write_ty, write_h),
                                _ => (mut_ty, mut_h),
                            };
                            exprs[i] = ty;
                            res.push((id, Res::Ctor(ctor, 0, Substs::EMPTY)));
                            res.push((callee, Res::Ctor(ctor, 0, Substs::EMPTY)));
                        }
                    }
                    Expr::Path(..) => {
                        exprs[i] = mut_u8;
                        res.push((id, Res::Local(0)));
                    }
                    _ => {}
                }
            }
            build_typed(
                exprs,
                res,
                Vec::new(),
                Vec::new(),
                vec![Local { ty: mut_u8 }],
                vec![0],
                Some(ret_ty),
            )
        })
    };
    run_split(&mut cx, ok_split, ok_split_ret).expect("shared ctor tuple split");
    assert!(run_split(&mut cx, bad_split, bad_split_ret).is_err());

    let run_holder = |cx: &mut Context<'_>, f, ret_ty| {
        check_case(cx, f, |_, body, _| {
            let mut exprs = vec![u8_ty; body.exprs.len()];
            let mut res = Vec::new();
            for (i, e) in body.exprs.iter().enumerate() {
                match *e {
                    Expr::Field(..) => exprs[i] = mut_u8,
                    Expr::Path(..) => {
                        exprs[i] = ref_holder;
                        res.push((ExprId(i as u32), Res::Local(0)));
                    }
                    _ => {}
                }
            }
            build_typed(
                exprs,
                res,
                Vec::new(),
                Vec::new(),
                vec![Local { ty: ref_holder }],
                vec![0],
                Some(ret_ty),
            )
        })
    };
    run_holder(&mut cx, ok_ref, ref_u8).expect("shared reborrow return out of &self");
    assert!(run_holder(&mut cx, bad_mut, mut_u8).is_err());

    let run_pair = |cx: &mut Context<'_>, f, ret_ty| {
        check_case(cx, f, |_, body, src| {
            let mut exprs = vec![ret_ty; body.exprs.len()];
            let mut res = Vec::new();
            for (i, e) in body.exprs.iter().enumerate() {
                match *e {
                    Expr::Tuple(..) => exprs[i] = ret_ty,
                    Expr::Ref { .. } => exprs[i] = ref_u8,
                    Expr::Field(_, name) => {
                        exprs[i] = if src.text(name) == "rx" {
                            mut_u8
                        } else {
                            u8_ty
                        };
                    }
                    Expr::Path(..) => {
                        exprs[i] = pair_ty;
                        res.push((ExprId(i as u32), Res::Local(0)));
                    }
                    _ => {}
                }
            }
            build_typed(
                exprs,
                res,
                Vec::new(),
                Vec::new(),
                vec![Local { ty: pair_ty }],
                vec![0],
                Some(ret_ty),
            )
        })
    };
    run_pair(&mut cx, ok_pair, ok_pair_ret).expect("disjoint mut and shared field tuple");
    assert!(run_pair(&mut cx, bad_pair, bad_pair_ret).is_err());
}

#[test]
fn owned_proj_deref_and_underscore_place_cases() {
    let code = concat!(
        "pub struct BufProj<'a> { pub buf: &'a mut [u8; 4] }\n",
        "pub fn make_empty<'a>() -> BufProj<'a> { loop {} }\n",
        "pub fn make_proj<'a>(x: &'a mut [u8; 4]) -> BufProj<'a> { loop {} }\n",
        "pub fn ok_deref<'a>(_dummy: &'a u8) -> &'a u8 {\n",
        "    let me = make_empty();\n",
        "    &me.buf[0]\n",
        "}\n",
        "pub fn bad_local() -> &'static u8 {\n",
        "    let mut x = [0u8; 4];\n",
        "    let me = make_proj(&mut x);\n",
        "    &me.buf[0]\n",
        "}\n",
        "pub struct SynBuf { pub v: u8 }\n",
        "pub fn consume_buf(s: SynBuf) -> u8 { s.v }\n",
        "pub fn read_buf(s: &SynBuf) -> u8 { s.v }\n",
        "pub fn ok_braced(s: SynBuf) -> u8 {\n",
        "    let content;\n",
        "    content = s;\n",
        "    _ = content;\n",
        "    read_buf(&content)\n",
        "}\n",
        "pub fn bad_braced(s: SynBuf) -> u8 {\n",
        "    let content = s;\n",
        "    _ = consume_buf(content);\n",
        "    read_buf(&content)\n",
        "}\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let (
        proj_adt,
        make_empty,
        make_proj,
        ok_deref,
        bad_local,
        syn_adt,
        consume_buf,
        read_buf,
        ok_braced,
        bad_braced,
    ) = (
        find_item_in(&outline, "BufProj"),
        find_item_in(&outline, "make_empty"),
        find_item_in(&outline, "make_proj"),
        find_item_in(&outline, "ok_deref"),
        find_item_in(&outline, "bad_local"),
        find_item_in(&outline, "SynBuf"),
        find_item_in(&outline, "consume_buf"),
        find_item_in(&outline, "read_buf"),
        find_item_in(&outline, "ok_braced"),
        find_item_in(&outline, "bad_braced"),
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let usize_ty = cx.types.prim(Primitive::Usize);
    let ref_u8 = cx.types.intern(Kind::Ref(Mut::Not, u8_ty));
    let arr_ty = cx.types.array(u8_ty, 4);
    let mut_arr = cx.types.intern(Kind::Ref(Mut::Mut, arr_ty));
    let proj_ty = cx.types.adt(proj_adt, &[]);
    let syn_ty = cx.types.adt(syn_adt, &[]);
    let ref_syn = cx.types.intern(Kind::Ref(Mut::Not, syn_ty));

    check_case(&mut cx, ok_deref, |_, body, src| {
        let mut binds = Vec::new();
        for s in &body.stmts {
            if let Stmt::Let { pat, .. } = *s {
                binds.push((pat, 1));
            }
        }
        let mut exprs = vec![u8_ty; body.exprs.len()];
        let mut res = Vec::new();
        for (i, e) in body.exprs.iter().enumerate() {
            let id = ExprId(i as u32);
            match *e {
                Expr::Call(callee, ..) => {
                    exprs[i] = proj_ty;
                    res.push((id, Res::Fn(make_empty, Substs::EMPTY)));
                    res.push((callee, Res::Fn(make_empty, Substs::EMPTY)));
                }
                Expr::Ref { .. } => exprs[i] = ref_u8,
                Expr::Field(..) => exprs[i] = mut_arr,
                Expr::Lit(..) => exprs[i] = usize_ty,
                Expr::Path(pid) if path_last(body, src, pid) == Some("me") => {
                    exprs[i] = proj_ty;
                    res.push((id, Res::Local(1)));
                }
                _ => {}
            }
        }
        build_typed(
            exprs,
            res,
            binds,
            Vec::new(),
            vec![Local { ty: ref_u8 }, Local { ty: proj_ty }],
            vec![0],
            Some(ref_u8),
        )
    })
    .expect("deref through &mut slice field in projection struct");

    let bad_res = check_case(&mut cx, bad_local, |_, body, src| {
        let mut binds = Vec::new();
        let mut next = 0u32;
        for s in &body.stmts {
            if let Stmt::Let { pat, .. } = *s {
                binds.push((pat, next));
                next += 1;
            }
        }
        let mut exprs = vec![u8_ty; body.exprs.len()];
        let mut res = Vec::new();
        for (i, e) in body.exprs.iter().enumerate() {
            let id = ExprId(i as u32);
            match *e {
                Expr::Call(callee, ..) => {
                    exprs[i] = proj_ty;
                    res.push((id, Res::Fn(make_proj, Substs::EMPTY)));
                    res.push((callee, Res::Fn(make_proj, Substs::EMPTY)));
                }
                Expr::Array(..) => exprs[i] = arr_ty,
                Expr::Ref { mutable: true, .. } => exprs[i] = mut_arr,
                Expr::Ref { mutable: false, .. } => exprs[i] = ref_u8,
                Expr::Field(..) => exprs[i] = mut_arr,
                Expr::Path(pid) => match path_last(body, src, pid) {
                    Some("x") => {
                        exprs[i] = arr_ty;
                        res.push((id, Res::Local(0)));
                    }
                    Some("me") => {
                        exprs[i] = proj_ty;
                        res.push((id, Res::Local(1)));
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        build_typed(
            exprs,
            res,
            binds,
            Vec::new(),
            vec![Local { ty: arr_ty }, Local { ty: proj_ty }],
            Vec::new(),
            Some(ref_u8),
        )
    });
    assert!(bad_res.is_err());

    let run_braced = |cx: &mut Context<'_>, f| {
        check_case(cx, f, |_, body, src| {
            let mut binds = Vec::new();
            for s in &body.stmts {
                if let Stmt::Let { pat, .. } = *s {
                    binds.push((pat, 1));
                }
            }
            let mut exprs = vec![u8_ty; body.exprs.len()];
            let mut res = Vec::new();
            for (i, e) in body.exprs.iter().enumerate() {
                let id = ExprId(i as u32);
                match *e {
                    Expr::Ref { .. } => exprs[i] = ref_syn,
                    Expr::Call(callee, ..) => {
                        if let Expr::Path(pid) = *body.expr(callee) {
                            let target = if path_last(body, src, pid) == Some("read_buf") {
                                read_buf
                            } else {
                                consume_buf
                            };
                            res.push((id, Res::Fn(target, Substs::EMPTY)));
                            res.push((callee, Res::Fn(target, Substs::EMPTY)));
                        }
                    }
                    Expr::Path(pid) => match path_last(body, src, pid) {
                        Some("s") => {
                            exprs[i] = syn_ty;
                            res.push((id, Res::Local(0)));
                        }
                        Some("content") => {
                            exprs[i] = syn_ty;
                            res.push((id, Res::Local(1)));
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            build_typed(
                exprs,
                res,
                binds,
                Vec::new(),
                vec![Local { ty: syn_ty }, Local { ty: syn_ty }],
                vec![0],
                Some(u8_ty),
            )
        })
    };
    run_braced(&mut cx, ok_braced).expect("_ = place should not move content");
    assert!(run_braced(&mut cx, bad_braced).is_err());
}

#[test]
fn pin_poll_through_closure_and_typed_self_named_ref_cases() {
    let code = concat!(
        "pub struct PinWrap<P> { pub p: P }\n",
        "pub struct PollRes<T> { pub v: T }\n",
        "pub trait MiniFut { type Out; fn poll(self: PinWrap<&mut Self>) -> PollRes<Self::Out>; }\n",
        "pub fn pin_new<T>(v: &mut T) -> PinWrap<&mut T> { PinWrap { p: v } }\n",
        "pub fn pin_as_mut<T>(this: &mut PinWrap<&mut T>) -> PinWrap<&mut T> { loop {} }\n",
        "pub fn coop_budget<R>(f: impl FnOnce() -> R) -> R { loop {} }\n",
        "pub fn unwrap_ready<T>(p: PollRes<T>) -> T { loop {} }\n",
        "pub fn ok_block_on<F: MiniFut>(f: F) -> F::Out {\n",
        "    let mut f = f;\n",
        "    let mut f = pin_new(&mut f);\n",
        "    let v = unwrap_ready(coop_budget(|| pin_as_mut(&mut f).poll()));\n",
        "    return v;\n",
        "}\n",
        "pub struct BorrowFut<'a>(pub &'a u8);\n",
        "impl<'a> MiniFut for BorrowFut<'a> {\n",
        "    type Out = &'a u8;\n",
        "    fn poll(self: PinWrap<&mut Self>) -> PollRes<&'a u8> { loop {} }\n",
        "}\n",
        "pub fn bad_block_on() -> &'static u8 {\n",
        "    let x = 0u8;\n",
        "    let mut f = BorrowFut(&x);\n",
        "    let mut f = pin_new(&mut f);\n",
        "    let v = unwrap_ready(coop_budget(|| pin_as_mut(&mut f).poll()));\n",
        "    return v;\n",
        "}\n",
        "pub struct Boxed<T>(pub T);\n",
        "pub struct ProjPin<'a, T>(pub &'a mut T);\n",
        "pub fn make_proj_pin<'p, U>(u: U) -> ProjPin<'p, U> { loop {} }\n",
        "impl<T> Boxed<T> {\n",
        "    pub fn ok_project<'__pin>(self: PinWrap<&'__pin mut Self>) -> ProjPin<'__pin, T> {\n",
        "        make_proj_pin(self)\n",
        "    }\n",
        "    pub fn bad_project<'a, 'b>(self: PinWrap<&'b mut Self>) -> ProjPin<'a, T> {\n",
        "        make_proj_pin(self)\n",
        "    }\n",
        "}\n",
    );
    let (outline, names) = test_outline(code, "2021");
    let (pin_adt, poll_adt, borrow_adt, boxed_adt, proj_adt) = (
        find_item_in(&outline, "PinWrap"),
        find_item_in(&outline, "PollRes"),
        find_item_in(&outline, "BorrowFut"),
        find_item_in(&outline, "Boxed"),
        find_item_in(&outline, "ProjPin"),
    );
    let (poll_fn, pin_new, pin_as_mut, coop_budget, unwrap_ready, make_proj_pin) = (
        find_item_in(&outline, "poll"),
        find_item_in(&outline, "pin_new"),
        find_item_in(&outline, "pin_as_mut"),
        find_item_in(&outline, "coop_budget"),
        find_item_in(&outline, "unwrap_ready"),
        find_item_in(&outline, "make_proj_pin"),
    );
    let (ok_block_on, bad_block_on, ok_project, bad_project) = (
        find_item_in(&outline, "ok_block_on"),
        find_item_in(&outline, "bad_block_on"),
        find_item_in(&outline, "ok_project"),
        find_item_in(&outline, "bad_project"),
    );
    let mut cx = Context::new(&outline, &names, Cache::default());
    let u8_ty = cx.types.prim(Primitive::U8);
    let ref_u8 = cx.types.intern(Kind::Ref(Mut::Not, u8_ty));
    let param_f = cx.types.intern(Kind::Param(0));
    let param_t = cx.types.intern(Kind::Param(0));
    let borrow_ty = cx.types.adt(borrow_adt, &[]);

    let run_block_on = |cx: &mut Context<'_>, fn_item, fut_ty, out_ty, has_x: bool| {
        check_case(cx, fn_item, |cx, body, src| {
            let mut_fut = cx.types.intern(Kind::Ref(Mut::Mut, fut_ty));
            let pin_fut = cx.types.adt(pin_adt, &[mut_fut]);
            let mut_pin_fut = cx.types.intern(Kind::Ref(Mut::Mut, pin_fut));
            let poll_out = cx.types.adt(poll_adt, &[out_ty]);
            let subs_fut = cx.types.intern_substs(&[fut_ty]);
            let subs_out = cx.types.intern_substs(&[out_ty]);
            let mut binds = Vec::new();
            let mut next = u32::from(!has_x);
            let mut f_locals = Vec::new();
            let mut v_local = 0u32;
            for stmt in &body.stmts {
                if let Stmt::Let { pat, .. } = *stmt {
                    binds.push((pat, next));
                    if let crate::body::Pat::Bind { name, .. } = *body.pat(pat) {
                        match src.text(name) {
                            "f" => f_locals.push(next),
                            "v" => v_local = next,
                            _ => {}
                        }
                    }
                    next += 1;
                }
            }
            let (f1, f2) = (f_locals[0], f_locals[1]);
            let mut f_exprs = Vec::new();
            for (i, e) in body.exprs.iter().enumerate() {
                if let Expr::Path(pid) = *e
                    && path_last(body, src, pid) == Some("f")
                {
                    f_exprs.push(ExprId(i as u32));
                }
            }
            let mut f_iter = f_exprs.into_iter();
            let f_init_e = (!has_x).then(|| f_iter.next()).flatten();
            let (f_pin_e, f_cls_e) = (f_iter.next(), f_iter.next());

            let mut exprs = vec![out_ty; body.exprs.len()];
            let mut res = Vec::new();
            let mut closures = Vec::new();
            for (i, e) in body.exprs.iter().enumerate() {
                let id = ExprId(i as u32);
                match *e {
                    Expr::Method { .. } => {
                        exprs[i] = poll_out;
                        res.push((id, Res::Fn(poll_fn, subs_fut)));
                    }
                    Expr::Closure { body: c_body, .. } => {
                        let c_ty = cx.types.intern(Kind::Closure(fn_item, Substs::EMPTY, id.0));
                        exprs[i] = c_ty;
                        let subs_bud = cx.types.intern_substs(&[poll_out, c_ty]);
                        for (ci, ce) in body.exprs.iter().enumerate() {
                            if let Expr::Call(callee, ..) = *ce
                                && let Expr::Path(pid) = *body.expr(callee)
                                && path_last(body, src, pid) == Some("coop_budget")
                            {
                                res.push((ExprId(ci as u32), Res::Fn(coop_budget, subs_bud)));
                                res.push((callee, Res::Fn(coop_budget, subs_bud)));
                            }
                        }
                        closures.push((
                            id,
                            Closure {
                                params: Vec::new(),
                                ret: poll_out,
                                body: c_body,
                                captures: vec![Capture {
                                    local: f2,
                                    path: Vec::new(),
                                    by: By::Mut,
                                }],
                                consumes: false,
                                coroutine: false,
                            },
                        ));
                    }
                    Expr::Call(callee, ..) => {
                        if let Expr::Path(pid) = *body.expr(callee) {
                            match path_last(body, src, pid) {
                                Some("pin_new") => {
                                    exprs[i] = pin_fut;
                                    res.push((id, Res::Fn(pin_new, subs_fut)));
                                    res.push((callee, Res::Fn(pin_new, subs_fut)));
                                }
                                Some("pin_as_mut") => {
                                    exprs[i] = pin_fut;
                                    res.push((id, Res::Fn(pin_as_mut, subs_fut)));
                                    res.push((callee, Res::Fn(pin_as_mut, subs_fut)));
                                }
                                Some("coop_budget") => exprs[i] = poll_out,
                                Some("unwrap_ready") => {
                                    exprs[i] = out_ty;
                                    res.push((id, Res::Fn(unwrap_ready, subs_out)));
                                    res.push((callee, Res::Fn(unwrap_ready, subs_out)));
                                }
                                Some("BorrowFut") => {
                                    exprs[i] = borrow_ty;
                                    res.push((id, Res::Ctor(borrow_adt, 0, Substs::EMPTY)));
                                    res.push((callee, Res::Ctor(borrow_adt, 0, Substs::EMPTY)));
                                }
                                _ => {}
                            }
                        }
                    }
                    Expr::Ref {
                        expr: inner,
                        mutable,
                        ..
                    } => {
                        exprs[i] = if !mutable {
                            ref_u8
                        } else if Some(inner) == f_cls_e {
                            mut_pin_fut
                        } else {
                            mut_fut
                        };
                    }
                    Expr::Lit(..) => exprs[i] = u8_ty,
                    Expr::Path(pid) => match path_last(body, src, pid) {
                        Some("x") => {
                            exprs[i] = u8_ty;
                            res.push((id, Res::Local(0)));
                        }
                        Some("v") => {
                            exprs[i] = out_ty;
                            res.push((id, Res::Local(v_local)));
                        }
                        Some("f") => {
                            if Some(id) == f_init_e {
                                exprs[i] = fut_ty;
                                res.push((id, Res::Local(0)));
                            } else if Some(id) == f_pin_e {
                                exprs[i] = fut_ty;
                                res.push((id, Res::Local(f1)));
                            } else if Some(id) == f_cls_e {
                                exprs[i] = pin_fut;
                                res.push((id, Res::Local(f2)));
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            let mut locals = vec![Local { ty: fut_ty }; next as usize];
            if has_x {
                locals[0] = Local { ty: u8_ty };
            }
            locals[f1 as usize] = Local { ty: fut_ty };
            locals[f2 as usize] = Local { ty: pin_fut };
            locals[v_local as usize] = Local { ty: out_ty };
            build_typed(
                exprs,
                res,
                binds,
                closures,
                locals,
                if has_x { Vec::new() } else { vec![0] },
                Some(out_ty),
            )
        })
    };
    run_block_on(&mut cx, ok_block_on, param_f, param_f, false)
        .expect("pinned generic future polled in coop budget closure should allow return");
    assert!(run_block_on(&mut cx, bad_block_on, borrow_ty, ref_u8, true).is_err());

    let boxed_ty = cx.types.adt(boxed_adt, &[param_t]);
    let mut_boxed = cx.types.intern(Kind::Ref(Mut::Mut, boxed_ty));
    let pin_self = cx.types.adt(pin_adt, &[mut_boxed]);
    let proj_ret = cx.types.adt(proj_adt, &[param_t]);
    let subs_proj = cx.types.intern_substs(&[pin_self]);
    let build_proj = |body: &Body, src: &Source| {
        let mut exprs = vec![proj_ret; body.exprs.len()];
        let mut res = Vec::new();
        for (i, e) in body.exprs.iter().enumerate() {
            let id = ExprId(i as u32);
            match *e {
                Expr::Call(callee, ..) => {
                    res.push((id, Res::Fn(make_proj_pin, subs_proj)));
                    res.push((callee, Res::Fn(make_proj_pin, subs_proj)));
                }
                Expr::Path(pid) if path_last(body, src, pid) == Some("self") => {
                    exprs[i] = pin_self;
                    res.push((id, Res::Local(0)));
                }
                _ => {}
            }
        }
        build_typed(
            exprs,
            res,
            Vec::new(),
            Vec::new(),
            vec![Local { ty: pin_self }],
            vec![0],
            Some(proj_ret),
        )
    };
    let ok_sig = cx.signature(ok_project).expect("ok_project sig");
    let ok_body = cx.body(ok_project).expect("ok_project body");
    let ok_src = &outline.crates[0].files[0].source;
    let ok_typed = build_proj(&ok_body, ok_src);
    let ok_info = super::super::calls::compute_param_lifetimes(
        &mut cx,
        ok_project,
        &ok_sig,
        ok_src,
        &ok_typed,
        (1, &[]),
    );
    assert!(!ok_info.has_elided_lt[0]);
    assert_eq!(ok_info.named_lts[0], vec!["'__pin".to_owned()]);
    super::super::check(&mut cx, ok_project, (&ok_body, &ok_sig), ok_src, &ok_typed)
        .expect("Pin<&'__pin mut Self> should satisfy '__pin return");

    let bad_sig = cx.signature(bad_project).expect("bad_project sig");
    let bad_body = cx.body(bad_project).expect("bad_project body");
    let bad_src = &outline.crates[0].files[0].source;
    let bad_typed = build_proj(&bad_body, bad_src);
    assert!(
        super::super::check(
            &mut cx,
            bad_project,
            (&bad_body, &bad_sig),
            bad_src,
            &bad_typed
        )
        .is_err()
    );
}
