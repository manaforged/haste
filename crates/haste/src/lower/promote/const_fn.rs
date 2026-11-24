use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::body::{BlockKind, Expr, ExprId, List, Pat, PatId, Stmt, UnOp};
use crate::names::{ItemRef, Primitive};
use crate::sema::infer::{LazyTables, Res, Shape, Typed};
use crate::sema::substs::Substs;
use crate::sema::ty::{Ty, int_info, mask};

use super::super::Lower;
use super::super::stencil::RodataReloc;
use super::const_val::{ConstEnv, Flow, Frame, Val};
use super::literal::{MAX_CONST_DEPTH, Scope};

fn is_readable_const_body(
    cx: &crate::sema::Context<'_>,
    owner: ItemRef,
    source: ItemRef,
    typed: &Typed,
) -> bool {
    let env = &cx.outline.crates[source.krate as usize].env;
    matches!(typed.shape, Shape::Body)
        && !cx.foreign(source)
        && !matches!(typed.tables, LazyTables::CheckOnly(_))
        && (env.release || env.fixed || source.krate == owner.krate)
}

impl<'l, 'a, 'r> ConstEnv<'l, 'a, 'r> {
    /// Looks up a concrete, readable `Shape::Body` function or const item without side effects.
    pub(super) fn const_body_source(
        &mut self,
        item: ItemRef,
        substs: Substs,
    ) -> Option<(ItemRef, Arc<Typed>)> {
        if self.lower.cx.types.substs_have_params(substs) {
            return None;
        }
        let owner = self.lower.reached.list[self.lower.index].0.0;
        if let Some(&index) = self.lower.reached.index.get(&(item, substs)) {
            if self.lower.reached.is_untyped(index) {
                return None;
            }
            let ((source, _), Ok(typed)) = &self.lower.reached.list[index] else {
                return None;
            };
            return is_readable_const_body(self.lower.cx, owner, *source, typed)
                .then(|| (*source, typed.clone()));
        }
        let env = &self.lower.cx.outline.crates[item.krate as usize].env;
        if self.lower.cx.foreign(item) || !(env.release || env.fixed || item.krate == owner.krate) {
            return None;
        }
        let (before, mut unlaid) = (self.lower.cx.quiet_mark(), Vec::new());
        let typed =
            crate::sema::reach::typed_once(self.lower.cx, &(item, substs), &mut unlaid).ok()?;
        (unlaid.is_empty()
            && self.lower.cx.quiet_mark() == before
            && is_readable_const_body(self.lower.cx, owner, item, &typed))
        .then_some((item, typed))
    }

    /// Resolves and caches an immutable static item inside the interpreter arena.
    pub(super) fn static_obj(&mut self, item: ItemRef, substs: Substs, ty: Ty) -> Option<u32> {
        if let Some(&obj_id) = self.statics.get(&(item, substs)) {
            return Some(obj_id);
        }
        if self.depth == 0
            && let Some((bytes, relocs)) = self
                .lower
                .static_rodata_blob(item, substs, ty)
                .ok()
                .flatten()
            && relocs.is_empty()
        {
            let obj_id = self.alloc_val(Val {
                bytes,
                ptrs: Vec::new(),
            })?;
            self.statics.insert((item, substs), obj_id);
            return Some(obj_id);
        }
        self.eval_static_init(item, substs, ty)
    }

    fn eval_static_init(&mut self, item: ItemRef, substs: Substs, ty: Ty) -> Option<u32> {
        let disallowed = self.lower.cx.foreign(item)
            || self.lower.cx.thread_local(item)
            || self.lower.is_mutable_static(item);
        if disallowed
            || !super::super::is_freeze_ty(self.lower.cx, ty, 0)
            || self.depth >= MAX_CONST_DEPTH
        {
            return None;
        }
        let (source, typed) = self.const_body_source(item, substs)?;
        let typed = typed.scoped(&self.lower.cx.types);
        if typed.ret_ty() != ty {
            return None;
        }
        let body = self.lower.cx.body(source).ok()?;
        let root = body.root?;
        let src = self.lower.cx.body_src(source);
        self.depth += 1;
        let mut frame = Frame {
            locals: vec![None; typed.locals.len()],
        };
        let val = self.eval_expr_val(
            Scope {
                body: &body,
                src,
                typed: &typed,
                in_const: true,
            },
            root,
            &mut frame,
        );
        self.depth -= 1;
        let obj_id = self.alloc_val(val?)?;
        self.statics.insert((item, substs), obj_id);
        Some(obj_id)
    }

    /// Evaluates expression `expr` in `scope` and unwraps normal `Flow::Val`.
    pub(super) fn eval_expr_val(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        frame: &mut Frame,
    ) -> Option<Val> {
        match self.eval_expr_flow(scope, expr, frame)? {
            Flow::Val(val) => Some(val),
            _ => None,
        }
    }

    /// Evaluates expression `expr` in `scope`, applying any `Coercion` recorded by inference.
    pub(super) fn eval_expr_flow(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        frame: &mut Frame,
    ) -> Option<Flow> {
        self.tick()?;
        let flow = self.eval_uncoerced_flow(scope, expr, frame)?;
        match flow {
            Flow::Val(val) => self.apply_coercion(scope, expr, val).map(Flow::Val),
            other => Some(other),
        }
    }

    fn eval_uncoerced_flow(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        frame: &mut Frame,
    ) -> Option<Flow> {
        let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
        if ty == Ty::NONE {
            return None;
        }
        match *scope.body.expr(expr) {
            Expr::Paren(inner) => self.eval_uncoerced_flow(scope, inner, frame),
            Expr::Lit(lit, at) => self.eval_lit_val(scope, lit, at, ty).map(Flow::Val),
            Expr::Path(_) => self.eval_path_val(scope, expr, ty, frame).map(Flow::Val),
            Expr::Unary(UnOp::Deref, operand) => self
                .eval_deref_read(scope, expr, operand, frame)
                .map(Flow::Val),
            Expr::Unary(op, operand) => self
                .eval_unary_val(scope, (expr, op, operand), ty, frame)
                .map(Flow::Val),
            Expr::Binary(op, lhs, rhs) => self
                .eval_binary_val(scope, (expr, op, lhs, rhs), ty, frame)
                .map(Flow::Val),
            Expr::Assign(lhs, rhs) => self
                .eval_assign_val(scope, (lhs, rhs), frame)
                .map(Flow::Val),
            Expr::AssignOp(op, lhs, rhs) => self
                .eval_compound_val(scope, (expr, op, lhs, rhs), frame)
                .map(Flow::Val),
            Expr::Cast(inner, _) => self.eval_cast_val(scope, inner, ty, frame).map(Flow::Val),
            Expr::Call(callee, args) => self
                .eval_call(scope, expr, callee, args, frame)
                .map(Flow::Val),
            Expr::Method {
                receiver, params, ..
            } => self
                .eval_method(scope, expr, receiver, params, frame)
                .map(Flow::Val),
            Expr::OffsetOf { .. } => Some(Flow::Val(Val::scalar(
                8,
                u128::from(*scope.typed.offsets.get(&expr)?),
            ))),
            _ => self.eval_control_flow(scope, expr, ty, frame),
        }
    }

    fn eval_control_flow(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Flow> {
        match *scope.body.expr(expr) {
            Expr::Block {
                stmts,
                tail,
                label: None,
                kind: BlockKind::Plain | BlockKind::Expansion | BlockKind::Unsafe | BlockKind::Const,
            } => self.eval_block_flow(scope, stmts, tail, frame),
            Expr::Block {
                stmts,
                tail,
                label: Some(label),
                ..
            } => self.eval_labeled_block(scope, stmts, tail, label, frame),
            Expr::If(cond, then, otherwise) => {
                self.eval_if_flow(scope, cond, then, otherwise, frame)
            }
            Expr::While { label, cond, body } => {
                self.eval_while_flow(scope, label, cond, body, frame)
            }
            Expr::Loop { label, body } => self.eval_loop_flow(scope, label, body, frame),
            Expr::Match(scrut, arms) => self.eval_match_flow(scope, scrut, arms, frame),
            Expr::Return(..) | Expr::Break(..) | Expr::Continue(..) => {
                self.eval_jump_flow(scope, expr, frame)
            }
            _ => self.eval_data_val(scope, expr, ty, frame).map(Flow::Val),
        }
    }

    fn eval_jump_flow(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        frame: &mut Frame,
    ) -> Option<Flow> {
        match *scope.body.expr(expr) {
            Expr::Return(val) => self.eval_opt_flow(scope, val, frame).map(|r| match r {
                Flow::Val(v) => Flow::Return(v),
                other => other,
            }),
            Expr::Break(label, val) => self.eval_opt_flow(scope, val, frame).map(|r| match r {
                Flow::Val(v) => Flow::Break(label, v),
                other => other,
            }),
            Expr::Continue(label) => Some(Flow::Continue(label)),
            _ => None,
        }
    }

    fn eval_opt_flow(
        &mut self,
        scope: Scope<'_>,
        val: Option<ExprId>,
        frame: &mut Frame,
    ) -> Option<Flow> {
        match val {
            Some(e) => self.eval_expr_flow(scope, e, frame),
            None => Some(Flow::Val(Val::zst())),
        }
    }

    fn eval_data_val(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        ty: Ty,
        frame: &mut Frame,
    ) -> Option<Val> {
        match *scope.body.expr(expr) {
            Expr::Ref {
                expr: inner, raw, ..
            } => self.eval_ref_val(scope, inner, raw, ty, frame),
            Expr::Field(..) | Expr::TupleField(..) | Expr::Index(..) => {
                let place = self.eval_place(scope, expr, frame)?;
                self.read_place(&place)
            }
            Expr::Tuple(list) => self.eval_tuple_val(scope, list, ty, frame),
            Expr::Array(list) => self.eval_array_val(scope, list, ty, frame),
            Expr::Repeat(item, _) => self.eval_repeat_val(scope, item, ty, frame),
            Expr::Struct { fields, base, .. } => {
                self.eval_struct_val(scope, (expr, fields, base), ty, frame)
            }
            _ => None,
        }
    }

    /// Evaluates a block's statements and tail, propagating `Return`/`Break`/`Continue`.
    pub(super) fn eval_block_flow(
        &mut self,
        scope: Scope<'_>,
        stmts: List,
        tail: Option<ExprId>,
        frame: &mut Frame,
    ) -> Option<Flow> {
        for &raw in scope.body.list(stmts) {
            let flow = self.exec_stmt(scope, scope.body.stmts[raw as usize], frame)?;
            if !matches!(flow, Flow::Val(_)) {
                return Some(flow);
            }
        }
        self.eval_opt_flow(scope, tail, frame)
    }

    fn eval_labeled_block(
        &mut self,
        scope: Scope<'_>,
        stmts: List,
        tail: Option<ExprId>,
        label: u32,
        frame: &mut Frame,
    ) -> Option<Flow> {
        match self.eval_block_flow(scope, stmts, tail, frame)? {
            Flow::Break(target, val)
                if crate::body::jumps_to(scope.src, (Some(label), true), target) =>
            {
                Some(Flow::Val(val))
            }
            other => Some(other),
        }
    }

    fn exec_stmt(&mut self, scope: Scope<'_>, stmt: Stmt, frame: &mut Frame) -> Option<Flow> {
        self.tick()?;
        match stmt {
            Stmt::Empty | Stmt::Item(..) => Some(Flow::Val(Val::zst())),
            Stmt::Expr(expr, _) => match self.eval_expr_flow(scope, expr, frame)? {
                Flow::Val(_) => Some(Flow::Val(Val::zst())),
                other => Some(other),
            },
            Stmt::Let {
                pat,
                init: None,
                otherwise: None,
                ..
            } if matches!(*scope.body.pat(pat), Pat::Bind { .. }) => Some(Flow::Val(Val::zst())),
            Stmt::Let {
                pat,
                init: Some(init),
                otherwise,
                ..
            } => self.exec_let_stmt(scope, pat, init, otherwise, frame),
            _ => None,
        }
    }

    fn exec_let_stmt(
        &mut self,
        scope: Scope<'_>,
        pat: PatId,
        init: ExprId,
        otherwise: Option<ExprId>,
        frame: &mut Frame,
    ) -> Option<Flow> {
        let place = self.eval_place_or_temp(scope, init, frame)?;
        if self.match_and_bind(scope, pat, &place, frame, false)? {
            self.match_and_bind(scope, pat, &place, frame, true)?;
            return Some(Flow::Val(Val::zst()));
        }
        match self.eval_expr_flow(scope, otherwise?, frame)? {
            Flow::Val(_) => None,
            flow => Some(flow),
        }
    }

    fn eval_cond_flow(
        &mut self,
        scope: Scope<'_>,
        cond: ExprId,
        frame: &mut Frame,
    ) -> Option<Result<bool, Flow>> {
        if let Expr::Let(pat, init) = *scope.body.expr(cond) {
            let place = self.eval_place_or_temp(scope, init, frame)?;
            let ok = self.match_and_bind(scope, pat, &place, frame, false)?;
            if ok {
                self.match_and_bind(scope, pat, &place, frame, true)?;
            }
            return Some(Ok(ok));
        }
        match self.eval_expr_flow(scope, cond, frame)? {
            Flow::Val(v) => Some(Ok(v.as_u128()? != 0)),
            other => Some(Err(other)),
        }
    }

    fn eval_if_flow(
        &mut self,
        scope: Scope<'_>,
        cond: ExprId,
        then: ExprId,
        otherwise: Option<ExprId>,
        frame: &mut Frame,
    ) -> Option<Flow> {
        let taken = match self.eval_cond_flow(scope, cond, frame)? {
            Ok(taken) => taken,
            Err(flow) => return Some(flow),
        };
        if taken {
            self.eval_expr_flow(scope, then, frame)
        } else {
            self.eval_opt_flow(scope, otherwise, frame)
        }
    }

    fn eval_while_flow(
        &mut self,
        scope: Scope<'_>,
        label: Option<u32>,
        cond: ExprId,
        body: ExprId,
        frame: &mut Frame,
    ) -> Option<Flow> {
        loop {
            self.tick()?;
            match self.eval_cond_flow(scope, cond, frame)? {
                Ok(true) => {}
                Ok(false) => return Some(Flow::Val(Val::zst())),
                Err(flow) => return Some(flow),
            }
            match self.eval_expr_flow(scope, body, frame)? {
                Flow::Val(_) => {}
                Flow::Continue(target)
                    if crate::body::jumps_to(scope.src, (label, false), target) => {}
                Flow::Break(target, _)
                    if crate::body::jumps_to(scope.src, (label, false), target) =>
                {
                    return Some(Flow::Val(Val::zst()));
                }
                other => return Some(other),
            }
        }
    }

    fn eval_loop_flow(
        &mut self,
        scope: Scope<'_>,
        label: Option<u32>,
        body: ExprId,
        frame: &mut Frame,
    ) -> Option<Flow> {
        loop {
            self.tick()?;
            match self.eval_expr_flow(scope, body, frame)? {
                Flow::Val(_) => {}
                Flow::Continue(target)
                    if crate::body::jumps_to(scope.src, (label, false), target) => {}
                Flow::Break(target, val)
                    if crate::body::jumps_to(scope.src, (label, false), target) =>
                {
                    return Some(Flow::Val(val));
                }
                other => return Some(other),
            }
        }
    }

    fn eval_match_flow(
        &mut self,
        scope: Scope<'_>,
        scrut: ExprId,
        arms: List,
        frame: &mut Frame,
    ) -> Option<Flow> {
        let place = self.eval_place_or_temp(scope, scrut, frame)?;
        for &raw in scope.body.list(arms) {
            let arm = scope.body.arms[raw as usize];
            if !self.match_and_bind(scope, arm.pat, &place, frame, false)? {
                continue;
            }
            self.match_and_bind(scope, arm.pat, &place, frame, true)?;
            if let Some(guard) = arm.guard {
                let g = self.eval_expr_flow(scope, guard, frame)?;
                let Flow::Val(v) = g else { return Some(g) };
                if v.as_u128()? == 0 {
                    continue;
                }
            }
            return self.eval_expr_flow(scope, arm.body, frame);
        }
        None
    }

    /// Executes a concrete function `item` at `substs` with argument values `args`.
    pub(super) fn call_fn(&mut self, item: ItemRef, substs: Substs, args: Vec<Val>) -> Option<Val> {
        if self.depth >= MAX_CONST_DEPTH {
            return None;
        }
        if let Some(which) = self.lower.cx.intrinsic(item) {
            let slice = self.lower.cx.types.substs(substs).to_vec();
            return self.eval_intrinsic(which, &slice, &args);
        }
        let (source, typed) = self.const_body_source(item, substs)?;
        let typed = typed.scoped(&self.lower.cx.types);
        let body = self.lower.cx.body(source).ok()?;
        let root = body.root?;
        let src = self.lower.cx.body_src(source);
        self.depth += 1;
        let out = self.run_fn_body(
            Scope {
                body: &body,
                src,
                typed: &typed,
                in_const: true,
            },
            root,
            args,
        );
        self.depth -= 1;
        out
    }

    fn run_fn_body(&mut self, scope: Scope<'_>, root: ExprId, args: Vec<Val>) -> Option<Val> {
        let mut locals = vec![None; scope.typed.locals.len()];
        let mut arg_iter = args.into_iter();
        for &local in &scope.typed.params {
            let val = arg_iter.next()?;
            locals[local as usize] = Some(self.alloc_val(val)?);
        }
        let mut frame = Frame { locals };
        match self.eval_expr_flow(scope, root, &mut frame)? {
            Flow::Val(v) | Flow::Return(v) => Some(v),
            _ => None,
        }
    }

    fn eval_call(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        callee: ExprId,
        args: List,
        frame: &mut Frame,
    ) -> Option<Val> {
        if let Some(&Res::Fn(item, substs)) = scope.typed.res.get(&expr) {
            let vals = self.eval_args(scope, args, Vec::new(), frame)?;
            return self.call_fn(item, substs, vals);
        }
        let res = scope
            .body
            .names_callee(callee)
            .then(|| scope.typed.res.get(&callee))
            .flatten()
            .copied()?;
        match res {
            Res::Ctor(_, variant, _) => {
                let ty = scope.typed.expr_ty(&self.lower.cx.types, expr);
                let vals = self.eval_args(scope, args, Vec::new(), frame)?;
                self.construct_adt(
                    ty,
                    variant,
                    vals.into_iter()
                        .enumerate()
                        .map(|(i, v)| (i as u32, v))
                        .collect(),
                    None,
                )
            }
            Res::Fn(item, substs)
                if !matches!(
                    self.lower.cx.item(item).1.kind,
                    crate::outline::item::ItemKind::Const | crate::outline::item::ItemKind::Static
                ) =>
            {
                let vals = self.eval_args(scope, args, Vec::new(), frame)?;
                self.call_fn(item, substs, vals)
            }
            _ => None,
        }
    }

    fn eval_method(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        receiver: ExprId,
        params: List,
        frame: &mut Frame,
    ) -> Option<Val> {
        let &Res::Fn(item, substs) = scope.typed.res.get(&expr)? else {
            return None;
        };
        let &adjust = scope.typed.adjusts.get(&expr)?;
        let first = self.eval_receiver(scope, (expr, receiver), adjust, frame)?;
        let vals = self.eval_args(scope, params, vec![first], frame)?;
        self.call_fn(item, substs, vals)
    }

    fn eval_args(
        &mut self,
        scope: Scope<'_>,
        args: List,
        mut out: Vec<Val>,
        frame: &mut Frame,
    ) -> Option<Vec<Val>> {
        for &raw in scope.body.list(args) {
            let arg = ExprId(raw);
            if !scope.typed.const_args.contains(&arg) {
                out.push(self.eval_expr_val(scope, arg, frame)?);
            }
        }
        Some(out)
    }
}

impl Lower<'_, '_> {
    /// Evaluates `expr` into `out[at..]` and `relocs` using the bounded `const fn` interpreter.
    pub(super) fn eval_const_into(
        &mut self,
        scope: Scope<'_>,
        (expr, ty): (ExprId, Ty),
        at: usize,
        (out, relocs): (&mut [u8], &mut Vec<RodataReloc>),
        depth: u32,
    ) -> Option<()> {
        if !scope.in_const
            && !matches!(
                *scope.body.expr(expr),
                Expr::Block {
                    kind: BlockKind::Const,
                    ..
                }
            )
        {
            return None;
        }
        let mut env = ConstEnv::new(self, depth);
        let mut frame = Frame {
            locals: vec![None; scope.typed.locals.len()],
        };
        let val = env.eval_expr_val(scope, expr, &mut frame)?;
        let expected = env.ty_size(ty)?;
        let mut exported = FxHashMap::default();
        let (bytes, child_relocs) = env.export_val(val, &mut exported)?;
        if bytes.len() != expected {
            return None;
        }
        out.get_mut(at..at + bytes.len())?.copy_from_slice(&bytes);
        for reloc in child_relocs {
            relocs.push(RodataReloc {
                offset: reloc.offset.checked_add(u32::try_from(at).ok()?)?,
                target: reloc.target,
            });
        }
        Some(())
    }

    /// Evaluates scalar primitive `expr` using the bounded `const fn` interpreter.
    pub(super) fn eval_const_scalar_fn(
        &mut self,
        scope: Scope<'_>,
        expr: ExprId,
        prim: Primitive,
        depth: u32,
    ) -> Option<u128> {
        if !scope.in_const
            && !matches!(
                *scope.body.expr(expr),
                Expr::Block {
                    kind: BlockKind::Const,
                    ..
                }
            )
        {
            return None;
        }
        let mut env = ConstEnv::new(self, depth);
        let mut frame = Frame {
            locals: vec![None; scope.typed.locals.len()],
        };
        let raw = env.eval_expr_val(scope, expr, &mut frame)?.as_u128()?;
        let bits = match prim {
            Primitive::Bool => return Some(u128::from(raw != 0)),
            Primitive::Char => {
                return u32::try_from(raw)
                    .ok()
                    .and_then(char::from_u32)
                    .map(|_| raw);
            }
            _ => int_info(prim)?.0,
        };
        Some(raw & mask(bits))
    }

    /// Evaluates a `const { ... }` block expression `expr` of type `ty` at compile time.
    pub(super) fn eval_const_block(
        &mut self,
        expr: ExprId,
        ty: Ty,
    ) -> Option<(Vec<u8>, Vec<RodataReloc>)> {
        if ty == Ty::NONE || self.cx.types.has_params(ty) {
            return None;
        }
        let scope = Scope {
            body: self.body,
            src: self.src,
            typed: self.typed,
            in_const: true,
        };
        let mut env = ConstEnv::new(self, 0);
        let expected = env.ty_size(ty)?;
        let mut frame = Frame {
            locals: vec![None; scope.typed.locals.len()],
        };
        let val = env.eval_expr_val(scope, expr, &mut frame)?;
        if val.bytes.len() != expected {
            return None;
        }
        let mut exported = FxHashMap::default();
        env.export_val(val, &mut exported)
    }
}
