use crate::body::{BlockKind, Expr, ExprId, List, Pat, PatId, Stmt, UnOp};
use crate::sema::infer::{By, Res};
use crate::sema::ty::{Kind, Mut};

use super::super::Checker;
use super::{FieldKey, Place, PlaceStep, push_step};

impl Checker<'_, '_> {
    pub(in super::super) fn closure_capture_info(
        &mut self,
        closure_expr: ExprId,
        by: By,
        target: u32,
    ) -> Vec<(Vec<PlaceStep>, bool, bool, bool)> {
        let root_expr = match *self.body.expr(closure_expr) {
            Expr::Closure { body, .. } => body,
            _ => closure_expr,
        };
        let mut walk = CapturePathWalk {
            checker: self,
            target,
            paths: Vec::new(),
        };
        walk.walk_expr(root_expr);
        if walk.paths.is_empty() {
            return vec![(Vec::new(), by == By::Mut, false, false)];
        }
        if by == By::Ref {
            for entry in &mut walk.paths {
                if !entry.3 {
                    entry.1 = false;
                }
            }
        }
        walk.paths
    }
}

fn capture_prefix(p: &Place) -> Vec<PlaceStep> {
    let limit = p.raw_ptr_prefix.unwrap_or(p.steps.len());
    p.steps[..limit]
        .iter()
        .take_while(|s| !matches!(s, PlaceStep::Opaque(_) | PlaceStep::Union))
        .cloned()
        .collect()
}

struct CapturePathWalk<'a, 'c, 'x> {
    checker: &'a mut Checker<'c, 'x>,
    target: u32,
    paths: Vec<(Vec<PlaceStep>, bool, bool, bool)>,
}

impl CapturePathWalk<'_, '_, '_> {
    fn record_path(
        &mut self,
        steps: Vec<PlaceStep>,
        mutated: bool,
        through_ref: bool,
        mut_ref_root: bool,
    ) {
        if let Some(existing) = self.paths.iter_mut().find(|(s, ..)| *s == steps) {
            existing.1 |= mutated;
            existing.2 |= through_ref;
            existing.3 |= mut_ref_root;
        } else {
            self.paths.push((steps, mutated, through_ref, mut_ref_root));
        }
    }

    fn record_place(&mut self, p: &Place, mutated: bool) {
        let (through_ref, mut_ref_root) = if p.raw_ptr_prefix.is_some() {
            (false, false)
        } else {
            (p.through_ref, p.mut_ref_root && !p.reborrows_shared_field)
        };
        self.record_path(
            capture_prefix(p),
            mutated && p.raw_ptr_prefix.is_none(),
            through_ref,
            mut_ref_root,
        );
    }

    fn walk_list(&mut self, list: List) {
        for &id in self.checker.body.list(list) {
            self.walk_expr(ExprId(id));
        }
    }

    fn walk_place_indices(&mut self, expr: ExprId) {
        match *self.checker.body.expr(expr) {
            Expr::Paren(inner)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Unary(_, inner) => {
                self.walk_place_indices(inner);
            }
            Expr::Index(base, idx) => {
                self.walk_place_indices(base);
                self.walk_expr(idx);
            }
            _ => {}
        }
    }

    fn check_nested_capture(&mut self, closure_expr: ExprId) -> bool {
        if let Some(c) = self.checker.typed.closures.get(&closure_expr) {
            let caps: Vec<_> = c
                .captures
                .iter()
                .filter(|cap| cap.local == self.target)
                .cloned()
                .collect();
            if !caps.is_empty() {
                let mut walk_body = false;
                for cap in caps {
                    let local_ty = self.checker.local_ty(self.target);
                    let cap_ty = cap.ty(self.checker.cx, local_ty).unwrap_or(local_ty);
                    let is_copy = self.checker.ty_is_copy(cap_ty);
                    let is_mut_ref = self.checker.is_mut_ref_ty(cap_ty);
                    let mut steps: Vec<PlaceStep> = cap
                        .path
                        .iter()
                        .map(|&(_, f)| PlaceStep::Field(FieldKey::Tuple(f)))
                        .collect();
                    if cap.by == By::Value && is_mut_ref {
                        push_step(&mut steps, PlaceStep::Deref);
                        self.record_path(steps, true, true, true);
                    } else if cap.by == By::Value && !is_copy {
                        self.record_path(steps, true, false, false);
                    } else if cap.by == By::Mut || (cap.by == By::Ref && cap.path.is_empty()) {
                        walk_body = true;
                    } else {
                        self.record_path(steps, false, false, false);
                    }
                }
                return walk_body;
            }
        }
        false
    }

    fn pat_res_inspects(&mut self, pat: PatId) -> bool {
        match self.checker.typed.pat_res.get(&pat).copied() {
            Some(Res::Ctor(item, _, _)) => self
                .checker
                .cx
                .adt_def(item)
                .is_ok_and(|def| def.variants.len() > 1),
            Some(Res::Fn(..)) => true,
            _ => false,
        }
    }

    fn walk_scrutinee_pat(&mut self, scrut: ExprId, pats: &[PatId]) {
        if let Some(p) = self.checker.root_place(scrut)
            && p.local == self.target
        {
            self.walk_place_indices(scrut);
            if p.raw_ptr_prefix.is_some() {
                self.record_place(&p, false);
                return;
            }
            let prefix = capture_prefix(&p);
            let mut_ref_root = p.mut_ref_root && !p.reborrows_shared_field;
            for &pat in pats {
                self.walk_pat(
                    pat,
                    &prefix,
                    (p.through_ref, mut_ref_root, p.reborrows_shared_field),
                );
            }
        } else {
            self.walk_expr(scrut);
        }
    }

    fn walk_tuple_pat(
        &mut self,
        pat: PatId,
        list: List,
        cur_steps: &[PlaceStep],
        (through_ref, mut_ref_root, shared): (bool, bool, bool),
    ) {
        if matches!(*self.checker.body.pat(pat), Pat::TupleStruct(..)) && self.pat_res_inspects(pat)
        {
            self.record_path(
                cur_steps.to_vec(),
                false,
                through_ref,
                mut_ref_root && !shared,
            );
        }
        let mut saw_rest = false;
        for (idx, &id) in self.checker.body.list(list).iter().enumerate() {
            let sub = PatId(id);
            if matches!(self.checker.body.pat(sub), Pat::Rest) {
                saw_rest = true;
                continue;
            }
            if saw_rest {
                self.record_path(
                    cur_steps.to_vec(),
                    false,
                    through_ref,
                    mut_ref_root && !shared,
                );
            } else {
                let mut next_steps = cur_steps.to_vec();
                push_step(
                    &mut next_steps,
                    PlaceStep::Field(FieldKey::Tuple(idx as u32)),
                );
                self.walk_pat(sub, &next_steps, (through_ref, mut_ref_root, shared));
            }
        }
    }

    fn walk_struct_pat(
        &mut self,
        pat: PatId,
        fields: List,
        cur_steps: &[PlaceStep],
        (through_ref, mut_ref_root, shared): (bool, bool, bool),
    ) {
        if self.pat_res_inspects(pat) {
            self.record_path(
                cur_steps.to_vec(),
                false,
                through_ref,
                mut_ref_root && !shared,
            );
        }
        for &fp_id in self.checker.body.list(fields) {
            let fp = self.checker.body.field_pats[fp_id as usize];
            let mut next_steps = cur_steps.to_vec();
            push_step(
                &mut next_steps,
                PlaceStep::Field(FieldKey::Named(self.checker.src.text(fp.name).to_owned())),
            );
            self.walk_pat(fp.pat, &next_steps, (through_ref, mut_ref_root, shared));
        }
    }

    fn walk_pat(
        &mut self,
        pat: PatId,
        steps: &[PlaceStep],
        (mut through_ref, mut mut_ref_root, mut shared): (bool, bool, bool),
    ) {
        let derefs = self.checker.typed.pat_derefs(pat);
        let mut cur_steps = steps.to_vec();
        if derefs > 0 {
            through_ref = true;
            if !shared && self.pat_has_mut_ref_bind(pat) {
                mut_ref_root = true;
            }
            for _ in 0..derefs {
                push_step(&mut cur_steps, PlaceStep::Deref);
            }
        }
        match *self.checker.body.pat(pat) {
            Pat::Wild | Pat::Rest => {}
            Pat::Paren(inner) | Pat::Box(inner) => {
                self.walk_pat(inner, &cur_steps, (through_ref, mut_ref_root, shared));
            }
            Pat::Ref {
                mutable,
                pat: inner,
            } => {
                through_ref = true;
                if !mutable {
                    shared = true;
                    mut_ref_root = false;
                } else if !shared {
                    mut_ref_root = true;
                }
                push_step(&mut cur_steps, PlaceStep::Deref);
                self.walk_pat(inner, &cur_steps, (through_ref, mut_ref_root, shared));
            }
            Pat::Bind { sub, .. } => {
                if self.checker.typed.pat_res.contains_key(&pat) {
                    if self.pat_res_inspects(pat) {
                        self.record_path(cur_steps, false, through_ref, mut_ref_root && !shared);
                    }
                    return;
                }
                let is_leaf =
                    sub.is_none_or(|s| matches!(*self.checker.body.pat(s), Pat::Rest | Pat::Wild));
                if let Some(&mutability) = self.checker.typed.ref_binds.get(&pat) {
                    self.record_path(
                        cur_steps.clone(),
                        mutability == Mut::Mut,
                        through_ref,
                        mut_ref_root && !shared,
                    );
                } else if is_leaf {
                    self.record_path(
                        cur_steps.clone(),
                        false,
                        through_ref,
                        mut_ref_root && !shared,
                    );
                }
                if let Some(sub) = sub {
                    self.walk_pat(sub, &cur_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Lit(_) | Pat::Range { .. } | Pat::Path(_) | Pat::Never | Pat::Macro { .. } => {
                self.record_path(cur_steps, false, through_ref, mut_ref_root && !shared);
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                self.walk_tuple_pat(pat, list, &cur_steps, (through_ref, mut_ref_root, shared));
            }
            Pat::Struct { fields, .. } => {
                self.walk_struct_pat(pat, fields, &cur_steps, (through_ref, mut_ref_root, shared));
            }
            Pat::Slice(list) => {
                if self
                    .checker
                    .typed
                    .pats
                    .get(&pat)
                    .is_some_and(|&ty| matches!(self.checker.cx.types.kind(ty), Kind::Slice(_)))
                {
                    self.record_path(
                        cur_steps.clone(),
                        false,
                        through_ref,
                        mut_ref_root && !shared,
                    );
                }
                for (sub, step) in self.checker.slice_pat_steps(pat, list) {
                    let mut next_steps = cur_steps.clone();
                    push_step(&mut next_steps, step);
                    self.walk_pat(sub, &next_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Or(list) => {
                for &alt in self.checker.body.list(list) {
                    self.walk_pat(PatId(alt), &cur_steps, (through_ref, mut_ref_root, shared));
                }
            }
            Pat::Deref(inner) => {
                self.record_path(
                    cur_steps.clone(),
                    false,
                    through_ref,
                    mut_ref_root && !shared,
                );
                self.walk_pat(inner, &cur_steps, (through_ref, mut_ref_root, shared));
            }
        }
    }

    fn pat_has_mut_ref_bind(&self, pat: PatId) -> bool {
        match *self.checker.body.pat(pat) {
            Pat::Bind { sub, .. } => {
                self.checker.typed.ref_binds.get(&pat) == Some(&Mut::Mut)
                    || sub.is_some_and(|s| self.pat_has_mut_ref_bind(s))
            }
            Pat::Paren(inner) | Pat::Box(inner) | Pat::Deref(inner) => {
                self.pat_has_mut_ref_bind(inner)
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) | Pat::Slice(list) | Pat::Or(list) => self
                .checker
                .body
                .list(list)
                .iter()
                .any(|&id| self.pat_has_mut_ref_bind(PatId(id))),
            Pat::Struct { fields, .. } => self.checker.body.list(fields).iter().any(|&id| {
                self.pat_has_mut_ref_bind(self.checker.body.field_pats[id as usize].pat)
            }),
            _ => false,
        }
    }

    fn walk_place_expr(&mut self, expr: ExprId) {
        if let Some(p) = self.checker.root_place(expr)
            && p.local == self.target
        {
            self.record_place(&p, false);
            self.walk_place_indices(expr);
            return;
        }
        match *self.checker.body.expr(expr) {
            Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Unary(_, base) => {
                self.walk_expr(base);
            }
            Expr::Index(base, idx) => {
                self.walk_expr(base);
                self.walk_expr(idx);
            }
            _ => {}
        }
    }

    fn walk_method(&mut self, expr: ExprId, receiver: ExprId, params: List) {
        if let Some(mut p) = self.checker.root_place(receiver)
            && p.local == self.target
        {
            let adjust = self.checker.typed.adjusts.get(&expr).copied();
            let autoref = adjust.and_then(|a| a.autoref);
            let derefs = adjust.map_or(0, |a| a.derefs);
            let recv_ty = self.checker.expr_ty(receiver);
            if derefs > 0 {
                self.checker
                    .apply_deref_steps(expr, recv_ty, derefs, &mut p);
            } else if autoref.is_none() && self.checker.is_mut_ref_ty(recv_ty) {
                self.checker.deref_ref_step(recv_ty, &mut p);
                push_step(&mut p.steps, PlaceStep::Deref);
            }
            let is_mut = autoref == Some(Mut::Mut)
                || (autoref.is_none() && self.checker.is_mut_ref_ty(recv_ty));
            self.record_place(&p, is_mut);
            self.walk_place_indices(receiver);
        } else {
            self.walk_expr(receiver);
        }
        self.walk_list(params);
    }

    fn walk_block(&mut self, expr: ExprId, kind: BlockKind, stmts: List, tail: Option<ExprId>) {
        if matches!(kind, BlockKind::Async | BlockKind::AsyncMove)
            && !self.check_nested_capture(expr)
        {
            return;
        }
        for &id in self.checker.body.list(stmts) {
            match self.checker.body.stmts[id as usize] {
                Stmt::Let {
                    pat,
                    init,
                    otherwise,
                    ..
                } => {
                    if let Some(e) = init {
                        self.walk_scrutinee_pat(e, &[pat]);
                    }
                    if let Some(e) = otherwise {
                        self.walk_expr(e);
                    }
                }
                Stmt::Expr(e, _) => self.walk_expr(e),
                Stmt::Item(..) | Stmt::Macro { .. } | Stmt::Empty => {}
            }
        }
        if let Some(t) = tail {
            self.walk_expr(t);
        }
    }

    fn walk_match(&mut self, scrut: ExprId, arms: List) {
        let arm_ids = self.checker.body.list(arms);
        let pats: Vec<PatId> = arm_ids
            .iter()
            .map(|&id| self.checker.body.arms[id as usize].pat)
            .collect();
        self.walk_scrutinee_pat(scrut, &pats);
        for &id in arm_ids {
            let arm = self.checker.body.arms[id as usize];
            if let Some(g) = arm.guard {
                self.walk_expr(g);
            }
            self.walk_expr(arm.body);
        }
    }

    fn walk_expr(&mut self, expr: ExprId) {
        match *self.checker.body.expr(expr) {
            Expr::Ref {
                mutable,
                raw: _,
                expr: inner,
            } => {
                if let Some(p) = self.checker.root_place(inner)
                    && p.local == self.target
                {
                    self.record_place(&p, mutable);
                    self.walk_place_indices(inner);
                    return;
                }
                self.walk_expr(inner);
            }
            Expr::Paren(inner)
            | Expr::Cast(inner, _)
            | Expr::Try(inner)
            | Expr::Await(inner)
            | Expr::Become(inner) => self.walk_expr(inner),
            Expr::Field(..)
            | Expr::TupleField(..)
            | Expr::Index(..)
            | Expr::Unary(UnOp::Deref, _) => self.walk_place_expr(expr),
            Expr::Unary(_, inner) => self.walk_expr(inner),
            Expr::Path(_) => {
                if self.checker.typed.res.get(&expr) == Some(&Res::Local(self.target)) {
                    self.record_path(Vec::new(), false, false, false);
                }
            }
            Expr::Assign(lhs, rhs) | Expr::AssignOp(_, lhs, rhs) => {
                if let Some(p) = self.checker.root_place(lhs)
                    && p.local == self.target
                {
                    self.record_place(&p, true);
                    self.walk_place_indices(lhs);
                } else {
                    self.walk_expr(lhs);
                }
                self.walk_expr(rhs);
            }
            Expr::Binary(_, a, b) | Expr::Repeat(a, b) => {
                self.walk_expr(a);
                self.walk_expr(b);
            }
            Expr::Call(callee, args) => {
                self.walk_expr(callee);
                self.walk_list(args);
            }
            Expr::Method {
                receiver, params, ..
            } => self.walk_method(expr, receiver, params),
            Expr::Block {
                kind, stmts, tail, ..
            } => self.walk_block(expr, kind, stmts, tail),
            Expr::If(c, t, e) => {
                self.walk_expr(c);
                self.walk_expr(t);
                if let Some(alt) = e {
                    self.walk_expr(alt);
                }
            }
            Expr::Let(pat, init) => self.walk_scrutinee_pat(init, &[pat]),
            Expr::While { cond, body, .. } => {
                self.walk_expr(cond);
                self.walk_expr(body);
            }
            Expr::Loop { body, .. } => self.walk_expr(body),
            Expr::For { iter, body, .. } => {
                self.walk_expr(iter);
                self.walk_expr(body);
            }
            Expr::Match(scrut, arms) => self.walk_match(scrut, arms),
            Expr::Closure { body, .. } => {
                if self.check_nested_capture(expr) {
                    self.walk_expr(body);
                }
            }
            Expr::Tuple(list) | Expr::Array(list) => self.walk_list(list),
            Expr::Struct { fields, base, .. } => {
                for &id in self.checker.body.list(fields) {
                    self.walk_expr(self.checker.body.fields[id as usize].value);
                }
                if let Some(b) = base {
                    self.walk_expr(b);
                }
            }
            Expr::Range { from, to, .. } => {
                if let Some(f) = from {
                    self.walk_expr(f);
                }
                if let Some(t) = to {
                    self.walk_expr(t);
                }
            }
            Expr::Return(e) | Expr::Break(_, e) | Expr::Yield(e) => {
                if let Some(e) = e {
                    self.walk_expr(e);
                }
            }
            Expr::Lit(..)
            | Expr::Continue(_)
            | Expr::Macro { .. }
            | Expr::Underscore
            | Expr::FormatArgs(_)
            | Expr::OffsetOf { .. } => {}
        }
    }
}
