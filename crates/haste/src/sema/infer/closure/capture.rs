use anyhow::Result;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::body::{Expr, ExprId, List, Pat, PatId, Stmt, UnOp};
use crate::names::ItemRef;
use crate::outline::item::ItemKind;

use super::super::super::ty::{Kind, Mut, Ty};
use super::super::{Infer, Res};
use super::{By, Capture, Closure, step_field_ty};

#[derive(Clone, Default)]
pub(in super::super) struct Captures {
    pub(super) first: u32,
    pub(super) used: Vec<u32>,
    pub(super) mutated: Vec<(u32, ExprId)>,
    pub(super) reads: Vec<(ExprId, u32)>,
    pub(super) inner_closures: Vec<ExprId>,
}

struct PatCaptureOut<'a> {
    local: u32,
    known: &'a mut bool,
    out: &'a mut Vec<Capture>,
}

impl PatCaptureOut<'_> {
    fn push(&mut self, by: By, path: &[(u32, u32)]) {
        self.out.push(Capture {
            local: self.local,
            by,
            path: path.to_vec(),
        });
    }
}

struct PlaceUses {
    places: HashSet<ExprId>,
    parents: HashMap<ExprId, ExprId>,
    deref_parents: HashMap<ExprId, ExprId>,
    scrutinees: HashMap<ExprId, Vec<PatId>>,
}

impl<'a> Infer<'_, 'a> {
    fn max_by(a: By, b: By) -> By {
        match (a, b) {
            (By::Value, _) | (_, By::Value) => By::Value,
            (By::Mut, _) | (_, By::Mut) => By::Mut,
            _ => By::Ref,
        }
    }

    fn resolve_opt(&self, ty: Ty) -> Ty {
        if ty == Ty::NONE {
            Ty::NONE
        } else {
            self.resolve(ty)
        }
    }

    fn append_step(path: &[(u32, u32)], idx: u32) -> Vec<(u32, u32)> {
        let mut p = path.to_vec();
        p.push((0, idx));
        p
    }

    fn projectable_fields(&mut self, ty: Ty) -> Result<Option<usize>> {
        let ty = self.resolve(ty);
        Ok(match *self.cx.types.kind(ty) {
            Kind::Tuple(items) => Some(self.cx.types.substs(items).len()),
            Kind::Adt(item, _)
                if self.cx.boxed(ty).is_none()
                    && self.cx.lang(item) != Some("manually_drop")
                    && matches!(self.cx.item(item).1.kind, ItemKind::Struct | ItemKind::Enum) =>
            {
                let def = self.cx.adt_def(item)?;
                (def.variants.len() == 1).then(|| def.variants[0].fields.len())
            }
            _ => None,
        })
    }

    fn resolve_path(
        &mut self,
        mut ty: Ty,
        path: &[(u32, u32)],
        stop_drop: bool,
        known: &mut bool,
    ) -> Result<(Vec<(u32, u32)>, Ty)> {
        let mut out = Vec::new();
        let mut drop_at = None;
        for &step in path {
            ty = self.resolve_opt(ty);
            if ty == Ty::NONE {
                return Ok((out, Ty::NONE));
            }
            if self.has_vars(ty) {
                *known = false;
                return Ok((out, if stop_drop { ty } else { Ty::NONE }));
            }
            if stop_drop
                && drop_at.is_none()
                && matches!(self.cx.types.kind(ty), Kind::Adt(..))
                && self.drop_impl(ty)?.is_some()
            {
                drop_at = Some((out.len(), ty));
            }
            out.push(step);
            ty = step_field_ty(self.cx, ty, step)?;
        }
        let ty = self.resolve_opt(ty);
        if ty != Ty::NONE && self.has_vars(ty) {
            *known = false;
        }
        if let Some((len, drop_ty)) = drop_at
            && (ty == Ty::NONE || self.has_vars(ty) || !self.is_copy(ty)? || self.needs_drop(ty)?)
        {
            out.truncate(len);
            return Ok((out, drop_ty));
        }
        Ok((out, ty))
    }

    fn truncate_value_paths(&mut self, caps: &mut [Capture], known: &mut bool) -> Result<()> {
        for cap in caps {
            if cap.by == By::Value && !cap.path.is_empty() {
                (cap.path, _) = self.resolve_path(
                    self.out.locals[cap.local as usize].ty,
                    &cap.path,
                    true,
                    known,
                )?;
            }
        }
        Ok(())
    }

    fn expr_proj(&mut self, expr: ExprId) -> Result<(Vec<(u32, u32)>, bool)> {
        let (mut rev, mut cur, mut known) = (Vec::new(), expr, true);
        loop {
            match *self.body.expr(cur) {
                Expr::Paren(inner) => cur = inner,
                Expr::TupleField(base, idx) if !self.out.autoderefs.contains_key(&cur) => {
                    rev.push((base, (0, idx)));
                    cur = base;
                }
                Expr::Field(base, _) if !self.out.autoderefs.contains_key(&cur) => {
                    let Some(&idx) = self.out.fields.get(&cur) else {
                        rev.clear();
                        break;
                    };
                    rev.push((base, (0, idx)));
                    cur = base;
                }
                Expr::Index(base, _) | Expr::Unary(UnOp::Deref, base) => {
                    rev.clear();
                    cur = base;
                }
                _ => break,
            }
        }
        let mut out = Vec::with_capacity(rev.len());
        for (base, step) in rev.into_iter().rev() {
            let ty = self.resolve_opt(self.out.exprs[base.0 as usize]);
            if ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Var(_)) {
                known = false;
            }
            if ty == Ty::NONE || self.projectable_fields(ty)?.is_none() {
                break;
            }
            out.push(step);
        }
        Ok((out, known))
    }

    pub(in super::super) fn note_read(&mut self, expr: ExprId, local: u32) {
        self.uses
            .iter_mut()
            .filter(|uses| local < uses.first)
            .for_each(|uses| uses.reads.push((expr, local)));
    }

    pub(in super::super) fn settle_captures(&mut self) -> Result<()> {
        while !self.unsure.is_empty() {
            let (expr, uses, by_move, forced) = self.unsure.remove(0);
            let (captures, consumes, _) = self.decide_captures(&uses, (by_move, &forced))?;
            if let Some(closure) = self.out.closures.get_mut(&expr) {
                (closure.consumes, closure.captures) = (closure.coroutine || consumes, captures);
                self.var_epoch += 1;
                self.capture_epoch += 1;
            }
        }
        Ok(())
    }

    fn walk_read_proj(
        &mut self,
        read: ExprId,
        parents: &HashMap<ExprId, ExprId>,
        known: &mut bool,
    ) -> Result<(ExprId, Vec<(u32, u32)>)> {
        let (mut at, mut proj, mut ok) = (read, Vec::new(), true);
        while let Some(&parent) = parents.get(&at) {
            if ok {
                let cur_ty = self.resolve_opt(self.out.exprs[at.0 as usize]);
                if cur_ty != Ty::NONE && matches!(self.cx.types.kind(cur_ty), Kind::Var(_)) {
                    *known = false;
                }
                let step = match *self.body.expr(parent) {
                    Expr::Paren(_) => {
                        at = parent;
                        continue;
                    }
                    Expr::TupleField(_, idx) if !self.out.autoderefs.contains_key(&parent) => {
                        Some((0, idx))
                    }
                    Expr::Field(_, _) if !self.out.autoderefs.contains_key(&parent) => {
                        self.out.fields.get(&parent).map(|&idx| (0, idx))
                    }
                    _ => None,
                };
                match step {
                    Some(s) if cur_ty != Ty::NONE && self.projectable_fields(cur_ty)?.is_some() => {
                        proj.push(s)
                    }
                    _ => ok = false,
                }
            }
            at = parent;
        }
        Ok((at, proj))
    }

    pub(super) fn decide_captures(
        &mut self,
        uses: &Captures,
        (by_move, forced): (bool, &[u32]),
    ) -> Result<(Vec<Capture>, bool, bool)> {
        let mut known = true;
        let mut raw: Vec<Capture> = forced
            .iter()
            .map(|&local| Capture {
                local,
                by: By::Value,
                path: Vec::new(),
            })
            .collect();
        let mut read_locals = HashSet::default();
        if !uses.reads.is_empty() || !uses.inner_closures.is_empty() {
            let place_uses = self.place_uses();
            self.collect_read_captures(uses, &place_uses, &mut known, &mut read_locals, &mut raw)?;
            self.collect_child_captures(uses, &mut known, &mut raw)?;
        }
        for &(local, mut_expr) in &uses.mutated {
            let (mut_proj, mut_known) = self.expr_proj(mut_expr)?;
            known &= mut_known;
            raw.push(Capture {
                local,
                by: By::Mut,
                path: mut_proj,
            });
        }
        if uses.inner_closures.is_empty() {
            uses.used
                .iter()
                .filter(|l| !read_locals.contains(l))
                .for_each(|&local| {
                    raw.push(Capture {
                        local,
                        by: By::Ref,
                        path: Vec::new(),
                    })
                });
        }
        let consumes = raw.iter().any(|c| c.by == By::Value);
        if by_move {
            raw.iter_mut().for_each(|cap| cap.by = By::Value);
        }
        self.truncate_value_paths(&mut raw, &mut known)?;
        raw.sort_by(|a, b| (a.local, a.path.len(), &a.path).cmp(&(b.local, b.path.len(), &b.path)));
        let mut captures: Vec<Capture> = Vec::new();
        for cap in raw {
            if let Some(prev) = captures
                .iter_mut()
                .find(|p| p.local == cap.local && cap.path.starts_with(&p.path))
            {
                prev.by = Self::max_by(prev.by, cap.by);
            } else {
                captures.push(cap);
            }
        }
        self.truncate_value_paths(&mut captures, &mut known)?;
        Ok((captures, consumes, known))
    }

    fn collect_read_captures(
        &mut self,
        uses: &Captures,
        place_uses: &PlaceUses,
        known: &mut bool,
        read_locals: &mut HashSet<u32>,
        raw: &mut Vec<Capture>,
    ) -> Result<()> {
        for &(read, local) in &uses.reads {
            read_locals.insert(local);
            let (at, proj) = self.walk_read_proj(read, &place_uses.parents, known)?;
            if let Some(pats) = place_uses.scrutinees.get(&at) {
                let at_ty = self.resolve_opt(self.out.exprs[at.0 as usize]);
                *known &= at_ty == Ty::NONE || !matches!(self.cx.types.kind(at_ty), Kind::Var(_));
                let mut sink = PatCaptureOut {
                    local,
                    known: &mut *known,
                    out: &mut *raw,
                };
                for &pat in pats {
                    self.pat_captures(pat, &proj, (false, true), &mut sink)?;
                }
            } else if place_uses.places.contains(&at) {
                let mut cur = at;
                while let Some(&p) = place_uses
                    .parents
                    .get(&cur)
                    .or_else(|| place_uses.deref_parents.get(&cur))
                {
                    cur = p;
                }
                let mut sub = Vec::new();
                let mut sink = PatCaptureOut {
                    local,
                    known: &mut *known,
                    out: &mut sub,
                };
                for &pat in place_uses.scrutinees.get(&cur).into_iter().flatten() {
                    self.pat_captures(pat, &[], (true, false), &mut sink)?;
                }
                let by = if sub.iter().any(|c| c.by == By::Mut) {
                    By::Mut
                } else {
                    By::Ref
                };
                raw.push(Capture {
                    local,
                    by,
                    path: proj,
                });
            } else {
                let ty = self.resolve_opt(self.out.exprs[at.0 as usize]);
                if ty == Ty::NONE {
                    continue;
                }
                *known &= !matches!(self.cx.types.kind(ty), Kind::Var(_));
                let by = if self.moves(ty)? { By::Value } else { By::Ref };
                raw.push(Capture {
                    local,
                    by,
                    path: proj,
                });
            }
        }
        Ok(())
    }

    fn collect_child_captures(
        &mut self,
        uses: &Captures,
        known: &mut bool,
        raw: &mut Vec<Capture>,
    ) -> Result<()> {
        for &child in &uses.inner_closures {
            let Some(caps) = self.out.closures.get(&child).map(|c| c.captures.clone()) else {
                continue;
            };
            for cap in caps.into_iter().filter(|c| c.local < uses.first) {
                let (_, cap_ty) = self.resolve_path(
                    self.out.locals[cap.local as usize].ty,
                    &cap.path,
                    false,
                    known,
                )?;
                let by = if cap.by == By::Value
                    && cap_ty != Ty::NONE
                    && (!self.is_copy(cap_ty)? || self.needs_drop(cap_ty)?)
                {
                    By::Value
                } else if cap.by == By::Mut {
                    By::Mut
                } else {
                    By::Ref
                };
                raw.push(Capture {
                    local: cap.local,
                    by,
                    path: cap.path,
                });
            }
        }
        Ok(())
    }

    fn pat_res_inspects(&mut self, pat: PatId) -> bool {
        match self.out.pat_res.get(&pat).copied() {
            Some(Res::Ctor(item, _, _)) => self
                .cx
                .adt_def(item)
                .is_ok_and(|def| def.variants.len() > 1),
            Some(Res::Fn(..)) => true,
            _ => false,
        }
    }

    fn pat_captures(
        &mut self,
        pat: PatId,
        path: &[(u32, u32)],
        (through_ref, allow_proj): (bool, bool),
        sink: &mut PatCaptureOut<'_>,
    ) -> Result<()> {
        let ty = self.resolve_opt(self.out.pats.get(&pat).copied().unwrap_or(Ty::NONE));
        if ty != Ty::NONE && matches!(self.cx.types.kind(ty), Kind::Var(_)) {
            *sink.known = false;
        }
        let derefs = self.out.pat_derefs.get(&pat).copied().unwrap_or(0);
        let through_ref = through_ref || derefs > 0;
        let allow_proj = allow_proj && derefs == 0;
        match *self.body.pat(pat) {
            Pat::Wild | Pat::Rest => {}
            Pat::Paren(inner) => self.pat_captures(inner, path, (through_ref, allow_proj), sink)?,
            Pat::Ref { pat: inner, .. } => self.pat_captures(inner, path, (true, false), sink)?,
            Pat::Box(inner) | Pat::Deref(inner) => {
                sink.push(By::Ref, path);
                self.pat_captures(inner, path, (through_ref, false), sink)?;
            }
            Pat::Bind { sub, .. } => {
                self.pat_bind_capture(pat, sub, ty, path, (through_ref, allow_proj), sink)?
            }
            Pat::Lit(_) | Pat::Range { .. } | Pat::Path(_) | Pat::Never | Pat::Macro { .. } => {
                sink.push(By::Ref, path)
            }
            Pat::Tuple(list) | Pat::TupleStruct(_, list) => {
                self.pat_tuple_captures(pat, list, ty, path, (through_ref, allow_proj), sink)?
            }
            Pat::Struct { fields, .. } => {
                self.pat_struct_captures(pat, fields, ty, path, (through_ref, allow_proj), sink)?
            }
            Pat::Or(list) | Pat::Slice(list) => {
                if matches!(*self.body.pat(pat), Pat::Slice(_))
                    && ty != Ty::NONE
                    && matches!(self.cx.types.kind(ty), Kind::Slice(_))
                {
                    sink.push(By::Ref, path);
                }
                for &at in &self.body.list(list).to_vec() {
                    self.pat_captures(PatId(at), path, (through_ref, false), sink)?;
                }
            }
        }
        Ok(())
    }

    fn pat_bind_capture(
        &mut self,
        pat: PatId,
        sub: Option<PatId>,
        ty: Ty,
        path: &[(u32, u32)],
        (through_ref, allow_proj): (bool, bool),
        sink: &mut PatCaptureOut<'_>,
    ) -> Result<()> {
        if self.out.pat_res.contains_key(&pat) {
            if self.pat_res_inspects(pat) {
                sink.push(By::Ref, path);
            }
        } else if let Some(&m) = self.out.ref_binds.get(&pat) {
            sink.push(if m == Mut::Mut { By::Mut } else { By::Ref }, path);
        } else {
            let by = if !through_ref && ty != Ty::NONE && self.moves(ty)? {
                By::Value
            } else {
                By::Ref
            };
            sink.push(by, path);
        }
        if let Some(sub) = sub {
            self.pat_captures(sub, path, (through_ref, allow_proj), sink)?;
        }
        Ok(())
    }

    fn pat_tuple_captures(
        &mut self,
        pat: PatId,
        list: List,
        ty: Ty,
        path: &[(u32, u32)],
        (through_ref, allow_proj): (bool, bool),
        sink: &mut PatCaptureOut<'_>,
    ) -> Result<()> {
        if allow_proj
            && !self.pat_res_inspects(pat)
            && ty != Ty::NONE
            && let Some(count) = self.projectable_fields(ty)?
        {
            let (pats, rest) = super::super::rest_split(self.body, list)?;
            for (at, &sub) in pats.to_vec().iter().enumerate() {
                if let Some(idx) = super::super::position(at, pats.len(), rest, count) {
                    let next = Self::append_step(path, idx as u32);
                    self.pat_captures(sub, &next, (through_ref, true), sink)?;
                }
            }
            return Ok(());
        }
        if matches!(*self.body.pat(pat), Pat::TupleStruct(..)) && self.pat_res_inspects(pat) {
            sink.push(By::Ref, path);
        }
        for &at in &self.body.list(list).to_vec() {
            self.pat_captures(PatId(at), path, (through_ref, false), sink)?;
        }
        Ok(())
    }

    fn pat_struct_captures(
        &mut self,
        pat: PatId,
        fields: List,
        ty: Ty,
        path: &[(u32, u32)],
        (through_ref, allow_proj): (bool, bool),
        sink: &mut PatCaptureOut<'_>,
    ) -> Result<()> {
        let can_proj = allow_proj
            && !self.pat_res_inspects(pat)
            && ty != Ty::NONE
            && self.projectable_fields(ty)?.is_some();
        if !can_proj && self.pat_res_inspects(pat) {
            sink.push(By::Ref, path);
        }
        for &at in &self.body.list(fields).to_vec() {
            let sub = self.body.field_pats[at as usize].pat;
            if can_proj && let Some(&idx) = self.out.field_pats.get(&sub) {
                let next = Self::append_step(path, idx);
                self.pat_captures(sub, &next, (through_ref, true), sink)?;
            } else {
                if can_proj {
                    *sink.known = false;
                }
                self.pat_captures(sub, path, (through_ref, false), sink)?;
            }
        }
        Ok(())
    }

    fn moves(&mut self, ty: Ty) -> Result<bool> {
        if let Some(captures) = self.own_closure(ty).map(|c| c.captures.clone()) {
            for cap in captures {
                let mut known = true;
                let (_, cap_ty) = self.resolve_path(
                    self.out.locals[cap.local as usize].ty,
                    &cap.path,
                    false,
                    &mut known,
                )?;
                if cap.by == By::Mut
                    || (cap.by == By::Value && cap_ty != Ty::NONE && self.moves(cap_ty)?)
                {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        let reborrows = matches!(self.cx.types.kind(self.resolve(ty)), Kind::Ref(Mut::Mut, _));
        Ok(self.needs_drop(ty)? || (!reborrows && !self.is_copy(ty)?))
    }

    fn own_closure(&self, ty: Ty) -> Option<&Closure> {
        let ty = self.resolve(ty);
        let Kind::Closure(parent, substs, at) = self.cx.types.kind(ty) else {
            return None;
        };
        let here = self
            .instance
            .as_ref()
            .is_some_and(|(item, args)| item == parent && args == substs);
        here.then(|| self.out.closures.get(&ExprId(*at)))
            .flatten()
            .filter(|_| !self.cx.has_closure(ty))
    }

    fn place_uses(&mut self) -> PlaceUses {
        let (mut places, mut parents, mut deref_parents, mut scrutinees): (
            HashSet<ExprId>,
            HashMap<ExprId, ExprId>,
            HashMap<ExprId, ExprId>,
            HashMap<ExprId, Vec<PatId>>,
        ) = Default::default();
        let call_once = self.cx.call_once();
        for (at, expr) in self.body.exprs.iter().enumerate() {
            let at = ExprId(at as u32);
            match *expr {
                Expr::Field(base, _) | Expr::TupleField(base, _) | Expr::Paren(base) => {
                    parents.insert(base, at);
                }
                Expr::Unary(UnOp::Deref, base) => {
                    let base_ty = self.resolve_opt(self.out.exprs[base.0 as usize]);
                    if base_ty != Ty::NONE
                        && !self.out.derefs.contains_key(&(at, 0))
                        && self.cx.boxed(base_ty).is_some()
                    {
                        parents.insert(base, at);
                    } else {
                        places.insert(base);
                        if base_ty != Ty::NONE
                            && !matches!(self.cx.types.kind(base_ty), Kind::Ptr(..))
                        {
                            deref_parents.insert(base, at);
                        }
                    }
                }
                Expr::Call(callee, _) if self.consumes_callee(at, callee, call_once) => {}
                Expr::Match(scrut, arms) => {
                    places.insert(scrut);
                    scrutinees.entry(scrut).or_default().extend(
                        self.body
                            .list(arms)
                            .iter()
                            .map(|&id| self.body.arms[id as usize].pat),
                    );
                }
                Expr::Let(pat, scrut) => {
                    places.insert(scrut);
                    scrutinees.entry(scrut).or_default().push(pat);
                }
                Expr::Ref { expr, .. }
                | Expr::Index(expr, _)
                | Expr::Assign(expr, _)
                | Expr::AssignOp(_, expr, _)
                | Expr::Call(expr, _) => {
                    places.insert(expr);
                }
                Expr::Binary(op, lhs, rhs) if super::super::expr::is_compare(op) => {
                    places.insert(lhs);
                    places.insert(rhs);
                }
                Expr::Method { receiver, .. }
                    if self
                        .out
                        .adjusts
                        .get(&at)
                        .is_some_and(|a| a.autoref.is_some() || a.derefs > 0) =>
                {
                    places.insert(receiver);
                }
                _ => {}
            }
        }
        for stmt in &self.body.stmts {
            if let Stmt::Let {
                pat,
                init: Some(init),
                ..
            } = *stmt
            {
                scrutinees.entry(init).or_default().push(pat);
            }
        }
        PlaceUses {
            places,
            parents,
            deref_parents,
            scrutinees,
        }
    }

    fn consumes_callee(&self, call: ExprId, callee: ExprId, call_once: Option<ItemRef>) -> bool {
        let Some(Res::Fn(_method, substs)) = self
            .out
            .res
            .get(&call)
            .filter(|r| matches!(r, Res::Fn(m, _) if Some(*m) == call_once))
        else {
            return false;
        };
        let ty = self.resolve(
            self.cx
                .types
                .substs(*substs)
                .first()
                .copied()
                .unwrap_or(self.out.exprs[callee.0 as usize]),
        );
        match self.cx.types.kind(ty) {
            Kind::Closure(..) => self
                .own_closure(ty)
                .map_or_else(|| self.cx.closure_consumes(ty), |c| c.consumes),
            _ => true,
        }
    }

    pub(in super::super) fn note_use(&mut self, local: u32) {
        self.uses
            .iter_mut()
            .filter(|uses| local < uses.first)
            .for_each(|uses| uses.used.push(local));
    }

    pub(in super::super) fn note_mut(&mut self, expr: ExprId) -> Result<()> {
        self.deref_mutably(expr)?;
        let Some(local) = self.place_local(expr) else {
            return Ok(());
        };
        self.uses
            .iter_mut()
            .filter(|uses| local < uses.first)
            .for_each(|uses| uses.mutated.push((local, expr)));
        Ok(())
    }

    fn place_local(&self, expr: ExprId) -> Option<u32> {
        match *self.body.expr(expr) {
            Expr::Paren(inner)
            | Expr::Field(inner, _)
            | Expr::TupleField(inner, _)
            | Expr::Index(inner, _)
            | Expr::Unary(UnOp::Deref, inner) => {
                let inner_ty = self.resolve_opt(self.out.exprs[inner.0 as usize]);
                (inner_ty == Ty::NONE || !matches!(self.cx.types.kind(inner_ty), Kind::Ptr(..)))
                    .then(|| self.place_local(inner))
                    .flatten()
            }
            Expr::Path(_) => match self.out.res.get(&expr) {
                Some(Res::Local(local)) => Some(*local),
                _ => None,
            },
            _ => None,
        }
    }
}
