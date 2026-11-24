use std::hash::{Hash, Hasher};

use crate::body::ExprId;
use crate::names::ItemRef;

use super::super::super::impls::ImplRef;
use super::super::super::substs::Substs;
use super::super::super::ty::{Kind, Ty};
use super::super::Infer;
use super::Picked;

pub(super) type SelectionKey = (ItemRef, Ty, Substs);

impl<'a> Infer<'_, 'a> {
    pub(super) fn cached_selection(
        &self,
        key: Option<&SelectionKey>,
        local_key: Option<&SelectionKey>,
    ) -> Option<Option<(ImplRef, Substs)>> {
        key.and_then(|k| self.cx.selection(k))
            .or_else(|| local_key.and_then(|k| self.local_selections.get(k).copied()))
    }

    pub(super) fn store_selection(
        &mut self,
        key: Option<SelectionKey>,
        local_key: Option<SelectionKey>,
        picked: Option<&Picked>,
        state: (((usize, usize), usize, usize), u32),
    ) {
        if self.cache_state() != state {
            return;
        }
        let Some((k, local)) = key
            .map(|k| (k, false))
            .or_else(|| local_key.map(|k| (k, true)))
        else {
            return;
        };
        let Some(kept) = (match picked {
            None => Some(None),
            Some((b, a)) => (if local {
                self.locally_closed_substs(a)
            } else {
                self.closed_substs(a)
            })
            .map(|a| Some((*b, a))),
        }) else {
            return;
        };
        if local {
            self.local_selections.insert(k, kept);
        } else {
            self.cx.set_selection(k, kept);
        }
    }

    pub(in super::super) fn substs_if(
        &mut self,
        tys: &[Ty],
        ok: fn(&Self, Ty) -> bool,
    ) -> Option<Substs> {
        if tys.is_empty() {
            return Some(Substs::EMPTY);
        }
        if !tys.iter().all(|&ty| ok(self, ty)) {
            return None;
        }
        if tys.len() <= 8 {
            let mut buf = [Ty::NONE; 8];
            for (dst, &ty) in buf.iter_mut().zip(tys) {
                *dst = self.substitute(ty);
            }
            return Some(self.cx.types.intern_substs(&buf[..tys.len()]));
        }
        let sub: Vec<Ty> = tys.iter().map(|&ty| self.substitute(ty)).collect();
        Some(self.cx.types.intern_substs(&sub))
    }

    pub(in super::super) fn closed_substs(&mut self, tys: &[Ty]) -> Option<Substs> {
        self.substs_if(tys, Self::closed)
    }

    pub(in super::super) fn locally_closed_substs(&mut self, tys: &[Ty]) -> Option<Substs> {
        self.substs_if(tys, Self::locally_closed)
    }

    pub(super) fn selection_keys(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
    ) -> (Option<SelectionKey>, Option<SelectionKey>) {
        if args.is_empty() && self.cx.types.is_ground(self_ty) {
            return (Some((of, self_ty, Substs::EMPTY)), None);
        }
        let build = |this: &mut Self, ok: fn(&Self, Ty) -> bool| {
            ok(this, self_ty)
                .then(|| this.substs_if(args, ok))
                .flatten()
                .map(|a| (of, this.substitute(self_ty), a))
        };
        if let Some(k) = build(self, Self::closed) {
            (Some(k), None)
        } else {
            (None, build(self, Self::locally_closed))
        }
    }

    pub(in super::super) fn closed(&self, ty: Ty) -> bool {
        self.closed_at(ty, 0, false, false)
    }

    pub(in super::super) fn locally_closed(&self, ty: Ty) -> bool {
        self.closed_at(ty, 0, true, false)
    }

    fn closed_at(&self, mut ty: Ty, depth: u32, local: bool, check_trail: bool) -> bool {
        if depth > 24 || ty == Ty::NONE {
            return false;
        }
        if self.cx.types.is_ground(ty) {
            return true;
        }
        if !local && self.cx.types.has_closure(ty) {
            return false;
        }
        while let Some(index) = self.var_of(ty) {
            let Some(bound) = self.vars.get(index as usize).copied().flatten() else {
                break;
            };
            if check_trail
                && self.trail_depth > 0
                && self.var_trail.iter().any(|&(i, _)| i == index)
            {
                return false;
            }
            ty = bound;
        }
        if self.cx.types.is_ground(ty) {
            return true;
        }
        if self.has_params(ty) {
            return false;
        }
        let kind = self.cx.types.kind(ty);
        if matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_))
            || (!local && matches!(kind, Kind::Closure(..)))
        {
            return false;
        }
        if !kind
            .children(&self.cx.types)
            .all(|item| self.closed_at(item, depth + 1, local, check_trail))
        {
            return false;
        }
        let Kind::Closure(parent, args, at) = kind else {
            return true;
        };
        if self.instance == Some((*parent, *args)) {
            let check = check_trail || self.trail_depth > 0;
            return self
                .out
                .closures
                .get(&ExprId(*at))
                .is_some_and(|c| self.closure_closed(c, &self.out.locals, depth + 1, check));
        }
        self.cx.closure(ty).is_some_and(|info| {
            !info.typed.exprs.is_empty()
                && self.closure_closed(&info.closure, &info.typed.locals, depth + 1, check_trail)
        })
    }

    fn closure_closed(
        &self,
        c: &super::super::closure::Closure,
        locals: &[super::super::Local],
        depth: u32,
        check_trail: bool,
    ) -> bool {
        self.closed_at(c.ret, depth, true, check_trail)
            && c.params
                .iter()
                .all(|&p| self.closed_at(locals[p as usize].ty, depth, true, check_trail))
    }

    pub(in crate::sema) fn has_vars(&self, ty: Ty) -> bool {
        if !self.cx.types.has_var_flag(ty) {
            return false;
        }
        let mut seen = rustc_hash::FxHashSet::default();
        self.has_vars_seen(ty, &mut seen)
    }

    pub(in crate::sema) fn has_vars_seen(
        &self,
        ty: Ty,
        seen: &mut rustc_hash::FxHashSet<Ty>,
    ) -> bool {
        let ty = self.resolve(ty);
        if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
            return false;
        }
        let kind = self.cx.types.kind(ty);
        matches!(kind, Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_))
            || kind
                .children(&self.cx.types)
                .any(|item| self.has_vars_seen(item, seen))
    }

    pub(in crate::sema) fn has_params(&self, ty: Ty) -> bool {
        if !self.cx.types.has_var_flag(ty) {
            return self.cx.types.has_params(ty);
        }
        let mut seen = rustc_hash::FxHashSet::default();
        self.has_params_seen(ty, &mut seen)
    }

    fn has_params_seen(&self, ty: Ty, seen: &mut rustc_hash::FxHashSet<Ty>) -> bool {
        let ty = self.resolve(ty);
        if self.cx.types.has_params(ty) {
            return true;
        }
        if !self.cx.types.has_var_flag(ty) || !seen.insert(ty) {
            return false;
        }
        match self.cx.types.kind(ty) {
            Kind::Closure(_, args, _) => self
                .cx
                .types
                .substs(*args)
                .iter()
                .any(|&item| self.has_params_seen(item, seen)),
            kind => kind
                .children(&self.cx.types)
                .any(|item| self.has_params_seen(item, seen)),
        }
    }

    pub(in crate::sema) fn any_has_params(&self, tys: &[Ty]) -> bool {
        tys.iter().any(|&ty| self.has_params(ty))
    }

    pub(in crate::sema) fn substs_have_params(&self, substs: Substs) -> bool {
        self.any_has_params(self.cx.types.substs(substs))
    }

    pub(in super::super) fn obligation_open(&self, ty: Ty, args: &[Ty]) -> bool {
        self.has_vars(ty) || args.iter().any(|&arg| self.has_vars(arg))
    }

    pub(super) fn query(&self, of: ItemRef, ty: Ty, args: &[Ty]) -> u64 {
        let (mut hasher, mut seen) = (
            rustc_hash::FxHasher::default(),
            ([Ty::NONE; 16], 0usize, Vec::new()),
        );
        of.hash(&mut hasher);
        self.hash_canonical(ty, &mut hasher, &mut seen);
        self.hash_list(args, &mut hasher, &mut seen);
        hasher.finish()
    }

    fn hash_canonical(
        &self,
        ty: Ty,
        hasher: &mut rustc_hash::FxHasher,
        seen: &mut ([Ty; 16], usize, Vec<Ty>),
    ) {
        let ty = self.resolve(ty);
        let kind = self.cx.types.kind(ty);
        let compound = !matches!(
            kind,
            Kind::Var(_)
                | Kind::IntVar(_)
                | Kind::FloatVar(_)
                | Kind::Never
                | Kind::Prim(_)
                | Kind::Value(_)
                | Kind::Param(_)
        );
        if compound {
            let (buf, len, extra) = seen;
            if let Some(pos) = buf[..(*len).min(16)].iter().position(|&t| t == ty) {
                (0xFFu8, pos as u32).hash(hasher);
                return;
            }
            if *len > 16
                && let Some(pos) = extra.iter().position(|&t| t == ty)
            {
                (0xFFu8, (16 + pos) as u32).hash(hasher);
                return;
            }
            if *len < 16 {
                buf[*len] = ty;
            } else {
                extra.push(ty);
            }
            *len += 1;
        }
        std::mem::discriminant(kind).hash(hasher);
        match kind {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Never => {}
            Kind::Prim(prim) => prim.hash(hasher),
            Kind::Value(bits) => bits.hash(hasher),
            Kind::Param(index) => index.hash(hasher),
            Kind::Proj(id) => id.hash(hasher),
            Kind::Ref(m, inner) | Kind::Ptr(m, inner) => {
                m.hash(hasher);
                self.hash_canonical(*inner, hasher, seen);
            }
            Kind::Slice(inner) => self.hash_canonical(*inner, hasher, seen),
            Kind::Array(inner, len) => self.hash_list(&[*inner, *len], hasher, seen),
            Kind::Tuple(items) => self.hash_list(self.cx.types.substs(*items), hasher, seen),
            Kind::Adt(item, items) | Kind::FnDef(item, items) => {
                item.hash(hasher);
                self.hash_list(self.cx.types.substs(*items), hasher, seen);
            }
            Kind::FnPtr(params, ret) => {
                self.hash_list(self.cx.types.substs(*params), hasher, seen);
                self.hash_canonical(*ret, hasher, seen);
            }
            Kind::Closure(item, items, id) => {
                (item, id).hash(hasher);
                self.hash_list(self.cx.types.substs(*items), hasher, seen);
            }
            Kind::Dyn(of, items, members, autos) => {
                (
                    of,
                    self.cx.types.items(*members),
                    self.cx.types.items(*autos),
                )
                    .hash(hasher);
                self.hash_list(self.cx.types.substs(*items), hasher, seen);
            }
            Kind::Pat(base, range) => {
                self.cx.types.pat(*range).hash(hasher);
                self.hash_canonical(*base, hasher, seen);
            }
        }
    }

    fn hash_list(
        &self,
        items: &[Ty],
        hasher: &mut rustc_hash::FxHasher,
        seen: &mut ([Ty; 16], usize, Vec<Ty>),
    ) {
        items.len().hash(hasher);
        for &item in items {
            self.hash_canonical(item, hasher, seen);
        }
    }
}
