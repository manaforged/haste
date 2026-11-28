use std::hash::{Hash, Hasher};

use crate::body::ExprId;
use crate::names::ItemRef;

use super::super::super::impls::ImplRef;
use super::super::super::substs::Substs;
use super::super::super::ty::{Kind, Ty, Types};
use super::super::unify::{CacheState, LocalTyInfo, replay_obligations};
use super::super::{Infer, LocalObligations, LocalSelectionKey, LocalSelectionValue};
use super::Picked;

pub(super) type SelectionKey = (ItemRef, Ty, Substs);

fn lookup_selection(
    map: &rustc_hash::FxHashMap<LocalSelectionKey, LocalSelectionValue>,
    types: &Types,
    obligations: &mut Vec<(ItemRef, Vec<Ty>)>,
    k: LocalSelectionKey,
) -> Option<Option<(ImplRef, Substs)>> {
    if let Some((found, obs)) = map.get(&k) {
        replay_obligations(obligations, obs);
        return Some(*found);
    }
    let for_bound = k.4;
    let alt = (k.0, k.1, k.2, k.3, !for_bound);
    let (found, obs) = map.get(&alt)?;
    if for_bound {
        if found.is_none() {
            return None;
        }
    } else if found.is_some()
        && types
            .substs(k.2)
            .iter()
            .any(|&a| matches!(types.kind(a), Kind::Var(_)))
    {
        return None;
    }
    replay_obligations(obligations, obs);
    Some(*found)
}

impl<'a> Infer<'_, 'a> {
    pub(super) fn cached_selection(
        &mut self,
        key: Option<&SelectionKey>,
        local_key: Option<&(LocalSelectionKey, usize)>,
    ) -> Option<Option<(ImplRef, Substs)>> {
        if let Some(found) = key.and_then(|k| self.cx.selection(k)) {
            return Some(found);
        }
        let &(k, key_limit) = local_key?;
        if key_limit == 0
            && self.local_closed_cache_epoch == self.capture_epoch
            && let Some(hit) = lookup_selection(
                &self.local_closed_selections,
                &self.cx.types,
                &mut self.obligations,
                k,
            )
        {
            return Some(hit);
        }
        if self.local_cache_epoch != self.var_epoch {
            return None;
        }
        lookup_selection(
            &self.local_selections,
            &self.cx.types,
            &mut self.obligations,
            k,
        )
    }

    pub(in super::super) fn cache_check(&self, state: CacheState, min_var: usize) -> (bool, bool) {
        let base_ok = self.cx.quiet_mark() == state.quiet
            && self.projections.len() == state.projections
            && self.cycle_hits == state.cycle_hits
            && min_var >= state.vars_len
            && self.capture_epoch == state.capture_epoch;
        (
            base_ok && self.obligations.len() == state.obligations,
            base_ok && self.obligations.len() >= state.obligations,
        )
    }

    pub(in super::super) fn prepare_local_store(
        &mut self,
        key_limit: usize,
        val_tys: &[Ty],
        state: CacheState,
    ) -> Option<(LocalObligations, usize)> {
        if key_limit > state.vars_len {
            return None;
        }
        let mut seen = rustc_hash::FxHashSet::default();
        let mut info = LocalTyInfo {
            var_limit: key_limit,
            has_ty_var: false,
        };
        if !self.local_slice_info(val_tys, &mut seen, &mut info) || info.var_limit > state.vars_len
        {
            return None;
        }
        let obs = self.collect_local_obligations(state, &mut seen, &mut info)?;
        self.sync_local_cache();
        if info.var_limit > 0 {
            self.local_cache_var_limit = self.local_cache_var_limit.max(info.var_limit);
        }
        Some((obs, info.var_limit))
    }

    pub(super) fn store_selection(
        &mut self,
        key: Option<SelectionKey>,
        local_key: Option<(LocalSelectionKey, usize)>,
        picked: Option<&Picked>,
        state: CacheState,
        min_var: usize,
    ) {
        let (global_ok, local_ok) = self.cache_check(state, min_var);
        if global_ok && let Some(k) = key {
            let kept = match picked {
                None => Some(None),
                Some((b, a)) => self.closed_substs(a).map(|a| Some((*b, a))),
            };
            if let Some(kept) = kept {
                self.cx.set_selection(k, kept);
            }
            return;
        }
        if !local_ok {
            return;
        }
        let Some((k, key_limit)) = local_key else {
            return;
        };
        let val_tys = picked.map_or(&[][..], |(_, a)| a.as_slice());
        let Some((obs, var_limit)) = self.prepare_local_store(key_limit, val_tys, state) else {
            return;
        };
        let kept = picked.map(|(b, a)| (*b, self.substitute_substs(a)));
        if var_limit == 0 {
            self.local_closed_selections.insert(k, (kept, obs));
        } else {
            self.local_selections.insert(k, (kept, obs));
        }
    }

    pub(in super::super) fn substitute_substs(&mut self, tys: &[Ty]) -> Substs {
        if tys.is_empty() {
            return Substs::EMPTY;
        }
        if tys.len() <= 8 {
            let mut buf = [Ty::NONE; 8];
            for (dst, &ty) in buf.iter_mut().zip(tys) {
                *dst = self.substitute(ty);
            }
            return self.cx.types.intern_substs(&buf[..tys.len()]);
        }
        let sub: Vec<Ty> = tys.iter().map(|&ty| self.substitute(ty)).collect();
        self.cx.types.intern_substs(&sub)
    }

    pub(in super::super) fn substs_if(
        &mut self,
        tys: &[Ty],
        ok: fn(&Self, Ty) -> bool,
    ) -> Option<Substs> {
        tys.iter()
            .all(|&ty| ok(self, ty))
            .then(|| self.substitute_substs(tys))
    }

    pub(in super::super) fn closed_substs(&mut self, tys: &[Ty]) -> Option<Substs> {
        self.substs_if(tys, Self::closed)
    }

    pub(in super::super) fn local_key_parts(
        &mut self,
        ty: Ty,
        args: &[Ty],
        own: &[Ty],
    ) -> Option<(Ty, Substs, Substs, Option<u32>, usize)> {
        if matches!(self.cx.types.kind(self.resolve(ty)), Kind::Var(_)) {
            return None;
        }
        let mut seen = rustc_hash::FxHashSet::default();
        let mut info = LocalTyInfo::default();
        if !self.local_ty_info(ty, false, &mut seen, &mut info)
            || !self.local_slice_info(args, &mut seen, &mut info)
            || !self.local_slice_info(own, &mut seen, &mut info)
        {
            return None;
        }
        let sub_ty = self.substitute(ty);
        let sub_args = self.substitute_substs(args);
        let sub_own = self.substitute_substs(own);
        let krate = info.has_ty_var.then_some(self.home.krate);
        Some((sub_ty, sub_args, sub_own, krate, info.var_limit))
    }

    pub(super) fn selection_keys(
        &mut self,
        of: ItemRef,
        self_ty: Ty,
        args: &[Ty],
        for_bound: bool,
    ) -> (Option<SelectionKey>, Option<(LocalSelectionKey, usize)>) {
        self.sync_local_cache();
        if args.is_empty() && self.cx.types.is_ground(self_ty) {
            return (Some((of, self_ty, Substs::EMPTY)), None);
        }
        if self.closed(self_ty)
            && let Some(a) = self.closed_substs(args)
        {
            return (Some((of, self.substitute(self_ty), a)), None);
        }
        let Some((sub_self, sub_args, _, krate, var_limit)) =
            self.local_key_parts(self_ty, args, &[])
        else {
            return (None, None);
        };
        (
            None,
            Some(((of, sub_self, sub_args, krate, for_bound), var_limit)),
        )
    }

    pub(in super::super) fn closed(&self, ty: Ty) -> bool {
        self.closed_at(ty, 0)
    }

    fn closed_at(&self, ty: Ty, depth: u32) -> bool {
        if depth > 24 || ty == Ty::NONE {
            return false;
        }
        if self.cx.types.is_ground(ty) {
            return true;
        }
        let ty = self.resolve(ty);
        if self.cx.types.is_ground(ty) {
            return true;
        }
        if self.cx.types.has_params(ty) {
            return false;
        }
        let kind = self.cx.types.kind(ty);
        if matches!(
            kind,
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) | Kind::Param(_) | Kind::Proj(_)
        ) {
            return false;
        }
        if let Kind::Closure(parent, args, _) = *kind {
            return self.instance != Some((parent, args))
                && self
                    .cx
                    .closure(ty)
                    .is_some_and(|c| !c.typed.exprs.is_empty())
                && self
                    .cx
                    .types
                    .substs(args)
                    .iter()
                    .all(|&item| self.closed_at(item, depth + 1));
        }
        kind.children(&self.cx.types)
            .all(|item| self.closed_at(item, depth + 1))
    }

    pub(in super::super) fn local_slice_info(
        &self,
        tys: &[Ty],
        seen: &mut rustc_hash::FxHashSet<(Ty, bool)>,
        info: &mut LocalTyInfo,
    ) -> bool {
        tys.iter()
            .all(|&ty| self.local_ty_info(ty, false, seen, info))
    }

    pub(in super::super) fn local_ty_info(
        &self,
        root: Ty,
        in_closure: bool,
        seen: &mut rustc_hash::FxHashSet<(Ty, bool)>,
        info: &mut LocalTyInfo,
    ) -> bool {
        if root == Ty::NONE {
            return false;
        }
        if self.cx.types.is_ground(root) {
            return true;
        }
        let mut stack = vec![(root, in_closure)];
        while let Some((mut cur, closed_ctx)) = stack.pop() {
            if cur == Ty::NONE {
                return false;
            }
            if self.cx.types.is_ground(cur) {
                continue;
            }
            while let Some(index) = self.var_of(cur) {
                if closed_ctx {
                    info.var_limit = info.var_limit.max(index as usize + 1);
                }
                match self.vars.get(index as usize).copied().flatten() {
                    Some(bound) => cur = bound,
                    None => {
                        info.var_limit = info.var_limit.max(index as usize + 1);
                        if matches!(self.cx.types.kind(cur), Kind::Var(_)) {
                            info.has_ty_var = true;
                        }
                        break;
                    }
                }
            }
            if cur == Ty::NONE || self.cx.types.has_params(cur) {
                return false;
            }
            if self.cx.types.is_ground(cur) || self.var_of(cur).is_some() {
                continue;
            }
            if !seen.insert((cur, closed_ctx)) {
                continue;
            }
            let kind = self.cx.types.kind(cur);
            for child in kind.children(&self.cx.types) {
                stack.push((child, closed_ctx));
            }
            if let Kind::Closure(parent, args, at) = kind
                && !self.push_closure_tys(cur, (*parent, *args, *at), &mut stack)
            {
                return false;
            }
        }
        true
    }

    fn push_closure_tys(
        &self,
        ty: Ty,
        (parent, args, at): (ItemRef, Substs, u32),
        stack: &mut Vec<(Ty, bool)>,
    ) -> bool {
        if self.instance == Some((parent, args)) {
            if !self.captures_settled && self.unsure.iter().any(|(e, ..)| e.0 == at) {
                return false;
            }
            let Some(c) = self.out.closures.get(&ExprId(at)) else {
                return false;
            };
            stack.push((c.ret, true));
            for &p in &c.params {
                stack.push((self.out.locals[p as usize].ty, true));
            }
            for cap in &c.captures {
                stack.push((self.out.locals[cap.local as usize].ty, true));
            }
            return true;
        }
        self.cx
            .closure(ty)
            .is_some_and(|closure| !closure.typed.exprs.is_empty())
    }

    pub(in super::super) fn collect_local_obligations(
        &mut self,
        state: CacheState,
        seen: &mut rustc_hash::FxHashSet<(Ty, bool)>,
        info: &mut LocalTyInfo,
    ) -> Option<LocalObligations> {
        if self.obligations.len() == state.obligations {
            return Some(Box::default());
        }
        let mut out: Vec<(ItemRef, Vec<Ty>)> = Vec::new();
        for idx in state.obligations..self.obligations.len() {
            let of = self.obligations[idx].0;
            let len = self.obligations[idx].1.len();
            let mut sub = Vec::with_capacity(len);
            for j in 0..len {
                let ty = self.obligations[idx].1[j];
                if !self.local_ty_info(ty, false, seen, info) || info.var_limit > state.vars_len {
                    return None;
                }
                sub.push(self.substitute(ty));
            }
            let entry = (of, sub);
            if !out.contains(&entry) {
                out.push(entry);
            }
        }
        Some(out.into_boxed_slice())
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
        let ty = if self.cx.types.has_var_flag(ty) {
            self.resolve(ty)
        } else {
            ty
        };
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
            Kind::FnPtr(params, ret) | Kind::UnsafeFnPtr(params, ret) => {
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
