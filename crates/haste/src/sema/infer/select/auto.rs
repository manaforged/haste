use crate::names::ItemRef;

use super::super::super::impls::Head;
use super::super::super::substs::{ItemList, Substs};
use super::super::super::ty::{Kind, Mut, Ty};
use super::super::Infer;

impl<'a> Infer<'_, 'a> {
    fn dyn_holds_auto(&mut self, of: ItemRef, object: ItemRef, autos: ItemList) -> bool {
        object == of
            || self.cx.types.items(autos).contains(&of)
            || match self.implied_traits(vec![object]) {
                Ok(t) => t.contains(&of),
                Err(_) => true,
            }
    }

    fn unpin_holds(&mut self, of: ItemRef, ty: Ty, seen: &mut Vec<Ty>) -> bool {
        let ty = self.resolve(ty);
        if seen.contains(&ty) {
            return true;
        }
        match *self.cx.types.kind(ty) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => return true,
            Kind::Param(_) | Kind::Proj(_) => {
                return !self.root_is_user()
                    || (self.poly.is_some()
                        && self.poly_holds_in_env(of, ty, &[]).unwrap_or(false));
            }
            Kind::Slice(inner) | Kind::Pat(inner, _) => {
                seen.push(ty);
                return self.unpin_holds(of, inner, seen);
            }
            Kind::Dyn(object, _, _, autos) => return self.dyn_holds_auto(of, object, autos),
            Kind::Closure(parent, substs, at) => {
                return !self.ensure_coroutine(ty)
                    && self.closure_unpin_holds(of, ty, (parent, substs, at), seen);
            }
            Kind::Adt(item, args) => {
                if let Some(explicit) = self.adt_explicit_unpin(of, ty, item, args, seen) {
                    return explicit;
                }
            }
            Kind::Tuple(_) | Kind::Array(..) => {}
            _ => return true,
        }
        seen.push(ty);
        self.parts(ty)
            .into_iter()
            .all(|part| self.unpin_holds(of, part, seen))
    }

    fn adt_explicit_unpin(
        &mut self,
        of: ItemRef,
        ty: Ty,
        item: ItemRef,
        args: Substs,
        seen: &mut Vec<Ty>,
    ) -> Option<bool> {
        let holders = self.holders(of, ty, &[]);
        if self.cx.lang(item) == Some("unsafe_pinned")
            || (self.cx.outline.is_sysroot(item.krate) && self.cx.name(item) == "PhantomPinned")
            || self.cx.negative_impl(of, Head::Adt(item), &holders)
        {
            return Some(false);
        }
        if self.cx.boxed(ty).is_some() {
            return Some(true);
        }
        if self
            .cx
            .has_specific_trait_impl_for_head(of, Head::Adt(item), &holders)
        {
            return Some(self.select_auto_impl(of, ty).0);
        }
        if self.cx.lang(item) == Some("phantom_data") {
            seen.push(ty);
            let first = self.cx.types.substs(args).first().copied();
            return Some(first.is_none_or(|t| self.unpin_holds(of, t, seen)));
        }
        None
    }

    fn closure_unpin_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        (parent, substs, at): (ItemRef, Substs, u32),
        seen: &mut Vec<Ty>,
    ) -> bool {
        seen.push(ty);
        if self.instance == Some((parent, substs)) {
            if !self.captures_settled {
                return true;
            }
            let Some(captures) = self
                .out
                .closures
                .get(&crate::body::ExprId(at))
                .map(|c| c.captures.clone())
            else {
                return true;
            };
            return captures.into_iter().all(|cap| {
                cap.by != super::super::By::Value || {
                    let local_ty = self.resolve(self.out.locals[cap.local as usize].ty);
                    let cap_ty = cap.ty(self.cx, local_ty).unwrap_or(local_ty);
                    self.unpin_holds(of, cap_ty, seen)
                }
            });
        }
        self.cx
            .ensure_closure(ty)
            .is_none_or(|info| self.unpin_holds(of, info.env, seen))
    }

    pub(in super::super) fn thread_auto_holds(&mut self, of: ItemRef, ty: Ty) -> bool {
        if self.cx.lang(of) == Some("unpin") {
            return self.unpin_holds(of, ty, &mut Vec::new());
        }
        let (holds, dep) = self.thread_auto_holds_dep(of, ty);
        self.auto_dep = self.auto_dep.min(dep);
        if self.auto_stack.is_empty() {
            self.auto_provisional.clear();
            self.auto_dep = usize::MAX;
        }
        holds
    }

    fn finish_auto_node(
        &mut self,
        key: (ItemRef, Ty),
        cacheable: bool,
        my_depth: usize,
        holds: bool,
        dep: usize,
    ) -> (bool, usize) {
        if !holds {
            self.auto_provisional.clear();
            if cacheable {
                self.cx.set_auto_trait(key, false);
            }
            return (false, usize::MAX);
        }
        if dep >= my_depth {
            if cacheable {
                self.cx.set_auto_trait(key, true);
            }
            let promoted: Vec<(ItemRef, Ty)> = self
                .auto_provisional
                .iter()
                .filter_map(|(&k, &d)| (d >= my_depth).then_some(k))
                .collect();
            for k in promoted {
                self.auto_provisional.remove(&k);
                self.cx.set_auto_trait(k, true);
            }
            (true, usize::MAX)
        } else {
            if cacheable {
                self.auto_provisional.insert(key, dep);
            }
            (true, dep)
        }
    }

    fn thread_auto_holds_dep(&mut self, of: ItemRef, ty: Ty) -> (bool, usize) {
        let ty = self.resolve(ty);
        if let Some(pos) = self.auto_stack.iter().position(|&e| e == (of, ty)) {
            self.cycle_hits += 1;
            return (true, pos);
        }
        if self.auto_stack.len() >= crate::RECURSION_LIMIT {
            self.cycle_hits += 1;
            return (true, 0);
        }
        let cacheable = self.cx.types.is_ground(ty);
        if cacheable {
            if let Some(cached) = self.cx.auto_trait(&(of, ty)) {
                return (cached, usize::MAX);
            }
            if let Some(&dep) = self.auto_provisional.get(&(of, ty)) {
                return (true, dep);
            }
        }
        match *self.cx.types.kind(ty) {
            Kind::Prim(_)
            | Kind::Never
            | Kind::FnDef(..)
            | Kind::FnPtr(..)
            | Kind::UnsafeFnPtr(..)
            | Kind::Value(_)
            | Kind::Var(_)
            | Kind::IntVar(_)
            | Kind::FloatVar(_) => (true, usize::MAX),
            Kind::Param(_) | Kind::Proj(_) => {
                let holds = !self.root_is_user()
                    || (self.poly.is_some()
                        && self.poly_holds_in_env(of, ty, &[]).unwrap_or(false));
                (holds, usize::MAX)
            }
            Kind::Ptr(..) => (false, usize::MAX),
            Kind::Ref(Mut::Not, inner) => match self.cx.lang_item("sync") {
                Some(sync_of) => self.thread_auto_holds_dep(sync_of, inner),
                None => (true, usize::MAX),
            },
            Kind::Ref(Mut::Mut, inner)
            | Kind::Array(inner, _)
            | Kind::Slice(inner)
            | Kind::Pat(inner, _) => self.thread_auto_holds_dep(of, inner),
            Kind::Dyn(object, _, _, autos) => (self.dyn_holds_auto(of, object, autos), usize::MAX),
            Kind::Tuple(_) => {
                let my_depth = self.auto_stack.len();
                self.auto_stack.push((of, ty));
                let (holds, dep) = self.parts_auto_holds(of, ty);
                self.auto_stack.pop();
                self.finish_auto_node((of, ty), cacheable, my_depth, holds, dep)
            }
            Kind::Closure(parent, substs, at) => {
                self.closure_auto_holds(of, ty, (parent, substs, at))
            }
            Kind::Adt(item, args) => self.adt_auto_holds(of, ty, item, args, cacheable),
        }
    }

    fn closure_auto_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        (parent, substs, at): (ItemRef, Substs, u32),
    ) -> (bool, usize) {
        self.auto_stack.push((of, ty));
        let mut dep = usize::MAX;
        let (mut holds, extra_tys) = if self.instance == Some((parent, substs)) {
            let Some(c) = self.out.closures.get(&crate::body::ExprId(at)) else {
                self.auto_stack.pop();
                return (true, usize::MAX);
            };
            let (captures, is_cor, cor_body) = (c.captures.clone(), c.coroutine, c.body);
            let mut ok = true;
            for cap in captures {
                let local_ty = self.resolve(self.out.locals[cap.local as usize].ty);
                let cap_ty = cap.ty(self.cx, local_ty).unwrap_or(local_ty);
                let held = match cap.by {
                    super::super::By::Ref => self.cx.types.intern(Kind::Ref(Mut::Not, cap_ty)),
                    super::super::By::Mut => self.cx.types.intern(Kind::Ref(Mut::Mut, cap_ty)),
                    super::super::By::Value => cap_ty,
                };
                let (h, child_dep) = self.thread_auto_holds_dep(of, held);
                if !h {
                    ok = false;
                    break;
                }
                dep = dep.min(child_dep);
            }
            let extra = if ok && is_cor {
                super::super::drop::coroutine_body_tys(
                    self.body,
                    super::super::drop::CoroutineSource::Infer(&self.out),
                    cor_body,
                )
            } else {
                Vec::new()
            };
            (ok, extra)
        } else if let Some(info) = self.cx.ensure_closure(ty) {
            let (h, child_dep) = self.thread_auto_holds_dep(of, info.env);
            dep = child_dep;
            let extra = if h && info.closure.coroutine {
                if !matches!(info.typed.tables, super::super::LazyTables::CheckOnly(_))
                    && let Ok(body) = self.cx.body(info.parent)
                {
                    let scoped = info.typed.scoped(&self.cx.types);
                    super::super::drop::coroutine_body_tys(
                        &body,
                        super::super::drop::CoroutineSource::Typed(&**scoped, &scoped.locals),
                        info.closure.body,
                    )
                } else {
                    info.typed.locals.iter().map(|l| l.ty).collect()
                }
            } else {
                Vec::new()
            };
            (h, extra)
        } else {
            (true, Vec::new())
        };
        if holds {
            for raw in extra_tys {
                let lt = self.resolve(raw);
                if lt == Ty::NONE {
                    continue;
                }
                let (lh, ldep) = self.thread_auto_holds_dep(of, lt);
                if !lh {
                    holds = false;
                    break;
                }
                dep = dep.min(ldep);
            }
        }
        self.auto_stack.pop();
        (holds, if holds { dep } else { usize::MAX })
    }

    fn parts_auto_holds(&mut self, of: ItemRef, ty: Ty) -> (bool, usize) {
        let (mut dep, mut ok) = (usize::MAX, true);
        for part in self.parts(ty) {
            let (h, child_dep) = self.thread_auto_holds_dep(of, part);
            if !h {
                ok = false;
                break;
            }
            dep = dep.min(child_dep);
        }
        (ok, dep)
    }

    fn collect_open_vars(&self, ty: Ty, out: &mut Vec<Ty>) {
        if !self.cx.types.has_var_flag(ty) {
            return;
        }
        let ty = self.resolve(ty);
        match *self.cx.types.kind(ty) {
            Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_) => {
                if !out.contains(&ty) {
                    out.push(ty);
                }
            }
            kind => {
                for child in kind.children(&self.cx.types) {
                    self.collect_open_vars(child, out);
                }
            }
        }
    }

    fn select_auto_impl(&mut self, of: ItemRef, ty: Ty) -> (bool, usize) {
        let mut open_vars = Vec::new();
        if self.has_vars(ty) {
            self.collect_open_vars(ty, &mut open_vars);
        }
        let saved_dep = std::mem::replace(&mut self.auto_dep, usize::MAX);
        let asks_before = self.cx.asks.count();
        let (picked, cycled, bound_var) = self.probe(|infer| {
            let before = infer.cycle_hits;
            let picked = infer.select(of, ty, &[]);
            let bound_var = matches!(picked, Ok(Some(_)))
                && open_vars.iter().any(|&v| {
                    !matches!(
                        infer.cx.types.kind(infer.resolve(v)),
                        Kind::Var(_) | Kind::IntVar(_) | Kind::FloatVar(_)
                    )
                });
            (picked, infer.cycle_hits != before, bound_var)
        });
        let dep = std::mem::replace(&mut self.auto_dep, saved_dep);
        match picked {
            Ok(Some(_)) if !bound_var => (true, dep),
            Ok(Some(_)) => (false, usize::MAX),
            Ok(None) if cycled => {
                self.cycle_hits += 1;
                (true, 0)
            }
            Ok(None) if self.cx.asks.count() == asks_before => (false, usize::MAX),
            Ok(None) => (true, usize::MAX),
            Err(_) => {
                self.cycle_hits += 1;
                (true, 0)
            }
        }
    }

    fn adt_auto_holds(
        &mut self,
        of: ItemRef,
        ty: Ty,
        item: ItemRef,
        args: Substs,
        cacheable: bool,
    ) -> (bool, usize) {
        let asks_before = self.cx.asks.count();
        let holders = self.holders(of, ty, &[]);
        if self.cx.negative_impl(of, Head::Adt(item), &holders) {
            if cacheable {
                self.cx.set_auto_trait((of, ty), false);
            }
            return (false, usize::MAX);
        }
        let my_depth = self.auto_stack.len();
        self.auto_stack.push((of, ty));
        let specific = self
            .cx
            .has_specific_trait_impl_for_head(of, Head::Adt(item), &holders);
        let (holds, dep) = if specific {
            self.select_auto_impl(of, ty)
        } else if self.cx.lang(item) == Some("phantom_data") {
            self.cx
                .types
                .substs(args)
                .first()
                .copied()
                .map_or((true, usize::MAX), |t| self.thread_auto_holds_dep(of, t))
        } else {
            self.parts_auto_holds(of, ty)
        };
        self.auto_stack.pop();
        let cache = cacheable && self.cx.asks.count() == asks_before;
        self.finish_auto_node((of, ty), cache, my_depth, holds, dep)
    }
}
